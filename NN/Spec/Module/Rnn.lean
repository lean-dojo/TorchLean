/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Gru
public import NN.Spec.Layers.Lstm
public import NN.Spec.Module.Core

/-!
# RNN/LSTM/GRU module wrappers

The layer specs (`NN/Spec/Layers/Rnn.lean`, `lstm.lean`, `gru.lean`) expose step-level and
sequence-level recurrence definitions.

This file wraps the "sequence forward" functions as `Spec.Module`s so recurrent blocks can be
composed with other modules in a `Spec.Module.Chain`.

Design choices:

- These wrappers are **stateless** modules: they pick a canonical initial hidden/state (all zeros).
  `Spec.Module` remains a pure `forward`; more stateful variants can be built at the layer-spec
  level if needed.
- The exported `forward` returns the *full output sequence*, including every intermediate state,
  matching common encoder usage.

If you think in PyTorch: these are the `nn.RNN`/`nn.LSTM`/`nn.GRU` "return the full output sequence"
wrappers, with the initial hidden/state fixed to zeros.

The GRU wrappers are the exception to an exact match. They wrap the reset-before `GRUSpec`, while
PyTorch's `nn.GRU` and `nn.GRUCell` use the reset-after candidate equation. Their `pythonExpr`
exports the PyTorch module with the same shapes, which is an architectural stand-in; it computes a
different function for the same weights.
-/

@[expose] public section


namespace Spec.Module
open TorchLean TorchLean.Tensor

variable {α : Type} [TorchLean.Storage α] [Context α]

-- RNN module specification wrapper
/-- RNN sequence wrapper with a zero initial hidden state. -/
def rnn {seqLen inputSize hiddenSize : Nat}
  (rnn : RNNSpec α inputSize hiddenSize) :
  Spec.Module α
    ([seqLen, inputSize])
    ([seqLen, hiddenSize]) :=
{
  forward := fun x =>
    let initialHidden := Tensor.full ([hiddenSize]) 0
    rnnSequenceSpec rnn x initialHidden,
  kind := "RNN",
  pythonExpr := s!"RNNOnlyOutput({inputSize}, {hiddenSize})"
}

-- LSTM module specification wrapper
/-- LSTM sequence wrapper with a zero initial state; returns the output sequence. -/
def lstm {seqLen inputSize hiddenSize : Nat}
  (lstm : LSTMSpec α inputSize hiddenSize) :
  Spec.Module α
    ([seqLen, inputSize])
    ([seqLen, hiddenSize]) :=
{
  forward := fun x =>
    let initialState : LSTMState α hiddenSize := {
      hidden := Tensor.full ([hiddenSize]) 0,
      cell := Tensor.full ([hiddenSize]) 0
    }
    (lstmSequenceSpec lstm x initialState).1,
  kind := "LSTM",
  pythonExpr := s!"LSTMOnlyOutput({inputSize}, {hiddenSize})"
}

-- GRU module specification wrapper
/-- Reset-before GRU sequence wrapper with a zero initial hidden state; returns the output sequence.

The exported `GRUOnlyOutput` wraps PyTorch's reset-after `nn.GRU`, so it matches shapes only. -/
def gru {seqLen inputSize hiddenSize : Nat}
  (gru : GRUSpec α inputSize hiddenSize) :
  Spec.Module α
    ([seqLen, inputSize])
    ([seqLen, hiddenSize]) :=
{
  forward := fun x =>
    let initialHidden := Tensor.full ([hiddenSize]) 0
    gruSequenceSpec gru x initialHidden,
  kind := "GRUResetBefore",
  pythonExpr := s!"GRUOnlyOutput({inputSize}, {hiddenSize})"
}

/-- Bidirectional LSTM wrapper (concatenates forward/backward features). -/
def bidirectionalLstm {seqLen inputSize hiddenSize : Nat}
  (forwardLstm : LSTMSpec α inputSize hiddenSize)
  (backwardLstm : LSTMSpec α inputSize hiddenSize) :
  Spec.Module α
    ([seqLen, inputSize])
    ([seqLen, (hiddenSize + hiddenSize)]) :=
{
  forward := fun x =>
    let initialState : LSTMState α hiddenSize := {
      hidden := Tensor.full ([hiddenSize]) 0,
      cell := Tensor.full ([hiddenSize]) 0
    }
    let (forwardOut, _) := lstmSequenceSpec forwardLstm x initialState
    let reversedInputs := Tensor.reverseAxis 0 x
    let (reversedBackwardOut, _) := lstmSequenceSpec backwardLstm reversedInputs initialState
    let backwardOut := Tensor.reverseAxis 0 reversedBackwardOut
    Tensor.zipEach ([seqLen]) ([(hiddenSize + hiddenSize)])
      (Tensor.concatAxisSpec .scalar) forwardOut backwardOut,
  kind := "BiLSTM",
  pythonExpr := s!"LSTMOnlyOutput({inputSize}, {hiddenSize}, bidirectional=True)"
}

-- RNN Cell module (for single timestep processing)
/-- Wrap `rnnCellSpec` as an `Spec.Module` for a single timestep.

Input convention: we take a single vector `[x; h]` (concatenated input and previous hidden state),
so the module is shape-safe and easy to compose.
-/
def rnnCell {inputSize hiddenSize : Nat}
  (rnn : RNNSpec α inputSize hiddenSize) :
  Spec.Module α
    ([(inputSize + hiddenSize)])
    ([hiddenSize]) :=
{
  forward := fun x =>
    let input := sliceRangeSpec x 0 inputSize (by
      simp)
    let hidden := sliceRangeSpec x inputSize hiddenSize (by simp)
    rnnCellSpec rnn input hidden,
  kind := "RNNCell",
  pythonExpr := s!"nn.RNNCell({inputSize}, {hiddenSize})"
}

-- LSTM Cell module (for single timestep processing)
/-- Wrap `lstmCellSpec` as an `Spec.Module` for a single timestep.

Input convention: a single concatenated vector `[x; h; c]` (input, previous hidden, previous cell).
Output convention: the concatenated new state `[h'; c']`.
-/
def lstmCell {inputSize hiddenSize : Nat}
  (lstm : LSTMSpec α inputSize hiddenSize) :
  Spec.Module α
    ([(inputSize + hiddenSize + hiddenSize)])
    ([(hiddenSize + hiddenSize)]) :=
{
  forward := fun x =>
    let input := sliceRangeSpec x 0 inputSize (by
      simp [Nat.add_assoc])
    let hidden := sliceRangeSpec x inputSize hiddenSize (by
      simp)
    let cell := sliceRangeSpec x (inputSize + hiddenSize) hiddenSize (by simp)
    let state : LSTMState α hiddenSize := ⟨hidden, cell⟩
    let nextState := lstmCellSpec lstm input state
    concatAxisSpec .scalar nextState.hidden nextState.cell,
  kind := "LSTMCell",
  pythonExpr := s!"nn.LSTMCell({inputSize}, {hiddenSize})"
}

-- GRU Cell module (for single timestep processing)
/-- Wrap the reset-before `gruCellSpec` as an `Spec.Module` for a single timestep, using input
`[x; h]`.

The exported `nn.GRUCell` is PyTorch's reset-after cell, so it matches shapes only. -/
def gruCell {inputSize hiddenSize : Nat}
  (gru : GRUSpec α inputSize hiddenSize) :
  Spec.Module α
    ([(inputSize + hiddenSize)])
    ([hiddenSize]) :=
{
  forward := fun x =>
    let input := sliceRangeSpec x 0 inputSize (by
      simp)
    let hidden := sliceRangeSpec x inputSize hiddenSize (by simp)
    gruCellSpec gru input hidden,
  kind := "GRUResetBeforeCell",
  pythonExpr := s!"nn.GRUCell({inputSize}, {hiddenSize})"
}

/-- Bidirectional RNN wrapper (concatenates forward/backward features).

We run the RNNSpec forward over `x`, run it again over the reversed sequence, then reverse outputs
back and concatenate along the feature axis.
-/
def bidirectionalRnn {seqLen inputSize hiddenSize : Nat}
  (forwardRnn : RNNSpec α inputSize hiddenSize)
  (backwardRnn : RNNSpec α inputSize hiddenSize) :
  Spec.Module α
    ([seqLen, inputSize])
    ([seqLen, (hiddenSize + hiddenSize)]) :=
{
  forward := fun x =>
    let initialHidden := Tensor.full ([hiddenSize]) 0
    let forwardOut := rnnSequenceSpec forwardRnn x initialHidden
    let reversedInputs := Tensor.reverseAxis 0 x
    let reversedBackwardOut := rnnSequenceSpec backwardRnn reversedInputs initialHidden
    let backwardOut := Tensor.reverseAxis 0 reversedBackwardOut
    Tensor.zipEach ([seqLen]) ([(hiddenSize + hiddenSize)])
      (Tensor.concatAxisSpec .scalar) forwardOut backwardOut,
  kind := "BiRNN",
  pythonExpr := s!"RNNOnlyOutput({inputSize}, {hiddenSize}, bidirectional=True)"
}

end Spec.Module
