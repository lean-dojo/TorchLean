/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Sequence
public import NN.Spec.Layers.Gru
public import NN.Spec.Layers.Lstm
public import NN.Spec.Module.SpecModule

/-!
# RNN/LSTM/GRU module wrappers

The layer specs (`NN/Spec/Layers/Rnn.lean`, `lstm.lean`, `gru.lean`) expose step-level and
sequence-level recurrence definitions.

This file wraps the "sequence forward" functions as `NNModuleSpec`s so recurrent blocks can be
composed with other modules in a `SpecChain`.

Design choices:

- These wrappers are **stateless** modules: they pick a canonical initial hidden/state (all zeros).
  This keeps `NNModuleSpec` simple (pure `forward`), and makes the semantics explicit. More
  stateful variants can always be built at the layer-spec level if needed.
- The exported `forward` returns the *full output sequence* (not just the final hidden state),
  matching common encoder usage.

If you think in PyTorch: these are the `nn.RNN`/`nn.LSTM`/`nn.GRU` "return the full output sequence"
wrappers, with the initial hidden/state fixed to zeros.
-/

@[expose] public section


namespace Spec
open Tensor
open ModSpec

variable {α : Type} [Context α]

-- RNN module specification wrapper
/-- RNN sequence wrapper with a zero initial hidden state. -/
def RNNModuleSpec {seqLen inputSize hiddenSize : Nat}
  (rnn : RNNSpec α inputSize hiddenSize) :
  NNModuleSpec α
    (.dim seqLen (.dim inputSize .scalar))
    (.dim seqLen (.dim hiddenSize .scalar)) :=
{
  forward := fun x =>
    let initial_hidden := fill 0 (.dim hiddenSize .scalar)
    rnnSequenceSpec rnn x initial_hidden,
  kind := "RNN",
  export_func := {
    toPyTorch := s!"RNNOnlyOutput({inputSize}, {hiddenSize})",
    dimensions := (inputSize, hiddenSize)
  }
}

-- LSTM module specification wrapper
/-- LSTM sequence wrapper with a zero initial state; returns the output sequence. -/
def LSTMModuleSpec {seqLen inputSize hiddenSize : Nat}
  (lstm : LSTMSpec α inputSize hiddenSize) :
  NNModuleSpec α
    (.dim seqLen (.dim inputSize .scalar))
    (.dim seqLen (.dim hiddenSize .scalar)) :=
{
  forward := fun x =>
    let initial_state : LSTMState α hiddenSize := {
      hidden := fill 0 (.dim hiddenSize .scalar),
      cell := fill 0 (.dim hiddenSize .scalar)
    }
    (lstmSequenceSpec lstm x initial_state).1,
  kind := "LSTM",
  export_func := {
    toPyTorch := s!"LSTMOnlyOutput({inputSize}, {hiddenSize})",
    dimensions := (inputSize, hiddenSize)
  }
}

-- GRU module specification wrapper
/-- GRU sequence wrapper with a zero initial hidden state; returns the output sequence. -/
def GRUModuleSpec {seqLen inputSize hiddenSize : Nat}
  (gru : GRUSpec α inputSize hiddenSize) :
  NNModuleSpec α
    (.dim seqLen (.dim inputSize .scalar))
    (.dim seqLen (.dim hiddenSize .scalar)) :=
{
  forward := fun x =>
    let initial_hidden := fill 0 (.dim hiddenSize .scalar)
    gruSequenceSpec gru x initial_hidden,
  kind := "GRU",
  export_func := {
    toPyTorch := s!"GRUOnlyOutput({inputSize}, {hiddenSize})",
    dimensions := (inputSize, hiddenSize)
  }
}

-- Bidirectional LSTM module
/-- Bidirectional LSTM wrapper (concatenates forward/backward features). -/
def BiLSTMModuleSpec {seqLen inputSize hiddenSize : Nat}
  (forward_lstm : LSTMSpec α inputSize hiddenSize)
  (backward_lstm : LSTMSpec α inputSize hiddenSize) :
  NNModuleSpec α
    (.dim seqLen (.dim inputSize .scalar))
    (.dim seqLen (.dim (hiddenSize + hiddenSize) .scalar)) :=
{
  forward := fun x =>
    let initial_state : LSTMState α hiddenSize := {
      hidden := fill 0 (.dim hiddenSize .scalar),
      cell := fill 0 (.dim hiddenSize .scalar)
    }
    -- Forward pass
    let (forward_out, _) := lstmSequenceSpec forward_lstm x initial_state
    -- Backward pass (reverse inputs, then reverse outputs back to original order).
    let reversed_inputs := reverseSequenceSpec x
    let (backward_out_rev, _) := lstmSequenceSpec backward_lstm reversed_inputs initial_state
    let backward_out := reverseSequenceSpec backward_out_rev
    concatSequenceSpec forward_out backward_out,
  kind := "BiLSTM",
  export_func := {
    toPyTorch := s!"LSTMOnlyOutput({inputSize}, {hiddenSize}, bidirectional=True)",
    dimensions := (inputSize, hiddenSize * 2)
  }
}

-- RNN Cell module (for single timestep processing)
/-- Wrap `rnn_cell_spec` as an `NNModuleSpec` for a single timestep.

Input convention: we take a single vector `[x; h]` (concatenated input and previous hidden state),
so the module is shape-safe and easy to compose.
-/
def RNNCellModuleSpec {inputSize hiddenSize : Nat}
  (rnn : RNNSpec α inputSize hiddenSize) :
  NNModuleSpec α
    (.dim (inputSize + hiddenSize) .scalar)
    (.dim hiddenSize .scalar) :=
{
  forward := fun x =>
    -- Split concatenated input back into input and hidden
    let input := sliceVectorSpec x 0 inputSize (by
      simp)
    let hidden := sliceVectorSpec x inputSize hiddenSize (by simp)
    rnnCellSpec rnn input hidden,
  kind := "RNNCell",
  export_func := {
    toPyTorch := s!"nn.RNNCell({inputSize}, {hiddenSize})",
    dimensions := (inputSize + hiddenSize, hiddenSize)
  }
}

-- LSTM Cell module (for single timestep processing)
/-- Wrap `lstm_cell_spec` as an `NNModuleSpec` for a single timestep.

Input convention: a single concatenated vector `[x; h; c]` (input, previous hidden, previous cell).
Output convention: the concatenated new state `[h'; c']`.
-/
def LSTMCellModuleSpec {inputSize hiddenSize : Nat}
  (lstm : LSTMSpec α inputSize hiddenSize) :
  NNModuleSpec α
    (.dim (inputSize + hiddenSize + hiddenSize) .scalar)
    (.dim (hiddenSize + hiddenSize) .scalar) :=
{
  forward := fun x =>
    -- Split concatenated input back into input, hidden, and cell
    let input := sliceVectorSpec x 0 inputSize (by
      simp [Nat.add_assoc])
    let hidden := sliceVectorSpec x inputSize hiddenSize (by
      simp)
    let cell := sliceVectorSpec x (inputSize + hiddenSize) hiddenSize (by simp)
    let state : LSTMState α hiddenSize := ⟨hidden, cell⟩
    let new_state := lstmCellSpec lstm input state
    concatVectorsSpec new_state.hidden new_state.cell,
  kind := "LSTMCell",
  export_func := {
    toPyTorch := s!"nn.LSTMCell({inputSize}, {hiddenSize})",
    dimensions := (inputSize + hiddenSize + hiddenSize, hiddenSize + hiddenSize)
  }
}

-- GRU Cell module (for single timestep processing)
/-- Wrap `gru_cell_spec` as an `NNModuleSpec` for a single timestep, using input `[x; h]`. -/
def GRUCellModuleSpec {inputSize hiddenSize : Nat}
  (gru : GRUSpec α inputSize hiddenSize) :
  NNModuleSpec α
    (.dim (inputSize + hiddenSize) .scalar)
    (.dim hiddenSize .scalar) :=
{
  forward := fun x =>
    -- Split concatenated input back into input and hidden
    let input := sliceVectorSpec x 0 inputSize (by
      simp)
    let hidden := sliceVectorSpec x inputSize hiddenSize (by simp)
    gruCellSpec gru input hidden,
  kind := "GRUCell",
  export_func := {
    toPyTorch := s!"nn.GRUCell({inputSize}, {hiddenSize})",
    dimensions := (inputSize + hiddenSize, hiddenSize)
  }
}

-- Bidirectional RNN module
/-- Bidirectional RNN wrapper (concatenates forward/backward features).

We run the RNNSpec forward over `x`, run it again over the reversed sequence, then reverse outputs
back and concatenate along the feature axis.
-/
def BiRNNModuleSpec {seqLen inputSize hiddenSize : Nat}
  (forward_rnn : RNNSpec α inputSize hiddenSize)
  (backward_rnn : RNNSpec α inputSize hiddenSize) :
  NNModuleSpec α
    (.dim seqLen (.dim inputSize .scalar))
    (.dim seqLen (.dim (hiddenSize + hiddenSize) .scalar)) :=
{
  forward := fun x =>
    let initial_hidden := fill 0 (.dim hiddenSize .scalar)
    -- Forward pass
    let forward_out := rnnSequenceSpec forward_rnn x initial_hidden
    -- Backward pass (reverse inputs, then reverse outputs back).
    let reversed_inputs := reverseSequenceSpec x
    let backward_out_rev := rnnSequenceSpec backward_rnn reversed_inputs initial_hidden
    let backward_out := reverseSequenceSpec backward_out_rev
    concatSequenceSpec forward_out backward_out,
  kind := "BiRNN",
  export_func := {
    toPyTorch := s!"RNNOnlyOutput({inputSize}, {hiddenSize}, bidirectional=True)",
    dimensions := (inputSize, hiddenSize * 2)
  }
}

end Spec
