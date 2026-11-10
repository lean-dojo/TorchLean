/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Sequence
public import TorchLean.Tensor.ConcatSlice
public import TorchLean.Tensor.Movement
public import TorchLean.Tensor.Layers.Activation
public import TorchLean.Tensor.Layers.Linear

/-!
# LSTM (spec layer)

A small LSTM specification, laid out the way PyTorch documents `nn.LSTMCell` and `nn.LSTM`. The
shapes are part of the types, so a dimension mismatch is a type error, and the gate equations are
written out for proofs to refer to.

The forget gate isn't in Hochreiter and Schmidhuber (1997). Gers, Schmidhuber and Cummins added it
in 2000, and `LSTM.cell` is the cell with that gate, typed in by hand from the equations of
`torch.nn.LSTMCell` and `torch.nn.LSTM` (PyTorch 2.8 docs). `nn.Layer.lstmCell_eq` and
`Recurrent.scan_lstmCell` (`TorchLean/NN/Layer/Recurrent/Correctness.lean`) prove that the module
layer's cell and its scan give `LSTM.cell` and `LSTM.sequence` under the pure interpreter. We didn't
find a test that runs `LSTM.cell` against torch.

## References (math + PyTorch behavior)

- Hochreiter and Schmidhuber, "Long Short-Term Memory", Neural Computation 9(8):1735-1780, 1997:
  https://doi.org/10.1162/neco.1997.9.8.1735
- Gers, Schmidhuber and Cummins, "Learning to Forget: Continual Prediction with LSTM", Neural
  Computation 12(10):2451-2471, 2000, the forget gate: https://doi.org/10.1162/089976600300015015
- PyTorch 2.8 `LSTMCell` equations:
  https://docs.pytorch.org/docs/2.8/generated/torch.nn.LSTMCell.html
- PyTorch 2.8 `LSTM` equations: https://docs.pytorch.org/docs/2.8/generated/torch.nn.LSTM.html

## Notes on parameterization

Many libraries expose two matrices per gate (`W_ih` and `W_hh`) and add their products. Here each
gate is one `Linear` applied to the concatenated vector `[x_t; h_{t-1}]`, which gives the same
equations over the reals. The products over the concatenated vector fix their own reduction order,
and that order can round differently from separate input and hidden products. The fields run
forget, input, candidate, output, and torch packs its rows input first, as `(W_ii|W_if|W_ig|W_io)`
in `weight_ih`. Each gate has one bias, which over the reals stands for the sum of torch's
`bias_ih` and `bias_hh` blocks.
-/

@[expose] public section


namespace TorchLean

open TorchLean.Tensor

variable {α : Type} [Storage α] [Context α]

/-- Parameters for an LSTM cell: one `Linear` per gate, whose weight
`(hiddenSize × (inputSize + hiddenSize))` acts on `[x_t; h_{t-1}]`.

Each weight is the usual pair `(W_ih, W_hh)` of libraries like PyTorch, placed side by side as one
matrix on the concatenated vector.
-/
structure LSTM (α : Type) [Storage α]
    (inputSize hiddenSize : Nat) where
  /-- Forget gate, `f_t = sigmoid(W_f [x_t; h_{t-1}] + b_f)`. -/
  forget : Linear α (inputSize + hiddenSize) hiddenSize
  /-- Input gate, `i_t = sigmoid(W_i [x_t; h_{t-1}] + b_i)`. -/
  input : Linear α (inputSize + hiddenSize) hiddenSize
  /-- Candidate cell values, `g_t = tanh(W_g [x_t; h_{t-1}] + b_g)`. -/
  candidate : Linear α (inputSize + hiddenSize) hiddenSize
  /-- Output gate, `o_t = sigmoid(W_o [x_t; h_{t-1}] + b_o)`. -/
  output : Linear α (inputSize + hiddenSize) hiddenSize

/-- LSTM recurrent state: hidden vector `h_t` and cell vector `c_t`. -/
structure LSTM.State (α : Type) [Storage α] (hiddenSize : Nat) where
  /-- Exposed hidden state `h_t`. -/
  hidden : Tensor α [hiddenSize]  -- h_t
  /-- Internal memory/cell state `c_t`. -/
  cell   : Tensor α [hiddenSize]  -- c_t

/-- One LSTM cell step: update `(h_{t-1}, c_{t-1})` given `x_t` and parameters. -/
def LSTM.cell {inputSize hiddenSize : Nat}
  (lstm : LSTM α inputSize hiddenSize)
  (input : Tensor α [inputSize])
  (prevState : State α hiddenSize) :
  State α hiddenSize :=
  -- The standard LSTM equations in the notation of the PyTorch docs, listed in field order:
  --
  --   `f_t = sigmoid(W_f [x_t; h_{t-1}] + b_f)`    (forget gate)
  --   `i_t = sigmoid(W_i [x_t; h_{t-1}] + b_i)`    (input gate)
  --   `g_t = tanh(W_g [x_t; h_{t-1}] + b_g)`       (candidate / cell proposal)
  --   `o_t = sigmoid(W_o [x_t; h_{t-1}] + b_o)`    (output gate)
  --   `c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t`            (cell state update)
  --   `h_t = o_t ⊙ tanh(c_t)`                      (exposed hidden state)
  --
  -- The `cell` component is what lets information persist over long ranges.
  let concat := Tensor.cat (s := [0]) ⟨0, by simp⟩ ![inputSize, hiddenSize]
      (Fin.cons input (Fin.cons prevState.hidden finZeroElim)) [inputSize + hiddenSize]
  let f := Tensor.sigmoid (Linear.forward lstm.forget concat)
  let i := Tensor.sigmoid (Linear.forward lstm.input concat)
  let g := Tensor.tanh (Linear.forward lstm.candidate concat)
  let o := Tensor.sigmoid (Linear.forward lstm.output concat)
  let c := (f * prevState.cell) + (i * g)
  ⟨o * Tensor.tanh c, c⟩

/-- Run an LSTM cell over a length-`seqLen` input sequence, returning the hidden state of every
step and the final state. -/
def LSTM.sequence {seqLen inputSize hiddenSize : Nat}
  (lstm : LSTM α inputSize hiddenSize)
  (inputs : Tensor α [seqLen, inputSize])
  (initialState : State α hiddenSize) :
  (Tensor α [seqLen, hiddenSize] × State α hiddenSize) :=
  let (finalState, outputs) := Sequence.mapAccum seqLen initialState fun i previous =>
    let state := cell lstm (Tensor.unstack inputs i) previous
    (state, state.hidden)
  (Tensor.dim outputs.getScalar, finalState)

end TorchLean
