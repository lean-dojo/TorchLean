/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Layers.Activation
public import TorchLean.Tensor.Layers.Linear
public import TorchLean.Tensor.ConcatSlice

/-!
# RNN (spec layer)

The Elman cell `RNN.cell` and its unrolling over time `RNN.sequence`. The cell's parameters
are one `Linear` on the concatenated vector `[x_t; h_{t-1}]`. Gradients come from the tape:
`TorchLean/NN/Layer/Recurrent/Correctness.lean` proves the runtime cell `rnnCell` and its scan
equal to these definitions under the pure interpreter (`nn.Layer.rnnCell_eq` and
`Recurrent.scan_rnnCell`).

PyTorch analogy:

- `RNN.cell` corresponds to `torch.nn.RNNCell` with `nonlinearity="tanh"`.
- `RNN.sequence` corresponds to `torch.nn.RNN` unrolled over `seqLen`.

By hand, from `torch.nn.RNNCell`'s `h' = tanh(W_ih x + b_ih + W_hh h + b_hh)` (PyTorch 2.8 docs),
with our single bias `b` standing for `b_ih + b_hh`. The `relu` option has no counterpart here, and
`RNN.sequence` takes `initialHidden` from the caller, where `torch.nn.RNN` starts from zeros when
`hx` isn't given. No test we found compares them with PyTorch.

## References

- Elman, "Finding Structure in Time", Cognitive Science 14(2):179-211, 1990:
  https://doi.org/10.1207/s15516709cog1402_1
- PyTorch 2.8 `RNNCell`: https://docs.pytorch.org/docs/2.8/generated/torch.nn.RNNCell.html
- PyTorch 2.8 `RNN`: https://docs.pytorch.org/docs/2.8/generated/torch.nn.RNN.html
-/

@[expose] public section


namespace TorchLean

open TorchLean.Tensor

variable {α : Type} [Storage α] [Context α]

/-!
## Recurrent tensor shapes

Recurrent vectors and matrices use ordinary rank-one and rank-two tensors. Sequences are
time-major, with `seqLen` as the outermost axis, because that layout follows the step-by-step
recursion of the definitions and proofs.
-/

/--
Single RNN cell forward pass, `h_t = tanh(W [x_t; h_{t-1}] + b)`, where `(W, b)` is one
`Linear` on the concatenated vector `[x_t; h_{t-1}]`.

Over the reals this is the common split-parameter form `h_t = tanh(W_ih x_t + W_hh h_{t-1} + b)`.
The fused form accumulates one dot product across the input and hidden coordinates, and two
separately rounded products and their sum can give different IEEE results.

PyTorch analogy: `RNNCell(input, hidden)` with `tanh` nonlinearity.
-/
def RNN.cell {inputSize hiddenSize : Nat}
  (rnn : Linear α (inputSize + hiddenSize) hiddenSize)
  (input : Tensor α [inputSize])
  (hidden : Tensor α [hiddenSize]) :
  Tensor α [hiddenSize] :=
  let concat := Tensor.cat (s := [0]) ⟨0, by simp⟩ ![inputSize, hiddenSize]
      (Fin.cons input (Fin.cons hidden finZeroElim)) [inputSize + hiddenSize]
  Tensor.tanh (Linear.forward rnn concat)

/--
Unroll an RNN over `seqLen` steps (time-major) from `initialHidden`.

Returns the sequence of hidden states `[h_0, ..., h_{seqLen-1}]` and the final hidden state, which
is `initialHidden` for an empty sequence.
-/
def RNN.sequence {seqLen inputSize hiddenSize : Nat}
  (rnn : Linear α (inputSize + hiddenSize) hiddenSize)
  (inputs : Tensor α [seqLen, inputSize])
  (initialHidden : Tensor α [hiddenSize]) :
  Tensor α [seqLen, hiddenSize] × Tensor α [hiddenSize] :=
  let (finalHidden, outputs) := Sequence.mapAccum seqLen initialHidden fun i previous =>
    let hidden := cell rnn (Tensor.unstack inputs i) previous
    (hidden, hidden)
  (Tensor.dim outputs.getScalar, finalHidden)

end TorchLean
