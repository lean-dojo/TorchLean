/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.RuntimeApprox.NF.Ops

/-!
# Axis Softmax Bounds

NF (rounded) backend: notes on *axis softmax*.

TorchLean's NF proof stack provides end-to-end approximation bounds for the scalar
logistic helper used throughout the NF runtime layer and the autograd proofs.

Important: the NF node named `softmax` below is the scalar logistic-like function
`exp(x) / (exp(x) + 1)` (so its derivative is `s(x) * (1 - s(x))`). This is closer to what PyTorch
calls `torch.sigmoid` than to the vector-valued `torch.softmax` that normalizes over an axis.

The vector/axis softmax theorem family (normalization across a dimension) is separate:
it requires bounding the *coupled* denominator `∑ exp(xᵢ)` and tracking correlations between
  entries.
A future axis-softmax theorem should live in this file, next to the scalar logistic bounds, so the
two proof obligations stay visibly distinct.

For the scalar logistic bounds and nodes, see:
- `NN.Proofs.RuntimeApprox.NF.Ops` (`approxT_softmax_spec`, `softmaxNode`)

We do not state an axis-softmax theorem until the coupled-denominator proof is present: a fake
theorem here would blur the difference between scalar logistic-style bounds and the coupled vector
normalization used in attention.
- `NN.Proofs.RuntimeApprox.NF.BackwardOps` (`softmaxRevNode`)

PyTorch reference (axis softmax, for naming context):
`torch.nn.functional.softmax` (docs).
-/

@[expose] public section
