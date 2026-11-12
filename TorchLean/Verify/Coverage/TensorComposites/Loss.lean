/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Verify.RuntimeApprox.Roundoff.Coverage
public import TorchLean.Tensor.Layers.Embedding
public import TorchLean.Tensor.Layers.Linear
public import TorchLean.Tensor.Layers.Attention
public import TorchLean.Tensor.Layers.MoE
public import TorchLean.Tensor.Layers.PositionalEncoding

/-!
# Roundoff coverage of the general cross-entropy and cosine losses

`#roundoff_coverage` runs `roundoff` on the value of cross-entropy for probabilities and for logits
and of its gradients, and of the cosine loss, at a coordinate of the result. For logits the
log-probabilities are those of `CrossEntropy.vectorLogProbs`: the one-level `LogSoftmax.vec` by
default, and the two-level normalizer when another class exists, neither the pivot logit nor the
runner-up is infinite, the computed sum of the exponentials other than the pivot's is zero, and
the computed exponential of the log-sum-exp of the other classes and every entry of the two-level
result are finite. `roundoff` analyzes both branches and keeps the test `CrossEntropy.twoLevelAt`
as one opaque branch condition. Shapes, axes, block sizes and every other parameter are variables,
so each theorem holds at every axis and rank. Each route is analyzed for the configured scalar of an
arbitrary IEEE format, for native `Float32` and for the rounded reals `FP32`. The theorems are
declared in the namespace `TorchLean.Coverage.TensorComposites`, which every module under
`TorchLean/Verify/Coverage/TensorComposites/` shares.
-/

public section

namespace TorchLean.Coverage.TensorComposites

open FloatLib.Floats FloatLib.Floats.Formats.BinaryInterchange

#roundoff_coverage
  loss :=
    Tensor.crossEntropy, Tensor.crossEntropyBackward, Tensor.crossEntropyTargetBackward,
    Tensor.cosineLoss

end TorchLean.Coverage.TensorComposites
