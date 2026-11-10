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
# Roundoff coverage of softmax, log-softmax and log-sum-exp

`#roundoff_coverage` runs `roundoff` on the value of softmax (with and without a mask, by either
schedule), log-softmax and log-sum-exp along any axes and of their JVP and VJPs, at a coordinate of
the result. Shapes, axes, block sizes and every other parameter are variables, so each theorem holds
at every axis and rank. Each route is analyzed for the configured scalar of an arbitrary IEEE
format, for native `Float32` and for the rounded reals `FP32`. Softmax has one route per mask and
schedule constructor. The online schedule folds a triple of running values over its blocks, which
`roundoff` does not analyze. `TorchLean.Verify.RuntimeApprox.OnlineSoftmax` bounds the online
softmax weights by hand (`onlineWeight_error`). The theorems are declared in the namespace
`TorchLean.Coverage.TensorComposites`, which every module under
`TorchLean/Verify/Coverage/TensorComposites/` shares.
-/

public section

namespace TorchLean.Coverage.TensorComposites

open FloatLib.Floats FloatLib.Floats.Formats.BinaryInterchange

#roundoff_coverage
  softmax :=
    Tensor.softmax, Tensor.softmaxBackward, Tensor.logSoftmax, Tensor.logSoftmaxJVP,
    Tensor.logSoftmaxBackward, Tensor.logsumexp

end TorchLean.Coverage.TensorComposites
