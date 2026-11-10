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
# Roundoff coverage of the general statistics and normalizations

`#roundoff_coverage` runs `roundoff` on the value of the variance and the mean, of layer, RMS, batch
and group normalization and of batch normalization with batch and with running statistics, at a
coordinate of the result. Shapes, axes, block sizes and every other parameter are variables, so each
theorem holds at every axis and rank. Each route is analyzed for the configured scalar of an
arbitrary IEEE format, for native `Float32` and for the rounded reals `FP32`. The theorems are
declared in the namespace `TorchLean.Coverage.TensorComposites`, which every
module under `TorchLean/Verify/Coverage/TensorComposites/` shares.

`#roundoff_propagation` declares the mixed propagation theorems of
`TorchLean.Normalization.normalize`, `TorchLean.Tensor.abs` and `TorchLean.Tensor.detach`, each an
error bound under real bounds on its inputs.
-/

public section

namespace TorchLean.Coverage.TensorComposites

open FloatLib.Floats FloatLib.Floats.Formats.BinaryInterchange

#roundoff_coverage
  statistics := Tensor.var, Tensor.mean
  normalization := Tensor.layerNorm, Tensor.rmsNorm, Tensor.batchNorm, Tensor.groupNorm

#roundoff_propagation
  mixed := Normalization.normalize, Tensor.abs, Tensor.detach

end TorchLean.Coverage.TensorComposites
