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
# Roundoff coverage of the general positional encodings

`#roundoff_coverage` runs `roundoff` on the value of the sinusoidal positional encoding, of the RoPE
factor tables and of RoPE, at a coordinate of the result. Shapes, axes, block sizes and every other
parameter are variables, so each theorem holds at every axis and rank. Each route is analyzed for
the configured scalar of an arbitrary IEEE format, for native `Float32` and for the rounded reals
`FP32`. The theorems are declared in the namespace
`TorchLean.Coverage.TensorComposites`, which every module under
`TorchLean/Verify/Coverage/TensorComposites/` shares.
-/

public section

namespace TorchLean.Coverage.TensorComposites

open FloatLib.Floats FloatLib.Floats.Formats.BinaryInterchange

#roundoff_coverage
  positional := Tensor.sinusoidalPositionalEncoding, Tensor.ropeCos, Tensor.ropeSin, Tensor.rope

end TorchLean.Coverage.TensorComposites
