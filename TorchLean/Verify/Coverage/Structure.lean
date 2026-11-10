/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Verify.RuntimeApprox.Roundoff.Coverage
public import TorchLean.NN.VJPOp
public import TorchLean.Tensor.Elementwise
public import TorchLean.Tensor.Movement
public import TorchLean.Tensor.Layers.Dropout
public import TorchLean.Tensor.Layers.Embedding
public import TorchLean.Tensor.Layers.Linear
public import TorchLean.Tensor.Layers.Loss
public import TorchLean.Tensor.Layers.Loss.Axis
public import TorchLean.Tensor.Layers.Softmax
public import TorchLean.NN.Models.MLP
public import TorchLean.NN.Models.LinearRegression
public import TorchLean.NN.FDeriv

/-!
# Roundoff coverage of the structure specifications

`#roundoff_coverage` runs `roundoff` on the forward map and the pullback of the structure
specifications: the composition combinators and the model specifications, for the configured
scalar of an arbitrary IEEE format, for native `Float32` and for the rounded reals
`FP32`. It declares a theorem for each route it proves.
-/

public section

namespace TorchLean.Coverage.Structure

open FloatLib.Floats FloatLib.Floats.Formats.BinaryInterchange

#roundoff_coverage
  composition := VJPOp.id, VJPOp.compose
  model := MLP.vjpOp, LinearRegression.vjpOp

end TorchLean.Coverage.Structure
