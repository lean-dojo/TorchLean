/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.IR.Infer

/-!
# IR Shape Contract Tests

Regression checks for graph-level shape inference. These examples ensure that malformed nodes are
rejected and that the documented empty-output and broadcasting semantics remain stable.
-/

@[expose] public section

namespace NN.Tests.IR.ShapeContracts

open NN
open NN.IR
open Spec TorchLean

private def smallSpatialInput : Spec.Shape := .dim 1 (.dim 2 (.dim 2 .scalar))

private def squareWindowConfig (kernel stride padding : Nat) : WindowConfig :=
  { spatialRank := 2
    kernel := TorchLean.Tensor.ofFn fun _ => kernel
    stride := TorchLean.Tensor.ofFn fun _ => stride
    padding := TorchLean.Tensor.ofFn fun _ => padding }

private def squareConvConfig
    (inChannels outChannels kernel stride padding : Nat) : ConvConfig :=
  { spatialRank := 2
    kernel := TorchLean.Tensor.ofFn fun _ => kernel
    stride := TorchLean.Tensor.ofFn fun _ => stride
    padding := TorchLean.Tensor.ofFn fun _ => padding
    channelAxis := 0
    inChannels := inChannels
    outChannels := outChannels }

private def node (kind : OpKind) (outShape : Spec.Shape) : Node :=
  { id := 0, parents := #[0], kind := kind, outShape := outShape }

example :
    (Infer.nodeOutShape
      (node (.conv (squareConvConfig 1 1 3 1 0)) smallSpatialInput)
      #[smallSpatialInput]).isOk = false := by
  decide

example :
    Infer.nodeOutShape
      (node (.maxPool (squareWindowConfig 3 1 0)) smallSpatialInput) #[smallSpatialInput] =
      .ok (.dim 1 (.dim 0 (.dim 0 .scalar))) := by
  rfl

example :
    (Infer.nodeOutShape
      (node (.broadcastTo (.dim 2 .scalar) (.dim 3 .scalar)) (.dim 3 .scalar))
      #[.dim 2 .scalar]).isOk = false := by
  decide

example :
    Infer.nodeOutShape
        (node (.broadcastTo .scalar (.dim 3 .scalar)) (.dim 3 .scalar))
        #[.scalar] =
      .ok (.dim 3 .scalar) := by
  rfl

example :
    (Infer.nodeOutShape (node (.layernorm 1) (.dim 0 .scalar)) #[.dim 0 .scalar]).isOk =
      false := by
  decide

/-! Pointwise operators: `op.arity` parents of one shape, which is the result shape. -/

example :
    Infer.nodeOutShape (node (.pointwise (.div .floor)) (.dim 2 .scalar))
        #[.dim 2 .scalar, .dim 2 .scalar] =
      .ok (.dim 2 .scalar) := by
  rfl

example :
    (Infer.nodeOutShape (node (.pointwise .pow) (.dim 2 .scalar))
        #[.dim 2 .scalar, .scalar]).isOk = false := by
  decide

example :
    (Infer.nodeOutShape (node (.pointwise .neg) (.dim 2 .scalar))
        #[.dim 2 .scalar, .dim 2 .scalar]).isOk = false := by
  decide

example :
    (Infer.nodeOutShape (node (.pointwise (.gelu .tanh)) .scalar) #[]).isOk = false := by
  decide

end NN.Tests.IR.ShapeContracts
