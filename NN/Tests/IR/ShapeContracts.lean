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

private def rejects (x : Except String Spec.Shape) : Bool :=
  match x with
  | .error _ => true
  | .ok _ => false

example :
    rejects
        (Infer.nodeOutShape
          (node (.conv (squareConvConfig 1 1 3 1 0)) smallSpatialInput)
          #[smallSpatialInput]) = true := by
  decide

example :
    rejects (Infer.nodeOutShape
      (node (.maxPool (squareWindowConfig 3 1 0)) smallSpatialInput) #[smallSpatialInput]) =
      false := by
  decide

example :
    rejects
      (Infer.nodeOutShape
        (node (.broadcastTo (.dim 2 .scalar) (.dim 3 .scalar)) (.dim 3 .scalar))
        #[.dim 2 .scalar]) = true := by
  decide

example :
    Infer.nodeOutShape
        (node (.broadcastTo .scalar (.dim 3 .scalar)) (.dim 3 .scalar))
        #[.scalar] =
      .ok (.dim 3 .scalar) := by
  decide

example :
    rejects (Infer.nodeOutShape (node (.layernorm 1) (.dim 0 .scalar)) #[.dim 0 .scalar]) =
      true := by
  decide

end NN.Tests.IR.ShapeContracts
