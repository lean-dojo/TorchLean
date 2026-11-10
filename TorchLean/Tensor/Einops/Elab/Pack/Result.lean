/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Datatype

/-!
# Packed Tensors

The public result type shared by `pack` and `unpack`. Component shapes live in
the type, so ordinary unpacking does not expose an anonymous metadata tuple.
-/

@[expose] public section

namespace TorchLean.Tensor

/--
A tensor produced by `pack`, indexed by the star shape of every source tensor.

Use `unpack packed pattern` to recover the source tensors. The packed tensor is
available as `packed.tensor`, and `packed.shapes` exposes the recorded component shapes.

The `pack` elaborator computes this metadata, and `unpack` checks it against the pattern
and packed shape. The public constructor itself carries no proof that arbitrary
`componentShapes` describe a valid partition of `shape`.
-/
structure Packed (α : Type) (shape : Shape)
    (componentShapes : List Shape) [TorchLean.Storage α] where
  /-- Row-major tensor containing every packed component.
  Scalar storage is chosen by `Storage α`. -/
  tensor : TorchLean.Tensor α shape

namespace Packed

/-- Return the star shapes recorded in the component-shape index. -/
def shapes {α : Type} {shape : Shape}
    {componentShapes : List Shape} [TorchLean.Storage α]
    (_ : Packed α shape componentShapes) : List Shape :=
  componentShapes

end Packed

end TorchLean.Tensor
