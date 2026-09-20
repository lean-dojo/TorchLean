/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

import Mathlib.Tactic.Bound.Init
public import NN.Spec.Core.Tensor -- shake: keep

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
available as `packed.tensor`, and `packed.shapes` exposes the certified
component shapes when an application needs to inspect them.
-/
structure Packed (α : Type) (shape : Spec.Shape)
    (componentShapes : List Spec.Shape) [TorchLean.Storage α] where
  /-- Contiguous tensor containing every packed component. -/
  tensor : TorchLean.Tensor α shape

namespace Packed

/-- Return the certified star shape of every component. -/
def shapes {α : Type} {shape : Spec.Shape}
    {componentShapes : List Spec.Shape} [TorchLean.Storage α]
    (_ : Packed α shape componentShapes) : List Spec.Shape :=
  componentShapes

end Packed

end TorchLean.Tensor
