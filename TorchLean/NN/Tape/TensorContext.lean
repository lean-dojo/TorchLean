/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Tape.Idx
public import TorchLean.NN.TypedGraph.Context

/-!
# Simp lemmas for tensor lookups and contexts

The tensor lookups and contexts of `TorchLean.NN.TypedGraph.Context` convert to and
from typed packs. This module proves the simp lemmas for those conversions.

## Main declarations

- `TensorLookup.read_ofPack`, `TensorLookup.toPack_ofPack` and `TensorLookup.ofPack_toPack`: lookups
  of a pack read its typed selection, and the conversions are mutually inverse.
- `TensorContext.toPack_cast` and `TensorContext.toPack_push`: `toPack` turns `cast` into
  `Pack.cast` and `push` into `Pack.snoc`.
-/

@[expose] public section

namespace TorchLean.Tape

namespace TensorLookup

variable {α : Type} [Storage α]

/-- Reading a pack through the lookup agrees with its typed selection. -/
@[simp] theorem read_ofPack {shapes : List Shape} (xs : TensorPack α shapes)
    {shape : Shape} (idx : Idx shapes shape) :
    (ofPack xs).read idx = TensorPack.getIdx xs idx := rfl

/-- Materializing the lookup of a pack gives the pack. -/
@[simp] theorem toPack_ofPack {shapes : List Shape} (xs : TensorPack α shapes) :
    (ofPack xs).toPack = xs := by
  apply TensorPack.ext_getIdx
  intro shape idx
  simp only [getIdx_toPack, ofPack]

/-- The lookup of a materialized reader is the reader. -/
@[simp] theorem ofPack_toPack {shapes : List Shape} (lookup : TensorLookup α shapes) :
    ofPack lookup.toPack = lookup := by
  apply TensorLookup.ext
  funext shape idx
  exact getIdx_toPack lookup idx

end TensorLookup

namespace TensorContext

variable {α : Type} [Storage α] {shapes : List Shape}

/-- `toPack` turns `TensorContext.cast` into `Pack.cast`. -/
@[simp] theorem toPack_cast {other : List Shape} (same : shapes = other)
    (ctx : TensorContext α shapes) :
    (ctx.cast same).toPack = Pack.cast same ctx.toPack := by
  cases same
  rfl

/-- `toPack` turns `TensorContext.push` into `Pack.snoc`. -/
@[simp] theorem toPack_push {shape : Shape} (ctx : TensorContext α shapes)
    (value : Tensor α shape) :
    (ctx.push value).toPack = ctx.toPack.snoc value := by
  conv_lhs => rw [← ofPack_toPack ctx]
  simp only [push_ofPack, toPack_ofPack]

end TensorContext

end TorchLean.Tape
