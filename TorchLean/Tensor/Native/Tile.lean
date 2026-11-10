/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Tensor.Native.Fold
public import TorchLean.Tensor.Einops.Lowering.Einsum
public import Batteries.Data.Vector.Lemmas -- shake: keep

/-!
# Width-polymorphic semantics for einsum output tiles

An output tile is a finite family of independent contraction accumulators, and
the definitions below cover any number of lanes. Concrete code generators may
keep selected widths in separate scalar registers, but their correctness proofs
reduce to these definitions and theorems.

No theorem in this module selects a hardware width or changes scalar
evaluation order.
-/

@[expose] public section

namespace TorchLean

universe u

/--
Add each lane's value to that lane's total in a contraction tile.

The lane values arrive as a function, so generated code can expose each
selected lane value without allocating an intermediate container.
-/
@[inline] def Native.updateTile
    {α : Type u} [Add α] {lanes : Nat}
    (totals : Vector α lanes) (values : Fin lanes → α) :
    Vector α lanes :=
  Vector.ofFn fun lane => totals[lane] + values lane

/-- Projecting a tile update gives the corresponding scalar update. -/
@[grind =] theorem Native.updateTile_get
    {α : Type u} [Add α] {lanes : Nat}
    (totals : Vector α lanes) (values : Fin lanes → α)
    (lane : Fin lanes) :
    (updateTile totals values)[lane] = totals[lane] + values lane := by
  simp [updateTile]

/--
A vector-valued coordinate fold is the vector of its scalar lane sums when
every update is pointwise addition.
-/
theorem Shape.Coord.foldl_vector
    {α : Type u} [Add α] {lanes : Nat} (shape : Shape)
    (update : Vector α lanes → Coord shape → Vector α lanes)
    (value : Fin lanes → Coord shape → α)
    (initial : Vector α lanes)
    (hUpdate :
      ∀ totals coordinate selectedLane,
        (update totals coordinate)[selectedLane] =
          totals[selectedLane] + value selectedLane coordinate) :
    foldl shape update initial =
      Vector.ofFn fun lane =>
        foldl shape
          (fun total coordinate => total + value lane coordinate)
          initial[lane] := by
  rw [← Vector.ofFn_getElem (xs := foldl shape update initial)]
  apply congrArg Vector.ofFn
  funext lane
  exact foldl_project shape
    (fun totals : Vector α lanes => totals[lane])
    update
    (fun total coordinate => total + value lane coordinate)
    initial (by
      intro totals coordinate
      exact (hUpdate totals coordinate lane).symm)

/--
An arbitrary-width contraction tile equals the vector of its scalar
contractions, so every lane retains the scalar fold's original coordinate
order.
-/
theorem Shape.Coord.foldl_updateTile
    {α : Type u} [Add α] {lanes : Nat}
    (shape : Shape) (values : Fin lanes → Coord shape → α)
    (initial : α) :
    foldl shape
        (fun totals coordinate =>
          Native.updateTile totals fun lane => values lane coordinate)
        (Vector.replicate lanes initial) =
      Vector.ofFn fun lane =>
        foldl shape
          (fun total coordinate => total + values lane coordinate)
          initial := by
  calc
    _ = Vector.ofFn fun lane =>
          foldl shape
            (fun total coordinate => total + values lane coordinate)
            (Vector.replicate lanes initial)[lane] := by
      apply foldl_vector
      intro totals coordinate lane
      exact Native.updateTile_get totals
        (fun selectedLane => values selectedLane coordinate) lane
    _ = _ := by simp

/--
Any concrete tile update that agrees pointwise with `updateTile` has the same
arbitrary-width contraction semantics.

Scalar-register kernels use this theorem as their only semantic boundary, so
their implementation-specific proofs need only identify one update step.
-/
theorem Shape.Coord.foldl_updateTile_of_eq
    {α : Type u} [Add α] {lanes : Nat}
    (shape : Shape)
    (update : Vector α lanes → Coord shape → Vector α lanes)
    (values : Fin lanes → Coord shape → α) (initial : α)
    (hUpdate :
      ∀ totals coordinate,
        update totals coordinate =
          Native.updateTile totals fun lane => values lane coordinate) :
    foldl shape update (Vector.replicate lanes initial) =
      Vector.ofFn fun lane =>
        foldl shape
          (fun total coordinate => total + values lane coordinate)
          initial := by
  have hUpdateFunction :
      update =
        fun totals coordinate =>
          Native.updateTile totals fun lane => values lane coordinate := by
    funext totals coordinate
    exact hUpdate totals coordinate
  rw [hUpdateFunction]
  exact foldl_updateTile shape values initial

end TorchLean
