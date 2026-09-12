/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Tensor.Internal.Elab.Einsum.Loop
public import NN.Tensor.Internal.Lowering.Einsum
public import Batteries.Data.Vector.Lemmas -- shake: keep
public import Init.Data.Vector.OfFn -- shake: keep

/-!
# Width-polymorphic semantics for einsum output tiles

An output tile is a finite family of independent contraction accumulators.
This module describes that family for an arbitrary number of lanes. Concrete
code generators may keep selected widths in separate scalar registers, but
their correctness proofs reduce to the definitions and theorems here.

No theorem in this module selects a hardware width or changes scalar
evaluation order.
-/

@[expose] public section

namespace TorchLean.Tensor.Internal.Elab.Impl

universe u

/--
Add one value to every lane of a contraction tile.

The lane family is a function rather than another vector so generated code
can expose each selected lane value without allocating an intermediate
container.
-/
@[inline] def updateTile
    {α : Type u} [Add α] {lanes : Nat}
    (totals : Vector α lanes) (values : Fin lanes → α) :
    Vector α lanes :=
  Vector.ofFn fun lane => totals[lane] + values lane

/-- Projecting a tile update gives the corresponding scalar update. -/
@[grind =] theorem updateTile_get
    {α : Type u} [Add α] {lanes : Nat}
    (totals : Vector α lanes) (values : Fin lanes → α)
    (lane : Fin lanes) :
    (updateTile totals values)[lane] = totals[lane] + values lane := by
  simp [updateTile]

/-- Equal lane families determine equal function-backed vectors. -/
theorem ofFn_congr
    {α : Type u} {lanes : Nat} {values reference : Fin lanes → α}
    (h : values = reference) :
    Vector.ofFn values = Vector.ofFn reference :=
  congrArg Vector.ofFn h

/--
The additive coordinate fold is the executable presentation of
`Semantics.coordinateSum`.
-/
theorem coordinateFoldl_add_eq_coordinateSum
    {α : Type u} [Add α] [OfNat α 0] (shape : Shape)
    (values : Coord shape → α) (initial : α) :
    coordinateFoldl shape
        (fun total coordinate => total + values coordinate)
        initial =
      Semantics.coordinateSum shape values initial := by
  induction shape generalizing initial with
  | nil => simp only [coordinateFoldl, Semantics.coordinateSum]
  | cons length shape inductionHypothesis =>
      simp only [coordinateFoldl, Semantics.coordinateSum]
      apply congrArg fun step => Fin.foldl length step initial
      funext total head
      exact inductionHypothesis
        (fun tail => values (head, tail)) total

/--
A vector-valued coordinate fold is the vector of its scalar lane sums when
every update is pointwise addition.
-/
theorem coordinateFoldl_vector
    {α : Type u} [Add α] [OfNat α 0] {lanes : Nat} (shape : Shape)
    (update : Vector α lanes → Coord shape → Vector α lanes)
    (value : Fin lanes → Coord shape → α)
    (initial : Vector α lanes)
    (hUpdate :
      ∀ totals coordinate selectedLane,
        (update totals coordinate)[selectedLane] =
          totals[selectedLane] + value selectedLane coordinate) :
    coordinateFoldl shape update initial =
      Vector.ofFn fun lane =>
        Semantics.coordinateSum shape (value lane) initial[lane] := by
  rw [← Vector.ofFn_getElem (xs := coordinateFoldl shape update initial)]
  apply congrArg Vector.ofFn
  funext lane
  calc
    _ = coordinateFoldl shape
          (fun total coordinate => total + value lane coordinate)
          initial[lane] := by
      exact coordinateFoldl_project shape
        (fun totals : Vector α lanes => totals[lane])
        update
        (fun total coordinate => total + value lane coordinate)
        initial (by
          intro totals coordinate
          exact (hUpdate totals coordinate lane).symm)
    _ = _ := coordinateFoldl_add_eq_coordinateSum _ _ _

/--
An arbitrary-width contraction tile equals the vector of its scalar
contractions. Every lane therefore retains the scalar fold's original
coordinate order.
-/
theorem coordinateFoldl_updateTile
    {α : Type u} [Add α] [OfNat α 0] {lanes : Nat}
    (shape : Shape) (values : Fin lanes → Coord shape → α)
    (initial : α) :
    coordinateFoldl shape
        (fun totals coordinate =>
          updateTile totals fun lane => values lane coordinate)
        (Vector.replicate lanes initial) =
      Vector.ofFn fun lane =>
        Semantics.coordinateSum shape (values lane) initial := by
  calc
    _ = Vector.ofFn fun lane =>
          Semantics.coordinateSum shape
            (values lane) (Vector.replicate lanes initial)[lane] := by
      apply coordinateFoldl_vector
      intro totals coordinate lane
      exact updateTile_get totals
        (fun selectedLane => values selectedLane coordinate) lane
    _ = _ := by simp

/--
Any concrete tile update that agrees pointwise with `updateTile` has the same
arbitrary-width contraction semantics.

Scalar-register kernels use this theorem as their only semantic boundary.
Their implementation-specific proof need only identify one update step.
-/
theorem coordinateFoldl_updateTile_of_eq
    {α : Type u} [Add α] [OfNat α 0] {lanes : Nat}
    (shape : Shape)
    (update : Vector α lanes → Coord shape → Vector α lanes)
    (values : Fin lanes → Coord shape → α) (initial : α)
    (hUpdate :
      ∀ totals coordinate,
        update totals coordinate =
          updateTile totals fun lane => values lane coordinate) :
    coordinateFoldl shape update (Vector.replicate lanes initial) =
      Vector.ofFn fun lane =>
        Semantics.coordinateSum shape (values lane) initial := by
  have hUpdateFunction :
      update =
        fun totals coordinate =>
          updateTile totals fun lane => values lane coordinate := by
    funext totals coordinate
    exact hUpdate totals coordinate
  rw [hUpdateFunction]
  exact coordinateFoldl_updateTile shape values initial

end TorchLean.Tensor.Internal.Elab.Impl
