/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import Mathlib.Data.Fintype.Prod
public import Mathlib.Logic.Equiv.Fin.Basic

/-!
# Static tensor shapes and coordinates

This module gives the finite geometry used by every TorchLean.Tensor.Internal denotation.
A shape is an outermost-first list of natural-number dimensions. Its
coordinates are the corresponding iterated product of finite types:

```text
Coord [d0, ..., dn] = Fin d0 x ... x Fin dn x PUnit.
```

The empty shape therefore has one coordinate, while a shape containing a
zero-length axis has none. `Coord.equivFin` uses mathlib's
`finProdFinEquiv`, so linear indices follow the conventional row-major order.

The representation is intentionally self-contained. Static shape expressions
and their coordinate spaces are the shared foundation for semantics, lowering,
and native execution.

## References

The row-major grouping convention follows einops v0.8.2, especially the
reshape stages in `einops.einops._reconstruct_from_shape_uncached` and
`einops.einops._apply_recipe`, pinned at commit
`8e911db71f2e693a0c434b041180388c685ed06f`.
-/

@[expose] public section

namespace TorchLean.Tensor.Internal

/-- A tensor shape, listed from the outermost axis to the innermost axis. -/
abbrev Shape := List Nat

namespace Shape

/-- The number of scalar entries in a shape. The empty product is one. -/
def size : Shape → Nat
  | [] => 1
  | n :: s => n * size s

/-- The number of axes in a shape. -/
def rank (s : Shape) : Nat :=
  s.length

/-- The rank-zero shape contains one scalar entry. -/
@[simp] theorem size_nil : size [] = 1 := rfl

/-- Prepending an axis multiplies the number of entries by its length. -/
@[simp] theorem size_cons (n : Nat) (s : Shape) : size (n :: s) = n * size s := rfl

/-- The empty shape has rank zero. -/
@[simp] theorem rank_nil : rank [] = 0 := rfl

/-- Prepending an axis increases the rank by one. -/
@[simp] theorem rank_cons (n : Nat) (s : Shape) : rank (n :: s) = rank s + 1 := by
  simp [rank]

/-- Shape size is the ordinary product of its dimension list. -/
theorem size_eq_prod (s : Shape) : size s = s.prod := by
  induction s with
  | nil => rfl
  | cons dimension shape ih =>
      simp only [size_cons, List.prod_cons, ih]

/-- Concatenating shapes multiplies their numbers of entries. -/
@[simp] theorem size_append (s t : Shape) : size (s ++ t) = size s * size t := by
  induction s with
  | nil => simp
  | cons n s ih => simp [ih, Nat.mul_assoc]

/-- A shape has no entries exactly when one of its axes has length zero. -/
theorem size_eq_zero_iff {s : Shape} : size s = 0 ↔ 0 ∈ s := by
  induction s with
  | nil => simp
  | cons n s ih =>
      rw [size_cons, List.mem_cons, Nat.mul_eq_zero, ih]
      exact or_congr eq_comm Iff.rfl

end Shape

/--
A coordinate of a finite shape.

Rank-zero coordinates are represented by `PUnit`. A zero-length dimension
contributes `Fin 0`, making the entire coordinate type empty.
-/
@[reducible]
def Coord : Shape → Type
  | [] => PUnit
  | n :: s => Fin n × Coord s

namespace Coord

/-- Coordinates of every finite shape have decidable equality. -/
instance instDecidableEq : (s : Shape) → DecidableEq (Coord s)
  | [] => by
      change DecidableEq PUnit
      infer_instance
  | n :: s => by
      change DecidableEq (Fin n × Coord s)
      letI := instDecidableEq s
      infer_instance

/-- Coordinates of every finite shape can be enumerated. -/
@[reducible] instance instFintype : (s : Shape) → Fintype (Coord s)
  | [] => by
      change Fintype PUnit
      infer_instance
  | n :: s => by
      change Fintype (Fin n × Coord s)
      letI := instFintype s
      infer_instance

/-- The coordinate type has exactly as many elements as the shape specifies. -/
@[simp] theorem card (s : Shape) : Fintype.card (Coord s) = Shape.size s := by
  induction s with
  | nil => simp [Coord, Shape.size]
  | cons n s ih =>
      change Fintype.card (Fin n × Coord s) = n * Shape.size s
      rw [Fintype.card_prod, Fintype.card_fin, ih]

/--
The row-major equivalence between multidimensional coordinates and flat
indices.

For a nonempty shape, the outer coordinate is the high-order component and
the tail coordinate is the low-order component.
-/
def equivFin : (s : Shape) → Coord s ≃ Fin (Shape.size s)
  | [] => (Equiv.equivPUnit (Fin 1)).symm
  | n :: s =>
      (Equiv.prodCongr (Equiv.refl (Fin n)) (equivFin s)).trans
        (finProdFinEquiv :
          Fin n × Fin (Shape.size s) ≃ Fin (n * Shape.size s))

/-- Convert a multidimensional coordinate to its row-major flat index. -/
def linearize {s : Shape} : Coord s → Fin (Shape.size s) :=
  equivFin s

/-- Recover a multidimensional coordinate from a row-major flat index. -/
def unlinearize {s : Shape} : Fin (Shape.size s) → Coord s :=
  (equivFin s).symm

/-- Unlinearizing and then linearizing recovers the flat index. -/
@[simp] theorem linearize_unlinearize {s : Shape} (i : Fin (Shape.size s)) :
    linearize (unlinearize i) = i :=
  (equivFin s).apply_symm_apply i

/-- Linearizing and then unlinearizing recovers the multidimensional coordinate. -/
@[simp] theorem unlinearize_linearize {s : Shape} (i : Coord s) :
    unlinearize (linearize i) = i :=
  (equivFin s).symm_apply_apply i

/-- Linearization is injective. -/
theorem linearize_injective {s : Shape} : Function.Injective (@linearize s) :=
  (equivFin s).injective

/-- Linearization is surjective. -/
theorem linearize_surjective {s : Shape} : Function.Surjective (@linearize s) :=
  (equivFin s).surjective

/-- The recursive linearization step is mathlib's row-major product index. -/
@[simp] theorem linearize_cons {n : Nat} {s : Shape} (i : Fin n) (j : Coord s) :
    linearize (s := n :: s) (i, j) = finProdFinEquiv (i, linearize j) := by
  change
    finProdFinEquiv ((Equiv.prodCongr (Equiv.refl (Fin n)) (equivFin s)) (i, j)) =
      finProdFinEquiv (i, (equivFin s) j)
  rfl

/--
The numerical row-major index of a coordinate is its tail index plus the
outer coordinate times the size of the tail shape.
-/
theorem linearize_cons_val {n : Nat} {s : Shape}
    (i : Fin n) (j : Coord s) :
    (linearize (s := n :: s) (i, j)).val =
      (linearize j).val + Shape.size s * i.val := by
  simp only [linearize_cons, finProdFinEquiv_apply_val]

/--
Project a coordinate from a dimensionwise broadcast target to its source.

Each source dimension must either equal the corresponding target dimension
or be a singleton. Equal dimensions retain the target coordinate; singleton
dimensions select their unique coordinate. Recursion on the two shape lists
keeps the proof in `Prop` while constructing coordinate data in `Type`.
-/
def broadcast :
    (sourceShape targetShape : Shape) →
      List.Forall₂
        (fun sourceLength targetLength =>
          sourceLength = targetLength ∨ sourceLength = 1)
        sourceShape targetShape →
      Coord targetShape → Coord sourceShape
  | [], [], _, _ => PUnit.unit
  | [], _ :: _, hShape, _ =>
      False.elim (by simpa using hShape.length_eq)
  | _ :: _, [], hShape, _ =>
      False.elim (by simpa using hShape.length_eq)
  | sourceLength :: sourceShape, targetLength :: targetShape,
      hShape, targetCoordinate =>
    if hEqual : sourceLength = targetLength then
      (Fin.cast hEqual.symm targetCoordinate.1,
        broadcast sourceShape targetShape
          (List.forall₂_cons.mp hShape).2 targetCoordinate.2)
    else
      have hSingleton : sourceLength = 1 :=
        (List.forall₂_cons.mp hShape).1.resolve_left hEqual
      (Fin.cast hSingleton.symm ⟨0, Nat.zero_lt_one⟩,
        broadcast sourceShape targetShape
          (List.forall₂_cons.mp hShape).2 targetCoordinate.2)

/--
Broadcasting a coordinate between identical shapes leaves every dimension
unchanged, independently of the proof used to certify compatibility.
-/
@[simp] theorem broadcast_self {shape : Shape}
    (hShape :
      List.Forall₂
        (fun sourceLength targetLength =>
          sourceLength = targetLength ∨ sourceLength = 1)
        shape shape)
    (coordinate : Coord shape) :
    broadcast shape shape hShape coordinate = coordinate := by
  induction shape with
  | nil =>
      cases coordinate
      rfl
  | cons length shape induction =>
      rcases coordinate with ⟨head, tail⟩
      simp [broadcast, induction (List.forall₂_cons.mp hShape).2 tail]

/-- A coordinate of a zero-size shape would yield an element of `Fin 0`. -/
theorem elim_of_size_eq_zero {s : Shape} (h : Shape.size s = 0) (i : Coord s) : False := by
  have flat : Fin (Shape.size s) := linearize i
  exact Fin.elim0 (h ▸ flat)

/-- A shape has a coordinate exactly when its size is positive. -/
theorem nonempty_iff_size_pos {s : Shape} : Nonempty (Coord s) ↔ 0 < Shape.size s := by
  rw [← card, Fintype.card_pos_iff]

/-- A shape has no coordinates exactly when its size is zero. -/
theorem isEmpty_iff_size_eq_zero {s : Shape} : IsEmpty (Coord s) ↔ Shape.size s = 0 := by
  rw [← not_nonempty_iff, nonempty_iff_size_pos, Nat.not_lt]
  omega

end Coord

end TorchLean.Tensor.Internal
