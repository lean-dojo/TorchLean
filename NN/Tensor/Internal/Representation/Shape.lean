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

/-- Horner-form row-major index: the result is `acc * size s` plus the index of the coordinate. -/
def linearizeAux : (s : Shape) → Nat → Coord s → Nat
  | [], acc, _ => acc
  | n :: s, acc, coordinate => linearizeAux s (acc * n + coordinate.1.val) coordinate.2

/-- The Horner accumulator adds `acc * size s` to the row-major index. -/
theorem linearizeAux_eq :
    (s : Shape) → (acc : Nat) → (coordinate : Coord s) →
      linearizeAux s acc coordinate = acc * Shape.size s + (linearize coordinate).val
  | [], acc, coordinate => by
      have h : (linearize (s := []) coordinate).val = 0 := by
        have := (linearize (s := []) coordinate).isLt
        simp only [Shape.size_nil] at this
        omega
      simp [linearizeAux, h]
  | n :: s, acc, (i, j) => by
      rw [linearizeAux, linearizeAux_eq s, linearize_cons_val, Shape.size_cons]
      simp only [Nat.add_mul, Nat.mul_assoc]
      rw [Nat.mul_comm i.val (Shape.size s)]
      omega

/--
Arithmetic row-major index of a coordinate.

This is `linearize` without the equivalence structures that `equivFin` builds at every level, and
it replaces `linearize` in compiled code.
-/
def linearizeFast {s : Shape} (coordinate : Coord s) : Fin (Shape.size s) :=
  ⟨linearizeAux s 0 coordinate, by
    rw [linearizeAux_eq, Nat.zero_mul, Nat.zero_add]
    exact (linearize coordinate).isLt⟩

/-- Compiled code runs `linearizeFast` in place of `linearize`. -/
@[csimp] theorem linearize_eq_linearizeFast : @linearize = @linearizeFast := by
  funext s coordinate
  apply Fin.ext
  simp [linearizeFast, linearizeAux_eq]

/-- Arithmetic inverse of the row-major index: divide by the tail size and recurse on the rest. -/
def unlinearizeAux : (s : Shape) → Fin (Shape.size s) → Coord s
  | [], _ => PUnit.unit
  | n :: s, index =>
      have h : index.val < n * Shape.size s := index.isLt
      have hTail : 0 < Shape.size s :=
        Nat.pos_of_ne_zero fun hZero => by simp [hZero] at h
      (⟨index.val / Shape.size s, (Nat.div_lt_iff_lt_mul hTail).2 h⟩,
        unlinearizeAux s ⟨index.val % Shape.size s, Nat.mod_lt _ hTail⟩)

/-- The arithmetic inverse is a right inverse of `linearize`. -/
theorem linearize_unlinearizeAux :
    (s : Shape) → (index : Fin (Shape.size s)) → linearize (unlinearizeAux s index) = index
  | [], index => by
      apply Fin.ext
      have h1 : index.val < 1 := index.isLt
      have h2 : (linearize (s := []) (unlinearizeAux [] index)).val < 1 :=
        (linearize (s := []) (unlinearizeAux [] index)).isLt
      omega
  | n :: s, index => by
      apply Fin.ext
      rw [unlinearizeAux, linearize_cons_val, linearize_unlinearizeAux s]
      exact Nat.mod_add_div _ _

/-- Arithmetic `unlinearize`, which replaces it in compiled code. -/
def unlinearizeFast {s : Shape} (index : Fin (Shape.size s)) : Coord s :=
  unlinearizeAux s index

/-- Compiled code runs `unlinearizeFast` in place of `unlinearize`. -/
@[csimp] theorem unlinearize_eq_unlinearizeFast : @unlinearize = @unlinearizeFast := by
  funext s index
  rw [unlinearizeFast, ← linearize_unlinearizeAux s index, unlinearize_linearize,
    linearize_unlinearizeAux]

/--
The row-major equivalence built from the arithmetic index maps. Compiled code uses it in place of
`equivFin`, whose `trans` and `prodCongr` layers are otherwise rebuilt at every application.
-/
def equivFinFast (s : Shape) : Coord s ≃ Fin (Shape.size s) where
  toFun := linearizeFast
  invFun := unlinearizeFast
  left_inv coordinate := by
    rw [← linearize_eq_linearizeFast, ← unlinearize_eq_unlinearizeFast]
    exact unlinearize_linearize coordinate
  right_inv index := by
    rw [← linearize_eq_linearizeFast, ← unlinearize_eq_unlinearizeFast]
    exact linearize_unlinearize index

/-- Compiled code runs `equivFinFast` in place of `equivFin`. -/
@[csimp] theorem equivFin_eq_equivFinFast : @equivFin = @equivFinFast := by
  funext s
  apply Equiv.ext
  intro coordinate
  change linearize coordinate = linearizeFast coordinate
  rw [linearize_eq_linearizeFast]

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
