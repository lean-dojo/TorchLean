/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Tensor.Rep.Pointwise
public import Mathlib.Data.Fintype.BigOperators

/-!
# Tensor Pullbacks, Reindexing, and Reshape

A pullback reads each output entry from the source at a mapped coordinate, or at a mapped
row-major flat index. Broadcast and equivalence-based reindexing are pullbacks along particular
coordinate maps, and pointwise maps commute with all of them. Reshape is zero-copy: it keeps the
source buffer, and `reshape_eq_reindex` identifies it with a coordinate reindexing for proofs.

The module also holds `dot`, the finite sum of corresponding entry products, with its stacking and
reindexing laws `dot_stack` and `dot_reindex_reindex`.
-/

@[expose] public section

namespace TorchLean.Tensor.Rep

universe u v w

section

open Shape (Coord)

/--
Pull a tensor back along a coordinate map: the entry at output coordinate `i` is the source entry
at `f i`.

Rearrangement and coordinate replication are both defined as pullbacks.
-/
def pull {α : Type u} [Storage α] {s t : Shape} (f : Coord t → Coord s) (x : Rep α s) :
    Rep α t :=
  ofFn fun i => x (f i)

/-- Evaluating a pullback selects the source coordinate given by its map. -/
@[simp, grind =] theorem pull_apply {α : Type u} [Storage α] {s t : Shape}
    (f : Coord t → Coord s)
    (x : Rep α s) (i : Coord t) : pull f x i = x (f i) := by
  simp [pull]

/--
Pull a tensor back along a row-major flat-index map.

This is the native execution form for coordinate programs whose compiler has
already certified their linear-index behavior. It fills one output array and
reads the source array directly, without constructing multidimensional
coordinates in the scalar loop.
-/
def pullFlat {α : Type u} [Storage α] {s t : Shape}
    (f : Fin (Shape.size t) → Fin (Shape.size s))
    (x : Rep α s) : Rep α t :=
  ofFlatFn fun outputIndex => x.getFlat (f outputIndex)

/-- Reading a flat pullback applies its certified source-index map. -/
@[simp, grind =] theorem getFlat_pullFlat {α : Type u} [Storage α] {s t : Shape}
    (f : Fin (Shape.size t) → Fin (Shape.size s))
    (x : Rep α s) (outputIndex : Fin (Shape.size t)) :
    getFlat (pullFlat f x) outputIndex =
      x.getFlat (f outputIndex) := by
  simp [pullFlat]

/--
A flat-index pullback equals its coordinate counterpart when both select the
same source index for every row-major output index.
-/
theorem pullFlat_eq_pull {α : Type u} [Storage α] {s t : Shape}
    (flatMap : Fin (Shape.size t) → Fin (Shape.size s))
    (coordinateMap : Coord t → Coord s)
    (hMap :
      ∀ outputIndex,
        flatMap outputIndex =
          Coord.linearize
            (coordinateMap (Coord.unlinearize outputIndex)))
    (x : Rep α s) :
    pullFlat flatMap x = pull coordinateMap x := by
  ext outputCoordinate
  simp only [pullFlat, get_ofFlatFn, pull_apply]
  rw [hMap (Coord.linearize outputCoordinate)]
  simp only [Coord.unlinearize_linearize, get]

/--
Successive flat-index pullbacks compose into one source-index calculation.

The first map selects entries from the original source, while the second map
selects entries from the intermediate tensor.
-/
theorem pullFlat_comp {α : Type u} [Storage α] {r s t : Shape}
    (first : Fin (Shape.size s) → Fin (Shape.size r))
    (second : Fin (Shape.size t) → Fin (Shape.size s))
    (x : Rep α r) :
    pullFlat second (pullFlat first x) =
      pullFlat (first ∘ second) x := by
  ext outputCoordinate
  simp only [get, getFlat_pullFlat, Function.comp_apply]

/-- Pointwise equal flat-index programs produce equal pullback tensors. -/
theorem pullFlat_congr {α : Type u} [Storage α] {s t : Shape}
    (first second : Fin (Shape.size t) → Fin (Shape.size s))
    (hMap : ∀ outputIndex, first outputIndex = second outputIndex)
    (x : Rep α s) :
    pullFlat first x = pullFlat second x := by
  ext outputCoordinate
  simp only [get, getFlat_pullFlat]
  rw [hMap]

/--
Broadcast a tensor dimensionwise.

Source and target shapes have the same rank, and in each dimension the source length either
equals the target length or is one. Singleton dimensions are replicated, which needs no operation
on the scalar type. A leading rank expansion is a reshape that adds singleton dimensions, followed
by this broadcast.
-/
def broadcast {α : Type u} [Storage α] {sourceShape targetShape : Shape}
    (hShape :
      List.Forall₂
        (fun sourceLength targetLength =>
          sourceLength = targetLength ∨ sourceLength = 1)
        sourceShape targetShape)
    (sourceTensor : Rep α sourceShape) : Rep α targetShape :=
  pull (Coord.broadcast sourceShape targetShape hShape) sourceTensor

/-- Pulling a tensor along the identity coordinate map changes nothing. -/
@[simp, grind =, einops =, einops_simps]
theorem pull_id {α : Type u} [Storage α] {s : Shape} (x : Rep α s) :
    pull id x = x := by
  ext i
  simp

/--
Pullbacks compose in the reverse order of their coordinate maps.

The law is not a global `grind` rule because repeated e-matching can synthesize arbitrarily
nested function compositions.
-/
@[einops_simps] theorem pull_comp {α : Type u} [Storage α] {r s t : Shape}
    (f : Coord s → Coord r)
    (g : Coord t → Coord s) (x : Rep α r) :
    pull g (pull f x) = pull (f ∘ g) x := by
  ext i
  simp

/--
Two coordinate pullbacks are equal when their source coordinates have the
same row-major index at every output index.

The premise is stated over linear output indices so reflected compilers can
reduce composed coordinate programs to arithmetic without enumerating a
tensor.
-/
theorem pull_eq_of_linearIndex_eq {α : Type u} [Storage α] {s t : Shape}
    (leftMap rightMap : Coord t → Coord s)
    (hLinearIndex :
      ∀ outputIndex : Fin (Shape.size t),
        (Coord.linearize
          (leftMap (Coord.unlinearize outputIndex))).val =
        (Coord.linearize
          (rightMap (Coord.unlinearize outputIndex))).val)
    (inputTensor : Rep α s) :
    pull leftMap inputTensor = pull rightMap inputTensor := by
  ext outputCoordinate
  simp only [pull_apply]
  congr 1
  apply Coord.linearize_injective
  apply Fin.ext
  simpa only [Coord.unlinearize_linearize] using
    hLinearIndex (Coord.linearize outputCoordinate)

/-- Every pointwise scalar map commutes with a coordinate pullback. -/
@[grind =] theorem map_pull
    {α : Type u} [Storage α]
    {β : Type v} [Storage β] {s t : Shape}
    (f : α → β) (coordinateMap : Coord t → Coord s)
    (x : Rep α s) :
    map f (pull coordinateMap x) = pull coordinateMap (map f x) := by
  ext outputCoordinate
  simp only [map_apply, pull_apply]

/--
Every pointwise binary operation commutes with a shared coordinate pullback.
-/
@[grind =] theorem zipWith_pull
    {α : Type u} [Storage α]
    {β : Type v} [Storage β]
    {γ : Type w} [Storage γ]
    {s t : Shape} (f : α → β → γ) (coordinateMap : Coord t → Coord s)
    (x : Rep α s) (y : Rep β s) :
    zipWith f (pull coordinateMap x) (pull coordinateMap y) =
      pull coordinateMap (zipWith f x y) := by
  ext outputCoordinate
  simp only [zipWith_apply, pull_apply]

/-- Reindex a tensor along an equivalence of coordinate spaces. -/
def reindex {α : Type u} [Storage α] {s t : Shape} (e : Coord t ≃ Coord s) (x : Rep α s) :
    Rep α t :=
  pull e x

/-- Evaluating a reindexed tensor applies the coordinate equivalence first. -/
@[simp, grind =] theorem reindex_apply {α : Type u} [Storage α] {s t : Shape}
    (e : Coord t ≃ Coord s)
    (x : Rep α s) (i : Coord t) : reindex e x i = x (e i) := by
  simp [reindex]

/-- Reindexing by the identity coordinate equivalence leaves a tensor unchanged. -/
@[simp, grind =, einops =, einops_simps] theorem reindex_refl {α : Type u} [Storage α] {s : Shape}
    (x : Rep α s) :
    reindex (Equiv.refl (Coord s)) x = x := by
  simpa only [reindex, Equiv.coe_refl] using pull_id x

/--
Reindexing by `first` and then by `second` is one reindexing by `second.trans first`.

The coordinate maps compose in the opposite order from the tensor operations, because the second
equivalence acts on the tensor that the first one produced.
-/
@[einops_simps] theorem reindex_trans {α : Type u} [Storage α] {r s t : Shape}
    (first : Coord s ≃ Coord r) (second : Coord t ≃ Coord s)
    (x : Rep α r) :
    reindex second (reindex first x) = reindex (second.trans first) x := by
  simpa only [reindex, Equiv.coe_trans] using pull_comp first second x

/-- Reindexing by an equivalence and its inverse recovers the original tensor. -/
@[simp, grind =, einops =, einops_simps]
theorem reindex_symm_reindex {α : Type u} [Storage α] {s t : Shape}
    (e : Coord t ≃ Coord s) (x : Rep α s) :
    reindex e.symm (reindex e x) = x := by
  ext i
  simp [reindex]

/-- Reindexing is injective because inverse reindexing recovers its input. -/
@[grind inj] theorem reindex_injective {α : Type u} [Storage α] {s t : Shape}
    (e : Coord t ≃ Coord s) :
    Function.Injective (reindex e : Rep α s → Rep α t) := by
  intro x y hxy
  have hInverse := congrArg (reindex e.symm) hxy
  simpa only [reindex_symm_reindex] using hInverse

/-- Pointwise scalar maps commute with coordinate reindexing. -/
@[grind =, einops =, einops_simps] theorem map_reindex
    {α : Type u} [Storage α]
    {β : Type v} [Storage β] {s t : Shape}
    (f : α → β) (e : Coord t ≃ Coord s) (x : Rep α s) :
    map f (reindex e x) = reindex e (map f x) := by
  exact map_pull f e x

/-- Pointwise binary operators commute with a shared coordinate reindexing. -/
@[grind =, einops =, einops_simps] theorem zipWith_reindex
    {α : Type u} [Storage α]
    {β : Type v} [Storage β]
    {γ : Type w} [Storage γ]
    {s t : Shape} (f : α → β → γ) (e : Coord t ≃ Coord s)
    (x : Rep α s) (y : Rep β s) :
    zipWith f (reindex e x) (reindex e y) =
      reindex e (zipWith f x y) := by
  exact zipWith_pull f e x y

/--
The coordinate equivalence underlying a reshape between equally sized
shapes.
-/
def _root_.TorchLean.Shape.Coord.reshapeEquiv {s t : Shape} (h : Shape.size s = Shape.size t) :
    Coord t ≃ Coord s :=
  (Coord.equivFin t).trans <|
    (finCongr h.symm).trans (Coord.equivFin s).symm

/-- Reversing a reshape size equality reverses its coordinate equivalence. -/
theorem _root_.TorchLean.Shape.Coord.reshapeEquiv_symm {s t : Shape}
    (h : Shape.size s = Shape.size t) :
    Coord.reshapeEquiv h.symm = (Coord.reshapeEquiv h).symm := by
  ext coordinate
  rfl

/-- A reshape preserves the value of the row-major linear index. -/
theorem _root_.TorchLean.Shape.Coord.linearize_reshapeEquiv_val {s t : Shape}
    (h : Shape.size s = Shape.size t) (coordinate : Coord t) :
    (Coord.linearize (Coord.reshapeEquiv h coordinate)).val = (Coord.linearize coordinate).val := by
  change
    ((Coord.equivFin s)
      ((Coord.equivFin s).symm (finCongr h.symm ((Coord.equivFin t) coordinate)))).val =
      ((Coord.equivFin t) coordinate).val
  rw [Equiv.apply_symm_apply]
  rfl

/--
Reshape a tensor while preserving its row-major scalar sequence.

The equality argument is proof data establishing that the source and target
shapes have the same number of entries.
-/
def reshape {α : Type u} [Storage α] {s t : Shape} (h : Shape.size s = Shape.size t)
    (x : Rep α s) : Rep α t where
  buffer := x.buffer
  size_eq := x.size_eq.trans h

/-- Reshape preserves the row-major flat index of every output coordinate. -/
@[simp] theorem reshape_apply {α : Type u} [Storage α] {s t : Shape}
    (h : Shape.size s = Shape.size t) (x : Rep α s) (i : Coord t) :
    reshape h x i =
      x (Coord.unlinearize (finCongr h.symm (Coord.linearize i))) := by
  simp [reshape, get, getFlat]

/--
Observing a zero-copy reshape applies the canonical row-major coordinate
equivalence.

This is the proof-facing form of `reshape_apply`: execution reuses the source
array, while extensional arguments may reason about an ordinary coordinate
reindexing.
-/
@[grind =] theorem reshape_apply_reshapeEquiv {α : Type u} [Storage α] {s t : Shape}
    (h : Shape.size s = Shape.size t) (x : Rep α s) (i : Coord t) :
    reshape h x i = x (Coord.reshapeEquiv h i) := by
  simp [Coord.reshapeEquiv, Coord.linearize, Coord.unlinearize]

/--
A zero-copy reshape is extensionally the corresponding coordinate
reindexing.

The left side is the native implementation and allocates no array. The right
side is used only as a mathematical description in compiler-correctness
proofs.
-/
@[grind =, einops =] theorem reshape_eq_reindex {α : Type u} [Storage α] {s t : Shape}
    (h : Shape.size s = Shape.size t) (x : Rep α s) :
    reshape h x = reindex (Coord.reshapeEquiv h) x := by
  ext coordinate
  simp only [reshape_apply_reshapeEquiv, reindex_apply]

/-- Pointwise scalar maps commute with a zero-copy row-major reshape. -/
@[grind =, einops =, einops_simps] theorem map_reshape
    {α : Type u} [Storage α]
    {β : Type v} [Storage β] {s t : Shape}
    (f : α → β) (h : Shape.size s = Shape.size t) (x : Rep α s) :
    map f (reshape h x) = reshape h (map f x) := by
  ext coordinate
  simp only [map_apply, reshape_apply_reshapeEquiv]

/--
Pointwise binary operations commute with reshaping both inputs by the same
row-major size equality.
-/
@[grind =, einops =, einops_simps]
theorem zipWith_reshape {α : Type u} [Storage α] {β : Type v} [Storage β]
    {γ : Type w} [Storage γ] {s t : Shape} (f : α → β → γ)
    (h : Shape.size s = Shape.size t) (x : Rep α s)
    (y : Rep β s) :
    zipWith f (reshape h x) (reshape h y) =
      reshape h (zipWith f x y) := by
  ext coordinate
  simp only [zipWith_apply, reshape_apply_reshapeEquiv]

/-- Flat reads of a reshape are the source's flat reads at the cast index. -/
theorem getFlat_reshape {α : Type u} [Storage α] {s t : Shape}
    (h : Shape.size s = Shape.size t) (x : Rep α s) :
    (reshape h x).getFlat = fun i => x.getFlat (finCongr h.symm i) := by
  rfl

/-- Reshaping to the same shape is extensionally the identity. -/
@[simp, grind =, einops =, einops_simps] theorem reshape_rfl {α : Type u} [Storage α] {s : Shape}
    (x : Rep α s) :
    reshape (s := s) (t := s) rfl x = x :=
  rfl

/-- Successive zero-copy reshapes are one reshape along the composed size equality. -/
@[simp, grind =, einops =, einops_simps]
theorem reshape_reshape {α : Type u} [Storage α] {r s t : Shape}
    (h₁ : Shape.size r = Shape.size s) (h₂ : Shape.size s = Shape.size t) (x : Rep α r) :
    reshape h₂ (reshape h₁ x) = reshape (h₁.trans h₂) x :=
  rfl

/-- Reshaping into a leading singleton axis stacks the tensor along that axis. -/
theorem reshape_one_cons {α : Type u} [Storage α] {s : Shape}
    (h : Shape.size s = Shape.size (1 :: s)) (x : Rep α s) :
    reshape h x = dim fun _ => x := by
  ext ⟨i, j⟩
  simp only [reshape_apply, dim_apply]
  have hi : i.val = 0 := Nat.lt_one_iff.mp i.isLt
  have hIndex : finCongr h.symm (Coord.linearize (i, j)) = Coord.linearize j := by
    apply Fin.ext
    rw [finCongr_apply, Fin.val_cast, Coord.linearize_cons_val, hi, Nat.mul_zero, Nat.add_zero]
  rw [hIndex, Coord.unlinearize_linearize]

/-! ## The finite tensor pairing -/

/-- The finite sum of corresponding entry products. -/
def dot {α : Type u} [Storage α] [AddCommMonoid α] [Mul α] {s : Shape}
    (x y : Rep α s) : α :=
  ∑ i, x i * y i

/--
Stacking and leading-axis slicing are adjoint for the finite tensor pairing.

The pairing of a stacked family with a tensor is the sum of the pairings with
each corresponding leading-axis slice.
-/
@[grind =, einops =] theorem dot_stack {α : Type u} [Storage α] [AddCommMonoid α] [Mul α]
    {n : Nat} {s : Shape}
    (components : Fin n → Rep α s) (tensor : Rep α (n :: s)) :
    dot (dim components) tensor =
      ∑ component, dot (components component) (unstack tensor component) := by
  classical
  unfold dot
  simp only [dim_apply, unstack_apply]
  exact Fintype.sum_prod_type _

/--
Reindexing and inverse reindexing are adjoint for the finite tensor pairing. The proof moves the
summation index along the coordinate equivalence, with no ordering or positivity assumption on the
shape dimensions.
-/
@[grind =, einops_simps] theorem dot_reindex_left {α : Type u}
    [Storage α] [AddCommMonoid α] [Mul α]
    {s t : Shape} (e : Coord t ≃ Coord s)
    (x : Rep α s) (y : Rep α t) :
    dot (reindex e x) y = dot x (reindex e.symm y) := by
  simpa only [dot, reindex_apply, Equiv.symm_apply_apply] using
    e.sum_comp (fun inputCoordinate => x inputCoordinate * y (e.symm inputCoordinate))

/-- Applying the same coordinate reindexing to both tensors preserves their pairing. -/
@[grind =, einops =, einops_simps] theorem dot_reindex_reindex {α : Type u} [Storage α]
    [AddCommMonoid α] [Mul α] {s t : Shape}
    (e : Coord t ≃ Coord s) (x y : Rep α s) :
    dot (reindex e x) (reindex e y) = dot x y := by
  rw [dot_reindex_left, reindex_symm_reindex]

end

end TorchLean.Tensor.Rep
