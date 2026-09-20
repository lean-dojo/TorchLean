/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Tensor.Internal.Representation.Basic.Pointwise

/-!
# Tensor Pullbacks, Reindexing, and Reshape

Coordinate and flat-index pullbacks, broadcast, equivalence-based reindexing,
row-major flattening, and zero-copy reshape.
-/

@[expose] public section

namespace TorchLean.Tensor.Internal

universe u v w

namespace Rep

/--
Pull a tensor back along a coordinate map.

This is the basic semantics of rearrangement and coordinate replication.
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

Source and target shapes have the same rank. In each dimension the source
length must either equal the target length or be one; singleton dimensions
are replicated without imposing any operation on the scalar type. Leading
rank expansion can be represented explicitly by reshaping in singleton
dimensions first, which is the normalization used by repeat lowering.
-/
def broadcast {α : Type u} [Storage α] {sourceShape targetShape : Shape}
    (hShape :
      List.Forall₂
        (fun sourceLength targetLength =>
          sourceLength = targetLength ∨ sourceLength = 1)
        sourceShape targetShape)
    (sourceTensor : Rep α sourceShape) : Rep α targetShape :=
  pull (Coord.broadcast sourceShape targetShape hShape) sourceTensor

/-- Broadcasting between empty shapes is the identity. -/
theorem broadcast_nil {α : Type u} [Storage α]
    (h : List.Forall₂ (fun a b => a = b ∨ a = 1)
      ([] : Shape) [])
    (x : Rep α []) :
    broadcast h x = x := by
  ext c
  cases c
  rw [broadcast, pull_apply]

/-- Broadcasting along equal leading extents acts slice by slice. -/
theorem broadcast_cons_eq {α : Type u} [Storage α] {n : Nat} {s t : Shape}
    (h : List.Forall₂ (fun a b => a = b ∨ a = 1)
      (n :: s) (n :: t))
    (x : Rep α (n :: s)) :
    broadcast h x = stack fun i => broadcast (List.forall₂_cons.mp h).2 (unstack x i) := by
  ext ⟨i, j⟩
  simp [broadcast, Coord.broadcast]

/-- Broadcasting a leading singleton extent replicates the single slice. -/
theorem broadcast_cons_one {α : Type u} [Storage α] {n : Nat} {s t : Shape}
    (h : List.Forall₂ (fun a b => a = b ∨ a = 1)
      (1 :: s) (n :: t))
    (x : Rep α (1 :: s)) :
    broadcast h x = stack fun _ => broadcast (List.forall₂_cons.mp h).2 (unstack x 0) := by
  ext ⟨i, j⟩
  simp only [broadcast, pull_apply, stack_apply, unstack_apply, Coord.broadcast]
  split
  · congr 1
    exact Prod.ext (Subsingleton.elim _ _) rfl
  · congr 1

/-- Pulling a tensor along the identity coordinate map changes nothing. -/
@[simp, grind =] theorem pull_id {α : Type u} [Storage α] {s : Shape} (x : Rep α s) :
    pull id x = x := by
  ext i
  simp

/--
Pullbacks compose in the reverse order of their coordinate maps.

This associativity-style law is intentionally not a global `grind` rule:
repeated e-matching can synthesize arbitrarily nested function compositions.
-/
theorem pull_comp {α : Type u} [Storage α] {r s t : Shape}
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
@[simp, grind =] theorem reindex_refl {α : Type u} [Storage α] {s : Shape}
    (x : Rep α s) :
    reindex (Equiv.refl (Coord s)) x = x := by
  simpa only [reindex, Equiv.coe_refl] using pull_id x

/--
Successive coordinate equivalences compose to one reindexing operation.

The first equivalence is applied to the tensor, while the second is applied
to the resulting tensor, so their coordinate maps compose in the opposite
order from the tensor operations.
-/
theorem reindex_trans {α : Type u} [Storage α] {r s t : Shape}
    (first : Coord s ≃ Coord r) (second : Coord t ≃ Coord s)
    (x : Rep α r) :
    reindex second (reindex first x) = reindex (second.trans first) x := by
  simpa only [reindex, Equiv.coe_trans] using pull_comp first second x

/-- Reindexing by an equivalence and its inverse recovers the original tensor. -/
@[simp, grind =] theorem reindex_symm_reindex {α : Type u} [Storage α] {s t : Shape}
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
@[grind =] theorem map_reindex
    {α : Type u} [Storage α]
    {β : Type v} [Storage β] {s t : Shape}
    (f : α → β) (e : Coord t ≃ Coord s) (x : Rep α s) :
    map f (reindex e x) = reindex e (map f x) := by
  ext outputCoordinate
  simp only [map_apply, reindex_apply]

/-- Pointwise binary operators commute with a shared coordinate reindexing. -/
@[grind =] theorem zipWith_reindex
    {α : Type u} [Storage α]
    {β : Type v} [Storage β]
    {γ : Type w} [Storage γ]
    {s t : Shape} (f : α → β → γ) (e : Coord t ≃ Coord s)
    (x : Rep α s) (y : Rep β s) :
    zipWith f (reindex e x) (reindex e y) =
      reindex e (zipWith f x y) := by
  ext outputCoordinate
  simp only [zipWith_apply, reindex_apply]

/-- Flatten a tensor in row-major coordinate order. -/
def flatten {α : Type u} [Storage α] {s : Shape} (x : Rep α s) :
    Fin (Shape.size s) → α :=
  x.getFlat

/-- Build a tensor from its row-major scalar sequence. -/
def unflatten {α : Type u} [Storage α] {s : Shape} (x : Fin (Shape.size s) → α) :
    Rep α s :=
  ofFlatFn x

/-- Flattening reads the tensor at the coordinate represented by a flat index. -/
@[simp, grind =] theorem flatten_apply {α : Type u} [Storage α] {s : Shape}
    (x : Rep α s)
    (i : Fin (Shape.size s)) : flatten x i = x (Coord.unlinearize i) := by
  simp [flatten, get, Coord.linearize_unlinearize]

/-- Unflattening reads the flat function at a coordinate's row-major index. -/
@[simp, grind =] theorem unflatten_apply {α : Type u} [Storage α] {s : Shape}
    (x : Fin (Shape.size s) → α) (i : Coord s) :
    unflatten x i = x (Coord.linearize i) := by
  change getFlat (ofFlatFn x) (Coord.linearize i) = x (Coord.linearize i)
  exact getFlat_ofFlatFn x (Coord.linearize i)

/-- Flattening an unflattened row-major function recovers that function. -/
@[simp, grind =] theorem flatten_unflatten {α : Type u} [Storage α] {s : Shape}
    (x : Fin (Shape.size s) → α) : flatten (unflatten x) = x := by
  funext i
  exact getFlat_ofFlatFn x i

/-- Unflattening a flattened tensor recovers the original tensor. -/
@[simp, grind =] theorem unflatten_flatten {α : Type u} [Storage α] {s : Shape}
    (x : Rep α s) :
    unflatten (flatten x) = x := by
  ext i
  change getFlat (ofFlatFn x.getFlat) (Coord.linearize i) =
    getFlat x (Coord.linearize i)
  exact getFlat_ofFlatFn x.getFlat (Coord.linearize i)

/-- Coordinate tensors are equivalent to their row-major flat functions. -/
def flatEquiv (α : Type u) [Storage α] (s : Shape) :
    Rep α s ≃ (Fin (Shape.size s) → α) where
  toFun := flatten
  invFun := unflatten
  left_inv := unflatten_flatten
  right_inv := flatten_unflatten

/--
The coordinate equivalence underlying a reshape between equally sized
shapes.
-/
def reshapeCoordEquiv {s t : Shape} (h : Shape.size s = Shape.size t) :
    Coord t ≃ Coord s :=
  (Coord.equivFin t).trans <|
    (finCongr h.symm).trans (Coord.equivFin s).symm

/-- Reversing a reshape size equality reverses its coordinate equivalence. -/
theorem reshapeCoordEquiv_symm {s t : Shape}
    (h : Shape.size s = Shape.size t) :
    reshapeCoordEquiv h.symm = (reshapeCoordEquiv h).symm := by
  ext coordinate
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
@[grind =] theorem reshape_apply_coordEquiv {α : Type u} [Storage α] {s t : Shape}
    (h : Shape.size s = Shape.size t) (x : Rep α s) (i : Coord t) :
    reshape h x i = x (reshapeCoordEquiv h i) := by
  simp [reshapeCoordEquiv, Coord.linearize, Coord.unlinearize]

/--
A zero-copy reshape is extensionally the corresponding coordinate
reindexing.

The left side is the native implementation and allocates no array. The right
side is used only as a mathematical description in compiler-correctness
proofs.
-/
@[grind =] theorem reshape_eq_reindex {α : Type u} [Storage α] {s t : Shape}
    (h : Shape.size s = Shape.size t) (x : Rep α s) :
    reshape h x = reindex (reshapeCoordEquiv h) x := by
  ext coordinate
  simp only [reshape_apply_coordEquiv, reindex_apply]

/-- Pointwise scalar maps commute with a zero-copy row-major reshape. -/
@[grind =] theorem map_reshape
    {α : Type u} [Storage α]
    {β : Type v} [Storage β] {s t : Shape}
    (f : α → β) (h : Shape.size s = Shape.size t) (x : Rep α s) :
    map f (reshape h x) = reshape h (map f x) := by
  ext coordinate
  simp only [map_apply, reshape_apply_coordEquiv]

/--
Pointwise binary operations commute with reshaping both inputs by the same
row-major size equality.
-/
@[grind =] theorem zipWith_reshape {α : Type u} [Storage α] {β : Type v} [Storage β]
    {γ : Type w} [Storage γ] {s t : Shape} (f : α → β → γ)
    (h : Shape.size s = Shape.size t) (x : Rep α s)
    (y : Rep β s) :
    zipWith f (reshape h x) (reshape h y) =
      reshape h (zipWith f x y) := by
  ext coordinate
  simp only [zipWith_apply, reshape_apply_coordEquiv]

/-- Flattening a reshape changes only the finite index type. -/
theorem flatten_reshape {α : Type u} [Storage α] {s t : Shape}
    (h : Shape.size s = Shape.size t) (x : Rep α s) :
    flatten (reshape h x) = fun i => flatten x (finCongr h.symm i) := by
  funext i
  simp only [flatten_apply, reshape_apply, Coord.linearize_unlinearize]

/-- Reshaping to the same shape is extensionally the identity. -/
@[simp, grind =] theorem reshape_rfl {α : Type u} [Storage α] {s : Shape}
    (x : Rep α s) :
    reshape (s := s) (t := s) rfl x = x := by
  ext i
  simp only [reshape_apply]
  change x (Coord.unlinearize (Coord.linearize i)) = x i
  rw [Coord.unlinearize_linearize]

/-- Successive zero-copy reshapes are one reshape along the composed size equality. -/
@[simp] theorem reshape_reshape {α : Type u} [Storage α] {r s t : Shape}
    (h₁ : Shape.size r = Shape.size s) (h₂ : Shape.size s = Shape.size t) (x : Rep α r) :
    reshape h₂ (reshape h₁ x) = reshape (h₁.trans h₂) x :=
  rfl

/-- Reshaping into a leading singleton axis stacks the tensor along that axis. -/
theorem reshape_one_cons {α : Type u} [Storage α] {s : Shape}
    (h : Shape.size s = Shape.size (1 :: s)) (x : Rep α s) :
    reshape h x = stack fun _ => x := by
  ext ⟨i, j⟩
  simp only [reshape_apply, stack_apply]
  have hi : i.val = 0 := Nat.lt_one_iff.mp i.isLt
  have hIndex : finCongr h.symm (Coord.linearize (i, j)) = Coord.linearize j := by
    apply Fin.ext
    rw [finCongr_apply, Fin.val_cast, Coord.linearize_cons_val, hi, Nat.mul_zero, Nat.add_zero]
  rw [hIndex, Coord.unlinearize_linearize]

/-- Reshaping back along the symmetric size equality recovers the input. -/
@[simp, grind =] theorem reshape_symm_reshape {α : Type u} [Storage α] {s t : Shape}
    (h : Shape.size s = Shape.size t) (x : Rep α s) :
    reshape h.symm (reshape h x) = x := by
  apply (flatEquiv α s).injective
  change flatten (reshape h.symm (reshape h x)) = flatten x
  rw [flatten_reshape]
  funext i
  rw [congrFun (flatten_reshape h x) (finCongr h i)]
  have hcast : finCongr h.symm (finCongr h i) = i := by
    exact (finCongr h).symm_apply_apply i
  rw [hcast]

end Rep

end TorchLean.Tensor.Internal
