/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.LastAxis
public import TorchLean.Tensor.AdjacentSwap
public import TorchLean.Tensor.Sequence
public import Mathlib.GroupTheory.Perm.Fin

@[expose] public section

open TorchLean.Shape
open Coord (linearize unlinearize)
open TorchLean.Shape (rank rank_cons)
open TorchLean.Shape (size)
open TorchLean.Shape (rank_eq_length)

/-!
# Fibers along an axis and slices at a batch coordinate

The fiber of a tensor along an axis through a coordinate holds the entries whose coordinates
differ from it only at that axis. `Tensor.alongAxis` moves an axis last with one `Tensor.permute`
by the cycle `Fin.cycleIcc dim (rank - 1)` (`TorchLean.Shape.permute_cycleIcc` gives the moved
shape), applies a map acting on last-axis rows and moves the axis back, as softmax along a dimension
does. It is read at a coordinate as that map on the fibers through it (`Tensor.alongAxis_apply`, and
`Tensor.alongAxis₂_apply` for maps of two tensors). The proof compares coordinates through their
indices.

The slice of a tensor of shape `batch ++ s` at a coordinate of `batch` is a tensor of shape
`s`. A map defined by recursion on a batch prefix that acts on the outer slices separately, as the
batched attention products do, is read at a coordinate as the map without batch on the slices at
its batch coordinate (`Tensor.batchwise_apply`). `Tensor.mapLeading_apply` and
`Tensor.zipEach_apply` are two instances, and `Tensor.mapLeading_eq_reshape` flattens a leading
shape into one batch axis.
-/

namespace TorchLean.Shape

/-- The cycle `Fin.cycleIcc dim (s.rank - 1)` permutes the axes of `s` into the other axes in
order followed by `dim`. -/
theorem permute_cycleIcc (s : Shape) (dim : Fin s.rank) :
    s.permute (Fin.cycleIcc dim ⟨s.rank - 1, Nat.sub_one_lt_of_lt dim.isLt⟩) =
      s.eraseIdx dim ++ [s.extent dim] := by
  have : NeZero s.rank := NeZero.of_pos (Nat.zero_lt_of_lt dim.isLt)
  have hd := dim.isLt
  have hl : s.rank = s.length := Shape.rank_eq_length s
  have he : Shape.rank (s.eraseIdx dim) = s.rank - 1 := rank_eraseIdx dim.isLt
  have hrank : (s.eraseIdx dim).length = s.length - 1 := by
    rw [← Shape.rank_eq_length, ← Shape.rank_eq_length, rank_eraseIdx dim.isLt]
  have hera (i j : Nat) (hi : i < s.length - 1) (hj : j = if i < dim.val then i else i + 1)
      (hjs : j < s.rank) : (s.eraseIdx dim)[i]? = some (s.extent ⟨j, hjs⟩) := by
    subst hj
    rw [List.getElem?_eq_getElem (by omega)]
    exact congrArg some (extent_eraseIdx ⟨i, by omega⟩)
  apply List.ext_getElem?
  intro k
  rw [permute, List.getElem?_ofFn, List.getElem?_append, hrank]
  rcases Nat.lt_or_ge k s.rank with hk | hk
  · rw [dite_eq_left hk]
    rcases Nat.lt_or_ge k dim.val with hkd | hkd
    · rw [ite_eq_left (by omega : k < s.length - 1),
        Fin.cycleIcc_of_lt (show (⟨k, hk⟩ : Fin s.rank) < dim from Fin.lt_def.mpr hkd),
        hera k k (by omega) (ite_eq_left hkd).symm hk]
    · rcases Nat.lt_or_ge k (s.rank - 1) with hkl | hkl
      · have hsucc : (⟨k, hk⟩ + 1 : Fin s.rank) = ⟨k + 1, by omega⟩ := by
          apply Fin.ext
          rw [Fin.val_add, Fin.val_one', Nat.add_mod_mod,
            Nat.mod_eq_of_lt (show k + 1 < s.rank by omega)]
        rw [ite_eq_left (by omega : k < s.length - 1),
          Fin.cycleIcc_of_ge_of_lt (show dim ≤ (⟨k, hk⟩ : Fin s.rank) from Fin.le_def.mpr hkd)
            (show (⟨k, hk⟩ : Fin s.rank) < ⟨s.rank - 1, Nat.sub_one_lt_of_lt dim.isLt⟩ from
              Fin.lt_def.mpr hkl),
          hsucc, hera k (k + 1) (by omega) (ite_eq_right (by omega : ¬k < dim.val)).symm (by omega)]
      · have hlast : (⟨k, hk⟩ : Fin s.rank) = ⟨s.rank - 1, Nat.sub_one_lt_of_lt dim.isLt⟩ :=
          Fin.ext (show k = s.rank - 1 by omega)
        rw [ite_eq_right (by omega : ¬k < s.length - 1), hlast,
          Fin.cycleIcc_of_last (show dim ≤ ⟨s.rank - 1, Nat.sub_one_lt_of_lt dim.isLt⟩ from
            Fin.le_def.mpr (show dim.val ≤ s.rank - 1 by omega)),
          show k - (s.length - 1) = 0 by omega, List.getElem?_cons_zero]
  · rw [dite_eq_right (by omega : ¬k < s.rank), ite_eq_right (by omega : ¬k < s.length - 1)]
    exact (List.getElem?_eq_none (by simp only [List.length_singleton]; omega)).symm

namespace Coord

/-- Moving along a row replaces the index along the last axis and keeps the others. -/
theorem index_updateLast : {s : Shape} → (c : s.Coord) → (j : Fin s.rowWidth) → (k : Nat) →
    index (updateLast c j) k = if k + 1 = s.rank then j.val else index c k
  | .scalar, _, _, _ => by simp
  | .dim _ .scalar, _, _, 0 => by simp [updateLast]
  | .dim _ .scalar, _, _, _ + 1 => by simp [updateLast, index_succ]
  | .dim _ (.dim _ _), _, _, 0 => by simp [updateLast]
  | .dim _ (.dim n s), c, j, k + 1 => by
      show index (updateLast (s := .dim n s) c.2 j) k = _
      rw [index_updateLast c.2 j k]
      simp [index_succ]

/-- The last index of a coordinate is its index along the last axis. -/
theorem val_getLast {s : Shape} (c : s.Coord) :
    (getLast c).val = index c (s.rank - 1) := by
  cases s with
  | scalar => rfl
  | dim n s =>
      have h := index_updateLast c (getLast c) (Shape.rank (Shape.dim n s) - 1)
      rw [updateLast_getLast, ite_eq_left (by simp only [Shape.rank_cons]; omega)] at h
      exact h.symm

/-- Moving along a row replaces the last index. -/
theorem toList_updateLast {s : Shape} (hs : s ≠ .scalar) (c : s.Coord) (j : Fin s.rowWidth) :
    Coord.toList s (updateLast c j) = (Coord.toList s c).dropLast ++ [j.val] := by
  induction s with
  | scalar => exact absurd rfl hs
  | dim n s ih =>
      cases s with
      | scalar => rfl
      | dim m s =>
          change c.1.val :: Coord.toList _ (updateLast c.2 j) = _
          rw [ih (List.cons_ne_nil _ _) c.2 j]
          rfl

/-- Reinserting an index along `axis` inserts it into the indices of the other axes. -/
theorem toList_axisEquiv_symm (axis : Nat) {s : Shape} [AxisInBounds axis s]
    (j : Fin (s.axisSize axis)) (c : Shape.Coord (s.eraseIdx axis)) :
    Coord.toList s ((axisEquiv axis (shape := s)).symm (j, c)) =
      (Coord.toList _ c).insertIdx axis j.val := by
  induction s generalizing axis with
  | scalar => exact absurd (inferInstance : AxisInBounds axis .scalar).proof (Nat.not_lt_zero axis)
  | dim n s ih =>
      cases axis with
      | zero => rfl
      | succ axis =>
          have : AxisInBounds axis s :=
            ⟨Nat.lt_of_succ_lt_succ (inferInstance : AxisInBounds (axis + 1) (.dim n s)).proof⟩
          exact congrArg (c.1.val :: ·) (ih axis j c.2)

/-- The coordinates of the other axes keep every index but the one at `axis`. -/
theorem toList_axisEquiv_snd (axis : Nat) {s : Shape} [AxisInBounds axis s]
    (c : s.Coord) :
    Coord.toList _ (axisEquiv axis (shape := s) c).2 =
      (Coord.toList s c).eraseIdx axis := by
  conv_rhs => rw [← (axisEquiv axis (shape := s)).symm_apply_apply c]
  rw [toList_axisEquiv_symm, List.eraseIdx_insertIdx_self]

/-- The index `Coord.axisEquiv` separates is the index at `axis`. -/
theorem val_axisEquiv_fst_eq_getElem (axis : Nat) {s : Shape} [AxisInBounds axis s] (c : s.Coord)
    (h : axis < (Coord.toList s c).length) :
    (axisEquiv axis (shape := s) c).1.val = (Coord.toList s c)[axis] := by
  have e := toList_axisEquiv_symm axis
    (axisEquiv axis (shape := s) c).1
    (axisEquiv axis (shape := s) c).2
  rw [Prod.mk.eta, Equiv.symm_apply_apply] at e
  exact ((List.getElem_of_eq e h).trans (List.getElem_insertIdx_self _)).symm

end Coord

end TorchLean.Shape

namespace TorchLean.Tensor

variable {α : Type} [Storage α]

/-- The fiber of `tensor` along `dim` through `c`: its entries at the coordinates that differ
from `c` only at `dim`, in index order. -/
def fiber {s : Shape} (tensor : Tensor α s) (dim : Fin s.rank)
    (c : s.Coord) : Tensor α [s.axisSize dim] :=
  ofFn fun j => tensor ((Coord.axisEquiv dim (shape := s)).symm
    (j, (Coord.axisEquiv dim (shape := s) c).2))

/-- Reading a fiber reads the tensor with the index at `dim` replaced. -/
theorem fiber_apply {s : Shape} (tensor : Tensor α s) (dim : Fin s.rank)
    (c : s.Coord) (j : Shape.Coord [s.axisSize dim]) :
    tensor.fiber dim c j = tensor ((Coord.axisEquiv dim (shape := s)).symm
      (j.1, (Coord.axisEquiv dim (shape := s) c).2)) :=
  ofFn_apply _ j

/-- A shape-polymorphic map of two vectors reads equal entries from inputs that agree after a
cast of their length. -/
private theorem getScalar_cast {β : Type} [Storage β]
    (map : ∀ {s : Shape}, Tensor α s → Tensor β s → Tensor α s)
    {m n : Nat} (h : m = n) {first : Tensor α [m]} {second : Tensor β [m]} {first' : Tensor α [n]}
    {second' : Tensor β [n]}
    (hfirst : ∀ i, first.getScalar i = first'.getScalar (Fin.cast h i))
    (hsecond : ∀ i, second.getScalar i = second'.getScalar (Fin.cast h i)) (i : Fin m) :
    (map first second).getScalar i = (map first' second').getScalar (Fin.cast h i) := by
  subst h
  have ext {γ : Type} [Storage γ] (t t' : Tensor γ [m])
      (ht : ∀ i, t.getScalar i = t'.getScalar (Fin.cast rfl i)) : t = t' :=
    Rep.ext fun c => by simpa only [getScalar_eq_apply, Fin.cast_eq_self] using ht c.1
  rw [ext first first' hfirst, ext second second' hsecond, Fin.cast_eq_self]

/-- `kernel` applied along `dim`: one `permute` by `Fin.cycleIcc dim (s.rank - 1)` moves `dim`
last, `kernel` acts on the last-axis rows and a second `permute` moves the axis back. -/
def alongAxis {s : Shape} (dim : Fin s.rank) (kernel : ∀ {t : Shape}, Tensor α t → Tensor α t)
    (x : Tensor α s) : Tensor α s :=
  if dim.val + 1 = s.rank then kernel x else
    let σ := Fin.cycleIcc dim ⟨s.rank - 1, Nat.sub_one_lt_of_lt dim.isLt⟩
    permute (kernel (permute x σ (s.permute σ) rfl)) (s.unpermute σ rfl) s
      (s.permute_unpermute σ rfl).symm

/-- `kernel` of two tensors applied along `dim`, moving `dim` last in both as `alongAxis` does. -/
def alongAxis₂ {β : Type} [Storage β] {s : Shape} (dim : Fin s.rank)
    (kernel : ∀ {t : Shape}, Tensor α t → Tensor β t → Tensor α t) (x : Tensor α s)
    (y : Tensor β s) : Tensor α s :=
  if dim.val + 1 = s.rank then kernel x y else
    let σ := Fin.cycleIcc dim ⟨s.rank - 1, Nat.sub_one_lt_of_lt dim.isLt⟩
    permute (kernel (permute x σ (s.permute σ) rfl) (permute y σ (s.permute σ) rfl))
      (s.unpermute σ rfl) s (s.permute_unpermute σ rfl).symm

/-- Along the last axis, `alongAxis` applies the kernel to the tensor itself. -/
theorem alongAxis_of_last {s : Shape} (dim : Fin s.rank)
    (kernel : ∀ {t : Shape}, Tensor α t → Tensor α t) (x : Tensor α s)
    (h : dim.val + 1 = s.rank) : alongAxis dim kernel x = kernel x :=
  ite_eq_left h

/-- Along the last axis, `alongAxis₂` applies the kernel to the two tensors themselves. -/
theorem alongAxis₂_of_last {β : Type} [Storage β] {s : Shape} (dim : Fin s.rank)
    (kernel : ∀ {t : Shape}, Tensor α t → Tensor β t → Tensor α t) (x : Tensor α s)
    (y : Tensor β s) (h : dim.val + 1 = s.rank) : alongAxis₂ dim kernel x y = kernel x y :=
  ite_eq_left h

/-- A coordinate with index `j` along `dim` and the indices of `c` along the other axes is
rebuilt by `Coord.axisEquiv` from `j` and the other indices of `c`. -/
private theorem eq_axisCoordinateEquiv_symm {s : Shape} (dim : Fin s.rank) {p c : s.Coord}
    {j : Fin (s.axisSize dim)} (hdim : Shape.Coord.index p dim = j.val)
    (hother : ∀ k < s.rank, k ≠ dim.val → Shape.Coord.index p k = Shape.Coord.index c k) :
    p = (Coord.axisEquiv dim (shape := s)).symm
      (j, (Coord.axisEquiv dim (shape := s) c).2) := by
  rw [Equiv.eq_symm_apply]
  refine Prod.ext (Fin.ext ?_) (Shape.Coord.ext_index fun k hk => ?_)
  · rw [Shape.Coord.val_axisEquiv_fst_eq_getElem dim p
      (by rw [Shape.Coord.length_toList, ← Shape.rank_eq_length]; exact dim.isLt),
      Shape.Coord.getElem_toList, hdim]
  · have hd := dim.isLt
    have hk' : k < s.rank - 1 := Shape.rank_eraseIdx dim.isLt ▸ hk
    rw [Coord.index_axisEquiv_snd, Coord.index_axisEquiv_snd]
    split_ifs with h
    · exact hother k (by omega) (by omega)
    · exact hother (k + 1) (by omega) (by omega)

/-- Suppose `pull` sends the row through `q` to the fiber through `c` along `dim`. Then a map acting
on last-axis rows, read at `q` of the tensors whose entries are those of `first` and `second` at
`pull`, is its value on the fibers through `c`. -/
private theorem rowwise_fiber {β : Type} [Storage β]
    (map : ∀ {s : Shape}, Tensor α s → Tensor β s → Tensor α s)
    (outer : ∀ {m n : Nat} {s : Shape} (first : Tensor α (.dim (m + 1) (.dim n s)))
      (second : Tensor β (.dim (m + 1) (.dim n s))),
      map first second = Tensor.dim fun i => map (first.unstack i) (second.unstack i))
    {s t : Shape} (dim : Fin s.rank) (hne : t ≠ .scalar) (hw : t.rowWidth = s.axisSize dim)
    {first : Tensor α s} {second : Tensor β s} {first' : Tensor α t} {second' : Tensor β t}
    (pull : t.Coord → s.Coord) (hfirst : ∀ p, first' p = first (pull p))
    (hsecond : ∀ p, second' p = second (pull p)) (q : t.Coord) (c : s.Coord)
    (hdim : ∀ j, Shape.Coord.index (pull (Shape.Coord.updateLast q j)) dim = j.val)
    (hother : ∀ j, ∀ k < s.rank, k ≠ dim.val →
      Shape.Coord.index (pull (Shape.Coord.updateLast q j)) k = Shape.Coord.index c k)
    (hlast : (Shape.Coord.getLast q).val = Shape.Coord.index c dim) :
    map first' second' q = (map (first.fiber dim c) (second.fiber dim c)).getScalar
      (Coord.axisEquiv dim (shape := s) c).1 := by
  have hc : Fin.cast hw (Shape.Coord.getLast q) =
      (Coord.axisEquiv dim (shape := s) c).1 :=
    Fin.ext (by
      rw [Fin.val_cast, hlast, Shape.Coord.val_axisEquiv_fst_eq_getElem dim c
        (by rw [Shape.Coord.length_toList, ← Shape.rank_eq_length]; exact dim.isLt),
        Shape.Coord.getElem_toList])
  have hrow (i : Fin t.rowWidth) : pull (Shape.Coord.updateLast q i) =
      (Coord.axisEquiv dim (shape := s)).symm
        (Fin.cast hw i, (Coord.axisEquiv dim (shape := s) c).2) :=
    eq_axisCoordinateEquiv_symm dim (hdim i) (hother i)
  rw [rowwise_apply_of_ne map outer hne, ← hc]
  refine getScalar_cast map hw (fun i => ?_) (fun i => ?_) _
  · rw [getScalar_eq_apply, getScalar_eq_apply, row_apply, hfirst, hrow, fiber_apply]
  · rw [getScalar_eq_apply, getScalar_eq_apply, row_apply, hsecond, hrow, fiber_apply]

/-- A shape-polymorphic map acting on last-axis rows (`rowwise_apply_of_ne`), applied along `dim`
by `alongAxis₂`, is read at `c` as its value on the fibers through `c`. -/
theorem alongAxis₂_apply {β : Type} [Storage β]
    (map : ∀ {s : Shape}, Tensor α s → Tensor β s → Tensor α s)
    (outer : ∀ {m n : Nat} {s : Shape} (first : Tensor α (.dim (m + 1) (.dim n s)))
      (second : Tensor β (.dim (m + 1) (.dim n s))),
      map first second = Tensor.dim fun i => map (first.unstack i) (second.unstack i))
    {s : Shape} (dim : Fin s.rank) (first : Tensor α s) (second : Tensor β s)
    (c : s.Coord) :
    alongAxis₂ dim map first second c =
      (map (first.fiber dim c) (second.fiber dim c)).getScalar
        (Coord.axisEquiv dim (shape := s) c).1 := by
  have : NeZero s.rank := NeZero.of_pos (Nat.zero_lt_of_lt dim.isLt)
  have hd := dim.isLt
  have hle : dim ≤ ⟨s.rank - 1, Nat.sub_one_lt_of_lt dim.isLt⟩ :=
    Fin.le_iff_val_le_val.mpr (by simp only; omega)
  have hshape := s.permute_cycleIcc dim
  by_cases hlast : dim.val + 1 = s.rank
  · have hs : s = s.eraseIdx dim ++ [s.extent dim] := by
      rw [← hshape, show (⟨s.rank - 1, Nat.sub_one_lt_of_lt dim.isLt⟩ : Fin s.rank) = dim from
        Fin.ext (by simp only; omega), Fin.cycleIcc_eq]
      exact (Shape.permute_refl s).symm
    rw [alongAxis₂, ite_eq_left hlast]
    refine rowwise_fiber map outer dim (t := s) (fun h => ?_)
      ((congrArg Shape.rowWidth hs).trans
        (by rw [Shape.rowWidth_append_singleton, Shape.axisSize_eq_extent]))
      id (fun _ => rfl) (fun _ => rfl) c c (fun j => ?_) (fun j k hk hkd => ?_) ?_
    · subst h
      exact dim.elim0
    · rw [id, Shape.Coord.index_updateLast, ite_eq_left hlast]
    · rw [id, Shape.Coord.index_updateLast, ite_eq_right (by omega)]
    · rw [Shape.Coord.val_getLast]
      exact congrArg (Shape.Coord.index c) (by omega)
  · simp only [alongAxis₂, ite_eq_right hlast]
    rw [permute_apply]
    set σ := Fin.cycleIcc dim ⟨s.rank - 1, Nat.sub_one_lt_of_lt dim.isLt⟩ with hσdef
    have hσ : σ.symm dim = ⟨s.rank - 1, Nat.sub_one_lt_of_lt dim.isLt⟩ :=
      (Equiv.symm_apply_eq _).mpr (Fin.cycleIcc_of_last hle).symm
    have hne : s.permute σ ≠ .scalar := by
      rw [hshape]
      exact List.concat_ne_nil _ _
    have hw : (s.permute σ).rowWidth = s.axisSize dim := by
      rw [hshape, Shape.rowWidth_append_singleton, Shape.axisSize_eq_extent]
    refine rowwise_fiber map outer dim hne hw _ (fun p => permute_apply first σ _ rfl p)
      (fun p => permute_apply second σ _ rfl p) _ c (fun j => ?_) (fun j k hk hkd => ?_) ?_
    · simp only [Shape.Coord.index_ofFun, dite_eq_left dim.isLt, Fin.val_cast, Shape.Coord.val_get,
        Shape.Coord.index_reindexEq, Shape.Coord.index_updateLast, Fin.eta, hσ,
        Shape.rank_permute]
      rw [ite_eq_left (by omega)]
    · have hk' : σ.symm ⟨k, hk⟩ ≠ ⟨s.rank - 1, Nat.sub_one_lt_of_lt dim.isLt⟩ := by
        rw [← hσ]
        exact fun h => hkd (congrArg Fin.val (σ.symm.injective h))
      have hk'' : (σ.symm ⟨k, hk⟩).val + 1 ≠ s.rank := fun h => hk' (Fin.ext (by simp only; omega))
      simp only [Shape.Coord.index_ofFun, dite_eq_left hk, Fin.val_cast, Shape.Coord.val_get,
        Shape.Coord.index_reindexEq, Shape.Coord.index_updateLast, Shape.rank_permute,
        ite_eq_right hk'']
      rw [dite_eq_left (σ.symm ⟨k, hk⟩).isLt]
      exact congrArg (fun x : Fin s.rank => Shape.Coord.index c x.val) (σ.apply_symm_apply ⟨k, hk⟩)
    · rw [Shape.Coord.val_getLast, Shape.Coord.index_ofFun,
        dite_eq_left (by simp only [Shape.rank_permute]; omega)]
      simp only [Fin.val_cast, Shape.Coord.val_get, Shape.Coord.index_reindexEq]
      have hk : (⟨(s.permute σ).rank - 1, by simp only [Shape.rank_permute]; omega⟩ : Fin s.rank) =
          ⟨s.rank - 1, Nat.sub_one_lt_of_lt dim.isLt⟩ := Fin.ext (by simp only [Shape.rank_permute])
      exact congrArg (fun x : Fin s.rank => Shape.Coord.index c x.val)
        ((congrArg σ hk).trans (Fin.cycleIcc_of_last hle))

/-- A shape-polymorphic map acting on last-axis rows, applied along `dim` by `alongAxis`, is read
at `c` as its value on the fiber through `c`. -/
theorem alongAxis_apply (kernel : ∀ {s : Shape}, Tensor α s → Tensor α s)
    (outer : ∀ {m n : Nat} {s : Shape} (x : Tensor α (.dim (m + 1) (.dim n s))),
      kernel x = Tensor.dim fun i => kernel (x.unstack i))
    {s : Shape} (dim : Fin s.rank) (x : Tensor α s) (c : s.Coord) :
    alongAxis dim kernel x c =
      (kernel (x.fiber dim c)).getScalar (Coord.axisEquiv dim (shape := s) c).1 :=
  alongAxis₂_apply (fun x _ => kernel x) (fun first _ => outer first) dim x x c

/-- The slice of `tensor` at the batch coordinate `b`. -/
def batchSlice {batch s : Shape} (tensor : Tensor α (batch ++ s)) (b : batch.Coord) :
    Tensor α s :=
  Rep.ofFn fun q => tensor ((Coord.appendEquiv batch s).symm (b, q))

/-- Reading a batch slice reads the tensor at the batch coordinate followed by the coordinate. -/
theorem batchSlice_apply {batch s : Shape} (tensor : Tensor α (batch ++ s)) (b : batch.Coord)
    (q : s.Coord) :
    tensor.batchSlice b q = tensor ((Coord.appendEquiv batch s).symm (b, q)) :=
  Rep.get_ofFn _ q

/-- A cast of a coordinate along an equality of nonempty shapes with equal outer extent keeps the
outer index. -/
private theorem cast_dim {source target : Shape} {n : Nat} (h : source = target)
    (c : Shape.Coord (.dim n source)) :
    _root_.cast (congrArg Shape.Coord (congrArg (List.cons n) h)) c =
      (c.1, _root_.cast (congrArg Shape.Coord h) c.2) := by
  cases h
  rfl

/-- The slice at `(i, b)` is the slice at `b` of the outer slice `i`. -/
theorem batchSlice_mk {n : Nat} {batch s : Shape}
    (tensor : Tensor α ((Shape.dim n batch) ++ s)) (i : Fin n) (b : batch.Coord) :
    tensor.batchSlice (i, b) = (tensor.unstack i).batchSlice b :=
  Rep.ext fun q => by
    rw [batchSlice_apply, batchSlice_apply, Tensor.unstack_apply]
    rfl

/-- A map of tensors with a shared batch prefix that acts separately on the slices of the
outermost batch axis is read at a coordinate as the map without batch on the slices at its batch
coordinate. -/
theorem batchwise_apply {ι : Type} {shapes : ι → Shape} {t : Shape}
    (map : ∀ {batch : Shape}, ((i : ι) → Tensor α (batch ++ (shapes i))) →
      Tensor α (batch ++ t))
    (outer : ∀ {n : Nat} {batch : Shape}
      (inputs : (i : ι) → Tensor α ((Shape.dim n batch) ++ (shapes i))),
      map inputs = Tensor.dim fun j => map fun i => (inputs i).unstack j)
    {batch : Shape} (inputs : (i : ι) → Tensor α (batch ++ (shapes i)))
    (c : (batch ++ t).Coord) :
    map inputs c =
      let split := Coord.appendEquiv batch t
        c
      map (batch := .scalar) (fun i => (inputs i).batchSlice split.1) split.2 := by
  induction batch with
  | scalar =>
      have h : (fun i => (inputs i).batchSlice PUnit.unit) = inputs :=
        funext fun i => Rep.ext fun q => Rep.get_ofFn _ q
      exact congrArg (fun inputs => map (batch := .scalar) inputs c) h.symm
  | dim n batch ih =>
      rw [outer, Tensor.dim_apply, ih]
      exact congrArg (fun inputs => map (batch := .scalar) inputs _)
        (funext fun i => (batchSlice_mk (inputs i) c.1 _).symm)

/-- Reading `mapLeading` reads the map on the slice at the leading part of the coordinate. -/
theorem mapLeading_apply (leading : Shape) {inShape outShape : Shape}
    (f : Tensor α inShape → Tensor α outShape) (x : Tensor α (leading ++ inShape))
    (c : (leading ++ outShape).Coord) :
    mapLeading leading f x c =
      let split := Coord.appendEquiv leading outShape
        c
      f (x.batchSlice split.1) split.2 :=
  batchwise_apply (ι := Unit) (shapes := fun _ => inShape) (t := outShape)
    (fun {batch} inputs => mapLeading batch f (inputs ())) (fun _ => rfl) (fun _ => x) c

/-- Reading `zipEach` reads the map on the slices of both operands at the leading part of the
coordinate. -/
theorem zipEach_apply (leading : Shape) {leftShape rightShape : Shape} (outShape : Shape)
    (f : Tensor α leftShape → Tensor α rightShape → Tensor α outShape)
    (left : Tensor α (leading ++ leftShape)) (right : Tensor α (leading ++ rightShape))
    (c : (leading ++ outShape).Coord) :
    zipEach leading outShape f left right c =
      let split := Coord.appendEquiv leading outShape
        c
      f (left.batchSlice split.1) (right.batchSlice split.1) split.2 :=
  batchwise_apply (ι := Bool) (shapes := fun b => if b then leftShape else rightShape)
    (t := outShape) (fun {batch} inputs => zipEach batch outShape f (inputs true) (inputs false))
    (fun _ => rfl) (fun b => match b with | true => left | false => right) c

/-- Mapping over the leading shape `L` is mapping over one batch axis of `L.size` slices, between
the row-major views of the input and of the result, for any map `f`. -/
theorem mapLeading_eq_reshape (L : Shape) {s t : Shape} (f : Tensor α s → Tensor α t)
    (x : Tensor α (L ++ s)) (h₁ : (L ++ s).size = Shape.size (L.size :: s))
    (h₂ : Shape.size (L.size :: t) = (L ++ t).size) :
    mapLeading L f x = reshape (mapLeading [L.size] f (reshape x _ h₁)) _ h₂ := by
  apply Rep.ext
  intro c
  obtain ⟨⟨b, d⟩, hc⟩ := (Coord.appendEquiv L t).symm.surjective
    c
  have hc' : c = ((Coord.appendEquiv L t).symm (b, d)) := by rw [hc]
  subst hc'
  let b₀ : Shape.Coord [L.size] := Coord.unlinearize ⟨(Coord.linearize b).val,
    by
      rw [Shape.size_vector]
      exact (Coord.linearize b).isLt⟩
  have hb₀ : (Coord.linearize b₀).val = (Coord.linearize b).val := by
    show (Coord.linearize (Coord.unlinearize _)).val = _
    rw [Coord.linearize_unlinearize]
  have hout : Coord.unlinearize (Fin.cast h₂.symm (Coord.linearize
      ((Coord.appendEquiv L t).symm (b, d)))) =
      ((Coord.appendEquiv [L.size] t).symm (b₀, d)) := by
    apply Coord.linearize_injective
    apply Fin.ext
    simp only [Coord.linearize_unlinearize, Fin.val_cast]
    rw [Coord.linearize_appendEquiv_symm_val, Coord.linearize_appendEquiv_symm_val, hb₀]
  rw [reshape_apply, hout, mapLeading_apply, mapLeading_apply]
  simp only [Equiv.apply_symm_apply]
  have hslice : x.batchSlice b = (reshape x ([L.size] ++ s) h₁).batchSlice b₀ := by
    apply Rep.ext
    intro q
    rw [batchSlice_apply, batchSlice_apply, reshape_apply]
    congr 1
    apply Coord.linearize_injective
    apply Fin.ext
    simp only [Coord.linearize_unlinearize, Fin.val_cast]
    rw [Coord.linearize_appendEquiv_symm_val, Coord.linearize_appendEquiv_symm_val, hb₀]
  rw [hslice]

end TorchLean.Tensor
