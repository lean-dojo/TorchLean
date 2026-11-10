/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Data.Fintype.Basic
public import Mathlib.Data.List.NodupEquivFin
public import Mathlib.Order.Monotone.Basic
public import TorchLean.Tensor.Shape
public import TorchLean.Tensor.Shape.Coord.Conversion
public import TorchLean.Tensor.Einops.AxisTuple
public import TorchLean.Tensor.Tactic.TensorShape

/-!
# Axes of a shape

The general tensor operators name the axes they act on by `Fin s.rank`, so an axis is in range by
construction. They share the following vocabulary.

* `Shape.extent s d` is the extent of axis `d`. `Shape.Coord.get c d` is the index of a coordinate
  along it, `Shape.Coord.ofFun` builds a coordinate from its indices, and `Shape.Coord.equivFun`
  identifies the two views. A coordinate equation is proved one axis at a time with
  `Shape.Coord.ext`.
* `Shape.permute s σ` is the shape whose axis `i` is axis `σ i` of `s`.
* `Shape.BroadcastDims s t dims` says that `dims` places the axes of `s` among those of `t`, as
  JAX's `broadcast_in_dim` requires. `Shape.broadcastDims_axis s d` places a vector of extent
  `s.extent d` at axis `d`, and `Shape.insertAxis_broadcastDims` places `s` in the shape with one
  more axis, `s.insertIdx axis n`.
* `Shape.slice`, `Shape.pad` and `Shape.eraseAxes` are the output shapes of slicing, padding and
  gathering, with their rank and extent equations. `Shape.valueAt` reads the value listed for an
  axis in an axis list.
* `Shape.axis?`, `Shape.axes?` and `Shape.perm?` read signed axes at import and runtime boundaries,
  in PyTorch's convention that `-1` is the innermost axis. A negative axis is shifted once by the
  rank and then checked, with no reduction modulo the rank, so an out-of-range axis is rejected.

We wrote the signed readers by hand to refuse what torch 2.8.0 refuses. At rank 3 `x.sum(-4)`
raises "Dimension out of range (expected to be in range of [-3, 2], but got -4)" and `axis?` names
no axis there. `x.sum((0, -3))` raises "dim 0 appears multiple times in the list of dims", and
`axes?` refuses the pair (`nodup_of_axes?_eq_some`). We part ways at the scalar shape. Torch
accepts `0` and `-1` on a 0-dim tensor ("expected to be in range of [-1, 0]"), so
`torch.tensor(5.).sum(-1)` is `5`, while `axis?` names no axis of `[]`. The range rule is
`axis?_eq_some_iff`.
-/

@[expose] public section

open TorchLean.Shape (rank rank_cons rank_eq_length)

namespace TorchLean.Shape

/-! ## Extents and coordinates -/

/-- The extent of axis `d` of `s`. -/
def extent (s : Shape) (d : Fin s.rank) : Nat :=
  s[d.val]'(d.isLt.trans_eq s.rank_eq_length)

/-- The outermost axis of `n :: s` has extent `n`. -/
@[simp] theorem extent_cons_zero (n : Nat) (s : Shape) (h : 0 < Shape.rank (n :: s)) :
    extent (n :: s) ⟨0, h⟩ = n :=
  rfl

/-- An inner axis of `n :: s` has its extent in `s`. -/
@[simp] theorem extent_cons_succ (n : Nat) (s : Shape) (d : Fin s.rank) :
    extent (n :: s) d.succ = extent s d :=
  rfl

/-- A typed axis is in bounds as a natural number. -/
instance axisInBoundsFin (s : Shape) (d : Fin s.rank) : AxisInBounds d.val s :=
  ⟨d.isLt⟩

/-- `axisSize` of an axis in `Fin s.rank` is its extent. -/
theorem axisSize_eq_extent (s : Shape) (d : Fin s.rank) : s.axisSize d.val = s.extent d :=
  axisSize_eq_getElem s d.val

/-- Equal shapes have equal extents at corresponding axes. -/
theorem extent_congr {s t : Shape} (h : s = t) (d : Fin s.rank) :
    s.extent d = t.extent (d.cast (congrArg Shape.rank h)) := by
  subst h
  rfl

namespace Coord

/-- The index of a coordinate along axis `d`. -/
def get : {s : Shape} → s.Coord → (d : Fin s.rank) → Fin (s.extent d)
  | [], _, d => d.elim0
  | _ :: _, c, ⟨0, _⟩ => c.1
  | _ :: _, c, ⟨d + 1, h⟩ => get c.2 ⟨d, Nat.lt_of_succ_lt_succ h⟩

/-- The coordinate whose index along each axis `d` is `f d`. -/
def ofFun : {s : Shape} → ((d : Fin s.rank) → Fin (s.extent d)) → s.Coord
  | [], _ => PUnit.unit
  | _ :: _, f => (f ⟨0, Nat.succ_pos _⟩, ofFun fun d => f d.succ)

/-- The outermost index of `(i, c)` is `i`. -/
@[simp] theorem get_cons_zero {n : Nat} {s : Shape} (i : Fin n) (c : s.Coord)
    (h : 0 < Shape.rank (n :: s)) : get (s := n :: s) (i, c) ⟨0, h⟩ = i :=
  rfl

/-- An inner index of `(i, c)` is the index of `c`. -/
@[simp] theorem get_cons_succ {n : Nat} {s : Shape} (i : Fin n) (c : s.Coord) (d : Fin s.rank) :
    get (s := n :: s) (i, c) d.succ = get c d :=
  rfl

/-- `ofFun f` has index `f d` along axis `d`. -/
@[simp] theorem get_ofFun {s : Shape} (f : (d : Fin s.rank) → Fin (s.extent d))
    (d : Fin s.rank) : get (ofFun f) d = f d := by
  induction s with
  | scalar => exact d.elim0
  | dim n s ih =>
      match d with
      | ⟨0, _⟩ => rfl
      | ⟨d + 1, h⟩ => exact ih (fun e => f e.succ) ⟨d, Nat.lt_of_succ_lt_succ h⟩

/-- A coordinate is determined by its indices. -/
@[simp] theorem ofFun_get {s : Shape} (c : s.Coord) : ofFun (get c) = c := by
  induction s with
  | scalar => rfl
  | dim n s ih =>
      obtain ⟨i, c⟩ := c
      exact Prod.ext rfl (ih c)

/-- Two coordinates with the same index along every axis are equal. -/
theorem ext {s : Shape} {c c' : s.Coord} (h : ∀ d, get c d = get c' d) : c = c' := by
  rw [← ofFun_get c, ← ofFun_get c']
  exact congrArg ofFun (funext h)

/-- Two coordinates whose indices have the same values along every axis are equal. -/
theorem ext_val {s : Shape} {c c' : s.Coord} (h : ∀ d, (get c d).val = (get c' d).val) :
    c = c' :=
  ext fun d => Fin.ext (h d)

/-- The coordinates of a shape are the functions giving an index along every axis. -/
def equivFun (s : Shape) : s.Coord ≃ ((d : Fin s.rank) → Fin (s.extent d)) where
  toFun := get
  invFun := ofFun
  left_inv := ofFun_get
  right_inv f := funext (get_ofFun f)

/-- The index of a coordinate along the axis numbered `k`, and `0` past the last axis. Proofs
about coordinate maps compare these natural numbers, which needs no transport between the axis
types of equal shapes. -/
def index : {s : Shape} → s.Coord → Nat → Nat
  | [], _, _ => 0
  | _ :: _, c, 0 => c.1.val
  | _ :: _, c, k + 1 => index c.2 k

/-- An inner index of `(i, c)` is an index of `c`. -/
@[simp] theorem index_cons_succ {n : Nat} {s : Shape} (i : Fin n) (c : s.Coord) (k : Nat) :
    index (s := n :: s) (i, c) (k + 1) = index c k :=
  rfl

/-- The outermost index of a coordinate is its first component. -/
@[simp] theorem index_zero {n : Nat} {s : Shape} (c : Coord (n :: s)) : index c 0 = c.1.val :=
  rfl

/-- An inner index of a coordinate is an index of its tail. -/
theorem index_succ {n : Nat} {s : Shape} (c : Coord (n :: s)) (k : Nat) :
    index c (k + 1) = index c.2 k :=
  rfl

/-- Every index of a scalar coordinate is `0`. -/
@[simp] theorem index_nil (c : Coord []) (k : Nat) : index c k = 0 :=
  rfl

/-- The index along a typed axis is the index along its number. -/
theorem val_get {s : Shape} (c : s.Coord) (d : Fin s.rank) : (get c d).val = index c d.val := by
  induction s with
  | scalar => exact d.elim0
  | dim n s ih =>
      match d with
      | ⟨0, _⟩ => rfl
      | ⟨d + 1, h⟩ => exact ih c.2 ⟨d, Nat.lt_of_succ_lt_succ h⟩

/-- Past the last axis, the index is `0`. -/
theorem index_of_rank_le {s : Shape} (c : s.Coord) {k : Nat} (h : s.rank ≤ k) : index c k = 0 := by
  induction s generalizing k with
  | scalar => rfl
  | dim n s ih =>
      cases k with
      | zero => exact absurd h (by simp)
      | succ k => exact ih c.2 (by simp only [Shape.rank_cons] at h; omega)

/-- The index of `c` along axis `k` is below its extent. -/
theorem index_lt {s : Shape} (c : s.Coord) {k : Nat} (h : k < s.rank) :
    index c k < s.extent ⟨k, h⟩ := by
  rw [← val_get c ⟨k, h⟩]
  exact (get c ⟨k, h⟩).isLt

/-- The indices of `ofFun f` are those of `f`. -/
theorem index_ofFun {s : Shape} (f : (d : Fin s.rank) → Fin (s.extent d)) (k : Nat) :
    index (ofFun f) k = if h : k < s.rank then (f ⟨k, h⟩).val else 0 := by
  split_ifs with h
  · have hget := val_get (ofFun f) ⟨k, h⟩
    rw [get_ofFun] at hget
    exact hget.symm
  · exact index_of_rank_le _ (Nat.le_of_not_lt h)

/-- Two coordinates with the same `index` at every axis number below the rank are equal. -/
theorem ext_index {s : Shape} {c c' : s.Coord} (h : ∀ k < s.rank, index c k = index c' k) :
    c = c' :=
  ext_val fun d => by rw [val_get, val_get, h d.val d.isLt]

/-- Transport along a shape equality keeps every index. -/
@[simp] theorem index_cast {s t : Shape} (h : s = t) (c : s.Coord) (k : Nat) :
    index (_root_.cast (congrArg Coord h) c) k = index c k := by
  subst h
  rfl

/-- Reading an axis of a coordinate transported along a shape equality reads the corresponding
axis of the original coordinate. -/
theorem val_get_cast {s t : Shape} (h : s = t) (c : s.Coord) (d : Fin t.rank) :
    (get (_root_.cast (congrArg Coord h) c) d).val =
      (get c (d.cast (congrArg Shape.rank h).symm)).val := by
  subst h
  rfl

/-- Entry `k` of the index list of a coordinate is its index along axis `k`. -/
theorem getElem_toList : {s : Shape} → (c : s.Coord) → (k : Nat) →
    (h : k < (Coord.toList s c).length) → (Coord.toList s c)[k] = index c k
  | [], _, _, h => absurd h (Nat.not_lt_zero _)
  | _ :: _, _, 0, _ => rfl
  | _ :: _, c, k + 1, h => getElem_toList c.2 k (Nat.lt_of_succ_lt_succ h)

/-- The coordinate of `b` with the indices of `c`, for shapes `a = b`. It is built from the indices
of `c` with no cast along `a = b`, and `reindexEq_eq_cast` proves it equal to that cast. -/
def reindexEq {a b : Shape} (h : a = b) (c : a.Coord) : b.Coord :=
  ofFun fun d => ⟨index c d.val, by subst h; exact index_lt c d.isLt⟩

/-- `reindexEq` keeps every index. -/
@[simp] theorem index_reindexEq {a b : Shape} (h : a = b) (c : a.Coord) (k : Nat) :
    index (reindexEq h c) k = index c k := by
  rw [reindexEq, index_ofFun]
  split_ifs with hk
  · rfl
  · subst h
    exact (index_of_rank_le c (Nat.le_of_not_lt hk)).symm

/-- `reindexEq` along `a = a` is the identity. -/
@[simp] theorem reindexEq_self {a : Shape} (h : a = a) (c : a.Coord) : reindexEq h c = c :=
  ext_index fun k _ => index_reindexEq h c k

/-- `reindexEq h c` equals the transport of `c` along `h`. -/
theorem reindexEq_eq_cast {a b : Shape} (h : a = b) (c : a.Coord) :
    reindexEq h c = _root_.cast (congrArg Coord h) c := by
  subst h
  exact reindexEq_self rfl c

end Coord

/-! ## Permutations -/

/-- The shape whose axis `i` is axis `σ i` of `s`. -/
def permute (s : Shape) (σ : Equiv.Perm (Fin s.rank)) : Shape :=
  List.ofFn fun i => s.extent (σ i)

/-- Permuting the axes keeps the rank. -/
@[simp] theorem rank_permute (s : Shape) (σ : Equiv.Perm (Fin s.rank)) :
    (s.permute σ).rank = s.rank := by
  simp [permute, Shape.rank_eq_length]

/-- Axis `i` of `s.permute σ` has the extent of axis `σ i` of `s`. -/
theorem extent_permute (s : Shape) (σ : Equiv.Perm (Fin s.rank)) (i : Fin (s.permute σ).rank) :
    (s.permute σ).extent i = s.extent (σ (i.cast (s.rank_permute σ))) := by
  simp only [extent, permute, List.getElem_ofFn]
  rfl

/-- The axis of `s.permute σ` that reads axis `j` of `s` has the extent of axis `j`. -/
@[simp] theorem extent_permute_symm (s : Shape) (σ : Equiv.Perm (Fin s.rank)) (j : Fin s.rank) :
    (s.permute σ).extent ((σ.symm j).cast (s.rank_permute σ).symm) = s.extent j := by
  rw [extent_permute]
  simp

/-- Permuting by the identity permutation keeps the shape. -/
theorem permute_refl (s : Shape) : s.permute (Equiv.refl _) = s := by
  apply List.ext_getElem
  · simp [permute, Shape.rank_eq_length]
  · intro k h₁ h₂
    simp [permute, extent]

/-- Permuting with `σ` and then `τ` gives the shape of permuting with the composite, in which axis
`i` is axis `σ (τ i)` with the two ranks identified. -/
theorem permute_permute (s : Shape) (σ : Equiv.Perm (Fin s.rank))
    (τ : Equiv.Perm (Fin (s.permute σ).rank)) :
    s.permute (((finCongr (s.rank_permute σ)).permCongr τ).trans σ) =
      (s.permute σ).permute τ := by
  apply List.ext_getElem
  · simp [permute, Shape.rank_eq_length]
  · intro k h₁ h₂
    simp only [permute, List.getElem_ofFn]
    simp only [extent, Equiv.trans_apply, List.getElem_ofFn]
    rfl

/-- The permutation of the axes of `out = s.permute σ` that undoes `σ`: axis `j` of the result is
axis `σ⁻¹ j` of `out`. -/
def unpermute (s : Shape) (σ : Equiv.Perm (Fin s.rank)) {out : Shape}
    (h : out = s.permute σ) : Equiv.Perm (Fin out.rank) :=
  (finCongr ((congrArg Shape.rank h).trans (s.rank_permute σ)).symm).permCongr σ.symm

/-- Permuting `s.permute σ` by `s.unpermute σ` gives back `s`. -/
theorem permute_unpermute (s : Shape) (σ : Equiv.Perm (Fin s.rank)) {out : Shape}
    (h : out = s.permute σ) : out.permute (s.unpermute σ h) = s := by
  subst h
  rw [← s.permute_permute σ (s.unpermute σ rfl)]
  convert s.permute_refl using 2
  ext i
  simp [unpermute, Equiv.permCongr_apply]

/-! ## Broadcasting -/

/-- `dims` places the axes of `s` among the axes of `t` in increasing order, and each extent of
`s` is one or equals the extent of the axis of `t` it is placed at, the condition of JAX's
`broadcast_in_dim`. -/
structure BroadcastDims (s t : Shape) (dims : Fin s.rank → Fin t.rank) : Prop where
  /-- The axes keep their order. -/
  strictMono : StrictMono dims
  /-- Each extent is one or the extent of its target axis. -/
  extent_eq : ∀ i, s.extent i = 1 ∨ s.extent i = t.extent (dims i)

/-- A scalar broadcasts to every shape, with no axes to place. -/
theorem BroadcastDims.scalar (t : Shape) :
    BroadcastDims [] t (fun i => i.elim0) :=
  ⟨fun i => i.elim0, fun i => i.elim0⟩

/-- A vector along axis `d` broadcasts to `s` along that axis. -/
theorem broadcastDims_axis (s : Shape) (d : Fin s.rank) :
    BroadcastDims [s.extent d] s fun _ => d where
  strictMono i j hij := absurd hij (by
    have := j.isLt
    simp only [Shape.rank_eq_length, List.length_cons, List.length_nil] at this
    rw [Fin.lt_def]
    omega)
  extent_eq i := by
    match i with
    | ⟨0, _⟩ => exact Or.inr rfl

/-- An axis of `n :: t` other than the outermost has its extent in `t`, one position lower. -/
theorem extent_cons_of_ne_zero {n : Nat} {t : Shape} (d : Fin (Shape.dim n t).rank)
    (hd : d.val ≠ 0) :
    extent (.dim n t) d =
      t.extent ⟨d.val - 1, by have := d.isLt; simp only [Shape.rank_cons] at this; omega⟩ :=
  match d, hd with
  | ⟨_ + 1, _⟩, _ => rfl

/-- A placement into `n :: t` that avoids its outermost axis, as a placement into `t`. -/
def BroadcastDims.shiftDims {r n : Nat} {t : Shape} (dims : Fin r → Fin (Shape.dim n t).rank)
    (avoids : ∀ i, (dims i).val ≠ 0) : Fin r → Fin t.rank :=
  fun i => ⟨(dims i).val - 1, by
    have := (dims i).isLt; have := avoids i; simp only [Shape.rank_cons] at *; omega⟩

/-- A placement of `s` into `n :: t` that leaves the outermost axis of `n :: t` unplaced is a
placement into `t`. -/
theorem BroadcastDims.shift {s : Shape} {n : Nat} {t : Shape}
    {dims : Fin s.rank → Fin (Shape.dim n t).rank} (h : BroadcastDims s (.dim n t) dims)
    (avoids : ∀ i, (dims i).val ≠ 0) : BroadcastDims s t (shiftDims dims avoids) where
  strictMono a b hab := by
    have := h.strictMono hab; have := avoids a; have := avoids b
    simp only [Fin.lt_def, shiftDims] at *; omega
  extent_eq i := by
    rcases h.extent_eq i with e | e
    · exact Or.inl e
    · exact Or.inr (e.trans (extent_cons_of_ne_zero _ (avoids i)))

/-- The inner axes of `m :: s` are placed past the outermost axis of `n :: t`. -/
theorem BroadcastDims.avoids_succ {m n : Nat} {s t : Shape}
    {dims : Fin (Shape.dim m s).rank → Fin (Shape.dim n t).rank}
    (h : BroadcastDims (.dim m s) (.dim n t) dims) (i : Fin s.rank) :
    (dims i.succ).val ≠ 0 := by
  have := h.strictMono (show (⟨0, Nat.succ_pos _⟩ : Fin (s.rank + 1)) < i.succ from
    by simp [Fin.lt_def])
  simp only [Fin.lt_def] at this; omega

/-- When the outermost axis of `m :: s` is not placed at the outermost axis of `n :: t`, no axis
is. -/
theorem BroadcastDims.avoids_of_head {m n : Nat} {s t : Shape}
    {dims : Fin (Shape.dim m s).rank → Fin (Shape.dim n t).rank}
    (h : BroadcastDims (.dim m s) (.dim n t) dims)
    (unplaced : (dims ⟨0, Nat.succ_pos _⟩).val ≠ 0) (i : Fin (Shape.dim m s).rank) :
    (dims i).val ≠ 0 := by
  have head : dims ⟨0, Nat.succ_pos _⟩ ≤ dims i :=
    h.strictMono.monotone (Fin.le_def.mpr (Nat.zero_le _))
  rw [Fin.le_def] at head; omega

/-- The placement of the inner axes of `m :: s` in the inner axes of `n :: t`. -/
theorem BroadcastDims.tail {m n : Nat} {s t : Shape}
    {dims : Fin (Shape.dim m s).rank → Fin (Shape.dim n t).rank}
    (h : BroadcastDims (.dim m s) (.dim n t) dims) :
    BroadcastDims s t (shiftDims (fun i => dims i.succ) h.avoids_succ) where
  strictMono a b hab := by
    have := h.strictMono (Fin.succ_lt_succ_iff.mpr hab : a.succ < b.succ)
    have := h.avoids_succ a; have := h.avoids_succ b
    simp only [Fin.lt_def, shiftDims] at *; omega
  extent_eq i := by
    rcases h.extent_eq i.succ with e | e
    · exact Or.inl e
    · exact Or.inr (e.trans (extent_cons_of_ne_zero _ (h.avoids_succ i)))

/-- Decide `BroadcastDims s t dims`: `dims` is strictly increasing and each extent of `s` is one
or the extent of `t` at its image. -/
instance (s t : Shape) (dims : Fin s.rank → Fin t.rank) : Decidable (BroadcastDims s t dims) :=
  decidable_of_iff ((∀ i j, i < j → dims i < dims j) ∧
      ∀ i, s.extent i = 1 ∨ s.extent i = t.extent (dims i))
    ⟨fun h => ⟨fun _ _ hij => h.1 _ _ hij, h.2⟩, fun h => ⟨fun _ _ hij => h.1 hij, h.2⟩⟩

/-- Axis `i` of `s` has its extent at position `i` or `i + 1` of `s.insertIdx axis n`. -/
theorem extent_insertAxis {s : Shape} {axis : Nat} (h : axis ≤ s.rank)
    (n : Nat) (i : Fin s.rank) :
    Shape.extent (s.insertIdx axis n) ⟨if i.val < axis then i.val else i.val + 1, by
        rw [Shape.rank_eq_length (s.insertIdx axis n), List.length_insertIdx_of_le_length
          (h.trans_eq s.rank_eq_length), ← Shape.rank_eq_length s]
        split <;> omega⟩ = s.extent i := by
  simp only [Shape.extent]
  rw [← Option.some_inj, ← List.getElem?_eq_getElem, ← List.getElem?_eq_getElem,
    List.getElem?_insertIdx]
  split_ifs <;> first | omega | congr 1

/-- The axes of `s` keep their extents in `s.insertIdx axis n`, shifted past `axis`. -/
theorem insertAxis_broadcastDims {s : Shape} {axis : Nat} (h : axis ≤ s.rank)
    (n : Nat) :
    Shape.BroadcastDims s (s.insertIdx axis n)
      (fun i => ⟨if i.val < axis then i.val else i.val + 1, by
        rw [Shape.rank_eq_length (s.insertIdx axis n), List.length_insertIdx_of_le_length
          (h.trans_eq s.rank_eq_length), ← Shape.rank_eq_length s]
        split <;> omega⟩) where
  strictMono a b hab := by
    simp only [Fin.lt_def] at hab ⊢
    split_ifs <;> omega
  extent_eq i := Or.inr (Shape.extent_insertAxis h n i).symm

/-! ## Axis lists -/

/-- The value listed for axis `d`: `values[k]` for the first position `k` of `d` in `dims`, and
`default` if `d` is not listed or has no value. -/
def valueAt {r : Nat} : List (Fin r) → List Nat → Nat → Fin r → Nat
  | d :: dims, v :: values, default, a => if a = d then v else valueAt dims values default a
  | _, _, default, _ => default

/-- An axis that is not listed takes the default value. -/
theorem valueAt_of_not_mem {r : Nat} {dims : List (Fin r)} {a : Fin r} (h : a ∉ dims)
    (values : List Nat) (default : Nat) : valueAt dims values default a = default := by
  induction dims generalizing values with
  | nil => rfl
  | cons d dims ih =>
      cases values with
      | nil => rfl
      | cons v values =>
          simp only [List.mem_cons, not_or] at h
          simp [valueAt, h.1, ih h.2]

/-- The value listed for the only axis of a one-axis list. -/
@[simp] theorem valueAt_singleton {r : Nat} (d a : Fin r) (v default : Nat) :
    valueAt [d] [v] default a = if a = d then v else default := by
  by_cases h : a = d <;> simp [valueAt, h]

/-! ## Slicing -/

/-- The shape of `slice dims starts stops steps`: along a listed axis, the number of indices
`start, start + step, ...` below `stop`, and every other axis kept whole. -/
def slice (s : Shape) (dims : List (Fin s.rank)) (starts stops steps : List Nat) : Shape :=
  List.ofFn fun d =>
    (valueAt dims stops (s.extent d) d - valueAt dims starts 0 d + valueAt dims steps 1 d - 1) /
      valueAt dims steps 1 d

/-- Slicing keeps the rank. -/
@[simp] theorem rank_slice (s : Shape) (dims : List (Fin s.rank)) (starts stops steps : List Nat) :
    (s.slice dims starts stops steps).rank = s.rank := by
  simp [slice, Shape.rank_eq_length]

/-- The extent of a sliced axis. -/
theorem extent_slice (s : Shape) (dims : List (Fin s.rank)) (starts stops steps : List Nat)
    (d : Fin (s.slice dims starts stops steps).rank) :
    (s.slice dims starts stops steps).extent d =
      let d' := d.cast (s.rank_slice dims starts stops steps)
      (valueAt dims stops (s.extent d') d' - valueAt dims starts 0 d' + valueAt dims steps 1 d' -
        1) / valueAt dims steps 1 d' := by
  simp only [extent, slice, List.getElem_ofFn]
  rfl

/-- The arguments of `slice dims starts stops steps`: distinct axes, one start, stop and step for
each, every stop at most the extent of its axis and every step positive. A start at or past its
stop gives an empty axis. -/
structure SliceValid (s : Shape) (dims : List (Fin s.rank)) (starts stops steps : List Nat) :
    Prop where
  /-- The axes are distinct. -/
  nodup : dims.Nodup
  /-- There is one start per axis. -/
  length_starts : starts.length = dims.length
  /-- There is one stop per axis. -/
  length_stops : stops.length = dims.length
  /-- There is one step per axis. -/
  length_steps : steps.length = dims.length
  /-- Every stop is at most the extent of its axis. -/
  stop_le : ∀ d, valueAt dims stops (s.extent d) d ≤ s.extent d
  /-- Every step is positive. -/
  step_pos : ∀ d, 0 < valueAt dims steps 1 d

/-- Decide `SliceValid` by checking each of its conditions. -/
instance (s : Shape) (dims : List (Fin s.rank)) (starts stops steps : List Nat) :
    Decidable (SliceValid s dims starts stops steps) :=
  decidable_of_iff (dims.Nodup ∧ starts.length = dims.length ∧ stops.length = dims.length ∧
      steps.length = dims.length ∧ (∀ d, valueAt dims stops (s.extent d) d ≤ s.extent d) ∧
      ∀ d, 0 < valueAt dims steps 1 d)
    ⟨fun ⟨a, b, c, d, e, f⟩ => ⟨a, b, c, d, e, f⟩, fun ⟨a, b, c, d, e, f⟩ => ⟨a, b, c, d, e, f⟩⟩

/-- Index `i` of a sliced axis is in range: `start + step * i` is below the stop. -/
theorem add_mul_lt_of_lt_sliceExtent {start stop step i : Nat} (hstep : 0 < step)
    (hi : i < (stop - start + step - 1) / step) : start + step * i < stop := by
  have hle := (Nat.le_div_iff_mul_le hstep).mp (Nat.succ_le_of_lt hi)
  rw [Nat.succ_mul, Nat.mul_comm] at hle
  omega

/-- Index `i` along a sliced axis `d` reads index `start + step * i` along `d`, which is in
range. -/
theorem SliceValid.add_mul_lt {s : Shape} {dims : List (Fin s.rank)} {starts stops steps : List Nat}
    (h : SliceValid s dims starts stops steps) (d : Fin s.rank)
    (i : Fin ((s.slice dims starts stops steps).extent
      (d.cast (s.rank_slice dims starts stops steps).symm))) :
    valueAt dims starts 0 d + valueAt dims steps 1 d * i.val < s.extent d := by
  have hi := i.isLt
  simp only [extent_slice, Fin.cast_cast, Fin.cast_eq_self] at hi
  exact (add_mul_lt_of_lt_sliceExtent (h.step_pos d) hi).trans_le (h.stop_le d)

/-- A one-axis slice with step one replaces the extent of that axis by the length of the slice. -/
theorem slice_singleton (s : Shape) (d : Fin s.rank) (start count : Nat) :
    s.slice [d] [start] [start + count] [1] = s.set d count := by
  apply List.ext_getElem
  · simp [slice, Shape.rank_eq_length]
  · intro k h₁ h₂
    simp only [slice, List.getElem_ofFn, valueAt_singleton, List.getElem_set,
      Fin.ext_iff, extent]
    split_ifs <;> omega

/-- A one-axis slice with step one and stop at most the extent is valid. -/
theorem sliceValid_singleton (s : Shape) (d : Fin s.rank) (start count : Nat)
    (h : start + count ≤ s.extent d) : SliceValid s [d] [start] [start + count] [1] where
  nodup := List.nodup_singleton d
  length_starts := rfl
  length_stops := rfl
  length_steps := rfl
  stop_le e := by
    rw [valueAt_singleton]
    split_ifs with he
    · exact he ▸ h
    · exact le_rfl
  step_pos e := by
    rw [valueAt_singleton]
    split_ifs <;> exact Nat.one_pos

/-! ## Padding -/

/-- The shape of `pad low high interior`: along axis `d`, `low d` added indices before the input,
`high d` after it, and `interior d` between consecutive input indices. -/
def pad (s : Shape) (low high interior : Fin s.rank → Nat) : Shape :=
  List.ofFn fun d => low d + high d + (s.extent d + (s.extent d - 1) * interior d)

/-- Padding keeps the rank. -/
@[simp] theorem rank_pad (s : Shape) (low high interior : Fin s.rank → Nat) :
    (s.pad low high interior).rank = s.rank := by
  simp [pad, Shape.rank_eq_length]

/-- The extent of a padded axis. -/
theorem extent_pad (s : Shape) (low high interior : Fin s.rank → Nat)
    (d : Fin (s.pad low high interior).rank) :
    (s.pad low high interior).extent d =
      let d' := d.cast (s.rank_pad low high interior)
      low d' + high d' + (s.extent d' + (s.extent d' - 1) * interior d') := by
  simp only [extent, pad, List.getElem_ofFn]
  rfl

/-! ## Erasing axes -/

/-- The axes of `s` outside `dims`, in increasing order. -/
def otherAxes (s : Shape) (dims : List (Fin s.rank)) : List (Fin s.rank) :=
  (List.finRange s.rank).filter (· ∉ dims)

/-- An axis is among the other axes exactly when it is not listed. -/
@[simp] theorem mem_otherAxes {s : Shape} {dims : List (Fin s.rank)} {d : Fin s.rank} :
    d ∈ s.otherAxes dims ↔ d ∉ dims := by
  simp [otherAxes]

/-- The other axes are distinct. -/
theorem nodup_otherAxes (s : Shape) (dims : List (Fin s.rank)) : (s.otherAxes dims).Nodup :=
  (List.nodup_finRange s.rank).filter _

/-- The shape of the axes of `s` outside `dims`, in their order in `s`. -/
def eraseAxes (s : Shape) (dims : List (Fin s.rank)) : Shape :=
  (s.otherAxes dims).map s.extent

/-- Erasing every axis leaves the scalar shape. -/
theorem eraseAxes_finRange (s : Shape) : s.eraseAxes (List.finRange s.rank) = [] := by
  simp [eraseAxes, otherAxes]

/-- The arguments of `gather x dims batchDims indices`: the indexed axes `dims` and the batch axes
of `x` are distinct, no axis of the index tensor batches two axes of `x`, and paired axes have
equal extents. -/
structure GatherValid (s t : Shape) (dims : List (Fin s.rank))
    (batchDims : List (Fin s.rank × Fin t.rank)) : Prop where
  /-- The indexed and batch axes of the operand are distinct. -/
  nodup : (dims ++ batchDims.map Prod.fst).Nodup
  /-- The batch axes of the index tensor are distinct. -/
  nodup_snd : (batchDims.map Prod.snd).Nodup
  /-- Paired batch axes have equal extents. -/
  extent_eq : ∀ p ∈ batchDims, s.extent p.1 = t.extent p.2

/-- The operand coordinate that `gather` reads: index `position k` along the indexed axis
`dims[k]`, the index of `ct` along the paired axis of the index tensor for a batch axis, and the
index of `cr` for each remaining axis. -/
def GatherValid.source {s t : Shape} {dims : List (Fin s.rank)}
    {batchDims : List (Fin s.rank × Fin t.rank)} (h : GatherValid s t dims batchDims)
    (position : Einops.AxisTuple s.extent dims) (ct : t.Coord)
    (cr : (s.eraseAxes (dims ++ batchDims.map Prod.fst)).Coord) : s.Coord :=
  Coord.ofFun fun a =>
    if ha : a ∈ dims then
      Fin.cast (congrArg s.extent (List.idxOf_get _))
        (position ⟨dims.idxOf a, List.idxOf_lt_length_iff.mpr ha⟩)
    else
      match hb : batchDims.find? (fun p => p.1 = a) with
      | some p =>
          Fin.cast
            ((h.extent_eq p (List.mem_of_find?_eq_some hb)).symm.trans
              (congrArg s.extent
                (of_decide_eq_true
                  (List.find?_some (p := fun q : Fin s.rank × Fin t.rank => decide (q.1 = a))
                    hb))))
            (Coord.get ct p.2)
      | none =>
          have hRest : a ∈ s.otherAxes (dims ++ batchDims.map Prod.fst) := by
            simp only [mem_otherAxes, List.mem_append, List.mem_map, not_or, not_exists, not_and]
            exact ⟨ha, fun p hp hpa => by
              simpa [hpa] using List.find?_eq_none.mp hb p hp⟩
          Fin.cast (congrArg s.extent (List.idxOf_get _))
            ((Einops.AxisTuple.coordEquiv s.extent
                (s.otherAxes (dims ++ batchDims.map Prod.fst))) cr
              ⟨(s.otherAxes _).idxOf a, List.idxOf_lt_length_iff.mpr hRest⟩)

/-! ## Index selection -/

/-- The axis of an index selection along `dim`, as the one-axis list `gather` indexes. -/
def selectDims (s : Shape) (dim : Fin s.rank) : List (Fin s.rank) :=
  [dim]

/-- The listed axis of `s.selectDims dim` is `dim`. -/
theorem get_selectDims (s : Shape) (dim : Fin s.rank)
    (i : Fin (s.selectDims dim).length) : (s.selectDims dim).get i = dim := by
  obtain ⟨_ | k, hk⟩ := i
  · rfl
  · simp [selectDims] at hk

/-- The batch pairs of an index selection along `dim`: every other axis of `s`, paired with the
same axis of `s.set dim count`. -/
def selectBatch (s : Shape) (dim : Fin s.rank) (count : Nat) :
    List (Fin s.rank × Fin (Shape.rank (s.set dim count))) :=
  (s.otherAxes (s.selectDims dim)).map fun a => (a, a.cast rank_set.symm)

/-- The batch axes of an index selection are the other axes. -/
theorem map_fst_selectBatch (s : Shape) (dim : Fin s.rank) (count : Nat) :
    (s.selectBatch dim count).map Prod.fst = s.otherAxes (s.selectDims dim) := by
  simp [selectBatch, List.map_map, Function.comp_def]

/-- An index selection along `dim` is a valid gather: it indexes `dim` and batches every other
axis with itself. -/
theorem GatherValid.select (s : Shape) (dim : Fin s.rank) (count : Nat) :
    GatherValid s (s.set dim count) (s.selectDims dim) (s.selectBatch dim count) where
  nodup := by
    rw [map_fst_selectBatch, selectDims, List.singleton_append, List.nodup_cons]
    exact ⟨by simp, s.nodup_otherAxes _⟩
  nodup_snd := by
    simp only [selectBatch, List.map_map, Function.comp_def]
    exact (s.nodup_otherAxes _).map fun a b h => Fin.ext (by simpa using congrArg Fin.val h)
  extent_eq := by
    intro p hp
    simp only [selectBatch, selectDims, List.mem_map, mem_otherAxes, List.mem_singleton] at hp
    obtain ⟨a, ha, rfl⟩ := hp
    simp only [extent, Fin.val_cast]
    rw [List.getElem_set_ne (by intro h; exact ha (Fin.ext h).symm)]

/-- An index selection leaves no operand axis outside its indexed axis and its batch axes. -/
@[tensor_shape_simps] theorem eraseAxes_selectBatch (s : Shape) (dim : Fin s.rank)
    (count : Nat) :
    s.eraseAxes (s.selectDims dim ++ (s.selectBatch dim count).map Prod.fst) = [] := by
  rw [map_fst_selectBatch, eraseAxes, List.map_eq_nil_iff, otherAxes, List.filter_eq_nil_iff]
  intro a _
  by_cases h : a ∈ s.selectDims dim <;> simp [h, mem_otherAxes]

attribute [tensor_shape_simps] List.append_nil

/-! ## Signed axes at the boundaries -/

/-- The axis of `s` named by a signed index: `-1` is the innermost axis, as in PyTorch. A negative
index is shifted once by the rank, and the result must lie in `[0, rank)`. -/
def axis? (s : Shape) (dim : Int) : Option (Fin s.rank) :=
  if h : 0 ≤ dim ∧ dim < s.rank then some ⟨dim.toNat, by omega⟩
  else if h' : dim < 0 ∧ 0 ≤ dim + s.rank then some ⟨(dim + s.rank).toNat, by omega⟩
  else none

/-- `s.axis? dim` names `d` exactly when `dim` is `d`, or `dim` is negative and `dim + rank` is
`d`. In particular no index is reduced modulo the rank. -/
theorem axis?_eq_some_iff {s : Shape} {dim : Int} {d : Fin s.rank} :
    s.axis? dim = some d ↔ (0 ≤ dim ∧ dim = d) ∨ (dim < 0 ∧ dim + s.rank = d) := by
  have hd := d.isLt
  unfold axis?
  split_ifs with h h'
  · simp only [Option.some.injEq, Fin.ext_iff]
    omega
  · simp only [Option.some.injEq, Fin.ext_iff]
    omega
  · simp only [false_iff]
    omega

/-- A natural axis names itself. -/
@[simp] theorem axis?_natCast (s : Shape) (d : Fin s.rank) : s.axis? (d.val : Int) = some d :=
  axis?_eq_some_iff.mpr (Or.inl ⟨Int.natCast_nonneg _, rfl⟩)

/-- Axis `-1` is the innermost axis. -/
theorem axis?_neg_one {s : Shape} (h : 0 < s.rank) :
    s.axis? (-1) = some ⟨s.rank - 1, Nat.sub_lt h Nat.one_pos⟩ :=
  axis?_eq_some_iff.mpr (Or.inr ⟨by omega, by simp; omega⟩)

/-- An index at or beyond the rank names no axis. -/
theorem axis?_eq_none_of_rank_le {s : Shape} {dim : Int} (h : (s.rank : Int) ≤ dim) :
    s.axis? dim = none := by
  unfold axis?
  split_ifs <;> first | rfl | omega

/-- An index below `-rank` names no axis. -/
theorem axis?_eq_none_of_lt_neg {s : Shape} {dim : Int} (h : dim < -(s.rank : Int)) :
    s.axis? dim = none := by
  unfold axis?
  split_ifs <;> first | rfl | omega

/-- The distinct axes of `s` named by an array of signed indices, or `none` if an index names no
axis or two indices name the same axis. -/
def axes? (s : Shape) (dims : Array Int) : Option (List (Fin s.rank)) := do
  let axes ← dims.toList.mapM s.axis?
  if axes.Nodup then some axes else none

/-- The axes read by `axes?` are distinct. -/
theorem nodup_of_axes?_eq_some {s : Shape} {dims : Array Int} {axes : List (Fin s.rank)}
    (h : s.axes? dims = some axes) : axes.Nodup := by
  unfold axes? at h
  cases hm : dims.toList.mapM s.axis? with
  | none => simp [hm] at h
  | some read =>
      by_cases hn : read.Nodup
      · simp only [hm, hn, Option.bind_eq_bind, Option.bind_some, ite_true,
          Option.some.injEq] at h
        exact h ▸ hn
      · simp [hm, hn] at h

/-- The permutation `σ` with `σ i` the axis named by `dims[i]`, as in PyTorch's
`x.permute(dims)`: axis `i` of the result is axis `dims[i]` of `x`. It is `none` unless the
indices name every axis exactly once. -/
def perm? (s : Shape) (dims : Array Int) : Option (Equiv.Perm (Fin s.rank)) :=
  match hAxes : s.axes? dims with
  | none => none
  | some axes =>
      if h : axes.length = s.rank ∧ ∀ d, d ∈ axes then
        some ((finCongr h.1.symm).trans
          (List.Nodup.getEquivOfForallMemList axes (nodup_of_axes?_eq_some hAxes) h.2))
      else none

end TorchLean.Shape
