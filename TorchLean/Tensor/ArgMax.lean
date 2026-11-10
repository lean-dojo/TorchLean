/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Reduce.Backward

/-!
# Indices of extrema along an axis

`argmax x dim keepdim` and `argmin x dim keepdim` hold, at each coordinate of the other axes, the
index along `dim` of a largest or smallest entry of its fiber, as PyTorch's `argmax` and `argmin`
with `dim` and `keepdim`. Each output entry is one left fold over the fiber in index order, from
index `0`.

For both operators, an entry that is not equal to itself (NaN under IEEE equality) is never
replaced and replaces every entry that equals itself, so the first NaN wins as in PyTorch.
Between entries that equal themselves, a later entry replaces the current choice only when it is
strictly larger (strictly smaller for `argmin`), so ties keep the first index. These policies
hold for every `BEq` and `LT`, and the extremality theorems also assume that the order reflects a
real reading of the entries.

The tie rule is the documented one: the PyTorch 2.8 pages of `torch.argmax` and `torch.argmin`
say they return the first of several extremal entries. The NaN rule isn't on those pages, so we
measured it on torch 2.8.0, where `[1, 3, 3, NaN, 2, NaN]` gives `3` for both operators. `-0` and
`0` are equal, so they tie and the first index stays. The capture test
`test_index_outputs_match_eager` in `scripts/checks/test_ir_capture.py` compares the indices with
eager PyTorch in binary32, binary64, bfloat16 and binary16.

* `Shape.axisSource` is the input coordinate a reduction over one axis reads at an output
  coordinate and an index along that axis, `Shape.axisTarget` the output coordinate whose fiber
  holds an input coordinate, and `Shape.Coord.update` replaces the index of a coordinate along one
  axis.
* `argmax_apply` and `argmin_apply` read an output entry as its fold over the fiber, and
  `item_argmax_vector` and `item_argmin_vector` read the index of a vector as a fold over its
  entries.
* `le_foldl_select` is the invariant of such a fold. From it, `le_argmax`, `argmin_le`,
  `le_getScalar_argmax` and `getScalar_argmin_le` show that the selected entry is an extremum for
  every real reading that reflects the strict order of the entries, when each entry equals itself.
-/

@[expose] public section

open TorchLean.Shape (rank_eq_length)

namespace TorchLean.Shape

/-- The coordinate of `s` that a reduction over the one axis `dim` reads at the output coordinate
`o` and the index `j` along `dim`. -/
def axisSource (s : Shape) (dim : Fin s.rank) (keepdim : Bool) (o : (s.reduce [dim] keepdim).Coord)
    (j : Fin (s.extent dim)) : s.Coord :=
  s.reduceSource [dim] keepdim o (Coord.reindexEq (s.selectAxes_singleton dim).symm (j, PUnit.unit))

/-- The coordinate `axisSource` reads has index `j` along `dim`. -/
@[simp] theorem get_axisSource (s : Shape) (dim : Fin s.rank) (keepdim : Bool)
    (o : (s.reduce [dim] keepdim).Coord) (j : Fin (s.extent dim)) :
    Coord.get (s.axisSource dim keepdim o j) dim = j := by
  apply Fin.ext
  rw [Coord.val_get, axisSource, index_reduceSource, dite_eq_left dim.isLt,
    ite_eq_left (List.mem_singleton_self _), sortedAxes_singleton, List.idxOf_cons_self,
    Coord.index_reindexEq, Coord.index_zero]

/-- Along another axis, the coordinate `axisSource` reads keeps the index of the output coordinate
at that axis's position. -/
theorem get_axisSource_of_ne (s : Shape) {dim : Fin s.rank} (keepdim : Bool)
    (o : (s.reduce [dim] keepdim).Coord) (j : Fin (s.extent dim)) {d : Fin s.rank}
    (hd : d ≠ dim) :
    (Coord.get (s.axisSource dim keepdim o j) d).val =
      Coord.index o (s.keptIndex [dim] keepdim d) := by
  rw [Coord.val_get, axisSource, index_reduceSource, dite_eq_left d.isLt,
    ite_eq_right (by simpa using hd)]

/-- Two indices along `dim` read the same coordinate of the other axes. -/
theorem axisSource_eq_iff (s : Shape) (dim : Fin s.rank) (keepdim : Bool)
    (o : (s.reduce [dim] keepdim).Coord) (i j : Fin (s.extent dim)) :
    s.axisSource dim keepdim o i = s.axisSource dim keepdim o j ↔ i = j := by
  constructor
  · intro h
    simpa using congrArg (fun c => Coord.get c dim) h
  · rintro rfl
    rfl

namespace Coord

/-- The coordinate `c` with its index along `dim` replaced by `j`. -/
def update {s : Shape} (c : s.Coord) (dim : Fin s.rank) (j : Fin (s.extent dim)) : s.Coord :=
  ofFun (Function.update (get c) dim j)

/-- An updated coordinate has index `j` along the updated axis. -/
@[simp] theorem get_update_self {s : Shape} (c : s.Coord) (dim : Fin s.rank)
    (j : Fin (s.extent dim)) : get (c.update dim j) dim = j := by
  simp [update]

/-- An updated coordinate keeps its index along the other axes. -/
theorem get_update_of_ne {s : Shape} (c : s.Coord) {dim d : Fin s.rank} (j : Fin (s.extent dim))
    (h : d ≠ dim) : get (c.update dim j) d = get c d := by
  simp [update, Function.update_of_ne h]

/-- Updating a coordinate with its own index changes nothing. -/
@[simp] theorem update_get {s : Shape} (c : s.Coord) (dim : Fin s.rank) :
    c.update dim (get c dim) = c := by
  simp [update]

/-- A second update along the same axis replaces the first. -/
@[simp] theorem update_update {s : Shape} (c : s.Coord) (dim : Fin s.rank)
    (i j : Fin (s.extent dim)) : (c.update dim i).update dim j = c.update dim j := by
  apply ext
  intro d
  by_cases h : d = dim
  · subst h
    simp
  · rw [get_update_of_ne _ _ h, get_update_of_ne _ _ h, get_update_of_ne _ _ h]

end Coord

/-- The coordinates a reduction over one axis reads at one output coordinate differ only along that
axis. -/
theorem update_axisSource (s : Shape) (dim : Fin s.rank) (keepdim : Bool)
    (o : (s.reduce [dim] keepdim).Coord) (i j : Fin (s.extent dim)) :
    (s.axisSource dim keepdim o i).update dim j = s.axisSource dim keepdim o j := by
  apply Coord.ext
  intro d
  by_cases h : d = dim
  · subst h
    simp
  · rw [Coord.get_update_of_ne _ _ h]
    exact Fin.ext ((get_axisSource_of_ne s keepdim o i h).trans
      (get_axisSource_of_ne s keepdim o j h).symm)

/-- The output coordinate of a reduction over the one axis `dim`, without `keepdim`, whose fiber
holds `c`. -/
def axisTarget (s : Shape) (dim : Fin s.rank) (c : s.Coord) : (s.reduce [dim] false).Coord :=
  Coord.ofFun fun d =>
    Fin.cast (by
        simp only [reduce, Bool.cond_false, eraseAxes, extent, List.getElem_map])
      (Coord.get c ((s.otherAxes [dim])[d.val]'(by
        have := d.isLt
        simp only [reduce, Bool.cond_false, eraseAxes, Shape.rank_eq_length,
            List.length_map] at this
        exact this)))

/-- The fiber of `axisTarget c` is the fiber through `c`. -/
theorem axisSource_axisTarget (s : Shape) (dim : Fin s.rank) (c : s.Coord)
    (j : Fin (s.extent dim)) :
    s.axisSource dim false (s.axisTarget dim c) j = c.update dim j := by
  apply Coord.ext_val
  intro d
  by_cases hd : d = dim
  · subst hd
    simp
  · rw [get_axisSource_of_ne s false _ j hd, Coord.get_update_of_ne _ _ hd, keptIndex,
      Bool.cond_false]
    have hmem : d ∈ s.otherAxes [dim] := mem_otherAxes.mpr (by simpa using hd)
    have hlt : (s.otherAxes [dim]).idxOf d < (s.otherAxes [dim]).length :=
      List.idxOf_lt_length_iff.mpr hmem
    rw [axisTarget, Coord.index_ofFun]
    rw [dite_eq_left (by simpa [reduce, eraseAxes, Shape.rank_eq_length] using hlt)]
    exact congrArg (fun e => (Coord.get c e).val) (List.getElem_idxOf hlt)

end TorchLean.Shape

namespace TorchLean.Tensor

/-! ## The selection fold -/

/-- The left fold that moves from its current index `j` to the next index `i` when
`better (value i) (value j)` holds ends at a largest entry of `start :: indices` for `observe`,
provided a move never lowers `observe` and staying never skips a larger entry. -/
theorem le_foldl_select {ι β : Type} (better : β → β → Bool) (value : ι → β) (observe : β → ℝ)
    (hmove : ∀ i j, better (value i) (value j) = true → observe (value j) ≤ observe (value i))
    (hstay : ∀ i j, better (value i) (value j) = false → observe (value i) ≤ observe (value j))
    (indices : List ι) (start i : ι) (hi : i ∈ start :: indices) :
    observe (value i) ≤ observe (value (indices.foldl
      (fun best k => if better (value k) (value best) then k else best) start)) := by
  induction indices generalizing start i with
  | nil =>
      rw [List.mem_singleton.mp hi]
      exact le_rfl
  | cons k rest ih =>
      rw [List.foldl_cons]
      have hstep : observe (value start) ≤
            observe (value (if better (value k) (value start) then k else start)) ∧
          observe (value k) ≤
            observe (value (if better (value k) (value start) then k else start)) := by
        cases hb : better (value k) (value start) with
        | false =>
            rw [ite_eq_right Bool.false_ne_true]
            exact ⟨le_rfl, hstay k start hb⟩
        | true =>
            rw [ite_eq_left rfl]
            exact ⟨hmove k start hb, le_rfl⟩
      have htop := ih (if better (value k) (value start) then k else start) _
        List.mem_cons_self
      rcases List.mem_cons.mp hi with rfl | hi
      · exact hstep.1.trans htop
      · rcases List.mem_cons.mp hi with rfl | hi
        · exact hstep.2.trans htop
        · exact ih _ i (List.mem_cons_of_mem _ hi)

/-- A left fold that either keeps its choice or moves to the next element ends at its start or at
an element of the list. -/
theorem foldl_select_mem {ι : Type} (move : ι → ι → Bool) (indices : List ι) (start : ι) :
    indices.foldl (fun best k => if move best k then k else best) start ∈ start :: indices := by
  induction indices generalizing start with
  | nil => exact List.mem_singleton_self _
  | cons k rest ih =>
      rw [List.foldl_cons]
      by_cases hm : move start k = true
      · rw [ite_eq_left hm]
        exact List.mem_cons_of_mem _ (ih k)
      · rw [ite_eq_right hm]
        rcases List.mem_cons.mp (ih start) with he | hr
        · rw [he]
          exact List.mem_cons_self
        · exact List.mem_cons_of_mem _ (List.mem_cons_of_mem _ hr)

variable {α : Type} [Storage α] {s : Shape} [BEq α] [LT α]
  [DecidableRel ((· > ·) : α → α → Prop)]

/-! ## The operators -/

/-- Indices along `dim` of the largest entries of `x`, as PyTorch's `argmax(dim, keepdim)`. Each
output entry folds over its fiber in index order from index `0`. It keeps the current choice when
that is not equal to itself (NaN), and otherwise moves to an entry that is NaN or strictly larger.
The first NaN therefore wins, and ties keep the first index. The axis must be nonempty. -/
def argmax (x : Tensor α s) (dim : Fin s.rank) (keepdim : Bool)
    (h : 0 < s.extent dim := by first | decide | simp) :
    Tensor (Fin (s.extent dim)) (s.reduce [dim] keepdim) :=
  Rep.ofFn fun o => (List.finRange (s.extent dim)).foldl (fun best j =>
    if (x (s.axisSource dim keepdim o best) == x (s.axisSource dim keepdim o best)) &&
        (!(x (s.axisSource dim keepdim o j) == x (s.axisSource dim keepdim o j)) ||
          decide (x (s.axisSource dim keepdim o j) > x (s.axisSource dim keepdim o best)))
    then j else best) ⟨0, h⟩

/-- Indices along `dim` of the smallest entries of `x`, as PyTorch's `argmin(dim, keepdim)`. The
fold is the one of `argmax` with the strict comparison reversed, so the first NaN wins and ties
keep the first index. The axis must be nonempty. -/
def argmin (x : Tensor α s) (dim : Fin s.rank) (keepdim : Bool)
    (h : 0 < s.extent dim := by first | decide | simp) :
    Tensor (Fin (s.extent dim)) (s.reduce [dim] keepdim) :=
  Rep.ofFn fun o => (List.finRange (s.extent dim)).foldl (fun best j =>
    if (x (s.axisSource dim keepdim o best) == x (s.axisSource dim keepdim o best)) &&
        (!(x (s.axisSource dim keepdim o j) == x (s.axisSource dim keepdim o j)) ||
          decide (x (s.axisSource dim keepdim o best) > x (s.axisSource dim keepdim o j)))
    then j else best) ⟨0, h⟩

/-- An entry of `argmax` is the fold of its definition over the fiber. -/
theorem argmax_apply (x : Tensor α s) (dim : Fin s.rank) (keepdim : Bool) (h : 0 < s.extent dim)
    (o : (s.reduce [dim] keepdim).Coord) :
    argmax x dim keepdim h o = (List.finRange (s.extent dim)).foldl (fun best j =>
      if (x (s.axisSource dim keepdim o best) == x (s.axisSource dim keepdim o best)) &&
          (!(x (s.axisSource dim keepdim o j) == x (s.axisSource dim keepdim o j)) ||
            decide (x (s.axisSource dim keepdim o j) > x (s.axisSource dim keepdim o best)))
      then j else best) ⟨0, h⟩ :=
  Rep.get_ofFn _ o

/-- An entry of `argmin` is the fold of its definition over the fiber. -/
theorem argmin_apply (x : Tensor α s) (dim : Fin s.rank) (keepdim : Bool) (h : 0 < s.extent dim)
    (o : (s.reduce [dim] keepdim).Coord) :
    argmin x dim keepdim h o = (List.finRange (s.extent dim)).foldl (fun best j =>
      if (x (s.axisSource dim keepdim o best) == x (s.axisSource dim keepdim o best)) &&
          (!(x (s.axisSource dim keepdim o j) == x (s.axisSource dim keepdim o j)) ||
            decide (x (s.axisSource dim keepdim o best) > x (s.axisSource dim keepdim o j)))
      then j else best) ⟨0, h⟩ :=
  Rep.get_ofFn _ o

/-! ## Extremality -/

/-- The entry of `x` at its `argmax` is a largest entry of the fiber for every real reading that
reflects the strict order of the fiber's entries, when each of them equals itself. -/
theorem le_argmax (x : Tensor α s) (dim : Fin s.rank) (keepdim : Bool) (h : 0 < s.extent dim)
    (o : (s.reduce [dim] keepdim).Coord) (observe : α → ℝ)
    (hself : ∀ j, (x (s.axisSource dim keepdim o j) == x (s.axisSource dim keepdim o j)) = true)
    (horder : ∀ i j, x (s.axisSource dim keepdim o i) > x (s.axisSource dim keepdim o j) ↔
      observe (x (s.axisSource dim keepdim o i)) > observe (x (s.axisSource dim keepdim o j)))
    (i : Fin (s.extent dim)) :
    observe (x (s.axisSource dim keepdim o i)) ≤
      observe (x (s.axisSource dim keepdim o (argmax x dim keepdim h o))) := by
  rw [argmax_apply]
  refine le_foldl_select
    (fun a b => (b == b) && (!(a == a) || decide (a > b)))
    (fun j => x (s.axisSource dim keepdim o j)) observe (fun i j hb => ?_) (fun i j hb => ?_)
    _ _ i (List.mem_cons_of_mem _ (List.mem_finRange i))
  · simp only [hself, Bool.not_true, Bool.true_and, Bool.false_or, decide_eq_true_eq] at hb
    exact ((horder i j).mp hb).le
  · simp only [hself, Bool.not_true, Bool.true_and, Bool.false_or,
      decide_eq_false_iff_not] at hb
    exact not_lt.mp fun hlt => hb ((horder i j).mpr hlt)

/-- The entry of `x` at its `argmin` is a smallest entry of the fiber for every real reading that
reflects the strict order of the fiber's entries, when each of them equals itself. -/
theorem argmin_le (x : Tensor α s) (dim : Fin s.rank) (keepdim : Bool) (h : 0 < s.extent dim)
    (o : (s.reduce [dim] keepdim).Coord) (observe : α → ℝ)
    (hself : ∀ j, (x (s.axisSource dim keepdim o j) == x (s.axisSource dim keepdim o j)) = true)
    (horder : ∀ i j, x (s.axisSource dim keepdim o i) > x (s.axisSource dim keepdim o j) ↔
      observe (x (s.axisSource dim keepdim o i)) > observe (x (s.axisSource dim keepdim o j)))
    (i : Fin (s.extent dim)) :
    observe (x (s.axisSource dim keepdim o (argmin x dim keepdim h o))) ≤
      observe (x (s.axisSource dim keepdim o i)) := by
  rw [argmin_apply]
  refine neg_le_neg_iff.mp (le_foldl_select
    (fun a b => (b == b) && (!(a == a) || decide (b > a)))
    (fun j => x (s.axisSource dim keepdim o j)) (fun a => -observe a)
    (fun i j hb => ?_) (fun i j hb => ?_) _ _ i (List.mem_cons_of_mem _ (List.mem_finRange i)))
  · simp only [hself, Bool.not_true, Bool.true_and, Bool.false_or, decide_eq_true_eq] at hb
    exact neg_le_neg ((horder j i).mp hb).le
  · simp only [hself, Bool.not_true, Bool.true_and, Bool.false_or,
      decide_eq_false_iff_not] at hb
    exact neg_le_neg (not_lt.mp fun hlt => hb ((horder j i).mpr hlt))

/-! ## Vectors -/

/-- In a reduction of a vector over its only axis, index `j` reads entry `j`. -/
theorem axisSource_vector {n : Nat} (j : Fin n) :
    Shape.axisSource [n] ⟨0, Nat.one_pos⟩ false PUnit.unit j = (j, PUnit.unit) := by
  apply Shape.Coord.ext
  intro d
  match d with
  | ⟨0, _⟩ => exact Shape.get_axisSource [n] ⟨0, Nat.one_pos⟩ false PUnit.unit j

/-- The `argmax` of a vector is the fold over its entries. -/
theorem item_argmax_vector {n : Nat} (x : Tensor α [n]) (h : 0 < n) :
    (argmax x ⟨0, Nat.one_pos⟩ false h).item = (List.finRange n).foldl (fun best j =>
      if (x.getScalar best == x.getScalar best) &&
          (!(x.getScalar j == x.getScalar j) || decide (x.getScalar j > x.getScalar best))
      then j else best) ⟨0, h⟩ := by
  refine (argmax_apply x ⟨0, Nat.one_pos⟩ false h PUnit.unit).trans ?_
  simp only [getScalar_eq_apply]
  rfl

/-- The `argmin` of a vector is the fold over its entries. -/
theorem item_argmin_vector {n : Nat} (x : Tensor α [n]) (h : 0 < n) :
    (argmin x ⟨0, Nat.one_pos⟩ false h).item = (List.finRange n).foldl (fun best j =>
      if (x.getScalar best == x.getScalar best) &&
          (!(x.getScalar j == x.getScalar j) || decide (x.getScalar best > x.getScalar j))
      then j else best) ⟨0, h⟩ := by
  refine (argmin_apply x ⟨0, Nat.one_pos⟩ false h PUnit.unit).trans ?_
  simp only [getScalar_eq_apply]
  rfl

/-- The entry of a vector at its `argmax` is a largest entry for every real reading that reflects
the strict order of the entries, when each of them equals itself. -/
theorem le_getScalar_argmax {n : Nat} (x : Tensor α [n]) (h : 0 < n) (observe : α → ℝ)
    (hself : ∀ j, (x.getScalar j == x.getScalar j) = true)
    (horder : ∀ i j,
      x.getScalar i > x.getScalar j ↔ observe (x.getScalar i) > observe (x.getScalar j))
    (i : Fin n) :
    observe (x.getScalar i) ≤ observe (x.getScalar (argmax x ⟨0, Nat.one_pos⟩ false h).item) := by
  simp only [getScalar_eq_apply] at hself horder ⊢
  exact (le_argmax x ⟨0, Nat.one_pos⟩ false h PUnit.unit observe hself horder i).trans_eq
    (congrArg observe (getScalar_eq_apply x (argmax x ⟨0, Nat.one_pos⟩ false h).item).symm)

/-- The entry of a vector at its `argmin` is a smallest entry for every real reading that
reflects the strict order of the entries, when each of them equals itself. -/
theorem getScalar_argmin_le {n : Nat} (x : Tensor α [n]) (h : 0 < n) (observe : α → ℝ)
    (hself : ∀ j, (x.getScalar j == x.getScalar j) = true)
    (horder : ∀ i j,
      x.getScalar i > x.getScalar j ↔ observe (x.getScalar i) > observe (x.getScalar j))
    (i : Fin n) :
    observe (x.getScalar (argmin x ⟨0, Nat.one_pos⟩ false h).item) ≤ observe (x.getScalar i) := by
  simp only [getScalar_eq_apply] at hself horder ⊢
  exact (congrArg observe (getScalar_eq_apply x (argmin x ⟨0, Nat.one_pos⟩ false h).item)).trans_le
    (argmin_le x ⟨0, Nat.one_pos⟩ false h PUnit.unit observe hself horder i)

end TorchLean.Tensor
