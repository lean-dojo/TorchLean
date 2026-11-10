/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.ArgMax

/-!
# Sorting along an axis

`sort x dim descending` returns the entries of every fiber along `dim` in sorted order together
with the index each one came from, as PyTorch's `sort(dim, descending, stable=True)`, and
`topk x dim k largest` is the first `k` entries of that sort, as PyTorch's `topk(k, dim, largest)`
with ties broken toward the lower index.

The order of a fiber is a stable merge sort (`List.mergeSort`) of its indices. In ascending order
the entries that equal themselves come first, increasing, and the entries that do not (NaN under
IEEE equality) come last. Descending order puts NaN first and then the decreasing entries, so NaN
counts as the largest value either way, as in PyTorch. Entries neither of which is placed before
the other, such as equal values, `-0` and `+0`, or two NaNs, keep their input order.

The stable order comes from the PyTorch 2.8 page of `torch.sort`, the NaN placement from
torch 2.8.0: the stable sort of `[1, NaN, 0, 3, -0, 1, NaN, -∞, 3]` has the indices
`[7, 2, 4, 0, 5, 3, 8, 1, 6]` ascending and `[1, 6, 3, 8, 0, 5, 2, 4, 7]` descending, and so does
`sort`. The lower-index ties of `topk` are our choice. The `torch.topk` page says the indices of
tied elements aren't guaranteed to be stable, so the exporter writes `topk` as a prefix of the
stable sort (`TorchLean.NN.PyTorch.Export.IR`). The capture test
`test_index_outputs_match_eager` in `scripts/checks/test_ir_capture.py` compares stable-sort
indices with eager PyTorch.

* `sort_snd_apply` and `sort_fst_apply` read the indices and the values, and `sortFast_eq`
  runs `sort` with each fiber's order computed once.
* `perm_sort_snd` shows the indices of a fiber are a permutation of it, and `sort_fst_le` and
  `le_sort_fst` show the values are monotone for a real reading that reflects the strict order of
  entries that equal themselves.
* `topk` is `slice` of the sort along `dim`, so its reads are those of `slice` (`topk_fst`,
  `topk_snd`). `Shape.slice_prefix` shows that the prefix slice replaces the extent of `dim` by
  `k`, and `slice_prefixCoord` reads it at `Shape.prefixCoord`, which takes an index `j` along
  `dim` and the indices of a coordinate `c` of `s` along the other axes.
* `sortBackward` and `topkBackward` write each incoming gradient entry at its source index in a
  tensor of zeros.
-/

@[expose] public section

namespace TorchLean.Shape

/-- The coordinate of the prefix slice `s.slice [dim] [0] [k] [1]` with index `j` along `dim` and
the indices of `c` along the other axes. -/
def prefixCoord (s : Shape) (dim : Fin s.rank) (k : Nat) (c : s.Coord) (j : Fin k) :
    (s.slice [dim] [0] [k] [1]).Coord :=
  Coord.ofFun fun d =>
    if hd : d.cast (s.rank_slice [dim] [0] [k] [1]) = dim then
      ⟨j.val, by rw [extent_slice]; simp [hd]⟩
    else
      ⟨(Coord.get c (d.cast (s.rank_slice [dim] [0] [k] [1]))).val,
        by rw [extent_slice]; simp [hd]⟩

/-- The prefix slice of `k` entries along `dim` replaces the extent of `dim` by `k`. -/
theorem slice_prefix (s : Shape) (dim : Fin s.rank) (k : Nat) :
    s.slice [dim] [0] [k] [1] = s.set dim k := by
  simpa using s.slice_singleton dim 0 k

end TorchLean.Shape

namespace TorchLean.Tensor

variable {α : Type} [Storage α] {s : Shape} [BEq α] [LT α]
  [DecidableRel ((· > ·) : α → α → Prop)]

/-- Sort `x` along `dim`, returning the sorted values and the index along `dim` each came from, as
PyTorch's stable `sort`. The indices of a fiber are `List.mergeSort` of its index list under the
relation "`j` is not placed before `i`", where ascending order places an entry that equals itself
before a NaN or before a strictly larger entry, and descending order places a NaN, or a strictly
larger entry, before an entry that equals itself. -/
def sort (x : Tensor α s) (dim : Fin s.rank) (descending : Bool) :
    Tensor α s × Tensor (Fin (s.extent dim)) s :=
  let indices : Tensor (Fin (s.extent dim)) s := Rep.ofFn fun c =>
    ((List.finRange (s.extent dim)).mergeSort fun i j =>
      !(bif descending then
          (x (Shape.Coord.update c dim i) == x (Shape.Coord.update c dim i)) &&
            (!(x (Shape.Coord.update c dim j) == x (Shape.Coord.update c dim j)) ||
              decide (x (Shape.Coord.update c dim j) > x (Shape.Coord.update c dim i)))
        else
          (x (Shape.Coord.update c dim j) == x (Shape.Coord.update c dim j)) &&
            (!(x (Shape.Coord.update c dim i) == x (Shape.Coord.update c dim i)) ||
              decide (x (Shape.Coord.update c dim i) > x (Shape.Coord.update c dim j)))))[
      (Shape.Coord.get c dim).val]'(by simp)
  (Rep.ofFn fun c => x (Shape.Coord.update c dim (indices c)), indices)

/-- An index of `sort` is the entry of the merge-sorted index list of its fiber at its own index
along `dim`. -/
theorem sort_snd_apply (x : Tensor α s) (dim : Fin s.rank) (descending : Bool) (c : s.Coord) :
    (sort x dim descending).2 c =
      ((List.finRange (s.extent dim)).mergeSort (fun i j =>
        !(bif descending then
            (x (Shape.Coord.update c dim i) == x (Shape.Coord.update c dim i)) &&
              (!(x (Shape.Coord.update c dim j) == x (Shape.Coord.update c dim j)) ||
                decide (x (Shape.Coord.update c dim j) > x (Shape.Coord.update c dim i)))
          else
            (x (Shape.Coord.update c dim j) == x (Shape.Coord.update c dim j)) &&
              (!(x (Shape.Coord.update c dim i) == x (Shape.Coord.update c dim i)) ||
                decide (x (Shape.Coord.update c dim i) > x (Shape.Coord.update c dim j))))))[
        (Shape.Coord.get c dim).val]'(by simp) :=
  Rep.get_ofFn _ c

/-- A value of `sort` is the input at its index. -/
theorem sort_fst_apply (x : Tensor α s) (dim : Fin s.rank) (descending : Bool) (c : s.Coord) :
    (sort x dim descending).1 c = x (Shape.Coord.update c dim ((sort x dim descending).2 c)) :=
  Rep.get_ofFn _ c

/-- `sort` with the merge-sorted index list of each fiber computed once and read at each entry. -/
def sortFast (x : Tensor α s) (dim : Fin s.rank) (descending : Bool) :
    Tensor α s × Tensor (Fin (s.extent dim)) s :=
  let order : Tensor (Array (Fin (s.extent dim))) (s.reduce [dim] false) :=
    Rep.ofFn fun o => ((List.finRange (s.extent dim)).mergeSort fun i j =>
      !(bif descending then
          (x (s.axisSource dim false o i) == x (s.axisSource dim false o i)) &&
            (!(x (s.axisSource dim false o j) == x (s.axisSource dim false o j)) ||
              decide (x (s.axisSource dim false o j) > x (s.axisSource dim false o i)))
        else
          (x (s.axisSource dim false o j) == x (s.axisSource dim false o j)) &&
            (!(x (s.axisSource dim false o i) == x (s.axisSource dim false o i)) ||
              decide (x (s.axisSource dim false o i) > x (s.axisSource dim false o j))))).toArray
  let indices : Tensor (Fin (s.extent dim)) s := Rep.ofFn fun c =>
    (order (s.axisTarget dim c))[(Shape.Coord.get c dim).val]?.getD (Shape.Coord.get c dim)
  (Rep.ofFn fun c => x (Shape.Coord.update c dim (indices c)), indices)

/-- The indices of `sortFast` are those of `sort`. -/
private theorem sortFast_snd (x : Tensor α s) (dim : Fin s.rank) (descending : Bool) :
    (sortFast x dim descending).2 = (sort x dim descending).2 := by
  apply Rep.ext
  intro c
  rw [sort_snd_apply]
  simp only [sortFast, Rep.get_ofFn, Shape.axisSource_axisTarget]
  rw [List.getElem?_toArray, List.getElem?_eq_getElem (by simp), Option.getD_some]

/-- Compiled code runs `sort` as `sortFast`, which computes each fiber's order once. -/
@[csimp] theorem sortFast_eq : @TorchLean.Tensor.sort = @sortFast := by
  funext α _ s _ _ _ x dim descending
  refine Prod.ext ?_ (sortFast_snd x dim descending).symm
  apply Rep.ext
  intro c
  rw [sort_fst_apply, ← sortFast_snd]
  exact (Rep.get_ofFn (fun c =>
    x (Shape.Coord.update c dim ((sortFast x dim descending).2 c))) c).symm

/-- Along the fiber through `c`, the indices of `sort` are the merge-sorted index list. -/
private theorem map_sort_snd (x : Tensor α s) (dim : Fin s.rank) (descending : Bool)
    (c : s.Coord) :
    (List.finRange (s.extent dim)).map
        (fun j => (sort x dim descending).2 (Shape.Coord.update c dim j)) =
      (List.finRange (s.extent dim)).mergeSort (fun i j =>
        !(bif descending then
            (x (Shape.Coord.update c dim i) == x (Shape.Coord.update c dim i)) &&
              (!(x (Shape.Coord.update c dim j) == x (Shape.Coord.update c dim j)) ||
                decide (x (Shape.Coord.update c dim j) > x (Shape.Coord.update c dim i)))
          else
            (x (Shape.Coord.update c dim j) == x (Shape.Coord.update c dim j)) &&
              (!(x (Shape.Coord.update c dim i) == x (Shape.Coord.update c dim i)) ||
                decide (x (Shape.Coord.update c dim i) > x (Shape.Coord.update c dim j))))) := by
  apply List.ext_getElem
  · simp
  · intro p h₁ h₂
    rw [List.getElem_map, sort_snd_apply]
    simp only [Shape.Coord.update_update, Shape.Coord.get_update_self, List.getElem_finRange]
    rfl

/-- The indices of `sort` along a fiber are a permutation of the fiber's indices. -/
theorem perm_sort_snd (x : Tensor α s) (dim : Fin s.rank) (descending : Bool) (c : s.Coord) :
    ((List.finRange (s.extent dim)).map
        fun j => (sort x dim descending).2 (Shape.Coord.update c dim j)).Perm
      (List.finRange (s.extent dim)) := by
  rw [map_sort_snd]
  exact List.mergeSort_perm _ _

/-- With every fiber entry equal to itself and an order-reflecting real reading, the merge-sorted
index list of a fiber is pairwise ordered by the reading of its entries: increasing for ascending
order, decreasing for descending order. -/
private theorem pairwise_sort (x : Tensor α s) (dim : Fin s.rank) (descending : Bool)
    (c : s.Coord) (observe : α → ℝ)
    (hself : ∀ j, (x (Shape.Coord.update c dim j) == x (Shape.Coord.update c dim j)) = true)
    (horder : ∀ i j, x (Shape.Coord.update c dim i) > x (Shape.Coord.update c dim j) ↔
      observe (x (Shape.Coord.update c dim i)) > observe (x (Shape.Coord.update c dim j))) :
    ((List.finRange (s.extent dim)).mergeSort (fun i j =>
        !(bif descending then
            (x (Shape.Coord.update c dim i) == x (Shape.Coord.update c dim i)) &&
              (!(x (Shape.Coord.update c dim j) == x (Shape.Coord.update c dim j)) ||
                decide (x (Shape.Coord.update c dim j) > x (Shape.Coord.update c dim i)))
          else
            (x (Shape.Coord.update c dim j) == x (Shape.Coord.update c dim j)) &&
              (!(x (Shape.Coord.update c dim i) == x (Shape.Coord.update c dim i)) ||
                decide (x (Shape.Coord.update c dim i) > x (Shape.Coord.update c dim j))))
        )).Pairwise
      fun i j => bif descending then
          observe (x (Shape.Coord.update c dim j)) ≤ observe (x (Shape.Coord.update c dim i))
        else
          observe (x (Shape.Coord.update c dim i)) ≤ observe (x (Shape.Coord.update c dim j)) := by
  have hle : ∀ i j, (fun i j =>
      !(bif descending then
          (x (Shape.Coord.update c dim i) == x (Shape.Coord.update c dim i)) &&
            (!(x (Shape.Coord.update c dim j) == x (Shape.Coord.update c dim j)) ||
              decide (x (Shape.Coord.update c dim j) > x (Shape.Coord.update c dim i)))
        else
          (x (Shape.Coord.update c dim j) == x (Shape.Coord.update c dim j)) &&
            (!(x (Shape.Coord.update c dim i) == x (Shape.Coord.update c dim i)) ||
              decide (x (Shape.Coord.update c dim i) > x (Shape.Coord.update c dim j)))))
        i j = true ↔
      bif descending then
          observe (x (Shape.Coord.update c dim j)) ≤ observe (x (Shape.Coord.update c dim i))
        else
          observe (x (Shape.Coord.update c dim i)) ≤ observe (x (Shape.Coord.update c dim j)) := by
    intro i j
    cases descending
    · simp only [hself, Bool.not_true, Bool.false_or, Bool.true_and, Bool.cond_false,
        Bool.not_eq_true', decide_eq_false_iff_not]
      exact (not_congr (horder i j)).trans not_lt
    · simp only [hself, Bool.not_true, Bool.false_or, Bool.true_and, Bool.cond_true,
        Bool.not_eq_true', decide_eq_false_iff_not]
      exact (not_congr (horder j i)).trans not_lt
  refine (List.pairwise_mergeSort (fun a b c hab hbc => ?_) (fun a b => ?_) _).imp
    fun {i j} h => (hle i j).mp h
  · rw [hle] at hab hbc ⊢
    cases descending
    · exact le_trans hab hbc
    · exact le_trans hbc hab
  · rcases le_total (observe (x (Shape.Coord.update c dim a)))
      (observe (x (Shape.Coord.update c dim b))) with hab | hba
    · cases descending
      · exact Bool.or_eq_true_iff.mpr (Or.inl ((hle a b).mpr hab))
      · exact Bool.or_eq_true_iff.mpr (Or.inr ((hle b a).mpr hab))
    · cases descending
      · exact Bool.or_eq_true_iff.mpr (Or.inr ((hle b a).mpr hba))
      · exact Bool.or_eq_true_iff.mpr (Or.inl ((hle a b).mpr hba))

/-- The value of `sort` at position `p` of the fiber through `c` is the input at entry `p` of the
fiber's merge-sorted index list. -/
private theorem sort_fst_update (x : Tensor α s) (dim : Fin s.rank) (descending : Bool)
    (c : s.Coord) (p : Fin (s.extent dim)) :
    (sort x dim descending).1 (Shape.Coord.update c dim p) =
      x (Shape.Coord.update c dim
        (((List.finRange (s.extent dim)).mergeSort (fun i j =>
        !(bif descending then
            (x (Shape.Coord.update c dim i) == x (Shape.Coord.update c dim i)) &&
              (!(x (Shape.Coord.update c dim j) == x (Shape.Coord.update c dim j)) ||
                decide (x (Shape.Coord.update c dim j) > x (Shape.Coord.update c dim i)))
          else
            (x (Shape.Coord.update c dim j) == x (Shape.Coord.update c dim j)) &&
              (!(x (Shape.Coord.update c dim i) == x (Shape.Coord.update c dim i)) ||
                decide (x (Shape.Coord.update c dim i) > x (Shape.Coord.update c dim j))))
        ))[p.val]'(by simp))) := by
  rw [sort_fst_apply, Shape.Coord.update_update]
  have hp := List.getElem_of_eq (map_sort_snd x dim descending c) (by simp : p.val < _)
  rw [List.getElem_map, List.getElem_finRange] at hp
  exact congrArg (fun j => x (Shape.Coord.update c dim j)) hp

/-- The ascending values of `sort` increase along `dim` for every real reading that reflects the
strict order of the fiber's entries, when each of them equals itself. -/
theorem sort_fst_le (x : Tensor α s) (dim : Fin s.rank) (c : s.Coord) (observe : α → ℝ)
    (hself : ∀ j, (x (Shape.Coord.update c dim j) == x (Shape.Coord.update c dim j)) = true)
    (horder : ∀ i j, x (Shape.Coord.update c dim i) > x (Shape.Coord.update c dim j) ↔
      observe (x (Shape.Coord.update c dim i)) > observe (x (Shape.Coord.update c dim j)))
    {p q : Fin (s.extent dim)} (hpq : p ≤ q) :
    observe ((sort x dim false).1 (Shape.Coord.update c dim p)) ≤
      observe ((sort x dim false).1 (Shape.Coord.update c dim q)) := by
  rcases hpq.lt_or_eq with hpq | rfl
  · rw [sort_fst_update, sort_fst_update]
    simpa using List.pairwise_iff_getElem.mp (pairwise_sort x dim false c observe hself horder)
      p.val q.val (by simp) (by simp) hpq
  · exact le_rfl

/-- The descending values of `sort` decrease along `dim` for every real reading that reflects the
strict order of the fiber's entries, when each of them equals itself. -/
theorem le_sort_fst (x : Tensor α s) (dim : Fin s.rank) (c : s.Coord) (observe : α → ℝ)
    (hself : ∀ j, (x (Shape.Coord.update c dim j) == x (Shape.Coord.update c dim j)) = true)
    (horder : ∀ i j, x (Shape.Coord.update c dim i) > x (Shape.Coord.update c dim j) ↔
      observe (x (Shape.Coord.update c dim i)) > observe (x (Shape.Coord.update c dim j)))
    {p q : Fin (s.extent dim)} (hpq : p ≤ q) :
    observe ((sort x dim true).1 (Shape.Coord.update c dim q)) ≤
      observe ((sort x dim true).1 (Shape.Coord.update c dim p)) := by
  rcases hpq.lt_or_eq with hpq | rfl
  · rw [sort_fst_update, sort_fst_update]
    simpa using List.pairwise_iff_getElem.mp (pairwise_sort x dim true c observe hself horder)
      p.val q.val (by simp) (by simp) hpq
  · exact le_rfl

/-! ## Top-k -/

/-- The `k` largest entries of every fiber along `dim` with their indices, or with `largest = false`
the `k` smallest, as PyTorch's `topk(k, dim, largest, sorted=True)`: the first `k` entries of
`sort x dim largest` along `dim`. Ties go to the lower index and NaN counts as the largest value. -/
def topk (x : Tensor α s) (dim : Fin s.rank) (k : Nat) (largest : Bool) (h : k ≤ s.extent dim) :
    Tensor α (s.slice [dim] [0] [k] [1]) ×
      Tensor (Fin (s.extent dim)) (s.slice [dim] [0] [k] [1]) :=
  (slice (sort x dim largest).1 [dim] [0] [k] [1]
      (by simpa using Shape.sliceValid_singleton s dim 0 k (by simpa using h)),
    slice (sort x dim largest).2 [dim] [0] [k] [1]
      (by simpa using Shape.sliceValid_singleton s dim 0 k (by simpa using h)))

/-- The values of `topk` are the prefix of the values of `sort`. -/
theorem topk_fst (x : Tensor α s) (dim : Fin s.rank) (k : Nat) (largest : Bool)
    (h : k ≤ s.extent dim) (hv : Shape.SliceValid s [dim] [0] [k] [1]) :
    (topk x dim k largest h).1 = slice (sort x dim largest).1 [dim] [0] [k] [1] hv :=
  rfl

/-- The indices of `topk` are the prefix of the indices of `sort`. -/
theorem topk_snd (x : Tensor α s) (dim : Fin s.rank) (k : Nat) (largest : Bool)
    (h : k ≤ s.extent dim) (hv : Shape.SliceValid s [dim] [0] [k] [1]) :
    (topk x dim k largest h).2 = slice (sort x dim largest).2 [dim] [0] [k] [1] hv :=
  rfl

/-- An entry of the values of `topk` is the entry of the sorted values that its prefix slice
reads. -/
theorem topk_fst_apply (x : Tensor α s) (dim : Fin s.rank) (k : Nat) (largest : Bool)
    (h : k ≤ s.extent dim) (c : (s.slice [dim] [0] [k] [1]).Coord) :
    (topk x dim k largest h).1 c = slice (sort x dim largest).1 [dim] [0] [k] [1]
      (by simpa using Shape.sliceValid_singleton s dim 0 k (by simpa using h)) c :=
  rfl

/-- An entry of the indices of `topk` is the entry of the sorted indices that its prefix slice
reads. -/
theorem topk_snd_apply (x : Tensor α s) (dim : Fin s.rank) (k : Nat) (largest : Bool)
    (h : k ≤ s.extent dim) (c : (s.slice [dim] [0] [k] [1]).Coord) :
    (topk x dim k largest h).2 c = slice (sort x dim largest).2 [dim] [0] [k] [1]
      (by simpa using Shape.sliceValid_singleton s dim 0 k (by simpa using h)) c :=
  rfl

/-- At `Shape.prefixCoord s dim k c j`, the prefix slice of `y` reads `y` at `c` with index `j`
along `dim`. -/
theorem slice_prefixCoord {β : Type} [Storage β] (y : Tensor β s) (dim : Fin s.rank) (k : Nat)
    (h : k ≤ s.extent dim) (hv : Shape.SliceValid s [dim] [0] [k] [1]) (c : s.Coord)
    (j : Fin k) :
    slice y [dim] [0] [k] [1] hv (Shape.prefixCoord s dim k c j) =
      y (Shape.Coord.update c dim (Fin.castLE h j)) := by
  rw [slice_apply]
  congr 1
  apply Shape.Coord.ext_val
  intro d
  by_cases hd : d = dim
  · subst hd
    simp [Shape.prefixCoord]
  · rw [Shape.Coord.get_update_of_ne _ _ hd]
    simp [Shape.prefixCoord, hd]

/-! ## Vector-Jacobian products -/

/-- The vector-Jacobian product of the values of `sort`: each incoming gradient entry is written at
the source index of its value. The indices of a fiber are a permutation of it (`perm_sort_snd`), so
every index receives exactly one entry. -/
def sortBackward [Zero α] (x g : Tensor α s) (dim : Fin s.rank) (descending : Bool) :
    Tensor α s :=
  Rep.ofFn fun c => (List.finRange (s.extent dim)).foldl (fun acc j =>
    if (sort x dim descending).2 (Shape.Coord.update c dim j) = Shape.Coord.get c dim then
      g (Shape.Coord.update c dim j)
    else acc) 0

/-- The vector-Jacobian product of the values of `topk`: each incoming gradient entry is written at
the source index of its value, and every other entry is zero. -/
def topkBackward [Zero α] (x : Tensor α s) (dim : Fin s.rank) (k : Nat) (largest : Bool)
    (h : k ≤ s.extent dim) (g : Tensor α (s.slice [dim] [0] [k] [1])) : Tensor α s :=
  Rep.ofFn fun c => (List.finRange k).foldl (fun acc j =>
    if (topk x dim k largest h).2 (Shape.prefixCoord s dim k c j) = Shape.Coord.get c dim then
      g (Shape.prefixCoord s dim k c j)
    else acc) 0

/-- An entry of `sortBackward` is the fold of its definition. -/
theorem sortBackward_apply [Zero α] (x g : Tensor α s) (dim : Fin s.rank) (descending : Bool)
    (c : s.Coord) :
    sortBackward x g dim descending c = (List.finRange (s.extent dim)).foldl (fun acc j =>
      if (sort x dim descending).2 (Shape.Coord.update c dim j) = Shape.Coord.get c dim then
        g (Shape.Coord.update c dim j)
      else acc) 0 :=
  Rep.get_ofFn _ c

/-- An entry of `topkBackward` is the fold of its definition. -/
theorem topkBackward_apply [Zero α] (x : Tensor α s) (dim : Fin s.rank) (k : Nat)
    (largest : Bool) (h : k ≤ s.extent dim) (g : Tensor α (s.slice [dim] [0] [k] [1]))
    (c : s.Coord) :
    topkBackward x dim k largest h g c = (List.finRange k).foldl (fun acc j =>
      if (topk x dim k largest h).2 (Shape.prefixCoord s dim k c j) = Shape.Coord.get c dim then
        g (Shape.prefixCoord s dim k c j)
      else acc) 0 :=
  Rep.get_ofFn _ c


end TorchLean.Tensor
