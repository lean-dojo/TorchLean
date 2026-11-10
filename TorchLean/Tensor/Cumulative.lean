/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.ArgMax

/-!
# Cumulative reductions along an axis

`cumsum x dim`, `cumprod x dim`, `cummax x dim` and `cummin x dim` are PyTorch's functions of the
same names at any axis of any shape: the entry at a coordinate reduces the entries of its fiber
along `dim` up to and including its own index.

* `cumsum` and `cumprod` fold the prefix in index order from `0` and from `1`, the seeds of
  `reduce .sum` and `reduce .prod`, so their last entries are those reductions
  (`cumsum_last`, `cumprod_last`). The first entry of `cumsum` is `0 + x₀`, which differs from
  `x₀` only in the sign of a zero.
* `cummax` and `cummin` return values and indices. The index folds over the prefix from index
  `0` and moves to an entry that is not equal to itself (NaN), or, when the current choice equals
  itself, to an entry that the choice does not strictly exceed (for `cummin`, does not strictly
  undercut). So a NaN propagates with the index of the latest NaN and ties take the latest index,
  as PyTorch's `cummax` and `cummin` do, and the value is the input at that index. `-0` and `0`
  tie, so `cummax` of `[0, -0]` ends at `-0`, where `reduce .max` gives `0`.
* `cumsumBackward`, `cumprodBackward`, `cummaxBackward` and `cumminBackward` are their
  vector-Jacobian products: the reversed cumulative sum, the sums of exclusive products (exact
  derivatives, with no condition on zeros), and the incoming gradient added at the source index
  of each value, in index order from zero. `cumprodJVP` is the Jacobian-vector product of
  `cumprod`.
* The `_apply` lemmas read every entry. `le_cummax` and `cummin_le` state the extremal property
  for real readings that reflect the order of entries that equal themselves.
* `cumsumFast_eq`, `cumprodFast_eq`, `cummaxFast_eq` and
  `cumminFast_eq` run the cumulative operators with one scan of each fiber
  (`prefixFolds`, `selectionScans`).

Like the reductions, the scans are folds written by hand. The PyTorch 2.8 pages of `torch.cummax`
and `torch.cummin` give the running extremum and its index but no rule for ties or NaN, so we took
the rule from torch 2.8.0 itself. Its `cummax` of `[1, 3, 3, 2, 3]` has indices `[0, 1, 2, 2, 4]`,
and of `[1, NaN, 2, NaN, 0]` indices `[0, 1, 1, 3, 3]`. PyTorch seeds `cumsum` with zero too, so
`[-0, -0]` gives `[0, 0]` in both. The capture test `test_index_outputs_match_eager` in
`scripts/checks/test_ir_capture.py` compares the indices of `cummax` and `cummin` with eager
PyTorch on rows of repeats, signed zeros and NaNs.
-/

@[expose] public section

namespace TorchLean.Tensor

variable {α : Type} [Storage α] {s : Shape}

/-! ## Sums and products -/

/-- Cumulative sum along `dim`, as PyTorch's `cumsum(dim)`: the entry at `c` adds the entries of
its fiber with index at most that of `c`, in index order, to zero. -/
def cumsum [Add α] [Zero α] (x : Tensor α s) (dim : Fin s.rank) : Tensor α s :=
  Rep.ofFn fun c => Tensor.foldl (· + ·) 0
    (Tensor.ofFn fun i : Fin ((Shape.Coord.get c dim).val + 1) =>
      x (Shape.Coord.update c dim (Fin.castLE (Shape.Coord.get c dim).isLt i)))

/-- Cumulative product along `dim`, as PyTorch's `cumprod(dim)`: the entry at `c` multiplies the
entries of its fiber with index at most that of `c`, in index order, into one. -/
def cumprod [Mul α] [One α] (x : Tensor α s) (dim : Fin s.rank) : Tensor α s :=
  Rep.ofFn fun c => Tensor.foldl (· * ·) 1
    (Tensor.ofFn fun i : Fin ((Shape.Coord.get c dim).val + 1) =>
      x (Shape.Coord.update c dim (Fin.castLE (Shape.Coord.get c dim).isLt i)))

/-- An entry of `cumsum` is the sum of its prefix. -/
theorem cumsum_apply [Add α] [Zero α] (x : Tensor α s) (dim : Fin s.rank) (c : s.Coord) :
    cumsum x dim c = Tensor.foldl (· + ·) 0
      (Tensor.ofFn fun i : Fin ((Shape.Coord.get c dim).val + 1) =>
        x (Shape.Coord.update c dim (Fin.castLE (Shape.Coord.get c dim).isLt i))) :=
  Rep.get_ofFn _ c

/-- An entry of `cumprod` is the product of its prefix. -/
theorem cumprod_apply [Mul α] [One α] (x : Tensor α s) (dim : Fin s.rank) (c : s.Coord) :
    cumprod x dim c = Tensor.foldl (· * ·) 1
      (Tensor.ofFn fun i : Fin ((Shape.Coord.get c dim).val + 1) =>
        x (Shape.Coord.update c dim (Fin.castLE (Shape.Coord.get c dim).isLt i))) :=
  Rep.get_ofFn _ c

/-- A left fold over the prefix of the last index along `dim` is the fold over the whole fiber that
a reduction over `dim` reads. -/
private theorem foldl_prefix_last (step : α → α → α) (seed : α) (x : Tensor α s)
    (dim : Fin s.rank) (h : 0 < s.extent dim) (o : (s.reduce [dim] false).Coord) :
    let c := s.axisSource dim false o ⟨s.extent dim - 1, by omega⟩
    Tensor.foldl step seed
        (Tensor.ofFn fun i : Fin ((Shape.Coord.get c dim).val + 1) =>
          x (Shape.Coord.update c dim (Fin.castLE (Shape.Coord.get c dim).isLt i))) =
      (List.finRange (Shape.size (s.selectAxes [dim]))).foldl
        (fun acc i => step acc (x (s.reduceSource [dim] false o (Shape.Coord.unlinearize i))))
        seed := by
  intro c
  rw [Rep.foldl_eq_fin_foldl, Fin.foldl_eq_finRange_foldl]
  simp only [Rep.getFlat_eq_get, Tensor.ofFn, Rep.get_ofFn]
  refine foldl_finRange_congr ?_
    (fun acc (r : Shape.Coord [(Shape.Coord.get c dim).val + 1]) =>
      step acc (x (Shape.Coord.update c dim (Fin.castLE (Shape.Coord.get c dim).isLt r.1))))
    (fun acc r => step acc (x (s.reduceSource [dim] false o r))) ?_ seed
  · rw [Shape.get_axisSource, Shape.selectAxes_singleton]
    congr 1
    show s.extent dim - 1 + 1 = s.extent dim
    omega
  · intro acc r r' hrr
    show step acc (x _) = step acc (x _)
    rw [Shape.update_axisSource, Shape.axisSource]
    congr 3
    apply Shape.Coord.ext_index
    intro k _
    rw [Shape.Coord.index_reindexEq, ← hrr k]
    cases k with
    | zero => simp
    | succ k => simp [Shape.Coord.index_succ]

/-- The entry of `cumsum` at the last index along `dim` is the sum over that axis. -/
theorem cumsum_last [Add α] [Zero α] (x : Tensor α s) (dim : Fin s.rank) (h : 0 < s.extent dim)
    (o : (s.reduce [dim] false).Coord) :
    cumsum x dim (s.axisSource dim false o ⟨s.extent dim - 1, by omega⟩) =
      reduce .sum x [dim] false (by simp) o := by
  rw [cumsum_apply, reduce_sum_apply]
  exact foldl_prefix_last (· + ·) 0 x dim h o

/-- The entry of `cumprod` at the last index along `dim` is the product over that axis. -/
theorem cumprod_last [Mul α] [One α] (x : Tensor α s) (dim : Fin s.rank) (h : 0 < s.extent dim)
    (o : (s.reduce [dim] false).Coord) :
    cumprod x dim (s.axisSource dim false o ⟨s.extent dim - 1, by omega⟩) =
      reduce .prod x [dim] false (by simp) o := by
  rw [cumprod_apply, reduce_prod_apply]
  exact foldl_prefix_last (· * ·) 1 x dim h o

/-- A left fold over `Tensor.ofFn f` is the fold of `f` over `List.finRange n`. -/
private theorem foldl_ofFn {β γ : Type} [Storage β] {n : Nat} (step : γ → β → γ) (init : γ)
    (f : Fin n → β) :
    Tensor.foldl step init (Tensor.ofFn f) =
      (List.finRange n).foldl (fun acc i => step acc (f i)) init := by
  rw [Tensor.ofFn, Rep.ofFn_eq_map_coordinates, Rep.foldl_map,
    ← Fin.foldl_eq_foldl_finRange, fin_foldl_eq_coordinates]

/-- The prefix folds of every fiber along `dim`, each computed by one scan. -/
def prefixFolds (step : α → α → α) (seed : α) (x : Tensor α s) (dim : Fin s.rank) :
    Tensor (Array α) (s.reduce [dim] false) :=
  Rep.ofFn fun o =>
    (((List.finRange (s.extent dim)).map fun j => x (s.axisSource dim false o j)).scanl step
      seed).toArray

/-- The first `k + 1` entries of `List.finRange n` are `List.finRange (k + 1)` cast up. -/
private theorem take_finRange_succ {n : Nat} (k : Fin n) :
    (List.finRange n).take (k.val + 1) = (List.finRange (k.val + 1)).map (Fin.castLE k.isLt) := by
  apply List.ext_getElem
  · simp only [List.length_take, List.length_finRange, List.length_map]
    exact Nat.min_eq_left k.isLt
  · intro i h₁ h₂
    simp [List.getElem_take, List.getElem_finRange]

/-- The prefix fold through `c` along `dim` is the scan entry of its fiber after its index. -/
private theorem foldl_prefix_eq_prefixFolds (step : α → α → α) (seed : α) (x : Tensor α s)
    (dim : Fin s.rank) (c : s.Coord) :
    Tensor.foldl step seed
        (Tensor.ofFn fun i : Fin ((Shape.Coord.get c dim).val + 1) =>
          x (Shape.Coord.update c dim (Fin.castLE (Shape.Coord.get c dim).isLt i))) =
      (prefixFolds step seed x dim (s.axisTarget dim c))[
        (Shape.Coord.get c dim).val + 1]?.getD seed := by
  have htake := take_finRange_succ (Shape.Coord.get c dim)
  rw [foldl_ofFn, prefixFolds, Rep.get_ofFn, List.getElem?_toArray,
    List.getElem?_scanl, ite_eq_left (by simp),
    Option.getD_some, ← List.map_take, htake, List.foldl_map, List.foldl_map]
  simp only [Shape.axisSource_axisTarget]

/-- `cumsum` reading the prefix sums of each fiber from one scan of it. -/
def cumsumFast [Add α] [Zero α] (x : Tensor α s) (dim : Fin s.rank) : Tensor α s :=
  let sums := Tensor.prefixFolds (· + ·) 0 x dim
  Rep.ofFn fun c => (sums (s.axisTarget dim c))[(Shape.Coord.get c dim).val + 1]?.getD 0

/-- `cumprod` reading the prefix products of each fiber from one scan of it. -/
def cumprodFast [Mul α] [One α] (x : Tensor α s) (dim : Fin s.rank) : Tensor α s :=
  let products := Tensor.prefixFolds (· * ·) 1 x dim
  Rep.ofFn fun c =>
    (products (s.axisTarget dim c))[(Shape.Coord.get c dim).val + 1]?.getD 1

/-- `cumsum` runs as `cumsumFast`, which scans each fiber once. -/
@[csimp] theorem cumsumFast_eq : @cumsum = @cumsumFast := by
  funext α _ s _ _ x dim
  apply Rep.ext
  intro c
  rw [cumsum_apply, foldl_prefix_eq_prefixFolds]
  exact (Rep.get_ofFn (fun c => (prefixFolds (· + ·) 0 x dim
    (s.axisTarget dim c))[(Shape.Coord.get c dim).val + 1]?.getD 0) c).symm

/-- `cumprod` runs as `cumprodFast`, which scans each fiber once. -/
@[csimp] theorem cumprodFast_eq : @cumprod = @cumprodFast := by
  funext α _ s _ _ x dim
  apply Rep.ext
  intro c
  rw [cumprod_apply, foldl_prefix_eq_prefixFolds]
  exact (Rep.get_ofFn (fun c => (prefixFolds (· * ·) 1 x dim
    (s.axisTarget dim c))[(Shape.Coord.get c dim).val + 1]?.getD 1) c).symm

/-- The prefix folds of a running selection through every fiber along `dim`, each computed by one
scan: from index `0`, the choice moves to the next index `j` when `move` holds of the entries at
the current choice and at `j`. -/
def selectionScans (move : α → α → Bool) (x : Tensor α s) (dim : Fin s.rank) :
    Tensor (Array (Fin (s.extent dim))) (s.reduce [dim] false) :=
  Rep.ofFn fun o =>
    if h : 0 < s.extent dim then
      ((List.finRange (s.extent dim)).scanl (fun best j =>
        if move (x (s.axisSource dim false o best)) (x (s.axisSource dim false o j)) then j
        else best) ⟨0, h⟩).toArray
    else #[]

/-- The scan entry of a fiber after the index of `c` is the selection fold over the prefix
through `c`. -/
private theorem selectionScans_read (move : α → α → Bool) (x : Tensor α s) (dim : Fin s.rank)
    (c : s.Coord) :
    (selectionScans move x dim (s.axisTarget dim c))[
        (Shape.Coord.get c dim).val + 1]?.getD (Shape.Coord.get c dim) =
      ((List.finRange ((Shape.Coord.get c dim).val + 1)).map
        (Fin.castLE (Shape.Coord.get c dim).isLt)).foldl (fun best j =>
          if move (x (Shape.Coord.update c dim best)) (x (Shape.Coord.update c dim j)) then j
          else best) ⟨0, Nat.zero_lt_of_lt (Shape.Coord.get c dim).isLt⟩ := by
  have htake := take_finRange_succ (Shape.Coord.get c dim)
  simp only [selectionScans, Rep.get_ofFn,
    dite_eq_left (Nat.zero_lt_of_lt (Shape.Coord.get c dim).isLt), List.getElem?_toArray,
    List.getElem?_scanl, Shape.axisSource_axisTarget]
  rw [ite_eq_left (by simp), Option.getD_some, htake]

/-! ## Running extrema -/

section Extrema

variable [BEq α] [LT α] [DecidableRel ((· > ·) : α → α → Prop)]

/-- Running maxima along `dim` with their indices, as PyTorch's `cummax(dim)`. The index at `c`
folds over the prefix of its fiber in index order from `0`, moving to an entry that is not equal to
itself (NaN), or, when the current choice equals itself, to an entry that the choice does not
strictly exceed: a NaN propagates with the index of the latest NaN and ties, `-0` and `0` among
them, take the latest index. The value is the input at that index. -/
def cummax (x : Tensor α s) (dim : Fin s.rank) : Tensor α s × Tensor (Fin (s.extent dim)) s :=
  let indices : Tensor (Fin (s.extent dim)) s := Rep.ofFn fun c =>
    ((List.finRange ((Shape.Coord.get c dim).val + 1)).map
      (Fin.castLE (Shape.Coord.get c dim).isLt)).foldl (fun best j =>
        if !(x (Shape.Coord.update c dim j) == x (Shape.Coord.update c dim j)) ||
            ((x (Shape.Coord.update c dim best) == x (Shape.Coord.update c dim best)) &&
              !decide (x (Shape.Coord.update c dim best) > x (Shape.Coord.update c dim j)))
        then j else best) ⟨0, Nat.zero_lt_of_lt (Shape.Coord.get c dim).isLt⟩
  (Rep.ofFn fun c => x (Shape.Coord.update c dim (indices c)), indices)

/-- Running minima along `dim` with their indices, as PyTorch's `cummin(dim)`. The index at `c`
folds over the prefix of its fiber in index order from `0`, moving to an entry that is not equal to
itself (NaN), or, when the current choice equals itself, to an entry that does not strictly exceed
the choice: a NaN propagates with the index of the latest NaN and ties, `-0` and `0` among them,
take the latest index. The value is the input at that index. -/
def cummin (x : Tensor α s) (dim : Fin s.rank) : Tensor α s × Tensor (Fin (s.extent dim)) s :=
  let indices : Tensor (Fin (s.extent dim)) s := Rep.ofFn fun c =>
    ((List.finRange ((Shape.Coord.get c dim).val + 1)).map
      (Fin.castLE (Shape.Coord.get c dim).isLt)).foldl (fun best j =>
        if !(x (Shape.Coord.update c dim j) == x (Shape.Coord.update c dim j)) ||
            ((x (Shape.Coord.update c dim best) == x (Shape.Coord.update c dim best)) &&
              !decide (x (Shape.Coord.update c dim j) > x (Shape.Coord.update c dim best)))
        then j else best) ⟨0, Nat.zero_lt_of_lt (Shape.Coord.get c dim).isLt⟩
  (Rep.ofFn fun c => x (Shape.Coord.update c dim (indices c)), indices)

/-- An index of `cummax` is the fold of its definition over the prefix. -/
theorem cummax_snd_apply (x : Tensor α s) (dim : Fin s.rank) (c : s.Coord) :
    (cummax x dim).2 c =
      ((List.finRange ((Shape.Coord.get c dim).val + 1)).map
        (Fin.castLE (Shape.Coord.get c dim).isLt)).foldl (fun best j =>
          if !(x (Shape.Coord.update c dim j) == x (Shape.Coord.update c dim j)) ||
              ((x (Shape.Coord.update c dim best) == x (Shape.Coord.update c dim best)) &&
                !decide (x (Shape.Coord.update c dim best) > x (Shape.Coord.update c dim j)))
          then j else best) ⟨0, Nat.zero_lt_of_lt (Shape.Coord.get c dim).isLt⟩ :=
  Rep.get_ofFn _ c

/-- An index of `cummin` is the fold of its definition over the prefix. -/
theorem cummin_snd_apply (x : Tensor α s) (dim : Fin s.rank) (c : s.Coord) :
    (cummin x dim).2 c =
      ((List.finRange ((Shape.Coord.get c dim).val + 1)).map
        (Fin.castLE (Shape.Coord.get c dim).isLt)).foldl (fun best j =>
          if !(x (Shape.Coord.update c dim j) == x (Shape.Coord.update c dim j)) ||
              ((x (Shape.Coord.update c dim best) == x (Shape.Coord.update c dim best)) &&
                !decide (x (Shape.Coord.update c dim j) > x (Shape.Coord.update c dim best)))
          then j else best) ⟨0, Nat.zero_lt_of_lt (Shape.Coord.get c dim).isLt⟩ :=
  Rep.get_ofFn _ c

/-- A value of `cummax` is the input at its index. -/
theorem cummax_fst_apply (x : Tensor α s) (dim : Fin s.rank) (c : s.Coord) :
    (cummax x dim).1 c = x (Shape.Coord.update c dim ((cummax x dim).2 c)) :=
  Rep.get_ofFn _ c

/-- A value of `cummin` is the input at its index. -/
theorem cummin_fst_apply (x : Tensor α s) (dim : Fin s.rank) (c : s.Coord) :
    (cummin x dim).1 c = x (Shape.Coord.update c dim ((cummin x dim).2 c)) :=
  Rep.get_ofFn _ c

/-- `cummax` reading its indices from one scan of each fiber. -/
def cummaxFast (x : Tensor α s) (dim : Fin s.rank) :
    Tensor α s × Tensor (Fin (s.extent dim)) s :=
  let scans := Tensor.selectionScans (fun best a =>
    !(a == a) || ((best == best) && !decide (best > a))) x dim
  let indices : Tensor (Fin (s.extent dim)) s := Rep.ofFn fun c =>
    (scans (s.axisTarget dim c))[(Shape.Coord.get c dim).val + 1]?.getD (Shape.Coord.get c dim)
  (Rep.ofFn fun c => x (Shape.Coord.update c dim (indices c)), indices)

/-- `cummin` reading its indices from one scan of each fiber. -/
def cumminFast (x : Tensor α s) (dim : Fin s.rank) :
    Tensor α s × Tensor (Fin (s.extent dim)) s :=
  let scans := Tensor.selectionScans (fun best a =>
    !(a == a) || ((best == best) && !decide (a > best))) x dim
  let indices : Tensor (Fin (s.extent dim)) s := Rep.ofFn fun c =>
    (scans (s.axisTarget dim c))[(Shape.Coord.get c dim).val + 1]?.getD (Shape.Coord.get c dim)
  (Rep.ofFn fun c => x (Shape.Coord.update c dim (indices c)), indices)

/-- The indices of `cummaxFast` are those of `cummax`. -/
private theorem cummaxFast_snd (x : Tensor α s) (dim : Fin s.rank) :
    (cummaxFast x dim).2 = (cummax x dim).2 :=
  Rep.ext fun c => by
    rw [cummax_snd_apply]
    simp only [cummaxFast, Rep.get_ofFn]
    exact selectionScans_read _ x dim c

/-- The indices of `cumminFast` are those of `cummin`. -/
private theorem cumminFast_snd (x : Tensor α s) (dim : Fin s.rank) :
    (cumminFast x dim).2 = (cummin x dim).2 :=
  Rep.ext fun c => by
    rw [cummin_snd_apply]
    simp only [cumminFast, Rep.get_ofFn]
    exact selectionScans_read _ x dim c

/-- `cummax` runs as `cummaxFast`, which scans each fiber once. -/
@[csimp] theorem cummaxFast_eq : @cummax = @cummaxFast := by
  funext α _ s _ _ _ x dim
  refine Prod.ext ?_ (cummaxFast_snd x dim).symm
  apply Rep.ext
  intro c
  rw [cummax_fst_apply, ← cummaxFast_snd]
  exact (Rep.get_ofFn (fun c =>
    x (Shape.Coord.update c dim ((cummaxFast x dim).2 c))) c).symm

/-- `cummin` runs as `cumminFast`, which scans each fiber once. -/
@[csimp] theorem cumminFast_eq : @cummin = @cumminFast := by
  funext α _ s _ _ _ x dim
  refine Prod.ext ?_ (cumminFast_snd x dim).symm
  apply Rep.ext
  intro c
  rw [cummin_fst_apply, ← cumminFast_snd]
  exact (Rep.get_ofFn (fun c =>
    x (Shape.Coord.update c dim ((cumminFast x dim).2 c))) c).symm

/-- Every index that the prefix fold through `k` visits, the start `0` and then `0` up to `k`, is
at most `k`. -/
private theorem le_of_mem_prefix {n k : Nat} (hk : k < n) {j : Fin n}
    (hj : j ∈ (⟨0, Nat.zero_lt_of_lt hk⟩ : Fin n) ::
      (List.finRange (k + 1)).map (Fin.castLE hk)) : j.val ≤ k := by
  rcases List.mem_cons.mp hj with rfl | hj
  · exact Nat.zero_le _
  · obtain ⟨i, _, rfl⟩ := List.mem_map.mp hj
    exact Nat.lt_succ_iff.mp i.isLt

/-- Every index at most `k` is one that the prefix fold through `k` visits. -/
private theorem mem_prefix {n k : Nat} (hk : k < n) {i : Fin n} (hi : i.val ≤ k) :
    i ∈ (⟨0, Nat.zero_lt_of_lt hk⟩ : Fin n) :: (List.finRange (k + 1)).map (Fin.castLE hk) :=
  List.mem_cons_of_mem _ (List.mem_map.mpr ⟨⟨i.val, Nat.lt_succ_of_le hi⟩, List.mem_finRange _,
    Fin.ext rfl⟩)

/-- The index of `cummax` at `c` is at most the index of `c` along `dim`. -/
theorem cummax_snd_le (x : Tensor α s) (dim : Fin s.rank) (c : s.Coord) :
    ((cummax x dim).2 c).val ≤ (Shape.Coord.get c dim).val := by
  rw [cummax_snd_apply]
  exact le_of_mem_prefix _ (foldl_select_mem _ _ _)

/-- The index of `cummin` at `c` is at most the index of `c` along `dim`. -/
theorem cummin_snd_le (x : Tensor α s) (dim : Fin s.rank) (c : s.Coord) :
    ((cummin x dim).2 c).val ≤ (Shape.Coord.get c dim).val := by
  rw [cummin_snd_apply]
  exact le_of_mem_prefix _ (foldl_select_mem _ _ _)

/-- When every entry of the fiber equals itself (no NaN) and the real reading `observe` reflects
their strict order, the reading of a `cummax` value is at least that of every entry of its
prefix. -/
theorem le_cummax (x : Tensor α s) (dim : Fin s.rank) (c : s.Coord) (observe : α → ℝ)
    (hself : ∀ j, (x (Shape.Coord.update c dim j) == x (Shape.Coord.update c dim j)) = true)
    (horder : ∀ i j, x (Shape.Coord.update c dim i) > x (Shape.Coord.update c dim j) ↔
      observe (x (Shape.Coord.update c dim i)) > observe (x (Shape.Coord.update c dim j)))
    (i : Fin (s.extent dim)) (hi : i.val ≤ (Shape.Coord.get c dim).val) :
    observe (x (Shape.Coord.update c dim i)) ≤ observe ((cummax x dim).1 c) := by
  rw [cummax_fst_apply, cummax_snd_apply]
  refine le_foldl_select (fun a b => !(a == a) || ((b == b) && !decide (b > a)))
    (fun j => x (Shape.Coord.update c dim j)) observe (fun i j hb => ?_) (fun i j hb => ?_)
    _ _ i (mem_prefix _ hi)
  · simp only [hself, Bool.not_true, Bool.false_or, Bool.true_and, Bool.not_eq_true',
      decide_eq_false_iff_not] at hb
    exact not_lt.mp fun hlt => hb ((horder j i).mpr hlt)
  · simp only [hself, Bool.not_true, Bool.false_or, Bool.true_and, Bool.not_eq_false',
      decide_eq_true_eq] at hb
    exact ((horder j i).mp hb).le

/-- When every entry of the fiber equals itself (no NaN) and the real reading `observe` reflects
their strict order, the reading of a `cummin` value is at most that of every entry of its
prefix. -/
theorem cummin_le (x : Tensor α s) (dim : Fin s.rank) (c : s.Coord) (observe : α → ℝ)
    (hself : ∀ j, (x (Shape.Coord.update c dim j) == x (Shape.Coord.update c dim j)) = true)
    (horder : ∀ i j, x (Shape.Coord.update c dim i) > x (Shape.Coord.update c dim j) ↔
      observe (x (Shape.Coord.update c dim i)) > observe (x (Shape.Coord.update c dim j)))
    (i : Fin (s.extent dim)) (hi : i.val ≤ (Shape.Coord.get c dim).val) :
    observe ((cummin x dim).1 c) ≤ observe (x (Shape.Coord.update c dim i)) := by
  rw [cummin_fst_apply, cummin_snd_apply]
  refine neg_le_neg_iff.mp (le_foldl_select
    (fun a b => !(a == a) || ((b == b) && !decide (a > b)))
    (fun j => x (Shape.Coord.update c dim j)) (fun a => -observe a) (fun i j hb => ?_)
    (fun i j hb => ?_) _ _ i (mem_prefix _ hi))
  · simp only [hself, Bool.not_true, Bool.false_or, Bool.true_and, Bool.not_eq_true',
      decide_eq_false_iff_not] at hb
    exact neg_le_neg (not_lt.mp fun hlt => hb ((horder i j).mpr hlt))
  · simp only [hself, Bool.not_true, Bool.false_or, Bool.true_and, Bool.not_eq_false',
      decide_eq_true_eq] at hb
    exact neg_le_neg ((horder i j).mp hb).le

end Extrema

/-! ## Vector-Jacobian products -/

/-- The vector-Jacobian product of `cumsum`: the cumulative sum of the incoming gradient from the
last index along `dim` down, as `flip (cumsum (flip g [dim]) dim) [dim]`. -/
def cumsumBackward [Add α] [Zero α] (g : Tensor α s) (dim : Fin s.rank) : Tensor α s :=
  flip (cumsum (flip g [dim]) dim) [dim]

/-- An entry of `cumsumBackward` is the entry of the flipped cumulative sum of the flipped
gradient. -/
theorem cumsumBackward_apply [Add α] [Zero α] (g : Tensor α s) (dim : Fin s.rank) (c : s.Coord) :
    cumsumBackward g dim c = flip (cumsum (flip g [dim]) dim) [dim] c :=
  rfl

/-- The vector-Jacobian product of `cumprod`. The entry at index `k` along `dim` adds, over the
indices `j ≥ k` in increasing order and from zero, the incoming gradient at `j` times the product,
in index order from one, of the entries with index at most `j` other than `k`. The formula divides
by no entry, so it gives the derivative of the cumulative products also where entries are zero. -/
def cumprodBackward [Add α] [Mul α] [Zero α] [One α] (x g : Tensor α s) (dim : Fin s.rank) :
    Tensor α s :=
  Rep.ofFn fun c => (List.finRange (s.extent dim)).foldl (fun acc j =>
    if Shape.Coord.get c dim ≤ j then
      acc + g (Shape.Coord.update c dim j) *
        ((List.finRange (j.val + 1)).map (Fin.castLE j.isLt)).foldl (fun product i =>
          if i = Shape.Coord.get c dim then product
          else product * x (Shape.Coord.update c dim i)) 1
    else acc) 0

/-- An entry of `cumprodBackward` is the fold of its definition. -/
theorem cumprodBackward_apply [Add α] [Mul α] [Zero α] [One α] (x g : Tensor α s)
    (dim : Fin s.rank) (c : s.Coord) :
    cumprodBackward x g dim c = (List.finRange (s.extent dim)).foldl (fun acc j =>
      if Shape.Coord.get c dim ≤ j then
        acc + g (Shape.Coord.update c dim j) *
          ((List.finRange (j.val + 1)).map (Fin.castLE j.isLt)).foldl (fun product i =>
            if i = Shape.Coord.get c dim then product
            else product * x (Shape.Coord.update c dim i)) 1
      else acc) 0 :=
  Rep.get_ofFn _ c

/-- The Jacobian-vector product of `cumprod`. The entry at `c` adds, over the indices `i` of its
prefix in increasing order and from zero, the tangent at `i` times the product, in index order from
one, of the other entries of the prefix. -/
def cumprodJVP [Add α] [Mul α] [Zero α] [One α] (x dx : Tensor α s) (dim : Fin s.rank) :
    Tensor α s :=
  Rep.ofFn fun c =>
    ((List.finRange ((Shape.Coord.get c dim).val + 1)).map
      (Fin.castLE (Shape.Coord.get c dim).isLt)).foldl (fun acc i =>
        acc + dx (Shape.Coord.update c dim i) *
          ((List.finRange ((Shape.Coord.get c dim).val + 1)).map
            (Fin.castLE (Shape.Coord.get c dim).isLt)).foldl (fun product j =>
              if j = i then product else product * x (Shape.Coord.update c dim j)) 1) 0

section ExtremaBackward

variable [BEq α] [LT α] [DecidableRel ((· > ·) : α → α → Prop)] [Add α] [Zero α]

/-- The vector-Jacobian product of the values of `cummax`: from zero, each incoming gradient entry
is added at the index its value comes from, in index order along `dim`. -/
def cummaxBackward (x g : Tensor α s) (dim : Fin s.rank) : Tensor α s :=
  Rep.ofFn fun c => (List.finRange (s.extent dim)).foldl (fun acc j =>
    if (cummax x dim).2 (Shape.Coord.update c dim j) = Shape.Coord.get c dim then
      acc + g (Shape.Coord.update c dim j)
    else acc) 0

/-- The vector-Jacobian product of the values of `cummin`: from zero, each incoming gradient entry
is added at the index its value comes from, in index order along `dim`. -/
def cumminBackward (x g : Tensor α s) (dim : Fin s.rank) : Tensor α s :=
  Rep.ofFn fun c => (List.finRange (s.extent dim)).foldl (fun acc j =>
    if (cummin x dim).2 (Shape.Coord.update c dim j) = Shape.Coord.get c dim then
      acc + g (Shape.Coord.update c dim j)
    else acc) 0

/-- An entry of `cummaxBackward` adds the incoming gradient entries routed to it. -/
theorem cummaxBackward_apply (x g : Tensor α s) (dim : Fin s.rank) (c : s.Coord) :
    cummaxBackward x g dim c = (List.finRange (s.extent dim)).foldl (fun acc j =>
      if (cummax x dim).2 (Shape.Coord.update c dim j) = Shape.Coord.get c dim then
        acc + g (Shape.Coord.update c dim j)
      else acc) 0 :=
  Rep.get_ofFn _ c

/-- An entry of `cumminBackward` adds the incoming gradient entries routed to it. -/
theorem cumminBackward_apply (x g : Tensor α s) (dim : Fin s.rank) (c : s.Coord) :
    cumminBackward x g dim c = (List.finRange (s.extent dim)).foldl (fun acc j =>
      if (cummin x dim).2 (Shape.Coord.update c dim j) = Shape.Coord.get c dim then
        acc + g (Shape.Coord.update c dim j)
      else acc) 0 :=
  Rep.get_ofFn _ c

end ExtremaBackward

end TorchLean.Tensor
