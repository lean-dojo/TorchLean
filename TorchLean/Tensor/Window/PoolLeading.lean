/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Window

/-!
# Pooling past untouched leading axes

Pooling the axes after an unpooled leading shape pools each slice at every index of the leading
shape through `mapLeading` (`TorchLean.Tensor.pool_concat`, with the shape equation
`Shape.pool_concat` and the step `pool_cons` for one outermost axis). An unpooled axis has a
one-cell window, so the windows, their row-major order, the sums, the divisors and the selections,
ties and NaNs included, are those of the slices.
-/

@[expose] public section

open TorchLean.Tensor TorchLean.Shape
open TorchLean.Shape (rank rank_cons rank_eq_length)

namespace TorchLean.Shape

/-! ## Pooled shapes with untouched leading axes -/

section PoolLeading

/-- A one-cell window with unit stride and dilation and no padding keeps the extent of an axis. -/
theorem poolExtent_one (n : Nat) (ceilMode : Bool) : poolExtent n 1 1 0 0 1 ceilMode = n := by
  unfold poolExtent
  by_cases hn : n = 0
  · simp [hn]
  · cases ceilMode <;> simp [hn, slidingWindowOutDimDilated, dilatedKernelExtent] <;>
      (try split_ifs) <;> omega

section ShiftedAxes

variable {r r' : Nat} {dims : List (Fin r)} {dims' : List (Fin r')}

/-- An axis list whose axes are those of `dims` shifted by one does not list axis zero, so axis zero
takes the default. -/
theorem valueAt_of_map_val_eq_zero (hd : dims'.map Fin.val = dims.map fun a => a.val + 1)
    (values : List Nat) (default : Nat) {a : Fin r'} (ha : a.val = 0) :
    valueAt dims' values default a = default := by
  refine valueAt_of_not_mem (fun hm => ?_) values default
  have : a.val ∈ dims'.map Fin.val := List.mem_map_of_mem hm
  rw [hd, List.mem_map] at this
  obtain ⟨b, _, hb⟩ := this
  omega

/-- Along an axis list whose axes are those of `dims` shifted by one, axis `a + 1` takes the value
that axis `a` takes along `dims`. -/
theorem valueAt_of_map_val_eq_succ (hd : dims'.map Fin.val = dims.map fun a => a.val + 1)
    (values : List Nat) (default : Nat) {a' : Fin r'} {a : Fin r} (ha : a'.val = a.val + 1) :
    valueAt dims' values default a' = valueAt dims values default a := by
  induction dims generalizing dims' values with
  | nil =>
      have : dims' = [] := List.map_eq_nil_iff.1 hd
      subst this
      rfl
  | cons d dims ih =>
      obtain ⟨d', dims'', rfl, hd', htail⟩ : ∃ d' dims'', dims' = d' :: dims'' ∧
          d'.val = d.val + 1 ∧ dims''.map Fin.val = dims.map fun a => a.val + 1 := by
        cases dims' with
        | nil => simp at hd
        | cons d' dims'' =>
            simp only [List.map_cons, List.cons.injEq] at hd
            exact ⟨d', dims'', rfl, hd.1, hd.2⟩
      cases values with
      | nil => rfl
      | cons v values =>
          have hiff : a' = d' ↔ a = d := by
            rw [Fin.ext_iff, Fin.ext_iff]
            omega
          simp only [valueAt, hiff]
          split_ifs
          · rfl
          · exact ih htail values

end ShiftedAxes

variable {n : Nat} {s : Shape} {dims : List (Fin s.rank)} {dims' : List (Fin (Shape.rank (n :: s)))}

/-- Axis `a + 1` of `n :: s` has the extent of axis `a` of `s`. -/
theorem extent_cons_of_val_eq_succ {a' : Fin (Shape.rank (n :: s))} {a : Fin s.rank}
    (ha : a'.val = a.val + 1) : extent (n :: s) a' = extent s a := by
  obtain ⟨v, hv⟩ := a'
  simp only at ha
  subst ha
  rfl

/-- Pooling the axes of `s` shifted past an outermost axis keeps that axis. -/
theorem pool_cons (hd : dims'.map Fin.val = dims.map fun a => a.val + 1)
    (kernel stride padLow padHigh dilation : List Nat) (ceilMode : Bool) :
    Shape.pool (n :: s) dims' kernel stride padLow padHigh dilation ceilMode =
      n :: s.pool dims kernel stride padLow padHigh dilation ceilMode := by
  unfold Shape.pool
  erw [List.ofFn_succ (n := s.rank)]
  congr 1
  · simp only [valueAt_of_map_val_eq_zero hd _ _ (a := (0 : Fin (s.rank + 1))) rfl,
      poolExtent_one]
    rfl
  · congr 1
    funext a
    have hv : ∀ (values : List Nat) (default : Nat),
        valueAt dims' values default a.succ = valueAt dims values default a :=
      fun values default => valueAt_of_map_val_eq_succ hd values default rfl
    simp only [hv]
    rfl

/-- A pool over the axes of `s` shifted past an outermost axis has a window of one cell along that
axis. -/
theorem poolWindow_cons (hd : dims'.map Fin.val = dims.map fun a => a.val + 1)
    (kernel : List Nat) :
    poolWindow (n :: s) dims' kernel = 1 :: s.poolWindow dims kernel := by
  unfold poolWindow
  erw [List.ofFn_succ]
  congr 1
  · exact valueAt_of_map_val_eq_zero hd _ _ rfl
  · congr 1
    funext a
    exact valueAt_of_map_val_eq_succ hd _ _ rfl

section PoolPositionCons

variable (hd : dims'.map Fin.val = dims.map fun a => a.val + 1)
  {kernel stride padLow padHigh dilation : List Nat} {ceilMode : Bool}
  {O : (Shape.pool (n :: s) dims' kernel stride padLow padHigh dilation ceilMode).Coord}
  {K : (Shape.poolWindow (n :: s) dims' kernel).Coord}
  {o : (s.pool dims kernel stride padLow padHigh dilation ceilMode).Coord}
  {k : (s.poolWindow dims kernel).Coord}

include hd in
/-- Along the outermost axis, which is not pooled, a window cell of an output coordinate sits at
the output's own index. -/
theorem poolPosition_cons_zero (hK₀ : Coord.index K 0 = 0) {a : Fin (Shape.rank (n :: s))}
    (ha : a.val = 0) :
    poolPosition dims' kernel stride padLow padHigh dilation ceilMode O K a = Coord.index O 0 := by
  unfold poolPosition
  rw [valueAt_of_map_val_eq_zero hd _ _ ha, valueAt_of_map_val_eq_zero hd _ _ ha, ha, hK₀]
  simp

include hd in
/-- Along an axis after the outermost one, a window cell sits where the cell with the same inner
indices sits in the pool of a slice. -/
theorem poolPosition_cons_succ (hO : ∀ b, Coord.index O (b + 1) = Coord.index o b)
    (hK : ∀ b, Coord.index K (b + 1) = Coord.index k b) {a' : Fin (Shape.rank (n :: s))}
    {a : Fin s.rank}
    (ha : a'.val = a.val + 1) :
    poolPosition dims' kernel stride padLow padHigh dilation ceilMode O K a' =
      poolPosition dims kernel stride padLow padHigh dilation ceilMode o k a := by
  unfold poolPosition
  rw [valueAt_of_map_val_eq_succ hd _ _ ha, valueAt_of_map_val_eq_succ hd _ _ ha, ha, hO, hK]

end PoolPositionCons

/-- Pooling the axes of `S` shifted past a leading shape `L`, none of which is pooled, keeps `L`. -/
theorem pool_concat (L : Shape) {S : Shape} {dims : List (Fin S.rank)}
    {dims' : List (Fin (L ++ S).rank)} (hd : dims'.map Fin.val = dims.map (L.rank + ·.val))
    (kernel stride padLow padHigh dilation : List Nat) (ceilMode : Bool) :
    (L ++ S).pool dims' kernel stride padLow padHigh dilation ceilMode =
      L ++ (S.pool dims kernel stride padLow padHigh dilation ceilMode) := by
  induction L with
  | scalar =>
      have : dims' = dims := List.map_injective_iff.2 Fin.val_injective (by simpa using hd)
      subst this
      rfl
  | dim m L ih =>
      have hmid : ∀ a : Fin S.rank, L.rank + a.val < (L ++ S).rank := fun a => by
        have := a.isLt
        simp only [Shape.rank_eq_length, List.length_append] at this ⊢
        omega
      have hd' : (dims.map fun a => (⟨L.rank + a.val, hmid a⟩ : Fin (L ++ S).rank)).map
          Fin.val = dims.map (L.rank + ·.val) := by
        simp [Function.comp_def]
      have hcons : dims'.map Fin.val =
          (dims.map fun a => (⟨L.rank + a.val, hmid a⟩ : Fin (L ++ S).rank)).map
            fun a => a.val + 1 := by
        rw [hd]
        simp only [List.map_map, Function.comp_def, Shape.rank_cons]
        congr 1
        funext a
        omega
      show Shape.pool (m :: (L ++ S)) dims' kernel stride padLow padHigh dilation ceilMode = _
      rw [pool_cons (s := L ++ S) hcons, ih hd']
      rfl

end PoolLeading

end TorchLean.Shape

namespace TorchLean

/-! ## Pooled tensors with untouched leading axes -/

section PoolLeading

variable {α : Type} [Storage α] [Context α]

/-- The row-major coordinate of a shape with an outermost axis of one: index zero there and the
same flat index into the rest. -/
theorem Shape.Coord.unlinearize_one_cons {w : Shape} (j : Fin (Shape.size (1 :: w))) :
    Coord.unlinearize j =
      ((0 : Fin 1), Coord.unlinearize (Fin.cast (Nat.one_mul _) j)) := by
  apply Coord.linearize_injective
  rw [Coord.linearize_unlinearize, Coord.linearize_cons]
  apply Fin.ext
  simp [finProdFinEquiv_apply_val]
  rfl

/-- `List.finRange m` is `List.finRange k` cast along `k = m`. -/
private theorem Tensor.finRange_eq_map_cast {m k : Nat} (h : k = m) :
    List.finRange m = (List.finRange k).map (Fin.cast h) := by
  subst h
  simp

/-- The coordinate of a shape `1 :: w` at a flat index of `w` is index zero followed by the
coordinate of `w` there. -/
theorem Shape.Coord.unlinearize_cast_one_cons {w w' : Shape} (hw : w' = 1 :: w)
    (h : Shape.size w = Shape.size w') (j : Fin (Shape.size w)) :
    Coord.unlinearize (s := w') (Fin.cast h j) =
      _root_.cast (congrArg Shape.Coord hw.symm) ((0 : Fin 1), Coord.unlinearize j) := by
  subst hw
  rw [unlinearize_one_cons]
  rfl

section PoolCons

variable {n : Nat} {s : Shape} {dims : List (Fin s.rank)} {dims' : List (Fin (Shape.rank (n :: s)))}
  {kernel stride padLow padHigh dilation : List Nat} {ceilMode : Bool}

/-- Let `O` be an output coordinate of the pool over the shifted axes and `K` a cell of its window
with index zero on the outermost axis. `K` reads the input at the outermost index of `O` followed
by the coordinate that the matching cell `k` of the matching output `o` of a slice reads. Where
that cell reads padding, so does `K`. -/
theorem Tensor.poolSource_cons (hd : dims'.map Fin.val = dims.map fun a => a.val + 1)
    {O : (Shape.pool (n :: s) dims' kernel stride padLow padHigh dilation ceilMode).Coord}
    {K : (Shape.poolWindow (n :: s) dims' kernel).Coord} {i : Fin n}
    {o : (s.pool dims kernel stride padLow padHigh dilation ceilMode).Coord}
    {k : (s.poolWindow dims kernel).Coord}
    (hO₀ : Shape.Coord.index O 0 = i.val)
    (hO : ∀ b, Shape.Coord.index O (b + 1) = Shape.Coord.index o b)
    (hK₀ : Shape.Coord.index K 0 = 0)
    (hK : ∀ b, Shape.Coord.index K (b + 1) = Shape.Coord.index k b) :
    poolSource O K = (poolSource o k).map (Prod.mk i) := by
  have hv₀ := fun values default => Shape.valueAt_of_map_val_eq_zero hd values default
    (a := (0 : Fin (s.rank + 1))) rfl
  have hv := fun values default (b : Fin s.rank) =>
    Shape.valueAt_of_map_val_eq_succ hd values default
    (a' := b.succ) (a := b) rfl
  have hiff : (∀ a, Shape.valueAt dims' padLow 0 a ≤
        Shape.poolPosition dims' kernel stride padLow padHigh dilation ceilMode O K a ∧
      Shape.poolPosition dims' kernel stride padLow padHigh dilation ceilMode O K a -
        Shape.valueAt dims' padLow 0 a < Shape.extent (n :: s) a) ↔
      ∀ b, Shape.valueAt dims padLow 0 b ≤
          Shape.poolPosition dims kernel stride padLow padHigh dilation ceilMode o k b ∧
        Shape.poolPosition dims kernel stride padLow padHigh dilation ceilMode o k b -
          Shape.valueAt dims padLow 0 b < s.extent b := by
    have hp₀ := Shape.poolPosition_cons_zero hd hK₀ (O := O)
      (a := (0 : Fin (s.rank + 1))) rfl
    have hp := fun b : Fin s.rank => Shape.poolPosition_cons_succ hd hO hK
      (a' := b.succ) (a := b) rfl
    have he := fun b : Fin s.rank => Shape.extent_cons_of_val_eq_succ (n := n)
      (a' := b.succ) (a := b) rfl
    have he₀ : Shape.extent (n :: s) (0 : Fin (s.rank + 1)) = n := rfl
    erw [Fin.forall_fin_succ (n := s.rank)]
    simp only [hv₀, hv, hp₀, hp, he₀, he, hO₀]
    simp
  unfold poolSource
  split_ifs with h₁ h₂ h₂
  · simp only [Option.map_some, Option.some.injEq]
    apply Shape.Coord.ext_index
    intro m hm
    cases m with
    | zero =>
        rw [Shape.Coord.index_ofFun, dite_eq_left hm, Shape.Coord.index_zero]
        dsimp only
        rw [Shape.poolPosition_cons_zero hd hK₀ rfl, Shape.valueAt_of_map_val_eq_zero hd _ _ rfl,
          hO₀, Nat.sub_zero]
    | succ b =>
        have hb : b < s.rank := by simpa using hm
        rw [Shape.Coord.index_ofFun, dite_eq_left hm, Shape.Coord.index_cons_succ,
          Shape.Coord.index_ofFun, dite_eq_left hb]
        dsimp only
        rw [Shape.poolPosition_cons_succ hd hO hK (a := ⟨b, hb⟩) rfl,
          Shape.valueAt_of_map_val_eq_succ hd _ _ (a := ⟨b, hb⟩) rfl]
  · exact absurd (hiff.1 h₁) h₂
  · exact absurd (hiff.2 h₂) h₁
  · rfl

/-- The window over the shifted axes has as many cells as the window of a slice. -/
theorem Shape.size_poolWindow_cons (hd : dims'.map Fin.val = dims.map fun a => a.val + 1) :
    Shape.size (s.poolWindow dims kernel) =
      Shape.size (Shape.poolWindow (n :: s) dims' kernel) := by
  rw [Shape.poolWindow_cons hd kernel, Shape.size_cons, Nat.one_mul]

/-- The window cell over the shifted axes at a flat index is index zero on the outermost axis
followed by the window cell of a slice at that index. -/
theorem Shape.Coord.unlinearize_poolWindow_cons
    (hd : dims'.map Fin.val = dims.map fun a => a.val + 1)
    (j : Fin (Shape.size (s.poolWindow dims kernel))) :
    Coord.unlinearize (Fin.cast (size_poolWindow_cons hd) j) =
      _root_.cast (congrArg Shape.Coord (Shape.poolWindow_cons hd kernel).symm)
        ((0 : Fin 1), Coord.unlinearize j) :=
  unlinearize_cast_one_cons (Shape.poolWindow_cons hd kernel) _ j

variable (hd : dims'.map Fin.val = dims.map fun a => a.val + 1) (x : Tensor α (n :: s))
  {O : (Shape.pool (n :: s) dims' kernel stride padLow padHigh dilation ceilMode).Coord} {i : Fin n}
  {o : (s.pool dims kernel stride padLow padHigh dilation ceilMode).Coord}

include hd in
/-- A window cell reads the input where the matching cell of the slice at the outermost index reads
the slice. -/
theorem Tensor.poolInput_cons (hO₀ : Shape.Coord.index O 0 = i.val)
    (hO : ∀ b, Shape.Coord.index O (b + 1) = Shape.Coord.index o b)
    (k : (s.poolWindow dims kernel).Coord) :
    poolInput x O (_root_.cast (congrArg Shape.Coord (Shape.poolWindow_cons hd kernel).symm)
        ((0 : Fin 1), k)) = poolInput (unstack x i) o k := by
  have hsrc : poolSource O (_root_.cast (congrArg Shape.Coord
      (Shape.poolWindow_cons hd kernel).symm) ((0 : Fin 1), k)) =
        (poolSource o k).map (Prod.mk i) :=
    poolSource_cons hd hO₀ hO
      (Shape.Coord.index_cast (Shape.poolWindow_cons hd kernel).symm ((0 : Fin 1), k) 0) fun b =>
      (Shape.Coord.index_cast (Shape.poolWindow_cons hd kernel).symm ((0 : Fin 1), k) (b + 1)).trans
        rfl
  unfold poolInput
  rw [hsrc]
  cases poolSource o k with
  | none => rfl
  | some c => exact (Tensor.unstack_apply x i c).symm

include hd in
/-- At an output coordinate, the window sum is the window sum of the matching coordinate of the
slice at its outermost index, taken in the same order. -/
theorem Tensor.poolSum_cons (hO₀ : Shape.Coord.index O 0 = i.val)
    (hO : ∀ b, Shape.Coord.index O (b + 1) = Shape.Coord.index o b) :
    poolSum x O = poolSum (unstack x i) o := by
  unfold poolSum
  rw [finRange_eq_map_cast (size_poolWindow_cons hd), List.foldl_map]
  simp only [Coord.unlinearize_poolWindow_cons hd, poolInput_cons hd x hO₀ hO]

include hd in
/-- An average divides by the cell count of the matching coordinate of the slice. -/
theorem Tensor.poolCount_cons (countIncludePad : Bool) (hO₀ : Shape.Coord.index O 0 = i.val)
    (hO : ∀ b, Shape.Coord.index O (b + 1) = Shape.Coord.index o b) :
    poolCount countIncludePad O = poolCount countIncludePad o := by
  unfold poolCount
  rw [finRange_eq_map_cast (size_poolWindow_cons hd), List.countP_map]
  congr 1
  funext j
  simp only [Function.comp_apply, Coord.unlinearize_poolWindow_cons hd]
  have hK₀ := Shape.Coord.index_cast (Shape.poolWindow_cons hd kernel).symm
    ((0 : Fin 1), Coord.unlinearize j) 0
  have hK := fun b => (Shape.Coord.index_cast (Shape.poolWindow_cons hd kernel).symm
    ((0 : Fin 1), Coord.unlinearize j) (b + 1)).trans rfl
  cases countIncludePad with
  | false =>
      simp only [Bool.false_eq_true, ite_false]
      rw [poolSource_cons hd hO₀ hO hK₀ hK, Option.isSome_map]
  | true =>
      simp only [ite_true, decide_eq_decide]
      have hv₀ := fun values default => Shape.valueAt_of_map_val_eq_zero hd values default
        (a := (0 : Fin (s.rank + 1))) rfl
      have hv := fun values default (b : Fin s.rank) =>
        Shape.valueAt_of_map_val_eq_succ hd values default
          (a' := b.succ) (a := b) rfl
      have hp₀ := Shape.poolPosition_cons_zero hd hK₀ (O := O)
        (a := (0 : Fin (s.rank + 1))) rfl
      have hp := fun b : Fin s.rank => Shape.poolPosition_cons_succ hd hO hK
        (a' := b.succ) (a := b) rfl
      have he := fun b : Fin s.rank => Shape.extent_cons_of_val_eq_succ (n := n)
        (a' := b.succ) (a := b) rfl
      have he₀ : Shape.extent (n :: s) (0 : Fin (s.rank + 1)) = n := rfl
      erw [Fin.forall_fin_succ (n := s.rank)]
      simp only [hv₀, hv, hp₀, hp, he₀, he, hO₀, Nat.add_zero, i.isLt, true_and]

include hd in
/-- At an output coordinate, a maximum or minimum pool selects the coordinate that the slice at its
outermost index selects at the matching output, behind that index. Both run the same scan in the
same order, so they make the same choice at ties and NaNs. -/
theorem Tensor.poolIndices_cons (largest : Bool) (hO₀ : Shape.Coord.index O 0 = i.val)
    (hO : ∀ b, Shape.Coord.index O (b + 1) = Shape.Coord.index o b) :
    poolIndices largest x dims' kernel stride padLow padHigh dilation ceilMode O =
      (poolIndices largest (unstack x i) dims kernel stride padLow padHigh dilation ceilMode
        o).map (Prod.mk i) := by
  unfold poolIndices
  rw [Rep.get_ofFn, Rep.get_ofFn,
    finRange_eq_map_cast (size_poolWindow_cons hd), List.foldl_map]
  simp only [Coord.unlinearize_poolWindow_cons hd]
  refine List.foldl_hom (Option.map (Prod.mk i)) (init := none) fun best j => ?_
  have hK₀ := Shape.Coord.index_cast (Shape.poolWindow_cons hd kernel).symm
    ((0 : Fin 1), Coord.unlinearize j) 0
  have hK := fun b => (Shape.Coord.index_cast (Shape.poolWindow_cons hd kernel).symm
    ((0 : Fin 1), Coord.unlinearize j) (b + 1)).trans rfl
  rw [poolSource_cons hd hO₀ hO hK₀ hK]
  cases poolSource o (Coord.unlinearize j) with
  | none => rfl
  | some c =>
      cases best with
      | none => rfl
      | some b =>
          simp only [Option.map_some, Tensor.unstack_apply]
          split_ifs <;> rfl

include hd in
/-- Pooling the axes of `s` shifted past an outermost axis, which is not pooled, pools each slice
along that axis: the same windows in the same order, so the same sums, divisors and selections. -/
theorem Tensor.pool_cons (op : PoolOp) :
    pool op x dims' kernel stride padLow padHigh dilation ceilMode =
      castShape (dim fun i => pool op (unstack x i) dims kernel stride padLow padHigh dilation
        ceilMode) (Shape.pool_cons hd kernel stride padLow padHigh dilation ceilMode).symm := by
  apply Rep.ext
  intro O
  rw [castShape_apply]
  obtain ⟨i, o, hc⟩ : ∃ i o, _root_.cast (congrArg Shape.Coord
      (Shape.pool_cons hd kernel stride padLow padHigh dilation ceilMode).symm.symm) O = (i, o) :=
    ⟨_, _, rfl⟩
  have hO₀ : Shape.Coord.index O 0 = i.val := by
    rw [← Shape.Coord.index_cast
      (Shape.pool_cons hd kernel stride padLow padHigh dilation ceilMode) O 0, hc]
    rfl
  have hO : ∀ b, Shape.Coord.index O (b + 1) = Shape.Coord.index o b := by
    intro b
    rw [← Shape.Coord.index_cast
      (Shape.pool_cons hd kernel stride padLow padHigh dilation ceilMode) O (b + 1), hc]
    rfl
  rw [hc, Tensor.dim_apply]
  cases op with
  | sum =>
      rw [pool_sum_apply, pool_sum_apply]
      exact poolSum_cons hd x hO₀ hO
  | avg countIncludePad =>
      rw [pool_avg_apply, pool_avg_apply]
      exact congrArg₂ Context.divNat (poolSum_cons hd x hO₀ hO)
        (poolCount_cons hd countIncludePad hO₀ hO)
  | max =>
      rw [pool_max_apply, pool_max_apply, poolIndices_cons hd x true hO₀ hO]
      cases poolIndices true (unstack x i) dims kernel stride padLow padHigh dilation ceilMode
          o with
      | none => rfl
      | some c => exact (Tensor.unstack_apply x i c).symm
  | min =>
      rw [pool_min_apply, pool_min_apply, poolIndices_cons hd x false hO₀ hO]
      cases poolIndices false (unstack x i) dims kernel stride padLow padHigh dilation ceilMode
          o with
      | none => rfl
      | some c => exact (Tensor.unstack_apply x i c).symm

/-- Pooling the axes of `S` shifted past an unpooled leading shape `L` pools each slice at every
index of `L` (`mapLeading`). Each slice keeps its windows and their order, so the sums, divisors and
selections agree, ties and NaNs included. -/
theorem Tensor.pool_concat (op : PoolOp) (L : Shape) {S : Shape} {dims : List (Fin S.rank)}
    {dims' : List (Fin (L ++ S).rank)} (hd : dims'.map Fin.val = dims.map (L.rank + ·.val))
    (x : Tensor α (L ++ S)) (kernel stride padLow padHigh dilation : List Nat)
    (ceilMode : Bool) :
    pool op x dims' kernel stride padLow padHigh dilation ceilMode =
      castShape (mapLeading L (fun t : Tensor α S =>
          pool op t dims kernel stride padLow padHigh dilation ceilMode) x)
        (Shape.pool_concat L hd kernel stride padLow padHigh dilation ceilMode).symm := by
  induction L with
  | scalar =>
      have : dims' = dims :=
        List.map_injective_iff.2 Fin.val_injective (by simpa using hd)
      subst this
      rfl
  | dim m L ih =>
      have hmid : ∀ a : Fin S.rank, L.rank + a.val < (L ++ S).rank := fun a => by
        have := a.isLt
        simp only [Shape.rank_eq_length, List.length_append] at this ⊢
        omega
      have hd' : (dims.map fun a => (⟨L.rank + a.val, hmid a⟩ : Fin (L ++ S).rank)).map
          Fin.val = dims.map (L.rank + ·.val) := by
        simp [Function.comp_def]
      have hcons : dims'.map Fin.val =
          (dims.map fun a => (⟨L.rank + a.val, hmid a⟩ : Fin (L ++ S).rank)).map
            fun a => a.val + 1 := by
        rw [hd]
        simp only [List.map_map, Function.comp_def, Shape.rank_cons]
        congr 1
        funext a
        omega
      have key : ∀ {A B : Shape} (h : A = B) (g : Fin m → Tensor α A),
          dim (fun i => castShape (g i) h) = castShape (dim g) (congrArg (m :: ·) h) := by
        intro A B h g
        subst h
        rfl
      rw [pool_cons hcons x op]
      simp only [ih hd', key, castShape_trans]
      rfl

end PoolCons

end PoolLeading

end TorchLean
