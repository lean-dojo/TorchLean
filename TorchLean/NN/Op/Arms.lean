/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Op.Table.Movement
public import TorchLean.Tensor.Op.Table.Reduce

/-!
# Bridges from the operator table to the per-frontend definitions

Each theorem here says that one arm of `Op.rule?` computes what a separate definition computes, so
results about the definition can be read about the table. `(op.rule?).get rfl` is the arm's rule.
The arms are the one-axis and total reductions, the pullbacks of the scatter-add along one axis,
the selection at one index and the concatenation of two inputs.

The one-axis sum and mean and the total sum are the arms of `Op.reduce` and `Op.mean`, read
through the cast of `Shape.reduce_singleton` or `Shape.reduce_finRange` to the shape without the
reduced axes. `reduce_sum_singleton` and `mean_singleton` give their forward maps and pullbacks on
a nonempty axis, and `reduce_sum` gives the total `Tensor.sum` and its pullback. The scatter-add
along one axis is the sum arm of `Op.scatter` over every axis at the index tuples of
`Tensor.selectSources` (`scatter_select`). Its base receives the upstream gradient, and its updates
read that gradient at the selected indices, the way the index selection along one axis reads its
input. The selection at one index of an axis is the one-index `Op.slice` followed by the
`Op.reshape` that drops the axis (`indexAt_slice`). The concatenation of two inputs along the
leading axis is the arm of `Op.cat` at two sizes (`cat_two`): its value is the two-block
`Tensor.cat`, and the input gradients are the leading-axis slices of the upstream gradient.
`cat_catElementParts_two` and `catGrads_two` state the two parts of the arm at any storage type.
-/

@[expose] public section

open TorchLean.Shape (size size_cons size_nil)

namespace TorchLean.Op

open TorchLean.Tensor Shape

variable {α : Type} [Storage α] [Context α]

/-! ## Reductions -/

section Reductions

variable {s : Shape} (x : Tensor α s)

/-- Read through the cast of `Shape.reduce_singleton`, the arm of `Op.reduce` summing over the one
axis `axis` is `Tensor.reduce .sum` over that axis, and its pullback broadcasts the upstream
gradient along `axis`. -/
theorem reduce_sum_singleton (axis : Nat) [Shape.HasNonemptyAxis axis s] [Shape.WellFormed s]
    (dz : Tensor α (s.eraseIdx axis)) :
    let rule := (Op.reduce (s := s) .sum [⟨axis, Shape.AxisInBounds.proof⟩] false
      (fun h => absurd h (by decide))).rule? (α := α) |>.get rfl
    castShape (rule.forward (x, ())).1
          (Shape.reduce_singleton s ⟨axis, Shape.AxisInBounds.proof⟩) =
        castShape (Tensor.reduce .sum x [⟨axis, Shape.AxisInBounds.proof⟩] false)
          (Shape.reduce_singleton _ _) ∧
      (rule.backward (x, ()) (castShape dz
          (Shape.reduce_singleton s ⟨axis, Shape.AxisInBounds.proof⟩).symm, ())).1 =
        Tensor.broadcast dz s _ (s.eraseIdx_broadcastDims axis) := by
  refine ⟨rfl, ?_⟩
  simp only [Op.rule?_reduce, Option.get_some, Tensor.reduceBackward]
  apply Tensor.broadcast_castShape
  intro i
  simp only [Shape.reduceAxes]
  rw [Shape.val_getElem_otherAxes_singleton]
  rfl

/-- The total `Tensor.sum` is the arm of `Op.reduce` summing over every axis, read through the
cast of `Shape.reduce_finRange`, and its pullback fills the input shape with the upstream
scalar. -/
theorem reduce_sum :
    let rule := (Op.reduce (s := s) .sum (List.finRange s.rank) false
      (fun h => absurd h (by decide))).rule? (α := α) |>.get rfl
    castShape (rule.forward (x, ())).1 (Shape.reduce_finRange s) = Tensor.scalar (Tensor.sum x) ∧
      ∀ dz : Tensor α Shape.scalar,
        (rule.backward (x, ()) (castShape dz (Shape.reduce_finRange s).symm, ())).1 =
          Tensor.full s dz.item := by
  refine ⟨?_, fun dz => ?_⟩
  · rw [Tensor.sum, Tensor.scalar_item]
    rfl
  · apply Tensor.Rep.ext
    intro c
    simp only [Op.rule?_reduce, Option.get_some, Tensor.reduceBackward, Tensor.broadcast_apply,
      Tensor.castShape_apply, Tensor.full_apply, Tensor.item]

/-- The arm of `Op.mean` over the one nonempty axis `axis`, read through the cast of
`Shape.reduce_singleton`: it divides the sum once by the axis size, and its pullback broadcasts
the upstream gradient along `axis` and divides each entry once by that size (`Context.divNat`). -/
theorem mean_singleton (axis : Nat) [hv : Shape.HasNonemptyAxis axis s] [Shape.WellFormed s]
    (dz : Tensor α (s.eraseIdx axis)) :
    let rule := (Op.mean (s := s) [⟨axis, Shape.AxisInBounds.proof⟩] false).rule? (α := α)
      |>.get rfl
    castShape (rule.forward (x, ())).1
          (Shape.reduce_singleton s ⟨axis, Shape.AxisInBounds.proof⟩) =
        Tensor.map (Context.divNat · (Shape.axisSize s axis))
          (castShape (Tensor.reduce .sum x [⟨axis, Shape.AxisInBounds.proof⟩] false)
            (Shape.reduce_singleton _ _)) ∧
      (rule.backward (x, ()) (castShape dz
          (Shape.reduce_singleton s ⟨axis, Shape.AxisInBounds.proof⟩).symm, ())).1 =
        Tensor.map (Context.divNat · (Shape.axisSize s axis))
          (Tensor.broadcast dz s _ (s.eraseIdx_broadcastDims axis)) := by
  have hpos : 0 < s.extent ⟨axis, Shape.AxisInBounds.proof⟩ := hv.proof.extent_pos _
  have hcount : (if Shape.size (s.selectAxes [⟨axis, Shape.AxisInBounds.proof⟩]) = 0
      then 1 else Shape.size (s.selectAxes [⟨axis, Shape.AxisInBounds.proof⟩])) =
      Shape.axisSize s axis := by
    rw [Shape.selectAxes_singleton]
    simp only [Shape.size_nil, Shape.size_cons, Nat.mul_one]
    rw [ite_eq_right (Nat.pos_iff_ne_zero.mp hpos)]
    exact (Shape.axisSize_eq_extent s ⟨axis, Shape.AxisInBounds.proof⟩).symm
  constructor
  · show castShape (Tensor.mean x _ false) _ = _
    rw [Tensor.mean, ← Tensor.map_castShape]
    simp only [hcount]
  · show Tensor.meanBackward _ false _ = _
    rw [Tensor.meanBackward]
    simp only [hcount]
    congr 1
    apply Tensor.broadcast_castShape
    intro i
    simp only [Shape.reduceAxes]
    rw [Shape.val_getElem_otherAxes_singleton]
    rfl

end Reductions

/-! ## Index selection -/

section Select

/-- For the scatter-add along `dim`, the sum arm of `Op.scatter` over every axis at the index tuples
of `Tensor.selectSources`, the base receives the upstream gradient. The updates, read at shape
`s.set dim count`, receive it gathered at the selected indices, as the index selection
reads its input. -/
theorem scatter_select {s : Shape} (dim : Fin s.rank) {count : Nat}
    (base : Tensor α s) (indices : Tensor (Fin (s.axisSize dim)) [count])
    (source : Tensor α (s.set dim count)) (g : Tensor α s) :
    let rule := (Op.scatter (t := s.set dim count) .sum (List.finRange s.rank)
      (List.nodup_finRange _)).rule? (α := α) |>.get rfl
    let index := Tensor.map (Op.indexOfAxisTuples s.extent (List.finRange s.rank))
      (Tensor.selectSources dim indices)
    let hshape : s.set dim count ++ s.eraseAxes (List.finRange s.rank) =
        s.set dim count := by rw [Shape.eraseAxes_finRange, List.append_nil]
    (rule.backward (base, index, castShape source hshape.symm, ()) (g, ())).1 = g ∧
      castShape (rule.backward (base, index, castShape source hshape.symm, ()) (g, ())).2.2.1
          hshape =
        Tensor.gather g (s.selectDims dim) (s.selectBatch dim count)
          (Tensor.selectIndices dim indices) (s.set dim count)
          (.select s dim count) := by
  intro rule index hshape
  have hidx : Tensor.map (Op.axisTuples s.extent (List.finRange s.rank)) index =
      Tensor.selectSources dim indices := by
    refine Rep.ext fun u => ?_
    simp only [index, Tensor.map_apply]
    exact axisTuples_indexOfAxisTuples _ _ _
  refine ⟨rfl, ?_⟩
  show castShape (Tensor.scatterUpdatesBackward .sum (List.finRange s.rank)
      (List.nodup_finRange _) (Tensor.map (Op.axisTuples _ _) index) g) hshape = _
  rw [hidx]
  refine Rep.ext fun u => ?_
  rw [castShape_apply, gather_select_apply]
  simp only [Tensor.scatterUpdatesBackward, gather_apply]
  congr 1
  apply Shape.Coord.ext_val
  intro a
  rw [Shape.GatherValid.source, Shape.Coord.get_ofFun, dite_eq_left (List.mem_finRange a)]
  simp only [Fin.val_cast, selectSources, Rep.get_ofFn, Shape.Coord.val_get]
  rw [Coord.appendEquiv_reindexEq_cast_fst hshape]
  simp

/-- The selection at `index` along `dim` is the one-index slice along `dim` followed by the
reshape that drops the axis: its value is the reshape arm's value at the slice arm's value, and its
pullback is the slice arm's pullback of the reshape arm's pullback. -/
theorem indexAt_slice {s : Shape} (dim : Fin s.rank) (index : Fin (s.extent dim))
    (x : Tensor α s) (g : Tensor α (s.eraseIdx dim)) :
    let hslice : Shape.SliceValid s [dim] [index.val] [index.val + 1] [1] :=
      Shape.sliceValid_singleton _ dim index.val 1 index.isLt
    let hsize :
        (s.slice [dim] [index.val] [index.val + 1] [1]).size = Shape.size (s.eraseIdx dim) := by
      rw [Shape.slice_singleton]; exact Tensor.size_shapeAfterSumKeepDim _ dim
    let rule := (Op.indexAt dim index).rule? (α := α) |>.get rfl
    let slice := (Op.slice [dim] [index.val] [index.val + 1] [1] hslice).rule? (α := α) |>.get rfl
    let reshape := (Op.reshape hsize).rule? (α := α) |>.get rfl
    (rule.forward (x, ())).1 = (reshape.forward ((slice.forward (x, ())).1, ())).1 ∧
      (rule.backward (x, ()) (g, ())).1 =
        (slice.backward (x, ()) (reshape.backward ((slice.forward (x, ())).1, ()) (g, ()))).1 := by
  intro hslice hsize rule slice reshape
  exact ⟨rfl, rfl⟩

end Select

/-! ## Concatenation -/

section Concat

omit [Context α] in
/-- Two `cat`s along one axis with equal sizes and equal blocks are equal. -/
private theorem cat_congr {s : Shape} {k : Nat} (dim : Fin s.rank) {sizes sizes' : Fin k → Nat}
    (hs : sizes = sizes') {xs : (i : Fin k) → Tensor α (s.set dim (sizes i))}
    {xs' : (i : Fin k) → Tensor α (s.set dim (sizes' i))} (hx : ∀ i, HEq (xs i) (xs' i))
    (out : Shape) (hout : out = s.set dim (∑ i, sizes i))
    (hout' : out = s.set dim (∑ i, sizes' i)) :
    Tensor.cat dim sizes xs out hout = Tensor.cat dim sizes' xs' out hout' := by
  subst hs
  obtain rfl : xs = xs' := funext fun i => eq_of_heq (hx i)
  rfl

omit [Context α] in
/-- The inputs of `Op.cat` along the leading axis at two sizes, concatenated, are the two-block
`Tensor.cat` of the two inputs. -/
theorem cat_catElementParts_two {n m : Nat} {s : Shape} (a : Tensor α (.dim n s))
    (b : Tensor α (.dim m s)) :
    Tensor.cat (s := .dim 0 s) ⟨0, Shape.AxisInBounds.proof⟩ [n, m].get
        (Op.catElementParts (e := .scalar) ⟨0, Shape.AxisInBounds.proof⟩ [n, m] (a, b, ()))
        ((Shape.dim 0 s).set 0 [n, m].sum) (by simp) =
      Tensor.cat (s := .dim 0 s) ⟨0, Shape.AxisInBounds.proof⟩ ![n, m]
        (Fin.cons a (Fin.cons b finZeroElim)) (.dim (n + m) s) := by
  exact cat_congr _ (by funext i; fin_cases i <;> rfl) (fun i => by fin_cases i <;> rfl) _ _ _

omit [Context α] in
/-- Along the leading axis, `Op.catGrads` gives two inputs of sizes `n` and `m` the slices of the
upstream gradient from `0` and from `n`, reshaped to the input shapes. -/
theorem catGrads_two {n m : Nat} {s : Shape} (g : Tensor α (.dim (n + m) s)) :
    Op.catGrads (s := .dim 0 s) (total := n + m) ⟨0, Shape.AxisInBounds.proof⟩ g 0 [n, m]
        (by simp) =
      (Tensor.reshape
        (Tensor.slice g [⟨0, Shape.AxisInBounds.proof⟩] [0] [0 + n] [1]
          (Shape.sliceValid_singleton (.dim (n + m) s) ⟨0, Shape.AxisInBounds.proof⟩ 0 n
            (by rw [Nat.zero_add]; exact Nat.le_add_right n m)))
        (.dim n s) (congrArg Shape.size (Shape.slice_singleton _ _ _ _)),
      Tensor.reshape
        (Tensor.slice g [⟨0, Shape.AxisInBounds.proof⟩] [n] [n + m] [1]
          (Shape.sliceValid_singleton (.dim (n + m) s) ⟨0, Shape.AxisInBounds.proof⟩ n m
            (Nat.le_refl _)))
        (.dim m s) (congrArg Shape.size (Shape.slice_singleton _ _ _ _)), ()) := by
  have hcast : ∀ {t u : Shape} (x : Tensor α t) (h : t.size = u.size) (heq : t = u),
      Tensor.reshape x u h = castShape x heq := by
    intro t u x h heq
    subst heq
    rfl
  have hR : ∀ (start count : Nat) (hc : start + count ≤ n + m)
      (heq : (Shape.dim (n + m) s).slice [⟨0, Shape.AxisInBounds.proof⟩] [start]
        [start + count] [1] = .dim count s) (c : Shape.Coord (.dim count s)),
      castShape (Tensor.slice g _ _ _ _ (Shape.sliceValid_singleton (.dim (n + m) s)
        ⟨0, Shape.AxisInBounds.proof⟩ start count hc)) heq c =
        g (⟨start + c.1.val, by omega⟩, c.2) := by
    intro start count hc heq c
    rw [castShape_apply, slice_apply]
    congr 1
    refine Shape.Coord.ext_index fun k hk => ?_
    rw [Shape.Coord.index_ofFun, dite_eq_left hk]
    rcases k with _ | k
    · simp only [Shape.valueAt_singleton, Shape.Coord.val_get, Fin.val_cast]
      show _ + _ * Shape.Coord.index (_root_.cast (congrArg Shape.Coord heq.symm) c) 0 = _
      rw [Shape.Coord.index_cast heq.symm]
      simp only [Nat.one_mul, ↓reduceIte]
      rfl
    · simp only [Shape.valueAt_singleton, Shape.Coord.val_get, Fin.val_cast]
      show _ + _ * Shape.Coord.index (_root_.cast (congrArg Shape.Coord heq.symm) c) (k + 1) = _
      rw [Shape.Coord.index_cast heq.symm]
      simp [Fin.ext_iff]
      rfl
  simp only [Op.catGrads]
  refine Prod.ext ?_ (Prod.ext ?_ rfl)
  · have heq : (Shape.dim (n + m) s).slice [⟨0, Shape.AxisInBounds.proof⟩] [0] [0 + n] [1] =
        .dim n s := Shape.slice_singleton (.dim (n + m) s) ⟨0, Shape.AxisInBounds.proof⟩ 0 n
    refine Eq.trans ?_ (hcast _ (congrArg Shape.size heq) heq).symm
    refine Rep.ext fun c => ?_
    simp only [Op.catBlock, Rep.get_ofFn]
    exact Eq.trans rfl (hR 0 n (by rw [Nat.zero_add]; exact Nat.le_add_right n m) heq c).symm
  · have heq : (Shape.dim (n + m) s).slice [⟨0, Shape.AxisInBounds.proof⟩] [n] [n + m] [1] =
        .dim m s := Shape.slice_singleton (.dim (n + m) s) ⟨0, Shape.AxisInBounds.proof⟩ n m
    refine Eq.trans ?_ (hcast _ (congrArg Shape.size heq) heq).symm
    refine Rep.ext fun c => ?_
    simp only [Op.catBlock, Rep.get_ofFn]
    exact Eq.trans (by simp only [Nat.zero_add]; rfl) (hR n m (Nat.le_refl _) heq c).symm

/-- The concatenation of two inputs along the leading axis is the arm of `Op.cat` at the two
sizes: its value is the two-block `Tensor.cat`, and the gradients of the inputs are the
leading-axis slices of the upstream gradient from `0` and from `n`, reshaped to the input shapes. -/
theorem cat_two {n m : Nat} {s : Shape} (a : Tensor α (.dim n s)) (b : Tensor α (.dim m s))
    (g : Tensor α (.dim (n + m) s)) :
    let rule :=
      (Op.cat (s := .dim 0 s) ⟨0, Shape.AxisInBounds.proof⟩ [n, m]).rule? (α := α) |>.get rfl
    (rule.forward (a, b, ())).1 = Tensor.cat (s := .dim 0 s) ⟨0, Shape.AxisInBounds.proof⟩ ![n, m]
        (Fin.cons a (Fin.cons b finZeroElim)) (.dim (n + m) s) ∧
      (rule.backward (a, b, ()) (g, ())).1 = Tensor.reshape
        (Tensor.slice g [⟨0, Shape.AxisInBounds.proof⟩] [0] [0 + n] [1]
          (Shape.sliceValid_singleton (.dim (n + m) s) ⟨0, Shape.AxisInBounds.proof⟩ 0 n
            (by rw [Nat.zero_add]; exact Nat.le_add_right n m)))
        (.dim n s) (congrArg Shape.size (Shape.slice_singleton _ _ _ _)) ∧
      (rule.backward (a, b, ()) (g, ())).2.1 = Tensor.reshape
        (Tensor.slice g [⟨0, Shape.AxisInBounds.proof⟩] [n] [n + m] [1]
          (Shape.sliceValid_singleton (.dim (n + m) s) ⟨0, Shape.AxisInBounds.proof⟩ n m
            (Nat.le_refl _)))
        (.dim m s) (congrArg Shape.size (Shape.slice_singleton _ _ _ _)) := by
  simp only [Op.rule?_cat, Option.get_some]
  exact ⟨cat_catElementParts_two a b, congrArg Prod.fst (catGrads_two g),
    congrArg (fun p => p.2.1) (catGrads_two g)⟩

end Concat

end TorchLean.Op
