/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Fiber
public import TorchLean.Tensor.Reduce
public import TorchLean.Tensor.Movement.SpecialForms

/-!
# Sums over one axis, and the pullbacks of reductions

`reduce` and its read laws are in `TorchLean.Tensor.Reduce`. A sum over one
axis is cast to the shape without that axis (`Shape.reduce_singleton`), where an entry is the fold
of the vector along the axis through it. `castShape_reduce_sum_singleton_apply` reads an entry so
at an axis `dim : Fin s.rank`, and `castShape_reduce_sum_singleton_mk_apply` and
`castShape_reduce_sum_singleton_apply_of_lt` at a natural axis whose bound is an instance or a
hypothesis. `castShape_reduce_sum_singleton` and `castShape_reduce_sum_singleton_mk` state the cast
sum as the tensor of the sums along the axis. Over the outer axis,
`castShape_reduce_sum_zero_apply` and `castShape_reduce_sum_zero` read the entries along the axis at
plain coordinate pairs. The proofs read the rebuilt coordinates through
`Coord.index_axisEquiv_symm` of `TorchLean.Tensor.Movement.SpecialForms`.
`broadcast_eraseIdx_apply` reads the broadcast back along the same axis at the other indices, and
`ofFun_broadcast_eraseIdx` rewrites the coordinate that `broadcast_apply` gives to that form.

The output of a reduction over `dims` broadcasts back to its input along `Shape.reduceAxes`
(`Shape.reduce_broadcastDims`). `reduceBackward` and `meanBackward` are the pullbacks of `reduce`
and `mean` that the operator table registers. A sum sends each input entry the upstream gradient of
the output entry it is added into, and a product sends that gradient times the product of the other
entries of its fiber (a fold that skips the entry, so a zero needs no division). A maximum or
minimum sends each entry equal to the extremum that gradient divided by the number of such entries,
and the others zero. Equal means equal by the scalar's `==` or both NaN. A mean sends that gradient
divided once by the fiber's size, or by one for an empty fiber. `reduceJVP` is the tangent of
`reduce`, the transpose of `reduceBackward`.

We derived these pullbacks by hand. On a product torch 2.8.0 agrees at the zeros we tried, with
`[6, 0, 0]` for `[0, 2, 3]` and zeros for `[0, 0, 3]`. At `[NaN, 1, 2]` it sends a NaN to every
entry, where the fold that skips the entry gives `[2, NaN, NaN]`. No test compares
`reduceBackward` or `meanBackward` with torch autograd.
-/

@[expose] public section

open TorchLean.Shape (rank rank_eq_length)

namespace TorchLean.Shape

/-- The axis of `s` that axis `i` of the output of a reduction over `dims` is: `i` itself with
`keepdim`, otherwise the `i`-th axis outside `dims`. -/
def reduceAxes (s : Shape) (dims : List (Fin s.rank)) :
    (keepdim : Bool) → Fin (s.reduce dims keepdim).rank → Fin s.rank
  | true, i => ⟨i.val, by
      have h := i.isLt
      simpa [Shape.reduce, keepAxes, Shape.rank_eq_length] using h⟩
  | false, i => (s.otherAxes dims)[i.val]'(by
      have h := i.isLt
      simpa [Shape.reduce, eraseAxes, Shape.rank_eq_length] using h)

/-- The output of a reduction over `dims` broadcasts back to `s`, its axis `i` placed at
`reduceAxes dims keepdim i`. -/
theorem reduce_broadcastDims (s : Shape) (dims : List (Fin s.rank)) (keepdim : Bool) :
    BroadcastDims (s.reduce dims keepdim) s (s.reduceAxes dims keepdim) := by
  cases keepdim with
  | false =>
    refine ⟨fun a b hab => ?_, fun i => Or.inr ?_⟩
    · exact List.pairwise_iff_getElem.mp (s.pairwise_otherAxes dims) a.val b.val _ _ hab
    · simp only [extent, reduceAxes, Shape.reduce, Bool.cond_false, eraseAxes,
        List.getElem_map]
  | true =>
    refine ⟨fun a b hab => hab, fun i => ?_⟩
    simp only [extent, reduceAxes, Shape.reduce, Bool.cond_true, keepAxes, List.getElem_ofFn]
    split_ifs
    · exact Or.inl rfl
    · exact Or.inr rfl

end TorchLean.Shape

namespace TorchLean.Tensor

variable {α : Type} [Storage α] {s : Shape}

/-- An entry of a sum over one axis, read through a cast to the shape without that axis, adds the
entries along that axis to zero in increasing order. -/
theorem castShape_reduce_sum_singleton_apply_of_lt [Add α] [Zero α] (x : Tensor α s) (axis : Nat)
    (hd : axis < s.rank)
    (h : (ReduceOp.Scalar.seed (op := .sum) (α := α)).isNone →
      ∀ d ∈ [(⟨axis, hd⟩ : Fin s.rank)], 0 < s.extent d)
    (q : s.reduce [⟨axis, hd⟩] false = s.eraseIdx axis) (c : Shape.Coord (s.eraseIdx axis)) :
    castShape (reduce .sum x [⟨axis, hd⟩] false h) q c =
      Tensor.foldl (· + ·) 0
        (Tensor.ofFn fun i : Fin (s.axisSize axis (h := ⟨hd⟩)) =>
          x ((Shape.Coord.axisEquiv axis (shape := s) (_h := ⟨hd⟩)).symm (i, c))) := by
  have : Shape.AxisInBounds axis s := ⟨hd⟩
  have hindex : ∀ {t u : Shape} (e : t = u) (c : u.Coord) (k : Nat),
      Shape.Coord.index (_root_.cast (congrArg Shape.Coord e.symm) c) k =
        Shape.Coord.index c k := by
    intro t u e c k
    subst e
    rfl
  rw [castShape_apply, reduce_sum_apply, Rep.foldl_eq_fin_foldl,
    Fin.foldl_eq_finRange_foldl]
  simp only [Rep.getFlat_eq_get, Tensor.ofFn, Rep.get_ofFn]
  refine (foldl_finRange_congr
    (by rw [Shape.selectAxes_singleton, ← Shape.axisSize_eq_extent])
    (fun a (r : Shape.Coord [Shape.axisSize s axis]) =>
      a + x ((Shape.Coord.axisEquiv axis (shape := s)).symm (r.1, c)))
    (fun a (r : Shape.Coord (s.selectAxes [⟨axis, hd⟩])) =>
      a + x (s.reduceSource [⟨axis, hd⟩] false
        (_root_.cast (congrArg Shape.Coord q.symm) c) r)) ?_ 0).symm
  intro a r r' hrr
  show a + x _ = a + x _
  have hsrc : (Shape.Coord.axisEquiv axis (shape := s)).symm (r.1, c) =
      s.reduceSource [⟨axis, hd⟩] false (_root_.cast (congrArg Shape.Coord q.symm) c) r' := by
    apply Shape.Coord.ext_index
    intro k hk
    rw [Shape.index_reduceSource, dite_eq_left hk, Shape.Coord.index_axisEquiv_symm]
    by_cases hka : k = axis
    · have hd' : (⟨k, hk⟩ : Fin s.rank) = ⟨axis, hd⟩ := Fin.ext hka
      rw [ite_eq_left hka, hd', ite_eq_left (List.mem_singleton_self _), Shape.sortedAxes_singleton,
        List.idxOf_cons_self]
      exact hrr 0
    · have hd' : (⟨k, hk⟩ : Fin s.rank) ≠ ⟨axis, hd⟩ :=
        fun e => hka (congrArg Fin.val e)
      rw [ite_eq_right hka, ite_eq_right (mt List.mem_singleton.mp hd'), Shape.keptIndex,
        Bool.cond_false, Shape.idxOf_otherAxes_singleton s hd', hindex q]
      simp only [Fin.mk_lt_mk]
      split_ifs <;> rfl
  rw [hsrc]

/-- `castShape_reduce_sum_singleton_apply_of_lt` with the axis bound an instance, which also
indexes the entries along the axis. -/
theorem castShape_reduce_sum_singleton_mk_apply [Add α] [Zero α] (x : Tensor α s) (axis : Nat)
    [hb : Shape.AxisInBounds axis s]
    (h : (ReduceOp.Scalar.seed (op := .sum) (α := α)).isNone →
      ∀ d ∈ [(⟨axis, hb.proof⟩ : Fin s.rank)], 0 < s.extent d)
    (q : s.reduce [⟨axis, hb.proof⟩] false = s.eraseIdx axis) (c : Shape.Coord (s.eraseIdx axis)) :
    castShape (reduce .sum x [⟨axis, hb.proof⟩] false h) q c =
      Tensor.foldl (· + ·) 0
        (Tensor.ofFn fun i : Fin (s.axisSize axis) =>
          x ((Shape.Coord.axisEquiv axis (shape := s)).symm (i, c))) :=
  castShape_reduce_sum_singleton_apply_of_lt x axis hb.proof h q c

/-- A sum over one axis, cast to the shape without that axis, is the tensor of the sums of the
vectors along that axis. -/
theorem castShape_reduce_sum_singleton_mk [Add α] [Zero α] (x : Tensor α s) (axis : Nat)
    [hb : Shape.AxisInBounds axis s]
    (h : (ReduceOp.Scalar.seed (op := .sum) (α := α)).isNone →
      ∀ d ∈ [(⟨axis, hb.proof⟩ : Fin s.rank)], 0 < s.extent d)
    (q : s.reduce [⟨axis, hb.proof⟩] false = s.eraseIdx axis) :
    castShape (reduce .sum x [⟨axis, hb.proof⟩] false h) q =
      Rep.ofFn fun c : Shape.Coord (s.eraseIdx axis) =>
        sum (Tensor.ofFn fun i : Fin (s.axisSize axis) =>
          x ((Shape.Coord.axisEquiv axis (shape := s)).symm (i, c))) :=
  Rep.ext fun c => by
    rw [castShape_reduce_sum_singleton_mk_apply, Rep.get_ofFn, sum_eq_foldl]

/-- An entry of a sum over the axis `dim`, read through a cast to the shape without it, adds the
entries along `dim` to zero in increasing order. -/
theorem castShape_reduce_sum_singleton_apply [Add α] [Zero α] (x : Tensor α s) (dim : Fin s.rank)
    (h : (ReduceOp.Scalar.seed (op := .sum) (α := α)).isNone → ∀ d ∈ [dim], 0 < s.extent d)
    (q : s.reduce [dim] false = s.eraseIdx dim) (c : Shape.Coord (s.eraseIdx dim)) :
    castShape (reduce .sum x [dim] false h) q c =
      Tensor.foldl (· + ·) 0
        (Tensor.ofFn fun i : Fin (s.axisSize dim) =>
          x ((Shape.Coord.axisEquiv dim (shape := s)).symm (i, c))) :=
  castShape_reduce_sum_singleton_apply_of_lt x dim dim.isLt h q c

/-- A sum over the axis `dim`, cast to the shape without it, is the tensor of the sums of the
vectors along `dim`. -/
theorem castShape_reduce_sum_singleton [Add α] [Zero α] (x : Tensor α s) (dim : Fin s.rank)
    (h : (ReduceOp.Scalar.seed (op := .sum) (α := α)).isNone → ∀ d ∈ [dim], 0 < s.extent d)
    (q : s.reduce [dim] false = s.eraseIdx dim) :
    castShape (reduce .sum x [dim] false h) q =
      Rep.ofFn fun c : Shape.Coord (s.eraseIdx dim) =>
        sum (Tensor.ofFn fun i : Fin (s.axisSize dim) =>
          x ((Shape.Coord.axisEquiv dim (shape := s)).symm (i, c))) :=
  castShape_reduce_sum_singleton_mk x dim h q

/-- An entry of a sum over the outer axis, read through a cast to the shape without it, adds the
entries `x (i, c)` to zero in increasing order of `i`. -/
theorem castShape_reduce_sum_zero_apply [Add α] [Zero α] {n : Nat} {inner : Shape}
    (x : Tensor α (.dim n inner))
    (h : (ReduceOp.Scalar.seed (op := .sum) (α := α)).isNone →
      ∀ d ∈ [(⟨0, Shape.AxisInBounds.proof⟩ : Fin (Shape.dim n inner).rank)],
        0 < (Shape.dim n inner).extent d)
    (q : (Shape.dim n inner).reduce [⟨0, Shape.AxisInBounds.proof⟩] false =
      (Shape.dim n inner).eraseIdx 0)
    (c : inner.Coord) :
    castShape (reduce .sum x [⟨0, Shape.AxisInBounds.proof⟩] false h) q c =
      Tensor.foldl (· + ·) 0 (Tensor.ofFn fun i : Fin n => x (i, c)) :=
  castShape_reduce_sum_singleton_mk_apply x 0 h q c

/-- A sum over the outer axis, cast to the shape without it, is the tensor of the sums of the
entries `x (i, c)` over `i`. -/
theorem castShape_reduce_sum_zero [Add α] [Zero α] {n : Nat} {inner : Shape}
    (x : Tensor α (.dim n inner))
    (h : (ReduceOp.Scalar.seed (op := .sum) (α := α)).isNone →
      ∀ d ∈ [(⟨0, Shape.AxisInBounds.proof⟩ : Fin (Shape.dim n inner).rank)],
        0 < (Shape.dim n inner).extent d)
    (q : (Shape.dim n inner).reduce [⟨0, Shape.AxisInBounds.proof⟩] false =
      (Shape.dim n inner).eraseIdx 0) :
    castShape (reduce .sum x [⟨0, Shape.AxisInBounds.proof⟩] false h) q =
      Rep.ofFn fun c : inner.Coord => sum (Tensor.ofFn fun i : Fin n => x (i, c)) :=
  castShape_reduce_sum_singleton_mk x 0 h q

/-- Broadcasting a tensor without axis `axis` back to `s`, placing its axes around `axis`, reads
it at the indices along the other axes. -/
theorem broadcast_eraseIdx_apply (axis : Nat) [Shape.AxisInBounds axis s]
    (x : Tensor α (s.eraseIdx axis)) (c : s.Coord) :
    broadcast x s _ (s.eraseIdx_broadcastDims axis) c =
      x (Shape.Coord.axisEquiv axis (shape := s) c).2 := by
  rw [broadcast_apply_of_extent_eq x _ _ fun i => Shape.extent_eraseIdx i]
  congr 1
  apply Shape.Coord.ext_index
  intro k hk
  rw [Shape.Coord.index_ofFun, dite_eq_left hk, Fin.val_cast, Shape.Coord.val_get,
    Shape.Coord.index_axisEquiv_snd]

/-- The coordinate that `broadcast_apply` reads for a broadcast back along `axis` is the
coordinate of the other axes: an axis of extent one reads index zero, which is its only index. -/
theorem ofFun_broadcast_eraseIdx (axis : Nat) [Shape.AxisInBounds axis s] (c : s.Coord) :
    (Shape.Coord.ofFun fun i : Fin (Shape.rank (s.eraseIdx axis)) =>
      if hi : Shape.extent (s.eraseIdx axis) i = 1 then (⟨0, by omega⟩ : Fin _)
      else Fin.cast (((s.eraseIdx_broadcastDims axis).extent_eq i).resolve_left hi).symm
        (Shape.Coord.get c ⟨if i.val < axis then i.val else i.val + 1,
          Shape.lt_rank_of_lt_rank_eraseIdx i.isLt⟩)) =
      (Shape.Coord.axisEquiv axis (shape := s) c).2 := by
  apply Shape.Coord.ext_index
  intro k hk
  rw [Shape.Coord.index_ofFun, dite_eq_left hk, Shape.Coord.index_axisEquiv_snd]
  by_cases hi : Shape.extent (s.eraseIdx axis) ⟨k, hk⟩ = 1
  · rw [dite_eq_left hi]
    have hlt := Shape.Coord.index_lt c (Shape.lt_rank_of_lt_rank_eraseIdx hk)
    have he := Shape.extent_eraseIdx (s := s) (axis := axis) ⟨k, hk⟩
    rw [hi] at he
    have h1 := lt_of_lt_of_eq hlt he.symm
    simp only at h1 ⊢
    omega
  · rw [dite_eq_right hi, Fin.val_cast, Shape.Coord.val_get]

/-- The sum of a vector over its one axis, cast to the scalar shape, is its total. -/
theorem castShape_reduce_sum_vector [Add α] [Zero α] {n : Nat} (x : Tensor α [n])
    (d : Fin (Shape.rank [n])) (q : Shape.reduce [n] [d] false = []) :
    castShape (reduce .sum x [d] false) q = Tensor.scalar (sum x) := by
  have hd : d = ⟨0, Nat.zero_lt_one⟩ := Fin.ext
      (by have := d.isLt; simp at this; omega)
  subst hd
  apply Rep.ext
  intro c
  have : Shape.AxisInBounds 0 [n] := inferInstance
  erw [castShape_reduce_sum_singleton_mk_apply x 0 _ q c]
  rw [scalar_apply, sum_eq_foldl]
  congr 1
  apply Rep.ext
  rintro ⟨i, u⟩
  cases u
  simp only [Tensor.ofFn, Rep.get_ofFn]
  rfl

/-! ## Pullbacks of reductions -/

/-- Broadcasting `castShape g h` is broadcasting `g` when the axis placements agree along the
cast. -/
theorem broadcast_castShape {s' t : Shape} (g : Tensor α s) (h : s = s')
    (dims : Fin s.rank → Fin t.rank) (dims' : Fin s'.rank → Fin t.rank)
    (hb : Shape.BroadcastDims s t dims) (hb' : Shape.BroadcastDims s' t dims')
    (hdims : ∀ i, (dims' (i.cast (congrArg Shape.rank h))).val = (dims i).val) :
    broadcast (castShape g h) t dims' hb' = broadcast g t dims hb := by
  subst h
  have : dims' = dims := funext fun i => Fin.ext (hdims i)
  subst this
  rfl

/-- The pullback of `reduce op x dims keepdim` at `g`. A sum sends each entry the upstream gradient
of the output entry it is added into. A product sends that gradient times the product of the other
entries of its fiber, in row-major order. A maximum or minimum sends it divided by the number of
entries of the fiber equal to the extremum to each of them and zero to the others. Equal is the
scalar's `==`, with a NaN equal to a NaN extremum, as in PyTorch's even split of the gradient of a
full `max` or `min`. The `torch.amax` page (PyTorch 2.8 docs) says `amax` splits this way too, but
at `[NaN, 1, NaN]` torch 2.8.0's `amax` gives three NaNs, where we and `max` give `[0.5, 0, 0.5]`.
-/
def reduceBackward [Context α] (op : ReduceOp) (x : Tensor α s) (dims : List (Fin s.rank))
    (keepdim : Bool) (h : op = .max ∨ op = .min → ∀ d ∈ dims, 0 < s.extent d)
    (g : Tensor α (s.reduce dims keepdim)) : Tensor α s :=
  let spread := broadcast g s (s.reduceAxes dims keepdim) (s.reduce_broadcastDims dims keepdim)
  let fiber := fun c : s.Coord =>
    (List.finRange (Shape.size (s.selectAxes dims))).map fun i =>
      s.reduceSource dims keepdim (Shape.Coord.ofFun fun j =>
        if hj : (s.reduce dims keepdim).extent j = 1 then ⟨0, by omega⟩
        else Fin.cast (((s.reduce_broadcastDims dims keepdim).extent_eq j).resolve_left hj).symm
          (Shape.Coord.get c (s.reduceAxes dims keepdim j)))
        (Shape.Coord.unlinearize i)
  match op, h with
  | .sum, _ => spread
  | .prod, _ => Rep.ofFn fun c =>
      spread c * (fiber c).foldl (fun acc source => if source = c then acc else acc * x source) 1
  | .max, h =>
      let y := broadcast (Tensor.reduce .max x dims keepdim fun _ => h (Or.inl rfl)) s
        (s.reduceAxes dims keepdim) (s.reduce_broadcastDims dims keepdim)
      Rep.ofFn fun c =>
        if extremumTie (x c) (y c) then
          Context.divNat (spread c)
            ((fiber c).countP fun source => extremumTie (x source) (y c))
        else 0
  | .min, h =>
      let y := broadcast (Tensor.reduce .min x dims keepdim fun _ => h (Or.inr rfl)) s
        (s.reduceAxes dims keepdim) (s.reduce_broadcastDims dims keepdim)
      Rep.ofFn fun c =>
        if extremumTie (x c) (y c) then
          Context.divNat (spread c)
            ((fiber c).countP fun source => extremumTie (x source) (y c))
        else 0

/-- The tangent of `reduce op x dims keepdim` along `dx`, the transpose of `reduceBackward`. A sum
adds the tangents of each fiber. A product adds to zero, in row-major order, each tangent times the
product of the other entries of its fiber. A maximum or minimum adds to zero the tangents of the
entries equal to the extremum (`extremumTie`) and divides that sum once by their number. -/
def reduceJVP [Context α] (op : ReduceOp) (x : Tensor α s) (dims : List (Fin s.rank))
    (keepdim : Bool) (h : op = .max ∨ op = .min → ∀ d ∈ dims, 0 < s.extent d)
    (dx : Tensor α s) : Tensor α (s.reduce dims keepdim) :=
  let fiber := fun o : (s.reduce dims keepdim).Coord =>
    (List.finRange (Shape.size (s.selectAxes dims))).map fun i =>
      s.reduceSource dims keepdim o (Shape.Coord.unlinearize i)
  let average := fun y : Tensor α (s.reduce dims keepdim) => Rep.ofFn fun o =>
    let tie := fun c => extremumTie (x c) (y o)
    Context.divNat ((fiber o).foldl (fun acc c => if tie c then acc + dx c else acc) 0)
      ((fiber o).countP tie)
  match op, h with
  | .sum, _ => Tensor.reduce .sum dx dims keepdim
  | .prod, _ => Rep.ofFn fun o =>
      (fiber o).foldl (fun acc c => acc +
        (fiber o).foldl (fun p source => if source = c then p else p * x source) 1 * dx c) 0
  | .max, h => average (Tensor.reduce .max x dims keepdim fun _ => h (Or.inl rfl))
  | .min, h => average (Tensor.reduce .min x dims keepdim fun _ => h (Or.inr rfl))

/-- The pullback of `mean x dims keepdim` at `g`: the upstream gradient of the output entry each
entry contributes to, divided once (`Context.divNat`) by the number of entries of its fiber, or by
one for an empty fiber. -/
def meanBackward [Context α] (dims : List (Fin s.rank)) (keepdim : Bool)
    (g : Tensor α (s.reduce dims keepdim)) : Tensor α s :=
  let count := Shape.size (s.selectAxes dims)
  map (Context.divNat · (if count = 0 then 1 else count))
    (broadcast g s (s.reduceAxes dims keepdim) (s.reduce_broadcastDims dims keepdim))

end TorchLean.Tensor
