/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Order.Interval.Finset.Fin
public import TorchLean.Tensor.Axes
public import TorchLean.Tensor.Elementwise
public import TorchLean.Tensor.ShapeChange
public import TorchLean.Tensor.Arithmetic
public import TorchLean.Tensor.Rep.Traversal
public import Mathlib.Logic.Equiv.Prod

/-!
# Reductions

`reduce op x dims keepdim` folds `op` over the axes `dims` of `x`, at any rank. Each output entry
is one left fold over the entries of its fiber, in row-major order of `x` (listed axes in
increasing index, whatever their order in `dims`), so a reduction over several axes is one flat
fold, whose rounding can differ from that of a chain of one-axis reductions.
Sums and products start from their seed, and `max` and `min` from the first entry of the fiber,
which they then visit again, as `Softmax.maxVec` does.
The step of `max` and `min` is `TorchLean.Context.maximum` and `TorchLean.Context.minimum`, IEEE
754-2019 9.6 `maximum` and `minimum` in a binary format, `Float` and `Float32` included: a fiber
with a NaN gives its first NaN, quieted, and `+0` is above `-0` in either order.
`sum x` is the one entry of the sum over every axis: every entry of `x` added to zero in row-major
order.

The reductions are folds we wrote by hand. At a tie of `-0` and `0` torch 2.8.0 keeps one of the
two, the first in `torch.amax` (`[-0, 0]` gives `-0`) and the last in a full `torch.max` (`[0, -0]`
gives `-0`), and we give `0` in both orders. Empty fibers sum to `0` and multiply to `1` in both.
Where `torch.amax` raises on one, `reduce .max` and `.min` take a proof `h` that the listed axes are
nonempty. The capture test `test_reductions_are_one_node` in `scripts/checks/test_ir_capture.py`
compares `torch.amax`, `torch.amin`, `sum`, `prod` and `mean` over several axes with eager PyTorch.

* `Shape.sortedAxes`, `Shape.selectAxes`, `Shape.keepAxes` and `Shape.reduce` are the listed axes,
  their shape, the shape with extent one along them, and the output shape.
* `Shape.reduceSource` is the input coordinate read at an output coordinate and a coordinate of
  the reduced axes.
* `reduce_sum_apply`, `reduce_prod_apply`, `reduce_max_apply` and `reduce_min_apply` read an
  output entry as its fold. `reduce_sum_finRange_apply` reads a sum over every axis as the flat
  fold of its input, and `sum_eq_foldl` reads `sum` as that fold. A sum over one axis, cast
  to the shape without that axis (`Shape.reduce_singleton`), is read in
  `TorchLean.Tensor.Reduce.Backward`.
* `maxAbs`, `Tensor.all` and `Tensor.dot` fold every entry, and the row-major fold `foldl` is
  defined with the buffer in `TorchLean.Tensor.Rep`. `mean` sums over the listed axes and divides
  each sum once by the size of its fiber.
* `Coord.axisEquiv` separates the index along one axis from the indices along the
  others, and `Coord.setEquiv` does the same for `shape.set axis count`, the shape with extent
  `count` at that axis. `TorchLean.Tensor.reduceDimWith` applies an aggregator that is not a
  `ReduceOp`, such as a directed-rounding sum or `maxAbs`, to the vectors along one axis.
-/

@[expose] public section

open FloatLib.Numerics (MathFunctions)
open TorchLean.Shape (rank rank_cons rank_eq_length size)
open TorchLean.Shape (size_eq_prod)

namespace TorchLean.Shape

/-! ## Reduced axes and output shapes -/

/-- The axes of `s` listed in `dims`, in increasing order and each once. -/
def sortedAxes (s : Shape) (dims : List (Fin s.rank)) : List (Fin s.rank) :=
  (List.finRange s.rank).filter (· ∈ dims)

/-- The sorted axes outside `dims` are the axes outside `dims`. -/
theorem sortedAxes_otherAxes (s : Shape) (dims : List (Fin s.rank)) :
    s.sortedAxes (s.otherAxes dims) = s.otherAxes dims :=
  List.filter_congr fun a _ => by simp

/-- An axis is among the sorted axes exactly when it is listed. -/
@[simp] theorem mem_sortedAxes {s : Shape} {dims : List (Fin s.rank)} {d : Fin s.rank} :
    d ∈ s.sortedAxes dims ↔ d ∈ dims := by
  simp [sortedAxes]

/-- The sorted axes are strictly increasing. -/
theorem pairwise_sortedAxes (s : Shape) (dims : List (Fin s.rank)) :
    (s.sortedAxes dims).Pairwise (· < ·) :=
  (List.pairwise_lt_finRange s.rank).filter _

/-- The other axes are strictly increasing. -/
theorem pairwise_otherAxes (s : Shape) (dims : List (Fin s.rank)) :
    (s.otherAxes dims).Pairwise (· < ·) :=
  (List.pairwise_lt_finRange s.rank).filter _

/-- The shape of the axes of `s` listed in `dims`, in their order in `s`. -/
def selectAxes (s : Shape) (dims : List (Fin s.rank)) : Shape :=
  (s.sortedAxes dims).map s.extent

/-- The shape `s` with extent one along every axis listed in `dims`. -/
def keepAxes (s : Shape) (dims : List (Fin s.rank)) : Shape :=
  List.ofFn fun d => if d ∈ dims then 1 else s.extent d

/-- The output shape of a reduction over `dims`: the other axes, or with `keepdim` the shape `s`
with extent one along `dims`. -/
def reduce (s : Shape) (dims : List (Fin s.rank)) (keepdim : Bool) : Shape :=
  bif keepdim then s.keepAxes dims else s.eraseAxes dims

/-- With every axis listed, the sorted axes are all the axes in increasing order. -/
@[simp] theorem sortedAxes_finRange (s : Shape) :
    s.sortedAxes (List.finRange s.rank) = List.finRange s.rank :=
  List.filter_eq_self.mpr (by simp)

/-- No axis is outside all axes. -/
@[simp] theorem otherAxes_finRange (s : Shape) : s.otherAxes (List.finRange s.rank) = [] :=
  List.filter_eq_nil_iff.mpr (by simp)

/-- A reduction over every axis has one entry. -/
@[simp] theorem size_reduce_finRange (s : Shape) (keepdim : Bool) :
    (s.reduce (List.finRange s.rank) keepdim).size = 1 := by
  cases keepdim
  · rw [reduce, Bool.cond_false, eraseAxes, otherAxes_finRange]
    rfl
  · rw [reduce, Bool.cond_true, keepAxes, Shape.size_eq_prod]
    simp

/-- One listed axis is its own sorted list. -/
@[simp] theorem sortedAxes_singleton (s : Shape) (d : Fin s.rank) : s.sortedAxes [d] = [d] := by
  have hnd : (s.sortedAxes [d]).Nodup := (List.nodup_finRange s.rank).filter _
  have hall : ∀ b ∈ s.sortedAxes [d], b = d := fun b hb => by simpa using hb
  have hrep := List.eq_replicate_iff.mpr ⟨rfl, hall⟩
  rw [hrep] at hnd
  have hle := List.nodup_replicate.mp hnd
  have hpos : 0 < (s.sortedAxes [d]).length :=
    List.length_pos_of_mem (mem_sortedAxes.mpr (List.mem_singleton_self d))
  rw [hrep, show (s.sortedAxes [d]).length = 1 by omega]
  rfl

/-- The extents of all the axes of `s` make up `s`. -/
theorem map_extent_finRange (s : Shape) : (List.finRange s.rank).map s.extent = s := by
  apply List.ext_getElem
  · simp [Shape.rank_eq_length]
  · intro k h₁ h₂
    simp [extent]

/-- Reducing over every axis selects the whole shape. -/
@[simp] theorem selectAxes_finRange (s : Shape) : s.selectAxes (List.finRange s.rank) = s := by
  rw [selectAxes, sortedAxes_finRange, map_extent_finRange]

/-- The selected shape of one axis is its extent. -/
@[simp] theorem selectAxes_singleton (s : Shape) (d : Fin s.rank) :
    s.selectAxes [d] = [s.extent d] := by
  simp [selectAxes]

/-- A reduced shape has a positive number of entries when every listed axis is nonempty. -/
theorem size_selectAxes_pos {s : Shape} {dims : List (Fin s.rank)}
    (h : ∀ d ∈ dims, 0 < s.extent d) :
    0 < Shape.size (s.selectAxes dims) := by
  rw [Shape.size_eq_prod, List.prod_pos_iff_forall_pos_nat]
  intro n hn
  simp only [selectAxes, List.mem_map, mem_sortedAxes] at hn
  obtain ⟨d, hd, rfl⟩ := hn
  exact h d hd

/-! ## Positions of axes -/

/-- In a strictly increasing list of axes, an axis sits at the number of entries below it. -/
theorem idxOf_eq_length_filter_lt {n : Nat} {l : List (Fin n)} (hl : l.Pairwise (· < ·))
    {a : Fin n} (ha : a ∈ l) : l.idxOf a = (l.filter (· < a)).length := by
  induction l with
  | nil => simp at ha
  | cons b l ih =>
      rw [List.pairwise_cons] at hl
      by_cases hb : b = a
      · subst hb
        rw [List.idxOf_cons_self, List.filter_cons_of_neg (by simp)]
        have : l.filter (· < b) = [] :=
          List.filter_eq_nil_iff.mpr fun x hx => by simpa using (hl.1 x hx).le
        rw [this]
        rfl
      · have haL : a ∈ l := (List.mem_cons.mp ha).resolve_left (Ne.symm hb)
        have hba : b < a := hl.1 a haL
        rw [List.idxOf_cons, ite_eq_right (by simpa using hb), ih hl.2 haL,
          List.filter_cons_of_pos (by simpa using hba)]
        rfl

/-- The index along a listed axis is below its extent. -/
theorem Coord.index_idxOf_lt {s : Shape} {l : List (Fin s.rank)} (r : Coord (l.map s.extent))
    {d : Fin s.rank} (hd : d ∈ l) : Coord.index r (l.idxOf d) < s.extent d := by
  have hk : l.idxOf d < l.length := List.idxOf_lt_length_iff.mpr hd
  have hr : l.idxOf d < Shape.rank (l.map s.extent) := by simpa [Shape.rank_eq_length] using hk
  have hlt := Coord.index_lt r hr
  have he : extent (l.map s.extent) ⟨_, hr⟩ = s.extent d := by
    simp only [extent, List.getElem_map, List.getElem_idxOf]
  omega

/-- The index of a coordinate of `s.keepAxes dims` along an unlisted axis is below its extent
in `s`. -/
theorem Coord.index_lt_keepAxes {s : Shape} {dims : List (Fin s.rank)}
    (o : (s.keepAxes dims).Coord) {d : Fin s.rank} (hd : d ∉ dims) :
    Coord.index o d.val < s.extent d := by
  have hr : d.val < (s.keepAxes dims).rank := by simpa [keepAxes, Shape.rank_eq_length] using d.isLt
  have hlt := Coord.index_lt o hr
  have he : (s.keepAxes dims).extent ⟨d.val, hr⟩ = s.extent d := by
    simp only [extent, keepAxes, List.getElem_ofFn]
    simp [hd]
  omega

/-- The position in an output coordinate of a reduction of the unlisted axis `d`. -/
def keptIndex (s : Shape) (dims : List (Fin s.rank)) (keepdim : Bool) (d : Fin s.rank) : Nat :=
  bif keepdim then d.val else (s.otherAxes dims).idxOf d

/-- The index of an output coordinate of a reduction along an unlisted axis is below its extent
in `s`. -/
theorem Coord.index_keptIndex_lt {s : Shape} {dims : List (Fin s.rank)} (keepdim : Bool)
    (o : (s.reduce dims keepdim).Coord) {d : Fin s.rank} (hd : d ∉ dims) :
    Coord.index o (s.keptIndex dims keepdim d) < s.extent d := by
  cases keepdim
  · exact Coord.index_idxOf_lt (l := s.otherAxes dims) o (mem_otherAxes.mpr hd)
  · exact Coord.index_lt_keepAxes o hd

/-- The coordinate of `s` read by a reduction over `dims` at the output coordinate `o` and the
reduced coordinate `r`: the indices of `r` along the listed axes, in increasing order, and those
of `o` along the other axes. -/
def reduceSource (s : Shape) (dims : List (Fin s.rank)) (keepdim : Bool)
    (o : (s.reduce dims keepdim).Coord) (r : (s.selectAxes dims).Coord) : s.Coord :=
  Coord.ofFun fun d =>
    if hd : d ∈ dims then
      ⟨Coord.index r ((s.sortedAxes dims).idxOf d),
        Coord.index_idxOf_lt (l := s.sortedAxes dims) r (mem_sortedAxes.mpr hd)⟩
    else ⟨Coord.index o (s.keptIndex dims keepdim d), Coord.index_keptIndex_lt keepdim o hd⟩

/-- The indices of `reduceSource`. -/
theorem index_reduceSource {s : Shape} (dims : List (Fin s.rank)) (keepdim : Bool)
    (o : (s.reduce dims keepdim).Coord) (r : (s.selectAxes dims).Coord) (k : Nat) :
    Coord.index (s.reduceSource dims keepdim o r) k =
      if h : k < s.rank then
        if (⟨k, h⟩ : Fin s.rank) ∈ dims then
          Coord.index r ((s.sortedAxes dims).idxOf ⟨k, h⟩)
        else Coord.index o (s.keptIndex dims keepdim ⟨k, h⟩)
      else 0 := by
  rw [reduceSource, Coord.index_ofFun]
  split_ifs <;> rfl

/-! ## Erasing one axis -/

/-- Erasing an axis shifts the later extents down, and an out-of-range axis erases nothing. -/
private theorem getElem?_eraseIdx_ite (s : Shape) (axis k : Nat) :
    (s.eraseIdx axis)[k]? = s[if k < axis then k else k + 1]? := by
  rw [List.getElem?_eraseIdx]
  by_cases h : k < axis <;> simp [h]

/-- The axes of `s.eraseIdx axis` sit at positions `i` or `i + 1` of `s`. -/
theorem lt_rank_of_lt_rank_eraseIdx {s : Shape} {axis i : Nat}
    (h : i < Shape.rank (s.eraseIdx axis)) : (if i < axis then i else i + 1) < s.rank := by
  have key := getElem?_eraseIdx_ite s axis i
  rw [Shape.rank_eq_length] at h ⊢
  rw [List.getElem?_eq_getElem h] at key
  exact (List.getElem?_eq_some_iff.mp key.symm).1

/-- Axis `i` of `s.eraseIdx axis` has the extent of axis `i` or `i + 1` of `s`. -/
theorem extent_eraseIdx {s : Shape} {axis : Nat}
    (i : Fin (Shape.rank (s.eraseIdx axis))) :
    Shape.extent (s.eraseIdx axis) i =
      s.extent ⟨if i.val < axis then i.val else i.val + 1,
        Shape.lt_rank_of_lt_rank_eraseIdx i.isLt⟩ := by
  have key := getElem?_eraseIdx_ite s axis i.val
  simp only [Shape.extent]
  rw [← Option.some_inj, ← List.getElem?_eq_getElem, ← List.getElem?_eq_getElem, key]

/-- The axes of `s.eraseIdx axis` keep their extents in `s`, placed around `axis`. -/
theorem eraseIdx_broadcastDims (s : Shape) (axis : Nat) :
    Shape.BroadcastDims (s.eraseIdx axis) s
      (fun i => ⟨if i.val < axis then i.val else i.val + 1,
        Shape.lt_rank_of_lt_rank_eraseIdx i.isLt⟩) where
  strictMono a b hab := by
    simp only [Fin.lt_def] at hab ⊢
    split_ifs <;> omega
  extent_eq i := Or.inr (Shape.extent_eraseIdx i)

/-! ## One reduced axis -/

/-- With one axis `d` listed, another axis `a` sits at its own index, or one lower past `d`. -/
theorem idxOf_otherAxes_singleton (s : Shape) {d a : Fin s.rank} (ha : a ≠ d) :
    (s.otherAxes [d]).idxOf a = if a < d then a.val else a.val - 1 := by
  rw [idxOf_eq_length_filter_lt (s.pairwise_otherAxes [d])
      (mem_otherAxes.mpr (by simpa using ha)),
    ← List.toFinset_card_of_nodup ((s.nodup_otherAxes [d]).filter _)]
  have he : ((s.otherAxes [d]).filter (· < a)).toFinset = (Finset.Iio a).erase d := by
    ext b
    simp only [List.mem_toFinset, List.mem_filter, mem_otherAxes, List.mem_singleton,
      decide_eq_true_eq, Finset.mem_erase, Finset.mem_Iio]
  rw [he, Finset.card_erase_eq_ite, Fin.card_Iio]
  have hne : a.val ≠ d.val := fun h => ha (Fin.ext h)
  simp only [Finset.mem_Iio, Fin.lt_def]
  split_ifs <;> omega

/-- With one axis listed, every other axis is left. -/
theorem length_otherAxes_singleton (s : Shape) (d : Fin s.rank) :
    (s.otherAxes [d]).length = s.rank - 1 := by
  rw [← List.toFinset_card_of_nodup (s.nodup_otherAxes [d])]
  have he : (s.otherAxes [d]).toFinset = Finset.univ.erase d := by
    ext b
    simp [otherAxes]
  rw [he, Finset.card_erase_of_mem (Finset.mem_univ d), Finset.card_univ, Fintype.card_fin]

/-- With one axis `d` listed, the other axis at position `k` is axis `k`, or `k + 1` past `d`. -/
theorem val_getElem_otherAxes_singleton (s : Shape) (d : Fin s.rank) {k : Nat}
    (hk : k < (s.otherAxes [d]).length) :
    ((s.otherAxes [d])[k]).val = if k < d.val then k else k + 1 := by
  have hidx : (s.otherAxes [d]).idxOf (s.otherAxes [d])[k] = k :=
    (s.nodup_otherAxes [d]).idxOf_getElem k hk
  have hne : (s.otherAxes [d])[k] ≠ d := by
    simpa using mem_otherAxes.mp (List.getElem_mem hk)
  rw [s.idxOf_otherAxes_singleton hne] at hidx
  have hne' : ((s.otherAxes [d])[k]).val ≠ d.val := fun h => hne (Fin.ext h)
  simp only [Fin.lt_def] at hidx
  split_ifs at hidx ⊢ <;> omega

/-- Erasing one listed axis is `List.eraseIdx`. -/
theorem eraseAxes_singleton (s : Shape) (d : Fin s.rank) :
    s.eraseAxes [d] = s.eraseIdx d.val := by
  apply List.ext_getElem
  · rw [eraseAxes, List.length_map, length_otherAxes_singleton, ← Shape.rank_eq_length,
      rank_eraseIdx d.isLt]
  · intro k h₁ h₂
    have hk : k < (s.otherAxes [d]).length := by simpa [eraseAxes] using h₁
    have he := extent_eraseIdx (s := s) (axis := d.val) ⟨k, by rwa [Shape.rank_eq_length]⟩
    simp only [extent] at he
    simp only [eraseAxes, List.getElem_map, he, extent, val_getElem_otherAxes_singleton s d hk]

/-- Reducing one axis without keeping it leaves the shape without that axis.
`castShape_reduce_sum_singleton_apply` reads through a cast along this equality. -/
theorem reduce_singleton (s : Shape) (d : Fin s.rank) : s.reduce [d] false = s.eraseIdx d.val :=
  s.eraseAxes_singleton d

/-- A reduction over every axis without `keepdim` leaves no axis. -/
theorem reduce_finRange (s : Shape) : s.reduce (List.finRange s.rank) false = [] := by
  rw [reduce, Bool.cond_false, eraseAxes, otherAxes_finRange]
  rfl

end TorchLean.Shape

namespace TorchLean.Tensor

/-! ## The reduction -/

/-- The reduction that `reduce` applies. -/
inductive ReduceOp
  /-- Sum, from zero. -/
  | sum
  /-- Product, from one. -/
  | prod
  /-- Maximum, from the first entry. -/
  | max
  /-- Minimum, from the first entry. -/
  | min
  deriving DecidableEq, Repr

/-- The scalar step of a reduction and its seed. A reduction without a seed starts from the first
entry of the fiber. -/
class ReduceOp.Scalar (op : ReduceOp) (α : Type) where
  /-- One step of the fold. -/
  step : α → α → α
  /-- The starting value, if the reduction has one. -/
  seed : Option α

/-- A sum adds, from zero. -/
instance ReduceOp.scalarSum {α : Type} [Add α] [Zero α] : ReduceOp.Scalar .sum α :=
  ⟨(· + ·), some 0⟩

/-- A product multiplies, from one. -/
instance ReduceOp.scalarProd {α : Type} [Mul α] [One α] : ReduceOp.Scalar .prod α :=
  ⟨(· * ·), some 1⟩

/-- A maximum takes `Context.maximum`, IEEE maximum in a binary format, with no seed. -/
instance ReduceOp.scalarMax {α : Type} [Context α] : ReduceOp.Scalar .max α :=
  ⟨Context.maximum, none⟩

/-- A minimum takes `Context.minimum`, IEEE minimum in a binary format, with no seed. -/
instance ReduceOp.scalarMin {α : Type} [Context α] : ReduceOp.Scalar .min α :=
  ⟨Context.minimum, none⟩

/-- The seed of a sum is zero. -/
@[simp] theorem ReduceOp.seed_sum {α : Type} [Add α] [Zero α] :
    ReduceOp.Scalar.seed (op := .sum) (α := α) = some 0 :=
  rfl

/-- The seed of a product is one. -/
@[simp] theorem ReduceOp.seed_prod {α : Type} [Mul α] [One α] :
    ReduceOp.Scalar.seed (op := .prod) (α := α) = some 1 :=
  rfl

/-- A maximum has no seed. -/
@[simp] theorem ReduceOp.seed_max {α : Type} [Context α] :
    ReduceOp.Scalar.seed (op := .max) (α := α) = none :=
  rfl

/-- A minimum has no seed. -/
@[simp] theorem ReduceOp.seed_min {α : Type} [Context α] :
    ReduceOp.Scalar.seed (op := .min) (α := α) = none :=
  rfl

section

variable {α : Type} [Storage α] {s : Shape}

/-- Reduce `x` over the axes `dims` with `op`. Each output entry is one left fold of the step of
`op` over the entries of its fiber in row-major order of `x`, from the seed of `op`, or else from
the first entry of the fiber, which the fold then visits again. A reduction without a seed needs
every listed axis nonempty. `keepdim` keeps the listed axes with extent one. -/
def reduce (op : ReduceOp) [ReduceOp.Scalar op α] (x : Tensor α s) (dims : List (Fin s.rank))
    (keepdim : Bool)
    (h : (ReduceOp.Scalar.seed (op := op) (α := α)).isNone → ∀ d ∈ dims, 0 < s.extent d := by
      simp) :
    Tensor α (s.reduce dims keepdim) :=
  Rep.ofFn fun o =>
    let value := fun i : Fin (Shape.size (s.selectAxes dims)) =>
      x (s.reduceSource dims keepdim o (Shape.Coord.unlinearize i))
    match hs : ReduceOp.Scalar.seed (op := op) (α := α) with
    | some seed =>
        (List.finRange _).foldl (fun acc i => ReduceOp.Scalar.step (op := op) acc (value i)) seed
    | none =>
        (List.finRange _).foldl (fun acc i => ReduceOp.Scalar.step (op := op) acc (value i))
          (value ⟨0, Shape.size_selectAxes_pos (h (by simp [hs]))⟩)

/-- An entry of a sum over `dims` adds the entries of its fiber to zero in row-major order. -/
@[simp] theorem reduce_sum_apply [Add α] [Zero α] (x : Tensor α s) (dims : List (Fin s.rank))
    (keepdim : Bool)
    (h : (ReduceOp.Scalar.seed (op := .sum) (α := α)).isNone → ∀ d ∈ dims, 0 < s.extent d)
    (o : (s.reduce dims keepdim).Coord) :
    reduce .sum x dims keepdim h o =
      (List.finRange (Shape.size (s.selectAxes dims))).foldl
        (fun acc i => acc + x (s.reduceSource dims keepdim o (Shape.Coord.unlinearize i))) 0 :=
  Rep.get_ofFn _ o

/-- An entry of a product over `dims` multiplies the entries of its fiber into one in row-major
order. -/
@[simp] theorem reduce_prod_apply [Mul α] [One α] (x : Tensor α s) (dims : List (Fin s.rank))
    (keepdim : Bool)
    (h : (ReduceOp.Scalar.seed (op := .prod) (α := α)).isNone → ∀ d ∈ dims, 0 < s.extent d)
    (o : (s.reduce dims keepdim).Coord) :
    reduce .prod x dims keepdim h o =
      (List.finRange (Shape.size (s.selectAxes dims))).foldl
        (fun acc i => acc * x (s.reduceSource dims keepdim o (Shape.Coord.unlinearize i))) 1 :=
  Rep.get_ofFn _ o

/-- An entry of a maximum over `dims` folds `Context.maximum` over the entries of its fiber in
row-major order, starting from the first entry. -/
@[simp] theorem reduce_max_apply [Context α] (x : Tensor α s) (dims : List (Fin s.rank))
    (keepdim : Bool)
    (h : (ReduceOp.Scalar.seed (op := .max) (α := α)).isNone → ∀ d ∈ dims, 0 < s.extent d)
    (o : (s.reduce dims keepdim).Coord) :
    reduce .max x dims keepdim h o =
      (List.finRange (Shape.size (s.selectAxes dims))).foldl
        (fun acc i =>
          Context.maximum acc (x (s.reduceSource dims keepdim o (Shape.Coord.unlinearize i))))
        (x (s.reduceSource dims keepdim o (Shape.Coord.unlinearize
          ⟨0, Shape.size_selectAxes_pos (h rfl)⟩))) :=
  Rep.get_ofFn _ o

/-- An entry of a minimum over `dims` folds `Context.minimum` over the entries of its fiber in
row-major order, starting from the first entry. -/
@[simp] theorem reduce_min_apply [Context α] (x : Tensor α s) (dims : List (Fin s.rank))
    (keepdim : Bool)
    (h : (ReduceOp.Scalar.seed (op := .min) (α := α)).isNone → ∀ d ∈ dims, 0 < s.extent d)
    (o : (s.reduce dims keepdim).Coord) :
    reduce .min x dims keepdim h o =
      (List.finRange (Shape.size (s.selectAxes dims))).foldl
        (fun acc i =>
          Context.minimum acc (x (s.reduceSource dims keepdim o (Shape.Coord.unlinearize i))))
        (x (s.reduceSource dims keepdim o (Shape.Coord.unlinearize
          ⟨0, Shape.size_selectAxes_pos (h rfl)⟩))) :=
  Rep.get_ofFn _ o

/-! ## Sums as reductions -/

/-- Two folds over the coordinates of equal shapes agree when their steps agree at coordinates
with the same indices. -/
theorem foldl_finRange_congr {β : Type} {t u : Shape} (h : t = u) (f : β → t.Coord → β)
    (g : β → u.Coord → β)
    (hfg : ∀ a (c : t.Coord) (c' : u.Coord),
      (∀ k, Shape.Coord.index c k = Shape.Coord.index c' k) → f a c = g a c')
    (b : β) :
    (List.finRange (Shape.size t)).foldl
        (fun a i => f a (Shape.Coord.unlinearize i)) b =
      (List.finRange (Shape.size u)).foldl
        (fun a i => g a (Shape.Coord.unlinearize i)) b := by
  subst h
  have hfg' : f = g := funext fun a => funext fun c => hfg a c c fun _ => rfl
  rw [hfg']


/-- An entry of a sum over every axis adds every entry of `x` to zero in row-major order, at any
output coordinate and with or without `keepdim`. -/
theorem reduce_sum_finRange_apply [Add α] [Zero α] (x : Tensor α s) (keepdim : Bool)
    (h : (ReduceOp.Scalar.seed (op := .sum) (α := α)).isNone →
      ∀ d ∈ List.finRange s.rank, 0 < s.extent d)
    (o : (s.reduce (List.finRange s.rank) keepdim).Coord) :
    reduce .sum x (List.finRange s.rank) keepdim h o = Tensor.foldl (· + ·) 0 x := by
  rw [reduce_sum_apply, Rep.foldl_eq_fin_foldl, Fin.foldl_eq_finRange_foldl]
  simp only [Rep.getFlat_eq_get]
  refine foldl_finRange_congr s.selectAxes_finRange
    (fun a r => a + x (s.reduceSource (List.finRange s.rank) keepdim o r)) (fun a c => a + x c)
    ?_ 0
  intro a r c hrc
  show a + x _ = a + x _
  have hsrc : s.reduceSource (List.finRange s.rank) keepdim o r = c := by
    apply Shape.Coord.ext_index
    intro k hk
    rw [Shape.index_reduceSource, dite_eq_left hk, ite_eq_left (List.mem_finRange _),
      Shape.sortedAxes_finRange, List.idxOf_finRange]
    exact hrc k
  rw [hsrc]


/-- A map of scalars that preserves zero and addition commutes with a sum over any axes. -/
theorem map_reduce_sum {β : Type} [Storage β] [Add α] [Zero α] [Add β] [Zero β] (f : α → β)
    (zero : f 0 = 0) (add : ∀ a b, f (a + b) = f a + f b) (x : Tensor α s)
    (dims : List (Fin s.rank)) (keepdim : Bool) :
    map f (reduce .sum x dims keepdim) = reduce .sum (map f x) dims keepdim := by
  refine Rep.ext fun o => ?_
  simp only [Tensor.map_apply, reduce_sum_apply]
  rw [← zero]
  refine (List.foldl_hom f fun a i => ?_).symm
  exact (add a _).symm

/-- A map of scalars commutes with a shape cast. -/
theorem map_castShape {β : Type} [Storage β] (f : α → β) {t u : Shape} (x : Tensor α t)
    (h : t = u) : map f (castShape x h) = castShape (map f x) h := by
  subst h
  rfl

end

/-! ## Folds over every entry -/

variable {α : Type} [TorchLean.Storage α] [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)]

/-- Largest absolute entry, or zero for an empty tensor. The first entry not equal to itself (a
NaN) is the result if there is one, since the scalar `max` can drop a NaN. -/
def maxAbs {α : Type} [TorchLean.Storage α] [MathFunctions α] [BEq α] [Max α] [Zero α]
    {shape : Shape} (tensor : Tensor α shape) : α :=
  foldl (fun largest value =>
    if !(largest == largest) then largest
    else if !(value == value) then value
    else max largest (MathFunctions.abs value)) 0 tensor

/-- Folding a scalar tensor visits its single element once. -/
@[simp] theorem foldl_scalar {α β : Type} [TorchLean.Storage α]
    (f : β → α → β) (init : β) (value : α) :
    foldl f init (Tensor.scalar value) = f init value := by
  rw [Tensor.Rep.foldl_eq_data_foldl]
  simp only [Tensor.scalar, Tensor.Rep.data, Tensor.Rep.ofFn,
    Tensor.Rep.ofFlatFn, TorchLean.Storage.toArray_ofFn]
  rw [← Array.foldl_toList]
  simp

namespace foldlEntries

/--
Proof-facing fold over a family of equal-shaped slices, starting at slice `index`.

`foldl` traverses the packed buffer, and `foldl_dim` reads its fold of a stacked tensor as this
fold over the slices, which suits proofs by induction on the shape.
-/
def go {α β : Type} [TorchLean.Storage α]
    (f : β → α → β) (length : Nat) (shape : Shape)
    (values : Fin length → Tensor α shape) (index : Nat) (accumulator : β) : β :=
  ((List.finRange length).drop index).foldl
    (fun value component => foldl f value (values component))
    accumulator

/-- Starting the proof-facing slice fold at zero is the ordinary finite fold. -/
theorem go_zero_eq_fin_foldl {α β : Type} [TorchLean.Storage α]
    (f : β → α → β) (length : Nat) (shape : Shape)
    (values : Fin length → Tensor α shape) (accumulator : β) :
    go f length shape values 0 accumulator =
      Fin.foldl length
        (fun value component => foldl f value (values component))
        accumulator := by
  simp [go, Fin.foldl_eq_foldl_finRange]

end foldlEntries

/-- Folding a tensor with a leading dimension folds its slices in row-major order. -/
@[simp] theorem foldl_dim {α β : Type} [TorchLean.Storage α]
    (f : β → α → β) (init : β) {length : Nat} {shape : Shape}
    (values : Fin length → Tensor α shape) :
    foldl f init (Tensor.dim values) =
      foldlEntries.go f length shape values 0 init := by
  rw [Tensor.Rep.foldl_stack,
    foldlEntries.go_zero_eq_fin_foldl]

/-- A loop over `Fin n` is the fold, in the same increasing order, over the coordinates of a
vector of length `n`, for any step and accumulator. -/
theorem fin_foldl_eq_coordinates {δ : Type} (n : Nat) (step : δ → Fin n → δ)
    (initial : δ) :
    Fin.foldl n step initial =
      (Rep.coordinates [n]).foldl (fun state coordinate => step state coordinate.1)
        initial := by
  rw [Rep.foldl_eq_fin_foldl]
  simp only [Rep.coordinates, Rep.ofFn, Rep.getFlat_ofFlatFn]
  have size : Shape.size [n] = n := by simp [Shape.size]
  rw [Fin.foldl_congr size]
  congr 1
  funext state index
  congr 1
  apply Fin.ext
  simp [Shape.Coord.unlinearizeFast_eq, Shape.Coord.unlinearizeFast]
  rfl

/-! ## The sum of every entry -/

/-- The sum of every entry of `x`, added to zero in row-major order: the one entry of the sum over
every axis. -/
def sum {α : Type} [Storage α] [Add α] [Zero α] {s : Shape} (x : Tensor α s) : α :=
  item (castShape (reduce .sum x (List.finRange s.rank) false) (Shape.reduce_finRange s))

/-- `sum x` adds every entry of `x` to zero in row-major order. -/
theorem sum_eq_foldl {α : Type} [Storage α] [Add α] [Zero α] {s : Shape} (x : Tensor α s) :
    sum x = foldl (· + ·) 0 x := by
  rw [sum, item, castShape_apply, reduce_sum_finRange_apply]

/-- The flat fold over the storage that `sum` compiles to. -/
def sumFast {α : Type} [Storage α] [Add α] [Zero α] {s : Shape} (x : Tensor α s) : α :=
  foldl (· + ·) 0 x

/-- `sum` runs as the flat fold over the storage. -/
@[csimp] theorem sumFast_eq : @sum = @sumFast := by
  funext α _ _ _ s x
  exact sum_eq_foldl x

/-- The sum of a scalar tensor is its value. -/
@[simp] theorem sum_scalar {α : Type} [TorchLean.Storage α]
    [AddZeroClass α] (value : α) :
    sum (Tensor.scalar value) = value := by
  simp [sum_eq_foldl]

/-- A map of scalars that preserves zero and addition commutes with a tensor sum. -/
theorem map_sum {α β : Type} [Storage α] [Add α] [Zero α] [Storage β] [Add β] [Zero β]
    (f : α → β) (zero : f 0 = 0) (add : ∀ a b, f (a + b) = f a + f b) {s : Shape}
    (t : Tensor α s) : f (sum t) = sum (map f t) := by
  rw [sum_eq_foldl, sum_eq_foldl,
    Rep.map_foldl t (· + ·) 0 f f (· + ·) add, zero]

/-- `true` if all entries satisfy `p`. -/
def all {s : Shape} (p : α → Bool) (t : Tensor α s) : Bool :=
  foldl (fun acc x => acc && p x) true t

/-- Dot product in row-major order, with a zero-seeded left fold. -/
def dot {α : Type} [TorchLean.Storage α] [Add α] [Mul α] [Zero α]
    {s : Shape} (a b : Tensor α s) : α :=
  sum (a * b)

-- Means and the mean squared error.

/-- Denominator of a totalized mean: the element count, or one for an empty shape.

Dividing by the raw size would make the mean of an empty tensor depend on the scalar type's
convention for `x / 0`. With one, the mean of an empty tensor is its zero sum, as in the loss layer
and in `TorchLean.Tensor.mean`, which divides by one for an empty fiber. -/
def meanDenominator (s : Shape) : Nat :=
  if s.size = 0 then 1 else s.size

/-- On a nonempty shape the mean denominator is the element count. -/
theorem meanDenominator_of_size_pos {s : Shape} (h : 0 < s.size) :
    meanDenominator s = s.size := by
  rw [meanDenominator, ite_eq_right (Nat.pos_iff_ne_zero.mp h)]

/-- On an empty shape the mean denominator is one. -/
theorem meanDenominator_of_size_eq_zero {s : Shape} (h : s.size = 0) :
    meanDenominator s = 1 := by
  rw [meanDenominator, ite_eq_left h]

/-- The mean denominator is never zero. -/
theorem meanDenominator_pos (s : Shape) : 0 < meanDenominator s := by
  unfold meanDenominator
  split <;> grind

/-- The mean of `x` over the axes `dims` (`torch.mean(x, dims, keepdim)`): the sum
`reduce .sum` of each fiber, divided once by its number of entries (`Context.divNat`), or by one
for an empty fiber. The mean of every entry is `Context.divNat (sum x) (meanDenominator s)`
(`mean_finRange_item`). The `torch.mean` page (PyTorch 2.8 docs) makes the mean of an empty tensor
a NaN, so the importer refuses a mean over an empty fiber and the exporter writes the sum there
(`TorchLean.NN.PyTorch.Export.IR`). -/
def mean {α : Type} [Storage α] [Context α] {s : Shape}
    (x : Tensor α s) (dims : List (Fin s.rank)) (keepdim : Bool) :
    Tensor α (s.reduce dims keepdim) :=
  let count := Shape.size (s.selectAxes dims)
  map (Context.divNat · (if count = 0 then 1 else count)) (reduce .sum x dims keepdim)

/-- The mean over every axis, read at its one entry, is the sum of every entry divided by
`meanDenominator s`: the same row-major sum and the same single division. -/
theorem mean_finRange_item {α : Type} [Storage α] [Context α]
    {s : Shape} (x : Tensor α s) :
    item (castShape (mean x (List.finRange s.rank) false) (Shape.reduce_finRange s)) =
      Context.divNat (sum x) (meanDenominator s) := by
  have hcount : (if Shape.size (s.selectAxes (List.finRange s.rank)) = 0 then 1
      else Shape.size (s.selectAxes (List.finRange s.rank))) = meanDenominator s := by
    rw [Shape.selectAxes_finRange, meanDenominator]
  rw [mean, ← map_castShape, item_map, sum, hcount]

/-- Mean squared difference between two equally shaped tensors.

Subtraction and squaring are pointwise, and the squares add in row-major order. A finite sum is
divided once by the totalized denominator `n = meanDenominator shape`, including for empty shapes.
When every difference `d` is finite but the sum overflows, the result is the row-major sum of
`d * (d / n)`, each quotient rounded once (`Context.divNat`), whose exact terms do not exceed the
exact mean. Binary32 differences `[2e19, 0, 0, 0]`, whose sum of squares overflows, give `1e38`.
A nonfinite difference keeps the division of its sum. Torch 2.8.0's `mse_loss` gives `+∞` on that
input and a NaN on empty tensors, where we give `0`. -/
def meanSquaredError {α : Type} [TorchLean.Storage α] [Context α]
    {shape : Shape} (predicted target : Tensor α shape) : α :=
  let diff := predicted - target
  let total := sum (diff * diff)
  if Context.isFinite total || !all Context.isFinite diff then
    Context.divNat total (meanDenominator shape)
  else sum (map (fun d => d * Context.divNat d (meanDenominator shape)) diff)

/-- A mean squared error whose sum of squares is finite, as every real one is, is that sum divided
once by `meanDenominator`. -/
theorem meanSquaredError_of_isFinite {α : Type} [TorchLean.Storage α] [Context α]
    {shape : Shape} (predicted target : Tensor α shape)
    (finite : Context.isFinite (sum ((predicted - target) * (predicted - target))) = true) :
    meanSquaredError predicted target =
      Context.divNat (sum ((predicted - target) * (predicted - target)))
        (meanDenominator shape) := by
  simp only [meanSquaredError, finite, Bool.true_or, ↓reduceIte]

-- Shape-level bookkeeping for reductions that drop one axis.

/-- Shape obtained by replacing the selected axis with a singleton dimension. -/
def shapeAfterSumKeepDim (shape : Shape) (axis : Nat) : Shape :=
  shape.set axis 1

/-- Keeping the reduced axis as a singleton does not change the element count. -/
@[simp] theorem size_shapeAfterSumKeepDim (s : Shape) (axis : Nat) :
    Shape.size (shapeAfterSumKeepDim s axis) = Shape.size (s.eraseIdx axis) := by
  induction s generalizing axis with
  | scalar => rfl
  | dim n inner ih =>
      cases axis with
      | zero =>
          change 1 * inner.size = inner.size
          exact Nat.one_mul _
      | succ axis =>
          change n * (shapeAfterSumKeepDim inner axis).size =
            n * Shape.size (inner.eraseIdx axis)
          exact congrArg (n * ·) (ih axis)

/-- Keeping the reduced axis as a singleton preserves the rank. -/
@[simp] theorem rank_shapeAfterSumKeepDim (s : Shape) (axis : Nat) :
    Shape.rank (shapeAfterSumKeepDim s axis) = Shape.rank s := by
  induction s generalizing axis with
  | scalar => rfl
  | dim n inner ih =>
      cases axis with
      | zero => rfl
      | succ axis =>
          change (shapeAfterSumKeepDim inner axis).rank + 1 = inner.rank + 1
          exact congrArg (· + 1) (ih axis)

/-- A keep-dimension reduction shape broadcasts back to its input shape. -/
theorem shapeAfterSumKeepDim_canBroadcastTo : (s : Shape) → (axis : Nat) →
    Shape.CanBroadcastTo (shapeAfterSumKeepDim s axis) s
  | .scalar, _ => .scalar
  | .dim _ inner, 0 => .dim_one (Shape.CanBroadcastTo.refl inner)
  | .dim _ inner, Nat.succ axis =>
      letI : Shape.SameRank (inner.set axis 1) inner :=
        ⟨rank_shapeAfterSumKeepDim inner axis⟩
      .dim_eq (shapeAfterSumKeepDim_canBroadcastTo inner axis)

/-- Coordinates of `shape` are pairs of an index along `axis` and a coordinate of the other axes,
which keep their relative order. -/
def _root_.TorchLean.Shape.Coord.axisEquiv :
    (axis : Nat) → {shape : Shape} → [_h : Shape.AxisInBounds axis shape] →
      Shape.Coord shape ≃
        Fin (Shape.axisSize shape axis) × Shape.Coord (shape.eraseIdx axis)
  | 0, Shape.dim n inner, _ => Equiv.refl (Fin n × Shape.Coord inner)
  | axis + 1, Shape.dim n inner, h => by
      let innerAxis : Shape.AxisInBounds axis inner := ⟨by
        have valid := h.proof
        simp only [Shape.rank_cons] at valid
        omega⟩
      letI := innerAxis
      let selected := Fin (Shape.axisSize inner axis)
      let retained := Shape.Coord (inner.eraseIdx axis)
      exact
        (Equiv.prodCongr (Equiv.refl (Fin n)) (@axisEquiv axis inner innerAxis)).trans
          ((Equiv.prodAssoc (Fin n) selected retained).symm.trans
            ((Equiv.prodCongr (Equiv.prodComm (Fin n) selected) (Equiv.refl retained)).trans
              (Equiv.prodAssoc selected (Fin n) retained)))

/-- Coordinates of `shape.set axis count` are pairs of an index in `Fin count` along `axis`
and a coordinate of the other axes, which keep their relative order. This is `axisEquiv` for the
shape with extent `count` at `axis`. -/
def _root_.TorchLean.Shape.Coord.setEquiv :
    (axis : Nat) → {shape : Shape} → [_h : Shape.AxisInBounds axis shape] →
      (count : Nat) → Shape.Coord (shape.set axis count) ≃
        Fin count × Shape.Coord (shape.eraseIdx axis)
  | 0, Shape.dim _ inner, _, count => Equiv.refl (Fin count × Shape.Coord inner)
  | axis + 1, Shape.dim n inner, h, count => by
      let innerAxis : Shape.AxisInBounds axis inner := ⟨by
        have valid := h.proof
        simp only [Shape.rank_cons] at valid
        omega⟩
      letI := innerAxis
      let retained := Shape.Coord (inner.eraseIdx axis)
      exact
        (Equiv.prodCongr (Equiv.refl (Fin n)) (setEquiv axis count)).trans
          ((Equiv.prodAssoc (Fin n) (Fin count) retained).symm.trans
            ((Equiv.prodCongr (Equiv.prodComm (Fin n) (Fin count)) (Equiv.refl retained)).trans
              (Equiv.prodAssoc (Fin count) (Fin n) retained)))

-- Reducers parameterized by the scalar aggregation operation.


/-- Apply `f` to the vectors along the outer axis, dropping that axis: the case of axis zero of
`reduceDimWith`. -/
def reduceOuterAxis {α : Type} [TorchLean.Storage α]
    {innerShape : Shape} {n : Nat}
    (f : ∀ {sliceShape : Shape}, Tensor α sliceShape → α)
    (t : Tensor α (.dim n innerShape)) : Tensor α innerShape :=
    match innerShape with
    | .scalar => .scalar (f t)
    | .dim _ _ =>
        .dim (fun j =>
          let sliceAtJ := .dim (fun i => Tensor.unstack (Tensor.unstack t i) j)
          reduceOuterAxis f sliceAtJ)

/-- Reducing a vector along its only axis applies the scalar aggregator to that vector. -/
@[simp] theorem reduceOuterAxis_vector
    {α : Type} [TorchLean.Storage α] {n : Nat}
    (f : ∀ {sliceShape : Shape}, Tensor α sliceShape → α)
    (tensor : Tensor α [n]) :
    reduceOuterAxis f tensor = Tensor.scalar (f tensor) := by
  rfl

/-- An entry of `reduceOuterAxis f input` is `f` on the entries of `input` along the outer axis
through it, in index order, for any scalar type and aggregator. -/
theorem reduceOuterAxis_apply {α : Type} [Storage α]
    (f : ∀ {sliceShape : Shape}, Tensor α sliceShape → α)
    {n : Nat} {inner : Shape} (input : Tensor α (.dim n inner))
    (coordinate : inner.Coord) :
    reduceOuterAxis f input coordinate =
      f (Tensor.ofFn fun index : Fin n => input (index, coordinate)) := by
  induction inner generalizing n with
  | scalar =>
      cases coordinate
      simp only [reduceOuterAxis, Tensor.scalar_apply]
      congr 1
      apply Tensor.Rep.ext
      rintro ⟨index, tailCoordinate⟩
      cases tailCoordinate
      simp only [Tensor.ofFn_apply]
  | dim m inner ih =>
      rcases coordinate with ⟨column, coordinate⟩
      simp only [reduceOuterAxis, Tensor.dim_apply]
      rw [ih]
      congr 1
      apply Tensor.Rep.ext
      rintro ⟨index, tailCoordinate⟩
      simp only [Tensor.ofFn_apply, Tensor.dim_apply,
        Tensor.unstack_apply]

/-- Apply `f` to the vectors along `axis`, dropping that axis: the reduction along one axis for an
aggregator that is not a `ReduceOp`, such as `maxAbs` or a directed-rounding sum. An axis past the
rank leaves the tensor unchanged. -/
def reduceDimWith
    {α : Type} [TorchLean.Storage α]
    (f : ∀ {sliceShape : Shape}, Tensor α sliceShape → α) :
    (s : Shape) → (axis : Nat) → Tensor α s → Tensor α (s.eraseIdx axis)
  | .scalar => fun _ tensor => tensor
  | .dim _ inner => fun axis tensor =>
      match axis with
      | 0 => reduceOuterAxis f tensor
      | Nat.succ axis =>
          Tensor.dim (fun index =>
            reduceDimWith f inner axis (Tensor.unstack tensor index))

/-- There is no axis to reduce in a scalar tensor, so the tensor is returned unchanged. -/
@[simp] theorem reduceDimWith_scalar
    {α : Type} [TorchLean.Storage α]
    (f : ∀ {sliceShape : Shape}, Tensor α sliceShape → α)
    (axis : Nat) (tensor : Tensor α .scalar) :
    reduceDimWith f .scalar axis tensor = tensor := by
  rfl

/-- Reducing axis zero is the outer-axis reduction. -/
@[simp] theorem reduceDimWith_dim_zero
    {α : Type} [TorchLean.Storage α]
    (f : ∀ {sliceShape : Shape}, Tensor α sliceShape → α)
    {n : Nat} {inner : Shape} (tensor : Tensor α (.dim n inner)) :
    reduceDimWith f (.dim n inner) 0 tensor = reduceOuterAxis f tensor := by
  rfl

/-- Reducing a deeper axis pushes the reduction into every outer slice.

With `reduceDimWith_scalar` and `reduceDimWith_dim_zero`, this equation is the whole computation
rule of `reduceDimWith`, and the three `simp` lemmas unfold a reduction on a literal shape without
mentioning the recursion. -/
@[simp] theorem reduceDimWith_dim_succ
    {α : Type} [TorchLean.Storage α]
    (f : ∀ {sliceShape : Shape}, Tensor α sliceShape → α)
    {n axis : Nat} {inner : Shape} (values : Fin n → Tensor α inner) :
    reduceDimWith f (.dim n inner) (axis + 1) (Tensor.dim values) =
      Tensor.dim (fun index => reduceDimWith f inner axis (values index)) := by
  simp [reduceDimWith]


/-- An entry of `reduceDimWith f s axis x` is `f` applied to the entries of `x` along `axis`
through it, in increasing index order. An empty axis supplies an empty vector. -/
theorem reduceDimWith_apply {α : Type} [Storage α]
    (f : ∀ {sliceShape : Shape}, Tensor α sliceShape → α) :
    (axis : Nat) → {shape : Shape} → [_h : Shape.AxisInBounds axis shape] →
      (input : Tensor α shape) → (coordinate : Shape.Coord (shape.eraseIdx axis)) →
      reduceDimWith f shape axis input coordinate =
        f (Tensor.ofFn fun index : Fin (Shape.axisSize shape axis) =>
          input ((Shape.Coord.axisEquiv axis (shape := shape)).symm
            (index, coordinate)))
  | 0, .dim _ _, _, input, coordinate => by
      change reduceOuterAxis f input coordinate =
        f (Tensor.ofFn fun index => input (index, coordinate))
      exact reduceOuterAxis_apply f input coordinate
  | axis + 1, .dim outerSize inner, h, input, ⟨outer, coordinate⟩ => by
      let innerAxis : Shape.AxisInBounds axis inner := ⟨by
        have valid := h.proof
        simp only [Shape.rank_cons] at valid
        omega⟩
      let := innerAxis
      simp only [reduceDimWith]
      erw [Tensor.dim_apply]
      change reduceDimWith f inner axis (Tensor.unstack input outer) coordinate = _
      rw [@reduceDimWith_apply α _ f axis inner innerAxis (Tensor.unstack input outer) coordinate]
      congr 1
      apply Tensor.Rep.ext
      rintro ⟨index, tailCoordinate⟩
      simp only [Tensor.ofFn, Tensor.Rep.get_ofFn,
        Tensor.unstack_apply]
      exact (Tensor.Rep.get_ofFn
        (fun pair : Shape.Coord [Shape.axisSize inner axis] =>
          input ((Shape.Coord.axisEquiv (axis + 1)
            (shape := .dim outerSize inner)).symm (pair.1, outer, coordinate)))
        (index, tailCoordinate)).symm


end TorchLean.Tensor
