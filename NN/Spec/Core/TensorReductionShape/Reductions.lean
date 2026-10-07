/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.TensorReductionShape.ShapeChange
public import NN.Spec.Core.TensorOps
public import NN.Tensor.Internal.Representation.Basic.Traversal

@[expose] public section


open Spec TorchLean

namespace TorchLean.Tensor

variable {α : Type} [TorchLean.Storage α] [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)]

/-!
# Reductions

Fold, sum, mean, variance, and axis reductions.
-/

/-- Left fold over all tensor elements. -/
def foldlSpec {α β : Type} [TorchLean.Storage α]
    (f : β → α → β) (init : β) {s : Shape} (tensor : Tensor α s) : β :=
  tensor.foldl f init

/-- Folding a scalar tensor visits its single element once. -/
@[simp] theorem foldlSpec_scalar {α β : Type} [TorchLean.Storage α]
    (f : β → α → β) (init : β) (value : α) :
    foldlSpec f init (Tensor.scalar value) = f init value := by
  rw [foldlSpec, TorchLean.Tensor.Internal.Rep.foldl_eq_data_foldl]
  simp only [Tensor.scalar, TorchLean.Tensor.Internal.Rep.data, TorchLean.Tensor.Internal.Rep.ofFn,
    TorchLean.Tensor.Internal.Rep.ofFlatFn, TorchLean.Storage.toArray_ofFn]
  rw [← Array.foldl_toList]
  simp

namespace foldlSpec

/--
Proof-facing fold over a family of equal-shaped slices, starting at slice `index`.

The public `foldlSpec` traverses the packed buffer directly. This helper retains the
slice-by-slice recursion useful in shape-inductive proofs without changing the executable path.
-/
def go {α β : Type} [TorchLean.Storage α]
    (f : β → α → β) (length : Nat) (shape : Shape)
    (values : Fin length → Tensor α shape) (index : Nat) (accumulator : β) : β :=
  ((List.finRange length).drop index).foldl
    (fun value component => foldlSpec f value (values component))
    accumulator

/-- Starting the proof-facing slice fold at zero is the ordinary finite fold. -/
theorem go_zero_eq_fin_foldl {α β : Type} [TorchLean.Storage α]
    (f : β → α → β) (length : Nat) (shape : Shape)
    (values : Fin length → Tensor α shape) (accumulator : β) :
    go f length shape values 0 accumulator =
      Fin.foldl length
        (fun value component => foldlSpec f value (values component))
        accumulator := by
  simp [go, Fin.foldl_eq_foldl_finRange]

end foldlSpec

/-- Folding a tensor with a leading dimension folds its slices in row-major order. -/
@[simp] theorem foldlSpec_dim {α β : Type} [TorchLean.Storage α]
    (f : β → α → β) (init : β) {length : Nat} {shape : Shape}
    (values : Fin length → Tensor α shape) :
    foldlSpec f init (Tensor.dim values) =
      foldlSpec.go f length shape values 0 init := by
  change (Tensor.dim values).foldl f init = _
  rw [Tensor.dim, TorchLean.Tensor.Internal.Rep.foldl_stack,
    foldlSpec.go_zero_eq_fin_foldl]
  rfl

-- Reductions that collapse a tensor to scalar values.
/-- Sum all elements of a tensor. -/
def sumSpec {α : Type} [TorchLean.Storage α] [Add α] [Zero α]
    {s : Shape} (t : Tensor α s) : α :=
  foldlSpec (· + ·) 0 t

/-- The sum of a scalar tensor is its value. -/
@[simp] theorem sumSpec_scalar {α : Type} [TorchLean.Storage α]
    [AddZeroClass α] (value : α) :
    sumSpec (Tensor.scalar value) = value := by
  simp [sumSpec]

/-- Flattened row-major index of the first maximal entry in a nonempty tensor.

The explicit nonemptiness hypothesis keeps the result total without inventing an index for an
empty tensor. Ties retain the smaller flattened index, matching a left-to-right traversal. -/
def argmax {s : Shape} (h : 0 < Shape.size s) (x : Tensor α s) : Fin (Shape.size s) :=
  let flat := flattenSpec x
  let value (i : Fin (Shape.size s)) := flat.getScalar i
  let rec loop (i : Nat) (best : Fin (Shape.size s)) : Fin (Shape.size s) :=
    if hi : i < Shape.size s then
      let current : Fin (Shape.size s) := ⟨i, hi⟩
      loop (i + 1) (if value current > value best then current else best)
    else
      best
  termination_by Shape.size s - i
  decreasing_by
    simpa using Nat.sub_succ_lt_self (a := Shape.size s) (i := i) hi
  loop 1 ⟨0, h⟩

/-- `true` if all entries satisfy `p`. -/
def allSpec {s : Shape} (p : α → Bool) (t : Tensor α s) : Bool :=
  foldlSpec (fun acc x => acc && p x) true t

/-- Dot product in row-major order, with a zero-seeded left fold. -/
def dotSpec {α : Type} [TorchLean.Storage α] [Add α] [Mul α] [Zero α]
    {s : Shape} (a b : Tensor α s) : α :=
  sumSpec (mulSpec a b)

-- Statistics computed over all scalar leaves of a tensor.

/-- Denominator of a totalized mean: the element count, or one for an empty shape.

Dividing by the raw size would make the mean of an empty tensor depend on the scalar type's
convention for `x / 0`. Using one instead makes the empty mean equal to the (zero) sum, the same
convention as the loss layer and `TorchLean.Tensor.mean`. -/
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

/-- Mean of all elements (treats nested dims as one big collection).

An empty tensor evaluates as `0 / (1 : α)`: `meanDenominator` uses one when the shape has no
entries. This is zero under ordinary scalar arithmetic. -/
def meanSpec {α : Type} [TorchLean.Storage α] [Add α] [Zero α] [Div α] [NatCast α]
    {s : Shape} (tensor : Tensor α s) : α :=
  sumSpec tensor / (meanDenominator s : α)

/-- Mean squared difference between two equally shaped tensors.

Subtraction and squaring are pointwise. The final mean uses row-major accumulation and the
totalized denominator, including for empty shapes. -/
def meanSquaredError {α : Type} [TorchLean.Storage α]
    [Add α] [Sub α] [Mul α] [Div α] [Zero α] [NatCast α]
    {shape : Shape} (predicted target : Tensor α shape) : α :=
  let diff := subSpec predicted target
  meanSpec (mulSpec diff diff)

omit [DecidableRel ((· > ·) : α → α → Prop)] in
/-- On a nonempty tensor the mean is the sum divided by the element count. -/
theorem meanSpec_of_size_pos {s : Shape} (h : 0 < s.size) (tensor : Tensor α s) :
    meanSpec tensor = sumSpec tensor / (s.size : α) := by
  rw [meanSpec, meanDenominator_of_size_pos h]

/-- Variance of all scalar leaves (population variance, divides by the total leaf count).

For a higher-rank tensor this centers every entry around the tensor-wide mean. In particular, it
does not collapse each outer slice to its mean before measuring dispersion. An empty tensor has
variance zero, using the same guarded denominator as `meanSpec`.
-/
def varianceSpec {s : Shape} (tensor : Tensor α s) : α :=
  let mean := meanSpec tensor
  let sumSquaredDifference := foldlSpec (fun accumulator value =>
    let difference := value - mean
    accumulator + difference * difference) 0 tensor
  sumSquaredDifference / (meanDenominator s : α)

-- Shape-level bookkeeping for reductions that drop one axis.
/-- Output shape after summing along `axis` (drops that dimension). -/
@[reducible] def shapeAfterSum : Shape → Nat → Shape :=
  Shape.eraseAxis

/-- Shape obtained by replacing the selected axis with a singleton dimension. -/
def shapeAfterSumKeepDim (shape : Shape) (axis : Nat) : Shape :=
  shape.replaceAxis axis 1

/-- Keeping the reduced axis as a singleton does not change the element count. -/
@[simp] theorem size_shapeAfterSumKeepDim (s : Shape) (axis : Nat) :
    Shape.size (shapeAfterSumKeepDim s axis) = Shape.size (shapeAfterSum s axis) := by
  induction s generalizing axis with
  | scalar => rfl
  | dim n inner ih =>
      cases axis with
      | zero =>
          change 1 * inner.size = inner.size
          exact Nat.one_mul _
      | succ axis =>
          change n * (shapeAfterSumKeepDim inner axis).size =
            n * (shapeAfterSum inner axis).size
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

/-- Dropping axis zero from `.dim n inner` yields `inner`, including when `n = 0`. -/
@[simp] theorem shapeAfterSum_zero (n : Nat) (inner : Shape) :
    shapeAfterSum (.dim n inner) 0 = inner := by
  rfl

/-- `simp` lemma: dropping axis `k+1` recurses into the tail shape. -/
@[simp] theorem shapeAfterSum_succ {n s k} :
    shapeAfterSum (.dim n s) (k + 1) = .dim n (shapeAfterSum s k) := by
  rfl

/-- A keep-dimension reduction shape broadcasts back to its input shape. -/
theorem shapeAfterSumKeepDimBroadcast : (s : Shape) → (axis : Nat) →
    Shape.CanBroadcastTo (shapeAfterSumKeepDim s axis) s
  | .scalar, _ => .scalar
  | .dim _ inner, 0 => .dim_1_to_n (Shape.CanBroadcastTo.refl inner)
  | .dim _ inner, Nat.succ axis =>
      letI : Shape.SameRank (inner.replaceAxis axis 1) inner :=
        ⟨rank_shapeAfterSumKeepDim inner axis⟩
      .dim_eq (shapeAfterSumKeepDimBroadcast inner axis)

/-- Reinsert an axis dropped by `shapeAfterSum`, repeating the reduced tensor along that axis.

This is deliberately separate from `broadcastTo`. Generic broadcasting aligns dimensions from the
right, whereas a reduction backward pass must restore the exact axis that was removed. -/
def broadcastAfterSum {α : Type} [TorchLean.Storage α] :
    (s : Shape) → (axis : Nat) → Tensor α (shapeAfterSum s axis) → Tensor α s
  | .scalar, _, x => x
  | .dim n _, 0, x => Tensor.dim (fun _ : Fin n => x)
  | .dim _ inner, Nat.succ axis, tensor =>
      Tensor.dim (fun i =>
        broadcastAfterSum inner axis (Tensor.unstack tensor i))

-- Reducers parameterized by the scalar aggregation operation.

namespace Reduction
namespace Internal

/-- Reduce a tensor by applying `f` across its outer axis.

This is the basic “reduce over axis 0” primitive that we reuse to implement broadcast-adjoints and
multi-axis reducers.
-/
def reduceOuterAxis {α : Type} [TorchLean.Storage α]
    {innerShape : Shape} {n : Nat}
    (f : ∀ {sliceShape : Shape}, Tensor α sliceShape → α)
    (t : Tensor α (.dim n innerShape)) : Tensor α innerShape :=
    match innerShape with
    | .scalar => .scalar (f t)
    | .dim _ _ =>
        .dim (fun j =>
          let sliceAtJ := .dim (fun i => get (Tensor.unstack t i) j)
          reduceOuterAxis f sliceAtJ)

/-- Reducing a vector along its only axis applies the scalar aggregator to that vector. -/
@[simp] theorem reduceOuterAxis_vector
    {α : Type} [TorchLean.Storage α] {n : Nat}
    (f : ∀ {sliceShape : Shape}, Tensor α sliceShape → α)
    (tensor : Tensor α [n]) :
    reduceOuterAxis f tensor = Tensor.scalar (f tensor) := by
  rfl

end Internal
end Reduction

/-!
Reduce a gradient from a broadcast target shape back to the original input shape.

This is the adjoint of `broadcastTo` for sum-reduction: broadcast duplicates values, so the
backward pass sums contributions across broadcasted dimensions.

PyTorch analogy: this is the logic behind "sum over broadcasted dimensions" that happens in
autograd for `expand` + elementwise ops.
-/

/-- Adjoint of `broadcastTo` under sum-reduction: collapse broadcast axes by summing.

Target axes that only raise the rank are summed away. A source axis of extent one that was
expanded to a larger target extent is summed and reinserted as a singleton axis. Axes with equal
extents pass through slice by slice. The recursion is on the shapes alone, so the result does not
depend on how the broadcast relation was proved. -/
def reduceFromBroadcastTo {α : Type} [TorchLean.Storage α] [Add α] [Zero α] :
    {s₁ s₂ : Shape} → Shape.CanBroadcastTo s₁ s₂ → Tensor α s₂ → Tensor α s₁
  | .scalar, .scalar, _, t => t
  | .dim _ _, .scalar, h, _ => absurd h Shape.not_canBroadcastTo_dim_scalar
  | .scalar, .dim _ _, h, t =>
      reduceFromBroadcastTo (Shape.canBroadcastTo_scalar_dim.mp h)
        (Reduction.Internal.reduceOuterAxis (fun {sliceShape} => sumSpec (s := sliceShape)) t)
  | .dim m s₁, .dim n s₂, h, t =>
      if hRank : s₁.rank = s₂.rank then
        have tail := ((Shape.canBroadcastTo_dim_dim_of_rank_eq hRank).mp h).2
        if hExtent : m = n then
          Tensor.dim fun i => reduceFromBroadcastTo tail (Tensor.unstack t (Fin.cast hExtent i))
        else
          let summed := reduceFromBroadcastTo tail
            (Reduction.Internal.reduceOuterAxis (fun {sliceShape} => sumSpec (s := sliceShape)) t)
          Tensor.dim fun _ => summed
      else
        reduceFromBroadcastTo ((Shape.canBroadcastTo_dim_dim_of_rank_ne hRank).mp h)
          (Reduction.Internal.reduceOuterAxis (fun {sliceShape} => sumSpec (s := sliceShape)) t)

namespace Reduction
namespace Internal

/-- Recursive evaluator underlying `reduceDim`; kept separate so proofs can use its equations. -/
def reduceDimCore
    {α : Type} [TorchLean.Storage α]
    (f : ∀ {sliceShape : Shape}, Tensor α sliceShape → α) :
    (s : Shape) → (axis : Nat) → Tensor α s → Tensor α (shapeAfterSum s axis)
  | .scalar => fun _ tensor => tensor
  | .dim _ inner => fun axis tensor =>
      match axis with
      | 0 => reduceOuterAxis f tensor
      | Nat.succ axis =>
          Tensor.dim (fun index =>
            reduceDimCore f inner axis (Tensor.unstack tensor index))

/-- There is no axis to reduce in a scalar tensor, so the tensor is returned unchanged. -/
@[simp] theorem reduceDimCore_scalar
    {α : Type} [TorchLean.Storage α]
    (f : ∀ {sliceShape : Shape}, Tensor α sliceShape → α)
    (axis : Nat) (tensor : Tensor α .scalar) :
    reduceDimCore f .scalar axis tensor = tensor := by
  rfl

/-- Reducing axis zero is the outer-axis reduction. -/
@[simp] theorem reduceDimCore_dim_zero
    {α : Type} [TorchLean.Storage α]
    (f : ∀ {sliceShape : Shape}, Tensor α sliceShape → α)
    {n : Nat} {inner : Shape} (tensor : Tensor α (.dim n inner)) :
    reduceDimCore f (.dim n inner) 0 tensor = reduceOuterAxis f tensor := by
  rfl

/-- Reducing a deeper axis pushes the reduction into every outer slice.

These three equations are the whole computation rule for `reduceDimCore`, and stating them as simp
lemmas is what lets a reduction on a literal shape unfold without ever mentioning the recursion. -/
@[simp] theorem reduceDimCore_dim_succ
    {α : Type} [TorchLean.Storage α]
    (f : ∀ {sliceShape : Shape}, Tensor α sliceShape → α)
    {n axis : Nat} {inner : Shape} (values : Fin n → Tensor α inner) :
    reduceDimCore f (.dim n inner) (axis + 1) (Tensor.dim values) =
      Tensor.dim (fun index => reduceDimCore f inner axis (values index)) := by
  simp [reduceDimCore]

end Internal
end Reduction

/-- Generic reduction along an axis.

`reduceDim f axis x` applies `f` to the slices along `axis`, and returns a tensor whose shape is
`shapeAfterSum s axis` (that axis is dropped). The axis may be empty: `f` then receives empty
slices and returns whatever it does on them, such as zero for `sumSpec`. The typed `reduceMax`
requires an initial entry, and `reduceMean` divides by the axis extent; both require nonemptiness.
-/
def reduceDim
    {α : Type} [TorchLean.Storage α]
    {s : Shape}
    (f : ∀ {sliceShape : Shape}, Tensor α sliceShape → α)
    (axis : Nat)
    (x : Tensor α s) : Tensor α (shapeAfterSum s axis) :=
  Reduction.Internal.reduceDimCore f s axis x

/-- Sum-reduction along a given axis.

The evidence matches `IR.OpContracts.checkReductionAxis`, which rejects invalid axes and
in-bounds axes of extent zero during both inference and evaluation. The underlying computation
also supports empty sums: use `reduceDim sumSpec` when that broader domain is intended. -/
def reduceSum {α : Type} [TorchLean.Storage α] [Add α] [Zero α]
    {s : Shape} (axis : Nat) (t : Tensor α s) (_h : Shape.NonemptyAxis axis s) :
    Tensor α (shapeAfterSum s axis) :=
  reduceDim sumSpec axis t

/-- Mean-reduction along a given axis. -/
def reduceMean {α : Type} [TorchLean.Storage α] [Add α] [Zero α] [Div α] [NatCast α]
    {s : Shape} (axis : Nat) (t : Tensor α s) (h : Shape.NonemptyAxis axis s) :
  Tensor α (shapeAfterSum s axis) :=
  let summed := reduceSum axis t h
  letI : Shape.AxisInBounds axis s := h.toAxisInBounds
  Tensor.map (fun x => x / (Shape.axisSize s axis : α)) summed

/-- Variance-reduction along a given axis (population variance, divides by `n`).

The reduced axis is centered first and squared second. This two-pass arrangement avoids the
catastrophic cancellation of $\mathbb{E}[X^2]-\mathbb{E}[X]^2$ when values are large but tightly
clustered.
-/
def reduceVar
  {s : Shape} (axis : Nat) (t : Tensor α s) (h : Shape.NonemptyAxis axis s) :
  Tensor α (shapeAfterSum s axis) :=
  match s with
  | .scalar => nomatch h
  | .dim n inner =>
    match axis with
    | 0 =>
      -- PyTorch analogy: `torch.var(x, dim=0, unbiased=False)` (population variance).
      let mean := reduceMean 0 t h
      let centered :=
        Tensor.dim (fun i => subSpec (Tensor.unstack t i) mean)
      let squared := Tensor.map (fun x => x * x) centered
      reduceMean 0 squared h

    | Nat.succ k =>
      -- Reducing along axis k+1 in the inner dimensions
      -- Apply reduce_var recursively to each slice along the first dimension
      let innerReducible : Shape.NonemptyAxis k inner := by
        cases h with
        | succ inner => exact inner
      Tensor.dim fun i =>
        reduceVar k (Tensor.unstack t i) innerReducible

/-- Max-reduction along a given axis. -/
def reduceMax {s : Shape}
  (axis : Nat) (t : Tensor α s) (h : Shape.NonemptyAxis axis s) :
  Tensor α (shapeAfterSum s axis) :=
  match s with
  | .scalar => t
  | .dim n inner =>
    match axis with
    | 0 =>
      -- PyTorch analogy: `torch.amax(x, dim=0)`.
      match n with
      | 0 => nomatch h
      | Nat.succ n' =>
        let rec loop (i : Nat) (acc : Tensor α inner) : Tensor α inner :=
          if h_lt : i < n' then
            let next_idx : Fin (Nat.succ n') := ⟨i + 1, Nat.succ_lt_succ h_lt⟩
            loop (i + 1) (maxSpec acc (Tensor.unstack t next_idx))
          else
            acc
        let first_idx : Fin (Nat.succ n') := ⟨0, Nat.succ_pos n'⟩
        loop 0 (Tensor.unstack t first_idx)
    | Nat.succ k =>
      let innerReducible : Shape.NonemptyAxis k inner := by
        cases h with
        | succ inner => exact inner
      Tensor.dim fun i =>
        reduceMax k (Tensor.unstack t i) innerReducible

-- Transpose operations live in the linear-algebra extension modules.
end TorchLean.Tensor
