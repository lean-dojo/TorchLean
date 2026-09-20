/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Context
public import NN.Spec.Core.Tensor -- shake: keep

/-!
# Elementwise tensor operations (`TorchLean.Tensor.*_spec`)

This file defines shape-preserving, elementwise operations on `Tensor α s`.

Naming convention:

- `foo_spec` means “pure spec definition” (no runtime side effects).
- most functions use packed pointwise kernels via `mapSpec` / `map2Spec`.

## Domain / smoothness notes

Some operations are domain-sensitive or non-smooth:

- `sqrtSpec` uses `sqrt (max x 0)` to stay total on ordered rings.
- `logSpec` is total as a function call, but analytic properties require positivity assumptions.
- `relu` / `clamp` / comparisons are non-smooth; analytic backprop theorems treat these via
  pointwise assumptions, or by switching to smooth surrogates in verification workflows.

The **spec layer** is where these semantics are defined; the **proof layer** decides which
assumptions/variants to use for theorems.
-/

@[expose] public section


open Spec TorchLean

namespace TorchLean.Tensor

variable {α : Type} [TorchLean.Storage α] [Context α]

/--
Keep a tensor's primal values while removing scalar differentiation metadata.

For ordinary numeric tensors, this returns the same tensor. Dual-valued tensors need a pointwise
map as well as the graph's zero JVP and VJP: otherwise a later operation could still read a tangent
from the detached value while computing a higher-order derivative.
-/
def detachSpec {s : Shape} (x : Tensor α s) : Tensor α s :=
  match Context.stopGradient? (α := α) with
  | none => x
  | some clear => Tensor.map clear x

/-! ## Axis-parametric indexing -/

/-- Place one selected slice back into an otherwise-zero tensor.

This is the adjoint of `selectSpec` and therefore its reverse-mode rule. -/
def selectBackwardSpec {α : Type} [TorchLean.Storage α] [Zero α] :
    (axis : Nat) → {s : Shape} →
      [_h : Shape.AxisInBounds axis s] →
      Fin (Shape.axisSize s axis) → Tensor α (s.eraseAxis axis) → Tensor α s
  | 0, .dim n rest, _, selected, gradient =>
      .dim fun coordinate =>
        if coordinate.val = selected.val then gradient else Tensor.full rest 0
  | axis + 1, .dim n rest, h, selected, gradients =>
      .dim fun outer =>
        @selectBackwardSpec α _ _ axis rest
          ⟨by
            have := h.proof
            simp only [Shape.rank] at this
            grind⟩
          (Fin.cast (by rfl) selected)
          (Spec.get gradients outer)

/-- Select coordinates from an arbitrary axis using an index vector.

The selected axis is replaced by the index count. Repeated indices are preserved, matching
`torch.index_select`. -/
def indexSelectSpec {α : Type} [TorchLean.Storage α] :
    (axis : Nat) → {s : Shape} → (tensor : Tensor α s) →
      [_h : Shape.AxisInBounds axis s] →
      {count : Nat} → Tensor (Fin (Shape.axisSize s axis)) [count] →
      Tensor α (s.replaceAxis axis count)
  | 0, .dim _ _, tensor, _, count, indices =>
      .dim fun j =>
        Spec.get tensor (indices.getScalar j)
  | axis + 1, .dim n rest, tensor, h, count, indices =>
      .dim fun outer =>
        @indexSelectSpec α _ axis rest (Spec.get tensor outer)
          ⟨by
            have := h.proof
            simp only [Shape.rank] at this
            grind⟩
          count
          (cast (by rfl) indices)

/-- Insert a sliced gradient back along an arbitrary axis, filling coordinates outside it with
zero. -/
def sliceAxisRangeBackwardSpec {α : Type}
    [TorchLean.Storage α] [Zero α] :
    (axis : Nat) → {s : Shape} →
      [_h : Shape.AxisInBounds axis s] →
      (start count : Nat) → (start + count ≤ Shape.axisSize s axis) →
      Tensor α (s.replaceAxis axis count) → Tensor α s
  | 0, .dim n rest, _, start, count, _hRange, gradients =>
      .dim fun coordinate =>
        if hBefore : coordinate.val < start then
          Tensor.full rest 0
        else if hInside : coordinate.val < start + count then
          Spec.get gradients ⟨coordinate.val - start,
            Nat.sub_lt_left_of_lt_add (Nat.not_lt.mp hBefore) hInside⟩
        else
          Tensor.full rest 0
  | axis + 1, .dim n rest, hAxis, start, count, hRange, gradients =>
      let innerAxis : Shape.AxisInBounds axis rest :=
        ⟨by
          have := hAxis.proof
          simp only [Shape.rank] at this
          grind⟩
      letI := innerAxis
      have hRange' : start + count ≤ Shape.axisSize rest axis := by
        simpa only [Shape.axisSize_succ] using hRange
      .dim fun outer =>
        @sliceAxisRangeBackwardSpec α _ _ axis rest innerAxis start count hRange'
          (Spec.get gradients outer)

/--
Map a scalar function over a tensor (shape preserved).

This is the core packed pointwise combinator for spec tensors.
Most elementwise ops are direct instances of `mapSpec f`.

PyTorch analogy: `f` applied pointwise (like `torch.<op>` broadcasting over all entries),
but here shape is fixed and enforced by the type.
-/
def mapSpec {s : Shape} (f : α → α) : Tensor α s → Tensor α s :=
  Tensor.map f

omit [Context α] in
/-- Elementwise mapping computes directly on a scalar tensor. -/
@[simp] theorem mapSpec_scalar (f : α → α) (x : α) :
    mapSpec f (Tensor.scalar x) = Tensor.scalar (f x) := by
  exact Tensor.map_scalar f x

omit [Context α] in
/-- Elementwise mapping distributes over the leading tensor dimension. -/
@[simp] theorem mapSpec_dim {n : Nat} {s : Shape} (f : α → α)
    (values : Fin n → Tensor α s) :
    mapSpec f (Tensor.dim values) = Tensor.dim (fun i => mapSpec f (values i)) := by
  exact Tensor.map_dim f values

omit [Context α] in
/-- Extracting a scalar after an elementwise map applies the scalar function once. -/
@[simp] theorem toScalar_mapSpec (f : α → α) (x : Tensor α .scalar) :
    (mapSpec f x).item = f x.item := by
  simp [mapSpec, Tensor.item, Tensor.map]

omit [Context α] in
/-- Vector indexing commutes with an elementwise operation. -/
@[simp] theorem getScalar_mapSpec {n : Nat} (f : α → α)
    (tensor : Tensor α [n]) (index : Fin n) :
    (mapSpec f tensor).getScalar index = f (tensor.getScalar index) := by
  exact Tensor.getScalar_map f tensor index

omit [Context α] in
/-- Transport a pointwise tensor property through an elementwise operation. -/
theorem forall_mapSpec
    {p q : α → Prop} {f : α → α} {s : Shape} {x : Tensor α s}
    (hx : Forall p x) (hf : ∀ a, p a → q (f a)) :
    Forall q (mapSpec f x) := by
  induction s with
  | scalar =>
      have hx' : p x.item := by simpa [Forall] using hx
      change q ((mapSpec f x).item)
      rw [toScalar_mapSpec]
      exact hf x.item hx'
  | dim n inner ih =>
      intro i
      rw [show Tensor.unstack (mapSpec f x) i =
          mapSpec f (Tensor.unstack x i) by
        exact (TorchLean.Tensor.Internal.Rep.map_unstack f x i).symm]
      exact ih (hx i)

/--
Map a binary function over two tensors of the same shape.

This is the packed `zipWith` combinator for spec tensors.
It is intentionally *shape-preserving*: if the shapes differ, the term is not well-typed.

PyTorch analogy: elementwise binary ops when tensors already have the same shape (no broadcasting).
Broadcasting is handled separately in `NN/Spec/Core/TensorReductionShape.lean`.
-/
def map2Spec {α β γ : Type}
    [TorchLean.Storage α] [TorchLean.Storage β]
    [TorchLean.Storage γ] {s : Shape}
    (f : α → β → γ) (left : Tensor α s) (right : Tensor β s) :
    Tensor γ s :=
  TorchLean.Tensor.Internal.Rep.zipWith f left right

/-- A pointwise binary operation computes directly on scalar tensors. -/
@[simp] theorem map2Spec_scalar {α β γ : Type}
    [TorchLean.Storage α] [TorchLean.Storage β]
    [TorchLean.Storage γ]
    (f : α → β → γ) (left : α) (right : β) :
    map2Spec f (Tensor.scalar left) (Tensor.scalar right) =
      Tensor.scalar (f left right) := by
  apply TorchLean.Tensor.Internal.Rep.ext
  intro coordinate
  cases coordinate
  simp [map2Spec, Tensor.scalar]

/-- Extracting a scalar after a pointwise binary map combines the two scalar values. -/
@[simp] theorem toScalar_map2Spec {α β γ : Type}
    [TorchLean.Storage α] [TorchLean.Storage β]
    [TorchLean.Storage γ]
    (f : α → β → γ) (left : Tensor α .scalar)
    (right : Tensor β .scalar) :
    (map2Spec f left right).item = f left.item right.item := by
  simp [map2Spec, Tensor.item]

/-- Pointwise binary operations distribute over the leading tensor dimension. -/
@[simp] theorem map2Spec_dim {α β γ : Type}
    [TorchLean.Storage α] [TorchLean.Storage β]
    [TorchLean.Storage γ] {n : Nat} {s : Shape}
    (f : α → β → γ) (left : Fin n → Tensor α s)
    (right : Fin n → Tensor β s) :
    map2Spec f (Tensor.dim left) (Tensor.dim right) =
      Tensor.dim fun i => map2Spec f (left i) (right i) := by
  exact TorchLean.Tensor.Internal.Rep.zipWith_stack f left right

/-- Evaluating a pointwise binary operation combines both scalar entries. -/
@[simp] theorem map2Spec_apply {α β γ : Type}
    [TorchLean.Storage α] [TorchLean.Storage β]
    [TorchLean.Storage γ] {s : Shape}
    (f : α → β → γ) (left : Tensor α s) (right : Tensor β s)
    (coordinate : s.Coord) :
    map2Spec f left right coordinate = f (left coordinate) (right coordinate) := by
  exact TorchLean.Tensor.Internal.Rep.zipWith_apply f left right coordinate

/-- Indexing a pointwise binary operation applies the scalar operation after indexing. -/
@[simp] theorem get_map2Spec {α β γ : Type}
    [TorchLean.Storage α] [TorchLean.Storage β]
    [TorchLean.Storage γ]
    {f : α → β → γ} {n : Nat} {s : Shape}
    (left : Tensor α (.dim n s)) (right : Tensor β (.dim n s)) (i : Fin n) :
    Spec.get (map2Spec f left right) i =
      map2Spec f (Spec.get left i) (Spec.get right i) := by
  apply TorchLean.Tensor.Internal.Rep.ext
  intro coordinate
  simp [map2Spec, Spec.get, Tensor.unstack]

/-- Vector indexing commutes with a pointwise binary tensor operation. -/
@[simp] theorem getScalar_map2Spec {α β γ : Type}
    [TorchLean.Storage α] [TorchLean.Storage β]
    [TorchLean.Storage γ]
    (f : α → β → γ) {n : Nat}
    (left : Tensor α [n]) (right : Tensor β [n]) (i : Fin n) :
    TorchLean.Tensor.getScalar (map2Spec f left right) i =
      f (TorchLean.Tensor.getScalar left i)
        (TorchLean.Tensor.getScalar right i) := by
  rw [Tensor.getScalar_eq_apply, map2Spec_apply,
    ← Tensor.getScalar_eq_apply, ← Tensor.getScalar_eq_apply]

/-- Matrix indexing commutes with a pointwise binary tensor operation. -/
@[simp] theorem get2_map2Spec {α β γ : Type}
    [TorchLean.Storage α] [TorchLean.Storage β]
    [TorchLean.Storage γ]
    {f : α → β → γ} {m n : Nat}
    (left : Tensor α [m, n])
    (right : Tensor β [m, n]) (i : Fin m) (j : Fin n) :
    Spec.get2 (map2Spec f left right) i j =
      f (Spec.get2 left i j) (Spec.get2 right i j) := by
  simp [map2Spec, Spec.get2, Tensor.getScalar,
    Spec.get, Tensor.unstack, Tensor.item]

omit [Context α] in
/-- Matrix indexing commutes with a pointwise unary tensor operation. -/
@[simp] theorem get2_mapSpec
    {m n : Nat} (f : α → α) (tensor : Tensor α [m, n])
    (i : Fin m) (j : Fin n) :
    Spec.get2 (mapSpec f tensor) i j =
      f (Spec.get2 tensor i j) := by
  simp [mapSpec, Spec.get2, Tensor.getScalar,
    Spec.get, Tensor.unstack, Tensor.item, Tensor.map]

/-- Transport two pointwise properties through a binary shape-preserving operation. -/
theorem forall_map2Spec {α β γ : Type}
    [TorchLean.Storage α] [TorchLean.Storage β]
    [TorchLean.Storage γ]
    {p : α → Prop} {q : β → Prop} {r : γ → Prop}
    {f : α → β → γ} {s : Shape} {x : Tensor α s} {y : Tensor β s}
    (hx : Forall p x) (hy : Forall q y) (hf : ∀ a b, p a → q b → r (f a b)) :
    Forall r (map2Spec f x y) := by
  induction s with
  | scalar =>
      have hx' : p x.item := by simpa [Forall] using hx
      have hy' : q y.item := by simpa [Forall] using hy
      change r ((map2Spec f x y).item)
      simp only [map2Spec, Tensor.item, TorchLean.Tensor.Internal.Rep.zipWith_apply]
      exact hf x.item y.item hx' hy'
  | dim n inner ih =>
      intro i
      rw [show Tensor.unstack (map2Spec f x y) i =
          map2Spec f (Tensor.unstack x i) (Tensor.unstack y i) by
        exact (TorchLean.Tensor.Internal.Rep.zipWith_unstack f x y i).symm]
      exact ih (hx i) (hy i)

/-- Element‑wise addition (shape preserved). -/
def addSpec {α : Type} [TorchLean.Storage α] [Add α]
    {s : Shape} (T₁ T₂ : Tensor α s) : Tensor α s :=
  map2Spec (· + ·) T₁ T₂

/-- Shape transport commutes with pointwise tensor addition. -/
theorem castShape_addSpec {α : Type} [TorchLean.Storage α]
    [Add α] {shape shape' : Shape}
    (left right : Tensor α shape) (h : shape = shape') :
    Tensor.castShape (addSpec left right) h =
      addSpec (Tensor.castShape left h) (Tensor.castShape right h) := by
  cases h
  rfl

/-- Add indexed source slices into an arbitrary axis of a tensor.

Repeated indices accumulate. This is the pure tensor semantics used by the backward rule for
`indexSelectSpec` and corresponds to `torch.scatter_add` with an index vector. -/
def scatterAddSpec {α : Type} [TorchLean.Storage α] [Add α] :
    (axis : Nat) → {s : Shape} → (base : Tensor α s) →
      [_h : Shape.AxisInBounds axis s] →
      {count : Nat} → Tensor (Fin (Shape.axisSize s axis)) [count] →
      Tensor α (s.replaceAxis axis count) → Tensor α s
  | 0, .dim _ _, base, _, count, indices, source =>
      .dim fun destination =>
        (List.finRange count).foldl
          (fun value sourceIndex =>
            let index := indices.getScalar sourceIndex
            if index.val = destination.val then
              addSpec value (Spec.get source sourceIndex)
            else
              value)
          (Spec.get base destination)
  | axis + 1, .dim n rest, base, h, count, indices, source =>
      .dim fun outer =>
        @scatterAddSpec α _ _ axis rest (Spec.get base outer)
          ⟨by
            have := h.proof
            simp only [Shape.rank] at this
            grind⟩
          count
          (cast (by rfl) indices)
          (Spec.get source outer)

/-- `addSpec` is the tensor `+` (both are `Rep.zipWith (· + ·)`). -/
theorem addSpec_eq_add {α : Type} [TorchLean.Storage α] [Add α]
    {s : Shape} (left right : Tensor α s) : addSpec left right = left + right :=
  rfl

/-- Element‑wise multiplication (shape preserved). -/
def mulSpec {α : Type} [TorchLean.Storage α] [Mul α]
    {s : Shape} (T₁ T₂ : Tensor α s) : Tensor α s :=
  map2Spec (· * ·) T₁ T₂

/-- Scalar extraction commutes with pointwise tensor multiplication. -/
@[simp] theorem toScalar_mulSpec {α : Type} [TorchLean.Storage α] [Mul α]
    (left right : Tensor α .scalar) :
    (mulSpec left right).item = left.item * right.item := by
  simp [mulSpec, map2Spec, Tensor.item]

/-- Multiplication by a filled tensor is coordinatewise multiplication by its scalar value. -/
@[simp] theorem mulSpec_full_left {α : Type}
    [TorchLean.Storage α] [Mul α] {s : Shape} (coefficient : α)
    (tensor : Tensor α s) :
    mulSpec (Tensor.full s coefficient) tensor = mapSpec (coefficient * ·) tensor := by
  apply TorchLean.Tensor.Internal.Rep.ext
  intro coordinate
  simp [mulSpec, map2Spec, mapSpec, Tensor.map]

/-- `Mul` instance for shape-indexed tensors: multiply pointwise, preserving the shape. -/
instance {α : Type} [TorchLean.Storage α] [Mul α]
    {s : Shape} : Mul (Tensor α s) :=
  ⟨mulSpec⟩

/-- Element‑wise subtraction (shape preserved). -/
def subSpec {α : Type} [TorchLean.Storage α] [Sub α]
    {s : Shape} : Tensor α s → Tensor α s → Tensor α s :=
  map2Spec (· - ·)

/-- `subSpec` is the tensor `-` (both are `Rep.zipWith (· - ·)`). -/
theorem subSpec_eq_sub {α : Type} [TorchLean.Storage α] [Sub α]
    {s : Shape} (left right : Tensor α s) : subSpec left right = left - right :=
  rfl

/-- Element‑wise division (shape preserved). -/
def divSpec {s : Shape} : Tensor α s → Tensor α s → Tensor α s :=
  map2Spec (· / ·)

/-- `Div` instance for shape-indexed tensors: divide pointwise, preserving the shape. -/
instance {s : Shape} : Div (Tensor α s) :=
  ⟨divSpec⟩

/-- Epsilon-shifted division, $x/(y+\varepsilon)$. The denominator can still be zero. -/
def safedivSpec {s : Shape} (t1 t2 : Tensor α s) : Tensor α s :=
  map2Spec (fun x y => x / (y + Context.defaultEpsilon)) t1 t2

/-- Scale a tensor by a scalar. -/
def scaleSpec {α : Type} [TorchLean.Storage α] [Mul α]
    {s : Shape} (t : Tensor α s) (scalar : α) : Tensor α s :=
  mapSpec (fun x => x * scalar) t

/-- Square each element of a tensor. -/
def squareSpec {s : Shape} (t : Tensor α s) : Tensor α s :=
  mapSpec (fun x => x * x) t

/-- Squaring is elementwise multiplication with the same tensor on both inputs. -/
theorem squareSpec_eq_mulSpec {s : Shape} (t : Tensor α s) :
    squareSpec t = mulSpec t t := by
  apply TorchLean.Tensor.Internal.Rep.ext
  intro coordinate
  simp [squareSpec, mapSpec, Tensor.map, mulSpec, map2Spec]

/-- Square root of each element (clamped to `max x 0` to stay total). -/
def sqrtSpec {s : Shape} (t : Tensor α s) : Tensor α s :=
  mapSpec (fun x => MathFunctions.sqrt (Max.max x 0)) t

/-- Absolute value of each element. -/
def absSpec {s : Shape} (t : Tensor α s) : Tensor α s :=
  mapSpec MathFunctions.abs t

/-- Element‑wise natural log. -/
def logSpec {s : Shape} (t : Tensor α s) : Tensor α s :=
  mapSpec MathFunctions.log t

/-- Element‑wise exponential. -/
def expSpec {s : Shape} (t : Tensor α s) : Tensor α s :=
  mapSpec MathFunctions.exp t

/-- Element‑wise negation. -/
def negSpec {s : Shape} (t : Tensor α s) : Tensor α s :=
  mapSpec Neg.neg t

/-- `negSpec` is the tensor negation (both are `Rep.map Neg.neg`). -/
theorem negSpec_eq_neg {s : Shape} (t : Tensor α s) : negSpec t = -t :=
  rfl

/-- Element‑wise power. -/
def powSpec {s : Shape} (t1 t2 : Tensor α s) : Tensor α s :=
  map2Spec HPow.hPow t1 t2

/-- Element‑wise comparisons (returning Bool tensors). -/
def greaterThanSpec {s : Shape} (x y : Tensor α s) : Tensor Bool s :=
  map2Spec (fun a b => decide (a > b)) x y

/-- Element-wise $\le$ test, implemented via $\neg(>)$ so we only depend on
`DecidableRel (· > ·)`. This agrees with `≤` for a total order; IEEE unordered
comparisons involving NaN return `true` here. -/
def lessEqualSpec {s : Shape} (x y : Tensor α s) : Tensor Bool s :=
  map2Spec (fun a b => decide (¬(a > b))) x y

/-- Element‑wise `<` test (defined as `y > x`). -/
def lessThanSpec {s : Shape} (x y : Tensor α s) : Tensor Bool s :=
  map2Spec (fun a b => decide (b > a)) x y

/-- Element-wise $\ge$ test (defined as $\neg(y>x)$). Like `lessEqualSpec`, this returns
`true` for IEEE unordered comparisons involving NaN. -/
def greaterEqualSpec {s : Shape} (x y : Tensor α s) : Tensor Bool s :=
  map2Spec (fun a b => decide (¬(b > a))) x y

/-- Boolean NOT, pointwise on a Bool tensor. -/
def notSpec {s : Shape} (x : Tensor Bool s) : Tensor Bool s :=
  mapSpec Bool.not x

/-- Element‑wise reciprocal (`1/x`). -/
def invSpec {s : Shape} (x : Tensor α s) : Tensor α s :=
  mapSpec (fun x => 1 / x) x

/-- Scalar extraction commutes with coordinatewise reciprocal. -/
@[simp] theorem toScalar_invSpec (x : Tensor α .scalar) :
    (invSpec x).item = 1 / x.item := by
  simp [invSpec, mapSpec, Tensor.item, Tensor.map]

/--
Clamp each entry into `[minVal, maxVal]`.

The runtime uses zero derivative outside the open interval, including at either endpoint.
Clearing scalar tangents on that branch gives dual-number execution the same convention as the
recorded JVP and VJP. The underlying minimum and maximum still determine the primal value.
-/
def clampSpec {s : Shape} (x : Tensor α s) (minVal maxVal : α) : Tensor α s :=
  mapSpec (fun v =>
    let clipped := Min.min maxVal (Max.max minVal v)
    if v > minVal ∧ maxVal > v then clipped else Context.stopGradient clipped) x

/-- Element‑wise minimum. -/
def minSpec {s : Shape} (t1 t2 : Tensor α s) : Tensor α s :=
  map2Spec (fun x y => Min.min x y) t1 t2

/-- Element‑wise maximum. -/
def maxSpec {s : Shape} (t1 t2 : Tensor α s) : Tensor α s :=
  map2Spec (fun x y => Max.max x y) t1 t2

/-- Element‑wise sign function: returns `-1`, `0`, or `1`. -/
def signSpec {s : Shape} (t : Tensor α s) : Tensor α s :=
  mapSpec (fun x => if x > 0 then 1 else if x < 0 then -1 else 0) t

/-- Elementwise sine, with each input interpreted as an angle in radians. -/
def sinSpec {s : Shape} : Tensor α s → Tensor α s :=
  mapSpec MathFunctions.sin

/-- Elementwise cosine, with each input interpreted as an angle in radians. -/
def cosSpec {s : Shape} : Tensor α s → Tensor α s :=
  mapSpec MathFunctions.cos

/-- Element‑wise cosh. -/
def coshSpec {s : Shape} : Tensor α s → Tensor α s :=
  mapSpec MathFunctions.cosh

/-- Element‑wise sinh. -/
def sinhSpec {s : Shape} : Tensor α s → Tensor α s :=
  mapSpec MathFunctions.sinh

/-- Derivative mask for clamp: `1` strictly inside `(minVal, maxVal)`, else `0`. -/
def clampDerivativeSpec {s : Shape} (x : Tensor α s) (minVal maxVal : α) : Tensor α s :=
  mapSpec (fun v => if v > minVal ∧ v < maxVal then 1 else 0) x

/-- Numeric mask: `1` where `a > b`, else `0`. -/
def gtMaskSpec {s : Shape} (a b : Tensor α s) : Tensor α s :=
  map2Spec (fun x y => if x > y then 1 else 0) a b

/-- Numeric mask: `1` where `a < b`, else `0`. -/
def ltMaskSpec {s : Shape} (a b : Tensor α s) : Tensor α s :=
  map2Spec (fun x y => if x < y then 1 else 0) a b

/-- Convert a Bool to `α` using `1`/`0`. -/
def boolToAlphaSpec : Bool → α :=
  fun b => if b then 1 else 0

/-- Multiply a tensor by a Bool mask (casts the mask to `0/1`). -/
def mulBoolMaskSpec {s : Shape} (t : Tensor α s) (mask : Tensor Bool s)
  : Tensor α s :=
  map2Spec (fun x b => x * boolToAlphaSpec b) t mask

/-- Apply a Huber-style clamp on entries selected by `mask` (leaves others unchanged). -/
def clampHuberMaskSpec {s : Shape}
  (t : Tensor α s) (mask : Tensor Bool s) (delta : α) : Tensor α s :=
  map2Spec (fun x m =>
    if m then
      if x > delta then delta
      else if (-delta > x) then -delta
      else x
    else
      x
  ) t mask

/-- Update a tensor at a runtime index path.

The index path is interpreted outermost-first. Out-of-bounds indices leave the tensor unchanged.
The implementation validates the path once, then performs one copy-on-write
physical-buffer replacement.
-/
def updateTensorSpec {α : Type} [TorchLean.Storage α]
    [TorchLean.Storage.Update α]
    {s : Shape} (tensor : Tensor α s) (indices : List Nat)
    (newValue : α) : Tensor α s :=
  match Shape.Coord.ofList? s indices with
  | some coordinate =>
      TorchLean.Tensor.Internal.Rep.set tensor coordinate newValue
  | none => tensor

/-- Like `updateTensorSpec`, but replaces a subtree with another tensor. -/
def updateTensorWithTensorSpec {α : Type} [TorchLean.Storage α] :
    ∀ {s : Shape}, Tensor α s → List Nat → Tensor α s → Tensor α s
  | .scalar, _, [], newTensor => newTensor
  | .scalar, tensor, _ :: _, _ => tensor
  | .dim _ _, tensor, [], _ => tensor
  | .dim n _, tensor, i :: rest, newTensor =>
      if h : i < n then
        .dim (Function.update (Tensor.unstack tensor) ⟨i, h⟩
          (updateTensorWithTensorSpec
            (Tensor.unstack tensor ⟨i, h⟩) rest
            (Tensor.unstack newTensor ⟨i, h⟩)))
      else
        tensor

/-- Specialization of `updateTensorSpec` for a top-level vector dimension. -/
def updateSpec {α : Type} [TorchLean.Storage α]
    [TorchLean.Storage.Update α]
    {n : ℕ} {s : Shape} (tensor : Tensor α (.dim n s))
    (indices : List Nat) (newValue : α) : Tensor α (.dim n s) :=
  updateTensorSpec tensor indices newValue

end TorchLean.Tensor
