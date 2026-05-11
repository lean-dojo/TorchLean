/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Context
public import NN.Spec.Core.Tensor.Core

/-!
# Elementwise tensor operations (`Spec.Tensor.*_spec`)

This file defines shape-preserving, elementwise operations on `Tensor α s`.

Naming convention:

- `foo_spec` means “pure spec definition” (no runtime side effects).
- most functions are defined by recursion on the tensor structure via `map_spec` / `map2_spec`.

## Domain / smoothness notes

Some ops are inherently domain-sensitive or non-smooth:

- `sqrt_spec` uses `sqrt (max x 0)` to stay total on ordered rings.
- `log_spec` is total as a function call, but analytic properties require positivity assumptions.
- `relu` / `clamp` / comparisons are non-smooth; analytic backprop theorems treat these via
  pointwise assumptions, or by switching to smooth surrogates in verification workflows.

The **spec layer** is where these semantics are defined; the **proof layer** decides which
assumptions/variants to use for theorems.
-/

@[expose] public section


namespace Spec
namespace Tensor

variable {α : Type} [Context α]

/--
Map a scalar function over a tensor (shape preserved).

This is the core "structural recursion" combinator for spec tensors.
Most elementwise ops are direct instances of `mapSpec f`.

PyTorch analogy: `f` applied pointwise (like `torch.<op>` broadcasting over all entries),
but here shape is fixed and enforced by the type.
-/
def mapSpec {s : Shape} (f : α → α) : Tensor α s → Tensor α s
  | Tensor.scalar x => Tensor.scalar (f x)
  | Tensor.dim g => Tensor.dim (fun i => mapSpec f (g i))

/--
Map a binary function over two tensors of the same shape.

This is the "zipWith" combinator for the spec tensor tree.
It is intentionally *shape-preserving*: if the shapes differ, the term is not well-typed.

PyTorch analogy: elementwise binary ops when tensors already have the same shape (no broadcasting).
Broadcasting is handled separately in `NN/Spec/Core/TensorReductionShape.lean`.
-/
def map2Spec {α β γ : Type} (f : α → β → γ) : ∀ {s : Shape}, Tensor α s → Tensor β s → Tensor γ s
  | Shape.scalar, Tensor.scalar x, Tensor.scalar y => Tensor.scalar (f x y)
  | Shape.dim _ _, Tensor.dim fx, Tensor.dim fy => Tensor.dim (fun i => map2Spec f (fx i) (fy i))

/-- Element‑wise addition (shape preserved). -/
def addSpec {α : Type} [Add α] {s : Shape} (T₁ T₂ : Tensor α s) : Tensor α s :=
  map2Spec (· + ·) T₁ T₂

/-- `Add` instance for shape-indexed tensors: add pointwise, preserving the shape. -/
instance {α : Type} [Add α] {s : Shape} : Add (Tensor α s) :=
  ⟨addSpec⟩

/-- Element‑wise multiplication (shape preserved). -/
def mulSpec {α : Type} [Mul α] {s : Shape} (T₁ T₂ : Tensor α s) : Tensor α s :=
  map2Spec (· * ·) T₁ T₂

/-- `Mul` instance for shape-indexed tensors: multiply pointwise, preserving the shape. -/
instance {α : Type} [Mul α] {s : Shape} : Mul (Tensor α s) :=
  ⟨mulSpec⟩

/-- Element‑wise subtraction (shape preserved). -/
def subSpec {α : Type} [Sub α] {s : Shape} : Tensor α s → Tensor α s → Tensor α s :=
  map2Spec (· - ·)

/-- `Sub` instance for shape-indexed tensors: subtract pointwise, preserving the shape. -/
instance {α : Type} [Sub α] {s : Shape} : Sub (Tensor α s) :=
  ⟨subSpec⟩

/-- Element‑wise division (shape preserved). -/
def divSpec {s : Shape} : Tensor α s → Tensor α s → Tensor α s :=
  map2Spec (· / ·)

/-- `Div` instance for shape-indexed tensors: divide pointwise, preserving the shape. -/
instance {s : Shape} : Div (Tensor α s) :=
  ⟨divSpec⟩

/-- Safe division with epsilon protection (`x / (y + ε)`). -/
def safedivSpec {s : Shape} (t1 t2 : Tensor α s) : Tensor α s :=
  map2Spec (fun x y => x / (y + Numbers.epsilon)) t1 t2

/-- Scale a tensor by a scalar. -/
def scaleSpec {α : Type} [Mul α] {s : Shape} (t : Tensor α s) (scalar : α) : Tensor α s :=
  mapSpec (fun x => x * scalar) t

/-- Square each element of a tensor. -/
def squareSpec {s : Shape} (t : Tensor α s) : Tensor α s :=
  mapSpec (fun x => x * x) t

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

/-- `Neg` instance for shape-indexed tensors: negate pointwise, preserving the shape. -/
instance {s : Shape} : Neg (Tensor α s) :=
  ⟨negSpec⟩

/-- Multiply tensor by a constant. -/
def mulConstantSpec {s : Shape} (t : Tensor α s) (constant : α) : Tensor α s :=
  mapSpec (fun x => x * constant) t

/-- Element‑wise power. -/
def powSpec {s : Shape} (t1 t2 : Tensor α s) : Tensor α s :=
  map2Spec HPow.hPow t1 t2

/-- Element‑wise comparisons (returning Bool tensors). -/
def greaterThanSpec {s : Shape} (x y : Tensor α s) : Tensor Bool s :=
  map2Spec (fun a b => decide (a > b)) x y

/-- Element‑wise `≤` test, implemented via `¬(>)` so we only depend on `DecidableRel (·>·)`. -/
def lessEqualSpec {s : Shape} (x y : Tensor α s) : Tensor Bool s :=
  map2Spec (fun a b => decide (¬(a > b))) x y

/-- Element‑wise `<` test (defined as `y > x`). -/
def lessThanSpec {s : Shape} (x y : Tensor α s) : Tensor Bool s :=
  map2Spec (fun a b => decide (b > a)) x y

/-- Element‑wise `≥` test (defined as `¬(y > x)`). -/
def greaterEqualSpec {s : Shape} (x y : Tensor α s) : Tensor Bool s :=
  map2Spec (fun a b => decide (¬(b > a))) x y

/-- Boolean NOT, pointwise on a Bool tensor. -/
def notSpec {s : Shape} (x : Tensor Bool s) : Tensor Bool s :=
  mapSpec Bool.not x

/-- Element‑wise reciprocal (`1/x`). -/
def invSpec {s : Shape} (x : Tensor α s) : Tensor α s :=
  mapSpec (fun x => 1 / x) x

/-- Element‑wise clamp into `[min_val, max_val]`. -/
def clampSpec {s : Shape} (x : Tensor α s) (min_val max_val : α) : Tensor α s :=
  mapSpec (fun v => Min.min max_val (Max.max min_val v)) x

/-- Element‑wise minimum. -/
def minSpec {s : Shape} (t1 t2 : Tensor α s) : Tensor α s :=
  map2Spec (fun x y => Min.min x y) t1 t2

/-- Element‑wise maximum. -/
def maxSpec {s : Shape} (t1 t2 : Tensor α s) : Tensor α s :=
  map2Spec (fun x y => Max.max x y) t1 t2

/-- Element‑wise sign function: returns `-1`, `0`, or `1`. -/
def signSpec {s : Shape} (t : Tensor α s) : Tensor α s :=
  mapSpec (fun x => if x > 0 then 1 else if x < 0 then -1 else 0) t

/-- Element‑wise cosh. -/
def coshSpec {s : Shape} : Tensor α s → Tensor α s :=
  mapSpec MathFunctions.cosh

/-- Element‑wise sinh. -/
def sinhSpec {s : Shape} : Tensor α s → Tensor α s :=
  mapSpec MathFunctions.sinh

/-- Derivative mask for clamp: `1` strictly inside `(min_val, max_val)`, else `0`. -/
def clampDerivativeSpec {s : Shape} (x : Tensor α s) (min_val max_val : α) : Tensor α s :=
  mapSpec (fun v => if v > min_val ∧ v < max_val then 1 else 0) x

/-- Numeric mask: `1` where `a > b`, else `0`. -/
def gtMaskSpec {s : Shape} (a b : Tensor α s) : Tensor α s :=
  map2Spec (fun x y => if x > y then 1 else 0) a b

/-- Numeric mask: `1` where `a < b`, else `0`. -/
def ltMaskSpec {s : Shape} (a b : Tensor α s) : Tensor α s :=
  map2Spec (fun x y => if x < y then 1 else 0) a b

/-- Convert a Bool to `α` using `1`/`0`. -/
def boolToAlphaSpec [One α] [Zero α] : Bool → α :=
  fun b => if b then 1 else 0

/-- Multiply a tensor by a Bool mask (casts the mask to `0/1`). -/
def mulBoolMaskSpec [Mul α] [One α] [Zero α] {s : Shape} (t : Tensor α s) (mask : Tensor Bool s)
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

/-- Update a tensor at a (runtime) index path.

The index path is interpreted outermost-first. Out-of-bounds indices leave the tensor unchanged.
This is an executable convenience helper; most proof-facing code prefers total, shape-indexed access.
-/
def updateTensorSpec {α : Type} : ∀ {s : Shape}, Tensor α s → List Nat → α → Tensor α s
  | .scalar, .scalar _, [], new_val => .scalar new_val
  | .scalar, .scalar val, _ :: _, _ => .scalar val  -- Can't index into scalar
  | .dim _ _, .dim values, [], _ => .dim values     -- No index provided
  | .dim n _, .dim values, i :: rest, new_val =>
      if h : i < n then
        .dim (Function.update values ⟨i, h⟩
          (updateTensorSpec (values ⟨i, h⟩) rest new_val))
      else
        .dim values  -- Index out of bounds

/-- Like `update_tensor_spec`, but replaces a subtree with another tensor. -/
def updateTensorWithTensorSpec {α : Type} : ∀ {s : Shape}, Tensor α s → List Nat → Tensor α s →
  Tensor α s
  | .scalar, Tensor.scalar _, [], new_tensor => new_tensor
  | .scalar, Tensor.scalar val, _ :: _, _ => Tensor.scalar val  -- Can't index into scalar
  | .dim _ _, Tensor.dim values, [], _ => Tensor.dim values     -- No index provided
  | .dim n _, Tensor.dim values, i :: rest, new_tensor =>
      if h : i < n then
        .dim (Function.update values ⟨i, h⟩
          (updateTensorWithTensorSpec (values ⟨i, h⟩) rest (match new_tensor with
            | Tensor.dim new_values => new_values ⟨i, h⟩)))
      else
        .dim values  -- Index out of bounds

/-- Specialization of `update_tensor_spec` for a top-level vector dimension. -/
def updateSpec {α : Type} {n : ℕ} {s : Shape} :
     Tensor α (.dim n s) → List Nat → α → Tensor α (.dim n s)
  | .dim values, [], _ => .dim values  -- No index provided, return original
  | .dim values, i :: rest, new_val =>
    if h : i < n then
      .dim (Function.update values ⟨i, h⟩
         (updateTensorSpec (values ⟨i, h⟩) rest new_val))
    else
      .dim values  -- Index out of bounds, return original

/-- Slice a vector `[start, start+len)` (with a proof that the slice stays in-bounds). -/
def sliceVectorSpec {α : Type} {n : Nat}
  (v : Tensor α (.dim n .scalar))
  (start len : Nat) (h : start + len ≤ n := by decide) :
  Tensor α (.dim len .scalar) :=
  match v with
  | .dim f =>
    .dim fun i =>
      f ⟨start + i.val, Nat.lt_of_lt_of_le
          (Nat.add_lt_add_left i.is_lt start)
          h⟩


end Tensor
end Spec
