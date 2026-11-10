/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Context
public import TorchLean.Tensor.OfFn -- shake: keep
public import TorchLean.Tensor.LinearAlgebra.Identity -- shake: keep

/-!
# Elementwise tensor operations

Shape-preserving elementwise operations on `Tensor α s`, most of them the packed pointwise kernels
of `Tensor.map` and `zipWith` applied to a scalar function.

## Domains and smoothness

Some operations are domain-sensitive or non-smooth:

- `Tensor.sqrt x .clampNegative` (`TorchLean/Tensor/Elementwise.lean`) takes the root of
  `max x 0` to stay total on ordered rings.
- `Tensor.log` is total as a function, and its analytic properties need positivity assumptions.
- `relu`, `clamp` and the comparisons are non-smooth. Analytic backpropagation theorems treat them
  through pointwise assumptions, or verification workflows switch to smooth surrogates.

The spec layer defines these semantics, and the proof layer decides which assumptions and variants
its theorems use.
-/

@[expose] public section


open FloatLib.Numerics (MathFunctions)

namespace TorchLean.Tensor

variable {α : Type} [TorchLean.Storage α] [Context α]

/--
Keep a tensor's primal values while removing scalar differentiation metadata.

For ordinary numeric tensors, this returns the same tensor. Dual-valued tensors need a pointwise
map as well as the graph's zero JVP and VJP: otherwise a later operation could still read a tangent
from the detached value while computing a higher-order derivative.
-/
def detach {s : Shape} (x : Tensor α s) : Tensor α s :=
  match Context.stopGradient? (α := α) with
  | none => x
  | some clear => Tensor.map clear x

/-- A pointwise binary operation computes directly on scalar tensors. -/
@[simp] theorem zipWith_scalar {α β γ : Type}
    [TorchLean.Storage α] [TorchLean.Storage β]
    [TorchLean.Storage γ]
    (f : α → β → γ) (left : α) (right : β) :
    zipWith f (Tensor.scalar left) (Tensor.scalar right) =
      Tensor.scalar (f left right) := by
  apply Tensor.Rep.ext
  intro coordinate
  cases coordinate
  simp [Tensor.scalar]

/-- Extracting a scalar after a pointwise binary map combines the two scalar values. -/
@[simp] theorem toScalar_zipWith {α β γ : Type}
    [TorchLean.Storage α] [TorchLean.Storage β]
    [TorchLean.Storage γ]
    (f : α → β → γ) (left : Tensor α .scalar)
    (right : Tensor β .scalar) :
    (zipWith f left right).item = f left.item right.item := by
  simp [Tensor.item]

/-- Indexing a pointwise binary operation applies the scalar operation after indexing. -/
@[simp] theorem unstack_zipWith {α β γ : Type}
    [TorchLean.Storage α] [TorchLean.Storage β]
    [TorchLean.Storage γ]
    {f : α → β → γ} {n : Nat} {s : Shape}
    (left : Tensor α (.dim n s)) (right : Tensor β (.dim n s)) (i : Fin n) :
    Tensor.unstack (zipWith f left right) i =
      zipWith f (Tensor.unstack left i) (Tensor.unstack right i) := by
  exact (Tensor.Rep.zipWith_unstack f left right i).symm

/-- Vector indexing commutes with a pointwise binary tensor operation. -/
@[simp] theorem getScalar_zipWith {α β γ : Type}
    [TorchLean.Storage α] [TorchLean.Storage β]
    [TorchLean.Storage γ]
    (f : α → β → γ) {n : Nat}
    (left : Tensor α [n]) (right : Tensor β [n]) (i : Fin n) :
    TorchLean.Tensor.getScalar (zipWith f left right) i =
      f (TorchLean.Tensor.getScalar left i)
        (TorchLean.Tensor.getScalar right i) := by
  rw [Tensor.getScalar_eq_apply, zipWith_apply,
    ← Tensor.getScalar_eq_apply, ← Tensor.getScalar_eq_apply]

/-- Matrix indexing commutes with a pointwise binary tensor operation. -/
@[simp] theorem get2_zipWith {α β γ : Type}
    [TorchLean.Storage α] [TorchLean.Storage β]
    [TorchLean.Storage γ]
    {f : α → β → γ} {m n : Nat}
    (left : Tensor α [m, n])
    (right : Tensor β [m, n]) (i : Fin m) (j : Fin n) :
    get2 (zipWith f left right) i j =
      f (get2 left i j) (get2 right i j) := by
  simp [get2, Tensor.getScalar,
    Tensor.item]

omit [Context α] in
/-- Matrix indexing commutes with a pointwise unary tensor operation. -/
@[simp] theorem get2_map
    {m n : Nat} (f : α → α) (tensor : Tensor α [m, n])
    (i : Fin m) (j : Fin n) :
    get2 (Tensor.map f tensor) i j =
      f (get2 tensor i j) := by
  simp [get2, Tensor.getScalar,
    Tensor.item]

/-- Transport two pointwise properties through a binary shape-preserving operation. -/
theorem forall_zipWith {α β γ : Type}
    [TorchLean.Storage α] [TorchLean.Storage β]
    [TorchLean.Storage γ]
    {p : α → Prop} {q : β → Prop} {r : γ → Prop}
    {f : α → β → γ} {s : Shape} {x : Tensor α s} {y : Tensor β s}
    (hx : Forall p x) (hy : Forall q y) (hf : ∀ a b, p a → q b → r (f a b)) :
    Forall r (zipWith f x y) := by
  rw [forall_iff] at hx hy ⊢
  intro coordinate
  simpa only [zipWith_apply] using hf (x coordinate) (y coordinate) (hx coordinate) (hy coordinate)

/-! ## Pointwise arithmetic

`x + y`, `x - y`, `x * y` and `x / y` on equally shaped tensors of one element type apply the
scalar operation at every coordinate (`zipWith`). The instances for `+` and `-` live on the
representation (`TorchLean.Tensor.Rep.Pointwise`), those for `*` and `/`
here. The promoting `Tensor.add`, `Tensor.sub`, `Tensor.mul` and `Tensor.div` of
`TorchLean.Tensor.Operations` also accept two element types. Their notation instances have low
priority, so the notation on one element type is always these. The `*_def` lemmas unfold the
notation to its `zipWith`. -/

/-- Tensor addition is `zipWith (· + ·)`. -/
theorem add_def {α : Type} [TorchLean.Storage α] [Add α] {s : Shape} (x y : Tensor α s) :
    x + y = zipWith (· + ·) x y :=
  rfl

/-- Adding the all-zero tensor on the left is the identity. -/
@[simp] theorem full_zero_add {α : Type} [TorchLean.Storage α] [AddZeroClass α] {s : Shape}
    (x : Tensor α s) : Tensor.full s (0 : α) + x = x := by
  apply Tensor.Rep.ext
  intro coordinate
  simp [add_def]

/-- Adding the all-zero tensor on the right is the identity. -/
@[simp] theorem add_full_zero {α : Type} [TorchLean.Storage α] [AddZeroClass α] {s : Shape}
    (x : Tensor α s) : x + Tensor.full s (0 : α) = x := by
  apply Tensor.Rep.ext
  intro coordinate
  simp [add_def]

/-- Shape transport commutes with pointwise tensor addition. -/
theorem castShape_add {α : Type} [TorchLean.Storage α]
    [Add α] {shape shape' : Shape}
    (left right : Tensor α shape) (h : shape = shape') :
    Tensor.castShape (left + right) h = Tensor.castShape left h + Tensor.castShape right h := by
  cases h
  rfl

/-- Tensor multiplication multiplies coordinatewise, preserving the shape. -/
instance {α : Type} [TorchLean.Storage α] [Mul α]
    {s : Shape} : Mul (Tensor α s) :=
  ⟨zipWith (· * ·)⟩

/-- Tensor multiplication is `zipWith (· * ·)`. -/
theorem mul_def {α : Type} [TorchLean.Storage α] [Mul α] {s : Shape} (x y : Tensor α s) :
    x * y = zipWith (· * ·) x y :=
  rfl

/-- Tensor multiplication is coordinatewise scalar multiplication. -/
@[simp] protected theorem mul_apply {α : Type} [TorchLean.Storage α] [Mul α] {s : Shape}
    (x y : Tensor α s) (i : s.Coord) : (x * y) i = x i * y i :=
  zipWith_apply _ x y i

/-- Scalar extraction commutes with pointwise tensor multiplication. -/
@[simp] theorem item_mul {α : Type} [TorchLean.Storage α] [Mul α]
    (left right : Tensor α .scalar) :
    (left * right).item = left.item * right.item := by
  simp [mul_def, Tensor.item]

/-- The entry of the zero scalar tensor is zero. -/
@[simp] theorem item_zero {α : Type} [TorchLean.Storage α] [Zero α] :
    (0 : Tensor α .scalar).item = 0 :=
  Tensor.Rep.zero_apply PUnit.unit

/-- Multiplication by a filled tensor is coordinatewise multiplication by its scalar value. -/
@[simp] theorem full_mul {α : Type}
    [TorchLean.Storage α] [Mul α] {s : Shape} (coefficient : α)
    (tensor : Tensor α s) :
    Tensor.full s coefficient * tensor = Tensor.map (coefficient * ·) tensor := by
  apply Tensor.Rep.ext
  intro coordinate
  simp [mul_def]

/-- Tensor subtraction is `zipWith (· - ·)`. -/
theorem sub_def {α : Type} [TorchLean.Storage α] [Sub α] {s : Shape} (x y : Tensor α s) :
    x - y = zipWith (· - ·) x y :=
  rfl

/-- Tensor division divides coordinatewise, preserving the shape. -/
instance {α : Type} [TorchLean.Storage α] [Div α] {s : Shape} : Div (Tensor α s) :=
  ⟨zipWith (· / ·)⟩

/-- Tensor division is `zipWith (· / ·)`. -/
theorem div_def {α : Type} [TorchLean.Storage α] [Div α] {s : Shape} (x y : Tensor α s) :
    x / y = zipWith (· / ·) x y :=
  rfl

/-- Tensor division is coordinatewise scalar division. -/
@[simp] protected theorem div_apply {α : Type} [TorchLean.Storage α] [Div α] {s : Shape}
    (x y : Tensor α s) (i : s.Coord) : (x / y) i = x i / y i :=
  zipWith_apply _ x y i

/-- Epsilon-shifted division, $x/(y+\varepsilon)$ with $\varepsilon$ the backend's
`Context.defaultEpsilon`. The shifted denominator can be zero. -/
def safeDiv {s : Shape} (t1 t2 : Tensor α s) : Tensor α s :=
  zipWith (fun x y => x / (y + Context.defaultEpsilon)) t1 t2

/-- Scale a tensor by a scalar. -/
def scale {α : Type} [TorchLean.Storage α] [Mul α]
    {s : Shape} (t : Tensor α s) (scalar : α) : Tensor α s :=
  Tensor.map (fun x => x * scalar) t

/-- Square each element of a tensor. -/
def square {α : Type} [TorchLean.Storage α] [Mul α]
    {s : Shape} (t : Tensor α s) : Tensor α s :=
  Tensor.map (fun x => x * x) t

/-- Squaring is elementwise multiplication with the same tensor on both inputs. -/
theorem square_eq_mul {α : Type} [TorchLean.Storage α] [Mul α]
    {s : Shape} (t : Tensor α s) :
    square t = t * t := by
  apply Tensor.Rep.ext
  intro coordinate
  simp [square, mul_def]

/-- Absolute value of each element. -/
def abs {s : Shape} (t : Tensor α s) : Tensor α s :=
  Tensor.map MathFunctions.abs t

/-- Elementwise natural logarithm. -/
def log {s : Shape} (t : Tensor α s) : Tensor α s :=
  Tensor.map MathFunctions.log t

/-- Elementwise exponential. -/
def exp {s : Shape} (t : Tensor α s) : Tensor α s :=
  Tensor.map MathFunctions.exp t

/-- Elementwise negation. -/
def neg {α : Type} [TorchLean.Storage α] [Neg α]
    {s : Shape} (t : Tensor α s) : Tensor α s :=
  Tensor.map Neg.neg t

/-- Elementwise power, each entry of `t1` raised to the matching entry of `t2`. -/
def pow {s : Shape} (t1 t2 : Tensor α s) : Tensor α s :=
  zipWith HPow.hPow t1 t2

/-- Elementwise sine, with each input interpreted as an angle in radians. -/
def sin {s : Shape} : Tensor α s → Tensor α s :=
  Tensor.map MathFunctions.sin

/-- Elementwise cosine, with each input interpreted as an angle in radians. -/
def cos {s : Shape} : Tensor α s → Tensor α s :=
  Tensor.map MathFunctions.cos

/-- Elementwise hyperbolic cosine. -/
def cosh {s : Shape} : Tensor α s → Tensor α s :=
  Tensor.map MathFunctions.cosh

/-- Elementwise hyperbolic sine. -/
def sinh {s : Shape} : Tensor α s → Tensor α s :=
  Tensor.map MathFunctions.sinh

/-- Update a tensor at a runtime index path.

The path is read outermost first, and a path that is not one in-bounds index per axis leaves the
tensor unchanged. The implementation validates the path once and then performs one copy-on-write
replacement in the physical buffer.
-/
def update {α : Type} [TorchLean.Storage α]
    [TorchLean.Storage.Update α]
    {s : Shape} (tensor : Tensor α s) (indices : List Nat)
    (newValue : α) : Tensor α s :=
  match Shape.Coord.ofList? s indices with
  | some coordinate =>
      Tensor.set tensor coordinate newValue
  | none => tensor

/-- Adding to the entry at a runtime index path changes only the coordinate the path names. -/
theorem update_add_apply {α : Type} [TorchLean.Storage α]
    [TorchLean.Storage.Update α] [Zero α] [Add α] {s : Shape} (tensor : Tensor α s)
    (indices : List Nat) (increment : α) (coordinate : s.Coord) :
    update tensor indices (getAtOrZero tensor indices + increment) coordinate =
      if Shape.Coord.ofList? s indices = some coordinate then tensor coordinate + increment
      else tensor coordinate := by
  unfold update getAtOrZero get?
  cases h : Shape.Coord.ofList? s indices with
  | none => simp
  | some updated =>
      by_cases same : updated = coordinate
      · subst same
        simp
      · simp [same, Tensor.Rep.get_set_ne _ _ same]

end TorchLean.Tensor
