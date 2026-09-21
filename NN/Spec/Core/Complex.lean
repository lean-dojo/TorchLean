/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Context
public import NN.Core.Numeric.Angle

/-!
# Complex scalar (`TorchLean.Complex α`)

TorchLean is scalar-polymorphic, and some model components (e.g. FFT/FNO-style blocks) want a
complex-valued scalar type.

Mathlib’s `ℂ` is specialized to `ℝ` and intentionally has no order instance; TorchLean’s generic
`Context` includes order-like operations (`LT/LE`, `max/min`) for ReLU/argmax-style code paths.

To avoid changing mathlib’s global behavior (and to support runtime-friendly backends like
`ExecFloat.Binary 8 23`), we provide a small parametric complex scalar:

`TorchLean.Complex α := α × α` with fields `re` and `im`.

Complex square roots use the principal branch. Complex logarithms retain the polar angle through
`Atan2 α`; a backend must supply that real-coordinate operation to obtain the complex `Context`.
Arithmetic operations inherit the rounding and exceptional-value behavior of the component type.

The `Context` instance supports explicit complex programs. For real-valued losses, the application
API `autograd.complex.grad` differentiates both real coordinates by forward-mode seeding; its
gradients work with `nn.sgdStep`, and `Checkpoint.State` preserves both components. The ordinary
supervised trainer still has a real-valued data and result boundary. Complex-linear reverse rules
must not be substituted for real-coordinate differentiation of a nonholomorphic loss.
-/

@[expose] public section

namespace TorchLean

/-- Parametric complex numbers $a+ib$ over a scalar type `α`. -/
structure Complex (α : Type) where
  /-- Real part. -/
  re : α
  /-- Imaginary part. -/
  im : α
  deriving Repr

namespace Complex

variable {α : Type}

/-- Embed a real scalar as a complex scalar with zero imaginary part. -/
def ofReal [Zero α] (a : α) : Complex α := ⟨a, 0⟩

/-- Complex conjugation keeps the real coordinate and negates the imaginary coordinate. -/
def conj [Neg α] (z : Complex α) : Complex α := ⟨z.re, -z.im⟩

/-- Squared magnitude, returned in the real component type. -/
def normSq [Mul α] [Add α] (z : Complex α) : α := z.re * z.re + z.im * z.im

/-- The real part of a real embedded as a complex number is itself. -/
@[simp] theorem re_ofReal [Zero α] (a : α) : (ofReal a).re = a := rfl
/-- A real embedded as a complex number has zero imaginary part. -/
@[simp] theorem im_ofReal [Zero α] (a : α) : (ofReal a).im = 0 := rfl

/-- Imaginary unit `i`. -/
def I [Zero α] [One α] : Complex α := ⟨0, 1⟩

/-- `i` has zero real part. -/
@[simp] theorem re_I [Zero α] [One α] : (I (α := α)).re = 0 := rfl
/-- `i` has imaginary part one. -/
@[simp] theorem im_I [Zero α] [One α] : (I (α := α)).im = 1 := rfl

/-! ## Basic algebraic structure -/

instance [Inhabited α] [Zero α] : Inhabited (Complex α) := ⟨⟨default, 0⟩⟩
instance [Zero α] : Zero (Complex α) := ⟨⟨0, 0⟩⟩
instance [One α] [Zero α] : One (Complex α) := ⟨⟨1, 0⟩⟩
instance [Neg α] : Neg (Complex α) := ⟨fun z => ⟨-z.re, -z.im⟩⟩
instance [Add α] : Add (Complex α) := ⟨fun x y => ⟨x.re + y.re, x.im + y.im⟩⟩
instance [Sub α] : Sub (Complex α) := ⟨fun x y => ⟨x.re - y.re, x.im - y.im⟩⟩

/-- Complex multiplication using the usual real/imaginary component formula. -/
instance [Mul α] [Add α] [Sub α] : Mul (Complex α) :=
  ⟨fun x y =>
    ⟨x.re * y.re - x.im * y.im, x.re * y.im + x.im * y.re⟩⟩

/-!
Division uses the standard formula
$$
\frac{a+bi}{c+di}=\frac{(ac+bd)+i(bc-ad)}{c^2+d^2}.
$$
-/
instance [Mul α] [Add α] [Sub α] [Div α] : Div (Complex α) :=
  ⟨fun x y =>
    let denom : α := y.re * y.re + y.im * y.im
    let re' : α := (x.re * y.re + x.im * y.im) / denom
    let im' : α := (x.im * y.re - x.re * y.im) / denom
    ⟨re', im'⟩⟩

instance [BEq α] : BEq (Complex α) :=
  ⟨fun x y => x.re == y.re && x.im == y.im⟩

/-!
Order is only used in TorchLean for branchy ops like ReLU/max/min. Complex numbers do not have a
canonical order, so we pick a simple *real-part* order: compare `re` and ignore `im`.

This instance is local to TorchLean’s branchy tensor operations and does not change mathlib’s `ℂ`.
-/
instance [LT α] : LT (Complex α) := ⟨fun x y => x.re < y.re⟩
instance [LE α] : LE (Complex α) := ⟨fun x y => x.re ≤ y.re⟩

instance [ToString α] : ToString (Complex α) where
  toString z := s!"({toString z.re} + {toString z.im}i)"

/-! ## Numeric literals and constants -/

instance [Context α] : NatCast (Complex α) where
  natCast n := ofReal (n : α)

/-! ## Transcendentals -/

namespace Internal

/-- Real magnitude evaluated with scaling to avoid unnecessary overflow from squaring.

This is mathematically `sqrt(re² + im²)` for real coordinates; floating-point evaluation can differ
in the last bits from the unscaled expression. No IEEE complex special-value contract is asserted.
-/
def absReal [Context α] (z : Complex α) : α :=
  let scale := Max.max (MathFunctions.abs z.re) (MathFunctions.abs z.im)
  if scale == 0 then 0 else
    let a := z.re / scale
    let b := z.im / scale
    scale * MathFunctions.sqrt (a * a + b * b)

end Internal

instance [Context α] [Atan2 α] : MathFunctions (Complex α) where
  exp z :=
    -- exp(a+bi) = exp(a) * (cos(b) + i sin(b))
    let ea := MathFunctions.exp z.re
    ⟨ea * MathFunctions.cos z.im, ea * MathFunctions.sin z.im⟩
  tanh z :=
    -- tanh(z) = sinh(z) / cosh(z)
    let sh : Complex α :=
      ⟨MathFunctions.sinh z.re * MathFunctions.cos z.im,
        MathFunctions.cosh z.re * MathFunctions.sin z.im⟩
    let ch : Complex α :=
      ⟨MathFunctions.cosh z.re * MathFunctions.cos z.im,
        MathFunctions.sinh z.re * MathFunctions.sin z.im⟩
    sh / ch
  cosh z :=
    -- cosh(a+bi) = cosh(a)cos(b) + i sinh(a)sin(b)
    ⟨MathFunctions.cosh z.re * MathFunctions.cos z.im,
      MathFunctions.sinh z.re * MathFunctions.sin z.im⟩
  sqrt z :=
    if z.im == 0 then
      if z.re > 0 then
        let root := MathFunctions.sqrt z.re
        ⟨root, z.im / (2 * root)⟩
      else
        let magnitude := MathFunctions.sqrt (MathFunctions.abs z.re)
        ⟨0, if Atan2.atan2 z.im z.re < 0 then -magnitude else magnitude⟩
    else
      -- Scaling avoids squaring large components or subtracting nearly equal magnitudes.
      let scale := Max.max (MathFunctions.abs z.re) (MathFunctions.abs z.im)
      let a := z.re / scale
      let b := z.im / scale
      let radius := MathFunctions.sqrt (a * a + b * b)
      let t := MathFunctions.sqrt scale *
        MathFunctions.sqrt ((radius + MathFunctions.abs a) / (2 : α))
      if z.re > 0 then
        ⟨t, (z.im / t) / (2 : α)⟩
      else
        ⟨(MathFunctions.abs z.im / t) / (2 : α), if z.im < 0 then -t else t⟩
  abs z := ofReal (Internal.absReal z)
  log z :=
    let scale := Max.max (MathFunctions.abs z.re) (MathFunctions.abs z.im)
    let magnitudeLog := if scale == 0 then MathFunctions.log scale else
      let a := z.re / scale
      let b := z.im / scale
      MathFunctions.log scale + MathFunctions.log (a * a + b * b) / (2 : α)
    ⟨magnitudeLog, Atan2.atan2 z.im z.re⟩
  pi := ofReal MathFunctions.pi
  cos z :=
    -- cos(a+bi) = cos(a)cosh(b) - i sin(a)sinh(b)
    ⟨MathFunctions.cos z.re * MathFunctions.cosh z.im,
      -(MathFunctions.sin z.re * MathFunctions.sinh z.im)⟩
  sin z :=
    -- sin(a+bi) = sin(a)cosh(b) + i cos(a)sinh(b)
    ⟨MathFunctions.sin z.re * MathFunctions.cosh z.im,
      MathFunctions.cos z.re * MathFunctions.sinh z.im⟩
  sinh z :=
    -- sinh(a+bi) = sinh(a)cos(b) + i cosh(a)sin(b)
    ⟨MathFunctions.sinh z.re * MathFunctions.cos z.im,
      MathFunctions.cosh z.re * MathFunctions.sin z.im⟩

/-! ## max/min and `Context` -/

instance [Context α] : Max (Complex α) where
  max x y := if x.re > y.re then x else y

instance [Context α] : Min (Complex α) where
  min x y := if x.re > y.re then y else x

instance [Context α] [Atan2 α] : Pow (Complex α) (Complex α) where
  pow x y := MathFunctions.exp (y * MathFunctions.log x)

/-- Lift a scalar `Context` to TorchLean complex scalars. -/
instance [Context α] [Atan2 α] : Context (Complex α) where
  defaultEpsilon := ofReal (Context.defaultEpsilon (α := α))
  decidableGT := fun x y => (Context.decidableGT) x.re y.re
  stopGradient? := (Context.stopGradient? (α := α)).map fun clear z =>
    ⟨clear z.re, clear z.im⟩
  ratCast value := ofReal (value : α)

end Complex

end TorchLean
