/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Context

/-!
# Complex scalar (`TorchLean.Complex α`)

TorchLean is scalar-polymorphic, and some model components (e.g. FFT/FNO-style blocks) want a
complex-valued scalar type.

Mathlib’s `ℂ` is specialized to `ℝ` and intentionally has no order instance; TorchLean’s generic
`Context` includes order-like operations (`LT/LE`, `max/min`) for ReLU/argmax-style code paths.

To avoid changing mathlib’s global behavior (and to support runtime-friendly backends like
`IEEE32Exec`), we provide a small parametric complex scalar:

`TorchLean.Complex α := α × α` with fields `re` and `im`.

The provided `Context` instance is designed to be:
- good enough for TorchLean’s tensor/model surface (arithmetic + common transcendental functions),
- runtime-friendly when `α` is runtime-friendly, and
- conservative for the few operations that are fundamentally about angles/branches (`log`, `sqrt`):
  we pick a simple real-part-based approximation that is exact on real inputs (imag part `0`).

This is *not* meant to be a full replacement for complex analysis; for deep analytic theorems, use
mathlib’s `ℂ` directly.
-/

@[expose] public section

namespace TorchLean

/-- Parametric complex numbers `a + i·b` over a scalar type `α`. -/
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

@[simp] theorem re_ofReal [Zero α] (a : α) : (ofReal a).re = a := rfl
@[simp] theorem im_ofReal [Zero α] (a : α) : (ofReal a).im = 0 := rfl

/-- Imaginary unit `i`. -/
def I [Zero α] [One α] : Complex α := ⟨0, 1⟩

@[simp] theorem re_I [Zero α] [One α] : (I (α := α)).re = 0 := rfl
@[simp] theorem im_I [Zero α] [One α] : (I (α := α)).im = 1 := rfl

/-! ## Basic algebraic structure -/

instance [Inhabited α] [Zero α] : Inhabited (Complex α) := ⟨⟨default, 0⟩⟩
instance [Zero α] : Zero (Complex α) := ⟨⟨0, 0⟩⟩
instance [One α] [Zero α] : One (Complex α) := ⟨⟨1, 0⟩⟩
instance [Neg α] : Neg (Complex α) := ⟨fun z => ⟨-z.re, -z.im⟩⟩
instance [Add α] : Add (Complex α) := ⟨fun x y => ⟨x.re + y.re, x.im + y.im⟩⟩
instance [Sub α] : Sub (Complex α) := ⟨fun x y => ⟨x.re - y.re, x.im - y.im⟩⟩

/-- Complex multiplication. -/
instance [Mul α] [Add α] [Sub α] : Mul (Complex α) :=
  ⟨fun x y =>
    ⟨x.re * y.re - x.im * y.im, x.re * y.im + x.im * y.re⟩⟩

/-!
Division uses the standard formula
`(a+bi)/(c+di) = ((ac+bd) + i(bc-ad)) / (c^2 + d^2)`.
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

This keeps the instance lightweight and avoids polluting mathlib’s `ℂ` with ad-hoc orderings.
-/
instance [LT α] : LT (Complex α) := ⟨fun x y => x.re < y.re⟩
instance [LE α] : LE (Complex α) := ⟨fun x y => x.re ≤ y.re⟩

instance [ToString α] : ToString (Complex α) where
  toString z := s!"({toString z.re} + {toString z.im}i)"

/-! ## Numeric literals and constants -/

instance [Context α] : Coe Nat (Complex α) where
  coe n := ofReal (n : α)

instance [Numbers α] [Zero α] : Numbers (Complex α) where
  neg_point_five := ofReal Numbers.neg_point_five
  neg_one := ofReal Numbers.neg_one
  pointone := ofReal Numbers.pointone
  pointfive := ofReal Numbers.pointfive
  one := ofReal Numbers.one
  zero := ofReal Numbers.zero
  two := ofReal Numbers.two
  three := ofReal Numbers.three
  four := ofReal Numbers.four
  five := ofReal Numbers.five
  ten := ofReal Numbers.ten
  log10 := ofReal Numbers.log10
  log10000 := ofReal Numbers.log10000
  epsilon := ofReal Numbers.epsilon
  neg_thousand := ofReal Numbers.neg_thousand

/-! ## Transcendentals -/

namespace Internal

/-- Squared magnitude `re^2 + im^2` (helper for `abs/log`). -/
def normSq [Mul α] [Add α] (z : Complex α) : α := z.re * z.re + z.im * z.im

/-- Real magnitude `sqrt(normSq z)`. -/
def absReal [Context α] (z : Complex α) : α := MathFunctions.sqrt (normSq z)

end Internal

instance [Context α] : MathFunctions (Complex α) where
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
    -- A small branch choice:
    -- - exact for real inputs (`im = 0`): sqrt(x) for x>=0, i*sqrt(-x) for x<0
    -- - otherwise: return sqrt(|z|) as a real (imag=0) approximation
    if z.im == 0 then
      if z.re > 0 then
        ofReal (MathFunctions.sqrt z.re)
      else
        ⟨0, MathFunctions.sqrt (MathFunctions.abs z.re)⟩
    else
      ofReal (MathFunctions.sqrt (Internal.absReal z))
  abs z := ofReal (Internal.absReal z)
  log z :=
    -- We intentionally ignore the complex argument; this is exact on positive reals.
    ofReal (MathFunctions.log (Internal.absReal z))
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

instance [Context α] : Pow (Complex α) (Complex α) where
  pow x y := MathFunctions.exp (y * MathFunctions.log x)

/-- Lift a scalar `Context` to TorchLean complex scalars. -/
instance [Context α] : Context (Complex α) where
  decidable_gt := fun x y => (Context.decidable_gt) x.re y.re

end Complex

end TorchLean
