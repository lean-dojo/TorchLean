/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Context
public import TorchLean.Tensor.Numeric.InverseTrig

/-!
# Complex scalar (`TorchLean.Complex α`)

TorchLean is scalar-polymorphic, and some model components (e.g. FFT/FNO-style blocks) want a
complex-valued scalar type.

Mathlib’s `ℂ` is specialized to `ℝ` and intentionally has no order instance, while TorchLean’s
generic `Context` includes order-like operations (`LT/LE`, `max/min`) for ReLU/argmax-style code
paths.

To avoid changing mathlib’s global behavior (and to support runtime-friendly backends like
`ExecFloat.Binary 8 23`), we provide a small parametric complex scalar:
`TorchLean.Complex α` stores two coordinates with fields `re` and `im`.

Complex square roots use the principal branch. Complex logarithms take their imaginary part, the
polar angle, from `InverseTrig.atan2`, so a backend must supply `InverseTrig α` to obtain the
complex `Context`. The real part of a logarithm is `log s + log (a² + b²) / 2`, with
`s = max |re| |im|` and `a`, `b` the parts divided by `s`, so near `|z| = 1` it loses its low
digits: `log (1 + 10⁻¹⁰ i)` has real part `0`, not `5·10⁻²¹`, and `z ^ w = exp (w log z)`
inherits the error. The inverse hyperbolic sine and cosine are the principal branches, by Kahan's
formulas. The hyperbolic tangent of `x + iy` is
`(sinh x cosh x + i sin y cos y) / (sinh² x + cos² y)` with numerator and denominator divided by
`cosh² x`, which is never formed. The tangent of `z` is `-i tanh (i z)`. At a zero imaginary part
`tanh` and `asinh` are the real functions of the real part, with that zero. At an infinite part
`acosh` is `+∞ + i atan2 y x`, the value of C99's `cacosh` (Annex G.6.2.1). Its imaginary part is
`±π/2` at `x ± ∞i` for a finite `x`, and `±0` at `+∞ ± yi` and `±π` at `-∞ ± yi` for a finite `y`.
It is `±π/4` at `+∞ ± ∞i`, `±3π/4` at `-∞ ± ∞i` and NaN when the other part is a NaN. At an infinite
part `asinh` is `±∞ + i atan2 y |x|` with the sign of `x`, the value of C99's `casinh` (Annex
G.6.2.2). Its imaginary part is `±π/2` at `x ± ∞i` for a finite `x`, and `±0` at `+∞ ± yi` and
`-∞ ± yi` for a finite `y`. It is `±π/4` at `+∞ ± ∞i` and `-∞ ± ∞i` and NaN when the other part is a
NaN. A sum in Kahan's formulas that overflows at a finite `z` is recomputed from halved factors. The
test is `Context.isInf` of the computed sum, so in a format without infinities, where an overflow
gives a NaN or the largest finite value, the sum is used as computed. Arithmetic operations inherit
the rounding and exceptional-value behavior of the component type.

We transcribed `asinh` and `acosh` by hand from section 9, "The Procedures", of W. Kahan (1987),
"Branch Cuts for Complex Elementary Functions, or Much Ado About Nothing's Sign Bit", in
A. Iserles and M. J. D. Powell (eds.), the proceedings of the joint IMA/SIAM conference at the
University of Birmingham in April 1986 (Clarendon Press, Oxford), 165-211. `asinh` is his
`CASINH(z) = -i CASIN(i z)` and `acosh` his `CACOSH`. "C99" above means the WG14 draft N1256 of
ISO/IEC 9899:TC3 (2007), whose Annex G we read for the infinite and NaN cases.

The `Context` instance supports complex programs written against it. For real-valued losses, the
application API `autograd.complex.grad` differentiates both real coordinates by forward-mode
seeding. Its gradients work with `nn.sgdStep`, and `Checkpoint.State` preserves both components.
The ordinary supervised trainer has a real-valued data and result boundary. Complex-linear reverse
rules must not be substituted for real-coordinate differentiation of a nonholomorphic loss.
-/

@[expose] public section

open FloatLib.Numerics (MathFunctions)

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

/-- The default real part with imaginary part zero. -/
instance [Inhabited α] [Zero α] : Inhabited (Complex α) := ⟨⟨default, 0⟩⟩
/-- Zero is `0 + 0i`. -/
instance [Zero α] : Zero (Complex α) := ⟨⟨0, 0⟩⟩
/-- One is `1 + 0i`. -/
instance [One α] [Zero α] : One (Complex α) := ⟨⟨1, 0⟩⟩
/-- Negation negates both parts. -/
instance [Neg α] : Neg (Complex α) := ⟨fun z => ⟨-z.re, -z.im⟩⟩
/-- Addition adds the parts. -/
instance [Add α] : Add (Complex α) := ⟨fun x y => ⟨x.re + y.re, x.im + y.im⟩⟩
/-- Subtraction subtracts the parts. -/
instance [Sub α] : Sub (Complex α) := ⟨fun x y => ⟨x.re - y.re, x.im - y.im⟩⟩

/-- Complex multiplication by the component formula `(a + bi)(c + di) = (ac - bd) + (ad + bc)i`. -/
instance [Mul α] [Add α] [Sub α] : Mul (Complex α) :=
  ⟨fun x y =>
    ⟨x.re * y.re - x.im * y.im, x.re * y.im + x.im * y.re⟩⟩

/--
Division uses the standard formula
$$
\frac{a+bi}{c+di}=\frac{(ac+bd)+i(bc-ad)}{c^2+d^2}.
$$
The denominator is computed as written, so in a binary format it can overflow or underflow where
the quotient is representable.
-/
instance [Mul α] [Add α] [Sub α] [Div α] : Div (Complex α) :=
  ⟨fun x y =>
    let denom : α := y.re * y.re + y.im * y.im
    let re' : α := (x.re * y.re + x.im * y.im) / denom
    let im' : α := (x.im * y.re - x.re * y.im) / denom
    ⟨re', im'⟩⟩

/-- `x == y` holds when both parts are equal under `==`. -/
instance [BEq α] : BEq (Complex α) :=
  ⟨fun x y => x.re == y.re && x.im == y.im⟩

/-!
TorchLean uses order only in branching operations such as ReLU, `max` and `min`. Complex numbers
have no canonical order, so we pick the *real-part* order: compare `re` and ignore `im`.

These instances apply to `TorchLean.Complex α` throughout the program and leave mathlib’s `ℂ`
unchanged. They do not form a linear order: `1 + i` and `1 - i` are distinct, and each is `≤` the
other.
-/
/-- `x < y` compares the real parts. -/
instance [LT α] : LT (Complex α) := ⟨fun x y => x.re < y.re⟩
/-- `x ≤ y` compares the real parts. -/
instance [LE α] : LE (Complex α) := ⟨fun x y => x.re ≤ y.re⟩

/-- Print `z` as `(re + imi)` from the strings of its parts. -/
instance [ToString α] : ToString (Complex α) where
  toString z := s!"({toString z.re} + {toString z.im}i)"

/-! ## Numeric literals and constants -/

/-- A natural `n` casts to `n + 0i`. -/
instance [Context α] : NatCast (Complex α) where
  natCast n := ofReal (n : α)

/-! ## Transcendentals -/

/-- The magnitude `√(re² + im²)`, computed as `scale * √(a² + b²)`, where `a` and `b` are the parts
divided by `scale = max |re| |im|`, so squaring a large part does not overflow. In floating point
the result can differ in the last bits from the unscaled expression. In an IEEE format an infinite
part gives a NaN (from `∞ / ∞`), where C99's `cabs` gives `+∞`.
-/
def absReal [Context α] (z : Complex α) : α :=
  let scale := Max.max (MathFunctions.abs z.re) (MathFunctions.abs z.im)
  if scale == 0 then 0 else
    let a := z.re / scale
    let b := z.im / scale
    scale * MathFunctions.sqrt (a * a + b * b)

/-- Complex elementary functions from those of `α`, through their component formulas. -/
instance instMathFunctions [Context α] [InverseTrig α] : MathFunctions (Complex α) where
  exp z :=
    -- exp(a+bi) = exp(a) * (cos(b) + i sin(b))
    let ea := MathFunctions.exp z.re
    ⟨ea * MathFunctions.cos z.im, ea * MathFunctions.sin z.im⟩
  -- `tanh (x + iy) = (sinh x cosh x + i sin y cos y) / (sinh² x + cos² y)`. Dividing through by
  -- `c²`, with `th = tanh x`, `c = cosh x`, `q = cos y / c` and `r = sin y / c`, gives
  -- `(th + i r q) / (th² + q²)`, which forms no `c²` and whose denominator cannot cancel. A zero
  -- `y` gives `tanh x` and keeps that zero, and an infinite `x` gives `±1` and a zero of the sign
  -- of `r q`: the values of C99's `ctanh`.
  tanh z :=
    let th := MathFunctions.tanh z.re
    if Context.isZero z.im then ⟨th, z.im⟩ else
      let c := MathFunctions.cosh z.re
      let q := MathFunctions.cos z.im / c
      let r := MathFunctions.sin z.im / c
      if Context.isInf z.re then ⟨th, Context.copysign 0 (r * q)⟩ else
        let d := th * th + q * q
        ⟨th / d, r * q / d⟩
  cosh z :=
    -- cosh(a+bi) = cosh(a)cos(b) + i sinh(a)sin(b)
    ⟨MathFunctions.cosh z.re * MathFunctions.cos z.im,
      MathFunctions.sinh z.re * MathFunctions.sin z.im⟩
  -- C99's `csqrt` (Annex G.6.4.2): `∞ ± ∞i` at an infinite imaginary part, `NaN + NaN i` at a NaN
  -- real part, `∞ ± 0i` at `∞ ± yi` and `0 ± ∞i` at `-∞ ± yi`, with a NaN for the zero when `y` is
  -- a NaN, and `NaN + NaN i` from the scaled formula at a finite real part and a NaN imaginary
  -- part. Each `±` is the sign that `Context.copysign` reads from the imaginary part, also from a
  -- NaN at `-∞ + NaN i`, where C99 leaves it open. A NaN result is quiet (`Context.quietNaN`), and
  -- an infinite part is the root `√|x|` of an infinite part `x`, whose dual tangent is zero when
  -- the tangent of `x` is finite.
  sqrt z :=
    if Context.isInf z.im then
      let root := MathFunctions.sqrt (MathFunctions.abs z.im)
      ⟨root, Context.copysign root z.im⟩
    else if Context.isNaN z.re then ⟨Context.quietNaN z.re, Context.quietNaN z.re⟩
    else if Context.isInf z.re then
      let root := MathFunctions.sqrt (MathFunctions.abs z.re)
      if z.re > 0 then
        ⟨root, if Context.isNaN z.im then Context.quietNaN z.im else Context.copysign 0 z.im⟩
      else
        ⟨if Context.isNaN z.im then Context.quietNaN z.im else 0, Context.copysign root z.im⟩
    else if z.im == 0 then
      if z.re > 0 then
        let root := MathFunctions.sqrt z.re
        ⟨root, z.im / (2 * root)⟩
      else if z.re < 0 then
        -- On the cut the imaginary part takes the sign of the zero `z.im`, so that
        -- `sqrt (conj z) = conj (sqrt z)` as for C99's `csqrt`. The real part `z.im / (2 * im)` is
        -- `+0`, and over dual numbers both tangents are those of `dz / (2 √z)` on that side.
        let im := Context.copysign (MathFunctions.sqrt (MathFunctions.abs z.re)) z.im
        ⟨z.im / (2 * im), im⟩
      else
        ⟨0, Context.copysign (MathFunctions.sqrt (MathFunctions.abs z.re)) z.im⟩
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
  abs z := ofReal (absReal z)
  log z :=
    let scale := Max.max (MathFunctions.abs z.re) (MathFunctions.abs z.im)
    let magnitudeLog := if scale == 0 then MathFunctions.log scale else
      let a := z.re / scale
      let b := z.im / scale
      MathFunctions.log scale + MathFunctions.log (a * a + b * b) / (2 : α)
    ⟨magnitudeLog, InverseTrig.atan2 z.im z.re⟩
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

/-- `max x y` is `x` when `x` has the greater real part, and `y` otherwise. -/
instance [Context α] : Max (Complex α) where
  max x y := if x.re > y.re then x else y

/-- `min x y` is `y` when `x` has the greater real part, and `x` otherwise. -/
instance [Context α] : Min (Complex α) where
  min x y := if x.re > y.re then y else x

/-- `x ^ y` is `exp (y * log x)`, except that a zero `x` gives `0 ^ 0 = 1` and `0 ^ y = 0` for
`y.re > 0`. -/
instance [Context α] [InverseTrig α] : Pow (Complex α) (Complex α) where
  pow x y :=
    if Context.isZero x.re && Context.isZero x.im then
      if Context.isZero y.re && Context.isZero y.im then ofReal 1
      else if y.re > 0 then ofReal 0
      else MathFunctions.exp (y * MathFunctions.log x)
    else MathFunctions.exp (y * MathFunctions.log x)

/-- Lift a scalar `Context` to TorchLean complex scalars. -/
instance [Context α] [InverseTrig α] : Context (Complex α) where
  sqrtNat n := ofReal (Context.sqrtNat (α := α) n)
  isZero z := Context.isZero z.re && Context.isZero z.im
  defaultEpsilon := ofReal (Context.defaultEpsilon (α := α))
  decidableGT := fun x y => (Context.decidableGT) x.re y.re
  stopGradient? := (Context.stopGradient? (α := α)).map fun clear z =>
    ⟨clear z.re, clear z.im⟩
  ratCast value := ofReal (value : α)
  -- Each part divided once by the count. The quotient by `n + 0i` would divide by `n * n` instead:
  -- binary16 `(1 + 0i) / (256 + 0i)` is zero because `256 * 256` overflows.
  divNat z n := ⟨Context.divNat z.re n, Context.divNat z.im n⟩
  quietNaN z := ⟨Context.quietNaN z.re, Context.quietNaN z.im⟩
  -- `tan z = -i tanh (i z)`, so `tan` keeps the scaling of `tanh`.
  tan z :=
    let t := MathFunctions.tanh (⟨-z.im, z.re⟩ : Complex α)
    ⟨t.im, -t.re⟩
  -- Kahan's `asinh`: with `s1 = √(1 - iz)` and `s2 = √(1 + iz)`, the real part is
  -- `asinh (Im (conj s1 * s2))`, a sum of two products of the sign of `x`, and the imaginary part
  -- is `atan2 y (Re (s1 * s2))`, whose second argument is a sum of two nonnegative products. The
  -- sign of a zero `x` selects the side of the cut. A zero `y` gives `asinh x` and keeps that
  -- zero, and an infinite part gives an infinite real part of the sign of `x` and the imaginary
  -- part `atan2 y |x|`: the values of C99's `casinh`. That real part is the real `asinh` of an
  -- infinite part with the sign of `x`, so that its dual tangent is zero when the tangent of that
  -- part is finite.
  asinh z :=
    if Context.isZero z.im then ⟨Context.asinh z.re, z.im⟩
    else if Context.isInf z.re || Context.isInf z.im then
      ⟨Context.asinh (if Context.isInf z.re then z.re else Context.copysign z.im z.re),
        InverseTrig.atan2 z.im (MathFunctions.abs z.re)⟩
    else
      let s1 := MathFunctions.sqrt (⟨1 + z.im, -z.re⟩ : Complex α)
      let s2 := MathFunctions.sqrt (⟨1 - z.im, z.re⟩ : Complex α)
      -- `|s|` is about `|z|` and `c` about `|x|`, so either can overflow at a finite `z`. The sum
      -- `w` with one factor of each product halved is about `s / 2` and finite, and the real part
      -- `asinh w ± log 2`, with the sign of `w`, is within `3 / (16 w²)` of `asinh (2w)`. An
      -- infinite `c` is halved together with `y`, which keeps the angle.
      let s := s1.re * s2.im - s1.im * s2.re
      let c := s1.re * s2.re - s1.im * s2.im
      ⟨if Context.isInf s then
          let w := (s1.re / 2) * s2.im - (s1.im / 2) * s2.re
          Context.asinh w + Context.copysign (MathFunctions.log 2) w
        else Context.asinh s,
        if Context.isInf c then
          InverseTrig.atan2 (z.im / 2) ((s1.re / 2) * s2.re - (s1.im / 2) * s2.im)
        else InverseTrig.atan2 z.im c⟩
  -- Kahan's `acosh`: with `u = √(z - 1)` and `v = √(z + 1)`, the real part is
  -- `asinh (Re (conj u * v))`, a sum of two nonnegative products, and the imaginary part is
  -- `2 atan2 (Im u) (Re v)`. Shifting only the real part keeps the sign of a zero imaginary part,
  -- which selects the side of the cut. An infinite part gives C99's `cacosh`, `+∞ + i atan2 y x`,
  -- whose real part is the real `acosh` of the magnitude of that part, so that its dual tangent is
  -- zero when the tangent of that part is finite.
  acosh z :=
    if Context.isInf z.re || Context.isInf z.im then
      ⟨Context.acosh (MathFunctions.abs (if Context.isInf z.re then z.re else z.im)),
        InverseTrig.atan2 z.im z.re⟩
    else
      let u := MathFunctions.sqrt (⟨z.re - 1, z.im⟩ : Complex α)
      let v := MathFunctions.sqrt (⟨z.re + 1, z.im⟩ : Complex α)
      let angle := InverseTrig.atan2 u.im v.re
      -- `s` is about `|z|`, so it can overflow at a finite `z`. The sum `w` with one factor of each
      -- product halved is about `s / 2` and finite, and the real part is `asinh w + log 2`, as in
      -- `asinh`.
      let s := u.re * v.re + u.im * v.im
      ⟨if Context.isInf s then
          Context.asinh ((u.re / 2) * v.re + (u.im / 2) * v.im) + MathFunctions.log 2
        else Context.asinh s, angle + angle⟩

end Complex

end TorchLean
