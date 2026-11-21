/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Elementwise
public import TorchLean.Tensor.Rep.Segment
public import Mathlib.Analysis.SpecialFunctions.Complex.Arctan
public import Mathlib.Analysis.SpecialFunctions.Pow.Complex
public import Mathlib.Analysis.Calculus.Deriv.Basic

/-!
# The complex functions

`ComplexFn` names the eighteen complex functions of one argument that `jax.lax` applies to complex
arrays: `exp`, `expm1`, `log`, `log1p`, `sqrt`, `rsqrt`, the trigonometric and hyperbolic functions
and their inverses. `ComplexFn.value` defines each with Mathlib's complex functions, the principal
logarithm `Complex.log` and the principal square root `w ^ (2⁻¹ : ℂ)`. Mathlib has no complex
inverse sine, cosine or hyperbolic functions, so `asin`, `asinh`, `acosh` and `atanh` are the
principal logarithmic expressions FloatLib states its exact results with (`asinExact_value`,
`asinhExact_value`, `acoshExact_value`, `atanhExact_value`), and `acos` is `π / 2 - asin`.

`ComplexFunctions α` is the optional scalar capability of `Op.complex` and `Op.complexPow`: each
function and its complex derivative, and the principal power `z ^ w` and its partial derivatives in
`z` and `w`, on pairs of scalars, the real part first. `ℝ` has an instance here: the value and
Mathlib's `deriv` of each function, Mathlib's `z ^ w` (`Complex.cpow`) and the partial derivatives
`w z ^ (w - 1)` and `z ^ w log z`, which are those of `z ^ w` where `z` is off the closed negative
real axis (`Complex.hasStrictFDerivAt_cpow`). The configured binary formats have FloatLib's
`ExecComplex` (`TorchLean.Tensor.ConfiguredBinary.Complex`).

A complex tensor of shape `s` is a real tensor of shape `s ++ [2]`, the real part at index `0` of
the last axis and the imaginary part at `1`, the layout of the JAX importer. `Tensor.pairAt` reads
the pair at a coordinate of `s` and `Tensor.ofPairs` builds a tensor from its pairs.

## References

- JAX documentation, `jax.lax`: `https://docs.jax.dev/en/latest/jax.lax.html`
-/

@[expose] public section

noncomputable section

namespace TorchLean

/-- The complex functions of one argument, each named after its `jax.lax` primitive. -/
inductive ComplexFn where
  /-- `lax.exp`. -/
  | exp
  /-- `lax.expm1`. -/
  | expm1
  /-- `lax.log`. -/
  | log
  /-- `lax.log1p`. -/
  | log1p
  /-- `lax.sqrt`. -/
  | sqrt
  /-- `lax.rsqrt`. -/
  | rsqrt
  /-- `lax.sin`. -/
  | sin
  /-- `lax.cos`. -/
  | cos
  /-- `lax.tan`. -/
  | tan
  /-- `lax.sinh`. -/
  | sinh
  /-- `lax.cosh`. -/
  | cosh
  /-- `lax.tanh`. -/
  | tanh
  /-- `lax.asin`. -/
  | asin
  /-- `lax.acos`. -/
  | acos
  /-- `lax.atan`. -/
  | atan
  /-- `lax.asinh`. -/
  | asinh
  /-- `lax.acosh`. -/
  | acosh
  /-- `lax.atanh`. -/
  | atanh
  deriving DecidableEq, Repr

namespace ComplexFn

open Complex in
/-- Each function as a function of a complex number. `expm1` is `exp w - 1`, `log1p` is
`log (1 + w)`, `sqrt` is `w ^ (1 / 2)`, `rsqrt` its inverse, `asin` is
`-i log (i w + (1 - w²) ^ (1 / 2))`, `acos` is `π / 2 - asin w`, `asinh` is
`log (w + (1 + w²) ^ (1 / 2))`, `acosh` is
`2 log (((w + 1) / 2) ^ (1 / 2) + ((w - 1) / 2) ^ (1 / 2))` and `atanh` is
`log ((1 + w) / (1 - w)) / 2`, in the principal logarithm and power. -/
def value : ComplexFn → ℂ → ℂ
  | exp => Complex.exp
  | expm1 => fun w => Complex.exp w - 1
  | log => Complex.log
  | log1p => fun w => Complex.log (1 + w)
  | sqrt => fun w => w ^ (2⁻¹ : ℂ)
  | rsqrt => fun w => (w ^ (2⁻¹ : ℂ))⁻¹
  | sin => Complex.sin
  | cos => Complex.cos
  | tan => Complex.tan
  | sinh => Complex.sinh
  | cosh => Complex.cosh
  | tanh => Complex.tanh
  | asin => fun w => -I * Complex.log (I * w + (1 - w ^ 2) ^ (2⁻¹ : ℂ))
  | acos => fun w => (Real.pi : ℂ) / 2 + I * Complex.log (I * w + (1 - w ^ 2) ^ (2⁻¹ : ℂ))
  | atan => Complex.arctan
  | asinh => fun w => Complex.log (w + (1 + w ^ 2) ^ (2⁻¹ : ℂ))
  | acosh => fun w =>
      2 * Complex.log (((w + 1) / 2) ^ (2⁻¹ : ℂ) + ((w - 1) / 2) ^ (2⁻¹ : ℂ))
  | atanh => fun w => Complex.log ((1 + w) / (1 - w)) / 2

/-- The `jax.lax` name of each function. -/
def jaxName : ComplexFn → String
  | exp => "exp"
  | expm1 => "expm1"
  | log => "log"
  | log1p => "log1p"
  | sqrt => "sqrt"
  | rsqrt => "rsqrt"
  | sin => "sin"
  | cos => "cos"
  | tan => "tan"
  | sinh => "sinh"
  | cosh => "cosh"
  | tanh => "tanh"
  | asin => "asin"
  | acos => "acos"
  | atan => "atan"
  | asinh => "asinh"
  | acosh => "acosh"
  | atanh => "atanh"

/-- The complex functions. -/
def all : List ComplexFn :=
  [exp, expm1, log, log1p, sqrt, rsqrt, sin, cos, tan, sinh, cosh, tanh, asin, acos, atan, asinh,
    acosh, atanh]

/-- The function with a given name, the inverse of `jaxName`. -/
def ofJaxName? (name : String) : Option ComplexFn :=
  all.find? (·.jaxName == name)

/-- Reading back the name of a function gives the function. -/
@[simp] theorem ofJaxName?_jaxName (f : ComplexFn) : ofJaxName? f.jaxName = some f := by
  cases f <;> rfl

end ComplexFn

/-- The complex functions and the principal power, and their complex derivatives, on pairs of
scalars, the real part first, for scalar types that have them. -/
class ComplexFunctions (α : Type) where
  /-- The function `f` at the complex number of parts `z.1` and `z.2`. -/
  apply : ComplexFn → α × α → α × α
  /-- The complex derivative of `f` at the complex number of parts `z.1` and `z.2`. -/
  deriv : ComplexFn → α × α → α × α
  /-- The principal power `z ^ w` (`lax.pow`). -/
  pow : α × α → α × α → α × α
  /-- The partial complex derivatives of `z ^ w` in `z` and in `w`. -/
  powDeriv : α × α → α × α → (α × α) × (α × α)

/-- The reals take `ComplexFn.value` and its complex derivative, Mathlib's `deriv` (`0` where the
function is not complex differentiable), and Mathlib's `z ^ w` with the partial derivatives
`w z ^ (w - 1)` and `z ^ w log z`. -/
instance : ComplexFunctions ℝ where
  apply f z := ((f.value ⟨z.1, z.2⟩).re, (f.value ⟨z.1, z.2⟩).im)
  deriv f z := ((_root_.deriv f.value ⟨z.1, z.2⟩).re, (_root_.deriv f.value ⟨z.1, z.2⟩).im)
  pow z w := let p : ℂ := (⟨z.1, z.2⟩ : ℂ) ^ (⟨w.1, w.2⟩ : ℂ); (p.re, p.im)
  powDeriv z w :=
    let a : ℂ := ⟨z.1, z.2⟩
    let b : ℂ := ⟨w.1, w.2⟩
    (((b * a ^ (b - 1)).re, (b * a ^ (b - 1)).im),
      ((a ^ b * Complex.log a).re, (a ^ b * Complex.log a).im))

namespace Tensor

variable {α : Type} [Storage α] {s : Shape}

/-- The coordinate of part `k` of the pair at `c` in the layout `s ++ [2]`: the real part at `0`
and the imaginary part at `1`. -/
def pairCoord (c : s.Coord) (k : Fin 2) : (s ++ [2]).Coord :=
  (Shape.Coord.appendEquiv s [2]).symm (c, k, ())

/-- The pair at `c` of a tensor in the layout `s ++ [2]`. -/
def pairAt (x : Tensor α (s ++ [2])) (c : s.Coord) : α × α :=
  (x (pairCoord c 0), x (pairCoord c 1))

/-- The tensor in the layout `s ++ [2]` whose pair at each `c` is `g c`. -/
def ofPairs (g : s.Coord → α × α) : Tensor α (s ++ [2]) :=
  Rep.ofFn fun e =>
    let parts := Shape.Coord.appendEquiv s [2] e
    if parts.2.1 = 0 then (g parts.1).1 else (g parts.1).2

/-- Part `k` of the pair at `c` of `ofPairs g` is part `k` of `g c`. -/
@[simp] theorem ofPairs_pairCoord (g : s.Coord → α × α) (c : s.Coord) (k : Fin 2) :
    ofPairs g (pairCoord c k) = if k = 0 then (g c).1 else (g c).2 := by
  simp [ofPairs, pairCoord]

/-- The pairs of `ofPairs g` are those of `g`. -/
@[simp] theorem pairAt_ofPairs (g : s.Coord → α × α) (c : s.Coord) :
    pairAt (ofPairs g) c = g c := by
  simp [pairAt]

end Tensor

end TorchLean
