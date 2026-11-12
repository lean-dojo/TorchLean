/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import FloatLib.Floats.Formats.BinaryInterchange.Algebraic.Rounding.Runtime
public import FloatLib.Floats.Formats.BinaryInterchange.Dyadic.Decode
public import FloatLib.Numerics.Exact.Elementary.Runtime

/-!
# Signed rounding from a comparator

`Elementary.roundSigned fmt c` rounds a real number `y` once, given a comparator `c` of `y` with
rationals: `c q = cmp y q` for every rational `q`. The sign bit comes from `c 0`, and
`Model.AlgebraicRounding.round` rounds the magnitude through `c` or through its reflection
`b ↦ (c (-b)).swap`, which compares `-y`. No floating-point approximation enters: the rounding
reads `c` and nothing else. A zero target gives `+0`.

`Elementary.once c fmt x` lifts that rounding to `n` inputs: when every input is finite, `c` maps
the rationals they decode to (`rats`) to a comparator, and otherwise the result is the invalid
value.

Comparators come from comparators of simpler reals. Given a comparator `E` of a real `e`,
`linCompare E c c₀` compares `c * e + c₀` with zero, and `ratioCompare E α β γ δ` compares the
ratio `(α * e + β) / (γ * e + δ)` with rationals, for rational coefficients. With `e = exp r`,
`expRatioCompare α β γ δ r` compares that ratio with rationals, so `roundSigned` of it rounds the
ratio once. It covers the logistic function, the exponential and `exp r - 1 - r`.

`TorchLean.Verify.RuntimeApprox.RoundSigned` proves that in an IEEE format `roundSigned` is the
nearest-even rounding of `y` when `|y|` is below the overflow threshold and the infinity of the
sign of `y` otherwise, and that the comparators compare what they name for every real `e`.
-/

@[expose] public section

namespace TorchLean

open FloatLib.Floats FloatLib.Floats.Formats.BinaryInterchange
open FloatLib.Floats.Formats.BinaryInterchange.Model

namespace Elementary

/-- The comparison rounding of the real number `y` that `c` compares with rationals: the sign bit
is set when `c 0 = .lt`, and the magnitude is rounded through `c` or its reflection. In an IEEE
format it is the nearest-even rounding of `y` when `|y|` is below the overflow threshold and the
infinity of the sign of `y` otherwise. -/
def roundSigned (fmt : FloatFormat) (c : ℚ → Ordering) : Model fmt :=
  if c 0 = .lt then AlgebraicRounding.round fmt true fun b => (c (-b)).swap
  else AlgebraicRounding.round fmt false c

/-- The comparison of `c * e + c₀` with zero, where `E` compares `e` with rationals. -/
def linCompare (E : ℚ → Ordering) (c c₀ : ℚ) : Ordering :=
  if c = 0 then cmp c₀ 0
  else if 0 < c then E (-c₀ / c)
  else (E (-c₀ / c)).swap

/-- The comparison of `(α * e + β) / (γ * e + δ)` with `q`, where `E` compares `e` with
rationals. A zero denominator gives the ratio zero. -/
def ratioCompare (E : ℚ → Ordering) (α β γ δ q : ℚ) : Ordering :=
  match linCompare E γ δ with
  | .gt => linCompare E (α - q * γ) (β - q * δ)
  | .lt => (linCompare E (α - q * γ) (β - q * δ)).swap
  | .eq => cmp 0 q

/-- The rationals that the inputs decode to, zero at an input that is not finite. -/
def rats {fmt : FloatFormat} {n : ℕ} (x : Fin n → Model fmt) (i : Fin n) : ℚ :=
  (toRat? (x i)).getD 0

/-- `roundSigned` of the comparator `c` at the decoded inputs when every input is finite, and the
invalid result otherwise. -/
def once {n : ℕ} (c : (Fin n → ℚ) → ℚ → Ordering) (fmt : FloatFormat) (x : Fin n → Model fmt) :
    Model fmt :=
  if ∀ i, isFinite (x i) = true then roundSigned fmt (c (rats x)) else invalidResult fmt

/-- The comparison of `(α * exp r + β) / (γ * exp r + δ)` with rationals, which `ratioCompare`
builds from the comparator of `exp r`. -/
def expRatioCompare (α β γ δ r : ℚ) : ℚ → Ordering :=
  ratioCompare (FloatLib.Numerics.ElementaryComparison.prepareExp r 16).compare α β γ δ

end Elementary

end TorchLean
