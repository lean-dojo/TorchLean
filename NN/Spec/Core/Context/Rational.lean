/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Context
import Mathlib.Data.Rat.Cast.Order
import Mathlib.Tactic.NormNum.Abs
import Mathlib.Tactic.NormNum.DivMod
import Mathlib.Tactic.NormNum.OfScientific
import Mathlib.Tactic.NormNum.Pow

/-!
# The opt-in rational algebraic backend

Split out of `NN.Spec.Core.Context` so that only the handful of algebraic tests that `open scoped
Spec.RationalAlgebraic` pay for the rational instances.
-/

@[expose] public section

/-!
## Rational Backend

`Context` includes transcendental functions and real-valued exponentiation (`Pow α α`) because many
models (softmax, tanh, etc.) need them when instantiated over `Float` / `ℝ` / interval scalars.

For `ℚ`, most transcendental functions do not map rationals to rationals, so there is no canonical
exact interpretation. TorchLean therefore does **not** install the rational `Context` globally.
Purely algebraic tests can opt in explicitly with:

```lean
open scoped Spec.RationalAlgebraic
```

Current policy:
- `abs` is exact.
- `pow x y` is supported only when `y` is an integer rational (`y.den = 1`); otherwise it returns
  `0`.
- Other transcendental functions are defined as `0` only in this explicitly scoped algebraic
  backend. This makes accidental softmax/GELU/tanh-over-`ℚ` use a typeclass error by default.
-/

namespace Spec.RationalAlgebraic

/--
`Pow ℚ ℚ` instance used for the rational backend.

Policy: support `x^y` only when `y` is an integer rational (`y.den = 1`); otherwise return `0`.
The instance is scoped so it is unavailable unless the caller explicitly opens
`Spec.RationalAlgebraic`.
-/
scoped instance instPowRatRat : Pow ℚ ℚ where
  pow x y :=
    if y.den = 1 then
      x ^ y.num
    else
      0

/--
`MathFunctions ℚ` dictionary for the rational backend.

Only `abs` is meaningful; other transcendental functions are defined as `0` in this scoped backend
and should not be used for semantic claims. Keeping this scoped makes unsupported transcendental
rational models fail at elaboration unless a file deliberately opts into the algebraic-test
backend.
-/
scoped instance instMathFunctionsRat : MathFunctions ℚ where
  exp := fun _ => 0
  tanh := fun _ => 0
  cosh := fun _ => 0
  sqrt := fun _ => 0
  abs := fun x => if x < 0 then -x else x
  log := fun _ => 0
  pi := 0
  cos := fun _ => 0
  sin := fun _ => 0
  sinh := fun _ => 0

/-- Full opt-in `Context` dictionary for exact rational algebraic fragments. -/
scoped instance instContextRat : Context ℚ where
  defaultEpsilon := 1 / 1000000
  decidableGT := inferInstance

end Spec.RationalAlgebraic
