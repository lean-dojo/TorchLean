/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Algebra.Order.Group.Unbundled.Abs
public import Mathlib.Analysis.Complex.Exponential
public import Mathlib.Analysis.Complex.Trigonometric
public import Mathlib.Analysis.SpecialFunctions.Log.Basic
public import Mathlib.Analysis.SpecialFunctions.Pow.Real
public import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
public import Mathlib.Data.Real.Basic
public import Mathlib.Data.Real.Sqrt
public import Mathlib.Logic.Basic

/-!
# `Context α`: scalar interface for models + proofs

TorchLean is designed to be *scalar-polymorphic*: the same model/layer definitions can be
instantiated over many numeric backends:

- `Float` (fast execution; trusted runtime semantics),
- `TorchLean.Floats.IEEE754.IEEE32Exec` (executable bit-level IEEE-754 binary32),
- interval enclosures for verification (see `NN/Floats/Interval/*`),
- `ℝ` (proof-level mathematics).

Why we designed it this way:

- We did not want separate "Float model code", "proof model code", and "verification model code"
  that slowly diverge and become inconsistent.
- In practice, we iterate across phases: execute a small model, state the proof-level contract, then
  run verification bounds. Rewriting each model for each phase is error-prone.
- A single scalar-polymorphic spec gives one source of truth for layers/models, while letting us
  swap numeric meaning by changing the scalar instance.
- This keeps cross-checking honest: if behavior changes between backends, the difference is visible
  at the scalar semantics layer, not hidden inside duplicated model definitions.
- The tradeoff is a slightly larger scalar interface (`Context α`), but we accept that complexity
  to keep architecture-level duplication low and proofs/reuse high.

Some relevant literature we were following / taking inspiration from:

- Bezanson et al., "Julia: A Fresh Approach to Numerical Computing" (generic numeric code across
  many scalar types; performance via specialization): https://arxiv.org/abs/1411.1607
- Spitters and van der Weegen, "Type classes for mathematics in type theory" (typeclass-based
  algebraic interfaces for reusable formalization and instances):
  https://doi.org/10.1017/S0960129511000119
- Elliott, "The Simple Essence of Automatic Differentiation" (one abstract formulation specialized
  to multiple concrete semantics/representations): https://arxiv.org/abs/1804.00746
- Mirman et al., "The Fundamental Limits of Interval Arithmetic for Neural Networks" (why interval
  backends are useful and where they become conservative): https://arxiv.org/abs/2112.05235

Our `Context α` is the same engineering pattern in a Lean setting: one model/layer definition, many
scalar interpretations, and explicit tradeoffs about semantics.

To make this practical, we collect the numeric operations required by neural networks into a single
typeclass:

`Context α`

This is intentionally broader than a standard algebraic structure: it bundles arithmetic,
ordering, and common transcendental functions (exp/tanh/log/sqrt) used by activations and losses.

## Notes

- Many spec definitions assume `[Context α]` so they can be re‑used at multiple dtypes.
- For "paper theorems", the spec layer fixes `Spec.SpecScalar := ℝ` (see
  `NN/Spec/Core/Scalar.lean`).
- `Context.decidable_gt` is included so executable code can decide comparisons (e.g. ReLU / argmax).
- For executable examples, `Context.gtBool` converts `x > y` into a printable `Bool`.
- For interval arithmetic, we override some order/comparison behavior (see `namespace Interval`
  below).
-/

@[expose] public section

-- Define generic type and instances of Floats (computation) and Reals (proofs)

/-- Scalar transcendental functions used in activations and losses. -/
class MathFunctions (α : Type) where
  exp : α → α
  tanh : α → α
  cosh : α → α
  sqrt : α → α
  abs : α → α
  log : α → α
  pi : α
  cos : α → α
  sin : α → α
  sinh : α → α

/-- Common scalar constants used in model definitions. -/
class Numbers (α : Type) where
  neg_point_five : α
  neg_one : α
  pointone : α
  pointfive : α
  one : α
  zero : α
  two : α
  three : α
  four : α
  five : α
  ten : α
  log10 : α
  log10000 : α
  epsilon : α
  neg_thousand : α

/-!
Friendly aliases: these keep the original names (`MathFunctions`, `Numbers`) and give more
descriptive names for scalar-facing code.
-/
/-- Alias for `MathFunctions`. -/
abbrev ScalarMath := MathFunctions
/-- Alias for `Numbers`. -/
abbrev ScalarConstants := Numbers

/-- The full scalar interface required by spec‑level tensors and models. -/
class Context (α : Type) extends
  Inhabited α, One α, Zero α,
  Add α, Sub α, Mul α, Div α, Neg α, Pow α α, Max α, Min α,
  BEq α, LT α, LE α, -- For ordering
  MathFunctions α, Numbers α,
  Coe Nat α where -- For converting natural numbers to the type
  decidable_gt : DecidableRel (· > · : α → α → Prop) -- For ordering

namespace Context

/-- Decide `x > y` as a `Bool` using the `Context`'s `decidable_gt`. -/
def gtBool {α : Type} [Context α] (x y : α) : Bool :=
  let _ : Decidable (x > y) := (Context.decidable_gt) x y
  decide (x > y)

end Context

/-- A `Context` includes a decidable `>` relation; expose it as a standard typeclass. -/
instance {α : Type} [Context α] : DecidableRel ((· > ·) : α → α → Prop) :=
  Context.decidable_gt

-- Open for convenience
open MathFunctions

/-- `MathFunctions` instance for Lean's `Float` (runtime-oriented backend). -/
instance : MathFunctions Float where
  exp := Float.exp
  tanh := Float.tanh
  cosh := Float.cosh
  sqrt := Float.sqrt
  abs := Float.abs
  log := Float.log
  pi := 3.14159265358979323846
  cos := Float.cos
  sin := Float.sin
  sinh := Float.sinh

/-- `MathFunctions` instance for `ℝ` (proof backend, noncomputable). -/
noncomputable instance : MathFunctions ℝ where
  exp := Real.exp
  tanh := Real.tanh
  cosh := Real.cosh
  sinh := Real.sinh
  sqrt := Real.sqrt
  abs := fun x => |x|
  log := Real.log
  pi := Real.pi
  cos := Real.cos
  sin := Real.sin

/-- `Numbers` literals for `Float` (runtime-oriented backend). -/
instance : Numbers Float where
  neg_point_five := -0.5
  neg_one := -1
  pointone := 0.1
  pointfive := 0.5
  zero      := 0
  one       := 1
  two       := 2
  three     := 3
  four      := 4
  five      := 5
  ten       := 10
  log10     := Float.log 10
  log10000  := Float.log 10000
  epsilon   := 1e-6
  neg_thousand := -1000

/-- `Numbers` literals for `ℝ` (proof backend, noncomputable). -/
noncomputable instance : Numbers ℝ where
  neg_point_five := -0.5
  neg_one := -1
  pointone := 0.1
  pointfive := 0.5
  zero      := 0
  one       := 1
  two       := 2
  three     := 3
  four      := 4
  five      := 5
  ten       := 10
  log10     := Real.log 10
  log10000  := Real.log 10000
  epsilon   := 1e-6
  neg_thousand := -1000

/-- Coerce naturals into `Float` using `Float.ofNat`. -/
instance : Coe Nat Float where
  coe := Float.ofNat

/-- Coerce naturals into `ℝ` via the standard `Nat`-to-real coercion. -/
instance : Coe Nat ℝ where
  coe := fun n => (n : ℕ) -- this uses Real.ofNat

/-- Coerce naturals into `ℚ` via the standard `Nat`-to-rational coercion. -/
instance : Coe Nat ℚ where
  coe := fun n => (n : Nat)

/-!
## A lightweight `ℚ` backend

`Context` includes transcendental functions and real-valued exponentiation (`Pow α α`) because many
models (softmax, tanh, etc.) need them when instantiated over `Float` / `ℝ` / interval scalars.

For `ℚ`, most transcendental functions do not map rationals to rationals, so there is no canonical
exact interpretation. TorchLean therefore does **not** install the rational `Context` globally.
Purely algebraic tests can opt in explicitly with:

```lean
open scoped NN.Spec.RationalAlgebraic
```

Current policy:
- `abs` is exact.
- `pow x y` is supported only when `y` is an integer rational (`y.den = 1`); otherwise it returns
  `0`.
- Other transcendental functions are defined as `0` only in this explicitly scoped algebraic
  backend. This makes accidental softmax/GELU/tanh-over-`ℚ` use a typeclass error by default.
-/

namespace NN.Spec.RationalAlgebraic

/--
`Pow ℚ ℚ` instance used for the lightweight rational backend.

Policy: support `x^y` only when `y` is an integer rational (`y.den = 1`); otherwise return `0`.
The instance is scoped so it is unavailable unless the caller explicitly opens
`NN.Spec.RationalAlgebraic`.
-/
scoped instance instPowRatRat : Pow ℚ ℚ where
  pow x y :=
    if y.den = 1 then
      x ^ y.num
    else
      0

/--
`MathFunctions ℚ` dictionary for the lightweight rational backend.

Only `abs` is meaningful; other transcendental functions are defined as `0` in this scoped backend
and should not be used for semantic claims. Keeping this scoped makes unsupported transcendental rational models fail at
elaboration unless a file deliberately opts into the algebraic-test backend.
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

/--
`Numbers ℚ` dictionary for the lightweight rational backend.

The transcendentals (`log10`, etc.) are defined as `0` in this scoped backend; the basic rational
literals are exact.
-/
scoped instance instNumbersRat : Numbers ℚ where
  neg_point_five := -1/2
  neg_one := -1
  pointone := 1/10
  pointfive := 1/2
  zero := 0
  one := 1
  two := 2
  three := 3
  four := 4
  five := 5
  ten := 10
  log10 := 0
  log10000 := 0
  epsilon := 1/1000000
  neg_thousand := -1000

/-- Full opt-in `Context` dictionary for exact rational algebraic fragments. -/
scoped instance instContextRat : Context ℚ where
  decidable_gt := inferInstance

end NN.Spec.RationalAlgebraic

/-- Full `Context` instance for `Float` (runtime backend). -/
instance : Context Float := { decidable_gt := inferInstance }
/-- Full `Context` instance for `ℝ` (proof backend, noncomputable). -/
noncomputable instance : Context ℝ := { decidable_gt := Classical.decRel _ }
