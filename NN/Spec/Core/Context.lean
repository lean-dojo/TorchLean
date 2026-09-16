/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Core.Numeric
public import NN.Core.Numeric.Quotient
public import Mathlib.Algebra.Field.Defs
public import Mathlib.Algebra.Order.Ring.Defs
public import Mathlib.Algebra.Order.Group.Unbundled.Abs
public import Mathlib.Data.Rat.Cast.Defs

/-!
# `Context α`: scalar interface for models + proofs

TorchLean is designed to be *scalar-polymorphic*: the same model/layer definitions can be
instantiated over many numeric backends:

- `Float` (fast binary64 execution with a logical model and a separate native boundary),
- FloatLib configured binary values, with the exponent and fraction widths selected in their type
  (import `NN.Spec.Core.FloatInstances` for their adapters),
- interval enclosures for verification,
- `ℝ` (proof-level mathematics).

The scalar parameter serves several purposes:

- We did not want separate "Float model code", "proof model code", and "verification model code"
  that slowly diverge and become inconsistent.
- In practice, we iterate across phases: execute a compact model, state the proof-level contract,
  then run verification bounds. Rewriting each model for each phase is error-prone.
- A scalar-polymorphic specification keeps the layer and model definitions shared while the scalar
  instance determines their numerical meaning.
- Cross-checking happens at the scalar semantics layer, not inside duplicated model definitions.
- `Context α` is larger than a minimal arithmetic interface, but avoids duplicating architectures
  across execution, proof, and verification code.

Related work:

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

This is broader than a standard algebraic structure: it bundles arithmetic, ordering, and common
transcendental functions (exp/tanh/log/sqrt) used by activations and losses.

## Notes

- Many spec definitions assume `[Context α]` so they can be re‑used at multiple dtypes.
- For "paper theorems", the spec layer fixes `Spec.SpecScalar := ℝ` (see
  `NN/Spec/Core/Scalar.lean`).
- `Context.decidableGT` is included so executable code can decide comparisons (e.g. ReLU / argmax).
  The derived global `DecidableRel` instance has low priority so native decision procedures win.
- `LawfulContext α` (below) records when a `Context` agrees with a Mathlib ordered field on `α`.
- The `ℝ` context dictionary lives in `NN.Spec.Core.Context.Real` and the opt-in rational one in
  `NN.Spec.Core.Context.Rational`. FloatLib's shared elementary-function class already imports real
  analysis; selecting a context does not add an accuracy theorem for finite-precision arithmetic.
- For executable examples, `Context.gtBool` converts `x > y` into a printable `Bool`.
- For interval arithmetic, we override some order/comparison behavior (see `namespace Interval`
  below).
-/

@[expose] public section

/-- The full scalar interface required by spec-level tensors and models. -/
class Context (α : Type) extends
  Inhabited α, One α, Zero α,
  Add α, Sub α, Mul α, Div α, Neg α, Pow α α, Max α, Min α,
  BEq α, LT α, LE α, -- For ordering
  MathFunctions α,
  NatCast α, RatCast α, TorchLean.Numeric.QuotientArithmetic α where
  /--
  Convert a rational constant using the backend's arithmetic.

  The default casts its numerator and denominator separately. Native `Float`, `Float32`, and
  configured binary backends override this to round the exact fraction, avoiding overflow in
  either intermediate integer cast.
  -/
  ratCast value :=
    let numerator : α := match value.num with
      | .ofNat n => (n : α)
      | .negSucc n => -((n + 1 : Nat) : α)
    numerator / (value.den : α)
  /-- Backend-selected safeguard used by default in guarded formulas; not machine epsilon. -/
  defaultEpsilon : α
  /-- Decision procedure for the scalar type's strict order. -/
  decidableGT : DecidableRel (· > · : α → α → Prop)
  /--
  Remove differentiation metadata from a scalar at a `detach` boundary.

  Ordinary numeric carriers leave this as `none`, so detaching a tensor can reuse its storage.
  A carrier such as `Dual α` supplies a map that keeps the primal value and clears its tangents.
  This is needed when a reverse pass runs over dual numbers: cutting the tape edge alone would
  leave the detached value's forward tangent available to later operations.
  -/
  stopGradient? : Option (α → α) := none

namespace Context

/-- Decide `x > y` as a `Bool` using the `Context`'s `decidableGT`. -/
def gtBool {α : Type} [Context α] (x y : α) : Bool :=
  let _ : Decidable (x > y) := (Context.decidableGT) x y
  decide (x > y)

/-- Clear a scalar's differentiation metadata, if its carrier has any. -/
def stopGradient {α : Type} [Context α] (x : α) : α :=
  match Context.stopGradient? (α := α) with
  | none => x
  | some clear => clear x

end Context

/--
A `Context` includes a decidable `>` relation; expose it as a standard typeclass.

The instance has low priority so that a type's own decision procedure (`Real.decidableLT`,
`Float.decLt`, ...) wins whenever one exists.
-/
instance (priority := low) {α : Type} [Context α] :
    DecidableRel ((· > ·) : α → α → Prop) :=
  Context.decidableGT

/-!
## Lawful contexts

`Context α` carries no laws: it only bundles operations. `LawfulContext α` records, for a scalar
type that also carries a Mathlib ordered field structure, that the `Context` dictionary computes
exactly the same operations. Generic proofs about scalar-polymorphic specifications can then
rewrite the dictionary operations into the ordinary ring operations and finish with Mathlib.

Only exact facts are recorded. Transcendental constants (`lnTen`, `pi`, ...) and the total power
operation are backend specific and stay unconstrained.
-/

/--
Compatibility of a `Context` dictionary with a linearly ordered field structure on the same type.

Each field equates a `Context` projection (written with the explicit instance path
`Context.to*`) with the corresponding Mathlib operation.
-/
class LawfulContext (α : Type) [Context α] [Field α] [LinearOrder α]
    [IsStrictOrderedRing α] : Prop where
  /-- Dictionary addition is ring addition. -/
  add_eq (x y : α) :
    @HAdd.hAdd α α α (@instHAdd α Context.toAdd) x y =
      @HAdd.hAdd α α α (@instHAdd α Distrib.toAdd) x y
  /-- Dictionary multiplication is ring multiplication. -/
  mul_eq (x y : α) :
    @HMul.hMul α α α (@instHMul α Context.toMul) x y =
      @HMul.hMul α α α (@instHMul α Distrib.toMul) x y
  /-- Dictionary subtraction is ring subtraction. -/
  sub_eq (x y : α) :
    @HSub.hSub α α α (@instHSub α Context.toSub) x y =
      @HSub.hSub α α α (@instHSub α SubNegMonoid.toSub) x y
  /-- Dictionary division is field division. -/
  div_eq (x y : α) :
    @HDiv.hDiv α α α (@instHDiv α Context.toDiv) x y =
      @HDiv.hDiv α α α (@instHDiv α DivisionRing.toDiv) x y
  /-- Dictionary negation is ring negation. -/
  neg_eq (x : α) :
    @Neg.neg α Context.toNeg x = @Neg.neg α NegZeroClass.toNeg x
  /-- The dictionary zero is the ring zero. -/
  zero_eq : @Zero.zero α Context.toZero = @Zero.zero α MulZeroClass.toZero
  /-- The dictionary one is the ring one. -/
  one_eq : @One.one α Context.toOne = @One.one α Monoid.toOne
  /-- The dictionary strict order is the field order. -/
  lt_iff (x y : α) : @LT.lt α Context.toLT x y ↔ @LT.lt α Preorder.toLT x y
  /-- The dictionary order is the field order. -/
  le_iff (x y : α) : @LE.le α Context.toLE x y ↔ @LE.le α Preorder.toLE x y
  /-- Dictionary `max` is the lattice maximum. -/
  max_eq (x y : α) : @Max.max α Context.toMax x y = @Max.max α LinearOrder.toMax x y
  /-- Dictionary `min` is the lattice minimum. -/
  min_eq (x y : α) : @Min.min α Context.toMin x y = @Min.min α LinearOrder.toMin x y
  /-- Boolean equality decides propositional equality. -/
  beq_iff (x y : α) : (x == y) = true ↔ x = y
  /-- The natural-number cast is the semiring cast. -/
  natCast_eq (n : Nat) :
    @Nat.cast α Context.toNatCast n = @Nat.cast α Semiring.toNatCast n
  /-- The rational-number cast is the field cast. -/
  ratCast_eq (value : Rat) :
    @Rat.cast α Context.toRatCast value = @Rat.cast α DivisionRing.toRatCast value
  /-- The dictionary absolute value is the lattice absolute value. -/
  abs_eq (x : α) : MathFunctions.abs x = |x|
  /-- The backend tolerance is strictly positive. -/
  defaultEpsilon_pos : (0 : α) < Context.defaultEpsilon

attribute [simp] LawfulContext.add_eq LawfulContext.mul_eq LawfulContext.sub_eq
  LawfulContext.div_eq LawfulContext.neg_eq LawfulContext.zero_eq LawfulContext.one_eq
  LawfulContext.lt_iff LawfulContext.le_iff LawfulContext.max_eq LawfulContext.min_eq
  LawfulContext.beq_iff LawfulContext.natCast_eq LawfulContext.abs_eq

/-- Full `Context` instance for `Float` (runtime backend). -/
@[inline] instance : Context Float where
  defaultEpsilon := 1e-6
  decidableGT := inferInstance
  ratCast value := TorchLean.Numeric.QuotientArithmetic.roundFloat value
/-- Full `Context` instance for native binary32 execution. -/
@[inline] instance : Context Float32 where
  defaultEpsilon := (1e-6 : Float).toFloat32
  decidableGT := inferInstance
  ratCast value :=
    TorchLean.Numeric.QuotientArithmetic.roundFloat32 value
