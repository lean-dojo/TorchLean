/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Elementwise

/-!
# Directed-rounding primitives for interval propagation

TorchLean's IBP and CROWN code stores bounds as endpoint pairs (`lo`, `hi`) in a `Box` or a
`FlatBox`. What an endpoint means depends on the scalar, so we abstract the primitive endpoint
operations. Over an exact scalar such as `ℝ`, or `ℚ` with its scoped exact instance, ordinary `+`
and `*` already give enclosures. On a finite grid such as `ExecFloat.Binary 8 23`, interval
propagation needs directed rounding (`addDown`/`addUp`, `mulDown`/`mulUp` and so on) to enclose
the exact real operation.

Two interfaces carry this:

- `BoundOps` for directed elementary arithmetic, and
- `NonlinearBoundOps` for interval transfers that a backend may not offer.

`BoundOps` has no generic fallback on purpose: ordinary finite-precision arithmetic does not round
in a fixed direction and must not slip into a sound bound-propagation path. `NonlinearBoundOps`
has a conservative fallback for the bounded operations, but operations with unbounded ranges
return `none` unless the scalar backend supplies an implementation. Soundness is a separate
obligation, recorded by `LawfulNonlinearBoundOps`.

## Where the operations are used

Graphs and layers stay scalar-polymorphic over `[TorchLean.Storage α] [Context α]`. A routine that
propagates bounds (IBP, affine, CROWN) also requires `[BoundOps α]` and uses
`addDown`/`addUp`/`subDown`/`subUp`/`mulDown`/`mulUp` at the endpoints:

- `TorchLean/Verify/CROWN.lean`
  - `TorchLean.Linear.evalOnBox`: the min/max over products and the accumulation use `BoundOps`.
  - `IBP.linear`: interval propagation through a linear layer uses `BoundOps`.
- `TorchLean/Verify/CROWN/FlatBox/Transfer.lean`
  - `boxAdd`, `boxSub`, `boxMulElem`: endpoint propagation uses `BoundOps`.

Executable certificate replay relies on this separation. A backend may have directed binary
arithmetic but no correctly rounded `exp` or `log`, and then the graph checker leaves the
corresponding node unresolved instead of treating an ordinary library call as an enclosure.
-/

@[expose] public section

open FloatLib.Numerics (MathFunctions)

namespace TorchLean.CROWN

variable {α : Type} [Storage α] [Context α]

/-!
## `BoundOps α`

`BoundOps` supplies directed-rounding versions of the arithmetic primitives that IBP needs for
affine and linear layers and for basic arithmetic nodes. A quantized backend joins bound
propagation by providing a `BoundOps` instance for its scalar type.
-/
/-- Directed endpoint arithmetic, plus one capability flag, for computing interval and affine
bounds. Soundness needs the separate laws of `LawfulBoundOps` for the chosen scalar. -/
class BoundOps (α : Type) [Storage α] [Context α] where
  /-- Addition directed downward, for lower endpoints. -/
  addDown : α → α → α
  /-- Addition directed upward, for upper endpoints. -/
  addUp   : α → α → α
  /-- Subtraction directed downward, for lower endpoints. -/
  subDown : α → α → α
  /-- Subtraction directed upward, for upper endpoints. -/
  subUp   : α → α → α
  /-- Multiplication directed downward, for lower endpoints. -/
  mulDown : α → α → α
  /-- Multiplication directed upward, for upper endpoints. -/
  mulUp   : α → α → α
  /-- Whether ordinary scalar algebra may be reassociated without a rounding error. -/
  exactReassociation : Bool := false

namespace BoundOps

/-- Endpoint arithmetic that does not round, for exact scalars such as `ℝ` and `ℚ` only: both
directions are the scalar's own `+`, `-` and `*`, and affine reassociation is allowed. It is a
definition and never an instance, so no rounded backend picks it up by accident. -/
abbrev exact (α : Type) [Storage α] [Context α] : BoundOps α where
  addDown := (· + ·)
  addUp := (· + ·)
  subDown := (· - ·)
  subUp := (· - ·)
  mulDown := (· * ·)
  mulUp := (· * ·)
  exactReassociation := true

end BoundOps

/-!
## Nonlinear enclosure operations

`NonlinearBoundOps` is operational, like `BoundOps`: its methods compute endpoint pairs, and what
those pairs mean is stated separately by `LawfulNonlinearBoundOps`. The proof layer establishes
those laws for the concrete implementations that checked workflows use. An external instance
without a lawful instance stays inside the backend's trust boundary.
-/

/-- Interval enclosures for the nonlinear scalar operations a bound-propagation pass needs.

Each method takes the endpoints of the input interval (division takes both operands' endpoints) and
returns the endpoints of an enclosure of the image, or `none` when the backend declines to bound
that input, for example a logarithm on an interval reaching zero. -/
class NonlinearBoundOps (α : Type) [Storage α] [Context α] where
  /-- Enclose `x / y` from the endpoints of `x` and then of `y`. -/
  divBounds : α → α → α → α → Option (α × α)
  /-- Enclose `exp x` on `[l, u]`. -/
  expBounds : α → α → Option (α × α)
  /-- Enclose `log x` on `[l, u]`. -/
  logBounds : α → α → Option (α × α)
  /-- Enclose `sqrt x` on `[l, u]`. -/
  sqrtBounds : α → α → Option (α × α)
  /-- Enclose `sigmoid x` on `[l, u]`. -/
  sigmoidBounds : α → α → Option (α × α)
  /-- Enclose `tanh x` on `[l, u]`. -/
  tanhBounds : α → α → Option (α × α)
  /-- Enclose `sin x` on `[l, u]`. -/
  sinBounds : α → α → Option (α × α)
  /-- Enclose `cos x` on `[l, u]`. -/
  cosBounds : α → α → Option (α × α)
  /-- Uniform absolute bound for one last-axis layer-normalization row. -/
  layerNormAbsBound : Nat → Option α

namespace NonlinearBoundOps

/-- Whether the denominator interval `[lo, hi]` avoids zero: `0 < lo` or `hi < 0`. -/
@[inline] def denominatorAvoidsZero (lo hi : α) : Bool :=
  decide (lo > 0) || decide (0 > hi)

/-- A coarse softplus enclosure that avoids evaluating an exponential.

For a real input, `max x 0 ≤ softplus x ≤ max x 0 + 1`: after the sign branch, the
remaining logarithm lies between zero and `log 2 < 1`. Directed addition therefore gives an
enclosure even when an endpoint is too large for an exponential-based transfer. The upper
endpoint can still overflow during directed addition. The graph checker can use this range as a
constant affine bound on every backend that supplies directed arithmetic. We derived the envelope
by hand. `Activation.softplus_envelope_real` proves it over `ℝ`, and
`Directed.softplusBounds_enclosure` proves the directed interval at every lawful backend. -/
def softplusBounds [BoundOps α] (lo hi : α) : Option (α × α) :=
  some (Context.maximum lo 0, BoundOps.addUp (Context.maximum hi 0) 1)

/-- Enclose safeLog using the complete input and epsilon intervals.

The lower endpoint of `softplus(x) + epsilon` must be positive, and the check uses the supplied
epsilon, including values smaller than the context default. A backend's logarithm transfer gives
the first choice of bounds. If it is unavailable, `1 - 1/z ≤ log z ≤ z - 1` supplies a coarser
enclosure using directed reciprocal and subtraction. Neither route evaluates an unbounded
exponential. `Directed.safeLogBounds_enclosure` proves both routes. -/
def safeLogBounds [BoundOps α] [NonlinearBoundOps α]
    (lo hi epsilonLo epsilonHi : α) : Option (α × α) := do
  let (softLo, softHi) ← softplusBounds lo hi
  let sumLo := BoundOps.addDown softLo epsilonLo
  let sumHi := BoundOps.addUp softHi epsilonHi
  if sumLo > 0 then
    match logBounds sumLo sumHi with
    | some bounds => pure bounds
    | none => do
        let (_, reciprocalHi) ← divBounds 1 1 sumLo sumHi
        pure (BoundOps.subDown 1 reciprocalHi, BoundOps.subUp sumHi 1)
  else
    none

end NonlinearBoundOps

/--
Conservative nonlinear ranges available for every scalar context.

Sigmoid, tanh, sine and cosine have format-independent codomain bounds. Unbounded operations and
layer normalization return `none`, and a concrete backend with directed implementations overrides
them.
-/
instance (priority := 100) NonlinearBoundOps.conservative : NonlinearBoundOps α where
  divBounds := fun _ _ _ _ => none
  expBounds := fun _ _ => none
  logBounds := fun _ _ => none
  sqrtBounds := fun _ _ => none
  sigmoidBounds := fun _ _ => some (0, 1)
  tanhBounds := fun _ _ => some ((-1), 1)
  sinBounds := fun _ _ => some ((-1), 1)
  cosBounds := fun _ _ => some ((-1), 1)
  layerNormAbsBound := fun _ => none

/-!
## Host binary endpoints

Lean's `Float` and `Float32` operations round to nearest in binary64 and binary32. For executable
checking we widen every finite result by one adjacent representable value, which turns a
nearest-rounded result into an outward endpoint. These are named instances and never a generic
fallback, because their soundness depends on the host IEEE 754 arithmetic boundary documented by
Lean. `instBoundOpsReal` is exact, and the `ExecFloat.Binary 8 23` instance is connected to
TorchLean's bit-level binary32 proofs.

One adjacent value is enough. Rounding to nearest leaves the exact sum, difference or product `x`
within half a gap of a finite result `r`, so `nextDown r ≤ x ≤ nextUp r`. After an overflow,
`nextDown` of `+∞` is the largest finite value `M` and `nextUp` of `-∞` is `-M`, and both stay on
the correct side of `x`. Neither `Float` nor `Float32` has a `LawfulBoundOps` instance, so no Lean
theorem states this bracket.
-/

namespace HostFloat

open TorchLean.Numeric (Word)

variable [Numeric.HostFloat α]

omit [Storage α] in
/-- IEEE 754-2019 `nextUp x`: the least host value above `x`, and `x` itself at `+∞` and at a
NaN. -/
@[inline] def nextUp (x : α) : α :=
  if Context.isNaN x || x == Numeric.HostFloat.posInf then x
  else if x == 0 then Word.ofBits 1
  else if Context.gtBool 0 x then Word.pred x
  else Word.succ x

omit [Storage α] in
/-- IEEE 754-2019 `nextDown x`: the greatest host value below `x`, and `x` itself at `-∞` and at
a NaN. -/
@[inline] def nextDown (x : α) : α :=
  if Context.isNaN x || x == Numeric.HostFloat.negInf then x
  else if x == 0 then -Word.ofBits 1
  else if Context.gtBool 0 x then Word.succ x
  else Word.pred x

end HostFloat

/-- Endpoint arithmetic for a host binary format: each nearest-rounded `+`, `-` and `*` widened by
one adjacent value, down with `HostFloat.nextDown` and up with `HostFloat.nextUp`. -/
abbrev BoundOps.host (α : Type) [Storage α] [Context α]
    [Numeric.HostFloat α] : BoundOps α where
  addDown a b := HostFloat.nextDown (a + b)
  addUp a b := HostFloat.nextUp (a + b)
  subDown a b := HostFloat.nextDown (a - b)
  subUp a b := HostFloat.nextUp (a - b)
  mulDown a b := HostFloat.nextDown (a * b)
  mulUp a b := HostFloat.nextUp (a * b)

/-- Nonlinear enclosures for a host binary format of precision `p`: host division of a finite
interval by a finite interval avoiding zero and square root widened by one adjacent value, and the
bounds `[0, 1]` of sigmoid and `[-1, 1]` of tanh, sine and cosine. A row of `n ≤ 2 ^ p` entries gets
`√n` widened up as its layer-normalization bound, and the host `exp` and `log` give no enclosure. -/
abbrev NonlinearBoundOps.host (α : Type) [Storage α] [Context α]
    [Numeric.HostFloat α] : NonlinearBoundOps α where
  divBounds aLo aHi bLo bHi :=
    -- Every endpoint must be finite, since a quotient such as `∞ / ∞` is a NaN.
    let finite (x : α) : Bool := !Context.isNaN x && !Context.isInf x
    if finite aLo && finite aHi && finite bLo && finite bHi &&
        NonlinearBoundOps.denominatorAvoidsZero bLo bHi then
      let p1 := aLo / bLo
      let p2 := aLo / bHi
      let p3 := aHi / bLo
      let p4 := aHi / bHi
      let lo := Context.minimum (Context.minimum p1 p2) (Context.minimum p3 p4)
      let hi := Context.maximum (Context.maximum p1 p2) (Context.maximum p3 p4)
      some (HostFloat.nextDown lo, HostFloat.nextUp hi)
    else
      none
  expBounds _ _ := none
  logBounds _ _ := none
  sqrtBounds lo hi :=
    if Context.gtBool 0 hi then
      none
    else
      -- Clip to the domain of the root with IEEE 754-2019 `maximum lo 0`, which keeps a NaN `lo`.
      let lo' := if Context.isNaN lo || Context.gtBool lo 0 then lo else 0
      some (HostFloat.nextDown (MathFunctions.sqrt lo'),
        HostFloat.nextUp (MathFunctions.sqrt hi))
  sigmoidBounds _ _ := some (0, 1)
  tanhBounds _ _ := some (-1, 1)
  sinBounds _ _ := some (-1, 1)
  cosBounds _ _ := some (-1, 1)
  -- Every count up to `2 ^ p` is a host value, so its cast is exact.
  layerNormAbsBound n :=
    if n ≤ 2 ^ ((Numeric.HostFloat.format α).fracWidth + 1) then
      some (HostFloat.nextUp (MathFunctions.sqrt (n : α)))
    else
      none

/-- Endpoint arithmetic for `Float`, widened outward by one binary64 value. -/
instance instBoundOpsFloat : BoundOps Float := .host Float

/-- Endpoint arithmetic for `Float32`, widened outward by one binary32 value. -/
instance instBoundOpsFloat32 : BoundOps Float32 := .host Float32

/-- Nonlinear enclosures for `Float`, from `NonlinearBoundOps.host`. -/
instance instNonlinearBoundOpsFloat : NonlinearBoundOps Float := .host Float

/-- Nonlinear enclosures for `Float32`: division, square root, sine and cosine as in
`NonlinearBoundOps.host`. Sigmoid, tanh, `exp`, `log` and the layer-normalization bound give
none. -/
instance instNonlinearBoundOpsFloat32 : NonlinearBoundOps Float32 :=
  { NonlinearBoundOps.host Float32 with
    sigmoidBounds := fun _ _ => none
    tanhBounds := fun _ _ => none
    layerNormAbsBound := fun _ => none }

end TorchLean.CROWN
