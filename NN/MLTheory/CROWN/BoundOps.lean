/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Context
public import NN.Tensor.Internal.Representation.Storage

/-!
# Directed-rounding primitives for interval propagation

TorchLean’s IBP/CROWN code represents bounds as endpoint pairs (`lo`/`hi`) inside `Box`/`FlatBox`.
To make those bounds meaningful under different numeric semantics, we abstract the *primitive*
endpoint operations (directed rounding).

Intuition:
- For pure real/interval backends, using ordinary `+`/`*` is already enclosure-safe (because the
  scalar itself is an interval type with outward rounding).
- For finite-precision backends with discrete grids (e.g. `ExecFloat.Binary 8 23`), we want
*directed rounding*
  primitives like `addDown/addUp` and `mulDown/mulUp` so that interval propagation encloses the
  corresponding exact real operation.

This file defines two small numeric interfaces:

- `BoundOps` for directed elementary arithmetic; and
- `NonlinearBoundOps` for interval transfers that may be unavailable on a backend.

There is intentionally no generic fallback for `BoundOps`: ordinary finite-precision arithmetic is
not directed rounding and must not silently enter a sound bound-propagation path. The nonlinear
interface does have a conservative fallback for bounded activations, but operations with unbounded
ranges return `none` unless the scalar backend supplies an implementation. Soundness is a separate
obligation, recorded by `LawfulNonlinearBoundOps`.

## Integration points in the current codebase

The intended usage is:

- Keep graphs/layers scalar-polymorphic over `[TorchLean.Storage α] [Context α]`.
- When a routine *propagates bounds* (IBP/affine/CROWN), also require `[BoundOps α]` and use
  `addDown/addUp/subDown/subUp/mulDown/mulUp` at the endpoints.

Concretely:

- `NN/MLTheory/CROWN/Core.lean`
  - `AffineVec.evalOnBox`: min/max over products and accumulation use `BoundOps`.
  - `IBP.linear`: interval linear layer propagation uses `BoundOps`.
- `NN.MLTheory.CROWN.Graph`
  - `boxAdd`, `boxSub`, `boxMulElem`: endpoint propagation uses `BoundOps`.

Executable certificate replay relies on this separation. A backend may support directed binary
arithmetic without having a correctly rounded `exp` or `log`; in that case the graph checker leaves
the corresponding node unresolved instead of treating an ordinary library call as an enclosure.
-/

@[expose] public section

namespace NN.MLTheory.CROWN

variable {α : Type} [TorchLean.Storage α] [Context α]

/-!
## `BoundOps α`

`BoundOps` supplies directed-rounding versions of the arithmetic
primitives that appear in IBP for affine/linear layers and basic arithmetic nodes.

If you want to swap in a quantized backend, the key is to provide an instance of `BoundOps` for
your scalar type.
-/
class BoundOps (α : Type) [TorchLean.Storage α] [Context α] where
  addDown : α → α → α
  addUp   : α → α → α
  subDown : α → α → α
  subUp   : α → α → α
  mulDown : α → α → α
  mulUp   : α → α → α
  /-- Whether ordinary scalar algebra may be reassociated without a rounding error. -/
  supportsExactAffineReassociation : Bool := false

namespace BoundOps

/-- Minimum of two scalar endpoints. -/
@[inline] def min2 (a b : α) : α :=
  if decide (a > b) then b else a

/-- Maximum of two scalar endpoints. -/
@[inline] def max2 (a b : α) : α :=
  if decide (a > b) then a else b

end BoundOps

/-!
## Nonlinear enclosure operations

Each method consumes a closed interval `[lo, hi]`. A successful result is another endpoint pair;
`none` means that this backend does not implement a finite transfer for the requested operation.
Division receives both numerator and denominator intervals. The executable result alone makes no
soundness claim; `LawfulNonlinearBoundOps` supplies that claim when a theorem needs it.

The interface is deliberately operational, like `BoundOps`. The proof layer establishes soundness
for the concrete implementations used by checked workflows; an external instance without a lawful
instance remains part of the backend trust boundary.
-/

/-- Interval enclosures for the nonlinear scalar operations a bound-propagation pass needs.

Each method takes the endpoints of the input interval (division takes both operands' endpoints) and
returns the endpoints of an enclosure of the image, or `none` when the backend declines to bound
that input, for example a logarithm on an interval reaching zero. -/
class NonlinearBoundOps (α : Type) [TorchLean.Storage α] [Context α] where
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
  /-- Whether coupled softmax/layer-normalization derivative formulas use exact scalar
  arithmetic. -/
  supportsIdealCoupledDerivatives : Bool

namespace NonlinearBoundOps

/-- Minimum of four endpoints. -/
def min4 (a b c d : α) : α :=
  BoundOps.min2 (BoundOps.min2 a b) (BoundOps.min2 c d)

/-- Maximum of four endpoints. -/
def max4 (a b c d : α) : α :=
  BoundOps.max2 (BoundOps.max2 a b) (BoundOps.max2 c d)

/-- The denominator interval avoids zero. -/
def denominatorAvoidsZero (lo hi : α) : Bool :=
  decide (lo > 0) || decide (0 > hi)

/-- A coarse softplus enclosure that stays finite without evaluating an exponential.

For a real input, `max x 0 ≤ softplus x ≤ max x 0 + 1`: after the sign branch, the
remaining logarithm lies between zero and `log 2 < 1`. Directed addition therefore gives an
enclosure even when an endpoint is too large for an exponential-based transfer. The graph checker
can use this range as a constant affine bound on every backend that supplies directed arithmetic. -/
def softplusBounds [BoundOps α] (lo hi : α) : Option (α × α) :=
  some (BoundOps.max2 lo 0, BoundOps.addUp (BoundOps.max2 hi 0) 1)

/-- Enclose safeLog using the complete input and epsilon intervals.

The lower endpoint of `softplus(x) + epsilon` must be positive. This checks the supplied epsilon,
including values smaller than the context default. A backend's logarithm transfer gives the first
choice of bounds. If it is unavailable, `1 - 1/z ≤ log z ≤ z - 1` supplies a coarser enclosure
using directed reciprocal and subtraction. Neither route evaluates an unbounded exponential. -/
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

Sigmoid, tanh, sine, and cosine have format-independent codomain bounds. Unbounded operations and
layer normalization remain unavailable until a concrete backend provides directed implementations.
-/
instance (priority := 100) instNonlinearBoundOpsConservative : NonlinearBoundOps α where
  divBounds := fun _ _ _ _ => none
  expBounds := fun _ _ => none
  logBounds := fun _ _ => none
  sqrtBounds := fun _ _ => none
  sigmoidBounds := fun _ _ => some (0, 1)
  tanhBounds := fun _ _ => some ((-1), 1)
  sinBounds := fun _ _ => some ((-1), 1)
  cosBounds := fun _ _ => some ((-1), 1)
  layerNormAbsBound := fun _ => none
  supportsIdealCoupledDerivatives := false

/-!
## Host binary64 endpoints

Lean's `Float` operations round to nearest on the host binary64 format. For executable checking we
widen every finite result by one adjacent representable value. This is deliberately an explicit
instance rather than a generic fallback: its soundness depends on the host IEEE-754 arithmetic
boundary documented by Lean, whereas `instBoundOpsReal` is exact and the `ExecFloat.Binary 8 23`
instance is
connected to TorchLean's bit-level binary32 proofs.
-/

namespace HostFloat

/-- Sign bit of a binary64 word: clearing it gives the magnitude, testing it gives the sign. -/
def signMask : UInt64 := 0x8000000000000000

/-- Bit pattern of `+∞` in binary64. Stepping up from here has to stay put. -/
def posInfBits : UInt64 := 0x7ff0000000000000

/-- Bit pattern of `-∞` in binary64. Stepping down from here has to stay put. -/
def negInfBits : UInt64 := 0xfff0000000000000

/-- Adjacent binary64 value above `x`, with the usual IEEE behavior at infinities and zeros. -/
def nextUp (x : Float) : Float :=
  let bits := x.toBits
  if x.isNaN || bits = posInfBits then
    x
  else if bits = signMask || bits = 0 then
    Float.ofBits 1
  else if bits &&& signMask = 0 then
    Float.ofBits (bits + 1)
  else
    Float.ofBits (bits - 1)

/-- Adjacent binary64 value below `x`, with the usual IEEE behavior at infinities and zeros. -/
def nextDown (x : Float) : Float :=
  let bits := x.toBits
  if x.isNaN || bits = negInfBits then
    x
  else if bits = signMask || bits = 0 then
    Float.ofBits (signMask + 1)
  else if bits &&& signMask = 0 then
    Float.ofBits (bits - 1)
  else
    Float.ofBits (bits + 1)

end HostFloat

/--
Outward-widened host binary64 operations.

This instance is suitable for executable certificate replay under the trusted host-Float boundary.
Use `ExecFloat.Binary 8 23` when the binary32 endpoint calculation itself must be connected to Lean
proofs.
-/
instance instBoundOpsFloat : BoundOps Float where
  addDown a b := HostFloat.nextDown (a + b)
  addUp a b := HostFloat.nextUp (a + b)
  subDown a b := HostFloat.nextDown (a - b)
  subUp a b := HostFloat.nextUp (a - b)
  mulDown a b := HostFloat.nextDown (a * b)
  mulUp a b := HostFloat.nextUp (a * b)

/--
Nonlinear enclosures supplied by host binary64 arithmetic.

Division and square root use hardware operations widened by one adjacent binary64 value. We do not
make the same claim for host transcendental-library calls, so `exp` and `log` remain unsupported.
-/
instance instNonlinearBoundOpsFloat : NonlinearBoundOps Float where
  divBounds aLo aHi bLo bHi :=
    if NonlinearBoundOps.denominatorAvoidsZero bLo bHi then
      let p1 := aLo / bLo
      let p2 := aLo / bHi
      let p3 := aHi / bLo
      let p4 := aHi / bHi
      let lo := NonlinearBoundOps.min4 p1 p2 p3 p4
      let hi := NonlinearBoundOps.max4 p1 p2 p3 p4
      some (HostFloat.nextDown lo, HostFloat.nextUp hi)
    else
      none
  expBounds := fun _ _ => none
  logBounds := fun _ _ => none
  sqrtBounds lo hi :=
    if hi < 0 then
      none
    else
      let lo' := if lo > 0 then lo else 0
      some (HostFloat.nextDown (MathFunctions.sqrt lo'),
        HostFloat.nextUp (MathFunctions.sqrt hi))
  sigmoidBounds := fun _ _ => some (0, 1)
  tanhBounds := fun _ _ => some ((-1), 1)
  sinBounds := fun _ _ => some ((-1), 1)
  cosBounds := fun _ _ => some ((-1), 1)
  layerNormAbsBound := fun n =>
    some (HostFloat.nextUp (MathFunctions.sqrt (Float.ofNat n)))
  supportsIdealCoupledDerivatives := false

/-!
## Native binary32 endpoints

The same one-ULP widening policy is available for Lean's native `Float32`. The operations execute
with binary32 rounding; moving to the adjacent representable value turns each nearest-rounded
result into an outward endpoint.
-/

namespace HostFloat32

/-- Sign bit of a binary32 word, the `Float32` counterpart of `HostFloat.signMask`. -/
def signMask : UInt32 := 0x80000000

/-- Bit pattern of `+∞` in binary32. -/
def posInfBits : UInt32 := 0x7f800000

/-- Bit pattern of `-∞` in binary32. -/
def negInfBits : UInt32 := 0xff800000

/-- Adjacent binary32 value above `x`, preserving NaNs and positive infinity. -/
def nextUp (x : Float32) : Float32 :=
  let bits := x.toBits
  if x.isNaN || bits = posInfBits then
    x
  else if bits = signMask || bits = 0 then
    Float32.ofBits 1
  else if bits &&& signMask = 0 then
    Float32.ofBits (bits + 1)
  else
    Float32.ofBits (bits - 1)

/-- Adjacent binary32 value below `x`, preserving NaNs and negative infinity. -/
def nextDown (x : Float32) : Float32 :=
  let bits := x.toBits
  if x.isNaN || bits = negInfBits then
    x
  else if bits = signMask || bits = 0 then
    Float32.ofBits (signMask + 1)
  else if bits &&& signMask = 0 then
    Float32.ofBits (bits - 1)
  else
    Float32.ofBits (bits + 1)

end HostFloat32

/-- Outward-widened native binary32 operations. -/
instance instBoundOpsFloat32 : BoundOps Float32 where
  addDown a b := HostFloat32.nextDown (a + b)
  addUp a b := HostFloat32.nextUp (a + b)
  subDown a b := HostFloat32.nextDown (a - b)
  subUp a b := HostFloat32.nextUp (a - b)
  mulDown a b := HostFloat32.nextDown (a * b)
  mulUp a b := HostFloat32.nextUp (a * b)

/-- Native binary32 nonlinear enclosures for division and square root. -/
instance instNonlinearBoundOpsFloat32 : NonlinearBoundOps Float32 where
  divBounds aLo aHi bLo bHi :=
    if NonlinearBoundOps.denominatorAvoidsZero bLo bHi then
      let p1 := aLo / bLo
      let p2 := aLo / bHi
      let p3 := aHi / bLo
      let p4 := aHi / bHi
      let lo := NonlinearBoundOps.min4 p1 p2 p3 p4
      let hi := NonlinearBoundOps.max4 p1 p2 p3 p4
      some (HostFloat32.nextDown lo, HostFloat32.nextUp hi)
    else
      none
  expBounds := fun _ _ => none
  logBounds := fun _ _ => none
  sqrtBounds lo hi :=
    if hi < 0 then none
    else some (HostFloat32.nextDown (Float32.sqrt (max lo 0)),
      HostFloat32.nextUp (Float32.sqrt hi))
  sigmoidBounds := fun _ _ => none
  tanhBounds := fun _ _ => none
  sinBounds := fun _ _ => some (-1, 1)
  cosBounds := fun _ _ => some (-1, 1)
  layerNormAbsBound := fun _ => none
  supportsIdealCoupledDerivatives := false

end NN.MLTheory.CROWN
