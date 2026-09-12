/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

/-!
# Executable IEEE-754 binary32 (`IEEE32Exec`)

TorchLean uses two complementary ways to talk about "float32":

- `NN/Floats/NeuralFloat/*` and `NN/Floats/FP32/*` model rounding-on-`ℝ`. This is suited to proofs
  and for compositional "real computation + rounding error" arguments.
- `IEEE32Exec` (in this file) models bit-level binary32 values and the IEEE-754 operations exposed
  by this subsystem. This is what you want for corner cases such as NaN/Inf payload propagation,
  signed zero, and exact tie-breaking.

We implement `IEEE32Exec` as raw `UInt32` bits and provide:

- decoders/encoders for the binary32 layout,
- `nextUp`/`nextDown` (adjacent representable floats),
- basic arithmetic (`+ - * / fma`) by decoding to an exact dyadic/rational intermediate and then
  rounding once (round-to-nearest, ties-to-even),
- comparisons and `min`/`max` with IEEE-754 NaN rules,
- `sqrt` via integer arithmetic on the exact input value, rounded back to binary32.

TorchLean's tensor specifications provide their `Context IEEE32Exec` adapter separately. The
executable kernel itself has no dependency on tensors, models, or runtime code.

## About transcendentals

IEEE-754 does not specify implementations for transcendental functions (`exp`, `tanh`, ...). In
practice those are provided by `libm` (or vendor math libraries) and vary across platforms.

The transcendental operations in this namespace use deterministic, pure Lean algorithms so examples
do not delegate to the host math library. Their formulas, range reduction, and rounding path are
fixed by the source. They are **not** claimed to be correctly rounded or to match a particular
hardware or `libm`; each proved or tested accuracy statement remains an explicit contract.

## References

- IEEE Standard for Floating-Point Arithmetic, IEEE 754-2019.
- David Goldberg, “What Every Computer Scientist Should Know About Floating-Point Arithmetic”,
  *ACM Computing Surveys* (1991). DOI: 10.1145/103162.103163
- Jean-Michel Muller et al., *Handbook of Floating-Point Arithmetic*, 2nd ed. (2018).
- S. Boldo, G. Melquiond, “Flocq: a unified Coq library for proving floating-point algorithms
  correct” (ARITH 2011). DOI: 10.1109/ARITH.2011.40
-/

@[expose] public section


namespace TorchLean.Floats.IEEE754

/-- Executable IEEE-754 binary32 value, stored as raw bits. -/
structure IEEE32Exec where
  /-- The 32 bits of the encoding, laid out as IEEE-754 requires: sign in bit 31, biased
  exponent in bits 30..23, trailing significand in bits 22..0. Storing the encoding rather than a
  sign/exponent/mantissa triple is what makes `IEEE32Exec` a faithful model of hardware: NaN
  payloads and the two zeros survive a round trip. -/
  bits : UInt32
  deriving DecidableEq, Repr

namespace IEEE32Exec

/-- Wrap raw binary32 bits as an `IEEE32Exec`. -/
@[inline] def ofBits (b : UInt32) : IEEE32Exec := ⟨b⟩

/-- Extract the raw binary32 bits of an `IEEE32Exec`. -/
@[inline] def toBits (x : IEEE32Exec) : UInt32 := x.bits

/-- `toBits (ofBits b) = b`. -/
@[simp] theorem toBits_ofBits (b : UInt32) : toBits (ofBits b) = b := rfl

/-- `ofBits (toBits x) = x`. -/
@[simp] theorem ofBits_toBits (x : IEEE32Exec) : ofBits (toBits x) = x := by
  cases x
  rfl

/-- Default inhabitant: all bits zero, i.e. `+0.0`. -/
instance : Inhabited IEEE32Exec where
  default := ofBits 0

/-!
## Binary32 bit layout

IEEE-754 binary32 is stored as:

- sign bit `s` in bit 31,
- exponent field `e` in bits 30..23 (8 bits, bias 127),
- fraction field `f` in bits 22..0 (23 bits).

For NaNs, the "quiet" bit is the top fraction bit.
-/

-- Masks/constants (binary32 layout: sign[31] exp[30..23] frac[22..0]).
/-- Mask selecting the sign bit (bit 31). -/
def signMask : UInt32 := 0x80000000

/-- Mask selecting the 8-bit exponent field (bits 30..23). -/
def expMask : UInt32 := 0x7F800000

/-- Mask selecting the 23-bit fraction field (bits 22..0). -/
def fracMask : UInt32 := 0x007FFFFF

/-- The IEEE-754 "quiet NaN" indicator bit (top fraction bit). -/
def quietBit : UInt32 := 0x00400000

/-- 8-bit value `0xFF`, used to test the “all ones” exponent field. -/
def expAllOnes : UInt32 := 0xFF

/-- True iff the sign bit (bit 31) is set. -/
@[inline] def signBit (x : IEEE32Exec) : Bool :=
  (x.bits &&& signMask) != 0

/-- Extract the 8-bit exponent field (bits 30..23). -/
@[inline] def expField (x : IEEE32Exec) : UInt32 :=
  (x.bits >>> 23) &&& expAllOnes

/-- Extract the 23-bit fraction field (bits 22..0). -/
@[inline] def fracField (x : IEEE32Exec) : UInt32 :=
  x.bits &&& fracMask

/-- Predicate for NaN: exponent all ones and fraction nonzero. -/
@[inline] def isNaN (x : IEEE32Exec) : Bool :=
  expField x == expAllOnes && fracField x != 0

/-- Predicate for quiet NaN (NaN with the quiet bit set). -/
@[inline] def isQNaN (x : IEEE32Exec) : Bool :=
  isNaN x && (x.bits &&& quietBit) != 0

/-- Predicate for signaling NaN (NaN with the quiet bit clear). -/
@[inline] def isSNaN (x : IEEE32Exec) : Bool :=
  isNaN x && (x.bits &&& quietBit) == 0

/-- Predicate for infinity: exponent all ones and fraction zero. -/
@[inline] def isInf (x : IEEE32Exec) : Bool :=
  expField x == expAllOnes && fracField x == 0

/-- Predicate for signed zero (both `+0` and `-0`). -/
@[inline] def isZero (x : IEEE32Exec) : Bool :=
  expField x == 0 && fracField x == 0

/-- Predicate for finiteness: exponent field is not all ones (excludes NaN/Inf). -/
@[inline] def isFinite (x : IEEE32Exec) : Bool :=
  expField x != expAllOnes

/-- `+0.0` as an executable binary32 constant. -/
@[inline] def posZero : IEEE32Exec := ofBits 0
/-- `-0.0` as an executable binary32 constant. -/
@[inline] def negZero : IEEE32Exec := ofBits signMask

/-- `+1.0` as an IEEE-754 binary32 constant. -/
@[inline] def posOne : IEEE32Exec := ofBits (0x3F800000 : UInt32)
/-- `-1.0` as an IEEE-754 binary32 constant. -/
@[inline] def negOne : IEEE32Exec := ofBits (0xBF800000 : UInt32)

/-- `+∞` as an executable binary32 constant. -/
@[inline] def posInf : IEEE32Exec := ofBits expMask
/-- `-∞` as an executable binary32 constant. -/
@[inline] def negInf : IEEE32Exec := ofBits (signMask ||| expMask)

/-- A canonical quiet NaN payload used by the executable kernel. -/
@[inline] def canonicalNaN : IEEE32Exec := ofBits (expMask ||| quietBit)

/-!
## NaN selection / payload propagation

IEEE-754 leaves some freedom in how NaNs are "chosen" when multiple NaNs appear.
For reproducibility (and nicer debugging), we make the choice deterministic (left-to-right) and we
quiet signaling NaNs by setting the quiet bit.
-/

/-- Quiet a NaN by setting the quiet bit (and leave non-NaNs unchanged). -/
@[inline] def quietNaN (x : IEEE32Exec) : IEEE32Exec :=
  if isNaN x then
    -- IEEE754: quiet NaN has the top fraction bit set.
    ofBits (x.bits ||| quietBit)
  else
    x

/-- If `x` is a NaN, return it (quieted). -/
@[inline] def chooseNaN1 (x : IEEE32Exec) : Option IEEE32Exec :=
  if isNaN x then some (quietNaN x) else none

/--
Choose a NaN from two operands.

This is the "NaN propagation" policy used by most binary ops in this file:

- if any operand is a signaling NaN, return that operand (quieted), left-to-right,
- otherwise if any operand is a quiet NaN, return that operand, left-to-right,
- otherwise return `none`.
-/
@[inline] def chooseNaN2 (x y : IEEE32Exec) : Option IEEE32Exec :=
  -- Prefer signaling NaNs (quieted), then quiet NaNs; deterministic left-to-right choice.
  if isSNaN x then some (quietNaN x)
  else if isSNaN y then some (quietNaN y)
  else if isNaN x then some (quietNaN x)
  else if isNaN y then some (quietNaN y)
  else none

/-- Like `chooseNaN2`, but for ternary ops (used for `fma`). -/
@[inline] def chooseNaN3 (x y z : IEEE32Exec) : Option IEEE32Exec :=
  if isSNaN x then some (quietNaN x)
  else if isSNaN y then some (quietNaN y)
  else if isSNaN z then some (quietNaN z)
  else if isNaN x then some (quietNaN x)
  else if isNaN y then some (quietNaN y)
  else if isNaN z then some (quietNaN z)
  else none

/-! ## Adjacent floats (`nextUp`/`nextDown`) -/

-- Smallest positive subnormal (2^-149) and its negative.
/-- Smallest positive subnormal (bit pattern `0x00000001`, value `2^-149`). -/
@[inline] def posMinSubnormal : IEEE32Exec := ofBits 0x00000001
/-- Smallest negative subnormal (bit pattern `0x80000001`, value `-2^-149`). -/
@[inline] def negMinSubnormal : IEEE32Exec := ofBits (signMask ||| 0x00000001)

-- Largest finite magnitude (just below ±Inf).
/-- Largest finite positive float32 (bit pattern `0x7F7FFFFF`). -/
@[inline] def posMaxFinite : IEEE32Exec := ofBits 0x7F7FFFFF
/-- Largest finite negative float32 (bit pattern `0xFF7FFFFF`). -/
@[inline] def negMaxFinite : IEEE32Exec := ofBits 0xFF7FFFFF

/--
`nextUp x` is the next representable float32 strictly greater than `x`.

IEEE-754 special cases:
- NaN propagates (quieted).
- `nextUp (+∞) = +∞`.
- `nextUp (-0) = +minSubnormal` (since `+0` is not strictly greater than `-0`).
-/
@[inline] def nextUp (x : IEEE32Exec) : IEEE32Exec :=
  if isNaN x then
    quietNaN x
  else if x.bits == posInf.bits then
    posInf
  else if isZero x && signBit x then
    posMinSubnormal
  else if signBit x then
    ofBits (x.bits - 1)
  else
    ofBits (x.bits + 1)

/--
`nextDown x` is the next representable float32 strictly less than `x`.

IEEE-754 special cases:
- NaN propagates (quieted).
- `nextDown (-∞) = -∞`.
- `nextDown (+0) = -minSubnormal` (since `-0` is not strictly less than `+0`).
-/
@[inline] def nextDown (x : IEEE32Exec) : IEEE32Exec :=
  if isNaN x then
    quietNaN x
  else if x.bits == negInf.bits then
    negInf
  else if isZero x && !signBit x then
    negMinSubnormal
  else if signBit x then
    ofBits (x.bits + 1)
  else
    ofBits (x.bits - 1)

/--
Flip only the sign bit.

IEEE sign-bit operations do not quiet signaling NaNs: their exponent, payload, and quiet/signaling
classification are preserved. Arithmetic operations quiet an sNaN when they consume it.
-/
@[inline] def neg (x : IEEE32Exec) : IEEE32Exec :=
  ofBits (x.bits ^^^ signMask)

/-- Clear only the sign bit, preserving every NaN payload bit and its quiet/signaling class. -/
@[inline] def abs (x : IEEE32Exec) : IEEE32Exec :=
  ofBits (x.bits &&& (~~~signMask))


end IEEE32Exec

end TorchLean.Floats.IEEE754
