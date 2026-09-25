/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import FloatLib.Floats.Formats.BinaryInterchange.Format.Catalog
public import FloatLib.Floats.Formats.Flocq

import Mathlib.Algebra.Order.Algebra

/-!
# `FP32`: TorchLean's proof-oriented float32 semantics

`FP32` is a proof-oriented float32 semantics: it models float computations as real-number operations
over $\mathbb{R}$ followed by rounding to a fixed binary32 grid after each primitive operation. This
abstraction is designed to support compositional rounding-error bounds over long computations.

What this model *does* cover:

- binary radix $\beta=2$
- IEEE-754-like binary32 exponent/precision parameters, including gradual underflow
- rounding to nearest with ties-to-even

What this model does *not* cover:
- NaN/Inf payload rules, signed-zero corner cases, and other IEEE “special values”

Special-value semantics live in the executable `ExecFloat.Binary 8 23` model. Bridge lemmas relate
`ExecFloat.Binary 8 23`
back to `FP32` on the finite/no-overflow fragment.
-/

@[expose] public section

open FloatLib.Numerics FloatLib.Floats.Formats.Flocq
open FloatLib.Floats.Formats.BinaryInterchange (FloatFormat)


namespace TorchLean.Floats

/-! ## Canonical IEEE-754 binary32 configuration -/

/--
Exponent function for the gradual-underflow part of IEEE-754 binary32, expressed in Flocq style.

Two numbers here matter:

- `prec = 24`: binary32 has 23 stored fraction bits, but 24 bits of precision for *normal* numbers
  once you include the implicit leading $1$.
- `emin = -149`: the smallest positive *subnormal* is $2^{-149}$. Using `emin = -149` is the usual
  way to encode gradual underflow in this “rounding on $\mathbb{R}$” model.

`fltExp` has no upper exponent bound. Overflow and the transition to infinity belong to
`ExecFloat.Binary 8 23`, not this exponent function.
-/
def fexp32 : ℤ → ℤ := fltExp (-149) 24

/-- `fexp32` is the binary32-compatible exponent function used by `FP32`. -/
instance : ValidExp fexp32 :=
  fltValidExp (emin := (-149)) (prec := 24) (by decide)

/--
Round-to-nearest, ties-to-even (binary32-style default rounding).

This is the rounding mode people typically assume when they say “IEEE float32 rounding”.
-/
noncomputable def rnd32 : ℝ → ℤ := nearestEven

/-- `rnd32` is a valid monotone rounding mode in the generic neural-float sense. -/
instance : ValidRnd rnd32 := by
  dsimp [rnd32]
  infer_instance

/-- `rnd32` is round-to-nearest in the `ValidRndToNearest` sense. -/
instance : ValidRndToNearest rnd32 := by
  dsimp [rnd32]
  infer_instance

/--
`FP32`: finite float32 rounding model, as a rounded real value.

This is the type you want if you are proving numerical stability/error bounds without dealing with
NaN/Inf behavior.
-/
abbrev FP32 : Type := NF binaryRadix fexp32 rnd32

namespace FP32

/-- Forgetful projection: treat an `FP32` value as a real number. -/
abbrev toReal (x : FP32) : ℝ := x.val

/-- Canonical effective mantissa/exponent representation of binary32 rounded-real rounding. -/
theorem round_eq_computed (x : ℝ) :
    round (β := binaryRadix) (fexp := fexp32) rnd32 x =
      FloatLib.Floats.Formats.Flocq.toReal (β := binaryRadix) {
        mantissa := nearestEvenMantissa
          (scaledMantissa binaryRadix fexp32 x)
        exponent := cexp binaryRadix fexp32 x } :=
  round_nearestEven_computed (β := binaryRadix) (fexp := fexp32) x

/--
The result of FP32 addition has the canonical mantissa/exponent representation computed by the
effective nearest-even rounding layer.
-/
theorem add_toReal_eq_computed (a b : FP32) :
    toReal (a + b) =
      FloatLib.Floats.Formats.Flocq.toReal (β := binaryRadix) {
        mantissa := nearestEvenMantissa
          (scaledMantissa binaryRadix fexp32 (a.val + b.val))
        exponent := cexp binaryRadix fexp32 (a.val + b.val) } :=
  round_eq_computed (a.val + b.val)

/--
Convenience constant: the smallest positive normal binary32 number, exactly $2^{-126}$
(`minNormal_eq_bpow`).

Subnormals exist below this; this constant is mainly useful when you want to distinguish
“normal-range” arguments from “subnormal-range” arguments in proofs.
-/
noncomputable def minNormal : ℝ :=
  bpow binaryRadix FloatFormat.binary32.minNormalExponent

/-- The binary32 minimum normal value is $2^{-126}$. -/
@[simp] theorem minNormal_eq_bpow : minNormal = bpow binaryRadix (-126) := by
  rfl

end FP32

end TorchLean.Floats
