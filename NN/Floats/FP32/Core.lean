/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.NeuralFloat.Formats
public import NN.Floats.NeuralFloat.NF
import Mathlib.Algebra.Order.Algebra

/-!
# `FP32`: TorchLean's proof-oriented float32 semantics

`FP32` is a proof-oriented float32 semantics: it models float computations as real-number operations
(`ℝ`) followed by rounding to a fixed binary32 grid after each primitive operation. This
abstraction is designed to support compositional rounding-error bounds over long computations.

What this model *does* cover:
- binary radix (`β = 2`)
- IEEE-754-like binary32 exponent/precision parameters, including gradual underflow
- rounding to nearest with ties-to-even

What this model does *not* cover:
- NaN/Inf payload rules, signed-zero corner cases, and other IEEE “special values”

Special-value semantics live in the executable `IEEE32Exec` model. Bridge lemmas relate `IEEE32Exec`
back to `FP32` on the finite/no-overflow fragment.
-/

@[expose] public section


namespace TorchLean.Floats

/-! ## Canonical IEEE-754 binary32 configuration -/

/--
Exponent function for IEEE-754 binary32 (gradual underflow), expressed in Flocq style.

Two numbers here matter:

- `prec = 24`: binary32 has 23 stored fraction bits, but 24 bits of precision for *normal* numbers
  once you include the implicit leading `1`.
- `emin = -149`: the smallest positive *subnormal* is `2^-149`. Using `emin = -149` is the usual
  way to encode gradual underflow in this “rounding-on-ℝ” model.
-/
def fexp32 : ℤ → ℤ := FLTExp (-149) 24

/-- `fexp32` is the binary32-compatible exponent function used by `FP32`. -/
instance : NeuralValidExp fexp32 :=
  fltValidExp (emin := (-149)) (prec := 24) (by decide)

/--
Round-to-nearest, ties-to-even (binary32-style default rounding).

This is the rounding mode people typically assume when they say “IEEE float32 rounding”.
-/
noncomputable def rnd32 : ℝ → ℤ := neuralNearestEven

/-- `rnd32` is a valid monotone rounding mode in the generic neural-float sense. -/
instance : NeuralValidRnd rnd32 := by
  dsimp [rnd32]
  infer_instance

/-- `rnd32` is round-to-nearest in the `NeuralValidRndToNearest` sense. -/
instance : NeuralValidRndToNearest rnd32 := by
  dsimp [rnd32]
  infer_instance

/--
`FP32`: finite float32 rounding model, as a rounded-`ℝ` value.

This is the type you want if you are proving numerical stability/error bounds without dealing with
NaN/Inf behavior.
-/
abbrev FP32 : Type := NF binaryRadix fexp32 rnd32

namespace FP32

/-- Forgetful projection: treat an `FP32` value as a real number. -/
abbrev toReal (x : FP32) : ℝ := x.val

/--
Convenience constant: the largest finite magnitude representable by the binary32 parameters used by
this model (i.e. roughly `(2 - 2^-23) * 2^127`).

We keep this in `FP32` because it is a useful *proof-level* guard when you want to state “no
overflow” side-conditions in a readable way.
-/
noncomputable def maxFinite : ℝ :=
  (binaryRadix.toReal - NeuralPrecision.machineEpsilon NeuralPrecision.ieee_single) *
    neuralBpow binaryRadix (2^(NeuralPrecision.expBits NeuralPrecision.ieee_single - 1) - 1)

/--
Convenience constant: the smallest positive normal binary32 number (roughly `2^-126`).

Subnormals exist below this; this constant is mainly useful when you want to distinguish
“normal-range” arguments from “subnormal-range” arguments in proofs.
-/
noncomputable def minNormal : ℝ :=
  neuralBpow binaryRadix (-(2^(NeuralPrecision.expBits NeuralPrecision.ieee_single - 1) : ℤ) + 2)

end FP32

end TorchLean.Floats
