/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import FloatLib.Floats.Formats.BinaryInterchange.Rounding.Proof
public import NN.Spec.Core.FloatInstances
import FloatLib.Floats.Formats.BinaryInterchange.Algebraic.Rounding.SqrtProof
import FloatLib.Floats.Formats.BinaryInterchange.Configured.Value.CoreProof

/-!
# Roots of dimension constants

A count can overflow its scalar representation while its square root is representable. The
configured scalar context then uses FloatLib's rational-root algorithm before packing the result.
The refinement below concerns that exceptional branch, for any IEEE descriptor and lossless codec.
Ordinary counts keep the cast-then-root schedule. Native hardware execution is a separate boundary.
-/

@[expose] public section

namespace TorchLean.ConfiguredBinary

open FloatLib.Floats
open FloatLib.Floats.Formats.BinaryInterchange

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
  [ExecFloat.ModelCodec plan (Model format) code]

local notation "Value" => ExecFloat (Configured.Family format code plan)

/-- Recover a representable root without first rounding an overflowing count.

The result is nearest-even rounding of the exact real square root, not the square root of a
saturated or infinite cast. Finiteness is required because the real rounding model has no infinity.
-/
theorem toReal_sqrtNat_of_cast_overflow (n : Nat) (hformat : format.isIEEE = true)
    (hoverflow : (Model.ofRat format (n : Rat) (status := true)).status.overflow = true)
    (hfinite : ExecFloat.Binary.isFinite (Context.sqrtNat (α := Value) n) = true) :
    Model.toReal (ExecFloat.Binary.toModel (Context.sqrtNat (α := Value) n)) =
      Model.roundAt format (Real.sqrt (n : ℝ)) := by
  have hmodel : ExecFloat.Binary.toModel (Context.sqrtNat (α := Value) n) =
      Model.AlgebraicRounding.sqrtRat format false (n : Rat) := by
    simp [Context.sqrtNat, sqrtNat, hoverflow, ExecFloat.Binary.toModel_ofModel]
  change Model.isFinite (ExecFloat.Binary.toModel (Context.sqrtNat (α := Value) n)) = true
    at hfinite
  rw [hmodel] at hfinite ⊢
  simpa using Model.AlgebraicRounding.toReal_sqrtRat_eq_roundAt format hformat false (n : Rat)
    (by positivity) hfinite

end TorchLean.ConfiguredBinary
