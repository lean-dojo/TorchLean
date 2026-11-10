/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import FloatLib.Floats.Formats.BinaryInterchange.Rounding.Proof
public import TorchLean.Tensor.ConfiguredBinary
import FloatLib.Floats.Formats.BinaryInterchange.Algebraic.Rounding.SqrtProof
import FloatLib.Floats.Formats.BinaryInterchange.Configured.Value.CoreProof

/-!
# Roots of dimension constants

A count can overflow its scalar format while its square root is representable. In that case the
configured scalar context computes the root with FloatLib's rational-root algorithm and packs the
result, and ordinary counts keep the cast-then-root schedule. `toReal_sqrtNat_of_overflow`
covers that overflow branch for any IEEE descriptor and lossless codec. Native hardware execution
is outside its scope.
-/

@[expose] public section

namespace TorchLean.ConfiguredBinary

open FloatLib.Floats
open FloatLib.Floats.Formats.BinaryInterchange

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
  [ExecFloat.ModelCodec plan (Model format) code]

local notation "Value" => ExecFloat (Configured.Family format code plan)

/-- For an IEEE format in which the cast of `n` overflows, a finite `sqrtNat n` has the real value
`√n` rounded once to nearest even. No saturated or infinite cast enters the root.

Finiteness is a hypothesis because the real rounding model has no infinity.
-/
theorem toReal_sqrtNat_of_overflow (n : Nat) (hformat : format.isIEEE = true)
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
