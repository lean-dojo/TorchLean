/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.RL.PolicyGradient
public import TorchLean.Tensor.ConfiguredBinary
public import FloatLib.Floats.Formats.BinaryInterchange.Configured.CorrectlyRounded
public import FloatLib.Floats.Formats.BinaryInterchange.Configured.MixedError.Rounding

/-!
# PPO Ratio At Unchanged Parameters

At an IEEE configured format the importance ratio of a finite log-probability with itself decodes
to exactly one: `x - x` is zero and the correctly rounded exponential of zero is one. At unchanged
parameters PPO's ratio has this form when the objective's interpreter recomputes bit for bit the
value `PPO.collect` cached. `PPO.Autograd.objective` says which interpreters do.
-/

@[expose] public section

open FloatLib.Floats
open FloatLib.Floats.Formats.BinaryInterchange

namespace TorchLean
namespace RL
namespace PolicyGradient

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
    [ExecFloat.ModelCodec plan (Model format) code]

local notation "Value" => ExecFloat (Configured.Family format code plan)

/-- At an IEEE configured format, the importance ratio of a finite log-probability with itself
decodes to exactly one, for instance of a cached `PolicyGradient.logProb logits action` that the
objective recomputes bit for bit. -/
theorem toReal_ratio_self (hformat : format.isIEEE = true) {x : Value}
    (hx : ExecFloat.Binary.isFinite x = true) :
    ExecFloat.Binary.toReal (RL.PolicyGradient.ratio (α := Value) x x) = 1 := by
  unfold RL.PolicyGradient.ratio
  rw [ConfiguredBinary.exp_of_isIEEE hformat]
  have hfinite : ExecFloat.Binary.isFinite (x - x) = true :=
    ExecFloat.Binary.isFinite_sub_self hx hformat
  have hzero : ExecFloat.Binary.toReal (x - x) = 0 := ExecFloat.Binary.toReal_sub_self hx hformat
  rw [ExecFloat.Binary.CorrectlyRounded.toReal_exp_eq_roundAt hfinite
      (by rw [hzero, Real.exp_zero]
          exact Model.Transcendentals.CorrectlyRounded.one_lt_overflowThreshold hformat)
      hformat,
    hzero, Real.exp_zero, Model.roundAt_one]

end PolicyGradient
end RL
end TorchLean
