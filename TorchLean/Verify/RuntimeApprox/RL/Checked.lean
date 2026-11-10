/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.RuntimeApprox.RL.Backup
public import TorchLean.NN.RL.Numerics.Advantage

/-!
# Rounding Chains of the Checked RL Runtime

`TorchLean.Verify.RuntimeApprox.RL.Backup` proves the rounding chains of the discounted backup and
the TD residual under one `isFinite … = true` hypothesis per intermediate. The checked runtime
`TorchLean.RL.Checked` fails as soon as an intermediate is not finite, so a run that returns `.ok`
discharges those hypotheses. This module states both chains for a successful run in any configured
IEEE format, binary32 `ExecFloat.Binary 8 23` among them.

## Main declarations

* `toReal_of_discountedBackup`
* `toReal_of_tdResidual`

References:

- IEEE 754-2019 (binary floating-point arithmetic): https://doi.org/10.1109/IEEESTD.2019.8766229
- Goldberg, “What Every Computer Scientist Should Know About Floating-Point Arithmetic” (1991):
  https://doi.org/10.1145/103162.103163
-/

@[expose] public section

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)

namespace TorchLean
namespace RL
namespace Checked

open FloatLib.Floats.Formats.BinaryInterchange

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
    [ExecFloat.ModelCodec plan (Model format) code]

local notation "Value" => ExecFloat (Configured.Family format code plan)

/--
If `RL.Checked.discountedBackup` returns `.ok out` in an IEEE format, then the real
value of `out` is the reward's when `done`, and otherwise that of `reward + gamma * bootstrap` with
the product and the sum each rounded by `Model.roundAt format`.
-/
theorem toReal_of_discountedBackup
    [ExecFloat.Backend.PolicyFor (Configured.Family format code plan)]
    [addCap : ExecFloat.Add (Configured.Family format code plan)]
    [ExecFloat.Capability.HasSpec addCap Configured.Spec.add]
    [mulCap : ExecFloat.Mul (Configured.Family format code plan)]
    [ExecFloat.Capability.HasSpec mulCap Configured.Spec.mul]
    (reward gamma bootstrap : Value) (done : Bool)
    (out : Value)
    (h : RL.Checked.discountedBackup reward gamma bootstrap done
      = .ok out)
    (hformat : format.isIEEE = true := by ieee_format) :
    (ExecFloat.Binary.toModel out).toReal =
      if done then (ExecFloat.Binary.toModel reward).toReal
      else
        Model.roundAt format
          ((ExecFloat.Binary.toModel reward).toReal +
            Model.roundAt format
              ((ExecFloat.Binary.toModel gamma).toReal *
                (ExecFloat.Binary.toModel bootstrap).toReal)) := by
  obtain ⟨hfinite, hout⟩ :=
    RL.Checked.discountedBackup_eq_ok
      (reward := reward) (gamma := gamma) (bootstrap := bootstrap) (done := done) (out := out) h
  rw [hout]
  exact toReal_discountedBackup reward gamma bootstrap done
    (fun hd => by subst hd; exact (by simpa using hfinite : _ ∧ _).1)
    (fun hd => by subst hd; exact (by simpa using hfinite : _ ∧ _).2) hformat

/--
If `RL.Checked.tdResidual` returns `.ok out` in an IEEE format, then the real value
of `out` is `Model.roundAt format (b - value)`. Here `b` is the reward when `done`, and otherwise
`reward + gamma * nextValue` with the product and the sum each rounded by `Model.roundAt format`.
-/
theorem toReal_of_tdResidual
    [ExecFloat.Backend.PolicyFor (Configured.Family format code plan)]
    [addCap : ExecFloat.Add (Configured.Family format code plan)]
    [ExecFloat.Capability.HasSpec addCap Configured.Spec.add]
    [mulCap : ExecFloat.Mul (Configured.Family format code plan)]
    [ExecFloat.Capability.HasSpec mulCap Configured.Spec.mul]
    [subCap : ExecFloat.Sub (Configured.Family format code plan)]
    [ExecFloat.Capability.HasSpec subCap Configured.Spec.sub]
    (value reward gamma nextValue : Value) (done : Bool)
    (out : Value)
    (h : RL.Checked.tdResidual value reward gamma nextValue done
      = .ok out)
    (hformat : format.isIEEE = true := by ieee_format) :
    (ExecFloat.Binary.toModel out).toReal =
      Model.roundAt format
        ((if done then (ExecFloat.Binary.toModel reward).toReal
          else
            Model.roundAt format
              ((ExecFloat.Binary.toModel reward).toReal +
                Model.roundAt format
                  ((ExecFloat.Binary.toModel gamma).toReal *
                    (ExecFloat.Binary.toModel nextValue).toReal))) -
          (ExecFloat.Binary.toModel value).toReal) := by
  obtain ⟨hfinite, -, hsub, hout⟩ :=
    RL.Checked.tdResidual_eq_ok
      (value := value) (reward := reward) (gamma := gamma) (nextValue := nextValue) (done := done)
      (out := out) h
  rw [hout]
  exact toReal_tdResidual value reward gamma nextValue done
    (fun hd => by subst hd; exact (by simpa using hfinite : _ ∧ _).1)
    (fun hd => by subst hd; exact (by simpa using hfinite : _ ∧ _).2) hsub hformat

end Checked
end RL
end TorchLean
