/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.RL
public import TorchLean.Verify.Floats.IEEE754

/-!
# RL Backups in Executable IEEE Semantics

Two of TorchLean's views of floating point meet here:

- `ExecFloat (Configured.Family format code plan)`: executable, bit-level IEEE-754 values of a
  configured binary format, such as binary32 `ExecFloat.Binary 8 23` (can run inside Lean),
- `FP32`: the proof-oriented binary32 model that rounds each real result and has no infinities,
  NaNs or upper exponent bound.

FloatLib's finite rounding equalities, such as `ExecFloat.Binary.toReal_add_eq_roundAt`, prove
that, on the **finite path**, executable arithmetic refines the standard mathematical model:
compute the real operation and round to the format at each primitive operation.

The theorems here apply that pattern to the one-step discounted backup and the TD residual, two RL
formulas that TD learning, value iteration and advantage estimation share. When runtime code has
checked that the relevant executable intermediates are finite (no NaN or infinity), the checked
fact yields a real-rounding semantics for reasoning and error analysis, the `FP32`-style model in
the binary32 case.

References:
- IEEE 754-2019 (binary floating-point arithmetic): https://doi.org/10.1109/IEEESTD.2019.8766229
- Goldberg, “What Every Computer Scientist Should Know About Floating-Point Arithmetic” (1991):
  https://doi.org/10.1145/103162.103163
- Sutton and Barto, *Reinforcement Learning: An Introduction* (2nd ed., discounted backups and
  TD learning): http://incompleteideas.net/book/the-book-2nd.html
-/

@[expose] public section

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)

namespace TorchLean
namespace RL

open FloatLib.Floats.Formats.BinaryInterchange

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
    [ExecFloat.ModelCodec plan (Model format) code]

local notation "Value" => ExecFloat (Configured.Family format code plan)

/-- The decoded one-step discounted backup is the reward at a terminal step and otherwise, when
the product `gamma * bootstrap` and the sum are finite in an IEEE format, the real product and
sum each rounded once by `Model.roundAt format`. -/
theorem toReal_discountedBackup
    [ExecFloat.Backend.PolicyFor (Configured.Family format code plan)]
    [addCap : ExecFloat.Add (Configured.Family format code plan)]
    [ExecFloat.Capability.HasSpec addCap Configured.Spec.add]
    [mulCap : ExecFloat.Mul (Configured.Family format code plan)]
    [ExecFloat.Capability.HasSpec mulCap Configured.Spec.mul]
    (reward gamma bootstrap : Value) (done : Bool)
    (h₁ : done = false → ExecFloat.Binary.isFinite (ExecFloat.mul gamma bootstrap) = true)
    (h₂ : done = false →
      ExecFloat.Binary.isFinite (ExecFloat.add reward (ExecFloat.mul gamma bootstrap)) = true)
    (hformat : format.isIEEE = true := by ieee_format) :
    (ExecFloat.Binary.toModel (discountedBackup (α := Value) reward gamma bootstrap
      done)).toReal =
      if done then (ExecFloat.Binary.toModel reward).toReal
      else
        Model.roundAt format
          ((ExecFloat.Binary.toModel reward).toReal +
            Model.roundAt format
              ((ExecFloat.Binary.toModel gamma).toReal *
                (ExecFloat.Binary.toModel bootstrap).toReal)) := by
  cases done with
  | true => rfl
  | false =>
      change ExecFloat.Binary.toReal (reward + gamma * bootstrap) = _
      rw [ExecFloat.Binary.toReal_add_eq_roundAt (left := reward) (right := gamma * bootstrap)
          (h₂ rfl) hformat,
        ExecFloat.Binary.toReal_mul_eq_roundAt (left := gamma) (right := bootstrap)
          (h₁ rfl) hformat]
      rfl

/--
Refinement theorem for the TD residual / Bellman error in executable IEEE semantics:
`discountedBackup reward gamma nextValue done - value`.

When the computed intermediates are finite, the decoded real value of the TD residual is the real
computation with each primitive result rounded by `Model.roundAt format`. At a terminal step the
backup inside the residual is the reward itself.
-/
theorem toReal_tdResidual
    [ExecFloat.Backend.PolicyFor (Configured.Family format code plan)]
    [addCap : ExecFloat.Add (Configured.Family format code plan)]
    [ExecFloat.Capability.HasSpec addCap Configured.Spec.add]
    [mulCap : ExecFloat.Mul (Configured.Family format code plan)]
    [ExecFloat.Capability.HasSpec mulCap Configured.Spec.mul]
    [subCap : ExecFloat.Sub (Configured.Family format code plan)]
    [ExecFloat.Capability.HasSpec subCap Configured.Spec.sub]
    (value reward gamma nextValue : Value) (done : Bool)
    (h₁ : done = false → ExecFloat.Binary.isFinite (ExecFloat.mul gamma nextValue) = true)
    (h₂ : done = false →
      ExecFloat.Binary.isFinite (ExecFloat.add reward (ExecFloat.mul gamma nextValue)) = true)
    (hsub :
      ExecFloat.Binary.isFinite (ExecFloat.sub (discountedBackup (α := Value)
        reward gamma nextValue done) value)
        = true)
    (hformat : format.isIEEE = true := by ieee_format) :
    (ExecFloat.Binary.toModel (tdResidual (α := Value) value reward gamma
      nextValue done)).toReal =
      Model.roundAt format
        ((if done then (ExecFloat.Binary.toModel reward).toReal
          else
            Model.roundAt format
              ((ExecFloat.Binary.toModel reward).toReal +
                Model.roundAt format
                  ((ExecFloat.Binary.toModel gamma).toReal *
                    (ExecFloat.Binary.toModel nextValue).toReal))) -
          (ExecFloat.Binary.toModel value).toReal) := by
  change ExecFloat.Binary.toReal (discountedBackup reward gamma nextValue done - value) = _
  rw [ExecFloat.Binary.toReal_sub_eq_roundAt hsub hformat]
  exact congrArg (fun r => Model.roundAt format (r - ExecFloat.Binary.toReal value))
    (toReal_discountedBackup reward gamma nextValue done h₁ h₂ hformat)

end RL
end TorchLean
