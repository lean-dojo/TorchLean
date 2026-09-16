/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.RL.Core
public import NN.Proofs.RuntimeApprox.IEEE32.Arithmetic

/-!
# RL Float32 Semantics (IEEE32Exec)

TorchLean provides multiple “views” of float32:

- `ExecFloat.Binary 8 23`: executable, bit-level IEEE-754 binary32 (can run inside Lean),
- `FP32`: proof-oriented “round-on-$\mathbb R$” float32 model (finite-only).

The IEEE32Exec bridge files prove that, on the **finite path**, executable float32 arithmetic
refines the standard mathematical model: compute the real operation and round to float32 at each
primitive operation.

This module packages that theorem pattern for one of the most common RL formulas: the one-step
discounted backup and TD residual used by TD learning, value iteration, and advantage estimation.

Practical takeaway:

If your runtime code checks that the relevant IEEE32Exec intermediates are finite (no NaN/Inf),
then you can immediately “upgrade” that checked fact into a clean `FP32`-style real-rounding
semantics for reasoning and error analysis.

References:
- IEEE 754-2019 (binary32 arithmetic): https://doi.org/10.1109/IEEESTD.2019.8766229
- Goldberg, “What Every Computer Scientist Should Know About Floating-Point Arithmetic” (1991):
  https://doi.org/10.1145/103162.103163
- Sutton and Barto, *Reinforcement Learning: An Introduction* (2nd ed., discounted backups and
  TD learning): http://incompleteideas.net/book/the-book-2nd.html
-/

@[expose] public section

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)


namespace Proofs
namespace RL
namespace Float32Exec

open Spec.RL

open TorchLean.Floats
open TorchLean.Floats.IEEE754

open TorchLean.Floats.IEEE754.IEEE32Exec

/--
Refinement theorem for the RL one-step discounted backup in executable float32 semantics.

Assuming the relevant IEEE32Exec intermediates are finite, the decoded real value of the backup
agrees with the standard “real op + round-to-float32” model (`fp32Round`) at each primitive op.

This is a useful building block for connecting executable RL code (IEEE32Exec) to textbook-style
reasoning and error bounds phrased over `ℝ` + rounding.
-/
theorem toReal_discountedBackup_eq_fp32Round_chain_of_isFinite
    (reward gamma bootstrap : ExecFloat.Binary 8 23) (done : Bool)
    (h₁ : ExecFloat.Binary.isFinite (ExecFloat.mul gamma (continueMask (α := (ExecFloat.Binary 8
      23)) done)) = true)
    (h₂ : ExecFloat.Binary.isFinite (ExecFloat.mul (ExecFloat.mul gamma (continueMask (α :=
      (ExecFloat.Binary 8 23)) done)) bootstrap) = true)
    (h₃ :
      ExecFloat.Binary.isFinite (ExecFloat.add reward (ExecFloat.mul (ExecFloat.mul gamma
        (continueMask (α := (ExecFloat.Binary 8 23)) done)) bootstrap))
        = true) :
    (ExecFloat.Binary.toModel (discountedBackup (α := (ExecFloat.Binary 8 23)) reward gamma
      bootstrap done)).toReal =
      fp32Round
        ((ExecFloat.Binary.toModel reward).toReal +
          fp32Round
            (fp32Round ((ExecFloat.Binary.toModel gamma).toReal * (ExecFloat.Binary.toModel
              (continueMask (α := (ExecFloat.Binary 8 23)) done)).toReal) *
              (ExecFloat.Binary.toModel bootstrap).toReal)) := by
  change (ExecFloat.Binary.toModel (ExecFloat.add reward (ExecFloat.mul (ExecFloat.mul gamma
    (continueMask (α := (ExecFloat.Binary 8 23)) done)) bootstrap))).toReal = _
  rw [toReal_add_eq_fp32Round_of_isFinite h₃,
    toReal_mul_eq_fp32Round_of_isFinite h₂, toReal_mul_eq_fp32Round_of_isFinite h₁]

/--
Refinement theorem for the TD residual / Bellman error in executable float32 semantics.

Formula:
`r + γ * (1-done) * nextValue - value`.

Assuming the relevant IEEE32Exec intermediates are finite, the decoded real value of the TD
residual agrees with the standard “real op + round-to-float32” model (`fp32Round`) at each
primitive operation.
-/
theorem toReal_tdResidual_eq_fp32Round_chain_of_isFinite
    (value reward gamma nextValue : ExecFloat.Binary 8 23) (done : Bool)
    (h₁ : ExecFloat.Binary.isFinite (ExecFloat.mul gamma (continueMask (α := (ExecFloat.Binary 8
      23)) done)) = true)
    (h₂ : ExecFloat.Binary.isFinite (ExecFloat.mul (ExecFloat.mul gamma (continueMask (α :=
      (ExecFloat.Binary 8 23)) done)) nextValue) = true)
    (h₃ :
      ExecFloat.Binary.isFinite (ExecFloat.add reward (ExecFloat.mul (ExecFloat.mul gamma
        (continueMask (α := (ExecFloat.Binary 8 23)) done)) nextValue))
        = true)
    (hval : ExecFloat.Binary.isFinite value = true)
    (hsub :
      ExecFloat.Binary.isFinite (ExecFloat.sub (discountedBackup (α := (ExecFloat.Binary 8 23))
        reward gamma nextValue done) value)
        = true) :
    (ExecFloat.Binary.toModel (tdResidual (α := (ExecFloat.Binary 8 23)) value reward gamma
      nextValue done)).toReal =
      fp32Round
        (fp32Round
            ((ExecFloat.Binary.toModel reward).toReal +
              fp32Round
                (fp32Round ((ExecFloat.Binary.toModel gamma).toReal * (ExecFloat.Binary.toModel
                  (continueMask (α := (ExecFloat.Binary 8 23)) done)).toReal) *
                  (ExecFloat.Binary.toModel nextValue).toReal)) -
          (ExecFloat.Binary.toModel value).toReal) := by
  -- First, refine the discounted backup part.
  have hbackup :
      (ExecFloat.Binary.toModel (discountedBackup (α := (ExecFloat.Binary 8 23)) reward gamma
        nextValue done)).toReal =
        fp32Round
          ((ExecFloat.Binary.toModel reward).toReal +
            fp32Round
              (fp32Round ((ExecFloat.Binary.toModel gamma).toReal * (ExecFloat.Binary.toModel
                (continueMask (α := (ExecFloat.Binary 8 23)) done)).toReal) *
                (ExecFloat.Binary.toModel nextValue).toReal)) :=
    toReal_discountedBackup_eq_fp32Round_chain_of_isFinite
      (reward := reward) (gamma := gamma) (bootstrap := nextValue) (done := done)
      (h₁ := h₁) (h₂ := h₂) (h₃ := h₃)

  -- Then apply the subtraction refinement for the final TD residual step.
  have hbackupFin :
      ExecFloat.Binary.isFinite (discountedBackup (α := (ExecFloat.Binary 8 23)) reward gamma
        nextValue done) = true := by
    exact h₃
  have hsubReal :
      (ExecFloat.Binary.toModel (ExecFloat.sub (discountedBackup (α := (ExecFloat.Binary 8 23))
        reward gamma nextValue done) value)).toReal =
        fp32Round
          ((ExecFloat.Binary.toModel (discountedBackup (α := (ExecFloat.Binary 8 23)) reward gamma
            nextValue done)).toReal -
            (ExecFloat.Binary.toModel value).toReal) :=
    toReal_sub_eq_fp32Round_of_isFinite
      (x := discountedBackup (α := (ExecFloat.Binary 8 23)) reward gamma nextValue done) (y :=
        value)
      hbackupFin hval hsub

  -- Unfold the RL definition and substitute the refined discounted-backup real meaning.
  -- `tdResidual = tdTarget - value` and `tdTarget = discountedBackup`.
  change (ExecFloat.Binary.toModel (ExecFloat.sub (discountedBackup reward gamma nextValue done)
    value)).toReal = _
  rw [hsubReal, hbackup]

end Float32Exec
end RL
end Proofs
