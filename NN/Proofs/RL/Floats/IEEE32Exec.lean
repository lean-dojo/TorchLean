/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.RL.Core
public import NN.Floats.IEEEExec.BridgeFP32Total
public import NN.Proofs.RL.Tactics

/-!
# RL Float32 Semantics (IEEE32Exec)

TorchLean provides multiple “views” of float32:

- `IEEE32Exec`: executable, bit-level IEEE-754 binary32 (can run inside Lean),
- `FP32`: proof-oriented “round-on-ℝ” float32 model (finite-only).

The IEEE32Exec bridge files prove that, on the **finite path**, executable float32 arithmetic
refines the standard mathematical model: compute the real operation and round to float32 at each
primitive operation.

This module packages that theorem pattern for one of the most common RL formulas: the one-step discounted
backup and TD residual used by TD learning, value iteration, and advantage estimation.

Practical takeaway:

If your runtime code checks that the relevant IEEE32Exec intermediates are finite (no NaN/Inf),
then you can immediately “upgrade” that checked fact into a clean `FP32`-style real-rounding
semantics for reasoning and error analysis.

References:
- IEEE 754-2019 (binary32 arithmetic): https://doi.org/10.1109/IEEESTD.2019.8766229
- Goldberg, “What Every Computer Scientist Should Know About Floating-Point Arithmetic” (1991):
  https://doi.org/10.1145/103162.103163
- Sutton and Barto, *Reinforcement Learning: An Introduction* (2nd ed., discounted backups/TD learning):
  http://incompleteideas.net/book/the-book-2nd.html
-/

@[expose] public section

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
    (reward gamma bootstrap : IEEE32Exec) (done : Bool)
    (h₁ : isFinite (mul gamma (continueMask (α := IEEE32Exec) done)) = true)
    (h₂ : isFinite (mul (mul gamma (continueMask (α := IEEE32Exec) done)) bootstrap) = true)
    (h₃ :
      isFinite (add reward (mul (mul gamma (continueMask (α := IEEE32Exec) done)) bootstrap)) = true) :
    toReal (discountedBackup (α := IEEE32Exec) reward gamma bootstrap done) =
      fp32Round
        (toReal reward +
          fp32Round (fp32Round (toReal gamma * toReal (continueMask (α := IEEE32Exec) done)) * toReal bootstrap)) := by
  -- Unfold the RL definition and apply the per-op bridge lemmas (`*_of_isFinite`) from BridgeFP32Total.
  simp_rl
  set mask : IEEE32Exec := continueMask (α := IEEE32Exec) done
  have ht1 :
      toReal (mul gamma mask) = fp32Round (toReal gamma * toReal mask) :=
    toReal_mul_eq_fp32Round_of_isFinite (x := gamma) (y := mask) h₁
  have ht2 :
      toReal (mul (mul gamma mask) bootstrap) =
        fp32Round (toReal (mul gamma mask) * toReal bootstrap) :=
    toReal_mul_eq_fp32Round_of_isFinite (x := (mul gamma mask)) (y := bootstrap) h₂
  have hout :
      toReal (add reward (mul (mul gamma mask) bootstrap)) =
        fp32Round (toReal reward + toReal (mul (mul gamma mask) bootstrap)) :=
    toReal_add_eq_fp32Round_of_isFinite (x := reward) (y := (mul (mul gamma mask) bootstrap)) h₃
  -- Substitute the intermediate refinement equations to expose the nested rounding structure.
  simpa [mask, ht1, ht2] using hout

/--
Refinement theorem for the TD residual / Bellman error in executable float32 semantics.

Formula:
`r + γ * (1-done) * nextValue - value`.

Assuming the relevant IEEE32Exec intermediates are finite, the decoded real value of the TD
residual agrees with the standard “real op + round-to-float32” model (`fp32Round`) at each
primitive operation.
-/
theorem toReal_tdResidual_eq_fp32Round_chain_of_isFinite
    (value reward gamma nextValue : IEEE32Exec) (done : Bool)
    (h₁ : isFinite (mul gamma (continueMask (α := IEEE32Exec) done)) = true)
    (h₂ : isFinite (mul (mul gamma (continueMask (α := IEEE32Exec) done)) nextValue) = true)
    (h₃ :
      isFinite (add reward (mul (mul gamma (continueMask (α := IEEE32Exec) done)) nextValue)) = true)
    (hval : isFinite value = true)
    (hsub :
      isFinite (sub (discountedBackup (α := IEEE32Exec) reward gamma nextValue done) value) = true) :
    toReal (tdResidual (α := IEEE32Exec) value reward gamma nextValue done) =
      fp32Round
        (fp32Round
            (toReal reward +
              fp32Round
                (fp32Round (toReal gamma * toReal (continueMask (α := IEEE32Exec) done)) *
                  toReal nextValue)) -
          toReal value) := by
  -- First, refine the discounted backup part.
  have hbackup :
      toReal (discountedBackup (α := IEEE32Exec) reward gamma nextValue done) =
        fp32Round
          (toReal reward +
            fp32Round
              (fp32Round (toReal gamma * toReal (continueMask (α := IEEE32Exec) done)) *
                toReal nextValue)) :=
    toReal_discountedBackup_eq_fp32Round_chain_of_isFinite
      (reward := reward) (gamma := gamma) (bootstrap := nextValue) (done := done)
      (h₁ := h₁) (h₂ := h₂) (h₃ := h₃)

  -- Then apply the subtraction refinement for the final TD residual step.
  have hbackupFin :
      isFinite (discountedBackup (α := IEEE32Exec) reward gamma nextValue done) = true := by
    simpa [discountedBackup] using h₃
  have hsubReal :
      toReal (sub (discountedBackup (α := IEEE32Exec) reward gamma nextValue done) value) =
        fp32Round
          (toReal (discountedBackup (α := IEEE32Exec) reward gamma nextValue done) - toReal value) :=
    toReal_sub_eq_fp32Round_of_isFinite
      (x := discountedBackup (α := IEEE32Exec) reward gamma nextValue done) (y := value)
      hbackupFin hval hsub

  -- Unfold the RL definition and substitute the refined discounted-backup real meaning.
  -- `tdResidual = tdTarget - value` and `tdTarget = discountedBackup`.
  simpa [Spec.RL.tdResidual, Spec.RL.tdTarget, hbackup, sub_eq_add_neg]
    using hsubReal

end Float32Exec
end RL
end Proofs
