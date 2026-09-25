/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.RL.Numerics.Float32.Returns
public import NN.Spec.Core.Tensor.Numerics

/-!
# Checked Float32 TD Residuals, GAE, and Advantage Normalization

This module contains the advantage-estimation pieces that sit on top of checked discounted backups:
TD residuals, fixed-horizon $\operatorname{GAE}(\lambda)$, and z-score normalization. Keeping these
separate from plain
returns makes it clearer which routines are value-learning recurrences and which are PPO pipeline
preprocessing.

References: Sutton and Barto, *Reinforcement Learning: An Introduction*; Schulman et al.,
"High-Dimensional Continuous Control Using Generalized Advantage Estimation" (2015).
-/

@[expose] public section

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.ExecFloat (Binary)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)

namespace Runtime
namespace RL
namespace Numerics
namespace Float32

open Spec TorchLean
open TorchLean TorchLean.Tensor
open Spec.RL


/-!
## Checked value-learning and advantage-estimation building blocks (configured binary32)

These helpers are “PPO-shaped” but still live in `Runtime.RL.Numerics.Float32` because they are
useful as general diagnostics/hardening tools whenever you want an explicit float32 execution
semantics plus checked finiteness.
-/

/--
Checked TD residual / Bellman error:

`r + γ * (1-done) * nextValue - value`.
-/
def tdResidualChecked
    (value reward gamma nextValue : Binary 8 23) (done : Bool) :
    Except String (Binary 8 23) :=
  match discountedBackupChecked (reward := reward) (gamma := gamma)
      (bootstrap := nextValue) (done := done) with
  | .error e => .error e
  | .ok target =>
      match requireFinite "tdResidual/value" value with
      | .error e => .error e
      | .ok _ =>
          checkedSub "tdResidual/sub(target,value)" target value

/--
If `tdResidualChecked` returns `.ok out`, then:

- the checked discounted-backup intermediates are finite,
- the final subtraction intermediate is finite, and
- `out` agrees with the spec-layer `tdResidual` formula.

This is the runtime-checker analogue of `discountedBackup_eq_ok`.
-/
theorem tdResidual_eq_ok
    (value reward gamma nextValue : Binary 8 23) (done : Bool) (out : Binary 8 23)
    (h : tdResidualChecked value reward gamma nextValue done = .ok out) :
    Binary.isFinite
        (ExecFloat.mul gamma (continueMask (α := Binary 8 23) done)) =
      true ∧
      Binary.isFinite
          (ExecFloat.mul
            (ExecFloat.mul gamma (continueMask (α := Binary 8 23) done))
            nextValue) =
        true ∧
        Binary.isFinite
            (ExecFloat.add reward
              (ExecFloat.mul
                (ExecFloat.mul gamma
                  (continueMask (α := Binary 8 23) done))
                nextValue)) =
          true ∧
          Binary.isFinite value = true ∧
          Binary.isFinite
              (ExecFloat.sub
                (discountedBackup (α := Binary 8 23) reward gamma nextValue done) value) =
            true ∧
            out = tdResidual (α := Binary 8 23) value reward gamma nextValue done := by
  -- First, extract the checked discounted-backup call.
  cases htarget : discountedBackupChecked (reward := reward) (gamma := gamma)
      (bootstrap := nextValue) (done := done) with
  | error e =>
      -- Contradiction: the TD residual is an `.error` in this branch.
      have : False := by
        have h' := h
        simp [tdResidualChecked, htarget] at h'
      exact this.elim
  | ok target =>
      -- The `.ok` TD residual means the value finiteness check and the subsequent checked
      -- subtraction both succeeded.
      have hval : Binary.isFinite value = true := by
        cases hf : Binary.isFinite value with
        | true =>
            rfl
        | false =>
            have : False := by
              have h' := h
              simp [tdResidualChecked, htarget, requireFinite, hf] at h'
            exact this.elim

      have hsub :
          checkedSub "tdResidual/sub(target,value)" target value = .ok out := by
        simpa [tdResidualChecked, htarget, requireFinite, hval] using h

      -- Pull out the discounted-backup finiteness hypotheses and spec equality.
      obtain ⟨h₁, h₂, h₃, htargetEq⟩ :=
        discountedBackup_eq_ok
          (reward := reward) (gamma := gamma) (bootstrap := nextValue) (done := done)
          (out := target) htarget

      -- Now handle the checked subtraction.
      set out0 : Binary 8 23 := ExecFloat.sub target value
      have hout0 : Binary.isFinite out0 = true := by
        cases hf : Binary.isFinite out0 with
        | true =>
            rfl
        | false =>
            have : False := by
              have h' := hsub
              simp [checkedSub, requireFinite, out0, hf] at h'
            exact this.elim

      have hout : out = out0 := by
        have : checkedSub "tdResidual/sub(target,value)" target value = .ok out0 := by
          simp [checkedSub, requireFinite, out0, hout0]
        have hok : (Except.ok out : Except String (Binary 8 23)) = Except.ok out0 := by
          exact hsub.symm.trans this
        have : out = out0 := by
          injection hok
        exact this

      -- Assemble the final statement.
      refine ⟨h₁, h₂, h₃, hval, ?_, ?_⟩
      · -- subtraction finiteness, rewritten to the spec-layer target expression
        simpa [out0, htargetEq] using hout0
      · -- returned value equals the spec TD residual
        -- `out0` is definitionally `target - value`.
        -- Rewrite `target` to the spec discounted backup and unfold `tdResidual`.
        rw [hout]
        simp [out0, htargetEq, Spec.RL.tdResidual, Spec.RL.tdTarget,
          HSub.hSub, Sub.sub]

/--
Checked fixed-horizon Generalized Advantage Estimation ($\operatorname{GAE}(\lambda)$), specialized
to `Binary 8 23`.

This is the checked/finite counterpart to `Runtime.RL.Core.generalizedAdvantageEstimation`.

Reference:
- Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation"
  (2015): https://arxiv.org/abs/1506.02438
-/
def generalizedAdvantageEstimationChecked {n : Nat}
    (gamma lam : Binary 8 23)
    (rewards values nextValues : Tensor (Binary 8 23) [n])
    (dones : Tensor Bool [n]) :
    Except String (Tensor (Binary 8 23) [n]) := do
  let indices : Tensor (Fin n) [n] := Tensor.ofFn id
  Tensor.scanrM (fun idx advNext => do
    let done := dones[idx]
    let mask : Binary 8 23 := continueMask (α := Binary 8 23) done
    -- delta = r + γ * mask * nextValue - value
    let t1 ← checkedMul "gae/mul(gamma,mask)" gamma mask
    let t2 ← checkedMul "gae/mul(t1,nextValue)" t1 nextValues[idx]
    let t3 ← checkedAdd "gae/add(reward,t2)" rewards[idx] t2
    let delta ← checkedSub "gae/sub(t3,value)" t3 values[idx]
    -- adv = delta + γ * λ * mask * advNext
    let u1 ← checkedMul "gae/mul(gamma,lam)" gamma lam
    let u2 ← checkedMul "gae/mul(u1,mask)" u1 mask
    let u3 ← checkedMul "gae/mul(u2,advNext)" u2 advNext
    let adv ← checkedAdd "gae/add(delta,u3)" delta u3
    pure adv) 0 indices

/--
Checked z-score normalization (mean-center then divide by standard deviation), specialized to
`Binary 8 23`.

This is used by PPO to normalize advantages.

Implementation note:
we reuse `Spec.normalizeZscoreSpec` for the math, but additionally enforce that all outputs are
finite. If the computed standard deviation is zero, `normalizeZscoreSpec` returns the centered
vector, which is still validated for finiteness here.
-/
def normalizeZScoreChecked {n : Nat}
    (x : Tensor (Binary 8 23) [n]) :
    Except String (Tensor (Binary 8 23) [n]) := do
  let y : Tensor (Binary 8 23) [n] :=
    Spec.normalizeZscoreSpec (α := Binary 8 23) (n := n) x
  if Boundary.tensorAll (α := Binary 8 23) (s := .dim n .scalar)
      (fun z => Binary.isFinite z) y then
    .ok y
  else
    .error "RL float32: normalizeZScore produced a non-finite entry."


end Float32
end Numerics
end RL
end Runtime
