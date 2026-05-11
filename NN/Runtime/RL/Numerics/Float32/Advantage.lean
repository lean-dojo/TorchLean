/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.RL.Numerics.Float32.Returns

/-!
# Checked Float32 TD Residuals, GAE, and Advantage Normalization

This module contains the advantage-estimation pieces that sit on top of checked discounted backups:
TD residuals, fixed-horizon GAE(λ), and z-score normalization. Keeping these separate from plain
returns makes it clearer which routines are value-learning recurrences and which are PPO pipeline
preprocessing.

References: Sutton and Barto, *Reinforcement Learning: An Introduction*; Schulman et al.,
"High-Dimensional Continuous Control Using Generalized Advantage Estimation" (2015).
-/

@[expose] public section

namespace Runtime
namespace RL
namespace Numerics
namespace Float32

open Spec
open Tensor
open Spec.RL

open TorchLean.Floats
open TorchLean.Floats.IEEE754

/-!
## Checked value-learning and advantage-estimation building blocks (IEEE32Exec)

These helpers are “PPO-shaped” but still live in `Runtime.RL.Numerics.Float32` because they are useful as
general diagnostics/hardening tools whenever you want an explicit float32 execution semantics plus
checked finiteness.
-/

/--
Checked TD residual / Bellman error:

`r + γ * (1-done) * nextValue - value`.
-/
def tdResidualIEEE32ExecChecked
    (value reward gamma nextValue : Float32Exec) (done : Bool) :
    Except String Float32Exec :=
  match discountedBackupIEEE32ExecChecked (reward := reward) (gamma := gamma)
      (bootstrap := nextValue) (done := done) with
  | .error e => .error e
  | .ok target =>
      match requireFinite "tdResidual/value" value with
      | .error e => .error e
      | .ok _ =>
          checkedSub "tdResidual/sub(target,value)" target value

/--
If `tdResidualIEEE32ExecChecked` returns `.ok out`, then:

- the checked discounted-backup intermediates are finite,
- the final subtraction intermediate is finite, and
- `out` agrees with the spec-layer `tdResidual` formula.

This is the runtime-checker analogue of `discountedBackupIEEE32ExecChecked_eq_ok`.
-/
theorem tdResidualIEEE32ExecChecked_eq_ok
    (value reward gamma nextValue : Float32Exec) (done : Bool) (out : Float32Exec)
    (h : tdResidualIEEE32ExecChecked value reward gamma nextValue done = .ok out) :
    TorchLean.Floats.IEEE754.IEEE32Exec.isFinite
        (TorchLean.Floats.IEEE754.IEEE32Exec.mul gamma (continueMask (α := Float32Exec) done)) =
      true ∧
      TorchLean.Floats.IEEE754.IEEE32Exec.isFinite
          (TorchLean.Floats.IEEE754.IEEE32Exec.mul
            (TorchLean.Floats.IEEE754.IEEE32Exec.mul gamma (continueMask (α := Float32Exec) done))
            nextValue) =
        true ∧
        TorchLean.Floats.IEEE754.IEEE32Exec.isFinite
            (TorchLean.Floats.IEEE754.IEEE32Exec.add reward
              (TorchLean.Floats.IEEE754.IEEE32Exec.mul
                (TorchLean.Floats.IEEE754.IEEE32Exec.mul gamma
                  (continueMask (α := Float32Exec) done))
                nextValue)) =
          true ∧
          TorchLean.Floats.IEEE754.IEEE32Exec.isFinite value = true ∧
          TorchLean.Floats.IEEE754.IEEE32Exec.isFinite
              (TorchLean.Floats.IEEE754.IEEE32Exec.sub
                (discountedBackup (α := Float32Exec) reward gamma nextValue done) value) =
            true ∧
            out = tdResidual (α := Float32Exec) value reward gamma nextValue done := by
  -- First, extract the checked discounted-backup call.
  cases htarget : discountedBackupIEEE32ExecChecked (reward := reward) (gamma := gamma)
      (bootstrap := nextValue) (done := done) with
  | error e =>
      -- Contradiction: the TD residual is an `.error` in this branch.
      have : False := by
        have h' := h
        simp [tdResidualIEEE32ExecChecked, htarget] at h'
      exact this.elim
  | ok target =>
      -- The `.ok` TD residual means the value finiteness check and subsequent checked subtraction succeeded.
      have hval : TorchLean.Floats.IEEE754.IEEE32Exec.isFinite value = true := by
        cases hf : TorchLean.Floats.IEEE754.IEEE32Exec.isFinite value with
        | true =>
            rfl
        | false =>
            have : False := by
              have h' := h
              simp [tdResidualIEEE32ExecChecked, htarget, requireFinite, hf] at h'
            exact this.elim

      have hsub :
          checkedSub "tdResidual/sub(target,value)" target value = .ok out := by
        simpa [tdResidualIEEE32ExecChecked, htarget, requireFinite, hval] using h

      -- Pull out the discounted-backup finiteness hypotheses and spec equality.
      obtain ⟨h₁, h₂, h₃, htargetEq⟩ :=
        discountedBackupIEEE32ExecChecked_eq_ok
          (reward := reward) (gamma := gamma) (bootstrap := nextValue) (done := done) (out := target)
          htarget

      -- Now handle the checked subtraction.
      set out0 : Float32Exec := TorchLean.Floats.IEEE754.IEEE32Exec.sub target value
      have hout0 : TorchLean.Floats.IEEE754.IEEE32Exec.isFinite out0 = true := by
        cases hf : TorchLean.Floats.IEEE754.IEEE32Exec.isFinite out0 with
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
        have hok : (Except.ok out : Except String Float32Exec) = Except.ok out0 := by
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
        simp [out0, htargetEq, Spec.RL.tdResidual, Spec.RL.tdTarget, HSub.hSub, Sub.sub]

/--
Checked fixed-horizon Generalized Advantage Estimation (GAE(λ)), specialized to `IEEE32Exec`.

This is the checked/finite counterpart to `Runtime.RL.Core.generalizedAdvantageEstimationVec`.

Reference:
- Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation"
  (2015): https://arxiv.org/abs/1506.02438
-/
def generalizedAdvantageEstimationVecIEEE32ExecChecked {n : Nat}
    (gamma lam : Float32Exec)
    (rewards values nextValues : Tensor Float32Exec (.dim n .scalar))
    (dones : Tensor Bool (.dim n .scalar)) :
    Except String (Tensor Float32Exec (.dim n .scalar)) := do
  let rArr : Array Float32Exec :=
    Array.ofFn (fun i : Fin n => Tensor.toScalar (get rewards i))
  let vArr : Array Float32Exec :=
    Array.ofFn (fun i : Fin n => Tensor.toScalar (get values i))
  let nvArr : Array Float32Exec :=
    Array.ofFn (fun i : Fin n => Tensor.toScalar (get nextValues i))
  let dArr : Array Bool :=
    Array.ofFn (fun i : Fin n => Tensor.toScalar (get dones i))

  let mut out : Array Float32Exec := Array.replicate n (0 : Float32Exec)
  let mut advNext : Float32Exec := 0
  for t in [0:n] do
    let idx := n - 1 - t
    let done := dArr[idx]!
    let mask : Float32Exec := continueMask (α := Float32Exec) done
    -- delta = r + γ * mask * nextValue - value
    let t1 ← checkedMul "gae/mul(gamma,mask)" gamma mask
    let t2 ← checkedMul "gae/mul(t1,nextValue)" t1 (nvArr[idx]!)
    let t3 ← checkedAdd "gae/add(reward,t2)" (rArr[idx]!) t2
    let delta ← checkedSub "gae/sub(t3,value)" t3 (vArr[idx]!)
    -- adv = delta + γ * λ * mask * advNext
    let u1 ← checkedMul "gae/mul(gamma,lam)" gamma lam
    let u2 ← checkedMul "gae/mul(u1,mask)" u1 mask
    let u3 ← checkedMul "gae/mul(u2,advNext)" u2 advNext
    let adv ← checkedAdd "gae/add(delta,u3)" delta u3
    advNext := adv
    out := out.set! idx adv

  pure <| Tensor.dim (fun i : Fin n => Tensor.scalar (out[i.val]!))

/--
Checked z-score normalization (mean-center then divide by standard deviation), specialized to
`IEEE32Exec`.

This is used by PPO to normalize advantages.

Implementation note:
we reuse `Spec.normalize_zscore_spec` for the math, but additionally enforce that all outputs are
finite. If the computed standard deviation is zero, `normalize_zscore_spec` returns the centered
vector, which is still validated for finiteness here.
-/
def normalizeZScoreIEEE32ExecChecked {n : Nat}
    (x : Tensor Float32Exec (.dim n .scalar)) :
    Except String (Tensor Float32Exec (.dim n .scalar)) := do
  let y : Tensor Float32Exec (.dim n .scalar) :=
    Spec.normalizeZscoreSpec (α := Float32Exec) (n := n) x
  if Boundary.tensorAll (α := Float32Exec) (s := .dim n .scalar)
      (fun z => TorchLean.Floats.IEEE754.IEEE32Exec.isFinite z) y then
    .ok y
  else
    .error "RL float32: normalizeZScore produced a non-finite entry."


end Float32
end Numerics
end RL
end Runtime
