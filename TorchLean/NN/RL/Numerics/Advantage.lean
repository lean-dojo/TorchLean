/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.RL.Numerics.Returns
public import TorchLean.Tensor.LinearAlgebra.Algorithms

/-!
# Checked TD Residuals, GAE and Advantage Normalization

This module holds the advantage estimation that runs on top of the checked discounted backup, in
any configured FloatLib binary format: TD residuals, fixed-horizon $\operatorname{GAE}(\lambda)$
and z-score normalization. Each routine performs the operations of its `RL` counterpart in the
same order. The TD residual and GAE check every result, and the z-score normalization checks the
entries it returns.

We wrote them by hand. The residual is the TD error (6.5) of Sutton and Barto, which the GAE paper
writes $\delta^V_t$, and `gae` is that paper's eq. (16) cut off at the horizon. In an IEEE format,
`RL.Checked.toReal_of_tdResidual` gives the real value of a successful residual with each
operation rounded once. We found no theorem or test for `gae` or `normalizeZScore`.

## Main declarations

* `tdResidual` and `tdResidual_eq_ok`: the checked TD residual, and the finiteness and
  `TorchLean.RL.tdResidual` equation of a successful run.
* `gae`: checked $\operatorname{GAE}(\lambda)$ with separate
  termination and boundary masks.
* `normalizeZScore`: checked z-score normalization of advantages.

References:
- Sutton and Barto, *Reinforcement Learning: An Introduction* (2nd ed., MIT Press, 2018),
  Section 6.1, eq. (6.5): http://incompleteideas.net/book/the-book-2nd.html
- Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation"
  (2015), ICLR 2016, Section 3 and eq. (16): https://arxiv.org/abs/1506.02438
-/

@[expose] public section

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.ExecFloat (Binary)

namespace TorchLean
namespace RL
namespace Checked

open TorchLean.Tensor
open FloatLib.Floats.Formats.BinaryInterchange

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
    [ExecFloat.ModelCodec plan (Model format) code]

local notation "Value" => ExecFloat (Configured.Family format code plan)

/--
Checked TD residual / Bellman error: `discountedBackup` minus a checked `value`.
-/
def tdResidual
    [ExecFloat.Backend.PolicyFor (Configured.Family format code plan)]
    [ExecFloat.Add (Configured.Family format code plan)]
    [ExecFloat.Mul (Configured.Family format code plan)]
    [ExecFloat.Sub (Configured.Family format code plan)]
    (value reward gamma nextValue : Value) (done : Bool) :
    Except String Value :=
  match discountedBackup (reward := reward) (gamma := gamma)
      (bootstrap := nextValue) (done := done) with
  | .error e => .error e
  | .ok target =>
      match requireFinite "tdResidual/value" value with
      | .error e => .error e
      | .ok _ =>
          sub "tdResidual/sub(target,value)" target value

/--
If `tdResidual` returns `.ok out`, then:

- the values computed by the checked discounted backup are finite,
- the value and the final subtraction are finite, and
- `out` agrees with the spec-layer `tdResidual` formula.

The theorem extends `discountedBackup_eq_ok` from the backup to the residual.
-/
theorem tdResidual_eq_ok
    [ExecFloat.Backend.PolicyFor (Configured.Family format code plan)]
    [ExecFloat.Add (Configured.Family format code plan)]
    [ExecFloat.Mul (Configured.Family format code plan)]
    [ExecFloat.Sub (Configured.Family format code plan)]
    (value reward gamma nextValue : Value) (done : Bool) (out : Value)
    (h : tdResidual value reward gamma nextValue done = .ok out) :
    (if done then Binary.isFinite reward = true
      else Binary.isFinite (ExecFloat.mul gamma nextValue) = true ∧
        Binary.isFinite (ExecFloat.add reward (ExecFloat.mul gamma nextValue)) = true) ∧
      Binary.isFinite value = true ∧
      Binary.isFinite
          (ExecFloat.sub (RL.discountedBackup (α := Value) reward gamma nextValue done) value) =
        true ∧
      out = RL.tdResidual (α := Value) value reward gamma nextValue done := by
  cases htarget : discountedBackup (reward := reward) (gamma := gamma)
      (bootstrap := nextValue) (done := done) with
  | error e =>
      simp [tdResidual, htarget] at h
  | ok target =>
      have hval : Binary.isFinite value = true := by
        cases hf : Binary.isFinite value with
        | true => rfl
        | false => simp [tdResidual, htarget, requireFinite, hf] at h
      have hsub :
          sub "tdResidual/sub(target,value)" target value = .ok out := by
        simpa [tdResidual, htarget, requireFinite, hval] using h
      obtain ⟨hbackup, htargetEq⟩ :=
        discountedBackup_eq_ok
          (reward := reward) (gamma := gamma) (bootstrap := nextValue) (done := done)
          (out := target) htarget
      set out0 : Value := ExecFloat.sub target value
      have hout0 : Binary.isFinite out0 = true := by
        cases hf : Binary.isFinite out0 with
        | true => rfl
        | false => simp [sub, requireFinite, Except.map, out0, hf] at hsub
      have hout : out = out0 := by
        have : sub "tdResidual/sub(target,value)" target value = .ok out0 := by
          simp [sub, requireFinite, Except.map, out0, hout0]
        exact Except.ok.inj (hsub.symm.trans this)
      refine ⟨hbackup, hval, ?_, ?_⟩
      · simpa [out0, htargetEq] using hout0
      · rw [hout]
        simp [out0, htargetEq, RL.tdResidual, HSub.hSub, Sub.sub]

/--
`RL.gae` with every result checked, performing its operations in
the same order. `terminated[t]` drops the bootstrap value `nextValues[t]` and `boundaries[t]` stops
the recursion from reading step `t + 1`, so a time-limit truncation sets only `boundaries[t]`.
-/
def gae
    [ExecFloat.Backend.PolicyFor (Configured.Family format code plan)]
    [ExecFloat.Add (Configured.Family format code plan)]
    [ExecFloat.Mul (Configured.Family format code plan)]
    [ExecFloat.Sub (Configured.Family format code plan)]
    {n : Nat}
    (gamma lam : Value)
    (rewards values nextValues : Tensor Value [n])
    (terminated boundaries : Tensor Bool [n]) :
    Except String (Tensor Value [n]) := do
  let indices : Tensor (Fin n) [n] := Tensor.ofFn id
  Tensor.scanrM (fun idx advNext => do
    -- Terms that a mask discards are never computed, so they cannot fail a finiteness check.
    let delta ← if terminated[idx] then
        sub "gae/sub(reward,value)" rewards[idx] values[idx]
      else do
        let future ← mul "gae/mul(gamma,nextValue)" gamma nextValues[idx]
        let target ← add "gae/add(reward,future)" rewards[idx] future
        sub "gae/sub(target,value)" target values[idx]
    if boundaries[idx] then return delta
    let discount ← mul "gae/mul(gamma,lam)" gamma lam
    let continuation ← mul "gae/mul(discount,advNext)" discount advNext
    add "gae/add(delta,continuation)" delta continuation) 0 indices

/--
`Tensor.normalizeZscore x`, the population z-score, when every entry of it is finite, otherwise
an error. When the largest magnitude of the centered entries is not positive the spec returns the
centered vector, which is checked the same way.
-/
def normalizeZScore {n : Nat}
    (x : Tensor Value [n]) :
    Except String (Tensor Value [n]) := do
  let y : Tensor Value [n] :=
    Tensor.normalizeZscore (α := Value) (n := n) x
  if y.all Binary.isFinite then
    .ok y
  else
    .error "RL numerics: normalizeZScore produced a non-finite entry."


end Checked
end RL
end TorchLean
