/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.RL.Numerics.Float32.Types

/-!
# Checked Float32 Discounted Returns

This module contains the value-learning recurrences that need explicit finite-intermediate checks:
discounted backups and fixed-horizon discounted returns. The public names stay in
`Runtime.RL.Numerics.Float32`; this file only separates the implementation so the runtime tree is
easier to audit.

Reference: Sutton and Barto, *Reinforcement Learning: An Introduction*.
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
## Checked RL core transforms (IEEE32Exec)
-/

/-- Require that an `IEEE32Exec` value is finite, producing a tagged error on failure. -/
def requireFinite (label : String) (x : Float32Exec) : Except String Unit :=
  if TorchLean.Floats.IEEE754.IEEE32Exec.isFinite x = true then
    .ok ()
  else
    .error s!"RL float32: non-finite IEEE32Exec value at {label}: {x}"

/-!
## Checked IEEE32Exec primitives

The checked RL helpers below are intentionally written in terms of a few small “checked primitive”
combinators (`checkedAdd`, `checkedMul`, …). This keeps larger routines (GAE, PPO objectives, …)
readable while still producing *precise* error locations when non-finite values occur.
-/

/-- Checked IEEE32Exec addition. -/
def checkedAdd (label : String) (x y : Float32Exec) : Except String Float32Exec :=
  let z := TorchLean.Floats.IEEE754.IEEE32Exec.add x y
  match requireFinite label z with
  | .ok _ => .ok z
  | .error e => .error e

/-- Checked IEEE32Exec subtraction. -/
def checkedSub (label : String) (x y : Float32Exec) : Except String Float32Exec :=
  let z := TorchLean.Floats.IEEE754.IEEE32Exec.sub x y
  match requireFinite label z with
  | .ok _ => .ok z
  | .error e => .error e

/-- Checked IEEE32Exec multiplication. -/
def checkedMul (label : String) (x y : Float32Exec) : Except String Float32Exec :=
  let z := TorchLean.Floats.IEEE754.IEEE32Exec.mul x y
  match requireFinite label z with
  | .ok _ => .ok z
  | .error e => .error e

/-- Checked IEEE32Exec division. -/
def checkedDiv (label : String) (x y : Float32Exec) : Except String Float32Exec :=
  let z := TorchLean.Floats.IEEE754.IEEE32Exec.div x y
  match requireFinite label z with
  | .ok _ => .ok z
  | .error e => .error e

/-- Checked IEEE32Exec exponentiation (base-e). -/
def checkedExp (label : String) (x : Float32Exec) : Except String Float32Exec :=
  let z := TorchLean.Floats.IEEE754.IEEE32Exec.exp x
  match requireFinite label z with
  | .ok _ => .ok z
  | .error e => .error e

/-- Checked IEEE32Exec logarithm (natural log). -/
def checkedLog (label : String) (x : Float32Exec) : Except String Float32Exec :=
  let z := TorchLean.Floats.IEEE754.IEEE32Exec.log x
  match requireFinite label z with
  | .ok _ => .ok z
  | .error e => .error e

/-- Checked IEEE32Exec square root. -/
def checkedSqrt (label : String) (x : Float32Exec) : Except String Float32Exec :=
  let z := TorchLean.Floats.IEEE754.IEEE32Exec.sqrt x
  match requireFinite label z with
  | .ok _ => .ok z
  | .error e => .error e

/-- Checked IEEE32Exec `min` using IEEE-754 `minimum`. -/
def checkedMin (label : String) (x y : Float32Exec) : Except String Float32Exec :=
  let z := TorchLean.Floats.IEEE754.IEEE32Exec.minimum x y
  match requireFinite label z with
  | .ok _ => .ok z
  | .error e => .error e

/-- Checked IEEE32Exec `max` using IEEE-754 `maximum`. -/
def checkedMax (label : String) (x y : Float32Exec) : Except String Float32Exec :=
  let z := TorchLean.Floats.IEEE754.IEEE32Exec.maximum x y
  match requireFinite label z with
  | .ok _ => .ok z
  | .error e => .error e

/--
Checked version of the one-step discounted backup

`reward + γ * (1-done) * bootstrap`

specialized to `IEEE32Exec`.

The runtime return type is `Except String …` so training code can choose to:
- fail fast, or
- fall back to a safer scalar backend (interval/oracle), or
- log and skip a bad sample.
-/
def discountedBackupIEEE32ExecChecked
    (reward gamma bootstrap : Float32Exec) (done : Bool) :
    Except String Float32Exec :=
  let mask : Float32Exec := Spec.RL.continueMask (α := Float32Exec) done
  match checkedMul "discountedBackup/mul(gamma,mask)" gamma mask with
  | .error e => .error e
  | .ok t1 =>
      match checkedMul "discountedBackup/mul(t1,bootstrap)" t1 bootstrap with
      | .error e => .error e
      | .ok t2 => checkedAdd "discountedBackup/add(reward,t2)" reward t2

/-!
## Checked preconditions → proof hypotheses

The `NN/Proofs/RL/Floats/*` bridge theorems for `IEEE32Exec` are usually stated with explicit
`isFinite … = true` hypotheses for each intermediate.

The lemma below is the glue between runtime safety checks and those proof hypotheses:

*If the checked routine returns `.ok`, then all the finiteness side-conditions needed by the
semantic bridge theorems hold automatically.*
-/

/--
If `discountedBackupIEEE32ExecChecked` returns `.ok out`, then:

- every IEEE32Exec intermediate used by the refinement theorem is finite, and
- `out` agrees with the spec-layer `discountedBackup` formula.
-/
theorem discountedBackupIEEE32ExecChecked_eq_ok
    (reward gamma bootstrap : Float32Exec) (done : Bool) (out : Float32Exec)
    (h : discountedBackupIEEE32ExecChecked reward gamma bootstrap done = .ok out) :
    TorchLean.Floats.IEEE754.IEEE32Exec.isFinite
        (TorchLean.Floats.IEEE754.IEEE32Exec.mul gamma (continueMask (α := Float32Exec) done)) =
      true ∧
      TorchLean.Floats.IEEE754.IEEE32Exec.isFinite
          (TorchLean.Floats.IEEE754.IEEE32Exec.mul
            (TorchLean.Floats.IEEE754.IEEE32Exec.mul gamma (continueMask (α := Float32Exec) done))
            bootstrap) =
        true ∧
        TorchLean.Floats.IEEE754.IEEE32Exec.isFinite
            (TorchLean.Floats.IEEE754.IEEE32Exec.add reward
              (TorchLean.Floats.IEEE754.IEEE32Exec.mul
                (TorchLean.Floats.IEEE754.IEEE32Exec.mul gamma
                  (continueMask (α := Float32Exec) done))
                bootstrap)) =
          true ∧
          out = discountedBackup (α := Float32Exec) reward gamma bootstrap done := by
  -- Abbreviate the intermediate values so we can reason by contradiction on each check.
  set mask : Float32Exec := continueMask (α := Float32Exec) done
  set t1 : Float32Exec := TorchLean.Floats.IEEE754.IEEE32Exec.mul gamma mask
  set t2 : Float32Exec := TorchLean.Floats.IEEE754.IEEE32Exec.mul t1 bootstrap
  set out0 : Float32Exec := TorchLean.Floats.IEEE754.IEEE32Exec.add reward t2

  have ht1 : TorchLean.Floats.IEEE754.IEEE32Exec.isFinite t1 = true := by
    cases hft1 : TorchLean.Floats.IEEE754.IEEE32Exec.isFinite t1 with
    | true =>
        rfl
    | false =>
        -- If the first intermediate is not finite, the checked routine must return `.error _`,
        -- contradicting `h`.
        have : False := by
          have h' := h
          simp [discountedBackupIEEE32ExecChecked, checkedMul, requireFinite, mask, t1, hft1] at h'
        exact this.elim

  have ht2 : TorchLean.Floats.IEEE754.IEEE32Exec.isFinite t2 = true := by
    cases hft2 : TorchLean.Floats.IEEE754.IEEE32Exec.isFinite t2 with
    | true =>
        rfl
    | false =>
        have : False := by
          have h' := h
          simp [discountedBackupIEEE32ExecChecked, checkedMul, requireFinite, mask, t1, t2, ht1, hft2] at h'
        exact this.elim

  have hout0 : TorchLean.Floats.IEEE754.IEEE32Exec.isFinite out0 = true := by
    cases hfout : TorchLean.Floats.IEEE754.IEEE32Exec.isFinite out0 with
    | true =>
        rfl
    | false =>
        have : False := by
          have h' := h
          simp [discountedBackupIEEE32ExecChecked, checkedMul, checkedAdd, requireFinite,
            mask, t1, t2, out0, ht1, ht2, hfout] at h'
        exact this.elim

  -- If all checks passed, the routine returns the plain `discountedBackup` expression.
  have hout : out = out0 := by
    have : discountedBackupIEEE32ExecChecked reward gamma bootstrap done = .ok out0 := by
      simp [discountedBackupIEEE32ExecChecked, checkedMul, checkedAdd, requireFinite, mask, t1, t2, out0, ht1, ht2, hout0]
    -- Both `h` and `this` identify the return value; compare them by constructor injection.
    have hok : (Except.ok out : Except String Float32Exec) = Except.ok out0 := by
      exact h.symm.trans this
    have : out = out0 := by
      injection hok
    exact this

  refine ⟨?_, ?_, ?_, ?_⟩
  · -- First intermediate is exactly `mul gamma mask`.
    simpa [t1, mask] using ht1
  · -- Second intermediate is `mul (mul gamma mask) bootstrap`.
    simpa [t2, t1, mask] using ht2
  · -- Output intermediate.
    simpa [out0, t2, t1, mask] using hout0
  · -- Result equality.
    -- `out0` is definitionally the spec-layer discounted backup.
    have : out0 = discountedBackup (α := Float32Exec) reward gamma bootstrap done := by
      -- After unfolding the local abbreviations, this is definitional.
      rfl
    simp [hout, this]

/--
Checked fixed-horizon discounted returns (no `done` flags), specialized to `IEEE32Exec`.

This is the checked/finite counterpart to `Runtime.RL.Core.discountedReturnsVecFrom`.
-/
def discountedReturnsVecFromIEEE32ExecChecked {n : Nat}
    (gamma : Float32Exec) (rewards : Tensor Float32Exec (.dim n .scalar))
    (bootstrap : Float32Exec := (0 : Float32Exec)) :
    Except String (Tensor Float32Exec (.dim n .scalar)) := do
  let rArr : Array Float32Exec :=
    Array.ofFn (fun i : Fin n => Tensor.toScalar (get rewards i))

  let mut out : Array Float32Exec := Array.replicate n (0 : Float32Exec)
  let mut g : Float32Exec := bootstrap
  for t in [0:n] do
    let idx := n - 1 - t
    g ← discountedBackupIEEE32ExecChecked (reward := rArr[idx]!) (gamma := gamma) (bootstrap := g) (done := false)
    out := out.set! idx g

  pure <| Tensor.dim (fun i : Fin n => Tensor.scalar (out[i.val]!))


end Float32
end Numerics
end RL
end Runtime
