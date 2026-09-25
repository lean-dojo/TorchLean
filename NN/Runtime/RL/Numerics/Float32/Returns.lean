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
## Checked RL core transforms (configured binary32)
-/

/-- Require that an `Binary 8 23` value is finite, producing a tagged error on failure. -/
def requireFinite (label : String) (x : Binary 8 23) : Except String Unit :=
  if Binary.isFinite x = true then
    .ok ()
  else
    .error s!"RL float32: non-finite configured binary32 value at {label}: {x}"

/-!
## Checked configured binary32 primitives

The checked RL helpers below are intentionally written in terms of a few small “checked primitive”
combinators (`checkedAdd`, `checkedMul`, …). Larger routines (GAE, PPO objectives, …) remain
readable while still producing *precise* error locations when non-finite values occur.

Exponential and logarithm use FloatLib's configured deterministic approximations. Their checks
reject nonfinite results; they do not certify correct rounding or a real-error bound.
-/

/-- Checked configured binary32 addition. -/
def checkedAdd (label : String) (x y : Binary 8 23) : Except String (Binary 8 23) :=
  let z := ExecFloat.add x y
  match requireFinite label z with
  | .ok _ => .ok z
  | .error e => .error e

/-- Checked configured binary32 subtraction. -/
def checkedSub (label : String) (x y : Binary 8 23) : Except String (Binary 8 23) :=
  let z := ExecFloat.sub x y
  match requireFinite label z with
  | .ok _ => .ok z
  | .error e => .error e

/-- Checked configured binary32 multiplication. -/
def checkedMul (label : String) (x y : Binary 8 23) : Except String (Binary 8 23) :=
  let z := ExecFloat.mul x y
  match requireFinite label z with
  | .ok _ => .ok z
  | .error e => .error e

/-- Checked configured binary32 division. -/
def checkedDiv (label : String) (x y : Binary 8 23) : Except String (Binary 8 23) :=
  let z := ExecFloat.div x y
  match requireFinite label z with
  | .ok _ => .ok z
  | .error e => .error e

/-- Approximate `eˣ` in binary32 and reject a nonfinite result. -/
def checkedExp (label : String) (x : Binary 8 23) : Except String (Binary 8 23) :=
  let z := FloatLib.Floats.ExecFloat.Binary.exp x
  match requireFinite label z with
  | .ok _ => .ok z
  | .error e => .error e

/-- Approximate the natural logarithm in binary32 and reject a nonfinite result. -/
def checkedLog (label : String) (x : Binary 8 23) : Except String (Binary 8 23) :=
  let z := FloatLib.Floats.ExecFloat.Binary.log x
  match requireFinite label z with
  | .ok _ => .ok z
  | .error e => .error e

/-- Checked configured binary32 square root. -/
def checkedSqrt (label : String) (x : Binary 8 23) : Except String (Binary 8 23) :=
  let z := (Binary.sqrt (rounding := .nearestEven)) x
  match requireFinite label z with
  | .ok _ => .ok z
  | .error e => .error e

/-- Checked configured binary32 `min` using IEEE-754 `minimum`. -/
def checkedMin (label : String) (x y : Binary 8 23) : Except String (Binary 8 23) :=
  let z := min x y
  match requireFinite label z with
  | .ok _ => .ok z
  | .error e => .error e

/-- Checked configured binary32 `max` using IEEE-754 `maximum`. -/
def checkedMax (label : String) (x y : Binary 8 23) : Except String (Binary 8 23) :=
  let z := max x y
  match requireFinite label z with
  | .ok _ => .ok z
  | .error e => .error e

/--
Checked version of the one-step discounted backup

`reward + γ * (1-done) * bootstrap`

specialized to `Binary 8 23`.

The runtime return type is `Except String …` so training code can choose to:
- fail fast, or
- fall back to a safer scalar backend (interval/oracle), or
- log and skip a bad sample.
-/
def discountedBackupChecked
    (reward gamma bootstrap : Binary 8 23) (done : Bool) :
    Except String (Binary 8 23) :=
  let mask : Binary 8 23 := Spec.RL.continueMask (α := Binary 8 23) done
  match checkedMul "discountedBackup/mul(gamma,mask)" gamma mask with
  | .error e => .error e
  | .ok t1 =>
      match checkedMul "discountedBackup/mul(t1,bootstrap)" t1 bootstrap with
      | .error e => .error e
      | .ok t2 => checkedAdd "discountedBackup/add(reward,t2)" reward t2

/-!
## Checked preconditions → proof hypotheses

The `NN/Proofs/RL/Floats/*` bridge theorems for `Binary 8 23` are usually stated with
explicit
`isFinite … = true` hypotheses for each intermediate.

The lemma below is the glue between runtime safety checks and those proof hypotheses:

*If the checked routine returns `.ok`, then all the finiteness side-conditions needed by the
semantic bridge theorems hold automatically.*
-/

/--
If `discountedBackupChecked` returns `.ok out`, then:

- every configured binary32 intermediate used by the refinement theorem is finite, and
- `out` agrees with the spec-layer `discountedBackup` formula.
-/
theorem discountedBackup_eq_ok
    (reward gamma bootstrap : Binary 8 23) (done : Bool) (out : Binary 8 23)
    (h : discountedBackupChecked reward gamma bootstrap done = .ok out) :
    Binary.isFinite
        (ExecFloat.mul gamma (continueMask (α := Binary 8 23) done)) =
      true ∧
      Binary.isFinite
          (ExecFloat.mul
            (ExecFloat.mul gamma (continueMask (α := Binary 8 23) done))
            bootstrap) =
        true ∧
        Binary.isFinite
            (ExecFloat.add reward
              (ExecFloat.mul
                (ExecFloat.mul gamma
                  (continueMask (α := Binary 8 23) done))
                bootstrap)) =
          true ∧
          out = discountedBackup (α := Binary 8 23) reward gamma bootstrap done := by
  -- Abbreviate the intermediate values so we can reason by contradiction on each check.
  set mask : Binary 8 23 := continueMask (α := Binary 8 23) done
  set t1 : Binary 8 23 := ExecFloat.mul gamma mask
  set t2 : Binary 8 23 := ExecFloat.mul t1 bootstrap
  set out0 : Binary 8 23 := ExecFloat.add reward t2

  have ht1 : Binary.isFinite t1 = true := by
    cases hft1 : Binary.isFinite t1 with
    | true =>
        rfl
    | false =>
        -- If the first intermediate is not finite, the checked routine must return `.error _`,
        -- contradicting `h`.
        have : False := by
          have h' := h
          simp [discountedBackupChecked, checkedMul, requireFinite, mask, t1, hft1] at h'
        exact this.elim

  have ht2 : Binary.isFinite t2 = true := by
    cases hft2 : Binary.isFinite t2 with
    | true =>
        rfl
    | false =>
        have : False := by
          have h' := h
          simp [discountedBackupChecked, checkedMul, requireFinite, mask, t1, t2, ht1, hft2] at h'
        exact this.elim

  have hout0 : Binary.isFinite out0 = true := by
    cases hfout : Binary.isFinite out0 with
    | true =>
        rfl
    | false =>
        have : False := by
          have h' := h
          simp [discountedBackupChecked, checkedMul, checkedAdd, requireFinite,
            mask, t1, t2, out0, ht1, ht2, hfout] at h'
        exact this.elim

  -- If all checks passed, the routine returns the plain `discountedBackup` expression.
  have hout : out = out0 := by
    have : discountedBackupChecked reward gamma bootstrap done = .ok out0 := by
      simp [discountedBackupChecked, checkedMul, checkedAdd, requireFinite, mask, t1, t2, out0,
        ht1, ht2, hout0]
    -- Both `h` and `this` identify the return value; compare them by constructor injection.
    have hok : (Except.ok out : Except String (Binary 8 23)) = Except.ok out0 := by
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
    have : out0 = discountedBackup (α := Binary 8 23) reward gamma bootstrap done := by
      -- After unfolding the local abbreviations, this is definitional.
      rfl
    simp [hout, this]

/--
Checked fixed-horizon discounted returns (no `done` flags), specialized to `Binary 8 23`.

This is the checked/finite counterpart to `Runtime.RL.Core.discountedReturnsFrom`.
-/
def discountedReturnsChecked {n : Nat}
    (gamma : Binary 8 23) (rewards : Tensor (Binary 8 23) [n])
    (bootstrap : Binary 8 23 := (0 : Binary 8 23)) :
    Except String (Tensor (Binary 8 23) [n]) := do
  Tensor.scanrM
    (fun reward future => discountedBackupChecked reward gamma future false)
    bootstrap rewards


end Float32
end Numerics
end RL
end Runtime
