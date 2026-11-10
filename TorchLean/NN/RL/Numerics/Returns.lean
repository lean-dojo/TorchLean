/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.RL.Numerics.Types

/-!
# Checked Discounted Returns

This module holds the checked arithmetic of the RL numerics, in any configured FloatLib binary
format, and the value-learning recurrences built on it: the one-step discounted backup and
fixed-horizon discounted returns. A checked operation fails with an error naming the operation as
soon as its result is not finite. `discountedBackup_eq_ok` turns a successful backup into the
finiteness hypotheses of the rounding chains in `TorchLean.Verify.RuntimeApprox.RL.Backup`.

We wrote the checks by hand around the spec layer. `RL.discountedBackup` is one step of the return
recursion (3.9) of Sutton and Barto, and `RL.discountedReturns` runs it from the last reward back.
The default `bootstrap` of 0 is their $G_T = 0$, and like `RL.discountedReturns` the checked scan
takes no `done` flags. An error names the failing operation and carries no step index. In an IEEE
format, `RL.Checked.toReal_of_discountedBackup` gives the real value of a successful backup, with
the product and the sum each rounded once. Neither a theorem nor a test covers the checked
`discountedReturns`.

## Main declarations

* `requireFinite`, `add`, `sub`, `mul`, `exp`, `min` and
  `max`: the checked primitives.
* `discountedBackup` and `discountedReturns`: the checked recurrences.
* `discountedBackup_eq_ok`: a successful backup equals `TorchLean.RL.discountedBackup`, and every
  value it computed is finite.

Reference: Sutton and Barto, *Reinforcement Learning: An Introduction* (2nd ed., MIT Press, 2018),
Section 3.3, eqs. (3.8) and (3.9): http://incompleteideas.net/book/the-book-2nd.html
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

/-!
## Checked primitives
-/

/-- `.ok ()` when `x` is finite, otherwise an error naming `label` and `x`. -/
def requireFinite (label : String) (x : Value) : Except String Unit :=
  if Binary.isFinite x = true then
    .ok ()
  else
    .error s!"RL numerics: non-finite value at {label}: {x}"

/-- The sum `ExecFloat.add x y` when it is finite, otherwise an error naming `label`. -/
def add
    [ExecFloat.Backend.PolicyFor (Configured.Family format code plan)]
    [ExecFloat.Add (Configured.Family format code plan)]
    (label : String) (x y : Value) : Except String Value :=
  let z := ExecFloat.add x y
  (requireFinite label z).map fun _ => z

/-- The difference `ExecFloat.sub x y` when it is finite, otherwise an error naming `label`. -/
def sub
    [ExecFloat.Backend.PolicyFor (Configured.Family format code plan)]
    [ExecFloat.Sub (Configured.Family format code plan)]
    (label : String) (x y : Value) : Except String Value :=
  let z := ExecFloat.sub x y
  (requireFinite label z).map fun _ => z

/-- The product `ExecFloat.mul x y` when it is finite, otherwise an error naming `label`. -/
def mul
    [ExecFloat.Backend.PolicyFor (Configured.Family format code plan)]
    [ExecFloat.Mul (Configured.Family format code plan)]
    (label : String) (x y : Value) : Except String Value :=
  let z := ExecFloat.mul x y
  (requireFinite label z).map fun _ => z

/--
`CorrectlyRounded.exp x` when it is finite, otherwise an error naming `label`. In an IEEE format
a finite result for a finite `x` is the nearest-even rounding of the real exponential.
-/
def exp (label : String) (x : Value) : Except String Value :=
  let z := Binary.CorrectlyRounded.exp x
  (requireFinite label z).map fun _ => z

/--
The IEEE 754-2019 `minimum` of `x` and `y` when it is finite, otherwise an error naming `label`.
-/
def min (label : String) (x y : Value) : Except String Value :=
  let z := Min.min x y
  (requireFinite label z).map fun _ => z

/--
The IEEE 754-2019 `maximum` of `x` and `y` when it is finite, otherwise an error naming `label`.
-/
def max (label : String) (x y : Value) : Except String Value :=
  let z := Max.max x y
  (requireFinite label z).map fun _ => z

/-!
## Checked recurrences
-/

/--
`RL.discountedBackup` with every result checked: the reward when `done`, otherwise
`reward + gamma * bootstrap`. A terminal step checks only the reward and reads neither `gamma` nor
`bootstrap`.
-/
def discountedBackup
    [ExecFloat.Backend.PolicyFor (Configured.Family format code plan)]
    [ExecFloat.Add (Configured.Family format code plan)]
    [ExecFloat.Mul (Configured.Family format code plan)]
    (reward gamma bootstrap : Value) (done : Bool) :
    Except String Value :=
  if done then
    (requireFinite "discountedBackup/reward" reward).map fun _ => reward
  else do
    let future ← mul "discountedBackup/mul(gamma,bootstrap)" gamma bootstrap
    add "discountedBackup/add(reward,future)" reward future

/--
If `discountedBackup` returns `.ok out`, then every value it computed is finite (the reward
at a terminal step, otherwise the product `gamma * bootstrap` and the sum) and `out` is the
spec-layer `discountedBackup`.
-/
theorem discountedBackup_eq_ok
    [ExecFloat.Backend.PolicyFor (Configured.Family format code plan)]
    [ExecFloat.Add (Configured.Family format code plan)]
    [ExecFloat.Mul (Configured.Family format code plan)]
    (reward gamma bootstrap : Value) (done : Bool) (out : Value)
    (h : discountedBackup reward gamma bootstrap done = .ok out) :
    (if done then Binary.isFinite reward = true
      else Binary.isFinite (ExecFloat.mul gamma bootstrap) = true ∧
        Binary.isFinite (ExecFloat.add reward (ExecFloat.mul gamma bootstrap)) = true) ∧
      out = RL.discountedBackup (α := Value) reward gamma bootstrap done := by
  cases done with
  | true =>
      cases hr : Binary.isFinite reward with
      | false => simp [discountedBackup, requireFinite, Except.map, hr] at h
      | true =>
          simp only [discountedBackup, requireFinite, Except.map, hr, ite_true] at h
          exact ⟨rfl, (Except.ok.inj h).symm⟩
  | false =>
      have hmul : Binary.isFinite (ExecFloat.mul gamma bootstrap) = true := by
        cases hf : Binary.isFinite (ExecFloat.mul gamma bootstrap) with
        | true => rfl
        | false =>
            simp [discountedBackup, mul, requireFinite, Except.map,
              Bind.bind, Except.bind, hf] at h
      have hadd :
          Binary.isFinite (ExecFloat.add reward (ExecFloat.mul gamma bootstrap)) = true := by
        cases hf : Binary.isFinite (ExecFloat.add reward (ExecFloat.mul gamma bootstrap)) with
        | true => rfl
        | false =>
            simp [discountedBackup, mul, add, requireFinite, Except.map,
              Bind.bind, Except.bind, hmul, hf] at h
      have hok : discountedBackup reward gamma bootstrap false =
          .ok (ExecFloat.add reward (ExecFloat.mul gamma bootstrap)) := by
        simp [discountedBackup, mul, add, requireFinite, Except.map,
          Bind.bind, Except.bind, hmul, hadd]
      exact ⟨⟨hmul, hadd⟩, Except.ok.inj (h.symm.trans hok)⟩

/--
`RL.discountedReturns` with every backup checked by `discountedBackup`, over a fixed
horizon without `done` flags.
-/
def discountedReturns
    [ExecFloat.Backend.PolicyFor (Configured.Family format code plan)]
    [ExecFloat.Add (Configured.Family format code plan)]
    [ExecFloat.Mul (Configured.Family format code plan)]
    {n : Nat} (gamma : Value) (rewards : Tensor Value [n]) (bootstrap : Value := (0 : Value)) :
    Except String (Tensor Value [n]) := do
  Tensor.scanrM
    (fun reward future => discountedBackup reward gamma future false)
    bootstrap rewards


end Checked
end RL
end TorchLean
