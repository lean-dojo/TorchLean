/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.RL.Numerics.Types

/-!
# Interval Diagnostics for RL

This module evaluates the return, TD residual, GAE and PPO formulas of the checked RL numerics in
FloatLib's outward-rounded intervals over a configured binary format. The intervals are
executable diagnostics. They don't replace the exact RL specs, but they flag overflow, invalid
endpoints and unstable recurrences.

We wrote each interval form by hand from its checked counterpart, branch for branch, with point
intervals for the scalars and FloatLib's outward-rounded operations for the arithmetic. The PPO
form is the one that departs. `Checked.ppoClippedObjective` takes the `min` of the two products,
and `ppoClippedObjectiveInterval` returns the hull of their enclosures, which covers the larger
product too. At a ratio of 1.5, an advantage of 1 and `clipEps` 0.2 the objective is 1.2, and the
hull runs from about 1.2 to 1.5.

## Main declarations

* `discountedBackupInterval`, `tdResidualInterval` and `ppoClippedObjectiveInterval`:
  the interval forms of the scalar formulas.
* `discountedReturnsIntervals` and `gaeIntervals`: the interval forms
  of the recurrences.
* `returnsWithinIntervals`: the executable check that computed returns lie in their intervals.

References:
- IEEE Std 1788-2015, "IEEE Standard for Interval Arithmetic", for interval arithmetic semantics:
  https://doi.org/10.1109/IEEESTD.2015.7140721
- Sutton and Barto, *Reinforcement Learning: An Introduction* (2nd ed., MIT Press, 2018),
  Sections 3.3 and 6.1, for the return and TD recurrences:
  http://incompleteideas.net/book/the-book-2nd.html
- Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation"
  (2015), ICLR 2016, Section 3: https://arxiv.org/abs/1506.02438
- Schulman et al., "Proximal Policy Optimization Algorithms" (2017), arXiv:1707.06347, Section 3:
  https://arxiv.org/abs/1707.06347
-/

@[expose] public section

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.ExecFloat (Binary)
open FloatLib.Numerics (Interval)

namespace TorchLean
namespace RL
namespace Checked

open TorchLean.Tensor
open FloatLib.Floats.Formats.BinaryInterchange

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
    [ExecFloat.ModelCodec plan (Model format) code]

local notation "Value" => ExecFloat (Configured.Family format code plan)


/-!
## Interval enclosures
-/

/--
Outward-rounded interval for the one-step discounted backup: the reward's point interval at a
terminal step, otherwise the interval sum of the reward and `gamma * bootstrap`.

As in `Checked.discountedBackup`, a terminal step does not read `gamma` or `bootstrap`. The interval
form always returns an interval, where the checked form can fail.

Reference:
- Sutton and Barto, *Reinforcement Learning: An Introduction* (2nd ed., 2018), eq. (3.9), the
  return recursion.
-/
def discountedBackupInterval
    (reward gamma bootstrap : Value) (done : Bool) : Interval Value :=
  if done then
    Binary.Interval.point reward
  else
    let r : Interval Value := Binary.Interval.point reward
    let prod : Interval Value :=
      Binary.Interval.mul
        (Binary.Interval.point gamma)
        (Binary.Interval.point bootstrap)
    Binary.Interval.add r prod

/--
Outward-rounded interval for the TD residual: `discountedBackupInterval` minus the value's point
interval, the interval form of `Checked.tdResidual`.

Reference:
- Sutton and Barto, *Reinforcement Learning: An Introduction* (2nd ed., 2018), eq. (6.5), the TD
  error.
-/
def tdResidualInterval
    (value reward gamma nextValue : Value) (done : Bool) : Interval Value :=
  let target : Interval Value := discountedBackupInterval reward gamma nextValue done
  Binary.Interval.sub target
    (Binary.Interval.point value)

/--
The hull of the outward-rounded enclosures of `ratio * A` and `clip ratio * A`, which contains the
PPO clipped surrogate objective, with the thresholds `1 - clipEps` and `1 + clipEps` themselves
outward-rounded intervals. Clipping is monotone in the ratio and in both thresholds, so its range
over the threshold intervals is exact at the endpoints, and a NaN input gives `whole`. The
objective is the summand of eq. (7) in Schulman et al. (2017), arXiv:1707.06347.
-/
def ppoClippedObjectiveInterval
    (ratio advantage clipEps : Value) : Interval Value :=
  let one : Interval Value := Binary.Interval.point (1 : Value)
  let eps : Interval Value := Binary.Interval.point clipEps
  let lo : Interval Value := Binary.Interval.sub one eps
  let hi : Interval Value := Binary.Interval.add one eps
  let clippedRatio : Interval Value :=
    Binary.Interval.ofBounds
      (min hi.lo (max lo.lo ratio))
      (min hi.hi (max lo.hi ratio))
  let unclipped : Interval Value :=
    Binary.Interval.mul
      (Binary.Interval.point ratio)
      (Binary.Interval.point advantage)
  let clipped : Interval Value :=
    Binary.Interval.mul clippedRatio
      (Binary.Interval.point advantage)
  Binary.Interval.hull unclipped clipped

/--
Outward-rounded intervals for fixed-horizon discounted returns, built from the point intervals of
`gamma`, the rewards and `bootstrap`.

FloatLib's interval soundness lemmas (`add_sound` and `mul_sound` in
`FloatLib.Floats.Formats.BinaryInterchange.Configured.Interval.Proof`) take valid operands with
finite endpoints, so the entries enclose the real return recursion when every leaf is finite and
every interval of the scan is valid. No theorem about the whole recursion is proved here. A
non-finite leaf gives an infinite or invalid endpoint, which this executable diagnostic exposes
when run alongside `Checked.discountedReturns`.

Reference:
- Sutton and Barto, *Reinforcement Learning: An Introduction* (2nd ed., 2018), eqs. (3.8) and
  (3.9), with their $G_T = 0$ as the default `bootstrap`.
-/
def discountedReturnsIntervals {n : Nat}
    (gamma : Value) (rewards : Tensor Value [n])
    (bootstrap : Value := (0 : Value)) :
    Tensor (Interval Value) [n] :=
  let gammaInterval := Binary.Interval.point gamma
  Tensor.scanr (fun reward future =>
    Binary.Interval.add
      (Binary.Interval.point reward)
      (Binary.Interval.mul gammaInterval future))
    (Binary.Interval.point bootstrap) rewards

/--
Outward-rounded intervals for fixed-horizon $\operatorname{GAE}(\lambda)$ with separate
termination and boundary masks, branching as `Checked.gae` does: a
terminated step drops the bootstrap value and a boundary stops the recursion, so no unused interval
is multiplied by zero.

It is a numerical diagnostic for the checked GAE and encloses the real recursion under the same
conditions as `discountedReturnsIntervals`.

Reference:
- Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation"
  (2015), ICLR 2016, eq. (16): https://arxiv.org/abs/1506.02438
-/
def gaeIntervals {n : Nat}
    (gamma lam : Value)
    (rewards values nextValues : Tensor Value [n])
    (terminated boundaries : Tensor Bool [n]) :
    Tensor (Interval Value) [n] :=
  let indices : Tensor (Fin n) [n] := Tensor.ofFn id
  let discount := Binary.Interval.mul (Binary.Interval.point gamma) (Binary.Interval.point lam)
  Tensor.scanr (fun idx advNext =>
    let delta := tdResidualInterval values[idx] rewards[idx] gamma nextValues[idx]
      terminated[idx]
    if boundaries[idx] then delta
    else Binary.Interval.add delta (Binary.Interval.mul discount advNext))
    (Binary.Interval.point 0) indices

/--
Executable check that every `returns[i]` lies inside `intervals[i]` in the configured scalar
order, used as a regression check in examples and tests. The interval soundness lemmas are
FloatLib's (`FloatLib.Floats.Formats.BinaryInterchange.Configured.Interval.Proof`).
-/
def returnsWithinIntervals {n : Nat}
    (returns : Tensor Value [n])
    (intervals : Tensor (Interval Value) [n]) : Bool :=
  (List.finRange n).all fun i =>
    let x : Value := Tensor.item (Tensor.unstack returns i)
    let I : Interval Value := Tensor.item (Tensor.unstack intervals i)
    Binary.Interval.leB I.lo x &&
      Binary.Interval.leB x I.hi

end Checked
end RL
end TorchLean
