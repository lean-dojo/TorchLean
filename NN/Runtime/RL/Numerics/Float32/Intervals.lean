/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.RL.Numerics.Float32.Types

/-!
# Float32 Interval Diagnostics for RL

This module uses FloatLib's outward-rounded binary32 intervals for return, GAE, TD-residual,
and PPO scalar formulas used by the checked binary32 runtime. These intervals are executable
diagnostics: they do not replace the exact RL specs, but they flag overflow, invalid endpoints, and
unstable recurrences in examples and regression tests.

References: IEEE 1788-2015 for interval arithmetic semantics; Sutton and Barto for return and TD
recurrences; Schulman et al. for GAE and PPO.
-/

@[expose] public section

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.ExecFloat (Binary)
open FloatLib.Numerics (Interval)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)

namespace Runtime
namespace RL
namespace Numerics
namespace Float32

open Spec TorchLean
open TorchLean TorchLean.Tensor
open Spec.RL


/-!
## Interval Enclosures (configured binary32 endpoint intervals)
-/

/--
Outward-rounded interval enclosure for the one-step discounted backup:

`reward + γ * (1-done) * bootstrap`.

This is the interval analogue of `discountedBackupChecked` (but purely functional, and
returning an enclosure rather than failing).

Reference:
- Sutton and Barto, *Reinforcement Learning: An Introduction* (discounted backups / returns).
-/
def discountedBackupInterval
    (reward gamma bootstrap : Binary 8 23) (done : Bool) : Interval (Binary 8 23) :=
  if done then
    Binary.Interval.point reward
  else
    let r : Interval (Binary 8 23) := Binary.Interval.point reward
    let prod : Interval (Binary 8 23) :=
      Binary.Interval.mul
        (Binary.Interval.point gamma)
        (Binary.Interval.point bootstrap)
    Binary.Interval.add r prod

/--
Outward-rounded interval enclosure for the TD residual:

`reward + γ * (1-done) * nextValue - value`.

This is the interval analogue of `tdResidualChecked`.

Reference:
- Sutton and Barto, *Reinforcement Learning: An Introduction* (TD error / Bellman error).
-/
def tdResidualInterval
    (value reward gamma nextValue : Binary 8 23) (done : Bool) : Interval (Binary 8 23) :=
  let target : Interval (Binary 8 23) := discountedBackupInterval reward gamma nextValue done
  Binary.Interval.sub target
    (Binary.Interval.point value)

/--
Outward-rounded interval enclosure for the PPO clipped surrogate objective from a precomputed ratio.

This is a **conservative hull enclosure**: it encloses both of the candidate products
`ratio * A` and `clippedRatio * A`, then returns their interval hull. The definition is simple and
still provides a useful non-finite/divergence detector for the PPO objective.

Reference:
- Schulman et al., "Proximal Policy Optimization Algorithms" (2017):
  https://arxiv.org/abs/1707.06347
-/
def ppoClippedObjectiveFromRatioInterval
    (ratio advantage clipEps : Binary 8 23) : Interval (Binary 8 23) :=
  let one : Binary 8 23 := (1 : Binary 8 23)
  -- Clipping thresholds are computed as float32 values (round-to-nearest). The main goal of this
  -- enclosure is to bound the subsequent products.
  let lo : Binary 8 23 := ExecFloat.sub one clipEps
  let hi : Binary 8 23 := ExecFloat.add one clipEps
  let clippedRatio : Binary 8 23 :=
    min hi
      (max lo ratio)
  let unclipped : Interval (Binary 8 23) :=
    Binary.Interval.mul
      (Binary.Interval.point ratio)
      (Binary.Interval.point advantage)
  let clipped : Interval (Binary 8 23) :=
    Binary.Interval.mul
      (Binary.Interval.point clippedRatio)
      (Binary.Interval.point advantage)
  Binary.Interval.hull unclipped clipped

/--
Outward-rounded interval enclosure for fixed-horizon discounted returns.

If you pass point intervals at the leaves (`Binary.Interval.point`), the output is a conservative
enclosure for the exact real return recursion (interpreting leaves via `Model.toReal` after
decoding).

This is an *executable* diagnostic: you can run it alongside `discountedReturnsChecked`
to detect blow-ups (endpoints becoming `±Inf` or `Valid` failing).

Reference:
- Sutton and Barto, *Reinforcement Learning: An Introduction* (returns / bootstrapping).
-/
def discountedReturnsIntervals {n : Nat}
    (gamma : Binary 8 23) (rewards : Tensor (Binary 8 23) [n])
    (bootstrap : Binary 8 23 := (0 : Binary 8 23)) :
    Tensor (Interval (Binary 8 23)) [n] :=
  let gammaInterval := Binary.Interval.point gamma
  Tensor.scanr (fun reward future =>
    Binary.Interval.add
      (Binary.Interval.point reward)
      (Binary.Interval.mul gammaInterval future))
    (Binary.Interval.point bootstrap) rewards

/--
Outward-rounded interval enclosure for fixed-horizon $\operatorname{GAE}(\lambda)$.

This is useful as a coarse numerical diagnostic alongside
`generalizedAdvantageEstimationChecked`.

Reference:
- Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation"
  (2015): https://arxiv.org/abs/1506.02438
-/
def generalizedAdvantageEstimationIntervals {n : Nat}
    (gamma lam : Binary 8 23)
    (rewards values nextValues : Tensor (Binary 8 23) [n])
    (dones : Tensor Bool [n]) :
    Tensor (Interval (Binary 8 23)) [n] :=
  let indices : Tensor (Fin n) [n] := Tensor.ofFn id
  let γ := Binary.Interval.point gamma
  let lamI := Binary.Interval.point lam
  Tensor.scanr (fun idx advNext =>
    let done := dones[idx]
    let mask : Interval (Binary 8 23) :=
      Binary.Interval.point
        (continueMask (α := Binary 8 23) done)
    let r : Interval (Binary 8 23) := Binary.Interval.point rewards[idx]
    let v : Interval (Binary 8 23) := Binary.Interval.point values[idx]
    let nv : Interval (Binary 8 23) := Binary.Interval.point nextValues[idx]

    -- delta = r + γ*mask*nv - v
    let t1 := Binary.Interval.mul γ mask
    let t2 := Binary.Interval.mul t1 nv
    let t3 := Binary.Interval.add r t2
    let delta := Binary.Interval.sub t3 v
    -- adv = delta + γ*λ*mask*advNext
    let u1 := Binary.Interval.mul γ lamI
    let u2 := Binary.Interval.mul u1 mask
    let u3 := Binary.Interval.mul u2 advNext
    let adv := Binary.Interval.add delta u3
    adv) (Binary.Interval.point 0) indices

/--
Executable check: every `returns[i]` lies inside `intervals[i]` in the configured scalar order.

This is an executable regression check for examples and tests; formal enclosure theorems live in
`NN/Floats/Interval/*`.
-/
def returnsWithinIntervals {n : Nat}
    (returns : Tensor (Binary 8 23) [n])
    (intervals : Tensor (Interval (Binary 8 23)) [n]) : Bool :=
  (List.finRange n).all fun i =>
    let x : Binary 8 23 := Tensor.item (get returns i)
    let I : Interval (Binary 8 23) := Tensor.item (get intervals i)
    Binary.Interval.leB I.lo x &&
      Binary.Interval.leB x I.hi

end Float32
end Numerics
end RL
end Runtime
