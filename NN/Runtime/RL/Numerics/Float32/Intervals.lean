/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.RL.Numerics.Float32.PPO

/-!
# Float32 Interval Diagnostics for RL

This module contains outward-rounded `Interval32` enclosures for the same return, GAE, TD-residual,
and PPO scalar formulas used by the checked binary32 runtime. These intervals are executable
diagnostics: they do not replace the exact RL specs, but they flag overflow, invalid endpoints, and
unstable recurrences in examples and regression tests.

References: IEEE 1788-2015 for interval arithmetic semantics; Sutton and Barto for return and TD
recurrences; Schulman et al. for GAE and PPO.
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
## Interval Enclosures (IEEE32Exec endpoint intervals)
-/

/--
Outward-rounded interval enclosure for the one-step discounted backup:

`reward + γ * (1-done) * bootstrap`.

This is the interval analogue of `discountedBackupIEEE32ExecChecked` (but purely functional, and
returning an enclosure rather than failing).

Reference:
- Sutton and Barto, *Reinforcement Learning: An Introduction* (discounted backups / returns).
-/
def discountedBackupInterval32
    (reward gamma bootstrap : Float32Exec) (done : Bool) : Interval32 :=
  if done then
    TorchLean.Floats.IEEE754.IEEE32Exec.Interval32.point reward
  else
    let r : Interval32 := TorchLean.Floats.IEEE754.IEEE32Exec.Interval32.point reward
    let prod : Interval32 :=
      TorchLean.Floats.IEEE754.IEEE32Exec.Interval32.mul
        (TorchLean.Floats.IEEE754.IEEE32Exec.Interval32.point gamma)
        (TorchLean.Floats.IEEE754.IEEE32Exec.Interval32.point bootstrap)
    TorchLean.Floats.IEEE754.IEEE32Exec.Interval32.add r prod

/--
Outward-rounded interval enclosure for the TD residual:

`reward + γ * (1-done) * nextValue - value`.

This is the interval analogue of `tdResidualIEEE32ExecChecked`.

Reference:
- Sutton and Barto, *Reinforcement Learning: An Introduction* (TD error / Bellman error).
-/
def tdResidualInterval32
    (value reward gamma nextValue : Float32Exec) (done : Bool) : Interval32 :=
  let target : Interval32 := discountedBackupInterval32 reward gamma nextValue done
  TorchLean.Floats.IEEE754.IEEE32Exec.Interval32.sub target
    (TorchLean.Floats.IEEE754.IEEE32Exec.Interval32.point value)

/--
Outward-rounded interval enclosure for the PPO clipped surrogate objective from a precomputed ratio.

This is a **conservative hull enclosure**: it encloses both of the candidate products
`ratio * A` and `clippedRatio * A`, then returns their interval hull. This keeps the definition
simple while still providing a useful non-finite/divergence detector for the PPO objective.

Reference:
- Schulman et al., "Proximal Policy Optimization Algorithms" (2017): https://arxiv.org/abs/1707.06347
-/
def ppoClippedObjectiveFromRatioInterval32
    (ratio advantage clipEps : Float32Exec) : Interval32 :=
  let one : Float32Exec := (1 : Float32Exec)
  -- Clipping thresholds are computed as float32 values (round-to-nearest). The main goal of this
  -- enclosure is to bound the subsequent products.
  let lo : Float32Exec := TorchLean.Floats.IEEE754.IEEE32Exec.sub one clipEps
  let hi : Float32Exec := TorchLean.Floats.IEEE754.IEEE32Exec.add one clipEps
  let clippedRatio : Float32Exec :=
    TorchLean.Floats.IEEE754.IEEE32Exec.minimum hi (TorchLean.Floats.IEEE754.IEEE32Exec.maximum lo ratio)
  let unclipped : Interval32 :=
    TorchLean.Floats.IEEE754.IEEE32Exec.Interval32.mul
      (TorchLean.Floats.IEEE754.IEEE32Exec.Interval32.point ratio)
      (TorchLean.Floats.IEEE754.IEEE32Exec.Interval32.point advantage)
  let clipped : Interval32 :=
    TorchLean.Floats.IEEE754.IEEE32Exec.Interval32.mul
      (TorchLean.Floats.IEEE754.IEEE32Exec.Interval32.point clippedRatio)
      (TorchLean.Floats.IEEE754.IEEE32Exec.Interval32.point advantage)
  TorchLean.Floats.IEEE754.IEEE32Exec.Interval32.hull unclipped clipped

/--
Outward-rounded interval enclosure for fixed-horizon discounted returns.

If you pass point intervals at the leaves (`Interval32.point`), the output is a conservative
enclosure for the exact real return recursion (interpreting leaves via `IEEE32Exec.toReal`).

This is an *executable* diagnostic: you can run it alongside `discountedReturnsVecFromIEEE32ExecChecked`
to detect blow-ups (endpoints becoming `±Inf` or `Valid` failing).

Reference:
- Sutton and Barto, *Reinforcement Learning: An Introduction* (returns / bootstrapping).
-/
def discountedReturnsIntervals32 {n : Nat}
    (gamma : Float32Exec) (rewards : Tensor Float32Exec (.dim n .scalar))
    (bootstrap : Float32Exec := (0 : Float32Exec)) :
    Tensor Interval32 (.dim n .scalar) :=
  let rArr : Array Float32Exec :=
    Array.ofFn (fun i : Fin n => Tensor.toScalar (get rewards i))
  let out : Array Interval32 :=
    Id.run do
      let mut out : Array Interval32 :=
        Array.replicate n (TorchLean.Floats.IEEE754.IEEE32Exec.Interval32.point 0)
      let mut g : Interval32 := TorchLean.Floats.IEEE754.IEEE32Exec.Interval32.point bootstrap
      let γ : Interval32 := TorchLean.Floats.IEEE754.IEEE32Exec.Interval32.point gamma
      for t in [0:n] do
        let idx := n - 1 - t
        let r : Interval32 := TorchLean.Floats.IEEE754.IEEE32Exec.Interval32.point (rArr[idx]!)
        g :=
          TorchLean.Floats.IEEE754.IEEE32Exec.Interval32.add r
            (TorchLean.Floats.IEEE754.IEEE32Exec.Interval32.mul γ g)
        out := out.set! idx g
      return out
  Tensor.dim (fun i : Fin n => Tensor.scalar (out[i.val]!))

/--
Outward-rounded interval enclosure for fixed-horizon GAE(λ).

This is useful as a coarse “numerical sanity” diagnostic alongside
`generalizedAdvantageEstimationVecIEEE32ExecChecked`.

Reference:
- Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation"
  (2015): https://arxiv.org/abs/1506.02438
-/
def generalizedAdvantageEstimationIntervals32 {n : Nat}
    (gamma lam : Float32Exec)
    (rewards values nextValues : Tensor Float32Exec (.dim n .scalar))
    (dones : Tensor Bool (.dim n .scalar)) :
    Tensor Interval32 (.dim n .scalar) :=
  let rArr : Array Float32Exec :=
    Array.ofFn (fun i : Fin n => Tensor.toScalar (get rewards i))
  let vArr : Array Float32Exec :=
    Array.ofFn (fun i : Fin n => Tensor.toScalar (get values i))
  let nvArr : Array Float32Exec :=
    Array.ofFn (fun i : Fin n => Tensor.toScalar (get nextValues i))
  let dArr : Array Bool :=
    Array.ofFn (fun i : Fin n => Tensor.toScalar (get dones i))

  let out : Array Interval32 :=
    Id.run do
      let mut out : Array Interval32 :=
        Array.replicate n (TorchLean.Floats.IEEE754.IEEE32Exec.Interval32.point 0)
      let mut advNext : Interval32 := TorchLean.Floats.IEEE754.IEEE32Exec.Interval32.point 0
      let γ : Interval32 := TorchLean.Floats.IEEE754.IEEE32Exec.Interval32.point gamma
      let lamI : Interval32 := TorchLean.Floats.IEEE754.IEEE32Exec.Interval32.point lam
      for t in [0:n] do
        let idx := n - 1 - t
        let done := dArr[idx]!
        let mask : Interval32 :=
          TorchLean.Floats.IEEE754.IEEE32Exec.Interval32.point (continueMask (α := Float32Exec) done)
        let r : Interval32 := TorchLean.Floats.IEEE754.IEEE32Exec.Interval32.point (rArr[idx]!)
        let v : Interval32 := TorchLean.Floats.IEEE754.IEEE32Exec.Interval32.point (vArr[idx]!)
        let nv : Interval32 := TorchLean.Floats.IEEE754.IEEE32Exec.Interval32.point (nvArr[idx]!)

        -- delta = r + γ*mask*nv - v
        let t1 := TorchLean.Floats.IEEE754.IEEE32Exec.Interval32.mul γ mask
        let t2 := TorchLean.Floats.IEEE754.IEEE32Exec.Interval32.mul t1 nv
        let t3 := TorchLean.Floats.IEEE754.IEEE32Exec.Interval32.add r t2
        let delta := TorchLean.Floats.IEEE754.IEEE32Exec.Interval32.sub t3 v
        -- adv = delta + γ*λ*mask*advNext
        let u1 := TorchLean.Floats.IEEE754.IEEE32Exec.Interval32.mul γ lamI
        let u2 := TorchLean.Floats.IEEE754.IEEE32Exec.Interval32.mul u1 mask
        let u3 := TorchLean.Floats.IEEE754.IEEE32Exec.Interval32.mul u2 advNext
        let adv := TorchLean.Floats.IEEE754.IEEE32Exec.Interval32.add delta u3
        advNext := adv
        out := out.set! idx adv
      return out

  Tensor.dim (fun i : Fin n => Tensor.scalar (out[i.val]!))

/--
Executable check: every `returns[i]` lies inside `intervals[i]` in the `IEEE32Exec.le` order.

This is an executable regression check for examples and tests; formal enclosure theorems live in
`NN/Floats/Interval/*`.
-/
def returnsWithinIntervals32 {n : Nat}
    (returns : Tensor Float32Exec (.dim n .scalar))
    (intervals : Tensor Interval32 (.dim n .scalar)) : Bool :=
  let idxs : Array (Fin n) := Array.ofFn (fun i => i)
  idxs.all fun i =>
    let x : Float32Exec := Tensor.toScalar (get returns i)
    let I : Interval32 := Tensor.toScalar (get intervals i)
    TorchLean.Floats.IEEE754.IEEE32Exec.Interval32.leB I.lo x &&
      TorchLean.Floats.IEEE754.IEEE32Exec.Interval32.leB x I.hi

end Float32
end Numerics
end RL
end Runtime
