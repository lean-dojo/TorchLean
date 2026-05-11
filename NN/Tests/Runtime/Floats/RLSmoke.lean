/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.RL
public import NN.Tests.Runtime.Floats.Utils
public import Std

/-!
# RLSmoke

Small compile-and-run smoke tests for TorchLean's RL helper surface.
-/

@[expose] public section

open Spec
open Tensor
open Tests.Floats.Utils

namespace Tests
namespace Floats
namespace RLSmoke

def assertBool (msg : String) (b : Bool) : IO Unit := do
  if !b then
    throw <| IO.userError msg

def run : IO Unit := do
  IO.println "rl_smoke: begin"

  let returns := Spec.RL.discountedReturns (α := Float) 0.5 [1.0, 2.0, 3.0]
  match returns with
  | [g0, g1, g2] =>
      assertApprox "discounted return[0]" g0 2.75 1e-6
      assertApprox "discounted return[1]" g1 3.5 1e-6
      assertApprox "discounted return[2]" g2 3.0 1e-6
  | _ => throw <| IO.userError "discounted returns length mismatch"

  let rewardsVec : Tensor Float (.dim 3 .scalar) := Spec.fromList1d [1.0, 2.0, 3.0]
  let returnsVec := Runtime.RL.Core.discountedReturnsVec (α := Float) 0.5 rewardsVec
  assertApprox "discountedReturnsVec[0]" (Tensor.vecGet returnsVec ⟨0, by decide⟩) 2.75 1e-6
  assertApprox "discountedReturnsVec[1]" (Tensor.vecGet returnsVec ⟨1, by decide⟩) 3.5 1e-6
  assertApprox "discountedReturnsVec[2]" (Tensor.vecGet returnsVec ⟨2, by decide⟩) 3.0 1e-6

  -- Run the same return recursion in executable float32 semantics (`IEEE32Exec`), with:
  -- 1) a checked Float→float32 cast (catches binary64→binary32 overflow), and
  -- 2) a simple interval enclosure (`Interval32`) as a diagnostic.
  let gamma32 : Runtime.RL.Numerics.Float32.Float32Exec ←
    match Runtime.RL.Numerics.Float32.ofFloatIEEE32ExecChecked 0.5 with
    | .ok g => pure g
    | .error e => throw <| IO.userError e
  let rewards32 : Tensor Runtime.RL.Numerics.Float32.Float32Exec (.dim 3 .scalar) ←
    match Runtime.RL.Numerics.Float32.castTensorIEEE32ExecChecked (s := .dim 3 .scalar) rewardsVec with
    | .ok t => pure t
    | .error e => throw <| IO.userError e
  let returns32 : Tensor Runtime.RL.Numerics.Float32.Float32Exec (.dim 3 .scalar) ←
    match Runtime.RL.Numerics.Float32.discountedReturnsVecFromIEEE32ExecChecked (n := 3) gamma32 rewards32 with
    | .ok t => pure t
    | .error e => throw <| IO.userError e
  assertApprox "discountedReturnsVec IEEE32Exec[0]"
    (TorchLean.Floats.IEEE754.IEEE32Exec.toFloat (Tensor.vecGet returns32 ⟨0, by decide⟩)) 2.75 1e-5
  assertApprox "discountedReturnsVec IEEE32Exec[1]"
    (TorchLean.Floats.IEEE754.IEEE32Exec.toFloat (Tensor.vecGet returns32 ⟨1, by decide⟩)) 3.5 1e-5
  assertApprox "discountedReturnsVec IEEE32Exec[2]"
    (TorchLean.Floats.IEEE754.IEEE32Exec.toFloat (Tensor.vecGet returns32 ⟨2, by decide⟩)) 3.0 1e-5
  let intervals32 : Tensor Runtime.RL.Numerics.Float32.Interval32 (.dim 3 .scalar) :=
    Runtime.RL.Numerics.Float32.discountedReturnsIntervals32 (n := 3) gamma32 rewards32
  assertBool "interval enclosure should contain IEEE32Exec returns"
    (Runtime.RL.Numerics.Float32.returnsWithinIntervals32 (n := 3) returns32 intervals32)

  -- Sanity: a huge binary64 value should be rejected by the checked Float→float32 cast.
  let huge : Float := 1e100
  match Runtime.RL.Numerics.Float32.ofFloatIEEE32ExecChecked huge with
  | .ok _ => throw <| IO.userError "expected Float→IEEE32Exec cast to reject huge value"
  | .error _ => pure ()

  let gaeRewards : Tensor Float (.dim 3 .scalar) := Spec.fromList1d [1.0, 1.0, 1.0]
  let gaeValues : Tensor Float (.dim 3 .scalar) := fill 0 (.dim 3 .scalar)
  let gaeNext : Tensor Float (.dim 3 .scalar) := fill 0 (.dim 3 .scalar)
  let gaeDones : Tensor Bool (.dim 3 .scalar) := Spec.fromList1d [false, false, false]

  -- Run PPO-relevant transforms (TD residual, GAE, z-score normalization, PPO clip objective).
  let one32 : Runtime.RL.Numerics.Float32.Float32Exec ←
    match Runtime.RL.Numerics.Float32.ofFloatIEEE32ExecChecked 1.0 with
    | .ok x => pure x
    | .error e => throw <| IO.userError e
  let lam32 : Runtime.RL.Numerics.Float32.Float32Exec ←
    match Runtime.RL.Numerics.Float32.ofFloatIEEE32ExecChecked 1.0 with
    | .ok x => pure x
    | .error e => throw <| IO.userError e
  let tdRes32 : Runtime.RL.Numerics.Float32.Float32Exec ←
    match Runtime.RL.Numerics.Float32.tdResidualIEEE32ExecChecked (value := 0) (reward := one32) (gamma := gamma32)
        (nextValue := 0) (done := false) with
    | .ok x => pure x
    | .error e => throw <| IO.userError e
  assertApprox "tdResidual IEEE32Exec"
    (TorchLean.Floats.IEEE754.IEEE32Exec.toFloat tdRes32) 1.0 1e-5

  let gaeRewards32 : Tensor Runtime.RL.Numerics.Float32.Float32Exec (.dim 3 .scalar) ←
    match Runtime.RL.Numerics.Float32.castTensorIEEE32ExecChecked (s := .dim 3 .scalar) gaeRewards with
    | .ok t => pure t
    | .error e => throw <| IO.userError e
  let gaeValues32 : Tensor Runtime.RL.Numerics.Float32.Float32Exec (.dim 3 .scalar) ←
    match Runtime.RL.Numerics.Float32.castTensorIEEE32ExecChecked (s := .dim 3 .scalar) gaeValues with
    | .ok t => pure t
    | .error e => throw <| IO.userError e
  let gaeNext32 : Tensor Runtime.RL.Numerics.Float32.Float32Exec (.dim 3 .scalar) ←
    match Runtime.RL.Numerics.Float32.castTensorIEEE32ExecChecked (s := .dim 3 .scalar) gaeNext with
    | .ok t => pure t
    | .error e => throw <| IO.userError e

  let advantages32 : Tensor Runtime.RL.Numerics.Float32.Float32Exec (.dim 3 .scalar) ←
    match Runtime.RL.Numerics.Float32.generalizedAdvantageEstimationVecIEEE32ExecChecked (n := 3)
        (gamma := gamma32) (lam := lam32) gaeRewards32 gaeValues32 gaeNext32 gaeDones with
    | .ok t => pure t
    | .error e => throw <| IO.userError e
  assertApprox "gaeVec IEEE32Exec[0]"
    (TorchLean.Floats.IEEE754.IEEE32Exec.toFloat (Tensor.vecGet advantages32 ⟨0, by decide⟩)) 1.75 1e-4
  assertApprox "gaeVec IEEE32Exec[1]"
    (TorchLean.Floats.IEEE754.IEEE32Exec.toFloat (Tensor.vecGet advantages32 ⟨1, by decide⟩)) 1.5 1e-4
  assertApprox "gaeVec IEEE32Exec[2]"
    (TorchLean.Floats.IEEE754.IEEE32Exec.toFloat (Tensor.vecGet advantages32 ⟨2, by decide⟩)) 1.0 1e-4

  let normIn32 : Tensor Runtime.RL.Numerics.Float32.Float32Exec (.dim 3 .scalar) ←
    match Runtime.RL.Numerics.Float32.castTensorIEEE32ExecChecked (s := .dim 3 .scalar) rewardsVec with
    | .ok t => pure t
    | .error e => throw <| IO.userError e
  let normed32 : Tensor Runtime.RL.Numerics.Float32.Float32Exec (.dim 3 .scalar) ←
    match Runtime.RL.Numerics.Float32.normalizeZScoreIEEE32ExecChecked (n := 3) normIn32 with
    | .ok t => pure t
    | .error e => throw <| IO.userError e
  -- Mean-centered input has a 0 entry; after z-score it should remain 0 (finite).
  assertApprox "zscore IEEE32Exec[1]"
    (TorchLean.Floats.IEEE754.IEEE32Exec.toFloat (Tensor.vecGet normed32 ⟨1, by decide⟩)) 0.0 1e-6

  let ratio32 : Runtime.RL.Numerics.Float32.Float32Exec ←
    match Runtime.RL.Numerics.Float32.ofFloatIEEE32ExecChecked 1.5 with
    | .ok x => pure x
    | .error e => throw <| IO.userError e
  let clipEps32 : Runtime.RL.Numerics.Float32.Float32Exec ←
    match Runtime.RL.Numerics.Float32.ofFloatIEEE32ExecChecked 0.2 with
    | .ok x => pure x
    | .error e => throw <| IO.userError e
  let ppoObj32 : Runtime.RL.Numerics.Float32.Float32Exec ←
    match Runtime.RL.Numerics.Float32.ppoClippedObjectiveFromRatioIEEE32ExecChecked ratio32 one32 clipEps32 with
    | .ok x => pure x
    | .error e => throw <| IO.userError e
  assertApprox "ppoClipFromRatio IEEE32Exec"
    (TorchLean.Floats.IEEE754.IEEE32Exec.toFloat ppoObj32) 1.2 1e-5

  let advantagesVec :=
    Runtime.RL.Core.generalizedAdvantageEstimationVec (α := Float) 0.5 1.0 gaeRewards gaeValues gaeNext gaeDones
  assertApprox "gaeVec[0]" (Tensor.vecGet advantagesVec ⟨0, by decide⟩) 1.75 1e-6
  assertApprox "gaeVec[1]" (Tensor.vecGet advantagesVec ⟨1, by decide⟩) 1.5 1e-6
  assertApprox "gaeVec[2]" (Tensor.vecGet advantagesVec ⟨2, by decide⟩) 1.0 1e-6
  let returnsFromAdv := Runtime.RL.Core.returnsFromAdvantagesVec (α := Float) advantagesVec gaeValues
  assertApprox "returnsFromAdvantagesVec[0]" (Tensor.vecGet returnsFromAdv ⟨0, by decide⟩) 1.75 1e-6

  let bandit0 : Runtime.RL.Bandits.ValueState Float 3 :=
    { counts := fill 0 (.dim 3 .scalar)
      values := fill 0 (.dim 3 .scalar) }
  let bandit1 := Runtime.RL.Bandits.sampleAverageStep bandit0 ⟨1, by decide⟩ 4.0
  assertApprox "bandit count" (Tensor.vecGet bandit1.counts ⟨1, by decide⟩) 1.0 1e-6
  assertApprox "bandit value" (Tensor.vecGet bandit1.values ⟨1, by decide⟩) 4.0 1e-6
  let greedy := Runtime.RL.Bandits.greedyAction? bandit1
  assertBool "bandit greedy action should be arm 1" (greedy = some ⟨1, by decide⟩)

  let q0 : Tensor Float (.dim 2 (.dim 2 .scalar)) := fill 0 (.dim 2 (.dim 2 .scalar))
  let q1 :=
    Runtime.RL.Tabular.qLearningUpdate q0 ⟨0, by decide⟩ ⟨1, by decide⟩ 1.0 ⟨1, by decide⟩ 0.9 0.5
  assertApprox "q-learning update"
    (get2 q1 ⟨0, by decide⟩ ⟨1, by decide⟩) 0.5 1e-6

  let qPred : Tensor Float (.dim 3 .scalar) :=
    Spec.fromList1d [1.0, 2.0, 0.5]
  let qNext : Tensor Float (.dim 3 .scalar) :=
    Spec.fromList1d [0.1, 1.4, 0.3]
  let dqnTarget := Runtime.RL.ValueLearning.dqnTarget (α := Float) 1.0 0.9 false qNext
  assertApprox "dqn target" dqnTarget 2.26 1e-6
  let dqnLoss := Runtime.RL.ValueLearning.dqnMSELoss qPred ⟨1, by decide⟩ 1.0 0.9 false qNext
  assertApprox "dqn mse loss" dqnLoss ((2.0 - 2.26) * (2.0 - 2.26)) 1e-6

  -- Replay + minibatch DQN layer: store typed transitions, sample deterministically, and compute
  -- the same DQN loss through caller-provided Q-functions.
  let obs2 : Tensor Float (.dim 2 .scalar) := Spec.fromList1d [0.0, 1.0]
  let nextObs2 : Tensor Float (.dim 2 .scalar) := Spec.fromList1d [1.0, 0.0]
  let tr0 : Runtime.RL.Core.Transition Float (.dim 2 .scalar) 3 :=
    { state := obs2
      action := ⟨1, by decide⟩
      reward := 1.0
      nextState := nextObs2
      done := false }
  let rb0 : Runtime.RL.Replay.Buffer Float (.dim 2 .scalar) 3 :=
    Runtime.RL.Replay.Buffer.empty 4
  let rb1 := rb0.push tr0
  let replayBatch := rb1.sampleContiguous 0 2
  assertBool "replay sample should wrap over one stored transition" (replayBatch.size == 2)
  let onlineQ (_ : Tensor Float (.dim 2 .scalar)) : Tensor Float (.dim 3 .scalar) := qPred
  let targetQ (_ : Tensor Float (.dim 2 .scalar)) : Tensor Float (.dim 3 .scalar) := qNext
  let replayLoss :=
    Runtime.RL.DQN.minibatchMSELoss (α := Float) onlineQ targetQ 0.9 replayBatch
  assertApprox "replay dqn minibatch loss" replayLoss dqnLoss 1e-6
  let soft := Runtime.RL.DQN.softUpdateScalar (α := Float) 0.1 10.0 0.0
  assertApprox "soft target update" soft 1.0 1e-6

  let logits : Tensor Float (.dim 2 .scalar) := Spec.fromList1d [0.0, 1.0]
  let logp := Runtime.RL.PolicyGradient.actionLogProbability (α := Float) logits ⟨1, by decide⟩
  assertBool "log-prob should be finite" (!Float.isNaN logp)
  let ppoObj := Runtime.RL.PolicyGradient.ppoClippedObjective (α := Float) logits ⟨1, by decide⟩
    (-0.2) 1.5 0.2
  assertBool "ppo objective should be finite" (!Float.isNaN ppoObj)
  let klSame := Runtime.RL.PolicyGradient.categoricalKLFromLogits (α := Float) logits logits
  assertApprox "categorical KL same policy" klSame 0.0 1e-6
  let a2cLoss := Runtime.RL.PolicyGradient.a2cLoss (α := Float) logits ⟨1, by decide⟩
    1.0 0.2 0.5 1.0 0.01
  assertBool "a2c loss should be finite" (!Float.isNaN a2cLoss)
  let qForPolicy : Tensor Float (.dim 2 .scalar) := Spec.fromList1d [0.1, 0.8]
  let sacActor := Runtime.RL.PolicyGradient.sacCategoricalActorLoss (α := Float)
    logits qForPolicy ⟨1, by decide⟩ 0.2
  assertBool "sac categorical actor loss should be finite" (!Float.isNaN sacActor)

  -- Boundary-contract sanity: validate a small discrete-action transition.
  let c0 : Runtime.RL.Boundary.Contract (.dim 2 .scalar) 3 := {}
  match Runtime.RL.Boundary.checkTransition (obsShape := .dim 2 .scalar) (nActions := 3) c0
      obs2 nextObs2 1 0.0 false false with
  | .ok _ => pure ()
  | .error e => throw <| IO.userError s!"boundary check should accept valid transition: {e}"

  match Runtime.RL.Boundary.checkTransition (obsShape := .dim 2 .scalar) (nActions := 3) c0
      obs2 nextObs2 3 0.0 false false with
  | .ok _ => throw <| IO.userError "boundary check should reject out-of-range action"
  | .error _ => pure ()

  let nanReward : Float := (0.0 / 0.0)
  match Runtime.RL.Boundary.checkTransition (obsShape := .dim 2 .scalar) (nActions := 3) c0
      obs2 nextObs2 1 nanReward false false with
  | .ok _ => throw <| IO.userError "boundary check should reject NaN reward"
  | .error _ => pure ()

  let cExclusive : Runtime.RL.Boundary.Contract (.dim 2 .scalar) 3 :=
    { requireExclusiveDoneFlags := true }
  match Runtime.RL.Boundary.checkTransition (obsShape := .dim 2 .scalar) (nActions := 3) cExclusive
      obs2 nextObs2 1 0.0 true true with
  | .ok _ => throw <| IO.userError "boundary check should reject terminated && truncated"
  | .error _ => pure ()

  IO.println "rl_smoke: ok"

end RLSmoke
end Floats
end Tests
