/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.RL
public import NN.Tests.Runtime.Floats.DQN
public import NN.Tests.Runtime.Floats.Utils

/-!
# RL Runtime Checks

Small compile-and-run runtime checks for TorchLean's RL helper surface.
-/

@[expose] public section

open FloatLib.Floats (ExecFloat)
open FloatLib.Numerics (Interval)
open FloatLib.Floats.ExecFloat (Binary)
open FloatLib.Floats.ExecFloat.Binary (ofModel toModel)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)

open Spec TorchLean
open TorchLean TorchLean.Tensor
open Tests.Utils
open Tests.Floats.Utils

namespace Tests
namespace Floats
namespace RLCheck

/-- Assert a boolean runtime condition with a labeled failure message. -/
def assertBool (msg : String) (b : Bool) : IO Unit := do
  if !b then
    throw <| IO.userError msg

/-- Check discounted returns in host, executable float32, and interval semantics. -/
def checkReturns : IO Unit := do
  let toHostFloat (value : Binary 8 23) : Float :=
    Binary.toFloat (ofModel (Model.cast .binary32 .binary64 (toModel value)))
  let rewards : Tensor Float [3] := [1.0, 2.0, 3.0]
  let checkTensorReturns (tensorReturns : Tensor Float [3]) : IO Unit := do
    assertApprox "discountedReturns[0]"
      (Tensor.getScalar tensorReturns ⟨0, by decide⟩) 2.75 1e-6
    assertApprox "discountedReturns[1]"
      (Tensor.getScalar tensorReturns ⟨1, by decide⟩) 3.5 1e-6
    assertApprox "discountedReturns[2]"
      (Tensor.getScalar tensorReturns ⟨2, by decide⟩) 3.0 1e-6
  checkTensorReturns <| Runtime.RL.Core.discountedReturns (α := Float) 0.5 rewards

  -- Run the same return recursion in executable float32 semantics (`Binary 8 23`), with:
  -- 1) a checked Float→float32 cast (catches binary64→binary32 overflow), and
  -- 2) a FloatLib interval enclosure as a diagnostic.
  let gamma32 : Binary 8 23 ←
    match Runtime.RL.Numerics.Float32.ofFloatChecked 0.5 with
    | .ok g => pure g
    | .error e => throw <| IO.userError e
  let rewards32 : Tensor (Binary 8 23) [3] ←
    match Runtime.RL.Numerics.Float32.castTensorChecked (s := [3]) rewards with
    | .ok t => pure t
    | .error e => throw <| IO.userError e
  let returns32 : Tensor (Binary 8 23) [3] ←
    match Runtime.RL.Numerics.Float32.discountedReturnsChecked (n := 3) gamma32 rewards32 with
    | .ok t => pure t
    | .error e => throw <| IO.userError e
  assertApprox "discountedReturns configured binary32[0]"
    (toHostFloat (Tensor.getScalar returns32 ⟨0, by decide⟩))
    2.75 1e-5
  assertApprox "discountedReturns configured binary32[1]"
    (toHostFloat (Tensor.getScalar returns32 ⟨1, by decide⟩))
    3.5 1e-5
  assertApprox "discountedReturns configured binary32[2]"
    (toHostFloat (Tensor.getScalar returns32 ⟨2, by decide⟩))
    3.0 1e-5
  let intervals32 : Tensor (Interval (Binary 8 23)) [3] :=
    Runtime.RL.Numerics.Float32.discountedReturnsIntervals (n := 3) gamma32 rewards32
  assertBool "interval enclosure should contain configured binary32 returns"
    (Runtime.RL.Numerics.Float32.returnsWithinIntervals (n := 3) returns32 intervals32)

  -- A huge binary64 value should be rejected by the checked Float→float32 cast.
  let huge : Float := 1e100
  match Runtime.RL.Numerics.Float32.ofFloatChecked huge with
  | .ok _ => throw <| IO.userError "expected Float→configured binary32 cast to reject huge value"
  | .error _ => pure ()

open Runtime.RL.Numerics.Float32 in
/-- Checked scans preserve empty horizons, termination resets, and failure order. -/
def checkCheckedScans : IO Unit := do
  let emptyRewards : Tensor (Binary 8 23) [0] := []
  match discountedReturnsChecked (1 / 2) emptyRewards with
  | .ok _ => pure ()
  | .error error => throw (IO.userError s!"empty checked returns: {error}")
  let rewards : Tensor (Binary 8 23) [3] := [1, 2, 3]
  let baseline : Tensor (Binary 8 23) [3] := [0, 0, 0]
  let dones : Tensor Bool [3] := [false, true, false]
  let advantages ← match generalizedAdvantageEstimationChecked
      (1 / 2) 1 rewards baseline baseline dones with
    | .ok values => pure values
    | .error error => throw (IO.userError error)
  let expected : Tensor (Binary 8 23) [3] := [2, 2, 3]
  assertBool "terminal GAE step resets the future advantage" (advantages == expected)
  -- A truncated step bootstraps from its next value but does not continue into the next episode.
  let nextValues : Tensor (Binary 8 23) [3] := [5, 5, 5]
  let noTermination : Tensor Bool [3] := [false, false, false]
  let truncated ← IO.ofExcept <| generalizedAdvantageEstimationWithBoundariesChecked
    (1 / 2) 1 rewards baseline nextValues noTermination dones
  let expectedTruncated : Tensor (Binary 8 23) [3] := [5.75, 4.5, 5.5]
  assertBool "truncated GAE step bootstraps from its next value" (truncated == expectedTruncated)
  let terminal ← IO.ofExcept <| generalizedAdvantageEstimationWithBoundariesChecked
    (1 / 2) 1 rewards baseline nextValues dones dones
  let singleMask ← IO.ofExcept <| generalizedAdvantageEstimationChecked
    (1 / 2) 1 rewards baseline nextValues dones
  let expectedTerminal : Tensor (Binary 8 23) [3] := [4.5, 2, 5.5]
  assertBool "terminal GAE step drops its next value" (terminal == expectedTerminal)
  assertBool "single-mask GAE matches equal masks" (singleMask == terminal)
  let intervals := generalizedAdvantageEstimationIntervals
    (1 / 2) 1 rewards baseline baseline dones
  assertBool "terminal GAE interval enclosure" (returnsWithinIntervals advantages intervals)
  let invalid : Tensor (Binary 8 23) [2] :=
    [(Binary.infinity false : Binary 8 23),
     (Binary.maxFinite false : Binary 8 23)]
  match discountedReturnsChecked 2 invalid
      (Binary.maxFinite false : Binary 8 23) with
  | .ok _ => throw (IO.userError "checked returns accepted multiplication overflow")
  | .error error =>
      assertBool "rightmost overflow is reported before the earlier nonfinite reward"
        (error.contains "mul")

open Runtime.RL.Numerics.Float32 in
/-- The PPO objective interval encloses the exact clipped product when `1 ± clipEps` rounds. -/
def checkPPOClipThresholdEnclosure : IO Unit := do
  let toHostFloat (value : Binary 8 23) : Float :=
    Binary.toFloat (ofModel (Model.cast .binary32 .binary64 (toModel value)))
  -- `0.1` is not a binary32 value, and `1 ± eps` is exact in binary64 but not in binary32.
  let eps ← IO.ofExcept <| ofFloatChecked 0.1
  let two ← IO.ofExcept <| ofFloatChecked 2.0
  let zero ← IO.ofExcept <| ofFloatChecked 0.0
  let one ← IO.ofExcept <| ofFloatChecked 1.0
  let contains (label : String) (I : Interval (Binary 8 23)) (exact : Float) : IO Unit :=
    assertBool s!"{label}: [{toHostFloat I.lo}, {toHostFloat I.hi}] misses {exact}"
      (toHostFloat I.lo ≤ exact && exact ≤ toHostFloat I.hi)
  contains "upper clip threshold"
    (ppoClippedObjectiveFromRatioInterval two one eps) (1.0 + toHostFloat eps)
  contains "lower clip threshold"
    (ppoClippedObjectiveFromRatioInterval zero one eps) (1.0 - toHostFloat eps)

/-- Check the numerical transforms used by PPO advantage estimation. -/
def checkAdvantages : IO Unit := do
  let toHostFloat (value : Binary 8 23) : Float :=
    Binary.toFloat (ofModel (Model.cast .binary32 .binary64 (toModel value)))
  let rewards : Tensor Float [3] := [1.0, 2.0, 3.0]
  let gaeRewards : Tensor Float [3] := [1.0, 1.0, 1.0]
  let gaeValues : Tensor Float [3] := Tensor.full [3] (0 : Float)
  let gaeNext : Tensor Float [3] := Tensor.full [3] (0 : Float)
  let gaeDones : Tensor Bool [3] :=
    [false, false, false]

  let gamma32 ←
    IO.ofExcept <| Runtime.RL.Numerics.Float32.ofFloatChecked 0.5

  -- Run PPO-relevant transforms (TD residual, GAE, z-score normalization, PPO clip objective).
  let one32 ←
    IO.ofExcept <| Runtime.RL.Numerics.Float32.ofFloatChecked 1.0
  let lam32 ←
    IO.ofExcept <| Runtime.RL.Numerics.Float32.ofFloatChecked 1.0
  let tdRes32 ←
    IO.ofExcept <| Runtime.RL.Numerics.Float32.tdResidualChecked
      (value := 0) (reward := one32) (gamma := gamma32) (nextValue := 0) (done := false)
  assertApprox "tdResidual configured binary32"
    (toHostFloat tdRes32) 1.0 1e-5

  let gaeRewards32 ←
    IO.ofExcept <| Runtime.RL.Numerics.Float32.castTensorChecked (s := [3]) gaeRewards
  let gaeValues32 ←
    IO.ofExcept <| Runtime.RL.Numerics.Float32.castTensorChecked (s := [3]) gaeValues
  let gaeNext32 ←
    IO.ofExcept <| Runtime.RL.Numerics.Float32.castTensorChecked (s := [3]) gaeNext

  let advantages32 ←
    IO.ofExcept <| Runtime.RL.Numerics.Float32.generalizedAdvantageEstimationChecked (n := 3)
      (gamma := gamma32) (lam := lam32) gaeRewards32 gaeValues32 gaeNext32 gaeDones
  assertApprox "gaeTensor configured binary32[0]"
    (toHostFloat (Tensor.getScalar advantages32 ⟨0, by decide⟩))
    1.75 1e-4
  assertApprox "gaeTensor configured binary32[1]"
    (toHostFloat (Tensor.getScalar advantages32 ⟨1, by decide⟩))
    1.5 1e-4
  assertApprox "gaeTensor configured binary32[2]"
    (toHostFloat (Tensor.getScalar advantages32 ⟨2, by decide⟩))
    1.0 1e-4

  let normIn32 ←
    IO.ofExcept <| Runtime.RL.Numerics.Float32.castTensorChecked (s := [3]) rewards
  let normed32 ←
    IO.ofExcept <| Runtime.RL.Numerics.Float32.normalizeZScoreChecked (n := 3) normIn32
  -- Mean-centered input has a 0 entry; after z-score it should remain 0 (finite).
  assertApprox "zscore configured binary32[1]"
    (toHostFloat (Tensor.getScalar normed32 ⟨1, by decide⟩))
    0.0 1e-6

  let ratio32 ←
    IO.ofExcept <| Runtime.RL.Numerics.Float32.ofFloatChecked 1.5
  let clipEps32 ←
    IO.ofExcept <| Runtime.RL.Numerics.Float32.ofFloatChecked 0.2
  let ppoObj32 ←
    IO.ofExcept <| Runtime.RL.Numerics.Float32.ppoClippedObjectiveFromRatioChecked
      ratio32 one32 clipEps32
  assertApprox "ppoClipFromRatio configured binary32"
    (toHostFloat ppoObj32) 1.2 1e-5

  let advantages :=
    Runtime.RL.Core.generalizedAdvantageEstimation (α := Float)
      0.5 1.0 gaeRewards gaeValues gaeNext gaeDones
  assertApprox "gaeTensor[0]" (Tensor.getScalar advantages ⟨0, by decide⟩) 1.75 1e-6
  assertApprox "gaeTensor[1]" (Tensor.getScalar advantages ⟨1, by decide⟩) 1.5 1e-6
  assertApprox "gaeTensor[2]" (Tensor.getScalar advantages ⟨2, by decide⟩) 1.0 1e-6
  let returnsFromAdv :=
    Runtime.RL.Core.returnsFromAdvantages (α := Float) advantages gaeValues
  assertApprox "returnsFromAdvantages[0]"
    (Tensor.getScalar returnsFromAdv ⟨0, by decide⟩) 1.75 1e-6

/-- Check bandit, tabular, DQN, and policy-gradient runtime helpers. -/
def checkValueLearning : IO Unit := do
  let bandit0 : Runtime.RL.Bandits.ValueState Float 3 :=
    { counts := Tensor.full [3] (0 : Float)
      values := Tensor.full [3] (0 : Float) }
  let bandit1 := Runtime.RL.Bandits.sampleAverageStep bandit0 ⟨1, by decide⟩ 4.0
  assertApprox "bandit count" (Tensor.getScalar bandit1.counts ⟨1, by decide⟩) 1.0 1e-6
  assertApprox "bandit value" (Tensor.getScalar bandit1.values ⟨1, by decide⟩) 4.0 1e-6
  let greedy := Runtime.RL.Bandits.greedyAction? bandit1
  assertBool "bandit greedy action should be arm 1" (greedy = some ⟨1, by decide⟩)

  let q0 : Tensor Float [2, 2] := Tensor.full [2, 2] (0 : Float)
  let q1 :=
    Runtime.RL.Tabular.qLearningUpdate q0 ⟨0, by decide⟩ ⟨1, by decide⟩ 1.0 ⟨1, by decide⟩ 0.9 0.5
  assertApprox "q-learning update"
    (get2 q1 ⟨0, by decide⟩ ⟨1, by decide⟩) 0.5 1e-6

  let qPred : Tensor Float [3] := [1.0, 2.0, 0.5]
  let qNext : Tensor Float [3] := [0.1, 1.4, 0.3]
  let dqnTarget := Runtime.RL.ValueLearning.dqnTarget (α := Float) 1.0 0.9 false qNext
  assertApprox "dqn target" dqnTarget 2.26 1e-6
  let dqnLoss := Runtime.RL.ValueLearning.dqnMSELoss qPred ⟨1, by decide⟩ 1.0 0.9 false qNext
  assertApprox "dqn mse loss" dqnLoss ((2.0 - 2.26) * (2.0 - 2.26)) 1e-6

  -- Replay + minibatch DQN layer: store typed transitions, sample deterministically, and compute
  -- the same DQN loss through caller-provided Q-functions.
  let obs2 : Tensor Float [2] := [0.0, 1.0]
  let nextObs2 : Tensor Float [2] := [1.0, 0.0]
  let tr0 : Runtime.RL.Core.Transition Float [2] 3 :=
    { state := obs2
      action := ⟨1, by decide⟩
      reward := 1.0
      nextState := nextObs2
      done := false }
  let rb0 : Runtime.RL.Replay.Buffer Float [2] 3 :=
    Runtime.RL.Replay.Buffer.empty 4
  let rb1 := rb0.push tr0
  let replayBatch := rb1.sampleContiguous 0 2
  assertBool "replay sample should wrap over one stored transition" (replayBatch.size == 2)
  let onlineQ (_ : Tensor Float [2]) : Tensor Float [3] := qPred
  let targetQ (_ : Tensor Float [2]) : Tensor Float [3] := qNext
  let replayLoss :=
    Runtime.RL.DQN.minibatchMSELoss (α := Float) onlineQ targetQ 0.9 replayBatch
  assertApprox "replay dqn minibatch loss" replayLoss dqnLoss 1e-6
  let soft := Runtime.RL.DQN.softUpdateScalar (α := Float) 0.1 10.0 0.0
  assertApprox "soft target update" soft 1.0 1e-6

  -- A time-limit truncation keeps the bootstrap term; only termination drops it.
  let observed (terminated truncated : Bool) :
      Spec.RL.ObservedTransition (Tensor Float [2]) (Fin 3) Float :=
    { observation := obs2, action := ⟨1, by decide⟩, reward := 1.0
      nextObservation := nextObs2, terminated := terminated, truncated := truncated }
  let truncatedTr := Runtime.RL.Replay.ofObservedTransition (observed false true)
  let terminatedTr := Runtime.RL.Replay.ofObservedTransition (observed true false)
  assertBool "truncated replay transition should bootstrap" (!truncatedTr.done)
  assertBool "terminated replay transition should not bootstrap" terminatedTr.done
  assertApprox "truncated replay dqn loss"
    (Runtime.RL.DQN.minibatchMSELoss (α := Float) onlineQ targetQ 0.9 #[truncatedTr])
    dqnLoss 1e-6
  assertApprox "terminated replay dqn loss"
    (Runtime.RL.DQN.minibatchMSELoss (α := Float) onlineQ targetQ 0.9 #[terminatedTr])
    ((2.0 - 1.0) * (2.0 - 1.0)) 1e-6

  let logits : Tensor Float [2] := [0.0, 1.0]
  let logp := Runtime.RL.PolicyGradient.actionLogProbability (α := Float) logits ⟨1, by decide⟩
  assertFinite "action log-probability" logp
  let ppoObj := Runtime.RL.PolicyGradient.ppoClippedObjective (α := Float) logits ⟨1, by decide⟩
    (-0.2) 1.5 0.2
  assertFinite "ppo objective" ppoObj
  let klSame := Runtime.RL.PolicyGradient.categoricalKLFromLogits (α := Float) logits logits
  assertApprox "categorical KL same policy" klSame 0.0 1e-6
  let a2cLoss := Runtime.RL.PolicyGradient.a2cLoss (α := Float) logits ⟨1, by decide⟩
    1.0 0.2 0.5 1.0 0.01
  assertFinite "a2c loss" a2cLoss
  let qForPolicy : Tensor Float [2] := [0.1, 0.8]
  let sacActor := Runtime.RL.PolicyGradient.sacCategoricalActorLoss (α := Float)
    logits qForPolicy 0.2
  assertFinite "sac categorical actor loss" sacActor

/-- Check validation at an external RL environment boundary. -/
def checkBoundary : IO Unit := do
  let obs2 : Tensor Float [2] := [0.0, 1.0]
  let nextObs2 : Tensor Float [2] := [1.0, 0.0]

  -- Boundary-contract check: validate a small discrete-action transition.
  let c0 : Runtime.RL.Boundary.Contract [2] 3 := {}
  match Runtime.RL.Boundary.checkTransition (obsShape := [2]) (nActions := 3) c0
      obs2 nextObs2 1 0.0 false false with
  | .ok _ => pure ()
  | .error e => throw <| IO.userError s!"boundary check should accept valid transition: {e}"

  match Runtime.RL.Boundary.checkTransition (obsShape := [2]) (nActions := 3) c0
      obs2 nextObs2 3 0.0 false false with
  | .ok _ => throw <| IO.userError "boundary check should reject out-of-range action"
  | .error _ => pure ()

  let nanReward : Float := (0.0 / 0.0)
  match Runtime.RL.Boundary.checkTransition (obsShape := [2]) (nActions := 3) c0
      obs2 nextObs2 1 nanReward false false with
  | .ok _ => throw <| IO.userError "boundary check should reject NaN reward"
  | .error _ => pure ()

  let cExclusive : Runtime.RL.Boundary.Contract [2] 3 :=
    { requireExclusiveDoneFlags := true }
  match Runtime.RL.Boundary.checkTransition (obsShape := [2]) (nActions := 3) cExclusive
      obs2 nextObs2 1 0.0 true true with
  | .ok _ => throw <| IO.userError "boundary check should reject terminated && truncated"
  | .error _ => pure ()

/-- Mean PPO clipped objective for a two-sample batch of three-action logits. -/
def ppoMeanObjective :
    ∀ {β : Type}, [TorchLean.Storage β] → [Context β] →
      Runtime.Autograd.Model.Program β
        [Shape.ofList [2, 3], Shape.ofList [2, 3], Shape.ofList [2], Shape.ofList [2]] [] :=
  fun {β} _ _ => fun {m} _ _ => fun logits actions oldLogProb advantage =>
    (do
      let objective ← Runtime.RL.PolicyGradient.Autograd.ppoClippedObjective (m := m) (α := β)
        (batch := 2) (nActions := 3) logits actions oldLogProb advantage
      Runtime.Autograd.Model.F.mean (m := m) (α := β) (s := .dim 2 .scalar) objective :
      m (Runtime.Autograd.Model.RefTy (m := m) (α := β) Shape.scalar))

/-- Old log-probabilities recorded during collection must match the autograd objective's new
log-probabilities, so the PPO ratio is exactly 1 at identical parameters. The second sample picks
an action whose probability is far below the clamp used by `actionLogProbability`. -/
def checkPPORatioAtIdenticalParams : IO Unit := do
  let rows : Array (Array Float) := #[#[0.0, 1.0, -0.5], #[0.0, 30.0, -30.0]]
  let chosen : Array (Fin 3) := #[⟨1, by decide⟩, ⟨2, by decide⟩]
  let logits : Tensor Float [2, 3] :=
    (Tensor.ofFn fun i : Fin 6 => (rows[i.val / 3]!)[i.val % 3]!).reshape [2, 3] (by decide)
  let actions : Tensor Float [2, 3] :=
    (Tensor.ofFn fun i : Fin 6 =>
      if chosen[i.val / 3]!.val == i.val % 3 then (1 : Float) else 0).reshape [2, 3] (by decide)
  let oldLogProb : Tensor Float [2] := Tensor.ofFn fun i =>
    Runtime.RL.PolicyGradient.actionLogSoftmax (α := Float)
      (Tensor.ofFn fun j : Fin 3 => (rows[i.val]!)[j.val]!) chosen[i.val]!
  let advantage : Tensor Float [2] := Tensor.ofFn fun _ => 1
  let graph ← Runtime.Autograd.Model.Autodiff.lowerScalarToTypedGraph (α := Float)
    (paramShapes := [Shape.ofList [2, 3]])
    (inputShapes := [Shape.ofList [2, 3], Shape.ofList [2], Shape.ofList [2]]) ppoMeanObjective
  let (_, objective) ← Runtime.Autograd.Model.Autodiff.Impl.vjpWithValue graph
    (.cons logits (.cons actions (.cons oldLogProb (.cons advantage .nil))))
    (Tensor.scalar (1 : Float))
  -- With ratio 1 and unit advantages every per-sample objective is exactly 1.
  let value := objective.getFlat ⟨0, by decide⟩
  assertBool s!"PPO objective at identical parameters should be 1, got {value}" (value == 1)

/-- A logit of `-inf` on an unselected action must not turn the one-hot log-probability into NaN.
The objective stays exactly 1 at identical parameters and the logit gradient stays finite. -/
def checkPPONegativeInfinityLogit : IO Unit := do
  let negInf : Float := -1.0 / 0.0
  let rows : Array (Array Float) := #[#[0.0, 1.0, negInf], #[negInf, 0.0, 2.0]]
  let chosen : Array (Fin 3) := #[⟨1, by decide⟩, ⟨2, by decide⟩]
  let logits : Tensor Float [2, 3] :=
    (Tensor.ofFn fun i : Fin 6 => (rows[i.val / 3]!)[i.val % 3]!).reshape [2, 3] (by decide)
  let actions : Tensor Float [2, 3] :=
    (Tensor.ofFn fun i : Fin 6 =>
      if chosen[i.val / 3]!.val == i.val % 3 then (1 : Float) else 0).reshape [2, 3] (by decide)
  let oldLogProb : Tensor Float [2] := Tensor.ofFn fun i =>
    Runtime.RL.PolicyGradient.actionLogSoftmax (α := Float)
      (Tensor.ofFn fun j : Fin 3 => (rows[i.val]!)[j.val]!) chosen[i.val]!
  let advantage : Tensor Float [2] := Tensor.ofFn fun _ => 1
  let graph ← Runtime.Autograd.Model.Autodiff.lowerScalarToTypedGraph (α := Float)
    (paramShapes := [Shape.ofList [2, 3]])
    (inputShapes := [Shape.ofList [2, 3], Shape.ofList [2], Shape.ofList [2]]) ppoMeanObjective
  let (gradients, objective) ← Runtime.Autograd.Model.Autodiff.Impl.vjpWithValue graph
    (.cons logits (.cons actions (.cons oldLogProb (.cons advantage .nil))))
    (Tensor.scalar (1 : Float))
  let value := objective.getFlat ⟨0, by decide⟩
  assertBool s!"PPO objective with a -inf logit should be 1, got {value}" (value == 1)
  let .cons logitGradient _ := gradients
  assertBool "PPO logit gradient with a -inf logit should be finite"
    (Tensor.allSpec (fun g : Float => g.isFinite) logitGradient)

/-- Run the complete RL runtime check suite. -/
def run : IO Unit := do
  IO.println "rl_check: begin"
  checkReturns
  checkCheckedScans
  checkAdvantages
  checkValueLearning
  checkBoundary
  checkPPORatioAtIdenticalParams
  checkPPONegativeInfinityLogit
  checkPPOClipThresholdEnclosure
  DQN.run
  IO.println "rl_check: ok"

end RLCheck
end Floats
end Tests
