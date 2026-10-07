/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.RL
public import NN.API.RL.Runtime
public import NN.API.Seeded
public import NN.Runtime.Autograd.Model.Autodiff
public import NN.Runtime.RL.DQN.Autograd
public import NN.Tests.Utils

/-!
# RL Runtime Checks

Runtime checks for termination versus truncation, binary32 overflow and interval endpoints,
PPO clamp and zero-probability behavior, strict JSON integers, and actor/critic state selection.
These exercise floating-point and external-input cases not settled by exact-real RL identities.
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

namespace Tests
namespace Floats
namespace RLCheck

/-! ## DQN target detachment and Huber gradients

Real-valued Huber identities do not check tape detachment or native cancellation. These cases
exercise both Float and Float32, including action masking and residuals below unit roundoff.
-/

namespace DQN

open Spec TorchLean Runtime.Autograd.Model

def expect (label : String) (condition : Bool) : IO Unit := do
  unless condition do throw <| IO.userError s!"DQN loss check failed: {label}"

def close {α : Type} [Context α] (actual expected : α) : Bool :=
  Context.gtBool (1 / 100000 : α) (MathFunctions.abs (actual - expected))

/-- Detach the Bellman target while testing the two Huber reduction schedules. -/
def vectorLoss (delta : Rat) (reduction : Loss.Reduction) :
    ∀ {β : Type}, [TorchLean.Storage β] → [Context β] →
      Program β [Shape.ofList [7], Shape.ofList [7]] [] :=
  fun {β} _ _ => fun {m} _ _ => fun prediction target =>
    Runtime.RL.DQN.Autograd.huber (m := m) (α := β)
      prediction target (delta : β) reduction

/-- One-hot action selection must not send gradients into actions or target values. -/
def actionLoss :
    ∀ {β : Type}, [TorchLean.Storage β] → [Context β] →
      Program β [Shape.ofList [2, 3], Shape.ofList [2, 3], Shape.ofList [2]] [] :=
  fun {β} _ _ => fun {m} _ _ => fun prediction actions target =>
    Runtime.RL.DQN.Autograd.loss (m := m) (α := β)
      prediction actions target

/-- Cover both tails, the joins, and the quadratic branch with one residual tensor. -/
def checkVector {α : Type} [TorchLean.Storage α] [Context α] : IO Unit := do
  let prediction : Tensor α [7] := [-4, -2, -1, 0, 1, 2, 4]
  let target : Tensor α [7] := Tensor.zeros [7]
  for reduction in [Loss.Reduction.sum, .mean] do
    let divisor : α := if reduction == .mean then 7 else 1
    let graph ← Autodiff.lowerToTypedGraph (α := α)
      (paramShapes := [Shape.ofList [7]]) (inputShapes := [Shape.ofList [7]])
      (vectorLoss 2 reduction)
    let (gradients, loss) ← Autodiff.Impl.pullback graph
      (.cons prediction (.cons target .nil)) (Tensor.scalar (1 : α))
    expect "Huber value with delta two" (close (loss.getFlat ⟨0, by decide⟩) (17 / divisor))
    let expected := ([-2, -2, -1, 0, 1, 2, 2] : Tensor α [7]).map (· / divisor)
    for i in List.finRange 7 do
      expect "Huber prediction gradient"
        (close ((gradients.get ⟨0, by decide⟩).getFlat i) (expected.getFlat i))
      expect "Bellman target is detached" (close ((gradients.get ⟨1, by decide⟩).getFlat i) 0)

/-- Unselected action values cannot affect the transition mean or receive gradients. -/
def checkActions {α : Type} [TorchLean.Storage α] [Context α] : IO Unit := do
  let prediction : Tensor α [2, 3] := [[-3, 100, -100], [100, 1 / 2, -100]]
  let actions : Tensor α [2, 3] := [[1, 0, 0], [0, 1, 0]]
  let target : Tensor α [2] := Tensor.zeros [2]
  let graph ← Autodiff.lowerToTypedGraph (α := α)
    (paramShapes := [Shape.ofList [2, 3]])
    (inputShapes := [Shape.ofList [2, 3], Shape.ofList [2]]) actionLoss
  let (gradients, loss) ← Autodiff.Impl.pullback graph
    (.cons prediction (.cons actions (.cons target .nil))) (Tensor.scalar (1 : α))
  expect "mean over transitions" (close (loss.getFlat ⟨0, by decide⟩) (21 / 16))
  let expected : Tensor α [2, 3] := [[-1 / 2, 0, 0], [0, 1 / 4, 0]]
  for i in List.finRange 6 do
    expect "only selected actions receive gradients"
      (close ((gradients.get ⟨0, by decide⟩).getFlat i) (expected.getFlat i))
    expect "action encoding is detached" (close ((gradients.get ⟨1, by decide⟩).getFlat i) 0)
  for i in List.finRange 2 do
    expect "selected-action target is detached" (close ((gradients.get ⟨2, by decide⟩).getFlat i) 0)

/-- Small quadratic-region residuals must retain their derivative without cancellation. -/
def checkSmallResidual {α : Type} [TorchLean.Storage α] [Context α] : IO Unit := do
  let small : α := 1 / 1073741824
  let prediction : Tensor α [7] := Tensor.ofFn fun _ => small
  let target : Tensor α [7] := Tensor.zeros [7]
  let (gradient, _) ← Autodiff.gradients (α := α)
    (paramShapes := [Shape.ofList [7]]) (inputShapes := [Shape.ofList [7]])
    (vectorLoss 1 .sum) (.cons prediction .nil) (.cons target .nil)
  for i in List.finRange 7 do
    expect "small residual derivative is preserved"
      ((gradient.get ⟨0, by decide⟩).getFlat i == small)

def run : IO Unit := do
  checkVector (α := Float)
  checkActions (α := Float)
  checkSmallResidual (α := Float)
  checkVector (α := Float32)
  checkActions (α := Float32)
  checkSmallResidual (α := Float32)
  IO.println "DQN semi-gradient Huber checks passed (Float, Float32)."


end DQN

/-- Assert a boolean runtime condition with a labeled failure message. -/
def assertBool (msg : String) (b : Bool) : IO Unit := do
  if !b then
    throw <| IO.userError msg

/-- Widen a configured binary32 value through its binary64 model for host assertions. -/
def toHostFloat (value : Binary 8 23) : Float :=
  Binary.toFloat (ofModel (Model.cast .binary32 .binary64 (toModel value)))

/-- Check discounted returns in host, executable float32, and interval semantics. -/
def checkReturns : IO Unit := do
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

/-- Replay sampling wraps correctly and truncation retains the bootstrap term. -/
def checkReplay : IO Unit := do
  let qPred : Tensor Float [3] := [1.0, 2.0, 0.5]
  let qNext : Tensor Float [3] := [0.1, 1.4, 0.3]
  let dqnLoss := (2.0 - 2.26) * (2.0 - 2.26)

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
    Runtime.RL.DQN.loss (α := Float) onlineQ targetQ 0.9 replayBatch (batch := true)
  assertApprox "replay dqn minibatch loss" replayLoss dqnLoss 1e-6
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
    (Runtime.RL.DQN.loss (α := Float) onlineQ targetQ 0.9 #[truncatedTr] (batch := true))
    dqnLoss 1e-6
  assertApprox "terminated replay dqn loss"
    (Runtime.RL.DQN.loss (α := Float) onlineQ targetQ 0.9 #[terminatedTr] (batch := true))
    ((2.0 - 1.0) * (2.0 - 1.0)) 1e-6

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
      let objective ← Runtime.RL.PolicyGradient.Autograd.PPO.objective (m := m) (α := β)
        (batch := 2) (nActions := 3) logits actions oldLogProb advantage
      Runtime.Autograd.Model.F.mean (m := m) (α := β) (s := .dim 2 .scalar) objective :
      m (Runtime.Autograd.Model.Ref (m := m) (α := β) Shape.scalar))

/-- Within the one-hot log-probability clamp interval, collection and autograd reductions agree
and the PPO ratio is exactly 1 at identical parameters. The second sample picks an action whose
probability is far below the separate probability clamp used by `logProbability`. -/
def checkPPORatioAtIdenticalParams : IO Unit := do
  let rows : Array (Array Float) := #[#[0.0, 1.0, -0.5], #[0.0, 30.0, -30.0]]
  let chosen : Array (Fin 3) := #[⟨1, by decide⟩, ⟨2, by decide⟩]
  let logits : Tensor Float [2, 3] :=
    (Tensor.ofFn fun i : Fin 6 => (rows[i.val / 3]!)[i.val % 3]!).reshape [2, 3] (by decide)
  let actions : Tensor Float [2, 3] :=
    (Tensor.ofFn fun i : Fin 6 =>
      if chosen[i.val / 3]!.val == i.val % 3 then (1 : Float) else 0).reshape [2, 3] (by decide)
  let oldLogProb : Tensor Float [2] := Tensor.ofFn fun i =>
    Runtime.RL.PolicyGradient.logSoftmax (α := Float)
      (Tensor.ofFn fun j : Fin 3 => (rows[i.val]!)[j.val]!) chosen[i.val]!
  let advantage : Tensor Float [2] := Tensor.ofFn fun _ => 1
  let graph ← Runtime.Autograd.Model.Autodiff.lowerToTypedGraph (α := Float)
    (paramShapes := [Shape.ofList [2, 3]])
    (inputShapes := [Shape.ofList [2, 3], Shape.ofList [2], Shape.ofList [2]]) ppoMeanObjective
  let (_, objective) ← Runtime.Autograd.Model.Autodiff.Impl.pullback graph
    (.cons logits (.cons actions (.cons oldLogProb (.cons advantage .nil))))
    (Tensor.scalar (1 : Float))
  -- With ratio 1 and unit advantages every per-sample objective is exactly 1.
  let value := objective.getFlat ⟨0, by decide⟩
  assertBool s!"PPO objective at identical parameters should be 1, got {value}" (value == 1)

/-- Outside the one-hot clamp interval, a raw cached tail log-probability need not give
unit importance ratio at identical parameters. Unit advantage clips the objective to 1.2. -/
def checkPPOTailOutsideClamp : IO Unit := do
  let logits : Tensor Float [2, 3] :=
    (Tensor.ofFn fun i : Fin 6 => if i.val % 3 == 1 then -1.0e40 else 0.0).reshape
      [2, 3] (by decide)
  let actions : Tensor Float [2, 3] :=
    (Tensor.ofFn fun i : Fin 6 => if i.val % 3 == 1 then (1 : Float) else 0).reshape
      [2, 3] (by decide)
  let rawLogProb := Runtime.RL.PolicyGradient.logSoftmax (α := Float)
    (Tensor.ofFn fun i : Fin 3 => if i.val == 1 then -1.0e40 else 0.0) ⟨1, by decide⟩
  assertBool "PPO cached tail lies outside the one-hot clamp" (rawLogProb < -1.0e30)
  let oldLogProb : Tensor Float [2] := Tensor.full [2] rawLogProb
  let advantage : Tensor Float [2] := Tensor.full [2] 1
  let graph ← Runtime.Autograd.Model.Autodiff.lowerToTypedGraph (α := Float)
    (paramShapes := [Shape.ofList [2, 3]])
    (inputShapes := [Shape.ofList [2, 3], Shape.ofList [2], Shape.ofList [2]]) ppoMeanObjective
  let (_, objective) ← Runtime.Autograd.Model.Autodiff.Impl.pullback graph
    (.cons logits (.cons actions (.cons oldLogProb (.cons advantage .nil))))
    (Tensor.scalar (1 : Float))
  assertApprox "PPO raw cached tail gives clipped nonunit objective" objective.item 1.2 1e-12

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
    Runtime.RL.PolicyGradient.logSoftmax (α := Float)
      (Tensor.ofFn fun j : Fin 3 => (rows[i.val]!)[j.val]!) chosen[i.val]!
  let advantage : Tensor Float [2] := Tensor.ofFn fun _ => 1
  let graph ← Runtime.Autograd.Model.Autodiff.lowerToTypedGraph (α := Float)
    (paramShapes := [Shape.ofList [2, 3]])
    (inputShapes := [Shape.ofList [2, 3], Shape.ofList [2], Shape.ofList [2]]) ppoMeanObjective
  let (gradients, objective) ← Runtime.Autograd.Model.Autodiff.Impl.pullback graph
    (.cons logits (.cons actions (.cons oldLogProb (.cons advantage .nil))))
    (Tensor.scalar (1 : Float))
  let value := objective.getFlat ⟨0, by decide⟩
  assertBool s!"PPO objective with a -inf logit should be 1, got {value}" (value == 1)
  let .cons logitGradient _ := gradients
  assertBool "PPO logit gradient with a -inf logit should be finite"
    (Tensor.allSpec (fun g : Float => g.isFinite) logitGradient)

/-- Strict parsing compares the original decimal, including fractional tails lost by Float. -/
def checkStrictNaturalJson : IO Unit := do
  for (source, expected) in [("0", 0), ("1.00", 1), ("1e3", 1000),
      ("9007199254740992", 9007199254740992),
      ("18446744073709549568", 18446744073709549568),
      ("18446744073709551615", 18446744073709551615)] do
    let json ← IO.ofExcept (Lean.Json.parse source)
    let parsed ← IO.ofExcept (Runtime.RL.Boundary.parseNatStrict json)
    assertBool s!"strict natural JSON {source}" (parsed == expected)
  for source in ["1.5", "1.00000000000000001", "9007199254740993",
      "18446744073709551616", "-1", "-1e-400", "1e-400"] do
    let json ← IO.ofExcept (Lean.Json.parse source)
    match Runtime.RL.Boundary.parseNatStrict json with
    | .ok value => throw <| IO.userError s!"strict natural JSON accepted {source} as {value}"
    | .error _ => pure ()

/-- Entropy objective for the zero-probability and finite-tail regressions. -/
def entropyObjective :
    ∀ {β : Type}, [TorchLean.Storage β] → [Context β] →
      Runtime.Autograd.Model.Program β [Shape.ofList [1, 2]] [] :=
  fun {β} _ _ => fun {m} _ _ => fun logits =>
    Runtime.RL.PolicyGradient.Autograd.entropy (m := m) (α := β) logits

/-- A zero-probability action contributes zero to entropy and its logit gradient. -/
def checkEntropyZeroProbability : IO Unit := do
  let graph ← Runtime.Autograd.Model.Autodiff.lowerToTypedGraph (α := Float)
    (paramShapes := [Shape.ofList [1, 2]]) (inputShapes := []) entropyObjective
  for tail in [-(1.0 / 0.0), -1000.0, -1.0e40, 0.0] do
    let logits : Tensor Float [1, 2] :=
      (Tensor.ofFn fun i : Fin 2 => if i.val == 0 then 0.0 else tail).reshape [1, 2] (by decide)
    let (gradients, entropy) ← Runtime.Autograd.Model.Autodiff.Impl.pullback graph
      (.cons logits .nil) (Tensor.scalar (1 : Float))
    let expected := if tail == 0 then Float.log 2.0 else 0.0
    assertBool "entropy must be finite" entropy.item.isFinite
    assertApprox "entropy with zero or equal probabilities" entropy.item expected 1e-12
    let .cons gradient .nil := gradients
    assertBool "entropy gradient must be finite and zero"
      (Tensor.allSpec (fun g : Float => g.isFinite && Float.abs g < 1e-12) gradient)

/-- A carrier's negative infinity yields zero entropy and a zero logit gradient. -/
def checkEntropyAtNegativeInfinity {α : Type} [TorchLean.Storage α] [Context α]
    (label : String) (negativeInfinity : α) (isZero : α → Bool) : IO Unit := do
  let graph ← Runtime.Autograd.Model.Autodiff.lowerToTypedGraph (α := α)
    (paramShapes := [Shape.ofList [1, 2]]) (inputShapes := []) entropyObjective
  let logits : Tensor α [1, 2] :=
    (Tensor.ofFn fun i : Fin 2 => if i.val == 0 then 0 else negativeInfinity).reshape
      [1, 2] (by decide)
  let (gradients, entropy) ← Runtime.Autograd.Model.Autodiff.Impl.pullback graph
    (.cons logits .nil) (Tensor.scalar (1 : α))
  assertBool s!"{label} zero-probability entropy" (isZero entropy.item)
  let .cons gradient .nil := gradients
  assertBool s!"{label} zero-probability entropy VJP" (Tensor.allSpec isZero gradient)

-- Already-lowered policies only read typed graphs and stored scalars.
example {α : Type} [Storage α] (graph : nn.TypedGraphModel [] [] [] α) :
    Tensor α [] → α :=
  rl.ppo.value graph (.id []) (.id []) nn.State.empty

example {α : Type} [Storage α] (graph : nn.TypedGraphModel [] [] [] α) :
    Tensor α [] → Tensor α [] :=
  rl.ppo.policy graph (.id []) (.id []) nn.State.empty

def checkShape (shape : Shape) (oneValue : shape.size = 1) : IO Unit := do
  let model : nn.Sequential shape shape := .id shape
  let graph ← nn.lowerToTypedGraph model (α := Float)
  let value := rl.ppo.value graph model model nn.State.empty (oneValue := oneValue)
  for bits in (#[0x0000000000000000, 0x8000000000000000, 0x3ff4000000000000,
      0xc004000000000000, 0x7ff0000000000000, 0xfff0000000000000,
      0x7ff8000000000042, 0xfff8000000000042] : Array UInt64) do
    -- Float.ofBits canonicalizes NaNs; extraction must preserve the constructed scalar.
    let scalar := Float.ofBits bits
    let input := Tensor.full shape scalar
    let actual := (value input).toBits
    assertBool s!"shape {shape}: expected scalar bits {scalar.toBits}, got {actual}"
      (actual == scalar.toBits)

def checkStateSplit : IO Unit := do
  let model := nn.build 0 (nn.linear 1 1)
  let graph ← nn.lowerToTypedGraph model (α := Float)
  let actorState : nn.State Float (nn.stateShapes model) := nn.State.full 50
  let criticState : nn.State Float (nn.stateShapes model) := nn.State.full 2
  let value := rl.ppo.value graph model model (actorState.append criticState)
  let input : Tensor Float [1] := [4]
  assertBool "critic reads its state after the actor state" (value input == 10)

/-- The scalar adapter cannot accept empty or multi-element outputs. -/
example : ¬ (Shape.size [0] = 1) ∧ ¬ (Shape.size [1, 2, 1] = 1) := by decide

/-- Run the complete RL runtime check suite. -/
def run : IO Unit := do
  IO.println "rl_check: begin"
  checkShape [] (by decide)
  checkShape [1] (by decide)
  checkShape [1, 1] (by decide)
  checkShape [1, 1, 1, 1, 1] (by decide)
  checkStateSplit
  checkReturns
  checkCheckedScans
  checkAdvantages
  checkReplay
  checkBoundary
  checkStrictNaturalJson
  checkEntropyZeroProbability
  checkEntropyAtNegativeInfinity "native binary32" (Float32.ofBits 0xff800000) (· == 0)
  checkEntropyAtNegativeInfinity "executable binary32"
    (Binary.infinity true : Binary 8 23) Binary.isZero
  checkPPORatioAtIdenticalParams
  checkPPOTailOutsideClamp
  checkPPONegativeInfinityLogit
  checkPPOClipThresholdEnclosure
  DQN.run
  IO.println "rl_check: ok"

end RLCheck
end Floats
end Tests
