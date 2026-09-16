/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

End-to-end PPO example: train an actor-critic on a Lean-native GridWorld environment.
-/

module

public import NN.API
public import NN.Examples.Support
public import NN.Runtime.RL.Artifacts.GridWorld
public import NN.Runtime.RL.Artifacts.DefaultPaths
public import NN.Spec.RL.Envs.GridWorld
public import NN.Proofs.RL.Envs.GridWorld
import Mathlib.Tactic.NormNum.Ineq

/-!
# PPO on Lean-native GridWorld (Executable Example + Formal Model)

This example complements `torchlean ppo_cartpole`:

- `torchlean ppo_cartpole` uses an external Python Gymnasium environment and checks every step
  against a Lean side trust-boundary contract (`Runtime.RL.Boundary.Contract`).
- This example uses a **Lean-native** GridWorld and still runs the same PPO update in Lean.

Even though the environment is defined in Lean, we *still* validate every transition with the
boundary checker. That keeps the data model unified: downstream training code consumes
`Spec.RL.ObservedTransition` in a single format regardless of whether the source is a Lean-native
environment or an external sampler.

## Formal hooks

1. The environment has an induced finite stochastic MDP (`Spec.RL.FiniteStochastic.MDP`) and we
   import a proof that it is well-formed (row-stochastic transition rows, $0\le\gamma<1$).
2. The boundary checker can be turned into a Prop-level hypothesis via
   `Proofs.RL.Boundary.contractHolds_of_checkTransitionFin_eq_ok`
   (see `NN/Proofs/RL/Boundary.lean`), or you can use the proof-layer Gymnasium checked step
   `Runtime.RL.Gymnasium.Session.stepCheckedWithProof`
   (`NN/Proofs/RL/Gymnasium.lean`) for external environments.

## CLI flags

- `--device cuda`: run the Torch backend on CUDA (requires building with `-K cuda=true`).
- `--updates <n>`: number of PPO updates to run.
- `--eval-every <n>`: evaluate the greedy policy every `n` updates.
- `--eval-episodes <n>`: number of evaluation episodes per checkpoint.
- `--eval-max-steps <n>`: maximum steps per evaluation episode.
- `--log <path>`, `--policy <path>`, `--path <path>`: override artifact output paths.

Run (from the repo root):

```bash
lake -R -K cuda=true build
lake -R -K cuda=true exe torchlean ppo_gridworld --device cuda --updates 1 --eval-every 1 \
  --eval-episodes 1 --eval-max-steps 8
```

Artifacts:
- The executable writes widget-friendly JSON snapshots to `data/rl/` by default:
  `ppo_gridworld_trainlog.json`, `ppo_gridworld_policy.json`, `ppo_gridworld_path.json`
  (override with `--log`, `--policy`, `--path`).
- You can tune runtime cost with:
  `--updates`, `--eval-every`, `--eval-episodes`, `--eval-max-steps`.
- Visualize them in the editor via `NN/Examples/Models/RL/Views/PPOGridWorld.lean`.

## What this example does (and does not) guarantee

- Because the environment dynamics are Lean code, you can reason about its properties directly
  (e.g. determinism, Markov property w.r.t. the explicit state, bounded rewards).
- The PPO/GAE update is implemented as Lean definitions and a TorchLean autograd program, so it is
  a natural target for formal proofs about the update equation.
- As in most practical PPO code, convergence and optimality are not guaranteed by this example; it
  is tuned for inspectability and type safety, not leaderboard performance.

References (primary):
- Schulman et al., "Proximal Policy Optimization Algorithms" (2017):
  https://arxiv.org/abs/1707.06347
- Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation"
  (2015): https://arxiv.org/abs/1506.02438
- Williams, "Simple statistical gradient-following algorithms for connectionist reinforcement
  learning" (REINFORCE, 1992): https://doi.org/10.1007/BF00992696
- Sutton and Barto, *Reinforcement Learning: An Introduction* (2nd ed., GridWorld examples):
  http://incompleteideas.net/book/the-book-2nd.html
- Puterman, *Markov Decision Processes* (finite discounted MDPs):
  https://doi.org/10.1002/9780470316887
-/

@[expose] public section

open Spec TorchLean TorchLean.Tensor
open TorchLean

namespace NN.Examples.Models.RL.PPOGridWorld

/-- Name of this executable target (used in CLI error messages and banners). -/
def exeName : String := "ppo_gridworld"

/-!
## Configuration
-/

/-- Grid width (number of columns). -/
def width : Nat := 4

/-- Grid height (number of rows). -/
def height : Nat := 4

/-- Total number of discrete states (`width * height`). -/
def stateCount : Nat := height * width

/-- Number of discrete actions (up/down/left/right). -/
def actionCount : Nat := 4

/-- Width of the hidden layer in the actor and critic MLPs. -/
def hiddenWidth : Nat := 32

/-- PPO rollout horizon (also the training batch size for this example). -/
def horizon : Nat := 64

/-- Discount factor used in returns / GAE. -/
def discountFactor : Float := 0.99

/-- GAE(λ) parameter controlling the bias/variance tradeoff of advantage estimates. -/
def gaeLambda : Float := 0.95

/-- Adam learning rate used for the GridWorld actor-critic update. -/
def learningRate : Float := 3e-3

/-- Number of PPO optimization epochs per collected rollout batch. -/
def updateEpochs : Nat := 8

/-- Default maximum number of PPO updates (can be overridden by `--updates`). -/
def maxUpdates : Nat := 2000

/-- Default evaluation checkpoint interval (can be overridden by `--eval-every`). -/
def defaultEvaluationInterval : Nat := 50

/-- Default evaluation episodes per checkpoint (can be overridden by `--eval-episodes`). -/
def defaultEvaluationEpisodes : Nat := 20

instance : NeZero horizon := ⟨by decide⟩
instance : NeZero actionCount := ⟨by decide⟩

/-- The observation tensor shape used by this example: `[..., stateCount]` one-hot vectors. -/
def observation : Shape := [stateCount]

def rollout : Shape := [horizon]
def rolloutStates : Shape := rl.ppo.StateBatchShape horizon observation
/-- Logits for a whole rollout: one row of action logits per timestep. -/
def rolloutLogits : Shape := rl.ppo.LogitsBatchShape horizon actionCount
/-- Value estimates for a whole rollout, one per timestep. -/
def rolloutValues : Shape := rl.ppo.ValueBatchShape horizon

/-- Action logits at a single timestep. -/
def actionLogits : Shape := [actionCount]
/-- A single value estimate, shaped `[1]` so it composes with the batched shapes above. -/
def value : Shape := [1]

/-!
## Formal GridWorld model (spec/proof layer)

We define a real-valued GridWorld model and record the proof that its stochastic-MDP view is valid.

This proof is not used by the executable training loop directly; it exists so that downstream
theorems about the induced MDP can refer to a concrete environment used in an example.
-/

/-- Start position (top-left cell). -/
def startPos : Spec.RL.Envs.GridWorld.State width height :=
  (⟨0, by decide⟩, ⟨0, by decide⟩)

/-- Goal position (bottom-right cell). -/
def goalPos : Spec.RL.Envs.GridWorld.State width height :=
  (⟨height - 1, by decide⟩, ⟨width - 1, by decide⟩)

noncomputable section

/-- A discount factor in `[0,1)` at the proof layer (`ℝ`) for the MDP instance. -/
def proofDiscountFactor : ℝ := (99 : ℝ) / 100

/-- Proof-layer GridWorld instance over `ℝ` rewards/discounts. -/
def proofGridWorld : Spec.RL.Envs.GridWorld width height :=
  { start := startPos
    goal := goalPos
    discount := proofDiscountFactor }

/-- The induced finite stochastic MDP for `proofGridWorld` is well-formed
($0\le\gamma<1$, with row-stochastic transitions). -/
theorem proofGridWorld_valid :
    Spec.RL.FiniteStochastic.Valid
      (Spec.RL.Envs.GridWorld.toFiniteStochasticMDP
        (width := width) (height := height) proofGridWorld) := by
  have hγ₀ : 0 ≤ proofGridWorld.discount := by
    norm_num [proofGridWorld, proofDiscountFactor]
  have hγ₁ : proofGridWorld.discount < 1 := by
    norm_num [proofGridWorld, proofDiscountFactor]
  exact Proofs.RL.Envs.GridWorld.toFiniteStochasticMDP_valid
    (width := width) (height := height) (gw := proofGridWorld) hγ₀ hγ₁

end

/-!
## Lean-native runtime environment

We implement a Gym-style environment (`Spec.RL.Env`) whose observations are **one-hot** vectors
over the flattened finite state space `Fin stateCount`.

The policy sees the same tensor shape as the Gymnasium-backed example and produces logits over a
finite action set.
-/

/-- Start state encoded as `Fin stateCount`. -/
def startState : Fin stateCount :=
  Spec.RL.Envs.GridWorld.encode (width := width) (height := height) startPos

/-- Encode a discrete state as a one-hot observation. -/
def observationOfState (s : Fin stateCount) : Tensor Float observation :=
  Tensor.oneHot (α := Float) stateCount s

/-- Absolute difference on natural-number coordinates, returned as a `Float`. -/
def coordDist (a b : Nat) : Float :=
  Float.ofNat (if a ≤ b then b - a else a - b)

/-- Manhattan distance to the goal. -/
def goalDistance (pos : Spec.RL.Envs.GridWorld.State width height) : Float :=
  let (x, y) := pos
  let (goalX, goalY) := goalPos
  coordDist x.val goalX.val + coordDist y.val goalY.val

/--
Deterministic GridWorld transition function with dense progress rewards.

The original sparse `-1 until terminal` reward gave short runs too little learning signal: random
rollouts rarely found the goal, so PPO received almost no useful signal. This shaped reward keeps
the same goal-reaching task, but gives the learner immediate credit for moving closer to the goal
and a small penalty for dithering.
-/
def stepState (state : Fin stateCount) (action : Fin actionCount) :
    Spec.RL.StepResult (Fin stateCount) Float :=
  let pos :=
    Spec.RL.Envs.GridWorld.decode (width := width) (height := height) state
  if _hGoal : pos = goalPos then
    { state := state
      reward := 0
      terminated := true
      truncated := false }
  else
    let nextPos :=
      Spec.RL.Envs.GridWorld.nextState (width := width) (height := height) pos action
    let nextState :=
      Spec.RL.Envs.GridWorld.encode (width := width) (height := height) nextPos
    if _hNextGoal : nextPos = goalPos then
      { state := nextState
        reward := 1
        terminated := true
        truncated := false }
    else
      let progress := goalDistance pos - goalDistance nextPos
      { state := nextState
        reward := progress - 0.05
        terminated := false
        truncated := false }

/-- Lean-native environment packaged as a `Spec.RL.Env` for reuse with the generic RL runtime. -/
def env :
    Spec.RL.Env
      (Fin stateCount) (Fin actionCount) (Tensor Float observation) Float :=
  { initialState := startState
    observe := observationOfState
    step := stepState }

/-!
## Trust boundary contract

Even though this environment is Lean-native, we keep a contract in play to exercise the
“checked preconditions” workflow and to keep the interface identical to the external Gymnasium
collector.
-/

def contract : rl.boundary.Contract observation actionCount :=
  { checkObsFinite := true
    checkRewardFinite := true
    obsRange? := some (0, 1)
    rewardRange? := some (-1.05, 1)
    requireExclusiveDoneFlags := false }

/-!
## Model (Actor + Critic)
-/

abbrev modelConfig : nn.models.PPO.Config :=
  { observationWidth := stateCount
    hiddenWidth := hiddenWidth
    actionCount := actionCount }

/-- Construct the actor network as an MLP mapping one-hot observations to action logits. -/
def actor (batchShape : Shape := []) :
    nn.Builder (nn.Sequential (modelConfig.input batchShape)
      (modelConfig.actorOutput batchShape)) :=
  nn.models.PPO.actor modelConfig batchShape

/--
Construct the critic network as an MLP mapping one-hot observations to a scalar value estimate.
-/
def critic (batchShape : Shape := []) :
    nn.Builder (nn.Sequential (modelConfig.input batchShape)
      (modelConfig.criticOutput batchShape)) :=
  nn.models.PPO.critic modelConfig batchShape

/-!
## Rollout collection (Lean-native environment)
-/

/-- Collect a fixed-horizon PPO rollout from the Lean-native environment using a checked session.

This is an example-local rollout operation around `rl.ppo.collectRolloutFromSession` that packages:
- a `Spec.RL.Env` as a `rl.session.CheckedSession`, and
- the (actor, critic) prediction functions at observation shape.
-/
def collectRolloutFromEnvironment
    (predictLogits : Tensor Float observation → Tensor Float [actionCount])
    (predictValue : Tensor Float observation → Float)
    (rngSeed rngCounter : Nat)
    (resetOnDone : Bool := true) :
    IO (rl.ppo.Rollout Float observation actionCount horizon × Nat) := do
  let sess : rl.session.CheckedSession observation actionCount :=
    rl.session.ofEnv
      (State := Fin stateCount) (obsShape := observation) (nActions := actionCount)
      env contract (resetOnDone := resetOnDone)
  rl.ppo.collectRolloutFromSession
    (α := Float) (obsShape := observation) (nActions := actionCount)
    (horizon := horizon)
    sess (castObservation := id) (castReward := id)
      (predictLogits := predictLogits) (predictValue := predictValue)
    (rngSeed := rngSeed) (rngCounter := rngCounter)

/-!
## Evaluation

Evaluation APIs live in `rl.eval`.
-/

/-!
## Main Training Loop
-/

/-- Entry point for `lake exe torchlean ppo_gridworld`.

This executable:
- runs PPO updates against a Lean-native GridWorld environment,
- periodically evaluates the greedy policy and logs the average return,
- writes widget-friendly JSON artifacts (training curve, greedy policy snapshot, greedy path
  snapshot).
-/
def main (args : List String) : IO UInt32 := do
  Module.Command.run
    (config := {
      banner? := some <| Support.bannerWithDeviceDetails
        exeName
        s!"PPO on Lean-native GridWorld ({width}x{height}, horizon={horizon})"
        "  env: pure Lean dynamics + boundary contract check + formal MDP validity proof available"
      usage? := some <| rl.cli.PPOOptions.usage exeName #[
        "",
        "Artifacts:",
        "  --policy PATH      greedy-policy JSON output",
        "  --path PATH        greedy-path JSON output"
      ]
      printSuccess := true })
    exeName args
    (.native fun runtime rest => do
      let (policyPath, rest) ← CLI.orThrow exeName <|
        CLI.takePathFlag rest "policy"
          (default := Runtime.RL.Artifacts.DefaultPaths.ppoGridWorldPolicy)
      let (pathPath, rest) ← CLI.orThrow exeName <|
        CLI.takePathFlag rest "path" (default := Runtime.RL.Artifacts.DefaultPaths.ppoGridWorldPath)
      let (ppo, rest) ← CLI.orThrow exeName <|
        rl.cli.PPOOptions.parse
          exeName rest Runtime.RL.Artifacts.DefaultPaths.ppoGridWorldTrainLog
          (defaultUpdateCount := maxUpdates)
          (defaultEvaluationInterval := defaultEvaluationInterval)
          (defaultEvaluationEpisodes := defaultEvaluationEpisodes)
          (defaultMaximumEvaluationSteps := 128)
      CLI.orThrow exeName <| CLI.checkNoArgs rest

      let updateCount : Nat := ppo.updateCount
      let evaluationInterval : Nat := ppo.evaluationInterval
      let evaluationEpisodes : Nat := ppo.evaluationEpisodes
      let maximumEvaluationSteps : Nat := ppo.maximumEvaluationSteps

      let seedActor ← rand.nextSeedGlobal
      let seedCritic ← rand.nextSeedGlobal
      let actorObs : nn.Sequential observation actionLogits :=
        nn.build seedActor (actor [])
      let criticObs : nn.Sequential observation value :=
        nn.build seedCritic (critic [])
      let actorRollout : nn.Sequential rolloutStates rolloutLogits :=
        nn.build seedActor (actor rollout)
      let criticRollout : nn.Sequential rolloutStates rolloutValues :=
        nn.build seedCritic (critic rollout)

      let actorGraph ← nn.lowerToTypedGraph actorObs
      let criticGraph ← nn.lowerToTypedGraph criticObs

      let m ← rl.ppo.instantiateActorCritic
        (α := Float) (options := runtime)
        (batch := horizon) (nActions := actionCount)
        actorRollout criticRollout

      let stepSample ←
        rl.ppo.trainingStep m
          (optim.adam { learningRate := learningRate })


      -- Training curve: greedy-policy return before training, then at each evaluation checkpoint.
      -- We keep evaluation logs as a compact `Curve` because it targets JSON/widget display.
      let mut curve : Training.Curve := {}

      -- Helper: build a fresh (checked) session for evaluation rollouts/paths.
      let evaluationSessionAt :
          Nat → rl.session.CheckedSession observation actionCount :=
        fun _seed =>
          rl.session.ofEnv
            (State := Fin stateCount) (obsShape := observation) (nActions := actionCount)
            env contract (resetOnDone := false)

      -- Evaluate + snapshot the untrained policy.
      let psAll0 ← rl.ppo.state (α := Float) m
      let policyLogits0 : Tensor Float observation → Tensor Float [actionCount] :=
        rl.ppo.actorPolicy actorGraph actorRollout criticRollout psAll0
      let avg0 ←
        rl.eval.averageEpisodeTotalReward
          (obsShape := observation) (nActions := actionCount)
          evaluationSessionAt policyLogits0 (baseSeed := runtime.seed)
          (episodes := evaluationEpisodes)
          (maxSteps := maximumEvaluationSteps)
      curve := curve.push 0 avg0
      IO.println s!"  eval(step=0) avg_return={avg0}"

      let policyBefore : Tensor Nat [stateCount] :=
        Tensor.ofFn (fun (s : Fin stateCount) =>
          let obs := observationOfState s
          let logits := policyLogits0 obs
          (rl.eval.greedyActionFromLogits
            (α := Float) (nActions := actionCount) logits).val)
      let pathBeforeStates ←
        rl.eval.episodeSessPath (obsShape := observation) (nActions := actionCount)
          (evaluationSessionAt runtime.seed) policyLogits0
          (maxSteps := maximumEvaluationSteps)
      let pathBefore : Array (Nat × Nat) :=
        pathBeforeStates.map (fun s =>
          let (x, y) :=
            Spec.RL.Envs.GridWorld.decode (width := width) (height := height) s
          (x.val, y.val))

      curve ← rl.ppo.train discountFactor gaeLambda
        { updates := updateCount, epochs := updateEpochs,
          evaluationEvery := evaluationInterval, seed := runtime.seed }
        (fun _update rngSeed rngCounter => do
            let psAll ← rl.ppo.state (α := Float) m
            let predictLogits : Tensor Float observation → Tensor Float [actionCount] :=
              rl.ppo.actorPolicy actorGraph actorRollout criticRollout psAll
            let predictValue : Tensor Float observation → Float :=
              rl.ppo.criticValue criticGraph actorRollout criticRollout psAll

            let (rollout, rngCounter') ←
              collectRolloutFromEnvironment predictLogits predictValue
                (rngSeed := rngSeed) (rngCounter := rngCounter) (resetOnDone := true)
            pure (rollout, rngCounter'))
        stepSample
        (fun completedUpdates => do
            let psAll' ← rl.ppo.state (α := Float) m
            let policyLogits : Tensor Float observation → Tensor Float [actionCount] :=
              rl.ppo.actorPolicy actorGraph actorRollout criticRollout psAll'
            let avg ←
              rl.eval.averageEpisodeTotalReward
                (obsShape := observation) (nActions := actionCount)
                evaluationSessionAt policyLogits (baseSeed := runtime.seed)
                (episodes := evaluationEpisodes)
                (maxSteps := maximumEvaluationSteps)
            IO.println s!"  update={completedUpdates} avg_return={avg}"
            pure (avg, false))
        curve

      -- Snapshot the final greedy policy and a single episode path.
      let psAllF ← rl.ppo.state (α := Float) m
      let policyLogitsF : Tensor Float observation → Tensor Float [actionCount] :=
        rl.ppo.actorPolicy actorGraph actorRollout criticRollout psAllF
      let policyAfter : Tensor Nat [stateCount] :=
        Tensor.ofFn (fun (s : Fin stateCount) =>
          let obs := observationOfState s
          let logits := policyLogitsF obs
          (rl.eval.greedyActionFromLogits
            (α := Float) (nActions := actionCount) logits).val)
      let pathAfterStates ←
        rl.eval.episodeSessPath (obsShape := observation) (nActions := actionCount)
          (evaluationSessionAt runtime.seed) policyLogitsF
          (maxSteps := maximumEvaluationSteps)
      let pathAfter : Array (Nat × Nat) :=
        pathAfterStates.map (fun s =>
          let (x, y) :=
            Spec.RL.Envs.GridWorld.decode (width := width) (height := height) s
          (x.val, y.val))

      Training.Curve.writeLog curve
        ppo.logDestination
        s!"PPO GridWorld {width}x{height} (TorchLean)"
        "avg_return"
        (color := "#4e79a7")
        #[
          s!"width={width}",
          s!"height={height}",
          s!"horizon={horizon}",
          s!"gamma={discountFactor}",
          s!"lambda={gaeLambda}",
          s!"lr={learningRate}",
          s!"updates={updateCount}",
          s!"eval_every={evaluationInterval}",
          s!"eval_episodes={evaluationEpisodes}",
          s!"eval_max_steps={maximumEvaluationSteps}",
          Support.deviceNote runtime
        ]

      let polDiff : Runtime.RL.Artifacts.GridWorld.PolicyDiff :=
        { width := width, height := height, before := policyBefore.to (Array Nat),
          after := policyAfter.to (Array Nat)
          notes := #["greedy policy (argmax over logits)"] }
      Runtime.RL.Artifacts.GridWorld.PolicyDiff.writeJson policyPath polDiff
      IO.println s!"{exeName}: wrote policy snapshot to {policyPath}"

      let pathDiff : Runtime.RL.Artifacts.GridWorld.PathDiff :=
        { width := width, height := height, before := pathBefore, after := pathAfter
          notes := #["greedy episode path (states decoded to (row,col))"] }
      Runtime.RL.Artifacts.GridWorld.PathDiff.writeJson pathPath pathDiff
      IO.println s!"{exeName}: wrote path snapshot to {pathPath}"

      IO.println s!"{exeName}: done"
    )

end NN.Examples.Models.RL.PPOGridWorld
