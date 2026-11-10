/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Data.Nat.Dist
public import TorchLean.NN.Models.PPO
public import TorchLean.NN.RL.CLI
public import TorchLean.NN.RL.PPO.ActorCritic
public import TorchLeanExamples.Models.Training
public import TorchLeanExamples.Models.RL.GridWorld.Artifacts
public import TorchLeanExamples.Models.RL.PPOAgent
public import TorchLeanExamples.Models.RL.GridWorld.Properties
public import TorchLean.NN.Data.Source

/-!
# PPO on Lean-native GridWorld (Executable Example + Formal Model)

The example complements `torchlean ppo_cartpole`:

- `torchlean ppo_cartpole` uses an external Python Gymnasium environment and checks every step
  against a Lean-side trust-boundary contract (`TorchLean.RL.Boundary.Contract`).
- `torchlean ppo_gridworld` runs the same PPO update in Lean on a **Lean-native** GridWorld.

Every transition of the Lean environment also goes through the boundary checker, so downstream
training code consumes `RL.ObservedTransition` values in one format, whether they come from a
Lean-native environment or from an external sampler.

## Formal hooks

1. `realGridWorld` has an induced finite stochastic MDP (`TorchLean.RL.StochasticMDP`) and a
   proof that it is well-formed (row-stochastic transition rows, $0\le\gamma<1$). That model keeps
   the sparse rewards of the specification, while the training environment below uses shaped
   rewards.
2. The boundary checker can be turned into a Prop-level hypothesis via
   `TorchLean.RL.Boundary.checkTransitionFin_sound`
   (see `TorchLean/NN/RL/Boundary/Soundness.lean`), or `TorchLean.RL.Boundary.stepChecked` can take
   `TorchLean.RL.Boundary.checkTransitionFinWithProof` as its checker for external environments.

## CLI flags

- `--device cuda`: run the Torch backend on CUDA (requires building with `-K cuda=true`).
- `--updates <n>`: number of PPO updates to run (default 2000).
- `--eval-every <n>`: evaluate the greedy policy every `n` updates (default 50).
- `--eval-episodes <n>`: number of evaluation episodes per checkpoint (default 20).
- `--eval-max-steps <n>`: maximum steps per evaluation episode (default 128).
- `--log <path>`, `--policy <path>`, `--path <path>`: override artifact output paths.

Run (from the repo root):

```bash
scripts/lake.sh -Kcuda=true build
scripts/lake.sh -Kcuda=true exe torchlean ppo_gridworld --device cuda \
    --updates 1 --eval-every 1 \
  --eval-episodes 1 --eval-max-steps 8
```

Artifacts:
- The executable writes widget-friendly JSON snapshots to `data/rl/` by default:
  `ppo_gridworld_trainlog.json`, `ppo_gridworld_policy.json`, `ppo_gridworld_path.json`
  (override with `--log`, `--policy`, `--path`).
- The runtime cost is set by `--updates`, `--eval-every`, `--eval-episodes` and `--eval-max-steps`.
- `TorchLeanExamples/Models/RL/PPOGridWorld/View.lean` displays the snapshots in the editor.

## What this example does (and does not) guarantee

- The environment dynamics are Lean code, so properties such as determinism, the Markov property
  in the grid state and bounded rewards can be proved in Lean.
- The PPO/GAE update is implemented as Lean definitions and a TorchLean autograd program, so it is
  a natural target for formal proofs about the update equation.
- As in most practical PPO code, the example guarantees neither convergence nor optimality. Its
  settings favor inspectability and type safety over leaderboard performance.

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

open TorchLean.RL TorchLean.Tensor

namespace TorchLean.Examples.PPOGridWorld

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
def lr : Float := 3e-3

/-- Number of PPO optimization epochs per collected rollout batch. -/
def updateEpochs : Nat := 8

/--
Default number of PPO updates (`--updates`). The GridWorld run has no early-stop criterion, so it
performs all of them.
-/
def maxUpdates : Nat := 2000

/-- Default number of PPO updates between greedy-policy evaluations (`--eval-every`). -/
def defaultEvaluationInterval : Nat := 50

/-- Default number of evaluation episodes per checkpoint (`--eval-episodes`). -/
def defaultEvaluationEpisodes : Nat := 20

instance : NeZero horizon := ⟨by decide⟩
instance : NeZero actionCount := ⟨by decide⟩

/-- Shape of one GridWorld observation, a one-hot vector of length `stateCount`. -/
def observation : Shape := [stateCount]

/-!
## Formal GridWorld model (spec/proof layer)

We define a real-valued GridWorld model and record the proof that its stochastic-MDP view is valid.

The specification pays reward `-1` for each step that does not reach the goal and `0` on arrival,
while the executable's `stepState` below uses progress rewards and a goal reward of `1`. The
theorem concerns `realGridWorld` alone and establishes no equivalence with the shaped-reward
training environment.
-/

/-- Start position (top-left cell). -/
def startPos : GridWorld.State width height :=
  (⟨0, by decide⟩, ⟨0, by decide⟩)

/-- Goal position (bottom-right cell). -/
def goalPos : GridWorld.State width height :=
  (⟨height - 1, by decide⟩, ⟨width - 1, by decide⟩)

noncomputable section

/-- A discount factor in `[0,1)` at the proof layer (`ℝ`) for the MDP instance. -/
def realDiscountFactor : ℝ := (99 : ℝ) / 100

/-- Proof-layer GridWorld instance over `ℝ` rewards/discounts. -/
def realGridWorld : GridWorld width height :=
  { start := startPos
    goal := goalPos
    discount := realDiscountFactor }

/-- The induced finite stochastic MDP for `realGridWorld` is well-formed
($0\le\gamma<1$, with row-stochastic transitions). -/
theorem realGridWorld_valid :
    RL.StochasticMDP.Valid
      (GridWorld.toStochasticMDP
        (width := width) (height := height) realGridWorld) := by
  have hγ₀ : 0 ≤ realGridWorld.discount := by
    norm_num [realGridWorld, realDiscountFactor]
  have hγ₁ : realGridWorld.discount < 1 := by
    norm_num [realGridWorld, realDiscountFactor]
  exact GridWorld.toStochasticMDP_valid
    (width := width) (height := height) (gw := realGridWorld) hγ₀ hγ₁

end

/-!
## Lean-native runtime environment

We implement a Gym-style environment (`RL.Env`) whose observations are **one-hot** vectors
over the flattened finite state space `Fin stateCount`.

As in the Gymnasium-backed example, the policy maps one observation vector, here of length
`stateCount`, to logits over a finite action set.
-/

/-- Start state encoded as `Fin stateCount`. -/
def startState : Fin stateCount :=
  GridWorld.encode (width := width) (height := height) startPos

/-- Encode a discrete state as a one-hot observation. -/
def observationOfState (s : Fin stateCount) : Tensor Float observation :=
  Tensor.oneHot (α := Float) stateCount s

/-- Manhattan distance to the goal. -/
def goalDistance (pos : GridWorld.State width height) : Float :=
  let (x, y) := pos
  let (goalX, goalY) := goalPos
  Float.ofNat (Nat.dist x.val goalX.val) + Float.ofNat (Nat.dist y.val goalY.val)

/--
Deterministic GridWorld transition function with dense progress rewards.

A move onto the goal earns `1`, and any other move earns the decrease in Manhattan distance to the
goal minus `0.05`. The goal is absorbing with reward `0`. These shaped rewards differ from those of
`realGridWorld`, and the file proves no policy equivalence between the two environments.
-/
def stepState (state : Fin stateCount) (action : Fin actionCount) :
    RL.StepResult (Fin stateCount) Float :=
  let pos :=
    GridWorld.decode (width := width) (height := height) state
  if _hGoal : pos = goalPos then
    { state := state
      reward := 0
      terminated := true
      truncated := false }
  else
    let nextPos :=
      GridWorld.nextState (width := width) (height := height) pos action
    let nextState :=
      GridWorld.encode (width := width) (height := height) nextPos
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

/-- Lean-native environment packaged as a `RL.Env` for reuse with the generic RL runtime. -/
def env :
    RL.Env
      (Fin stateCount) (Fin actionCount) (Tensor Float observation) Float :=
  { initialState := startState
    observe := observationOfState
    step := stepState }

/-!
## Trust boundary contract

Although the environment is Lean-native, its transitions go through a contract check. That
exercises the checked-preconditions workflow and gives the session the interface of the external
Gymnasium collector. The contract requires finite observations in $[0,1]$ and finite rewards in
$[-1.05,1]$, a range that covers every shaped reward, and it allows both done flags at once.
-/

/-- The check on every transition: finite observations in `[0, 1]` and finite rewards in
`[-1.05, 1]`, with both done flags allowed at once. -/
def contract : Boundary.Contract observation actionCount :=
  { checkObsFinite := true
    checkRewardFinite := true
    obsRange? := some (0, 1)
    rewardRange? := some (-1.05, 1)
    requireExclusiveDoneFlags := false }

/-!
## Model (Actor + Critic)
-/

/-- Actor and critic sizes: one-hot observations of width `stateCount`, hidden layers of width
`hiddenWidth` and `actionCount` actor logits. -/
abbrev modelConfig : nn.PPO.Config :=
  { observationWidth := stateCount
    hiddenWidth := hiddenWidth
    actionCount := actionCount }

/-!
## Rollout collection (Lean-native environment)
-/

/-- Collect a fixed-horizon PPO rollout from the Lean-native environment using a checked session.

The session wraps `env` and `contract` as a `Session.Checked`, and `PPO.collect`
drives it with the actor and critic prediction functions at observation shape. With `resetOnDone`
set, an episode that ends restarts from the start state. The result pairs the rollout with the
advanced random counter.
-/
def collectRollout
    (predictLogits : Tensor Float observation → Tensor Float [actionCount])
    (predictValue : Tensor Float observation → Float)
    (rngSeed rngCounter : Nat)
    (resetOnDone : Bool := true) :
    IO (PPO.Rollout Float observation actionCount horizon × Nat) := do
  let sess : Session.Checked observation actionCount :=
    Session.Checked.ofEnv
      (State := Fin stateCount) (obsShape := observation) (nActions := actionCount)
      env contract (resetOnDone := resetOnDone)
  PPO.collect
    (α := Float) (obsShape := observation) (nActions := actionCount)
    (horizon := horizon)
    sess (castObservation := id) (castReward := id)
      (predictLogits := predictLogits) (predictValue := predictValue)
    (rngSeed := rngSeed) (rngCounter := rngCounter)

/-!
## Main Training Loop
-/

/-- Entry point for `scripts/lake.sh exe torchlean ppo_gridworld`.

The command:
- runs PPO updates against the Lean-native GridWorld environment,
- evaluates the greedy policy before training and every `--eval-every` updates, logging the mean
  return,
- writes widget-friendly JSON artifacts: the training curve, and greedy-policy and greedy-path
  snapshots from before and after training.
-/
def main (args : List String) : IO UInt32 := do
  Module.Command.run
    (config := {
      banner? := some <| Training.banner
        exeName
        s!"PPO on Lean-native GridWorld ({width}x{height}, horizon={horizon})"
        (details := some <| "  env: pure Lean dynamics + boundary contract check " ++
          "+ formal MDP validity proof available")
      usage? := some <| CLI.Options.usage exeName #[
        "",
        "Artifacts:",
        "  --policy PATH      greedy-policy JSON output",
        "  --path PATH        greedy-path JSON output"
      ]
      printSuccess := true })
    exeName args
    (fun runtime rest => do
      let (policyPath, rest) ← CLI.orThrow exeName <|
        CLI.takePathFlag rest "policy"
          (default := Training.artifactPath "ppo_gridworld" "policy" "rl")
      let (episodePath, rest) ← CLI.orThrow exeName <|
        CLI.takePathFlag rest "path" (default := Training.artifactPath "ppo_gridworld" "path" "rl")
      let (ppo, rest) ← CLI.orThrow exeName <|
        CLI.Options.parse
          exeName rest (Training.artifactPath "ppo_gridworld" (dir := "rl"))
          (defaultUpdateCount := maxUpdates)
          (defaultEvaluationInterval := defaultEvaluationInterval)
          (defaultEvaluationEpisodes := defaultEvaluationEpisodes)
          (defaultMaximumEvaluationSteps := 128)
      CLI.orThrow exeName <| CLI.checkNoArgs rest

      let agent : PPOAgent observation actionCount horizon ←
        PPOAgent.create modelConfig horizon runtime (optim.adam { lr := lr })

      -- Training curve: greedy-policy return before training, then at each evaluation checkpoint.
      -- It is a compact `Curve` of arrays because it only feeds the JSON log and the widget.
      let mut curve : Training.Curve := {}

      -- Each evaluation episode gets a fresh checked session that does not reset on done. The
      -- seed is unused because the dynamics are deterministic.
      let evaluationSessionAt :
          Nat → Session.Checked observation actionCount :=
        fun _seed =>
          Session.Checked.ofEnv
            (State := Fin stateCount) (obsShape := observation) (nActions := actionCount)
            env contract (resetOnDone := false)

      -- Evaluate + snapshot the untrained policy.
      let psAll0 ← agent.state
      let policyLogits0 : Tensor Float observation → Tensor Float [actionCount] :=
        agent.policy psAll0
      let avg0 ←
        Eval.meanReward
          (obsShape := observation) (nActions := actionCount)
          evaluationSessionAt policyLogits0 (baseSeed := runtime.seed)
          (episodes := ppo.evaluationEpisodes)
          (maxSteps := ppo.maximumEvaluationSteps)
      curve := curve.push 0 avg0
      IO.println s!"  eval(step=0) avg_return={avg0}"

      let policyBefore : Tensor Nat [stateCount] :=
        Tensor.ofFn (fun (s : Fin stateCount) =>
          let obs := observationOfState s
          let logits := policyLogits0 obs
          (Eval.greedy
            (α := Float) (nActions := actionCount) logits).val)
      let pathBeforeStates ←
        Eval.path (obsShape := observation) (nActions := actionCount)
          (evaluationSessionAt runtime.seed) policyLogits0
          (maxSteps := ppo.maximumEvaluationSteps)
      let pathBefore : Array (Nat × Nat) :=
        pathBeforeStates.map (fun s =>
          let (x, y) :=
            GridWorld.decode (width := width) (height := height) s
          (x.val, y.val))

      curve ← PPO.train discountFactor gaeLambda
        { updates := ppo.updateCount, epochs := updateEpochs,
          evaluationEvery := ppo.evaluationInterval, seed := runtime.seed }
        (fun _update rngSeed rngCounter => do
            let psAll ← agent.state
            let predictLogits : Tensor Float observation → Tensor Float [actionCount] :=
              agent.policy psAll
            let predictValue : Tensor Float observation → Float :=
              agent.value psAll

            let (rollout, rngCounter') ←
              collectRollout predictLogits predictValue
                (rngSeed := rngSeed) (rngCounter := rngCounter) (resetOnDone := true)
            pure (rollout, rngCounter'))
        agent.updateBatch
        (fun completedUpdates => do
            let psAll' ← agent.state
            let policyLogits : Tensor Float observation → Tensor Float [actionCount] :=
              agent.policy psAll'
            let avg ←
              Eval.meanReward
                (obsShape := observation) (nActions := actionCount)
                evaluationSessionAt policyLogits (baseSeed := runtime.seed)
                (episodes := ppo.evaluationEpisodes)
                (maxSteps := ppo.maximumEvaluationSteps)
            IO.println s!"  update={completedUpdates} avg_return={avg}"
            pure (avg, false))
        curve

      -- Snapshot the final greedy policy and a single episode path.
      let psAllF ← agent.state
      let policyLogitsF : Tensor Float observation → Tensor Float [actionCount] :=
        agent.policy psAllF
      let policyAfter : Tensor Nat [stateCount] :=
        Tensor.ofFn (fun (s : Fin stateCount) =>
          let obs := observationOfState s
          let logits := policyLogitsF obs
          (Eval.greedy
            (α := Float) (nActions := actionCount) logits).val)
      let pathAfterStates ←
        Eval.path (obsShape := observation) (nActions := actionCount)
          (evaluationSessionAt runtime.seed) policyLogitsF
          (maxSteps := ppo.maximumEvaluationSteps)
      let pathAfter : Array (Nat × Nat) :=
        pathAfterStates.map (fun s =>
          let (x, y) :=
            GridWorld.decode (width := width) (height := height) s
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
          s!"lr={lr}",
          s!"updates={ppo.updateCount}",
          s!"eval_every={ppo.evaluationInterval}",
          s!"eval_episodes={ppo.evaluationEpisodes}",
          s!"eval_max_steps={ppo.maximumEvaluationSteps}",
          Training.deviceNote runtime
        ]

      let polDiff : GridWorld.Diff Nat :=
        { width := width, height := height, before := policyBefore.to (Array Nat),
          after := policyAfter.to (Array Nat)
          notes := #["greedy policy (argmax over logits)"] }
      GridWorld.Diff.writeJson .policy policyPath polDiff
      IO.println s!"{exeName}: wrote policy snapshot to {policyPath}"

      let pathDiff : GridWorld.Diff (Nat × Nat) :=
        { width := width, height := height, before := pathBefore, after := pathAfter
          notes := #["greedy episode path (states decoded to (row,col))"] }
      GridWorld.Diff.writeJson .path episodePath pathDiff
      IO.println s!"{exeName}: wrote path snapshot to {episodePath}"

      IO.println s!"{exeName}: done"
    )

end TorchLean.Examples.PPOGridWorld
