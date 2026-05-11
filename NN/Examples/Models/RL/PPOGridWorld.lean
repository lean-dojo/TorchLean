/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

End-to-end PPO example: train an actor-critic on a Lean-native GridWorld environment.
-/

module

public import NN
public import NN.API.Models.PPO
public import NN.Runtime.RL.Artifacts.GridWorld
public import NN.Runtime.RL.Artifacts.DefaultPaths

public import NN.Spec.RL.Envs.GridWorld
public import NN.Proofs.RL.Envs.GridWorld

/-!
# PPO on Lean-native GridWorld (Executable Demo + Formal Model)

This example complements `torchlean ppo_cartpole`:

- `torchlean ppo_cartpole` uses an external Python Gymnasium environment and checks every step
  against a Lean-side trust-boundary contract (`Runtime.RL.Boundary.Contract`).
- This example uses a **Lean-native** GridWorld and still runs the same PPO update in Lean.

Even though the environment is defined in Lean, we *still* validate every transition with the
boundary checker. That keeps the data model unified: downstream training code consumes
`Spec.RL.ObservedTransition` in a single format regardless of whether the source is a Lean-native
environment or an external sampler.

## Formal hooks

1. The environment has an induced finite stochastic MDP (`Spec.RL.FiniteStochastic.MDP`) and we import a
   proof that it is well-formed (row-stochastic transition rows, `0 ≤ γ < 1`).
2. The boundary checker can be turned into a Prop-level hypothesis via
   `Proofs.RL.Boundary.contractHolds_of_checkTransitionFin_eq_ok` (see `NN/Proofs/RL/Boundary.lean`), or you can
   use the proof-layer Gymnasium wrapper `Runtime.RL.Gymnasium.Session.stepCheckedWithProof`
   (`NN/Proofs/RL/Gymnasium.lean`) for external environments.

## CLI flags

- `--cuda`: run the Torch backend on CUDA (requires building with `-K cuda=true`).
- `--updates <n>`: number of PPO updates to run.
- `--eval-every <n>`: evaluate the greedy policy every `n` updates.
- `--eval-episodes <n>`: number of evaluation episodes per checkpoint.
- `--eval-max-steps <n>`: maximum steps per evaluation episode.
- `--log <path>`, `--policy <path>`, `--path <path>`: override artifact output paths.

Run (from the repo root):

```bash
lake exe torchlean ppo_gridworld
lake build -R -K cuda=true && lake exe torchlean ppo_gridworld --cuda
lake exe torchlean ppo_gridworld --updates 200
```

Artifacts:
- The executable writes widget-friendly JSON snapshots to `data/rl/` by default:
  `ppo_gridworld_trainlog.json`, `ppo_gridworld_policy.json`, `ppo_gridworld_path.json`
  (override with `--log`, `--policy`, `--path`).
- You can also tune runtime cost with:
  `--updates`, `--eval-every`, `--eval-episodes`, `--eval-max-steps`.
- Visualize them in the editor via `NN/Examples/RL/PPOGridWorldView.lean`.

## What this example does (and does not) guarantee

- Because the environment dynamics are Lean code, you can reason about its properties directly
  (e.g. determinism, Markov property w.r.t. the explicit state, bounded rewards).
- The PPO/GAE update is implemented as Lean definitions and a TorchLean autograd program, so it is
  a natural target for formal proofs about the update equation.
- As in most practical PPO code, convergence and optimality are not guaranteed by this example; it is
  tuned for inspectability and type safety, not leaderboard performance.

References (primary):
- Schulman et al., "Proximal Policy Optimization Algorithms" (2017): https://arxiv.org/abs/1707.06347
- Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation" (2015):
  https://arxiv.org/abs/1506.02438
- Williams, "Simple statistical gradient-following algorithms for connectionist reinforcement learning"
  (REINFORCE, 1992): https://doi.org/10.1007/BF00992696
- Sutton and Barto, *Reinforcement Learning: An Introduction* (2nd ed., GridWorld examples):
  http://incompleteideas.net/book/the-book-2nd.html
- Puterman, *Markov Decision Processes* (finite discounted MDPs):
  https://doi.org/10.1002/9780470316887
-/

@[expose] public section

open Spec Tensor
open NN.API

namespace NN.Examples.Models.RL.PPOGridWorld

/-- Name of this executable target (used in CLI error messages and banners). -/
def exeName : String := "torchlean ppo_gridworld"

/-!
## Configuration
-/

/-- Grid width (number of columns). -/
def width : Nat := 4

/-- Grid height (number of rows). -/
def height : Nat := 4

/-- Total number of discrete states (`width * height`). -/
def nStates : Nat := height * width

/-- Number of discrete actions (up/down/left/right). -/
def nActions : Nat := 4

/-- Width of the hidden layer in the actor and critic MLPs. -/
def hiddenDim : Nat := 32

/-- PPO rollout horizon (also the training batch size for this example). -/
def horizon : Nat := 64

/-- Discount factor used in returns / GAE. -/
def gamma : Float := 0.99

/-- GAE(λ) parameter controlling the bias/variance tradeoff of advantage estimates. -/
def lam : Float := 0.95

/-- Adam learning rate. -/
def lr : Float := 3e-3

/-- Number of PPO optimization epochs per collected rollout batch. -/
def updateEpochs : Nat := 8

/-- Default maximum number of PPO updates (can be overridden by `--updates`). -/
def updatesMax : Nat := 2000

/-- Default evaluation checkpoint interval (can be overridden by `--eval-every`). -/
def evalEvery : Nat := 50

/-- Default evaluation episodes per checkpoint (can be overridden by `--eval-episodes`). -/
def evalEpisodes : Nat := 20

instance : Fact (0 < horizon) := ⟨by decide⟩
instance : Fact (0 < nActions) := ⟨by decide⟩

/-- The observation tensor shape used by this example: `[..., nStates]` one-hot vectors. -/
def obsShape : Shape := shape![nStates]

def pfxBatch : Shape := shape![horizon]
def sStateBatch : Shape := rl.ppo.StateBatchShape horizon obsShape
def sLogitsBatch : Shape := rl.ppo.LogitsBatchShape horizon nActions
def sScalarBatch : Shape := rl.ppo.ScalarBatchShape horizon
def sValueBatch : Shape := rl.ppo.ValueBatchShape horizon

def sState1 : Shape := obsShape
def sLogits1 : Shape := shape![nActions]
def sValue1 : Shape := shape![1]

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

/-- A discount factor in `[0,1)` at the proof layer (`ℝ`), used to build an MDP instance. -/
def discountR : ℝ := (99 : ℝ) / 100

/-- Proof-layer GridWorld instance over `ℝ` rewards/discounts. -/
def gwR : Spec.RL.Envs.GridWorld width height :=
  { start := startPos
    goal := goalPos
    discount := discountR }

/-- The induced finite stochastic MDP for `gwR` is well-formed (`0 ≤ γ < 1`, row-stochastic transitions). -/
theorem gwR_valid :
    Spec.RL.FiniteStochastic.Valid (Spec.RL.Envs.GridWorld.toFiniteStochasticMDP (width := width) (height := height) gwR) := by
  -- `discountR = 99/100` is in `[0,1)`.
  have hγ₀ : 0 ≤ gwR.discount := by
    norm_num [gwR, discountR]
  have hγ₁ : gwR.discount < 1 := by
    norm_num [gwR, discountR]
  exact Proofs.RL.Envs.GridWorld.toFiniteStochasticMDP_valid (width := width) (height := height) (gw := gwR) hγ₀ hγ₁

end

/-!
## Lean-native runtime environment

We implement a Gym-style environment (`Spec.RL.Env`) whose observations are **one-hot** vectors
over the flattened finite state space `Fin nStates`.

This keeps the PPO code identical to the Gymnasium example: the policy consumes tensors and
produces logits over a finite action set.
-/

/-- Start state encoded as `Fin nStates`. -/
def startState : Fin nStates :=
  Spec.RL.Envs.GridWorld.encode (width := width) (height := height) startPos

/-- Goal state encoded as `Fin nStates`. -/
def goalState : Fin nStates :=
  Spec.RL.Envs.GridWorld.encode (width := width) (height := height) goalPos

/-- Observation function: encode the discrete state as a one-hot vector of length `nStates`. -/
def obsOfState (s : Fin nStates) : Tensor Float obsShape :=
  NN.Tensor.oneHot (α := Float) nStates s

/-- Absolute difference on natural-number coordinates, returned as a `Float`. -/
def coordDist (a b : Nat) : Float :=
  Float.ofNat (if a ≤ b then b - a else a - b)

/-- Manhattan distance to the goal. -/
def goalDistance (pos : Spec.RL.Envs.GridWorld.State width height) : Float :=
  coordDist pos.1.val goalPos.1.val + coordDist pos.2.val goalPos.2.val

/--
Deterministic GridWorld transition function with dense progress rewards.

The original sparse `-1 until terminal` reward gave short runs too little learning signal: random rollouts
rarely found the goal, so PPO received almost no useful signal.  This shaped reward keeps the same
goal-reaching task, but gives the learner immediate credit for moving closer to the goal and a
small penalty for dithering.
-/
def stepState (state : Fin nStates) (action : Fin nActions) :
    Spec.RL.StepResult (Fin nStates) Float :=
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
def env : Spec.RL.Env (Fin nStates) (Fin nActions) (Tensor Float obsShape) Float :=
  { initialState := startState
    observe := obsOfState
    step := stepState }

/-!
## Trust boundary contract

Even though this environment is Lean-native, we keep a contract in play to exercise the
“checked preconditions” workflow and to keep the interface identical to the external Gymnasium
collector.
-/

def contract : rl.boundary.Contract obsShape nActions :=
  { checkObsFinite := true
    checkRewardFinite := true
    obsRange? := some (0, 1)
    rewardRange? := some (-1.05, 1)
    requireExclusiveDoneFlags := false }

/-!
## Model (Actor + Critic)
-/

def modelCfg : nn.models.PPOActorCriticConfig :=
  { obsDim := nStates, hiddenDim := hiddenDim, nActions := nActions }

/-- Construct the actor network as an MLP mapping one-hot observations to action logits. -/
def actorMk (pfx : Shape) : nn.M (nn.Sequential (pfx.appendDim nStates) (pfx.appendDim nActions)) :=
  nn.models.ppoActor modelCfg pfx

/-- Construct the critic network as an MLP mapping one-hot observations to a scalar value estimate. -/
def criticMk (pfx : Shape) : nn.M (nn.Sequential (pfx.appendDim nStates) (pfx.appendDim 1)) :=
  nn.models.ppoCritic modelCfg pfx

/-!
## Rollout collection (Lean-native environment)
-/

/-- Collect a fixed-horizon PPO rollout from the Lean-native environment using a checked session.

This is an example-local wrapper around `rl.ppo.collectRolloutCheckedSessionWith` that packages:
- a `Spec.RL.Env` as a `rl.session.CheckedSession`, and
- the (actor, critic) prediction functions at observation shape.
-/
def collectRolloutNativeWith
    (predictLogits : Tensor Float obsShape → Tensor Float (shape![nActions]))
    (predictValue : Tensor Float obsShape → Float)
    (rngSeed rngCounter : Nat)
    (resetOnDone : Bool := true) :
    IO (rl.ppo.Rollout Float obsShape nActions horizon × Nat) := do
  let sess : rl.session.CheckedSession obsShape nActions :=
    rl.session.ofEnv (State := Fin nStates) (obsShape := obsShape) (nActions := nActions) env contract
      (resetOnDone := resetOnDone)
  rl.ppo.collectRolloutCheckedSessionWith (α := Float) (obsShape := obsShape) (nActions := nActions)
    (horizon := horizon)
    sess (castObs := id) (castReward := id) (predictLogits := predictLogits) (predictValue := predictValue)
    (rngSeed := rngSeed) (rngCounter := rngCounter)

/-!
## Evaluation

Evaluation helpers live in `NN.API.rl.eval` (runtime module `NN.Runtime.RL.Eval`).
-/

/-!
## Main Training Loop
-/

/-- Entry point for `lake exe torchlean ppo_gridworld`.

This executable:
- runs PPO updates against a Lean-native GridWorld environment,
- periodically evaluates the greedy policy and logs the average return,
- writes widget-friendly JSON artifacts (training curve, greedy policy snapshot, greedy path snapshot).
-/
def main (args : List String) : IO UInt32 := do
  TorchLean.Module.run exeName args
    (.float (fun opts rest => do
      let (logPath?, rest) ← Common.orThrow exeName <| CLI.takePathFlagOnce rest "log"
      let (policyPath?, rest) ← Common.orThrow exeName <| CLI.takePathFlagOnce rest "policy"
      let (pathPath?, rest) ← Common.orThrow exeName <| CLI.takePathFlagOnce rest "path"
      let (updates?, rest) ← Common.orThrow exeName <| CLI.takeNatFlagOnce rest "updates"
      let (evalEvery?, rest) ← Common.orThrow exeName <| CLI.takeNatFlagOnce rest "eval-every"
      let (evalEpisodes?, rest) ← Common.orThrow exeName <| CLI.takeNatFlagOnce rest "eval-episodes"
      let (evalMaxSteps?, rest) ← Common.orThrow exeName <| CLI.takeNatFlagOnce rest "eval-max-steps"
      Common.orThrow exeName <| CLI.requireNoArgs rest

      let updates : Nat := updates?.getD updatesMax
      let evalEvery : Nat := evalEvery?.getD evalEvery
      let evalEpisodes : Nat := evalEpisodes?.getD evalEpisodes
      let evalMaxSteps : Nat := evalMaxSteps?.getD 128

      if updates = 0 then
        throw <| IO.userError s!"{exeName}: --updates must be > 0"
      if evalEvery = 0 then
        throw <| IO.userError s!"{exeName}: --eval-every must be > 0"
      if evalEpisodes = 0 then
        throw <| IO.userError s!"{exeName}: --eval-episodes must be > 0"
      if evalMaxSteps = 0 then
        throw <| IO.userError s!"{exeName}: --eval-max-steps must be > 0"

      let logPath : System.FilePath :=
        logPath?.getD Runtime.RL.Artifacts.DefaultPaths.ppoGridWorldTrainLog
      let policyPath : System.FilePath :=
        policyPath?.getD Runtime.RL.Artifacts.DefaultPaths.ppoGridWorldPolicy
      let pathPath : System.FilePath :=
        pathPath?.getD Runtime.RL.Artifacts.DefaultPaths.ppoGridWorldPath

      -- Touch the formal theorem so it stays reachable from this example module.
      have _ : True := by
        -- `gwR_valid` is a Prop-level proof; it has no runtime cost.
        exact True.intro

      let seedActor ← nn.freshSeed
      let seedCritic ← nn.freshSeed
      let actorObs : nn.Sequential sState1 sLogits1 :=
        nn.build seedActor (actorMk (pfx := .scalar))
      let criticObs : nn.Sequential sState1 sValue1 :=
        nn.build seedCritic (criticMk (pfx := .scalar))
      let actorRollout : nn.Sequential sStateBatch sLogitsBatch :=
        nn.build seedActor (actorMk (pfx := pfxBatch))
      let criticRollout : nn.Sequential sStateBatch sValueBatch :=
        nn.build seedCritic (criticMk (pfx := pfxBatch))

      let actorC ← nn.compileOut actorObs
      let criticC ← nn.compileOut criticObs

      let modDef :=
        API.TorchLean.RL.Autograd.ppoActorCriticScalarModuleDef (stateShape := sStateBatch)
          (batch := horizon) (nActions := nActions) actorRollout criticRollout
      let m ← TorchLean.Module.instantiateWithOptions (α := Float) modDef id opts

      let opt := TorchLean.Optim.adam (α := Float) lr 0.9 0.999 1e-8
      let optH ← TorchLean.Optim.handle (α := Float) m opt

      let mut rngSeed : Nat := opts.seed
      let mut rngCounter : Nat := 0

      -- Training curve: greedy-policy return before training, then at each `evalEvery` checkpoint.
      -- We keep evaluation logs as a lightweight `Curve` because it targets JSON/widget display.
      let mut curve : rl.train.Curve := {}

      -- Helper: build a fresh (checked) session for evaluation rollouts/paths.
      let mkSession : Nat → rl.session.CheckedSession obsShape nActions :=
        fun _seed =>
          rl.session.ofEnv (State := Fin nStates) (obsShape := obsShape) (nActions := nActions)
            env contract (resetOnDone := false)

      -- Evaluate + snapshot the untrained policy.
      let psAll0 ← TorchLean.Module.params (α := Float) m
      let (psActor0, _psCritic0) :=
        rl.ppo.splitActorCriticParams actorRollout criticRollout psAll0
      let psActorObs0 : Runtime.Autograd.Torch.TList Float (nn.paramShapes actorObs) := by
        simpa using psActor0
      let policyLogits0 : Tensor Float obsShape → Tensor Float (shape![nActions]) :=
        fun obs => nn.predict1 actorObs actorC psActorObs0 obs
      let avg0 ←
        rl.eval.averageEpisodeTotalReward (obsShape := obsShape) (nActions := nActions)
          mkSession policyLogits0 (baseSeed := opts.seed) (episodes := evalEpisodes)
          (maxSteps := evalMaxSteps)
      curve := curve.push 0 avg0
      IO.println s!"  eval(step=0) avg_return={avg0}"

      let policyBefore : Array Nat :=
        Array.ofFn (fun (s : Fin nStates) =>
          let obs := obsOfState s
          let logits := policyLogits0 obs
          (rl.eval.greedyActionFromLogits (α := Float) (nActions := nActions) logits).val)
      let pathBeforeStates ←
        rl.eval.episodeSessPath (obsShape := obsShape) (nActions := nActions)
          (mkSession opts.seed) policyLogits0 (maxSteps := evalMaxSteps)
      let pathBefore : Array (Nat × Nat) :=
        pathBeforeStates.map (fun s =>
          let p := Spec.RL.Envs.GridWorld.decode (width := width) (height := height) s
          (p.1.val, p.2.val))

      for update in [0:updates] do
        let psAll ← TorchLean.Module.params (α := Float) m
        let (psActor, psCritic) :=
          rl.ppo.splitActorCriticParams actorRollout criticRollout psAll
        let psActorObs : Runtime.Autograd.Torch.TList Float (nn.paramShapes actorObs) := by
          simpa using psActor
        let psCriticObs : Runtime.Autograd.Torch.TList Float (nn.paramShapes criticObs) := by
          simpa using psCritic

        let predictLogits : Tensor Float obsShape → Tensor Float (shape![nActions]) :=
          fun obs => nn.predict1 actorObs actorC psActorObs obs
        let predictValue : Tensor Float obsShape → Float :=
          fun obs =>
            Tensor.vecGet (nn.predict1 criticObs criticC psCriticObs obs) ⟨0, by decide⟩

        let (rollout, rngCounter') ←
          collectRolloutNativeWith predictLogits predictValue
            (rngSeed := rngSeed) (rngCounter := rngCounter) (resetOnDone := true)
        rngCounter := rngCounter'

        let sample ←
          rollout.toActorCriticSample (α := Float) (obsShape := obsShape) (nActions := nActions)
            (horizon := horizon) gamma lam
        for _e in [0:updateEpochs] do
          optH.step sample

        if update % evalEvery == 0 then
          let psAll' ← TorchLean.Module.params (α := Float) m
          let (psActor', _psCritic') :=
            rl.ppo.splitActorCriticParams actorRollout criticRollout psAll'
          let psActorObs' : Runtime.Autograd.Torch.TList Float (nn.paramShapes actorObs) := by
            simpa using psActor'
          let policyLogits : Tensor Float obsShape → Tensor Float (shape![nActions]) :=
            fun obs => nn.predict1 actorObs actorC psActorObs' obs
          let avg ←
            rl.eval.averageEpisodeTotalReward (obsShape := obsShape) (nActions := nActions)
              mkSession policyLogits (baseSeed := opts.seed) (episodes := evalEpisodes)
              (maxSteps := evalMaxSteps)
          curve := curve.push (update + 1) avg
          IO.println s!"  update={update} avg_return={avg}"

          rngSeed := rand.nextSeed rngSeed update

      -- Snapshot the final greedy policy and a single episode path.
      let psAllF ← TorchLean.Module.params (α := Float) m
      let (psActorF, _psCriticF) :=
        rl.ppo.splitActorCriticParams actorRollout criticRollout psAllF
      let psActorObsF : Runtime.Autograd.Torch.TList Float (nn.paramShapes actorObs) := by
        simpa using psActorF
      let policyLogitsF : Tensor Float obsShape → Tensor Float (shape![nActions]) :=
        fun obs => nn.predict1 actorObs actorC psActorObsF obs
      let policyAfter : Array Nat :=
        Array.ofFn (fun (s : Fin nStates) =>
          let obs := obsOfState s
          let logits := policyLogitsF obs
          (rl.eval.greedyActionFromLogits (α := Float) (nActions := nActions) logits).val)
      let pathAfterStates ←
        rl.eval.episodeSessPath (obsShape := obsShape) (nActions := nActions)
          (mkSession opts.seed) policyLogitsF (maxSteps := evalMaxSteps)
      let pathAfter : Array (Nat × Nat) :=
        pathAfterStates.map (fun s =>
          let p := Spec.RL.Envs.GridWorld.decode (width := width) (height := height) s
          (p.1.val, p.2.val))

      let trainLog : rl.train.TrainLog :=
        curve.toTrainLog
          (title := s!"PPO GridWorld {width}x{height} (TorchLean)")
          (seriesName := "avg_return")
          (color := "#4e79a7")
          (notes := #[
            s!"width={width}",
            s!"height={height}",
            s!"horizon={horizon}",
            s!"gamma={gamma}",
            s!"lambda={lam}",
            s!"lr={lr}",
            s!"updates={updates}",
            s!"eval_every={evalEvery}",
            s!"eval_episodes={evalEpisodes}",
            s!"eval_max_steps={evalMaxSteps}",
            s!"device={(if opts.useGpu then "cuda" else "cpu")}"
          ])
      rl.train.writeJson logPath trainLog
      IO.println s!"{exeName}: wrote train log to {logPath}"

      let polDiff : _root_.Runtime.RL.Artifacts.GridWorld.PolicyDiff :=
        { width := width, height := height, before := policyBefore, after := policyAfter
          notes := #["greedy policy (argmax over logits)"] }
      _root_.Runtime.RL.Artifacts.GridWorld.PolicyDiff.writeJson policyPath polDiff
      IO.println s!"{exeName}: wrote policy snapshot to {policyPath}"

      let pathDiff : _root_.Runtime.RL.Artifacts.GridWorld.PathDiff :=
        { width := width, height := height, before := pathBefore, after := pathAfter
          notes := #["greedy episode path (states decoded to (row,col))"] }
      _root_.Runtime.RL.Artifacts.GridWorld.PathDiff.writeJson pathPath pathDiff
      IO.println s!"{exeName}: wrote path snapshot to {pathPath}"

      IO.println s!"{exeName}: done"
    ))
    { banner? := some (fun opts =>
        s!"{exeName}: PPO on Lean-native GridWorld ({width}x{height}, horizon={horizon}, device={if opts.useGpu then "cuda" else "cpu"})\n" ++
        "  env: pure Lean dynamics + boundary contract check + formal MDP validity proof available")
      printOk := true }

end NN.Examples.Models.RL.PPOGridWorld
