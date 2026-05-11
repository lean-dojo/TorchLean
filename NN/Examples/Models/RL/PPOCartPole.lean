/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

End-to-end PPO example: train an actor-critic on Gymnasium `CartPole-v1` using TorchLean.
-/

module

public import NN
public import NN.API.Models.PPO
public import NN.Runtime.RL.Artifacts.DefaultPaths

/-!
# PPO on Gymnasium CartPole (Executable Demo)

This example is intentionally “small but complete”:

- **Environment**: external Python Gymnasium (started as a subprocess).
- **Trust boundary**: every step is checked against a Lean-side contract
  (`Runtime.RL.Boundary.Contract`) before being used for training data.
- **Algorithm**: PPO with GAE (all update math is Lean definitions; the PPO loss is a TorchLean
  autograd program).

More concretely:

- The policy is a categorical distribution over discrete actions parameterized by logits
  `π_θ(a | s) = softmax(logits_θ(s))`.
- Advantages are computed using Generalized Advantage Estimation (GAE(λ)).
- PPO uses the clipped surrogate objective (plus a value-loss and optional entropy bonus, depending
  on the runtime configuration).

## CLI flags

- `--cuda`: run the Torch backend on CUDA (requires building with `-K cuda=true`).
- `--seed <n>`: deterministic seed for TorchLean RNG streams (and evaluation seeding).
- `--updates <n>`: limit the number of PPO rollout/update cycles.
- `--log <path>`: write the widget log JSON to a custom path.

Run (from the repo root):

```bash
python3 -m pip install --user 'gymnasium>=1.0'
lake exe torchlean ppo_cartpole
lake build -R -K cuda=true && lake exe torchlean ppo_cartpole --cuda
```

Artifacts:
- The executable writes a widget-friendly training curve JSON to
  `data/rl/ppo_cartpole_trainlog.json` (override with `--log <path>`).
- Visualize it in the editor via `NN/Examples/RL/PPOCartPoleView.lean`.

## What this run does (and does not) guarantee

- The PPO/GAE math and the autograd loss program are Lean definitions, so they are suitable targets
  for formal reasoning.
- When Gymnasium is external, TorchLean cannot prove the environment satisfies Markov/measurability
  assumptions. The trust-boundary contract turns some common assumptions (finite tensors, reward
  bounds, done-flag semantics) into checked preconditions.
- This is not a tuned “benchmark PPO” implementation. It is designed to be readable, typed, and
  easy to inspect with widgets.

References (primary):
- Schulman et al., "Proximal Policy Optimization Algorithms" (2017): https://arxiv.org/abs/1707.06347
- Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation" (2015):
  https://arxiv.org/abs/1506.02438
- Williams, "Simple statistical gradient-following algorithms for connectionist reinforcement learning"
  (REINFORCE, 1992): https://doi.org/10.1007/BF00992696
- Brockman et al., "OpenAI Gym" (2016): https://arxiv.org/abs/1606.01540
- Gymnasium API docs (reset/step, `terminated` vs `truncated`): https://gymnasium.farama.org/
- CartPole environment docs: https://gymnasium.farama.org/environments/classic_control/cart_pole/
-/

@[expose] public section

open Spec Tensor
open NN.API

namespace NN.Examples.Models.RL.PPOCartPole

/-- Name of this executable target (used in CLI error messages and banners). -/
def exeName : String := "torchlean ppo_cartpole"

/-!
## Configuration

We keep this example discrete-action and small (CartPole) so it runs quickly in a native Lean
executable.
-/

/-- Gymnasium environment id passed to the Python subprocess (see Gymnasium docs for supported ids). -/
def envId : String := "CartPole-v1"

/-- Relative path to the Python Gymnasium bridge script (spawned as a subprocess). -/
def gymServerScript : String := "scripts/rl/gymnasium_server.py"

/-- Observation vector dimension for CartPole (`Gymnasium` reports 4 floats). -/
def stateDim : Nat := 4

/-- Number of discrete actions for CartPole (left/right). -/
def nActions : Nat := 2

/-- Width of the hidden layer in the actor and critic MLPs. -/
def hiddenDim : Nat := 32

/-- PPO rollout horizon (also the training batch size for this run). -/
def horizon : Nat := 64

/-- Discount factor used in returns / GAE. -/
def gamma : Float := 0.99

/-- GAE(λ) parameter controlling the bias/variance tradeoff of advantage estimates. -/
def lam : Float := 0.95

/-- Adam learning rate. -/
def lr : Float := 3e-4

/-- Number of PPO optimization epochs per collected rollout batch. -/
def updateEpochs : Nat := 2

/-- Maximum number of PPO updates (training stops early if the "solved" criterion triggers). -/
def updatesMax : Nat := 1000

/-- Evaluate (greedy policy) every `evalEvery` PPO updates. -/
def evalEvery : Nat := 50

/-- Number of evaluation episodes per checkpoint. -/
def evalEpisodes : Nat := 5

/-- Stop early if average return meets/exceeds this threshold. -/
def solvedAvgReturn : Float := 475.0

instance : Fact (0 < horizon) := ⟨by decide⟩
instance : Fact (0 < nActions) := ⟨by decide⟩

/-- The observation tensor shape used by this run: `[..., stateDim]`. -/
def obsShape : Shape := shape![stateDim]

def pfxBatch : Shape := shape![horizon]
def sStateBatch : Shape := rl.ppo.StateBatchShape horizon obsShape
def sLogitsBatch : Shape := rl.ppo.LogitsBatchShape horizon nActions
def sScalarBatch : Shape := rl.ppo.ScalarBatchShape horizon
def sValueBatch : Shape := rl.ppo.ValueBatchShape horizon

def sState1 : Shape := Shape.scalar.appendDim stateDim
def sLogits1 : Shape := Shape.scalar.appendDim nActions
def sValue1 : Shape := Shape.scalar.appendDim 1

/-!
## Model (Actor + Critic)

We use the public `API.nn` facade, which provides "prefix-shape preserving" layers:
if `x` has shape `[..., inDim]`, `nn.linear inDim outDim (pfx := ...)` maps it to `[..., outDim]`.
-/

def modelCfg : nn.models.PPOActorCriticConfig :=
  { obsDim := stateDim, hiddenDim := hiddenDim, nActions := nActions }

/-- Construct the actor network as an MLP mapping observations to action logits. -/
def actorMk (pfx : Shape) : nn.M (nn.Sequential (pfx.appendDim stateDim) (pfx.appendDim nActions)) :=
  nn.models.ppoActor modelCfg pfx

/-- Construct the critic network as an MLP mapping observations to a scalar value estimate. -/
def criticMk (pfx : Shape) : nn.M (nn.Sequential (pfx.appendDim stateDim) (pfx.appendDim 1)) :=
  nn.models.ppoCritic modelCfg pfx

/-!
## Gymnasium Bridge

We talk to a small Python helper (`scripts/rl/gymnasium_server.py`) using the reusable runtime
bridge in `Runtime.RL.Gymnasium` (exposed as `rl.gym.*`).

The Lean-side trust-boundary contract (`rl.boundary.Contract`) is enforced on every step.
-/

/-!
## Evaluation

Evaluation helpers live in `NN.API.rl.eval` (runtime module `NN.Runtime.RL.Eval`).
-/

/-!
## Main Training Loop
-/

/-- Entry point for `lake exe torchlean ppo_cartpole`.

This executable:
- launches a Python Gymnasium subprocess for `CartPole-v1`,
- collects checked rollouts under `rl.boundary.Contract`,
- performs PPO updates on the Torch backend (CPU or CUDA via `--cuda`),
- writes a widget-friendly training curve JSON (default: `data/rl/ppo_cartpole_trainlog.json`).
-/
def main (args : List String) : IO UInt32 := do
  TorchLean.Module.run exeName args
    (.float (fun opts rest => do
      let (logPath?, rest) ← Common.orThrow exeName <| CLI.takePathFlagOnce rest "log"
      let (updates?, rest) ← Common.orThrow exeName <| CLI.takeNatFlagOnce rest "updates"
      Common.orThrow exeName <| CLI.requireNoArgs rest
      let logPath : System.FilePath :=
        logPath?.getD Runtime.RL.Artifacts.DefaultPaths.ppoCartPoleTrainLog
      let updatesLimit := updates?.getD updatesMax
      let contract : rl.boundary.Contract obsShape nActions :=
        { checkObsFinite := true
          checkRewardFinite := true
          obsRange? := none
          rewardRange? := none
          requireExclusiveDoneFlags := false }

      let gym ←
        rl.gym.client.spawn (obsShape := obsShape) (nActions := nActions) gymServerScript envId contract
      try
        -- Build actor/critic once from a single seed, then reuse them at both observation-time shapes
        -- (`pfx = scalar`) and rollout-time batched shapes (`pfx = horizon`).
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

        -- Training curve: greedy-policy evaluation return before training, then at each
        -- `evalEvery` checkpoint. We keep this as a lightweight `Curve` (arrays) because it is
        -- destined for JSON/widget display; the actual learning data is stored as typed tensors.
        let mut curve : rl.train.Curve := {}

        let mkSession : Nat → rl.session.CheckedSession obsShape nActions :=
          fun seed =>
            rl.session.gymnasium (obsShape := obsShape) (nActions := nActions) gym
              (seed? := some seed) (resetOnDone := false)

        -- Evaluate the untrained policy once (step=0).
        do
          let psAll0 ← TorchLean.Module.params (α := Float) m
          let (psActor0, _psCritic0) :=
            rl.ppo.splitActorCriticParams actorRollout criticRollout psAll0
          let psActorObs0 : Runtime.Autograd.Torch.TList Float (nn.paramShapes actorObs) := by
            simpa using psActor0
          let policyLogits0 : Tensor Float obsShape → Tensor Float (shape![nActions]) :=
            fun obs => nn.predict1 actorObs actorC psActorObs0 obs
          let avg0 ←
            rl.eval.averageEpisodeTotalReward (obsShape := obsShape) (nActions := nActions)
              mkSession policyLogits0 (baseSeed := 1000) (episodes := evalEpisodes)
              (maxSteps := 500)
          curve := curve.push 0 avg0
          IO.println s!"  eval(step=0) avg_return={avg0}"

        for update in [0:updatesLimit] do
          let psAll ← TorchLean.Module.params (α := Float) m
          let (psActor, psCritic) :=
            rl.ppo.splitActorCriticParams actorRollout criticRollout psAll
          let psActorObs : Runtime.Autograd.Torch.TList Float (nn.paramShapes actorObs) := by
            simpa using psActor
          let psCriticObs : Runtime.Autograd.Torch.TList Float (nn.paramShapes criticObs) := by
            simpa using psCritic

          let predictLogits : Tensor Float obsShape → Tensor Float sLogits1 :=
            fun obs => nn.predict1 actorObs actorC psActorObs obs
          let predictValue : Tensor Float obsShape → Float :=
            fun obs =>
              Tensor.vecGet (nn.predict1 criticObs criticC psCriticObs obs) ⟨0, by decide⟩
          let (rollout, rngCounter') ←
            rl.ppo.collectRolloutWith (α := Float) (obsShape := obsShape) (nActions := nActions)
              (horizon := horizon) (castObs := id) (castReward := id) gym predictLogits predictValue
              (rngSeed := rngSeed) (rngCounter := rngCounter) (resetSeed := update)
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
                mkSession policyLogits (baseSeed := 1000 + update) (episodes := evalEpisodes)
                (maxSteps := 500)
            curve := curve.push (update + 1) avg
            IO.println s!"  update={update} avg_return={avg}"
            if avg ≥ solvedAvgReturn then
              IO.println s!"{exeName}: solved (avg_return ≥ {solvedAvgReturn})"
              break

          -- keep the RNG seed moving so repeated runs can explore different traces if desired
          rngSeed := rand.nextSeed rngSeed update

        let trainLog : rl.train.TrainLog :=
          curve.toTrainLog
            (title := s!"PPO {envId} (TorchLean)")
            (seriesName := "avg_return")
            (color := "#4e79a7")
            (notes := #[
              s!"env_id={envId}",
              s!"horizon={horizon}",
              s!"gamma={gamma}",
              s!"lambda={lam}",
              s!"lr={lr}",
              s!"updates={updatesLimit}",
              s!"eval_every={evalEvery}",
              s!"eval_episodes={evalEpisodes}",
              s!"device={(if opts.useGpu then "cuda" else "cpu")}"
            ])
        rl.train.writeJson logPath trainLog
        IO.println s!"{exeName}: wrote train log to {logPath}"
        IO.println s!"{exeName}: done"
      finally
        gym.close
    ))
    { banner? := some (fun opts =>
        s!"{exeName}: PPO on {envId} (horizon={horizon}, device={if opts.useGpu then "cuda" else "cpu"})\n" ++
        "  env: Python Gymnasium subprocess (JSON-lines bridge) + Lean boundary contract")
      printOk := true }

end NN.Examples.Models.RL.PPOCartPole
