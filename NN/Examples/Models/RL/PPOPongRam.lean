/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

End-to-end PPO example: train an actor-critic on Atari Pong (ALE) using TorchLean.
-/

module

public import NN
public import NN.API.Models.PPO
public import NN.Runtime.RL.Artifacts.DefaultPaths

/-!
# PPO on Atari Pong (RAM Observations) (Executable Demo)

This example mirrors `NN/Examples/Models/RL/PPOCartPole.lean`, but targets an Atari game via the
Arcade Learning Environment (ALE) registered into Gymnasium as `ALE/Pong-v5`.

Why "RAM" observations?
- Pixel-based Atari PPO is absolutely doable, but a JSON-lines subprocess bridge is not the right
  transport if you want millions of steps/hour. RAM observations (`obs_type="ram"`, shape `128`)
  keep the bridge lightweight and make this run viable as a native Lean executable.

The key TorchLean interface remains the same:

- **Algorithm math** (GAE, PPO clipped objective) is Lean definitions.
- **Autograd program** (PPO loss) is a TorchLean backend-generic program (CPU or CUDA).
- **Trust boundary** is explicit: every externally sampled transition is checked by
  `Runtime.RL.Boundary.Contract` before it can influence training.

## Dependencies

Atari/ALE environments require `ale-py` and a recent `gymnasium`:

```bash
python3 -m pip install --user gymnasium>=1.0 ale-py
```

## CLI flags

- `--cuda`: run the Torch backend on CUDA (requires building with `-K cuda=true`).
- `--updates <n>`: number of PPO updates to run.
- `--eval-every <n>`: evaluate the greedy policy every `n` updates.
- `--eval-episodes <n>`: number of evaluation episodes per checkpoint.
- `--eval-max-steps <n>`: maximum steps per evaluation episode.
- `--log <path>`: write the widget log JSON to a custom path.

Run (from the repo root):

```bash
python3 -m pip install --user gymnasium>=1.0 ale-py
lake exe torchlean ppo_pong_ram
lake build -R -K cuda=true && lake exe torchlean ppo_pong_ram --cuda
```

Artifacts:
- Writes `data/rl/ppo_pong_ram_trainlog.json` by default (override with `--log`).
- Visualize it in the editor via `NN/Examples/RL/PPOPongRamView.lean`.

References (primary):
- Schulman et al., "Proximal Policy Optimization Algorithms" (2017): https://arxiv.org/abs/1707.06347
- Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation" (2015):
  https://arxiv.org/abs/1506.02438
- Williams, "Simple statistical gradient-following algorithms for connectionist reinforcement learning"
  (REINFORCE, 1992): https://doi.org/10.1007/BF00992696
- Machado et al., "Revisiting the Arcade Learning Environment: Evaluation Protocols and Open Problems"
  (2018): https://arxiv.org/abs/1709.06009
- ALE docs (environment catalogue and versioned `ALE/...-v5` ids): https://ale.farama.org/
- Gymnasium API docs (reset/step, `terminated` vs `truncated`): https://gymnasium.farama.org/
-/

@[expose] public section

open Spec Tensor
open NN.API

namespace NN.Examples.Models.RL.PPOPongRam

/-- Name of this executable target (used in CLI error messages and banners). -/
def exeName : String := "torchlean ppo_pong_ram"

/-!
## Configuration
-/

/-- Atari environment id passed to the Python subprocess. -/
def envId : String := "ALE/Pong-v5"

/-- Relative path to the Python Gymnasium bridge script (spawned as a subprocess). -/
def gymServerScript : String := "scripts/rl/gymnasium_server.py"

/--
Pong RAM observation dimension.

Gymnasium exposes RAM as `Box(0, 255, (128,), uint8)` when `obs_type="ram"`.
-/
def stateDim : Nat := 128

/-- Number of discrete actions in Pong under ALE's reduced action set. -/
def nActions : Nat := 6

/-- Width of the hidden layer in the actor and critic MLPs. -/
def hiddenDim : Nat := 64

/-- PPO rollout horizon (also the training batch size for this run). -/
def horizon : Nat := 128

/-- Discount factor used in returns / GAE. -/
def gamma : Float := 0.99

/-- GAE(λ) parameter controlling the bias/variance tradeoff of advantage estimates. -/
def lam : Float := 0.95

/-- Adam learning rate. -/
def lr : Float := 2.5e-4

/-- Number of PPO optimization epochs per collected rollout batch. -/
def updateEpochs : Nat := 1

/-- Default maximum number of PPO updates (override with `--updates`). -/
def updatesMax : Nat := 2000

/-- Default evaluation checkpoint interval (override with `--eval-every`). -/
def evalEvery : Nat := 100

/-- Default evaluation episodes per checkpoint (override with `--eval-episodes`). -/
def evalEpisodes : Nat := 5

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

We use MLPs over RAM. For pixel observations you would typically use a CNN (see
`NN.GraphSpec.Models.TorchLean.Cnn`) and wrap the environment with Atari preprocessing.
-/

def modelCfg : nn.models.PPOActorCriticConfig :=
  { obsDim := stateDim, hiddenDim := hiddenDim, nActions := nActions }

/-- Construct the actor network as an MLP mapping RAM observations to action logits. -/
def actorMk (pfx : Shape) : nn.M (nn.Sequential (pfx.appendDim stateDim) (pfx.appendDim nActions)) :=
  nn.models.ppoActor modelCfg pfx

/-- Construct the critic network as an MLP mapping RAM observations to a scalar value estimate. -/
def criticMk (pfx : Shape) : nn.M (nn.Sequential (pfx.appendDim stateDim) (pfx.appendDim 1)) :=
  nn.models.ppoCritic modelCfg pfx

/-!
## Gymnasium / ALE bridge

We request RAM observations by passing `{"obs_type": "ram"}` to `gym.make` through the bridge's
`--make-kwargs` option. The server also auto-registers `ale_py` when `envId` starts with `ALE/`.
-/

def makeKwargs : List (String × Lean.Json) :=
  [("obs_type", .str "ram")]

def contract : rl.boundary.Contract obsShape nActions :=
  { checkObsFinite := true
    checkRewardFinite := true
    -- RAM bytes live in `[0,255]` by construction. This range check is a cheap sanity guard
    -- against protocol bugs or unexpected wrappers.
    obsRange? := some (0, 255)
    rewardRange? := none
    requireExclusiveDoneFlags := false }

/-!
## Main Training Loop
-/

def main (args : List String) : IO UInt32 := do
  TorchLean.Module.run exeName args
    (.float (fun opts rest => do
      let (logPath?, rest) ← Common.orThrow exeName <| CLI.takePathFlagOnce rest "log"
      let (updates?, rest) ← Common.orThrow exeName <| CLI.takeNatFlagOnce rest "updates"
      let (evalEvery?, rest) ← Common.orThrow exeName <| CLI.takeNatFlagOnce rest "eval-every"
      let (evalEpisodes?, rest) ← Common.orThrow exeName <| CLI.takeNatFlagOnce rest "eval-episodes"
      let (evalMaxSteps?, rest) ← Common.orThrow exeName <| CLI.takeNatFlagOnce rest "eval-max-steps"
      Common.orThrow exeName <| CLI.requireNoArgs rest

      let updates : Nat := updates?.getD updatesMax
      let evalEvery : Nat := evalEvery?.getD evalEvery
      let evalEpisodes : Nat := evalEpisodes?.getD evalEpisodes
      let evalMaxSteps : Nat := evalMaxSteps?.getD 10000

      if updates = 0 then
        throw <| IO.userError s!"{exeName}: --updates must be > 0"
      if evalEvery = 0 then
        throw <| IO.userError s!"{exeName}: --eval-every must be > 0"
      if evalEpisodes = 0 then
        throw <| IO.userError s!"{exeName}: --eval-episodes must be > 0"
      if evalMaxSteps = 0 then
        throw <| IO.userError s!"{exeName}: --eval-max-steps must be > 0"

      let logPath : System.FilePath :=
        logPath?.getD Runtime.RL.Artifacts.DefaultPaths.ppoPongRamTrainLog

      IO.eprintln s!"  starting env: {envId} (obs_type=ram)"
      let gym ←
        rl.gym.client.spawn (obsShape := obsShape) (nActions := nActions) gymServerScript envId contract
          (makeKwargs := makeKwargs)
      try
        IO.eprintln "  building actor/critic..."
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

        IO.eprintln "  compiling actor/critic..."
        let actorC ← nn.compileOut actorObs
        let criticC ← nn.compileOut criticObs

        IO.eprintln "  initializing module + optimizer..."
        let modDef :=
          API.TorchLean.RL.Autograd.ppoActorCriticScalarModuleDef (stateShape := sStateBatch)
            (batch := horizon) (nActions := nActions) actorRollout criticRollout
        let m ← TorchLean.Module.instantiateWithOptions (α := Float) modDef id opts
        IO.eprintln "  module ready"

        let opt := TorchLean.Optim.adam (α := Float) lr 0.9 0.999 1e-8
        let optH ← TorchLean.Optim.handle (α := Float) m opt
        IO.eprintln "  optimizer ready"

        let mut rngSeed : Nat := opts.seed
        let mut rngCounter : Nat := 0

        let mut curve : rl.train.Curve := {}

        let mkSession : Nat → rl.session.CheckedSession obsShape nActions :=
          fun seed =>
            rl.session.gymnasium (obsShape := obsShape) (nActions := nActions) gym
              (seed? := some seed) (resetOnDone := false)

        -- Evaluate once before training (step=0).
        do
          IO.eprintln "  evaluating initial policy..."
          let psAll0 ← TorchLean.Module.params (α := Float) m
          let (psActor0, _psCritic0) :=
            rl.ppo.splitActorCriticParams actorRollout criticRollout psAll0
          let psActorObs0 : Runtime.Autograd.Torch.TList Float (nn.paramShapes actorObs) := by
            simpa using psActor0
          let policyLogits0 : Tensor Float obsShape → Tensor Float (shape![nActions]) :=
            fun obs => nn.predict1 actorObs actorC psActorObs0 (Spec.mapTensor (fun x => x / 255.0) obs)
          let avg0 ←
            rl.eval.averageEpisodeTotalReward (obsShape := obsShape) (nActions := nActions)
              mkSession policyLogits0 (baseSeed := 9000) (episodes := evalEpisodes)
              (maxSteps := evalMaxSteps)
          curve := curve.push 0 avg0
          IO.eprintln s!"  eval(step=0) avg_return={avg0}"

        for update in [0:updates] do
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
              (horizon := horizon)
              (castObs := fun x => x / 255.0) (castReward := id)
              gym predictLogits predictValue
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
              fun obs => nn.predict1 actorObs actorC psActorObs' (Spec.mapTensor (fun x => x / 255.0) obs)
            let avg ←
              rl.eval.averageEpisodeTotalReward (obsShape := obsShape) (nActions := nActions)
                mkSession policyLogits (baseSeed := 9000 + update) (episodes := evalEpisodes)
                (maxSteps := evalMaxSteps)
            curve := curve.push (update + 1) avg
            IO.eprintln s!"  update={update} avg_return={avg}"

          rngSeed := rand.nextSeed rngSeed update

        let trainLog : rl.train.TrainLog :=
          curve.toTrainLog
            (title := s!"PPO {envId} (RAM, TorchLean)")
            (seriesName := "avg_return")
            (color := "#f28e2b")
            (notes := #[
              s!"env_id={envId}",
              s!"obs_type=ram",
              s!"horizon={horizon}",
              s!"gamma={gamma}",
              s!"lambda={lam}",
              s!"lr={lr}",
              s!"eval_every={evalEvery}",
              s!"eval_episodes={evalEpisodes}",
              s!"device={(if opts.useGpu then "cuda" else "cpu")}"
            ])
        rl.train.writeJson logPath trainLog
        IO.eprintln s!"{exeName}: wrote train log to {logPath}"
        IO.eprintln s!"{exeName}: done"
      finally
        gym.close
    ))
    { banner? := some (fun opts =>
        s!"{exeName}: PPO on {envId} (obs=ram, horizon={horizon}, device={if opts.useGpu then "cuda" else "cpu"})\n" ++
        "  env: Python Gymnasium subprocess (ALE) + Lean boundary contract")
      printOk := true }

end NN.Examples.Models.RL.PPOPongRam
