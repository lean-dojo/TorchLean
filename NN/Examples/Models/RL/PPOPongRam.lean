/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

End-to-end PPO example: train an actor-critic on Atari Pong (ALE) using TorchLean.
-/

module

public import NN.API
public import NN.Examples.Support
public import NN.Runtime.RL.Artifacts.DefaultPaths

/-!
# PPO on Atari Pong (RAM Observations) (Executable Example)

This example mirrors `NN/Examples/Models/RL/PPOCartPole.lean`, but targets an Atari game via the
Arcade Learning Environment (ALE) registered into Gymnasium as `ALE/Pong-v5`.

Why "RAM" observations?
- Pixel-based Atari PPO is absolutely doable, but a JSON-lines subprocess bridge is not the right
  transport if you want millions of steps/hour. RAM observations (`obs_type="ram"`, shape `128`)
  keep the bridge compact and make this run viable as a native Lean executable.

The key TorchLean interface remains the same:

- **Algorithm math** (GAE, PPO clipped objective) is Lean definitions.
- **Autograd program** (PPO loss) is a TorchLean backend-generic program (CPU or CUDA).
- **Trust boundary** is explicit: every externally sampled transition is checked by
  `Runtime.RL.Boundary.Contract` before it can influence training.

## Dependencies

Atari/ALE environments require `ale-py` and a recent `gymnasium`:

```bash
python3 -m pip install --user 'gymnasium>=1.0' ale-py
```

## CLI flags

- `--device cuda`: run the Torch backend on CUDA (requires building with `-K cuda=true`).
- `--updates <n>`: number of PPO updates to run.
- `--eval-every <n>`: evaluate the greedy policy every `n` updates.
- `--eval-episodes <n>`: number of evaluation episodes per checkpoint.
- `--eval-max-steps <n>`: maximum steps per evaluation episode.
- `--log <path>`: write the widget log JSON to a custom path.

This command is optional in the sense that it depends on a compatible external ALE/Gymnasium
installation. It is available through the runner but is not part of the default quick-check list.

Artifacts:
- Writes `data/rl/ppo_pong_ram_trainlog.json` by default (override with `--log`).
- Visualize it in the editor via `NN/Examples/Models/RL/Views/PPOPongRam.lean`.

References (primary):
- Schulman et al., "Proximal Policy Optimization Algorithms" (2017):
  https://arxiv.org/abs/1707.06347
- Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation"
  (2015): https://arxiv.org/abs/1506.02438
- Williams, "Simple statistical gradient-following algorithms for connectionist reinforcement
  learning" (REINFORCE, 1992): https://doi.org/10.1007/BF00992696
- Machado et al., "Revisiting the Arcade Learning Environment: Evaluation Protocols and Open
  Problems" (2018): https://arxiv.org/abs/1709.06009
- ALE docs (environment catalogue and versioned `ALE/...-v5` ids): https://ale.farama.org/
- Gymnasium API reference (reset/step, `terminated` vs `truncated`): https://gymnasium.farama.org/
-/

@[expose] public section

open TorchLean

namespace NN.Examples.Models.RL.PPOPongRam

/-- Name used in CLI error messages and banners. -/
def exeName : String := "ppo_pong_ram"

/-- Help text for the optional ALE/Pong RAM PPO runner. -/
def usage : String :=
  String.intercalate "\n"
    [ "Usage:"
    , "  lake -R -K cuda=true exe torchlean ppo_pong_ram --device cuda [PPO flags]"
    , ""
    , "PPO flags:"
    , "  --updates N          number of PPO updates"
    , "  --eval-every N       evaluate every N updates"
    , "  --eval-episodes N    evaluation episodes per checkpoint"
    , "  --eval-max-steps N   maximum steps per evaluation episode"
    , "  --log PATH|off       training-curve JSON path, or disable logging"
    , "  --check-env-only     start ALE, reset once, take one checked step, then exit"
    , ""
    , "External dependency:"
    , "  python3 -m pip install --user 'gymnasium>=1.0' ale-py"
    ]

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
def observationWidth : Nat := 128

/-- Number of discrete actions in Pong under ALE's reduced action set. -/
def actionCount : Nat := 6

/-- Width of the hidden layer in the actor and critic MLPs. -/
def hiddenWidth : Nat := 64

/-- PPO rollout horizon (also the training batch size for this run). -/
def horizon : Nat := 128

/-- Discount factor used in returns / GAE. -/
def discountFactor : Float := 0.99

/-- GAE(λ) parameter controlling the bias/variance tradeoff of advantage estimates. -/
def gaeLambda : Float := 0.95

/-- Adam learning rate used for the Pong RAM actor-critic update. -/
def learningRate : Float := 2.5e-4

/-- Number of PPO optimization epochs per collected rollout batch. -/
def updateEpochs : Nat := 1

/-- Default maximum number of PPO updates (override with `--updates`). -/
def maxUpdates : Nat := 2000

/-- Default evaluation checkpoint interval (override with `--eval-every`). -/
def defaultEvaluationInterval : Nat := 100

/-- Default evaluation episodes per checkpoint (override with `--eval-episodes`). -/
def defaultEvaluationEpisodes : Nat := 5

instance : NeZero horizon := ⟨by decide⟩
instance : NeZero actionCount := ⟨by decide⟩

/-- The observation tensor shape used by this run: `[..., observationWidth]`. -/
def observation : Shape := [observationWidth]

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
## Model (Actor + Critic)

We use MLPs over RAM. Pixel observations can instead use the arbitrary-rank convolutional model
from `TorchLean.nn.models.cnn` after applying the appropriate Atari preprocessing.
-/

abbrev modelConfig : nn.models.PPO.Config :=
  { observationWidth := observationWidth
    hiddenWidth := hiddenWidth
    actionCount := actionCount }

/-- Construct the actor network as an MLP mapping RAM observations to action logits. -/
def actor (batchShape : Shape := []) :
    nn.Builder (nn.Sequential (modelConfig.input batchShape)
      (modelConfig.actorOutput batchShape)) :=
  nn.models.PPO.actor modelConfig batchShape

/-- Construct the critic network as an MLP mapping RAM observations to a scalar value estimate. -/
def critic (batchShape : Shape := []) :
    nn.Builder (nn.Sequential (modelConfig.input batchShape)
      (modelConfig.criticOutput batchShape)) :=
  nn.models.PPO.critic modelConfig batchShape

/-!
## Gymnasium / ALE bridge

We request RAM observations by passing `{"obs_type": "ram"}` to `gym.make` through the bridge's
`--make-kwargs` option. The server also auto-registers `ale_py` when `envId` starts with `ALE/`.
-/

def makeKwargs : Array (String × Lean.Json) :=
  #[("obs_type", .str "ram")]

/--
What Lean insists on before it will believe anything the Python environment sends.

Every observation and reward must be finite, and RAM bytes must lie in `[0, 255]`. That last check
is
not about the emulator, which cannot produce anything else; it is about the protocol and the adapter
between them, where a byte-order or dtype mistake would show up as out-of-range values.
-/
def contract : rl.boundary.Contract observation actionCount :=
  { checkObsFinite := true
    checkRewardFinite := true
    -- RAM bytes live in `[0,255]` by construction. This range check guards
    -- against protocol bugs or unexpected adapters.
    obsRange? := some (0, 255)
    rewardRange? := none
    requireExclusiveDoneFlags := false }

/--
Start ALE, reset once, take one checked step, and close the subprocess.

This exercises the Gymnasium subprocess, ALE registration, RAM observation shape handshake, and
Lean side boundary contract as the full PPO runner, without collecting a 128-step rollout.
-/
def checkEnvOnly : IO Unit := do
  IO.eprintln s!"  starting env: {envId} (obs_type=ram)"
  let gym ←
    rl.gym.client.spawn
      (obsShape := observation) (nActions := actionCount) gymServerScript envId contract
      (makeKwargs := makeKwargs)
  try
    let session ← rl.gym.session.start gym (seed? := some 0)
    let (transition, _) ←
      rl.gym.session.stepChecked session 0 (resetOnDone := false)
    IO.println <|
      s!"{exeName}: env check ok reward={transition.reward} " ++
      s!"terminated={transition.terminated} truncated={transition.truncated}"
  finally
    rl.gym.client.close gym

/-!
## Main Training Loop
-/

def main (args : List String) : IO UInt32 := do
  if args.contains "--help" || args.contains "-h" then
    IO.println usage
    return 0
  if args.contains "--check-env-only" then
    let args := args.erase "--check-env-only"
    return ←
      Module.Command.run
        (config := {
          banner? := some <| Support.bannerWithDeviceDetails
            exeName
            s!"PPO on {envId} (obs=ram, env check only)"
            "  env: Python Gymnasium subprocess (ALE) + Lean boundary contract"
          printSuccess := true })
        exeName args
        (.native fun _opts rest => do
          CLI.orThrow exeName <| CLI.checkNoArgs rest
          checkEnvOnly)
  Module.Command.run
    (config := {
      banner? := some <| Support.bannerWithDeviceDetails
        exeName
        s!"PPO on {envId} (obs=ram, horizon={horizon})"
        "  env: Python Gymnasium subprocess (ALE) + Lean boundary contract"
      printSuccess := true })
    exeName args
    (.native fun runtime rest => do
      let (ppo, rest) ← CLI.orThrow exeName <|
        rl.cli.PPOOptions.parse
          exeName rest Runtime.RL.Artifacts.DefaultPaths.ppoPongRamTrainLog
          (defaultUpdateCount := maxUpdates)
          (defaultEvaluationInterval := defaultEvaluationInterval)
          (defaultEvaluationEpisodes := defaultEvaluationEpisodes)
          (defaultMaximumEvaluationSteps := 10000)
      CLI.orThrow exeName <| CLI.checkNoArgs rest

      let updateCount : Nat := ppo.updateCount
      let evaluationInterval : Nat := ppo.evaluationInterval
      let evaluationEpisodes : Nat := ppo.evaluationEpisodes
      let maximumEvaluationSteps : Nat := ppo.maximumEvaluationSteps

      IO.eprintln s!"  starting env: {envId} (obs_type=ram)"
      let gym ←
        rl.gym.client.spawn
          (obsShape := observation) (nActions := actionCount) gymServerScript envId contract
          (makeKwargs := makeKwargs)
      try
        IO.eprintln "  building actor/critic..."
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

        IO.eprintln "  lowering actor/critic to typed graphs..."
        let actorGraph ← nn.lowerToTypedGraph actorObs
        let criticGraph ← nn.lowerToTypedGraph criticObs

        IO.eprintln "  initializing module + optimizer..."
        let m ← rl.ppo.instantiateActorCritic
          (α := Float) (options := runtime)
          (batch := horizon) (nActions := actionCount)
          actorRollout criticRollout
        IO.eprintln "  module ready"

        let stepSample ←
          rl.ppo.trainingStep m
            (optim.adam { learningRate := learningRate })
        IO.eprintln "  optimizer ready"


        let mut curve : Training.Curve := {}

        let evaluationSessionAt :
            Nat → rl.session.CheckedSession observation actionCount :=
          fun seed =>
            rl.session.gymnasium
              (obsShape := observation) (nActions := actionCount) gym
              (seed? := some seed) (resetOnDone := false)

        -- Evaluate once before training (step=0).
        do
          IO.eprintln "  evaluating initial policy..."
          let psAll0 ← rl.ppo.state (α := Float) m
          let policy0 := rl.ppo.actorPolicy actorGraph actorRollout criticRollout psAll0
          let policyLogits0 :
              Tensor Float observation → Tensor Float actionLogits :=
            fun obs => policy0 (Tensor.map (fun x => x / 255.0) obs)
          let avg0 ←
            rl.eval.averageEpisodeTotalReward
              (obsShape := observation) (nActions := actionCount)
              evaluationSessionAt policyLogits0 (baseSeed := 9000)
              (episodes := evaluationEpisodes)
              (maxSteps := maximumEvaluationSteps)
          curve := curve.push 0 avg0
          IO.eprintln s!"  eval(step=0) avg_return={avg0}"

        curve ← rl.ppo.train discountFactor gaeLambda
          { updates := updateCount, epochs := updateEpochs,
            evaluationEvery := evaluationInterval, seed := runtime.seed }
          (fun update rngSeed rngCounter => do
              let psAll ← rl.ppo.state (α := Float) m
              let predictLogits :
                  Tensor Float observation → Tensor Float actionLogits :=
                rl.ppo.actorPolicy actorGraph actorRollout criticRollout psAll
              let predictValue : Tensor Float observation → Float :=
                rl.ppo.criticValue criticGraph actorRollout criticRollout psAll

              let (rollout, rngCounter') ←
                rl.ppo.collectRolloutFromGymnasium
                  (α := Float) (obsShape := observation) (nActions := actionCount)
                  (horizon := horizon)
                  (castObservation := fun x => x / 255.0) (castReward := id)
                  gym predictLogits predictValue
                  (rngSeed := rngSeed) (rngCounter := rngCounter) (resetSeed := update)
              pure (rollout, rngCounter'))
          stepSample
          (fun completedUpdates => do
              let psAll' ← rl.ppo.state (α := Float) m
              let policy := rl.ppo.actorPolicy actorGraph actorRollout criticRollout psAll'
              let policyLogits :
                  Tensor Float observation → Tensor Float actionLogits :=
                fun obs => policy (Tensor.map (fun x => x / 255.0) obs)
              let avg ←
                rl.eval.averageEpisodeTotalReward
                  (obsShape := observation) (nActions := actionCount)
                  evaluationSessionAt policyLogits (baseSeed := 9000 + completedUpdates)
                    (episodes := evaluationEpisodes)
                  (maxSteps := maximumEvaluationSteps)
              IO.eprintln s!"  update={completedUpdates} avg_return={avg}"
              pure (avg, false))
          curve

        Training.Curve.writeLog curve
          ppo.logDestination
          s!"PPO {envId} (RAM, TorchLean)"
          "avg_return"
          (color := "#f28e2b")
          #[
            s!"env_id={envId}",
            s!"obs_type=ram",
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
        IO.eprintln s!"{exeName}: done"
      finally
        gym.close
    )

end NN.Examples.Models.RL.PPOPongRam
