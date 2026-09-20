/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

End-to-end PPO example: train an actor-critic on Gymnasium `CartPole-v1` using TorchLean.
-/

module

public import NN.API
public import NN.Examples.Support
public import NN.Runtime.RL.Artifacts.DefaultPaths

/-!
# PPO on Gymnasium CartPole (Executable Example)

This example is small but complete:

- **Environment**: external Python Gymnasium (started as a subprocess).
- **Trust boundary**: every step is checked against a Lean side contract
  (`Runtime.RL.Boundary.Contract`) before being used for training data.
- **Algorithm**: PPO with GAE (all update math is Lean definitions; the PPO loss is a TorchLean
  autograd program).

More concretely:

- The policy is a categorical distribution over discrete actions parameterized by logits:
  $\pi_\theta(a\mid s)=\operatorname{softmax}(\mathrm{logits}_\theta(s))$.
- Advantages are computed using Generalized Advantage Estimation (GAE(λ)).
- PPO uses the clipped surrogate objective (plus a value-loss and optional entropy bonus, depending
  on the runtime configuration).

## CLI flags

- `--device cuda`: run the Torch backend on CUDA (requires building with `-K cuda=true`).
- `--seed <n>`: deterministic seed for TorchLean RNG streams (and evaluation seeding).
- `--updates <n>`: limit the number of PPO rollout/update cycles.
- `--log <path>`: write the widget log JSON to a custom path.

Run (from the repo root):

```bash
python3 -m pip install --user 'gymnasium>=1.0'
lake -R -K cuda=true exe torchlean ppo_cartpole --device cuda --updates 1 --eval-every 1 \
  --eval-episodes 1 --eval-max-steps 8
```

Artifacts:
- The executable writes a widget-friendly training curve JSON to
  `data/rl/ppo_cartpole_trainlog.json` (override with `--log <path>`).
- Visualize it in the editor via `NN/Examples/Models/RL/Views/PPOCartPole.lean`.

## What this run does (and does not) guarantee

- The PPO/GAE math and the autograd loss program are Lean definitions, so they are suitable targets
  for formal reasoning.
- When Gymnasium is external, TorchLean cannot prove the environment satisfies Markov/measurability
  assumptions. The trust-boundary contract turns some common assumptions (finite tensors, reward
  bounds, done-flag semantics) into checked preconditions.
- The run favors readability, typed boundaries, and widget inspection over benchmark-specific PPO
  tuning.

References (primary):
- Schulman et al., "Proximal Policy Optimization Algorithms" (2017):
  https://arxiv.org/abs/1707.06347
- Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation"
  (2015): https://arxiv.org/abs/1506.02438
- Williams, "Simple statistical gradient-following algorithms for connectionist reinforcement
  learning" (REINFORCE, 1992): https://doi.org/10.1007/BF00992696
- Brockman et al., "OpenAI Gym" (2016): https://arxiv.org/abs/1606.01540
- Gymnasium API reference (reset/step, `terminated` vs `truncated`): https://gymnasium.farama.org/
- CartPole environment docs: https://gymnasium.farama.org/environments/classic_control/cart_pole/
-/

@[expose] public section

open TorchLean

namespace NN.Examples.Models.RL.PPOCartPole

/-- Name of this executable target (used in CLI error messages and banners). -/
def exeName : String := "ppo_cartpole"

/-!
## Configuration

This stays with discrete-action CartPole so the native Lean executable remains easy to run and
inspect.
-/

/--
Gymnasium environment id passed to the Python subprocess (see Gymnasium docs for supported ids).
-/
def envId : String := "CartPole-v1"

/-- Relative path to the Python Gymnasium bridge script (spawned as a subprocess). -/
def gymServerScript : String := "scripts/rl/gymnasium_server.py"

/-- Observation vector dimension for CartPole (`Gymnasium` reports 4 floats). -/
def observationWidth : Nat := 4

/-- Number of discrete actions for CartPole (left/right). -/
def actionCount : Nat := 2

/-- Width of the hidden layer in the actor and critic MLPs. -/
def hiddenWidth : Nat := 32

/-- PPO rollout horizon (also the training batch size for this run). -/
def horizon : Nat := 64

/-- Discount factor used in returns / GAE. -/
def discountFactor : Float := 0.99

/-- GAE(λ) parameter controlling the bias/variance tradeoff of advantage estimates. -/
def gaeLambda : Float := 0.95

/-- Adam learning rate used for the CartPole actor-critic update. -/
def learningRate : Float := 3e-4

/-- Number of PPO optimization epochs per collected rollout batch. -/
def updateEpochs : Nat := 2

/-- Maximum number of PPO updates (training stops early if the "solved" criterion triggers). -/
def maxUpdates : Nat := 1000

/-- Evaluate the greedy policy after this many PPO updates. -/
def defaultEvaluationInterval : Nat := 50

/-- Number of evaluation episodes per checkpoint. -/
def defaultEvaluationEpisodes : Nat := 5

/-- Stop early if average return meets/exceeds this threshold. -/
def solvedAverageReturn : Float := 475.0

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
/--
A single value estimate. Kept as `[1]` rather than a scalar so it composes with the batched shapes
above without a reshape.
-/
def value : Shape := [1]

/-!
## Model (Actor + Critic)

We use the public `TorchLean.nn` surface, which provides prefix-shape preserving layers:
if `x` has shape `[..., inputWidth]`, `nn.linear inputWidth outputWidth` maps it to
`[..., outputWidth]`.
-/

abbrev modelConfig : nn.models.PPO.Config :=
  { observationWidth := observationWidth
    hiddenWidth := hiddenWidth
    actionCount := actionCount }

/-- Construct the actor network as an MLP mapping observations to action logits. -/
def actor (batchShape : Shape := []) :
    nn.Builder (nn.Sequential (modelConfig.input batchShape)
      (modelConfig.actorOutput batchShape)) :=
  nn.models.PPO.actor modelConfig batchShape

/-- Construct the critic network as an MLP mapping observations to a scalar value estimate. -/
def critic (batchShape : Shape := []) :
    nn.Builder (nn.Sequential (modelConfig.input batchShape)
      (modelConfig.criticOutput batchShape)) :=
  nn.models.PPO.critic modelConfig batchShape

/-!
## Gymnasium Bridge

We talk to a small Python service (`scripts/rl/gymnasium_server.py`) using the reusable runtime
bridge exposed as `rl.gym.*`.

The Lean side trust-boundary contract (`rl.boundary.Contract`) is enforced on every step.
-/

/-!
## Evaluation

Evaluation APIs live in `rl.eval`.
-/

/-!
## Main Training Loop
-/

/-- Entry point for `lake exe torchlean ppo_cartpole`.

This executable:
- launches a Python Gymnasium subprocess for `CartPole-v1`,
- collects checked rollouts under `rl.boundary.Contract`,
- performs PPO updates on the selected Torch backend device,
- writes a widget-friendly training curve JSON (default: `data/rl/ppo_cartpole_trainlog.json`).
-/
def main (args : List String) : IO UInt32 := do
  Module.Command.run
    (config := {
      banner? := some <| Support.bannerWithDeviceDetails
        exeName
        s!"PPO on {envId} (horizon={horizon})"
        "  env: Python Gymnasium subprocess (JSON-lines bridge) + Lean boundary contract"
      usage? := some <| rl.cli.PPOOptions.usage exeName
      printSuccess := true })
    exeName args
    (.native fun runtime rest => do
      let (ppo, rest) ← CLI.orThrow exeName <|
        rl.cli.PPOOptions.parse
          exeName rest Runtime.RL.Artifacts.DefaultPaths.ppoCartPoleTrainLog
          (defaultUpdateCount := maxUpdates)
          (defaultEvaluationInterval := defaultEvaluationInterval)
          (defaultEvaluationEpisodes := defaultEvaluationEpisodes)
          (defaultMaximumEvaluationSteps := 500)
      CLI.orThrow exeName <| CLI.checkNoArgs rest
      let updateCount : Nat := ppo.updateCount
      let evaluationInterval : Nat := ppo.evaluationInterval
      let evaluationEpisodes : Nat := ppo.evaluationEpisodes
      let maximumEvaluationSteps : Nat := ppo.maximumEvaluationSteps
      let contract : rl.boundary.Contract observation actionCount :=
        { checkObsFinite := true
          checkRewardFinite := true
          obsRange? := none
          rewardRange? := none
          requireExclusiveDoneFlags := false }

      let gym ←
        rl.gym.client.spawn
          (obsShape := observation) (nActions := actionCount) gymServerScript envId contract
      try
        -- Build actor and critic once per seed, then reuse those parameters for one observation
        -- and for the horizon-sized rollout batch.
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


        -- Training curve: greedy-policy evaluation return before training, then at each
        -- evaluation checkpoint. We keep this as a compact `Curve` (arrays) because it is
        -- destined for JSON/widget display; the actual learning data is stored as typed tensors.
        let mut curve : Training.Curve := {}

        let evaluationSessionAt :
            Nat → rl.session.CheckedSession observation actionCount :=
          fun seed =>
            rl.session.gymnasium
              (obsShape := observation) (nActions := actionCount) gym
              (seed? := some seed) (resetOnDone := false)

        -- Evaluate the untrained policy once (step=0).
        do
          let psAll0 ← rl.ppo.state (α := Float) m
          let policyLogits0 :
              Tensor Float observation → Tensor Float actionLogits :=
            rl.ppo.actorPolicy actorGraph actorRollout criticRollout psAll0
          let avg0 ←
            rl.eval.averageEpisodeTotalReward
              (obsShape := observation) (nActions := actionCount)
              evaluationSessionAt policyLogits0 (baseSeed := 1000)
              (episodes := evaluationEpisodes)
              (maxSteps := maximumEvaluationSteps)
          curve := curve.push 0 avg0
          IO.println s!"  eval(step=0) avg_return={avg0}"

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
                  (horizon := horizon) (castObservation := id) (castReward := id)
                  gym predictLogits predictValue
                  (rngSeed := rngSeed) (rngCounter := rngCounter) (resetSeed := update)
              pure (rollout, rngCounter'))
          stepSample
          (fun completedUpdates => do
              let psAll' ← rl.ppo.state (α := Float) m
              let policyLogits :
                  Tensor Float observation → Tensor Float actionLogits :=
                rl.ppo.actorPolicy actorGraph actorRollout criticRollout psAll'
              let avg ←
                rl.eval.averageEpisodeTotalReward
                  (obsShape := observation) (nActions := actionCount)
                  evaluationSessionAt policyLogits (baseSeed := 1000 + completedUpdates)
                    (episodes := evaluationEpisodes)
                  (maxSteps := maximumEvaluationSteps)
              IO.println s!"  update={completedUpdates} avg_return={avg}"
              if avg ≥ solvedAverageReturn then
                IO.println s!"{exeName}: solved (avg_return ≥ {solvedAverageReturn})"
                return (avg, true)

              pure (avg, false))
          curve

        Training.Curve.writeLog curve
          ppo.logDestination
          s!"PPO {envId} (TorchLean)"
          "avg_return"
          (color := "#4e79a7")
          #[
            s!"env_id={envId}",
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
        IO.println s!"{exeName}: done"
      finally
        gym.close
    )

end NN.Examples.Models.RL.PPOCartPole
