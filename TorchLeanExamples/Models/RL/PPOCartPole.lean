/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Models.PPO
public import TorchLean.NN.RL.CLI
public import TorchLean.NN.RL.PPO.ActorCritic
public import TorchLeanExamples.Models.Training
public import TorchLeanExamples.Models.RL.PPOAgent
public import TorchLean.NN.Data.Source

/-!
# PPO on Gymnasium CartPole (Executable Example)

A small but complete PPO run:

- **Environment**: external Python Gymnasium, started as a subprocess.
- **Trust boundary**: every step is checked against a Lean-side contract
  (`TorchLean.RL.Boundary.Contract`) before it becomes training data.
- **Algorithm**: PPO with GAE. The update math is Lean definitions, and the PPO loss is a TorchLean
  autograd program.

More concretely:

- The policy is a categorical distribution over discrete actions parameterized by logits:
  $\pi_\theta(a\mid s)=\operatorname{softmax}(\mathrm{logits}_\theta(s))$.
- Advantages come from Generalized Advantage Estimation (GAE(λ)).
- PPO uses the clipped surrogate objective (plus a value-loss and optional entropy bonus, depending
  on the runtime configuration).

## CLI flags

- `--device cuda`: run the Torch backend on CUDA (requires building with `-K cuda=true`).
- `--seed <n>`: deterministic seed for TorchLean RNG streams. Gymnasium resets do not depend on
  it: the rollout of update `u` (counted from `0`) resets with seed `u`, and evaluation episode `k`
  after `n` completed updates resets with seed `1000 + n + k`.
- `--updates <n>`: limit the number of PPO rollout/update cycles (default 1000).
- `--eval-every <n>`, `--eval-episodes <n>`, `--eval-max-steps <n>`: evaluation cadence, episode
  count and episode length (defaults 50, 5 and 500).
- `--log <path>`: write the widget log JSON to a custom path.

Run (from the repo root):

```bash
python3 -m pip install --user 'gymnasium>=1.0'
scripts/lake.sh -Kcuda=true exe torchlean ppo_cartpole --device cuda \
    --updates 1 --eval-every 1 \
  --eval-episodes 1 --eval-max-steps 8
```

Artifacts:
- The executable writes a widget-friendly training curve JSON to
  `data/rl/ppo_cartpole_trainlog.json` (override with `--log <path>`).
- `#train_log_file_view` displays it, and this directory's README has artifact examples.

## What this run does (and does not) guarantee

- The PPO/GAE math and the autograd loss program are Lean definitions, so they are suitable targets
  for formal reasoning.
- When Gymnasium is external, TorchLean cannot prove the environment satisfies Markov/measurability
  assumptions. The boundary contract of this command checks that observations and rewards are
  finite. It leaves their ranges unrestricted and permits both done flags to be true.
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

open TorchLean.RL

namespace TorchLean.Examples.PPOCartPole

/-- Name of this executable target (used in CLI error messages and banners). -/
def exeName : String := "ppo_cartpole"

/-!
## Configuration

Discrete-action CartPole keeps the native Lean executable easy to run and inspect.
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

/--
PPO rollout horizon (also the training batch size for this run). Every update restarts the
environment, so training never sees an episode past step `horizon`, while evaluation episodes run
up to `--eval-max-steps` steps (default 500).
-/
def horizon : Nat := 64

/-- Discount factor used in returns / GAE. -/
def discountFactor : Float := 0.99

/-- GAE(λ) parameter controlling the bias/variance tradeoff of advantage estimates. -/
def gaeLambda : Float := 0.95

/-- Adam learning rate used for the CartPole actor-critic update. -/
def lr : Float := 3e-4

/-- Number of PPO optimization epochs per collected rollout batch. -/
def updateEpochs : Nat := 2

/--
Default cap on PPO updates (`--updates`). Training stops early once the solved criterion is met.
-/
def maxUpdates : Nat := 1000

/-- Default number of PPO updates between greedy-policy evaluations (`--eval-every`). -/
def defaultEvaluationInterval : Nat := 50

/-- Default number of evaluation episodes per checkpoint (`--eval-episodes`). -/
def defaultEvaluationEpisodes : Nat := 5

/--
Training stops once the average greedy evaluation return reaches this value, the CartPole-v1
solved threshold.
-/
def solvedAverageReturn : Float := 475.0

instance : NeZero horizon := ⟨by decide⟩
instance : NeZero actionCount := ⟨by decide⟩

/-- Shape of one CartPole observation, `[observationWidth]`. -/
def observation : Shape := [observationWidth]

/-- Action logits at a single timestep. -/
def actionLogits : Shape := [actionCount]

/-!
## Model (Actor + Critic)

The actor and critic are MLPs from the public `TorchLean.nn` surface, whose layers preserve a shape
prefix: if `x` has shape `[..., inputWidth]`, `nn.linear inputWidth outputWidth` maps it to
`[..., outputWidth]`. The same builder therefore serves one observation and a whole rollout.
-/

/-- Actor and critic sizes: observations of width `observationWidth`, hidden layers of width
`hiddenWidth` and `actionCount` actor logits. -/
abbrev modelConfig : nn.PPO.Config :=
  { observationWidth := observationWidth
    hiddenWidth := hiddenWidth
    actionCount := actionCount }

/-!
## Gymnasium Bridge

The executable talks to a small Python service (`scripts/rl/gymnasium_server.py`) through the
reusable runtime bridge `RL.Gymnasium.*`. The Lean-side trust-boundary contract
(`Boundary.Contract`)
checks every step.
-/

/-!
## Main Training Loop
-/

/-- Entry point for `scripts/lake.sh exe torchlean ppo_cartpole`.

The command:
- launches a Python Gymnasium subprocess for `CartPole-v1`,
- collects checked rollouts under `Boundary.Contract`,
- performs PPO updates on the selected Torch backend device,
- writes a widget-friendly training curve JSON (default: `data/rl/ppo_cartpole_trainlog.json`).
-/
def main (args : List String) : IO UInt32 := do
  Module.Command.run
    (config := {
      banner? := some <| Training.banner
        exeName
        s!"PPO on {envId} (horizon={horizon})"
        (details := some
          "  env: Python Gymnasium subprocess (JSON-lines bridge) + Lean boundary contract")
      usage? := some <| CLI.Options.usage exeName
      printSuccess := true })
    exeName args
    (fun runtime rest => do
      let (ppo, rest) ← CLI.orThrow exeName <|
        CLI.Options.parse
          exeName rest (Training.artifactPath "ppo_cartpole" (dir := "rl"))
          (defaultUpdateCount := maxUpdates)
          (defaultEvaluationInterval := defaultEvaluationInterval)
          (defaultEvaluationEpisodes := defaultEvaluationEpisodes)
          (defaultMaximumEvaluationSteps := 500)
      CLI.orThrow exeName <| CLI.checkNoArgs rest
      let contract : Boundary.Contract observation actionCount :=
        { checkObsFinite := true
          checkRewardFinite := true
          obsRange? := none
          rewardRange? := none
          requireExclusiveDoneFlags := false }

      let gym ←
        Gymnasium.Client.spawn
          (obsShape := observation) (nActions := actionCount) gymServerScript envId contract
      try
        let agent : PPOAgent observation actionCount horizon ←
          PPOAgent.create modelConfig horizon runtime (optim.adam { lr := lr })

        -- Training curve: the greedy-policy evaluation return before training and at each
        -- evaluation checkpoint. It is a compact `Curve` of arrays because it only feeds the JSON
        -- log and the widget, while the learning data lives in typed tensors.
        let mut curve : Training.Curve := {}

        let evaluationSessionAt :
            Nat → Session.Checked observation actionCount :=
          fun seed =>
            Session.Checked.gymnasium
              (obsShape := observation) (nActions := actionCount) gym
              (seed? := some seed) (resetOnDone := false)

        -- Evaluate the untrained policy once (step=0).
        do
          let psAll0 ← agent.state
          let policyLogits0 :
              Tensor Float observation → Tensor Float actionLogits :=
            agent.policy psAll0
          let avg0 ←
            Eval.meanReward
              (obsShape := observation) (nActions := actionCount)
              evaluationSessionAt policyLogits0 (baseSeed := 1000)
              (episodes := ppo.evaluationEpisodes)
              (maxSteps := ppo.maximumEvaluationSteps)
          curve := curve.push 0 avg0
          IO.println s!"  eval(step=0) avg_return={avg0}"

        curve ← PPO.train discountFactor gaeLambda
          { updates := ppo.updateCount, epochs := updateEpochs,
            evaluationEvery := ppo.evaluationInterval, seed := runtime.seed }
          (fun update rngSeed rngCounter => do
              let psAll ← agent.state
              let predictLogits :
                  Tensor Float observation → Tensor Float actionLogits :=
                agent.policy psAll
              let predictValue : Tensor Float observation → Float :=
                agent.value psAll
              let (rollout, rngCounter') ←
                PPO.collect
                  (α := Float) (obsShape := observation) (nActions := actionCount)
                  (horizon := horizon)
                  (Session.Checked.gymnasium (obsShape := observation) (nActions := actionCount) gym
                    (seed? := some update) (resetOnDone := true))
                  (castObservation := id) (castReward := id) predictLogits predictValue
                  (rngSeed := rngSeed) (rngCounter := rngCounter)
              pure (rollout, rngCounter'))
          agent.updateBatch
          (fun completedUpdates => do
              let psAll' ← agent.state
              let policyLogits :
                  Tensor Float observation → Tensor Float actionLogits :=
                agent.policy psAll'
              let avg ←
                Eval.meanReward
                  (obsShape := observation) (nActions := actionCount)
                  evaluationSessionAt policyLogits (baseSeed := 1000 + completedUpdates)
                    (episodes := ppo.evaluationEpisodes)
                  (maxSteps := ppo.maximumEvaluationSteps)
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
            s!"lr={lr}",
            s!"updates={ppo.updateCount}",
            s!"eval_every={ppo.evaluationInterval}",
            s!"eval_episodes={ppo.evaluationEpisodes}",
            s!"eval_max_steps={ppo.maximumEvaluationSteps}",
            Training.deviceNote runtime
          ]
        IO.println s!"{exeName}: done"
      finally
        gym.close
    )

end TorchLean.Examples.PPOCartPole
