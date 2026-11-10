/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.RL.PPO.Rollout
public import TorchLean.NN.Training
public import TorchLean.Tensor.Random

/-!
# PPO Training

The schedule shared by fixed-horizon PPO runs: collect a rollout, compute GAE, update for several
epochs, and evaluate at a fixed cadence. Callbacks run the models and the environment, and the loop
owns the seed and sampler-counter progression and the update counts.

We wrote `train` by hand as the outer loop of Algorithm 1 in Schulman et al. (2017), "Proximal
Policy Optimization Algorithms", arXiv:1707.06347. Each update collects one rollout, so their $N$
is 1, and each of the $K$ = `config.epochs` passes hands `updateBatch` the whole batch. With
`ActorCritic.bind` as `updateBatch`, their minibatch size $M$ is the horizon $T$.
-/

@[expose] public section

namespace TorchLean.RL.PPO

/-- PPO update counts, evaluation cadence, and the initial rollout seed. -/
structure TrainConfig where
  /-- Number of PPO updates, each on a freshly collected rollout. -/
  updates : Nat
  /-- Number of optimization passes over the batch of each rollout. -/
  epochs : Nat
  /-- Number of updates between evaluations, where `0` disables evaluation. -/
  evaluationEvery : Nat
  /-- Seed of the first rollout. -/
  seed : Nat

/-- Run `config.updates` PPO updates and append evaluation points to `initialCurve`. Each update
collects a rollout, builds its `TrainingBatch` and passes it to `updateBatch` `config.epochs` times.

`collect` receives the zero-based update index, seed, and random counter. `evaluate` receives the
completed update count and returns a metric plus an early-stop flag. A stop occurs after recording
that metric and before advancing the seed. `evaluationEvery = 0` disables evaluation, and with it
early stopping.
-/
def train {α : Type} [Storage α] [Context α] {observation : Shape} {actions horizon : Nat}
    [NeZero actions] [NeZero horizon] (gamma lam : α) (config : TrainConfig)
    (collect : Nat → Nat → Nat → IO (Rollout α observation actions horizon × Nat))
    (updateBatch : TrainingBatch α observation actions horizon → IO Unit)
    (evaluate : Nat → IO (Float × Bool))
    (initialCurve : Training.Curve := {}) : IO Training.Curve := do
  let mut seed := config.seed
  let mut counter := 0
  let mut curve := initialCurve
  for update in [0:config.updates] do
    let (rollout, nextCounter) ← collect update seed counter
    counter := nextCounter
    let batch ← rollout.trainingBatch gamma lam
    for _ in [0:config.epochs] do
      updateBatch batch
    let completed := update + 1
    if config.evaluationEvery != 0 && completed % config.evaluationEvery == 0 then
      let (metric, stop) ← evaluate completed
      curve := curve.push completed metric
      if stop then break
    seed := Random.nextSeed seed completed
  pure curve

end TorchLean.RL.PPO
