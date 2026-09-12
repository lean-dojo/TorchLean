/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.RL.PPO.Rollout
public import NN.Runtime.Training.Log
public import NN.Spec.Core.Random

/-!
# PPO Training

Shared rollout, GAE, repeated-update, and evaluation schedule for fixed-horizon PPO. Callbacks own
model execution and environment interaction; the loop owns RNG progression and update counts.
-/

@[expose] public section

namespace Runtime.RL.PPO

open Spec TorchLean

/-- PPO update counts, evaluation cadence, and the initial rollout seed. -/
structure TrainConfig where
  updates : Nat
  epochs : Nat
  evaluationEvery : Nat
  seed : Nat

/-- Train from fixed-horizon rollouts and append evaluation points to an existing curve.

`collect` receives the zero-based update index, seed, and random counter. `evaluate` receives the
completed update count and returns a metric plus an early-stop flag. A stop occurs after recording
that metric and before advancing the seed. `evaluationEvery = 0` disables periodic evaluation.
-/
def train {α : Type} [Storage α] [Context α] {observation : Shape} {actions horizon : Nat}
    [NeZero actions] [NeZero horizon] (gamma lam : α) (config : TrainConfig)
    (collect : Nat → Nat → Nat → IO (Rollout α observation actions horizon × Nat))
    (updateBatch : TrainingBatch α observation actions horizon → IO Unit)
    (evaluate : Nat → IO (Float × Bool))
    (initialCurve : Runtime.Training.Curve := {}) : IO Runtime.Training.Curve := do
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
    seed := Spec.Random.nextSeed seed completed
  pure curve

end Runtime.RL.PPO
