/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN
public import NN.API.RL

/-!
# DQN Replay Mini-Example

This example runs the runtime pieces used by an off-policy DQN-style update:

1. construct typed transitions;
2. insert them into a bounded replay buffer;
3. sample a minibatch;
4. evaluate a DQN minibatch loss from caller-provided online/target Q-functions.

It is deliberately compact: the Q-functions are hand-written closures rather than neural networks. That
keeps the file focused on the replay/minibatch API. A full trainable DQN example can later swap those
closures for compiled TorchLean models and an optimizer step.

Run from the repo root through the maintained example runner:

```bash
lake exe torchlean dqn_replay
```

References:
- Mnih et al., "Human-level control through deep reinforcement learning" (2015):
  https://doi.org/10.1038/nature14236
- Lin, "Self-Improving Reactive Agents Based on Reinforcement Learning, Planning and Teaching"
  (1992), early experience replay.
-/

@[expose] public section

open Spec
open Tensor

namespace NN.Examples.Models.RL.DQNReplay

abbrev ObsShape : Shape := .dim 2 .scalar
abbrev NActions : Nat := 3

/-- A compact two-feature observation. -/
def obsA : Tensor Float ObsShape := Spec.fromList1d [0.0, 1.0]

/-- A second observation used as the next state. -/
def obsB : Tensor Float ObsShape := Spec.fromList1d [1.0, 0.0]

/-- One typed transition inserted into the replay buffer. -/
def transitionA : Runtime.RL.Core.Transition Float ObsShape NActions :=
  { state := obsA
    action := ⟨1, by decide⟩
    reward := 1.0
    nextState := obsB
    done := false }

/-- A second transition, marked terminal, so the sample contains both bootstrap modes. -/
def transitionB : Runtime.RL.Core.Transition Float ObsShape NActions :=
  { state := obsB
    action := ⟨0, by decide⟩
    reward := 0.5
    nextState := obsA
    done := true }

/-- Compact online Q-function used by the example. -/
def onlineQ (obs : Tensor Float ObsShape) : Tensor Float (.dim NActions .scalar) :=
  let x0 := Tensor.vecGet obs ⟨0, by decide⟩
  let x1 := Tensor.vecGet obs ⟨1, by decide⟩
  Spec.fromList1d [x0 + 0.2, x1 + 1.0, 0.5]

/-- Compact target Q-function used by the example. -/
def targetQ (obs : Tensor Float ObsShape) : Tensor Float (.dim NActions .scalar) :=
  let x0 := Tensor.vecGet obs ⟨0, by decide⟩
  let x1 := Tensor.vecGet obs ⟨1, by decide⟩
  Spec.fromList1d [0.1 + x1, 1.4 + x0, 0.3]

/-- Build a replay buffer, sample a minibatch, and compute DQN losses. -/
def run : IO Unit := do
  IO.println "dqn_replay: begin"

  let buffer0 : Runtime.RL.Replay.Buffer Float ObsShape NActions :=
    Runtime.RL.Replay.Buffer.empty 8
  let buffer :=
    buffer0.pushMany #[transitionA, transitionB]
  IO.println s!"stored transitions: {buffer.size}"

  let batch := buffer.sampleContiguous (start := 0) (batchSize := 4)
  IO.println s!"sampled transitions: {batch.size}"

  let gamma : Float := 0.9
  let mse :=
    Runtime.RL.DQN.minibatchMSELoss (α := Float) onlineQ targetQ gamma batch
  let huber :=
    Runtime.RL.DQN.minibatchHuberLoss (α := Float) onlineQ targetQ gamma 1.0 batch
  IO.println s!"DQN minibatch MSE loss:   {mse}"
  IO.println s!"DQN minibatch Huber loss: {huber}"

  let targetParam := 0.0
  let onlineParam := 10.0
  let synced := Runtime.RL.DQN.softUpdateScalar (α := Float) 0.1 onlineParam targetParam
  IO.println s!"soft target update example: {synced}"
  IO.println "dqn_replay: ok"

/-- Runner entrypoint used by `lake exe torchlean dqn_replay`. -/
def main (_args : List String) : IO UInt32 := do
  run
  pure 0

end NN.Examples.Models.RL.DQNReplay
