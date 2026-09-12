/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API

/-!
# DQN Replay Mini-Example

This example runs the runtime pieces used by an off-policy DQN-style update:

1. construct typed transitions;
2. insert them into a bounded replay buffer;
3. sample a minibatch;
4. evaluate a DQN minibatch loss from caller-provided online/target Q-functions.

The Q-functions are hand-written closures rather than neural networks, so the example stays focused
on replay buffers and minibatch losses. A full trainable DQN run can later swap those closures for
typed graph TorchLean models and an optimizer step.

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

open TorchLean

namespace NN.Examples.Models.RL.DqnReplay

/-- Subcommand name. -/
def exeName : String := "dqn_replay"

/--
Two observation features, small enough that the replay buffer's contents can be printed in full.
-/
abbrev observation : Shape := [2]
/-- Three discrete actions. -/
abbrev actionCount : Nat := 3

/-- A compact two-feature observation. -/
def obsA : Tensor Float observation := [0.0, 1.0]

/-- A second observation used as the next state. -/
def obsB : Tensor Float observation := [1.0, 0.0]

/-- One typed transition inserted into the replay buffer. -/
def transitionA : rl.core.Transition Float observation actionCount :=
  { state := obsA
    action := 1
    reward := 1.0
    nextState := obsB
    done := false }

/-- A second transition, marked terminal, so the sample contains both bootstrap modes. -/
def transitionB : rl.core.Transition Float observation actionCount :=
  { state := obsB
    action := 0
    reward := 0.5
    nextState := obsA
    done := true }

/-- Compact online Q-function used by the example. -/
def onlineQ (obs : Tensor Float observation) : Tensor Float [actionCount] :=
  let x0 := obs[0]
  let x1 := obs[1]
  [x0 + 0.2, x1 + 1.0, 0.5]

/-- Compact target Q-function used by the example. -/
def targetQ (obs : Tensor Float observation) : Tensor Float [actionCount] :=
  let x0 := obs[0]
  let x1 := obs[1]
  [0.1 + x1, 1.4 + x0, 0.3]

/-- Build a replay buffer, sample a minibatch, and compute DQN losses. -/
def run : IO Unit := do
  IO.println "dqn_replay: begin"

  let buffer0 : rl.replay.Buffer Float observation actionCount :=
    rl.replay.empty 8
  let buffer :=
    rl.replay.pushMany buffer0 #[transitionA, transitionB]
  IO.println s!"stored transitions: {rl.replay.size buffer}"

  let batch := rl.replay.sampleContiguous buffer (start := 0) (batchSize := 4)
  IO.println s!"sampled transitions: {batch.size}"

  let gamma : Float := 0.9
  let mse :=
    rl.dqn.minibatchMSELoss (α := Float) onlineQ targetQ gamma batch
  let huber :=
    rl.dqn.minibatchHuberLoss (α := Float) onlineQ targetQ gamma 1.0 batch
  IO.println s!"DQN minibatch MSE loss:   {mse}"
  IO.println s!"DQN minibatch Huber loss: {huber}"

  let targetParam := 0.0
  let onlineParam := 10.0
  let synced := rl.dqn.softUpdateScalar (α := Float) 0.1 onlineParam targetParam
  IO.println s!"soft target update example: {synced}"
  IO.println "dqn_replay: ok"

/-- Command-line help for the replay-buffer mini-example. -/
def usage : String :=
  String.intercalate "\n"
    [ "Usage:"
    , "  lake exe torchlean dqn_replay"
    , ""
    , "Runs a fixed replay-buffer and DQN-loss executable check. "
        ++ "This command has no training flags."
    ]

/-- Runner entrypoint used by `lake exe torchlean dqn_replay`. -/
def main (args : List String) : IO UInt32 := do
  if CLI.hasHelp args then
    IO.println usage
  else
    CLI.requireNoArgs exeName args
    run
  pure 0

end NN.Examples.Models.RL.DqnReplay
