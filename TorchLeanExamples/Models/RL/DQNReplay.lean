/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.RL.PPO.ActorCritic
public import TorchLean.NN.Data.Source
public import TorchLean.NN.Builder
public import TorchLean.NN.Training.Reporting

/-!
# DQN Replay Mini-Example

The runtime pieces of an off-policy DQN-style update, run once on fixed data:

1. Construct typed transitions.
2. Insert them into a bounded replay buffer.
3. Sample a minibatch.
4. Evaluate the DQN minibatch loss, with squared and with Huber error, from caller-provided online
   and target Q-functions.
5. Apply one soft target-network update.

The Q-functions are hand-written closures, which keeps the example on replay buffers and minibatch
losses.

Run from the repository root:

```bash
scripts/lake.sh exe torchlean dqn_replay
```

References:
- Mnih et al., "Human-level control through deep reinforcement learning" (2015):
  https://doi.org/10.1038/nature14236
- Lin, "Self-Improving Reactive Agents Based on Reinforcement Learning, Planning and Teaching"
  (1992), early experience replay.
-/

@[expose] public section

open TorchLean.RL

namespace TorchLean.Examples.DQNReplay

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
def transitionA : RL.Transition Float observation actionCount :=
  { state := obsA
    action := 1
    reward := 1.0
    nextState := obsB
    done := false }

/-- A second transition, marked terminal, so the sample contains both bootstrap modes. -/
def transitionB : RL.Transition Float observation actionCount :=
  { state := obsB
    action := 0
    reward := 0.5
    nextState := obsA
    done := true }

/-- Hand-written online Q-function, affine in the two observation features. -/
def onlineQ (obs : Tensor Float observation) : Tensor Float [actionCount] :=
  [obs[0] + 0.2, obs[1] + 1.0, 0.5]

/-- Hand-written target Q-function, also affine in the observation. -/
def targetQ (obs : Tensor Float observation) : Tensor Float [actionCount] :=
  [0.1 + obs[1], 1.4 + obs[0], 0.3]

/--
Fill a replay buffer of capacity 8 with the two transitions, sample a contiguous minibatch of four
(the sample wraps around), and print its DQN losses for `γ = 0.9` and one soft target update with
`τ = 0.1`.
-/
def run : IO Unit := do
  IO.println "dqn_replay: begin"

  let buffer := Replay.Buffer.pushMany (Replay.Buffer.empty 8) #[transitionA, transitionB]
  IO.println s!"stored transitions: {Replay.Buffer.size buffer}"

  let batch := Replay.Buffer.sampleContiguous buffer (start := 0) (batchSize := 4)
  IO.println s!"sampled transitions: {batch.size}"

  let gamma : Float := 0.9
  let mse :=
    DQN.loss onlineQ targetQ gamma batch (batch := true)
  let huber :=
    DQN.loss onlineQ targetQ gamma batch (batch := true)
      (error := fun prediction target => Huber.term prediction target 1.0)
  IO.println s!"DQN minibatch MSE loss:   {mse}"
  IO.println s!"DQN minibatch Huber loss: {huber}"

  let targetParam := 0.0
  let onlineParam := 10.0
  let synced := DQN.updateTarget (α := Float) 0.1 onlineParam targetParam
  IO.println s!"soft target update example: {synced}"
  IO.println "dqn_replay: ok"

/-- Command-line help for the replay-buffer mini-example. -/
def usage : String :=
  String.intercalate "\n"
    [ "Usage:"
    , "  scripts/lake.sh exe torchlean dqn_replay"
    , ""
    , "Stores two fixed transitions in a replay buffer, samples a minibatch, and prints the sizes,"
    , "the minibatch DQN losses and one soft target update. This command has no training flags."
    ]

/-- Runner entrypoint used by `scripts/lake.sh exe torchlean dqn_replay`. -/
def main (args : List String) : IO UInt32 := do
  if CLI.hasHelp args then
    IO.println usage
  else
    CLI.requireNoArgs exeName args
    run
  pure 0

end TorchLean.Examples.DQNReplay
