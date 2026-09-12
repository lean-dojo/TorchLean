/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.CLI.Trainer

/-!
# Training Quickstart Flag Parsing

Flag parsing for `SimpleMlpTrain`, kept separate from the model and training example.

This is example support, not part of the training API. User code starts from `Trainer.new` and
`trainer.train`; this file only keeps flag handling out of the tutorial bodies.
-/

@[expose] public section

namespace NN.Examples.Quickstart

open TorchLean

/-- Command-line choices accepted by a training quickstart. -/
structure Flags where
  /-- Model initialization seed from `--seed`. -/
  seed : Nat
  /-- Number of optimizer updates from `--steps`. -/
  steps : Nat
  /-- Runtime settings from `--arithmetic`, `--execution`, `--device`, and `--show-backend`. -/
  runtime : Trainer.RunConfig

/-- Parse `--seed`, `--steps`, and the runtime flags, rejecting anything else. -/
def parseFlags (exeName : String) (args : List String) (defaultSteps : Nat) : IO Flags := do
  let (seed, args) ← CLI.seed exeName args
  let (steps, args) ← CLI.positiveNatFlag exeName args "steps" defaultSteps
  let runtime ← CLI.Trainer.parseCommandLine exeName args
  pure { seed, steps, runtime }

end NN.Examples.Quickstart
