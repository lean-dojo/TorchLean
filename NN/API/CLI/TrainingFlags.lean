/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.CLI.Parser

/-!
# Training Flag Presets

Small parser compositions for the training flags shared by runnable examples.
-/

@[expose] public section

namespace TorchLean.CLI

/-- Common training flags for epoch-oriented loader and tutorial commands. -/
structure EpochBatch where
  /-- Number of epochs to train for. -/
  epochs : Nat
  /-- Batch size. -/
  batchSize : Nat

/-- Parse `--epochs` and `--batch`, requiring both selected values to be positive. -/
def takePositiveEpochBatch
    (arguments : List String)
    (exeName : String)
    (defaultEpochs defaultBatch : Nat) :
    Except String (EpochBatch × List String) := do
  let (epochs, arguments) ← takeNatFlag arguments "epochs" (default := defaultEpochs)
  let (batchSize, arguments) ← takeNatFlag arguments "batch" (default := defaultBatch)
  if epochs = 0 then
    throw s!"{exeName}: --epochs must be > 0"
  if batchSize = 0 then
    throw s!"{exeName}: --batch must be > 0"
  pure ({ epochs, batchSize }, arguments)

end TorchLean.CLI
