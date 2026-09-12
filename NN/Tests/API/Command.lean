/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Module.Command

/-!
# Module Command API Tests

Regression checks for runtime defaults and executable command requirements.
-/

@[expose] public section

namespace NN.Tests.API.Command

open TorchLean

def expectEqual {α : Type} [BEq α] [Repr α]
    (label : String) (expected actual : α) : IO Unit := do
  unless expected == actual do
    throw <| IO.userError
      s!"command API check failed: {label} (expected {repr expected}, got {repr actual})"

def parseSelection (args : List String) :
    IO (Module.RuntimeSelection × List String) :=
  match Module.RuntimeSelection.parse args .native with
  | .ok result => pure result
  | .error message => throw <| IO.userError s!"unexpected runtime parse error: {message}"

def run : IO Unit := do
  let cudaEager : Module.Command.RuntimeRequirements :=
    { device? := some .cuda, execution? := some .eager }
  let (defaults, rest) ←
    parseSelection <| cudaEager.applyDefaults ["--steps", "3"]
  expectEqual "default device" NN.Backend.Device.cuda defaults.device
  expectEqual "default execution" Runtime.ExecutionMode.eager defaults.execution
  expectEqual "preserve command arguments" ["--steps", "3"] rest

  let explicitCpu := cudaEager.applyDefaults ["--device=cpu", "--steps", "1"]
  expectEqual "preserve explicit device"
    ["--execution", "eager", "--device=cpu", "--steps", "1"] explicitCpu
  let (cpuSelection, _) ← parseSelection explicitCpu
  match cudaEager.validate (← IO.ofExcept cpuSelection.toConfig) with
  | .ok () => throw <| IO.userError "command API check failed: accepted conflicting CPU device"
  | .error message =>
      expectEqual "device mismatch message"
        "this command requires --device cuda, not --device cpu" message

  let typedGraph := cudaEager.applyDefaults ["--device", "cuda", "--execution=typed-graph"]
  let (typedGraphSelection, _) ← parseSelection typedGraph
  match cudaEager.validate (← IO.ofExcept typedGraphSelection.toConfig) with
  | .ok () =>
      throw <| IO.userError "command API check failed: accepted conflicting execution mode"
  | .error message =>
      expectEqual "execution mismatch message"
        "this command requires --execution eager, not --execution typed-graph" message

  let matching : Runtime.Config := { device := .cuda, execution := .eager }
  match cudaEager.validate matching with
  | .ok () => pure ()
  | .error message =>
      throw <| IO.userError s!"command API check failed: rejected matching runtime: {message}"

end NN.Tests.API.Command
