/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Rand
public import NN.API.Module.Execution

/-!
# Executable Module Commands

Command-line support for executable TorchLean programs, including arithmetic and device selection,
help output, seed parsing, banners, and exit codes.
-/

@[expose] public section

namespace TorchLean
namespace Module

/-- Runtime choices parsed from the shared command-line flags. -/
structure RuntimeSelection where
  /-- Arithmetic semantics for this execution. -/
  arithmetic : TorchLean.Runtime.Arithmetic := .native
  /-- Immediate or typed-graph execution. -/
  execution : Runtime.ExecutionMode := .eager
  /-- Requested execution device. -/
  device : NN.Backend.Device := .cpu
  /-- Whether to print each backend capsule when first used. -/
  showBackend : Bool := false
  deriving Repr, DecidableEq

namespace RuntimeSelection

/--
Parse the shared arithmetic, execution, device, and backend-reporting flags.

Named devices without an installed runtime remain parseable so diagnostics can report the intended
target. `Runtime.Config.validateForExecution` rejects such a configuration before execution.
-/
def parse
    (arguments : List String) (defaultArithmetic : TorchLean.Runtime.Arithmetic := .native) :
    Except String (RuntimeSelection × List String) := do
  let (arithmetic, arguments) ←
    TorchLean.Runtime.Arithmetic.parseAndStrip arguments (default := defaultArithmetic)
  let (execution, arguments) ←
    TorchLean.CLI.takeParsedFlag arguments "execution" (default := "eager")
      TorchLean.Runtime.ExecutionMode.parse
  let rec go (device : NN.Backend.Device) (showBackend : Bool) (acc : List String) :
      List String → Except String (NN.Backend.Device × Bool × List String)
    | .nil => pure (device, showBackend, acc.reverse)
    | .cons "--device" (.cons value rest) => do
        go (← TorchLean.Runtime.Device.parse value) showBackend acc rest
    | .cons "--device" .nil =>
        throw ("missing value after --device (supported: auto | cpu | cuda | rocm | metal | wasm | "
          ++ "tpu | trainium | custom | external)")
    | .cons arg rest =>
        if arg.startsWith "--device=" then do
          let device ← TorchLean.Runtime.Device.parse ((arg.drop "--device=".length).toString)
          go device showBackend acc rest
        else if arg == "--show-backend" then
          go device true acc rest
        else
          go device showBackend (arg :: acc) rest
  let (device, showBackend, remainingArguments) ← go .cpu false [] arguments
  pure ({ arithmetic, execution, device, showBackend }, remainingArguments)

/-- Convert parsed command-line choices to an explicit runtime configuration. -/
def toConfig (selection : RuntimeSelection) (seed : Nat := 0) :
    Except String Runtime.Config := do
  if (NN.Backend.BackendProfile.maintainedForDevice? selection.device).isNone then
    throw (s!"device `{selection.device.cliName}` has no maintained runtime profile; "
      ++ "use a programmatic backend profile")
  pure
    { execution := selection.execution
      device := selection.device
      seed
      backendProfile? := none
      showBackend := selection.showBackend }

/-- Print the selected arithmetic, execution strategy, and device. -/
def log (selection : RuntimeSelection) : IO Unit := do
  TorchLean.Runtime.Arithmetic.log selection.arithmetic
  IO.println
    s!"[TorchLean] execution: {TorchLean.Runtime.ExecutionMode.cliName selection.execution}"
  IO.println s!"[TorchLean] device: {selection.device.cliName}"

end RuntimeSelection

/--
Parse the shared runtime flags, select executable arithmetic, and call `continuation` with the
corresponding literal conversion and explicit runtime options.
-/
def withSelectedRuntime
    (arguments : List String)
    (continuation :
      ∀ {α : Type}, [TorchLean.Storage α] → [Context α] →
        [ToString α] →
        [TorchLean.Runtime.FromFloat α] →
        (cast : Float → α) → (runtime : Runtime.Config) →
        (remainingArguments : List String) → IO Unit) :
    IO Unit := do
  let (selection, remainingArguments) ← match RuntimeSelection.parse arguments with
    | .ok result => pure result
    | .error message => throw <| IO.userError message
  RuntimeSelection.log selection
  let runtime ← match RuntimeSelection.toConfig selection with
    | .ok result => pure result
    | .error message => throw <| IO.userError message
  runtime.validateForExecution
  TorchLean.Runtime.Arithmetic.withRuntime selection.arithmetic (fun {α} _ _ _ =>
    continuation (α := α) (TorchLean.Runtime.ofFloat (α := α))
      runtime remainingArguments)

end Module
end TorchLean

namespace TorchLean.Module.Command

/-- Runtime properties required by an executable command. -/
structure RuntimeRequirements where
  /-- Required execution device, or no device restriction. -/
  device? : Option NN.Backend.Device := none
  /-- Required execution strategy, or no execution-strategy restriction. -/
  execution? : Option Runtime.ExecutionMode := none
  deriving Inhabited

namespace RuntimeRequirements

/-- Supply required runtime flags only when the caller did not choose them explicitly. -/
def applyDefaults (requirements : RuntimeRequirements) (arguments : List String) : List String :=
  let arguments :=
    match requirements.device? with
    | none => arguments
    | some device =>
        if TorchLean.CLI.hasFlagValue arguments "device" then arguments
        else "--device" :: device.cliName :: arguments
  match requirements.execution? with
  | none => arguments
  | some execution =>
      if TorchLean.CLI.hasFlagValue arguments "execution" then arguments
      else "--execution" :: TorchLean.Runtime.ExecutionMode.cliName execution :: arguments

/-- Reject a parsed runtime selection that conflicts with the command's requirements. -/
def validate (requirements : RuntimeRequirements) (runtime : Runtime.Config) :
    Except String Unit := do
  match requirements.device? with
  | none => pure ()
  | some required =>
      unless runtime.device == required do
        throw (s!"this command requires --device {required.cliName}, "
          ++ s!"not --device {runtime.device.cliName}")
  match requirements.execution? with
  | none => pure ()
  | some required =>
      unless runtime.execution == required do
        throw (s!"this command requires --execution "
          ++ s!"{TorchLean.Runtime.ExecutionMode.cliName required}, not --execution "
          ++ s!"{TorchLean.Runtime.ExecutionMode.cliName runtime.execution}")

end RuntimeRequirements

/-- Banner, success-message, and flushing configuration for an executable command. -/
structure Config where
  /-- Optional banner to print before executing the program. -/
  banner? : Option (Runtime.Config → String) := none
  /-- Command-specific help text; the generic runtime help is used when absent. -/
  usage? : Option String := none
  /-- Flush stdout after printing the banner, when present. -/
  flush : Bool := true
  /-- Print `"{exeName}: ok"` after successful execution. -/
  printSuccess : Bool := false
  /-- Device or execution-mode requirements imposed by this command. -/
  runtime : RuntimeRequirements := {}
deriving Inhabited

namespace Config

/-- Print the configured executable banner, if one was supplied. -/
def printBanner (config : Config) (runtime : Runtime.Config) : IO Unit := do
  match config.banner? with
  | none => pure ()
  | some banner =>
      IO.println (banner runtime)
      if config.flush then
        (← IO.getStdout).flush

end Config

/-- How an executable command chooses its arithmetic semantics. -/
inductive Action where
  /-- Allow arithmetic selection; the continuation must work for every executable backend. -/
  | selectedArithmetic
      (continuation :
        ∀ {α : Type}, [TorchLean.Storage α] → [Context α] →
          [ToString α] →
          [TorchLean.Runtime.FromFloat α] →
          (cast : Float → α) → (runtime : Runtime.Config) →
          (remainingArguments : List String) → IO Unit)
  /-- Run a command fixed to native arithmetic rather than exposing arithmetic selection. -/
  | native
      (continuation :
        (runtime : Runtime.Config) → (remainingArguments : List String) → IO Unit)

/-- Generic help text for executables built on `TorchLean.Module.Command.run`. -/
def usage (exeName : String) : String :=
  String.intercalate "\n"
    [ s!"Usage: {exeName} [runtime flags] [command flags]"
    , ""
    , "Quick examples:"
    , s!"  {exeName} --device cpu --steps 10"
    , s!"  {exeName} --device cuda --steps 10"
    , ""
    , "Runtime flags:"
    , "  -h, --help"
    , "  --device auto|cpu|cuda|rocm|metal|wasm|tpu|trainium|custom|external"
    , "      cpu and cuda are implemented by the current eager runtime;"
    , "      other names are planning targets and fail until a runtime is registered."
    , "  --arithmetic native|ieee|complex"
    , "      native is the default; ieee uses FloatLib's executable binary32 format."
    , "  --execution eager|typed-graph"
    , "      eager executes immediately; typed-graph records and reuses a shape-indexed SSA graph."
    , "  --seed N"
    , "  --show-backend"
    , "      print the backend capsules selected by the current device profile."
    , ""
    , "Verification commands:"
    , "  lake exe verify -- list"
    , "  lake exe verify -- margin-report"
    , "  lake exe verify -- abcrown-leaf"
    , "  lake exe verify -- torchlean-mlp-workflow"
    , ""
    , "Use `lake exe torchlean --help` for the full example list."
    ]

/--
Run a TorchLean executable after parsing the shared seed and runtime flags.

The selected seed initializes TorchLean's global random stream and is also stored in
`Runtime.Config`, so
model initialization and either execution mode observe the same seed.
-/
def run
    (exeName : String)
    (arguments : List String)
    (action : Action)
    (config : Config := {}) :
    IO UInt32 := do
  let arguments := TorchLean.CLI.dropDashDash arguments
  if arguments.contains "--help" || arguments.contains "-h" then
    IO.println (config.usage?.getD (usage exeName))
    return 0
  let (seed, arguments) ←
    match TorchLean.CLI.takeSeed arguments (default := 0) with
    | .ok result => pure result
    | .error message => throw <| IO.userError s!"{exeName}: {message}"
  let arguments := config.runtime.applyDefaults arguments

  TorchLean.rand.manualSeed seed

  let printSuccess : IO Unit := do
    if config.printSuccess then
      IO.println s!"{exeName}: ok"

  match action with
  | .selectedArithmetic continuation =>
      withSelectedRuntime arguments
        (fun {α} _ _ _ _ cast runtime remainingArguments => do
        let runtime : Runtime.Config := { runtime with seed := seed }
        match config.runtime.validate runtime with
        | .ok () => pure ()
        | .error message => throw <| IO.userError s!"{exeName}: {message}"
        config.printBanner runtime
        continuation (α := α) cast runtime remainingArguments
        printSuccess)
      pure 0
  | .native continuation =>
      let (selection, remainingArguments) ←
        match RuntimeSelection.parse arguments with
        | .ok result => pure result
        | .error message => throw <| IO.userError message
      if selection.arithmetic != .native then
        throw <| IO.userError s!"{exeName}: this program only supports `--arithmetic native`"
      RuntimeSelection.log selection
      let runtime ← match RuntimeSelection.toConfig selection seed with
        | .ok runtime => pure runtime
        | .error message => throw <| IO.userError message
      runtime.validateForExecution
      match config.runtime.validate runtime with
      | .ok () => pure ()
      | .error message => throw <| IO.userError s!"{exeName}: {message}"
      config.printBanner runtime
      continuation runtime remainingArguments
      printSuccess
      pure 0

end TorchLean.Module.Command
