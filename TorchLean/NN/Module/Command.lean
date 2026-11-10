/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Random
public import TorchLean.NN.Module.Execution

/-!
# Executable Module Commands

`Module.Command.run` is the command-line entry point of executable TorchLean programs, covering
help output, seed and runtime flags, the command's `Requirements`, banners and the exit
code. `RuntimeFlags` holds the parsed arithmetic and device flags, and `withSelectedRuntime`
hands the selected arithmetic to a continuation that is polymorphic in the scalar type.
-/

@[expose] public section

namespace TorchLean
namespace Module

/-- Runtime choices parsed from the shared command-line flags. -/
structure RuntimeFlags where
  /-- Arithmetic semantics for this execution. -/
  arithmetic : Arithmetic := .native
  /-- Requested execution device. -/
  device : Backend.Device := .cpu
  /-- Whether to print each backend capsule when first used. -/
  showBackend : Bool := false
  deriving Repr, DecidableEq

namespace RuntimeFlags

/-- Parse the shared `--arithmetic`, `--device` and `--show-backend` flags, each at most once. -/
def parse
    (arguments : List String) (defaultArithmetic : Arithmetic := .native) :
    Except String (RuntimeFlags × List String) := do
  let (arithmetic, arguments) ←
    Arithmetic.parseAndStrip arguments (default := defaultArithmetic)
  let (device, arguments) ←
    TorchLean.CLI.takeParsedFlag arguments "device" "cpu" Backend.Device.parseSelector
  let (showBackend, arguments) ← TorchLean.CLI.takeBoolFlag arguments "show-backend"
  pure ({ arithmetic, device, showBackend }, arguments)

/-- Convert parsed command-line choices and `seed` to a `Session.Config`. -/
def toConfig (selection : RuntimeFlags) (seed : Nat := 0) : Session.Config :=
  { device := selection.device
    seed
    backendProfile? := none
    showBackend := selection.showBackend }

/-- Print the selected arithmetic and device. -/
def log (selection : RuntimeFlags) : IO Unit := do
  Arithmetic.log selection.arithmetic
  IO.println s!"[TorchLean] device: {selection.device.cliName}"

end RuntimeFlags

/--
Parse the shared runtime flags, validate the resulting `Session.Config` and call `continuation` at
the selected arithmetic with `ofFloat`, that configuration and the remaining arguments.
No `--seed` flag is read here, so the configuration has seed `0`.
-/
def withSelectedRuntime
    (arguments : List String)
    (continuation :
      ∀ {α : Type}, [TorchLean.Storage α] → [Context α] →
        [ToString α] →
        [FromFloat α] →
        (cast : Float → α) → (runtime : Session.Config) →
        (remainingArguments : List String) → IO Unit) :
    IO Unit := do
  let (selection, remainingArguments) ← IO.ofExcept (RuntimeFlags.parse arguments)
  RuntimeFlags.log selection
  let runtime := RuntimeFlags.toConfig selection
  runtime.validateForExecution
  Arithmetic.withRuntime selection.arithmetic (fun {α} _ _ _ =>
    continuation (α := α) (ofFloat (α := α))
      runtime remainingArguments)

end Module
end TorchLean

namespace TorchLean.Module.Command

/-- Runtime properties required by an executable command. -/
structure Requirements where
  /-- Required execution device, or no device restriction. -/
  device? : Option Backend.Device := none
  deriving Inhabited

namespace Requirements

/-- Prepend `--device` for a required device unless the arguments already choose one. -/
def applyDefaults (requirements : Requirements) (arguments : List String) : List String :=
  match requirements.device? with
  | none => arguments
  | some device =>
      if TorchLean.CLI.hasFlagValue arguments "device" then arguments
      else "--device" :: device.cliName :: arguments

/-- Reject a runtime configuration whose device differs from the required one. -/
def validate (requirements : Requirements) (runtime : Session.Config) :
    Except String Unit := do
  match requirements.device? with
  | none => pure ()
  | some required =>
      unless runtime.device == required do
        throw (s!"this command requires --device {required.cliName}, "
          ++ s!"not --device {runtime.device.cliName}")

end Requirements

/-- Banner, help text, flushing, success message and device requirements of one command. -/
structure Config where
  /-- Optional banner to print before executing the program. -/
  banner? : Option (Session.Config → String) := none
  /-- Command-specific help text. When absent, `--help` and `-h` print the generic `usage` text. -/
  usage? : Option String := none
  /-- Flush stdout after printing the banner, when present. -/
  flush : Bool := true
  /-- Print `"{exeName}: ok"` after successful execution. -/
  printSuccess : Bool := false
  /-- Device requirements imposed by this command. -/
  runtime : Requirements := {}
deriving Inhabited

namespace Config

/-- Print the configured banner for `runtime`, if any, then flush stdout when `flush` is set. -/
def printBanner (config : Config) (runtime : Session.Config) : IO Unit := do
  match config.banner? with
  | none => pure ()
  | some banner =>
      IO.println (banner runtime)
      if config.flush then
        (← IO.getStdout).flush

end Config

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
    , s!"  --device auto|{Backend.Device.help}"
    , "  --arithmetic native"
    , "      these programs run in native binary32 only."
    , "  --seed N"
    , "  --show-backend"
    , "      print the backend capsules selected by the device profile."
    , ""
    , "Verification commands:"
    , "  scripts/lake.sh exe verify -- list"
    , "  scripts/lake.sh exe verify -- abcrown-leaf"
    , "  scripts/lake.sh exe verify -- torchlean-mlp-workflow"
    , ""
    , "Use `scripts/lake.sh exe torchlean --help` for the full example list."
    ]

/--
Run a TorchLean executable after parsing the shared seed and runtime flags.

The selected seed initializes TorchLean's global random stream and is also stored in
`Session.Config`, so model initialization and training observe the same seed. With `--help` or
`-h`, `run` prints the usage text and returns `0` before parsing any other flag, and it rejects
arithmetic other than `native`.
-/
def run
    (exeName : String)
    (arguments : List String)
    (action : (runtime : Session.Config) → (remainingArguments : List String) → IO Unit)
    (config : Config := {}) :
    IO UInt32 := do
  let arguments := TorchLean.CLI.dropDashDash arguments
  if TorchLean.CLI.hasHelp arguments then
    IO.println (config.usage?.getD (usage exeName))
    return 0
  let (seed, arguments) ←
    TorchLean.CLI.orThrow exeName (TorchLean.CLI.takeSeed arguments (default := 0))
  let arguments := config.runtime.applyDefaults arguments

  Random.manualSeed seed

  let printSuccess : IO Unit := do
    if config.printSuccess then
      IO.println s!"{exeName}: ok"

  let (selection, remainingArguments) ← IO.ofExcept (RuntimeFlags.parse arguments)
  if selection.arithmetic != .native then
    throw <| IO.userError s!"{exeName}: this program only supports `--arithmetic native`"
  RuntimeFlags.log selection
  let runtime := RuntimeFlags.toConfig selection seed
  runtime.validateForExecution
  TorchLean.CLI.orThrow exeName (config.runtime.validate runtime)
  config.printBanner runtime
  action runtime remainingArguments
  printSuccess
  pure 0

end TorchLean.Module.Command
