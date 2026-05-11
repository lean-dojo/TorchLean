/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Lean.Data.Json
import Lean

/-!
# External process helpers

TorchLean frequently calls out to external tools (Python, Julia, ...), typically in the
“untrusted producer, trusted checker” pattern:

- the external tool produces an artifact (often JSON),
- Lean parses it and checks it against a small trusted kernel.

This file centralizes utilities shared across wrappers:
- environment-variable overrides for selecting the executable, and
- availability checks with consistent optional-dependency error messages, and
- “run and parse stdout as JSON” helpers with good error messages.

It intentionally does **not** provide any claims of correctness about the external tool. It only
keeps subprocess handling consistent and non-duplicated.
-/

@[expose] public section

namespace Runtime
namespace External
namespace Process

open Lean

/-!
## Executable resolution
-/

/--
Resolve an executable command name, allowing an environment-variable override.

Example:
- `resolveCmdFromEnv "TORCHLEAN_JULIA" "julia"` uses `$TORCHLEAN_JULIA` if set, otherwise `"julia"`.
-/
def resolveCmdFromEnv (envVar : String) (defaultCmd : String) : IO String := do
  pure <| (← IO.getEnv envVar) |>.getD defaultCmd

/-!
## Availability checks
-/

/--
Check whether a command is available by running it with lightweight version-style arguments.

Any exception (including “executable not found”) is treated as `false`. This helper is for optional
dependencies; callers that require the command should use `ensureCmdAvailable` so users get a
helpful message.
-/
def isCmdAvailable (cmd : String) (args : Array String := #["--version"]) : IO Bool := do
  try
    let out ← IO.Process.output { cmd := cmd, args := args }
    pure (out.exitCode == 0)
  catch _ =>
    pure false

/--
Require a command to be available and return the command name/path.

The `toolName` and `envVar` fields are only used in error messages. Wrappers such as Julia or
Python oracles use this to keep optional-dependency diagnostics consistent across the codebase.
-/
def ensureCmdAvailable (toolName : String) (cmd : String)
    (args : Array String := #["--version"]) (envVar : Option String := none) : IO String := do
  let hint :=
    match envVar with
    | none => s!"Install {toolName} or put `{cmd}` on PATH."
    | some v => s!"Install {toolName} or set `{v}=/path/to/{cmd}`."
  let out ←
    try
      IO.Process.output { cmd := cmd, args := args }
    catch e =>
      throw <|
        IO.userError
          (s!"{toolName} is not available. This feature is optional.\n" ++
           hint ++ "\n" ++
           s!"error: {e}\n")
  if out.exitCode != 0 then
    throw <|
      IO.userError
        (s!"{toolName} is not available. This feature is optional.\n" ++
         hint ++ "\n" ++
         s!"cmd={cmd}\nargs={args.toList}\nexit={out.exitCode}\nstderr:\n{out.stderr}\n")
  pure cmd

/-!
## Running a subprocess and parsing JSON
-/

/--
Run a subprocess and return its captured `stdout`.

On nonzero exit code, raises `IO.userError` including `stderr`. Any exception thrown by the
process runner (e.g. executable not found) is propagated to the caller.
-/
def runStdoutChecked (ctx : String)
    (cmd : String) (args : Array String) (cwd : Option String := some ".") : IO String := do
  let out ←
    try
      IO.Process.output { cmd := cmd, args := args, cwd := cwd }
    catch e =>
      throw <|
        IO.userError
          (s!"{ctx}: failed to start subprocess.\n" ++
           s!"cmd={cmd}\nargs={args.toList}\nerror: {e}\n")
  if out.exitCode != 0 then
    throw <|
      IO.userError
        (s!"{ctx}: subprocess failed.\n" ++
         s!"cmd={cmd}\nargs={args.toList}\nexit={out.exitCode}\nstderr:\n{out.stderr}\n")
  pure out.stdout

/--
Run a subprocess, treat its `stdout` as a single JSON payload, and parse it.

On nonzero exit code, raises `IO.userError`. If JSON parsing fails, raises `IO.userError` including
the raw stdout (which is usually the most helpful debug output).
-/
def runJsonStdoutChecked (ctx : String)
    (cmd : String) (args : Array String) (cwd : Option String := some ".") : IO Json := do
  let out ← runStdoutChecked (ctx := ctx) (cmd := cmd) (args := args) (cwd := cwd)
  match Json.parse out with
  | .ok j => pure j
  | .error msg =>
      throw <| IO.userError s!"{ctx}: JSON parse error: {msg}\nstdout:\n{out}"

end Process
end External
end Runtime
