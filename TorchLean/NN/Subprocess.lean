/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Lean.Data.Json.Parser

/-!
# External Process Helpers

TorchLean calls out to external tools such as Python and Julia, usually as an untrusted producer
with a trusted checker: the tool writes an artifact (often JSON), and Lean parses it and checks it
against a small trusted kernel. The wrappers for those tools share these helpers for scratch paths,
executable overrides, optional-dependency checks and subprocess runs, so a missing dependency or a
failing tool is reported the same way everywhere. Correctness of a tool's output is left to the
checker.
-/

@[expose] public section

namespace TorchLean
namespace Subprocess

open Lean

/-!
## Artifact paths
-/

/-- Build-directory scratch space for subprocess adapters and generated check artifacts. -/
def directory (stem : String) : System.FilePath :=
  System.FilePath.mk s!".lake/build/torchlean_{stem}"

/-!
## Executable resolution
-/

/--
Resolve an executable command name, allowing an environment-variable override.

Example:
- `command "TORCHLEAN_JULIA" "julia"` uses `$TORCHLEAN_JULIA` if set, otherwise `"julia"`.
-/
def command (envVar : String) (defaultCmd : String) : IO String := do
  pure <| (← IO.getEnv envVar) |>.getD defaultCmd

/-!
## Availability checks
-/

/--
Whether `cmd` runs and exits with code 0 when given `args` (`--version` by default).

Any exception, "executable not found" included, counts as `false`. The check suits optional
dependencies, and callers that need the command use `require`, whose error says how to install it.
-/
def available (cmd : String) (args : Array String := #["--version"]) : IO Bool := do
  try
    let out ← IO.Process.output { cmd := cmd, args := args }
    pure (out.exitCode == 0)
  catch _ =>
    pure false

/-- Whether `pythonCmd` can import all of `modules` in one `import` statement. An empty list passes
without running Python. -/
def pythonCanImport (modules : Array String) (pythonCmd : String := "python3") : IO Bool := do
  if modules.isEmpty then
    pure true
  else
    let code := "import " ++ String.intercalate ", " modules.toList
    available pythonCmd #["-c", code]

/--
Return `cmd` when it runs and exits with code 0 given `args`. Otherwise the error says that
`toolName` is unavailable and the feature optional. It suggests installing the tool, or else
setting `envVar` to the path of `cmd` when `envVar` is given and putting `cmd` on PATH when it is
not.

`toolName` and `envVar` appear only in that message. Wrappers for Julia and Python call `require`,
so a missing optional dependency is reported the same way across the codebase.
-/
def require (toolName : String) (cmd : String)
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

On nonzero exit code, raises `IO.userError` including `stderr`. An exception thrown by the process
runner (e.g. executable not found) is wrapped in `IO.userError` with the context, command, and
arguments.
-/
def run (ctx : String)
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
def runJson (ctx : String)
    (cmd : String) (args : Array String) (cwd : Option String := some ".") : IO Json := do
  let out ← run (ctx := ctx) (cmd := cmd) (args := args) (cwd := cwd)
  match Json.parse out with
  | .ok j => pure j
  | .error msg =>
      throw <| IO.userError s!"{ctx}: JSON parse error: {msg}\nstdout:\n{out}"

end Subprocess
end TorchLean
