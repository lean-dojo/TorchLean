/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Core.ExternalProcess

/-!
# Julia subprocess integration (optional)

This module provides a small wrapper around `IO.Process` for calling out to Julia.

Why this exists:
- TorchLean sometimes treats external tools as **untrusted producers** (e.g. Julia doing a heavy
  numeric search on CPU/GPU) and then checks a small certificate artifact in Lean.
- The IO boundary is explicit: any result returned by Julia should be treated as untrusted unless
  validated in Lean.

This mirrors the existing patterns used in TorchLean for:
- Arb via Python (`NN/Floats/Arb/Oracle.lean`), and
- Gymnasium via Python (`NN/Runtime/RL/Gymnasium/Client.lean`).

Optional dependency:
- This file compiles without Julia installed.
- Julia is only required if you execute code that calls `run`/`runJson`.
- You can override the Julia executable path via the `TORCHLEAN_JULIA` environment variable.

References:
- Lean `IO.Process`: https://leanprover-community.github.io/mathlib4_docs/Init/System/IO.html
- Julia language: https://julialang.org/
-/

@[expose] public section

namespace Runtime
namespace External
namespace Julia

open Lean

/-!
## Resolving the Julia executable
-/

/--
Resolve which Julia executable to use.

If the environment variable `TORCHLEAN_JULIA` is set, it takes precedence. Otherwise we fall back
to `juliaCmd` (default: `"julia"`), which is expected to be on `PATH`.
-/
def resolveJuliaCmd (juliaCmd : String := "julia") : IO String := do
  TorchLean.External.Process.resolveCmdFromEnv "TORCHLEAN_JULIA" juliaCmd

/-!
## Availability checks
-/

/--
Require Julia to be available and return the resolved command.

This is suitable for example runners that want a friendly error message when Julia is missing.
-/
def ensureAvailable (juliaCmd : String := "julia") : IO String := do
  let cmd ← resolveJuliaCmd juliaCmd
  TorchLean.External.Process.ensureCmdAvailable "Julia" cmd #["--version"] (some "TORCHLEAN_JULIA")

/-!
## Running Julia
-/

/--
Run Julia with the given CLI arguments and return `stdout`.

On nonzero exit code, we raise `IO.userError` including the captured stderr.
The `cwd` defaults to `"."` so relative script paths behave like in other TorchLean subprocess
integrations.
-/
def run (args : Array String) (cwd : Option String := some ".") (juliaCmd : String := "julia") :
    IO String := do
  let cmd ← ensureAvailable juliaCmd
  TorchLean.External.Process.runStdoutChecked (ctx := "Julia") (cmd := cmd) (args := args)
    (cwd := cwd)

/--
Run Julia and parse `stdout` as JSON.

Certificate producer scripts commonly print one JSON payload to stdout; this entrypoint checks the
Julia process and parses that payload.
-/
def runJson (args : Array String) (cwd : Option String := some ".") (juliaCmd : String := "julia") :
    IO Json := do
  let cmd ← ensureAvailable juliaCmd
  TorchLean.External.Process.runJsonStdoutChecked (ctx := "Julia")
    (cmd := cmd) (args := args) (cwd := cwd)

end Julia
end External
end Runtime
