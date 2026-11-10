/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Subprocess

/-!
# Julia subprocess

A small wrapper around `IO.Process` for calling Julia. With `--regen`, the spline example runs a
Julia script as an untrusted producer, and Lean checks the certificate it prints.

The file compiles without Julia installed. Julia is needed only when code calls `runJulia`, and the
`TORCHLEAN_JULIA` environment variable overrides the Julia executable.

References:
- Lean `IO.Process`: https://leanprover-community.github.io/mathlib4_docs/Init/System/IO.html
- Julia language: https://julialang.org/
-/

@[expose] public section

namespace TorchLean
namespace Examples.Splines

/-!
## Availability checks
-/

/--
Resolve the Julia command, where `TORCHLEAN_JULIA` takes precedence over `juliaCmd`, and return it
after checking that `--version` runs. A missing Julia fails with an install hint.
-/
def juliaCommand (juliaCmd : String) : IO String := do
  let cmd ← Subprocess.command "TORCHLEAN_JULIA" juliaCmd
  Subprocess.require "Julia" cmd #["--version"] (some "TORCHLEAN_JULIA")

/-!
## Running Julia
-/

/--
Run Julia with the given CLI arguments and return `stdout`.

A nonzero exit code raises `IO.userError` with the captured stderr. `cwd` defaults to `"."`, so
relative script paths resolve as in the other TorchLean subprocess integrations.
-/
def runJulia (args : Array String) (cwd : Option String := some ".")
    (juliaCmd : String := "julia") :
    IO String := do
  let cmd ← juliaCommand juliaCmd
  Subprocess.run (ctx := "Julia") (cmd := cmd) (args := args)
    (cwd := cwd)

end Examples.Splines
end TorchLean
