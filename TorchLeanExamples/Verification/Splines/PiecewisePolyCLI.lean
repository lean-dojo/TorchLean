/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLeanExamples.Verification.Splines.Julia
public import TorchLeanExamples.Verification.Splines.PiecewisePolyCert
public import TorchLean.NN.CLI.Parser

/-!
# Piecewise-polynomial spline certificate CLI

The workflow follows the “external producer, Lean checker” pattern:

- Julia acts as an *untrusted producer* that emits a small JSON certificate describing a
  piecewise-polynomial interpolant of a small dataset, piecewise linear in the bundled example.
- Lean parses and checks the certificate exactly over `Rat` using
  `TorchLean.Examples.Splines`, and with `--arithmetic=ieee` it also replays the
  endpoint equalities under binary32 arithmetic.

The workflow is dependency-free:
- the Julia script uses only Julia Base (no packages),
- the default `scripts/lake.sh exe verify -- spline-cert` path checks a bundled JSON file,
  without requiring Julia,
- passing `--regen` calls Julia to regenerate the JSON and checks the output.

Run via the unified verification CLI:

- Check bundled cert (no Julia required):
  `scripts/lake.sh exe verify -- spline-cert`

- Regenerate by calling Julia (requires `julia` on `PATH` or `TORCHLEAN_JULIA` set):
  `scripts/lake.sh exe verify -- spline-cert --regen`
-/

@[expose] public section

namespace TorchLean.Examples.Splines

open Lean
open Lean.Json

/-- Arithmetic of the check: `exact` checks over `Rat` alone, and `ieee` adds the binary32
replay. -/
inductive Arithmetic where
  /-- `Splines.checkJson`, over `Rat` alone. -/
  | exact
  /-- `Splines.checkJsonBinary32`, over `Rat` and then with the binary32 replay. -/
  | ieee
  deriving DecidableEq, Repr

/-- Parse the command-line arithmetic selector, which must be `exact` or `ieee`. -/
def Arithmetic.parse (value : String) : Except String Arithmetic :=
  match value with
  | "exact" => pure .exact
  | "ieee" => pure .ieee
  | _ => throw s!"--arithmetic: expected exact or ieee; got `{value}`"

/-- Repository-relative path of the bundled piecewise-linear certificate. -/
def defaultCertPath : String :=
  "TorchLeanExamples/Verification/Splines/piecewise_linear_cert.json"

/-- Repository-relative path of the Julia certificate producer. -/
def defaultJuliaScript : String :=
  "TorchLeanExamples/Verification/Splines/fit_piecewise_linear.jl"

/-- Help text for the piecewise-polynomial certificate command. -/
def usage : String :=
  String.intercalate "\n" [
    "Usage:",
    "  lake exe verify -- spline-cert [<path>]",
    "  lake exe verify -- spline-cert --regen",
    "",
    "Arguments:",
    s!"  <path>            certificate JSON path (default: {defaultCertPath})",
    "  --regen            call Julia to regenerate the JSON and check the stdout payload",
    "  --arithmetic=exact|ieee  check exactly (default) or also replay under IEEE arithmetic",
    s!"  --script=PATH     override Julia script path (default: {defaultJuliaScript})",
  ]

/--
Entry point used by the unified verification CLI.

It checks the certificate at the positional path, by default the bundled JSON file. With `--regen`
it runs the Julia producer named by `--script` and checks the JSON payload that it prints on
stdout.
-/
def main (args : List String) : IO Unit := do
  let args := CLI.dropDashDash args

  if CLI.hasHelp args then
    IO.println usage
    return

  -- Every flag error is followed by the usage text.
  let orUsage {β : Type} (result : Except String β) : IO β :=
    IO.ofExcept <| result.mapError fun e => s!"{e}\n\n{usage}"
  let (regen, args) ← orUsage <| CLI.takeBoolFlag args "regen"
  let (arithmetic, args) ← orUsage <|
    CLI.takeParsedFlag args "arithmetic" (default := "exact") Arithmetic.parse
  let (scriptPath, args) ← orUsage <|
    CLI.takeFlagValue args "script" (default := defaultJuliaScript)
  let (certPath, args) ← orUsage <| CLI.takePositional args (default := defaultCertPath)
  orUsage <| CLI.checkNoArgs args
  let check :=
    match arithmetic with
    | .exact => Splines.checkJson
    | .ieee => Splines.checkJsonBinary32

  let j ←
    if regen then
      let jsonStr ←
        runJulia (args := #["--color=no", "--startup-file=no", scriptPath])
      match Json.parse jsonStr with
      | .ok j => pure j
      | .error msg => throw <| IO.userError s!"Julia stdout was not valid JSON: {msg}"
    else
      Json.readFile certPath
  check j

end TorchLean.Examples.Splines
