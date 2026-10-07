/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.External.Julia
public import NN.API.CLI
public import NN.Verification.Splines.PiecewisePolyCert

/-!
# Piecewise-linear spline certificate CLI

The workflow follows the “external producer, Lean checker” pattern:

- Julia acts as an *untrusted producer* that emits a small JSON certificate describing a
  piecewise-polynomial (piecewise-linear) interpolant of a small dataset.
- Lean parses and checks the certificate exactly over `Rat` using
  `NN.Verification.Splines.PiecewisePolyCert`.

This is dependency-free:
- the Julia script uses only Julia Base (no packages),
- the default `scripts/lake.sh exe verify -- spline-cert` path checks a bundled JSON file,
  without requiring Julia,
- passing `--regen` calls Julia to regenerate the JSON and checks the output.

Run via the unified verification CLI:

- Check bundled cert (no Julia required):
  `scripts/lake.sh exe verify -- spline-cert`

- Regenerate by calling Julia (requires `julia` on `PATH` or `TORCHLEAN_JULIA` set):
  `scripts/lake.sh exe verify -- spline-cert --regen`

References:
- “untrusted producer, trusted checker” workflow: see
  bundled certificates and `scripts/verification/*` producers.
-/

@[expose] public section

namespace NN.Verification.Splines.PiecewiseLinearCLI

open Lean
open Json

/-- Arithmetic used for the optional runtime cross-check. -/
inductive Arithmetic where
  | exact
  | ieee
  deriving DecidableEq, Repr

/-- Parse the command-line arithmetic selector. -/
def parseArithmetic (value : String) : Except String Arithmetic :=
  match value with
  | "exact" => pure .exact
  | "ieee" => pure .ieee
  | _ => throw s!"--arithmetic: expected exact or ieee; got `{value}`"

/-- Repository-relative path of the bundled piecewise-linear certificate. -/
def defaultCertPath : String :=
  "NN/Examples/Verification/Splines/piecewise_linear_cert.json"

/-- Repository-relative path of the Julia certificate producer. -/
def defaultJuliaScript : String :=
  "scripts/verification/splines/fit_piecewise_linear.jl"

/-- Help text for the piecewise-linear certificate command. -/
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

By default, checks the bundled JSON cert on disk. With `--regen`, calls Julia and checks its
stdout JSON payload instead.
-/
def main (args : List String) : IO Unit := do
  let args := TorchLean.CLI.dropDashDash args

  if TorchLean.CLI.hasHelp args then
    IO.println usage
    return

  -- Every flag error is followed by the usage text.
  let orUsage {β : Type} (result : Except String β) : IO β :=
    IO.ofExcept <| result.mapError fun e => s!"{e}\n\n{usage}"
  let (regen, args) ← orUsage <| TorchLean.CLI.takeBoolFlag args "regen"
  let (arithmetic, args) ← orUsage <|
    TorchLean.CLI.takeParsedFlag args "arithmetic" (default := "exact") parseArithmetic
  let (scriptPath, args) ← orUsage <|
    TorchLean.CLI.takeFlagValue args "script" (default := defaultJuliaScript)
  let (certPath, args) ← orUsage <| TorchLean.CLI.takePositional args (default := defaultCertPath)
  orUsage <| TorchLean.CLI.checkNoArgs args
  let check :=
    match arithmetic with
    | .exact => NN.Verification.Splines.PiecewisePolyCert.checkJson
    | .ieee => NN.Verification.Splines.PiecewisePolyCert.checkJsonIEEE32ExecExact

  let j ←
    if regen then
      let jsonStr ←
        Runtime.External.Julia.run (args := #["--color=no", "--startup-file=no", scriptPath])
      match Json.parse jsonStr with
      | .ok j => pure j
      | .error msg => throw <| IO.userError s!"Julia stdout was not valid JSON: {msg}"
    else
      NN.Verification.Json.readJsonFile certPath
  check j

end NN.Verification.Splines.PiecewiseLinearCLI
