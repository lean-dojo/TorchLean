/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.External.Julia
public import NN.Verification.Splines.PiecewisePolyCert
import Lean.Data.Json

/-!
# Julia spline certificate: external producer + Lean checker

This file is the runnable entry point for the “external producer, Lean checker” pattern:

- Julia acts as an *untrusted producer* that emits a small JSON certificate describing a
  piecewise-polynomial (piecewise-linear) interpolant of a small dataset.
- Lean parses and checks the certificate exactly over `Rat` using
  `NN.Verification.Splines.PiecewisePolyCert`.

This is intentionally small and dependency-free:
- the Julia script uses only Julia Base (no packages),
- the default `lake exe verify -- spline-cert` path checks a **bundled** JSON file and does not
  require Julia,
- passing `--regen` calls Julia to regenerate the JSON and checks the output.

Run via the unified verification CLI:

- Check bundled cert (no Julia required):
  `lake exe verify -- spline-cert`

- Regenerate by calling Julia (requires `julia` on `PATH` or `TORCHLEAN_JULIA` set):
  `lake exe verify -- spline-cert --regen`

References:
- “untrusted producer, trusted checker” workflow: see
  `NN/Examples/Verification/*` (bundled certs) and `scripts/verification/*` (producers).
-/

@[expose] public section

namespace NN.Examples.Verification.Splines.PiecewiseLinearVerify

open Lean
open Json

def defaultCertPath : String :=
  "NN/Examples/Verification/Splines/piecewise_linear_cert.json"

def defaultJuliaScript : String :=
  "scripts/verification/splines/fit_piecewise_linear.jl"

def usage : String :=
  String.intercalate "\n" [
    "Usage:",
    "  lake exe verify -- spline-cert [<path>]",
    "  lake exe verify -- spline-cert --regen",
    "",
    "Arguments:",
    s!"  <path>            certificate JSON path (default: {defaultCertPath})",
    "  --regen            call Julia to regenerate the JSON and check the stdout payload",
    "  --ieee32           additionally check the same equalities under IEEE32Exec semantics",
    s!"  --script=PATH     override Julia script path (default: {defaultJuliaScript})",
  ]

def getOptArg? (optPrefix : String) (args : List String) : Option String :=
  args.findSome? (fun a =>
    if a.startsWith optPrefix then
      some (a.drop optPrefix.length).toString
    else
      none)

/--
Entry point used by the unified verification CLI.

By default, checks the bundled JSON cert on disk. With `--regen`, calls Julia and checks its
stdout JSON payload instead.
-/
def main (args : List String) : IO Unit := do
  let args :=
    match args with
    | "--" :: rest => rest
    | _ => args

  if args.any (· = "--help") || args.any (· = "-h") then
    IO.println usage
    return

  let regen : Bool := args.any (· = "--regen")
  let ieee32 : Bool := args.any (· = "--ieee32")
  let scriptPath : String := (getOptArg? "--script=" args).getD defaultJuliaScript
  let certPath : String :=
    args.find? (fun a => !(a.startsWith "--")) |>.getD defaultCertPath
  let check :=
    if ieee32 then
      NN.Verification.Splines.PiecewisePolyCert.checkJsonIEEE32ExecExact
    else
      NN.Verification.Splines.PiecewisePolyCert.checkJson

  let j ←
    if regen then
      let jsonStr ←
        -- Call Julia and validate its stdout payload.
        Runtime.External.Julia.run (args := #["--color=no", "--startup-file=no", scriptPath])
      match Json.parse jsonStr with
      | .ok j => pure j
      | .error msg => throw <| IO.userError s!"Julia stdout was not valid JSON: {msg}"
    else
      NN.Verification.Json.readJsonFile certPath
  check j

end NN.Examples.Verification.Splines.PiecewiseLinearVerify
