/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.CLI
public import NN.MLTheory.CROWN.Lyapunov.Verification

/-!
# Pipeline (i): Python-produced Lyapunov bounds

This file corresponds to **Figure 7 (i)** in the TorchLean paper (`arXiv:2602.22631`):

- Stage 1 + Stage 2 run in PyTorch (float32) and produce candidate networks + numeric bounds.
- Lean **does not** re-run α/β-CROWN here.
- Lean assigns a precise meaning to the exported numbers and proves the arithmetic consequences of
  those bounds. The connection between the numbers and the network remains an explicit hypothesis
  until a Lean checker establishes it.

The required semantic hypothesis is

$$
\forall x\in R,\quad
V_{\mathrm{lo}}\leq V(x)\leq V_{\mathrm{hi}}
\quad\text{and}\quad
\dot V_{\mathrm{lo}}\leq\dot V(x)\leq\dot V_{\mathrm{hi}}.
$$

Everything after that is ordinary real arithmetic.
-/

@[expose] public section


open Spec TorchLean
open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Lyapunov

namespace NN.MLTheory.CROWN.Lyapunov.TwoStage.PipelineI.PythonOnly

/-!
## How to use in practice

1. Run the Python-side verifier to produce a Lean file containing:
   - a concrete `RealCert n` (the numeric bounds and region), and
   - proofs of the strict numeric margins reported in that record.

   Concretely:
   `python NN/MLTheory/CROWN/Tactics/crown_verifier.py verify --model ... --region ... --dynamics
     ... --format lean-full`

2. Import that generated file in a proof module.
3. Replay the network bounds with a checked TorchLean verifier to construct
   `LyapunovCert.ValidFor`.
4. Apply the theorems from `NN.MLTheory.CROWN.Lyapunov.Verification`.

This file contains **no hardcoded numeric certificate**. In pipeline (i), the numbers come from the
external verifier and are reified into Lean via generated code.
-/

/-!
## What remains to be checked

The generated module proves arithmetic facts about its own endpoints, such as `0 < V_lo`. It does
not prove that those endpoints enclose a network. The semantic replay step must establish both
fields of `LyapunovCert.ValidFor`; only then do positivity and decay follow from the numeric
margins.
-/

/--
Convenience runner for pipeline (i): call the external verifier (`crown_verifier.py`) and emit a
  Lean data file (using `--format lean-full`) into `NN/MLTheory/CROWN/Lyapunov/Generated/`.

Why this is an *IO runner* instead of a theorem:
- Lean imports are resolved at compile time, so we cannot “generate a file and then import it”
  within the same compilation unit.
- The intended workflow is:
  1) run this generator (or run `crown_verifier.py` directly),
  2) import the produced module in a proof file, and
  3) run a semantic checker that constructs `LyapunovCert.ValidFor` for the imported bounds.

Usage (via the CLI tool registered in `NN/Verification/CLI.lean`):
`lake exe verify -- twostage-pythononly-certgen --model <path>.pth --region
  \"[-1,1]x[-1,1]\" --dynamics van_der_pol`
-/
def main (args : List String) : IO Unit := do
  let args := TorchLean.CLI.dropDashDash args

  let modelPath ←
    match TorchLean.CLI.flagValue? args "model" with
    | .ok (some path) => pure path
    | .ok none => throw <| IO.userError "expected `--model <path>`"
    | .error e => throw <| IO.userError e

  let baseName : String :=
    match modelPath.splitOn "/" |>.reverse with
    | [] => modelPath
    | b :: _ => b

  let stem : String :=
    match baseName.splitOn "." |>.reverse with
    | [] => baseName
    | _ext :: restRev =>
        match restRev.reverse with
        | [] => baseName
        | xs => String.intercalate "." xs

  let safeName : String :=
    stem.replace "-" "_" |>.replace "." "_"

  let outPath ←
    match TorchLean.CLI.flagValue? args "out" with
    | .ok (some path) => pure path
    | .ok none => pure s!"NN/MLTheory/CROWN/Lyapunov/Generated/{safeName}.lean"
    | .error e => throw <| IO.userError e

  -- ensure output directory exists (best-effort)
  let outDir : String :=
    match outPath.splitOn "/" |>.reverse with
    | [] => "."
    | _file :: restRev =>
        match restRev.reverse with
        | [] => "."
        | xs => String.intercalate "/" xs
  let mkdirProc ← IO.Process.spawn
    { cmd := "mkdir", args := #["-p", outDir], stdout := .inherit, stderr := .inherit }
  let _ ← mkdirProc.wait

  let script : String := "NN/MLTheory/CROWN/Tactics/crown_verifier.py"
  let baseArgs : Array String :=
    #[script, "verify", "--format", "lean-full", "--lean-namespace",
      "NN.MLTheory.CROWN.Lyapunov.Generated"]

  let forwarded : Array String :=
    (TorchLean.CLI.stripFlagValues args ["out", "format", "lean-namespace"]).toArray

  let proc ← IO.Process.spawn
    { cmd := "python3", args := baseArgs ++ forwarded, stdout := .piped, stderr := .inherit }
  let out ← proc.stdout.readToEnd
  let code := (← proc.wait)
  if code != 0 then
    throw <| IO.userError s!"crown_verifier.py exited with code {code}"
  IO.FS.writeFile outPath out
  IO.println s!"wrote Lean certificate module: {outPath}"
  IO.println <|
    s!"next: import `NN.MLTheory.CROWN.Lyapunov.Generated.{safeName}` and replay " ++
    s!"its bounds with a checker that constructs `LyapunovCert.ValidFor`"

end NN.MLTheory.CROWN.Lyapunov.TwoStage.PipelineI.PythonOnly
