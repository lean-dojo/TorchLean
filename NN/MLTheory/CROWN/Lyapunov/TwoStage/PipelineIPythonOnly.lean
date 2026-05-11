/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Lyapunov.Verification

/-!
# Pipeline (i): Python-only training/verification, oracle-backed Lean theorem packaging

This file corresponds to **Figure 7 (i)** in the TorchLean paper (`arXiv:2602.22631`):

- Stage 1 + Stage 2 run in PyTorch (float32) and produce candidate networks + numeric bounds.
- Lean **does not** re-run α/β-CROWN here.
- Lean’s role is to assign a precise meaning to the exported numbers and derive the usual
  “Lyapunov inequalities hold on region R” statement under a single oracle trust boundary.

Concretely, the trusted boundary is:
`crown_oracle : CrownOracleWitness lyap cert →
  (∀ x ∈ R, V_lo ≤ V(x) ≤ V_hi) ∧ (∀ x ∈ R, Vdot_lo ≤ V̇(x) ≤ Vdot_hi)`.

Everything after that is ordinary real arithmetic.
-/

@[expose] public section


open Spec
open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Lyapunov

namespace NN.MLTheory.CROWN.Lyapunov.TwoStage.PipelineI.PythonOnly

/-!
## How to use in practice

1. Run the Python-side verifier to produce a Lean file containing:
   - a concrete `RealCert n` (the numeric bounds + region), and
   - theorems discharging the side conditions needed by `Lyapunov.Verification`.

   Concretely:
   `python NN/MLTheory/CROWN/Tactics/crown_verifier.py verify --model ... --region ... --dynamics
     ... --format lean-full`

2. Import that generated file in a proof module and instantiate the theorems from
   `NN.MLTheory.CROWN.Lyapunov.Verification`.

This file intentionally contains **no hardcoded numeric certificate**.
The point of pipeline (i) is that the numbers come from the external verifier and are reified into
Lean via generated code.
-/

/-!
## What Lean proves (schema)

Once the external tool produces a concrete certificate, the Lean side is essentially:

- a `LyapunovCert ℝ n` packaging a box region + scalar bounds, and
- a derivation of the usual Lyapunov inequalities on that region, using only:
  `V_lo > 0` and `Vdot_hi < 0`.

This file keeps that statement *parametric* (no numerals), and the generated file supplies the
concrete instance.
-/

section

variable {n : Nat}
variable (lyap : NeuralLyapunov ℝ n)
variable (cert : LyapunovCert ℝ n)

/--
If a certificate provides a strictly-positive lower bound for `V` and a strictly-negative upper
bound for `V̇` on a region, then `V > 0` and `V̇ < 0` hold everywhere on that region.

In pipeline (i), `hV` and `hVdot` are discharged by the **generated** Lean file emitted by
`crown_verifier.py`.
-/
theorem lyapunov_conditions_schema :
    CrownOracleWitness lyap cert →
    cert.V_lo > 0 →
    cert.Vdot_hi < 0 →
    (∀ x, Box.contains cert.region x → lyap.V x > 0) ∧
    (∀ x, Box.contains cert.region x → lyap.Vdot x < 0) :=
by
  intro w hV hVdot
  exact NN.MLTheory.CROWN.Lyapunov.Real.lyapunov_conditions lyap cert w hV hVdot

end

/--
Convenience runner for pipeline (i): call the external verifier (`crown_verifier.py`) and emit a
Lean file (using `--format lean-full`) into `NN/MLTheory/CROWN/Lyapunov/Generated/`.

Why this is an *IO runner* instead of a theorem:
- Lean imports are resolved at compile time, so we cannot “generate a file and then import it”
  within the same compilation unit.
- The intended workflow is:
  1) run this generator (or run `crown_verifier.py` directly),
  2) then `import` the produced module in a proof file.

Usage (via the CLI tool registered in `NN/Verification/CLI.lean`):
`lake exe verify -- twostage-pythononly-certgen --model <path>.pth --region
  \"[-1,1]x[-1,1]\" --dynamics van_der_pol`
-/
def main (args : List String) : IO Unit := do
  let rec stripHandledFlags : List String → List String
    | [] => []
    | "--out" :: _val :: rest => stripHandledFlags rest
    | a :: rest =>
        if a.startsWith "--out=" || a = "--format" || a.startsWith "--format=" ||
           a = "--lean-namespace" || a.startsWith "--lean-namespace=" then
          stripHandledFlags rest
        else
          a :: stripHandledFlags rest

  -- parse: `--model PATH`
  let modelPath ←
    match args.dropWhile (· != "--model") with
    | _ :: path :: _ => pure path
    | _ => throw <| IO.userError "expected `--model <path>`"

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

  let outPath : String :=
    match args.dropWhile (· != "--out") with
    | _ :: p :: _ => p
    | _ =>
        match args.findSome? (fun a => if a.startsWith "--out=" then some (a.drop 6).toString else
          none) with
        | some p => p
        | none => s!"NN/MLTheory/CROWN/Lyapunov/Generated/{safeName}.lean"

  -- ensure output directory exists (best-effort)
  let outDir : String :=
    match outPath.splitOn "/" |>.reverse with
    | [] => "."
    | _file :: restRev =>
        match restRev.reverse with
        | [] => "."
        | xs => String.intercalate "/" xs
  let _ ← (← IO.Process.spawn { cmd := "mkdir", args := #["-p", outDir], stdout := .inherit, stderr := .inherit }).wait

  let script : String := "NN/MLTheory/CROWN/Tactics/crown_verifier.py"
  let baseArgs : Array String :=
    #[script, "verify", "--format", "lean-full", "--lean-namespace",
      "NN.MLTheory.CROWN.Lyapunov.Generated"]

  -- forward user args, but drop any `--out=...` (handled by Lean runner)
  let forwarded : Array String := (stripHandledFlags args).toArray

  let proc := (← IO.Process.spawn { cmd := "python3", args := baseArgs ++ forwarded, stdout := .piped, stderr := .inherit })
  let out ← proc.stdout.readToEnd
  let code := (← proc.wait)
  if code != 0 then
    throw <| IO.userError s!"crown_verifier.py exited with code {code}"
  IO.FS.writeFile outPath out
  IO.println s!"wrote Lean certificate module: {outPath}"
  IO.println <|
    s!"next: import `NN.MLTheory.CROWN.Lyapunov.Generated.{safeName}` in a " ++
    s!"proof file and apply `lyapunov_conditions_schema`"

end NN.MLTheory.CROWN.Lyapunov.TwoStage.PipelineI.PythonOnly
