/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Lyapunov.TwoStage.PipelineIIHybrid
public import NN.MLTheory.CROWN.Lyapunov.TwoStage.PipelineIIIAllInLean
public import NN.MLTheory.CROWN.Lyapunov.TwoStage.PipelineIPythonOnly

/-!
# Run

Executable entrypoint for the Three-Pipeline TwoStage workflow (paper Figure 7).

We keep this separate from `verify` / `verify_all` so the TwoStage workflow can be built/run without
depending on the full repo-wide registry.
-/

@[expose] public section


namespace NN.MLTheory.CROWN.Lyapunov.TwoStage.Run

def usage : String :=
  String.intercalate "\n" [
    "Usage:",
    "  lake env lean --run NN/MLTheory/CROWN/Lyapunov/TwoStage/Run.lean -- allinlean [args...]",
    "  lake env lean --run NN/MLTheory/CROWN/Lyapunov/TwoStage/Run.lean -- hybrid [args...]",
    "  lake env lean --run NN/MLTheory/CROWN/Lyapunov/TwoStage/Run.lean -- certgen [args...]",
    "",
    "Commands:",
    "  allinlean  Pipeline (iii): all-in-Lean TwoStage refinement + IBP/CROWN",
    "  hybrid     Pipeline (ii): PyTorch stage1 (auto) + Lean stage2 + IBP/CROWN",
    "  certgen    Pipeline (i): run crown_verifier.py and emit a Lean cert module",
  ]

/-- CLI entrypoint that dispatches between the TwoStage pipeline variants. -/
def main (args : List String) : IO Unit := do
  let args :=
    match args with
    | "--" :: rest => rest
    | _ => args
  match args with
  | [] =>
      IO.println usage
  | "allinlean" :: rest =>
      NN.MLTheory.CROWN.Lyapunov.TwoStage.PipelineIII.AllInLean.main rest
  | "hybrid" :: rest =>
      NN.MLTheory.CROWN.Lyapunov.TwoStage.PipelineII.Hybrid.main rest
  | "certgen" :: rest =>
      NN.MLTheory.CROWN.Lyapunov.TwoStage.PipelineI.PythonOnly.main rest
  | "--help" :: _ | "-h" :: _ =>
      IO.println usage
  | cmd :: _ =>
      throw <| IO.userError s!"unknown command: {cmd}\n\n{usage}"

end NN.MLTheory.CROWN.Lyapunov.TwoStage.Run

/-!
Lake executables expect an unqualified `main` at the module root.

Keep the implementation namespaced and provide a small root-level wrapper for `lean --run`.
-/

def main (args : List String) : IO Unit :=
  NN.MLTheory.CROWN.Lyapunov.TwoStage.Run.main args
