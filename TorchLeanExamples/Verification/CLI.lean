/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.Certificate.AbCROWNLeaf
public import TorchLean.Verify.Certificate.CROWNQuery.JSON
public import TorchLeanExamples.Verification.Geometry3D.CLI
public import TorchLean.Verify.Capture
public import TorchLean.NN.CLI.Parser
public import TorchLean.NN.IR.Table
public import TorchLeanExamples.Verification.IBPCert.MLP
public import TorchLeanExamples.Verification.PINN.CLI
public import TorchLeanExamples.Verification.PINN.Certificate
public import TorchLeanExamples.Verification.PINN.DatasetCheck
public import TorchLeanExamples.Verification.ODE.Verify
public import TorchLeanExamples.Verification.IBPWorkflow
public import TorchLeanExamples.Verification.CROWNOpsWorkflow
public import TorchLeanExamples.Verification.TransformerIBPWorkflow
public import TorchLeanExamples.Verification.MLPTrainVerifyWorkflow
public import TorchLeanExamples.Verification.Splines.PiecewisePolyCLI
public import TorchLeanExamples.Verification.VNNComp.MNISTFC
public import TorchLeanExamples.Verification.Lyapunov.AllInLean

/-!
# CLI

Unified verification CLI registry.

The repository exposes several verification entry points: an IBP certificate check for a small
MLP, PINN certificate recomputation, exact CROWN output queries, consistency checks for converted
α,β-CROWN leaf artifacts, ODE enclosure checks, and model-to-IR bound propagation workflows. One
dispatcher runs them all, each as `scripts/lake.sh exe verify -- <tool> [args...]`.

References (background on the verifier families exposed here):

- IBP (interval bound propagation): arXiv:1810.12715.
- CROWN / α,β-CROWN style linear relaxations: see arXiv:1811.00866 and arXiv:2103.06624.
- VNN-COMP (benchmark format / suites): see the VNN-COMP competition pages and papers.
-/

@[expose] public section

namespace TorchLean.Examples.Verify

/-!
## Tool registry

Each runnable verifier or checker is a `Tool` record, and `dispatch` picks one from `tools` by
name.
-/

/--
A single entry in the unified verification CLI: one command name (such as `torchlean-ibp`) and its
handler `run : List String → IO Unit`.

The `all` command skips tools with `includeInAll = false`, such as the interactive and long-running
workflows.
-/
structure Tool where
  /-- Command-line name (first positional arg) used to select this tool. -/
  name : String
  /-- Short help text shown in `list` output. -/
  description : String
  /-- Default path argument, shown in the usage line. Each tool's `run` applies its own default. -/
  defaultArg : Option String := none
  /-- Whether `scripts/lake.sh exe verify -- all` should run this tool. -/
  includeInAll : Bool := true
  /-- Implementation: run the tool with the remaining CLI args. -/
  run : List String → IO Unit

/-- One-line usage string for a single tool in the unified verification CLI. -/
def Tool.usage (t : Tool) : String :=
  match t.defaultArg with
  | none => s!"  {t.name}   -- {t.description}"
  | some d => s!"  {t.name} [<path>]   -- {t.description} (default: {d})"

/-- Usage string for the unified verification CLI dispatcher. -/
def usage (tools : List Tool) : String :=
  let header :=
    String.intercalate "\n" [
      "Usage:",
      "  scripts/lake.sh exe verify -- list",
      "  scripts/lake.sh exe verify -- all",
      "  scripts/lake.sh exe verify -- <tool> [args...]",
      "",
      "Tools:"
    ]
  header ++ "\n" ++ String.intercalate "\n" (tools.map Tool.usage)

/-- The single optional positional path in `args`, or `defaultPath` when there is none. Any further
argument is an error. -/
def getPathOrDefault (args : List String) (defaultPath : String) : Except String String := do
  let args := CLI.dropDashDash args
  let (path, rest) ← CLI.takePositional args (default := defaultPath)
  CLI.checkNoArgs rest
  pure path

/-- Print the lines of a checker's report on standard output, then throw `IO.userError` with the
report's rejection reason if it has one. -/
def printReport (report : Certificate.Report) : IO Unit := do
  for line in report.lines do
    IO.println line
  if let some reason := report.rejection then
    throw <| IO.userError reason

/-- Bundled sample α,β-CROWN leaf artifact. -/
def abcrownLeafSample : String :=
  "TorchLeanExamples/Verification/AbCROWN/sample_abcrown_leaf_artifact_v0_1.json"

/-- The registered verification tools, in the order `list` shows them and `all` runs them. -/
def tools : List Tool :=
  let ibpCertPath := "TorchLeanExamples/Verification/IBPCert/mlp_cert.json"
  [
    { name := "ibp-cert"
      description := "IBP certificate check for a 3 → 4 → 2 ReLU MLP"
      defaultArg := some ibpCertPath
      run := fun args => do
        let path ← IO.ofExcept <| getPathOrDefault args ibpCertPath
        printReport (← IBPCert.verifyCert path) }
  , { name := "ir"
      description := "typed graph checks and optional binary sample comparison (JSON report)"
      includeInAll := false
      run := fun
        | "--op-table" :: args => do
            let path :: names@(_ :: _) := args
              | throw (IO.userError "usage: verify ir --op-table <table.json> <dtype>...")
            IO.FS.writeFile path ((← IO.ofExcept (IR.tableOfNames names)).compress)
        | path :: requests => do
            let artifact ← Json.readFile path
            let requests := if requests.isEmpty then #["structure", "types"] else requests.toArray
            let report := Capture.check artifact requests
            IO.println report.compress
            unless (← IO.ofExcept ((← IO.ofExcept (report.getObjVal? "ok")).getBool?)) do
              throw (IO.userError "one or more IR checks failed; see the JSON report")
        | [] => throw <| IO.userError <| "\n".intercalate [
            "usage: verify ir <document.json> [structure|types|execute|compare|compare_bitwise]",
            "       verify ir --op-table <table.json> <dtype>..."] }
  , { name := "camera-box3d-cert"
      description := "3D camera-box projection certificate check"
      defaultArg := some Geometry3D.defaultCertPath
      run := fun args =>
        Geometry3D.main args }
  ,
    { name := "pinn-cert"
      description := "PINN certificate recomputation check"
      defaultArg := some PINN.Certificate.defaultPath
      run := fun args => do
        let path ← IO.ofExcept <|
          getPathOrDefault args PINN.Certificate.defaultPath
        PINN.Certificate.verify path }
  , { name := "spline-cert"
      description := "piecewise-polynomial certificate checker (optional Julia regen)"
      defaultArg := some Splines.defaultCertPath
      run := fun args =>
        Splines.main args }
  , { name := "pinn-cli"
      description := "interactive PINN residual-bounding CLI"
      includeInAll := false
      run := fun args =>
        PINN.main args }
  , { name := "abcrown-leaf"
      description := "consistency check of a converted α,β-CROWN leaf artifact (no bound recheck)"
      defaultArg := some abcrownLeafSample
      run := fun args => do
        let usage := String.intercalate "\n" [
          "Usage:",
          "  lake exe verify -- abcrown-leaf [<path/to/artifact.json>]",
          "",
          "If no path is provided, runs a small bundled sample artifact:",
          s!"  {abcrownLeafSample}"]
        if CLI.hasHelp args then
          IO.println usage
          return
        match getPathOrDefault args abcrownLeafSample with
        | .ok path => printReport (← Certificate.AbCROWNLeaf.check path)
        | .error e => throw <| IO.userError s!"{e}\n\n{usage}" }
  , { name := "crown-query"
      description := "exact rational CROWN output query (crown_query_v1) with a soundness theorem"
      includeInAll := false
      run := fun args => do
        let usage := "Usage:\n  lake exe verify -- crown-query <path/to/query.json>"
        if CLI.hasHelp args then
          IO.println usage
          return
        let (path, rest) ←
          IO.ofExcept <| CLI.takePositional? (CLI.dropDashDash args)
        IO.ofExcept <| CLI.checkNoArgs rest
        let some path := path
          | throw <| IO.userError s!"missing query path\n\n{usage}"
        let source ← IO.FS.readFile path
        if Certificate.CROWNQuery.acceptsText source then
          IO.println <| s!"[crown-query] {path}: proved for every real input in the box " ++
            "(acceptsText_sound)"
        else
          let reason := Certificate.CROWNQuery.rejectionReason source
          throw <| IO.userError s!"[crown-query] {path}: not proved: {reason}" }
  , { name := "torchlean-ibp"
      description := "TorchLean → IR → IBP workflow (MLP)"
      includeInAll := false
      run := fun args =>
        IBPWorkflow.main args }
  , { name := "torchlean-transformer-ibp"
      description := "TorchLean → IR → IBP workflow (attention/encoder; optional --with-crown)"
      includeInAll := false
      run := fun args =>
        TransformerIBPWorkflow.main args }
  , { name := "torchlean-crown-ops"
      description := "TorchLean → IR → IBP+CROWN workflow (softmax/mse_loss ops)"
      includeInAll := false
      run := fun args =>
        CROWNOpsWorkflow.main args }
  , { name := "torchlean-mlp-workflow"
      description := "train a classifier, then check robustness with CROWN"
      includeInAll := false
      run := fun args =>
        MLPTrainVerifyWorkflow.main args }
  , { name := "pinn-dataset-check"
      description := "PINN dataset pointwise interval containment check"
      defaultArg := some PINN.Dataset.defaultPath
      includeInAll := false
      run := fun args =>
        PINN.Dataset.main args }
  , { name := "ode"
      description := "ODE enclosure verification (sub/super NN bounds)"
      includeInAll := false
      run := fun args =>
        ODE.main args }
  , { name := "vnncomp-mnistfc"
      description := "VNN-COMP-style suite: MNIST-FC (vnncomp2022) via exported JSON"
      includeInAll := false
      run := fun args =>
        VNNComp.main args }
  , { name := "twostage-torchlean-cegis-van"
      description := "all-in-Lean two-stage/CEGIS Lyapunov refinement and bound check"
      includeInAll := false
      run := fun args =>
        Lyapunov.AllInLean.main args }
  ]

/-- Look up a tool by its command-line name. -/
def findTool? (name : String) : Option Tool :=
  tools.find? (fun t => t.name = name)

/--
Dispatch a unified verification command. A leading `--` is dropped, and so is one right after the
tool name.

- no arguments, `list`, `--help` or `-h`: print the usage and the tool list
- `all`: run every tool with `includeInAll = true`, passing it no arguments
- `<tool> [args...]`: run the named tool on `args`, or fail with the usage for an unknown name
-/
def dispatch (args : List String) : IO Unit := do
  let args :=
    match args with
    | "--" :: rest => rest
    | _ => args
  let help := usage tools
  match args with
  | [] =>
      IO.println help
  | "list" :: _ =>
      IO.println help
  | "all" :: _ =>
      -- `all` runs each tool with no arguments. Call a tool by name to pass it arguments.
      for t in tools do
        if !t.includeInAll then
          continue
        IO.println s!"\n== {t.name} =="
        t.run []
  | "--help" :: _ | "-h" :: _ =>
      IO.println help
  | toolName :: rest =>
      let rest :=
        match rest with
        | "--" :: more => more
        | _ => rest
      match findTool? toolName with
      | none =>
          throw <| IO.userError s!"Unknown tool: {toolName}\n\n{help}"
      | some t =>
          t.run rest

end TorchLean.Examples.Verify
