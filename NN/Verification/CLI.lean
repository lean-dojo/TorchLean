/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.Cert.AbCrownLeafCert
public import NN.Examples.Verification
public import NN.Verification.Geometry3D.CLI
public import NN.Verification.PINN.CLI
public import NN.Verification.PINN.Certificate
public import NN.Verification.PINN.DatasetCheck
public import NN.Verification.ODE.Verify

/-!
# CLI

Unified verification CLI registry.

The repository exposes several verification entry points: LiRPA-style bound checks, PINN
certificate recomputation, α,β-CROWN leaf artifact checks, robustness margin certificates, ODE
enclosure checks, and model-to-IR bound propagation workflows.

This file defines a single dispatcher so users can run everything from:
  `lake exe verify -- <tool> [args...]`

References (background on the verifier families exposed here):

- IBP (interval bound propagation): arXiv:1810.12715.
- CROWN / α,β-CROWN style linear relaxations: see arXiv:1811.00866 and arXiv:2103.06624.
- VNN-COMP (benchmark format / suites): see the VNN-COMP competition pages and papers.
-/

@[expose] public section


namespace NN.Verification.CLI

/-!
## Tool registry

We represent each runnable verifier or checker as a `Tool` record, then expose a simple dispatcher
over a list of tools.
-/

/--
A single entry in the unified verification CLI.

Each `Tool` corresponds to one command name (e.g. `torchlean-ibp`) and a handler
`run : List String → IO Unit`.

Design notes:
- Tools with `includeInAll = false` are interactive or long-running workflows and are excluded from
  the `all` command.
- Tools may declare a default path argument for convenience.
-/
structure Tool where
  /-- Command-line name (first positional arg) used to select this tool. -/
  name : String
  /-- Short help text shown in `list` output. -/
  description : String
  /-- Default file/path argument, if the tool expects a path. -/
  defaultArg : Option String := none
  /-- Whether `lake exe verify -- all` should run this tool. -/
  includeInAll : Bool := true
  /-- Implementation: run the tool with the remaining CLI args. -/
  run : List String → IO Unit

/-- One-line usage string for a single tool in the unified verification CLI. -/
def toolUsage (t : Tool) : String :=
  match t.defaultArg with
  | none => s!"  {t.name}   -- {t.description}"
  | some d => s!"  {t.name} [<path>]   -- {t.description} (default: {d})"

/-- Usage string for the unified verification CLI dispatcher. -/
def usage (tools : List Tool) : String :=
  let header :=
    String.intercalate "\n" [
      "Usage:",
      "  lake exe verify -- list",
      "  lake exe verify -- all",
      "  lake exe verify -- <tool> [args...]",
      "",
      "Tools:"
    ]
  header ++ "\n" ++ String.intercalate "\n" (tools.map toolUsage)

/-- Extract a path argument if present; otherwise fall back to a default path. -/
def getPathOrDefault (args : List String) (defaultPath : String) : String :=
  match args with
  | [] => defaultPath
  | "--" :: rest => rest.getD 0 defaultPath
  | a :: _ => a

/-- Tool group: LiRPA-style bound propagation and certificate checking. -/
def lirpaTools : List Tool :=
  let mk (name desc defaultPath : String) (k : String → IO Unit) : Tool :=
    { name := name
      description := desc
      defaultArg := some defaultPath
      run := fun args => k (getPathOrDefault args defaultPath) }
  [
    mk "lirpa-mlp" "IBP cert: feed-forward MLP"
      "NN/Examples/Verification/LiRPA/mlp_cert.json"
      NN.Examples.Verification.LiRPA.MlpVerify.verifyCert
  , mk "lirpa-cnn" "IBP cert: CNN conv→head"
      "NN/Examples/Verification/LiRPA/cnn_cert.json"
      NN.Examples.Verification.LiRPA.CnnVerify.verifyCert
  , mk "lirpa-attention" "IBP cert: attention softmax block"
    "NN/Examples/Verification/LiRPA/attention_softmax_cert.json"
      NN.Examples.Verification.LiRPA.AttentionVerify.verifyCert
  , mk "lirpa-gru" "IBP cert: GRU gate"
      "NN/Examples/Verification/LiRPA/gru_gate_cert.json"
      NN.Examples.Verification.LiRPA.GruVerify.verifyCert
  , mk "lirpa-encoder" "IBP cert: transformer encoder block"
    "NN/Examples/Verification/LiRPA/transformer_encoder_cert.json"
      NN.Examples.Verification.LiRPA.TransformerEncoderVerify.verifyCert
  ]

/-- Tool group: other verification-related utilities. -/
def otherTools : List Tool :=
  [
    { name := "camera-box3d-cert"
      description := "3D camera-box projection certificate check"
      defaultArg := some "NN/Verification/Geometry3D/smoke_box3d_camera_cert.json"
      run := fun args =>
        NN.Verification.Geometry3D.CLI.main args }
  ,
    { name := "pinn-cert"
      description := "PINN certificate recomputation check"
      defaultArg := some "NN/Examples/Verification/PINN/pinn_cert.json"
      run := fun args =>
        NN.Verification.PINN.Certificate.verifyCert
          (getPathOrDefault args "NN/Examples/Verification/PINN/pinn_cert.json") }
  , { name := "spline-cert"
      description := "piecewise-polynomial certificate checker (optional Julia regen)"
      defaultArg := some "NN/Examples/Verification/Splines/piecewise_linear_cert.json"
      run := fun args =>
        NN.Examples.Verification.Splines.PiecewiseLinearVerify.main args }
  , { name := "pinn-cli"
      description := "interactive PINN residual-bounding CLI"
      includeInAll := false
      run := fun args =>
        NN.Verification.PINN.CLI.main args }
  , { name := "abcrown-leaf"
      description := "α,β-CROWN leaf artifact structural check"
      defaultArg := some "NN/Examples/Verification/AbCrown/sample_abcrown_leaf_cert_v0_1.json"
      run := fun args => NN.Verification.Cert.AbCrownLeafCert.run args }
  , { name := "margin-cert"
      description := "logit-margin certificate check (bounds ⇒ certified label)"
      defaultArg := some "NN/Examples/Verification/Robustness/digits_linear_margin_cert.json"
      run := fun args => NN.Examples.Verification.Robustness.VerifyMarginCert.run args }
  , { name := "torchlean-robustness"
      description := "TorchLean → IR margin certification workflow"
      includeInAll := false
      run := fun args =>
        NN.Examples.Verification.Robustness.TorchLeanRobustness.main args }
  , { name := "torchlean-ibp"
      description := "TorchLean → IR → IBP workflow (MLP)"
      includeInAll := false
      run := fun args =>
        NN.Examples.Verification.TorchLean.TorchLeanIBP.main args }
  , { name := "torchlean-transformer-ibp"
      description := "TorchLean → IR → IBP workflow (attention/encoder; optional --with-crown)"
      includeInAll := false
      run := fun args =>
        NN.Examples.Verification.TorchLean.TorchLeanTransformerIBP.main args }
  , { name := "torchlean-crown-ops"
      description := "TorchLean → IR → IBP+CROWN workflow (softmax/mse_loss ops)"
      includeInAll := false
      run := fun args =>
        NN.Examples.Verification.TorchLean.TorchLeanCrownOps.main args }
  , { name := "torchlean-mlp-workflow"
      description := "TorchLean MLP: train with compiled backend, then run IBP+CROWN"
      includeInAll := false
      run := fun args =>
        NN.Examples.Verification.TorchLean.TorchLeanMlpWorkflow.main args }
  , { name := "pinn-dataset-check"
      description := "PINN dataset pointwise interval containment check"
      defaultArg := some "NN/Examples/Verification/PINN/sample_dataset_1d.json"
      includeInAll := false
      run := fun args =>
        let defaultPath := "NN/Examples/Verification/PINN/sample_dataset_1d.json"
        let args :=
          match args with
          | [] => [s!"--dataset={defaultPath}"]
          | a :: rest =>
              if a.startsWith "--" then
                a :: rest
              else
                s!"--dataset={a}" :: rest
        NN.Verification.PINN.DatasetCheck.main args }
  , { name := "ode"
      description := "ODE enclosure verification (sub/super NN bounds)"
      includeInAll := false
      run := fun args =>
        NN.Verification.ODE.Verify.main args }
  , { name := "digits"
      description := "run Lean IBP/CROWN certified-accuracy workflow (not a cert checker)"
      includeInAll := false
      run := fun args =>
        NN.Verification.Robustness.Digits.main args }
  , { name := "vnncomp-mnistfc"
      description := "VNN-COMP-style suite: MNIST-FC (vnncomp2022) via exported JSON"
      includeInAll := false
      run := fun args =>
        NN.Examples.Verification.VNNComp.MnistFcVerify.main args }
  ]

/-- The full list of registered verification tools (LiRPA plus other workflows/checkers). -/
def tools : List Tool :=
  lirpaTools ++ otherTools

/-- Look up a tool by its command-line name. -/
def findTool? (name : String) : Option Tool :=
  tools.find? (fun t => t.name = name)

/--
Dispatch a unified verification command.

Supported invocations:
- `list`: show usage and tool list
- `all`: run all non-interactive tools (ignores extra args)
- `<tool> [args...]`: run a named tool
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
      -- `all` ignores extra args in this command; use individual tools for extra args.
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

end NN.Verification.CLI

/-!
`lake exe verify` executes the unqualified `main`.

We keep the implementation namespaced and provide a thin root-level wrapper so the Lake executable
entrypoint remains stable.
-/

/--
`lake exe verify` entry point.

This is a thin wrapper around the unified dispatcher `NN.Verification.CLI.dispatch`.
-/
def main (args : List String) : IO Unit :=
  NN.Verification.CLI.dispatch args
