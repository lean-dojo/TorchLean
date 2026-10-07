/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.PyTorch.Import.Core
public import NN.Verification.VNNComp.Spec
public import NN.API.CLI.Parser
public import NN.MLTheory.CROWN.Graph.Engine.CROWN.Run

/-!
# VNN-COMP MNIST-FC Checker

VNN-COMP-style mini-suite checker for the MNIST-FC benchmark.

The command consumes exported ONNX/VNNLIB JSON artifacts, runs TorchLean IBP or CROWN bounds on the
imported MLP, and checks the VNNLIB disjunction-of-conjunctions constraints using a sufficient
condition on the output box.

The checker expects exported JSON artifacts. Keep large VNN-COMP snapshots outside git, for example
under `_external/vnncomp/mnist_fc/`, and pass explicit paths when needed.

Run (Lean):
  `scripts/lake.sh exe verify -- vnncomp-mnistfc`
-/

@[expose] public section


namespace NN.Verification.VNNComp.MnistFC

open Lean
open _root_.Spec _root_.TorchLean
open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Graph
open NN.Verification.Json
open Import.PyTorch

/-- Bound propagation/checking mode for the MNIST-FC mini-suite. -/
inductive Mode where
  /-- IBP only: fast but loose. -/
  | ibp
  /-- Forward CROWN output box. -/
  | crown
  /-- Backward CROWN objective pass for each VNNLIB row. -/
  | crownObj
  /-- Backward CROWN objective pass with imported ReLU slopes. -/
  | crownObjAlpha
  deriving DecidableEq, Repr

namespace Mode

/-- CLI spelling for a VNN-COMP checker mode. -/
def toString : Mode → String
  | .ibp => "ibp"
  | .crown => "crown"
  | .crownObj => "crownobj"
  | .crownObjAlpha => "crownobj-alpha"

instance : ToString Mode where
  toString := toString

/-- Parse the `--mode` value accepted by the MNIST-FC checker. -/
def parse (s : String) : Except String Mode :=
  match s with
  | "ibp" => pure .ibp
  | "crown" => pure .crown
  | "crownobj" => pure .crownObj
  | "crownobj-alpha" => pure .crownObjAlpha
  | _ => throw s!"--mode: expected ibp, crown, crownobj, or crownobj-alpha; got `{s}`"

end Mode

/-- Default ignored-local MNIST-FC weights export. -/
def defaultWeightsPath : String :=
  "_external/vnncomp/mnist_fc/model_weights.json"

/-- Default ignored-local MNIST-FC suite export. -/
def defaultSuitePath : String :=
  "_external/vnncomp/mnist_fc/suite.json"

/--
Command-line options for the MNIST-FC VNN-COMP mini-suite verifier.

The defaults point at ignored local artifact paths under `_external/vnncomp/mnist_fc/`.
-/
structure Options where
  /-- Path to exported weights (`model_weights.json`). -/
  weights : String := defaultWeightsPath
  /-- Path to exported instance suite (`suite.json`). -/
  suite   : String := defaultSuitePath
  /-- Maximum number of instances to check (useful for local checks). -/
  max     : Nat := 30
  /--
  Bound propagation mode:
  - `ibp`: IBP only (fast, loose)
  - `crown`: forward CROWN (still produces just an output box)
  - `crownobj`: backward objective pass for each spec row
  - `crownobj-alpha`: objective pass with precomputed ReLU slopes (`--alphas`); the directed Float
    backend rejects imported slopes
  -/
  mode    : Mode := .ibp
  /-- Path to exported alpha slopes JSON (required for `mode = crownobj-alpha`). -/
  alphas  : String := ""
  deriving Repr

/-- Usage text printed by this tool (returned as an error message on `--help`). -/
def usage : String :=
  String.intercalate "\n" [
    "Usage:",
    "  lake exe verify -- vnncomp-mnistfc",
    ("    [--weights=PATH.json] [--suite=PATH.json] [--max=30] " ++
      "[--mode=ibp|crown|crownobj|crownobj-alpha]"),
    "    [--alphas=PATH.json]   (required for crownobj-alpha)",
    "",
    "Default inputs (ignored local artifacts):",
    s!"  {defaultWeightsPath}",
    s!"  {defaultSuitePath}",
    "",
    "Large VNN-COMP exports are not committed. Generate or download them first, then pass",
    "--weights=... and --suite=... explicitly if you store them elsewhere."
  ]

/-- Parse CLI flags into `Options`. -/
def parseArgs (args : List String) : Except String Options := do
  let args := TorchLean.CLI.dropDashDash args
  if TorchLean.CLI.hasHelp args then
    throw usage
  let (weights, args) ← TorchLean.CLI.takeFlagValue args "weights" (default := defaultWeightsPath)
  let (suite, args) ← TorchLean.CLI.takeFlagValue args "suite" (default := defaultSuitePath)
  let (max, args) ← TorchLean.CLI.takeNatFlag args "max" (default := 30)
  let (mode, args) ← TorchLean.CLI.takeParsedFlag args "mode" (default := "ibp") Mode.parse
  let (alphas, args) ← TorchLean.CLI.takeFlagValue args "alphas" (default := "")
  TorchLean.CLI.checkNoArgs args
  pure { weights := weights, suite := suite, max := max, mode := mode, alphas := alphas }

/--
Fail early with a helpful message when an external artifact is missing.

Unlike `Data.requireFile`, this appends the full `usage` text, since a missing VNN-COMP snapshot
usually means the artifacts were never exported. It also keeps `NN.API.Data.Sources` and its
CSV/`.npy` loader stack off this checker's import path.
-/
def requireArtifact (label path : String) : IO Unit := do
  unless (← (path : System.FilePath).pathExists) do
    throw <| IO.userError
      s!"missing {label}: {path}\n\n{usage}"

/--
Weights for one exported fully-connected layer ($y=Wx+b$).

Each layer stores PyTorch's row-major weights and bias with their checked dimensions.
-/
structure LayerWB where
  /-- Input dimension for this layer. -/
  inDim : Nat
  /-- Output dimension for this layer. -/
  outDim : Nat
  /-- Weight matrix, shape `(outDim × inDim)`. -/
  w : _root_.TorchLean.Tensor Float [outDim, inDim]
  /-- Bias vector, shape `(outDim)`. -/
  b : _root_.TorchLean.Tensor Float [outDim]

/-- State-dict keys for the `i`-th linear layer exported by the Python script. -/
def keysForLayer (i : Nat) : (String × String) :=
  (s!"layers.{i}.weight", s!"layers.{i}.bias")

/--
Load an exported MNIST-FC model from JSON weights.

This expects keys like `layers.0.weight`, `layers.0.bias`, etc, and checks that the linear layer
dimensions form a consistent chain.
-/
def loadWeights (path : String) : IO (Array LayerWB) := do
  let j ← readJsonObjectFile path
  let some sd := Import.PyTorch.loadWeights? j
    | throw <| IO.userError "Weights JSON must be an object (or {\"params\": {...}})"

  let layerIdxs : Array Nat :=
    sd.foldl (init := #[]) (fun acc k _v =>
      match Import.PyTorch.parseIndexedKey "layers." ".weight" k with
      | some i => acc.push i
      | none => acc)
  let layerIdxsSorted := (layerIdxs.qsort (· < ·)).eraseReps
  if layerIdxsSorted.isEmpty then
    throw <| IO.userError "No keys of the form layers.<i>.weight found"

  let mut layers : Array LayerWB := #[]
  for i in layerIdxsSorted do
    let (kW, kB) := keysForLayer i
    let some wJ := sd.get? kW
      | throw <| IO.userError s!"Missing key: {kW}"
    let some bJ := sd.get? kB
      | throw <| IO.userError s!"Missing key: {kB}"
    let some (rows, cols) := Import.PyTorch.inferMatrixDims wJ
      | throw <| IO.userError s!"Bad matrix for {kW}"
    let wArr ← NN.Verification.Json.expectFiniteFloatMatrix wJ kW
    let some w := NN.Verification.Util.Tensor.ofRows? rows cols wArr
      | throw <| IO.userError s!"Bad matrix shape for {kW} (expected {rows}x{cols})"
    let bArr ← NN.Verification.Json.expectFiniteFloatArray bJ kB
    let bT ← NN.Verification.Util.Tensor.requireArray kB [rows] bArr
    layers := layers.push { inDim := cols, outDim := rows, w := w, b := bT }

  for i in [0:layers.size - 1] do
    match layers[i]?, layers[i + 1]? with
    | some a, some b =>
        if a.outDim != b.inDim then
          throw <| IO.userError s!"Layer dim mismatch: out={a.outDim} ≠ in={b.inDim}"
    | _, _ =>
        throw <| IO.userError "Internal error while validating the layer chain"

  pure layers

/--
One set of precomputed ReLU alpha slopes for `crownobj-alpha` mode.

This checker assumes the MNIST-FC graph has ReLU nodes at fixed ids (`2` and `4`), so we store the
alpha vectors in a format that can be mapped back to those nodes.
-/
structure AlphaEntry where
  /-- Instance id this alpha payload applies to. -/
  id : Nat
  /-- Alpha vectors for ReLU node 2, indexed by disjunct term. -/
  alpha2 : Array (Array Float)
  /-- Alpha vectors for ReLU node 4, indexed by disjunct term. -/
  alpha4 : Array (Array Float)

/--
Load the exported alpha slopes database (for `mode = crownobj-alpha`).

This expects the `mnist_fc_crownobj_alpha_v0_1` format produced by the Python export script.
-/
def loadAlphas (path : String) : IO (Array AlphaEntry) := do
  let top ← readJsonObjectFile path
  expectFormat top "mnist_fc_crownobj_alpha_v0_1"
  let instArr ← expectFieldArray top "instances" "top-level"
  let mut out : Array AlphaEntry := #[]
  for ex in instArr do
    let exo ← expectObject ex "alpha instance"
    let id ← expectFieldNat exo "id" "alpha instance"
    let alphaObj ← expectFieldObject exo "alpha" "alpha instance"
    let a2J ← expectField alphaObj "2" "alpha"
    let a4J ← expectField alphaObj "4" "alpha"
    let a2 ← NN.Verification.Json.expectFiniteFloatMatrix a2J "alpha.2"
    let a4 ← NN.Verification.Json.expectFiniteFloatMatrix a4J "alpha.4"
    out := out.push { id := id, alpha2 := a2, alpha4 := a4 }
  pure out

/--
Lower the loaded weights into a CROWN `Graph` + `ParamStore`.

The graph alternates linear layers and ReLUs, ending with a linear layer. The number of layers
comes from the export. Return its parameters, input/output dimensions, and output node id.
-/
def buildGraphAndParams (layers : Array LayerWB) : IO (Graph × ParamStore Float × Nat × Nat × Nat)
  := do
  let some first := layers[0]?
    | throw <| IO.userError "Empty layer array"
  let some last := layers.back?
    | throw <| IO.userError "Empty layer array"
  let inDim := first.inDim
  let outDim := last.outDim

  let nodes0 : Array Node :=
    #[{ id := 0
        parents := #[]
        kind := .input
        outShape := .dim inDim .scalar }]
  let baseParams : ParamStore Float := {}
  let (nodes, ps, outId) := layers.zipIdx.foldl (init := (nodes0, baseParams, 0))
    fun (nodes, ps, parentId) (layer, index) =>
      let linId := parentId + 1
      let nodes := nodes.push
        { id := linId
          parents := #[parentId]
          kind := .linear
          outShape := .dim layer.outDim .scalar }
      let ps := { ps with
        linearWB := ps.linearWB.insert linId
          { m := layer.outDim
            n := layer.inDim
            w := layer.w
            b := layer.b } }
      if index + 1 = layers.size then
        (nodes, ps, linId)
      else
        let reluId := linId + 1
        let nodes := nodes.push
          { id := reluId
            parents := #[linId]
            kind := .relu
            outShape := .dim layer.outDim .scalar }
        (nodes, ps, reluId)
  let g : Graph := { nodes := nodes }
  pure (g, ps, inDim, outDim, outId)

/-- Compute an output interval box using IBP (fast, loose). -/
def outputBoxIBP (g : Graph) (ps : ParamStore Float) (outId : Nat) : IO (FlatBox Float) :=
  let ibp := runIBP (α := Float) g ps
  IO.ofExcept <| (NN.MLTheory.CROWN.Graph.outputBox? ibp outId).mapError
    fun msg => s!"IBP produced no output box: {msg}"

/-- Compute an output interval box by running forward CROWN and evaluating the affine bounds on the
  input box. -/
def outputBoxCROWN (g : Graph) (ps : ParamStore Float) (xB : FlatBox Float)
    (inId outId inDim : Nat) : IO (FlatBox Float) :=
  IO.ofExcept (NN.MLTheory.CROWN.Graph.outputBoxCROWN? g ps xB inId outId inDim)

/--
Compute per-node interval boxes to be used by the backward objective pass.

We start from IBP boxes and (when the input dimension is small) refine a small set of parent-node
intervals by evaluating forward CROWN affines on the input box. This makes objective bounds tighter
without paying the full cost of "evaluate every affine on the input box".
-/
def boxesForObjective (g : Graph) (ps : ParamStore Float) (xB : FlatBox Float)
    (inId inDim : Nat) : IO (Array (Option (FlatBox Float))) := do
  let ibp0 := runIBP (α := Float) g ps
  -- For large input dimensions, evaluating forward CROWN affines on the input box is too slow
  -- in this compact checker. Retain the baseline IBP boxes instead.
  if inDim > 64 then
    return ibp0
  let ctx : AffineCtx := { inputId := inId, inputDim := inDim }
  let crown := runCROWN (α := Float) g ps ctx ibp0
  let mut out : Array (Option (FlatBox Float)) := ibp0
  -- Only refine boxes for nodes that will be queried as parents of a unary relaxation
  -- in the backward objective pass (these are the only nodes where tighter pre-activation
  -- intervals matter for CROWN objective bounds).
  let mut need : Array Bool := Array.replicate g.nodes.size false
  for id in List.finRange g.nodes.size do
    let node := g.nodes[id.val]'id.isLt
    match node.parents[0]? with
    | some p1 =>
      let isUnaryRelax : Bool :=
        match node.kind with
        | .relu | .exp | .log | .inv | .sigmoid | .tanh | .softmax _ | .layernorm _ => true
        | _ => false
      if isUnaryRelax && p1 < g.nodes.size then
        need := need.set! p1 true
    | none => pure ()

  for p in List.finRange g.nodes.size do
    if (need[p.val]?).getD false then
      if let some (some b) := crown[p.val]? then
        if hXB : xB.dim = inDim then
          if hIn : b.inDim = inDim then
            let outB := b.evalOnFlatBox xB (hXB.trans hIn.symm)
            let newB : FlatBox Float := { dim := b.outDim, lo := outB.lo, hi := outB.hi }
            out := out.set! p.val (some newB)
  pure out

/--
Try to refute a single spec row using a backward CROWN objective bound.

We build an objective $\mathrm{obj}=\mathrm{row}$ and lower-bound
$\mathrm{row}^{\mathsf{T}}y$ over the input box. If the lower bound is
strictly greater than `rhs`, then the constraint `rowᵀ y <= rhs` cannot hold.
-/
def refutesRowByCROWNObjective
    (g : Graph) (ps : ParamStore Float) (xB : FlatBox Float)
    (ibp : Array (Option (FlatBox Float))) (ctx : AffineCtx)
    (outId outDim : Nat) (row : Tensor Float [outDim]) (rhs : Float) : IO Bool := do
  let obj : FlatTensor Float := { n := outDim, v := row }
  let outB ← IO.ofExcept (backwardObjectiveBox? (α := Float) g ps ctx ibp xB outId obj)
  let lo := getAtOrZero outB.lo [0]
  pure (lo > rhs)

/--
Like `refutesRowByCROWNObjective`, but use precomputed ReLU alpha slopes.

This is a workflow hook for comparing against "fixed alpha" CROWN objective bounds exported from a
reference implementation.
-/
def refutesRowByCROWNObjectiveWithReluAlpha
    (g : Graph) (ps : ParamStore Float) (xB : FlatBox Float)
    (ibp : Array (Option (FlatBox Float))) (inId outId inDim outDim : Nat)
    (row : Tensor Float [outDim]) (rhs : Float)
    (reluAlpha : Array (Option (FlatTensor Float))) : IO Bool := do
  let obj : FlatTensor Float := { n := outDim, v := row }
  let ctx : AffineCtx := { inputId := inId, inputDim := inDim }
  let some loAff := runCROWNBackwardObjectiveLowerWithReluAlpha (α := Float) g ps ctx ibp outId obj
    reluAlpha
    | throw <| IO.userError <|
        "crownobj-alpha: the Float backend's directed CROWN pass has no ReLU relaxation, so " ++
        "imported slopes cannot be used; run with --mode=crownobj"
  if hXB : xB.dim = inDim then
    let loBox := loAff.evalOnFlatBox xB hXB
    let lo := getAtOrZero loBox.lo [0]
    pure (lo > rhs)
  else
    throw <| IO.userError "Input box dim mismatch"

/--
Refute a VNNLIB disjunction-of-conjunctions spec using per-row CROWN objective bounds.

Each row is bounded directly, avoiding some correlations lost when evaluating a row over an
output interval box. This implementation does not guarantee that every resulting bound is tighter.
-/
def vnnlibRefutedByCROWNObjectives
    (g : Graph) (ps : ParamStore Float) (xB : FlatBox Float)
    (inId outId inDim outDim : Nat)
    (spec : NN.Verification.VNNComp.VNNLib.Spec) : IO Bool := do
  let ibp ← boxesForObjective g ps xB inId inDim
  let ctx : AffineCtx := { inputId := inId, inputDim := inDim }
  for term in spec do
    if hCols : term.cols = outDim then
      let mut termRefuted := false
      for i in List.finRange term.rows do
        let row : Tensor Float [outDim] := hCols ▸ term.mat.unstack i
        let rhsVal := term.rhs.getScalar i
        let ok ← refutesRowByCROWNObjective g ps xB ibp ctx outId outDim row rhsVal
        if ok then
          termRefuted := true
          break
      if !termRefuted then
        return false
    else
      return false
  return true

/--
Refute a VNNLIB spec using CROWN objectives with externally-provided ReLU alphas.

The artifact supplies slopes for fixed ReLU node ids `2` and `4`. The current directed Float
backend rejects imported slopes, so this interface does not provide a working alpha-CROWN
comparison mode for that backend.
-/
def vnnlibRefutedByCROWNObjectivesAlpha
    (g : Graph) (ps : ParamStore Float) (xB : FlatBox Float)
    (inId outId inDim outDim : Nat)
    (spec : NN.Verification.VNNComp.VNNLib.Spec)
    (alphas : AlphaEntry) (hid1 hid2 : Nat) : IO Bool := do
  let ibp ← boxesForObjective g ps xB inId inDim
  for termIdx in List.finRange spec.size do
    let term := spec[termIdx.val]'termIdx.isLt
    if hCols : term.cols = outDim then
      -- Build per-node α vector for this disjunct term (objective index = termIdx).
      let a2Row? := alphas.alpha2[termIdx.val]?
      let a4Row? := alphas.alpha4[termIdx.val]?
      let mut reluAlpha : Array (Option (FlatTensor Float)) := Array.replicate g.nodes.size none
      match a2Row?, a4Row? with
      | some a2Row, some a4Row =>
        let some a2T := NN.Verification.Util.Tensor.ofArray? [hid1] a2Row
          | throw <| IO.userError s!"Alpha dim mismatch for node 2 (expected {hid1})"
        let some a4T := NN.Verification.Util.Tensor.ofArray? [hid2] a4Row
          | throw <| IO.userError s!"Alpha dim mismatch for node 4 (expected {hid2})"
        if 2 < g.nodes.size then
          reluAlpha := reluAlpha.set! 2 (some { n := hid1, v := a2T })
        else
          throw <| IO.userError "Graph too small for ReLU node 2"
        if 4 < g.nodes.size then
          reluAlpha := reluAlpha.set! 4 (some { n := hid2, v := a4T })
        else
          throw <| IO.userError "Graph too small for ReLU node 4"
      | _, _ =>
        throw <| IO.userError s!"Missing alphas for instance {alphas.id} termIdx={termIdx.val}"

      let mut termRefuted := false
      for i in List.finRange term.rows do
        let row : Tensor Float [outDim] := hCols ▸ term.mat.unstack i
        let rhsVal := term.rhs.getScalar i
        let ok ← refutesRowByCROWNObjectiveWithReluAlpha g ps xB ibp inId outId inDim outDim row
          rhsVal reluAlpha
        if ok then
          termRefuted := true
          break
      if !termRefuted then
        return false
    else
      return false
  return true

/--
CLI entry point.

This is wired into `scripts/lake.sh exe verify -- vnncomp-mnistfc`.
-/
def main (args : List String) : IO Unit := do
  let args := TorchLean.CLI.dropDashDash args
  if TorchLean.CLI.hasHelp args then
    IO.println usage
    return
  let options ← IO.ofExcept (parseArgs args)
  requireArtifact "weights JSON" options.weights
  requireArtifact "suite JSON" options.suite
  if options.mode = .crownObjAlpha then
    if options.alphas.isEmpty then
      throw <| IO.userError "--alphas is required for --mode=crownobj-alpha"
    requireArtifact "alpha-slope JSON" options.alphas
  IO.println "== TorchLean VNN-COMP mini-suite: MNIST-FC (vnncomp2022) =="
  let layers ← loadWeights options.weights
  let instances0 ← NN.Verification.VNNComp.VNNLib.loadSuite options.suite
  let instances := instances0.take options.max
  let alphaDB? ←
    if options.mode = .crownObjAlpha then
      pure (some (← loadAlphas options.alphas))
    else
      pure none
  let (g, baseParams, inDim, outDim, outId) ← buildGraphAndParams layers
  let inId := 0
  IO.println s!"[mnist_fc] layers={layers.size} nodes={g.nodes.size} inDim={inDim} outDim={outDim}"
  IO.println s!"[mnist_fc] instances={instances.size} mode={options.mode}"

  let mut numericallyRefuted : Nat := 0
  let mut unknown : Nat := 0
  for inst in instances do
    if inst.input.dim != inDim then
      throw <| IO.userError s!"Instance {inst.id}: input dim mismatch"
    let xB := inst.input
    let ps : ParamStore Float := baseParams.seedInputBox inId xB
    let isRefuted ←
      if options.mode = .crownObj then
        vnnlibRefutedByCROWNObjectives g ps xB inId outId inDim outDim inst.spec
      else if options.mode = .crownObjAlpha then
        let some alphaDB := alphaDB?
          | throw <| IO.userError "internal: alphaDB missing"
        let some entry := alphaDB.find? (fun e => e.id = inst.id)
          | throw <| IO.userError s!"No alpha entry for instance id {inst.id}"
        let (hid1, hid2) ←
          match layers[0]?, layers[1]? with
          | some l0, some l1 => pure (l0.outDim, l1.outDim)
          | _, _ => throw <| IO.userError "Need at least 2 hidden layers for MNIST-FC"
        vnnlibRefutedByCROWNObjectivesAlpha g ps xB inId outId inDim outDim inst.spec entry hid1
          hid2
      else
        let box ←
          if options.mode = .crown then
            outputBoxCROWN g ps xB inId outId inDim
          else
            outputBoxIBP g ps outId
        pure (NN.Verification.VNNComp.VNNLib.refutedByOutputBox box inst.spec)
    if isRefuted then
      numericallyRefuted := numericallyRefuted + 1
    else
      unknown := unknown + 1

  IO.println <|
    (s!"[mnist_fc] numerically_refuted={numericallyRefuted} unknown={unknown} " ++
      s!"(mode={options.mode}; outward-rounded host Float; not a Lean safety theorem)")

end NN.Verification.VNNComp.MnistFC
