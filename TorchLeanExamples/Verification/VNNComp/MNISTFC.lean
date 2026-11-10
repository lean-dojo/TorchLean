/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.PyTorch.Import
public import TorchLeanExamples.Verification.VNNComp.VNNLib
public import TorchLean.NN.CLI.Parser
public import TorchLean.Verify.CROWN.Op.ExecCROWN
public import TorchLean.NN.IR.Builder

/-!
# VNN-COMP MNIST-FC Checker

A checker for a VNN-COMP-style mini-suite of the MNIST-FC benchmark. The command reads the
benchmark's ONNX network and VNNLIB properties as exported JSON and bounds the imported MLP with
TorchLean IBP or CROWN. An instance counts as refuted when every term of its VNNLIB disjunction has
a row whose lower bound exceeds the right-hand side. Modes `ibp` and `crown` take that bound over an
output box, and mode `crownobj` runs a backward CROWN pass for each row.

The JSON artifacts are not committed. Keep large VNN-COMP snapshots outside git, for example under
`_external/vnncomp/mnist_fc/`, and pass `--weights` and `--suite` when they live elsewhere.

Run (Lean):
  `scripts/lake.sh exe verify -- vnncomp-mnistfc`
-/

@[expose] public section

open TorchLean.Tensor

namespace TorchLean.Examples.VNNComp

open Lean
open CROWN
open Graph

/-- Bound propagation/checking mode for the MNIST-FC mini-suite. -/
inductive Mode where
  /-- IBP only: fast but loose. -/
  | ibp
  /-- CROWN output box from backward sweeps per output coordinate. -/
  | crown
  /-- Backward CROWN objective pass for each VNNLIB row. -/
  | crownObj
  deriving DecidableEq, Repr

namespace Mode

/-- CLI spelling for a VNN-COMP checker mode. -/
def toString : Mode → String
  | .ibp => "ibp"
  | .crown => "crown"
  | .crownObj => "crownobj"

instance : ToString Mode where
  toString := toString

/-- Parse the `--mode` value accepted by the MNIST-FC checker. -/
def parse (s : String) : Except String Mode :=
  match s with
  | "ibp" => pure .ibp
  | "crown" => pure .crown
  | "crownobj" => pure .crownObj
  | _ => throw s!"--mode: expected ibp, crown, or crownobj; got `{s}`"

end Mode

/-- Default location of the MNIST-FC weights export, a gitignored local file. -/
def defaultWeightsPath : String :=
  "_external/vnncomp/mnist_fc/model_weights.json"

/-- Default location of the MNIST-FC suite export, a gitignored local file. -/
def defaultSuitePath : String :=
  "_external/vnncomp/mnist_fc/suite.json"

/--
Command-line options for the MNIST-FC VNN-COMP mini-suite verifier.

The defaults point at gitignored local artifact paths under `_external/vnncomp/mnist_fc/`.
-/
structure Options where
  /-- Path to exported weights (`model_weights.json`). -/
  weights : String := defaultWeightsPath
  /-- Path to exported instance suite (`suite.json`). -/
  suite   : String := defaultSuitePath
  /-- Check at most this many instances, taken from the start of the suite. -/
  max     : Nat := 30
  /--
  Bound propagation mode:
  - `ibp`: IBP only (fast, loose)
  - `crown`: an output box from CROWN backward sweeps, checked like the IBP box
  - `crownobj`: a backward CROWN objective pass for each spec row
  -/
  mode    : Mode := .ibp
  deriving Repr

/-- Usage text for `--help`: `parseArgs` returns it as its error message, and `main` prints it. -/
def usage : String :=
  String.intercalate "\n" [
    "Usage:",
    "  lake exe verify -- vnncomp-mnistfc",
    ("    [--weights=PATH.json] [--suite=PATH.json] [--max=30] " ++
      "[--mode=ibp|crown|crownobj]"),
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
  let args := CLI.dropDashDash args
  if CLI.hasHelp args then
    throw usage
  let (weights, args) ← CLI.takeFlagValue args "weights" (default := defaultWeightsPath)
  let (suite, args) ← CLI.takeFlagValue args "suite" (default := defaultSuitePath)
  let (max, args) ← CLI.takeNatFlag args "max" (default := 30)
  let (mode, args) ← CLI.takeParsedFlag args "mode" (default := "ibp") Mode.parse
  CLI.checkNoArgs args
  pure { weights := weights, suite := suite, max := max, mode := mode }

/--
Fail early when an external artifact is missing, with the full `usage` text appended, since a
missing VNN-COMP snapshot usually means the export step has not run.

The check is written here because `Data.requireFile` would bring `TorchLean.NN.Data.Source` and
its CSV and `.npy` loaders onto this checker's import path.
-/
def requireArtifact (label path : String) : IO Unit := do
  unless (← (path : System.FilePath).pathExists) do
    throw <| IO.userError
      s!"missing {label}: {path}\n\n{usage}"

/--
Load an exported MNIST-FC model from JSON weights.

Every key `layers.<i>.weight` gives one layer, taken in increasing order of `i` and paired with
`layers.<i>.bias`. Each weight matrix must be rectangular with finite entries, its bias must have
one entry per row, and each layer's output dimension must equal the next layer's input dimension.
-/
def loadWeights (path : String) : IO (Array (SomeLinear Float)) := do
  let j ← Json.readFile path
  let _ ← IO.ofExcept <| Json.object "top-level" j
  let some sd := PyTorch.loadWeights? j
    | throw <| IO.userError "Weights JSON must be an object"
  IO.ofExcept (PyTorch.linearLayers sd)

/--
Build the graph of registered operators for the loaded layers, with its payload. Graph input `0`
feeds the first layer, each layer is `Op.linear` on its weight and bias constants, and a ReLU
follows every layer but the last, whose output is the graph's one output. Returns the graph, its
payload and the input and output dimensions.
-/
def model (layers : Array (SomeLinear Float)) :
    IO (IR.Graph × IR.Payload Float × Nat × Nat) := do
  let some first := layers[0]?
    | throw <| IO.userError "Empty layer array"
  let some last := layers.back?
    | throw <| IO.userError "Empty layer array"
  let (g, P) ← IO.ofExcept <|
    IR.Graph.mlp (fun n => ⟨_, _, Op.relu (s := [n])⟩) layers
  pure (g, P, first.inDim, last.outDim)

/-- Compute the output interval box of graph `G` from the box `xB` of its input using IBP (fast,
loose). -/
def outputBoxIBP (G : IR.Graph) (P : IR.Payload Float) (xB : FlatBox Float) :
    IO (FlatBox Float) :=
  IO.ofExcept <| (Op.outputBox? G (Op.runIBP G P (Op.inputSeeds G [xB])) 0).mapError
    fun msg => s!"IBP produced no output box: {msg}"

/-- Compute an output interval box by CROWN on the graph of registered operators: interval
  propagation, then a lower and an upper directed backward sweep per output coordinate, evaluated
  on the input box and met with the IBP output box. The box must have the `inDim` entries of graph
  input `0`. -/
def outputBoxCROWN (G : IR.Graph) (P : IR.Payload Float) (xB : FlatBox Float)
    (inDim : Nat) : IO (FlatBox Float) :=
  IO.ofExcept do
    unless xB.dim = inDim do throw "expected the box of graph input 0"
    Op.outputBoxCROWN? G P xB 0

/--
Compute the per-node interval boxes for the backward objective pass.

These are the IBP boxes, and when the input dimension is at most 64 we narrow every node's box by
full CROWN (`Op.crownBoxes`: each node's box met with its backward affine bounds on the input box).
In that case the box must have the `inDim` entries of graph input `0`.
-/
def boxesForObjective (G : IR.Graph) (P : IR.Payload Float) (xB : FlatBox Float)
    (inDim : Nat) : IO (Array (Option (FlatBox Float))) := do
  let seeds := Op.inputSeeds G [xB]
  let ibp0 := Op.runIBP G P seeds
  -- Full CROWN on every node is too slow for large inputs in this compact checker, so those
  -- keep the IBP boxes.
  if inDim > 64 then
    return ibp0
  IO.ofExcept (ε := String) do
    unless xB.dim = inDim do throw "expected the box of graph input 0"
    let some inputId := Op.inputId? G 0 | throw "the graph has no input 0"
    pure (Op.crownBoxes G P { inputId, inputDim := xB.dim } seeds ibp0 xB)

/--
Try to refute one spec row `rowᵀ y <= rhs` with a backward CROWN objective bound.

The objective is $\mathrm{obj}=\mathrm{row}$, and we lower-bound $\mathrm{row}^{\mathsf{T}}y$ over
the input box. A lower bound strictly greater than `rhs` shows that no output reachable from the
box satisfies the row.
-/
def refutesRowByCROWNObjective
    (G : IR.Graph) (P : IR.Payload Float) (xB : FlatBox Float)
    (ibp : Array (Option (FlatBox Float))) (inDim outDim : Nat) (row : Tensor Float [outDim])
    (rhs : Float) : IO Bool := do
  let obj : FlatTensor Float := { n := outDim, v := row }
  let outB ← IO.ofExcept do
    unless xB.dim = inDim do throw "expected the box of graph input 0"
    Op.backwardObjectiveBox? G P ibp xB 0 obj
  let lo := getAtOrZero outB.lo [0]
  pure (lo > rhs)

/--
Try to refute a VNNLIB disjunction-of-conjunctions spec with a CROWN objective bound for each row.
The spec is refuted when every term has a row that `refutesRowByCROWNObjective` refutes, and a term
whose width differs from the network output makes the result `false`.

Bounding each row on its own keeps correlations between outputs that an output interval box loses.
`Op.backwardObjectiveBox?` also meets each row bound with the row over the output's interval box,
so it is never looser than that box bound.
-/
def refutedByCROWNObjectives
    (G : IR.Graph) (P : IR.Payload Float) (xB : FlatBox Float)
    (inDim outDim : Nat) (spec : VNNComp.Property) : IO Bool := do
  let ibp ← boxesForObjective G P xB inDim
  for term in spec do
    if hCols : term.cols = outDim then
      let mut termRefuted := false
      for i in List.finRange term.rows do
        let row : Tensor Float [outDim] := hCols ▸ term.mat.unstack i
        let rhsVal := term.rhs.getScalar i
        let ok ← refutesRowByCROWNObjective G P xB ibp inDim outDim row rhsVal
        if ok then
          termRefuted := true
          break
      if !termRefuted then
        return false
    else
      return false
  return true

/--
CLI entry point for `scripts/lake.sh exe verify -- vnncomp-mnistfc`. It checks the first `max`
suite instances in the chosen mode and prints the counts of numerically refuted and unknown
instances.
-/
def main (args : List String) : IO Unit := do
  let args := CLI.dropDashDash args
  if CLI.hasHelp args then
    IO.println usage
    return
  let options ← IO.ofExcept (parseArgs args)
  requireArtifact "weights JSON" options.weights
  requireArtifact "suite JSON" options.suite
  IO.println "== TorchLean VNN-COMP mini-suite: MNIST-FC (vnncomp2022) =="
  let layers ← loadWeights options.weights
  let instances0 ← VNNComp.loadSuite options.suite
  let instances := instances0.take options.max
  let (g, P, inDim, outDim) ← model layers
  IO.println s!"[mnist_fc] layers={layers.size} nodes={g.nodes.size} inDim={inDim} outDim={outDim}"
  IO.println s!"[mnist_fc] instances={instances.size} mode={options.mode}"

  let mut numericallyRefuted : Nat := 0
  let mut unknown : Nat := 0
  for inst in instances do
    if inst.input.dim != inDim then
      throw <| IO.userError s!"Instance {inst.id}: input dim mismatch"
    let xB := inst.input
    let isRefuted ←
      if options.mode = .crownObj then
        refutedByCROWNObjectives g P xB inDim outDim inst.property
      else
        let box ←
          if options.mode = .crown then
            outputBoxCROWN g P xB inDim
          else
            outputBoxIBP g P xB
        pure (VNNComp.refutedByOutputBox box inst.property)
    if isRefuted then
      numericallyRefuted := numericallyRefuted + 1
    else
      unknown := unknown + 1

  IO.println <|
    (s!"[mnist_fc] numerically_refuted={numericallyRefuted} unknown={unknown} " ++
      s!"(mode={options.mode}; outward-rounded host Float; not a Lean safety theorem)")

end TorchLean.Examples.VNNComp
