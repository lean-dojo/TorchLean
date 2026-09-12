/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.PINN.Core
public import NN.Verification.PINN.Dataset
public import NN.API.CLI.Parser
public import NN.Verification.PINN.PyTorch.ParamStore
public import NN.API.CLI -- shake: keep
public import NN.Verification.PINN.PdeParse -- shake: keep
public import NN.MLTheory.CROWN.Graph -- shake: keep
public import NN.Verification.PINN.PyTorch -- shake: keep
public import NN.Verification.PINN.Architecture -- shake: keep
public import NN.Verification.Util.Json -- shake: keep
import Lean.Data.Json -- shake: keep

/-!
# PINN Dataset Check

Dataset-backed PINN certificate checker.

This is small and explicit: it reads a dataset JSON in the same schema
as `train_pinn_1d.py --dataset-json`, evaluates the network on the dataset's
`initial`/`boundary`/`data` points, and reports whether the ground-truth `u`
value is contained in the output interval (with a tolerance).

This does **not** prove that the network solves the PDE; it is a bridge for
using a real reference dataset while exercising the same Lean side bound
propagation machinery as the PINN CLI.

References:
- PINNs: `https://arxiv.org/abs/1711.10561`
- IBP (interval bounds): `https://arxiv.org/abs/1810.12715`
- CROWN (linear bounds): `https://arxiv.org/abs/1811.00866`
 -/

@[expose] public section


namespace NN.Verification.PINN.DatasetCheck

open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Graph
open NN.Verification.PINN
open NN.Verification.PINN.PdeParse
open Import
open Spec TorchLean
open TorchLean.Tensor

/-- Bundled dataset sample used by `lake exe verify -- pinn-dataset-check`. -/
def defaultDatasetPath : String :=
  "NN/Examples/Verification/PINN/sample_dataset_1d.json"

/-- CLI options for `pinn-dataset-check`. -/
structure Options where
  /-- Optional JSON file containing exported PINN weights. -/
  weights : Option String := none
  /-- Dataset JSON containing initial, boundary, or supervised data points to check. -/
  dataset : Option String := none
  /-- Radius used to seed the input interval around each dataset point. -/
  eps     : Float := 0.0
  /-- Allowed endpoint tolerance when checking that the reference value is enclosed. -/
  tol     : Float := 1e-3
  /-- Maximum number of points checked from each dataset section. -/
  maxPts  : Nat := 200
  /-- Treat any failed enclosure check as a command failure. -/
  strict  : Bool := false
  deriving Repr

/-- Help text for the dataset-backed PINN checker. -/
def usage : String :=
  "Usage:\n" ++
  ("  lake exe verify -- pinn-dataset-check --dataset=PATH.json " ++
    "[--weights=WEIGHTS.json] [--eps=0.0] [--tol=1e-3] [--max=200] " ++
    "[--strict]\n")

/-- Parse command-line flags for `pinn-dataset-check`. -/
def parseArgs (args : List String) : Except String Options := do
  let args := TorchLean.CLI.dropDashDash args
  if TorchLean.CLI.hasHelp args then
    throw usage
  let (weights?, args) ← TorchLean.CLI.takeFlagValue? args "weights"
  let (dataset?, args) ← TorchLean.CLI.takeFlagValue? args "dataset"
  let (eps, args) ← TorchLean.CLI.takeFloatFlag args "eps" (default := 0.0)
  let (tol, args) ← TorchLean.CLI.takeFloatFlag args "tol" (default := 1e-3)
  let (maxPts, args) ← TorchLean.CLI.takeNatFlag args "max" (default := 200)
  let (strict, args) ← TorchLean.CLI.takeBoolFlag args "strict"
  TorchLean.CLI.checkNoArgs args
  unless eps.isFinite && eps ≥ 0.0 do
    throw s!"--eps must be finite and nonnegative, got {eps}"
  unless tol.isFinite && tol ≥ 0.0 do
    throw s!"--tol must be finite and nonnegative, got {tol}"
  pure { weights := weights?
         dataset := dataset?
         eps := eps
         tol := tol
         maxPts := maxPts
         strict := strict }

/-- Load a PINN graph and parameters, using built-in seed parameters when no weights are
supplied. -/
def loadGraphAndParams (weightsPath? : Option String) : IO (Graph × ParamStore Float) := do
  match weightsPath? with
  | none =>
    pure (buildReferenceGraph 2, referenceParams 2)
  | some path =>
    let j ← NN.Verification.Json.readJsonFile path
    match Import.PINNPyTorch.loadPinnState j with
    | some sd =>
      pure (Import.PINNPyTorch.buildGraph sd, Import.PINNPyTorch.toParamStore sd)
    | none =>
      throw <| IO.userError "Weights JSON did not match expected shapes"

/-- Check one dataset section and return `(contained, missed, maxAbsMidpointError)`. -/
def checkSection
    (g : Graph) (baseParams : ParamStore Float) (options : Options)
    (sectionName : String) (pts : Array Dataset.Point) : IO (Nat × Nat × Float) := do
  let outId := SequentialPINNArch.graphOutputId g
  let pts := pts.take options.maxPts
  let mut okCount : Nat := 0
  let mut badCount : Nat := 0
  let mut maxAbsErr : Float := 0.0
  for point in pts do
    let center : TorchLean.Tensor Float [2] :=
      TorchLean.Tensor.dim fun i =>
        TorchLean.Tensor.scalar <| if i.val = 0 then point.x else point.yOrT
    let ps := seedInput baseParams center options.eps
    let ibp := runIBP (α := Float) g ps
    let outB ←
      match NN.MLTheory.CROWN.Graph.outputBox? ibp outId with
      | .ok outB => pure outB
      | .error msg => throw <| IO.userError s!"IBP failed at output for {sectionName}: {msg}"
    let lo := TorchLean.Tensor.sumSpec outB.lo
    let hi := TorchLean.Tensor.sumSpec outB.hi
    let mid := (lo + hi) / 2.0
    maxAbsErr := max maxAbsErr (Dataset.absDiff mid point.u)
    if Dataset.containsWithTol point.u lo hi options.tol then
      okCount := okCount + 1
    else
      badCount := badCount + 1
  pure (okCount, badCount, maxAbsErr)

/--
Entry point: dataset-backed interval containment check for a PINN model.

This is wired into the unified dispatcher as:
`lake exe verify -- pinn-dataset-check [PATH.json]`

The JSON schema matches the exporter used by `train_pinn_1d.py --dataset-json`.
-/
def main (args : List String) : IO Unit := do
  let args := TorchLean.CLI.normalizePathFlag args "dataset" defaultDatasetPath
  let options ←
    match parseArgs args with
    | .ok o => pure o
    | .error msg => throw <| IO.userError s!"{msg}\n\n{usage}"
  let some datasetPath := options.dataset | throw <| IO.userError usage
  let (g, baseParams) ← loadGraphAndParams options.weights
  let initial ← Dataset.loadSection datasetPath "initial"
  let boundary ← Dataset.loadSection datasetPath "boundary"
  let data ← Dataset.loadSection datasetPath "data"

  let (okI, badI, maxI) ← checkSection g baseParams options "initial" initial
  let (okB, badB, maxB) ← checkSection g baseParams options "boundary" boundary
  let (okD, badD, maxD) ← checkSection g baseParams options "data" data

  IO.println s!"[PINN dataset] initial: ok={okI} bad={badI} max|err|≈{maxI}"
  IO.println s!"[PINN dataset] boundary: ok={okB} bad={badB} max|err|≈{maxB}"
  IO.println s!"[PINN dataset] data: ok={okD} bad={badD} max|err|≈{maxD}"

  if options.strict && (badI + badB + badD) > 0 then
    throw <| IO.userError
      "dataset check failed (--strict): some points not contained by output interval"

end NN.Verification.PINN.DatasetCheck
