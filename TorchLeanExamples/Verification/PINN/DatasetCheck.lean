/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLeanExamples.Verification.PINN
public import TorchLeanExamples.Verification.PINN.Dataset
public import TorchLean.NN.CLI.Command -- shake: keep
public import TorchLeanExamples.Verification.PINN.PyTorch.Model -- shake: keep

/-!
# PINN Dataset Check

A dataset-backed PINN checker. It reads a dataset JSON in the schema of
`train_pinn.py evolution --dataset-json` and bounds the network by IBP on a box around each point of
the dataset's `initial`, `boundary` and `data` sections. For each section it counts the points whose
ground-truth `u` value lies in the output interval up to a tolerance.

The check ties the network to a real reference dataset with the same Lean bound propagation
machinery as the PINN CLI. It tests the sampled points only and proves nothing about the network
solving the PDE.

References:
- PINNs: `https://arxiv.org/abs/1711.10561`
- IBP (interval bounds): `https://arxiv.org/abs/1810.12715`
- CROWN (linear bounds): `https://arxiv.org/abs/1811.00866`
 -/

@[expose] public section


namespace TorchLean.Examples.PINN

/-- CLI options for `pinn-dataset-check`. -/
structure DatasetOptions where
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

namespace Dataset

open CROWN
open TorchLean.Tensor

/-- Bundled dataset sample used by `scripts/lake.sh exe verify -- pinn-dataset-check`. -/
def defaultPath : String :=
  "TorchLeanExamples/Verification/PINN/sample_dataset_1d.json"

/-- Help text for the dataset-backed PINN checker. -/
def usage : String :=
  "Usage:\n" ++
  ("  lake exe verify -- pinn-dataset-check --dataset=PATH.json " ++
    "[--weights=WEIGHTS.json] [--eps=0.0] [--tol=1e-3] [--max=200] " ++
    "[--strict]\n")

/-- Parse command-line flags for `pinn-dataset-check`. -/
def parseArgs (args : List String) : Except String DatasetOptions := do
  let args := CLI.dropDashDash args
  if CLI.hasHelp args then
    throw usage
  let (weights?, args) ← CLI.takeFlagValue? args "weights"
  let (dataset?, args) ← CLI.takeFlagValue? args "dataset"
  let (eps, args) ← CLI.takeFloatFlag args "eps" (default := 0.0)
  let (tol, args) ← CLI.takeFloatFlag args "tol" (default := 1e-3)
  let (maxPts, args) ← CLI.takeNatFlag args "max" (default := 200)
  let (strict, args) ← CLI.takeBoolFlag args "strict"
  CLI.checkNoArgs args
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

/-- Load a PINN as a graph of registered operators with its payload. Without a weights file it uses
the two-input `referenceModel 2`, and a weights file that does not match the expected shapes is
rejected. -/
def loadModel (weightsPath? : Option String) :
    IO (IR.Graph × IR.Payload Float) := do
  match weightsPath? with
  | none => IO.ofExcept (referenceModel 2)
  | some path =>
    let j ← Json.readFile path
    match Checkpoint.ofJson? j with
    | some sd => IO.ofExcept (Checkpoint.model (fun x => x) sd)
    | none =>
      throw <| IO.userError "Weights JSON did not match expected shapes"

/-- Check the first `maxPts` points of one dataset section and return
`(contained, missed, maxAbsMidpointError)`. -/
def checkSection
    (g : IR.Graph) (payload : IR.Payload Float) (options : DatasetOptions)
    (sectionName : String) (pts : Array Sample) : IO (Nat × Nat × Float) := do
  let pts := pts.take options.maxPts
  let mut okCount : Nat := 0
  let mut badCount : Nat := 0
  let mut maxAbsErr : Float := 0.0
  for point in pts do
    let center : Tensor Float [2] :=
      Tensor.dim fun i =>
        Tensor.scalar <| if i.val = 0 then point.x else point.yOrT
    let ibp := Op.runIBP g payload (Op.inputSeeds g [FlatBox.lInfBall center options.eps])
    let box ← IO.ofExcept (outputInterval s!"IBP failed at output for {sectionName}" g ibp 0)
    let mid := (box.lo + box.hi) / 2.0
    maxAbsErr := max maxAbsErr (Float.abs (mid - point.u))
    if Dataset.containsWithTol point.u box.lo box.hi options.tol then
      okCount := okCount + 1
    else
      badCount := badCount + 1
  pure (okCount, badCount, maxAbsErr)

/--
Entry point of the dataset-backed interval containment check, registered in the unified dispatcher
as `scripts/lake.sh exe verify -- pinn-dataset-check [PATH.json]`. Without a path it checks the
bundled `defaultPath`, and with `--strict` any missed point fails the command.
-/
def main (args : List String) : IO Unit := do
  let args := CLI.normalizePathFlag args "dataset" defaultPath
  let options ←
    match parseArgs args with
    | .ok o => pure o
    | .error msg => throw <| IO.userError s!"{msg}\n\n{usage}"
  let some datasetPath := options.dataset | throw <| IO.userError usage
  let (g, payload) ← loadModel options.weights
  let initial ← Dataset.loadSection datasetPath "initial"
  let boundary ← Dataset.loadSection datasetPath "boundary"
  let data ← Dataset.loadSection datasetPath "data"

  let (okI, badI, maxI) ← checkSection g payload options "initial" initial
  let (okB, badB, maxB) ← checkSection g payload options "boundary" boundary
  let (okD, badD, maxD) ← checkSection g payload options "data" data

  IO.println s!"[PINN dataset] initial: ok={okI} bad={badI} max|err|≈{maxI}"
  IO.println s!"[PINN dataset] boundary: ok={okB} bad={badB} max|err|≈{maxB}"
  IO.println s!"[PINN dataset] data: ok={okD} bad={badD} max|err|≈{maxD}"

  if options.strict && (badI + badB + badD) > 0 then
    throw <| IO.userError
      "dataset check failed (--strict): some points not contained by output interval"

end Dataset

end TorchLean.Examples.PINN
