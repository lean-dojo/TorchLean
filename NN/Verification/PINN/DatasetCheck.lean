/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.PINN.Core
public import NN.Verification.PINN.PdeParse
public import NN.MLTheory.CROWN.Graph
public import NN.Verification.PINN.PyTorch
public import NN.Verification.PINN.Architecture
public import NN.Verification.Util.Json
import Lean.Data.Json

/-!
# PINN Dataset Check

Dataset-backed PINN "sanity checker".

This is intentionally lightweight: it reads a dataset JSON in the same schema
as `train_pinn_1d.py --dataset-json`, evaluates the network on the dataset's
`initial`/`boundary`/`data` points, and reports whether the ground-truth `u`
value is contained in the output interval (with a tolerance).

This does **not** prove that the network solves the PDE; it is a bridge for
using a real reference dataset while exercising the same Lean-side bound
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
open Lean
open Json
open _root_.Spec
open _root_.Spec.Tensor

def JsonObj := Std.TreeMap.Raw String Json

/-- CLI options for `pinn-dataset-check`. -/
structure DatasetCheckOpts where
  /-- weights. -/
  weights : Option String := none
  /-- dataset. -/
  dataset : Option String := none
  /-- eps. -/
  eps     : Float := 0.0
  /-- tol. -/
  tol     : Float := 1e-3
  /-- max Pts. -/
  maxPts  : Nat := 200
  /-- strict. -/
  strict  : Bool := false
  deriving Repr

def usage : String :=
  "Usage:\n" ++
  ("  lake exe verify -- pinn-dataset-check --dataset=PATH.json " ++
    "[--weights=WEIGHTS.json] [--eps=0.0] [--tol=1e-3] [--max=200] " ++
    "[--strict]\n")

def parseFloatLit (s : String) : Option Float :=
  match parseNumber { s := s } with
  | .ok (v, st) => if st.i = s.rawEndPos then some v else none
  | .error _ => none

def parseArgs : List String → Except String DatasetCheckOpts
  | [] => .ok {}
  | a :: rest =>
    let o? : Except String DatasetCheckOpts := parseArgs rest
    match o? with
    | .error e => .error e
    | .ok o =>
      if a.startsWith "--weights=" then
        .ok { o with weights := some (a.drop 10).toString }
      else if a.startsWith "--dataset=" then
        .ok { o with dataset := some (a.drop 10).toString }
      else if a.startsWith "--eps=" then
        match parseFloatLit (a.drop 6).toString with
        | some v => .ok { o with eps := v }
        | none => .error s!"bad --eps: {a}"
      else if a.startsWith "--tol=" then
        match parseFloatLit (a.drop 6).toString with
        | some v => .ok { o with tol := v }
        | none => .error s!"bad --tol: {a}"
      else if a.startsWith "--max=" then
        match (a.drop 6).toString.toNat? with
        | some v => .ok { o with maxPts := v }
        | none => .error s!"bad --max: {a}"
      else if a = "--strict" then
        .ok { o with strict := true }
      else
        .error s!"unknown arg: {a}"

def getFloat (o : JsonObj) (k : String) : Except String Float :=
  match o.get? k with
  | some (.num v) => .ok v.toFloat
  | some (.str s) =>
    match parseFloatLit s with
    | some v => .ok v
    | none => .error s!"field '{k}' is not a float: {s}"
  | some _ => .error s!"field '{k}' is not a number"
  | none => .error s!"missing field '{k}'"

def getYorT (o : JsonObj) : Except String Float := do
  match o.get? "y" with
  | some _ => getFloat o "y"
  | none => getFloat o "t"

def getArrD (o : JsonObj) (k : String) : Except String (Array Json) :=
  match o.get? k with
  | none => .ok #[]
  | some .null => .ok #[]
  | some (.arr a) => .ok a
  | some _ => .error s!"field '{k}' is not an array"

def parseXYZ (j : Json) : Except String (Float × Float × Float) := do
  let o ← j.getObj?
  let x ← getFloat o "x"
  let y ← getYorT o
  let u ← getFloat o "u"
  pure (x, y, u)

def loadDatasetXYZ (path : String) (sectionName : String) : IO (Array (Float × Float × Float)) := do
  let j ← NN.Verification.Json.readJsonFile path
  let o ← match j with
    | .obj o => pure o
    | _ => throw <| IO.userError "Dataset JSON must be an object"
  let arr ← match getArrD o sectionName with
    | .ok a => pure a
    | .error msg => throw <| IO.userError s!"Dataset.{sectionName}: {msg}"
  let mut out : Array (Float × Float × Float) := #[]
  for entry in arr do
    match parseXYZ entry with
    | .ok xyz => out := out.push xyz
    | .error msg => throw <| IO.userError s!"Dataset.{sectionName}: {msg}"
  pure out

def loadGraphAndParams (weightsPath? : Option String) : IO (Graph × ParamStore Float) := do
  match weightsPath? with
  | none =>
    pure (buildGraph2D, seedParamsFloat2D)
  | some path =>
    let j ← NN.Verification.Json.readJsonFile path
    match Import.PINNPyTorch.loadPinnState j with
    | some sd =>
      pure (Import.PINNPyTorch.buildGraph sd, Import.PINNPyTorch.toParamStore sd)
    | none =>
      throw <| IO.userError "Weights JSON did not match expected shapes"

def inInterval (u lo hi tol : Float) : Bool :=
  (u ≥ lo - tol) && (u ≤ hi + tol)

def absDiff (a b : Float) : Float :=
  if a ≥ b then a - b else b - a

def checkSection
    (g : Graph) (ps0 : ParamStore Float) (opts : DatasetCheckOpts)
    (sectionName : String) (pts : Array (Float × Float × Float)) : IO (Nat × Nat × Float) := do
  let outId := SequentialPINNArch.graphOutputId g
  let pts := pts.take opts.maxPts
  let mut okCount : Nat := 0
  let mut badCount : Nat := 0
  let mut maxAbsErr : Float := 0.0
  for (x, y, u) in pts do
    let ps := seedInputFloat2D ps0 x y opts.eps
    let ibp := runIBP (α := Float) g ps
    let some outB := ibp[outId]! | throw <| IO.userError s!"IBP failed at output for {sectionName}"
    let lo := Spec.Tensor.sumSpec outB.lo
    let hi := Spec.Tensor.sumSpec outB.hi
    let mid := (lo + hi) / 2.0
    maxAbsErr := max maxAbsErr (absDiff mid u)
    if inInterval u lo hi opts.tol then
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
  let opts ←
    match parseArgs args with
    | .ok o => pure o
    | .error msg => throw <| IO.userError s!"{msg}\n\n{usage}"
  let some datasetPath := opts.dataset | throw <| IO.userError usage
  let (g, ps0) ← loadGraphAndParams opts.weights
  let initial ← loadDatasetXYZ datasetPath "initial"
  let boundary ← loadDatasetXYZ datasetPath "boundary"
  let data ← loadDatasetXYZ datasetPath "data"

  let (okI, badI, maxI) ← checkSection g ps0 opts "initial" initial
  let (okB, badB, maxB) ← checkSection g ps0 opts "boundary" boundary
  let (okD, badD, maxD) ← checkSection g ps0 opts "data" data

  IO.println s!"[PINN dataset] initial: ok={okI} bad={badI} max|err|≈{maxI}"
  IO.println s!"[PINN dataset] boundary: ok={okB} bad={badB} max|err|≈{maxB}"
  IO.println s!"[PINN dataset] data: ok={okD} bad={badD} max|err|≈{maxD}"

  if opts.strict && (badI + badB + badD) > 0 then
    throw <| IO.userError
      "dataset check failed (--strict): some points not contained by output interval"

end NN.Verification.PINN.DatasetCheck
