/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Entrypoint.Spec
public import NN.MLTheory.CROWN.Core
public import NN.MLTheory.CROWN.Graph
public import NN.MLTheory.CROWN.Lyapunov.TwoStage.Core
public import NN.MLTheory.CROWN.Lyapunov.TwoStage.ExecUtils
public import NN.Runtime.Autograd.TorchLean.Autodiff
public import NN.Runtime.Autograd.TorchLean.Backend
public import NN.Runtime.Autograd.TorchLean.Module
public import NN.Verification.TorchLean.Compile
public import NN.Verification.Util.Json

/-!
# Pipeline (ii): Hybrid PyTorch → TorchLean (exact float32) → IBP/CROWN post-check

This file corresponds to **Figure 7 (ii)** in the TorchLean paper (`arXiv:2602.22631`).

What is “hybrid” here:
- **Stage 1** (outside Lean): train in PyTorch using ordinary float32.
  The output is exported as *bit-exact* float32 parameters (a JSON array of uint32 bit patterns).
- **Stage 2** (inside Lean): load those exact bits into `α = IEEE32Exec` (Lean's executable model of
  IEEE-754 float32), run a small refinement loop (PGD on input, SGD on parameters), and then
  compile the TorchLean loss to the shared verifier IR and run in-repo IBP/CROWN bounds on a box.

Trust boundary:
- Stage-1 training is untrusted and only provides an initialization.
- The Stage-2 computation and the final IBP/CROWN bound propagation are *inside Lean*.

Stage-1 export script:
`python3 scripts/verification/two_stage/export_van_stage1_bits.py --width 100
  --steps 10`

Run this pipeline:
`lake exe verify -- twostage-hybrid-van-stage2`

Convenience flags for this workflow runner:
- omit the weights path to auto-use `_external/van_stage1_w{width}_bits.json`
- if the weights file is missing (or `--stage1` is passed), we run the stage-1 exporter:
  `python3 scripts/verification/two_stage/export_van_stage1_bits.py ...`
- `--stage1-steps=N` controls stage-1 SGD steps (default: 10)
- `--weights=PATH` overrides the weights JSON path
-/

@[expose] public section


open Spec
open Tensor

namespace NN.MLTheory.CROWN.Lyapunov.TwoStage.PipelineII.Hybrid

open Lean
open Lean.Data
open Lean.Json
open NN.Verification.Json

open _root_.TorchLean.Floats.IEEE754
open Runtime
open Runtime.Autograd
open NN.MLTheory.CROWN.Graph
open NN.MLTheory.CROWN

open NN.MLTheory.CROWN.Lyapunov.TwoStage.Core
open NN.MLTheory.CROWN.Lyapunov.TwoStage.ExecUtils

abbrev α : Type := IEEE32Exec

abbrev xDim : Nat := Core.xDim
abbrev uDim : Nat := Core.uDim

abbrev xShape : Shape := Core.xShape
abbrev uShape : Shape := Core.uShape

abbrev paramShapes (width : Nat) : List Shape :=
  Core.paramShapes width

/-- Local alias for `ExecUtils.nat` (coercion `Nat → IEEE32Exec`). -/
abbrev nat (k : Nat) : α := ExecUtils.nat k

/-- Learning rate for the stage-2 SGD loop. -/
def lr : α := ExecUtils.defaultLr

/-- PGD step size when searching for counterexample-ish inputs. -/
def pgdStepSize : α := ExecUtils.defaultPgdStepSize

/-- Radius of the training box `[-rad, rad]^2` (also used for clamping PGD iterates). -/
def rad : α := ExecUtils.defaultRad

/-- Half-width of the small box around the origin used for the final IBP/CROWN post-check. -/
def epsCheck : α := ExecUtils.defaultEpsCheck

def lossProg (width : Nat) :
    ∀ {β : Type}, [Context β] → [DecidableEq Shape] →
      TorchLean.Program β (paramShapes width ++ [xShape]) Shape.scalar :=
  Core.lossProgram width

/-- Convert a `Nat` JSON payload into a `UInt32`, or raise a user-facing error with context. -/
def expectU32 (ctx : String) (n : Nat) : IO UInt32 := do
  let limit : Nat := 4294967296 -- 2^32
  if _h : n < limit then
    pure (UInt32.ofNat n)
  else
    throw <| IO.userError s!"{ctx}: expected uint32 in [0,2^32), got {n}"

/-- Parse an array of float32 bit patterns (`UInt32`) encoded as nat/decimal-strings in JSON. -/
def parseBitsArray (j : Json) (ctx : String) : IO (Array UInt32) := do
  let arr ←
    match j with
    | .arr xs => pure xs
    | _ => throw <| IO.userError s!"{ctx}: expected JSON array"
  let ns ←
    match arr.mapM asNat? with
    | some ns => pure ns
    | none => throw <| IO.userError s!"{ctx}: expected nat/decimal-string array"
  ns.mapM (expectU32 ctx)

/-- Turn `UInt32` float32 bit patterns into executable float32 values (`IEEE32Exec`). -/
def bitsToα (bs : Array UInt32) : Array α :=
  bs.map IEEE32Exec.ofBits

/-- Build a length-`n` vector tensor from an array (with a length check). -/
def mkVec (n : Nat) (xs : Array α) : IO (Tensor α (.dim n .scalar)) := do
  if xs.size != n then
    throw <| IO.userError s!"expected length {n}, got {xs.size}"
  pure <| Tensor.dim (n := n) (s := .scalar) (fun i => Tensor.scalar xs[i.val]!)

/-- Build an `m×n` matrix tensor from a flat array (row-major, with a length check). -/
def mkMat (m n : Nat) (xs : Array α) : IO (Tensor α (.dim m (.dim n .scalar))) := do
  let expected := m * n
  if xs.size != expected then
    throw <| IO.userError s!"expected length {expected} (matrix {m}x{n}), got {xs.size}"
  pure <|
    Tensor.dim (n := m) (s := .dim n .scalar) (fun i =>
      Tensor.dim (n := n) (s := .scalar) (fun j =>
        let idx := i.val * n + j.val
        Tensor.scalar xs[idx]!))

/--
Load stage-1 parameters exported by PyTorch as *float32 bit patterns*.

We do this (instead of parsing JSON floats) so stage-2 runs under *bit-exact* float32 semantics
(`IEEE32Exec`) without decimal conversion error.
-/
def loadStage1Params (width : Nat) (path : String) : IO (TorchLean.TList α (paramShapes
  width)) := do
  let jsonStr ← IO.FS.readFile path
  let j ← match Json.parse jsonStr with
    | .ok j => pure j
    | .error e => throw <| IO.userError s!"Bad JSON: {e}"
  let top ← expectObj j "top-level"

  let wJ ← expectField top "width" "top-level"
  let some w := asNat? wJ | throw <| IO.userError "top-level.width must be nat/decimal-string"
  if w != width then
    throw <| IO.userError s!"width mismatch: file has {w}, workflow expects {width}"

  let wCBits ← parseBitsArray (← expectField top "wC" "top-level") "wC"
  let bCBits ← parseBitsArray (← expectField top "bC" "top-level") "bC"
  let w1Bits ← parseBitsArray (← expectField top "w1" "top-level") "w1"
  let b1Bits ← parseBitsArray (← expectField top "b1" "top-level") "b1"
  let w2Bits ← parseBitsArray (← expectField top "w2" "top-level") "w2"
  let b2Bits ← parseBitsArray (← expectField top "b2" "top-level") "b2"

  let wC ← mkMat uDim xDim (bitsToα wCBits)
  let bC ← mkVec uDim (bitsToα bCBits)
  let w1 ← mkMat width xDim (bitsToα w1Bits)
  let b1 ← mkVec width (bitsToα b1Bits)
  let w2 ← mkMat 1 width (bitsToα w2Bits)
  let b2 ← mkVec 1 (bitsToα b2Bits)

  pure <| .cons wC (.cons bC (.cons w1 (.cons b1 (.cons w2 (.cons b2 .nil)))))

abbrev lossΓ (width : Nat) : List Shape := paramShapes width ++ [xShape]

/--
One PGD step on the input `x` using a *compiled* loss graph.

We differentiate only w.r.t. `x` (parameters are treated as constants during the PGD inner loop),
then clamp to the training box `[-rad, rad]^2`.
-/
def pgdStepCompiled
    (width : Nat)
    (cLoss : _root_.Runtime.Autograd.Torch.CompiledScalar α (lossΓ width))
    (params : TorchLean.TList α (paramShapes width))
    (x : Tensor α xShape) : Tensor α xShape :=
  let args : TorchLean.TList α (lossΓ width) :=
    _root_.Runtime.Autograd.Torch.Proofs.Autograd.Algebra.TList.append (α := α)
      (ss₁ := paramShapes width) (ss₂ := [xShape]) params (.cons x .nil)
  let gAll : TorchLean.TList α (lossΓ width) :=
    _root_.Runtime.Autograd.Torch.CompiledScalar.backward (α := α) (Γ := lossΓ width) cLoss args
  let gx : TorchLean.TList α [xShape] :=
    (_root_.Runtime.Autograd.Torch.Proofs.Autograd.Algebra.TList.splitAppend (α := α)
      (ss₁ := paramShapes width) (ss₂ := [xShape]) gAll).2
  let .cons g .nil := gx
  let x' := Tensor.addSpec x (Tensor.scaleSpec g pgdStepSize)
  clampVec2 (-rad) rad x'

/--
Final post-check: compile the TorchLean loss to the shared verifier IR, then run IBP and CROWN
over a small box around the origin.

This is a *sanity check* that the training objective is provably small on that box; it is not
claimed to match α/β-CROWN tightness.
-/
def checkBox (width : Nat) (params : TorchLean.TList α (paramShapes width)) (eps : α :=
  epsCheck) : IO Unit := do
  IO.println "Stage 2 check: IBP + CROWN on the scalar loss over a small box"
  let compiled ←
    match NN.Verification.TorchLean.compileForward1
          (α := α) (paramShapes := paramShapes width) (inShape := xShape) (outShape := Shape.scalar)
          (lossProg width (β := α)) params with
    | .ok c => pure c
    | .error e => throw <| IO.userError e

  IO.println s!"compiled IR nodes: {compiled.graph.nodes.size}"

  let x0 : Tensor α xShape := Spec.zeros (α := α) xShape
  let radT : Tensor α xShape := Spec.fill (α := α) eps xShape
  let xB : FlatBox α :=
    { dim := xDim
      lo := Tensor.subSpec x0 radT
      hi := Tensor.addSpec x0 radT }

  let ps : ParamStore α :=
    { compiled.ps with inputBoxes := compiled.ps.inputBoxes.insert compiled.inputId xB }

  let ibp := runIBP (α := α) compiled.graph ps
  let some outB := ibp[compiled.outputId]! | throw <| IO.userError "IBP produced no output box"
  IO.println s!"[IBP] scalar loss box dim={outB.dim}"

  let ctx : AffineCtx := { inputId := compiled.inputId, inputDim := xDim }
  let crown := runCROWN (α := α) compiled.graph ps ctx ibp
  match crown[compiled.outputId]! with
  | none => IO.println "[CROWN] no affine bounds for scalar loss"
  | some outAff =>
      if hIn : outAff.inDim = xDim then
        let xBox : Box α (.dim outAff.inDim .scalar) :=
          { lo := Tensor.castVecDim (α := α) (n := xDim) (m := outAff.inDim) hIn.symm xB.lo
            hi := Tensor.castVecDim (α := α) (n := xDim) (m := outAff.inDim) hIn.symm xB.hi }
        let outLo := NN.MLTheory.CROWN.AffineVec.evalOnBox (α := α) outAff.loAff xBox
        let outHi := NN.MLTheory.CROWN.AffineVec.evalOnBox (α := α) outAff.hiAff xBox
        IO.println s!"[CROWN] loss lo = {pretty outLo.lo}"
        IO.println s!"[CROWN] loss hi = {pretty outHi.hi}"
      else
        IO.println s!"[CROWN] unexpected input dim {outAff.inDim} (expected {xDim})"

/--
Run the external PyTorch Stage-1 exporter (if needed) and return the JSON path.

This is the only place pipeline (ii) depends on Python. The trust boundary is still clean:
- Stage 1 provides an **initialization only** (untrusted),
- Stage 2 and the IBP/CROWN post-check run inside Lean under exact `IEEE32Exec` semantics.
-/
def ensureStage1Weights (width : Nat) (args : List String) : IO String := do
  let defaultPath : String := s!"_external/van_stage1_w{width}_bits.json"
  let weightsPath : String :=
    match args.find? (fun a => a.startsWith "--weights=") with
    | some a => (a.drop 10).toString
    | none =>
        match args.find? (fun a => !a.startsWith "--") with
        | some a => a
        | none => defaultPath

  let force : Bool := args.any (· = "--stage1")
  let steps : Nat :=
    match args.find? (fun a => a.startsWith "--stage1-steps=") with
    | some a => (a.drop 15).toString.toNat?.getD 10
    | none => 10

  let weightsExists := (← System.FilePath.pathExists (System.FilePath.mk weightsPath))
  if weightsExists && !force then
    return weightsPath

  IO.println s!"[stage1] running PyTorch exporter (width={width} steps={steps}) → {weightsPath}"
  let script : String :=
    "scripts/verification/two_stage/export_van_stage1_bits.py"
  let proc := (← IO.Process.spawn
    { cmd := "python3"
      args := #[script, "--width", toString width, "--steps", toString steps, "--out", weightsPath]
      stdout := .inherit
      stderr := .inherit })
  let code := (← proc.wait)
  if code != 0 then
    throw <| IO.userError s!"Stage-1 exporter failed with exit code {code}"
  let weightsExists' := (← System.FilePath.pathExists (System.FilePath.mk weightsPath))
  if !weightsExists' then
    throw <| IO.userError s!"Stage-1 exporter did not produce expected file: {weightsPath}"
  return weightsPath

/-- Main entrypoint for the hybrid pipeline (width is a parameter; CLI default is `defaultWidth`).
  -/
def run (width : Nat) (args : List String) : IO Unit := do
  let weightsPath ← ensureStage1Weights width args

  let longRun : Bool := args.any (· = "--long")
  let paperRun : Bool := args.any (· = "--paper")
  let stage2Rounds : Nat := (if longRun then 10 else if paperRun then 10 else 1)
  let pgdSteps : Nat := (if longRun then 20 else if paperRun then 10 else 1)
  let candidates : Nat :=
    match args.find? (fun a => a.startsWith "--candidates=") with
    | some a => (a.drop 13).toNat?.getD 1
    | none => 1

  IO.println "== TwoStage Hybrid workflow: Stage1=PyTorch (bits), Stage2=TorchLean (IEEE32Exec) =="
  IO.println
    (s!"weights={weightsPath} width={width} stage2Rounds={stage2Rounds} " ++
      s!"candidates={candidates} pgdSteps={pgdSteps}")

  let initParams ← loadStage1Params width weightsPath
  let mod ← _root_.Runtime.Autograd.TorchLean.Module.ScalarModule.create
    (α := α) (paramShapes := paramShapes width) (inputShapes := [xShape])
    (opts := { backend := .compiled })
    (initRequiresGrad := List.replicate (paramShapes width).length true)
    (loss := lossProg width (β := α))
    initParams
  let tr := mod.trainer

  let cLoss ← TorchLean.Autodiff.compileLoss
    (α := α) (paramShapes := paramShapes width) (inputShapes := [xShape]) (lossProg width)

  -- Stage 2: PGD on x to find violations, then train on them
  let mut seed : UInt64 := 1
  let mut foundViolations : Nat := 0
  for round in [0:stage2Rounds] do
    for _ci in [0:candidates] do
      let (seed', x0) := sampleVec2 seed rad
      seed := seed'
      let loss0 := _root_.Runtime.Autograd.Torch.scalarOf (←
        _root_.Runtime.Autograd.Torch.ScalarTrainer.forwardT tr (.cons x0 .nil))
      let params ← tr.getParams
      let mut x := x0
      for _k in [0:pgdSteps] do
        x := pgdStepCompiled width cLoss params x
      let xs : TorchLean.TList α [xShape] := .cons x .nil
      let lossFound := _root_.Runtime.Autograd.Torch.scalarOf (←
        _root_.Runtime.Autograd.Torch.ScalarTrainer.forwardT tr xs)
      if (0 : α) < lossFound then
        foundViolations := foundViolations + 1
      _root_.Runtime.Autograd.Torch.ScalarTrainer.stepT tr lr xs
      IO.println s!"[stage2] round {round}: lossBefore={loss0} lossAfterPGD={lossFound}"

  let params ← tr.getParams
  IO.println
    (s!"[stage2] PGD counterexample candidates={stage2Rounds * candidates} " ++
      s!"(positive-loss={foundViolations})")
  checkBox width params (eps := epsCheck)

/-- Default hidden width used by the hybrid workflow. -/
def defaultWidth : Nat := 500

/-- CLI entrypoint (hybrid pipeline, default width). -/
def main (args : List String) : IO Unit :=
  run (width := defaultWidth) args

end NN.MLTheory.CROWN.Lyapunov.TwoStage.PipelineII.Hybrid
