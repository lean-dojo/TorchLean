/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN
public import NN.API.Data
public import NN.API.Runtime
public import NN.API.Samples
public import NN.MLTheory.CROWN.Core
public import NN.MLTheory.CROWN.Graph
public import NN.Verification.TorchLean.Compile

/-!
# TorchLean MLP training and verification workflow

This is a native TorchLean workflow: the model is built, trained, compiled to verifier IR, and
checked without importing an external certificate.

Training:
- build a 2-layer ReLU MLP with a scalar MSE loss
- train it for a few SGD steps under the compiled TorchLean backend

Verification:
- compile the trained model's forward pass to the verifier IR
- run IBP and (basic) CROWN bounds on a small input box around one sample

Run:
  `lake exe verify -- torchlean-mlp-workflow --dtype float`
  `lake exe verify -- torchlean-mlp-workflow --dtype float32`
-/

@[expose] public section


namespace NN.Examples.Verification.TorchLean.TorchLeanMlpWorkflow

open _root_.Spec
open _root_.Spec.Tensor
open NN.API

open NN.MLTheory.CROWN.Graph
open NN.MLTheory.CROWN

/-- Input dimension for the workflow model. -/
def inDim : Nat := 2
/-- Hidden width for the workflow model. Kept as local config, not as part of the public name. -/
def hiddenDim : Nat := 100
/-- Output dimension for the workflow model. -/
def outDim : Nat := 1

/-- Input shape for the workflow model. -/
def xShape : Shape := NN.Tensor.Shape.Vec inDim
/-- Output shape for the workflow model. -/
def yShape : Shape := NN.Tensor.Shape.Vec outDim

/-- Batched inputs for training. -/
def XFloat : Spec.Tensor Float (.dim 3 xShape) :=
  tensor! [[1.0, 0.0],
           [0.0, 1.0],
           [1.0, 1.0]]

/-- Batched targets for training. -/
def YFloat : Spec.Tensor Float (.dim 3 yShape) :=
  API.Samples.regression2to1Float XFloat (API.Samples.affine2 2.0 (-3.0) 0.0)

/-- TorchLean model used for training and verification. -/
def mkModel : nn.M (nn.Sequential xShape yShape) :=
  nn.sequential![
    nn.linear inDim hiddenDim (pfx := Spec.Shape.scalar),
    nn.relu,
    nn.linear hiddenDim outDim (pfx := Spec.Shape.scalar)
  ]

def model : nn.Sequential xShape yShape :=
  nn.build 0 mkModel

/-- Training task: model plus regression loss. -/
def task :=
  train.regression model

/--
Run training and verification under a chosen scalar backend `α`.

The verifier compiles the trained model to IR and runs IBP/CROWN on a small input box.
-/
def runOnce {α : Type} [Semantics.Scalar α] [DecidableEq Shape] [ToString α]
    [Runtime.Scalar α] (opts : API.TorchLean.Options) : IO Unit := do
  let cast : Float → α := Runtime.ofFloat
  let dataset := _root_.NN.API.Data.supervisedDim0F (α := α) XFloat YFloat

  IO.println s!"== TorchLean MLP workflow ({inDim} → {hiddenDim} → {outDim}) =="
  IO.println s!"Training with backend={reprStr opts.backend}, device={if opts.useGpu then "cuda" else "cpu"}"
  let runner ← train.instantiateWithOptions task (α := α) opts
  let cfg := train.steps 10 (optim.sgd 0.05) (logEvery := 1)
  let _report ← train.fitDataset runner cfg dataset
  let avg ← train.meanLossDataset runner dataset
  IO.println s!"avg_loss(on samples)={avg}"

  IO.println "Checking bounds (IBP + CROWN) on a small input box"
  let params ← train.params runner

  let paramShapes := nn.paramShapes model
  let forward : TorchLean.Program α (paramShapes ++ [xShape]) yShape :=
    nn.program (model := model) (α := α)

  let compiled ←
    match NN.Verification.TorchLean.compileForward1
          (α := α) (paramShapes := paramShapes) (inShape := xShape) (outShape := yShape)
          forward params with
    | .ok c => pure c
    | .error e => throw <| IO.userError e

  IO.println s!"compiled IR nodes: {compiled.graph.nodes.size}"

  let x0F : Spec.Tensor Float xShape := _root_.Spec.get XFloat ⟨0, by decide⟩
  let x0 := NN.API.Common.castTensor cast x0F
  let eps : α := cast 0.05
  let rad : Tensor α xShape := Spec.fill (α := α) eps xShape
  let xB : FlatBox α :=
    { dim := inDim
      lo := Tensor.subSpec x0 rad
      hi := Tensor.addSpec x0 rad }

  let ps : ParamStore α :=
    { compiled.ps with inputBoxes := compiled.ps.inputBoxes.insert compiled.inputId xB }

  let ibp := runIBP (α := α) compiled.graph ps
  let some outB := ibp[compiled.outputId]! | throw <| IO.userError "IBP produced no output box"
  if hDim : outB.dim = 1 then
    let loY : Tensor α yShape := by
      simpa [yShape] using Tensor.castVecDim (α := α) (n := outB.dim) (m := 1) hDim outB.lo
    let hiY : Tensor α yShape := by
      simpa [yShape] using Tensor.castVecDim (α := α) (n := outB.dim) (m := 1) hDim outB.hi
    IO.println s!"[IBP] y lo = {pretty loY}"
    IO.println s!"[IBP] y hi = {pretty hiY}"
  else
    IO.println s!"[IBP] unexpected output dim {outB.dim} (expected 1)"

  let ctx : AffineCtx := { inputId := compiled.inputId, inputDim := inDim }
  let crown := runCROWN (α := α) compiled.graph ps ctx ibp
  match crown[compiled.outputId]! with
  | none =>
      IO.println "[CROWN] no affine bounds for output"
  | some outAff =>
      if hIn : outAff.inDim = inDim then
        if hOut : outAff.outDim = 1 then
          let xBox : Box α (.dim outAff.inDim .scalar) :=
            { lo := Tensor.castVecDim (α := α) (n := inDim) (m := outAff.inDim) hIn.symm xB.lo
              hi := Tensor.castVecDim (α := α) (n := inDim) (m := outAff.inDim) hIn.symm xB.hi }
          let outLo := NN.MLTheory.CROWN.AffineVec.evalOnBox (α := α) outAff.loAff xBox
          let outHi := NN.MLTheory.CROWN.AffineVec.evalOnBox (α := α) outAff.hiAff xBox
          IO.println s!"[CROWN] y lo = {pretty outLo.lo}"
          IO.println s!"[CROWN] y hi = {pretty outHi.hi}"
        else
          IO.println s!"[CROWN] unexpected output dim {outAff.outDim} (expected 1)"
      else
        IO.println s!"[CROWN] unexpected input dim {outAff.inDim} (expected {inDim})"

/--
CLI entry point for the native TorchLean MLP workflow.

This is wired into `lake exe verify -- torchlean-mlp-workflow`.
-/
def main (args : List String) : IO Unit := do
  let args :=
    if NN.API.CLI.hasFlagValue args "backend" then
      args
    else
      "--backend=compiled" :: args
  NN.API.TorchLean.Module.withRuntime args (fun {α} _ _ _ _ _ opts rest => do
    NN.API.Common.orThrow "torchlean-mlp-workflow" <| NN.API.CLI.requireNoArgs rest
    if opts.useGpu then
      throw <| IO.userError
        "torchlean-mlp-workflow: CUDA eager training is not used here; this workflow keeps trained parameters as Lean tensors so the verifier can compile and check them. Use the model-training examples for CUDA runtime training, or run this verifier workflow without --cuda."
    runOnce (α := α) opts)

end NN.Examples.Verification.TorchLean.TorchLeanMlpWorkflow
