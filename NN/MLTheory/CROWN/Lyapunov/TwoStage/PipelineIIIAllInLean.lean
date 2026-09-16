/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec
public import NN.API.Neural.State
public import NN.API.Sample
public import NN.MLTheory.CROWN.Core
public import NN.MLTheory.CROWN.Graph
public import NN.MLTheory.CROWN.Lyapunov.TwoStage.Core
public import NN.MLTheory.CROWN.Lyapunov.TwoStage.LossAnalysis
public import NN.MLTheory.CROWN.Lyapunov.TwoStage.Execution
public import NN.Runtime.Autograd.Model.Autodiff
public import NN.Runtime.Autograd.Model.Program
public import NN.Runtime.Autograd.Model.Module
public import NN.Verification.Builtin.Lowering

/-!
# Pipeline (iii): All-in-Lean TwoStage refinement + IBP/CROWN check

This file corresponds to **Figure 7 (iii)** in the TorchLean paper (`arXiv:2602.22631`).

Everything runs *inside Lean*:
- Stage 1: sample training points in a box and train parameters (SGD) under exact `ExecFloat.Binary
8 23`.
- Stage 2: for each round, run a small PGD loop on the input `x` to find “counterexample-ish”
  points, then train on them (CEGIS flavor).
- Final: lower the same TorchLean loss program to the shared verifier IR and run in-repo IBP/CROWN
  bound propagation to check the loss on a small box around the origin.

The workflow uses the in-repo IBP/CROWN engine rather than reproducing every optimization from
external α/β-CROWN systems. The complete computation, including its float32 semantics, runs inside
Lean and requires no external verifier.

Run:
`lake exe verify -- twostage-torchlean-cegis-van`
-/

@[expose] public section

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)


open Spec TorchLean
open TorchLean TorchLean.Tensor

namespace NN.MLTheory.CROWN.Lyapunov.TwoStage.PipelineIII.AllInLean

open _root_.Runtime
open _root_.Runtime.Autograd
open NN.MLTheory.CROWN.Graph
open NN.MLTheory.CROWN

open NN.MLTheory.CROWN.Lyapunov.TwoStage.Core
open NN.MLTheory.CROWN.Lyapunov.TwoStage.Execution

local notation "Scalar" => (ExecFloat.Binary 8 23)


/-- Learning rate for the stage-1 and stage-2 SGD loops. -/
def lr : Scalar := Execution.defaultLr

/-- PGD step size when searching for counterexample-ish inputs. -/
def pgdStepSize : Scalar := Execution.defaultPgdStepSize

/-- Radius of the training box `[-rad, rad]^2` (also used for clamping PGD iterates). -/
def rad : Scalar := Execution.defaultRad

/-- Half-width of the small box around the origin used for the final IBP/CROWN post-check. -/
def epsCheck : Scalar := Execution.defaultEpsCheck

/-- Initial parameters: Xavier-uniform weights and zero biases, with fixed seeds per tensor.

The seeds are hard-coded so a pipeline run is reproducible from the source alone; that matters
because the verification result at the end is reported for this particular network. -/
def initialState (width : Nat) : TorchLean.nn.State Float (Core.paramShapes width) :=
  let wC : Tensor Float [Core.uDim, Core.xDim] :=
    Runtime.Autograd.Torch.Init.xavierUniform Core.uDim Core.xDim (seed := 0)
  let bC : Tensor Float [Core.uDim] :=
    Runtime.Autograd.Torch.Init.tensor (s := [Core.uDim]) (sch := .zeros)
      (seed := 1)
  let w1 : Tensor Float [width, Core.xDim] :=
    Runtime.Autograd.Torch.Init.xavierUniform width Core.xDim (seed := 2)
  let b1 : Tensor Float [width] :=
    Runtime.Autograd.Torch.Init.tensor (s := [width]) (sch := .zeros) (seed := 3)
  let w2 : Tensor Float [1, width] :=
    Runtime.Autograd.Torch.Init.xavierUniform 1 width (seed := 4)
  let b2 : Tensor Float [1] :=
    Runtime.Autograd.Torch.Init.tensor (s := [1]) (sch := .zeros) (seed := 5)
  TorchLean.nn.State.empty
    |>.push wC
    |>.push bC
    |>.push w1
    |>.push b1
    |>.push w2
    |>.push b2

/-- Packages the initial state and the loss program into the objective the optimizer loops over. -/
def objectiveDef (width : Nat) :
    Runtime.Autograd.Model.Module.ObjectiveDef Unit (Core.paramShapes width) [Core.xShape]
  :=
  { initState := TorchLean.nn.State.Internal.toTensorPack (initialState width)
    loss := Core.lossProgram width }

/-- Main entrypoint for the all-in-Lean pipeline (width is a parameter; CLI default is
  `defaultWidth`). -/
def run (width : Nat) (args : List String) : IO Unit := do
  let longRun : Bool := args.any (· = "--long")
  let twoRun : Bool := args.any (· = "--two")
  let paperRun : Bool := args.any (· = "--paper")
  let stage1Steps : Nat := (if longRun then 20 else if paperRun then 10 else if twoRun then 2 else
    1)
  let stage2Rounds : Nat := (if longRun then 10 else if paperRun then 10 else 1)
  let pgdSteps : Nat := (if longRun then 20 else if paperRun then 10 else 1)

  IO.println "== TwoStage TorchLean CEGIS workflow (IEEE32Exec) =="
  IO.println
    s!"width={width} stage1Steps={stage1Steps} stage2Rounds={stage2Rounds} pgdSteps={pgdSteps}"

  let mod ← Runtime.Autograd.Model.Module.ObjectiveDef.instantiate (α := Scalar)
    (objectiveDef width) (fun x => (ExecFloat.Binary.ofModel (Model.cast FloatFormat.binary64
      FloatFormat.binary32 (ExecFloat.Binary.toModel (ExecFloat.Binary.ofFloat x))) :
      ExecFloat.Binary 8 23)) .typedGraph
  let tr := mod.trainer
  let cLoss ← Runtime.Autograd.Model.Autodiff.lowerScalarToTypedGraph
    (α := Scalar) (paramShapes := Core.paramShapes width) (inputShapes := [Core.xShape])
      (Core.lossProgram width)

  -- Stage 1: initialization pass on random x in [-rad, rad]^2
  let mut seed : UInt64 := 1
  for i in [0:stage1Steps] do
    let (seed', x) := sampleStateTensor seed rad
    seed := seed'
    let xs := TorchLean.TensorPack.singleton x
    let currentLoss := (←
      Runtime.Autograd.Torch.ScalarTrainer.runLoss tr xs .nil).item
    Runtime.Autograd.Torch.ScalarTrainer.runStep tr lr xs .nil
    if i % 5 = 0 then
      IO.println s!"[stage1] step {i}: loss={currentLoss}"

  -- Stage 2: PGD on x to find violations, then train on them
  for round in [0:stage2Rounds] do
    let (seed', x0) := sampleStateTensor seed rad
    seed := seed'
    let params := TorchLean.nn.State.Internal.fromTensorPack (← tr.getState)
    let mut x := x0
    for _k in [0:pgdSteps] do
      x := LossAnalysis.projectedGradientStep
        width cLoss params x pgdStepSize rad
    let xs := TorchLean.TensorPack.singleton x
    let lossFound := (←
      Runtime.Autograd.Torch.ScalarTrainer.runLoss tr xs .nil).item
    Runtime.Autograd.Torch.ScalarTrainer.runStep tr lr xs .nil
    IO.println s!"[stage2] round {round}: loss={lossFound}"

  let params := TorchLean.nn.State.Internal.fromTensorPack (← tr.getState)
  LossAnalysis.checkLossBox width params epsCheck

/-- Default hidden width used by the Pipeline III all-in-Lean workflow. -/
def defaultWidth : Nat := 100

/-- CLI entrypoint (all-in-Lean pipeline, default width). -/
def main (args : List String) : IO Unit :=
  run (width := defaultWidth) args

end NN.MLTheory.CROWN.Lyapunov.TwoStage.PipelineIII.AllInLean
