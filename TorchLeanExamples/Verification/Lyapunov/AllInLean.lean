/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLeanExamples.Verification.Lyapunov.LossAnalysis
public import TorchLean.NN.Module.Instantiation

/-!
# Pipeline (iii): All-in-Lean TwoStage refinement + IBP/CROWN bounds

The pipeline of **Figure 7 (iii)** in the TorchLean paper (`arXiv:2602.22631`) runs everything
*inside Lean*:
- Stage 1 samples training points in a box and trains the parameters (SGD) under exact
  `ExecFloat.Binary 8 23`.
- Stage 2 runs, in each round, a small PGD loop on the input `x` to find counterexample
  candidates, then trains on them (CEGIS style).
- At the end the same TorchLean loss program is lowered to the shared verifier IR, and in-repo
  IBP/CROWN bound propagation reports loss bounds on a small box around the origin.

The bounds come from the in-repo IBP/CROWN engine, which leaves out many optimizations of external
α/β-CROWN systems. The complete computation, including its float32 semantics, runs inside Lean
and requires no external verifier.

Run:
`scripts/lake.sh exe verify -- twostage-torchlean-cegis-van`

Without flags each stage takes a single step. `--long` gives 20 stage-1 steps and 10 rounds of 20
PGD steps, `--paper` gives 10 steps and 10 rounds of 10, and `--two` gives 2 stage-1 steps.
-/

@[expose] public section

open FloatLib.Floats (ExecFloat Binary32)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)


open TorchLean.Tensor

namespace TorchLean.Examples.Lyapunov.AllInLean



local notation "Scalar" => FloatLib.Floats.Binary32


/-- Initial parameters: Xavier-uniform weights and zero biases, with a fixed seed per tensor.

The seeds are hard-coded so a pipeline run is reproducible from the source alone, which matters
because the verification result at the end is reported for this particular network. -/
def initialState (width : Nat) : TensorPack Float (paramShapes width) :=
  let wC : Tensor Float [uDim, xDim] :=
    Init.xavierUniform (outputWidth := uDim) (inputWidth := xDim) (seed := 0)
  let bC : Tensor Float [uDim] :=
    Init.tensor (shape := [uDim]) (scheme := .zeros) (seed := 1)
  let w1 : Tensor Float [width, xDim] :=
    Init.xavierUniform (outputWidth := width) (inputWidth := xDim) (seed := 2)
  let b1 : Tensor Float [width] :=
    Init.tensor (shape := [width]) (scheme := .zeros) (seed := 3)
  let w2 : Tensor Float [1, width] :=
    Init.xavierUniform (outputWidth := 1) (inputWidth := width) (seed := 4)
  let b2 : Tensor Float [1] :=
    Init.tensor (shape := [1]) (scheme := .zeros) (seed := 5)
  Pack.nil
    |>.snoc wC
    |>.snoc bC
    |>.snoc w1
    |>.snoc b1
    |>.snoc w2
    |>.snoc b2

/-- Packages the initial state and the loss program into the objective the optimizer loops over. -/
def objective (width : Nat) :
    ObjectiveDef Unit (paramShapes width) [xShape]
  :=
  { initState := initialState width
    loss := lossProgram width }

/-- Main entry point of the all-in-Lean pipeline at width `width` (the CLI uses `defaultWidth`). -/
def run (width : Nat) (args : List String) : IO Unit := do
  let longRun : Bool := args.any (· = "--long")
  let twoRun : Bool := args.any (· = "--two")
  let paperRun : Bool := args.any (· = "--paper")
  let stage1Steps : Nat := (if longRun then 20 else if paperRun then 10 else if twoRun then 2 else
    1)
  let stage2Rounds : Nat := (if longRun then 10 else if paperRun then 10 else 1)
  let pgdSteps : Nat := (if longRun then 20 else if paperRun then 10 else 1)

  IO.println "== TwoStage TorchLean CEGIS workflow (Binary32) =="
  IO.println
    s!"width={width} stage1Steps={stage1Steps} stage2Rounds={stage2Rounds} pgdSteps={pgdSteps}"

  let mod ← ObjectiveDef.instantiate (α := Scalar)
    (objective width) (fun x => (ExecFloat.Binary.ofModel (Model.cast FloatFormat.binary64
      FloatFormat.binary32 (ExecFloat.Binary.toModel
        (ExecFloat.Binary.ofFloat x : FloatLib.Floats.Binary64))) :
      Binary32))
  let tr := mod.trainer
  let sgd ← Objective.bindOptimizer mod
    (Optimizer.sgd defaultLr)
  let cLoss ← Autodiff.lowerToTypedGraph
    (α := Scalar) (paramShapes := Lyapunov.paramShapes width) (inputShapes := [xShape])
      (lossProgram width)

  -- Stage 1: initialization pass on random x in [-rad, rad]^2
  let mut seed : UInt64 := 1
  for i in [0:stage1Steps] do
    let (seed', x) := sampleStateTensor seed defaultRad
    seed := seed'
    let xs := Pack.singleton x
    let currentLoss := (←
      ScalarTrainer.loss tr xs .nil).item
    sgd.step xs .nil
    if i % 5 = 0 then
      IO.println s!"[stage1] step {i}: loss={currentLoss}"

  -- Stage 2: PGD on x to find violations, then train on them
  for round in [0:stage2Rounds] do
    let (seed', x0) := sampleStateTensor seed defaultRad
    seed := seed'
    let params ← tr.getState
    let mut x := x0
    for _k in [0:pgdSteps] do
      x := projectedGradientAscentStep
        width cLoss params x defaultPgdStepSize defaultRad
    let xs := Pack.singleton x
    let lossFound := (←
      ScalarTrainer.loss tr xs .nil).item
    sgd.step xs .nil
    IO.println s!"[stage2] round {round}: loss={lossFound}"

  let params ← tr.getState
  reportLossBox width params defaultEpsCheck

/-- Default hidden width used by the Pipeline III all-in-Lean workflow. -/
def defaultWidth : Nat := 100

/-- CLI entrypoint (all-in-Lean pipeline, default width). -/
def main (args : List String) : IO Unit :=
  run (width := defaultWidth) args

end TorchLean.Examples.Lyapunov.AllInLean
