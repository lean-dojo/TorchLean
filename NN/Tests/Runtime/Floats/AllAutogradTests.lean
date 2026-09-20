/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.TypedGraph.GraphM
public import NN.API.Neural.Execution
public import NN.API.Optim
public import NN.Runtime.Autograd.Train
public import NN.Spec.Models.Mlp
public import NN.Tests.Utils

/-!
# Consolidated Float Runtime Autograd Tests

This file collects runtime tests that exercise the *dynamic autograd tape*.
-/

@[expose] public section


/-! ## autograd_engine_test.lean -/

/-!
Regression tests for `Runtime.Autograd` dynamic tape.

We check that for a simple 2-layer MLP, the tape-based gradients match the existing
hand-derived `Examples.mlpBackward`.
-/

open Spec TorchLean
open TorchLean.Tensor
open Examples

namespace Tests
namespace Floats
namespace AutogradEngine

open Runtime.Autograd

abbrev inDim  := 2
abbrev hidDim := 3
abbrev outDim := 1

-- Small tag used for readable error messages.
abbrev tag : String := "autograd_engine_test"

-- The parameter-id record is shared with the `ℚ` transpose of this test; see `Tests.Utils`.
open Tests.Utils (ParamIds)

/-!
## Fixed inputs and parameters

We use a small deterministic 2-layer MLP so the gradients are stable.
-/
def hiddenWeight : Tensor Float [hidDim, inDim] :=
  (Tensor.from #[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).reshape [hidDim, inDim] (by dsimp; decide)

def hiddenBias : Tensor Float [hidDim] :=
  (Tensor.from #[0.1, 0.2, 0.3]).reshape [hidDim] (by dsimp; decide)

def outputWeight : Tensor Float [outDim, hidDim] :=
  (Tensor.from #[0.7, 0.8, 0.9]).reshape [outDim, hidDim] (by dsimp; decide)

def outputBias : Tensor Float [outDim] :=
  (Tensor.from #[0.4]).reshape [outDim] (by dsimp; decide)

def x : Tensor Float [inDim] :=
  (Tensor.from #[0.5, 0.8]).reshape [inDim] (by dsimp; decide)

def dLdy : Tensor Float [outDim] :=
  (Tensor.from #[1.0]).reshape [outDim] (by dsimp; decide)

def hiddenLayer : Spec.LinearSpec Float inDim hidDim :=
  { weights := hiddenWeight, bias := hiddenBias }
def outputLayer : Spec.LinearSpec Float hidDim outDim :=
  { weights := outputWeight, bias := outputBias }

def expected :=
  Examples.mlpBackward hiddenLayer outputLayer x dLdy

/-!
## Test: dynamic tape gradients vs. reference

We compare the autograd tape gradients against the hand-derived MLP backward pass.
-/
def checkMlpGrads :
  Runtime.Autograd.Result Bool := do
  let t0 : Tape Float := Tape.empty

  -- Build the graph in TapeM for readability.
  let m : TapeM Float _ := do
    let hiddenWeightId ← Train.TapeM.param hiddenWeight (name := some "hiddenWeight")
    let hiddenBiasId ← Train.TapeM.param hiddenBias (name := some "hiddenBias")
    let outputWeightId ← Train.TapeM.param outputWeight (name := some "outputWeight")
    let outputBiasId ← Train.TapeM.param outputBias (name := some "outputBias")
    let xId ← Train.TapeM.const x (name := some "x")

    -- Forward pass: linear -> relu -> linear
    let z1Id ← TapeM.linear (inDim:=inDim) (outDim:=hidDim) hiddenWeightId hiddenBiasId xId
    let a1Id ← TapeM.relu (s := [hidDim]) z1Id
    let yId ← TapeM.linear (inDim:=hidDim) (outDim:=outDim) outputWeightId outputBiasId a1Id

    let t ← TapeM.getTape
    let grads ← liftM (Tape.backward (t:=t) yId (Spec.SomeTensor.ofTensor dLdy))

    let ids : ParamIds :=
      { hiddenWeightId := hiddenWeightId, hiddenBiasId := hiddenBiasId,
        outputWeightId := outputWeightId, outputBiasId := outputBiasId }
    pure (ids, grads)

  let ((ids, grads), _) ← TapeM.run t0 m

  let (dW1_exp, db1_exp, dW2_exp, db2_exp, _dX_exp) := expected

  let dW1_dyn ← Train.requireGradTensor (tag := tag)
    (s := [hidDim, inDim]) grads ids.hiddenWeightId
  let db1_dyn ← Train.requireGradTensor (tag := tag)
    (s := [hidDim]) grads ids.hiddenBiasId
  let dW2_dyn ← Train.requireGradTensor (tag := tag)
    (s := [outDim, hidDim]) grads ids.outputWeightId
  let db2_dyn ← Train.requireGradTensor (tag := tag)
    (s := [outDim]) grads ids.outputBiasId

  -- Compare the coordinates directly; formatted tensors can hide small gradient errors.
  let close {s : Shape} (actual expected : Tensor Float s) : Bool :=
    let left := actual.to (Array Float)
    let right := expected.to (Array Float)
    left.size == right.size && (left.zip right).all fun (x, y) =>
      x.isFinite && y.isFinite && decide (Float.abs (x - y) ≤ 1e-12)
  pure (close dW1_dyn dW1_exp && close db1_dyn db1_exp &&
    close dW2_dyn dW2_exp && close db2_dyn db2_exp)

def run : IO Unit := do
  match checkMlpGrads with
  | .ok true => IO.println "autograd_engine_test (Float): OK"
  | .ok false => throw <| IO.userError "autograd_engine_test (Float): FAILED"
  | .error msg => throw <| IO.userError s!"autograd_engine_test (Float): {msg}"

end AutogradEngine
end Floats
end Tests

/-!
Dynamic-tape training with AdamW, linear warmup, and evaluation over a finite sample stream.
The checks below exercise the low-level trainer and evaluation APIs together.
-/

open Spec TorchLean
open TorchLean.Tensor

namespace Tests
namespace Floats
namespace AutogradLinearRegression

open Runtime.Autograd

-- A short tag used for readable error messages.
abbrev tag : String := "autograd_linear_regression_test"

abbrev inDim := 1
abbrev outDim := 1

-- One training example: (x, y)
abbrev Sample := Prod Float Float

-- A small dataset: y = 2x + 1
def dataset : Array Sample :=
  #[ (0.0, 1.0)
   , (1.0, 3.0)
   , (2.0, 5.0)
   , (3.0, 7.0)
   ]

-- Expose the examples through the reusable finite stream abstraction.
def testDataset : TorchLean.Data.SampleStream Sample :=
  TorchLean.Data.SampleStream.fromArray dataset

-- Model parameters (W, b) for y = W * x + b
structure Parameters where
  /-- Weight matrix of the scalar affine model. -/
  W : Tensor Float [outDim, inDim]
  /-- Bias of the scalar affine model. -/
  b : Tensor Float [outDim]

-- Initial parameters (not too close to the target).
def initialParameters : Parameters :=
  { W := Tensor.full [outDim, inDim] (0.5 : Float)
  , b := Tensor.full [outDim] (0.0 : Float)
  }

-- Optimizer config: ids are stable because we create W then b each step.
def learningRateScheduler : Train.LearningRateScheduler Float :=
  .linearWarmup (Optim.Scheduler.LinearWarmup.create
    (initialLearningRate := 0.2) (warmupSteps := 2) (startingLearningRate := 0.05))

def initialOptimizerState : Train.OptimizerState Float :=
  { algorithm := .adamw
  , parameterGroups :=
      #[{ parameterIds := #[0, 1]
        , learningRate := 0.2
        , weightDecay := 0.0
        , scheduler := some learningRateScheduler
        }]
  }

-- Training state for the trainer API.
structure TrainState where
  /-- Current model parameters. -/
  parameters : Parameters
  /-- Current optimizer state. -/
  optimizerState : Train.OptimizerState Float

def initialState : TrainState :=
  { parameters := initialParameters, optimizerState := initialOptimizerState }

-- Single-sample loss using the tape.
def sampleLoss (weightId biasId : Nat) (sample : Sample) :
  Runtime.Autograd.TapeM Float Nat := do
  let input : Tensor Float [inDim] := Tensor.full [inDim] sample.fst
  let target : Tensor Float [outDim] := Tensor.full [outDim] sample.snd
  let inputId ← Train.TapeM.const input (name := some "x")
  let targetId ← Train.TapeM.const target (name := some "y")
  let predictionId ←
    TapeM.linear (inDim := inDim) (outDim := outDim) weightId biasId inputId
  let lossId ← TapeM.mseLoss (s := [outDim]) predictionId targetId
  pure lossId

-- One optimizer-backed training step over a batch of samples.
def trainStep
  (state : TrainState) (batch : Array Sample) :
  Runtime.Autograd.Result (Train.StepResult TrainState Float) := do
  let initialTape : Tape Float := Tape.empty
  let computation : TapeM Float _ := do
    let weightId ← Train.TapeM.param state.parameters.W (name := some "W")
    let biasId ← Train.TapeM.param state.parameters.b (name := some "b")
    let lossId ← Train.TapeM.meanScalarOver (tag := tag) batch
      (fun sample => sampleLoss weightId biasId sample)
    let tape ← TapeM.getTape
    let loss ← liftM (Train.requireScalarValue (tag := tag) tape lossId)
    let gradients ← liftM (Tape.backwardScalar (t := tape) lossId)
    pure (weightId, biasId, loss, gradients)

  let ((weightId, biasId, loss, gradients), _) ← TapeM.run initialTape computation

  let parameterTable : Train.ParameterTable Float :=
    #[Train.Parameter.create weightId state.parameters.W (name := some "W")
    , Train.Parameter.create biasId state.parameters.b (name := some "b")
    ]

  let optimizerStep ←
    Train.Optimizer.step state.optimizerState parameterTable gradients

  let newW ← Train.ParameterTable.get (tag := tag)
    (s := [outDim, inDim]) optimizerStep.parameters weightId
  let newb ← Train.ParameterTable.get (tag := tag)
    (s := [outDim]) optimizerStep.parameters biasId

  let parameters : Parameters := { W := newW, b := newb }
  pure
    { nextState :=
        { parameters := parameters, optimizerState := optimizerStep.optimizerState }
      output := loss }

-- One trainer step over the fixed dataset.
def step (state : TrainState) :
  Runtime.Autograd.Result (Train.StepResult TrainState (Train.StepReport Float)) := do
  let result ← trainStep state dataset
  pure
    { nextState := result.nextState
      output := { loss := result.output, metrics := #[] } }

def trainer : Train.Trainer Runtime.Autograd.Result TrainState Float :=
  Train.Trainer.withoutLogging initialState step

/-- Evaluate one sample using constant parameters on a fresh tape. -/
def evalSample (parameters : Parameters) :
    Sample -> Runtime.Autograd.Result (Train.StepReport Float)
  | sample => do
      let t0 : Tape Float := Tape.empty
      let m : TapeM Float _ := do
        let wId ← Train.TapeM.const parameters.W (name := some "W")
        let bId ← Train.TapeM.const parameters.b (name := some "b")
        let lossId ← sampleLoss wId bId sample
        let t ← TapeM.getTape
        let lossVal ← liftM (Train.requireScalarValue (tag := tag) t lossId)
        pure lossVal
      let (lossVal, _) ← TapeM.run t0 m
      pure { loss := lossVal, metrics := #[] }

def evalDataset (parameters : Parameters) : Runtime.Autograd.Result (Train.StepReport Float) :=
  Train.Eval.evalDataset (tag := tag) testDataset (evalSample parameters)

def run : IO Unit := do
  let res :=
    (Train.Trainer.run (steps := 5) trainer) >>= fun result => do
      let evalReport ← evalDataset result.finalState.parameters
      pure (result.outputs, evalReport)
  match res with
  | .error msg => throw <| IO.userError s!"autograd_linear_regression_test (Float): {msg}"
  | .ok (reports, evalReport) =>
    for report in reports do
      Tests.Utils.assertFinite "linear regression training loss" report.loss
    Tests.Utils.assertFinite "linear regression evaluation loss" evalReport.loss
    IO.println "autograd_linear_regression_test (Float): OK"

end AutogradLinearRegression
end Floats
end Tests

/-!
Dynamic-tape MLP training through `Train.runSteps` and direct SGD parameter updates.
-/

open Spec TorchLean
open TorchLean.Tensor

namespace Tests
namespace Floats
namespace AutogradTrain

open Runtime.Autograd

abbrev inDim  := 2
abbrev hidDim := 3
abbrev outDim := 1

-- Small tag used for readable error messages.
abbrev tag : String := "autograd_train_test"

/-- The four parameter tensors updated together by each MLP training step. -/
structure Parameters where
  /-- Weight matrix for layer 1. -/
  hiddenWeight : Tensor Float [hidDim, inDim]
  /-- Bias for layer 1. -/
  hiddenBias : Tensor Float [hidDim]
  /-- Weight matrix for layer 2. -/
  outputWeight : Tensor Float [outDim, hidDim]
  /-- Bias for layer 2. -/
  outputBias : Tensor Float [outDim]

-- A fixed initialization so the test is deterministic.
def initialParameters : Parameters :=
  {
    hiddenWeight :=
      (Tensor.from #[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).reshape [hidDim, inDim] (by dsimp; decide),
    hiddenBias := (Tensor.from #[0.1, 0.2, 0.3]).reshape [hidDim] (by dsimp; decide),
    outputWeight := (Tensor.from #[0.7, 0.8, 0.9]).reshape [outDim, hidDim] (by dsimp; decide),
    outputBias := (Tensor.from #[0.4]).reshape [outDim] (by dsimp; decide)
  }

def x : Tensor Float [inDim] :=
  (Tensor.from #[0.5, 0.8]).reshape [inDim] (by dsimp; decide)

def yTarget : Tensor Float [outDim] :=
  (Tensor.from #[1.0]).reshape [outDim] (by dsimp; decide)

/-- Build the loss tape, obtain all four parameter gradients, and apply one SGD update. -/
def trainStep (parameters : Parameters) (learningRate : Float := 0.1) :
    Runtime.Autograd.Result (Train.StepResult Parameters Float) := do
  let t0 : Tape Float := Tape.empty
  let (t1, hiddenWeightId) :=
    Tape.leaf (t := t0) parameters.hiddenWeight (name := some "hiddenWeight")
  let (t2, hiddenBiasId) :=
    Tape.leaf (t := t1) parameters.hiddenBias (name := some "hiddenBias")
  let (t3, outputWeightId) :=
    Tape.leaf (t := t2) parameters.outputWeight (name := some "outputWeight")
  let (t4, outputBiasId) :=
    Tape.leaf (t := t3) parameters.outputBias (name := some "outputBias")
  let (t5, xId)  := Tape.leaf (t:=t4) x (name := some "x") (requiresGrad := false)
  let (t6, yId)  := Tape.leaf (t:=t5) yTarget (name := some "y") (requiresGrad := false)

  -- Forward pass: linear -> relu -> linear -> mse_loss
  let (t7, z1Id) ←
    Tape.linear (t:=t6) (inDim:=inDim) (outDim:=hidDim) hiddenWeightId hiddenBiasId xId
  let (t8, a1Id) ← Tape.relu (t := t7) (s := [hidDim]) z1Id
  let (t9, yhatId) ←
    Tape.linear (t:=t8) (inDim:=hidDim) (outDim:=outDim) outputWeightId outputBiasId a1Id
  let (t10, lossId) ← Tape.mseLoss (t := t9) (s := [outDim]) yhatId yId

  -- Read loss and backpropagate from the scalar loss node.
  let lossVal ← Train.requireScalarValue (tag := tag) t10 lossId
  let gradients ← Tape.backwardScalar (t := t10) lossId

  -- Extract typed gradients and apply SGD updates.
  let hiddenWeightGrad ← Train.requireGradTensor (tag := tag)
    (s := [hidDim, inDim]) gradients hiddenWeightId
  let hiddenBiasGrad ← Train.requireGradTensor (tag := tag)
    (s := [hidDim]) gradients hiddenBiasId
  let outputWeightGrad ← Train.requireGradTensor (tag := tag)
    (s := [outDim, hidDim]) gradients outputWeightId
  let outputBiasGrad ← Train.requireGradTensor (tag := tag)
    (s := [outDim]) gradients outputBiasId

  let hiddenWeightStep :=
    Optim.SGD.update { learningRate := learningRate } parameters.hiddenWeight hiddenWeightGrad
  let hiddenBiasStep :=
    Optim.SGD.update { learningRate := learningRate } parameters.hiddenBias hiddenBiasGrad
  let outputWeightStep :=
    Optim.SGD.update { learningRate := learningRate } parameters.outputWeight outputWeightGrad
  let outputBiasStep :=
    Optim.SGD.update { learningRate := learningRate } parameters.outputBias outputBiasGrad

  pure
    { nextState :=
        { hiddenWeight := hiddenWeightStep.parameters
          hiddenBias := hiddenBiasStep.parameters
          outputWeight := outputWeightStep.parameters
          outputBias := outputBiasStep.parameters }
      output := lossVal }

/-- Run the low-level step driver and retain each loss for the finite-value checks. -/
def train (epochs : Nat) (learningRate : Float := 0.1) :
  Runtime.Autograd.Result (Array Float) := do
  let result ← Train.runSteps (m := Runtime.Autograd.Result) epochs initialParameters
    (fun parameters => trainStep parameters learningRate)
  pure result.outputs

def run : IO Unit := do
  match train 6 0.1 with
  | .ok losses =>
    for loss in losses do
      Tests.Utils.assertFinite "MLP training loss" loss
    IO.println "autograd_train_test (Float): OK"
  | .error msg => throw <| IO.userError s!"autograd_train_test (Float): {msg}"

end AutogradTrain
end Floats
end Tests

/-!
CPU LayerNorm tape execution, including lookup and finite-value checks for all three gradients.
-/

open Spec TorchLean
open TorchLean.Tensor

namespace Tests
namespace Floats
namespace AutogradLayerNorm

open Runtime.Autograd

abbrev seqLen := 2
abbrev embedDim := 3

def x : Tensor Float [seqLen, embedDim] :=
  (Tensor.from #[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).reshape [seqLen, embedDim] (by dsimp; decide)

def gamma : Tensor Float [embedDim] :=
  (Tensor.from #[1.0, 0.9, 1.1]).reshape [embedDim] (by dsimp; decide)

def beta : Tensor Float [embedDim] :=
  (Tensor.from #[0.0, 0.1, -0.1]).reshape [embedDim] (by dsimp; decide)

def checkLayerNormGrads :
  Runtime.Autograd.Result
    (Float × Tensor Float [seqLen, embedDim] × Tensor Float [embedDim] ×
      Tensor Float [embedDim]) := do
  let t0 : Tape Float := Tape.empty
  let m : TapeM Float _ := do
    let xId ← Train.TapeM.param x (name := some "x")
    let gammaId ← Train.TapeM.param gamma (name := some "gamma")
    let betaId ← Train.TapeM.param beta (name := some "beta")
    let yId ← TapeM.layerNorm (seqLen := seqLen) (embedDim := embedDim) (by decide) (by decide) xId
      gammaId betaId
    let lossId ← TapeM.sum (s := [seqLen, embedDim]) yId
    let t ← TapeM.getTape
    let lossVal ← liftM (Train.requireScalarValue (tag := "layer_norm") t lossId)
    let grads ← liftM (Tape.backwardScalar (t := t) lossId)
    pure (xId, gammaId, betaId, lossVal, grads)

  let ((xId, gammaId, betaId, lossVal, grads), _) ← TapeM.run t0 m

  let dX ← Train.requireGradTensor (tag := "layer_norm")
    (s := [seqLen, embedDim]) grads xId
  let dGamma ← Train.requireGradTensor (tag := "layer_norm")
    (s := [embedDim]) grads gammaId
  let dBeta ← Train.requireGradTensor (tag := "layer_norm")
    (s := [embedDim]) grads betaId

  pure (lossVal, dX, dGamma, dBeta)

def run : IO Unit := do
  match checkLayerNormGrads with
  | .error msg => throw <| IO.userError s!"autograd_layernorm_test (Float): {msg}"
  | .ok (loss, dX, dGamma, dBeta) =>
    Tests.Utils.assertFinite "LayerNorm loss" loss
    for value in Tensor.to dX (Array Float) do
      Tests.Utils.assertFinite "LayerNorm input gradient" value
    for value in Tensor.to dGamma (Array Float) do
      Tests.Utils.assertFinite "LayerNorm scale gradient" value
    for value in Tensor.to dBeta (Array Float) do
      Tests.Utils.assertFinite "LayerNorm bias gradient" value
    IO.println "autograd_layernorm_test (Float): OK"

end AutogradLayerNorm
end Floats
end Tests

/-!
CPU convolution tape execution with two spatial axes and finite kernel/bias gradients.
-/

open Spec TorchLean
open TorchLean.Tensor

namespace Tests
namespace Floats
namespace AutogradConv

open Runtime.Autograd

abbrev inC := 1
abbrev outC := 1
abbrev kH := 2
abbrev kW := 2
abbrev stride := 1
abbrev padding := 0
abbrev inH := 2
abbrev inW := 2

theorem h1 : inC ≠ 0 := by decide
theorem h2 : kH ≠ 0 := by decide
theorem h3 : kW ≠ 0 := by decide

def outH : Nat := Spec.Shape.slidingWindowOutDim inH kH stride padding
def outW : Nat := Spec.Shape.slidingWindowOutDim inW kW stride padding

def kernel : Tensor Float [outC, inC, kH, kW] :=
  (Tensor.from #[0.2, -0.1, 0.3, 0.4]).reshape [outC, inC, kH, kW] (by dsimp; decide)

def bias : Tensor Float [outC] :=
  (Tensor.from #[0.05]).reshape [outC] (by dsimp; decide)

def input : Tensor Float [inC, inH, inW] :=
  (Tensor.from #[1.0, 2.0, 3.0, 4.0]).reshape [inC, inH, inW] (by dsimp; decide)

def checkConvGrads :
  Runtime.Autograd.Result
    (Tensor Float [outC, inC, kH, kW] × Tensor Float [outC]) := do
  let t0 : Tape Float := Tape.empty
  let m : TapeM Float _ := do
    let kId ← Train.TapeM.param kernel (name := some "kernel")
    let bId ← Train.TapeM.param bias (name := some "bias")
    let xId ← Train.TapeM.const input (name := some "input")
    let yId ← TapeM.conv (d := 2) (inC := inC) (outC := outC)
      (kernel := [kH, kW]) (stride := [stride, stride])
      (padding := [padding, padding]) (inSpatial := [inH, inW]) kId bId xId
    let lossId ← TapeM.sum (s := [outC, outH, outW]) yId
    let t ← TapeM.getTape
    let grads ← liftM (Tape.backwardScalar (t := t) lossId)
    pure (kId, bId, grads)

  let ((kId, bId, grads), _) ← TapeM.run t0 m
  let dK ← Train.requireGradTensor (tag := "conv")
    (s := [outC, inC, kH, kW]) grads kId
  let dB ← Train.requireGradTensor (tag := "conv")
    (s := [outC]) grads bId
  pure (dK, dB)

def run : IO Unit := do
  match checkConvGrads with
  | .error msg => throw <| IO.userError s!"autograd_conv_test (Float): {msg}"
  | .ok (dK, dB) =>
    for value in Tensor.to dK (Array Float) do
      Tests.Utils.assertFinite "convolution kernel gradient" value
    for value in Tensor.to dB (Array Float) do
      Tests.Utils.assertFinite "convolution bias gradient" value
    IO.println "autograd_conv_test (Float): OK"

end AutogradConv
end Floats
end Tests

/-! ## Typed graph log-softmax JVP -/

namespace Tests
namespace Floats
namespace TypedGraphLogSoftmaxJvp

open Spec TorchLean
open TorchLean.Tensor

/-- Check that typed graph log-softmax uses its JVP rather than its distinct reverse-mode VJP. -/
def run : IO Unit := do
  let vectorShape : Shape := [2]
  let build :
      Runtime.Autograd.TypedGraph.GraphM.M Float [vectorShape]
        (Runtime.Autograd.TypedGraph.GraphM.Var vectorShape) := do
    let x ← Runtime.Autograd.TypedGraph.GraphM.arg
      (α := Float) (Γ := [vectorShape]) 0 vectorShape
    Runtime.Autograd.TypedGraph.GraphM.logSoftmax 0 x
  let graph ←
    match Runtime.Autograd.Torch.lowerToTypedGraph
        (α := Float) (Γ := [vectorShape]) (τ := vectorShape) build with
    | .ok c => pure c
    | .error e => throw <| IO.userError s!"typed graph log-softmax JVP: lowering failed: {e}"
  let logits : Tensor Float vectorShape :=
    (Tensor.from #[0.0, Float.log 2.0]).reshape [2] (by dsimp; decide)
  let tangent : Tensor Float vectorShape := (Tensor.from #[1.0, 0.0]).reshape [2] (by dsimp; decide)
  let inputs : TorchLean.TensorPack Float [vectorShape] := .cons logits .nil
  let tangents : TorchLean.TensorPack Float [vectorShape] := .cons tangent .nil
  let got := Runtime.Autograd.Torch.TypedGraph.jvp graph inputs tangents
  let got0 := Tensor.getScalar got ⟨0, by decide⟩
  let got1 := Tensor.getScalar got ⟨1, by decide⟩
  unless Float.abs (got0 - 2.0 / 3.0) ≤ 1e-5 &&
      Float.abs (got1 - (-1.0 / 3.0)) ≤ 1e-5 do
    throw <| IO.userError s!"typed graph log-softmax JVP: got {pretty got}, expected [2/3, -1/3]"
  IO.println "typed_graph_log_softmax_jvp_test (Float): OK"

end TypedGraphLogSoftmaxJvp
end Floats
end Tests

/-! ## Typed graph output references -/

namespace Tests
namespace Floats
namespace TypedGraphOutputReference

open Spec TorchLean
open TorchLean.Tensor

/--
Typed graph lowering accepts an input as the output, even when no node is recorded or later nodes
are not selected as the result. Forward, JVP, and VJP must all follow that same output reference.
-/
def run : IO Unit := do
  let identityBuild :
      Runtime.Autograd.TypedGraph.GraphM.M Float [Shape.scalar]
        (Runtime.Autograd.TypedGraph.GraphM.Var Shape.scalar) := do
    Runtime.Autograd.TypedGraph.GraphM.arg
      (α := Float) (Γ := [Shape.scalar]) 0 Shape.scalar
  let identity ←
    match Runtime.Autograd.Torch.lowerToTypedGraph
        (α := Float) (Γ := [Shape.scalar]) (τ := Shape.scalar) identityBuild with
    | .ok graph => pure graph
    | .error e => throw <| IO.userError s!"typed graph identity lowering failed: {e}"
  unless identity.nodeShapes.isEmpty do
    throw <| IO.userError "typed graph identity lowering unexpectedly recorded a node"

  let earlierOutputBuild :
      Runtime.Autograd.TypedGraph.GraphM.M Float [Shape.scalar]
        (Runtime.Autograd.TypedGraph.GraphM.Var Shape.scalar) := do
    let x ← Runtime.Autograd.TypedGraph.GraphM.arg
      (α := Float) (Γ := [Shape.scalar]) 0 Shape.scalar
    let _unused ← Runtime.Autograd.TypedGraph.GraphM.add x x
    pure x
  let earlierOutput ←
    match Runtime.Autograd.Torch.lowerToTypedGraph
        (α := Float) (Γ := [Shape.scalar]) (τ := Shape.scalar) earlierOutputBuild with
    | .ok graph => pure graph
    | .error e => throw <| IO.userError s!"typed graph earlier-output lowering failed: {e}"
  unless earlierOutput.nodeShapes.length == 1 do
    throw <| IO.userError "typed graph earlier-output lowering lost the unused recorded node"

  let inputs : TorchLean.TensorPack Float [Shape.scalar] :=
    .cons (Tensor.scalar 3.0) .nil
  let tangents : TorchLean.TensorPack Float [Shape.scalar] :=
    .cons (Tensor.scalar 2.0) .nil
  let checkGraph (label : String)
      (graph : Runtime.Autograd.Torch.TypedGraph Float [Shape.scalar] Shape.scalar) : IO Unit := do
    let output := Tensor.item (Runtime.Autograd.Torch.TypedGraph.forward graph inputs)
    let tangent := Tensor.item (Runtime.Autograd.Torch.TypedGraph.jvp graph inputs tangents)
    let gradients := Runtime.Autograd.Torch.TypedGraph.vjpWithSeed
      graph inputs (Tensor.scalar 5.0)
    let gradient := match gradients with
      | .cons grad .nil => Tensor.item grad
    unless output == 3.0 && tangent == 2.0 && gradient == 5.0 do
      throw <| IO.userError
        s!"{label}: got forward={output}, jvp={tangent}, vjp={gradient}; expected 3, 2, 5"
  checkGraph "typed graph identity output" identity
  checkGraph "typed graph earlier output" earlierOutput

  let publicModel : TorchLean.nn.TypedGraphModel [] Shape.scalar Shape.scalar Float := identity
  let noParams : TorchLean.nn.State Float [] := TorchLean.nn.State.empty
  let publicOutput := Tensor.item <|
    TorchLean.nn.TypedGraphModel.forward publicModel noParams (Tensor.scalar 3.0)
  let publicTangent := Tensor.item <|
    TorchLean.nn.TypedGraphModel.jvp publicModel noParams noParams
      (Tensor.scalar 3.0) (Tensor.scalar 2.0)
  let (_, publicInputGradient) :=
    TorchLean.nn.TypedGraphModel.vjp publicModel noParams
      (Tensor.scalar 3.0) (Tensor.scalar 5.0)
  let publicInputGradient := Tensor.item publicInputGradient
  unless publicOutput == 3.0 && publicTangent == 2.0 &&
      publicInputGradient == 5.0 do
    throw <| IO.userError <|
      s!"typed graph public API: got forward={publicOutput}, jvp={publicTangent}, " ++
      s!"vjp={publicInputGradient}; expected 3, 2, 5"
  IO.println "typed_graph_output_reference_test (Float): OK"

end TypedGraphOutputReference
end Floats
end Tests

/-! ## Typed graph smooth-max parameter checks -/

namespace Tests
namespace Floats
namespace TypedGraphSmoothMaxDomain

open Spec TorchLean

/-- Typed `GraphM` rejects an undefined zero inverse temperature while building the graph. -/
def run : IO Unit := do
  let inputShape : Shape := [1, 1, 2]
  let spatial : TorchLean.Tensor Nat [2] := [1, 2]
  let kernel : TorchLean.Tensor Nat [2] := [1, 2]
  let stride : TorchLean.Tensor Nat [2] := [1, 1]
  let padding : TorchLean.Tensor Nat [2] := [0, 0]
  let outputShape : Shape :=
    Shape.ofList
      (1 :: Tensor.to (Spec.poolOutSpatialPad spatial kernel stride padding) (List Nat))
  let build :
      Runtime.Autograd.TypedGraph.GraphM.M Float [inputShape]
        (Runtime.Autograd.TypedGraph.GraphM.Var outputShape) := do
    let x ← Runtime.Autograd.TypedGraph.GraphM.arg
      (α := Float) (Γ := [inputShape]) 0 inputShape
    Runtime.Autograd.TypedGraph.GraphM.smoothMaxPool
      (d := 2) (C := 1) (inSpatial := spatial) (kernel := kernel)
      (stride := stride) (padding := padding) x 0.0
  match Runtime.Autograd.Torch.lowerToTypedGraph
      (α := Float) (Γ := [inputShape]) (τ := outputShape) build with
  | .error _ => IO.println "typed_graph_smooth_max_domain_test (Float): OK"
  | .ok _ => throw <| IO.userError "typed graph smooth-max accepted zero beta"

end TypedGraphSmoothMaxDomain
end Floats
end Tests

/-! ## Dense gradients for disconnected nodes -/

namespace Tests
namespace Floats
namespace DisconnectedDenseGradient

open Spec TorchLean
open TorchLean.Tensor
open Runtime.Autograd

/-- A disconnected reciprocal at zero must not turn an unrelated leaf gradient into `NaN`. -/
def run : IO Unit := do
  let t0 : Tape Float := Tape.empty
  let (t1, xId) := Tape.leaf (t := t0) (Tensor.scalar 0.0) (name := some "x")
  let (t2, outId) := Tape.leaf (t := t1) (Tensor.scalar 3.0) (name := some "output")
  let (t3, invId) ← okOrThrow <|
    Tape.inv (α := Float) (t := t2) (s := Shape.scalar) xId
  let grads ← okOrThrow <|
    Tape.backwardDenseAll (t := t3) outId (Spec.SomeTensor.ofTensor (Tensor.scalar 1.0))
  unless grads.size = t3.nodes.size do
    throw <| IO.userError "disconnected dense gradient: result length mismatch"
  let checkFiniteZero (label : String) (id : Nat) : IO Unit := do
    let grad ← match grads[id]? with
      | some grad => pure grad
      | none => throw <| IO.userError s!"{label}: gradient id out of bounds"
    if h : grad.shape = Shape.scalar then
      let value := Tensor.item (grad.cast h)
      unless value.isFinite && value == 0.0 do
        throw <| IO.userError s!"{label}: expected finite zero, got {value}"
    else
      throw <| IO.userError s!"{label}: expected a scalar gradient"
  checkFiniteZero "disconnected reciprocal input gradient" xId
  checkFiniteZero "disconnected reciprocal output gradient" invId
  IO.println "disconnected_dense_gradient_test (Float): OK"

end DisconnectedDenseGradient
end Floats
end Tests

/-! ## Optimizer and scheduler edge-case regressions -/

namespace Tests
namespace Floats
namespace OptimizerNumerics

open Runtime.Autograd

/-- Finite approximate equality used by the optimizer numerical regressions. -/
def close (x y : Float) (tol : Float := 1e-5) : Bool :=
  x.isFinite && y.isFinite && (x - y).abs ≤ tol

/-- Construct one scalar parameter for a compact optimizer test. -/
def scalarParameter (id : Nat) (value : Float) : Train.Parameter Float :=
  Train.Parameter.create id (Tensor.scalar value)

/-- Construct a one-entry scalar gradient map. -/
def scalarGradient (id : Nat) (value : Float) : Std.HashMap Nat (Spec.SomeTensor Float) :=
  ({} : Std.HashMap Nat (Spec.SomeTensor Float)).insert id
    (Spec.SomeTensor.ofTensor (Tensor.scalar value))

/-- Read a scalar parameter while preserving the runtime's error reporting. -/
def scalarParameterValue (tag : String) (parameters : Train.ParameterTable Float) (id : Nat) :
    Runtime.Autograd.Result Float := do
  let value ← Train.ParameterTable.get (tag := tag) (s := .scalar) parameters id
  pure value.item

/-- Adam bias correction advances only when that particular parameter receives a gradient. -/
def checkSparseAdamSteps : Runtime.Autograd.Result Bool := do
  let initialOptimizerState : Train.OptimizerState Float :=
    { algorithm := .adam
      parameterGroups :=
        #[{ parameterIds := #[0, 1]
            learningRate := 0.1
            beta1 := 0.9
            beta2 := 0.999
            epsilon := 1e-8 }] }
  let initialParameters : Train.ParameterTable Float :=
    #[scalarParameter 0 1.0, scalarParameter 1 1.0]
  let firstStep ← Train.Optimizer.step initialOptimizerState initialParameters
    (scalarGradient 0 1.0)
  let secondStep ← Train.Optimizer.step firstStep.optimizerState firstStep.parameters
    (scalarGradient 1 1.0)
  let firstParameter ← scalarParameterValue "sparse Adam" secondStep.parameters 0
  let secondParameter ← scalarParameterValue "sparse Adam" secondStep.parameters 1
  let restored := Train.OptimizerState.restore secondStep.optimizerState.snapshot
  pure <|
    close firstParameter 0.9 && close secondParameter 0.9 &&
      secondStep.optimizerState.stepCount == 2 &&
      secondStep.optimizerState.parameterStepCount? 0 == some 1 &&
      secondStep.optimizerState.parameterStepCount? 1 == some 1 &&
      restored.parameterStepCount? 0 == some 1 &&
      restored.parameterStepCount? 1 == some 1

/-- Momentum dampening does not scale the first buffer, matching the standard SGD convention. -/
def checkMomentumInitialization : Runtime.Autograd.Result Bool := do
  let initialOptimizerState : Train.OptimizerState Float :=
    { algorithm := .momentum
      parameterGroups :=
        #[{ parameterIds := #[0]
            learningRate := 0.1
            momentum := 0.9
            dampening := 0.5 }] }
  let initialParameters : Train.ParameterTable Float := #[scalarParameter 0 1.0]
  let firstStep ← Train.Optimizer.step initialOptimizerState initialParameters
    (scalarGradient 0 2.0)
  let first ← scalarParameterValue "momentum initialization" firstStep.parameters 0
  let secondStep ← Train.Optimizer.step firstStep.optimizerState firstStep.parameters
    (scalarGradient 0 2.0)
  let second ← scalarParameterValue "momentum initialization" secondStep.parameters 0
  pure (close first 0.8 && close second 0.52)

/-- Adadelta's update accumulator stores the unscaled update, independently of the learning rate. -/
def checkAdadeltaAccumulator : Bool :=
  let parameters : Tensor Float .scalar := Tensor.scalar 10.0
  let gradients : Tensor Float .scalar := Tensor.scalar 2.0
  let state := Optim.Adadelta.init 0.5 0.0 1.0 parameters
  let result := Optim.Adadelta.update state parameters gradients
  close result.optimizerState.squaredUpdateAverage.item 0.8 &&
    close result.parameters.item (10.0 - 1.0 / Float.sqrt 5.0)

/-- Warmup-cosine decay remains at zero after its finite schedule has ended. -/
def checkWarmupCosineStops : Bool :=
  let base : Optim.Scheduler.WarmupCosine Float :=
    Optim.Scheduler.WarmupCosine.create 1.0 2 10
  let atEnd := { base with currentStep := 10 }
  let afterEnd := { base with currentStep := 20 }
  atEnd.current == 0.0 && afterEnd.current == 0.0

/-- Public optimizer configurations reject domains that make their updates undefined. -/
def checkPublicOptimizerValidation : Bool :=
  let rejected : Except String Unit -> Bool
    | .error _ => true
    | .ok () => false
  (TorchLean.optim.adam { learningRate := 1e-3 }).validate.isOk &&
    rejected (TorchLean.optim.adam { learningRate := 1e-3, beta1 := 1.0 }).validate &&
    rejected (TorchLean.optim.adamW { learningRate := 1e-3, weightDecay := -0.1 }).validate &&
    rejected (TorchLean.optim.rmsProp { learningRate := 1e-3, epsilon := 0.0 }).validate &&
    rejected (TorchLean.optim.sgd
      { learningRate := 0.1, momentum := Float.ofBits 0x7ff8000000000000 }).validate

/-- Run the optimizer and scheduler edge-case regressions. -/
def run : IO Unit := do
  match checkSparseAdamSteps with
  | .error msg => throw <| IO.userError s!"optimizer numerics (sparse Adam): {msg}"
  | .ok false => throw <| IO.userError "optimizer numerics (sparse Adam): FAILED"
  | .ok true => pure ()
  match checkMomentumInitialization with
  | .error msg => throw <| IO.userError s!"optimizer numerics (momentum): {msg}"
  | .ok false => throw <| IO.userError "optimizer numerics (momentum): FAILED"
  | .ok true => pure ()
  unless checkAdadeltaAccumulator do
    throw <| IO.userError "optimizer numerics (Adadelta accumulator): FAILED"
  unless checkWarmupCosineStops do
    throw <| IO.userError "optimizer numerics (warmup cosine): FAILED"
  unless checkPublicOptimizerValidation do
    throw <| IO.userError "optimizer configuration validation: FAILED"
  IO.println "optimizer and scheduler edge cases (Float): OK"

end OptimizerNumerics
end Floats
end Tests

namespace Tests
namespace Floats

def runAllAutogradTests : IO Unit := do
  IO.println "=== Runtime autograd test suite (Float) ==="
  AutogradEngine.run
  AutogradLinearRegression.run
  AutogradTrain.run
  AutogradLayerNorm.run
  AutogradConv.run
  TypedGraphLogSoftmaxJvp.run
  TypedGraphOutputReference.run
  TypedGraphSmoothMaxDomain.run
  DisconnectedDenseGradient.run
  OptimizerNumerics.run
  IO.println "=== Autograd test suite completed ==="

end Floats
end Tests
