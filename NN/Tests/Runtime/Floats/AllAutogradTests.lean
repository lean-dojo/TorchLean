/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Utils
public import NN.Spec.Core.Tensor
public import NN.Spec.Core.Utils
public import NN.Spec.Models.Mlp
public import NN.Entrypoint.Tensor

/-!
# Consolidated Float Runtime Autograd Tests

This file collects runtime tests that exercise the *dynamic autograd tape*.
-/

@[expose] public section


/-! ## autograd_engine_test.lean -/

/-!
Regression tests for `Runtime.Autograd` dynamic tape.

We check that for a simple 2-layer MLP, the tape-based gradients match the existing
hand-derived `Examples.mlp_backward`.
-/

open _root_.Spec
open _root_.Spec.Tensor
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

-- Parameter node ids we want to read gradients for.
structure ParamIds where
  /-- w 1 Id. -/
  w1Id : Nat
  /-- b 1 Id. -/
  b1Id : Nat
  /-- w 2 Id. -/
  w2Id : Nat
  /-- b 2 Id. -/
  b2Id : Nat

/-!
## Fixed inputs and parameters

We use a small deterministic 2-layer MLP so the gradients are stable.
-/
def W1 : Tensor Float (.dim hidDim (.dim inDim .scalar)) :=
  tensorND! [hidDim, inDim] [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

def b1 : Tensor Float (.dim hidDim .scalar) :=
  tensorND! [hidDim] [0.1, 0.2, 0.3]

def W2 : Tensor Float (.dim outDim (.dim hidDim .scalar)) :=
  tensorND! [outDim, hidDim] [0.7, 0.8, 0.9]

def b2 : Tensor Float (.dim outDim .scalar) :=
  tensorND! [outDim] [0.4]

def x : Tensor Float (.dim inDim .scalar) :=
  tensorND! [inDim] [0.5, 0.8]

def dLdy : Tensor Float (.dim outDim .scalar) :=
  tensorND! [outDim] [1.0]

def layer1 : Spec.LinearSpec Float inDim hidDim := { weights := W1, bias := b1 }
def layer2 : Spec.LinearSpec Float hidDim outDim := { weights := W2, bias := b2 }

def expected :=
  Examples.mlpBackward layer1 layer2 x dLdy

/-!
## Test: dynamic tape gradients vs. reference

We compare the autograd tape gradients against the hand-derived MLP backward pass.
-/
def checkMlpGrads :
  Runtime.Autograd.Result Bool := do
  let t0 : Tape Float := Tape.empty

  -- Build the graph in TapeM for readability.
  let m : TapeM Float _ := do
    let w1Id ← Train.TapeM.param W1 (name := some "W1")
    let b1Id ← Train.TapeM.param b1 (name := some "b1")
    let w2Id ← Train.TapeM.param W2 (name := some "W2")
    let b2Id ← Train.TapeM.param b2 (name := some "b2")
    let xId ← Train.TapeM.const x (name := some "x")

    -- Forward pass: linear -> relu -> linear
    let z1Id ← TapeM.linear (inDim:=inDim) (outDim:=hidDim) w1Id b1Id xId
    let a1Id ← TapeM.relu (s:=.dim hidDim .scalar) z1Id
    let yId ← TapeM.linear (inDim:=hidDim) (outDim:=outDim) w2Id b2Id a1Id

    let t ← TapeM.getTape
    let grads ← liftM (Tape.backward (t:=t) yId (Runtime.Autograd.AnyTensor.mk dLdy))

    let ids : ParamIds := { w1Id := w1Id, b1Id := b1Id, w2Id := w2Id, b2Id := b2Id }
    pure (ids, grads)

  let ((ids, grads), _) ← TapeM.run t0 m

  let (dW1_exp, db1_exp, dW2_exp, db2_exp, _dX_exp) := expected

  let dW1_dyn ← Train.requireGradTensor (tag := tag)
    (s:=.dim hidDim (.dim inDim .scalar)) grads ids.w1Id
  let db1_dyn ← Train.requireGradTensor (tag := tag)
    (s:=.dim hidDim .scalar) grads ids.b1Id
  let dW2_dyn ← Train.requireGradTensor (tag := tag)
    (s:=.dim outDim (.dim hidDim .scalar)) grads ids.w2Id
  let db2_dyn ← Train.requireGradTensor (tag := tag)
    (s:=.dim outDim .scalar) grads ids.b2Id

  let ok1 := decide (pretty dW1_dyn = pretty dW1_exp)
  let ok2 := decide (pretty db1_dyn = pretty db1_exp)
  let ok3 := decide (pretty dW2_dyn = pretty dW2_exp)
  let ok4 := decide (pretty db2_dyn = pretty db2_exp)
  pure (ok1 && ok2 && ok3 && ok4)

def run : IO Unit := do
  match checkMlpGrads with
  | .ok true => IO.println "autograd_engine_test (Float): OK"
  | .ok false => throw <| IO.userError "autograd_engine_test (Float): FAILED"
  | .error msg => throw <| IO.userError s!"autograd_engine_test (Float): {msg}"

end AutogradEngine
end Floats
end Tests

/-! ## autograd_linear_regression_test.lean -/

/-!
# Autograd linear regression (Float)

This file is a small, end-to-end training regression test for the *dynamic autograd tape*.

We fit a 1D linear model to a small dataset:

  `y = 2x + 1`

using SGD on the mean-squared-error (MSE) loss.

Key things to notice when reading the code:
* The forward pass is written in the `TapeM` style, so the tape is threaded implicitly.
* Dataset inputs/targets are created with `requires_grad = false` (they are constants).
* After the forward pass, we call `backwardScalar` to get a gradient map `id -> grad`.
* Parameters are updated with a simple tensor-level SGD update rule.
-/

open _root_.Spec
open _root_.Spec.Tensor

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
def dataset : List Sample :=
  [ (0.0, 1.0)
  , (1.0, 3.0)
  , (2.0, 5.0)
  , (3.0, 7.0)
  ]

-- Wrap into the reusable Dataset abstraction.
def testDataset : Train.Dataset Sample :=
  Train.Dataset.ofList dataset

-- Model parameters (W, b) for y = W * x + b
structure Params where
  /-- W. -/
  W : Tensor Float (.dim outDim (.dim inDim .scalar))
  /-- b. -/
  b : Tensor Float (.dim outDim .scalar)

-- Initial parameters (not too close to the target).
def initParams : Params :=
  { W := fill (0.5 : Float) (.dim outDim (.dim inDim .scalar))
  , b := fill (0.0 : Float) (.dim outDim .scalar)
  }

-- Optimizer config: ids are stable because we create W then b each step.
def lrScheduler : Train.LRScheduler Float :=
  .linearWarmup (Optim.createLinearWarmupScheduler (initialLr := 0.2) (warmupSteps := 2)
    (startLr := 0.05))

def initOptim : Train.OptimizerState Float :=
  { kind := .adamw
  , groups :=
      [ { params := [0, 1]
        , lr := 0.2
        , weight_decay := 0.0
        , scheduler := some lrScheduler
        } ]
  }

-- Training state for the trainer API.
structure TrainState where
  /-- params. -/
  params : Params
  /-- opt. -/
  opt : Train.OptimizerState Float

def initState : TrainState := { params := initParams, opt := initOptim }

-- Single-sample loss using the tape.
def sampleLoss (WId bId : Nat) (sample : Sample) :
  Runtime.Autograd.TapeM Float Nat := do
  let xVal : Tensor Float (.dim inDim .scalar) := fill sample.fst (.dim inDim .scalar)
  let yVal : Tensor Float (.dim outDim .scalar) := fill sample.snd (.dim outDim .scalar)
  let xId ← Train.TapeM.const xVal (name := some "x")
  let yId ← Train.TapeM.const yVal (name := some "y")
  let yHatId ← TapeM.linear (inDim:=inDim) (outDim:=outDim) WId bId xId
  let lossId ← TapeM.mseLoss (s:=.dim outDim .scalar) yHatId yId
  pure lossId

-- One optimizer-backed training step over a batch of samples.
def trainStep
  (s : TrainState) (batch : List Sample) :
  Runtime.Autograd.Result (Prod TrainState Float) := do
  let t0 : Tape Float := Tape.empty
  let m : TapeM Float _ := do
    let wId ← Train.TapeM.param s.params.W (name := some "W")
    let bId ← Train.TapeM.param s.params.b (name := some "b")
    let lossId ← Train.TapeM.meanScalarOver (tag := tag) batch (fun sample => sampleLoss wId bId
      sample)
    let t ← TapeM.getTape
    let lossVal ← liftM (Train.requireScalarValue (tag := tag) t lossId)
    let grads ← liftM (Tape.backwardScalar (t:=t) lossId)
    pure (wId, bId, lossVal, grads)

  let ((wId, bId, lossVal, grads), _) ← TapeM.run t0 m

  let paramTable : Train.ParamTable Float :=
    [ Train.ParamEntry.ofTensor wId s.params.W (name := some "W")
    , Train.ParamEntry.ofTensor bId s.params.b (name := some "b")
    ]

  let (opt', paramTable') ← Train.Optim.step s.opt paramTable grads

  let newW ← Train.ParamTable.getTensor (tag := tag)
    (s:=.dim outDim (.dim inDim .scalar)) paramTable' wId
  let newb ← Train.ParamTable.getTensor (tag := tag)
    (s:=.dim outDim .scalar) paramTable' bId

  let newParams : Params := { W := newW, b := newb }
  pure ({ params := newParams, opt := opt' }, lossVal)

-- One trainer step over the fixed dataset.
def step (s : TrainState) :
  Runtime.Autograd.Result (Prod TrainState (Train.StepReport Float)) := do
  let (s', loss) ← trainStep s dataset
  pure (s', { loss := loss, metrics := [] })

def trainer : Train.Trainer Runtime.Autograd.Result TrainState Float :=
  Train.Trainer.noLog initState step

/-!
## Training

Run a small number of epochs and report the per-epoch loss.
-/
def train (epochs : Nat) :
  Runtime.Autograd.Result (Prod Params (List Float)) := do
  let (s, losses) ← Train.Trainer.runLosses (steps := epochs) trainer
  pure (s.params, losses)

/-!
## Evaluation

This runs a forward pass only (no backprop) and averages loss over a dataset.
-/
def evalSample (p : Params) : Sample -> Runtime.Autograd.Result (Train.StepReport Float)
  | sample => do
      let t0 : Tape Float := Tape.empty
      let m : TapeM Float _ := do
        let wId ← Train.TapeM.const p.W (name := some "W")
        let bId ← Train.TapeM.const p.b (name := some "b")
        let lossId ← sampleLoss wId bId sample
        let t ← TapeM.getTape
        let lossVal ← liftM (Train.requireScalarValue (tag := tag) t lossId)
        pure lossVal
      let (lossVal, _) ← TapeM.run t0 m
      pure { loss := lossVal, metrics := [] }

def evalDataset (p : Params) : Runtime.Autograd.Result (Train.StepReport Float) :=
  Train.Eval.evalDataset (tag := tag) testDataset (evalSample p)

def run : IO Unit := do
  let res :=
    (Train.Trainer.run (steps := 5) trainer) >>= fun (s, reports) => do
      let evalReport ← evalDataset s.params
      pure (Train.renderReports reports, Train.renderReport 0 evalReport,
        pretty s.params.W, pretty s.params.b)
  match res with
  | .error msg => throw <| IO.userError s!"autograd_linear_regression_test (Float): {msg}"
  | .ok (reports, evalReport, wStr, bStr) =>
    IO.println "=== Autograd linear regression (Float) ==="
    for line in reports do
      IO.println line
    IO.println evalReport
    IO.println s!"W: {wStr}"
    IO.println s!"b: {bStr}"

end AutogradLinearRegression
end Floats
end Tests

/-! ## autograd_train_test.lean -/

/-!
End-to-end training smoke test using the dynamic autograd tape.

This mirrors the hand-written SGD loop in `mlp_test.lean`, but gradients are produced
via the dynamic tape (`Runtime.Autograd.Tape.backwardScalar`).
-/

open _root_.Spec
open _root_.Spec.Tensor

namespace Tests
namespace Floats
namespace AutogradTrain

open Runtime.Autograd

abbrev inDim  := 2
abbrev hidDim := 3
abbrev outDim := 1

-- Small tag used for readable error messages.
abbrev tag : String := "autograd_train_test"

/-!
## Parameter record

We keep parameters in a small structure so we can update them together after each step.
-/
structure Params where
  /-- Weight matrix for layer 1. -/
  W1 : Tensor Float (.dim hidDim (.dim inDim .scalar))
  /-- Bias for layer 1. -/
  b1 : Tensor Float (.dim hidDim .scalar)
  /-- Weight matrix for layer 2. -/
  W2 : Tensor Float (.dim outDim (.dim hidDim .scalar))
  /-- Bias for layer 2. -/
  b2 : Tensor Float (.dim outDim .scalar)

-- A fixed initialization so the test is deterministic.
def initParams : Params :=
  {
    W1 := tensorND! [hidDim, inDim] [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    b1 := tensorND! [hidDim] [0.1, 0.2, 0.3],
    W2 := tensorND! [outDim, hidDim] [0.7, 0.8, 0.9],
    b2 := tensorND! [outDim] [0.4]
  }

def x : Tensor Float (.dim inDim .scalar) :=
  tensorND! [inDim] [0.5, 0.8]

def yTarget : Tensor Float (.dim outDim .scalar) :=
  tensorND! [outDim] [1.0]

/-!
## One training step

We explicitly build the tape with `Tape.*` operations, then call `backwardScalar`.
-/
def trainStep (p : Params) (lr : Float := 0.1) : Runtime.Autograd.Result (Prod Params Float) := do
  let t0 : Tape Float := Tape.empty
  let (t1, w1Id) := Tape.leaf (t:=t0) p.W1 (name := some "W1")
  let (t2, b1Id) := Tape.leaf (t:=t1) p.b1 (name := some "b1")
  let (t3, w2Id) := Tape.leaf (t:=t2) p.W2 (name := some "W2")
  let (t4, b2Id) := Tape.leaf (t:=t3) p.b2 (name := some "b2")
  let (t5, xId)  := Tape.leaf (t:=t4) x (name := some "x") (requires_grad := false)
  let (t6, yId)  := Tape.leaf (t:=t5) yTarget (name := some "y") (requires_grad := false)

  -- Forward pass: linear -> relu -> linear -> mse_loss
  let (t7, z1Id) ← Tape.linear (t:=t6) (inDim:=inDim) (outDim:=hidDim) w1Id b1Id xId
  let (t8, a1Id) ← Tape.relu (t:=t7) (s:=.dim hidDim .scalar) z1Id
  let (t9, yhatId) ← Tape.linear (t:=t8) (inDim:=hidDim) (outDim:=outDim) w2Id b2Id a1Id
  let (t10, lossId) ← Tape.mseLoss (t:=t9) (s:=.dim outDim .scalar) yhatId yId

  -- Read loss and backpropagate from the scalar loss node.
  let lossVal ← Train.requireScalarValue (tag := tag) t10 lossId
  let grads ← Tape.backwardScalar (t:=t10) lossId

  -- Extract typed gradients and apply SGD updates.
  let dW1 ← Train.requireGradTensor (tag := tag) (s:=.dim hidDim (.dim inDim .scalar)) grads w1Id
  let db1 ← Train.requireGradTensor (tag := tag) (s:=.dim hidDim .scalar) grads b1Id
  let dW2 ← Train.requireGradTensor (tag := tag) (s:=.dim outDim (.dim hidDim .scalar)) grads w2Id
  let db2 ← Train.requireGradTensor (tag := tag) (s:=.dim outDim .scalar) grads b2Id

  let newW1 := Train.sgdUpdateTensor p.W1 dW1 lr
  let newb1 := Train.sgdUpdateTensor p.b1 db1 lr
  let newW2 := Train.sgdUpdateTensor p.W2 dW2 lr
  let newb2 := Train.sgdUpdateTensor p.b2 db2 lr

  pure ({ W1 := newW1, b1 := newb1, W2 := newW2, b2 := newb2 }, lossVal)

/-!
## Training loop

Run a small fixed number of epochs and collect the per-epoch loss.
-/
def train (epochs : Nat) (lr : Float := 0.1) :
  Runtime.Autograd.Result (List Float) := do
  let (_, losses) ← Train.runStepsM (m := Runtime.Autograd.Result) epochs initParams
    (fun p => trainStep p lr)
  pure losses

def run : IO Unit := do
  match train 6 0.1 with
  | .ok losses =>
    IO.println "=== Autograd train smoke test (Float) ==="
    IO.println s!"losses: {losses}"
  | .error msg => throw <| IO.userError s!"autograd_train_test (Float): {msg}"

end AutogradTrain
end Floats
end Tests

/-! ## autograd_layernorm_test.lean -/

/-!
Small layer norm gradient smoke test using the dynamic tape.
-/

open _root_.Spec
open _root_.Spec.Tensor

namespace Tests
namespace Floats
namespace AutogradLayerNorm

open Runtime.Autograd

abbrev seqLen := 2
abbrev embedDim := 3

def x : Tensor Float (.dim seqLen (.dim embedDim .scalar)) :=
  tensorND! [seqLen, embedDim] [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

def gamma : Tensor Float (.dim embedDim .scalar) :=
  tensorND! [embedDim] [1.0, 0.9, 1.1]

def beta : Tensor Float (.dim embedDim .scalar) :=
  tensorND! [embedDim] [0.0, 0.1, -0.1]

def checkLayerNormGrads :
  Runtime.Autograd.Result (String × String × String) := do
  let t0 : Tape Float := Tape.empty
  let m : TapeM Float _ := do
    let xId ← Train.TapeM.param x (name := some "x")
    let gammaId ← Train.TapeM.param gamma (name := some "gamma")
    let betaId ← Train.TapeM.param beta (name := some "beta")
    let yId ← TapeM.layerNorm (seqLen := seqLen) (embedDim := embedDim) (by decide) (by decide) xId
      gammaId betaId
    let lossId ← TapeM.sum (s := .dim seqLen (.dim embedDim .scalar)) yId
    let t ← TapeM.getTape
    let lossVal ← liftM (Train.requireScalarValue (tag := "layer_norm") t lossId)
    let grads ← liftM (Tape.backwardScalar (t := t) lossId)
    pure (xId, gammaId, betaId, lossVal, grads)

  let ((xId, gammaId, betaId, lossVal, grads), _) ← TapeM.run t0 m

  let dX ← Train.requireGradTensor (tag := "layer_norm")
    (s := .dim seqLen (.dim embedDim .scalar)) grads xId
  let dGamma ← Train.requireGradTensor (tag := "layer_norm")
    (s := .dim embedDim .scalar) grads gammaId
  let dBeta ← Train.requireGradTensor (tag := "layer_norm")
    (s := .dim embedDim .scalar) grads betaId

  pure (s!"loss={lossVal}", pretty dGamma, pretty dBeta)

def run : IO Unit := do
  match checkLayerNormGrads with
  | .error msg => throw <| IO.userError s!"autograd_layernorm_test (Float): {msg}"
  | .ok (lossStr, dGammaStr, dBetaStr) =>
    IO.println "=== Autograd layer norm grad smoke test (Float) ==="
    IO.println lossStr
    IO.println s!"dGamma: {dGammaStr}"
    IO.println s!"dBeta: {dBetaStr}"

end AutogradLayerNorm
end Floats
end Tests

/-! ## autograd_conv2d_test.lean -/

/-!
Conv2D gradient smoke test using the dynamic tape.
-/

open _root_.Spec
open _root_.Spec.Tensor

namespace Tests
namespace Floats
namespace AutogradConv2d

open Runtime.Autograd

abbrev inC := 1
abbrev outC := 1
abbrev kH := 2
abbrev kW := 2
abbrev stride := 1
abbrev padding := 0
abbrev inH := 2
abbrev inW := 2

def h1 : inC ≠ 0 := by decide
def h2 : kH ≠ 0 := by decide
def h3 : kW ≠ 0 := by decide

def outH : Nat := (inH + 2 * padding - kH) / stride + 1
def outW : Nat := (inW + 2 * padding - kW) / stride + 1

def kernel : Tensor Float (.dim outC (.dim inC (.dim kH (.dim kW .scalar)))) :=
  tensorND! [outC, inC, kH, kW] [0.2, -0.1, 0.3, 0.4]

def bias : Tensor Float (.dim outC .scalar) :=
  tensorND! [outC] [0.05]

def input : Tensor Float (.dim inC (.dim inH (.dim inW .scalar))) :=
  tensorND! [inC, inH, inW] [1.0, 2.0, 3.0, 4.0]

def checkConv2dGrads :
  Runtime.Autograd.Result (String × String) := do
  let t0 : Tape Float := Tape.empty
  let m : TapeM Float _ := do
    let kId ← Train.TapeM.param kernel (name := some "kernel")
    let bId ← Train.TapeM.param bias (name := some "bias")
    let xId ← Train.TapeM.const input (name := some "input")
    let yId ← TapeM.conv2d (inC := inC) (outC := outC) (kH := kH) (kW := kW)
      (stride := stride) (padding := padding) (inH := inH) (inW := inW)
      (h1 := h1) (h2 := h2) (h3 := h3) kId bId xId
    let lossId ← TapeM.sum (s := .dim outC (.dim outH (.dim outW .scalar))) yId
    let t ← TapeM.getTape
    let grads ← liftM (Tape.backwardScalar (t := t) lossId)
    pure (kId, bId, grads)

  let ((kId, bId, grads), _) ← TapeM.run t0 m
  let dK ← Train.requireGradTensor (tag := "conv2d")
    (s := .dim outC (.dim inC (.dim kH (.dim kW .scalar)))) grads kId
  let dB ← Train.requireGradTensor (tag := "conv2d")
    (s := .dim outC .scalar) grads bId
  pure (pretty dK, pretty dB)

def run : IO Unit := do
  match checkConv2dGrads with
  | .error msg => throw <| IO.userError s!"autograd_conv2d_test (Float): {msg}"
  | .ok (dKStr, dBStr) =>
    IO.println "=== Autograd conv2d grad smoke test (Float) ==="
    IO.println s!"dK: {dKStr}"
    IO.println s!"dB: {dBStr}"

end AutogradConv2d
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
  AutogradConv2d.run
  IO.println "=== Autograd test suite completed ==="

end Floats
end Tests
