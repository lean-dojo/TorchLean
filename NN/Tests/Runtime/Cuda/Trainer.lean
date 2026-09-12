/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Trainer.Constructor
public import NN.API.Trainer.Session
public import NN.Runtime.Autograd.Model.Session.Types
public import NN.Runtime.Autograd.Torch.Core.Ops.ShapeReduction
public import NN.Runtime.Autograd.Torch.Core.Trainer

/-!
# CUDA Trainer Coverage

User-facing trainer checks whose behavior depends on selecting a distinct evaluation program.
-/

@[expose] public section

namespace Tests
namespace Cuda
namespace Trainer

open TorchLean

def dropoutModel : nn.Builder (nn.Sequential [1] [1]) :=
  nn.dropout (shape := [1]) 1.0

def sample : Sample.Supervised Float [1] [1] :=
  { input := [2.0], target := [0.0] }

def close (actual expected : Float) : Bool :=
  Float.abs (actual - expected) <= 1e-5

/--
Evaluation must use the evaluation-mode CUDA objective without changing subsequent training.

Full-probability dropout makes the distinction exact: training sees zero, while evaluation is the
identity and therefore has mean-squared error four against the zero target.
-/
def checkEvaluationMode : IO Unit := do
  let trainer := TorchLean.Trainer.new dropoutModel
    { device := .cuda, execution := .eager }
  let session ← trainer.open
  let prediction ← session.predict sample.input
  let evaluationLoss ← session.loss sample
  unless close prediction[0] 2.0 do
    throw <| IO.userError
      s!"CUDA trainer evaluation prediction: got {prediction[0]}, expected 2"
  unless close evaluationLoss 4.0 do
    throw <| IO.userError
      s!"CUDA trainer evaluation loss: got {evaluationLoss}, expected 4"
  let trainingLoss ← session.step sample
  unless close trainingLoss 0.0 do
    throw <| IO.userError
      s!"CUDA trainer training loss after evaluation: got {trainingLoss}, expected 0"
  let repeatedEvaluationLoss ← session.loss sample
  unless close repeatedEvaluationLoss 4.0 do
    throw <| IO.userError
      s!"CUDA trainer repeated evaluation loss: got {repeatedEvaluationLoss}, expected 4"

/-- Reject a shape whose element count wraps to zero in the CUDA size ABI. -/
def checkBufferSizeBoundary : IO Unit := do
  let buffer := Runtime.Autograd.Cuda.Buffer.zeros 0
  let check := Runtime.Autograd.Torch.Internal.EagerSession.checkCudaAnyBufferSize
  check "empty buffer" { s := [0], buf := buffer }
  let rejected ← try
    check "oversized buffer" { s := [UInt32.size], buf := buffer }
    pure false
  catch _ => pure true
  unless rejected do
    throw <| IO.userError "CUDA buffer size validation accepted a wrapped element count"

/-- A public CPU typed-graph session must read the current value of a CUDA parameter. -/
def checkTypedGraphCudaParameter : IO Unit := do
  let parameter ← Runtime.Autograd.Torch.Param.Internal.create (Tensor.scalar 1.0)
  let current := Runtime.Autograd.Cuda.Buffer.full 1 2.0
  Runtime.Autograd.Torch.Internal.setParamCudaValue parameter { s := [], buf := current }
  let session ← Runtime.Autograd.Model.Session.new (α := Float) { execution := .typedGraph }
  let reference ← session.use parameter
  let value ← session.getValue reference
  unless value.item == 2.0 do
    throw <| IO.userError
      s!"typed graph read stale CUDA parameter: got {value.item}, expected 2"

/-- Resetting views must preserve the parameter allocation, including frozen parameters. -/
def checkParameterViews : IO Unit := do
  for requiresGrad in #[true, false] do
    let parameter ← Runtime.Autograd.Torch.Param.Internal.create
      ([1.0, 2.0] : Tensor Float [2]) (requiresGrad := requiresGrad)
    let session ← Runtime.Autograd.Torch.Internal.EagerSession.new (α := Float)
      { device := .cuda, execution := .eager }
    for _ in [0:3] do
      let reference ← session.use parameter
      let detached ← session.detach reference
      let reshaped ← session.reshape (sh2 := [1, 2]) detached (by decide)
      let flattened ← session.flatten reshaped
      let value ← session.getValue flattened
      unless close value[0] 1.0 && close value[1] 2.0 do
        throw <| IO.userError "CUDA parameter view changed after a tape reset"
      session.resetTape

/--
Repeating one sample in an accumulated batch leaves its mean gradient unchanged. Adam and AdamW
must therefore follow the same parameter trajectory when batch sizes and loss reporting vary.
-/
def checkMixedBatchOptimizer (optimizer : optim.Optimizer) : IO Unit := do
  let trainer := TorchLean.Trainer.new (nn.linear 1 1)
    { device := .cuda, execution := .eager, optimizer, seed := 37 }
  let reference ← trainer.open
  let mixed ← trainer.open
  for i in [0:8] do
    let x := (i % 3).toFloat - 1.0
    let y := if i % 2 == 0 then 2.0 else -3.0
    let current : Sample.Supervised Float [1] [1] := { input := [x], target := [y] }
    let expectedLoss ← reference.step current
    let batch := Array.replicate (if i % 3 == 0 then 1 else 3) current
    if i % 2 == 0 then
      let actualLoss ← mixed.stepBatch batch
      unless Float.abs (actualLoss - expectedLoss) ≤ 1e-4 do
        throw <| IO.userError s!"CUDA mixed batch loss diverged at update {i}"
    else
      mixed.updateBatch batch
    for probe in #[-1.0, 0.0, 2.0] do
      let expected ← reference.predict [probe]
      let actual ← mixed.predict [probe]
      unless Float.abs (actual[0] - expected[0]) ≤ 1e-4 do
        throw <| IO.userError
          s!"CUDA mixed batch optimizer diverged at update {i}: \
            got {actual[0]}, expected {expected[0]}"

/-- Average large finite losses and gradients without overflowing their unnormalized sums. -/
def checkLargeBatchMean : IO Unit := do
  let makeTrainer := Runtime.Autograd.Torch.scalarTrainer
    (α := Float) (δ := Float) (paramShapes := [[]]) (inputShapes := [])
    (dataInputShapes := []) (options := { device := .cuda, execution := .eager })
    (loss := fun {m} _ _ parameter =>
      (show m (Runtime.Autograd.Torch.Ops.Ref m Float []) from do
        let large ← Runtime.Autograd.Torch.Ops.const
          (m := m) (α := Float) (Tensor.scalar 2e38)
        let product ← Runtime.Autograd.Torch.Ops.mul (m := m) parameter large
        Runtime.Autograd.Torch.Ops.add (m := m) large product))
  let trainer ← makeTrainer (Tensor.scalar 0.0)
  let some step := trainer.nativeBatchStep?
    | throw <| IO.userError "CUDA trainer omitted native batch update"
  let some mean ← step (.sgd 2e-38) #[(.nil, .nil), (.nil, .nil)] true
    | throw <| IO.userError "CUDA trainer omitted requested batch loss"
  unless Float.abs (mean.item / 2e38 - 1.0) < 1e-5 do
    throw <| IO.userError s!"CUDA batch mean overflowed: {mean.item}"
  let .cons parameter .nil ← trainer.getState
  unless Float.abs (parameter.item + 4.0) < 1e-5 do
    throw <| IO.userError s!"CUDA batch gradient overflowed: {parameter.item}"

/-- A scalar square gives Adam a changing gradient with just one parameter. -/
def quadraticObjective (initial : Float) :
    IO (Module.Objective Float Float [[]] []) := do
  let objective ← Runtime.Autograd.Model.Module.Objective.create
    (α := Float) (β := Float) (stateShapes := [[]]) (inputShapes := [])
    (dataInputShapes := []) (runtime := { device := .cuda, execution := .eager })
    (loss := fun {m} _ _ parameter =>
      Runtime.Autograd.Torch.Ops.mul (m := m) parameter parameter)
    (initState := .cons (Tensor.scalar initial) .nil)
  pure (Module.Objective.Internal.fromRuntime objective)

/-- Starting a fresh optimizer history resets device moments and permits changing update paths. -/
def checkOptimizerReset : IO Unit := do
  let optimizer := Runtime.Autograd.Model.Optim.adam
    (α := Float) (paramShapes := [[]]) 0.3 0.9 0.999 1e-8
  let objective ← quadraticObjective 1.0
  let initialState ← objective.initOptimizer optimizer
  let state ← objective.step optimizer initialState .empty .empty
  let state ← objective.step optimizer state .empty .empty
  let before := (← objective.state).get 0
  let gradients ← objective.grad .empty .empty
  let rejected ← try
    let _ ← objective.update optimizer state gradients
    pure false
  catch _ => pure true
  unless rejected && close ((← objective.state).get 0).item before.item do
    throw <| IO.userError "CUDA accepted a generic update into native Adam history"

  let fresh ← quadraticObjective before.item
  let freshState ← fresh.initOptimizer optimizer
  let resetState ← objective.initOptimizer optimizer
  let _ ← fresh.step optimizer freshState .empty .empty
  let _ ← objective.step optimizer resetState .empty .empty
  unless close ((← fresh.state).get 0).item ((← objective.state).get 0).item do
    throw <| IO.userError "CUDA optimizer reinitialization retained old Adam moments"

  let genericState ← objective.initOptimizer optimizer
  let gradients ← objective.grad .empty .empty
  let genericState ← objective.update optimizer genericState gradients
  let before := (← objective.state).get 0
  let rejected ← try
    let _ ← objective.step optimizer genericState .empty .empty
    pure false
  catch _ => pure true
  unless rejected && close ((← objective.state).get 0).item before.item do
    throw <| IO.userError "CUDA accepted a native update into generic Adam history"
  let resetState ← objective.initOptimizer optimizer
  let _ ← objective.step optimizer resetState .empty .empty
  pure ()

/-- A CUDA result owns its parameter snapshot after the live trainer moves on. -/
def checkSnapshot : IO Unit := do
  let trainer := TorchLean.Trainer.new (nn.linear 1 1)
    { device := .cuda, execution := .eager
      optimizer := optim.adam { learningRate := 0.1 }, seed := 37 }
  let session ← trainer.open
  let sample : Sample.Supervised Float [1] [1] := { input := [1.0], target := [4.0] }
  let before ← session.loss sample
  session.update sample
  let after ← session.loss sample
  let result ← session.finish { before, after }
  let savedPrediction ← result.predict sample.input
  for _ in [0:4] do
    session.updateBatch #[sample, sample]
  let livePrediction ← session.predict sample.input
  let repeatedPrediction ← result.predict sample.input
  unless close savedPrediction[0] repeatedPrediction[0] do
    throw <| IO.userError "CUDA result prediction changed after further training"
  unless Float.abs (livePrediction[0] - savedPrediction[0]) > 0.01 do
    throw <| IO.userError "CUDA snapshot regression did not move the live parameters"
  unless result.report.steps == 1 do
    throw <| IO.userError "CUDA result step count changed after further training"

def run : IO Unit := do
  IO.println "=== CUDA trainer coverage ==="
  checkEvaluationMode
  checkBufferSizeBoundary
  checkTypedGraphCudaParameter
  checkParameterViews
  checkMixedBatchOptimizer (optim.adam { learningRate := 0.03 })
  checkMixedBatchOptimizer (optim.adamW { learningRate := 0.03, weightDecay := 0.1 })
  checkLargeBatchMean
  checkOptimizerReset
  checkSnapshot

end Trainer
end Cuda
end Tests
