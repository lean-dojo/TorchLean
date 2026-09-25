/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core.Trainer.Types
public import NN.Runtime.Autograd.Torch.Core.Trainer.GraphOps
public import NN.Runtime.Autograd.Torch.Core.Functional.GraphInputs
public import NN.Runtime.Autograd.Torch.Core.TypedGraph

/-!
# Graph Trainer

Build the graph once, then compile it with the current parameters on each call. Each backward
pass uses the local programs saved by that forward execution.
-/

public section

namespace Runtime.Autograd.Torch

open Spec TorchLean
open TorchLean.Tensor

/-- Construct the CPU graph backend for a scalar trainer. -/
def Internal.graphScalarTrainer {α δ : Type} [TorchLean.Storage α] [TorchLean.Storage δ]
    [Context α] [TensorTransfer α] {paramShapes inputShapes dataInputShapes : List Shape}
    (options : Config) (parameters : ParamList α paramShapes)
    (validateDataInputsIO : TorchLean.TensorPack δ dataInputShapes → IO Unit)
    (loss : ScalarLoss α δ paramShapes inputShapes dataInputShapes) :
    IO (ScalarTrainer α δ paramShapes inputShapes dataInputShapes) := do
  if options.device != .cpu then
    throw <| IO.userError <|
      s!"typed graph execution currently supports device `cpu`; "
        ++ s!"requested `{options.deviceName}`"
  let argumentShapes : List Shape := paramShapes ++ inputShapes
  let dataInputPack : Type := TorchLean.TensorPack δ dataInputShapes
  let buildLoss : Runtime.Autograd.TypedGraph.GraphM.MWith α dataInputPack argumentShapes
      (Runtime.Autograd.TypedGraph.GraphM.Var []) := do
    let argumentVariables ←
      Runtime.Autograd.TypedGraph.GraphM.args (α := α) (Γ := argumentShapes)
    let withDataInputs :=
      CurriedRef.applyVarList (Γ := argumentShapes)
        (β := CurriedRef (fun shape => dataInputPack → Tensor δ shape) dataInputShapes
          (Runtime.Autograd.TypedGraph.GraphM.MWith α dataInputPack argumentShapes
            (Runtime.Autograd.TypedGraph.GraphM.Var [])))
        (loss (m := Runtime.Autograd.TypedGraph.GraphM.MWith α dataInputPack argumentShapes))
        argumentVariables
    CurriedRef.applyPackProjections (full := dataInputShapes) id withDataInputs
  let graph ← okOrThrow
    (lowerToTypedGraphWithData
      (α := α) (Δ := dataInputPack) (Γ := argumentShapes) (τ := []) buildLoss)
  let nodeShapes : List Shape := graph.nodeShapes
  let graphData : Proofs.Autograd.Algebra.GraphData α dataInputPack argumentShapes nodeShapes :=
    graph.data
  let rec collectParameters : {ss : List Shape} → ParamList α ss → Array (AnyParam α) →
      Array (AnyParam α)
    | [], .nil, result => result
    | _ :: _, .cons parameter rest, result =>
        collectParameters rest (result.push (AnyParam.ofParam parameter))
  let parameterSlots := collectParameters parameters #[]

  let getScalar (compiled : Runtime.Autograd.TypedGraph.Compiled α argumentShapes nodeShapes) :
      Tensor α [] :=
    compiled.context.lookup.read graph.output

  let runCompiled (inputs : TorchLean.TensorPack α inputShapes) (dataInputs : dataInputPack) :
      IO (Runtime.Autograd.TypedGraph.Compiled α argumentShapes nodeShapes) := do
    validateDataInputsIO dataInputs
    let parameterValues ← ParamList.values (α := α) parameters
    let arguments := TorchLean.TensorPack.append (α := α) (ss₁ := paramShapes)
      (ss₂ := inputShapes) parameterValues inputs
    let compiled ← okOrThrow <|
      Runtime.Autograd.TypedGraph.compileChecked graphData arguments dataInputs
    let getValue {s : Shape} (id : Nat) : IO (Tensor α s) := do
      let some value := compiled.context.values[id]?
        | throw <| IO.userError "typed graph buffer update: missing recorded value"
      if h : value.shape = s then
        pure (value.cast h)
      else
        throw <| IO.userError "typed graph buffer update: recorded shape mismatch"
    let getParameter (id : Nat) : IO (AnyParam α) :=
      match parameterSlots[id]? with
      | some parameter => pure parameter
      | none => throw <| IO.userError "typed graph buffer update: expected a state reference"
    let getState {s : Shape} (id : Nat) : IO (Tensor α s) := do
      let value ← (← getParameter id).get
      if h : value.shape = s then
        pure (value.cast h)
      else
        throw <| IO.userError "typed graph buffer update: state shape mismatch"
    for update in graph.bufferUpdates do
      for (id, value) in ← update getState getValue do
        let parameter ← getParameter id
        unless parameter.requiresGrad do
          parameter.set value
    pure compiled
  -- Parameters are the first graph arguments, so their gradients form the leading pack.
  let parameterGradients
      (compiled : Runtime.Autograd.TypedGraph.Compiled α argumentShapes nodeShapes) :
      IO (TorchLean.TensorPack α paramShapes) := do
    let gradients := compiled.backwardDenseAllFrom graph.output (Tensor.scalar (1 : α))
    let values ← okOrThrow (TorchLean.TensorPack.ofShapeErasedArray
      (α := α) gradients (shapes := paramShapes))
    let rec retainTrainable : {shapes : List Shape} → ParamList α shapes →
        TorchLean.TensorPack α shapes → TorchLean.TensorPack α shapes
      | [], .nil, .nil => .nil
      | shape :: _, .cons parameter rest, .cons gradient gradients =>
          .cons (if parameter.requiresGrad then gradient else Tensor.zeros shape)
            (retainTrainable rest gradients)
    pure (retainTrainable parameters values)
  let lossFn :
      Curried.Fn α inputShapes
        (Curried.Fn δ dataInputShapes (IO (Tensor α []))) :=
    Curried.curry (α := α) (ss := inputShapes)
      (β := Curried.Fn δ dataInputShapes (IO (Tensor α []))) (fun inputs =>
        Curried.curry (α := δ) (ss := dataInputShapes)
          (β := IO (Tensor α [])) (fun dataInputs =>
            getScalar <$> runCompiled inputs dataInputs))
  let diff :
      Curried.Fn α inputShapes (Curried.Fn δ dataInputShapes
        (IO (Tensor α [] × TorchLean.TensorPack α paramShapes))) :=
    Curried.curry (α := α) (ss := inputShapes)
      (β := Curried.Fn δ dataInputShapes
        (IO (Tensor α [] × TorchLean.TensorPack α paramShapes))) (fun inputs =>
        Curried.curry (α := δ) (ss := dataInputShapes)
          (β := IO (Tensor α [] × TorchLean.TensorPack α paramShapes)) (fun dataInputs => do
            let compiled ← runCompiled inputs dataInputs
            let lossValue := getScalar compiled
            let gradients ← parameterGradients compiled
            pure (lossValue, gradients)))
  let grad :
      Curried.Fn α inputShapes
        (Curried.Fn δ dataInputShapes (IO (TorchLean.TensorPack α paramShapes))) :=
    Curried.curry (α := α) (ss := inputShapes)
      (β := Curried.Fn δ dataInputShapes (IO (TorchLean.TensorPack α paramShapes))) (fun inputs =>
        Curried.curry (α := δ) (ss := dataInputShapes)
          (β := IO (TorchLean.TensorPack α paramShapes)) (fun dataInputs => do
            let compiled ← runCompiled inputs dataInputs
            parameterGradients compiled))
  let stepWithLoss (learningRate : α) :
      Curried.Fn α inputShapes
        (Curried.Fn δ dataInputShapes (IO (Tensor α []))) :=
    Curried.curry (α := α) (ss := inputShapes)
      (β := Curried.Fn δ dataInputShapes (IO (Tensor α []))) (fun inputs =>
        Curried.curry (α := δ) (ss := dataInputShapes)
          (β := IO (Tensor α [])) (fun dataInputs => do
            let diffForData :=
              Curried.uncurry (α := α) (ss := inputShapes)
                (β := Curried.Fn δ dataInputShapes
                  (IO (Tensor α [] × TorchLean.TensorPack α paramShapes)))
                diff inputs
            let (lossValue, gradients) ←
              Curried.uncurry (α := δ) (ss := dataInputShapes)
                (β := IO (Tensor α [] × TorchLean.TensorPack α paramShapes))
                diffForData dataInputs
            ParamList.sgdStep (α := α) (ss := paramShapes) parameters learningRate gradients
            pure lossValue))
  let step (learningRate : α) :
      Curried.Fn α inputShapes (Curried.Fn δ dataInputShapes (IO Unit)) :=
    Curried.curry (α := α) (ss := inputShapes)
      (β := Curried.Fn δ dataInputShapes (IO Unit)) (fun inputs =>
        Curried.curry (α := δ) (ss := dataInputShapes) (β := IO Unit)
          (fun dataInputs => do
            let gradForData :=
              Curried.uncurry (α := α) (ss := inputShapes)
                (β := Curried.Fn δ dataInputShapes
                  (IO (TorchLean.TensorPack α paramShapes))) grad inputs
            let gradients ← Curried.uncurry (α := δ) (ss := dataInputShapes)
              (β := IO (TorchLean.TensorPack α paramShapes)) gradForData dataInputs
            ParamList.sgdStep (α := α) (ss := paramShapes) parameters learningRate gradients))
  pure
    { state := parameters
      lossFn := lossFn
      diffFn := diff
      gradFn := grad
      stepWithLossFn := stepWithLoss
      stepFn := step
      adamStep? := none
      adamStepWithLoss? := none
      adamWStep? := none
      adamWStepWithLoss? := none
      optimizerStateCheckpoint? := none
      getState := ParamList.values (α := α) (ss := paramShapes) parameters }

end Runtime.Autograd.Torch
