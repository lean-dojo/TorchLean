/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core.Trainer.Types

/-!
# Scalar Trainer Operations

Packed loss, gradient, and update operations, plus simple training and evaluation loops.
Differentiable inputs use scalar type `α`; non-differentiable data, such as token identifiers,
labels, or masks, use a separate type `δ`.
-/

@[expose] public section

namespace Runtime.Autograd.Torch

open Spec TorchLean TorchLean.Tensor

namespace ScalarTrainer

/-- Evaluate the scalar loss on packed differentiable and non-differentiable inputs. -/
def runLoss {α δ : Type} [Storage α] [Storage δ]
    {paramShapes inputShapes dataInputShapes : List Shape}
    (trainer : ScalarTrainer α δ paramShapes inputShapes dataInputShapes)
    (inputs : TensorPack α inputShapes)
    (dataInputs : TensorPack δ dataInputShapes) : IO (Tensor α .scalar) :=
  let withData := Curried.uncurry (α := α) (ss := inputShapes)
    (β := Curried.Fn δ dataInputShapes (IO (Tensor α .scalar))) trainer.loss inputs
  Curried.uncurry (α := δ) (ss := dataInputShapes)
    (β := IO (Tensor α .scalar)) withData dataInputs

/-- Evaluate one loss and its parameter gradients from the same tape. -/
def runDiff {α δ : Type} [Storage α] [Storage δ]
    {paramShapes inputShapes dataInputShapes : List Shape}
    (trainer : ScalarTrainer α δ paramShapes inputShapes dataInputShapes)
    (inputs : TensorPack α inputShapes)
    (dataInputs : TensorPack δ dataInputShapes) :
    IO (Tensor α .scalar × TensorPack α paramShapes) :=
  let withData := Curried.uncurry (α := α) (ss := inputShapes)
    (β := Curried.Fn δ dataInputShapes
      (IO (Tensor α .scalar × TensorPack α paramShapes)))
    trainer.diff inputs
  Curried.uncurry (α := δ) (ss := dataInputShapes)
    (β := IO (Tensor α .scalar × TensorPack α paramShapes))
    withData dataInputs

/-- Evaluate parameter gradients on packed inputs. -/
def runGrad {α δ : Type} [Storage α] [Storage δ]
    {paramShapes inputShapes dataInputShapes : List Shape}
    (trainer : ScalarTrainer α δ paramShapes inputShapes dataInputShapes)
    (inputs : TensorPack α inputShapes)
    (dataInputs : TensorPack δ dataInputShapes) : IO (TensorPack α paramShapes) :=
  let withData := Curried.uncurry (α := α) (ss := inputShapes)
    (β := Curried.Fn δ dataInputShapes (IO (TensorPack α paramShapes))) trainer.grad inputs
  Curried.uncurry (α := δ) (ss := dataInputShapes)
    (β := IO (TensorPack α paramShapes)) withData dataInputs

/-- Apply the trainer's SGD update to packed inputs. -/
def runStep {α δ : Type} [Storage α] [Storage δ]
    {paramShapes inputShapes dataInputShapes : List Shape}
    (trainer : ScalarTrainer α δ paramShapes inputShapes dataInputShapes)
    (learningRate : α) (inputs : TensorPack α inputShapes)
    (dataInputs : TensorPack δ dataInputShapes) : IO Unit :=
  let withData := Curried.uncurry (α := α) (ss := inputShapes)
    (β := Curried.Fn δ dataInputShapes (IO Unit)) (trainer.step learningRate) inputs
  Curried.uncurry (α := δ) (ss := dataInputShapes) (β := IO Unit) withData dataInputs

/-- Apply the trainer's SGD update and return the loss used for the update. -/
def runStepWithLoss {α δ : Type} [Storage α] [Storage δ]
    {paramShapes inputShapes dataInputShapes : List Shape}
    (trainer : ScalarTrainer α δ paramShapes inputShapes dataInputShapes)
    (learningRate : α) (inputs : TensorPack α inputShapes)
    (dataInputs : TensorPack δ dataInputShapes) : IO (Tensor α .scalar) :=
  let withData := Curried.uncurry (α := α) (ss := inputShapes)
    (β := Curried.Fn δ dataInputShapes (IO (Tensor α .scalar)))
    (trainer.stepWithLoss learningRate) inputs
  Curried.uncurry (α := δ) (ss := dataInputShapes)
    (β := IO (Tensor α .scalar)) withData dataInputs

end ScalarTrainer

/--
Apply `steps` SGD updates while cycling through samples without auxiliary data tensors.

Logging reuses each update's loss. Set `logEvery := 0` to disable logging.
Rejects an empty dataset.
-/
def trainCycleSGD {α : Type} [Storage α] [ToString α]
    {paramShapes inputShapes : List Shape}
    (trainer : ScalarTrainer α Unit paramShapes inputShapes)
    (learningRate : α) (steps : Nat)
    (samples : List (TensorPack α inputShapes))
    (logEvery : Nat := 1) : IO Unit := do
  -- Convert once so each step has constant-time sample lookup.
  let samples := samples.toArray
  if empty : samples.size = 0 then
    throw <| IO.userError "trainCycleSGD: empty dataset"
  else
    for step in [0:steps] do
      let inputs := samples[step % samples.size]'(Nat.mod_lt _ (Nat.pos_of_ne_zero empty))
      if logEvery != 0 && step % logEvery = 0 then
        let loss ← ScalarTrainer.runStepWithLoss trainer learningRate inputs .nil
        IO.println s!"step {step}: loss={loss.item}"
      else
        ScalarTrainer.runStep trainer learningRate inputs .nil

/--
Evaluate the arithmetic mean loss over samples without auxiliary data tensors.

Rejects an empty dataset.
-/
def meanLoss {α : Type} [Storage α] [Add α] [Div α] [Zero α] [NatCast α]
    {paramShapes inputShapes : List Shape}
    (trainer : ScalarTrainer α Unit paramShapes inputShapes)
    (samples : List (TensorPack α inputShapes)) : IO α := do
  if samples.isEmpty then
    throw <| IO.userError "meanLoss: empty dataset"
  let mut total : α := 0
  for inputs in samples do
    let loss ← ScalarTrainer.runLoss trainer inputs .nil
    total := total + loss.item
  pure (total / (samples.length : α))

end Runtime.Autograd.Torch
