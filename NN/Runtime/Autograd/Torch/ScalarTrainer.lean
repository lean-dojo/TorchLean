/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core.Trainer.Types

/-!
# Scalar Trainer Operations

Packed loss, gradient, and update operations. Differentiable inputs use scalar type `α`;
non-differentiable data, such as token identifiers, labels, or masks, use a separate type `δ`.
-/

@[expose] public section

namespace Runtime.Autograd.Torch

open Spec TorchLean TorchLean.Tensor

namespace ScalarTrainer

/-- Evaluate the scalar loss on packed differentiable and non-differentiable inputs. -/
def loss {α δ : Type} [Storage α] [Storage δ]
    {paramShapes inputShapes dataInputShapes : List Shape}
    (trainer : ScalarTrainer α δ paramShapes inputShapes dataInputShapes)
    (inputs : TensorPack α inputShapes)
    (dataInputs : TensorPack δ dataInputShapes) : IO (Tensor α .scalar) :=
  let withData := Curried.uncurry (α := α) (ss := inputShapes)
    (β := Curried.Fn δ dataInputShapes (IO (Tensor α .scalar))) trainer.lossFn inputs
  Curried.uncurry (α := δ) (ss := dataInputShapes)
    (β := IO (Tensor α .scalar)) withData dataInputs

/--
Evaluate parameter gradients on packed inputs.

Set `value := true` to return `(gradients, loss)` from the same forward tape. The default uses
the backend's gradient-only operation.
-/
def grad {α δ : Type} [Storage α] [Storage δ]
    {paramShapes inputShapes dataInputShapes : List Shape}
    (trainer : ScalarTrainer α δ paramShapes inputShapes dataInputShapes)
    (inputs : TensorPack α inputShapes)
    (dataInputs : TensorPack δ dataInputShapes) (value : Bool := false) :
    IO (match value with
      | false => TensorPack α paramShapes
      | true => TensorPack α paramShapes × Tensor α .scalar) := by
  cases value with
  | false =>
      exact
        let withData := Curried.uncurry (α := α) (ss := inputShapes)
          (β := Curried.Fn δ dataInputShapes (IO (TensorPack α paramShapes))) trainer.gradFn inputs
        Curried.uncurry (α := δ) (ss := dataInputShapes)
          (β := IO (TensorPack α paramShapes)) withData dataInputs
  | true =>
      exact do
        let withData := Curried.uncurry (α := α) (ss := inputShapes)
          (β := Curried.Fn δ dataInputShapes
            (IO (Tensor α .scalar × TensorPack α paramShapes))) trainer.diffFn inputs
        let (lossValue, gradient) ← Curried.uncurry (α := δ) (ss := dataInputShapes)
          (β := IO (Tensor α .scalar × TensorPack α paramShapes)) withData dataInputs
        pure (gradient, lossValue)

/--
Apply the trainer's SGD update to packed inputs.

Set `loss := true` to return the loss used for the update. The default uses the backend's
update operation without reading the objective value back.
-/
def step {α δ : Type} [Storage α] [Storage δ]
    {paramShapes inputShapes dataInputShapes : List Shape}
    (trainer : ScalarTrainer α δ paramShapes inputShapes dataInputShapes)
    (learningRate : α) (inputs : TensorPack α inputShapes)
    (dataInputs : TensorPack δ dataInputShapes) (loss : Bool := false) :
    IO (match loss with | false => Unit | true => Tensor α .scalar) := by
  cases loss with
  | false =>
      exact
        let withData := Curried.uncurry (α := α) (ss := inputShapes)
          (β := Curried.Fn δ dataInputShapes (IO Unit)) (trainer.stepFn learningRate) inputs
        Curried.uncurry (α := δ) (ss := dataInputShapes) (β := IO Unit) withData dataInputs
  | true =>
      exact
        let withData := Curried.uncurry (α := α) (ss := inputShapes)
          (β := Curried.Fn δ dataInputShapes (IO (Tensor α .scalar)))
          (trainer.stepWithLossFn learningRate) inputs
        Curried.uncurry (α := δ) (ss := dataInputShapes)
          (β := IO (Tensor α .scalar)) withData dataInputs

end ScalarTrainer

end Runtime.Autograd.Torch
