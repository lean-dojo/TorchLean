/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Model.Module.Objective

/-!
# Module Evaluators

Reusable no-gradient evaluators over live module state, including scalar-objective evaluation.
-/

@[expose] public section


namespace Runtime
namespace Autograd
namespace Model

open Spec TorchLean
open TorchLean TorchLean.Tensor
open Proofs.Autograd.Algebra

namespace Module

/--
A reusable no-gradient evaluator over existing model state.

The evaluator accepts differentiable tensors and non-differentiable data tensors separately. It can
therefore run both scalar objectives and tensor-valued forward programs without rebuilding a
session for each input batch.
-/
structure Evaluator (α β : Type) [TorchLean.Storage α] [TorchLean.Storage β]
    (stateShapes inputShapes dataInputShapes : List Shape)
    (outputShape : Shape) where
  /-- Evaluate one input pack and return the program output. -/
  evaluate : Torch.Curried.Fn α inputShapes
    (Torch.Curried.Fn β dataInputShapes (IO (Tensor α outputShape)))

/-- Scalar-output specialization of `Evaluator`. -/
abbrev ObjectiveEvaluator (α β : Type) [TorchLean.Storage α] [TorchLean.Storage β]
    (stateShapes inputShapes : List Shape)
    (dataInputShapes : List Shape := []) :=
  Evaluator α β stateShapes inputShapes dataInputShapes Shape.scalar

namespace Evaluator

/-- Apply a reusable evaluator to shape-indexed ordinary and discrete inputs. -/
def run {α β : Type} [TorchLean.Storage α] [TorchLean.Storage β]
    {stateShapes inputShapes dataInputShapes : List Shape}
    {outputShape : Shape}
    (evaluator : Evaluator α β stateShapes inputShapes dataInputShapes outputShape)
    (xs : TorchLean.TensorPack α inputShapes)
    (dataInputs : TorchLean.TensorPack β dataInputShapes) :
    IO (Tensor α outputShape) :=
  let withData := Torch.Curried.uncurry (α := α) (ss := inputShapes)
    (β := Torch.Curried.Fn β dataInputShapes (IO (Tensor α outputShape)))
    evaluator.evaluate xs
  Torch.Curried.uncurry (α := β) (ss := dataInputShapes)
    (β := IO (Tensor α outputShape)) withData dataInputs

/--
Create a reusable no-gradient evaluator for an execution-polymorphic program.

The evaluator shares the supplied live parameter objects. Its eager session is reset after every
call, so validation and generation do not retain one execution graph per input batch.
-/
def withState
    {α β : Type} [TorchLean.Storage α] [TorchLean.Storage β]
    [Context α]
    [tensorTransfer : Runtime.Autograd.Torch.TensorTransfer α]
    {stateShapes inputShapes dataInputShapes : List Shape} {outputShape : Shape}
    (program : ProgramWithDataInputs α β (stateShapes ++ inputShapes) dataInputShapes outputShape)
    (options : Torch.Config)
    (state : Torch.ParamList α stateShapes)
    (validateDataInputs : TorchLean.TensorPack β dataInputShapes → Except String Unit :=
      fun _ => pure ()) (rngCounter : Option (IO.Ref Nat) := none) :
    IO (Evaluator α β stateShapes inputShapes dataInputShapes outputShape) := do
  let options := { options with gradEnabled := false }
  let sess ← Torch.Internal.EagerSession.new (α := α) options
  let sess := match rngCounter with
    | some counter => { sess with rngCounter := counter }
    | none => sess
  let programEager := program (m := Torch.Internal.EagerM α)
  let evaluate : Torch.Curried.Fn α inputShapes
      (Torch.Curried.Fn β dataInputShapes (IO (Tensor α outputShape))) :=
    Torch.Curried.curry (α := α) (ss := inputShapes)
      (β := Torch.Curried.Fn β dataInputShapes (IO (Tensor α outputShape))) (fun xs =>
        Torch.Curried.curry (α := β) (ss := dataInputShapes)
          (β := IO (Tensor α outputShape)) (fun dataInputs => do
            match validateDataInputs dataInputs with
            | .error message => throw <| IO.userError message
            | .ok () => pure ()
            sess.resetTape
            try
              let outRef ← (do
                let pRefs ← Torch.Internal.useParams (α := α) (ss := stateShapes) state
                let xRefs ← Torch.Internal.useInputs (α := α) (ss := inputShapes) xs
                let allRefs := Torch.RefList.append
                  (ss₁ := stateShapes) (ss₂ := inputShapes) pRefs xRefs
                let withData := Torch.CurriedRef.uncurry
                  (ss := stateShapes ++ inputShapes) programEager allRefs
                Torch.CurriedRef.uncurryPack
                  (α := β) (ss := dataInputShapes) withData dataInputs) |>.run sess
              Torch.Internal.EagerSession.getValue (α := α) sess outRef
            finally
              sess.resetTape
              if options.usesCuda then
                Runtime.Autograd.Cuda.Buffer.collectGarbage))
  pure { evaluate := evaluate }

end Evaluator

namespace ObjectiveDef

/--
Create a reusable no-gradient evaluator over an existing live parameter list.

This is useful when two definitions share the same parameter layout but differ in execution mode,
for example training and evaluation losses for a model containing dropout. The evaluator shares
the parameter objects and their current backend storage with the training module. Its eager session
is reset after every call, so repeated validation does not retain one execution graph per batch.
-/
def evaluatorWithState
    {α β : Type} [TorchLean.Storage α] [TorchLean.Storage β]
    [Context α]
    [tensorTransfer : Runtime.Autograd.Torch.TensorTransfer α]
    {stateShapes inputShapes dataInputShapes : List Shape}
    (d : ObjectiveDef β stateShapes inputShapes dataInputShapes)
    (options : Torch.Config)
    (state : Torch.ParamList α stateShapes) :
    IO (ObjectiveEvaluator α β stateShapes inputShapes dataInputShapes) :=
  match d.validate with
  | .error message => throw <| IO.userError message
  | .ok () =>
      Evaluator.withState (program := d.loss (α := α)) options state
        (validateDataInputs := d.validateDataInputs)

end ObjectiveDef

end Module

end Model
end Autograd
end Runtime
