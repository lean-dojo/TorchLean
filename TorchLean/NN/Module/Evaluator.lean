/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Module.Objective

/-!
# Module Evaluators

Reusable no-gradient evaluators over live module state, including scalar-objective evaluation.
-/

@[expose] public section


namespace TorchLean

open TorchLean.Tensor

/--
A reusable no-gradient evaluator over existing model state.

The evaluator accepts differentiable tensors and non-differentiable data tensors separately. It can
therefore run both scalar objectives and tensor-valued forward programs without rebuilding a
session for each input batch.
-/
structure Evaluator (α β : Type) [Storage α] [Storage β]
    (stateShapes inputShapes dataInputShapes : List Shape)
    (outputShape : Shape) where
  /-- Evaluate one input pack and return the program output. -/
  evaluate : Curried.Function α inputShapes
    (Curried.Function β dataInputShapes (IO (Tensor α outputShape)))

/-- Scalar-output specialization of `Evaluator`. -/
abbrev ObjectiveEvaluator (α β : Type) [Storage α] [Storage β]
    (stateShapes inputShapes : List Shape)
    (dataInputShapes : List Shape := []) :=
  Evaluator α β stateShapes inputShapes dataInputShapes Shape.scalar

namespace Evaluator

/-- Apply a reusable evaluator to shape-indexed ordinary and discrete inputs. -/
def run {α β : Type} [Storage α] [Storage β]
    {stateShapes inputShapes dataInputShapes : List Shape}
    {outputShape : Shape}
    (evaluator : Evaluator α β stateShapes inputShapes dataInputShapes outputShape)
    (xs : TensorPack α inputShapes)
    (dataInputs : TensorPack β dataInputShapes) :
    IO (Tensor α outputShape) :=
  let withData := CurriedRef.uncurry (Ref := (Tensor α ·)) (ss := inputShapes)
    (β := Curried.Function β dataInputShapes (IO (Tensor α outputShape)))
    evaluator.evaluate xs
  CurriedRef.uncurry (Ref := (Tensor β ·)) (ss := dataInputShapes)
    (β := IO (Tensor α outputShape)) withData dataInputs

/--
Create a reusable no-gradient evaluator for an execution-polymorphic program.

The evaluator shares the supplied live parameter objects. Its eager session is reset after every
call, so validation and generation do not retain one execution graph per input batch.
-/
def create
    {α β : Type} [Storage α] [Storage β]
    [Context α]
    [tensorTransfer : TensorTransfer α]
    {stateShapes inputShapes dataInputShapes : List Shape} {outputShape : Shape}
    (program : Program.Data α β (stateShapes ++ inputShapes) dataInputShapes outputShape)
    (options : Session.Config)
    (state : ParamList α stateShapes)
    (validateDataInputs : TensorPack β dataInputShapes → Except String Unit :=
      fun _ => pure ()) (rngCounter : Option (IO.Ref Nat) := none) :
    IO (Evaluator α β stateShapes inputShapes dataInputShapes outputShape) := do
  let sess ← Session.create (α := α) options (gradEnabled := false)
  let sess := match rngCounter with
    | some counter => { sess with rngCounter := counter }
    | none => sess
  let programEager := program (m := SessionM α)
  let evaluate : Curried.Function α inputShapes
      (Curried.Function β dataInputShapes (IO (Tensor α outputShape))) :=
    CurriedRef.curry (Ref := (Tensor α ·)) (ss := inputShapes)
      (β := Curried.Function β dataInputShapes (IO (Tensor α outputShape))) (fun xs =>
        CurriedRef.curry (Ref := (Tensor β ·)) (ss := dataInputShapes)
          (β := IO (Tensor α outputShape)) (fun dataInputs => do
            IO.ofExcept (validateDataInputs dataInputs)
            sess.resetTape
            try
              let outRef ← (do
                let pRefs ← ParamList.use (α := α) (ss := stateShapes) state
                let xRefs ← Session.useInputs (α := α) (ss := inputShapes) xs
                let allRefs := Pack.append
                  (ss₁ := stateShapes) (ss₂ := inputShapes) pRefs xRefs
                let withData := CurriedRef.uncurry
                  (ss := stateShapes ++ inputShapes) programEager allRefs
                CurriedRef.uncurry
                  (Ref := (Tensor β ·)) (ss := dataInputShapes) withData dataInputs) |>.run sess
              Session.getValue (α := α) sess outRef
            finally
              sess.resetTape
              if sess.options.usesCUDA then
                LibTorch.Buffer.collectGarbage))
  pure { evaluate := evaluate }

end Evaluator

namespace ObjectiveDef

/--
Create a reusable no-gradient evaluator for the loss of `d` over an existing live parameter list.

It suits two definitions that share a parameter layout and differ in train/eval behavior, such as
the training and evaluation losses of a model with dropout. The evaluator shares the parameter
objects and their current backend storage with the training module. It runs `d.validate` once
and checks every data pack with `d.validateDataInputs`. Its eager session is reset after every
call, so repeated validation keeps no execution graph per batch.
-/
def evaluator
    {α β : Type} [Storage α] [Storage β]
    [Context α]
    [tensorTransfer : TensorTransfer α]
    {stateShapes inputShapes dataInputShapes : List Shape}
    (d : ObjectiveDef β stateShapes inputShapes dataInputShapes)
    (options : Session.Config)
    (state : ParamList α stateShapes) :
    IO (ObjectiveEvaluator α β stateShapes inputShapes dataInputShapes) := do
  IO.ofExcept d.validate
  Evaluator.create (program := d.loss (α := α)) options state
    (validateDataInputs := d.validateDataInputs)

end ObjectiveDef

end TorchLean
