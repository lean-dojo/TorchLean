/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core.Trainer.Types
public import NN.Runtime.Autograd.Torch.Core.Trainer.Eager
public import NN.Runtime.Autograd.Torch.Core.Trainer.Graph

/-!
# Scalar Trainer Construction

Select the execution backend and allocate parameter state. The public contract is in
`Trainer.Types`; the eager and graph implementations stay behind this constructor.
-/

public section

namespace Runtime.Autograd.Torch

open Spec TorchLean TorchLean.Tensor

/--
Build a `ScalarTrainer` from an initial parameter pack and an operation-generic loss definition.

`loss` is written once against the `Ops` interface over a concatenated context
`paramShapes ++ inputShapes`. Depending on `options.execution`, TorchLean either records the loss
once as a typed SSA graph or executes it immediately while building a dynamic tape.
-/
def scalarTrainer {α δ : Type} [TorchLean.Storage α] [TorchLean.Storage δ]
    [Context α] [TensorTransfer α] {paramShapes inputShapes dataInputShapes : List Shape}
    (options : Config := {})
    (initRequiresGrad : Array Bool := Array.replicate paramShapes.length true)
    (validateDataInputs : TorchLean.TensorPack δ dataInputShapes → Except String Unit :=
      fun _ => pure ())
    (loss : ScalarLoss α δ paramShapes inputShapes dataInputShapes) :
    Curried.Fn α paramShapes
      (IO (ScalarTrainer α δ paramShapes inputShapes dataInputShapes)) :=
    Curried.curry (α := α) (ss := paramShapes)
      (β := IO (ScalarTrainer α δ paramShapes inputShapes dataInputShapes))
    (fun initParams => do
      let parameters ← ParamList.ofPackWithRequiresGrad (α := α) initParams initRequiresGrad
      let validateDataInputsIO (inputs : TorchLean.TensorPack δ dataInputShapes) : IO Unit :=
        match validateDataInputs inputs with
        | .error message => throw <| IO.userError message
        | .ok () => pure ()
      match options.execution with
      | .typedGraph => Internal.graphScalarTrainer options parameters validateDataInputsIO loss
      | .eager => Internal.eagerScalarTrainer options parameters validateDataInputsIO loss)

end Runtime.Autograd.Torch
