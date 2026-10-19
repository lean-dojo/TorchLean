/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core.Trainer.Types
public import NN.Runtime.Autograd.Torch.Core.Trainer.Eager

/-!
# Scalar Trainer Construction

Allocate parameter state and build the eager trainer. The public contract is in `Trainer.Types`.
-/

public section

namespace Runtime.Autograd.Torch

open Spec TorchLean TorchLean.Tensor

/--
Build a `ScalarTrainer` from an initial parameter pack and an operation-generic loss definition.

`loss` is written once against the `Ops` interface over a concatenated context
`paramShapes ++ inputShapes`. Each step executes it on the autograd tape, on the CPU or on CUDA
as `options.device` selects.
-/
def scalarTrainer {α δ : Type} [TorchLean.Storage α] [TorchLean.Storage δ]
    [Context α] [TensorTransfer α] {paramShapes inputShapes dataInputShapes : List Shape}
    (options : Config := {})
    (initRequiresGrad : Array Bool := Array.replicate paramShapes.length true)
    (validateDataInputs : TorchLean.TensorPack δ dataInputShapes → Except String Unit :=
      fun _ => pure ())
    (loss : ScalarLoss α δ paramShapes inputShapes dataInputShapes) :
    Curried.Function α paramShapes
      (IO (ScalarTrainer α δ paramShapes inputShapes dataInputShapes)) :=
    Curried.curry (α := α) (ss := paramShapes)
      (β := IO (ScalarTrainer α δ paramShapes inputShapes dataInputShapes))
    (fun initParams => do
      let parameters ← ParamList.ofPackWithRequiresGrad (α := α) initParams initRequiresGrad
      let validateDataInputsIO (inputs : TorchLean.TensorPack δ dataInputShapes) : IO Unit :=
        IO.ofExcept (validateDataInputs inputs)
      Internal.eagerScalarTrainer options parameters validateDataInputsIO loss)

end Runtime.Autograd.Torch
