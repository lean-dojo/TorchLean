/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.ScalarTrainer
public import TorchLean.NN.ScalarTrainer.Eager

/-!
# Scalar Trainer Construction

Allocate parameter state and build the eager trainer. The public contract is in
`TorchLean.NN.ScalarTrainer`.
-/

public section

namespace TorchLean

open TorchLean.Tensor

/--
Build a `ScalarTrainer` from initial parameters, passed curried in `paramShapes` order, and an
operation-generic loss definition.

`loss` is written once against the `Ops` interface over the concatenated context
`paramShapes ++ inputShapes`, followed by the data inputs. Each step executes it on the autograd
tape, on the CPU or on CUDA as `options.device` selects. Parameters whose `initRequiresGrad` entry
is `false` are persistent buffers.
-/
def ScalarTrainer.create {α δ : Type} [Storage α] [Storage δ]
    [Context α] [TensorTransfer α] {paramShapes inputShapes dataInputShapes : List Shape}
    (options : Session.Config := {})
    (initRequiresGrad : Array Bool := Array.replicate paramShapes.length true)
    (validateDataInputs : TensorPack δ dataInputShapes → Except String Unit :=
      fun _ => pure ())
    (loss : ScalarLoss α δ paramShapes inputShapes dataInputShapes) :
    Curried.Function α paramShapes
      (IO (ScalarTrainer α δ paramShapes inputShapes dataInputShapes)) :=
    CurriedRef.curry (Ref := (Tensor α ·)) (ss := paramShapes)
      (β := IO (ScalarTrainer α δ paramShapes inputShapes dataInputShapes))
    (fun initParams => do
      let parameters ← ParamList.ofPackWithRequiresGrad (α := α) initParams initRequiresGrad
      let validateDataInputsIO (inputs : TensorPack δ dataInputShapes) : IO Unit :=
        IO.ofExcept (validateDataInputs inputs)
      eager options parameters validateDataInputsIO loss)

end TorchLean
