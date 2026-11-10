/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Sequential.Execution

/-!
# SGD on Typed Model State

`sgdStep` updates a model state that the caller holds, with the gradients and the learning rate in
the same scalar type. It applies the SGD tensor kernel `SGD.update` and reads the trainable
flags of the original sequential model.
-/

@[expose] public section

namespace TorchLean.nn

/--
Apply one SGD update to the trainable tensors of an immutable model state.

Use the original model passed to `Sequential.lowerToTypedGraph`. `TypedGraphModel.vjp` supplies a
compatible state gradient. Each trainable tensor uses `SGD.update`. Entries whose
`requiresGrad` flag is false are returned unchanged, including persistent buffers, even when their
gradient is nonzero. State shapes and gradient shapes agree by construction. Malformed model
metadata is rejected.

All arithmetic stays in `α`, with the rounding and exceptional-value behavior of its `Context`, and
the learning rate goes to the SGD kernel as given, with no transfer through `Float` or device
storage. The result is a fresh state. Neither input is mutated, no optimizer history is shared,
and the storage-alias contract of the mutable runtime optimizer does not apply.
-/
def sgdStep {σ τ : Shape} {α : Type} [Storage α] [Context α]
    (model : Sequential σ τ) (lr : α)
    (state gradients : TensorPack α (Sequential.stateShapes model)) :
    Except String (TensorPack α (Sequential.stateShapes model)) := do
  Sequential.validate model
  if Sequential.hasBufferUpdates model then
    throw "nn.sgdStep: models with buffer-update hooks are unsupported"
  let flags := Sequential.requiresGrad model
  let rec
    /-- Take one SGD step on each remaining trainable tensor and keep each frozen one, reading the
    flags of `flags` from the given index on. -/
    update : {shapes : List Shape} →
        TensorPack α shapes → TensorPack α shapes → Nat →
        Except String (TensorPack α shapes)
      | [], .nil, .nil, index =>
          if index == flags.size then pure .nil
          else throw "nn.sgdStep: trainable flags do not match the state layout"
      | _ :: _, .cons parameter rest, .cons gradient restGradients, index => do
          let some trainable := flags[index]?
            | throw "nn.sgdStep: trainable flags do not match the state layout"
          let next :=
            if trainable then
              (SGD.update (SGD.init lr parameter) parameter gradient).parameters
            else parameter
          return .cons next (← update rest restGradients (index + 1))
  update state gradients 0

end TorchLean.nn
