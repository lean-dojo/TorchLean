/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Neural.Execution

/-!
# SGD on explicit typed model state

The caller supplies state, gradients, and a learning rate in the same scalar type. Updates use
the existing SGD tensor kernel and the original sequential model's trainable-state flags.
-/

@[expose] public section

namespace TorchLean.nn

/--
Apply one SGD update to the trainable entries of immutable model state.

Use the original model passed to `lowerToTypedGraph`; `TypedGraphModel.vjp` supplies a compatible
state gradient. Each trainable tensor uses `Optim.SGD.update`. Entries whose `requiresGrad` flag
is false are returned unchanged, including persistent buffers, even when their gradient is nonzero.
State shapes and gradient shapes agree by construction; malformed model metadata is rejected.

Models with buffer-update hooks are rejected. This entrypoint does not run their mode-dependent
updates, including BatchNorm running statistics. It supports models without those hooks, with
the forward mode chosen explicitly when lowering the graph.

All arithmetic stays in `α`, with the rounding and exceptional-value behavior of its `Context`.
The learning rate is passed directly to the canonical SGD kernel. There is no transfer through
`Float` or device storage. The result is a new value: this operation does not mutate either input,
share optimizer history, or implement the mutable runtime optimizer's storage-alias contract.
-/
def sgdStep {σ τ : Shape} {α : Type} [Storage α] [Context α]
    (model : Sequential σ τ) (learningRate : α)
    (state gradients : State α (stateShapes model)) :
    Except String (State α (stateShapes model)) := do
  validate model
  if hasBufferUpdates model then
    throw "nn.sgdStep: models with buffer-update hooks are unsupported"
  let flags := requiresGrad model
  let rec update : {shapes : List Shape} →
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
            (Optim.SGD.update (Optim.SGD.init learningRate parameter) parameter gradient).parameters
          else parameter
        return .cons next (← update rest restGradients (index + 1))
  let tensors ← update
    (State.Internal.toTensorPack state) (State.Internal.toTensorPack gradients) 0
  return State.Internal.fromTensorPack tensors

end TorchLean.nn
