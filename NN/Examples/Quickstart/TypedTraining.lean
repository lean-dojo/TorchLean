/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Precision

/-!
# Training with explicit precision

Fit an affine model to one sample using the loss `(prediction - target)^2 / 2`.
The derivative with respect to the prediction is its residual. A typed VJP converts that
residual into parameter gradients, and `nn.sgdStep` applies the existing SGD kernel.

The model has no buffer-update hooks. State, data, gradients, and learning rate use the selected
scalar throughout. Call `run` for a binary128 example, or `fit` with another supported `Context`.
This is CPU software execution; no arbitrary-precision CUDA or checkpoint support is implied.
-/

@[expose] public section

namespace NN.Examples.Quickstart.TypedTraining

open TorchLean
open FloatLib.Floats

/-- The concrete scalar selected for the runnable example. -/
abbrev Binary128 := ExecFloat.Binary (exponentBits := 15) (fractionBits := 112)

/-- A scalar affine model with caller-supplied typed weight and bias. -/
def model : nn.Sequential [1] [1] :=
  nn.build 0 (nn.linear 1 1)

/-- Reuse one typed graph for several SGD updates of half squared error on one sample. -/
def fit {α : Type} [Storage α] [Context α]
    (initial : nn.State α (nn.stateShapes model))
    (input target : Tensor α [1]) (learningRate : α) (steps : Nat) :
    IO (nn.State α (nn.stateShapes model)) := do
  let graph ← nn.lowerToTypedGraph model (α := α) (mode := .train)
  let mut state := initial
  for _ in [:steps] do
    let prediction := nn.TypedGraphModel.forward graph state input
    let residual := Tensor.sub prediction target
    let (gradient, _) := nn.TypedGraphModel.vjp graph state input residual
    state ← match nn.sgdStep model learningRate state gradient with
      | .ok next => pure next
      | .error message => throw <| IO.userError message
  return state

/-- Train from a coefficient that cannot be represented in binary64, then print exact rationals. -/
def run : IO Unit := do
  let initial : nn.State Binary128 (nn.stateShapes model) :=
    nn.State.full (Rat.cast (1 + 1 / (2 ^ 100 : Nat) : Rat))
  let input : Tensor Binary128 [1] := Tensor.full [1] 2
  let trained ← fit initial input (Tensor.zeros [1]) (Rat.cast (1 / 8 : Rat)) 2
  let graph ← nn.lowerToTypedGraph model (α := Binary128)
  let before := nn.TypedGraphModel.forward graph initial input
  let after := nn.TypedGraphModel.forward graph trained input
  IO.println s!"initial prediction: {ExecFloat.Binary.toRat? (before.getScalar ⟨0, by decide⟩)}"
  IO.println s!"trained prediction: {ExecFloat.Binary.toRat? (after.getScalar ⟨0, by decide⟩)}"

end NN.Examples.Quickstart.TypedTraining
