/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

-- This module supplies public namespace exports used by downstream consumers. Import shaking
-- cannot see those downstream lookups, so keep the marked imports.
module -- shake: keep-downstream

public import NN.API.Neural.State -- shake: keep
public import NN.Runtime.Autograd.Torch.Core.TypedGraph -- shake: keep
public import NN.API.Neural.Builders -- shake: keep
public import NN.API.Runtime -- shake: keep
public import NN.Tensor -- shake: keep

/-!
# Executing Sequential Models

Execution operations for shape-checked sequential models and the typed graphs obtained by lowering
them.
-/

@[expose] public section

namespace TorchLean

namespace nn

export Runtime.Autograd.Model.Layers.Seq (forward lowerToTypedGraph)

/-- A typed graph whose inputs are model state followed by one model input. -/
abbrev TypedGraphModel (stateShapes : List Shape) (σ τ : Shape) (α : Type)
    [TorchLean.Storage α] :=
  Runtime.Autograd.Torch.TypedGraph α (stateShapes ++ [σ]) τ

namespace TypedGraphModel

/-- Evaluate a lowered model with explicit model state and one input tensor. -/
def forward {σ τ : Shape} {α : Type} [TorchLean.Storage α]
    {stateShapes : List Shape}
    (model : TypedGraphModel stateShapes σ τ α)
    (state : State α stateShapes)
    (input : Tensor α σ) : Tensor α τ :=
  Runtime.Autograd.Torch.TypedGraph.forward model <|
    TensorPack.append (ss₁ := stateShapes) (ss₂ := [σ])
      (State.Internal.toTensorPack state) (TensorPack.singleton input)

/--
Evaluate a Jacobian-vector product with separate tangents for the model state and input.
-/
def jvp {σ τ : Shape} {α : Type} [TorchLean.Storage α]
    {stateShapes : List Shape}
    (model : TypedGraphModel stateShapes σ τ α)
    (state stateTangent : State α stateShapes)
    (input inputTangent : Tensor α σ) : Tensor α τ :=
  Runtime.Autograd.Torch.TypedGraph.jvp model
    (TensorPack.append (ss₁ := stateShapes) (ss₂ := [σ])
      (State.Internal.toTensorPack state) (TensorPack.singleton input))
    (TensorPack.append (ss₁ := stateShapes) (ss₂ := [σ])
      (State.Internal.toTensorPack stateTangent)
      (TensorPack.singleton inputTangent))

/--
Evaluate a vector-Jacobian product.

The returned pair is `(stateGrad, inputGrad)`.
-/
def vjp {σ τ : Shape} {α : Type} [TorchLean.Storage α] [Add α] [Zero α]
    {stateShapes : List Shape}
    (model : TypedGraphModel stateShapes σ τ α)
    (state : State α stateShapes)
    (input : Tensor α σ) (outputGradient : Tensor α τ) :
    State α stateShapes × Tensor α σ :=
  let gradients := Runtime.Autograd.Torch.TypedGraph.vjpWithSeed model
    (TensorPack.append (ss₁ := stateShapes) (ss₂ := [σ])
      (State.Internal.toTensorPack state) (TensorPack.singleton input))
    outputGradient
  let (stateGrad, inputGrads) := TensorPack.split
    (α := α) (ss₁ := stateShapes) (ss₂ := [σ]) gradients
  (State.Internal.fromTensorPack stateGrad, TensorPack.head inputGrads)

end TypedGraphModel

end nn

end TorchLean
