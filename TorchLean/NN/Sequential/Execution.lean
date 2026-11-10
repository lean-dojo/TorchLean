/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Builder.Layer

/-!
# Executing Sequential Models

Three operations on the typed graphs lowered from shape-checked sequential models:
`TypedGraphModel.forward`, `TypedGraphModel.jvp` and `TypedGraphModel.vjp`.
-/

@[expose] public section

namespace TorchLean

namespace nn

/-- A typed graph whose inputs are model state followed by one model input. -/
abbrev TypedGraphModel (stateShapes : List Shape) (σ τ : Shape) (α : Type)
    [TorchLean.Storage α] :=
  TypedGraph α (stateShapes ++ [σ]) τ

namespace TypedGraphModel

/-- Evaluate a lowered model on the given model state and one input tensor. -/
def forward {σ τ : Shape} {α : Type} [TorchLean.Storage α]
    {stateShapes : List Shape}
    (model : TypedGraphModel stateShapes σ τ α)
    (state : TensorPack α stateShapes)
    (input : Tensor α σ) : Tensor α τ :=
  TypedGraph.WithData.forward model (d := ()) <|
    Pack.append (ss₁ := stateShapes) (ss₂ := [σ])
      state (Pack.singleton input)

/--
Evaluate a Jacobian-vector product with separate tangents for the model state and input.
-/
def jvp {σ τ : Shape} {α : Type} [TorchLean.Storage α]
    {stateShapes : List Shape}
    (model : TypedGraphModel stateShapes σ τ α)
    (state stateTangent : TensorPack α stateShapes)
    (input inputTangent : Tensor α σ) : Tensor α τ :=
  TypedGraph.WithData.jvp model (d := ())
    (Pack.append (ss₁ := stateShapes) (ss₂ := [σ])
      state (Pack.singleton input))
    (Pack.append (ss₁ := stateShapes) (ss₂ := [σ])
      stateTangent
      (Pack.singleton inputTangent))

/--
Vector-Jacobian product of a lowered model, which pulls `outputGradient` back to the pair
`(stateGrad, inputGrad)`.
-/
def vjp {σ τ : Shape} {α : Type} [TorchLean.Storage α] [Add α] [Zero α]
    {stateShapes : List Shape}
    (model : TypedGraphModel stateShapes σ τ α)
    (state : TensorPack α stateShapes)
    (input : Tensor α σ) (outputGradient : Tensor α τ) :
    TensorPack α stateShapes × Tensor α σ :=
  let gradients := TypedGraph.WithData.vjp model (d := ())
    (Pack.append (ss₁ := stateShapes) (ss₂ := [σ])
      state (Pack.singleton input))
    outputGradient
  let (stateGrad, inputGrads) := Pack.split
    (ss₁ := stateShapes) (ss₂ := [σ]) gradients
  (stateGrad, Pack.head inputGrads)

end TypedGraphModel

end nn

end TorchLean
