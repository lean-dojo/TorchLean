/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Autograd.Model
public import NN.API.Neural.Execution
public import NN.Runtime.Autograd.Model.Dual.Nested

/-!
# Differentiating neural fields

Coordinate derivatives of a model remain functions of its parameters. `model.derivativeVjp`
pulls a cotangent through those derivatives, which lets a PDE residual contribute to a training
gradient. Directions are arbitrary tensors of the input shape, not a fixed list of spatial axes.
Repeated directions compute higher derivatives; different directions compute mixed derivatives.

The implementation nests forward-mode dual scalars around the existing model evaluator and
reverse pass. It does not approximate derivatives by finite differences. Directions and output
cotangents are held constant during differentiation. At nonsmooth points the scalar and graph
rules retain their existing branch conventions; these executable transforms are not a theorem
that a nonsmooth model has classical derivatives of every order.
-/

@[expose] public section

namespace TorchLean.autograd.model

open Runtime.Autograd.Model

/--
Evaluate an iterated input-directional derivative of a model in evaluation mode.

An empty direction list evaluates the model itself. For `[v, w]`, the result is
`D_w D_v model(state, input)`. State is held fixed, and all directions are constant vectors.
Every direction has the entire input shape, so the operation also covers vector-valued fields,
multiple spatial coordinates, and batched input layouts.
-/
def derivative {σ τ : Shape} (model : nn.Sequential σ τ)
    {α : Type} [Storage α] [Context α]
    (state : State model α) (input : Tensor α σ)
    (directions : List (Tensor α σ)) : IO (Tensor α τ) := do
  let order := directions.length
  let graph ← nn.lowerToTypedGraph model (α := Dual.Nested α order)
  let state := state.map (Tensor.map (Dual.Nested.ofPrimal order))
  let input := Dual.Nested.seedTensor (fun i : Fin order => directions[i]) input
  pure (Dual.Nested.tangentTensor (nn.TypedGraphModel.forward graph state input))

/--
Pull an output cotangent through an iterated input derivative of a model.

Returns `(stateGradient, inputGradient)` for the scalar pairing of `outputGradient` with
`derivative model state input directions`. Use the state gradient to train on PDE residuals;
the input gradient differentiates the same pairing with respect to collocation coordinates.
An empty direction list is the ordinary model VJP. The cotangent and directions are constants,
even when the caller computed them from the current residual or coordinates.
-/
def derivativeVjp {σ τ : Shape} (model : nn.Sequential σ τ)
    {α : Type} [Storage α] [Context α]
    (state : State model α) (input : Tensor α σ)
    (directions : List (Tensor α σ)) (outputGradient : Tensor α τ) :
    IO (State model α × Tensor α σ) := do
  let order := directions.length
  let state := state.map (Tensor.map (Dual.Nested.ofPrimal order))
  let input := Dual.Nested.seedTensor (fun i : Fin order => directions[i]) input
  let outputGradient := Tensor.map (Dual.Nested.ofPrimal order) outputGradient
  let (stateGrad, inputGrad) ← vjp model state input outputGradient
  pure (stateGrad.map Dual.Nested.tangentTensor, Dual.Nested.tangentTensor inputGrad)

end TorchLean.autograd.model
