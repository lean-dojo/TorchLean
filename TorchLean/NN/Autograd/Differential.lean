/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Autograd.Model
public import TorchLean.NN.Sequential.Execution
public import TorchLean.NN.Dual.Nested

/-!
# Differentiating neural fields

Coordinate derivatives of a model remain functions of its parameters.
`autograd.model.derivative` evaluates them, and `autograd.model.derivativeVJP` pulls a cotangent
through them, which lets a PDE residual contribute to a training gradient. Directions are
arbitrary tensors of the input shape, so a derivative can follow any combination of input
coordinates. Repeated directions compute higher derivatives, and different directions compute
mixed derivatives.

The implementation nests forward-mode dual scalars around the existing model evaluator and
reverse pass. It does not approximate derivatives by finite differences. Directions and output
cotangents are held constant during differentiation. At nonsmooth points the scalar and graph
rules keep their branch conventions, so the result there follows those conventions and need not
be a classical derivative of any order.
-/

@[expose] public section

namespace TorchLean.autograd.model

/--
Evaluate an iterated input-directional derivative of a model in evaluation mode.

An empty leading axis evaluates the model itself. For two rows `[v, w]`, the result is
`D_w D_v model(state, input)`, so `[v, v]` gives the second derivative along `v`. State is held
fixed, and all directions are constant vectors. Every direction has the entire input shape, so the
operation also covers vector-valued fields, multiple spatial coordinates, and batched input
layouts. It differentiates the forward evaluation over nested dual numbers, not the registered JVP
rules, so it can differ from `jacfwdInput` and the model VJP: a frozen reference keeps its tangent,
and a tie of three or more entries in a max or min reduction weights the tangents by fold position
(1/4, 1/4 and 1/2 for three entries) instead of taking their mean. It evaluates the graph with
`TypedGraph.lowerToTapeChecked`, which runs each node's `NodeData.validate` check before the node,
as `vjp`, `derivativeVJP` and `jacrev` do, and throws the first failure, such as a zero
`smooth_max_pool` beta.
-/
def derivative {σ τ : Shape} {order : Nat} (model : nn.Sequential σ τ)
    {α : Type} [Storage α] [Context α]
    (state : State model α) (input : Tensor α σ)
    (directions : Tensor α (Shape.dim order σ)) : IO (Tensor α τ) := do
  let graph ← nn.Sequential.lowerToTypedGraph model (α := Dual.Nested α order)
  let state := state.map (Tensor.map (Dual.Nested.ofPrimal order))
  let input := Dual.Nested.seedTensor (fun i : Fin order => directions[i]) input
  let (_, context) ← IO.ofExcept
    (TypedGraph.lowerToTapeChecked graph.data (Pack.append state (Pack.singleton input)) ())
  pure (Dual.Nested.tangentTensor (TensorPack.getIdx context graph.output))

/--
Pull an output cotangent through an iterated input derivative of a model.

Returns `(stateGradient, inputGradient)` for the scalar pairing of `outputGradient` with
`derivative model state input directions`. Use the state gradient to train on PDE residuals. The
input gradient differentiates the same pairing with respect to collocation coordinates. An empty
leading axis gives the ordinary model VJP. The cotangent and directions are constants, even when
the caller computed them from the current residual or coordinates.
-/
def derivativeVJP {σ τ : Shape} {order : Nat} (model : nn.Sequential σ τ)
    {α : Type} [Storage α] [Context α]
    (state : State model α) (input : Tensor α σ)
    (directions : Tensor α (Shape.dim order σ)) (outputGradient : Tensor α τ) :
    IO (State model α × Tensor α σ) := do
  let state := state.map (Tensor.map (Dual.Nested.ofPrimal order))
  let input := Dual.Nested.seedTensor (fun i : Fin order => directions[i]) input
  let outputGradient := Tensor.map (Dual.Nested.ofPrimal order) outputGradient
  let (stateGrad, inputGrad) ← vjp model state input outputGradient
  pure (stateGrad.map Dual.Nested.tangentTensor, Dual.Nested.tangentTensor inputGrad)

end TorchLean.autograd.model
