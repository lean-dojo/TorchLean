/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Model.Composition
public import NN.Proofs.Autograd.Model.Reverse
import Batteries.Lean.LawfulMonad
import NN.Tactic.Autograd.Scalar

/-!
# Higher derivatives through the model API

The public transforms lower a model, embed fixed state, and seed input directions. The forward
theorem connects coefficient extraction to mathlib's iterated derivative. `Model.Reverse` also
connects the returned state and input gradients to its pullback. Their hypotheses identify the
recorded graph and certify its operations; successful recording alone does not imply
differentiability.
-/

public section

open Spec TorchLean Runtime.Autograd.Model Proofs.Autograd

/-- A model with no state tensors evaluates on its input alone, regardless of the empty state value.
This equation lets clients reason about execution without unpacking the opaque state wrapper.
-/
theorem TorchLean.nn.TypedGraphModel.forward_empty {σ τ : Shape} {α : Type} [Storage α]
    (model : nn.TypedGraphModel [] σ τ α) (state : nn.State α []) (input : Tensor α σ) :
    nn.TypedGraphModel.forward model state input =
      Runtime.Autograd.Torch.TypedGraph.forward model (TensorPack.singleton input) := by
  unfold nn.TypedGraphModel.forward
  cases nn.State.Internal.toTensorPack state
  rfl

namespace TorchLean.autograd.model

/-- Successful certified lowering makes `model.derivative` compute the iterated Fréchet derivative.

State is fixed, directions can repeat, and the list length is the derivative order. `real` supplies
the reference function, while the jet certificate relates every recorded operation to the nested
graph actually returned by the IO lowering call. No assumption identifies native floats with reals.
-/
@[autograd] theorem derivative_eq {σ τ : Shape} (model : nn.Sequential σ τ)
    (state : State model ℝ) (input : Tensor ℝ σ) (directions : List (Tensor ℝ σ))
    (real : nn.TypedGraphModel (nn.stateShapes model) σ τ ℝ)
    (nested : nn.TypedGraphModel (nn.stateShapes model) σ τ
      (Dual.Nested ℝ directions.length))
    (lowered : nn.lowerToTypedGraph model (α := Dual.Nested ℝ directions.length) = pure nested)
    (sameShapes : nested.nodeShapes = real.nodeShapes)
    (hgraph : Algebra.GraphData.PreservesJet (Tensor ℝ σ) directions.length
      real.data (sameShapes ▸ nested.data))
    (sameOutput : sameShapes ▸ nested.output = real.output) :
    derivative model state input directions =
      pure (iteratedFDeriv ℝ directions.length
        (fun y => nn.TypedGraphModel.forward real state y) input
        (fun i : Fin directions.length => directions[i])) := by
  simp only [derivative, lowered, pure_bind]
  congr 1
  let ds := fun i : Fin directions.length => directions[i]
  have hinputs := TensorPack.JetRelated.append
    (TensorPack.JetRelated.const ds input (nn.State.Internal.toTensorPack state))
    (TensorPack.JetRelated.singleton_seed ds input)
  simpa only [nn.TypedGraphModel.forward, Runtime.Autograd.Torch.TypedGraph.forward,
    nn.State.map, nn.State.Internal.toTensorPack_fromTensorPack] using
    Runtime.Autograd.Torch.TypedGraphWithData.tangent_forward real nested sameShapes hgraph
      sameOutput ds input
      (fun y => TensorPack.append (nn.State.Internal.toTensorPack state) (TensorPack.singleton y))
      _ () hinputs

end TorchLean.autograd.model
