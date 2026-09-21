/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Autograd.Differential
public import NN.Proofs.Autograd.Model.Seeding
public import NN.Proofs.Autograd.Runtime.Link.HigherOrderFDeriv
import Batteries.Lean.LawfulMonad
import NN.Tactic.Autograd.Scalar
import all Init.System.IO

/-!
# Pullbacks of higher model derivatives

The public reverse transform seeds input directions while holding model state constant, runs
the checked reverse pass, and returns state and input gradients. The proof connects that complete
IO call to the adjoint derivative of a higher input derivative. It uses the existing flattened
context only at the calculus boundary; callers still receive the usual typed state and tensor.

Successful recording and execution are explicit hypotheses, separate from the first-order graph
certificate, higher-order jet laws, and smoothness needed to interchange derivatives. No theorem
here identifies rounded floating-point execution with exact real differentiation.
-/

public section

open Spec TorchLean Runtime.Autograd.Model Proofs.Autograd
open Runtime.Autograd.Torch (TypedGraphWithData)

namespace TorchLean.autograd.model

/-- Validation and a successful checked graph execution determine the public VJP result.

This execution equation works with every runtime scalar context. It preserves the existing
state/input split and makes no claim that the graph's backward rules are analytic derivatives.
-/
theorem vjp_eq_checked {σ τ : Shape} (model : nn.Sequential σ τ)
    {α : Type} [Storage α] [Context α]
    (state : State model α) (input : Tensor α σ) (seed : Tensor α τ)
    (graph : nn.TypedGraphModel (nn.stateShapes model) σ τ α)
    (valid : nn.validate model = .ok ())
    (lowered : nn.lowerToTypedGraph model (α := α) = pure graph)
    (result : TensorPack α (nn.stateShapes model ++ [σ]) × Tensor α τ)
    (checked : graph.vjpChecked
      ((nn.State.Internal.toTensorPack state).append (TensorPack.singleton input)) () seed =
      .ok result) :
    vjp model state input seed =
      pure (nn.State.Internal.fromTensorPack
        (TensorPack.split (ss₁ := nn.stateShapes model) (ss₂ := [σ]) result.1).1,
        (TensorPack.split (ss₁ := nn.stateShapes model) (ss₂ := [σ]) result.1).2.head) := by
  simp only [nn.lowerToTypedGraph, valid] at lowered
  simp only [vjp, valid, IO.ofExcept, Autodiff.vjp, lowered, pure_bind,
    checked, Runtime.Autograd.okOrThrow]

/-- The public higher-order VJP returns the pullback of the requested iterated input derivative.

The context includes state and input entries. Directions have zero state components, but the
final adjoint differentiates with respect to all entries, giving both parameter and input
gradients. The output cotangent and directions are fixed, including when directions repeat.
-/
@[autograd] theorem derivativeVjp_eq {σ τ : Shape} (model : nn.Sequential σ τ)
    (state : State model ℝ) (input : Tensor ℝ σ) (directions : List (Tensor ℝ σ))
    (seed : Tensor ℝ τ)
    (real : nn.TypedGraphModel (nn.stateShapes model) σ τ ℝ)
    (nested : nn.TypedGraphModel (nn.stateShapes model) σ τ
      (Dual.Nested ℝ directions.length))
    (valid : nn.validate model = .ok ())
    (lowered : nn.lowerToTypedGraph model (α := Dual.Nested ℝ directions.length) = pure nested)
    (sameShapes : nested.nodeShapes = real.nodeShapes)
    (hgraph : Algebra.GraphData.PreservesPullbackJet
      (CtxVec (nn.stateShapes model ++ [σ])) directions.length
      real.data (sameShapes ▸ nested.data))
    (sameOutput : sameShapes ▸ nested.output = real.output)
    (proofGraph : Algebra.Graph (α := ℝ) Unit (nn.stateShapes model ++ [σ]) real.nodeShapes)
    (same : proofGraph.toData = real.data)
    (correct : ∀ y, GraphFDerivCorrectAt (Algebra.Graph.toReal proofGraph ()) y)
    (smooth : ContDiff ℝ (directions.length + 1)
      (fun z => tensorToVec (TypedGraphWithData.forward real (unflattenCtx z) ())))
    (result : TensorPack (Dual.Nested ℝ directions.length) (nn.stateShapes model ++ [σ]) ×
      Tensor (Dual.Nested ℝ directions.length) τ)
    (checked : nested.vjpChecked
      (((nn.State.Internal.toTensorPack state).map (Tensor.map (Dual.Nested.ofPrimal
        directions.length))).append (TensorPack.singleton
          (Dual.Nested.seedTensor (fun i : Fin directions.length => directions[i]) input))) ()
      (Tensor.map (Dual.Nested.ofPrimal directions.length) seed) = .ok result) :
    let ds := fun i : Fin directions.length => flattenCtx
      ((TensorPack.zero (ss := nn.stateShapes model)).append
        (TensorPack.singleton directions[i]))
    let x := flattenCtx ((nn.State.Internal.toTensorPack state).append
      (TensorPack.singleton input))
    let gradient := unflattenCtx
      ((fderiv ℝ (fun y => iteratedFDeriv ℝ directions.length
        (fun z => tensorToVec (TypedGraphWithData.forward real (unflattenCtx z) ()))
        y ds) x).adjoint
          (tensorToVec seed))
    derivativeVjp model state input directions seed = pure
      (nn.State.Internal.fromTensorPack
        (TensorPack.split (ss₁ := nn.stateShapes model) (ss₂ := [σ]) gradient).1,
        (TensorPack.split (ss₁ := nn.stateShapes model) (ss₂ := [σ]) gradient).2.head) := by
  intro ds x gradient
  have hpack : result.1.map Dual.Nested.tangentTensor = gradient := by
    apply TensorPack.ext_getIdx
    intro shape index
    rw [Proofs.getIdx_map]
    exact Runtime.Autograd.Torch.TypedGraphWithData.tangent_vjpChecked_iteratedFDeriv
      real nested sameShapes hgraph sameOutput proofGraph same _ _ _ () correct
      (TensorPack.JetRelated.const_append_seed (nn.State.Internal.toTensorPack state)
        input (fun i : Fin directions.length => directions[i])) smooth seed result checked index
  rw [← hpack]
  unfold derivativeVjp
  dsimp only
  rw [vjp_eq_checked model _ _ _ nested valid lowered result
    (by simpa only [nn.State.map, nn.State.Internal.toTensorPack_fromTensorPack] using checked)]
  simp only [pure_bind, nn.State.map, nn.State.Internal.toTensorPack_fromTensorPack,
    TensorPack.split_map, TensorPack.head_map]

end TorchLean.autograd.model
