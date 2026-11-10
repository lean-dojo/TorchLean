/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Autograd.Differential
public import TorchLean.NN.Autograd.Model.Seeding
public import TorchLean.NN.TypedGraph.WithData.HigherOrder
import Batteries.Lean.LawfulMonad
import TorchLean.NN.Data.Monad
import TorchLean.NN.Tactic.Autograd.Scalar
import all Init.System.IO

/-!
# Higher derivatives through the model API

The public transforms lower a model, hold its state fixed, and seed input directions.
`derivative_eq` connects the forward transform's coefficient extraction to Mathlib's iterated
derivative. `derivativeVJP_eq` connects the reverse transform, which runs the checked reverse pass
and returns state and input gradients, to the adjoint of the derivative of a higher input
derivative. The flattened context appears only at the calculus boundary, and callers receive the
usual typed state and tensor. A model without state tensors evaluates on its input alone
(`TorchLean.nn.TypedGraphModel.forward_empty`).

Successful lowering and checked execution enter as hypotheses, separate from the jet laws, the
first-order graph certificate and the smoothness needed to interchange derivatives. The derivative
theorems work over `ℝ` and the nested dual numbers `Dual.Nested ℝ order`, where the arithmetic is
exact.
-/

public section

namespace TorchLean

open TensorPack Tensor Tape.RealGraph
open TypedGraph (WithData)

/-- A model with no state tensors evaluates on its input alone: every state over the empty shape
list is the empty pack.
-/
theorem nn.TypedGraphModel.forward_empty {σ τ : Shape} {α : Type} [Storage α]
    (model : nn.TypedGraphModel [] σ τ α) (state : TensorPack α []) (input : Tensor α σ) :
    nn.TypedGraphModel.forward model state input =
      WithData.forward model (Pack.singleton input) () := by
  unfold nn.TypedGraphModel.forward
  cases state
  rfl

namespace autograd.model

/-- Validation and a successful checked graph execution determine the public VJP result.

The equation holds in every runtime scalar context and keeps the state/input split. It is a
statement about execution, and analytic derivatives enter only through the hypotheses of
`derivativeVJP_eq`.
-/
theorem vjp_eq_checked {σ τ : Shape} (model : nn.Sequential σ τ)
    {α : Type} [Storage α] [Context α]
    (state : State model α) (input : Tensor α σ) (seed : Tensor α τ)
    (graph : nn.TypedGraphModel (nn.Sequential.stateShapes model) σ τ α)
    (valid : nn.Sequential.validate model = .ok ())
    (lowered : nn.Sequential.lowerToTypedGraph model (α := α) = pure graph)
    (result : TensorPack α (nn.Sequential.stateShapes model ++ [σ]) × Tensor α τ)
    (checked : graph.vjpChecked
      (state.append (Pack.singleton input)) () seed =
      .ok result) :
    vjp model state input seed =
      pure ((Pack.split (ss₁ := nn.Sequential.stateShapes model) (ss₂ := [σ]) result.1).1,
        (Pack.split (ss₁ := nn.Sequential.stateShapes model) (ss₂ := [σ])
          result.1).2.head) := by
  simp only [nn.Sequential.lowerToTypedGraph, valid] at lowered
  simp only [vjp, valid, IO.ofExcept, Autodiff.vjp, Autodiff.pullback, lowered, pure_bind,
    checked]

/-- Successful certified lowering makes `model.derivative` compute the iterated Fréchet derivative,
after the checks of the nested graph's nodes.

State is held fixed, directions may repeat, and the leading axis of `directions` is the derivative
order. `real` supplies the reference function, and the jet certificate `hgraph` relates every
recorded operation to the nested graph that the IO lowering call returned. `model.derivative`
throws the first `NodeData.validate` failure of that graph on the seeded inputs, which
`TypedGraph.lowerToTapeChecked` reports, and otherwise returns the derivative. Everything is stated
over exact reals, so native float execution needs its own rounding argument.
-/
@[autograd] theorem derivative_eq {σ τ : Shape} {order : Nat} (model : nn.Sequential σ τ)
    (state : State model ℝ) (input : Tensor ℝ σ)
    (directions : Tensor ℝ (Shape.dim order σ))
    (real : nn.TypedGraphModel (nn.Sequential.stateShapes model) σ τ ℝ)
    (nested : nn.TypedGraphModel (nn.Sequential.stateShapes model) σ τ
      (Dual.Nested ℝ order))
    (lowered : nn.Sequential.lowerToTypedGraph model (α := Dual.Nested ℝ order) = pure nested)
    (sameShapes : nested.nodeShapes = real.nodeShapes)
    (hgraph : Tape.GraphData.PreservesJet (Tensor ℝ σ) order
      real.data (sameShapes ▸ nested.data))
    (sameOutput : sameShapes ▸ nested.output = real.output) :
    derivative model state input directions =
      IO.ofExcept (TypedGraph.lowerToTapeChecked nested.data
          (Pack.append (state.map (Tensor.map (Dual.Nested.ofPrimal order)))
            (Pack.singleton (Dual.Nested.seedTensor (fun i : Fin order => directions[i]) input)))
          ()) >>= fun _ =>
        pure (iteratedFDeriv ℝ order
          (fun y => nn.TypedGraphModel.forward real state y) input
          (fun i : Fin order => directions[i])) := by
  simp only [derivative, lowered, pure_bind]
  cases checked : TypedGraph.lowerToTapeChecked nested.data
      (Pack.append (state.map (Tensor.map (Dual.Nested.ofPrimal order)))
        (Pack.singleton (Dual.Nested.seedTensor (fun i : Fin order => directions[i]) input)))
      () with
  | error message => simp only [IO.ofExcept_error, EIO.throw_bind]
  | ok result =>
    have hresult := TypedGraph.lowerToTapeChecked_eq _ _ _ result checked
    subst hresult
    simp only [IO.ofExcept_ok, pure_bind, Tape.GraphData.toTape_ctx_eq_eval]
    congr 1
    let ds := fun i : Fin order => directions[i]
    have hinputs := TensorPack.JetRelated.append
      (TensorPack.JetRelated.const ds input state)
      (TensorPack.JetRelated.singleton_seed ds input)
    simpa only [nn.TypedGraphModel.forward, WithData.forward] using
      WithData.tangent_forward real nested sameShapes hgraph
        sameOutput ds input
        (fun y => Pack.append state (Pack.singleton y))
        _ () hinputs

/-- The public higher-order VJP returns the pullback of the requested iterated input derivative.

The context includes state and input entries. Directions have zero state components, but the
final adjoint differentiates with respect to all entries, giving both parameter and input
gradients. The output cotangent and directions are fixed, including when directions repeat.
-/
@[autograd] theorem derivativeVJP_eq {σ τ : Shape} {order : Nat} (model : nn.Sequential σ τ)
    (state : State model ℝ) (input : Tensor ℝ σ)
    (directions : Tensor ℝ (Shape.dim order σ))
    (seed : Tensor ℝ τ)
    (real : nn.TypedGraphModel (nn.Sequential.stateShapes model) σ τ ℝ)
    (nested : nn.TypedGraphModel (nn.Sequential.stateShapes model) σ τ
      (Dual.Nested ℝ order))
    (valid : nn.Sequential.validate model = .ok ())
    (lowered : nn.Sequential.lowerToTypedGraph model (α := Dual.Nested ℝ order) = pure nested)
    (sameShapes : nested.nodeShapes = real.nodeShapes)
    (hgraph : Tape.GraphData.PreservesPullbackJet
      (CtxVec (nn.Sequential.stateShapes model ++ [σ])) order
      real.data (sameShapes ▸ nested.data))
    (sameOutput : sameShapes ▸ nested.output = real.output)
    (proofGraph :
      Tape.Graph (α := ℝ) Unit (nn.Sequential.stateShapes model ++ [σ]) real.nodeShapes)
    (same : proofGraph.toData = real.data)
    (correct : ∀ y, FDerivCorrectAt proofGraph.toData () y)
    (smooth : ContDiff ℝ (order + 1)
      (fun z => Tensor.toVec (WithData.forward real (TensorPack.ofVec z) ())))
    (result : TensorPack (Dual.Nested ℝ order) (nn.Sequential.stateShapes model ++ [σ]) ×
      Tensor (Dual.Nested ℝ order) τ)
    (checked : nested.vjpChecked
      ((state.map (Tensor.map (Dual.Nested.ofPrimal
        order))).append (Pack.singleton
          (Dual.Nested.seedTensor (fun i : Fin order => directions[i]) input))) ()
      (Tensor.map (Dual.Nested.ofPrimal order) seed) = .ok result) :
    let ds := fun i : Fin order => TensorPack.toVec
      ((TensorPack.zero (ss := nn.Sequential.stateShapes model)).append
        (Pack.singleton directions[i]))
    let x := TensorPack.toVec (state.append
      (Pack.singleton input))
    let gradient := TensorPack.ofVec
      ((fderiv ℝ (fun y => iteratedFDeriv ℝ order
        (fun z => Tensor.toVec (WithData.forward real (TensorPack.ofVec z) ()))
        y ds) x).adjoint
          (Tensor.toVec seed))
    derivativeVJP model state input directions seed = pure
      ((Pack.split (ss₁ := nn.Sequential.stateShapes model) (ss₂ := [σ]) gradient).1,
        (Pack.split (ss₁ := nn.Sequential.stateShapes model) (ss₂ := [σ])
          gradient).2.head) := by
  intro ds x gradient
  have hpack : result.1.map Dual.Nested.tangentTensor = gradient := by
    apply TensorPack.ext_getIdx
    intro shape index
    rw [TensorPack.getIdx_map]
    exact WithData.tangent_vjpChecked_iteratedFDeriv
      real nested sameShapes hgraph sameOutput proofGraph same _ _ _ () correct
      (TensorPack.JetRelated.const_append_seed state
        input (fun i : Fin order => directions[i])) smooth seed result checked index
  rw [← hpack]
  unfold derivativeVJP
  dsimp only
  rw [vjp_eq_checked model _ _ _ nested valid lowered result checked]
  simp only [pure_bind, Pack.split_map, Pack.head_map]

end autograd.model

end TorchLean
