/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Runtime.Link.Checked
public import NN.Proofs.Autograd.Runtime.Link.HigherOrderReverse
public import NN.Proofs.Autograd.FDeriv.Interchange
public import NN.Proofs.Autograd.FDeriv.TensorVectorization

/-!
# Higher derivatives of certified pullbacks

First-order graph certificates identify the real reverse pass with an adjoint Fréchet derivative.
Jet preservation identifies nested-dual execution with repeated differentiation of that pass.
Combining them proves higher-order reverse-mode correctness against mathlib, including smoothly
varying output cotangents and arbitrary real normed parameter spaces.

Successful checked execution is connected to the same pure graph calculation. The derivative
interpretation is exact-real; model recording must supply matching graphs and their local laws.
-/

@[expose] public section

open Spec TorchLean Runtime.Autograd.Model Proofs Proofs.Autograd

namespace Proofs.Autograd.Algebra.Graph

/-- Nested-dual reverse execution computes every derivative of the exact adjoint derivative.

The first-order certificate identifies the real pullback with mathlib's `fderiv`; the jet laws
then justify differentiating that pullback. Neither hypothesis can replace the other. -/
theorem tangent_backprop_adjoint_fderiv
    {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] {n : Nat}
    {Δ : Type} {Γ shapes : List Shape} (graph : Graph (α := ℝ) Δ Γ shapes)
    (nested : GraphData (Dual.Nested ℝ n) Δ Γ shapes)
    (h : GraphData.PreservesPullbackJet E n graph.toData nested)
    (directions : Fin n → E) (x : E) (inputs : E → TorchLean.TensorPack ℝ Γ)
    (values : TorchLean.TensorPack (Dual.Nested ℝ n) Γ) (data : Δ)
    (correct : GraphFDerivCorrect (graph.toReal data))
    (hinputs : TorchLean.TensorPack.JetRelated directions x inputs values)
    (seed : E → TorchLean.TensorPack ℝ (Γ ++ shapes))
    (seedValues : TorchLean.TensorPack (Dual.Nested ℝ n) (Γ ++ shapes))
    (hseed : TorchLean.TensorPack.JetRelated directions x seed seedValues)
    {shape : Shape} (input : Idx Γ shape) :
    Dual.Nested.tangentTensor (getIdx (nested.backpropCtx values data seedValues) input) =
      iteratedFDeriv ℝ n (fun y => getIdx (unflattenCtx
        ((fderiv ℝ (Proofs.Autograd.Graph.evalVec (graph.toReal data))
          (flattenCtx (inputs y))).adjoint (flattenCtx (seed y)))) input) x directions := by
  have hderiv := h.tangent_backprop directions x inputs values data hinputs
    seed seedValues hseed input
  have hfun : (fun y => getIdx (graph.toData.backpropCtx (inputs y) data (seed y)) input) =
      fun y => getIdx (unflattenCtx
        ((fderiv ℝ (Proofs.Autograd.Graph.evalVec (graph.toReal data))
          (flattenCtx (inputs y))).adjoint (flattenCtx (seed y)))) input := by
    funext y
    have heq := congrArg unflattenCtx
      (graph.backpropCtx_eq_adjoint_fderiv data correct (inputs y) (seed y))
    simp only [unflattenCtx_flattenCtx] at heq
    exact congrArg (fun ctx => getIdx ctx input) heq
  rwa [hfun] at hderiv

end Proofs.Autograd.Algebra.Graph

namespace Runtime.Autograd.Torch.TypedGraphWithData

/-- Extracting a nested graph's output computes the iterated derivative of its real execution. -/
theorem tangent_forward {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] {n : Nat}
    {Δ : Type} {Γ : List Shape} {shape : Shape}
    (real : TypedGraphWithData ℝ Δ Γ shape)
    (nested : TypedGraphWithData (Dual.Nested ℝ n) Δ Γ shape)
    (sameShapes : nested.nodeShapes = real.nodeShapes)
    (hgraph : Algebra.GraphData.PreservesJet E n real.data (sameShapes ▸ nested.data))
    (sameOutput : sameShapes ▸ nested.output = real.output)
    (directions : Fin n → E) (x : E) (inputs : E → TensorPack ℝ Γ)
    (values : TensorPack (Dual.Nested ℝ n) Γ) (data : Δ)
    (hinputs : TensorPack.JetRelated directions x inputs values) :
    Dual.Nested.tangentTensor (nested.forward values data) =
      iteratedFDeriv ℝ n (fun y => real.forward (inputs y) data) x directions := by
  rcases real with ⟨rs, rg, ro, rb⟩
  rcases nested with ⟨ns, ng, no, nb⟩
  dsimp only at sameShapes hgraph sameOutput ⊢
  cases sameShapes
  have ho : no = ro := sameOutput
  subst no
  exact hgraph.tangent_eval directions x inputs values data hinputs ro

/-- Every extracted input cotangent is the iterated derivative of the real typed graph's VJP.

The two runs must record the same node shapes and select the same output. Buffer observers do
not enter this pure method. The seed can vary smoothly with the parameters being differentiated.
-/
theorem tangent_vjp {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] {n : Nat}
    {Δ : Type} {Γ : List Shape} {shape : Shape}
    (real : TypedGraphWithData ℝ Δ Γ shape)
    (nested : TypedGraphWithData (Dual.Nested ℝ n) Δ Γ shape)
    (sameShapes : nested.nodeShapes = real.nodeShapes)
    (hgraph : Algebra.GraphData.PreservesPullbackJet E n real.data (sameShapes ▸ nested.data))
    (sameOutput : sameShapes ▸ nested.output = real.output)
    (directions : Fin n → E) (x : E) (inputs : E → TensorPack ℝ Γ)
    (values : TensorPack (Dual.Nested ℝ n) Γ) (data : Δ)
    (hinputs : TensorPack.JetRelated directions x inputs values)
    (seed : E → Tensor ℝ shape) (hseed : ContDiff ℝ n seed)
    {inputShape : Shape} (input : Idx Γ inputShape) :
    Dual.Nested.tangentTensor
      (getIdx (nested.vjpWithSeed values data (DualTensor.jet directions seed x)) input) =
      iteratedFDeriv ℝ n
        (fun y => getIdx (real.vjpWithSeed (inputs y) data (seed y)) input) x directions := by
  rcases real with ⟨rs, rg, ro, rb⟩
  rcases nested with ⟨ns, ng, no, nb⟩
  dsimp only at sameShapes hgraph sameOutput ⊢
  cases sameShapes
  have ho : no = ro := sameOutput
  subst no
  exact hgraph.tangent_backprop directions x inputs values data hinputs
    (fun y => Algebra.TensorPack.single ro (seed y))
    (Algebra.TensorPack.single ro (DualTensor.jet directions seed x))
    (TensorPack.JetRelated.single ro directions x hseed) input

/-- A successful checked reverse pass has the same higher derivatives as the real pullback.

Validation is required at the nested input actually executed, not throughout a neighbourhood.
Smoothness and the derivative interpretation come from the graph's separate jet certificate. -/
theorem tangent_vjpChecked {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] {n : Nat}
    {Δ : Type} {Γ : List Shape} {shape : Shape}
    (real : TypedGraphWithData ℝ Δ Γ shape)
    (nested : TypedGraphWithData (Dual.Nested ℝ n) Δ Γ shape)
    (sameShapes : nested.nodeShapes = real.nodeShapes)
    (hgraph : Algebra.GraphData.PreservesPullbackJet E n real.data (sameShapes ▸ nested.data))
    (sameOutput : sameShapes ▸ nested.output = real.output)
    (directions : Fin n → E) (x : E) (inputs : E → TensorPack ℝ Γ)
    (values : TensorPack (Dual.Nested ℝ n) Γ) (data : Δ)
    (hinputs : TensorPack.JetRelated directions x inputs values)
    (seed : E → Tensor ℝ shape) (hseed : ContDiff ℝ n seed)
    (result : TensorPack (Dual.Nested ℝ n) Γ × Tensor (Dual.Nested ℝ n) shape)
    (checked : nested.vjpChecked values data (DualTensor.jet directions seed x) = .ok result)
    {inputShape : Shape} (input : Idx Γ inputShape) :
    Dual.Nested.tangentTensor (getIdx result.1 input) =
      iteratedFDeriv ℝ n (fun y => getIdx (real.vjpWithSeed (inputs y) data (seed y)) input)
        x directions := by
  obtain ⟨tape, lowered⟩ : ∃ tape,
      Runtime.Autograd.TypedGraph.lowerToTapeChecked nested.data values data = .ok tape := by
    cases lowered : Runtime.Autograd.TypedGraph.lowerToTapeChecked nested.data values data with
    | error message =>
      simp only [vjpChecked, lowered, Bind.bind, Except.bind] at checked
      cases checked
    | ok tape => exact ⟨tape, rfl⟩
  have sameResult := Except.ok.inj
    ((vjpChecked_eq nested values data (DualTensor.jet directions seed x) tape lowered).symm.trans
      checked)
  rw [← sameResult]
  exact tangent_vjp real nested sameShapes hgraph sameOutput directions x inputs values data
    hinputs seed hseed input

/-- Checked nested execution differentiates the adjoint of the selected forward derivative.

The first-order proof is needed at each input in the smooth family. Runtime validation is only
needed for the particular nested execution returning `result`. These are separate obligations.
-/
theorem tangent_vjpChecked_adjoint_fderiv
    {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] {n : Nat}
    {Δ : Type} {Γ : List Shape} {shape : Shape}
    (real : TypedGraphWithData ℝ Δ Γ shape)
    (nested : TypedGraphWithData (Dual.Nested ℝ n) Δ Γ shape)
    (sameShapes : nested.nodeShapes = real.nodeShapes)
    (hgraph : Algebra.GraphData.PreservesPullbackJet E n real.data (sameShapes ▸ nested.data))
    (sameOutput : sameShapes ▸ nested.output = real.output)
    (proofGraph : Algebra.Graph (α := ℝ) Δ Γ real.nodeShapes)
    (same : proofGraph.toData = real.data)
    (directions : Fin n → E) (x : E) (inputs : E → TensorPack ℝ Γ)
    (values : TensorPack (Dual.Nested ℝ n) Γ) (data : Δ)
    (correct : ∀ y, GraphFDerivCorrectAt (Algebra.Graph.toReal proofGraph data)
      (flattenCtx (inputs y)))
    (hinputs : TensorPack.JetRelated directions x inputs values)
    (seed : E → Tensor ℝ shape) (hseed : ContDiff ℝ n seed)
    (result : TensorPack (Dual.Nested ℝ n) Γ × Tensor (Dual.Nested ℝ n) shape)
    (checked : nested.vjpChecked values data (DualTensor.jet directions seed x) = .ok result)
    {inputShape : Shape} (input : Idx Γ inputShape) :
    Dual.Nested.tangentTensor (getIdx result.1 input) =
      iteratedFDeriv ℝ n (fun y => getIdx (unflattenCtx
        ((fderiv ℝ (fun z => tensorToVec (real.forward (unflattenCtx z) data))
          (flattenCtx (inputs y))).adjoint (tensorToVec (seed y)))) input) x directions := by
  have hderiv := tangent_vjpChecked real nested sameShapes hgraph sameOutput directions x
    inputs values data hinputs seed hseed result checked input
  have hfun : (fun y => getIdx (real.vjpWithSeed (inputs y) data (seed y)) input) =
      fun y => getIdx (unflattenCtx
        ((fderiv ℝ (fun z => tensorToVec (real.forward (unflattenCtx z) data))
          (flattenCtx (inputs y))).adjoint (tensorToVec (seed y)))) input := by
    funext y
    have heq := congrArg unflattenCtx
      (vjpWithSeed_adjoint_fderiv real proofGraph same (inputs y) data (seed y) (correct y))
    simp only [unflattenCtx_flattenCtx] at heq
    exact congrArg (fun ctx => getIdx ctx input) heq
  rwa [hfun] at hderiv

/-- Checked nested reverse execution computes the pullback of any fixed higher derivative.

All context entries, including parameters, can occur in the direction tuple. Holding the output
cotangent fixed is essential: a varying cotangent contributes additional derivative terms.
The forward map must be `C^(n+1)` to interchange the reverse derivative with the `n` directions.
-/
theorem tangent_vjpChecked_iteratedFDeriv
    {n : Nat} {Δ : Type} {Γ : List Shape} {shape : Shape}
    (real : TypedGraphWithData ℝ Δ Γ shape)
    (nested : TypedGraphWithData (Dual.Nested ℝ n) Δ Γ shape)
    (sameShapes : nested.nodeShapes = real.nodeShapes)
    (hgraph : Algebra.GraphData.PreservesPullbackJet (CtxVec Γ) n
      real.data (sameShapes ▸ nested.data))
    (sameOutput : sameShapes ▸ nested.output = real.output)
    (proofGraph : Algebra.Graph (α := ℝ) Δ Γ real.nodeShapes)
    (same : proofGraph.toData = real.data)
    (directions : Fin n → CtxVec Γ) (x : CtxVec Γ)
    (values : TensorPack (Dual.Nested ℝ n) Γ) (data : Δ)
    (correct : ∀ y, GraphFDerivCorrectAt (Algebra.Graph.toReal proofGraph data) y)
    (hinputs : TensorPack.JetRelated directions x unflattenCtx values)
    (smooth : ContDiff ℝ (n + 1) (fun z => tensorToVec (real.forward (unflattenCtx z) data)))
    (seed : Tensor ℝ shape)
    (result : TensorPack (Dual.Nested ℝ n) Γ × Tensor (Dual.Nested ℝ n) shape)
    (checked : nested.vjpChecked values data (Tensor.map (Dual.Nested.ofPrimal n) seed) =
      .ok result)
    {inputShape : Shape} (input : Idx Γ inputShape) :
    Dual.Nested.tangentTensor (getIdx result.1 input) =
      getIdx (unflattenCtx
        ((fderiv ℝ (fun y => iteratedFDeriv ℝ n
          (fun z => tensorToVec (real.forward (unflattenCtx z) data)) y directions)
          x).adjoint (tensorToVec seed))) input := by
  have hderiv := tangent_vjpChecked_adjoint_fderiv real nested sameShapes hgraph sameOutput
    proofGraph same directions x unflattenCtx values data (fun y => correct _)
    hinputs (fun _ => seed) contDiff_const result
    (by simpa only [DualTensor.jet_const] using checked) input
  simp only [flattenCtx_unflattenCtx] at hderiv
  rw [hderiv]
  let f := fun z => tensorToVec (real.forward (unflattenCtx z) data)
  let adj : (CtxVec Γ →L[ℝ] Vec (Shape.size shape)) →L[ℝ] CtxVec Γ :=
    (ContinuousLinearMap.apply ℝ (CtxVec Γ) (tensorToVec seed)).comp
      ContinuousLinearMap.adjoint.toContinuousLinearEquiv.toContinuousLinearMap
  have hp : ContDiff ℝ n (fun y => (fderiv ℝ f y).adjoint (tensorToVec seed)) :=
    adj.contDiff.comp (smooth.fderiv_right (m := n) (by simp))
  simp only [← CtxVec.getTensorCLM_apply]
  change iteratedFDeriv ℝ n (CtxVec.getTensorCLM input ∘
    (fun y => (fderiv ℝ f y).adjoint (tensorToVec seed))) x directions = _
  rw [(CtxVec.getTensorCLM input).iteratedFDeriv_comp_left hp.contDiffAt (by simp)]
  change CtxVec.getTensorCLM input (iteratedFDeriv ℝ n
    (fun y => (fderiv ℝ f y).adjoint (tensorToVec seed)) x directions) = _
  rw [iteratedFDeriv_adjoint_fderiv smooth]

end Runtime.Autograd.Torch.TypedGraphWithData
