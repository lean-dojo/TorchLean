/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.TypedGraph.WithData.Checked
public import TorchLean.NN.Tape.Lowering.HigherOrderReverse
public import TorchLean.NN.FDeriv.Interchange
public import TorchLean.NN.FDeriv.TensorVectorization

/-!
# Higher derivatives of certified pullbacks

First-order graph certificates identify the real reverse pass with an adjoint Fréchet derivative,
and jet preservation identifies nested-dual execution with repeated differentiation of that pass.
Together they prove higher-order reverse-mode correctness against mathlib, with output cotangents
that may vary smoothly and parameters in any real normed space.

Successful checked execution is connected to the same pure graph calculation. The derivative
interpretation is exact-real, and model recording must supply matching graphs and their local
laws.
-/

@[expose] public section

open TorchLean.TensorPack TorchLean.Tensor TorchLean.Tape.RealGraph
open TorchLean.Shape (size)

namespace TorchLean.Tape.Graph

/-- Nested-dual reverse execution computes the `n`-th iterated derivative, along `directions`, of
the adjoint Fréchet derivative applied to the seed.

The first-order certificate `correct` identifies the real pullback with mathlib's `fderiv`, and
the jet laws `h` then justify differentiating that pullback. Neither hypothesis can replace the
other. -/
theorem tangent_backprop_eq_fderiv
    {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] {n : Nat}
    {Δ : Type} {Γ shapes : List Shape} (graph : Graph (α := ℝ) Δ Γ shapes)
    (nested : GraphData (Dual.Nested ℝ n) Δ Γ shapes)
    (h : GraphData.PreservesPullbackJet E n graph.toData nested)
    (directions : Fin n → E) (x : E) (inputs : E → TensorPack ℝ Γ)
    (values : TensorPack (Dual.Nested ℝ n) Γ) (data : Δ)
    (correct : FDerivCorrect graph.toData data)
    (hinputs : TensorPack.JetRelated directions x inputs values)
    (seed : E → TensorPack ℝ (Γ ++ shapes))
    (seedValues : TensorPack (Dual.Nested ℝ n) (Γ ++ shapes))
    (hseed : TensorPack.JetRelated directions x seed seedValues)
    {shape : Shape} (input : Idx Γ shape) :
    Dual.Nested.tangentTensor (getIdx (nested.backpropCtx values data seedValues) input) =
      iteratedFDeriv ℝ n (fun y => getIdx (TensorPack.ofVec
        ((fderiv ℝ (graph.toData.evalVec data)
          (TensorPack.toVec (inputs y))).adjoint (TensorPack.toVec (seed y)))) input) x
            directions := by
  have hderiv := h.tangent_backprop directions x inputs values data hinputs
    seed seedValues hseed input
  have hfun : (fun y => getIdx (graph.toData.backpropCtx (inputs y) data (seed y)) input) =
      fun y => getIdx (TensorPack.ofVec
        ((fderiv ℝ (graph.toData.evalVec data)
          (TensorPack.toVec (inputs y))).adjoint (TensorPack.toVec (seed y)))) input := by
    funext y
    have heq := congrArg TensorPack.ofVec
      (graph.backpropCtx_eq_fderiv data (inputs y) (correct.at (TensorPack.toVec (inputs y)))
        (seed y))
    simp only [TensorPack.ofVec_toVec] at heq
    exact congrArg (fun ctx => getIdx ctx input) heq
  rwa [hfun] at hderiv

end TorchLean.Tape.Graph

namespace TorchLean.TypedGraph.WithData

/-- If the nested graph records the real graph's node shapes and output and its nodes preserve
jets, `tangentTensor` of its output is the `n`-th iterated derivative of the real forward map
along `directions`. -/
theorem tangent_forward {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] {n : Nat}
    {Δ : Type} {Γ : List Shape} {shape : Shape}
    (real : WithData ℝ Δ Γ shape)
    (nested : WithData (Dual.Nested ℝ n) Δ Γ shape)
    (sameShapes : nested.nodeShapes = real.nodeShapes)
    (hgraph : Tape.GraphData.PreservesJet E n real.data (sameShapes ▸ nested.data))
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

private theorem decEq?_nested_eq_none (n : Nat) :
    Storage.decEq? (α := Dual.Nested ℝ n) = none := by
  cases n <;> rfl

/-- `tangentTensor` of each input gradient of the nested `vjp` is the `n`-th iterated derivative of
the real typed graph's `vjp` along `directions`.

The two runs must record the same node shapes and select the same output, and buffer observers do
not enter this pure method. The seed can vary smoothly with the parameters being differentiated.
-/
theorem tangent_vjp {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] {n : Nat}
    {Δ : Type} {Γ : List Shape} {shape : Shape}
    (real : WithData ℝ Δ Γ shape)
    (nested : WithData (Dual.Nested ℝ n) Δ Γ shape)
    (sameShapes : nested.nodeShapes = real.nodeShapes)
    (hgraph : Tape.GraphData.PreservesPullbackJet E n real.data (sameShapes ▸ nested.data))
    (sameOutput : sameShapes ▸ nested.output = real.output)
    (directions : Fin n → E) (x : E) (inputs : E → TensorPack ℝ Γ)
    (values : TensorPack (Dual.Nested ℝ n) Γ) (data : Δ)
    (hinputs : TensorPack.JetRelated directions x inputs values)
    (seed : E → Tensor ℝ shape) (hseed : ContDiff ℝ n seed)
    {inputShape : Shape} (input : Idx Γ inputShape) :
    Dual.Nested.tangentTensor
      (getIdx (nested.vjp values data (DualTensor.jet directions seed x)) input) =
      iteratedFDeriv ℝ n
        (fun y => getIdx (real.vjp (inputs y) data (seed y)) input) x directions := by
  have realPullback (y : E) := vjp_eq_backpropCtx real (inputs y) data (seed y) (by rfl)
  simp_rw [vjp_eq_backpropCtx nested values data (DualTensor.jet directions seed x)
    (decEq?_nested_eq_none n), realPullback]
  rcases real with ⟨rs, rg, ro, rb⟩
  rcases nested with ⟨ns, ng, no, nb⟩
  dsimp only at sameShapes hgraph sameOutput ⊢
  cases sameShapes
  have ho : no = ro := sameOutput
  subst no
  exact hgraph.tangent_backprop directions x inputs values data hinputs
    (fun y => TensorPack.single ro (seed y))
    (TensorPack.single ro (DualTensor.jet directions seed x))
    (TensorPack.JetRelated.single ro directions x hseed) input

/-- If the checked nested reverse pass succeeds, `tangentTensor` of each input gradient is the
`n`-th iterated derivative of the real pullback `vjp` along `directions`.

Validation is required only at the nested input actually executed. Smoothness and the derivative
interpretation come from the graph's separate jet certificate. -/
theorem tangent_vjpChecked {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] {n : Nat}
    {Δ : Type} {Γ : List Shape} {shape : Shape}
    (real : WithData ℝ Δ Γ shape)
    (nested : WithData (Dual.Nested ℝ n) Δ Γ shape)
    (sameShapes : nested.nodeShapes = real.nodeShapes)
    (hgraph : Tape.GraphData.PreservesPullbackJet E n real.data (sameShapes ▸ nested.data))
    (sameOutput : sameShapes ▸ nested.output = real.output)
    (directions : Fin n → E) (x : E) (inputs : E → TensorPack ℝ Γ)
    (values : TensorPack (Dual.Nested ℝ n) Γ) (data : Δ)
    (hinputs : TensorPack.JetRelated directions x inputs values)
    (seed : E → Tensor ℝ shape) (hseed : ContDiff ℝ n seed)
    (result : TensorPack (Dual.Nested ℝ n) Γ × Tensor (Dual.Nested ℝ n) shape)
    (checked : nested.vjpChecked values data (DualTensor.jet directions seed x) = .ok result)
    {inputShape : Shape} (input : Idx Γ inputShape) :
    Dual.Nested.tangentTensor (getIdx result.1 input) =
      iteratedFDeriv ℝ n (fun y => getIdx (real.vjp (inputs y) data (seed y)) input)
        x directions := by
  rw [eq_of_vjpChecked_ok nested values data (DualTensor.jet directions seed x) result checked]
  exact tangent_vjp real nested sameShapes hgraph sameOutput directions x inputs values data
    hinputs seed hseed input

/-- If the checked nested reverse pass succeeds, `tangentTensor` of each input gradient is the
`n`-th iterated derivative of the adjoint of the selected forward derivative applied to the seed.

The first-order proof is needed at each input in the smooth family, while runtime validation
concerns only the nested execution that returned `result`.
-/
theorem tangent_vjpChecked_eq_fderiv
    {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] {n : Nat}
    {Δ : Type} {Γ : List Shape} {shape : Shape}
    (real : WithData ℝ Δ Γ shape)
    (nested : WithData (Dual.Nested ℝ n) Δ Γ shape)
    (sameShapes : nested.nodeShapes = real.nodeShapes)
    (hgraph : Tape.GraphData.PreservesPullbackJet E n real.data (sameShapes ▸ nested.data))
    (sameOutput : sameShapes ▸ nested.output = real.output)
    (proofGraph : Tape.Graph (α := ℝ) Δ Γ real.nodeShapes)
    (same : proofGraph.toData = real.data)
    (directions : Fin n → E) (x : E) (inputs : E → TensorPack ℝ Γ)
    (values : TensorPack (Dual.Nested ℝ n) Γ) (data : Δ)
    (correct : ∀ y, FDerivCorrectAt proofGraph.toData data (TensorPack.toVec (inputs y)))
    (hinputs : TensorPack.JetRelated directions x inputs values)
    (seed : E → Tensor ℝ shape) (hseed : ContDiff ℝ n seed)
    (result : TensorPack (Dual.Nested ℝ n) Γ × Tensor (Dual.Nested ℝ n) shape)
    (checked : nested.vjpChecked values data (DualTensor.jet directions seed x) = .ok result)
    {inputShape : Shape} (input : Idx Γ inputShape) :
    Dual.Nested.tangentTensor (getIdx result.1 input) =
      iteratedFDeriv ℝ n (fun y => getIdx (TensorPack.ofVec
        ((fderiv ℝ (fun z => Tensor.toVec (real.forward (TensorPack.ofVec z) data))
          (TensorPack.toVec (inputs y))).adjoint (Tensor.toVec (seed y)))) input) x directions := by
  have hderiv := tangent_vjpChecked real nested sameShapes hgraph sameOutput directions x
    inputs values data hinputs seed hseed result checked input
  have hfun : (fun y => getIdx (real.vjp (inputs y) data (seed y)) input) =
      fun y => getIdx (TensorPack.ofVec
        ((fderiv ℝ (fun z => Tensor.toVec (real.forward (TensorPack.ofVec z) data))
          (TensorPack.toVec (inputs y))).adjoint (Tensor.toVec (seed y)))) input := by
    funext y
    have heq := congrArg TensorPack.ofVec
      (vjp_eq_fderiv real proofGraph same (inputs y) data (seed y) (correct y))
    simp only [TensorPack.ofVec_toVec] at heq
    exact congrArg (fun ctx => getIdx ctx input) heq
  rwa [hfun] at hderiv

/-- For a fixed seed, a successful checked nested reverse pass returns the pullback of the seed
through the derivative of `y ↦ iteratedFDeriv ℝ n f y directions`, where `f` is the forward map.

All context entries, including parameters, can occur in the direction tuple. Holding the output
cotangent fixed is essential: a varying cotangent contributes additional derivative terms.
The forward map must be `C^(n+1)` to interchange the reverse derivative with the `n` directions.
-/
theorem tangent_vjpChecked_iteratedFDeriv
    {n : Nat} {Δ : Type} {Γ : List Shape} {shape : Shape}
    (real : WithData ℝ Δ Γ shape)
    (nested : WithData (Dual.Nested ℝ n) Δ Γ shape)
    (sameShapes : nested.nodeShapes = real.nodeShapes)
    (hgraph : Tape.GraphData.PreservesPullbackJet (CtxVec Γ) n
      real.data (sameShapes ▸ nested.data))
    (sameOutput : sameShapes ▸ nested.output = real.output)
    (proofGraph : Tape.Graph (α := ℝ) Δ Γ real.nodeShapes)
    (same : proofGraph.toData = real.data)
    (directions : Fin n → CtxVec Γ) (x : CtxVec Γ)
    (values : TensorPack (Dual.Nested ℝ n) Γ) (data : Δ)
    (correct : ∀ y, FDerivCorrectAt proofGraph.toData data y)
    (hinputs : TensorPack.JetRelated directions x TensorPack.ofVec values)
    (smooth : ContDiff ℝ (n + 1) (fun z => Tensor.toVec (real.forward (TensorPack.ofVec z) data)))
    (seed : Tensor ℝ shape)
    (result : TensorPack (Dual.Nested ℝ n) Γ × Tensor (Dual.Nested ℝ n) shape)
    (checked : nested.vjpChecked values data (Tensor.map (Dual.Nested.ofPrimal n) seed) =
      .ok result)
    {inputShape : Shape} (input : Idx Γ inputShape) :
    Dual.Nested.tangentTensor (getIdx result.1 input) =
      getIdx (TensorPack.ofVec
        ((fderiv ℝ (fun y => iteratedFDeriv ℝ n
          (fun z => Tensor.toVec (real.forward (TensorPack.ofVec z) data)) y directions)
          x).adjoint (Tensor.toVec seed))) input := by
  have hderiv := tangent_vjpChecked_eq_fderiv real nested sameShapes hgraph sameOutput
    proofGraph same directions x TensorPack.ofVec values data (fun y => correct _)
    hinputs (fun _ => seed) contDiff_const result
    (by simpa only [DualTensor.jet_const] using checked) input
  simp only [TensorPack.toVec_ofVec] at hderiv
  rw [hderiv]
  let f := fun z => Tensor.toVec (real.forward (TensorPack.ofVec z) data)
  let adj : (CtxVec Γ →L[ℝ] Vec (Shape.size shape)) →L[ℝ] CtxVec Γ :=
    (ContinuousLinearMap.apply ℝ (CtxVec Γ) (Tensor.toVec seed)).comp
      ContinuousLinearMap.adjoint.toContinuousLinearEquiv.toContinuousLinearMap
  have hp : ContDiff ℝ n (fun y => (fderiv ℝ f y).adjoint (Tensor.toVec seed)) :=
    adj.contDiff.comp (smooth.fderiv_right (m := n) (by simp))
  simp only [← CtxVec.getTensorCLM_apply]
  change iteratedFDeriv ℝ n (CtxVec.getTensorCLM input ∘
    (fun y => (fderiv ℝ f y).adjoint (Tensor.toVec seed))) x directions = _
  rw [(CtxVec.getTensorCLM input).iteratedFDeriv_comp_left hp.contDiffAt (by simp)]
  change CtxVec.getTensorCLM input (iteratedFDeriv ℝ n
    (fun y => (fderiv ℝ f y).adjoint (Tensor.toVec seed)) x directions) = _
  rw [iteratedFDeriv_adjoint_fderiv smooth]

end TorchLean.TypedGraph.WithData
