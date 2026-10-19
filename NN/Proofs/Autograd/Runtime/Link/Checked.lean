/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core.TypedGraph
public import NN.Proofs.Autograd.Runtime.Link.FDeriv
public import NN.Proofs.Autograd.Tape.Nodes.Context

/-!
# Correctness of checked differentiation

The public autograd transforms validate and evaluate the graph before sweeping saved reverse
rules from its selected output. The theorems here connect that checked path to the graph semantics
and, for analytically correct real
graphs, to mathlib's Fréchet derivative. Validation remains a separate hypothesis: passing a
runtime domain check alone is not a derivative-correctness certificate.
-/

@[expose] public section

namespace Runtime.Autograd.Torch.TypedGraphWithData

open Spec TorchLean
open Proofs.Autograd

/-- Successful validation makes the checked VJP agree with the stored graph reverse rules.
This requires no algebraic laws on the scalar backend. The single-output API uses certified
support when available, rather than claiming equality with an arbitrary dense all-seeds VJP. -/
theorem vjpChecked_eq {α Δ : Type} [Storage α] [Add α] [Zero α]
    {Γ : List Shape} {τ : Shape}
    (graph : TypedGraphWithData α Δ Γ τ) (inputs : TensorPack α Γ) (data : Δ)
    (seed : Tensor α τ)
    (result : Runtime.Autograd.Tape (Spec.SomeTensor α) × TensorPack α (Γ ++ graph.nodeShapes))
    (checked : Runtime.Autograd.TypedGraph.lowerToTapeChecked graph.data inputs data =
      .ok result) :
    graph.vjpChecked inputs data seed =
      .ok (graph.vjp inputs data seed, graph.forward inputs data) := by
  cases saved : Algebra.SavedGraph.lowerChecked graph.data inputs data with
  | error message =>
      have lowered := Runtime.Autograd.TypedGraph.compileChecked_eq_lowerToTapeChecked
        graph.data inputs data
      simp only [Runtime.Autograd.TypedGraph.compileChecked, saved, Except.map, checked] at lowered
      cases lowered
  | ok execution => exact vjpChecked_eq_of_lowerChecked graph inputs data seed execution saved

/-- A successful checked reverse pass returns the graph pullback and forward value.
Validation is required at the input actually executed; no scalar algebraic laws are needed. -/
theorem eq_of_vjpChecked_ok {α Δ : Type} [Storage α] [Add α] [Zero α]
    {Γ : List Shape} {τ : Shape}
    (graph : TypedGraphWithData α Δ Γ τ) (inputs : TensorPack α Γ) (data : Δ)
    (seed : Tensor α τ) (result : TensorPack α Γ × Tensor α τ)
    (checked : graph.vjpChecked inputs data seed = .ok result) :
    result = (graph.vjp inputs data seed, graph.forward inputs data) := by
  obtain ⟨execution, lowered⟩ : ∃ execution,
      Algebra.SavedGraph.lowerChecked graph.data inputs data = .ok execution := by
    cases lowered : Algebra.SavedGraph.lowerChecked graph.data inputs data with
    | error message =>
      simp only [vjpChecked, Runtime.Autograd.TypedGraph.compileChecked, lowered, Except.map,
        Bind.bind, Except.bind] at checked
      cases checked
    | ok execution => exact ⟨execution, rfl⟩
  exact (Except.ok.inj
    ((vjpChecked_eq_of_lowerChecked graph inputs data seed execution lowered).symm.trans
      checked)).symm

/-- Without storage equality, single-output differentiation retains the exact dense reference.
This connects real-valued and symbolic carriers to the existing derivative theorems. -/
theorem vjp_eq_backpropCtx {α Δ : Type} [Storage α] [Add α] [Zero α]
    {Γ : List Shape} {τ : Shape} (graph : TypedGraphWithData α Δ Γ τ)
    (inputs : TorchLean.TensorPack α Γ) (data : Δ) (seed : Tensor α τ)
    (equality : Storage.decEq? (α := α) = none) :
    graph.vjp inputs data seed = graph.data.backpropCtx inputs data
      (Algebra.TensorPack.single graph.output seed) := by
  simp only [vjp, Algebra.SavedGraph.backpropFrom, equality]
  rw [Algebra.SavedGraph.ofGraph_backpropStored, ← Algebra.TensorPack.takeLeft_eq_split,
    Algebra.GraphData.takeLeft_backpropAllCtx]

/-- The selected output has the derivative obtained by projecting the graph derivative. -/
theorem hasFDerivAt_forward {Δ : Type} {Γ : List Shape} {τ : Shape}
    (graph : TypedGraphWithData ℝ Δ Γ τ)
    (proofGraph : Algebra.Graph (α := ℝ) Δ Γ graph.nodeShapes)
    (same : proofGraph.toData = graph.data)
    (inputs : TensorPack ℝ Γ) (data : Δ)
    (correct : GraphFDerivCorrectAt (Algebra.Graph.toReal proofGraph data) (flattenCtx inputs)) :
    HasFDerivAt (fun x => tensorToVec (graph.forward (unflattenCtx x) data))
      ((CtxVec.getCLM graph.output).comp
        (fderiv ℝ (Graph.evalVec (Algebra.Graph.toReal proofGraph data)) (flattenCtx inputs)))
      (flattenCtx inputs) := by
  have forward_eq : (fun x => tensorToVec (graph.forward (unflattenCtx x) data)) =
      (CtxVec.getCLM graph.output) ∘ Graph.evalVec (Algebra.Graph.toReal proofGraph data) := by
    funext x
    rw [Function.comp_apply, CtxVec.getCLM_apply, ← flattenCtx_unflattenCtx x,
      Algebra.Graph.toReal_evalVec, CtxVec.get_flattenCtx]
    simp only [unflattenCtx_flattenCtx, forward, Algebra.Graph.eval, same]
  obtain ⟨D, hD, _⟩ := Graph.hasFDerivAt_evalVec_and_jvp_at
    (Algebra.Graph.toReal proofGraph data) (flattenCtx inputs) correct
  rw [forward_eq, hD.fderiv]
  exact (CtxVec.getCLM graph.output).hasFDerivAt.comp _ hD

/-- The checked forward-mode tangent is mathlib's derivative applied to the input direction. -/
theorem jvpChecked_fderiv {Δ : Type} {Γ : List Shape} {τ : Shape}
    (graph : TypedGraphWithData ℝ Δ Γ τ)
    (proofGraph : Algebra.Graph (α := ℝ) Δ Γ graph.nodeShapes)
    (same : proofGraph.toData = graph.data)
    (inputs tangents : TensorPack ℝ Γ) (data : Δ)
    (correct : GraphFDerivCorrectAt (Algebra.Graph.toReal proofGraph data) (flattenCtx inputs))
    (result : TensorPack ℝ (Γ ++ graph.nodeShapes) × TensorPack ℝ (Γ ++ graph.nodeShapes))
    (checked : Runtime.Autograd.TypedGraph.jvpChecked graph.data inputs tangents data =
      .ok result) :
    tensorToVec (Proofs.getIdx result.2 graph.output) =
      fderiv ℝ (fun x => tensorToVec (graph.forward (unflattenCtx x) data))
        (flattenCtx inputs) (flattenCtx tangents) := by
  rw [Runtime.Autograd.TypedGraph.jvpChecked_eq _ _ _ _ _ checked,
    (hasFDerivAt_forward graph proofGraph same inputs data correct).fderiv,
    ContinuousLinearMap.comp_apply, CtxVec.getCLM_apply,
    ← Graph.jvpVec_eq_fderiv_at _ _ _ correct, Graph.jvpVec_flattenCtx,
    Algebra.Graph.toReal_jvpCtx, CtxVec.get_flattenCtx]
  simp only [Algebra.Graph.jvpCtx, same]

/-- The pure graph pullback is the adjoint derivative of its selected output. -/
theorem vjp_adjoint_fderiv {Δ : Type} {Γ : List Shape} {τ : Shape}
    (graph : TypedGraphWithData ℝ Δ Γ τ)
    (proofGraph : Algebra.Graph (α := ℝ) Δ Γ graph.nodeShapes)
    (same : proofGraph.toData = graph.data)
    (inputs : TensorPack ℝ Γ) (data : Δ) (seed : Tensor ℝ τ)
    (correct : GraphFDerivCorrectAt (Algebra.Graph.toReal proofGraph data) (flattenCtx inputs)) :
    flattenCtx (graph.vjp inputs data seed) =
      (fderiv ℝ (fun x => tensorToVec (graph.forward (unflattenCtx x) data))
        (flattenCtx inputs)).adjoint (tensorToVec seed) := by
  have seed_eq : (CtxVec.getCLM graph.output).adjoint (tensorToVec seed) =
      flattenCtx (Algebra.TensorPack.single graph.output seed) := by
    apply ext_inner_left ℝ
    intro x
    rw [ContinuousLinearMap.adjoint_inner_right, CtxVec.getCLM_apply,
      ← flattenCtx_unflattenCtx x, CtxVec.get_flattenCtx,
      ← dotList_eq_inner_flattenCtx, TensorPack.dotList_eq_algebra_dotList,
      Algebra.TensorPack.dotList_single, ← dot_eq_tensorAlgebra_dot,
      dot_eq_inner_tensorToVec]
  rw [(hasFDerivAt_forward graph proofGraph same inputs data correct).fderiv,
    ContinuousLinearMap.adjoint_comp, ContinuousLinearMap.comp_apply, seed_eq]
  rw [vjp_eq_backpropCtx graph inputs data seed (by rfl), ← same]
  exact Algebra.Graph.backpropCtx_eq_adjoint_fderiv_at proofGraph data inputs correct _

/--
The checked API returns the adjoint of mathlib's derivative of its selected forward output.

The proof graph must describe the exact operation data that execution uses. Differentiability is
required only at this input and the corresponding intermediate values. The carrier is `ℝ`;
floating-point error and native kernels are not identified with exact real arithmetic.
-/
theorem vjpChecked_adjoint_fderiv {Δ : Type} {Γ : List Shape} {τ : Shape}
    (graph : TypedGraphWithData ℝ Δ Γ τ)
    (proofGraph : Algebra.Graph (α := ℝ) Δ Γ graph.nodeShapes)
    (same : proofGraph.toData = graph.data)
    (inputs : TensorPack ℝ Γ) (data : Δ) (seed : Tensor ℝ τ)
    (correct : GraphFDerivCorrectAt (Algebra.Graph.toReal proofGraph data) (flattenCtx inputs))
    (result : TensorPack ℝ Γ × Tensor ℝ τ)
    (checked : graph.vjpChecked inputs data seed = .ok result) :
    flattenCtx result.1 =
      (fderiv ℝ (fun x => tensorToVec (graph.forward (unflattenCtx x) data))
        (flattenCtx inputs)).adjoint (tensorToVec seed) := by
  rw [eq_of_vjpChecked_ok graph inputs data seed result checked]
  exact vjp_adjoint_fderiv graph proofGraph same inputs data seed correct

end Runtime.Autograd.Torch.TypedGraphWithData
