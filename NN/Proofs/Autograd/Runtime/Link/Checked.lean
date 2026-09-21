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

The public autograd transforms run domain validation, lower a graph to a tape, sweep its stored
reverse rules, and recover the typed input gradients from a heterogeneous array. The theorems here
connect that complete checked path to the graph semantics and, for analytically correct real
graphs, to mathlib's Fréchet derivative. Validation remains a separate hypothesis: passing a
runtime domain check alone is not a derivative-correctness certificate.
-/

@[expose] public section

namespace Runtime.Autograd.Torch.TypedGraphWithData

open Spec TorchLean
open Proofs.Autograd

/-- Successful validation makes the checked VJP agree with the stored graph reverse rules.
This preserves the order of additions and requires no algebraic laws on the scalar backend. -/
theorem vjpChecked_eq {α Δ : Type} [Storage α] [Add α] [Zero α]
    {Γ : List Shape} {τ : Shape}
    (graph : TypedGraphWithData α Δ Γ τ) (inputs : TensorPack α Γ) (data : Δ)
    (seed : Tensor α τ)
    (result : Runtime.Autograd.Tape α × TensorPack α (Γ ++ graph.nodeShapes))
    (checked : Runtime.Autograd.TypedGraph.lowerToTapeChecked graph.data inputs data =
      .ok result) :
    graph.vjpChecked inputs data seed =
      .ok (graph.vjpWithSeed inputs data seed, graph.forward inputs data) := by
  have same := Runtime.Autograd.TypedGraph.lowerToTapeChecked_eq
    graph.data inputs data result checked
  have checked' := checked.trans (congrArg Except.ok same)
  simp only [vjpChecked, checked', Bind.bind, Except.bind, vjpFromTape,
    Runtime.Autograd.TypedGraph.backwardDenseAllFrom]
  rw [Algebra.Graph.backwardDenseFrom_lowerGraphDataToTape_eq_backpropAllCtx]
  simp only [TensorPack.ofShapeErasedArray_toShapeErasedArray_prefix]
  rw [← Algebra.TensorPack.takeLeft_eq_split, Algebra.GraphData.takeLeft_backpropAllCtx,
    Algebra.Graph.lowerGraphDataToTape_ctx_eq_eval]
  rfl

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
theorem vjpWithSeed_adjoint_fderiv {Δ : Type} {Γ : List Shape} {τ : Shape}
    (graph : TypedGraphWithData ℝ Δ Γ τ)
    (proofGraph : Algebra.Graph (α := ℝ) Δ Γ graph.nodeShapes)
    (same : proofGraph.toData = graph.data)
    (inputs : TensorPack ℝ Γ) (data : Δ) (seed : Tensor ℝ τ)
    (correct : GraphFDerivCorrectAt (Algebra.Graph.toReal proofGraph data) (flattenCtx inputs)) :
    flattenCtx (graph.vjpWithSeed inputs data seed) =
      (fderiv ℝ (fun x => tensorToVec (graph.forward (unflattenCtx x) data))
        (flattenCtx inputs)).adjoint (tensorToVec seed) := by
  have seed_eq : (CtxVec.getCLM graph.output).adjoint (tensorToVec seed) =
      flattenCtx (Algebra.TensorPack.single graph.output seed) := by
    apply ext_inner_left ℝ
    intro x
    rw [ContinuousLinearMap.adjoint_inner_right, CtxVec.getCLM_apply,
      ← flattenCtx_unflattenCtx x, CtxVec.get_flattenCtx,
      ← dotList_eq_inner_flattenCtx, dotList_eq_algebra_dotList,
      Algebra.TensorPack.dotList_single, ← dot_eq_tensorAlgebra_dot,
      dot_eq_inner_tensorToVec]
  rw [(hasFDerivAt_forward graph proofGraph same inputs data correct).fderiv,
    ContinuousLinearMap.adjoint_comp, ContinuousLinearMap.comp_apply, seed_eq]
  unfold vjpWithSeed
  rw [← same]
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
  obtain ⟨tape, lowered⟩ : ∃ tape,
      Runtime.Autograd.TypedGraph.lowerToTapeChecked graph.data inputs data = .ok tape := by
    cases lowered : Runtime.Autograd.TypedGraph.lowerToTapeChecked graph.data inputs data with
    | error message =>
      simp only [vjpChecked, lowered, Bind.bind, Except.bind] at checked
      cases checked
    | ok tape => exact ⟨tape, rfl⟩
  have sameResult := Except.ok.inj ((vjpChecked_eq graph inputs data seed tape lowered).symm.trans
    checked)
  rw [← sameResult]
  exact vjpWithSeed_adjoint_fderiv graph proofGraph same inputs data seed correct

end Runtime.Autograd.Torch.TypedGraphWithData
