/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.TypedGraph.WithData
public import TorchLean.NN.Tape.Lowering.FDeriv
public import TorchLean.NN.Tape.RealNode.Context
public import TorchLean.NN.Tape.Lowering.Compiled

/-!
# Correctness of checked differentiation

The public autograd transforms validate and evaluate the graph before sweeping saved reverse rules
from its selected output. The theorems here connect that checked path to the graph semantics and,
for analytically correct real graphs, to mathlib's Fréchet derivative. Validation and analytic
correctness (`RealGraph.FDerivCorrectAt`) are separate hypotheses, since passing the runtime domain
checks says nothing about derivatives.
-/

@[expose] public section

namespace TorchLean.TypedGraph.WithData

open TensorPack Tensor Tape.RealGraph

/-- If checked lowering succeeds, the checked VJP returns the pure `vjp` of the stored reverse
rules and the forward value, with no algebraic laws on the scalar. `vjp` visits only the output's
dependencies when storage equality and certified compact support are available, and
`vjp_eq_backpropCtx` identifies it with the dense reference on carriers without storage equality.
-/
theorem vjpChecked_eq {α Δ : Type} [Storage α] [Add α] [Zero α]
    {Γ : List Shape} {τ : Shape}
    (graph : WithData α Δ Γ τ) (inputs : TensorPack α Γ) (data : Δ)
    (seed : Tensor α τ)
    (result : Tape (SomeTensor α) × TensorPack α (Γ ++ graph.nodeShapes))
    (checked : TypedGraph.lowerToTapeChecked graph.data inputs data =
      .ok result) :
    graph.vjpChecked inputs data seed =
      .ok (graph.vjp inputs data seed, graph.forward inputs data) := by
  cases saved : Tape.SavedGraph.lowerChecked graph.data inputs data with
  | error message =>
      have lowered := TypedGraph.compileChecked_eq
        graph.data inputs data
      simp only [TypedGraph.compileChecked, saved, Except.map, checked] at lowered
      cases lowered
  | ok execution => exact vjpChecked_eq_of_lowerChecked graph inputs data seed execution saved

/-- If `vjpChecked` succeeds with `result`, then `result` is the graph pullback `vjp` paired with
the forward value. The only hypothesis is that success at the executed input, and the scalar needs
no algebraic laws. -/
theorem eq_of_vjpChecked_ok {α Δ : Type} [Storage α] [Add α] [Zero α]
    {Γ : List Shape} {τ : Shape}
    (graph : WithData α Δ Γ τ) (inputs : TensorPack α Γ) (data : Δ)
    (seed : Tensor α τ) (result : TensorPack α Γ × Tensor α τ)
    (checked : graph.vjpChecked inputs data seed = .ok result) :
    result = (graph.vjp inputs data seed, graph.forward inputs data) := by
  obtain ⟨execution, lowered⟩ : ∃ execution,
      Tape.SavedGraph.lowerChecked graph.data inputs data = .ok execution := by
    cases lowered : Tape.SavedGraph.lowerChecked graph.data inputs data with
    | error message =>
      simp only [vjpChecked, TypedGraph.compileChecked, lowered, Except.map,
        Bind.bind, Except.bind] at checked
      cases checked
    | ok execution => exact ⟨execution, rfl⟩
  exact (Except.ok.inj
    ((vjpChecked_eq_of_lowerChecked graph inputs data seed execution lowered).symm.trans
      checked)).symm

/-- On a carrier without storage equality (`Storage.decEq? = none`), the single-output `vjp` is
the dense reference `backpropCtx` seeded with `single graph.output seed`. Real-valued and symbolic
carriers fall in this case, which connects them to the derivative theorems about `backpropCtx`. -/
theorem vjp_eq_backpropCtx {α Δ : Type} [Storage α] [Add α] [Zero α]
    {Γ : List Shape} {τ : Shape} (graph : WithData α Δ Γ τ)
    (inputs : TensorPack α Γ) (data : Δ) (seed : Tensor α τ)
    (equality : Storage.decEq? (α := α) = none) :
    graph.vjp inputs data seed = graph.data.backpropCtx inputs data
      (TensorPack.single graph.output seed) := by
  simp only [vjp, Tape.SavedGraph.backpropFrom, equality]
  rw [Tape.SavedGraph.ofGraph_backpropStored]
  exact Tape.GraphData.takeLeft_backpropAllCtx _ _ _ _

/-- If `proofGraph` has the executed graph's data and that data is `Tape.RealGraph.FDerivCorrectAt`
at `inputs`, the selected output's forward map has as derivative the graph derivative projected
onto that output. -/
theorem hasFDerivAt_forward {Δ : Type} {Γ : List Shape} {τ : Shape}
    (graph : WithData ℝ Δ Γ τ)
    (proofGraph : Tape.Graph (α := ℝ) Δ Γ graph.nodeShapes)
    (same : proofGraph.toData = graph.data)
    (inputs : TensorPack ℝ Γ) (data : Δ)
    (correct : FDerivCorrectAt proofGraph.toData data (TensorPack.toVec inputs)) :
    HasFDerivAt (fun x => Tensor.toVec (graph.forward (TensorPack.ofVec x) data))
      ((CtxVec.getCLM graph.output).comp
        (fderiv ℝ (Tape.GraphData.evalVec proofGraph.toData data) (TensorPack.toVec inputs)))
      (TensorPack.toVec inputs) := by
  have forward_eq : (fun x => Tensor.toVec (graph.forward (TensorPack.ofVec x) data)) =
      (CtxVec.getCLM graph.output) ∘ Tape.GraphData.evalVec proofGraph.toData data := by
    funext x
    rw [Function.comp_apply, CtxVec.getCLM_apply, ← TensorPack.toVec_ofVec x,
      Tape.GraphData.evalVec_flattenCtx, CtxVec.get_toVec]
    simp only [TensorPack.ofVec_toVec, forward, same]
  obtain ⟨D, hD, _⟩ := Tape.GraphData.hasFDerivAt_evalVec_and_jvp
    proofGraph.toData data (TensorPack.toVec inputs) correct
  rw [forward_eq, hD.fderiv]
  exact (CtxVec.getCLM graph.output).hasFDerivAt.comp _ hD

/-- Under the hypotheses of `hasFDerivAt_forward`, the output tangent of a successful `jvpChecked`
is mathlib's derivative of the selected output applied to the input direction. -/
theorem jvpChecked_eq_fderiv {Δ : Type} {Γ : List Shape} {τ : Shape}
    (graph : WithData ℝ Δ Γ τ)
    (proofGraph : Tape.Graph (α := ℝ) Δ Γ graph.nodeShapes)
    (same : proofGraph.toData = graph.data)
    (inputs tangents : TensorPack ℝ Γ) (data : Δ)
    (correct : FDerivCorrectAt proofGraph.toData data (TensorPack.toVec inputs))
    (result : TensorPack ℝ (Γ ++ graph.nodeShapes) × TensorPack ℝ (Γ ++ graph.nodeShapes))
    (checked : TypedGraph.jvpChecked graph.data inputs tangents data =
      .ok result) :
    Tensor.toVec (TensorPack.getIdx result.2 graph.output) =
      fderiv ℝ (fun x => Tensor.toVec (graph.forward (TensorPack.ofVec x) data))
        (TensorPack.toVec inputs) (TensorPack.toVec tangents) := by
  rw [TypedGraph.jvpChecked_eq _ _ _ _ _ checked,
    (hasFDerivAt_forward graph proofGraph same inputs data correct).fderiv,
    ContinuousLinearMap.comp_apply, CtxVec.getCLM_apply,
    ← Tape.GraphData.jvpVec_eq_fderiv _ _ _ _ correct, Tape.GraphData.jvpVec_flattenCtx,
    CtxVec.get_toVec]
  simp only [same]

/-- With `same` and `correct` as in `hasFDerivAt_forward`, the pure pullback `vjp` is the adjoint
of mathlib's derivative of the selected output, applied to the seed. -/
theorem vjp_eq_fderiv {Δ : Type} {Γ : List Shape} {τ : Shape}
    (graph : WithData ℝ Δ Γ τ)
    (proofGraph : Tape.Graph (α := ℝ) Δ Γ graph.nodeShapes)
    (same : proofGraph.toData = graph.data)
    (inputs : TensorPack ℝ Γ) (data : Δ) (seed : Tensor ℝ τ)
    (correct : FDerivCorrectAt proofGraph.toData data (TensorPack.toVec inputs)) :
    TensorPack.toVec (graph.vjp inputs data seed) =
      (fderiv ℝ (fun x => Tensor.toVec (graph.forward (TensorPack.ofVec x) data))
        (TensorPack.toVec inputs)).adjoint (Tensor.toVec seed) := by
  have seed_eq : (CtxVec.getCLM graph.output).adjoint (Tensor.toVec seed) =
      TensorPack.toVec (TensorPack.single graph.output seed) := by
    apply ext_inner_left ℝ
    intro x
    rw [ContinuousLinearMap.adjoint_inner_right, CtxVec.getCLM_apply,
      ← TensorPack.toVec_ofVec x, CtxVec.get_toVec,
      ← realDotList_eq_inner_toVec, TensorPack.dotList_single, dot_eq_inner_toVec]
  rw [(hasFDerivAt_forward graph proofGraph same inputs data correct).fderiv,
    ContinuousLinearMap.adjoint_comp, ContinuousLinearMap.comp_apply, seed_eq]
  rw [vjp_eq_backpropCtx graph inputs data seed (by rfl), ← same]
  exact Tape.Graph.backpropCtx_eq_fderiv proofGraph data inputs correct _

/--
If `vjpChecked` succeeds, its input gradients are the adjoint of mathlib's derivative of the
selected forward output, applied to the seed.

The proof graph must describe the exact operation data that execution uses, and differentiability
is required only at this input and the intermediate values it reaches. Because the carrier is
`ℝ`, the statement says nothing about floating-point error or native kernels.
-/
theorem vjpChecked_eq_fderiv {Δ : Type} {Γ : List Shape} {τ : Shape}
    (graph : WithData ℝ Δ Γ τ)
    (proofGraph : Tape.Graph (α := ℝ) Δ Γ graph.nodeShapes)
    (same : proofGraph.toData = graph.data)
    (inputs : TensorPack ℝ Γ) (data : Δ) (seed : Tensor ℝ τ)
    (correct : FDerivCorrectAt proofGraph.toData data (TensorPack.toVec inputs))
    (result : TensorPack ℝ Γ × Tensor ℝ τ)
    (checked : graph.vjpChecked inputs data seed = .ok result) :
    TensorPack.toVec result.1 =
      (fderiv ℝ (fun x => Tensor.toVec (graph.forward (TensorPack.ofVec x) data))
        (TensorPack.toVec inputs)).adjoint (Tensor.toVec seed) := by
  rw [eq_of_vjpChecked_ok graph inputs data seed result checked]
  exact vjp_eq_fderiv graph proofGraph same inputs data seed correct

end TorchLean.TypedGraph.WithData
