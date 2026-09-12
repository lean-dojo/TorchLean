/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Tape.Ops.Norm.LayerNormEval
public import NN.Proofs.Autograd.Tape.Ops.Norm.RowNormalization
public import NN.Proofs.Autograd.Tape.Ops.Norm.MatrixEntries
public import NN.Proofs.Autograd.Tape.Ops.Norm.LayerNormAdjoint

/-!
# LayerNorm

Pointwise analytic correctness for a **LayerNorm** graph.

This is spec-level over `ℝ`. It is the proof-tape counterpart of the runtime/spec LayerNorm in
`Spec.layerNorm`: a `seqLen × embedDim` tensor is normalized across the last axis, the row-wise
normalizer is broadcast back over each token, and affine parameters `gamma`/`beta` are broadcast
over the sequence dimension. The runtime API and typed graph path both route through that spec
definition; this file proves the corresponding reverse-mode graph rule.

Because the proof graph uses the differentiable scalar nodes `sqrt (max x 0)` and `inv`, the main
theorem is pointwise (`GraphFDerivCorrectAt`). The `_of_domain` variants take the two domain
assumptions (positive `var + ε`, nonzero `std`) at the execution point; the main statements take
only `0 < ε`, since the row variance is a mean of squares and therefore nonnegative
(`LayerNormEval.varEps_pos_of_eps_pos`, `LayerNormEval.std_ne_zero_of_eps_pos`). Away from the
clamp kink and zero denominator, backprop is the adjoint of the Fréchet derivative. The executable
`Spec.layerNorm` additionally clamps the raw variance before adding epsilon as a numerical guard;
over exact real variance this is the same contract on the positive branch used by the proof.

The last section connects the graph to the spec. `outputCLM_evalVec_layerNormGraph` shows that the
output block of the graph evaluation is `tensorToVec (Spec.layerNorm X gamma beta)` for the packed
inputs, so `hasFDerivAt_specLayerNormVec` and `backpropVec_single_eq_adjoint_specLayerNorm` state
differentiability and the adjoint rule for the spec function itself. The adjointness of
`Spec.layerNormJvp` and `Spec.layerNormBackward` is proved in
`NN.Proofs.Autograd.Tape.Ops.Norm.LayerNormAdjoint`. The companion `LayerNormFDeriv` module
identifies that JVP with the actual derivative and proves that the primitive backward rule
returns the same cotangents as this graph.

## PyTorch correspondence / citations
- Conceptually corresponds to `torch.nn.LayerNorm` (without batching/running stats): normalize along
  the last dimension, then apply affine parameters `(gamma,beta)`.
  https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
-/


@[expose] public section


namespace Proofs
namespace Autograd

open Spec TorchLean

open scoped BigOperators

noncomputable section

namespace LayerNorm

open TapeNodes

-- ---------------------------------------------------------------------------
-- Pointwise `GraphFDerivCorrectAt` for LayerNorm
-- ---------------------------------------------------------------------------

/--
Pointwise proof that `layerNormGraph` satisfies `GraphFDerivCorrectAt`, from explicit domain
assumptions.

The hypotheses `hVarEpsPos` and `hStdNe0` ensure that `sqrt` and `inv` are differentiable at the
execution point. `layerNormGraphFderivCorrectAt` discharges both from `0 < ε`.
-/
def layerNormGraphFderivCorrectAtOfDomain
    {m n : Nat} (ε : ℝ) (xV : CtxVec (ΓLN m n))
    (hVarEpsPos :
      ∀ i : Fin (Spec.Shape.size (VecShape m)),
        0 < CtxVec.get (Γ := ΓLN m n ++ ssPrefix6 m n) (s := VecShape m) (idxVarEps (m := m) (n :=
          n))
          (Graph.evalVec (Γ := ΓLN m n) (ss := ssPrefix6 m n) (layerNormPrefix6 (m := m) (n := n) ε)
            xV) i)
    (hStdNe0 :
      ∀ i : Fin (Spec.Shape.size (VecShape m)),
        CtxVec.get (Γ := ΓLN m n ++ ssPrefix7 m n) (s := VecShape m) (idxStd (m := m) (n := n))
          (Graph.evalVec (Γ := ΓLN m n) (ss := ssPrefix7 m n) (layerNormPrefix7 (m := m) (n := n) ε)
            xV) i ≠ 0) :
    GraphFDerivCorrectAt (Γ := ΓLN m n) (ss := ssLayerNorm m n) (layerNormGraph (m := m) (n := n) ε)
      xV := by
  classical
  -- Prefix 6
  have hg0 : GraphFDerivCorrectAt (Γ := ΓLN m n) (ss := []) (.nil) xV := PUnit.unit
  have hg1 :
      GraphFDerivCorrectAt (Γ := ΓLN m n) (ss := [VecShape m]) (g1 (m := m) (n := n)) xV := by
    refine ⟨hg0, ?_⟩
    exact
      (rowMeanFderiv (idx := idxX (m := m) (n := n) (ss := []))).at
        (Graph.evalVec (Γ := ΓLN m n) (ss := []) (.nil) xV)
  have hg2 :
      GraphFDerivCorrectAt (Γ := ΓLN m n) (ss := [VecShape m, MatShape m n]) (g2 (m := m) (n := n))
        xV := by
    refine ⟨hg1, ?_⟩
    exact
      (broadcastRowFderiv (idx := idxMean (m := m) (n := n))).at
        (Graph.evalVec (Γ := ΓLN m n) (ss := [VecShape m]) (g1 (m := m) (n := n)) xV)
  have hg3 :
      GraphFDerivCorrectAt (Γ := ΓLN m n) (ss := [VecShape m, MatShape m n, MatShape m n]) (g3 (m :=
        m) (n := n)) xV := by
    refine ⟨hg2, ?_⟩
    exact
      (subFderiv (s := MatShape m n)
        (a := idxX (m := m) (n := n) (ss := [VecShape m, MatShape m n]))
        (b := idxMeanB (m := m) (n := n))).at
        (Graph.evalVec (Γ := ΓLN m n) (ss := [VecShape m, MatShape m n]) (g2 (m := m) (n := n)) xV)
  have hg4 :
      GraphFDerivCorrectAt (Γ := ΓLN m n) (ss := [VecShape m, MatShape m n, MatShape m n, MatShape m
        n]) (g4 (m := m) (n := n)) xV := by
    refine ⟨hg3, ?_⟩
    exact
      (mulFderiv (s := MatShape m n) (a := idxCentered (m := m) (n := n)) (b := idxCentered (m :=
        m) (n := n))).at
        (Graph.evalVec (Γ := ΓLN m n) (ss := [VecShape m, MatShape m n, MatShape m n]) (g3 (m := m)
          (n := n)) xV)
  have hg5 :
      GraphFDerivCorrectAt (Γ := ΓLN m n) (ss := [VecShape m, MatShape m n, MatShape m n, MatShape m
        n, VecShape m])
        (g5 (m := m) (n := n)) xV := by
    refine ⟨hg4, ?_⟩
    exact
      (rowMeanFderiv (idx := idxCenteredSq (m := m) (n := n))).at
        (Graph.evalVec (Γ := ΓLN m n) (ss := [VecShape m, MatShape m n, MatShape m n, MatShape m n])
          (g4 (m := m) (n := n)) xV)
  have hg6 :
      GraphFDerivCorrectAt (Γ := ΓLN m n) (ss := ssPrefix6 m n) (layerNormPrefix6 (m := m) (n := n)
        ε) xV := by
    refine ⟨hg5, ?_⟩
    have hderiv : NodeFDerivCorrect (nodeVarEps (m := m) (n := n) ε) :=
      elemwiseFderiv (Γ := ΓLN m n ++ [VecShape m, MatShape m n, MatShape m n, MatShape m n,
        VecShape m])
        (s := VecShape m) (idx := idxVar (m := m) (n := n))
        (f := fun z => z + ε) (f' := fun _ => 1) (hf := fun z => (hasDerivAt_id z).add_const ε)
    exact
      NodeFDerivCorrect.at hderiv
        (Graph.evalVec (Γ := ΓLN m n)
          (ss := [VecShape m, MatShape m n, MatShape m n, MatShape m n, VecShape m]) (g5 (m := m) (n
            := n)) xV)

  -- Prefix 7 (std)
  have hg7 :
      GraphFDerivCorrectAt (Γ := ΓLN m n) (ss := ssPrefix7 m n) (layerNormPrefix7 (m := m) (n := n)
        ε) xV := by
    refine ⟨hg6, ?_⟩
    have hStdAt :
        NodeFDerivCorrectAt (nodeStd (m := m) (n := n))
          (Graph.evalVec (Γ := ΓLN m n) (ss := ssPrefix6 m n) (layerNormPrefix6 (m := m) (n := n) ε)
            xV) :=
      sqrtClampFderivAt (Γ := ΓLN m n ++ ssPrefix6 m n) (s := VecShape m) (idx := idxVarEps (m :=
        m) (n := n))
        (xV := Graph.evalVec (Γ := ΓLN m n) (ss := ssPrefix6 m n) (layerNormPrefix6 (m := m) (n :=
          n) ε) xV)
        (hx := hVarEpsPos)
    simpa [layerNormPrefix7, nodeStd] using hStdAt

  -- inv_std
  have hg8 :
      GraphFDerivCorrectAt (Γ := ΓLN m n) (ss := ssPrefix7 m n ++ [VecShape m]) (g8 (m := m) (n :=
        n) ε) xV := by
    refine ⟨hg7, ?_⟩
    have hInvAt :
        NodeFDerivCorrectAt (nodeInvStd (m := m) (n := n))
          (Graph.evalVec (Γ := ΓLN m n) (ss := ssPrefix7 m n) (layerNormPrefix7 (m := m) (n := n) ε)
            xV) :=
      invFderivAt (Γ := ΓLN m n ++ ssPrefix7 m n) (s := VecShape m) (idx := idxStd (m := m) (n :=
        n))
        (xV := Graph.evalVec (Γ := ΓLN m n) (ss := ssPrefix7 m n) (layerNormPrefix7 (m := m) (n :=
          n) ε) xV)
        (hx := hStdNe0)
    simpa [g8, nodeInvStd] using hInvAt

  -- inv_std_b
  have hg9 :
      GraphFDerivCorrectAt (Γ := ΓLN m n) (ss := ssPrefix7 m n ++ [VecShape m, MatShape m n]) (g9 (m
        := m) (n := n) ε) xV := by
    refine ⟨hg8, ?_⟩
    exact
      (broadcastRowFderiv (idx := idxInvStd (m := m) (n := n))).at
        (Graph.evalVec (Γ := ΓLN m n) (ss := ssPrefix7 m n ++ [VecShape m]) (g8 (m := m) (n := n) ε)
          xV)

  -- normalized
  have hg10 :
      GraphFDerivCorrectAt (Γ := ΓLN m n) (ss := ssPrefix7 m n ++ [VecShape m, MatShape m n,
        MatShape m n]) (g10 (m := m) (n := n) ε) xV := by
    refine ⟨hg9, ?_⟩
    exact
      (mulFderiv (s := MatShape m n) (a := idxCentered9 (m := m) (n := n)) (b := idxInvStdB9 (m :=
        m) (n := n))).at
        (Graph.evalVec (Γ := ΓLN m n) (ss := ssPrefix7 m n ++ [VecShape m, MatShape m n]) (g9 (m :=
          m) (n := n) ε) xV)

  -- gamma_b
  have hg11 :
      GraphFDerivCorrectAt (Γ := ΓLN m n) (ss := ssPrefix7 m n ++ [VecShape m, MatShape m n,
        MatShape m n, MatShape m n])
        (g11 (m := m) (n := n) ε) xV := by
    refine ⟨hg10, ?_⟩
    exact
      (broadcastColFderiv
        (idx := idxGamma (m := m) (n := n) (ss := ssPrefix7 m n ++ [VecShape m, MatShape m n,
          MatShape m n]))).at
        (Graph.evalVec (Γ := ΓLN m n) (ss := ssPrefix7 m n ++ [VecShape m, MatShape m n, MatShape m
          n]) (g10 (m := m) (n := n) ε) xV)

  -- scaled
  have hg12 :
      GraphFDerivCorrectAt (Γ := ΓLN m n) (ss := ssPrefix7 m n ++ [VecShape m, MatShape m n,
        MatShape m n, MatShape m n, MatShape m n])
        (g12 (m := m) (n := n) ε) xV := by
    refine ⟨hg11, ?_⟩
    exact
      (mulFderiv (s := MatShape m n) (a := idxNorm11 (m := m) (n := n)) (b := idxGammaB11 (m := m)
        (n := n))).at
        (Graph.evalVec (Γ := ΓLN m n) (ss := ssPrefix7 m n ++ [VecShape m, MatShape m n, MatShape m
          n, MatShape m n]) (g11 (m := m) (n := n) ε) xV)

  -- beta_b
  have hg13 :
      GraphFDerivCorrectAt (Γ := ΓLN m n) (ss := ssPrefix7 m n ++ [VecShape m, MatShape m n,
        MatShape m n, MatShape m n, MatShape m n, MatShape m n])
        (g13 (m := m) (n := n) ε) xV := by
    refine ⟨hg12, ?_⟩
    exact
      (broadcastColFderiv
        (idx := idxBeta (m := m) (n := n)
          (ss := ssPrefix7 m n ++ [VecShape m, MatShape m n, MatShape m n, MatShape m n, MatShape m
            n]))).at
        (Graph.evalVec (Γ := ΓLN m n)
          (ss := ssPrefix7 m n ++ [VecShape m, MatShape m n, MatShape m n, MatShape m n, MatShape m
            n]) (g12 (m := m) (n := n) ε) xV)

  -- y
  refine ⟨hg13, ?_⟩
  exact
    (addFderiv (s := MatShape m n) (a := idxScaled13 (m := m) (n := n)) (b := idxBetaB13 (m := m)
      (n := n))).at
      (Graph.evalVec (Γ := ΓLN m n)
        (ss := ssPrefix7 m n ++ [VecShape m, MatShape m n, MatShape m n, MatShape m n, MatShape m n,
          MatShape m n])
        (g13 (m := m) (n := n) ε) xV)

/-- Pointwise proof that `layerNormGraph` satisfies `GraphFDerivCorrectAt` whenever `0 < ε`. -/
def layerNormGraphFderivCorrectAt
    {m n : Nat} (ε : ℝ) (xV : CtxVec (ΓLN m n)) (hε : 0 < ε) :
    GraphFDerivCorrectAt (Γ := ΓLN m n) (ss := ssLayerNorm m n) (layerNormGraph (m := m) (n := n) ε)
      xV :=
  layerNormGraphFderivCorrectAtOfDomain (m := m) (n := n) ε xV
    (varEps_pos_of_eps_pos hε xV) (std_ne_zero_of_eps_pos hε xV)

/--
Pointwise end-to-end result from explicit domain assumptions: backprop equals `(fderiv eval)†`
for `layerNormGraph`.

The hypotheses `hVarEpsPos` and `hStdNe0` are the domain assumptions needed for differentiability
of `sqrt` (after clamp) and `inv` at the actual execution point.
-/
theorem backprop_eq_adjoint_fderiv_layerNorm_at_of_domain
    {m n : Nat} (ε : ℝ)
    (xV : CtxVec (ΓLN m n))
    (seedV : CtxVec (ΓLN m n ++ ssLayerNorm m n))
    (hVarEpsPos :
      ∀ i : Fin (Spec.Shape.size (VecShape m)),
        0 < CtxVec.get (Γ := ΓLN m n ++ ssPrefix6 m n) (s := VecShape m) (idxVarEps (m := m) (n :=
          n))
          (Graph.evalVec (Γ := ΓLN m n) (ss := ssPrefix6 m n) (layerNormPrefix6 (m := m) (n := n) ε)
            xV) i)
    (hStdNe0 :
      ∀ i : Fin (Spec.Shape.size (VecShape m)),
        CtxVec.get (Γ := ΓLN m n ++ ssPrefix7 m n) (s := VecShape m) (idxStd (m := m) (n := n))
          (Graph.evalVec (Γ := ΓLN m n) (ss := ssPrefix7 m n) (layerNormPrefix7 (m := m) (n := n) ε)
            xV) i ≠ 0) :
    Graph.backpropVec (Γ := ΓLN m n) (ss := ssLayerNorm m n) (layerNormGraph (m := m) (n := n) ε) xV
      seedV
      =
    (fderiv ℝ
        (Graph.evalVec (Γ := ΓLN m n) (ss := ssLayerNorm m n) (layerNormGraph (m := m) (n := n) ε))
        xV).adjoint seedV := by
  classical
  have hg :
      GraphFDerivCorrectAt (Γ := ΓLN m n) (ss := ssLayerNorm m n) (layerNormGraph (m := m) (n := n)
        ε) xV :=
    layerNormGraphFderivCorrectAtOfDomain (m := m) (n := n) ε xV hVarEpsPos hStdNe0
  exact
    Graph.backpropVec_eq_adjoint_fderiv_at (Γ := ΓLN m n) (ss := ssLayerNorm m n)
      (g := layerNormGraph (m := m) (n := n) ε) xV seedV hg

/-- Pointwise end-to-end result: backprop equals `(fderiv eval)†` for `layerNormGraph` whenever
`0 < ε`. -/
theorem backprop_eq_adjoint_fderiv_layerNorm_at
    {m n : Nat} (ε : ℝ)
    (xV : CtxVec (ΓLN m n))
    (seedV : CtxVec (ΓLN m n ++ ssLayerNorm m n)) (hε : 0 < ε) :
    Graph.backpropVec (Γ := ΓLN m n) (ss := ssLayerNorm m n) (layerNormGraph (m := m) (n := n) ε) xV
      seedV
      =
    (fderiv ℝ
        (Graph.evalVec (Γ := ΓLN m n) (ss := ssLayerNorm m n) (layerNormGraph (m := m) (n := n) ε))
        xV).adjoint seedV :=
  Graph.backpropVec_eq_adjoint_fderiv_at (Γ := ΓLN m n) (ss := ssLayerNorm m n)
    (g := layerNormGraph (m := m) (n := n) ε) xV seedV
    (layerNormGraphFderivCorrectAt (m := m) (n := n) ε xV hε)

-- ---------------------------------------------------------------------------
-- Generic whole-node adapter
-- ---------------------------------------------------------------------------

/--
LayerNorm inputs inside an arbitrary tape context.

This is the model-level interface we use once LayerNorm is no longer the root graph. For example,
in a post-norm Transformer block, `x` is the residual stream produced by an earlier SSA node, while
`gamma` and `beta` are carried parameters in the surrounding context.
-/
structure Inputs (Γ : List Shape) (m n : Nat) where
  /-- Sequence/residual matrix normalized across its last axis. -/
  x : Idx Γ (MatShape m n)
  /-- Affine scale vector. -/
  gamma : Idx Γ (VecShape n)
  /-- Affine shift vector. -/
  beta : Idx Γ (VecShape n)

/--
Linear map that packs arbitrary-context LayerNorm inputs into the canonical context
`[X, gamma, beta]`.
-/
def packInputsCLM {Γ : List Shape} {m n : Nat} (inputs : Inputs Γ m n) :
    CtxVec Γ →L[ℝ] CtxVec (ΓLN m n) := by
  let xCLM : CtxVec Γ →L[ℝ] Vec (Spec.Shape.size (MatShape m n)) :=
    CtxVec.getCLM (Γ := Γ) (s := MatShape m n) inputs.x
  let gammaCLM : CtxVec Γ →L[ℝ] Vec (Spec.Shape.size (VecShape n)) :=
    CtxVec.getCLM (Γ := Γ) (s := VecShape n) inputs.gamma
  let betaCLM : CtxVec Γ →L[ℝ] Vec (Spec.Shape.size (VecShape n)) :=
    CtxVec.getCLM (Γ := Γ) (s := VecShape n) inputs.beta
  let gbCLM :=
    (Graph.appendCLM (Spec.Shape.size (VecShape n)) (Spec.Shape.size (VecShape n))).comp
      (gammaCLM.prod betaCLM)
  let allCLM :=
    (Graph.appendCLM (Spec.Shape.size (MatShape m n))
      (Spec.Shape.size (VecShape n) + Spec.Shape.size (VecShape n))).comp
      (xCLM.prod gbCLM)
  let h :
      Spec.Shape.size (MatShape m n) + (Spec.Shape.size (VecShape n) + Spec.Shape.size (VecShape n))
        =
      ctxSize (ΓLN m n) := by
    simp [ctxSize, Spec.Shape.size]
  exact (Graph.castCLM (h := h)).comp allCLM

/-- Project the final LayerNorm output from the full canonical graph context. -/
def outputCLM {m n : Nat} :
    CtxVec (ΓLN m n ++ ssLayerNorm m n) →L[ℝ] Vec (Spec.Shape.size (MatShape m n)) :=
  CtxVec.getCLM (Γ := ΓLN m n ++ ssLayerNorm m n) (s := MatShape m n) (idxY (m := m) (n := n))

/--
LayerNorm as one reusable pointwise node over arbitrary context indices.

Internally this node runs the already-proved detailed LayerNorm graph. Its JVP is defined as the
Fréchet derivative of that composed map at the current point, and its VJP is the adjoint of that
derivative. This is exactly the block-level abstraction needed for large model proofs: the detailed
LayerNorm proof remains in this file, while Transformer/GPT/ViT proofs can treat LayerNorm as a
single pointwise node with explicit domain assumptions.
-/
def wholeNode {Γ : List Shape} {m n : Nat} (inputs : Inputs Γ m n) (ε : ℝ) :
    Node Γ (MatShape m n) :=
  let pack := packInputsCLM (Γ := Γ) (m := m) (n := n) inputs
  let f : CtxVec Γ → Vec (Spec.Shape.size (MatShape m n)) :=
    fun xV =>
      outputCLM (m := m) (n := n)
        (Graph.evalVec (Γ := ΓLN m n) (ss := ssLayerNorm m n)
          (layerNormGraph (m := m) (n := n) ε) (pack xV))
  Node.ofFn (Γ := Γ) (τ := MatShape m n)
    (f := f)
    (jvp := fun xV dxV => (fderiv ℝ f xV) dxV)
    (vjp := fun xV δV => (fderiv ℝ f xV).adjoint δV)
    (correct_inner := by
      intro xV dxV δV
      simpa using
        (ContinuousLinearMap.adjoint_inner_right (A := fderiv ℝ f xV) (x := dxV) (y := δV)).symm)

/--
Pointwise derivative certificate for `wholeNode`.

Only `0 < ε` is needed: the row variance computed inside the packed LayerNorm graph is a mean of
squares, so `var + ε` is positive and the clamped standard deviation is nonzero at every point.
-/
def wholeNodeFDerivCorrectAt {Γ : List Shape} {m n : Nat}
    (inputs : Inputs Γ m n) (ε : ℝ) (xV : CtxVec Γ) (hε : 0 < ε) :
    NodeFDerivCorrectAt (wholeNode (Γ := Γ) (m := m) (n := n) inputs ε) xV := by
  classical
  let pack := packInputsCLM (Γ := Γ) (m := m) (n := n) inputs
  let g := layerNormGraph (m := m) (n := n) ε
  let out := outputCLM (m := m) (n := n)
  let f : CtxVec Γ → Vec (Spec.Shape.size (MatShape m n)) :=
    fun z =>
      out (Graph.evalVec (Γ := ΓLN m n) (ss := ssLayerNorm m n) g (pack z))
  have hgAt :
      GraphFDerivCorrectAt (Γ := ΓLN m n) (ss := ssLayerNorm m n) g (pack xV) :=
    layerNormGraphFderivCorrectAt (m := m) (n := n) ε (pack xV) hε
  let hEval := Graph.hasFDerivAt_evalVec_and_jvp_at
      (Γ := ΓLN m n) (ss := ssLayerNorm m n) (g := g) (xV := pack xV) hgAt
  let Dg := Classical.choose hEval
  have hDg :
      HasFDerivAt (Graph.evalVec (Γ := ΓLN m n) (ss := ssLayerNorm m n) g) Dg (pack xV) :=
    (Classical.choose_spec hEval).1
  have hEvalComp :
      HasFDerivAt
        (fun z : CtxVec Γ => Graph.evalVec (Γ := ΓLN m n) (ss := ssLayerNorm m n) g (pack z))
        (Dg.comp pack) xV :=
    hDg.comp xV (pack.hasFDerivAt (x := xV))
  have hOut :
      HasFDerivAt
        (fun z : CtxVec Γ =>
          out (Graph.evalVec (Γ := ΓLN m n) (ss := ssLayerNorm m n) g (pack z)))
        (out.comp (Dg.comp pack)) xV :=
    out.hasFDerivAt.comp xV hEvalComp
  refine
    { deriv := fderiv ℝ f xV
      hasFDerivAt := ?_
      jvp_eq := ?_ }
  · have hfEq :
        (fun z : CtxVec Γ =>
          out (Graph.evalVec (Γ := ΓLN m n) (ss := ssLayerNorm m n) g (pack z))) = f := by
      rfl
    have hFderiv :
        fderiv ℝ f xV = out.comp (Dg.comp pack) := by
      rw [← hfEq]
      exact hOut.fderiv
    rw [hFderiv]
    simpa [wholeNode, f, pack, g, out] using hOut
  · intro dxV
    simp [wholeNode, f, pack, g, out]

-- ---------------------------------------------------------------------------
-- Bridge to `Spec.layerNorm`
-- ---------------------------------------------------------------------------

/-! ## The graph computes `Spec.layerNorm`

The packed context `[X, gamma, beta]` determines three spec tensors. Evaluating the LayerNorm graph
on it and reading the output block gives exactly `tensorToVec (Spec.layerNorm X gamma beta)`.
Consequently the spec function is differentiable wherever the graph is, with the graph's backprop
as the adjoint of its derivative. -/

open TapeNodes.Matmul in
/-- The input matrix of a packed LayerNorm context, as a spec tensor. -/
def specX {m n : Nat} (xV : CtxVec (ΓLN m n)) : Tensor ℝ [m, n] :=
  vecToTensor (s := MatShape m n) (valX xV)

/-- The scale vector of a packed LayerNorm context, as a spec tensor. -/
def specGamma {m n : Nat} (xV : CtxVec (ΓLN m n)) : Tensor ℝ [n] :=
  vecToTensor (s := VecShape n) (CtxVec.get (Γ := ΓLN m n) (s := VecShape n) idxGamma0 xV)

/-- The shift vector of a packed LayerNorm context, as a spec tensor. -/
def specBeta {m n : Nat} (xV : CtxVec (ΓLN m n)) : Tensor ℝ [n] :=
  vecToTensor (s := VecShape n) (CtxVec.get (Γ := ΓLN m n) (s := VecShape n) idxBeta0 xV)

/-- `Spec.layerNorm` as a map on the packed context vector `[X, gamma, beta]`. -/
def specLayerNormVec {m n : Nat} (hm : 0 < m) (hn : 0 < n) (ε : ℝ) (xV : CtxVec (ΓLN m n)) :
    Vec (Spec.Shape.size (MatShape m n)) :=
  tensorToVec (Spec.layerNorm (specX xV) (specGamma xV) (specBeta xV) hm hn ε)

/-- Pack three spec tensors into a LayerNorm context vector. -/
def packLN {m n : Nat} (X : Tensor ℝ [m, n]) (gamma beta : Tensor ℝ [n]) : CtxVec (ΓLN m n) :=
  flattenCtx (Γ := ΓLN m n) (.cons X (.cons gamma (.cons beta .nil)))

section EntryLemmas

open TapeNodes.Matmul

variable {m n : Nat}

/-- Entries of the input block. -/
theorem valX_idxMN (xV : CtxVec (ΓLN m n)) (i : Fin m) (j : Fin n) :
    valX xV (idxMN (m := m) (n := n) i j) = Spec.get2 (specX xV) i j := by
  rw [specX, ← Norm.tensorToVec_idxMN, tensorToVec_vecToTensor]

/-- Entries of the scale block. -/
theorem valGamma_apply (xV : CtxVec (ΓLN m n)) (j : Fin n) :
    valGamma xV j = TorchLean.Tensor.getScalar (specGamma xV) j := by
  simp only [valGamma, getVec, castVec_apply]
  rw [show CtxVec.get (Γ := ΓLN m n) (s := VecShape n) idxGamma0 xV = tensorToVec (specGamma xV)
    from (tensorToVec_vecToTensor _).symm, Norm.tensorToVec_vec]
  rfl

/-- Entries of the shift block. -/
theorem valBeta_apply (xV : CtxVec (ΓLN m n)) (j : Fin n) :
    valBeta xV j = TorchLean.Tensor.getScalar (specBeta xV) j := by
  simp only [valBeta, getVec, castVec_apply]
  rw [show CtxVec.get (Γ := ΓLN m n) (s := VecShape n) idxBeta0 xV = tensorToVec (specBeta xV)
    from (tensorToVec_vecToTensor _).symm, Norm.tensorToVec_vec]
  rfl

/-- The mean block holds the spec row means. -/
theorem valMean_apply (xV : CtxVec (ΓLN m n)) (i : Fin m) :
    valMean xV (Fin.cast (vecShape_size m).symm i) = Norm.rowMeanE (specX xV) i := by
  show TapeNodes.MatrixLinear.rowMeanCLM (m := m) (n := n) (valX xV) i = _
  rw [RowNorm.rowMeanCLM_eq, RowNorm.rowMean, Norm.rowMeanE]
  congr 1
  exact Finset.sum_congr rfl fun j _ => valX_idxMN xV i j

/-- The broadcast mean block. -/
theorem valMeanB_idxMN (xV : CtxVec (ΓLN m n)) (i : Fin m) (j : Fin n) :
    valMeanB xV (idxMN (m := m) (n := n) i j) = Norm.rowMeanE (specX xV) i := by
  rw [valMeanB, RowNorm.broadcastRowCLM_idxMN, castVec_apply, valMean_apply]

/-- The centered block. -/
theorem valCentered_idxMN (xV : CtxVec (ΓLN m n)) (i : Fin m) (j : Fin n) :
    valCentered xV (idxMN (m := m) (n := n) i j) =
      Spec.get2 (specX xV) i j - Norm.rowMeanE (specX xV) i := by
  simp only [valCentered, PiLp.sub_apply, valX_idxMN, valMeanB_idxMN]

/-- The variance block holds the spec row variances. -/
theorem valVar_apply (xV : CtxVec (ΓLN m n)) (i : Fin m) :
    valVar xV (Fin.cast (vecShape_size m).symm i) = Norm.rowVarE (specX xV) i := by
  show TapeNodes.MatrixLinear.rowMeanCLM (m := m) (n := n) (valCenteredSq xV) i = _
  rw [RowNorm.rowMeanCLM_eq, RowNorm.rowMean, Norm.rowVarE]
  congr 1
  refine Finset.sum_congr rfl fun j _ => ?_
  simp only [valCenteredSq, vecOfFun_apply, valCentered_idxMN]

/-- The broadcast inverse standard deviation block. -/
theorem valInvStdB_idxMN (xV : CtxVec (ΓLN m n)) (ε : ℝ) (i : Fin m) (j : Fin n) :
    valInvStdB xV ε (idxMN (m := m) (n := n) i j) =
      (Real.sqrt (max (Norm.rowVarE (specX xV) i + ε) 0))⁻¹ := by
  rw [valInvStdB, RowNorm.broadcastRowCLM_idxMN, castVec_apply]
  simp only [valInvStd, valStd, valVarEps, vecOfFun_apply, valVar_apply]

/-- The output block, entrywise. -/
theorem valY_idxMN (xV : CtxVec (ΓLN m n)) (ε : ℝ) (i : Fin m) (j : Fin n) :
    valY xV ε (idxMN (m := m) (n := n) i j) =
      (Spec.get2 (specX xV) i j - Norm.rowMeanE (specX xV) i) *
          (Real.sqrt (max (Norm.rowVarE (specX xV) i + ε) 0))⁻¹ *
            TorchLean.Tensor.getScalar (specGamma xV) j +
        TorchLean.Tensor.getScalar (specBeta xV) j := by
  simp only [valY, PiLp.add_apply, valScaled, valNorm, vecOfFun_apply, valGammaB, valBetaB,
    RowNorm.broadcastColCLM_idxMN, valCentered_idxMN, valInvStdB_idxMN, valGamma_apply,
    valBeta_apply]

end EntryLemmas

/-- The spec LayerNorm of the packed inputs is the closed form of the graph output. -/
theorem specLayerNormVec_eq_valY {m n : Nat} (hm : 0 < m) (hn : 0 < n) (ε : ℝ)
    (xV : CtxVec (ΓLN m n)) :
    specLayerNormVec hm hn ε xV = valY xV ε := by
  apply Norm.vec_ext_idxMN
  intro i j
  rw [specLayerNormVec, Norm.tensorToVec_idxMN, Norm.get2_layerNorm hm hn, valY_idxMN,
    div_eq_mul_inv]

/-- Forward bridge: the output block of the LayerNorm graph is `Spec.layerNorm` of the packed
inputs. -/
theorem outputCLM_evalVec_layerNormGraph {m n : Nat} (hm : 0 < m) (hn : 0 < n) (ε : ℝ)
    (xV : CtxVec (ΓLN m n)) :
    outputCLM (m := m) (n := n)
        (Graph.evalVec (Γ := ΓLN m n) (ss := ssLayerNorm m n)
          (layerNormGraph (m := m) (n := n) ε) xV) =
      specLayerNormVec hm hn ε xV := by
  rw [specLayerNormVec_eq_valY, outputCLM, CtxVec.getCLM_apply, get_idxY]

/-- The spec LayerNorm map is the output projection of the graph evaluation. -/
theorem specLayerNormVec_eq {m n : Nat} (hm : 0 < m) (hn : 0 < n) (ε : ℝ) :
    specLayerNormVec (m := m) (n := n) hm hn ε =
      fun xV =>
        outputCLM (m := m) (n := n)
          (Graph.evalVec (Γ := ΓLN m n) (ss := ssLayerNorm m n)
            (layerNormGraph (m := m) (n := n) ε) xV) := by
  funext xV
  exact (outputCLM_evalVec_layerNormGraph hm hn ε xV).symm

/-- `Spec.layerNorm` is differentiable in `[X, gamma, beta]` for `0 < ε`, with derivative the
output projection of the graph derivative. -/
theorem hasFDerivAt_specLayerNormVec {m n : Nat} (hm : 0 < m) (hn : 0 < n) {ε : ℝ}
    (hε : 0 < ε) (xV : CtxVec (ΓLN m n)) :
    HasFDerivAt (specLayerNormVec hm hn ε)
      ((outputCLM (m := m) (n := n)).comp
        (fderiv ℝ
          (Graph.evalVec (Γ := ΓLN m n) (ss := ssLayerNorm m n)
            (layerNormGraph (m := m) (n := n) ε)) xV))
      xV := by
  rw [specLayerNormVec_eq]
  have hg := layerNormGraphFderivCorrectAt (m := m) (n := n) ε xV hε
  rcases Graph.hasFDerivAt_evalVec_and_jvp_at (Γ := ΓLN m n) (ss := ssLayerNorm m n)
    (g := layerNormGraph (m := m) (n := n) ε) xV hg with ⟨D, hD, _⟩
  rw [hD.fderiv]
  exact (outputCLM (m := m) (n := n)).hasFDerivAt.comp xV hD

/-- `Spec.layerNorm` is differentiable in `[X, gamma, beta]` for `0 < ε`. -/
theorem differentiableAt_specLayerNormVec {m n : Nat} (hm : 0 < m) (hn : 0 < n) {ε : ℝ}
    (hε : 0 < ε) (xV : CtxVec (ΓLN m n)) :
    DifferentiableAt ℝ (specLayerNormVec hm hn ε) xV :=
  (hasFDerivAt_specLayerNormVec hm hn hε xV).differentiableAt

/-- The adjoint of the output projection injects a cotangent into the output block. -/
theorem adjoint_outputCLM {m n : Nat} (δ : Vec (Spec.Shape.size (MatShape m n))) :
    (outputCLM (m := m) (n := n)).adjoint δ =
      CtxVec.single (Γ := ΓLN m n ++ ssLayerNorm m n) (s := MatShape m n) (idxY (m := m) (n := n))
        δ := by
  apply ext_inner_left ℝ
  intro w
  rw [ContinuousLinearMap.adjoint_inner_right, CtxVec.inner_get_single, outputCLM,
    CtxVec.getCLM_apply]

/-- Reverse-mode bridge: backprop of the LayerNorm graph seeded on the output block is the adjoint
derivative of `Spec.layerNorm` in `[X, gamma, beta]`. -/
theorem backpropVec_single_eq_adjoint_specLayerNorm {m n : Nat} (hm : 0 < m) (hn : 0 < n)
    {ε : ℝ} (hε : 0 < ε) (xV : CtxVec (ΓLN m n)) (δ : Vec (Spec.Shape.size (MatShape m n))) :
    Graph.backpropVec (Γ := ΓLN m n) (ss := ssLayerNorm m n) (layerNormGraph (m := m) (n := n) ε)
        xV
        (CtxVec.single (Γ := ΓLN m n ++ ssLayerNorm m n) (s := MatShape m n)
          (idxY (m := m) (n := n)) δ) =
      (fderiv ℝ (specLayerNormVec hm hn ε) xV).adjoint δ := by
  rw [backprop_eq_adjoint_fderiv_layerNorm_at (m := m) (n := n) ε xV _ hε,
    (hasFDerivAt_specLayerNormVec hm hn hε xV).fderiv, ContinuousLinearMap.adjoint_comp,
    ContinuousLinearMap.comp_apply, adjoint_outputCLM]

/-! ### Packed tensors -/

section Pack

variable {m n : Nat}

/-- The input block of a packed context. -/
theorem valX_packLN (X : Tensor ℝ [m, n]) (gamma beta : Tensor ℝ [n]) :
    valX (packLN X gamma beta) = tensorToVec X := by
  apply PiLp.ext
  intro j
  simp only [valX, CtxVec.get, packLN, idxX0]
  rw [CtxVec.getBlock_flattenCtx_zero]
  rfl

/-- The scale block of a packed context. -/
theorem get_idxGamma0_packLN (X : Tensor ℝ [m, n]) (gamma beta : Tensor ℝ [n]) :
    CtxVec.get (Γ := ΓLN m n) (s := VecShape n) idxGamma0 (packLN X gamma beta) =
      tensorToVec gamma := by
  apply PiLp.ext
  intro j
  simp only [CtxVec.get, packLN, idxGamma0]
  rw [CtxVec.getBlock_flattenCtx_succ, CtxVec.getBlock_flattenCtx_zero]
  rfl

/-- The shift block of a packed context. -/
theorem get_idxBeta0_packLN (X : Tensor ℝ [m, n]) (gamma beta : Tensor ℝ [n]) :
    CtxVec.get (Γ := ΓLN m n) (s := VecShape n) idxBeta0 (packLN X gamma beta) =
      tensorToVec beta := by
  apply PiLp.ext
  intro j
  simp only [CtxVec.get, packLN, idxBeta0]
  rw [CtxVec.getBlock_flattenCtx_succ, CtxVec.getBlock_flattenCtx_succ,
    CtxVec.getBlock_flattenCtx_zero]
  rfl

/-- Packing and unpacking round-trip on the input matrix. -/
theorem specX_packLN (X : Tensor ℝ [m, n]) (gamma beta : Tensor ℝ [n]) :
    specX (packLN X gamma beta) = X := by
  rw [specX, valX_packLN, vecToTensor_tensorToVec]

/-- Packing and unpacking round-trip on the scale vector. -/
theorem specGamma_packLN (X : Tensor ℝ [m, n]) (gamma beta : Tensor ℝ [n]) :
    specGamma (packLN X gamma beta) = gamma := by
  rw [specGamma, get_idxGamma0_packLN, vecToTensor_tensorToVec]

/-- Packing and unpacking round-trip on the shift vector. -/
theorem specBeta_packLN (X : Tensor ℝ [m, n]) (gamma beta : Tensor ℝ [n]) :
    specBeta (packLN X gamma beta) = beta := by
  rw [specBeta, get_idxBeta0_packLN, vecToTensor_tensorToVec]

/-- Forward bridge in tensor form: evaluating the LayerNorm graph on packed spec tensors and
reading the output block gives `Spec.layerNorm`. -/
theorem outputCLM_evalVec_layerNormGraph_packLN (hm : 0 < m) (hn : 0 < n) (ε : ℝ)
    (X : Tensor ℝ [m, n]) (gamma beta : Tensor ℝ [n]) :
    outputCLM (m := m) (n := n)
        (Graph.evalVec (Γ := ΓLN m n) (ss := ssLayerNorm m n)
          (layerNormGraph (m := m) (n := n) ε) (packLN X gamma beta)) =
      tensorToVec (Spec.layerNorm X gamma beta hm hn ε) := by
  rw [outputCLM_evalVec_layerNormGraph hm hn ε, specLayerNormVec, specX_packLN, specGamma_packLN,
    specBeta_packLN]

end Pack

end LayerNorm

end

end Autograd
end Proofs
