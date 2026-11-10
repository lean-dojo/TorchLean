/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Norm.LayerNorm.Eval
public import TorchLean.NN.Norm.RowNorm
public import TorchLean.NN.Norm.LayerNorm.Adjoint

/-!
# LayerNorm

Pointwise analytic correctness of the LayerNorm graph over `ℝ`, the proof-tape counterpart of the
spec `Tensor.layerNorm`. A `seqLen × embedDim` tensor is normalized across the last axis, the
row-wise normalizer is broadcast back over each token, and the affine parameters `gamma`/`beta` are
broadcast over the sequence dimension. The runtime API and the typed graph path both route through
that spec definition, and the theorems here prove the corresponding reverse-mode graph rule.

The proof graph uses the clamped square root `sqrt (max x 0)` and the reciprocal, so the main
theorem is pointwise (`RealGraph.FDerivCorrectAt`): away from the clamp kink and a zero
denominator, backprop is the adjoint of the Fréchet derivative. The `OfDomain` and `_of_domain`
variants take the two domain assumptions at the execution point, positive `var + ε` and nonzero
`std`. The main statements take only `0 < ε`, since the row variance is a mean of squares and
therefore nonnegative (`varEps_pos_of_eps_pos` and `std_ne_zero_of_eps_pos` in
`TorchLean.NN.Norm.LayerNorm.Eval`).

The last section connects the graph to the spec. Before squaring, `Tensor.layerNorm` divides each
centered row by `Tensor.normalizationScale`, so that no unscaled square is summed, and it clamps
the scaled `var + ε` at zero inside the square root. Over the reals the scale cancels, so for a
nonempty last axis `outputCLM_evalVec_graph` shows that the output block of the graph
evaluation is `Tensor.toVec (Tensor.layerNorm X gamma beta)` for the packed inputs.
`LayerNorm.hasFDerivAt_forwardVec` and `LayerNorm.backprop_eq_fderiv_forwardVec` then state
differentiability and the adjoint rule for the spec function itself. The adjointness of
`TorchLean.LayerNorm.jvp` and `TorchLean.LayerNorm.backward` is proved in
`TorchLean.NN.Norm.LayerNorm.Adjoint`. The companion `TorchLean.NN.Norm.LayerNorm.FDeriv`
module identifies that JVP with the Fréchet derivative and proves that the primitive backward rule
returns the same cotangents as this graph.

## PyTorch correspondence / citations
- Conceptually corresponds to `torch.nn.LayerNorm`: normalize along
  the last dimension, then apply affine parameters `(gamma,beta)`.
  https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
-/

@[expose] public section

open TorchLean.Shape (size size_cons)

namespace TorchLean

open scoped BigOperators

noncomputable section

namespace LayerNorm

open Tape.RealNode CtxVec

-- ---------------------------------------------------------------------------
-- Pointwise `Tape.RealGraph.FDerivCorrectAt` for LayerNorm
-- ---------------------------------------------------------------------------

/--
Pointwise proof that `graph` satisfies `Tape.RealGraph.FDerivCorrectAt` at `xV`, under the two
domain conditions there.

`hVarEpsPos` (every entry of `var + ε` is positive) makes the clamped square root differentiable
at the execution point, and `hStdNe0` (every entry of `std` is nonzero) does the same for the
reciprocal. `graphFDerivAt` discharges both from `0 < ε`.
-/
def graphFDerivAtOfDomain
    {m n : Nat} (ε : ℝ) (xV : CtxVec (ctx m n))
    (hVarEpsPos :
      ∀ i : Fin (Shape.size [m]),
        0 < CtxVec.get (Γ := ctx m n ++ ssVarEps m n) (s := [m])
          (idxVarEps (m := m) (n := n))
          (Tape.GraphData.evalVec (Γ := ctx m n) (ss := ssVarEps m n)
            (graphVarEps (m := m) (n := n) ε).toData () xV) i)
    (hStdNe0 :
      ∀ i : Fin (Shape.size [m]),
        CtxVec.get (Γ := ctx m n ++ ssStd m n) (s := [m]) (idxStd (m := m) (n := n))
          (Tape.GraphData.evalVec (Γ := ctx m n) (ss := ssStd m n)
            (graphStd (m := m) (n := n) ε).toData () xV) i ≠ 0) :
    Tape.RealGraph.FDerivCorrectAt (Γ := ctx m n) (ss := LayerNorm.ss m n)
      (graph (m := m) (n := n) ε).toData () xV := by
  classical
  -- Through the epsilon-shifted variance (`ssVarEps`)
  have hEmpty : Tape.RealGraph.FDerivCorrectAt (Γ := ctx m n) (ss := []) (.nil) () xV := PUnit.unit
  have hMean :
      Tape.RealGraph.FDerivCorrectAt (Γ := ctx m n) (ss := [[m]])
        (graphMean (m := m) (n := n)).toData () xV := by
    refine ⟨hEmpty, ?_⟩
    exact
      (rowMeanFDeriv (idx := idxX (m := m) (n := n) (ss := []))).at
        (Tape.GraphData.evalVec (Γ := ctx m n) (ss := []) (.nil) () xV)
  have hMeanBroadcast :
      Tape.RealGraph.FDerivCorrectAt (Γ := ctx m n) (ss := [[m], [m, n]])
        (graphMeanBroadcast (m := m) (n := n)).toData () xV := by
    refine ⟨hMean, ?_⟩
    exact
      (broadcastRowFDeriv (idx := idxMean (m := m) (n := n))).at
        (Tape.GraphData.evalVec (Γ := ctx m n) (ss := [[m]])
          (graphMean (m := m) (n := n)).toData () xV)
  have hCentered :
      Tape.RealGraph.FDerivCorrectAt (Γ := ctx m n) (ss := [[m], [m, n], [m, n]])
        (graphCentered (m := m) (n := n)).toData () xV := by
    refine ⟨hMeanBroadcast, ?_⟩
    exact
      (FDerivCorrect.ofArm₂ (s := [m, n]) (a := idxX (m := m) (n := n) (ss := [[m], [m, n]]))
        (b := idxMeanB (m := m) (n := n)) (cert := Op.sub_prod_backward_eq_adjoint)).at
        (Tape.GraphData.evalVec (Γ := ctx m n) (ss := [[m], [m, n]])
          (graphMeanBroadcast (m := m) (n := n)).toData () xV)
  have hCenteredSq :
      Tape.RealGraph.FDerivCorrectAt (Γ := ctx m n)
        (ss := [[m], [m, n], [m, n], [m, n]])
        (graphCenteredSq (m := m) (n := n)).toData () xV := by
    refine ⟨hCentered, ?_⟩
    exact
      (FDerivCorrect.ofArm₂ (s := [m, n]) (a := idxCentered (m := m) (n := n))
        (b := idxCentered (m := m) (n := n)) (cert := Op.mul_prod_backward_eq_adjoint)).at
        (Tape.GraphData.evalVec (Γ := ctx m n) (ss := [[m], [m, n], [m, n]])
          (graphCentered (m := m) (n := n)).toData () xV)
  have hVar :
      Tape.RealGraph.FDerivCorrectAt (Γ := ctx m n)
        (ss := [[m], [m, n], [m, n], [m, n], [m]])
        (graphVar (m := m) (n := n)).toData () xV := by
    refine ⟨hCenteredSq, ?_⟩
    exact
      (rowMeanFDeriv (idx := idxCenteredSq (m := m) (n := n))).at
        (Tape.GraphData.evalVec (Γ := ctx m n)
          (ss := [[m], [m, n], [m, n], [m, n]])
          (graphCenteredSq (m := m) (n := n)).toData () xV)
  have hVarEps :
      Tape.RealGraph.FDerivCorrectAt (Γ := ctx m n) (ss := ssVarEps m n)
        (graphVarEps (m := m) (n := n) ε).toData () xV := by
    refine ⟨hVar, ?_⟩
    have hderiv : FDerivCorrect (nodeVarEps (m := m) (n := n) ε).toNodeData () :=
      elemwiseFDeriv
        (Γ := ctx m n ++ [[m], [m, n], [m, n], [m, n], [m]])
        (s := [m]) (idx := idxVar (m := m) (n := n))
        (f := fun z => z + ε) (f' := fun _ => 1) (hf := fun z => (hasDerivAt_id z).add_const ε)
    exact
      FDerivCorrect.at hderiv
        (Tape.GraphData.evalVec (Γ := ctx m n)
          (ss := [[m], [m, n], [m, n], [m, n], [m]])
          (graphVar (m := m) (n := n)).toData () xV)

  -- std (`ssStd`)
  have hStd :
      Tape.RealGraph.FDerivCorrectAt (Γ := ctx m n) (ss := ssStd m n)
        (graphStd (m := m) (n := n) ε).toData () xV := by
    refine ⟨hVarEps, ?_⟩
    have hStdAt :
        FDerivCorrectAt (nodeStd (m := m) (n := n)).toNodeData ()
          (Tape.GraphData.evalVec (Γ := ctx m n) (ss := ssVarEps m n)
            (graphVarEps (m := m) (n := n) ε).toData () xV) :=
      FDerivCorrectAt.ofArm₁ (Γ := ctx m n ++ ssVarEps m n) (s := [m])
        (idx := idxVarEps (m := m) (n := n))
        (xV := Tape.GraphData.evalVec (Γ := ctx m n) (ss := ssVarEps m n)
          (graphVarEps (m := m) (n := n) ε).toData () xV)
        (cert := fun δ => Op.sqrt_clampNegative_backward_eq_adjoint _ δ
          ((CtxVec.forall_getTensorCLM_iff _ _ (0 < ·)).2 hVarEpsPos))
    simpa [graphStd, nodeStd] using hStdAt

  -- inv_std
  have hInvStd :
      Tape.RealGraph.FDerivCorrectAt (Γ := ctx m n) (ss := ssStd m n ++ [[m]])
        (graphInvStd (m := m) (n := n) ε).toData () xV := by
    refine ⟨hStd, ?_⟩
    have hInvAt :
        FDerivCorrectAt (nodeInvStd (m := m) (n := n)).toNodeData ()
          (Tape.GraphData.evalVec (Γ := ctx m n) (ss := ssStd m n)
            (graphStd (m := m) (n := n) ε).toData () xV) :=
      FDerivCorrectAt.ofArm₁ (Γ := ctx m n ++ ssStd m n) (s := [m])
        (idx := idxStd (m := m) (n := n))
        (xV := Tape.GraphData.evalVec (Γ := ctx m n) (ss := ssStd m n)
          (graphStd (m := m) (n := n) ε).toData () xV)
        (cert := fun δ => Op.reciprocal_backward_eq_adjoint _ δ
          ((CtxVec.forall_getTensorCLM_iff _ _ (· ≠ 0)).2 hStdNe0))
    simpa [graphInvStd, nodeInvStd] using hInvAt

  -- inv_std_b
  have hInvStdBroadcast :
      Tape.RealGraph.FDerivCorrectAt (Γ := ctx m n) (ss := ssStd m n ++ [[m], [m, n]])
        (graphInvStdBroadcast (m := m) (n := n) ε).toData () xV := by
    refine ⟨hInvStd, ?_⟩
    exact
      (broadcastRowFDeriv (idx := idxInvStd (m := m) (n := n))).at
        (Tape.GraphData.evalVec (Γ := ctx m n) (ss := ssStd m n ++ [[m]])
          (graphInvStd (m := m) (n := n) ε).toData () xV)

  -- normalized
  have hNormalized :
      Tape.RealGraph.FDerivCorrectAt (Γ := ctx m n)
        (ss := ssStd m n ++ [[m], [m, n], [m, n]])
        (graphNormalized (m := m) (n := n) ε).toData () xV := by
    refine ⟨hInvStdBroadcast, ?_⟩
    exact
      (FDerivCorrect.ofArm₂ (s := [m, n]) (a := idxCenteredForNorm (m := m) (n := n))
        (b := idxInvStdBroadcast (m := m) (n := n)) (cert := Op.mul_prod_backward_eq_adjoint)).at
        (Tape.GraphData.evalVec (Γ := ctx m n) (ss := ssStd m n ++ [[m], [m, n]])
          (graphInvStdBroadcast (m := m) (n := n) ε).toData () xV)

  -- gamma_b
  have hGammaBroadcast :
      Tape.RealGraph.FDerivCorrectAt (Γ := ctx m n)
        (ss := ssStd m n ++ [[m], [m, n], [m, n], [m, n]])
        (graphGammaBroadcast (m := m) (n := n) ε).toData () xV := by
    refine ⟨hNormalized, ?_⟩
    exact
      (broadcastColFDeriv
        (idx := idxGamma (m := m) (n := n)
          (ss := ssStd m n ++ [[m], [m, n], [m, n]]))).at
        (Tape.GraphData.evalVec (Γ := ctx m n)
          (ss := ssStd m n ++ [[m], [m, n], [m, n]])
          (graphNormalized (m := m) (n := n) ε).toData () xV)

  -- scaled
  have hScaled :
      Tape.RealGraph.FDerivCorrectAt (Γ := ctx m n)
        (ss := ssStd m n ++ [[m], [m, n], [m, n], [m, n], [m, n]])
        (graphScaled (m := m) (n := n) ε).toData () xV := by
    refine ⟨hGammaBroadcast, ?_⟩
    exact
      (FDerivCorrect.ofArm₂ (s := [m, n]) (a := idxNormForScale (m := m) (n := n))
        (b := idxGammaBroadcast (m := m) (n := n)) (cert := Op.mul_prod_backward_eq_adjoint)).at
        (Tape.GraphData.evalVec (Γ := ctx m n)
          (ss := ssStd m n ++ [[m], [m, n], [m, n], [m, n]])
          (graphGammaBroadcast (m := m) (n := n) ε).toData () xV)

  -- beta_b
  have hBetaBroadcast :
      Tape.RealGraph.FDerivCorrectAt (Γ := ctx m n)
        (ss := ssStd m n ++ [[m], [m, n], [m, n], [m, n], [m, n], [m, n]])
        (graphBetaBroadcast (m := m) (n := n) ε).toData () xV := by
    refine ⟨hScaled, ?_⟩
    exact
      (broadcastColFDeriv
        (idx := idxBeta (m := m) (n := n)
          (ss := ssStd m n ++ [[m], [m, n], [m, n], [m, n], [m, n]]))).at
        (Tape.GraphData.evalVec (Γ := ctx m n)
          (ss := ssStd m n ++ [[m], [m, n], [m, n], [m, n], [m, n]])
          (graphScaled (m := m) (n := n) ε).toData () xV)

  -- y
  refine ⟨hBetaBroadcast, ?_⟩
  exact
    (FDerivCorrect.ofArm₂ (s := [m, n]) (a := idxScaledForOutput (m := m) (n := n))
      (b := idxBetaBroadcast (m := m) (n := n)) (cert := Op.add_prod_backward_eq_adjoint)).at
      (Tape.GraphData.evalVec (Γ := ctx m n)
        (ss := ssStd m n ++ [[m], [m, n], [m, n], [m, n], [m, n], [m, n]])
        (graphBetaBroadcast (m := m) (n := n) ε).toData () xV)

/-- Pointwise proof that `graph` satisfies `Tape.RealGraph.FDerivCorrectAt` whenever `0 < ε`. -/
def graphFDerivAt
    {m n : Nat} (ε : ℝ) (xV : CtxVec (ctx m n)) (hε : 0 < ε) :
    Tape.RealGraph.FDerivCorrectAt (Γ := ctx m n) (ss := LayerNorm.ss m n)
      (graph (m := m) (n := n) ε).toData () xV :=
  graphFDerivAtOfDomain (m := m) (n := n) ε xV
    (varEps_pos_of_eps_pos hε xV) (std_ne_zero_of_eps_pos hε xV)

/--
Pointwise end-to-end result under the two domain conditions at `xV`: backprop of `graph`
equals `(fderiv eval)†`.

`hVarEpsPos` and `hStdNe0` are the conditions under which the clamped square root and the
reciprocal are differentiable at the execution point.
-/
theorem backprop_eq_fderiv_of_domain
    {m n : Nat} (ε : ℝ)
    (xV : CtxVec (ctx m n))
    (seedV : CtxVec (ctx m n ++ LayerNorm.ss m n))
    (hVarEpsPos :
      ∀ i : Fin (Shape.size [m]),
        0 < CtxVec.get (Γ := ctx m n ++ ssVarEps m n) (s := [m])
          (idxVarEps (m := m) (n := n))
          (Tape.GraphData.evalVec (Γ := ctx m n) (ss := ssVarEps m n)
            (graphVarEps (m := m) (n := n) ε).toData () xV) i)
    (hStdNe0 :
      ∀ i : Fin (Shape.size [m]),
        CtxVec.get (Γ := ctx m n ++ ssStd m n) (s := [m]) (idxStd (m := m) (n := n))
          (Tape.GraphData.evalVec (Γ := ctx m n) (ss := ssStd m n)
            (graphStd (m := m) (n := n) ε).toData () xV) i ≠ 0) :
    Tape.GraphData.backpropVec (Γ := ctx m n) (ss := LayerNorm.ss m n)
      (graph (m := m) (n := n) ε).toData () xV seedV
      =
    (fderiv ℝ
        (Tape.GraphData.evalVec (Γ := ctx m n) (ss := LayerNorm.ss m n)
          (graph (m := m) (n := n) ε).toData ())
        xV).adjoint seedV := by
  classical
  have hg :
      Tape.RealGraph.FDerivCorrectAt (Γ := ctx m n) (ss := LayerNorm.ss m n)
        (graph (m := m) (n := n) ε).toData () xV :=
    graphFDerivAtOfDomain (m := m) (n := n) ε xV hVarEpsPos hStdNe0
  exact
    Tape.Graph.backpropVec_eq_fderiv (Γ := ctx m n) (ss := LayerNorm.ss m n)
      (g := graph (m := m) (n := n) ε) () xV seedV hg

/-- Pointwise end-to-end result: backprop equals `(fderiv eval)†` for `graph` whenever
`0 < ε`. -/
theorem backprop_eq_fderiv
    {m n : Nat} (ε : ℝ)
    (xV : CtxVec (ctx m n))
    (seedV : CtxVec (ctx m n ++ LayerNorm.ss m n)) (hε : 0 < ε) :
    Tape.GraphData.backpropVec (Γ := ctx m n) (ss := LayerNorm.ss m n)
      (graph (m := m) (n := n) ε).toData () xV seedV
      =
    (fderiv ℝ
        (Tape.GraphData.evalVec (Γ := ctx m n) (ss := LayerNorm.ss m n)
          (graph (m := m) (n := n) ε).toData ())
        xV).adjoint seedV :=
  Tape.Graph.backpropVec_eq_fderiv (Γ := ctx m n) (ss := LayerNorm.ss m n)
    (g := graph (m := m) (n := n) ε) () xV seedV
    (graphFDerivAt (m := m) (n := n) ε xV hε)

-- ---------------------------------------------------------------------------
-- Generic whole-node adapter
-- ---------------------------------------------------------------------------

/--
LayerNorm inputs inside an arbitrary tape context.

Model-level proofs use this interface when LayerNorm is one node of a larger graph. In a post-norm
Transformer block, for example, `x` is the residual stream produced by an earlier SSA node, and
`gamma` and `beta` are parameters carried in the surrounding context.
-/
structure Inputs (Γ : List Shape) (m n : Nat) where
  /-- Sequence/residual matrix normalized across its last axis. -/
  x : Idx Γ [m, n]
  /-- Affine scale vector. -/
  gamma : Idx Γ [n]
  /-- Affine shift vector. -/
  beta : Idx Γ [n]

/--
Continuous linear map that reads the three LayerNorm inputs out of an arbitrary context and packs
them into the canonical context `[X, gamma, beta]`.
-/
def packInputsCLM {Γ : List Shape} {m n : Nat} (inputs : Inputs Γ m n) :
    CtxVec Γ →L[ℝ] CtxVec (ctx m n) := by
  let xCLM : CtxVec Γ →L[ℝ] Vec (Shape.size [m, n]) :=
    CtxVec.getCLM (Γ := Γ) (s := [m, n]) inputs.x
  let gammaCLM : CtxVec Γ →L[ℝ] Vec (Shape.size [n]) :=
    CtxVec.getCLM (Γ := Γ) (s := [n]) inputs.gamma
  let betaCLM : CtxVec Γ →L[ℝ] Vec (Shape.size [n]) :=
    CtxVec.getCLM (Γ := Γ) (s := [n]) inputs.beta
  let gbCLM :=
    (EuclideanSpace.finAddEquivProd (𝕜 := ℝ)).symm.toContinuousLinearMap.comp
      (gammaCLM.prod betaCLM)
  let allCLM :=
    (EuclideanSpace.finAddEquivProd (𝕜 := ℝ)).symm.toContinuousLinearMap.comp (xCLM.prod gbCLM)
  let h :
      Shape.size [m, n] + (Shape.size [n] + Shape.size [n])
        =
      ctxSize (ctx m n) := by
    simp [ctxSize]
  exact (LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ
    (finCongr h)).toContinuousLinearEquiv.toContinuousLinearMap.comp allCLM

/-- Project the final LayerNorm output from the full canonical graph context. -/
def outputCLM {m n : Nat} :
    CtxVec (ctx m n ++ ss m n) →L[ℝ] Vec (Shape.size [m, n]) :=
  CtxVec.getCLM (Γ := ctx m n ++ ss m n) (s := [m, n]) (idxY (m := m) (n := n))

/--
LayerNorm as one reusable pointwise node over arbitrary context indices.

The forward map packs the inputs, evaluates the LayerNorm graph and reads its output block. Its
JVP is the Fréchet derivative of that map and its VJP the adjoint, both noncomputable. Transformer
proofs use the node as one block without expanding the graph inside, and
`TorchLean.NN.Norm.LayerNorm.FDeriv` identifies the derivative of the LayerNorm map with the
primitive JVP and VJP formulas.
-/
def node {Γ : List Shape} {m n : Nat} (inputs : Inputs Γ m n) (ε : ℝ) :
    Tape.LawfulNode (α := ℝ) Unit Γ [m, n] :=
  let f : CtxVec Γ → Vec (Shape.size [m, n]) :=
    fun xV =>
      outputCLM (m := m) (n := n)
        (Tape.GraphData.evalVec (Γ := ctx m n) (ss := LayerNorm.ss m n)
          (graph (m := m) (n := n) ε).toData () (packInputsCLM inputs xV))
  Tape.LawfulNode.ofFn (Γ := Γ) (τ := [m, n])
    (f := f)
    (jvp := fun xV dxV => (fderiv ℝ f xV) dxV)
    (vjp := fun xV δV => (fderiv ℝ f xV).adjoint δV)
    (correct_inner := by
      intro xV dxV δV
      simpa using
        (ContinuousLinearMap.adjoint_inner_right (A := fderiv ℝ f xV) (x := dxV) (y := δV)).symm)

/--
Pointwise derivative certificate for `node`.

Only `0 < ε` is needed: the row variance computed inside the packed LayerNorm graph is a mean of
squares, so `var + ε` is positive and the clamped standard deviation is nonzero at every point.
-/
def nodeFDerivAt {Γ : List Shape} {m n : Nat}
    (inputs : Inputs Γ m n) (ε : ℝ) (xV : CtxVec Γ) (hε : 0 < ε) :
    FDerivCorrectAt (node (Γ := Γ) (m := m) (n := n) inputs ε).toNodeData () xV := by
  classical
  -- We write the packed map out in full: binding it with `let` makes the elaboration of this
  -- definition exceed the recursion limit.
  have hgAt := graphFDerivAt (m := m) (n := n) ε (packInputsCLM inputs xV) hε
  have hOut :
      HasFDerivAt (fun z => outputCLM (m := m) (n := n)
          (Tape.GraphData.evalVec (graph (m := m) (n := n) ε).toData ()
            (packInputsCLM inputs z)))
        ((outputCLM (m := m) (n := n)).comp
          ((fderiv ℝ (Tape.GraphData.evalVec (graph (m := m) (n := n) ε).toData ())
            (packInputsCLM inputs xV)).comp (packInputsCLM inputs))) xV :=
    (outputCLM (m := m) (n := n)).hasFDerivAt.comp xV
      (Tape.GraphData.hasFDerivAt_evalVec_comp _ () xV hgAt (packInputsCLM inputs).hasFDerivAt)
  refine
    { deriv := fderiv ℝ (fun z => outputCLM (m := m) (n := n)
          (Tape.GraphData.evalVec (graph (m := m) (n := n) ε).toData ()
            (packInputsCLM inputs z))) xV
      hasFDerivAt := ?_
      jvp_eq := ?_ }
  · rw [hOut.fderiv]
    -- Here and below, simp leaves out `size_cons`. With it, simp rewrites the sizes inside the
    -- `Vec` types and runs out of heartbeats.
    simpa [node, -Shape.size_cons] using hOut
  · intro dxV
    simp [node, -Shape.size_cons]

-- ---------------------------------------------------------------------------
-- Bridge to `Tensor.layerNorm`
-- ---------------------------------------------------------------------------

/-! ## The graph computes `Tensor.layerNorm`

The packed context `[X, gamma, beta]` determines three spec tensors. For `0 < n`, evaluating the
LayerNorm graph on it and reading the output block gives
`Tensor.toVec (Tensor.layerNorm X gamma beta)`. For `0 < ε` the spec function is therefore
differentiable, with the graph's backprop as the adjoint of its derivative. -/

/-- The input matrix of a packed LayerNorm context, as a spec tensor. -/
def inputX {m n : Nat} (xV : CtxVec (ctx m n)) : Tensor ℝ [m, n] :=
  Tensor.ofVec (s := [m, n]) (valX xV)

/-- The scale vector of a packed LayerNorm context, as a spec tensor. -/
def inputGamma {m n : Nat} (xV : CtxVec (ctx m n)) : Tensor ℝ [n] :=
  Tensor.ofVec (s := [n])
    (CtxVec.get (Γ := ctx m n) (s := [n]) (idxGamma (m := m) (n := n) (ss := [])) xV)

/-- The shift vector of a packed LayerNorm context, as a spec tensor. -/
def inputBeta {m n : Nat} (xV : CtxVec (ctx m n)) : Tensor ℝ [n] :=
  Tensor.ofVec (s := [n])
    (CtxVec.get (Γ := ctx m n) (s := [n]) (idxBeta (m := m) (n := n) (ss := [])) xV)

/-- Layer normalization over the last axis as a map on the packed context vector
`[X, gamma, beta]`. -/
def forwardVec {m n : Nat} (ε : ℝ) (xV : CtxVec (ctx m n)) :
    Vec (Shape.size [m, n]) :=
  Tensor.toVec (Tensor.layerNorm (inputX xV) [⟨1, Nat.one_lt_two⟩] (inputGamma xV) (inputBeta xV) ε)

/-- Pack three spec tensors into a LayerNorm context vector. -/
def pack {m n : Nat} (X : Tensor ℝ [m, n]) (gamma beta : Tensor ℝ [n]) : CtxVec (ctx m n) :=
  TensorPack.toVec (Γ := ctx m n) (.cons X (.cons gamma (.cons beta .nil)))

section EntryLemmas

open Matmul

variable {m n : Nat}

/-- Entries of the input block. -/
theorem valX_idxMN (xV : CtxVec (ctx m n)) (i : Fin m) (j : Fin n) :
    valX xV (idxMN (m := m) (n := n) i j) = Tensor.get2 (inputX xV) i j := by
  rw [inputX, ← Matmul.toVec_idxMN, Tensor.toVec_ofVec]

/-- Entries of the scale block. -/
theorem valGamma_apply (xV : CtxVec (ctx m n)) (j : Fin n) :
    valGamma xV j = Tensor.getScalar (inputGamma xV) j := by
  simp only [valGamma, getVec, LinearIsometryEquiv.piLpCongrLeft_apply, Equiv.piCongrLeft'_apply,
    finCongr_symm_apply]
  rw [show CtxVec.get (Γ := ctx m n) (s := [n]) (idxGamma (m := m) (n := n) (ss := []))
      xV = Tensor.toVec (inputGamma xV)
    from (Tensor.toVec_ofVec _).symm, Tensor.toVec_vec]
  rfl

/-- Entries of the shift block. -/
theorem valBeta_apply (xV : CtxVec (ctx m n)) (j : Fin n) :
    valBeta xV j = Tensor.getScalar (inputBeta xV) j := by
  simp only [valBeta, getVec, LinearIsometryEquiv.piLpCongrLeft_apply, Equiv.piCongrLeft'_apply,
    finCongr_symm_apply]
  rw [show CtxVec.get (Γ := ctx m n) (s := [n]) (idxBeta (m := m) (n := n) (ss := []))
      xV = Tensor.toVec (inputBeta xV)
    from (Tensor.toVec_ofVec _).symm, Tensor.toVec_vec]
  rfl

/-- The mean block holds the spec row means. -/
theorem valMean_apply (xV : CtxVec (ctx m n)) (i : Fin m) :
    valMean xV (Fin.cast (Shape.size_vector m).symm i) = 𝔼 j, Tensor.get2 (inputX xV) i j := by
  show MatrixLinear.rowMeanCLM (m := m) (n := n) (valX xV) i = _
  rw [RowNorm.rowMeanCLM_eq]
  exact Finset.expect_congr rfl fun j _ => valX_idxMN xV i j

/-- The broadcast mean block. -/
theorem valMeanB_idxMN (xV : CtxVec (ctx m n)) (i : Fin m) (j : Fin n) :
    valMeanB xV (idxMN (m := m) (n := n) i j) = 𝔼 k, Tensor.get2 (inputX xV) i k := by
  rw [valMeanB, RowNorm.broadcastRowCLM_idxMN, LinearIsometryEquiv.piLpCongrLeft_apply,
    Equiv.piCongrLeft'_apply, finCongr_symm_apply, valMean_apply]

/-- The centered block. -/
theorem valCentered_idxMN (xV : CtxVec (ctx m n)) (i : Fin m) (j : Fin n) :
    valCentered xV (idxMN (m := m) (n := n) i j) =
      Tensor.get2 (inputX xV) i j - 𝔼 k, Tensor.get2 (inputX xV) i k := by
  simp only [valCentered, PiLp.sub_apply, valX_idxMN, valMeanB_idxMN]

/-- The variance block holds the spec row variances. -/
theorem valVar_apply (xV : CtxVec (ctx m n)) (i : Fin m) :
    valVar xV (Fin.cast (Shape.size_vector m).symm i) =
      TorchLean.var (Tensor.get2 (inputX xV) i) := by
  show MatrixLinear.rowMeanCLM (m := m) (n := n) (valCenteredSq xV) i = _
  rw [RowNorm.rowMeanCLM_eq, TorchLean.var]
  refine Finset.expect_congr rfl fun j _ => ?_
  simp only [valCenteredSq, PiLp.toLp_apply, valCentered_idxMN]

/-- The broadcast inverse standard deviation block. -/
theorem valInvStdB_idxMN (xV : CtxVec (ctx m n)) (ε : ℝ) (i : Fin m) (j : Fin n) :
    valInvStdB xV ε (idxMN (m := m) (n := n) i j) =
      (Real.sqrt (max (TorchLean.var (Tensor.get2 (inputX xV) i) + ε) 0))⁻¹ := by
  rw [valInvStdB, RowNorm.broadcastRowCLM_idxMN, LinearIsometryEquiv.piLpCongrLeft_apply,
    Equiv.piCongrLeft'_apply, finCongr_symm_apply]
  simp only [valInvStd, valStd, valVarEps, PiLp.toLp_apply, valVar_apply]

/-- The output block, entrywise. -/
theorem valY_idxMN (xV : CtxVec (ctx m n)) (ε : ℝ) (i : Fin m) (j : Fin n) :
    valY xV ε (idxMN (m := m) (n := n) i j) =
      (Tensor.get2 (inputX xV) i j - 𝔼 k, Tensor.get2 (inputX xV) i k) *
          (Real.sqrt (max (TorchLean.var (Tensor.get2 (inputX xV) i) + ε) 0))⁻¹ *
            Tensor.getScalar (inputGamma xV) j +
        Tensor.getScalar (inputBeta xV) j := by
  simp only [valY, PiLp.add_apply, valScaled, valNorm, PiLp.toLp_apply, valGammaB, valBetaB,
    RowNorm.broadcastColCLM_idxMN, valCentered_idxMN, valInvStdB_idxMN, valGamma_apply,
    valBeta_apply]

end EntryLemmas

/-- For `0 < n`, the spec LayerNorm of the packed inputs is `valY`, the closed form of the graph
output. -/
theorem forwardVec_eq_valY {m n : Nat} (hn : 0 < n) (ε : ℝ)
    (xV : CtxVec (ctx m n)) :
    forwardVec ε xV = valY xV ε := by
  apply Normalization.vec_ext_idxMN
  intro i j
  rw [forwardVec, Matmul.toVec_idxMN]
  erw [Normalization.get2_layerNorm hn]
  rw [valY_idxMN, div_eq_mul_inv]
  rfl

/-- Forward bridge: for `0 < n`, the output block of the LayerNorm graph is `Tensor.layerNorm` of
the packed inputs. -/
theorem outputCLM_evalVec_graph {m n : Nat} (hn : 0 < n) (ε : ℝ)
    (xV : CtxVec (ctx m n)) :
    outputCLM (m := m) (n := n)
        (Tape.GraphData.evalVec (Γ := ctx m n) (ss := LayerNorm.ss m n)
          (graph (m := m) (n := n) ε).toData () xV) =
      forwardVec ε xV := by
  rw [forwardVec_eq_valY hn, outputCLM, CtxVec.getCLM_apply, get_idxY]

/-- For `0 < n`, the spec LayerNorm map is the output projection of the graph evaluation. -/
theorem forwardVec_eq {m n : Nat} (hn : 0 < n) (ε : ℝ) :
    forwardVec (m := m) (n := n) ε =
      fun xV =>
        outputCLM (m := m) (n := n)
          (Tape.GraphData.evalVec (Γ := ctx m n) (ss := LayerNorm.ss m n)
            (graph (m := m) (n := n) ε).toData () xV) := by
  funext xV
  exact (outputCLM_evalVec_graph hn ε xV).symm

/-- For `0 < n` and `0 < ε`, `Tensor.layerNorm` is differentiable in `[X, gamma, beta]`, with
derivative the output projection of the graph derivative. -/
theorem hasFDerivAt_forwardVec {m n : Nat} (hn : 0 < n) {ε : ℝ}
    (hε : 0 < ε) (xV : CtxVec (ctx m n)) :
    HasFDerivAt (forwardVec ε)
      ((outputCLM (m := m) (n := n)).comp
        (fderiv ℝ
          (Tape.GraphData.evalVec (Γ := ctx m n) (ss := LayerNorm.ss m n)
            (graph (m := m) (n := n) ε).toData ()) xV))
      xV := by
  rw [forwardVec_eq hn]
  have hg := graphFDerivAt (m := m) (n := n) ε xV hε
  rcases Tape.GraphData.hasFDerivAt_evalVec_and_jvp (Γ := ctx m n) (ss := LayerNorm.ss m n)
    (g := (graph (m := m) (n := n) ε).toData) () xV hg with ⟨D, hD, _⟩
  rw [hD.fderiv]
  exact (outputCLM (m := m) (n := n)).hasFDerivAt.comp xV hD

/-- For `0 < n` and `0 < ε`, `Tensor.layerNorm` is differentiable in `[X, gamma, beta]`. -/
theorem differentiableAt_forwardVec {m n : Nat} (hn : 0 < n) {ε : ℝ}
    (hε : 0 < ε) (xV : CtxVec (ctx m n)) :
    DifferentiableAt ℝ (forwardVec ε) xV :=
  (hasFDerivAt_forwardVec hn hε xV).differentiableAt

/-- The adjoint of the output projection injects a cotangent into the output block. -/
theorem adjoint_outputCLM {m n : Nat} (δ : Vec (Shape.size [m, n])) :
    (outputCLM (m := m) (n := n)).adjoint δ =
      CtxVec.single (Γ := ctx m n ++ ss m n) (s := [m, n]) (idxY (m := m) (n := n))
        δ :=
  CtxVec.adjoint_getCLM _ δ

/-- Reverse-mode bridge: for `0 < n` and `0 < ε`, backprop of the LayerNorm graph seeded with `δ`
on the output block is the adjoint of the derivative of `Tensor.layerNorm` in `[X, gamma, beta]`,
applied to `δ`. -/
theorem backprop_eq_fderiv_forwardVec {m n : Nat} (hn : 0 < n)
    {ε : ℝ} (hε : 0 < ε) (xV : CtxVec (ctx m n)) (δ : Vec (Shape.size [m, n])) :
    Tape.GraphData.backpropVec (Γ := ctx m n) (ss := LayerNorm.ss m n)
        (graph (m := m) (n := n) ε).toData () xV
        (CtxVec.single (Γ := ctx m n ++ LayerNorm.ss m n) (s := [m, n])
          (idxY (m := m) (n := n)) δ) =
      (fderiv ℝ (forwardVec ε) xV).adjoint δ := by
  rw [backprop_eq_fderiv (m := m) (n := n) ε xV _ hε,
    (hasFDerivAt_forwardVec hn hε xV).fderiv, ContinuousLinearMap.adjoint_comp,
    ContinuousLinearMap.comp_apply, adjoint_outputCLM]

/-! ### Packed tensors -/

section Pack

variable {m n : Nat}

/-- The input block of a packed context. -/
theorem valX_pack (X : Tensor ℝ [m, n]) (gamma beta : Tensor ℝ [n]) :
    valX (pack X gamma beta) = Tensor.toVec X := by
  apply PiLp.ext
  intro j
  simp only [valX, CtxVec.get, pack, idxX]
  rw [CtxVec.getBlock_toVec_zero]
  rfl

/-- The scale block of a packed context. -/
theorem get_idxGamma_pack (X : Tensor ℝ [m, n]) (gamma beta : Tensor ℝ [n]) :
    CtxVec.get (Γ := ctx m n) (s := [n]) (idxGamma (m := m) (n := n) (ss := []))
      (pack X gamma beta) =
      Tensor.toVec gamma := by
  apply PiLp.ext
  intro j
  simp only [CtxVec.get, pack, idxGamma]
  rw [CtxVec.getBlock_toVec_succ, CtxVec.getBlock_toVec_zero]
  rfl

/-- The shift block of a packed context. -/
theorem get_idxBeta_pack (X : Tensor ℝ [m, n]) (gamma beta : Tensor ℝ [n]) :
    CtxVec.get (Γ := ctx m n) (s := [n]) (idxBeta (m := m) (n := n) (ss := []))
      (pack X gamma beta) =
      Tensor.toVec beta := by
  apply PiLp.ext
  intro j
  simp only [CtxVec.get, pack, idxBeta]
  rw [CtxVec.getBlock_toVec_succ, CtxVec.getBlock_toVec_succ,
    CtxVec.getBlock_toVec_zero]
  rfl

/-- Packing and unpacking round-trip on the input matrix. -/
theorem inputX_pack (X : Tensor ℝ [m, n]) (gamma beta : Tensor ℝ [n]) :
    inputX (pack X gamma beta) = X := by
  rw [inputX, valX_pack, Tensor.ofVec_toVec]

/-- Packing and unpacking round-trip on the scale vector. -/
theorem inputGamma_pack (X : Tensor ℝ [m, n]) (gamma beta : Tensor ℝ [n]) :
    inputGamma (pack X gamma beta) = gamma := by
  rw [inputGamma, get_idxGamma_pack, Tensor.ofVec_toVec]

/-- Packing and unpacking round-trip on the shift vector. -/
theorem inputBeta_pack (X : Tensor ℝ [m, n]) (gamma beta : Tensor ℝ [n]) :
    inputBeta (pack X gamma beta) = beta := by
  rw [inputBeta, get_idxBeta_pack, Tensor.ofVec_toVec]

/-- Forward bridge in tensor form: for `0 < n`, evaluating the LayerNorm graph on packed spec
tensors and reading the output block gives `Tensor.layerNorm`. -/
theorem outputCLM_evalVec_graph_pack (hn : 0 < n) (ε : ℝ)
    (X : Tensor ℝ [m, n]) (gamma beta : Tensor ℝ [n]) :
    outputCLM (m := m) (n := n)
        (Tape.GraphData.evalVec (Γ := ctx m n) (ss := LayerNorm.ss m n)
          (graph (m := m) (n := n) ε).toData () (pack X gamma beta)) =
      Tensor.toVec (Tensor.layerNorm X [⟨1, Nat.one_lt_two⟩] gamma beta ε) := by
  rw [outputCLM_evalVec_graph hn ε, forwardVec, inputX_pack]
  erw [inputGamma_pack, inputBeta_pack]

end Pack

end LayerNorm

end

end TorchLean
