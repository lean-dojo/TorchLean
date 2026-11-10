/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Norm.LayerNorm.Graph
public import TorchLean.NN.Tape.RealGraph.EvalVec
public import TorchLean.NN.Norm.RowNorm

/-!
# Evaluating the LayerNorm graph

Closed forms for every saved intermediate of `graph`. Each `val*` definition is the value
the corresponding node computes from the packed input `[X, gamma, beta]`, and each `get_*` lemma
identifies the block of `Tape.GraphData.evalVec` at that stage with its closed form.

Two consequences drive the calculus: the per-row variance block is a mean of squares, so it is
nonnegative, and therefore `var + ε` is positive and the clamped square root is nonzero as soon as
`0 < ε`. Those two facts are the domain hypotheses of the LayerNorm graph theorem, proved here as
`varEps_pos_of_eps_pos` and `std_ne_zero_of_eps_pos`.
-/

@[expose] public section

open TorchLean.Shape (size)

namespace TorchLean

open scoped BigOperators

noncomputable section

namespace LayerNorm

open Tape.RealNode CtxVec MatrixLinear

variable {m n : Nat}

/-! ## Closed forms of the intermediates -/

/-- The input matrix block. -/
def valX (xV : CtxVec (ctx m n)) : Vec (Shape.size [m, n]) :=
  CtxVec.get (Γ := ctx m n) (s := [m, n]) (idxX (m := m) (n := n) (ss := [])) xV

/-- The scale vector block. -/
def valGamma (xV : CtxVec (ctx m n)) : Vec n :=
  getVec (Γ := ctx m n) (n := n) (idxGamma (m := m) (n := n) (ss := [])) xV

/-- The shift vector block. -/
def valBeta (xV : CtxVec (ctx m n)) : Vec n :=
  getVec (Γ := ctx m n) (n := n) (idxBeta (m := m) (n := n) (ss := [])) xV

/-- Row means of `X`. -/
def valMean (xV : CtxVec (ctx m n)) : Vec (Shape.size [m]) :=
  LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ (finCongr (Shape.size_vector m).symm)
    (rowMeanCLM (m := m) (n := n) (valX xV))

/-- Row means broadcast back to the matrix shape. -/
def valMeanB (xV : CtxVec (ctx m n)) : Vec (Shape.size [m, n]) :=
  broadcastRowCLM (m := m) (n := n)
    (LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ (finCongr (Shape.size_vector m)) (valMean xV))

/-- Centered input `X - mean_b`. -/
def valCentered (xV : CtxVec (ctx m n)) : Vec (Shape.size [m, n]) :=
  valX xV - valMeanB xV

/-- Squared centered input. -/
def valCenteredSq (xV : CtxVec (ctx m n)) : Vec (Shape.size [m, n]) :=
  WithLp.toLp 2 fun i => valCentered xV i * valCentered xV i

/-- Row variances (means of the squared centered entries). -/
def valVar (xV : CtxVec (ctx m n)) : Vec (Shape.size [m]) :=
  LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ (finCongr (Shape.size_vector m).symm)
    (rowMeanCLM (m := m) (n := n) (valCenteredSq xV))

/-- Row variances shifted by `ε`. -/
def valVarEps (xV : CtxVec (ctx m n)) (ε : ℝ) : Vec (Shape.size [m]) :=
  WithLp.toLp 2 fun i => valVar xV i + ε

/-- Clamped standard deviation `sqrt (max (var + ε) 0)`. -/
def valStd (xV : CtxVec (ctx m n)) (ε : ℝ) : Vec (Shape.size [m]) :=
  WithLp.toLp 2 fun i => Real.sqrt (max (valVarEps xV ε i) 0)

/-- Inverse standard deviation. -/
def valInvStd (xV : CtxVec (ctx m n)) (ε : ℝ) : Vec (Shape.size [m]) :=
  WithLp.toLp 2 fun i => (valStd xV ε i)⁻¹

/-- Inverse standard deviation broadcast back to the matrix shape. -/
def valInvStdB (xV : CtxVec (ctx m n)) (ε : ℝ) : Vec (Shape.size [m, n]) :=
  broadcastRowCLM (m := m) (n := n)
    (LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ (finCongr (Shape.size_vector m)) (valInvStd xV ε))

/-- Normalized input `centered ⊙ inv_std_b`. -/
def valNorm (xV : CtxVec (ctx m n)) (ε : ℝ) : Vec (Shape.size [m, n]) :=
  WithLp.toLp 2 fun i => valCentered xV i * valInvStdB xV ε i

/-- Scale vector broadcast over rows. -/
def valGammaB (xV : CtxVec (ctx m n)) : Vec (Shape.size [m, n]) :=
  broadcastColCLM (m := m) (n := n) (valGamma xV)

/-- Scaled normalized input. -/
def valScaled (xV : CtxVec (ctx m n)) (ε : ℝ) : Vec (Shape.size [m, n]) :=
  WithLp.toLp 2 fun i => valNorm xV ε i * valGammaB xV i

/-- Shift vector broadcast over rows. -/
def valBetaB (xV : CtxVec (ctx m n)) : Vec (Shape.size [m, n]) :=
  broadcastColCLM (m := m) (n := n) (valBeta xV)

/-- LayerNorm output `scaled + beta_b`. -/
def valY (xV : CtxVec (ctx m n)) (ε : ℝ) : Vec (Shape.size [m, n]) :=
  valScaled xV ε + valBetaB xV

/-! ## Stage evaluations -/

/-- Stage `graphMean`: the mean block. -/
theorem get_graphMean (xV : CtxVec (ctx m n)) :
    CtxVec.get (Γ := ctx m n ++ [[m]]) (s := [m]) (idxMean (m := m) (n := n))
      (Tape.GraphData.evalVec (Γ := ctx m n) (ss := [[m]])
        (graphMean (m := m) (n := n)).toData () xV) =
      valMean xV := by
  refine (Tape.GraphData.get_evalVec_snoc_last (Γ := ctx m n) (ss := []) (τ := [m])
    .nil (nodeMean (m := m) (n := n)).toNodeData () xV (idxMean (m := m) (n := n)) rfl).trans ?_
  simp only [nodeMean, rowMean, Tape.LawfulNode.forwardVec_ofFn]
  rw [Tape.GraphData.get_evalVec_input (Γ := ctx m n) (ss := []) .nil () xV
    (idxX (m := m) (n := n) (ss := [])) (idxX (m := m) (n := n) (ss := [])) rfl]
  rfl

/-- Stage `graphMeanBroadcast`: the broadcast mean block. -/
theorem get_graphMeanBroadcast (xV : CtxVec (ctx m n)) :
    CtxVec.get (Γ := ctx m n ++ [[m], [m, n]]) (s := [m, n])
      (idxMeanB (m := m) (n := n))
      (Tape.GraphData.evalVec (Γ := ctx m n) (ss := [[m], [m, n]])
        (graphMeanBroadcast (m := m) (n := n)).toData () xV) =
      valMeanB xV := by
  refine (Tape.GraphData.get_evalVec_snoc_last (Γ := ctx m n) (ss := [[m]])
    (τ := [m, n]) (graphMean (m := m) (n := n)).toData
      (nodeMeanB (m := m) (n := n)).toNodeData () xV
    (idxMeanB (m := m) (n := n)) rfl).trans ?_
  simp only [nodeMeanB, broadcastRow, Tape.LawfulNode.forwardVec_ofFn, getVec]
  rw [get_graphMean]
  rfl

/-- Stage `graphCentered`: the centered block. -/
theorem get_graphCentered (xV : CtxVec (ctx m n)) :
    CtxVec.get (Γ := ctx m n ++ [[m], [m, n], [m, n]]) (s := [m, n])
      (idxCentered (m := m) (n := n))
      (Tape.GraphData.evalVec (Γ := ctx m n) (ss := [[m], [m, n], [m, n]])
        (graphCentered (m := m) (n := n)).toData () xV) =
      valCentered xV := by
  refine (Tape.GraphData.get_evalVec_snoc_last (Γ := ctx m n) (ss := [[m], [m, n]])
    (τ := [m, n]) (graphMeanBroadcast (m := m) (n := n)).toData
      (nodeCentered (m := m) (n := n)).toNodeData () xV
    (idxCentered (m := m) (n := n)) rfl).trans ?_
  rw [Tape.NodeData.forwardVec_of_zipWith
    (idxX (m := m) (n := n) (ss := [[m], [m, n]])) (idxMeanB (m := m) (n := n))
    (· - ·)]
  rw [Tape.GraphData.get_evalVec_input (Γ := ctx m n) (ss := [[m], [m, n]])
    (graphMeanBroadcast (m := m) (n := n)).toData () xV
    (idxX (m := m) (n := n) (ss := [[m], [m, n]]))
    (idxX (m := m) (n := n) (ss := [])) rfl, get_graphMeanBroadcast]
  rfl

/-- Stage `graphCenteredSq`: the squared centered block. -/
theorem get_graphCenteredSq (xV : CtxVec (ctx m n)) :
    CtxVec.get (Γ := ctx m n ++ [[m], [m, n], [m, n], [m, n]])
      (s := [m, n]) (idxCenteredSq (m := m) (n := n))
      (Tape.GraphData.evalVec (Γ := ctx m n)
        (ss := [[m], [m, n], [m, n], [m, n]])
        (graphCenteredSq (m := m) (n := n)).toData () xV) =
      valCenteredSq xV := by
  refine (Tape.GraphData.get_evalVec_snoc_last (Γ := ctx m n)
    (ss := [[m], [m, n], [m, n]]) (τ := [m, n])
    (graphCentered (m := m) (n := n)).toData
    (nodeCenteredSq (m := m) (n := n)).toNodeData () xV
      (idxCenteredSq (m := m) (n := n)) rfl).trans ?_
  rw [Tape.NodeData.forwardVec_of_zipWith (idxCentered (m := m) (n := n))
    (idxCentered (m := m) (n := n)) (· * ·)]
  rw [get_graphCentered]
  rfl

/-- Stage `graphVar`: the variance block. -/
theorem get_graphVar (xV : CtxVec (ctx m n)) :
    CtxVec.get
      (Γ := ctx m n ++ [[m], [m, n], [m, n], [m, n], [m]])
      (s := [m]) (idxVar (m := m) (n := n))
      (Tape.GraphData.evalVec (Γ := ctx m n)
        (ss := [[m], [m, n], [m, n], [m, n], [m]])
        (graphVar (m := m) (n := n)).toData () xV) =
      valVar xV := by
  refine (Tape.GraphData.get_evalVec_snoc_last (Γ := ctx m n)
    (ss := [[m], [m, n], [m, n], [m, n]]) (τ := [m])
    (graphCenteredSq (m := m) (n := n)).toData (nodeVar (m := m) (n := n)).toNodeData () xV
      (idxVar (m := m) (n := n)) rfl).trans ?_
  simp only [nodeVar, rowMean, Tape.LawfulNode.forwardVec_ofFn]
  rw [get_graphCenteredSq]
  rfl

/-- Stage `graphVarEps`: the shifted variance block. -/
theorem get_graphVarEps (ε : ℝ) (xV : CtxVec (ctx m n)) :
    CtxVec.get (Γ := ctx m n ++ ssVarEps m n) (s := [m]) (idxVarEps (m := m) (n := n))
      (Tape.GraphData.evalVec (Γ := ctx m n) (ss := ssVarEps m n)
        (graphVarEps (m := m) (n := n) ε).toData () xV) =
      valVarEps xV ε := by
  refine (Tape.GraphData.get_evalVec_snoc_last (Γ := ctx m n)
    (ss := [[m], [m, n], [m, n], [m, n], [m]]) (τ := [m])
    (graphVar (m := m) (n := n)).toData (nodeVarEps (m := m) (n := n) ε).toNodeData () xV
      (idxVarEps (m := m) (n := n))
    rfl).trans ?_
  simp only [nodeVarEps, elemwise, Tape.LawfulNode.forwardVec_ofFn]
  rw [get_graphVar]
  rfl

/-- Stage `graphStd`: the standard deviation block. -/
theorem get_graphStd (ε : ℝ) (xV : CtxVec (ctx m n)) :
    CtxVec.get (Γ := ctx m n ++ ssStd m n) (s := [m]) (idxStd (m := m) (n := n))
      (Tape.GraphData.evalVec (Γ := ctx m n) (ss := ssStd m n)
        (graphStd (m := m) (n := n) ε).toData () xV) =
      valStd xV ε := by
  refine (Tape.GraphData.get_evalVec_snoc_last (Γ := ctx m n) (ss := ssVarEps m n) (τ := [m])
    (graphVarEps (m := m) (n := n) ε).toData (nodeStd (m := m) (n := n)).toNodeData () xV
    (idxStd (m := m) (n := n)) rfl).trans ?_
  rw [Tape.NodeData.forwardVec_of_map (idxVarEps (m := m) (n := n))
    (fun v => FloatLib.Numerics.MathFunctions.sqrt (Max.max v 0))]
  rw [get_graphVarEps]
  rfl

/-- Stage `graphInvStd`: the inverse standard deviation block. -/
theorem get_graphInvStd (ε : ℝ) (xV : CtxVec (ctx m n)) :
    CtxVec.get (Γ := ctx m n ++ (ssStd m n ++ [[m]])) (s := [m])
      (idxInvStd (m := m) (n := n))
      (Tape.GraphData.evalVec (Γ := ctx m n) (ss := ssStd m n ++ [[m]])
        (graphInvStd (m := m) (n := n) ε).toData () xV) =
      valInvStd xV ε := by
  refine (Tape.GraphData.get_evalVec_snoc_last (Γ := ctx m n) (ss := ssStd m n) (τ := [m])
    (graphStd (m := m) (n := n) ε).toData (nodeInvStd (m := m) (n := n)).toNodeData () xV
    (idxInvStd (m := m) (n := n)) rfl).trans ?_
  rw [Tape.NodeData.forwardVec_of_map (idxStd (m := m) (n := n)) Context.reciprocal,
    get_graphStd]
  simp only [Context.reciprocal, one_div, valInvStd]

/-- Stage `graphInvStdBroadcast`: the broadcast inverse standard deviation block. -/
theorem get_graphInvStdBroadcast (ε : ℝ) (xV : CtxVec (ctx m n)) :
    CtxVec.get (Γ := ctx m n ++ (ssStd m n ++ [[m], [m, n]]))
      (s := [m, n]) (idxInvStdBroadcast (m := m) (n := n))
      (Tape.GraphData.evalVec (Γ := ctx m n) (ss := ssStd m n ++ [[m], [m, n]])
        (graphInvStdBroadcast (m := m) (n := n) ε).toData () xV) =
      valInvStdB xV ε := by
  refine (Tape.GraphData.get_evalVec_snoc_last (Γ := ctx m n) (ss := ssStd m n ++ [[m]])
    (τ := [m, n]) (graphInvStd (m := m) (n := n) ε).toData
      (nodeInvStdB (m := m) (n := n)).toNodeData () xV
    (idxInvStdBroadcast (m := m) (n := n)) rfl).trans ?_
  simp only [nodeInvStdB, broadcastRow, Tape.LawfulNode.forwardVec_ofFn, getVec]
  rw [get_graphInvStd]
  rfl

/-- At stage `graphInvStdBroadcast`, the block at `idxCenteredForNorm` is the centered input
`valCentered`. -/
theorem get_centeredForNorm (ε : ℝ) (xV : CtxVec (ctx m n)) :
    CtxVec.get (Γ := ctx m n ++ (ssStd m n ++ [[m], [m, n]]))
      (s := [m, n]) (idxCenteredForNorm (m := m) (n := n))
      (Tape.GraphData.evalVec (Γ := ctx m n) (ss := ssStd m n ++ [[m], [m, n]])
        (graphInvStdBroadcast (m := m) (n := n) ε).toData () xV) =
      valCentered xV := by
  refine (Tape.GraphData.get_evalVec_snoc_of_lt (Γ := ctx m n) (ss := ssStd m n ++ [[m]])
    (τ := [m, n]) (graphInvStd (m := m) (n := n) ε).toData
      (nodeInvStdB (m := m) (n := n)).toNodeData () xV
    (idxCenteredForNorm (m := m) (n := n))
    (Idx.weaken (idxCentered (m := m) (n := n))
      [[m, n], [m], [m], [m], [m]]) rfl).trans ?_
  refine (Tape.GraphData.get_evalVec_snoc_of_lt (Γ := ctx m n) (ss := ssStd m n)
    (τ := [m]) (graphStd (m := m) (n := n) ε).toData
      (nodeInvStd (m := m) (n := n)).toNodeData () xV _
    (Idx.weaken (idxCentered (m := m) (n := n))
      [[m, n], [m], [m], [m]]) rfl).trans ?_
  refine (Tape.GraphData.get_evalVec_snoc_of_lt (Γ := ctx m n) (ss := ssVarEps m n)
    (τ := [m]) (graphVarEps (m := m) (n := n) ε).toData
      (nodeStd (m := m) (n := n)).toNodeData () xV _
    (Idx.weaken (idxCentered (m := m) (n := n)) [[m, n], [m], [m]])
    rfl).trans ?_
  refine (Tape.GraphData.get_evalVec_snoc_of_lt (Γ := ctx m n)
    (ss := [[m], [m, n], [m, n], [m, n], [m]])
    (τ := [m]) (graphVar (m := m) (n := n)).toData
      (nodeVarEps (m := m) (n := n) ε).toNodeData () xV _
    (Idx.weaken (idxCentered (m := m) (n := n)) [[m, n], [m]]) rfl).trans ?_
  refine (Tape.GraphData.get_evalVec_snoc_of_lt (Γ := ctx m n)
    (ss := [[m], [m, n], [m, n], [m, n]])
    (τ := [m]) (graphCenteredSq (m := m) (n := n)).toData
      (nodeVar (m := m) (n := n)).toNodeData () xV _
    (Idx.weaken (idxCentered (m := m) (n := n)) [[m, n]]) rfl).trans ?_
  refine (Tape.GraphData.get_evalVec_snoc_of_lt (Γ := ctx m n)
    (ss := [[m], [m, n], [m, n]])
    (τ := [m, n]) (graphCentered (m := m) (n := n)).toData
      (nodeCenteredSq (m := m) (n := n)).toNodeData () xV _
    (idxCentered (m := m) (n := n)) rfl).trans ?_
  exact get_graphCentered xV

/-- Stage `graphNormalized`: the normalized block. -/
theorem get_graphNormalized (ε : ℝ) (xV : CtxVec (ctx m n)) :
    CtxVec.get (Γ := ctx m n ++ (ssStd m n ++ [[m], [m, n], [m, n]]))
      (s := [m, n])
      (Idx.last (Γ := ctx m n) (ss := ssStd m n ++ [[m], [m, n]])
        (τ := [m, n]))
      (Tape.GraphData.evalVec (Γ := ctx m n)
        (ss := ssStd m n ++ [[m], [m, n], [m, n]])
        (graphNormalized (m := m) (n := n) ε).toData () xV) =
      valNorm xV ε := by
  refine (Tape.GraphData.get_evalVec_snoc_last (Γ := ctx m n)
    (ss := ssStd m n ++ [[m], [m, n]]) (τ := [m, n])
    (graphInvStdBroadcast (m := m) (n := n) ε).toData (nodeNorm (m := m) (n := n)).toNodeData () xV
      _ rfl).trans ?_
  rw [Tape.NodeData.forwardVec_of_zipWith (idxCenteredForNorm (m := m) (n := n))
    (idxInvStdBroadcast (m := m) (n := n)) (· * ·)]
  rw [get_centeredForNorm, get_graphInvStdBroadcast]
  rfl

/-- Stage `graphGammaBroadcast`: the broadcast scale block. -/
theorem get_graphGammaBroadcast (ε : ℝ) (xV : CtxVec (ctx m n)) :
    CtxVec.get
      (Γ := ctx m n ++ (ssStd m n ++ [[m], [m, n], [m, n], [m, n]]))
      (s := [m, n]) (idxGammaBroadcast (m := m) (n := n))
      (Tape.GraphData.evalVec (Γ := ctx m n)
        (ss := ssStd m n ++ [[m], [m, n], [m, n], [m, n]])
        (graphGammaBroadcast (m := m) (n := n) ε).toData () xV) =
      valGammaB xV := by
  refine (Tape.GraphData.get_evalVec_snoc_last (Γ := ctx m n)
    (ss := ssStd m n ++ [[m], [m, n], [m, n]]) (τ := [m, n])
    (graphNormalized (m := m) (n := n) ε).toData (nodeGammaB (m := m) (n := n)).toNodeData () xV
    (idxGammaBroadcast (m := m) (n := n)) rfl).trans ?_
  simp only [nodeGammaB, broadcastCol, Tape.LawfulNode.forwardVec_ofFn, getVec]
  rw [Tape.GraphData.get_evalVec_input (Γ := ctx m n)
    (ss := ssStd m n ++ [[m], [m, n], [m, n]])
    (graphNormalized (m := m) (n := n) ε).toData () xV _
      (idxGamma (m := m) (n := n) (ss := [])) rfl]
  rfl

/-- At stage `graphGammaBroadcast`, the block at `idxNormForScale` is the normalized input
`valNorm`. -/
theorem get_normForScale (ε : ℝ) (xV : CtxVec (ctx m n)) :
    CtxVec.get
      (Γ := ctx m n ++ (ssStd m n ++ [[m], [m, n], [m, n], [m, n]]))
      (s := [m, n]) (idxNormForScale (m := m) (n := n))
      (Tape.GraphData.evalVec (Γ := ctx m n)
        (ss := ssStd m n ++ [[m], [m, n], [m, n], [m, n]])
        (graphGammaBroadcast (m := m) (n := n) ε).toData () xV) =
      valNorm xV ε := by
  refine (Tape.GraphData.get_evalVec_snoc_of_lt (Γ := ctx m n)
    (ss := ssStd m n ++ [[m], [m, n], [m, n]]) (τ := [m, n])
    (graphNormalized (m := m) (n := n) ε).toData (nodeGammaB (m := m) (n := n)).toNodeData () xV
    (idxNormForScale (m := m) (n := n))
    (Idx.last (Γ := ctx m n) (ss := ssStd m n ++ [[m], [m, n]])
      (τ := [m, n])) rfl).trans ?_
  exact get_graphNormalized ε xV

/-- Stage `graphScaled`: the scaled block. -/
theorem get_graphScaled (ε : ℝ) (xV : CtxVec (ctx m n)) :
    CtxVec.get
      (Γ := ctx m n ++ (ssStd m n ++ [[m], [m, n], [m, n], [m, n], [m, n]]))
      (s := [m, n])
      (Idx.last (Γ := ctx m n)
        (ss := ssStd m n ++ [[m], [m, n], [m, n], [m, n]])
        (τ := [m, n]))
      (Tape.GraphData.evalVec (Γ := ctx m n)
        (ss := ssStd m n ++ [[m], [m, n], [m, n], [m, n], [m, n]])
        (graphScaled (m := m) (n := n) ε).toData () xV) =
      valScaled xV ε := by
  refine (Tape.GraphData.get_evalVec_snoc_last (Γ := ctx m n)
    (ss := ssStd m n ++ [[m], [m, n], [m, n], [m, n]])
    (τ := [m, n]) (graphGammaBroadcast (m := m) (n := n) ε).toData
      (nodeScaled (m := m) (n := n)).toNodeData ()
    xV _ rfl).trans ?_
  rw [Tape.NodeData.forwardVec_of_zipWith (idxNormForScale (m := m) (n := n))
    (idxGammaBroadcast (m := m) (n := n)) (· * ·)]
  rw [get_normForScale, get_graphGammaBroadcast]
  rfl

/-- Stage `graphBetaBroadcast`: the broadcast shift block. -/
theorem get_graphBetaBroadcast (ε : ℝ) (xV : CtxVec (ctx m n)) :
    CtxVec.get
      (Γ := ctx m n ++ (ssStd m n ++ [[m], [m, n], [m, n], [m, n], [m, n], [m, n]]))
      (s := [m, n]) (idxBetaBroadcast (m := m) (n := n))
      (Tape.GraphData.evalVec (Γ := ctx m n)
        (ss := ssStd m n ++ [[m], [m, n], [m, n], [m, n], [m, n], [m, n]])
        (graphBetaBroadcast (m := m) (n := n) ε).toData () xV) =
      valBetaB xV := by
  refine (Tape.GraphData.get_evalVec_snoc_last (Γ := ctx m n)
    (ss := ssStd m n ++ [[m], [m, n], [m, n], [m, n], [m, n]])
    (τ := [m, n]) (graphScaled (m := m) (n := n) ε).toData
      (nodeBetaB (m := m) (n := n)).toNodeData () xV
    (idxBetaBroadcast (m := m) (n := n)) rfl).trans ?_
  simp only [nodeBetaB, broadcastCol, Tape.LawfulNode.forwardVec_ofFn, getVec]
  rw [Tape.GraphData.get_evalVec_input (Γ := ctx m n)
    (ss := ssStd m n ++ [[m], [m, n], [m, n], [m, n], [m, n]])
    (graphScaled (m := m) (n := n) ε).toData () xV _ (idxBeta (m := m) (n := n) (ss := [])) rfl]
  rfl

/-- At stage `graphBetaBroadcast`, the block at `idxScaledForOutput` is the scaled input
`valScaled`. -/
theorem get_scaledForOutput (ε : ℝ) (xV : CtxVec (ctx m n)) :
    CtxVec.get
      (Γ := ctx m n ++ (ssStd m n ++ [[m], [m, n], [m, n], [m, n], [m, n], [m, n]]))
      (s := [m, n]) (idxScaledForOutput (m := m) (n := n))
      (Tape.GraphData.evalVec (Γ := ctx m n)
        (ss := ssStd m n ++ [[m], [m, n], [m, n], [m, n], [m, n], [m, n]])
        (graphBetaBroadcast (m := m) (n := n) ε).toData () xV) =
      valScaled xV ε := by
  refine (Tape.GraphData.get_evalVec_snoc_of_lt (Γ := ctx m n)
    (ss := ssStd m n ++ [[m], [m, n], [m, n], [m, n], [m, n]])
    (τ := [m, n]) (graphScaled (m := m) (n := n) ε).toData
      (nodeBetaB (m := m) (n := n)).toNodeData () xV
    (idxScaledForOutput (m := m) (n := n))
    (Idx.last (Γ := ctx m n)
      (ss := ssStd m n ++ [[m], [m, n], [m, n], [m, n]])
      (τ := [m, n])) rfl).trans ?_
  exact get_graphScaled ε xV

/-- The output block of the full LayerNorm graph. -/
theorem get_idxY (ε : ℝ) (xV : CtxVec (ctx m n)) :
    CtxVec.get (Γ := ctx m n ++ LayerNorm.ss m n) (s := [m, n]) (idxY (m := m) (n := n))
      (Tape.GraphData.evalVec (Γ := ctx m n) (ss := LayerNorm.ss m n)
        (graph (m := m) (n := n) ε).toData () xV) =
      valY xV ε := by
  refine (Tape.GraphData.get_evalVec_snoc_last (Γ := ctx m n)
    (ss := ssStd m n ++ [[m], [m, n], [m, n], [m, n], [m, n], [m, n]])
    (τ := [m, n]) (graphBetaBroadcast (m := m) (n := n) ε).toData
      (nodeY (m := m) (n := n)).toNodeData () xV
    (idxY (m := m) (n := n)) rfl).trans ?_
  rw [Tape.NodeData.forwardVec_of_zipWith (idxScaledForOutput (m := m) (n := n))
    (idxBetaBroadcast (m := m) (n := n)) (· + ·)]
  rw [get_scaledForOutput, get_graphBetaBroadcast]
  rfl

/-! ## Positivity of the denominators -/

/-- The variance block is a mean of squares, hence nonnegative. -/
theorem valVar_nonneg (xV : CtxVec (ctx m n)) (i : Fin (Shape.size [m])) :
    0 ≤ valVar xV i := by
  simp only [valVar, LinearIsometryEquiv.piLpCongrLeft_apply, Equiv.piCongrLeft'_apply,
    finCongr_symm_apply]
  rw [RowNorm.rowMeanCLM_eq]
  refine Finset.expect_nonneg fun j _ => ?_
  simp only [valCenteredSq]
  exact mul_self_nonneg _

/-- With positive `ε`, every shifted variance is positive. -/
theorem valVarEps_pos {ε : ℝ} (hε : 0 < ε) (xV : CtxVec (ctx m n))
    (i : Fin (Shape.size [m])) :
    0 < valVarEps xV ε i := by
  simp only [valVarEps, PiLp.toLp_apply]
  exact add_pos_of_nonneg_of_pos (valVar_nonneg xV i) hε

/-- With positive `ε`, every clamped standard deviation is positive. -/
theorem valStd_pos {ε : ℝ} (hε : 0 < ε) (xV : CtxVec (ctx m n))
    (i : Fin (Shape.size [m])) :
    0 < valStd xV ε i := by
  simp only [valStd, PiLp.toLp_apply]
  exact Real.sqrt_pos.2 (lt_max_of_lt_left (valVarEps_pos hε xV i))

/-- The domain hypothesis of the LayerNorm square root follows from `0 < ε`. -/
theorem varEps_pos_of_eps_pos {ε : ℝ} (hε : 0 < ε) (xV : CtxVec (ctx m n)) :
    ∀ i : Fin (Shape.size [m]),
      0 < CtxVec.get (Γ := ctx m n ++ ssVarEps m n) (s := [m]) (idxVarEps (m := m) (n := n))
        (Tape.GraphData.evalVec (Γ := ctx m n) (ss := ssVarEps m n)
          (graphVarEps (m := m) (n := n) ε).toData () xV) i := by
  intro i
  rw [get_graphVarEps]
  exact valVarEps_pos hε xV i

/-- The domain hypothesis of the LayerNorm reciprocal follows from `0 < ε`. -/
theorem std_ne_zero_of_eps_pos {ε : ℝ} (hε : 0 < ε) (xV : CtxVec (ctx m n)) :
    ∀ i : Fin (Shape.size [m]),
      CtxVec.get (Γ := ctx m n ++ ssStd m n) (s := [m]) (idxStd (m := m) (n := n))
        (Tape.GraphData.evalVec (Γ := ctx m n) (ss := ssStd m n)
          (graphStd (m := m) (n := n) ε).toData () xV) i ≠ 0 := by
  intro i
  rw [get_graphStd]
  exact (valStd_pos hε xV i).ne'

end LayerNorm

end

end TorchLean
