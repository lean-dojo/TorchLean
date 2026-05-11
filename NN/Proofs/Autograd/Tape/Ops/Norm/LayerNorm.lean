/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Normalization
public import NN.Proofs.Autograd.Tape.Nodes.GraphComposition
public import NN.Proofs.Autograd.Tape.Util.Idx

public import Mathlib.Analysis.Calculus.Deriv.Add

/-!
# LayerNorm

Pointwise analytic correctness for a **LayerNorm** graph.

This is spec-level over `ℝ`. It is the proof-tape counterpart of the runtime/spec LayerNorm in
`Spec.layerNorm`: a `seqLen × embedDim` tensor is normalized across the last axis, the row-wise
normalizer is broadcast back over each token, and affine parameters `gamma`/`beta` are broadcast
over the sequence dimension. The runtime API and compiled IR path both route through that spec
definition; this file proves the corresponding reverse-mode graph rule.

Because the proof graph uses the differentiable scalar nodes `sqrt (max x 0)` and `inv`, the main
theorem is pointwise (`GraphFDerivCorrectAt`) with explicit domain assumptions. Those hypotheses
are the honest mathematical boundary: away from the clamp kink and zero denominator, backprop is the
adjoint of the Fréchet derivative. The executable `Spec.layerNorm` additionally clamps the raw
variance before adding epsilon as a numerical guard; over exact real variance this is the same
contract on the positive branch used by the proof.

## PyTorch correspondence / citations
- Conceptually corresponds to `torch.nn.LayerNorm` (without batching/running stats): normalize along
  the last dimension, then apply affine parameters `(gamma,beta)`.
  https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
-/

@[expose] public section


namespace Proofs
namespace Autograd

open Spec

open scoped BigOperators

noncomputable section

namespace LayerNorm

open TapeNodes

/-- Matrix shape `m×n`. -/
abbrev MatShape (m n : Nat) : Shape := .dim m (.dim n .scalar)
/-- Vector shape `k`. -/
abbrev VecShape (k : Nat) : Shape := .dim k .scalar

/-- Input context shapes: `[X, gamma, beta]` for layer norm over the last axis. -/
abbrev ΓLN (m n : Nat) : List Shape := [MatShape m n, VecShape n, VecShape n]

/-- First 6 intermediates in the LayerNorm computation (up to `var_eps`). -/
abbrev ssPrefix6 (m n : Nat) : List Shape :=
  [ VecShape m   -- mean
  , MatShape m n -- mean_b
  , MatShape m n -- centered
  , MatShape m n -- centered_sq
  , VecShape m   -- var
  , VecShape m   -- var_eps
  ]

/-- Prefix intermediates up to `std` (adds one more vector). -/
abbrev ssPrefix7 (m n : Nat) : List Shape := ssPrefix6 m n ++ [VecShape m] -- std

/-- Full list of intermediates for the LayerNorm graph in this file. -/
abbrev ssLayerNorm (m n : Nat) : List Shape :=
  ssPrefix7 m n ++
    [ VecShape m   -- inv_std
    , MatShape m n -- inv_std_b
    , MatShape m n -- normalized
    , MatShape m n -- gamma_b
    , MatShape m n -- scaled
    , MatShape m n -- beta_b
    , MatShape m n -- y
    ]

/-- Index of the input matrix `X` in the base LayerNorm context `ΓLN m n ++ ss`. -/
def idxX {m n : Nat} {ss : List Shape} : Idx (ΓLN m n ++ ss) (MatShape m n) :=
  ⟨⟨0, by simp [ΓLN]⟩, by simp [ΓLN]⟩

/-- Index of the scale vector `gamma` in the base LayerNorm context `ΓLN m n ++ ss`. -/
def idxGamma {m n : Nat} {ss : List Shape} : Idx (ΓLN m n ++ ss) (VecShape n) :=
  ⟨⟨1, by simp [ΓLN]⟩, by simp [ΓLN]⟩

/-- Index of the shift vector `beta` in the base LayerNorm context `ΓLN m n ++ ss`. -/
def idxBeta {m n : Nat} {ss : List Shape} : Idx (ΓLN m n ++ ss) (VecShape n) :=
  ⟨⟨2, by simp [ΓLN]⟩, by simp [ΓLN]⟩

/-- Index helper for the last element of an extended context `Γ ++ ss ++ [τ]`. -/
def idxLast {Γ : List Shape} {ss : List Shape} {τ : Shape} :
    Idx (Γ ++ ss ++ [τ]) τ :=
  _root_.Proofs.Autograd.Idx.last (Γ := Γ) (ss := ss) (τ := τ)

-- ---------------------------------------------------------------------------
-- LayerNorm graph (explicit `snoc` chain; no `let`-blocked reducibility)
-- ---------------------------------------------------------------------------

-- Prefix nodes (mean/variance + epsilon)

/-- Mean over the last axis: `mean : ℝ^{m×n} → ℝ^{m}`. -/
def nodeMean {m n : Nat} : Node (ΓLN m n) (VecShape m) :=
  rowMean (Γ := ΓLN m n) (m := m) (n := n) (idx := idxX (m := m) (n := n) (ss := []))

/-- Graph prefix producing `[mean]`. -/
def g1 {m n : Nat} : Graph (ΓLN m n) [VecShape m] :=
  .snoc (.nil) (nodeMean (m := m) (n := n))

/-- Index of `mean` in the extended context `ΓLN ++ [mean]`. -/
def idxMean {m n : Nat} : Idx (ΓLN m n ++ [VecShape m]) (VecShape m) :=
  idxLast (Γ := ΓLN m n) (ss := []) (τ := VecShape m)

/-- Broadcast `mean` back to `m×n` (row-wise). -/
def nodeMeanB {m n : Nat} : Node (ΓLN m n ++ [VecShape m]) (MatShape m n) :=
  broadcastRow (Γ := ΓLN m n ++ [VecShape m]) (m := m) (n := n) (idx := idxMean (m := m) (n := n))

/-- Graph prefix producing `[mean, mean_b]`. -/
def g2 {m n : Nat} : Graph (ΓLN m n) [VecShape m, MatShape m n] :=
  .snoc (g1 (m := m) (n := n)) (nodeMeanB (m := m) (n := n))

/-- Index of `mean_b` in `ΓLN ++ [mean, mean_b]`. -/
def idxMeanB {m n : Nat} : Idx (ΓLN m n ++ [VecShape m, MatShape m n]) (MatShape m n) :=
  idxLast (Γ := ΓLN m n) (ss := [VecShape m]) (τ := MatShape m n)

/-- Center: `centered := X - mean_b`. -/
def nodeCentered {m n : Nat} : Node (ΓLN m n ++ [VecShape m, MatShape m n]) (MatShape m n) :=
  sub (Γ := ΓLN m n ++ [VecShape m, MatShape m n]) (s := MatShape m n)
    (a := idxX (m := m) (n := n) (ss := [VecShape m, MatShape m n]))
    (b := idxMeanB (m := m) (n := n))

/-- Graph prefix producing `[mean, mean_b, centered]`. -/
def g3 {m n : Nat} : Graph (ΓLN m n) [VecShape m, MatShape m n, MatShape m n] :=
  .snoc (g2 (m := m) (n := n)) (nodeCentered (m := m) (n := n))

/-- Index of `centered` in `ΓLN ++ [mean, mean_b, centered]`. -/
def idxCentered {m n : Nat} :
    Idx (ΓLN m n ++ [VecShape m, MatShape m n, MatShape m n]) (MatShape m n) :=
  idxLast (Γ := ΓLN m n) (ss := [VecShape m, MatShape m n]) (τ := MatShape m n)

/-- Square `centered`: `centered_sq := centered ⊙ centered`. -/
def nodeCenteredSq {m n : Nat} : Node (ΓLN m n ++ [VecShape m, MatShape m n, MatShape m n])
  (MatShape m n) :=
  mul (Γ := ΓLN m n ++ [VecShape m, MatShape m n, MatShape m n]) (s := MatShape m n)
    (a := idxCentered (m := m) (n := n)) (b := idxCentered (m := m) (n := n))

/-- Graph prefix producing `[mean, mean_b, centered, centered_sq]`. -/
def g4 {m n : Nat} : Graph (ΓLN m n) [VecShape m, MatShape m n, MatShape m n, MatShape m n] :=
  .snoc (g3 (m := m) (n := n)) (nodeCenteredSq (m := m) (n := n))

/-- Index of `centered_sq` in the extended context. -/
def idxCenteredSq {m n : Nat} :
    Idx (ΓLN m n ++ [VecShape m, MatShape m n, MatShape m n, MatShape m n]) (MatShape m n) :=
  idxLast (Γ := ΓLN m n) (ss := [VecShape m, MatShape m n, MatShape m n]) (τ := MatShape m n)

/-- Variance per row: `var := mean(centered_sq)` producing a length-`m` vector. -/
def nodeVar {m n : Nat} :
    Node (ΓLN m n ++ [VecShape m, MatShape m n, MatShape m n, MatShape m n]) (VecShape m) :=
  rowMean (Γ := ΓLN m n ++ [VecShape m, MatShape m n, MatShape m n, MatShape m n])
    (m := m) (n := n) (idx := idxCenteredSq (m := m) (n := n))

/-- Graph prefix producing `[mean, mean_b, centered, centered_sq, var]`. -/
def g5 {m n : Nat} : Graph (ΓLN m n) [VecShape m, MatShape m n, MatShape m n, MatShape m n, VecShape
  m] :=
  .snoc (g4 (m := m) (n := n)) (nodeVar (m := m) (n := n))

/-- Index of `var` in the extended context. -/
def idxVar {m n : Nat} :
    Idx (ΓLN m n ++ [VecShape m, MatShape m n, MatShape m n, MatShape m n, VecShape m]) (VecShape m)
      :=
  idxLast (Γ := ΓLN m n) (ss := [VecShape m, MatShape m n, MatShape m n, MatShape m n]) (τ :=
    VecShape m)

/-- Add epsilon: `var_eps := var + ε`. -/
def nodeVarEps {m n : Nat} (ε : ℝ) :
    Node (ΓLN m n ++ [VecShape m, MatShape m n, MatShape m n, MatShape m n, VecShape m]) (VecShape
      m) :=
  elemwise (Γ := ΓLN m n ++ [VecShape m, MatShape m n, MatShape m n, MatShape m n, VecShape m])
    (s := VecShape m) (idxVar (m := m) (n := n)) (fun z => z + ε) (fun _ => 1)

/-- Graph prefix computing the first 6 intermediates (`ssPrefix6`). -/
def layerNormPrefix6 {m n : Nat} (ε : ℝ) : Graph (ΓLN m n) (ssPrefix6 m n) :=
  .snoc (g5 (m := m) (n := n)) (nodeVarEps (m := m) (n := n) ε)

/-- Index of `var_eps` in `ΓLN ++ ssPrefix6`. -/
def idxVarEps {m n : Nat} : Idx (ΓLN m n ++ ssPrefix6 m n) (VecShape m) :=
  idxLast (Γ := ΓLN m n)
    (ss := [VecShape m, MatShape m n, MatShape m n, MatShape m n, VecShape m])
    (τ := VecShape m)

/--
Standard deviation: `std := sqrt_clamp(var_eps)`.

This is where the development becomes pointwise: differentiability depends on the (clamped) input.
-/
def nodeStd {m n : Nat} : Node (ΓLN m n ++ ssPrefix6 m n) (VecShape m) :=
  sqrtClamp (Γ := ΓLN m n ++ ssPrefix6 m n) (s := VecShape m) (idxVarEps (m := m) (n := n))

/-- Graph prefix computing `ssPrefix7` (adds `std`). -/
def layerNormPrefix7 {m n : Nat} (ε : ℝ) : Graph (ΓLN m n) (ssPrefix7 m n) :=
  .snoc (layerNormPrefix6 (m := m) (n := n) ε) (nodeStd (m := m) (n := n))

-- Remaining nodes (normalize, scale, shift)

/-- Index of `std` in `ΓLN ++ ssPrefix7`. -/
def idxStd {m n : Nat} : Idx (ΓLN m n ++ ssPrefix7 m n) (VecShape m) :=
  idxLast (Γ := ΓLN m n) (ss := ssPrefix6 m n) (τ := VecShape m)

/-- Inverse standard deviation: `inv_std := 1/std`. -/
def nodeInvStd {m n : Nat} : Node (ΓLN m n ++ ssPrefix7 m n) (VecShape m) :=
  inv (Γ := ΓLN m n ++ ssPrefix7 m n) (s := VecShape m) (idxStd (m := m) (n := n))

/-- Graph prefix adding `inv_std`. -/
def g8 {m n : Nat} (ε : ℝ) : Graph (ΓLN m n) (ssPrefix7 m n ++ [VecShape m]) :=
  .snoc (layerNormPrefix7 (m := m) (n := n) ε) (nodeInvStd (m := m) (n := n))

/-- Index of `inv_std` in the extended context. -/
def idxInvStd {m n : Nat} : Idx (ΓLN m n ++ (ssPrefix7 m n ++ [VecShape m])) (VecShape m) :=
  idxLast (Γ := ΓLN m n) (ss := ssPrefix7 m n) (τ := VecShape m)

/-- Broadcast `inv_std` back to `m×n` (row-wise). -/
def nodeInvStdB {m n : Nat} :
    Node (ΓLN m n ++ (ssPrefix7 m n ++ [VecShape m])) (MatShape m n) :=
  broadcastRow (Γ := ΓLN m n ++ (ssPrefix7 m n ++ [VecShape m]))
    (m := m) (n := n) (idx := idxInvStd (m := m) (n := n))

/-- Graph prefix adding `inv_std_b`. -/
def g9 {m n : Nat} (ε : ℝ) : Graph (ΓLN m n) (ssPrefix7 m n ++ [VecShape m, MatShape m n]) :=
  .snoc (g8 (m := m) (n := n) ε) (nodeInvStdB (m := m) (n := n))

/-- Index of `centered` in the stage-`g9` context. -/
def idxCentered9 {m n : Nat} :
    Idx (ΓLN m n ++ (ssPrefix7 m n ++ [VecShape m, MatShape m n])) (MatShape m n) :=
  Idx.weaken (Γ := ΓLN m n ++ [VecShape m, MatShape m n, MatShape m n]) (idxCentered (m := m) (n :=
    n))
    (rest := [MatShape m n, VecShape m, VecShape m, VecShape m, VecShape m, MatShape m n])

/-- Index of `inv_std_b` in the stage-`g9` context. -/
def idxInvStdB9 {m n : Nat} :
    Idx (ΓLN m n ++ (ssPrefix7 m n ++ [VecShape m, MatShape m n])) (MatShape m n) :=
  idxLast (Γ := ΓLN m n) (ss := ssPrefix7 m n ++ [VecShape m]) (τ := MatShape m n)

/-- Node computing `normalized := centered ⊙ inv_std_b`. -/
def nodeNorm {m n : Nat} :
    Node (ΓLN m n ++ (ssPrefix7 m n ++ [VecShape m, MatShape m n])) (MatShape m n) :=
  mul (Γ := ΓLN m n ++ (ssPrefix7 m n ++ [VecShape m, MatShape m n])) (s := MatShape m n)
    (a := idxCentered9 (m := m) (n := n)) (b := idxInvStdB9 (m := m) (n := n))

/-- Graph prefix producing `normalized := centered ⊙ inv_std_b`. -/
def g10 {m n : Nat} (ε : ℝ) : Graph (ΓLN m n) (ssPrefix7 m n ++ [VecShape m, MatShape m n, MatShape
  m n]) :=
  .snoc (g9 (m := m) (n := n) ε) (nodeNorm (m := m) (n := n))

/-- Broadcast `gamma` to `m×n` (column-wise). -/
def nodeGammaB {m n : Nat} :
    Node (ΓLN m n ++ (ssPrefix7 m n ++ [VecShape m, MatShape m n, MatShape m n])) (MatShape m n) :=
  broadcastCol
    (Γ := ΓLN m n ++ (ssPrefix7 m n ++ [VecShape m, MatShape m n, MatShape m n]))
    (m := m) (n := n)
    (idx := idxGamma (m := m) (n := n) (ss := ssPrefix7 m n ++ [VecShape m, MatShape m n, MatShape m
      n]))

/-- Graph prefix adding `gamma_b`. -/
def g11 {m n : Nat} (ε : ℝ) :
    Graph (ΓLN m n) (ssPrefix7 m n ++ [VecShape m, MatShape m n, MatShape m n, MatShape m n]) :=
  .snoc (g10 (m := m) (n := n) ε) (nodeGammaB (m := m) (n := n))

/-- Index of `normalized` in the context at stage `g11`. -/
def idxNorm11 {m n : Nat} :
    Idx (ΓLN m n ++ (ssPrefix7 m n ++ [VecShape m, MatShape m n, MatShape m n, MatShape m n]))
      (MatShape m n) :=
  _root_.Proofs.Autograd.Idx.weaken (Γ := ΓLN m n ++ (ssPrefix7 m n ++ [VecShape m, MatShape m n,
    MatShape m n]))
    (idxLast (Γ := ΓLN m n) (ss := ssPrefix7 m n ++ [VecShape m, MatShape m n]) (τ := MatShape m n))
    (rest := [MatShape m n])

/-- Index of `gamma_b` at stage `g11`. -/
def idxGammaB11 {m n : Nat} :
    Idx (ΓLN m n ++ (ssPrefix7 m n ++ [VecShape m, MatShape m n, MatShape m n, MatShape m n]))
      (MatShape m n) :=
  idxLast (Γ := ΓLN m n) (ss := ssPrefix7 m n ++ [VecShape m, MatShape m n, MatShape m n]) (τ :=
    MatShape m n)

/-- Scale: `scaled := normalized ⊙ gamma_b`. -/
def nodeScaled {m n : Nat} :
    Node (ΓLN m n ++ (ssPrefix7 m n ++ [VecShape m, MatShape m n, MatShape m n, MatShape m n]))
      (MatShape m n) :=
  mul (Γ := ΓLN m n ++ (ssPrefix7 m n ++ [VecShape m, MatShape m n, MatShape m n, MatShape m n])) (s
    := MatShape m n)
    (a := idxNorm11 (m := m) (n := n)) (b := idxGammaB11 (m := m) (n := n))

/-- Graph prefix adding `scaled`. -/
def g12 {m n : Nat} (ε : ℝ) :
    Graph (ΓLN m n) (ssPrefix7 m n ++ [VecShape m, MatShape m n, MatShape m n, MatShape m n,
      MatShape m n]) :=
  .snoc (g11 (m := m) (n := n) ε) (nodeScaled (m := m) (n := n))

/-- Broadcast `beta` to `m×n` (column-wise). -/
def nodeBetaB {m n : Nat} :
    Node (ΓLN m n ++ (ssPrefix7 m n ++ [VecShape m, MatShape m n, MatShape m n, MatShape m n,
      MatShape m n])) (MatShape m n) :=
  broadcastCol
    (Γ := ΓLN m n ++ (ssPrefix7 m n ++ [VecShape m, MatShape m n, MatShape m n, MatShape m n,
      MatShape m n]))
    (m := m) (n := n)
    (idx := idxBeta (m := m) (n := n)
      (ss := ssPrefix7 m n ++ [VecShape m, MatShape m n, MatShape m n, MatShape m n, MatShape m n]))

/-- Graph prefix adding `beta_b`. -/
def g13 {m n : Nat} (ε : ℝ) :
    Graph (ΓLN m n) (ssPrefix7 m n ++ [VecShape m, MatShape m n, MatShape m n, MatShape m n,
      MatShape m n, MatShape m n]) :=
  .snoc (g12 (m := m) (n := n) ε) (nodeBetaB (m := m) (n := n))

/-- Index of `scaled` at stage `g13`. -/
def idxScaled13 {m n : Nat} :
    Idx (ΓLN m n ++ (ssPrefix7 m n ++ [VecShape m, MatShape m n, MatShape m n, MatShape m n,
      MatShape m n, MatShape m n])) (MatShape m n) :=
  _root_.Proofs.Autograd.Idx.weaken (Γ := ΓLN m n ++ (ssPrefix7 m n ++ [VecShape m, MatShape m n,
    MatShape m n, MatShape m n, MatShape m n]))
    (idxLast (Γ := ΓLN m n) (ss := ssPrefix7 m n ++ [VecShape m, MatShape m n, MatShape m n,
      MatShape m n]) (τ := MatShape m n))
    (rest := [MatShape m n])

/-- Index of `beta_b` at stage `g13`. -/
def idxBetaB13 {m n : Nat} :
    Idx (ΓLN m n ++ (ssPrefix7 m n ++ [VecShape m, MatShape m n, MatShape m n, MatShape m n,
      MatShape m n, MatShape m n])) (MatShape m n) :=
  idxLast (Γ := ΓLN m n) (ss := ssPrefix7 m n ++ [VecShape m, MatShape m n, MatShape m n, MatShape m
    n, MatShape m n]) (τ := MatShape m n)

/-- Output: `y := scaled + beta_b`. -/
def nodeY {m n : Nat} :
    Node (ΓLN m n ++ (ssPrefix7 m n ++ [VecShape m, MatShape m n, MatShape m n, MatShape m n,
      MatShape m n, MatShape m n])) (MatShape m n) :=
  add (Γ := ΓLN m n ++ (ssPrefix7 m n ++ [VecShape m, MatShape m n, MatShape m n, MatShape m n,
    MatShape m n, MatShape m n]))
    (s := MatShape m n) (a := idxScaled13 (m := m) (n := n)) (b := idxBetaB13 (m := m) (n := n))

/-- Full LayerNorm graph (as an explicit snoc chain). -/
def layerNormGraph {m n : Nat} (ε : ℝ) : Graph (ΓLN m n) (ssLayerNorm m n) :=
  .snoc (g13 (m := m) (n := n) ε) (nodeY (m := m) (n := n))

-- ---------------------------------------------------------------------------
-- Pointwise `GraphFDerivCorrectAt` for LayerNorm
-- ---------------------------------------------------------------------------

/--
Pointwise proof that `layerNormGraph` satisfies `GraphFDerivCorrectAt`.

The hypotheses `hVarEpsPos` and `hStdNe0` are explicit domain assumptions ensuring that `sqrt` and
`inv` are differentiable at the execution point.
-/
def layerNormGraphFderivCorrectAt
    {m n : Nat} (ε : ℝ) (xV : CtxVec (ΓLN m n))
    (hVarEpsPos :
      ∀ i : Fin (Shape.size (VecShape m)),
        0 < CtxVec.get (Γ := ΓLN m n ++ ssPrefix6 m n) (s := VecShape m) (idxVarEps (m := m) (n :=
          n))
          (Graph.evalVec (Γ := ΓLN m n) (ss := ssPrefix6 m n) (layerNormPrefix6 (m := m) (n := n) ε)
            xV) i)
    (hStdNe0 :
      ∀ i : Fin (Shape.size (VecShape m)),
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

/--
Pointwise end-to-end result: backprop equals `(fderiv eval)†` for `layerNormGraph`.

The hypotheses `hVarEpsPos` and `hStdNe0` are the explicit domain assumptions needed for
differentiability of `sqrt` (after clamp) and `inv` at the actual execution point.
-/
theorem backprop_eq_adjoint_fderiv_layerNorm_at
    {m n : Nat} (ε : ℝ)
    (xV : CtxVec (ΓLN m n))
    (seedV : CtxVec (ΓLN m n ++ ssLayerNorm m n))
    (hVarEpsPos :
      ∀ i : Fin (Shape.size (VecShape m)),
        0 < CtxVec.get (Γ := ΓLN m n ++ ssPrefix6 m n) (s := VecShape m) (idxVarEps (m := m) (n :=
          n))
          (Graph.evalVec (Γ := ΓLN m n) (ss := ssPrefix6 m n) (layerNormPrefix6 (m := m) (n := n) ε)
            xV) i)
    (hStdNe0 :
      ∀ i : Fin (Shape.size (VecShape m)),
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
    layerNormGraphFderivCorrectAt (m := m) (n := n) ε xV hVarEpsPos hStdNe0
  exact
    Graph.backpropVec_eq_adjoint_fderiv_at (Γ := ΓLN m n) (ss := ssLayerNorm m n)
      (g := layerNormGraph (m := m) (n := n) ε) xV seedV hg

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

/-- Saved tensors before the final LayerNorm output `y`. -/
abbrev ssBeforeY (m n : Nat) : List Shape :=
  ssPrefix7 m n ++
    [ VecShape m
    , MatShape m n
    , MatShape m n
    , MatShape m n
    , MatShape m n
    , MatShape m n
    ]

/-- Index of the final LayerNorm output in `ΓLN ++ ssLayerNorm`. -/
def idxY {m n : Nat} : Idx (ΓLN m n ++ ssLayerNorm m n) (MatShape m n) :=
  ⟨⟨16, by simp [ΓLN, ssLayerNorm]⟩,
    by simp [ΓLN, ssLayerNorm]⟩

/--
Linear map that packs arbitrary-context LayerNorm inputs into the canonical context
`[X, gamma, beta]`.
-/
def packInputsCLM {Γ : List Shape} {m n : Nat} (inputs : Inputs Γ m n) :
    CtxVec Γ →L[ℝ] CtxVec (ΓLN m n) := by
  let xCLM : CtxVec Γ →L[ℝ] Vec (Shape.size (MatShape m n)) :=
    CtxVec.getCLM (Γ := Γ) (s := MatShape m n) inputs.x
  let gammaCLM : CtxVec Γ →L[ℝ] Vec (Shape.size (VecShape n)) :=
    CtxVec.getCLM (Γ := Γ) (s := VecShape n) inputs.gamma
  let betaCLM : CtxVec Γ →L[ℝ] Vec (Shape.size (VecShape n)) :=
    CtxVec.getCLM (Γ := Γ) (s := VecShape n) inputs.beta
  let gbCLM :=
    (Graph.appendCLM (Shape.size (VecShape n)) (Shape.size (VecShape n))).comp
      (gammaCLM.prod betaCLM)
  let allCLM :=
    (Graph.appendCLM (Shape.size (MatShape m n))
      (Shape.size (VecShape n) + Shape.size (VecShape n))).comp
      (xCLM.prod gbCLM)
  let h :
      Shape.size (MatShape m n) + (Shape.size (VecShape n) + Shape.size (VecShape n))
        =
      ctxSize (ΓLN m n) := by
    simp [ctxSize, Shape.size]
  exact (Graph.castCLM (h := h)).comp allCLM

/-- Project the final LayerNorm output from the full canonical graph context. -/
def outputCLM {m n : Nat} :
    CtxVec (ΓLN m n ++ ssLayerNorm m n) →L[ℝ] Vec (Shape.size (MatShape m n)) :=
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
  let f : CtxVec Γ → Vec (Shape.size (MatShape m n)) :=
    fun xV =>
      outputCLM (m := m) (n := n)
        (Graph.evalVec (Γ := ΓLN m n) (ss := ssLayerNorm m n)
          (layerNormGraph (m := m) (n := n) ε) (pack xV))
  Node.ofVec (Γ := Γ) (τ := MatShape m n)
    (f := f)
    (jvp := fun xV dxV => (fderiv ℝ f xV) dxV)
    (vjp := fun xV δV => (fderiv ℝ f xV).adjoint δV)
    (correct_inner := by
      intro xV dxV δV
      simpa using
        (ContinuousLinearMap.adjoint_inner_right (A := fderiv ℝ f xV) (x := dxV) (y := δV)).symm)

/--
Pointwise derivative certificate for `wholeNode`.

The hypotheses are the same LayerNorm domain conditions as the detailed graph theorem, but evaluated
after packing the arbitrary context into `[X, gamma, beta]`.
-/
def wholeNodeFDerivCorrectAt {Γ : List Shape} {m n : Nat}
    (inputs : Inputs Γ m n) (ε : ℝ) (xV : CtxVec Γ)
    (hVarEpsPos :
      ∀ i : Fin (Shape.size (VecShape m)),
        0 < CtxVec.get (Γ := ΓLN m n ++ ssPrefix6 m n) (s := VecShape m)
          (idxVarEps (m := m) (n := n))
          (Graph.evalVec (Γ := ΓLN m n) (ss := ssPrefix6 m n)
            (layerNormPrefix6 (m := m) (n := n) ε)
            ((packInputsCLM (Γ := Γ) (m := m) (n := n) inputs) xV)) i)
    (hStdNe0 :
      ∀ i : Fin (Shape.size (VecShape m)),
        CtxVec.get (Γ := ΓLN m n ++ ssPrefix7 m n) (s := VecShape m)
          (idxStd (m := m) (n := n))
          (Graph.evalVec (Γ := ΓLN m n) (ss := ssPrefix7 m n)
            (layerNormPrefix7 (m := m) (n := n) ε)
            ((packInputsCLM (Γ := Γ) (m := m) (n := n) inputs) xV)) i ≠ 0) :
    NodeFDerivCorrectAt (wholeNode (Γ := Γ) (m := m) (n := n) inputs ε) xV := by
  classical
  let pack := packInputsCLM (Γ := Γ) (m := m) (n := n) inputs
  let g := layerNormGraph (m := m) (n := n) ε
  let out := outputCLM (m := m) (n := n)
  let f : CtxVec Γ → Vec (Shape.size (MatShape m n)) :=
    fun z =>
      out (Graph.evalVec (Γ := ΓLN m n) (ss := ssLayerNorm m n) g (pack z))
  have hgAt :
      GraphFDerivCorrectAt (Γ := ΓLN m n) (ss := ssLayerNorm m n) g (pack xV) :=
    layerNormGraphFderivCorrectAt (m := m) (n := n) ε (pack xV) hVarEpsPos hStdNe0
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
      simpa [hfEq] using hOut.fderiv
    rw [hFderiv]
    simpa [wholeNode, f, pack, g, out] using hOut
  · intro dxV
    simp [wholeNode, f, pack, g, out]

end LayerNorm

end

end Autograd
end Proofs
