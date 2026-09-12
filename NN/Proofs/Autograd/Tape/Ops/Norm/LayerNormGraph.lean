/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Tape.Nodes.Arithmetic
public import NN.Proofs.Autograd.Tape.Util.Idx
public import NN.Proofs.Autograd.Tape.Nodes.Matrix

/-!
# LayerNorm graph

The explicit SSA graph used by the LayerNorm proofs: a `seqLen × embedDim` input is normalized
across its last axis, the row statistics are broadcast back over each token, and the affine
parameters `gamma`/`beta` are broadcast over the sequence axis. The graph is written as an
explicit `snoc` chain so that every prefix has a name; the calculus proofs in
`NN.Proofs.Autograd.Tape.Ops.Norm.LayerNorm` and the evaluation lemmas in
`NN.Proofs.Autograd.Tape.Ops.Norm.LayerNormEval` refer to these prefixes directly.

The nodes are the shape-generic tape nodes: row mean, row and column broadcast, elementwise
arithmetic, the clamped square root `sqrt (max x 0)`, and inversion.
-/

@[expose] public section


namespace Proofs
namespace Autograd

open Spec TorchLean

open scoped BigOperators

noncomputable section

namespace LayerNorm

open TapeNodes

/-- Matrix shape `m×n`. -/
abbrev MatShape (m n : Nat) : Shape := .dim m (.dim n .scalar)
/-- Rank-one tensor shape `k`. -/
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
  Idx.last (Γ := ΓLN m n) (ss := []) (τ := VecShape m)

/-- Broadcast `mean` back to `m×n` (row-wise). -/
def nodeMeanB {m n : Nat} : Node (ΓLN m n ++ [VecShape m]) (MatShape m n) :=
  broadcastRow (Γ := ΓLN m n ++ [VecShape m]) (m := m) (n := n) (idx := idxMean (m := m) (n := n))

/-- Graph prefix producing `[mean, mean_b]`. -/
def g2 {m n : Nat} : Graph (ΓLN m n) [VecShape m, MatShape m n] :=
  .snoc (g1 (m := m) (n := n)) (nodeMeanB (m := m) (n := n))

/-- Index of `mean_b` in `ΓLN ++ [mean, mean_b]`. -/
def idxMeanB {m n : Nat} : Idx (ΓLN m n ++ [VecShape m, MatShape m n]) (MatShape m n) :=
  Idx.last (Γ := ΓLN m n) (ss := [VecShape m]) (τ := MatShape m n)

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
  Idx.last (Γ := ΓLN m n) (ss := [VecShape m, MatShape m n]) (τ := MatShape m n)

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
  Idx.last (Γ := ΓLN m n) (ss := [VecShape m, MatShape m n, MatShape m n]) (τ := MatShape m n)

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
  Idx.last (Γ := ΓLN m n) (ss := [VecShape m, MatShape m n, MatShape m n, MatShape m n]) (τ :=
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
  Idx.last (Γ := ΓLN m n)
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
  Idx.last (Γ := ΓLN m n) (ss := ssPrefix6 m n) (τ := VecShape m)

/-- Inverse standard deviation: `inv_std := 1/std`. -/
def nodeInvStd {m n : Nat} : Node (ΓLN m n ++ ssPrefix7 m n) (VecShape m) :=
  inv (Γ := ΓLN m n ++ ssPrefix7 m n) (s := VecShape m) (idxStd (m := m) (n := n))

/-- Graph prefix adding `invStd`. -/
def g8 {m n : Nat} (ε : ℝ) : Graph (ΓLN m n) (ssPrefix7 m n ++ [VecShape m]) :=
  .snoc (layerNormPrefix7 (m := m) (n := n) ε) (nodeInvStd (m := m) (n := n))

/-- Index of `invStd` in the extended context. -/
def idxInvStd {m n : Nat} : Idx (ΓLN m n ++ (ssPrefix7 m n ++ [VecShape m])) (VecShape m) :=
  Idx.last (Γ := ΓLN m n) (ss := ssPrefix7 m n) (τ := VecShape m)

/-- Broadcast `invStd` back to `m×n` (row-wise). -/
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
  Idx.last (Γ := ΓLN m n) (ss := ssPrefix7 m n ++ [VecShape m]) (τ := MatShape m n)

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
  Proofs.Idx.weaken (Γ := ΓLN m n ++ (ssPrefix7 m n ++ [VecShape m, MatShape m n,
    MatShape m n]))
    (Idx.last (Γ := ΓLN m n) (ss := ssPrefix7 m n ++ [VecShape m, MatShape m n])
      (τ := MatShape m n))
    (rest := [MatShape m n])

/-- Index of `gamma_b` at stage `g11`. -/
def idxGammaB11 {m n : Nat} :
    Idx (ΓLN m n ++ (ssPrefix7 m n ++ [VecShape m, MatShape m n, MatShape m n, MatShape m n]))
      (MatShape m n) :=
  Idx.last (Γ := ΓLN m n) (ss := ssPrefix7 m n ++ [VecShape m, MatShape m n, MatShape m n]) (τ :=
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
  Proofs.Idx.weaken (Γ := ΓLN m n ++ (ssPrefix7 m n ++ [VecShape m, MatShape m n,
    MatShape m n, MatShape m n, MatShape m n]))
    (Idx.last (Γ := ΓLN m n) (ss := ssPrefix7 m n ++ [VecShape m, MatShape m n, MatShape m n,
      MatShape m n]) (τ := MatShape m n))
    (rest := [MatShape m n])

/-- Index of `beta_b` at stage `g13`. -/
def idxBetaB13 {m n : Nat} :
    Idx (ΓLN m n ++ (ssPrefix7 m n ++ [VecShape m, MatShape m n, MatShape m n, MatShape m n,
      MatShape m n, MatShape m n])) (MatShape m n) :=
  Idx.last (Γ := ΓLN m n)
    (ss := ssPrefix7 m n ++ [VecShape m, MatShape m n, MatShape m n, MatShape m n, MatShape m n])
    (τ := MatShape m n)

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

/-- Index of the final LayerNorm output in `ΓLN ++ ssLayerNorm`. -/
def idxY {m n : Nat} : Idx (ΓLN m n ++ ssLayerNorm m n) (MatShape m n) :=
  ⟨⟨16, by simp [ΓLN, ssLayerNorm]⟩,
    by simp [ΓLN, ssLayerNorm]⟩

end LayerNorm

end

end Autograd
end Proofs
