/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Tape.RealNode.Arm
public import TorchLean.NN.Tape.RealNode.Matrix
public import TorchLean.NN.Op.Pointwise

/-!
# LayerNorm graph

The SSA graph used by the LayerNorm proofs: a `seqLen × embedDim` input is normalized across its
last axis, the row statistics are broadcast back over each token, and the affine parameters
`gamma`/`beta` are broadcast over the sequence axis. The graph is written as a `snoc` chain whose
prefixes are all named definitions, so the calculus proofs in
`TorchLean.NN.Norm.LayerNorm` and the evaluation lemmas in
`TorchLean.NN.Norm.LayerNorm.Eval` can refer to each prefix.

The row mean, the row and column broadcasts and the epsilon shift are shape-generic tape nodes.
The subtraction, the products, the output sum, the clamped square root `sqrt (max x 0)` and the
reciprocal are operator arms, made into nodes by `Tape.LawfulNode.ofArm` with their certificates.
-/

@[expose] public section

namespace TorchLean

open scoped BigOperators

noncomputable section

namespace LayerNorm

open Tape.RealNode

/-- Input context shapes: `[X, gamma, beta]` for layer norm over the last axis. -/
abbrev ctx (m n : Nat) : List Shape := [[m, n], [n], [n]]

/-- LayerNorm intermediates through the epsilon-shifted variance. -/
abbrev ssVarEps (m n : Nat) : List Shape :=
  [ [m]    -- mean
  , [m, n] -- mean_b
  , [m, n] -- centered
  , [m, n] -- centered_sq
  , [m]    -- var
  , [m]    -- var_eps
  ]

/-- Prefix intermediates up to `std` (adds one more vector). -/
abbrev ssStd (m n : Nat) : List Shape := ssVarEps m n ++ [[m]] -- std

/-- Full list of intermediates of `graph`. -/
abbrev ss (m n : Nat) : List Shape :=
  ssStd m n ++
    [ [m]    -- inv_std
    , [m, n] -- inv_std_b
    , [m, n] -- normalized
    , [m, n] -- gamma_b
    , [m, n] -- scaled
    , [m, n] -- beta_b
    , [m, n] -- y
    ]

/-- Index of the input matrix `X` in the base LayerNorm context `ctx m n ++ ss`. -/
def idxX {m n : Nat} {ss : List Shape} : Idx (ctx m n ++ ss) [m, n] :=
  ⟨⟨0, by simp [ctx]⟩, by simp [ctx]⟩

/-- Index of the scale vector `gamma` in the base LayerNorm context `ctx m n ++ ss`. -/
def idxGamma {m n : Nat} {ss : List Shape} : Idx (ctx m n ++ ss) [n] :=
  ⟨⟨1, by simp [ctx]⟩, by simp [ctx]⟩

/-- Index of the shift vector `beta` in the base LayerNorm context `ctx m n ++ ss`. -/
def idxBeta {m n : Nat} {ss : List Shape} : Idx (ctx m n ++ ss) [n] :=
  ⟨⟨2, by simp [ctx]⟩, by simp [ctx]⟩

-- ---------------------------------------------------------------------------
-- LayerNorm graph: a `snoc` chain of named prefixes, with no `let` to block unfolding
-- ---------------------------------------------------------------------------

-- Prefix nodes (mean/variance + epsilon)

/-- Mean over the last axis: `mean : ℝ^{m×n} → ℝ^{m}`. -/
def nodeMean {m n : Nat} : Tape.LawfulNode (α := ℝ) Unit (ctx m n) [m] :=
  rowMean (Γ := ctx m n) (m := m) (n := n) (idx := idxX (m := m) (n := n) (ss := []))

/-- Graph prefix producing `[mean]`. -/
def graphMean {m n : Nat} : Tape.Graph (α := ℝ) Unit (ctx m n) [[m]] :=
  .snoc (.nil) (nodeMean (m := m) (n := n))

/-- Index of `mean` in the extended context `ctx ++ [mean]`. -/
def idxMean {m n : Nat} : Idx (ctx m n ++ [[m]]) [m] :=
  Idx.last (Γ := ctx m n) (ss := []) (τ := [m])

/-- Broadcast `mean` back to `m×n` (row-wise). -/
def nodeMeanB {m n : Nat} : Tape.LawfulNode (α := ℝ) Unit (ctx m n ++ [[m]]) [m, n] :=
  broadcastRow (Γ := ctx m n ++ [[m]]) (m := m) (n := n) (idx := idxMean (m := m) (n := n))

/-- Graph prefix producing `[mean, mean_b]`. -/
def graphMeanBroadcast {m n : Nat} :
    Tape.Graph (α := ℝ) Unit (ctx m n) [[m], [m, n]] :=
  .snoc (graphMean (m := m) (n := n)) (nodeMeanB (m := m) (n := n))

/-- Index of `mean_b` in `ctx ++ [mean, mean_b]`. -/
def idxMeanB {m n : Nat} : Idx (ctx m n ++ [[m], [m, n]]) [m, n] :=
  Idx.last (Γ := ctx m n) (ss := [[m]]) (τ := [m, n])

/-- Center: `centered := X - mean_b`. -/
def nodeCentered {m n : Nat} :
    Tape.LawfulNode (α := ℝ) Unit (ctx m n ++ [[m], [m, n]]) [m, n] :=
  Tape.LawfulNode.ofArm (Op.sub (s := [m, n])) rfl (Γ := ctx m n ++ [[m], [m, n]])
    (idxX (m := m) (n := n) (ss := [[m], [m, n]]), idxMeanB (m := m) (n := n), ())
    (Arm.adj_of_cert₂ fun xs δ => Op.sub_prod_backward_eq_adjoint xs.1 xs.2.1 δ)

/-- Graph prefix producing `[mean, mean_b, centered]`. -/
def graphCentered {m n : Nat} :
    Tape.Graph (α := ℝ) Unit (ctx m n) [[m], [m, n], [m, n]] :=
  .snoc (graphMeanBroadcast (m := m) (n := n)) (nodeCentered (m := m) (n := n))

/-- Index of `centered` in `ctx ++ [mean, mean_b, centered]`. -/
def idxCentered {m n : Nat} :
    Idx (ctx m n ++ [[m], [m, n], [m, n]]) [m, n] :=
  Idx.last (Γ := ctx m n) (ss := [[m], [m, n]]) (τ := [m, n])

/-- Square `centered`: `centered_sq := centered ⊙ centered`. -/
def nodeCenteredSq {m n : Nat} :
    Tape.LawfulNode (α := ℝ) Unit (ctx m n ++ [[m], [m, n], [m, n]])
      [m, n] :=
  Tape.LawfulNode.ofArm (Op.mul (s := [m, n])) rfl (Γ := ctx m n ++ [[m], [m, n], [m, n]])
    (idxCentered (m := m) (n := n), idxCentered (m := m) (n := n), ())
    (Arm.adj_of_cert₂ fun xs δ => Op.mul_prod_backward_eq_adjoint xs.1 xs.2.1 δ)

/-- Graph prefix producing `[mean, mean_b, centered, centered_sq]`. -/
def graphCenteredSq {m n : Nat} :
    Tape.Graph (α := ℝ) Unit (ctx m n) [[m], [m, n], [m, n], [m, n]] :=
  .snoc (graphCentered (m := m) (n := n)) (nodeCenteredSq (m := m) (n := n))

/-- Index of `centered_sq` in the extended context. -/
def idxCenteredSq {m n : Nat} :
    Idx (ctx m n ++ [[m], [m, n], [m, n], [m, n]]) [m, n] :=
  Idx.last (Γ := ctx m n) (ss := [[m], [m, n], [m, n]]) (τ := [m, n])

/-- Variance per row: `var := mean(centered_sq)` producing a length-`m` vector. -/
def nodeVar {m n : Nat} :
    Tape.LawfulNode (α := ℝ) Unit (ctx m n ++ [[m], [m, n], [m, n], [m, n]])
      [m] :=
  rowMean (Γ := ctx m n ++ [[m], [m, n], [m, n], [m, n]])
    (m := m) (n := n) (idx := idxCenteredSq (m := m) (n := n))

/-- Graph prefix producing `[mean, mean_b, centered, centered_sq, var]`. -/
def graphVar {m n : Nat} :
    Tape.Graph (α := ℝ) Unit (ctx m n)
      [[m], [m, n], [m, n], [m, n], [m]] :=
  .snoc (graphCenteredSq (m := m) (n := n)) (nodeVar (m := m) (n := n))

/-- Index of `var` in the extended context. -/
def idxVar {m n : Nat} :
    Idx (ctx m n ++ [[m], [m, n], [m, n], [m, n], [m]])
      [m] :=
  Idx.last (Γ := ctx m n) (ss := [[m], [m, n], [m, n], [m, n]])
    (τ := [m])

/-- Add epsilon: `var_eps := var + ε`. -/
def nodeVarEps {m n : Nat} (ε : ℝ) :
    Tape.LawfulNode (α := ℝ) Unit
      (ctx m n ++ [[m], [m, n], [m, n], [m, n], [m]])
      [m] :=
  elemwise (Γ := ctx m n ++ [[m], [m, n], [m, n], [m, n], [m]])
    (s := [m]) (idxVar (m := m) (n := n)) (fun z => z + ε) (fun _ => 1)

/-- Graph prefix through the epsilon-shifted variance (`ssVarEps`). -/
def graphVarEps {m n : Nat} (ε : ℝ) : Tape.Graph (α := ℝ) Unit (ctx m n) (ssVarEps m n) :=
  .snoc (graphVar (m := m) (n := n)) (nodeVarEps (m := m) (n := n) ε)

/-- Index of `var_eps` in `ctx ++ ssVarEps`. -/
def idxVarEps {m n : Nat} : Idx (ctx m n ++ ssVarEps m n) [m] :=
  Idx.last (Γ := ctx m n)
    (ss := [[m], [m, n], [m, n], [m, n], [m]])
    (τ := [m])

/--
Standard deviation: `std := sqrt (max var_eps 0)`, the `clampNegative` square root arm.

The development becomes pointwise at this node, because its certificate
`Op.sqrt_clampNegative_backward_eq_adjoint` needs every entry of `var_eps` to be positive at the
evaluation point.
-/
def nodeStd {m n : Nat} : Tape.LawfulNode (α := ℝ) Unit (ctx m n ++ ssVarEps m n) [m] :=
  Tape.LawfulNode.ofArm (Op.sqrt (s := [m]) .clampNegative) rfl
    (Γ := ctx m n ++ ssVarEps m n) (idxVarEps (m := m) (n := n), ())
    (Arm.adj_of_jvp₁ rfl fun xs dxs δ => Op.sqrt_clampNegative_jvp_adjoint xs.1 dxs.1 δ)

/-- Graph prefix computing `ssStd` (adds `std`). -/
def graphStd {m n : Nat} (ε : ℝ) : Tape.Graph (α := ℝ) Unit (ctx m n) (ssStd m n) :=
  .snoc (graphVarEps (m := m) (n := n) ε) (nodeStd (m := m) (n := n))

-- Remaining nodes (normalize, scale, shift)

/-- Index of `std` in `ctx ++ ssStd`. -/
def idxStd {m n : Nat} : Idx (ctx m n ++ ssStd m n) [m] :=
  Idx.last (Γ := ctx m n) (ss := ssVarEps m n) (τ := [m])

/-- Inverse standard deviation: `inv_std := 1/std`, the reciprocal arm. -/
def nodeInvStd {m n : Nat} : Tape.LawfulNode (α := ℝ) Unit (ctx m n ++ ssStd m n) [m] :=
  Tape.LawfulNode.ofArm (Op.reciprocal (s := [m])) rfl (Γ := ctx m n ++ ssStd m n)
    (idxStd (m := m) (n := n), ())
    (Arm.adj_of_jvp₁ rfl fun xs dxs δ => Op.reciprocal_jvp_adjoint xs.1 dxs.1 δ)

/-- Graph prefix adding `invStd`. -/
def graphInvStd {m n : Nat} (ε : ℝ) :
    Tape.Graph (α := ℝ) Unit (ctx m n) (ssStd m n ++ [[m]]) :=
  .snoc (graphStd (m := m) (n := n) ε) (nodeInvStd (m := m) (n := n))

/-- Index of `invStd` in the extended context. -/
def idxInvStd {m n : Nat} : Idx (ctx m n ++ (ssStd m n ++ [[m]])) [m] :=
  Idx.last (Γ := ctx m n) (ss := ssStd m n) (τ := [m])

/-- Broadcast `invStd` back to `m×n` (row-wise). -/
def nodeInvStdB {m n : Nat} :
    Tape.LawfulNode (α := ℝ) Unit (ctx m n ++ (ssStd m n ++ [[m]])) [m, n] :=
  broadcastRow (Γ := ctx m n ++ (ssStd m n ++ [[m]]))
    (m := m) (n := n) (idx := idxInvStd (m := m) (n := n))

/-- Graph prefix adding `inv_std_b`. -/
def graphInvStdBroadcast {m n : Nat} (ε : ℝ) :
    Tape.Graph (α := ℝ) Unit (ctx m n) (ssStd m n ++ [[m], [m, n]]) :=
  .snoc (graphInvStd (m := m) (n := n) ε) (nodeInvStdB (m := m) (n := n))

/-- Index of `centered` in the stage-`graphInvStdBroadcast` context. -/
def idxCenteredForNorm {m n : Nat} :
    Idx (ctx m n ++ (ssStd m n ++ [[m], [m, n]])) [m, n] :=
  Idx.weaken (Γ := ctx m n ++ [[m], [m, n], [m, n]])
    (idxCentered (m := m) (n := n))
    (rest := [[m, n], [m], [m], [m], [m], [m, n]])

/-- Index of `inv_std_b` in the stage-`graphInvStdBroadcast` context. -/
def idxInvStdBroadcast {m n : Nat} :
    Idx (ctx m n ++ (ssStd m n ++ [[m], [m, n]])) [m, n] :=
  Idx.last (Γ := ctx m n) (ss := ssStd m n ++ [[m]]) (τ := [m, n])

/-- Node computing `normalized := centered ⊙ inv_std_b`. -/
def nodeNorm {m n : Nat} :
    Tape.LawfulNode (α := ℝ) Unit (ctx m n ++ (ssStd m n ++ [[m], [m, n]]))
      [m, n] :=
  Tape.LawfulNode.ofArm (Op.mul (s := [m, n])) rfl (Γ := ctx m n ++ (ssStd m n ++ [[m], [m, n]]))
    (idxCenteredForNorm (m := m) (n := n), idxInvStdBroadcast (m := m) (n := n), ())
    (Arm.adj_of_cert₂ fun xs δ => Op.mul_prod_backward_eq_adjoint xs.1 xs.2.1 δ)

/-- Graph prefix producing `normalized := centered ⊙ inv_std_b`. -/
def graphNormalized {m n : Nat} (ε : ℝ) :
    Tape.Graph (α := ℝ) Unit (ctx m n) (ssStd m n ++ [[m], [m, n], [m, n]]) :=
  .snoc (graphInvStdBroadcast (m := m) (n := n) ε) (nodeNorm (m := m) (n := n))

/-- Broadcast `gamma` to `m×n` (column-wise). -/
def nodeGammaB {m n : Nat} :
    Tape.LawfulNode (α := ℝ) Unit (ctx m n ++ (ssStd m n ++ [[m], [m, n], [m, n]]))
      [m, n] :=
  broadcastCol
    (Γ := ctx m n ++ (ssStd m n ++ [[m], [m, n], [m, n]]))
    (m := m) (n := n)
    (idx := idxGamma (m := m) (n := n)
      (ss := ssStd m n ++ [[m], [m, n], [m, n]]))

/-- Graph prefix adding `gamma_b`. -/
def graphGammaBroadcast {m n : Nat} (ε : ℝ) :
    Tape.Graph (α := ℝ) Unit (ctx m n)
      (ssStd m n ++ [[m], [m, n], [m, n], [m, n]]) :=
  .snoc (graphNormalized (m := m) (n := n) ε) (nodeGammaB (m := m) (n := n))

/-- Index of `normalized` in the context at stage `graphGammaBroadcast`. -/
def idxNormForScale {m n : Nat} :
    Idx (ctx m n ++ (ssStd m n ++ [[m], [m, n], [m, n], [m, n]]))
      [m, n] :=
  Idx.weaken (Γ := ctx m n ++ (ssStd m n ++ [[m], [m, n], [m, n]]))
    (Idx.last (Γ := ctx m n) (ss := ssStd m n ++ [[m], [m, n]])
      (τ := [m, n]))
    (rest := [[m, n]])

/-- Index of `gamma_b` at stage `graphGammaBroadcast`. -/
def idxGammaBroadcast {m n : Nat} :
    Idx (ctx m n ++ (ssStd m n ++ [[m], [m, n], [m, n], [m, n]]))
      [m, n] :=
  Idx.last (Γ := ctx m n) (ss := ssStd m n ++ [[m], [m, n], [m, n]])
    (τ := [m, n])

/-- Scale: `scaled := normalized ⊙ gamma_b`. -/
def nodeScaled {m n : Nat} :
    Tape.LawfulNode (α := ℝ) Unit
      (ctx m n ++ (ssStd m n ++ [[m], [m, n], [m, n], [m, n]]))
      [m, n] :=
  Tape.LawfulNode.ofArm (Op.mul (s := [m, n])) rfl
    (Γ := ctx m n ++ (ssStd m n ++ [[m], [m, n], [m, n], [m, n]]))
    (idxNormForScale (m := m) (n := n), idxGammaBroadcast (m := m) (n := n), ())
    (Arm.adj_of_cert₂ fun xs δ => Op.mul_prod_backward_eq_adjoint xs.1 xs.2.1 δ)

/-- Graph prefix adding `scaled`. -/
def graphScaled {m n : Nat} (ε : ℝ) :
    Tape.Graph (α := ℝ) Unit (ctx m n)
      (ssStd m n ++ [[m], [m, n], [m, n], [m, n], [m, n]]) :=
  .snoc (graphGammaBroadcast (m := m) (n := n) ε) (nodeScaled (m := m) (n := n))

/-- Broadcast `beta` to `m×n` (column-wise). -/
def nodeBetaB {m n : Nat} :
    Tape.LawfulNode (α := ℝ) Unit (ctx m n ++ (ssStd m n ++ [[m], [m, n], [m, n], [m, n], [m, n]]))
      [m, n] :=
  broadcastCol
    (Γ := ctx m n ++ (ssStd m n ++ [[m], [m, n], [m, n], [m, n], [m, n]]))
    (m := m) (n := n)
    (idx := idxBeta (m := m) (n := n)
      (ss := ssStd m n ++ [[m], [m, n], [m, n], [m, n], [m, n]]))

/-- Graph prefix adding `beta_b`. -/
def graphBetaBroadcast {m n : Nat} (ε : ℝ) :
    Tape.Graph (α := ℝ) Unit (ctx m n)
      (ssStd m n ++ [[m], [m, n], [m, n], [m, n], [m, n], [m, n]]) :=
  .snoc (graphScaled (m := m) (n := n) ε) (nodeBetaB (m := m) (n := n))

/-- Index of `scaled` at stage `graphBetaBroadcast`. -/
def idxScaledForOutput {m n : Nat} :
    Idx (ctx m n ++ (ssStd m n ++ [[m], [m, n], [m, n], [m, n], [m, n], [m, n]])) [m, n] :=
  Idx.weaken (Γ := ctx m n ++ (ssStd m n ++ [[m], [m, n], [m, n], [m, n], [m, n]]))
    (Idx.last (Γ := ctx m n) (ss := ssStd m n ++ [[m], [m, n], [m, n], [m, n]]) (τ := [m, n]))
    (rest := [[m, n]])

/-- Index of `beta_b` at stage `graphBetaBroadcast`. -/
def idxBetaBroadcast {m n : Nat} :
    Idx (ctx m n ++ (ssStd m n ++ [[m], [m, n], [m, n], [m, n], [m, n], [m, n]])) [m, n] :=
  Idx.last (Γ := ctx m n)
    (ss := ssStd m n ++ [[m], [m, n], [m, n], [m, n], [m, n]])
    (τ := [m, n])

/-- Output: `y := scaled + beta_b`. -/
def nodeY {m n : Nat} :
    Tape.LawfulNode (α := ℝ) Unit
      (ctx m n ++ (ssStd m n ++ [[m], [m, n], [m, n], [m, n], [m, n], [m, n]])) [m, n] :=
  Tape.LawfulNode.ofArm (Op.add (s := [m, n])) rfl
    (Γ := ctx m n ++ (ssStd m n ++ [[m], [m, n], [m, n], [m, n], [m, n], [m, n]]))
    (idxScaledForOutput (m := m) (n := n), idxBetaBroadcast (m := m) (n := n), ())
    (Arm.adj_of_cert₂ fun xs δ => Op.add_prod_backward_eq_adjoint xs.1 xs.2.1 δ)

/-- Full LayerNorm graph, the `snoc` chain that ends in the output node `nodeY`. -/
def graph {m n : Nat} (ε : ℝ) : Tape.Graph (α := ℝ) Unit (ctx m n) (ss m n) :=
  .snoc (graphBetaBroadcast (m := m) (n := n) ε) (nodeY (m := m) (n := n))

/-- Index of the final LayerNorm output in `ctx ++ LayerNorm.ss`. -/
def idxY {m n : Nat} : Idx (ctx m n ++ LayerNorm.ss m n) [m, n] :=
  Idx.last (Γ := ctx m n)
    (ss := ssStd m n ++ [[m], [m, n], [m, n], [m, n], [m, n], [m, n]])
    (τ := [m, n])

end LayerNorm

end

end TorchLean
