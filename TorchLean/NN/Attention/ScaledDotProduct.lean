/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Tape.DGraph
public import TorchLean.NN.Tape.RealNode.Arithmetic
public import TorchLean.NN.Tape.RealNode.Arm
public import TorchLean.NN.Tape.RealNode.Matrix
public import TorchLean.NN.Op.Pointwise
public import TorchLean.NN.Op.Softmax

/-!
# ScaledDotProduct

A tape graph for **scaled dot-product attention** on `m×d` inputs `Q`, `K` and `V`, built from the
proven nodes `matrixTranspose` and `matmul`, the `Op.scale` and `Op.softmax` arms of
`Tape.LawfulNode.ofArm`, and a second `matmul`.
`scaledDotProductDGraph` bundles the graph with the `Tape.RealNode.FDerivCorrect` certificate of
each node. Over `ℝ` the general theorem `Tape.Graph.backpropVec_eq_fderiv` then identifies backprop
on this graph with the adjoint of its `fderiv`, and `ScaledDotProduct/FDeriv.lean` carries that
result over to `Attention.scaledDotProduct`.

## PyTorch correspondence / citations
- The graph follows the usual mathematical definition of scaled dot-product attention,
  `softmax(c * Q Kᵀ) V`. PyTorch builds the same computation from `torch.matmul` and
  `torch.softmax`, or provides it as a dedicated helper:
  https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
-/

@[expose] public section

namespace TorchLean

open scoped BigOperators

noncomputable section

namespace Attention


/-- Input context shapes: `[Q, K, V]`, each `m×d`. -/
abbrev qkvCtx (m d : Nat) : List Shape := [[m, d], [m, d], [m, d]]

/-- Context index of `Q` in `qkvCtx`. -/
def idxQ {m d : Nat} {ss : List Shape} : Idx (qkvCtx m d ++ ss) [m, d] :=
  ⟨⟨0, by simp [qkvCtx]⟩, by simp [qkvCtx]⟩

/-- Context index of `K` in `qkvCtx`. -/
def idxK {m d : Nat} {ss : List Shape} : Idx (qkvCtx m d ++ ss) [m, d] :=
  ⟨⟨1, by simp [qkvCtx]⟩, by simp [qkvCtx]⟩

/-- Context index of `V` in `qkvCtx`. -/
def idxV {m d : Nat} {ss : List Shape} : Idx (qkvCtx m d ++ ss) [m, d] :=
  ⟨⟨2, by simp [qkvCtx]⟩, by simp [qkvCtx]⟩

/-- Saved tensors of the attention graph: `Kᵀ`, logits, scaled logits, probabilities, output. -/
abbrev ssScaledDotProduct (m d : Nat) : List Shape :=
  [ .dim d (.dim m .scalar)
  , .dim m (.dim m .scalar)
  , .dim m (.dim m .scalar)
  , .dim m (.dim m .scalar)
  , .dim m (.dim d .scalar) ]

/-- Node 1: `Kᵀ`. -/
abbrev nodeKt (m d : Nat) : Tape.LawfulNode (α := ℝ) Unit (qkvCtx m d) (.dim d (.dim m .scalar)) :=
  Tape.RealNode.matrixTranspose (Γ := qkvCtx m d) (m := m) (n := d) (A := idxK (m := m) (d := d))

/-- Node 2: logits `Q Kᵀ`. -/
abbrev nodeLogits (m d : Nat) :
    Tape.LawfulNode (α := ℝ) Unit (qkvCtx m d ++ [.dim d (.dim m .scalar)])
      (.dim m (.dim m .scalar)) :=
  Tape.RealNode.matmul (Γ := qkvCtx m d ++ [.dim d (.dim m .scalar)]) (m := m) (n := d) (p := m)
    (A := idxQ (m := m) (d := d) (ss := [.dim d (.dim m .scalar)]))
    (B := Idx.last (Γ := qkvCtx m d) (ss := []) (τ := .dim d (.dim m .scalar)))

/-- Node 3: scaled logits `c * (Q Kᵀ)`. -/
abbrev nodeScaled (m d : Nat) (c : ℝ) :
    Tape.LawfulNode (α := ℝ) Unit (qkvCtx m d ++ [.dim d (.dim m .scalar), .dim m (.dim m .scalar)])
      (.dim m (.dim m .scalar)) :=
  Tape.LawfulNode.ofArm (Op.scale (s := .dim m (.dim m .scalar))) rfl
    (Γ := qkvCtx m d ++ [.dim d (.dim m .scalar), .dim m (.dim m .scalar)])
    (Idx.last (Γ := qkvCtx m d) (ss := [.dim d (.dim m .scalar)]) (τ := .dim m (.dim m .scalar)),
      Tensor.scalar c, ())
    (Arm.adj_of_cert₁ fun xs δ => Op.scale_backward_eq_adjoint xs.1 δ xs.2.1)

/-- Node 4: row-wise softmax probabilities. -/
abbrev nodeProbs (m d : Nat) :
    Tape.LawfulNode (α := ℝ) Unit
      (qkvCtx m d ++ [.dim d (.dim m .scalar), .dim m (.dim m .scalar), .dim m (.dim m .scalar)])
      (.dim m (.dim m .scalar)) :=
  Tape.LawfulNode.ofArm (Op.softmax (s := .dim m (.dim m .scalar)) ⟨1, Nat.one_lt_two⟩) rfl
    (Γ := qkvCtx m d ++ [.dim d (.dim m .scalar), .dim m (.dim m .scalar), .dim m (.dim m .scalar)])
    (Idx.last (Γ := qkvCtx m d) (ss := [.dim d (.dim m .scalar), .dim m (.dim m .scalar)])
        (τ := .dim m (.dim m .scalar)), ())
    (Arm.adj_of_cert₁ fun xs δ => Op.softmax_backward_eq_adjoint _ xs.1 δ)

/-- Node 5: output `probs * V`. -/
abbrev nodeOut (m d : Nat) :
    Tape.LawfulNode (α := ℝ) Unit (qkvCtx m d ++ [.dim d (.dim m .scalar), .dim m (.dim m .scalar),
      .dim m (.dim m .scalar), .dim m (.dim m .scalar)]) (.dim m (.dim d .scalar)) :=
  Tape.RealNode.matmul
    (Γ := qkvCtx m d ++ [.dim d (.dim m .scalar), .dim m (.dim m .scalar), .dim m (.dim m .scalar),
      .dim m (.dim m .scalar)])
    (m := m) (n := m) (p := d)
    (A := Idx.last (Γ := qkvCtx m d)
      (ss := [.dim d (.dim m .scalar), .dim m (.dim m .scalar), .dim m (.dim m .scalar)])
      (τ := .dim m (.dim m .scalar)))
    (B := idxV (m := m) (d := d)
      (ss := [.dim d (.dim m .scalar), .dim m (.dim m .scalar), .dim m (.dim m .scalar),
        .dim m (.dim m .scalar)]))

/-- Attention graph prefix through `Kᵀ`. -/
abbrev graphKt (m d : Nat) : Tape.Graph (α := ℝ) Unit (qkvCtx m d) [.dim d (.dim m .scalar)] :=
  Tape.Graph.snoc (ss := []) Tape.Graph.nil (nodeKt m d)

/-- Attention graph prefix through the logits. -/
abbrev graphLogits (m d : Nat) :
    Tape.Graph (α := ℝ) Unit (qkvCtx m d) [.dim d (.dim m .scalar), .dim m (.dim m .scalar)] :=
  Tape.Graph.snoc (ss := [.dim d (.dim m .scalar)]) (graphKt m d) (nodeLogits m d)

/-- Attention graph prefix through the scaled logits. -/
abbrev graphScaled (m d : Nat) (c : ℝ) :
    Tape.Graph (α := ℝ) Unit (qkvCtx m d)
      [.dim d (.dim m .scalar), .dim m (.dim m .scalar), .dim m (.dim m .scalar)] :=
  Tape.Graph.snoc (ss := [.dim d (.dim m .scalar), .dim m (.dim m .scalar)]) (graphLogits m d)
    (nodeScaled m d c)

/-- Attention graph prefix through the softmax probabilities. -/
abbrev graphProbs (m d : Nat) (c : ℝ) :
    Tape.Graph (α := ℝ) Unit (qkvCtx m d) [.dim d (.dim m .scalar), .dim m (.dim m .scalar),
      .dim m (.dim m .scalar), .dim m (.dim m .scalar)] :=
  Tape.Graph.snoc
    (ss := [.dim d (.dim m .scalar), .dim m (.dim m .scalar), .dim m (.dim m .scalar)])
    (graphScaled m d c) (nodeProbs m d)

/--
Scaled dot-product attention as a tape graph.

It maps `Q K V ↦ softmax(c * (Q * Kᵀ)) * V` and keeps each intermediate value on the tape for
backprop. Being a plain `Tape.Graph.snoc` chain, the graph lets `Tape.GraphData.evalVec` unfold
one node at a time when the tape is matched against the specification forward pass.
-/
def scaledDotProductGraph {m d : Nat} (c : ℝ) :
    Tape.Graph (α := ℝ) Unit (qkvCtx m d) (ssScaledDotProduct m d) :=
  Tape.Graph.snoc (ss := [.dim d (.dim m .scalar), .dim m (.dim m .scalar),
      .dim m (.dim m .scalar), .dim m (.dim m .scalar)])
    (graphProbs m d c) (nodeOut m d)

/--
Scaled dot-product attention as a proved-correct `DGraph`.

Every node of `scaledDotProductGraph` carries its `Tape.RealNode.FDerivCorrect` certificate.
-/
def scaledDotProductDGraph {m d : Nat} (c : ℝ) :
    Tape.DGraph (qkvCtx m d) (ssScaledDotProduct m d) :=
  ⟨scaledDotProductGraph c,
    ⟨⟨⟨⟨⟨PUnit.unit,
      Tape.RealNode.matrixTransposeFDeriv (Γ := qkvCtx m d) (m := m) (n := d)
        (A := idxK (m := m) (d := d))⟩,
      Tape.RealNode.matmulFDeriv (Γ := qkvCtx m d ++ [.dim d (.dim m .scalar)]) (m := m)
        (n := d) (p := m) (A := idxQ (m := m) (d := d) (ss := [.dim d (.dim m .scalar)]))
        (B := Idx.last (Γ := qkvCtx m d) (ss := []) (τ := .dim d (.dim m .scalar)))⟩,
      Tape.RealNode.FDerivCorrect.ofArm₁
        (Γ := qkvCtx m d ++ [.dim d (.dim m .scalar), .dim m (.dim m .scalar)])
        (s := .dim m (.dim m .scalar))
        (idx := Idx.last (Γ := qkvCtx m d) (ss := [.dim d (.dim m .scalar)])
          (τ := .dim m (.dim m .scalar)))
        (cert := fun x δ => Op.scale_backward_eq_adjoint x δ (Tensor.scalar c))⟩,
      Tape.RealNode.FDerivCorrect.ofArm₁
        (Γ := qkvCtx m d ++
          [.dim d (.dim m .scalar), .dim m (.dim m .scalar), .dim m (.dim m .scalar)])
        (idx := Idx.last (Γ := qkvCtx m d)
          (ss := [.dim d (.dim m .scalar), .dim m (.dim m .scalar)])
          (τ := .dim m (.dim m .scalar)))
        (cert := Op.softmax_backward_eq_adjoint (s := .dim m (.dim m .scalar))
          ⟨1, Nat.one_lt_two⟩)⟩,
      Tape.RealNode.matmulFDeriv
        (Γ := qkvCtx m d ++
          [.dim d (.dim m .scalar), .dim m (.dim m .scalar), .dim m (.dim m .scalar),
            .dim m (.dim m .scalar)])
        (m := m) (n := m) (p := d)
        (A := Idx.last (Γ := qkvCtx m d)
          (ss := [.dim d (.dim m .scalar), .dim m (.dim m .scalar), .dim m (.dim m .scalar)])
          (τ := .dim m (.dim m .scalar)))
        (B := idxV (m := m) (d := d)
          (ss := [.dim d (.dim m .scalar), .dim m (.dim m .scalar), .dim m (.dim m .scalar),
            .dim m (.dim m .scalar)]))⟩⟩

end Attention

end

end TorchLean
