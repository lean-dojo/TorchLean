/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Tape.Nodes.GraphComposition
public import NN.Proofs.Autograd.Tape.Util.Idx
public import NN.Proofs.Autograd.Tape.Nodes.Arithmetic
public import NN.Proofs.Autograd.Tape.Nodes.Matrix
public import NN.Proofs.Autograd.Tape.Nodes.Softmax

/-!
# ScaledDotProduct

End-to-end `fderiv`/backprop correctness for a **scaled dot-product attention** graph,
built out of the proven tape nodes (`matmul`, `matrixTranspose`, `scale`, `softmaxLast`).

This is spec-level over `ℝ`. It is a corollary of the general graph theorem once each node
used by the graph has a `NodeFDerivCorrect` instance.

## PyTorch correspondence / citations
- This file matches the usual mathematical definition of scaled dot-product attention:
  `softmax(c * Q Kᵀ) V`. In PyTorch this corresponds to building the same computation with
  `torch.matmul` + `torch.softmax`, or using the dedicated helper:
  https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
-/

@[expose] public section


namespace Proofs
namespace Autograd

open Spec TorchLean

open scoped BigOperators

noncomputable section

namespace Attention

open TapeNodes
open DGraph

/-- Matrix shape for `m×d` Q/K/V inputs. -/
abbrev QKVShape (m d : Nat) : Shape := Shape.dim m (Shape.dim d Shape.scalar)

/-- Input context shapes: `[Q, K, V]`, each `m×d`. -/
abbrev ΓQKV (m d : Nat) : List Shape := [QKVShape m d, QKVShape m d, QKVShape m d]

/-- Context index of `Q` in `ΓQKV`. -/
def idxQ {m d : Nat} {ss : List Shape} : Idx (ΓQKV m d ++ ss) (QKVShape m d) :=
  ⟨⟨0, by simp [ΓQKV]⟩, by simp [ΓQKV]⟩

/-- Context index of `K` in `ΓQKV`. -/
def idxK {m d : Nat} {ss : List Shape} : Idx (ΓQKV m d ++ ss) (QKVShape m d) :=
  ⟨⟨1, by simp [ΓQKV]⟩, by simp [ΓQKV]⟩

/-- Context index of `V` in `ΓQKV`. -/
def idxV {m d : Nat} {ss : List Shape} : Idx (ΓQKV m d ++ ss) (QKVShape m d) :=
  ⟨⟨2, by simp [ΓQKV]⟩, by simp [ΓQKV]⟩

/-- Saved tensors of the attention graph: `Kᵀ`, logits, scaled logits, probabilities, output. -/
abbrev ssScaledDotProduct (m d : Nat) : List Shape :=
  [ .dim d (.dim m .scalar)
  , .dim m (.dim m .scalar)
  , .dim m (.dim m .scalar)
  , .dim m (.dim m .scalar)
  , .dim m (.dim d .scalar) ]

/-- Node 1: `Kᵀ`. -/
abbrev nodeKt (m d : Nat) : Node (ΓQKV m d) (.dim d (.dim m .scalar)) :=
  TapeNodes.matrixTranspose (Γ := ΓQKV m d) (m := m) (n := d) (A := idxK (m := m) (d := d))

/-- Node 2: logits `Q Kᵀ`. -/
abbrev nodeLogits (m d : Nat) :
    Node (ΓQKV m d ++ [.dim d (.dim m .scalar)]) (.dim m (.dim m .scalar)) :=
  TapeNodes.matmul (Γ := ΓQKV m d ++ [.dim d (.dim m .scalar)]) (m := m) (n := d) (p := m)
    (A := idxQ (m := m) (d := d) (ss := [.dim d (.dim m .scalar)]))
    (B := Idx.last (Γ := ΓQKV m d) (ss := []) (τ := .dim d (.dim m .scalar)))

/-- Node 3: scaled logits `c * (Q Kᵀ)`. -/
abbrev nodeScaled (m d : Nat) (c : ℝ) :
    Node (ΓQKV m d ++ [.dim d (.dim m .scalar), .dim m (.dim m .scalar)])
      (.dim m (.dim m .scalar)) :=
  TapeNodes.scale (Γ := ΓQKV m d ++ [.dim d (.dim m .scalar), .dim m (.dim m .scalar)])
    (s := .dim m (.dim m .scalar))
    (idx := Idx.last (Γ := ΓQKV m d) (ss := [.dim d (.dim m .scalar)])
      (τ := .dim m (.dim m .scalar)))
    c

/-- Node 4: row-wise softmax probabilities. -/
abbrev nodeProbs (m d : Nat) :
    Node (ΓQKV m d ++ [.dim d (.dim m .scalar), .dim m (.dim m .scalar), .dim m (.dim m .scalar)])
      (.dim m (.dim m .scalar)) :=
  TapeNodes.softmaxLast
    (Γ := ΓQKV m d ++ [.dim d (.dim m .scalar), .dim m (.dim m .scalar), .dim m (.dim m .scalar)])
    (m := m) (n := m)
    (idx := Idx.last (Γ := ΓQKV m d) (ss := [.dim d (.dim m .scalar), .dim m (.dim m .scalar)])
      (τ := .dim m (.dim m .scalar)))

/-- Node 5: output `probs * V`. -/
abbrev nodeOut (m d : Nat) :
    Node (ΓQKV m d ++ [.dim d (.dim m .scalar), .dim m (.dim m .scalar), .dim m (.dim m .scalar),
      .dim m (.dim m .scalar)]) (.dim m (.dim d .scalar)) :=
  TapeNodes.matmul
    (Γ := ΓQKV m d ++ [.dim d (.dim m .scalar), .dim m (.dim m .scalar), .dim m (.dim m .scalar),
      .dim m (.dim m .scalar)])
    (m := m) (n := m) (p := d)
    (A := Idx.last (Γ := ΓQKV m d)
      (ss := [.dim d (.dim m .scalar), .dim m (.dim m .scalar), .dim m (.dim m .scalar)])
      (τ := .dim m (.dim m .scalar)))
    (B := idxV (m := m) (d := d)
      (ss := [.dim d (.dim m .scalar), .dim m (.dim m .scalar), .dim m (.dim m .scalar),
        .dim m (.dim m .scalar)]))

/-- Attention graph prefix through `Kᵀ`. -/
abbrev graphKt (m d : Nat) : Graph (ΓQKV m d) [.dim d (.dim m .scalar)] :=
  Graph.snoc (ss := []) Graph.nil (nodeKt m d)

/-- Attention graph prefix through the logits. -/
abbrev graphLogits (m d : Nat) :
    Graph (ΓQKV m d) [.dim d (.dim m .scalar), .dim m (.dim m .scalar)] :=
  Graph.snoc (ss := [.dim d (.dim m .scalar)]) (graphKt m d) (nodeLogits m d)

/-- Attention graph prefix through the scaled logits. -/
abbrev graphScaled (m d : Nat) (c : ℝ) :
    Graph (ΓQKV m d) [.dim d (.dim m .scalar), .dim m (.dim m .scalar), .dim m (.dim m .scalar)] :=
  Graph.snoc (ss := [.dim d (.dim m .scalar), .dim m (.dim m .scalar)]) (graphLogits m d)
    (nodeScaled m d c)

/-- Attention graph prefix through the softmax probabilities. -/
abbrev graphProbs (m d : Nat) (c : ℝ) :
    Graph (ΓQKV m d) [.dim d (.dim m .scalar), .dim m (.dim m .scalar), .dim m (.dim m .scalar),
      .dim m (.dim m .scalar)] :=
  Graph.snoc (ss := [.dim d (.dim m .scalar), .dim m (.dim m .scalar), .dim m (.dim m .scalar)])
    (graphScaled m d c) (nodeProbs m d)

/--
Scaled dot-product attention as an explicit tape graph.

Computes `Q K V ↦ softmax(c * (Q * Kᵀ)) * V` and records intermediate values needed by backprop.
The graph is a plain `Graph.snoc` chain so that `Graph.evalVec` can be computed node by node when
relating the tape to the specification forward pass.
-/
def scaledDotProductGraph {m d : Nat} (c : ℝ) : Graph (ΓQKV m d) (ssScaledDotProduct m d) :=
  Graph.snoc (ss := [.dim d (.dim m .scalar), .dim m (.dim m .scalar), .dim m (.dim m .scalar),
      .dim m (.dim m .scalar)])
    (graphProbs m d c) (nodeOut m d)

/--
Scaled dot-product attention as a proved-correct `DGraph`.

Every node of `scaledDotProductGraph` carries its `NodeFDerivCorrect` certificate.
-/
def scaledDotProductDGraph {m d : Nat} (c : ℝ) : DGraph (ΓQKV m d) (ssScaledDotProduct m d) :=
  ⟨scaledDotProductGraph c,
    ⟨⟨⟨⟨⟨PUnit.unit,
      TapeNodes.matrixTransposeFderiv (Γ := ΓQKV m d) (m := m) (n := d)
        (A := idxK (m := m) (d := d))⟩,
      TapeNodes.matmulFderiv (Γ := ΓQKV m d ++ [.dim d (.dim m .scalar)]) (m := m) (n := d) (p := m)
        (A := idxQ (m := m) (d := d) (ss := [.dim d (.dim m .scalar)]))
        (B := Idx.last (Γ := ΓQKV m d) (ss := []) (τ := .dim d (.dim m .scalar)))⟩,
      TapeNodes.scaleFderiv
        (Γ := ΓQKV m d ++ [.dim d (.dim m .scalar), .dim m (.dim m .scalar)])
        (s := .dim m (.dim m .scalar))
        (idx := Idx.last (Γ := ΓQKV m d) (ss := [.dim d (.dim m .scalar)])
          (τ := .dim m (.dim m .scalar)))
        (c := c)⟩,
      TapeNodes.softmaxLastFderiv
        (Γ := ΓQKV m d ++
          [.dim d (.dim m .scalar), .dim m (.dim m .scalar), .dim m (.dim m .scalar)])
        (m := m) (n := m)
        (idx := Idx.last (Γ := ΓQKV m d) (ss := [.dim d (.dim m .scalar), .dim m (.dim m .scalar)])
          (τ := .dim m (.dim m .scalar)))⟩,
      TapeNodes.matmulFderiv
        (Γ := ΓQKV m d ++
          [.dim d (.dim m .scalar), .dim m (.dim m .scalar), .dim m (.dim m .scalar),
            .dim m (.dim m .scalar)])
        (m := m) (n := m) (p := d)
        (A := Idx.last (Γ := ΓQKV m d)
          (ss := [.dim d (.dim m .scalar), .dim m (.dim m .scalar), .dim m (.dim m .scalar)])
          (τ := .dim m (.dim m .scalar)))
        (B := idxV (m := m) (d := d)
          (ss := [.dim d (.dim m .scalar), .dim m (.dim m .scalar), .dim m (.dim m .scalar),
            .dim m (.dim m .scalar)]))⟩⟩

/--
Corollary of the general DAG theorem: backprop equals `(fderiv eval)†` for the attention graph.

This is the formal statement that the tape reverse pass computes the VJP for the full attention
computation.
-/
theorem backprop_eq_adjoint_fderiv_scaledDotProduct {m d : Nat} (c : ℝ) :
    ∀ (xV : CtxVec (ΓQKV m d))
      (seedV :
        CtxVec (ΓQKV m d ++
          [ .dim d (.dim m .scalar)
          , .dim m (.dim m .scalar)
          , .dim m (.dim m .scalar)
          , .dim m (.dim m .scalar)
          , .dim m (.dim d .scalar)
          ])),
      Graph.backpropVec (Γ := ΓQKV m d)
          (ss :=
            [ .dim d (.dim m .scalar)
            , .dim m (.dim m .scalar)
            , .dim m (.dim m .scalar)
            , .dim m (.dim m .scalar)
            , .dim m (.dim d .scalar)
            ])
          (scaledDotProductDGraph (m := m) (d := d) c).g xV seedV
        =
      (fderiv ℝ
          (Graph.evalVec (Γ := ΓQKV m d)
            (ss :=
              [ .dim d (.dim m .scalar)
              , .dim m (.dim m .scalar)
              , .dim m (.dim m .scalar)
              , .dim m (.dim m .scalar)
              , .dim m (.dim d .scalar)
              ])
            (scaledDotProductDGraph (m := m) (d := d) c).g)
          xV).adjoint seedV :=
  DGraph.backpropVec_eq_adjoint_fderiv (dg := scaledDotProductDGraph (m := m) (d := d) c)

end Attention

end

end Autograd
end Proofs
