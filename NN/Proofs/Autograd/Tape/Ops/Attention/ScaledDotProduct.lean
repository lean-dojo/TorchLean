/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Tape.Nodes.GraphComposition
public import NN.Proofs.Autograd.Tape.Util.Idx

/-!
# ScaledDotProduct

End-to-end `fderiv`/backprop correctness for a **scaled dot-product attention** graph,
built out of the proven tape nodes (`matmul`, `matrix_transpose`, `scale`, `softmax_last`).

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

open Spec

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

/-- Index of the most-recently appended tensor in a `DGraph` context. -/
def idxLast {Γ : List Shape} {ss : List Shape} {τ : Shape} :
    Idx (Γ ++ ss ++ [τ]) τ :=
  _root_.Proofs.Autograd.Idx.last (Γ := Γ) (ss := ss) (τ := τ)

/--
Scaled dot-product attention as a proved-correct `DGraph`.

Computes `Q K V ↦ softmax(c * (Q * Kᵀ)) * V` and records intermediate values needed by backprop.
-/
def scaledDotProductDGraph {m d : Nat} (c : ℝ) :
    DGraph (ΓQKV m d)
      [ .dim d (.dim m .scalar)           -- Kᵀ
      , .dim m (.dim m .scalar)           -- Q*Kᵀ
      , .dim m (.dim m .scalar)           -- scaled logits
      , .dim m (.dim m .scalar)           -- softmax probs
      , .dim m (.dim d .scalar)           -- output
      ] := by
  classical

  -- Start with an empty graph.
  let dg0 : DGraph (ΓQKV m d) [] := DGraph.nil

  -- 1) Kᵀ
  let nodeKt : Node (ΓQKV m d) (.dim d (.dim m .scalar)) :=
    TapeNodes.matrixTranspose (Γ := ΓQKV m d) (m := m) (n := d) (A := idxK (m := m) (d := d))
  let dg1 :=
    DGraph.snoc (dg := dg0) (node := nodeKt)
      (hn := TapeNodes.matrixTransposeFderiv (Γ := ΓQKV m d) (m := m) (n := d) (A := idxK (m := m)
        (d := d)))

  -- 2) logits := Q * Kᵀ
  let idxKt : Idx (ΓQKV m d ++ [.dim d (.dim m .scalar)]) (.dim d (.dim m .scalar)) :=
    idxLast (Γ := ΓQKV m d) (ss := []) (τ := .dim d (.dim m .scalar))
  let nodeLogits :
      Node (ΓQKV m d ++ [.dim d (.dim m .scalar)]) (.dim m (.dim m .scalar)) :=
    TapeNodes.matmul (Γ := ΓQKV m d ++ [.dim d (.dim m .scalar)])
      (m := m) (n := d) (p := m)
      (A := idxQ (m := m) (d := d) (ss := [.dim d (.dim m .scalar)]))
      (B := idxKt)
  let dg2 :=
    DGraph.snoc (dg := dg1) (node := nodeLogits)
      (hn := TapeNodes.matmulFderiv (Γ := ΓQKV m d ++ [.dim d (.dim m .scalar)])
        (m := m) (n := d) (p := m)
        (A := idxQ (m := m) (d := d) (ss := [.dim d (.dim m .scalar)]))
        (B := idxKt))

  -- 3) scaled := scale logits c
  let idxLogits :
      Idx (ΓQKV m d ++ [.dim d (.dim m .scalar), .dim m (.dim m .scalar)]) (.dim m (.dim m .scalar))
        :=
    idxLast (Γ := ΓQKV m d) (ss := [.dim d (.dim m .scalar)]) (τ := .dim m (.dim m .scalar))
  let nodeScaled :
      Node (ΓQKV m d ++ [.dim d (.dim m .scalar), .dim m (.dim m .scalar)]) (.dim m (.dim m
        .scalar)) :=
    TapeNodes.scale (Γ := ΓQKV m d ++ [.dim d (.dim m .scalar), .dim m (.dim m .scalar)])
      (s := .dim m (.dim m .scalar)) (idx := idxLogits) c
  let dg3 :=
    DGraph.snoc (dg := dg2) (node := nodeScaled)
      (hn := TapeNodes.scaleFderiv (Γ := ΓQKV m d ++ [.dim d (.dim m .scalar), .dim m (.dim m
        .scalar)])
        (s := .dim m (.dim m .scalar)) (idx := idxLogits) (c := c))

  -- 4) probs := softmax_last scaled
  let idxScaled :
      Idx (ΓQKV m d ++ [.dim d (.dim m .scalar), .dim m (.dim m .scalar), .dim m (.dim m .scalar)])
        (.dim m (.dim m .scalar)) :=
    idxLast (Γ := ΓQKV m d) (ss := [.dim d (.dim m .scalar), .dim m (.dim m .scalar)]) (τ := .dim m
      (.dim m .scalar))
  let nodeProbs :
      Node (ΓQKV m d ++ [.dim d (.dim m .scalar), .dim m (.dim m .scalar), .dim m (.dim m .scalar)])
        (.dim m (.dim m .scalar)) :=
    TapeNodes.softmaxLast (Γ := ΓQKV m d ++ [.dim d (.dim m .scalar), .dim m (.dim m .scalar), .dim
      m (.dim m .scalar)])
      (m := m) (n := m) (idx := idxScaled)
  let dg4 :=
    DGraph.snoc (dg := dg3) (node := nodeProbs)
      (hn := TapeNodes.softmaxLastFderiv
        (Γ := ΓQKV m d ++ [.dim d (.dim m .scalar), .dim m (.dim m .scalar), .dim m (.dim m
          .scalar)])
        (m := m) (n := m) (idx := idxScaled))

  -- 5) out := probs * V
  let idxProbs :
      Idx
        (ΓQKV m d ++
          [.dim d (.dim m .scalar), .dim m (.dim m .scalar), .dim m (.dim m .scalar), .dim m (.dim m
            .scalar)])
        (.dim m (.dim m .scalar)) :=
    idxLast (Γ := ΓQKV m d)
      (ss := [.dim d (.dim m .scalar), .dim m (.dim m .scalar), .dim m (.dim m .scalar)])
      (τ := .dim m (.dim m .scalar))
  let nodeOut :
      Node
        (ΓQKV m d ++
          [.dim d (.dim m .scalar), .dim m (.dim m .scalar), .dim m (.dim m .scalar), .dim m (.dim m
            .scalar)])
        (.dim m (.dim d .scalar)) :=
    TapeNodes.matmul
      (Γ := ΓQKV m d ++
        [.dim d (.dim m .scalar), .dim m (.dim m .scalar), .dim m (.dim m .scalar), .dim m (.dim m
          .scalar)])
      (m := m) (n := m) (p := d)
      (A := idxProbs)
      (B := idxV (m := m) (d := d)
        (ss := [.dim d (.dim m .scalar), .dim m (.dim m .scalar), .dim m (.dim m .scalar), .dim m
          (.dim m .scalar)]))
  let dg5 :=
    DGraph.snoc (dg := dg4) (node := nodeOut)
      (hn := TapeNodes.matmulFderiv
        (Γ := ΓQKV m d ++
          [.dim d (.dim m .scalar), .dim m (.dim m .scalar), .dim m (.dim m .scalar), .dim m (.dim m
            .scalar)])
        (m := m) (n := m) (p := d)
        (A := idxProbs)
        (B := idxV (m := m) (d := d)
          (ss := [.dim d (.dim m .scalar), .dim m (.dim m .scalar), .dim m (.dim m .scalar), .dim m
            (.dim m .scalar)])))

  simpa using dg5

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
