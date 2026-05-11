/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Tape.Nodes.Batched
public import NN.Proofs.Autograd.Tape.Nodes.Shape
public import NN.Proofs.Autograd.Tape.Util.Idx

/-!
# MultiHeadSelfAttention

End-to-end `fderiv`/backprop correctness for a **Multi-Head Self-Attention** graph,
decomposed into proven tape nodes:
- linear projections via `matmul`,
- head split/merge via `reshape` + `swap_first_two3d`,
- attention core via batched `matmul` + `transpose3d_last_two` + `scale` + batched `softmax_last`.

This is spec-level over `ℝ`. It is a corollary of the general graph theorem once each node
used by the graph has a `NodeFDerivCorrect` instance.

## PyTorch correspondence / citations
- The construction matches the usual “project → split heads → scaled dot-product attention →
  concat heads → output projection” pipeline used by `torch.nn.MultiheadAttention`.
  https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
- The core attention step corresponds to `torch.nn.functional.scaled_dot_product_attention`.
  https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
-/

@[expose] public section


namespace Proofs
namespace Autograd

open Spec

open scoped BigOperators

noncomputable section

namespace MultiHeadAttention

open TapeNodes
open TapeNodes.ShapeOps
open TapeNodes.Batched
open DGraph

/-- Sequence input shape `n×dModel`. -/
abbrev XShape (n dModel : Nat) : Shape := .dim n (.dim dModel .scalar)

/-- Concatenated-head representation `n×(numHeads*headDim)`. -/
abbrev BigShape (n numHeads headDim : Nat) : Shape := .dim n (.dim (numHeads * headDim) .scalar)

/-- Split-head representation `(numHeads)×n×headDim`. -/
abbrev HeadsShape (n numHeads headDim : Nat) : Shape := .dim numHeads (.dim n (.dim headDim
  .scalar))

/-- Key-transposed shape `(numHeads)×headDim×n` used for `Q Kᵀ`. -/
abbrev KtShape (n numHeads headDim : Nat) : Shape := .dim numHeads (.dim headDim (.dim n .scalar))

/-- Attention scores shape `(numHeads)×n×n`. -/
abbrev ScoresShape (n numHeads : Nat) : Shape := .dim numHeads (.dim n (.dim n .scalar))

/-- Intermediate shape after swapping axes for concatenation `n×numHeads×headDim`. -/
abbrev SwappedShape (n numHeads headDim : Nat) : Shape := .dim n (.dim numHeads (.dim headDim
  .scalar))

/-- Intermediate node output shapes (tape “saved tensors”) for the MHA graph. -/
abbrev ssMHA (n dModel numHeads headDim : Nat) : List Shape :=
  [ BigShape n numHeads headDim
  , HeadsShape n numHeads headDim
  , BigShape n numHeads headDim
  , HeadsShape n numHeads headDim
  , KtShape n numHeads headDim
  , BigShape n numHeads headDim
  , HeadsShape n numHeads headDim
  , ScoresShape n numHeads
  , ScoresShape n numHeads
  , ScoresShape n numHeads
  , HeadsShape n numHeads headDim
  , SwappedShape n numHeads headDim
  , BigShape n numHeads headDim
  , XShape n dModel
  ]

/-- Projection weight shape `dModel×(numHeads*headDim)` (used for Q/K/V). -/
abbrev WqShape (dModel numHeads headDim : Nat) : Shape := .dim dModel (.dim (numHeads * headDim)
  .scalar)

/-- Output projection weight shape `(numHeads*headDim)×dModel`. -/
abbrev WoShape (dModel numHeads headDim : Nat) : Shape := .dim (numHeads * headDim) (.dim dModel
  .scalar)

/-- Input context shapes: `[x, Wq, Wk, Wv, Wo]`. -/
abbrev ΓMHA (n dModel numHeads headDim : Nat) : List Shape :=
  [ XShape n dModel
  , WqShape dModel numHeads headDim  -- Wq
  , WqShape dModel numHeads headDim  -- Wk
  , WqShape dModel numHeads headDim  -- Wv
  , WoShape dModel numHeads headDim  -- Wo
  ]

/-- Context index of the sequence input `x` in `ΓMHA`. -/
def idxX {n dModel numHeads headDim : Nat} {ss : List Shape} :
    Idx (ΓMHA n dModel numHeads headDim ++ ss) (XShape n dModel) :=
  ⟨⟨0, by simp [ΓMHA]⟩, by simp [ΓMHA]⟩

/-- Context index of `Wq` in `ΓMHA`. -/
def idxWq {n dModel numHeads headDim : Nat} {ss : List Shape} :
    Idx (ΓMHA n dModel numHeads headDim ++ ss) (WqShape dModel numHeads headDim) :=
  ⟨⟨1, by simp [ΓMHA]⟩, by simp [ΓMHA]⟩

/-- Context index of `Wk` in `ΓMHA`. -/
def idxWk {n dModel numHeads headDim : Nat} {ss : List Shape} :
    Idx (ΓMHA n dModel numHeads headDim ++ ss) (WqShape dModel numHeads headDim) :=
  ⟨⟨2, by simp [ΓMHA]⟩, by simp [ΓMHA]⟩

/-- Context index of `Wv` in `ΓMHA`. -/
def idxWv {n dModel numHeads headDim : Nat} {ss : List Shape} :
    Idx (ΓMHA n dModel numHeads headDim ++ ss) (WqShape dModel numHeads headDim) :=
  ⟨⟨3, by simp [ΓMHA]⟩, by simp [ΓMHA]⟩

/-- Context index of `Wo` in `ΓMHA`. -/
def idxWo {n dModel numHeads headDim : Nat} {ss : List Shape} :
    Idx (ΓMHA n dModel numHeads headDim ++ ss) (WoShape dModel numHeads headDim) :=
  ⟨⟨4, by simp [ΓMHA]⟩, by simp [ΓMHA]⟩

/-- Index of the most-recently appended tensor in a `DGraph` context. -/
def idxLast {Γ : List Shape} {ss : List Shape} {τ : Shape} :
    Idx (Γ ++ ss ++ [τ]) τ :=
  _root_.Proofs.Autograd.Idx.last (Γ := Γ) (ss := ss) (τ := τ)

lemma size_big_to_heads (n numHeads headDim : Nat) :
    Shape.size (BigShape n numHeads headDim) = Shape.size (HeadsShape n numHeads headDim) := by
  -- `Shape.size` multiplies dimension sizes; the remaining goal is a commutative-monoid identity.
  simp [Shape.size]
  ac_rfl

lemma size_swap_to_concat (n numHeads headDim : Nat) :
    Shape.size (.dim n (.dim numHeads (.dim headDim .scalar))) = Shape.size (BigShape n numHeads
      headDim) := by
  simp [Shape.size]

section

set_option maxHeartbeats 20000000

/--
Multi-head self-attention as a proof-carrying graph.

This implements:
`x Wq Wk Wv Wo ↦ Wo (concat_heads (softmax(c * (Q Kᵀ)) V))`,
with `Q/K/V` projected from `x`.

The graph is laid out to match typical runtime implementations:
`view(...).transpose(...)` is modeled by `reshape` + `swap_first_two3d`.
-/
def mhaDGraph {n dModel numHeads headDim : Nat} (c : ℝ) :
    DGraph (ΓMHA n dModel numHeads headDim) (ssMHA n dModel numHeads headDim) := by
  classical

  let dg0 : DGraph (ΓMHA n dModel numHeads headDim) [] := DGraph.nil

  -- 1) Qbig := x * Wq
  let nodeQbig :
      Node (ΓMHA n dModel numHeads headDim) (BigShape n numHeads headDim) :=
    TapeNodes.matmul (Γ := ΓMHA n dModel numHeads headDim)
      (m := n) (n := dModel) (p := numHeads * headDim)
      (A := idxX (n := n) (dModel := dModel) (numHeads := numHeads) (headDim := headDim) (ss := []))
      (B := idxWq (n := n) (dModel := dModel) (numHeads := numHeads) (headDim := headDim) (ss :=
        []))
  let dg1 :=
    DGraph.snoc (dg := dg0) (node := nodeQbig)
      (hn := TapeNodes.matmulFderiv
        (Γ := ΓMHA n dModel numHeads headDim)
        (m := n) (n := dModel) (p := numHeads * headDim)
        (A := idxX (n := n) (dModel := dModel) (numHeads := numHeads) (headDim := headDim) (ss :=
          []))
        (B := idxWq (n := n) (dModel := dModel) (numHeads := numHeads) (headDim := headDim) (ss :=
          [])))

  -- 2) Qheads := reshape Qbig to (numHeads, n, headDim)
  let idxQbig :
      Idx (ΓMHA n dModel numHeads headDim ++ [BigShape n numHeads headDim]) (BigShape n numHeads
        headDim) :=
    idxLast (Γ := ΓMHA n dModel numHeads headDim) (ss := []) (τ := BigShape n numHeads headDim)
  let nodeQheads :
      Node (ΓMHA n dModel numHeads headDim ++ [BigShape n numHeads headDim]) (HeadsShape n numHeads
        headDim) :=
    reshape (Γ := ΓMHA n dModel numHeads headDim ++ [BigShape n numHeads headDim])
      (s₁ := BigShape n numHeads headDim) (s₂ := HeadsShape n numHeads headDim)
      idxQbig (size_big_to_heads (n := n) (numHeads := numHeads) (headDim := headDim))
  let dg2 :=
    DGraph.snoc (dg := dg1) (node := nodeQheads)
      (hn := reshapeFderiv
        (Γ := ΓMHA n dModel numHeads headDim ++ [BigShape n numHeads headDim])
        (s₁ := BigShape n numHeads headDim) (s₂ := HeadsShape n numHeads headDim)
        idxQbig (size_big_to_heads (n := n) (numHeads := numHeads) (headDim := headDim)))

  -- 3) Kbig := x * Wk
  let nodeKbig :
      Node (ΓMHA n dModel numHeads headDim ++ [BigShape n numHeads headDim, HeadsShape n numHeads
        headDim])
        (BigShape n numHeads headDim) :=
    TapeNodes.matmul (Γ := ΓMHA n dModel numHeads headDim ++ [BigShape n numHeads headDim,
      HeadsShape n numHeads headDim])
      (m := n) (n := dModel) (p := numHeads * headDim)
      (A := idxX (n := n) (dModel := dModel) (numHeads := numHeads) (headDim := headDim)
        (ss := [BigShape n numHeads headDim, HeadsShape n numHeads headDim]))
      (B := idxWk (n := n) (dModel := dModel) (numHeads := numHeads) (headDim := headDim)
        (ss := [BigShape n numHeads headDim, HeadsShape n numHeads headDim]))
  let dg3 :=
    DGraph.snoc (dg := dg2) (node := nodeKbig)
      (hn := TapeNodes.matmulFderiv
        (Γ := ΓMHA n dModel numHeads headDim ++ [BigShape n numHeads headDim, HeadsShape n numHeads
          headDim])
        (m := n) (n := dModel) (p := numHeads * headDim)
        (A := idxX (n := n) (dModel := dModel) (numHeads := numHeads) (headDim := headDim)
          (ss := [BigShape n numHeads headDim, HeadsShape n numHeads headDim]))
        (B := idxWk (n := n) (dModel := dModel) (numHeads := numHeads) (headDim := headDim)
          (ss := [BigShape n numHeads headDim, HeadsShape n numHeads headDim])))

  -- 4) Kheads := reshape Kbig
  let idxKbig :
      Idx
        (ΓMHA n dModel numHeads headDim ++
          [BigShape n numHeads headDim, HeadsShape n numHeads headDim, BigShape n numHeads headDim])
        (BigShape n numHeads headDim) :=
    idxLast (Γ := ΓMHA n dModel numHeads headDim)
      (ss := [BigShape n numHeads headDim, HeadsShape n numHeads headDim]) (τ := BigShape n numHeads
        headDim)
  let nodeKheads :
      Node
        (ΓMHA n dModel numHeads headDim ++
          [BigShape n numHeads headDim, HeadsShape n numHeads headDim, BigShape n numHeads headDim])
        (HeadsShape n numHeads headDim) :=
    reshape (Γ := ΓMHA n dModel numHeads headDim ++
          [BigShape n numHeads headDim, HeadsShape n numHeads headDim, BigShape n numHeads headDim])
      (s₁ := BigShape n numHeads headDim) (s₂ := HeadsShape n numHeads headDim)
      idxKbig (size_big_to_heads (n := n) (numHeads := numHeads) (headDim := headDim))
  let dg4 :=
    DGraph.snoc (dg := dg3) (node := nodeKheads)
      (hn := reshapeFderiv
        (Γ := ΓMHA n dModel numHeads headDim ++
          [BigShape n numHeads headDim, HeadsShape n numHeads headDim, BigShape n numHeads headDim])
        (s₁ := BigShape n numHeads headDim) (s₂ := HeadsShape n numHeads headDim)
        idxKbig (size_big_to_heads (n := n) (numHeads := numHeads) (headDim := headDim)))

  -- 5) Kᵀ (per head): transpose last two axes of Kheads
  let idxKheads :
      Idx
        (ΓMHA n dModel numHeads headDim ++
          [BigShape n numHeads headDim, HeadsShape n numHeads headDim, BigShape n numHeads headDim,
            HeadsShape n numHeads headDim])
        (HeadsShape n numHeads headDim) :=
    idxLast (Γ := ΓMHA n dModel numHeads headDim)
      (ss := [BigShape n numHeads headDim, HeadsShape n numHeads headDim, BigShape n numHeads
        headDim])
      (τ := HeadsShape n numHeads headDim)
  let nodeKt :
      Node
        (ΓMHA n dModel numHeads headDim ++
          [BigShape n numHeads headDim, HeadsShape n numHeads headDim, BigShape n numHeads headDim,
            HeadsShape n numHeads headDim])
        (.dim numHeads (.dim headDim (.dim n .scalar))) :=
    transpose3dLastTwo
      (Γ := ΓMHA n dModel numHeads headDim ++
        [BigShape n numHeads headDim, HeadsShape n numHeads headDim, BigShape n numHeads headDim,
          HeadsShape n numHeads headDim])
      (a := numHeads) (b := n) (c := headDim) idxKheads
  let dg5 :=
    DGraph.snoc (dg := dg4) (node := nodeKt)
      (hn := transpose3dLastTwoFderiv
        (Γ := ΓMHA n dModel numHeads headDim ++
          [BigShape n numHeads headDim, HeadsShape n numHeads headDim, BigShape n numHeads headDim,
            HeadsShape n numHeads headDim])
        (a := numHeads) (b := n) (c := headDim) idxKheads)

  -- 6) Vbig := x * Wv
  let nodeVbig :
      Node
        (ΓMHA n dModel numHeads headDim ++
          [ BigShape n numHeads headDim, HeadsShape n numHeads headDim
          , BigShape n numHeads headDim, HeadsShape n numHeads headDim
          , .dim numHeads (.dim headDim (.dim n .scalar))
          ])
        (BigShape n numHeads headDim) :=
    TapeNodes.matmul
      (Γ := ΓMHA n dModel numHeads headDim ++
        [ BigShape n numHeads headDim, HeadsShape n numHeads headDim
        , BigShape n numHeads headDim, HeadsShape n numHeads headDim
        , .dim numHeads (.dim headDim (.dim n .scalar))
        ])
      (m := n) (n := dModel) (p := numHeads * headDim)
      (A := idxX (n := n) (dModel := dModel) (numHeads := numHeads) (headDim := headDim)
        (ss :=
          [ BigShape n numHeads headDim, HeadsShape n numHeads headDim
          , BigShape n numHeads headDim, HeadsShape n numHeads headDim
          , .dim numHeads (.dim headDim (.dim n .scalar))
          ]))
      (B := idxWv (n := n) (dModel := dModel) (numHeads := numHeads) (headDim := headDim)
        (ss :=
          [ BigShape n numHeads headDim, HeadsShape n numHeads headDim
          , BigShape n numHeads headDim, HeadsShape n numHeads headDim
          , .dim numHeads (.dim headDim (.dim n .scalar))
          ]))
  let dg6 :=
    DGraph.snoc (dg := dg5) (node := nodeVbig)
      (hn := TapeNodes.matmulFderiv
        (Γ := ΓMHA n dModel numHeads headDim ++
          [ BigShape n numHeads headDim, HeadsShape n numHeads headDim
          , BigShape n numHeads headDim, HeadsShape n numHeads headDim
          , .dim numHeads (.dim headDim (.dim n .scalar))
          ])
        (m := n) (n := dModel) (p := numHeads * headDim)
        (A := idxX (n := n) (dModel := dModel) (numHeads := numHeads) (headDim := headDim)
          (ss :=
            [ BigShape n numHeads headDim, HeadsShape n numHeads headDim
            , BigShape n numHeads headDim, HeadsShape n numHeads headDim
            , .dim numHeads (.dim headDim (.dim n .scalar))
            ]))
        (B := idxWv (n := n) (dModel := dModel) (numHeads := numHeads) (headDim := headDim)
          (ss :=
            [ BigShape n numHeads headDim, HeadsShape n numHeads headDim
            , BigShape n numHeads headDim, HeadsShape n numHeads headDim
            , .dim numHeads (.dim headDim (.dim n .scalar))
            ])))

  -- 7) Vheads := reshape Vbig
  let idxVbig :
      Idx
        (ΓMHA n dModel numHeads headDim ++
          [ BigShape n numHeads headDim, HeadsShape n numHeads headDim
          , BigShape n numHeads headDim, HeadsShape n numHeads headDim
          , .dim numHeads (.dim headDim (.dim n .scalar))
          , BigShape n numHeads headDim
          ])
        (BigShape n numHeads headDim) :=
    idxLast
      (Γ := ΓMHA n dModel numHeads headDim)
      (ss :=
        [ BigShape n numHeads headDim, HeadsShape n numHeads headDim
        , BigShape n numHeads headDim, HeadsShape n numHeads headDim
        , .dim numHeads (.dim headDim (.dim n .scalar))
        ])
      (τ := BigShape n numHeads headDim)
  let nodeVheads :
      Node
        (ΓMHA n dModel numHeads headDim ++
          [ BigShape n numHeads headDim, HeadsShape n numHeads headDim
          , BigShape n numHeads headDim, HeadsShape n numHeads headDim
          , .dim numHeads (.dim headDim (.dim n .scalar))
          , BigShape n numHeads headDim
          ])
        (HeadsShape n numHeads headDim) :=
    reshape
      (Γ := ΓMHA n dModel numHeads headDim ++
        [ BigShape n numHeads headDim, HeadsShape n numHeads headDim
        , BigShape n numHeads headDim, HeadsShape n numHeads headDim
        , .dim numHeads (.dim headDim (.dim n .scalar))
        , BigShape n numHeads headDim
        ])
      (s₁ := BigShape n numHeads headDim) (s₂ := HeadsShape n numHeads headDim)
      idxVbig (size_big_to_heads (n := n) (numHeads := numHeads) (headDim := headDim))
  let dg7 :=
    DGraph.snoc (dg := dg6) (node := nodeVheads)
      (hn := reshapeFderiv
        (Γ := ΓMHA n dModel numHeads headDim ++
          [ BigShape n numHeads headDim, HeadsShape n numHeads headDim
          , BigShape n numHeads headDim, HeadsShape n numHeads headDim
          , .dim numHeads (.dim headDim (.dim n .scalar))
          , BigShape n numHeads headDim
          ])
        (s₁ := BigShape n numHeads headDim) (s₂ := HeadsShape n numHeads headDim)
        idxVbig (size_big_to_heads (n := n) (numHeads := numHeads) (headDim := headDim)))

  -- 8) scores := Qheads * Kᵀ (batched matmul)
  let ss7 :=
      [ BigShape n numHeads headDim
      , HeadsShape n numHeads headDim
      , BigShape n numHeads headDim
      , HeadsShape n numHeads headDim
      , .dim numHeads (.dim headDim (.dim n .scalar))
      , BigShape n numHeads headDim
      , HeadsShape n numHeads headDim
      ]
  let idxQheads0 :
      Idx (ΓMHA n dModel numHeads headDim ++ [BigShape n numHeads headDim, HeadsShape n numHeads
        headDim])
        (HeadsShape n numHeads headDim) :=
    idxLast (Γ := ΓMHA n dModel numHeads headDim) (ss := [BigShape n numHeads headDim])
      (τ := HeadsShape n numHeads headDim)
  let idxQheads7 :
      Idx (ΓMHA n dModel numHeads headDim ++ ss7) (HeadsShape n numHeads headDim) :=
    _root_.Proofs.Autograd.Idx.weaken
      (Γ := ΓMHA n dModel numHeads headDim ++ [BigShape n numHeads headDim, HeadsShape n numHeads
        headDim])
      idxQheads0
      (rest := [ BigShape n numHeads headDim, HeadsShape n numHeads headDim
               , .dim numHeads (.dim headDim (.dim n .scalar))
               , BigShape n numHeads headDim, HeadsShape n numHeads headDim ])
  let idxKt0 :
      Idx
        (ΓMHA n dModel numHeads headDim ++
          [ BigShape n numHeads headDim, HeadsShape n numHeads headDim
          , BigShape n numHeads headDim, HeadsShape n numHeads headDim
          , .dim numHeads (.dim headDim (.dim n .scalar))
          ])
        (.dim numHeads (.dim headDim (.dim n .scalar))) :=
    idxLast
      (Γ := ΓMHA n dModel numHeads headDim)
      (ss := [ BigShape n numHeads headDim, HeadsShape n numHeads headDim
             , BigShape n numHeads headDim, HeadsShape n numHeads headDim ])
      (τ := .dim numHeads (.dim headDim (.dim n .scalar)))
  let idxKt7 :
      Idx (ΓMHA n dModel numHeads headDim ++ ss7) (.dim numHeads (.dim headDim (.dim n .scalar))) :=
    _root_.Proofs.Autograd.Idx.weaken
      (Γ := ΓMHA n dModel numHeads headDim ++
        [ BigShape n numHeads headDim, HeadsShape n numHeads headDim
        , BigShape n numHeads headDim, HeadsShape n numHeads headDim
        , .dim numHeads (.dim headDim (.dim n .scalar))
        ])
      idxKt0
      (rest := [BigShape n numHeads headDim, HeadsShape n numHeads headDim])
  let nodeScores :
      Node (ΓMHA n dModel numHeads headDim ++ ss7) (.dim numHeads (.dim n (.dim n .scalar))) :=
    TapeNodes.Batched.matmul
      (Γ := ΓMHA n dModel numHeads headDim ++ ss7)
      (h := numHeads) (m := n) (n := headDim) (p := n)
      (A := idxQheads7) (B := idxKt7)
  let dg8 :=
    DGraph.snoc (dg := dg7) (node := nodeScores)
      (hn := TapeNodes.Batched.matmulFderiv
        (Γ := ΓMHA n dModel numHeads headDim ++ ss7)
        (h := numHeads) (m := n) (n := headDim) (p := n)
        (A := idxQheads7) (B := idxKt7))

  -- 9) scaled := scale scores c
  let idxScores :
      Idx (ΓMHA n dModel numHeads headDim ++ ss7 ++ [.dim numHeads (.dim n (.dim n .scalar))])
        (.dim numHeads (.dim n (.dim n .scalar))) :=
    idxLast (Γ := ΓMHA n dModel numHeads headDim) (ss := ss7) (τ := .dim numHeads (.dim n (.dim n
      .scalar)))
  let nodeScaled :
      Node (ΓMHA n dModel numHeads headDim ++ ss7 ++ [.dim numHeads (.dim n (.dim n .scalar))])
        (.dim numHeads (.dim n (.dim n .scalar))) :=
    TapeNodes.scale
      (Γ := ΓMHA n dModel numHeads headDim ++ ss7 ++ [.dim numHeads (.dim n (.dim n .scalar))])
      (s := .dim numHeads (.dim n (.dim n .scalar))) (idx := idxScores) c
  let dg9 :=
    DGraph.snoc (dg := dg8) (node := nodeScaled)
      (hn := TapeNodes.scaleFderiv
        (Γ := ΓMHA n dModel numHeads headDim ++ ss7 ++ [.dim numHeads (.dim n (.dim n .scalar))])
        (s := .dim numHeads (.dim n (.dim n .scalar))) (idx := idxScores) (c := c))

  -- 10) probs := batched softmax_last scaled
  let idxScaled :
      Idx
        (ΓMHA n dModel numHeads headDim ++ ss7 ++
          [.dim numHeads (.dim n (.dim n .scalar)), .dim numHeads (.dim n (.dim n .scalar))])
        (.dim numHeads (.dim n (.dim n .scalar))) :=
    idxLast
      (Γ := ΓMHA n dModel numHeads headDim)
      (ss := ss7 ++ [.dim numHeads (.dim n (.dim n .scalar))])
      (τ := .dim numHeads (.dim n (.dim n .scalar)))
  let nodeProbs :
      Node
        (ΓMHA n dModel numHeads headDim ++ ss7 ++
          [.dim numHeads (.dim n (.dim n .scalar)), .dim numHeads (.dim n (.dim n .scalar))])
        (.dim numHeads (.dim n (.dim n .scalar))) :=
    TapeNodes.Batched.softmaxLast
      (Γ := ΓMHA n dModel numHeads headDim ++ ss7 ++
        [.dim numHeads (.dim n (.dim n .scalar)), .dim numHeads (.dim n (.dim n .scalar))])
      (h := numHeads) (m := n) (n := n) (idx := idxScaled)
  let dg10 :=
    DGraph.snoc (dg := dg9) (node := nodeProbs)
      (hn := TapeNodes.Batched.softmaxLastFderiv
        (Γ := ΓMHA n dModel numHeads headDim ++ ss7 ++
          [.dim numHeads (.dim n (.dim n .scalar)), .dim numHeads (.dim n (.dim n .scalar))])
        (h := numHeads) (m := n) (n := n) (idx := idxScaled))

  -- 11) headOut := probs * Vheads (batched matmul)
  let ss10 := ss7 ++
      [.dim numHeads (.dim n (.dim n .scalar))       -- scores
      , .dim numHeads (.dim n (.dim n .scalar))       -- scaled
      , .dim numHeads (.dim n (.dim n .scalar))]      -- probs
  let idxVheads0 :
      Idx (ΓMHA n dModel numHeads headDim ++ ss7) (HeadsShape n numHeads headDim) :=
    idxLast
      (Γ := ΓMHA n dModel numHeads headDim)
      (ss :=
        [ BigShape n numHeads headDim
        , HeadsShape n numHeads headDim
        , BigShape n numHeads headDim
        , HeadsShape n numHeads headDim
        , .dim numHeads (.dim headDim (.dim n .scalar))
        , BigShape n numHeads headDim
        ])
      (τ := HeadsShape n numHeads headDim)
  let idxVheads10 :
      Idx (ΓMHA n dModel numHeads headDim ++ ss10) (HeadsShape n numHeads headDim) :=
    _root_.Proofs.Autograd.Idx.weaken
      (Γ := ΓMHA n dModel numHeads headDim ++ ss7)
      idxVheads0
      (rest :=
        [.dim numHeads (.dim n (.dim n .scalar))
        , .dim numHeads (.dim n (.dim n .scalar))
        , .dim numHeads (.dim n (.dim n .scalar))])
  let idxProbs :
      Idx (ΓMHA n dModel numHeads headDim ++ ss10) (.dim numHeads (.dim n (.dim n .scalar))) :=
    idxLast
      (Γ := ΓMHA n dModel numHeads headDim)
      (ss := ss7 ++
        [.dim numHeads (.dim n (.dim n .scalar)), .dim numHeads (.dim n (.dim n .scalar))])
      (τ := .dim numHeads (.dim n (.dim n .scalar)))
  let nodeHeadOut :
      Node (ΓMHA n dModel numHeads headDim ++ ss10) (HeadsShape n numHeads headDim) :=
    TapeNodes.Batched.matmul
      (Γ := ΓMHA n dModel numHeads headDim ++ ss10)
      (h := numHeads) (m := n) (n := n) (p := headDim)
      (A := idxProbs) (B := idxVheads10)
  let dg11 :=
    DGraph.snoc (dg := dg10) (node := nodeHeadOut)
      (hn := TapeNodes.Batched.matmulFderiv
        (Γ := ΓMHA n dModel numHeads headDim ++ ss10)
        (h := numHeads) (m := n) (n := n) (p := headDim)
        (A := idxProbs) (B := idxVheads10))

  -- 12) swapped := swap_first_two3d headOut  (numHeads,n,headDim) → (n,numHeads,headDim)
  let idxHeadOut :
      Idx (ΓMHA n dModel numHeads headDim ++ ss10 ++ [HeadsShape n numHeads headDim])
        (HeadsShape n numHeads headDim) :=
    idxLast (Γ := ΓMHA n dModel numHeads headDim) (ss := ss10) (τ := HeadsShape n numHeads headDim)
  let nodeSwapped :
      Node (ΓMHA n dModel numHeads headDim ++ ss10 ++ [HeadsShape n numHeads headDim])
        (.dim n (.dim numHeads (.dim headDim .scalar))) :=
    swapFirstTwo3d
      (Γ := ΓMHA n dModel numHeads headDim ++ ss10 ++ [HeadsShape n numHeads headDim])
      (m := numHeads) (n := n) (rest := .dim headDim .scalar) idxHeadOut
  let dg12 :=
    DGraph.snoc (dg := dg11) (node := nodeSwapped)
      (hn := swapFirstTwo3dFderiv
        (Γ := ΓMHA n dModel numHeads headDim ++ ss10 ++ [HeadsShape n numHeads headDim])
        (m := numHeads) (n := n) (rest := .dim headDim .scalar) idxHeadOut)

  -- 13) concat := reshape swapped to (n, numHeads*headDim)
  let idxSwapped :
      Idx
        (ΓMHA n dModel numHeads headDim ++ ss10 ++
          [HeadsShape n numHeads headDim, .dim n (.dim numHeads (.dim headDim .scalar))])
        (.dim n (.dim numHeads (.dim headDim .scalar))) :=
    idxLast
      (Γ := ΓMHA n dModel numHeads headDim)
      (ss := ss10 ++ [HeadsShape n numHeads headDim])
      (τ := .dim n (.dim numHeads (.dim headDim .scalar)))
  let nodeConcat :
      Node
        (ΓMHA n dModel numHeads headDim ++ ss10 ++
          [HeadsShape n numHeads headDim, .dim n (.dim numHeads (.dim headDim .scalar))])
        (BigShape n numHeads headDim) :=
    reshape
      (Γ := ΓMHA n dModel numHeads headDim ++ ss10 ++
        [HeadsShape n numHeads headDim, .dim n (.dim numHeads (.dim headDim .scalar))])
      (s₁ := .dim n (.dim numHeads (.dim headDim .scalar)))
      (s₂ := BigShape n numHeads headDim)
      idxSwapped (size_swap_to_concat (n := n) (numHeads := numHeads) (headDim := headDim))
  let dg13 :=
    DGraph.snoc (dg := dg12) (node := nodeConcat)
      (hn := reshapeFderiv
        (Γ := ΓMHA n dModel numHeads headDim ++ ss10 ++
          [HeadsShape n numHeads headDim, .dim n (.dim numHeads (.dim headDim .scalar))])
        (s₁ := .dim n (.dim numHeads (.dim headDim .scalar)))
        (s₂ := BigShape n numHeads headDim)
        idxSwapped (size_swap_to_concat (n := n) (numHeads := numHeads) (headDim := headDim)))

  -- 14) out := concat * Wo
  let idxConcat :
      Idx (ΓMHA n dModel numHeads headDim ++ ss10 ++
        [ HeadsShape n numHeads headDim
        , .dim n (.dim numHeads (.dim headDim .scalar))
        , BigShape n numHeads headDim
        ])
        (BigShape n numHeads headDim) :=
    idxLast
      (Γ := ΓMHA n dModel numHeads headDim)
      (ss := ss10 ++ [HeadsShape n numHeads headDim, .dim n (.dim numHeads (.dim headDim .scalar))])
      (τ := BigShape n numHeads headDim)
  let nodeOut :
      Node (ΓMHA n dModel numHeads headDim ++ ss10 ++
        [ HeadsShape n numHeads headDim
        , .dim n (.dim numHeads (.dim headDim .scalar))
        , BigShape n numHeads headDim
        ])
        (XShape n dModel) :=
    TapeNodes.matmul
      (Γ := ΓMHA n dModel numHeads headDim ++ ss10 ++
        [ HeadsShape n numHeads headDim
        , .dim n (.dim numHeads (.dim headDim .scalar))
        , BigShape n numHeads headDim
        ])
      (m := n) (n := numHeads * headDim) (p := dModel)
      (A := idxConcat)
      (B := idxWo (n := n) (dModel := dModel) (numHeads := numHeads) (headDim := headDim)
        (ss := ss10 ++
          [ HeadsShape n numHeads headDim
          , .dim n (.dim numHeads (.dim headDim .scalar))
          , BigShape n numHeads headDim
          ]))
  let dg14 :=
    DGraph.snoc (dg := dg13) (node := nodeOut)
      (hn := TapeNodes.matmulFderiv
        (Γ := ΓMHA n dModel numHeads headDim ++ ss10 ++
          [ HeadsShape n numHeads headDim
          , .dim n (.dim numHeads (.dim headDim .scalar))
          , BigShape n numHeads headDim
          ])
        (m := n) (n := numHeads * headDim) (p := dModel)
        (A := idxConcat)
        (B := idxWo (n := n) (dModel := dModel) (numHeads := numHeads) (headDim := headDim)
          (ss := ss10 ++
            [ HeadsShape n numHeads headDim
            , .dim n (.dim numHeads (.dim headDim .scalar))
            , BigShape n numHeads headDim
            ])))

  exact dg14

end

/--
Corollary of the general DAG theorem: backprop equals `(fderiv eval)†` for the MHA graph.

This is the formal “VJP correctness” statement for the full MHA computation (as laid out by
  `mhaDGraph`).
-/
theorem mha_backpropVec_eq_adjoint_fderiv {n dModel numHeads headDim : Nat} (c : ℝ)
    (xV : CtxVec (ΓMHA n dModel numHeads headDim))
    (seedV : CtxVec (ΓMHA n dModel numHeads headDim ++ ssMHA n dModel numHeads headDim)) :
    Graph.backpropVec
        (Γ := ΓMHA n dModel numHeads headDim)
        (ss := ssMHA n dModel numHeads headDim)
        (mhaDGraph (n := n) (dModel := dModel) (numHeads := numHeads) (headDim := headDim) c).g xV
          seedV
      =
    (fderiv ℝ
        (Graph.evalVec
          (Γ := ΓMHA n dModel numHeads headDim)
          (ss := ssMHA n dModel numHeads headDim)
          (mhaDGraph (n := n) (dModel := dModel) (numHeads := numHeads) (headDim := headDim) c).g)
        xV).adjoint seedV :=
  DGraph.backpropVec_eq_adjoint_fderiv
    (dg := mhaDGraph (n := n) (dModel := dModel) (numHeads := numHeads) (headDim := headDim) c)
    xV seedV

end MultiHeadAttention

end

end Autograd
end Proofs
