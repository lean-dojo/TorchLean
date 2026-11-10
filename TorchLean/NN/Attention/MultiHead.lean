/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Tape.RealNode.Batched
public import TorchLean.NN.Tape.DGraph
public import TorchLean.NN.Tape.RealNode.Arithmetic
public import TorchLean.NN.Tape.RealNode.Arm
public import TorchLean.NN.Op.Pointwise
public import TorchLean.NN.Op.Softmax
public import TorchLean.Tensor.Layers.Attention

/-!
# Multi-head self-attention

Derivative and backprop certificates for multi-head self-attention, decomposed into proven tape
nodes:
- linear projections via `matmul`,
- head split/merge via `reshape` and coordinate reindexing,
- attention core via batched `matmul`, coordinate reindexing, scaling, and row-wise softmax.

`dGraph` bundles the graph with the `FDerivCorrect` certificate of each node, so over `ℝ`
the general theorem `Tape.Graph.backpropVec_eq_fderiv` identifies its backprop with the
adjoint of its `fderiv`.

Here the Q/K/V split reindexes by `splitHeadsEquiv`, the coordinate map of
`TorchLean.MultiHeadAttention.splitHeads`, which reshapes `[n, numHeads * headDim]` to
`[n, numHeads, headDim]` and then swaps the token and head axes. For `n = numHeads = headDim = 2`,
the split lists the flat projection entries in the order `0,1,4,5,2,3,6,7`.
`toVec_splitHeads` and `toVec_combineHeads` prove that the split and merge
nodes compute `TorchLean.MultiHeadAttention.splitHeads` and
`TorchLean.MultiHeadAttention.combineHeads` on vectorized tensors. The certificate covers this
declared graph, and no result here relates it to `MultiHeadAttention.forward` or to the
executable attention implementation.

## PyTorch correspondence / citations
- The construction has the “project → split heads → scaled dot-product attention → concat heads →
  output projection” stages of `torch.nn.MultiheadAttention`, which splits the feature axis
  into heads in the same order.
  https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
- The core attention step corresponds to `torch.nn.functional.scaled_dot_product_attention`.
  https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
-/

@[expose] public section

open TorchLean.Shape (size)

namespace TorchLean

open scoped BigOperators

noncomputable section

namespace MultiHeadAttention

open Tape.RealNode
open Tape (DGraph)

/-- The shape `n×(numHeads*headDim)` of `n` rows of concatenated heads. With `n = dModel` it is the
shape of the Q/K/V projection weights. -/
abbrev flatHeadsShape (n numHeads headDim : Nat) : Shape :=
  .dim n (.dim (numHeads * headDim) .scalar)

/-- Split-head representation `(numHeads)×n×headDim`. -/
abbrev headsShape (n numHeads headDim : Nat) : Shape :=
  .dim numHeads (.dim n (.dim headDim .scalar))

/-- Key-transposed shape `(numHeads)×headDim×n` used for `Q Kᵀ`. -/
abbrev ktShape (n numHeads headDim : Nat) : Shape :=
  .dim numHeads (.dim headDim (.dim n .scalar))

/-- Attention scores shape `(numHeads)×n×n`. -/
abbrev scoresShape (n numHeads : Nat) : Shape := .dim numHeads (.dim n (.dim n .scalar))

/-- Intermediate shape after swapping axes for concatenation `n×numHeads×headDim`. -/
abbrev swappedShape (n numHeads headDim : Nat) : Shape :=
  .dim n (.dim numHeads (.dim headDim .scalar))

/-- Intermediate node output shapes (tape “saved tensors”) for the MHA graph. -/
abbrev ss (n dModel numHeads headDim : Nat) : List Shape :=
  [ flatHeadsShape n numHeads headDim
  , headsShape n numHeads headDim
  , flatHeadsShape n numHeads headDim
  , headsShape n numHeads headDim
  , ktShape n numHeads headDim
  , flatHeadsShape n numHeads headDim
  , headsShape n numHeads headDim
  , scoresShape n numHeads
  , scoresShape n numHeads
  , scoresShape n numHeads
  , headsShape n numHeads headDim
  , swappedShape n numHeads headDim
  , flatHeadsShape n numHeads headDim
  , [n, dModel]
  ]

/-- Saved tensors produced while projecting the input into query, key, and value heads. -/
abbrev ssProjections (n numHeads headDim : Nat) : List Shape :=
  [ flatHeadsShape n numHeads headDim
  , headsShape n numHeads headDim
  , flatHeadsShape n numHeads headDim
  , headsShape n numHeads headDim
  , ktShape n numHeads headDim
  , flatHeadsShape n numHeads headDim
  , headsShape n numHeads headDim
  ]

/-- Saved tensors through scaled query-key scores, before softmax is applied. -/
abbrev ssScores (n numHeads headDim : Nat) : List Shape :=
  ssProjections n numHeads headDim ++
    [ scoresShape n numHeads
    , scoresShape n numHeads
    ]

/-- Saved tensors through the head-wise attention output, before heads are merged. -/
abbrev ssAttention (n numHeads headDim : Nat) : List Shape :=
  ssScores n numHeads headDim ++
    [ scoresShape n numHeads
    , headsShape n numHeads headDim
    ]

/-- Output projection weight shape `(numHeads*headDim)×dModel`. -/
abbrev woShape (dModel numHeads headDim : Nat) : Shape :=
  .dim (numHeads * headDim) (.dim dModel .scalar)

/-- Input context shapes: `[x, Wq, Wk, Wv, Wo]`. -/
abbrev ctx (n dModel numHeads headDim : Nat) : List Shape :=
  [ [n, dModel]
  , flatHeadsShape dModel numHeads headDim  -- Wq
  , flatHeadsShape dModel numHeads headDim  -- Wk
  , flatHeadsShape dModel numHeads headDim  -- Wv
  , woShape dModel numHeads headDim  -- Wo
  ]

/-- Context index of the sequence input `x` in `ctx`. -/
def idxX {n dModel numHeads headDim : Nat} {ss : List Shape} :
    Idx (ctx n dModel numHeads headDim ++ ss) [n, dModel] :=
  ⟨⟨0, by simp [ctx]⟩, by simp [ctx]⟩

/-- Context index of `Wq` in `ctx`. -/
def idxWq {n dModel numHeads headDim : Nat} {ss : List Shape} :
    Idx (ctx n dModel numHeads headDim ++ ss) (flatHeadsShape dModel numHeads headDim) :=
  ⟨⟨1, by simp [ctx]⟩, by simp [ctx]⟩

/-- Context index of `Wk` in `ctx`. -/
def idxWk {n dModel numHeads headDim : Nat} {ss : List Shape} :
    Idx (ctx n dModel numHeads headDim ++ ss) (flatHeadsShape dModel numHeads headDim) :=
  ⟨⟨2, by simp [ctx]⟩, by simp [ctx]⟩

/-- Context index of `Wv` in `ctx`. -/
def idxWv {n dModel numHeads headDim : Nat} {ss : List Shape} :
    Idx (ctx n dModel numHeads headDim ++ ss) (flatHeadsShape dModel numHeads headDim) :=
  ⟨⟨3, by simp [ctx]⟩, by simp [ctx]⟩

/-- Context index of `Wo` in `ctx`. -/
def idxWo {n dModel numHeads headDim : Nat} {ss : List Shape} :
    Idx (ctx n dModel numHeads headDim ++ ss) (woShape dModel numHeads headDim) :=
  ⟨⟨4, by simp [ctx]⟩, by simp [ctx]⟩

/-- Reshaping `[n, numHeads * headDim]` directly to `[numHeads, n, headDim]` preserves the
element count. The count alone does not give the token and head axis exchange of
`splitHeads`, which `splitHeadsEquiv` gives. -/
theorem size_flatHeads_eq_size_heads (n numHeads headDim : Nat) :
    Shape.size (flatHeadsShape n numHeads headDim) =
      Shape.size (headsShape n numHeads headDim) := by
  -- `simp` multiplies out `Shape.size` and leaves a commutative-monoid identity.
  simp
  ac_rfl

/-- Splitting the feature axis into heads, `[n, numHeads, headDim]`, preserves the element
count of `[n, numHeads * headDim]`. -/
theorem size_swapped_eq_size_flatHeads (n numHeads headDim : Nat) :
    Shape.size (swappedShape n numHeads headDim) =
      Shape.size (flatHeadsShape n numHeads headDim) := by
  simp

/-- The coordinate map of `splitHeads`: reshape `[n, numHeads * headDim]` to
`[n, numHeads, headDim]`, then exchange the token and head axes. -/
def splitHeadsEquiv (n numHeads headDim : Nat) :
    Fin (Shape.size (flatHeadsShape n numHeads headDim)) ≃
      Fin (Shape.size (headsShape n numHeads headDim)) :=
  (finCongr (size_swapped_eq_size_flatHeads n numHeads headDim).symm).trans
    (swapAdjacentEquiv n numHeads (Shape.size (Shape.dim headDim .scalar)))

/-- Vectorizing `splitHeads` reindexes the vectorized input by `splitHeadsEquiv`. -/
theorem toVec_splitHeads {n numHeads headDim : Nat}
    (x : Tensor ℝ (flatHeadsShape n numHeads headDim)) :
    Tensor.toVec (splitHeads x numHeads headDim rfl) =
      LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ (splitHeadsEquiv n numHeads headDim)
        (Tensor.toVec x) := by
  dsimp only [splitHeads, Shape.swapAdjacentAtDepth]
  rw [toVec_permute_adjacentSwap_zero, toVec_reshape]
  rfl

/-- Vectorizing `combineHeads` exchanges the head and token axes of the vectorized heads
and casts the result to `[n, numHeads * headDim]`. -/
theorem toVec_combineHeads {n numHeads headDim : Nat}
    (heads : Tensor ℝ (headsShape n numHeads headDim)) :
    Tensor.toVec (combineHeads heads) =
      LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ
        (finCongr (size_swapped_eq_size_flatHeads n numHeads headDim))
        (LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ
          (swapAdjacentEquiv numHeads n (Shape.size (Shape.dim headDim .scalar)))
          (Tensor.toVec heads)) := by
  dsimp only [combineHeads]
  rw [toVec_reshape, toVec_permute_adjacentSwap_zero]

/-- Append a node that scales the last saved tensor by the constant `c`. -/
def appendScaleLast {Γ ss : List Shape} {s : Shape}
    (dg : DGraph Γ (ss ++ [s])) (c : ℝ) : DGraph Γ ((ss ++ [s]) ++ [s]) :=
  let hctx : Γ ++ ss ++ [s] = Γ ++ (ss ++ [s]) := by simp [List.append_assoc]
  let idx : Idx (Γ ++ (ss ++ [s])) s :=
    hctx ▸ Idx.last (Γ := Γ) (ss := ss) (τ := s)
  DGraph.snoc dg
    (Tape.LawfulNode.ofArm (Op.scale (s := s)) rfl (Γ := Γ ++ (ss ++ [s]))
      (idx, Tensor.scalar c, ())
        (Arm.adj_of_cert₁ fun xs δ => Op.scale_backward_eq_adjoint xs.1 δ xs.2.1))
    (FDerivCorrect.ofArm₁ (Γ := Γ ++ (ss ++ [s])) (s := s) (idx := idx)
        (cert := fun x δ => Op.scale_backward_eq_adjoint x δ (Tensor.scalar c)))

/-- Append a row-wise softmax node on the last saved batched matrix. -/
def appendBatchedSoftmaxLast {Γ ss : List Shape} {h m n : Nat}
    (dg : DGraph Γ (ss ++ [.dim h (.dim m (.dim n .scalar))])) :
    DGraph Γ
      ((ss ++ [.dim h (.dim m (.dim n .scalar))]) ++ [.dim h (.dim m (.dim n .scalar))]) :=
  let s : Shape := .dim h (.dim m (.dim n .scalar))
  let hctx : Γ ++ ss ++ [s] = Γ ++ (ss ++ [s]) := by simp [List.append_assoc]
  let idx : Idx (Γ ++ (ss ++ [s])) s :=
    hctx ▸ Idx.last (Γ := Γ) (ss := ss) (τ := s)
  DGraph.snoc dg
    (Tape.LawfulNode.ofArm (Op.softmax (s := .dim h (.dim m (.dim n .scalar)))
      ⟨2, Nat.lt_succ_self 2⟩) rfl (Γ := Γ ++ (ss ++ [s])) (idx, ())
      (Arm.adj_of_cert₁ fun xs δ => Op.softmax_backward_eq_adjoint _ xs.1 δ))
    (FDerivCorrect.ofArm₁ (Γ := Γ ++ (ss ++ [s])) (idx := idx)
      (cert := Op.softmax_backward_eq_adjoint (s := .dim h (.dim m (.dim n .scalar)))
        ⟨2, Nat.lt_succ_self 2⟩))

/-- Project the sequence input into query, transposed-key, and value head tensors. -/
def projectionDGraph {n dModel numHeads headDim : Nat} :
    DGraph (ctx n dModel numHeads headDim) (ssProjections n numHeads headDim) := by

  let dg0 : DGraph (ctx n dModel numHeads headDim) [] := DGraph.nil

  -- 1) Qbig := x * Wq
  let nodeQbig :
      Tape.LawfulNode (α := ℝ) Unit (ctx n dModel numHeads headDim)
        (flatHeadsShape n numHeads headDim) :=
    Tape.RealNode.matmul (Γ := ctx n dModel numHeads headDim)
      (m := n) (n := dModel) (p := numHeads * headDim)
      (A := idxX (n := n) (dModel := dModel) (numHeads := numHeads) (headDim := headDim) (ss := []))
      (B := idxWq (n := n) (dModel := dModel) (numHeads := numHeads) (headDim := headDim) (ss :=
        []))
  let dg1 :=
    DGraph.snoc (dg := dg0) (node := nodeQbig)
      (hn := Tape.RealNode.matmulFDeriv
        (Γ := ctx n dModel numHeads headDim)
        (m := n) (n := dModel) (p := numHeads * headDim)
        (A := idxX (n := n) (dModel := dModel) (numHeads := numHeads) (headDim := headDim) (ss :=
          []))
        (B := idxWq (n := n) (dModel := dModel) (numHeads := numHeads) (headDim := headDim) (ss :=
          [])))

  -- 2) Qheads := split Qbig into heads as `splitHeads` does
  let idxQbig :
      Idx (ctx n dModel numHeads headDim ++ [flatHeadsShape n numHeads headDim])
        (flatHeadsShape n numHeads headDim) :=
    Idx.last (Γ := ctx n dModel numHeads headDim) (ss := [])
      (τ := flatHeadsShape n numHeads headDim)
  let nodeQheads :
      Tape.LawfulNode (α := ℝ) Unit
        (ctx n dModel numHeads headDim ++ [flatHeadsShape n numHeads headDim])
        (headsShape n numHeads headDim) :=
    reindex (Γ := ctx n dModel numHeads headDim ++ [flatHeadsShape n numHeads headDim])
      (source := flatHeadsShape n numHeads headDim)
      (target := headsShape n numHeads headDim)
      idxQbig (splitHeadsEquiv n numHeads headDim)
  let dg2 :=
    DGraph.snoc (dg := dg1) (node := nodeQheads)
      (hn := reindexFDeriv
        (Γ := ctx n dModel numHeads headDim ++ [flatHeadsShape n numHeads headDim])
        (source := flatHeadsShape n numHeads headDim)
        (target := headsShape n numHeads headDim)
        idxQbig (splitHeadsEquiv n numHeads headDim))

  -- 3) Kbig := x * Wk
  let nodeKbig :
      Tape.LawfulNode (α := ℝ) Unit
        (ctx n dModel numHeads headDim ++
          [flatHeadsShape n numHeads headDim, headsShape n numHeads headDim])
        (flatHeadsShape n numHeads headDim) :=
    Tape.RealNode.matmul (Γ := ctx n dModel numHeads headDim ++ [flatHeadsShape n numHeads headDim,
      headsShape n numHeads headDim])
      (m := n) (n := dModel) (p := numHeads * headDim)
      (A := idxX (n := n) (dModel := dModel) (numHeads := numHeads) (headDim := headDim)
        (ss := [flatHeadsShape n numHeads headDim, headsShape n numHeads headDim]))
      (B := idxWk (n := n) (dModel := dModel) (numHeads := numHeads) (headDim := headDim)
        (ss := [flatHeadsShape n numHeads headDim, headsShape n numHeads headDim]))
  let dg3 :=
    DGraph.snoc (dg := dg2) (node := nodeKbig)
      (hn := Tape.RealNode.matmulFDeriv
        (Γ := ctx n dModel numHeads headDim ++
          [flatHeadsShape n numHeads headDim, headsShape n numHeads headDim])
        (m := n) (n := dModel) (p := numHeads * headDim)
        (A := idxX (n := n) (dModel := dModel) (numHeads := numHeads) (headDim := headDim)
          (ss := [flatHeadsShape n numHeads headDim, headsShape n numHeads headDim]))
        (B := idxWk (n := n) (dModel := dModel) (numHeads := numHeads) (headDim := headDim)
          (ss := [flatHeadsShape n numHeads headDim, headsShape n numHeads headDim])))

  -- 4) Kheads := split Kbig into heads
  let idxKbig :
      Idx
        (ctx n dModel numHeads headDim ++
          [flatHeadsShape n numHeads headDim, headsShape n numHeads headDim,
            flatHeadsShape n numHeads headDim])
        (flatHeadsShape n numHeads headDim) :=
    Idx.last (Γ := ctx n dModel numHeads headDim)
      (ss := [flatHeadsShape n numHeads headDim, headsShape n numHeads headDim])
      (τ := flatHeadsShape n numHeads headDim)
  let nodeKheads :
      Tape.LawfulNode (α := ℝ) Unit
        (ctx n dModel numHeads headDim ++
          [flatHeadsShape n numHeads headDim, headsShape n numHeads headDim,
            flatHeadsShape n numHeads headDim])
        (headsShape n numHeads headDim) :=
    reindex (Γ := ctx n dModel numHeads headDim ++
          [flatHeadsShape n numHeads headDim, headsShape n numHeads headDim,
            flatHeadsShape n numHeads headDim])
      (source := flatHeadsShape n numHeads headDim)
      (target := headsShape n numHeads headDim)
      idxKbig (splitHeadsEquiv n numHeads headDim)
  let dg4 :=
    DGraph.snoc (dg := dg3) (node := nodeKheads)
      (hn := reindexFDeriv
        (Γ := ctx n dModel numHeads headDim ++
          [flatHeadsShape n numHeads headDim, headsShape n numHeads headDim,
            flatHeadsShape n numHeads headDim])
        (source := flatHeadsShape n numHeads headDim)
        (target := headsShape n numHeads headDim)
        idxKbig (splitHeadsEquiv n numHeads headDim))

  -- 5) Kᵀ (per head): exchange the sequence and feature axes of `Kheads`.
  let idxKheads :
      Idx
        (ctx n dModel numHeads headDim ++
          [flatHeadsShape n numHeads headDim, headsShape n numHeads headDim,
            flatHeadsShape n numHeads headDim, headsShape n numHeads headDim])
        (headsShape n numHeads headDim) :=
    Idx.last (Γ := ctx n dModel numHeads headDim)
      (ss := [flatHeadsShape n numHeads headDim, headsShape n numHeads headDim,
        flatHeadsShape n numHeads headDim])
      (τ := headsShape n numHeads headDim)
  let transposeKeys :
      Fin (Shape.size (headsShape n numHeads headDim)) ≃
        Fin (Shape.size (ktShape n numHeads headDim)) :=
    mapOuterEquiv numHeads (swapAdjacentEquiv n headDim (Shape.size Shape.scalar))
  let nodeKt :
      Tape.LawfulNode (α := ℝ) Unit
        (ctx n dModel numHeads headDim ++
          [flatHeadsShape n numHeads headDim, headsShape n numHeads headDim,
            flatHeadsShape n numHeads headDim, headsShape n numHeads headDim])
        (ktShape n numHeads headDim) :=
    reindex
      (Γ := ctx n dModel numHeads headDim ++
        [flatHeadsShape n numHeads headDim, headsShape n numHeads headDim,
          flatHeadsShape n numHeads headDim, headsShape n numHeads headDim])
      (source := headsShape n numHeads headDim)
      (target := ktShape n numHeads headDim)
      idxKheads transposeKeys
  let dg5 :=
    DGraph.snoc (dg := dg4) (node := nodeKt)
      (hn := reindexFDeriv
        (Γ := ctx n dModel numHeads headDim ++
          [flatHeadsShape n numHeads headDim, headsShape n numHeads headDim,
            flatHeadsShape n numHeads headDim, headsShape n numHeads headDim])
        (source := headsShape n numHeads headDim)
        (target := ktShape n numHeads headDim)
        idxKheads transposeKeys)

  -- 6) Vbig := x * Wv
  let nodeVbig :
      Tape.LawfulNode (α := ℝ) Unit
        (ctx n dModel numHeads headDim ++
          [ flatHeadsShape n numHeads headDim, headsShape n numHeads headDim
          , flatHeadsShape n numHeads headDim, headsShape n numHeads headDim
          , ktShape n numHeads headDim
          ])
        (flatHeadsShape n numHeads headDim) :=
    Tape.RealNode.matmul
      (Γ := ctx n dModel numHeads headDim ++
        [ flatHeadsShape n numHeads headDim, headsShape n numHeads headDim
        , flatHeadsShape n numHeads headDim, headsShape n numHeads headDim
        , ktShape n numHeads headDim
        ])
      (m := n) (n := dModel) (p := numHeads * headDim)
      (A := idxX (n := n) (dModel := dModel) (numHeads := numHeads) (headDim := headDim)
        (ss :=
          [ flatHeadsShape n numHeads headDim, headsShape n numHeads headDim
          , flatHeadsShape n numHeads headDim, headsShape n numHeads headDim
          , ktShape n numHeads headDim
          ]))
      (B := idxWv (n := n) (dModel := dModel) (numHeads := numHeads) (headDim := headDim)
        (ss :=
          [ flatHeadsShape n numHeads headDim, headsShape n numHeads headDim
          , flatHeadsShape n numHeads headDim, headsShape n numHeads headDim
          , ktShape n numHeads headDim
          ]))
  let dg6 :=
    DGraph.snoc (dg := dg5) (node := nodeVbig)
      (hn := Tape.RealNode.matmulFDeriv
        (Γ := ctx n dModel numHeads headDim ++
          [ flatHeadsShape n numHeads headDim, headsShape n numHeads headDim
          , flatHeadsShape n numHeads headDim, headsShape n numHeads headDim
          , ktShape n numHeads headDim
          ])
        (m := n) (n := dModel) (p := numHeads * headDim)
        (A := idxX (n := n) (dModel := dModel) (numHeads := numHeads) (headDim := headDim)
          (ss :=
            [ flatHeadsShape n numHeads headDim, headsShape n numHeads headDim
            , flatHeadsShape n numHeads headDim, headsShape n numHeads headDim
            , ktShape n numHeads headDim
            ]))
        (B := idxWv (n := n) (dModel := dModel) (numHeads := numHeads) (headDim := headDim)
          (ss :=
            [ flatHeadsShape n numHeads headDim, headsShape n numHeads headDim
            , flatHeadsShape n numHeads headDim, headsShape n numHeads headDim
            , ktShape n numHeads headDim
            ])))

  -- 7) Vheads := split Vbig into heads
  let idxVbig :
      Idx
        (ctx n dModel numHeads headDim ++
          [ flatHeadsShape n numHeads headDim, headsShape n numHeads headDim
          , flatHeadsShape n numHeads headDim, headsShape n numHeads headDim
          , ktShape n numHeads headDim
          , flatHeadsShape n numHeads headDim
          ])
        (flatHeadsShape n numHeads headDim) :=
    Idx.last
      (Γ := ctx n dModel numHeads headDim)
      (ss :=
        [ flatHeadsShape n numHeads headDim, headsShape n numHeads headDim
        , flatHeadsShape n numHeads headDim, headsShape n numHeads headDim
        , ktShape n numHeads headDim
        ])
      (τ := flatHeadsShape n numHeads headDim)
  let nodeVheads :
      Tape.LawfulNode (α := ℝ) Unit
        (ctx n dModel numHeads headDim ++
          [ flatHeadsShape n numHeads headDim, headsShape n numHeads headDim
          , flatHeadsShape n numHeads headDim, headsShape n numHeads headDim
          , ktShape n numHeads headDim
          , flatHeadsShape n numHeads headDim
          ])
        (headsShape n numHeads headDim) :=
    reindex
      (Γ := ctx n dModel numHeads headDim ++
        [ flatHeadsShape n numHeads headDim, headsShape n numHeads headDim
        , flatHeadsShape n numHeads headDim, headsShape n numHeads headDim
        , ktShape n numHeads headDim
        , flatHeadsShape n numHeads headDim
        ])
      (source := flatHeadsShape n numHeads headDim)
      (target := headsShape n numHeads headDim)
      idxVbig (splitHeadsEquiv n numHeads headDim)
  let dg7 :=
    DGraph.snoc (dg := dg6) (node := nodeVheads)
      (hn := reindexFDeriv
        (Γ := ctx n dModel numHeads headDim ++
          [ flatHeadsShape n numHeads headDim, headsShape n numHeads headDim
          , flatHeadsShape n numHeads headDim, headsShape n numHeads headDim
          , ktShape n numHeads headDim
          , flatHeadsShape n numHeads headDim
          ])
        (source := flatHeadsShape n numHeads headDim)
        (target := headsShape n numHeads headDim)
        idxVbig (splitHeadsEquiv n numHeads headDim))

  exact dg7

/-- Form and scale the query-key score matrices for every head. -/
def scoresDGraph {n dModel numHeads headDim : Nat} (c : ℝ) :
    DGraph (ctx n dModel numHeads headDim) (ssScores n numHeads headDim) := by
  let dg7 := projectionDGraph (n := n) (dModel := dModel)
    (numHeads := numHeads) (headDim := headDim)

  -- 8) scores := Qheads * Kᵀ (batched matmul)
  let ss7 := ssProjections n numHeads headDim
  let idxQheads0 :
      Idx
        (ctx n dModel numHeads headDim ++
          [flatHeadsShape n numHeads headDim, headsShape n numHeads headDim])
        (headsShape n numHeads headDim) :=
    Idx.last (Γ := ctx n dModel numHeads headDim) (ss := [flatHeadsShape n numHeads headDim])
      (τ := headsShape n numHeads headDim)
  let idxQheads7 :
      Idx (ctx n dModel numHeads headDim ++ ss7) (headsShape n numHeads headDim) :=
    Idx.weaken
      (Γ := ctx n dModel numHeads headDim ++
        [flatHeadsShape n numHeads headDim, headsShape n numHeads headDim])
      idxQheads0
      (rest := [ flatHeadsShape n numHeads headDim, headsShape n numHeads headDim
               , ktShape n numHeads headDim
               , flatHeadsShape n numHeads headDim, headsShape n numHeads headDim ])
  let idxKt0 :
      Idx
        (ctx n dModel numHeads headDim ++
          [ flatHeadsShape n numHeads headDim, headsShape n numHeads headDim
          , flatHeadsShape n numHeads headDim, headsShape n numHeads headDim
          , ktShape n numHeads headDim
          ])
        (ktShape n numHeads headDim) :=
    Idx.last
      (Γ := ctx n dModel numHeads headDim)
      (ss := [ flatHeadsShape n numHeads headDim, headsShape n numHeads headDim
             , flatHeadsShape n numHeads headDim, headsShape n numHeads headDim ])
      (τ := ktShape n numHeads headDim)
  let idxKt7 :
      Idx (ctx n dModel numHeads headDim ++ ss7) (ktShape n numHeads headDim) :=
    Idx.weaken
      (Γ := ctx n dModel numHeads headDim ++
        [ flatHeadsShape n numHeads headDim, headsShape n numHeads headDim
        , flatHeadsShape n numHeads headDim, headsShape n numHeads headDim
        , ktShape n numHeads headDim
        ])
      idxKt0
      (rest := [flatHeadsShape n numHeads headDim, headsShape n numHeads headDim])
  let nodeScores :
      Tape.LawfulNode (α := ℝ) Unit (ctx n dModel numHeads headDim ++ ss7)
        (scoresShape n numHeads) :=
    Batched.matmul
      (Γ := ctx n dModel numHeads headDim ++ ss7)
      (h := numHeads) (m := n) (n := headDim) (p := n)
      (A := idxQheads7) (B := idxKt7)
  let dg8 :=
    DGraph.snoc (dg := dg7) (node := nodeScores)
      (hn := Batched.matmulFDeriv
        (Γ := ctx n dModel numHeads headDim ++ ss7)
        (h := numHeads) (m := n) (n := headDim) (p := n)
        (A := idxQheads7) (B := idxKt7))

  -- 9) scaled := scale scores c
  let dg9 := appendScaleLast (dg := dg8) c

  exact dg9

/-- Normalize the score rows and combine the resulting probabilities with the value heads. -/
def attentionDGraph {n dModel numHeads headDim : Nat} (c : ℝ) :
    DGraph (ctx n dModel numHeads headDim) (ssAttention n numHeads headDim) := by
  let dg9 := scoresDGraph (n := n) (dModel := dModel)
    (numHeads := numHeads) (headDim := headDim) c
  let ss7 := ssProjections n numHeads headDim

  -- 10) probs := batched softmax_last scaled
  let dg9' : DGraph (ctx n dModel numHeads headDim)
      ((ss7 ++ [scoresShape n numHeads]) ++ [scoresShape n numHeads]) := by
    exact dg9
  let dg10 := appendBatchedSoftmaxLast (dg := dg9')

  -- 11) headOut := probs * Vheads (batched matmul)
  let ss10 := ss7 ++
      [scoresShape n numHeads       -- scores
      , scoresShape n numHeads       -- scaled
      , scoresShape n numHeads]      -- probs
  let idxVheads0 :
      Idx (ctx n dModel numHeads headDim ++ ss7) (headsShape n numHeads headDim) :=
    Idx.last
      (Γ := ctx n dModel numHeads headDim)
      (ss :=
        [ flatHeadsShape n numHeads headDim
        , headsShape n numHeads headDim
        , flatHeadsShape n numHeads headDim
        , headsShape n numHeads headDim
        , ktShape n numHeads headDim
        , flatHeadsShape n numHeads headDim
        ])
      (τ := headsShape n numHeads headDim)
  let idxVheads10 :
      Idx (ctx n dModel numHeads headDim ++ ss10) (headsShape n numHeads headDim) :=
    Idx.weaken
      (Γ := ctx n dModel numHeads headDim ++ ss7)
      idxVheads0
      (rest :=
        [scoresShape n numHeads
        , scoresShape n numHeads
        , scoresShape n numHeads])
  let idxProbs :
      Idx (ctx n dModel numHeads headDim ++ ss10) (scoresShape n numHeads) :=
    Idx.last
      (Γ := ctx n dModel numHeads headDim)
      (ss := ss7 ++
        [scoresShape n numHeads, scoresShape n numHeads])
      (τ := scoresShape n numHeads)
  let nodeHeadOut :
      Tape.LawfulNode (α := ℝ) Unit (ctx n dModel numHeads headDim ++ ss10)
        (headsShape n numHeads headDim) :=
    Batched.matmul
      (Γ := ctx n dModel numHeads headDim ++ ss10)
      (h := numHeads) (m := n) (n := n) (p := headDim)
      (A := idxProbs) (B := idxVheads10)
  let dg11 :=
    DGraph.snoc (dg := dg10) (node := nodeHeadOut)
      (hn := Batched.matmulFDeriv
        (Γ := ctx n dModel numHeads headDim ++ ss10)
        (h := numHeads) (m := n) (n := n) (p := headDim)
        (A := idxProbs) (B := idxVheads10))

  exact dg11

/--
Multi-head self-attention as a proof-carrying graph.

It computes `x Wq Wk Wv Wo ↦ concat_heads (softmax(c * (Q Kᵀ)) V) * Wo`, with `Q`, `K`, and `V`
projected from `x`. Each projection is split into heads as
`TorchLean.MultiHeadAttention.splitHeads` does, and the output merge swaps the head and token axes
before flattening, as `TorchLean.MultiHeadAttention.combineHeads` does.
-/
def dGraph {n dModel numHeads headDim : Nat} (c : ℝ) :
    DGraph (ctx n dModel numHeads headDim) (MultiHeadAttention.ss n dModel numHeads headDim) := by
  let dg11 := attentionDGraph (n := n) (dModel := dModel)
    (numHeads := numHeads) (headDim := headDim) c
  let ss10 := ssProjections n numHeads headDim ++
    [ scoresShape n numHeads
    , scoresShape n numHeads
    , scoresShape n numHeads
    ]

  -- 12) Exchange the head and sequence axes before concatenating heads.
  let idxHeadOut :
      Idx (ctx n dModel numHeads headDim ++ ss10 ++ [headsShape n numHeads headDim])
        (headsShape n numHeads headDim) :=
    Idx.last (Γ := ctx n dModel numHeads headDim) (ss := ss10) (τ := headsShape n numHeads headDim)
  let swapHeadSequence :
      Fin (Shape.size (headsShape n numHeads headDim)) ≃
        Fin (Shape.size (swappedShape n numHeads headDim)) :=
    swapAdjacentEquiv numHeads n (Shape.size (Shape.dim headDim .scalar))
  let nodeSwapped :
      Tape.LawfulNode (α := ℝ) Unit
        (ctx n dModel numHeads headDim ++ ss10 ++ [headsShape n numHeads headDim])
        (swappedShape n numHeads headDim) :=
    reindex
      (Γ := ctx n dModel numHeads headDim ++ ss10 ++ [headsShape n numHeads headDim])
      (source := headsShape n numHeads headDim)
      (target := swappedShape n numHeads headDim)
      idxHeadOut swapHeadSequence
  let dg12 :=
    DGraph.snoc (dg := dg11) (node := nodeSwapped)
      (hn := reindexFDeriv
        (Γ := ctx n dModel numHeads headDim ++ ss10 ++ [headsShape n numHeads headDim])
        (source := headsShape n numHeads headDim)
        (target := swappedShape n numHeads headDim)
        idxHeadOut swapHeadSequence)

  -- 13) concat := reshape swapped to (n, numHeads*headDim)
  let idxSwapped :
      Idx
        (ctx n dModel numHeads headDim ++ ss10 ++
          [headsShape n numHeads headDim, swappedShape n numHeads headDim])
        (swappedShape n numHeads headDim) :=
    Idx.last
      (Γ := ctx n dModel numHeads headDim)
      (ss := ss10 ++ [headsShape n numHeads headDim])
      (τ := swappedShape n numHeads headDim)
  let nodeConcat :
      Tape.LawfulNode (α := ℝ) Unit
        (ctx n dModel numHeads headDim ++ ss10 ++
          [headsShape n numHeads headDim, swappedShape n numHeads headDim])
        (flatHeadsShape n numHeads headDim) :=
    reshape
      (Γ := ctx n dModel numHeads headDim ++ ss10 ++
        [headsShape n numHeads headDim, swappedShape n numHeads headDim])
      (s₁ := swappedShape n numHeads headDim)
      (s₂ := flatHeadsShape n numHeads headDim)
      idxSwapped (size_swapped_eq_size_flatHeads (n := n) (numHeads := numHeads)
        (headDim := headDim))
  let dg13 :=
    DGraph.snoc (dg := dg12) (node := nodeConcat)
      (hn := reshapeFDeriv
        (Γ := ctx n dModel numHeads headDim ++ ss10 ++
          [headsShape n numHeads headDim, swappedShape n numHeads headDim])
        (s₁ := swappedShape n numHeads headDim)
        (s₂ := flatHeadsShape n numHeads headDim)
        idxSwapped (size_swapped_eq_size_flatHeads (n := n) (numHeads := numHeads)
          (headDim := headDim)))

  -- 14) out := concat * Wo
  let idxConcat :
      Idx (ctx n dModel numHeads headDim ++ ss10 ++
        [ headsShape n numHeads headDim
        , swappedShape n numHeads headDim
        , flatHeadsShape n numHeads headDim
        ])
        (flatHeadsShape n numHeads headDim) :=
    Idx.last
      (Γ := ctx n dModel numHeads headDim)
      (ss := ss10 ++ [headsShape n numHeads headDim, swappedShape n numHeads headDim])
      (τ := flatHeadsShape n numHeads headDim)
  let nodeOut :
      Tape.LawfulNode (α := ℝ) Unit (ctx n dModel numHeads headDim ++ ss10 ++
        [ headsShape n numHeads headDim
        , swappedShape n numHeads headDim
        , flatHeadsShape n numHeads headDim
        ])
        [n, dModel] :=
    Tape.RealNode.matmul
      (Γ := ctx n dModel numHeads headDim ++ ss10 ++
        [ headsShape n numHeads headDim
        , swappedShape n numHeads headDim
        , flatHeadsShape n numHeads headDim
        ])
      (m := n) (n := numHeads * headDim) (p := dModel)
      (A := idxConcat)
      (B := idxWo (n := n) (dModel := dModel) (numHeads := numHeads) (headDim := headDim)
        (ss := ss10 ++
          [ headsShape n numHeads headDim
          , swappedShape n numHeads headDim
          , flatHeadsShape n numHeads headDim
          ]))
  let dg14 :=
    DGraph.snoc (dg := dg13) (node := nodeOut)
      (hn := Tape.RealNode.matmulFDeriv
        (Γ := ctx n dModel numHeads headDim ++ ss10 ++
          [ headsShape n numHeads headDim
          , swappedShape n numHeads headDim
          , flatHeadsShape n numHeads headDim
          ])
        (m := n) (n := numHeads * headDim) (p := dModel)
        (A := idxConcat)
        (B := idxWo (n := n) (dModel := dModel) (numHeads := numHeads) (headDim := headDim)
          (ss := ss10 ++
            [ headsShape n numHeads headDim
            , swappedShape n numHeads headDim
            , flatHeadsShape n numHeads headDim
            ])))

  exact dg14

end MultiHeadAttention

end

end TorchLean
