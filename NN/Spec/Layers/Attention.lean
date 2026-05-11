/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Activation

/-!
# Attention (spec layer)

This file defines the standard **scaled dot-product attention** primitive and a simple
multi-head wrapper.

`Attention(Q,K,V) = softmax(Q Kᵀ / √d) V`

TorchLean goal here is to mirror the math you see in deep learning libraries (especially PyTorch),
but keep everything as pure functions on `Spec.Tensor` so the same definitions can be reused for:

- proofs (e.g. reasoning about shapes and gradients),
- reference implementations (runtime extraction),
- verification backends (e.g. interval semantics).

## Shapes and conventions

We model the "single batch element" case. Batched attention is obtained by adding an outer `.dim B`
and mapping over it.

Core shapes:

- `Q : (nQ × d)` queries
- `K : (nK × d)` keys
- `V : (nK × dV)` values

In many transformer blocks `dV = d`, and this file uses that common choice for simplicity.

The optional Boolean mask has shape `(nQ × nK)`. In the main spec, masks use the true `-∞`
semantics: blocked entries receive zero numerator before row normalization, so their attention
weight is definitionally zero. This is the finite-scalar encoding of the PyTorch pattern
`scores.masked_fill(~mask, -torch.inf)`.


PyTorch analogy:

- `scaledDotProductAttention` corresponds to `torch.nn.functional.scaled_dot_product_attention`
  (no dropout), with Boolean masks interpreted as true `-∞` masks.
- `MultiHeadAttention.forward` corresponds to the core computation inside `nn.MultiheadAttention`
  / transformer blocks, ignoring biases and dropout.
-/

@[expose] public section


open Spec
open Tensor
open Shape
open MathFunctions
open Numbers

namespace Spec
open Tensor
open Shape

variable {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]

/-!
## Scaled Dot-Product Attention

We separate out the single-head primitive (`scaledDotProductAttention`) because:

- it is the core mathematical object, reused in multi-head attention,
- it is a good target for proofs and for "spec vs runtime" comparisons.
-/

/-!
## Boolean masks

TorchLean uses the same boolean mask convention as PyTorch SDPA:

- `true` means a key/value position is **allowed to be attended to**,
- `false` means it is blocked (its softmax numerator is exactly zero).

PyTorch reference: `torch.nn.functional.scaled_dot_product_attention` uses the same convention for
boolean `attn_mask` entries: `True` entries are included, and `False` entries are blocked.
-/

/-- A `(nQ × nK)` mask where every position is allowed (`true`). -/
def allTrueMask (nQ nK : Nat) : Tensor Bool (.dim nQ (.dim nK .scalar)) :=
  Tensor.dim (fun _ => Tensor.dim (fun _ => Tensor.scalar true))

/-- A `(nQ × nK)` mask where every position is blocked (`false`). -/
def allFalseMask (nQ nK : Nat) : Tensor Bool (.dim nQ (.dim nK .scalar)) :=
  Tensor.dim (fun _ => Tensor.dim (fun _ => Tensor.scalar false))

/-- Causal (lower-triangular) self-attention mask of shape `(n, n)`.

`mask[i,j] = true` iff `j ≤ i`, i.e. each query position can attend to itself and past positions.
-/
def causalMask (n : Nat) : Tensor Bool (.dim n (.dim n .scalar)) :=
  Tensor.dim (fun i =>
    Tensor.dim (fun j =>
      Tensor.scalar (decide (j.1 ≤ i.1))))

/-- Future-only (upper-triangular) self-attention mask of shape `(n, n)`.

This is the (strict) complement of `causal_mask`: `mask[i,j] = true` iff `i < j`.
-/
def futureMask (n : Nat) : Tensor Bool (.dim n (.dim n .scalar)) :=
  Tensor.dim (fun i =>
    Tensor.dim (fun j =>
      Tensor.scalar (decide (i.1 < j.1))))

/-- Bundled inputs and mask needed for scaled dot-product attention. -/
structure AttentionContext (α : Type) [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  (nQ nK dModel : Nat) (h1 : nQ ≠ 0) (h2 : nK ≠ 0) where
  Q : Tensor α (.dim nQ (.dim dModel .scalar))
  K : Tensor α (.dim nK (.dim dModel .scalar))
  V : Tensor α (.dim nK (.dim dModel .scalar))
  bc_sum_to_target :
    BroadcastTo
      (.dim nQ .scalar)
      (.dim nQ (.dim nK .scalar))
  mask : Option (Tensor Bool (.dim nQ (.dim nK .scalar)))

/-!
## Exact hard masking

TorchLean encodes the usual "true `-∞` before softmax" behavior without requiring the tensor scalar
type itself to contain infinities. Instead of replacing blocked logits by a finite sentinel, we form
softmax numerators directly:

`numerator_j = if mask_j then exp(score_j) else 0`.

This is exactly what `exp(-∞)=0` contributes to softmax. Blocked positions therefore have exactly
zero attention mass, which is the property causal proofs need.
-/

/-- Hard-masked softmax on one vector.

`mask[j] = false` makes the `j`-th numerator exactly zero before normalization. This is the
ordinary finite-scalar encoding of softmax with true `-∞` masked logits.

If every mask entry is false, the denominator is zero; the definition remains total because the
spec scalar division is total, but well-formed attention call sites should provide at least one
allowed key per row. Causal self-attention satisfies that by construction because each token can
attend to itself.
-/
def hardMaskedSoftmaxVecSpec {n : Nat}
    (scores : Tensor α (.dim n .scalar))
    (mask : Tensor Bool (.dim n .scalar)) :
    Tensor α (.dim n .scalar) :=
  let numerators : Tensor α (.dim n .scalar) :=
    map2Spec (fun score allowed => if allowed then MathFunctions.exp score else 0) scores mask
  let denom : α := sumSpec numerators
  divSpec numerators (replicate (Tensor.scalar denom))

/-- Row-wise hard-masked softmax for attention score matrices. -/
def hardMaskedSoftmaxSpec {nQ nK : Nat}
    (scores : Tensor α (.dim nQ (.dim nK .scalar)))
    (mask : Tensor Bool (.dim nQ (.dim nK .scalar))) :
    Tensor α (.dim nQ (.dim nK .scalar)) :=
  match scores, mask with
  | Tensor.dim scoreRows, Tensor.dim maskRows =>
      Tensor.dim (fun i => hardMaskedSoftmaxVecSpec (scoreRows i) (maskRows i))

/-- VJP/JVP helper for a softmax-like row-normalization when the forward weights are already known.

For ordinary softmax, `weights = softmax(scores)`. For hard-masked softmax, blocked entries have
`weights = 0`, and the same formula gives zero gradient through blocked logits:

`dScores = weights ⊙ (dWeights - Σⱼ dWeightsⱼ * weightsⱼ)`.
-/
def softmaxBackwardFromWeightsSpec : {s : Shape} → Tensor α s → Tensor α s → Tensor α s
  | .scalar, _weights, _dWeights => Tensor.scalar 0
  | .dim _n .scalar, weights, dWeights =>
      let rowDot : α := sumSpec (mulSpec dWeights weights)
      mulSpec weights (subSpec dWeights (replicate (Tensor.scalar rowDot)))
  | .dim n inner, Tensor.dim weightRows, Tensor.dim dWeightRows =>
      Tensor.dim (fun i : Fin n =>
        softmaxBackwardFromWeightsSpec (s := inner) (weightRows i) (dWeightRows i))

/-- Scaled dot-product attention (forward).

Given:

- `Q : (nQ × d)`, `K : (nK × d)`, `V : (nK × d)`,

we compute:

1. scores `S = Q Kᵀ` with shape `(nQ × nK)`
2. scaled scores `S' = S / √d`
3. (optional) mask: for each `(i,j)`, if `mask[i,j] = false`, its softmax numerator is exactly zero
   (the finite-scalar encoding of true `-∞` masking)
4. attention weights `A = softmax(S')` (softmax over the last axis, i.e. each query row sums to 1)
5. output `Out = A V` with shape `(nQ × d)`

Mask convention:

`mask[i,j] = true` means "this key position is allowed", and `false` means "mask it out".

PyTorch analogy: `torch.softmax(scores.masked_fill(~mask, -torch.inf), dim=-1)` row-wise, then a
final matrix multiply by `V`.
-/
def scaledDotProductAttention
  {nQ nK dModel : Nat} {h1 : nQ ≠ 0} {h2 : nK ≠ 0}
  (ctx : AttentionContext α nQ nK dModel h1 h2) :
  Tensor α (.dim nQ (.dim dModel .scalar)) :=
  let scale := MathFunctions.sqrt (dModel : α)
  let scores := matMulSpec ctx.Q (matrixTransposeSpec ctx.K)
  let scaledScores := scaleSpec scores (1 / scale)
  let attentionWeights :=
    match ctx.mask with
    | none => Activation.softmaxSpec scaledScores
    | some m => hardMaskedSoftmaxSpec scaledScores m
  matMulSpec attentionWeights ctx.V

/-- Alias documenting that the main attention spec uses exact hard-mask semantics. -/
def hardMaskedScaledDotProductAttention
  {nQ nK dModel : Nat} {h1 : nQ ≠ 0} {h2 : nK ≠ 0}
  (ctx : AttentionContext α nQ nK dModel h1 h2) :
  Tensor α (.dim nQ (.dim dModel .scalar)) :=
  scaledDotProductAttention ctx

/-- Backward/VJP for scaled dot-product attention.

Returns `(dQ, dK, dV)` given an upstream gradient `dOut`.

We recompute the forward intermediates locally so this spec stays self-contained and does not rely
on a global tape.

For masked calls, this is the VJP for true hard masking. Blocked logits have zero forward weight,
and `softmaxBackwardFromWeightsSpec` therefore gives zero gradient through those blocked positions.
 -/
def scaledDotProductAttentionBackward
  {nQ nK dModel : Nat} {h1 : nQ ≠ 0} {h2 : nK ≠ 0}
  (ctx : AttentionContext α nQ nK dModel h1 h2)
  (dOut : Tensor α (.dim nQ (.dim dModel .scalar))) :
  (Tensor α (.dim nQ (.dim dModel .scalar)) ×
   Tensor α (.dim nK (.dim dModel .scalar)) ×
   Tensor α (.dim nK (.dim dModel .scalar))) :=
  let scale := MathFunctions.sqrt (dModel : α)
  let scores := matMulSpec ctx.Q (matrixTransposeSpec ctx.K)
  let scaledScores := scaleSpec scores (1 / scale)
  let attentionWeights :=
    match ctx.mask with
    | none => Activation.softmaxSpec scaledScores
    | some m => hardMaskedSoftmaxSpec scaledScores m

  -- Backprop through `Out = A V`.
  let dAttentionWeights := matMulSpec dOut (matrixTransposeSpec ctx.V)
  let dV := matMulSpec (matrixTransposeSpec attentionWeights) dOut

  -- Backprop through row normalization. Hard-masked blocked entries already have zero weight, so
  -- their score gradients are zero by the formula.
  let dScaledScores := softmaxBackwardFromWeightsSpec attentionWeights dAttentionWeights

  -- Backprop through scaling: `scaledScores = scores * (1 / scale)`.
  let dScores := scaleSpec dScaledScores (1 / scale)

  -- Backprop through `scores = Q Kᵀ`.
  let dQ := matMulSpec dScores ctx.K
  let dK := matMulSpec (matrixTransposeSpec dScores) ctx.Q

  (dQ, dK, dV)

/--
Forward-mode JVP for scaled dot-product attention.

This differentiates the pure attention equation

`Out = softmax(mask(Q Kᵀ / sqrt(d))) V`

in the direction `(dQ,dK,dV)`. For hard-masked calls, blocked logits have zero forward weight, so
their tangent contribution is zero in `softmaxBackwardFromWeightsSpec`. The row-wise softmax
Jacobian is symmetric, so the same formula serves as both VJP and JVP once the forward weights are
known.
-/
def scaledDotProductAttentionJvp
  {nQ nK dModel : Nat} {h1 : nQ ≠ 0} {h2 : nK ≠ 0}
  (ctx : AttentionContext α nQ nK dModel h1 h2)
  (dQ : Tensor α (.dim nQ (.dim dModel .scalar)))
  (dK dV : Tensor α (.dim nK (.dim dModel .scalar))) :
  Tensor α (.dim nQ (.dim dModel .scalar)) :=
  let scale := MathFunctions.sqrt (dModel : α)
  let scores := matMulSpec ctx.Q (matrixTransposeSpec ctx.K)
  let dScores :=
    addSpec
      (matMulSpec dQ (matrixTransposeSpec ctx.K))
      (matMulSpec ctx.Q (matrixTransposeSpec dK))
  let scaledScores := scaleSpec scores (1 / scale)
  let dScaledScores := scaleSpec dScores (1 / scale)
  let attentionWeights :=
    match ctx.mask with
    | none => Activation.softmaxSpec scaledScores
    | some m => hardMaskedSoftmaxSpec scaledScores m
  let dAttentionWeights :=
    softmaxBackwardFromWeightsSpec attentionWeights dScaledScores
  addSpec (matMulSpec dAttentionWeights ctx.V) (matMulSpec attentionWeights dV)


/-
  Multi-Head Attention
  Splits input into multiple heads, applies attention, then combines
-/
  /-- Multi-head attention parameters (projection matrices).

PyTorch analogy: this corresponds to the four linear maps used in attention blocks:

- `Wq`, `Wk`, `Wv` project `dModel -> (numHeads * headDim)`
- `Wo` projects `(numHeads * headDim) -> dModel`

This spec keeps them as explicit matrices (no bias terms) to keep the math simple and to make the
gradients easy to audit.
-/
  structure MultiHeadAttention (α : Type) (numHeads dModel headDim : Nat) where
    /-- Wq. -/
    Wq : Tensor α (.dim dModel (.dim (numHeads * headDim) .scalar))
    -- Query projection: dModel × (numHeads * headDim)
    /-- Wk. -/
    Wk : Tensor α (.dim dModel (.dim (numHeads * headDim) .scalar))  -- Key projection
    /-- Wv. -/
    Wv : Tensor α (.dim dModel (.dim (numHeads * headDim) .scalar))  -- Value projection
    /-- Wo. -/
    Wo : Tensor α (.dim (numHeads * headDim) (.dim dModel .scalar))
    -- Output projection: (numHeads * headDim) × dModel

/-
  Split tensor into multiple attention heads
-/
  /-- Split `(n, dModel)` into `(numHeads, n, headDim)` by reshaping.

We store heads as the outermost axis so that "per-head computation" is just a `Tensor.dim` over
`Fin numHeads`.

PyTorch analogy: conceptually similar to reshaping `(n, numHeads*headDim)` into
`(n, numHeads, headDim)` and then transposing to make heads a separate axis; here we go directly to
`(numHeads, n, headDim)` because it is convenient for later definitions.
-/
  def splitHeadsSpec
    {α : Type} [Inhabited α]
    {n dModel : Nat}
  (x : Tensor α (.dim n (.dim dModel .scalar)))
  (numHeads headDim : Nat)
  (h : dModel = numHeads * headDim)
  : Tensor α (.dim numHeads (.dim n (.dim headDim .scalar))) :=
  let s₁ := .dim n (.dim dModel .scalar)
  let s₂ := .dim numHeads (.dim n (.dim headDim .scalar))
  have size_eq : Shape.size s₁ = Shape.size s₂ := Shape.size_eq_of_dModel_eq_numHeads_mul_headDim n
    numHeads dModel headDim h
  reshapeSpec x size_eq

/-
  Concatenate attention heads back into single tensor
-/
  /-- Concatenate a list of `numHeads` head tensors into a single `(n, numHeads*headDim)` tensor.

This is a straightforward list-based definition. The newer `combine_heads_spec` below does the same
thing starting from a tensor-of-heads representation.
-/
  def concatHeadsSpec {n numHeads headDim : Nat}
    (heads : List (Tensor α (.dim n (.dim headDim .scalar))))
    (h : heads.length = numHeads) :
    Tensor α (.dim n (.dim (numHeads * headDim) .scalar)) :=
    concatSpec numHeads heads h

/-!
`concat_heads_spec` above is the original (list-based) definition.

For proofs/automation, it's often easier to work with a **tensor of heads**
`Tensor α (.dim numHeads (.dim n (.dim headDim .scalar)))` and then use shape-only transforms
to combine heads back into a single `(n, numHeads*headDim)` tensor.
-/

  /-- Combine a tensor-of-heads back into a single `(n, numHeads*headDim)` tensor.

Implementation detail:

1. `swap_first_two_spec` converts `(numHeads, n, headDim)` into `(n, numHeads, headDim)`
2. `reshape_spec` flattens the last two axes into `(n, numHeads*headDim)`
-/
  def combineHeadsSpec
    {α : Type} [Context α]
    {n numHeads headDim : Nat}
    (heads : Tensor α (.dim numHeads (.dim n (.dim headDim .scalar)))) :
    Tensor α (.dim n (.dim (numHeads * headDim) .scalar)) :=
  let swapped : Tensor α (.dim n (.dim numHeads (.dim headDim .scalar))) :=
    Tensor.swapFirstTwoSpec heads
  let s₁ : Shape := .dim n (.dim numHeads (.dim headDim .scalar))
  let s₂ : Shape := .dim n (.dim (numHeads * headDim) .scalar)
  have hSize : Shape.size s₁ = Shape.size s₂ := by
    simp [s₁, s₂, Shape.size]
  reshapeSpec swapped hSize

  /-- Convenience proof that `(n)` broadcasts to `(n,n)`.

  This is kept as a small helper because some attention-style proofs and wrappers want an explicit
  `BroadcastTo` witness rather than relying on typeclass search.
  -/
  @[reducible]
  def buildBcProof (n : Nat) : BroadcastTo (Shape.dim n Shape.scalar) (Shape.dim n (Shape.dim n
    Shape.scalar)) :=
  -- Shape.dim n Shape.scalar broadcasts to Shape.dim n (Shape.dim n Shape.scalar)
  -- by first broadcasting Shape.scalar to Shape.dim n Shape.scalar
  broadcastToExpandDims

  /-- Multi-head attention forward pass (self-attention when `mask` is square).

High-level structure (PyTorch mental model):

1. project `x` into `Q,K,V`
2. split the projection dimension into heads
3. run scaled dot-product attention per head (sharing the same mask)
4. combine heads back and project with `Wo`
-/
  def MultiHeadAttention.forward
    {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
    {numHeads dModel headDim : Nat}
  (n : Nat) (h1 : n ≠ 0)
  (mha : MultiHeadAttention α numHeads dModel headDim)
  (x : Tensor α (.dim n (.dim dModel .scalar)))
  (mask : Option (Tensor Bool (.dim n (.dim n Shape.scalar)))) :
  Tensor α (.dim n (.dim dModel .scalar)) :=

  let h : numHeads * headDim = numHeads * headDim := by rfl

  -- Project inputs to big Q, K, V
  let Q := matMulSpec x mha.Wq
  let K := matMulSpec x mha.Wk
  let V := matMulSpec x mha.Wv

    -- Split heads: we represent heads as the outer axis `(numHeads, n, headDim)`.
    let QHeads := splitHeadsSpec Q numHeads headDim h
    let KHeads := splitHeadsSpec K numHeads headDim h
    let VHeads := splitHeadsSpec V numHeads headDim h

  -- Compute attention per head as a tensor indexed by `Fin numHeads`.
  let attentionHeads : Tensor α (.dim numHeads (.dim n (.dim headDim .scalar))) :=
    match QHeads, KHeads, VHeads with
    | Tensor.dim qF, Tensor.dim kF, Tensor.dim vF =>
        Tensor.dim (fun headIdx =>
          let ctx : AttentionContext α n n headDim h1 h1 :=
            { Q := qF headIdx
              K := kF headIdx
              V := vF headIdx
              bc_sum_to_target := buildBcProof n
              mask := mask }
          scaledDotProductAttention ctx)

    -- Combine heads back to `(n, numHeads * headDim)`.
    let concatenated := combineHeadsSpec (α := α) (n := n) (numHeads := numHeads) (headDim :=
      headDim) attentionHeads

    -- Final output projection: back to (seqLen, dModel)
    matMulSpec concatenated mha.Wo

  /-- Multi-head attention backward pass.

  Returns gradients for input `x` and all projection matrices `(Wq,Wk,Wv,Wo)`.
  We recompute forward intermediates locally so we don’t rely on a global tape.
  -/
def MultiHeadAttentionBackward
  {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  {n numHeads dModel headDim : Nat} (h1 : n ≠ 0)
  (mha : MultiHeadAttention α numHeads dModel headDim)
  (x : Tensor α (.dim n (.dim dModel .scalar)))
  (mask : Option (Tensor Bool (.dim n (.dim n .scalar))))
  (grad_output : Tensor α (.dim n (.dim dModel .scalar))) :
  ( Tensor α (.dim n (.dim dModel .scalar))      -- ∂L/∂x
  × Tensor α (.dim dModel (.dim (numHeads * headDim) .scalar))  -- ∂L/∂Wq
  × Tensor α (.dim dModel (.dim (numHeads * headDim) .scalar))  -- ∂L/∂Wk
  × Tensor α (.dim dModel (.dim (numHeads * headDim) .scalar))  -- ∂L/∂Wv
  × Tensor α (.dim (numHeads * headDim) (.dim dModel .scalar))  -- ∂L/∂Wo
  ) :=

  -- Forward recomputation for intermediate values:
  let Q := matMulSpec x mha.Wq
  let K := matMulSpec x mha.Wk
  let V := matMulSpec x mha.Wv

  let h : numHeads * headDim = numHeads * headDim := by rfl
  let QHeads := splitHeadsSpec Q numHeads headDim h
  let KHeads := splitHeadsSpec K numHeads headDim h
  let VHeads := splitHeadsSpec V numHeads headDim h

  let attentionHeads : Tensor α (.dim numHeads (.dim n (.dim headDim .scalar))) :=
    match QHeads, KHeads, VHeads with
    | Tensor.dim qF, Tensor.dim kF, Tensor.dim vF =>
        Tensor.dim (fun headIdx =>
          let ctx : AttentionContext α n n headDim h1 h1 :=
            { Q := qF headIdx
              K := kF headIdx
              V := vF headIdx
              bc_sum_to_target := buildBcProof n
              mask := mask }
          scaledDotProductAttention ctx)

  let concatenated := combineHeadsSpec (α := α) (n := n) (numHeads := numHeads) (headDim :=
    headDim) attentionHeads

  -- Backprop through output projection Wo:
  let (grad_concat, grad_Wo) := matMulBackwardSpec concatenated mha.Wo grad_output

  -- Backprop through combine-heads (reshape/swap):
  let grad_attentionHeads := splitHeadsSpec grad_concat numHeads headDim h

  -- Backprop through each head's scaledDotProductAttention:
  let (grad_QHeads, grad_KHeads, grad_VHeads) :
      Tensor α (.dim numHeads (.dim n (.dim headDim .scalar))) ×
        Tensor α (.dim numHeads (.dim n (.dim headDim .scalar))) ×
        Tensor α (.dim numHeads (.dim n (.dim headDim .scalar))) :=
    match QHeads, KHeads, VHeads, grad_attentionHeads with
    | Tensor.dim qF, Tensor.dim kF, Tensor.dim vF, Tensor.dim dF =>
        let gQ : Fin numHeads → Tensor α (.dim n (.dim headDim .scalar)) :=
          fun headIdx =>
            let ctx : AttentionContext α n n headDim h1 h1 :=
              { Q := qF headIdx
                K := kF headIdx
                V := vF headIdx
                bc_sum_to_target := buildBcProof n
                mask := mask }
            (scaledDotProductAttentionBackward ctx (dF headIdx)).1
        let gK : Fin numHeads → Tensor α (.dim n (.dim headDim .scalar)) :=
          fun headIdx =>
            let ctx : AttentionContext α n n headDim h1 h1 :=
              { Q := qF headIdx
                K := kF headIdx
                V := vF headIdx
                bc_sum_to_target := buildBcProof n
                mask := mask }
            (scaledDotProductAttentionBackward ctx (dF headIdx)).2.1
        let gV : Fin numHeads → Tensor α (.dim n (.dim headDim .scalar)) :=
          fun headIdx =>
            let ctx : AttentionContext α n n headDim h1 h1 :=
              { Q := qF headIdx
                K := kF headIdx
                V := vF headIdx
                bc_sum_to_target := buildBcProof n
                mask := mask }
            (scaledDotProductAttentionBackward ctx (dF headIdx)).2.2
        (Tensor.dim gQ, Tensor.dim gK, Tensor.dim gV)

  -- Backprop through split heads (reshape) for Q, K, V:
  let grad_Q := combineHeadsSpec (α := α) (n := n) (numHeads := numHeads) (headDim := headDim)
    grad_QHeads
  let grad_K := combineHeadsSpec (α := α) (n := n) (numHeads := numHeads) (headDim := headDim)
    grad_KHeads
  let grad_V := combineHeadsSpec (α := α) (n := n) (numHeads := numHeads) (headDim := headDim)
    grad_VHeads

  -- Backprop through input projections:
  let (grad_x_Q, grad_Wq) := matMulBackwardSpec x mha.Wq grad_Q
  let (grad_x_K, grad_Wk) := matMulBackwardSpec x mha.Wk grad_K
  let (grad_x_V, grad_Wv) := matMulBackwardSpec x mha.Wv grad_V

  -- Sum grads w.r.t. x from Q, K, V branches:
  let grad_x := addSpec (addSpec grad_x_Q grad_x_K) grad_x_V

  (grad_x, grad_Wq, grad_Wk, grad_Wv, grad_Wo)

/--
Forward-mode JVP for multi-head attention.

The rule follows the same computational graph as `MultiHeadAttention.forward`:

1. project tangents through `Q/K/V`,
2. split primal and tangent projections into heads,
3. apply `scaledDotProductAttentionJvp` head-wise,
4. combine head tangents, then differentiate the final output projection.

This keeps attention forward-mode AD explicit at the spec layer instead of hiding it behind a
runtime-only implementation.
-/
def MultiHeadAttentionJvp
  {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  {n numHeads dModel headDim : Nat} (h1 : n ≠ 0)
  (mha dmha : MultiHeadAttention α numHeads dModel headDim)
  (x dx : Tensor α (.dim n (.dim dModel .scalar)))
  (mask : Option (Tensor Bool (.dim n (.dim n .scalar)))) :
  Tensor α (.dim n (.dim dModel .scalar)) :=

  let Q := matMulSpec x mha.Wq
  let K := matMulSpec x mha.Wk
  let V := matMulSpec x mha.Wv

  let dQ := addSpec (matMulSpec dx mha.Wq) (matMulSpec x dmha.Wq)
  let dK := addSpec (matMulSpec dx mha.Wk) (matMulSpec x dmha.Wk)
  let dV := addSpec (matMulSpec dx mha.Wv) (matMulSpec x dmha.Wv)

  let h : numHeads * headDim = numHeads * headDim := by rfl
  let QHeads := splitHeadsSpec Q numHeads headDim h
  let KHeads := splitHeadsSpec K numHeads headDim h
  let VHeads := splitHeadsSpec V numHeads headDim h
  let dQHeads := splitHeadsSpec dQ numHeads headDim h
  let dKHeads := splitHeadsSpec dK numHeads headDim h
  let dVHeads := splitHeadsSpec dV numHeads headDim h

  let attentionHeads : Tensor α (.dim numHeads (.dim n (.dim headDim .scalar))) :=
    match QHeads, KHeads, VHeads with
    | Tensor.dim qF, Tensor.dim kF, Tensor.dim vF =>
        Tensor.dim (fun headIdx =>
          let ctx : AttentionContext α n n headDim h1 h1 :=
            { Q := qF headIdx
              K := kF headIdx
              V := vF headIdx
              bc_sum_to_target := buildBcProof n
              mask := mask }
          scaledDotProductAttention ctx)

  let dAttentionHeads : Tensor α (.dim numHeads (.dim n (.dim headDim .scalar))) :=
    match QHeads, KHeads, VHeads, dQHeads, dKHeads, dVHeads with
    | Tensor.dim qF, Tensor.dim kF, Tensor.dim vF, Tensor.dim dqF, Tensor.dim dkF,
        Tensor.dim dvF =>
        Tensor.dim (fun headIdx =>
          let ctx : AttentionContext α n n headDim h1 h1 :=
            { Q := qF headIdx
              K := kF headIdx
              V := vF headIdx
              bc_sum_to_target := buildBcProof n
              mask := mask }
          scaledDotProductAttentionJvp ctx (dqF headIdx) (dkF headIdx) (dvF headIdx))

  let concatenated := combineHeadsSpec (α := α) (n := n) (numHeads := numHeads) (headDim :=
    headDim) attentionHeads
  let dConcatenated := combineHeadsSpec (α := α) (n := n) (numHeads := numHeads) (headDim :=
    headDim) dAttentionHeads

  addSpec (matMulSpec dConcatenated mha.Wo) (matMulSpec concatenated dmha.Wo)

/-- Self-attention on a single sequence.

This uses the same input `x` for Q/K/V, runs scaled dot-product attention, then applies the output
projection `Wo`.

PyTorch mental model: the core of `nn.MultiheadAttention` / `TransformerEncoderLayer` (ignoring the
batch axis).
-/
def selfAttention
  {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  {n dModel projDim : Nat}
  (x : Tensor α (.dim n (.dim dModel .scalar)))
  (Wq : Tensor α (.dim dModel (.dim projDim .scalar)))
  (Wk : Tensor α (.dim dModel (.dim projDim .scalar)))
  (Wv : Tensor α (.dim dModel (.dim projDim .scalar)))
  (Wo : Tensor α (.dim projDim (.dim dModel .scalar)))
  (h1 : n ≠ 0) :
  Tensor α (.dim n (.dim dModel .scalar)) := by
  let Q := matMulSpec x Wq
  let K := matMulSpec x Wk
  let V := matMulSpec x Wv
  let ctx : AttentionContext α n n projDim h1 h1 :=
    { Q := Q, K := K, V := V,
      bc_sum_to_target := inferInstance,
      mask := none }
  exact matMulSpec (scaledDotProductAttention ctx) Wo


/-- Cross-attention between two sequences.

`query` is length `n1` and attends to `key/value` of length `n2`.

PyTorch mental model: the attention block in a Transformer decoder layer (`nn.MultiheadAttention`
with distinct query and key/value inputs).
-/
def crossAttention {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  {n1 n2 dModel projDim : Nat}
  (query : Tensor α (.dim n1 (.dim dModel .scalar)))
  (key   : Tensor α (.dim n2 (.dim dModel .scalar)))
  (value : Tensor α (.dim n2 (.dim dModel .scalar)))
  (Wq : Tensor α (.dim dModel (.dim projDim .scalar)))
  (Wk : Tensor α (.dim dModel (.dim projDim .scalar)))
  (Wv : Tensor α (.dim dModel (.dim projDim .scalar)))
  (Wo : Tensor α (.dim projDim (.dim dModel .scalar)))
  (h1 : n1 ≠ 0) (h2 : n2 ≠ 0) :
  Tensor α (.dim n1 (.dim dModel .scalar)) :=
  let Q := matMulSpec query Wq
  let K := matMulSpec key Wk
  let V := matMulSpec value Wv
  let ctx : AttentionContext α n1 n2 projDim h1 h2 :=
    { Q := Q, K := K, V := V,
      bc_sum_to_target := inferInstance,
      mask := none }
  let attention := scaledDotProductAttention ctx
  matMulSpec attention Wo

/--
  Sparse Attention
  Uses sparse attention patterns for efficiency
-/
def sparseAttention {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  {n dModel projDim : Nat}
  (x : Tensor α (.dim n (.dim dModel .scalar)))
  (sparsityPattern : Tensor Bool (.dim n (.dim n .scalar)))
  (Wq : Tensor α (.dim dModel (.dim projDim .scalar)))
  (Wk : Tensor α (.dim dModel (.dim projDim .scalar)))
  (Wv : Tensor α (.dim dModel (.dim projDim .scalar)))
  (Wo : Tensor α (.dim projDim (.dim dModel .scalar)))
  (h1 : n ≠ 0) :
  Tensor α (.dim n (.dim dModel .scalar)) :=
  let Q := matMulSpec x Wq
  let K := matMulSpec x Wk
  let V := matMulSpec x Wv
  let ctx : AttentionContext α n n projDim h1 h1 :=
    { Q := Q, K := K, V := V,
      bc_sum_to_target := inferInstance,
      mask := sparsityPattern }
  let attention := scaledDotProductAttention ctx
  matMulSpec attention Wo

end Spec
