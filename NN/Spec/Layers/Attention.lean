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
but keep everything as pure functions on `TorchLean.Tensor` so the same definitions can be reused
for:

- proofs (e.g. reasoning about shapes and gradients),
- reference implementations (runtime extraction),
- verification backends (e.g. interval semantics).

## Shapes and conventions

We model the "single batch element" case. Batched attention is obtained by prepending `[B]`
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

Rows with no allowed entries evaluate to the zero vector. This total convention agrees with the
native TorchLean and SDPA paths and avoids the undefined `0 / 0` normalization of an empty row.


PyTorch analogy:

- `scaledDotProductAttention` corresponds to `torch.nn.functional.scaled_dot_product_attention`
  (no dropout), with Boolean masks interpreted as true `-∞` masks.
- `MultiHeadAttention.forward` corresponds to the core computation inside `nn.MultiheadAttention`
  / transformer blocks, ignoring biases and dropout.
-/

@[expose] public section


open Spec TorchLean
open TorchLean TorchLean.Tensor
open Shape
open MathFunctions

open TorchLean

namespace Spec
open TorchLean TorchLean.Tensor
open Shape

variable {α : Type} [TorchLean.Storage α] [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)]

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

If an entire row is `false`, every output weight in that row is zero.

PyTorch reference: `torch.nn.functional.scaled_dot_product_attention` uses the same convention for
boolean `attn_mask` entries: `True` entries are included, and `False` entries are blocked.
-/

/-- A `(nQ × nK)` mask where every position is allowed (`true`). -/
def allTrueMask (nQ nK : Nat) : Tensor Bool [nQ, nK] :=
  Tensor.dim (fun _ => Tensor.dim (fun _ => Tensor.scalar true))

/-- A `(nQ × nK)` mask where every position is blocked (`false`). -/
def allFalseMask (nQ nK : Nat) : Tensor Bool [nQ, nK] :=
  Tensor.dim (fun _ => Tensor.dim (fun _ => Tensor.scalar false))

/-- Causal (lower-triangular) self-attention mask of shape `(n, n)`.

`mask[i,j] = true` iff `j ≤ i`, i.e. each query position can attend to itself and past positions.
-/
def causalMask (n : Nat) : Tensor Bool [n, n] :=
  Tensor.dim (fun i =>
    Tensor.dim (fun j =>
      Tensor.scalar (decide (j.1 ≤ i.1))))

/-- Future-only (upper-triangular) self-attention mask of shape `(n, n)`.

This is the (strict) complement of `causalMask`: `mask[i,j] = true` iff `i < j`.
-/
def futureMask (n : Nat) : Tensor Bool [n, n] :=
  Tensor.dim (fun i =>
    Tensor.dim (fun j =>
      Tensor.scalar (decide (i.1 < j.1))))

/-- Bundled inputs and mask needed for scaled dot-product attention. -/
structure AttentionContext (α : Type) [TorchLean.Storage α] [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)]
  (nQ nK dModel : Nat) (h1 : nQ ≠ 0) (h2 : nK ≠ 0) where
  Q : Tensor α [nQ, dModel]
  K : Tensor α [nK, dModel]
  V : Tensor α [nK, dModel]
  mask : Option (Tensor Bool [nQ, nK])

/-- Denominator used by scaled dot-product attention.

Standard attention requires a positive feature dimension and divides scores by `sqrt(dModel)`.
TorchLean's tensor shapes also admit zero dimensions. In that degenerate case the result has no
feature coordinates, so choosing denominator `1` gives the unique empty-feature result without
introducing a division by zero. -/
def attentionScaleDenom (dModel : Nat) : α :=
  if dModel = 0 then 1 else MathFunctions.sqrt (dModel : α)

/-!
## Exact hard masking

TorchLean encodes the usual "true `-∞` before softmax" behavior without requiring the tensor scalar
type itself to contain infinities. Instead of replacing blocked logits by a finite sentinel, we form
stable softmax numerators directly. If `rowMax` is the greatest allowed score in the row, then

`numerator_j = if mask_j then exp(score_j - rowMax) else 0`.

This is exactly what `exp(-∞)=0` contributes to softmax. Blocked positions therefore have exactly
zero attention mass, which is the property causal proofs need. A row with no allowed positions is
defined to contain only zeros.
-/

/-- Maximum allowed score in one hard-masked row, or `none` when every entry is blocked. -/
def hardMaskedMax? {n : Nat}
    (scores : Tensor α [n])
    (mask : Tensor Bool [n]) : Option α :=
  (List.finRange n).foldl (fun best i =>
    let score := Tensor.getScalar scores i
    let allowed := Tensor.getScalar mask i
    if allowed then
      match best with
      | none => some score
      | some current => some (if score > current then score else current)
    else
      best) none

/-- Hard-masked softmax on one vector.

`mask[j] = false` makes the `j`-th numerator exactly zero before normalization. This is the
ordinary finite-scalar encoding of softmax with true `-∞` masked logits.

The maximum and denominator are computed only over allowed entries. Subtracting the allowed-row
maximum gives the usual numerically stable softmax formula. If every mask entry is false, the result
is the zero vector, matching PyTorch SDPA and TorchLean's native CUDA providers.
-/
def hardMaskedSoftmaxVecSpec {n : Nat}
    (scores : Tensor α [n])
    (mask : Tensor Bool [n]) :
    Tensor α [n] :=
  match hardMaskedMax? scores mask with
  | none => replicate (Tensor.scalar 0)
  | some rowMax =>
      let numerators : Tensor α [n] :=
        map2Spec
          (fun score allowed =>
            if allowed then MathFunctions.exp (score - rowMax) else 0)
          scores mask
      let denom : α := sumSpec numerators
      divSpec numerators (replicate (Tensor.scalar denom))

/-- Hard-masked softmax along the innermost axis of an arbitrary tensor. -/
def hardMaskedSoftmaxSpec : {s : Shape} → Tensor α s → Tensor Bool s → Tensor α s
  | .scalar, _scores, mask =>
      Tensor.scalar (if mask.item then 1 else 0)
  | .dim _ .scalar, scores, mask => hardMaskedSoftmaxVecSpec scores mask
  | .dim n inner, scores, mask =>
      Tensor.dim (fun i : Fin n =>
        hardMaskedSoftmaxSpec (s := inner)
          (Tensor.unstack scores i) (Tensor.unstack mask i))

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
  | .dim n inner, weights, dWeights =>
      Tensor.dim (fun i : Fin n =>
        softmaxBackwardFromWeightsSpec (s := inner)
          (Tensor.unstack weights i) (Tensor.unstack dWeights i))

/-- Scaled dot-product attention (forward).

Given:

- `Q : (nQ × d)`, `K : (nK × d)`, `V : (nK × d)`,

we compute:

1. scores `S = Q Kᵀ` with shape `(nQ × nK)`
2. scaled scores `S' = S / √d`
3. (optional) mask: for each `(i,j)`, if `mask[i,j] = false`, its softmax numerator is exactly zero
   (the finite-scalar encoding of true `-∞` masking)
4. attention weights `A` by row normalization over the last axis
5. output `Out = A V` with shape `(nQ × d)`

Mask convention:

`mask[i,j] = true` means "this key position is allowed", and `false` means "mask it out".

For unmasked attention, each attention row sums to `1`. A masked row with at least one allowed key
has the same normalization. A fully blocked row is defined to have all-zero weights, matching
PyTorch SDPA and avoiding a `0/0` result.

PyTorch analogy: `torch.softmax(scores.masked_fill(~mask, -torch.inf), dim=-1)` row-wise, then a
final matrix multiply by `V`.
-/
def scaledDotProductAttention
  {nQ nK dModel : Nat} {h1 : nQ ≠ 0} {h2 : nK ≠ 0}
  (ctx : AttentionContext α nQ nK dModel h1 h2) :
  Tensor α [nQ, dModel] :=
  let scale := attentionScaleDenom (α := α) dModel
  let scores := matMulSpec ctx.Q (swapAdjacentAxes ctx.K 0)
  let scaledScores := scaleSpec scores (1 / scale)
  let attentionWeights :=
    match ctx.mask with
    | none => Activation.softmaxSpec 1 scaledScores
    | some m => hardMaskedSoftmaxSpec scaledScores m
  matMulSpec attentionWeights ctx.V

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
  (dOut : Tensor α [nQ, dModel]) :
  (Tensor α [nQ, dModel] ×
   Tensor α [nK, dModel] ×
   Tensor α [nK, dModel]) :=
  let scale := attentionScaleDenom (α := α) dModel
  let scores := matMulSpec ctx.Q (swapAdjacentAxes ctx.K 0)
  let scaledScores := scaleSpec scores (1 / scale)
  let attentionWeights :=
    match ctx.mask with
    | none => Activation.softmaxSpec 1 scaledScores
    | some m => hardMaskedSoftmaxSpec scaledScores m

  -- Backprop through `Out = A V`.
  let dAttentionWeights := matMulSpec dOut (swapAdjacentAxes ctx.V 0)
  let dV := matMulSpec (swapAdjacentAxes attentionWeights 0) dOut

  -- Backprop through row normalization. Hard-masked blocked entries already have zero weight, so
  -- their score gradients are zero by the formula.
  let dScaledScores := softmaxBackwardFromWeightsSpec attentionWeights dAttentionWeights

  -- Backprop through scaling: `scaledScores = scores * (1 / scale)`.
  let dScores := scaleSpec dScaledScores (1 / scale)

  -- Backprop through `scores = Q Kᵀ`.
  let dQ := matMulSpec dScores ctx.K
  let dK := matMulSpec (swapAdjacentAxes dScores 0) ctx.Q

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
  (dQ : Tensor α [nQ, dModel])
  (dK dV : Tensor α [nK, dModel]) :
  Tensor α [nQ, dModel] :=
  let scale := attentionScaleDenom (α := α) dModel
  let scores := matMulSpec ctx.Q (swapAdjacentAxes ctx.K 0)
  let dScores :=
    addSpec
      (matMulSpec dQ (swapAdjacentAxes ctx.K 0))
      (matMulSpec ctx.Q (swapAdjacentAxes dK 0))
  let scaledScores := scaleSpec scores (1 / scale)
  let dScaledScores := scaleSpec dScores (1 / scale)
  let attentionWeights :=
    match ctx.mask with
    | none => Activation.softmaxSpec 1 scaledScores
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

This spec keeps them as explicit matrices, without bias terms, so the parameterization remains
visible in statements about the forward and derivative maps.
-/
  structure MultiHeadAttention (α : Type) [TorchLean.Storage α]
      (numHeads dModel headDim : Nat) where
    /-- Query projection from `dModel` to all attention heads. -/
    queryWeight : Tensor α [dModel, (numHeads * headDim)]
    /-- Key projection from `dModel` to all attention heads. -/
    keyWeight : Tensor α [dModel, (numHeads * headDim)]
    /-- Value projection from `dModel` to all attention heads. -/
    valueWeight : Tensor α [dModel, (numHeads * headDim)]
    /-- Projection from the concatenated heads back to `dModel`. -/
    outputWeight : Tensor α [(numHeads * headDim), dModel]

/-
  Split tensor into multiple attention heads
-/
  /-- Split `(n, dModel)` into `(numHeads, n, headDim)`.

We store heads as the outermost axis so that "per-head computation" is just a `Tensor.dim` over
`Fin numHeads`.

The feature coordinate is interpreted as `(head, coordinate-within-head)`: first reshape to
`(n, numHeads, headDim)`, then swap the token and head axes. Reshaping directly to
`(numHeads, n, headDim)` would preserve the wrong row-major coordinate order.
-/
  def splitHeadsSpec
    {α : Type} [TorchLean.Storage α] [Context α]
    {n dModel : Nat}
  (x : Tensor α [n, dModel])
  (numHeads headDim : Nat)
  (h : dModel = numHeads * headDim)
  : Tensor α [numHeads, n, headDim] :=
  let s₁ : Shape := .dim n (.dim dModel .scalar)
  let s₂ : Shape := .dim n (.dim numHeads (.dim headDim .scalar))
  have size_eq : Spec.Shape.size s₁ = Spec.Shape.size s₂ := by
    change n * (dModel * 1) = n * (numHeads * (headDim * 1))
    rw [h]
    simp
  Tensor.swapAdjacentAxes (reshapeSpec x size_eq) 0

/-
  Concatenate attention heads back into single tensor
-/
  /-- Combine a tensor-of-heads back into a single `(n, numHeads*headDim)` tensor.

Implementation detail:

1. `Tensor.swapAdjacentAxes` exchanges the head and token axes.
2. `reshapeSpec` flattens the final two axes into `(n, numHeads * headDim)`.
-/
  def combineHeadsSpec
    {α : Type} [TorchLean.Storage α] [Context α]
    {n numHeads headDim : Nat}
    (heads : Tensor α [numHeads, n, headDim]) :
    Tensor α [n, (numHeads * headDim)] :=
  let swapped : Tensor α [n, numHeads, headDim] :=
    Tensor.swapAdjacentAxes heads 0
  let s₁ : Shape := .dim n (.dim numHeads (.dim headDim .scalar))
  let s₂ : Shape := .dim n (.dim (numHeads * headDim) .scalar)
  have hSize : Spec.Shape.size s₁ = Spec.Shape.size s₂ := by
    simp [s₁, s₂, Spec.Shape.size]
  reshapeSpec swapped hSize

  /-- Multi-head attention forward pass (self-attention when `mask` is square).

High-level structure:

1. project `x` into `Q,K,V`
2. split the projection dimension into heads
3. run scaled dot-product attention per head (sharing the same mask)
4. combine heads back and project with `Wo`
-/
  def MultiHeadAttention.forward
    {α : Type} [TorchLean.Storage α] [Context α]
    [DecidableRel ((· > ·) : α → α → Prop)]
    {numHeads dModel headDim : Nat}
  (n : Nat) (h1 : n ≠ 0)
  (mha : MultiHeadAttention α numHeads dModel headDim)
  (x : Tensor α [n, dModel])
  (mask : Option (Tensor Bool [n, n])) :
  Tensor α [n, dModel] :=

  -- Project inputs to big Q, K, V.
  let Q := matMulSpec x mha.queryWeight
  let K := matMulSpec x mha.keyWeight
  let V := matMulSpec x mha.valueWeight

  -- Split heads: we represent heads as the outer axis `(numHeads, n, headDim)`.
  let QHeads := splitHeadsSpec Q numHeads headDim rfl
  let KHeads := splitHeadsSpec K numHeads headDim rfl
  let VHeads := splitHeadsSpec V numHeads headDim rfl

  -- Compute attention per head as a tensor indexed by `Fin numHeads`.
  let attentionHeads : Tensor α [numHeads, n, headDim] :=
    Tensor.dim (fun headIdx =>
      let ctx : AttentionContext α n n headDim h1 h1 :=
        { Q := Tensor.unstack QHeads headIdx
          K := Tensor.unstack KHeads headIdx
          V := Tensor.unstack VHeads headIdx
          mask := mask }
      scaledDotProductAttention ctx)

  -- Combine heads back to `(n, numHeads * headDim)`.
  let concatenated := combineHeadsSpec (α := α) (n := n) (numHeads := numHeads)
    (headDim := headDim) attentionHeads

  -- Project the concatenated heads back to the model dimension.
  matMulSpec concatenated mha.outputWeight

/--
Parameter gradients for multi-head attention: one per projection matrix.

The record lives here, beside the backward pass that produces it, rather than in the transformer
model file where it was originally declared. All four fields are named after the corresponding field
of `MultiHeadAttention`, so `{ queryWeight, keyWeight, valueWeight, outputWeight }` works with the
anonymous constructor at every call site.

PyTorch analogue: the `.grad` of `nn.MultiheadAttention.in_proj_weight` split into its three blocks,
plus `out_proj.weight.grad`.
-/
structure MultiHeadAttentionParameterGradients (numHeads dModel headDim : Nat) (α : Type)
    [TorchLean.Storage α] where
  /-- Gradient of the query projection matrix. -/
  queryWeight : Tensor α [dModel, numHeads * headDim]
  /-- Gradient of the key projection matrix. -/
  keyWeight : Tensor α [dModel, numHeads * headDim]
  /-- Gradient of the value projection matrix. -/
  valueWeight : Tensor α [dModel, numHeads * headDim]
  /-- Gradient of the output projection matrix. -/
  outputWeight : Tensor α [numHeads * headDim, dModel]

/-- Everything `multiHeadAttentionBackward` sends backwards: the four parameter gradients and the
gradient with respect to the attended sequence. -/
structure MultiHeadAttentionGradients (n numHeads dModel headDim : Nat) (α : Type)
    [TorchLean.Storage α] where
  /-- Gradients for the four projection matrices. -/
  parameters : MultiHeadAttentionParameterGradients numHeads dModel headDim α
  /-- Gradient with respect to the input sequence `x`. -/
  input : Tensor α [n, dModel]

/-- Multi-head attention backward pass.

Returns gradients for input `x` and all projection matrices `(Wq,Wk,Wv,Wo)`.
The forward intermediates are recomputed locally instead of relying on a global tape.
-/
def multiHeadAttentionBackward
  {α : Type} [TorchLean.Storage α] [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)]
  {n numHeads dModel headDim : Nat} (h1 : n ≠ 0)
  (mha : MultiHeadAttention α numHeads dModel headDim)
  (x : Tensor α [n, dModel])
  (mask : Option (Tensor Bool [n, n]))
  (gradOutput : Tensor α [n, dModel]) :
  MultiHeadAttentionGradients n numHeads dModel headDim α :=

  -- Forward recomputation for intermediate values.
  let Q := matMulSpec x mha.queryWeight
  let K := matMulSpec x mha.keyWeight
  let V := matMulSpec x mha.valueWeight

  let QHeads := splitHeadsSpec Q numHeads headDim rfl
  let KHeads := splitHeadsSpec K numHeads headDim rfl
  let VHeads := splitHeadsSpec V numHeads headDim rfl

  let attentionHeads : Tensor α [numHeads, n, headDim] :=
    Tensor.dim (fun headIdx =>
      let ctx : AttentionContext α n n headDim h1 h1 :=
        { Q := Tensor.unstack QHeads headIdx
          K := Tensor.unstack KHeads headIdx
          V := Tensor.unstack VHeads headIdx
          mask := mask }
      scaledDotProductAttention ctx)

  let concatenated := combineHeadsSpec (α := α) (n := n) (numHeads := numHeads)
    (headDim := headDim) attentionHeads

  -- Backpropagate through the output projection.
  let (gradConcat, gradOutputWeight) :=
    matmulBackwardSpec (Shape.CanBroadcastTo.refl .scalar)
      (Shape.CanBroadcastTo.refl .scalar) concatenated mha.outputWeight gradOutput

  -- Backpropagate through the reshape and axis swap used to combine heads.
  let gradAttentionHeads := splitHeadsSpec gradConcat numHeads headDim rfl

  -- Backpropagate independently through each attention head.
  let (gradQHeads, gradKHeads, gradVHeads) :
      Tensor α [numHeads, n, headDim] ×
        Tensor α [numHeads, n, headDim] ×
        Tensor α [numHeads, n, headDim] :=
    let backwardHead (headIdx : Fin numHeads) :=
      let ctx : AttentionContext α n n headDim h1 h1 :=
        { Q := Tensor.unstack QHeads headIdx
          K := Tensor.unstack KHeads headIdx
          V := Tensor.unstack VHeads headIdx
          mask := mask }
      scaledDotProductAttentionBackward ctx
        (Tensor.unstack gradAttentionHeads headIdx)
    let gQ : Fin numHeads → Tensor α [n, headDim] :=
      fun headIdx => (backwardHead headIdx).1
    let gK : Fin numHeads → Tensor α [n, headDim] :=
      fun headIdx => (backwardHead headIdx).2.1
    let gV : Fin numHeads → Tensor α [n, headDim] :=
      fun headIdx => (backwardHead headIdx).2.2
    (Tensor.dim gQ, Tensor.dim gK, Tensor.dim gV)

  -- Undo the head split for Q, K, and V.
  let gradQ := combineHeadsSpec (α := α) (n := n) (numHeads := numHeads)
    (headDim := headDim) gradQHeads
  let gradK := combineHeadsSpec (α := α) (n := n) (numHeads := numHeads)
    (headDim := headDim) gradKHeads
  let gradV := combineHeadsSpec (α := α) (n := n) (numHeads := numHeads)
    (headDim := headDim) gradVHeads

  -- Backpropagate through the input projections.
  let (gradXQuery, gradQueryWeight) :=
    matmulBackwardSpec (Shape.CanBroadcastTo.refl .scalar)
      (Shape.CanBroadcastTo.refl .scalar) x mha.queryWeight gradQ
  let (gradXKey, gradKeyWeight) :=
    matmulBackwardSpec (Shape.CanBroadcastTo.refl .scalar)
      (Shape.CanBroadcastTo.refl .scalar) x mha.keyWeight gradK
  let (gradXValue, gradValueWeight) :=
    matmulBackwardSpec (Shape.CanBroadcastTo.refl .scalar)
      (Shape.CanBroadcastTo.refl .scalar) x mha.valueWeight gradV

  -- Add the contributions to the input gradient from the Q, K, and V branches.
  let gradX := addSpec (addSpec gradXQuery gradXKey) gradXValue

  { parameters :=
      { queryWeight := gradQueryWeight
        keyWeight := gradKeyWeight
        valueWeight := gradValueWeight
        outputWeight := gradOutputWeight }
    input := gradX }

/-- Forward-mode JVP for multi-head attention.

The rule follows the same computational graph as `MultiHeadAttention.forward`:

1. project tangents through `Q/K/V`,
2. split primal and tangent projections into heads,
3. apply `scaledDotProductAttentionJvp` head-wise,
4. combine head tangents, then differentiate the final output projection.

Attention forward-mode AD is explicit at the specification layer rather than hidden behind a
runtime-only implementation.
-/
def multiHeadAttentionJvp
  {α : Type} [TorchLean.Storage α] [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)]
  {n numHeads dModel headDim : Nat} (h1 : n ≠ 0)
  (mha dmha : MultiHeadAttention α numHeads dModel headDim)
  (x dx : Tensor α [n, dModel])
  (mask : Option (Tensor Bool [n, n])) :
  Tensor α [n, dModel] :=

  let Q := matMulSpec x mha.queryWeight
  let K := matMulSpec x mha.keyWeight
  let V := matMulSpec x mha.valueWeight

  let dQ := addSpec (matMulSpec dx mha.queryWeight) (matMulSpec x dmha.queryWeight)
  let dK := addSpec (matMulSpec dx mha.keyWeight) (matMulSpec x dmha.keyWeight)
  let dV := addSpec (matMulSpec dx mha.valueWeight) (matMulSpec x dmha.valueWeight)

  let QHeads := splitHeadsSpec Q numHeads headDim rfl
  let KHeads := splitHeadsSpec K numHeads headDim rfl
  let VHeads := splitHeadsSpec V numHeads headDim rfl
  let dQHeads := splitHeadsSpec dQ numHeads headDim rfl
  let dKHeads := splitHeadsSpec dK numHeads headDim rfl
  let dVHeads := splitHeadsSpec dV numHeads headDim rfl

  let attentionHeads : Tensor α [numHeads, n, headDim] :=
    Tensor.dim (fun headIdx =>
      let ctx : AttentionContext α n n headDim h1 h1 :=
        { Q := Tensor.unstack QHeads headIdx
          K := Tensor.unstack KHeads headIdx
          V := Tensor.unstack VHeads headIdx
          mask := mask }
      scaledDotProductAttention ctx)

  let dAttentionHeads : Tensor α [numHeads, n, headDim] :=
    Tensor.dim (fun headIdx =>
      let ctx : AttentionContext α n n headDim h1 h1 :=
        { Q := Tensor.unstack QHeads headIdx
          K := Tensor.unstack KHeads headIdx
          V := Tensor.unstack VHeads headIdx
          mask := mask }
      scaledDotProductAttentionJvp ctx
        (Tensor.unstack dQHeads headIdx)
        (Tensor.unstack dKHeads headIdx)
        (Tensor.unstack dVHeads headIdx))

  let concatenated := combineHeadsSpec (α := α) (n := n) (numHeads := numHeads)
    (headDim := headDim) attentionHeads
  let dConcatenated := combineHeadsSpec (α := α) (n := n) (numHeads := numHeads)
    (headDim := headDim) dAttentionHeads

  addSpec (matMulSpec dConcatenated mha.outputWeight)
    (matMulSpec concatenated dmha.outputWeight)

/-- Self-attention on a single sequence.

This uses the same input `x` for Q/K/V, runs scaled dot-product attention, then applies the output
projection `Wo`.

PyTorch analogue: the core of `nn.MultiheadAttention` / `TransformerEncoderLayer` (ignoring the
batch axis).
-/
def selfAttention
  {α : Type} [TorchLean.Storage α] [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)]
  {n dModel projDim : Nat}
  (x : Tensor α [n, dModel])
  (Wq : Tensor α [dModel, projDim])
  (Wk : Tensor α [dModel, projDim])
  (Wv : Tensor α [dModel, projDim])
  (Wo : Tensor α [projDim, dModel])
  (h1 : n ≠ 0) :
  Tensor α [n, dModel] := by
  let Q := matMulSpec x Wq
  let K := matMulSpec x Wk
  let V := matMulSpec x Wv
  let ctx : AttentionContext α n n projDim h1 h1 :=
    { Q := Q, K := K, V := V, mask := none }
  exact matMulSpec (scaledDotProductAttention ctx) Wo

/-- Cross-attention between two sequences.

`query` is length `n1` and attends to `key/value` of length `n2`.

PyTorch analogue: the attention block in a Transformer decoder layer (`nn.MultiheadAttention`
with distinct query and key/value inputs).
-/
def crossAttention {α : Type} [TorchLean.Storage α] [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)]
  {n1 n2 dModel projDim : Nat}
  (query : Tensor α [n1, dModel])
  (key : Tensor α [n2, dModel])
  (value : Tensor α [n2, dModel])
  (Wq : Tensor α [dModel, projDim])
  (Wk : Tensor α [dModel, projDim])
  (Wv : Tensor α [dModel, projDim])
  (Wo : Tensor α [projDim, dModel])
  (h1 : n1 ≠ 0) (h2 : n2 ≠ 0) :
  Tensor α [n1, dModel] :=
  let Q := matMulSpec query Wq
  let K := matMulSpec key Wk
  let V := matMulSpec value Wv
  let ctx : AttentionContext α n1 n2 projDim h1 h2 :=
    { Q := Q, K := K, V := V, mask := none }
  let attention := scaledDotProductAttention ctx
  matMulSpec attention Wo

/-- Sparse attention using a Boolean attention pattern. -/
def sparseAttention {α : Type} [TorchLean.Storage α] [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)]
  {n dModel projDim : Nat}
  (x : Tensor α [n, dModel])
  (sparsityPattern : Tensor Bool [n, n])
  (Wq : Tensor α [dModel, projDim])
  (Wk : Tensor α [dModel, projDim])
  (Wv : Tensor α [dModel, projDim])
  (Wo : Tensor α [projDim, dModel])
  (h1 : n ≠ 0) :
  Tensor α [n, dModel] :=
  let Q := matMulSpec x Wq
  let K := matMulSpec x Wk
  let V := matMulSpec x Wv
  let ctx : AttentionContext α n n projDim h1 h1 :=
    { Q := Q, K := K, V := V, mask := sparsityPattern }
  let attention := scaledDotProductAttention ctx
  matMulSpec attention Wo

end Spec
