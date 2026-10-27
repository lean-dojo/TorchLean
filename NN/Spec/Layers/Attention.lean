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

The optional Boolean mask has shape `(nQ × nK)`. Blocked entries receive zero numerator before
row normalization. Over real scalars with a nonzero denominator, their attention weight is zero.
This is the finite-scalar encoding of the PyTorch pattern
`scores.masked_fill(~mask, -torch.inf)`.

Rows with no allowed entries return zero weights directly, avoiding division by zero. This
guarantees zero weights even for exceptional scores; the final multiplication by `V` still obeys
the scalar backend's arithmetic. For IEEE scalars, `0 * ∞` can produce NaN. On a partially allowed
row, exceptional allowed scores can also make the denominator and blocked weights NaN.


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
structure AttentionContext (α : Type) [TorchLean.Storage α]
  (nQ nK dModel : Nat) (h1 : nQ ≠ 0) (h2 : nK ≠ 0) where
  Q : Tensor α [nQ, dModel]
  K : Tensor α [nK, dModel]
  V : Tensor α [nK, dModel]
  mask : Option (Tensor Bool [nQ, nK])

/-- Denominator used by scaled dot-product attention.

Standard attention requires a positive feature dimension and divides scores by `sqrt(dModel)`.
TorchLean's tensor shapes also admit zero dimensions. In that degenerate case the result has no
feature coordinates, so choosing denominator `1` gives the unique empty-feature result without
introducing a division by zero. The scalar context constructs the root from the count: binary
backends can recover a finite root even when casting the dimension itself would overflow. -/
def attentionScaleDenom (dModel : Nat) : α :=
  if dModel = 0 then 1 else Context.sqrtNat dModel

/-!
## Exact hard masking

TorchLean encodes the usual "true `-∞` before softmax" behavior without requiring the
tensor scalar
type itself to contain infinities. Instead of replacing blocked logits by a finite sentinel, we form
stable softmax numerators directly. If `rowMax` is the greatest allowed score in the row, then

`numerator_j = if mask_j then exp(score_j - rowMax) else 0`.

This models the zero numerator contributed by `exp(-∞)`. The zero-mass conclusion used in causal
proofs requires the scalar division laws and a nonzero denominator. A row with no allowed positions
returns zeros directly.
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
is the zero vector, matching PyTorch SDPA and the LibTorch CUDA attention bridge
(`NN/Backend/LibTorch.lean`).
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
      Activation.Internal.normalizeWeights numerators

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

For ordinary softmax, `weights = softmax(scores)`. For hard-masked softmax over real scalars with
`weights = 0` at blocked entries, the same formula gives zero gradient through blocked logits.
Exceptional floating-point operands need not satisfy that zero-product conclusion:

`dScores = weights ⊙ (dWeights - Σⱼ dWeightsⱼ * weightsⱼ)`.

Attention uses `center := true`: subtract the upstream coordinate at a maximal weight before
taking the weighted sum. For normalized real weights this leaves the derivative unchanged; for
an all-zero row both formulas return zero. In floating point it retains a small dominant-logit
gradient after its probability rounds to one. The default retains the uncentered algebraic
formula for arbitrary supplied weights, whose sum need not be one.
-/
def softmaxBackwardFromWeightsSpec {s : Shape} (weights dWeights : Tensor α s)
    (center : Bool := false) : Tensor α s :=
  match s with
  | .scalar => Tensor.scalar 0
  | .dim 0 .scalar => dWeights
  | .dim (Nat.succ n) .scalar =>
      if center then
        let pivot : Fin (Nat.succ n) :=
          Fin.cast (by simp [Shape.size]) (Tensor.argmax (by simp [Shape.size]) weights)
        Activation.Internal.softmaxPullback weights dWeights (dWeights.getScalar pivot)
      else
        let rowDot : α := sumSpec (mulSpec dWeights weights)
        mulSpec weights (subSpec dWeights (replicate (Tensor.scalar rowDot)))
  | .dim n inner =>
      Tensor.dim (fun i : Fin n =>
        softmaxBackwardFromWeightsSpec (s := inner)
          (Tensor.unstack weights i) (Tensor.unstack dWeights i) (center := center))

namespace Internal

/-- Reconstruct a complete finite scaled contraction without rounding its coefficients. -/
@[noinline] def productCoefficients? {α : Type} [Storage α] [Context α]
    {m n p groups : Nat} (left : Fin groups → Tensor α [m, n])
    (right : Fin groups → Tensor α [n, p]) (scale : α)
    (row : Fin m) (column : Fin p) :
    Option (Numeric.QuotientCoefficients (Numeric.QuotientArithmetic.depth α)) := do
  let ops := (inferInstance : Numeric.QuotientArithmetic α)
  let coefficient ← ops.encode scale
  let zero ← ops.encode (0 : α)
  let total ← (List.finRange groups).foldl (fun accumulated group => do
    let initial ← accumulated
    (List.finRange n).foldl (fun accumulated index => do
      let total ← accumulated
      let a ← ops.encode (get2 (left group) row index)
      let b ← ops.encode (get2 (right group) index column)
      pure (Numeric.QuotientCoefficients.add total (Numeric.QuotientCoefficients.mul a b)))
      (some initial)) (some zero)
  pure (Numeric.QuotientCoefficients.mul total coefficient)

/-- Round a finite contraction only after reconstructing every original coefficient. -/
@[noinline] def replayProducts? {α : Type} [Storage α] [Context α]
    {m n p groups : Nat} (left : Fin groups → Tensor α [m, n])
    (right : Fin groups → Tensor α [n, p]) (scale ordinary : α)
    (row : Fin m) (column : Fin p) : Option α := do
  let ops := (inferInstance : Numeric.QuotientArithmetic α)
  let result ← productCoefficients? left right scale row column
  let rounded ← ops.decode result
  pure <| if result.primal == 0 && ordinary == 0 then ops.copyPrimal ordinary rounded else rounded

/-- Scaled sums of matrix products used by attention's forward and differential rules.

One product serves the forward or VJP; two products retain the JVP's joint contraction. Within
and between products, checked arithmetic follows the original left-fold schedule. `scaleFirst`
keeps the VJP's input-scaling convention. Range loss or an absorbed nonzero term replays all
original finite coefficient trees, including the scale, before rounding. Missing finite input
encodings retain the ordinary result. Unsupported carriers use the original expression directly.

This is a reference scalar repair, not a numerical contract for an expanded IR or foreign kernel.
-/
def scaledProducts {α : Type} [Storage α] [Context α]
    {m n p groups : Nat} (left : Fin groups → Tensor α [m, n])
    (right : Fin groups → Tensor α [n, p]) (scale : α) (scaleFirst : Bool := false) :
    Tensor α [m, p] :=
  let ops := (inferInstance : Numeric.QuotientArithmetic α)
  if !ops.supported then
    let combined := match groups, left, right with
      | 0, _, _ => Tensor.zeros [m, p]
      | groups + 1, left, right =>
          let product := fun group => matMulSpec
            (if scaleFirst then scaleSpec (left group) scale else left group) (right group)
          (List.finRange groups).foldl (fun total group =>
            addSpec total (product group.succ)) (product 0)
    if scaleFirst then combined else scaleSpec combined scale
  else
    Tensor.Internal.Rep.ofFn fun coordinate =>
      let (total, flags) := match groups, left, right with
        | 0, _, _ => (0, 0)
        | groups + 1, left, right =>
          let product := fun group => (List.finRange n).foldl (fun (total, flags) index =>
            let a := get2 (left group) coordinate.1 index
            let (a, inputFlags) := if scaleFirst then ops.mulWithFlags a scale else (a, 0)
            let (value, productFlags) := ops.mulWithFlags a
              (get2 (right group) index coordinate.2.1)
            let (next, sumFlags) := ops.addWithFlags total value
            let stalled := (value != 0 && next == total).toUInt8
            (next, flags ||| inputFlags ||| productFlags ||| sumFlags ||| stalled))
            ((0 : α), (0 : UInt8))
          (List.finRange groups).foldl (fun (total, flags) group =>
              let (part, partFlags) := product group.succ
              let (next, sumFlags) := ops.addWithFlags total part
              let stalled := (part != 0 && next == total).toUInt8
              (next, flags ||| partFlags ||| sumFlags ||| stalled)) (product 0)
      let (ordinary, scaleFlags) := if scaleFirst then (total, 0)
        else ops.mulWithFlags total scale
      if flags ||| scaleFlags == 0 then ordinary else
        (replayProducts? left right scale ordinary coordinate.1 coordinate.2.1).getD ordinary

/-- Scaled contractions over a shared batch prefix, without materializing an unscaled product. -/
def scaledBatchProducts {α : Type} [Storage α] [Context α]
    {batch : Shape} {m n p groups : Nat}
    (left : Fin groups → Tensor α (batch.concat [m, n]))
    (right : Fin groups → Tensor α (batch.concat [n, p])) (scale : α) :
    Tensor α (batch.concat [m, p]) :=
  match batch with
  | .scalar => scaledProducts left right scale
  | .dim _ _ => Tensor.dim fun index =>
      scaledBatchProducts (fun group => (left group).unstack index)
        (fun group => (right group).unstack index) scale

/-- A scaled matrix product with broadcast batch prefixes and one retained contraction boundary. -/
def scaledMatmul {α : Type} [Storage α] [Context α]
    {batchA batchB batch : Shape} {m n p : Nat}
    (broadcastA : Shape.CanBroadcastTo batchA batch)
    (broadcastB : Shape.CanBroadcastTo batchB batch)
    (left : Tensor α (batchA.concat [m, n]))
    (right : Tensor α (batchB.concat [n, p])) (scale : α) :
    Tensor α (batch.concat [m, p]) :=
  let a := broadcastTo (Tensor.LinearAlgebra.Internal.extendBroadcastSuffix [m, n] broadcastA) left
  let b := broadcastTo (Tensor.LinearAlgebra.Internal.extendBroadcastSuffix [n, p] broadcastB) right
  scaledBatchProducts (fun _ : Fin 1 => a) (fun _ => b) scale

/-- The two product-rule terms share one replay boundary, so large opposite tangents can cancel. -/
def scaledMatmulJvp {α : Type} [Storage α] [Context α]
    {batchA batchB batch : Shape} {m n p : Nat}
    (broadcastA : Shape.CanBroadcastTo batchA batch)
    (broadcastB : Shape.CanBroadcastTo batchB batch)
    (left deltaLeft : Tensor α (batchA.concat [m, n]))
    (right deltaRight : Tensor α (batchB.concat [n, p])) (scale : α) :
    Tensor α (batch.concat [m, p]) :=
  let a := broadcastTo (Tensor.LinearAlgebra.Internal.extendBroadcastSuffix [m, n] broadcastA) left
  let da := broadcastTo (Tensor.LinearAlgebra.Internal.extendBroadcastSuffix [m, n] broadcastA)
    deltaLeft
  let b := broadcastTo (Tensor.LinearAlgebra.Internal.extendBroadcastSuffix [n, p] broadcastB) right
  let db := broadcastTo (Tensor.LinearAlgebra.Internal.extendBroadcastSuffix [n, p] broadcastB)
    deltaRight
  scaledBatchProducts (fun group : Fin 2 => if group.val == 0 then da else a)
    (fun group => if group.val == 0 then b else db) scale

/-- Cotangents of a scaled, broadcast matrix product, retaining its fixed scale in each product. -/
def scaledMatmulBackward {α : Type} [Storage α] [Context α]
    {batchA batchB batch : Shape} {m n p : Nat}
    (broadcastA : Shape.CanBroadcastTo batchA batch)
    (broadcastB : Shape.CanBroadcastTo batchB batch)
    (left : Tensor α (batchA.concat [m, n]))
    (right : Tensor α (batchB.concat [n, p])) (gradient : Tensor α (batch.concat [m, p]))
    (scale : α) : Tensor α (batchA.concat [m, n]) × Tensor α (batchB.concat [n, p]) :=
  let a := broadcastTo (Tensor.LinearAlgebra.Internal.extendBroadcastSuffix [m, n] broadcastA) left
  let b := broadcastTo (Tensor.LinearAlgebra.Internal.extendBroadcastSuffix [n, p] broadcastB) right
  let bt : Tensor α (batch.concat [p, n]) := by
    simpa using swapAdjacentAxes b batch.rank
  let transposed : Tensor α (batch.concat [n, m]) := by
    simpa using swapAdjacentAxes a batch.rank
  (reduceFromBroadcastTo (Tensor.LinearAlgebra.Internal.extendBroadcastSuffix [m, n] broadcastA)
      (scaledBatchProducts (fun _ : Fin 1 => gradient) (fun _ => bt) scale),
    reduceFromBroadcastTo (Tensor.LinearAlgebra.Internal.extendBroadcastSuffix [n, p] broadcastB)
      (scaledBatchProducts (fun _ : Fin 1 => transposed) (fun _ => gradient) scale))

/-- Replay one softmax row from retained finite cotangent coefficients. -/
def replayPullbackCoefficientsRow? {α : Type} [Storage α] [Context α] {p : Nat}
    (weights ordinary : Tensor α [p])
    (coefficients : Fin p → Option
      (Numeric.QuotientCoefficients (Numeric.QuotientArithmetic.depth α))) :
    Option (Tensor α [p]) :=
  match p, weights, ordinary, coefficients with
  | 0, _, ordinary, _ => some ordinary
  | p + 1, weights, ordinary, coefficients =>
      let pivot : Fin (p + 1) := Fin.cast (by simp [Shape.size])
        (Tensor.argmax (by simp [Shape.size]) weights)
      Activation.Internal.replaySoftmaxPullback? weights ordinary coefficients (coefficients pivot)

/-- Replay one softmax row from its original gradient contractions and saved weights. -/
def replayPullbackProductsRow? {α : Type} [Storage α] [Context α] {m n p groups : Nat}
    (weights ordinary : Tensor α [p]) (left : Fin groups → Tensor α [m, n])
    (right : Fin groups → Tensor α [n, p]) (scale : α) (row : Fin m) :
    Option (Tensor α [p]) :=
  replayPullbackCoefficientsRow? weights ordinary (productCoefficients? left right scale row)

/-- Multiply a retained contraction by one coordinate's factor before encoding its result. -/
def weightedProductCoefficients? {α : Type} [Storage α] [Context α] {m n p : Nat}
    (left : Tensor α [m, n]) (right : Tensor α [n, p])
    (factor : Tensor α [m, p]) (row : Fin m) (column : Fin p) :
    Option (Numeric.QuotientCoefficients (Numeric.QuotientArithmetic.depth α)) := do
  let ops := (inferInstance : Numeric.QuotientArithmetic α)
  let product ← productCoefficients? (fun _ : Fin 1 => left) (fun _ => right) 1 row column
  let coefficient ← ops.encode (get2 factor row column)
  pure (Numeric.QuotientCoefficients.mul product coefficient)

/-- Retain both pointwise factors through the transposed value-gradient contraction. -/
def weightedValueCoefficients? {α : Type} [Storage α] [Context α] {rows keys width : Nat}
    (weights multiplier : Tensor α [rows, keys]) (gradient : Tensor α [rows, width])
    (key : Fin keys) (column : Fin width) :
    Option (Numeric.QuotientCoefficients (Numeric.QuotientArithmetic.depth α)) := do
  let ops := (inferInstance : Numeric.QuotientArithmetic α)
  let zero ← ops.encode (0 : α)
  (List.finRange rows).foldl (fun accumulated row => do
    let total ← accumulated
    let p ← ops.encode (get2 weights row key)
    let m ← ops.encode (get2 multiplier row key)
    let g ← ops.encode (get2 gradient row column)
    pure (Numeric.QuotientCoefficients.add total
      (Numeric.QuotientCoefficients.mul (Numeric.QuotientCoefficients.mul p m) g))) (some zero)

/-- Conservative range guard for an ordinary product whose coefficients must survive another map.
Only flags escape this check; the caller keeps the original value-producing schedule. -/
def productsLoseRange {α : Type} [Storage α] [Context α] {m n p : Nat}
    (left : Tensor α [m, n]) (right : Tensor α [n, p]) : Bool := Id.run do
  let ops := (inferInstance : Numeric.QuotientArithmetic α)
  if !ops.supported then return false
  for row in List.finRange m do
    for column in List.finRange p do
      let mut total : α := 0
      for index in List.finRange n do
        let (product, productFlags) := ops.mulWithFlags (get2 left row index)
          (get2 right index column)
        let (next, sumFlags) := ops.addWithFlags total product
        if productFlags != 0 || sumFlags != 0 || (product != 0 && next == total) then
          return true
        total := next
  return false

/-- Retain the original gradient contractions through the centered softmax pullback.

The ordinary tensor calculation is unchanged unless an output coefficient loses range. Replaying
the contraction and pullback together can recover a finite result even when the intervening
gradient tensor cannot represent it. This uses the same coefficient replay as ordinary softmax.
-/
def pullbackProducts {α : Type} [Storage α] [Context α] {m n p groups : Nat}
    (weights gradient : Tensor α [m, p]) (left : Fin groups → Tensor α [m, n])
    (right : Fin groups → Tensor α [n, p]) (scale : α) : Tensor α [m, p] :=
  let ordinary := softmaxBackwardFromWeightsSpec weights gradient (center := true)
  let ops := (inferInstance : Numeric.QuotientArithmetic α)
  if !ops.supported then ordinary
  else match p, ordinary, weights, right with
  | 0, ordinary, _, _ => ordinary
  | p + 1, ordinary, weights, right => Tensor.dim fun row =>
      let values := ordinary.unstack row
      if !(List.finRange (p + 1)).any (fun i =>
          (ops.addWithFlags (values.getScalar i) 0).2 != 0) then values
      else
        (replayPullbackProductsRow? (weights.unstack row) values left right scale row).getD values

end Internal

/-- Row normalization followed by a differentiable multiplier and a value contraction.
The forward schedule is deliberately the ordinary softmax, pointwise product, then matrix product.
The compound boundary matters for derivatives, where the unweighted cotangent can overflow. -/
def softmaxMatmulWeights {α : Type} [Storage α] [Context α]
    {batch : Shape} {rows keys : Nat} (scores : Tensor α (batch.concat [rows, keys]))
    (mask : Option (Tensor Bool [rows, keys]) := none) :
    Tensor α (batch.concat [rows, keys]) :=
  match batch, scores with
  | .scalar, scores => match mask with
      | none => Activation.Internal.softmaxInnermostSpec scores
      | some mask => hardMaskedSoftmaxSpec scores mask
  | .dim _ _, scores => Tensor.dim fun i => softmaxMatmulWeights (scores.unstack i) mask

/-- Mix value rows from saved weights, keeping pointwise multiplication before the contraction. -/
def Internal.softmaxMatmulFromWeights {α : Type} [Storage α] [Context α]
    {batch : Shape} {rows keys width : Nat}
    (weights : Tensor α (batch.concat [rows, keys]))
    (value : Tensor α (batch.concat [keys, width]))
    (multiplier : Tensor α (batch.concat [rows, keys])) :
    Tensor α (batch.concat [rows, width]) :=
  match batch, weights, value, multiplier with
  | .scalar, weights, value, multiplier => matMulSpec (mulSpec weights multiplier) value
  | .dim _ _, weights, value, multiplier => Tensor.dim fun i =>
      Internal.softmaxMatmulFromWeights (weights.unstack i) (value.unstack i)
        (multiplier.unstack i)

/-- Forward value of the normalized matrix product, without changing its association. -/
def softmaxMatmul {α : Type} [Storage α] [Context α]
    {batch : Shape} {rows keys width : Nat}
    (scores : Tensor α (batch.concat [rows, keys]))
    (value : Tensor α (batch.concat [keys, width]))
    (multiplier : Tensor α (batch.concat [rows, keys]))
    (mask : Option (Tensor Bool [rows, keys]) := none) :
    Tensor α (batch.concat [rows, width]) :=
  Internal.softmaxMatmulFromWeights (softmaxMatmulWeights scores mask) value multiplier

/-- Cotangents of normalized scores, values and their differentiable multiplier.
Finite coefficient replay keeps the original output cotangent and values through both the
multiplier and centered softmax. Unsupported scalar carriers use the ordinary local formula. -/
def softmaxMatmulBackward {α : Type} [Storage α] [Context α]
    {batch : Shape} {rows keys width : Nat}
    (weights : Tensor α (batch.concat [rows, keys]))
    (value : Tensor α (batch.concat [keys, width]))
    (multiplier : Tensor α (batch.concat [rows, keys]))
    (gradient : Tensor α (batch.concat [rows, width])) :
    Tensor α (batch.concat [rows, keys]) × Tensor α (batch.concat [keys, width]) ×
      Tensor α (batch.concat [rows, keys]) :=
  match batch, weights, value, multiplier, gradient with
  | .dim _ _, weights, value, multiplier, gradient =>
      let result := fun i => softmaxMatmulBackward (weights.unstack i) (value.unstack i)
        (multiplier.unstack i) (gradient.unstack i)
      (Tensor.dim fun i => (result i).1, Tensor.dim fun i => (result i).2.1,
        Tensor.dim fun i => (result i).2.2)
  | .scalar, weights, value, multiplier, gradient =>
      let ops := (inferInstance : Numeric.QuotientArithmetic α)
      let active := fun row => !(List.finRange keys).all fun key =>
        Context.isZero (get2 weights row key)
      let gradient := if ops.supported then
        Tensor.dim fun row =>
          if active row then gradient.unstack row else Tensor.zeros [width]
        else gradient
      let transposed := swapAdjacentAxes value 0
      let upstream := matMulSpec gradient transposed
      let upstream := if ops.supported then
        Tensor.dim fun row =>
          if active row then upstream.unstack row else Tensor.zeros [keys]
        else upstream
      let ordinary := softmaxBackwardFromWeightsSpec weights (mulSpec upstream multiplier)
        (center := true)
      let multiplierGradient := mulSpec weights upstream
      let contractionRisk := Internal.productsLoseRange gradient transposed
      let scoreGradient := if !ops.supported then ordinary else Tensor.dim fun row =>
        let previous := ordinary.unstack row
        let coefficient := Internal.weightedProductCoefficients? gradient transposed multiplier row
        if !contractionRisk && !(List.finRange keys).any (fun column =>
            previous.getScalar column == 0 ||
              (ops.addWithFlags (previous.getScalar column) 0).2 != 0 ||
              (ops.mulWithFlags (get2 upstream row column) (get2 multiplier row column)).2 != 0)
          then previous
        else (Internal.replayPullbackCoefficientsRow? (weights.unstack row) previous coefficient)
          |>.getD previous
      let multiplierGradient := if !ops.supported then multiplierGradient else
        Tensor.dim fun row => Tensor.dim fun column => Tensor.scalar <|
          let previous := get2 multiplierGradient row column
          if !contractionRisk && previous != 0 && (ops.addWithFlags previous 0).2 == 0 &&
              (ops.mulWithFlags (get2 weights row column) (get2 upstream row column)).2 == 0
            then previous else
            (do
              let result ← Internal.weightedProductCoefficients? gradient transposed weights
                row column
              let rounded ← ops.decode result
              pure <| if result.primal == 0 && previous == 0 then
                ops.copyPrimal previous rounded else rounded)
            |>.getD previous
      let valueGradient := Internal.scaledProducts
        (fun _ : Fin 1 => swapAdjacentAxes (mulSpec weights multiplier) 0) (fun _ => gradient) 1
      let valueGradient := if !ops.supported then valueGradient else
        Tensor.dim fun key => Tensor.dim fun column => Tensor.scalar <|
          let previous := get2 valueGradient key column
          let factorRisk := (List.finRange rows).any fun row =>
            (ops.mulWithFlags (get2 weights row key) (get2 multiplier row key)).2 != 0
          if !factorRisk && previous != 0 && (ops.addWithFlags previous 0).2 == 0 then previous else
            (do
              let total ← Internal.weightedValueCoefficients? weights multiplier gradient key column
              let rounded ← ops.decode total
              pure <| if total.primal == 0 && previous == 0 then
                ops.copyPrimal previous rounded else rounded) |>.getD previous
      (scoreGradient, valueGradient, multiplierGradient)

/-- Product-rule tangent of the normalized matrix product, using its saved row weights. -/
def softmaxMatmulJvp {α : Type} [Storage α] [Context α]
    {batch : Shape} {rows keys width : Nat}
    (weights : Tensor α (batch.concat [rows, keys]))
    (value : Tensor α (batch.concat [keys, width]))
    (multiplier deltaScores : Tensor α (batch.concat [rows, keys]))
    (deltaValue : Tensor α (batch.concat [keys, width]))
    (deltaMultiplier : Tensor α (batch.concat [rows, keys])) :
    Tensor α (batch.concat [rows, width]) :=
  match batch, weights, value, multiplier, deltaScores, deltaValue, deltaMultiplier with
  | .dim _ _, weights, value, multiplier, deltaScores, deltaValue, deltaMultiplier =>
      Tensor.dim fun i => softmaxMatmulJvp (weights.unstack i) (value.unstack i)
        (multiplier.unstack i) (deltaScores.unstack i) (deltaValue.unstack i)
        (deltaMultiplier.unstack i)
  | .scalar, weights, value, multiplier, deltaScores, deltaValue, deltaMultiplier =>
      let deltaWeights := softmaxBackwardFromWeightsSpec weights deltaScores (center := true)
      let deltaMixed := addSpec (mulSpec deltaWeights multiplier) (mulSpec weights deltaMultiplier)
      let ordinary := Internal.scaledProducts
        (fun group : Fin 2 => if group.val == 0 then deltaMixed else mulSpec weights multiplier)
        (fun group => if group.val == 0 then value else deltaValue) 1
      let ops := (inferInstance : Numeric.QuotientArithmetic α)
      if !ops.supported then ordinary else
        match keys, weights, value, multiplier, deltaScores, deltaValue, deltaMultiplier,
            deltaWeights with
        | 0, _, _, _, _, _, _, _ => ordinary
        | keys + 1, weights, value, multiplier, deltaScores, deltaValue, deltaMultiplier,
            deltaWeights =>
          Tensor.dim fun row =>
            let probabilities := weights.unstack row
            let pivot : Fin (keys + 1) := Fin.cast (by simp [Shape.size])
              (Tensor.argmax (by simp [Shape.size]) probabilities)
            let incoming := fun key => ops.encode (get2 deltaScores row key)
            let normalized := Activation.Internal.softmaxPullbackCoefficients?
              probabilities incoming (incoming pivot)
            let factorRisk := (List.finRange (keys + 1)).any fun key =>
              let dp := get2 deltaWeights row key
              let p := get2 weights row key
              let m := get2 multiplier row key
              let dm := get2 deltaMultiplier row key
              let (first, firstFlags) := ops.mulWithFlags dp m
              let (second, secondFlags) := ops.mulWithFlags p dm
              firstFlags != 0 || secondFlags != 0 ||
                (ops.addWithFlags first second).2 != 0 ||
                (ops.mulWithFlags p m).2 != 0 || (ops.addWithFlags dp 0).2 != 0
            Tensor.dim fun column => Tensor.scalar <|
              let previous := get2 ordinary row column
              if !factorRisk && previous != 0 && (ops.addWithFlags previous 0).2 == 0
                then previous else
                (do
                  let dp ← normalized
                  let zero ← ops.encode (0 : α)
                  let total ← (List.finRange (keys + 1)).foldl (fun accumulated key => do
                    let total ← accumulated
                    let p ← ops.encode (get2 weights row key)
                    let m ← ops.encode (get2 multiplier row key)
                    let v ← ops.encode (get2 value key column)
                    let dm ← ops.encode (get2 deltaMultiplier row key)
                    let dv ← ops.encode (get2 deltaValue key column)
                    let dp ← dp key
                    let mixed := Numeric.QuotientCoefficients.add
                      (Numeric.QuotientCoefficients.mul dp m)
                      (Numeric.QuotientCoefficients.mul p dm)
                    let first := Numeric.QuotientCoefficients.mul mixed v
                    let second := Numeric.QuotientCoefficients.mul
                      (Numeric.QuotientCoefficients.mul p m) dv
                    pure (Numeric.QuotientCoefficients.add total
                      (Numeric.QuotientCoefficients.add first second))) (some zero)
                  let rounded ← ops.decode total
                  pure <| if total.primal == 0 && previous == 0 then
                    ops.copyPrimal previous rounded else rounded) |>.getD previous

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

Over real scalars, each unmasked attention row sums to `1`; a masked row with at least one allowed
key has the same normalization. These equalities are not promised for arbitrary numeric contexts.
A fully blocked row returns all-zero weights directly, avoiding a `0/0` result.

The ordinary matrix product and scaling order is retained. Finite-coefficient backends replay
the complete scaled contraction after range loss or an absorbed nonzero term. This prevents an
unscaled score overflow from poisoning otherwise representable scaled scores, without pre-scaling
tiny queries to zero. Scalar conversion and native execution remain separate trust obligations.

PyTorch analogy: `torch.softmax(scores.masked_fill(~mask, -torch.inf), dim=-1)` row-wise, then a
final matrix multiply by `V`.
-/
def scaledDotProductAttention
  {nQ nK dModel : Nat} {h1 : nQ ≠ 0} {h2 : nK ≠ 0}
  (ctx : AttentionContext α nQ nK dModel h1 h2) :
  Tensor α [nQ, dModel] :=
  let scale := attentionScaleDenom (α := α) dModel
  let scaledScores := Internal.scaledProducts (fun _ : Fin 1 => ctx.Q)
    (fun _ => swapAdjacentAxes ctx.K 0) (1 / scale)
  let attentionWeights :=
    match ctx.mask with
    | none => Activation.softmaxSpec 1 scaledScores
    | some m => hardMaskedSoftmaxSpec scaledScores m
  matMulSpec attentionWeights ctx.V

/-- Backward/VJP for scaled dot-product attention.

Returns `(dQ, dK, dV)` given an upstream gradient `dOut`.

We recompute the forward intermediates locally so this spec stays self-contained and does not rely
on a global tape.

Over real scalars, masked calls use the VJP for hard masking: blocked logits have zero forward
weight and therefore zero gradient. Finite-coefficient replay retains `dOut` and `V` through the
centered pullback: their matrix product can overflow even when the score cotangent is finite.
Nonfinite original operands retain the ordinary backend arithmetic.
 -/
def scaledDotProductAttentionBackward
  {nQ nK dModel : Nat} {h1 : nQ ≠ 0} {h2 : nK ≠ 0}
  (ctx : AttentionContext α nQ nK dModel h1 h2)
  (dOut : Tensor α [nQ, dModel]) :
  (Tensor α [nQ, dModel] ×
   Tensor α [nK, dModel] ×
   Tensor α [nK, dModel]) :=
  let scale := attentionScaleDenom (α := α) dModel
  let scaledScores := Internal.scaledProducts (fun _ : Fin 1 => ctx.Q)
    (fun _ => swapAdjacentAxes ctx.K 0) (1 / scale)
  let attentionWeights :=
    match ctx.mask with
    | none => Activation.softmaxSpec 1 scaledScores
    | some m => hardMaskedSoftmaxSpec scaledScores m

  -- Backprop through `Out = A V`.
  let dAttentionWeights := matMulSpec dOut (swapAdjacentAxes ctx.V 0)
  let dV := matMulSpec (swapAdjacentAxes attentionWeights 0) dOut

  -- Center before the weighted sum, not after a dominant weight has rounded to one.
  let dScaledScores :=
    Internal.pullbackProducts attentionWeights dAttentionWeights (fun _ : Fin 1 => dOut)
      (fun _ => swapAdjacentAxes ctx.V 0) 1

  -- Keep the score scale inside the replay boundary for both cotangent contractions.
  let dQ := Internal.scaledProducts (fun _ : Fin 1 => dScaledScores)
    (fun _ => ctx.K) (1 / scale) (scaleFirst := true)
  let dK := Internal.scaledProducts (fun _ : Fin 1 => swapAdjacentAxes dScaledScores 0)
    (fun _ => ctx.Q) (1 / scale) (scaleFirst := true)

  (dQ, dK, dV)

/--
Forward-mode JVP for scaled dot-product attention.

This differentiates the pure attention equation

`Out = softmax(mask(Q Kᵀ / sqrt(d))) V`

in the direction `(dQ,dK,dV)`. Over real scalars, hard-masked blocked logits have zero
forward weight,
so their tangent contribution is zero in `softmaxBackwardFromWeightsSpec`. Upstream centering
preserves that zero-row convention. The row-wise softmax Jacobian is symmetric, so the same
centered formula serves as both VJP and JVP once the forward weights are known.

When a tangent contraction overflows before normalization, finite-coefficient replay carries both
original product terms through the pullback. This also preserves zero tangents for a singleton
softmax without rounding an unrepresentable intermediate tangent to infinity first.
-/
def scaledDotProductAttentionJvp
  {nQ nK dModel : Nat} {h1 : nQ ≠ 0} {h2 : nK ≠ 0}
  (ctx : AttentionContext α nQ nK dModel h1 h2)
  (dQ : Tensor α [nQ, dModel])
  (dK dV : Tensor α [nK, dModel]) :
  Tensor α [nQ, dModel] :=
  let scale := attentionScaleDenom (α := α) dModel
  let scaledScores := Internal.scaledProducts (fun _ : Fin 1 => ctx.Q)
    (fun _ => swapAdjacentAxes ctx.K 0) (1 / scale)
  -- Enclose both terms together: either unscaled tangent product may overflow on its own.
  let dScaledScores := Internal.scaledProducts
    (fun i : Fin 2 => if i.val == 0 then dQ else ctx.Q)
    (fun i => if i.val == 0 then swapAdjacentAxes ctx.K 0 else swapAdjacentAxes dK 0)
    (1 / scale)
  let attentionWeights :=
    match ctx.mask with
    | none => Activation.softmaxSpec 1 scaledScores
    | some m => hardMaskedSoftmaxSpec scaledScores m
  let dAttentionWeights :=
    Internal.pullbackProducts attentionWeights dScaledScores
      (fun i : Fin 2 => if i.val == 0 then dQ else ctx.Q)
      (fun i => if i.val == 0 then swapAdjacentAxes ctx.K 0 else swapAdjacentAxes dK 0)
      (1 / scale)
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
    {α : Type} [TorchLean.Storage α]
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
    {α : Type} [TorchLean.Storage α]
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

The record lives beside the backward pass that produces it. All four fields are named after the
corresponding field of `MultiHeadAttention`, so `{ queryWeight, keyWeight, valueWeight,
outputWeight }` works with the anonymous constructor at every call site.

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

end Spec
