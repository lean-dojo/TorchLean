/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Layers.Activation
public import TorchLean.Tensor.Layers.Softmax
public import TorchLean.Tensor.Contract

/-!
# Attention (spec layer)

Scaled dot-product attention over any leading batch axes, `Tensor.attention`, and the
single-sequence specification `Attention.scaledDotProduct` with its multi-head wrapper, each with a
backward pass and a forward-mode derivative.

`Attention(Q,K,V) = softmax(Q Kᵀ / √d) V`

The definitions follow the mathematics of deep learning libraries, PyTorch above all, as pure
functions on `TorchLean.Tensor`, so that proofs about shapes and gradients, reference
implementations for runtime extraction and verification backends such as interval semantics can
share them.

## Shapes and conventions

`Attention.scaledDotProduct` models one batch element. `Tensor.attention`,
`Attention.scaledProducts` and the kernels `Attention.weights`, `Attention.values`,
`Attention.valuesBackward` and `Attention.valuesJVP` take any leading batch axes and
read the matrices at each batch coordinate, with one optional `[rows, keys]` mask shared by every
batch entry.

`Tensor.attention` takes the softmax schedule as an argument. `twoPass` normalizes every score row
before the value product. `online blocks` is FlashAttention's schedule, an IO-aware strategy that
tiles the computation and keeps online softmax summaries so the full score matrix is not
materialized. For one query it makes one pass over blocks of keys of any size, keeping the running
maximum `m`, the normalizer `l = ∑ exp (s - m)` and the weighted sum `o = ∑ exp (s - m) * v`. The
maximum starts at the first allowed score that is not `-∞`, or at `-∞` when every allowed score is
`-∞`. A block raises the maximum to `m'`, multiplies `l` and `o` by `exp (m - m')` and adds its
own terms, and the output is `o / l`. Over the reals this is the softmax average for every
partition of the keys into blocks (`Softmax.onlineAverage_eq_div`, `Tensor.attention_online_eq`),
and `TorchLean.Verify.RuntimeApprox.OnlineSoftmax` bounds its rounding error in every IEEE format.
No memory-traffic model is defined. The backward and JVP recompute the two-pass weights for both
schedules.

The CUDA runtime composes attention in Lean over LibTorch numerical primitives
(`TorchLean/NN/LibTorch/Attention.lean`). It materializes full score
matrices, saves probabilities and does not select a fused FlashAttention implementation. Lean does
not verify LibTorch or CUDA code, and a fused implementation needs its own refinement argument.

Core shapes:

- `Q : (nQ × d)` queries
- `K : (nK × d)` keys
- `V : (nK × dV)` values

`Attention.scaledDotProduct` takes `dV = d`, as many transformer blocks do, and `Tensor.attention`
allows any value width.

The optional Boolean mask has shape `(nQ × nK)`. Blocked entries get a zero numerator before row
normalization, the finite-scalar encoding of the PyTorch pattern
`scores.masked_fill(~mask, -torch.inf)`, so over real scalars with a nonzero denominator their
attention weight is zero.

In the two-pass schedule a row with no allowed entry gets zero weights from its own branch, with no
division, whatever its scores. The product with `V` follows the arithmetic of the scalar backend,
so in IEEE arithmetic a zero weight against an infinite value gives `0 * ∞`, a NaN. On a partially
allowed row, exceptional allowed scores can also make the denominator and the blocked weights NaN.
The online schedule folds over the allowed keys only, so a blocked score or value does not enter
its output, and a row without an allowed key is zero.

Vaswani et al.'s equations went in by hand: eq. (1) of Sec. 3.2.1 gives the weights and the
output, Sec. 3.2.2 the heads of `MultiHeadAttention.forward`, and Sec. 3.2.3 the `-∞` mask.
`Tensor.attention` takes the scale as an argument, and `Attention.scaledDotProduct` fixes it at
`1 / √d`, the default of `torch.nn.functional.scaled_dot_product_attention` (PyTorch 2.8 docs).
Like that page's reference code, we multiply the scores by it, where eq. (1) divides by `√d`. A
mask of `MultiHeadAttention.forward` is the negation of a Boolean `attn_mask` for
`torch.nn.MultiheadAttention`, where `True` blocks a position, and `numHeads * headDim` need not
equal `dModel`, where torch gives each head `embed_dim // num_heads` features.

PyTorch analogy:

- `Tensor.attention` and `Attention.scaledDotProduct` correspond to
  `torch.nn.functional.scaled_dot_product_attention`, with a Boolean mask read as a true `-∞`
  mask. The multiplier `weights` of `Tensor.attention` carries dropout on the probabilities.
- `MultiHeadAttention.forward` corresponds to the core computation inside `nn.MultiheadAttention`
  / transformer blocks, ignoring biases and dropout.

References:
- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz
  Kaiser, Illia Polosukhin, "Attention Is All You Need", NIPS 2017, arXiv:1706.03762.
- Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré, "FlashAttention: Fast and
  Memory-Efficient Exact Attention with IO-Awareness" (2022), arXiv:2205.14135.
- Tri Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"
  (2023), arXiv:2307.08691. Its Algorithm 1 is the forward pass that `online blocks` follows.
- Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao,
  "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision" (2024),
  arXiv:2407.08608.
-/

@[expose] public section


open TorchLean.Tensor TorchLean.Softmax
open TorchLean.Shape (size)

namespace TorchLean

variable {α : Type} [Storage α] [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)]

/-!
## Scaled Dot-Product Attention

We keep the single-head primitive `Attention.scaledDotProduct` separate. It is the core
mathematical object, which multi-head attention reuses for every head, and a good target for proofs
and for comparisons of the specification with the runtime.
-/

/-!
## Boolean masks

TorchLean uses the Boolean `attn_mask` convention of PyTorch's
`torch.nn.functional.scaled_dot_product_attention`: `true` means a key/value position is
**allowed to be attended to**, and `false` means it is blocked, with a softmax numerator of
exactly zero.

If an entire row is `false`, every output weight in that row is zero.
-/

/-- A `(nQ × nK)` mask where every position is allowed (`true`). -/
def Attention.allTrueMask (nQ nK : Nat) : Tensor Bool [nQ, nK] :=
  Tensor.dim (fun _ => Tensor.dim (fun _ => Tensor.scalar true))

/-- A `(nQ × nK)` mask where every position is blocked (`false`). -/
def Attention.allFalseMask (nQ nK : Nat) : Tensor Bool [nQ, nK] :=
  Tensor.dim (fun _ => Tensor.dim (fun _ => Tensor.scalar false))

/-- Causal (lower-triangular) self-attention mask of shape `(n, n)`.

`mask[i,j] = true` iff `j ≤ i`, so each query position can attend to itself and to past positions.
-/
def Attention.causalMask (n : Nat) : Tensor Bool [n, n] :=
  Tensor.dim (fun i =>
    Tensor.dim (fun j =>
      Tensor.scalar (decide (j.1 ≤ i.1))))

/-- Future-only (upper-triangular) self-attention mask of shape `(n, n)`: `mask[i,j] = true` iff
`i < j`, the complement of `causalMask`. -/
def Attention.futureMask (n : Nat) : Tensor Bool [n, n] :=
  Tensor.dim (fun i =>
    Tensor.dim (fun j =>
      Tensor.scalar (decide (i.1 < j.1))))

/-- Bundled inputs and mask needed for scaled dot-product attention. -/
structure Attention.Inputs (α : Type) [Storage α]
  (nQ nK dModel : Nat) (h1 : nQ ≠ 0) (h2 : nK ≠ 0) where
  /-- The `nQ` query rows. -/
  query : Tensor α [nQ, dModel]
  /-- The `nK` key rows. -/
  key : Tensor α [nK, dModel]
  /-- The `nK` value rows, one for each key. -/
  value : Tensor α [nK, dModel]
  /-- The optional mask: key `j` is blocked for query `i` where `mask[i,j]` is false. -/
  mask : Option (Tensor Bool [nQ, nK])

/-- The denominator `√dModel` of scaled dot-product attention, and `1` when `dModel = 0`.

TorchLean's tensor shapes admit zero dimensions, and when `dModel = 0` the result has no feature
coordinates, so the denominator `1` gives the unique empty-feature result with no division by zero.
The scalar context constructs the root from the count, so a binary backend can recover a finite
root even when casting the dimension itself would overflow. -/
def Attention.scaleDenom (dModel : Nat) : α :=
  if dModel = 0 then 1 else Context.sqrtNat dModel

/-- VJP of a softmax-like row normalization along the last axis from its known forward weights,
which by the symmetry of the Jacobian is also its JVP:

`dScores = weights ⊙ (dWeights - Σⱼ dWeightsⱼ * weightsⱼ)`.

For ordinary softmax `weights = softmax(scores)`. For a hard-masked softmax over real scalars, with
`weights = 0` at blocked entries, the same formula gives blocked logits zero gradient, a zero
product that exceptional floating-point operands need not give.

Attention uses `center := true`, which subtracts the upstream coordinate at a maximal weight before
taking the weighted sum. For normalized real weights the derivative is unchanged, and an all-zero
row gives zero under both formulas. In floating point the centered form keeps a small gradient at a
dominant logit after its probability rounds to one. The default computes the uncentered algebraic
formula, which applies to arbitrary supplied weights whose sum need not be one. A scalar, which has
no row, gives zero.
-/
def Softmax.backwardFromWeights {s : Shape} (weights dWeights : Tensor α s)
    (center : Bool := false) : Tensor α s :=
  match s with
  | .scalar => Tensor.scalar 0
  | .dim 0 .scalar => dWeights
  | .dim (Nat.succ n) .scalar =>
      if center then
        let pivot : Fin (Nat.succ n) := Softmax.pivot weights
        pullback weights dWeights (dWeights.getScalar pivot)
      else
        let rowDot : α := Tensor.sum (dWeights * weights)
        weights * (dWeights - Tensor.full _ rowDot)
  | .dim n inner =>
      Tensor.dim (fun i : Fin n =>
        backwardFromWeights (s := inner)
          (Tensor.unstack weights i) (Tensor.unstack dWeights i) (center := center))

namespace Attention

/-- The exact coefficients of entry `(row, column)` of the scaled contraction
`scale * Σ_g left g * right g`, or `none` when an operand has no finite encoding. -/
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

/-- Entry `(row, column)` of the scaled contraction, rounded once from the coefficients of
`productCoefficients?`. An exact zero takes the sign of `ordinary` when that is zero too. -/
@[noinline] def replayProducts? {α : Type} [Storage α] [Context α]
    {m n p groups : Nat} (left : Fin groups → Tensor α [m, n])
    (right : Fin groups → Tensor α [n, p]) (scale ordinary : α)
    (row : Fin m) (column : Fin p) : Option α := do
  let ops := (inferInstance : Numeric.QuotientArithmetic α)
  let result ← productCoefficients? left right scale row column
  let rounded ← ops.decode result
  pure <| if result.primal == 0 && ordinary == 0 then ops.copyPrimal ordinary rounded else rounded

/-- Scaled sums of the matrix products of one batch slice, the kernel of `scaledProducts`.

One product serves the forward pass or the VJP and two serve the JVP's joint contraction, and
`scaleFirst` scales the left operand before the product, the VJP's convention. On a supported
carrier the checked arithmetic follows the ordinary left-fold schedule within and between products.
An entry that loses range or absorbs a nonzero term is replayed from the exact coefficient trees of
every product and the scale, then rounded once, and without finite encodings of its inputs it keeps
the ordinary result. Unsupported carriers compute the ordinary expression.

The replay belongs to this reference scalar semantics, and an expanded IR or a foreign kernel need
not reproduce it.
-/
def scaledProductsSlice {α : Type} [Storage α] [Context α]
    {m n p groups : Nat} (left : Fin groups → Tensor α [m, n])
    (right : Fin groups → Tensor α [n, p]) (scale : α) (scaleFirst : Bool := false) :
    Tensor α [m, p] :=
  let ops := (inferInstance : Numeric.QuotientArithmetic α)
  if !ops.supported then
    let combined := match groups, left, right with
      | 0, _, _ => Tensor.zeros [m, p]
      | groups + 1, left, right =>
          let product := fun group => Tensor.matmul
            (if scaleFirst then Tensor.scale (left group) scale else left group) (right group)
          (List.finRange groups).foldl (fun total group =>
            total + product group.succ) (product 0)
    if scaleFirst then combined else Tensor.scale combined scale
  else
    Tensor.Rep.ofFn fun coordinate =>
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

/-- On an unsupported carrier, a one-product slice is the matrix product scaled afterwards, or
with `scaleFirst` the product of the scaled left operand. -/
theorem scaledProductsSlice_one {α : Type} [Storage α] [Context α]
    (h : Numeric.QuotientArithmetic.supported (α := α) = false) {m n p : Nat}
    (a : Tensor α [m, n]) (b : Tensor α [n, p]) (scale : α) (scaleFirst : Bool) :
    scaledProductsSlice (fun _ : Fin 1 => a) (fun _ => b) scale scaleFirst =
      if scaleFirst then Tensor.matmul (Tensor.scale a scale) b
      else Tensor.scale (Tensor.matmul a b) scale := by
  cases scaleFirst <;> simp [scaledProductsSlice, h]

/-- On an unsupported carrier, the JVP's product-rule slice is `matmul da b + matmul a db` scaled
afterwards, or with `scaleFirst` the same sum with each left operand scaled before its product. -/
theorem scaledProductsSlice_two {α : Type} [Storage α] [Context α]
    (h : Numeric.QuotientArithmetic.supported (α := α) = false) {m n p : Nat}
    (a da : Tensor α [m, n]) (b db : Tensor α [n, p]) (scale : α) (scaleFirst : Bool) :
    scaledProductsSlice (fun i : Fin 2 => if i.val == 0 then da else a)
        (fun i => if i.val == 0 then b else db) scale scaleFirst =
      if scaleFirst then
        Tensor.matmul (Tensor.scale da scale) b + Tensor.matmul (Tensor.scale a scale) db
      else Tensor.scale (Tensor.matmul da b + Tensor.matmul a db) scale := by
  cases scaleFirst <;> simp [scaledProductsSlice, h, List.finRange_succ]

/-- Scaled sums of matrix products over a shared batch prefix, used by attention's forward and
differential rules: each entry is the entry of `scaledProductsSlice` on the batch slices of the
operands at its batch coordinate. -/
def scaledProducts {α : Type} [Storage α] [Context α]
    {batch : Shape} {m n p groups : Nat}
    (left : Fin groups → Tensor α (batch ++ [m, n]))
    (right : Fin groups → Tensor α (batch ++ [n, p])) (scale : α) (scaleFirst : Bool := false) :
    Tensor α (batch ++ [m, p]) :=
  Tensor.Rep.ofFn fun c =>
    let split := Shape.Coord.appendEquiv batch [m, p]
      c
    scaledProductsSlice (fun group => (left group).batchSlice split.1)
      (fun group => (right group).batchSlice split.1) scale scaleFirst split.2

/-- An entry of `scaledProducts` is the entry of `scaledProductsSlice` on the batch slices at its
batch coordinate. -/
theorem scaledProducts_apply {α : Type} [Storage α] [Context α] {batch : Shape}
    {m n p groups : Nat} (left : Fin groups → Tensor α (batch ++ [m, n]))
    (right : Fin groups → Tensor α (batch ++ [n, p])) (scale : α) (scaleFirst : Bool)
    (c : (batch ++ [m, p]).Coord) :
    scaledProducts left right scale scaleFirst c =
      let split := Shape.Coord.appendEquiv batch [m, p]
        c
      scaledProductsSlice (fun group => (left group).batchSlice split.1)
        (fun group => (right group).batchSlice split.1) scale scaleFirst split.2 :=
  Tensor.Rep.get_ofFn _ c

/-- A left fold of tensor sums, read at one coordinate, is the fold of the entries there. -/
private theorem foldl_add_apply {α β : Type} [Storage α] [Add α] {s : Shape}
    (f : β → Tensor α s) (c : s.Coord) :
    ∀ (l : List β) (t : Tensor α s),
      (l.foldl (fun total g => total + f g) t) c = l.foldl (fun total g => total + f g c) (t c)
  | [], _ => rfl
  | _ :: l, t => by
      simp only [List.foldl_cons]
      rw [foldl_add_apply f c l, Tensor.add_apply]

/-- Entry `(i, j)` of `scaledProductsSlice` reads only row `i` of each left operand and column `j`
of each right operand, in both of its branches and in the replay: operands that agree there give
the same entry. -/
theorem scaledProductsSlice_congr {α : Type} [Storage α] [Context α]
    {m n p m' p' groups : Nat} {left : Fin groups → Tensor α [m, n]}
    {right : Fin groups → Tensor α [n, p]} {left' : Fin groups → Tensor α [m', n]}
    {right' : Fin groups → Tensor α [n, p']} (scale : α) (scaleFirst : Bool)
    {i : Fin m} {j : Fin p} {i' : Fin m'} {j' : Fin p'}
    (hl : ∀ g k, get2 (left g) i k = get2 (left' g) i' k)
    (hr : ∀ g k, get2 (right g) k j = get2 (right' g) k j') :
    scaledProductsSlice left right scale scaleFirst (i, j, PUnit.unit) =
      scaledProductsSlice left' right' scale scaleFirst (i', j', PUnit.unit) := by
  have hreplay : ∀ ordinary, replayProducts? left right scale ordinary i j =
      replayProducts? left' right' scale ordinary i' j' := by
    intro ordinary
    simp only [replayProducts?, productCoefficients?, hl, hr]
  unfold scaledProductsSlice
  by_cases hs : (inferInstance : Numeric.QuotientArithmetic α).supported
  · simp only [hs, Bool.not_true, Bool.false_eq_true, ite_false, Tensor.Rep.get_ofFn]
    cases groups with
    | zero => rfl
    | succ g => simp only [hl, hr, hreplay]
  · simp only [hs, Bool.not_false, ite_true]
    have hrow : ∀ (A : Tensor α [m, n]) (A' : Tensor α [m', n]) (B : Tensor α [n, p])
        (B' : Tensor α [n, p']), (∀ k, get2 A i k = get2 A' i' k) →
        (∀ k, get2 B k j = get2 B' k j') →
        Tensor.matmul A B (i, j, PUnit.unit) = Tensor.matmul A' B' (i', j', PUnit.unit) := by
      intro A A' B B' ha hb
      rw [Tensor.matmul_apply, Tensor.matmul_apply]
      simp only [ha, hb]
    have hscale : ∀ g k, get2 (Tensor.scale (left g) scale) i k =
        get2 (Tensor.scale (left' g) scale) i' k := by
      intro g k
      simp only [get2_eq_apply, Tensor.scale, Tensor.map_apply] at hl ⊢
      rw [hl]
    cases groups with
    | zero =>
        cases scaleFirst <;>
          simp [Tensor.scale, Tensor.map_apply]
    | succ g =>
        cases scaleFirst
        · simp only [Bool.false_eq_true, ite_false, Tensor.scale,
            Tensor.map_apply, foldl_add_apply, hrow _ _ _ _ (hl _) (hr _)]
        · simp only [ite_true, foldl_add_apply, hrow _ _ _ _ (hscale _) (hr _)]

/-- `scaledProducts` computed one outer batch index at a time, without reading a slice per entry.
-/
def scaledProductsFast {α : Type} [Storage α] [Context α]
    {batch : Shape} {m n p groups : Nat}
    (left : Fin groups → Tensor α (batch ++ [m, n]))
    (right : Fin groups → Tensor α (batch ++ [n, p])) (scale : α) (scaleFirst : Bool := false) :
    Tensor α (batch ++ [m, p]) :=
  match batch with
  | .scalar => scaledProductsSlice left right scale scaleFirst
  | .dim _ _ => Tensor.dim fun index =>
      scaledProductsFast (fun group => (left group).unstack index)
        (fun group => (right group).unstack index) scale scaleFirst

/-- `scaledProductsFast` computes `scaledProducts`. -/
@[csimp] theorem scaledProductsFast_eq :
    @scaledProducts = @scaledProductsFast := by
  funext α _ _ batch m n p groups left right scale scaleFirst
  refine Tensor.Rep.ext fun c => ?_
  rw [scaledProducts_apply]
  exact Tensor.batchwise_apply (shapes := Sum.elim (fun _ : Fin groups => [m, n]) fun _ => [n, p])
    (fun inputs => scaledProductsFast (fun group => inputs (.inl group))
      (fun group => inputs (.inr group)) scale scaleFirst)
    (fun _ => rfl) (fun | .inl group => left group | .inr group => right group) c |>.symm

/-- `scale` times the matrix product of `left` and `right`, with both batch prefixes broadcast to
`batch`: one `scaledProducts` contraction, so the scale lies inside the replay of each entry. -/
def scaledMatmul {α : Type} [Storage α] [Context α]
    {batchA batchB batch : Shape} {m n p : Nat}
    (broadcastA : Shape.CanBroadcastTo batchA batch)
    (broadcastB : Shape.CanBroadcastTo batchB batch)
    (left : Tensor α (batchA ++ [m, n]))
    (right : Tensor α (batchB ++ [n, p])) (scale : α) :
    Tensor α (batch ++ [m, p]) :=
  have hA := Shape.CanBroadcastTo.concat_right [m, n] broadcastA
  let a := Tensor.broadcast left _ hA.dims hA.broadcastDims
  have hB := Shape.CanBroadcastTo.concat_right [n, p] broadcastB
  let b := Tensor.broadcast right _ hB.dims hB.broadcastDims
  scaledProducts (fun _ : Fin 1 => a) (fun _ => b) scale

/-- The tangent `scale * (deltaLeft * right + left * deltaRight)` of `scaledMatmul`. Both
product-rule terms share one replay, so large terms of opposite sign can cancel before rounding. -/
def scaledMatmulJVP {α : Type} [Storage α] [Context α]
    {batchA batchB batch : Shape} {m n p : Nat}
    (broadcastA : Shape.CanBroadcastTo batchA batch)
    (broadcastB : Shape.CanBroadcastTo batchB batch)
    (left deltaLeft : Tensor α (batchA ++ [m, n]))
    (right deltaRight : Tensor α (batchB ++ [n, p])) (scale : α) :
    Tensor α (batch ++ [m, p]) :=
  have hA := Shape.CanBroadcastTo.concat_right [m, n] broadcastA
  let a := Tensor.broadcast left _ hA.dims hA.broadcastDims
  let da := Tensor.broadcast deltaLeft _ hA.dims hA.broadcastDims
  have hB := Shape.CanBroadcastTo.concat_right [n, p] broadcastB
  let b := Tensor.broadcast right _ hB.dims hB.broadcastDims
  let db := Tensor.broadcast deltaRight _ hB.dims hB.broadcastDims
  scaledProducts (fun group : Fin 2 => if group.val == 0 then da else a)
    (fun group => if group.val == 0 then b else db) scale

/-- Cotangents of `scaledMatmul`: `scale * gradient * rightᵀ` for `left` and
`scale * leftᵀ * gradient` for `right`, each summed back over its broadcast batch axes, with the
scale inside each product's replay. -/
def scaledMatmulBackward {α : Type} [Storage α] [Context α]
    {batchA batchB batch : Shape} {m n p : Nat}
    (broadcastA : Shape.CanBroadcastTo batchA batch)
    (broadcastB : Shape.CanBroadcastTo batchB batch)
    (left : Tensor α (batchA ++ [m, n]))
    (right : Tensor α (batchB ++ [n, p])) (gradient : Tensor α (batch ++ [m, p]))
    (scale : α) : Tensor α (batchA ++ [m, n]) × Tensor α (batchB ++ [n, p]) :=
  have hA := Shape.CanBroadcastTo.concat_right [m, n] broadcastA
  let a := Tensor.broadcast left _ hA.dims hA.broadcastDims
  have hB := Shape.CanBroadcastTo.concat_right [n, p] broadcastB
  let b := Tensor.broadcast right _ hB.dims hB.broadcastDims
  let bt : Tensor α (batch ++ [p, n]) :=
    Tensor.permute b (Shape.adjacentSwap (batch ++ [n, p]) batch.rank) (batch ++ [p, n])
  let transposed : Tensor α (batch ++ [n, m]) :=
    Tensor.permute a (Shape.adjacentSwap (batch ++ [m, n]) batch.rank) (batch ++ [n, m])
  (Tensor.reduceBroadcast hA.dims hA.broadcastDims
      (scaledProducts (fun _ : Fin 1 => gradient) (fun _ => bt) scale),
    Tensor.reduceBroadcast hB.dims hB.broadcastDims
      (scaledProducts (fun _ : Fin 1 => transposed) (fun _ => gradient) scale))

/-- One row of the centered softmax pullback, replayed from the exact coefficients of its incoming
gradient by `Softmax.replayPullback?` and anchored at the pivot of `weights`. -/
def replayPullbackCoefficientsRow? {α : Type} [Storage α] [Context α] {p : Nat}
    (weights ordinary : Tensor α [p])
    (coefficients : Fin p → Option
      (Numeric.QuotientCoefficients (Numeric.QuotientArithmetic.depth α))) :
    Option (Tensor α [p]) :=
  match p, weights, ordinary, coefficients with
  | 0, _, ordinary, _ => some ordinary
  | p + 1, weights, ordinary, coefficients =>
      let pivot : Fin (p + 1) := Softmax.pivot weights
      Softmax.replayPullback? weights ordinary coefficients (coefficients pivot)

/-- `replayPullbackCoefficientsRow?` with the exact coefficients of row `row` of the scaled
contraction of `left` and `right` as its incoming gradient. -/
def replayPullbackProductsRow? {α : Type} [Storage α] [Context α] {m n p groups : Nat}
    (weights ordinary : Tensor α [p]) (left : Fin groups → Tensor α [m, n])
    (right : Fin groups → Tensor α [n, p]) (scale : α) (row : Fin m) :
    Option (Tensor α [p]) :=
  replayPullbackCoefficientsRow? weights ordinary (productCoefficients? left right scale row)

/-- The exact coefficients of entry `(row, column)` of `left * right` times the same entry of
`factor`, or `none` when an operand has no finite encoding. -/
def weightedProductCoefficients? {α : Type} [Storage α] [Context α] {m n p : Nat}
    (left : Tensor α [m, n]) (right : Tensor α [n, p])
    (factor : Tensor α [m, p]) (row : Fin m) (column : Fin p) :
    Option (Numeric.QuotientCoefficients (Numeric.QuotientArithmetic.depth α)) := do
  let ops := (inferInstance : Numeric.QuotientArithmetic α)
  let product ← productCoefficients? (fun _ : Fin 1 => left) (fun _ => right) 1 row column
  let coefficient ← ops.encode (get2 factor row column)
  pure (Numeric.QuotientCoefficients.mul product coefficient)

/-- The exact coefficients of entry `(key, column)` of `(weights ⊙ multiplier)ᵀ * gradient`, both
pointwise factors of every term unrounded, or `none` when an operand has no finite encoding. -/
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

/-- Entry `(row, column)` of `left * right` times `factor`, rounded once from the coefficients of
`weightedProductCoefficients?`. An exact zero takes the sign of `ordinary` when that is zero too. -/
@[noinline] def replayWeightedProducts? {α : Type} [Storage α] [Context α] {m n p : Nat}
    (left : Tensor α [m, n]) (right : Tensor α [n, p]) (factor : Tensor α [m, p]) (ordinary : α)
    (row : Fin m) (column : Fin p) : Option α := do
  let ops := (inferInstance : Numeric.QuotientArithmetic α)
  let result ← weightedProductCoefficients? left right factor row column
  let rounded ← ops.decode result
  pure <| if result.primal == 0 && ordinary == 0 then ops.copyPrimal ordinary rounded else rounded

/-- Entry `(key, column)` of `(weights ⊙ multiplier)ᵀ * gradient`, rounded once from the
coefficients of `weightedValueCoefficients?`. An exact zero keeps the sign of a zero `ordinary`. -/
@[noinline] def replayWeightedValues? {α : Type} [Storage α] [Context α] {rows keys width : Nat}
    (weights multiplier : Tensor α [rows, keys]) (gradient : Tensor α [rows, width]) (ordinary : α)
    (key : Fin keys) (column : Fin width) : Option α := do
  let ops := (inferInstance : Numeric.QuotientArithmetic α)
  let total ← weightedValueCoefficients? weights multiplier gradient key column
  let rounded ← ops.decode total
  pure <| if total.primal == 0 && ordinary == 0 then ops.copyPrimal ordinary rounded else rounded

/-- Conservative range test for an ordinary product that feeds a further map: whether a product or
partial sum in some entry of `left * right` loses range or absorbs a nonzero term, and `false` on an
unsupported carrier. Only the Boolean leaves the test, and the caller keeps its own schedule for the
values. -/
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

/-- The centered softmax pullback of `gradient` through `weights`, row by row, for a `gradient`
that is the ordinary scaled contraction of `left` and `right`.

The ordinary result stands unless the carrier is supported and an entry of a row loses range. That
row is then replayed from the exact coefficients of the contraction through the pullback, which can
give a finite result even where `gradient` cannot represent the intermediate values. Ordinary
softmax uses the same coefficient replay.
-/
def pullbackProducts {α : Type} [Storage α] [Context α] {m n p groups : Nat}
    (weights gradient : Tensor α [m, p]) (left : Fin groups → Tensor α [m, n])
    (right : Fin groups → Tensor α [n, p]) (scale : α) : Tensor α [m, p] :=
  let ordinary := backwardFromWeights weights gradient (center := true)
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

/-- An entry of `pullbackProducts` is read from the centered softmax pullback of its row, so that
no case split on either extent remains. -/
theorem pullbackProducts_apply {α : Type} [Storage α] [Context α] {m n p groups : Nat}
    (weights gradient : Tensor α [m, p]) (left : Fin groups → Tensor α [m, n])
    (right : Fin groups → Tensor α [n, p]) (scale : α) (c : Shape.Coord [m, p]) :
    pullbackProducts weights gradient left right scale c =
      let values := backwardFromWeights (weights.unstack c.1) (gradient.unstack c.1)
        (center := true)
      let ops := (inferInstance : Numeric.QuotientArithmetic α)
      if !ops.supported then values c.2
      else if !(List.finRange p).any (fun i => (ops.addWithFlags (values.getScalar i) 0).2 != 0)
        then values c.2
      else
        (replayPullbackProductsRow? (weights.unstack c.1) values left right scale c.1).getD
          values c.2 := by
  obtain ⟨row, column, ⟨⟩⟩ := c
  cases p with
  | zero => exact Fin.elim0 column
  | succ p =>
      cases m with
      | zero => exact Fin.elim0 row
      | succ m =>
          have hrow : (backwardFromWeights weights gradient (center := true)).unstack
              row = backwardFromWeights (weights.unstack row) (gradient.unstack row)
                (center := true) := Tensor.unstack_dim _ row
          unfold pullbackProducts
          dsimp only
          rw [← hrow]
          split
          · exact (Tensor.unstack_apply _ _ _).symm
          · simp only [Tensor.dim_apply]
            split <;> rfl

end Attention

/-- The `Attention.weights` kernel on one score matrix: the row softmax, or the hard-masked
softmax when a mask is given, so a row with no allowed key has zero weights. -/
def Attention.weightsSlice {α : Type} [Storage α] [Context α] {rows keys : Nat}
    (scores : Tensor α [rows, keys]) (mask : Option (Tensor Bool [rows, keys])) :
    Tensor α [rows, keys] :=
  match mask with
  | none => Tensor.softmax scores ⟨1, Nat.one_lt_two⟩
  | some mask => hardMasked scores mask

/-- Row weights of score matrices with any leading batch axes, as `Tensor.attention` uses them: each
matrix is normalized by `Attention.weightsSlice`, sharing one optional mask. -/
@[irreducible] def Attention.weights {α : Type} [Storage α] [Context α]
    {batch : Shape} {rows keys : Nat} (scores : Tensor α (batch ++ [rows, keys]))
    (mask : Option (Tensor Bool [rows, keys]) := none) :
    Tensor α (batch ++ [rows, keys]) :=
  Tensor.Rep.ofFn fun c =>
    let split := Shape.Coord.appendEquiv batch [rows, keys]
      c
    Attention.weightsSlice (scores.batchSlice split.1) mask split.2

/-- An entry of `Attention.weights` is the entry of `Attention.weightsSlice` on the
score matrix at its batch coordinate. -/
theorem Attention.weights_apply {α : Type} [Storage α] [Context α] {batch : Shape}
    {rows keys : Nat} (scores : Tensor α (batch ++ [rows, keys]))
    (mask : Option (Tensor Bool [rows, keys])) (c : (batch ++ [rows, keys]).Coord) :
    Attention.weights scores mask c =
      let split := Shape.Coord.appendEquiv batch [rows, keys]
        c
      Attention.weightsSlice (scores.batchSlice split.1) mask split.2 := by
  unfold Attention.weights
  exact Tensor.Rep.get_ofFn _ c

/-- `Attention.weights` computed one outer batch index at a time. -/
def Attention.weightsFast {α : Type} [Storage α] [Context α]
    {batch : Shape} {rows keys : Nat} (scores : Tensor α (batch ++ [rows, keys]))
    (mask : Option (Tensor Bool [rows, keys]) := none) :
    Tensor α (batch ++ [rows, keys]) :=
  match batch, scores with
  | .scalar, scores => Attention.weightsSlice scores mask
  | .dim _ _, scores => Tensor.dim fun i =>
      weightsFast (scores.unstack i) mask

/-- `Attention.weightsFast` computes `Attention.weights`. -/
@[csimp] theorem Attention.weightsFast_eq :
    @Attention.weights = @Attention.weightsFast := by
  funext α _ _ batch rows keys scores mask
  refine Tensor.Rep.ext fun c => ?_
  rw [Attention.weights_apply]
  exact Tensor.batchwise_apply (shapes := fun _ : Unit => [rows, keys])
    (fun inputs => Attention.weightsFast (inputs ()) mask) (fun _ => rfl)
    (fun _ => scores) c |>.symm

/-- Mix value rows from saved weights over any leading batch axes: each matrix of weights, times the
matching matrix of `multiplier` when one is given, is multiplied by the value matrix, keeping the
pointwise product before the contraction. -/
@[irreducible] def Attention.values {α : Type} [Storage α] [Context α]
    {batch : Shape} {rows keys width : Nat}
    (weights : Tensor α (batch ++ [rows, keys]))
    (value : Tensor α (batch ++ [keys, width]))
    (multiplier : Option (Tensor α (batch ++ [rows, keys]))) :
    Tensor α (batch ++ [rows, width]) :=
  Tensor.Rep.ofFn fun c =>
    let split := Shape.Coord.appendEquiv batch [rows, width]
      c
    Tensor.matmul (multiplier.elim (weights.batchSlice split.1) fun multiplier =>
        weights.batchSlice split.1 * multiplier.batchSlice split.1)
      (value.batchSlice split.1) split.2

/-- The weighted value product at a coordinate: the product of the matrices at its batch
coordinate. -/
theorem Attention.values_apply {α : Type} [Storage α] [Context α]
    {batch : Shape} {rows keys width : Nat} (weights : Tensor α (batch ++ [rows, keys]))
    (value : Tensor α (batch ++ [keys, width]))
    (multiplier : Option (Tensor α (batch ++ [rows, keys])))
    (c : (batch ++ [rows, width]).Coord) :
    Attention.values weights value multiplier c =
      let split := Shape.Coord.appendEquiv batch [rows, width]
        c
      Tensor.matmul (multiplier.elim (weights.batchSlice split.1) fun multiplier =>
          weights.batchSlice split.1 * multiplier.batchSlice split.1)
        (value.batchSlice split.1) split.2 := by
  unfold Attention.values
  exact Tensor.Rep.get_ofFn _ c

/-- `Attention.values` computed one outer batch index at a time. -/
def Attention.valuesFast {α : Type} [Storage α] [Context α]
    {batch : Shape} {rows keys width : Nat}
    (weights : Tensor α (batch ++ [rows, keys]))
    (value : Tensor α (batch ++ [keys, width]))
    (multiplier : Option (Tensor α (batch ++ [rows, keys]))) :
    Tensor α (batch ++ [rows, width]) :=
  match batch, weights, value, multiplier with
  | .scalar, weights, value, none => Tensor.matmul weights value
  | .scalar, weights, value, some multiplier => Tensor.matmul (weights * multiplier) value
  | .dim _ _, weights, value, multiplier => Tensor.dim fun i =>
      valuesFast (weights.unstack i) (value.unstack i)
        (multiplier.map (·.unstack i))

/-- `Attention.valuesFast` computes `Attention.values`. -/
@[csimp] theorem Attention.valuesFast_eq :
    @Attention.values = @Attention.valuesFast := by
  funext α _ _ batch rows keys width weights value multiplier
  refine Tensor.Rep.ext fun c => ?_
  rw [Attention.values_apply]
  cases multiplier with
  | none =>
    exact Tensor.batchwise_apply
      (shapes := Sum.elim (fun _ : Unit => [rows, keys]) fun _ : Unit => [keys, width])
      (fun inputs => Attention.valuesFast (inputs (.inl ())) (inputs (.inr ())) none)
      (fun _ => rfl) (fun | .inl () => weights | .inr () => value) c |>.symm
  | some multiplier =>
    exact Tensor.batchwise_apply
      (shapes := Sum.elim (fun _ : Bool => [rows, keys]) fun _ : Unit => [keys, width])
      (fun inputs => Attention.valuesFast (inputs (.inl true)) (inputs (.inr ()))
        (some (inputs (.inl false))))
      (fun _ => rfl) (fun | .inl true => weights | .inl false => multiplier | .inr () => value) c
      |>.symm

/-- The `Attention.valuesBackward` kernel on one matrix of each operand, giving the score,
value and multiplier cotangents. On a supported carrier a row whose weights are all zero passes no
gradient, and entries that are zero or at risk of range loss are replayed over finite coefficients,
which carry the output cotangent and the values through the multiplier and the centered softmax.
Unsupported scalar carriers use the ordinary local formula. -/
def Attention.valuesBackwardSlice {α : Type} [Storage α] [Context α]
    {rows keys width : Nat} (weights : Tensor α [rows, keys]) (value : Tensor α [keys, width])
    (multiplier : Tensor α [rows, keys]) (gradient : Tensor α [rows, width]) :
    Tensor α [rows, keys] × Tensor α [keys, width] × Tensor α [rows, keys] :=
  let ops := (inferInstance : Numeric.QuotientArithmetic α)
  let active := fun row => !(List.finRange keys).all fun key =>
    Context.isZero (get2 weights row key)
  let gradient := if ops.supported then
    Tensor.dim fun row =>
      if active row then gradient.unstack row else Tensor.zeros [width]
    else gradient
  let transposed := Tensor.permute value
      (Shape.adjacentSwap [keys, width] 0)
      (Shape.swapAdjacentAtDepth [keys, width] 0)
  let upstream := Tensor.matmul gradient transposed
  let upstream := if ops.supported then
    Tensor.dim fun row =>
      if active row then upstream.unstack row else Tensor.zeros [keys]
    else upstream
  let ordinary := backwardFromWeights weights (upstream * multiplier)
    (center := true)
  let multiplierGradient := weights * upstream
  let contractionRisk := Attention.productsLoseRange gradient transposed
  let scoreGradient := if !ops.supported then ordinary else Tensor.dim fun row =>
    let previous := ordinary.unstack row
    let coefficient := Attention.weightedProductCoefficients? gradient transposed multiplier row
    if !contractionRisk && !(List.finRange keys).any (fun column =>
        previous.getScalar column == 0 ||
          (ops.addWithFlags (previous.getScalar column) 0).2 != 0 ||
          (ops.mulWithFlags (get2 upstream row column) (get2 multiplier row column)).2 != 0)
      then previous
    else (Attention.replayPullbackCoefficientsRow? (weights.unstack row) previous coefficient)
      |>.getD previous
  let multiplierGradient := if !ops.supported then multiplierGradient else
    Tensor.dim fun row => Tensor.dim fun column => Tensor.scalar <|
      let previous := get2 multiplierGradient row column
      if !contractionRisk && previous != 0 && (ops.addWithFlags previous 0).2 == 0 &&
          (ops.mulWithFlags (get2 weights row column) (get2 upstream row column)).2 == 0
        then previous else
        (Attention.replayWeightedProducts? gradient transposed weights previous row column).getD
          previous
  let valueGradient := Attention.scaledProductsSlice
    (fun _ : Fin 1 => Tensor.permute (weights * multiplier)
        (Shape.adjacentSwap [rows, keys] 0)
        (Shape.swapAdjacentAtDepth [rows, keys] 0)) (fun _ => gradient) 1
  let valueGradient := if !ops.supported then valueGradient else
    Tensor.dim fun key => Tensor.dim fun column => Tensor.scalar <|
      let previous := get2 valueGradient key column
      let factorRisk := (List.finRange rows).any fun row =>
        (ops.mulWithFlags (get2 weights row key) (get2 multiplier row key)).2 != 0
      if !factorRisk && previous != 0 && (ops.addWithFlags previous 0).2 == 0 then previous else
        (Attention.replayWeightedValues? weights multiplier gradient previous key column).getD
          previous
  (scoreGradient, valueGradient, multiplierGradient)

/-- Cotangents of normalized scores, values and their differentiable multiplier over any leading
batch axes: each matrix is that of `Attention.valuesBackwardSlice` at its batch
coordinate.
-/
@[irreducible] def Attention.valuesBackward {α : Type} [Storage α] [Context α]
    {batch : Shape} {rows keys width : Nat}
    (weights : Tensor α (batch ++ [rows, keys]))
    (value : Tensor α (batch ++ [keys, width]))
    (multiplier : Tensor α (batch ++ [rows, keys]))
    (gradient : Tensor α (batch ++ [rows, width])) :
    Tensor α (batch ++ [rows, keys]) × Tensor α (batch ++ [keys, width]) ×
      Tensor α (batch ++ [rows, keys]) :=
  (Tensor.Rep.ofFn fun c =>
      let split := Shape.Coord.appendEquiv batch [rows, keys]
        c
      (Attention.valuesBackwardSlice (weights.batchSlice split.1) (value.batchSlice split.1)
        (multiplier.batchSlice split.1) (gradient.batchSlice split.1)).1 split.2,
    Tensor.Rep.ofFn fun c =>
      let split := Shape.Coord.appendEquiv batch [keys, width]
        c
      (Attention.valuesBackwardSlice (weights.batchSlice split.1) (value.batchSlice split.1)
        (multiplier.batchSlice split.1) (gradient.batchSlice split.1)).2.1 split.2,
    Tensor.Rep.ofFn fun c =>
      let split := Shape.Coord.appendEquiv batch [rows, keys]
        c
      (Attention.valuesBackwardSlice (weights.batchSlice split.1) (value.batchSlice split.1)
        (multiplier.batchSlice split.1) (gradient.batchSlice split.1)).2.2 split.2)

/-- The score cotangent at a coordinate: the score cotangent of the matrices at its batch
coordinate. -/
theorem Attention.valuesBackward_fst_apply {α : Type} [Storage α] [Context α]
    {batch : Shape} {rows keys width : Nat} (weights : Tensor α (batch ++ [rows, keys]))
    (value : Tensor α (batch ++ [keys, width]))
    (multiplier : Tensor α (batch ++ [rows, keys]))
    (gradient : Tensor α (batch ++ [rows, width])) (c : (batch ++ [rows, keys]).Coord) :
    (Attention.valuesBackward weights value multiplier gradient).1 c =
      let split := Shape.Coord.appendEquiv batch [rows, keys]
        c
      (Attention.valuesBackwardSlice (weights.batchSlice split.1) (value.batchSlice split.1)
        (multiplier.batchSlice split.1) (gradient.batchSlice split.1)).1 split.2 := by
  unfold Attention.valuesBackward
  exact Tensor.Rep.get_ofFn _ c

/-- The value cotangent at a coordinate: the value cotangent of the matrices at its batch
coordinate. -/
theorem Attention.valuesBackward_snd_fst_apply {α : Type} [Storage α] [Context α]
    {batch : Shape} {rows keys width : Nat} (weights : Tensor α (batch ++ [rows, keys]))
    (value : Tensor α (batch ++ [keys, width]))
    (multiplier : Tensor α (batch ++ [rows, keys]))
    (gradient : Tensor α (batch ++ [rows, width])) (c : (batch ++ [keys, width]).Coord) :
    (Attention.valuesBackward weights value multiplier gradient).2.1 c =
      let split := Shape.Coord.appendEquiv batch [keys, width]
        c
      (Attention.valuesBackwardSlice (weights.batchSlice split.1) (value.batchSlice split.1)
        (multiplier.batchSlice split.1) (gradient.batchSlice split.1)).2.1 split.2 := by
  unfold Attention.valuesBackward
  exact Tensor.Rep.get_ofFn _ c

/-- The multiplier cotangent at a coordinate: the multiplier cotangent of the matrices at its batch
coordinate. -/
theorem Attention.valuesBackward_snd_snd_apply {α : Type} [Storage α] [Context α]
    {batch : Shape} {rows keys width : Nat} (weights : Tensor α (batch ++ [rows, keys]))
    (value : Tensor α (batch ++ [keys, width]))
    (multiplier : Tensor α (batch ++ [rows, keys]))
    (gradient : Tensor α (batch ++ [rows, width])) (c : (batch ++ [rows, keys]).Coord) :
    (Attention.valuesBackward weights value multiplier gradient).2.2 c =
      let split := Shape.Coord.appendEquiv batch [rows, keys]
        c
      (Attention.valuesBackwardSlice (weights.batchSlice split.1) (value.batchSlice split.1)
        (multiplier.batchSlice split.1) (gradient.batchSlice split.1)).2.2 split.2 := by
  unfold Attention.valuesBackward
  exact Tensor.Rep.get_ofFn _ c

/-- `Attention.valuesBackward` computed one outer batch index at a time. -/
def Attention.valuesBackwardFast {α : Type} [Storage α] [Context α]
    {batch : Shape} {rows keys width : Nat}
    (weights : Tensor α (batch ++ [rows, keys]))
    (value : Tensor α (batch ++ [keys, width]))
    (multiplier : Tensor α (batch ++ [rows, keys]))
    (gradient : Tensor α (batch ++ [rows, width])) :
    Tensor α (batch ++ [rows, keys]) × Tensor α (batch ++ [keys, width]) ×
      Tensor α (batch ++ [rows, keys]) :=
  match batch, weights, value, multiplier, gradient with
  | .scalar, weights, value, multiplier, gradient =>
      Attention.valuesBackwardSlice weights value multiplier gradient
  | .dim _ _, weights, value, multiplier, gradient =>
      let result := fun i => valuesBackwardFast (weights.unstack i)
        (value.unstack i) (multiplier.unstack i) (gradient.unstack i)
      (Tensor.dim fun i => (result i).1, Tensor.dim fun i => (result i).2.1,
        Tensor.dim fun i => (result i).2.2)

/-- The inputs of `valuesBackward` as one family over a batch prefix. -/
private abbrev Attention.valuesShapes (rows keys width : Nat) : Fin 4 → Shape :=
  ![[rows, keys], [keys, width], [rows, keys], [rows, width]]

/-- A component of `valuesBackwardFast` at a coordinate: that component of the
kernel on the matrices at its batch coordinate. -/
private theorem Attention.valuesBackwardFast_component_apply {α : Type} [Storage α] [Context α]
    {batch : Shape} {rows keys width : Nat} {t : Shape}
    (component : ∀ {batch : Shape},
      Tensor α (batch ++ [rows, keys]) × Tensor α (batch ++ [keys, width]) ×
        Tensor α (batch ++ [rows, keys]) → Tensor α (batch ++ t))
    (outer : ∀ {n : Nat} {batch : Shape}
      (inputs : (i : Fin 4) →
        Tensor α ((Shape.dim n batch) ++ (valuesShapes rows keys width i))),
      component (valuesBackwardFast (inputs 0) (inputs 1) (inputs 2) (inputs 3)) =
        Tensor.dim fun j => component (valuesBackwardFast ((inputs 0).unstack j)
          ((inputs 1).unstack j) ((inputs 2).unstack j) ((inputs 3).unstack j)))
    (weights : Tensor α (batch ++ [rows, keys]))
    (value : Tensor α (batch ++ [keys, width]))
    (multiplier : Tensor α (batch ++ [rows, keys]))
    (gradient : Tensor α (batch ++ [rows, width])) (c : (batch ++ t).Coord) :
    component (valuesBackwardFast weights value multiplier gradient) c =
      let split := Shape.Coord.appendEquiv batch t
        c
      component (batch := .scalar) (valuesBackwardSlice
        (weights.batchSlice split.1) (value.batchSlice split.1) (multiplier.batchSlice split.1)
        (gradient.batchSlice split.1)) split.2 :=
  Tensor.batchwise_apply (shapes := valuesShapes rows keys width)
    (fun inputs => component
      (valuesBackwardFast (inputs 0) (inputs 1) (inputs 2) (inputs 3)))
    outer (fun | 0 => weights | 1 => value | 2 => multiplier | 3 => gradient) c

/-- `Attention.valuesBackwardFast` computes `Attention.valuesBackward`. -/
@[csimp] theorem Attention.valuesBackwardFast_eq :
    @Attention.valuesBackward = @Attention.valuesBackwardFast := by
  funext α _ _ batch rows keys width weights value multiplier gradient
  refine Prod.ext (Tensor.Rep.ext fun c => ?_)
    (Prod.ext (Tensor.Rep.ext fun c => ?_) (Tensor.Rep.ext fun c => ?_))
  · rw [Attention.valuesBackward_fst_apply]
    exact (valuesBackwardFast_component_apply (fun result => result.1) (fun _ => rfl)
      weights value multiplier gradient c).symm
  · rw [Attention.valuesBackward_snd_fst_apply]
    exact (valuesBackwardFast_component_apply (fun result => result.2.1) (fun _ => rfl)
      weights value multiplier gradient c).symm
  · rw [Attention.valuesBackward_snd_snd_apply]
    exact (valuesBackwardFast_component_apply (fun result => result.2.2) (fun _ => rfl)
      weights value multiplier gradient c).symm

/-- The `Attention.valuesJVP` kernel on one matrix of each operand, from its saved row
weights. Entries that are zero or whose factors or ordinary sum lose range are replayed over finite
coefficients. -/
def Attention.valuesJVPSlice {α : Type} [Storage α] [Context α] {rows keys width : Nat}
    (weights : Tensor α [rows, keys]) (value : Tensor α [keys, width])
    (multiplier deltaScores : Tensor α [rows, keys]) (deltaValue : Tensor α [keys, width])
    (deltaMultiplier : Tensor α [rows, keys]) : Tensor α [rows, width] :=
  let deltaWeights := backwardFromWeights weights deltaScores (center := true)
  let deltaMixed := (deltaWeights * multiplier) + (weights * deltaMultiplier)
  let ordinary := Attention.scaledProductsSlice
    (fun group : Fin 2 => if group.val == 0 then deltaMixed else weights * multiplier)
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
        let pivot : Fin (keys + 1) := Softmax.pivot probabilities
        let incoming := fun key => ops.encode (get2 deltaScores row key)
        let normalized := Softmax.pullbackCoefficients?
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

/-- Product-rule tangent of the normalized matrix product over any leading batch axes, using its
saved row weights: each matrix is that of `Attention.valuesJVPSlice` at its batch
coordinate. -/
@[irreducible] def Attention.valuesJVP {α : Type} [Storage α] [Context α]
    {batch : Shape} {rows keys width : Nat}
    (weights : Tensor α (batch ++ [rows, keys]))
    (value : Tensor α (batch ++ [keys, width]))
    (multiplier deltaScores : Tensor α (batch ++ [rows, keys]))
    (deltaValue : Tensor α (batch ++ [keys, width]))
    (deltaMultiplier : Tensor α (batch ++ [rows, keys])) :
    Tensor α (batch ++ [rows, width]) :=
  Tensor.Rep.ofFn fun c =>
    let split := Shape.Coord.appendEquiv batch [rows, width]
      c
    Attention.valuesJVPSlice (weights.batchSlice split.1) (value.batchSlice split.1)
      (multiplier.batchSlice split.1) (deltaScores.batchSlice split.1)
      (deltaValue.batchSlice split.1) (deltaMultiplier.batchSlice split.1) split.2

/-- An entry of `Attention.valuesJVP` is the entry of `Attention.valuesJVPSlice` on
the matrices at its batch coordinate. -/
theorem Attention.valuesJVP_apply {α : Type} [Storage α] [Context α] {batch : Shape}
    {rows keys width : Nat} (weights : Tensor α (batch ++ [rows, keys]))
    (value : Tensor α (batch ++ [keys, width]))
    (multiplier deltaScores : Tensor α (batch ++ [rows, keys]))
    (deltaValue : Tensor α (batch ++ [keys, width]))
    (deltaMultiplier : Tensor α (batch ++ [rows, keys]))
    (c : (batch ++ [rows, width]).Coord) :
    Attention.valuesJVP weights value multiplier deltaScores deltaValue deltaMultiplier c =
      let split := Shape.Coord.appendEquiv batch [rows, width]
        c
      Attention.valuesJVPSlice (weights.batchSlice split.1) (value.batchSlice split.1)
        (multiplier.batchSlice split.1) (deltaScores.batchSlice split.1)
        (deltaValue.batchSlice split.1) (deltaMultiplier.batchSlice split.1) split.2 := by
  unfold Attention.valuesJVP
  exact Tensor.Rep.get_ofFn _ c

/-- `Attention.valuesJVP` computed one outer batch index at a time. -/
def Attention.valuesJVPFast {α : Type} [Storage α] [Context α]
    {batch : Shape} {rows keys width : Nat}
    (weights : Tensor α (batch ++ [rows, keys]))
    (value : Tensor α (batch ++ [keys, width]))
    (multiplier deltaScores : Tensor α (batch ++ [rows, keys]))
    (deltaValue : Tensor α (batch ++ [keys, width]))
    (deltaMultiplier : Tensor α (batch ++ [rows, keys])) :
    Tensor α (batch ++ [rows, width]) :=
  match batch, weights, value, multiplier, deltaScores, deltaValue, deltaMultiplier with
  | .scalar, weights, value, multiplier, deltaScores, deltaValue, deltaMultiplier =>
      Attention.valuesJVPSlice weights value multiplier deltaScores deltaValue
        deltaMultiplier
  | .dim _ _, weights, value, multiplier, deltaScores, deltaValue, deltaMultiplier =>
      Tensor.dim fun i => valuesJVPFast (weights.unstack i) (value.unstack i)
        (multiplier.unstack i) (deltaScores.unstack i) (deltaValue.unstack i)
        (deltaMultiplier.unstack i)

/-- The inputs of `valuesJVP` as one family over a batch prefix. -/
private abbrev Attention.valuesJVPShapes (rows keys width : Nat) : Fin 6 → Shape :=
  ![[rows, keys], [keys, width], [rows, keys], [rows, keys], [keys, width], [rows, keys]]

/-- `Attention.valuesJVPFast` computes `Attention.valuesJVP`. -/
@[csimp] theorem Attention.valuesJVPFast_eq :
    @Attention.valuesJVP = @Attention.valuesJVPFast := by
  funext α _ _ batch rows keys width weights value multiplier deltaScores deltaValue
    deltaMultiplier
  refine Tensor.Rep.ext fun c => ?_
  rw [Attention.valuesJVP_apply]
  exact Tensor.batchwise_apply (shapes := valuesJVPShapes rows keys width)
    (fun inputs => Attention.valuesJVPFast (inputs 0) (inputs 1) (inputs 2) (inputs 3)
      (inputs 4) (inputs 5))
    (fun _ => rfl)
    (fun
      | 0 => weights | 1 => value | 2 => multiplier | 3 => deltaScores | 4 => deltaValue
      | 5 => deltaMultiplier) c |>.symm

/-- Scaled dot-product attention (forward) of `Q : (nQ × d)`, `K : (nK × d)` and `V : (nK × d)`:

1. scores `S = Q Kᵀ` with shape `(nQ × nK)`
2. scaled scores `S' = S * (1 / Attention.scaleDenom d)`, which is `S / √d` over the reals when
   `d > 0`
3. (optional) mask: if `mask[i,j] = false`, key `j` is blocked for query `i` and its softmax
   numerator is exactly zero (the finite-scalar encoding of true `-∞` masking)
4. attention weights `A` by row normalization over the last axis
5. output `Out = A V` with shape `(nQ × d)`

Over real scalars every unmasked row of `A` sums to `1`, and so does a masked row with at least one
allowed key. Other numeric contexts need not meet these equalities. A fully blocked row gets
all-zero weights from its own branch, with no `0/0`.

The scores keep the ordinary order of matrix product, then scaling. Finite-coefficient backends
replay the complete scaled contraction of an entry after range loss or an absorbed nonzero term, so
an overflow of the unscaled score cannot spoil an otherwise representable scaled score, and tiny
queries are not scaled to zero beforehand. Scalar conversion and native execution are separate
trust obligations.

PyTorch analogy: `torch.softmax(scores.masked_fill(~mask, -torch.inf), dim=-1)` row-wise, then a
final matrix multiply by `V`.
-/
def Attention.scaledDotProduct
  {nQ nK dModel : Nat} {h1 : nQ ≠ 0} {h2 : nK ≠ 0}
  (ctx : Inputs α nQ nK dModel h1 h2) :
  Tensor α [nQ, dModel] :=
  let scale := scaleDenom (α := α) dModel
  let scaledScores := scaledProductsSlice (fun _ : Fin 1 => ctx.query)
    (fun _ => Tensor.permute ctx.key (Shape.adjacentSwap [nK, dModel] 0) [dModel, nK]) (1 / scale)
  let attentionWeights :=
    match ctx.mask with
    | none => Tensor.softmax scaledScores ⟨1, Shape.AxisInBounds.proof⟩
    | some m => hardMasked scaledScores m
  Tensor.matmul attentionWeights ctx.value

/-- Backward/VJP for scaled dot-product attention: `(dQ, dK, dV)` for an upstream gradient `dOut`.

We recompute the forward intermediates here, so the spec stands alone and needs no global tape.

Over real scalars, masked calls use the VJP for hard masking: blocked logits have zero forward
weight and therefore zero gradient. The finite-coefficient replay carries the exact product of
`dOut` and `V` through the centered pullback, because that product can overflow even when the
score cotangent is finite. Original operands that are not finite go through the ordinary backend
arithmetic.
-/
def Attention.scaledDotProductBackward
  {nQ nK dModel : Nat} {h1 : nQ ≠ 0} {h2 : nK ≠ 0}
  (ctx : Inputs α nQ nK dModel h1 h2)
  (dOut : Tensor α [nQ, dModel]) :
  (Tensor α [nQ, dModel] ×
   Tensor α [nK, dModel] ×
   Tensor α [nK, dModel]) :=
  let scale := scaleDenom (α := α) dModel
  let scaledScores := scaledProductsSlice (fun _ : Fin 1 => ctx.query)
    (fun _ => Tensor.permute ctx.key (Shape.adjacentSwap [nK, dModel] 0) [dModel, nK]) (1 / scale)
  let attentionWeights :=
    match ctx.mask with
    | none => Tensor.softmax scaledScores ⟨1, Shape.AxisInBounds.proof⟩
    | some m => hardMasked scaledScores m

  -- Backprop through `Out = A V`.
  let dAttentionWeights := Tensor.matmul dOut (Tensor.permute ctx.value
      (Shape.adjacentSwap [nK, dModel] 0) [dModel, nK])
  let dV := Tensor.matmul (Tensor.permute attentionWeights
      (Shape.adjacentSwap [nQ, nK] 0) [nK, nQ]) dOut

  -- Centering before the weighted sum keeps the gradient of a logit whose weight rounds to one.
  let dScaledScores :=
    pullbackProducts attentionWeights dAttentionWeights (fun _ : Fin 1 => dOut)
      (fun _ => Tensor.permute ctx.value (Shape.adjacentSwap [nK, dModel] 0) [dModel, nK]) 1

  -- Keep the score scale inside the replay boundary for both cotangent contractions.
  let dQ := scaledProductsSlice (fun _ : Fin 1 => dScaledScores)
    (fun _ => ctx.key) (1 / scale) (scaleFirst := true)
  let dK := scaledProductsSlice (fun _ : Fin 1 => Tensor.permute dScaledScores
      (Shape.adjacentSwap [nQ, nK] 0) [nK, nQ])
    (fun _ => ctx.query) (1 / scale) (scaleFirst := true)

  (dQ, dK, dV)

/--
Forward-mode JVP for scaled dot-product attention: the derivative of the pure attention equation

`Out = softmax(mask(Q Kᵀ / sqrt(d))) V`

in the direction `(dQ,dK,dV)`. Over real scalars a hard-masked blocked logit has zero forward
weight, so its tangent contribution in `backwardFromWeights` is zero, and upstream
centering keeps that zero-row convention. The row-wise softmax Jacobian is symmetric, so the same
centered formula serves as both VJP and JVP once the forward weights are known.

When a tangent contraction overflows before normalization, the finite-coefficient replay carries
both original product terms through the pullback. The replay also gives the zero tangent of a
softmax over one key without first rounding an unrepresentable intermediate tangent to infinity.
-/
def Attention.scaledDotProductJVP
  {nQ nK dModel : Nat} {h1 : nQ ≠ 0} {h2 : nK ≠ 0}
  (ctx : Inputs α nQ nK dModel h1 h2)
  (dQ : Tensor α [nQ, dModel])
  (dK dV : Tensor α [nK, dModel]) :
  Tensor α [nQ, dModel] :=
  let scale := scaleDenom (α := α) dModel
  let scaledScores := scaledProductsSlice (fun _ : Fin 1 => ctx.query)
    (fun _ => Tensor.permute ctx.key (Shape.adjacentSwap [nK, dModel] 0) [dModel, nK]) (1 / scale)
  -- Enclose both terms together: either unscaled tangent product may overflow on its own.
  let dScaledScores := scaledProductsSlice
    (fun i : Fin 2 => if i.val == 0 then dQ else ctx.query)
    (fun i => if i.val == 0 then Tensor.permute ctx.key
        (Shape.adjacentSwap [nK, dModel] 0) [dModel, nK] else Tensor.permute dK
            (Shape.adjacentSwap [nK, dModel] 0) [dModel, nK])
    (1 / scale)
  let attentionWeights :=
    match ctx.mask with
    | none => Tensor.softmax scaledScores ⟨1, Shape.AxisInBounds.proof⟩
    | some m => hardMasked scaledScores m
  let dAttentionWeights :=
    pullbackProducts attentionWeights dScaledScores
      (fun i : Fin 2 => if i.val == 0 then dQ else ctx.query)
      (fun i => if i.val == 0 then Tensor.permute ctx.key
          (Shape.adjacentSwap [nK, dModel] 0) [dModel, nK] else Tensor.permute dK
              (Shape.adjacentSwap [nK, dModel] 0) [dModel, nK])
      (1 / scale)
  Tensor.matmul dAttentionWeights ctx.value + Tensor.matmul attentionWeights dV


/-
  Multi-Head Attention
  Splits input into multiple heads, applies attention, then combines
-/
  /-- Multi-head attention parameters (projection matrices).

PyTorch analogy: the four linear maps of an attention block.

- `Wq`, `Wk`, `Wv` project `dModel -> (numHeads * headDim)`
- `Wo` projects `(numHeads * headDim) -> dModel`

The projections are plain matrices without bias terms, so statements about the forward and
derivative maps show the whole parameterization.
-/
  structure MultiHeadAttention (α : Type) [Storage α]
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

We store heads as the outermost axis, so a per-head computation is a `Tensor.dim` over
`Fin numHeads`.

A feature coordinate is read as `(head, coordinate-within-head)`: the input is reshaped to
`(n, numHeads, headDim)` and its token and head axes are then swapped. A single reshape to
`(numHeads, n, headDim)` would read the coordinates in the wrong row-major order.
-/
  def MultiHeadAttention.splitHeads
    {α : Type} [Storage α]
    {n dModel : Nat}
  (x : Tensor α [n, dModel])
  (numHeads headDim : Nat)
  (h : dModel = numHeads * headDim)
  : Tensor α [numHeads, n, headDim] :=
  let s₁ : Shape := .dim n (.dim dModel .scalar)
  let s₂ : Shape := .dim n (.dim numHeads (.dim headDim .scalar))
  have size_eq : Shape.size s₁ = Shape.size s₂ := by
    change n * (dModel * 1) = n * (numHeads * (headDim * 1))
    rw [h]
    simp
  Tensor.permute (Tensor.reshape x _ size_eq) (s₂.adjacentSwap 0) (s₂.swapAdjacentAtDepth 0)

/-
  Concatenate attention heads back into single tensor
-/
  /-- Combine a tensor of heads back into one `(n, numHeads * headDim)` tensor:
`Tensor.permute` by `Shape.adjacentSwap` exchanges the head and token axes, then `Tensor.reshape`
flattens the last two axes.
-/
  def MultiHeadAttention.combineHeads
    {α : Type} [Storage α]
    {n numHeads headDim : Nat}
    (heads : Tensor α [numHeads, n, headDim]) :
    Tensor α [n, (numHeads * headDim)] :=
  let swapped : Tensor α [n, numHeads, headDim] :=
    Tensor.permute heads (Shape.adjacentSwap [numHeads, n, headDim] 0) [n, numHeads, headDim]
  let s₁ : Shape := .dim n (.dim numHeads (.dim headDim .scalar))
  let s₂ : Shape := .dim n (.dim (numHeads * headDim) .scalar)
  have hSize : Shape.size s₁ = Shape.size s₂ := by
    simp [s₁, s₂]
  Tensor.reshape swapped _ hSize

  /-- Multi-head attention of the rows of `query` over the rows of `keyValue`, every head sharing
the optional `mask`:

1. project `query` into `Q` and `keyValue` into `K,V`
2. split the projection dimension into heads
3. run scaled dot-product attention per head, with the head width `headDim` as its `d`
4. combine heads back and project with `Wo`

Self-attention of `x` is `MultiHeadAttention.forward h h mha x x mask`.
-/
  def MultiHeadAttention.forward
    {α : Type} [Storage α] [Context α]
    [DecidableRel ((· > ·) : α → α → Prop)]
    {numHeads dModel headDim nQ nK : Nat}
  (hQ : nQ ≠ 0) (hK : nK ≠ 0)
  (mha : MultiHeadAttention α numHeads dModel headDim)
  (query : Tensor α [nQ, dModel]) (keyValue : Tensor α [nK, dModel])
  (mask : Option (Tensor Bool [nQ, nK])) :
  Tensor α [nQ, dModel] :=

  -- Project the inputs to the full-width Q, K and V of all heads.
  let Q := Tensor.matmul query mha.queryWeight
  let K := Tensor.matmul keyValue mha.keyWeight
  let V := Tensor.matmul keyValue mha.valueWeight

  -- Split heads: we represent heads as the outer axis, `(numHeads, nQ, headDim)` for `Q` and
  -- `(numHeads, nK, headDim)` for `K` and `V`.
  let QHeads := splitHeads Q numHeads headDim rfl
  let KHeads := splitHeads K numHeads headDim rfl
  let VHeads := splitHeads V numHeads headDim rfl

  -- Compute attention per head as a tensor indexed by `Fin numHeads`.
  let attentionHeads : Tensor α [numHeads, nQ, headDim] :=
    Tensor.dim (fun headIdx =>
      let ctx : Attention.Inputs α nQ nK headDim hQ hK :=
        { query := Tensor.unstack QHeads headIdx
          key := Tensor.unstack KHeads headIdx
          value := Tensor.unstack VHeads headIdx
          mask := mask }
      Attention.scaledDotProduct ctx)

  -- Combine heads back to `(nQ, numHeads * headDim)`.
  let concatenated := combineHeads (α := α) (n := nQ) (numHeads := numHeads)
    (headDim := headDim) attentionHeads

  -- Project the concatenated heads back to the model dimension.
  Tensor.matmul concatenated mha.outputWeight

/-- Everything `backward` sends backwards: the gradients of the four projection
matrices, as a `MultiHeadAttention`, and the gradient with respect to the attended sequence.

PyTorch analogue of `parameters`: the `.grad` of `nn.MultiheadAttention.in_proj_weight` split into
its three blocks, plus `out_proj.weight.grad`. -/
structure MultiHeadAttention.Gradients (n numHeads dModel headDim : Nat) (α : Type)
    [Storage α] where
  /-- Gradients for the four projection matrices. -/
  parameters : MultiHeadAttention α numHeads dModel headDim
  /-- Gradient with respect to the input sequence `x`. -/
  input : Tensor α [n, dModel]

/-- Multi-head attention backward pass of `MultiHeadAttention.forward h1 h1 mha x x mask`.

Returns gradients for input `x` and all projection matrices `(Wq,Wk,Wv,Wo)`.
The forward intermediates are recomputed locally instead of relying on a global tape.
-/
def MultiHeadAttention.backward
  {α : Type} [Storage α] [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)]
  {n numHeads dModel headDim : Nat} (h1 : n ≠ 0)
  (mha : MultiHeadAttention α numHeads dModel headDim)
  (x : Tensor α [n, dModel])
  (mask : Option (Tensor Bool [n, n]))
  (gradOutput : Tensor α [n, dModel]) :
  Gradients n numHeads dModel headDim α :=

  -- Forward recomputation for intermediate values.
  let Q := Tensor.matmul x mha.queryWeight
  let K := Tensor.matmul x mha.keyWeight
  let V := Tensor.matmul x mha.valueWeight

  let QHeads := splitHeads Q numHeads headDim rfl
  let KHeads := splitHeads K numHeads headDim rfl
  let VHeads := splitHeads V numHeads headDim rfl

  let attentionHeads : Tensor α [numHeads, n, headDim] :=
    Tensor.dim (fun headIdx =>
      let ctx : Attention.Inputs α n n headDim h1 h1 :=
        { query := Tensor.unstack QHeads headIdx
          key := Tensor.unstack KHeads headIdx
          value := Tensor.unstack VHeads headIdx
          mask := mask }
      Attention.scaledDotProduct ctx)

  let concatenated := combineHeads (α := α) (n := n) (numHeads := numHeads)
    (headDim := headDim) attentionHeads

  -- Backpropagate through the output projection.
  let (gradConcat, gradOutputWeight) :=
    matmulBackward (Shape.CanBroadcastTo.refl .scalar)
      (Shape.CanBroadcastTo.refl .scalar) concatenated mha.outputWeight gradOutput

  -- Backpropagate through the reshape and axis swap used to combine heads.
  let gradAttentionHeads := splitHeads gradConcat numHeads headDim rfl

  -- Backpropagate independently through each attention head.
  let (gradQHeads, gradKHeads, gradVHeads) :
      Tensor α [numHeads, n, headDim] ×
        Tensor α [numHeads, n, headDim] ×
        Tensor α [numHeads, n, headDim] :=
    let backwardHead (headIdx : Fin numHeads) :=
      let ctx : Attention.Inputs α n n headDim h1 h1 :=
        { query := Tensor.unstack QHeads headIdx
          key := Tensor.unstack KHeads headIdx
          value := Tensor.unstack VHeads headIdx
          mask := mask }
      Attention.scaledDotProductBackward ctx
        (Tensor.unstack gradAttentionHeads headIdx)
    let gQ : Fin numHeads → Tensor α [n, headDim] :=
      fun headIdx => (backwardHead headIdx).1
    let gK : Fin numHeads → Tensor α [n, headDim] :=
      fun headIdx => (backwardHead headIdx).2.1
    let gV : Fin numHeads → Tensor α [n, headDim] :=
      fun headIdx => (backwardHead headIdx).2.2
    (Tensor.dim gQ, Tensor.dim gK, Tensor.dim gV)

  -- Undo the head split for Q, K, and V.
  let gradQ := combineHeads (α := α) (n := n) (numHeads := numHeads)
    (headDim := headDim) gradQHeads
  let gradK := combineHeads (α := α) (n := n) (numHeads := numHeads)
    (headDim := headDim) gradKHeads
  let gradV := combineHeads (α := α) (n := n) (numHeads := numHeads)
    (headDim := headDim) gradVHeads

  -- Backpropagate through the input projections.
  let (gradXQuery, gradQueryWeight) :=
    matmulBackward (Shape.CanBroadcastTo.refl .scalar)
      (Shape.CanBroadcastTo.refl .scalar) x mha.queryWeight gradQ
  let (gradXKey, gradKeyWeight) :=
    matmulBackward (Shape.CanBroadcastTo.refl .scalar)
      (Shape.CanBroadcastTo.refl .scalar) x mha.keyWeight gradK
  let (gradXValue, gradValueWeight) :=
    matmulBackward (Shape.CanBroadcastTo.refl .scalar)
      (Shape.CanBroadcastTo.refl .scalar) x mha.valueWeight gradV

  -- Add the contributions to the input gradient from the Q, K, and V branches.
  let gradX := (gradXQuery + gradXKey) + gradXValue

  { parameters :=
      { queryWeight := gradQueryWeight
        keyWeight := gradKeyWeight
        valueWeight := gradValueWeight
        outputWeight := gradOutputWeight }
    input := gradX }

omit [DecidableRel ((· > ·) : α → α → Prop)] in
/-- The fieldwise sum of two attention parameter records, which is how the gradients of two
sequences accumulate. -/
def MultiHeadAttention.add {numHeads dModel headDim : Nat}
    (first second : MultiHeadAttention α numHeads dModel headDim) :
    MultiHeadAttention α numHeads dModel headDim :=
  { queryWeight := first.queryWeight + second.queryWeight
    keyWeight := first.keyWeight + second.keyWeight
    valueWeight := first.valueWeight + second.valueWeight
    outputWeight := first.outputWeight + second.outputWeight }

omit [DecidableRel ((· > ·) : α → α → Prop)] in
/-- The sum of a list of attention parameter records, such as the parameter gradients of a list of
sequences, added by a left fold from the first. For an empty list it is the zero record shaped like
the projections of `mha`. -/
def MultiHeadAttention.sum {numHeads dModel headDim : Nat}
    (mha : MultiHeadAttention α numHeads dModel headDim) :
    List (MultiHeadAttention α numHeads dModel headDim) →
      MultiHeadAttention α numHeads dModel headDim
  | [] =>
      { queryWeight := Tensor.map (fun _ => (0 : α)) mha.queryWeight
        keyWeight := Tensor.map (fun _ => (0 : α)) mha.keyWeight
        valueWeight := Tensor.map (fun _ => (0 : α)) mha.valueWeight
        outputWeight := Tensor.map (fun _ => (0 : α)) mha.outputWeight }
  | first :: rest => rest.foldl add first

/-- `backward` of every `(n, dModel)` sequence along the leading axes `leading`:
each sequence's input gradient at its index, and the parameter gradients of the sequences in
row-major order of the leading axes. -/
def MultiHeadAttention.backwardLeading
    {n numHeads dModel headDim : Nat} (h1 : n ≠ 0)
    (mha : MultiHeadAttention α numHeads dModel headDim) (mask : Option (Tensor Bool [n, n])) :
    (leading : Shape) → Tensor α (leading ++ [n, dModel]) →
      Tensor α (leading ++ [n, dModel]) →
      Tensor α (leading ++ [n, dModel]) ×
        List (MultiHeadAttention α numHeads dModel headDim)
  | .scalar, x, gradOutput =>
      let gradients := backward h1 mha x mask gradOutput
      (gradients.input, [gradients.parameters])
  | .dim m rest, x, gradOutput =>
      let parts := Array.ofFn fun i : Fin m =>
        backwardLeading h1 mha mask rest (Tensor.unstack x i)
          (Tensor.unstack gradOutput i)
      (Tensor.dim fun i => (parts[i.val]'(by simp [parts])).1, parts.toList.flatMap (·.2))

/-- One sequence: its input gradient and its parameter gradients. -/
theorem MultiHeadAttention.backwardLeading_scalar
    {n numHeads dModel headDim : Nat} (h1 : n ≠ 0)
    (mha : MultiHeadAttention α numHeads dModel headDim) (mask : Option (Tensor Bool [n, n]))
    (x gradOutput : Tensor α [n, dModel]) :
    backwardLeading h1 mha mask .scalar x gradOutput =
      ((backward h1 mha x mask gradOutput).input,
        [(backward h1 mha x mask gradOutput).parameters]) := rfl

/-- Along a leading axis of `m` entries: the input gradient at index `i` is that of the `i`-th
slice, and the parameter gradients are those of the slices in index order. -/
theorem MultiHeadAttention.backwardLeading_dim
    {n numHeads dModel headDim m : Nat} {rest : Shape} (h1 : n ≠ 0)
    (mha : MultiHeadAttention α numHeads dModel headDim) (mask : Option (Tensor Bool [n, n]))
    (x gradOutput : Tensor α ((Shape.dim m rest) ++ [n, dModel])) :
    backwardLeading h1 mha mask (.dim m rest) x gradOutput =
      (Tensor.dim fun i => (backwardLeading h1 mha mask rest
          (Tensor.unstack x i) (Tensor.unstack gradOutput i)).1,
        (List.ofFn fun i => backwardLeading h1 mha mask rest
          (Tensor.unstack x i) (Tensor.unstack gradOutput i)).flatMap (·.2)) := by
  simp only [backwardLeading, Array.getElem_ofFn, Array.toList_ofFn]

/-- Forward-mode JVP for multi-head attention, along the input tangent `dx` and the parameter
tangent `dmha`.

The rule follows the computational graph of `MultiHeadAttention.forward h1 h1 mha x x mask`:

1. project tangents through `Q/K/V` by the product rule, as `dx Wq + x dWq` for the queries,
2. split primal and tangent projections into heads,
3. apply `Attention.scaledDotProductJVP` head-wise,
4. combine head tangents, then differentiate the final output projection.
-/
def MultiHeadAttention.jvp
  {α : Type} [Storage α] [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)]
  {n numHeads dModel headDim : Nat} (h1 : n ≠ 0)
  (mha dmha : MultiHeadAttention α numHeads dModel headDim)
  (x dx : Tensor α [n, dModel])
  (mask : Option (Tensor Bool [n, n])) :
  Tensor α [n, dModel] :=

  let Q := Tensor.matmul x mha.queryWeight
  let K := Tensor.matmul x mha.keyWeight
  let V := Tensor.matmul x mha.valueWeight

  let dQ := Tensor.matmul dx mha.queryWeight + Tensor.matmul x dmha.queryWeight
  let dK := Tensor.matmul dx mha.keyWeight + Tensor.matmul x dmha.keyWeight
  let dV := Tensor.matmul dx mha.valueWeight + Tensor.matmul x dmha.valueWeight

  let QHeads := splitHeads Q numHeads headDim rfl
  let KHeads := splitHeads K numHeads headDim rfl
  let VHeads := splitHeads V numHeads headDim rfl
  let dQHeads := splitHeads dQ numHeads headDim rfl
  let dKHeads := splitHeads dK numHeads headDim rfl
  let dVHeads := splitHeads dV numHeads headDim rfl

  let attentionHeads : Tensor α [numHeads, n, headDim] :=
    Tensor.dim (fun headIdx =>
      let ctx : Attention.Inputs α n n headDim h1 h1 :=
        { query := Tensor.unstack QHeads headIdx
          key := Tensor.unstack KHeads headIdx
          value := Tensor.unstack VHeads headIdx
          mask := mask }
      Attention.scaledDotProduct ctx)

  let dAttentionHeads : Tensor α [numHeads, n, headDim] :=
    Tensor.dim (fun headIdx =>
      let ctx : Attention.Inputs α n n headDim h1 h1 :=
        { query := Tensor.unstack QHeads headIdx
          key := Tensor.unstack KHeads headIdx
          value := Tensor.unstack VHeads headIdx
          mask := mask }
      Attention.scaledDotProductJVP ctx
        (Tensor.unstack dQHeads headIdx)
        (Tensor.unstack dKHeads headIdx)
        (Tensor.unstack dVHeads headIdx))

  let concatenated := combineHeads (α := α) (n := n) (numHeads := numHeads)
    (headDim := headDim) attentionHeads
  let dConcatenated := combineHeads (α := α) (n := n) (numHeads := numHeads)
    (headDim := headDim) dAttentionHeads

  Tensor.matmul dConcatenated mha.outputWeight +
    Tensor.matmul concatenated dmha.outputWeight

/-- Self-attention on a single sequence with one head and no mask.

`x` is projected to Q, K and V by `Wq`, `Wk` and `Wv`, attended by `Attention.scaledDotProduct` at
width `projDim`, and projected back to `dModel` by `Wo`.

PyTorch analogue: the core of `nn.MultiheadAttention` / `TransformerEncoderLayer` (ignoring the
batch axis).
-/
def Tensor.selfAttention
  {α : Type} [Storage α] [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)]
  {n dModel projDim : Nat}
  (x : Tensor α [n, dModel])
  (Wq : Tensor α [dModel, projDim])
  (Wk : Tensor α [dModel, projDim])
  (Wv : Tensor α [dModel, projDim])
  (Wo : Tensor α [projDim, dModel])
  (h1 : n ≠ 0) :
  Tensor α [n, dModel] := by
  let Q := Tensor.matmul x Wq
  let K := Tensor.matmul x Wk
  let V := Tensor.matmul x Wv
  let ctx : Attention.Inputs α n n projDim h1 h1 :=
    { query := Q, key := K, value := V, mask := none }
  exact Tensor.matmul (Attention.scaledDotProduct ctx) Wo

/-- FlashAttention's schedule on score matrices with any leading batch axes: every output entry is
`Softmax.onlineAttend` over the blocks of the allowed keys of its score row, in index
order, against its value column times the `multiplier` row when one is given. -/
def Attention.online {α : Type} [Storage α] [Context α] {batch : Shape}
    {rows keys width : Nat} (blocks : Softmax.BlockPartition)
    (scores : Tensor α (batch ++ [rows, keys])) (value : Tensor α (batch ++ [keys, width]))
    (multiplier : Option (Tensor α (batch ++ [rows, keys])))
    (mask : Option (Tensor Bool [rows, keys])) : Tensor α (batch ++ [rows, width]) :=
  Tensor.Rep.ofFn fun c =>
    let split := Shape.Coord.appendEquiv batch [rows, width]
      c
    Softmax.onlineAttend blocks
      (fun j => scores.batchSlice split.1 (split.2.1, j, ()))
      (fun j => multiplier.elim (value.batchSlice split.1 (j, split.2.2.1, ())) fun multiplier =>
        multiplier.batchSlice split.1 (split.2.1, j, ()) *
          value.batchSlice split.1 (j, split.2.2.1, ()))
      (fun j => mask.elim true fun mask => mask (split.2.1, j, ()))

/-- An entry of `Attention.online` is `Softmax.onlineAttend` of the score row and
the value column through it. -/
theorem Attention.online_apply {α : Type} [Storage α] [Context α] {batch : Shape}
    {rows keys width : Nat} (blocks : Softmax.BlockPartition)
    (scores : Tensor α (batch ++ [rows, keys])) (value : Tensor α (batch ++ [keys, width]))
    (multiplier : Option (Tensor α (batch ++ [rows, keys])))
    (mask : Option (Tensor Bool [rows, keys])) (c : (batch ++ [rows, width]).Coord) :
    Attention.online blocks scores value multiplier mask c =
      let split := Shape.Coord.appendEquiv batch [rows, width]
        c
      Softmax.onlineAttend blocks
        (fun j => scores.batchSlice split.1 (split.2.1, j, ()))
        (fun j => multiplier.elim (value.batchSlice split.1 (j, split.2.2.1, ())) fun multiplier =>
          multiplier.batchSlice split.1 (split.2.1, j, ()) *
            value.batchSlice split.1 (j, split.2.2.1, ()))
        (fun j => mask.elim true fun mask => mask (split.2.1, j, ())) :=
  Tensor.Rep.get_ofFn _ c

end TorchLean

/-! ## Attention over any leading batch axes -/

namespace TorchLean.Tensor

variable {α : Type} [Storage α] [Context α]

/-- Scaled dot-product attention over any leading batch axes, PyTorch's
`scaled_dot_product_attention`: the softmax over the keys of the scores `scale * query keyᵀ`, with
the entries where `mask` is false blocked, times `weights` when it is given, times `value`. The
scores are `Attention.scaledMatmul`. From there `twoPass` applies
`Attention.weights` and `Attention.values`, and `online blocks` applies
`Attention.online`. -/
def attention {batch : Shape} {rows keys depth width : Nat}
    (query : Tensor α (batch ++ [rows, depth])) (key : Tensor α (batch ++ [keys, depth]))
    (value : Tensor α (batch ++ [keys, width])) (scale : α)
    (mask : Option (Tensor Bool [rows, keys]) := none)
    (weights : Option (Tensor α (batch ++ [rows, keys])) := none)
    (schedule : Softmax.Schedule := .twoPass) : Tensor α (batch ++ [rows, width]) :=
  let scores := Attention.scaledMatmul (Shape.CanBroadcastTo.refl batch)
    (Shape.CanBroadcastTo.refl batch) query
    (Tensor.permute key (Shape.adjacentSwap (batch ++ [keys, depth]) batch.rank)
      (batch ++ [depth, keys])) scale
  match schedule with
  | .twoPass =>
    Attention.values (Attention.weights scores mask) value weights
  | .online blocks => Attention.online blocks scores value weights mask

/-- The cotangents of the query, key, value and multiplier of `attention` for both schedules (the
last zero when `weights` is absent): `Attention.valuesBackward` on the two-pass
weights, multiplier one without `weights`, then `Attention.scaledMatmulBackward` and the key
transpose. -/
def attentionBackward {batch : Shape} {rows keys depth width : Nat}
    (query : Tensor α (batch ++ [rows, depth])) (key : Tensor α (batch ++ [keys, depth]))
    (value : Tensor α (batch ++ [keys, width])) (scale : α)
    (mask : Option (Tensor Bool [rows, keys]))
    (weights : Option (Tensor α (batch ++ [rows, keys])))
    (gradient : Tensor α (batch ++ [rows, width])) :
    Tensor α (batch ++ [rows, depth]) × Tensor α (batch ++ [keys, depth]) ×
      Tensor α (batch ++ [keys, width]) × Tensor α (batch ++ [rows, keys]) :=
  let keyTranspose := Tensor.permute key
    (Shape.adjacentSwap (batch ++ [keys, depth]) batch.rank) (batch ++ [depth, keys])
  let scores := Attention.scaledMatmul (Shape.CanBroadcastTo.refl batch)
    (Shape.CanBroadcastTo.refl batch) query keyTranspose scale
  let values := Attention.valuesBackward
    (Attention.weights scores mask) value (weights.getD (Tensor.full _ 1)) gradient
  let products := Attention.scaledMatmulBackward (Shape.CanBroadcastTo.refl batch)
    (Shape.CanBroadcastTo.refl batch) query keyTranspose values.1 scale
  (products.1, Tensor.permute products.2
      (Shape.adjacentSwap (batch ++ [depth, keys]) batch.rank) (batch ++ [keys, depth]),
    values.2.1, weights.elim (Tensor.zeros _) fun _ => values.2.2)

/-- The tangent of `attention` for both schedules along the query, key, value and multiplier
tangents (`dWeights` read only when `weights` is given): `Attention.valuesJVP` on the
two-pass weights at the score tangent of `Attention.scaledMatmulJVP`. -/
def attentionJVP {batch : Shape} {rows keys depth width : Nat}
    (query : Tensor α (batch ++ [rows, depth])) (key : Tensor α (batch ++ [keys, depth]))
    (value : Tensor α (batch ++ [keys, width])) (scale : α)
    (mask : Option (Tensor Bool [rows, keys]))
    (weights : Option (Tensor α (batch ++ [rows, keys])))
    (dQuery : Tensor α (batch ++ [rows, depth])) (dKey : Tensor α (batch ++ [keys, depth]))
    (dValue : Tensor α (batch ++ [keys, width]))
    (dWeights : Option (Tensor α (batch ++ [rows, keys]))) :
    Tensor α (batch ++ [rows, width]) :=
  let transpose := fun (t : Tensor α (batch ++ [keys, depth])) => Tensor.permute t
    (Shape.adjacentSwap (batch ++ [keys, depth]) batch.rank) (batch ++ [depth, keys])
  let scores := Attention.scaledMatmul (Shape.CanBroadcastTo.refl batch)
    (Shape.CanBroadcastTo.refl batch) query (transpose key) scale
  let dScores := Attention.scaledMatmulJVP (Shape.CanBroadcastTo.refl batch)
    (Shape.CanBroadcastTo.refl batch) query dQuery (transpose key) (transpose dKey) scale
  Attention.valuesJVP (Attention.weights scores mask) value
    (weights.getD (Tensor.full _ 1)) dScores dValue
    (weights.elim (Tensor.zeros _) fun _ => dWeights.getD (Tensor.zeros _))

end TorchLean.Tensor
