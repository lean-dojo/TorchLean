/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.TensorOps

/-!
# Positional encodings (spec layer)

This file provides the simplest positional encoding definition: a **learnable** per-position
embedding that is added to token embeddings.

PyTorch analogy:

- This is the same idea as having an `nn.Embedding(max_len, d_model)` (or a parameter tensor
  `[max_len, d_model]`) and doing `x + pos[:seqLen]`.

Why learnable positional encodings show up a lot in practice:

- they are easy to train and tend to work well for fixed-length settings (e.g. ViT with a chosen
  patch grid, or language models trained with a fixed max sequence length),
- they keep the spec algebraic: there is no trigonometry, complex numbers, or special casing for
  even/odd dimensions.

If you want sinusoidal encodings (Transformer) or RoPE/rotary encodings, those can be defined as
pure functions that produce a tensor of shape `(seqLen, embedDim)` and then reused with the same
`addPositionalEncodingSpec` below.

Reference (sinusoidal): "Attention Is All You Need" (Vaswani et al., 2017):
  https://arxiv.org/abs/1706.03762
-/

@[expose] public section


open TorchLean

namespace Spec

open TorchLean TorchLean.Tensor

variable {α : Type} [TorchLean.Storage α] [Context α]

/--
Learnable positional encoding parameters for a fixed `(seqLen, embedDim)`.

This record stores the trainable positional table. Higher-level models decide how to initialize it
and whether to share or resize it across different sequence lengths.
-/
structure PositionalEncodingSpec (seqLen embedDim : Nat) (α : Type) [TorchLean.Storage α] where
  /-- The learnable positional table, one `embedDim` vector per position. -/
  pos : Tensor α [seqLen, embedDim]

/--
Add positional encodings: `y = x + pos`.

Both `x` and `pos` have the same shape, so this is just elementwise addition (no broadcasting).
-/
def addPositionalEncodingSpec {seqLen embedDim : Nat}
    (pe : PositionalEncodingSpec seqLen embedDim α)
    (x : Tensor α [seqLen, embedDim]) :
    Tensor α [seqLen, embedDim] :=
  Tensor.addSpec x pe.pos

/-!
## Gradients

Learnable positional encoding is just an elementwise addition:

`y = x + pos`

So the adjoint is just:
- `δx   = δy`
- `δpos = δy`

This is trivial, but having it as a named spec makes higher-level models (e.g. ViT) easier to
wire up without re-deriving the same one-liner everywhere.
-/

/-- Backward/VJP for `addPositionalEncodingSpec`. -/
def addPositionalEncodingBackwardSpec {seqLen embedDim : Nat}
    (_pe : PositionalEncodingSpec seqLen embedDim α)
    (gradOutput : Tensor α [seqLen, embedDim]) :
    (Tensor α [seqLen, embedDim] ×  -- ∂L/∂pos
     Tensor α [seqLen, embedDim]) := -- ∂L/∂x
  (gradOutput, gradOutput)

/-!
## Sinusoidal positional encodings (pure functions)

These are the classic Transformer sinusoidal encodings from:

Vaswani et al. (2017), "Attention Is All You Need".

We implement them as **pure tensor generators** so they can be reused in multiple model specs
without adding new trainable parameters.
-/

/--
Frequency denominator used by both sinusoidal PE and RoPE:

`denom(i; d) = 10000^(2*i / d)`

implemented as:

`exp(log(10000) * (2*i / d))`

Each operation uses the selected scalar backend. On a grid that cannot represent `10000`, the
logarithm receives the rounded base; this is not a single rounding of the exact real logarithm.
-/
def posencFreqDenomSpec (i d : Nat) : α :=
  MathFunctions.exp ((MathFunctions.log 10000) * ((2 * (i : α)) / (d : α)))

/--
Common angle used by sinusoidal PE and RoPE:

`θ(pos, i; d) = pos / 10000^(2*i/d)`.
-/
def posencAngleSpec (pos i d : Nat) : α :=
  (pos : α) / posencFreqDenomSpec (α := α) i d

/--
Pure sinusoidal positional encoding tensor with shape `(seqLen, embedDim)`.

Definition (Transformer):
- `PE[pos, 2i]   = sin(θ(pos, i; embedDim))`
- `PE[pos, 2i+1] = cos(θ(pos, i; embedDim))`

`startPos` is an offset for the absolute positions; use it when generating a chunk of positions
for cached decoding (e.g. tokens `startPos .. startPos+seqLen-1`).

This definition is total for all `seqLen`/`embedDim`:
- if `embedDim = 0`, the inner dimension is empty, so no scalar computations are observed.
- if `embedDim` is odd, the last column uses the same `i = floor(j/2)` convention as usual.
-/
def sinusoidalPositionalEncodingSpec (seqLen embedDim : Nat) (startPos : Nat := 0) :
    Tensor α [seqLen, embedDim] :=
  Tensor.dim (fun (pos : Fin seqLen) =>
    Tensor.dim (fun (j : Fin embedDim) =>
      let posNat : Nat := startPos + pos.val
      let iNat : Nat := j.val / 2
      let θ : α := posencAngleSpec (α := α) posNat iNat embedDim
      let v : α := if j.val % 2 = 0 then MathFunctions.sin θ else MathFunctions.cos θ
      Tensor.scalar v))

/--
Add sinusoidal positional encodings: `y = x + sinusoidal(startPos, seqLen, embedDim)`.
-/
def addSinusoidalPositionalEncodingSpec {seqLen embedDim : Nat}
    (x : Tensor α [seqLen, embedDim])
    (startPos : Nat := 0) :
    Tensor α [seqLen, embedDim] :=
  Tensor.addSpec x (sinusoidalPositionalEncodingSpec (α := α) seqLen embedDim startPos)

/-!
## Rotary positional embeddings (RoPE) utilities (pure functions)

RoPE (Su et al., 2021) encodes position by applying a 2D rotation to each pair of features
in the last dimension.

In most transformer implementations, RoPE is applied to query/key head vectors:
- per head: `(seqLen, headDim)`
- all heads: `(numHeads, seqLen, headDim)`

This file provides **pure** RoPE helpers. Attention modules can apply these helpers before the
query/key dot product.

References:
- Su et al. (2021), "RoFormer: Enhanced Transformer with Rotary Position Embedding".
- Many implementations use the same `θ(pos,i;d)` frequency schedule as sinusoidal PE.
-/

/--
Rotate pairs on the last dimension:

`(x0, x1, x2, x3, ...) ↦ (-x1, x0, -x3, x2, ...)`.

This corresponds to multiplying each 2-vector `(x_even, x_odd)` by the matrix:
`[[0, -1], [1, 0]]`.

Design note:
- Standard RoPE assumes `headDim` is even.
- This spec function is total: if `headDim` is odd, the last (unpaired) entry is left unchanged.
-/
def ropeRotatePairsSpec {headDim : Nat}
    (x : Tensor α [headDim]) :
    Tensor α [headDim] :=
  Tensor.dim (fun (j : Fin headDim) =>
    let idx := j.val
    if idx % 2 = 0 then
      if hNext : idx + 1 < headDim then
        Tensor.scalar (-Tensor.getScalar x ⟨idx + 1, hNext⟩)
      else
        -- Unpaired last entry (only possible when `headDim` is odd).
        Tensor.scalar (Tensor.getScalar x j)
    else
      have hPrev : idx - 1 < headDim :=
        Nat.lt_of_le_of_lt (Nat.sub_le idx 1) j.isLt
      Tensor.scalar (Tensor.getScalar x ⟨idx - 1, hPrev⟩))

/--
Broadcast RoPE `cos(θ)` factors to a full `(headDim)` vector for one position.

For an odd width, the final unpaired coordinate receives cosine factor `1`.
-/
def ropeCosVectorSpec (pos headDim : Nat) : Tensor α [headDim] :=
  Tensor.dim (fun (j : Fin headDim) =>
    if j.val % 2 = 0 ∧ j.val + 1 = headDim then
      Tensor.scalar 1
    else
      let iNat : Nat := j.val / 2
      let θ : α := posencAngleSpec (α := α) pos iNat headDim
      Tensor.scalar (MathFunctions.cos θ))

/--
Broadcast RoPE `sin(θ)` factors to a full `(headDim)` vector for one position.

For an odd width, the final unpaired coordinate receives sine factor `0`, so applying RoPE leaves
that coordinate unchanged.
-/
def ropeSinVectorSpec (pos headDim : Nat) : Tensor α [headDim] :=
  Tensor.dim (fun (j : Fin headDim) =>
    if j.val % 2 = 0 ∧ j.val + 1 = headDim then
      Tensor.scalar 0
    else
      let iNat : Nat := j.val / 2
      let θ : α := posencAngleSpec (α := α) pos iNat headDim
      Tensor.scalar (MathFunctions.sin θ))

/--
Apply RoPE to a single head matrix `x : (seqLen, headDim)`.

Implementation matches the standard identity:

`rope(x) = x * cos + rotatePairs(x) * sin`

where `cos` and `sin` are position-dependent vectors broadcast across the last dimension.

`startPos` is an absolute-position offset (useful for KV-cache decoding).
When `headDim` is odd, the final unpaired coordinate is preserved.
-/
def ropeApplySpec {seqLen headDim : Nat}
    (x : Tensor α [seqLen, headDim])
    (startPos : Nat := 0) :
    Tensor α [seqLen, headDim] :=
  Tensor.dim (fun (pos : Fin seqLen) =>
    let posNat : Nat := startPos + pos.val
    let row : Tensor α [headDim] := Tensor.unstack x pos
    let c : Tensor α [headDim] := ropeCosVectorSpec (α := α) posNat headDim
    let s : Tensor α [headDim] := ropeSinVectorSpec (α := α) posNat headDim
    Tensor.addSpec (Tensor.mulSpec row c)
      (Tensor.mulSpec (ropeRotatePairsSpec (α := α) (headDim := headDim) row) s))

/-- Apply RoPE to `(numHeads, seqLen, headDim)` by applying `ropeApplySpec` independently per head.
-/
def ropeApplyHeadsSpec {numHeads seqLen headDim : Nat}
    (x : Tensor α [numHeads, seqLen, headDim])
    (startPos : Nat := 0) :
    Tensor α [numHeads, seqLen, headDim] :=
  Tensor.dim (fun h =>
    ropeApplySpec (α := α) (seqLen := seqLen) (headDim := headDim)
      (Tensor.unstack x h) startPos)

end Spec
