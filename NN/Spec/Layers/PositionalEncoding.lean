/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.TensorOps

/-!
# Positional encodings (spec layer)

This file provides the simplest positional encoding definition: a **learnable** per-position embedding
that is added to token embeddings.

PyTorch analogy:

- This is the same idea as having an `nn.Embedding(max_len, d_model)` (or a parameter tensor
  `[max_len, d_model]`) and doing `x + pos[:seqLen]`.

Why learnable positional encodings show up a lot in practice:

- they are easy to train and tend to work well for fixed-length settings (e.g. ViT with a chosen
  patch grid, or language models trained with a fixed max sequence length),
- they keep the spec lightweight: there is no trigonometry, complex numbers, or special casing for
  even/odd dimensions.

If you want sinusoidal encodings (Transformer) or RoPE/rotary encodings, those can be defined as
pure functions that produce a tensor of shape `(seqLen, embedDim)` and then reused with the same
`add_positional_encoding_spec` below.

Reference (sinusoidal): "Attention Is All You Need" (Vaswani et al., 2017):
  https://arxiv.org/abs/1706.03762
-/

@[expose] public section


namespace Spec

open Tensor

variable {α : Type} [Context α]

/--
Learnable positional encoding parameters for a fixed `(seqLen, embedDim)`.

This is intentionally just a tensor of trainable parameters. Higher-level models decide how to
initialize it and whether to share/resize it across different sequence lengths.
-/
structure PositionalEncodingSpec (seqLen embedDim : Nat) (α : Type) where
  /-- pos. -/
  pos : Tensor α (.dim seqLen (.dim embedDim .scalar))

/--
Add positional encodings: `y = x + pos`.

Both `x` and `pos` have the same shape, so this is just elementwise addition (no broadcasting).
-/
def addPositionalEncodingSpec {seqLen embedDim : Nat}
    (pe : PositionalEncodingSpec seqLen embedDim α)
    (x : Tensor α (.dim seqLen (.dim embedDim .scalar))) :
    Tensor α (.dim seqLen (.dim embedDim .scalar)) :=
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

/-- Backward/VJP for `add_positional_encoding_spec`. -/
def addPositionalEncodingBackwardSpec {seqLen embedDim : Nat}
    (_pe : PositionalEncodingSpec seqLen embedDim α)
    (grad_output : Tensor α (.dim seqLen (.dim embedDim .scalar))) :
    (Tensor α (.dim seqLen (.dim embedDim .scalar)) ×  -- ∂L/∂pos
     Tensor α (.dim seqLen (.dim embedDim .scalar))) := -- ∂L/∂x
  (grad_output, grad_output)

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

This keeps the definition in terms of the scalar `Context` operations.
-/
def posencFreqDenomSpec (i d : Nat) : α :=
  MathFunctions.exp (Numbers.log10000 * ((Numbers.two * (i : α)) / (d : α)))

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
    Tensor α (.dim seqLen (.dim embedDim .scalar)) :=
  Tensor.dim (fun (pos : Fin seqLen) =>
    Tensor.dim (fun (j : Fin embedDim) =>
      let posNat : Nat := startPos + pos.val
      let iNat : Nat := j.val / 2
      let θ : α := posencAngleSpec (α := α) posNat iNat embedDim
      let v : α := if j.val % 2 = 0 then MathFunctions.sin θ else MathFunctions.cos θ
      Tensor.ofScalar v))

/--
Add sinusoidal positional encodings: `y = x + sinusoidal(startPos, seqLen, embedDim)`.
-/
def addSinusoidalPositionalEncodingSpec {seqLen embedDim : Nat}
    (x : Tensor α (.dim seqLen (.dim embedDim .scalar)))
    (startPos : Nat := 0) :
    Tensor α (.dim seqLen (.dim embedDim .scalar)) :=
  Tensor.addSpec x (sinusoidalPositionalEncodingSpec (α := α) seqLen embedDim startPos)

/-!
## Rotary positional embeddings (RoPE) utilities (pure functions)

RoPE (Su et al., 2021) encodes position by applying a 2D rotation to each pair of features
in the last dimension.

In most transformer implementations, RoPE is applied to query/key head vectors:
- per head: `(seqLen, headDim)`
- all heads: `(numHeads, seqLen, headDim)`

This file intentionally provides only **pure** RoPE helpers; we do not integrate them into
attention yet.

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
def ropeRotatePairsLastdimSpec {headDim : Nat}
    (x : Tensor α (.dim headDim .scalar)) :
    Tensor α (.dim headDim .scalar) :=
  Tensor.dim (fun (j : Fin headDim) =>
    let idx := j.val
    if idx % 2 = 0 then
      if hNext : idx + 1 < headDim then
        Tensor.ofScalar (-Tensor.vecGet x ⟨idx + 1, hNext⟩)
      else
        -- Unpaired last entry (only possible when `headDim` is odd).
        Tensor.ofScalar (Tensor.vecGet x j)
    else
      have hPrev : idx - 1 < headDim :=
        Nat.lt_of_le_of_lt (Nat.sub_le idx 1) j.isLt
      Tensor.ofScalar (Tensor.vecGet x ⟨idx - 1, hPrev⟩))

/-- Broadcast RoPE `cos(θ)` factors to a full `(headDim)` vector for one position. -/
def ropeCosLastdimSpec (pos headDim : Nat) : Tensor α (.dim headDim .scalar) :=
  Tensor.dim (fun (j : Fin headDim) =>
    let iNat : Nat := j.val / 2
    let θ : α := posencAngleSpec (α := α) pos iNat headDim
    Tensor.ofScalar (MathFunctions.cos θ))

/-- Broadcast RoPE `sin(θ)` factors to a full `(headDim)` vector for one position. -/
def ropeSinLastdimSpec (pos headDim : Nat) : Tensor α (.dim headDim .scalar) :=
  Tensor.dim (fun (j : Fin headDim) =>
    let iNat : Nat := j.val / 2
    let θ : α := posencAngleSpec (α := α) pos iNat headDim
    Tensor.ofScalar (MathFunctions.sin θ))

/--
Apply RoPE to a single head matrix `x : (seqLen, headDim)`.

Implementation matches the standard identity:

`rope(x) = x * cos + rotatePairs(x) * sin`

where `cos` and `sin` are position-dependent vectors broadcast across the last dimension.

`startPos` is an absolute-position offset (useful for KV-cache decoding).
-/
def ropeApplySpec {seqLen headDim : Nat}
    (x : Tensor α (.dim seqLen (.dim headDim .scalar)))
    (startPos : Nat := 0) :
    Tensor α (.dim seqLen (.dim headDim .scalar)) :=
  Tensor.dim (fun (pos : Fin seqLen) =>
    let posNat : Nat := startPos + pos.val
    let row : Tensor α (.dim headDim .scalar) := x[pos]
    let c : Tensor α (.dim headDim .scalar) := ropeCosLastdimSpec (α := α) posNat headDim
    let s : Tensor α (.dim headDim .scalar) := ropeSinLastdimSpec (α := α) posNat headDim
    Tensor.addSpec (Tensor.mulSpec row c)
      (Tensor.mulSpec (ropeRotatePairsLastdimSpec (α := α) (headDim := headDim) row) s))

/-- Apply RoPE to `(numHeads, seqLen, headDim)` by applying `rope_apply_spec` independently per head. -/
def ropeApplyHeadsSpec {numHeads seqLen headDim : Nat}
    (x : Tensor α (.dim numHeads (.dim seqLen (.dim headDim .scalar))))
    (startPos : Nat := 0) :
    Tensor α (.dim numHeads (.dim seqLen (.dim headDim .scalar))) :=
  Tensor.dim (fun h => ropeApplySpec (α := α) (seqLen := seqLen) (headDim := headDim) x[h] startPos)

end Spec
