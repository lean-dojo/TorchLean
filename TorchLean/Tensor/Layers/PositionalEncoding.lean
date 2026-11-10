/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Arithmetic
public import TorchLean.Tensor.ArgMax

/-!
# Positional encodings (spec layer)

The sinusoidal positional encoding and the rotary position embedding (RoPE) work on tensors of any
shape, with positions along one axis `dim`, features along a different axis `featureDim`, and
every other axis a batch axis. The four tensor definitions take a proof `h : dim ≠ featureDim`
that the two axes differ, and no entry depends on it.

- `Tensor.sinusoidalPositionalEncoding s dim featureDim h startPos` is the table of
  "Attention Is All You Need": feature `2 * i` at position `p` is `sin θ` and feature `2 * i + 1`
  is `cos θ`, for `θ = (p + startPos) / exp (log 10000 * (2 * i / d))` and the feature extent `d`.
  The model adds it to its input with broadcasting.
- `Tensor.rope x dim featureDim h startPos` rotates each feature pair `(a, b)` at position `p` to
  `(a * cos θ + (-b) * sin θ, b * cos θ + a * sin θ)` with the same angle `θ` (Su et al., 2021),
  and returns an unpaired last feature (odd feature extent) unchanged. `Tensor.ropeCos` and
  `Tensor.ropeSin` are its factor tables, `1` and `0` at an unpaired feature.

A learnable positional table needs no definition of its own: it is a parameter tensor added to the
input, as `x + pos[:seqLen]` in PyTorch.

Vaswani et al.'s table is coded by hand, with the feature extent `d` as their `d_model`. RoPE,
also by hand, is eq. (34) of Su et al., the pairwise form of the block-diagonal rotation in
eq. (15). Their frequencies `θ_k = 10000^(-2 (k - 1) / d)` count pairs from `k = 1`, so our pair
`i` (features `2 * i` and `2 * i + 1`) uses `θ_(i+1)`. Su et al. take `d` even, and passing an odd
last feature through is our choice. Both encodings divide the position by `10000^(2 * i / d)`, the
way Vaswani et al. write it. Su et al. multiply `m` by `θ_(i+1)`, which agrees over `ℝ` and can
round differently in a binary format.

Reference (sinusoidal): Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser and Polosukhin,
"Attention Is All You Need", NIPS 2017, arXiv:1706.03762, Sec. 3.5:
  https://arxiv.org/abs/1706.03762
Reference (RoPE): Su, Lu, Pan, Murtadha, Wen and Liu, "RoFormer: Enhanced Transformer with Rotary
Position Embedding" (2021), arXiv:2104.09864, Sec. 3.2.2 and 3.4.2:
  https://arxiv.org/abs/2104.09864
-/

@[expose] public section


open FloatLib.Numerics (MathFunctions)

namespace TorchLean

open TorchLean.Tensor

variable {α : Type} [Storage α] [Context α]

/--
Frequency denominator used by both sinusoidal PE and RoPE:

`denom(i; d) = 10000^(2*i / d)`

implemented as:

`exp(log(10000) * (2*i / d))`

The exponent `2*i / d` is the exact ratio cast once into the scalar, and every other operation
uses the selected scalar backend. On a grid that cannot represent `10000`, the logarithm receives
the rounded base, so its result is not a single rounding of the real `log 10000`.
-/
def PositionalEncoding.freqDenom (i d : Nat) : α :=
  MathFunctions.exp ((MathFunctions.log 10000) * ((2 * i / d : Rat) : α))

/--
Common angle used by sinusoidal PE and RoPE:

`θ(pos, i; d) = pos / 10000^(2*i/d)`.
-/
def PositionalEncoding.angle (pos i d : Nat) : α :=
  (pos : α) / freqDenom (α := α) i d

end TorchLean

namespace TorchLean.Tensor

variable {α : Type} [Storage α] [Context α]

/-- The sinusoidal positional encoding of shape `s`, positions along `dim` and features along
`featureDim`: feature `2 * i` is `sin θ` and feature `2 * i + 1` is `cos θ` for the angle
`PositionalEncoding.angle (p + startPos) i d` of position `p` and feature extent `d`. -/
def sinusoidalPositionalEncoding (s : Shape) (dim featureDim : Fin s.rank)
    (_h : dim ≠ featureDim) (startPos : Nat := 0) : Tensor α s :=
  Rep.ofFn fun c =>
    let j := (Shape.Coord.get c featureDim).val
    let θ : α := PositionalEncoding.angle (startPos + (Shape.Coord.get c dim).val) (j / 2)
      (s.extent featureDim)
    if j % 2 = 0 then MathFunctions.sin θ else MathFunctions.cos θ

/-- The cosine factors of `rope` on shape `s`, positions along `dim` and features along
`featureDim`: `cos θ` at feature `j` for the angle `θ` of pair `j / 2`, and `1` at an unpaired
last feature. -/
def ropeCos (s : Shape) (dim featureDim : Fin s.rank) (_h : dim ≠ featureDim)
    (startPos : Nat := 0) : Tensor α s :=
  Rep.ofFn fun c =>
    let j := (Shape.Coord.get c featureDim).val
    if j % 2 = 0 ∧ j + 1 = s.extent featureDim then 1 else
      MathFunctions.cos (PositionalEncoding.angle (startPos + (Shape.Coord.get c dim).val) (j / 2)
        (s.extent featureDim))

/-- The sine factors of `rope` on shape `s`, positions along `dim` and features along
`featureDim`: `sin θ` at feature `j` for the angle `θ` of pair `j / 2`, and `0` at an unpaired
last feature. -/
def ropeSin (s : Shape) (dim featureDim : Fin s.rank) (_h : dim ≠ featureDim)
    (startPos : Nat := 0) : Tensor α s :=
  Rep.ofFn fun c =>
    let j := (Shape.Coord.get c featureDim).val
    if j % 2 = 0 ∧ j + 1 = s.extent featureDim then 0 else
      MathFunctions.sin (PositionalEncoding.angle (startPos + (Shape.Coord.get c dim).val) (j / 2)
        (s.extent featureDim))

/-- Rotary position embedding of `x`, positions along `dim` and feature pairs along `featureDim`:
the pair `(a, b)` becomes `(a * cos θ + (-b) * sin θ, b * cos θ + a * sin θ)` with the factors of
`ropeCos` and `ropeSin`, and an unpaired last feature is returned unchanged. -/
def rope {s : Shape} (x : Tensor α s) (dim featureDim : Fin s.rank) (h : dim ≠ featureDim)
    (startPos : Nat := 0) : Tensor α s :=
  Rep.ofFn fun c =>
    let j := Shape.Coord.get c featureDim
    let cosθ := ropeCos s dim featureDim h startPos c
    let sinθ := ropeSin s dim featureDim h startPos c
    if j.val % 2 = 0 then
      if hNext : j.val + 1 < s.extent featureDim then
        x c * cosθ + -x (Shape.Coord.update c featureDim ⟨j.val + 1, hNext⟩) * sinθ
      else x c
    else
      x c * cosθ + x (Shape.Coord.update c featureDim
        ⟨j.val - 1, Nat.lt_of_le_of_lt (Nat.sub_le _ 1) j.isLt⟩) * sinθ

end TorchLean.Tensor
