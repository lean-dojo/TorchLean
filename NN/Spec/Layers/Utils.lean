/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.TensorOps

/-!
# Image/tensor utilities (spec layer)

Convenience aliases and helpers for 2‑D images (`H×W`) and multi‑channel images (`C×H×W`),
plus padding and window‑extraction utilities used by conv/pooling layers.
-/

@[expose] public section


namespace Spec
open Tensor

-- Abbreviations
/-- A 2-D image tensor of shape `[H, W]`. -/
abbrev Image (H W : ℕ) (α : Type) := Tensor α (.dim H (.dim W .scalar))
/-- A `C`-channel image tensor of shape `[C, H, W]` (channels-first, like PyTorch `NCHW` without
  `N`). -/
abbrev MultiChannelImage (C H W : ℕ) (α : Type) := Tensor α (.dim C (.dim H (.dim W .scalar)))

/-- A 1-D signal tensor of shape `[L]`. -/
abbrev Signal (L : ℕ) (α : Type) := Tensor α (.dim L .scalar)

/-- A `C`-channel 1-D signal tensor of shape `[C, L]` (channels-first). -/
abbrev MultiChannelSignal (C L : ℕ) (α : Type) := Tensor α (.dim C (.dim L .scalar))

/-- A 3-D volume tensor of shape `[D, H, W]`. -/
abbrev Volume (D H W : ℕ) (α : Type) := Tensor α (.dim D (.dim H (.dim W .scalar)))

/-- A `C`-channel 3-D volume tensor of shape `[C, D, H, W]` (channels-first). -/
abbrev MultiChannelVolume (C D H W : ℕ) (α : Type) :=
  Tensor α (.dim C (.dim D (.dim H (.dim W .scalar))))

/--
Cast a `MultiChannelImage` along definitional equalities of its channel/height/width indices.

This is a dependent-type convenience: it does not change the underlying tensor data, only the
type-level shape indices.
-/
def rwMultiChannelImage {α : Type} {C1 C2 H1 H2 W1 W2 : ℕ} (img : MultiChannelImage C1 H1 W1 α)
  (h1 : C1 = C2) (h2 : H1 = H2) (h3 : W1 = W2) : MultiChannelImage C2 H2 W2 α :=
  have h : Shape.dim C1 (Shape.dim H1 (Shape.dim W1 .scalar)) = Shape.dim C2 (Shape.dim H2
    (Shape.dim W2 .scalar)) := by
    simp [h1, h2, h3]
  tensorCast (Shape.dim C2 (Shape.dim H2 (Shape.dim W2 .scalar))) h img

/--
Explicit-argument version of `rw_multi_channel_image`.

This is occasionally convenient when elaboration has trouble inferring `C2/H2/W2` from context.
-/
def rwMultiChannelImageExplicit {α : Type} {C1 H1 W1 : ℕ} (C2 H2 W2 : ℕ) (img :
  MultiChannelImage C1 H1 W1 α) (h1 : C1 = C2) (h2 : H1 = H2) (h3 : W1 = W2) : MultiChannelImage C2
  H2 W2 α :=
  have h : Shape.dim C1 (Shape.dim H1 (Shape.dim W1 .scalar)) = Shape.dim C2 (Shape.dim H2
    (Shape.dim W2 .scalar)) := by
    simp [h1, h2, h3]
  tensorCast (Shape.dim C2 (Shape.dim H2 (Shape.dim W2 .scalar))) h img

-- Get value at position with bounds checking.
/--
Read pixel `(x, y)` from an `Image`, returning `0` when out of bounds.

This helper is used by window-extraction and padding utilities for conv/pooling specs.
-/
def getValueAtPosition {α : Type} [Context α] {H W : ℕ} (img : Image H W α) (x y : ℕ) : Tensor α
  .scalar :=
  if h : x < H then
    if h2 : y < W then
      if x ≥ 0 ∧ y ≥ 0 then
        let x_val := getAtSpec img ⟨x, h⟩
        let y_val := getAtSpec x_val ⟨y, h2⟩
        y_val
      else Tensor.scalar 0
    else Tensor.scalar 0
  else Tensor.scalar 0

/--
`getValueAtPosition` agrees with the generic list-indexing helper `get_at_or_zero`.

In particular, reading a scalar via the specialized `(x, y)` accessor is the same as reading
with indices `[x, y]`, where both return `0` out of bounds.
-/
lemma get_at_or_zero_getValueAtPosition
    {α : Type} [Context α] {H W : ℕ} (img : Image H W α) (x y : ℕ) :
    getAtOrZero (getValueAtPosition (H := H) (W := W) img x y) [] = getAtOrZero img [x, y] :=
      by
  classical
  cases img with
  | dim f =>
      by_cases hx : x < H
      · by_cases hy : y < W
        · -- In bounds: both sides are the selected scalar.
          cases hrow : f ⟨x, hx⟩ with
          | dim fW =>
              cases hcell : fW ⟨y, hy⟩ with
              | scalar v =>
                  simp [getValueAtPosition, getAtSpec, hx, hy, hrow, hcell]
        · -- Out of bounds on y: both sides are `0`.
          cases hrow : f ⟨x, hx⟩ with
          | dim fW =>
              simp [getValueAtPosition, hx, hy, hrow]
      · -- Out of bounds on x.
        simp [getValueAtPosition, hx]

-- Extract a window from an image at a specific position.
/--
Extract a `kH × kW` patch from an image starting at `(start_i, start_j)`.

Out-of-bounds pixels are treated as `0`, matching the behavior of `getValueAtPosition`. This is
spec-level "im2col"-style logic (cf. PyTorch `nn.Unfold`, conceptually).
-/
def extractWindow {α : Type} [Context α] {H W : ℕ} (kW kH : ℕ)
  (img : Image H W α)
  (start_i start_j : ℕ) : Tensor α (.dim kH (.dim kW .scalar)) :=
  Tensor.dim (fun di =>
    Tensor.dim (fun dj =>
      let x := start_i + di.val
      let y := start_j + dj.val
      getValueAtPosition img x y))

-- Pad a multi-channel image (zero padding).
/--
Zero-pad a channels-first image by `padding` pixels on each spatial axis.

This is the spec analogue of `torch.nn.functional.pad` (with constant `0` padding). The output
shape is `[inC, inH + 2*padding, inW + 2*padding]`.
-/
def padMultiChannel {α : Type} [Context α] {inC inH inW : ℕ} (img : MultiChannelImage inC inH inW
  α) (padding : ℕ) :
  MultiChannelImage inC (inH + 2 * padding) (inW + 2 * padding) α :=
  Tensor.dim (fun c =>
    Tensor.dim (fun i =>
      Tensor.dim (fun j =>
        if _h : i.val < padding ∨ j.val < padding then
          Tensor.scalar 0
        else
          let x := i.val - padding
          let y := j.val - padding
          getValueAtPosition (getAtSpec img c) x y)))

/--
Characterization lemma for `pad_multi_channel` under list-indexing (`get_at_or_zero`).

Reading the padded tensor at `[c, p, q]` yields `0` in the top/left padding region, and otherwise
reads the original tensor at `[c, p - padding, q - padding]` (with out-of-bounds falling back to
`0` on both sides).
-/
lemma get_at_or_zero_pad_multi_channel
    {α : Type} [Context α] {inC inH inW padding : ℕ}
    (img : MultiChannelImage inC inH inW α) (c : Fin inC) (p q : ℕ) :
    getAtOrZero (padMultiChannel (inC := inC) (inH := inH) (inW := inW) img padding) [c.val, p,
      q]
      =
    (if _h : p < padding ∨ q < padding then
        (0 : α)
      else
        getAtOrZero img [c.val, p - padding, q - padding]) := by
  classical
  cases img with
  | dim fC =>
      by_cases hp : p < inH + 2 * padding
      · by_cases hq : q < inW + 2 * padding
        · -- In bounds: unfold to the `pad_multi_channel` definition.
          by_cases ht : p < padding ∨ q < padding
          · simp [padMultiChannel, c.isLt, hp, hq, ht]
          · simp [padMultiChannel, getAtSpec, c.isLt, hp, hq, ht,
            get_at_or_zero_getValueAtPosition]
        · -- `q` out of bounds: the padded read is `0`, and the source read is also `0`.
          have hq' : ¬ q < inW + 2 * padding := hq
          by_cases ht : p < padding ∨ q < padding
          · simp [padMultiChannel, c.isLt, hp, hq', ht]
          ·
            have hq_ge : inW ≤ q - padding := by
              have hq_ge2 : inW + 2 * padding ≤ q := Nat.le_of_not_gt hq'
              have hq_ge1 : inW + padding ≤ q - padding := by
                -- `(inW + padding) + padding = inW + 2*padding`
                have : (inW + padding) + padding ≤ q := by
                  simpa [two_mul, Nat.add_assoc, Nat.add_left_comm, Nat.add_comm] using hq_ge2
                exact Nat.le_sub_of_add_le this
              exact le_trans (Nat.le_add_right inW padding) hq_ge1
            have hq_out : ¬ q - padding < inW := Nat.not_lt_of_ge hq_ge
            -- Reduce RHS via the out-of-bounds check on the last axis.
            by_cases hp0 : p - padding < inH
            ·
            -- When `p - padding < inH`, show the RHS is `0` by forcing the final width check to
            -- fail.
              cases hfc : fC c with
              | dim fH =>
                  simp [padMultiChannel, getAtSpec, c.isLt, hp, hq', ht, hfc, hp0,
                    ]
                  cases hrow : fH ⟨p - padding, hp0⟩ with
                  | dim fW =>
                      simp [hq_out]
            · -- When `p - padding` is out of bounds, the source read is `0`.
              cases hfc : fC c with
              | dim fH =>
                  simp [padMultiChannel, getAtSpec, c.isLt, hp, hq', ht, hp0, hfc]
      · -- `p` out of bounds: the padded read is `0`, and the source read is also `0`.
        have hp' : ¬ p < inH + 2 * padding := hp
        by_cases ht : p < padding ∨ q < padding
        · simp [padMultiChannel, c.isLt, hp', ht]
        ·
          have hp_ge : inH ≤ p - padding := by
            have hp_ge2 : inH + 2 * padding ≤ p := Nat.le_of_not_gt hp'
            have hp_ge1 : inH + padding ≤ p - padding := by
              have : (inH + padding) + padding ≤ p := by
                simpa [two_mul, Nat.add_assoc, Nat.add_left_comm, Nat.add_comm] using hp_ge2
              exact Nat.le_sub_of_add_le this
            exact le_trans (Nat.le_add_right inH padding) hp_ge1
          have hp_out : ¬ p - padding < inH := Nat.not_lt_of_ge hp_ge
          -- RHS is `0` because the height check fails at the first spatial axis.
          cases hfc : fC c with
          | dim fH =>
              simp [padMultiChannel, getAtSpec, c.isLt, hp', ht, hp_out, hfc]

/--
Index-shift lemma for `pad_multi_channel`.

If `(i, j)` is in-bounds for the original image, then reading the padded image at
`(i + padding, j + padding)` returns the same value.
-/
lemma get_at_or_zero_pad_multi_channel_shift
    {α : Type} [Context α] {inC inH inW padding : ℕ}
    (img : MultiChannelImage inC inH inW α) (c : Fin inC) (i : Fin inH) (j : Fin inW) :
    getAtOrZero (padMultiChannel (inC := inC) (inH := inH) (inW := inW) img padding)
        [c.val, i.val + padding, j.val + padding]
      =
    getAtOrZero img [c.val, i.val, j.val] := by
  -- Use the general pad read lemma; this coordinate is never in the top/left padding.
  have hpad :=
    get_at_or_zero_pad_multi_channel (padding := padding)
      (img := img) (c := c) (p := i.val + padding) (q := j.val + padding)
  have hnp : ¬((i.val + padding) < padding ∨ (j.val + padding) < padding) := by
    have hge_i : padding ≤ i.val + padding := by
      simp [Nat.add_comm]
    have hge_j : padding ≤ j.val + padding := by
      simp [Nat.add_comm]
    intro h
    cases h with
    | inl hi' => exact (Nat.not_lt_of_ge hge_i) hi'
    | inr hj' => exact (Nat.not_lt_of_ge hge_j) hj'
  -- Specialize the `if`-formula and cancel the padding shift.
  have : getAtOrZero (padMultiChannel (inC := inC) (inH := inH) (inW := inW) img padding)
        [c.val, i.val + padding, j.val + padding]
        =
      getAtOrZero img [c.val, (i.val + padding) - padding, (j.val + padding) - padding] := by
    simpa [hnp] using hpad
  simpa [Nat.add_sub_cancel] using this

-- Extract window from multi-channel image
/--
Extract a `kH × kW` window from each channel of a channels-first image.

The input is typically a padded image, and the result has shape `[inC, kH, kW]`.
-/
def extractMultiWindow {α : Type} [Context α] {inC kH kW inH inW padding : ℕ}
  (img : MultiChannelImage inC (inH + 2 * padding) (inW + 2 * padding) α)
  (start_i start_j : ℕ) :
  Tensor α (.dim inC (.dim kH (.dim kW .scalar))) :=
  Tensor.dim (fun c =>
    extractWindow kW kH (getAtSpec img c) start_i start_j)

-- Pad channels with zeros (for ResNet shortcut connections)
-- Expands from inChannels to outChannels by zero-padding additional channels
/--
Increase the channel dimension by zero-padding extra channels.

This is used in some ResNet-style skip connections when `inChannels < outChannels`. Existing
channels are copied; newly introduced channels are identically zero.
-/
def padChannelsZero {α : Type} [Zero α] {inChannels outChannels height width : ℕ}
  (_h : inChannels ≤ outChannels)
  (img : MultiChannelImage inChannels height width α) :
  MultiChannelImage outChannels height width α :=
  Tensor.dim (fun c =>
    if h_lt : c.val < inChannels then
      -- Copy existing channel
      getAtSpec img ⟨c.val, h_lt⟩
    else
      -- Zero-pad additional channels
      Tensor.dim (fun _ => Tensor.dim (fun _ => Tensor.scalar (0 : α)))
  )

-- Identity mapping when channel dimensions match
/-- Identity on `MultiChannelImage` (useful as a "no-op" branch in higher-level specs). -/
def channelIdentity {α : Type} {channels height width : ℕ}
  (img : MultiChannelImage channels height width α) :
  MultiChannelImage channels height width α := img

-- Set value at position
/--
Write a value at pixel `(x, y)` if it is in-bounds; otherwise return the original image.

This uses `update_tensor_spec` under the hood and is intended for small spec-level utilities.
-/
def setValueAtPosition {α : Type} {H W : ℕ} (img : Image H W α) (x y : ℕ) (value : α) : Image H W α
  :=
  if _ : x < H then
    if _ : y < W then
      updateTensorSpec img [x, y] value
    else img
  else img

-- Add value at position
/--
Add `value` to pixel `(x, y)` if it is in-bounds; otherwise return the original image.

This is a small helper for accumulation-style specs (e.g. naive convolution).
-/
def addValueAtPosition {α : Type} [Add α] {H W : ℕ} (img : Image H W α) (x y : ℕ) (value : α) :
  Image H W α :=
  if _ : x < H then
    if _ : y < W then
      let current := getSpec img [x, y]
      match current with
      | some current_val => updateTensorSpec img [x, y] (current_val + value)
      | none => img
    else img
  else img

-- Create output image with zeros
/-- Construct an `H × W` image filled with zeros. -/
def createZeroImage {α : Type} [Zero α] (H W : ℕ) : Image H W α :=
  Tensor.dim (fun _ =>
    Tensor.dim (fun _ =>
      Tensor.scalar 0))

-- Create multi-channel output image with zeros
/-- Construct a `C × H × W` channels-first image filled with zeros. -/
def createZeroMultiChannelImage (α : Type) [Zero α] (C H W : ℕ) : MultiChannelImage C H W α :=
  Tensor.dim (fun _ =>
    createZeroImage H W)

end Spec
