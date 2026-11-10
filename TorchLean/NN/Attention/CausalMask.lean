/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Context.Real
public import TorchLean.Tensor.Layers.Attention

/-!
# Causal attention mask laws

Under `causalMask n` query `i` may attend to key `j` when `j ≤ i`, and under `futureMask n` when
`i < j`. The theorems here prove these readings, the blocking laws that follow from them and their
consequence for attention: `Softmax.hardMasked` gives a blocked entry a zero softmax
numerator and so zero weight.

Real numbers have no negative infinity, and the zero numerator is what a `-∞` logit would
contribute through `exp`. The laws are about the specification and do not cover native attention
implementations.

References:
- Vaswani et al., “Attention Is All You Need”, 2017.
- PyTorch `scaled_dot_product_attention`, whose mask interface is the runtime analogue of this spec.
  https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
-/

@[expose] public section

namespace TorchLean

open TorchLean.Tensor

/-!
## Pointwise access

The causal mask is lower triangular and the future mask strictly upper triangular, each a Boolean
tensor built entry by entry, so the access lemmas are definitional. As `[simp]` lemmas they let
larger attention proofs read a mask entry without unfolding the tensor constructors.
-/

/-- Reading `causalMask n` at row `i`, column `j` returns exactly `j ≤ i`. -/
@[simp] theorem Attention.get2_causalMask {n : Nat} (i j : Fin n) :
    Tensor.get2 (causalMask n) i j = decide (j.val ≤ i.val) := by
  simp [causalMask, Tensor.get2]

/-- Reading `futureMask n` at row `i`, column `j` returns exactly `i < j`. -/
@[simp] theorem Attention.get2_futureMask {n : Nat} (i j : Fin n) :
    Tensor.get2 (futureMask n) i j = decide (i.val < j.val) := by
  simp [futureMask, Tensor.get2]

/-!
## Causal blocking and past visibility

The causal mask blocks every strict future column and allows every past or present one, and the
future mask does the reverse.
-/

/-- A causal mask rejects every strict future key position. -/
theorem Attention.get2_causalMask_eq_false_of_lt {n : Nat} (i j : Fin n) (hij : i.val < j.val) :
    Tensor.get2 (causalMask n) i j = false := by
  simp [Nat.not_le_of_gt hij]

/-- A causal mask admits every past or current key position. -/
theorem Attention.get2_causalMask_eq_true_of_le {n : Nat} (i j : Fin n) (hji : j.val ≤ i.val) :
    Tensor.get2 (causalMask n) i j = true := by
  simp [hji]

/-- A future mask marks every strict future key position `true`. -/
theorem Attention.get2_futureMask_eq_true_of_lt {n : Nat} (i j : Fin n) (hij : i.val < j.val) :
    Tensor.get2 (futureMask n) i j = true := by
  simp [hij]

/-- A future mask rejects every past or current key position. -/
theorem Attention.get2_futureMask_eq_false_of_le {n : Nat} (i j : Fin n) (hji : j.val ≤ i.val) :
    Tensor.get2 (futureMask n) i j = false := by
  simp [Nat.not_lt_of_ge hji]

/-!
## Exact hard-mask attention weights

Hard masking gives a blocked entry a zero softmax numerator by definition, so over `ℝ` its weight
is zero. A causal non-interference argument for a model takes these zero weights as its
attention-level step.
-/

/-- Any blocked coordinate of a hard-masked softmax vector has exactly zero weight. -/
theorem Softmax.hardMaskedVec_blocked_eq_zero
    {n : Nat}
    (scores : Tensor ℝ [n])
    (mask : Tensor Bool [n])
    (j : Fin n)
    (hblocked : Tensor.getScalar mask j = false) :
    Tensor.getScalar (hardMaskedVec scores mask) j = 0 := by
  have hblocked' : mask (j, PUnit.unit) = false := by
    simpa [Tensor.getScalar_eq_apply] using hblocked
  unfold hardMaskedVec
  split
  · simp
  · change Tensor.getScalar (_ / _) j = 0
    simp [Tensor.getScalar_eq_apply, div_def, hblocked']

/-- Any blocked coordinate of a row-wise hard-masked softmax matrix has exactly zero weight. -/
theorem Softmax.hardMasked_blocked_eq_zero
    {nQ nK : Nat}
    (scores : Tensor ℝ [nQ, nK])
    (mask : Tensor Bool [nQ, nK])
    (i : Fin nQ) (j : Fin nK)
    (hblocked : Tensor.get2 mask i j = false) :
    Tensor.get2 (hardMasked scores mask) i j = 0 := by
  rw [Tensor.get2_eq_getScalar_unstack]
  simp only [hardMasked, Tensor.unstack_dim]
  apply hardMaskedVec_blocked_eq_zero
  exact hblocked

/-- Under the causal mask, hard-masked softmax over `ℝ` gives query `i` zero weight on every key
`j` with `i < j`. -/
theorem Softmax.hardMasked_causalMask_eq_zero
    {n : Nat}
    (scores : Tensor ℝ [n, n])
    (i j : Fin n) (hij : i.val < j.val) :
    Tensor.get2 (hardMasked scores (Attention.causalMask n)) i j = 0 := by
  exact hardMasked_blocked_eq_zero
    scores (Attention.causalMask n) i j (Attention.get2_causalMask_eq_false_of_lt i j hij)

end TorchLean
