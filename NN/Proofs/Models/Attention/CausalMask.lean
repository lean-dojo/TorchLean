/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Attention
public import NN.Proofs.Tensor.Basic

/-!
# Causal attention mask laws

This file proves the exact Boolean semantics of TorchLean's causal and future masks and connects
those mask facts to the true hard-masked attention primitive.

TorchLean's main attention spec uses the proof-facing semantics corresponding to
`scores.masked_fill(~mask, -torch.inf)`: blocked entries receive zero softmax numerator, hence zero
attention mass.

References:
- Vaswani et al., “Attention Is All You Need”, 2017.
- PyTorch `scaled_dot_product_attention`, whose mask interface is the runtime analogue of this spec.
  https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
-/

@[expose] public section

namespace NN.Proofs.Models.Attention

open Spec
open Spec.Tensor

/-!
## Pointwise access

The definitions are deliberately simple lower/upper-triangular Boolean tensors, so the access lemmas
are definitional. Keeping them as named `[simp]` theorems lets larger attention proofs use the mask
without unfolding the tensor constructors each time.
-/

/-- Reading `causalMask n` at row `i`, column `j` returns exactly `j ≤ i`. -/
@[simp] theorem causalMask_get2 {n : Nat} (i j : Fin n) :
    Spec.get2 (Spec.causalMask n) i j = decide (j.val ≤ i.val) := by
  rfl

/-- Reading `futureMask n` at row `i`, column `j` returns exactly `i < j`. -/
@[simp] theorem futureMask_get2 {n : Nat} (i j : Fin n) :
    Spec.get2 (Spec.futureMask n) i j = decide (i.val < j.val) := by
  rfl

/--
Elementwise binary maps commute with matrix indexing.

This small tensor lemma is useful for attention proofs because masking is implemented as
`map2Spec` over the score matrix and the Boolean mask.
-/
@[simp] theorem get2_map2Spec_matrix {α β γ : Type} {m n : Nat}
    (f : α → β → γ)
    (A : Spec.Tensor α (.dim m (.dim n .scalar)))
    (B : Spec.Tensor β (.dim m (.dim n .scalar)))
    (i : Fin m) (j : Fin n) :
    Spec.get2 (map2Spec f A B) i j = f (Spec.get2 A i j) (Spec.get2 B i j) := by
  cases A with
  | dim rowsA =>
    cases B with
    | dim rowsB =>
      cases hA : rowsA i with
      | dim colsA =>
        cases hB : rowsB i with
        | dim colsB =>
          cases hcA : colsA j
          cases hcB : colsB j
          simp [Spec.get2, Spec.get, Spec.getAtSpec, map2Spec, hA, hB, hcA, hcB]

/-!
## Causal blocking and past visibility

These are the two user-facing laws: causal attention blocks strict future columns and allows every
past-or-present column.
-/

/-- A causal mask rejects every strict future key position. -/
theorem causalMask_blocks_future {n : Nat} (i j : Fin n) (hij : i.val < j.val) :
    Spec.get2 (Spec.causalMask n) i j = false := by
  simp [Nat.not_le_of_gt hij]

/-- A causal mask admits every past or current key position. -/
theorem causalMask_allows_past {n : Nat} (i j : Fin n) (hji : j.val ≤ i.val) :
    Spec.get2 (Spec.causalMask n) i j = true := by
  simp [hji]

/-- A future mask is the strict complement direction of the causal lower triangle. -/
theorem futureMask_marks_future {n : Nat} (i j : Fin n) (hij : i.val < j.val) :
    Spec.get2 (Spec.futureMask n) i j = true := by
  simp [hij]

/-- A future mask rejects every past or current key position. -/
theorem futureMask_rejects_past {n : Nat} (i j : Fin n) (hji : j.val ≤ i.val) :
    Spec.get2 (Spec.futureMask n) i j = false := by
  simp [Nat.not_lt_of_ge hji]

/-!
## Exact hard-mask attention weights

For hard masking, blocked entries are not merely assigned a very small logit. Their softmax
numerator is definitionally zero. These lemmas are the attention-level facts needed for causal
non-interference proofs.
-/

/-- Any blocked coordinate of a hard-masked softmax vector has exactly zero weight. -/
theorem hardMaskedSoftmaxVecSpec_blocked_eq_zero
    {n : Nat}
    (scores : Spec.Tensor ℝ (.dim n .scalar))
    (mask : Spec.Tensor Bool (.dim n .scalar))
    (j : Fin n)
    (hblocked : Spec.Tensor.vecGet mask j = false) :
    Spec.Tensor.vecGet (Spec.hardMaskedSoftmaxVecSpec scores mask) j = 0 := by
  cases scores with
  | dim scoreRows =>
    cases mask with
    | dim maskRows =>
      cases hscore : scoreRows j with
      | scalar score =>
        cases hmask : maskRows j with
        | scalar allowed =>
          have hallowed : allowed = false := by
            simpa [Spec.Tensor.vecGet, Spec.get, Spec.getAtSpec, hmask] using hblocked
          change Spec.Tensor.toScalar
              (Spec.get
                (Spec.hardMaskedSoftmaxVecSpec (Spec.Tensor.dim scoreRows)
                  (Spec.Tensor.dim maskRows)) j) = 0
          simp [Spec.hardMaskedSoftmaxVecSpec, Spec.get, Spec.getAtSpec,
            map2Spec, divSpec, Spec.replicate, Spec.Tensor.toScalar, hscore, hmask, hallowed]

/-- Any blocked coordinate of a row-wise hard-masked softmax matrix has exactly zero weight. -/
theorem hardMaskedSoftmaxSpec_blocked_eq_zero
    {nQ nK : Nat}
    (scores : Spec.Tensor ℝ (.dim nQ (.dim nK .scalar)))
    (mask : Spec.Tensor Bool (.dim nQ (.dim nK .scalar)))
    (i : Fin nQ) (j : Fin nK)
    (hblocked : Spec.get2 mask i j = false) :
    Spec.get2 (Spec.hardMaskedSoftmaxSpec scores mask) i j = 0 := by
  cases scores with
  | dim scoreRows =>
    cases mask with
    | dim maskRows =>
      cases hscoreRow : scoreRows i with
      | dim scoreCols =>
        cases hmaskRow : maskRows i with
        | dim maskCols =>
          have hvec :
              Spec.Tensor.vecGet
                (Spec.hardMaskedSoftmaxVecSpec (scoreRows i) (maskRows i)) j = 0 := by
            apply hardMaskedSoftmaxVecSpec_blocked_eq_zero
            simpa [Spec.get2, Spec.get, Spec.getAtSpec, Spec.Tensor.vecGet, hmaskRow]
              using hblocked
          simpa [Spec.hardMaskedSoftmaxSpec, Spec.get2, Spec.get, Spec.getAtSpec,
            Spec.Tensor.vecGet, hscoreRow, hmaskRow] using hvec

/-- In exact hard-masked causal softmax, every strict-future attention weight is exactly zero. -/
theorem hardMaskedSoftmaxSpec_causal_future_zero
    {n : Nat}
    (scores : Spec.Tensor ℝ (.dim n (.dim n .scalar)))
    (i j : Fin n) (hij : i.val < j.val) :
    Spec.get2 (Spec.hardMaskedSoftmaxSpec scores (Spec.causalMask n)) i j = 0 := by
  exact hardMaskedSoftmaxSpec_blocked_eq_zero
    scores (Spec.causalMask n) i j (causalMask_blocks_future i j hij)

end NN.Proofs.Models.Attention
