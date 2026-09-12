/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Attention
public import NN.Proofs.Analysis.Softmax

/-!
# Attention weights sum to 1 (spec layer, `ℝ`)

For unmasked scaled dot-product attention, the attention weights are computed by applying
`Activation.softmaxSpec 1` to the scaled score matrix.

This file records the normalization theorem that each row of those unmasked weights sums to `1`.
Masked attention uses hard-mask semantics; its normalization theorem needs the additional
well-formedness assumption that each row admits at least one key.
-/

@[expose] public section

namespace NN.Proofs.Models.Attention

open Spec TorchLean
open TorchLean.Tensor
open scoped BigOperators

noncomputable section

/--
In unmasked scaled dot-product attention, each query row of the attention-weight matrix sums to `1`.

This is purely a property of axis-`1` softmax; it does not depend on the particular choice of
scores or scaling.
-/
theorem scaledDotProductAttention_unmasked_weights_row_sum_one
    {nQ nK dModel : Nat} {hQ : nQ ≠ 0} {hK : nK ≠ 0}
    (ctx : Spec.AttentionContext ℝ nQ nK dModel hQ hK) (i : Fin nQ) :
    let scale := MathFunctions.sqrt (dModel : ℝ)
    let scores := matMulSpec ctx.Q (swapAdjacentAxes ctx.K 0)
    let scaledScores := scaleSpec scores (1 / scale)
    let attentionWeights := Activation.softmaxSpec (α := ℝ) 1 scaledScores
    TorchLean.Tensor.sumSpec (Spec.get attentionWeights i) = 1 := by
  intro scale scores scaledScores attentionWeights
  simpa [attentionWeights] using
    (Proofs.sum_spec_softmax_spec_row (nQ := nQ) (nK := nK) (hK := hK)
      (scores := scaledScores) i)

end

end NN.Proofs.Models.Attention
