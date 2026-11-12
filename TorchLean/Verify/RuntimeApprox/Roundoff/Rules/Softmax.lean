/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Tensor.Layers.Attention
public import TorchLean.Tensor.Layers.Loss
public meta import TorchLean.Verify.RuntimeApprox.Roundoff.Rules
public import FloatLib.Floats.Formats.BinaryInterchange.Automation.Roundoff.Extension
public import TorchLean.Verify.RuntimeApprox.Roundoff.Rules

/-!
# Softmax reads for `roundoff`

Softmax, log-softmax, their derivatives, the log-probabilities of cross-entropy and its terms are
read at a coordinate as their vector kernels on the row or fiber through it. The last-axis kernels
are read by their row here, and their reads at any `Fin` axis are in
`TorchLean.Verify.RuntimeApprox.Roundoff.Rules.Composite`. The batched attention products and
weights, and the value and multiplier cotangents of the normalized matrix product, are read as their
unbatched forms on the slices at the batch coordinate. All of these reads are registered with
`roundoff_read`, so they apply where the shape is not literal, and elsewhere the producers unfold.

The maximum allowed score of a hard-masked row is read as the stored score at the hard-masked pivot
(`TorchLean.Softmax.hardMaskedMax?_eq_map`), so its reference reads that score through the producer
that computes it. An entry of the softmax pullback that retains the gradient contractions is read
from the centered pullback of its row (`TorchLean.Attention.pullbackProducts_apply`), with no case
split on either extent.
-/

public section

attribute [roundoff_read]
  TorchLean.Softmax.innermost_apply
  TorchLean.Softmax.innermostBackward_apply
  TorchLean.LogSoftmax.innermost_apply
  TorchLean.LogSoftmax.innermostJVP_apply
  TorchLean.LogSoftmax.innermostBackward_apply
  TorchLean.CrossEntropy.innermostLogProbs_apply
  TorchLean.CrossEntropy.terms_apply
  TorchLean.Attention.scaledProducts_apply
  TorchLean.Attention.weights_apply
  TorchLean.Attention.values_apply
  TorchLean.Attention.valuesBackward_snd_fst_apply
  TorchLean.Attention.valuesBackward_snd_snd_apply

attribute [roundoff_rule]
  TorchLean.Softmax.hardMaskedMax?_eq_map
  TorchLean.Attention.pullbackProducts_apply
