/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.TensorReductionShape.Broadcasting
public import NN.Spec.Core.TensorReductionShape.Reductions

/-!
# Normalization layers (spec layer)

This file collects a few normalization operators used throughout TorchLean's spec/model code.

The common pattern is:

- compute per-axis statistics (mean / variance or RMS),
- normalize with an `epsilon` for numerical stability,
- optionally apply an affine transform (`gamma`, `beta`) like PyTorch does.

The familiar normalization and differential interpretations require a positive `epsilon` and
suitable real-number laws; the raw scalar-polymorphic definitions do not validate that parameter.
For floating-point contexts, the forward, JVP, and VJP are separate rounded programs. Their
closed-form differential formulas do not assert a derivative of IEEE rounding or bitwise equality
with a native backend. In particular, LayerNorm computes `reduceVar` of already centered data,
which centers again; simplifying that second centering changes floating-point execution.

## References (papers + PyTorch behavior)

- LayerNorm: Ba et al., "Layer Normalization" (2016): https://arxiv.org/abs/1607.06450
- BatchNorm: Ioffe, Szegedy, "Batch Normalization" (2015): https://arxiv.org/abs/1502.03167
- GroupNorm: Wu, He, "Group Normalization" (2018): https://arxiv.org/abs/1803.08494
- RMSNorm: Zhang, Sennrich, "Root Mean Square Layer Normalization" (2019):
  https://arxiv.org/abs/1910.07467
- WeightNorm: Salimans, Kingma, "Weight Normalization" (2016): https://arxiv.org/abs/1602.07868

- PyTorch LayerNorm: https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
- PyTorch BatchNorm modules: https://docs.pytorch.org/docs/stable/nn.html#normalization-layers
-/

@[expose] public section


open TorchLean

namespace Spec
open TorchLean TorchLean.Tensor

variable {α : Type} [TorchLean.Storage α] [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)]

/-- Named reverse-mode result shared by affine normalization operators. -/
structure NormalizationGradients (α : Type) [TorchLean.Storage α]
    (inputShape parameterShape : Shape) where
  /-- Gradient with respect to the normalized input. -/
  inputGradient : Tensor α inputShape
  /-- Gradient with respect to the learned multiplicative scale. -/
  scaleGradient : Tensor α parameterShape
  /-- Gradient with respect to the learned additive bias. -/
  biasGradient : Tensor α parameterShape
deriving Repr

/-- Core normalization routine with explicit broadcast proofs.

This is the shared “math step” behind normalization layers:

`y = ((x - mean) / sqrt(variance + ε)) * gamma + beta`.
-/
def normalizeCore
  (s sMean sVar sGamma sBeta : Shape)
  (epsilon : α)
  (x : Tensor α s)
  (mean : Tensor α sMean)
  (variance : Tensor α sVar)
  (gamma : Tensor α sGamma)
  (beta : Tensor α sBeta)
  (cbMean : Shape.CanBroadcastTo sMean s)
  (cbVar : Shape.CanBroadcastTo sVar s)
  (cbGamma : Shape.CanBroadcastTo sGamma s)
  (cbBeta : Shape.CanBroadcastTo sBeta s) : Tensor α s :=

  let mean_broadcast := broadcastTo cbMean mean
  let varianceBroadcast := broadcastTo cbVar variance
  let gammaBroadcast := broadcastTo cbGamma gamma
  let betaBroadcast := broadcastTo cbBeta beta

  let centered := subSpec x mean_broadcast
  let std := sqrtSpec (addSpec varianceBroadcast (Tensor.full s epsilon))
  let normalized := divSpec centered std
  addSpec (mulSpec normalized gammaBroadcast) betaBroadcast


/-
  Layer Normalization
  Normalizes along the last dimension
-/
/-- LayerNorm over the last dimension of a `(seqLen, embedDim)` tensor.

Uses `epsilon` (default `TorchLean.normalizationEpsilon`) for numerical stability
in the denominator. The default can round to zero in tiny formats and has no fallback. For those
formats, pass a representable positive, finite `epsilon` explicitly; a constant row otherwise
produces a zero denominator. This raw scalar-polymorphic operation does not validate the argument.
-/
def layerNorm {seqLen embedDim : Nat}
  (x : Tensor α [seqLen, embedDim])
  (gamma : Tensor α [embedDim])
  (beta : Tensor α [embedDim])
  (h_seq_pos : seqLen > 0 := by norm_num)
  (h_embed_pos : embedDim > 0 := by norm_num)
  (epsilon : α := TorchLean.normalizationEpsilon) :
  Tensor α [seqLen, embedDim] :=

  -- Compute mean along last dimension (dim = 1)
  let _ : Shape.WellFormed (.dim seqLen (.dim embedDim .scalar)) :=
  ⟨⟨h_seq_pos, ⟨h_embed_pos, trivial⟩⟩⟩

  let s := Shape.dim seqLen (Shape.dim embedDim Shape.scalar)
  let h_rank : Spec.Shape.rank s > 0 := by simp [s, Spec.Shape.rank]
  let h_valid : Shape.HasNonemptyAxis (Spec.Shape.rank s - 1) s :=
    Shape.inferNonemptyAxis (Nat.sub_lt h_rank Nat.zero_lt_one)

  let mean := reduceMean (Spec.Shape.rank s - 1) x h_valid.proof

  have h₁ : (Shape.dim seqLen (Shape.dim embedDim Shape.scalar)).rank = 2 := by
    simp [Spec.Shape.rank]

  let mean_broadcast := broadcastAfterSum s (Spec.Shape.rank s - 1) mean
  let centered := subSpec x mean_broadcast

  have inst : Shape.HasNonemptyAxis (Spec.Shape.rank (.dim seqLen (.dim embedDim .scalar)) - 1)
      (.dim seqLen (.dim embedDim .scalar)) := by
    apply Shape.inferNonemptyAxis
    simp [h₁]

  let varianceRaw := reduceVar (Spec.Shape.rank s - 1) centered inst.proof
  -- Clamp variance to be nonnegative so `std` is always defined/bounded away from 0 even for
  -- approximate numeric contexts (Float/NF) where small negative variance can occur.
  let variance := maxSpec varianceRaw (Tensor.full (.dim seqLen .scalar) 0)

  let std := sqrtSpec (addSpec variance (Tensor.full (.dim seqLen .scalar) epsilon))

  let stdBroadcast := broadcastAfterSum s (Spec.Shape.rank s - 1) std
  let normalized := divSpec centered stdBroadcast

  have h5 : Shape.CanBroadcastTo (.dim embedDim .scalar) (.dim seqLen (.dim embedDim .scalar)) := by
    apply Shape.CanBroadcastTo.expand_dims
    apply Shape.CanBroadcastTo.dim_eq
    exact Shape.CanBroadcastTo.scalar

  let gammaBroadcast := broadcastTo h5 gamma
  let betaBroadcast := broadcastTo h5 beta
  let scaled := mulSpec normalized gammaBroadcast
  addSpec scaled betaBroadcast

/-- Backward/VJP for `layerNorm`, with named input, scale, and bias gradients. -/
def layerNormBackward
  {seqLen embedDim : Nat}
  (sequenceLengthPositive : seqLen > 0)
  (embeddingWidthPositive : embedDim > 0)
  (input : Tensor α [seqLen, embedDim])
  (scale : Tensor α [embedDim])
  (outputGradient : Tensor α [seqLen, embedDim])
  (epsilon : α := TorchLean.normalizationEpsilon) :
  NormalizationGradients α [seqLen, embedDim] [embedDim] :=

  -- Forward recomputation
  let _ : Shape.WellFormed (.dim seqLen (.dim embedDim .scalar)) :=
  ⟨⟨sequenceLengthPositive, ⟨embeddingWidthPositive, trivial⟩⟩⟩

  let s := Shape.dim seqLen (Shape.dim embedDim Shape.scalar)
  let h_rank : Spec.Shape.rank s > 0 := by simp [s, Spec.Shape.rank]
  let h_valid : Shape.HasNonemptyAxis (Spec.Shape.rank s - 1) s :=
    Shape.inferNonemptyAxis (Nat.sub_lt h_rank Nat.zero_lt_one)

  let mean := reduceMean (Spec.Shape.rank s - 1) input h_valid.proof

  have h₁ : (Shape.dim seqLen (Shape.dim embedDim Shape.scalar)).rank = 2 := by
    simp [Spec.Shape.rank]

  have h₂ : shapeAfterSum (Shape.dim seqLen (Shape.dim embedDim Shape.scalar)) 1
            = Shape.dim seqLen Shape.scalar := by
    simp

  have h3 : shapeAfterSum (Shape.dim seqLen (Shape.dim embedDim Shape.scalar)) ((Shape.dim seqLen
    (Shape.dim embedDim Shape.scalar)).rank - 1)
          = Shape.dim seqLen Shape.scalar := by
    rw [h₁]
    rw [h₂]

  let mean_broadcast := broadcastAfterSum s (Spec.Shape.rank s - 1) mean
  let centered := subSpec input mean_broadcast

  have inst : Shape.HasNonemptyAxis (Spec.Shape.rank (.dim seqLen (.dim embedDim .scalar)) - 1)
      (.dim seqLen (.dim embedDim .scalar)) := by
    apply Shape.inferNonemptyAxis
    simp [h₁]

  let varianceRaw := reduceVar (Spec.Shape.rank s - 1) centered inst.proof
  let variance := maxSpec varianceRaw (Tensor.full (.dim seqLen .scalar) 0)
  let std := sqrtSpec (addSpec variance (Tensor.full (.dim seqLen .scalar) epsilon))
  let invStd := divSpec (Tensor.full (.dim seqLen .scalar) 1) std

  let stdBroadcast := broadcastAfterSum s (Spec.Shape.rank s - 1) std
  let norm := divSpec centered stdBroadcast

  have h5 : Shape.CanBroadcastTo (.dim embedDim .scalar) (.dim seqLen (.dim embedDim .scalar)) := by
    apply Shape.CanBroadcastTo.expand_dims
    apply Shape.CanBroadcastTo.dim_eq
    exact Shape.CanBroadcastTo.scalar

  -- `gamma` and `beta` have shape `[embedDim]` and are shared across all `seqLen` positions, so
  -- their
  -- gradients sum over the sequence dimension (axis 0).
  let hSequenceAxis := Shape.hasNonemptyAxisZeroOfPos sequenceLengthPositive
  let biasGradient := reduceSum 0 outputGradient hSequenceAxis.proof
  let scaleGradient := reduceSum 0 (mulSpec outputGradient norm) hSequenceAxis.proof

  -- ∂L/∂x: standard LayerNorm VJP, using per-position statistics over the feature dimension.
  --
  -- Let `N = embedDim`, `xhat = norm`, and `dy = gradOutput`.
  -- With `dyGamma = dy ⊙ gamma`, the closed form is:
  --
  --   dx = invStd ⊙ ( dyGamma
  --                    - mean(dyGamma)
  --                    - xhat ⊙ mean(dyGamma ⊙ xhat) )
  --
  -- where the `mean` is taken over the last dimension (features) for each sequence position.
  let scaleBroadcast := broadcastTo h5 scale
  let invStdBroadcast := broadcastAfterSum s (Spec.Shape.rank s - 1) invStd
  let dyGamma := mulSpec outputGradient scaleBroadcast

  let sumDyGamma := reduceSum (Spec.Shape.rank s - 1) dyGamma inst.proof
  -- We interpret `embedDim` as the feature-count `N` in the closed-form LayerNorm VJP.
  --
  -- Note: this relies on the `Context`'s `NatCast α` behaving sensibly (in particular, that
  -- `(embedDim : α)` is nonzero when `embedDim > 0`). This holds for TorchLean's shipped backends
  -- (Float/ℝ/configured binary32), but for exotic saturating casts a specialized scalar interface
  -- may be
  -- preferable.
  let N : α := (embedDim : α)
  let meanDyGamma := divSpec sumDyGamma (Tensor.full (.dim seqLen .scalar) N)

  let sumDyGammaXhat :=
    reduceSum (Spec.Shape.rank s - 1) (mulSpec dyGamma norm) inst.proof
  let meanDyGammaXhat := divSpec sumDyGammaXhat (Tensor.full (.dim seqLen .scalar) N)

  let meanDyGammaBroadcast :=
    broadcastAfterSum s (Spec.Shape.rank s - 1) meanDyGamma
  let meanDyGammaXhatBroadcast :=
    broadcastAfterSum s (Spec.Shape.rank s - 1) meanDyGammaXhat

  let inputGradient :=
    mulSpec invStdBroadcast
      (subSpec (subSpec dyGamma meanDyGammaBroadcast) (mulSpec norm
        meanDyGammaXhatBroadcast))

  { inputGradient, scaleGradient, biasGradient }

/--
Forward-mode JVP for `layerNorm`.

For each sequence position, LayerNorm is the map
`y = gamma ⊙ xhat + beta` with `xhat = (x - mean(x)) / sqrt(var(x)+eps)`.
The input tangent is normalized by the standard closed form

`dxhat = invStd ⊙ (dx - mean(dx) - xhat ⊙ mean(dx ⊙ xhat))`,

and affine-parameter tangents contribute `xhat ⊙ dgamma + dbeta`. This is the forward-mode
counterpart of the closed-form VJP above and follows the same clamped-variance convention as the
forward pass.
-/
def layerNormJvp
  {seqLen embedDim : Nat} (h_seq_pos : seqLen > 0) (h_embed_pos : embedDim > 0)
  (x tangent : Tensor α [seqLen, embedDim])
  (gamma dgamma _beta dbeta : Tensor α [embedDim])
  (epsilon : α := TorchLean.normalizationEpsilon) :
  Tensor α [seqLen, embedDim] :=

  let _ : Shape.WellFormed (.dim seqLen (.dim embedDim .scalar)) :=
  ⟨⟨h_seq_pos, ⟨h_embed_pos, trivial⟩⟩⟩

  let s := Shape.dim seqLen (Shape.dim embedDim Shape.scalar)
  let h_rank : Spec.Shape.rank s > 0 := by simp [s, Spec.Shape.rank]
  let h_valid : Shape.HasNonemptyAxis (Spec.Shape.rank s - 1) s :=
    Shape.inferNonemptyAxis (Nat.sub_lt h_rank Nat.zero_lt_one)

  let mean := reduceMean (Spec.Shape.rank s - 1) x h_valid.proof

  have h₁ : (Shape.dim seqLen (Shape.dim embedDim Shape.scalar)).rank = 2 := by
    simp [Spec.Shape.rank]

  let mean_broadcast := broadcastAfterSum s (Spec.Shape.rank s - 1) mean
  let centered := subSpec x mean_broadcast

  have inst : Shape.HasNonemptyAxis (Spec.Shape.rank (.dim seqLen (.dim embedDim .scalar)) - 1)
      (.dim seqLen (.dim embedDim .scalar)) := by
    apply Shape.inferNonemptyAxis
    simp [h₁]

  let varianceRaw := reduceVar (Spec.Shape.rank s - 1) centered inst.proof
  let variance := maxSpec varianceRaw (Tensor.full (.dim seqLen .scalar) 0)
  let std := sqrtSpec (addSpec variance (Tensor.full (.dim seqLen .scalar) epsilon))
  let invStd := divSpec (Tensor.full (.dim seqLen .scalar) 1) std
  let invStdBroadcast := broadcastAfterSum s (Spec.Shape.rank s - 1) invStd
  let norm := mulSpec centered invStdBroadcast

  let sumTangent := reduceSum (Spec.Shape.rank s - 1) tangent inst.proof
  let N : α := (embedDim : α)
  let meanTangent := divSpec sumTangent (Tensor.full (.dim seqLen .scalar) N)

  let sumTangentNorm :=
    reduceSum (Spec.Shape.rank s - 1) (mulSpec tangent norm) inst.proof
  let meanTangentNorm := divSpec sumTangentNorm (Tensor.full (.dim seqLen .scalar) N)

  let meanTangentBroadcast := broadcastAfterSum s (Spec.Shape.rank s - 1) meanTangent
  let meanTangentNormBroadcast :=
    broadcastAfterSum s (Spec.Shape.rank s - 1) meanTangentNorm

  let dnorm :=
    mulSpec invStdBroadcast
      (subSpec (subSpec tangent meanTangentBroadcast)
        (mulSpec norm meanTangentNormBroadcast))

  have h5 : Shape.CanBroadcastTo (.dim embedDim .scalar) (.dim seqLen (.dim embedDim .scalar)) := by
    apply Shape.CanBroadcastTo.expand_dims
    apply Shape.CanBroadcastTo.dim_eq
    exact Shape.CanBroadcastTo.scalar

  let gammaBroadcast := broadcastTo h5 gamma
  let dgammaBroadcast := broadcastTo h5 dgamma
  let dbetaBroadcast := broadcastTo h5 dbeta
  addSpec (addSpec (mulSpec dnorm gammaBroadcast) (mulSpec norm dgammaBroadcast))
    dbetaBroadcast
/-! ## Group normalization -/

/--
Normalize each sample over groups of channels and every spatial position.

The spatial domain is an arbitrary `Shape`. Channels are split into `groups` contiguous groups;
each group is flattened together with the spatial axes, normalized, and then transformed by the
per-channel `gamma` and `beta` parameters.
-/
def groupNorm
    {batch channels groups : Nat} {spatial : Shape}
    (x : Tensor α (Shape.concat [batch, channels] spatial))
    (gamma beta : Tensor α [channels])
    (hGroups : groups > 0 := by norm_num)
    (hGroupsLe : channels ≥ groups)
    (hDiv : channels % groups = 0)
    (epsilon : α := TorchLean.normalizationEpsilon)
    [Shape.WellFormed (Shape.concat [batch, channels] spatial)] :
    Tensor α (Shape.concat [batch, channels] spatial) :=
  let channelsPerGroup := channels / groups
  let spatialSize := Shape.size spatial
  let groupSize := channelsPerGroup * spatialSize
  let inputShape : Shape := Shape.concat [batch, channels] spatial
  let groupedShape : Shape := [batch, groups, groupSize]
  let flatShape : Shape := [batch, channels, spatialSize]
  have hInput := Shape.WellFormed.proof (s := inputShape)
  have hBatch : 0 < batch := hInput.1
  have hChannels : 0 < channels := hInput.2.1
  have hSpatial : 0 < spatialSize := by
    simpa [spatialSize] using Shape.size_pos_of_well_formed hInput.2.2
  have hChannelsPerGroup : 0 < channelsPerGroup :=
    Nat.div_pos hGroupsLe hGroups
  have hGroupSize : 0 < groupSize :=
    Nat.mul_pos hChannelsPerGroup hSpatial
  have hChannelsEq : channels = groups * channelsPerGroup := by
    simpa [channelsPerGroup, hDiv] using (Nat.mod_add_div channels groups).symm
  letI : Shape.WellFormed groupedShape :=
    ⟨⟨hBatch, ⟨hGroups, ⟨hGroupSize, trivial⟩⟩⟩⟩
  letI : Shape.WellFormed flatShape :=
    ⟨⟨hBatch, ⟨hChannels, ⟨hSpatial, trivial⟩⟩⟩⟩
  have hGroupedSize : Shape.size inputShape = Shape.size groupedShape := by
    simp only [inputShape, groupedShape, Shape.size]
    rw [hChannelsEq]
    simp [groupSize, spatialSize, Nat.mul_assoc]
  have hFlatSize : Shape.size inputShape = Shape.size flatShape := by
    simp [inputShape, flatShape, spatialSize, Shape.size]
  let grouped : Tensor α groupedShape := reshapeSpec x hGroupedSize
  let axis := Shape.rank groupedShape - 1
  let hAxis : Shape.HasNonemptyAxis axis groupedShape :=
    Shape.inferNonemptyAxis (by simp [axis, groupedShape, Shape.rank])
  let mean := reduceMean axis grouped hAxis.proof
  let meanBroadcast := broadcastAfterSum groupedShape axis mean
  let centered := subSpec grouped meanBroadcast
  let variance := reduceMean axis (mulSpec centered centered) hAxis.proof
  let variance := maxSpec variance (Tensor.full (shapeAfterSum groupedShape axis) 0)
  let denominator :=
    sqrtSpec (addSpec variance (Tensor.full (shapeAfterSum groupedShape axis) epsilon))
  let normalized :=
    divSpec centered (broadcastAfterSum groupedShape axis denominator)
  let normalizedInput : Tensor α inputShape :=
    reshapeSpec normalized hGroupedSize.symm
  let normalizedFlat : Tensor α flatShape :=
    reshapeSpec normalizedInput hFlatSize
  let channelSpatialShape : Shape := [channels, spatialSize]
  letI : Shape.WellFormed channelSpatialShape :=
    ⟨⟨hChannels, ⟨hSpatial, trivial⟩⟩⟩
  let gammaSpatial : Tensor α channelSpatialShape :=
    broadcastAfterSum channelSpatialShape 1 gamma
  let betaSpatial : Tensor α channelSpatialShape :=
    broadcastAfterSum channelSpatialShape 1 beta
  let gammaFlat : Tensor α flatShape :=
    broadcastAfterSum flatShape 0 gammaSpatial
  let betaFlat : Tensor α flatShape :=
    broadcastAfterSum flatShape 0 betaSpatial
  let outputFlat := addSpec (mulSpec normalizedFlat gammaFlat) betaFlat
  reshapeSpec outputFlat hFlatSize.symm

/-
  Normalize along a specific dimension
-/
/--
Normalize along a chosen axis `dim` of a tensor `x`, using per-element affine parameters `gamma`
and `beta` of the same shape as `x`.

This is a "generic building block" that is handy in specs; it is closer to the raw math than to a
single PyTorch module. Most named normalizations (LayerNorm, GroupNorm, BatchNorm) are special
cases of this pattern with a specific choice of axis set and parameter shape.
-/
def normalizeAlongDim
  {s : Shape}
  (x : Tensor α s)
  (gamma : Tensor α s)
  (beta : Tensor α s)
  (dim : Nat)
  (h_valid : Shape.HasNonemptyAxis dim s)
  (_h_wf : Shape.WellFormed s)
  (epsilon : α := TorchLean.normalizationEpsilon)
  : Tensor α s :=

  -- mean shape: shape_after_sum s dimension
  let mean := reduceMean dim x h_valid.proof

  let mean_broadcast := broadcastAfterSum s dim mean
  -- center x by subtracting mean (broadcasted)
  let centered := subSpec x mean_broadcast

  -- variance shape: shape_after_sum s dimension (same shape as mean)
  let variance := reduceVar dim centered h_valid.proof

  -- broadcast variance to s for addition of epsilon and sqrt
  let varianceBroadcast := broadcastAfterSum s dim variance

  -- compute std = sqrt(variance + epsilon)
  let std := sqrtSpec (addSpec varianceBroadcast (Tensor.full s epsilon))
  -- normalize centered by dividing by std (broadcasted)
  let normalized := divSpec centered std
  -- multiply by gamma (shape s) and add beta (shape s)
  let result := addSpec (mulSpec normalized gamma) beta
  result

/-
  RMS Normalization
  Normalizes using RMS instead of mean/variance
-/
/--
RMSNorm over the last dimension of a `(seqLen, embedDim)` tensor.

Compared to LayerNorm, RMSNorm skips subtracting the mean and normalizes by:

`rms(x) = sqrt(mean(x^2) + eps)`.

This shows up in many Transformer-style models as a cheaper alternative to LayerNorm.
-/
def rmsNorm {seqLen embedDim : Nat}
  (x : Tensor α [seqLen, embedDim])
  (gamma : Tensor α [embedDim])
  (h_seq_pos : seqLen > 0 := by norm_num)
  (h_embed_pos : embedDim > 0 := by norm_num)
  (epsilon : α := TorchLean.normalizationEpsilon) :
  Tensor α [seqLen, embedDim] :=
  -- Compute RMS along last dimension
  let squared := squareSpec x

  -- Proofs
  let _ : Shape.WellFormed (.dim seqLen (.dim embedDim .scalar)) :=
  ⟨⟨h_seq_pos, ⟨h_embed_pos, trivial⟩⟩⟩
  let s := Shape.dim seqLen (Shape.dim embedDim Shape.scalar)
  let h_rank : Spec.Shape.rank s > 0 := by simp [s, Spec.Shape.rank]
  let h_valid : Shape.HasNonemptyAxis (Spec.Shape.rank s - 1) s :=
    Shape.inferNonemptyAxis (Nat.sub_lt h_rank Nat.zero_lt_one)

  -- Compute mean along last dimension (dim = 1)
  let meanSquared := reduceMean (Spec.Shape.rank s - 1) squared h_valid.proof
  let rms := sqrtSpec (addSpec meanSquared (Tensor.full (.dim seqLen .scalar) epsilon))
  -- shape: [seqLen]

  -- Normalize by RMS
  let rmsBroadcast := broadcastAfterSum s (Spec.Shape.rank s - 1) rms
  let normalized := divSpec x rmsBroadcast

  have h_gamma_broadcast : Shape.CanBroadcastTo (Shape.dim embedDim Shape.scalar) (Shape.dim seqLen
    (.dim embedDim .scalar)) := by
    apply Shape.CanBroadcastTo.expand_dims
    apply Shape.CanBroadcastTo.dim_eq
    exact Shape.CanBroadcastTo.scalar

  -- Scale
  let gammaBroadcast := broadcastTo h_gamma_broadcast gamma
  let result := mulSpec normalized gammaBroadcast
  result

/-
  Weight Normalization
  Normalizes the weight matrix
-/
/--
WeightNorm for a dense weight matrix `(outDim, inDim)`.

This implements the "normalize weight vectors then scale" idea:

- normalize each output row by its L2 norm,
- then rescale by `gamma` (one scalar per output row).

PyTorch analogy: weight normalization is typically applied as a parametrization of a module's
weights rather than as a standalone tensor operator.
-/
def weightNorm {inDim outDim : Nat}
  (weight : Tensor α [outDim, inDim])
  (gamma : Tensor α [outDim])
  (h_out_pos : outDim > 0 := by norm_num)
  (h_in_pos : inDim > 0 := by norm_num)
  (epsilon : α := TorchLean.normalizationEpsilon) :
  Tensor α [outDim, inDim] :=

  -- Compute L2 norm of each row
  let squared := squareSpec weight

  -- Register well-formedness and axis

  let s := Shape.dim outDim (Shape.dim inDim Shape.scalar)
  let _ : Shape.WellFormed s := ⟨⟨h_out_pos, ⟨h_in_pos, trivial⟩⟩⟩

  -- The selected innermost axis is statically known to be nonempty.
  let h_rank : Spec.Shape.rank s > 0 := by simp [s, Spec.Shape.rank]
  let hAxis : Shape.HasNonemptyAxis (Spec.Shape.rank s - 1) s :=
    Shape.inferNonemptyAxis (Nat.sub_lt h_rank Nat.zero_lt_one)

  -- Sum each row along its `inDim` axis.
  let rowSums := reduceSum (Spec.Shape.rank s - 1) squared hAxis.proof
  let rowNorms := sqrtSpec (addSpec rowSums (Tensor.full (.dim outDim .scalar) epsilon))
  -- shape: [outDim]

  -- Normalize weights
  let rowNormsBroadcast := broadcastAfterSum s (Spec.Shape.rank s - 1) rowNorms
  let normalized := divSpec weight rowNormsBroadcast

  -- Scale
  let gammaBroadcast := broadcastAfterSum s (Spec.Shape.rank s - 1) gamma
  let result := mulSpec normalized gammaBroadcast
  result

end Spec
