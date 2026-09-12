/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Normalization.Core

/-!
# Batch Normalization

Generic channel-first BatchNorm semantics together with its JVP and VJP. All spatial axes are
flattened only for the reduction, so the definitions apply uniformly at every tensor rank.
Training-time statistics and inference-time running statistics remain separate operations.
-/

@[expose] public section

open TorchLean

namespace Spec
open TorchLean TorchLean.Tensor

variable {α : Type} [TorchLean.Storage α] [Context α] [DecidableRel ((· > ·) : α → α → Prop)]

/-- Repeat a channel vector over every position of a spatial shape. -/
def broadcastChannel {channels : Nat} (sSpatial : Shape)
    (x : Tensor α [channels]) : Tensor α (Shape.concat [channels] sSpatial) :=
  let spatialSize := Spec.Shape.size sSpatial
  let sFlat : Shape := [channels, spatialSize]
  let expanded : Tensor α sFlat := broadcastAfterSum sFlat 1 x
  have hSize : Spec.Shape.size sFlat = Spec.Shape.size (Shape.concat [channels] sSpatial) := by
    simp [sFlat, spatialSize, Spec.Shape.size]
  reshapeSpec expanded hSize

namespace BatchNorm

/-- Differential of normalized, affine channel data after statistics have been computed. -/
def normalizedJvp
    {channels positions : Nat} (hPositions : 0 < positions)
    (tangent xHat : Tensor α [channels, positions])
    (invStd gamma dgamma dbeta : Tensor α [channels]) :
    Tensor α [channels, positions] :=
  let shape : Shape := [channels, positions]
  let hAxis : Shape.HasNonemptyAxis 1 shape :=
    ⟨Shape.NonemptyAxis.succ (Shape.hasNonemptyAxisZeroOfPos hPositions).proof⟩
  let meanTangent := reduceMean 1 tangent hAxis.proof
  let meanTangentXHat := reduceMean 1 (mulSpec tangent xHat) hAxis.proof
  let normalizedTangent :=
    mulSpec (broadcastAfterSum shape 1 invStd)
      (subSpec
        (subSpec tangent (broadcastAfterSum shape 1 meanTangent))
        (mulSpec xHat (broadcastAfterSum shape 1 meanTangentXHat)))
  addSpec
    (addSpec
      (mulSpec normalizedTangent (broadcastAfterSum shape 1 gamma))
      (mulSpec xHat (broadcastAfterSum shape 1 dgamma)))
    (broadcastAfterSum shape 1 dbeta)

/-- Reverse rule adjoint to `normalizedJvp`, including affine-parameter gradients. -/
def normalizedBackward
    {channels positions : Nat} (hPositions : 0 < positions)
    (gradOutput xHat : Tensor α [channels, positions])
    (invStd gamma : Tensor α [channels]) :
    Tensor α [channels, positions] ×
      Tensor α [channels] × Tensor α [channels] :=
  let shape : Shape := [channels, positions]
  let hAxis : Shape.HasNonemptyAxis 1 shape :=
    ⟨Shape.NonemptyAxis.succ (Shape.hasNonemptyAxisZeroOfPos hPositions).proof⟩
  let gradGammaScaled := mulSpec gradOutput (broadcastAfterSum shape 1 gamma)
  let meanGradGamma := reduceMean 1 gradGammaScaled hAxis.proof
  let meanGradGammaXHat := reduceMean 1 (mulSpec gradGammaScaled xHat) hAxis.proof
  let gradInput :=
    mulSpec (broadcastAfterSum shape 1 invStd)
      (subSpec
        (subSpec gradGammaScaled (broadcastAfterSum shape 1 meanGradGamma))
        (mulSpec xHat (broadcastAfterSum shape 1 meanGradGammaXHat)))
  let gradGamma := reduceSum 1 (mulSpec gradOutput xHat) hAxis.proof
  let gradBeta := reduceSum 1 gradOutput hAxis.proof
  (gradInput, gradGamma, gradBeta)

end BatchNorm

/-
  Batch Normalization (spec layer)

  TorchLean models *pure* (stateless) BatchNorm operators:

  - `batchNorm`: "training-mode" normalization using statistics computed from the current input
    (TorchLean does not model the running-statistics update),
  - `batchNormInference`: inference-time normalization using fixed running mean/variance.
-/

/--
Stateless BatchNorm for channel-first tensors with shape `[channels] ++ sSpatial`.

This computes per-channel mean/variance over the `sSpatial` axes and applies:

`y = ((x - mean) / sqrt(var + eps)) * gamma + beta`.

PyTorch analogy: `torch.nn.BatchNorm{1,2,3}d` in training mode on an input with batch size `N=1`.
TorchLean does **not** model the running-statistics update here. A singleton spatial domain is
accepted and has zero variance; PyTorch training BatchNorm requires more than one value per channel.
The usual normalization interpretation assumes positive `epsilon`, which this raw spec does not
validate.
-/
def batchNorm
  {channels : Nat} {sSpatial : Shape}
  (x : Tensor α (Shape.concat [channels] sSpatial))
  (gamma : Tensor α [channels])
  (beta : Tensor α [channels])
  (epsilon : α := TorchLean.normalizationEpsilon)
  [Shape.WellFormed (Shape.concat [channels] sSpatial)] :
  Tensor α (Shape.concat [channels] sSpatial) :=
  let spatialSize : Nat := Spec.Shape.size sSpatial
  let s_flat : Shape := [channels, spatialSize]
  have h_reshape :
      Spec.Shape.size (Shape.concat [channels] sSpatial) = Spec.Shape.size s_flat := by
    simp [s_flat, spatialSize, Spec.Shape.size]
  let x2 : Tensor α s_flat := reshapeSpec x h_reshape
  have hwf_x : (Shape.concat [channels] sSpatial).wellFormed := Shape.WellFormed.proof
  have h_channels : channels > 0 := hwf_x.1
  have h_spatial_wf : sSpatial.wellFormed := hwf_x.2
  have h_spatialSize : spatialSize > 0 := by
    simpa [spatialSize] using Shape.size_pos_of_well_formed (s := sSpatial) h_spatial_wf
  letI : Shape.WellFormed s_flat := ⟨⟨h_channels, ⟨h_spatialSize, trivial⟩⟩⟩
  let h_rank : Spec.Shape.rank s_flat > 0 := by simp [s_flat, Spec.Shape.rank]
  let h_valid : Shape.HasNonemptyAxis (Spec.Shape.rank s_flat - 1) s_flat :=
    Shape.inferNonemptyAxis (Nat.sub_lt h_rank Nat.zero_lt_one)
  let mean : Tensor α [channels] := reduceMean (Shape.rank s_flat - 1) x2 h_valid.proof
  let centered := subSpec x2 (broadcastAfterSum s_flat 1 mean)
  let centered_sq := mulSpec centered centered
  let varianceRaw : Tensor α [channels] :=
    reduceMean (Shape.rank s_flat - 1) centered_sq h_valid.proof
  let variance := maxSpec varianceRaw (Tensor.full ([channels]) 0)
  let meanB := broadcastChannel sSpatial mean
  let varianceB := broadcastChannel sSpatial variance
  let gammaB := broadcastChannel sSpatial gamma
  let betaB := broadcastChannel sSpatial beta
  let centered := subSpec x meanB
  let std := sqrtSpec (addSpec varianceB (Tensor.full (Shape.concat [channels] sSpatial) epsilon))
  addSpec (mulSpec (divSpec centered std) gammaB) betaB

/--
Forward-mode JVP for `batchNorm`.

Stateless BatchNorm computes one set of statistics per channel over every spatial position. The
input tangent therefore uses the same closed-form normalization differential as LayerNorm, with
the mean taken over the flattened spatial shape for each channel:

`dxhat = inv_std * (dx - mean(dx) - xhat * mean(dx*xhat))`.

Affine tangents contribute `xhat * dgamma + dbeta` channel-wise.
-/
def batchNormJvp
  {channels : Nat} {sSpatial : Shape}
  (x tangent : Tensor α (Shape.concat [channels] sSpatial))
  (gamma dgamma _beta dbeta : Tensor α [channels])
  (epsilon : α := TorchLean.normalizationEpsilon)
  [Shape.WellFormed (Shape.concat [channels] sSpatial)] :
  Tensor α (Shape.concat [channels] sSpatial) :=
  let spatialSize := Shape.size sSpatial
  let sFlat : Shape := [channels, spatialSize]
  have hReshape : Shape.size (Shape.concat [channels] sSpatial) = Shape.size sFlat := by
    simp [sFlat, spatialSize, Shape.size]
  let xFlat : Tensor α sFlat := reshapeSpec x hReshape
  let tangentFlat : Tensor α sFlat := reshapeSpec tangent hReshape
  have hWellFormed : (Shape.concat [channels] sSpatial).wellFormed := Shape.WellFormed.proof
  have hChannels : channels > 0 := hWellFormed.1
  have hSpatial : spatialSize > 0 := by
    simpa [spatialSize] using Shape.size_pos_of_well_formed hWellFormed.2
  letI : Shape.WellFormed sFlat := ⟨⟨hChannels, ⟨hSpatial, trivial⟩⟩⟩
  let hAxis : Shape.HasNonemptyAxis (Shape.rank sFlat - 1) sFlat :=
    Shape.inferNonemptyAxis (by simp [sFlat, Shape.rank])
  let mean : Tensor α [channels] := reduceMean (Shape.rank sFlat - 1) xFlat hAxis.proof
  let meanB := broadcastAfterSum sFlat 1 mean
  let centered := subSpec xFlat meanB
  let varianceRaw : Tensor α [channels] :=
    reduceMean (Shape.rank sFlat - 1) (mulSpec centered centered) hAxis.proof
  let variance := maxSpec varianceRaw (Tensor.full ([channels]) 0)
  let invStd := divSpec (Tensor.full ([channels]) 1)
    (sqrtSpec (addSpec variance (Tensor.full ([channels]) epsilon)))
  let invStdB := broadcastAfterSum sFlat 1 invStd
  let xHat := mulSpec centered invStdB
  let yFlat := BatchNorm.normalizedJvp hSpatial tangentFlat xHat invStd gamma dgamma dbeta
  reshapeSpec yFlat hReshape.symm

/--
Backward/VJP for `batchNorm`.

Statistics and affine-parameter gradients are reduced over every spatial axis, independently for
each channel.
-/
def batchNormBackward
  {channels : Nat} {sSpatial : Shape}
  (x : Tensor α (Shape.concat [channels] sSpatial))
  (gamma : Tensor α [channels])
  (gradOutput : Tensor α (Shape.concat [channels] sSpatial))
  (epsilon : α := TorchLean.normalizationEpsilon)
  [Shape.WellFormed (Shape.concat [channels] sSpatial)] :
  NormalizationGradients α (Shape.concat [channels] sSpatial) [channels] :=
  let spatialSize := Shape.size sSpatial
  let sFlat : Shape := [channels, spatialSize]
  have hReshape : Shape.size (Shape.concat [channels] sSpatial) = Shape.size sFlat := by
    simp [sFlat, spatialSize, Shape.size]
  let xFlat : Tensor α sFlat := reshapeSpec x hReshape
  let gradFlat : Tensor α sFlat := reshapeSpec gradOutput hReshape
  have hWellFormed : (Shape.concat [channels] sSpatial).wellFormed := Shape.WellFormed.proof
  have hChannels : channels > 0 := hWellFormed.1
  have hSpatial : spatialSize > 0 := by
    simpa [spatialSize] using Shape.size_pos_of_well_formed hWellFormed.2
  letI : Shape.WellFormed sFlat := ⟨⟨hChannels, ⟨hSpatial, trivial⟩⟩⟩
  let hAxis : Shape.HasNonemptyAxis (Shape.rank sFlat - 1) sFlat :=
    Shape.inferNonemptyAxis (by simp [sFlat, Shape.rank])
  let mean : Tensor α [channels] := reduceMean (Shape.rank sFlat - 1) xFlat hAxis.proof
  let centered := subSpec xFlat (broadcastAfterSum sFlat 1 mean)
  let varianceRaw : Tensor α [channels] :=
    reduceMean (Shape.rank sFlat - 1) (mulSpec centered centered) hAxis.proof
  let variance := maxSpec varianceRaw (Tensor.full ([channels]) 0)
  let invStd := divSpec (Tensor.full ([channels]) 1)
    (sqrtSpec (addSpec variance (Tensor.full ([channels]) epsilon)))
  let invStdB := broadcastAfterSum sFlat 1 invStd
  let xHat := mulSpec centered invStdB
  let (flatInputGradient, scaleGradient, biasGradient) :=
    BatchNorm.normalizedBackward hSpatial gradFlat xHat invStd gamma
  { inputGradient := reshapeSpec flatInputGradient hReshape.symm
    scaleGradient
    biasGradient }

/-!
## BatchNorm (inference-time, running statistics)

PyTorch distinction:

- *training*: normalize using batch statistics (and update running mean/variance);
- *inference*: normalize using the stored running mean/variance.

TorchLean keeps things pure and explicit: inference-time BatchNorm takes the running statistics
as arguments.
-/

/--
Inference-time BatchNorm for channel-first tensors with shape `[channels] ++ sSpatial`, using fixed
running statistics.

Formula (per channel `c`):

`y = ((x - μ) / sqrt(σ² + eps)) * γ + β`

This matches the standard evaluation-time behavior of `torch.nn.BatchNorm{1,2,3}d` (no
batch-statistics computation, no running-statistics update).

Over real arithmetic, `(μ, σ², γ, β)` are constants, so this is an **affine** map in `x`.
Floating-point evaluation still rounds the individual operations. See
`Proofs.Normalization.batchNorm_inference_eq_mul_add`.
-/
def batchNormInference
  {channels : Nat} {sSpatial : Shape}
  (x : Tensor α (Shape.concat [channels] sSpatial))
  (runningMean : Tensor α [channels])
  (runningVar : Tensor α [channels])
  (gamma : Tensor α [channels])
  (beta : Tensor α [channels])
  (epsilon : α := TorchLean.normalizationEpsilon) :
  Tensor α (Shape.concat [channels] sSpatial) :=
  -- Clamp the variance to stay nonnegative in approximate numeric backends.
  let runningVar := maxSpec runningVar (Tensor.full ([channels]) 0)
  let meanB := broadcastChannel sSpatial runningMean
  let varianceB := broadcastChannel sSpatial runningVar
  let gammaB := broadcastChannel sSpatial gamma
  let betaB := broadcastChannel sSpatial beta
  let centered := subSpec x meanB
  let std := sqrtSpec (addSpec varianceB (Tensor.full (Shape.concat [channels] sSpatial) epsilon))
  addSpec (mulSpec (divSpec centered std) gammaB) betaB

end Spec
