/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Model.Functional.Core

/-!
# Norm

Normalization programs built from `Ops`, so they can run eagerly or be recorded in a typed graph.
Spatial dimensions are represented by an arbitrary `Shape`; the same definitions cover vectors,
images, volumes, and higher-rank data.

BatchNorm has separate training and evaluation programs here. Training computes statistics from
the input, while evaluation receives the stored mean and variance as arguments. `Layers.batchNorm`
owns those running-statistics buffers and updates them through `Layer.updateBuffers`.
-/

@[expose] public section


namespace Runtime
namespace Autograd
namespace Model

open Spec TorchLean
open TorchLean TorchLean.Tensor

namespace Norm

namespace Internal

/-- Flatten an arbitrary spatial shape while preserving the batch and channel axes. -/
theorem reshapeBatchChannelFlatSize {batch channels : Nat} {spatial : Shape} :
    Shape.size (.dim batch (.dim channels spatial)) =
      Shape.size (.dim batch (.dim channels (.dim (Shape.size spatial) .scalar))) := by
  simp [Spec.Shape.size]

/-- Flatten an arbitrary spatial shape while preserving the channel axis. -/
theorem reshapeChannelFlatSize {channels : Nat} {spatial : Shape} :
    Shape.size (.dim channels spatial) =
      Shape.size (.dim channels (.dim (Shape.size spatial) .scalar)) := by
  simp [Spec.Shape.size]

/-- Repeat a channel vector over the batch and flattened spatial axes. -/
def broadcastChannelToBatchSpatial {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    (batch channels spatialSize : Nat)
    (x : RefTy (m := m) (α := α) (.dim channels .scalar)) :
    m (RefTy (m := m) (α := α)
      (.dim batch (.dim channels (.dim spatialSize .scalar)))) := do
  let acrossSpatial ← Runtime.Autograd.Torch.broadcastAfterSum
    (m := m) (α := α) (.dim channels (.dim spatialSize .scalar)) 1 x
  Runtime.Autograd.Torch.broadcastAfterSum
    (m := m) (α := α) (.dim batch (.dim channels (.dim spatialSize .scalar))) 0 acrossSpatial

end Internal

/--
Normalize each final-axis vector by its root mean square, then apply `gamma`.

For each leading index, we compute `meanSq = mean(x * x)` and divide its vector by
`sqrt(max(meanSq, 0) + ε)`. The scale `gamma` has shape `[width]` and is broadcast across the
leading axes. We square the entries of `x` directly, without first subtracting their mean.

`width` must be positive. If a leading axis is empty, the result is an empty tensor of the same
shape, so we return it before constructing the reduction.
-/
def rmsNorm {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {leading : Shape} {width : Nat} (hWidth : width > 0)
    (x : RefTy (m := m) (α := α) (leading.appendDim width))
    (gamma : RefTy (m := m) (α := α) (.dim width .scalar))
    (ε : α := TorchLean.normalizationEpsilon) :
    m (RefTy (m := m) (α := α) (leading.appendDim width)) := by
  by_cases hLeading : Shape.size leading = 0
  · exact const (m := m) (α := α) (s := leading.appendDim width)
      (Tensor.full (leading.appendDim width) (0 : α))
  · exact do
      let s := leading.appendDim width
      let hSize : 0 < Shape.size s := by
        simpa [s, Shape.size_appendDim] using
          Nat.mul_pos (Nat.pos_of_ne_zero hLeading) hWidth
      let _ : Shape.WellFormed s := ⟨Shape.wellFormed_of_size_pos hSize⟩
      let sq ← F.square (m := m) (α := α) (s := s) x
      let axis := Shape.rank s - 1
      let _ : Shape.HasNonemptyAxis axis s :=
        Shape.inferNonemptyAxis (by simp [axis, s])
      let meanSq ← reduceMean (m := m) (α := α) (s := s) axis sq
      let meanSqShape := shapeAfterSum s axis
      let zero ← const (m := m) (α := α) (s := meanSqShape) (Tensor.full meanSqShape (0 : α))
      let meanSqClamped ← max (m := m) (α := α) (s := meanSqShape) meanSq zero
      let epsT ← const (m := m) (α := α) (s := meanSqShape) (Tensor.full meanSqShape ε)
      let denom ← sqrt (m := m) (α := α) (s := meanSqShape)
        (← add (m := m) (α := α) (s := meanSqShape) meanSqClamped epsT)
      let invDenom ← inv (m := m) (α := α) (s := meanSqShape) denom
      let invDenomB ← Runtime.Autograd.Torch.broadcastAfterSum
        (m := m) (α := α) s axis invDenom
      let normalized ← mul (m := m) (α := α) (s := s) x invDenomB
      let gammaB ← broadcastTo (m := m) (α := α) (s₁ := .dim width .scalar) (s₂ := s)
        (by
          simpa [s, Shape.appendDim_eq_concat] using
            Shape.CanBroadcastTo.prependTarget leading (.dim width .scalar)) gamma
      mul (m := m) (α := α) (s := s) normalized gammaB

/--
Divide each final-axis vector by `sqrt(sum(x * x) + epsilon)`.

`epsilon` is a scalar tensor reference, shared by all vectors. It is added to the squared norm
before the square root, so its effect follows this formula even for vectors whose norm is very
small. An empty leading axis produces an empty output of the same shape.
-/
def l2Normalize {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {leading : Shape} {width : Nat} (hWidth : width > 0)
    (x : RefTy (m := m) (α := α) (leading.appendDim width))
    (epsilon : RefTy (m := m) (α := α) .scalar) :
    m (RefTy (m := m) (α := α) (leading.appendDim width)) := by
  by_cases hLeading : Shape.size leading = 0
  · exact const (m := m) (α := α) (s := leading.appendDim width)
      (Tensor.full (leading.appendDim width) (0 : α))
  · exact do
      let s := leading.appendDim width
      let hSize : 0 < Shape.size s := by
        simpa [s, Shape.size_appendDim] using
          Nat.mul_pos (Nat.pos_of_ne_zero hLeading) hWidth
      let _ : Shape.WellFormed s := ⟨Shape.wellFormed_of_size_pos hSize⟩
      let squared ← mul (m := m) (α := α) (s := s) x x
      let axis := Shape.rank s - 1
      let _ : Shape.HasNonemptyAxis axis s :=
        Shape.inferNonemptyAxis (by simp [axis, s])
      let normSquared ← reduceSum (m := m) (α := α) (s := s) axis squared
      let reducedShape := shapeAfterSum s axis
      let epsilonB ← broadcastTo (m := m) (α := α) (s₁ := .scalar) (s₂ := reducedShape)
        (Shape.CanBroadcastTo.scalarTo reducedShape) epsilon
      let denominator ← sqrt (m := m) (α := α) (s := reducedShape)
        (← add (m := m) (α := α) (s := reducedShape) normSquared epsilonB)
      let inverse ← inv (m := m) (α := α) (s := reducedShape) denominator
      let inverseB ← Runtime.Autograd.Torch.broadcastAfterSum
        (m := m) (α := α) s axis inverse
      mul (m := m) (α := α) (s := s) x inverseB

/--
Normalize each sample and channel using its own spatial mean and variance.

We flatten the spatial axes to a vector of length `spatial.size`, subtract that vector's mean, and
divide by `sqrt(max(mean((x - mean) * (x - mean)), 0) + ε)`. The batch and channel axes remain
separate throughout this calculation.

`gamma` and `beta` each have one entry per channel. They are broadcast across the batch and spatial
positions before restoring the original shape. All statistics come from the current input.
-/
def instanceNorm {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {batch channels : Nat} {spatial : Shape}
    (hWellFormed : (Shape.dim batch (Shape.dim channels spatial)).wellFormed)
    (x : RefTy (m := m) (α := α) (.dim batch (.dim channels spatial)))
    (gamma beta : RefTy (m := m) (α := α) (.dim channels .scalar))
    (ε : α := TorchLean.normalizationEpsilon) :
    m (RefTy (m := m) (α := α) (.dim batch (.dim channels spatial))) := do
  let spatialSize := Shape.size spatial
  let inputShape : Shape := .dim batch (.dim channels spatial)
  let flatShape : Shape := .dim batch (.dim channels (.dim spatialSize .scalar))
  let hBatch := hWellFormed.1
  let hChannels := hWellFormed.2.1
  let hSpatial := Shape.size_pos_of_well_formed hWellFormed.2.2
  let _ : Shape.WellFormed inputShape := ⟨hWellFormed⟩
  let _ : Shape.WellFormed flatShape := ⟨⟨hBatch, ⟨hChannels, ⟨hSpatial, trivial⟩⟩⟩⟩
  let xFlat ← reshape (m := m) (α := α) (s₁ := inputShape) (s₂ := flatShape) x
    (Internal.reshapeBatchChannelFlatSize (batch := batch) (channels := channels)
      (spatial := spatial))
  let axis := Shape.rank flatShape - 1
  let _ : Shape.HasNonemptyAxis axis flatShape :=
    Shape.inferNonemptyAxis (by simp [axis, flatShape, Shape.rank])
  let mean ← reduceMean (m := m) (α := α) (s := flatShape) axis xFlat
  let meanShape := shapeAfterSum flatShape axis
  let meanB ← Runtime.Autograd.Torch.broadcastAfterSum
    (m := m) (α := α) flatShape axis mean
  let centered ← sub (m := m) (α := α) (s := flatShape) xFlat meanB
  let sq ← F.square (m := m) (α := α) (s := flatShape) centered
  let var ← reduceMean (m := m) (α := α) (s := flatShape) axis sq
  let zero ← const (m := m) (α := α) (s := meanShape) (Tensor.full meanShape (0 : α))
  let varClamped ← max (m := m) (α := α) (s := meanShape) var zero
  let epsT ← const (m := m) (α := α) (s := meanShape) (Tensor.full meanShape ε)
  let denom ← sqrt (m := m) (α := α) (s := meanShape) (← add (m := m) (α := α) (s := meanShape)
    varClamped epsT)
  let invDenom ← inv (m := m) (α := α) (s := meanShape) denom
  let invDenomB ← Runtime.Autograd.Torch.broadcastAfterSum
    (m := m) (α := α) flatShape axis invDenom
  let normalized ← mul (m := m) (α := α) (s := flatShape) centered invDenomB
  let gammaB ← Internal.broadcastChannelToBatchSpatial
    (m := m) (α := α) batch channels spatialSize gamma
  let betaB ← Internal.broadcastChannelToBatchSpatial
    (m := m) (α := α) batch channels spatialSize beta
  let yFlat ← add (m := m) (α := α) (s := flatShape)
    (← mul (m := m) (α := α) (s := flatShape) normalized gammaB) betaB
  reshape (m := m) (α := α) (s₁ := flatShape) (s₂ := inputShape) yFlat
    (Internal.reshapeBatchChannelFlatSize (batch := batch) (channels := channels)
      (spatial := spatial)).symm

/--
Normalize equal, contiguous channel groups independently within each sample.

The input is viewed as `[batch, groups, channelsPerGroup * spatial.size]`. Reducing the last axis
therefore combines the channels and spatial positions belonging to one group. We subtract the
group mean and divide by `sqrt(max(groupVariance, 0) + ε)`, using the mean squared deviation for
`groupVariance`.

After normalization, we restore the channel axis and apply `gamma` and `beta`, which each have
one entry per channel. The shape hypotheses ensure that every group has the same positive size.
-/
def groupNorm {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {batch channels groups : Nat} {spatial : Shape}
    (hWellFormed : (Shape.dim batch (Shape.dim channels spatial)).wellFormed)
    (hGroups : groups > 0) (hGroupsLe : channels ≥ groups)
    (hDiv : channels % groups = 0)
    (x : RefTy (m := m) (α := α) (.dim batch (.dim channels spatial)))
    (gamma beta : RefTy (m := m) (α := α) (.dim channels .scalar))
    (ε : α := TorchLean.normalizationEpsilon) :
    m (RefTy (m := m) (α := α) (.dim batch (.dim channels spatial))) := do
  let channelsPerGroup := channels / groups
  let spatialSize := Shape.size spatial
  let groupSize := channelsPerGroup * spatialSize
  let inputShape : Shape := .dim batch (.dim channels spatial)
  let groupedShape : Shape := .dim batch (.dim groups (.dim groupSize .scalar))
  let hBatch := hWellFormed.1
  let hSpatial := Shape.size_pos_of_well_formed hWellFormed.2.2
  have hChannelsPerGroup : channelsPerGroup > 0 := by
    exact Nat.div_pos hGroupsLe hGroups
  have hGroupSize : groupSize > 0 := Nat.mul_pos hChannelsPerGroup hSpatial
  have hChannelsEq : channels = groups * channelsPerGroup := by
    simpa [channelsPerGroup, hDiv] using (Nat.mod_add_div channels groups).symm
  let _ : Shape.WellFormed inputShape := ⟨hWellFormed⟩
  let _ : Shape.WellFormed groupedShape :=
    ⟨⟨hBatch, ⟨hGroups, ⟨hGroupSize, trivial⟩⟩⟩⟩
  have hReshape : Shape.size inputShape = Shape.size groupedShape := by
    rw [show inputShape = .dim batch (.dim channels spatial) by rfl,
      show groupedShape = .dim batch (.dim groups (.dim groupSize .scalar)) by rfl]
    simp only [Shape.size]
    rw [hChannelsEq]
    simp [groupSize, spatialSize, Nat.mul_assoc]
  let xGrouped ← reshape (m := m) (α := α) (s₁ := inputShape) (s₂ := groupedShape) x hReshape
  let axis := Shape.rank groupedShape - 1
  let _ : Shape.HasNonemptyAxis axis groupedShape :=
    Shape.inferNonemptyAxis (by simp [axis, groupedShape, Shape.rank])
  let mean ← reduceMean (m := m) (α := α) (s := groupedShape) axis xGrouped
  let meanShape := shapeAfterSum groupedShape axis
  let meanB ← Runtime.Autograd.Torch.broadcastAfterSum
    (m := m) (α := α) groupedShape axis mean
  let centered ← sub (m := m) (α := α) (s := groupedShape) xGrouped meanB
  let sq ← F.square (m := m) (α := α) (s := groupedShape) centered
  let var ← reduceMean (m := m) (α := α) (s := groupedShape) axis sq
  let zeroVar ← const (m := m) (α := α) (s := meanShape) (Tensor.full meanShape (0 : α))
  let varClamped ← max (m := m) (α := α) (s := meanShape) var zeroVar
  let epsT ← const (m := m) (α := α) (s := meanShape) (Tensor.full meanShape ε)
  let denom ← sqrt (m := m) (α := α) (s := meanShape)
    (← add (m := m) (α := α) (s := meanShape) varClamped epsT)
  let invDenom ← inv (m := m) (α := α) (s := meanShape) denom
  let invDenomB ← Runtime.Autograd.Torch.broadcastAfterSum
    (m := m) (α := α) groupedShape axis invDenom
  let normalized ← mul (m := m) (α := α) (s := groupedShape) centered invDenomB
  let normalizedInput ← reshape (m := m) (α := α)
    (s₁ := groupedShape) (s₂ := inputShape) normalized hReshape.symm
  let flatShape : Shape := .dim batch (.dim channels (.dim spatialSize .scalar))
  let _ : Shape.WellFormed flatShape :=
    ⟨⟨hBatch, ⟨hWellFormed.2.1, ⟨hSpatial, trivial⟩⟩⟩⟩
  let yFlat ← reshape (m := m) (α := α) (s₁ := inputShape) (s₂ := flatShape)
    normalizedInput (Internal.reshapeBatchChannelFlatSize (batch := batch)
      (channels := channels) (spatial := spatial))
  let gammaB ← Internal.broadcastChannelToBatchSpatial
    (m := m) (α := α) batch channels spatialSize gamma
  let betaB ← Internal.broadcastChannelToBatchSpatial
    (m := m) (α := α) batch channels spatialSize beta
  let yFlat ← add (m := m) (α := α) (s := flatShape)
    (← mul (m := m) (α := α) (s := flatShape) yFlat gammaB) betaB
  reshape (m := m) (α := α) (s₁ := flatShape) (s₂ := inputShape) yFlat
    (Internal.reshapeBatchChannelFlatSize (batch := batch) (channels := channels)
      (spatial := spatial)).symm

/--
Normalize a batch and return `(output, mean, variance)`.

Statistics have shape `[channels]`. We first average over spatial positions and then over the
batch, so every entry of a channel contributes equally. The variance is the mean squared
deviation over `batch * spatial.size` entries, clamped below by zero.

The output uses `sqrt(variance + ε)`, followed by the per-channel scale `gamma` and bias `beta`.
The returned variance is the biased estimate used in this forward pass; a running-statistics
update can apply `Layers.unbiasedRunningVariance` before storing it.
-/
def batchNormTrainStats {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {batch channels : Nat} {spatial : Shape}
    (hWellFormed : (Shape.dim batch (Shape.dim channels spatial)).wellFormed)
    (x : RefTy (m := m) (α := α) (.dim batch (.dim channels spatial)))
    (gamma beta : RefTy (m := m) (α := α) (.dim channels .scalar))
    (ε : α := TorchLean.normalizationEpsilon) :
    m (RefTy (m := m) (α := α) (.dim batch (.dim channels spatial)) ×
       RefTy (m := m) (α := α) (.dim channels .scalar) ×
       RefTy (m := m) (α := α) (.dim channels .scalar)) := do
  let spatialSize := Shape.size spatial
  let inputShape : Shape := .dim batch (.dim channels spatial)
  let flatShape : Shape := .dim batch (.dim channels (.dim spatialSize .scalar))
  let hBatch := hWellFormed.1
  let hChannels := hWellFormed.2.1
  let hSpatial := Shape.size_pos_of_well_formed hWellFormed.2.2
  let _ : Shape.WellFormed inputShape := ⟨hWellFormed⟩
  let _ : Shape.WellFormed flatShape := ⟨⟨hBatch, ⟨hChannels, ⟨hSpatial, trivial⟩⟩⟩⟩
  let xFlat ← reshape (m := m) (α := α) (s₁ := inputShape) (s₂ := flatShape) x
    (Internal.reshapeBatchChannelFlatSize (batch := batch) (channels := channels)
      (spatial := spatial))
  let spatialAxis := Shape.rank flatShape - 1
  let _ : Shape.HasNonemptyAxis spatialAxis flatShape :=
    Shape.inferNonemptyAxis (by simp [spatialAxis, flatShape, Shape.rank])
  let meanSpatial ← reduceMean (m := m) (α := α) (s := flatShape) spatialAxis xFlat
  let batchChannelShape := shapeAfterSum flatShape spatialAxis
  have hBatchChannelShape : batchChannelShape = .dim batch (.dim channels .scalar) := by
    simp [batchChannelShape, flatShape, spatialAxis, Shape.rank, shapeAfterSum]
  let _ : Shape.WellFormed batchChannelShape := by
    simpa [hBatchChannelShape] using
      (show Shape.WellFormed (.dim batch (.dim channels .scalar)) from
        ⟨⟨hBatch, ⟨hChannels, trivial⟩⟩⟩)
  let batchAxis := 0
  let _ : Shape.HasNonemptyAxis batchAxis batchChannelShape := by
    simpa [hBatchChannelShape] using
      Shape.hasNonemptyAxisZeroOfPos (n := batch) (s := .dim channels .scalar) hBatch
  let mean ← reduceMean (m := m) (α := α) (s := batchChannelShape) batchAxis meanSpatial
  let meanB ← Internal.broadcastChannelToBatchSpatial
    (m := m) (α := α) batch channels spatialSize mean
  let centered ← sub (m := m) (α := α) (s := flatShape) xFlat meanB
  let sq ← F.square (m := m) (α := α) (s := flatShape) centered
  let varSpatial ← reduceMean (m := m) (α := α) (s := flatShape) spatialAxis sq
  let var ← reduceMean (m := m) (α := α) (s := batchChannelShape) batchAxis varSpatial
  let channelShape : Shape := .dim channels .scalar
  let zero ← const (m := m) (α := α) (s := channelShape)
    (Tensor.full channelShape (0 : α))
  let varClamped ← max (m := m) (α := α) (s := channelShape) var zero
  let epsT ← const (m := m) (α := α) (s := channelShape) (Tensor.full channelShape ε)
  let denom ← sqrt (m := m) (α := α) (s := channelShape)
    (← add (m := m) (α := α) (s := channelShape) varClamped epsT)
  let invDenom ← inv (m := m) (α := α) (s := channelShape) denom
  let invDenomB ← Internal.broadcastChannelToBatchSpatial
    (m := m) (α := α) batch channels spatialSize invDenom
  let normalized ← mul (m := m) (α := α) (s := flatShape) centered invDenomB
  let gammaB ← Internal.broadcastChannelToBatchSpatial
    (m := m) (α := α) batch channels spatialSize gamma
  let betaB ← Internal.broadcastChannelToBatchSpatial
    (m := m) (α := α) batch channels spatialSize beta
  let yFlat ← add (m := m) (α := α) (s := flatShape)
    (← mul (m := m) (α := α) (s := flatShape) normalized gammaB) betaB
  let y ← reshape (m := m) (α := α) (s₁ := flatShape) (s₂ := inputShape) yFlat
    (Internal.reshapeBatchChannelFlatSize (batch := batch) (channels := channels)
      (spatial := spatial)).symm
  pure (y, mean, varClamped)

/--
Batch normalization using the current batch and spatial statistics.

This returns the output of `batchNormTrainStats`. Use that function when the caller also needs
the per-channel mean and biased variance, for example to update running statistics.
-/
def batchNormTrain {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {batch channels : Nat} {spatial : Shape}
    (hWellFormed : (Shape.dim batch (Shape.dim channels spatial)).wellFormed)
    (x : RefTy (m := m) (α := α) (.dim batch (.dim channels spatial)))
    (gamma beta : RefTy (m := m) (α := α) (.dim channels .scalar))
    (ε : α := TorchLean.normalizationEpsilon) :
    m (RefTy (m := m) (α := α) (.dim batch (.dim channels spatial))) := do
  let (y, _mean, _var) ← batchNormTrainStats
    (α := α) (m := m) hWellFormed x gamma beta (ε := ε)
  pure y

/--
Normalize with supplied per-channel mean and variance.

For each channel, the output is `gamma * (x - mean) / sqrt(max(var, 0) + ε) + beta`, broadcast over
the batch and spatial axes. This program reads the supplied references and leaves buffer updates
to the caller. `Layers.batchNorm` uses it in evaluation mode with the stored running statistics.
-/
def batchNormEval {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {batch channels : Nat} {spatial : Shape}
    (hWellFormed : (Shape.dim batch (Shape.dim channels spatial)).wellFormed)
    (x : RefTy (m := m) (α := α) (.dim batch (.dim channels spatial)))
    (gamma beta mean var : RefTy (m := m) (α := α) (.dim channels .scalar))
    (ε : α := TorchLean.normalizationEpsilon) :
    m (RefTy (m := m) (α := α) (.dim batch (.dim channels spatial))) := do
  let spatialSize := Shape.size spatial
  let inputShape : Shape := .dim batch (.dim channels spatial)
  let flatShape : Shape := .dim batch (.dim channels (.dim spatialSize .scalar))
  let hSpatial := Shape.size_pos_of_well_formed hWellFormed.2.2
  let _ : Shape.WellFormed inputShape := ⟨hWellFormed⟩
  let _ : Shape.WellFormed flatShape :=
    ⟨⟨hWellFormed.1, ⟨hWellFormed.2.1, ⟨hSpatial, trivial⟩⟩⟩⟩
  let xFlat ← reshape (m := m) (α := α) (s₁ := inputShape) (s₂ := flatShape) x
    (Internal.reshapeBatchChannelFlatSize (batch := batch) (channels := channels)
      (spatial := spatial))
  let meanB ← Internal.broadcastChannelToBatchSpatial
    (m := m) (α := α) batch channels spatialSize mean
  let centered ← sub (m := m) (α := α) (s := flatShape) xFlat meanB
  let channelShape : Shape := .dim channels .scalar
  let zero ← const (m := m) (α := α) (s := channelShape)
    (Tensor.full channelShape (0 : α))
  let varClamped ← max (m := m) (α := α) (s := channelShape) var zero
  let epsT ← const (m := m) (α := α) (s := channelShape) (Tensor.full channelShape ε)
  let denom ← sqrt (m := m) (α := α) (s := channelShape)
    (← add (m := m) (α := α) (s := channelShape) varClamped epsT)
  let invDenom ← inv (m := m) (α := α) (s := channelShape) denom
  let invDenomB ← Internal.broadcastChannelToBatchSpatial
    (m := m) (α := α) batch channels spatialSize invDenom
  let normalized ← mul (m := m) (α := α) (s := flatShape) centered invDenomB
  let gammaB ← Internal.broadcastChannelToBatchSpatial
    (m := m) (α := α) batch channels spatialSize gamma
  let betaB ← Internal.broadcastChannelToBatchSpatial
    (m := m) (α := α) batch channels spatialSize beta
  let yFlat ← add (m := m) (α := α) (s := flatShape)
    (← mul (m := m) (α := α) (s := flatShape) normalized gammaB) betaB
  reshape (m := m) (α := α) (s₁ := flatShape) (s₂ := inputShape) yFlat
    (Internal.reshapeBatchChannelFlatSize (batch := batch) (channels := channels)
      (spatial := spatial)).symm

end Norm

end Model
end Autograd
end Runtime
