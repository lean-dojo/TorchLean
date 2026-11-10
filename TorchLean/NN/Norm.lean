/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Functional

/-!
# Norm

Normalization programs built from `Ops`, so they can run eagerly or be recorded in a typed graph.
Spatial dimensions are an arbitrary `Shape`, and the same definitions cover vectors, images,
volumes and higher-rank data.

`rms` and `group` divide each normalized vector by a detached scale before squaring it, as
`Tensor.rmsNorm` and `Tensor.groupNorm` do. Instance normalization is `group` with one group per
channel. `Layer.batchNorm` applies the registered operators, `Op.batchNorm` in training and
`Op.batchNormRunning` in evaluation.
-/

@[expose] public section


open TorchLean.Shape (rank size size_cons size_nil)

namespace TorchLean
namespace Norm

open TorchLean.Tensor Shape
open Ops

/-- Flatten an arbitrary spatial shape while preserving the batch and channel axes. -/
theorem reshape_flat_size {batch channels : Nat} {spatial : Shape} :
    Shape.size (Shape.dim batch (.dim channels spatial)) =
      Shape.size (Shape.dim batch (.dim channels (.dim (Shape.size spatial) .scalar))) := by
  simp

/-- Repeat a channel vector over the batch and flattened spatial axes. -/
def broadcast {α : Type} [Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    (batch channels spatialSize : Nat)
    (x : Ref (m := m) (α := α) (.dim channels .scalar)) :
    m (Ref (m := m) (α := α)
      (.dim batch (.dim channels (.dim spatialSize .scalar)))) := do
  let acrossSpatial ← Ops.apply (m := m) (α := α)
      (.broadcast (shapeAfterSumKeepDim_canBroadcastTo
        (.dim channels (.dim spatialSize .scalar)) 1).dims
        (shapeAfterSumKeepDim_canBroadcastTo (.dim channels (.dim spatialSize .scalar))
          1).broadcastDims)
      ((← Ops.apply (m := m) (α := α)
          (.reshape (t := shapeAfterSumKeepDim (.dim channels (.dim spatialSize .scalar)) 1)
            (size_shapeAfterSumKeepDim (.dim channels (.dim spatialSize .scalar)) 1).symm) (x, ())),
        ())
  Ops.apply (m := m) (α := α)
      (.broadcast
        (shapeAfterSumKeepDim_canBroadcastTo (.dim batch (.dim channels (.dim spatialSize .scalar)))
          0).dims (shapeAfterSumKeepDim_canBroadcastTo
          (.dim batch (.dim channels (.dim spatialSize .scalar))) 0).broadcastDims)
      ((← Ops.apply (m := m) (α := α)
          (.reshape
            (t := shapeAfterSumKeepDim (.dim batch (.dim channels (.dim spatialSize .scalar))) 0)
            (size_shapeAfterSumKeepDim (.dim batch (.dim channels (.dim spatialSize .scalar)))
              0).symm) (acrossSpatial, ())), ())

/-- Normalize final-axis vectors by their root mean square after scaling both the input and its
regularizer. The selected scale is detached because it cancels from the real function. -/
def normalize {α : Type} [Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {leading : Shape} {width : Nat} (hWidth : 0 < width)
    (x : Ref (m := m) (α := α) (leading ++ [width]))
    (epsilon : Ref (m := m) (α := α) .scalar) :
    m (Ref (m := m) (α := α) (leading ++ [width])) := by
  by_cases hLeading : Shape.size leading = 0
  · exact Ops.const (Tensor.zeros (leading ++ [width]))
  · exact do
      let s := leading ++ [width]
      let _ : Shape.WellFormed s := ⟨Shape.positive_of_size_pos (by
        simpa [s, Shape.size_append_singleton] using
          Nat.mul_pos (Nat.pos_of_ne_zero hLeading) hWidth)⟩
      let axis := s.rank - 1
      let _ : Shape.HasNonemptyAxis axis s :=
        Shape.hasNonemptyAxis_of_lt_rank (by simp [axis, s])
      let reduced := s.eraseIdx axis
      let magnitude ← Ops.apply .abs (x, ())
      let selected ← Ops.apply
        (.topKMask ⟨axis, Shape.AxisInBounds.proof⟩ 1) (magnitude, ())
      let largest ←
          (Ref.cast
            (Shape.reduce_singleton _ ⟨axis, Shape.AxisInBounds.proof⟩) <$>
            Ops.apply
              (.reduce .sum [⟨axis, Shape.AxisInBounds.proof⟩] false
                (fun h => by rcases h with h | h <;> cases h))
              ((← Ops.apply .mul (magnitude, selected, ())), ()))
      let epsilon ← Ops.apply
          (.broadcast (Shape.scalar_canBroadcastTo reduced).dims
            (Shape.scalar_canBroadcastTo reduced).broadcastDims) (epsilon, ())
      let epsilonScale ← Ops.apply (.sqrt .clampNegative)
          ((← Ops.apply .abs (epsilon, ())), ())
      let scale ← Ops.apply .detach
          ((← Ops.apply .max (largest, epsilonScale, ())), ())
      let expanded ← Ops.apply
          (.broadcast (shapeAfterSumKeepDim_canBroadcastTo s axis).dims
            (shapeAfterSumKeepDim_canBroadcastTo s axis).broadcastDims)
          ((← Ops.apply
              (.reshape (t := shapeAfterSumKeepDim s axis) (size_shapeAfterSumKeepDim s axis).symm)
              (scale, ())), ())
      let scaled ← Ops.apply .div (x, expanded, ())
      let squares ← Ops.apply .mul (scaled, scaled, ())
      let meanSquare ←
          (Ref.cast
            (Shape.reduce_singleton _ ⟨axis, Shape.AxisInBounds.proof⟩) <$>
            Ops.apply (.mean [⟨axis, Shape.AxisInBounds.proof⟩] false) (squares, ()))
      let regularizer ← Ops.apply .div
          ((← Ops.apply .div (epsilon, scale, ())), scale, ())
      let denominator ← Ops.apply (.sqrt .clampNegative)
          ((← Ops.apply .add (meanSquare, regularizer, ())), ())
      let denominator ← Ops.apply
          (.broadcast (shapeAfterSumKeepDim_canBroadcastTo s axis).dims
            (shapeAfterSumKeepDim_canBroadcastTo s axis).broadcastDims)
          ((← Ops.apply
              (.reshape (t := shapeAfterSumKeepDim s axis) (size_shapeAfterSumKeepDim s axis).symm)
              (denominator, ())), ())
      Ops.apply .div (scaled, denominator, ())

/--
Normalize each final-axis vector by its root mean square, then apply `gamma`.

For each leading index, the final-axis vector is divided by `sqrt(meanSq + ε)`, where
`meanSq = mean(x * x)`. Before squaring, the input and epsilon are scaled together to keep large or
small finite squares in range. `gamma` has shape `[width]` and is broadcast across leading axes. The
reduction and its gradients are subject to the chosen scalar's finite range, and each operation
rounds separately. For nonfinite inputs the NaN pattern can differ from that of the unscaled
formula.

`width` must be positive. If a leading axis is empty, the result is an empty tensor of the same
shape, returned before the reduction is constructed.
-/
def rms {α : Type} [Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {leading : Shape} {width : Nat} (hWidth : width > 0)
    (x : Ref (m := m) (α := α) (leading ++ [width]))
    (gamma : Ref (m := m) (α := α) (.dim width .scalar))
    (ε : α := normalizationEpsilon) :
    m (Ref (m := m) (α := α) (leading ++ [width])) := by
  by_cases hLeading : Shape.size leading = 0
  · exact Ops.const (m := m) (α := α) (s := leading ++ [width])
      (Tensor.full (leading ++ [width]) (0 : α))
  · exact do
      let s := leading ++ [width]
      let epsT ← Ops.const (Tensor.scalar ε)
      let normalized ← normalize hWidth x epsT
      let cb : Shape.CanBroadcastTo (.dim width .scalar) s := by
        simpa [s] using
          Shape.canBroadcastTo_concat_right leading (.dim width .scalar)
      let gammaB ← Ops.apply (m := m) (α := α)
        (.broadcast cb.dims cb.broadcastDims) (gamma, ())
      Ops.apply (m := m) (α := α) (.mul (s := s)) (normalized, gammaB, ())

/--
Normalize equal, contiguous channel groups independently within each sample.

The input is viewed as `[batch, groups, channelsPerGroup * spatial.size]`, so reducing the last
axis combines the channels and spatial positions of one group. Each group is centered by its mean
and then normalized as `rms` normalizes a vector: divided by the detached scale
`max(largest magnitude, sqrt |ε|)`, then by `sqrt(meanSq + ε / scale / scale)`, the steps of
`Tensor.groupNorm` along axis 1. Instance normalization is the case `groups = channels`.

After normalization the channel axis is restored and `gamma` and `beta`, which each have one entry
per channel, are applied. The shape hypotheses ensure that every group has the same positive size.
-/
def group {α : Type} [Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {batch channels groups : Nat} {spatial : Shape}
    (hWellFormed : (Shape.dim batch (Shape.dim channels spatial)).Positive)
    (hGroups : groups > 0) (hGroupsLe : channels ≥ groups)
    (hDiv : channels % groups = 0)
    (x : Ref (m := m) (α := α) (.dim batch (.dim channels spatial)))
    (gamma beta : Ref (m := m) (α := α) (.dim channels .scalar))
    (ε : α := normalizationEpsilon) :
    m (Ref (m := m) (α := α) (.dim batch (.dim channels spatial))) := do
  let channelsPerGroup := channels / groups
  let spatialSize := Shape.size spatial
  let groupSize := channelsPerGroup * spatialSize
  let inputShape : Shape := .dim batch (.dim channels spatial)
  let groupedShape : Shape := .dim batch (.dim groups (.dim groupSize .scalar))
  let hBatch := hWellFormed.1
  let hSpatial := Shape.size_pos_of_positive hWellFormed.2.2
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
    simp only [Shape.size_nil, Shape.size_cons]
    rw [hChannelsEq]
    simp [groupSize, spatialSize, Nat.mul_assoc]
  let xGrouped ← Ops.apply (m := m) (α := α)
      (.reshape (s := inputShape) (t := groupedShape) hReshape) (x, ())
  let axis := Shape.rank groupedShape - 1
  let _ : Shape.HasNonemptyAxis axis groupedShape :=
    Shape.hasNonemptyAxis_of_lt_rank (by simp [axis, groupedShape])
  let mean ←
      (Ref.cast
        (Shape.reduce_singleton groupedShape ⟨axis, Shape.AxisInBounds.proof⟩) <$>
        Ops.apply (m := m) (α := α)
          (.mean (s := groupedShape) [⟨axis, Shape.AxisInBounds.proof⟩] false) (xGrouped, ()))
  let meanB ← Ops.apply (m := m) (α := α)
      (.broadcast (shapeAfterSumKeepDim_canBroadcastTo groupedShape axis).dims
        (shapeAfterSumKeepDim_canBroadcastTo groupedShape axis).broadcastDims)
      ((← Ops.apply (m := m) (α := α)
          (.reshape (t := shapeAfterSumKeepDim groupedShape axis)
            (size_shapeAfterSumKeepDim groupedShape axis).symm) (mean, ())), ())
  let centered ← Ops.apply (m := m) (α := α) (.sub (s := groupedShape))
      (xGrouped, meanB, ())
  let epsT ← Ops.const (m := m) (α := α) (Tensor.scalar ε)
  let normalized ← normalize (m := m) (α := α)
      (leading := .dim batch (.dim groups .scalar)) hGroupSize centered epsT
  let normalizedInput ← Ops.apply (m := m) (α := α)
      (.reshape (s := groupedShape) (t := inputShape) hReshape.symm) (normalized, ())
  let flatShape : Shape := .dim batch (.dim channels (.dim spatialSize .scalar))
  let _ : Shape.WellFormed flatShape :=
    ⟨⟨hBatch, ⟨hWellFormed.2.1, ⟨hSpatial, trivial⟩⟩⟩⟩
  let yFlat ← Ops.apply (m := m) (α := α)
      (.reshape (s := inputShape) (t := flatShape)
        (reshape_flat_size (batch := batch)
      (channels := channels) (spatial := spatial))) (normalizedInput, ())
  let gammaB ← broadcast
    (m := m) (α := α) batch channels spatialSize gamma
  let betaB ← broadcast
    (m := m) (α := α) batch channels spatialSize beta
  let yFlat ← Ops.apply (m := m) (α := α) (.add (s := flatShape))
      ((← Ops.apply (m := m) (α := α) (.mul (s := flatShape))
          (yFlat, gammaB, ())), betaB, ())
  Ops.apply (m := m) (α := α)
      (.reshape (s := flatShape) (t := inputShape)
        (reshape_flat_size (batch := batch) (channels := channels)
      (spatial := spatial)).symm) (yFlat, ())

end Norm
end TorchLean
