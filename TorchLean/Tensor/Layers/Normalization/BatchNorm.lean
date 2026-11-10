/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Layers.Normalization.LayerNorm

/-!
# Batch normalization derivatives

The JVP and VJP of channel-first batch normalization with batch statistics, `Tensor.batchNorm x 0
gamma beta .batch epsilon` for `x : Tensor α ([channels] ++ sSpatial)`. All spatial axes are
flattened only for the reduction, so the definitions apply at every tensor rank.
`Tensor.batchNorm` (`TorchLean.Tensor.Layers.Normalization`) is the forward operator, with
batch or running statistics along any channel axis.

Both rules are hand-derived. Over `ℝ` and with every extent positive (`Shape.WellFormed`),
`TorchLean.BatchNorm.fderiv_forwardVec_eq_jvp` (`TorchLean/NN/Norm/BatchNorm/FDeriv.lean`) proves
that `BatchNorm.jvp` is the derivative of the `.batch` forward when `epsilon` is positive, and
`TorchLean.BatchNorm.jvp_backward_adjoint` (`TorchLean/NN/Norm/BatchNorm.lean`) that
`BatchNorm.backward` is adjoint to it at any `epsilon`.
-/

@[expose] public section

open TorchLean.Shape (size)

namespace TorchLean
open TorchLean.Tensor
open Normalization (normalizedRows)

variable {α : Type} [Storage α] [Context α] [DecidableRel ((· > ·) : α → α → Prop)]

namespace BatchNorm

/-- Tangent of affine batch normalization on a `[channels, positions]` matrix, given the forward's
normalized input `xHat` and inverse standard deviation `invStd` per channel. -/
def normalizedJVP
    {channels positions : Nat}
    (tangent xHat : Tensor α [channels, positions])
    (invStd gamma dgamma dbeta : Tensor α [channels]) :
    Tensor α [channels, positions] :=
  let shape : Shape := [channels, positions]
  let meanTangent := Tensor.map (Context.divNat · positions)
    (castShape (reduce .sum tangent [⟨1, Nat.one_lt_two⟩] false) (Shape.reduce_singleton _ _))
  let meanTangentXHat := Tensor.map (Context.divNat · positions)
    (castShape (reduce .sum (tangent * xHat) [⟨1, Nat.one_lt_two⟩] false)
        (Shape.reduce_singleton _ _))
  let normalizedTangent :=
    (Tensor.broadcast (s := shape.eraseIdx 1) invStd shape _
        (shape.eraseIdx_broadcastDims 1)) *
      ((tangent - Tensor.broadcast meanTangent shape _ (shape.eraseIdx_broadcastDims 1)) -
        (xHat * Tensor.broadcast meanTangentXHat shape _ (shape.eraseIdx_broadcastDims 1)))
  ((normalizedTangent * (Tensor.broadcast (s := shape.eraseIdx 1) gamma shape _
        (shape.eraseIdx_broadcastDims 1))) +
      (xHat * (Tensor.broadcast (s := shape.eraseIdx 1) dgamma shape _
        (shape.eraseIdx_broadcastDims 1)))) +
    (Tensor.broadcast (s := shape.eraseIdx 1) dbeta shape _
        (shape.eraseIdx_broadcastDims 1))

/-- Reverse rule adjoint to `normalizedJVP`, including affine-parameter gradients. -/
def normalizedBackward
    {channels positions : Nat}
    (gradOutput xHat : Tensor α [channels, positions])
    (invStd gamma : Tensor α [channels]) :
    Tensor α [channels, positions] ×
      Tensor α [channels] × Tensor α [channels] :=
  let shape : Shape := [channels, positions]
  let gradGammaScaled := gradOutput *
      (Tensor.broadcast (s := shape.eraseIdx 1) gamma shape _
        (shape.eraseIdx_broadcastDims 1))
  let meanGradGamma := Tensor.map (Context.divNat · positions)
    (castShape (reduce .sum gradGammaScaled [⟨1, Nat.one_lt_two⟩] false)
        (Shape.reduce_singleton _ _))
  let meanGradGammaXHat := Tensor.map (Context.divNat · positions)
    (castShape (reduce .sum (gradGammaScaled * xHat) [⟨1, Nat.one_lt_two⟩] false)
        (Shape.reduce_singleton _ _))
  let gradInput :=
    (Tensor.broadcast (s := shape.eraseIdx 1) invStd shape _
        (shape.eraseIdx_broadcastDims 1)) *
      ((gradGammaScaled -
            Tensor.broadcast meanGradGamma shape _ (shape.eraseIdx_broadcastDims 1)) -
        (xHat *
            Tensor.broadcast meanGradGammaXHat shape _ (shape.eraseIdx_broadcastDims 1)))
  let gradGamma := castShape (reduce .sum (gradOutput * xHat) [⟨1, Nat.one_lt_two⟩] false)
      (Shape.reduce_singleton _ _)
  let gradBeta := castShape (reduce .sum gradOutput [⟨1, Nat.one_lt_two⟩] false)
      (Shape.reduce_singleton _ _)
  (gradInput, gradGamma, gradBeta)

end BatchNorm

/--
Forward-mode JVP of `Tensor.batchNorm` along the leading channel axis with batch statistics.

With batch statistics each channel has one mean and variance over all its spatial positions, so
the input tangent goes through the closed-form normalization differential of LayerNorm, with each
mean over the flattened spatial shape of a channel:

`dxhat = inv_std * (dx - mean(dx) - xhat * mean(dx*xhat))`.

Affine tangents contribute `xhat * dgamma + dbeta` channel-wise. The normalized input and inverse
standard deviations are the forward's (`normalizedRows` with `recenter` false).
-/
def BatchNorm.jvp
  {channels : Nat} {sSpatial : Shape}
  (x tangent : Tensor α (channels :: sSpatial))
  (gamma dgamma _beta dbeta : Tensor α [channels])
  (epsilon : α := normalizationEpsilon) :
  Tensor α (channels :: sSpatial) :=
  let spatialSize := Shape.size sSpatial
  let sFlat : Shape := [channels, spatialSize]
  have hReshape : Shape.size (channels :: sSpatial) = Shape.size sFlat := by
    simp [sFlat, spatialSize]
  let xFlat : Tensor α sFlat := Tensor.reshape x _ hReshape
  let tangentFlat : Tensor α sFlat := Tensor.reshape tangent _ hReshape
  let normalized := normalizedRows (rows := channels) (cols := spatialSize) xFlat epsilon false
  let yFlat := BatchNorm.normalizedJVP tangentFlat normalized.1 normalized.2 gamma dgamma dbeta
  Tensor.reshape yFlat _ hReshape.symm

/--
Backward/VJP of `Tensor.batchNorm` along the leading channel axis with batch statistics.

Statistics and affine-parameter gradients are reduced over every spatial axis, independently for
each channel.
-/
def BatchNorm.backward
  {channels : Nat} {sSpatial : Shape}
  (x : Tensor α (channels :: sSpatial))
  (gamma : Tensor α [channels])
  (gradOutput : Tensor α (channels :: sSpatial))
  (epsilon : α := normalizationEpsilon) :
  Normalization.Gradients α (channels :: sSpatial) [channels] :=
  let spatialSize := Shape.size sSpatial
  let sFlat : Shape := [channels, spatialSize]
  have hReshape : Shape.size (channels :: sSpatial) = Shape.size sFlat := by
    simp [sFlat, spatialSize]
  let xFlat : Tensor α sFlat := Tensor.reshape x _ hReshape
  let gradFlat : Tensor α sFlat := Tensor.reshape gradOutput _ hReshape
  let normalized := normalizedRows (rows := channels) (cols := spatialSize) xFlat epsilon false
  let (flatInputGradient, scaleGradient, biasGradient) :=
    BatchNorm.normalizedBackward gradFlat normalized.1 normalized.2 gamma
  { inputGradient := Tensor.reshape flatInputGradient _ hReshape.symm
    scaleGradient
    biasGradient }

end TorchLean
