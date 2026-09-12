/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Model.Layers.Core
public import NN.Runtime.Autograd.Model.Norm
public import NN.Spec.Core.Context.Constants

/-!
# Normalization Layers

These layers package normalization programs with their learned parameters and, for BatchNorm,
their running statistics. LayerNorm and RMSNorm operate on the final axis. BatchNorm, InstanceNorm,
and GroupNorm take inputs of shape `[batch, channels, ...spatial]`, where `spatial` can describe a
sequence, image, volume, or any other collection of spatial axes.

Each constructor accepts a positive `eps`, added under the square root. We store it as a rational
and convert it with `Context.ofRat` when the forward program chooses its scalar type. This keeps
the layer configuration usable with both real-valued specifications and floating-point execution.
Floating-point execution requires the converted epsilon to remain positive and finite; the static
configuration check only enforces positivity of the rational input.
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace Model

open Spec TorchLean
open TorchLean TorchLean.Tensor
open Proofs.Autograd.Algebra

namespace Layers

namespace Internal

/--
Require a positive epsilon before constructing runtime state.

A constant input has zero variance. Adding `eps` under the square root gives that input a positive
denominator in the real-valued normalization formula.
-/
def validateEpsilon (kind : String) (eps : Rat) : Except String Unit :=
  if eps > 0 then pure () else .error s!"{kind}: epsilon must be positive"

/--
Build a layer around a normalization program that expects scale, bias, and input references.

With `affine := true`, the scale starts at one and the optional bias starts at zero. Setting
`bias := false` leaves only the scale in model state; setting `affine := false` leaves neither
parameter. The forward program supplies constant ones and zeros for the omitted arguments, so
the same normalization program can run eagerly or be recorded in a typed graph.
-/
def affineNormalization (kind : String) (shape parameterShape : Shape)
    (affine bias : Bool) (validateConfig : Except String Unit)
    (normalize : ∀ {α : Type}, [Storage α] → [Context α] →
      Program α [parameterShape, parameterShape, shape] shape) : Layer shape shape :=
  if affine then
    if bias then
      { kind
        stateShapes := [parameterShape, parameterShape]
        initState := .cons (Tensor.ones parameterShape) (.cons (Tensor.zeros parameterShape) .nil)
        runtimeInit := some (.cons .ones (.cons .zeros .nil))
        validateConfig
        forward := fun _ => normalize }
    else
      { kind
        stateShapes := [parameterShape]
        initState := .cons (Tensor.ones parameterShape) .nil
        runtimeInit := some (.cons .ones .nil)
        validateConfig
        forward := fun _ {α} _ _ {m} _ _ weight input =>
          show m (RefTy (m := m) (α := α) shape) from do
            let bias ← const (m := m) (α := α) (Tensor.zeros parameterShape)
            normalize (α := α) (m := m) weight bias input }
  else
    { kind
      stateShapes := []
      initState := .nil
      validateConfig
      forward := fun _ {α} _ _ {m} _ _ input =>
        show m (RefTy (m := m) (α := α) shape) from do
          let weight ← const (m := m) (α := α) (Tensor.ones parameterShape)
          let bias ← const (m := m) (α := α) (Tensor.zeros parameterShape)
          normalize (α := α) (m := m) weight bias input }

end Internal

/--
Layer normalization over the final axis.

For an input of shape `[batch, tokens, width]`, each `(batch, token)` position has its own mean and
variance over the `width` entries. We subtract that mean, divide by `sqrt(variance + eps)`, then
apply the learned scale and bias. The variance divides by `width`.

Scale and bias each have shape `[width]` and are shared across the leading axes. The scale starts
at one and the bias at zero. Setting `bias := false` keeps only the scale; setting `affine := false`
removes both parameters. Training and evaluation use the same input statistics.
-/
def layerNorm
    (leading : Shape) (width : Nat)
    {hWidth : width > 0} (eps : Rat := 1e-5) (affine bias : Bool := true) :
    Layer (leading.appendDim width) (leading.appendDim width) :=
  Internal.affineNormalization "LayerNorm" (leading.appendDim width) [width] affine bias
    (Internal.validateEpsilon "LayerNorm" eps)
    (fun {α} _ _ {m} _ _ weight bias input =>
      Runtime.Autograd.Model.layerNorm (m := m) (α := α)
        (leading := leading) (width := width) hWidth input weight bias
        (epsilon := Context.ofRat eps))

/--
Root-mean-square normalization over the final axis.

Each row is divided by `sqrt(mean(x * x) + eps)` and multiplied by a learned scale of shape
`[width]`. RMSNorm squares the entries directly, without first subtracting the row's mean. Its only
learned parameter is the scale.

The scale starts at one. Setting `affine := false` removes it from model state and uses a constant
scale of one. The default `eps` is `1e-5` for every scalar type; pass a different value when the
model you are reproducing uses another epsilon.
-/
def rmsNorm
    (leading : Shape) (width : Nat)
    {hWidth : width > 0} (eps : Rat := 1e-5) (affine : Bool := true) :
    Layer (leading.appendDim width) (leading.appendDim width) :=
  let weightShape : Shape := .dim width .scalar
  if affine then
    { kind := "RMSNorm"
      stateShapes := [weightShape]
      initState := .cons (Tensor.ones weightShape) .nil
      runtimeInit := some (.cons .ones .nil)
      validateConfig := Internal.validateEpsilon "RMSNorm" eps
      forward := fun _ {α} _ _ {m} _ _ weight input =>
        Norm.rmsNorm (m := m) (α := α)
          (leading := leading) (width := width) hWidth input weight (ε := Context.ofRat eps) }
  else
    { kind := "RMSNorm"
      stateShapes := []
      initState := .nil
      validateConfig := Internal.validateEpsilon "RMSNorm" eps
      forward := fun _ {α} _ _ {m} _ _ input =>
        show m (RefTy (m := m) (α := α) (leading.appendDim width)) from do
          let weight ← const (m := m) (α := α) (Tensor.ones weightShape)
          Norm.rmsNorm (m := m) (α := α)
            (leading := leading) (width := width) hWidth input weight (ε := Context.ofRat eps) }

/--
Batch normalization over a batch, channel axis, and arbitrary spatial shape.

For each channel, training computes the mean and variance over the batch and every spatial
position. The forward pass divides the variance sum by `batch * spatial.size`, then normalizes
with `sqrt(variance + eps)` and applies the learned scale and bias.

Running statistics are updated separately with
`next = (1 - momentum) * running + momentum * batch`. The running variance uses the unbiased
estimate when there is more than one sample; for a single sample it keeps the finite biased
value. Evaluation reads these stored statistics and leaves them unchanged.

Model state contains scale, bias, running mean, running variance, and momentum in that order.
Only scale and bias receive gradients. Scale and running variance start at one; bias and running
mean start at zero.
-/
def batchNorm
    (batch channels : Nat) (spatial : Shape)
    (hWellFormed : (Shape.dim batch (Shape.dim channels spatial)).wellFormed)
    (momentum : Float := 0.1) (eps : Rat := 1e-5) :
    Layer (.dim batch (.dim channels spatial)) (.dim batch (.dim channels spatial)) :=
  let weightShape : Shape := .dim channels .scalar
  let biasShape : Shape := .dim channels .scalar
  let runningMeanShape : Shape := .dim channels .scalar
  let runningVarianceShape : Shape := .dim channels .scalar
  let momentumShape : Shape := Shape.scalar
  let initialWeight : Tensor Float weightShape := Tensor.ones (α := Float) weightShape
  let initialBias : Tensor Float biasShape := Tensor.zeros (α := Float) biasShape
  let initialRunningMean : Tensor Float runningMeanShape :=
    Tensor.zeros (α := Float) runningMeanShape
  let initialRunningVariance : Tensor Float runningVarianceShape :=
    Tensor.ones (α := Float) runningVarianceShape
  let momentumTensor : Tensor Float momentumShape := Tensor.scalar momentum
  { kind := "BatchNorm"
    stateShapes := [weightShape, biasShape, runningMeanShape, runningVarianceShape, momentumShape]
    initState := .cons initialWeight <| .cons initialBias <| .cons initialRunningMean <|
      .cons initialRunningVariance <| .cons momentumTensor .nil
    runtimeInit := some (.cons .ones (.cons .zeros (.cons .zeros (.cons .ones
      (.cons (.flat (FloatArray.mk #[momentum])) .nil)))))
    requiresGrad := #[true, true, false, false, false]
    validateConfig := do
      Internal.validateEpsilon "BatchNorm" eps
      unless momentum.isFinite do
        throw "BatchNorm: momentum must be finite"
      unless 0.0 ≤ momentum && momentum ≤ 1.0 do
        throw "BatchNorm: momentum must be between 0 and 1"
    updateBuffers := some (fun mode {_α} _ _ ps x => do
      match mode, ps with
      | .eval, _ => pure ps
      | .train, .cons weight (.cons bias (.cons runningMean (.cons runningVariance
          (.cons momentumT .nil)))) =>
          let (batchMean, batchVar) := batchChannelStats x
          let sampleCount := batch * Shape.size spatial
          let runningBatchVar := unbiasedRunningVariance batchVar sampleCount
          let nextMean := updateRunning runningMean batchMean momentumT
          let nextVariance := updateRunning runningVariance runningBatchVar momentumT
          pure (.cons weight (.cons bias <| .cons nextMean <|
            .cons nextVariance <| .cons momentumT .nil))
      | .train, _ => pure ps)
    forward := fun mode {α} _ _ =>
      fun {m} _ _ =>
        fun weight bias runningMean runningVariance _momentum x =>
          match mode with
          | .train =>
              Runtime.Autograd.Model.Norm.batchNormTrain (m := m) (α := α)
                hWellFormed x weight bias (ε := Context.ofRat eps)
          | .eval =>
              Runtime.Autograd.Model.Norm.batchNormEval (m := m) (α := α)
                hWellFormed x weight bias runningMean runningVariance (ε := Context.ofRat eps)
  }

/--
Instance normalization over the spatial axes of each sample and channel.

For an image tensor `[batch, channels, height, width]`, each `(batch, channel)` pair has its own
mean and variance over `height * width` entries. Both training and evaluation compute these
statistics from the current input; this layer has no running-statistics buffers.

The optional scale and bias have shape `[channels]` and are shared across samples and spatial
positions. Both are enabled by default. Setting `bias := false` keeps only the scale, while
`affine := false` removes both. The variance divides by `spatial.size`, and `eps` is added before
taking its square root.
-/
def instanceNorm
    (batch channels : Nat) (spatial : Shape)
    (hWellFormed : (Shape.dim batch (Shape.dim channels spatial)).wellFormed)
    (eps : Rat := 1e-5) (affine bias : Bool := true) :
    Layer (.dim batch (.dim channels spatial)) (.dim batch (.dim channels spatial)) :=
  Internal.affineNormalization "InstanceNorm"
    (.dim batch (.dim channels spatial)) [channels] affine bias
    (Internal.validateEpsilon "InstanceNorm" eps)
    (fun {α} _ _ {m} _ _ weight bias input =>
      Norm.instanceNorm (m := m) (α := α) hWellFormed input weight bias (ε := Context.ofRat eps))

/--
Group normalization over channel groups and their spatial positions, separately for each sample.

Channels are split into `groups` equal, contiguous groups. For example, six channels with
`groups := 2` form two groups of three channels. Each group's mean and variance include all three
channels and all their spatial positions. The variance divides by the number of entries in that
group, and `eps` is added before taking its square root.

Scale and bias still have one entry per channel, shared across samples and spatial positions.
Setting `bias := false` keeps only the scale; setting `affine := false` removes both parameters.
Training and evaluation use the same input statistics.
-/
def groupNorm
    (batch channels groups : Nat) (spatial : Shape)
    (hWellFormed : (Shape.dim batch (Shape.dim channels spatial)).wellFormed)
    (hGroups : groups > 0) (hGroupsLe : channels ≥ groups)
    (hDiv : channels % groups = 0) (eps : Rat := 1e-5) (affine bias : Bool := true) :
    Layer (.dim batch (.dim channels spatial)) (.dim batch (.dim channels spatial)) :=
  Internal.affineNormalization s!"GroupNorm(groups={groups})"
    (.dim batch (.dim channels spatial)) [channels] affine bias
    (Internal.validateEpsilon "GroupNorm" eps)
    (fun {α} _ _ {m} _ _ weight bias input =>
      Norm.groupNorm (m := m) (α := α) hWellFormed hGroups hGroupsLe hDiv input weight bias
        (ε := Context.ofRat eps))

end Layers

end Model
end Autograd
end Runtime
