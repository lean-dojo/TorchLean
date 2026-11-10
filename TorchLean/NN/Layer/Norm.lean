/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Layer
public import TorchLean.NN.Norm

/-!
# Normalization Layers

The normalization layers package normalization programs with their learned parameters and, for
BatchNorm, its running statistics. LayerNorm and RMSNorm operate on the final axis. BatchNorm,
InstanceNorm, and GroupNorm take inputs of shape `[batch, channels, ...spatial]`, where `spatial`
can describe a sequence, image, volume, or any other collection of spatial axes.

Each constructor accepts a positive `eps`, added under the square root. It is stored as a rational
and cast with the scalar type's rational cast once the forward program fixes its scalar type, so
the same configuration serves real-valued specifications and floating-point execution. `Float`,
`Float32` and the configured binary formats round the exact fraction once, and the result can
underflow or overflow. Floating-point execution needs a converted epsilon that is positive and
finite, and the static configuration check enforces only the positivity of the rational input.
In tiny formats the default can round to zero. Neither eager execution nor typed-graph lowering
substitutes `Context.defaultEpsilon`, so a constant LayerNorm input can then produce NaNs.
Pass an `eps` whose converted value is positive and finite before lowering the model.
-/

@[expose] public section

open TorchLean.Shape (size)

namespace TorchLean
namespace nn.Layer

open TorchLean.Tensor
open Ops

/--
Require a positive epsilon before constructing runtime state.

A constant input has zero variance, and adding `eps` under the square root gives that input a
positive denominator in the real-valued normalization formula.
-/
def validateEpsilon (kind : String) (eps : Rat) : Except String Unit :=
  if eps > 0 then pure () else .error s!"{kind}: epsilon must be positive"

/--
Build a layer around a normalization program that expects scale, bias, and input references.

With `affine := true`, the scale starts at one and the optional bias starts at zero. Setting
`bias := false` leaves only the scale in model state, and setting `affine := false` leaves neither
parameter. The forward program supplies constant ones and zeros for the omitted arguments, so one
normalization program serves all three configurations, eager or recorded in a typed graph.
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
        runtimeInit := some (.cons (.scheme .ones) (.cons (.scheme .zeros) .nil))
        validateConfig
        forward := fun _ => normalize }
    else
      { kind
        stateShapes := [parameterShape]
        initState := .cons (Tensor.ones parameterShape) .nil
        runtimeInit := some (.cons (.scheme .ones) .nil)
        validateConfig
        forward := fun _ {α} _ _ {m} _ _ weight input =>
          show m (Ref (m := m) (α := α) shape) from do
            let bias ← Ops.const (m := m) (α := α) (Tensor.zeros parameterShape)
            normalize (α := α) (m := m) weight bias input }
  else
    { kind
      stateShapes := []
      initState := .nil
      validateConfig
      forward := fun _ {α} _ _ {m} _ _ input =>
        show m (Ref (m := m) (α := α) shape) from do
          let weight ← Ops.const (m := m) (α := α) (Tensor.ones parameterShape)
          let bias ← Ops.const (m := m) (α := α) (Tensor.zeros parameterShape)
          normalize (α := α) (m := m) weight bias input }

/--
Layer normalization over the final axis.

For an input of shape `[batch, tokens, width]`, each `(batch, token)` position has its own mean and
variance over the `width` entries. The layer subtracts that mean, divides by
`sqrt(variance + eps)` and then applies the learned scale and bias. The variance divides by
`width`.

Scale and bias each have shape `[width]` and are shared across the leading axes. The scale starts
at one and the bias at zero. Setting `bias := false` keeps only the scale, and setting
`affine := false` removes both parameters. Training and evaluation use the same input statistics.
-/
def layerNorm
    (leading : Shape) (width : Nat)
    {hWidth : width > 0} (eps : Rat := 1e-5) (affine bias : Bool := true) :
    Layer (leading ++ [width]) (leading ++ [width]) :=
  affineNormalization "LayerNorm" (leading ++ [width]) [width] affine bias
    (validateEpsilon "LayerNorm" eps)
    (fun {α} _ _ {m} _ _ weight bias input =>
      functional.layerNorm (m := m) (α := α)
        (leading := leading) (width := width) hWidth input weight bias
        (epsilon := (eps : α)))

/--
Root-mean-square normalization over the final axis.

Each row is divided by `sqrt(mean(x * x) + eps)` and multiplied by a learned scale of shape
`[width]`. RMSNorm squares the entries themselves and subtracts no row mean, and its only learned
parameter is the scale.

The scale starts at one. Setting `affine := false` removes it from model state and uses a constant
scale of one. The default `eps` is `1e-5` for every scalar type. Pass a different value to match a
model that uses another epsilon.
-/
def rmsNorm
    (leading : Shape) (width : Nat)
    {hWidth : width > 0} (eps : Rat := 1e-5) (affine : Bool := true) :
    Layer (leading ++ [width]) (leading ++ [width]) :=
  let weightShape : Shape := .dim width .scalar
  if affine then
    { kind := "RMSNorm"
      stateShapes := [weightShape]
      initState := .cons (Tensor.ones weightShape) .nil
      runtimeInit := some (.cons (.scheme .ones) .nil)
      validateConfig := validateEpsilon "RMSNorm" eps
      forward := fun _ {α} _ _ {m} _ _ weight input =>
        Norm.rms (m := m) (α := α)
          (leading := leading) (width := width) hWidth input weight (ε := (eps : α)) }
  else
    { kind := "RMSNorm"
      stateShapes := []
      initState := .nil
      validateConfig := validateEpsilon "RMSNorm" eps
      forward := fun _ {α} _ _ {m} _ _ input =>
        show m (Ref (m := m) (α := α) (leading ++ [width])) from do
          let weight ← Ops.const (m := m) (α := α) (Tensor.ones weightShape)
          Norm.rms (m := m) (α := α)
            (leading := leading) (width := width) hWidth input weight (ε := (eps : α)) }

/--
Batch normalization over a batch, channel axis, and arbitrary spatial shape.

Training applies `Op.batchNorm` to the input with its channel axis moved first: each channel is
normalized with the mean and the population variance of its entries over the batch and every
spatial position (`Tensor.batchNorm` with batch statistics), then scaled and shifted. Evaluation
applies `Op.batchNormRunning` along axis 1 with the stored statistics, which it reads as frozen
inputs and leaves unchanged.

A training step updates the running statistics with
`next = (1 - momentum) * running + momentum * batch`. The batch mean and variance are
`Tensor.mean` and `Tensor.var` of the same channel-first view over its other axes: over the reals,
the statistics the forward normalizes with. At a float they differ: the forward divides each
centered channel by its scale before squaring it, while the sum of squared deviations in
`Tensor.var` is unscaled and can overflow to infinity where the variance itself is finite. The
running variance uses the unbiased estimate when there is more than one sample, and the finite
biased value for a single sample.

Model state contains scale, bias, running mean, running variance, and momentum in that order.
Only scale and bias receive gradients. Scale and running variance start at one, and bias and
running mean start at zero. The configuration check requires a positive `eps` and a finite
`momentum` in `[0, 1]`.
-/
def batchNorm
    (batch channels : Nat) (spatial : Shape)
    (hWellFormed : (Shape.dim batch (Shape.dim channels spatial)).Positive)
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
    runtimeInit := some (.cons (.scheme .ones) (.cons (.scheme .zeros)
      (.cons (.scheme .zeros) (.cons (.scheme .ones)
      (.cons (.flat (FloatArray.mk #[momentum])) .nil)))))
    requiresGrad := #[true, true, false, false, false]
    validateConfig := do
      validateEpsilon "BatchNorm" eps
      unless momentum.isFinite do
        throw "BatchNorm: momentum must be finite"
      unless 0.0 ≤ momentum && momentum ≤ 1.0 do
        throw "BatchNorm: momentum must be between 0 and 1"
    updateBuffers := some (fun mode {_α} _ _ ps x => do
      match mode, ps with
      | .eval, _ => pure ps
      | .train, .cons weight (.cons bias (.cons runningMean (.cons runningVariance
          (.cons momentumT .nil)))) =>
          let channelFirst : Tensor _ (.dim channels (.dim batch spatial)) :=
            Tensor.permute x (Shape.adjacentSwap (.dim batch (.dim channels spatial)) 0)
              (Shape.swapAdjacentAtDepth (.dim batch (.dim channels spatial)) 0)
          let others := Shape.otherAxes (.dim channels (.dim batch spatial)) [⟨0, Nat.succ_pos _⟩]
          let hStats := Shape.eraseAxes_otherAxes_singleton (.dim channels (.dim batch spatial))
            ⟨0, Nat.succ_pos _⟩
          let batchMean : Tensor _ [channels] :=
            Tensor.castShape (Tensor.mean channelFirst others false) hStats
          let batchVar : Tensor _ [channels] :=
            Tensor.castShape (Tensor.var channelFirst others false) hStats
          let sampleCount := batch * Shape.size spatial
          let runningBatchVar := unbiasedRunningVariance batchVar sampleCount
          let nextMean := updateRunning runningMean batchMean momentumT
          let nextVariance := updateRunning runningVariance runningBatchVar momentumT
          pure (.cons weight (.cons bias <| .cons nextMean <|
            .cons nextVariance <| .cons momentumT .nil)))
    forward := fun mode {α} _ _ =>
      fun {m} _ _ =>
        fun weight bias runningMean runningVariance _momentum x =>
          let epsilon := Ops.dataConst (m := m) (α := α)
            (Tensor.scalar (eps : α))
          match mode with
          | .train =>
              show m (Ops.Ref m α _) from do
              let channelFirst :
                  Ops.Ref m α (.dim channels (.dim batch spatial)) ←
                Ops.apply (m := m) (α := α)
                  (.permute (s := .dim batch (.dim channels spatial))
                    (Shape.adjacentSwap (.dim batch (.dim channels spatial)) 0)
                    (Shape.swapAdjacentAtDepth (.dim batch (.dim channels spatial)) 0)
                    (by tensor_shape)) (x, ())
              let normalized ← Ops.apply (m := m) (α := α)
                  (.batchNorm (channels := channels) (spatial := .dim batch spatial)
                    ⟨hWellFormed.2.1, hWellFormed.1, hWellFormed.2.2⟩)
                  (channelFirst, weight, bias, .data epsilon, ())
              Ops.apply (m := m) (α := α)
                  (.permute (s := .dim channels (.dim batch spatial))
                    (Shape.adjacentSwap (.dim channels (.dim batch spatial)) 0)
                    (Shape.swapAdjacentAtDepth (.dim channels (.dim batch spatial)) 0)
                    (by tensor_shape)) (normalized, ())
          | .eval =>
              Ops.apply (m := m) (α := α)
                (.batchNormRunning (s := .dim batch (.dim channels spatial))
                  ⟨1, Nat.succ_lt_succ (Nat.succ_pos _)⟩)
                (x, weight, bias, .ref runningMean, .ref runningVariance, .data epsilon, ())
  }

/--
Instance normalization over the spatial axes of each sample and channel.

For an image tensor `[batch, channels, height, width]`, each `(batch, channel)` pair has its own
mean and variance over `height * width` entries. Training and evaluation both compute these
statistics from the current input, and the layer has no running-statistics buffers.

The optional scale and bias have shape `[channels]` and are shared across samples and spatial
positions. Both are enabled by default. Setting `bias := false` keeps only the scale, while
`affine := false` removes both. The variance divides by `spatial.size`, and `eps` is added before
taking its square root. The layer runs `Norm.group` with one group per channel.
-/
def instanceNorm
    (batch channels : Nat) (spatial : Shape)
    (hWellFormed : (Shape.dim batch (Shape.dim channels spatial)).Positive)
    (eps : Rat := 1e-5) (affine bias : Bool := true) :
    Layer (.dim batch (.dim channels spatial)) (.dim batch (.dim channels spatial)) :=
  affineNormalization "InstanceNorm"
    (.dim batch (.dim channels spatial)) [channels] affine bias
    (validateEpsilon "InstanceNorm" eps)
    (fun {α} _ _ {m} _ _ weight bias input =>
      Norm.group (m := m) (α := α) (groups := channels) hWellFormed hWellFormed.2.1
        (Nat.le_refl channels) (Nat.mod_self channels) input weight bias (ε := (eps : α)))

/--
Group normalization over channel groups and their spatial positions, separately for each sample.

Channels are split into `groups` equal, contiguous groups. For example, six channels with
`groups := 2` form two groups of three channels, and each group's mean and variance cover its three
channels at all their spatial positions. The variance divides by the number of entries in that
group, and `eps` is added before taking its square root.

Scale and bias have one entry per channel, shared across samples and spatial positions. Setting
`bias := false` keeps only the scale, and setting `affine := false` removes both parameters.
Training and evaluation use the same input statistics.
-/
def groupNorm
    (batch channels groups : Nat) (spatial : Shape)
    (hWellFormed : (Shape.dim batch (Shape.dim channels spatial)).Positive)
    (hGroups : groups > 0) (hGroupsLe : channels ≥ groups)
    (hDiv : channels % groups = 0) (eps : Rat := 1e-5) (affine bias : Bool := true) :
    Layer (.dim batch (.dim channels spatial)) (.dim batch (.dim channels spatial)) :=
  affineNormalization s!"GroupNorm(groups={groups})"
    (.dim batch (.dim channels spatial)) [channels] affine bias
    (validateEpsilon "GroupNorm" eps)
    (fun {α} _ _ {m} _ _ weight bias input =>
      Norm.group (m := m) (α := α) hWellFormed hGroups hGroupsLe hDiv input weight bias
        (ε := (eps : α)))

end nn.Layer
end TorchLean
