/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Neural.Indexed
public import NN.API.Neural.Leading
public import NN.API.Neural.Positional
public import NN.API.Neural.Transformer
public import NN.Runtime.Autograd.Model.Layers.Mamba

/-!
# Layer Implementations

Explicit-seed layer constructors behind the public `nn.*` builders in `NN.API.Seeded`.

Only `NN.API.Seeded` imports this module. Every constructor here takes its initialization seeds as
ordinary arguments; the public builders draw those seeds from `nn.Builder` and are the supported
way to construct layers. The bodies stay exposed so that downstream proofs can unfold a built model
through the public builders.
-/

@[expose] public section

namespace TorchLean
namespace nn
namespace Impl

/-!
## Leading dimensions
-/

/--
Reshape arbitrary leading dimensions into the single outer dimension expected by `layer`.

For an input of shape `leading.concat σ`, the layer receives shape
`[leading.size].concat σ`; its output is then reshaped from `[leading.size].concat τ` to
`leading.concat τ`. The adapter reuses the layer's parameters and buffer-update function.
-/
def adaptLeadingShape (leading : Spec.Shape) {σ τ : Spec.Shape}
    (layer : Layer (σ.prependDim leading.size) (τ.prependDim leading.size)) :
    Layer (leading.concat σ) (leading.concat τ) :=
  { kind := layer.kind
    stateShapes := layer.stateShapes
    initState := layer.initState
    runtimeInit := layer.runtimeInit
    requiresGrad := layer.requiresGrad
    validateConfig := layer.validateConfig
    updatesBuffersInForward := layer.updatesBuffersInForward
    updateBuffers := layer.updateBuffers.map fun update mode {α} _ _ state input =>
      update mode state <| input.reshape _ (by
        simp [Spec.Shape.size_concat, Spec.Shape.size])
    forward := fun mode {α} _ _ =>
      fun {m} _ _ =>
        Runtime.Autograd.Torch.CurriedRef.curry
          (Ref := fun shape => TorchLean.Runtime.ValueRef (m := m) (α := α) shape)
          (ss := layer.stateShapes ++ [leading.concat σ])
          (β := m (TorchLean.Runtime.ValueRef (m := m) (α := α) (leading.concat τ)))
          (fun arguments => do
            let (state, input) :=
              Runtime.Autograd.Torch.RefList.splitLast
                (Ref := fun shape =>
                  TorchLean.Runtime.ValueRef (m := m) (α := α) shape)
                (ss := layer.stateShapes) (τ := leading.concat σ) arguments
            let flattenedInput ←
              Runtime.Autograd.Torch.reshape (m := m) (α := α)
                (s₁ := leading.concat σ) (s₂ := σ.prependDim leading.size)
                input (by simp [Spec.Shape.size_concat, Spec.Shape.size])
            let flattenedOutput ←
              Runtime.Autograd.Torch.CurriedRef.uncurry
                (Ref := fun shape =>
                  TorchLean.Runtime.ValueRef (m := m) (α := α) shape)
                (ss := layer.stateShapes ++ [σ.prependDim leading.size])
                (β := m (TorchLean.Runtime.ValueRef (m := m) (α := α)
                  (τ.prependDim leading.size)))
                (layer.forward mode (α := α) (m := m))
                (Runtime.Autograd.Torch.RefList.append state (.cons flattenedInput .nil))
            Runtime.Autograd.Torch.reshape (m := m) (α := α)
              (s₁ := τ.prependDim leading.size) (s₂ := leading.concat τ)
              flattenedOutput (by simp [Spec.Shape.size_concat, Spec.Shape.size])) }

/-- Share a single-sequence recurrent core over every index of `batchShape`. -/
def batchedRecurrent (batchShape : Spec.Shape) {sequenceLength inputWidth hiddenWidth : Nat}
    (core : Sequential [sequenceLength, inputWidth] [sequenceLength, hiddenWidth]) :
    Sequential
      ((batchShape.appendDim sequenceLength).appendDim inputWidth)
      ((batchShape.appendDim sequenceLength).appendDim hiddenWidth) := by
  simpa only [Spec.Shape.appendDim_appendDim_eq_concat] using mapLeading batchShape core

/-!
## Affine and recurrent layers
-/

/--
Linear layer on the last axis (prefix-shape preserving).

PyTorch analogue: `torch.nn.Linear`.
See `https://pytorch.org/docs/stable/generated/torch.nn.Linear.html`.

If `input` has shape `[..., inputWidth]`, `linear inputWidth outputWidth` returns a model of shape
`[..., outputWidth]`. The leading dimensions are treated as a batch: they are flattened to
`(numel(prefix), inputWidth)`, the affine map is applied once, and the result is reshaped back.
-/
def linear (inputWidth outputWidth : Nat) (weightSeed biasSeed : Nat := 0)
    (batchShape : Spec.Shape := []) (config : Linear.Config := {}) :
    Sequential (batchShape.appendDim inputWidth) (batchShape.appendDim outputWidth) :=
  let weightShape : Spec.Shape := [outputWidth, inputWidth]
  let biasShape : Spec.Shape := [outputWidth]
  let weightInitialization :=
    config.weightInitialization?.getD (.xavierUniform inputWidth outputWidth)
  let initialWeight : TorchLean.Tensor Float weightShape := Runtime.Autograd.Torch.Init.tensor
    (s := weightShape) (sch := weightInitialization) (seed := weightSeed)
  let initialBias : TorchLean.Tensor Float biasShape := Runtime.Autograd.Torch.Init.tensor
    (s := biasShape) (sch := config.biasInitialization) (seed := biasSeed)
  let batch : Nat := batchShape.size
  Sequential.fromLayer
    { kind := s!"Linear({inputWidth}, {outputWidth})"
      stateShapes := [weightShape, biasShape]
      initState := TorchLean.TensorPack.pair initialWeight initialBias
      runtimeInit := some (.cons
        (Runtime.Autograd.Model.Module.RuntimeInit.FloatInit.ofScheme
          weightInitialization weightSeed)
        (.cons
          (Runtime.Autograd.Model.Module.RuntimeInit.FloatInit.ofScheme
            config.biasInitialization biasSeed)
          .nil))
      requiresGrad := #[true, true]
      validateConfig := config.validate inputWidth outputWidth
      forward := fun _ {α} _ _ =>
        fun {m} _ _ =>
          fun w b x =>
            let sIn := batchShape.appendDim inputWidth
            let sOut := batchShape.appendDim outputWidth
            ((do
              let x2d ←
                Runtime.Autograd.Torch.reshape (m := m) (α := α)
                  (s₁ := sIn)
                  (s₂ := [batch, inputWidth])
                  x (by
                    rw [Spec.Shape.size_appendDim]
                    simp [batch, Spec.Shape.size])

              let wT ←
                Runtime.Autograd.Torch.swapAdjacentAtDepth (m := m) (α := α)
                  (s := [outputWidth, inputWidth]) 0 w
              let y ← Runtime.Autograd.Torch.matmul (m := m) (α := α)
                (batchA := []) (batchB := []) (batch := [])
                (mDim := batch) (nDim := inputWidth) (pDim := outputWidth) x2d wT
              let y2d ←
                Runtime.Autograd.Model.F.addB (m := m) (α := α)
                  (t := [batch, outputWidth]) y b
              Runtime.Autograd.Torch.reshape (m := m) (α := α)
                (s₁ := [batch, outputWidth])
                (s₂ := sOut)
                y2d (by
                  rw [Spec.Shape.size_appendDim]
                  simp [batch, Spec.Shape.size])
            ) : m (TorchLean.Runtime.ValueRef (m := m) (α := α) sOut))
    }

/--
Vanilla RNN layer (time-major sequence, no batch axis).

Semantics:

$$
h_t=\tanh\!\left(W[x_t;h_{t-1}]+b\right),\qquad h_{-1}=0.
$$

This is implemented by unrolling `sequenceLength` steps using existing TorchLean ops, so it runs on
both CPU and CUDA backends.

PyTorch analogy: `torch.nn.RNN(inputWidth, hiddenWidth, nonlinearity="tanh")` with
`batch_first=false`, specialized to a single batch element.
-/
def rnn (sequenceLength inputWidth hiddenWidth : Nat)
    (weightSeed : Nat := 0) :
    Sequential
      [sequenceLength, inputWidth]
      [sequenceLength, hiddenWidth] :=
  Sequential.fromLayer
    (Runtime.Autograd.Model.Layers.rnn
      (sequenceLength := sequenceLength) (inputWidth := inputWidth) (hiddenWidth := hiddenWidth)
      weightSeed)

/--
GRU layer (time-major sequence, no batch axis).

This is implemented by unrolling `sequenceLength` Cho-style steps using existing TorchLean ops, so
it runs on both CPU and CUDA backends. PyTorch uses a different reset-after candidate
parameterization; its GRU checkpoints are not directly compatible with this constructor.
-/
def gru (sequenceLength inputWidth hiddenWidth : Nat)
    (resetWeightSeed updateWeightSeed candidateWeightSeed : Nat := 0) :
    Sequential
      [sequenceLength, inputWidth]
      [sequenceLength, hiddenWidth] :=
  Sequential.fromLayer
    (Runtime.Autograd.Model.Layers.gru
      (sequenceLength := sequenceLength) (inputWidth := inputWidth) (hiddenWidth := hiddenWidth)
      resetWeightSeed updateWeightSeed candidateWeightSeed)

/--
Trainable selective Mamba layer.

The input has shape `(sequenceLength × inputWidth)` and the output has shape
`(sequenceLength × hiddenWidth)`. Each token passes through a causal depthwise convolution and
produces its own time steps, input coefficients, and readout coefficients for a diagonal state
update. `options` controls the expanded channels, states per channel, and convolution width.
The recurrence is unrolled with differentiable tensor operations.
-/
def mamba (sequenceLength inputWidth hiddenWidth : Nat)
    (inputWeightSeed stateWeightSeed gateWeightSeed : Nat := 0)
    (options : Runtime.Autograd.Model.Mamba.Options := {}) :
    Sequential
      [sequenceLength, inputWidth]
      [sequenceLength, hiddenWidth] :=
  Sequential.fromLayer
    (Runtime.Autograd.Model.Layers.mamba
      (sequenceLength := sequenceLength) (inputWidth := inputWidth) (hiddenWidth := hiddenWidth)
      inputWeightSeed stateWeightSeed gateWeightSeed (options := options))

/--
LSTM layer (time-major sequence, no batch axis).

This is implemented by unrolling `sequenceLength` steps using existing TorchLean ops, so it runs on
both CPU and CUDA backends.

PyTorch analogy: `torch.nn.LSTM(inputWidth, hiddenWidth)` with `batch_first=false`, specialized to
a single batch element.
-/
def lstm (sequenceLength inputWidth hiddenWidth : Nat)
    (forgetWeightSeed inputWeightSeed candidateWeightSeed outputWeightSeed : Nat := 0) :
    Sequential
      [sequenceLength, inputWidth]
      [sequenceLength, hiddenWidth] :=
  Sequential.fromLayer
    (Runtime.Autograd.Model.Layers.lstm
      (sequenceLength := sequenceLength) (inputWidth := inputWidth) (hiddenWidth := hiddenWidth)
      forgetWeightSeed inputWeightSeed candidateWeightSeed outputWeightSeed)

/-!
## Shape and reduction layers
-/

/-- Softmax over a tensor dimension, rejected by model validation when the axis is out of bounds. -/
def softmax {s : Spec.Shape} (axis : Nat) : Sequential s s :=
  if hAxis : axis < s.rank then
    letI : Spec.Shape.AxisInBounds axis s :=
      Spec.Shape.AxisInBounds.ofRank hAxis
    Sequential.fromLayer <| Runtime.Autograd.Model.Layers.softmax (s := s) axis
  else
    nn.Internal.invalidConfiguration s s "Softmax"
      s!"Softmax: axis {axis} is out of bounds for rank {s.rank}"

/--
Stable log-softmax over a tensor dimension, rejected by model validation when the axis is out of
bounds.
-/
def logSoftmax {s : Spec.Shape} (axis : Nat) : Sequential s s :=
  if hAxis : axis < s.rank then
    letI : Spec.Shape.AxisInBounds axis s :=
      Spec.Shape.AxisInBounds.ofRank hAxis
    Sequential.fromLayer <| Runtime.Autograd.Model.Layers.logSoftmax (s := s) axis
  else
    nn.Internal.invalidConfiguration s s "LogSoftmax"
      s!"LogSoftmax: axis {axis} is out of bounds for rank {s.rank}"

/-- Reduce-sum to a scalar. PyTorch analogue: `torch.sum`. -/
def sum {s : Spec.Shape} : Sequential s [] :=
  Sequential.fromLayer <| Runtime.Autograd.Model.Layers.sum (s := s)

/-- Flatten any tensor into a 1D vector of length `size s`. PyTorch analogue: `torch.flatten`. -/
def flatten {s : Spec.Shape} : Sequential s [Spec.Shape.size s] :=
  Sequential.fromLayer <| Runtime.Autograd.Model.Layers.flatten (s := s)

/--
View a tensor with a new shape containing the same number of scalar entries.

This is the shape-typed counterpart of `torch.reshape`. A size mismatch is represented as an
invalid model configuration and rejected by the ordinary validation path.
-/
def reshape (source target : Spec.Shape) : Sequential source target :=
  if sameSize : Spec.Shape.size source = Spec.Shape.size target then
    Sequential.fromLayer
      { kind := "Reshape"
        stateShapes := []
        initState := .nil
        requiresGrad := #[]
        forward := fun _ {α} _ _ =>
          fun {m} _ _ =>
            fun x =>
              Runtime.Autograd.Torch.reshape (m := m) (α := α)
                (s₁ := source) (s₂ := target) x sameSize }
  else
    nn.Internal.invalidConfiguration source target "Reshape"
      s!"Reshape: source has {source.size} elements but target has {target.size}"

/--
Flatten each tensor after an arbitrary batch shape.

For `batchShape = [batch]`, this is the typed counterpart of
`torch.flatten(x, start_dim=1)`. Multiple batch dimensions are preserved without introducing a
separate batched tensor type.
-/
def flattenAfter (batchShape : Spec.Shape := []) {shape : Spec.Shape} :
    Sequential (batchShape.concat shape) (batchShape.appendDim shape.size) :=
  let source := batchShape.concat shape
  let target := batchShape.appendDim shape.size
  Sequential.fromLayer
    { kind := "FlattenAfter"
      stateShapes := []
      initState := .nil
      requiresGrad := #[]
      forward := fun _ {α} _ _ =>
        fun {m} _ _ =>
          fun x =>
            Runtime.Autograd.Torch.reshape (m := m) (α := α)
              (s₁ := source)
              (s₂ := target)
              x (by
                rw [Spec.Shape.size_concat, Spec.Shape.size_appendDim])
    }

/--
Dropout layer (active in train mode, identity in eval mode).

PyTorch analogue: `torch.nn.Dropout`.
-/
def dropout {s : Spec.Shape} (p : Float) (seed : Nat := 0) : Sequential s s :=
  Sequential.fromLayer <| Runtime.Autograd.Model.Layers.dropout (s := s) p seed

/-!
## Convolution and pooling
-/

/--
Apply an arbitrary-dimensional convolution to the channel and spatial suffix of a tensor.

The input suffix is `(inputChannels, spatial...)`. Any axes in `batchShape` are preserved;
internally they are flattened into one runtime batch and restored after the convolution.
-/
def conv (batchShape : Spec.Shape := []) {d : Nat} {inputChannels : Nat}
    (spatial : Tensor Nat [d]) (config : Convolution.Config d)
    (kernelSeed : Nat := 0) :
    Sequential
      (batchShape.concat ((spatial.to Spec.Shape).prependDim inputChannels))
      (batchShape.concat
        (((config.outputSpatial spatial).to Spec.Shape).prependDim config.outChannels)) :=
  nn.Sequential.fromLayer <| adaptLeadingShape batchShape <|
      Runtime.Autograd.Model.Layers.conv
        batchShape.size d
        inputChannels config.outChannels
        config.kernelSize config.stride config.padding spatial
        kernelSeed config.weightInitialization

/--
Apply an arbitrary-dimensional transpose convolution to the channel and spatial suffix.

The input suffix is `(inputChannels, spatial...)`. Any axes in `batchShape` are mapped independently
and restored after the operation.
-/
def convTranspose (batchShape : Spec.Shape := []) {d : Nat} {inputChannels : Nat}
    (spatial : Tensor Nat [d]) (config : TransposedConvolution.Config d)
    (kernelSeed : Nat := 0) :
    Sequential
      (batchShape.concat ((spatial.to Spec.Shape).prependDim inputChannels))
      (batchShape.concat
        (((config.outputSpatial spatial).to Spec.Shape).prependDim config.outChannels)) :=
  nn.Sequential.fromLayer <| adaptLeadingShape batchShape <|
      Runtime.Autograd.Model.Layers.convTranspose
        batchShape.size d
        inputChannels config.outChannels
        config.kernelSize config.stride config.padding spatial
        kernelSeed config.weightInitialization

/-- Apply max pooling to the channel and spatial suffix of a tensor. -/
def maxPool (batchShape : Spec.Shape := []) {d channels : Nat} (spatial : Tensor Nat [d])
    (config : Pooling.Config d) :
    Sequential
      (batchShape.concat ((spatial.to Spec.Shape).prependDim channels))
      (batchShape.concat
        (((config.outputSpatial spatial).to Spec.Shape).prependDim channels)) :=
  nn.Sequential.fromLayer <| adaptLeadingShape batchShape <|
      Runtime.Autograd.Model.Layers.maxPool
        batchShape.size d channels
        config.kernelSize config.stride config.padding spatial

/-- Apply average pooling to the channel and spatial suffix of a tensor. -/
def avgPool (batchShape : Spec.Shape := []) {d channels : Nat} (spatial : Tensor Nat [d])
    (config : Pooling.Config d) :
    Sequential
      (batchShape.concat ((spatial.to Spec.Shape).prependDim channels))
      (batchShape.concat
        (((config.outputSpatial spatial).to Spec.Shape).prependDim channels)) :=
  nn.Sequential.fromLayer <| adaptLeadingShape batchShape <|
      Runtime.Autograd.Model.Layers.avgPool
        batchShape.size d channels
        config.kernelSize config.stride config.padding spatial

/-- Global average pooling over every spatial axis, preserving the batch axes and channels. -/
def globalAvgPool (batchShape : Spec.Shape := []) {d channels : Nat}
    (spatial : Tensor Nat [d]) :
    Sequential
      (batchShape.concat ((spatial.to Spec.Shape).prependDim channels))
      (batchShape.appendDim channels) := by
  let spatialShape := spatial.to Spec.Shape
  let meanSpatial : Layer spatialShape [] :=
    { kind := s!"GlobalAvgPool(rank={d})"
      stateShapes := []
      initState := .nil
      validateConfig := do
        if channels = 0 then
          throw "GlobalAvgPool: channel count must be positive"
        if spatial.prod = 0 then
          throw "GlobalAvgPool: spatial dimensions must be positive"
      requiresGrad := #[]
      forward := fun _ {α} _ _ =>
        fun {m} _ _ =>
          fun x =>
            Runtime.Autograd.Model.F.mean
              (m := m) (α := α) (s := spatialShape) x }
  let pooled :=
    mapLeading (batchShape.appendDim channels) (Sequential.fromLayer meanSpatial)
  simpa [spatialShape, Spec.Shape.appendDim_eq_concat, Spec.Shape.concat_assoc] using pooled

/-!
## Normalization
-/

/--
Layer normalization over the final axis of a tensor.

Every index in `batchShape` selects one vector of length `width`. Its entries share a mean and
variance, while scale and bias are shared across all leading indices. `batchShape := []` describes
a single vector, and zero-sized leading axes are also allowed. Only `width` must be positive.

`eps` is added to the variance before the square root. Setting `bias := false` keeps only the
learned scale; setting `affine := false` removes both learned parameters.
-/
def layerNorm (batchShape : Spec.Shape := []) {width : Nat}
    (eps : Rat := 1e-5) (affine bias : Bool := true) :
    Sequential (batchShape.appendDim width) (batchShape.appendDim width) :=
  if hWidth : width = 0 then
    nn.Internal.invalidConfiguration
      (batchShape.appendDim width) (batchShape.appendDim width)
      "LayerNorm" "LayerNorm: normalized width must be positive"
  else
    Sequential.fromLayer <| Runtime.Autograd.Model.Layers.layerNorm
      batchShape width (hWidth := Nat.pos_of_ne_zero hWidth)
      (eps := eps) (affine := affine) (bias := bias)

/--
Divide each final-axis vector by `sqrt(mean(x * x) + eps)`, then apply a learned scale.

The scale has shape `[width]` and is shared across all leading indices. Setting `affine := false`
removes it from model state. The default `eps` is `1e-5`, independent of the scalar type.
-/
def rmsNorm (batchShape : Spec.Shape := []) {width : Nat}
    (eps : Rat := 1e-5) (affine : Bool := true) :
    Sequential (batchShape.appendDim width) (batchShape.appendDim width) :=
  if hWidth : width = 0 then
    nn.Internal.invalidConfiguration
      (batchShape.appendDim width) (batchShape.appendDim width)
      "RMSNorm" "RMSNorm: normalized width must be positive"
  else
    Sequential.fromLayer <| Runtime.Autograd.Model.Layers.rmsNorm
      batchShape width (hWidth := Nat.pos_of_ne_zero hWidth) (eps := eps) (affine := affine)

/--
Batch normalization over `(batchShape..., channels, spatial...)` for any spatial rank.

All leading batch axes and spatial axes contribute to each channel's training statistics.
Evaluation uses the running mean and variance. `momentum` controls their moving-average updates,
and `eps` is added to the variance before taking the square root.
-/
def batchNorm (batchShape : Spec.Shape := []) {d channels : Nat}
    (spatial : Tensor Nat [d]) (momentum : Float := 0.1) (eps : Rat := 1e-5) :
    Sequential
      (batchShape.concat ((spatial.to Spec.Shape).prependDim channels))
      (batchShape.concat ((spatial.to Spec.Shape).prependDim channels)) := by
  let shape :=
    batchShape.concat ((spatial.to Spec.Shape).prependDim channels)
  if hBatch : batchShape.size = 0 then
    exact nn.Internal.invalidConfiguration shape shape "BatchNorm"
      "BatchNorm: batch shape must contain at least one element"
  else if hChannels : channels = 0 then
    exact nn.Internal.invalidConfiguration shape shape "BatchNorm"
      "BatchNorm: channel count must be positive"
  else if hSpatial : spatial.prod = 0 then
    exact nn.Internal.invalidConfiguration shape shape "BatchNorm"
      "BatchNorm: spatial shape must contain at least one element"
  else
  let spatialShape := spatial.to Spec.Shape
  let batch := batchShape.size
  have hSpatialSize : spatialShape.size ≠ 0 := by
    simpa only [spatialShape, Tensor.size_to_shape] using hSpatial
  let hInput :
      ((spatialShape.prependDim channels).prependDim batch).wellFormed :=
    ⟨Nat.pos_of_ne_zero (by simpa [batch] using hBatch),
      Nat.pos_of_ne_zero hChannels,
      Spec.Shape.wellFormed_of_size_pos (Nat.pos_of_ne_zero hSpatialSize)⟩
  exact Sequential.fromLayer <| adaptLeadingShape batchShape <|
    Runtime.Autograd.Model.Layers.batchNorm batch channels spatialShape hInput
      (momentum := momentum) (eps := eps)

/--
Instance normalization over `(batchShape..., channels, spatial...)` for any spatial rank.

Each sample and channel uses its own spatial mean and variance in both training and evaluation.
Scale and bias are shared across samples. Setting `bias := false` keeps only the scale, while
`affine := false` removes both parameters.
-/
def instanceNorm (batchShape : Spec.Shape := []) {d channels : Nat}
    (spatial : Tensor Nat [d]) (eps : Rat := 1e-5) (affine bias : Bool := true) :
    Sequential
      (batchShape.concat ((spatial.to Spec.Shape).prependDim channels))
      (batchShape.concat ((spatial.to Spec.Shape).prependDim channels)) := by
  let shape :=
    batchShape.concat ((spatial.to Spec.Shape).prependDim channels)
  if hBatch : batchShape.size = 0 then
    exact nn.Internal.invalidConfiguration shape shape "InstanceNorm"
      "InstanceNorm: batch shape must contain at least one element"
  else if hChannels : channels = 0 then
    exact nn.Internal.invalidConfiguration shape shape "InstanceNorm"
      "InstanceNorm: channel count must be positive"
  else if hSpatial : spatial.prod = 0 then
    exact nn.Internal.invalidConfiguration shape shape "InstanceNorm"
      "InstanceNorm: spatial shape must contain at least one element"
  else
  let spatialShape := spatial.to Spec.Shape
  let batch := batchShape.size
  have hSpatialSize : spatialShape.size ≠ 0 := by
    simpa only [spatialShape, Tensor.size_to_shape] using hSpatial
  let hInput :
      ((spatialShape.prependDim channels).prependDim batch).wellFormed :=
    ⟨Nat.pos_of_ne_zero (by simpa [batch] using hBatch),
      Nat.pos_of_ne_zero hChannels,
      Spec.Shape.wellFormed_of_size_pos (Nat.pos_of_ne_zero hSpatialSize)⟩
  exact Sequential.fromLayer <| adaptLeadingShape batchShape <|
    Runtime.Autograd.Model.Layers.instanceNorm batch channels spatialShape hInput
      (eps := eps) (affine := affine) (bias := bias)

/--
Group normalization over `(batchShape..., channels, spatial...)` for any spatial rank.

Within each sample, each group shares a mean and variance across its channels and spatial
positions. Scale and bias have one entry per channel. Setting `bias := false` keeps only the
scale; setting `affine := false` removes both parameters.
-/
def groupNorm (batchShape : Spec.Shape := []) {d channels : Nat}
    (spatial : Tensor Nat [d]) (groups : Nat)
    (eps : Rat := 1e-5) (affine bias : Bool := true) :
    Sequential
      (batchShape.concat ((spatial.to Spec.Shape).prependDim channels))
      (batchShape.concat ((spatial.to Spec.Shape).prependDim channels)) := by
  let shape :=
    batchShape.concat ((spatial.to Spec.Shape).prependDim channels)
  if hBatch : batchShape.size = 0 then
    exact nn.Internal.invalidConfiguration shape shape "GroupNorm"
      "GroupNorm: batch shape must contain at least one element"
  else if hChannels : channels = 0 then
    exact nn.Internal.invalidConfiguration shape shape "GroupNorm"
      "GroupNorm: channel count must be positive"
  else if hSpatial : spatial.prod = 0 then
    exact nn.Internal.invalidConfiguration shape shape "GroupNorm"
      "GroupNorm: spatial shape must contain at least one element"
  else if hGroups : groups = 0 then
    exact nn.Internal.invalidConfiguration shape shape "GroupNorm"
      "GroupNorm: group count must be positive"
  else if hGroupsLe : groups ≤ channels then
    if hDiv : channels % groups = 0 then
      let spatialShape := spatial.to Spec.Shape
      let batch := batchShape.size
      have hSpatialSize : spatialShape.size ≠ 0 := by
        simpa only [spatialShape, Tensor.size_to_shape] using hSpatial
      let hInput :
          ((spatialShape.prependDim channels).prependDim batch).wellFormed :=
        ⟨Nat.pos_of_ne_zero (by simpa [batch] using hBatch),
          Nat.pos_of_ne_zero hChannels,
          Spec.Shape.wellFormed_of_size_pos (Nat.pos_of_ne_zero hSpatialSize)⟩
      exact Sequential.fromLayer <| adaptLeadingShape batchShape <|
        Runtime.Autograd.Model.Layers.groupNorm batch channels groups spatialShape
          hInput (Nat.pos_of_ne_zero hGroups) hGroupsLe hDiv
          (eps := eps) (affine := affine) (bias := bias)
    else
      exact nn.Internal.invalidConfiguration shape shape "GroupNorm"
        "GroupNorm: channel count must be divisible by group count"
  else
    exact nn.Internal.invalidConfiguration shape shape "GroupNorm"
      "GroupNorm: group count cannot exceed channel count"

/-!
## Attention
-/

/--
Multi-head self-attention over a trailing `(sequenceLength × modelWidth)` shape.

If `mask` is provided, it is a boolean attention mask of shape `(n × n)` (e.g. causal masking).
-/
def multiHeadAttention (batchShape : Spec.Shape := []) {sequenceLength modelWidth : Nat}
    (config : MultiHeadAttention.Config)
    (queryWeightSeed keyWeightSeed valueWeightSeed outputWeightSeed : Nat := 0)
    (mask : Option (Tensor Bool [sequenceLength, sequenceLength]) := none)
    (dropoutSeed : Nat := 0) :
    Sequential
      (batchShape.concat [sequenceLength, modelWidth])
      (batchShape.concat [sequenceLength, modelWidth]) :=
  if hSequence : sequenceLength = 0 then
    nn.Internal.invalidConfiguration
      (batchShape.concat [sequenceLength, modelWidth])
      (batchShape.concat [sequenceLength, modelWidth])
      "MultiHeadAttention"
      "MultiHeadAttention: sequence length must be positive"
  else if config.inputBias || config.dropout?.isSome then
    Sequential.fromLayer <| adaptLeadingShape batchShape <|
      Runtime.Autograd.Model.Layers.multiHeadAttentionConfigured
        batchShape.size sequenceLength modelWidth config.headCount config.headWidth
        (sequenceLengthNonzero := hSequence)
        (queryWeightSeed := queryWeightSeed)
        (keyWeightSeed := keyWeightSeed)
        (valueWeightSeed := valueWeightSeed)
        (outputWeightSeed := outputWeightSeed)
        (weightInitialization? := config.weightInitialization?)
        (outputWeightInitialization? := config.outputWeightInitialization?)
        (mask := mask) (inputBias := config.inputBias) (outputBias := config.outputBias)
        (dropout? := config.dropout?) (dropoutSeed := dropoutSeed)
  else if config.outputBias then
      Sequential.fromLayer <| adaptLeadingShape batchShape <|
        Runtime.Autograd.Model.Layers.multiHeadAttentionOutputBias
        batchShape.size sequenceLength modelWidth
        config.headCount config.headWidth
        (sequenceLengthNonzero := hSequence)
        (queryWeightSeed := queryWeightSeed)
        (keyWeightSeed := keyWeightSeed)
        (valueWeightSeed := valueWeightSeed)
        (outputWeightSeed := outputWeightSeed)
        (weightInitialization? := config.weightInitialization?)
        (outputWeightInitialization? := config.outputWeightInitialization?) (mask := mask)
  else
    Sequential.fromLayer <| adaptLeadingShape batchShape <|
      Runtime.Autograd.Model.Layers.multiHeadAttention
      batchShape.size sequenceLength modelWidth
      config.headCount config.headWidth
      (sequenceLengthNonzero := hSequence)
      (queryWeightSeed := queryWeightSeed)
      (keyWeightSeed := keyWeightSeed)
      (valueWeightSeed := valueWeightSeed)
      (outputWeightSeed := outputWeightSeed)
      (weightInitialization? := config.weightInitialization?)
      (outputWeightInitialization? := config.outputWeightInitialization?) (mask := mask)

/-!
## Positional encodings
-/

/--
Add learned positional embeddings to the `(sequenceLength × embeddingWidth)` suffix of a tensor.

PyTorch analogue: `x + position[:sequenceLength]` where `position` is a parameter table.
-/
def learnedPositionalEmbedding (batchShape : Spec.Shape := [])
    {sequenceLength embeddingWidth : Nat}
    (config : LearnedPositionalEmbedding.Config := {}) (positionSeed : Nat := 0) :
    Sequential
      (batchShape.concat [sequenceLength, embeddingWidth])
      (batchShape.concat [sequenceLength, embeddingWidth]) :=
  let posShape : Spec.Shape := [sequenceLength, embeddingWidth]
  let xShape := batchShape.concat posShape
  let pos0 : Tensor Float posShape :=
    Runtime.Autograd.Torch.Init.tensor
      (s := posShape) (sch := config.initialization) (seed := positionSeed)
  letI : Spec.Shape.BroadcastTo posShape xShape := by
    exact ⟨Spec.Shape.CanBroadcastTo.prependTarget batchShape posShape⟩
  letI : Spec.Shape.BroadcastTo xShape xShape :=
    ⟨Spec.Shape.CanBroadcastTo.refl xShape⟩
  Sequential.fromLayer
    { kind := "LearnedPositionalEmbedding"
      stateShapes := [posShape]
      initState := TorchLean.TensorPack.singleton pos0
      runtimeInit := some (.cons
        (Runtime.Autograd.Model.Module.RuntimeInit.FloatInit.ofScheme
          config.initialization positionSeed) .nil)
      requiresGrad := #[true]
      validateConfig := config.validate sequenceLength embeddingWidth
      forward := fun _ {α} _ _ =>
        fun {m} _ _ =>
          fun pos x =>
            -- Broadcast the positional table across `batchShape`.
            (Runtime.Autograd.Model.F.addB (m := m) (α := α)
              (s₁ := posShape) (s₂ := xShape) (t := xShape) pos x)
    }

/--
Add sinusoidal positional encodings to the `(sequenceLength × embeddingWidth)` suffix of a tensor.

Implementation:
- precompute `PE : (sequenceLength × embeddingWidth)` at initialization time
  (stored as a non-trainable buffer),
- broadcast it across `batchShape` and add it to the input.
-/
def sinusoidalPositionalEncoding (batchShape : Spec.Shape := [])
    {sequenceLength embeddingWidth : Nat}
    (config : SinusoidalPositionalEncoding.Config := {}) :
    Sequential
      (batchShape.concat [sequenceLength, embeddingWidth])
      (batchShape.concat [sequenceLength, embeddingWidth]) :=
  let peShape : Spec.Shape := [sequenceLength, embeddingWidth]
  let xShape := batchShape.concat peShape
  let pe0 : Tensor Float peShape :=
    Spec.sinusoidalPositionalEncodingSpec
      (α := Float) sequenceLength embeddingWidth config.startPosition
  letI : Spec.Shape.BroadcastTo peShape xShape := by
    exact ⟨Spec.Shape.CanBroadcastTo.prependTarget batchShape peShape⟩
  letI : Spec.Shape.BroadcastTo xShape xShape :=
    ⟨Spec.Shape.CanBroadcastTo.refl xShape⟩
  Sequential.fromLayer
    { kind := "SinusoidalPositionalEncoding"
      stateShapes := [peShape]
      initState := TorchLean.TensorPack.singleton pe0
      runtimeInit := some (.cons (.flat (pe0.to FloatArray)) .nil)
      requiresGrad := #[false]
      validateConfig := config.validate sequenceLength embeddingWidth
      forward := fun _ {α} _ _ =>
        fun {m} _ _ =>
          fun pe x =>
            -- Broadcast `PE : (sequenceLength × embeddingWidth)` across `batchShape`.
            (Runtime.Autograd.Model.F.addB (m := m) (α := α)
              (s₁ := peShape) (s₂ := xShape) (t := xShape) pe x)
    }

/--
Apply RoPE to the `(sequenceLength × headWidth)` suffix of a tensor.

This matches the standard identity:

$$
\operatorname{rope}(x)
  = x \odot \cos + \operatorname{rotatePairs}(x) \odot \sin
$$

where `cos` and `sin` depend only on `(pos, dim)` and broadcast across `batchShape`.

Notes:
- This layer is *differentiable* (gradients flow through the rotation), but it has no trainable
  parameters; the precomputed `cos`/`sin` tables are stored as non-trainable buffers.
- The pure spec version is in `NN.Spec.Layers.PositionalEncoding` (`Spec.ropeApplyHeadsSpec`).
-/
def rope (batchShape : Spec.Shape := []) {sequenceLength headWidth : Nat}
    (config : RotaryEmbedding.Config := {}) :
    Sequential
      (batchShape.concat [sequenceLength, headWidth])
      (batchShape.concat [sequenceLength, headWidth]) :=
  let xShape : Spec.Shape := batchShape.concat [sequenceLength, headWidth]
  let csShape : Spec.Shape := [sequenceLength, headWidth]

  -- Precompute cos/sin tables from the sequence length, head width, and starting position.
  let cos0 : Tensor Float csShape :=
    TorchLean.Tensor.stackLeading (fun (position : Fin sequenceLength) =>
      Spec.ropeCosVectorSpec
        (α := Float) (config.startPosition + position.val) headWidth)
  let sin0 : Tensor Float csShape :=
    TorchLean.Tensor.stackLeading (fun (position : Fin sequenceLength) =>
      Spec.ropeSinVectorSpec
        (α := Float) (config.startPosition + position.val) headWidth)

  -- Column permutation indices implementing pairwise swap `(0↔1, 2↔3, ...)`.
  -- When `headWidth` is odd, the last index is left unchanged.
  let permIdx : Tensor (Fin headWidth) [headWidth] :=
    TorchLean.Tensor.ofFn (fun (j : Fin headWidth) =>
      let idx := j.val
      let out : Fin headWidth :=
        if h : idx % 2 = 0 ∧ idx + 1 < headWidth then
          ⟨idx + 1, h.2⟩
        else if idx % 2 = 0 then
          j
        else
          ⟨idx - 1, Nat.lt_of_le_of_lt (Nat.sub_le idx 1) j.isLt⟩
      out)

  Sequential.fromLayer
    { kind := "RoPE"
      stateShapes := [csShape, csShape]
      initState := TorchLean.TensorPack.pair cos0 sin0
      runtimeInit := some (.cons (.flat (cos0.to FloatArray))
        (.cons (.flat (sin0.to FloatArray)) .nil))
      requiresGrad := #[false, false]
      validateConfig := config.validate sequenceLength headWidth
      forward := fun _ {α} _ _ =>
        fun {m} _ _ =>
          fun cos sin x =>
            ((do
            -- Rotate last-dim pairs by a fixed 2D permutation/sign pattern.
            let rowsFold : Nat := batchShape.size * sequenceLength
            let flatShape : Spec.Shape := [rowsFold, headWidth]

            let x2d ←
              Runtime.Autograd.Torch.reshape (m := m) (α := α)
                (s₁ := xShape) (s₂ := flatShape)
                x (by
                  simp [xShape, flatShape, rowsFold, Spec.Shape.size_concat,
                    Spec.Shape.size, Nat.mul_assoc])

            let xT ←
              Runtime.Autograd.Torch.swapAdjacentAtDepth (m := m) (α := α)
                (s := flatShape) 0 x2d

            let xPerm ←
              Runtime.Autograd.Torch.indexSelect (m := m) (α := α)
                (s := [headWidth, rowsFold]) 0 headWidth xT
                (Runtime.Autograd.Torch.dataConst (m := m) (α := α) permIdx)

            let xBack ←
              Runtime.Autograd.Torch.swapAdjacentAtDepth (m := m) (α := α)
                (s := [headWidth, rowsFold]) 0 xPerm

            -- Sign pattern for `rotatePairs`: even outputs get a negation (except the final
            -- unpaired entry).
            let signT : Tensor α [headWidth] :=
              TorchLean.Tensor.ofFn (fun (j : Fin headWidth) =>
                let idx := j.val
                let value : α :=
                  if idx % 2 = 0 ∧ idx + 1 < headWidth then (-1 : α) else (1 : α)
                value)
            let sign ←
              Runtime.Autograd.Torch.const (m := m) (α := α) (s := [headWidth]) signT

            let xRot2d ←
              Runtime.Autograd.Model.F.mulB (m := m) (α := α)
                (s₁ := flatShape) (s₂ := [headWidth]) (t := flatShape)
                xBack sign

            let xRot ←
              Runtime.Autograd.Torch.reshape (m := m) (α := α)
                (s₁ := flatShape) (s₂ := xShape)
                xRot2d (by
                  simp [xShape, flatShape, rowsFold, Spec.Shape.size_concat,
                    Spec.Shape.size, Nat.mul_assoc])

            -- Apply RoPE with cosine and sine tables broadcast over `batchShape`. The broadcast
            -- proof is supplied by hand rather than through `BroadcastTo`: instance search cannot
            -- see past the let-bound shapes in this closure.
            let cosFull ←
              Runtime.Autograd.Torch.broadcastTo (m := m) (α := α)
                (s₁ := csShape) (s₂ := xShape)
                (Spec.Shape.CanBroadcastTo.prependTarget batchShape csShape) cos
            let sinFull ←
              Runtime.Autograd.Torch.broadcastTo (m := m) (α := α)
                (s₁ := csShape) (s₂ := xShape)
                (Spec.Shape.CanBroadcastTo.prependTarget batchShape csShape) sin
            let xCos ←
              Runtime.Autograd.Torch.mul (m := m) (α := α) (s := xShape) x cosFull
            let rotSin ←
              Runtime.Autograd.Torch.mul (m := m) (α := α) (s := xShape) xRot sinFull
            Runtime.Autograd.Torch.add (m := m) (α := α) (s := xShape) xCos rotSin
            ) : m (TorchLean.Runtime.ValueRef (m := m) (α := α) xShape))
    }

/-!
## Embeddings
-/

/--
Linear projection for one-hot or soft token-distribution inputs.

Input shape: `[..., vocabularySize]`
Output shape: `[..., embeddingWidth]`

This is not an indexed embedding: it multiplies the final input axis by a trainable table. Use
`embedding` for bounded token ids.
-/
def oneHotEmbedding (vocabularySize embeddingWidth : Nat)
    (config : Embedding.Config := {})
    (seed : Nat := 0)
    (batchShape : Spec.Shape := []) :
    Sequential
      (batchShape.appendDim vocabularySize)
      (batchShape.appendDim embeddingWidth) :=
  let weightShape : Spec.Shape := [vocabularySize, embeddingWidth]
  let initialWeight : Tensor Float weightShape :=
    Runtime.Autograd.Torch.Init.tensor
      (s := weightShape) (sch := config.weightInitialization) (seed := seed)
  let batch : Nat := batchShape.size
  Sequential.fromLayer
    { kind := s!"OneHotEmbedding({vocabularySize}, {embeddingWidth})"
      stateShapes := [weightShape]
      initState := TorchLean.TensorPack.singleton initialWeight
      runtimeInit := some (.cons
        (Runtime.Autograd.Model.Module.RuntimeInit.FloatInit.ofScheme
          config.weightInitialization seed) .nil)
      requiresGrad := #[!config.freeze]
      validateConfig := config.validate vocabularySize embeddingWidth
      forward := fun _ {α} _ _ =>
        fun {m} _ _ =>
          fun w x =>
            let sIn := batchShape.appendDim vocabularySize
            let sOut := batchShape.appendDim embeddingWidth
            ((do
              let x2d ←
                Runtime.Autograd.Torch.reshape (m := m) (α := α)
                  (s₁ := sIn)
                  (s₂ := [batch, vocabularySize])
                  x (by
                    simp [sIn, batch, Spec.Shape.size_appendDim, Spec.Shape.size])
              let y ←
                Runtime.Autograd.Torch.matmul (m := m) (α := α)
                  (batchA := []) (batchB := []) (batch := [])
                  (mDim := batch) (nDim := vocabularySize) (pDim := embeddingWidth)
                  x2d w
              Runtime.Autograd.Torch.reshape (m := m) (α := α)
                (s₁ := [batch, embeddingWidth])
                (s₂ := sOut)
                y (by
                  simp [sOut, batch, Spec.Shape.size_appendDim, Spec.Shape.size])
            ) : m (TorchLean.Runtime.ValueRef (m := m) (α := α) sOut))
    }

/-- Build a trainable table for bounded token ids. -/
def embedding (vocabularySize embeddingWidth : Nat) (config : Embedding.Config := {})
    (seed : Nat := 0) :
    Embedding vocabularySize embeddingWidth :=
  let weightShape : Spec.Shape := [vocabularySize, embeddingWidth]
  let initialWeight : Tensor Float weightShape :=
    Runtime.Autograd.Torch.Init.tensor
      (s := weightShape) (sch := config.weightInitialization) (seed := seed)
  Embedding.Internal.create initialWeight
    (.cons
      (Runtime.Autograd.Model.Module.RuntimeInit.FloatInit.ofScheme
        config.weightInitialization seed) .nil)
    (trainable := !config.freeze)
    (validation := config.validate vocabularySize embeddingWidth)

/-!
## Blocks and heads
-/

/-- Build a rank-polymorphic convolution/activation block. -/
def convBlock (batchShape : Spec.Shape := []) {d : Nat} {inputChannels : Nat}
    (spatial : Tensor Nat [d]) (config : ConvBlock.Config d)
    (kernelSeed dropoutSeed : Nat := 0) :
    Sequential
      (batchShape.concat ((spatial.to Spec.Shape).prependDim inputChannels))
      (batchShape.concat
        (((config.convolution.outputSpatial spatial).to Spec.Shape)
          |>.prependDim config.convolution.outChannels)) := by
  let output := batchShape.concat
    (((config.convolution.outputSpatial spatial).to Spec.Shape)
      |>.prependDim config.convolution.outChannels)
  let actLayer : Sequential output output :=
    activation (s := output) config.activation
  let core : Sequential
      (batchShape.concat ((spatial.to Spec.Shape).prependDim inputChannels))
      output := nn.compose![
    conv batchShape spatial config.convolution kernelSeed,
    actLayer]
  exact match config.dropout? with
  | none => core
  | some p =>
      let dropoutLayer : Sequential output output :=
        dropout (s := output) p (seed := dropoutSeed)
      nn.compose![core, dropoutLayer]

/--
Transformer encoder block.

With post-normalization (the default), this follows:
`LayerNorm(x + MHA(x)) -> LayerNorm(x + FFN(x))`.

With `normalizeFirst := true`, each branch is normalized before its learned transform:
`x + MHA(LayerNorm(x)) -> x + FFN(LayerNorm(x))`.

PyTorch analogue:
- `torch.nn.TransformerEncoderLayer`
  (`https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html`)
-/
def transformerEncoderBlock (batchShape : Spec.Shape := []) {sequenceLength modelWidth : Nat}
    (config : TransformerEncoder.Block.Config)
    (queryWeightSeed keyWeightSeed valueWeightSeed outputWeightSeed : Nat := 0)
    (firstFeedForwardWeightSeed secondFeedForwardWeightSeed : Nat := 0)
    (attentionDropoutSeed feedForwardDropoutSeed : Nat := 0)
    (mask : Option (Tensor Bool [sequenceLength, sequenceLength]) := none)
    (attentionProbabilityDropoutSeed feedForwardHiddenDropoutSeed : Nat := 0) :
    Sequential
      (batchShape.concat [sequenceLength, modelWidth])
      (batchShape.concat [sequenceLength, modelWidth]) := by
  let tokenBatchShape := batchShape.appendDim sequenceLength
  let modelShape := tokenBatchShape.appendDim modelWidth
  let attention : Sequential modelShape modelShape := by
    simpa [modelShape, tokenBatchShape, Spec.Shape.appendDim_appendDim_eq_concat] using
      (multiHeadAttention batchShape
        (sequenceLength := sequenceLength) (modelWidth := modelWidth)
        { headCount := config.headCount, headWidth := config.headWidth,
          outputBias := config.attentionOutputBias,
          inputBias := config.attentionInputBias,
          dropout? := config.attentionDropout?,
          weightInitialization? := config.weightInitialization?,
          outputWeightInitialization? := config.residualOutputInitialization? }
        (queryWeightSeed := queryWeightSeed)
        (keyWeightSeed := keyWeightSeed)
        (valueWeightSeed := valueWeightSeed)
        (outputWeightSeed := outputWeightSeed)
        (mask := mask) (dropoutSeed := attentionProbabilityDropoutSeed))
  let attentionBranch :=
    match config.dropout? with
    | none => attention
    | some probability =>
        nn.compose![attention,
          dropout (s := modelShape) probability (seed := attentionDropoutSeed)]
  let firstNormalization : Sequential modelShape modelShape :=
    layerNorm tokenBatchShape (width := modelWidth)

  let hiddenShape := tokenBatchShape.appendDim config.feedForwardWidth
  let hiddenDropout : Sequential hiddenShape hiddenShape :=
    match config.feedForwardDropout? with
    | none => Sequential.identity hiddenShape
    | some probability =>
        dropout (s := hiddenShape) probability (seed := feedForwardHiddenDropoutSeed)
  let feedForward : Sequential modelShape modelShape :=
    nn.compose![
      linear modelWidth config.feedForwardWidth
        firstFeedForwardWeightSeed 0
        (batchShape := tokenBatchShape)
        (config := { weightInitialization? := config.weightInitialization? }),
      activation
        (s := tokenBatchShape.appendDim config.feedForwardWidth)
        config.activation,
      hiddenDropout,
      linear config.feedForwardWidth modelWidth
        secondFeedForwardWeightSeed 0
        (batchShape := tokenBatchShape)
        (config := {
          weightInitialization? :=
            config.residualOutputInitialization?.orElse
              (fun _ => config.weightInitialization?) })]
  let feedForwardBranch :=
    match config.dropout? with
    | none => feedForward
    | some probability =>
        nn.compose![feedForward,
          dropout (s := modelShape) probability (seed := feedForwardDropoutSeed)]
  let secondNormalization : Sequential modelShape modelShape :=
    layerNorm tokenBatchShape (width := modelWidth)

  let result : Sequential modelShape modelShape :=
    if config.normalizeFirst then
      nn.compose![
        residual (nn.compose![firstNormalization, attentionBranch]),
        residual (nn.compose![secondNormalization, feedForwardBranch])]
    else
      nn.compose![
        residual attentionBranch,
        firstNormalization,
        residual feedForwardBranch,
        secondNormalization]
  simpa [modelShape, tokenBatchShape, Spec.Shape.appendDim_appendDim_eq_concat] using result

/-- Flatten the feature suffix and apply an affine map, preserving every batch axis. -/
def affineHead (batchShape : Spec.Shape := []) {featureShape : Spec.Shape}
    (outputWidth : Nat) (weightSeed : Nat := 0) :
    Sequential
      (batchShape.concat featureShape)
      (batchShape.appendDim outputWidth) :=
  nn.compose![
    flattenAfter batchShape (shape := featureShape),
    linear featureShape.size outputWidth
      (weightSeed := weightSeed) (biasSeed := 0) (batchShape := batchShape)]

end Impl
end nn
end TorchLean
