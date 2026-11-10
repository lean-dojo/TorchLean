/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Builder.Indexed
public import TorchLean.NN.Sequential.Leading
public import TorchLean.NN.Builder.Positional
public import TorchLean.NN.Builder.Transformer
public import TorchLean.NN.Layer.Mamba

/-!
# Layer Implementations

The layer constructors behind the public `nn.*` builders of `TorchLean.NN.Builder`, each taking
its initialization seeds as ordinary arguments. The public builders draw those seeds from
`nn.Builder`, are the only callers outside this file and are the supported way to construct
layers. Some model builders, FNO and `nn.moeMLP` among them, draw seeds with `nn.withSeed` and
build their seeded layers themselves.

The constructors cover leading dimensions (`adaptLeadingShape`, `batchedRecurrent`), the recurrent
layers `rnn`, `gru`, `mamba` and `lstm`, `sum`, `dropout`, convolution and `convBlock`, pooling,
normalization, `attention`, positional encodings, embeddings and `transformerBlock`. Their
bodies stay exposed so that downstream proofs can unfold a built model through the public builders.
-/

@[expose] public section

open TorchLean.Shape (size_append)

namespace TorchLean
namespace nn

/-!
## Leading dimensions
-/

/--
Reshape arbitrary leading dimensions into the single outer dimension expected by `layer`.

For an input of shape `leading ++ σ`, the layer receives shape `leading.size :: σ`. Its output is
reshaped from `leading.size :: τ` to `leading ++ τ`. The adapter reuses the layer's parameters and
buffer-update function.
-/
def Layer.adaptLeadingShape (leading : Shape) {σ τ : Shape}
    (layer : Layer (Shape.dim leading.size σ) (Shape.dim leading.size τ)) :
    Layer (leading ++ σ) (leading ++ τ) :=
  { kind := layer.kind
    stateShapes := layer.stateShapes
    initState := layer.initState
    runtimeInit := layer.runtimeInit
    requiresGrad := layer.requiresGrad
    validateConfig := layer.validateConfig
    updatesBuffersInForward := layer.updatesBuffersInForward
    updateBuffers := layer.updateBuffers.map fun update mode {α} _ _ state input =>
      update mode state <| input.reshape _ (by
        simp [Shape.size_append])
    forward := fun mode {α} _ _ =>
      fun {m} _ _ =>
        CurriedRef.curry
          (Ref := fun shape => Ops.Ref (m := m) (α := α) shape)
          (ss := layer.stateShapes ++ [leading ++ σ])
          (β := m (Ops.Ref (m := m) (α := α) (leading ++ τ)))
          (fun arguments => do
            let (state, input) :=
              TorchLean.Pack.unsnoc
                (F := fun shape =>
                  Ops.Ref (m := m) (α := α) shape)
                (ss := layer.stateShapes) (τ := leading ++ σ) arguments
            let flattenedInput ←
              Ops.apply (m := m) (α := α)
                  (.reshape (s := leading ++ σ) (t := Shape.dim leading.size σ)
                    (by simp [Shape.size_append])) (input, ())
            let flattenedOutput ←
              layer.forwardRef (α := α) (m := m) mode state flattenedInput
            Ops.apply (m := m) (α := α)
                (.reshape (s := Shape.dim leading.size τ) (t := leading ++ τ)
                  (by simp [Shape.size_append])) (flattenedOutput, ())) }

/-- Share a single-sequence recurrent core over every index of `batchShape`. -/
def Sequential.batchedRecurrent (batchShape : Shape) {sequenceLength inputWidth hiddenWidth : Nat}
    (core : Sequential [sequenceLength, inputWidth] [sequenceLength, hiddenWidth]) :
    Sequential
      (batchShape ++ [sequenceLength] ++ [inputWidth])
      (batchShape ++ [sequenceLength] ++ [hiddenWidth]) := by
  simpa only [← List.append_cons] using mapLeading batchShape core

/-!
## Recurrent layers
-/

/--
Vanilla RNN layer over a time-major sequence with no batch axis. Its hidden states are

$$
h_t=\tanh\!\left(W[x_t;h_{t-1}]+b\right),\qquad h_{-1}=0.
$$

The runtime layer unrolls the `sequenceLength` steps into TorchLean operators, so it runs on both
the CPU and the CUDA backend.

PyTorch analogy: `torch.nn.RNN(inputWidth, hiddenWidth, nonlinearity="tanh")` with
`batch_first=false`, specialized to a single batch element.
-/
def Sequential.rnn (sequenceLength inputWidth hiddenWidth : Nat)
    (weightSeed : Nat := 0) :
    Sequential
      [sequenceLength, inputWidth]
      [sequenceLength, hiddenWidth] :=
  Sequential.fromLayer
    (Layer.rnn
      (sequenceLength := sequenceLength) (inputWidth := inputWidth) (hiddenWidth := hiddenWidth)
      weightSeed)

/--
GRU layer (time-major sequence, no batch axis) with the Cho-style reset-before candidate.

The runtime layer unrolls `sequenceLength` steps into TorchLean operators, which run on both
backends. PyTorch applies the reset gate after the recurrent product, so its GRU checkpoints do
not load into this constructor, and `nn.gru` with `convention := .resetAfter` builds the PyTorch
recurrence.
-/
def Sequential.gru (sequenceLength inputWidth hiddenWidth : Nat)
    (resetWeightSeed updateWeightSeed candidateWeightSeed : Nat := 0) :
    Sequential
      [sequenceLength, inputWidth]
      [sequenceLength, hiddenWidth] :=
  Sequential.fromLayer
    (Layer.gru
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
def Sequential.mamba (sequenceLength inputWidth hiddenWidth : Nat)
    (inputWeightSeed stateWeightSeed gateWeightSeed : Nat := 0)
    (options : Mamba.Options := {}) :
    Sequential
      [sequenceLength, inputWidth]
      [sequenceLength, hiddenWidth] :=
  Sequential.fromLayer
    (Layer.mamba
      (sequenceLength := sequenceLength) (inputWidth := inputWidth) (hiddenWidth := hiddenWidth)
      inputWeightSeed stateWeightSeed gateWeightSeed (options := options))

/--
LSTM layer (time-major sequence, no batch axis).

The runtime layer unrolls the standard four-gate recurrence from $(h_{-1},c_{-1})=(0,0)$ into
`sequenceLength` steps of TorchLean operators, which run on both backends.

PyTorch analogy: `torch.nn.LSTM(inputWidth, hiddenWidth)` with `batch_first=false`, specialized to
a single batch element.
-/
def Sequential.lstm (sequenceLength inputWidth hiddenWidth : Nat)
    (forgetWeightSeed inputWeightSeed candidateWeightSeed outputWeightSeed : Nat := 0) :
    Sequential
      [sequenceLength, inputWidth]
      [sequenceLength, hiddenWidth] :=
  Sequential.fromLayer
    (Layer.lstm
      (sequenceLength := sequenceLength) (inputWidth := inputWidth) (hiddenWidth := hiddenWidth)
      forgetWeightSeed inputWeightSeed candidateWeightSeed outputWeightSeed)

/-!
## Sum and dropout layers
-/

/-- Reduce-sum to a scalar. PyTorch analogue: `torch.sum`. -/
def Sequential.sum {s : Shape} : Sequential s [] :=
  Sequential.fromLayer <| Layer.sum (s := s)

/--
Dropout layer (active in train mode, identity in eval mode).

PyTorch analogue: `torch.nn.Dropout`.
-/
def Sequential.dropout {s : Shape} (p : Float) (seed : Nat := 0) : Sequential s s :=
  Sequential.fromLayer <| Layer.dropout (s := s) p seed

/-!
## Convolution and pooling
-/

/--
Apply an arbitrary-dimensional convolution to the channel and spatial suffix of a tensor.

The input suffix is `(inputChannels, spatial...)`. The axes of `batchShape` are flattened into one
runtime batch for the convolution and restored afterwards.
-/
def Sequential.conv (batchShape : Shape := []) {d : Nat} {inputChannels : Nat}
    (spatial : Tensor Nat [d]) (config : Conv.Config d)
    (kernelSeed : Nat := 0) :
    Sequential
      (batchShape ++ (Shape.dim inputChannels (spatial.to Shape)))
      (batchShape ++
        (Shape.dim config.outChannels ((config.convOutputSpatial spatial).to Shape))) :=
  nn.Sequential.fromLayer <| Layer.adaptLeadingShape batchShape <|
      Layer.conv
        batchShape.size d
        inputChannels config.outChannels
        config.kernelSize config.stride config.padding spatial
        kernelSeed config.weightInit

/--
Apply an arbitrary-dimensional transpose convolution to the channel and spatial suffix.

The input suffix is `(inputChannels, spatial...)`. As in `nn.Sequential.conv`, the axes of
`batchShape` are flattened into one runtime batch and restored after the operation.
-/
def Sequential.convTranspose (batchShape : Shape := []) {d : Nat} {inputChannels : Nat}
    (spatial : Tensor Nat [d]) (config : Conv.Config d)
    (kernelSeed : Nat := 0) :
    Sequential
      (batchShape ++ (Shape.dim inputChannels (spatial.to Shape)))
      (batchShape ++
        (Shape.dim config.outChannels ((config.transposedOutputSpatial spatial).to Shape))) :=
  nn.Sequential.fromLayer <| Layer.adaptLeadingShape batchShape <|
      Layer.convTranspose
        batchShape.size d
        inputChannels config.outChannels
        config.kernelSize config.stride config.padding spatial
        kernelSeed config.weightInit

/-- Apply max pooling to the channel and spatial suffix of a tensor. -/
def Sequential.maxPool (batchShape : Shape := []) {d channels : Nat} (spatial : Tensor Nat [d])
    (config : Window d) :
    Sequential
      (batchShape ++ (Shape.dim channels (spatial.to Shape)))
      (batchShape ++
        (Shape.dim channels ((config.poolOutputSpatial spatial).to Shape))) :=
  nn.Sequential.fromLayer <| Layer.adaptLeadingShape batchShape <|
      Layer.maxPool
        batchShape.size d channels
        config.kernelSize config.stride config.padding spatial

/-- Apply average pooling to the channel and spatial suffix of a tensor. -/
def Sequential.avgPool (batchShape : Shape := []) {d channels : Nat} (spatial : Tensor Nat [d])
    (config : Window d) :
    Sequential
      (batchShape ++ (Shape.dim channels (spatial.to Shape)))
      (batchShape ++
        (Shape.dim channels ((config.poolOutputSpatial spatial).to Shape))) :=
  nn.Sequential.fromLayer <| Layer.adaptLeadingShape batchShape <|
      Layer.avgPool
        batchShape.size d channels
        config.kernelSize config.stride config.padding spatial

/-- Global average pooling over every spatial axis, preserving the batch axes and channels. -/
def Sequential.globalAvgPool (batchShape : Shape := []) {d channels : Nat}
    (spatial : Tensor Nat [d]) :
    Sequential
      (batchShape ++ (Shape.dim channels (spatial.to Shape)))
      (batchShape ++ [channels]) := by
  let spatialShape := spatial.to Shape
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
            functional.mean
              (m := m) (α := α) (s := spatialShape) x }
  let pooled :=
    mapLeading (batchShape ++ [channels]) (Sequential.fromLayer meanSpatial)
  simpa [spatialShape, List.append_assoc] using pooled

/-!
## Normalization
-/

/--
Layer normalization over the final axis of a tensor.

Every index in `batchShape` selects one vector of length `width`. Its entries share a mean and
variance, while scale and bias are shared across all leading indices. `batchShape := []` describes
a single vector, and zero-sized leading axes are also allowed. Only `width` must be positive.

`eps` is added to the variance before the square root. Setting `bias := false` keeps only the
learned scale, and `affine := false` removes both learned parameters.
-/
def Sequential.layerNorm (batchShape : Shape := []) {width : Nat}
    (eps : Rat := 1e-5) (affine bias : Bool := true) :
    Sequential (batchShape ++ [width]) (batchShape ++ [width]) :=
  if hWidth : width = 0 then
    nn.Sequential.invalid
      (batchShape ++ [width]) (batchShape ++ [width])
      "LayerNorm" "LayerNorm: normalized width must be positive"
  else
    Sequential.fromLayer <| Layer.layerNorm
      batchShape width (hWidth := Nat.pos_of_ne_zero hWidth)
      (eps := eps) (affine := affine) (bias := bias)

/--
Divide each final-axis vector by `sqrt(mean(x * x) + eps)`, then apply a learned scale.

The scale has shape `[width]` and is shared across all leading indices. Setting `affine := false`
removes it from model state. The default `eps` is `1e-5`, independent of the scalar type.
-/
def Sequential.rmsNorm (batchShape : Shape := []) {width : Nat}
    (eps : Rat := 1e-5) (affine : Bool := true) :
    Sequential (batchShape ++ [width]) (batchShape ++ [width]) :=
  if hWidth : width = 0 then
    nn.Sequential.invalid
      (batchShape ++ [width]) (batchShape ++ [width])
      "RMSNorm" "RMSNorm: normalized width must be positive"
  else
    Sequential.fromLayer <| Layer.rmsNorm
      batchShape width (hWidth := Nat.pos_of_ne_zero hWidth) (eps := eps) (affine := affine)

/--
Batch normalization over `(batchShape..., channels, spatial...)` for any spatial rank.

All leading batch axes and spatial axes contribute to each channel's training statistics.
Evaluation uses the running mean and variance. `momentum` controls their moving-average updates,
and `eps` is added to the variance before taking the square root.
-/
def Sequential.batchNorm (batchShape : Shape := []) {d channels : Nat}
    (spatial : Tensor Nat [d]) (momentum : Float := 0.1) (eps : Rat := 1e-5) :
    Sequential
      (batchShape ++ (Shape.dim channels (spatial.to Shape)))
      (batchShape ++ (Shape.dim channels (spatial.to Shape))) := by
  let shape :=
    batchShape ++ (Shape.dim channels (spatial.to Shape))
  if hBatch : batchShape.size = 0 then
    exact nn.Sequential.invalid shape shape "BatchNorm"
      "BatchNorm: batch shape must contain at least one element"
  else if hChannels : channels = 0 then
    exact nn.Sequential.invalid shape shape "BatchNorm"
      "BatchNorm: channel count must be positive"
  else if hSpatial : spatial.prod = 0 then
    exact nn.Sequential.invalid shape shape "BatchNorm"
      "BatchNorm: spatial shape must contain at least one element"
  else
  let spatialShape := spatial.to Shape
  let batch := batchShape.size
  have hSpatialSize : spatialShape.size ≠ 0 := by
    simpa only [spatialShape, Tensor.size_to_shape] using hSpatial
  let hInput :
      (Shape.dim batch (Shape.dim channels spatialShape)).Positive :=
    ⟨Nat.pos_of_ne_zero (by simpa [batch] using hBatch),
      Nat.pos_of_ne_zero hChannels,
      Shape.positive_of_size_pos (Nat.pos_of_ne_zero hSpatialSize)⟩
  exact Sequential.fromLayer <| Layer.adaptLeadingShape batchShape <|
    Layer.batchNorm batch channels spatialShape hInput
      (momentum := momentum) (eps := eps)

/--
Instance normalization over `(batchShape..., channels, spatial...)` for any spatial rank.

Each sample and channel uses its own spatial mean and variance in both training and evaluation.
Scale and bias are shared across samples. Setting `bias := false` keeps only the scale, while
`affine := false` removes both parameters.
-/
def Sequential.instanceNorm (batchShape : Shape := []) {d channels : Nat}
    (spatial : Tensor Nat [d]) (eps : Rat := 1e-5) (affine bias : Bool := true) :
    Sequential
      (batchShape ++ (Shape.dim channels (spatial.to Shape)))
      (batchShape ++ (Shape.dim channels (spatial.to Shape))) := by
  let shape :=
    batchShape ++ (Shape.dim channels (spatial.to Shape))
  if hBatch : batchShape.size = 0 then
    exact nn.Sequential.invalid shape shape "InstanceNorm"
      "InstanceNorm: batch shape must contain at least one element"
  else if hChannels : channels = 0 then
    exact nn.Sequential.invalid shape shape "InstanceNorm"
      "InstanceNorm: channel count must be positive"
  else if hSpatial : spatial.prod = 0 then
    exact nn.Sequential.invalid shape shape "InstanceNorm"
      "InstanceNorm: spatial shape must contain at least one element"
  else
  let spatialShape := spatial.to Shape
  let batch := batchShape.size
  have hSpatialSize : spatialShape.size ≠ 0 := by
    simpa only [spatialShape, Tensor.size_to_shape] using hSpatial
  let hInput :
      (Shape.dim batch (Shape.dim channels spatialShape)).Positive :=
    ⟨Nat.pos_of_ne_zero (by simpa [batch] using hBatch),
      Nat.pos_of_ne_zero hChannels,
      Shape.positive_of_size_pos (Nat.pos_of_ne_zero hSpatialSize)⟩
  exact Sequential.fromLayer <| Layer.adaptLeadingShape batchShape <|
    Layer.instanceNorm batch channels spatialShape hInput
      (eps := eps) (affine := affine) (bias := bias)

/--
Group normalization over `(batchShape..., channels, spatial...)` for any spatial rank.

All extents must be positive, and `groups` must be positive and divide `channels`. Within each
sample, each group shares a mean and variance across its channels and spatial positions. Scale and
bias have one entry per channel. Setting `bias := false` keeps only the scale, and
`affine := false` removes both parameters.
-/
def Sequential.groupNorm (batchShape : Shape := []) {d channels : Nat}
    (spatial : Tensor Nat [d]) (groups : Nat)
    (eps : Rat := 1e-5) (affine bias : Bool := true) :
    Sequential
      (batchShape ++ (Shape.dim channels (spatial.to Shape)))
      (batchShape ++ (Shape.dim channels (spatial.to Shape))) := by
  let shape :=
    batchShape ++ (Shape.dim channels (spatial.to Shape))
  if hBatch : batchShape.size = 0 then
    exact nn.Sequential.invalid shape shape "GroupNorm"
      "GroupNorm: batch shape must contain at least one element"
  else if hChannels : channels = 0 then
    exact nn.Sequential.invalid shape shape "GroupNorm"
      "GroupNorm: channel count must be positive"
  else if hSpatial : spatial.prod = 0 then
    exact nn.Sequential.invalid shape shape "GroupNorm"
      "GroupNorm: spatial shape must contain at least one element"
  else if hGroups : groups = 0 then
    exact nn.Sequential.invalid shape shape "GroupNorm"
      "GroupNorm: group count must be positive"
  else if hGroupsLe : groups ≤ channels then
    if hDiv : channels % groups = 0 then
      let spatialShape := spatial.to Shape
      let batch := batchShape.size
      have hSpatialSize : spatialShape.size ≠ 0 := by
        simpa only [spatialShape, Tensor.size_to_shape] using hSpatial
      let hInput :
          (Shape.dim batch (Shape.dim channels spatialShape)).Positive :=
        ⟨Nat.pos_of_ne_zero (by simpa [batch] using hBatch),
          Nat.pos_of_ne_zero hChannels,
          Shape.positive_of_size_pos (Nat.pos_of_ne_zero hSpatialSize)⟩
      exact Sequential.fromLayer <| Layer.adaptLeadingShape batchShape <|
        Layer.groupNorm batch channels groups spatialShape
          hInput (Nat.pos_of_ne_zero hGroups) hGroupsLe hDiv
          (eps := eps) (affine := affine) (bias := bias)
    else
      exact nn.Sequential.invalid shape shape "GroupNorm"
        "GroupNorm: channel count must be divisible by group count"
  else
    exact nn.Sequential.invalid shape shape "GroupNorm"
      "GroupNorm: group count cannot exceed channel count"

/-!
## Attention
-/

/--
Multi-head self-attention over a trailing `(sequenceLength × modelWidth)` shape, with a positive
`sequenceLength`.

An optional boolean `mask` of shape `(sequenceLength × sequenceLength)` lets query `i` attend to key
`j` where entry `[i, j]` is `true`. `Attention.causalMask` is the causal case, `true` for `j ≤ i`.
-/
def Sequential.attention (batchShape : Shape := []) {sequenceLength modelWidth : Nat}
    (config : Attention.Options)
    (queryWeightSeed keyWeightSeed valueWeightSeed outputWeightSeed : Nat := 0)
    (mask : Option (Tensor Bool [sequenceLength, sequenceLength]) := none)
    (dropoutSeed : Nat := 0) :
    Sequential
      (batchShape ++ [sequenceLength, modelWidth])
      (batchShape ++ [sequenceLength, modelWidth]) :=
  if hSequence : sequenceLength = 0 then
    nn.Sequential.invalid
      (batchShape ++ [sequenceLength, modelWidth])
      (batchShape ++ [sequenceLength, modelWidth])
      "MultiHeadAttention"
      "MultiHeadAttention: sequence length must be positive"
  else
    Sequential.fromLayer <| Layer.adaptLeadingShape batchShape <|
      Layer.attention
        batchShape.size sequenceLength modelWidth config.headCount config.headWidth
        (sequenceLengthNonzero := hSequence)
        (queryWeightSeed := queryWeightSeed)
        (keyWeightSeed := keyWeightSeed)
        (valueWeightSeed := valueWeightSeed)
        (outputWeightSeed := outputWeightSeed)
        (weightInitialization? := config.weightInit?)
        (outputWeightInitialization? := config.outputWeightInit?)
        (mask := mask) (inputBias := config.inputBias) (outputBias := config.outputBias)
        (dropout? := config.dropout?) (dropoutSeed := dropoutSeed)

/-!
## Positional encodings
-/

/--
Add learned positional embeddings to the `(sequenceLength × embeddingWidth)` suffix of a tensor.

PyTorch analogue: `x + position[:sequenceLength]` where `position` is a parameter table.
-/
def Sequential.learnedPositionalEmbedding (batchShape : Shape := [])
    {sequenceLength embeddingWidth : Nat}
    (config : LearnedPositionalEmbedding.Config := {}) (positionSeed : Nat := 0) :
    Sequential
      (batchShape ++ [sequenceLength, embeddingWidth])
      (batchShape ++ [sequenceLength, embeddingWidth]) :=
  let posShape : Shape := [sequenceLength, embeddingWidth]
  let xShape := batchShape ++ posShape
  let pos0 : Tensor Float posShape :=
    TorchLean.Init.tensor
      (shape := posShape) (scheme := config.init) (seed := positionSeed)
  letI : Shape.BroadcastTo posShape xShape := by
    exact ⟨Shape.canBroadcastTo_concat_right batchShape posShape⟩
  letI : Shape.BroadcastTo xShape xShape :=
    ⟨Shape.CanBroadcastTo.refl xShape⟩
  Sequential.fromLayer
    { kind := "LearnedPositionalEmbedding"
      stateShapes := [posShape]
      initState := TorchLean.Pack.singleton pos0
      runtimeInit := some (.cons
        (Init.Source.scheme
          config.init positionSeed) .nil)
      requiresGrad := #[true]
      validateConfig := config.validate sequenceLength embeddingWidth
      forward := fun _ {α} _ _ =>
        fun {m} _ _ =>
          fun pos x =>
            -- Broadcast the positional table across `batchShape`.
            (functional.addB (m := m) (α := α)
              (s₁ := posShape) (s₂ := xShape) (t := xShape) pos x)
    }

/--
Add sinusoidal positional encodings to the `(sequenceLength × embeddingWidth)` suffix of a tensor.

The table `PE : (sequenceLength × embeddingWidth)` is `Tensor.sinusoidalPositionalEncoding` with
positions offset by `config.offset`. It is computed at initialization and stored as a
non-trainable buffer, and the forward pass broadcasts it across `batchShape` and adds it to the
input.
-/
def Sequential.sinusoidalPositionalEncoding (batchShape : Shape := [])
    {sequenceLength embeddingWidth : Nat}
    (config : SinusoidalPositionalEncoding.Config := {}) :
    Sequential
      (batchShape ++ [sequenceLength, embeddingWidth])
      (batchShape ++ [sequenceLength, embeddingWidth]) :=
  let peShape : Shape := [sequenceLength, embeddingWidth]
  let xShape := batchShape ++ peShape
  let pe0 : Tensor Float peShape :=
    TorchLean.Tensor.sinusoidalPositionalEncoding peShape ⟨0, Nat.zero_lt_two⟩
      ⟨1, Nat.one_lt_two⟩ (Fin.ne_of_val_ne Nat.zero_ne_one) config.offset
  letI : Shape.BroadcastTo peShape xShape := by
    exact ⟨Shape.canBroadcastTo_concat_right batchShape peShape⟩
  letI : Shape.BroadcastTo xShape xShape :=
    ⟨Shape.CanBroadcastTo.refl xShape⟩
  Sequential.fromLayer
    { kind := "SinusoidalPositionalEncoding"
      stateShapes := [peShape]
      initState := TorchLean.Pack.singleton pe0
      runtimeInit := some (.cons (.flat (pe0.to FloatArray)) .nil)
      requiresGrad := #[false]
      validateConfig := config.validate sequenceLength embeddingWidth
      forward := fun _ {α} _ _ =>
        fun {m} _ _ =>
          fun pe x =>
            -- Broadcast `PE : (sequenceLength × embeddingWidth)` across `batchShape`.
            (functional.addB (m := m) (α := α)
              (s₁ := peShape) (s₂ := xShape) (t := xShape) pe x)
    }

/--
Apply RoPE to the `(sequenceLength × headWidth)` suffix of a tensor.

The layer computes the standard identity

$$
\operatorname{rope}(x)
  = x \odot \cos + \operatorname{rotatePairs}(x) \odot \sin
$$

where `cos` and `sin` depend on the position, offset by `config.offset`, and on the feature
pair, and broadcast across `batchShape`. Gradients flow through the rotation, but the layer has no
trainable parameters: the precomputed `cos` and `sin` tables are non-trainable buffers. The pure
spec version is `Tensor.rope` in `TorchLean.Tensor.Layers.PositionalEncoding`.

At an odd `headWidth` the rotation pairs the last feature with a zero, which the sign pattern makes
`-0`, so with the factors `1` and `0` the feature comes out as `x * 1 + (-0) * 0 = x`, infinities
and NaN included, as `Tensor.rope` returns it.
-/
def Sequential.rope (batchShape : Shape := []) {sequenceLength headWidth : Nat}
    (config : RotaryEmbedding.Config := {}) :
    Sequential
      (batchShape ++ [sequenceLength, headWidth])
      (batchShape ++ [sequenceLength, headWidth]) :=
  let xShape : Shape := batchShape ++ [sequenceLength, headWidth]
  let angleShape : Shape := [sequenceLength, headWidth]

  -- Precompute cos/sin tables from the sequence length, head width, and starting position.
  let cosineValues : Tensor Float angleShape :=
    TorchLean.Tensor.ropeCos angleShape ⟨0, Nat.zero_lt_two⟩ ⟨1, Nat.one_lt_two⟩
      (Fin.ne_of_val_ne Nat.zero_ne_one) config.offset
  let sineValues : Tensor Float angleShape :=
    TorchLean.Tensor.ropeSin angleShape ⟨0, Nat.zero_lt_two⟩ ⟨1, Nat.one_lt_two⟩
      (Fin.ne_of_val_ne Nat.zero_ne_one) config.offset

  -- The row of each feature's partner in the features with a zero row appended: the pairwise
  -- swap `(0↔1, 2↔3, ...)`, and the zero row `headWidth` for an unpaired last feature.
  let pairIndices : Tensor (Fin (headWidth + 1)) [headWidth] :=
    TorchLean.Tensor.ofFn (fun (j : Fin headWidth) =>
      let idx := j.val
      let out : Fin (headWidth + 1) :=
        if h : idx % 2 = 0 ∧ idx + 1 < headWidth then
          ⟨idx + 1, Nat.lt_succ_of_lt h.2⟩
        else if idx % 2 = 0 then
          Fin.last headWidth
        else
          ⟨idx - 1, Nat.lt_succ_of_lt (Nat.lt_of_le_of_lt (Nat.sub_le idx 1) j.isLt)⟩
      out)

  Sequential.fromLayer
    { kind := "RoPE"
      stateShapes := [angleShape, angleShape]
      initState := TorchLean.Pack.pair cosineValues sineValues
      runtimeInit := some (.cons (.flat (cosineValues.to FloatArray))
        (.cons (.flat (sineValues.to FloatArray)) .nil))
      requiresGrad := #[false, false]
      validateConfig := config.validate sequenceLength headWidth
      forward := fun _ {α} _ _ =>
        fun {m} _ _ =>
          fun cos sin x =>
            ((do
            -- Rotate adjacent feature pairs after flattening the batch and sequence axes.
            let rowCount : Nat := batchShape.size * sequenceLength
            let flatShape : Shape := [rowCount, headWidth]

            let flatInput ←
              Ops.apply (m := m) (α := α)
                  (.reshape (s := xShape) (t := flatShape) (by
                  simp [xShape, flatShape, rowCount, Shape.size_append, Nat.mul_assoc])) (x,
                      ())

            let transposedInput ←
              Ops.apply (m := m) (α := α)
                  (.permute (s := flatShape) (Shape.adjacentSwap flatShape 0)
                    (Shape.swapAdjacentAtDepth flatShape 0) (by tensor_shape)) (flatInput, ())

            -- A zero row after the features: the partner of an unpaired last feature.
            let zeroRow ← Ops.const (m := m) (α := α) (s := [1, rowCount])
              (TorchLean.Tensor.full [1, rowCount] (0 : α))
            let paddedInput : Ops.Ref (m := m) (α := α)
                [headWidth + 1, rowCount] ←
              Ops.apply (m := m) (α := α)
                  (.cat (s := .dim 0 [rowCount]) ⟨0, by simp⟩ [headWidth, 1])
                  (transposedInput, zeroRow, ())

            -- The rows of the padded input in the order `pairIndices`: a gather along axis 0.
            let permutedInput : Ops.Ref (m := m) (α := α)
                [headWidth, rowCount] ←
              Ops.Ref.cast (m := m) (α := α)
                  (by rw [Shape.eraseAxes_selectBatch, List.append_nil]; rfl) <$>
                Ops.apply (m := m) (α := α)
                  (.gather (Shape.selectDims [headWidth + 1, rowCount]
                      ⟨0, Shape.AxisInBounds.proof⟩)
                    (Shape.selectBatch [headWidth + 1, rowCount]
                      ⟨0, Shape.AxisInBounds.proof⟩ headWidth)
                    (.select [headWidth + 1, rowCount] ⟨0, Shape.AxisInBounds.proof⟩
                      headWidth))
                  (paddedInput, Ops.dataConst (m := m) (α := α)
                    (TorchLean.Tensor.map (fun tuple => tuple ⟨0, Nat.one_pos⟩)
                      (TorchLean.Tensor.selectIndices (s := [headWidth + 1, rowCount])
                        ⟨0, Shape.AxisInBounds.proof⟩ pairIndices)), ())

            let pairedInput ←
              Ops.apply (m := m) (α := α)
                  (.permute (s := [headWidth, rowCount])
                    (Shape.adjacentSwap [headWidth, rowCount] 0)
                    (Shape.swapAdjacentAtDepth [headWidth, rowCount] 0) (by tensor_shape))
                  (permutedInput, ())

            -- Sign pattern of the pair rotation: even outputs get a negation, which makes the zero
            -- partner of an unpaired last feature `-0`.
            let signValues : Tensor α [headWidth] :=
              TorchLean.Tensor.ofFn (fun (j : Fin headWidth) =>
                if j.val % 2 = 0 then (-1 : α) else (1 : α))
            let sign ←
              Ops.const (m := m) (α := α) (s := [headWidth]) signValues

            let flatRotated ←
              functional.mulB (m := m) (α := α)
                (s₁ := flatShape) (s₂ := [headWidth]) (t := flatShape)
                pairedInput sign

            let rotatedInput ←
              Ops.apply (m := m) (α := α)
                  (.reshape (s := flatShape) (t := xShape) (by
                  simp [xShape, flatShape, rowCount, Shape.size_append,
                      Nat.mul_assoc])) (flatRotated, ())

            -- Apply RoPE with cosine and sine tables broadcast over `batchShape`. The broadcast
            -- proof is written out by hand because instance search for `BroadcastTo` cannot see
            -- past the let-bound shapes in this closure.
            let cosFull ←
              Ops.apply (m := m) (α := α)
                  (.broadcast (s := angleShape) (t := xShape)
                    (Shape.canBroadcastTo_concat_right batchShape angleShape).dims
                    (Shape.canBroadcastTo_concat_right batchShape angleShape).broadcastDims)
                  (cos, ())
            let sinFull ←
              Ops.apply (m := m) (α := α)
                  (.broadcast (s := angleShape) (t := xShape)
                    (Shape.canBroadcastTo_concat_right batchShape angleShape).dims
                    (Shape.canBroadcastTo_concat_right batchShape angleShape).broadcastDims)
                  (sin, ())
            let xCos ←
              Ops.apply (m := m) (α := α) (.mul (s := xShape))
                  (x, cosFull, ())
            let rotSin ←
              Ops.apply (m := m) (α := α) (.mul (s := xShape))
                  (rotatedInput, sinFull, ())
            Ops.apply (m := m) (α := α) (.add (s := xShape))
                (xCos, rotSin, ())
            ) : m (Ops.Ref (m := m) (α := α) xShape))
    }

/-!
## Embeddings
-/

/--
Linear projection for one-hot or soft token-distribution inputs.

Input shape: `[..., vocabularySize]`
Output shape: `[..., embeddingWidth]`

The layer multiplies the final input axis by a `[vocabularySize, embeddingWidth]` table, trainable
unless `config.freeze` is set, and `Embedding.fromSeed` is the indexed form for bounded token ids.
-/
def Sequential.oneHotEmbedding (vocabularySize embeddingWidth : Nat)
    (config : Embedding.Config := {})
    (seed : Nat := 0)
    (batchShape : Shape := []) :
    Sequential
      (batchShape ++ [vocabularySize])
      (batchShape ++ [embeddingWidth]) :=
  let weightShape : Shape := [vocabularySize, embeddingWidth]
  let initialWeight : Tensor Float weightShape :=
    TorchLean.Init.tensor
      (shape := weightShape) (scheme := config.weightInit) (seed := seed)
  let batch : Nat := batchShape.size
  Sequential.fromLayer
    { kind := s!"OneHotEmbedding({vocabularySize}, {embeddingWidth})"
      stateShapes := [weightShape]
      initState := TorchLean.Pack.singleton initialWeight
      runtimeInit := some (.cons
        (Init.Source.scheme
          config.weightInit seed) .nil)
      requiresGrad := #[!config.freeze]
      validateConfig := config.validate vocabularySize embeddingWidth
      forward := fun _ {α} _ _ =>
        fun {m} _ _ =>
          fun w x =>
            let sIn := batchShape ++ [vocabularySize]
            let sOut := (batchShape ++ [embeddingWidth])
            ((do
              let flatInput ←
                Ops.apply (m := m) (α := α)
                    (.reshape (s := sIn) (t := [batch, vocabularySize]) (by
                    simp [sIn, batch])) (x, ())
              let y ←
                Ops.apply (m := m) (α := α)
                    (.matmul (batchA := []) (batchB := []) (batch := []) (m := batch)
                      (n := vocabularySize) (p := embeddingWidth) false false (fun h => nomatch h)
                      (fun h => nomatch h) Shape.BroadcastTo.proof
                      Shape.BroadcastTo.proof)
                    (flatInput, w, ())
              Ops.apply (m := m) (α := α)
                  (.reshape (s := [batch, embeddingWidth]) (t := sOut) (by
                  simp [sOut, batch])) (y, ())
            ) : m (Ops.Ref (m := m) (α := α) sOut))
    }

/-- Build an embedding table for bounded token ids, trainable unless `config.freeze` is set. -/
def Embedding.fromSeed (vocabularySize embeddingWidth : Nat) (config : Embedding.Config := {})
    (seed : Nat := 0) :
    Embedding vocabularySize embeddingWidth :=
  let weightShape : Shape := [vocabularySize, embeddingWidth]
  let initialWeight : Tensor Float weightShape :=
    TorchLean.Init.tensor
      (shape := weightShape) (scheme := config.weightInit) (seed := seed)
  Embedding.create initialWeight
    (.cons
      (Init.Source.scheme
        config.weightInit seed) .nil)
    (trainable := !config.freeze)
    (validation := config.validate vocabularySize embeddingWidth)

/-!
## Blocks and heads
-/

/-- Convolution, then the activation and optional dropout of `config`, for any spatial rank. -/
def Sequential.convBlock (batchShape : Shape := []) {d : Nat} {inputChannels : Nat}
    (spatial : Tensor Nat [d]) (config : ConvBlock.Config d)
    (kernelSeed dropoutSeed : Nat := 0) :
    Sequential
      (batchShape ++ (Shape.dim inputChannels (spatial.to Shape)))
      (batchShape ++
        (Shape.dim config.convolution.outChannels
          ((config.convolution.convOutputSpatial spatial).to Shape))) := by
  let output := batchShape ++
    (Shape.dim config.convolution.outChannels
      ((config.convolution.convOutputSpatial spatial).to Shape))
  let actLayer : Sequential output output :=
    Sequential.fromLayer
      (Layer.op config.activation.op config.activation.need_op)
  let core : Sequential
      (batchShape ++ (Shape.dim inputChannels (spatial.to Shape)))
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

The two affine layers of the feed-forward branch are arguments, so that `nn.transformerBlock`
builds them with `nn.linear` and their seeds.

PyTorch analogue:
- `torch.nn.TransformerEncoderLayer`
  (`https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html`)
-/
def Sequential.transformerBlock (batchShape : Shape := []) {sequenceLength modelWidth : Nat}
    (config : TransformerBlock.Config)
    (queryWeightSeed keyWeightSeed valueWeightSeed outputWeightSeed : Nat := 0)
    (firstFeedForward : Sequential
      (batchShape ++ [sequenceLength] ++ [modelWidth])
      (batchShape ++ [sequenceLength] ++ [config.feedForwardWidth]))
    (secondFeedForward : Sequential
      (batchShape ++ [sequenceLength] ++ [config.feedForwardWidth])
      (batchShape ++ [sequenceLength] ++ [modelWidth]))
    (attentionDropoutSeed feedForwardDropoutSeed : Nat := 0)
    (mask : Option (Tensor Bool [sequenceLength, sequenceLength]) := none)
    (attentionProbabilityDropoutSeed feedForwardHiddenDropoutSeed : Nat := 0) :
    Sequential
      (batchShape ++ [sequenceLength, modelWidth])
      (batchShape ++ [sequenceLength, modelWidth]) := by
  let tokenBatchShape := batchShape ++ [sequenceLength]
  let modelShape := tokenBatchShape ++ [modelWidth]
  let attention : Sequential modelShape modelShape := by
    simpa [modelShape, tokenBatchShape] using
      (attention batchShape
        (sequenceLength := sequenceLength) (modelWidth := modelWidth)
        { headCount := config.headCount, headWidth := config.headWidth,
          outputBias := config.attentionOutputBias,
          inputBias := config.attentionInputBias,
          dropout? := config.attentionDropout?,
          weightInit? := config.weightInit?,
          outputWeightInit? := config.residualInit? }
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

  let hiddenShape := tokenBatchShape ++ [config.feedForwardWidth]
  let hiddenDropout : Sequential hiddenShape hiddenShape :=
    match config.feedForwardDropout? with
    | none => Sequential.id hiddenShape
    | some probability =>
        dropout (s := hiddenShape) probability (seed := feedForwardHiddenDropoutSeed)
  let feedForward : Sequential modelShape modelShape :=
    nn.compose![
      firstFeedForward,
      Sequential.fromLayer
        (Layer.op
          (config.activation.op (s := tokenBatchShape ++ [config.feedForwardWidth]))
          config.activation.need_op),
      hiddenDropout,
      secondFeedForward]
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
  simpa [modelShape, tokenBatchShape] using result

end nn
end TorchLean
