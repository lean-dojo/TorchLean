/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

-- This module supplies public namespace exports used by downstream consumers. Import shaking
-- cannot see those downstream lookups, so keep the marked imports.
module -- shake: keep-downstream

public import NN.API.Rand -- shake: keep
public import NN.API.Neural.Impl -- shake: keep
public import NN.Runtime.Autograd.Model.Functional.SelectiveScan
public import NN.Runtime.Autograd.Model.Functional.Spectral
public import NN.Runtime.Autograd.Model.Functional.Fourier.Transform

@[expose] public section

namespace TorchLean

/-!
# Seeded model builders

Every public layer constructor lives here and returns `nn.Builder`, drawing deterministic
initialization seeds from an explicit seed stream. The explicit-seed implementations behind these
builders live in `NN.API.Neural.Impl`, which only this module imports; `nn.build seed` is the way to
turn a builder into a model with fixed seeds.

`nn.Sequential` lives in `Type 1` because every layer stores an execution-polymorphic forward
program, so it cannot be returned directly from `IO`. We draw a base seed in `IO`, then use
`nn.build` to construct the model purely. `nn.buildIO` returns the drawn seed in a `Built` value
that `IO` can carry, and `nn.withModel` passes the resulting model to a continuation.
-/

namespace nn

universe u

/-- Deterministic model builder that threads an explicit initialization seed stream. -/
abbrev Builder := rand.SeedM

namespace functional
export Runtime.Autograd.Model.F
  (square checkpoint
   exp sin cos log scale shift affine
   detach
   addB mulB
   embedding mean
   dropoutSeeded
   fft rfft1d irfft1d selectiveScanDiag selectiveScanDiagVar spectralConv1dRfft SpectralPath)
end functional

open Spec TorchLean

-- Recurrent builders and runtime layer constructors use the same dimension checks. Calling the
-- runtime helper here keeps validation consistent before either path allocates model state.
open Runtime.Autograd.Model.Layers.Internal (validateRecurrentDimensions)

/--
Build a value from a deterministic initialization seed.

Example:
```lean
def model : nn.Builder (nn.Sequential [2] [1]) :=
  nn.Sequential![nn.linear 2 8, nn.relu, nn.linear 8 1]

-- Same seed, same weights, on every machine and every run.
def built : nn.Sequential [2] [1] := nn.build 7 model
```
-/
def build {α : Type u} (seed : Nat) (builder : Builder α) : α :=
  let (value, _) := builder (rand.SeedStream.init seed)
  value

/-- Consume one initialization seed and continue building in the same result universe. -/
def withSeed {α : Type u} (continuation : Nat → Builder α) : Builder α :=
  fun state =>
    let (seed, nextState) := rand.SeedStream.next state
    continuation seed nextState

/-- Consume a seed exactly when an initialization scheme is stochastic. -/
def withInitializationSeed {α : Type u}
    (initialization : Init.Scheme) (continuation : Nat → Builder α) : Builder α :=
  match initialization with
  | .zeros | .ones => continuation 0
  | _ => withSeed continuation

/-- Consume a seed exactly when training dropout requires a random mask. -/
def withDropoutSeed {α : Type u}
    (probability : Float) (continuation : Nat → Builder α) : Builder α :=
  if probability == 0.0 || probability == 1.0 then
    continuation 0
  else
    withSeed continuation

/-- An absent dropout site consumes no key, just like a deterministic endpoint probability. -/
def withOptionalDropoutSeed {α : Type u}
    (probability? : Option Float) (continuation : Nat → Builder α) : Builder α :=
  match probability? with
  | none => continuation 0
  | some probability => withDropoutSeed probability continuation

/-!
## Layer constructors

Each builder validates its configuration, consumes exactly the seeds its stochastic initializers
need, and delegates to `nn.Impl`.
-/

/--
Build global average pooling without consuming an initialization seed.

Example:
```lean
-- Average every `8 x 8` map down to one number per channel, the usual last step before a
-- classifier head.
def model : nn.Builder (nn.Sequential [3, 8, 8] [3]) :=
  nn.globalAvgPool [8, 8] (channels := 3)
```
-/
def globalAvgPool {d channels : Nat}
    (spatial : Tensor Nat [d]) (batchShape : Shape := []) :
    Builder (Sequential
      (batchShape.concat ((spatial.to Shape).prependDim channels))
      (batchShape.appendDim channels)) :=
  pure <| Impl.globalAvgPool batchShape (channels := channels) spatial

namespace heads

/-- Build a classification head that flattens the feature suffix and seeds its weight. -/
def classifier {featureShape : Shape} (classCount : Nat)
    (batchShape : Shape := []) :
    Builder (Sequential
      (batchShape.concat featureShape)
      (batchShape.appendDim classCount)) :=
  let input := batchShape.concat featureShape
  let output := batchShape.appendDim classCount
  if featureShape.size = 0 then
    pure <| nn.Internal.invalidConfiguration input output "Classifier"
      "Classifier: feature size must be positive"
  else if classCount = 0 then
    pure <| nn.Internal.invalidConfiguration input output "Classifier"
      "Classifier: class count must be positive"
  else
    withSeed fun weightSeed =>
      pure <| Impl.affineHead batchShape classCount weightSeed

/-- Build a regression head that flattens the feature suffix and seeds its weight. -/
def regressor {featureShape : Shape} (outputWidth : Nat := 1)
    (batchShape : Shape := []) :
    Builder (Sequential
      (batchShape.concat featureShape)
      (batchShape.appendDim outputWidth)) :=
  let input := batchShape.concat featureShape
  let output := batchShape.appendDim outputWidth
  if featureShape.size = 0 then
    pure <| nn.Internal.invalidConfiguration input output "Regressor"
      "Regressor: feature size must be positive"
  else if outputWidth = 0 then
    pure <| nn.Internal.invalidConfiguration input output "Regressor"
      "Regressor: output width must be positive"
  else
    withSeed fun weightSeed =>
      pure <| Impl.affineHead batchShape outputWidth weightSeed

end heads

/-- Build an elementwise ReLU layer without consuming an initialization seed. -/
def relu {shape : Shape} : Builder (Sequential shape shape) :=
  pure (activation (s := shape) .relu)

/-- Build an elementwise SiLU layer without consuming an initialization seed. -/
def silu {shape : Shape} : Builder (Sequential shape shape) :=
  pure (activation (s := shape) .silu)

/-- Build GELU's cubic tanh approximation without consuming an initialization seed. -/
def gelu {shape : Shape} : Builder (Sequential shape shape) :=
  pure (activation (s := shape) .gelu)

/-- Build an elementwise sigmoid layer without consuming an initialization seed. -/
def sigmoid {shape : Shape} : Builder (Sequential shape shape) :=
  pure (activation (s := shape) .sigmoid)

/-- Build an elementwise hyperbolic-tangent layer without consuming an initialization seed. -/
def tanh {shape : Shape} : Builder (Sequential shape shape) :=
  pure (activation (s := shape) .tanh)

/--
Build a softmax layer along any valid tensor dimension without consuming a seed.

Example:
```lean
-- Axis `0` of a rank-one shape: a probability vector over ten classes.
def model : nn.Builder (nn.Sequential [10] [10]) :=
  nn.softmax 0
```
-/
def softmax {shape : Shape} (axis : Nat) :
    Builder (Sequential shape shape) :=
  pure (Impl.softmax (s := shape) axis)

/-- Build a stable log-softmax layer along any valid tensor dimension without consuming a seed. -/
def logSoftmax {shape : Shape} (axis : Nat) :
    Builder (Sequential shape shape) :=
  pure (Impl.logSoftmax (s := shape) axis)

/-- Build a reduction that sums every tensor entry to a scalar. -/
def sum {shape : Shape} : Builder (Sequential shape []) :=
  pure (Impl.sum (s := shape))

/-- Build a layer that flattens the entire input shape into one vector. -/
def flatten {shape : Shape} : Builder (Sequential shape [shape.size]) :=
  pure <| Impl.flatten (s := shape)

/-- Build a reshape that is rejected by model validation when the element counts differ. -/
def reshape (source target : Shape) :
    Builder (Sequential source target) :=
  pure (Impl.reshape source target)

/-- Flatten each tensor after an arbitrary batch shape. -/
def flattenAfter (batchShape : Shape := []) {shape : Shape} :
    Builder
      (Sequential
        (batchShape.concat shape)
        (batchShape.appendDim shape.size)) :=
  pure (Impl.flattenAfter batchShape (shape := shape))

/--
Build max pooling over arbitrary spatial rank using the supplied pooling configuration.

Example:
```lean
def spatial : Tensor Nat [2] := [8, 8]

def pooling : nn.Pooling.Config 2 :=
  { kernelSize := [2, 2], stride := [2, 2] }

-- Non-overlapping `2 x 2` windows halve both spatial axes and leave the channel count alone.
def model : nn.Builder (nn.Sequential [3, 8, 8] [3, 4, 4]) :=
  nn.maxPool spatial pooling (channels := 3)
```
-/
def maxPool {d channels : Nat} (spatial : Tensor Nat [d])
    (config : Pooling.Config d) (batchShape : Shape := []) :
    Builder (Sequential
      (batchShape.concat ((spatial.to Shape).prependDim channels))
      (batchShape.concat
        (((config.outputSpatial spatial).to Shape).prependDim channels))) :=
  match config.validate channels spatial (kind := "MaxPool") with
  | .error message =>
      pure <| nn.Internal.invalidConfiguration
        (batchShape.concat ((spatial.to Shape).prependDim channels))
        (batchShape.concat
          (((config.outputSpatial spatial).to Shape).prependDim channels))
        "MaxPool" message
  | .ok () =>
      pure (Impl.maxPool batchShape (channels := channels) spatial config)

/-- Build average pooling over arbitrary spatial rank using the supplied pooling configuration. -/
def avgPool {d channels : Nat} (spatial : Tensor Nat [d])
    (config : Pooling.Config d) (batchShape : Shape := []) :
    Builder (Sequential
      (batchShape.concat ((spatial.to Shape).prependDim channels))
      (batchShape.concat
        (((config.outputSpatial spatial).to Shape).prependDim channels))) :=
  match config.validate channels spatial (kind := "AvgPool") with
  | .error message =>
      pure <| nn.Internal.invalidConfiguration
        (batchShape.concat ((spatial.to Shape).prependDim channels))
        (batchShape.concat
          (((config.outputSpatial spatial).to Shape).prependDim channels))
        "AvgPool" message
  | .ok () =>
      pure (Impl.avgPool batchShape (channels := channels) spatial config)

/-- Build a transpose convolution over arbitrary spatial rank. -/
def convTranspose {d : Nat} {inputChannels : Nat}
    (spatial : Tensor Nat [d]) (config : TransposedConvolution.Config d)
    (batchShape : Shape := []) :
    Builder (Sequential
      (batchShape.concat ((spatial.to Shape).prependDim inputChannels))
      (batchShape.concat
        (((config.outputSpatial spatial).to Shape).prependDim config.outChannels))) :=
  match config.validate inputChannels spatial (kind := "ConvTranspose") with
  | .error message =>
      pure <| nn.Internal.invalidConfiguration
        (batchShape.concat ((spatial.to Shape).prependDim inputChannels))
        (batchShape.concat
          (((config.outputSpatial spatial).to Shape).prependDim config.outChannels))
        "ConvTranspose" message
  | .ok () =>
      withInitializationSeed config.weightInitialization fun kernelSeed =>
        pure <| Impl.convTranspose batchShape spatial config kernelSeed

/--
Build an affine layer, consuming seeds only for stochastic initializers.

Example:
```lean
-- Two affine layers around a ReLU: `2 -> 8 -> 1`.
def model : nn.Builder (nn.Sequential [2] [1]) :=
  nn.Sequential![
    nn.linear 2 8,
    nn.relu,
    nn.linear 8 1
  ]

-- With a batch axis in front, the same call reads `[16, 2] -> [16, 8]`.
def batched : nn.Builder (nn.Sequential [16, 2] [16, 8]) :=
  nn.linear 2 8 (batchShape := [16])
```
-/
def linear (inputWidth outputWidth : Nat) (batchShape : Shape := [])
    (config : Linear.Config := {}) :
    Builder (Sequential
      (batchShape.appendDim inputWidth)
      (batchShape.appendDim outputWidth)) :=
  match config.validate inputWidth outputWidth with
  | .error message =>
      pure <| nn.Internal.invalidConfiguration
        (batchShape.appendDim inputWidth)
        (batchShape.appendDim outputWidth)
        "Linear" message
  | .ok () =>
      let weightInitialization :=
        config.weightInitialization?.getD (.xavierUniform inputWidth outputWidth)
      withInitializationSeed weightInitialization fun weightSeed =>
        withInitializationSeed config.biasInitialization fun biasSeed =>
          pure <| Impl.linear inputWidth outputWidth weightSeed biasSeed
            (batchShape := batchShape) (config := config)

/--
Build a seeded recurrent neural network over a fixed sequence length.

The recurrent computation acts independently over every index in `batchShape`; its parameters are
shared across those indices. The scalar default is a single sequence, while a shape such as
`[batch]` gives the usual batched model.
-/
def rnn (sequenceLength inputWidth hiddenWidth : Nat) (batchShape : Shape := []) :
    Builder (Sequential
      ((batchShape.appendDim sequenceLength).appendDim inputWidth)
      ((batchShape.appendDim sequenceLength).appendDim hiddenWidth)) :=
  let input := (batchShape.appendDim sequenceLength).appendDim inputWidth
  let output := (batchShape.appendDim sequenceLength).appendDim hiddenWidth
  match validateRecurrentDimensions
      "RNN" sequenceLength inputWidth hiddenWidth with
  | .error message =>
      pure <| nn.Internal.invalidConfiguration input output "RNN" message
  | .ok () =>
      withSeed fun weightSeed =>
        pure <| Impl.batchedRecurrent batchShape
          (Impl.rnn sequenceLength inputWidth hiddenWidth weightSeed)

/--
Build a seeded GRU, shared over every index in `batchShape`.

`resetBefore` preserves the original constructor and its three initialization draws. Choose
`resetAfter` for PyTorch's recurrence and four packed parameter tensors; that version draws one
seed for each of its input and recurrent weight matrices and keeps both biases independent.
-/
def gru (sequenceLength inputWidth hiddenWidth : Nat) (batchShape : Shape := [])
    (convention : Spec.GRUConvention := .resetBefore) :
    Builder (Sequential
      ((batchShape.appendDim sequenceLength).appendDim inputWidth)
      ((batchShape.appendDim sequenceLength).appendDim hiddenWidth)) :=
  let input := (batchShape.appendDim sequenceLength).appendDim inputWidth
  let output := (batchShape.appendDim sequenceLength).appendDim hiddenWidth
  match validateRecurrentDimensions
      "GRU" sequenceLength inputWidth hiddenWidth with
  | .error message =>
      pure <| nn.Internal.invalidConfiguration input output "GRU" message
  | .ok () =>
      match convention with
      | .resetBefore =>
          withSeed fun resetWeightSeed =>
            withSeed fun updateWeightSeed =>
              withSeed fun candidateWeightSeed =>
                pure <| Impl.batchedRecurrent batchShape
                  (Impl.gru sequenceLength inputWidth hiddenWidth
                    resetWeightSeed updateWeightSeed candidateWeightSeed)
      | .resetAfter =>
          withSeed fun inputWeightSeed =>
            withSeed fun hiddenWeightSeed =>
              pure <| Impl.batchedRecurrent batchShape
                (Sequential.fromLayer <| Runtime.Autograd.Model.Layers.gruResetAfter
                  sequenceLength inputWidth hiddenWidth inputWeightSeed hiddenWeightSeed)

/-- Build a seeded selective Mamba layer, shared over every index in `batchShape`. -/
def mamba (sequenceLength inputWidth hiddenWidth : Nat) (batchShape : Shape := [])
    (options : Runtime.Autograd.Model.Mamba.Options := {}) :
    Builder (Sequential
      ((batchShape.appendDim sequenceLength).appendDim inputWidth)
      ((batchShape.appendDim sequenceLength).appendDim hiddenWidth)) :=
  let input := (batchShape.appendDim sequenceLength).appendDim inputWidth
  let output := (batchShape.appendDim sequenceLength).appendDim hiddenWidth
  match (do
      validateRecurrentDimensions "Mamba" sequenceLength inputWidth hiddenWidth
      options.validate) with
  | .error message =>
      pure <| nn.Internal.invalidConfiguration input output "Mamba" message
  | .ok () =>
      withSeed fun inputWeightSeed =>
        withSeed fun stateWeightSeed =>
          withSeed fun gateWeightSeed =>
            pure <| Impl.batchedRecurrent batchShape
              (Impl.mamba sequenceLength inputWidth hiddenWidth
                 inputWeightSeed stateWeightSeed gateWeightSeed (options := options))

/--
Build a seeded long short-term memory layer, shared over every index in `batchShape`.

Example:
```lean
-- Sixteen timesteps of width 8 in, sixteen hidden states of width 32 out.
def model : nn.Builder (nn.Sequential [16, 8] [16, 32]) :=
  nn.lstm 16 8 32
```
-/
def lstm (sequenceLength inputWidth hiddenWidth : Nat) (batchShape : Shape := []) :
    Builder (Sequential
      ((batchShape.appendDim sequenceLength).appendDim inputWidth)
      ((batchShape.appendDim sequenceLength).appendDim hiddenWidth)) :=
  let input := (batchShape.appendDim sequenceLength).appendDim inputWidth
  let output := (batchShape.appendDim sequenceLength).appendDim hiddenWidth
  match validateRecurrentDimensions
      "LSTM" sequenceLength inputWidth hiddenWidth with
  | .error message =>
      pure <| nn.Internal.invalidConfiguration input output "LSTM" message
  | .ok () =>
      withSeed fun forgetWeightSeed =>
        withSeed fun inputWeightSeed =>
          withSeed fun candidateWeightSeed =>
            withSeed fun outputWeightSeed =>
              pure <| Impl.batchedRecurrent batchShape
                (Impl.lstm sequenceLength inputWidth hiddenWidth
                  forgetWeightSeed inputWeightSeed candidateWeightSeed outputWeightSeed)

/--
Build an arbitrary-rank convolution, seeding its kernel when initialization is stochastic.

Example:
```lean
def spatial : Tensor Nat [2] := [8, 8]

def convolution : nn.Convolution.Config 2 :=
  { outChannels := 4, kernelSize := [3, 3] }

-- One `8 x 8` channel in, four `6 x 6` feature maps out: no padding, so the kernel eats a
-- one-pixel border on each side.
def model : nn.Builder (nn.Sequential [1, 8, 8] [4, 6, 6]) :=
  nn.conv spatial convolution (inputChannels := 1)
```
-/
def conv {d : Nat} {inputChannels : Nat} (spatial : Tensor Nat [d])
    (config : Convolution.Config d) (batchShape : Shape := []) :
    Builder (Sequential
      (batchShape.concat ((spatial.to Shape).prependDim inputChannels))
      (batchShape.concat
        (((config.outputSpatial spatial).to Shape).prependDim config.outChannels))) :=
  match config.validate inputChannels spatial (kind := "Conv") with
  | .error message =>
      pure <| nn.Internal.invalidConfiguration
        (batchShape.concat ((spatial.to Shape).prependDim inputChannels))
        (batchShape.concat
          (((config.outputSpatial spatial).to Shape).prependDim config.outChannels))
        "Conv" message
  | .ok () =>
      withInitializationSeed config.weightInitialization fun kernelSeed =>
        pure <| Impl.conv batchShape (inputChannels := inputChannels) spatial config
          kernelSeed

/--
Build a pointwise convolution over any spatial rank.

The unit kernel, unit stride, and zero padding preserve every spatial axis. This is the common
channel-projection operation used by residual, diffusion, and encoder-decoder models.
-/
def pointwiseConv {d : Nat} {inputChannels : Nat}
    (spatial : Tensor Nat [d]) (outputChannels : Nat) (batchShape : Shape := []) :
    Builder (Sequential
      (batchShape.concat ((spatial.to Shape).prependDim inputChannels))
      (batchShape.concat ((spatial.to Shape).prependDim outputChannels))) := by
  let config : Convolution.Config d :=
    { outChannels := outputChannels
      kernelSize := Tensor.ones [d]
      stride := Tensor.ones [d]
      padding := Tensor.zeros [d] }
  have hOut : config.outputSpatial spatial = spatial := by
    rw [Convolution.Config.outputSpatial]
    exact Spec.convOutSpatial_unit spatial
  let result :=
    conv spatial config (batchShape := batchShape) (inputChannels := inputChannels)
  rw [hOut] at result
  exact result

/--
Build batch normalization with learned scale and bias and stored running statistics.

Training computes one mean and variance per channel across the batch and spatial axes.
Evaluation uses the stored statistics. `momentum` controls how much each training batch changes
those buffers; `eps` is added to the variance before taking its square root. Scale and running
variance start at one, while bias and running mean start at zero.

Example:
```lean
-- Shape in equals shape out. What changes is the running mean and variance this layer keeps as
-- persistent buffers, updated in `.train` mode and only read in `.eval` mode.
def model : nn.Builder (nn.Sequential [3, 8, 8] [3, 8, 8]) :=
  nn.batchNorm [8, 8] (momentum := 0.1) (channels := 3)
```
-/
def batchNorm {d channels : Nat}
    (spatial : Tensor Nat [d]) (momentum : Float := 0.1)
    (batchShape : Shape := []) (eps : Rat := 1e-5) :
    Builder (Sequential
      (batchShape.concat ((spatial.to Shape).prependDim channels))
      (batchShape.concat ((spatial.to Shape).prependDim channels))) :=
  pure <| Impl.batchNorm batchShape spatial (momentum := momentum) (eps := eps)

/--
Build instance normalization using each sample and channel's spatial mean and variance.

The layer uses current input statistics in both training and evaluation. Scale and bias each
have one entry per channel and start at one and zero. Both are enabled by default; `bias := false`
keeps only the scale, and `affine := false` removes both. `eps` is added to the variance before
taking its square root.
-/
def instanceNorm {d channels : Nat}
    (spatial : Tensor Nat [d]) (batchShape : Shape := [])
    (eps : Rat := 1e-5) (affine bias : Bool := true) :
    Builder (Sequential
      (batchShape.concat ((spatial.to Shape).prependDim channels))
      (batchShape.concat ((spatial.to Shape).prependDim channels))) :=
  pure <| Impl.instanceNorm batchShape spatial (eps := eps) (affine := affine) (bias := bias)

/--
Build group normalization with equal, contiguous groups of channels.

Each sample's groups have separate means and variances, computed over their channels and spatial
positions. The channel count must be divisible by the positive group count. Scale and bias have
one entry per channel; `bias := false` keeps only the scale, and `affine := false` removes both.
`eps` is added to each group's variance before taking its square root.
-/
def groupNorm {d channels : Nat}
    (spatial : Tensor Nat [d]) (groups : Nat) (batchShape : Shape := [])
    (eps : Rat := 1e-5) (affine bias : Bool := true) :
    Builder (Sequential
      (batchShape.concat ((spatial.to Shape).prependDim channels))
      (batchShape.concat ((spatial.to Shape).prependDim channels))) :=
  pure <| Impl.groupNorm batchShape spatial groups (eps := eps) (affine := affine) (bias := bias)

/-- Build an embedding lookup layer from a freshly seeded embedding table. -/
def oneHotEmbedding (vocabularySize embeddingWidth : Nat)
    (config : Embedding.Config := {})
    (batchShape : Shape := []) :
    Builder (Sequential
      (batchShape.appendDim vocabularySize)
      (batchShape.appendDim embeddingWidth)) :=
  match config.validate vocabularySize embeddingWidth with
  | .error message =>
      pure <| nn.Internal.invalidConfiguration
        (batchShape.appendDim vocabularySize)
        (batchShape.appendDim embeddingWidth)
        "OneHotEmbedding" message
  | .ok () =>
      withInitializationSeed config.weightInitialization fun seed =>
        pure <| Impl.oneHotEmbedding vocabularySize embeddingWidth config seed
          (batchShape := batchShape)

/--
Build a trainable lookup table for a tensor of natural-number indices.

Example:
```lean
-- A vocabulary of 256 byte tokens, each mapped to a 32-dimensional row.
def table : nn.Embedding 256 32 :=
  nn.build 0 (nn.embedding 256 32)

-- `table.model` fixes the index shape: a length-16 token window becomes `[16, 32]`.
def model : nn.IndexedModel [16] [16, 32] (Fin 256) :=
  table.model [16]
```
-/
def embedding (vocabularySize embeddingWidth : Nat) (config : Embedding.Config := {}) :
    Builder (Embedding vocabularySize embeddingWidth) :=
  match config.validate vocabularySize embeddingWidth with
  | .error message =>
      pure <| Embedding.Internal.invalid vocabularySize embeddingWidth message
  | .ok () =>
      withInitializationSeed config.weightInitialization fun seed =>
        pure <| Impl.embedding vocabularySize embeddingWidth config seed

/-- Build deterministic sinusoidal positional encoding over a sequence suffix. -/
def sinusoidalPositionalEncoding (batchShape : Shape := [])
    {sequenceLength embeddingWidth : Nat}
    (config : SinusoidalPositionalEncoding.Config := {}) :
    Builder (Sequential
      (batchShape.concat [sequenceLength, embeddingWidth])
      (batchShape.concat [sequenceLength, embeddingWidth])) :=
  match config.validate sequenceLength embeddingWidth with
  | .error message =>
      let shape := batchShape.concat [sequenceLength, embeddingWidth]
      pure <| nn.Internal.invalidConfiguration
        shape shape "SinusoidalPositionalEncoding" message
  | .ok () =>
      pure <| Impl.sinusoidalPositionalEncoding batchShape
        (sequenceLength := sequenceLength) (embeddingWidth := embeddingWidth) config

/-- Build deterministic rotary positional encoding for multi-head sequence features. -/
def rope (batchShape : Shape := []) {sequenceLength headWidth : Nat}
    (config : RotaryEmbedding.Config := {}) :
    Builder (Sequential
      (batchShape.concat [sequenceLength, headWidth])
      (batchShape.concat [sequenceLength, headWidth])) :=
  match config.validate sequenceLength headWidth with
  | .error message =>
      let shape := batchShape.concat [sequenceLength, headWidth]
      pure <| nn.Internal.invalidConfiguration shape shape "RoPE" message
  | .ok () =>
      pure <| Impl.rope batchShape
        (sequenceLength := sequenceLength) (headWidth := headWidth) config

/-- Build learned positional embeddings from a freshly allocated parameter seed. -/
def learnedPositionalEmbedding (batchShape : Shape := [])
    {sequenceLength embeddingWidth : Nat}
    (config : LearnedPositionalEmbedding.Config := {}) :
    Builder (Sequential
      (batchShape.concat [sequenceLength, embeddingWidth])
      (batchShape.concat [sequenceLength, embeddingWidth])) :=
  match config.validate sequenceLength embeddingWidth with
  | .error message =>
      let shape := batchShape.concat [sequenceLength, embeddingWidth]
      pure <| nn.Internal.invalidConfiguration
        shape shape "LearnedPositionalEmbedding" message
  | .ok () =>
      withInitializationSeed config.initialization fun positionSeed =>
        pure <| Impl.learnedPositionalEmbedding batchShape
          (sequenceLength := sequenceLength) (embeddingWidth := embeddingWidth)
          config (positionSeed := positionSeed)

/--
Build layer normalization over the final axis.

Each row uses its own mean and variance. Scale and bias have shape `[width]` and are shared across
the leading axes, starting at one and zero. `eps` is added to the variance before taking its square
root. Setting `bias := false` keeps only the scale; setting `affine := false` removes both
parameters.

The rational `eps` must remain positive and finite after conversion to the execution scalar.
The default can round to zero in tiny formats, producing NaNs on constant rows in a typed graph.
Validation checks rational positivity only. Choose a representable positive value with `eps`;
for three exponent bits and two fraction bits, `(eps := (1 / 16 : Rat))` is such a value.

Example:
```lean
-- Normalizes across the final axis of each `[16, 64]` row, the Transformer convention.
def model : nn.Builder (nn.Sequential [16, 64] [16, 64]) :=
  nn.layerNorm [16] (width := 64)
```
-/
def layerNorm (batchShape : Shape := []) {width : Nat}
    (eps : Rat := 1e-5) (affine bias : Bool := true) :
    Builder (Sequential (batchShape.appendDim width) (batchShape.appendDim width)) :=
  pure <| Impl.layerNorm batchShape (width := width) (eps := eps) (affine := affine) (bias := bias)

/--
Build RMS normalization over the final axis.

Each row is divided by `sqrt(mean(x * x) + eps)`, then multiplied by a scale of shape `[width]`.
There is no mean subtraction or bias. The scale starts at one; `affine := false` removes it from
model state. The default `eps` is `1e-5` for every scalar type.
The converted epsilon must remain positive and finite. If it rounds to zero in a tiny format,
zero input can produce NaNs; pass a representable positive `eps` before lowering the model.
-/
def rmsNorm (batchShape : Shape := []) {width : Nat}
    (eps : Rat := 1e-5) (affine : Bool := true) :
    Builder (Sequential (batchShape.appendDim width) (batchShape.appendDim width)) :=
  pure <| Impl.rmsNorm batchShape (width := width) (eps := eps) (affine := affine)

/--
Build seeded multi-head self-attention with an optional fixed attention mask.

Example:
```lean
-- Two heads of width 4 give an internal attention width of 8, which here happens to match the
-- model width; the two are independent, so `headCount * headWidth` may differ from it.
def model : nn.Builder (nn.Sequential [4, 8] [4, 8]) :=
  nn.multiHeadAttention { headCount := 2, headWidth := 4 }
    (sequenceLength := 4) (modelWidth := 8)

-- Causal masking is a separate argument rather than a config field, because the mask is a value
-- with the sequence length in its type.
def causal : nn.Builder (nn.Sequential [4, 8] [4, 8]) :=
  nn.multiHeadAttention { headCount := 2, headWidth := 4 }
    (mask := some (Spec.causalMask 4)) (sequenceLength := 4) (modelWidth := 8)
```
-/
def multiHeadAttention {sequenceLength modelWidth : Nat}
    (config : MultiHeadAttention.Config)
    (mask : Option (Tensor Bool [sequenceLength, sequenceLength]) := none)
    (batchShape : Shape := []) :
    Builder (Sequential
      (batchShape.concat [sequenceLength, modelWidth])
      (batchShape.concat [sequenceLength, modelWidth])) :=
  match config.validate sequenceLength modelWidth with
  | .error message =>
      let shape := batchShape.concat [sequenceLength, modelWidth]
      pure <| nn.Internal.invalidConfiguration shape shape "MultiHeadAttention" message
  | .ok () =>
      let projectionWidth := config.headCount * config.headWidth
      let projectionInitialization :=
        config.weightInitialization?.getD (.xavierUniform modelWidth projectionWidth)
      let outputInitialization :=
        config.outputWeightInitialization?.orElse (fun _ => config.weightInitialization?)
          |>.getD (.xavierUniform projectionWidth modelWidth)
      withInitializationSeed projectionInitialization fun queryWeightSeed =>
        withInitializationSeed projectionInitialization fun keyWeightSeed =>
          withInitializationSeed projectionInitialization fun valueWeightSeed =>
            withInitializationSeed outputInitialization fun outputWeightSeed =>
              withOptionalDropoutSeed config.dropout? fun dropoutSeed =>
                pure <| Impl.multiHeadAttention batchShape
                  (sequenceLength := sequenceLength) (modelWidth := modelWidth)
                  config
                  (queryWeightSeed := queryWeightSeed)
                  (keyWeightSeed := keyWeightSeed)
                  (valueWeightSeed := valueWeightSeed)
                  (outputWeightSeed := outputWeightSeed)
                  (mask := mask) (dropoutSeed := dropoutSeed)

/--
Build one seeded transformer encoder block, optionally applying a fixed attention mask.

Example:
```lean
-- Pre-norm block, GELU feed-forward, no dropout: attention and feed-forward each sit inside their
-- own residual connection, so shapes in and out agree.
def model : nn.Builder (nn.Sequential [16, 64] [16, 64]) :=
  nn.transformerEncoderBlock
    { headCount := 4
      headWidth := 16
      feedForwardWidth := 256
      normalizeFirst := true }
    (sequenceLength := 16) (modelWidth := 64)
```
-/
def transformerEncoderBlock {sequenceLength modelWidth : Nat}
    (config : TransformerEncoder.Block.Config)
    (mask : Option (Tensor Bool [sequenceLength, sequenceLength]) := none)
    (batchShape : Shape := []) :
    Builder (Sequential
      (batchShape.concat [sequenceLength, modelWidth])
      (batchShape.concat [sequenceLength, modelWidth])) :=
  let shape := batchShape.concat [sequenceLength, modelWidth]
  let invalid (message : String) : Builder (Sequential shape shape) :=
    pure <| nn.Internal.invalidConfiguration shape shape "TransformerEncoder" message
  if sequenceLength = 0 then
    invalid "TransformerEncoder: sequence length must be positive"
  else if modelWidth = 0 then
    invalid "TransformerEncoder: model width must be positive"
  else match config.validate with
  | .error message => invalid message
  | .ok () =>
      let projectionWidth := config.headCount * config.headWidth
      let projectionInitialization :=
        config.weightInitialization?.getD (.xavierUniform modelWidth projectionWidth)
      let outputInitialization :=
        config.residualOutputInitialization?.orElse
          (fun _ => config.weightInitialization?)
          |>.getD (.xavierUniform projectionWidth modelWidth)
      let firstFeedForwardInitialization :=
        config.weightInitialization?.getD
          (.xavierUniform modelWidth config.feedForwardWidth)
      let secondFeedForwardInitialization :=
        config.residualOutputInitialization?.orElse
          (fun _ => config.weightInitialization?)
          |>.getD (.xavierUniform config.feedForwardWidth modelWidth)
      withInitializationSeed projectionInitialization fun queryWeightSeed =>
        withInitializationSeed projectionInitialization fun keyWeightSeed =>
          withInitializationSeed projectionInitialization fun valueWeightSeed =>
            withInitializationSeed outputInitialization fun outputWeightSeed =>
              withInitializationSeed firstFeedForwardInitialization
                  fun firstFeedForwardWeightSeed =>
                withInitializationSeed secondFeedForwardInitialization
                    fun secondFeedForwardWeightSeed =>
                  -- Keep the original residual-dropout keys first. Adding an opt-in site does
                  -- not change the weight initialization or the keys of those existing sites.
                  withOptionalDropoutSeed config.dropout? fun attentionDropoutSeed =>
                    withOptionalDropoutSeed config.dropout? fun feedForwardDropoutSeed =>
                      withOptionalDropoutSeed config.attentionDropout?
                          fun attentionProbabilityDropoutSeed =>
                        withOptionalDropoutSeed config.feedForwardDropout?
                            fun feedForwardHiddenDropoutSeed =>
                          pure <| Impl.transformerEncoderBlock batchShape
                            (sequenceLength := sequenceLength) (modelWidth := modelWidth)
                            config
                            (queryWeightSeed := queryWeightSeed)
                            (keyWeightSeed := keyWeightSeed)
                            (valueWeightSeed := valueWeightSeed)
                            (outputWeightSeed := outputWeightSeed)
                            (firstFeedForwardWeightSeed := firstFeedForwardWeightSeed)
                            (secondFeedForwardWeightSeed := secondFeedForwardWeightSeed)
                            (attentionDropoutSeed := attentionDropoutSeed)
                            (feedForwardDropoutSeed := feedForwardDropoutSeed)
                            (mask := mask)
                            (attentionProbabilityDropoutSeed := attentionProbabilityDropoutSeed)
                            (feedForwardHiddenDropoutSeed := feedForwardHiddenDropoutSeed)

/-- Build a seeded stack of transformer encoder blocks with an optional attention mask. -/
def transformerEncoderStack {sequenceLength modelWidth : Nat}
    (config : TransformerEncoder.Stack.Config)
    (mask : Option (Tensor Bool [sequenceLength, sequenceLength]) := none)
    (batchShape : Shape := []) :
    Builder (Sequential
      (batchShape.concat [sequenceLength, modelWidth])
      (batchShape.concat [sequenceLength, modelWidth])) :=
  let shape := batchShape.concat [sequenceLength, modelWidth]
  let invalid (message : String) : Builder (Sequential shape shape) :=
    pure <| nn.Internal.invalidConfiguration shape shape "TransformerEncoder" message
  let rec buildLayers : Nat → Builder (Sequential shape shape)
    | 0 => pure <| Sequential.identity shape
    | remaining + 1 => do
        let current ← transformerEncoderBlock config.block (mask := mask)
          (batchShape := batchShape)
          (sequenceLength := sequenceLength) (modelWidth := modelWidth)
        let rest ← buildLayers remaining
        pure <| compose current rest
  if sequenceLength = 0 then
    invalid "TransformerEncoder: sequence length must be positive"
  else if modelWidth = 0 then
    invalid "TransformerEncoder: model width must be positive"
  else match config.block.validate with
  | .error message => invalid message
  | .ok () => buildLayers config.layerCount

/--
Build dropout, consuming a key only when a random training mask is possible.

Example:
```lean
-- Active in `.train` mode and the identity in `.eval` mode, which the trainer selects for you.
def model : nn.Builder (nn.Sequential [64] [64]) :=
  nn.dropout 0.1
```
-/
def dropout {shape : Shape} (p : Float) : Builder (Sequential shape shape) :=
  if p.isFinite && 0.0 <= p && p <= 1.0 then
    withDropoutSeed p fun seed =>
      pure <| Impl.dropout (s := shape) p (seed := seed)
  else
    pure <| nn.Internal.invalidConfiguration shape shape "Dropout"
      s!"Dropout: probability must be finite and in [0, 1], got {p}"

/-- Build a seeded convolution, activation, and optional dropout block. -/
def convBlock {d : Nat} {inputChannels : Nat}
    (spatial : Tensor Nat [d]) (config : ConvBlock.Config d)
    (batchShape : Shape := []) :
    Builder (Sequential
      (batchShape.concat ((spatial.to Shape).prependDim inputChannels))
      (batchShape.concat
        (((config.convolution.outputSpatial spatial).to Shape)
          |>.prependDim config.convolution.outChannels))) :=
  match config.validate inputChannels spatial (kind := "ConvBlock") with
  | .error message =>
      pure <| nn.Internal.invalidConfiguration
        (batchShape.concat ((spatial.to Shape).prependDim inputChannels))
        (batchShape.concat
          (((config.convolution.outputSpatial spatial).to Shape)
            |>.prependDim config.convolution.outChannels))
        "ConvBlock" message
  | .ok () =>
      withInitializationSeed config.convolution.weightInitialization fun kernelSeed =>
        withOptionalDropoutSeed config.dropout? fun dropoutSeed =>
          pure <| Impl.convBlock batchShape spatial config kernelSeed dropoutSeed

/-- Build a seeded convolution/activation block followed by max pooling. -/
def convPoolBlock {d : Nat} {inputChannels : Nat}
    (spatial : Tensor Nat [d]) (config : ConvPoolBlock.Config d)
    (batchShape : Shape := []) :
    Builder (Sequential
      (batchShape.concat ((spatial.to Shape).prependDim inputChannels))
      (batchShape.concat
        (((config.pooling.outputSpatial
          (config.block.convolution.outputSpatial spatial)).to Shape)
            |>.prependDim config.block.convolution.outChannels))) :=
  match config.validate inputChannels spatial (kind := "ConvPoolBlock") with
  | .error message =>
      pure <| nn.Internal.invalidConfiguration
        (batchShape.concat ((spatial.to Shape).prependDim inputChannels))
        (batchShape.concat
          (((config.pooling.outputSpatial
            (config.block.convolution.outputSpatial spatial)).to Shape)
              |>.prependDim config.block.convolution.outChannels))
        "ConvPoolBlock" message
  | .ok () => do
      let block ← convBlock spatial config.block batchShape
      let pool ← maxPool (config.block.convolution.outputSpatial spatial) config.pooling batchShape
      pure <| compose block pool

/--
Build a multilayer perceptron over any `batchShape`.

Each hidden width contributes a linear layer followed by the configured activation and optional
dropout. Initialization seeds come from the surrounding `Builder` seed stream.

Example:
```lean
-- `16 -> 32 -> 32 -> 1`, ReLU between hidden layers, dropout after each one.
def model : nn.Builder (nn.Sequential [16] [1]) :=
  nn.mlp 16 1 { hiddenWidths := [32, 32], activation := .relu, dropout? := some 0.1 }
```
-/
def mlp (inputWidth outputWidth : Nat) (config : MLP.Config := {})
    (batchShape : Shape := []) :
    Builder
      (Sequential
        (batchShape.appendDim inputWidth)
        (batchShape.appendDim outputWidth)) :=
  let rec buildStages (currentWidth : Nat) (hiddenWidths : List Nat) :
      Builder
        (Sequential
          (batchShape.appendDim currentWidth)
          (batchShape.appendDim outputWidth)) :=
    match hiddenWidths with
    | .nil => linear currentWidth outputWidth (batchShape := batchShape)
    | .cons hiddenWidth remainingWidths => do
        let affine ← linear currentWidth hiddenWidth (batchShape := batchShape)
        let hiddenShape := batchShape.appendDim hiddenWidth
        let activated : Sequential hiddenShape hiddenShape :=
          activation config.activation
        let stage : Sequential
            (batchShape.appendDim currentWidth) hiddenShape :=
          compose affine activated
        let stage ←
          match config.dropout? with
          | none => pure stage
          | some probability => do
              let dropped ← dropout (shape := hiddenShape) probability
              pure <| compose stage dropped
        let rest ← buildStages hiddenWidth remainingWidths
        pure <| compose stage rest
  match config.validate inputWidth outputWidth with
  | .error message =>
      pure <| nn.Internal.invalidConfiguration
        (batchShape.appendDim inputWidth)
        (batchShape.appendDim outputWidth)
        "MLP"
        message
  | .ok () => buildStages inputWidth config.hiddenWidths

/--
A model built from `builder` with a seed drawn at run time.

`nn.Sequential` lives in `Type 1`, so `IO` cannot return it directly. This record stores only the
drawn seed; because `builder` is deterministic, `Built.model` recovers the same model every time.
`Built builder` lives in `Type`, so it can be returned from `IO` and stored in ordinary records.
-/
structure Built {σ τ : Shape} (builder : Builder (Sequential σ τ)) : Type where
  /-- The initialization seed drawn for this model. -/
  seed : Nat

/-- The model determined by a `Built` value. -/
def Built.model {σ τ : Shape} {builder : Builder (Sequential σ τ)} (built : Built builder) :
    Sequential σ τ :=
  build built.seed builder

/--
Draw the next global seed for `builder` and return it as an `IO` value.

This is the direct-construction counterpart of `nn.withModel`: `(← nn.buildIO builder).model` is
the model that `nn.withModel builder` would pass to its continuation.
-/
def buildIO {σ τ : Shape} (builder : Builder (Sequential σ τ)) : IO (Built builder) := do
  let seed ← rand.nextSeedGlobal
  pure ⟨seed⟩

/--
Build a model using the next global seed, then run a continuation.

`nn.Sequential` lives in `Type 1`, so executable code cannot receive the model directly from `IO`.
`nn.buildIO` covers most uses without a continuation; this form remains for code that already
uses continuation-passing style.

Example:
```lean
def builder : nn.Builder (nn.Sequential [2] [1]) :=
  nn.Sequential![nn.linear 2 8, nn.relu, nn.linear 8 1]

-- `nn.Sequential` lives in `Type 1`, so it cannot be returned from `IO`. Drawing the seed in `IO`
-- and handing the model to a continuation keeps model building pure.
def main : IO Unit :=
  nn.withModel builder fun model => nn.printSummary model
```
-/
def withModel {σ τ : Shape} {β : Type}
    (builder : Builder (Sequential σ τ))
    (continuation : Sequential σ τ → IO β) : IO β := do
  let built ← buildIO builder
  continuation built.model

end nn

end TorchLean
