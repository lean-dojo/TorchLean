/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

-- Downstream modules reach the marked imports through this module. Import shaking cannot see
-- those downstream uses, so keep the marked imports.
module -- shake: keep-downstream

public import TorchLean.NN.Random -- shake: keep
public import TorchLean.NN.Builder.Sequential -- shake: keep
public import TorchLean.NN.Functional.SelectiveScan
public import TorchLean.NN.Functional.Spectral
public import TorchLean.NN.Functional.Fourier.Transform

@[expose] public section

namespace TorchLean

/-!
# Seeded model builders

The standard layer constructors live here. Each returns an `nn.Builder`, which draws deterministic
initialization seeds from a seed stream. Most correspond to `torch.nn` modules: `nn.linear` to
`torch.nn.Linear`, `nn.conv` to `torch.nn.Conv1d`, `Conv2d` or `Conv3d` by spatial rank,
`nn.lstm` to `torch.nn.LSTM`, and so on. Their implementations, which take seeds as arguments,
mostly live in `nn.Sequential` (`TorchLean.NN.Builder.Sequential`). `nn.linear` and `nn.op` build
their runtime layers here, and `nn.classifierHead` and `nn.mlp` compose other builders.
`nn.withSeed`, `nn.withInitializationSeed` and `nn.withDropoutSeed` draw seeds inside a builder.

`nn.build seed` turns a builder into a model with fixed seeds. `nn.Sequential` lives in `Type 1`
because every layer stores an execution-polymorphic forward program, so it cannot be returned
directly from `IO`. A program draws a base seed in `IO` and then calls `nn.build` to construct the
model purely. `nn.buildIO` returns the drawn seed in a `Built` value that `IO` can carry, and
`(← nn.buildIO builder).model` is the model.
-/

namespace nn

universe u


-- Recurrent builders and runtime layer constructors use the same dimension checks. Calling the
-- runtime helper here keeps validation consistent before either path allocates model state.
open Layer (validateRecurrent)

/--
Build a value from a deterministic initialization seed.

Example:
```lean
def model : nn.Builder (nn.Sequential [2] [1]) :=
  nn.Sequential![nn.linear 2 8, nn.op .relu, nn.linear 8 1]

-- Same seed, same weights, on every machine and every run.
def built : nn.Sequential [2] [1] := nn.build 7 model
```
-/
def build {α : Type u} (seed : Nat) (builder : Builder α) : α :=
  let (value, _) := builder { seed := seed }
  value

/-- Consume one initialization seed and continue building in the same result universe. -/
def withSeed {α : Type u} (continuation : Nat → Builder α) : Builder α :=
  fun state =>
    let (seed, nextState) := Random.SeedStream.next state
    continuation seed nextState

/--
Draw a seed for `continuation` unless `initialization` is `.zeros` or `.ones`, which need none.
In those two cases `continuation` receives 0 and the stream does not advance.
-/
def withInitializationSeed {α : Type u}
    (initialization : Init.Scheme) (continuation : Nat → Builder α) : Builder α :=
  match initialization with
  | .zeros | .ones => continuation 0
  | _ => withSeed continuation

/--
Draw a seed for `continuation` unless `probability` is 0 or 1, the two values whose training mask
is not random. In those two cases `continuation` receives 0 and the stream does not advance.
-/
def withDropoutSeed {α : Type u}
    (probability : Float) (continuation : Nat → Builder α) : Builder α :=
  if probability == 0.0 || probability == 1.0 then
    continuation 0
  else
    withSeed continuation

/--
Draw a dropout seed for an optional site, as `withDropoutSeed` does. An absent site (`none`) draws
no seed and passes 0, as a probability of 0 or 1 does.
-/
def withOptionalDropoutSeed {α : Type u}
    (probability? : Option Float) (continuation : Nat → Builder α) : Builder α :=
  match probability? with
  | none => continuation 0
  | some probability => withDropoutSeed probability continuation

/-!
## Layer constructors

Each builder validates its configuration and draws only the seeds that its random initializers and
dropout masks need. Most delegate to an implementation in `nn.Sequential`. `nn.linear` and `nn.op`
build their runtime layers here, and `nn.classifierHead` and blocks such as `nn.mlp` compose other
builders. A configuration that fails validation yields a placeholder model, which model validation
rejects before any state is allocated.
-/

/--
Build global average pooling, which averages each channel over all spatial positions and draws no
seed. PyTorch analogue, for two spatial axes: `torch.nn.AdaptiveAvgPool2d(1)` followed by
`torch.nn.Flatten()`.

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
      (batchShape ++ (Shape.dim channels (spatial.to Shape)))
      (batchShape ++ [channels])) :=
  pure <| nn.Sequential.globalAvgPool batchShape (channels := channels) spatial

/--
Build the parameter-free layer `Layer.op o` of one registered operator without consuming a seed.

Example:
```lean
-- ReLU between two linear layers, then a softmax over the one axis of the output.
def model : nn.Builder (nn.Sequential [2] [3]) :=
  nn.Sequential![nn.linear 2 8, nn.op .relu, nn.linear 8 3, nn.op (.softmax ⟨0, by decide⟩)]
```
-/
def op {σ τ : Shape} {rest : List Port} (o : Op [.value σ] (.value τ :: rest))
    (h : o.need = .none := by rfl) : Builder (Sequential σ τ) :=
  pure (Sequential.fromLayer (Layer.op o h))

/-- Build a reduction that sums every tensor entry to a scalar (PyTorch: `torch.sum`). -/
def sum {shape : Shape} : Builder (Sequential shape []) :=
  pure (nn.Sequential.sum (s := shape))

/--
Build max pooling over arbitrary spatial rank with the pooling window `config`. PyTorch analogue:
`torch.nn.MaxPool2d` for two spatial axes, and `MaxPool1d` or `MaxPool3d` for one or three.

Example:
```lean
def spatial : Tensor Nat [2] := [8, 8]

def pooling : TorchLean.Window 2 :=
  { kernelSize := [2, 2], stride := [2, 2] }

-- Non-overlapping `2 x 2` windows halve both spatial axes and leave the channel count alone.
def model : nn.Builder (nn.Sequential [3, 8, 8] [3, 4, 4]) :=
  nn.maxPool spatial pooling (channels := 3)
```
-/
def maxPool {d channels : Nat} (spatial : Tensor Nat [d])
    (config : Window d) (batchShape : Shape := []) :
    Builder (Sequential
      (batchShape ++ (Shape.dim channels (spatial.to Shape)))
      (batchShape ++
        (Shape.dim channels ((config.poolOutputSpatial spatial).to Shape)))) :=
  match config.validatePool channels spatial (kind := "MaxPool") with
  | .error message =>
      pure <| nn.Sequential.invalid
        (batchShape ++ (Shape.dim channels (spatial.to Shape)))
        (batchShape ++
          (Shape.dim channels ((config.poolOutputSpatial spatial).to Shape)))
        "MaxPool" message
  | .ok () =>
      pure (nn.Sequential.maxPool batchShape (channels := channels) spatial config)

/--
Build average pooling over arbitrary spatial rank with the pooling window `config`. PyTorch
analogue: `torch.nn.AvgPool2d` for two spatial axes, and `AvgPool1d` or `AvgPool3d` for one or
three.
-/
def avgPool {d channels : Nat} (spatial : Tensor Nat [d])
    (config : Window d) (batchShape : Shape := []) :
    Builder (Sequential
      (batchShape ++ (Shape.dim channels (spatial.to Shape)))
      (batchShape ++
        (Shape.dim channels ((config.poolOutputSpatial spatial).to Shape)))) :=
  match config.validatePool channels spatial (kind := "AvgPool") with
  | .error message =>
      pure <| nn.Sequential.invalid
        (batchShape ++ (Shape.dim channels (spatial.to Shape)))
        (batchShape ++
          (Shape.dim channels ((config.poolOutputSpatial spatial).to Shape)))
        "AvgPool" message
  | .ok () =>
      pure (nn.Sequential.avgPool batchShape (channels := channels) spatial config)

/--
Build a transpose convolution over arbitrary spatial rank, seeding its kernel when initialization
is stochastic. `validateConvolution` checks the configuration on the transposed output grid
`config.transposedOutputSpatial spatial`. PyTorch analogue: `torch.nn.ConvTranspose2d` for two
spatial axes, and `ConvTranspose1d` or `ConvTranspose3d` for one or three.
-/
def convTranspose {d : Nat} {inputChannels : Nat}
    (spatial : Tensor Nat [d]) (config : Conv.Config d)
    (batchShape : Shape := []) :
    Builder (Sequential
      (batchShape ++ (Shape.dim inputChannels (spatial.to Shape)))
      (batchShape ++
        (Shape.dim config.outChannels ((config.transposedOutputSpatial spatial).to Shape)))) :=
  match nn.validateConvolution inputChannels config.outChannels spatial
      config.kernelSize config.stride (config.transposedOutputSpatial spatial)
      config.weightInit "ConvTranspose" with
  | .error message =>
      pure <| nn.Sequential.invalid
        (batchShape ++ (Shape.dim inputChannels (spatial.to Shape)))
        (batchShape ++
          (Shape.dim config.outChannels ((config.transposedOutputSpatial spatial).to Shape)))
        "ConvTranspose" message
  | .ok () =>
      withInitializationSeed config.weightInit fun kernelSeed =>
        pure <| nn.Sequential.convTranspose batchShape spatial config kernelSeed

/--
Build an affine layer, consuming seeds only for stochastic initializers.

PyTorch analogue: `torch.nn.Linear`. On an input of shape `[..., inputWidth]` the leading
dimensions are flattened to `(numel(prefix), inputWidth)`, multiplied by the transposed weight,
the bias is broadcast and added, and the result is reshaped back to `[..., outputWidth]`.

Example:
```lean
-- Two affine layers around a ReLU: `2 -> 8 -> 1`.
def model : nn.Builder (nn.Sequential [2] [1]) :=
  nn.Sequential![
    nn.linear 2 8,
    nn.op .relu,
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
      (batchShape ++ [inputWidth])
      (batchShape ++ [outputWidth])) :=
  match config.validate inputWidth outputWidth with
  | .error message =>
      pure <| nn.Sequential.invalid
        (batchShape ++ [inputWidth])
        (batchShape ++ [outputWidth])
        "Linear" message
  | .ok () =>
      let weightInitialization :=
        config.weightInit?.getD (.xavierUniform inputWidth outputWidth)
      let weightShape : Shape := [outputWidth, inputWidth]
      let biasShape : Shape := [outputWidth]
      let batch : Nat := batchShape.size
      withInitializationSeed weightInitialization fun weightSeed =>
        withInitializationSeed config.biasInit fun biasSeed =>
          let initialWeight : Tensor Float weightShape := Init.tensor
            (shape := weightShape) (scheme := weightInitialization) (seed := weightSeed)
          let initialBias : Tensor Float biasShape := Init.tensor
            (shape := biasShape) (scheme := config.biasInit) (seed := biasSeed)
          pure <| Sequential.fromLayer
            { kind := s!"Linear({inputWidth}, {outputWidth})"
              stateShapes := [weightShape, biasShape]
              initState := TorchLean.Pack.pair initialWeight initialBias
              runtimeInit := some (.cons
                (Init.Source.scheme
                  weightInitialization weightSeed)
                (.cons
                  (Init.Source.scheme
                    config.biasInit biasSeed)
                  .nil))
              requiresGrad := #[true, true]
              validateConfig := config.validate inputWidth outputWidth
              forward := fun _ {α} _ _ =>
                fun {m} _ _ =>
                  fun w b x =>
                    let sIn := batchShape ++ [inputWidth]
                    let sOut := (batchShape ++ [outputWidth])
                    ((do
                      let flatInput ←
                        Ops.apply (m := m) (α := α)
                          (.reshape (s := sIn) (t := [batch, inputWidth]) (by
                            rw [Shape.size_append_singleton]
                            simp [batch])) (x, ())
                      let wT ←
                        Ops.apply (m := m) (α := α)
                          (.permute (s := [outputWidth, inputWidth])
                            (Shape.adjacentSwap [outputWidth, inputWidth] 0)
                            (Shape.swapAdjacentAtDepth [outputWidth, inputWidth] 0)
                            (by tensor_shape))
                          (w, ())
                      let y : Ops.Ref (m := m) (α := α) [batch, outputWidth] ←
                        Ops.apply (m := m) (α := α)
                          (.matmul (batchA := []) (batchB := []) (batch := []) (m := batch)
                            (n := inputWidth) (p := outputWidth) false false
                            (fun h => nomatch h) (fun h => nomatch h)
                            Shape.BroadcastTo.proof Shape.BroadcastTo.proof)
                          (flatInput, wT, ())
                      let flatOutput ←
                        functional.addB (m := m) (α := α)
                          (t := [batch, outputWidth]) y b
                      Ops.apply (m := m) (α := α)
                        (.reshape (s := [batch, outputWidth]) (t := sOut) (by
                          rw [Shape.size_append_singleton]
                          simp [batch])) (flatOutput, ())
                    ) : m (Ops.Ref (m := m) (α := α) sOut))
            }

/--
Build a classification head: flatten the feature suffix and apply a seeded `nn.linear` with
`classCount` outputs. The feature size and `classCount` must be positive. PyTorch analogue:
`torch.nn.Flatten()` followed by `torch.nn.Linear(featureShape.size, classCount)`.
-/
def classifierHead {featureShape : Shape} (classCount : Nat)
    (batchShape : Shape := []) :
    Builder (Sequential
      (batchShape ++ featureShape)
      (batchShape ++ [classCount])) :=
  let input := batchShape ++ featureShape
  let output := batchShape ++ [classCount]
  if featureShape.size = 0 then
    pure <| nn.Sequential.invalid input output "Classifier"
      "Classifier: feature size must be positive"
  else if classCount = 0 then
    pure <| nn.Sequential.invalid input output "Classifier"
      "Classifier: class count must be positive"
  else
    (fun head => nn.compose![
        Sequential.fromLayer (Layer.op
          (.reshape (Shape.size_append_eq_size_append_singleton batchShape featureShape))),
        head]) <$> linear featureShape.size classCount (batchShape := batchShape)


/--
Build a seeded recurrent neural network over a fixed sequence length.

The recurrent computation acts independently over every index in `batchShape`, and its parameters
are shared across those indices. The default `batchShape := []` is a single sequence, and `[batch]`
gives the usual batched model. PyTorch analogue: `torch.nn.RNN(inputWidth, hiddenWidth)` with its
default tanh, and with `batch_first=True` when `batchShape` is `[batch]`.
-/
def rnn (sequenceLength inputWidth hiddenWidth : Nat) (batchShape : Shape := []) :
    Builder (Sequential
      (batchShape ++ [sequenceLength] ++ [inputWidth])
      (batchShape ++ [sequenceLength] ++ [hiddenWidth])) :=
  let input := batchShape ++ [sequenceLength] ++ [inputWidth]
  let output := batchShape ++ [sequenceLength] ++ [hiddenWidth]
  match validateRecurrent
      "RNN" sequenceLength inputWidth hiddenWidth with
  | .error message =>
      pure <| nn.Sequential.invalid input output "RNN" message
  | .ok () =>
      withSeed fun weightSeed =>
        pure <| nn.Sequential.batchedRecurrent batchShape
          (nn.Sequential.rnn sequenceLength inputWidth hiddenWidth weightSeed)

/--
Build a seeded GRU, shared over every index in `batchShape`.

The default convention `resetBefore` resets the previous hidden state before multiplying it by the
recurrent matrix, and draws one seed for each of its reset, update and candidate weight matrices.
Choose `resetAfter` for the recurrence of `torch.nn.GRU` and its four packed parameter tensors.
That version draws one seed for each of its input and recurrent weight matrices and keeps both
biases independent.
-/
def gru (sequenceLength inputWidth hiddenWidth : Nat) (batchShape : Shape := [])
    (convention : GRU.Convention := .resetBefore) :
    Builder (Sequential
      (batchShape ++ [sequenceLength] ++ [inputWidth])
      (batchShape ++ [sequenceLength] ++ [hiddenWidth])) :=
  let input := batchShape ++ [sequenceLength] ++ [inputWidth]
  let output := batchShape ++ [sequenceLength] ++ [hiddenWidth]
  match validateRecurrent
      "GRU" sequenceLength inputWidth hiddenWidth with
  | .error message =>
      pure <| nn.Sequential.invalid input output "GRU" message
  | .ok () =>
      match convention with
      | .resetBefore =>
          withSeed fun resetWeightSeed =>
            withSeed fun updateWeightSeed =>
              withSeed fun candidateWeightSeed =>
                pure <| nn.Sequential.batchedRecurrent batchShape
                  (nn.Sequential.gru sequenceLength inputWidth hiddenWidth
                    resetWeightSeed updateWeightSeed candidateWeightSeed)
      | .resetAfter =>
          withSeed fun inputWeightSeed =>
            withSeed fun hiddenWeightSeed =>
              pure <| nn.Sequential.batchedRecurrent batchShape
                (Sequential.fromLayer <| Layer.gruResetAfter
                  sequenceLength inputWidth hiddenWidth inputWeightSeed hiddenWeightSeed)

/-- Build a seeded selective Mamba layer, shared over every index in `batchShape`. -/
def mamba (sequenceLength inputWidth hiddenWidth : Nat) (batchShape : Shape := [])
    (options : Mamba.Options := {}) :
    Builder (Sequential
      (batchShape ++ [sequenceLength] ++ [inputWidth])
      (batchShape ++ [sequenceLength] ++ [hiddenWidth])) :=
  let input := batchShape ++ [sequenceLength] ++ [inputWidth]
  let output := batchShape ++ [sequenceLength] ++ [hiddenWidth]
  match (do
      validateRecurrent "Mamba" sequenceLength inputWidth hiddenWidth
      options.validate) with
  | .error message =>
      pure <| nn.Sequential.invalid input output "Mamba" message
  | .ok () =>
      withSeed fun inputWeightSeed =>
        withSeed fun stateWeightSeed =>
          withSeed fun gateWeightSeed =>
            pure <| nn.Sequential.batchedRecurrent batchShape
              (nn.Sequential.mamba sequenceLength inputWidth hiddenWidth
                 inputWeightSeed stateWeightSeed gateWeightSeed (options := options))

/--
Build a seeded long short-term memory layer, shared over every index in `batchShape`. PyTorch
analogue: `torch.nn.LSTM(inputWidth, hiddenWidth)`, with `batch_first=True` when `batchShape` is
`[batch]`.

Example:
```lean
-- Sixteen timesteps of width 8 in, sixteen hidden states of width 32 out.
def model : nn.Builder (nn.Sequential [16, 8] [16, 32]) :=
  nn.lstm 16 8 32
```
-/
def lstm (sequenceLength inputWidth hiddenWidth : Nat) (batchShape : Shape := []) :
    Builder (Sequential
      (batchShape ++ [sequenceLength] ++ [inputWidth])
      (batchShape ++ [sequenceLength] ++ [hiddenWidth])) :=
  let input := batchShape ++ [sequenceLength] ++ [inputWidth]
  let output := batchShape ++ [sequenceLength] ++ [hiddenWidth]
  match validateRecurrent
      "LSTM" sequenceLength inputWidth hiddenWidth with
  | .error message =>
      pure <| nn.Sequential.invalid input output "LSTM" message
  | .ok () =>
      withSeed fun forgetWeightSeed =>
        withSeed fun inputWeightSeed =>
          withSeed fun candidateWeightSeed =>
            withSeed fun outputWeightSeed =>
              pure <| nn.Sequential.batchedRecurrent batchShape
                (nn.Sequential.lstm sequenceLength inputWidth hiddenWidth
                  forgetWeightSeed inputWeightSeed candidateWeightSeed outputWeightSeed)

/--
Build an arbitrary-rank convolution, seeding its kernel when initialization is stochastic.
PyTorch analogue: `torch.nn.Conv2d` for two spatial axes, and `Conv1d` or `Conv3d` for one or
three.

Example:
```lean
def spatial : Tensor Nat [2] := [8, 8]

def convolution : nn.Conv.Config 2 :=
  { outChannels := 4, kernelSize := [3, 3] }

-- One `8 x 8` channel in, four `6 x 6` feature maps out: no padding, so the kernel eats a
-- one-pixel border on each side.
def model : nn.Builder (nn.Sequential [1, 8, 8] [4, 6, 6]) :=
  nn.conv spatial convolution (inputChannels := 1)
```
-/
def conv {d : Nat} {inputChannels : Nat} (spatial : Tensor Nat [d])
    (config : Conv.Config d) (batchShape : Shape := []) :
    Builder (Sequential
      (batchShape ++ (Shape.dim inputChannels (spatial.to Shape)))
      (batchShape ++
        (Shape.dim config.outChannels ((config.convOutputSpatial spatial).to Shape)))) :=
  match config.validate inputChannels spatial (kind := "Conv") with
  | .error message =>
      pure <| nn.Sequential.invalid
        (batchShape ++ (Shape.dim inputChannels (spatial.to Shape)))
        (batchShape ++
          (Shape.dim config.outChannels ((config.convOutputSpatial spatial).to Shape)))
        "Conv" message
  | .ok () =>
      withInitializationSeed config.weightInit fun kernelSeed =>
        pure <| nn.Sequential.conv batchShape (inputChannels := inputChannels) spatial config
          kernelSeed

/--
Build a pointwise convolution over any spatial rank.

The unit kernel, unit stride and zero padding preserve every spatial axis, and residual, diffusion
and encoder-decoder models use it as their channel projection. PyTorch analogue:
`torch.nn.Conv2d(inputChannels, outputChannels, kernel_size=1)` for two spatial axes.
-/
def pointwiseConv {d : Nat} {inputChannels : Nat}
    (spatial : Tensor Nat [d]) (outputChannels : Nat) (batchShape : Shape := []) :
    Builder (Sequential
      (batchShape ++ (Shape.dim inputChannels (spatial.to Shape)))
      (batchShape ++ (Shape.dim outputChannels (spatial.to Shape)))) := by
  let config : Conv.Config d :=
    { outChannels := outputChannels
      kernelSize := Tensor.ones [d]
      stride := Tensor.ones [d]
      padding := Tensor.zeros [d] }
  have hOut : config.convOutputSpatial spatial = spatial :=
    Window.convOutputSpatial_unit spatial
  let result :=
    conv spatial config (batchShape := batchShape) (inputChannels := inputChannels)
  rw [hOut] at result
  exact result

/--
Build batch normalization with learned scale and bias and stored running statistics.

Training computes one mean and variance per channel across the batch and spatial axes, and updates
each stored statistic to `(1 - momentum) * running + momentum * batch`. Evaluation uses the stored
statistics. `eps` is added to the variance before taking its square root. Scale and running
variance start at one, while bias and running mean start at zero. PyTorch analogue:
`torch.nn.BatchNorm2d(channels, momentum=0.1)` for two spatial axes, and `BatchNorm1d` or
`BatchNorm3d` for one or three.

Example:
```lean
-- Shape in equals shape out. What changes is the running mean and variance this layer keeps as
-- persistent buffers, updated in training mode and only read in eval mode.
def model : nn.Builder (nn.Sequential [3, 8, 8] [3, 8, 8]) :=
  nn.batchNorm [8, 8] (momentum := 0.1) (channels := 3)
```
-/
def batchNorm {d channels : Nat}
    (spatial : Tensor Nat [d]) (momentum : Float := 0.1)
    (batchShape : Shape := []) (eps : Rat := 1e-5) :
    Builder (Sequential
      (batchShape ++ (Shape.dim channels (spatial.to Shape)))
      (batchShape ++ (Shape.dim channels (spatial.to Shape)))) :=
  pure <| nn.Sequential.batchNorm batchShape spatial (momentum := momentum) (eps := eps)

/--
Build instance normalization using each sample and channel's spatial mean and variance.

The layer uses current input statistics in both training and evaluation. Scale and bias each
have one entry per channel and start at one and zero. Both are enabled by default. `bias := false`
keeps only the scale, and `affine := false` removes both. `eps` is added to the variance before
taking its square root. PyTorch analogue: `torch.nn.InstanceNorm2d(channels, affine=True)` for
two spatial axes.
-/
def instanceNorm {d channels : Nat}
    (spatial : Tensor Nat [d]) (batchShape : Shape := [])
    (eps : Rat := 1e-5) (affine bias : Bool := true) :
    Builder (Sequential
      (batchShape ++ (Shape.dim channels (spatial.to Shape)))
      (batchShape ++ (Shape.dim channels (spatial.to Shape)))) :=
  pure <| nn.Sequential.instanceNorm batchShape spatial (eps := eps) (affine := affine)
    (bias := bias)

/--
Build group normalization with equal, contiguous groups of channels.

Each sample's groups have separate means and variances, computed over their channels and spatial
positions. The channel count must be divisible by the positive group count. Scale and bias have
one entry per channel. `bias := false` keeps only the scale, and `affine := false` removes both.
`eps` is added to each group's variance before taking its square root. PyTorch analogue:
`torch.nn.GroupNorm(groups, channels)`.
-/
def groupNorm {d channels : Nat}
    (spatial : Tensor Nat [d]) (groups : Nat) (batchShape : Shape := [])
    (eps : Rat := 1e-5) (affine bias : Bool := true) :
    Builder (Sequential
      (batchShape ++ (Shape.dim channels (spatial.to Shape)))
      (batchShape ++ (Shape.dim channels (spatial.to Shape)))) :=
  pure <| nn.Sequential.groupNorm batchShape spatial groups (eps := eps) (affine := affine)
    (bias := bias)

/--
Build an embedding layer whose input is a dense vector of vocabulary weights.

A one-hot input selects a table row, and any other input forms a weighted combination of rows.
For token indices, `embedding` does the lookup without one-hot tensors.
-/
def oneHotEmbedding (vocabularySize embeddingWidth : Nat)
    (config : Embedding.Config := {})
    (batchShape : Shape := []) :
    Builder (Sequential
      (batchShape ++ [vocabularySize])
      (batchShape ++ [embeddingWidth])) :=
  match config.validate vocabularySize embeddingWidth with
  | .error message =>
      pure <| nn.Sequential.invalid
        (batchShape ++ [vocabularySize])
        (batchShape ++ [embeddingWidth])
        "OneHotEmbedding" message
  | .ok () =>
      withInitializationSeed config.weightInit fun seed =>
        pure <| nn.Sequential.oneHotEmbedding vocabularySize embeddingWidth config seed
          (batchShape := batchShape)

/--
Build a lookup table for bounded token indices of type `Fin vocabularySize`. PyTorch analogue:
`torch.nn.Embedding(vocabularySize, embeddingWidth)`.

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
      pure <| Embedding.invalid vocabularySize embeddingWidth message
  | .ok () =>
      withInitializationSeed config.weightInit fun seed =>
        pure <| nn.Embedding.fromSeed vocabularySize embeddingWidth config seed

/-- Build deterministic sinusoidal positional encoding over a sequence suffix. -/
def sinusoidalPositionalEncoding (batchShape : Shape := [])
    {sequenceLength embeddingWidth : Nat}
    (config : SinusoidalPositionalEncoding.Config := {}) :
    Builder (Sequential
      (batchShape ++ [sequenceLength, embeddingWidth])
      (batchShape ++ [sequenceLength, embeddingWidth])) :=
  match config.validate sequenceLength embeddingWidth with
  | .error message =>
      let shape := batchShape ++ [sequenceLength, embeddingWidth]
      pure <| nn.Sequential.invalid
        shape shape "SinusoidalPositionalEncoding" message
  | .ok () =>
      pure <| nn.Sequential.sinusoidalPositionalEncoding batchShape
        (sequenceLength := sequenceLength) (embeddingWidth := embeddingWidth) config

/-- Build deterministic rotary positional encoding for multi-head sequence features. -/
def rope (batchShape : Shape := []) {sequenceLength headWidth : Nat}
    (config : RotaryEmbedding.Config := {}) :
    Builder (Sequential
      (batchShape ++ [sequenceLength, headWidth])
      (batchShape ++ [sequenceLength, headWidth])) :=
  match config.validate sequenceLength headWidth with
  | .error message =>
      let shape := batchShape ++ [sequenceLength, headWidth]
      pure <| nn.Sequential.invalid shape shape "RoPE" message
  | .ok () =>
      pure <| nn.Sequential.rope batchShape
        (sequenceLength := sequenceLength) (headWidth := headWidth) config

/-- Build learned positional embeddings, consuming a seed only for stochastic initialization. -/
def learnedPositionalEmbedding (batchShape : Shape := [])
    {sequenceLength embeddingWidth : Nat}
    (config : LearnedPositionalEmbedding.Config := {}) :
    Builder (Sequential
      (batchShape ++ [sequenceLength, embeddingWidth])
      (batchShape ++ [sequenceLength, embeddingWidth])) :=
  match config.validate sequenceLength embeddingWidth with
  | .error message =>
      let shape := batchShape ++ [sequenceLength, embeddingWidth]
      pure <| nn.Sequential.invalid
        shape shape "LearnedPositionalEmbedding" message
  | .ok () =>
      withInitializationSeed config.init fun positionSeed =>
        pure <| nn.Sequential.learnedPositionalEmbedding batchShape
          (sequenceLength := sequenceLength) (embeddingWidth := embeddingWidth)
          config (positionSeed := positionSeed)

/--
Build layer normalization over the final axis.

Each row uses its own mean and variance. Scale and bias have shape `[width]` and are shared across
the leading axes, starting at one and zero. `eps` is added to the variance before taking its square
root. Setting `bias := false` keeps only the scale, and setting `affine := false` removes both
parameters. PyTorch analogue: `torch.nn.LayerNorm(width)`, whose `elementwise_affine` is `affine`.

The rational `eps` must remain positive and finite after conversion to the execution scalar.
The default can round to zero in tiny formats, producing NaNs on constant rows in a typed graph.
Validation checks rational positivity only. Choose a representable positive value with `eps`.
For three exponent bits and two fraction bits, `(eps := (1 / 16 : Rat))` is such a value.

Example:
```lean
-- Normalizes across the final axis of each `[16, 64]` row, the Transformer convention.
def model : nn.Builder (nn.Sequential [16, 64] [16, 64]) :=
  nn.layerNorm [16] (width := 64)
```
-/
def layerNorm (batchShape : Shape := []) {width : Nat}
    (eps : Rat := 1e-5) (affine bias : Bool := true) :
    Builder (Sequential (batchShape ++ [width]) (batchShape ++ [width])) :=
  pure <| nn.Sequential.layerNorm batchShape (width := width) (eps := eps) (affine := affine)
    (bias := bias)

/--
Build RMS normalization over the final axis.

Each row is divided by `sqrt(mean(x * x) + eps)`, then multiplied by a scale of shape `[width]`.
There is no mean subtraction or bias. The scale starts at one, and `affine := false` removes it
from model state. The default `eps` is `1e-5` for every scalar type. PyTorch analogue:
`torch.nn.RMSNorm(width, eps=1e-5)`.

The converted epsilon must remain positive and finite. If it rounds to zero in a tiny format,
zero input can produce NaNs, so pass a representable positive `eps` before lowering the model.
-/
def rmsNorm (batchShape : Shape := []) {width : Nat}
    (eps : Rat := 1e-5) (affine : Bool := true) :
    Builder (Sequential (batchShape ++ [width]) (batchShape ++ [width])) :=
  pure <| nn.Sequential.rmsNorm batchShape (width := width) (eps := eps) (affine := affine)

/--
Build seeded multi-head self-attention with an optional fixed attention mask.

PyTorch analogue when `headWidth = modelWidth / headCount`:
`torch.nn.MultiheadAttention(modelWidth, headCount, bias=False, batch_first=True)` applied as
`mha(x, x, x)`. A `mask` entry `true` lets the query attend to the key, the opposite of a Boolean
`attn_mask`, so the analogue of `mask` is `attn_mask=~mask`.

Example:
```lean
-- Two heads of width 4 give an internal attention width of 8, which here matches the model
-- width. The two are independent, so `headCount * headWidth` may differ from it.
def model : nn.Builder (nn.Sequential [4, 8] [4, 8]) :=
  nn.attention { headCount := 2, headWidth := 4 }
    (sequenceLength := 4) (modelWidth := 8)

-- The causal mask is an argument of its own: its type carries the sequence length, which
-- `Attention.Options` does not.
def causal : nn.Builder (nn.Sequential [4, 8] [4, 8]) :=
  nn.attention { headCount := 2, headWidth := 4 }
    (mask := some (TorchLean.Attention.causalMask 4)) (sequenceLength := 4) (modelWidth := 8)
```
-/
def attention {sequenceLength modelWidth : Nat}
    (config : Attention.Options)
    (mask : Option (Tensor Bool [sequenceLength, sequenceLength]) := none)
    (batchShape : Shape := []) :
    Builder (Sequential
      (batchShape ++ [sequenceLength, modelWidth])
      (batchShape ++ [sequenceLength, modelWidth])) :=
  match config.validate sequenceLength modelWidth with
  | .error message =>
      let shape := batchShape ++ [sequenceLength, modelWidth]
      pure <| nn.Sequential.invalid shape shape "MultiHeadAttention" message
  | .ok () =>
      let projectionWidth := config.headCount * config.headWidth
      let (projectionInitialization, outputInitialization) :=
        Layer.Attention.initializations modelWidth projectionWidth config.weightInit?
          config.outputWeightInit?
      withInitializationSeed projectionInitialization fun queryWeightSeed =>
        withInitializationSeed projectionInitialization fun keyWeightSeed =>
          withInitializationSeed projectionInitialization fun valueWeightSeed =>
            withInitializationSeed outputInitialization fun outputWeightSeed =>
              withOptionalDropoutSeed config.dropout? fun dropoutSeed =>
                pure <| nn.Sequential.attention batchShape
                  (sequenceLength := sequenceLength) (modelWidth := modelWidth)
                  config
                  (queryWeightSeed := queryWeightSeed)
                  (keyWeightSeed := keyWeightSeed)
                  (valueWeightSeed := valueWeightSeed)
                  (outputWeightSeed := outputWeightSeed)
                  (mask := mask) (dropoutSeed := dropoutSeed)

/--
Build one seeded transformer encoder block, optionally applying a fixed attention mask.

PyTorch analogue: `torch.nn.TransformerEncoderLayer(modelWidth, headCount, feedForwardWidth)` with
`batch_first=True`, `norm_first=normalizeFirst`, `dropout=0.0` and
`activation=nn.GELU(approximate='tanh')`. The defaults here are post-normalization, GELU in its
tanh approximation, no dropout and no attention biases, where PyTorch's are ReLU, dropout `0.1`
and attention biases.

Example:
```lean
-- Pre-norm block, GELU feed-forward, no dropout: attention and feed-forward each sit inside their
-- own residual connection, so shapes in and out agree.
def model : nn.Builder (nn.Sequential [16, 64] [16, 64]) :=
  nn.transformerBlock
    { headCount := 4
      headWidth := 16
      feedForwardWidth := 256
      normalizeFirst := true }
    (sequenceLength := 16) (modelWidth := 64)
```
-/
def transformerBlock {sequenceLength modelWidth : Nat}
    (config : TransformerBlock.Config)
    (mask : Option (Tensor Bool [sequenceLength, sequenceLength]) := none)
    (batchShape : Shape := []) :
    Builder (Sequential
      (batchShape ++ [sequenceLength, modelWidth])
      (batchShape ++ [sequenceLength, modelWidth])) :=
  let shape := batchShape ++ [sequenceLength, modelWidth]
  let invalid (message : String) : Builder (Sequential shape shape) :=
    pure <| nn.Sequential.invalid shape shape "TransformerEncoder" message
  if sequenceLength = 0 then
    invalid "TransformerEncoder: sequence length must be positive"
  else if modelWidth = 0 then
    invalid "TransformerEncoder: model width must be positive"
  else match config.validate with
  | .error message => invalid message
  | .ok () =>
      let projectionWidth := config.headCount * config.headWidth
      let (projectionInitialization, outputInitialization) :=
        Layer.Attention.initializations modelWidth projectionWidth config.weightInit?
          config.residualInit?
      withInitializationSeed projectionInitialization fun queryWeightSeed =>
        withInitializationSeed projectionInitialization fun keyWeightSeed =>
          withInitializationSeed projectionInitialization fun valueWeightSeed =>
            withInitializationSeed outputInitialization fun outputWeightSeed => do
              -- The feed-forward layers draw their weight seeds right after the attention ones.
              let firstFeedForward ← linear modelWidth config.feedForwardWidth
                (batchShape := batchShape ++ [sequenceLength])
                (config := { weightInit? := config.weightInit? })
              let secondFeedForward ← linear config.feedForwardWidth modelWidth
                (batchShape := batchShape ++ [sequenceLength])
                (config := {
                  weightInit? :=
                    config.residualInit?.orElse
                      (fun _ => config.weightInit?) })
              -- Dropout keys: residual attention and residual feed-forward first, then attention
              -- probabilities and the hidden feed-forward layer.
              withOptionalDropoutSeed config.dropout? fun attentionDropoutSeed =>
                withOptionalDropoutSeed config.dropout? fun feedForwardDropoutSeed =>
                  withOptionalDropoutSeed config.attentionDropout?
                      fun attentionProbabilityDropoutSeed =>
                    withOptionalDropoutSeed config.feedForwardDropout?
                        fun feedForwardHiddenDropoutSeed =>
                      pure <| nn.Sequential.transformerBlock batchShape
                        (sequenceLength := sequenceLength) (modelWidth := modelWidth)
                        config
                        (queryWeightSeed := queryWeightSeed)
                        (keyWeightSeed := keyWeightSeed)
                        (valueWeightSeed := valueWeightSeed)
                        (outputWeightSeed := outputWeightSeed)
                        (firstFeedForward := firstFeedForward)
                        (secondFeedForward := secondFeedForward)
                        (attentionDropoutSeed := attentionDropoutSeed)
                        (feedForwardDropoutSeed := feedForwardDropoutSeed)
                        (mask := mask)
                        (attentionProbabilityDropoutSeed := attentionProbabilityDropoutSeed)
                        (feedForwardHiddenDropoutSeed := feedForwardHiddenDropoutSeed)

/--
Build a stack of `config.layerCount` transformer encoder blocks with an optional attention mask.
Each block draws its own seeds, and zero layers give the identity. PyTorch analogue:
`torch.nn.TransformerEncoder(layer, num_layers=config.layerCount)`.
-/
def transformerStack {sequenceLength modelWidth : Nat}
    (config : TransformerStack.Config)
    (mask : Option (Tensor Bool [sequenceLength, sequenceLength]) := none)
    (batchShape : Shape := []) :
    Builder (Sequential
      (batchShape ++ [sequenceLength, modelWidth])
      (batchShape ++ [sequenceLength, modelWidth])) :=
  let shape := batchShape ++ [sequenceLength, modelWidth]
  let invalid (message : String) : Builder (Sequential shape shape) :=
    pure <| nn.Sequential.invalid shape shape "TransformerEncoder" message
  let rec
    /-- A stack of the given number of transformer blocks, the identity for `0`. -/
    buildLayers : Nat → Builder (Sequential shape shape)
      | 0 => pure <| Sequential.id shape
      | remaining + 1 => do
          let current ← transformerBlock config.block (mask := mask)
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
Build dropout with probability `p`, which must be finite and in $[0, 1]$. A seed is drawn only
for $0 < p < 1$, the values whose training mask is random. PyTorch analogue: `torch.nn.Dropout(p)`.

Example:
```lean
-- Active in training mode and the identity in eval mode, which the trainer selects for you.
def model : nn.Builder (nn.Sequential [64] [64]) :=
  nn.dropout 0.1
```
-/
def dropout {shape : Shape} (p : Float) : Builder (Sequential shape shape) :=
  if functional.inUnitInterval p then
    withDropoutSeed p fun seed =>
      pure <| nn.Sequential.dropout (s := shape) p (seed := seed)
  else
    pure <| nn.Sequential.invalid shape shape "Dropout"
      s!"Dropout: probability must be finite and in [0, 1], got {p}"

/-- Build a seeded convolution, activation, and optional dropout block. -/
def convBlock {d : Nat} {inputChannels : Nat}
    (spatial : Tensor Nat [d]) (config : ConvBlock.Config d)
    (batchShape : Shape := []) :
    Builder (Sequential
      (batchShape ++ (Shape.dim inputChannels (spatial.to Shape)))
      (batchShape ++
        (Shape.dim config.convolution.outChannels
          ((config.convolution.convOutputSpatial spatial).to Shape)))) :=
  match config.validate inputChannels spatial (kind := "ConvBlock") with
  | .error message =>
      pure <| nn.Sequential.invalid
        (batchShape ++ (Shape.dim inputChannels (spatial.to Shape)))
        (batchShape ++
          (Shape.dim config.convolution.outChannels
            ((config.convolution.convOutputSpatial spatial).to Shape)))
        "ConvBlock" message
  | .ok () =>
      withInitializationSeed config.convolution.weightInit fun kernelSeed =>
        withOptionalDropoutSeed config.dropout? fun dropoutSeed =>
          pure <| nn.Sequential.convBlock batchShape spatial config kernelSeed dropoutSeed

/-- Build a seeded convolution/activation block followed by max pooling. -/
def convPoolBlock {d : Nat} {inputChannels : Nat}
    (spatial : Tensor Nat [d]) (config : ConvPoolBlock.Config d)
    (batchShape : Shape := []) :
    Builder (Sequential
      (batchShape ++ (Shape.dim inputChannels (spatial.to Shape)))
      (batchShape ++
        (Shape.dim config.block.convolution.outChannels
          ((config.pooling.poolOutputSpatial
            (config.block.convolution.convOutputSpatial spatial)).to Shape)))) :=
  match config.validate inputChannels spatial (kind := "ConvPoolBlock") with
  | .error message =>
      pure <| nn.Sequential.invalid
        (batchShape ++ (Shape.dim inputChannels (spatial.to Shape)))
        (batchShape ++
          (Shape.dim config.block.convolution.outChannels
            ((config.pooling.poolOutputSpatial
              (config.block.convolution.convOutputSpatial spatial)).to Shape)))
        "ConvPoolBlock" message
  | .ok () => do
      let block ← convBlock spatial config.block batchShape
      let pool ← maxPool (config.block.convolution.convOutputSpatial spatial) config.pooling
        batchShape
      pure <| compose block pool

/--
Build a multilayer perceptron over any `batchShape`.

Each hidden width contributes a linear layer followed by the configured activation and optional
dropout. Initialization seeds come from the surrounding `Builder` seed stream.

Example:
```lean
-- `16 -> 32 -> 32 -> 1`, ReLU between hidden layers, dropout after each one.
def model : nn.Builder (nn.Sequential [16] [1]) :=
  nn.mlp 16 1 { hiddenWidths := [32, 32], activation := relu, dropout? := some 0.1 }
```
-/
def mlp (inputWidth outputWidth : Nat) (config : MLP.Config := {})
    (batchShape : Shape := []) :
    Builder
      (Sequential
        (batchShape ++ [inputWidth])
        (batchShape ++ [outputWidth])) :=
  let rec
    /-- Linear layers from `currentWidth` through `hiddenWidths` to `outputWidth`, with the
    activation and the optional dropout after each hidden layer. -/
    buildStages (currentWidth : Nat) (hiddenWidths : List Nat) :
        Builder
          (Sequential
            (batchShape ++ [currentWidth])
            (batchShape ++ [outputWidth])) :=
      match hiddenWidths with
      | .nil => linear currentWidth outputWidth (batchShape := batchShape)
      | .cons hiddenWidth remainingWidths => do
          let affine ← linear currentWidth hiddenWidth (batchShape := batchShape)
          let hiddenShape := batchShape ++ [hiddenWidth]
          let activated : Sequential hiddenShape hiddenShape :=
            Sequential.fromLayer
              (Layer.op config.activation.op config.activation.need_op)
          let stage : Sequential
              (batchShape ++ [currentWidth]) hiddenShape :=
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
      pure <| nn.Sequential.invalid
        (batchShape ++ [inputWidth])
        (batchShape ++ [outputWidth])
        "MLP"
        message
  | .ok () => buildStages inputWidth config.hiddenWidths

/--
A model built from `builder` with a seed drawn at run time.

`nn.Sequential` lives in `Type 1`, so `IO` cannot return it directly. The record stores only the
drawn seed. Because `builder` is deterministic, `Built.model` recovers the same model every time.
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
Draw the next seed of the global seed stream for `builder` and return it as an `IO` value.
`(← nn.buildIO builder).model` is the model built from that seed, and `Random.manualSeed` resets
the stream, as `torch.manual_seed` does.
-/
def buildIO {σ τ : Shape} (builder : Builder (Sequential σ τ)) : IO (Built builder) := do
  let seed ← Random.nextSeedGlobal
  pure ⟨seed⟩


end nn


end TorchLean
