import VersoManual
import VersoBlueprint

open Verso.Genre Manual

#doc (Manual) "Building Models With Layers" =>
%%%
tag := "building-models"
%%%

Once tensors have shapes, a model can be read as a typed map between tensor spaces. TorchLean's
layer API is meant to feel familiar: linear layers, activations, convolutions, residual blocks, and
attention blocks compose into sequential models. The difference is that the input and output shapes
are visible before the model runs.

```
model : Tensor alpha inputShape -> Tensor alpha outputShape
```

The three names to remember are `nn.M`, `nn.Sequential`, and `nn.build`.

- `nn.M` is the model builder. It allocates parameter seeds and assembles layers.
- `nn.Sequential sigma tau` is a model from shape `sigma` to shape `tau`.
- `nn.build seed mkModel` fixes the initialization seed and returns the trainable model package.

The result still feels close to PyTorch's `nn.Sequential`, but the input and output shapes are part
of the Lean type.

# A First MLP

The smallest useful pattern is a multilayer perceptron:

```
open Spec
open NN.Tensor
open NN.API

def inDim : Nat := 2
def hidden : Nat := 8
def outDim : Nat := 1

def mkModel : nn.M (nn.Sequential (Shape.Vec inDim) (Shape.Vec outDim)) :=
  nn.sequential![
    nn.linear inDim hidden (pfx := Spec.Shape.scalar),
    nn.relu,
    nn.linear hidden outDim (pfx := Spec.Shape.scalar)
  ]

def task (seed : Nat) :=
  train.regression (nn.build seed mkModel)
```

There are three things to notice.

First, the model type says exactly what the model accepts and returns:

```
nn.Sequential (Shape.Vec 2) (Shape.Vec 1)
```

Second, `nn.linear inDim hidden` is not just a runtime operation. It also describes the parameter
shapes for the weight and bias. A layer `nn.linear 2 8` introduces the usual affine parameters for
mapping two features to eight features under the selected prefix convention. When `nn.build` runs,
it creates the initial parameter bundle for that layer.

Third, `train.regression` attaches a loss convention to the model. The model definition and the
training task are separate: the same model shape can appear in a regression task, a classification
task, an export path, or a proof statement.

The runnable file is
[NN.Examples.Quickstart.SimpleMlpTrain](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Quickstart/SimpleMlpTrain.lean).

# Prefix Shapes: The Batch Axis Story

If you are coming from PyTorch, the prefix shape is the one new idea to slow down for. A linear
layer acts on the last dimension. The dimensions before it are the prefix. This matches the PyTorch
intuition that `Linear(inDim, outDim)` can be applied to inputs shaped `[..., inDim]`. TorchLean asks
you to name that prefix.

For one vector:

```
nn.linear 2 8 (pfx := Spec.Shape.scalar)
-- Shape.Vec 2 -> Shape.Vec 8
```

For a minibatch:

```
nn.linear 2 8 (pfx := Shape.Vec batch)
-- Shape.Mat batch 2 -> Shape.Mat batch 8
```

That prefix is why the minibatch MLP can be written with the same layer vocabulary:

```
def mkBatched {batch : Nat} :
    nn.M (nn.Sequential (Shape.Mat batch 2) (Shape.Mat batch 1)) :=
  nn.sequential![
    nn.linear 2 8 (pfx := Shape.Vec batch),
    nn.relu,
    nn.linear 8 1 (pfx := Shape.Vec batch)
  ]
```

The layer still transforms features from `2` to `8` to `1`. The prefix says that the operation is
applied across a batch.

The runnable minibatch example is
[NN.Examples.Quickstart.MinibatchMlpTrain](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Quickstart/MinibatchMlpTrain.lean).

# Images and CNNs

Image models use the same `nn.sequential!` style, but the input shape is now an image batch:

```
Shape.Images batch channels height width
```

A small CNN looks like this:

```
def mkCnn {batch : Nat} :
    nn.M (nn.Sequential (Shape.Images batch 1 4 4) (shape![batch, 2])) :=
  let outC : Nat := 3
  let outH : Nat := (4 - 2) / 1 + 1
  let outW : Nat := (4 - 2) / 1 + 1
  let featInner : Shape := Shape.Image outC outH outW
  let featSize : Nat := Spec.Shape.size featInner
  nn.sequential![
    nn.conv (n := batch) (inC := 1) (inH := 4) (inW := 4)
      { outC := outC, kH := 2, kW := 2, stride := 1, padding := 0 },
    nn.relu,
    nn.flattenBatch,
    nn.linear featSize 2 (pfx := Shape.Vec batch)
  ]
```

Here the shape bookkeeping is part of the model definition:

- the convolution maps `N x 1 x 4 x 4` to `N x 3 x 3 x 3`;
- `nn.flattenBatch` keeps the batch axis and flattens the feature axes;
- the final linear layer maps each flattened image to two logits.

The CNN example is valuable because the axes are no longer implicit. The type records that a batch
of images enters, the convolution changes the channel and spatial axes, and two logits per image
leave. Later chapters use the same information when they lower the model to graphs or discuss
verification conditions.

The runnable CNN tutorial is
[NN.Examples.Quickstart.SimpleCnnTrain](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Quickstart/SimpleCnnTrain.lean).

# Residual Blocks

Residual models are useful because they force the API to express a shape preserving path:

```
input -> block(input) + skip(input)
```

The public builder for the small ResNet tutorial is `nn.resnetBasicBlock`. A typical shape is:

```
nn.resnetBasicBlock (n := batch) (inC := 8) (h := 4) (w := 4)
  { outC := 8, stride := 1 }
```

Read the type as a contract: if the block is used in the no downsample case, the residual path and
the main path have compatible output shapes. If a downsample is requested, the block records the
shape change explicitly.

The tutorial file is
[NN.Examples.Quickstart.ResnetBasicblockTrain](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Quickstart/ResnetBasicblockTrain.lean).
The larger model wrappers live under [NN/API/Models/Resnet.lean](https://github.com/lean-dojo/TorchLean/blob/main/NN/API/Models/Resnet.lean).

# Transformer Shaped Blocks

Sequence models use the same principle. A transformer block is a typed map over a batched sequence:

```
shape![batch, seqLen, dModel]
```

The public constructors include:

- `nn.multiheadAttention`,
- `nn.layerNorm`,
- `nn.transformerEncoderBlock`,
- and model wrappers under [NN/API/Models/Transformer.lean](https://github.com/lean-dojo/TorchLean/blob/main/NN/API/Models/Transformer.lean),
  [NN/API/Models/Gpt2.lean](https://github.com/lean-dojo/TorchLean/blob/main/NN/API/Models/Gpt2.lean), and
  [NN/API/Models/Vit.lean](https://github.com/lean-dojo/TorchLean/blob/main/NN/API/Models/Vit.lean).

A small block reads as:

```
nn.transformerEncoderBlock
  (batch := batch) (n := seqLen) (dModel := dModel)
  { numHeads := 2, headDim := dModel / 2, ffnHidden := 4 * dModel }
```

The advanced-model chapters give the longer model zoo tour. Here the lesson is simpler: MLPs, CNNs,
ResNets, and transformer blocks all enter through the same typed model-building path. State the
shape, choose the layers, build parameters, then train or inspect the resulting task.

# Parameters Are Explicit

PyTorch stores parameters inside module objects. TorchLean keeps the parameter bundle explicit. That
choice makes the training loop and the verification path much easier to read.

Informally:

```
nn.build seed mkModel
-- produces a model structure together with initialized parameters
```

That package is what `train.regression`, `train.classificationOneHot`, and the other task builders
consume. Training updates the parameter bundle. Prediction evaluates the same structure with the
current parameter values.

If this looks verbose for a two-layer MLP, remember that the same structure is what later lets us
lower the model to a graph and check a certificate without rediscovering the parameter shapes.

# Choosing a Task

After a model is built, choose the task that matches the target:

- use `train.regression` for mean squared error style vector targets;
- use `train.classificationOneHot` for one hot classification targets;
- use the model zoo wrappers when a family has a specialized objective or data shape.

The shape of the task matches the shape of the model:

```
train.Task inputShape targetShape
```

That is the bridge between model building and training. The next page explains how datasets and
loaders produce samples with those same shapes.
