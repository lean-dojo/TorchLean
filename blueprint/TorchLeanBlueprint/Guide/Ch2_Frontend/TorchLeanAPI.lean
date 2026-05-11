import VersoManual
import VersoBlueprint

open Verso.Genre Manual

#doc (Manual) "TorchLean API" =>
%%%
tag := "torchlean-api"
%%%

This page is a map of the public API. The short version is:

```
import NN
open NN.API
```

That is enough for ordinary examples. It gives access to the names used throughout the tutorials:
`nn` for models, `Data` for datasets, `train` for fitting and reports, `optim` for optimizers, and
`autograd` for lower-level differentiation tools.

For a first pass through TorchLean, read the API this way:

- `nn` builds models;
- `Data` builds datasets and loaders;
- `train` runs fitting, prediction, callbacks, and reports;
- `optim` configures updates;
- `autograd` exposes lower level gradient tools when the training helper is not enough.

The first thing to keep in mind is the layering:

- `NN.API.Public` is the public user surface.
- `NN.API.Runtime` exposes the lower layer executable runtime surface.
- `NN.API.TorchLean` is the namespace that the runtime surface re-exports for ordinary code.
- `NN.Entrypoint.*` modules provide narrow wrappers for specialized proof or runtime paths.

Rule of thumb: use `import NN` and `open NN.API` first. Drop to the runtime layer only when the
runtime itself is the subject.

# The First Line Of A Tutorial

The shortest useful setup is:

```
import NN

open NN.API

#check API.nn
#check API.train
#check API.optim
#check TorchLean.Session.new
```

That is enough for the ordinary tutorial path. Users should not have to choose between a dozen
internal imports before they have built their first model.

# The Public API

`NN.API.Public` is the canonical module for ordinary user code. It is designed to feel familiar to
readers who know PyTorch:

- `API.nn` for layer builders and model constructors,
- `API.Data` for datasets, loaders, and small preprocessing helpers,
- `API.train` for fit/predict loops and training utilities,
- `API.optim` for optimizer configuration,
- `API.autograd` for gradient, VJP, and Jacobian helpers,
- `API.rand` for deterministic RNG helpers.

Tutorials, notebooks, and examples should prefer this layer. It does not hide the runtime; it gives
the runtime a stable and legible public interface.

The useful namespace map is:

- `nn`: layers, blocks, and model builders.
- `Data`: datasets, sources, loaders, and transforms.
- `train`: fitting, prediction, callbacks, reports, and manual steps.
- `optim`: SGD, Adam, AdamW, and scheduler configuration.
- `autograd`: gradients, VJPs, Jacobians, and lower-level differentiation.
- `NN.Entrypoint.IR`: graph-level examples.
- `NN.Entrypoint.Verification`: verifier and certificate examples.
- `NN.Entrypoint.Floats`: Float32 and numeric-semantics examples.

The user should not need to rewrite the model to move from a small training run to graph inspection
or to a verifier fixture. The lower layer may require extra hypotheses, but it should be about the
same model.

Two parts of the public API deserve special emphasis:

- `API.Data`
  includes typed CSV/NPY readers, deterministic shuffling, minibatch loaders, and helper
  constructors for supervised and labeled datasets;
- `API.train`
  spans both full fitting helpers and manual step APIs that feel familiar from PyTorch.

# Data And Transform Surface

Closest TorchLean analogue to `torch.utils.data` plus a small, predictable preprocessing surface:

- file sources: `Data.TensorSource`, `Data.SupervisedSource`, `Data.LabeledSource`,
  `Data.TabularSupervisedSource`
- loading: `src.load` (on each source)
- deterministic minibatching: `Data.batchLoader` and `Data.BatchLoader.epoch`
- map-style transforms: `Data.Transforms` (combinators you can apply to samples or datasets)

The most important design decision is the boundary format: TorchLean expects numeric tensors in
`.npy` (and small numeric CSV for tabular data). For anything else, we usually convert once with Python
tools.

The intended data contract is:

$$`\mathrm{loader}(\mathrm{file})
\in
\mathrm{Except}\;\mathrm{String}\;
  \left(\mathrm{Tensor}(\alpha,s_x)\times\mathrm{Tensor}(\alpha,s_y)\right)`

Loading can fail, and shape checks happen before a dataset becomes training data. That is less
convenient than accepting every file and hoping for the best, but it gives later training and
verification code a typed object instead of an unchecked blob.

See the data contract documentation:

- [NN/API/Data/README.md](https://github.com/lean-dojo/TorchLean/blob/main/NN/API/Data/README.md)
- [NN/Examples/Data/README.md](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Data/README.md)

The public tutorials that exercise this surface are:

- [CSV loader example](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Data/Loaders/Csv.lean)
- [NPY loader example](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Data/Loaders/Npy.lean)
- [CIFAR image loader example](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Data/Loaders/Cifar10Images.lean)

This is worth saying explicitly because TorchLean examples now cover both small tutorial fixtures and
data backed training runs.

# The Runtime Surface

`NN.API.Runtime` and `NN.API.TorchLean` expose the lower layer machinery that underlies the public
API. That surface includes:

- typed and untyped tensor operations,
- backend-neutral programs and compiled outputs,
- session and execution control,
- losses, norms, optimizers, and training helpers,
- the explicit `eager` versus `compiled` execution distinction.

The inventory matters because the runtime layer keeps executable semantics visible enough for graph
inspection, verification, and interop work.

Typical names in that layer include:

- `add`, `matmul`, `reshape`, `transpose2d`, `broadcastTo`,
- `linear`, `conv2d`, `layer_norm`, `multi_head_attention`,
- `trainCycleSGD`, `trainCycleOptim`, `meanLoss`,
- `Backend`, `Options`, `Program`, `CompiledOut`.

Those names represent the lower half of the same public API:

$$`\mathrm{Program}_\alpha
\;\xrightarrow{\mathrm{run}}\;
\mathrm{value},\mathrm{tape/log/graph},\mathrm{state'}`

The returned artifact matters. It is what widgets display, what compiled execution replays, and
what proof chapters relate back to the spec.

# Training, Reporting, And Schedulers

The training API is best understood as two compatible layers.

## High-level fit helpers

- `train.fitDataset`
- `train.fitLoaderWith`
- `train.meanLossDataset`
- `train.predict`

These are the helpers most public tutorials use.

## Low-level manual-loop helpers

- `train.stepper`
- `train.step`
- `train.evalMode`
- `train.withMode`

Closest Lean analogue to a handwritten PyTorch training loop:

## Hooks and reporting

- `train.Callbacks`
- `train.onTrainStart`, `train.onStep`, `train.onEpochEnd`, `train.onTrainEnd`
- `train.Report.*`

These API entry points matter because a surprising amount of tutorial quality comes from reporting and
monitoring code, not only in the model definition.

## Optimizers and scheduler math

- `optim.sgd`, `optim.adam`
- `API.TorchLean.Schedulers.constant`
- `API.TorchLean.Schedulers.step`
- `API.TorchLean.Schedulers.exponential`

The scheduler layer is intentionally small and pure: it gives higher layer training code a stable
place to compute learning-rate policies without entangling that logic with runtime state.

# Execution Modes

TorchLean uses the same public model code with two execution styles:

- `eager` for debugging, gradients, and step by step inspection,
- `compiled` for a graph artifact that can be replayed or verified more directly.

That distinction matters because it keeps the choice of semantics explicit. The code does not change
just because the backend changes; only the execution mode changes.

Device note: the same APIs also compose with an optional CUDA backed float32 path when the
project is built with `-K cuda=true`. Public tutorials often show `--cpu` and `--cuda` on the
`torchlean` runner; the `NN.API` types do not change. Only the runtime buffer implementation does. See
*Runtime and Autograd* for the trust boundary and *Example Walkthroughs* for commands.

```
lake env lean --run NN/Examples/Quickstart/SimpleMlpTrain.lean -- --steps 200 --dtype float --backend compiled
```

The command above is a good reminder of one API invariant: the public API stays constant while the
backend varies.
Swapping `--backend eager` and `--backend compiled` shows the behavioral difference without rewriting
the model.

# When To Use A Lower Layer

Most users can stay at `import NN`. Use a lower layer when the lower layer is exactly what you are
explaining:

1. `NN.API.Public` for ordinary examples.
2. `NN.API.Runtime` or `NN.API.TorchLean` only when the lower layer API is the topic.
3. `NN.Entrypoint.*` for a narrow import tied to a specific narrative (specs, runtime, floats, or
   verification).

This ordering keeps the manual coherent: the public API appears first, then the runtime assembly.

Concrete import choices:

```
-- Beginner model/training example
import NN
open NN.API
```

```
-- Focused verifier example
import NN.Entrypoint.Verification
```

```
-- Focused Float32 example
import NN.Entrypoint.Floats
```

# Model Families Beyond The Smallest Tutorials

The public tutorials understandably emphasize MLPs and small CNNs, but the API/runtime surface has
grown further:

- residual CNN support via `nn.blocks.resnetBasicBlock`,
- transformer style blocks via `nn.multiheadAttention`, `nn.layerNorm`, and
  `nn.blocks.transformerEncoderBlock`,
- GraphSpec-backed runtime model wrappers such as `resnet18Program`,
- verifier friendly operator learning work such as the `fno1d` runtime model wrapper.

Not all of these families are beginner examples, but they are part of the model building surface that
advanced tutorials and experiments use.

# GraphSpec And Tooling

Three adjacent namespaces are easy to overlook from `import NN` alone.

## `NN.Entrypoint.*`

These are one-import wrappers for focused subsystems:

- `NN.Entrypoint.Spec`
- `NN.Entrypoint.Runtime`
- `NN.Entrypoint.Floats`
- `NN.Entrypoint.Verification`
- `NN.Entrypoint.GraphSpec`

They are especially useful in focused guide files or demos where the import surface should say
what the file is about.

## `NN.GraphSpec`

GraphSpec is the typed architecture DSL described in its own guide. It is not the default public
API, but it is already connected to the runtime model zoo and to public demos such as
`graphspec_mlp_demo`.

## Navigation tools

For exact declarations, use the generated API docs. For import structure, use the module graph. The
guide keeps only the names needed to explain the main contracts, so it stays readable while the
reference pages remain complete.

# How The Pieces Fit Together

The split between public and runtime namespaces gives the project three practical benefits:

- tutorials stay concise,
- implementation details stay inspectable rather than hidden behind opaque objects,
- verification chapters can refer to the same names as the examples.

The API rule is simple: start broad, then narrow only when the topic demands it. Use `import NN` for
ordinary model code. Use `NN.Entrypoint.*` when a file is specifically about graphs, floats,
verification, widgets, or runtime internals. That keeps examples readable while preserving precise
entry points for the lower layers.

See *Training From Scratch* for workflows, *Example Walkthroughs* for curated demos, and *PyTorch Round-Trip*
then *TorchLean vs PyTorch* for interop.

# References

- TorchLean paper: https://arxiv.org/abs/2602.22631
- Lean module-system reference:
  https://lean-lang.org/doc/reference/latest/Source-Files-and-Modules/
- PyTorch documentation for the corresponding surface concepts:
  https://pytorch.org/docs/stable/index.html
