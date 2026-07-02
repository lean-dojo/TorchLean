import VersoManual

open Verso.Genre Manual

#doc (Manual) "Execution Modes and Runners" =>
%%%
tag := "execution-modes"
%%%

By now we can build a model and give it data. The next question is how the model runs.

Most TorchLean examples make three runtime choices: the scalar type, the execution backend, and the
device. These choices affect the artifact produced by the run. Eager mode produces a tape that is
easy to inspect. Compiled mode produces a reusable graph-shaped artifact. CUDA mode places supported
Float32 work on device buffers. The model architecture and parameter shapes should remain fixed.

# The Three Runtime Choices

Most runnable examples answer three questions.

1. Which scalar type should the program use?
2. Which execution backend should run the graph?
3. Which device should hold the numeric buffers?

Typical command line flags look like:

```
lake env lean --run NN/Examples/Quickstart/SimpleMlpTrain.lean -- \
  --dtype float --backend eager --steps 100

lake env lean --run NN/Examples/Quickstart/SimpleMlpTrain.lean -- \
  --dtype float --backend compiled --steps 100
```

With CUDA enabled, model examples that support device buffers add a device choice:

```
lake exe -K cuda=true torchlean mlp --cuda --steps 100
lake exe -K cuda=true torchlean mlp --cuda --steps 1000 --cuda-mem-watch 100
```

The flags change how the model is evaluated. They do not change the layer structure or parameter
shapes. The `--cuda-mem-watch` flag is only a runtime diagnostic: it asks the example to print
CUDA allocator samples during training so that long runs expose memory drift while they are still
running. The public model examples use the same step-counted training convention here: `--steps`
means optimizer updates, and loader-based examples stop after that many updates rather than after an
accidental number of data-loader passes.

The runtime choices are easiest to read this way:

- Scalar choice: `--dtype float` or `--dtype float32` changes the numeric representation. The model
  shape and architecture stay fixed.
- Backend choice: `--backend eager` or `--backend compiled` changes whether the run produces an
  eager tape or a reusable graph-shaped artifact.
- Device choice: `--cpu` or `--cuda` changes whether supported numeric buffers live on the host or
  on CUDA device memory.
- CUDA diagnostics: `--cuda-mem-watch N` samples native allocator state every `N` training updates.
  Long CUDA model runs choose a small default cadence so the terminal shows whether device memory is
  steady, growing slowly, or approaching exhaustion.
- Mode choice: train/eval mode changes runtime behavior for mode-sensitive layers such as dropout
  or normalization. The declared model and parameter payload stay visible.

# What The Public Runtime Owns

Most tutorial code does not instantiate a runner manually. It builds one trainer with a public
`Trainer.Config`, then lets `trainer.train data trainOptions` run with those scalar/backend/device
settings.

That public runtime path still creates one executable runtime object under the hood. The useful mental
model is:

- the declared model stays the same,
- the runtime layer chooses the scalar/backend/device interpretation,
- the instantiated runtime object owns parameters, buffers, mode, and compiled executable artifacts.

If the model definition is the recipe, the runtime object is the instantiated kitchen: it has the
ingredients, backend, and mode needed to actually run the recipe.

The quickstart API surface is:

- `Trainer.Config`
- `Trainer.TrainOptions`
- `Trainer.new`
- `trainer.train data trainOptions`
- `trained.eval ...` / `trained.printPrediction ...`

The advanced runner API still exists, but it sits under
`Trainer.Advanced`.  Use that path when you really need manual callbacks, explicit mode changes, or
custom runtime loops.

# Eager Mode

Eager mode is the easiest backend to reason about while debugging. It records operations as they run,
keeps a tape of parent links and local reverse rules, and returns explicit gradients.

The closest PyTorch analogy is:

```
loss = model(x).loss(y)
loss.backward()
```

In TorchLean, the corresponding data remains explicit:

- the tape is a Lean value,
- gradients are returned as tensors or parameter bundles,
- widgets can inspect the recorded graph.

Use eager mode to understand one step, inspect a gradient, or explain what a small example is doing.

# Compiled Mode

Compiled mode builds a stable graph-shaped runtime artifact before repeated execution. It is the
better default for longer runs when the model and loss are fixed, because the graph structure is
constructed once and then reused.

The closest PyTorch analogy is the intuition behind `torch.compile`: keep the model code, but first
turn its computation into a reusable graph. TorchLean's compiled path is more explicit because the
artifact is a Lean runtime object and can be related to the IR and proof layers.

Use compiled mode when the model and loss are fixed and the training loop will run many steps or
many batches.

# Train Mode and Eval Mode

Some layers behave differently during training than they do while evaluating. Dropout and batch
normalization are the common examples. TorchLean keeps this state in the runner, rather than
pretending that every layer is purely stateless at runtime.

The mental model is the same as PyTorch's `model.train()` and `model.eval()`, but the mode is part
of the explicit runtime object.

# Loaders and Epochs

Training calls run the same runner repeatedly over datasets or loaders. The runtime choice is
independent of whether examples arrive as one batch or many minibatches.

The public declarations to read first are:

- `Trainer.Config`
- `Trainer.TrainOptions`
- `Trainer.new`
- `trainer.train data trainOptions`

Lower-level dataset and loader loops still exist for custom runtime code, but ordinary examples
should keep dtype, backend, device, and optimizer in the config passed to `Trainer.new`, and
per-training steps/logging in `Trainer.TrainOptions`.

# Same Architecture, Different Artifact

It helps to keep this small map in mind:

- `--dtype float` versus `--dtype float32` changes the scalar representation. Tensor shapes and
  model structure stay fixed.
- `--backend eager` changes the runtime artifact to tape style execution. The public model
  definition stays fixed.
- `--backend compiled` changes the runtime artifact to a reusable compiled graph. The public model
  definition stays fixed.
- `--cpu` versus `--cuda` changes where supported numeric buffers live. The Lean specification and
  declared runtime assumptions stay fixed.

That separation is the reason TorchLean can be used as a tutorial framework, an executable
experiment harness, and a verification codebase at the same time.

# What To Read Next

For a small runnable example, open [SimpleMlpTrain](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Quickstart/SimpleMlpTrain.lean).
For minibatches and epochs, open
[MinibatchMlpTrain](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Quickstart/MinibatchMlpTrain.lean). For the declarations behind
the runner, open [NN.API.Runtime](https://github.com/lean-dojo/TorchLean/blob/main/NN/API/Runtime.lean).
