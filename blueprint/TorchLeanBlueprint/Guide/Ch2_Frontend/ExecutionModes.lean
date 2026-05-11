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

With CUDA enabled, model demos that support device buffers add a device choice:

```
lake exe torchlean mlp --cuda --steps 100
```

The flags change how the model is evaluated. They do not change the layer structure or parameter
shapes.

The runtime choices are easiest to read this way:

- Scalar choice: `--dtype float` or `--dtype float32` changes the numeric representation. The model
  shape and architecture stay fixed.
- Backend choice: `--backend eager` or `--backend compiled` changes whether the run produces an
  eager tape or a reusable graph-shaped artifact.
- Device choice: `--cpu` or `--cuda` changes whether supported numeric buffers live on the host or
  on CUDA device memory.
- Mode choice: train/eval mode changes runtime behavior for mode-sensitive layers such as dropout
  or normalization. The declared model and parameter payload stay visible.

# What A Runner Owns

The public training examples usually enter the runtime through `train.run`:

```
train.run task args (fun {α} _ _ _ _ runner rest => do
  ...
)
```

Read this callback as:

- `α` is the concrete scalar type selected by the command line, such as `Float` or `Float32`.
- `runner` is the instantiated runtime object for the task.
- `rest` contains the remaining command line arguments after the common runtime flags have been
  parsed.

A runner is the object that turns a task into repeated executable steps. If the model definition is
the recipe, the runner is the instantiated kitchen: it has the ingredients, backend, and mode needed
to actually run the recipe.

The runner owns the pieces a training loop needs:

- initialized parameters and buffers,
- compiled predictors for train and eval mode,
- compiled losses for train and eval mode,
- the current mode flag,
- update helpers used by optimizers and fitting loops.

The main API surface is:

- [NN.API.Runtime API](https://github.com/lean-dojo/TorchLean/blob/main/NN/API/Runtime.lean)
- [NN.API.Public API](https://github.com/lean-dojo/TorchLean/blob/main/NN/API/Public.lean)

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

Use eager mode when the goal is to understand one step, inspect a gradient, or explain what a small
example is doing.

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

Some layers behave differently while fitting than they do while evaluating. Dropout and batch
normalization are the common examples. TorchLean keeps this state in the runner, rather than
pretending that every layer is purely stateless at runtime.

The mental model is the same as PyTorch's `model.train()` and `model.eval()`, but the mode is part
of the explicit runtime object.

# Loaders and Epochs

Fitting helpers call the same runner repeatedly over datasets or loaders. The runtime choice is
independent of whether examples arrive as one batch or many minibatches.

The declarations to open when reading the code are:

- `train.FitConfig`
- `train.LoaderFitConfig`
- `train.fitDataset`
- `train.fitLoader`

Those live in [NN.API.Runtime API](https://github.com/lean-dojo/TorchLean/blob/main/NN/API/Runtime.lean), with public re-exports in
[NN.API.Public API](https://github.com/lean-dojo/TorchLean/blob/main/NN/API/Public.lean).

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
