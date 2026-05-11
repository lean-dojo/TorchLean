# Quickstart

This folder is the starting path for TorchLean. It focuses on core moves rather than the full model zoo. The goal is
to show the core moves clearly:

- build typed tensors,
- inspect values with editor widgets,
- use autograd helpers,
- train one small MLP,
- and see how ordinary Lean proofs fit into the workflow.

For larger architectures, use `NN/Examples/Models`. For datasets and loaders backed by files, use
`NN/Examples/Data/Loaders`. For runnable verification examples and bundled certificate fixtures,
start with `NN/Examples/Verification`; reusable checkers live under `NN/Verification`.

## Recommended Order

1. `TensorBasics.lean`
   Command: `lake exe torchlean quickstart_tensors`

2. `Widgets.lean`
   Editor-only: open the file and put the cursor on the `#tensor_view`, `#ir_view`, or
   `#train_log_view` commands.

3. `AutogradBasics.lean`
   Command: `lake exe torchlean quickstart_autograd --dtype float --backend eager`

4. `SimpleMlpTrain.lean`
   Command: `lake exe torchlean quickstart_mlp --steps 200 --dtype float --backend compiled`

5. `Proofs.lean`
   Build check: `lake build NN.Examples.Quickstart.Proofs`

## What About The CNN / ResNet Files?

`SimpleCnnTrain.lean`, `MinibatchMlpTrain.lean`, and `ResnetBasicblockTrain.lean` are still useful
follow up examples, but they are no longer the main quickstart path. They overlap with the model and
data tutorials, so new users should read them after the five files above.

## Scalar Constraints

Many examples are polymorphic in a scalar type `α`, for example:

```lean
def buildDataset {α : Type} [Semantics.Scalar α] [Runtime.Scalar α] := ...
```

Read those constraints as two separate roles:

- `Semantics.Scalar α`: the model and loss can do their mathematical scalar operations over `α`.
- `Runtime.Scalar α`: examples can inject host `Float` literals into the selected backend.

That split is what lets the same tutorial run with `--dtype float`, `--dtype float32`, or other
runtime scalar backends.
