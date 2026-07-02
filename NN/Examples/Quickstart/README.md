# Quickstart

This folder is the starting path for TorchLean. It focuses on the moves people copy first:

- build typed tensors,
- inspect values with editor widgets,
- use autograd APIs,
- train small MLP and CNN models,
- and see how ordinary Lean proofs fit into the workflow.

New code should start from the root API:

```lean
import NN
open TorchLean
```

The quickstarts should not make you learn subsystem namespaces first. Use `TorchLean.nn`,
`TorchLean.Tensor`, `TorchLean.autograd`, `TorchLean.Trainer`, and `TorchLean.Data` here. Drop into
`NN.*` when you are extending TorchLean itself, proving runtime facts, or working with subsystem
entrypoints.

For larger architectures, use `NN/Examples/Models`. For datasets and loaders backed by files, use
`NN/Examples/Data/Loaders`. Runnable verification examples and bundled certificate artifacts live
under `NN/Examples/Verification`; reusable checkers live under `NN/Verification`.

## Recommended Order

1. `TensorBasics.lean`
   Command: `lake exe torchlean quickstart_tensors`

2. `StarterWorkflow.lean`
   Build check: `lake build NN.Examples.Quickstart.StarterWorkflow`
   This is the API shape we want people to copy first: `import NN`, `nn.Sequential![...]`,
   `Data.tensorDataset xs ys`, `Trainer.new model { task := .regression, optimizer := ... }`,
   `trainer.eval x`, `trainer.train data { steps := ..., batchSize := ..., logEvery := ... }`,
   and then one trained-handle prediction plus one `ℓ∞` IBP verification call. The task field chooses
   the loss kind: regression means MSE, classification means one-hot cross entropy.

3. `Widgets.lean`
   Editor-only: open the file and put the cursor on the `#tensor_view`, `#ir_view`, or
   `#train_log_view` commands.

4. `AutogradBasics.lean`
   Command: `lake exe torchlean quickstart_autograd`
   This uses `TorchLean.autograd` directly, so the example shows VJP/Jacobian/HVP without exposing
   the runtime callback machinery.

5. `SimpleMlpTrain.lean`
   Command: `lake exe torchlean quickstart_mlp --steps 200 --dtype float --backend compiled`

6. `MinibatchMlpTrain.lean`
   Command: `lake exe torchlean quickstart_minibatch_mlp --steps 30 --batch 5 --dtype float --backend eager`

7. `SimpleCnnTrain.lean`
   Command: `lake exe torchlean quickstart_cnn --steps 5 --batch 2 --dtype float --backend eager`

8. `Proofs.lean`
   Build check: `lake build NN.Examples.Quickstart.Proofs`

## Where The Bigger Examples Live

The quickstarts use the `NN` public API and keep ordinary training behind `Trainer` and
`Data`. Larger architectures, real datasets, PyTorch interop, RL, and verification examples live in
their specialized folders under `NN/Examples`.
