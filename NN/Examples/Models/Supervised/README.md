# Supervised Examples

This folder contains examples where the primary object is a supervised map from inputs to targets.
The models use the ordinary public path on purpose: trainer construction, data loaders, losses,
optimizers, scalar/backend choices, prediction, training, and logging. That makes this folder the
place to check whether the everyday API still feels like TorchLean user code rather than runtime
plumbing.

## Files

- `Mlp.lean`: tabular regression over the Auto MPG example data.
- `Kan.lean`: a compact KAN-style supervised model path using the same trainer API.
- `LstmRegression.lean`: time-series forecasting from UCI household-power windows.

The architecture alone does not decide the folder. `LstmRegression.lean` uses an LSTM, but the task
is supervised forecasting (`past window -> target window`), so it belongs here. The `Sequence/`
folder is for sequence-model behavior itself: recurrent layer checks, Transformer blocks,
GPT-style language modeling, Mamba-style state updates, and synthetic sequence curricula.

## Useful Commands

```bash
python3 scripts/datasets/download_example_data.py --auto-mpg
lake exe torchlean mlp --device cpu --steps 10
lake exe torchlean kan --device cpu --steps 10

python3 scripts/datasets/download_example_data.py --household-power --household-power-windows 512
lake -R -K cuda=true exe torchlean lstm_regression --device cuda --steps 200 --windows 96
```

Pass `--log PATH` to preserve the training curve:

```bash
lake exe torchlean mlp --device cpu --steps 50 --log data/examples/mlp_trainlog.json
```

The log is the stable artifact for comparing runs. Printed predictions are useful for a quick read,
but the log records the run metadata, metric names, steps, and values.

## What Each Example Exercises

| Example | Data boundary | Main runtime path | Artifact to inspect |
| --- | --- | --- | --- |
| `Mlp.lean` | Auto MPG CSV or generated tabular tensors | public `Trainer` regression path | `TrainLog`, before/after predictions |
| `Kan.lean` | same tabular loader conventions | public trainer with KAN-style model structure | loss curve and parameter shapes |
| `LstmRegression.lean` | household-power windows exported to `.npy` | CUDA-capable sequence layer used for supervised forecasting | forecast rows and `TrainLog` |

These examples are useful when changing the public API because they touch the common path: build a
model, choose a backend, iterate batches, compute a loss, update parameters, and emit a `TrainLog`.
For certificate-producing workflows, use `NN/Examples/Verification`. For scientific operator
learning, use `NN/Examples/Models/Operators`. If a supervised run later feeds a proof or checker,
the verification page should name the exported artifact and the Lean statement that consumes it.

## Public API Expectations

Supervised examples should keep the public path direct:

- start from `import NN.API` and `open TorchLean`;
- construct a model with `TorchLean.nn`;
- load data through `TorchLean.Data`;
- construct with `Trainer.new`, train with `trainer.train`, and predict with the trained result;
- keep manual runtime hooks out of the tutorial path unless the example explicitly explains why.

That consistency matters because these files exercise the application API. If a
backend or optimizer change forces ordinary supervised examples to know about implementation
internals, the application API probably needs cleanup.
