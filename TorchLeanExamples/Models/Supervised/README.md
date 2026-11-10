# Supervised Examples

Each example here learns a supervised map from inputs to targets. The models use the ordinary
public path on purpose: trainer construction, data loaders, losses, optimizers, scalar and backend
choices, prediction, training, and logging. That makes them the place to check that the everyday
API reads like TorchLean user code and keeps runtime plumbing out of sight.

## Files

- `MLP.lean`: tabular regression over the Auto MPG example data.
- `KAN.lean`: a compact KAN-style supervised model path using the same trainer API.
- `LSTMRegression.lean`: time-series forecasting from UCI household-power windows.

`LSTMRegression.lean` uses an LSTM, but its task is supervised forecasting
(`past window -> target window`), so it belongs here. `TorchLeanExamples/Models/Sequence/` holds
the sequence-model mechanics: recurrent layer checks, Transformer blocks, GPT-style language
modeling, Mamba-style state updates, and synthetic sequence curricula.

## Useful Commands

```bash
python3 scripts/datasets/download_example_data.py --auto-mpg
scripts/lake.sh exe torchlean mlp --device cpu --steps 10
scripts/lake.sh exe torchlean kan --device cpu --steps 10

python3 scripts/datasets/download_example_data.py --household-power --household-power-windows 512
scripts/lake.sh -Kcuda=true exe torchlean lstm_regression --device cuda --steps 200 --windows 96
```

Training commands write a JSON log by default, and `--log PATH` sets its path (`--log off`
disables it). The run below writes `data/examples/mlp_trainlog.json`:

```bash
scripts/lake.sh exe torchlean mlp --device cpu --steps 50 --log data/examples/mlp_trainlog.json
```

The log is the stable artifact for comparing runs. Printed predictions are useful for a quick read,
but the log records the run metadata, metric names, steps, and values.

## What Each Example Exercises

| Example | Data boundary | Main runtime path | Artifact to inspect |
| --- | --- | --- | --- |
| `MLP.lean` | Auto MPG CSV or a custom CSV with the same columns | public `Trainer` regression path | `TrainLog` and loss summary |
| `KAN.lean` | same tabular loader conventions | public trainer with KAN-style model structure | loss curve and parameter shapes |
| `LSTMRegression.lean` | household-power windows exported to `.npy` | CUDA-capable sequence layer used for supervised forecasting | forecast rows and `TrainLog` |

Run these examples after a change to the public API, since they cover the common path: build a
model, choose a backend, iterate batches, compute a loss, update parameters, and emit a
`TrainLog`. Certificate-producing workflows are in `TorchLeanExamples/Verification`, and scientific
operator learning is in `TorchLeanExamples/Models/Operators`.

For contributor conventions, see [contributing](../../../docs/CONTRIBUTING.md).
