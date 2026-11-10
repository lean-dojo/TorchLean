# Data Examples

The data examples run both sides of TorchLean's data boundary. Python downloads or converts
ecosystem formats. Lean loads `.npy`, numeric CSV, or text, checks the declared shape, and
produces typed datasets.

For the API design and source types, read `TorchLean/NN/Data/README.md`.

## Start Here

Generate the small local fixtures and run all three loader commands:

```bash
python3 TorchLeanExamples/Data/generate_small_data.py
scripts/lake.sh exe torchlean data_csv
scripts/lake.sh exe torchlean data_npy
scripts/lake.sh exe torchlean data_cifar10 --check-only --epochs 1 --batch 4 --train-size 8 --n-total 20
```

The generator writes these git-ignored fixtures:

| Task | Files | Shape |
| --- | --- | --- |
| tabular regression | `small_regression.csv` | rows `x1,x2,y` |
| tensor regression | `small_regression_{X,y}.npy` | `(25, 2)`, `(25, 1)` |
| image classification | `small_cifar10like_{X,y}.npy` | `(200, 3, 32, 32)`, `(200,)` |

## Real Example Data

```bash
python3 scripts/datasets/download_example_data.py --auto-mpg --cifar10 --tiny-shakespeare
python3 scripts/datasets/download_example_data.py --household-power --household-power-windows 512
```

The download script fills `data/real/` for the MLP, vision, sequence, and forecasting examples. Each
model command's `--help` lists its data flags, and a missing file stops the command with a
preparation hint.

## Convert Other Formats

Use the converter for `.pt`, `.npz`, `.mat`, image folders, and other Python-side sources:

```bash
python3 scripts/datasets/torchlean_data_convert.py --help
```

A typical paired export is:

```bash
python3 scripts/datasets/torchlean_data_convert.py tensor \
  --input dataset.npz --key images --output data/real/mytask/X.npy --manifest
python3 scripts/datasets/torchlean_data_convert.py tensor \
  --input dataset.npz --key labels --output data/real/mytask/y.npy --manifest
```

With `--manifest`, the converter writes a JSON sidecar next to each array (`X.npy.json` for `X.npy`)
that records the source file, the key, the shape, and the dtype. Lean checks each array's shape
against the shape declared in the Lean code before it builds a dataset.

## Lean Side

Application code imports:

```lean
import TorchLean.NN.Data.CausalLM
open TorchLean
```

`TorchLean.NN.Data.Dataset` provides every source below except text, which needs
`TorchLean.NN.Data.CausalLM`.

Use the source that matches the boundary:

| Boundary | Source |
| --- | --- |
| tensors already in Lean | `Data.fromTensors`, `Data.fromSamples` |
| one tensor file | `Tensor.load` |
| `X.npy` and tensor targets | `Data.SupervisedSource` |
| `X.npy` and class labels | `Data.LabeledSource` |
| one numeric CSV | `Data.TabularSupervisedSource` |
| text | `Data.CausalLM.byteSamples`, `Data.CausalLM.tokenBatch` |

Class labels are numeric ids, each checked against the class count (`Fin classes`) before
one-hot conversion. The resulting sample shape is part of `Trainer.Dataset input target`.

## Boundary Rule

Training data may be shuffled and summarized by logs. Verification fixtures should be small and
reproducible, and they should state their ordering and tolerances. When an exported prediction
becomes a verification input, document the command that produced it and the checker that
consumes it.

List every public dataset and third-party artifact in `docs/THIRD_PARTY_NOTICES.md`.
