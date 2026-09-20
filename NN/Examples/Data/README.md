# Data Examples

These examples show TorchLean's data boundary in action. Python downloads or converts ecosystem
formats; Lean loads `.npy`, numeric CSV, or text, checks the declared shape, and produces typed
datasets.

For the API design and source types, read `NN/API/Data/README.md`.

## Start Here

Generate the small local fixtures and run all three loader commands:

```bash
python3 NN/Examples/Data/generate_small_data.py
lake exe torchlean data_csv
lake exe torchlean data_npy
lake exe torchlean data_cifar10 --check-only --epochs 1 --batch 4 --train-size 8 --n-total 20
```

The generator writes ignored tutorial files for:

| Task | Files | Shape |
| --- | --- | --- |
| tabular regression | `small_regression.csv` | rows `x1,x2,y` |
| tensor regression | `small_regression_{X,y}.npy` | `(25, 2)`, `(25, 1)` |
| image classification | `small_cifar10like_{X,y}.npy` | `(200, 3, 32, 32)`, `(200,)` |
| operator learning | `small_fno1d_{X,y}.npy` | `(4, 32)`, `(4, 32)` |

## Real Example Data

```bash
python3 scripts/datasets/download_example_data.py --auto-mpg --cifar10 --tiny-shakespeare
python3 scripts/datasets/download_example_data.py --household-power --household-power-windows 512
```

These commands populate `data/real/` for the MLP, vision, sequence, and forecasting examples. Use
each model command's `--help` output for its exact paths and shapes.

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

The manifest records the producer, selected keys, and exported shapes. Lean still validates those
shapes before constructing a dataset.

## Lean Side

Application code imports:

```lean
import NN.API
open TorchLean
```

Use the source that matches the boundary:

| Boundary | Source |
| --- | --- |
| tensors already in Lean | `Data.fromTensors`, `Data.fromSamples` |
| one tensor file | `Tensor.load` |
| `X.npy` and tensor targets | `Data.SupervisedSource` |
| `X.npy` and class labels | `Data.LabeledSource` |
| one numeric CSV | `Data.TabularSupervisedSource` |
| text | `TorchLean.text` |

Classification label vectors contain numeric class ids and are checked before one-hot conversion.
The resulting sample shape is part of `Trainer.Dataset input target`.

## Boundary Rule

Training data may be shuffled and summarized by logs. Verification fixtures should be small,
reproducible, and explicit about ordering and tolerances. When an exported prediction becomes a
verification input, document the producing command and the checker that consumes it.

Public datasets and third-party artifacts must remain listed in `docs/THIRD_PARTY_NOTICES.md`.
