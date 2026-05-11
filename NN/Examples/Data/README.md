# Data Examples

TorchLean keeps the data boundary deliberately small. Python handles ecosystem formats and
downloads; Lean receives `.npy`, numeric CSV, or text, then checks shapes and builds typed datasets.

```text
external dataset -> Python converter/downloader -> .npy or numeric CSV -> NN.API.Data -> typed loader
```

Use Python for JPEG folders, `.pt`, `.npz`, `.mat`, and dataset downloads. Use Lean for typed
tensors, typed batching, training, verification, and reproducible evaluation.

## Local Tutorial Fixtures

Generate the small ignored fixtures used by the loader tutorials:

```bash
python3 NN/Examples/Data/generate_toy_data.py
```

This writes:

| File | Shape / role |
| --- | --- |
| `toy_regression.csv` | numeric rows `x1,x2,y` |
| `toy_regression_X.npy` | `(25, 2)` `float32` features |
| `toy_regression_y.npy` | `(25, 1)` `float32` targets |
| `toy_cifar10like_X.npy` | `(200, 3, 32, 32)` `float32` image-shaped features |
| `toy_cifar10like_y.npy` | `(200,)` integer labels stored as `float32` |

## Dataset Shapes

| Task | Input file | Target file | Loader |
| --- | --- | --- | --- |
| regression / operator learning | `X.npy : (N, ...)` | `Y.npy : (N, ...)` | `Data.SupervisedSource` |
| classification | `X.npy : (N, ...)` | `y.npy : (N,)` labels | `Data.LabeledSource` |
| tabular regression | numeric CSV with `x...,y...` | same CSV | `Data.TabularSupervisedSource` |
| text modeling | UTF-8 text | tokenizer/windowing builds targets | `NN.API.Text` |
| images | `X.npy : (N, C, H, W)` | `y.npy : (N,)` labels | `Data.LabeledSource` |

Classification labels are numeric values such as `0.0`, `1.0`, ..., and are converted to one-hot
targets on the Lean side.

## Downloaded Example Data

```bash
python3 scripts/datasets/download_example_data.py --auto-mpg --cifar10 --tiny-shakespeare
python3 scripts/datasets/download_example_data.py --household-power --household-power-windows 512
python3 scripts/datasets/download_example_data.py --tinystories-valid
```

Common outputs:

| Dataset | Output path |
| --- | --- |
| CIFAR-10 | `data/real/cifar10/cifar10_{train,test}_{X,y}.npy` |
| UCI Auto MPG | `data/real/auto_mpg/auto_mpg.csv` |
| UCI household power | `data/real/household_power/household_power_{X,Y}.npy` |
| Tiny Shakespeare | `data/real/text/tiny_shakespeare.txt` |
| TinyStories validation | `data/real/text/tinystories_valid.txt` |

The CIFAR-10 export defaults to a small real subset for interactive examples. Pass
`--cifar10-limit-train -1 --cifar10-limit-test -1` to export the full split.

## Run The Loader Examples

```bash
lake exe torchlean data_csv
lake exe torchlean data_npy
lake exe torchlean data_cifar10 --quick --epochs 1 --batch 4 --train-size 8 --n-total 20
```

Model examples use the same API:

```bash
lake exe -K cuda=true torchlean mlp --cuda --epochs 10
lake exe -K cuda=true torchlean cnn --cuda --n-total 200 --epochs 2
lake exe -K cuda=true torchlean lstm_regression --cuda --steps 200 --windows 96
lake exe -K cuda=true torchlean gpt2 --cuda --tiny-shakespeare --steps 100
```

## Convert External Data

Use `.npy` as the stable tensor boundary:

```bash
python3 scripts/datasets/torchlean_data_convert.py tensor \
  --input features.pt --key x --output data/real/mytask/X.npy --manifest

python3 scripts/datasets/torchlean_data_convert.py pair \
  --x-input dataset.npz --x-key images --x-output data/real/mytask/X.npy \
  --y-input dataset.npz --y-key labels --y-output data/real/mytask/y.npy \
  --manifest

python3 scripts/datasets/torchlean_data_convert.py image-folder \
  --input imagenette2/train --height 160 --width 160 --labels-from-dirs \
  --x-output data/real/imagenette/train_X.npy \
  --y-output data/real/imagenette/train_y.npy \
  --manifest
```

## Lean API

Import the public data surface with:

```lean
import NN.API.Data
```

Classification source:

```lean
open NN.API

def src : Data.LabeledSource :=
  Data.LabeledSource.ofPaths
    .npy
    "data/real/imagenette/train_X.npy"
    "data/real/imagenette/train_y.npy"
    1000
    [3, 160, 160]
    10
```

Supervised regression source:

```lean
open NN.API

def src : Data.SupervisedSource :=
  Data.SupervisedSource.ofPaths
    .npy
    "data/real/fno/burgers_X.npy"
    "data/real/fno/burgers_Y.npy"
    1000
    [64]
    [64]
```

Wrap loaded datasets with `Data.batchLoader` for typed minibatches:

```lean
let loader := Data.batchLoader ds 8 (shuffle := true) (seed := 0) (dropLast := true)
let (_loader', batches) ← Common.orThrow "trainer" <| Data.BatchLoader.epoch "trainer" loader
```

The batch axis is reflected in the type, so image minibatches have shapes such as:

```text
X : Tensor α (Shape.dim batch (Shape.Image C H W))
Y : Tensor α (Shape.dim batch (Shape.Vec classes))
```

## Provenance

CIFAR-10, UCI Auto MPG, UCI household power, Tiny Shakespeare, and TinyStories keep their upstream
licenses and citations. The generated tutorial fixtures are local examples and are ignored by git.
