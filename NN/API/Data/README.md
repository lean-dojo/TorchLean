# `NN.API.Data`

`NN.API.Data` is TorchLean's public data API.

The data boundary is intentionally small:

- `.npy` for numeric tensors;
- numeric CSV for small tabular data;
- UTF-8 text for language-model examples.

For other formats, use the converter:

```bash
python3 scripts/datasets/torchlean_data_convert.py --help
```

## Which Source Should I Use?

| Data shape | Use |
| --- | --- |
| one tensor file | `Data.TensorSource` |
| supervised `X.npy`, `Y.npy` | `Data.SupervisedSource` |
| image/classification labels | `Data.LabeledSource` |
| small numeric CSV | `Data.TabularSupervisedSource` |
| repeated training batches | `Data.batchLoader` |
| simple preprocessing | `Data.Transforms` |

The main Lean entry points are:

- `Data.TensorSource`: one tensor file plus expected dimensions.
- `Data.SupervisedSource`: two batched tensors, `X : (N, xDims...)` and `Y : (N, yDims...)`.
- `Data.LabeledSource`: batched inputs plus label vector, one-hot encoded when loaded.
- `Data.TabularSupervisedSource`: one CSV where each row contains `x..., y...`.
- `Data.batchLoader`: typed, deterministic minibatching.
- `Data.Transforms`: map transforms over samples and datasets.

Examples and conversion recipes live in:

```text
NN/Examples/Data/README.md
```
