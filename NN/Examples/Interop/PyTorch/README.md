# PyTorch state-dict round trips

`Roundtrip.lean` exports Python modules or loads JSON parameters into Lean tensors and runs a
forward pass. The reusable model-family adapters live under `NN/Runtime/PyTorch/{Import,Export}`.

From the repository root:

```bash
lake exe torchlean pytorch_roundtrip --model mlp --action import
lake exe torchlean pytorch_roundtrip --model cnn --action import
lake exe torchlean pytorch_roundtrip --model transformer --action import
lake exe torchlean pytorch_roundtrip --model mlp --action export
```

Import uses the checked-in weights. MLP evaluates tensor operations; CNN and Transformer load an
executable CPU `nn` module. Export writes Python source beside the corresponding JSON file, with
an additional weight-loading script when that JSON exists. It does not run Python or train.

| Folder | Model and producer |
| --- | --- |
| `MLP/` | Two linear layers and ReLU; `train_mlp.py` fits one input/target pair. |
| `CNN/` | Two convolution/ReLU/pooling blocks and a linear head; `train_cnn.py` fits one image. |
| `Transformer/` | One seeded encoder block; `train_transformer.py` exports initialization without training. |

Each folder documents the tensor shapes and parameter orientation. Producers overwrite their local
JSON fixture when run. The importers accept bare state dictionaries or a `params` wrapper, check
required tensor shapes, and require `float32` when dtype metadata is present. They allow extra keys
and do not infer the architecture from `meta.format`.

For model-agnostic graph capture, use `NN.Runtime.PyTorch.Export.TorchExport` and
`NN.Runtime.PyTorch.Import.TorchExport`. The maintained regression suite exercises capture,
unsupported operators, malformed artifacts, and numerical parity:

```bash
lake exe pytorch_export_check
```

That suite lives in `NN/Tests/Interop/PyTorch.lean`. It compares every captured output against
PyTorch at absolute tolerance `2e-5`; the round-trip commands above print a Lean prediction without
asserting cross-framework parity. Graph parsing establishes the importer's shape predicate;
semantic preservation and native execution require their corresponding library contracts.
