# PyTorch state-dict round trips

`Roundtrip.lean` reads the PyTorch state dict of an MLP, a CNN or a Transformer encoder block into
the `TensorPack Float` of the same Lean model and prints the model's output on a fixed input. The
script beside each fixture trains or initializes the PyTorch module and writes its `state_dict()`
with `python/torchlean/state.py`. The importer, `TorchLean.PyTorch.state` in
`TorchLean/NN/PyTorch/Import/State.lean`, reads each tensor under the name PyTorch gives it.

From the repository root:

```bash
scripts/lake.sh exe torchlean pytorch_roundtrip --model mlp
scripts/lake.sh exe torchlean pytorch_roundtrip --model cnn
scripts/lake.sh exe torchlean pytorch_roundtrip --model transformer
```

The driver uses the checked-in state dicts and runs the ordinary executable `nn` model on the CPU.
It does not run Python or train.

| Folder | Model and producer |
| --- | --- |
| `MLP/` | Two linear layers and ReLU. `train_mlp.py` fits one input/target pair. |
| `CNN/` | Two convolution/ReLU/pooling blocks and a linear head. `train_cnn.py` fits one image. |
| `Transformer/` | One seeded encoder block. `export_transformer.py` writes its initialization. |

Each folder lists its PyTorch names and shapes. The producers overwrite their local JSON fixture
when run. A fixture is a `torchlean.state` document holding, for each name, the dtype, the shape
and the binary word of every element. `Roundtrip.lean` accepts a document whose names are the names
the Lean model lists, each entry with the dtype `float64` of `Float` and the shape of its state
tensor. The importer's messages for missing keys, unexpected keys and shape mismatches follow
PyTorch's `load_state_dict`.

For model-agnostic graph capture, run
`PYTHONPATH=python python3 -m torchlean.export MODULE CTOR JSON --example-shape 1,4` from the
repository root. `TorchLean.IR.load` reads the document it writes: `TorchLean.IR.Artifact.ofJson`
decodes its `torchlean.ir` artifact and `TorchLean.IR.Artifact.load` checks it. The
[graph-import guide](https://lean-dojo.github.io/TorchLean/examples/verify/) explains capture and
reference comparisons. The round-trip commands above print a Lean prediction and do not compare it
with PyTorch. Loading checks every step's ports and element types
(`TorchLean.IR.Program.check`). Semantic preservation and native execution require their
corresponding library contracts.
