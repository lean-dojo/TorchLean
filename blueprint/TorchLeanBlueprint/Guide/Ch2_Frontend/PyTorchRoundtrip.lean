import VersoManual

open Verso.Genre Manual

#doc (Manual) "PyTorch Round-Trip" =>
%%%
tag := "pytorch-roundtrip"
%%%

Many useful TorchLean workflows will still involve Python. A model may be trained in PyTorch
because the dataset, optimizer, or engineering environment belongs there. The question for
TorchLean is what happens after that training run. Can the result come back into Lean as a checked
artifact, with names, shapes, and parameter order made explicit?

The round trip is deliberately narrow. TorchLean does not import arbitrary `nn.Module` objects. It
supports known model families with known layouts. That restriction is a feature: a small bridge can
be audited, tested, and connected to later graph and verification work.

For runnable examples around this boundary, see:

- [NN.Examples.Interop.PyTorch.Roundtrip API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Interop/PyTorch/Roundtrip.lean)
- [NN.Examples.Interop.PyTorch.TorchExportSmoke API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Interop/PyTorch/TorchExportSmoke.lean)
- [Torch IR and PyTorch example](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Advanced/TorchIRPyTorch.lean)

The main declarations for this boundary are
[PyTorch export](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/PyTorch/Export.lean),
[PyTorch import](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/PyTorch/Import.lean),
[interop examples](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Interop/PyTorch.lean), and the spec pages for
[modules](https://github.com/lean-dojo/TorchLean/blob/main/NN/Spec/Module.lean) and [models](https://github.com/lean-dojo/TorchLean/blob/main/NN/Spec/Models.lean).

# The Contract

The round trip has five steps:

1. Lean defines the expected model family.
2. Lean exports a matching PyTorch scaffold.
3. Python trains or modifies the weights.
4. Python writes a named tensor payload.
5. Lean imports the payload and checks the family tag, parameter names, order, shapes, and scalar
   format.

That design keeps the boundary between Lean and PyTorch small enough that, when something goes
wrong, the failure is usually local and comprehensible: a mismatched name, a mismatched shape, or an
unexpected model family.

If any of those checks fail, the artifact stops at the boundary. It does not become a TorchLean
model by accident.

# What Lean Checks On Import

The importer is intentionally strict. It checks the payload before constructing the typed parameter
bundle used by the rest of TorchLean.

The checks include:

- the model family tag,
- the parameter names Lean expects for that family,
- the parameter order used by the typed bundle,
- the shape attached to each tensor,
- the number of values implied by that shape,
- the scalar payload format.

For example, if Lean expects:

```
linear1.weight : shape![8, 2]
```

but the JSON contains:

```
linear1.weight : shape![2, 8]
```

the import fails before the value becomes a model parameter. That is the behavior we want: a
round-trip should either reconstruct the same typed layout, or stop at the boundary with a concrete
error.

# What Gets Serialized

The round-trip does not serialize an arbitrary Python object graph. Instead it serializes a small
amount of model family metadata plus a named tensor payload. The exact schema depends on the family,
but the shape of the contract is always the same:

- choose a known architecture family,
- agree on parameter order and tensor layout,
- serialize tensors by name into JSON,
- re-check that layout on the Lean side before constructing typed parameters.

Lean then reconstructs the typed parameter bundle that the rest of TorchLean expects. The JSON file
is transport, not semantics.

# Model Families

The repository uses three family demos:

- MLP: the smallest parameter and shape contract.
- CNN: convolution weights and image layouts.
- Transformer encoder: attention projections and many named parameters.

## MLP

Open:

- Lean entrypoint: [PyTorch roundtrip API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Interop/PyTorch/Roundtrip.lean)
- Python: [MLP training script](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Interop/PyTorch/MLP/train_mlp.py)
- Output JSON: `NN/Examples/Interop/PyTorch/MLP/mlp.json`

The MLP path is the smallest place to read the round-trip as a contract rather than as infrastructure.
The architecture is small enough to inspect exported names, compare them against PyTorch's
`state_dict`, and see exactly what Lean checks on re-import.

This is also the best place to learn the failure modes: if the exported JSON is wrong, Lean usually
rejects it because the family name, tensor shape, or parameter ordering does not match the expected
typed layout.

Example command sequence:

```
lake env lean --run NN/Examples/Interop/PyTorch/Roundtrip.lean -- --model mlp --action export
python3 NN/Examples/Interop/PyTorch/MLP/train_mlp.py
lake env lean --run NN/Examples/Interop/PyTorch/Roundtrip.lean -- --model mlp --action import
```

## CNN

Open:

- Lean entrypoint: [PyTorch roundtrip API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Interop/PyTorch/Roundtrip.lean)
- Python: [CNN training script](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Interop/PyTorch/CNN/train_cnn.py)
- Output JSON: `NN/Examples/Interop/PyTorch/CNN/cnn.json`

The CNN path is a useful middle ground. It exercises nontrivial tensor shapes and weight layouts,
but it is still small enough to debug by hand when the importer reports a mismatch.

Example command sequence:

```
lake env lean --run NN/Examples/Interop/PyTorch/Roundtrip.lean -- --model cnn --action export
python3 NN/Examples/Interop/PyTorch/CNN/train_cnn.py
lake env lean --run NN/Examples/Interop/PyTorch/Roundtrip.lean -- --model cnn --action import
```

## Transformer (Encoder)

Open:

- Lean entrypoint: [PyTorch roundtrip API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Interop/PyTorch/Roundtrip.lean)
- Python: [Transformer training script](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Interop/PyTorch/Transformer/train_transformer.py)
- Output JSON: `NN/Examples/Interop/PyTorch/Transformer/transformer_encoder.json`

The transformer example makes the boundary especially clear. Once attention layers and multiple
projections appear, the reason for a family-based round-trip is concrete: Lean can check a known
layout instead of claiming to import every PyTorch module.

The same pattern scales to larger model families: Lean checks the declared shapes and parameter
layout, Python runs the training loop, and the JSON payload transports only named parameters back
across the boundary.

Example command sequence:

```
lake env lean --run NN/Examples/Interop/PyTorch/Roundtrip.lean -- --model transformer --action export
python3 NN/Examples/Interop/PyTorch/Transformer/train_transformer.py
lake env lean --run NN/Examples/Interop/PyTorch/Roundtrip.lean -- --model transformer --action import
```

# TorchLean, IR, and Generated PyTorch Code

A second interop path complements the JSON round-trip workflow:

- file: [Torch IR and PyTorch example](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Advanced/TorchIRPyTorch.lean)

This example compiles a TorchLean model to the shared IR and then emits runnable PyTorch code for a
curated set of architectures:

- `linear`, `mlp`, `sum`, `autoencoder`
- `cnn`, `conv-mlp`
- `mha`, `mha-mask`
- `transformer`

Typical usage:

```
lake exe torchlean torch_ir_pytorch --arch mlp > exported_model.py
python3 exported_model.py
```

This is useful for two reasons:

- it shows the compiled IR as an interchange boundary, not only as an internal verifier artifact;
- it gives PyTorch users a concrete translation example for architectures richer than a single
  linear layer.

# A Small Python View

The Python half is intentionally ordinary. The goal is not to replace the PyTorch workflow; the
goal is to make the boundary between PyTorch and Lean explicit enough that it can be audited.

```
import torch

model = build_exported_model()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(steps):
    pred = model(x_train)
    loss = torch.nn.functional.mse_loss(pred, y_train)
    opt.zero_grad()
    loss.backward()
    opt.step()

save_torchlean_json(model, "mlp.json")
```

That is the basic shape of the round-trip: Lean defines the expected structure, Python performs the
training, and Lean checks the returned payload against the same structure.

# Where This Fits In The Guide

Use the round-trip when the training workflow belongs in Python but the returned artifact should
enter the TorchLean world with typed parameters and auditable metadata. Use the runtime chapters
when the training loop itself should run in Lean. Use the graph and verification chapters when the
next step is to inspect the model as an IR object or connect it to a theorem.

# Guarantees And Limits

When the round trip succeeds, the result is a controlled bridge between Lean and Python:

- Lean can emit a model skeleton and helper files.
- Python can train or export weights using a matching layout.
- Lean can read the exported payload back in and continue on the proof or verification side.

What it does not claim:

- this is not a proof that arbitrary PyTorch training is semantically identical to Lean execution,
- this is not a universal importer for the full PyTorch ecosystem,
- this does not by itself settle the floating-point semantics.

Binary32 claims still require the relevant TorchLean float backend and the theorems in the
floating-point chapters.

For a PyTorch-side-by-side comparison, read *TorchLean vs PyTorch*.
Verification after import: *Graphs and IR*, then *Verification*.

The round trip should be read as an artifact discipline, not as a universal conversion tool. Python
can remain the right place to train. Lean becomes the place where the returned object is named,
shaped, inspected, and prepared for graph-level or verification work.

# References

- TorchLean paper (George et al., 2026): project overview, shared IR architecture, and verification-driven
  architecture. https://arxiv.org/abs/2602.22631
- PyTorch serialization notes: the official reference for `state_dict`-style save/load workflows.
  https://pytorch.org/docs/stable/notes/serialization.html
- PyTorch `nn.Module` documentation: helpful background for parameter naming and module-family
  conventions. https://pytorch.org/docs/stable/generated/torch.nn.Module.html
