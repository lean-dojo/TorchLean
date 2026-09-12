# Common Model Example Helpers

This folder contains shared plumbing for runnable model examples. It should stay predictable:
repository data conventions, model-specific flag defaults, and common
"prepare this dataset" messages. Model definitions, runtime kernels, and proof logic belong
elsewhere.

## Files

- `RealData.lean`: shared real-data helpers for CIFAR-10, ImageNet-style tensors, local text
  corpora, cropping, typed minibatches, and missing-data hints.
- `Train.lean`: dataset-specific command adapters delegating to
  `TorchLean.CLI.Training.Command`. Generic runtime parsing lives in `TorchLean.CLI.Trainer`;
  curves and metric-history writers live in `TorchLean.Training`.

## Boundary

Model examples should use this folder when they need the same local data convention or CLI shape.
They should not hide a new mathematical model here. If a helper becomes a public API, move it under
`NN/API` or the relevant `NN/Runtime` module and keep this folder as thin command glue.

The data boundary remains:

```text
download/convert with Python
  -> .npy, CSV, or text under data/
  -> typed TorchLean loader
  -> public Trainer command
```

This keeps examples consistent without letting dataset scripts, runtime details, and model
semantics blur together.

## What Belongs In Common Helpers

Good common helpers do one of three things:

- make a repeated CLI convention consistent across model commands;
- check a local file path and print the command that prepares the missing data;
- adapt an already-documented data source into the typed `Trainer` path.

They should not introduce a new loss, architecture, verifier, backend, or theorem statement. Those
belong in the subsystem that owns the semantics. Keeping this folder small makes it easier to see
whether an example is using the ordinary public trainer or crossing into a special runtime path.

## Adding A New Model Command

When adding a model command, use this folder only for shared boilerplate. The model file itself
should still say:

- what data it expects and how to prepare it;
- which shapes enter the model;
- which `Trainer` objective and optimizer are used;
- what artifacts are written, such as `TrainLog` JSON, prediction CSV, images, or saved params;
- whether CUDA, ATen/libtorch, or another backend is part of the runtime boundary.

If the command produces an artifact that a verifier later consumes, document that handoff in the
verification README too. The model command is the producer; the checker owns the verified claim.
