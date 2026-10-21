/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Examples.Quickstart.TensorBasics
public import NN.Examples.Quickstart.AutogradBasics
public import NN.Examples.Quickstart.SimpleMlpTrain
public import NN.Examples.Quickstart.Precision
public import NN.Examples.Quickstart.TypedTraining
public import NN.Examples.Quickstart.Proofs

/-!
# Quickstart

Curated first-tour examples for TorchLean.

This source umbrella teaches the primitives a new user needs before opening the larger model
examples:

- typed tensors and runtime arithmetic,
- one tensor-function and one model-state gradient,
- an end-to-end MLP training command,
- typed model execution at a caller-selected binary precision,
- an explicit typed SGD loop that preserves the selected precision, and
- small proofs over TorchLean definitions.

Data loading, CNNs, larger models, editor widgets, advanced autograd transforms, interoperability,
and verification live in their focused example directories. `Quickstart` is a learning path, not a
model catalog. The `torchlean` executable imports the three runnable quickstarts directly; precision
and proof modules are only included in this source umbrella.
-/

@[expose] public section
