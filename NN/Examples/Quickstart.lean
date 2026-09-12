/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Examples.Quickstart.TensorBasics
public import NN.Examples.Quickstart.AutogradBasics
public import NN.Examples.Quickstart.SimpleMlpTrain
public import NN.Examples.Quickstart.Proofs
public import NN.Examples.Quickstart.Widgets

/-!
# Quickstart

Curated first-tour examples for TorchLean.

This source umbrella teaches the primitives a new user needs before opening the larger model
examples:

- typed tensors and runtime arithmetic,
- one tensor-function and one model-state gradient,
- an end-to-end MLP training command,
- small proofs over TorchLean definitions, and
- optional editor widgets.

Data loading, CNNs, larger models, advanced autograd transforms, interoperability, and verification
live in their focused example directories. `Quickstart` is a learning path, not a model catalog.
The `torchlean` executable imports the three runnable quickstarts directly; proof and widget
modules are only included in this source umbrella.
-/

@[expose] public section
