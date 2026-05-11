/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Examples.Quickstart.TensorBasics
public import NN.Examples.Quickstart.Widgets
public import NN.Examples.Quickstart.AutogradBasics
public import NN.Examples.Quickstart.SimpleMlpTrain
public import NN.Examples.Quickstart.Proofs

/-!
# Quickstart

Curated first-tour examples for TorchLean.

This umbrella is intentionally smaller than `NN.Examples.Zoo`. It teaches the primitives a new user
needs before opening the model zoo:

- typed tensors and runtime scalar choices,
- editor widgets for inspecting tensors, floats, graphs, and logs,
- autograd helpers for gradients, Jacobians, Hessian-vector products, and detach,
- one compact end-to-end MLP training loop, and
- the proof/compile-time side of TorchLean's shape-indexed API.

Larger CNN, ResNet, data-loader, PyTorch interop, RL, and verification examples remain in their
specialized folders. The point of `Quickstart` is a clean on-ramp, not another model zoo.
-/

@[expose] public section
