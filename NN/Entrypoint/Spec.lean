/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Autograd
public import NN.Spec.Core
public import NN.Spec.Dynamics
public import NN.Spec.Generative
public import NN.Spec.Layers
public import NN.Spec.Models
public import NN.Spec.Module
public import NN.Spec.RL
public import NN.Entrypoint.Tensor


/-!
# Spec entrypoint

This is the one-stop import for TorchLean’s *spec* layer: shapes, tensors, layers, modules, and
model constructors.

If you're writing specs or proofs, this is usually the right place to start. If you're trying to
*run* models (autograd, training loops, import/export), prefer the runtime entrypoints instead.

Structure:
- `NN.Tensor.API` for the core tensor/shape layer and ergonomic constructors,
- `NN.Spec.Layers.*` for layer-level denotational semantics,
- `NN.Spec.Module.*` for PyTorch-style module wrappers over those specs,
- `NN.Spec.Models.*` for reusable model constructors,
- `NN.Spec.Autograd.*` / `NN.Spec.Dynamics.*` for auxiliary math-first interfaces.

This module is the chapter boundary for the spec surface. It imports the focused `NN.Spec.*`
subsystems directly and re-exports the ergonomic tensor entrypoint beside them.
-/

@[expose] public section
