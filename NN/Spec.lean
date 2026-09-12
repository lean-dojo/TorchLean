/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Autograd
public import NN.Spec.Core
public import NN.Spec.Core.FloatInstances
public import NN.Spec.Core.FloatInstances.NF
public import NN.Spec.Dynamics
public import NN.Spec.Generative
public import NN.Spec.Layers
public import NN.Spec.Models
public import NN.Spec.Module
public import NN.Spec.Quantization
public import NN.Spec.RL


/-!
# Mathematical Specifications

Import this file when you want TorchLean’s *spec* layer: shapes, tensors, layers, modules, and model
constructors.

If you're writing specs or proofs, this is usually the right place to start. If you're trying to
*run* models (autograd, training loops, import/export), use `NN.API` for the public application
interface or `NN.Runtime` for the complete executable subsystem.

Structure:
- `NN.Spec.Core` for the core tensor/shape mathematics,
- `NN.Spec.Layers.*` for layer-level denotational semantics,
- `NN.Spec.Module.*` for PyTorch-style module wrappers over those specs,
- `NN.Spec.Models.*` for reusable model constructors,
- `NN.Spec.Autograd.*` / `NN.Spec.Dynamics.*` for auxiliary math-first interfaces.

The user-facing tensor constructors and notation live in `NN.Tensor` and are intentionally not
re-exported here.
-/

@[expose] public section
