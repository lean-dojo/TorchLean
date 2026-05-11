/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.GraphSpec.Primitives.Vision

/-!
# GraphSpec Primitive Packs

This umbrella re-exports the non-core GraphSpec primitive packs.

GraphSpec primitives are **operation adapters**, not models:

- a primitive wraps one reusable operation, such as convolution or pooling;
- a model composes primitives into an architecture, such as `cnn2` or `ResNet18.model`.

The small always-available primitives (`linear`, `relu`, `softmax`) live in
`NN.GraphSpec.Core` because they are part of the minimal sequential DSL examples and lowering
interface. Larger domain-specific packs live under `NN.GraphSpec.Primitives/*` so the core language does
not become a grab bag of every layer TorchLean supports.

Current extension pack:

- `NN.GraphSpec.Primitives.Vision`: CHW convolution, max-pool, flatten, BatchNorm, and global
  average pooling adapters.

Adding a new operation to GraphSpec means giving it both meanings:

- pure Spec semantics for proof/reference use;
- TorchLean program semantics for execution/training.
-/

@[expose] public section
