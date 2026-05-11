/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Widgets.Core.Docs
public import NN.Widgets.Core.Tensor
public import NN.Widgets.Core.UI
public import NN.Widgets.IR.ExecutionTrace
public import NN.Widgets.IR.Graph
public import NN.Widgets.IR.Rewrite
public import NN.Widgets.IR.ShapeInference
public import NN.Widgets.Interop.PyTorchTranslator
public import NN.Widgets.Models.Sequence.Gpt2
public import NN.Widgets.Numerics.Float32
public import NN.Widgets.RL.GridWorld
public import NN.Widgets.RL.Boundary
public import NN.Widgets.RL.PPO
public import NN.Widgets.Runtime.Autograd
public import NN.Widgets.Runtime.Context
public import NN.Widgets.Runtime.Training
public import NN.Widgets.Verification.CROWN

/-!
# Widgets entrypoint

Umbrella import for TorchLean's optional Infoview / widget tooling.

This entrypoint imports the widget implementation modules directly. We avoid a second top-level
`NN.Widgets` alias so the root namespace stays focused on the library umbrella and subsystem
entrypoints.

The entrypoint is intentionally editor-facing:

- ordinary runtime/proof files should import the concrete library modules they need;
- tutorial and inspection files can import this one module to get all widget commands;
- adding a widget here does not make it part of TorchLean's trusted semantics.

`NN.Widgets.Interop.PyTorchTranslator` is included here because it is exactly that kind of
editor-side assistant: useful for navigating PyTorch-to-TorchLean workflows, but not a proof that
arbitrary Python has been verified.
-/

@[expose] public section
