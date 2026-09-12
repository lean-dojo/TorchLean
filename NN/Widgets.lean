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
public import NN.Widgets.Numerics.Float32
public import NN.Widgets.RL.GridWorld
public import NN.Widgets.RL.Boundary
public import NN.Widgets.RL.PPO
public import NN.Widgets.Runtime.Autograd
public import NN.Widgets.Runtime.Training
public import NN.Widgets.Verification.CROWN

/-!
# Widgets

Umbrella import for TorchLean's optional Infoview / widget tooling.

This module imports the widget implementation modules directly.

The import is for editor and inspection workflows:

- ordinary runtime/proof files should import the concrete library modules they need;
- example and inspection files can import this one module to get all widget commands;
- adding a widget here does not make it part of TorchLean's trusted semantics.

`NN.Widgets.Interop.PyTorchTranslator` is included here because it is exactly that kind of
editor-side assistant. It helps navigate PyTorch-to-TorchLean workflows; checked import still goes
through the explicit artifact bridges and verifier/proof layers.
-/

@[expose] public section
