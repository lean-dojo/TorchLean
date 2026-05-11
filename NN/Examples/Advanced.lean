/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Examples.Advanced.Floats.ArbIEEEExecCompare
public import NN.Examples.Advanced.Floats.Float32Modes
public import NN.Examples.Advanced.GraphSpec.Tutorial
public import NN.Examples.Advanced.IRAxisOps
public import NN.Examples.Advanced.OneSemanticUniverse
public import NN.Examples.Advanced.Tensors.Basic
public import NN.Examples.Advanced.TorchIRPyTorch
public import NN.Examples.Advanced.Widgets

/-!
# `NN.Examples.Advanced`

Curated umbrella for advanced TorchLean examples.

These files are intentionally not beginner quickstarts. They demonstrate cross-cutting boundaries:

- float/runtime semantics and Arb-backed interval comparison;
- GraphSpec lowering into the training API;
- IR axis semantics and PyTorch export;
- the “one semantic universe” contract connecting execution, CROWN/IBP, and widgets;
- tensor construction/indexing/bridging basics and editor-only widget panels.

Rule of thumb for this folder: examples may contain small concrete graphs, parameters, and display
data, but reusable algorithms and data structures belong in `NN.*` library modules. Import this
umbrella when you want the whole advanced examples surface.
-/

@[expose] public section
