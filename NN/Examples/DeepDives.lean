/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Examples.DeepDives.AutogradTransforms
public import NN.Examples.DeepDives.Floats.ArbIEEEExecCompare
public import NN.Examples.DeepDives.Floats.EffectiveRounding
public import NN.Examples.DeepDives.Floats.Float32Semantics
public import NN.Examples.DeepDives.Floats.GraphNumericalCertificate
public import NN.Examples.DeepDives.GraphSpec.Tutorial
public import NN.Examples.DeepDives.IRAxisOps
public import NN.Examples.DeepDives.OneSemanticUniverse
public import NN.Examples.DeepDives.TensorOperations
public import NN.Examples.DeepDives.TorchIRPyTorch
public import NN.Examples.DeepDives.Widgets

/-!
# `NN.Examples.DeepDives`

Curated umbrella for deep-dive TorchLean examples.

These files are not beginner introductory examples. They cover cross-cutting boundaries:

- higher-order, directional, and stopped-gradient autograd transforms;
- float/runtime semantics, numerical certificates, and Arb-backed interval comparison;
- GraphSpec lowering into the training API;
- IR axis semantics and PyTorch export;
- the “one semantic universe” contract connecting execution, CROWN/IBP, and widgets;
- tensor operations, indexing, reshaping, `einsum`, and editor-only widget panels.

Rule of thumb for this folder: examples may contain small concrete graphs, parameters, and display
data, but reusable algorithms and data structures belong in `NN.*` library modules. Import this
umbrella when you want the whole deep-dive example surface.
-/

@[expose] public section
