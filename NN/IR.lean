/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.IR.Check
public import NN.IR.Graph
public import NN.IR.Infer
public import NN.IR.Operator
public import NN.IR.OpContracts
public import NN.IR.Pretty
public import NN.IR.Semantics
public import NN.IR.ShapeSoundness

/-!
# Intermediate Representation

Curated umbrella import for TorchLean's op-tagged intermediate representation.

Use this when you want the whole IR subsystem: graph syntax, operation contracts, shape inference,
validation wrappers, denotational semantics, the shape-soundness theorems relating the two, and
pretty-printers. The individual `NN.IR.*` files
remain focused implementation modules for internal code that needs a smaller dependency footprint.
-/

@[expose] public section
