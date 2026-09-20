/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.GraphSpec.Chain.Lowering
public import NN.GraphSpec.Chain.ToDAG

/-!
# Sequential GraphSpec

This is the canonical import for GraphSpec's sequential model language. It exposes:

- the extensible `Primitive` interface and shape-indexed `Chain` syntax;
- standard linear, ReLU, and softmax chain constructors;
- direct pure tensor semantics through `Interp.spec`;
- execution-polymorphic lowering through `Chain.toProgram`;
- structural conversion to the canonical GraphSpec DAG representation.

The implementation is organized by ownership in `Chain.Syntax`, `Chain.Primitives`,
`Chain.Semantics`, `Chain.Lowering`, and `Chain.ToDAG`. Existing users should continue to import
`NN.GraphSpec.Core`.

For skip connections, shared intermediates, residual additions, and other multi-input nodes, use
`NN.GraphSpec.DAG.Core` directly.
-/
