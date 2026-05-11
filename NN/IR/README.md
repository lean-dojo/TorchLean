# `NN/IR`

`NN.IR` is TorchLean's op tagged SSA/DAG intermediate representation. It is the small shared graph
language that model compilers, verification passes, exporters, widgets, and compiled-runtime
correctness proofs can all point at without each inventing a private graph format.

For public use, prefer the broad library import or the IR entrypoint. Internal code that only needs
one IR component should import the focused leaf directly.

```lean
import NN.Library
-- or, if you only want this subsystem:
import NN.Entrypoint.IR
-- or a focused internal leaf, for example:
import NN.IR.Graph
import NN.IR.Semantics
```

There is intentionally no extra top level `NN.IR` umbrella: `NN.Library` is the broad
surface, `NN.Entrypoint.IR` is the subsystem umbrella, and the individual `NN.IR.*` files remain the
precise internal dependency boundaries.

## What Belongs Here

- `Graph.lean`: graph syntax, node ids, op tags, arity conventions, and topological well formedness.
- `OpContracts.lean`: shared shape arithmetic for ops such as concat, matmul, pooling, conv, and
  axis-moving utilities.
- `Infer.lean`: the single source of truth for declared-output-shape validation.
- `Check.lean`: public validation wrappers and proposition-level `WellFormed` / `WellShaped` names.
- `Semantics.lean`: denotational evaluator into spec-layer tensor operations with explicit payloads,
  plus the scoped `IR` notation for graph denotation.
- `Pretty.lean`: readable text and GraphViz renderers for debugging.

## Relationship To `NN.GraphSpec`

`NN.GraphSpec` is a typed authoring DSL for model architectures, with a pure semantics and lowering
to TorchLean runtime programs. `NN.IR` sits lower in the stack: it is the op tagged graph target that
verification/export/runtime tooling can consume after a model has been compiled or traced.

In PyTorch terms, `NN.GraphSpec` is closer to a typed model construction DSL; `NN.IR` is closer to
an FX/TorchScript-style graph with explicit shapes and external parameter payloads.

## Release Invariants

- Node ids are array indices: `g.nodes[i].id = i`.
- Parents always point backward: every parent id is smaller than the child id.
- Parameter tensors are not embedded in `Graph`; `const`, `linear`, and `conv2d` use external
  payload stores keyed by node id.
- Shape checking is centralized through `Infer.inferNodeOutShape`; `Graph.checkShapes` delegates to
  that implementation to avoid duplicate op-contract logic.
