# `NN/IR`

`NN.IR` is TorchLean's shape-annotated, op-tagged topological DAG interchange representation. It is
the small shared graph language used by model lowering, verification, export, widgets, and checked
forward execution. It is distinct from the dependent `GraphData` representation recorded by the
differentiable typed-graph session.

For public use, prefer the broad library import or the IR entrypoint. Internal code that only needs
one IR component should import the focused leaf directly.

```lean
import NN
-- or, if you only want this subsystem:
import NN.IR
-- or a focused internal leaf, for example:
import NN.IR.Graph
import NN.IR.Semantics
```

`NN/IR.lean` is the subsystem umbrella imported by `NN.IR`. Internal modules should import the
smallest `NN.IR.*` dependency they need.

## Directory Layout

- `Operator.lean`: `OpKind`, its static attributes, and `OpTag` constructor identities with parent
  counts and diagnostic names. This module has no graph, runtime, or artifact-format dependency.
- `Graph.lean`: nodes, dependency edges, declared shapes, and topological well formedness.
- `OpContracts.lean`: shared shape arithmetic for ops such as concat, matmul, pooling, conv, and
  axis-moving utilities. Concat and matmul use list-indexed arbitrary-rank contracts.
- `Infer.lean`: canonical declared-output-shape validation. `Graph.inferShapesFrom` walks the node
  array once, structurally recursive on the remaining node count, and `Graph.checkShapes` wraps it.
- `Check.lean`: public validation wrappers and proposition-level `WellFormed` / `WellShaped` names.
- `Semantics.lean`: denotational evaluator into spec-layer tensor operations with explicit payloads,
  including arbitrary-leading linear and matrix multiplication, checked arbitrary-axis concat, and
  the scoped `IR` notation for graph denotation. `Graph.evalNodeRaw` is the operator dispatch and
  `Graph.evalNode` adds the declared-shape normalization.
- `Payload.lean`: the external payload stores (constants, weights, input boxes) keyed by node id.
- `HardMask.lean`: conversions between typed Boolean tensors and the row-major masks stored in
  `OpKind`, shared by graph builders, evaluators, and verifier passes.
- `ShapeSoundness.lean`: the theorems relating `Infer` and `Semantics` (see below).
- `Pretty.lean`: readable text and GraphViz renderers for debugging.

## One Shape Rule Per Operation

`Infer.nodeOutShape` and `Graph.evalNodeRaw` are both matches over `OpKind`, but they do not
contain two copies of the shape arithmetic. Every operation whose output shape is not simply a
parent shape or a shape written in the operation goes through a single rule in `OpContracts`, and
both passes call it on the same inputs:

| Operation | Shared rule |
| --- | --- |
| `matmul` | `OpContracts.matmulDims` (operand decomposition); inference reads `MatmulDims.outShape`, the evaluator recovers the typed operands from `leftShape` / `rightShape` |
| `maxPool`, `avgPool` | `OpContracts.planPool` via `inferWindowOutShape`; the plan's `outShape` is the type of the evaluator's result |
| `reduceSum`, `reduceMean` | `OpContracts.checkReductionAxis` (a nonempty axis, as the typed reductions require) and `Tensor.shapeAfterSum` |
| `flatten` | `ShapeUtil.flattenOutShape` |
| `concat` | `OpContracts.inferConcatOutShape`, recomputed by the evaluator on the parent values |
| `conv`, `batchNormEval`, `broadcastTo`, `layernorm`, `transpose` | `inferConvConfigOutShape`, `inferBatchNormEvalOutShape`, the decidable `Spec.Shape.CanBroadcastTo`, `layerNormMatrixDims`, `transposePerm` |

The permute rule is `Spec.Shape.permute?`; the evaluator realizes the permutation by adjacent
swaps (`Graph.swapDepthsForPerm`) and checks the realized shape against the declared one.

## Shape Soundness

`NN/IR/ShapeSoundness.lean` proves that the two passes agree, organized as one lemma per operator
family (`Graph.evalNodeRaw_shape_*`) plus a lockstep induction over the node array:

- `Graph.evalNodeRaw_shape_of_infer`: if inference assigns the declared shape to a node and the
  node's parents carry the shapes inference saw (`Graph.ParentShapesOf`), then the raw evaluator
  value already has the declared shape.
- `Graph.denoteAllRaw_eq_denoteAll`: on a graph accepted by `checkShapes`, evaluating without the
  per-node declared-shape normalization (`Graph.denoteAllRaw`) gives the same result as
  `denoteAll`. The normalization is defense in depth, not a second shape rule.
- `Graph.checkShapes_sound` / `Graph.denoteAll_shape`: every value produced by `denoteAll` has the
  declared shape of its node. The `checkShapes` hypothesis is not needed for this conclusion
  because `evalNode` normalizes; the previous theorem is where it does work.

For `permute`, `transpose`, and `conv` the evaluator compares the shape it realizes with the
declared shape, so soundness holds through that comparison; proving that the adjacent-swap
lowering realizes `Shape.permute?`, and that the convolution contract's list arithmetic equals
`Spec.convOutSpatialDilated`, are separate lemmas not attempted here.

Note on partiality: `denoteAll` is not total on well-shaped graphs. `.log` rejects nonpositive
input data, and payload-backed nodes reject inconsistent payloads. See the header of
`Semantics.lean`.

## Relationship To `NN.GraphSpec`

`NN.GraphSpec` is a typed authoring DSL for model architectures, with a pure semantics and lowering
to TorchLean runtime programs. `NN.IR` is the lower-level op tagged graph target that
verification/export/runtime tooling can consume after a model has been lowered or traced.

In PyTorch terms, `NN.GraphSpec` is closer to a typed model construction DSL; `NN.IR` is closer to
an FX/TorchScript-style graph with explicit shapes and external parameter payloads.

## How A Model Reaches IR

There are several routes into the same graph language:

- TorchLean model code can lower supported fragments into IR for execution, inspection, or
  verification.
- GraphSpec models can be lowered when the architecture needs explicit sharing and
  named parameter layouts.
- PyTorch `torch.export` and ONNX adapters can write `torchlean.ir.v1` JSON, which Lean then parses
  and validates.
- Verification examples can build small graphs directly when the graph itself is the artifact under
  study.

Those routes are intentionally different producers with one consumer contract. Once a graph reaches
`NN.IR.Graph`, downstream code should be able to ask the same questions: are node ids topological,
are shapes inferred by the shared op contracts, which payloads are required, and what denotation does
the graph have in the spec layer?

## Role And Scope

The IR gives the runtime, checkers, exporters, and future compiler passes one graph object to share. Write
ordinary models through `TorchLean.nn`, `Trainer`, or `GraphSpec`, then lower them. Construct `Node`
arrays directly only when testing an IR consumer.

Each runtime backend retains its own proof status. Proofs, tests, and trust-boundary statements say
how a particular runtime, lowering fragment, or certificate checker relates to the shared graph.

## Current Consumers

| Consumer | How it uses IR |
| --- | --- |
| Checked IR execution | `IRExec` validates and lowers supported IR nodes to a forward-only, shape-indexed `ForwardGraph`. The differentiable typed-graph session records `GraphData` directly instead. |
| Verification | Runs IBP/CROWN-style passes, margin checks, and certificate replay over node ids and payloads. |
| TorchLean lowering fragments | Prove that supported source fragments lower to IR with the same denotation. |
| PyTorch/ONNX/export paths | Use a small graph format to make parameter order and tensor shapes explicit at the boundary. |
| Widgets and graph pages | Render graphs, inferred shapes, execution traces, and dependency structure for debugging. |

## Proof And Runtime Status

The IR is the object shared by proofs and runtime code, but proof coverage is still named
fragment-by-fragment. Current theorem work covers supported evaluator bridges, graph well-formedness
conditions, selected checked IR-to-`ForwardGraph` lowering fragments, and verification-oriented
bound propagation. Separately, `GraphData` lowers to the autograd tape; that typed lowering is not
an assertion that every `NN.IR` node is differentiable. A new
operator should therefore add three things in the right places:

- its shape contract in `OpContracts`/`Infer`;
- its semantics in `Semantics` or a documented payload-backed evaluator;
- the runtime/proof/checker coverage that makes the operator usable in the intended workflow.

Adding a tag without these follow-up pieces creates a graph that can be printed but not responsibly
used for verified claims.

## Payload Discipline

The graph syntax stores operation structure. It does not smuggle learned tensors into node fields.
Weights, constants, and input boxes live in payload stores keyed by node id. This is slightly more
ceremonial than embedding everything in the node, but it is much easier to audit:

- graph topology can be checked independently of parameter values,
- payload shape mismatches are explicit errors,
- certificate and verifier code can cite the node id that owns each parameter or input box,
- imported weights can be treated as artifacts rather than trusted syntax.

## Release Invariants

- Node ids are array indices: `g.nodes[i].id = i`.
- Parents always point backward: every parent id is smaller than the child id.
- Parameter tensors are not embedded in `Graph`; `const`, `linear`, and `conv` use external
  payload stores keyed by node id.
- Shape checking is centralized through `Infer.nodeOutShape`; `Graph.checkShapes` delegates to
  that implementation to avoid duplicate op-contract logic.
