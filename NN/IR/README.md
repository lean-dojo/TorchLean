# `NN/IR`

`NN.IR` is TorchLean's shape-annotated, op-tagged topological DAG interchange representation. It is
the small shared graph language used by model lowering, verification, export, widgets, and forward
execution by its semantics. It is distinct from the dependent `GraphData` representation that the
`autograd` transforms record.

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
- `DType.lean` and `Program.lean`: per-node element types, explicit casts, typed extension
  signatures, and designated inputs/results on that same graph.
- `Interpreter.lean`: heterogeneous tensor evaluation, reusing `Graph.evalNodeRaw` for ordinary
  arithmetic and `Graph.evalTransform` for storage-only operations. Its scalar family and
  implementations are explicit; unsupported operations fail.
- `Binary.lean` and `BinaryProofs.lean`: FloatLib-backed binary execution at each node's precision
  and the theorem relating interpreter casts to FloatLib's directed conversions.
- `Properties.lean`: induction and relational rules for proving properties through the existing
  typed interpreter, including its accumulated conversion flags and designated outputs.
- `Reference.lean`: the same binary arithmetic plus Boolean and arbitrary-width integer storage
  for layout operations, with checked complete-word encoding and a lossless decoding theorem.
- `OpContracts.lean`: shared shape arithmetic for ops such as concat, matmul, pooling, conv, and
  axis-moving utilities. Concat and matmul use list-indexed arbitrary-rank contracts.
- `Infer.lean`: canonical declared-output-shape validation. `Graph.inferShapesFrom` walks the node
  array once, structurally recursive on the remaining node count, and `Graph.checkShapes` wraps it.
- `Check.lean`: public validation wrappers and proposition-level `WellFormed` / `WellShaped` names.
- `Semantics.lean`: denotational evaluator into spec-layer tensor operations with explicit payloads,
  including arbitrary-leading linear and matrix multiplication, checked arbitrary-axis concat, and
  the scoped `IR` notation for graph denotation. `Graph.evalNodeRaw` is the operator dispatch and
  `Graph.evalNode` adds the declared-shape normalization.
- `Payload.lean`: external constants, weights, normalization parameters, and checked custom
  computation bodies keyed by node id.
- `HardMask.lean`: conversions between typed Boolean tensors and the row-major masks stored in
  `OpKind`, shared by graph builders, evaluators, and verifier passes.
- `ShapeSoundness.lean`: the theorems relating `Infer` and `Semantics` (see below).
- `Pretty.lean`: readable text and GraphViz renderers for debugging.

## One Shape Rule Per Operation

Custom operations declare their input shapes in operand order and their output shape. Their
`NN.Kernel.Program` bodies live in the payload, not in a display name or an opaque source string.
Shape inference checks the declared signature; evaluation checks the actual parent tensors and
runs the body with bounded reads. The body's source-correspondence proof also covers failed reads.
This does not imply a derivative rule, a range rule, or an export translation. Those consumers must
report unsupported custom operations until the corresponding implementation is supplied.

Typed `Program` extensions use the same custom-node shape contract, with their element signature
and implementation supplied separately. A typed cast must run through a typed interpreter;
extracting `program.graph` and interpreting it homogeneously is not mixed-precision execution.

Reshape, flatten, broadcast, permutation, transpose, concatenation and slicing need tensor storage, not
scalar arithmetic. Their shared evaluator works unchanged for Boolean, integer, rational, symbolic
and user-defined elements when the interpreter supplies their representation. `Interpreter.reference`
supplies Boolean and fixed-width integer storage alongside FloatLib binary arithmetic. The
binary-only interpreter remains available separately. `detach` remains on the scalar-context path
because dual-valued tensors
must also clear differentiation metadata, rather than merely copy their stored values.

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
| `slice` | `Spec.Shape.replaceAxis` after an axis and range check; the evaluator applies `Tensor.sliceAxisRangeSpec` and checks the declared shape |
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

## How A Tensor Program Reaches IR

The same IR represents general tensor computations. A program need not contain a layer, a loss,
parameters, or an optimizer: its interface is an ordered collection of tensor inputs and results.
Supported array operations and numerical algorithms use the same shape and element-type checks
as model computations. This is not a compiler for arbitrary Lean or Python programs; control flow
and operations still need explicit lowering rules.

From Python, `torchlean.verify(model, *inputs)` automatically captures a supported PyTorch
computation, translates it to the shared IR, and requests the typed graph checks. Select
`frontend="jax"` for Jaxpr capture. No manual export is required. Other producers can pass a typed
artifact dictionary directly. Optional execution and comparison concern only supplied arguments.

Add `export="Model.lean"` to retain that same capture as a self-contained Lean module. It embeds
the artifact and defines `Model.program` through the shared importer. Lean checks the included
import and shape/type/interface proofs by kernel reduction when the module is compiled; export
does not launch a compiler. Mathematical properties can refer
to that program and the embedded constants; they still require their own proofs. A `.json` export
keeps the interchange artifact instead. Neither export replaces an existing file. See the
[captured-computation example](../../home_page/examples/verify/index.md).

There are several routes into the same graph language:

- TorchLean model code can lower supported fragments into IR for execution, inspection, or
  verification.
- GraphSpec models can be lowered when the architecture needs explicit sharing and
  named parameter layouts.
- PyTorch `torch.export` records positional tensor arguments, including unused arguments, from
  its graph signature. One input uses version one; multiple inputs use version two's `input_ids`.
  The ONNX adapter emits version one. Lean parses and validates the resulting tensor graphs.
- The Jaxpr adapter in `NN/Runtime/JAX/jaxpr.py` lowers static tensor computations to the same
  format. Multiple tensor arguments use version two's ordered `input_ids`; each input has its
  own shape and dtype. `Import.JAX.parseProgram` checks per-node types and explicit casts.
  `Interpreter.binary` uses FloatLib for mixed binary precision; the homogeneous graph path remains
  scalar-polymorphic. Structured control flow is not yet lowered; see the
  [JAX example](../../home_page/examples/jaxpr/index.md).
- Verification examples can build small graphs directly when the graph itself is the artifact under
  study.

Mixture-of-experts routing is one operation, `top_k_mask k`, whose meaning is
`Spec.MoE.topKMaskSpec`: one for the `k` largest entries of every row, ties to the lower index. The
PyTorch adapter captures the usual idiom `torch.zeros_like(p).scatter(-1, torch.topk(p, k).indices,
1)` as that node; `torch.topk` leaves the order of ties unspecified, so the two agree whenever the
`k`-th and `k + 1`-th largest entries of a row differ. `NN/Tests/IR/MoEImport.lean` exports a
Mixtral-style block from PyTorch and checks that the imported graph computes `Spec.MoE.forwardSpec`
bit for bit.

Frontend coverage is narrower than the graph's operator vocabulary. JAX division is rejected
until its arithmetic schedule has a direct lowering; JAX mean and LayerNorm expansions therefore
remain unsupported. Square root needs a structurally nonnegative argument to agree with the
IR's negative-input clamp. The full JAX softmax expansion also needs a `reduce_max` lowering,
so importing arbitrary attention code is not supported yet.

## Role And Scope

Write ordinary models through `TorchLean.nn`, `Trainer`, or `GraphSpec`, then lower them.
Direct `Node` construction is useful for developing IR consumers and certificate examples.

## Current Consumers

| Consumer | How it uses IR |
| --- | --- |
| Execution | `Graph.denote` and `Graph.denoteAll` run a graph by its semantics, and the typed interpreter in `Interpreter.lean` runs it at runtime precision. |
| Verification | Runs IBP/CROWN-style passes, margin checks, and certificate replay over node ids and payloads. |
| TorchLean lowering fragments | Prove that supported source fragments lower to IR with the same denotation. |
| PyTorch/ONNX/export paths | Use a small graph format to make parameter order and tensor shapes explicit at the boundary. |
| Widgets and graph pages | Render graphs, inferred shapes, execution traces, and dependency structure for debugging. |

## Proof And Runtime Status

The IR is the object shared by proofs and runtime code, but proof coverage is still named
fragment-by-fragment. Current theorem work covers supported evaluator bridges, graph well-formedness
conditions, and verification-oriented bound propagation. Separately, `GraphData` lowers to the autograd tape; that typed lowering is not
an assertion that every `NN.IR` node is differentiable. A new
operator should therefore add three things in the right places:

- its shape contract in `OpContracts`/`Infer`;
- its semantics in `Semantics` or a documented payload-backed evaluator;
- the runtime/proof/checker coverage that makes the operator usable in the intended workflow.

Adding a tag without these follow-up pieces creates a graph that can be printed but not responsibly
used for verified claims.

## Proving Properties Across A Program

An imported `Program` already has Lean semantics. A proof can work with those operations directly,
without first identifying the graph as an MLP, transformer, or another named architecture.
`NN.IR.Properties` provides the common traversal argument:

- `Interpreter.run_induction` carries a predicate on intermediate values and conversion flags.
  To establish a bound, prove it for the initial state and show that each successful operation
  preserves the predicate.
- `Interpreter.run_rel` relates two executions of the same typed program. The interpreters,
  arguments, and parameter payloads can differ. A relation can compare exact and rounded values,
  or inputs related by a permutation, provided its local rules account for every operation.
- `Interpreter.eval_rel` carries a pointwise relation through to the designated outputs, including
  repeated outputs and arbitrary return order.

The proof rules use the existing `Tensor` values, `Program`, and interpreter. Numerical relations
must use that interpreter's actual arithmetic, casts and parameters. The traversal theorem does not
supply error bounds or invariance rules for an operation whose corresponding proof is missing.

These theorems assume the executions being compared succeed. They do not establish success for
every input, prove a frontend translation faithful, or verify external kernels. Comparing two
different graph structures additionally needs a correspondence between those computations; the
same-program relational rule does not establish arbitrary optimization equivalence.

## Element Descriptors And Extensions

`DType.toJson` and `DType.fromJson` share one frontend-independent representation. Binary
descriptors retain exponent width, fraction width, bias, and exceptional-value encoding. Decimal
descriptors retain FloatLib's declet count, exponent width, and signed bias. Integer widths,
posit widths, complex component types, and named extensions are represented explicitly too.
Named-type component arrays preserve qualified names without losing numeric or escaped components.
Element descriptors and custom calls share the same total decoder for component arrays and Lean
qualified names, including escaped components.
`DType.fromJson_valid` proves that a successfully decoded descriptor passes its parameter checks.

`Reference.read_write` proves lossless element-word decoding for every supported format and
integer width. Boolean words are zero or one. Integers use Lean's `BitVec n`, retaining signedness
in the descriptor; no conversion to a host float is involved. This reference implements their
layout operations, not integer arithmetic or casts between distinct integer types. Other scalar
families still need their own implementation.

`Import.IR.parsePayload` takes an interpreter and a word decoder explicitly. The same loader
checks constants and linear parameters at their declared types and shapes, including decoder
results. A custom decoder cannot silently return a tensor at a different type or shape.

A producer can declare a custom tensor node with `primitive = "call"`, a qualified `name`,
`input_shapes`, `input_dtypes`, and `output_dtype`. The shared checker compares this signature
with the actual edges and result. Execution additionally requires an implementation in
`Interpreter.Payload.call`; merely declaring the signature does not register executable code or
prove its mathematical properties. The default binary payload rejects unimplemented calls.
Likewise, a valid decimal, posit, or named descriptor does not imply that the binary interpreter
can execute it. Other interpreters must provide the corresponding scalar operations.

## Payload Discipline

`NN.IR.Payload` stores constants, weights, normalization parameters, and custom bodies by node id.
Topology checking needs no parameter values; evaluation rejects missing or inconsistent payloads.
Verifier input boxes use a separate payload. Imported weights remain artifacts to validate.

## Release Invariants

- Node ids are array indices: `g.nodes[i].id = i`.
- Parents always point backward: every parent id is smaller than the child id.
- Parameter tensors are not embedded in `Graph`; `const`, `linear`, and `conv` use external
  payload stores keyed by node id.
- Shape checking is centralized through `Infer.nodeOutShape`; `Graph.checkShapes` delegates to
  that implementation to avoid duplicate op-contract logic.
