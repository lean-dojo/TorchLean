import VersoManual
import VersoBlueprint

open Verso.Genre Manual

#doc (Manual) "Graphs and IR" =>
%%%
tag := "graphs-ir"
%%%

A neural network can become several different graph-like objects. During eager execution, the
runtime records a tape. During compiled execution, it builds a reusable graph-shaped artifact.
During verification, the checker needs a symbolic graph whose nodes have operation tags.

These are related, but they are not interchangeable. A tape node may contain a closure. A verifier
cannot reason generically about an arbitrary closure. It needs to know that a node is `.linear`,
`.relu`, `.conv2d`, or `.softmax`, and it needs the shape and payload convention for that operation.
That is the role of `NN.IR.Graph`.

When a model, a verifier, and a runtime disagree, the most useful diagnostic question is often the
simplest one: which graph is under discussion, and which denotation does each component attach to
that graph?

# One Word, Several Meanings

At a glance, the graph pipeline is:

- the spec layer says what operations mean;
- GraphSpec can describe architectures with typed parameter interfaces;
- runtime execution produces tapes or compiled graph artifacts;
- `NN.IR.Graph` gives the op-tagged graph consumed by widgets, exporters, runtime bridges, and
  verification passes.

We start with the low-level graph because it is the object verifiers consume. The next two pages
move upward: the spec layer explains the mathematical meanings behind the operations, and GraphSpec
explains how architectures can be authored before they are lowered to IR.

# Three Graphs, Not One

The word "graph" appears in several places in TorchLean. The distinction is worth making early.

An eager tape records what happened during one execution. It stores runtime values, parent links,
and local backward closures. It is excellent for debugging and backpropagation, but it is not the
object a verifier wants to analyze.

A compiled runtime graph is a reusable execution artifact. It is useful when the same model and
loss are evaluated many times. Its nodes are still execution objects, not the external symbolic
contract used by bound propagation.

An `NN.IR.Graph` is the symbolic graph used by inspection, export, and verification. Its nodes carry
operation tags, parent ids, and output shapes. Parameters live in a separate payload.

The verifier wants the third object. It can run bound propagation over an op-tagged graph. It cannot
soundly inspect arbitrary runtime closures as if they were mathematical operators.

# The Graph Pipeline

Here is how the graph layer sits in the full system:

$$`\text{Spec layer}
\;\to\; \text{GraphSpec / runtime layer}
\;\to\; NN.IR.Graph + \text{ Payload/ParamStore}
\;\to\; \{\text{verification},\text{widgets},\text{export}\}`

The surrounding chapters explain how this graph is produced, executed, inspected, and checked.

## Where CUDA fits (and where it does not)

Training and forward evaluation can use optional CUDA buffers and kernels for speed (*Runtime
and Autograd*). The canonical verifier IR (`NN.IR.Graph`) and its Lean denotation (`Graph.denote`
/ `denoteAll`) are still defined and executed in Lean for the verification pipeline: IBP, CROWN, and
certificate tooling consume that graph, not the GPU's internal kernel schedule. In other words: GPU
mode changes *how* some float32 primitives are implemented at runtime; it should not change the
*meaning* of the shared IR you export for verification, modulo the normal float-soundness caveats in
*Floating-Point Semantics*.

# The Canonical IR

The graph that matters most in TorchLean is the symbolic DAG carrying explicit operation tags.

The [IR graph API](https://github.com/lean-dojo/TorchLean/blob/main/NN/IR/Graph.lean) introduces `NN.IR.OpKind`, `NN.IR.Node`, and
`NN.IR.Graph`. The graph stores only structure: nodes, parent links, operation tags, and output
shapes. Parameters stay outside the graph in payload tables keyed by node id.

That keeps the IR small enough to diff, inspect, rewrite, and share across runtime, verification,
and export code without smuggling full tensors into the graph itself.

# A Tiny Network As IR

Take a small two-layer classifier:

```
x
  -> linear(W1, b1)
  -> relu
  -> linear(W2, b2)
```

The IR sees the dataflow and the operation tags:

```
node 0 : input      parents []
node 1 : linear     parents [0]   payload[1] = (W1, b1)
node 2 : relu       parents [1]
node 3 : linear     parents [2]   payload[3] = (W2, b2)
```

The graph stores topology and op tags. The payload stores weights, biases, and constants. That split
is why a verifier can say exactly which node it propagated through and which parameter tensor it
used.

# Reading An IR Node

An IR node is intentionally small. The important fields are:

- `id`: the node number, also expected to be its array index,
- `parents`: the ids of earlier nodes this node reads from,
- `kind`: an `OpKind` tag such as `.input`, `.const`, `.add`, `.linear`, `.conv2d`, `.relu`, or
  `.softmax axis`,
- `outShape`: the declared output shape.

That is the main difference from a runtime tape node. A tape node may carry closures and runtime
values. An IR node carries a symbolic operation tag. Verifiers need the latter because an IBP or
CROWN pass must be able to ask "what operation is this?" without executing arbitrary runtime code.

For example, a linear layer node has one parent: the activation input. Its weights and bias are not
extra parents. They live in the payload store keyed by the linear node id. That convention keeps the
dataflow graph readable:

```
node 0 : input        parents []
node 1 : linear       parents [0]   -- W and b live in payload.linear? 1
node 2 : relu         parents [1]
```

When a verifier says it propagated bounds through node `1`, it means it used the `.linear` opcode,
the parent bounds from node `0`, and the parameters stored for node `1`.

# The Op Vocabulary

`OpKind` is the shared vocabulary used by the IR evaluator, widgets, exporters, and verification
passes. It includes:

- structural nodes such as `.input`, `.const`, `.reshape`, `.flatten`, `.concat`, and `.permute`,
- elementwise arithmetic such as `.add`, `.sub`, `.mul_elem`, `.sqrt`, `.inv`, `.maxElem`, and
  `.minElem`,
- reductions and broadcasts such as `.broadcastTo`, `.reduceSum`, `.reduceMean`, and `.sum`,
- neural network operations such as `.linear`, `.conv2d`, `.relu`, `.tanh`, `.sigmoid`, `.softmax`,
  and `.layernorm`,
- explicit randomness nodes such as `.randUniform seed` and `.bernoulliMask seed`.

Randomness is not ambient. A seeded random node is an explicit part of the graph. That matters for
replay, debugging, and verification: the graph records the source of the mask instead of asking a
backend to remember hidden state.

The IR account is built from three closely related groups of declarations:

- [NN.IR.Graph API](https://github.com/lean-dojo/TorchLean/blob/main/NN/IR/Graph.lean)
  - `OpKind` names the operation vocabulary.
  - `Node` records ids, parents, operation tags, and declared output shapes.
  - `Graph` stores the node array and exposes the basic well-formedness checks.

- [NN.IR.Infer API](https://github.com/lean-dojo/TorchLean/blob/main/NN/IR/Infer.lean) / [NN.IR.Check API](https://github.com/lean-dojo/TorchLean/blob/main/NN/IR/Check.lean)
  - `checkInferredShapes` recomputes every declared output shape from parent shapes and op tags.
  - `checkShapes` is the public compatibility name for the same inferred-shape contract.
  - These checks are meant for compiler and backend consistency checks.

- [NN.IR.Semantics API](https://github.com/lean-dojo/TorchLean/blob/main/NN/IR/Semantics.lean)
  - The denotation defined in Lean:
    - `Graph.denoteAll` (evaluate all nodes),
    - `Graph.denote` (evaluate a chosen output id).
  - Parameters are provided via an explicit `Payload` (`const?`, `linear?`, `conv2d?`).

When debugging compilation or a verifier failure, the infoview widgets are often the fastest
place to start:

- `#shape_infer_view g` (declared vs inferred shapes)
- `#ir_exec_trace_view g, payload, input` (step by step evaluation)

The *Widgets* chapter gives concrete examples of both views.

# A Few Invariants That Matter

- Do not confuse a runtime tape with the canonical IR. They are related, but they do different jobs.
- Do not skip `checkWellFormed` and `checkInferredShapes` when developing compiler passes.
- Do not hide parameters inside the graph structure if the API expects an external payload.
- Do not assume every runtime feature is automatically verifier-ready; reification has a scope.

TorchLean uses a small, explicit set of *executable* invariants for IR graphs. These checks are not
mere bureaucracy; they are the difference between:

- a verifier silently reasoning about a malformed artifact, and
- a verifier/compiler failing loudly at a precise node id with a readable error.

The most important invariants are:

- `Graph.checkWellFormed` (topology and ids)
  - ids are within bounds,
  - parents only refer to earlier nodes (topological order),
  - node ids match the index discipline expected by builders.

- `Graph.checkShapes` / `Graph.checkInferredShapes` (declared shapes agree with inference)
  - each opcode's arity is checked by the inference rule,
  - parent shapes are checked against what the opcode expects,
  - and the inferred output shape must match the node's declared `outShape`.

When writing a compiler pass or a rewrite, we treat these as the default consistency checks:

- run `checkWellFormed` and `checkInferredShapes` on every output graph while developing,
- and record "preserves well-formedness" as a proof obligation once the pass stabilizes.

# SSA Discipline

`NN.IR.Graph` follows a minimal SSA-style discipline:

- graph nodes are stored in an array,
- each node stores a list of parent ids,
- and parents must be smaller ids.

This buys several practical benefits:

- *execution* is a fold over ids (no scheduler),
- *debugging* is local (node `i` only depends on `0..(i-1)`),
- *proofs* are structural (induction over ids matches evaluation order),
- *verification passes* are dynamic programs (IBP/CROWN are prefix computations).

# Why Parameters Live Outside The Graph

The IR graph itself is just structure. Parameters are supplied by an explicit payload record
defined in [NN.IR.Semantics API](https://github.com/lean-dojo/TorchLean/blob/main/NN/IR/Semantics.lean):

- `Payload.const? : Nat → Option (ConstFlat α)`
- `Payload.linear? : Nat → Option (LinearWB α)`
- `Payload.conv2d? : Nat → Option (Conv2DParams α)`

This split is deliberate, for at least three reasons:

1. It keeps the graph small and shareable (diffs, pretty-printing, and rewrites stay lightweight
   without dragging around huge tensors).
2. It matches the real world (ONNX initializers, PyTorch `state_dict` / `nn.Module` parameters).
3. It is verification-friendly (verifiers want to attach extra metadata to parameters).

A node such as `OpKind.linear` with exactly one parent `x` is intentional: the weights and bias are
supplied through the payload map keyed by the node's id, not as extra parent edges in the DAG.

Verification code usually works with a richer `ParamStore`: it contains the payload-like parameters
needed to execute parameterized nodes, plus the metadata required by bound propagation. The helper
`payloadOfParamStore` converts that verification store back to the `Payload` expected by the IR
denotation, so the same graph can be evaluated and verified without introducing a second graph
language.

# A Compiler Checklist

When a pass emits a graph, the minimum reader checklist is:

1. Does `checkWellFormed` pass?
2. Does `checkInferredShapes` pass?
3. Does every parameterized node have the expected payload entry?
4. Does the intended output id point to the value we want to verify or export?

The first two checks are structural. The third is semantic bookkeeping. A graph can be perfectly
well formed and still fail evaluation if a `.const`, `.linear`, or `.conv2d` node is missing its
payload. That is a useful failure mode: the error is attached to a node id rather than becoming a
silent mismatch between the model and the verifier input.

# Denotation In Plain Terms

The [IR semantics API](https://github.com/lean-dojo/TorchLean/blob/main/NN/IR/Semantics.lean) centers on two functions:

- `Graph.denoteAll` evaluates the whole graph and returns a table of dynamic values
  (`DVal α := Σ s, Tensor α s`).
- `Graph.denote` evaluates the graph and returns the dynamic value at `outputId`.

Two practical notes are worth keeping in mind:

- evaluation returns `Except String ...` (so missing payloads and shape errors are explicit),
- and the evaluator performs `checkWellFormed` up front (fast path for compiler-produced graphs).

Informally, this is the compiler-output theorem shape:

$$`\operatorname{denoteAll}(g,payload,input)
= \text{the table whose entry } i \text{ is the denotation of node } i`

and the rewrite theorem shape:

$$`\operatorname{denote}(g,payload,input,outId)
=
\operatorname{denote}(\operatorname{rewrite}(g),payload,input,outId)`

# Why This IR Is Shared

Once the object of interest is a denotation of a symbolic DAG defined in Lean, the three major
workflows line up:

- Runtime: execute models through eager or compiled runtime graphs, and use the IR bridge where a
  symbolic op graph is needed for inspection, verification, or export.
- Verification: run IBP/CROWN passes on the same `NN.IR.Graph`, so bounds refer to the same denotation.
- Proofs: state correctness theorems and soundness theorems *about the IR denotation*, not about an
  opaque runtime.

This is the key to avoiding the classic verification failure mode: proving a property of the wrong
graph.

# A Tiny Worked Example

This is a minimal "input + const + add" graph, including:

- the graph structure,
- the external constant payload,
- and the denotation API.

```
import NN
import NN.Entrypoint.IR

open NN
open Spec
open NN.IR

def f2 (x y : Float) : Spec.Tensor Float (shape![2]) :=
  Tensor.dim (fun
    | ⟨0, _⟩ => Tensor.scalar x
    | ⟨_, _⟩ => Tensor.scalar y)

def g : Graph :=
  { nodes := #[
      { id := 0, parents := [], kind := .input,
        outShape := (shape![2]) },
      { id := 1, parents := [], kind := .const (shape![2]),
        outShape := (shape![2]) },
      { id := 2, parents := [0, 1], kind := .add,
        outShape := (shape![2]) }
    ] }

def payload : Payload Float :=
  { const? := fun id =>
      if id = 1 then some { n := 2, v := f2 0.25 0.25 } else none }

def input : DVal Float :=
  DVal.mk (α := Float) (shape![2]) (f2 0.6 (-0.2))

-- Typical debugging checks:
-- #eval g.checkWellFormed
-- #eval g.checkShapes
-- #eval g.checkInferredShapes

-- Evaluate the whole graph (values[2] is the output):
-- #eval g.denoteAll payload input
```

In the infoview, this graph can be inspected with:

- `#shape_infer_view g`
- `#ir_exec_trace_view g, payload, {s := ..., t := ...}`

Start from the [IR graph API](https://github.com/lean-dojo/TorchLean/blob/main/NN/IR/Graph.lean), the
[IR semantics API](https://github.com/lean-dojo/TorchLean/blob/main/NN/IR/Semantics.lean), and the
[IR checking API](https://github.com/lean-dojo/TorchLean/blob/main/NN/IR/Check.lean). That is enough context to make the rest of the
verification path much easier to follow.

# What Can Go Wrong

Most graph bugs fall into one of a few categories:

- a parent id points forward or outside the graph,
- an opcode has the wrong number of parents,
- a declared shape disagrees with the inferred shape,
- a parameterized node has no payload,
- a verifier pass supports the opcode only under additional assumptions.

TorchLean tries to make these failures local. Instead of discovering a bad certificate at the end of
a long verification run, the IR checks should identify the first malformed node or missing payload.
That is why the guide keeps returning to node ids, shapes, and payloads: they are the diagnostic
coordinates of the graph layer.

# Verification Reuses The Same Graph

Verification does *not* define a second graph type. It reuses `NN.IR.Graph` and adds per node
bound state.

The [CROWN graph API](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/CROWN/Graph.lean) does not introduce a second graph
language. Instead, it takes `NN.IR.Graph` as given and layers verification state on top of it:
interval boxes, affine forms, parameter stores, propagation state, and the passes that compute them.

This is the main reason the IR needs operation tags: without them, verification passes cannot be
generic over a runtime trace.

# Runtime Graphs Have A Different Job

Runtime graphs are designed around *execution*, so their shape is different from the verifier IR.
The [eager engine API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/Autograd/Engine/Core.lean) contains `Tape`, where each node
has runtime values and local VJP rules. The [compiled graph builder](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/Autograd/Compiled/GraphM.lean)
produces `Proofs.Autograd.Algebra.GraphData`, an SSA/DAG of closures. That is close to a "PyTorch
graph" in spirit, but the nodes are opaque closures rather than symbolic operations, so verifiers
need a separate reification step.

On the proof side, the [tape soundness API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/Autograd/Tape/Algebra/Soundness.lean)
contains the same idea again: `Proofs.Autograd.Algebra.Graph` and `GraphData` are SSA graphs whose
nodes are functions with local adjointness laws, not symbolic op codes.

A compact summary is: runtime graphs optimize for differentiable execution and debugging; the
operation-tagged IR optimizes for shared semantics, inspection, and verification passes that need an
explicit operator vocabulary.

# Where User Code Enters The IR Story

The public model-building and training APIs are covered earlier in the guide. The extra point here
is the semantic boundary:

- ordinary training may run through eager or compiled runtime execution;
- verification, export, and graph inspection need a symbolic `NN.IR.Graph`;
- the bridge succeeds by producing a graph plus payload/parameter store, or fails with an explicit
  unsupported-operator error.

So the IR chapter is not another tour of the training API. It is the contract for the artifact that
verification and graph-level tooling consume.

# Compiling TorchLean Programs To IR

The key bridge is simple to state even if the implementation takes work: compile a forward model
into the canonical IR graph.

The [TorchLean to IR compiler](https://github.com/lean-dojo/TorchLean/blob/main/NN/Verification/TorchLean/Compile.lean) turns a forward model
into an `NN.IR.Graph` plus the CROWN/LiRPA `ParamStore` payload. It supports a curated operator set,
and unsupported ops fail with explicit errors rather than silently changing semantics. The
[TorchLean correctness API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Verification/TorchLean/Correctness.lean) contains the helpers
for comparing compiled graphs with the IR semantics (`NN.IR.Graph.denote`).

GraphSpec is the typed architecture authoring layer. `NN.IR.Graph` is the lower shared IR used by
runtime compilation and verification. The compiler connects those levels while preserving a single
verifier-facing denotation.

# The Proved Forward Fragment

The semantics alignment theorem people usually want to cite is:

> evaluating the compiled IR graph equals evaluating the TorchLean forward model.

For the full public TorchLean embedding (`Runtime.Autograd.TorchLean.Program`), this is a big
theorem because that embedding is higher order and tagless final.

Rather than forcing the public API to become an AST, TorchLean takes an additive path:
a first-order forward fragment is introduced whose compiler correctness can be proved inside Lean,
and coverage grows op by op.

The [proved TorchLean compiler fragment](https://github.com/lean-dojo/TorchLean/blob/main/NN/Verification/TorchLean/Proved.lean) defines a
first order SSA/DAG language (`FGraph`) with typed indices (`Idx`) into the current context. It
provides a spec evaluator `evalForward1`, a compiler `compileForward1` into `NN.IR.Graph` plus
`ParamStore`, and the structural theorem `compileForward1_wellFormed`.
  - Proved (semantic alignment for the fragment):
    - `Proved.evalCompiledForward1_eq_evalForward1`:
      `evalCompiledForward1 (compileForward1 p params) x = evalForward1 p params x`.

The informal correctness statement is simple:

For the proved forward fragment, the compiler preserves denotation:

- take a forward program `p` (in the fragment),
- compile it to IR (`compileForward1 p params = (g, payload)`),
- then evaluating the IR graph under the IR semantics equals evaluating the fragment directly:

$$`\operatorname{Graph.denote}(g,payload,input)
=
\operatorname{evalForward1}(p,params,input)`

(with the required typing and shape well-formedness conditions, which are also proved).

Why this matters, from an ML and verification point of view:

- For the proved fragment, it eliminates the "verified the wrong graph" failure mode: proofs and
  certificates attach to the same denotation as execution.
- It makes the trust boundary explicit: if the only remaining gap is "runtime backend matches the IR evaluator",
  it's a narrow, auditable assumption.
- It scales: once the theorem exists for a fragment, coverage extends by adding one op at a time (plus its local lemma).

# Longer Term Direction

The long-term direction is to make the IR bridge broader: more public programs should be lowerable
to `NN.IR.Graph`, and more runtime paths should come with direct semantic alignment theorems. The
repository takes an incremental route: prove a first-order forward fragment, then extend coverage
operator by operator.

Next: *Verification* (TorchLean to IR to bounds), *Floating-Point Semantics* (which scalar backend
the verifier uses), *Widgets* (infoview inspection of graphs and bounds).

# References

- TorchLean paper: https://arxiv.org/abs/2602.22631
- [PyTorch FX docs](https://pytorch.org/docs/stable/fx.html)
- [ONNX](https://onnx.ai/) (common export boundary in practice)
- [GraphViz DOT language](https://graphviz.org/doc/info/lang.html)
- SSA overview: `https://en.wikipedia.org/wiki/Static_single_assignment_form`
