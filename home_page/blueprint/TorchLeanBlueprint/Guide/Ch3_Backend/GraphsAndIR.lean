import VersoManual
import NN.API
import NN.IR
import NN.Backend
import NN.MLTheory.CROWN.Extras.FP32
import NN.MLTheory.CROWN.Proofs.GraphRunibpEndToEnd
import NN.Examples.DeepDives.OneSemanticUniverse
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.Roles

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean
open TorchLean
open NN.IR
open NN.MLTheory.CROWN (NonlinearBoundOps)
open NN.MLTheory.CROWN.Graph (runIBP)
open NN.Examples.DeepDives.OneSemanticUniverse
open Lean.Elab.Tactic.GuardMsgs.WhitespaceMode (lax)

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)

#doc (Manual) "Graph IR" =>
%%%
tag := "graphs-and-ir"
file := "The-Canonical-Graph-IR"
%%%

GraphSpec preserves architecture and shape constraints in its type. Importers, exporters, and
verification passes need something else: a first-order representation that can be read before it is
known to be valid. `NN.IR.Graph` is that representation, an array of operation nodes that can be
traversed, serialized, checked, and assigned to kernels.

I'll keep the six-node network from
{src "NN/Examples/DeepDives/OneSemanticUniverse.lean"}[`OneSemanticUniverse.lean`] fixed as we
change its parameter payload and scalar interpretation. We can then compare the computed values,
propagated intervals, and kernel plan for the same nodes. A PyTorch trace provides a second view
of their connections. Every Lean
block below is elaborated while this page is built. The Python examples and shell commands let
you inspect the graph and compare its interpretations in your build.

# Executable Graph Types And Lowering

`Torch.TypedGraph` stores shape-indexed executable functions for
forward evaluation, JVPs, and VJPs, together with a typed reference selecting its output. The
canonical IR has no second executable form: `Graph.denoteAll` runs it directly, by its semantics.
Importing forward semantics does not supply a derivative rule, so the IR stays a forward object.
{ref "runtime-autograd"}[Runtime and Autograd] explains what the typed graph can justify.

The principal executable paths are:

```
-- Follow the executable export route separately from its
-- semantic evidence.
GraphSpec.Chain ── toProgram ──> TorchLean.Runtime.Program
                                             │
                                             ├── Autodiff.lowerToTypedGraph
                                             │   ──> Torch.TypedGraph
                                             │        (forward + JVP + VJP)
                                             │
                                             └── Verification.lowerForwardToIR
                                                 ──> NN.IR.Graph
                                                       │
                                                       └── Graph.denoteAll
                                                            (forward only)

NN.IR.Graph ── kernel selection ──> GraphKernelPlan
                                     (metadata only)
```

The diagram separates forward execution from execution with derivative rules. The `TypedGraph`
branch records forward, JVP, and VJP operations; the IR branch is evaluated by its semantics.
The kernel-selection branch produces metadata that must later be connected to runtime handlers.

`TorchLean.Runtime.Program` is an operation-polymorphic function, not a stored graph. The
broad `lowerForwardToIR` pass executes that function with an IR-building interpreter and validates
the graph it produces. It does not by itself carry an end-to-end source-lowering theorem.
Stable log-softmax and GELU currently return lowering errors, rather than replacing them with
numerically different expansions. The eager runtime still supports both operations.

The theorem-backed source path is deliberately separate:

```
-- This proved route has its own supported program fragment.
Proved.ForwardProgram ── Proved.lowerForwardProgramToIR ──> NN.IR.Graph
          │                                                        │
   Proved.evalForward                                      runForwardIR
          │                                                        │
          └──────────────────── equal by theorem ──────────────────┘
```

`Proved.ForwardProgram` is a smaller first-order let-chain whose constructors expose the fragment
covered by the theorem. It is not an alias for the general `TorchLean.Runtime.Program` interface.

# MLP Graph Structure

The running example is

$$`
G(x)=\tanh\!\left(
  \sum_k
  \left[W_2\operatorname{ReLU}(W_1x+b_1)+b_2\right]_k
\right),
`

with $`x\in\mathbb{R}^4`, a hidden width of five, and an output width of three.
$`W_1x+b_1` produces five hidden preactivations; ReLU preserves their shape. The second affine
layer produces three coordinates, the sum reduces them to a scalar, and `tanh` bounds that scalar.
This gives us external parameters, a piecewise-linear activation, a rank-changing reduction, and
a nonlinear interval rule in one small graph.

TorchLean stores it as six nodes, and the graph is ordinary Lean data, so we can just print it:

```lean (name := irPretty)
-- Print node ids, parent edges, and declared result shapes
-- for the running graph.
#eval IO.println graph.pretty
```

```leanOutput irPretty (whitespace := lax)
0: input parents=[] out=[4]
1: linear(payload=node_id) parents=[0] out=[5]
2: relu parents=[1] out=[5]
3: linear(payload=node_id) parents=[2] out=[3]
4: sum parents=[3] out=[]
5: tanh parents=[4] out=[]
```

The parent list gives data dependencies. Because every parent id is smaller than the node id, array
order is already a topological execution order, so validation and evaluation can scan left to right
without a graph search. `linear(payload=node_id)` is the pretty-printer telling you that this node's
weights are not in the node; they are fetched from a payload keyed by the node id, which is the
subject of a later section.

That listing renders the stored node array. Here is the source that produced it, quoted
from the example file:

```
-- The node array gives a topological program; parameters
-- remain in a separate payload.
def graph : NN.IR.Graph :=
  let inputNode : NN.IR.Node :=
    { id := 0, parents := #[], kind := .input, outShape := [4] }
  let hiddenLinearNode : NN.IR.Node :=
    { id := 1, parents := #[0], kind := .linear, outShape := [5] }
  let hiddenActivationNode : NN.IR.Node :=
    { id := 2, parents := #[1], kind := .relu, outShape := [5] }
  let outputLinearNode : NN.IR.Node :=
    { id := 3, parents := #[2], kind := .linear, outShape := [3] }
  let reductionNode : NN.IR.Node :=
    { id := 4, parents := #[3], kind := .sum, outShape := [] }
  let outputNode : NN.IR.Node :=
    { id := 5, parents := #[4], kind := .tanh, outShape := [] }
  { nodes :=
      #[inputNode, hiddenLinearNode, hiddenActivationNode, outputLinearNode, reductionNode,
        outputNode] }
```

Unlike GraphSpec, this datatype does not make every edge shape-correct by construction. Node ids,
raw axis numbers, and declared output shapes are ordinary data. That is deliberate: an importer must
be able to construct a candidate graph from an external document before Lean knows it is valid. The
representation therefore needs an explicit validation phase before its shape declarations can be
used.

There is also a GraphViz rendering for when six nodes become sixty:

```lean (name := irDot)
-- Render the same parent relation as Graphviz input.
#eval IO.println graph.toDot
```

```leanOutput irDot (whitespace := lax)
digraph IR {
  rankdir=LR;
  node [shape=box, fontsize=10];
  n0 [label="0: input\nout=[4]"];
  n1 [label="1: linear(payload=node_id)\nout=[5]"];
  n2 [label="2: relu\nout=[5]"];
  n3 [label="3: linear(payload=node_id)\nout=[3]"];
  n4 [label="4: sum\nout=[]"];
  n5 [label="5: tanh\nout=[]"];
  n0 -> n1;
  n1 -> n2;
  n2 -> n3;
  n3 -> n4;
  n4 -> n5;
}
```

Write that to a file and run `dot -Tpng g.dot -o g.png`. Edges point from producer to consumer,
matching the dataflow picture the frameworks use. Neither printer is a serialization format, and
{src "NN/IR/Pretty.lean"}[`Pretty.lean`] says so in its module docstring: the IR gains
operations and invariants over time, and the pretty output should not become a compatibility
obligation.

The shape `[]` on the last two nodes denotes a scalar, with one value. When inspecting a larger
graph, follow the edges as well as the shapes: an edge that bypasses ReLU may still connect
compatible shapes while changing the computation.

# PyTorch Graph Capture

PyTorch's op-tagged graph exposes the same computation through a different node interface.
Write the network with `nn.Linear`, copy in the same weights,
and trace it ({Informal.citep fx2022}[]):

```
# Build the matching tensor computation before comparing the
# exported FX structure.
import torch, torch.nn as nn, torch.fx as fx

W1 = torch.tensor([[.15, -.12, .08, .05], [.02, .11, -.09, .07],
                   [-.04, .06, .10, -.03], [.09, .01, .04, .13],
                   [-.07, .03, .12, -.02]])
b1 = torch.tensor([.01, -.02, .03, 0., .02])
W2 = torch.tensor([[.05, .08, -.06, .03, .07],
                   [-.04, .02, .09, -.01, .06],
                   [.10, -.03, .04, .05, -.08]])
b2 = torch.tensor([.02, -.01, 0.])

class G(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(4, 5); self.l2 = nn.Linear(5, 3)
        with torch.no_grad():
            self.l1.weight.copy_(W1); self.l1.bias.copy_(b1)
            self.l2.weight.copy_(W2); self.l2.bias.copy_(b2)
    def forward(self, x):
        return torch.tanh(self.l2(torch.relu(self.l1(x))).sum())

m = G().eval()
x0 = torch.tensor([0.3, -0.2, 0.1, 0.4])
print("torch y(x0) = %.6f" % m(x0).item())
g = fx.symbolic_trace(m)
for n in g.graph.nodes:
    print("%-14s %-8s %-10s %s" % (n.name, n.op, n.target, n.args))
print("fx node has a shape field:", hasattr(list(g.graph.nodes)[0], "shape"))
```

Compare the printed operation names, parent arguments, parameter references, and output selection
with the six-node listing above. Keep the output value for the numerical comparison below.

An FX node's `target` is a
Python object: a builtin, a submodule path, a method name, or any function you happened to call.
TorchLean's `kind` is a constructor of a single inductive type with a few dozen operations, so a
pass must provide exhaustive case handling. A wildcard branch can still accept or reject many
cases together, so compilation alone does not prove each operation has the intended support.

The FX node has no shape field
in its core record. Shape information can be attached as metadata, for example by propagating a
concrete input (other capture paths can use symbolic or fake-tensor metadata):

```
# Run shape propagation to see which metadata comes from
# executing example inputs.
from torch.fx.passes.shape_prop import ShapeProp
print("before:", {n.name: sorted(n.meta.keys()) for n in g.graph.nodes})
ShapeProp(g).propagate(x0)
print("after: ", {n.name: tuple(n.meta["tensor_meta"].shape)
                  for n in g.graph.nodes if "tensor_meta" in n.meta})
```

The expected feature widths are four at the input, five after the first linear layer and ReLU,
and three after the second linear layer. The sum and `tanh` return scalars. Shape propagation
records what happened on the supplied input by executing the graph. TorchLean's declared shapes
are checked before execution, with a theorem connecting that check to the semantics.

`call_module l1` means "go find the submodule named `l1` and
call it", so the parameters are inside the traced object. TorchLean keeps the graph free of tensors
and passes a payload separately, which is what makes it possible to run the same graph with interval
parameters later in this chapter.

FX adds an explicit `output` node, so its listing has seven nodes. TorchLean names the output by
id, and `Graph.denote` takes `outputId := 5`. The extra FX node selects the result; it adds no
tensor calculation.

We can compare the printed PyTorch output at `x0` with the IR evaluator after supplying the
parameter payload.

# The Node Record

The node record stores the information that validation and evaluation will use:

```
/-- Node in the graph. Edges are implicit via parent indices. -/
structure Node where
  /-- Node id. By convention this is also the node's index in `Graph.nodes`. -/
  id       : Nat
  /-- Parent node ids, i.e. data dependencies. Each parent must be smaller than `id`. -/
  parents  : Array Nat
  /-- Operation tag and any operation-local metadata. -/
  kind     : OpKind
  /-- Declared output shape. `NN.IR.Infer` can recompute/check this from parents. -/
  outShape : Shape
  deriving Repr
```

`kind` carries operation-local data such as an axis, a permutation, convolution geometry, or a hard
mask. Large tensor values do not belong there; constants and learned parameters live in the payload.
To evaluate this record safely, `id` must agree with the array position,
every parent must already exist, the parent count must suit the operation, and `outShape` must agree
with independent inference. These are checks on the record as a whole; the operation tag alone
does not establish them.

This separation of operation data from validation also appears in MLIR
({Informal.citep mlir2021}[]); ONNX similarly stores initializers separately from operations.
Here we can inspect both the checks and the Lean theorems connecting them to the graph semantics.

# Structural And Shape Validation

Validation separates graph structure from operation-specific shape consistency.

`Graph.checkWellFormed` checks structure only: node id equals array position, parents occur earlier
in topological order, the operation has an admissible number of parents, and designated input and
constant nodes have the required arity. `Graph.checkShapes` then runs the operation-specific shape
rules from {src "NN/IR/Infer.lean"}[`Infer.lean`]. Our graph passes both:

```lean (name := irChecks)
-- Check index consistency and the order of parent
-- references.
#eval graph.checkWellFormed
```

```leanOutput irChecks (whitespace := lax)
Except.ok ()
```

```lean (name := irChecks2)
-- Check the result shapes predicted by each operation.
#eval graph.checkShapes
```

```leanOutput irChecks2 (whitespace := lax)
Except.ok ()
```

Now break it three ways, one field at a time. First, declare the wrong output shape on the ReLU
node, leaving the structure untouched:

```lean (name := irBadShape)
-- Keep the edges valid while changing only one declared
-- shape.
/-- The same graph, with node 2 claiming `[4]` not `[5]`. -/
def irBadShape : Graph :=
  { nodes := graph.nodes.set! 2
      { id := 2, parents := #[1], kind := .relu
        outShape := [4] } }

#eval irBadShape.checkWellFormed
```

```leanOutput irBadShape (whitespace := lax)
Except.ok ()
```

```lean (name := irBadShape2)
-- Shape inference now detects the inconsistency that
-- structural checking accepted.
#eval irBadShape.checkShapes
```

```leanOutput irBadShape2 (whitespace := lax)
Except.error "IR graph: node 2: outShape mismatch:
  inferred=[5], declared=[4] (Node(id=2, kind=relu,
  parents=#[1], outShape=[4]))"
```

The parent edge still points to an earlier node, so the structural check succeeds. ReLU must
preserve its parent's shape, however, and `[4]` disagrees with the parent's `[5]`. The shape
checker reports that mismatch. The evaluator performs additional checks and can also reject it
before downstream use; a successful structural check alone does not establish shape consistency.

Second, break the topological order by pointing node 2 at node 3:

```lean (name := irCycle)
-- A reference to a later node violates the order required
-- by the evaluator.
/-- Node 2 now reads node 3, which does not exist yet. -/
def irCycle : Graph :=
  { nodes := graph.nodes.set! 2
      { id := 2, parents := #[3], kind := .relu
        outShape := [5] } }

#eval irCycle.checkWellFormed
```

```leanOutput irCycle (whitespace := lax)
Except.error "IR graph: node 2: parent id 3 is not < 2
  (Node(id=2, kind=relu, parents=#[3], outShape=[5]))"
```

Third, break the id discipline, which is the invariant that lets ids double as array indices:

```lean (name := irBadId)
-- The stored id must agree with the node’s array position.
/-- The node at index 2 claims to be node 7. -/
def irBadId : Graph :=
  { nodes := graph.nodes.set! 2
      { id := 7, parents := #[1], kind := .relu
        outShape := [5] } }

#eval irBadId.checkWellFormed
```

```leanOutput irBadId (whitespace := lax)
Except.error "IR graph: id discipline violated at
  index 2: nodes[2].id = 7"
```

Each failed check returns an `Except.error` identifying the inconsistent field, so an importer can
report the problem. A shape change cannot repair a forward reference, and renumbering cannot
repair a shape mismatch.

## Relating Shape Checking To Evaluation

The shape checker and evaluator use separate matches over `OpKind`. Their agreement is stated in
{src "NN/IR/ShapeSoundness.lean"}[`ShapeSoundness.lean`]:

```lean (name := irSound)
-- Read the successful-evaluation premise as well as the
-- successful-shape-check premise.
#check @NN.IR.Graph.checkShapes_sound
```

```leanOutput irSound (whitespace := lax)
@Graph.checkShapes_sound : ∀ {α : Type}
  [inst : Storage α] [inst_1 : Context α]
  (g : Graph) (payload : Payload α)
  (input : Spec.SomeTensor α)
  (vals : Array (Spec.SomeTensor α)),
  g.checkShapes = Except.ok () →
    g.denoteAll payload input = Except.ok vals →
      ∀ (i : ℕ) (hi : i < g.nodes.size)
        (hiv : i < vals.size),
        vals[i].shape = g.nodes[i].outShape
```

The theorem assumes that shape checking and evaluation both succeed. Its conclusion identifies
each computed value's shape with the corresponding node's declaration. The companion
`Graph.denoteAll_shape` says the same thing without assuming
`checkShapes`, because `evalNode` normalizes each value to the declared shape; its docstring points
at `denoteAllRaw_eq_denoteAll`, which shows the normalization never actually changes anything on a
graph the checker accepted. The declared shapes are therefore the shapes the semantics computes, and
we can watch that happen:

```lean (name := irShapesAgree)
-- These are the shapes declared by the graph artifact.
#eval graph.nodes.map fun n => n.outShape
```

```leanOutput irShapesAgree (whitespace := lax)
#[[4], [5], [5], [3], [], []]
```

```lean (name := irShapesAgree2)
-- These are the shapes of the values actually returned by
-- its reference evaluator.
#eval (Graph.denoteAll (α := Float) (g := graph)
    (payload := payload parameters)
    (input :=
      Spec.SomeTensor.ofTensor center)).map
  fun vals => vals.map fun v => v.shape
```

```leanOutput irShapesAgree2 (whitespace := lax)
Except.ok #[[4], [5], [5], [3], [], []]
```

The distinction also matters in the backend adapter. `NN.Backend.IR.checkedPlanGraph` calls
`checkWellFormed` before selecting kernels, and does not run the shape check. A caller accepting
untrusted graph data must not infer shape validity from a successful plan.

The successful-evaluation premise also leaves payload lookup to the evaluator. Shape checking
cannot supply a missing weight tensor, or ensure that a tensor stored under a node id has the
dimensions that node expects.

# Parameter Payloads

The two `.linear` nodes mention only their activation parent. Their weights and biases come from a
payload keyed by node id, and this is the real definition from the example:

```
-- Resolve the two linear nodes to parameter tensors with
-- their own checked dimensions.
def payload {α : Type} [TorchLean.Storage α] [Context α]
    (parameters : Parameters α) : NN.IR.Payload α :=
  { linear? := fun id =>
      if id = 1 then
        some {
          outDim := 5
          inDim := 4
          W := parameters.hiddenWeight
          b := parameters.hiddenBias
        }
      else if id = 3 then
        some {
          outDim := 3
          inDim := 5
          W := parameters.outputWeight
          b := parameters.outputBias
        }
      else
        none }
```

The scalar type parameter and the `Storage` and `Context` instances determine the payload's tensor
values. The node ids stay fixed when we change this scalar interpretation, so the same graph can
later read interval parameters.

Separating structure from values has practical consequences:

- one graph can be reused with initial, trained, or bounded parameters;
- checkpoint loading changes the payload without rebuilding the node array;
- a verifier can replace concrete parameters by interval metadata;
- an exporter can emit graph nodes and initializers through different channels.

It also creates an obligation that no type will discharge for you. The payload's `LinearWB` record
carries its own `outDim` and `inDim`, and its `W : Tensor α [outDim, inDim]` is shape-correct with
respect to *those* fields, not with respect to the node. So a payload can be internally consistent
and still disagree with the graph. Keying by node id makes that failure possible; here it is:

```lean (name := irBadPayload)
-- A present payload key is insufficient when the matrix
-- dimensions disagree.
/-- A payload whose key exists, whose weight is a perfectly
good matrix, and whose dimensions do not match node 1. -/
def irBadPayload : Payload Float :=
  { linear? := fun id =>
      if id = 1 then
        some { outDim := 3, inDim := 4
               W := Tensor.full [3, 4] 0.1
               b := Tensor.full [3] 0.0 }
      else if id = 3 then
        some { outDim := 3, inDim := 5
               W := parameters.outputWeight
               b := parameters.outputBias }
      else none }

#eval (Graph.denote (α := Float) (g := graph)
    (payload := irBadPayload)
    (input :=
      Spec.SomeTensor.ofTensor center)
    (outputId := 5)).map fun v => Spec.pretty v.tensor
```

```leanOutput irBadPayload (whitespace := lax)
Except.error "IR eval: linear 1: declared outShape
  mismatch: [5] vs expected [3]"
```

The lookup at key `1` succeeds, but its three-output matrix disagrees with node 1's declared
five-output shape. Evaluation reports both dimensions and names the node.

The shared `NN.IR.Payload` currently has typed records for constants, linear weights and bias,
convolution parameters, layer-normalization parameters, and eval-mode channel-axis BatchNorm
parameters. Other operations obtain their values from parent edges. Adding a new payload-backed
operation requires coordinated changes to the payload type, shape inference, denotation,
import/export adapters, and every runtime or verifier that claims to support it, which is a good
reason to prefer an operation that reads its inputs from edges when either would do.

Equal dimensions would still not prove that the correct checkpoint tensor had been assigned
to a node. The GraphSpec bias-swap example passed that check while changing the output.

# A Heterogeneous Value Table

During evaluation, node 0 holds a vector of four, nodes 1 and 2 hold five, node 3 holds three, and
nodes 4 and 5 hold scalars. One homogeneous Lean array cannot directly contain all those tensor
types, so the evaluator stores the specification layer's shape-erased tensor:

```
-- The existential package retains a tensor together with
-- the shape indexing its type.
structure Spec.SomeTensor (α : Type) where
  shape : Shape
  tensor : TorchLean.Tensor α shape
```

This is a dependent pair of a runtime shape and a tensor with exactly that shape.
`Graph.expectShape`
recovers a statically typed tensor only after checking the stored shape, and
{ref "runtime-autograd"}[Runtime and Autograd] shows the `cast` that demands a proof and the
round-trip theorem saying nothing was lost. Model code never builds this table; its inputs stay
`Tensor α shape`.

For node 1, evaluation performs:

1. fetch parent 0 from the value table;
2. fetch the linear payload keyed by `1`;
3. check the parent tag equals `[4]`;
4. check the declared output equals `[5]`;
5. call the pure `linearSpec`;
6. store the result as a `Spec.SomeTensor α`.

Step 4 is the one that produced the error message in the previous section. Failures are reported as
`Except String`; malformed imported data does not receive a fabricated proof cast.

We can read every intermediate value from this table to locate where two evaluations diverge:

```lean (name := irValues)
-- Inspect every intermediate, so the final scalar can be
-- traced through all six nodes.
#eval (Graph.denoteAll (α := Float) (g := graph)
    (payload := payload parameters)
    (input :=
      Spec.SomeTensor.ofTensor center)).map
  fun vals => vals.map fun v => Spec.pretty v.tensor
```

```leanOutput irValues (whitespace := lax)
Except.ok #["[0.300000, -0.200000, 0.100000, 0.400000]",
  "[0.107000, -0.017000, 0.004000, 0.081000, -0.003000]",
  "[0.107000, 0.000000, 0.004000, 0.081000, 0.000000]",
  "[0.027540, -0.014730, 0.014910]", "0.027720",
  "0.027713"]
```

Node 1 is the affine layer, node 2 is the same vector
with its two negative entries clamped to zero, node 3 is the second affine layer, node 4 sums the
three components to `0.027720`, and node 5 applies `tanh` to get `0.027713`. The tanh barely moves
the value because `tanh x ≈ x` near zero. This gives us a reference for the interval calculation:
the reduction and final output at the center are both close to `0.028`.

# Scalar Interpretations

`Graph.denote` folds over the node array using the spec operations and a scalar `Context α`. Nothing
in `NN.IR.Graph` mentions `α` at all, which is why one piece of graph data can be read at:

- `α := ℝ`, for exact-real theorem statements;
- `α := FP32`, for finite rounded-real analysis;
- `α := ExecFloat.Binary 8 23`, for executable binary32 behavior;
- another valid FloatLib binary format, for typed CPU execution at a different precision;
- `α := Float`, for the ordinary runtime scalar;
- interval endpoints over any of those, for bound propagation.

The graph is the same data and the meaning of arithmetic changes with `α`. This is the same
tagless-final move the runtime uses for its two `Ops` instances
({Informal.citet kiselyov2012}[]), except that here the interpreter is a fold over stored data
rather than an instance chosen by unification.

We can compare two executable choices directly:

```lean (name := irFloat)
-- Evaluate the graph using the host Float scalar
-- interpretation.
#eval (evaluate (α := Float) parameters
  center).map Spec.pretty
```

```leanOutput irFloat (whitespace := lax)
Except.ok "0.027713"
```

```lean (name := irIEEE)
-- Convert the same parameters to the executable binary32
-- interpretation.
def irIEEEParams : Parameters (ExecFloat.Binary 8 23) :=
  parameters.map Runtime.ofFloat

def irCenter : Tensor (ExecFloat.Binary 8 23) [4] :=
  Tensor.map Runtime.ofFloat center

#eval (evaluate (α := (ExecFloat.Binary 8 23))
  irIEEEParams irCenter).map fun output =>
    Spec.pretty (Tensor.map
      (Float32.toFloat ∘ ExecFloat.Binary.toFloat32)
      output)
```

```leanOutput irIEEE (whitespace := lax)
Except.ok "0.027713"
```

The two checked Lean evaluations agree to six decimals. The first uses hardware `Float`
double-precision arithmetic; the second uses FloatLib binary32 with explicit rounding
({Informal.citep goldberg1991}[]). Its final conversion selects a decimal display without changing
the arithmetic inside the graph. Compare these values with the PyTorch output from the command
above. Agreement at the displayed precision does not establish agreement of stored values.
{ref "floats"}[Floating-Point Semantics] follows individual rounding steps to explain how such
differences arise.

The other two instantiations are noncomputable, which is a fact about their scalars and not about
the graph:

```lean (name := irNoncomputable)
-- These functions return optional boxes; a type alone does
-- not promise a bound at every node.
open NN.MLTheory.CROWN in
#check runIBP (α := ℝ) graph

open NN.MLTheory.CROWN in
#check runIBP (α := Floats.FP32) graph
```

```leanOutput irNoncomputable (whitespace := lax)
runIBP graph : Graph.ParamStore ℝ →
  Array (Option (FlatBox ℝ))
```

```leanOutput irNoncomputable (whitespace := lax)
runIBP graph : Graph.ParamStore Floats.FP32 →
  Array (Option (FlatBox Floats.FP32))
```

These functions use the same graph and propagation algorithm to construct boxes for mathematical
reasoning. `ℝ` is Mathlib's real numbers ({Informal.citep mathlib2020}[]); `FP32` models
binary32 precision and gradual underflow using rounded real values, without an upper exponent
cutoff. Flocq provides a related formal account of floating-point rounding
({Informal.citep flocq2011}[]). Relating different scalar interpretations requires explicit
hypotheses; {ref "spec-layer"}[The Specification Layer] explains those relationships.

The `Option` entries in the return types allow a node to lack a derived bound. Even when the
graph evaluates successfully, propagation may leave part of the enclosure trace unavailable.

# Interval Bound Propagation

Bound propagation replaces each value by an interval and each operation by an operation on
intervals, then reads the output interval. This is interval bound propagation, the simplest member
of the family that CROWN and its descendants refine
({Informal.citep gowal2018}[]; {Informal.citep crown2018}[]). It runs on our graph unchanged: the
node array does not know it is being interpreted over boxes.

Take the center input $`x_0=(0.3,-0.2,0.1,0.4)` and an $`\ell_\infty` ball of radius `0.05` around
it, then print the box at every node:

```lean (name := irIBP)
-- Propagate the input box and display where the interval
-- becomes less precise.
def irBox := inputBox (α := (ExecFloat.Binary 8 23))
  (eps := 0.05)

def irFlatBox : NN.MLTheory.CROWN.FlatBox
    (ExecFloat.Binary 8 23) :=
  { dim := 4, lo := irBox.lo, hi := irBox.hi }

def irStore := parameterStore
  (α := (ExecFloat.Binary 8 23)) irIEEEParams irFlatBox

def irBoxes := runIBP (α := (ExecFloat.Binary 8 23)) graph
  irStore

#eval IO.println <| String.intercalate "\n"
  ((List.range 6).map fun id =>
    match irBoxes[id]? with
    | some (some b) =>
        let display :=
          Float32.toFloat ∘ ExecFloat.Binary.toFloat32
        s!"node {id}: {Spec.pretty
          (Tensor.map display b.lo)} .. " ++
          s!"{Spec.pretty (Tensor.map display b.hi)}"
    | _ => s!"node {id}: no box")
```

```leanOutput irIBP (whitespace := lax)
node 0: [0.250000, -0.250000, 0.050000, 0.350000] ..
  [0.350000, -0.150000, 0.150000, 0.450000]
node 1: [0.087000, -0.031500, -0.007500, 0.067500,
  -0.015000] .. [0.127000, -0.002500, 0.015500,
  0.094500, 0.009000]
node 2: [0.087000, 0.000000, 0.000000, 0.067500,
  0.000000] .. [0.127000, 0.000000, 0.015500,
  0.094500, 0.009000]
node 3: [0.025445, -0.016025, 0.011355] ..
  [0.029815, -0.012220, 0.018045]
node 4: [0.020775] .. [0.035640]
node 5: [-1.000000] .. [1.000000]
```

Node 0 is the input box, the center plus and minus `0.05` in each coordinate. Node 1 applies the
affine interval rule to that box and records one lower and upper bound per output coordinate.
These bounds include endpoint rounding; the printed decimals do not assert an exact image of the
box. Compare node 2 with node 1:
every negative lower bound has become exactly `0.000000`, and the upper bounds are unchanged except
where they were negative too. That is interval ReLU, and zero-based coordinate `1` of node 2 is the
degenerate interval `[0, 0]` because the whole input interval for that coordinate was negative.
Node 3 is the
second affine layer, node 4 sums the three coordinates, and the resulting scalar interval is
`[0.020775, 0.035640]`, which does contain the value `0.027720` we computed at the center.

Node 5 widens the result to `[-1, 1]`, a global enclosure of the range of `tanh`. To explain
that loss of
information, we need to inspect the interval rule selected for this scalar type.

An interval at a node encloses the values allowed by the input box and the transfer rules used so
far. It can lose correlations between two parents that depend on the same input. For example,
separately enclosing a value and its negation does not retain the fact that their sum is exactly
zero. This loss of dependence can widen later intervals even with exact endpoint arithmetic.
Rounded endpoints introduce another issue: they must move outward to preserve enclosure. A wide
interval can therefore reflect the abstraction, the rounding policy, or a deliberately coarse
nonlinear rule, and those causes call for different improvements.

## The Executable Tanh Interval Rule

The `NonlinearBoundOps` instance chooses the `tanh` interval rule for each scalar type. We can
ask the executable binary32 instance directly, using node 4's displayed endpoints:

```lean (name := irTanhBounds)
-- Ask the executable nonlinear policy for the tanh interval
-- directly.
#eval (NonlinearBoundOps.tanhBounds
    (α := (ExecFloat.Binary 8 23))
    (Runtime.ofFloat 0.020775)
    (Runtime.ofFloat 0.035640)).map
  fun (lo, hi) =>
    ((Float32.toFloat ∘ ExecFloat.Binary.toFloat32) lo,
      (Float32.toFloat ∘ ExecFloat.Binary.toFloat32) hi)
```

```leanOutput irTanhBounds (whitespace := lax)
some (-1.000000, 1.000000)
```

The executable instance in
{src "NN/MLTheory/CROWN/Extras/BoundOpsIEEE32Exec.lean"}[the binary32 bound operations] returns the
global codomain of `tanh` after checking that its input endpoints are finite and ordered. It then
discards their range. The specialized proof-oriented instances use that range: over `ℝ`
the rule is `some (Real.tanh lo, Real.tanh hi)`, and over `FP32` it is the same with the endpoints
rounded outward. Since `tanh` is increasing, the real endpoints are tight for that scalar interval
and the FP32
endpoints enclose them with outward rounding. Host `Float.tanh` illustrates their approximate
size; this next evaluation is not a directed-rounding enclosure:

```lean (name := irTanhTight)
-- Endpoint evaluations illustrate a tighter candidate,
-- without proving outward rounding.
#eval (Float.tanh 0.020775, Float.tanh 0.035640)
```

```leanOutput irTanhTight (whitespace := lax)
(0.020772, 0.035625)
```

For the displayed node-4 interval, the real endpoint image has width about `0.0149`; the executable
codomain fallback has width `2.0`. A full rerun with another scalar type also changes the rounding
of the preceding nodes, so this calculation compares the final transfer on fixed endpoints.
The endpoint-sensitive bound is much more informative here.
`IEEE32ExecBounds.tanhBounds_containsReal` proves that the executable fallback contains ideal real
`tanh` whenever the finite transfer succeeds. Agreement with a separately rounded native `tanh`
evaluation requires a further backend theorem.

The generic real interval construction is defined in `FloatLib.Floats.Interval.Quantized`.
Its `RInterval.tanh` is noncomputable: it applies `Real.tanh` to each endpoint and rounds outward.
The executable TorchLean instance above uses the global codomain instead. A tighter executable
interval needs justified upper and lower bounds for the transcendental operation;
`[-1, 1]` already encloses the ideal real `tanh` range. Relating executable transcendental outputs
to that range is a separate backend obligation. Nearest rounding at an endpoint can round inward:
a lower bound can become too large, or an upper bound too small. A tight executable enclosure
therefore needs an error bound or a directed-rounding implementation for the transcendental function
({Informal.citep boldo2015}[]). The global enclosure avoids that endpoint calculation, at the cost
of discarding the input interval.

`sigmoid` also uses its global codomain, `[0, 1]`, for finite ordered inputs. The `exp`, `log`,
`sqrt`, `sin`, and `cos` transfers use FloatLib's certified rational interval kernels followed by
checked outward binary32 rounding. Their theorems enclose every real input in the interval;
sine and cosine include interior extrema. Domain checks and finite-output checks can still
return `none`. Square root preserves intervals crossing zero by using the real convention
`Real.sqrt x = 0` for negative `x`. `layerNormAbsBound` remains unavailable, and
`supportsIdealCoupledDerivatives` remains false. When an executable IBP result looks uselessly
wide, check which nonlinearity the graph ends with before suspecting the propagation.

For a property that needs to distinguish this small positive output from zero, `[-1, 1]` is
too wide. The missing information comes from the final transfer rule, even though the preceding
nodes retained useful bounds.

# Sampled Containment And Enclosure Theorems

The whole experiment is one command:

```terminal
# Check sampled executions against the intervals derived for
# this example.
scripts/lake.sh exe torchlean one_semantic_universe --samples 50
```

The command evaluates the center input in FloatLib binary32, propagates an output interval, and
checks fifty deterministically generated inputs against it. It reports a counterexample and fails
if a sample is outside its input box or its evaluated output is outside the propagated box.

Even if every sample passes, containment in `[-1, 1]` supplies little information about tightness:
the final interval rule uses the global range of ideal real `tanh` and discards the input range.
The comparison can still expose NaNs, broken dispatch, or an invalid range result, making it
useful as a regression check.

The command also names the pointwise theorem behind its Boolean containment check:

```lean (name := irContains)
-- The Boolean-to-proposition bridge concerns the point
-- supplied to containsDecBool.
open NN.MLTheory.CROWN in
#check @Box.containsDecBool_sound
```

```leanOutput irContains (whitespace := lax)
@Box.containsDecBool_sound : ∀ {α : Type}
  [inst : Storage α] [inst_1 : Context α]
  [inst_2 : DecidableRel fun x1 x2 => x1 ≤ x2]
  {s : Shape} (b : Box α s) (x : Tensor α s),
  b.containsDecBool x = true → b.contains x
```

It upgrades a Boolean to a proposition about the point that was checked. It says nothing about the
other points in the box, and it is not the enclosure theorem.

The enclosure theorem quantifies over graph evaluations under explicit coverage and input
assumptions:

```lean (name := irEnclose)
-- The enclosure theorem also requires supported engine
-- operations and covered nodes.
open NN.MLTheory.CROWN.Graph.CertSoundness in
#check @runIBP_encloses_evalGraphRec
```

```leanOutput irEnclose (whitespace := lax)
runIBP_encloses_evalGraphRec :
  ∀ (g : NN.MLTheory.CROWN.Graph)
  (ps : NN.MLTheory.CROWN.Graph.ParamStore ℝ)
  (inputs : Std.HashMap ℕ Val),
  TopoSorted g →
    EngineCore g →
      IBPCovers g (runIBP? g ps) →
        InputsEnclosed g ps inputs →
          ∀ id < g.nodes.size,
            ∀ (B : NN.MLTheory.CROWN.FlatBox ℝ)
            (v : Val),
              (g.runIBP ps)[id]! = some B →
              (evalGraphRec g ps inputs)[id]! = some v →
              EnclosesBox B v
```

The theorem assumes that the graph is topologically sorted, its operations belong to `EngineCore`,
the proof-side pass covers every node, and the input boxes enclose the inputs. Under those
assumptions, a box returned by the engine encloses a value returned by the evaluator at the same
node. It does not by itself establish successful evaluation at every node.

The operation restriction matters for our example. `EngineCore` admits inputs, constants, detach,
addition, subtraction, elementwise multiplication, ReLU, linear layers, matrix multiplication,
concatenation, convolution, softplus, and safe log.
It excludes both `sum` and `tanh`, which occur at nodes 4 and 5. Consequently this theorem does not
cover the six-node graph as written, even after changing its scalar type to `ℝ`.
The separate `runIBP_encloses_all` theorem covers these operations and every other operation kind
under `RealNodeEquation` and the lawful arithmetic requirements described in
{ref "verification"}[Verification And Certificates]. The reals and rounded-real `FP32` satisfy
its scalar requirements; that result does not supply a global law for the FloatLib IEEE carrier.

The scalar in the theorem is `ParamStore ℝ`, whereas the command above uses FloatLib binary32.
Relating that rounded propagation to real arithmetic requires another argument, developed in
{ref "verification"}[Verification And Certificates] and
{ref "fp32-soundness"}[Floating-Point Soundness]. Applying the enclosure theorem to an execution
requires both its graph, coverage, and input hypotheses and this connection between scalar
semantics.

`IBPCovers` requires a box for every node in the proof-side pass. `InputsEnclosed` supplies the
initial enclosure from which the propagation proof starts. The sampled checker only compares an
already supplied point and box; it performs neither this induction nor a check of universal input
hypotheses.

# Executable Coverage And Proof Coverage

`Graph.denoteAll` is defined for every operation kind, so there is nothing to lower before a graph
runs: the value table shown earlier is the execution. It returns an error on a shape or axis that
its operation does not accept, and on a raw logarithm of a nonpositive input, rather than a NaN.
A custom node runs only when the payload supplies its checked body; recognizing its tag is not
enough to execute it.

The proof-bearing source lowering under `NN.Verification.Builtin.Proved` relates its first-order
source evaluator to IR denotation, including errors. Its theorem is not a wildcard over every
`TorchLean.Runtime.Program` or every `OpKind` the broad lowering accepts.

The equality theorems prove semantic preservation, including error results. They do not prove an
asymptotic complexity bound. Native graph evaluation still uses recursion and depends on stack
capacity.

# Axis Operations And Lowering Coverage

An axis is stored as data, but changing it changes which coordinates an operation combines. The
following example uses an interior axis so the shape rules must account for dimensions on both
sides:

```terminal
# Exercise nontrivial axes so a last-axis-only
# implementation cannot pass unnoticed.
scripts/lake.sh exe torchlean ir_axis_ops
```

```terminal +output
== IR axis ops tutorial ==
The IR evaluator validates each graph and computes its selected output on CPU.

-- softmax axis=1 on shape [2,3,4]
[softmax_middle_axis] output shape: [2, 3, 4]

-- layernorm axis=1 on shape [2,3,4]
PyTorch meaning: normalized_shape = x.shape[axis:] = [3,4]
[layernorm_middle_axis] output shape: [2, 3, 4]

-- concat axis=1: [2,3,4] ++ [2,5,4] -> [2,8,4]
[concat_middle_axis] output shape: [2, 8, 4]
```

The full transcript also prints the leading scalars. The example checks the graph's shapes with
`Graph.checkShapes`, then calls `Graph.denote`, which checks the input shape and returns the
selected output. The tutorial supplies the graphs and prints the results;
validation and execution live in the runtime library. This is the Lean tensor evaluator, not
LibTorch dispatch. Choose the scalar with `--arithmetic native`, `ieee`, or `complex`; this command
does not accept device or backend-reporting flags.

Change the graph's `concat (axis := 1)` to an out-of-range axis in
{src "NN/Examples/DeepDives/IRAxisOps.lean"}[`IRAxisOps.lean`] and shape inference rejects the node
before evaluation, with the same kind of error message the broken graphs produced earlier.

On `[2,3,4]`, softmax along axis one normalizes three entries while holding the first and last
coordinates fixed. LayerNorm starting at that axis combines twelve entries into each row's
statistics. Concatenation instead changes one extent and retains the others. These examples
exercise choices that a last-axis-only test would miss.

Changing the normalization axis can change values without changing the output shape. Shape
soundness alone cannot establish that the intended axis was used; semantic preservation must
identify the values too.

# Kernel Planning

The backend adapter maps each operation tag to a backend operation, chooses an admissible kernel
capsule for every runtime-relevant node, and preserves node identity so the plan can be audited
against the graph. On our graph, with the default CPU policy and the maintained capsule modules:

```lean (name := irPlan)
-- Inspect capsule selection for the executable nodes of
-- this graph.
#eval (NN.Backend.IR.checkedPlanGraph {} {}
    (NN.Backend.Registry.flatten
      NN.Backend.Registry.maintainedModules)
    graph).map fun plan =>
      (plan.nodeIds, plan.capsuleNames)
```

```leanOutput irPlan (whitespace := lax)
Except.ok (#[1, 2, 3, 4, 5],
  #["reference.linear", "reference.relu",
    "reference.linear", "reference.reduce_sum",
    "reference.tanh"])
```

Five capsules for six nodes. Node 0 is absent because `.input` needs no runtime work: `op?` maps it
to `none`, as it does for `.const` and `.detach`. Notice also that the `.sum` tag was planned as
`reference.reduce_sum`, because the backend vocabulary has one reduction operation where the IR has
both a full sum and an axis reduction.

A `KernelCapsule` describes the selected contract. It contains no executable closure, so planning
node 1 for `libtorch` does not call ATen. Eager execution must bind the selected capsule to a typed
handler with the same operation, provider, and device before that handler can run.
{ref "backend-selection"}[Backend Selection] follows the selection and assurance model in detail.

The distinction prevents a common architecture mistake:

```
-- Each stage below answers a different question about
-- backend support.
registered     ≠ selectable
selectable     ≠ executable
executable     ≠ proved correct
```

Moving between these states requires availability checks, runtime dispatch, and evidence for the
selected implementation. The capsule's numerical policy also determines whether an arithmetic
bound for a particular reduction order applies to the proposed kernel.

# Canonical IR And Autograd Tapes

The canonical IR records a persistent model computation. An eager autograd tape records one
execution, and carries what an execution has:

- concrete runtime tensor handles;
- which values require gradients;
- saved forward values needed by VJPs;
- the actual order in which wrappers ran.

The tape may contain enough information to reconstruct an IR-like graph, but it is not
`NN.IR.Graph`. Conversely, the canonical IR does not own mutable gradient buffers or optimizer
state. {ref "runtime-autograd"}[Runtime and Autograd] builds a tape by hand and shows exactly what
those four bullet points look like as data.

This difference is also why LibTorch can compute values in a TorchLean-owned backward path. The
TorchLean wrapper records the local tape node and chooses its VJP; ATen executes the requested
tensor operations with its autograd recording disabled. The canonical semantic graph and the
execution tape remain distinct objects connected by the operation contract.

A forward value table is also insufficient to reconstruct a proved reverse pass on its own. A
reverse computation needs the selected VJP for every operation, the saved inputs that rule reads,
and a consistent accumulation of cotangents at shared parents. Two tapes may agree on every
forward value while disagreeing on one of those backward choices. The forward lowering theorem is
valuable because it fixes the denotation being differentiated; the autograd link must additionally
connect the reverse rules to derivatives of that denotation.

# Graph Results And Validation Evidence

For any graph-based claim, locate:

1. the exact graph representation;
2. its concrete parameter payload;
3. the structural and shape checks that ran;
4. the scalar context used by denotation;
5. the lowering theorem and fragment, if lowering was used;
6. the capsule selected for each native operation;
7. the provider branch that actually executed it.

These records distinguish a shape-checking failure from a payload mismatch, a lowering restriction,
or an unavailable runtime provider.

The source map is:

- {src "NN/IR/Graph.lean"}[`Graph.lean`] for nodes and operation tags;
- {src "NN/IR/Infer.lean"}[`Infer.lean`] and {src "NN/IR/Check.lean"}[`Check.lean`] for shape
  inference and the two validation levels;
- {src "NN/IR/Semantics.lean"}[`Semantics.lean`] for pure denotation;
- {src "NN/IR/ShapeSoundness.lean"}[`ShapeSoundness.lean`] for the proof that validation and
  denotation agree on shapes;
- {src "NN/IR/Pretty.lean"}[`Pretty.lean`] for the listing and the GraphViz output;
- {src "NN/Backend/IR.lean"}[`Backend/IR.lean`] for capsule planning;
- {src "NN/Examples/DeepDives/OneSemanticUniverse.lean"}[`OneSemanticUniverse.lean`] for the
  complete six-node experiment.

# Variations On The Six-Node Graph

Changing one field at a time isolates the checks and interpretations used above.

1. Add a seventh node, `.relu` on node 5, and predict which of the two checkers complains before
   running either. Then give it output shape `[3]` and predict again.
2. Swap the two `.linear` payload keys so node 1 gets node 3's weights. Predict which of the six
   evaluation steps fails and with which message.
3. Change the input box radius from `0.05` to `0.5` and look at node 2. At what radius does the
   interval for zero-based coordinate `1` stop being `[0, 0]`?
4. Replace the final `.tanh` with `.sigmoid` and rerun the interval block. Explain the output box
   from the instance definitions rather than from the numbers.
5. Replace it with `.abs` instead, which has a real interval rule at FloatLib binary32, and see the
   output interval become informative again.
6. Add a `.log` node after the final `.tanh` and evaluate the graph. Find the input coordinates
   that make the evaluation fail, and read the message.
7. Plan the graph with a policy asking for a provider this build does not have, and read the error.
   It comes from the backend boundary, not from the graph.
