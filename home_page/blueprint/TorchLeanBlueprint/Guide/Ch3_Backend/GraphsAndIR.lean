import VersoManual
import TorchLean.NN.IR.Pretty
import TorchLean.NN.Backend.Graph
import TorchLean.Verify.CROWN.BoundOps.FP32
import TorchLean.Verify.CROWN.Op.Directed
import TorchLeanExamples.DeepDives.GraphSemantics
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.Roles
import TorchLean.NN.Loss
import TorchLean.NN.Session.Random
import TorchLean.Verify.Trainer

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean
open TorchLean
open IR
open CROWN (NonlinearBoundOps)
open Examples.GraphSemantics
open Lean.Elab.Tactic.GuardMsgs.WhitespaceMode (lax)

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)

#doc (Manual) "Graph IR" =>
%%%
tag := "graphs-and-ir"
file := "The-Canonical-Graph-IR"
%%%

A sequential model preserves architecture and shape constraints in its type. Importers, exporters,
and verification passes need something else: a first-order representation that can be read before it
is known to be valid. `IR.Graph` is that representation, an array of nodes of registered
operators that can be traversed, serialized, checked, and assigned to kernels.

I'll keep the ten-node graph from
{src "TorchLeanExamples/DeepDives/GraphSemantics.lean"}[`GraphSemantics.lean`] fixed as we
change its parameter payload and scalar interpretation. We can then compare the computed values,
propagated intervals, and kernel plan for the same nodes. A PyTorch trace provides a second view
of their connections. Every Lean
block below is elaborated while this page is built. The Python examples and shell commands let
you inspect the graph and compare its interpretations in your build.

# Executable Graph Types And Lowering

`TypedGraph` stores shape-indexed executable functions for forward evaluation, JVPs, and
VJPs, together with a typed reference selecting its output. The canonical IR has no second
executable form: `Graph.eval` runs the graph itself, by the rules of its registered operators, and
computes forward values only. {ref "runtime-autograd"}[Runtime and Autograd] explains what the
typed graph can justify.

Here are the principal executable paths:

```
-- Follow the executable export route separately from its
-- semantic evidence.
Sequential σ τ ── Sequential.forward ──────> TorchLean.Program
                                             │
                                             ├── Autodiff.lowerToTypedGraph
                                             │   ──> TorchLean.TypedGraph
                                             │        (forward + JVP + VJP)
                                             │
                                             └── Verification.lowerForwardToIR
                                                 ──> TorchLean.IR.Graph
                                                       │
                                                       └── Graph.eval
                                                            (forward only)

TorchLean.IR.Graph ── kernel selection ──> GraphKernelPlan
                                        (metadata only)
```

The diagram separates forward execution from execution with derivative rules. Its `TypedGraph`
branch records forward, JVP, and VJP operations, while the IR branch is evaluated by its operators'
rules. Kernel selection produces metadata that must later be connected to runtime implementations.

`Program` is an operation-polymorphic function, not a stored graph. A broad pass,
`IR.lowerForward`, executes that function with a recording interpreter that builds a typed graph
(`IR.OpGraph`), then builds and checks its operator graph. We prove that the lowered graph runs
as that typed graph (`IR.lowerForward_eq_ok`), but no theorem relates the graph to the source
program. Stable log-softmax lowers to the IR's own `logSoftmax`, the composition the source
computes. GELU returns a lowering error, not an expansion with a different rounding schedule,
though the eager runtime supports GELU.

# MLP Graph Structure

Our running example is

$$`
G(x)=\tanh\!\left(
  \sum_k
  \left[W_2\operatorname{ReLU}(W_1x+b_1)+b_2\right]_k
\right),
`

with $`x\in\mathbb{R}^4`, a hidden width of five, and an output width of three. $`W_1x+b_1`
produces five hidden preactivations, and ReLU keeps their shape. The second affine layer produces
three coordinates, the sum reduces them to a scalar, and `tanh` bounds that scalar. That's external
parameters, a piecewise-linear activation, a rank-changing reduction, and a nonlinear interval
rule, all in one small graph.

TorchLean stores it as ten nodes: the input, the four parameter tensors as constant nodes, and the
five operators. Since the graph is ordinary Lean data, we can just print it:

```lean (name := irPretty)
-- Print node ids, the references each operator reads, and
-- the ports each node produces.
#eval do
  let g ← IO.ofExcept graph
  for (n, id) in g.nodes.zipIdx do
    IO.println (n.prettyLine id)
```

```leanOutput irPretty (whitespace := lax)
0: input 0 out=[4]
1: const out=[5, 4]
2: const out=[5]
3: linear inputs=[1, 2, 0] out=[5]
4: relu inputs=[3] out=[5]
5: const out=[3, 5]
6: const out=[3]
7: linear inputs=[5, 6, 4] out=[3]
8: reduce inputs=[7] out=[]
9: tanh inputs=[8] out=[]
```

An operator lists the references it reads, one per input port in port order: node 3 reads the
weight at node 1, the bias at node 2 and the input at node 0, the order of `Op.linear`'s ports. A
reference names a node and one of its outputs, and the listing shows the output's position only
when it isn't the first. Every reference names an earlier node, so array order is already a
topological execution order, and validation and evaluation can scan left to right without a graph
search. Each operator prints the tag of its wire record (`Op.encode`), the record the
`torchlean.ir` artifact writes with the operator's fields. The printed line leaves those fields out
(here, the summed axis of `reduce`). Constant nodes hold no tensors: their values come from a
payload keyed by the node id, which gets its own section later.

That listing renders the stored node array. Here is the source that produced it, quoted
from the example file:

```
-- The node array gives a topological program; parameter
-- values remain in a separate payload.
def graph : Except String IR.Graph :=
  let (b, input) := ({} : IR.Graph.Builder).input 0 (.value [4])
  let (b, hiddenWeight) := b.const (.value [5, 4])
  let (b, hiddenBias) := b.const (.value [5])
  let (b, layer1) := b.op (Op.linear (lead := []) (inDim := 4) (outDim := 5))
    [hiddenWeight, hiddenBias, input]
  let (b, activation) := b.op (Op.relu (s := [5])) layer1
  let (b, outputWeight) := b.const (.value [3, 5])
  let (b, outputBias) := b.const (.value [3])
  let (b, output) := b.op (Op.linear (lead := []) (inDim := 5) (outDim := 3))
    ([outputWeight, outputBias] ++ activation)
  let (b, total) := b.op (Op.reduce (s := [3]) .sum [⟨0, by decide⟩] false
    (by decide)) output
  let (b, result) := b.op (Op.tanh (s := [])) total
  b.build result.toArray
```

`input`, `const` and `op` append a node and return references to its outputs (one `Ref` for an
input or a constant, a list for an operator), so `relu` reads `layer1` as it is and the second
`linear` reads the activation after its weight and bias. `build` returns the graph only if the
structural check `Graph.check` accepts it, which is why `graph` is an `Except`.

Unlike a sequential model, this datatype doesn't make every edge shape-correct by construction. An
operator's ports are part of its type (`TorchLean.Op ins outs`), so the ports an operator node
produces follow from its operator (`Node.outs`) and need no declaration. References, though, are
ordinary data: nothing in the type says that one names an earlier node, or an output at the port
its input reads. We made that choice on purpose, because an importer must be able to construct a
candidate graph from an external document (a `torchlean.ir` artifact, say) before Lean knows it's
valid. The representation therefore needs a separate validation phase before its references can be
trusted.

There is also a GraphViz rendering for when ten nodes become a hundred:

```lean (name := irDot)
-- Render the same references as Graphviz input.
#eval do IO.println (← IO.ofExcept graph).toDot
```

```leanOutput irDot (whitespace := lax)
digraph IR {
  rankdir=LR;
  node [shape=box, fontsize=10];
  n0 [label="0: input\n[4]"];
  n1 [label="1: const\n[5, 4]"];
  n2 [label="2: const\n[5]"];
  n3 [label="3: linear\n[5]"];
  n4 [label="4: relu\n[5]"];
  n5 [label="5: const\n[3, 5]"];
  n6 [label="6: const\n[3]"];
  n7 [label="7: linear\n[3]"];
  n8 [label="8: reduce\n[]"];
  n9 [label="9: tanh\n[]", peripheries=2];
  n1 -> n3;
  n2 -> n3;
  n0 -> n3;
  n3 -> n4;
  n5 -> n7;
  n6 -> n7;
  n4 -> n7;
  n7 -> n8;
  n8 -> n9;
}
```

Write that to a file and run `dot -Tpng g.dot -o g.png`. Edges point from producer to consumer,
matching the dataflow picture the frameworks use, and the double border marks node 9, the graph's
output. Neither printer is a serialization format, as
{src "TorchLean/NN/IR/Pretty.lean"}[`Pretty.lean`] says in its module docstring:
`TorchLean.NN.IR.Artifact` writes the `torchlean.ir` artifact, and `Artifact.ofJson_toJson`
proves that decoding it returns the artifact. The pretty output carries no such obligation, so it
can change with the IR.

On the last two nodes, the port `[]` denotes a scalar, with one value. When you inspect a larger
graph, follow the edges as well as the shapes: an edge that bypasses ReLU may still connect
compatible shapes while changing the computation.

# PyTorch Graph Capture

PyTorch's op-tagged graph exposes the same computation through a different node interface.
Write the network with `nn.Linear`, copy in the same weights,
and trace it {Informal.citep fx2022}[]:

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
with the ten-node listing above. Keep the output value for the numerical comparison below.

An FX node's `target` is a Python object: a builtin, a submodule path, a method name, or any
function you happened to call. TorchLean's operator is a constructor of a single inductive type
(the operator table `TorchLean.Op`), so a pass must handle every case. A wildcard branch can still
accept or reject many cases together, so compilation alone doesn't prove each operation has the
intended support.

An FX node has no shape field in its core record. Shape information can be attached as metadata,
for example by propagating a concrete input (other capture paths can use symbolic or fake-tensor
metadata):

```
# Run shape propagation to see which metadata comes from
# executing example inputs.
from torch.fx.passes.shape_prop import ShapeProp
print("before:", {n.name: sorted(n.meta.keys()) for n in g.graph.nodes})
ShapeProp(g).propagate(x0)
print("after: ", {n.name: tuple(n.meta["tensor_meta"].shape)
                  for n in g.graph.nodes if "tensor_meta" in n.meta})
```

You should see feature widths of four at the input, five after the first linear layer and ReLU, and
three after the second linear layer, while the sum and `tanh` return scalars. Shape propagation
records what happened on the supplied input by executing the graph. TorchLean's ports are part of
its operators and are checked against the references before execution, with theorems connecting
that check and those ports to the semantics.

`call_module l1` means "go find the submodule named `l1` and call it", so the parameters are inside
the traced object. TorchLean keeps the graph free of tensors. Each parameter is a constant node,
and a payload passed separately supplies its value, which is what lets us run the same graph with
interval parameters later on.

FX adds a separate `output` node, so its listing has seven nodes, and it keeps the parameters
inside its two `call_module` nodes, where TorchLean has four constant nodes. TorchLean names the
graph's outputs by reference in `Graph.outputs`, here output 0 of node 9, and `Graph.eval` returns
their values. That extra FX node selects the result and adds no tensor calculation.

We can compare the printed PyTorch output at `x0` with the IR evaluator after supplying the
parameter payload.

# The Node Record

Validation and evaluation use the information in the node record:

```
/-- Output `out` of node `node`. -/
structure Ref where
  /-- The node, by its position. -/
  node : Nat
  /-- The output position. -/
  out : Nat := 0
  deriving DecidableEq, Repr, Inhabited

/-- A node: a graph input, a constant of the payload, or a registered operator with one reference
per input port, in port order; for `Op.custom` the payload holds the checked program. -/
inductive Node where
  /-- Graph input `index`, read at `port`. -/
  | input (index : Nat) (port : Port)
  /-- The payload's constant at the node's id, read at `port`. -/
  | const (port : Port)
  /-- Operator `op` on the outputs `inputs` name. -/
  | op (op : Operator) (inputs : Array Ref)
  deriving DecidableEq, Repr
```

A port (`Port`) is an element sort and a shape: values of the scalar, frozen or detached values, or
index, Boolean or integer tensors. An operator (`Operator`, a `TorchLean.Op ins outs` with its
ports forgotten in the type) carries operation-local data such as an axis, a permutation,
convolution geometry, or the shapes of its ports. Large tensor values don't belong there. Constants
and learned parameters are constant nodes, and their values live in the payload. A node has no id
field, since its id is its position in `nodes`, and an operator node declares no output port, since
`Node.outs` computes its ports from the operator. To evaluate the graph safely, each operator needs
one reference per input port, each reference must name an existing output of an earlier node at a
port its input accepts, and the operator's fields must pass `Op.check`. Those are checks on the
graph as a whole, and the operator alone doesn't establish them.

MLIR separates operation data from validation in the same way {Informal.citep mlir2021}[], and
ONNX similarly stores initializers apart from operations. Here we can inspect both the checks and
the Lean theorems connecting them to the graph semantics.

# Structural And Shape Validation

Validation is one structural check. `Graph.check` checks each node in order (`Node.check`): an
operator has one reference per input port, each naming an existing output of an earlier node at a
port its input accepts, and its fields pass `Op.check`. Then every graph output must name an
existing output. A port carries its shape, so the same check compares the shapes at both ends of
every edge. `build` ran it on our graph, and running it again gives the same answer:

```lean (name := irChecks)
-- Check every reference against the ports of the node it
-- names.
#eval graph >>= Graph.check
```

```leanOutput irChecks (whitespace := lax)
Except.ok ()
```

Let's break it three ways, one field at a time. First, give the ReLU node the wrong shape, leaving
its reference untouched:

```lean (name := irBadShape)
-- Keep the edges valid while changing only the shape the
-- operator reads.
/-- The same graph, with the ReLU at node 4 reading `[4]`
not `[5]`. -/
def irBadShape : Except String Graph := do
  let g ← graph
  let relu : Node :=
    .op ⟨_, _, TorchLean.Op.relu (s := [4])⟩ #[⟨3, 0⟩]
  pure { g with nodes := g.nodes.set! 4 relu }

#eval irBadShape >>= Graph.check
```

```leanOutput irBadShape (whitespace := lax)
Except.error "IR check: node 4: output 0 of node 3 has
  port { kind := TorchLean.Port.Kind.value, shape := [5] },
  which input port { kind := TorchLean.Port.Kind.value,
  shape := [4] } does not accept"
```

In the broken graph, the reference still points to an earlier node, but ReLU at `[4]` reads a port
of shape `[4]`, and node 3 produces `[5]`. The check reports that mismatch, with both ports, before
any value is computed. Evaluation also checks every input against its port (`Op.invoke`), so an
unchecked copy of this graph fails there too, at the same node.

Second, break the topological order by pointing node 4 at node 7:

```lean (name := irCycle)
-- A reference to a later node violates the order required
-- by the evaluator.
/-- Node 4 reads node 7, which comes after it. -/
def irCycle : Except String Graph := do
  let g ← graph
  let relu : Node :=
    .op ⟨_, _, TorchLean.Op.relu (s := [5])⟩ #[⟨7, 0⟩]
  pure { g with nodes := g.nodes.set! 4 relu }

#eval irCycle >>= Graph.check
```

```leanOutput irCycle (whitespace := lax)
Except.error "IR check: node 4: no earlier output 0 of
  node 7"
```

Third, name an output that doesn't exist. A graph's outputs are references too, and node 9 has one
output, at position `0`:

```lean (name := irBadOutput)
-- An output reference must name an existing output.
/-- The graph's output is output 1 of node 9. -/
def irBadOutput : Except String Graph := do
  let g ← graph
  pure { g with outputs := #[⟨9, 1⟩] }

#eval irBadOutput >>= Graph.check
```

```leanOutput irBadOutput (whitespace := lax)
Except.error "IR check: no output 1 of node 9"
```

Each failed check returns an `Except.error` naming the node and the reference, so an importer can
report the problem. A shape change cannot repair a forward reference, and reordering cannot
repair a port mismatch.

## Relating Shape Checking To Evaluation

Checking and evaluation are separate functions, and
{src "TorchLean/NN/IR/Eval.lean"}[`Eval.lean`] states what the check accepts:

```lean (name := irSound)
-- The check succeeds exactly on the well-formed graphs.
#check @IR.Graph.check_ok_iff
```

```leanOutput irSound (whitespace := lax)
@Graph.check_ok_iff : ∀ {g : Graph},
  g.check = Except.ok () ↔ g.WellFormed
```

`Graph.WellFormed` is the proposition the check decides: every node is well formed with the outputs
of the nodes before it (`Node.WellFormed`: one reference per input port, each naming an existing
output at a port its input accepts, and `Op.check`), and every graph output names an existing
output. Ports relate to evaluation through the evaluator's own theorems. `Node.eval_ports` says
that every output of a successful node is at the port the node declares, and `Graph.run_ports` that
entry `j` of a successful run is at the ports of node `j`. Neither assumes the check, because each
operator's rule produces its values at its ports. We can watch that happen:

```lean (name := irShapesAgree)
-- These are the ports the graph's nodes produce.
#eval graph.map fun g => g.nodes.map Node.outsPretty
```

```leanOutput irShapesAgree (whitespace := lax)
Except.ok #["[4]", "[5, 4]", "[5]", "[5]", "[5]",
  "[3, 5]", "[3]", "[3]", "[]", "[]"]
```

```lean (name := irShapesAgree2)
-- These are the ports of the values actually returned by
-- its evaluator.
#eval (graph >>= fun g =>
    g.run (payload floatParameters)
      [⟨.value [4], center⟩]).map
  fun vals => vals.map fun vs => vs.map fun v => v.1.pretty
```

```leanOutput irShapesAgree2 (whitespace := lax)
Except.ok #[["[4]"], ["[5, 4]"], ["[5]"], ["[5]"],
  ["[5]"], ["[3, 5]"], ["[3]"], ["[3]"], ["[]"], ["[]"]]
```

`IR.Graph.plan`, in the backend adapter, relies on the same check: it runs `Graph.check`
before it selects a kernel for any node, so a successful plan implies that the check passed.

Payloads, too, are left to the evaluator. `Graph.check` reads no payload, so it can't supply a
missing weight tensor, or ensure that a tensor stored under a node id is at the port that node
declares.

# Parameter Payloads

Both `linear` nodes read their weights and biases through references to the constant nodes 1, 2, 5
and 6. A constant node declares only its port. Its value comes from a payload keyed by node id, and
here's the real definition from the example:

```
-- Resolve the four constant nodes to parameter tensors at
-- their ports.
def payload {α : Type} [Storage α] [Context α]
    (parameters : Parameters α) : IR.Payload α :=
  { const? := fun
      | 1 => some ⟨.value [5, 4], parameters.hiddenWeight⟩
      | 2 => some ⟨.value [5], parameters.hiddenBias⟩
      | 5 => some ⟨.value [3, 5], parameters.outputWeight⟩
      | 6 => some ⟨.value [3], parameters.outputBias⟩
      | _ => none }
```

The scalar type parameter and the `Storage` and `Context` instances determine the payload's tensor
values. Node ids stay fixed when we change this scalar interpretation, so the same graph and
payload later give interval propagation its parameters (as boxes of width zero).

Separating structure from values has practical consequences:

- one graph can be reused with initial, trained, or bounded parameters
- checkpoint loading changes the payload without rebuilding the node array
- a verifier can replace concrete parameters by interval metadata
- an exporter can emit graph nodes and initializers through different channels

But separating them also creates an obligation that no type will discharge for you. A payload entry
is a value with its own port (`TorchLean.Op.Value α`), and its tensor is shape-correct with respect
to *that* port, not with respect to the node. So a payload can be internally consistent and still
disagree with the graph. Keying by node id makes that failure possible, and here it is:

```lean (name := irBadPayload)
-- A present payload key is insufficient when the matrix
-- dimensions disagree.
/-- A payload whose key exists, whose weight is a perfectly
good matrix, and whose dimensions do not match node 1. -/
def irBadPayload : Payload Float :=
  { const? := fun id =>
      if id = 1 then
        some ⟨.value [3, 4], Tensor.full [3, 4] 0.1⟩
      else (payload floatParameters).const? id }

#eval (graph >>= fun g =>
    g.eval irBadPayload [⟨.value [4], center⟩]).map
  fun outputs => outputs.length
```

```leanOutput irBadPayload (whitespace := lax)
Except.error "op: an input of port { kind :=
  TorchLean.Port.Kind.value, shape := [3, 4] } where { kind :=
  TorchLean.Port.Kind.value, shape := [5, 4] } is expected"
```

Looking up key `1` succeeds, but its three-row matrix isn't at node 1's port `[5, 4]`. Evaluation
stops at node 1, where the constant node reads its payload value at its port (`Port.read`), and
reports both ports.

`IR.Payload` has two fields: the constant of each constant node (a value, index or Boolean
tensor at its port: parameters, masks, index tensors) and the checked program of each `Op.custom`
node. Every operator reads its operands through references, the parameters of `linear` as much as
its input, so supporting another operator needs no change to the payload type: its parameters are
more constant nodes.

Even equal dimensions wouldn't prove that the correct checkpoint tensor had been assigned to a
node: the bias-swap example of {ref "sequential"}[the sequential models chapter] passed that check
while changing the output.

# A Heterogeneous Value Table

During evaluation, node 0 holds a vector of four, node 1 a five-by-four matrix, nodes 3 and 4
vectors of five, node 7 a vector of three, and nodes 8 and 9 scalars. One homogeneous Lean array
cannot directly contain all those tensor types, so the evaluator stores each output with its port:

```
-- The dependent pair retains a tensor together with the
-- port indexing its type.
abbrev Op.Value (α : Type) [Storage α] := Σ p : Port, p.Carrier α
```

Each stored value is a dependent pair of a port and a tensor of that port's own carrier: a tensor
of the scalar at a value, frozen or detached port, and of indices, Booleans or machine integers at
the others. `Port.read` recovers a tensor at a given port only after checking the stored one. Each
node's entry is the list of its outputs, so an operator with several outputs (`sort`, `topk`) is
one node. Model code never builds this table: its inputs stay `Tensor α shape`.

For node 3, evaluation performs:

1. check the operator's fields (`Op.check`)
2. read the outputs its references name from the value table: nodes 1, 2 and 0
3. check each against its input port, `[5, 4]`, `[5]` and `[4]`
4. check the rule's value domain, which `linear` leaves unrestricted
5. call the rule's forward function
6. store its output, at port `[5]`

In the previous section, the wrong weight never reached step 3 because constant node 1 already
reads its payload value at its port. Failures are reported as `Except String`, and malformed
imported data doesn't get a fabricated proof cast.

We can read every intermediate value from this table to locate where two evaluations diverge:

```lean (name := irValues)
-- Inspect every intermediate, so the final scalar can be
-- traced through all ten nodes.
#eval (graph >>= fun g =>
    g.run (payload floatParameters)
      [⟨.value [4], center⟩]).map
  fun vals => vals.map fun outputs => outputs.map fun
    | ⟨⟨.value, _⟩, t⟩ => Tensor.pretty t
    | _ => "not a value"
```

```leanOutput irValues (whitespace := lax)
Except.ok #[["[0.300000, -0.200000, 0.100000, 0.400000]"],
  ["[[0.150000, -0.120000, 0.080000, 0.050000],
    [0.020000, 0.110000, -0.090000, 0.070000],
    [-0.040000, 0.060000, 0.100000, -0.030000],
    [0.090000, 0.010000, 0.040000, 0.130000],
    [-0.070000, 0.030000, 0.120000, -0.020000]]"],
  ["[0.010000, -0.020000, 0.030000, 0.000000, 0.020000]"],
  ["[0.107000, -0.017000, 0.004000, 0.081000, -0.003000]"],
  ["[0.107000, 0.000000, 0.004000, 0.081000, 0.000000]"],
  ["[[0.050000, 0.080000, -0.060000, 0.030000, 0.070000],
    [-0.040000, 0.020000, 0.090000, -0.010000, 0.060000],
    [0.100000, -0.030000, 0.040000, 0.050000, -0.080000]]"],
  ["[0.020000, -0.010000, 0.000000]"],
  ["[0.027540, -0.014730, 0.014910]"], ["0.027720"],
  ["0.027713"]]
```

In the table, constant nodes hold the parameters as the payload gives them. Node 3 is the affine
layer, node 4 is the same vector with its two negative entries clamped to zero, node 7 is the
second affine layer, node 8 sums the three components to `0.027720`, and node 9 applies `tanh` to
get `0.027713`. Applying tanh barely moves the value, because `tanh x ≈ x` near zero. That gives us
a reference for the interval calculation: at the center, the reduction and the final output are
both close to `0.028`.

# Scalar Interpretations

`Graph.eval` folds over the node array using the operators' rules at a scalar `Context α`. Nothing
in `IR.Graph` mentions `α` at all, which is why one piece of graph data can be read at:

- `α := ℝ`, for exact-real theorem statements
- `α := FP32`, for finite rounded-real analysis
- `α := ExecFloat.Binary 8 23`, for executable binary32 behavior
- another valid FloatLib binary format, for typed CPU execution at a different precision
- `α := Float`, for the ordinary runtime scalar
- interval endpoints over any of those, for bound propagation

Across those choices the graph is the same data, and the meaning of arithmetic changes with `α`.
It's the same tagless-final move the runtime uses for its `Ops` instances, the eager tape and the
typed graph {Informal.citep kiselyov2012}[], except that here the interpreter is a fold over
stored data instead of an instance chosen by unification.

We can compare two executable choices side by side:

```lean (name := irFloat)
-- Evaluate the graph using the host Float scalar
-- interpretation.
#eval (evaluate (α := Float) floatParameters
  center).map Tensor.pretty
```

```leanOutput irFloat (whitespace := lax)
Except.ok "0.027713"
```

```lean (name := irIEEE)
-- Convert the same parameters to the executable binary32
-- interpretation.
def irIEEEParams : Parameters (ExecFloat.Binary 8 23) :=
  floatParameters.map ofFloat

def irCenter : Tensor (ExecFloat.Binary 8 23) [4] :=
  Tensor.map ofFloat center

#eval (evaluate (α := (ExecFloat.Binary 8 23))
  irIEEEParams irCenter).map fun output =>
    Tensor.pretty (Tensor.map
      (Float32.toFloat ∘ ExecFloat.Binary.toFloat32)
      output)
```

```leanOutput irIEEE (whitespace := lax)
Except.ok "0.027713"
```

The two checked Lean evaluations agree to six decimals. The first uses hardware `Float`
double-precision arithmetic, and the second uses FloatLib binary32 with its rounding spelled out
{Informal.citep goldberg1991}[]. Its final conversion picks a decimal display without changing
the arithmetic inside the graph. Compare these values with the PyTorch output from the command
above. Agreement at the displayed precision doesn't establish agreement of stored values, and
{ref "floats"}[Floating-Point Semantics] follows individual rounding steps to explain how such
differences arise.

The other two instantiations are noncomputable, which is a fact about their scalars and not about
the graph:

```lean (name := irNoncomputable)
-- These functions return optional boxes, so a type alone
-- does not promise a bound at every node.
#check CROWN.Op.runIBP (α := ℝ)

#check CROWN.Op.runIBP (α := FP32)
```

```leanOutput irNoncomputable (whitespace := lax)
CROWN.Op.runIBP : Graph → Payload ℝ →
  (ℕ → Option (CROWN.FlatBox ℝ)) →
    Array (Option (CROWN.FlatBox ℝ))
```

```leanOutput irNoncomputable (whitespace := lax)
CROWN.Op.runIBP : Graph → Payload FP32 →
  (ℕ → Option (CROWN.FlatBox FP32)) →
    Array (Option (CROWN.FlatBox FP32))
```

Both instantiations use the same graph, payload and propagation algorithm to construct boxes for
mathematical reasoning. `ℝ` is Mathlib's real numbers {Informal.citep mathlib2020}[]. `FP32`
models binary32 precision and gradual underflow using rounded real values (with no upper exponent
cutoff). Flocq provides a related formal account of floating-point rounding
{Informal.citep flocq2011}[]. Relating different scalar interpretations requires hypotheses of
its own, and {ref "spec-layer"}[The Specification Layer] explains those relationships.

In the return types, the `Option` entries allow a node to lack a derived bound: even when the graph
evaluates successfully, propagation may leave part of the enclosure trace unavailable.

# Interval Bound Propagation

Bound propagation replaces each value by an interval and each operation by an operation on
intervals, then reads the output interval. That's interval bound propagation, the simplest member
of the family that CROWN and its descendants refine ({Informal.citep gowal2018}[],
{Informal.citep crown2018}[]). It runs on our graph unchanged, since the node array doesn't know
it's being interpreted over boxes.

Take the center input $`x_0=(0.3,-0.2,0.1,0.4)` and an $`\ell_\infty` ball of radius `0.05` around
it, then print the box at every node:

```lean (name := irIBP)
-- Propagate the input box and display where the interval
-- becomes less precise.
def irBox := inputBox (α := (ExecFloat.Binary 8 23))
  (eps := 0.05)

def irFlatBox : CROWN.FlatBox
    (ExecFloat.Binary 8 23) :=
  { dim := 4, lo := irBox.lo, hi := irBox.hi }

open CROWN in
def irBoxes := graph.map fun g =>
  Op.runIBP g (payload irIEEEParams)
    (Op.inputSeeds g [irFlatBox])

-- The constant nodes 1, 2, 5 and 6 hold the point boxes of
-- the parameters. Print the other nodes.
#eval IO.println <| String.intercalate "\n"
  ([0, 3, 4, 7, 8, 9].map fun id =>
    match irBoxes.toOption.bind (·[id]?) with
    | some (some b) =>
        let display :=
          Float32.toFloat ∘ ExecFloat.Binary.toFloat32
        s!"node {id}: {Tensor.pretty
          (Tensor.map display b.lo)} .. " ++
          s!"{Tensor.pretty (Tensor.map display b.hi)}"
    | _ => s!"node {id}: no box")
```

```leanOutput irIBP (whitespace := lax)
node 0: [0.250000, -0.250000, 0.050000, 0.350000] ..
  [0.350000, -0.150000, 0.150000, 0.450000]
node 3: [0.087000, -0.031500, -0.007500, 0.067500,
  -0.015000] .. [0.127000, -0.002500, 0.015500,
  0.094500, 0.009000]
node 4: [0.087000, 0.000000, 0.000000, 0.067500,
  0.000000] .. [0.127000, 0.000000, 0.015500,
  0.094500, 0.009000]
node 7: [0.025445, -0.016025, 0.011355] ..
  [0.029815, -0.012220, 0.018045]
node 8: [0.020775] .. [0.035640]
node 9: [-1.000000] .. [1.000000]
```

`Op.inputSeeds` gives each input node the box of its graph input, here node 0. Node 0 is the input
box (the center plus and minus `0.05` in each coordinate), and each constant node gets the point
box of its payload value. Node 3 applies the affine interval rule to those boxes and records one
lower and upper bound per output coordinate. These bounds include endpoint rounding, and the
printed decimals don't assert an exact image of the box. Compare node 4 with node 3: every negative
lower bound has become exactly `0.000000`, and the upper bounds are unchanged except where they
were negative too. That's interval ReLU, and zero-based coordinate `1` of node 4 is the degenerate
interval `[0, 0]` because the whole input interval for that coordinate was negative. Node 7 is the
second affine layer, node 8 sums the three coordinates, and the resulting scalar interval is
`[0.020775, 0.035640]`, which does contain the value `0.027720` we computed at the center.

Node 9 widens the result to `[-1, 1]`, a global enclosure of the range of `tanh`. To explain
that loss of
information, we need to inspect the interval rule selected for this scalar type.

An interval at a node encloses the values allowed by the input box and the transfer rules used so
far, and it can lose correlations between two parents that depend on the same input. Enclose a
value and its negation separately, for example, and you lose the fact that their sum is exactly
zero. That loss of dependence can widen later intervals even with exact endpoint arithmetic.
Rounded endpoints add another issue, because they must move outward to preserve enclosure. A wide
interval can therefore reflect the abstraction, the rounding policy, or a deliberately coarse
nonlinear rule, and those causes call for different improvements.

## The Executable Tanh Interval Rule

For each scalar type, the `NonlinearBoundOps` instance chooses the `tanh` interval rule. We can
query the executable binary32 instance on its own, using node 8's displayed endpoints:

```lean (name := irTanhBounds)
-- Ask the executable nonlinear policy for the tanh interval
-- directly.
#eval (NonlinearBoundOps.tanhBounds
    (α := (ExecFloat.Binary 8 23))
    (ofFloat 0.020775)
    (ofFloat 0.035640)).map
  fun (lo, hi) =>
    ((Float32.toFloat ∘ ExecFloat.Binary.toFloat32) lo,
      (Float32.toFloat ∘ ExecFloat.Binary.toFloat32) hi)
```

```leanOutput irTanhBounds (whitespace := lax)
some (-1.000000, 1.000000)
```

In {src "TorchLean/Verify/CROWN/BoundOps/Binary32.lean"}[the binary32 bound operations], the
executable instance returns the global codomain of `tanh` after checking that its input endpoints
are finite and ordered, so it discards their range. Specialized proof-oriented instances use that
range: over `ℝ` the rule is `some (Real.tanh lo, Real.tanh hi)`, and over `FP32` it's the same with
the endpoints rounded outward. Since `tanh` is increasing, the real endpoints are tight for that
scalar interval and the FP32 endpoints enclose them with outward rounding. Host `Float.tanh` shows
their approximate size, though this next evaluation isn't a directed-rounding enclosure:

```lean (name := irTanhTight)
-- Endpoint evaluations illustrate a tighter candidate,
-- without proving outward rounding.
#eval (Float.tanh 0.020775, Float.tanh 0.035640)
```

```leanOutput irTanhTight (whitespace := lax)
(0.020772, 0.035625)
```

For the displayed node-8 interval, the real endpoint image has width about `0.0149`, while the
executable codomain fallback has width `2.0`. A full rerun with another scalar type also changes
the rounding of the preceding nodes, so this calculation compares the final transfer on fixed
endpoints. The endpoint-sensitive bound is much more informative here.
`CROWN.Binary32.tanhBounds_sound` proves that the executable fallback contains ideal real
`tanh` whenever the finite transfer succeeds, but agreement with a separately rounded native `tanh`
evaluation requires a further backend theorem.

`FloatLib.Floats.Interval.Quantized` defines the generic real interval construction. Its
`RInterval.tanh` is noncomputable: it applies `Real.tanh` to each endpoint and rounds outward.
TorchLean's executable instance above uses the global codomain instead. A tighter executable
interval needs justified upper and lower bounds for the transcendental operation, while `[-1, 1]`
already encloses the ideal real `tanh` range. Relating executable transcendental outputs to that
range is a separate backend obligation. Nearest rounding at an endpoint can round inward (a lower
bound can become too large, or an upper bound too small), so a tight executable enclosure needs an
error bound or a directed-rounding implementation for the transcendental function
{Informal.citep boldo2015}[]. Using the global enclosure avoids that endpoint calculation, at the
cost of discarding the input interval.

`sigmoid` also uses its global codomain, `[0, 1]`, for finite ordered inputs. For `exp`, `log`,
`sqrt`, `sin`, and `cos`, the transfers use FloatLib's certified rational interval kernels followed
by checked outward binary32 rounding. Their theorems enclose the value at every real input in the
interval, and sine and cosine include interior extrema. Domain checks and finite-output checks can
still return `none`. Square root preserves intervals crossing zero by using the real convention
`Real.sqrt x = 0` for negative `x`. `layerNormAbsBound` remains unavailable. When an executable IBP
result looks uselessly wide, check which nonlinearity the graph ends with before suspecting the
propagation.

For a property that needs to distinguish this small positive output from zero, `[-1, 1]` is too
wide. It's the final transfer rule that loses the information, even though the preceding nodes
retained useful bounds.

# Containment And Enclosure Theorems

One command runs the whole experiment:

```terminal
# Evaluate this example and propagate its intervals.
scripts/lake.sh exe torchlean graph_semantics
```

The command evaluates the center input in FloatLib binary32 and propagates an output interval.

The command also names the pointwise theorem behind its Boolean containment check:

```lean (name := irContains)
-- The Boolean-to-proposition bridge concerns the point
-- supplied to containsDecBool.
open CROWN in
#check @Box.containsDecBool_sound
```

```leanOutput irContains (whitespace := lax)
@Box.containsDecBool_sound : ∀ {α : Type}
  [inst : Storage α] [inst_1 : Context α]
  [inst_2 : DecidableRel fun x1 x2 => x1 ≤ x2]
  {s : Shape} (b : Box α s) (x : Tensor α s),
  b.containsDecBool x = true → b.Contains x
```

The pointwise theorem upgrades a Boolean to a proposition about the point that was checked. It says
nothing about the other points in the box, and it isn't the enclosure theorem.

The enclosure theorem quantifies over real runs of the graph under stated payload and input
assumptions:

```lean (name := irEnclose)
-- The enclosure theorem relates the rounded boxes to a
-- real run of the same graph.
#check @CROWN.Op.Directed.encloses_of_run
```

```leanOutput irEnclose (whitespace := lax)
@CROWN.Op.Directed.encloses_of_run :
  ∀ {α : Type} [inst : Storage α] [inst_1 : Context α]
  [inst_2 : CROWN.BoundOps α]
  [inst_3 : CROWN.LawfulBoundOps α]
  [inst_4 : NonlinearBoundOps α]
  [CROWN.LawfulNonlinearBoundOps α]
  (G : Graph) (P : Payload α) (Pr : Payload ℝ)
  (seeds : ℕ → Option (CROWN.FlatBox α))
  (inputs : List (Op.Value ℝ))
  (values : Array (List (Op.Value ℝ))),
  CROWN.Op.Directed.RealConsts Pr P →
    (∀ (id k : ℕ) (p : Port),
        G.nodes[id]? = some (Node.input k p) →
          ∀ (B : CROWN.FlatBox α),
            seeds id = some B →
              ∀ (x : Op.Value ℝ) (y : Port.Carrier ℝ p)
                (v : CROWN.Graph.FlatTensor ℝ),
                inputs[k]? = some x →
                  p.read x = Except.ok y →
                    CROWN.Op.flatOf ⟨p, y⟩ = some v →
                      CROWN.Op.Directed.Encloses B v) →
      G.run Pr inputs = Except.ok values →
        ∀ id < G.nodes.size,
          ∀ (B : CROWN.FlatBox α)
            (v : CROWN.Graph.FlatTensor ℝ),
            (CROWN.Op.runIBP G P seeds)[id]! = some B →
              CROWN.Op.runVal values id = some v →
                CROWN.Op.Directed.Encloses B v
```

The theorem assumes lawful endpoint operations (`LawfulBoundOps`, `LawfulNonlinearBoundOps`), a
real payload `Pr` holding the real value of each rounded constant of `P` (`RealConsts`), seed boxes
that enclose the graph inputs their input nodes read, and a successful real run of the same graph
on `Pr`. Under those assumptions, every box the rounded pass returns encloses the value of its node
in that run. On its own, it doesn't establish that the real run succeeds, or that a node gets a
box.

No premise constrains the operators: every operator `Op.runIBP` gives a box has a proved rule (the
sum at node 8 and the `tanh` at node 9 among them), so the theorem covers the ten-node graph as
written. Its scalar requirements hold for the reals and for rounded-real `FP32`. But FloatLib's
executable binary32 carrier has no `LawfulBoundOps` instance, so the theorem doesn't apply to the
boxes the command above computes.

In the theorem, the real run is at `ℝ`, whereas the command above uses FloatLib binary32. Relating
that rounded propagation to real arithmetic takes another argument, which
{ref "verification"}[Verification And Certificates] and {ref "fp32-soundness"}[Floating-Point
Soundness] develop. Applying the enclosure theorem to an execution requires both its payload and
input hypotheses and this connection between scalar semantics.

`RealConsts` ties the real constants to the rounded ones, and the input hypothesis supplies the
initial enclosure from which the propagation proof starts. The Boolean containment check only
compares an already supplied point and box. It performs neither this induction nor a check of
universal input hypotheses.

# Executable Coverage And Proof Coverage

`Graph.eval` runs every registered operator by its rule in the operator table, so there's nothing
to lower before a graph runs: the value table shown earlier is the execution. It returns an error
when an operator's checks refuse its inputs (`Op.invoke`), while a raw logarithm of a nonpositive
input returns the scalar's IEEE values (a NaN below zero, $`-\infty` at zero), as `Tensor.log`
does. A custom node runs only when the payload supplies its checked program: recognizing its tag
isn't enough to execute it.

As for proofs, the evaluation theorems state the ports of a successful run, and `evalOp_eq` the
outputs of an operator node through its rule, but none of them proves an asymptotic complexity
bound.

# Axis Operations And Lowering Coverage

An axis is stored as data, but changing it changes which coordinates an operation combines. Here we
use an interior axis, so the shape rules must account for dimensions on both sides:

```terminal
# Exercise nontrivial axes so a last-axis-only
# implementation cannot pass unnoticed.
scripts/lake.sh exe torchlean ir_axis_ops
```

```terminal +output
== IR axis ops tutorial ==
Each operator graph passes Graph.check, then Graph.eval computes it on CPU.

-- softmax axis=1 on shape [2,3,4]
[softmax_middle_axis] output shape: [2, 3, 4]

-- layerNorm from axis 1 on shape [2,3,4]
PyTorch meaning: normalized_shape = x.shape[axis:] = [3,4]
[layer_norm_middle_axes] output shape: [2, 3, 4]

-- cat dim=1: [2,3,4] ++ [2,5,4] -> [2,8,4]
[cat_middle_dim] output shape: [2, 8, 4]
```

(The full transcript also prints the leading scalars.) Each graph of registered operators is built
with `Graph.Builder`, whose `build` runs the structural check `Graph.check`, and then the example
calls `Graph.eval`, which computes the graph's outputs with the operators' rules. LayerNorm over
the axes from one on is `Op.layerNorm` over one axis of twelve entries, between two reshapes. The
tutorial supplies the graphs and prints the results, and checking and execution live in the
library. This is the Lean tensor evaluator, not LibTorch dispatch. Choose the scalar with
`--arithmetic native`, `ieee`, or `complex` (this command doesn't accept device or
backend-reporting flags).

Softmax and cat take their axis as a `Fin` of the rank, so an axis value carries its bound: a
numeral is read modulo the rank, and `(⟨3, by decide⟩ : Fin 3)` does not elaborate. Change
`cat 1` to `cat 0` in {src "TorchLeanExamples/DeepDives/IRAxisOps.lean"}[`IRAxisOps.lean`] and
`build` rejects the concatenation before evaluation: its first input reads a port of shape
`[3, 0, 4]`, which the `[2, 3, 4]` tensor does not fit.

On `[2,3,4]`, softmax along axis one normalizes three entries while holding the first and last
coordinates fixed. LayerNorm starting at that axis combines twelve entries into each row's
statistics. Concatenation instead changes one extent and retains the others. These examples
exercise choices that a last-axis-only test would miss.

Changing the normalization axis can change values without changing the output shape, so shape
soundness alone can't establish that the intended axis was used. Semantic preservation must
identify the values too.

# Kernel Planning

For each operator node, the backend adapter reads its registered operator, chooses an admissible
kernel capsule among those the implementation tables offer for that operator, and keeps the node's
identity so the plan can be audited against the graph. On our graph, with the default CPU policy
and the maintained implementation tables:

```lean (name := irPlan)
-- Inspect capsule selection for the executable nodes of
-- this graph.
#eval (graph >>= fun g =>
    g.plan {} {} Backend.Registry.maintained).map
  fun plan => (plan.nodeIds, plan.capsuleNames)
```

```leanOutput irPlan (whitespace := lax)
Except.ok (#[3, 4, 7, 8, 9],
  #["reference.linear", "reference.relu",
    "reference.linear", "reference.reduce",
    "reference.tanh"])
```

Five capsules for ten nodes. Input node 0 and constant nodes 1, 2, 5 and 6 are absent because they
need no runtime work, and planning skips an `Op.detach` node too. Node 8 is the operator `reduce`
with its fields (a sum over axis `0` of a `[3]` input), and the reference table offers
`reference.reduce` for every sum reduction, whatever its axes. `IR.Graph.plan_sound` states
what each kernel in a plan is: it belongs to an operator node of the graph and records that node's
operator, and its capsule is offered for that operator by an available implementation and is
admissible under the policy.

A `KernelCapsule` describes the selected contract. It contains no executable closure, so planning
node 3 for `libtorch` does not call ATen. Eager execution runs the selected capsule only with an
implementation of the same operator, provider, and device.
{ref "backend-selection"}[Backend Selection] follows the selection and assurance model in detail.

Keeping that distinction prevents a common architecture mistake:

```
-- Each stage below answers a different question about
-- backend support.
registered     ≠ selectable
selectable     ≠ executable
executable     ≠ proved correct
```

Moving between these states requires availability checks, runtime dispatch, and evidence for the
selected implementation. A capsule's numerical policy also decides whether an arithmetic bound for
a particular reduction order applies to the proposed kernel.

# Canonical IR And Autograd Tapes

Where the canonical IR records a persistent model computation, an eager autograd tape records one
execution, and carries what an execution has:

- concrete runtime tensor handles
- which values require gradients
- saved forward values needed by VJPs
- the actual order in which wrappers ran

A tape may contain enough information to reconstruct an IR-like graph, but it isn't
`IR.Graph`. Conversely, the canonical IR doesn't own mutable gradient buffers or optimizer
state. {ref "runtime-autograd"}[Runtime and Autograd] builds a tape by hand and shows what those
four bullet points look like as data.

That difference is also why LibTorch can compute values in a TorchLean-owned backward path. An
operator's CUDA entry in `LibTorch.Kernel.ofOp` records the local tape node and its VJP, and ATen
executes the requested tensor operations with its autograd recording disabled. The canonical
semantic graph and the execution tape remain distinct objects, connected by the operation contract.

A forward value table isn't enough, on its own, to reconstruct a proved reverse pass either. A
reverse computation needs the selected VJP for every operation, the saved inputs that rule reads,
and a consistent accumulation of cotangents at shared parents, and two tapes may agree on every
forward value while disagreeing on one of those backward choices. What makes the forward lowering
theorem valuable is that it fixes the denotation being differentiated, but the autograd link must
also connect the reverse rules to derivatives of that denotation.

# Graph Results And Validation Evidence

For any graph-based claim, locate:

1. the exact graph representation
2. its concrete parameter payload
3. the structural check that ran
4. the scalar context used by evaluation
5. the lowering theorem and fragment, if lowering was used
6. the capsule selected for each native operation
7. the provider branch that actually executed it

These records distinguish a failed structural check from a payload mismatch, a lowering restriction,
or an unavailable runtime provider.

Here's the source map:

- {src "TorchLean/NN/IR/Graph.lean"}[`Graph.lean`] for nodes, references and payloads
- {src "TorchLean/NN/IR/Builder.lean"}[`Builder.lean`] for building checked graphs
- {src "TorchLean/NN/IR/Eval.lean"}[`Eval.lean`] for evaluation, the structural check, and the
  theorems relating both to ports
- {src "TorchLean/NN/IR/Pretty.lean"}[`Pretty.lean`] for the listing and the GraphViz output
- {src "TorchLean/NN/IR/Artifact.lean"}[`Artifact.lean`] for the `torchlean.ir` artifact
- {src "TorchLean/Verify/CROWN/Op/Exec.lean"}[`Exec.lean`] and
  {src "TorchLean/Verify/CROWN/Op/Directed.lean"}[`Directed.lean`] for interval propagation and its
  enclosure theorem
- {src "TorchLean/NN/Backend/Graph.lean"}[`Backend/Graph.lean`] for capsule planning
- {src "TorchLeanExamples/DeepDives/GraphSemantics.lean"}[`GraphSemantics.lean`] for the complete
  ten-node experiment

# Variations On The Ten-Node Graph

Changing one field at a time isolates the checks and interpretations used above.

1. Add an eleventh node, `Op.relu` at shape `[3]` on node 9, and predict what `build` reports
   before running it. Then give it shape `[]` and predict again.
2. Swap the payload's two weight keys so node 1 gets node 5's weights. Predict which node fails and
   with which message.
3. Change the input box radius from `0.05` to `0.5` and look at node 4. Then find the radius at
   which the interval for zero-based coordinate `1` stops being `[0, 0]`.
4. Replace the final `Op.tanh` with `Op.sigmoid` and rerun the interval block. Explain the output
   box from the instance definitions, not from the numbers.
5. Replace it with `Op.abs` instead, which has a real interval rule at FloatLib binary32, and see
   the output interval become informative again.
6. Add an `Op.log` node after the final `Op.tanh` and evaluate the graph. Predict which output
   coordinates are NaN or $`-\infty` (the IEEE logarithm of a negative input or of zero), and check
   that the evaluation itself doesn't fail.
7. Plan the graph with a policy asking for a provider this build doesn't have, and read the error.
   It comes from the backend boundary, not from the graph.
