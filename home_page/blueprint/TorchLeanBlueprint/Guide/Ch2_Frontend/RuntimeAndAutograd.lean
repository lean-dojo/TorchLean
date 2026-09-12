import VersoManual
import NN.API
import NN.IR
import NN.Backend
import NN.Runtime.Autograd.Engine
import NN.Runtime.Autograd.IRExec
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.Roles

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean
open TorchLean
open Runtime.Autograd
open Lean.Elab.Tactic.GuardMsgs.WhitespaceMode (lax)

#doc (Manual) "Runtime and Autograd" =>
%%%
tag := "runtime-autograd"
file := "What-Actually-Runs"
%%%

Several representations sit behind an autograd call. An eager tape stores the values and local
reverse rules for one execution. A typed graph records operations that can be reused with new
parameters and inputs. The canonical IR exposes operations as data for inspection and
verification, and its lowered forward graph executes those operations. A kernel plan records
the selected implementations. Their lifetimes and contents determine which questions each can
answer.

We will follow a branched calculation through the tape and typed graph, then examine the
boundaries between execution, inspection, and proof. {ref "autograd-walkthrough"}[Differentiation By
Example] covers the public derivative API; {ref "execution-modes"}[Choosing How A Model
Runs] covers runtime selection. Here the goal is to locate the values, rules, and checks behind
those interfaces, particularly when two executions disagree.

The named Lean blocks are elaborated while this page is built, together with checks of their
displayed outputs. The command-line and PyTorch transcripts record separate runs; they are not
executed by the page build.

# API Entry Points And Runtime Artifacts

Three levels of API can compute the same gradient, and it matters which one you call, because they
do not all build the same runtime object.

:::table +header
*
  * You call
  * What is built
  * Typical use
*
  * `Trainer` with `execution := .eager`
  * an eager tape per step
  * ordinary training
*
  * `Trainer` with `execution := .typedGraph`
  * one typed graph, reused
  * ordinary training, fewer rebuilds
*
  * `autograd.grad` and `autograd.vjp`
  * a typed graph for each transform invocation
  * differentiating a function or a loss
*
  * `Tape` and `TapeM`
  * a tape you own, as pure data
  * inspection, tests, and this chapter
:::

The `autograd.*` transforms do not consult the trainer's execution setting. In
{src "NN/Runtime/Autograd/Model/Autodiff.lean"}[`Autodiff.lean`], `gradients` starts with
`lowerScalarToTypedGraph`, then evaluates through checked tape lowering and a reverse pass.
The graph describes the reusable computation; the tape captures values for this invocation.
Higher-order transforms in the same file lower graphs to tapes over dual scalars, so the reverse
calculation also carries directional derivatives. The trainer's execution choice is separate
from these transforms.

Consequently, comparing `autograd.grad` with an eager training step compares two execution paths.
A discrepancy could come from either path or from the state supplied to it. Inspecting only the
eager tape would leave the transform's typed graph unexamined.

# Runtime Artifact Types

The type parameters reveal which information callers must supply statically and which information
the representation stores internally:

```lean (name := rtTypes)
-- Inspect the static information retained by each runtime
-- artifact.
#check @Tape
#check @Torch.TypedGraph
#check @IRExec.ForwardGraph
#check @NN.IR.Graph
#check @NN.Backend.IR.GraphKernelPlan
```

```leanOutput rtTypes (whitespace := lax)
Tape : (α : Type) → [Storage α] → Type
```

```leanOutput rtTypes (whitespace := lax)
Torch.TypedGraph : (α : Type) → [Storage α] →
  List Shape → Shape → Type
```

```leanOutput rtTypes (whitespace := lax)
IRExec.ForwardGraph : (α : Type) → [Storage α] → Type
```

```leanOutput rtTypes (whitespace := lax)
NN.IR.Graph : Type
```

```leanOutput rtTypes (whitespace := lax)
NN.Backend.IR.GraphKernelPlan : Type
```

`Tape α` takes an element type and nothing else. No shapes appear in the type, which means the tape
cannot enforce a shape agreement in the type system; it stores shapes as values and checks them at
runtime. That is a deliberate trade, and we come back to it below.

`Torch.TypedGraph α Γ τ` carries a context `Γ : List Shape` and an output shape `τ`. A node cannot
refer to a parent at the wrong shape, because the shape is part of the graph's type. This is the
same trick the sequential architecture language uses in {ref "graphspec"}[GraphSpec: One
Architecture, Several Meanings], applied one level lower.

`IRExec.ForwardGraph α` looks like the tape again, but for a different reason: its shapes are stored
in fields (`inShape`, `ss`) because the graph it was lowered from did not know them statically
either.

`NN.IR.Graph : Type` has no `α`. The canonical IR is not
parameterized by a scalar type, so the same graph can be interpreted over the
reals, over intervals, and over binary32 without being rewritten. See
{ref "graphs-and-ir"}[Graph IR].

`NN.Backend.IR.GraphKernelPlan : Type` has no element type and no shapes. It holds node identifiers
and capsule metadata, and it contains no tensors whatsoever. A plan therefore cannot witness that
anything executed. Accepting a plan is a statement about a choice, not about a computation.

Here is the same set with the two remaining columns filled in:

:::table +header
*
  * Object
  * Purpose
  * Contains
*
  * eager tape
  * reverse-mode execution
  * values, parents, local VJPs
*
  * `Torch.TypedGraph`
  * repeated runtime execution
  * shape-indexed forward, JVP, and VJP functions
*
  * `IRExec.ForwardGraph`
  * execute a lowered `NN.IR.Graph`
  * a forward-only shape-indexed node list
*
  * `NN.IR.Graph`
  * inspection and verification
  * explicit operation tags and payload references
*
  * `NN.Backend.IR.GraphKernelPlan`
  * kernel selection and audit
  * source node ids and selected capsule metadata
:::

A CUDA Graph capture is a separate device launch-and-replay mechanism; TorchLean's `.typedGraph`
setting does not select it. The other boundaries also require explicit connections. An accepted
kernel plan needs runtime binding before it says anything about execution, and an IR semantics
theorem needs a provider refinement result before it applies to a native tape node.

The five type signatures divide the debugging problem into useful pieces. A wrong tensor value
belongs to an execution artifact such as the tape or evaluated graph context. A wrong operation
tag belongs to the canonical IR. A surprising provider belongs to the kernel plan and its runtime
binding. None of the last two objects contains the floating-point intermediate that caused a loss
to become nonfinite. Keeping these objects distinct tells us which additional record to request
instead of treating every representation called a graph as interchangeable.

The scalar parameter also has consequences. The same shape-indexed program can be interpreted
with different scalar operations, but a tape already contains values of its chosen `α`. Changing
arithmetic means evaluating with the other scalar interpretation, not relabeling those stored
values. Shape agreement survives that choice; numerical agreement requires a separate argument.

# Manual Tape Construction

`Tape α` is a grow-only array of nodes; `TapeM α` is `StateT (Tape α) (Except String)`, so a
tape-building computation is a pure function and its errors are values. To examine the node
structure, record two elementwise products sharing the same input, followed by their sum:

$$`z = x \odot x + x \odot x, \qquad \ell = \textstyle\sum_i z_i`

Here $`\odot` denotes elementwise multiplication, $`z` has the same shape as $`x`, and
$`\ell` reduces it to a scalar. The following code uses $`x=(3,-1.5)`:

```lean (name := rtTape)
-- Record both square branches separately so their shared
-- input is visible.
/-- `z = x*x + x*x`, summed, built on the eager tape. -/
def rtTwice : Result (Nat × Tape Float) :=
  TapeM.run Tape.empty do
    let x ← TapeM.leaf (α := Float) (s := [2])
      [3.0, -1.5] (name := "x")
    let a ← TapeM.mul (α := Float) (s := [2]) x x
    let b ← TapeM.mul (α := Float) (s := [2]) x x
    let z ← TapeM.add (α := Float) (s := [2]) a b
    TapeM.sum (α := Float) (s := [2]) z

#eval rtTwice.map fun (out, tape) => (out, tape.size)
```

```leanOutput rtTape (whitespace := lax)
Except.ok (4, 5)
```

Five nodes, and the scalar output is node `4`. Each `TapeM` operation appends exactly one node and
returns its identifier, so identifiers are array indices and reverse-mode traversal is a loop from
`size - 1` down to `0`. Nothing here is hidden state: the tape is the value returned by `run`.

Ask the tape what it recorded:

```lean (name := rtNodes)
-- Print parent positions, including repeated uses of the
-- same node.
#eval rtTwice.map fun (_, tape) =>
  tape.nodes.map fun (node : Node Float) =>
    (node.name, node.parents)
```

```leanOutput rtNodes (whitespace := lax)
Except.ok #[(some "x", #[]), (some "mul", #[0, 0]),
  (some "mul", #[0, 0]), (some "add", #[1, 2]),
  (some "sum", #[3])]
```

The leaf has no parents. Both multiplications name node `0` twice, once for each factor, which is
literally what `x * x` means. The addition names both products. This printout is the graph, and it
came out of the same data structure the trainer uses.

A node also carries the piece we did not print, because it is a closure rather than data:

```
-- A local rule may emit several contributions to the same
-- parent.
backward : SomeTensor α → Result (Array (Nat × SomeTensor α))
```

Given an upstream cotangent for this node's value, return one contribution per parent. That
signature is the whole local interface of reverse mode. Nothing in it mentions the rest of the
graph, which is why local derivative rules can be stated and proved one operation at a time.

The result `(4, 5)` contains the output node identifier and the tape length. Node identifiers start
at zero, so the scalar sum at node `4` is the fifth recorded value. The result is not the loss
itself. At the chosen input, each square branch contains `[9, 2.25]`; adding them gives `[18, 4.5]`,
and the scalar sum is `22.5`. This calculation supplies a forward reference before inspecting any
derivative.

The two parent entries `[0, 0]` on each `mul` are intentional. Both operands refer to the same
input, but they occupy different argument positions in multiplication. A reverse rule owes one
contribution for each position. Deduplicating that parent list without changing the rule would
lose half of the derivative of a square. Reusing a value and using it only once are different
properties of a graph.

# Gradient Accumulation On The Tape

Two product branches run from `x` to `z`. Each squaring contributes $`2x`, and the
addition sends the output seed to both of its parents, so the leaf's cotangent is

$$`\bar x = 2x + 2x = 4x.`

At `x = (3, -1.5)` that predicts `(12, -6)`. Seed the scalar with `1` and read node `0`:

```lean (name := rtGrad)
-- Recover the input cotangent only after checking its
-- stored shape.
/-- Seed the scalar output, then read the leaf gradient. -/
def rtLeafGrad (built : Result (Nat × Tape Float)) :
    Result (Tensor Float [2]) := do
  let (out, tape) ← built
  let grads ← Tape.backwardScalar (α := Float) tape out
  match grads[0]? with
  | some g => Tape.requireGrad (α := Float) (τ := [2]) g
  | none => .error "leaf 0 received no gradient"

#eval rtLeafGrad rtTwice
```

```leanOutput rtGrad (whitespace := lax)
Except.ok [12.000000, -6.000000]
```

Each `mul` returns two contributions to node `0`, one for each occurrence of `x`, giving four
contributions in total. If later
contributions overwrote earlier ones instead of adding, the result would be wrong. Accumulation
happens in exactly one place,
`SomeTensor.add` in {src "NN/Runtime/Autograd/Engine/Core/Base.lean"}[`Engine/Core/Base.lean`],
which checks that the two shapes agree before adding. Reverse-mode AD on a DAG is a sum over paths
({Informal.citet baydin2018}[]), and this one function is where the sum is taken.

This is also why TorchLean's autograd proofs come in two layers rather than one. There are primitive
derivative facts, one per operation, and there is a global soundness statement about the traversal
that composes them. A library that proves only the first layer has proved that its rules are right
and said nothing about whether they are combined correctly.
{ref "autograd-proofs"}[Proving Autograd Correct] is where both layers are stated and discharged.

## PyTorch Gradient Accumulation

PyTorch accumulates the two branch contributions within a backward pass as well
({Informal.citep pytorch2019}[]). Detaching the second product removes its contribution:

```
# Compare two live branches with one branch detached from
# autograd.
import torch

x = torch.tensor([3.0, -1.5], requires_grad=True)
z = (x * x + x * x).sum()
z.backward()
print("both branches live:", x.grad.tolist())

x.grad = None
z = (x * x + (x * x).detach()).sum()
z.backward()
print("one branch detached:", x.grad.tolist())
```

On torch 2.13.0 that prints:

```
both branches live: [12.0, -6.0]
one branch detached: [6.0, -3.0]
```

The first line agrees with the tape we built. The second line is the next section.

The backward closure's result is a list of contributions, not a map with one entry per parent.
That lets multiplication return two entries for node `0`, and lets the traversal add contributions
from both square branches later. At `x = [3, -1.5]`, one square contributes `[6, -3]`; the second
contributes the same vector. Their sum explains every coordinate of `[12, -6]` without relying on
the implementation's own gradient formula as an oracle.

The scalar seed is `1`, so this is the gradient of the displayed scalar sum. A different scalar
seed would scale all the contributions. A vector-output seed would instead choose a weighted sum
of outputs and would need the output's shape. Runtime shape checks protect that interface even
though node identifiers themselves are plain natural numbers.

# Stop-Gradient Nodes

`detach` is often described as "turning off gradients", which suggests a mutable switch. In this
engine it is a node like any other, and the node is short enough to read in full. Here is exactly
what {src "NN/Runtime/Autograd/Torch/Core/Session.lean"}[`Core/Session.lean`] appends, transcribed
into the pure API:

```lean (name := rtStop)
-- Preserve the second branch value while returning no
-- parent cotangents.
/--
The stop-gradient node, built the way the eager session
builds it: the forward value is reused, the parent edge
stays, and the local rule returns no contributions.
-/
def rtDetachNode (id : Nat) :
    TapeM Float Nat := fun tape => do
  let value ←
    Tape.requireValue (α := Float) (t := tape) (s := [2]) id
  let (tape', newId) := Tape.addNode tape
    { name := some "detach"
      value := Spec.SomeTensor.ofTensor value
      requiresGrad := false
      parents := #[id]
      backward := fun _ => .ok #[] }
  pure (newId, tape')

/-- The same computation, second branch stopped. -/
def rtStopped : Result (Nat × Tape Float) :=
  TapeM.run Tape.empty do
    let x ← TapeM.leaf (α := Float) (s := [2])
      [3.0, -1.5] (name := "x")
    let a ← TapeM.mul (α := Float) (s := [2]) x x
    let b ← TapeM.mul (α := Float) (s := [2]) x x
    let bStop ← rtDetachNode b
    let z ← TapeM.add (α := Float) (s := [2]) a bStop
    TapeM.sum (α := Float) (s := [2]) z

#eval rtLeafGrad rtStopped
```

```leanOutput rtStop (whitespace := lax)
Except.ok [6.000000, -3.000000]
```

`2x` instead of `4x`, matching PyTorch's second line to the digit. Three fields did that work:
`backward` returns an empty array, so no contribution flows; `parents` still names the source, so
the forward edge remains visible to anything that reads the graph; and `requiresGrad := false`
causes parent accumulation to skip this node. The forward tensor value is reused without
recomputing or copying its payload, so the forward result is unchanged.

The node list makes the difference visible:

```lean (name := rtStopNodes)
-- Keep stopped forward nodes visible even though reverse
-- propagation cannot cross them.
#eval rtStopped.map fun (_, tape) =>
  tape.nodes.map fun (node : Node Float) =>
    (node.name, node.parents)
```

```leanOutput rtStopNodes (whitespace := lax)
Except.ok #[(some "x", #[]), (some "mul", #[0, 0]),
  (some "mul", #[0, 0]), (some "detach", #[2]),
  (some "add", #[1, 3]), (some "sum", #[4])]
```

Six nodes now, and the addition's parents are `1` and `3`: the live product and the detached copy.
The tape still knows the whole forward computation. Only the reverse rule changed.

## Reverse-Pass Reachability

Because the detached node hands back nothing, the traversal never asks node `2` for anything, and
`2` never asks the leaf. That is observable. `Tape.backward` returns a map keyed by node
identifier, containing exactly the nodes that received a cotangent:

```lean (name := rtReached)
-- Inspect presence in the gradient map separately from
-- numerical zero.
#eval rtStopped.bind fun (out, tape) =>
  (Tape.backwardScalar (α := Float) tape out).map
    fun grads =>
      (grads.toList.map fun (id, _) => id).mergeSort
        (· ≤ ·)
```

```leanOutput rtReached (whitespace := lax)
Except.ok [0, 1, 4, 5]
```

Nodes `2` and `3` are absent: the detached product and the detach node itself. Nodes `0`, `1`, `4`
and `5` are the surviving path from the leaf through the live product, the addition, and the sum.
The sparse map distinguishes an absent gradient from a gradient whose entries are zero.
A missing key means no
gradient was retained for that node, including when `requiresGrad` disables
it. A key holding zeros means the accumulated runtime cotangent is zero, which can also arise from
cancellation or rounding; an analytic derivative claim needs the usual correctness hypotheses.

Stopping the second branch leaves the forward scalar at `22.5`: the same two arrays are still
added. Its derivative becomes `[6, -3]` because only the first square sends a cotangent back to the
input. The printed node list keeps the stopped branch visible, which is useful when checking why
a forward result survived a change to differentiation behavior.

The reached identifiers `[0, 1, 4, 5]` describe the surviving reverse path through the input, first
square, addition, and sum. The detach node and the second square are not reached by that path.
An absent entry means no cotangent was propagated there. A present zero tensor could instead mean
that contributions canceled or that a reached local rule returned zero. These cases can have the
same final numerical gradient while exercising different reverse computations.

# Differentiation Through The Transform API

We now have a hand-built answer. Compare it with the answer the public transform API gives, which
takes an entirely different route: it lowers the program to a typed scalar graph and differentiates
that. Write the program once against the operation interface, so that the same text can be
interpreted either way:

```lean (name := rtTransform)
-- Express the same two branch choices through the public
-- differentiation interface.
/-- `sum (x*x + x*x)`, for the transform API. -/
def rtQuad : autograd.Function [2] [] :=
  fun x => do
    let a ← Torch.mul x x
    let b ← Torch.mul x x
    let z ← Torch.add a b
    Torch.sum z

/-- The same program, second branch stopped. -/
def rtQuadStopped : autograd.Function [2] [] :=
  fun x => do
    let a ← Torch.mul x x
    let b ← Torch.mul x x
    let bStop ← Torch.detach b
    let z ← Torch.add a bStop
    Torch.sum z

#eval do
  let x : Tensor Float [2] := [3.0, -1.5]
  let gradient ← autograd.grad rtQuad x
  let stopped ← autograd.grad rtQuadStopped x
  IO.println s!"both branches live = {gradient}"
  IO.println s!"second detached    = {stopped}"
```

```leanOutput rtTransform (whitespace := lax)
both branches live = [12.000000, -6.000000]
second detached    = [6.000000, -3.000000]
```

The tape, the typed graph, the closed-form calculation, and PyTorch agree on $`4x` and $`2x` at
this input. The comparison checks several implementations of the same branch structure. If only
one result changes, that narrows the investigation to its construction or execution path; shared
rules can still produce shared errors.

The public transform reproduces the two tape gradients by expressing the same branch structure
through functional operations. The comparison is useful because the manual tape spells out parent
identifiers while the transform records them through its handler. Agreement checks this small
construction at the chosen values. It does not establish that every operation's VJP is correct,
or that the transform and manual tape could not share a primitive bug.

A useful independent reference here is the algebraic calculation above: two square branches give
`4x`, and one stopped branch leaves `2x`. For a more complicated program, derive a small special
case or a directional check before comparing two execution routes that reuse the same rules.

# Eager And Graph Interpreters

The reason the same source text can be executed eagerly or recorded into a graph is not a flag
inside the runtime. It is two instances of one class. `Ops` is the operation interface that
`Torch.mul`, `Torch.add` and the rest are defined against, and
{src "NN/Runtime/Autograd/Torch/Core/Trainer.lean"}[`Core/Trainer.lean`] provides two instances of
it. The difference between them is visible in a single field, `Ref`, which says what a handle to an
intermediate value actually is:

```lean (name := rtRefs)
-- These equalities identify interpreter reference types
-- without evaluating a model.
example :
    Torch.Ops.Ref (Torch.Internal.EagerM Float) Float [2] =
      Torch.TensorRef Float [2] := rfl

example {Δ : Type} {Γ : List Shape} :
    Torch.Ops.Ref
        (TypedGraph.GraphM.MWith Float Δ Γ) Float [2] =
      TypedGraph.GraphM.Var [2] := rfl
```

Both equalities hold by reduction, as the `rfl` proofs show. Under the eager instance a reference
is a `TensorRef`, an identifier into the
session's tape, and each operation executes immediately and appends a node. Under the graph
instance a reference is a `GraphM.Var`, an SSA variable, and each operation records a node without
computing anything.

This is the tagless-final pattern ({Informal.citet kiselyov2012}[]): one object language, several
interpreters, with the object language's types carried by the host language. The two fields of a
GraphSpec primitive use the same design. A shared operation interface keeps the program text
aligned, while the interpreters still need correspondence checks for the operations they implement.

The two `rfl` proofs identify what the abstract reference type becomes in each interpreter. They
do not evaluate a tensor or compare two model predictions. In an eager session a reference names
a recorded value; a graph variable identifies an entry in a typed context. That distinction lets
the same authored operation sequence build either representation.

The maintained eager handle also records a session owner and generation. A reset can reuse numeric
node `0`, so its number alone cannot establish that an old handle belongs to the new recording.
The runtime checks those fields before accepting a session reference. This protects a different
property from shape safety: two tensors may both have shape `[2]` while only one belongs to the
current execution.

# Typed Graph Execution

Typed graph execution records the model's scalar loss once as a typed SSA graph. Nodes carry the
forward behavior and the executable rules used for JVPs and VJPs. Each training step supplies
current parameters and inputs and reuses that graph. A separate typed reference selects the result,
so the graph may return an input or an earlier node without adding a dummy final operation, and
forward and reverse execution read and seed that exact reference.

The graph's indices enforce shape agreement between node references.
Derivative correctness does not. The executable node payload holds three functions, `forward`, `jvp`
and `vjp`, and no proof field. The proof-carrying node in
{src "NN/Proofs/Autograd/Tape/Algebra/Soundness.lean"}[`Tape/Algebra/Soundness.lean`] adds the local
adjointness law relating its `jvp` and `vjp`, and the proof-carrying graph composes those local laws
into a graph theorem. Two representations exist because execution must not require a proof and
verification must not accept a graph without one; the erasure runs one way only.
{ref "autograd-proofs"}[Proving Autograd Correct] follows that ladder up.

The public model-level name specializes the context of this same graph type:

```lean (name := rtWrapper)
-- Model state precedes the input in the lowered graph
-- context.
example {α : Type} [Storage α]
    {stateShapes : List Shape} {σ τ : Shape} :
    nn.TypedGraphModel stateShapes σ τ α =
      Torch.TypedGraph α (stateShapes ++ [σ]) τ := rfl
```

`nn.TypedGraphModel` is an abbreviation whose context is model state followed by one input tensor.
It introduces no additional runtime representation; the equality unfolds to `rfl`.

The typed graph trainer currently supports CPU execution. A non-CPU request is rejected, so a
successful run's execution label continues to identify the path being measured. Recording and
lowering also do not imply the optimization, fusion, scheduling, or native code generation
associated with a compiler such as `torch.compile`.

The wrapper equality says where model state enters the graph: the leaf context is the list of
state shapes followed by the input shape. A caller can keep state and input separate at the API,
while lowering supplies one ordered context to the graph. The order is part of the interface; two
same-shaped parameter tensors are not interchangeable merely because either fits a slot.

A node's executable JVP and VJP fields are functions. Their presence makes evaluation possible,
but does not supply an adjointness proof. The proof-carrying graph adds the local laws needed for
a global theorem. This is why a type-safe executable graph can still contain a numerically wrong
rule, and why inspecting the graph type alone cannot settle derivative correctness.

# Canonical IR Representation

Typed graph nodes hold Lean functions. Those functions can execute an operation, but they do not
expose its structure as inspectable data. `NN.IR.Graph` instead represents operations such as
linear, ReLU, reduction, normalization and shape
transforms as data. Each node records an operation tag, parent identifiers, and its output shape.
Input shapes are obtained from parent nodes; parameter data is looked up in the separate payload.

Because the operation is a tag, an importer can validate it, a verifier can interpret it, and a code
generator can reject what it does not support without executing an arbitrary function. The
scalar-independent representation also permits several interpretations:
{ref "graphs-and-ir"}[Graph IR] builds
one small graph and evaluates it over the reals, over intervals, and over IEEE binary32 scalars
without touching the graph.

The bridge from the canonical IR produces `IRExec.ForwardGraph`, not `Torch.TypedGraph`, and it is
forward-only. The IR lowering theorem compares the lowered graph's value table with
`NN.IR.Semantics`. This bridge supplies no JVP or VJP rules; differentiating an imported operation
would require additional rules and corresponding correctness statements.

Both paths honor the homogeneous-element contract from {ref "tensors-shapes"}[Tensors And Shapes]: a
typed graph is parameterized by one `α`, and an IR denotation receives one payload over `α` and
produces values over that same `α`.

# Shape Erasure And Recovery

A parameter pack holds differently shaped tensors that share one element type:

```
-- Each state entry keeps its own shape; the pack also fixes
-- the order of the two layers.
[weight1 : Tensor α [8,2],
 bias1   : Tensor α [8],
 weight2 : Tensor α [1,8],
 bias2   : Tensor α [1]]
```

The pack type preserves that dependent list. A tape cannot, because `Tape α` has no shape index, and
neither can a registry that has to iterate over parameters by name. Both store `Spec.SomeTensor α`,
a structure holding a shape and a tensor indexed by that shape. The collection can therefore hold
different shapes while each entry retains its own shape-value relationship.

```lean (name := rtPack)
-- Recovery needs a shape equality; the second declaration
-- proves a pack/cast roundtrip.
#check @Spec.SomeTensor.cast
#check @Spec.SomeTensor.ofTensor_cast
```

```leanOutput rtPack (whitespace := lax)
@Spec.SomeTensor.cast : {α : Type} →
  [inst : Storage α] → {shape : Shape} →
  (value : Spec.SomeTensor α) →
  value.shape = shape → Tensor α shape
```

```leanOutput rtPack (whitespace := lax)
@Spec.SomeTensor.ofTensor_cast : ∀ {α : Type}
  [inst : Storage α] (value : Spec.SomeTensor α)
  {shape : Shape} (h : value.shape = shape),
  Spec.SomeTensor.ofTensor (value.cast h) = value
```

First, `cast` takes a proof. Its last explicit argument is `value.shape = shape`, so there is no way
to recover a typed tensor from a packed one without producing evidence that the stored shape is the
expected shape. A runtime comparison can supply that evidence, but `rfl` cannot establish an
arbitrary packed tensor's shape:

```lean +error (name := rtNoProof)
-- An arbitrary packed shape cannot be identified with [2]
-- by reflexivity.
example (value : Spec.SomeTensor Float) :
    Tensor Float [2] :=
  value.cast rfl
```

```leanOutput rtNoProof (whitespace := lax)
Application type mismatch: The argument
  rfl
has type
  ?m.11 = ?m.11
but is expected to have type
  value.shape = [2]
in the application
  value.cast ⋯
```

Second, `ofTensor_cast` says the round trip is the identity: pack a tensor, check its shape, and you
are back where you started with nothing lost along the way. So the shape did not vanish. It moved
from a compile-time index of the whole collection into a value stored beside each entry, and a
theorem says the move is reversible.

The instantiated runner also holds parameter
names, bounded token inputs, RNG state, optimizer memory, mutable model buffers, and the train or
eval flag.

The cast signature takes an equality from the stored shape to the requested shape. Its direction
matters: it transports the existing tensor; it does not reshape, truncate, pad, or reinterpret the
buffer. The roundtrip theorem says that packing a tensor and recovering it with the matching
shape returns that tensor. It does not manufacture an equality for arbitrary packed data.

The rejected `rfl` illustrates exactly that missing information. For an arbitrary `SomeTensor`,
Lean cannot reduce its hidden shape to `[2]`. A runtime equality test can provide evidence in its
successful branch, after which the cast is justified. Until that check succeeds, reporting an
error preserves the caller's shape contract. The printed metavariable names are elaborator details;
the useful part of the diagnostic is the demanded equality with `[2]`.

# Model Mode And Random State

Some operations depend on mode. Dropout samples a mask during training and applies its
deterministic inference behavior during evaluation ({Informal.citep dropout2014}[]); batch
normalization may update running statistics during training and use stored statistics at evaluation
({Informal.citep batchnorm2015}[]); other stochastic or stateful layers follow the same pattern.
`trained.predict` uses the retained runtime state in evaluation mode, and a `Trainer.Session`
follows the same rule in a caller-driven loop, with `step` in training mode and `predict` and `loss`
in evaluation mode. Mode is not a backend: one eager CUDA runner can switch modes without changing
device or provider profile. {ref "execution-modes"}[Execution Modes] measures the size of
the resulting difference, which is the part that matters when you are trying to tell a mode bug from
rounding noise.

For the tape, randomness raises a replay requirement. A stochastic forward pass and its reverse
pass must use the same random draw, or the reverse pass computes the derivative of a function nobody
evaluated. The tape interface makes replay deterministic, but correct capture of the forward mask
is still an obligation of the node implementation:

```lean (name := rtPure)
-- The result may report a malformed tape even though
-- backward has no IO effects.
#check @Tape.backwardScalar
```

```leanOutput rtPure (whitespace := lax)
@Tape.backwardScalar : {α : Type} →
  [inst : Storage α] → [Add α] → [One α] →
  Tape α → ℕ → Result (Std.HashMap ℕ (Spec.SomeTensor α))
```

No `IO`: in the pure Lean model, replay depends on the tape and its stored closures rather than an
external generator. A closure could still use an incorrect deterministic mask or an incorrect VJP;
purity alone does not prove it differentiates the recorded forward pass. Explicit capture and local
correctness laws supply that connection when treating reverse mode as a program on immutable data
({Informal.citet pearlmutter2008}[]). The grow-only array behind
it is efficient for the same reason Lean's own runtime can afford immutable structures
({Informal.citep immutablebeans2019}[]).

This also affects checkpoints. Inference can require persistent
buffers, mode, and preprocessing as well as parameters. Reproducing the next training update needs
additional state. Replaying that update also needs the generator state, the optimizer
state, and the loader position.

The pure backward type still returns `Result`: lack of `IO` does not mean that every supplied tape
is valid. The traversal can reject an invalid output id, an incompatible seed, or a parent
contribution with the wrong shape. Purity describes how the computation obtains its inputs;
validation describes which of those inputs it accepts.

The eager reverse engine also distinguishes an unreached node from a reached node with a zero
cotangent. It skips unreached local rules and fills missing entries with zeros when a dense result
is requested. This matters at singular values: evaluating a disconnected rule can produce an
expression such as `0 * (1 / 0)`, which need not be zero in floating-point arithmetic. The bridge
to a traversal that visits every node therefore needs a zero-preservation condition. That is a
mathematical obligation about the rules, not just a choice of array or hash-map storage.

# Checkpoint Wrapper Semantics

`nn.functional.checkpoint` currently provides a boundary for a future memory optimization. In
{src "NN/Runtime/Autograd/Model/Functional/Core.lean"}[`Functional/Core.lean`] the definition is

$$`\operatorname{checkpoint}(f, x) = f(x),`

an identity wrapper. Real checkpointing discards intermediate values during the forward pass and
recomputes them during the reverse sweep on a schedule that trades arithmetic for memory
({Informal.citet griewank2000}[]). None of that happens here.

A backend that implements recomputation can refine this wrapper without changing its mathematical
meaning. The boundary identifies where the execution strategy may change; the current
implementation provides no memory saving.

# Runtime And Proof

Relating execution to a derivative requires two kinds of correspondence.

The derivative ladder is internal to Lean:

```
calculus rule for each primitive
  -> semantic local VJP
  -> well-formed graph or tape composition
  -> global reverse result
```

The refinement ladder leaves Lean:

```
executable primitive
  -> declared semantic primitive
```

and it needs one rung per provider actually used by the run.

The first ladder can be climbed entirely inside Lean for a supported abstract graph, and
{ref "autograd-proofs"}[Proving Autograd Correct] climbs it: local adjointness laws compose into a
graph theorem, the executed dense reverse sweep is identified with that theorem's backpropagation,
and over the reals the result is identified with the adjoint of a Fréchet derivative from Mathlib
({Informal.citep mathlib2020}[]). The second ladder may be a theorem, a sound checked guard, a
numerical certificate, or an explicit trusted boundary. Native CUDA, LibTorch, the compiler, the
driver, and the hardware need their own refinement evidence or stated trust assumptions.

Relevant proof sources:

- {src "NN/Proofs/Autograd/Overview.lean"}[`Autograd/Overview.lean`];
- {src "NN/Proofs/Autograd/Tape/Algebra/Soundness.lean"}[`Tape/Algebra/Soundness.lean`];
- {srcDir "NN/Proofs/Autograd/Runtime/Link"}[`Runtime/Link`].

# Kernel Selection Records

Before an eager operation executes, the session resolves its backend profile and binds the accepted
capsule to a matching runtime handler; a missing provider or handler is an error. Run

```terminal
# Inspect the capsules selected for a short eager training
# run.
lake exe torchlean quickstart_mlp \
  --device cpu --steps 1 --seed 2026 --show-backend
```

and each operation reports one line the first time it is selected. Here is the entry for ReLU on
this machine, rewrapped to fit the page:

```
  relu: reference.relu provider=reference trust=checked
    vjp=torchlean-tape reduction=n/a
```

{ref "execution-modes"}[Execution Modes] reads the full report, tabulates the seven
capsules a CPU run selects, and explains the evidence lines and the empty banners. The field
`vjp=torchlean-tape` names which artifact owns the reverse rule for that operation. It connects
the tape we built by hand to the selected kernel: the
forward value came from the `reference.relu` provider, and the backward contribution came from
TorchLean's own tape node. When a provider owns its own backward instead, the corresponding
derivative claim also depends on that provider's reverse implementation.

Two deep dives follow a single operation the whole way down.
{ref "backend-selection"}[Backend Selection] gives the selection and assurance model, and
{ref "gpu-and-cuda"}[GPU and CUDA] follows an accepted CUDA operation
through storage checks, launch, and backward registration.

# Tape Inspection In VS Code

The pure API exposes a tape as a value that tests can compare. The editor widgets present its
structure visually when the cursor is placed on

```
-- Inspect recorded nodes and the cotangents associated with
-- them.
#tape_view ...
#tape_grads_view ...
#tape_trace_view ...
```

The first shows operation nodes and parent edges. The second evaluates a scalar-output reverse pass
and annotates gradients. The third exposes traversal order. In the branched example above, these
views show both products contributing to the leaf. With `detach`, the second product remains in
the forward graph but contributes nothing to the leaf's cotangent. The widget
reads a runtime artifact; it does not alter execution and it does not prove the tape correct. See
{ref "widgets"}[Interactive Widgets].

# Eager And Typed Graph Training Comparison

The same comparison can be made through the trainer, using a fixed seed, dataset, and arithmetic:

```terminal
# Keep seed and data fixed for the two-step execution
# comparison.
lake exe torchlean quickstart_mlp \
  --device cpu --execution eager --steps 2 --seed 2026
```

```terminal +output
== Quickstart: simple MLP training (seed=2026, steps=2) ==
target(heldout)    = [0.200000]
untrained(heldout) = [-0.088261]
dataset size = 25
mean_loss(before training) = 0.495227
step 0: loss=0.250488
mean_loss(after training) = 0.392821
steps=2 arithmetic=native scalar=Float32
  loss=0.495227 -> 0.392821
trained(heldout) = [0.019031]
```

Replacing `--execution eager` with `--execution typed-graph` prints the same nine lines, digit for
digit. For this model, changing the execution mode changes whether the loss is recorded once and
reused or rebuilt each step. The architecture, seed, data, and arithmetic remain fixed. The
matching transcript checks the displayed results of this run. The graph trainer still constructs
a tape for each evaluation of its stored graph, so this comparison does not establish a timing or
allocation improvement. {ref "execution-modes"}[Execution Modes] explains that reuse and compares
mode and arithmetic choices through their displayed results.

The two-step log reports a dataset mean falling from `0.495227` to `0.392821`. Its `step 0` line
is a single update loss, and the held-out value `0.019031` is a prediction with target `0.2`.
These quantities should not be ranked against one another as if they were three estimates of the
same loss. Matching their displayed values between execution modes checks several observable
results while leaving internal values, timings, and behavior on other inputs unmeasured.

# Runtime Debugging

A gradient discrepancy can originate in the mathematical reference, the recorded program, the
state supplied to it, or the provider that executes it. These checks separate those possibilities.

1. Derive a small reference case, such as $`4x`, including the reduction and output seed.
2. Print the arithmetic semantics, execution mode, device, and selected capsules, so you know which
   artifacts are even in play.
3. Inspect the forward tape and check that the branch you care about is reachable. A missing key in
   the gradient map and a zero value in it mean different things.
4. Check the output cotangent's shape and values. A wrong seed produces a perfectly correct
   derivative of the wrong scalar.
5. Check `detach`, train or eval mode, and stochastic state. They affect the derivative path,
   forward behavior, or random draw used by the same model source.
6. Compare eager and typed graph CPU execution on the same explicit parameters. Disagreement helps
   locate a boundary; agreement on one case does not rule out a shared bug.
7. Compare a native provider against the IEEE reference path on a small finite case.
8. Compare the gap with an error bound or a more precise reference. A small difference can be a
   semantic bug, and a large one can come from numerical instability.

# Tape Experiments

The hand-built tape also exposes cases that a successful scalar gradient does not distinguish:

1. A third `x*x` branch adds another $`2x` contribution. Its `mul` node names node `0` twice,
   just as the first two do.
2. Setting the leaf's `requiresGrad := false` prevents retention of its cotangent, so
   `rtLeafGrad` reports a missing gradient.
3. Moving `rtDetachNode` from `b` to `a` preserves the derivative because the products are equal,
   but changes the node identifiers on the surviving path.
4. A scalar output independent of the leaf leaves node `0` absent from the gradient map.
5. A `mul` node with incompatible operand shapes causes `TapeM.run` to return an error value
   in `Except String`.
6. Changing a constant only in `rtQuad` makes the typed graph represent a different function.
   Comparing its result with the unchanged tape then detects a construction mismatch.

Before defining the canonical IR we cross one more concrete boundary: a named parameter payload
moving between PyTorch and Lean, in {ref "pytorch-roundtrip"}[Exchanging Models With PyTorch]. The
graph chapters then give that exchanged computation an inspectable semantics, and relate runtime
approximation and verification claims to it.
