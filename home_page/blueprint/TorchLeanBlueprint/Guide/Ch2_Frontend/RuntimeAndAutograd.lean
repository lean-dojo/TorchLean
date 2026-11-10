import VersoManual
import TorchLean.NN.LoRA
import TorchLean.NN.Autograd.Function
import TorchLean.NN.Autograd.Frontend
import TorchLean.NN.Autograd.Differential
import TorchLean.NN.Autograd.Complex
import TorchLean.NN.CLI.Training.Command
import TorchLean.NN.Data.Synthetic
import TorchLean.NN.Data.CausalLM
import TorchLean.NN.Models.KAN
import TorchLean.NN.Models.CNN
import TorchLean.NN.Models.ResNet
import TorchLean.NN.Models.UNet
import TorchLean.NN.Models.Recurrent
import TorchLean.NN.Models.CausalTransformer.LanguageModel
import TorchLean.NN.Models.Mamba
import TorchLean.NN.Models.MaskedPatchReconstructor
import TorchLean.NN.Models.Diffusion.NoisePredictor
import TorchLean.NN.Models.FNO
import TorchLean.NN.Models.PPO
import TorchLean.NN.Builder.MoE
import TorchLean.NN.Sequential.Training
import TorchLean.NN.RL.CLI
import TorchLean.NN.RL.PPO.ActorCritic
import TorchLean.NN.SelfSupervised
import TorchLean.NN.Text.Vocabulary
import TorchLean.NN.Text.GPT2BPE
import TorchLean.NN.Text.Generation
import TorchLean.NN.Trainer.Constructor
import TorchLean.NN.Trainer.Train
import TorchLean.NN.Data.Image
import TorchLean.NN.IR.CustomRule
import TorchLean.NN.IR.Tangent
import TorchLean.NN.IR.Program
import TorchLean.NN.Backend.Graph
import TorchLean.Verify.LibTorch.TreeSum
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.Roles
import TorchLean.NN.Loss
import TorchLean.NN.Session.Random
import TorchLean.Tensor.Op.Pointwise.Real
import TorchLean.Verify.Trainer

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean
open TorchLean
open Tape
open Lean.Elab.Tactic.GuardMsgs.WhitespaceMode (lax)

#doc (Manual) "Runtime and Autograd" =>
%%%
tag := "runtime-autograd"
file := "What-Actually-Runs"
%%%

When `x` is used twice in a multiplication, reverse mode must account for both uses. If two
squares then share that same input, their contributions must meet there too. I'll record this
small branched calculation by hand so we can inspect every parent reference and follow every
contribution.

The eager tape keeps the values and reverse rules from that execution, while a typed graph lets us
record the operation sequence once and supply new values later. Comparing the two shows what the
public calls in {ref "autograd-walkthrough"}[Differentiation By Example] construct.
{ref "run-configuration"}[Choosing How A Model Runs] explains how a trainer's arithmetic and device
are chosen.

Named Lean blocks are elaborated while this page is built, together with checks of their displayed
outputs, but the command-line and PyTorch transcripts record separate runs that the page build
doesn't execute.

# API Entry Points And Runtime Artifacts

Training and a standalone `autograd.grad` call build different runtime artifacts:

:::table +header
*
  * You call
  * What is built
  * Typical use
*
  * `Trainer`
  * one eager tape per sample in a reused session, on the CPU or CUDA
  * ordinary training
*
  * `autograd.grad` and `autograd.vjp`
  * a typed graph for each transform invocation
  * differentiating a function or a loss
*
  * `Tape`
  * a tape you own, as pure data
  * inspection, tests, and this chapter
:::

The `autograd.*` transforms don't consult the trainer's settings. In
{src "TorchLean/NN/Autodiff.lean"}[`Autodiff.lean`], `gradients` uses the
general `vjp` path with a unit scalar seed. That path calls `lowerToTypedGraph`, then `vjpChecked`,
whose compiled implementation uses checked graph execution and saved reverse programs (a proved
compiler simplification replaces its raw-tape specification). Scalars use the same lowering as
other tensor outputs. That graph describes the reusable computation, while each invocation supplies
its own values and saved local rules. Higher-order transforms in the same file lower graphs to
tapes over dual scalars, so the reverse calculation also carries directional derivatives. The
trainer doesn't use these transforms: it records an eager tape.

So comparing `autograd.grad` with an eager training step compares two execution paths, and a
discrepancy could come from either path or from the state supplied to it. Inspecting only the eager
tape would leave the transform's typed graph unexamined.

# Runtime Artifact Types

Each type's parameters reveal which information callers must supply statically and which
information the representation stores internally:

```lean (name := rtTypes)
-- Inspect the static information retained by each runtime
-- artifact.
#check @Tape
#check @TypedGraph
#check @IR.Graph
#check @Backend.GraphKernelPlan
```

```leanOutput rtTypes (whitespace := lax)
Tape : Type → Type
```

```leanOutput rtTypes (whitespace := lax)
TypedGraph : (α : Type) → [Storage α] →
  List Shape → Shape → Type
```

```leanOutput rtTypes (whitespace := lax)
IR.Graph : Type
```

```leanOutput rtTypes (whitespace := lax)
Backend.GraphKernelPlan : Type
```

`Tape V` takes the type of the values it stores and nothing else: the CPU stores
`SomeTensor α`, CUDA stores device buffers, and the nodes, parents, and reverse sweep are the
same for both. Each stored value still carries its own shape, but the tape's type does not record a
context of node shapes, so agreement between a node and the parents named by its identifiers is
checked at runtime. That is a deliberate trade, and we come back to it below.

`TypedGraph α Γ τ` carries a context `Γ : List Shape` and an output shape `τ`. A node cannot
refer to a parent at the wrong shape, because the shape is part of the graph's type. This is the
same trick the sequential model language uses in {ref "sequential"}[Sequential Models: One
Architecture, Several Meanings], applied one level lower.

`IR.Graph : Type` has no `α`. Since the canonical IR isn't parameterized by a scalar type,
the same graph can be interpreted over the reals, over intervals, and over binary32 without being
rewritten. See {ref "graphs-and-ir"}[Graph IR].

`Backend.GraphKernelPlan : Type` has no scalar or static shape indices. It holds node
identifiers and capsule metadata, and it contains no tensors whatsoever. A plan cannot witness that
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
  * `TypedGraph`
  * repeated runtime execution
  * shape-indexed forward, JVP, and VJP functions
*
  * `IR.Graph`
  * inspection and verification
  * registered operators, ports, and references
*
  * `Backend.GraphKernelPlan`
  * kernel selection and audit
  * source node ids and selected capsule metadata
:::

A CUDA Graph capture is a separate device launch-and-replay mechanism, and TorchLean doesn't use
it. None of the other boundaries is crossed implicitly either. An accepted graph kernel plan says
nothing about execution: the runtime doesn't execute from it, and the eager session selects and
runs an implementation per operator. And an IR semantics theorem needs a provider refinement result
before it applies to a native tape node.

The four type signatures divide the debugging problem into useful pieces. A wrong tensor value
belongs to an execution artifact such as the tape or evaluated graph context. A wrong operator
belongs to the canonical IR. A surprising provider belongs to the kernel plan or to the eager
session's selection for that operator. None of the last two objects contains the floating-point
intermediate that caused a loss to become nonfinite. Keeping these objects distinct tells us which
additional record to request instead of treating every representation called a graph as
interchangeable.

The scalar parameter also has consequences. One shape-indexed program can be interpreted with
different scalar operations, but a tape already contains values of its chosen `α`, so changing
arithmetic means evaluating with the other scalar interpretation, not relabeling those stored
values. Shape agreement survives that choice, while numerical agreement requires a separate
argument.

# Manual Tape Construction

`Tape V` is a grow-only array of nodes. Each operation takes a tape and returns the extended tape
with the new node's identifier, inside `Except String`, so building a tape is a pure computation
and its errors are values. `Tape.apply` records a registered operator of `TorchLean.Op`, given the
node's name, the operator and its parents' node identifiers. The operator's rule in `Op.rule?`
supplies the forward value and the local backward rule, while the CUDA tape records the same
operator through its entry in `LibTorch.Kernel.ofOp` (`LibTorch.Tape.apply`). For a single value
output the result is `(tape, (nodeId, ()))`, so the final `_` ignores the terminating `Unit`. To
examine the node structure, record two elementwise products sharing the same input, followed by
their sum:

$$`z = x \odot x + x \odot x, \qquad \ell = \textstyle\sum_i z_i`

Here $`\odot` denotes elementwise multiplication, $`z` has the same shape as $`x`, and $`\ell`
reduces it to a scalar. Our code uses $`x=(3,-1.5)`:

```lean (name := rtTape)
-- Record both square branches separately so their shared
-- input is visible.
/-- `z = x*x + x*x`, summed, built on the eager tape. -/
def rtTwice :
    Result (Nat × Tape (SomeTensor Float)) := do
  let (t, x) := Tape.leaf Tape.empty
    (SomeTensor.ofTensor ([3.0, -1.5] : Tensor Float [2]))
    (name := "x")
  let (t, a, _) ← Tape.apply t "mul" (Op.mul (s := [2]))
    (x, x, ())
  let (t, b, _) ← Tape.apply t "mul" (Op.mul (s := [2]))
    (x, x, ())
  let (t, z, _) ← Tape.apply t "add" (Op.add (s := [2]))
    (a, b, ())
  let (t, out, _) ← Tape.apply t "sum"
    (Op.reduce (s := [2]) .sum [⟨0, by decide⟩] false
      (by decide)) (z, ())
  pure (out, t)

#eval rtTwice.map fun (out, tape) => (out, tape.size)
```

```leanOutput rtTape (whitespace := lax)
Except.ok (4, 5)
```

Five nodes, and the scalar output is node `4`. Each operation appends exactly one node and returns
its identifier, so identifiers are array indices and reverse-mode traversal is a loop from
`size - 1` down to `0`. Nothing here is hidden state: the tape is a value passed from one call to
the next.

Ask the tape what it recorded:

```lean (name := rtNodes)
-- Print parent positions, including repeated uses of the
-- same node.
#eval rtTwice.map fun (_, tape) =>
  tape.nodes.map fun (node : Node _) =>
    (node.name, node.parents)
```

```leanOutput rtNodes (whitespace := lax)
Except.ok #[(some "x", #[]), (some "mul", #[0, 0]),
  (some "mul", #[0, 0]), (some "add", #[1, 2]),
  (some "sum", #[3])]
```

The leaf has no parents. Both multiplications name node `0` twice (once for each factor), which is
literally what `x * x` means, and the addition names both products. This printout is the graph, and
it came out of the same data structure the trainer uses.

A node also carries the piece we didn't print, because it's a closure, not data:

```
-- A local rule may emit several contributions to the same
-- parent.
backward : SomeTensor α → Result (Array (Nat × SomeTensor α))
```

Given an upstream cotangent for the node's value, the closure returns one contribution per parent.
That signature is the whole local interface of reverse mode, and nothing in it mentions the rest of
the graph, which is why local derivative rules can be stated and proved one operation at a time.

`(4, 5)` pairs the output node identifier with the tape length (node identifiers start at zero, so
the scalar sum at node `4` is the fifth recorded value). Neither number is the loss itself. At the
chosen input, each square branch contains `[9, 2.25]`, so adding them gives `[18, 4.5]` and the
scalar sum is `22.5`. That gives us a forward reference before we inspect any derivative.

Keeping both parent entries `[0, 0]` on each `mul` is intentional: the two operands refer to the
same input, but they occupy different argument positions in the multiplication, and a reverse rule
owes one contribution for each position. Deduplicating that parent list without changing the rule
would lose half of the derivative of a square. Reusing a value and using it only once are different
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
def rtLeafGrad
    (built : Result (Nat × Tape (SomeTensor Float))) :
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

Each `mul` returns two contributions to node `0` (one for each occurrence of `x`), giving four
contributions in total. If later contributions overwrote earlier ones instead of adding, the result
would be wrong. Accumulation happens in exactly one place, `SomeTensor.add` in
{src "TorchLean/NN/Tape/CPU.lean"}[`Tape/CPU.lean`], which checks
that the two shapes agree before adding. Reverse-mode AD on a DAG is a sum over paths
{Informal.citep baydin2018}[], and this one function is where the sum is taken.

This is also why TorchLean's autograd proofs come in two layers, not one. There are primitive
derivative facts (one per operation), and there is a global soundness statement about the traversal
that composes them. A library that proves only the first layer has proved that its rules are right
and said nothing about whether they are combined correctly. {ref "autograd-proofs"}[Proving
Autograd Correct] is where both layers are stated and discharged.

## PyTorch Gradient Accumulation

PyTorch accumulates the two branch contributions within a backward pass as well
{Informal.citep pytorch2019}[]. Detaching the second product removes its contribution:

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

The first line agrees with the tape we built, and the stop-gradient nodes below explain the second.

Because a backward closure's result is an array of contributions, not a map with one entry per
parent, multiplication can return two entries for node `0` and the traversal can add contributions
from both square branches later. At `x = [3, -1.5]`, one square contributes `[6, -3]` and the
second contributes the same vector. Their sum explains every coordinate of `[12, -6]` without
relying on the implementation's own gradient formula as an oracle.

Since the scalar seed is `1`, this is the gradient of the displayed scalar sum. A different scalar
seed would scale all the contributions, while a vector-output seed would choose a weighted sum of
outputs and would need the output's shape. Runtime shape checks protect that interface even though
node identifiers themselves are plain natural numbers.

# Stop-Gradient Nodes

`detach` is often described as "turning off gradients", which suggests a mutable switch. In this
engine it is a node like any other, and the node is short enough to read in full. Here is the
ordinary `Float` case of what `Session.detach` in
{src "TorchLean/NN/Session/Apply.lean"}[`Session/Apply.lean`]
appends to the CPU tape, transcribed into the pure API. The generic session also clears scalar
differentiation metadata when the carrier has it. On CUDA it records the entry of `Op.detach`.

```lean (name := rtStop)
-- Preserve the second branch value while returning no
-- parent cotangents.
/--
The stop-gradient node, built the way the eager session
builds it: the forward value is reused, the parent edge
stays, and the local rule returns no contributions.
-/
def rtDetachNode (tape : Tape (SomeTensor Float))
    (id : Nat) :
    Result (Tape (SomeTensor Float) × Nat) := do
  let value ←
    Tape.requireValue (α := Float) (t := tape) (s := [2]) id
  pure <| Tape.addNode tape
    { name := some "detach"
      value := SomeTensor.ofTensor value
      requiresGrad := false
      parents := #[id]
      backward := fun _ => .ok #[] }

/-- The same computation, second branch stopped. -/
def rtStopped :
    Result (Nat × Tape (SomeTensor Float)) := do
  let (t, x) := Tape.leaf Tape.empty
    (SomeTensor.ofTensor ([3.0, -1.5] : Tensor Float [2]))
    (name := "x")
  let (t, a, _) ← Tape.apply t "mul" (Op.mul (s := [2]))
    (x, x, ())
  let (t, b, _) ← Tape.apply t "mul" (Op.mul (s := [2]))
    (x, x, ())
  let (t, bStop) ← rtDetachNode t b
  let (t, z, _) ← Tape.apply t "add" (Op.add (s := [2]))
    (a, bStop, ())
  let (t, out, _) ← Tape.apply t "sum"
    (Op.reduce (s := [2]) .sum [⟨0, by decide⟩] false
      (by decide)) (z, ())
  pure (out, t)

#eval rtLeafGrad rtStopped
```

```leanOutput rtStop (whitespace := lax)
Except.ok [6.000000, -3.000000]
```

`2x` instead of `4x`, matching PyTorch's second line to the digit. Three fields did that work.
`backward` returns an empty array, so no contribution flows. `parents` still names the source, so
the forward edge remains visible to anything that reads the graph. And `requiresGrad := false`
makes parent accumulation skip this node. The forward tensor value is reused without recomputing or
copying its payload, so the forward result is unchanged.

Printing the node list makes the difference visible:

```lean (name := rtStopNodes)
-- Keep stopped forward nodes visible even though reverse
-- propagation cannot cross them.
#eval rtStopped.map fun (_, tape) =>
  tape.nodes.map fun (node : Node _) =>
    (node.name, node.parents)
```

```leanOutput rtStopNodes (whitespace := lax)
Except.ok #[(some "x", #[]), (some "mul", #[0, 0]),
  (some "mul", #[0, 0]), (some "detach", #[2]),
  (some "add", #[1, 3]), (some "sum", #[4])]
```

Six nodes this time, and the addition's parents are `1` and `3`: the live product and the detached
copy. The tape still knows the whole forward computation. Only the reverse rule changed.

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
The sparse map distinguishes an absent gradient from a gradient whose entries are zero. A missing
key means no gradient was retained for that node, including when `requiresGrad` disables it. A key
holding zeros means the accumulated runtime cotangent is zero, which can also arise from
cancellation or rounding, so an analytic derivative claim needs the usual correctness hypotheses.

Stopping the second branch leaves the forward scalar at `22.5`: the same two arrays are still
added. The reported cotangent becomes `[6, -3]` because only the first square sends a contribution
back to the input. That's the chosen stop-gradient rule. But the ordinary derivative of the
unchanged real forward function $`2\sum_i x_i^2` is still $`4x`. The printed node list keeps the
stopped branch visible, which is useful when checking why a forward result survived a change to
differentiation behavior.

# Differentiation Through The Transform API

So far we have a hand-built answer. Compare it with the answer the public transform API gives,
which takes an entirely different route: it lowers the program to a typed scalar graph and
differentiates that. Write the program once against the operation interface, so that the same text
can be interpreted either way:

```lean (name := rtTransform)
-- Express the same two branch choices through the public
-- differentiation interface.
/-- `sum (x*x + x*x)`, for the transform API. -/
def rtQuad : autograd.Function [2] [] :=
  fun x => do
    let a ← Ops.apply .mul (x, x, ())
    let b ← Ops.apply .mul (x, x, ())
    let z ← Ops.apply .add (a, b, ())
    Ops.Ref.cast
        (Shape.reduce_finRange _) <$>
      Ops.apply
        (.reduce .sum (List.finRange _) false
          (fun h => by rcases h with h | h <;> cases h))
        (z, ())

/-- The same program, second branch stopped. -/
def rtQuadStopped : autograd.Function [2] [] :=
  fun x => do
    let a ← Ops.apply .mul (x, x, ())
    let b ← Ops.apply .mul (x, x, ())
    let bStop ←
      Ops.apply .detach (b, ())
    let z ←
      Ops.apply .add (a, bStop, ())
    Ops.Ref.cast
        (Shape.reduce_finRange _) <$>
      Ops.apply
        (.reduce .sum (List.finRange _) false
          (fun h => by rcases h with h | h <;> cases h))
        (z, ())

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
this input. That comparison checks several implementations of the same branch structure. If only
one result changes, the investigation narrows to its construction or execution path, but shared
rules can still produce shared errors.

A useful independent reference here is the algebraic calculation above: two square branches give
`4x`, and the chosen stop-gradient rule on one branch leaves `2x`. For a more complicated program,
derive a small special
case or a directional check before comparing two execution routes that reuse the same rules.

# Eager And Graph Interpreters

What lets the same source text be executed eagerly or recorded into a graph isn't a flag inside the
runtime. It's two instances of one class. `Ops` is the operation interface that models call
registered operators through (`Ops.apply`), and the trainer provides two instances of it, in
{src "TorchLean/NN/ScalarTrainer/EagerOps.lean"}[`Trainer/EagerOps.lean`] and
{src "TorchLean/NN/ScalarTrainer/GraphOps.lean"}[`Trainer/GraphOps.lean`].
You can see the difference between them in a single field, `Ref`, which says what a handle to an
intermediate value actually is:

```lean (name := rtRefs)
-- These equalities identify interpreter reference types
-- without evaluating a model.
example :
    Ops.Ref (SessionM Float) Float [2] =
      TensorRef Float [2] := rfl

example {Δ : Type} {Γ : List Shape} :
    Ops.Ref
        (TypedGraph.GraphM.MWith Float Δ Γ) Float [2] =
      TypedGraph.GraphM.Var [2] := rfl
```

Both equalities hold by reduction, as the `rfl` proofs show. Under the eager instance a reference
is a `TensorRef` (an identifier into the session's tape), and each operation records its registered
operators with `Session.apply`, which takes one operator with exactly one value output per
call, executes it immediately and appends its node. Under the graph instance a reference is a
`GraphM.Var` (an SSA variable), and each operation records a node without computing anything. A
third instance, `IR.RecordM`, records the same program into the verification
IR when a model is lowered, appending registered operators to a typed graph.

This is the tagless-final pattern {Informal.citep kiselyov2012}[]: one object language, several
interpreters, with the object language's types carried by the host language. Every layer of a
sequential model uses the same design: one forward program, which each interpreter runs. A shared
operation interface keeps the program text aligned, while the interpreters still need
correspondence checks for the operations they implement.

What a rule decides is visible on the eager CPU tape and in the typed graph, because both run it.
On CUDA, the eager tape runs the operators' LibTorch entries instead. Two inputs of a node that are
one handle use the operator's alias rule, chosen by node identity and not by equal values: `max` or
`min` of a handle with itself gives each entry the average of its cotangent with itself, while two
handles that hold equal values get the rule's two halves, whose sum can differ in the last bit at a
subnormal cotangent. A node sends its contributions in its rule's `order` (for the fused multi-head
attention, the input's gradient comes first), and the reverse sweep adds them in that order. A
smooth max pool with `beta = 0` is refused by the eager CPU tape, which validates the inputs before
it records. A typed graph does record it, and checked execution refuses it when it validates the
inputs. A convolution with leading axes is one node on the CPU tape and in the typed graph. When
the input must be reshaped to the batch shape of the CUDA entry, the CUDA tape folds the leading
axes into the batch axis instead: a reshape, the convolution and a reshape, recorded as one tape
transaction that keeps all three nodes on success and restores the tape on failure. Batched
attention is one node with its own reduction order, so its rounded parameter gradients can differ
from those of one single-sequence node per sequence. The one-ulp difference observed in the
projection weights' gradients is an observation on the tested inputs, not an error bound.

The maintained eager handle also records a session owner and generation. A reset can reuse numeric
node `0`, so its number alone can't establish that an old handle belongs to the new recording.
Before accepting a session reference, the runtime checks those fields. This protects a different
property from shape safety: two tensors may both have shape `[2]` while only one belongs to the
current execution.

# Typed Graph Execution

Typed graph execution records the model's scalar loss once as a typed SSA graph, whose nodes carry
the forward behavior and the executable rules used for JVPs and VJPs. Each evaluation supplies
current parameters and inputs to that recorded graph. A separate typed reference selects the
result, so the graph may return an input or an earlier node without adding a dummy final operation,
and forward and reverse execution read and seed that exact reference.

A typed graph's indices enforce shape agreement between node references, but they don't establish
derivative correctness. Each node's executable payload holds three functions (`forward`, `jvp` and
`vjp`) with runtime validation and optional certified preparation. Its preparation certificate
preserves the forward, validation, and dense VJP programs. It supplies no derivative-correctness
law. The proof-carrying node in
{src "TorchLean/NN/Tape/Graph.lean"}[`TorchLean/NN/Tape/Graph.lean`]
adds the local adjointness law relating its `jvp` and `vjp`, and the proof-carrying graph composes
those local laws into a graph theorem. Execution can proceed without this adjointness certificate,
while the graph verification theorem requires it. {ref "autograd-proofs"}[Proving Autograd Correct]
follows that ladder up.

At the model level, the public name specializes the context of this same graph type:

```lean (name := rtWrapper)
-- Model state precedes the input in the lowered graph
-- context.
example {α : Type} [Storage α]
    {stateShapes : List Shape} {σ τ : Shape} :
    nn.TypedGraphModel stateShapes σ τ α =
      TypedGraph α (stateShapes ++ [σ]) τ := rfl
```

`nn.TypedGraphModel` is an abbreviation whose context is model state followed by one input tensor.
It introduces no additional runtime representation: the equality unfolds to `rfl`.

{src "TorchLean/NN/TypedGraph/Compiled.lean"}[ `TypedGraph.compileChecked`] returns a `Compiled`
execution containing an indexed primal context and saved local VJP programs.
`Compiled.backwardDenseFrom` accepts a seed pack for the complete context. The ordinary checked and
pure TypedGraph VJP APIs use this path through proved compiler simplifications, while the
reverse-mode Jacobian transforms in `Autodiff.lean` use it directly. `Compiled.toTape` and
`lowerToTapeChecked` retain the raw Tape interface. `TypedGraph.compileChecked_eq` identifies the
complete checked error/result with that of tape lowering, and the theorems of
{src "TorchLean/NN/Tape/Lowering/Compiled.lean"}[`Lowering/Compiled.lean`] identify every dense
gradient with full graph backpropagation, under `Storage` and `Add` and without assuming
associative addition. They preserve the stored executable rules, but mathematical derivative
correctness and agreement with GPU kernels remain separate obligations.

Saved backward shares uniform contribution histories while preserving each dense scalar addition,
including the complete local contribution of a node with repeated parents. Compression uses the
certified carrier equality in `Storage.decEq?`, which native `Float`, `Float32`, and `UInt8`
provide. Other storage instances default to dense backward. This equality distinguishes signed
zeros but does not require raw NaN sign or payload identity. Custom nodes without certified local
preparation or compact contributions retain materializing adapters, and noncompressible histories
can require quadratic replay. The raw Tape adapter also retains full prefix contributions.

Construction and final disposal can require quadratic work because the graph representation
retains full shape-prefix lists. Efficient reusable execution does not imply a linear graph
lifecycle or a general complexity theorem.

Recording and lowering do not imply the optimization, fusion, scheduling, or native code generation
associated with a compiler such as `torch.compile`.

# Canonical IR Representation

Typed graph nodes hold Lean functions. Those functions can execute an operation, but they do not
expose its structure as inspectable data. `IR.Graph` instead represents operations such as
linear, ReLU, reduction, normalization and shape transforms as data. An operator node records a
registered operator `TorchLean.Op ins outs`, indexed by its input and output ports, and the
references (`IR.Ref`, a node and an output position) of the values it reads. Its output ports
follow from that operator (`IR.Node.outs`). Constants (parameters among them) are looked up
by node id in the separate payload (`IR.Payload`).

Because the operator is data (a constructor of `TorchLean.Op` with its attributes), an importer can
validate it (`TorchLean.Op.check`), a verifier can interpret it, and a code generator can reject
what it does not support without executing an arbitrary function. The
scalar-independent representation also permits several interpretations:
{ref "graphs-and-ir"}[Graph IR] builds
one small graph and evaluates it over the reals, over intervals, and over IEEE binary32 scalars
without touching the graph.

The canonical IR is executed by its own evaluator, `IR.Graph.eval`, which runs the graph
forward: an operator is invoked on the values its references name and returns its rule's forward
values (`TorchLean.Op.invoke`, `Op.rule?`), and a custom operation runs the checked
program that the payload holds for its node. Those same rules carry pullbacks and (for some
operators) forward-mode rules, but the evaluator computes no gradients: the reverse sweep belongs
to the eager tape. `IR.Graph.tangent` appends tangent calculations using a separate set of graph
rewrite rules. No theorem yet connects those rewrites to `Op.Rule.jvp?` or to a derivative, so a
tangent graph is not itself a checked derivative proof. An operator's correctness certificates are
separate theorems attached by the `op_cert` attribute.

Both paths honor the homogeneous-element contract from {ref "tensors-shapes"}[Tensors And Shapes]:
a typed graph is parameterized by one `α`, and an IR evaluation (`IR.Graph.eval`) receives
one payload over `α` and returns tensors over that same `α` at its value, frozen and detached
ports.

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

The pack type preserves that dependent list. A tape can't, because `Tape V` has no shape index, and
neither can a registry that has to iterate over parameters by name. Both store `SomeTensor α`
(a structure holding a shape and a tensor indexed by that shape). So the collection can hold
different shapes while each entry retains its own shape-value relationship.

```lean (name := rtPack)
-- Recovery needs a shape equality. The second declaration
-- proves a pack/cast roundtrip.
#check @SomeTensor.cast
#check @SomeTensor.ofTensor_cast
```

```leanOutput rtPack (whitespace := lax)
@SomeTensor.cast : {α : Type} →
  [inst : Storage α] → {shape : Shape} →
  (value : SomeTensor α) →
  value.shape = shape → Tensor α shape
```

```leanOutput rtPack (whitespace := lax)
@SomeTensor.ofTensor_cast : ∀ {α : Type}
  [inst : Storage α] (value : SomeTensor α)
  {shape : Shape} (h : value.shape = shape),
  SomeTensor.ofTensor (value.cast h) = value
```

First, `cast` takes a proof. Its last explicit argument is `value.shape = shape`, so there is no way
to recover a typed tensor from a packed one without producing evidence that the stored shape is the
expected shape. A runtime comparison can supply that evidence, but `rfl` cannot establish an
arbitrary packed tensor's shape:

```lean +error (name := rtNoProof)
-- An arbitrary packed shape cannot be identified with [2]
-- by reflexivity.
example (value : TorchLean.SomeTensor Float) :
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

Second, `ofTensor_cast` says that casting a packed tensor to the matching shape and repacking it
returns the original packed value. Its stored shape and values survive the round trip.

The instantiated runner also holds parameter
names, bounded token inputs, RNG state, optimizer memory, mutable model buffers, and the train or
eval flag.

The cast signature takes an equality from the stored shape to the requested shape. Its direction
matters: it transports the existing tensor without reshaping, truncating, padding, or
reinterpreting the buffer. The roundtrip theorem says that packing a tensor and recovering it with
the matching shape returns that tensor. It doesn't manufacture an equality for arbitrary packed
data.

In the diagnostic above, the printed metavariable names are elaborator details. Its useful part is
the demanded equality with `[2]`.

# Model Mode And Random State

Some operations depend on mode. Dropout samples a mask during training and applies its
deterministic inference behavior during evaluation {Informal.citep dropout2014}[]. Batch
normalization may update running statistics during training and use stored statistics at evaluation
{Informal.citep batchnorm2015}[], and other stochastic or stateful layers follow the same
pattern. `trained.predict` uses the retained runtime state in evaluation mode, and a
`Trainer.Session` follows the same rule in a caller-driven loop (`step` in training mode, `predict`
and `loss` in evaluation mode). Mode is not a backend: one eager CUDA runner can switch modes
without changing device or provider profile. {ref "run-configuration"}[Choosing How A Model Runs]
measures the size of the resulting difference, which is the part that matters when you are trying
to tell a mode bug from rounding noise.

For the tape, randomness raises a replay requirement. A stochastic forward pass and its reverse
pass must use the same random draw, or the reverse pass computes the derivative of a function
nobody evaluated. Replay through the tape interface is deterministic, but correct capture of the
forward mask is still an obligation of the node implementation:

```lean (name := rtPure)
-- The result may report a malformed tape even though
-- backward has no IO effects.
#check @Tape.backwardScalar
```

```leanOutput rtPure (whitespace := lax)
@backwardScalar : {α : Type} →
  [inst : Storage α] → [Add α] → [One α] →
  Tape (SomeTensor α) → ℕ →
    Result (Std.HashMap ℕ (SomeTensor α))
```

No `IO`: in the pure Lean model, replay depends on the tape and its stored closures, not on an
external generator. A closure could still use an incorrect deterministic mask or an incorrect VJP,
so purity alone doesn't prove it differentiates the recorded forward pass. Explicit capture and
local correctness laws supply that connection when treating reverse mode as a program on immutable
data {Informal.citep pearlmutter2008}[]. And the grow-only array behind it is efficient for the
same reason Lean's own runtime can afford immutable structures
{Informal.citep immutablebeans2019}[].

This also affects checkpoints. Inference can require persistent
buffers, mode, and preprocessing as well as parameters. Reproducing the next training update needs
additional state. Replaying that update also needs the generator state, the optimizer
state, and the loader position.

The pure backward type still returns `Result`: lack of `IO` doesn't mean that every supplied tape
is valid. The traversal can reject an invalid output id, an incompatible seed, or a parent
contribution with the wrong shape. Purity describes how the computation obtains its inputs, while
validation describes which of those inputs it accepts.

The eager reverse engine also distinguishes an unreached node from a reached node with a zero
cotangent: it skips unreached local rules and fills missing entries with zeros when a dense result
is requested. This matters at singular values: evaluating a disconnected rule can produce an
expression such as `0 * (1 / 0)`, which need not be zero in floating-point arithmetic. So the
bridge to a traversal that visits every node needs a zero-preservation condition. That is a
mathematical obligation about the rules, not just a choice of array or hash-map storage.

# Activation Recomputation

Saving model state to a checkpoint and checkpointing an activation graph solve different problems.
Activation checkpointing discards intermediate values during the forward pass and recomputes them
during the reverse sweep on a schedule that trades arithmetic for memory
{Informal.citep griewank2000}[]. The tape examples here retain their recorded values, so they
don't demonstrate that memory optimization.

Recomputation must reproduce the forward values used by the local reverse rules, and for a
stochastic operation that includes replaying the same random draw. Agreement of the forward
function alone does not establish a memory saving or validate a recomputation schedule.

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

For a supported abstract graph, the first ladder can be climbed entirely inside Lean, and
{ref "autograd-proofs"}[Proving Autograd Correct] climbs it: local adjointness laws compose into a
graph theorem, the executed dense reverse sweep is identified with that theorem's backpropagation,
and over the reals the result is identified with the adjoint of a Fréchet derivative from Mathlib
{Informal.citep mathlib2020}[]. The second ladder may be a theorem, a sound checked guard, a
numerical certificate, or an explicit trusted boundary. Native CUDA, LibTorch, the compiler, the
driver, and the hardware need their own refinement evidence or stated trust assumptions.

Relevant proof sources:

- {srcDir "TorchLean/NN/Tape"}[`TorchLean/NN/Tape`]
- {src "TorchLean/NN/Tape/Graph.lean"}[`TorchLean/NN/Tape/Graph.lean`]
- {srcDir "TorchLean/NN/Tape/Lowering"}[`Tape/Lowering`]

# Kernel Selection Records

Before an eager operator is recorded, the session's profile selects an accepted capsule for that
operator, and the capsule must name the provider and device of the session's tape: a reference CPU
capsule records the operator's rule with `Tape.apply`, a LibTorch CUDA capsule its
`LibTorch.Kernel.ofOp` entry. Any other capsule, or an operator without an entry, is an error. Run

```terminal
# Inspect the capsules selected for a short eager training
# run.
scripts/lake.sh exe torchlean quickstart_mlp \
  --device cpu --steps 1 --seed 2026 --show-backend
```

and each capsule name is reported the first time the session selects it. Here is the summary line of
the ReLU entry from the recorded run, rewrapped to fit the page:

```
  relu: reference.relu provider=reference trust=checked
    vjp=torchlean-tape reduction=n/a
```

{ref "run-configuration"}[Choosing How A Model Runs] reads the full report, tabulates the seven
capsules a CPU run selects, and explains the evidence lines and the empty banners. In the ReLU
line, the field `vjp=torchlean-tape` names which artifact owns the reverse rule for that operation.
It connects the tape we built by hand to the selected kernel: the forward value came from the
`reference.relu` provider, and the backward contribution came from TorchLean's own tape node. When
a provider owns its own backward instead, the corresponding derivative claim also depends on that
provider's reverse implementation.

Two deep dives follow a single operation the whole way down.
{ref "backend-selection"}[Backend Selection] gives the selection and assurance model, and
{ref "gpu-and-cuda"}[GPU and CUDA] follows an accepted CUDA operation
through storage checks, launch, and backward registration.

# Tape Inspection In VS Code

In the pure API a tape is a value that tests can compare. The editor widgets present its structure
visually when the cursor is placed on

```
-- Inspect recorded nodes and the cotangents associated with
-- them.
#tape_view ...
#tape_grads_view ...
#tape_trace_view ...
```

The first view shows operation nodes and parent edges, the second evaluates a scalar-output reverse
pass and annotates gradients, and the third exposes traversal order and the final accumulated
cotangents. In the branched example above, these views show both products contributing to the leaf.
With `detach`, the second product remains in the forward graph but contributes nothing to the
leaf's cotangent. See {ref "widgets"}[Interactive Widgets].

# Training Through The Trainer

This command makes a reproducible eager training run, with a fixed seed, dataset, and arithmetic:

```terminal
# Keep seed and data fixed for a two-step run.
scripts/lake.sh exe torchlean quickstart_mlp \
  --device cpu --steps 2 --seed 2026
```

```terminal +output
== Quickstart: simple MLP training (seed=2026, steps=2) ==
target(heldout)    = [0.200000]
untrained(heldout) = [-0.047640]
dataset size = 25
mean_loss(before training) = 0.629263
mean_loss(after training) = 0.491944
steps=2 arithmetic=native scalar=Float32
  loss=0.629263 -> 0.491944
trained(heldout) = [0.041230]
```

Each step records a fresh tape for the current parameters and the step's sample, then runs it
backward. {ref "run-configuration"}[Choosing How A Model Runs] compares arithmetic and device
choices through their displayed results.

Over the two steps, the log reports a dataset mean falling from `0.629263` to `0.491944`. The
held-out value `0.041230` is a prediction with target `0.2`, not another loss measurement.

# Runtime Debugging

A gradient discrepancy can originate in the mathematical reference, the recorded program, the
state supplied to it, or the provider that executes it. These checks separate those possibilities.

1. Derive a small reference case, such as $`4x`, including the reduction and output seed.
2. Print the arithmetic semantics, device, and selected capsules, so you know which artifacts are
   even in play.
3. Inspect the forward tape and check that the branch you care about is reachable. A missing key in
   the gradient map and a zero value in it mean different things.
4. Check the output cotangent's shape and values. A wrong seed produces a perfectly correct
   derivative of the wrong scalar.
5. Check `detach`, train or eval mode, and stochastic state. They affect the derivative path,
   forward behavior, or random draw used by the same model source.
6. Compare eager and typed graph CPU execution on the same explicit parameters. Disagreement helps
   locate a boundary, but agreement on one case doesn't rule out a shared bug.
7. Compare a native provider against the IEEE reference path on a small finite case.
8. Compare the gap with an error bound or a more precise reference. A small difference can be a
   semantic bug, and a large one can come from numerical instability.

# Tape Experiments

Our hand-built tape also exposes cases that a successful scalar gradient doesn't distinguish:

1. A third `x*x` branch adds another $`2x` contribution. Its `mul` node names node `0` twice,
   just as the first two do.
2. Setting the leaf's `requiresGrad := false` prevents retention of its cotangent, so
   `rtLeafGrad` reports a missing gradient.
3. Moving `rtDetachNode` from `b` to `a` preserves the reported cotangent because the products
   are equal, but changes the node identifiers on the surviving path.
4. A scalar output independent of the leaf leaves node `0` absent from the gradient map.
5. A `mul` node with incompatible operand shapes makes `Tape.apply` return an error value in
   `Except String`.
6. Changing a constant only in `rtQuad` makes the typed graph represent a different function.
   Comparing its result with the unchanged tape then detects a construction mismatch.

Before defining the canonical IR we cross one more concrete boundary: a named parameter payload
moving between PyTorch and Lean, in {ref "pytorch-roundtrip"}[Exchanging Models With PyTorch]. The
graph chapters then give that exchanged computation an inspectable semantics, and relate runtime
approximation and verification claims to it.
