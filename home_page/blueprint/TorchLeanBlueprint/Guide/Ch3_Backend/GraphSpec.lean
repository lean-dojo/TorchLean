import VersoManual
import NN.API
import NN.GraphSpec
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.Roles

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean
open TorchLean
open NN
open NN.GraphSpec
open Lean.Elab.Tactic.GuardMsgs.WhitespaceMode (lax)

#doc (Manual) "GraphSpec" =>
%%%
tag := "graphspec"
file := "GraphSpec___-One-Architecture___-Several-Meanings"
%%%

The previous chapter described the MLP by its pure formula:

$$`x\mapsto W_2\operatorname{ReLU}(W_1x+b_1)+b_2.`

Suppose we want to count its trainable scalars before allocating the weights, or replace the
activation while keeping the layers on either side. A plain function does not expose a layer
list to traverse. GraphSpec stores the architecture as data, with shape indices that describe its
parameters, input, and output. Tools can inspect that description and interpret it for several
execution targets.

Every Lean block on this page is elaborated when the guide is built, and every printed value below
was produced by that elaboration. The command transcripts come from the same checkout.

# Architecture Representation

The same layer connections are needed by initialization, forward evaluation, gradient extraction,
and parameter loading. GraphSpec records those connections in a shape-indexed syntax. Its
parameter ABI is the ordered list of tensor shapes that a caller must supply.

The syntax has several interpreters. Its `program` fields use the tagless-final operation
interface ({Informal.citep kiselyov2012}[]), letting an execution interface choose what each
operation does:

```
-- One model description has several interpretations with
-- different proof obligations.
                         ┌── Interp.spec ──> pure tensor function
GraphSpec Chain / DAG ──┤
                         └── toProgram ────> TorchLean.Program
                                                │
                                                ├── eager execution
                                                ├── typed graph lowering
                                                └── broad verification lowering
                                                    ──> NN.IR.Graph
                                                         ──> IRExec.ForwardGraph
```

The upper branch reads the architecture as a pure tensor function. The lower branch constructs a
program whose operations can be interpreted by eager execution or recorded by a graph builder.
These branches share the architecture and parameter layout; an agreement theorem must identify
their meanings where that is claimed.

The optional `toSeq` adapter supplies initialized sequential layers for the supported subset. It is
not required by the pure interpretation or by `toProgram`, and a section below defines a primitive
that has both of those meanings and no layer view at all.

The broad verification lowering validates the IR it produces, but that success is not an
end-to-end semantic theorem for every GraphSpec primitive. The smaller
`NN.Verification.Builtin.Proved.ForwardProgram` language has the corresponding source-to-IR
correctness theorem.

# MLP Chain Definition

I'll use two input features, three hidden features, and one output. In
`Chain ps σ τ`, `ps` is the ordered list of parameter shapes, `σ` is the input shape, and `τ`
is the output shape:

```lean (name := gsMlpDef)
-- The parameter list records both matrix dimensions and
-- their position in the chain.
def gsMlp :
    Chain [[3, 2], [3], [1, 3], [1]] [2] [1] :=
  Chain.linear 2 3 >>>
  Chain.relu [3] >>>
  Chain.linear 3 1
```

That is the width-2, hidden-3, width-1 instance of the library definition. The general one lives in
{src "NN/GraphSpec/Models/Mlp.lean"}[`NN/GraphSpec/Models/Mlp.lean`], and its type states the
parameter ABI for every choice of widths:

```lean (name := gsMlpType)
-- Read the widths and the resulting ordered parameter
-- shapes from the constructor type.
#check @Models.mlp
```

```leanOutput gsMlpType (whitespace := lax)
Models.mlp : (inputWidth hiddenWidth outputWidth : ℕ) →
  Chain
    [[hiddenWidth, inputWidth], [hiddenWidth],
      [outputWidth, hiddenWidth], [outputWidth]]
    [inputWidth] [outputWidth]
```

Read the type from the outside in:

- the chain consumes one tensor of shape `[inputWidth]`;
- it produces one tensor of shape `[outputWidth]`;
- its parameter environment contains exactly four tensors;
- the order is $`W_1`, $`b_1`, $`W_2`, $`b_2`.

For the $`2\to3\to1` model the parameter shapes are:

:::table +header
*
  * Slot
  * Shape
  * Scalars
*
  * $`W_1`
  * `[3, 2]`
  * 6
*
  * $`b_1`
  * `[3]`
  * 3
*
  * $`W_2`
  * `[1, 3]`
  * 3
*
  * $`b_2`
  * `[1]`
  * 1
:::

The four slots contain $`6+3+3+1=13` trainable scalars. Their shapes are available in the chain's
type before allocation, so a tool can compute this count without inspecting initialized tensors
or interpreting parameter names.

Each weight matrix has one row per output of its layer and one column per input. The bias
matches that layer's output width. A caller supplies four tensors in this order, rather than an
undifferentiated collection of thirteen scalars.

# Composition And The Parameter ABI

The composition operator `>>>` does more than connect two arrows. If

```
-- Composition requires the first output shape to equal the
-- second input shape.
g₁ : Chain ps₁ σ τ
g₂ : Chain ps₂ τ υ
```

then

```
-- Parameter lists concatenate in the same order as the two
-- model components.
g₁ >>> g₂ : Chain (ps₁ ++ ps₂) σ υ.
```

The intermediate shape must be the same $`\tau` on both sides, and the parameter lists concatenate
in construction order. In the MLP, the first linear layer contributes `[[3, 2], [3]]`, ReLU
contributes no parameters, and the second linear layer contributes `[[1, 3], [1]]`. Composition
therefore computes the full ABI shown above. A primitive can still misinterpret two same-shaped
slots, as the swapped-bias example below demonstrates.

Change the hidden width of the first linear layer to five while leaving the second layer's input
width at three:

```lean +error (name := gsBroken)
-- The hidden width disagrees at the connection, so
-- elaboration must reject this chain.
def gsBroken : Chain [[5, 2], [5], [1, 3], [1]] [2] [1] :=
  Chain.linear 2 5 >>>
  Chain.relu [5] >>>
  Chain.linear 3 1
```

```leanOutput gsBroken (whitespace := lax)
Application type mismatch: The argument
  Chain.linear 3 1
has type
  Chain [[1, 3], [1]] [3] [1]
but is expected to have type
  Chain ?m.45 [5] ?m.48
in the application
  Chain.relu [5] >>> Chain.linear 3 1
```

The report names the two shapes that disagree, `[3]` and `[5]`, and points at their connection.
Each linear layer is valid on its own, but the first produces five hidden values while the last
expects three. Lean rejects their composition while checking the architecture definition.

One practical note about reading these messages. `>>>` is overloaded: TorchLean also uses it for
sequential layer composition and for `Spec.OpSpec` composition. When a shape error additionally
rules out every candidate, Lean prints one branch per candidate and ends with a failed instance
search for `HShiftRight`. The first branch is the one about chains, and it is the one to read.

## Shape Mismatch In PyTorch

For comparison, here is the same architecture and the same typo in PyTorch
({Informal.citep pytorch2019}[]):

```
# Compare construction-time module sizes with the shape
# error discovered on execution.
import torch, torch.nn as nn

net = nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 1))
print([tuple(p.shape) for p in net.parameters()])
print(sum(p.numel() for p in net.parameters()))
print(list(net.state_dict().keys()))

bad = nn.Sequential(nn.Linear(2, 5), nn.ReLU(), nn.Linear(3, 1))
print(sum(p.numel() for p in bad.parameters()), "scalars")
bad(torch.tensor([0.5, 0.8]))
```

The recorded PyTorch run prints four lines and then raises, with the traceback ending in the last
line
below:

```
[(3, 2), (3,), (1, 3), (1,)]
13
['0.weight', '0.bias', '2.weight', '2.bias']
19 scalars
RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x5 and 3x1)
```

The correct model reports the same four parameter shapes and thirteen scalars as GraphSpec.
The difference is where the broken model gets caught.
`nn.Sequential(nn.Linear(2, 5), nn.ReLU(), nn.Linear(3, 1))` is a perfectly good Python object. It
constructs, it reports nineteen parameters, it can be moved to a device, it can be handed to an
optimizer, and it can be saved. It fails on the first tensor that reaches it.

The two representations make shape information available at different times. This Python object
checks the connection when data reaches the layers, and its `state_dict` identifies parameters by
string keys. GraphSpec requires shapes that the elaborator can see, which can make the types long.
It can then reject the incompatible connection while checking the architecture and expose the
parameter shapes through the type.

# Primitive Specifications And Runtime Programs

The layers in a chain are built from `Primitive`:

```lean (name := gsPrimitive)
-- Inspect the separate specification and runtime fields,
-- including their shape parameters.
#print NN.GraphSpec.Primitive
```

```leanOutput gsPrimitive (whitespace := lax)
structure NN.GraphSpec.Primitive (ps : List Shape)
    (σ τ : Shape) : Type 1
number of parameters: 3
fields:
  NN.GraphSpec.Primitive.name : String
  NN.GraphSpec.Primitive.specFwd : {α : Type} →
    [inst : Storage α] → [Context α] →
      TensorPack α ps → Tensor α σ → Tensor α τ
  NN.GraphSpec.Primitive.program : {α : Type} →
    [inst : Storage α] → [inst_1 : Context α] →
      Runtime.Autograd.Model.Program α (ps ++ [σ]) τ
  NN.GraphSpec.Primitive.toLayerM? :
      Option (ℕ → { l // l.stateShapes = ps }) :=
    none
  NN.GraphSpec.Primitive.countsAsLayer : Bool :=
    false
constructor:
  NN.GraphSpec.Primitive.mk {ps : List Shape} {σ τ : Shape}
    (name : String)
    (specFwd : {α : Type} → [inst : Storage α] →
      [Context α] → TensorPack α ps → Tensor α σ → Tensor α τ)
    (program : {α : Type} → [inst : Storage α] →
      [inst_1 : Context α] →
        Runtime.Autograd.Model.Program α (ps ++ [σ]) τ)
    (toLayerM? : Option (ℕ → { l // l.stateShapes = ps }))
    (countsAsLayer : Bool) : Primitive ps σ τ
```

`specFwd`, `program`, and the optional layer view share the shape indices. `specFwd` is the
mathematical meaning, a plain function on shape-indexed
tensors. `program` is the executable meaning, polymorphic in the execution monad, which is how one
primitive serves eager execution, the typed graph, and the verification path without being written
three times. For a linear primitive, `specFwd` calls the mathematical `linearSpec` while `program`
calls the runtime linear operation with the same parameter shapes.

:::table +header
*
  * Field
  * Role
*
  * `name`
  * a short identifier for summaries and errors
*
  * `specFwd`
  * the pure tensor function used by `Interp.spec`
*
  * `program`
  * the execution-polymorphic tensor program
*
  * `toLayerM?`
  * an optional lowering to an initialized `nn.Layer`
*
  * `countsAsLayer`
  * whether deterministic layer indexing advances here
:::

The shared indices ensure that `specFwd` and `program` accept and return the same shapes.
Agreement on values requires a theorem: a record can contain two functions with the same signature
that compute different answers. The record keeps the intended pair adjacent and gives it one name.
Such a theorem must identify the interpreter and the reference function being compared. The MLP
theorem later in this chapter, for example, compares `Interp.spec` with a pure reference model; it
does not establish agreement with native execution.

The remaining fields serve the sequential-layer adapter. `toLayerM?` can be absent even when both
forward interpretations are present. `countsAsLayer` controls the occurrence index used for
initialization; we will inspect those indices for the MLP below.

## A Primitive Without A Layer View

We can leave out `toLayerM?` in a primitive that doubles its input:

```lean (name := gsDoubleDef)
-- Both interpretations double the input; the optional layer
-- adapter is absent.
/-- Doubling primitive: two meanings, no layer view. -/
def gsDouble (s : Shape) : Primitive [] s s :=
  { name := "double"
    specFwd := fun {_α} _storage _ctx _params x =>
      Tensor.addSpec x x
    program := fun {α} _storage _ctx =>
      fun {m} _ _ =>
        fun x =>
          Runtime.Autograd.Torch.add
            (m := m) (α := α) (s := s) x x }
```

The pure interpretation runs immediately:

```lean (name := gsDoubleRun)
-- Run the pure interpretation with the empty parameter
-- pack.
#eval Interp.spec (Chain.prim (gsDouble [3]))
  (α := Float) .nil [1.0, -2.0, 3.5]
```

```leanOutput gsDoubleRun (whitespace := lax)
[2.000000, -4.000000, 7.000000]
```

The executable interpretation typechecks too, because `Chain.toProgram` only needs the `program`
field:

```lean (name := gsProgramType)
-- The program receives all parameters followed by the
-- ordinary input tensor.
#check @Chain.toProgram
```

```leanOutput gsProgramType (whitespace := lax)
@Chain.toProgram : {ps : List Shape} →
  {σ τ : Shape} →
    Chain ps σ τ →
      {α : Type} → [inst : Storage α] →
        [inst_1 : Context α] →
          Runtime.Autograd.Model.Program α (ps ++ [σ]) τ
```

The sequential view is the one that refuses:

```lean (name := gsToSeqFail)
-- Failure here concerns the optional sequential adapter,
-- not the primitive’s forward map.
#eval
  match ToSequential.toSeq (Chain.prim (gsDouble [3])) with
  | .error e => e
  | .ok _ => "ok"
```

```leanOutput gsToSeqFail (whitespace := lax)
"graphspec.toSeq: primitive `double` has no Seq
lowering (missing toLayerM?); use `Chain.toProgram` if
you only need execution"
```

`toSeq` returns
`Except String` and rejects a chain at the first primitive that has no layer constructor, rather
than inventing parameter initialization, buffer behavior, and layer metadata that the primitive
never specified. The three primitives of the sequential core, and the four in
{src "NN/GraphSpec/Primitives/Spatial.lean"}[`Primitives/Spatial.lean`], all do supply a layer, so
the partiality only shows up for primitives like the one above.

The pure call supplies `.nil` because doubling needs no parameters. In the program interface,
the ordinary input follows the parameter list, as the `ps ++ [σ]` type shows. The absent layer
adapter prevents initialization through `toSeq`; it leaves these two forward definitions
available. Derivative correctness would need its own statement.

# Parameter Order And Shape Checking

For a new primitive, check the parameter order in three places: the type-level list `ps`,
the pattern match inside `specFwd`, and the argument order inside `program`. Shape typing proves
that each slot has the right shape. Two tensors *of the same shape* can still be exchanged.

The demonstration needs a model with two same-shaped biases, so widen the middle layer to match:

```lean (name := gsSquareDef)
-- Equal bias shapes let a role swap pass shape checking in
-- this model.
def gsSquare : Chain [[3, 3], [3], [3, 3], [3]] [3] [3] :=
  Chain.linear 3 3 >>>
  Chain.relu [3] >>>
  Chain.linear 3 3
```

Both linear layers now contribute a `[3]` bias, so a parameter pack with the two biases exchanged
has exactly the same type as the correct one. Feed the model the zero input, identity weights, and
two distinguishable biases, in both orders:

```lean (name := gsSwapRun)
-- Hold the matrices fixed and swap only the bias roles to
-- isolate the ABI mistake.
#eval do
  let identity : Tensor Float [3, 3] :=
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
  let first : Tensor Float [3] := [-1.0, 0.0, 0.0]
  let second : Tensor Float [3] := [1.0, 0.0, 0.0]
  let x : Tensor Float [3] := [0.0, 0.0, 0.0]
  let inOrder := Interp.spec gsSquare
    (.cons identity (.cons first
      (.cons identity (.cons second .nil)))) x
  let exchanged := Interp.spec gsSquare
    (.cons identity (.cons second
      (.cons identity (.cons first .nil)))) x
  IO.println s!"in ABI order = {inOrder}"
  IO.println s!"exchanged    = {exchanged}"
```

```leanOutput gsSwapRun (whitespace := lax)
in ABI order = [1.000000, 0.000000, 0.000000]
exchanged    = [0.000000, 0.000000, 0.000000]
```

Both packs elaborate. The model computes $`\operatorname{ReLU}(x+b_1)+b_2`, so the correct order
gives $`\operatorname{ReLU}(-1)+1=1` in the first coordinate and the exchanged order gives
$`\operatorname{ReLU}(1)-1=0`. Exchanging the biases changes the function while preserving every
tensor's type. The activation between the biases makes their positions observable in the output.

The same swap passes PyTorch's checkpoint-loading checks:

```
# A state dictionary can accept all keys while placing
# same-shaped values in wrong roles.
same = nn.Sequential(nn.Linear(3, 3), nn.ReLU(), nn.Linear(3, 3))
with torch.no_grad():
    same[0].weight.copy_(torch.eye(3))
    same[2].weight.copy_(torch.eye(3))
    same[0].bias.copy_(torch.tensor([-1.0, 0.0, 0.0]))
    same[2].bias.copy_(torch.tensor([1.0, 0.0, 0.0]))
z = torch.zeros(3)
print("in ABI order: ", same(z).tolist())
sd = same.state_dict()
sd["0.bias"], sd["2.bias"] = sd["2.bias"].clone(), sd["0.bias"].clone()
print("load_state_dict result:", same.load_state_dict(sd))
print("after swapping the two biases:", same(z).tolist())
```

```
in ABI order:  [1.0, 0.0, 0.0]
load_state_dict result: <All keys matched successfully>
after swapping the two biases: [0.0, 0.0, 0.0]
```

`load_state_dict` reports that all keys matched, because they did: the names are present and the
shapes agree. The function changed anyway.

GraphSpec fixes the number and shapes of parameter slots and computes their positional order
through composition. The example shows the remaining obligation: the tensors assigned to those
slots must have the intended roles. A primitive's correctness theorem relates the slot order to
its computation.

A checkpoint bridge must therefore retain each tensor's layer and role. In this model,
one bias shifts values before thresholding and the other shifts them afterwards; exchanging
those roles changes the function.

# Deterministic Initialization

GraphSpec knows the parameter shapes and their order, but an architecture does not mathematically
require one initializer. The repository supports zero initialization for structural examples and
deterministic seeded initialization for executable models. Deterministic here means reproducible
from the layer's occurrence index, with no global random state involved:

```lean (name := gsInit)
-- Show the ordered parameter pack produced by the
-- deterministic initializer.
#eval LowerToDAG.Chain.detInitParams? gsMlp
```

```leanOutput gsInit (whitespace := lax)
Except.ok [[3, 2]:
 [[-0.435614, 0.404103], [-0.418974, -0.588339],
  [0.596124, -0.950010]],
 [3]:
 [0.000000, 0.000000, 0.000000],
 [1, 3]:
 [[-0.112675, -0.745444, 0.214802]],
 [1]:
 [0.000000]]
```

Four tensors, in ABI order, with the two biases exactly zero. The traversal returns `Except` because
it inherits the same partiality as the sequential lowering: a primitive with no layer view has no
initializer either.

The initializer theorem identifies the two layer occurrences:

```lean (name := gsInitThm)
-- The theorem identifies the actual per-layer initializers,
-- including their occurrence indices.
#check @Models.mlp_detInitParams_eq_torchlean_linear_inits
```

```leanOutput gsInitThm (whitespace := lax)
Models.mlp_detInitParams_eq_torchlean_linear_inits :
  ∀ (inputWidth hiddenWidth outputWidth : ℕ),
    LowerToDAG.Chain.detInitParams?
        (Models.mlp inputWidth hiddenWidth outputWidth) =
      Except.ok
        ((Runtime.Autograd.Model.Layers.linear inputWidth
              hiddenWidth).initState.append
          (Runtime.Autograd.Model.Layers.linear hiddenWidth
              outputWidth 1).initState)
```

Read the right-hand side as: whatever the ordinary TorchLean linear layer would produce at
occurrence index 0, appended to whatever it would produce at occurrence index 1. The first index is
implicit at its default value, and the second is the explicit `1`. So the GraphSpec traversal is
not a second initializer that happens to look similar; it is provably the concatenation of the
runtime's own initializers, in ABI order, with the occurrence counter advancing only at primitives
whose `countsAsLayer` is true. ReLU does not advance it, which is why the second linear layer is
index 1 and not index 2.

This theorem establishes reproducibility and agreement with the runtime initializer. It leaves
initialization quality, scaling with depth, and the provenance of imported checkpoints as separate
questions.

Adding or reordering parameter-bearing layers can change later occurrence indices and hence
their initialized weights, even if the total scalar count stays the same. When comparing two
execution routes, keep this initialization convention fixed along with the architecture.

# Pure, Runtime, And PyTorch Forward Evaluation

The tutorial command trains this MLP on a single sample, input `[0.5, 0.8]` and target `[1.0]`, with
mean squared error. So the pure interpretation should predict the loss the runtime reports before
any step is taken. Run the pure interpretation on the deterministic parameters:

```lean (name := gsForward)
-- Use that same generated parameter pack for a concrete
-- pure forward evaluation.
#eval do
  match LowerToDAG.Chain.detInitParams? gsMlp with
  | .error e => IO.println e
  | .ok params =>
      let x : Tensor Float [2] := [0.5, 0.8]
      IO.println s!"forward = {Interp.spec gsMlp params x}"
```

```leanOutput gsForward (whitespace := lax)
forward = [-0.011884]
```

The command below reports `mean_loss(before training) = 1.023910`. With one sample and one output,
mean squared error is $`(y-t)^2`, and

$$`(-0.011884-1)^2\approx1.023910.`

Then PyTorch, loaded with the same four tensors printed above and given the same input, agrees to
the digits both sides print:

```
forward: -0.011884444393217564
mse vs target 1.0: 1.0239101648330688
```

The pure GraphSpec evaluation, the runtime training loop, and PyTorch agree at the displayed
precision for this parameter pack and input. That comparison checks the wiring of this example.
The theorem below addresses a different scope: it relates the two pure model definitions for
every parameter pack and input.

# MLP Lowering

The repository contains an executable GraphSpec tutorial:

```terminal
# Run the GraphSpec example through its eager execution
# interpretation.
lake exe torchlean graphspec --device cpu --execution eager
```

The recorded run prints:

```terminal +output
== GraphSpec tutorial ==
GraphSpec architecture ladder:
  1. MLP: sequential layer stack; lowers to nn.Sequential and trains below.
  2. CNN: sequential vision graph with checked conv/pool shape arithmetic.
  3. residualLinear: minimal DAG-native skip connection.

model:
Sequential: [2] -> [1], layers=3, params=13, state=13
  [0] Linear(2, 3): [2] -> [3] params=9, state=9 [[3, 2], [3]]
  [1] ReLU: [3] -> [3] params=0, state=0 []
  [2] Linear(3, 1): [3] -> [1] params=4, state=4 [[1, 3], [1]]
dataset size = 1
mean_loss(before training) = 1.023910
mean_loss(after training) = 0.260259
forward: GraphSpec MLP lowered to TorchLean and executed
steps=3 arithmetic=native scalar=Float32 loss=1.023910 -> 0.260259
```

The model summary reads the type-level ABI back at runtime: `params=13` is the
thirteen scalars of the table above, and the per-layer `[[3, 2], [3]]` and `[[1, 3], [1]]` are the
two halves of the parameter list that `>>>` concatenated. `params` and `state` are equal here
because this model has no non-trainable state; a model with persistent BatchNorm statistics
would show a gap, as the case studies chapter does.

The execution path is:

```
-- The chain fixes parameter order before the execution
-- backend is selected.
Models.mlp
   │ Chain ps [2] [1]
   ▼
GraphSpec.ToSequential.toSeq
   │ Except String (nn.Sequential [2] [1])
   ▼
Trainer.new
   ▼
eager runtime and autograd tape
```

Now run it again on the other execution target:

```terminal
# Run the same example through the typed-graph
# interpretation.
lake exe torchlean graphspec --device cpu --execution typed-graph
```

The recorded runs print the same losses. The runtime chooses a different interpreter for the
polymorphic program: eager execution runs operations as encountered, while typed-graph execution
records a shape-indexed graph for replay. The architecture, parameter ordering, and initializer stay
fixed. Numerical differences still require investigation of reduction order and implementation;
matching printed digits is not a universal equivalence result.

The command takes the standard runtime flags, `--arithmetic native|ieee`,
`--execution eager|typed-graph`, `--device`, and `--show-backend`; see
{ref "execution-modes"}[the execution modes chapter] for what each target means.

Matching the three-step trajectory checks the optimizer and execution route as well as the
initial forward value. To reason about arbitrary parameters and inputs, we need the pure equality
below and the separate evidence connecting each execution route to it.

# The Equivalence Theorem

GraphSpec can be evaluated without the trainer:

```
-- The pure equality below compares functions at the same
-- parameters and input.
Interp.spec gsMlp params x
```

Here `params` is a tensor pack whose shape index is exactly `[[3, 2], [3], [1, 3], [1]]`. Pattern
matching on that list recovers $`W_1`, $`b_1`, $`W_2`, and $`b_2` in ABI order, and the interpreter
computes

$$`\operatorname{linearSpec}
  (W_2,b_2)
  \left(\operatorname{ReLU}
    \left(\operatorname{linearSpec}(W_1,b_1,x)\right)\right).`

`Models.mlp_interp_eq_spec_mlp_forward` identifies this composition with TorchLean's hand-written
MLP specification:

```lean (name := gsEquivThm)
-- Inspect the specification equality with its
-- parameter-pack pattern match exposed.
#check @Models.mlp_interp_eq_spec_mlp_forward
```

```leanOutput gsEquivThm (whitespace := lax)
@Models.mlp_interp_eq_spec_mlp_forward :
  ∀ {α : Type} [inst : Storage α] [inst_1 : Context α]
    {inputWidth hiddenWidth outputWidth : ℕ}
    (params : TensorPack α
      (Models.MLPParams inputWidth hiddenWidth outputWidth))
    (x : Tensor α [inputWidth]),
    Interp.spec
        (Models.mlp inputWidth hiddenWidth outputWidth) params x =
      match
        match params with
        | TensorPack.cons w1
            (TensorPack.cons b1
              (TensorPack.cons w2
                (TensorPack.cons b2 TensorPack.nil))) =>
          (w1, b1, w2, b2) with
      | (w1, b1, w2, b2) =>
        have l1 := { weights := w1, bias := b1 };
        have l2 := { weights := w2, bias := b2 };
        Examples.mlpForward l1 l2 x
```

The result is quantified over the scalar type with its `Storage` and `Context` instances, all
three widths, the parameter pack, and the input. The nested match on `params` extracts the four
tensors in ABI order, builds the two `LinearSpec` values, and passes them to `Examples.mlpForward`.
The equality therefore covers arbitrary well-shaped parameters, including but not limited to the
deterministic initialization used above.

The proof in {src "NN/GraphSpec/Models/MlpSpecEquivalence.lean"}[`MlpSpecEquivalence.lean`]
first pattern-matches on the four entries of the parameter pack. After those tensors are exposed,
the interpreter's parameter splits and both model definitions reduce to the same expression, so
`rfl` closes the goal. Its axiom report is:

```lean (name := gsAxioms)
-- List the logical dependencies of this particular
-- equivalence theorem.
#print axioms Models.mlp_interp_eq_spec_mlp_forward
```

```leanOutput gsAxioms (whitespace := lax)
'NN.GraphSpec.Models.mlp_interp_eq_spec_mlp_forward' depends on
axioms: [propext, Classical.choice, Quot.sound]
```

The report lists Lean's standard axioms used by Mathlib
({Informal.citep mathlib2020}[]), with no additional axiom or unproved placeholder.

The theorem relates two pure definitions. Runtime agreement requires separate evidence.
Conformance tests compare selected executions; IR-level theorems establish the stated semantic
equalities under their lowering and operator hypotheses. Neither extends this pure MLP equality
to arbitrary native kernels. {ref "verification"}[The verification chapter] explains the
additional connections.

# DAG Syntax And Shared Computations

A chain joins unary stages with `>>>`. To represent the sharing inside a residual block explicitly,
use a DAG:

$$`r(x)=\operatorname{ReLU}(Wx+b+x).`

The input $`x` is used twice, once by the linear branch and once by the skip. A dedicated
`ResidualLinear` primitive could hide those uses inside one chain stage. Exposing them as graph
edges instead lets a tool inspect and transform the sharing pattern. GraphSpec's DAG syntax
provides that representation for residual architectures ({Informal.citep resnet2016}[]) and other
models with shared intermediates.

A term has type

```
-- The environment lists available values; the final index
-- is the result shape.
DAG.Term Γ τ
```

meaning that, given a typed environment $`\Gamma`, it computes a tensor of shape $`\tau`. Its
essential constructors are `var`, which reads an existing value; `op`, which applies a primitive of
any arity; and `let1`, which computes an intermediate once and extends the environment. Two further
constructors, `cast` and `castEnv`, carry propositional equalities of shapes and environments; they
exist so that a programmatic lowering can keep terms in constructor form instead of blocking on an
equality that is true but not definitional.

`let1` expresses how a newly computed value becomes available to later operations:

```lean (name := gsLet1)
-- A let-bound intermediate extends the environment
-- available to the body.
#check @DAG.Term.let1
```

```leanOutput gsLet1 (whitespace := lax)
@DAG.Term.let1 : {Γ : List Shape} → {σ τ : Shape} →
  DAG.Term Γ σ → DAG.Term (Γ ++ [σ]) τ → DAG.Term Γ τ
```

The body is typed in the *extended* environment `Γ ++ [σ]`, and there is no constructor that
extends an environment without also supplying the value. Since the language has no recursion and
only ever appends previously computed values, its terms denote acyclic graphs by construction. No
cycle check is needed to construct such a term. A later erased IR still runs its own
structural validation.

Variables are shape-indexed rather than numeric:

```lean (name := gsVar)
-- Variable constructors preserve the shape of the entry
-- they select.
#print NN.GraphSpec.DAG.Var
```

```leanOutput gsVar (whitespace := lax)
inductive NN.GraphSpec.DAG.Var : List Shape → Shape → Type
number of parameters: 0
constructors:
NN.GraphSpec.DAG.Var.head : {s : Shape} → {Γ : List Shape} →
  DAG.Var (s :: Γ) s
NN.GraphSpec.DAG.Var.tail : {Γ : List Shape} →
  {s t : Shape} → DAG.Var Γ t → DAG.Var (s :: Γ) t
```

A `Var Γ s` is a de Bruijn position that also records that the entry it selects has shape `s`. The
alternative, a bare `Fin Γ.length` plus a lookup, would force every evaluator and every renaming
lemma to transport along `List.get` equalities. Intrinsically typed syntax of this kind is the
standard remedy ({Informal.citep benton2012}[]), and the payoff is visible in
{srcDir "NN/GraphSpec/DAG"}[`NN/GraphSpec/DAG`]: environment lookup reduces structurally, and the
renaming and substitution lemmas are stated without casts. A numeric position is still available
through `Var.ofFin` for lowerings that discover indices dynamically.

Two references to the same `let1` result express sharing without duplicating its computation.
In a residual block, the original input remains available after the linear result is bound.
The skip edge must select that original input.

## Residual Block Evaluation

The smallest DAG example is the residual linear block, whose type says it has two parameters, one
data input, and one output:

```lean (name := gsResidualType)
-- A residual model reuses its input while retaining the
-- linear layer’s parameter ABI.
#check @Models.residualLinear
```

```leanOutput gsResidualType (whitespace := lax)
Models.residualLinear : (d : ℕ) →
  DAG.Model (Models.ResidualLinearParams d) [[d]] [d]
```

Its body, in {src "NN/GraphSpec/Models/ResidualLinear.lean"}[`ResidualLinear.lean`], reads
essentially as

```
-- The skip edge must read the original input, not the newly
-- computed linear result.
let y = linear(W, b, x)
let z = add(y, x)
relu(z)
```

where `x` appears in both the linear arguments and the add arguments while `y` is bound once. Run it
on parameters chosen so the answer can be checked by hand: the swap matrix, a bias, and a small
input.

```lean (name := gsResidualRun)
-- These values distinguish the intended skip connection
-- from accidental reuse of y.
#eval do
  let w : Tensor Float [2, 2] := [[0.0, 1.0], [1.0, 0.0]]
  let b : Tensor Float [2] := [-3.0, 0.5]
  let x : Tensor Float [2] := [1.0, 2.0]
  let out := (Models.residualLinear (d := 2)).specFwd
    (α := Float) (.cons w (.cons b .nil)) (.cons x .nil)
  IO.println s!"residual = {out}"
```

```leanOutput gsResidualRun (whitespace := lax)
residual = [0.000000, 3.500000]
```

By hand: $`Wx=[2,1]`, then $`Wx+b=[-1,1.5]`, then adding the skip gives $`[0,3.5]`, and ReLU leaves
it alone. The first coordinate is exactly zero because the skip cancelled the bias, which is the
kind of check worth doing on a new graph language: a wrong sharing pattern, for instance one that
added the *output* of the linear branch to itself, would give $`[-2,3]` before ReLU and $`[0,3]`
afterwards.

## DAG Operations

A `PrimOp` is parameter-free: what the sequential language keeps in `ps` becomes an ordinary input
in the environment. That is what makes the DAG language flexible, and it also makes the type of an
operation read like a signature rather than a layer. Matrix multiplication is the clearest case:

```lean (name := gsMatmul)
-- Batch broadcasting and matrix compatibility appear as
-- separate shape requirements.
#check @DAG.PrimOp.matmul
```

```leanOutput gsMatmul (whitespace := lax)
DAG.PrimOp.matmul : (batchA batchB batch : Shape) →
  (mDim nDim pDim : ℕ) →
    batchA.CanBroadcastTo batch →
      batchB.CanBroadcastTo batch →
        DAG.PrimOp
          [batchA.concat [mDim, nDim],
            batchB.concat [nDim, pDim]]
          (batch.concat [mDim, pDim])
```

One primitive covers the shared-batch case, the pairwise-batched case, and the unbatched case,
because the two broadcast facts are arguments rather than assumptions. There is no separate
`bmm`, and no runtime branch that guesses which case you meant: the caller supplies the evidence,
and vector inputs reach the same primitive through a row reshape. Multi-head attention follows the
same convention as the public attention builder, taking any leading shape before the
`[sequence, model]` suffix, so one operation serves a batched and an unbatched call site.

Most of these operations also carry a `simp` lemma named after them, `add_specFwd`,
`matmul_specFwd`, and so on, that rewrites the primitive's pure meaning to the corresponding `Spec`
function. The proofs are `rfl` because the DAG operation unfolds to that specification. These lemmas
let a proof about a DAG model step through operations without unfolding the interpreter by hand.

For matrix multiplication, the matrix dimensions and the batch dimensions answer separate
questions. The contracted dimension must match for each scalar dot product to be meaningful.
The leading dimensions must support the declared broadcasting so the correct pair of matrices is
chosen for every output batch coordinate. The constructor type records both requirements. That
shape information determines the forward indexing, but a reverse theorem must additionally show
how contributions from repeated broadcast uses accumulate into the original operand.

## Structural Lowering From Chain To DAG

A chain can be lowered into a DAG term structurally, so that DAG-only tooling can consume a
pipeline that was written with `>>>`:

```lean (name := gsToDag)
-- The converted DAG places parameters and the model input
-- in its initial environment.
#check @LowerToDAG.Chain.toDAGTerm
```

```leanOutput gsToDag (whitespace := lax)
@LowerToDAG.Chain.toDAGTerm : {ps : List Shape} →
  {σ τ : Shape} → Chain ps σ τ → DAG.Term (ps ++ [σ]) τ
```

The environment of the result is `ps ++ [σ]`: the parameters that were type-level in the chain
become ordinary variables, and the data input is last. Each sequential primitive is embedded as a
DAG operation with inputs `ps ++ [σ]`, and that embedding does have a proof that nothing is lost:

```lean (name := gsEmbedThm)
-- The primitive embedding preserves the pure forward
-- function by construction.
#check @Primitive.toDAGPrimOp_specFwd_eq
```

```leanOutput gsEmbedThm (whitespace := lax)
@Primitive.toDAGPrimOp_specFwd_eq :
  ∀ {α : Type} [inst : Storage α] [inst_1 : Context α]
    {ps : List Shape} {σ τ : Shape} (p : Primitive ps σ τ)
    (params : TensorPack α ps) (x : Tensor α σ),
    (LowerToDAG.Primitive.toDAGPrimOp p).specFwd
        (params.append (TensorPack.cons x TensorPack.nil)) =
      p.specFwd params x
```

The primitive statement holds for every primitive, including one supplied by a user. The
whole-chain theorem lifts that agreement through every sequential composition:

```lean (name := gsChainDagPreservation)
-- The induction covers arbitrary chains because each
-- embedded primitive keeps specFwd.
#check @LowerToDAG.Chain.eval_toDAGTerm
```

```leanOutput gsChainDagPreservation (whitespace := lax)
@LowerToDAG.Chain.eval_toDAGTerm :
  ∀ {α : Type} [inst : Storage α] [inst_1 : Context α]
    {ps : List Shape} {σ τ : Shape} (g : Chain ps σ τ)
    (params : TensorPack α ps) (input : Tensor α σ),
    DAG.Term.eval (params.append (TensorPack.cons input TensorPack.nil))
        (LowerToDAG.Chain.toDAGTerm g) =
      Interp.spec g params input
```

The proof in {src "NN/GraphSpec/Chain/ToDAG/Semantics.lean"}[`Chain.ToDAG.Semantics`] follows
`Chain` composition, preserving parameter order and the intermediate value introduced by each
DAG `let1`. It allows arbitrary parameter shapes, custom primitives, and scalar `Context`
instances. No primitive-program agreement hypothesis is needed: both pure interpretations call
the same `specFwd`. Relating that field to `program`, a derivative implementation, or a native
kernel remains a separate obligation.

The numerical example below illustrates this proved equality on one initialized model:

```lean (name := gsChainDag)
-- Compare both pure representations with exactly the same
-- initialized parameters.
#eval do
  match LowerToDAG.Chain.detInitParams? gsMlp with
  | .error e => IO.println e
  | .ok params =>
      let x : Tensor Float [2] := [0.5, 0.8]
      let env := TensorPack.append (α := Float)
        params (.cons x .nil)
      let chainValue := Interp.spec gsMlp params x
      let dagValue := DAG.Term.eval env
        (LowerToDAG.Chain.toDAGTerm gsMlp)
      IO.println s!"chain = {chainValue}"
      IO.println s!"dag   = {dagValue}"
```

```leanOutput gsChainDag (whitespace := lax)
chain = [-0.011884]
dag   = [-0.011884]
```

The theorem establishes equality independently of these printed decimals, including for a chain
containing a custom primitive. In the reverse direction, preserving a DAG's fan-out in a chain
requires a special combinator; duplicating the branch would lose the explicit sharing.

# Chain Versus DAG Model

:::table +header
*
  * Form
  * Best use
  * Sharing
  * Parameter representation
*
  * `Chain ps σ τ`
  * readable layer chains
  * no explicit fan-out
  * type-indexed list `ps`
*
  * `DAG.Model ps ins τ`
  * residual and multi-input models
  * explicit variables and `let1`
  * typed model environment
:::

There is a third form worth knowing about. `DAG.MultiModel ps ins outs` returns a typed *list* of
tensors, which is what a recurrent layer needs when it produces an updated state alongside an
observable output. Keeping those as a typed list rather than flattening them into one buffer means
the shared `let1` binding that computes the new state is still shared after the model is inlined
into a larger graph, and `MultiModel.eval_inline` is the theorem that says inlining preserves every
output.

# Convolution And Pooling Shapes

The sequential vocabulary is not limited to dense layers. `NN.GraphSpec.Core` supplies `linear`,
`relu`, and axis-wise `softmax`; {src "NN/GraphSpec/Primitives/Spatial.lean"}[`Primitives/Spatial`]
adds spatial-rank-polymorphic convolution and max pooling, flattening, and BatchNorm with an
explicit channel axis. Rank-polymorphic means the same definition applies to signals, images, and
volumes: the spatial extents are a `Tensor Nat [d]` rather than a fixed pair.

The intermediate spatial arithmetic is part of the type. `Models.cnn` accepts a feature chain
and attaches flattening and a linear head. The chain determines its own depth and spatial
operations. Here are the tutorial's two stages, with their different channel widths:

```lean (name := gsCnnFeatures)
/-- Two spatial stages ending in three 2-by-2 maps. -/
def gsCnnFeatures :
    Chain [[2, 1, 3, 3], [2], [3, 2, 3, 3], [3]]
      [1, 8, 8] [3, 2, 2] :=
  Chain.conv 1 2 [3, 3] [1, 1] [1, 1] [8, 8] >>>
  Chain.relu [2, 8, 8] >>>
  Chain.maxPool 2 [2, 2] [2, 2] [0, 0] [8, 8]
    (hKernel := by intro i; fin_cases i <;> decide)
    (hStride := by intro i; fin_cases i <;> decide) >>>
  Chain.conv 2 3 [3, 3] [1, 1] [1, 1] [4, 4] >>>
  Chain.relu [3, 4, 4] >>>
  Chain.maxPool 3 [2, 2] [2, 2] [0, 0] [4, 4]
    (hKernel := by intro i; fin_cases i <;> decide)
    (hStride := by intro i; fin_cases i <;> decide)

example :
    Chain
      ([[2, 1, 3, 3], [2], [3, 2, 3, 3], [3]] ++
        [[4, 12], [4]])
      [1, 8, 8] [4] :=
  Models.cnn gsCnnFeatures 4
```

Padding 1 with a `3x3` kernel preserves the extent, so the two pooling stages halve `8` to `4` and
`4` to `2`. Three channels of `2x2` give twelve features. The feature chain's output shape
determines the head's weight shape `[4, 12]`; its parameters follow the four feature tensors.
Changing the chain changes that inferred head width. The explicit type above asks Lean to check
the complete parameter list and output shape, so an inconsistent annotation gives the same kind
of compile error as the width typo earlier in this chapter.

The feature chain can contain any composable sequence supported by its chosen interpretation.
An identity chain gives a classifier on the flattened input. The general chain-to-DAG conversion
applies to the resulting model; it does not require a separate CNN conversion.

# Checkpoint Import And Parameter Mapping

GraphSpec's parameter ABI is a positional convention that the type computes. An imported checkpoint
is a bag of named arrays produced by another program. Connecting the two needs an argument the types
cannot supply on their own:

1. parse the external names and arrays;
2. check each concrete shape and finite-value condition;
3. map values into the GraphSpec ABI;
4. state or assume that this mapping matches the source framework's layout.

Steps one to three are code, and {ref "external-tools-and-ffi"}[the boundary chapter] shows the
parser and its checks. Step four is a modeling assumption about someone else's file format, and the
swapped-bias experiment above is exactly what it looks like when that assumption is wrong: all keys
match, all shapes agree, the function is different. The import contract must record the mapping;
the architecture's shape constraints cannot authenticate training provenance.

Distinct parameter values help test the mapping: filling both biases with zero would have hidden
our swap. A passing sample still cannot establish equality for all parameters. To apply the MLP
equivalence theorem to an imported model, the importer must connect its named tensors to the
ordered parameter pack in that theorem.

# GraphSpec And The Backend IR

TorchLean keeps several graph representations because they carry different information.

`GraphSpec.Chain` is typed sequential architecture syntax, and `GraphSpec.DAG.Model` adds explicit
sharing. A model that connects incompatible shapes does not elaborate, and each primitive carries
both a pure interpretation and a TorchLean-program interpretation.

`NN.IR.Graph` is a serializable op-tagged DAG. Nodes carry numeric identifiers, parent identifiers,
output shapes, and attributes; tensors and parameters live in an external payload. That form lets
importers construct candidate graphs for validation, transformation, verification, and kernel
selection. A candidate node can claim a shape its parents cannot produce, so a validator must
check the declaration.

There is no universal lowering pass from every GraphSpec model to `NN.IR.Graph`. Selected frontend
and model paths lower to IR, and their semantic theorems state which meaning is preserved.
{ref "graphs-and-ir"}[The next chapter] builds and inspects that lower-level representation
directly.

# Further Architecture Checks

The following variations isolate shape composition, optional adapters, scalar interpretation, and
semantic preservation using the same architectures.

1. Change the hidden width at one of the two linear nodes in
   {src "NN/GraphSpec/Models/Mlp.lean"}[`Models/Mlp.lean`] and elaborate the file. The error names
   the two shapes that disagree, and it arrives before initialization, data loading, or execution.
2. Append a classifier head, `Models.mlp 2 3 3 >>> Chain.softmax [3] 0`, and check the type. The
   parameter list does not change, because softmax has no parameters, and the `AxisInBounds`
   instance is what stops you from writing axis `1`.
3. Write a primitive with no `toLayerM?`, as `gsDouble` does above, put it in a chain, and try both
   `Chain.toProgram` and `ToSequential.toSeq`. One succeeds and one gives you the error message
   quoted in this chapter.
4. Run the tutorial with `--arithmetic ieee` and compare the two losses against the transcript
   above. A difference here is a floating-point story, and {ref "floats"}[the floating-point
   chapter] is where that story is told.
5. Change a pooling stride in `gsCnnFeatures` from `[2, 2]` to `[3, 3]` and update the following
   spatial annotations. Inspect the head's inferred weight shape before annotating the classifier.
6. Apply `LowerToDAG.Chain.eval_toDAGTerm` to a chain containing your own primitive. Inspect why
   it needs no premise about that primitive's `program`, then state the additional agreement
   needed to transfer the pure equality to execution.
