import VersoManual
import TorchLean.NN.LoRA
import TorchLean.NN.Autograd.Function
import TorchLean.Tensor.Op.Table.Linear
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
import TorchLeanExamples.DeepDives.Sequential.MLP
import TorchLean.NN.Models.Classifier
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.Roles
import TorchLean.NN.Session.Random
import TorchLean.Verify.Trainer

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean
open TorchLean
open nn
open TorchLean (Ops)
open Ops (Ref)
open scoped TorchLean.nn.Sequential
open Examples.Sequential
open Lean.Elab.Tactic.GuardMsgs.WhitespaceMode (lax)

#doc (Manual) "Sequential Models" =>
%%%
tag := "sequential"
file := "Sequential-Models___-One-Architecture___-Several-Meanings"
%%%

In {ref "spec-layer"}[the specification chapter], we described the MLP by its pure formula:

$$`x\mapsto W_2\operatorname{ReLU}(W_1x+b_1)+b_2.`

Suppose we want to count its trainable scalars without running it, or replace the activation
while keeping the layers on either side. A plain function does not expose a layer list to
traverse. A sequential model, `nn.Sequential`, stores the architecture as data: a list of layers
whose type records the input and output shapes, and whose parameter shapes can be read without
examining tensor elements or running the model. The same value has a pure meaning for theorems and a
program that the runtimes execute.

Every Lean block on this page is elaborated when the guide is built, and every printed value below
is checked against that elaboration. Command transcripts below are recorded runs, separate from
the page build.

# Architecture Representation

Initialization, forward evaluation, gradient extraction, and parameter loading all need the same
layer connections. A sequential model is either the identity `Sequential.id σ` or a layer followed
by a model (`Sequential.cons`). Each layer carries the ordered shapes of its parameters and buffers
(`stateShapes`), their initial values, and one forward program written against the operation
interface `Ops`. In the tagless-final style {Informal.citep kiselyov2012}[], each interpreter of
that interface chooses what an operation does:

```
-- One model has a pure meaning and one program that several
-- interpreters run.
                            ┌── Sequential.eval ──────> pure tensor function (Ops Id)
Sequential σ τ (layers) ────┤
                            └── Sequential.forward ───> TorchLean.Program
                                                          │
                                                          ├── eager execution (CPU or CUDA tape)
                                                          ├── typed graph (lowerToTypedGraph)
                                                          └── verification lowering
                                                              ──> TorchLean.IR.Graph
```

The upper branch reads the model as a function on tensors with explicit parameters. It runs the
same layer programs as the lower branch, under the pure interpreter `Ops Id`, so a layer has one
definition of what it computes. The pure interpreter is a denotation, not a checked runtime: it
skips the runtime's attribute and value-domain checks, updates no buffers, and returns zeros where
an operator has no rule. A random operator draws with the first key under its seed
(`Random.keyOf seed 0`) on every call, and a keyless operator runs unchanged. That convention
fixes one draw and claims no correspondence with the runtimes' random streams. Agreement between
the pure meaning and an executable interpreter is a separate claim for each interpreter and each
operator it supports.

On the lower branch, the verification lowering validates the IR it produces, and the lowered IR
provably runs as the typed graph it records, but no theorem relates that graph to the source model.

# MLP Definition

I'll use two input features, three hidden features, and one output. In
{src "TorchLeanExamples/DeepDives/Sequential/MLP.lean"}[
`TorchLeanExamples/DeepDives/Sequential/MLP.lean`], the library MLP composes two linear layers and
a ReLU:

```
-- The definition composes three single-layer models.
def model (inDim hidDim outDim : Nat) : nn.Sequential [inDim] [outDim] :=
  (nn.Sequential.fromLayer (Layer.linear inDim hidDim 0)).comp
    ((nn.Sequential.fromLayer (Layer.op .relu)).comp
      (nn.Sequential.fromLayer (Layer.linear hidDim outDim 1)))
```

Its type states the input and output shapes for every choice of widths:

```lean (name := sqMlpType)
-- The hidden width does not appear in the type: it is
-- internal to the model.
#check @MLP.model
```

```leanOutput sqMlpType (whitespace := lax)
MLP.model : (inDim : ℕ) → ℕ → (outDim : ℕ) → Sequential [inDim] [outDim]
```

Parameter shapes are a function of the model (`stateShapes`), and they reduce to a literal list
once the widths are known:

```lean (name := sqShapes)
-- The ordered parameter shapes of the 2 → 3 → 1 model.
#eval nn.Sequential.stateShapes (MLP.model 2 3 1)
```

```leanOutput sqShapes (whitespace := lax)
[[3, 2], [3], [1, 3], [1]]
```

Read the list as the model's parameter ABI:

- the model consumes one tensor of shape `[2]` and produces one of shape `[1]`
- its parameter pack holds four tensors
- the order is $`W_1`, $`b_1`, $`W_2`, $`b_2`

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

Together, the four slots hold $`6+3+3+1=13` trainable scalars. A tool can compute this count from
`stateShapes` without initializing tensors or interpreting parameter names.

# Composition And The Parameter ABI

`Sequential.comp` (written `>>>`) joins two models when the first output shape is the second input
shape, and their parameter lists concatenate in construction order:

```lean (name := sqCompShapes)
-- Composition appends the parameter shapes of the second
-- model after those of the first.
#check @nn.Sequential.stateShapes_comp
```

```leanOutput sqCompShapes (whitespace := lax)
@Sequential.stateShapes_comp : ∀ {σ τ υ : Shape} (first : Sequential σ τ)
  (second : Sequential τ υ),
  (first >>> second).stateShapes = first.stateShapes ++ second.stateShapes
```

In the MLP, the first linear layer contributes `[[3, 2], [3]]`, ReLU contributes no parameters, and
the second linear layer contributes `[[1, 3], [1]]`. A layer can still misread two same-shaped
slots, as the swapped-bias example below shows.

Change the hidden width of the first linear layer to five while leaving the second layer's input
width at three:

```lean +error -keep (name := sqBroken)
-- The hidden width disagrees at the connection, so
-- elaboration must reject this model.
def sqBroken : nn.Sequential [2] [1] :=
  (nn.Sequential.fromLayer (Layer.linear 2 5)).comp <|
    (nn.Sequential.fromLayer (Layer.op .relu)).comp <|
      nn.Sequential.fromLayer (Layer.linear 3 1)
```

```leanOutput sqBroken (whitespace := lax)
Application type mismatch: The argument
  Layer.linear 3 1
has type
  Layer [3] [1]
but is expected to have type
  Layer [5] [1]
in the application
  Sequential.fromLayer (Layer.linear 3 1)
```

The report names the two shapes that disagree. Each linear layer is valid on its own, but the first
produces five hidden values while the last expects three. Lean rejects their composition while
checking the model definition, and it rejects the same model written with `>>>` too.

## Shape Mismatch In PyTorch

For comparison, here's the same architecture and the same typo in PyTorch
{Informal.citep pytorch2019}[]:

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

In the recorded run, PyTorch prints four lines and then raises, with the traceback ending in the
last line below:

```
[(3, 2), (3,), (1, 3), (1,)]
13
['0.weight', '0.bias', '2.weight', '2.bias']
19 scalars
RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x5 and 3x1)
```

PyTorch's correct model reports the same four parameter shapes and thirteen scalars as
`stateShapes`. What differs is where the broken model gets caught.
`nn.Sequential(nn.Linear(2, 5), nn.ReLU(), nn.Linear(3, 1))` is a perfectly good Python object. It
constructs, it reports nineteen parameters, it can be moved to a device, it can be handed to an
optimizer, and it can be saved. It fails on the first tensor that reaches it.

This Python object checks the connection when data reaches the layers, and its `state_dict`
identifies parameters by string keys. A Lean sequential model requires shapes that the elaborator
can see. It rejects the incompatible connection while checking the definition and exposes the
parameter shapes as a value computed from the model.

# The Pure Interpretation

`Sequential.eval` runs the model's forward program in evaluation mode under the pure interpreter,
at any scalar type with `Storage` and `Context` instances, with every parameter passed in as an
argument:

```lean (name := sqEval)
-- The parameter pack is indexed by the model's own
-- parameter shapes.
#check @nn.Sequential.eval
```

```leanOutput sqEval (whitespace := lax)
@Sequential.eval : {σ τ : Shape} →
  (model : Sequential σ τ) →
    {α : Type} → [inst : Storage α] → [Context α] →
      TensorPack α model.stateShapes → Tensor α σ → Tensor α τ
```

All of the model's state (parameters and buffers) goes in the pack, in `stateShapes` order. Nothing
is initialized, so a theorem about `Sequential.eval` holds for every parameter value. Composition
is evaluation in sequence:

```lean (name := sqEvalComp)
-- The cast moves the appended pack to the composed model's
-- parameter shapes.
#check @nn.Sequential.eval_comp
```

```leanOutput sqEvalComp (whitespace := lax)
@Sequential.eval_comp : ∀ {α : Type} [inst : Storage α] [inst_1 : Context α]
  {σ τ υ : Shape} (first : Sequential σ τ)
  (second : Sequential τ υ) (firstParams : TensorPack α first.stateShapes)
  (secondParams : TensorPack α second.stateShapes) (x : Tensor α σ),
  (first >>> second).eval (⋯ ▸ Pack.append firstParams secondParams) x =
    second.eval secondParams (first.eval firstParams x)
```

The equation is the pure specialization of `Sequential.forwardState_comp`, which holds for every
lawful execution monad and preserves the order of forward calls and buffer updates.

## Parameter Order And Shape Checking

Shape typing proves that each slot has the right shape, but two tensors of the same shape can still
be exchanged. In the square MLP `MLP.model 3 3 3`, both biases are `[3]`, so a pack with the biases
exchanged has the same type as the correct one. Feed it the zero input, identity weights, and two
distinguishable biases, in both orders:

```lean (name := sqSwapRun)
-- Hold the matrices fixed and swap only the bias roles to
-- isolate the ABI mistake.
#eval do
  let identity : Tensor Float [3, 3] :=
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
  let first : Tensor Float [3] := [-1.0, 0.0, 0.0]
  let second : Tensor Float [3] := [1.0, 0.0, 0.0]
  let x : Tensor Float [3] := [0.0, 0.0, 0.0]
  let inOrder := (MLP.model 3 3 3).eval
    (.cons identity (.cons first
      (.cons identity (.cons second .nil)))) x
  let exchanged := (MLP.model 3 3 3).eval
    (.cons identity (.cons second
      (.cons identity (.cons first .nil)))) x
  IO.println s!"in ABI order = {inOrder}"
  IO.println s!"exchanged    = {exchanged}"
```

```leanOutput sqSwapRun (whitespace := lax)
in ABI order = [1.000000, 0.000000, 0.000000]
exchanged    = [0.000000, 0.000000, 0.000000]
```

Both packs elaborate. Since the model computes $`\operatorname{ReLU}(x+b_1)+b_2`, the correct order
gives $`\operatorname{ReLU}(-1)+1=1` in the first coordinate and the exchanged order gives
$`\operatorname{ReLU}(1)-1=0`. Exchanging the biases changes the function while preserving every
tensor's type. With an activation between them, the biases' positions become observable in the
output.

PyTorch's checkpoint-loading checks accept the same swap:

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
shapes agree. But the function changed anyway.

A sequential model fixes the number and shapes of parameter slots and computes their positional
order through composition. As the example shows, one obligation remains: the tensors assigned to
those slots must have the intended roles. A layer's correctness theorem relates the slot order to
its computation, as the MLP theorem below does.

# The MLP Theorem

Evaluating the MLP on the two linear layers' weights and biases, in ABI order, is the hand-written
formula of the specification chapter:

```lean (name := sqEquivThm)
-- The pack is built from the fields of two linear layers,
-- in the order W₁, b₁, W₂, b₂.
#check @MLP.eval_model
```

```leanOutput sqEquivThm (whitespace := lax)
@MLP.eval_model : ∀ {α : Type} [inst : Storage α] [inst_1 : Context α] {inDim hidDim outDim : ℕ}
  (l1 : Linear α inDim hidDim) (l2 : Linear α hidDim outDim)
  (x : Tensor α [inDim]),
  (MLP.model inDim hidDim outDim).eval
      (Pack.cons l1.weights
        (Pack.cons l1.bias (Pack.cons l2.weights
          (Pack.cons l2.bias Pack.nil))))
      x =
    have z1 := l1.forward x;
    have a1 := z1.relu;
    l2.forward a1
```

The result is quantified over the scalar type with its `Storage` and `Context` instances, all three
widths, both linear layers, and the input. Every pack of shapes
`[[hidDim, inDim], [hidDim], [outDim, hidDim], [outDim]]` has this form, so the equality covers
arbitrary well-shaped parameters, the deterministic initialization used below among them.
`MLP.forward` of the specification chapter is the same formula by definition.

The proof in
{src "TorchLeanExamples/DeepDives/Sequential/MLP.lean"}[`DeepDives/Sequential/MLP.lean`] is
`rfl`: under the pure interpreter the linear operator's rule is `Linear.forward` and the ReLU
operator's rule is `Tensor.relu`, so both sides reduce to the same expression. Its axiom report is:

```lean (name := sqAxioms)
-- List the logical dependencies of this particular
-- evaluation theorem.
#print axioms MLP.eval_model
```

```leanOutput sqAxioms (whitespace := lax)
'TorchLean.Examples.Sequential.MLP.eval_model' depends on axioms:
[propext, Classical.choice, Quot.sound]
```

All three are Lean's standard axioms used by Mathlib {Informal.citep mathlib2020}[], and there's
no additional axiom or unproved placeholder.

Since the theorem is about the pure meaning, runtime agreement requires separate evidence:
conformance tests compare selected executions, and IR-level theorems establish the stated semantic
equalities under their lowering and operator hypotheses. Neither extends this equality to arbitrary
native kernels. {ref "verification"}[The verification chapter] explains the additional connections.

A classifier head on any feature model follows the same pattern. `Sequential.classifier features o`
flattens the output of `features` and applies a linear layer with `o` outputs:

```lean (name := sqClassifierThm)
-- The feature model is arbitrary. Its state comes first,
-- then the head's weight and bias.
#check @Sequential.eval_classifier
```

```leanOutput sqClassifierThm (whitespace := lax)
@Sequential.eval_classifier : ∀ {α : Type} [inst : Storage α] [inst_1 : Context α]
  {σ φ : Shape} {outputSize : ℕ}
  (weightSeed : ℕ) (features : Sequential σ φ) (state : TensorPack α features.stateShapes)
  (head : Linear α φ.size outputSize) (x : Tensor α σ),
  (features.classifier outputSize weightSeed).eval
      (⋯ ▸ Pack.append state (Pack.cons head.weights
        (Pack.cons head.bias Pack.nil))) x =
    head.forward ((features.eval state x).reshape [φ.size] ⋯)
```

# Deterministic Initialization

A model knows its parameter shapes and their order, but an architecture doesn't mathematically
require one initializer. Each layer stores its own seeded initial values (deterministic here means
reproducible from the layer's weight seed, with no global random state involved):

```lean (name := sqInit)
-- Show the ordered parameter pack of the initial state.
#eval nn.Sequential.initState (MLP.model 2 3 1)
```

```leanOutput sqInit (whitespace := lax)
[[3, 2]:
 [[-0.435614, 0.404103], [-0.418974, -0.588339],
  [0.596124, -0.950010]],
 [3]:
 [0.000000, 0.000000, 0.000000],
 [1, 3]:
 [[-0.112675, -0.745444, 0.214802]],
 [1]:
 [0.000000]]
```

Four tensors, in ABI order, with the two biases exactly zero. The initialization theorem identifies
the two layers:

```lean (name := sqInitThm)
-- The theorem identifies the per-layer initializers,
-- including their weight seeds.
#check @MLP.initState_model
```

```leanOutput sqInitThm (whitespace := lax)
MLP.initState_model : ∀ (inDim hidDim outDim : ℕ),
  (MLP.model inDim hidDim outDim).initState =
    Pack.append (Layer.linear inDim hidDim).initState (Layer.linear hidDim outDim 1).initState
```

Read the right-hand side as: whatever the linear layer produces with weight seed 0, appended to
whatever it produces with weight seed 1. The first seed is implicit at its default value, and the
second is the explicit `1`. ReLU has no state and contributes nothing. Initialization is total, so
the equation needs no success case.

That theorem establishes reproducibility and the parameter order, and it leaves initialization
quality, scaling with depth, and the provenance of imported checkpoints as separate questions. When
you compare two execution routes, keep the weight seeds fixed along with the architecture.

# Pure, Runtime, And PyTorch Forward Evaluation

Our tutorial command trains this MLP on a single sample (input `[0.5, 0.8]` and target `[1.0]`)
with mean squared error. So the pure interpretation should predict the loss the runtime reports
before any step is taken. Evaluate the model on its initial parameters:

```lean (name := sqForward)
-- Use the initial parameter pack for a concrete pure
-- forward evaluation.
#eval do
  let x : Tensor Float [2] := [0.5, 0.8]
  let model := MLP.model 2 3 1
  IO.println s!"forward = {model.eval model.initState x}"
```

```leanOutput sqForward (whitespace := lax)
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

The pure call uses `Float` (binary64), but the trainer transcript below uses `Float32`. Our
comparison with PyTorch agrees at the displayed digits, not at a shared precision or at the bit
level. It checks the wiring of this example. The MLP theorem addresses a different scope: it
relates the pure model to the formula for every parameter pack and input.

# Forward Programs And Lowering

A model's executable meaning is its forward program, which receives all parameters followed by the
input:

```lean (name := sqProgramType)
-- The program is polymorphic in the execution monad, and
-- the mode selects dropout and normalization behavior.
#check @nn.Sequential.forward
```

```leanOutput sqProgramType (whitespace := lax)
@Sequential.forward : {σ τ : Shape} →
  (model : Sequential σ τ) →
    optParam nn.Mode Mode.eval →
      {α : Type} → [inst : Storage α] → [inst_1 : Context α] →
        Program α (model.stateShapes ++ [σ]) τ
```

The eager runtime runs this program and records a tape for reverse mode. Typed-graph lowering
records it once as shape-indexed SSA data that can be evaluated and differentiated repeatedly:

```lean (name := sqLower)
-- Lowering records the same program as a typed graph.
#check @nn.Sequential.lowerToTypedGraph
```

```leanOutput sqLower (whitespace := lax)
@Sequential.lowerToTypedGraph : {σ τ : Shape} →
  (model : Sequential σ τ) →
    optParam nn.Mode Mode.eval →
      {α : Type} →
        [inst : Storage α] → [Context α] →
          IO (TypedGraph α (model.stateShapes ++ [σ]) τ)
```

An executable tutorial in the repository trains the MLP above:

```terminal
# Run the sequential example on the CPU.
scripts/lake.sh exe torchlean sequential --device cpu
```

The recorded run prints:

```terminal +output
== Sequential tutorial ==
pure forward (nn.Sequential.eval, Float) = [-0.011884]
model:
Sequential: [2] -> [1], layers=3, params=13, state=13
  [0] Linear(2, 3): [2] -> [3] params=9, state=9 [[3, 2], [3]]
  [1] relu: [3] -> [3] params=0, state=0 []
  [2] Linear(3, 1): [3] -> [1] params=4, state=4 [[1, 3], [1]]
dataset size = 1
mean_loss(before training) = 1.023910
mean_loss(after training) = 0.260259
steps=3 arithmetic=native scalar=Float32 loss=1.023910 -> 0.260259
```

The model summary reads the parameter ABI back at runtime: `params=13` is the thirteen scalars of
the table above, and the per-layer `[[3, 2], [3]]` and `[[1, 3], [1]]` are the two halves of the
parameter list that `>>>` concatenated. `params` and `state` are equal here because this model has
no non-trainable state. A model with persistent BatchNorm statistics would show a gap.

The command takes the standard runtime flags, `--arithmetic native|ieee`, `--device`, and
`--show-backend` (see {ref "run-configuration"}[Choosing How A Model Runs] for what each target
means). A three-step trajectory checks the optimizer and the tape as well as the initial forward
value. To reason about arbitrary parameters and inputs, we need the pure equality above and the
separate evidence connecting the runtime to it.

# A Custom Layer

A layer is its parameter shapes, their initial values, and one forward program. A layer that
doubles its input needs no parameters:

```lean (name := sqDoubleDef)
-- One program: every interpreter of `Ops` runs it, the
-- pure one included.
/-- Doubling layer: no parameters, one forward program. -/
def sqDouble (s : Shape) : nn.Layer s s :=
  { kind := "Double"
    stateShapes := []
    initState := .nil
    forward := fun _ {α} _ _ => fun {m} _ _ => fun x =>
      Ops.apply (m := m) (α := α)
        (.add (s := s)) (x, x, ()) }
```

In `forward`, the binders are the mode, the scalar type with its instances, and the execution monad
with its `Ops` instance. Under the pure interpretation, the layer runs immediately:

```lean (name := sqDoubleRun)
-- Run the pure interpretation with the empty parameter
-- pack.
#eval (nn.Sequential.fromLayer (sqDouble [3])).eval
  (α := Float) .nil [1.0, -2.0, 3.5]
```

```leanOutput sqDoubleRun (whitespace := lax)
[2.000000, -4.000000, 7.000000]
```

The layer's meaning at every scalar is the sum of the input with itself, by definition:

```lean (name := sqDoubleSpec)
-- The pure meaning of the program, for every scalar and
-- every input.
example {α : Type} [Storage α] [Context α]
    (x : Tensor α [3]) :
    (nn.Sequential.fromLayer (sqDouble [3])).eval .nil x =
      x + x := rfl
```

There is no second field holding a separate mathematical meaning that could disagree with the
program. Since the pure interpretation is the program read by the pure interpreter, a theorem about
`Sequential.eval` is a theorem about the code the runtimes run, up to each runtime's own agreement
with the operators' rules. Derivative correctness would need its own statement.

# Residual Blocks And Shared Values

A residual block uses its input twice:

$$`r(x)=\operatorname{ReLU}(Wx+b+x).`

`nn.residual inner` is the one-layer model whose program runs `inner` and adds the original input
to its output {Informal.citep resnet2016}[]:

```lean (name := sqResidualType)
-- The residual layer keeps the parameter shapes of the
-- inner model.
#check @nn.residual
```

```leanOutput sqResidualType (whitespace := lax)
@nn.residual : {s : Shape} → Sequential s s → Sequential s s
```

Run it on parameters chosen so the answer can be checked by hand: the swap matrix, a bias, and a
small input.

```lean (name := sqResidualRun)
-- These values distinguish the intended skip connection
-- from accidental reuse of the linear output.
#eval do
  let w : Tensor Float [2, 2] := [[0.0, 1.0], [1.0, 0.0]]
  let b : Tensor Float [2] := [-3.0, 0.5]
  let x : Tensor Float [2] := [1.0, 2.0]
  let block : nn.Sequential [2] [2] :=
    nn.residual (.fromLayer (Layer.linear 2 2)) >>>
      .fromLayer (Layer.op .relu)
  let y := block.eval (.cons w (.cons b .nil)) x
  IO.println s!"residual = {y}"
```

```leanOutput sqResidualRun (whitespace := lax)
residual = [0.000000, 3.500000]
```

By hand: $`Wx=[2,1]`, then $`Wx+b=[-1,1.5]`, then adding the skip gives $`[0,3.5]`, and ReLU leaves
it alone. We get exactly zero in the first coordinate because the skip cancelled the bias. A wrong
sharing pattern (for instance, one that added the output of the linear branch to itself) would give
$`[-2,3]` before ReLU and $`[0,3]` afterwards.

Sharing needs no graph syntax. A layer's program is a `do` block over `Ops`: a value bound once by
`←` can be read by any later operation, and the residual program reads the input after the inner
model has run. A computation with several results is a function over `Ops` that returns several
references. Recurrent cells are written this way, and the recurrent layers and models run them step
by step. One LSTM step returns both the hidden and the cell state:

```lean (name := sqLstmCell)
-- One LSTM step returns two references: hidden and
-- cell state.
#check @Layer.lstmCell
```

```leanOutput sqLstmCell (whitespace := lax)
@Layer.lstmCell : {α : Type} →
  [inst : Storage α] →
    [inst_1 : Context α] →
      {m : Type → Type} →
        [Monad m] →
          [inst_3 : Ops m α] →
            {inputWidth hiddenWidth : ℕ} →
              Pack (Ref m α) (Layer.lstmCellShapes inputWidth hiddenWidth) →
                Ref m α [inputWidth] → Ref m α [hiddenWidth] × Ref m α [hiddenWidth] →
                  m (Ref m α [hiddenWidth] × Ref m α [hiddenWidth])
```

Operators are typed the same way, and their types separate the shape questions they answer.
Matrix multiplication takes the contracted dimension `n` once, shared by both operands, and two
witnesses that the operands' leading shapes broadcast to the output's batch shape:

```lean (name := sqMatmul)
-- Both operands share the contracted dimension `n`, and
-- the leading shapes broadcast to `batch` by two witnesses.
#check @Op.matmul
```

```leanOutput sqMatmul (whitespace := lax)
@Op.matmul : {batchA batchB batch : Shape} →
  {m n p : ℕ} →
    (lv rv : Bool) →
      (lv = true → m = 1) →
        (rv = true → p = 1) →
          batchA.CanBroadcastTo batch →
            batchB.CanBroadcastTo batch →
              Op [Port.value (Shape.matmulLeft lv batchA m n),
                  Port.value (Shape.matmulRight rv batchB n p)]
                [Port.value (Shape.matmulOut lv rv batch m p)]
```

Each scalar dot product is meaningful only if the contracted dimension matches, and the broadcast
witnesses choose the pair of matrices for every output batch coordinate. One operator covers the
shared-batch, pairwise-batched, and unbatched cases, and a vector operand is a flag with a unit
dimension (`lv` with `m = 1`, `rv` with `p = 1`), so no runtime branch guesses the case. The type
determines the forward indexing. A reverse rule must also sum the contributions of a broadcast
operand's repeated uses back into that operand: with matrix operands the operator's backward is
`Tensor.matmulBackward` (`Op.rule?_matmul_matrix`), which reduces the common-batch gradients to
each operand's own batch shape.

# Convolution And Pooling Shapes

Dense layers aren't the whole vocabulary. Convolution and pooling layers take a batch size, a
spatial rank, channel counts, and per-axis kernel, stride, and padding sizes, and their output
shapes are computed in the type. Here are two spatial stages for a batch of one image, with their
different channel widths:

```lean (name := sqCnnFeatures)
/-- Two spatial stages ending in three 2-by-2 maps. -/
def sqCnnFeatures :
    nn.Sequential [1, 1, 8, 8] [1, 3, 2, 2] :=
  nn.Sequential.fromLayer
    (Layer.conv 1 2 1 2 [3, 3] [1, 1] [1, 1] [8, 8]) >>>
  nn.Sequential.fromLayer (Layer.op .relu) >>>
  nn.Sequential.fromLayer
    (Layer.maxPool 1 2 2 [2, 2] [2, 2] [0, 0] [8, 8]) >>>
  nn.Sequential.fromLayer
    (Layer.conv 1 2 2 3 [3, 3] [1, 1] [1, 1] [4, 4]) >>>
  nn.Sequential.fromLayer (Layer.op .relu) >>>
  nn.Sequential.fromLayer
    (Layer.maxPool 1 2 3 [2, 2] [2, 2] [0, 0] [4, 4])

example : nn.Sequential [1, 1, 8, 8] [4] :=
  Sequential.classifier sqCnnFeatures 4
```

Padding 1 with a `3x3` kernel preserves the extent, so the two pooling stages halve `8` to `4` and
`4` to `2`, and three channels of `2x2` give twelve features. The feature model's output shape
determines the head's weight shape, and the head's parameters follow the four feature tensors:

```lean (name := sqCnnShapes)
-- The head's weight shape [4, 12] is computed from the
-- feature model's output shape.
#eval nn.Sequential.stateShapes
  (Sequential.classifier sqCnnFeatures 4)
```

```leanOutput sqCnnShapes (whitespace := lax)
[[2, 1, 3, 3], [2], [3, 2, 3, 3], [3], [4, 12], [4]]
```

Annotating `sqCnnFeatures` with its type asks Lean to check every intermediate shape, so an
inconsistent annotation gives the same kind of compile error as the width typo above. An identity
feature model gives a classifier on the flattened input:

```lean (name := sqClassifierIdentity)
example : nn.Sequential [2, 3] [4] :=
  Sequential.classifier (nn.Sequential.id [2, 3]) 4
```

`Sequential.eval_classifier` applies to both, since its feature model is arbitrary.

# Checkpoint Import And Parameter Mapping

A model's parameter ABI is a positional convention computed by `stateShapes`, while an imported
checkpoint is a bag of named arrays produced by another program. Connecting the two needs an
argument the types can't supply on their own:

1. parse the external names and arrays
2. check each concrete shape and finite-value condition
3. map values into the model's ABI
4. state or assume that this mapping matches the source framework's layout

Steps one to three are code, and {ref "external-tools-and-ffi"}[the boundary chapter] shows the
parser and its checks. Step four is a modeling assumption about someone else's file format, and the
swapped-bias experiment above shows what it looks like when that assumption is wrong: all keys
match, all shapes agree, the function is different. The import contract must record the mapping,
because the architecture's shape constraints can't authenticate training provenance.

Distinct parameter values help test the mapping: filling both biases with zero would have hidden
our swap. A passing sample still can't establish equality for all parameters. To apply the MLP
theorem to an imported model, the importer must connect its named tensors to the ordered parameter
pack in that theorem.

# Sequential Models And The Backend IR

Sequential model syntax and the backend IR are the two ends of the pipeline drawn at the start of
this page (between them, the typed graph records one program for repeated evaluation and
differentiation). They carry different information.

A sequential model is typed layer syntax. A model that connects incompatible shapes does not
elaborate, and each layer's program is both its pure meaning and what the runtimes execute.

`IR.Graph` is a DAG of registered operators whose nodes serialize to JSON
(`IR.Node.toJson`). A node is a graph input at a port, a constant at a port, or a registered
operator with its attributes and the references of the outputs it reads. A node's id is its
position, and constants and parameters live in a separate payload. That form lets importers
construct candidate graphs for validation, transformation, verification, and kernel selection. An
operator node declares no output shape, because its ports follow from its operator
(`IR.Node.outs`), but a candidate operator node can still have the wrong number of
references, or a reference that names no earlier output or one at a port its input doesn't accept,
so a validator (`IR.Graph.check`) must check them.

For the operators it supports, the verification lowering records a model's forward program as IR,
and it refuses the others. Semantic theorems for selected paths state which meaning is preserved.
{ref "graphs-and-ir"}[The next chapter] builds and inspects that lower-level representation
directly.

# Further Architecture Checks

Using the same models, these variations isolate shape composition, parameter order, scalar
interpretation, and semantic preservation.

1. Change the hidden width of one linear layer in the definition of `MLP.model` and elaborate the
   file. You'll get an error naming the two shapes that disagree, and it arrives before
   initialization, data loading, or execution.
2. Write a layer like `sqDouble` with a different program (for instance, one that subtracts the
   input from itself) and prove its pure meaning by `rfl`. Then evaluate it at `Float` and at `ℚ`.
3. Evaluate `MLP.model 3 3 3` with the two weight matrices exchanged instead of the biases, and
   predict the output before running it.
4. Run the tutorial with `--arithmetic ieee` and compare the two losses against the transcript
   above. A difference here is a floating-point story, and {ref "floats"}[the floating-point
   chapter] is where that story is told.
5. Change a pooling stride in `sqCnnFeatures` from `[2, 2]` to `[3, 3]` and update the following
   spatial annotations. Inspect the head's weight shape with `stateShapes` before annotating the
   classifier.
6. Apply `Sequential.eval_comp` to two models of your own. Work out why it needs no premise about
   the layers' programs, then state the additional agreement needed to transfer the pure equality
   to execution on the eager tape.
