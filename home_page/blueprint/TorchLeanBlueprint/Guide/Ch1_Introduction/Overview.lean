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
import TorchLean.Verify.Lowering
import TorchLean.NN.IR.CustomRule
import TorchLean.NN.IR.Tangent
import TorchLean.NN.IR.Program
import FloatLib.Floats
import TorchLean.Verify.Floats.FP32
import FloatLib.Floats.Formats.BinaryInterchange
import FloatLib.Floats.Formats.Flocq.Theory.Rounding.Affine
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.Roles
import TorchLean.NN.Loss
import TorchLean.NN.Session.Random
import TorchLean.Tensor.Op.Pointwise.Real
import TorchLean.Verify.Trainer

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean
open TorchLean Trainer.Objective
open Lean.Elab.Tactic.GuardMsgs.WhitespaceMode (lax)

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)

#doc (Manual) "Overview" =>
%%%
tag := "overview"
file := "The-Problem-TorchLean-Solves"
%%%

Suppose we train a small network, save its parameters, run it on a GPU, and later claim that its
output stays in a safe range for every input near a test point. To check that claim, a verifier
must analyze the parameters and computation that actually run. Architecture alone can't identify
them: initialization chooses parameter values, training changes them, and exporting or compiling
determines how the operations will be represented and evaluated.

A mismatch can be hard to detect. A checkpoint loader may transpose a weight matrix. A verification
script may forget that the model expects normalized inputs. A compiler may replace separate
multiplication and addition with an FMA, changing the result by an amount that a test's tolerance
hides. Each program still runs, but computes a different function from the one we had in mind.

We designed TorchLean around that problem. It's a neural-network library in which the architecture,
parameter payload, graph, arithmetic, and property can be named in the same language. Numerical
kernels may still run through optimized native providers, but their inputs, outputs, and
assumptions stay attached to the computation we're studying.

# Model Definition And Initialization

The model and the trainer below need two imports:

```
-- The trainer modules provide the model builder and the
-- trainer used below.
import TorchLean.NN.Trainer.Constructor
import TorchLean.NN.Trainer.Train

open TorchLean
```

When this page is built, its named Lean blocks are elaborated and their output is checked against
the displayed results. Our first declaration is a two-layer regression model:

```lean
-- The hidden shape [8] must connect the first affine layer,
-- ReLU, and final affine layer.
def model :
    nn.Builder (nn.Sequential [2] [1]) :=
  nn.Sequential![
    nn.linear 2 8,
    nn.op .relu,
    nn.linear 8 1
  ]
```

Our model accepts a length-two tensor and returns a length-one prediction. The dimensions occur in
the model's type, so the output of a layer with width seven can't be fed to a layer expecting width
eight. `nn.Builder` means that this is a seeded model builder, which describes initialization
without choosing the random seed or producing concrete parameter tensors.

We can initialize it ourselves:

```lean
-- Run the initialization builder with an explicit seed to
-- obtain a model value.
def initialized :=
  nn.build 2026 model
```

or ask the trainer to initialize and execute it:

```lean
-- Configure the loss and optimizer without performing a
-- training step.
def trainer :=
  Trainer.create model
    { objective := mse
      optimizer := optim.adam
        { lr := 0.03 }
      seed := 2026 }
```

The notation is intentionally familiar to a PyTorch user {Informal.citep pytorch2019}[], but Lean
learns more from the declaration. It checks the input and output dimensions while elaborating the
file, and it distinguishes the seeded builder from the initialized model. Once initialized, the
model exposes the exact order and shape of its parameter tensors, so the trainer, the graph
lowering pass, and the verifier can use that information later without rediscovering it from
runtime metadata.

Since the seed sits in the trainer's configuration (not in a global mutable generator), an
untrained prediction is a value we can reproduce with the same initialization and numerical
backend:

```lean (name := untrainedPrediction)
-- Observe the seeded model at one input before fitting any
-- data.
#eval trainer.predict ([0.25, -0.75] : Tensor Float [2])
```
```leanOutput untrainedPrediction
[-0.047640]
```

# Core Objects

With the declarations above we can inspect several stages of the model's lifecycle. We print the
types of the builder, the initialized model, the parameter state, and the lowering result, and
check the prediction function's single-input type:

```lean (name := ovSix)
-- Inspect the types at each transition: builder, model,
-- state, execution, and lowering.
#check model
#check initialized
#check nn.initialState initialized
example : Tensor Float [2] → IO (Tensor Float [1]) :=
  trainer.predict
#check Verification.lowerForwardToIR (α := Float)
  initialized (nn.initialState initialized)
```
```leanOutput ovSix
model : nn.Builder (nn.Sequential [2] [1])
```
```leanOutput ovSix
initialized : nn.Sequential [2] [1]
```
```leanOutput ovSix (whitespace := lax)
nn.initialState initialized Float : TensorPack Float
  initialized.stateShapes
```
```leanOutput ovSix (whitespace := lax)
Verification.lowerForwardToIR initialized
  (nn.initialState initialized Float) : Except String (IR.LoweredModel Float)
```

`model` and `initialized` have different types. A builder is a recipe, while a
`nn.Sequential [2] [1]` is a network with concrete parameter tensors in it. Code that needs one and
is handed the other doesn't compile. So initialization is its own step in the interface.

Here the parameter payload's type mentions the model it came from:
`TensorPack Float initialized.stateShapes`. That list of shapes is computed from `initialized`, not
supplied alongside it, and a payload with a different ordered list of state shapes has a different
type. But models with the same state shapes can still have different operations or parameter
values, so this check doesn't establish checkpoint identity.

`trainer.predict` returns `IO (Tensor Float [1])`, so using the runtime requires an effectful
computation. For the same network, the specification-level forward function is a plain total
function with no `IO`, and the {ref "running-example"}[running example] checks that the two agree
on a held-out point.

Lowering returns `Except String (IR.LoweredModel Float)`. Its error alternative reports unsupported
forward programs or invalid inputs, and its success alternative contains the lowered artifact,
whose semantics need a correctness argument of their own.

Each stage of the workflow has corresponding objects in Lean and PyTorch:

:::table +header
*
  * Stage of the workflow
  * TorchLean object
  * Usual PyTorch counterpart
*
  * source-level architecture
  * `nn.Builder (nn.Sequential [2] [1])`
  * the body of an `nn.Module.__init__`
*
  * initialized parameter tensors
  * `initialized`, with `nn.initialState initialized`
  * the constructed module and its `state_dict()`
*
  * mutable parameters after training
  * the live state owned by an opened `Trainer.Session`
  * the same `Parameter` objects, mutated by `optimizer.step()`
*
  * exported operation graph
  * `IR.Graph` plus an `IR.Payload α`
  * an `fx.GraphModule` or a `torch.export` program
*
  * the program that actually runs
  * the backend profile the runtime selects
  * whatever kernel ATen dispatch picks after `.to(device)`
*
  * the object a verifier analyzes
  * `IR.LoweredModel α`
  * an ONNX or VNN-LIB file handed to an external checker
:::

PyTorch also distinguishes these objects and provides conversions between them. TorchLean puts
additional information in their types, including the network's input and output shapes and the
ordered shapes of its state. Those checks rule out some conversion mistakes, but they leave
value-level questions open, such as whether a checkpoint contains the intended weights.

# Model Representations

Trainer, proof, and verifier each need a different view of the initialized model.

## The model description

An `nn.Sequential` stores layer definitions, parameter shapes, initialization values, gradient
flags, and a forward program, and it's the object the high-level trainer uses. Training creates an
effectful runtime runner whose parameters change over time, but it doesn't rewrite the source
definition.

## The equations

The specification layer defines tensors as shape-indexed objects and gives operations such as
matrix multiplication, softmax, normalization, and loss functions their mathematical meanings.
Proofs use these definitions, so they don't have to reverse-engineer an opaque native buffer. Those
definitions are stated against Mathlib's real analysis {Informal.citep mathlib2020}[], so a
TorchLean theorem about a gradient is a statement in Mathlib's analysis. Analytic gradient theorems
use `fderiv`, while algebraic JVP/VJP adjointness theorems have a separate role and require an
analytic bridge to identify a derivative.

## The running program

TorchLean's eager runtime owns autograd tape nodes, parameter and optimizer state, and optional
accelerator buffers. Each operation it records is a registered operator of one table,
`TorchLean.Op`, whose rule `Op.rule?` gives the operator's forward value and pullback. On the CPU
tape, `Tape.apply` records an operator through that rule. On the CUDA tape, `LibTorch.Tape.apply`
records it through its entry in `LibTorch.Kernel.ofOp`, which calls LibTorch kernels that no
theorem relates to the rule. A backend profile selects implementations without changing the source
model, and the backend chapter explains the profiles we maintain and what their evidence supports.

## The operation graph

A lowered graph (`IR.Graph`) is a straight-line graph of registered operators. Each node is
the graph input, a constant, or an operator with one reference per input port. Constants, weights
and similar data live in a separate payload, so you can inspect the structure without inspecting
every parameter.

For the two linear layers and ReLU above, we can inspect the node count and input and output ids:

```lean (name := ovGraphIds)
-- Keep lowering errors visible while extracting the graph
-- size and its input and output nodes.
def ovGraphFacts :
    Except String (Nat × Nat × Nat) := do
  let lowered ← Verification.lowerForwardToIR
    (α := Float) initialized
    (nn.initialState initialized)
  return (lowered.graph.nodes.size,
    lowered.input.node, lowered.output.node)

#eval ovGraphFacts
```
```leanOutput ovGraphIds
Except.ok (18, 0, 17)
```

So the graph has eighteen nodes, input id `0`, and output id `17`. Seventeen identifies a node, not
a dimension (the output still has shape `[1]`). Nor is the node count a parameter count: a constant
node can hold an entire weight matrix.

Two linear layers and a ReLU expand to eighteen nodes because reshapes, broadcasts, matrix
products, and bias additions each have their own nodes. A proof can follow these operations
one at a time using the equation attached to each node.

By itself, the graph structure doesn't choose an element type. Evaluation does:
`IR.Payload α` and the graph's evaluation use one `α` for the numeric input, parameters,
intermediates, and output. We explain in the {ref "tensors-shapes"}[tensor chapter] how that
arithmetic is selected and recorded.

Lowering builds this graph by recording the forward program's operations as registered operators of
`TorchLean.Op` in a typed graph (`IR.record`), then building that typed
graph's graph of registered operators node for node (`IR.OpGraph.toOp`). On every input, the
lowered graph returns the typed graph's value at the output (`IR.lowerForward_eq_ok`). That
recording is a construction, though: no theorem relates the recorded graph to the forward program
it came from. The PyTorch and JAX captures write typed programs of these registered operators as
JSON, in the `torchlean.ir` artifact format (`IR.Artifact`), and `IR.load` checks and loads them.

## Verification Artifacts

Verification lowering converts supported forward programs and a concrete parameter payload to a
`IR.LoweredModel`. Interval bound propagation (IBP) carries lower and upper bounds through the
operations, while CROWN uses affine bounds to retain more information about how values depend on
the input. Both operate on the lowered graph's import as a graph of registered operators
(`IR.LoweredModel.toImported`), and numerical certificates can replay bit-level ranges and backend
policies over it. Other checkers consume external artifacts such as alpha-beta-CROWN leaves
{Informal.citep betacrown2021}[] or PINN residual certificates.

## Comparing Runtime And Graph Evaluation

Shrink the verifier's input region to the single point
$`(0.25,-0.75)`. This is the input used for the earlier prediction, so we can compare runtime
evaluation with interval propagation through the lowered graph:

```lean (name := ovAgree)
-- Feed the same input to the runtime and to a zero-radius
-- graph-bound calculation.
#eval show IO Unit from do
  let lowered ←
    match Verification.lowerForwardToIR (α := Float)
        initialized (nn.initialState initialized) with
    | .ok c => pure c
    | .error e => throw <| IO.userError e
  let x0 : Tensor Float [2] := [0.25, -0.75]
  -- Radius zero: the "region" is one point.
  let xB := CROWN.FlatBox.lInfBall
    (α := Float) x0 0.0
  let imported := lowered.toImported
  let box ← IO.ofExcept <|
    imported.outputBox? (imported.runIBP xB)
  let y ← trainer.predict x0
  IO.println s!"runtime  = {y}"
  IO.println s!"IBP lo   = {box.lo}"
  IO.println s!"IBP hi   = {box.hi}"
```

```leanOutput ovAgree
runtime  = [-0.047640]
IBP lo   = [-0.047640]
IBP hi   = [-0.047640]
```

All three values agree at the displayed precision. This checks initialization, payload transfer,
lowering, interval propagation, and runtime execution on one input. If they differed, we could
first compare their states and inputs, then inspect the lowered operations and rounding choices.
A zero-radius box removes input variation but leaves those numerical choices in place.
Agreement at one point and six decimal places does not establish an enclosure theorem for host
`Float` execution, or rule out smaller differences.

Since the graph is itself a value, we can print its eighteen nodes. This is the level of detail the
verifier sees:

```lean (name := ovKinds)
-- Print the primitive operations hidden inside the three
-- high-level layers.
#eval show IO Unit from do
  let lowered ←
    match Verification.lowerForwardToIR (α := Float)
        initialized (nn.initialState initialized) with
    | .ok c => pure c
    | .error e => throw <| IO.userError e
  let nodes := lowered.graph.nodes
  for i in [0:nodes.size] do
    match nodes[i]? with
    | some (.input _ p) =>
        IO.println s!"{i}: input {repr p.shape}"
    | some (.const p) =>
        IO.println s!"{i}: const {repr p.shape}"
    | some (.op o refs) =>
        let tag := (TorchLean.Op.encode o).tag
        let parents := refs.toList.map (·.node)
        IO.println s!"{i}: {tag} {parents}"
    | none => pure ()
```

```leanOutput ovKinds (whitespace := lax)
0: input [2]
1: reshape [0]
2: const [8, 2]
3: permute [2]
4: matmul [1, 3]
5: const [8]
6: broadcast [5]
7: add [4, 6]
8: reshape [7]
9: relu [8]
10: reshape [9]
11: const [1, 8]
12: permute [11]
13: matmul [10, 12]
14: const [1]
15: broadcast [14]
16: add [13, 15]
17: reshape [16]
```

Each operator lists the nodes it reads, and each input and constant its shape. Follow the first
affine layer through the listing. Its input vector becomes a one-row matrix of shape `[1, 2]`.
Since the stored weight has shape `[8, 2]`, permuting its two dimensions gives the `[2, 8]` matrix
needed on the right of that row. Their product has shape `[1, 8]`. Broadcasting the length-eight
bias gives it the same shape before addition, and the last reshape removes the temporary row axis.
ReLU then acts on the eight entries. With eight inputs and one output, the second affine layer
repeats the same construction. These structural nodes spell out the storage convention.

A soundness proof can relate each of these primitives to its equation and compose the results
along the graph.

## Interval And Affine Bounds

With a positive radius, the library offers three analyses of that graph's import. Interval
propagation, the CROWN output box, and the backward objective pass all read the same nodes and the
same payload:

```lean (name := ovCrown)
-- Use one input box and payload for all three analyses of
-- the scalar output.
#eval show IO Unit from do
  let lowered ←
    match Verification.lowerForwardToIR (α := Float)
        initialized (nn.initialState initialized) with
    | .ok c => pure c
    | .error e => throw <| IO.userError e
  let x0 : Tensor Float [2] := [0.25, -0.75]
  let eps : Float := 1.0
  let xB := CROWN.FlatBox.lInfBall
    (α := Float) x0 eps
  let imported := lowered.toImported
  let ibp := imported.runIBP xB
  let ibpBox ← IO.ofExcept (imported.outputBox? ibp)
  let crown ← IO.ofExcept (imported.outputBoxCROWN? xB)
  let obj : CROWN.Graph.FlatTensor Float :=
    { n := 1, v := Tensor.full [1] 1.0 }
  IO.println s!"IBP      = {ibpBox.lo}, {ibpBox.hi}"
  IO.println s!"CROWN    = {crown.lo}, {crown.hi}"
  match imported.backwardObjectiveBox? ibp xB obj with
  | .ok b => IO.println s!"backward = {b.lo}, {b.hi}"
  | .error m => IO.println s!"backward: {m}"
```

```leanOutput ovCrown
IBP      = [-1.099108], [0.227352]
CROWN    = [-1.099108], [0.227352]
backward = [-1.099108], [0.227352]
```

One extra ingredient goes into the backward pass: `obj` has one coefficient, equal to one, so the
objective is the scalar network output itself. In a classifier, a vector of coefficients could
instead ask about a difference between two logits. Here all three passes compute bounds for the
same scalar.

In the displayed transcript all three produce the same bounds. TorchLean's
{src "TorchLean/Verify/CROWN/Op/ExecCROWN.lean"}[CROWN pass] sweeps interval coefficients backward
from the output to the input, rounding outward at every scalar. Sums, reshapes, permutations, and
products with a constant operand pass their coefficients on, and a node without such a rule (ReLU
among them) bounds its coefficient by its IBP box.

So on this network each ReLU enters the bound as its interval, which is why the three bounds agree
here. For a crossing-zero ReLU, the α-CROWN relaxation belongs to the forward pass, which
`Op.outputBoxAlphaCROWN?` runs only at scalars with exact affine reassociation, and `Float` isn't
one of them.

Replacing an affine form with its enclosing interval can lose tightness, and the validity of that
interval still depends on the arithmetic and enclosure hypotheses. There's a theorem for the
real-valued ReLU relaxation, and {ref "motivation"}[the motivation chapter] proves its smallest
instance by hand. Products of two bounded operands (`mul`, `matmul`, `scaledMatmul`) use McCormick
planes, while softmax nodes fall back to IBP enclosures. So the agreement displayed here describes
this particular network and input box, not every graph the pass accepts.

External verifiers can also propose sharper bounds. Their artifacts need separate checks:
{ref "certificates"}[the certificates chapter] distinguishes recomputing an enclosure from checking
an α,β-CROWN leaf artifact's consistency {Informal.citep betacrown2021}[]. Accepting the latter
does not by itself prove the proposed numerical bound.

Reading an interval also requires a property to compare it with. Suppose we wanted to establish
that the output stays below zero throughout this input box. Then the displayed upper endpoint,
`0.227352`, wouldn't establish that claim, and it wouldn't refute the claim by itself either: a
bound can include values the network never attains. A counterexample would need an actual input and
its output. For a proof of the upper bound, we'd instead need an enclosure whose upper endpoint
satisfies the requested threshold, together with the soundness argument connecting that enclosure
to the intended computation.

Here the analysis has succeeded in producing a box, but that box doesn't settle the requested
inequality. A loose bound, an execution error, and a counterexample call for different follow-ups.

# Shape Contracts In A Loss Function

A TorchLean tensor has an element type $`\alpha` and a shape $`s`:

$$`\operatorname{Tensor}\;\alpha\;s`.

For example,

```lean
-- The leading extent agrees, but a vector of labels lacks
-- the final prediction axis.
def predictions : Tensor Float [32, 1] :=
  Tensor.full [32, 1] 0.0

def labels : Tensor Float [32] :=
  Tensor.full [32] 0.0
```

These are different types, and Lean says so before anything runs:

```lean +error (name := shapeClash)
-- A singleton axis is still part of the type, so it is not
-- removed implicitly.
example : Tensor Float [32] := predictions
```
```leanOutput shapeClash
Type mismatch
  predictions
has type
  Tensor Float [32, 1]
but is expected to have type
  Tensor Float [32]
```

A loss that expects equal shapes can't silently broadcast the label vector across the second axis.
Mean squared error on two tensors of the same shape is fine:

```lean (name := ovLossOk)
-- Matching prediction and target shapes permit a scalar
-- mean squared error.
#check Tensor.meanSquaredError predictions predictions
```
```leanOutput ovLossOk
predictions.meanSquaredError predictions : Float
```

Mixing the two shapes isn't, and the message names the argument:

```lean +error (name := ovLossClash)
-- The loss requires a common shape, so it cannot broadcast
-- these labels silently.
example : Float :=
  Tensor.meanSquaredError predictions labels
```
```leanOutput ovLossClash
Application type mismatch: The argument
  labels
has type
  Tensor Float [32]
but is expected to have type
  Tensor Float [32, 1]
in the application
  predictions.meanSquaredError labels
```

With all-zero tensors, even incorrectly paired entries would give zero loss, but the type error
exposes the mismatch without relying on the values. Under broadcasting, the `[32]` labels would
stretch across the second axis and subtraction would produce a `[32, 32]` matrix of all pairwise
residuals. On nonzero data, that loss could decrease while comparing every prediction with every
label.

TorchLean requires us to reshape the labels, squeeze the predictions, or select another loss.
That choice should follow the intended pairing of samples and targets.

# Limits Of Shape Checking

Shape typing checks dimensions, leaving properties of the entries to other checks. A square
matrix and its transpose, for example, have exactly the same tensor type.

We write the axis exchange in the einops-style notation from the tensor chapter:

```lean (name := ovTranspose)
-- A nonsymmetric square matrix exposes a transpose that
-- shape checking cannot detect.
def ovW : Tensor Float [2, 2] :=
  [[0.5, 1.5], [2.5, 3.5]]

open TorchLean.Tensor in
/-- The same four numbers, read the other way round. -/
def ovWT : Tensor Float [2, 2] :=
  rearrange ovW "i j -> j i"

def ovIn : Tensor Float [2] := [1.0, 0.0]

#eval Tensor.matvec ovW ovIn
#eval Tensor.matvec ovWT ovIn
```
```leanOutput ovTranspose
[0.500000, 2.500000]
```
```leanOutput ovTranspose
[0.500000, 1.500000]
```

In a matrix-vector product, the input `[1, 0]` selects the first column. Before transposition that
column contains 0.5 and 2.5, and afterwards it contains 0.5 and 1.5. A square matrix makes the
shape error invisible, while a nonsymmetric payload makes the semantic error visible. That gives us
a useful way to design an importer check: choose entries that distinguish row and column
conventions, then choose an input that isolates one of them. A matrix of all zeros or a symmetric
matrix would let the wrong convention pass this particular check.

A loader that reads this row-major buffer as column-major would introduce exactly that transpose
while preserving the tensor type `Tensor Float [2, 2]`.

It helps to remember what the contraction is doing. `Tensor.matvec` sums over the second axis,
so entry $`i` of the result is the dot product of row $`i` with the input:

```lean (name := ovDot)
-- Use basis and all-ones inputs to make contraction
-- directions easy to identify.
#eval Tensor.dot ovIn ovIn
#eval Tensor.matvec ovW (Tensor.full [2] 1.0)
```
```leanOutput ovDot
1.000000
```
```leanOutput ovDot
[2.000000, 6.000000]
```

Against the all-ones input the result is the vector of row sums, `[0.5 + 1.5, 2.5 + 3.5]`.
Feeding the transposed matrix the same input gives the column sums instead.

What catches a transposed checkpoint is a claim about values, not about shapes: an equation between
the tensor that was saved and the tensor that was loaded, or an end-to-end comparison against a
reference run. We do the second of those against real PyTorch state dicts in the
{ref "pytorch-roundtrip"}[round trip chapter], and the failure that comparison guards against is
this one.

Units, label quality, normalization, finiteness, and generalization need their own definitions and
hypotheses, because a tensor shape carries none of those guarantees.

A tensor type names its scalar parameter too. `Tensor α s` is homogeneous: every element of that
tensor has type `α`. We develop that point once, in the tensor chapter, alongside the distinction
between specification scalars and executable runtime arithmetic.

# Input Normalization

An input can also have the right shape and the wrong units. If the model was trained on normalized
features, passing raw measurements changes the prediction even though the call typechecks.

Suppose the two features were standardized during training using the training set's means and
standard deviations:

```lean (name := ovNormDefs)
-- Normalize each feature with its own mean and standard
-- deviation.
def ovMean : Tensor Float [2] := [10.0, 100.0]
def ovStd : Tensor Float [2] := [2.0, 50.0]

def ovNormalize (x : Tensor Float [2]) :
    Tensor Float [2] :=
  [(x[0] - ovMean[0]) / ovStd[0],
   (x[1] - ovMean[1]) / ovStd[1]]

def ovRaw : Tensor Float [2] := [11.0, 125.0]

#eval ovNormalize ovRaw
```
```leanOutput ovNormDefs
[0.500000, 0.500000]
```

In both features, the raw measurement sits half a standard deviation above the mean. We ask the
model about it once through the normalizer and once in raw units, which is the mistake:

```lean (name := ovNormBug)
-- Compare the normalized measurement with the same numbers
-- sent in raw units.
#eval trainer.predict (ovNormalize ovRaw)
#eval trainer.predict ovRaw
```
```leanOutput ovNormBug
[-0.071120]
```
```leanOutput ovNormBug
[-63.560242]
```

Depending on whether normalization is applied, the same raw measurement leads to predictions about
three orders of magnitude apart. Yet both calls accept `Tensor Float [2]` and return
`IO (Tensor Float [1])`. A shape can't distinguish raw coordinates from normalized coordinates, so
the caller needs a separate contract specifying which representation the model expects.

For verification, we must transform the whole input region as well as its center. An
$`\ell^\infty` ball of radius $`\varepsilon` allows each coordinate to vary by at most that amount.
Dividing each coordinate by its standard deviation also divides its permitted variation:

```lean (name := ovNormRadius)
-- Divide a raw perturbation radius by each feature scale
-- separately.
#eval (0.1 / ovStd[0], 0.1 / ovStd[1])
```
```leanOutput ovNormRadius
(0.050000, 0.002000)
```

Around the normalized center `[0.5, 0.5]`, these half-widths describe the rectangle `[0.45, 0.55]`
by `[0.498, 0.502]`. In normalized units the second feature has a much narrower range, because its
raw scale is larger. Replacing both half-widths by 0.05 would enlarge the region in that feature
(potentially making a bound much looser), while replacing both by 0.002 would fail to cover the
intended first-feature perturbations.

A ball of radius $`0.1` in raw units becomes a box of half-widths $`0.05` and $`0.002` in
normalized units. It's still a box, because the normalizer here is diagonal, but it's no longer a
*ball*: the two features are scaled by different amounts, so the certificate produced for
$`\|x-x_0\|_\infty\leq 0.1` in one coordinate system isn't a certificate for the same inequality in
the other. Conversely, a normalized radius of $`0.1` corresponds to raw half-widths $`0.2` and
$`5`. So the claim must state which coordinate system and which per-feature widths it covers.

We can account for this transformation in two ways. Either the normalizer is part of the lowered
forward program, so `eval` includes it and the input box is quantified over raw measurements, or it
isn't, and then the normalization becomes a hypothesis about the box being verified. Because the
graph and the box are separate arguments to the checker, they can be inspected separately. That's
why the {ref "verification"}[verification chapters] state the property against a named graph and a
named region, and the statistics that define a normalizer come from {ref "datasets-loaders"}[the
datasets chapter].

# Fused Multiply-Add

Replacing `x * y + z` with a single fused multiply-add can change the answer, because the fused
operation rounds only once. For this kind of question the float chapters use FloatLib's executable
semantics, which is available with one more import:

```
-- Import the executable binary32 operations used to
-- distinguish fused arithmetic.
import FloatLib

open FloatLib.Floats
```

Take $`a = 1 + 2^{-12}`, which is representable in binary32, and $`b = -1`:

```lean (name := ovFmaDefs)
-- Choose exactly representable inputs whose product needs
-- an extra significand bit.
def ovA : ExecFloat.Binary 8 23 :=
  1 + Rat.cast (1 / 4096 : Rat)

def ovB : ExecFloat.Binary 8 23 := -1
```

We compute $`a \cdot a + b` with separate operations and with a fused operation:

```lean (name := ovFma)
-- Compare two rounding schedules by equality, encoded
-- words, and their scaled difference.
#eval (ovA * ovA + ovB) ==
  ExecFloat.Binary.fma ovA ovA ovB .nearest
#eval (ExecFloat.Binary.toBits32 (ovA * ovA + ovB),
  ExecFloat.Binary.toBits32
    (ExecFloat.Binary.fma
      ovA ovA ovB .nearest))
#eval
  ((ExecFloat.Binary.toFloat32
      (ExecFloat.Binary.fma
        ovA ovA ovB .nearest)).toFloat
    - (ExecFloat.Binary.toFloat32
      (ovA * ovA + ovB)).toFloat) * 1000000000.0
```
```leanOutput ovFma
false
```
```leanOutput ovFma
(973078528, 973079552)
```
```leanOutput ovFma
59.604645
```

Cancellation explains the large ULP count. That product is close to one, but subtracting one leaves
a much smaller result with much finer spacing between representable numbers, so a small rounding
error in the product can span many ULPs of the result.

In exact arithmetic the product is $`a \cdot a = 1 + 2^{-11} + 2^{-24}`, and that value needs
twenty-five significand bits, one more than binary32 has. So the separate multiply rounds it to
$`1 + 2^{-11}`, after which adding $`-1` is exact and the answer is $`2^{-11}`. A fused
multiply-add keeps the full product, rounds once at the end, and returns $`2^{-11} + 2^{-24}`.
Those two results differ by $`2^{-24} \approx 5.96 \times 10^{-8}`, which is $`1024` ULPs at this
result's magnitude ($`\operatorname{ulp}(2^{-11})=2^{-34}`). That's the third `#eval` above,
reported as a multiple of $`10^{-9}` because `Float.toString` prints six decimals and both results
print as `0.000488`.

A regression test with absolute tolerance $`10^{-6}` can't see this difference. Larger computations
need an error analysis that accounts for their operation schedule. And compiler contraction
settings can permit this substitution. The CompCert work on verified compilation of floating-point
computations {Informal.citep boldo2015}[] treats it as a case the semantics has to model, not as
an optimizer bug. So a numerical guarantee must account for whether the execution permits fusion.

FloatLib gives `*`, `+`, and `fma` separate specifications, so a theorem has to commit to a
rounding schedule. We develop that semantics in the {ref "floats"}[floating-point chapter], compare
the design against the Rocq Flocq library {Informal.citep flocq2011}[], and state which backend
contracts assume that no fusion happened. For the underlying arithmetic, Goldberg's survey
{Informal.citep goldberg1991}[] explains rounding and the effect of intermediate precision.

# Output Properties

Suppose the trained model has parameters $`\theta`, its graph is $`g`, and the two input features
vary in a box $`B`. Let $`y^\star` be the desired output and $`\delta` the allowed error. Writing
`eval` for evaluation of the graph with those parameters, a range statement might be

$$`\forall x\in B,\qquad
  |\operatorname{eval}(g,\theta,x)_0-y^\star|\leq\delta`.

This line records the details that are easy to lose in prose:

- $`g` identifies the operation graph
- $`\theta` identifies the concrete trained parameters
- $`B` identifies the quantified input convention
- `eval` identifies the scalar and operator semantics
- index $`0`, target $`y^\star`, and tolerance $`\delta` identify the output property.

In that formula, `eval` isn't shorthand for "however the graph happens to run". It's a function in
the library, and its type lists the same ingredients:

```lean (name := ovEval)
-- Expose all arguments that determine a graph evaluation,
-- including its scalar semantics.
#check @IR.Graph.eval
```
```leanOutput ovEval (whitespace := lax)
@IR.Graph.eval : {α : Type} →
  [inst : Storage α] →
    [Context α] →
      [Op.Capabilities α] →
        IR.Graph →
          IR.Payload α →
            List (Op.Value α) →
              Except String (List (Op.Value α))
```

Those arguments fix the scalar type, its storage and arithmetic instances, and the optional
capabilities that some operators read (`Op.Capabilities`), then the graph, the payload, and the
list of graph inputs. Evaluation returns the graph's output values as a list. Our lowered model has
one output, the length-one vector of node `17`, and the formula reads its coordinate `0`.
`IR.Graph.run` returns the outputs of every node.

Each `Op.Value α` pairs a tensor with its port, because a graph supplied as data records its ports
in its nodes, not in its type. `Except String` reports the first error that stops an evaluation,
such as a missing graph input or constant, a reference that names no output, a value at a port that
the node reading it doesn't accept, or an operator whose attribute or value-domain check fails.
Evaluation gets the operations it needs from the `Storage`, `Context`, and `Op.Capabilities`
instances, but a bound on the returned values still requires a soundness argument.

An interval pass can compute lower and upper output bounds, and a Boolean check can then confirm
that the whole interval lies in $`[y^\star-\delta,y^\star+\delta]`. Last comes a soundness theorem
saying that the computed interval really encloses `eval` for this graph. We didn't invent interval
propagation or its linear-relaxation refinement: the shapes of those bounds come from the
robustness-verification literature {Informal.citep wongkolter2018 crown2018 autolirpa2020}[].
What TorchLean adds is a Lean proof that the propagation step really encloses the semantics it
claims to.

For the interval passes above (`CROWN.Op.runIBP`), that theorem is
`TorchLean.CROWN.Op.runIBP_encloses`: every box the pass computes from an input box
encloses its node's value in a successful real run of the graph, not in a native run, given lawful
endpoint operations (`LawfulNonlinearBoundOps`), an input box that encloses graph input `0`, and a
real payload that holds the real values of the rounded constants. A node whose operator has no
interval rule gets no box. `ℝ` and the rounded binary32 model `FP32` have lawful instances, but
host `Float` has none, so the `Float` passes above are outside the theorem.

Throughout the guide we use this vocabulary:

:::table +header
*
  * Evidence
  * What it establishes
*
  * a successful run
  * one execution produced a value
*
  * a parser or shape check
  * an artifact satisfies a structural predicate
*
  * a certificate replay
  * an artifact satisfies the checker's acceptance predicate
*
  * a soundness theorem
  * the accepted predicate implies a semantic proposition
*
  * a backend assumption
  * an external implementation is being trusted to meet a stated contract
:::

Nothing in that table is unique to machine learning. It is the proof-carrying-code discipline
{Informal.citep necula1997}[] applied to tensor programs: an untrusted producer
emits an artifact, a small checker accepts or rejects it, and a theorem relates acceptance to the
property we actually care about.

# Native Kernels And Backend Contracts

Numerical work can still use established native kernels. PyTorch provides operator coverage,
distributed training, compilers, and pretrained models that a project may already depend on.

TorchLean calls LibTorch's ATen operators for CUDA computation while retaining its own model,
parameter layout, tape, and backward traversal.
{ref "backend-selection"}[Backend Selection]
explains how each operation acquires an implementation and a contract stating the assumptions
that connect it to graph semantics.

# Imports

Application code imports the modules it uses, since no single import covers a whole library.
Building and training a model takes the two trainer modules:

```
-- The trainer modules also bring in tensors, the model
-- builder and the optimizers.
import TorchLean.NN.Trainer.Constructor
import TorchLean.NN.Trainer.Train
open TorchLean
```

A file that also verifies a model adds a module of `TorchLean.Verify`, such as the lowering
interface:

```
-- Verification is in TorchLean.Verify, which the trainer
-- modules don't import.
import TorchLean.Verify.Lowering
open TorchLean
```

TorchLean's Lean code is in four libraries, each built only on the ones named before it.
`TorchLean.Tensor` holds tensors, shapes, the operator table and the layer specifications.
`TorchLean.CUDA` adds the LibTorch bindings and the CUDA kernel emitter. `TorchLean.NN` has the
runtime and autograd, the application interface, the IR and the model specifications, and
`TorchLean.Verify` has bound propagation, certificates, the floating-point error proofs and the
verification interface. The worked examples are in `TorchLeanExamples` and the tests in
`TorchLeanTest`.

# References

TorchLean's frontend notation follows PyTorch {Informal.citep pytorch2019}[], and the analysis
our specification layer is stated against is Mathlib {Informal.citep mathlib2020}[]. The fused
multiply-add example is the situation modelled by {Informal.citet boldo2015}[], and
{Informal.citet goldberg1991}[] explains the arithmetic behind it. In the floating-point chapters
we compare against the Rocq library of {Informal.citet flocq2011}[]. Bound propagation and its
refinements come from {Informal.citet wongkolter2018}[], {Informal.citet crown2018}[],
{Informal.citet autolirpa2020}[], and {Informal.citet betacrown2021}[]. Our evidence vocabulary is
the proof-carrying-code discipline of {Informal.citet necula1997}[].
