import VersoManual
import NN.API
import NN.API.Verification.Lowering
import NN.IR
import NN.Floats
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.Roles

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean
open TorchLean
open Lean.Elab.Tactic.GuardMsgs.WhitespaceMode (lax)

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)

#doc (Manual) "Overview" =>
%%%
tag := "overview"
file := "The-Problem-TorchLean-Solves"
%%%

Suppose we train a small network, save its parameters, run it on a GPU, and later claim that its
output stays in a safe range for every input near a test point. The verifier must analyze the
parameters and computation that actually run. The architecture alone cannot identify them:
initialization chooses parameter values, training changes them, and exporting or compiling
determines how the operations will be represented and evaluated.

A mismatch can be hard to detect. A checkpoint loader may transpose a weight matrix. A verification
script may forget that the model expects normalized inputs. A compiler may replace separate
multiplication and addition with an FMA, changing the result by an amount that a test's tolerance
hides. Each program still runs, but computes a different function from the one we had in mind.

TorchLean was designed around that problem. It is a neural-network library in which the
architecture, parameter payload, graph, arithmetic, and property can be named in the same language.
Numerical kernels may still run through optimized native providers; their inputs, outputs, and
assumptions remain attached to the computation being studied.

# Model Definition And Initialization

Application code uses the focused `NN.API` import:

```
-- The application interface provides the model builder and
-- trainer used below.
import NN.API

open TorchLean
```

The named Lean blocks are elaborated when this page is built, and their output is checked against
the displayed results. The first declaration is a two-layer regression model:

```lean
-- The hidden shape [8] must connect the first affine layer,
-- ReLU, and final affine layer.
def model :
    nn.Builder (nn.Sequential [2] [1]) :=
  nn.Sequential![
    nn.linear 2 8,
    nn.relu,
    nn.linear 8 1
  ]
```

The model accepts a length-two tensor and returns a length-one prediction. The dimensions occur in
the model's type, so the output of a layer with width seven cannot be fed to a layer expecting width
eight. `nn.Builder` means that this is a seeded model builder. It describes initialization but has
not yet chosen the random seed or produced concrete parameter tensors.

We can initialize it directly:

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
  Trainer.new model
    { objective := .mse
      optimizer := optim.adam
        { learningRate := 0.03 }
      seed := 2026 }
```

The notation is intentionally familiar to a PyTorch user
({Informal.citet pytorch2019}[]), but Lean learns more from the declaration.
It checks the input and output dimensions while elaborating the file. It distinguishes the seeded
builder from the initialized model. Once initialized, the model exposes the exact order and shape of
its parameter tensors. That information is available later to the trainer, graph lowering pass, and
verifier without rediscovering it from runtime metadata.

Because the seed is part of the trainer's configuration rather than a global mutable generator, an
untrained prediction is a value we can reproduce with the same initialization and numerical backend:

```lean (name := untrainedPrediction)
-- Observe the seeded model at one input before fitting any
-- data.
#eval trainer.predict ([0.25, -0.75] : Tensor Float [2])
```
```leanOutput untrainedPrediction
[-0.088261]
```

# Core Objects

The declarations above let us inspect several stages of the model's lifecycle. We print the
builder, initialized model, parameter state, and lowering result types, and check the prediction
function's single-input type:

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
nn.initialState initialized Float : nn.State Float
  (Runtime.Autograd.Model.Layers.Seq.stateShapes initialized)
```
```leanOutput ovSix (whitespace := lax)
Verification.lowerForwardToIR initialized
  (nn.initialState initialized Float) : Except String (NN.Verification.Builtin.LoweredIR Float)
```

`model` and `initialized` have different types. A builder is a recipe; a `nn.Sequential [2] [1]` is
a network with concrete parameter tensors in it. Code that needs one and is handed the other does
not compile. Initialization is therefore an explicit step in the interface.

The type of the parameter payload mentions the model it came from:
`nn.State Float (Runtime.Autograd.Model.Layers.Seq.stateShapes initialized)`. The list of shapes is
computed from `initialized`, not supplied alongside it. A payload with a different ordered list of
state shapes has a different type. Models with the same state shapes can still have different
operations or parameter values, so this check does not establish checkpoint identity.

`trainer.predict` returns `IO (Tensor Float [1])`, so using the runtime requires an effectful
computation. The specification-level forward function of the same network is a plain total function
with no `IO`, and the {ref "running-example"}[running example] checks that the two agree on a
held-out point.

Lowering returns `Except String (LoweredIR Float)`. The error alternative reports unsupported
forward programs or invalid inputs; the success alternative contains the lowered artifact.
Its semantics still require a separate correctness argument.

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
  * `NN.IR.Graph` plus an `NN.IR.Payload α`
  * an `fx.GraphModule` or a `torch.export` program
*
  * the program that actually runs
  * the backend profile the runtime selects
  * whatever kernel ATen dispatch picks after `.to(device)`
*
  * the object a verifier analyzes
  * `LoweredIR α`
  * an ONNX or VNN-LIB file handed to an external checker
:::

PyTorch also distinguishes these objects and provides conversions between them. TorchLean places
additional information in their types, including the network's input and output shapes and the
ordered shapes of its state. These checks rule out some conversion mistakes; they still leave
value-level questions, such as whether a checkpoint contains the intended weights.

# Model Representations

The trainer, proof, and verifier each need a different view of the initialized model.

## The model description

An `nn.Sequential` stores layer definitions, parameter shapes, initialization values, gradient
flags, and a forward program. This is the object used by the high-level trainer. Training creates an
effectful runtime runner whose parameters change over time; it does not rewrite the source
definition.

## The equations

The specification layer defines tensors as shape-indexed objects and gives operations such as
matrix multiplication, softmax, normalization, and loss functions explicit mathematical meanings.
Proofs use these definitions rather than reverse-engineering an opaque native buffer. The real
analysis those definitions are stated against comes from Mathlib
({Informal.citep mathlib2020}[]), so a TorchLean theorem about a gradient is a
statement in Mathlib's analysis. Analytic gradient theorems use `fderiv`; algebraic JVP/VJP
adjointness theorems have a separate role and require an analytic bridge to identify a derivative.

## The running program

The eager runtime owns autograd tape nodes, parameter and optimizer state, and optional accelerator
buffers. A backend profile selects implementations without changing the source model. The backend
chapter explains the maintained profiles and what their evidence supports.

## The operation graph

`NN.IR.Graph` is a directed acyclic graph of operation-tagged nodes. Each node has parent ids and a
declared output shape. Constants, weights, convolution parameters, and similar data live in a
separate payload store, so inspecting the structure does not require inspecting every parameter.

For the two linear layers and ReLU above, we can inspect the node count and input and output ids:

```lean (name := ovGraphIds)
-- Keep lowering errors visible while extracting the graph
-- size and its boundary node ids.
def ovGraphFacts :
    Except String (Nat × Nat × Nat) := do
  let lowered ← Verification.lowerForwardToIR
    (α := Float) initialized
    (nn.initialState initialized)
  return (lowered.graph.nodes.size,
    lowered.inputId, lowered.outputId)

#eval ovGraphFacts
```
```leanOutput ovGraphIds
Except.ok (18, 0, 17)
```

The tuple gives eighteen nodes, input id `0`, and output id `17`. The output still has shape `[1]`;
seventeen identifies a node, not a dimension. Nor is the node count a parameter count: a constant
node can hold an entire weight matrix.

Two linear layers and a ReLU expand to eighteen nodes because reshapes, broadcasts, matrix
products, and bias additions each have their own nodes. A proof can follow these operations
one at a time using the equation attached to each node.

The graph structure itself does not choose an element type. Evaluation does: `NN.IR.Payload α` and
the graph denotation use one `α` for the numeric input, parameters, intermediates, and output. The
{ref "tensors-shapes"}[tensor chapter] explains how that arithmetic is selected and recorded.

## Verification Artifacts

The verification lowering converts supported forward programs and a concrete parameter payload
to a `LoweredIR`. Interval bound propagation (IBP) carries lower and upper bounds through the
operations. CROWN uses affine bounds to retain more information about how values depend on the
input. Both operate on the lowered graph. Numerical certificates can replay
bit-level ranges and backend policies over it. Other checkers consume external artifacts such as
alpha-beta-CROWN leaves ({Informal.citep betacrown2021}[]) or PINN residual
certificates.

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
  let ps := lowered.seedLInfBall x0 0.0
  let box ← lowered.outputBoxOrThrow (lowered.runIBP ps)
  let y ← trainer.predict x0
  IO.println s!"runtime  = {y}"
  IO.println s!"IBP lo   = {box.lo}"
  IO.println s!"IBP hi   = {box.hi}"
```

```leanOutput ovAgree
runtime  = [-0.088261]
IBP lo   = [-0.088261]
IBP hi   = [-0.088261]
```

All three values agree at the displayed precision. This checks initialization, payload transfer,
lowering, interval propagation, and runtime execution on one input. If they differed, we could
first compare their states and inputs, then inspect the lowered operations and rounding choices.
A zero-radius box removes input variation but leaves those numerical choices in place.
Agreement at one point and six decimal places does not establish an enclosure theorem for host
`Float` execution, or rule out smaller differences.

The graph itself is a value, so the eighteen nodes can simply be printed. This is the level of
detail the verifier sees:

```lean (name := ovKinds)
-- Print the primitive operations hidden inside the three
-- high-level layers.
#eval show IO Unit from do
  let lowered ←
    match Verification.lowerForwardToIR (α := Float)
        initialized (nn.initialState initialized) with
    | .ok c => pure c
    | .error e => throw <| IO.userError e
  for node in lowered.graph.nodes do
    IO.println s!"{node.id}: {repr node.kind}"
```

```leanOutput ovKinds (whitespace := lax)
0: NN.IR.OpKind.input
1: NN.IR.OpKind.reshape [2] [1, 2]
2: NN.IR.OpKind.const [8, 2]
3: NN.IR.OpKind.transpose 0 1
4: NN.IR.OpKind.matmul
5: NN.IR.OpKind.const [8]
6: NN.IR.OpKind.broadcastTo [8] [1, 8]
7: NN.IR.OpKind.add
8: NN.IR.OpKind.reshape [1, 8] [8]
9: NN.IR.OpKind.relu
10: NN.IR.OpKind.reshape [8] [1, 8]
11: NN.IR.OpKind.const [1, 8]
12: NN.IR.OpKind.transpose 0 1
13: NN.IR.OpKind.matmul
14: NN.IR.OpKind.const [1]
15: NN.IR.OpKind.broadcastTo [1] [1, 1]
16: NN.IR.OpKind.add
17: NN.IR.OpKind.reshape [1, 1] [1]
```

Follow the first affine layer through the listing. The input vector becomes a one-row matrix of
shape `[1, 2]`. The stored weight has shape `[8, 2]`, so transposing it gives the `[2, 8]` matrix
needed on the right of that row. Their product has shape `[1, 8]`. Broadcasting the length-eight
bias gives it the same shape before addition, and the last reshape removes the temporary row axis.
ReLU then acts on the eight entries. The second affine layer repeats the same construction with
eight inputs and one output. These structural nodes make the storage convention explicit.

A soundness proof can relate each of these primitives to its equation and compose the results
along the graph.

## Interval And Affine Bounds

With a positive radius, the library offers three analyses of that graph. Interval
propagation, the forward CROWN pass, and the backward objective pass all read the same nodes and the
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
  let xB := NN.Verification.Builtin.lInfBall
    (α := Float) x0 eps
  let ps := lowered.seedLInfBall x0 eps
  let ibp := lowered.runIBP ps
  let ibpBox ← lowered.outputBoxOrThrow ibp
  let fwd ← lowered.outputBoxCROWNOrThrow ps xB
  let obj : NN.MLTheory.CROWN.Graph.FlatTensor Float :=
    { n := 1, v := Tensor.full [1] 1.0 }
  IO.println s!"IBP      = {ibpBox.lo}, {ibpBox.hi}"
  IO.println s!"forward  = {fwd.lo}, {fwd.hi}"
  match lowered.backwardObjectiveBox? ps ibp xB obj with
  | .ok b => IO.println s!"backward = {b.lo}, {b.hi}"
  | .error m => IO.println s!"backward: {m}"
```

```leanOutput ovCrown
IBP      = [-1.033713], [0.242680]
forward  = [-1.033713], [0.242680]
backward = [-1.033713], [0.242680]
```

The backward pass takes one extra ingredient: `obj` has one coefficient, equal to one, so the
objective is the scalar network output itself. In a classifier, a vector of coefficients could
instead ask about a difference between two logits. Here all three passes compute bounds for the
same scalar.

All three produce the same displayed bounds here. The affine pass in
{src "NN/MLTheory/CROWN/Graph/Engine/CROWN/Node.lean"}[the CROWN engine] carries real affine forms
through `matmul`, `add`, `reshape`, and `transpose`, but at each ReLU it keeps the interval
enclosure as a constant affine form instead of the usual relaxation. The comment in the source says
why: the CROWN slope is a division, and until the executable arithmetic can supply directed
coefficients for it, computing that slope in host arithmetic would put an unproved rounding step
inside the bound. Since the only nonlinearity in this network is a ReLU, the affine information is
discarded at node `9` and everything after it is the interval answer again.

Replacing an affine form with its enclosing interval can lose tightness; the validity of that
interval still depends on the arithmetic and enclosure hypotheses. Tighter bounds in this library
come from two places, neither of them a floating-point CROWN slope computed
on the fly. The real-valued relaxation is available as a theorem, and
{ref "motivation"}[the motivation chapter] proves the smallest instance of it by hand. Sharper
executable bounds arrive as certificates from an outside search, which Lean then checks; that is
the arrangement {ref "certificates"}[the certificates chapter] uses for α,β-CROWN leaves
({Informal.citep betacrown2021}[]). Graphs with products or softmax do use the relaxations above,
because `mulElem` and two-argument `matmul` have McCormick transfers, so the agreement seen here is
a fact about ReLU networks rather than about the pass.

Reading an interval also requires a property to compare it with. Suppose we wanted to establish
that the output stays below zero throughout this input box. The displayed upper endpoint,
`0.242680`, would not establish that claim. It would not by itself refute the claim either: a
bound can include values the network never attains. A counterexample would need an actual input
and its output. For a proof of the upper bound, we would instead need an enclosure whose upper
endpoint satisfies the requested threshold, together with the soundness argument connecting that
enclosure to the intended computation.

The analysis has succeeded in producing a box, but that box does not settle the requested
inequality. A loose bound, an execution error, and a counterexample call for different next steps.

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
-- A singleton axis is still part of the type; it is not
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

A loss that expects equal shapes cannot silently broadcast the label vector across the second axis.
Mean squared error on two tensors of the same shape is fine:

```lean (name := ovLossOk)
-- Matching prediction and target shapes permit a scalar
-- mean squared error.
#check Spec.mseSpec predictions predictions
```
```leanOutput ovLossOk
Spec.mseSpec predictions predictions : Float
```

Mixing the two shapes is not, and the message names the argument:

```lean +error (name := ovLossClash)
-- The loss requires a common shape, so it cannot broadcast
-- these labels silently.
example : Float := Spec.mseSpec predictions labels
```
```leanOutput ovLossClash
Application type mismatch: The argument
  labels
has type
  Tensor Float [32]
but is expected to have type
  Tensor Float [32, 1]
in the application
  Spec.mseSpec predictions labels
```

With all-zero tensors, even incorrectly paired entries would give zero loss. The type error
exposes the mismatch without relying on the values. Under broadcasting, the `[32]` labels would
stretch across the second axis and subtraction would produce a `[32, 32]` matrix of all pairwise
residuals. On nonzero data, that loss could decrease while comparing every prediction with every
label.

TorchLean requires us to reshape the labels, squeeze the predictions, or select another loss.
That choice should follow the intended pairing of samples and targets.

# Limits Of Shape Checking

Shape typing checks dimensions, leaving properties of the entries to other checks. A square
matrix and its transpose, for example, have exactly the same tensor type.

The einops-style notation from the tensor chapter expresses the axis exchange:

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

#eval Spec.matVecMulSpec ovW ovIn
#eval Spec.matVecMulSpec ovWT ovIn
```
```leanOutput ovTranspose
[0.500000, 2.500000]
```
```leanOutput ovTranspose
[0.500000, 1.500000]
```

The input `[1, 0]` selects the first column in a matrix-vector product. Before transposition,
that column contains 0.5 and 2.5; afterwards it contains 0.5 and 1.5. A square matrix makes the
shape error invisible, while a nonsymmetric payload makes the semantic error visible. This is a
useful way to design an importer check: choose entries that distinguish row and column conventions,
then choose an input that isolates one of them. A matrix of all zeros or a symmetric matrix would
let the wrong convention pass this particular check.

A loader that reads this row-major buffer as column-major would introduce exactly that transpose
while preserving the tensor type `Tensor Float [2, 2]`.

It helps to remember what the contraction is doing. `Spec.matVecMulSpec` sums over the second axis,
so entry $`i` of the result is the dot product of row $`i` with the input:

```lean (name := ovDot)
-- Use basis and all-ones inputs to make contraction
-- directions easy to identify.
#eval Tensor.dotSpec ovIn ovIn
#eval Spec.matVecMulSpec ovW (Tensor.full [2] 1.0)
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
reference run. The {ref "pytorch-roundtrip"}[round trip chapter] does the second of those against
real PyTorch state dicts, and the failure it is guarding against is exactly this one.

Units, label quality, normalization, finiteness, and generalization need their own definitions and
hypotheses. The tensor shape carries none of those guarantees.

The scalar parameter is equally explicit. `Tensor α s` is homogeneous: every element of that
tensor has type `α`. The tensor chapter develops that point once, alongside the distinction between
specification scalars and executable runtime arithmetic.

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

The raw measurement sits half a standard deviation above the mean in both features. Now ask the
model about it, once through the normalizer and once directly, which is the mistake:

```lean (name := ovNormBug)
-- Compare the normalized measurement with the same numbers
-- sent in raw units.
#eval trainer.predict (ovNormalize ovRaw)
#eval trainer.predict ovRaw
```
```leanOutput ovNormBug
[0.024256]
```
```leanOutput ovNormBug
[-47.586140]
```

The same raw measurement leads to predictions three orders of magnitude apart depending on whether
normalization is applied. Both calls accept `Tensor Float [2]` and return `IO (Tensor Float [1])`.
The shape cannot distinguish raw coordinates from normalized coordinates, so the caller needs a
separate contract specifying which representation the model expects.

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

Around the normalized center `[0.5, 0.5]`, these half-widths describe the rectangle
`[0.45, 0.55]` by `[0.498, 0.502]`. The second feature has a much narrower normalized range
because its raw scale is larger. Replacing both half-widths by 0.05 would enlarge the region in
that feature, potentially making a bound much looser; replacing both by 0.002 would fail to cover
the intended first-feature perturbations.

A ball of radius $`0.1` in raw units becomes a box of half-widths $`0.05` and $`0.002` in normalized
units. It is still a box, because the normalizer here is diagonal, but it is no longer a *ball*: the
two features are scaled by different amounts, so the certificate produced for
$`\|x-x_0\|_\infty\leq 0.1` in one coordinate system is not a certificate for the same inequality in
the other. Conversely, a normalized radius of $`0.1` corresponds to raw half-widths $`0.2` and $`5`.
The claim must state which coordinate system and which per-feature widths it covers.

There are two ways to account for this transformation. Either
the normalizer is part of the lowered forward program, so `denote` includes it and the input box is
quantified over raw measurements, or it is not, and then the normalization becomes an explicit
hypothesis about the box being verified. Because the graph and the box are separate arguments to the
checker, they can be inspected separately. The
{ref "verification"}[verification chapters] state the property against a named graph and a named
region for exactly this reason, and {ref "datasets-loaders"}[the datasets chapter] is where the
statistics that define a normalizer come from.

# Fused Multiply-Add

Replacing `x * y + z` with a single fused multiply-add can change the answer because the fused
operation rounds only once. The
float chapters use FloatLib's executable semantics for exactly this kind of question, and it is
available with one more import:

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

Now compute $`a \cdot a + b` with separate operations and with a fused operation:

```lean (name := ovFma)
-- Compare two rounding schedules by equality, encoded
-- words, and their scaled difference.
#eval (ovA * ovA + ovB) ==
  ExecFloat.Binary.fmaWithRounding ovA ovA ovB .nearestEven
#eval (ExecFloat.Binary.toBits32 (ovA * ovA + ovB),
  ExecFloat.Binary.toBits32
    (ExecFloat.Binary.fmaWithRounding
      ovA ovA ovB .nearestEven))
#eval
  ((ExecFloat.Binary.toFloat32
      (ExecFloat.Binary.fmaWithRounding
        ovA ovA ovB .nearestEven)).toFloat
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

Cancellation explains the large ULP count. The product is close to one, but subtracting one leaves
a much smaller result with much finer spacing between representable numbers. A small rounding
error in the product can therefore span many ULPs of the result.

The exact product is $`a \cdot a = 1 + 2^{-11} + 2^{-24}`, and that value needs
twenty-five significand bits, one more than binary32 has. So the separate multiply rounds it to
$`1 + 2^{-11}`, after which adding $`-1` is exact and the answer is $`2^{-11}`. The fused form keeps
the full product, rounds once at the end, and returns $`2^{-11} + 2^{-24}`. The two results differ
by $`2^{-24} \approx 5.96 \times 10^{-8}`, which is $`1024` ULPs at this result's magnitude
($`\operatorname{ulp}(2^{-11})=2^{-34}`). That is the third
`#eval` above, reported as a multiple of $`10^{-9}` because `Float.toString` prints six decimals and
both results print as `0.000488`.

A regression test with absolute tolerance $`10^{-6}` cannot see this difference;
larger computations need an error analysis that accounts for their operation schedule. Also,
compiler contraction settings can permit this substitution. The CompCert work on verified
compilation of floating-point computations
({Informal.citet boldo2015}[]) treats it as a case the semantics has to model rather than an
optimizer bug. A numerical guarantee must therefore account for whether the execution permits
fusion.

FloatLib gives `*`, `+`, and `fma` separate specifications, so a theorem has to commit to a
rounding schedule. The
{ref "floats"}[floating-point chapter] develops that semantics, compares the design against the Rocq
Flocq library ({Informal.citep flocq2011}[]), and states which backend contracts assume that no
fusion happened. For the underlying arithmetic, Goldberg's survey
({Informal.citep goldberg1991}[]) explains rounding and the effect of intermediate precision.

# Output Properties

Suppose the trained model has parameters $`\theta`, its graph is $`g`, and the two input features
vary in a box $`B`. Let $`y^\star` be the desired output and $`\delta` the allowed error. Writing
`denote` for evaluation of the graph with those parameters, a range statement might be

$$`\forall x\in B,\qquad
  |\operatorname{denote}(g,\theta,x)_0-y^\star|\leq\delta`.

This line records the details that are easy to lose in prose:

- $`g` identifies the operation graph;
- $`\theta` identifies the concrete trained parameters;
- $`B` identifies the quantified input convention;
- `denote` identifies the scalar and operator semantics;
- index $`0`, target $`y^\star`, and tolerance $`\delta` identify the output property.

The `denote` in that formula is not shorthand for "however the graph happens to run". It is a
function in the library, and its type lists the same ingredients:

```lean (name := ovDenote)
-- Expose all arguments that determine a graph evaluation,
-- including its scalar semantics.
#check @NN.IR.Graph.denote
```
```leanOutput ovDenote (whitespace := lax)
@NN.IR.Graph.denote : {α : Type} →
  [inst : Storage α] →
    [inst_1 : Context α] → NN.IR.Graph → NN.IR.Payload α →
      Spec.SomeTensor α → ℕ → Except String (Spec.SomeTensor α)
```

The arguments fix the scalar type and its storage and arithmetic instances, then the graph,
payload, input tensor, and requested node id. For our model that id is `17`; another request can
select another node in the same graph.

`SomeTensor α` pairs the result with its shape, because a graph supplied as data determines
intermediate shapes during evaluation. `Except String` reports malformed graphs as errors.
The `Storage` and `Context` instances provide the operations needed to evaluate; a bound on the
returned values still requires a soundness argument.

An interval pass can compute lower and upper output bounds. A Boolean check can then confirm that
the whole interval lies in $`[y^\star-\delta,y^\star+\delta]`. The last ingredient is a soundness
theorem saying that the computed interval really encloses `denote` for this graph. Interval
propagation and its linear-relaxation refinement are not TorchLean inventions; the shapes of those
bounds come from the robustness-verification literature
({Informal.citep wongkolter2018 crown2018 autolirpa2020}[]), and what TorchLean
adds is a Lean proof that the propagation step really encloses the semantics it claims to.

The guide uses the following vocabulary throughout:

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
({Informal.citet necula1997}[]) applied to tensor programs: an untrusted producer
emits an artifact, a small checker accepts or rejects it, and a theorem relates acceptance to the
property we actually care about.

# Native Kernels And Backend Contracts

The numerical work can still use established native kernels. PyTorch provides operator coverage,
distributed training, compilers, and pretrained models that a project may already depend on.

TorchLean calls LibTorch's ATen operators for CUDA computation while retaining its own model,
parameter layout, tape, and backward traversal.
{ref "backend-selection"}[Backend Selection]
explains how each operation acquires an implementation and a contract stating the assumptions
that connect it to graph semantics.

# Imports

Use the focused TorchLean API for application code:

```
-- Use the application import when model construction and
-- training are the task.
import NN.API
open TorchLean
```

Use the complete library import when one file combines application code with proofs, verification,
floating-point semantics, backends, or widgets:

```
-- The full import also makes proof, verification, and
-- backend interfaces available.
import NN
open TorchLean
```

Focused subsystem imports such as `NN.IR`, `NN.Floats`, `NN.Proofs`, and `NN.Verification` avoid
loading the whole project when only one layer is needed.

# References

The frontend notation follows PyTorch ({Informal.citet pytorch2019}[]), and the analysis the
specification layer is stated against is Mathlib ({Informal.citet mathlib2020}[]). The fused
multiply-add example is the situation modelled by {Informal.citet boldo2015}[]; the arithmetic
behind it is explained in {Informal.citet goldberg1991}[], and the Rocq library the floating-point
chapters compare against is {Informal.citet flocq2011}[]. Bound propagation and its refinements come
from {Informal.citet wongkolter2018}[], {Informal.citet crown2018}[],
{Informal.citet autolirpa2020}[], and {Informal.citet betacrown2021}[]. The evidence vocabulary is
the proof-carrying-code discipline of {Informal.citet necula1997}[].
