import VersoManual
import NN.API
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.Roles

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean
open TorchLean
open Lean.Elab.Tactic.GuardMsgs.WhitespaceMode (lax)

#doc (Manual) "Models" =>
%%%
tag := "building-models"
file := "Building-A-Model"
%%%

A TorchLean model begins as a map between tensor shapes. Initialization, loss functions, optimizer
state, and execution devices are attached later. This separation is useful even before proving a
theorem: the architecture can be inspected without allocating parameters, and the same model can
be initialized twice with different seeds or interpreted by different runtimes.

I'll start with a $`2\to8\to1` regression MLP and follow the tensor through each layer. Its hidden
width determines two weight matrices, so changing that one number changes both the intermediate
shape and the parameter count. The same reasoning applies to the convolutional classifier and
transformer encoder block {Informal.citep transformer2017}[] below.

The Lean summaries expose each parameter shape alongside its count. Comparing those shapes with
PyTorch {Informal.citep pytorch2019}[] reveals a detail a total can hide: the transformer examples
use different attention-bias conventions. Later, masking changes the model's input while a
low-rank adapter changes a projection's weights.

# Layer Types And Shape Checking

The layer type is:

```
-- A layer names the shape it consumes and the shape it
-- produces.
nn.Layer input output
```

A sequential model has:

```
-- A sequential model retains its outer shapes while
-- checking each internal connection.
nn.Sequential input output
```

The input and output shapes are indices of the type, so composition requires equality at the
boundary. If:

$$`f:s_0\to s_1`

and

$$`g:s_1\to s_2`,

then $`g` may follow $`f`. A layer expecting $`s_3` cannot be inserted there merely because the two
shapes contain the same number of values.

For the MLP, the first map produces eight hidden features, so the second map must consume those
eight features as a vector. Reshaping them into a `[2, 4]` matrix would preserve the scalar count
but change that interface. An explicit reshape could be part of a different model; sequential
composition will not invent one. The type of the intermediate value is where one layer's
promise becomes the next layer's requirement.

Model builders use:

```
-- A builder delays the seed-dependent construction of its
-- result A.
nn.Builder A
```

which is a seeded construction of `A`. It allocates deterministic seeds for parameterized layers
but does not run a forward pass or optimizer update.

# Layer Definition Fields

An `nn.Layer σ τ` stores the pieces that must remain aligned from model construction through
execution:

:::table +header
*
  * Field
  * Meaning
*
  * `kind`
  * the name printed in model summaries
*
  * `stateShapes`
  * the ordered shapes of parameters and persistent buffers
*
  * `initState`
  * reproducible Float-authored initial values for that shape list
*
  * `runtimeInit`
  * an optional storage-first initializer for executable Float runtimes
*
  * `requiresGrad`
  * one gradient flag for each entry of `stateShapes`
*
  * `validateConfig`
  * value-level checks, such as a valid dropout probability, before runtime allocation or lowering
*
  * `updateBuffers`
  * an optional train/eval-dependent update for state such as running statistics
*
  * `forward`
  * the execution-polymorphic tensor program from `σ` to `τ`
:::

The forward program receives model state followed by the layer input. Its input shape list is
`stateShapes ++ [σ]`, and its result has shape `τ`. The initializer and forward program share the
same dependent shape list. This catches swaps between different shapes; two entries with equal
shapes can still be exchanged incorrectly.

`validateConfig` handles facts that equality of shapes does not express. A dropout probability
can be outside its allowed interval even though the layer preserves its input shape perfectly.
Likewise, the state layout says where a running statistic lives, while `updateBuffers` says how
training changes it. Reading the fields together separates three obligations: dimensions must
compose, configuration values must be accepted, and the forward program must use state in the
intended order.

A linear layer from two inputs to one output makes the relation between state shapes and gradient
flags concrete. Its state contains a weight matrix and a bias vector:

```lean (name := bmTiny)
-- Building the affine map fixes the weight and bias values
-- before we read their shape list.
def bmTiny (seed : Nat) : nn.Sequential [2] [1] :=
  nn.build seed nn.Sequential![nn.linear 2 1]

#eval (bmTiny 2026).stateShapes
```
```leanOutput bmTiny (whitespace := lax)
[[1, 2], [1]]
```

Weight first, then bias, in that order, with the weight stored as $`[\mathrm{out},\mathrm{in}]`.
Both entries are trainable. For this layer, the weight has two entries and the bias has one,
giving three scalar parameters. The two entries in `requiresGrad` below refer to those two
tensors, rather than to the three individual scalars. This convention scales to a large matrix
without storing a separate flag for every coordinate. Its position in the flag array follows
the same weight-then-bias order as the state list:

```lean (name := bmGrad)
-- The two flags belong to whole state tensors: the weight
-- and the bias are both trainable.
#eval (bmTiny 2026).requiresGrad
```
```leanOutput bmGrad (whitespace := lax)
#[true, true]
```

A frozen embedding table or a running batch-norm statistic is the case where one of those flags is
`false` while the shape stays in the list. That is why `requiresGrad` is a separate flag list rather
than a shorter shape list: the value is still state the module owns and a checkpoint must carry, it
just does not receive a gradient.

Sequential composition is correspondingly small. The internal type is `Seq`, exported to
application code as `nn.Sequential`:

```
-- The shared τ makes the first layer output exactly the
-- next sequence input.
inductive Seq : Shape → Shape → Type
  | id (s : Shape) : Seq s s
  | cons : nn.Layer σ τ → Seq τ υ → Seq σ υ
```

The middle shape `τ` appears on both sides of `cons`. This is the check performed by
`nn.Sequential!`; the macro saves syntax but does not weaken the type.

# Model Definitions And Runtime State

`nn.Sequential` is an immutable model definition. It contains the checked layer composition,
parameter layout, initializers, and forward program; it does not hide mutable weights inside the
value used by proofs and graph lowering.

Manual runtime code can instantiate that definition as a live module:

```
-- The same parameters can run in training mode and then
-- evaluation mode through the module.
nn.withModel model fun checked => do
  let module ← nn.Module.instantiate checked { device := .cpu }
  let trainingOutput ← module.forward input
  module.eval
  let evaluationOutput ← module.forward input
```

The module owns parameters, persistent buffers, and its train/eval mode. It begins in training
mode. `forward` executes without a backward tape and respects the current mode. Passing
`(mode := some .eval)` selects evaluation for one call without changing the stored mode.
Instantiation uses native `Float32` by default; code that needs
another executable or proof-facing representation selects it directly with `(α := ...)`. The
immutable `checked` value is still the object passed to `nn.lowerToTypedGraph` or a proof-facing
interpretation.

This distinction is intentional. Mutating an `nn.Sequential` or `nn.Layer` would make the meaning
of a lowered graph depend on hidden state. Mutation belongs to the instantiated `nn.Module`; the
definition passed to lowering and proofs remains an ordinary Lean value.

The snippet first constructs that definition's live state and then runs the same input in two
modes. `module.eval` changes how subsequent mode-sensitive operations behave; it does not create
a new architecture or replace learned weights. For a plain linear/ReLU MLP the modes have the
same forward formula. For dropout or normalization, the mode changes which formula or persistent
statistics are used, so it belongs to the execution state even though the layer shapes stay fixed.

# Token Models And Bounded Indices

Token models use the same split without turning indices into floating-point tensors. The untrusted
boundary is explicit: raw identifiers arrive as a `Tensor Nat`, and one checked step turns them into
bounded indices.

```lean (name := bmIndices)
-- Check the vocabulary bound without changing the [1, 3]
-- token layout.
/-- Three token identifiers for a five-word vocabulary. -/
def bmRawIds : Tensor Nat [1, 3] := [[0, 2, 4]]

def bmBadIds : Tensor Nat [1, 3] := [[0, 2, 9]]

#eval (Tensor.checkIndices 5 bmRawIds).isOk
```
```leanOutput bmIndices (whitespace := lax)
true
```

An identifier outside the vocabulary is rejected, with a message rather than a silent clamp or a
wrapped index:

```lean (name := bmBadIndices)
-- The value nine cannot inhabit Fin 5, so conversion
-- returns the diagnostic branch.
#eval
  match Tensor.checkIndices 5 bmBadIds with
  | .error message => message
  | .ok _ => "accepted"
```
```leanOutput bmBadIndices (whitespace := lax)
"tensor contains an index outside the valid range [0, 5)"
```

After the check, the identifiers have type `Tensor (Fin 5) [1, 3]`, and the embedding lookup can no
longer be handed an out-of-range row. The table itself is built like any other model, and its type
records the whole lookup:

```lean (name := bmEmbedding)
-- Embedding appends width three to the token shape; each
-- token selects one learned row.
def bmTable := nn.build 2026 (nn.embedding 5 3)

#check bmTable.model [1, 3]
```
```leanOutput bmEmbedding (whitespace := lax)
bmTable.model [1, 3] :
  nn.IndexedModel [1, 3] (Shape.appendDim [1, 3] 3)
    (Fin 5)
```

Reading that type left to right: token identifiers of shape `[1, 3]` map to vectors of shape
`[1, 3, 3]`, and the index type is `Fin 5`. The vocabulary bound is part of the model type, not a
comment. In runtime code the pieces fit together as:

```
-- Validate token bounds before running the indexed module
-- against the shared embedding table.
let table := nn.build 2026 <| nn.embedding vocab embedDim
let tokenShape := [batch, seqLen]
let tokenIds ← Tensor.checkIndices vocab rawTokenIds
let module ← nn.IndexedModule.instantiate (table.model tokenShape) { device := .cpu }
let vectors ← module.forward tokenIds (mode := some .eval)
```

Repeated identifiers select the same row and therefore accumulate into the same weight gradient.
Fresh tables use a seeded normal sampler with the standard-normal initialization convention of
`torch.nn.Embedding`; matching distributions does not imply matching generated values. Architectures
may replace that default, and GPT-2 uses its smaller initialization scale.

To start from an existing table, use the tensor itself:

```
-- Freezing changes whether this supplied weight receives
-- updates, not its embedding lookup.
let table := nn.Embedding.fromWeight weight
let frozenTable := nn.Embedding.fromWeight weight (freeze := true)
```

The type of `weight` fixes the vocabulary and embedding dimensions. The first table participates in
reverse mode; the second remains part of module state but receives no parameter gradient, which is
the `requiresGrad := false` case from the field list above. Current embedding modules use dense row
lookup and scatter-add gradients. Options that change PyTorch's forward or backward semantics, such
as `padding_idx`, `max_norm`, frequency-scaled gradients, and sparse gradients, require their own
checked definitions and are not accepted as ignored flags.

For `bmRawIds`, lookup selects table rows zero, two, and four. Each selected row has three
features, which explains the extra final dimension in `[1, 3, 3]`: the original batch and token
positions survive, and each identifier becomes a vector. The failed validation of `bmBadIds`
occurs before this lookup because row nine does not exist. The successful `true` above reports
only that the bounds check accepted the identifiers; it does not evaluate the embedding or say
anything about the quality of its learned vectors.

# MLP Architecture

The regression MLP uses two linear layers separated by a ReLU. Making its hidden width an argument
lets us compare parameter layouts while keeping the external input and output shapes fixed:

```lean (name := bmMlp)
-- The hidden width controls both adjoining weight shapes
-- while the public [2] → [1] map stays
-- fixed.
/-- The running example, given a hidden width. -/
def bmMlp (hidden : Nat) :
    nn.Builder (nn.Sequential [2] [1]) :=
  nn.Sequential![
    nn.linear 2 hidden,
    nn.relu,
    nn.linear hidden 1
  ]

#eval nn.printSummary (nn.build 2026 (bmMlp 8))
```
```leanOutput bmMlp (whitespace := lax)
Sequential: [2] -> [1], layers=3, params=33, state=33
  [0] Linear(2, 8): [2] -> [8] params=24, state=24 [[8, 2], [8]]
  [1] ReLU: [8] -> [8] params=0, state=0 []
  [2] Linear(8, 1): [8] -> [1] params=9, state=9 [[1, 8], [1]]
```

That summary is the counterpart of `print(model)` together with `torchinfo.summary`, and it is
printed from the immutable definition rather than from an instantiated module. Read the macro from
top to bottom:

```
-- ReLU preserves the hidden shape; only the linear maps
-- change widths.
input [2]
  -> linear 2 8
hidden [8]
  -> relu
hidden [8]
  -> linear 8 1
output [1]
```

`ReLU` is shape-preserving, which the summary shows as `[8] -> [8]` with `params=0`. The two linear
layers each change the final feature axis. The macro checks that the chain is composable and returns
one `Sequential [2] [1]`.

The parameter layout follows PyTorch's linear-layer convention:

$$`
W:\operatorname{Tensor}\;\alpha\;[\mathrm{out},\mathrm{in}],
\qquad
b:\operatorname{Tensor}\;\alpha\;[\mathrm{out}].
`

So the shapes in the summary read $`[8,2]`, $`[8]`, $`[1,8]`, $`[1]`, and the total is:

$$`8\cdot2+8+1\cdot8+1=33`.

`torch.nn.Sequential(nn.Linear(2,8), nn.ReLU(), nn.Linear(8,1))` reports the same total:

```
mlp params      : 33
```

The counts agree because both models store a weight and a bias for each linear layer, with the same
dimensions. Comparing the individual shapes also checks the weight orientation. A matching total
alone would not establish that correspondence; the transformer example below shows how a bias
convention changes it.

Change the width and the layout changes with it:

```lean (name := bmMlp16)
-- Doubling hidden width changes both affine layers and
-- therefore the total parameter count.
#eval nn.printSummary (nn.build 2026 (bmMlp 16))
```
```leanOutput bmMlp16 (whitespace := lax)
Sequential: [2] -> [1], layers=3, params=65, state=65
  [0] Linear(2, 16): [2] -> [16] params=48, state=48 [[16, 2], [16]]
  [1] ReLU: [16] -> [16] params=0, state=0 []
  [2] Linear(16, 1): [16] -> [1] params=17, state=17 [[1, 16], [1]]
```

$`16\cdot2+16+16+1=65`, and PyTorch's `Linear(2,16)`/`Linear(16,1)` stack also reports `65`.
For hidden width $`h`, the first layer contributes $`2h+h` parameters and the second contributes
$`h+1`, giving $`4h+1` in total. Widths `2`, `16`, and `64` therefore give `9`, `65`, and `257`
parameters. Inserting another hidden `linear` and `relu` pair also preserves the external map
$`[2] \to [1]`, while changing the internal shape chain and parameter payload.

Parameter order is part of the typed model interface. A pack with the two bias tensors exchanged
does not match the expected dependent list. The dependent pack records a list of shapes and follows
the homogeneous-element rule developed in *Tensors And Shapes*. Executing the model at another `α`
reuses the architecture without changing the ordered parameter layout.

# Composition Errors And Architecture Changes

Two kinds of mistake are possible in a model definition, and TorchLean reports them in two different
places. The first kind breaks composition, so it is a type error. Insert a layer whose input width
does not match the previous output:

```lean +error (name := bmMiddle)
-- The inserted layer expects nine hidden entries, but its
-- predecessor supplies eight.
def bmMiddle :
    nn.Builder (nn.Sequential [2] [1]) :=
  nn.Sequential![
    nn.linear 2 8,
    nn.linear 9 4,
    nn.relu,
    nn.linear 4 1
  ]
```
```leanOutput bmMiddle (whitespace := lax)
nn.Sequential!: layer 2 expects input shape [9], but layer 1 outputs [8].
Change layer 2's input shape or insert a layer that converts [8] to [9].
```

The message names the incompatible entries: layer 2 expects width `9`, while layer 1 produces `8`.
It appears at the definition. No data, parameters, or device are needed to expose it. The
{ref "torchlean-api"}[API chapter] shows the same failure at the last layer instead of the middle.

A shape-compatible insertion has a different consequence. This additional `8 → 8` layer is
accepted:

```lean (name := bmWider)
-- The new eight-to-eight map adds its own weight and bias
-- while preserving composability.
def bmWider :
    nn.Builder (nn.Sequential [2] [1]) :=
  nn.Sequential![
    nn.linear 2 8,
    nn.linear 8 8,
    nn.relu,
    nn.linear 8 1
  ]

#eval nn.printSummary (nn.build 2026 bmWider)
```
```leanOutput bmWider (whitespace := lax)
Sequential: [2] -> [1], layers=4, params=105, state=105
  [0] Linear(2, 8): [2] -> [8] params=24, state=24 [[8, 2], [8]]
  [1] Linear(8, 8): [8] -> [8] params=72, state=72 [[8, 8], [8]]
  [2] ReLU: [8] -> [8] params=0, state=0 []
  [3] Linear(8, 1): [8] -> [1] params=9, state=9 [[1, 8], [1]]
```

This version compiles because the shapes compose, and the summary shows what it really is: a
different architecture with another weight matrix and bias, 105 parameters instead of 33, and two
consecutive linear layers with no activation between them. Composing two affine maps gives an affine
map, so those two layers have the expressive power of one. Shape safety prevents malformed
composition; it does not declare two well-shaped networks equivalent, and it does not tell you that
an architecture suits the learning problem.

The additional layer contributes `8 * 8 + 8 = 72` scalars, exactly the difference between the
two summaries. In exact arithmetic, combining the first two affine maps would multiply their
weights and transform their biases. Training those factors separately still gives a different
parameterization and optimizer state, and rounded evaluation can depend on the chosen sequence
of operations. A shape-compatible edit is therefore a reason to reread the summary and the
forward formula, even though composition continues to typecheck.

The second kind of mistake is arithmetic on shapes rather than composition of them, and it is
reported by validation rather than by the type checker. We meet it in the convolutional section
below.

# Parameter Initialization

The model declaration describes how parameters should be created. Building it supplies the seed, and
the initial values are ordinary data we can print:

```lean (name := bmInit)
-- The same seed fixes both weight entries; the separately
-- stored bias starts at zero.
#eval (bmTiny 2026).initState
```
```leanOutput bmInit (whitespace := lax)
[[1, 2]: [[-0.616737, 0.962068]], [1]: [0.000000]]
```

Each entry is labeled with its shape. The weight was drawn from the layer's initialization scheme at
seed `2026`; TorchLean's bias starts at exactly zero. PyTorch's default `Linear` bias instead uses
a uniform random initialization. Matching parameter counts does not match initializers. Change only
the seed:

```lean (name := bmInit1)
-- Changing the seed changes initialized weights without
-- changing the parameter shapes.
#eval (bmTiny 1).initState
```
```leanOutput bmInit1 (whitespace := lax)
[[1, 2]: [[-1.225729, -0.830849]], [1]: [0.000000]]
```

Different weights, same zero bias, and both runs are repeatable: evaluating `bmTiny 2026` again
returns the first pair, because the seed is an argument rather than a global generator that other
code can advance. Changing only the seed is therefore a controlled experiment, and a seed printed in
a log reconstructs the starting point only together with the architecture, initializer settings,
and implementation version.

The trainer takes the same seed through its configuration:

```
-- Keep seed choice at construction so repeated runs can
-- reuse one architecture declaration.
def trainer (seed : Nat) :=
  Trainer.new model
    { objective := .mse
      optimizer := optim.adam { learningRate := 0.03 }
      seed := seed }
```

The seed now sits beside the objective and optimizer in the trainer configuration. We can
initialize the same architecture at another seed without rewriting the layer list.

# Prefix Shapes And Shared Parameters

Linear layers act on the final dimension and preserve everything in front of it. The batch shape is
an argument to the layer, so lifting the model to a fixed batch means passing that prefix rather
than reshaping data:

```lean (name := bmBatched)
-- Both linear layers share their parameters across all rows
-- of the leading batch axis.
def bmBatched (batch : Nat) :
    nn.Builder
      (nn.Sequential [batch, 2] [batch, 1]) :=
  nn.Sequential![
    nn.linear 2 8 [batch],
    nn.relu,
    nn.linear 8 1 [batch]
  ]

#eval nn.printSummary (nn.build 2026 (bmBatched 4))
```
```leanOutput bmBatched (whitespace := lax)
Sequential: [4, 2] -> [4, 1], layers=3, params=33, state=33
  [0] Linear(2, 8): [4, 2] -> [4, 8] params=24, state=24 [[8, 2], [8]]
  [1] ReLU: [4, 8] -> [4, 8] params=0, state=0 []
  [2] Linear(8, 1): [4, 8] -> [4, 1] params=9, state=9 [[1, 8], [1]]
```

The per-layer shapes now carry the prefix, while the parameter count remains `33`. The same four
parameter tensors apply to every sample in the batch. The prefix changes the activation shapes
without introducing a separate set of weights for each sample.

One row follows `[2] → [8] → [1]` inside that batched computation. The next row follows the
same maps with the same weights, so adding rows increases activation storage and arithmetic work
without adding trainable scalars. During training, contributions from the rows meet in the
gradients of those shared weights. This is why a batch prefix belongs to the input/output
contract, whereas the parameter layout remains the one printed for the unbatched MLP.

The batch prefix must be passed to each parameterized layer. It is an ordinary
argument with a default of `[]`, and the default is inserted before the composition is checked. The
error you get is the composition mismatch from the previous section.

Nothing in `nn.linear` calls the prefix “batch.” It could equally be `[time]`, `[batch, time]`, or a
higher-rank collection. The operation's contract is:

$$`
[\ldots,\mathrm{inFeatures}]
\longrightarrow
[\ldots,\mathrm{outFeatures}].
`

This is the same broad behavior users expect from PyTorch, with the full map recorded in the Lean
type. `Data.batch` later collates per-sample tensors into a model whose prefix begins with
the chosen batch size.

# A Rank-Generic Convolutional Model

TorchLean does not need separate tensor types for signals, images, and volumes. A convolution is
parameterized by a vector of spatial sizes. The length of that vector determines the spatial rank.

For a two-dimensional input, the conventional shape is:

$$`
[\mathrm{batch},\mathrm{channels},\mathrm{height},\mathrm{width}].
`

For one-dimensional signals it is:

$$`
[\mathrm{batch},\mathrm{channels},\mathrm{length}].
`

The public layer surface uses `nn.conv`, `nn.convTranspose`, `nn.maxPool`, `nn.avgPool`,
`nn.globalAvgPool`, and the rank-polymorphic channel-normalization constructors. The length of each
spatial tensor determines the rank. `nn.models.CNN.Config` records the input channels, spatial
sizes, an ordered list of feature stages, and the class count. Each stage chooses its own
convolution, activation, optional dropout, and pooling. Start with one stage so that we can
account for every parameter:

```lean (name := bmCnn)
-- The convolution and pooling geometries determine the
-- classifier input width.
/-- Four filters followed by two-by-two pooling. -/
def bmCnnStage : nn.ConvPoolBlock.Config 2 :=
  { block :=
      { convolution :=
          { outChannels := 4, kernelSize := [3, 3] } }
    pooling :=
      { kernelSize := [2, 2], stride := [2, 2] } }

/-- Three channels, 8 by 8, one stage, ten classes. -/
def bmCnnConfig : nn.models.CNN.Config 2 :=
  { inputChannels := 3
    spatial := [8, 8]
    stages := [bmCnnStage]
    classCount := 10 }

#eval nn.printSummary
  (nn.build 2026 (nn.models.cnn bmCnnConfig [1]))
```
```leanOutput bmCnn (whitespace := lax)
Sequential: [1, 3, 8, 8] -> [1, 10], layers=5, params=482, state=482
  [0] Conv(rank=2, in=3, out=4): [1, 3, 8, 8] -> [1, 4, 6, 6]
    params=112, state=112 [[4, 3, 3, 3], [4]]
  [1] ReLU: [1, 4, 6, 6] -> [1, 4, 6, 6] params=0, state=0 []
  [2] MaxPool(rank=2): [1, 4, 6, 6] -> [1, 4, 3, 3]
    params=0, state=0 []
  [3] FlattenAfter: [1, 4, 3, 3] -> [1, 36] params=0, state=0 []
  [4] Linear(36, 10): [1, 36] -> [1, 10]
    params=370, state=370 [[10, 36], [10]]
```

Every number in that summary is derived from the configuration. The kernel is
$`4\times3\times3\times3` because the layer has four filters over three input channels and a
$`3\times3` window, plus one bias per filter, so $`108+4=112`. The spatial geometry shrinks twice,
and both steps are computable on their own:

```lean (name := bmGeom)
-- Compute the spatial dimensions after convolution before
-- flattening any activations.
#eval bmCnnStage.block.convolution.outputSpatial
  bmCnnConfig.spatial
```
```leanOutput bmGeom (whitespace := lax)
[6, 6]
```

```lean (name := bmGeom2)
-- Pooling consumes the convolution output dimensions, not
-- the original image dimensions.
#eval bmCnnStage.pooling.outputSpatial
  (bmCnnStage.block.convolution.outputSpatial
    bmCnnConfig.spatial)
```
```leanOutput bmGeom2 (whitespace := lax)
[3, 3]
```

An unpadded $`3\times3` convolution with unit stride takes $`8` to $`8-3+1=6`, and $`2\times2`
pooling with stride two takes $`6` to $`3`. The classifier head therefore sees
$`4\cdot3\cdot3=36` features and holds $`10\cdot36+10=370` parameters, for a total of $`482`. The
same network in PyTorch,
`Sequential(Conv2d(3,4,3), ReLU(), MaxPool2d(2,2), Flatten(), Linear(36,10))`, reports:

```
cnn params      : 482
cnn out shape   : [1, 10]
```

Identical count and identical output shape, which is the check worth running whenever a layer's
convention is in doubt.

With several stages, each consumes the channels and spatial grid produced by the previous one.
The stages can have different widths, kernels, strides, and pooling choices; the linear head
always consumes the final feature map. An empty `stages` list flattens the input directly and
attaches that head; input channels, spatial extents, and class count must still be positive.
Parameters follow stage order, then the head's weight and bias.

## Validating Spatial Geometry

A convolution can have composable layer types and still request an empty spatial grid. Its
configuration validator checks the computed extents and returns an `Except` result:

```lean (name := bmValidOk)
-- Validation checks numeric configuration constraints that
-- shape annotations alone do not
-- express.
#eval bmCnnConfig.validate
```
```leanOutput bmValidOk (whitespace := lax)
Except.ok ()
```

```lean (name := bmValidBad)
-- A nine-by-nine kernel cannot fit this image under the
-- configured convolution geometry.
#eval { bmCnnConfig with
        stages :=
          [{ bmCnnStage with
             block :=
               { convolution :=
                   { outChannels := 4
                     kernelSize := [9, 9] } } }]
      }.validate
```
```leanOutput bmValidBad (whitespace := lax)
Except.error "CNN: geometry produced an empty spatial grid"
```

A $`9\times9` kernel on an $`8\times8` image leaves no output pixels, and the message says so before
any parameter is allocated. A degenerate class count is caught the same way:

```lean (name := bmValidClasses)
-- A zero class count is rejected even though zero is a
-- valid Nat.
#eval { bmCnnConfig with classCount := 0 }.validate
```
```leanOutput bmValidClasses (whitespace := lax)
Except.error "CNN: class count must be positive"
```

These checks concern inequalities between computed extents. Requiring a proof at every call would
also require callers to construct that proof for dimensions read from a command line. The
configuration API instead reports invalid geometry through `Except` at the boundary where those
dimensions arrive. The resulting layer types still record the computed shapes and check their
composition.

The successful validation result, `Except.ok ()`, carries no computed tensor. Its unit value
means that the configuration checks passed. The two error results identify different missing
conditions: usable spatial geometry and a positive class count. Once accepted, the same
configuration determines the activation shapes printed in the summary. Validation and summary
thus answer complementary questions: whether the requested model is admissible, and what model
that request actually describes.

## CIFAR-10 Training Example

The maintained example trains a convolutional classifier on prepared CIFAR-10 images; its
configuration and image dimensions differ from the small summary above:

```terminal
# Use one downloaded CIFAR-10 example to exercise the
# complete image training route.
python3 scripts/datasets/download_example_data.py --cifar10
lake -R -K cuda=false exe torchlean cnn \
  --device cpu --n-total 1 --steps 1 --seed 2026
```

Uniform ten-class probabilities give loss $`\ln 10 \approx 2.303`; an arbitrary untrained
classifier need not be uniform. The leading dimension of the input is an ordinary tensor
prefix preserved by the layers, not a separate image or batch container.

This one-image run exercises data loading, loss evaluation, and one parameter update. The
`TrainLog` records that execution; assessing classification accuracy would require predictions
on an evaluation set.

The relevant source files are:

- [`Cnn.lean`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/Vision/Cnn.lean);
- [`Cnn.lean`](https://github.com/lean-dojo/TorchLean/blob/main/NN/API/Models/Cnn.lean);
- [`ResNet.lean`](https://github.com/lean-dojo/TorchLean/blob/main/NN/API/Models/ResNet.lean).

Residual blocks require their main and skip paths to return the same shape before addition
{Informal.citep resnet2016}[]. A projection shortcut is therefore an explicit layer, not a runtime
broadcasting guess.

# A Transformer Encoder

A transformer encoder normally consumes:

$$`
[\mathrm{batch},\mathrm{sequenceLength},d_{\mathrm{model}}].
`

One block, small enough to read every parameter shape:

```lean (name := bmBlock)
-- Two heads split the width-eight representation into four
-- coordinates per head.
/-- Two heads of width four over a model width of eight. -/
def bmBlock :
    nn.Builder (nn.Sequential [1, 4, 8] [1, 4, 8]) :=
  nn.transformerEncoderBlock
    { headCount := 2
      headWidth := 4
      feedForwardWidth := 32 }
    (batchShape := [1])
    (sequenceLength := 4) (modelWidth := 8)

#eval nn.printSummary (nn.build 2026 bmBlock)
```
```leanOutput bmBlock (whitespace := lax)
Sequential: [1, 4, 8] -> [1, 4, 8], layers=4, params=840, state=840
  [0] Residual: [1, 4, 8] -> [1, 4, 8]
    params=256, state=256 [[8, 8], [8, 8], [8, 8], [8, 8]]
  [1] LayerNorm: [1, 4, 8] -> [1, 4, 8] params=16, state=16 [[8], [8]]
  [2] Residual: [1, 4, 8] -> [1, 4, 8]
    params=552, state=552 [[32, 8], [32], [8, 32], [8]]
  [3] LayerNorm: [1, 4, 8] -> [1, 4, 8] params=16, state=16 [[8], [8]]
```

The architecture is visible in four rows: attention inside a residual connection, a layer
normalization, the position-wise feed-forward network inside another residual connection, and a
second normalization. The first `Residual` holds four $`8\times8` matrices, which are the query,
key, value, and output projections. The second holds $`[32,8]`, $`[32]`, $`[8,32]`, $`[8]`: the
feed-forward network expands to `32` and comes back, with biases. Both normalizations hold a scale
and a shift of width `8`.

Within attention, the four sequence positions acquire query, key, and value representations.
Two heads split the projected features into groups of four, and attention combines positions
before the output projection restores the model width. The residual addition requires that
width to match the incoming eight features. The feed-forward branch then acts on each position's
features, expanding eight to thirty-two and returning to eight. This explains why a block can
change the representation substantially while keeping its external shape `[1, 4, 8]`.

## PyTorch Transformer Parameter Comparison

`torch.nn.TransformerEncoderLayer(d_model=8, nhead=2, dim_feedforward=32, dropout=0.0)` reports:

```
encoder params  : 872
    self_attn.in_proj_weight [24, 8] 192
    self_attn.in_proj_bias [24] 24
    self_attn.out_proj.weight [8, 8] 64
    self_attn.out_proj.bias [8] 8
    linear1.weight [32, 8] 256
    linear1.bias [32] 32
    linear2.weight [8, 32] 256
    linear2.bias [8] 8
    norm1.weight [8] 8
    norm1.bias [8] 8
    norm2.weight [8] 8
    norm2.bias [8] 8
```

`872` against TorchLean's `840`. The difference is $`872-840=32`, and the listing says exactly where
it lives: `in_proj_bias` contributes `24` and `out_proj.bias` contributes `8`. TorchLean's attention
projections are bias-free, while PyTorch's are not. Everything else matches term by term: PyTorch's
packed $`[24,8]` query/key/value projection is TorchLean's three separate $`[8,8]` matrices
($`192=3\cdot64`), the output projection is $`[8,8]` in both, and the two feed-forward layers and
the two normalizations agree exactly.

The parameter comparison makes the bias choice in this transformer block
{Informal.citep transformer2017}[] explicit. When porting the PyTorch block, its 32 attention bias
values need a decision: the TorchLean block shown here has no corresponding state entries.
The per-tensor summary exposes that mismatch before a forward comparison.

## Block Structure And Masks

The block constructor's configuration validates before any state is allocated. Zero sequence, model,
head, or feed-forward widths are rejected, in the same `Except` style as the convolutional
configuration above. The head configuration may use an internal projection width distinct from
`modelWidth`; `headCount * headWidth` is the projection width, and it equals `modelWidth` in the
example above only because we chose two heads of width four.

Post-normalization is the default, matching the original block. In the two displays below, each
arrow uses $`x` for the representation entering that stage; the second stage receives the result
of the first:

$$`
x \mapsto \mathrm{LayerNorm}(x + \mathrm{MHA}(x))
\mapsto \mathrm{LayerNorm}(x + \mathrm{FFN}(x)).
`

With `normalizeFirst := true`, each branch normalizes before its learned transform, which is the
pre-norm variant used by most later models:

$$`
x \mapsto x + \mathrm{MHA}(\mathrm{LayerNorm}(x))
\mapsto x + \mathrm{FFN}(\mathrm{LayerNorm}(x)).
`

The parameter count is identical for the two variants; only the composition order changes.

Boolean masks are explicit and use hard-mask semantics: blocked positions have zero softmax
numerator. TorchLean does not silently replace that mathematical operation with a finite additive
constant such as `-1000`. The {ref "bugzoo-catalog"}[BugZoo catalog] executes both the mask and its
proof, and shows what PyTorch returns for a fully masked row.

Run one optimizer step of the maintained example:

```terminal
# This command trains the complete sequence example, not
# just the isolated encoder block above.
lake exe torchlean transformer \
  --device cpu --steps 1 --log false
```

The current example reports:

```terminal +output
[TorchLean] arithmetic: native binary32
[TorchLean] execution: eager
[TorchLean] device: cpu
transformer: Causal Transformer next-byte model (device=cpu)
dataset size = 16
mean_loss(before training) = 1.836619
mean_loss(after training) = 1.773165
steps=1 arithmetic=native scalar=Float32 loss=1.836619 -> 1.773165
transformer: ok
```

The log identifies the arithmetic, execution mode, and device chosen for this run. The next runtime
chapters explain how those choices are made.

This transcript comes from the maintained causal next-byte example, whereas the summary above
describes a standalone encoder block. The log identifies a complete learning task, with a
dataset and loss attached to its model. Its sixteen items and before/after losses should be
read in that context; they are not measurements of the isolated four-layer summary. The shared
interface lets both examples use the same construction machinery while their masks and
objectives specify different computations.

# Model Family Constructors

KANs, GPT-style language models, vision transformers {Informal.citep vit2021}[], recurrent and
state-space models {Informal.citep mamba2024}[], neural operators {Informal.citep fno2021}[],
autoencoders, diffusion models, and reinforcement-learning policies all build from the same shape,
parameter, and runtime interfaces. None of them introduces a private tensor type, and each one is
inspectable with the same `nn.printSummary` used above.

For example, `nn.models.KAN.Config` records input/output widths, hidden widths, and an edge basis
family. The basis is explicit because a KAN edge performs a learned scalar function rather than an
ordinary affine weight. It still returns a seeded model builder that can be trained through the
same trainer boundary.

A Fourier neural operator uses spectral transforms and mode truncation, but its input and output
remain general tensors. The Burgers example later in the guide shows how a PDE trajectory dataset,
FNO model, exported prediction, and Lean-checkable residual artifact fit together.

# Masked Inputs And Reconstruction Targets

Self-supervised learning changes the training problem more than it changes the tensor foundation
{Informal.citep mae2022}[]. For a sequence of eight sensor readings, a block mask exposes
alternating pairs and hides the remaining pairs. The reconstruction target must retain the
original readings:

```lean (name := bmMask)
-- The mask policy groups adjacent signal entries into
-- blocks of width two.
def bmSignal : Tensor Float [8] :=
  [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

/-- One policy entry per axis: blocks of width two. -/
def bmBlocks : Tensor (Option Nat) [1] := [some 2]

#eval ssl.BlockMask.apply bmSignal bmBlocks 2 0
```
```leanOutput bmMask (whitespace := lax)
[0.000000, 0.000000, 3.000000, 4.000000,
 0.000000, 0.000000, 7.000000, 8.000000]
```

`some 2` divides the participating axis into blocks of width two. The period `2` and offset `0` hide
block indices congruent to zero modulo two, hence the first and third blocks. Change the offset and
the complement is hidden instead:

```lean (name := bmMask1)
-- Keep the number of hidden blocks fixed and change which
-- blocks the seed selects.
#eval ssl.BlockMask.apply bmSignal bmBlocks 2 1
```
```leanOutput bmMask1 (whitespace := lax)
[1.000000, 2.000000, 0.000000, 0.000000,
 5.000000, 6.000000, 0.000000, 0.000000]
```

Change the period and the mask thins out. Period `4` hides one block in four, so only the first two
readings disappear:

```lean (name := bmMask4)
-- Period four hides the first block, replacing its two
-- signal values with zero.
#eval ssl.BlockMask.apply bmSignal bmBlocks 4 0
```
```leanOutput bmMask4 (whitespace := lax)
[0.000000, 0.000000, 3.000000, 4.000000,
 5.000000, 6.000000, 7.000000, 8.000000]
```

An axis marked `none` is left out of the block index; an image policy such as
`[none, some 4, some 4]` therefore repeats the same 4-by-4 spatial mask across channels.

The loss needs to know which coordinates were hidden, and that is a separate query against the same
policy rather than a second convention to keep in sync:

```lean (name := bmHidden)
-- List the coordinates the reconstruction objective must
-- recover from the original target.
#eval ssl.BlockMAE.hiddenIndices
  (dataShape := [8]) bmBlocks 2 0
```
```leanOutput bmHidden (whitespace := lax)
#[0, 1, 4, 5]
```

Those are exactly the flattened positions that came back zero in the first transcript. For training,
`ssl.BlockMAE.sample` pairs the masked batch with a row-major prefix of the original, unmasked
batch:

```lean (name := bmSample)
-- Build a supervised pair whose input is masked and whose
-- target retains the original signal.
def bmBatchedSignal : Tensor Float [1, 8] :=
  [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]

#eval
  match ssl.BlockMAE.sample [1] (dataShape := [8]) 8
      bmBlocks 2 0 bmBatchedSignal with
  | .error message => s!"rejected: {message}"
  | .ok sample =>
      s!"{reprStr sample.input} / {reprStr sample.target}"
```
```leanOutput bmSample (whitespace := lax)
"[[0.000000, 0.000000, 3.000000, 4.000000, 0.000000,
 0.000000, 7.000000, 8.000000]] /
 [[1.000000, 2.000000, 3.000000, 4.000000, 5.000000,
 6.000000, 7.000000, 8.000000]]"
```

The input is masked and the target is not. Choosing which reconstructed coordinates contribute to
the loss is a separate part of the masked-reconstruction objective. The reconstruction width appears
in the target shape; an oversized prefix is reported as an
ordinary construction error rather than truncated. `hiddenIndices` returns the flattened coordinates
selected by the same mask, so a weighted loss can exclude visible coordinates without duplicating
the mask convention.

For the first mask, reconstruction errors at positions zero, one, four, and five concern hidden
readings. Errors at positions two, three, six, and seven concern readings the model could already
see. Summing both groups would train a different objective from summing only hidden positions.
The unchanged target in the transcript is essential in either case: replacing it with the masked
input would reward reproducing the inserted zeros. Shape agreement cannot catch that mistake
because masked and original tensors have identical dimensions.

The mask is deterministic. In the typed `BlockMask.apply` call, the policy and tensor ranks already
agree by construction, because the policy's length is `shape.rank`. A zero period or zero block
width hides nothing; the lower-level coordinate helpers likewise treat a rank mismatch or an
out-of-bounds coordinate as visible rather than sampling a fallback mask. Randomized mask selection
should choose the period/offset or another explicit mask from recorded generator state. The
coordinate theorems `TorchLean.ssl.BlockMask.hidden_scalar_eq_zero` and
`TorchLean.ssl.BlockMask.visible_scalar_eq_input` then describe exactly what the executable
transformation did: the transcripts above are instances of those two statements.

# Low-Rank Linear Adapters

Masks change which data reaches the model. A low-rank adapter instead changes a parameterized
projection without replacing its base weight {Informal.citep lora2022}[].

`NN.API` also exports the LoRA tensor helpers as `TorchLean.Adapters.LoRA`. An input whose final
axis has size `inputWidth` multiplies a base weight of shape `[inputWidth, outputWidth]` on the
right. Every leading axis is preserved, so the same definition handles a vector, a batch, or several
leading axes. An adapter stores

$$`
A:\operatorname{Tensor}\;\alpha\;[\mathrm{inputWidth},\mathrm{rank}],
\qquad
B:\operatorname{Tensor}\;\alpha\;[\mathrm{rank},\mathrm{outputWidth}],
`

and contributes $`\mathrm{scale}\cdot(AB)` to the base weight. This helper uses
`[inputWidth, outputWidth]` weights, so its orientation must be distinguished from the
`[out, in]` convention of `nn.linear` above. A rank-one adapter makes the update an outer product:

```lean (name := bmLora)
-- The adapter adds a scaled low-rank product to this fixed
-- base projection.
/-- A base projection keeping two of three coordinates. -/
def bmBase : Tensor Float [3, 2] :=
  [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]

def bmAdapter : Adapters.LoRA.Parameters Float 3 1 2 :=
  { inputFactor := [[1.0], [0.0], [2.0]]
    outputFactor := [[1.0, 3.0]] }

#eval Adapters.LoRA.weightUpdate bmAdapter (0.5 : Float)
```
```leanOutput bmLora (whitespace := lax)
[[0.500000, 1.500000], [0.000000, 0.000000],
 [1.000000, 3.000000]]
```

$`AB` is the outer product $`(1,0,2)^{\top}(1,3)`, and half of it is the matrix above. Adding it to
the base weight gives the weight the adapted layer actually applies:

```lean (name := bmLoraWeight)
-- Materialize the effective weight to see which entries the
-- rank-one update changes.
#eval Adapters.LoRA.effectiveWeight bmBase bmAdapter
  (0.5 : Float)
```
```leanOutput bmLoraWeight (whitespace := lax)
[[1.500000, 1.500000], [0.000000, 1.000000],
 [1.000000, 3.000000]]
```

The third row was all zeros in the base weight and now carries $`(1,3)`: the adapter gave the layer
a use for a coordinate it previously discarded. Applying it to two input rows:

```lean (name := bmLoraApply)
-- Apply the same effective projection to both rows;
-- batching does not duplicate adapter
-- weights.
def bmRows : Tensor Float [2, 3] :=
  [[1.0, 1.0, 1.0], [1.0, 0.0, 0.0]]

#eval Adapters.LoRA.linear (batchShape := [2])
  bmRows bmBase bmAdapter (0.5 : Float)
```
```leanOutput bmLoraApply (whitespace := lax)
[[2.500000, 5.500000], [1.500000, 1.500000]]
```

Against the base weight alone:

```lean (name := bmBaseApply)
-- The base-only result isolates the contribution already
-- present before the adapter is added.
#eval Tensor.matmul bmRows bmBase
```
```leanOutput bmBaseApply (whitespace := lax)
[[1.000000, 1.000000], [1.000000, 0.000000]]
```

Both rows moved. The adapter reads a combination of the first and third coordinates; the first row
has an additional contribution from its nonzero third coordinate. Note the explicit
`(batchShape := [2])` : the input type is
`Tensor α (batchShape.appendDim inputWidth)`, and Lean will not invert that from a literal shape
`[2, 3]` on its own.

The two rows also separate the adapter's contributions. The first row multiplies every row of
the effective weight, so its outputs add to `(2.5, 5.5)`. The second row selects only the first
weight row, giving `(1.5, 1.5)`. Keeping the base result beside the adapted result makes the change
visible independently of any training procedure. The chosen rank restricts how the weight update
is factored. Here we supplied both factors by hand, so the outputs show their effect before
any training.

Use `Adapters.LoRA.weightUpdate adapter scale` to inspect only the scaled low-rank update, or
`Adapters.LoRA.effectiveWeight base adapter scale` to construct the combined weight. Keeping
`scale` explicit supports the usual $`\alpha/\mathrm{rank}` choice as well as scheduled or
experimental scales.

These are pure tensor definitions, not yet a trainer-integrated LoRA workflow. They do not insert
an adapter into an `nn.Sequential`, initialize its factors, freeze the base weight, or tell an
optimizer to update only $`A` and $`B`. A training experiment must currently wire those choices into
its parameter pack and forward/loss program explicitly.

# Trainer Objectives

Architecture determines the output tensor, not what that tensor means. A `[classes]` output can be
used as logits for cross entropy, scores for a margin loss, or values passed to a custom objective.

TorchLean therefore writes:

```
-- The objective interprets model outputs and targets; it
-- does not alter the layer architecture.
Trainer.new model { objective := .mse }
Trainer.new model { objective := .oneHotCrossEntropy 0 }
Trainer.new model { objective := .custom lossProgram }
```

Mean-squared error is the default objective. Classification variants specify their target
convention and the zero-based class dimension. For example, `[batch, classes]` uses axis `1`, while
`[batch, time, vocabulary]` uses axis `2`. A custom objective supplies a checked scalar loss
program. This makes the loss visible in the training configuration rather than baking it into the
model architecture.

The next chapters pair the model with data and a loss, then instantiate the mutable state needed
for training. Graph lowering later consumes the same immutable definition, including its shapes
and parameter order.
