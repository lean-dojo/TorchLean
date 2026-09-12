import VersoManual
import NN.API
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.Roles

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean
open TorchLean
open Lean.Elab.Tactic.GuardMsgs.WhitespaceMode (lax)

#doc (Manual) "Training" =>
%%%
tag := "training-from-scratch"
file := "Training___-One-State-Transition-At-A-Time"
%%%

Training repeatedly transforms parameters, optimizer memory, data order, and runtime state.
TorchLean's high-level trainer packages that transition, while the lower manual API exposes each
piece.

The useful unit to follow is one update. A sample enters the current model, the objective reduces
its prediction to a scalar, and reverse mode produces one gradient tensor for each trainable state
tensor. The optimizer consumes those gradients together with its own memory and installs the next
parameters. Reading the chapter in that order explains why choosing a model is only the first part
of a training program, and why a saved prediction alone cannot tell us how the next update will
behave.

We will train the running $`2\to8\to1` MLP and then inspect its state transitions. Named Lean
blocks and their outputs are checked during the guide build. Command-line and Python transcripts
are separately recorded runs, with commands supplied for reproduction.

# MLP Training

From the repository root, train the model on the CPU for 200 updates. The fixed seed makes this
run reproducible under the same arithmetic and execution settings:

```terminal
# Fix the seed and update count for the MLP trace
# interpreted below.
lake exe torchlean quickstart_mlp \
  --device cpu --steps 200 --seed 2026
```

```terminal +output
== Quickstart: simple MLP training (seed=2026, steps=200) ==
target(heldout)    = [0.200000]
untrained(heldout) = [-0.088261]
dataset size = 25
mean_loss(before training) = 0.495227
step 0: loss=0.250488
step 25: loss=0.586318
step 50: loss=0.847444
step 75: loss=0.003933
step 100: loss=0.023397
step 125: loss=0.069799
step 150: loss=0.000061
step 175: loss=0.029500
mean_loss(after training) = 0.002402
steps=200 arithmetic=native scalar=Float32 loss=0.495227 -> 0.002402
trained(heldout) = [0.228325]
```

The source is
{src "NN/Examples/Quickstart/SimpleMlpTrain.lean"}[`SimpleMlpTrain.lean`].
Its program structure is:

```
-- Construction, data, and per-run options meet at the call
-- that returns the trained snapshot.
model builder
  + trainer configuration
  + dataset
  + train options
  -> trained result
```

The log separates individual training losses from the mean loss over the dataset. Start with
these entries:

- `dataset size = 25` is the number of items the trainer got after materializing the dataset, not
  the number of rows in a file. The {ref "datasets-loaders"}[data chapter] is where that number
  comes from.
- `mean_loss(before training)` and `mean_loss(after training)` are the only two full-dataset
  measurements in the run. The trainer takes them once each, at the boundaries.
- The `step k: loss=...` lines are single-sample losses, measured on the tape that produced that
  step's gradient. That is why they are not monotone: `0.586318` at step 25 and `0.847444` at step
  50 use different samples at different parameter points, so their increase alone does not establish
  that training diverged. The
  full-dataset mean fell from `0.495227` to `0.002402` while those numbers bounced around.
- `arithmetic=native scalar=Float32` records which of TorchLean's executable semantics ran. A loss
  curve without that label is ambiguous, and we say why below.
- `untrained(heldout)` and `trained(heldout)` are the same held-out input `(0.25, -0.75)` before and
  after: `-0.088261` moves to `0.228325` against a target of `0.200000`.

The loss values are measurements from this run. The model and optimizer definitions, by contrast,
are reusable objects that can appear in theorem statements or another runtime profile.

# Model Architecture

```lean (name := tfModelDef)
-- The hidden ReLU separates two affine maps with a shared
-- width of eight.
def tfModel : nn.Builder (nn.Sequential [2] [1]) :=
  nn.Sequential![
    nn.linear 2 8,
    nn.relu,
    nn.linear 8 1
  ]
```

Lean reports both the seeded construction stage and the model's input/output shapes:

```lean (name := tfModelType)
-- The builder type exposes the model boundary before
-- initialization or training.
#check tfModel
```

```leanOutput tfModelType
tfModel : nn.Builder (nn.Sequential [2] [1])
```

`nn.Builder` is the seed-consuming half and `nn.Sequential [2] [1]` is the checked model. At this
point we have:

- input and output shapes;
- layer structure;
- parameter shapes;
- seeded initialization actions.

We do not yet have a loss, optimizer, concrete parameter values, or device.

The intermediate width eight belongs to the architecture even though only `[2]` and `[1]` appear
at the outer boundary. It determines both the number of hidden activations for each sample and the
shapes of the two weight matrices. The builder delays choosing their values until a seed is
supplied.
This lets the same architecture be initialized repeatedly for the seed comparison below without
rewriting its forward computation or changing what counts as a well-shaped input.

# Objective And Optimizer Configuration

```lean (name := tfTrainerDef)
-- Bind the objective, Adam configuration, arithmetic, and
-- seed to one reusable trainer.
def tfTrainer (seed : Nat) : Trainer [2] [1] :=
  Trainer.new tfModel
    { objective := .meanSquaredError
      optimizer := optim.adam { learningRate := 0.03 }
      arithmetic := .native
      execution := .eager
      seed := seed }
```

`Trainer.new` runs the seeded builder and stores persistent choices. It still does not consume data
or update a parameter. Its summary shows the initialized model's layer shapes and state layout:

```lean (name := tfSummaryRun)
-- The summary counts trainable parameters and all
-- persistent state separately.
#eval (tfTrainer 2026).printSummary
```

```leanOutput tfSummaryRun (whitespace := lax)
model:
Sequential: [2] -> [1], layers=3, params=33, state=33
  [0] Linear(2, 8): [2] -> [8] params=24, state=24 [[8, 2], [8]]
  [1] ReLU: [8] -> [8] params=0, state=0 []
  [2] Linear(8, 1): [8] -> [1] params=9, state=9 [[1, 8], [1]]
```

Thirty three numbers: $`8\times2` weights and $`8` biases in the first affine layer, then
$`1\times8` weights and $`1` bias in the second. `params` counts trainable scalars and `state`
counts everything the checkpoint stores, so a model with normalization buffers would show the two
numbers disagreeing.

For regression, the default objective is mean-squared error. For a prediction and target with
$`n` entries, where $`n>0`, the loss is:

$$`
L(\theta;x,y)
=
\frac1n\sum_{i=1}^{n}
\left(F_\theta(x)_i-y_i\right)^2.
`

Changing `.meanSquaredError` to `.oneHotCrossEntropy axis` changes the objective and target
convention without changing the architecture. The zero-based `axis` may name any output dimension;
Lean rejects an axis outside the output shape. A custom objective supplies a checked scalar loss
program.

Here the output has shape `[1]`, so the mean contains one squared residual. In the batched version,
`[5, 1]` contains five residuals and the same objective averages all five. This reduction is part of
the loss definition: it fixes the scale of the gradients passed to Adam. Changing a sum to a mean
would therefore change more than the number printed in the log, even if the model and target tensors
were otherwise identical.

The optimizer is a small record, and it can print itself back in the syntax you would use to
construct it:

```lean (name := tfOptim)
-- Print the Adam configuration, including defaults omitted
-- from the constructor call.
#eval (optim.adam { learningRate := 0.03 }).describe
```

```leanOutput tfOptim (whitespace := lax)
"optim.adam { learningRate := 0.030000, beta1 := 0.900000,
  beta2 := 0.999000, epsilon := 1.000000e-8 }"
```

The moment coefficients and epsilon match PyTorch's defaults. The corresponding PyTorch
configuration prints:

```
lr, betas, eps = 0.03 (0.9, 0.999) 1e-08
```

The exponent notation in `epsilon := 1.000000e-8` preserves a distinction that fixed six-decimal
formatting would hide. Printing this value as `0.000000` would suggest a different update rule.
`describe` uses exponent form below `1e-4`, so small configuration values remain visible in logs.

Adam is {Informal.citet adam2015}[]; the single-sample step underneath it is older than deep
learning and goes back to {Informal.citet robbins1951}[].

# Initial Parameter Values

Before any data arrives, building the model has fixed all 33 parameter values. The first loss
depends on these values together with the data, objective, mode, and arithmetic. Reading the
initial state separates those choices from the later optimizer updates:

```lean (name := tfInitDef)
-- Name the four state shapes so each initialized weight or
-- bias can be read at its checked
-- index.
abbrev tfShapes : List Shape := [[8, 2], [8], [1, 8], [1]]

def tfInitial : nn.State Float tfShapes :=
  (nn.initialState (nn.build 2026 tfModel)).cast (by rfl)

#eval show IO Unit from do
  IO.println s!"input weights = {tfInitial.get 0}"
  IO.println s!"input bias    = {tfInitial.get 1}"
  IO.println s!"output bias   = {tfInitial.get 3}"
```

```leanOutput tfInitDef (whitespace := lax)
input weights = [[-0.337801, 0.526946],
 [-0.767575, 0.519218],
 [0.116565, 0.347734],
 [-0.242321, 0.244720],
 [-0.419680, 0.002862],
 [-0.461676, -0.360324],
 [-0.510277, 0.741238],
 [-0.108099, 0.091942]]
input bias    = [0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
 0.000000, 0.000000, 0.000000]
output bias   = [0.000000]
```

Two decisions are visible. Biases start at exactly zero. Weights are drawn from Xavier uniform
initialization {Informal.citep glorot2010}[], whose bound for a $`2\to8` layer is

```lean (name := tfXavier)
-- Xavier uses both fan-in two and fan-out eight to set this
-- layer weight bound.
#eval Float.sqrt (6.0 / (2.0 + 8.0))
```

```leanOutput tfXavier
0.774597
```

and every entry above sits inside $`\pm0.774597`, the largest magnitude being `0.767575`.

The four shapes identify which values belong to which computation. The first matrix forms eight
weighted combinations of the two input coordinates; the first bias shifts those combinations before
ReLU. The second matrix combines the eight resulting activations, and the final bias shifts the
prediction. Zero biases do not make the initial model zero, because the two weight matrices already
contain nonzero values. The negative initial held-out prediction is consistent with this layout.

PyTorch makes both decisions differently. Its `Linear` uses Kaiming uniform initialization
{Informal.citep he2015}[] with bound $`1/\sqrt{n_{\text{in}}}` and gives the bias a random value
from the same interval:

```terminal +output
== default Linear(2,8) initialization, seed 2026 ==
weight max |.| = 0.705845
kaiming bound  = 0.707107
bias           = [0.582698, 0.278374, 0.009535, 0.614985, -0.093329,
                  0.385542, 0.295646, 0.047354]
xavier bound   = 0.774597
```

Different initialization explains why the two runs begin with different losses on the same
architecture and seed: `0.495227` here against `0.510072` in PyTorch. The generators and
initialization schemes are not matched, so equal seed values do not establish equal parameters.
Where we do claim agreement with PyTorch, as in
{ref "pytorch-roundtrip"}[the round-trip chapter], the agreement is on imported weights and forward
values rather than on initialization.

# Training Dataset

The quickstart uses a deterministic grid over $`[-1,1]^2` and labels it with a small piecewise
linear function, so that the target is something a $`2\to8\to1` ReLU network can actually represent:

$$`
y(x)
=
0.8\,\mathrm{relu}(x_1+x_2)
-0.4\,\mathrm{relu}(x_2-x_1)
+0.2.
`

The mathematical coordinates $`x_1,x_2` correspond to Lean's zero-based `x[0]` and `x[1]`.
The label function applies the two ReLUs independently, then combines them:

```lean (name := tfDataDef)
-- Label each grid point using the fixed target,
-- independently of the model being trained.
def tfTarget (x : Tensor Float [2]) : Tensor Float [1] :=
  let relu (v : Float) := if v < 0.0 then 0.0 else v
  [0.8 * relu (x[0] + x[1])
     - 0.4 * relu (x[1] - x[0]) + 0.2]

def tfData : Trainer.Dataset [2] [1] :=
  let inputs := Data.Synthetic.squareGrid (-1.0) 1.0 5
  Data.fromTensors inputs
    (Tensor.mapLeading [5 * 5] tfTarget inputs)
```

`Data.Synthetic.squareGrid` builds a $`5\times5` grid as a `Tensor Float [25, 2]`, and
`Tensor.mapLeading` applies `tfTarget` to each of the 25 leading rows. Materializing the dataset
shows what the trainer will actually see:

```lean (name := tfDataRun)
-- Read the first materialized item to connect the grid
-- construction to the training stream.
#eval show IO Unit from do
  let stream ← tfData.materialize (α := Float)
  IO.println s!"samples = {stream.size}"
  match stream.get? 0 with
  | some sample =>
      IO.println s!"input  = {sample.input}"
      IO.println s!"target = {sample.target}"
  | none => IO.println "empty dataset"
```

```leanOutput tfDataRun
samples = 25
input  = [-1.000000, -1.000000]
target = [0.200000]
```

The corner $`(-1,-1)` lands on the constant term: both ReLUs are zero there, so the label is exactly
`0.2`. That is the same `dataset size = 25` the command-line transcript reports, and the same first
row PyTorch computes from the same grid.

The labels are computed once from the grid and do not depend on the trainer's parameters. During
training, only the model's attempt to reproduce those labels changes. The target's two ReLU terms
also explain the architecture choice: a hidden unit can form each required linear combination before
applying ReLU, and the output layer can combine them. Representability makes this a useful small
regression problem; it does not tell Adam which representation to find or how many updates it needs.

The dataset type matches the model map $`[2]\to[1]`. A batched model would require a batched dataset
with a leading dimension in both item shapes, which we do further down. Everything about where
samples come from, how they are shuffled, and what happens to a short final batch belongs to
{ref "datasets-loaders"}[the data chapter]. Fixing the grid here lets the experiments below vary
initialization, optimizer, or update grouping while using the same samples.

# Training And Results

A complete run is one `IO` action. This checked example trains the model and evaluates the same
held-out input used by the command-line program:

```lean (name := tfFullRun)
-- The returned result evaluates held-out data using the
-- parameters reached after 200 updates.
#eval show IO Unit from do
  let trained ← (tfTrainer 2026).train tfData
    { steps := 200, logEvery := 50 }
  trained.printSummary
  let heldout : Tensor Float [2] := [0.25, -0.75]
  let prediction ← trained.predict heldout
  IO.println s!"trained(heldout) = {prediction}"
```

```leanOutput tfFullRun
dataset size = 25
mean_loss(before training) = 0.495227
step 0: loss=0.250488
step 50: loss=0.847444
step 100: loss=0.023397
step 150: loss=0.000061
mean_loss(after training) = 0.002402
steps=200 arithmetic=native scalar=Float32 loss=0.495227 -> 0.002402
trained(heldout) = [0.228325]
```

The initial mean `0.495227`, the shared logged steps, the final mean `0.002402`, and the prediction
`[0.228325]` agree with the command-line transcript. Only the logging interval changed, from 25
updates to 50. The checked block recomputes these values when the page builds.

The returned `Trainer.Result` retains:

- final parameters and buffers, readable as `Float` tensors through `trained.state` and
  writable to a checkpoint file through `trained.save path`;
- the runtime model state;
- the completed step count and numeric before/after losses;
- prediction and verification closures.

`Trainer.load trainer path data` restores a saved result for the same trainer; it evaluates `data`
once so that the restored report carries the mean loss of the restored parameters rather than a
copied number.

`trained.report.loss.before` and `trained.report.loss.after` are ordinary host `Float` values. The
model still executes with the arithmetic selected by the trainer, including `.ieee` ;
`Session.meanLoss` converts per-item losses to host `Float` before averaging; the result stores
those before/after means. Callers can therefore compare,
serialize, or plot losses directly without parsing printed backend values.

Prediction accepts Float tensors and performs conversion into the arithmetic representation selected
by the trainer. It does not rebuild or reinitialize the model.

There is also an ownership boundary at the result. `Session.finish` freezes the model state used by
the result's prediction and verification closures. Further steps or a later checkpoint load on the
live session cannot silently change an already returned result. This matters when keeping several
training milestones in memory: a reported loss and the model saved beside it must continue to refer
to the same parameter snapshot while the experiment proceeds.

## Updated Parameters

`trained.state` returns the trained values in the layout printed by the model summary. Comparing
the output bias with its initial value exposes one component of the parameter update:

```lean (name := tfStateRun)
-- Check the returned shape list before reading the output
-- bias as the fourth state tensor.
#eval show IO Unit from do
  let trained ← (tfTrainer 2026).train tfData
    { steps := 40 }
  IO.println s!"state shapes = {trained.stateShapes}"
  let state ← trained.state
  if sameShapes : trained.stateShapes = tfShapes then
    let known := state.cast sameShapes
    IO.println s!"initial output bias = {tfInitial.get 3}"
    IO.println s!"trained output bias = {known.get 3}"
  else
    IO.println "unexpected state layout"
```

```leanOutput tfStateRun
dataset size = 25
mean_loss(before training) = 0.495227
mean_loss(after training) = 0.125425
state shapes = [[8, 2], [8], [1, 8], [1]]
initial output bias = [0.000000]
trained output bias = [0.279321]
```

`trained.stateShapes` is a field of a value returned by `IO`, so Lean does not know at elaboration
time that it is
`[[8, 2], [8], [1, 8], [1]]`; the equality is checked once at runtime and then used to transport the
state into a type where `get 3` is meaningful. This is the same check a checkpoint loader performs,
and it is why a checkpoint for a wider model cannot be quietly accepted here.

The displayed output-bias change isolates one coordinate of a much larger update. It does not
attribute the loss reduction to that bias: both matrices and the hidden bias have been trainable
throughout the forty steps. The shape check establishes how to interpret the returned pack, while
the printed numbers show the values found there in this run. Keeping those two roles separate makes
state inspection useful without treating a single changed coordinate as an explanation of training.

# Gradient Descent Updates

Let $`\theta_t` be the parameter pack at update $`t`, $`(x_t,y_t)` the selected sample, and
$`\eta` the learning rate. A plain SGD update is:

$$`
g_t=\nabla_\theta L(\theta_t;x_t,y_t),
\qquad
\theta_{t+1}=\theta_t-\eta g_t.
`

Every symbol is structured:

- $`\theta_t` is a shape-indexed pack of differently shaped tensors sharing element type `α`;
- $`g_t` has exactly the same pack structure;
- $`L` is a scalar tensor program;
- the subtraction and scaling occur coordinatewise in the selected arithmetic semantics.

An optimizer is therefore not merely a function from a flat vector to a flat vector. It owns
shape-aligned state.

For this model, reverse mode returns gradients with shapes `[[8, 2], [8], [1, 8], [1]]`, in the
same order as the parameters. A bias gradient cannot accidentally be used as a weight gradient
merely
because both are stored in the same pack. The optimizer can traverse corresponding tensors and then
corresponding scalar coordinates. That alignment is what allows the same update implementation to
work for this MLP and for a model with a different number of layers or tensor ranks.

You can watch a single update in isolation. `Session.step` returns the loss it measured *before*
applying the update, so feeding it the same sample twice shows the effect of exactly one step:

```lean (name := tfOneStep)
-- Repeating the same sample makes the second pre-update
-- loss measure the first update effect.
#eval show IO Unit from do
  let session ← (tfTrainer 2026).open
  let stream ← tfData.materialize (α := Float)
  match stream.get? 0 with
  | some sample =>
      let first ← session.step sample
      let second ← session.step sample
      IO.println s!"loss before update 1 = {first}"
      IO.println s!"loss before update 2 = {second}"
  | none => IO.println "empty dataset"
```

```leanOutput tfOneStep
loss before update 1 = 0.250488
loss before update 2 = 0.110670
```

One Adam step on the corner sample cut its loss from `0.250488` to `0.110670`. The first value
matches step 0 of the longer run because the seed, sample, and arithmetic match. The second call
measures the effect of the first update before applying another update of its own. These two
values concern the corner sample; a step that helps it can increase losses on other samples.
Dataset loss must be measured separately. For the stochastic approximation setting, see
{Informal.citet robbins1951}[].

# Mixed-Dtype Indexed Training

One tensor remains homogeneous, but a training problem need not use one element type everywhere.
Language models are the standard case: learned parameters, logits, and the scalar loss use a
floating-point type `α`, while token inputs and targets use bounded indices such as `Fin vocab`.
The public objective API keeps that distinction in its type:

```lean (name := tfDataStep)
-- The reusable step accepts β-valued data while its
-- objective and parameters use α.
#check @Module.Objective.dataStep
```

```leanOutput tfDataStep (whitespace := lax)
@Module.Objective.dataStep : {α β : Type} →
  [inst : Storage α] →
    [inst_1 : Storage β] →
      [inst_2 : Context α] →
        [Runtime.FromFloat α] →
          {stateShapes : List Shape} →
            {firstShape secondShape : Shape} →
              Module.Objective α β stateShapes [] [firstShape, secondShape] →
                optim.Optimizer →
                  IO (Tensor β firstShape → Tensor β secondShape → IO Unit)
```

The outer `IO` action creates a reusable step function taking two `β` tensors. Those tensors
contain tokens; the step updates module state and returns `Unit`. The floating-point type `α`
belongs to the objective and to the values returned by loss evaluation. A character-level GPT
uses this interface in
{src "NN/Examples/Models/Sequence/CharGpt.lean"}[`CharGpt.lean`]:

```
-- Keep discrete token tensors at the data boundary and
-- floating-point logits at the prediction
-- boundary.
let trainStep ← module.dataStep <|
  optim.adamW { learningRate := train.training.learningRate }
let evaluateLoss ← module.dataLossEvaluator evalDef
let predict ← module.indexedPredictor model
```

`inputTokens` and `targetTokens` keep their discrete element type; the returned loss and logits use
`α`. `dataLossEvaluator` creates a reusable evaluator when repeated loss measurements should share
the trained state. These functions hide the heterogeneous runtime context and its session
lifecycle. `TensorPack` still exists as the typed internal representation of differently shaped
parameters and graph inputs, but normal indexed-model code does not construct or deconstruct it.

The distinction between `α` and `β` is computational as well as notational. A token chooses a row
of an embedding table; the chosen row contains floating-point parameters that can receive gradients.
The integer identifying that row is not itself a floating-point parameter to optimize. Returning
`Unit` from the step makes its purpose explicit: the observable effect is the updated module state.
Use the loss evaluator when the caller also needs a scalar measurement, or the predictor when it
needs the full output tensor.

# Training State

It is tempting to describe training as repeated calls to `backward`, but the value passed from one
update to the next is larger than a gradient. For a reproducible run we need to account for:

:::table +header
*
  * State
  * Why the next update needs it
*
  * parameters
  * they are the point at which the next loss and gradient are evaluated
*
  * optimizer memory
  * Adam moments, momentum buffers, and step counters change the update
*
  * scheduler state
  * the update index determines the learning rate
*
  * loader or stream position
  * it determines the next samples and final partial-batch behavior
*
  * random-generator state
  * dropout, augmentation, sampling, and some data sources consume it
*
  * model buffers and mode
  * normalization statistics and other persistent state may change in training mode
*
  * backend profile
  * it fixes the providers and backward ownership used by the executable step
:::

The live session owns model and optimizer state; the loop manages sample order, and external
generators remain the caller's responsibility. A result is not a snapshot of every row in this
table. The manual API exposes them
when an experiment needs a custom loop or a checkpoint must record more than parameter tensors.
Saving only weights is enough for inference, but it is not enough to resume Adam at the same
update.

The buffers row is not hypothetical. Dropout consumes randomness and denotes a different function
in training and evaluation mode {Informal.citep dropout2014}[], and batch normalization keeps
running statistics that the forward pass updates without the optimizer touching them
{Informal.citep batchnorm2015}[]. Our `state=33` above equals `params=33` only because this model
has neither.

# Adam Moments And Step Counter

Adam {Informal.citep adam2015}[] maintains an exponentially weighted mean $`m_t` of the gradients
and a mean $`v_t` of their coordinatewise squares. Here $`t=1` denotes the first update:
$`\theta_0` is the initial parameter pack, $`m_0=v_0=0`, and $`g_t` is the gradient evaluated
at $`\theta_{t-1}`. The decay factors $`\beta_1,\beta_2\in[0,1)` determine how much of each
previous estimate survives:

$$`
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,
`

$$`
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2.
`

At the first update, the zero initial moments give $`m_1=(1-\beta_1)g_1` and
$`v_1=(1-\beta_2)g_1^2`. Dividing by $`1-\beta_1` and $`1-\beta_2` recovers
$`g_1` and $`g_1^2`. After $`t` updates, the weights assigned to the observed gradients
sum to $`(1-\beta_1)\sum_{k=0}^{t-1}\beta_1^k=1-\beta_1^t`; the corresponding
weights on their squares sum to $`1-\beta_2^t`. Bias correction normalizes each
estimate by this total weight:

$$`
\widehat m_t=\frac{m_t}{1-\beta_1^t},
\qquad
\widehat v_t=\frac{v_t}{1-\beta_2^t},
`

and the parameter update is:

$$`
\theta_t
=
\theta_{t-1}-\eta
\frac{\widehat m_t}{\sqrt{\widehat v_t}+\epsilon}.
`

This counter records completed updates, so the first iteration in a zero-based loop has
Adam counter $`t=1`. The square, square root, and division act coordinatewise.
The first moment supplies a smoothed
gradient, while the square root of the second moment rescales each coordinate; the positive
$`\epsilon` keeps a zero second-moment estimate from producing a zero denominator.
Both moment packs have the same dependent tensor shapes as $`\theta`. Restoring only parameters
from a checkpoint loses these estimates and the step counter used in bias correction, so it does
not resume the same Adam update.

PyTorch exposes the corresponding state for each parameter:

```
per-parameter state keys: ['exp_avg', 'exp_avg_sq', 'step']
step = 1.0 exp_avg shape [8, 2]
```

`exp_avg` is $`m_t`, `exp_avg_sq` is $`v_t`, and `step` is $`t`. The shape of `exp_avg` matches the
weight it belongs to, which is the untyped version of the statement we get from the type: a moment
pack has the same shape layout as the parameter pack. Momentum {Informal.citep polyak1964}[] is the
same story with one buffer instead of two, and decoupled weight decay
{Informal.citep adamw2019}[] changes the update without changing the state layout.

TorchLean also provides SGD, momentum SGD, AdamW, AdaGrad, RMSProp, Adadelta, and Muon-related
module-level runtime configuration. Muon requires an explicit orthogonalization backend and is
not one of the opaque `optim.Optimizer` constructors accepted by every high-level trainer path.
Their state and laws remain optimizer-specific.

# Update Counts And Gradient Accumulation

For the unbatched model $`[2]\to[1]`, this configuration:

```
-- Four samples contribute to each of the 200 optimizer
-- updates.
steps := 200
samplesPerStep := 4
```

means 200 optimizer updates. Each update consumes four samples, differentiates them at the same
parameter point, and averages their gradient packs. Logging reports the mean pre-update loss from
those same forward tapes.
The trainer does not run an extra forward pass merely to print that loss. Before building those
tapes, it advances mutable model buffers once per item; trainable parameters stay fixed until the
mean gradient is ready. This matters for dropout, BatchNorm, and any operation whose tape retains
data needed by backward: the displayed scalar, saved state, and gradient come from the same step.

Fifty updates with one sample each and fifty updates with four samples each are two different
amounts of work, and they land in different places:

```lean (name := tfAccum)
-- Hold update count fixed while changing how many sample
-- gradients are averaged per update.
#eval show IO Unit from do
  IO.println "-- samplesPerStep = 1"
  let _ ← (tfTrainer 2026).train tfData
    { steps := 50, samplesPerStep := 1 }
  IO.println "-- samplesPerStep = 4"
  let _ ← (tfTrainer 2026).train tfData
    { steps := 50, samplesPerStep := 4 }
```

```leanOutput tfAccum
-- samplesPerStep = 1
dataset size = 25
mean_loss(before training) = 0.495227
mean_loss(after training) = 0.335392
-- samplesPerStep = 4
dataset size = 25
mean_loss(before training) = 0.495227
mean_loss(after training) = 0.008248
```

The second run saw four times as many samples, so this is not evidence that averaging gradients is
better per unit of work. It is evidence of what the option means: `steps` counts updates, and the
starting point of both runs is identical because `samplesPerStep` does not touch initialization.

Averaging four gradients at one parameter point is also different from taking four consecutive
updates. In the consecutive case, the second gradient sees parameters changed by the first update;
in the accumulated case, all four gradients see the original parameters. Adam then updates its
moments once from their mean. The option therefore changes both how much data contributes to each
update and how often the optimizer's memory advances, even before considering a vectorized
implementation.

For a true vectorized minibatch, the batch axis goes into the model type, and `nn.linear` takes it
as `batchShape`:

```lean (name := tfBatchedDef)
-- Put five samples into each tensor item and carry that
-- axis through both linear layers.
def tfBatchedModel :
    nn.Builder (nn.Sequential [5, 2] [5, 1]) :=
  nn.Sequential![
    nn.linear 2 8 (batchShape := [5]),
    nn.relu,
    nn.linear 8 1 (batchShape := [5])
  ]

def tfBatchedData : Trainer.Dataset [5, 2] [5, 1] :=
  Data.batch 5 tfData (shuffle := true) (seed := 2026)
```

Training it is the same call, and the summary tells us that the batch axis cost us no parameters:

```lean (name := tfBatchedRun)
-- The model summary verifies shared parameters before the
-- batched training run starts.
#eval show IO Unit from do
  let trainer := Trainer.new tfBatchedModel
    { objective := .meanSquaredError
      optimizer := optim.adam { learningRate := 0.03 }
      seed := 2026 }
  trainer.printSummary "batched model"
  let trained ← trainer.train tfBatchedData { steps := 40 }
  trained.printSummary
```

```leanOutput tfBatchedRun (whitespace := lax)
batched model:
Sequential: [5, 2] -> [5, 1], layers=3, params=33, state=33
  [0] Linear(2, 8): [5, 2] -> [5, 8] params=24, state=24 [[8, 2], [8]]
  [1] ReLU: [5, 8] -> [5, 8] params=0, state=0 []
  [2] Linear(8, 1): [5, 8] -> [5, 1] params=9, state=9 [[1, 8], [1]]
dataset size = 5
mean_loss(before training) = 0.495227
mean_loss(after training) = 0.013059
steps=40 arithmetic=native scalar=Float32 loss=0.495227 -> 0.013059
```

The parameter count remains 33 because all five samples share the layer weights. The dataset now
contains five items, each holding five samples; no incomplete group remains in this example.
The initial mean loss again prints as `0.495227`: this MLP applies the same initial parameters to
the same 25 samples, grouped differently. The output lets us check the grouping separately from
the subsequent optimizer trajectory.

With the default `TrainOptions.samplesPerStep := 1`, an update consumes one item and therefore runs
one vectorized forward and backward pass over five samples. Setting `samplesPerStep` above one would
accumulate gradients across several of these tensor minibatches. Vectorization can change reduction
order and performance; it is not an optimization flag applied to the first model.

The maintained CSV example is already batched this way:

```terminal
# This CSV experiment uses tensor batches of five and its
# own regression data.
python3 NN/Examples/Data/generate_small_data.py
lake exe torchlean data_csv \
  --device cpu --batch 5 --steps 5 --seed 2026
```

```terminal +output
== CSV loader training tutorial ==
model:
Sequential: [5, 2] -> [5, 1], layers=3, params=33, state=33
  [0] Linear(2, 8): [5, 2] -> [5, 8] params=24, state=24 [[8, 2], [8]]
  [1] ReLU: [5, 8] -> [5, 8] params=0, state=0 []
  [2] Linear(8, 1): [5, 8] -> [5, 1] params=9, state=9 [[1, 8], [1]]
data_dir = NN/Examples/Data
csv_path  = NN/Examples/Data/small_regression.csv
seed      = 2026
train     = Adam(lr=0.05), steps=5, batch_size=5, shuffle=true, drop_last=true
dataset size = 5
mean_loss(before training) = 0.210192
mean_loss(after training) = 0.055578
steps=5 arithmetic=native scalar=Float32 loss=0.210192 -> 0.055578
predict(batch=heldout) = [[0.303611], [0.303611], [0.303611], [0.303611], [0.303611]]
```

The five identical predictions are not a bug: the held-out batch repeats one row five times, so a
batched model answers it five times.

The CSV loss cannot be read as the next stage of the earlier grid experiment. Its starting mean is
`0.210192`, not `0.495227`, because this command uses the CSV example's data and configuration.
Within that run, the before and after values measure the same dataset and are directly comparable.
Across the two examples, the matching architecture and parameter count explain the shared API, while
the different target values and learning rate explain why the logs describe separate experiments.

# Eager And Typed Graph Execution

Compare:

```terminal
# Keep the run configuration fixed except for eager versus
# typed graph execution.
lake exe torchlean quickstart_mlp \
  --device cpu --execution eager --steps 20 --seed 2026

lake exe torchlean quickstart_mlp \
  --device cpu --execution typed-graph --steps 20 --seed 2026
```

Both print the same three numbers:

```terminal +output
mean_loss(before training) = 0.495227
mean_loss(after training) = 0.401184
trained(heldout) = [0.365380]
```

The trainer method remains `train`; `execution := .typedGraph` changes how that method runs without
changing the model's `forward` definition. {ref "execution-modes"}[The execution chapter] develops
the graph reuse and proof boundaries in detail.

Agreement on twenty updates of a 33-parameter model is evidence, not proof. It is worth having
anyway, because the two execution modes take different code paths through the runtime, and a
disagreement deserves investigation of operations, reductions, and numeric tolerances.

Typed graph trainer execution is currently CPU-only. A non-CPU typed graph request is rejected
rather than silently falling back to a different semantics.

# Device Selection And Backend Profiles

Programmatically, `Trainer.Config` extends `Trainer.RunConfig`, so the device can be set in the
same record as the objective and seed. `RunConfig.withDevice` checks the device against the table of
maintained backend profiles and returns an `Except`:

```lean (name := tfDevices)
-- Profile lookup returns either an accepted configuration
-- or the reason no profile was found.
#eval show IO Unit from do
  let run : Trainer.RunConfig :=
    { optimizer := optim.adam { learningRate := 0.03 } }
  for device in [Runtime.Device.cpu, .cuda, .metal] do
    match run.withDevice device with
    | .ok _ => IO.println s!"{device.cliName}: accepted"
    | .error message =>
        IO.println s!"{device.cliName}: {message}"
```

```leanOutput tfDevices (whitespace := lax)
cpu: accepted
cuda: accepted
metal: device `metal` has no maintained runtime profile;
  provide an explicit backend profile
```

The CPU-only guide build still accepts `cuda` at this stage. `withDevice` checks whether a
maintained profile exists; opening a session later checks whether the binary can execute it.
Metal, ROCm, TPU, and Trainium have configuration names but no maintained profile accepted here.
CUDA execution additionally requires a binary linked with the native runtime:

```terminal
# The CUDA request needs a native-runtime build; the report
# records selected operation
# providers.
lake -R -K cuda=true exe torchlean quickstart_mlp \
  --device cuda --steps 20 --seed 2026 --show-backend
```

Execution validation rejects CUDA when the native runtime is unavailable. The profile lookup
above is intentionally weaker than opening an executable session.

Read the two successful lookup rows as configuration results. They return a run configuration with
a maintained device profile attached; they have not allocated a tensor on that device or executed a
kernel. This distinction lets configuration code remain usable on a CPU-only documentation build.
The later session boundary is where a CUDA request must meet the capabilities of the linked runtime,
so a successful lookup does not replace checking whether the requested execution can actually start.
`--show-backend` prints the selected implementation for each operation; the
{ref "backend-selection"}[backend chapter] explains the profile, provider, VJP, and evidence fields
in that report.

# Training Arithmetic

The common executable selections are:

```
# These alternative flags choose the scalar arithmetic used
# by a training run.
--arithmetic native
--arithmetic ieee
```

Native arithmetic uses Lean's builtin `Float32` operations. IEEE arithmetic uses TorchLean's
independent raw-bit binary32 reference and is much slower, but exposes precise finite and
exceptional behavior. On this run the two agree to the last printed digit, and only the label
changes:

```
steps=20 arithmetic=native scalar=Float32 loss=0.495227 -> 0.401184
steps=20 arithmetic=ieee scalar=IEEE32Exec loss=0.495227 -> 0.401184
```

That is the outcome we want and not one we are entitled to assume: the bit-level reference is a
separate implementation of binary32, so the displayed agreement is a regression check for this
workload; it does not establish
bit equality or identify the implementation of each operation. Proof-level `Real` and rounded-real
`FP32` are not executable trainer choices.
`FP32` has binary32 precision and gradual-underflow parameters, but no upper exponent bound, NaN,
infinity, or signed zero; bridge theorems therefore require finite and no-overflow hypotheses when
relating it to `IEEE32Exec`. {ref "floats"}[The floating-point chapter] is where those boundaries
are drawn.

A loss curve without its arithmetic semantics is incomplete. The same architecture and seed may
round differently in native binary32, the bit-level reference, a fused CUDA kernel, or an external
provider.

# Model And Optimizer Checkpoints

Native `Float32` modules use an exact binary32 checkpoint on CPU and CUDA. The binary64 `Float`
path retains its exact-bit JSON format on CPU. Neither representation passes
through decimal text. The expected state shapes come from the model, and every tensor must have the
right shape and scalar count before the checkpoint is accepted:

```
-- Use this model definition as the expected tensor manifest
-- for the loaded state.
def loadForThisModel (path : System.FilePath) :=
  Checkpoint.State.load (nn.build 2026 model) path
```

The result is an `IO` action returning tensors whose dependent shape list is exactly
`nn.stateShapes (nn.build 2026 model)`. The runtime checkpoint loader turns such a checked pack into
runtime state handles, while `Checkpoint.load` and `Checkpoint.save` work with
an already instantiated runtime module.

The loader requires an exact tensor manifest. Missing tensors, extra tensors, shape mismatches, and
malformed scalar payloads are rejected before any state is installed. This matters when two models
share an initial state prefix: a checkpoint for the larger model cannot be accepted as a checkpoint
for the smaller one merely because the first few shapes agree.

Here is the round trip, end to end, in a temporary directory that vanishes when the block finishes:

```lean (name := tfCheckpoint)
-- Restore into a differently seeded trainer so unchanged
-- fresh initialization would be visible.
#eval show IO Unit from do
  let trained ← (tfTrainer 2026).train tfData
    { steps := 40 }
  IO.FS.withTempDir fun dir => do
    let path := dir / "mlp.state"
    trained.save path
    let restored ← (tfTrainer 999).load path tfData
    restored.printSummary
    let heldout : Tensor Float [2] := [0.25, -0.75]
    let before ← trained.predict heldout
    let after ← restored.predict heldout
    IO.println s!"trained  = {before}"
    IO.println s!"restored = {after}"
```

```leanOutput tfCheckpoint
dataset size = 25
mean_loss(before training) = 0.495227
mean_loss(after training) = 0.125425
steps=0 arithmetic=native scalar=Float32 loss=0.125425 -> 0.125425
trained  = [0.189974]
restored = [0.189974]
```

The restoring trainer uses seed `999`, so matching predictions show that loading replaced its
initial state. `steps=0` says the restored result performed no updates. The two losses in the
restored report are both `0.125425`, which equals the trained run's final loss, because
`Trainer.load` evaluates the supplied dataset once rather than copying a number it was handed.

The equality of these two predictions checks a different property from loss reduction. It checks
that the saved state, reloaded into the same architecture, produces the displayed forward value.
The restoring seed is deliberately different so that retaining the fresh initialization would be
visible. It does not resume update forty-one of the old Adam process: the report's zero step count
and the model-only checkpoint boundary leave the optimizer history outside this round trip.

The supervised training paths wire the `loadCheckpoint?` and `saveCheckpoint?` fields of
`Trainer.TrainOptions` into the checkpoint format selected by the runtime arithmetic. For example:

```
-- The per-call save path records model state after the
-- requested training updates.
def saveClassifier : Trainer.TrainOptions :=
  { steps := 200
    logEvery := 25
    saveCheckpoint? := some "artifacts/classifier.state.json" }
```

The same file can be written after the fact with `trained.save` and read back with `Trainer.load`.
Loading through `loadCheckpoint?` continues training from the stored parameters; loading through
`Trainer.load` produces a result whose `steps` is zero and whose two losses are both the mean loss
of the restored parameters on the supplied data, exactly as printed above.

This is a *model-state* checkpoint, not a complete training snapshot. The eager CUDA runtime can
save Adam or AdamW moments and step counters separately with
`Checkpoint.Optimizer.save`, then restore them with
`Checkpoint.Optimizer.load`. That binary file records the optimizer kind, the
moment-defining hyperparameters, every parameter shape, and the `requiresGrad` mask. Loading rejects
a different optimizer configuration or parameter schema instead of silently attaching moments to
the wrong model. Integer metadata and float32 payloads use explicit little-endian encodings, and a
save is written to a fresh sibling file before it replaces the destination.

That optimizer file is still not a complete training snapshot. Replaying the next batch also needs
the loader or stream position; stochastic layers need generator state; and interpreting the result
needs the model, preprocessing, arithmetic semantics, backend profile, and device. A parameter-only
checkpoint remains appropriate for inference or a fresh optimizer run. Pairing it with native
optimizer state resumes more of an Adam trajectory, but only the state explicitly present in those
two files.

# Custom Training Loops

`trainer.train` owns the optimizer loop. For custom control at the same API level, open the trainer
as a session and own the loop in the calling program:

```lean (name := tfSession)
-- Own the six updates explicitly, then measure the full
-- dataset at the resulting parameters.
#eval show IO Unit from do
  let session ← (tfTrainer 2026).open
  let stream ← tfData.materialize (α := Float)
  for step in [0:6] do
    match stream.get? step with
    | some sample =>
        let loss ← session.step sample
        IO.println s!"step {step}: loss = {loss}"
    | none => pure ()
  let after ← session.eval tfData
  IO.println s!"mean loss after 6 updates = {after}"
```

```leanOutput tfSession
step 0: loss = 0.250488
step 1: loss = 0.031549
step 2: loss = 0.001341
step 3: loss = 0.031191
step 4: loss = 0.021362
step 5: loss = 0.003199
mean loss after 6 updates = 0.465379
```

The six update lines report pre-update losses on neighboring grid points. The final `eval` instead
measures all 25 samples at the parameters reached after those updates, returning `0.465379`
against an initial mean of `0.495227`. The low losses on several visited samples therefore
coexist with a much larger dataset mean. A custom loop can measure both by choosing when to
call `eval`.

`Trainer.Session` exposes:

```
-- State and measurements can be queried between updates
-- before finish freezes a result.
Session.step
Session.stepBatch
Session.steps
Session.predict
Session.loss
Session.eval
Session.state
Session.save
Session.load
Session.finish
```

Use it when the program needs a custom accumulation policy, multiple losses, generated batches,
reinforcement-learning interaction, or detailed instrumentation. Generated batches simply become
the samples passed to `step`; PINN collocation points and simulator batches fit this directly.

If `Trainer.new`, objective selection, and `trainer.train` can express the workflow, keep the
high-level API. A session is the same trainer with the loop handed over, not a second training
style: `trainer.train` itself is `open`, a loop of `step`, and `finish`.

The session does not change the model or autograd semantics. It exposes the live parameters that
`trainer.train` normally manages, and `finish` packages them as the same `Trainer.Result`.

Loading into an existing session has a different lifecycle from constructing a restored result.
`Session.load` replaces the model parameters and buffers but keeps that session's optimizer history
and completed-step counter. This is useful only when those retained values are intentional: old Adam
moments will affect the next update of the loaded parameters. Open a fresh session when the desired
experiment is a fresh optimizer run from saved weights. Saving and loading model state alone does
not decide that policy for the caller.

## Learning-rate schedules

`Trainer.Scheduler.Config` has constant, step-decay, exponential, and warmup-cosine schedules. Here
a rate of `0.1` is halved after every three step indices:

```lean (name := tfDecay)
-- Index the schedule by completed updates, including zero
-- before the first update.
def tfDecay := Trainer.Scheduler.step 0.1 3 0.5

#eval (List.range 10).map
  (Trainer.Scheduler.learningRateAt tfDecay)
```

```leanOutput tfDecay (whitespace := lax)
[0.100000, 0.100000, 0.100000, 0.050000, 0.050000, 0.050000,
 0.025000, 0.025000, 0.025000, 0.012500]
```

PyTorch's `StepLR(optimizer, step_size=3, gamma=0.5)` with initial optimizer rate `0.1` produces the
same ten numbers on the same box:

```
[0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.025, 0.025, 0.025, 0.0125]
```

The matching sequences check the schedule's indexing as well as its decay factor. The counter is
zero-indexed, which is why the first
decay occurs at index three. A step schedule with `stepSize := 0` deliberately stays at its base
rate instead of dividing by zero. The schedule is indexed by completed optimizer updates, so
`samplesPerStep` does not change its meaning.

Consequently, the rate at index three is used after three optimizer updates, whether each update
consumed one sample, four accumulated samples, or one tensor minibatch. It is not indexed by epochs
or by the number of rows read. When moving a schedule between these configurations, keep track of
how many examples each update represents; an unchanged sequence of learning rates can act at very
different points in a pass through the dataset.

The same schedule attaches to a high-level run through the `scheduler` field of
`Trainer.TrainOptions`, and to a caller-driven session through the `scheduler` argument of
`trainer.open`:

```
-- The same schedule can be attached to a managed run or a
-- caller-driven live session.
def scheduledOptions : Trainer.TrainOptions :=
  { steps := 10
    scheduler := some tfDecay }

def scheduledSession := trainer.open (scheduler := some tfDecay)
```

Attaching the schedule replaces the optimizer's configured learning rate for the whole run.
The following comparison keeps the model, data, and seed fixed:

```lean (name := tfScheduled)
-- The scheduled run replaces the constant Adam rate with
-- the entire tfDecay sequence.
#eval show IO Unit from do
  IO.println "-- constant 0.03 from optim.adam"
  let _ ← (tfTrainer 2026).train tfData { steps := 100 }
  IO.println "-- tfDecay, starting from 0.1"
  let _ ← (tfTrainer 2026).train tfData
    { steps := 100, scheduler := some tfDecay }
```

```leanOutput tfScheduled
-- constant 0.03 from optim.adam
dataset size = 25
mean_loss(before training) = 0.495227
mean_loss(after training) = 0.004741
-- tfDecay, starting from 0.1
dataset size = 25
mean_loss(before training) = 0.495227
mean_loss(after training) = 0.213333
```

The scheduled run finishes with a substantially higher loss. `tfDecay` starts at `0.1`, more than
three times the rate configured in `optim.adam`, and by index 100 it has decayed far below `0.03`.
This experiment changes the entire learning-rate sequence. The schedule is useful for exposing
the indexing convention, but these settings do not improve this run.

Transformer runs commonly warm up from a small rate and then decay toward a nonzero floor
{Informal.citep goyal2017}[]. The decay half is cosine annealing
{Informal.citep sgdr2017}[]. On a deliberately tiny configuration, with a peak of `0.001`, a floor
of `0.0001`, four warm-up updates and twelve total, the whole curve fits on one line:

```lean (name := tfWarmup)
-- Include the endpoint at index twelve to show where cosine
-- decay reaches its floor.
def tfWarmup :=
  Trainer.Scheduler.warmupCosine 0.001 0.0001 4 12

#eval (List.range 13).map
  (Trainer.Scheduler.learningRateAt tfWarmup)
```

```leanOutput tfWarmup (whitespace := lax)
[0.000250, 0.000500, 0.000750, 0.001000, 0.001000, 0.000966,
 0.000868, 0.000722, 0.000550, 0.000378, 0.000232, 0.000134,
 0.000100]
```

Index 0 is `peak / warmupSteps`, index 3 reaches the peak, and index 12 is the floor exactly.

The repeated peak at indices three and four follows from the two pieces of the function. The last
warm-up update reaches the peak; the cosine segment starts at that same peak before descending.
Requesting thirteen indices displays both the twelve-update interval and its endpoint at index
twelve. The small example makes these boundary conventions visible before the longer configuration
compresses them into a handful of printed samples. A
realistic pretraining configuration is the same function with bigger numbers, in the shape a
transformer run uses {Informal.citep transformer2017}[]:

```lean (name := tfPretraining)
-- Sample both warm-up and decay indices; six-decimal
-- printing can hide the smallest rates.
def tfPretraining :=
  Trainer.Scheduler.warmupCosine 0.0006 0.00006 2000 162761

#eval [0, 1, 1000, 2000, 100000, 162761].map
  (Trainer.Scheduler.learningRateAt tfPretraining)
```

```leanOutput tfPretraining (whitespace := lax)
[0.000000, 0.000001, 0.000300, 0.000600, 0.000239, 0.000060]
```

The first entry is not zero, it is $`3\times10^{-7}` displayed with six decimals, and the second is
$`6\times10^{-7}`. Halfway through warm-up the rate is half the peak, at update 2000 it is the peak
`0.0006`, and at the final update it is the floor `0.00006`. At and after `totalSteps`,
`learningRateAt` returns the floor exactly. The scheduler changes optimizer state only. It does not
depend on a particular model, loss, dataset, or device. If the requested warm-up is longer than the
run, TorchLean clamps it to `totalSteps`.

# Training Option Validation

An update that consumes zero samples has no meaning, so `TrainOptions.validate` refuses it before
any tape is built:

```lean +error (name := tfBadOptions)
-- Reject an update with no contributing samples before
-- constructing its training tape.
#eval show IO Unit from do
  let trained ← (tfTrainer 2026).train tfData
    { steps := 5, samplesPerStep := 0 }
  trained.printSummary
```

```leanOutput tfBadOptions
training: samplesPerStep must be positive
```

The quickstart command also requires a positive step count:

```terminal +output
$ lake exe torchlean quickstart_mlp --device cpu --steps 0 --seed 2026
error: quickstart_mlp: --steps must be > 0
```

{src "NN/API/CLI/Command.lean"}[`TorchLean.CLI.orThrow`] converts a parse failure to an `IO`
exception, adding the command name only if the message does not already include it. To inspect
initialization without updates, use a session evaluation as shown next.

# Training Experiments

## Initialization only

You do not need a zero-step run for this, because the quickstart already evaluates the untrained
model on the held-out point and prints both:

```terminal +output
untrained(heldout) = [-0.088261]
trained(heldout)   = [0.228325]
```

For the initial loss on the whole dataset, `mean_loss(before training)` in any run is exactly that
number, measured before the first update. If you want it without training at all, open a session and
evaluate the dataset before any step:

```lean (name := tfInitializationOnly)
-- Evaluate the initialized model without ever asking the
-- session to update it.
#eval show IO Unit from do
  let session ← (tfTrainer 2026).open
  IO.println s!"initial loss = {← session.eval tfData}"
```
```leanOutput tfInitializationOnly
initial loss = 0.495227
```

## Seed sensitivity

Changing the seed changes the initialized weights; the biases retain their zero initialization.
The dataset is a deterministic grid consumed in order, so this experiment isolates initialization:

```lean (name := tfSeeds)
-- The deterministic sample order stays fixed while each
-- seed changes initial weights.
#eval show IO Unit from do
  for seed in [2026, 2027, 2028] do
    IO.println s!"-- seed {seed}"
    let _ ← (tfTrainer seed).train tfData
      { steps := 100 }
```

```leanOutput tfSeeds
-- seed 2026
dataset size = 25
mean_loss(before training) = 0.495227
mean_loss(after training) = 0.004741
-- seed 2027
dataset size = 25
mean_loss(before training) = 0.607745
mean_loss(after training) = 0.064443
-- seed 2028
dataset size = 25
mean_loss(before training) = 0.371631
mean_loss(after training) = 0.005225
```

A factor of thirteen separates the best and worst final loss after the same 100 updates. The
experiment measures sensitivity to initialization; identifying which ReLU activation patterns or
parameter updates caused the spread would require inspecting those runs. Comparing optimizers
across several seeds gives a fuller account than selecting one final loss.

## Optimizer sensitivity

Keep the model, seed, dataset, and step count fixed and change only the optimizer:

```lean (name := tfOptimizers)
-- Compare SGD and Adam from the same seeded model using the
-- same number of updates.
def tfSgdTrainer : Trainer [2] [1] :=
  Trainer.new tfModel
    { objective := .meanSquaredError
      optimizer := optim.sgd { learningRate := 0.03 }
      seed := 2026 }

#eval show IO Unit from do
  IO.println "-- adam, lr 0.03"
  let _ ← (tfTrainer 2026).train tfData { steps := 100 }
  IO.println "-- sgd, lr 0.03"
  let _ ← tfSgdTrainer.train tfData { steps := 100 }
```

```leanOutput tfOptimizers
-- adam, lr 0.03
dataset size = 25
mean_loss(before training) = 0.495227
mean_loss(after training) = 0.004741
-- sgd, lr 0.03
dataset size = 25
mean_loss(before training) = 0.495227
mean_loss(after training) = 0.037862
```

The initial losses agree exactly, which is the control we wanted: the two runs start from the same
parameters and see the same samples in the same order. After 100 updates Adam is roughly eight times
lower. PyTorch, running the same recipe from its own initialization, reports the same ordering with
different magnitudes: `0.510072 -> 0.013152` for Adam against `0.510072 -> 0.049435` for SGD. Two
frameworks, two initializations, and lower final loss for Adam after this fixed number of updates.
This does not compare wall-clock speed or equally tuned optimizers.

## Backend report

Add `--show-backend`. On CPU, inspect the reference capsules. On a CUDA build, inspect which native
capsules are selected and whether any trusted external provider appears.

# Training Evidence And Limits

A successful run establishes that one configured pipeline executed:

```
-- The log records this configured execution; each arrow
-- still has its own semantic contract.
data -> forward -> loss -> reverse pass -> optimizer -> parameters
```

It can produce valuable evidence:

- loss and prediction traces;
- exact parameter artifacts;
- capsule audit rows;
- reproducible configuration;
- runtime errors or successful completion.

It does not automatically prove:

- convergence for all initializations;
- generalization to unseen data;
- robustness to an input region;
- equality of eager, typed graph, CUDA, and LibTorch paths;
- correctness of every native instruction.

The seed experiment demonstrates finite-run sensitivity, not failure of asymptotic convergence.
The six-step session illustrates the distinction between visited-sample losses and dataset loss;
it does not evaluate unseen-data generalization. Neither is a substitute for a statement with a
proof, which is
what {ref "optimization-theory"}[the optimization chapter] and
{ref "certificates"}[the certificate chapters] are for.

For a useful comparison, decide which quantity is being held fixed before interpreting a smaller
loss. The accumulation example fixes update count but changes samples consumed; the optimizer
example fixes samples and updates but changes the rule and its memory; the seed example changes the
starting parameters. Their logs answer those particular comparisons. None supplies a timing
measurement, and a displayed held-out prediction is one evaluation point rather than an estimate of
performance over a held-out population.

TorchLean gives the run enough structure for a theorem, numerical bound, backend contract, or
verification certificate to refer to the same model without erasing the boundary between them.
