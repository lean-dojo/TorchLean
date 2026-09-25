import VersoManual
import NN.API
import NN.Examples.Models.Vision.ResNet
import NN.Examples.Models.Vision.Vit
import NN.Examples.Models.Operators.Fno1dBurgers
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.Roles

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean
open TorchLean
open Lean.Elab.Tactic.GuardMsgs.WhitespaceMode (lax)

#doc (Manual) "Case Studies" =>
%%%
tag := "model-examples-deep-dive"
file := "Three-End-to-End-Case-Studies"
%%%

A training loss becomes easier to interpret when we know what the model predicts, how its
parameters are arranged, and which data enter the evaluation. CharGPT predicts the next character;
the vision models classify images; the Burgers neural operator predicts an entire field.
Each task gives its reported numbers a different meaning.

I use the same approach for all three: inspect the typed computation before interpreting the
training result. The Lean blocks derive shapes and parameter counts from the model definitions.
The commands produce checkpoints, logs, and predictions to inspect in a particular run. A count
can expose a missing bias, a different width, or accidental weight sharing, though it cannot
establish equality of forward computations.

# Uniform Predictions As A Loss Baseline

For a classifier over $`V` categories, cross entropy measures the negative logarithm of the
probability assigned to the target class. A uniform prediction assigns every class probability
$`1/V`, giving loss

$$`-\log\frac1V=\log V`

for every label. This is a baseline for a uniform predictor, not a guarantee about a randomly
initialized model:

```lean (name := dpLnV)
-- Compare the uniform-target loss at three vocabulary
-- sizes.
#eval show IO Unit from do
  for classes in [10, 65, 1000] do
    let value := Float.log (Nat.toFloat classes)
    IO.println s!"ln {classes} = {value}"
```
```leanOutput dpLnV
ln 10 = 2.302585
ln 65 = 4.174387
ln 1000 = 6.907755
```

Equal logits produce that uniform distribution. The corresponding PyTorch calculation is:

```
# Equal logits make the target label irrelevant to this
# uniform baseline.
import torch
import torch.nn.functional as F
F.cross_entropy(torch.zeros(1, 10), torch.tensor([3])).item()
F.cross_entropy(torch.zeros(1, 65), torch.tensor([21])).item()
```

The ten-class and 65-character examples below can be compared with 2.302585 and 4.174387.
A nonuniform initial prediction can give a higher or lower loss on one label. Its average loss
also depends on the predictions and label distribution, so a deviation from $`\log V` alone
does not diagnose a configuration error.

Cross entropy averages a log probability, so it emphasizes confidently wrong predictions.
Two models can assign the correct class the same mean probability and still have different mean
losses if one makes a few much more confident mistakes. The uniform baseline is useful here
because it can be calculated without inspecting a checkpoint. It gives a scale for the first
reported losses while leaving the actual distribution of predictions to further evaluation.

# CharGPT

The Tiny Shakespeare experiment begins with a text file. Its character inventory determines the
vocabulary, and the run writes both trained parameters and generated text.

Prepare the corpus:

```terminal
# Fetch the corpus whose character inventory determines the
# model vocabulary.
python3 scripts/datasets/download_example_data.py --tiny-shakespeare
```

Then run the two-update smoke configuration:

```terminal
# Save both the trained state and the validation trace from
# the compact preset.
lake -R -K cuda=true exe torchlean chargpt --device cuda \
  --tiny-shakespeare --preset smoke \
  --save-checkpoint /tmp/chargpt.state.json \
  --log /tmp/chargpt-trainlog.json
```

The smoke preset performs two optimizer updates and writes a checkpoint and training log.
Validation uses a disjoint ten-percent suffix of the corpus. Compare its cross entropy with
$`\log 65` when the corpus has 65 distinct characters, while remembering that proximity to the
uniform baseline does not establish that the predicted distributions are uniform.

Character-level prediction keeps tokenization visible. Each distinct corpus character is a
category, and training asks for the next category at every position of a short window. This
connects discrete input indices, continuous embedding vectors, and categorical logits without a
separate subword tokenizer artifact. During generation, sampled categories become later inputs.
The validation trace measures prediction on held-out windows; the continuation shows what happens
when the model has to continue from its own sampled choices. Two updates exercise these paths
without providing evidence of language-model quality.

## Parameter Layout

The 30,017 scalars come from embeddings, two Transformer blocks, normalization, and the output
head. The smoke preset fixes their dimensions through this configuration:

```lean (name := dpSmokeCfg)
-- Fix the same widths, depth, activation, and
-- initialization as the smoke architecture.
def dpSmoke : nn.models.CausalTransformer.Config :=
  { sequenceLength := 16, vocabularySize := 65
    headCount := 4, headWidth := 8
    feedForwardWidth := 128, layerCount := 2
    activation := .relu
    parameterInitialization? := some (.normal 0.0 0.02) }
```

Model width is not a field. It is `headCount * headWidth`, so four heads of width eight give a width
of 32. Writing it this way makes an invalid split unrepresentable rather than rejected later.

The indexed model uses that configuration with a batch of two windows:

```lean (name := dpGptDef)
-- Each token is a bounded vocabulary index; each window
-- produces a row of logits per position.
def dpGpt : nn.IndexedModel (dpSmoke.tokenShape [2])
    (dpSmoke.vocabularyShape [2])
    (Fin dpSmoke.vocabularySize) :=
  nn.build 0 <|
    nn.models.CausalTransformer.indexed dpSmoke [2]
```

The parameter list is ordered: token embedding, positional embedding, thirteen tensors per
Transformer block, the final layer normalization, and the vocabulary head. We can count each
contiguous group in that order:

```lean (name := dpCountDef)
/-- Scalars in a contiguous run of parameter shapes. -/
def dpRun (shapes : List Shape) (start len : Nat) : Nat :=
  ((shapes.drop start).take len).foldl
    (fun total s => total + Shape.size s) 0
```

```lean (name := dpCount)
-- Count contiguous parameter groups in the indexed model’s
-- declared state order.
#eval show IO Unit from do
  let shapes := dpGpt.stateShapes
  IO.println s!"tensors  = {shapes.length}"
  IO.println s!"token    = {dpRun shapes 0 1}"
  IO.println s!"position = {dpRun shapes 1 1}"
  IO.println s!"block 1  = {dpRun shapes 2 13}"
  IO.println s!"block 2  = {dpRun shapes 15 13}"
  IO.println s!"final ln = {dpRun shapes 28 2}"
  IO.println s!"head     = {dpRun shapes 30 2}"
  IO.println s!"total    = {dpRun shapes 0 shapes.length}"
```
```leanOutput dpCount
tensors  = 32
token    = 2080
position = 512
block 1  = 12608
block 2  = 12608
final ln = 64
head     = 2145
total    = 30017
```

The parameter groups expand as follows:

:::table +header
*
  * Part
  * Shapes
  * Scalars
*
  * token embedding
  * `[65, 32]`
  * 2,080
*
  * positional embedding
  * `[16, 32]`
  * 512
*
  * one block, layer norms
  * 4 tensors `[32]`
  * 128
*
  * one block, attention
  * 4 tensors `[32, 32]` plus one `[32]`
  * 4,128
*
  * one block, feed forward
  * `[128, 32]`, `[128]`, `[32, 128]`, `[32]`
  * 8,352
*
  * final layer norm
  * 2 tensors `[32]`
  * 64
*
  * vocabulary head
  * `[65, 32]`, `[65]`
  * 2,145
:::

Two blocks of 12,608 plus 2,080 plus 512 plus 64 plus 2,145 give 30,017. In attention, the query,
key, value, and output projections each contribute a square width-to-width matrix. Only the output
projection has a bias, giving 4,128 scalars rather than 4,224. This is the local architecture's
parameterization; the count does not establish GPT-2 checkpoint compatibility
{Informal.citep transformer2017}[].

The parameter groups also give a recipe for assembling corresponding PyTorch modules
{Informal.citep pytorch2019}[]:

```
# Architectural sketch of the corresponding modules.
tok = nn.Embedding(65, 32);  pos = nn.Embedding(16, 32)
block: LayerNorm(32); Linear(32,32,bias=False) x 3;
       Linear(32,32); LayerNorm(32);
       Linear(32,128); Linear(128,32)
lnf = nn.LayerNorm(32);  head = nn.Linear(32, 65)

```

Matching parameter counts can reveal a missing bias or unintended weight sharing. They cannot
exclude either when another change compensates for the count, and they do not establish that the
forward computations agree.

The embedding and vocabulary head have related dimensions but separate storage. Looking up
one token selects a row of the `[65, 32]` embedding table; producing logits compares a token
representation with all sixty-five output rows. Tying those matrices would remove one independent
set of weights and change the parameter count. Increasing batch size, in contrast, adds more
activation rows without creating another copy of either table. This is why the parameter layout
is a better checkpoint description than the number of token windows used in a run.

## Parameter Counts And Dropout State

Once the layout is known, the count is a closed form in five integers:

```lean (name := dpFormulaDef)
/-- Trainable scalars in a GPT with this shape. -/
def dpParameters
    (vocab context width feedForward layers : Nat) : Nat :=
  let perBlock :=
    4 * width * width + 2 * width * feedForward +
      6 * width + feedForward
  vocab * width + context * width + layers * perBlock +
    2 * width + width * vocab + vocab
```

```lean (name := dpFormula)
-- Evaluate the closed-form count without allocating the
-- larger model.
#eval show IO Unit from do
  IO.println s!"smoke    = {dpParameters 65 16 32 128 2}"
  IO.println s!"karpathy = {dpParameters 65 256 384 1536 6}"
```
```leanOutput dpFormula
smoke    = 30017
karpathy = 10788929
```

The formula agrees with the built smoke model. Applied to the lecture configuration below, it
gives 10,788,929 trainable scalars, about 359 times as many, without allocating those tensors.

A model's stored state is not always the same as its parameters. Turning on dropout adds state that
no optimizer updates {Informal.citep dropout2014}[]:

```lean (name := dpDropDef)
-- Enable dropout while retaining the same learned
-- projections and embeddings.
def dpDrop : nn.models.CausalTransformer.Config :=
  { dpSmoke with dropout? := some 0.2 }

def dpGptDrop : nn.IndexedModel (dpDrop.tokenShape [2])
    (dpDrop.vocabularyShape [2])
    (Fin dpDrop.vocabularySize) :=
  nn.build 0 <|
    nn.models.CausalTransformer.indexed dpDrop [2]
```

```lean (name := dpDropRun)
-- Exclude state marked non-trainable when counting
-- optimizer parameters.
#eval show IO Unit from do
  let shapes := dpGptDrop.stateShapes
  let trainable :=
    (shapes.toArray.zip dpGptDrop.requiresGrad).foldl
      (fun total entry =>
        let (s, grad) := entry
        if grad then total + Shape.size s else total) 0
  IO.println s!"tensors   = {shapes.length}"
  IO.println s!"stored    = {dpRun shapes 0 shapes.length}"
  IO.println s!"trainable = {trainable}"
```
```leanOutput dpDropRun
tensors   = 36
stored    = 30021
trainable = 30017
```

Dropout adds four scalar state tensors: one after attention and one after the feed-forward
network in each block. Their `requiresGrad` entries are false, so stored state rises to 30,021
scalars while the trainable count remains 30,017. The command reports a separate
`non_trainable_state_scalars` count when these differ. Checkpoint validation must use the expected
state layout, including these four slots.

The four additional dropout slots explain why a checkpoint can have the right number of
trainable weights and still have an incompatible state layout. Training must also preserve the
non-trainable state used by the model's execution. The filtered count above reads `requiresGrad`
from the same model that supplies `stateShapes`, so it pairs each decision with the correct
tensor. A separate hand-maintained parameter total would not catch an inserted state slot or a
change in that ordering.

## The Architecture

Let $`B` be the batch size, $`T` the window length, $`V` the vocabulary size, and $`D` the model
width. A batch of token windows $`I\in\mathbb{N}^{B\times T}` contains ids smaller than $`V`;
the indexed API represents each id with this bound. Looking up the token embedding table $`E`
produces

$$`E[I]\in\mathbb{R}^{B\times T\times D},`

The model adds a learned positional table and applies two pre-normalized Transformer blocks.
For a block input $`X`, attention mixes visible token positions and the feed-forward branch
transforms each token independently. Both branches normalize their inputs before adding their
result back to the token stream:

$$`\begin{aligned}
Z_1&=X+\operatorname{Dropout}
  \left(\operatorname{MHA}(\operatorname{LN}(X))\right),\\
Z_2&=Z_1+\operatorname{Dropout}
  \left(W_2\,\rho(W_1\operatorname{LN}(Z_1)+b_1)+b_2\right).
\end{aligned}`

The final layer normalization and linear projection produce logits

$$`\operatorname{logits}\in\mathbb{R}^{B\times T\times V}.`

Here $`\rho` is ReLU. CharGPT selects it explicitly to follow the lecture architecture, while
the shared `Config` defaults to GELU. The `activation := .relu` field in the configuration above
records the choice that determines the block's nonlinear computation.

The training target is the same token window shifted by one position. Cross entropy at location
$`t` therefore uses the target character $`x_{t+1}`. Token IDs stay discrete throughout the
embedding lookup. The reusable GPT API also supports tying the output projection to the embedding
table. This CharGPT command uses an independent output head, which is why the head contributes a
separate 2,080 weights to the table above.

The source of the shared architecture is
{src "NN/API/Models/CausalTransformer.lean"}[NN/API/Models/CausalTransformer.lean].
The corpus split, configuration presets, evaluation loop, generation, and checkpoint handling are
in
{src "NN/Examples/Models/Sequence/CharGpt.lean"}[NN/Examples/Models/Sequence/CharGpt.lean].

For the smoke batch, the input has shape `[2, 16]` and the logits have shape `[2, 16, 65]`.
There are thirty-two target predictions in one batch, each with sixty-five scores. The target
shift makes causality essential: if a query could read the following token from the input
window, it could inspect the very category it is being trained to predict. Shapes alone cannot
exclude this shortcut. The mask constrains which positions contribute while leaving the useful
parallel training computation over all thirty-two targets intact.

## Causal Attention

The causal mask blocks attention from query position $`i` to every future key position $`j`:

$$`j>i\quad\Longrightarrow\quad
\operatorname{attentionWeight}_{i,j}=0.`

With zero weight on every future position, the weighted sum cannot use those values. In PyTorch,
filling blocked scores with negative infinity gives the desired weights for this row, which has
one allowed entry:

```
# Block the three future keys and leave the first position
# available.
scores = torch.tensor([[0.5, 1.5, -0.5, 2.0]])
mask = torch.tensor([[False, True, True, True]])
torch.softmax(scores.masked_fill(mask, float('-inf')), dim=-1)
```

Position zero can attend only to itself, so its attention row is $`(1,0,0,0)` exactly, and the three
future scores contribute nothing at all.

TorchLean's `hardMaskedSoftmaxVecSpec` sets each numerator to
`if allowed then exp (score - rowMax) else 0` and normalizes over allowed entries. A large finite
penalty can leave a positive weight on a blocked position. The next calculation compares both
methods on the same row:

```lean (name := dpMask)
-- Compare exact Boolean exclusion with a finite score
-- penalty on the same row.
def maskScores : Tensor Float [4] :=
  [0.5, 1.5, -0.5, 2.0]

def maskAllow : Tensor Bool [4] :=
  [true, false, false, false]

def maskedRow : Tensor Float [4] :=
  Spec.hardMaskedSoftmaxVecSpec maskScores maskAllow

-- A large finite penalty instead of a mask.
def biasedScores : Tensor Float [4] :=
  [0.5, 1.5 - 30.0, -0.5 - 30.0, 2.0 - 30.0]

def biasedRow : Tensor Float [4] :=
  Activation.softmaxVecSpec biasedScores

#eval maskedRow
#eval biasedRow
#eval maskedRow.getScalar 3 == 0.0
#eval biasedRow.getScalar 3 == 0.0
#eval Float.log (biasedRow.getScalar 3)
```

```leanOutput dpMask
[1.000000, 0.000000, 0.000000, 0.000000]
```
```leanOutput dpMask
[1.000000, 0.000000, 0.000000, 0.000000]
```
```leanOutput dpMask
true
```
```leanOutput dpMask
false
```
```leanOutput dpMask
-28.500000
```

The two rows print identically at six decimals, but the equality tests distinguish them. The
masked entry is exactly `0.0`; the penalized entry is approximately
$`e^{-28.5}\approx4.19\times10^{-13}`. Its logarithm exposes the scale hidden by decimal
rounding. A test that compared only these rendered rows would miss the difference.

Increasing a finite penalty reduces the leak and may eventually cause floating-point underflow.
The hard mask instead encodes zero weight directly. This also affects the backward pass:
`softmaxBackwardFromWeightsSpec` computes
`dScores = weights ⊙ (dWeights - Σⱼ dWeightsⱼ * weightsⱼ)`.
When the remaining arithmetic is finite, a zero forward weight gives zero gradient into that logit.
A small positive weight can
carry a nonzero gradient, whose magnitude also depends on the upstream cotangent and weighted sum.

I use a row with one visible key because its expected answer does not depend on any score.
After normalization, the single permitted value receives all the weight. That isolates mask
semantics from whether a particular dot product happened to be large or small. The logarithm
of the penalized entry then gives information the six-decimal tensor printer cannot show. For
later rows with several allowed keys, score differences still matter within the allowed set;
the mask specifies support rather than choosing a uniform distribution over that support.

## Changing Width, Heads, And Context

Every structural parameter in the smoke preset can be overridden:

```terminal
# Increase context and depth together while keeping width
# divisible by head count.
lake -R -K cuda=true exe torchlean chargpt --device cuda \
  --tiny-shakespeare --preset smoke \
  --width 64 --heads 4 --layers 3 --seq-len 64 \
  --batch-size 8 --steps 20 --eval-every 5 --eval-iters 4
```

The parser and the attention computation impose different constraints on these settings.

`--heads` must divide `--width` so the width can be split into equal head dimensions. For example:

```terminal
# Request an invalid head split to inspect validation before
# model allocation.
lake -R -K cuda=true exe torchlean chargpt --device cuda \
  --tiny-shakespeare --preset smoke --width 30 --heads 4
```

The parser rejects this configuration before tokenizing the corpus or allocating parameters.

Attention compares every query with every key in each head. With $`H` heads of width $`D_h`,
increasing window length $`T` therefore increases attention work quadratically:

$$`\operatorname{cost}_{\mathrm{attention}}
=O(BHT^2D_h).`

Increasing width changes the projection and feed-forward matrix products; increasing context
also enlarges the score matrices and their stored autograd state. Parameter count alone does not
capture this cost. From smoke to lecture settings, the count grows by about 359 while each head's
score matrix grows by $`(256/16)^2=256`.

The corpus loader also accepts `--data-file PATH` for any UTF-8 text file, which is the quickest way
to see how vocabulary size responds to a different corpus.

The `karpathy` preset uses batch 64, context 256, width 384, six heads, six blocks, dropout 0.2,
and 5,000 AdamW updates {Informal.citep adamw2019}[]. Training its 10,788,929 parameters takes
substantially more work than the smoke run:

```terminal
# Select the larger lecture configuration, including its
# longer training budget.
lake -R -K cuda=true exe torchlean chargpt --device cuda \
  --tiny-shakespeare --preset karpathy
```

TorchLean follows the architecture and hyperparameter lineage of
[Karpathy's lecture model](https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py), but the
runtime and implementation are TorchLean's. It does not claim bit-for-bit identity with the Python
program or with a pretrained GPT-2 checkpoint.

# ResNet And ViT On CIFAR

ResNet {Informal.citep resnet2016}[] and ViT {Informal.citep vit2021}[] consume the same prepared
CIFAR arrays but impose different structure on them. The examples make that difference visible while
sharing the same trainer, optimizer, loss, and runtime options.

```terminal
# Use the same prepared arrays for two separately configured
# vision runs.
python3 scripts/datasets/download_example_data.py --cifar10

lake exe torchlean resnet --device cpu --n-total 1 --steps 1 \
  --log /tmp/resnet-trainlog.json

lake exe torchlean vit --device cpu --n-total 1 --steps 1 \
  --log /tmp/vit-trainlog.json
```

Both examples use batches of one image, so `--n-total 1` supplies one batch. Increasing
`--n-total` supplies more one-image batches; `--steps` controls the number of optimizer updates.
The log records the selected run's losses and configuration.

Both are ten-class models, but they organize computation differently.

## Residual Geometry

The {src "NN/Examples/Models/Vision/ResNet.lean"}[`ResNet application`] crops each image to
$`3\times8\times8`, lifts it to four hidden channels, applies two shape-preserving residual blocks,
globally averages the spatial axes, and emits ten logits. Its type and size:

```lean (name := dpResnet)
-- Inspect the application’s actual crop shape and
-- residual-model state.
open NN.Examples.Models.Vision.ResNet in
#eval show IO Unit from do
  let shapes := nn.stateShapes (nn.build 0 model)
  IO.println s!"{input} -> {output}"
  IO.println s!"tensors = {shapes.length}"
  IO.println s!"scalars = {dpRun shapes 0 shapes.length}"
  IO.println s!"shapes  = {shapes.take 3}"
```
```leanOutput dpResnet (whitespace := lax)
[1, 3, 8, 8] -> [1, 10]
tensors = 12
scalars = 754
shapes  = [[4, 3, 3, 3], [4], [4, 4, 3, 3]]
```

Twelve tensors: a `3 -> 4` stem convolution with bias, then four `4 -> 4` convolutions with bias,
which is two convolutions in each of the two residual blocks, then the `[10, 4]` classifier and its
bias. The same parameter arithmetic can be written by module:

```
stem  Conv2d(3, 4, 3, padding=1)   -> 112
mid   Conv2d(4, 4, 3, padding=1) x 4 -> 592
head  Linear(4, 10)                -> 50
total = 112 + 592 + 50 = 754
```

At every residual join, both branches have shape

$$`1\times4\times8\times8.`

The two branches' equal shapes allow elementwise addition without a projection shortcut. Model
construction enforces this requirement. It does not check label correctness, numerical conditioning,
or conformance of an unproved runtime kernel.

The residual output keeps a spatial grid until global averaging reduces it to four channel
features. A feature can therefore respond at many locations and contribute to the same final
class score. The shape-preserving shortcut also gives each residual block a direct path for its
input features while its convolutions learn a correction. The printed first three state shapes
locate the stem and the first residual convolution; their channel dimensions explain where the
RGB input becomes the four-channel internal representation.

## Patch Geometry

The {src "NN/Examples/Models/Vision/Vit.lean"}[`ViT application`] uses a convolution to create patch
embeddings, reshapes the patch grid into a token sequence, and applies two Transformer encoder
blocks. If the patch output grid is $`H'\times W'`, the token count is

$$`N=H'W'.`

This example crops to $`4\times4` and uses $`2\times2` patches with stride two, so
$`H'=W'=2` and $`N=4`. It appends a class token and pools that token's output. The learned
positional table therefore needs five rows, one for each patch or class-token position:

```lean (name := dpVit)
-- Locate the patch projection, class token, and positional
-- table in the stored state.
open NN.Examples.Models.Vision.Vit in
#eval show IO Unit from do
  let shapes := nn.stateShapes (nn.build 0 model)
  IO.println s!"{input} -> {output}"
  IO.println s!"tensors  = {shapes.length}"
  IO.println s!"scalars  = {dpRun shapes 0 shapes.length}"
  IO.println s!"patch    = {shapes.take 2}"
  IO.println s!"cls, pos = {(shapes.drop 2).take 2}"
```
```leanOutput dpVit (whitespace := lax)
[1, 3, 4, 4] -> [1, 10]
tensors  = 34
scalars  = 454
patch    = [[4, 3, 2, 2], [4]]
cls, pos = [[1, 4], [5, 4]]
```

The positional table is `[5, 4]`: five positions of width four, which is four image patches plus one
class token. The corresponding PyTorch patch projection is:

```
# A stride-two patch projection produces four spatial patch
# positions.
conv = nn.Conv2d(3, 4, kernel_size=2, stride=2)
conv(torch.zeros(1, 3, 4, 4)).shape
```

The convolution produces a two-by-two grid of width-four vectors. Flattening the two spatial
axes gives four patch tokens; adding the class token accounts for the fifth positional row.
The conversion

$$`B\times D\times H'\times W'
\longrightarrow B\times N\times D`

is an explicit layer in the reusable ViT model. It moves the channel axis behind the flattened
spatial axis. An incorrect axis order can preserve the total number of entries while changing
which entries constitute a token, so checking size alone is insufficient.

The class token is a learned width-four vector, not an additional image patch. It joins the
four patch representations so attention can aggregate image information into the token that
the classifier reads. The positional table distinguishes where each patch came from after the
grid has been flattened. Here the spatial grid and feature width both happen to contain four
entries, which makes axis mistakes particularly easy to hide behind equal sizes. The explicit
convolution shape and subsequent token interpretation disambiguate them.

## Initial Cross-Entropy Loss

A ten-class model's initial loss depends on the probability it assigns to each selected label.
A value above or below $`\log 10=2.302585` alone does not indicate a bug. Averaging over more
examples gives a broader measurement, although a nonuniform head need not have expected loss
$`\log 10`. To use eight samples and ten updates:

```terminal
# Increase the selected dataset rows independently of the
# optimizer update count.
lake exe torchlean resnet --device cpu --n-total 8 --steps 10
lake exe torchlean vit --device cpu --n-total 8 --steps 10
```

Cross entropy measures assigned probabilities, not the number of classes inferred from a loss
value. A ten-class loss of 4.6 indicates low target probability; it does not mean the head predicts
a hundred classes. Inspect logits, labels, and the selected examples before diagnosing the cause.

## Model Comparison

The examples differ in crop size, parameter count, and computation. A comparison intended to
isolate architecture needs a common data and evaluation protocol. These examples instead let us
inspect the objects each model builds:

- model summaries and parameter shapes, as printed above;
- the residual join, where both branches must have identical shape, against the spatial-to-token
  conversion, where the axis order must be right;
- the backend capsules printed by adding `--show-backend`;
- the JSON metadata written by `--log`.

Add `--show-backend` to inspect the selected capsules alongside the loss. Each capsule records
its provider and the evidence attached to its shape, value, and derivative claims. The portable
CPU profile selects reference operations; CUDA numerical operations use LibTorch. The capsule
records the selected implementation and its contract, while the loss measures the model on the
chosen examples.

# Burgers Neural Operator

The Fourier neural operator example {Informal.citep fno2021}[] learns the terminal-time solution map
for the viscous Burgers equation

$$`\partial_t u+u\,\partial_xu=\nu\,\partial_{xx}u,
\qquad x\in[0,1],`

from sampled initial conditions. Each training pair is

$$`u_0(x_i)\longmapsto u_T(x_i),\qquad i=0,\ldots,31.`

Prepare the dataset:

```terminal
# Prepare paired initial and terminal fields on the model’s
# fixed 32-point grid.
python3 NN/Examples/Data/prepare_fno1d_burgers.py \
  --download --grid 32 --ntrain 128 --ntest 32
```

A one-update CUDA run over four training fields and two held-out fields is:

```terminal
# Use the ATen real-FFT path and retain held-out
# predictions for inspection.
lake -R -K cuda=true exe torchlean fno1d_burgers --device cuda \
  --steps 1 --lr 0.003 \
  --train-rows 4 --test-rows 2 --eval-rows 2 \
  --log /tmp/fno-trainlog.json \
  --plot-csv /tmp/fno-predictions.csv
```

Plot the prediction artifact with:

```terminal
# Plot the saved field predictions against their paired
# targets.
python3 NN/Examples/Data/plot_fno1d_burgers.py \
  --csv /tmp/fno-predictions.csv
```

Each row in the Burgers dataset is an entire sampled field. The target is the later field
for that initial condition, so the learned map can couple distant grid locations. This differs
from fitting a scalar function of a spatial coordinate: the model receives thirty-two initial
values together and returns thirty-two terminal values together. The reported train and test
MSEs assess these field-to-field predictions on different rows. The prediction CSV can reveal
whether an error is localized around a steep feature or spread across the grid, information an
averaged loss removes.

## Spectral Layer

Let $`v\in\mathbb R^{N\times C}` be a field with $`C` latent channels. The spectral branch computes
a discrete Fourier transform, retains a configured frequency band, multiplies those modes by
learned complex weights, and transforms back:

$$`\widehat v_k
=\sum_{j=0}^{N-1}v_j e^{-2\pi i jk/N},`

$$`\widehat y_k
=R_{\theta,k}\widehat v_k
\quad\text{for retained }k,\qquad
y=\mathcal F^{-1}(\widehat y).`

A pointwise linear branch is added before the activation. The command uses grid $`N=32`, width
$`8`, a mode budget of eight, and one spectral residual block.

A retained Fourier coefficient summarizes a pattern across the whole periodic grid. Changing
one learned channel map can therefore alter the output at many spatial positions after the
inverse transform. The pointwise branch supplies a local channel transformation alongside that
global mixing. The mode budget controls which spectral patterns the learned branch can use;
it does not change the number of spatial values required by the input and output types.

## Portable Model Architecture

The CPU command uses the full-spectrum parameterization through the generic trainer:

```terminal
# Run the portable parameterization with the same row
# limits.
lake exe torchlean fno1d_burgers --device cpu \
  --steps 1 --train-rows 4 --test-rows 2 --eval-rows 2
```

A scalar field on 32 points gains a channel axis, is lifted to eight channels by a pointwise
linear map, passes through one FNO block, and is projected back to one channel and then to a bare
field. The reshape layers around the pointwise maps are explicit, as in the ViT patch conversion.

We can recover the parameter count directly from the model's state shapes:

```lean (name := dpFno)
-- Count stored real and imaginary spectral weights,
-- including masked frequency slices.
open NN.Examples.Models.Operators.Fno1dBurgers in
#eval show IO Unit from do
  let shapes := nn.stateShapes (nn.build 0 model)
  IO.println s!"scalars = {dpRun shapes 0 shapes.length}"
  IO.println s!"shapes  = {shapes}"
```
```leanOutput dpFno (whitespace := lax)
scalars = 4193
shapes  = [[1, 8], [8], [32, 8, 8], [32, 8, 8], [8, 8], [8],
  [8, 1], [1]]
```

The two `[32, 8, 8]` tensors are the real and imaginary parts of $`R_\theta`: one $`8\times8`
channel map for each of the 32 frequencies. This execution path stores real and imaginary
components separately, so a complex multiply is
assembled from four real matrix products,

$$`(a+bi)(c+di)=(ac-bd)+(ad+bc)i,`

and the inverse transform recombines them. This layer expresses complex arithmetic through real
tensor operations. Its default `spectralPath := .automatic` transforms each spatial axis in turn:
eager CUDA execution uses LibTorch FFT operations, while CPU and typed-graph execution use
dense per-axis operations.
Choosing `spectralPath := .denseReference` instead constructs the full-grid cosine and sine
matrices. The two choices use exactly the same state shapes and learned weights.

For a grid with axis lengths $`n_1,\ldots,n_d`, the phase is

$$`-2\pi i\sum_{a=1}^{d}\frac{k_a x_a}{n_a}.`

Flattening the spatial coordinates makes a convenient tensor view, but a one-dimensional FFT of
that flattened index would compute a different transform. The per-axis implementation preserves
the multidimensional phase, including when axis lengths differ or are odd. Its normalized inverse
divides by $`\prod_a n_a`.

## Fourier Mode Bands

The mode budget is the width of a band at each end of the frequency axis, not a count of retained
frequencies. The layer's membership test is one line, and it can be run:

```lean (name := dpBand)
-- Inspect the portable end-band predicate at half and full
-- frequency coverage.
open Runtime.Autograd.Model.Layers.FNO.Internal in
#eval show IO Unit from do
  let kept := (List.range 32).filter (keepCoordinate 32 8)
  IO.println s!"kept  = {kept}"
  IO.println s!"count = {kept.length} of 32"
  let wide := (List.range 32).filter (keepCoordinate 32 16)
  IO.println s!"modes=16 keeps {wide.length} of 32"
```
```leanOutput dpBand (whitespace := lax)
kept  = [0, 1, 2, 3, 4, 5, 6, 7, 24, 25, 26, 27, 28, 29, 30,
  31]
count = 16 of 32
modes=16 keeps 32 of 32
```

The two bands retain sixteen of the thirty-two bins. FFT arrays store negative frequencies in
the second half:

```
# Translate array positions into signed Fourier frequencies.
import numpy as np
(np.fft.fftfreq(32) * 32).astype(int)[:10]
(np.fft.fftfreq(32) * 32).astype(int)[-10:]
```

Bins 24 through 31 are frequencies $`-8` through $`-1`, so the retained set is exactly the
frequencies with $`|k|\le 8`, minus the positive $`8`. A real input has a conjugate-symmetric
spectrum, but this retained set omits positive frequency
8 while retaining -8, and the learned complex weights are independent. The portable layer takes the
real part of its inverse transform; keeping two bands alone does not establish conjugate symmetry
or equivalence to the CUDA real-FFT parameterization.

At `modes=16`, the bands meet and retain the whole axis. Each frequency has its own learned
channel map; this does not mix distinct frequency bins. The
{src "NN/API/Models/FNO.lean"}[FNO configuration] uses the same rule when the requested bands
overlap.

## Spectral Storage And Transform Cost

The portable layer masks the transformed field after the learned channel map. Discarded frequency
slices therefore receive zero data-loss gradient when the intermediate arithmetic is finite.
For grid 32 and eight modes, sixteen slices are discarded, each with real and imaginary
$`8\times8` weights, giving

$$`16\times8\times8\times2=2048`

of the portable path's 4,193 stored numbers have zero data-loss gradient on finite inputs.
An optimizer with weight decay or existing momentum can still change them; zero loss gradient is
not the same as immutable state.

The CUDA `Fno1dRfft` path allocates weights for the retained real-FFT bins and has a smaller
parameter vector. Its checkpoint layout therefore differs from the full-spectrum model.
The full-spectrum construction supports any number of spatial axes with a uniform representation,
at the cost of unused weight slices. For $`N=\prod_a n_a` spatial points, its full-grid dense
reference takes
$`O(N^2)` transform work per channel. The automatic path takes
$`O(N\sum_a n_a)` with dense per-axis transforms, or $`O(N\sum_a\log n_a)` with native FFTs.
These costs describe the transforms, not the learned channel maps or the whole training step.

The displayed spectral state contains two arrays of $`32\cdot8\cdot8=2048` scalars each.
Together they account for 4,096 of the 4,193 stored scalars, so spectral storage dominates this
small portable model even though only part of it contributes to the data loss. The remaining
ninety-seven scalars belong to the lifting, pointwise branch, and projection. This breakdown
explains why a smaller retained-bin representation can change checkpoint size substantially
without changing the external `[32] → [32]` contract.

## CPU And CUDA Parameterizations

The Burgers application chooses between two representations behind the same typed input and
output:

:::table +header
*
  * Property
  * `--device cpu`
  * `--device cuda`
*
  * spectral representation
  * full spectrum, with an end-band mask
  * retained real-FFT bins
*
  * execution
  * generic trainer and portable per-axis transforms
  * `Cuda.Fno1dRfft` tape runner and LibTorch operations
:::

The CUDA runner is
{src "NN/Runtime/Autograd/Engine/Cuda/Fno1dRfft.lean"}[`Fno1dRfft`]. TorchLean builds the tape,
retains the operands needed for backward, and applies its explicit gradient rules. The spectral
buffer operation calls LibTorch's real FFT, complex matrix multiplication, and inverse real FFT;
its backward operation also uses ATen computations. These calls do not construct a LibTorch
autograd graph. The application reports this choice as `spectral path=ATen RFFT tape op`.

The spectral parameterization and numerical provider both affect a comparison. Equal input and
output types do not identify the learned weights or retained frequency set. The reusable
{src "NN/API/Models/FNO.lean"}[`FNO constructor`] states grid and mode constraints independently
of the backend, and the
{src "NN/Examples/Models/Operators/Fno1dBurgers.lean"}[`Burgers application`] selects its runner.
The native numerical behavior remains part of the LibTorch execution boundary.

A controlled numerical comparison needs a correspondence between spectral parameter sets,
aligned retained frequencies, the same initial field, and matching optimizer state. A speed
comparison additionally needs matched workloads and measurements on the target device. The
commands above provide entry points for those experiments; choosing CUDA alone establishes
neither numerical agreement nor a speedup.

# Application Coverage

The artifact to retain depends on what we want to inspect from each run:

:::table +header
*
  * Application
  * Structural pressure
  * External boundary
  * Primary artifact
*
  * CharGPT
  * causal windows, heads, depth, token IDs
  * LibTorch execution and corpus file
  * checkpoint, validation log, generated text
*
  * ResNet / ViT
  * residual joins or patch-token layout
  * CIFAR arrays and selected runtime
  * classification loss log
*
  * FNO
  * field shape, retained Fourier modes
  * dataset preparation and LibTorch FFT operations
  * train/test loss and prediction CSV
:::

For generative models, the schedule and sampler also determine what happens after training.
For reinforcement learning, the environment and recorded rollout determine which observations
the learner receives. A model configuration alone cannot describe either complete experiment.

Checkpoint and dataset files produced by these commands are runtime artifacts, not proof objects.
Loading one checks its declared schema and dimensions where the command implements those checks.
It does not establish provenance, reproduce the optimizer history, or prove that two files with
the same shape encode the same model or dataset.
