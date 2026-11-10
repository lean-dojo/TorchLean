import VersoManual
import TorchLeanExamples.Models.Vision.ResNet
import TorchLeanExamples.Models.Vision.ViT
import TorchLeanExamples.Models.Operators.FNOBurgers
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.Roles
import TorchLean.NN.LoRA
import TorchLean.NN.Autograd.Complex
import TorchLean.NN.Autograd.Differential
import TorchLean.NN.Autograd.Frontend
import TorchLean.NN.Autograd.Function
import TorchLean.NN.Data.Image
import TorchLean.NN.Data.Synthetic
import TorchLean.NN.Data.CausalLM
import TorchLean.NN.Models.CNN
import TorchLean.NN.Models.CausalTransformer.LanguageModel
import TorchLean.NN.Models.Diffusion.NoisePredictor
import TorchLean.NN.Models.KAN
import TorchLean.NN.Models.Mamba
import TorchLean.NN.Models.PPO
import TorchLean.NN.Models.Recurrent
import TorchLean.NN.Models.MaskedPatchReconstructor
import TorchLean.NN.Models.UNet
import TorchLean.NN.Builder.MoE
import TorchLean.NN.Sequential.Training
import TorchLean.NN.RL.CLI
import TorchLean.NN.SelfSupervised
import TorchLean.NN.Text.GPT2BPE
import TorchLean.NN.Text.Generation
import TorchLean.NN.Text.Vocabulary
import TorchLean.NN.Loss
import TorchLean.NN.Session.Random
import TorchLean.Verify.Trainer

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean
open TorchLean Activation.Kind
open Shape (size)
open Lean.Elab.Tactic.GuardMsgs.WhitespaceMode (lax)

#doc (Manual) "Case Studies" =>
%%%
tag := "model-examples-deep-dive"
file := "Three-End-to-End-Case-Studies"
%%%

A training loss becomes easier to interpret when we know what the model predicts, how its
parameters are arranged, and which data enter the evaluation. CharGPT predicts the next character,
the vision models classify images, and the Burgers neural operator predicts an entire field. Each
task gives its reported numbers a different meaning.

I use the same approach for all three: inspect the typed computation before interpreting the
training result. Lean blocks on this page derive shapes and parameter counts from the model
definitions, while the commands produce checkpoints, logs, and predictions to inspect in a
particular run. A count can expose a missing bias, a different width, or accidental weight sharing,
though it can't establish equality of forward computations.

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

Equal logits produce that uniform distribution. In PyTorch, the corresponding calculation is:

```
# Equal logits make the target label irrelevant to this
# uniform baseline.
import torch
import torch.nn.functional as F
F.cross_entropy(torch.zeros(1, 10), torch.tensor([3])).item()
F.cross_entropy(torch.zeros(1, 65), torch.tensor([21])).item()
```

You can compare the ten-class and 65-character examples below with 2.302585 and 4.174387. A
nonuniform initial prediction can give a higher or lower loss on one label. Its average loss also
depends on the predictions and label distribution, so a deviation from $`\log V` alone doesn't
diagnose a configuration error.

Cross entropy averages a log probability, so it emphasizes confidently wrong predictions. Two
models can assign the correct class the same mean probability and still have different mean losses
if one makes a few much more confident mistakes. We can calculate the uniform baseline without
inspecting a checkpoint, which is what makes it useful here. It gives a scale for the first
reported losses while leaving the actual distribution of predictions to further evaluation.

# CharGPT

CharGPT's Tiny Shakespeare experiment begins with a text file. Its character inventory determines
the vocabulary, and the run writes both trained parameters and generated text.

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
scripts/lake.sh -R -K cuda=true exe torchlean chargpt --device cuda \
  --tiny-shakespeare --preset smoke \
  --save-checkpoint /tmp/chargpt.state \
  --log /tmp/chargpt-trainlog.json
```

With the smoke preset, the run performs two optimizer updates and writes a checkpoint and training
log. Validation uses a disjoint ten-percent suffix of the corpus. Compare its cross entropy with
$`\log 65` when the corpus has 65 distinct characters, but remember that proximity to the uniform
baseline doesn't establish that the predicted distributions are uniform.

Character-level prediction keeps tokenization visible. Each distinct corpus character is a
category, and training asks for the next category at every position of a short window. That setup
connects discrete input indices, continuous embedding vectors, and categorical logits without a
separate subword tokenizer artifact. During generation, sampled categories become later inputs. The
validation trace measures prediction on held-out windows, while the continuation shows what happens
when the model has to continue from its own sampled choices. Two updates exercise these paths
without providing evidence of language-model quality.

## Parameter Layout

Embeddings, two Transformer blocks, normalization, and the output head account for the 30,017
scalars, and the smoke preset fixes their dimensions through this configuration:

```lean (name := dpSmokeCfg)
-- Fix the same widths, depth, activation, and
-- initialization as the smoke architecture.
def dpSmoke : nn.CausalTransformer.Config :=
  { sequenceLength := 16, vocabularySize := 65
    headCount := 4, headWidth := 8
    feedForwardWidth := 128, layerCount := 2
    activation := relu
    parameterInit? := some (.normal 0.0 0.02) }
```

Model width isn't a field. It's `headCount * headWidth`: four heads of width eight give a width
of 32. Writing it this way makes an invalid split unrepresentable, so there's nothing to reject
later.

We build the indexed model from that configuration, with a batch of two windows:

```lean (name := dpGptDef)
-- Each token is a bounded vocabulary index, and each
-- window produces a row of logits per position.
def dpGpt : nn.IndexedModel (dpSmoke.tokenShape [2])
    (dpSmoke.vocabularyShape [2])
    (Fin dpSmoke.vocabularySize) :=
  nn.build 0 <|
    nn.CausalTransformer.indexed dpSmoke [2]
```

Parameters are listed in this order: token embedding, positional embedding, thirteen tensors per
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

Group by group, the parameters expand as follows:

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
projection has a bias, giving 4,128 scalars, not 4,224. That's the local architecture's
parameterization, and the count doesn't establish GPT-2 checkpoint compatibility
{Informal.citep transformer2017}[].

Those parameter groups also give a recipe for assembling corresponding PyTorch modules
{Informal.citep pytorch2019}[]:

```
# Architectural sketch of the corresponding modules.
tok = nn.Embedding(65, 32);  pos = nn.Embedding(16, 32)
block: LayerNorm(32); Linear(32,32,bias=False) x 3;
       Linear(32,32); LayerNorm(32);
       Linear(32,128); Linear(128,32)
lnf = nn.LayerNorm(32);  head = nn.Linear(32, 65)

```

Although the embedding and vocabulary head have related dimensions, they have separate storage.
Looking up one token selects a row of the `[65, 32]` embedding table, while producing logits
compares a token representation with all sixty-five output rows. Tying those matrices would remove
one independent set of weights and change the parameter count. Increasing batch size, in contrast,
adds more activation rows without creating another copy of either table. So the parameter layout is
a better checkpoint description than the number of token windows used in a run.

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

That closed form agrees with the built smoke model. Applied to the lecture configuration below, it
gives 10,788,929 trainable scalars (about 359 times as many) without allocating those tensors.

A model's stored state can hold tensors that no optimizer updates (running statistics, for
example). Dropout doesn't: its probability is configuration that `validateConfig` checks, so
turning it on leaves the stored state equal to the parameters {Informal.citep dropout2014}[]:

```lean (name := dpDropDef)
-- Enable dropout while retaining the same learned
-- projections and embeddings.
def dpDrop : nn.CausalTransformer.Config :=
  { dpSmoke with dropout? := some 0.2 }

def dpGptDrop : nn.IndexedModel (dpDrop.tokenShape [2])
    (dpDrop.vocabularyShape [2])
    (Fin dpDrop.vocabularySize) :=
  nn.build 0 <|
    nn.CausalTransformer.indexed dpDrop [2]
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
tensors   = 32
stored    = 30017
trainable = 30017
```

In each block, the dropout layers after attention and after the feed-forward network add no state
tensor, so a checkpoint of the smoke model fits the model with dropout. When stored and trainable
state differ (as they do for a model with running statistics), the command reports a separate
`non_trainable_state_scalars` count.

## The Architecture

Let $`B` be the batch size, $`T` the window length, $`V` the vocabulary size, and $`D` the model
width. A batch of token windows $`I\in\mathbb{N}^{B\times T}` contains ids smaller than $`V`, and
the indexed API represents each id with this bound. Looking up the token embedding table $`E`
produces

$$`E[I]\in\mathbb{R}^{B\times T\times D},`

To these embeddings, the model adds a learned positional table and then applies two pre-normalized
Transformer blocks. For a block input $`X`, attention mixes visible token positions and the
feed-forward branch transforms each token independently. Both branches normalize their inputs
before adding their result back to the token stream:

$$`\begin{aligned}
Z_1&=X+\operatorname{Dropout}
  \left(\operatorname{MHA}(\operatorname{LN}(X))\right),\\
Z_2&=Z_1+\operatorname{Dropout}
  \left(W_2\,\rho(W_1\operatorname{LN}(Z_1)+b_1)+b_2\right).
\end{aligned}`

A final layer normalization and linear projection produce logits

$$`\operatorname{logits}\in\mathbb{R}^{B\times T\times V}.`

Here $`\rho` is ReLU, which CharGPT selects to follow the lecture architecture while the shared
`Config` defaults to GELU. In the configuration above, the `activation := relu` field records the
choice that determines the block's nonlinear computation.

For training, the target is the same token window shifted by one position, so cross entropy at
location $`t` uses the target character $`x_{t+1}`. Token IDs stay discrete throughout the
embedding lookup. In the reusable GPT API, the output projection can also be tied to the embedding
table. This CharGPT command uses an independent output head, which is why the head contributes a
separate 2,080 weights to the table above.

We define the shared architecture in
{srcDir "TorchLean/NN/Models/CausalTransformer"}[TorchLean/NN/Models/CausalTransformer/].
CharGPT's corpus split, configuration presets, evaluation loop, generation, and checkpoint handling
are in {src "TorchLeanExamples/Models/Sequence/CharGPT.lean"}[
TorchLeanExamples/Models/Sequence/CharGPT.lean].

For the smoke batch, the input has shape `[2, 16]` and the logits have shape `[2, 16, 65]`. One
batch holds thirty-two target predictions, each with sixty-five scores. Because of the target
shift, causality is essential: if a query could read the following token from the input window, it
could inspect the very category it's being trained to predict. Shapes alone can't exclude this
shortcut, but the mask constrains which positions contribute while leaving the useful parallel
training computation over all thirty-two targets intact.

## Causal Attention

A causal mask blocks attention from query position $`i` to every future key position $`j`:

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

TorchLean's `Softmax.hardMaskedVec` sets each numerator to
`if allowed then exp (score - rowMax) else 0` and normalizes over allowed entries. A large finite
penalty can leave a positive weight on a blocked position. We compare both methods on the same row:

```lean (name := dpMask)
-- Compare exact Boolean exclusion with a finite score
-- penalty on the same row.
def maskScores : Tensor Float [4] :=
  [0.5, 1.5, -0.5, 2.0]

def maskAllow : Tensor Bool [4] :=
  [true, false, false, false]

def maskedRow : Tensor Float [4] :=
  Softmax.hardMaskedVec maskScores maskAllow

-- A large finite penalty instead of a mask.
def biasedScores : Tensor Float [4] :=
  [0.5, 1.5 - 30.0, -0.5 - 30.0, 2.0 - 30.0]

def biasedRow : Tensor Float [4] :=
  Tensor.softmax biasedScores ⟨0, Nat.one_pos⟩

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

Both rows print identically at six decimals, but the equality tests distinguish them. Whereas the
masked entry is exactly `0.0`, the penalized entry is approximately
$`e^{-28.5}\approx4.19\times10^{-13}`, and its logarithm exposes the scale hidden by decimal
rounding. A test that compared only these rendered rows would miss the difference.

Increasing a finite penalty reduces the leak, and a large enough penalty may cause floating-point
underflow. A hard mask instead sets blocked weights to zero by definition. That difference also
affects the backward pass: `Softmax.backwardFromWeights` computes
`dScores = weights ⊙ (dWeights - Σⱼ dWeightsⱼ * weightsⱼ)`. When the remaining arithmetic is
finite, a zero forward weight gives zero gradient into that logit. But a small positive weight can
carry a nonzero gradient, whose magnitude also depends on the upstream cotangent and weighted sum.

I use one visible key so the expected answer does not depend on its score. With several allowed
keys, score differences still matter: the mask restricts support, not the distribution within it.

## Changing Width, Heads, And Context

Every structural parameter in the smoke preset can be overridden:

```terminal
# Increase context and depth together while keeping width
# divisible by head count.
scripts/lake.sh -R -K cuda=true exe torchlean chargpt --device cuda \
  --tiny-shakespeare --preset smoke \
  --width 64 --heads 4 --layers 3 --seq-len 64 \
  --batch-size 8 --steps 20 --eval-every 5 --eval-iters 4
```

These settings face different constraints from the parser and from the attention computation.

`--heads` must divide `--width` so the width can be split into equal head dimensions. For example:

```terminal
# Request an invalid head split to inspect validation before
# model allocation.
scripts/lake.sh -R -K cuda=true exe torchlean chargpt --device cuda \
  --tiny-shakespeare --preset smoke --width 30 --heads 4
```

With this request, the parser rejects the configuration before tokenizing the corpus or allocating
parameters.

Attention compares every query with every key in each head. With $`H` heads of width $`D_h`,
increasing window length $`T` therefore increases attention work quadratically:

$$`\operatorname{cost}_{\mathrm{attention}}
=O(BHT^2D_h).`

Increasing width changes the projection and feed-forward matrix products, while increasing context
also enlarges the score matrices and their stored autograd state. Parameter count alone doesn't
capture this cost. From smoke to lecture settings, the count grows by a factor of about 359, while
each head's score matrix grows by a factor of $`(256/16)^2=256`.

You can also point the corpus loader at any UTF-8 text file with `--data-file PATH`, which is the
quickest way to see how vocabulary size responds to a different corpus.

The `karpathy` preset uses batch 64, context 256, width 384, six heads, six blocks, dropout 0.2,
and 5,000 AdamW updates {Informal.citep adamw2019}[]. Training its 10,788,929 parameters takes
substantially more work than the smoke run:

```terminal
# Select the larger lecture configuration, including its
# longer training budget.
scripts/lake.sh -R -K cuda=true exe torchlean chargpt --device cuda \
  --tiny-shakespeare --preset karpathy
```

TorchLean follows the architecture and hyperparameter lineage of [Karpathy's lecture
model](https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py), but the runtime and
implementation are TorchLean's. It doesn't guarantee bit-for-bit identity with the Python program
or with a pretrained GPT-2 checkpoint.

# ResNet And ViT On CIFAR

ResNet {Informal.citep resnet2016}[] and ViT {Informal.citep vit2021}[] consume the same prepared
CIFAR arrays but impose different structure on them. These examples make that difference visible
while sharing the same trainer, optimizer, loss, and runtime options.

```terminal
# Use the same prepared arrays for two separately configured
# vision runs.
python3 scripts/datasets/download_example_data.py --cifar10

scripts/lake.sh exe torchlean resnet --device cpu --n-total 1 --steps 1 \
  --log /tmp/resnet-trainlog.json

scripts/lake.sh exe torchlean vit --device cpu --n-total 1 --steps 1 \
  --log /tmp/vit-trainlog.json
```

Both examples use batches of one image, so `--n-total 1` supplies one batch. Increasing `--n-total`
supplies more one-image batches, while `--steps` controls the number of optimizer updates. The log
records the selected run's losses and configuration.

Both are ten-class models, but they organize computation differently.

## Residual Geometry

The {src "TorchLeanExamples/Models/Vision/ResNet.lean"}[`ResNet application`] crops each image to
$`3\times8\times8`, lifts it to four hidden channels, applies two shape-preserving residual blocks,
globally averages the spatial axes, and emits ten logits. Its type and size:

```lean (name := dpResnet)
-- Inspect the application’s actual crop shape and
-- residual-model state.
open Examples.ResNet in
#eval show IO Unit from do
  let shapes := nn.Sequential.stateShapes (nn.build 0 model)
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

Twelve tensors: a `3 -> 4` stem convolution with bias, then four `4 -> 4` convolutions with bias
(two convolutions in each of the two residual blocks), then the `[10, 4]` classifier and its bias.
Written by module, the same parameter arithmetic is:

```
stem  Conv2d(3, 4, 3, padding=1)   -> 112
mid   Conv2d(4, 4, 3, padding=1) x 4 -> 592
head  Linear(4, 10)                -> 50
total = 112 + 592 + 50 = 754
```

At every residual join, both branches have shape

$$`1\times4\times8\times8.`

Equal branch shapes allow elementwise addition without a projection shortcut. Model construction
enforces this requirement, but it doesn't check label correctness, numerical conditioning, or
conformance of an unproved runtime kernel.

The residual output keeps a spatial grid until global averaging reduces it to four channel
features, so a feature can respond at many locations and contribute to the same final class score.
Each residual block also gets a direct path for its input features from the shape-preserving
shortcut, while its convolutions learn a correction. In the printed output, the first three state
shapes locate the stem and the first residual convolution. Their channel dimensions explain where
the RGB input becomes the four-channel internal representation.

## Patch Geometry

The {src "TorchLeanExamples/Models/Vision/ViT.lean"}[`ViT application`] uses a convolution to create
patch embeddings, reshapes the patch grid into a token sequence, and applies two Transformer encoder
blocks. If the patch output grid is $`H'\times W'`, the token count is

$$`N=H'W'.`

This example crops to $`4\times4` and uses $`2\times2` patches with stride two, so $`H'=W'=2` and
$`N=4`. It prepends a class token and pools that token's output. So the learned positional table
needs five rows (one for each patch or class-token position):

```lean (name := dpVit)
-- Locate the patch projection, class token, and positional
-- table in the stored state.
open Examples.ViT in
#eval show IO Unit from do
  let shapes := nn.Sequential.stateShapes (nn.build 0 model)
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

In PyTorch, the corresponding patch projection is:

```
# A stride-two patch projection produces four spatial patch
# positions.
conv = nn.Conv2d(3, 4, kernel_size=2, stride=2)
conv(torch.zeros(1, 3, 4, 4)).shape
```

That convolution produces a two-by-two grid of width-four vectors. Flattening the two spatial axes
gives four patch tokens, and adding the class token accounts for the fifth positional row. The
conversion

$$`B\times D\times H'\times W'
\longrightarrow B\times N\times D`

is a layer of its own in the reusable ViT model. It moves the channel axis behind the flattened
spatial axis. An incorrect axis order can preserve the total number of entries while changing which
entries constitute a token, so checking size alone is insufficient.

The class token is learned, not an extra image patch, and its output is what the classifier reads
after attention. Each patch's location is identified by the positional table. Here the spatial grid
and feature width both have four entries, so an axis-order mistake can hide behind equal sizes.

## Initial Cross-Entropy Loss

A ten-class model's initial loss depends on the probability it assigns to each selected label.
A value above or below $`\log 10=2.302585` alone does not indicate a bug. Averaging over more
examples gives a broader measurement, although a nonuniform head need not have expected loss
$`\log 10`. To use eight samples and ten updates:

```terminal
# Increase the selected dataset rows independently of the
# optimizer update count.
scripts/lake.sh exe torchlean resnet --device cpu --n-total 8 --steps 10
scripts/lake.sh exe torchlean vit --device cpu --n-total 8 --steps 10
```

Cross entropy measures assigned probabilities, not the number of classes inferred from a loss
value. A ten-class loss of 4.6 indicates low target probability. It doesn't mean the head predicts
a hundred classes. Inspect logits, labels, and the selected examples before diagnosing the cause.

## Model Comparison

Crop size, parameter count, and computation all differ between the examples. A comparison intended
to isolate architecture needs a common data and evaluation protocol. These examples instead let us
inspect the objects each model builds:

- model summaries and parameter shapes, as printed above
- the residual join (where both branches must have identical shape) against the spatial-to-token
  conversion (where the axis order must be right)
- the backend capsules printed by adding `--show-backend`
- the JSON metadata written by `--log`.

Add `--show-backend` to inspect the selected capsules alongside the loss. Each capsule records its
provider and the evidence attached to its shape, value, and derivative claims. The portable CPU
profile selects reference operations, and CUDA numerical operations use LibTorch. A capsule records
the selected implementation and its contract, while the loss measures the model on the chosen
examples.

# Burgers Neural Operator

Here the Fourier neural operator example {Informal.citep fno2021}[] learns the terminal-time
solution map for the viscous Burgers equation

$$`\partial_t u+u\,\partial_xu=\nu\,\partial_{xx}u,
\qquad x\in[0,1],`

from sampled initial conditions. Each training pair is

$$`u_0(x_i)\longmapsto u_T(x_i),\qquad i=0,\ldots,31.`

Prepare the dataset:

```terminal
# Prepare paired initial and terminal fields on the model’s
# fixed 32-point grid.
python3 TorchLeanExamples/Data/prepare_fno1d_burgers.py \
  --download --grid 32 --ntrain 128 --ntest 32
```

A one-update CUDA run over four training fields and two held-out fields is:

```terminal
# Use the LibTorch FFT path and retain held-out
# predictions for inspection.
scripts/lake.sh -R -K cuda=true exe torchlean fno1d_burgers --device cuda \
  --steps 1 --lr 0.003 \
  --train-rows 4 --test-rows 2 --eval-rows 2 \
  --log /tmp/fno-trainlog.json \
  --plot-csv /tmp/fno-predictions.csv
```

Plot the prediction artifact with:

```terminal
# Plot the saved field predictions against their paired
# targets.
python3 TorchLeanExamples/Data/plot_fno1d_burgers.py \
  --csv /tmp/fno-predictions.csv
```

Each row in the Burgers dataset is an entire sampled field. Its target is the later field for that
initial condition, so the learned map can couple distant grid locations. That's different from
fitting a scalar function of a spatial coordinate: the model receives thirty-two initial values
together and returns thirty-two terminal values together. The reported train and test MSEs assess
these field-to-field predictions on different rows. With the prediction CSV, you can see whether an
error is localized around a steep feature or spread across the grid (information an averaged loss
removes).

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

A retained Fourier coefficient summarizes a pattern across the whole periodic grid. Changing one
learned channel map can therefore alter the output at many spatial positions after the inverse
transform. Alongside that global mixing, the pointwise branch supplies a local channel
transformation. The mode budget controls which spectral patterns the learned branch can use, but it
doesn't change the number of spatial values required by the input and output types.

## Portable Model Architecture

On CPU, the command uses the full-spectrum parameterization through the generic trainer:

```terminal
# Run the portable parameterization with the same row
# limits.
scripts/lake.sh exe torchlean fno1d_burgers --device cpu \
  --steps 1 --train-rows 4 --test-rows 2 --eval-rows 2
```

A scalar field on 32 points gains a channel axis, is lifted to eight channels by a pointwise linear
map, passes through one FNO block, and is projected back to one channel and then to a bare field.
Around the pointwise maps, the reshapes are layers of their own, as in the ViT patch conversion.

We can recover the parameter count from the model's state shapes:

```lean (name := dpFno)
-- Count stored real and imaginary spectral weights,
-- including masked frequency slices.
open Examples.FNOBurgers in
#eval show IO Unit from do
  let shapes := nn.Sequential.stateShapes (nn.build 0 model)
  IO.println s!"scalars = {dpRun shapes 0 shapes.length}"
  IO.println s!"shapes  = {shapes}"
```
```leanOutput dpFno (whitespace := lax)
scalars = 4193
shapes  = [[1, 8], [8], [32, 8, 8], [32, 8, 8], [8, 8], [8],
  [8, 1], [1]]
```

In the printed shapes, the two `[32, 8, 8]` tensors are the real and imaginary parts of
$`R_\theta`: one $`8\times8` channel map for each of the 32 frequencies. This execution path stores
real and imaginary components separately, so a complex multiply is assembled from four real matrix
products,

$$`(a+bi)(c+di)=(ac-bd)+(ad+bc)i,`

and the inverse transform recombines them. This layer expresses complex arithmetic through real
tensor operations. Its default `spectralPath := .automatic` transforms each spatial axis in turn:
CUDA execution uses LibTorch FFT operations, while CPU execution and typed graphs use dense
per-axis operations. Choosing `spectralPath := .denseReference` instead constructs the full-grid
cosine and sine matrices. Both choices use the same state shapes and learned weights.

For a grid with axis lengths $`n_1,\ldots,n_d`, the phase is

$$`-2\pi i\sum_{a=1}^{d}\frac{k_a x_a}{n_a}.`

Flattening the spatial coordinates makes a convenient tensor view, but a one-dimensional FFT of
that flattened index would compute a different transform. TorchLean's per-axis implementation
preserves the multidimensional phase, including when axis lengths differ or are odd. Its normalized
inverse divides by $`\prod_a n_a`.

## Fourier Mode Bands

In the FNO layer, the mode budget is the width of a band at each end of the frequency axis, not a
count of retained frequencies. The layer's membership test is one line, and you can run it:

```lean (name := dpBand)
-- Inspect the portable end-band predicate at half and full
-- frequency coverage.
open FNO in
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

Together, the two bands retain sixteen of the thirty-two bins. FFT arrays store negative
frequencies in the second half:

```
# Translate array positions into signed Fourier frequencies.
import numpy as np
(np.fft.fftfreq(32) * 32).astype(int)[:10]
(np.fft.fftfreq(32) * 32).astype(int)[-10:]
```

Bins 24 through 31 are frequencies $`-8` through $`-1`, so the retained set is exactly the
frequencies with $`|k|\le 8`, minus the positive $`8`. A real input has a conjugate-symmetric
spectrum, but this retained set omits positive frequency 8 while retaining -8, and the learned
complex weights are independent. The portable layer takes the real part of its inverse transform.
Keeping two bands alone doesn't establish conjugate symmetry.

At `modes=16`, the bands meet and retain the whole axis. Each frequency has its own learned channel
map, which doesn't mix distinct frequency bins. When the requested bands overlap, the
{src "TorchLean/NN/Models/FNO.lean"}[FNO configuration] uses the same rule.

## Spectral Storage And Transform Cost

After the learned channel map, the portable layer masks the transformed field, so discarded
frequency slices receive zero data-loss gradient when the intermediate arithmetic is finite. For
grid 32 and eight modes, sixteen slices are discarded, each with real and imaginary $`8\times8`
weights, giving

$$`16\times8\times8\times2=2048`

of the portable path's 4,193 stored numbers have zero data-loss gradient on finite inputs. An
optimizer with weight decay or existing momentum can still change them. Zero loss gradient isn't
the same as immutable state.

CPU and GPU use the same full-spectrum model and checkpoint layout. That full-spectrum construction
supports any number of spatial axes with a uniform representation, at the cost of unused weight
slices. For $`N=\prod_a n_a` spatial points, its full-grid dense reference takes $`O(N^2)`
transform work per channel. With dense per-axis transforms, the automatic path takes
$`O(N\sum_a n_a)`, or $`O(N\sum_a\log n_a)` with native FFTs. These costs describe the transforms,
not the learned channel maps or the whole training step.

Of the 4,193 stored scalars, the two spectral tensors account for 4,096 and the other ninety-seven
belong to the lifting, pointwise branch, and projection. So a retained-bin representation can
shrink the checkpoint without changing the external `[32] → [32]` contract.

## One Model Across Devices

On CPU and GPU, the Burgers application uses the same
{src "TorchLean/NN/Models/FNO.lean"}[`FNO constructor`] and shared trainer. Grid dimensions,
retained frequency bands, activation, initialization, and parameter layout are independent of the
selected device. There's no separate FNO-specific CUDA training loop.

TorchLean owns the tape, optimizer, and explicit gradient rules on both devices.

CPU/GPU runs share the model layout, but floating-point reduction order and rounding can still
differ. Numerical comparisons need tolerances and aligned optimizer state, while speed comparisons
need matched workloads and measurements on the target device.

# Application Coverage

Which artifact to retain depends on what we want to inspect from each run:

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
