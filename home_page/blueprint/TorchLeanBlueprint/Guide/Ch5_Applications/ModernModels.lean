import VersoManual
import NN.API
import NN.Spec.Layers.Attention
import NN.Proofs.Models.Attention.CausalMask
import NN.MLTheory.Proofs.StateSpace.MambaCausality
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.Roles

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean
open TorchLean
open Lean.Elab.Tactic.GuardMsgs.WhitespaceMode (lax)

#doc (Manual) "Modern Models" =>
%%%
tag := "modern-models"
%%%

A batch of eight examples should share one set of model parameters. A residual addition needs
both branches to produce the same shape. These are small requirements, but they are good places
to start reading a model: we can inspect the parameter list or try to construct mismatched
branches and see exactly what the API requires.

I will keep that level of detail as the models become more involved. For a vision Transformer,
we can follow the pixels that survive patch extraction; for causal attention and a recurrent
scan, we can append tokens and inspect the earlier outputs. Shapes identify the objects in these
calculations, while the mask and scan definitions determine which inputs can affect them.

The Lean blocks compute shapes from the model configurations. The accompanying PyTorch
transcripts show shapes obtained by executing the corresponding operations
{Informal.citep pytorch2019}[].

The reusable constructors live under {srcDir "NN/API/Models"}[`NN/API/Models`] and the runnable
applications under {srcDir "NN/Examples/Models"}[`NN/Examples/Models`]. A constructor such as
`nn.models.resnet` is reusable and rank-polymorphic; the `resnet` command chooses a deliberately
small CIFAR configuration so that a reader can run the whole data and training path locally.

Use the focused public imports when a file does not need the whole API:

```
-- Import the causal model constructor and the layers used
-- to compose it.
import NN.API.Models.CausalTransformer
import NN.API.Neural.Layers
```

# Batching And Parameter Sharing

Consider an ordinary one-hidden-layer network

$$`f_\theta(x)=W_2\,\rho(W_1x+b_1)+b_2.`

For a batch of $`B` vectors the shapes are

$$`X\in\mathbb{R}^{B\times d_{\mathrm{in}}},\qquad
W_1\in\mathbb{R}^{d_h\times d_{\mathrm{in}}},\qquad
W_2\in\mathbb{R}^{d_{\mathrm{out}}\times d_h}.`

In TorchLean the per-sample model and the batch axis are written separately. The per-sample model
maps `[3]` to `[2]`, and `nn.mapLeading` applies it at every index of a new leading axis, sharing
one parameter set.

The distinction becomes useful when changing batch size. Eight examples require eight sets of
activations, but there is still only one classifier to update. In this example the first affine
map contributes fifteen weights and five biases; the second contributes ten weights and two
biases. Those thirty-two scalars are shared by every row. The printed shapes let us account for
them without inspecting an optimizer or relying on a naming convention.

```lean (name := mmBatchedShapes)
-- Build one parameter set, then reuse it for all eight
-- input rows.
def mmPerSample : nn.Builder (nn.Sequential [3] [2]) :=
  nn.Sequential![
    nn.linear 3 5,
    nn.relu,
    nn.linear 5 2
  ]

def mmBatched :
    nn.Builder (nn.Sequential [8, 3] [8, 2]) := do
  pure (nn.mapLeading [8] (← mmPerSample))

#eval (nn.stateShapes (nn.build 1 mmBatched)).map
  Shape.toList
```
```leanOutput mmBatchedShapes
[[5, 3], [5], [2, 5], [2]]
```

The parameter list is the per-sample list: adding a batch axis added no parameters. Running the
batched model on eight copies of the same row gives eight copies of the same output row, which is
what parameter sharing means operationally.

Evaluation mode also makes this a controlled comparison: the model contains affine maps and a
ReLU, with no operation that depends on other batch members. Equal rows therefore take equal
paths through the same parameters. The values themselves are determined by the seeded initial
model; they are predictions before training, not class probabilities or evidence that a task has
been learned.

```lean (name := mmBatchedRun)
-- Identical rows in evaluation mode isolate parameter
-- sharing from input variation.
#eval do
  let model := nn.build 1 mmBatched
  let m ← nn.Module.instantiate model { device := .cpu }
  m.eval
  let out ← m.forward (Tensor.full [8, 3] 1.0)
  IO.println s!"{out}"
```
```leanOutput mmBatchedRun (whitespace := lax)
[[0.428865, 0.706478],
 [0.428865, 0.706478],
 [0.428865, 0.706478],
 [0.428865, 0.706478],
 [0.428865, 0.706478],
 [0.428865, 0.706478],
 [0.428865, 0.706478],
 [0.428865, 0.706478]]
```

PyTorch writes the same model without mentioning the batch axis at all, because `nn.Linear` maps
over any number of leading dimensions:

```
# Inspect the shared weights, then compare all output rows
# with the first.
mlp = nn.Sequential(nn.Linear(3, 5), nn.ReLU(), nn.Linear(5, 2))
print([tuple(p.shape) for p in mlp.parameters()])
out = mlp(torch.ones(8, 3))
print(tuple(out.shape), bool(torch.all(out == out[0])))
```

```
[(5, 3), (5,), (2, 5), (2,)]
(8, 2) True
```

Both versions share parameters across the batch. Because `[8, 3]` appears in the TorchLean model's
type, a theorem or an exported graph
about this model states which batch shape it is about, and an operation that is not batch-invariant
cannot hide behind an implicit leading axis; {ref "torchlean_vs_pytorch"}[TorchLean and PyTorch]
shows a real batched-versus-sliced discrepancy of that kind. The cost is that the batch axis has to
be named, and `nn.mapLeading` is where you name it.

The batch model fixes how these affine maps and activations compose. Adding a residual branch
introduces another requirement: the two paths must agree on the shape at their join.

# Residual Networks

A residual block computes

$$`y=F_\theta(x)+S_\theta(x),`

where $`S` is either the identity or a projection {Informal.citep resnet2016}[]. The addition is
defined only when the two branches produce the same shape, so `nn.addBranches` requires exactly
that, and the parameter list of the block is the concatenation of the two branch lists:

```lean (name := mmBranches)
-- Both learned branches map the same 32 features to eight
-- output coordinates.
def mmBranches : nn.Builder (nn.Sequential [32] [8]) := do
  let wide ← nn.Sequential![nn.linear 32 8, nn.relu]
  let shortcut ← nn.linear 32 8
  pure (nn.addBranches wide shortcut)

#eval (nn.stateShapes (nn.build 3 mmBranches)).map
  Shape.toList
```
```leanOutput mmBranches
[[8, 32], [8], [8, 32], [8]]
```

Give the shortcut the wrong width and there is no model to run:

```lean +error (name := mmClash)
-- This deliberate mismatch asks the shortcut for six
-- coordinates instead of eight.
def mmClash : nn.Builder (nn.Sequential [32] [8]) := do
  let wide ← nn.Sequential![nn.linear 32 8, nn.relu]
  let shortcut ← nn.linear 32 6
  pure (nn.addBranches wide shortcut)
```
```leanOutput mmClash (whitespace := lax)
Application type mismatch: The argument
  shortcut
has type
  nn.Sequential (Shape.appendDim [] 32) (Shape.appendDim [] 6)
but is expected to have type
  nn.Sequential [32] [8]
in the application
  nn.addBranches wide shortcut
```

With the identity as the shortcut and a single ReLU as the branch, the block has no parameters. It
computes $`x+\rho(x)`, so a negative coordinate
passes through unchanged and a positive one is doubled:

```lean (name := mmResidual)
-- With no learned weights, each coordinate exposes the
-- residual addition directly.
def mmRes : nn.Builder (nn.Sequential [2] [2]) := do
  pure (nn.residual (← nn.Sequential![nn.relu]))

#eval do
  let m ← nn.Module.instantiate (nn.build 0 mmRes)
    { device := .cpu }
  m.eval
  let out ← m.forward [-1.0, 2.0]
  IO.println s!"x + relu x = {out}"
```
```leanOutput mmResidual
x + relu x = [-1.000000, 4.000000]
```

```
# Use the same two coordinates to expose the sign-dependent
# residual behavior.
x = torch.tensor([-1.0, 2.0])
print((x + torch.relu(x)).tolist())
```

```
[-1.0, 4.0]
```

The two residual examples separate two choices that are easy to conflate. `mmBranches` learns a
new representation on both paths because its input and output widths differ. `mmRes` keeps the
input representation and adds a correction of the same width. The latter makes the identity path
visible in the negative coordinate: even when ReLU contributes zero, the original value survives.
For a learned residual branch the same addition permits a small correction without requiring that
branch to reproduce the whole input.

## ResNet Configuration

In {src "NN/API/Models/ResNet.lean"}[`NN.API.Models.ResNet`] the configuration is indexed by the
number $`d` of spatial axes:

```
-- Spatial rank controls the length of the grid and
-- kernel-radius vectors.
structure ResNet.Config (d : Nat) where
  inputChannels  : Nat
  spatial        : Tensor Nat [d]
  hiddenChannels : Nat
  kernelRadius   : Tensor Nat [d] := Tensor.ones [d]
  classCount     : Nat
```

Grid extents are ordinary values such as `[32, 32]`; no separate nonzero proof is exposed in the
configuration, and model validation rejects a zero extent before execution. The configuration
derives the three shapes a caller cares about, for any leading shape $`L`:

$$`\operatorname{input}
=L\mathbin{+\!+}(C_{\mathrm{in}},n_1,\ldots,n_d),\qquad
\operatorname{hidden}
=L\mathbin{+\!+}(C_{\mathrm{hidden}},n_1,\ldots,n_d),`

$$`\operatorname{output}=L\mathbin{+\!+}(C_{\mathrm{class}}).`

For CIFAR-shaped inputs with a batch of eight:

```lean (name := mmResNetShapes)
-- Preserve the image grid while replacing RGB channels with
-- learned features.
abbrev mmResNet : nn.models.ResNet.Config 2 :=
  { inputChannels := 3
    spatial := [32, 32]
    hiddenChannels := 16
    kernelRadius := [1, 1]
    classCount := 10 }

#eval (mmResNet.input [8], mmResNet.hidden [8],
  mmResNet.output [8])
```
```leanOutput mmResNetShapes (whitespace := lax)
([8, 3, 32, 32], [8, 16, 32, 32], [8, 10])
```

The hidden shape keeps the input grid. A `kernelRadius` of `[1, 1]` selects a same-padding
$`3\times3` convolution, and the
geometry helper comes with the preservation theorem the constructor uses internally:

```lean (name := mmSamePadding)
-- The equality quantifies over every input grid and kernel
-- radius.
#check @nn.Convolution.Geometry.output_samePadding
```
```leanOutput mmSamePadding (whitespace := lax)
@nn.Convolution.Geometry.output_samePadding :
  ∀ {d : ℕ} (input radius : Tensor ℕ [d]),
    (nn.Convolution.Geometry.samePadding radius).output
        input =
      input
```

That theorem is what lets the residual trunk typecheck without a hand-written shape proof at every
block, because `geometry.output config.spatial` and `config.spatial` are provably the same shape.
The PyTorch example obtains the same three shapes by executing the layers:

```
# Pool spatial positions only; the channel dimension becomes
# the classifier input.
conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
images = torch.zeros(8, 3, 32, 32)
print(tuple(conv(images).shape))
pooled = F.adaptive_avg_pool2d(conv(images), 1).flatten(1)
print(tuple(pooled.shape), tuple(nn.Linear(16, 10)(pooled).shape))
```

```
(8, 16, 32, 32)
(8, 16) (8, 10)
```

The constructor uses a convolutional stem, two residual blocks, global average pooling over all
$`d` spatial axes, and a linear classifier. The CIFAR example instantiates $`d=2` and $`L=(B)`; the
model API itself is not tied to images, to two dimensions, or to one batch axis.

The pooling step explains why the final shape has no spatial axes. Each hidden channel is
averaged across the image, producing sixteen features per sample, and the final affine map turns
those features into ten class scores. Same-padding preserves coordinates inside the trunk;
pooling deliberately discards their locations at the classifier boundary. These are different
operations with different purposes, even though both are summarized by a short shape expression.

## Residual Branch Shapes

Open {src "NN/Examples/Models/Vision/ResNet.lean"}[`NN/Examples/Models/Vision/ResNet.lean`] and
inspect `config`. The command uses an $`8\times8` crop and four hidden channels. Changing the hidden
width changes both residual branches and the classifier input at once, because all three come from
the same configuration. When constructing branches separately, we must preserve that agreement:
the `mmClash` block above fails to elaborate because its two output widths differ.

# Vision Transformers

A vision Transformer first turns a spatial field into a token sequence
{Informal.citep vit2021}[]. For an input with spatial extent $`n_1\times\cdots\times n_d`, patch
kernel $`k`, stride $`s`, and padding $`p`, each output extent is the usual convolution expression

$$`n'_i
=\left\lfloor\frac{n_i+2p_i-k_i}{s_i}\right\rfloor+1.`

If the patch convolution emits $`D` channels then the patch grid becomes

$$`B\times D\times n'_1\times\cdots\times n'_d
\;\longrightarrow\;
B\times N\times D,\qquad
N=\prod_i n'_i.`

{src "NN/API/Models/Vit.lean"}[`NN.API.Models.Vit`] defines that conversion as `patchesToTokens`:
the implementation reshapes the patch grid and moves the channel axis to the end, so the following
Transformer block receives the conventional `batch × sequence × embedding` layout. Every stage is a
derived field of the configuration, so we can print the whole pipeline. Take $`32\times32` inputs
with $`4\times4` non-overlapping patches, sixty-four embedding channels, four heads of width
sixteen, two blocks, and a learned class slot:

```lean (name := mmVitConfig)
-- Four heads of width sixteen match the 64-channel patch
-- embedding.
abbrev mmVit : nn.models.ViT.Config 2 :=
  { inputChannels := 3
    spatial := [32, 32]
    patchEmbedding :=
      { outChannels := 64
        kernelSize := [4, 4]
        stride := [4, 4]
        padding := [0, 0] }
    headCount := 4
    headWidth := 16
    feedForwardWidth := 128
    layerCount := 2
    pooling := .cls
    classCount := 10 }

#eval (mmVit.encoder.grid.to Shape,
  mmVit.encoder.patchCount, mmVit.encoder.sequenceLength)
```
```leanOutput mmVitConfig
([8, 8], 64, 65)
```

A $`32\times32` image with stride-four patches has an $`8\times8` patch grid, hence sixty-four
tokens, and the class slot makes the encoded sequence length sixty-five. The three intermediate
tensor shapes follow, for a batch of eight:

```lean (name := mmVitStages)
-- Track where spatial axes become a token axis and the
-- class token is inserted.
#eval (mmVit.encoder.patches [8], mmVit.encoder.tokens [8],
  mmVit.encoder.encoded [8])
```
```leanOutput mmVitStages (whitespace := lax)
([8, 64, 8, 8], [8, 64, 64], [8, 65, 64])
```

```lean (name := mmVitEnds)
-- Classification removes the sequence axis and returns ten
-- logits per image.
#eval (mmVit.input [8], mmVit.output [8])
```
```leanOutput mmVitEnds
([8, 3, 32, 32], [8, 10])
```

In `[8, 64, 64]`, the two axes of length sixty-four mean different things: one counts patches and
one counts features within a patch token. Their equal sizes happen to hide a possible transpose
mistake. The preceding `[8, 64, 8, 8]` and following `[8, 65, 64]` disambiguate them: adding the
class slot changes the sequence length, while the embedding width stays fixed. The endpoint
`[8, 10]` then says that the classifier produces one score vector per image, rather than one per
patch.

The corresponding PyTorch operations expose the reshape and axis permutation:

```
# Flatten only the patch grid, then move channels behind the
# sequence axis.
patch = nn.Conv2d(3, 64, kernel_size=4, stride=4)
grid = patch(torch.zeros(8, 3, 32, 32))
print(tuple(grid.shape))
tokens = grid.flatten(2).transpose(1, 2)
print(tuple(tokens.shape))
cls = torch.zeros(8, 1, 64)
print(tuple(torch.cat([cls, tokens], dim=1).shape))
```

```
(8, 64, 8, 8)
(8, 64, 64)
(8, 65, 64)
```

The intermediate shapes agree at every stage. `flatten(2).transpose(1, 2)` computes the token tensor
at runtime, while `mmVit.encoder.tokens [8]` computes its shape without running the model.
Computing that shape does not run validation; the model constructor separately calls
`ViT.EncoderConfig.validate` to reject an unusable patch grid.

## Patch Coverage

The floor in the extent formula determines how many complete patches fit. Keep the configuration
above and change the patches from $`4\times4` to $`5\times5`:

```lean (name := mmVitOdd)
-- Change patch geometry alone so the loss of border
-- coverage is visible.
abbrev mmVitOdd : nn.models.ViT.Config 2 :=
  { mmVit with
    patchEmbedding :=
      { outChannels := 64
        kernelSize := [5, 5]
        stride := [5, 5]
        padding := [0, 0] } }

#eval (mmVitOdd.encoder.grid.to Shape,
  mmVitOdd.encoder.patchCount,
  mmVitOdd.encoder.sequenceLength)
```
```leanOutput mmVitOdd
([6, 6], 36, 37)
```

Six patches per axis, thirty-six tokens, thirty-seven with the class slot. Six patches of width five
cover thirty of the thirty-two pixels, so the last two rows and the last two columns of every image
are not read by the model at all. The PyTorch convolution produces the same smaller grid:

```
# A stride-five convolution emits only complete patches from
# the 32-pixel grid.
patch = nn.Conv2d(3, 64, kernel_size=5, stride=5)
print(tuple(patch(torch.zeros(8, 3, 32, 32)).shape))
```

```
(8, 64, 6, 6)
```

Both results follow the standard convolution convention. Printing the derived grid before loading
images reveals that this configuration discards a border. The types remain consistent: the
Transformer receives exactly the number of tokens that the patch convolution produces. Whether
discarding those pixels is acceptable is a separate modeling decision.

This patch experiment changes the evidence available to the classifier before any attention
weight is computed. Increasing depth later cannot recover the pixels omitted at the boundary.
Conversely, choosing a smaller stride can preserve more coverage while producing more tokens.
The grid calculation is therefore a modeling check as well as a type-level calculation: it tells
us which observations survive preprocessing and how many positions the sequence model receives.

For $`H` attention heads of width $`D_h` the model width is

$$`D=H D_h,`

and scaled dot-product attention is

$$`\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{D_h}}+M\right)V.`

The reusable constructor accepts any number of encoder blocks. A learned positional table is added
before the stack, and the classifier can use either mean token pooling or a learned class slot. The
small CIFAR application in {src "NN/Examples/Models/Vision/Vit.lean"}[`Vision/Vit.lean`] selects a
$`4\times4` crop, $`2\times2` patches, and two blocks; those choices keep the example quick without
changing the architecture that the public API represents.

# Recurrence, Attention, And Causality

Sequence models add state or a causal dependency. Write $`h_{t-1}` for the state before input
$`x_t` and $`h_t` for the updated state. A vanilla recurrent network then computes

$$`h_t=\phi(W_xx_t+W_hh_{t-1}+b),\qquad
y_t=W_yh_t+b_y.`

An LSTM replaces the single update by input, forget, output, and candidate gates:

$$`\begin{aligned}
i_t&=\sigma(W_ix_t+U_ih_{t-1}+b_i),\\
f_t&=\sigma(W_fx_t+U_fh_{t-1}+b_f),\\
o_t&=\sigma(W_ox_t+U_oh_{t-1}+b_o),\\
\tilde c_t&=\tanh(W_cx_t+U_ch_{t-1}+b_c),\\
c_t&=f_t\odot c_{t-1}+i_t\odot\tilde c_t,\\
h_t&=o_t\odot\tanh(c_t).
\end{aligned}`

In the cell update, $`f_t` scales the old cell state and $`i_t` scales the candidate
$`\tilde c_t`. The output gate $`o_t` controls how much of the transformed cell state appears in
$`h_t`. These elementwise products require each gate to have the same width as the state it acts
on.

The hidden and cell states are explicit tensors whose dimensions have to stay stable across the
unrolled sequence. The runnable `rnn`, `lstm`, and `lstm_regression` commands use that typed state
path. There is currently no GRU training subcommand: the LiRPA verifier has a GRU-gate certificate
fixture, but that is a different artifact and should not be presented as a trainable GRU model.

An explicit state gives a recurrent model a streaming interface: a caller can supply the state
left by a previous chunk and continue the computation. Restarting from zero instead describes a
new sequence. For an LSTM there are two pieces to retain, the hidden state and the cell state;
keeping only the hidden output loses information needed by the next update. Their matching widths
make the gate products legal, but correct streaming also requires carrying the actual values.

## Causal Masks

Transformers remove recurrent state and introduce a mask instead
{Informal.citep transformer2017}[]. A causal language model factors

$$`p_\theta(x_0,\ldots,x_T)
=\prod_{t=0}^{T}p_\theta(x_t\mid x_0,\ldots,x_{t-1}),`

so attention row $`i` must assign exactly zero probability to every key $`j>i`. TorchLean's mask is
an ordinary boolean tensor. A `true` entry permits the corresponding query–key pair:

```lean (name := mmMask)
-- Row i permits keys through position i, including its own
-- diagonal entry.
#eval Spec.causalMask 4
```
```leanOutput mmMask (whitespace := lax)
[[true, false, false, false],
 [true, true, false, false],
 [true, true, true, false],
 [true, true, true, true]]
```

Attention with that mask is an ordinary function of three matrices, so we can run it. Take three
two-dimensional queries and keys, three values, and compare the masked result against the unmasked
one:

```lean (name := mmAttention)
-- Keep Q, K, and V fixed so only the visibility rule
-- changes between runs.
def mmQ : Tensor Float [3, 2] :=
  [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
def mmV : Tensor Float [3, 2] :=
  [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]

def mmCausal : Spec.AttentionContext Float 3 3 2
    (by decide) (by decide) :=
  { Q := mmQ, K := mmQ, V := mmV
    mask := some (Spec.causalMask 3) }

def mmFull : Spec.AttentionContext Float 3 3 2
    (by decide) (by decide) :=
  { Q := mmQ, K := mmQ, V := mmV, mask := none }

#eval Spec.scaledDotProductAttention mmCausal
#eval Spec.scaledDotProductAttention mmFull
```
```leanOutput mmAttention (whitespace := lax)
[[1.000000, 2.000000], [2.339523, 3.339523],
 [3.510470, 4.510470]]
```
```leanOutput mmAttention (whitespace := lax)
[[3.000000, 4.000000], [3.406673, 4.406673],
 [3.510470, 4.510470]]
```

The first masked row is exactly the first value row, because position zero may attend only to
itself. The last row agrees between the two runs, because the last query may attend everywhere in
either case. PyTorch, on the same inputs in double precision:

```
# Use binary64 to compare the same causal and unrestricted
# weighted sums.
q = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float64)
v = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float64)
print(F.scaled_dot_product_attention(q, q, v, is_causal=True).tolist())
print(F.scaled_dot_product_attention(q, q, v).tolist())
```

```
[[1.0, 2.0], [2.3395230986533138, 3.3395230986533138],
 [3.5104695304536615, 4.510469530453662]]
[[3.0, 4.0], [3.4066725560787154, 4.406672556078716],
 [3.5104695304536615, 4.510469530453662]]
```

Rounded to the six decimal places shown in the Lean output, the PyTorch values agree.

These small matrices make a weighted average interpretable. Every value row has second coordinate
one greater than its first, so any normalized attention row has that same difference. The middle
causal output lies between the first two value rows because those are its only permitted keys.
The unrestricted middle row can also use the third value, which explains its larger first
coordinate. The two transcripts preserve both relationships. Both
computations include the scale factor $`1/\sqrt{D_h}`. These causal rows all contain a visible key;
they do not exercise the separate convention that a fully blocked row returns zeros.

Causality requires that extending the sequence must
not change the outputs already produced. Adding a fourth token to the same three leaves the first
three rows untouched.

```lean (name := mmPrefix)
-- Append one query, key, and value without modifying the
-- original prefix.
def mmQ4 : Tensor Float [4, 2] :=
  [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 2.0]]
def mmV4 : Tensor Float [4, 2] :=
  [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]

def mmCausal4 : Spec.AttentionContext Float 4 4 2
    (by decide) (by decide) :=
  { Q := mmQ4, K := mmQ4, V := mmV4
    mask := some (Spec.causalMask 4) }

#eval Spec.scaledDotProductAttention mmCausal4
```
```leanOutput mmPrefix (whitespace := lax)
[[1.000000, 2.000000], [2.339523, 3.339523],
 [3.510470, 4.510470], [5.837654, 6.837654]]
```

The fourth output has a different role from the first three. It is newly computed from four
visible values; it is not expected to match any row in the shorter run. The unchanged prefix is
the property needed when generating tokens sequentially: making a later token available must not
retroactively alter a prediction whose context ended earlier. The displayed example makes that
property inspectable for these inputs.

## Permuting Keys, Values, And Queries

Attention has a symmetry that explains why every Transformer in this chapter adds a positional table
before its stack. Permute the keys and values by the same permutation, leave the queries alone, and
an unmasked attention row is unchanged, because the row is a sum over key positions and real
addition is invariant under permutation. Floating-point reduction can differ in its
last bits when that order changes. Reversing our three keys and values:

```lean (name := mmPermute)
-- Move keys and their associated values together while
-- leaving queries fixed.
def mmKRev : Tensor Float [3, 2] :=
  [[1.0, 1.0], [0.0, 1.0], [1.0, 0.0]]
def mmVRev : Tensor Float [3, 2] :=
  [[5.0, 6.0], [3.0, 4.0], [1.0, 2.0]]

def mmFullRev : Spec.AttentionContext Float 3 3 2
    (by decide) (by decide) :=
  { Q := mmQ, K := mmKRev, V := mmVRev, mask := none }

def mmCausalRev : Spec.AttentionContext Float 3 3 2
    (by decide) (by decide) :=
  { Q := mmQ, K := mmKRev, V := mmVRev
    mask := some (Spec.causalMask 3) }

#eval Spec.scaledDotProductAttention mmFullRev
#eval Spec.scaledDotProductAttention mmCausalRev
```
```leanOutput mmPermute (whitespace := lax)
[[3.000000, 4.000000], [3.406673, 4.406673],
 [3.510470, 4.510470]]
```
```leanOutput mmPermute (whitespace := lax)
[[5.000000, 6.000000], [4.000000, 5.000000],
 [3.510470, 4.510470]]
```

The first output agrees with the previous unmasked result to the six decimals shown. The queries
stayed fixed while the keys and their associated values moved together. For each query, this
permutes the attention scores and the values they weight by the same permutation, leaving their
weighted sum unchanged over real arithmetic.

Permuting the input to self-attention is a different operation: it permutes queries as well as
keys and values. Without a mask or positional information, the output rows then follow the same
permutation. This is permutation equivariance, proved for the real-valued spec by
`selfAttention_reindexOuter` in
{src "NN/Proofs/Models/Attention/PermutationEquivariance.lean"}[`PermutationEquivariance.lean`].
The learned positional tables in the GPT and ViT constructors give otherwise identical tokens
different inputs at different positions
{Informal.citep transformer2017}[]{Informal.citep vit2021}[].

The second output changes because the causal mask stays tied to array positions while the keys
and values move. The first query can still see only slot zero, but that slot now holds the value
that used to be last. A causal mask supplies an order-dependent visibility rule; positional
embeddings supply position-dependent features. Both can make a token's position affect its output.

The double-precision PyTorch output also shows what happens in the last displayed digits:

```
# Compare unrestricted permutation symmetry with the
# position-dependent causal mask.
q = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float64)
k = torch.tensor([[1.0, 1.0], [0.0, 1.0], [1.0, 0.0]], dtype=torch.float64)
v = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float64)
vp = torch.tensor([[5.0, 6.0], [3.0, 4.0], [1.0, 2.0]], dtype=torch.float64)
print(F.scaled_dot_product_attention(q, q, v).tolist())
print(F.scaled_dot_product_attention(q, k, vp).tolist())
print(F.scaled_dot_product_attention(q, k, vp, is_causal=True).tolist())
```

```
[[3.0, 4.0], [3.4066725560787154, 4.406672556078716],
 [3.5104695304536615, 4.510469530453662]]
[[3.0, 4.0], [3.4066725560787154, 4.406672556078715],
 [3.510469530453662, 4.510469530453662]]
[[5.0, 6.0], [4.0, 5.0],
 [3.510469530453662, 4.510469530453662]]
```

The last digit of the second row changes: `4.406672556078716` before the permutation and
`4.406672556078715` after it. The permutation symmetry is exact over $`\mathbb{R}` and approximate
in binary64, because reordering the terms reorders the rounding. Six printed decimals hide that in
the Lean transcript above. {ref "floats"}[The floating-point chapter] explains how such rounding
differences affect claims about executable computations.

The example supplies queries, keys, and values directly, so it isolates attention from the
operations that produce those tensors. A complete language model needs a dependency argument for
those operations too. If a query at the current position were computed using a future token,
zeroing strict-future attention weights would not remove that information from the query. The
causal-mask theorem therefore provides one part of a model-level argument. Its other parts must
show that embeddings, projections, normalization, and any state updates preserve the intended
prefix dependency. Their shapes identify the positions, but do not establish that dependency by
themselves.

## Finite Sentinels And Exact Masking

A common implementation replaces blocked logits by a large negative constant before the softmax.
TorchLean instead sets blocked softmax numerators to zero. The distinction matters even though a
sufficiently negative binary64 input can underflow:

```lean (name := mmSentinel)
-- This tests one floating-point exponential, not
-- normalization of a masked row.
#eval Float.exp (-1000.0)
```
```leanOutput mmSentinel
0.000000
```

This particular exponential underflows to zero. It does not make a finite sentinel safe in
general: stable softmax subtracts the row maximum, and an all-sentinel row becomes a row of zeros
before exponentiation, producing uniform weights rather than an all-zero masked row.

The specification layer states its causality theorem over $`\mathbb{R}`, where the exponential
of any real number is strictly positive. A finite sentinel therefore gives positive
strict-future weights; their size depends on the sentinel and the other logits.
An exact-zero theorem cannot apply.
TorchLean forms the softmax numerators directly, taking blocked entries to be zero by
definition, which makes the statement exact:

```lean (name := mmCausalThm)
-- Inspect the strict-future hypothesis and the exact zero
-- in the theorem conclusion.
open NN.Proofs.Models.Attention in
#check @hardMaskedSoftmaxSpec_causal_future_zero
```
```leanOutput mmCausalThm (whitespace := lax)
@hardMaskedSoftmaxSpec_causal_future_zero :
  ∀ {n : ℕ} (scores : Tensor ℝ [n, n]) (i j : Fin n),
    ↑i < ↑j →
      Spec.get2
          (Spec.hardMaskedSoftmaxSpec scores
            (Spec.causalMask n))
          i j =
        0
```

Every strict-future attention weight is exactly zero. The
executable path is the one whose numbers we compared against PyTorch above, and
{ref "runtime-approximation"}[Runtime Approximation] is where the rounded version of the same
softmax is related back to this real-valued one.

Read the theorem's indices as query row `i` and key column `j`. The assumption `i < j` is exactly
the strict-future case, and the conclusion concerns a weight in the normalized matrix. It is
stronger than observing a small printed number. At the same time, this statement is about the
masking operation itself: a complete language model must also avoid future information in its
other inputs and transformations, such as the construction of token features.

## The GPT-Family Constructor

The shared constructor in
{src "NN/API/Models/CausalTransformer.lean"}[`NN.API.Models.CausalTransformer`] has the architecture

```
token embedding
  -> learned positional embedding
  -> masked Transformer blocks
  -> layer normalization
  -> vocabulary projection
```

Its configuration records sequence length, vocabulary size, head count, head width, feed-forward
width, depth, activation, dropout, normalization order, and initialization. A model constructor then
receives the leading shape used by a particular run, so batch size is not part of the Transformer
architecture. The name “GPT-2-style” describes this architecture lineage; it does not mean that the
command loads OpenAI GPT-2 weights.

# State-Space Models

Mamba-style models return to explicit state but allow input-dependent scan parameters
{Informal.citep mamba2024}[]. Using the same convention of reading the updated state, a simplified
state-space recurrence is

$$`h_t=A_t h_{t-1}+B_t x_t,\qquad
y_t=C_t h_t+D_t x_t.`

{src "NN/API/Models/Mamba.lean"}[`NN.API.Models.Mamba`] holds the trainable side of this:

- `Mamba.Config`, which records the vocabulary size and the model width, and whose `validate`
  rejects a zero-length sequence before any tensor is allocated;
- `Mamba.languageModel`, a trainable selective state-space core followed by a linear
  projection back to vocabulary logits at every time step. The core is built from autograd-covered
  operations, so `--device cuda` trains the same parameters on the CUDA backend.

Validation is an explicit function on the configuration and sequence length. Calling it before
model construction rejects invalid extents before allocating model tensors:

```lean (name := mmMambaConfig)
-- Compare model validation with shape calculation for a
-- positive sequence length.
abbrev mmMamba : nn.models.Mamba.Config :=
  { vocabularySize := 256, modelWidth := 32 }

#eval mmMamba.validate 0
#eval mmMamba.validate 8
#eval (mmMamba.input 8, mmMamba.output 8)
```
```leanOutput mmMambaConfig
Except.error "Mamba: sequence length must be positive"
```
```leanOutput mmMambaConfig
Except.ok ()
```
```leanOutput mmMambaConfig
([8, 256], [8, 256])
```

`[0, 256]` is a valid tensor shape, but this model requires a positive sequence length.
`validate` checks that additional requirement and returns an `Except.error` before a scan runs.
The successful call with length eight separates this configuration check from the input and output
shape calculation printed below it.

Input and output shapes coincide here because the model predicts a distribution over the same
vocabulary at every position, which is what makes next-token training a matter of shifting the
target by one rather than of reshaping anything.

Here the input rows are one-hot vocabulary features and the output rows are logits. Equal shapes
do not mean equal representations: the input selects a token category, while a logit row scores
all possible next categories. A loss or sampling rule interprets those scores. The hidden
`modelWidth` can therefore be thirty-two even though the public input and output width is 256;
the vocabulary projection restores the external width after the recurrent computation.

The trainable core computes $`\Delta`, $`B_t` and $`C_t` from the current token features.
The specification describes the corresponding input-dependent recurrence as
`SelectiveMambaBlockSpec` in {src "NN/Spec/Models/Mamba.lean"}[`NN.Spec.Models.Mamba`], with
`runArray` and `runArrayWithHistory` runners. The API constructor supplies trainable parameters;
the spec runners expose the state and history needed to state properties of the recurrence.

The causality property proved for attention above has a recurrent counterpart, and it is stated for
the array runners in
{src "NN/MLTheory/Proofs/StateSpace/MambaCausality.lean"}[`MambaCausality.lean`]: for any initial
state $`h_0` and any sequences $`xs` and $`ys`,

$$`\operatorname{take}\,|xs|\;\bigl(\operatorname{run}(h_0,\;xs\mathbin{+\!+}ys)\bigr)
=\operatorname{run}(h_0,\;xs).`

Appending future tokens cannot change outputs already emitted for a prefix. The file proves this for
the diagonal S4 runner, the compact Mamba block, and the full selective block, including the variant
that carries a causal-convolution history, all on top of the scan algebra in
{src "NN/MLTheory/Proofs/StateSpace/Scan.lean"}[`Scan.lean`]. Stating it at the runner level is
deliberate: a runtime may use a chunked or parallel selective scan, but it has to refine these
runners.

The trainable constructor currently uses a selective Mamba-1 block with explicit convolution
history and input-dependent state parameters. Its time-step projection is stored as a dense
matrix. Loading factors from a low-rank checkpoint would require multiplying those factors to
match this forward parameterization; training the resulting dense matrix would then expose a
different set of independent parameters. The compact block below is chosen to make the recurrence
arithmetic visible, and should be read as that smaller spec example.

## Mamba Scan Example

Consider a block with one input channel, two state channels, one output channel, a zero gate
projection, and no skip term. Its state is small enough to follow through every token:

```lean (name := mmScanDefs)
-- Opposite decay signs make the two state channels
-- distinguishable in the output.
def mmSsm : Spec.Dynamics.DiagonalSSM Float 2 :=
  { A := [0.5, -0.5], B := [1.0, 1.0]
    C := [1.0, 2.0], D := [0.0, 0.0] }

def mmBlock : Models.MambaBlockSpec Float 1 2 1 :=
  { inProj := [[1.0, 1.0]]
    gateProj := [[0.0, 0.0]]
    outProj := [[1.0], [1.0]]
    ssm := mmSsm }

def mmH0 : Tensor Float [2] := [0.0, 0.0]
def mmXs : Array (Tensor Float [1]) :=
  #[[1.0], [0.0], [1.0]]

def mmOut (xs : Array (Tensor Float [1])) :
    Array (Tensor Float [1]) :=
  let (_, ys) := mmBlock.runArray mmH0 xs
  ys

#eval mmOut mmXs
```
```leanOutput mmScanDefs
#[[1.500000], [-0.250000], [1.875000]]
```

`runArray` returns both the final state and the outputs. `mmOut` selects the outputs so that we can
compare the emitted sequence before and after appending tokens.

I set the gate projection to zero so the gate is $`\sigma(0)=\tfrac12` at every
step and we can halve each readout by hand. Following the first token through, the input projects to
$`(1,1)`, the state becomes $`A\odot 0+B\odot(1,1)=(1,1)`, the readout is
$`C\odot(1,1)=(1,2)`, halving gives $`(0.5,1)`, and the output projection sums the two channels to
$`1.5`. The second token is a zero, so the state decays to $`(0.5,-0.5)`, the readout is
$`(0.5,-1)`, and the output is $`-0.25`. The third token brings the state to $`(1.25,1.25)` and the
output to $`1.875`.

Appending two tokens lets us compare the original outputs with the prefix of the longer run:

```lean (name := mmScanCausal)
-- The suffix adds outputs; compare the retained prefix with
-- the original run.
def mmYs : Array (Tensor Float [1]) :=
  #[[2.0], [-1.0]]

#eval mmOut (mmXs ++ mmYs)
#eval
  (mmOut (mmXs ++ mmYs)).take mmXs.size == mmOut mmXs
```
```leanOutput mmScanCausal (whitespace := lax)
#[[1.500000], [-0.250000], [1.875000], [2.687500], [-1.531250]]
```
```leanOutput mmScanCausal
true
```

The first three entries agree, and the Boolean comparison checks their equality beyond the printed
digits. This evaluates one block on one pair of sequences. The theorem quoted above covers every
block, every initial state, and every pair of sequences; applying it to these definitions gives:

```lean (name := mmScanProof)
-- Apply the general runner theorem to these concrete Float
-- parameters and inputs.
open NN.MLTheory.StateSpace in
example :
    (mmOut (mmXs ++ mmYs)).take mmXs.size = mmOut mmXs :=
  compactMamba_runArray_append_outputs_prefix
    mmBlock mmH0 mmXs mmYs
```

The theorem application proves the equality tested by the preceding `#eval`. It needs no new
calculation for these five tokens because the runner theorem already covers arbitrary sequence
lengths.

The trainable path uses TorchLean autograd operations on CPU or CUDA. The repository also has a
selective-scan CUDA operation for supported float execution. That runtime kernel is not thereby
proved equivalent to every equation in the high-level Mamba specification; the kernel boundary is
reported separately, and {ref "gpu-and-cuda"}[GPU And CUDA] describes what that boundary covers.

# Neural Operators

An FNO learns a map between functions sampled on a grid rather than a map between fixed feature
vectors {Informal.citep fno2021}[]. One spectral block has the schematic form

$$`v_{\ell+1}(x)
=\sigma\!\left(W_\ell v_\ell(x)
+\mathcal F^{-1}\!\left(R_\ell\cdot\mathcal F(v_\ell)\right)(x)\right).`

The two terms combine local and grid-wide information. $`W_\ell` mixes channels at each grid point.
The spectral term transforms the whole field, applies learned weights $`R_\ell` to retained Fourier
coefficients, and transforms back. The activation $`\sigma` acts on the sum. The retained mode set
therefore determines which frequencies the spectral branch can change.

The configuration in {src "NN/API/Models/FNO.lean"}[`NN.API.Models.FNO`] is again parameterized by
spatial rank, and the field-to-field boundary is visible in its shapes: a grid of sixty-four points
with a mode-band width of twelve, a latent width of thirty-two, and four blocks maps a sampled
field to a sampled field of the same extent.

```lean (name := mmFnoShapes)
-- Predict a scalar field on the same grid, with a separate
-- leading batch axis.
abbrev mmFno : nn.models.FNO.Config 1 :=
  { spatial := [64]
    modes := [12]
    width := 32
    layerCount := 4 }

#eval (mmFno.input [8], mmFno.output [8])
```
```leanOutput mmFnoShapes
([8, 64], [8, 64])
```

On each full-DFT axis, `modes` is the width of each of two index bands. A width of four on a
sixteen-point axis therefore retains eight indices. The second example uses an eight-point axis
whose bands overlap:

```lean (name := mmModes)
-- Compare disjoint retained frequency bands with
-- overlapping bands.
open Runtime.Autograd.Model.Layers.FNO.Internal
  (keepCoordinate) in
#eval ((List.range 16).filter (keepCoordinate 16 4),
  (List.range 8).filter (keepCoordinate 8 6))
```
```leanOutput mmModes (whitespace := lax)
([0, 1, 2, 3, 12, 13, 14, 15],
 [0, 1, 2, 3, 4, 5, 6, 7])
```

For an axis of length 16, indices $`0,\ldots,3` represent the zero and low positive frequencies;
indices $`12,\ldots,15` represent frequencies $`-4,\ldots,-1`. These are two retained bands, not
exact positive/negative pairs: the mask includes $`-4` but excludes $`+4`. When the bands overlap,
the whole axis is retained without double counting. The PyTorch transcript uses the same FFT
index convention:

```
# Translate storage indices into signed frequencies before
# selecting the two bands.
freqs = (torch.fft.fftfreq(16) * 16).to(torch.int64)
print(freqs.tolist())
print([i for i in range(16) if abs(int(freqs[i])) < 4 or int(freqs[i]) == -4])
```

```
[0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1]
[0, 1, 2, 3, 12, 13, 14, 15]
```

The frequency lists give a more precise description than “keep four modes.” The first list maps
every storage location to a signed frequency; the second selects the locations the layer actually
uses. Comparing them reveals the asymmetric endpoint at negative four. It also explains why
asking for six indices at each end of an eight-point axis keeps everything: the two sets cover
the axis. This is set membership, so their overlap does not multiply a coefficient twice.

The public constructor uses a dense multidimensional DFT. The CUDA Burgers command deliberately
selects a separate one-dimensional real-FFT parameterization backed by cuFFT. Both models have the
same typed field-to-field boundary, but they are not presented as numerically interchangeable
implementations, and the spectral-block chapter in
{ref "scientific-forward-models"}[Scientific Forward Models] says which parts are shared.

The field shape $`[8,64]` records eight independent sampled fields, each with sixty-four spatial
values. It does not describe the hidden channel width, the retained Fourier bands, or how samples
are spaced physically. Those choices live in the model and data configuration. Two operators can
therefore share this boundary while learning different maps. When comparing their predictions,
match the grid and coordinate convention as well as the tensor extents; a shifted field can have
the right number of entries while representing a different spatial function.

# References

The architectures are {Informal.citet resnet2016}[], {Informal.citet transformer2017}[],
{Informal.citet vit2021}[], {Informal.citet mamba2024}[], and {Informal.citet fno2021}[]. The
framework whose shapes and numbers we compare against throughout is
{Informal.citet pytorch2019}[].

- PyTorch pages for the operations used in the transcripts:
  [`nn.Conv2d`](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html),
  [`torch.fft`](https://pytorch.org/docs/stable/fft.html), and the entry for
  `scaled_dot_product_attention` on the
  [`torch.nn.functional`](https://pytorch.org/docs/stable/nn.functional.html) page.
- TorchLean sources shown above:
  {src "NN/API/Neural/Blocks.lean"}[`Blocks`],
  {src "NN/Spec/Layers/Attention.lean"}[`Attention`],
  {src "NN/Proofs/Models/Attention/CausalMask.lean"}[`CausalMask`], and
  {src "NN/Runtime/Autograd/Model/Fno.lean"}[`Fno`].
