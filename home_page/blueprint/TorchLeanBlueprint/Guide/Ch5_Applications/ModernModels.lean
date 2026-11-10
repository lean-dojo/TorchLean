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
import TorchLean.Tensor.Layers.Attention
import TorchLean.NN.Models.Recurrent.Stack
import TorchLean.NN.Attention.CausalMask
import TorchLean.NN.StateSpace.Causality
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.Roles
import TorchLean.NN.Loss
import TorchLean.NN.Session.Random
import TorchLean.Verify.Trainer

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean
open TorchLean Backend.Device nn.ResNet.Shortcut nn.ViT.Pooling
open Lean.Elab.Tactic.GuardMsgs.WhitespaceMode (lax)

#doc (Manual) "Modern Models" =>
%%%
tag := "modern-models"
%%%

A batch of eight examples should share one set of model parameters. A residual addition needs both
branches to produce the same shape. These are small requirements, but they're good places to start
reading a model: we can inspect the parameter list, or try to construct mismatched branches and see
what the API requires.

I'll keep that level of detail as the models get more involved. For a vision Transformer, we can
follow the pixels that survive patch extraction. For causal attention and a recurrent scan, we can
append tokens and inspect the earlier outputs. Shapes identify the objects in these calculations,
while the mask and scan definitions determine which inputs can affect them.

Our Lean blocks compute shapes from the model configurations. The PyTorch transcripts that
accompany them record shapes from corresponding operations {Informal.citep pytorch2019}[].

TorchLean's reusable constructors live under
{srcDir "TorchLean/NN/Models"}[`TorchLean/NN/Models`] and the runnable applications under
{srcDir "TorchLeanExamples/Models"}[`TorchLeanExamples/Models`]. A constructor such as
`nn.resnet` is reusable and rank-polymorphic. For the `resnet` command we chose a small
CIFAR configuration on purpose, so that you can run the whole data and training path locally.

A file imports only the modules it uses. For the causal Transformer that's a single module, and its
imports bring in the layers the model is built from:

```
import TorchLean.NN.Models.CausalTransformer.LanguageModel
```

# Batching And Parameter Sharing

Consider an ordinary one-hidden-layer network

$$`f_\theta(x)=W_2\,\rho(W_1x+b_1)+b_2.`

For a batch of $`B` vectors the shapes are

$$`X\in\mathbb{R}^{B\times d_{\mathrm{in}}},\qquad
W_1\in\mathbb{R}^{d_h\times d_{\mathrm{in}}},\qquad
W_2\in\mathbb{R}^{d_{\mathrm{out}}\times d_h}.`

In TorchLean you write the per-sample model and the batch axis separately. Here the per-sample
model maps `[3]` to `[2]`, and `nn.mapLeading` applies it at every index of a new leading axis,
sharing one parameter set.

The distinction becomes useful when the batch size changes. Eight examples require eight sets of
activations, but there's still only one classifier to update. In this example the first affine map
contributes fifteen weights and five biases, and the second contributes ten weights and two biases.
Every row shares those thirty-two scalars. From the printed shapes we can account for them without
inspecting an optimizer or relying on a naming convention.

```lean (name := mmBatchedShapes)
-- Build one parameter set, then reuse it for all eight
-- input rows.
def mmPerSample : nn.Builder (nn.Sequential [3] [2]) :=
  nn.Sequential![
    nn.linear 3 5,
    nn.op .relu,
    nn.linear 5 2
  ]

def mmBatched :
    nn.Builder (nn.Sequential [8, 3] [8, 2]) := do
  pure (nn.mapLeading [8] (← mmPerSample))

#eval nn.Sequential.stateShapes (nn.build 1 mmBatched)
```
```leanOutput mmBatchedShapes
[[5, 3], [5], [2, 5], [2]]
```

Those shapes are the per-sample parameter list. Adding a batch axis added no parameters. Run the
batched model on eight copies of the same row and you get eight copies of the same output row,
which is what parameter sharing means operationally.

Evaluation mode also makes this a controlled comparison. Here the model contains affine maps and a
ReLU, with no operation that depends on other batch members, so equal rows take equal paths through
the same parameters. The values themselves come from the seeded initial model: they're predictions
before training, not class probabilities or evidence that a task has been learned.

```lean (name := mmBatchedRun)
-- Identical rows in evaluation mode isolate parameter
-- sharing from input variation.
#eval do
  let model := nn.build 1 mmBatched
  let m ← nn.Module.instantiate model { device := cpu }
  m.eval
  let out ← m.forward (Tensor.full [8, 3] 1.0)
  IO.println s!"{out}"
```
```leanOutput mmBatchedRun (whitespace := lax)
[[0.133376, 0.625924],
 [0.133376, 0.625924],
 [0.133376, 0.625924],
 [0.133376, 0.625924],
 [0.133376, 0.625924],
 [0.133376, 0.625924],
 [0.133376, 0.625924],
 [0.133376, 0.625924]]
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
type, a theorem or an exported graph about this model states which batch shape it's about. That
annotation doesn't itself prove that rows are independent ({ref "torchlean_vs_pytorch"}[TorchLean
and PyTorch] shows a real batched-versus-sliced discrepancy of that kind). In exchange, you have to
name the batch axis, and `nn.mapLeading` is where you name it.

The batch model fixes how these affine maps and activations compose. A residual branch adds another
requirement: the two paths must agree on the shape at their join.

# Residual Networks

A residual block computes

$$`y=F_\theta(x)+S_\theta(x),`

where $`S` is either the identity or a projection {Informal.citep resnet2016}[]. The addition is
defined only when the two branches produce the same shape, so `nn.addBranches` requires matching
shapes, and the block's parameter list is the concatenation of the two branch lists:

```lean (name := mmBranches)
-- Both learned branches map the same 32 features to eight
-- output coordinates.
def mmBranches : nn.Builder (nn.Sequential [32] [8]) := do
  let wide ← nn.Sequential![nn.linear 32 8, nn.op .relu]
  let shortcut ← nn.linear 32 8
  pure (nn.addBranches wide shortcut)

#eval nn.Sequential.stateShapes (nn.build 3 mmBranches)
```
```leanOutput mmBranches
[[8, 32], [8], [8, 32], [8]]
```

Give the shortcut the wrong width and there's no model to run:

```lean +error -keep (name := mmClash)
-- This deliberate mismatch asks the shortcut for six
-- coordinates instead of eight.
def mmClash : nn.Builder (nn.Sequential [32] [8]) := do
  let wide ← nn.Sequential![nn.linear 32 8, nn.op .relu]
  let shortcut ← nn.linear 32 6
  pure (nn.addBranches wide shortcut)
```
```leanOutput mmClash (whitespace := lax)
Application type mismatch: The argument
  shortcut
has type
  nn.Sequential ([] ++ [32]) ([] ++ [6])
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
  pure (nn.residual (← nn.Sequential![nn.op .relu]))

#eval do
  let m ← nn.Module.instantiate (nn.build 0 mmRes)
    { device := cpu }
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

The two residual examples separate two choices that are easy to conflate. `mmBranches` learns a new
representation on both paths because its input and output widths differ. `mmRes` keeps the input
representation and adds a correction of the same width, and its negative coordinate makes the
identity path visible: even when ReLU contributes zero, the original value survives. For a learned
residual branch the same addition permits a small correction without requiring that branch to
reproduce the whole input.

## ResNet Configuration

In {src "TorchLean/NN/Models/ResNet.lean"}[`TorchLean.NN.Models.ResNet`] the configuration
is indexed by the number $`d` of spatial axes:

```
-- Each stage determines its own width and spatial geometry.
structure ResNet.Config (d : Nat) where
  inputChannels : Nat
  spatial : Tensor Nat [d]
  stem : Conv.Config d
  stages : List (ResNetStage.Config d) := []
  classCount : Nat
```

Grid extents are ordinary values such as `[32, 32]`. No separate nonzero proof is exposed in the
configuration (model validation rejects a zero extent before execution). Each stage contains
`first` and `second` convolutions and a `shortcut`. Its main branch applies the first convolution,
ReLU, and the second convolution, and the stage then adds the shortcut and applies ReLU. For any
leading shape $`L`, the model boundary is:

$$`\operatorname{input}
=L\mathbin{+\!+}(C_{\mathrm{in}},n_1,\ldots,n_d),`

$$`\operatorname{output}=L\mathbin{+\!+}(C_{\mathrm{class}}).`

Inside the model, stage $`i` has its own channel count $`C_i` and spatial grid.
Here a stem produces sixteen channels, and a residual stage doubles the channels while halving
both spatial extents:

```lean (name := mmResNetShapes)
/-- Downsample the branch and shortcut to the same shape. -/
abbrev mmDownsample : nn.ResNetStage.Config 2 :=
  { first :=
      { outChannels := 32, kernelSize := [3, 3]
        stride := [2, 2], padding := [1, 1] }
    second :=
      { outChannels := 32, kernelSize := [3, 3]
        padding := [1, 1] }
    shortcut := projection
      { outChannels := 32, kernelSize := [1, 1]
        stride := [2, 2] } }

/-- A stem, one downsampling stage, and ten class scores. -/
abbrev mmResNet : nn.ResNet.Config 2 :=
  { inputChannels := 3
    spatial := [32, 32]
    stem :=
      { outChannels := 16, kernelSize := [3, 3]
        padding := [1, 1] }
    stages := [mmDownsample]
    classCount := 10 }

#eval (mmResNet.inputShape [8],
  mmDownsample.outputShape [32, 32] [8],
  mmResNet.outputShape [8])
```
```leanOutput mmResNetShapes (whitespace := lax)
([8, 3, 32, 32], [8, 32, 16, 16], [8, 10])
```

```lean (name := mmResNetValidation)
#eval mmResNet.validate
```
```leanOutput mmResNetValidation
Except.ok ()
```

An identity shortcut is valid only when the branch preserves its complete feature shape. The
downsampling stage changes both width and grid, so its shortcut is a projection convolution. Before
allocating parameters, validation checks that the branch and shortcut produce the same channels and
spatial extents.

For stages that should preserve the grid, `Window.samePadding [1, 1]` supplies
a unit-stride $`3\times3` kernel with padding one. The window has a preservation theorem:

```lean (name := mmSamePadding)
-- The equality quantifies over every input grid and kernel
-- radius.
#check @Window.convOutputSpatial_samePadding
```
```leanOutput mmSamePadding (whitespace := lax)
@Window.convOutputSpatial_samePadding :
  ∀ {d : ℕ} (input radius : Tensor ℕ [d]),
    (Window.samePadding radius).convOutputSpatial input = input
```

That theorem states equality of the spatial grids for every input and radius. Channel agreement is
a separate requirement of an identity shortcut. Here a PyTorch example isolates the stem, global
pooling, and head, so we can see which axes pooling removes:

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

TorchLean's ResNet constructor uses a convolutional stem, the configured list of residual stages,
global average pooling over all $`d` spatial axes, and a linear classifier. An empty stage list
pools the stem output, and the stem geometry and a positive class count are still validated. And
the model API isn't tied to images, to two dimensions, or to one batch axis.

The isolated stem above produces sixteen pooled features, and the Lean model's residual stage
produces thirty-two. That final channel count, not the spatial grid, is what the classifier
consumes.

## Residual Branch Shapes

Open
{src "TorchLeanExamples/Models/Vision/ResNet.lean"}[`TorchLeanExamples/Models/Vision/ResNet.lean`]
and inspect `modelConfig`. Each stage's second convolution determines the next stage's input
channels, and the last stage determines the classifier input width. A width change also requires a
matching shortcut in that stage. When we build branches separately, we have to preserve that
agreement: the `mmClash` block above fails to elaborate because its two output widths differ.

# Vision Transformers

A vision Transformer first turns a spatial field into a token sequence
{Informal.citep vit2021}[]. For an input with spatial extent $`n_1\times\cdots\times n_d`, patch
kernel $`k`, stride $`s`, padding $`p`, and unit dilation, each output extent is

$$`n'_i
=\left\lfloor\frac{n_i+2p_i-k_i}{s_i}\right\rfloor+1.`

This formula assumes a positive stride and enough padded input for a complete kernel. Model
validation rejects unusable patch geometry instead of interpreting a negative extent.

If the patch convolution emits $`D` channels then the patch grid becomes

$$`B\times D\times n'_1\times\cdots\times n'_d
\;\longrightarrow\;
B\times N\times D,\qquad
N=\prod_i n'_i.`

{src "TorchLean/NN/Models/ViT.lean"}[`TorchLean.NN.Models.ViT`] implements that conversion
in `ViT.patchesToTokens`:
the implementation reshapes the patch grid and moves the channel axis to the end, so the following
Transformer block receives the conventional `batch × sequence × embedding` layout. Every stage is a
derived field of the configuration, so we can print the whole pipeline. Take $`32\times32` inputs
with $`4\times4` non-overlapping patches, sixty-four embedding channels, four heads of width
sixteen, two blocks, and a learned class slot:

```lean (name := mmVitConfig)
-- Four heads of width sixteen match the 64-channel patch
-- embedding.
abbrev mmVit : nn.ViT.Config 2 :=
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
    pooling := cls
    classCount := 10 }

#eval (mmVit.toEncoderConfig.patchGrid.to Shape,
  mmVit.toEncoderConfig.patchCount,
  mmVit.toEncoderConfig.sequenceLength)
```
```leanOutput mmVitConfig
([8, 8], 64, 65)
```

A $`32\times32` image with stride-four patches has an $`8\times8` patch grid, hence sixty-four
tokens, and the class slot makes the encoded sequence length sixty-five. For a batch of eight, the
three intermediate tensor shapes follow:

```lean (name := mmVitStages)
-- Track where spatial axes become a token axis and the
-- class token is inserted.
#eval (mmVit.toEncoderConfig.patchShape [8],
  mmVit.toEncoderConfig.patchTokenShape [8],
  mmVit.toEncoderConfig.outputShape [8])
```
```leanOutput mmVitStages (whitespace := lax)
([8, 64, 8, 8], [8, 64, 64], [8, 65, 64])
```

```lean (name := mmVitEnds)
-- Classification removes the sequence axis and returns ten
-- logits per image.
#eval (mmVit.inputShape [8], mmVit.outputShape [8])
```
```leanOutput mmVitEnds
([8, 3, 32, 32], [8, 10])
```

In `[8, 64, 64]`, the two axes of length sixty-four mean different things: one counts patches and
one counts features within a patch token. Their equal sizes happen to hide a possible transpose
mistake. Neighboring shapes, `[8, 64, 8, 8]` before and `[8, 65, 64]` after, disambiguate them:
adding the class slot changes the sequence length, while the embedding width stays fixed. At the
end, `[8, 10]` says that the classifier produces one score vector per image, not one per patch.

In PyTorch, the corresponding operations expose the reshape and axis permutation:

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

Lean and PyTorch agree on the intermediate shapes at every stage. `flatten(2).transpose(1, 2)`
computes the token tensor at runtime, while `mmVit.toEncoderConfig.patchTokenShape [8]` computes
its shape without running the model (`toEncoderConfig` selects the encoder settings inherited by
the classifier configuration). Computing that shape doesn't run validation: the model constructor
separately calls `ViT.EncoderConfig.validate` to reject an unusable patch grid.

## Patch Coverage

The floor in the extent formula determines how many complete patches fit. Keep the configuration
above and change the patches from $`4\times4` to $`5\times5`:

```lean (name := mmVitOdd)
-- Change patch geometry alone so the loss of border
-- coverage is visible.
abbrev mmVitOdd : nn.ViT.Config 2 :=
  { mmVit with
    patchEmbedding :=
      { outChannels := 64
        kernelSize := [5, 5]
        stride := [5, 5]
        padding := [0, 0] } }

#eval (mmVitOdd.toEncoderConfig.patchGrid.to Shape,
  mmVitOdd.toEncoderConfig.patchCount,
  mmVitOdd.toEncoderConfig.sequenceLength)
```
```leanOutput mmVitOdd
([6, 6], 36, 37)
```

Six patches per axis, thirty-six tokens, thirty-seven with the class slot. Six patches of width
five cover thirty of the thirty-two pixels, so in every image the model doesn't read the last two
rows and the last two columns at all. PyTorch's convolution produces the same smaller grid:

```
# A stride-five convolution emits only complete patches from
# the 32-pixel grid.
patch = nn.Conv2d(3, 64, kernel_size=5, stride=5)
print(tuple(patch(torch.zeros(8, 3, 32, 32)).shape))
```

```
(8, 64, 6, 6)
```

Both results follow the standard convolution convention. Print the derived grid before loading
images and you can see that this configuration discards a border. Types stay consistent: the
Transformer receives exactly the number of tokens that the patch convolution produces. Whether
discarding those pixels is acceptable is a separate modeling decision.

Increasing depth cannot recover omitted pixels. A smaller stride can preserve more coverage,
but also creates more tokens for attention to process.

For $`H` attention heads of width $`D_h` the model width is

$$`D=H D_h,`

and scaled dot-product attention is

$$`\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{D_h}}+M\right)V.`

FlashAttention evaluates the same softmax over blocks of keys, keeping a running maximum and
rescaling its partial sums whenever the maximum grows. We call that schedule
`Softmax.Schedule.online`. Over the reals it gives the two-pass result for every block partition,
mask and multiplier (`Tensor.attention_online_eq`), and
{src "TorchLean/Verify/RuntimeApprox/OnlineSoftmax.lean"}[`OnlineSoftmax.lean`] bounds its rounding
error in every IEEE format, with the scores as inputs: the rounding of $`QK^\top` isn't included.

TorchLean's reusable ViT constructor accepts any number of encoder blocks. It adds a learned
positional table before the stack, and its classifier can use either mean token pooling or a
learned class slot. For the small CIFAR application in
{src "TorchLeanExamples/Models/Vision/ViT.lean"}[`Vision/ViT.lean`] we picked a $`4\times4` crop,
$`2\times2` patches, and two blocks. Those choices keep the example quick without changing the
architecture that the public API represents.

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

Within each layer, the hidden and cell states keep stable dimensions across time. Different layers
can have different widths. `nn.Recurrent.Config.hiddenWidths` lists those widths in
execution order, and the `rnn`, `gru`, and `lstm` constructors append a linear head at every time
step:

```lean (name := mmRecurrent)
/-- Three recurrent layers with independent widths. -/
abbrev mmRecurrent : nn.Recurrent.Config :=
  { sequenceLength := 8
    inputWidth := 2
    hiddenWidths := [5, 3, 4]
    outputWidth := 1 }

example : nn.Builder (nn.Sequential [8, 2] [8, 1]) :=
  nn.Recurrent.gru mmRecurrent

example :
    { mmRecurrent with
      sequenceLength := 0
      hiddenWidths := [] }.validate = .ok () := by decide
```

The intermediate feature widths are $`2\to5\to3\to4\to1`, while the time axis stays eight
throughout. With `hiddenWidths := []`, only the time-distributed linear head remains. It accepts an
empty sequence because there's no recurrent core to unroll. Input and output widths must remain
positive. A nonempty stack also requires a positive sequence length and positive hidden widths.

These trainable constructors start every call from zero state and return the projected sequence.
They don't carry hidden state between calls. For explicit state, use the recurrent models in
`Recurrent`, which take an initial state. `sequence`, `stacked`,
`generator`, `languageModel` and `encoderDecoder` also return the final state. `classifier` returns
only its head output, `bidirectional` only the aligned output stream, and `residual` its last
residual feature (which isn't a recurrent state). A `Recurrent.Stack` holds cells whose widths
match: `.cons` adds a cell whose output width matches the next cell's input, and `.nil` preserves
the input with no recurrent state. `States` contains one state per layer, with each state indexed
by that layer's width (LSTM layers carry both hidden and cell tensors).

For two RNN cells with widths $`2\to3\to1`, the states have widths three and one:

```lean (name := mmReferenceStates)
open Ops nn.Layer in
example
    (first : Pack (Ref (m := Id) (α := Float))
      (rnnCellShapes 2 3))
    (second : Pack (Ref (m := Id) (α := Float))
      (rnnCellShapes 3 1))
    (h₃ : Tensor Float [3]) (h₁ : Tensor Float [1]) :
    (Recurrent.Stack.cons (Recurrent.Cell.rnn first)
        (.cons (Recurrent.Cell.rnn second) .nil) :
      Recurrent.Stack Id Float
        (fun width => Tensor Float [width]) 2 1).States :=
  (h₃, (h₁, ()))
```

`Recurrent.stacked` runs such a stack from those states, applies a head at every time step, and
returns both the projected sequence and the final states. An empty stack uses `()` for its state,
and an empty sequence preserves every supplied layer state. Here one RNN cell of width one (weight
$`\tfrac12` on the input and on the hidden state, and no bias) runs from zero over the inputs
$`1, 0` with the identity head:

```lean (name := mmReferenceRun)
#eval
  let layers :
      Recurrent.Stack Id Float
        (fun width => Tensor Float [width]) 1 1 :=
    .cons (.rnn (.cons ([[0.5, 0.5]] : Tensor Float [1, 2])
      (.cons ([0.0] : Tensor Float [1]) .nil))) .nil
  let result :
      Tensor Float [2, 1] × (Tensor Float [1] × Unit) :=
    Recurrent.stacked (m := Id) layers pure ([0.0], ())
      ([[1.0], [0.0]] : Tensor Float [2, 1])
  (result.1, result.2.1)
```
```leanOutput mmReferenceRun
([[0.462117], [0.227033]], [0.227033])
```

The final state is the last hidden row, and passing it as the state of the next call continues the
sequence. `Recurrent.sequence`, `classifier`, `generator`, `bidirectional`, `languageModel`,
`encoderDecoder` and `residual` are the other recurrent models, each for any cell and any head.

With the state passed in, the models that return their final state (`sequence`, `stacked`,
`generator` and `languageModel`) get a streaming interface: a caller can supply the state left by a
previous chunk and continue the computation. Restarting from zero instead describes a new sequence.
`bidirectional` reads the whole sequence and `residual` returns a feature, so neither streams. For
an LSTM there are two pieces to retain (the hidden state and the cell state). Keeping only the
hidden output loses information needed by the next update. Their matching widths make the gate
products legal, but correct streaming also requires carrying the actual values.

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
#eval Attention.causalMask 4
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

def mmCausal : Attention.Inputs Float 3 3 2
    (by decide) (by decide) :=
  { query := mmQ, key := mmQ, value := mmV
    mask := some (Attention.causalMask 3) }

def mmFull : Attention.Inputs Float 3 3 2
    (by decide) (by decide) :=
  { query := mmQ, key := mmQ, value := mmV, mask := none }

#eval Attention.scaledDotProduct mmCausal
#eval Attention.scaledDotProduct mmFull
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
itself. Both runs agree on the last row, because the last query may attend everywhere in either
case. Here's PyTorch on the same inputs, in double precision:

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
one greater than its first, so any normalized attention row has that same difference. In the causal
run, the middle output lies between the first two value rows because those are its only permitted
keys, while the unrestricted middle row can also use the third value (which explains its larger
first coordinate). The two transcripts preserve both relationships. Both computations include the
scale factor $`1/\sqrt{D_h}`. All these causal rows contain a visible key, so they don't exercise
the separate convention that a fully blocked row returns zeros.

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

def mmCausal4 : Attention.Inputs Float 4 4 2
    (by decide) (by decide) :=
  { query := mmQ4, key := mmQ4, value := mmV4
    mask := some (Attention.causalMask 4) }

#eval Attention.scaledDotProduct mmCausal4
```
```leanOutput mmPrefix (whitespace := lax)
[[1.000000, 2.000000], [2.339523, 3.339523],
 [3.510470, 4.510470], [5.837654, 6.837654]]
```

The fourth output uses all four visible values. Only the first three should match the shorter run.
This comparison checks prefix preservation on these inputs, not on every possible sequence.

## Permuting Keys, Values, And Queries

Attention has a symmetry that explains why the GPT and ViT constructors both add a positional table
before their stacks. Permute the keys and values by the same permutation, leave the queries alone,
and an unmasked attention row is unchanged, because the row is a sum over key positions and real
addition is invariant under permutation. Floating-point reduction can differ in its last bits when
that order changes. Reversing our three keys and values:

```lean (name := mmPermute)
-- Move keys and their associated values together while
-- leaving queries fixed.
def mmKRev : Tensor Float [3, 2] :=
  [[1.0, 1.0], [0.0, 1.0], [1.0, 0.0]]
def mmVRev : Tensor Float [3, 2] :=
  [[5.0, 6.0], [3.0, 4.0], [1.0, 2.0]]

def mmFullRev : Attention.Inputs Float 3 3 2
    (by decide) (by decide) :=
  { query := mmQ, key := mmKRev, value := mmVRev,
    mask := none }

def mmCausalRev : Attention.Inputs Float 3 3 2
    (by decide) (by decide) :=
  { query := mmQ, key := mmKRev, value := mmVRev
    mask := some (Attention.causalMask 3) }

#eval Attention.scaledDotProduct mmFullRev
#eval Attention.scaledDotProduct mmCausalRev
```
```leanOutput mmPermute (whitespace := lax)
[[3.000000, 4.000000], [3.406673, 4.406673],
 [3.510470, 4.510470]]
```
```leanOutput mmPermute (whitespace := lax)
[[5.000000, 6.000000], [4.000000, 5.000000],
 [3.510470, 4.510470]]
```

The first output agrees with the previous unmasked result to the six decimals shown. Our queries
stayed fixed while the keys and their associated values moved together, which for each query
permutes the attention scores and the values they weight by the same permutation and leaves their
weighted sum unchanged over real arithmetic.

Permuting the input to self-attention is a different operation: it permutes queries as well as keys
and values. Without a mask or positional information, the output rows then follow the same
permutation. That's permutation equivariance, which `selfAttention_reindexOuter` in
{src "TorchLean/NN/Attention/PermutationEquivariance.lean"}[
`PermutationEquivariance.lean`] proves for the real-valued spec. In the GPT and ViT constructors,
learned positional tables give otherwise identical tokens different inputs at different positions
{Informal.citep transformer2017}[]{Informal.citep vit2021}[].

The second output changes because the causal mask stays tied to array positions while the keys and
values move. The first query can still see only slot zero, but after the reversal that slot holds
the value that was last in the original order. A causal mask supplies an order-dependent visibility
rule, while positional embeddings supply position-dependent features. Both can make a token's
position affect its output.

PyTorch's double-precision output also shows what happens in the last displayed digits:

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

In the second row the last digit changes: `4.406672556078716` before the permutation and
`4.406672556078715` after it. Over $`\mathbb{R}` the permutation symmetry is exact, while in
binary64 it's approximate, because reordering the terms reorders the rounding. Six printed decimals
hide that in the Lean transcript above. {ref "floats"}[The floating-point chapter] explains how
such rounding differences affect claims about executable computations.

Because we wrote the queries, keys, and values by hand, the example isolates attention from the
operations that produce those tensors. A complete language model needs a dependency argument for
those operations too: if a query at the current position were computed using a future token,
zeroing strict-future attention weights wouldn't remove that information from the query. So the
causal-mask theorem provides one part of a model-level argument. Its other parts must show that
embeddings, projections, normalization, and any state updates preserve the intended prefix
dependency. Their shapes identify the positions, but don't establish that dependency by themselves.

## Finite Sentinels And Exact Masking

A common implementation replaces blocked logits by a large negative constant before the softmax. We
set blocked softmax numerators to zero instead. And the difference matters even though a
sufficiently negative binary64 input can underflow:

```lean (name := mmSentinel)
-- This tests one floating-point exponential, not
-- normalization of a masked row.
#eval Float.exp (-1000.0)
```
```leanOutput mmSentinel
0.000000
```

This particular exponential underflows to zero. That doesn't make a finite sentinel safe in
general: stable softmax subtracts the row maximum, and an all-sentinel row becomes a row of zeros
before exponentiation, producing uniform weights, not an all-zero masked row.

TorchLean's specification layer states its causality theorem over $`\mathbb{R}`, where the
exponential of any real number is strictly positive. So a finite sentinel gives positive
strict-future weights (their size depends on the sentinel and the other logits). An exact-zero
theorem can't apply. We build the softmax numerators with blocked entries equal to zero by
definition, which makes the statement exact:

```lean (name := mmCausalThm)
-- Inspect the strict-future hypothesis and the exact zero
-- in the theorem conclusion.
open Softmax in
#check @hardMasked_causalMask_eq_zero
```
```leanOutput mmCausalThm (whitespace := lax)
@hardMasked_causalMask_eq_zero :
  ∀ {n : ℕ} (scores : Tensor ℝ [n, n]) (i j : Fin n),
    ↑i < ↑j →
      (hardMasked scores (Attention.causalMask n)).get2 i j = 0
```

Every strict-future attention weight is exactly zero. We evaluated this same specification with
`Float` in the comparison above. In {ref "runtime-approximation"}[Runtime Approximation] we develop
error bounds for rounded operations, but those bounds have their own hypotheses and don't by
themselves prove this entire attention execution equivalent to its real-valued specification.

## The GPT-Family Constructor

In
{src "TorchLean/NN/Models/CausalTransformer.lean"}[`TorchLean.NN.Models.CausalTransformer`],
the shared constructor has the architecture

```
token embedding
  -> learned positional embedding
  -> masked Transformer blocks
  -> layer normalization
  -> vocabulary projection
```

The constructor's configuration records sequence length, vocabulary size, head count, head width,
feed-forward width, depth, activation, dropout, normalization order, and initialization. A model
constructor then receives the leading shape used by a particular run, so batch size isn't part of
the Transformer architecture. The name “GPT-2-style” describes this architecture lineage, and it
doesn't mean that the command loads OpenAI GPT-2 weights.

# State-Space Models

Mamba-style models return to explicit state but allow input-dependent scan parameters
{Informal.citep mamba2024}[]. Using the same convention of reading the updated state, a simplified
state-space recurrence is

$$`h_t=A_t h_{t-1}+B_t x_t,\qquad
y_t=C_t h_t+D_t x_t.`

{src "TorchLean/NN/Models/Mamba.lean"}[`TorchLean.NN.Models.Mamba`] holds the trainable side
of this:

- `Mamba.Config`, which records the vocabulary size and an ordered list of `modelWidths`, together
  with the expansion, state width, and convolution kernel width used by its layers.
- `Mamba.languageModel`, a stack of trainable selective state-space layers followed by a linear
  projection back to vocabulary logits at every time step. Its layers use operations covered by
  TorchLean's autograd tape, and LibTorch provides CUDA execution.

Validation is a function on the configuration and sequence length. Called before model
construction, it rejects invalid extents before allocating model tensors:

```lean (name := mmMambaConfig)
-- Compare model validation with shape calculation for a
-- positive sequence length.
abbrev mmMamba : nn.Mamba.Config :=
  { vocabularySize := 256, modelWidths := [32, 16] }

#eval mmMamba.validate 0
#eval mmMamba.validate 8
#eval (mmMamba.inputShape 8, mmMamba.outputShape 8)
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

`[0, 256]` is a valid tensor shape, but a nonempty Mamba stack requires a positive sequence length.
`validate` checks that additional requirement and returns an `Except.error` before a scan runs.
The successful call with length eight separates this configuration check from the input and output
shape calculation printed below it.

An empty `modelWidths` list builds only the vocabulary-to-vocabulary linear head. Such a model
accepts a zero-length sequence, and its validation doesn't check the unused scan configuration.
Vocabulary size must remain positive, and nonempty layers also require positive model widths:

```lean (name := mmMambaHead)
example :
    ({ vocabularySize := 256, modelWidths := []
       expansion := 0, stateWidth := 0, kernelWidth := 0 }
      : nn.Mamba.Config).validate 0 = .ok () := by
  decide
```

Input and output shapes coincide here because the model predicts a distribution over the same
vocabulary at every position, which is what makes next-token training a matter of shifting the
target by one, not of reshaping anything.

Here the input rows are one-hot vocabulary features and the output rows are logits. Equal shapes
don't mean equal representations: the input selects a token category, while a logit row scores all
possible next categories. A loss or sampling rule interprets those scores. So the hidden widths can
be thirty-two and then sixteen even though the public input and output width is 256: the vocabulary
projection restores the external width after the recurrent computation.

In the trainable core, $`\Delta`, $`B_t` and $`C_t` are computed from the current token features.
On the specification side, `Mamba.SelectiveBlock` in
{src "TorchLean/NN/Models/Mamba/Block.lean"}[`TorchLean.NN.Models.Mamba.Block`] describes the
corresponding input-dependent recurrence, with `runArray` and `runArrayWithHistory` runners. Our
API constructor supplies trainable parameters, while the spec runners expose the state and history
needed to state properties of the recurrence.

Attention's causality property, proved above, has a recurrent counterpart, stated for the array
runners in {src "TorchLean/NN/StateSpace/Causality.lean"}[`Causality.lean`]: for
any initial state $`h_0` and any sequences $`xs` and $`ys`,

$$`\operatorname{take}\,|xs|\;\bigl(\operatorname{run}(h_0,\;xs\mathbin{+\!+}ys)\bigr)
=\operatorname{run}(h_0,\;xs).`

Appending future tokens can't change outputs already emitted for a prefix. That file proves it for
the diagonal S4 runner, the compact Mamba block, and the full selective block (including the
variant that carries a causal-convolution history), all on top of the scan algebra in
{src "TorchLean/NN/StateSpace/Scan.lean"}[`Scan.lean`]. We stated it at the runner level
on purpose: a runtime may use a chunked or parallel selective scan, but it has to refine these
runners.

TorchLean's trainable constructor uses a selective Mamba-1 block that carries its convolution
history and has input-dependent state parameters. Its time-step projection is stored as a dense
matrix. Loading factors from a low-rank checkpoint would require multiplying those factors to match
this forward parameterization, and training the resulting dense matrix would then expose a
different set of independent parameters. We chose the compact block below to make the recurrence
arithmetic visible, and you should read it as that smaller spec example.

## Mamba Scan Example

Consider a block with one input channel, two state channels, one output channel, a zero gate
projection, and no skip term. Its state is small enough to follow through every token:

```lean (name := mmScanDefs)
-- Opposite decay signs make the two state channels
-- distinguishable in the output.
def mmSsm : Dynamics.DiagonalSSM Float 2 :=
  { A := [0.5, -0.5], B := [1.0, 1.0]
    C := [1.0, 2.0], D := [0.0, 0.0] }

def mmBlock : Mamba.CompactBlock Float 1 2 1 :=
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

I set the gate projection to zero so the gate is $`\sigma(0)=\tfrac12` at every step and we can
halve each readout by hand. If we follow the first token through, the input projects to $`(1,1)`,
the state becomes $`A\odot 0+B\odot(1,1)=(1,1)`, the readout is $`C\odot(1,1)=(1,2)`, halving gives
$`(0.5,1)`, and the output projection sums the two channels to $`1.5`. Next comes a zero token, so
the state decays to $`(0.5,-0.5)`, the readout is $`(0.5,-1)`, and the output is $`-0.25`. Then the
third token brings the state to $`(1.25,1.25)` and the output to $`1.875`.

Appending two tokens lets us compare the original outputs with the prefix of the longer run:

```lean (name := mmScanCausal)
-- The suffix adds outputs. Compare the retained prefix with
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

In the longer run the first three entries agree with the original outputs, and the Boolean
comparison checks their equality beyond the printed digits. This evaluates one block on one pair of
sequences. But the theorem quoted above covers every block, every initial state, and every pair of
sequences, and applying it to these definitions gives:

```lean (name := mmScanProof)
-- Apply the general runner theorem to these concrete Float
-- parameters and inputs.
open Mamba.CompactBlock in
example :
    (mmOut (mmXs ++ mmYs)).take mmXs.size = mmOut mmXs :=
  runArray_causal
    mmBlock mmH0 mmXs mmYs
```

Applying the theorem proves the equality tested by the preceding `#eval`. No new calculation is
needed for these five tokens, because the runner theorem already covers arbitrary sequence lengths.

On the trainable path, TorchLean autograd operations run on CPU or CUDA, and the CUDA selective
scan is an ATen composition with a backward rule that we chose. None of that proves the
implementation equivalent to every equation in the high-level Mamba specification.
{ref "gpu-and-cuda"}[GPU And CUDA] describes the runtime boundary.

# Neural Operators

An FNO learns a map between functions sampled on a grid, not a map between fixed feature vectors
{Informal.citep fno2021}[]. One spectral block has the schematic form

$$`v_{\ell+1}(x)
=\sigma\!\left(W_\ell v_\ell(x)
+\mathcal F^{-1}\!\left(R_\ell\cdot\mathcal F(v_\ell)\right)(x)\right).`

Its two terms combine local and grid-wide information. $`W_\ell` mixes channels at each grid point.
The spectral term transforms the whole field, applies learned weights $`R_\ell` to retained Fourier
coefficients, and transforms back. Then the activation $`\sigma` acts on the sum. So the retained
mode set determines which frequencies the spectral branch can change.

In {src "TorchLean/NN/Models/FNO.lean"}[`TorchLean.NN.Models.FNO`] the configuration is
again parameterized by spatial rank, and its shapes show the field-to-field boundary: a grid of
sixty-four points with a mode-band width of twelve, a latent width of thirty-two, and four blocks
maps a sampled field to a sampled field of the same extent.

```lean (name := mmFnoShapes)
-- Predict a scalar field on the same grid, with a separate
-- leading batch axis.
abbrev mmFno : nn.FNO.Config 1 :=
  { spatial := [64]
    modes := [12]
    width := 32
    layerCount := 4 }

#eval (mmFno.inputShape [8], mmFno.outputShape [8])
```
```leanOutput mmFnoShapes
([8, 64], [8, 64])
```

The defaults describe scalar fields. `inputFeatures` and `outputFeatures` instead let each grid
point carry a vector or a tensor. For example, three measured quantities can predict a two-by-two
tensor at each point without changing the spatial grid:

```lean (name := mmFnoFeatures)
abbrev mmTensorFno : nn.FNO.Config 1 :=
  { mmFno with
    inputFeatures := [3]
    outputFeatures := [2, 2] }

#eval (mmTensorFno.inputShape [8],
  mmTensorFno.outputShape [8])
```
```leanOutput mmFnoFeatures
([8, 64, 3], [8, 64, 2, 2])
```

Within the model, the lift combines the input features into the latent channels, and the projection
produces the output features. Fourier transforms still act only on spatial axes. Feature shapes
must have a positive number of entries (`[]` means one scalar per point, not zero entries), while
an empty batch is allowed.

On each full-DFT axis, `modes` is the width of each of two index bands, so a width of four on a
sixteen-point axis retains eight indices. The second example uses an eight-point axis whose bands
overlap:

```lean (name := mmModes)
-- Compare disjoint retained frequency bands with
-- overlapping bands.
open FNO (keepCoordinate) in
#eval ((List.range 16).filter (keepCoordinate 16 4),
  (List.range 8).filter (keepCoordinate 8 6))
```
```leanOutput mmModes (whitespace := lax)
([0, 1, 2, 3, 12, 13, 14, 15],
 [0, 1, 2, 3, 4, 5, 6, 7])
```

For an axis of length 16, indices $`0,\ldots,3` represent the zero and low positive frequencies,
and indices $`12,\ldots,15` represent frequencies $`-4,\ldots,-1`. They form two retained bands,
not exact positive/negative pairs: the mask includes $`-4` but excludes $`+4`. When the bands
overlap, the whole axis is retained without double counting. PyTorch's transcript uses the same FFT
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

The public constructor transforms each spatial axis separately. With `spectralPath := .automatic`,
eager CUDA execution uses LibTorch FFTs and other interpreters use dense per-axis operations, while
`spectralPath := .denseReference` selects full-grid DFT matrices without changing the weights or
checkpoint layout. Both paths compute the same full-spectrum model, up to floating-point
differences in their transform algorithms.

The Burgers command uses this same constructor on a one-dimensional grid. Its CPU and CUDA runs
share the model and weight layout, and the selected runtime determines how the transforms execute.
The Burgers neural operator case study in {ref "model-examples-deep-dive"}[Case Studies] follows
that computation in detail.

Here the field shape $`[8,64]` records eight independent sampled fields, each with sixty-four
spatial values. That shape doesn't describe the hidden channel width, the retained Fourier bands,
or how samples are spaced physically. Those choices live in the model and data configuration. So
two operators can share this boundary while learning different maps. When you compare their
predictions, match the grid and coordinate convention as well as the tensor extents: a shifted
field can have the right number of entries while representing a different spatial function.

# References

The architectures are {Informal.citet resnet2016}[], {Informal.citet transformer2017}[],
{Informal.citet vit2021}[], {Informal.citet mamba2024}[], and {Informal.citet fno2021}[]. We
compare shapes and numbers against {Informal.citet pytorch2019}[] throughout.

- PyTorch pages for the operations used in the transcripts:
  [`nn.Conv2d`](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html),
  [`torch.fft`](https://pytorch.org/docs/stable/fft.html), and the entry for
  `scaled_dot_product_attention` on the
  [`torch.nn.functional`](https://pytorch.org/docs/stable/nn.functional.html) page.
- TorchLean sources shown above:
  {src "TorchLean/NN/Builder/Blocks.lean"}[`Blocks`],
  {src "TorchLean/Tensor/Layers/Attention.lean"}[`Attention`],
  {src "TorchLean/NN/Attention/CausalMask.lean"}[`CausalMask`], and
  {src "TorchLean/NN/Models/FNO/Block.lean"}[`Fno`].
