import VersoManual
import VersoBlueprint

open Verso.Genre Manual

#doc (Manual) "Example Walkthroughs" =>
%%%
tag := "examples"
%%%

This page is a guided path through the examples. The repository has many demos, but the first pass
should be short: one tensor file, one training run, one autograd run, one graph/checker run, and one
BugZoo card.

Each example group has a teaching role: what it exercises, where data enters the system, what gets
checked, and what a successful run establishes.

For GPT text training, Mamba, diffusion, PPO, or FNO on the Burgers dataset, *Modern Models
and Training* gives the deeper model walkthrough. The map below helps choose a starting point.

# The Examples At A Glance

The tree is organized around questions:

| Goal | Run or read |
|---|---|
| first tensor example | `NN/Examples/Quickstart/TensorBasics.lean` |
| first training example | `lake exe torchlean mlp --cpu --steps 10` |
| first autograd example | `NN/Examples/Quickstart/AutogradBasics.lean` |
| first graph example | `lake exe verify -- torchlean-ibp` |
| first BugZoo example | `NN.Examples.BugZoo.All` |
| first CUDA smoke test | `lake exe -K cuda=true torchlean mlp --cuda --steps 20` |
| first scientific ML example | `lake exe verify -- ode` or the PINN/FNO examples |

- *Can I build and inspect small tensors?* Read [Quickstart](https://github.com/lean-dojo/TorchLean/tree/main/NN/Examples/Quickstart/).
- *Can I train a model in Lean?* Read [Models](https://github.com/lean-dojo/TorchLean/tree/main/NN/Examples/Models/) and
  [Data](https://github.com/lean-dojo/TorchLean/tree/main/NN/Examples/Data/).
- *Can I cross the PyTorch boundary without losing shapes?* Read
  [PyTorch interop](https://github.com/lean-dojo/TorchLean/tree/main/NN/Examples/Interop/PyTorch/).
- *Can I see the graph that a verifier sees?* Read [Advanced GraphSpec and IR demos](https://github.com/lean-dojo/TorchLean/tree/main/NN/Examples/Advanced/).
- *Can I check a property or certificate?* Read the verification examples and fixtures.
- *Can I see common ML bugs as checked Lean case studies?* Read [Bug Zoo](https://github.com/lean-dojo/TorchLean/tree/main/NN/Examples/BugZoo/).

Most model demos run through one executable:

```
lake exe torchlean <demo> [args...]
```

The source-level umbrella is `NN.Examples.Zoo`. Building it is the quick health check for the
curated example surface:

```
lake build NN.Examples.Zoo
```

# Three Walkthroughs To Read First

## 1. A small model trains

Start with the MLP and CNN examples. They show the basic pattern:

- the dataset is represented as tensors;
- the model is built through `NN.API.nn`;
- parameters are explicit;
- the optimizer returns updated state;
- the log records an ordinary loss trace.

The size of the model is not the point here. This is the same runtime surface used by the larger
examples.

```
lake exe torchlean mlp --cpu --steps 10
lake exe torchlean cnn --cpu --n-total 20 --steps 1
```

## 2. A graph is checked

The verification examples start from the shared IR. A small model is lowered into `NN.IR.Graph`, a
payload supplies parameters, and a verifier pass computes a bound or checks a certificate.

```
lake exe verify -- torchlean-ibp
```

Read this together with the *Graphs and IR* and *Verification* chapters. The useful lesson is that
the verifier is not guessing which Python object was meant. It consumes the same graph with operation tags
described earlier in the book.

## 3. A real ML failure mode becomes a tiny Lean case study

Bug Zoo examples are deliberately small. They are not meant to impress by scale; they are meant to
name the failure mode precisely. Examples cover attention masks, KV caches and RoPE positions,
tokenizer boundaries, normalization state, batch invariance, Float32 assumptions, and autograd
boundaries.

The style is:

```
-- State the behavior directly, then make the boundary explicit.
#check Spec.hardMaskedSoftmaxSpec
#check runtimeFloat32_add_rewrites_to_ieee32
```

For papers, talks, and documentation, this is often the most useful example family: it shows how
TorchLean turns a bug pattern into a checked statement.

# Fast Kernels, Bug Cards, and Verification

A useful way to read the examples is as one chain:

1. *Fast kernel*: a runtime path uses an optimized implementation, such as fused CUDA attention.
2. *Bug card*: BugZoo states the semantic hazard, such as mask polarity, cache position mismatch, or
   floating-point boundary mismatch.
3. *Contract*: the spec names the intended mathematical behavior.
4. *Checker or theorem*: Lean checks an artifact, proves a local theorem, or records the remaining
   producer/runtime assumption.

For attention, the chain looks like this:

- Runtime: CUDA `multi_head_attention` may call fused native `flashAttentionFwd` and fused VJP
  kernels.
- Spec: `Spec.flashAttention` and `Spec.scaledDotProductAttention` name the forward contract.
- Theorem: `flashAttention_eq_scaledDotProductAttention` states that the fused spec denotes SDPA.
- Bug card: `BugZoo.AttentionMask` records the exact mask property, including future tokens getting
  zero attention under causal masking.
- Verification fixture: `torchlean-transformer-ibp` pushes a tiny transformer style graph through
  the same verifier surface.

This is the preferred TorchLean pattern: faster execution is welcome, but it is paired with a named
semantic contract and a visible proof or test boundary.

The pattern is either equality with the spec, or membership in a safe enclosure around the spec.

If the equality is proved, the fast path becomes an optimization. If it is checked only by tests, we
say that plainly. If the kernel is external CUDA code, the CUDA chapter names the assumption instead
of letting it disappear into the demo.

# How The Tree Is Organized

The examples are grouped so each subtree has a job:

- [Quickstart examples](https://github.com/lean-dojo/TorchLean/tree/main/NN/Examples/Quickstart/)
  Typed-tensor warmups, autograd basics, and the smallest training tutorials.
- [Model examples](https://github.com/lean-dojo/TorchLean/tree/main/NN/Examples/Models/)
  The main model zoo used by `lake exe torchlean`: MLP, CNN, ResNet, ViT, RNN/LSTM,
  Transformer, GPT text demos, Mamba demos, diffusion, and RL examples.
- [Advanced examples](https://github.com/lean-dojo/TorchLean/tree/main/NN/Examples/Advanced/)
  GraphSpec, float-mode, tensor-bridge, and TorchLean-to-PyTorch IR demos.
- [PyTorch interop examples](https://github.com/lean-dojo/TorchLean/tree/main/NN/Examples/Interop/PyTorch/)
  PyTorch round-trip scripts and generated-code interop examples.
- [Verification fixtures](https://github.com/lean-dojo/TorchLean/tree/main/NN/Examples/Verification/)
  Small bundled certificate/checker demos used by `lake exe verify -- ...` (with external producers
  under `scripts/verification/`).
- [Widget gallery](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Advanced/Widgets.lean)
  Infoview widgets used in the editor rather than through a long training run.
- [RL widget viewers](https://github.com/lean-dojo/TorchLean/tree/main/NN/Examples/RL/)
  File-backed PPO/GridWorld dashboards colocated with the RL examples they visualize.
- [Bug Zoo](https://github.com/lean-dojo/TorchLean/tree/main/NN/Examples/BugZoo/)
  Small checked case studies for attention masks, KV caches, tokenizers, normalization state,
  batching, floats, and autograd boundaries.
- [real data paths](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Data/RealPaths.lean) and
  [model data helpers](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/Common/RealData.lean)
  Shared paths and loaders for prepared real datasets under `data/real`.

That structure helps the examples read as one system: start with tensors, run models, lower graphs,
then check properties.

# A Good First Pass Through The Examples

For one short pass that touches the main layers, use this order:

1. Build once:
   `lake build`
2. Build the curated examples umbrella:
   `lake build NN.Examples.Zoo`
3. Typed tensor demo:
   `lake env lean --run NN/Examples/Quickstart/TensorBasics.lean`
4. Float-backend comparison demo:
   `lake exe torchlean float32_modes`
5. Widgets demo, opened in the editor:
   open the [widgets example API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Advanced/Widgets.lean)
6. Autograd demo:
   `lake env lean --run NN/Examples/Quickstart/AutogradBasics.lean -- --dtype float`
7. Model-zoo training demo:
   `lake exe torchlean mlp --cpu --steps 10`
8. Small complete verification demo:
   `lake exe verify -- torchlean-ibp`

Read the output alongside this book: training-focused material starts with *Training From Scratch*
and *Runtime and Autograd*; graph and verification artifacts start with *Graphs and IR*, then
*Verification*.

# What Success Looks Like

Expected outputs on a healthy run:

- `tensor_api_demo`: small typed tensors print cleanly.
- `float32_modes_demo`: the same numeric expression is evaluated under different float32
  semantics.
- `autodiff_demo`: prints a value and a gradient, not only a successful build.
- `lake exe torchlean mlp --cpu --steps 10`: prints a short loss trace trending downward.
- `lake exe torchlean gpt_adder --steps 50 --cuda` (after `lake build -R -K cuda=true`): runs the minGPT
  addition demo on GPU when the CUDA backend has been built.
- `torchlean-ibp`: prints interval bounds for the output node, not raw graph internals.

This framing distinguishes a meaningful semantic check from a command that merely exits with code
zero.

# Dtype And Backend Flags

Most training and runtime examples are written once and then instantiated by the command line
wrapper with a scalar type and backend chosen on the command line.

## `--dtype`

Many examples accept `--dtype`:

- `--dtype float`
  Lean runtime `Float` (binary64).
- `--dtype float32`
  An executable float32 mode used by several tutorials.
- `--dtype ieee32exec` or `--dtype ieee754exec`
  TorchLean's executable IEEE-754 binary32 model defined in Lean (`IEEE32Exec`).

Proof-only backends such as `FP32` and `NF` are discussed in the floating-point chapters.

## `--backend`

Many examples accept `--backend eager|compiled`:

- `--backend eager`
  Build and inspect a tape as the run executes.
- `--backend compiled`
  Lower the run to a first-order graph representation and execute it as a graph.

For debugging compilation to IR or verifier-style passes, `--backend compiled` is the usual default.

# Model zoo and real data

The model surface is wide: classical feedforward and conv nets, ViT and ResNet blocks,
recurrent and attention language models, state space and diffusion demos, and PPO RL
environments, not only the original MLP/CNN tutorials. Model demos are driven by:

`lake exe torchlean <demo> [flags...]`

The [model examples API](https://github.com/lean-dojo/TorchLean/tree/main/NN/Examples/Models/) contains the implementations, and the
[example runner API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/Runner.lean) defines the public subcommand names.

First run:

`lake exe torchlean --help`

## `torchlean` subcommands (curated zoo)

Use these as starting points; each module's header documents the full flag surface.

- `mlp`: tabular/vector regression; synthetic or CSV data; a good first GPU smoke test after `cuda=true`.
- `cnn`: convolutional training on CIFAR-style tensors; prepare `data/real` with the download
  script for real data.
- `resnet`: residual CNN block stack; the usual architecture anchor is He et al., ResNet (2015).
- `vit`: patch embedding plus transformer-style vision; the usual anchor is Dosovitskiy et al.,
  ViT (2020).
- `rnn`, `lstm`: sequence modeling baselines over recurrent state.
- `transformer`: encoder-style sequence model; pairs naturally with the attention chapters.
- `gpt2`, `text_gpt2`: decoder-style language modeling; `text_gpt2` uses file-backed text.
- `mamba`: selective state-space model demo.
- `diffusion`: generative diffusion training loop.
- `ppo_*`: policy optimization in compact demonstration environments.
- `floats_arb_ieee_compare`: compares float semantics modes and pairs with *Floating-Point
  Semantics*.

Runnable via `torchlean`: `lake exe torchlean gpt_adder` (Karpathy-style single-CUDA-kernel addition learning;
not a `torchlean` subcommand).

Example invocations (CPU is always available; add `--cuda` after building with `-K cuda=true`):

```
lake exe torchlean mlp --cpu --steps 10
lake exe torchlean cnn --cpu --n-total 20 --steps 1
lake exe torchlean resnet --cuda --n-total 20 --steps 1
lake exe torchlean vit --cuda --n-total 20 --steps 1
lake exe torchlean rnn --cuda --tiny-shakespeare --steps 1
lake exe torchlean transformer --cuda --tiny-stories --steps 1
lake exe torchlean mamba --cuda --tiny-shakespeare --steps 25
lake exe torchlean diffusion --cpu --steps 5
lake exe torchlean gpt2 --cuda --steps 1
lake build -R -K cuda=true && lake exe torchlean gpt_adder --steps 50 --cuda
```

The real data helper prepares the small public corpora and CIFAR shards used by these examples:

`python3 scripts/datasets/download_example_data.py --tiny-shakespeare --tinystories-valid --cifar10 --cifar10-limit-train 200 --cifar10-limit-test 50`

Downloaded data is stored in `data/real` and is intentionally ignored by Git. Small checked-in toy
data remains in the [example data API](https://github.com/lean-dojo/TorchLean/tree/main/NN/Examples/Data/) so the repo can still run smoke
tests without network access.

These model examples matter because they show that TorchLean's public API is not limited to one
teaching example:

- MLPs, CNNs, attention blocks, and residual blocks share the same `NN.API` vocabulary,
- the `zero_grad / backward / step` rhythm can be written directly in Lean,
- and the same scripts can still switch dtype and backend from the CLI.

# Data Loading, Transforms, And Schedulers

The `NN/API/Data*` surface is richer than a first glance suggests. The current tutorial set covers
three increasingly realistic patterns:

- `csv_loader_train`
  Disk-backed tabular data, `Data.Transforms.Compose`, minibatch loaders, and a learning-rate
  schedule (`train.stepEpochLR`).
- `npy_loader_train`
  NumPy/PyTorch `.npy` interop with typed dataset reconstruction on the Lean side.
- `cifar10_npy_cnn_train`
  A more realistic offline image workflow: load `.npy`, split train/test, batch, train, and
  report accuracy.

Those are worth calling out because they answer a common question from PyTorch users:

> Is TorchLean only comfortable with in-memory tutorial tensors?

The answer is no. The examples cover CSV and `.npy` file readers, typed transforms,
deterministic batching, callbacks, and scheduler-driven training loops.

# GraphSpec And Export Demos

Two examples highlight newer architecture and interop work beyond the smallest tutorials.

## GraphSpec-authored training demo

- Lean demo: [GraphSpec tutorial API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Advanced/GraphSpec/Tutorial.lean)
- Command:
   `lake exe torchlean graphspec_mlp --backend compiled`

This demo defines an MLP once in GraphSpec, lowers it to the TorchLean training surface, and then
trains it with the ordinary public API. It is the fastest way to see GraphSpec as a real user
workflow rather than as a research-only subdirectory.

## TorchLean, IR, and Emitted PyTorch Code

- Lean demo: [Torch IR PyTorch export API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Advanced/TorchIRPyTorch.lean)
- Commands:
  `lake exe torchlean torch_ir_pytorch --arch mlp > exported_model.py`
  `lake exe torchlean torch_ir_pytorch --arch transformer > exported_model.py`

This demo sits at an important boundary in the repo:

- TorchLean model, compiled IR, and generated PyTorch code.

The supported architectures already include more than a single linear layer:

- `linear`, `mlp`, `sum`, `autoencoder`,
- `cnn`, `conv-mlp`,
- `mha`, `mha-mask`,
- `transformer`.

# BugZoo Case Studies

BugZoo deserves its own reading pass because it explains what kinds of mistakes TorchLean is trying
to make visible. The [BugZoo API](https://github.com/lean-dojo/TorchLean/tree/main/NN/Examples/BugZoo/) contains the checked cards, with the
full motivation in the [BugZoo overview](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/BugZoo/README.md).

The case studies are compact, but they cover a broad range:

- *Shape and broadcast*: missing axes and wrong broadcasts; TorchLean uses tensors indexed by shape
  and explicit broadcast evidence.
- *Stable loss*: unstable loss formulas and invalid domains; TorchLean names stable logit losses and
  safe-domain ops.
- *Ignored labels*: all-ignored label reductions; TorchLean uses an explicit contribution mask and
  empty-reduction policy.
- *Autograd domain*: masking after undefined division; TorchLean uses epsilon-protected division
  before masking.
- *Attention mask*: mask polarity and fully masked attention; TorchLean states exact causal-mask
  zero-weight properties.
- *Compiler boundary*: optimized graph wrong-code bugs; TorchLean asks for a source/target semantic
  preservation contract.
- *Float boundary*: real-proof versus float-deployment gaps; TorchLean names the bridge from runtime
  float32 to `IEEE32Exec`.
- *Normalization state*: BatchNorm formula/state mismatch; TorchLean makes epsilon placement and
  running statistics explicit.
- *Batch invariance*: serving output depends on batch composition; TorchLean states per-example
  batching behavior.
- *KV cache*: shifted or malformed decode cache; TorchLean names append-last key/value invariants.
- *RoPE position*: position mismatch during decoding; TorchLean names the position schedule.
- *Tokenizer boundary*: vocabulary/config mismatch; TorchLean represents token ids as
  `Fin vocabSize`.

These cards do not solve every production incident. They show a repeatable method: take a bug that
normally sits inside an opaque runtime, name the boundary, and attach a
small checked statement to it.

Useful build target:

```
lake build NN.Examples.BugZoo.All
```

For readers coming from ML systems, BugZoo is often the most concrete entry point: a recognizable
bug class, the exact object TorchLean uses as the contract, and the theorem or boundary that
follows.

# Verification Application Workflows

The verification side has also grown beyond one tiny IBP demo. The current example tree covers
several distinct application styles.

## Core TorchLean To IR Demos

- `torchlean-ibp`
  Tiny MLP, compile to IR, seed an input box, read off output bounds.
- `torchlean-crown-ops`
  Exercises a broader operator set, including softmax and loss-shaped examples.
- `torchlean-transformer-ibp`
  Tiny attention / encoder pipeline pushed through the same verifier path.

## Robustness and benchmark-style workflows

- `torchlean-robustness`
  Margin-certification demo over a TorchLean-authored model.
- `digits`
  Certified-accuracy style robustness demo.
- `vnncomp-mnistfc`
  A small VNN-COMP suite checker over exported JSON artifacts.

## PINN and ODE workflows

- `pinn-cert`
  Recompute-and-compare certificate check for a physics-informed neural network residual bound.
- `pinn-cli`
  Interactive PDE residual-bounding workbench with box splitting and backend selection.
- `pinn-dataset-check`
  Dataset-backed containment check against reference points.
- `ode`
  ODE enclosure verification for sub/super-solution style workflows.

Together, these examples show the range of the project: robustness checks, scientific-ML residual
bounds, and dynamical-system reasoning all use the same idea of a model plus a checked semantic
artifact.

# A Paper-Oriented Reading Path

To mirror the paper's example ordering, try:

1. `simple_mlp_train`
2. `pytorch_loop_mlp_train`
3. `graphspec_mlp_demo`
4. `torchlean-ibp`
5. `torchlean-crown-ops`
6. `pinn-cert` or `pinn-cli`
7. `ode`
8. `vnncomp-mnistfc`

That sequence starts with familiar training, then introduces typed authoring, then compiled IR,
then the verifier workflows that are specific to TorchLean's research program.

# Widgets In Practice

Some of the most useful examples are not command line programs at all. They are editor files that
render live views in the infoview.

The full widget family includes several views; the short list cited most often is:

- tensor views,
- graph and shape-inference views,
- float32 bit-layout views,
- verification-state views,
- autograd tape and gradient views.

Start with the [widgets example API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Advanced/Widgets.lean), then return here for
a compact map of each widget family.

# Smoke Tests

After edits, the fastest regression check for the public surface is:

`lake test`

# Adding A New Example

Checklist for adding a new example to the curated zoo:

- Put it in the right subtree:
  `NN/Examples/Models` for model zoo demos, `NN/Examples/Quickstart` for small
  tutorials, `NN/Examples/Advanced` for architecture/IR demos, and `NN/Examples/Verification` +
  `scripts/verification` for verifier workflows.
- If it belongs in the public runner, register it in the
  [model runner API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/Runner.lean) and make sure `NN.Examples.Zoo`
  imports it.
- Use the public facade where appropriate:
  `open NN.API` and stick to `nn`, `Data`, `train`, and `optim` unless the example is explicitly
  advanced.
- Add a short module docstring at the top with:
  1. what the example demonstrates, and
  2. the exact `lake env lean --run ...` command.
- Parse flags consistently:
  support `--seed` when randomness is involved, and use `train.run` to parse `--dtype` and
  `--backend`.
- Keep it cheap by default:
  small dataset, small model, few steps or epochs; heavier runs remain behind flags.

There is a central model runner for model demos (`lake exe torchlean ...`), while
advanced/tutorial files can still be run directly with `lake env lean --run ...`.

Command-line overview: *CLI Entry Points*. From examples into the verifier stack: *Graphs and IR*,
then *Verification*. For GPU build details, *Runtime and Autograd*.

# References and further reading

- George et al., *TorchLean* (2026): project overview. https://arxiv.org/abs/2602.22631
- Vaswani et al., *Attention Is All You Need* (2017): transformer baseline. https://arxiv.org/abs/1706.03762
- He et al., *Deep Residual Learning* (2015): ResNet. https://arxiv.org/abs/1512.03385
- Dosovitskiy et al., *An Image Is Worth 16x16 Words* (2020): ViT. https://arxiv.org/abs/2010.11929
- Ho, Jain, Abbeel, *Denoising Diffusion Probabilistic Models* (2020): DDPM. https://arxiv.org/abs/2006.11239
- Gu & Dao, *Mamba* (2023): selective state spaces. https://arxiv.org/abs/2312.00752
- Dao et al., *FlashAttention* (2022): IO-aware exact attention. https://arxiv.org/abs/2205.14135
- Odena et al., *TensorFuzz* (2019): fuzzing numerical failures. https://arxiv.org/abs/1807.10875
- Liu et al., *NNSmith* (2023): generated valid tests for DL compilers. https://arxiv.org/abs/2207.13066
- Schulman et al., *Proximal Policy Optimization* (2017): PPO. https://arxiv.org/abs/1707.06347
- PyTorch documentation: user-side analogy for `NN.API`. https://pytorch.org/docs/stable/index.html
