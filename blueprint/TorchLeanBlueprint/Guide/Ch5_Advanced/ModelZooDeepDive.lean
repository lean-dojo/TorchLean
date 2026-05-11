import VersoManual
import VersoBlueprint

open Verso.Genre Manual

#doc (Manual) "Model Zoo Deep Dive" =>
%%%
tag := "model-zoo-deep-dive"
%%%

The model zoo exercises TorchLean as a whole system. A tensor library can look good on
an MLP. A graph IR can look good on a two layer verifier fixture. A CUDA backend can look good on a
single matmul. The zoo asks a harder question: can the same API, runtime, graph boundary, data
helpers, and verification boundaries survive modern model shapes that users actually recognize?

We built the [model examples API](https://github.com/lean-dojo/TorchLean/tree/main/NN/Examples/Models/) to answer that question
as an executable model suite rather than a benchmark leaderboard. The default runs are compact, but
the stress they apply is real: causal windows, residual branches, patch tokens, selective scan,
spectral convolution, stochastic denoising, replay buffers, PPO rollouts, Python environment
bridges, CUDA fast paths, and graph/verification boundaries that must remain explicit.

Read each model along five axes:

- *API*: does the public model surface express the architecture?
- *Runtime*: can the example train, log, save, load, or sample?
- *Graph*: does the graph/IR story remain visible?
- *CUDA*: does the accelerated path match the intended runtime boundary?
- *Verification*: does the example point to a spec, theorem, checker, or explicit producer
  assumption?

# Model Families And Stress Tests

Most demos are dispatched by [NN.Examples.Models.Runner API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/Runner.lean):

```
lake exe torchlean <example> [flags...]
```

That runner keeps the public surface stable while each model file owns its data path, model
constructor, optimizer, logs, and caveats. When the docs say "run `gpt2`", the code path is not
hidden in a script; it ends in a namespaced Lean `main`.

The zoo is organized by stressor:

- *GPT-2 style and CharGPT*: causal language model commands stress causal windows, tokenization,
  generation probes, save/load, and logs.
- *Mamba*: the selective scan demo stresses sequence modeling without attention, recurrent state,
  and CUDA paths for selective scan.
- *RNN/LSTM/GRU style recurrence*: recurrent text and forecasting demos stress recurrent state,
  gated cells, short text windows, and forecasting data.
- *ResNet*: the residual vision classifier stresses residual wiring, convolution, batch data, and
  graph shape discipline.
- *ViT*: the patch-token vision transformer stresses image patches, token sequences, and attention
  blocks.
- *FNO*: the Burgers operator learning demo stresses scientific data, spectral kernels, and a CUDA
  primitive backed by cuFFT.
- *Diffusion and latent generators*: the generative examples stress stochastic schedules,
  reconstruction targets, and generator/discriminator losses.
- *RL demos*: PPO and replay examples stress trajectories, environment boundaries, replay, and
  policy/value losses.

There is no separate GRU command in the runner. We still discuss GRU in this section as
part of the gated recurrent design space: TorchLean's present runnable recurrent coverage is RNN,
LSTM, and LSTM forecasting; a future GRU demo would stress the same typed recurrent state and
runtime loop boundary with a smaller gate set.

The common model zoo contract is that a demo should make the data boundary, model, loss, optimizer,
and logs visible. It does not need to be large to be valuable.
It needs to make clear what was run, what data entered, what state changed, and which semantic
contract the example points back to.

The architecture equations are familiar, but the TorchLean reading is specific:

- causal language models estimate $`p_\theta(x_t\mid x_{<t})` and rely on the mask property
  $`j>i\Rightarrow A_{ij}=0`;
- residual blocks compute $`x\mapsto F_\theta(x)+S_\theta(x)`, so both branches must agree on
  output shape;
- state-space models use $`h_{t+1}=A_t h_t+B_t x_t` and $`y_t=C_t h_t+D_t x_t`, so prefix
  causality is a real contract;
- FNO layers use spectral convolution, informally
  $`u\mapsto \mathcal F^{-1}(R_\theta\cdot \mathcal F(u))`, which is why the cuFFT boundary matters.

# GPT-2-Style Models And CharGPT

The GPT-facing surfaces are:

- [NN.Examples.Models.Sequence.Gpt2 API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/Sequence/Gpt2.lean), a small
  byte level GPT-2 style causal language model.
- [NN.Examples.Models.Sequence.TextGpt2 API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/Sequence/TextGpt2.lean), a
  trainer over a text corpus that can use GPT-2 style BPE assets.
- [NN.Examples.Models.Sequence.CharGpt API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/Sequence/CharGpt.lean), a
  minGPT style character model with deterministic random windows and generation probes.
- [NN.Examples.Models.Sequence.Gpt2Saved API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/Sequence/Gpt2Saved.lean),
  which exercises saved parameter loading and prompting.
- [NN.Examples.Models.Sequence.GptAdder API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/Sequence/GptAdder.lean), a
  minGPT style algorithmic addition task that is especially useful for CUDA smoke testing.

A typical small run is:

```
python3 scripts/datasets/download_example_data.py --tiny-shakespeare
lake exe -K cuda=true torchlean gpt2 --cuda --fast-kernels \
  --tiny-shakespeare --steps 100 --windows 128 --prompt "ROMEO:"
```

For CharGPT:

```
lake exe -K cuda=true torchlean chargpt --cuda --tiny-shakespeare \
  --steps 200 --prompt "ROMEO:"
```

These examples stress the API in a way MLPs do not. The model shape depends on vocabulary, sequence
length, number of heads, head dimension, and transformer depth. The data path has to turn raw text
into bounded token windows. The runtime has to carry a long eager tape through embeddings, attention
blocks, layer style operations, and output logits. The logging path writes before and after loss and
decoded probes so a user can see whether the tiny local model is moving in the right direction.

The semantic core is still a typed next token map:

$$`\mathrm{tokens}_{0:T}
\longmapsto
\mathrm{logits}_{0:T,\;0:|\mathcal V|}`

Masking and positions are not decoration. A causal mask says token `i` may not attend to future
token `j>i`; RoPE or positional embeddings say which position each key and query belongs to. Those
are exactly the kinds of bugs the BugZoo and CUDA chapters keep naming.

They also stress the boundary between "GPT style" and "GPT-2". TorchLean's `gpt2` command is not
OpenAI's pretrained GPT-2-small. It is a small causal Transformer with GPT-2-like ingredients,
chosen so the Lean side runtime, tokenizer path, parameter handling, and CUDA hooks can be exercised
locally. The right citations for the architecture lineage are Radford et al., "Language Models are
Unsupervised Multitask Learners" (OpenAI technical report, 2019,
[PDF](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)),
Vaswani et al., "Attention Is All You Need" (NeurIPS 2017, https://arxiv.org/abs/1706.03762), and
Karpathy's minGPT educational implementation (https://github.com/karpathy/minGPT) for the compact
training demo style.

# Mamba

[NN.Examples.Models.Sequence.Mamba API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/Sequence/Mamba.lean) keeps the
sequence coverage broader than attention. The Mamba command trains a byte level language model through
the public Mamba API:

```
python3 scripts/datasets/download_example_data.py --tiny-shakespeare
lake exe -K cuda=true torchlean mamba --cuda --fast-kernels \
  --tiny-shakespeare --steps 200 --windows 128 --prompt "ROMEO:"
```

The specification side includes state space material that later theorems can use in
[NN.Spec.Models.Mamba API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Spec/Models/Mamba.lean), including run-list length facts for
compact Mamba style blocks. The API side in [NN.API.Models.Mamba API](https://github.com/lean-dojo/TorchLean/blob/main/NN/API/Models/Mamba.lean)
builds a trainable model with a gated recurrent core. The runtime side can use CUDA fast paths for
the float32 selective scan family when built with CUDA.

This example is valuable because it stresses a different runtime shape: recurrent state, scan
computation, gating, and generation. Attention bugs are not the only sequence bugs. Mamba style
models make us care about state update conventions, scan order, and assumptions for specific kernels.

The paper anchor is Gu and Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
(2023, https://arxiv.org/abs/2312.00752).

At the level of a mental model, the recurrent scan is:

$$`h_{t+1}=A_t h_t + B_t x_t,\qquad
y_t=C_t h_t + D_t x_t`

The details differ by implementation, but the proof pressure is the same: state shape, scan order,
and parameter broadcasting must be explicit.

# RNN, LSTM, And The GRU Gap

The current runnable recurrent examples are:

- [NN.Examples.Models.Sequence.Rnn API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/Sequence/Rnn.lean), a vanilla RNN
  text-window runtime check.
- [NN.Examples.Models.Sequence.Lstm API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/Sequence/Lstm.lean), an LSTM
  text-window runtime check.
- [NN.Examples.Models.Supervised.LstmRegression API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/Supervised/LstmRegression.lean),
  a more natural LSTM forecasting tutorial over household power windows.

The small text commands are intentionally short:

```
lake exe -K cuda=true torchlean rnn --cuda --tiny-shakespeare --steps 1
lake exe -K cuda=true torchlean lstm --cuda --tiny-shakespeare --steps 1
```

The forecasting run is the better tutorial:

```
python3 scripts/datasets/download_example_data.py --household-power --household-power-windows 512
lake exe -K cuda=true torchlean lstm_regression --cuda --steps 200 --windows 96
```

These examples stress recurrent state handling: hidden state shape, application at each time step,
heads distributed across time, and optimizer updates through unfolded computation. The LSTM regression
case also stresses real time series data and a loss that readers can interpret as forecasting
error, not just token reconstruction.

A recurrent cell has the shape:

$$`(h_t,x_t)\longmapsto(h_{t+1},y_t)`

An LSTM cell carries more state:

$$`(h_t,c_t,x_t)\longmapsto(h_{t+1},c_{t+1},y_t)`

That extra carried state is why recurrent examples are good documentation tests: framework code may
store the state implicitly, while a proof needs to reason about the actual transition that ran.

There is no `gru` subcommand today. That is worth saying plainly because "LSTM/GRU/RNN" is often
used as one category in ML writing. TorchLean covers vanilla RNN and LSTM executables;
GRU is a natural extension point rather than a file currently cited as present. The classic recurrent
citations are Hochreiter and Schmidhuber, "Long Short-Term Memory" (Neural Computation 1997,
https://www.bioinf.jku.at/publications/older/2604.pdf) and Cho et al., "Learning Phrase
Representations using RNN Encoder Decoder for Statistical Machine Translation" (2014,
https://arxiv.org/abs/1406.1078) for GRU style gating.

# ResNet

[NN.Examples.Models.Vision.Resnet API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/Vision/Resnet.lean) trains a
compact ResNet style classifier over CIFAR arrays:

```
python3 scripts/datasets/download_example_data.py --cifar10
lake exe -K cuda=true torchlean resnet --cuda --n-total 20 --steps 1
```

The spec side in [NN.Spec.Models.Resnet API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Spec/Models/Resnet.lean) is richer than a
smoke command. It defines residual blocks with identity/projection shortcuts, proves well formedness
facts for ResNet style configs, records convolution output size facts, and gives forward/backward
structure for a basic residual block.

ResNet stresses the graph boundary because residual connections create branching structure that later rejoins.
A linear stack can postpone many mistakes; a residual add makes shape agreement explicit. If the
shortcut branch and the convolution branch disagree, the model does not merely train poorly; the
typed construction has to confront the mismatch. That pressure is useful before verifiers consume
lowered graphs.

The residual block contract is:

$$`x \longmapsto F_\theta(x)+S_\theta(x)`

Both branches must land in the same shape before the addition exists. That is a small sentence in
the prose, but it is one of the main reasons typed architectures help.

The paper anchor is He et al., "Deep Residual Learning for Image Recognition" (CVPR 2016,
https://arxiv.org/abs/1512.03385).

# Vision Transformers

[NN.Examples.Models.Vision.Vit API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/Vision/Vit.lean) trains a compact
ViT style CIFAR classifier:

```
python3 scripts/datasets/download_example_data.py --cifar10
lake exe -K cuda=true torchlean vit --cuda --n-total 20 --steps 1
```

The corresponding spec material in [NN.Spec.Models.Vit API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Spec/Models/Vit.lean) names
ViT configurations and proves well formedness for standard patch 16 config records. The executable
uses much smaller dimensions, but it stresses the same class of concerns: converting images to
patches, token embeddings, dimensions compatible with attention, and classifier heads.

ViT is a useful bridge between the vision and sequence parts of TorchLean. It asks the image loader
to produce CIFAR shaped tensors, then asks the model API to treat patches as tokens. That crossover
is a common source of shape bugs in ordinary frameworks.

The shape contract is:

$$`\mathrm{image}(C,H,W)
\longmapsto
\mathrm{patches}(N_{\mathrm{patch}},d_{\mathrm{model}})
\longmapsto
\mathrm{tokens}`

The verifier and runtime should not have to guess which image layout or patch convention was used.

The paper anchor is Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image
Recognition at Scale" (ICLR 2021, https://arxiv.org/abs/2010.11929).

# FNO

[NN.Examples.Models.Operators.Fno1dBurgers API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/Operators/Fno1dBurgers.lean)
is the zoo's scientific ML example. It trains a Fourier neural operator on a prepared one dimensional
Burgers dataset:

```
python3 NN/Examples/Data/prepare_fno1d_burgers.py --download --grid 32 --ntrain 128 --ntest 32

lake exe -K cuda=true torchlean fno1d_burgers --cuda --fast-kernels \
  --steps 700 --lr 0.003 --plot-csv data/real/fno/predictions.csv
```

FNO stresses a runtime boundary that language and vision models do not: spectral convolution. The
CPU path uses a portable dense DFT reference implementation, while the CUDA run can use a fused
`spectralConv1dRfft` autograd primitive backed by cuFFT. That split gives us a readable reference
path for inspection and a practical path for training.

The operator learning mental model is:

$$`u_0(\cdot)\longmapsto u_T(\cdot)`

The FNO does not predict one label for one image; it learns an operator between function-like
signals. That is why the example belongs next to Scientific ML verification rather than only next
to ordinary supervised classification.

The example also stresses artifact hygiene. A helper prepares `.npy` arrays, Lean trains the model,
and a plotting helper renders a prediction CSV. The Python scripts are data/artifact utilities, not
the owner of the model semantics.

The paper anchor is Li et al., "Fourier Neural Operator for Parametric Partial Differential
Equations" (ICLR 2021, https://arxiv.org/abs/2010.08895).

# Diffusion And Other Generative Commands

The generative commands are:

```
lake exe torchlean autoencoder --steps 10
lake exe torchlean mae --steps 10
lake exe torchlean vae --steps 10
lake exe torchlean vqvae --steps 10
lake exe torchlean gan --steps 10
lake exe torchlean diffusion --steps 5
```

These examples stress stochastic and reconstruction oriented training. Diffusion adds timestep
schedules, noising, denoising, and sampling artifacts. VAE, VQ-VAE, and GAN examples connect a
runnable training surface to objective definitions used by theorems in the
[generative theory API](https://github.com/lean-dojo/TorchLean/tree/main/NN/MLTheory/Generative/). The companion generative guide explains
that boundary in more detail.

These examples are not just "more MLPs." They force the API to express latent shapes,
reconstruction targets, score targets, generator/discriminator pairs, patch masks, and image valued
sampling outputs.

# RL Demos

The RL examples are different from the supervised and generative zoo because the data is not just a
static tensor file. It is produced by an environment boundary:

- [NN.Examples.Models.RL.PPOCartPole API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/RL/PPOCartPole.lean) trains an
  actor critic on Gymnasium `CartPole-v1` through a Python subprocess bridge.
- [NN.Examples.Models.RL.PPOGridWorld API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/RL/PPOGridWorld.lean) gives a
  smaller controlled PPO environment.
- [NN.Examples.Models.RL.PPOPongRam API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/RL/PPOPongRam.lean) exercises a
  larger observation/action setting.
- [NN.Examples.Models.RL.DQNReplay API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/RL/DQNReplay.lean) focuses on the
  off policy replay and minibatch update pieces.

For CartPole:

```
lake exe -K cuda=true torchlean ppo_cartpole --cuda --updates 50
```

For the DQN replay mini example:

```
lake exe torchlean dqn_replay
```

RL stresses producer/checker boundaries harder than ordinary supervised learning. TorchLean can
typecheck the tensors that enter training, define the policy/value losses, and write logs that
widgets can read. When Gymnasium is the environment, raw observations are checked against a runtime
boundary contract before becoming training data. TorchLean verifies what enters the Lean side, while
the external simulator remains a named producer.

The paper anchors are Mnih et al., "Human-level control through deep reinforcement learning"
(Nature 2015, https://www.nature.com/articles/nature14236) for DQN and Schulman et al., "Proximal
Policy Optimization Algorithms" (2017, https://arxiv.org/abs/1707.06347) for PPO. The environment
API reference is Gymnasium by the Farama Foundation (https://gymnasium.farama.org/).

The interaction loop is:

$$`s_t \xrightarrow{\pi_\theta} a_t
\xrightarrow{\mathrm{env}} (r_t,s_{t+1})`

TorchLean can type and check the policy, rollout buffers, returns, and PPO losses. If the
environment is Gymnasium, the environment transition itself remains an external producer step.

# API, Runtime, Graph, CUDA, Verification

The model zoo is best read as five overlapping tests:

- *API*: can modern models be expressed as typed Lean constructors rather than opaque Python
  modules?
- *Runtime*: can eager autograd, optimizers, logs, save/load paths, and data loaders run end to end?
- *Graph*: can branchy, tokenized, recurrent, and spectral programs be lowered or related to shared
  graph concepts when needed?
- *CUDA*: can GPU execution accelerate examples without changing their public model shape?
- *Verification*: can examples point to specs used by theorems and make any remaining producer or
  runtime assumptions explicit?

The answer is intentionally layered. MLP, CNN, ResNet, ViT, GPT, Mamba, FNO, diffusion, and RL
commands all exercise real runtime surfaces. Some model families also have dedicated spec or theory
declarations with citeable theorem names. Not every command has full verification. Not every CUDA
kernel has a fully proved equivalence to a high level spec. Not every external data source is
trusted.

That layered view is a feature of the zoo. Readers can see where TorchLean is already a proof
artifact, where it is a runnable ML artifact, and where the bridge between those two worlds is still
being strengthened.
