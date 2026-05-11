import VersoManual
import VersoBlueprint

open Verso.Genre Manual

#doc (Manual) "Modern Models and Training" =>
%%%
tag := "modern-models"
%%%

The model zoo is the point where TorchLean stops being a clean semantic story and starts being an
ML system. A two-layer MLP is enough to explain typed tensors and autograd. It is not enough to test
whether the framework can handle the shapes that modern ML actually uses.

The examples in this section are best read as semantic stress tests. Each model family exercises a
different part of the stack: residual branches, patch tokens, causal masks, scan state, diffusion
schedules, spectral kernels, RL trajectories, and CUDA runtime paths.

# What Each Family Stresses

The examples are compact, but each one touches a real source of ML complexity:

| Model family | What it stresses |
|---|---|
| MLP / CNN | typed tensors, losses, optimizers, data loaders |
| ResNet | residual shape agreement and branch joins |
| ViT | image patches, token dimensions, attention blocks |
| GPT-style models | causal windows, token ids, masks, save/load |
| Mamba | recurrent state, selective scan, prefix causality |
| Diffusion | stochastic schedules, denoising objectives, sampling artifacts |
| FNO | spectral convolution, scientific data, cuFFT boundary |
| PPO / RL | trajectories, environment boundary, policy/value losses |

A formal ML library that only works for one MLP is easy to make look clean. The harder test is
whether the same ideas survive residual sharing, attention masks, token windows, recurrent state,
stochastic sampling, spectral transforms, and external environments. That is what the model zoo is
for.

# Start With The Runner

Most model demos go through one command shape:

```
lake exe torchlean <demo> [flags...]
```

The subcommands are registered in the [model runner API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/Runner.lean), and
the model implementations are collected by the [model examples API](https://github.com/lean-dojo/TorchLean/tree/main/NN/Examples/Models/).

The runner is there so users can learn one command shape instead of remembering twenty separate
`lean --run` invocations.

# A Small Tour

The best first run is still a small one:

```
lake exe torchlean mlp --cpu --steps 10
```

Once CUDA has been built, the same style works on GPU:

```
lake build -R -K cuda=true
lake exe -K cuda=true torchlean mlp --cuda --steps 20
```

That one command already exercises the public API, the eager tape, the optimizer, and the selected
runtime backend. The rest of the zoo grows from that same shape.

# Feedforward, Convolutional, And Vision Models

The classical supervised examples are useful because their expected behavior is easy to recognize.

```
lake exe torchlean mlp --cpu --steps 10
lake exe torchlean cnn --cpu --n-total 20 --steps 1
lake exe -K cuda=true torchlean resnet --cuda --n-total 20 --steps 1
lake exe -K cuda=true torchlean vit --cuda --n-total 20 --steps 1
```

What these demonstrate:

- the same training loop works for vectors, images, residual blocks, and patch/attention-style
  vision models;
- model parameters are ordinary TorchLean parameters, not hidden Python objects;
- the examples can use prepared real data under `data/real` when available.

The commands above are intentionally small.  They are smoke runs, not leaderboard experiments.

The model code follows the same pattern as the running example:

```
import NN

open Spec
open NN.Tensor
open NN.API

def smallVisionHead : nn.M (nn.Sequential (Shape.Vec 64) (Shape.Vec 10)) :=
  nn.sequential![
    nn.linear 64 32 (pfx := Shape.scalar),
    nn.relu,
    nn.linear 32 10 (pfx := Shape.scalar)
  ]
```

The larger CNN, ResNet, and ViT examples replace this small head with convolution, residual, or
patch/attention blocks, but the training surface is the same.

For residual models, the semantic shape is:

$$`\operatorname{block}(x)=x+F_\theta(x)`

The theorem burden, when we prove one, is to show both branches have the same shape and that the
runtime graph really computes that sum rather than merely producing a tensor with compatible size.

# Sequence Models: RNN, LSTM, Transformer

The sequence examples give a gentle path from recurrent state to attention:

```
lake exe -K cuda=true torchlean rnn --cuda --tiny-shakespeare --steps 1
lake exe -K cuda=true torchlean lstm --cuda --tiny-shakespeare --steps 1
lake exe -K cuda=true torchlean transformer --cuda --tiny-stories --steps 1
```

A one-step run does not learn language. The useful fact is that the full data path exists:
text is loaded, token windows are constructed, tensors are built in Lean, and the model trains
through TorchLean's runtime.

# GPT-Style Language Models

TorchLean has two GPT-facing demos:

- `gpt2`
  a small GPT model suitable for smoke tests;
- `text_gpt2`
  a file-backed corpus trainer, including a path that can use GPT-2 BPE vocabulary and merges.

Typical commands:

```
lake exe -K cuda=true torchlean gpt2 --cuda --steps 1
lake exe -K cuda=true torchlean text_gpt2 --cuda \
  --data-file data/real/text/tinystories_valid.txt --allow-small-data --steps 100 --log-every 10
```

The `text_gpt2` example is explicit about scale. It is a TorchLean-native miniature
language model. It can overfit small windows and exercise a real tokenizer/data path, but it is not
OpenAI GPT-2-small and should not be described as pretrained GPT-2.

The important data path is:

- read a corpus file;
- tokenize into bounded ids;
- form causal windows;
- train next token prediction;
- optionally save parameters;
- reload the parameter file and prompt the saved model.

That is the part worth studying. A tiny local run will not chat like a large pretrained model, but it
does exercise the same kind of boundary TorchLean needs for serious sequence-model work.

The language model objective is the familiar next token map:

$$`\operatorname{logits}_t =
\operatorname{model}_\theta(tokens_{<t}),
\qquad
L = -\sum_t \log p_\theta(tokens_t \mid tokens_{<t})`

The small examples make token ids, causal windows, parameter files, and generation traces explicit
objects so later correctness work has something concrete to talk about.

# Mamba-Style State-Space Models

The Mamba demo exercises selective-scan style computation:

```
lake exe -K cuda=true torchlean mamba --cuda --tiny-shakespeare --steps 25
```

Under the CUDA backend, TorchLean includes native selective-scan kernels for the float32 path.  The
model is small, but the architectural lesson is real: not every modern sequence model is attention,
and TorchLean's runtime has begun to cover that broader operator family.

The state space shape is:

$$`h_{t+1}=A_t h_t+B_t x_t,
\qquad
y_t=C_t h_t+D_t x_t`

That equation is why scan order and prefix causality matter. The state space proof chapters state
that future tokens do not change prefix outputs for the supported scan definitions.

# Diffusion

The diffusion demo is a compact denoising training loop:

```
lake exe torchlean diffusion --cpu --steps 5
```

Diffusion is included because it stresses a different style of ML program: a stochastic-noise
schedule, a denoising objective, and a generative sampling path. It is a compact reference
implementation for the runtime and training API.

A useful way to read the diffusion example is as a four-step pipeline:

1. load real image tensors or a small fallback dataset;
2. sample a timestep and add noise according to the schedule;
3. train a denoiser to predict the noise or clean signal;
4. run a sampler such as DDIM-style reverse steps to produce an image artifact.

Data and logging belong beside the tutorial because a poor sample usually raises ordinary ML
debugging questions first: dataset scale, resolution, schedule length, model capacity, training
time, and sampler settings.

# FNO And Burgers Operator Learning

The FNO example is the best current example of TorchLean training on a real scientific-ML dataset.
The Python helper only downloads/converts data and plots the result; the model and training loop are
native TorchLean.

Prepare data with the [Burgers data helper](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Data/prepare_fno1d_burgers.py):

```
python3 NN/Examples/Data/prepare_fno1d_burgers.py --download --grid 32 --ntrain 128 --ntest 32
```

Train on CUDA:

```
lake exe -K cuda=true torchlean fno1d_burgers --cuda --fast-kernels --steps 700 --lr 0.003 \
  --plot-csv data/real/fno/predictions.csv
```

Plot one held-out prediction with the [plot helper](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Data/plot_fno1d_burgers.py):

```
python3 NN/Examples/Data/plot_fno1d_burgers.py --csv data/real/fno/predictions.csv
```

CUDA mode uses the fused cuFFT-backed `spectralConv1dRfft` autograd primitive.  CPU mode keeps a
portable dense-DFT reference path.  We use that split because one path is convenient for
inspection and the other is practical for training.

The operator learning claim has a different type from classification:

$$`\mathcal{G}_\theta :
\operatorname{function\ samples\ on\ a\ grid}
\to
\operatorname{function\ samples\ on\ a\ grid}`

The FNO example is valuable because it exercises scientific data, spectral transforms, and CUDA
interop while still using the same typed tensor and runtime layer.

# Reinforcement Learning

The PPO examples show that TorchLean is not limited to supervised losses:

```
lake exe torchlean ppo_gridworld --steps 100
lake exe torchlean ppo_cartpole --cuda --steps 100
```

The GridWorld demo is written in Lean and has proof hooks.  The Gymnasium examples cross a Python
environment boundary and therefore use a runtime contract to check observations, actions, rewards,
and termination flags before they enter the learner.

For RL, the useful TorchLean idea is not "PPO exists." It is that an episode can be treated as data:
observations, actions, rewards, done flags, log probabilities, and value estimates are explicit
records that can be replayed, logged, checked, and visualized.

# Data Is Part Of The Example

Several examples use real or semi-real data paths:

- CSV and `.npy` loaders for tabular and tensor datasets;
- CIFAR-style prepared shards under `data/real`;
- Tiny Shakespeare and TinyStories text;
- Burgers `.mat` conversion into `.npy` tensors.

The [example-data helper](https://github.com/lean-dojo/TorchLean/blob/main/scripts/datasets/download_example_data.py) prepares common tiny datasets:

```
python3 scripts/datasets/download_example_data.py --tiny-shakespeare --tinystories-valid --cifar10
```

TorchLean examples should make the data source visible. If a file is generated, downloaded, or
converted, put the command beside the example or in the module header. Reproducibility starts before the
first tensor is allocated.

# Demo Scope

A successful model zoo run establishes a concrete execution fact:

- the code builds;
- the selected runtime path executes;
- the loss and optimizer connect correctly;
- for training demos, the loss usually moves in the expected direction.

Architectural optimality, GPU-kernel agreement, and mathematical model theorems are separate
claims. Those belong in *Verification*, *Floating-Point Semantics*, and *GPU and CUDA*.

# References

- He et al., [*Deep Residual Learning for Image Recognition*](https://arxiv.org/abs/1512.03385),
  2015.
- Vaswani et al., [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762), 2017.
- Radford et al., [GPT-2 technical report / model card line](https://openai.com/index/better-language-models/),
  2019.
- Dosovitskiy et al., [*An Image is Worth 16x16 Words*](https://arxiv.org/abs/2010.11929), 2020.
- Ho et al., [*Denoising Diffusion Probabilistic Models*](https://arxiv.org/abs/2006.11239), 2020.
- Li et al., [*Fourier Neural Operator for Parametric Partial Differential
  Equations*](https://arxiv.org/abs/2010.08895), 2020/2021.
- Gu and Dao, [*Mamba: Linear-Time Sequence Modeling with Selective State
  Spaces*](https://arxiv.org/abs/2312.00752), 2023.
- Schulman et al., [*Proximal Policy Optimization Algorithms*](https://arxiv.org/abs/1707.06347),
  2017.
