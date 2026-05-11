# Model Examples

This directory contains TorchLean model examples. The files are meant to be read like ordinary
training scripts: instantiate a model, prepare a loader or token stream, fit for several updates or
epochs, and optionally write a training curve.

For the narrative walkthrough, use the website guide. This page is the local command map.

## Directory Map

| Directory | Contents |
| --- | --- |
| `Common/` | shared real-data helpers used by several examples |
| `Supervised/` | MLP tabular regression and LSTM forecasting |
| `Vision/` | CNN, ResNet, and ViT image classifiers |
| `Sequence/` | RNN/LSTM checks, Transformer blocks, GPT-style models, Mamba, GPT-adder |
| `Generative/` | autoencoder, MAE, VAE, VQ-VAE, GAN, and diffusion examples |
| `Operators/` | FNO and operator-learning examples |
| `RL/` | DQN and PPO examples |
| `Runner.lean` | shared `lake exe torchlean ...` dispatcher |

## Data Preparation

Most examples use datasets prepared outside Lean and loaded through `NN.API.Data`.

```bash
python3 scripts/datasets/download_example_data.py --auto-mpg --cifar10 --tiny-shakespeare
python3 scripts/datasets/download_example_data.py --household-power --household-power-windows 512
```

For custom image folders or tensor archives, convert once to `.npy` with
`scripts/datasets/torchlean_data_convert.py`, then pass `--x` and `--y` to the relevant command.

## Training Curves

Most trainers accept `--log PATH`. The log is a `TrainLog` JSON file with metric names, steps,
values, and run metadata. Plot saved logs with:

```bash
python3 scripts/datasets/plot_trainlog.py data/model_zoo/*.json --out-dir plots/model_zoo
```

These logs are the right source for website plots. Samples and printed predictions are useful
qualitative checks, but claims about learning should point to the loss or accuracy curve.

## Supervised And Vision Runs

The tabular and vision examples use the same loader path: a dataset source is loaded, wrapped in a
typed minibatch loader, shuffled by epoch, and passed through the shared training loop. Here,
`--epochs` means passes over minibatches, not repeated updates on one fixed batch.

```bash
lake exe -K cuda=true torchlean mlp --cuda --epochs 100 --lr 0.003 \
  --log data/model_zoo/mlp_trainlog.json

lake exe -K cuda=true torchlean cnn --cuda --fast-kernels --n-total 2000 \
  --epochs 25 --lr 0.001 --log data/model_zoo/cnn_trainlog.json

lake exe -K cuda=true torchlean resnet --cuda --fast-kernels --n-total 2000 \
  --epochs 15 --lr 0.001 --log data/model_zoo/resnet_trainlog.json

lake exe -K cuda=true torchlean vit --cuda --fast-kernels --n-total 2000 \
  --epochs 10 --lr 0.001 --log data/model_zoo/vit_trainlog.json
```

The LSTM regression example trains on household-power windows and prints before/after forecast rows:

```bash
lake exe -K cuda=true torchlean lstm_regression --cuda --steps 200 --windows 96 \
  --log data/model_zoo/lstm_regression_trainlog.json
```

## Text Runs

Text models read a corpus, tokenize it, and build causal language-model windows with shifted targets.
The shared helpers live in `NN.API.Text`.

```bash
lake exe -K cuda=true torchlean mamba --cuda --fast-kernels --tiny-shakespeare \
  --steps 2000 --windows 384 --lr 0.004 --prompt "ROMEO:" --generate 260 \
  --temperature 0.75 --top-k 10 --sample-seed 11 \
  --log data/model_zoo/mamba_seq64_fixedsampler_2000.json

lake exe -K cuda=true torchlean gpt2 --cuda --fast-kernels --tiny-shakespeare \
  --steps 300 --windows 32 --lr 0.001 --prompt "ROMEO:" --generate 220 \
  --temperature 0.85 --top-k 24 --repeat-penalty 1.25 --repeat-window 24 \
  --sample-seed 11 --log data/model_zoo/gpt2_trainlog.json
```

`gpt2` here is a compact GPT-style causal Transformer, not a pretrained OpenAI checkpoint.

## Generative, Operator, And RL Runs

```bash
lake exe -K cuda=true torchlean diffusion --cuda --fast-kernels \
  --dataset cifar10 --n-total 800 --steps 200 --hidden-c 8 --T 100 --beta-end 0.12 \
  --sample-ppm data/model_zoo/diffusion_sample.ppm

lake exe -K cuda=true torchlean fno1d_burgers --cuda --steps 200 \
  --log data/model_zoo/fno1d_burgers_trainlog.json

lake exe torchlean mae --steps 25
lake exe torchlean gpt_adder --steps 1000 --a 7 --b 8
lake exe torchlean ppo_gridworld --updates 200
```

For 3D detector verification, use the verification examples rather than this model-training
directory. The geometry path exports detector tensors as certificates, checks the projection
envelope in Lean, and renders accepted/rejected overlays:

```bash
python3 scripts/verification/regenerate_assets.py --group geometry3d-wilddet3d --run
```
