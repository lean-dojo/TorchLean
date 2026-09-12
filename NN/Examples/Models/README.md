# Model Examples

This directory contains runnable TorchLean models, from a small MLP to attention and reinforcement
learning examples. Start with `mlp`; the larger models often require downloaded datasets or CUDA.
Each family README describes its inputs and outputs.

## Layout

| Directory | Workloads |
| --- | --- |
| `Supervised/` | MLP, KAN, and LSTM forecasting |
| `Vision/` | CNN, ResNet, and ViT |
| `Sequence/` | RNN, LSTM, Transformer, GPT-style models, and Mamba |
| `Generative/` | autoencoder, masked autoencoder, and diffusion |
| `Operators/` | Fourier neural operator examples |
| `RL/` | PPO and DQN examples |
| `Common/` | shared data and command plumbing |
| `../Runner.lean` | the application-wide command registry |

## Discover Commands

```bash
lake exe torchlean --help
lake exe torchlean --list
lake exe torchlean mlp --help
```

`--help` lists commands with descriptions; `--list` prints their names for scripts. Each command's
own `--help` explains its data and runtime options.

## Normal Training Path

Most examples use this training lifecycle:

```lean
let trainer := Trainer.new model config
let trained ← trainer.train dataset options
trained.printSummary
let prediction ← trained.predict input
```

The shared command helpers parse data and runtime flags around this lifecycle. They do not define a
second training API.

Examples with custom streams, reinforcement learning rollouts, or native fused steps use
`trainer.open` sessions to control each step directly.

## Prepare Data

```bash
python3 scripts/datasets/download_example_data.py --auto-mpg --cifar10 --tiny-shakespeare
python3 scripts/datasets/download_example_data.py --household-power --household-power-windows 512
```

For other formats:

```bash
python3 scripts/datasets/torchlean_data_convert.py --help
```

The data guide is `NN/Examples/Data/README.md`.

## Representative Runs

```bash
lake exe torchlean mlp --device cpu --steps 10
lake exe torchlean rnn --device cpu --steps 1
lake -R -K cuda=true exe torchlean cnn --device cuda --n-total 1 --steps 1
lake -R -K cuda=true exe torchlean gpt2 --device cuda --steps 1 --generate 0
```

Pass `--log PATH` when the training curve matters. `TrainLog` JSON is the stable quantitative
artifact; printed predictions and generated samples are previews.

## Verification Boundary

A model command trains, predicts, and exports artifacts. It does not prove that a generated result
is correct. When an artifact enters a formal claim, the relevant checker or theorem under
`NN/Verification`, `NN/Proofs`, or `NN/MLTheory` must state the checked property.

Compile the examples with `lake build NNExamples`, then run the affected model command on a
small dataset. Retain permanent tests for numerical and native-boundary behavior; use temporary
checks for routine command changes.
