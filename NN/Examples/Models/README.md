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
| `Operators/` | Fourier neural operators, complex regression, and physics-informed neural fields |
| `RL/` | PPO and DQN examples |
| `Common/` | shared data and command plumbing |
| `../Runner.lean` | the application-wide command registry |

## Discover Commands

```bash
scripts/lake.sh exe torchlean --help
scripts/lake.sh exe torchlean --list
scripts/lake.sh exe torchlean mlp --help
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

`Common/RealData.lean` loads prepared CIFAR-10 and ImageNet-style tensors and local text corpora,
crops images, builds typed minibatches, and reports missing-data preparation steps.
`Common/Train.lean` adapts CSV, NPY, and forecasting commands to
`TorchLean.CLI.Training.Command`. Flag parsing comes from `TorchLean.CLI.Trainer`; training logs
come from `TorchLean.Training`. Dataset scripts write under `data/`, and the helpers pass those
files through typed loaders to the public trainer. Model definitions stay in their example files.

Diffusion and FNO use `trainer.trainStream` for indexed samples and evaluation callbacks.
The PPO examples use the specialized `rl.ppo` runtime, while CharGPT uses the `Module` API for
mixed-dtype token inputs and floating-point parameters.

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
scripts/lake.sh exe torchlean mlp --device cpu --steps 10
scripts/lake.sh exe torchlean rnn --device cpu --steps 1
scripts/lake.sh -Kcuda=true exe torchlean cnn --device cuda --n-total 1 --steps 1
scripts/lake.sh -Kcuda=true exe torchlean gpt2 --device cuda --steps 1 --generate 0
```

Pass `--log PATH` when the training curve matters. `TrainLog` JSON is the stable quantitative
artifact; printed predictions and generated samples are previews.

## Verification Boundary

A model command trains, predicts, and exports artifacts. It does not prove that a generated result
is correct. When an artifact enters a formal claim, the relevant checker or theorem under
`NN/Verification`, `NN/Proofs`, or `NN/MLTheory` must state the checked property.

Compile the examples with `scripts/lake.sh build NNExamples`, then run the affected model command on
a small dataset. Retain permanent tests for numerical and native-boundary behavior; use temporary
checks for routine command changes.

See [contributing](../../../docs/CONTRIBUTING.md) for API and validation conventions.
