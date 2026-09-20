# Vision Examples

This folder contains runnable models over the prepared CIFAR-10 tensors. CIFAR is stored in a
channel-first layout, while the TorchLean tensor and layer APIs remain rank-polymorphic: convolution
and pooling consume a declared spatial suffix and preserve any leading axes.

## Files

- `Cnn.lean`: compact convolutional CIFAR-10 classifier. It uses a small crop so the command remains
  a practical runtime check while exercising real convolution and pooling data flow.
- `ResNet.lean`: residual classifier over an `8 × 8` crop, with global average pooling over both
  spatial axes. Architecture dimensions are source-level configuration values.
- `Vit.lean`: compact ViT-style CIFAR-10 classifier. It uses convolutional patch embedding,
  token reshape, an encoder stack configured in the source, learned positions, and class-slot pooling.

## Data

Prepare the small real CIFAR-10 subset with:

```bash
python3 scripts/datasets/download_example_data.py --cifar10
```

The loader reads `.npy` arrays under `data/real/cifar10/`. The examples crop the image tensors to
small typed shapes so they run quickly while still crossing the same data boundary as larger image
runs.

## Commands

```bash
lake exe torchlean cnn --n-total 1 --steps 1
lake exe torchlean resnet --n-total 1 --steps 1
lake exe torchlean vit --n-total 1 --steps 1
```

The same commands can target CUDA when TorchLean was built with CUDA support:

```bash
lake -R -K cuda=true exe torchlean cnn --device cuda --n-total 1 --steps 1
lake -R -K cuda=true exe torchlean resnet --device cuda --n-total 1 --steps 1
lake -R -K cuda=true exe torchlean vit --device cuda --n-total 1 --steps 1
```

## What To Inspect

These examples own the image-classification training path. Useful outputs are:

- the training loss and accuracy trace;
- the `TrainLog` JSON if `--log PATH` is passed;
- the typed image shapes in the Lean source;
- CPU/CUDA parity and regression evidence when changing image kernels.

For 3D detector certificates or projection verification, use `NN/Examples/Verification` and the
Geometry3D workflow. That path exports detector tensors as certificate artifacts, checks the camera
projection/enclosure conditions in Lean, and renders overlays for inspection.
