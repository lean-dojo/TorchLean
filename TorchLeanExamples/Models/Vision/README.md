# Vision Examples

The three models here classify the prepared CIFAR-10 tensors. CIFAR is stored channel-first, and
the TorchLean tensor and layer APIs are rank-polymorphic: convolution and pooling consume a
declared spatial suffix and keep any leading axes.

## Files

- `CNN.lean`: compact convolutional CIFAR-10 classifier on an `8 × 8` crop, small enough for a quick
  runtime check that runs real convolution and pooling.
- `ResNet.lean`: residual classifier over an `8 × 8` crop, with global average pooling over both
  spatial axes. Architecture dimensions are source-level configuration values.
- `ViT.lean`: compact ViT-style CIFAR-10 classifier on a `4 × 4` crop. It uses convolutional patch
  embedding, token reshape, an encoder stack configured in the source, learned positions, and
  class-token pooling.

## Data

Prepare the small real CIFAR-10 subset with:

```bash
python3 scripts/datasets/download_example_data.py --cifar10
```

The loader reads `.npy` arrays under `data/real/cifar10/`. The examples crop the images to small
typed shapes, so they run quickly and cross the same data boundary as larger image runs.

## Commands

```bash
scripts/lake.sh exe torchlean cnn --n-total 1 --steps 1
scripts/lake.sh exe torchlean resnet --n-total 1 --steps 1
scripts/lake.sh exe torchlean vit --n-total 1 --steps 1
```

With `-Kcuda=true` and `--device cuda`, the same commands run on CUDA:

```bash
scripts/lake.sh -Kcuda=true exe torchlean cnn --device cuda --n-total 1 --steps 1
scripts/lake.sh -Kcuda=true exe torchlean resnet --device cuda --n-total 1 --steps 1
scripts/lake.sh -Kcuda=true exe torchlean vit --device cuda --n-total 1 --steps 1
```

## What To Inspect

When you change the image-classification training path, check:

- the training loss before and after
- the `TrainLog` JSON, when `--log PATH` is passed
- the typed image shapes in the Lean source
- CPU and CUDA parity, and regression evidence when the change touches image kernels

Geometry3D detector certificates live elsewhere. Scripts in
`TorchLeanExamples/Verification/Geometry3D/` export detector outputs as certificate artifacts and
render overlays for inspection, and `scripts/lake.sh exe verify -- camera-box3d-cert` checks the
camera projection and box enclosure conditions in
`TorchLeanExamples/Verification/Geometry3D/Box3D.lean`.
