# Generative Examples

Here we train an autoencoder, reconstruct masked image patches, and generate images with diffusion.
Each command uses a small model and can save its training log; diffusion can also write PPM images.

## Files

- `Autoencoder.lean`: compact vector autoencoder over real CIFAR image batches.
- `Mae.lean`: ViT-MAE-style masked autoencoder path: patch masking, transformer tokens, and image
  reconstruction.
- `Diffusion.lean`: compact unconditional diffusion command with noise schedule, denoiser training,
  DDIM-style replay, and optional PPM image artifacts.

## Data

Most commands use CIFAR-10 through the shared real-data loader:

```bash
python3 scripts/datasets/download_example_data.py --cifar10
```

The diffusion command also supports ImageNet-style folders converted to `.npy`:

```bash
python3 scripts/datasets/torchlean_data_convert.py image-folder \
  --input /path/to/imagenet/train \
  --x-output data/real/imagenet64/imagenet64_train_X.npy \
  --y-output data/real/imagenet64/imagenet64_train_y.npy \
  --height 64 --width 64 --labels-from-dirs --limit 800
```

## Commands

Quick CPU checks:

```bash
scripts/lake.sh -Kcuda=false exe torchlean autoencoder --device cpu --steps 1 --n-total 1
scripts/lake.sh -Kcuda=false exe torchlean mae --device cpu --steps 1 --n-total 1
scripts/lake.sh -Kcuda=false exe torchlean diffusion --device cpu \
  --dataset cifar10 --n-total 1 --steps 1 --hidden-c 2 --T 2
```

The same commands accept `--device cuda` in a CUDA-enabled build. A longer CUDA diffusion run with
visual artifacts is:

```bash
scripts/lake.sh -Kcuda=true exe torchlean diffusion --device cuda \
  --dataset cifar10 --n-total 800 --steps 200 --hidden-c 8 --T 100 --beta-end 0.12 \
  --reference-ppm data/examples/diffusion_reference.ppm \
  --noisy-ppm data/examples/diffusion_noisy.ppm \
  --reconstruct-ppm data/examples/diffusion_reconstruct.ppm \
  --sample-ppm data/examples/diffusion_sample.ppm \
  --log data/examples/diffusion_trainlog.json
```

## Outputs

| Command | Training target | Written artifacts |
| --- | --- | --- |
| `autoencoder` | Reconstruct the first 16 flattened CIFAR values from a four-value latent representation | Training log JSON and terminal summary |
| `mae` | Reconstruct the masked coordinates of a `3 × 4 × 4` CIFAR crop | Training log JSON and terminal summary |
| `diffusion` | Predict injected noise at a sampled diffusion timestep | Training log JSON and optional reference, noisy, reconstructed, and sampled PPM images |

A falling reconstruction or noise-prediction loss measures this training task. It does not certify
image quality, generalization, or agreement between the LibTorch CUDA backend and a mathematical model.
These commands do not emit verification certificates; any later checked claim needs a separate
artifact format and checker.
