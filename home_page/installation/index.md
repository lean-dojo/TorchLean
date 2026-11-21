---
title: Installation
redirect_from:
  - /backends/
---

## CPU installation

On Linux and macOS, you’ll need Git, Bash, Python 3, and a C/C++ compiler.

- On **Ubuntu or Debian**, run `sudo apt install git curl bash python3 build-essential`.
- On **macOS**, run `xcode-select --install` and make sure Python 3 is available. The CPU build
  works on Intel and Apple silicon.
- On **Windows**, run `wsl --install -d Ubuntu` in an administrator PowerShell. Open Ubuntu after
  setup and follow the Linux instructions. TorchLean supports Windows through WSL2.

Install [Elan](https://github.com/leanprover/elan) once to get the Lean version pinned by the
project.

```bash
curl https://elan.lean-lang.org/elan-init.sh -sSf | sh
```

Open a new terminal, then clone, build, and run a ten-step training example.

```bash
git clone https://github.com/lean-dojo/TorchLean.git
cd TorchLean
scripts/lake.sh exe cache get
scripts/lake.sh build
scripts/lake.sh exe torchlean quickstart_mlp --device cpu --steps 10
```

The cache download is optional. Use `scripts/lake.sh` in this checkout so CPU and CUDA
builds get separate caches. If you previously used raw Lake, move a real `.lake/build`
directory aside once before switching.

## NVIDIA CUDA

For Linux or WSL2, install the NVIDIA driver, a compatible CUDA toolkit, CMake 3.22 or
later, and Make. Download a CUDA-enabled [LibTorch SDK](https://docs.pytorch.org/cppdocs/installing.html)
from torch 2.13 or later. On WSL2, install the driver on Windows and the toolkit inside Ubuntu.
[NVIDIA CUDA on WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)

Point `TORCHLEAN_LIBTORCH_HOME` to the extracted SDK directory.

```bash
export TORCHLEAN_LIBTORCH_HOME=/absolute/path/to/libtorch
scripts/lake.sh -Kcuda=true build
scripts/lake.sh -Kcuda=true exe torchlean quickstart_mlp \
  --device cuda --steps 10 --show-backend
```

`-Kcuda=true` builds the executable with LibTorch. `--device cuda` runs the example on
the GPU and reports an error if CUDA is unavailable. See the [CUDA guide]({{ '/cuda/' | relative_url }})
for device settings, operation coverage, and the CUDA test command.

## Use TorchLean from another Lean project

Use the same Lean toolchain as TorchLean, then add this dependency to your `lakefile.lean`.

```lean
require TorchLean from git "https://github.com/lean-dojo/TorchLean.git" @ "main"
```

If you want to evaluate tensors with `#eval` or in the editor, add the CPU shared library to
your package declaration.

```lean
package MyProject where
  dynlibs := #[`@TorchLean/torchlean_tensor_cpu_shared]
```

Build from the root of your project using Lake.

```bash
lake update
lake exe cache get
lake build
```

## From a model to a kernel

Pass `--show-backend` to a model command to see which implementations it uses. The
[backend planner chapter]({{ '/blueprint/Runtime___-Autograd___-and-Interop/Inside-The-Backend-Planner/' | relative_url }})
explains how provider selection and CUDA parity checks affect a run.
