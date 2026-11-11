---
title: Installation
layout: default
redirect_from:
  - /backends/
---

# Installation

## CPU Installation

TorchLean's build wrapper requires Bash and Python 3. Before building, install the compiler tools
described for your platform below.

Optional [dataset preparation](https://github.com/lean-dojo/TorchLean/tree/main/scripts/datasets)
requires Python 3.12+ and NumPy (WikiText also needs PyArrow). You don't need these packages to
build the CPU library or run its offline examples.

First install [Elan](https://github.com/leanprover/elan), the Lean toolchain manager. On Linux or
macOS:

```bash
curl https://elan.lean-lang.org/elan-init.sh -sSf | sh
```

Open a new terminal so that `elan`, `lean`, and `lake` are on your `PATH`, then clone and build
TorchLean:

```bash
git clone https://github.com/lean-dojo/TorchLean.git
cd TorchLean
scripts/lake.sh exe cache get
scripts/lake.sh build
```

The cache command downloads compatible prebuilt Lean dependencies when they're available. It's safe
to omit (`scripts/lake.sh build` compiles anything that is missing).

Use `scripts/lake.sh` for commands in this checkout: it keeps CPU and CUDA artifacts in separate
cache directories and holds a checkout lock while Lake runs. If you've built with raw Lake and
`.lake/build` is a real directory, move it aside once before using the wrapper.

Run a small model to check the executable path:

```bash
scripts/lake.sh exe torchlean quickstart_mlp --device cpu --steps 10
```

If those commands succeed, TorchLean is installed. You can inspect the available examples and
verification commands with:

```bash
scripts/lake.sh exe torchlean --help
scripts/lake.sh exe verify --help
```

For GPU execution, build with LibTorch as described below (the Lean model stays the same).

| Platform | CPU | NVIDIA GPU | LibTorch provider | Status |
| --- | --- | --- | --- | --- |
| Linux | &#10003; | &#10003; Through LibTorch | Tensor operations and local gradients | Supported |
| macOS, Intel or Apple silicon | &#10003; | Not applicable | Not implemented | CPU supported, GPU unsupported |
| Windows with WSL2 | &#10003; Linux path | &#10003; CUDA on WSL2 | Linux path | Recommended Windows setup |
| Native Windows | Unsupported | Not validated | Not wired | Unsupported (use WSL2) |

LibTorch is the standard CUDA backend: build with `-Kcuda=true` and run with `--device cuda`. You
can also request `--device gpu` (or `device := .cuda` in Lean), which selects
CUDA through LibTorch and fails if that GPU runtime is unavailable instead of falling back to CPU.
A plain build uses the portable CPU runtime and doesn't link LibTorch. On the GPU, TorchLean uses
ATen (LibTorch's tensor library) but retains its own tape, backward traversal, and optimizer state.
Native operations compute tensor values and local gradients with LibTorch autograd recording
disabled. The CPU build remains independent of LibTorch.

## Linux

### CPU

You need Git, `curl`, Bash, Python 3, and a C/C++ compiler. On Ubuntu or Debian:

```bash
sudo apt update
sudo apt install -y git curl bash python3 build-essential
```

Then follow the CPU installation steps above. By default the build uses the portable CPU runtime,
plus one unavailable-backend shim that supplies the GPU symbols so CPU-only machines can compile
the complete Lean project. GPU requests fail with instructions to rebuild with LibTorch.

### NVIDIA CUDA

Install a supported NVIDIA driver and CUDA toolkit using NVIDIA's [CUDA Installation Guide for
Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/). Download a CUDA-enabled SDK
of torch 2.13 or later from the [LibTorch installation
page](https://docs.pytorch.org/cppdocs/installing.html), using its
supported C++ compiler and matching CUDA toolkit (the SDK must contain `include/`, `lib/`, and
`share/cmake/Torch/TorchConfig.cmake`). To build the bridge, you need CMake 3.22 or later, Make,
and Python 3. Check the machine before rebuilding:

```bash
nvidia-smi
nvcc --version
```

Build and run the CUDA configuration, pointing to the extracted SDK:

```bash
export TORCHLEAN_LIBTORCH_HOME=/absolute/path/to/libtorch
scripts/lake.sh -Kcuda=true build
scripts/lake.sh -Kcuda=true exe torchlean quickstart_mlp \
  --device cuda --steps 10 --show-backend
```

The two CUDA choices happen at different times: `-Kcuda=true` tells Lake to compile the C++
adapters and link LibTorch, and `--device cuda` asks the executable to use it. A CPU-linked
executable rejects `--device cuda` instead of silently moving the run back to the CPU.

Our wrapper selects the CUDA build cache and passes `-R` to Lake automatically (so the build
description is recomputed when you switch configurations). For the CUDA regression suite, run:

```bash
TORCHLEAN_REQUIRE_CUDA=1 scripts/lake.sh -Kcuda=true exe nn_tests_suite
```

See the [CUDA guide]({{ '/cuda/' | relative_url }}) for deterministic reductions, parity checks,
sanitizers, and the remaining native-code trust boundary.

The SDK's CMake configuration supplies its compiler ABI flags and library dependencies. TorchLean
compiles C++ adapters, while the torch runtime supplies GPU kernels and their supported
architectures. The tensor operations use the LibTorch Stable ABI and can run with a later torch
runtime. Allocator statistics, memory-fraction controls and `emptyCache` require torch 2.13 to
2.15. You can also pass the SDK path to Lake:

```bash
scripts/lake.sh -Kcuda=true \
  -Klibtorch_home=/absolute/path/to/libtorch build
scripts/lake.sh -Kcuda=true \
  -Klibtorch_home=/absolute/path/to/libtorch test
```

Our maintained CUDA profile uses attention composed in Lean from matrix products, masking, softmax,
and a local VJP. LibTorch supplies the numerical primitives, while TorchLean's tape owns Q/K/V and
the saved probabilities. The full score matrices require quadratic memory in sequence length. The
[backend
chapter]({{ '/blueprint/Runtime___-Autograd___-and-Interop/Inside-The-Backend-Planner/' | relative_url }})
explains its per-operation selection and backward boundary.

## macOS

Install Apple's command-line developer tools and ensure Python 3 is available, then install Elan
and TorchLean:

```bash
xcode-select --install
curl https://elan.lean-lang.org/elan-init.sh -sSf | sh
git clone https://github.com/lean-dojo/TorchLean.git
cd TorchLean
scripts/lake.sh exe cache get
scripts/lake.sh build
scripts/lake.sh exe torchlean quickstart_mlp --device cpu --steps 10
```

The CPU path works on Intel and Apple silicon. Modern macOS has no NVIDIA CUDA execution path.
TorchLean has no Metal/MPS kernels, and `--device metal` fails as an unknown device instead of
quietly running the CPU implementation.

## Windows

### Recommended: WSL2

Ubuntu under WSL2 is the most reliable Windows setup. Open an administrator PowerShell prompt:

```powershell
wsl --install -d Ubuntu
```

After Windows restarts, open Ubuntu and follow the Linux instructions. For an NVIDIA GPU, follow
NVIDIA's [CUDA on WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html). Install the
Windows NVIDIA driver and the CUDA toolkit inside WSL, but don't install a second Linux display
driver inside WSL.

### Native Windows

Native Windows isn't a tested release path. Install Git, the Visual Studio C++ build tools, and the
Windows SDK, then install Elan from PowerShell:

```powershell
curl -O --location https://elan.lean-lang.org/elan-init.ps1
powershell -ExecutionPolicy Bypass -f elan-init.ps1
del elan-init.ps1
```

Installing these tools alone doesn't make the native runtime work on Windows, so use WSL2 for the
supported CPU and NVIDIA GPU paths.

## Use TorchLean From Another Lean Project

Add TorchLean to the downstream project's `lakefile.lean`:

```lean
require TorchLean from git "https://github.com/lean-dojo/TorchLean.git" @ "main"
```

For `#eval` and editor evaluation, add the CPU tensor library to your existing package declaration:

```lean
package MyProject where
  dynlibs := #[`@TorchLean/torchlean_tensor_cpu_shared]
```

Compiled executables link this library automatically, and the `dynlibs` entry also loads it in
Lean's interpreter.

Then update and build from the downstream project's root using its own Lake configuration:

```bash
lake update
lake exe cache get
lake build
```

A model file that uses the supervised trainer starts with:

```lean
import TorchLean.NN.Data.Dataset
import TorchLean.NN.Trainer.Constructor
import TorchLean.NN.Trainer.Train
open TorchLean
```

TorchLean includes [FloatLib](https://github.com/lean-dojo/FloatLib) for scalar arithmetic and
numerical proofs. For example:

```lean
import FloatLib
open FloatLib.Floats

abbrev Binary128 :=
  ExecFloat.Binary (exponentBits := 15) (fractionBits := 112)

def reading : Binary128 := 1.5
def scaled : Binary128 := reading * 2.25
```

Here the type chooses 113 bits of significand precision in a 128-bit IEEE layout, and literals are
rounded from exact rationals directly into the selected format. Wider software formats use the same
public arithmetic interface, but selecting one doesn't add arbitrary-precision hardware support to
CUDA or cuBLAS.

See the [floating-point guide]({{ '/blueprint/Floating-Point-and-Native-Boundaries/Floating-Point-Semantics/' | relative_url }})
for other formats, elementary functions, and numerical proofs.

For development against a neighboring checkout, use a path dependency:

```lean
require TorchLean from "../TorchLean"
```

## From A Model To A Kernel

When you change devices, the model API stays the same. Before each operation runs, the backend
planner checks its provider, device, and contract evidence against the selected profile.
Unavailable providers fail rather than changing the request. Add `--show-backend` to a model
command to see the selected implementations.

The default `libtorch_cuda` profile uses LibTorch numerical operations and TorchLean's tape.
Operations without retained CUDA parity checks record a LibTorch trust boundary, and the strict
`checked_cuda` profile rejects them. These profile names don't assert a Lean refinement proof for
LibTorch kernels. Reduction order remains part of each operation's numerical policy: an
implementation-defined CUDA reduction can't inherit a certificate for fixed-left accumulation.

Read [Inside the Backend
Planner]({{ '/blueprint/Runtime___-Autograd___-and-Interop/Inside-The-Backend-Planner/' | relative_url }})
for capsules, provider preference, VJP ownership, assurance policies, and backend reports. Read
[From a Tensor Operation to a GPU
Kernel]({{ '/blueprint/Floating-Point-and-Native-Boundaries/From-A-Tensor-Operation-To-A-GPU-Kernel/' | relative_url }})
for native CUDA dispatch, boundary checks, determinism, and operation coverage.

## Check An Installation

These commands cover the normal CPU installation:

```bash
scripts/lake.sh build
scripts/lake.sh lint
scripts/lake.sh exe nn_tests_suite
scripts/lake.sh exe torchlean --help
scripts/lake.sh exe verify --help
```

For CUDA, rebuild with `scripts/lake.sh -Kcuda=true build`, then run the suite with
`TORCHLEAN_REQUIRE_CUDA=1 scripts/lake.sh -Kcuda=true exe nn_tests_suite`. That environment
variable makes an unavailable CUDA runtime a failure instead of allowing CUDA tests to skip.

For a complete account of Lean axioms, executable checkers, CUDA and FFI code, external artifact
producers, and floating-point assumptions, read
[`docs/TRUST_BOUNDARIES.md`](https://github.com/lean-dojo/TorchLean/blob/main/docs/TRUST_BOUNDARIES.md).

With a working installation, [Getting Started]({{ '/start/' | relative_url }}) trains a small model
and runs interval bound propagation.

## References

- [Elan: Lean toolchain manager](https://github.com/leanprover/elan).
- [Lean reference: validating
  proofs](https://lean-lang.org/doc/reference/latest/ValidatingProofs/).
- [NVIDIA CUDA Installation Guide for
  Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/).
- [NVIDIA CUDA on WSL User Guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).
- [Installing LibTorch](https://docs.pytorch.org/cppdocs/installing.html).
- George C. Necula, ["Proof-Carrying Code"](https://doi.org/10.1145/263699.263712), POPL 1997.
  Kernel capsules are contract and provenance records, not proof-carrying binaries, and none of the
  maintained capsules carries a Lean refinement theorem for its implementation.
