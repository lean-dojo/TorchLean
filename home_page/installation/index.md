---
title: Installation
layout: default
redirect_from:
  - /backends/
---

# Installation

If you want to try TorchLean on a laptop, start with the CPU build. It does not require PyTorch,
CUDA, or a GPU. The repository pins its Lean version in `lean-toolchain`, so Elan will select the
right compiler for you.

## A Five-Minute CPU Install

First install [Elan](https://github.com/leanprover/elan), the Lean toolchain manager. On Linux or
macOS:

```bash
curl https://elan.lean-lang.org/elan-init.sh -sSf | sh
```

Open a new terminal so that `elan`, `lean`, and `lake` are on your `PATH`. Then clone and build
TorchLean:

```bash
git clone https://github.com/lean-dojo/TorchLean.git
cd TorchLean
lake exe cache get
lake build
```

The cache command downloads compatible prebuilt Lean dependencies when they are available. It is
safe to omit; `lake build` will compile anything that is missing.

Run a small model to check the executable path:

```bash
lake exe torchlean quickstart_mlp --device cpu --steps 10
```

If those commands succeed, TorchLean is installed. You can inspect the available examples and
verification commands with:

```bash
lake exe torchlean --help
lake exe verify --help
```

That CPU build is the common starting point on every platform. From there, TorchLean can build
its native CUDA runtime or link an external provider without changing the Lean model being run.
The table below separates paths that work today from targets that are represented in the backend
architecture but still need platform-specific runtime work.

| Platform | CPU | NVIDIA GPU | LibTorch provider | Current status |
| --- | --- | --- | --- | --- |
| Linux | &#10003; | &#10003; Native CUDA | SDPA forward with TorchLean backward | Supported |
| macOS, Intel or Apple silicon | &#10003; | Not applicable | Not yet | CPU supported; Metal is planned |
| Windows with WSL2 | &#10003; Linux path | &#10003; CUDA on WSL2 | Linux path | Recommended Windows setup |
| Native Windows | Bring-up target | Not validated | Not wired | Backend target exists; native toolchain work remains |

Here, "LibTorch provider" means the current scaled-dot-product-attention bridge, not a requirement
for ordinary TorchLean models and not a claim that every operation is delegated to PyTorch. The
CPU, native CUDA, and LibTorch sections below give the corresponding build commands.

## Linux

### CPU

You need Git, `curl`, and a C/C++ compiler. On Ubuntu or Debian:

```bash
sudo apt update
sudo apt install -y git curl build-essential
```

Then follow the five-minute install above. The default build uses the portable CPU runtime. It also
builds harmless CUDA stub archives so that CPU-only machines can compile the complete Lean project;
the stubs do not pretend that a GPU is present.

### NVIDIA CUDA

Install a supported NVIDIA driver and CUDA toolkit using NVIDIA's
[CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/).
TorchLean needs `nvcc`, cuBLAS, and cuFFT. Check the machine before rebuilding:

```bash
nvidia-smi
nvcc --version
```

Build and run the CUDA configuration:

```bash
lake -R -K cuda=true build
lake -R -K cuda=true exe torchlean quickstart_mlp \
  --device cuda --steps 10 --show-backend
```

The two CUDA choices happen at different times. `-K cuda=true` tells Lake to compile and link the
native CUDA implementation. `--device cuda` asks the executable to use it. A CPU-linked executable
rejects `--device cuda` instead of silently moving the run back to the CPU.

`nvcc` compiles the kernels for its own default architecture unless told otherwise. Add
`-K cuda_arch=sm_86` (or `native`, or a verbatim `-gencode` list) to compile for the GPU you deploy
on; the [CUDA guide]({{ '/cuda/' | relative_url }}) explains what the default costs you.

Use `-R` whenever you switch between CPU and CUDA configurations; it forces Lake to recompute the
build description. The CUDA regression suite is:

```bash
lake -R -K cuda=true exe nn_tests_suite
```

The [CUDA guide]({{ '/cuda/' | relative_url }}) covers deterministic reductions, parity checks,
sanitizers, and the remaining native-code trust boundary.

### Optional LibTorch Attention

The normal CPU and CUDA builds do not need LibTorch. TorchLean currently uses LibTorch only through
an optional scaled-dot-product-attention bridge. Download a matching GPU-enabled distribution from
the [official LibTorch installation page](https://docs.pytorch.org/cppdocs/installing.html) and
extract it somewhere outside the repository.

The extracted directory must contain `include/` and `lib/`. Pass its absolute path to Lake:

```bash
lake -R -K cuda=true -K libtorch=true \
  -K libtorch_home=/absolute/path/to/libtorch build
lake -R -K cuda=true -K libtorch=true \
  -K libtorch_home=/absolute/path/to/libtorch exe libtorch_sdpa_test
```

This enables the `libtorch_forward_cuda` profile for scaled-dot-product attention. The
[backend chapter]({{ '/blueprint/Runtime___-Autograd___-and-Interop/Inside-The-Backend-Planner/' | relative_url }})
explains its per-operation selection and backward boundary.

## macOS

Install Apple's command-line developer tools, then Elan and TorchLean:

```bash
xcode-select --install
curl https://elan.lean-lang.org/elan-init.sh -sSf | sh
git clone https://github.com/lean-dojo/TorchLean.git
cd TorchLean
lake exe cache get
lake build
lake exe torchlean quickstart_mlp --device cpu --steps 10
```

The CPU path works on Intel and Apple silicon. Modern macOS has no NVIDIA CUDA execution path.
TorchLean already reserves `--device metal` in its device vocabulary, but Metal/MPS kernels are not
implemented yet. Selecting Metal therefore returns an unsupported-device error; it does not quietly
run the CPU implementation.

## Windows

### Recommended: WSL2

The most reliable Windows setup is Ubuntu under WSL2. Open an administrator PowerShell prompt:

```powershell
wsl --install -d Ubuntu
```

After Windows restarts, open Ubuntu and follow the Linux instructions. For an NVIDIA GPU, follow
NVIDIA's [CUDA on WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html). Install the
Windows NVIDIA driver and the CUDA toolkit inside WSL; do not install a second Linux display driver
inside WSL.

### Native Windows

Native Windows is represented in the backend target vocabulary, but it is still a bring-up target
rather than a regularly tested release path. Install Git, the Visual Studio C++ build tools, and the
Windows SDK. Then install Elan from PowerShell:

```powershell
curl -O --location https://elan.lean-lang.org/elan-init.ps1
powershell -ExecutionPolicy Bypass -f elan-init.ps1
del elan-init.ps1
```

The intended native CPU commands are:

```powershell
lake exe cache get
lake build
```

The remaining work is platform engineering: the native libraries must be compiled with a compatible
Windows C/C++ toolchain; CUDA and LibTorch must be discovered as `.lib` and DLL artifacts; Linux
linker options such as `-Wl,-rpath` must be replaced; and the CPU stubs and GPU runtime must be tested
under the Windows loader and ABI. Once those pieces are wired, the existing device, provider, and
capsule abstractions do not need to be redesigned. Until then, WSL2 is the supported route for both
CPU and NVIDIA GPU execution on Windows.

## Use TorchLean From Another Lean Project

Add TorchLean to the downstream project's `lakefile.lean`:

```lean
require TorchLean from git "https://github.com/lean-dojo/TorchLean.git" @ "main"
```

Then update and build:

```bash
lake update
lake exe cache get
lake build
```

Most model files need only:

```lean
import NN.API
open TorchLean
```

The numerical library has a smaller independent import:

```lean
import NN.Floats
open TorchLean.Floats
```

It includes formats, rounding, finite binary32 semantics, executable IEEE binary32 operations,
interval rounders, and scalar quantization. It does not load the tensor, model, autograd, CUDA,
certificate, or external-tool layers. Use `NN.Spec.Quantization` when tensor quantization is needed,
and `NN.Proofs.RuntimeApprox.FP32` when connecting binary32 arithmetic to runtime-approximation
proofs.

For development against a neighboring checkout, use a path dependency:

```lean
require TorchLean from "../TorchLean"
```

## From A Model To A Kernel

Installation chooses a device or a complete backend profile; the model API stays the same. The
planner then selects an implementation per operation and rejects unavailable providers instead of
quietly changing the request.

Read [Inside the Backend Planner]({{ '/blueprint/Runtime___-Autograd___-and-Interop/Inside-The-Backend-Planner/' | relative_url }})
for capsules, provider preference, VJP ownership, assurance policies, and backend reports. Read
[From a Tensor Operation to a GPU Kernel]({{ '/blueprint/Floating-Point-and-Native-Boundaries/From-A-Tensor-Operation-To-A-GPU-Kernel/' | relative_url }})
for native CUDA dispatch, boundary checks, determinism, and the current operation coverage.

## Check An Installation

These commands cover the normal CPU installation:

```bash
lake build
lake lint
lake exe nn_tests_suite
lake exe torchlean --help
lake exe verify --help
```

For CUDA, rebuild and run the suite with `-R -K cuda=true`.

For a complete account of Lean axioms, executable checkers, CUDA and FFI code, external artifact
producers, and floating-point assumptions, read
[`TRUST_BOUNDARIES.md`](https://github.com/lean-dojo/TorchLean/blob/main/TRUST_BOUNDARIES.md).

## References

- [Elan: Lean toolchain manager](https://github.com/leanprover/elan).
- [Lean reference: validating proofs](https://lean-lang.org/doc/reference/latest/ValidatingProofs/).
- [NVIDIA CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/).
- [NVIDIA CUDA on WSL User Guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).
- [Installing LibTorch](https://docs.pytorch.org/cppdocs/installing.html).
- George C. Necula, ["Proof-Carrying Code"](https://doi.org/10.1145/263699.263712), POPL 1997.
  Ordinary kernel capsules are contract and provenance records, not proof-carrying binaries.
  TorchLean's separate typed `ProofCarryingKernel` interface retains a Lean refinement theorem with
  an implementation when such a proof is available.
