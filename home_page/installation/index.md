---
title: Installation
layout: default
redirect_from:
  - /backends/
---

# Installation

If you want to try TorchLean on a laptop, start with the CPU build. It does not require PyTorch,
CUDA, or a GPU. The repository pins its Lean version in `lean-toolchain`, so Elan will select the
right compiler for you: Lean 4.34.0. Mathlib uses the matching release. FloatLib tracks `main` in
`lakefile.lean`, with the exact revision recorded in `lake-manifest.json`. Run
`scripts/lake.sh update floatlib` to adopt newer FloatLib changes, then rebuild and test TorchLean.

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
The table below separates paths that work today from platforms that still need platform-specific
runtime work.

| Platform | CPU | NVIDIA GPU | LibTorch provider | Current status |
| --- | --- | --- | --- | --- |
| Linux | &#10003; | &#10003; Native CUDA | SDPA forward with TorchLean backward | Supported |
| macOS, Intel or Apple silicon | &#10003; | Not applicable | Not yet | CPU supported; Metal is planned |
| Windows with WSL2 | &#10003; Linux path | &#10003; CUDA on WSL2 | Linux path | Recommended Windows setup |
| Native Windows | Not yet | Not validated | Not wired | Native toolchain work remains; use WSL2 |

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

Linux native targets also build a private mimalloc 3.4.4 object from checksum-pinned source.
Position-independent code and initial-exec thread-local storage let it link into executables and
shared libraries. It includes a narrow arena-purge wakeup repair; the installed Lean compiler and
`#eval` keep their existing allocator. The
[native allocation boundary](https://github.com/lean-dojo/TorchLean/blob/main/docs/TRUST_BOUNDARIES.md#native-host-allocation)
describes the repair and its remaining assumptions.

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

Native Windows is not a tested release path yet. Install Git, the Visual Studio C++ build tools,
and the Windows SDK. Then install Elan from PowerShell:

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

Scalar arithmetic and numerical proofs come from the separate
[FloatLib package](https://github.com/lean-dojo/FloatLib), included as a pinned dependency:

```lean
import FloatLib
open FloatLib.Floats

abbrev Binary128 :=
  ExecFloat.Binary (exponentBits := 15) (fractionBits := 112)

def reading : Binary128 := 1.5
def scaled : Binary128 := reading * 2.25
```

Here the type chooses 113 bits of significand precision in a 128-bit IEEE layout. Literals are
rounded from exact rationals directly into the selected format. Wider software formats use the
same public arithmetic interface; selecting one does not add arbitrary-precision hardware support
to CUDA or cuBLAS.

`import FloatLib` supplies configured numerical types, reference semantics, refinement theorems,
and intervals without importing TorchLean's model or runtime layers. Binary elementary functions
have a separate import,
`FloatLib.Floats.Formats.BinaryInterchange.Configured.Transcendentals`; they are deterministic
approximations and need their own accuracy claims. Tensor quantization and graph-level numerical
proofs remain TorchLean integrations.

For development against a neighboring checkout, use a path dependency:

```lean
require TorchLean from "../TorchLean"
```

## From A Model To A Kernel

Installation chooses a device or a complete backend profile; the model API stays the same. Each
operation an eager session executes is first matched to a `KernelCapsule`: a record naming the
operation, provider, device, trust level (`checked` or `trustedExternal`), VJP mode, and four
contract descriptors (shape, layout, value, VJP) with their evidence. Its `numericalPolicy` has one
field, `reduction`, which says whether the kernel accumulates in the fixed left-fold order of the
tensor semantics or in an implementation-defined order. The numerical certificate registry reads
that field, so a CUDA or LibTorch capsule cannot inherit a fixed-left reduction certificate.

Selection has three steps. The planner picks the first capsule whose device, provider preference,
VJP mode, and trust level fit the profile. `checkContracts` then produces a `ContractCheck` that
rejects any selected descriptor whose evidence the profile's assurance policy does not admit. The
session finally calls `KernelCapsule.bind` to pair the capsule with a handler for the same
operation, provider, and device before running it. Unavailable providers fail at that point instead
of quietly changing the request. `--show-backend` prints each selected capsule the first time a
session uses it.

The attention provider is the one place where the profile changes which implementation runs.
`checked_cuda` prefers TorchLean's composed attention (CUDA batched matrix multiplication with
TorchLean's hard-masked softmax). `libtorch_forward_cuda` prefers the LibTorch SDPA forward
capsule. Both keep `vjpMode := .torchLeanTape`, so TorchLean records the tape node and owns the
backward pass in either case.

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
[`docs/TRUST_BOUNDARIES.md`](https://github.com/lean-dojo/TorchLean/blob/main/docs/TRUST_BOUNDARIES.md).

## References

- [Elan: Lean toolchain manager](https://github.com/leanprover/elan).
- [Lean reference: validating proofs](https://lean-lang.org/doc/reference/latest/ValidatingProofs/).
- [NVIDIA CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/).
- [NVIDIA CUDA on WSL User Guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).
- [Installing LibTorch](https://docs.pytorch.org/cppdocs/installing.html).
- George C. Necula, ["Proof-Carrying Code"](https://doi.org/10.1145/263699.263712), POPL 1997.
  Kernel capsules are contract and provenance records, not proof-carrying binaries; none of the
  maintained capsules carries a Lean refinement theorem for its implementation.
