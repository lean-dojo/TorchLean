---
title: CUDA
layout: default
---

# CUDA

CUDA runs supported Float32 operations on an NVIDIA GPU without changing the model's tensor types
or graph. The linked guide chapters give the full provider and trust account; the material below
collects the commands used to build and test the native path.

## Build and Run

Build with CUDA support:

```bash
lake -R -K cuda=true build
```

Run a small CUDA example:

```bash
lake -R -K cuda=true exe torchlean mlp --device cuda --steps 100
```

Run a broader CUDA regression pass:

```bash
scripts/checks/example_regression.sh --cuda
```

Run the CUDA sanitizer suite when changing native kernels:

```bash
scripts/checks/cuda_sanitize_tests.sh --all-tools
```

## Compilation Target

`-K cuda=true` says to compile the native kernels; it does not say which GPU to compile them
*for*. That second choice belongs to `nvcc`, which without instruction applies a built-in default
that changes with the toolkit version — CUDA 13.0, for instance, emits `sm_75` machine code plus
`compute_75` PTX. Such a binary runs at full speed on that architecture, reaches a newer GPU only
through forward PTX just-in-time compilation, and does not load on an older one at all. Name the
target when the deployment GPU is known:

```bash
lake -R -K cuda=true -K cuda_arch=sm_86 build     # an A10G or an RTX A4500
lake -R -K cuda=true -K cuda_arch=native build    # whatever GPU this machine has
```

A value starting with `-` is passed to `nvcc` verbatim, which is how one binary is compiled for
several architectures at once:

```bash
lake -R -K cuda=true \
  -K cuda_arch="-gencode arch=compute_75,code=sm_75 \
                -gencode arch=compute_90,code=[sm_90,compute_90]" build
```

`TORCHLEAN_CUDA_ARCH` carries the same value when the Lake option is absent, so a container image
or CI job can select a target without rewriting its build command. Both spellings participate in
Lake's build trace: changing the target recompiles the kernels. `nvcc`'s own `NVCC_APPEND_FLAGS`
does not — Lake cannot see it, judges the existing objects current, and links kernels compiled for
the previous target, which is visible only as unexplained throughput. Prefer the option, and delete
`.lake/build/torchlean_*.o` if a target was ever set that way.

`scripts/checks/cuda_arch_target.sh` asserts this behavior, and needs neither a toolkit nor a GPU.

## What CUDA Covers

The CUDA path is used for supported Float32 tensor operations: elementwise arithmetic, reductions,
matmul/cuBLAS paths, convolution/pooling kernels, shape/view operations, attention kernels, FFT/FNO
support where enabled, and model examples that choose `--device cuda`.

## Determinism

Some CUDA reduction and backward paths use floating-point accumulation. Because Float32 addition is
not associative, atomic accumulation can be schedule-dependent. TorchLean also provides an opt-in
deterministic reductions mode for the covered reduction, gather/scatter, and pooling-backward paths:

```lean
let _ := Runtime.Autograd.Cuda.Buffer.setDeterministicReductionsChecked true
```

or:

```bash
TORCHLEAN_CUDA_DETERMINISTIC_REDUCTIONS=1 lake -R -K cuda=true exe torchlean mlp --device cuda
```

The setting covers the reduction, gather/scatter, and pooling-backward paths named in the
[GPU chapter]({{ '/blueprint/Floating-Point-and-Native-Boundaries/From-A-Tensor-Operation-To-A-GPU-Kernel/' | relative_url }}),
which also explains what the CUDA tests establish. For provider selection, VJP ownership, and
assurance policies, read
[Inside the Backend Planner]({{ '/blueprint/Runtime___-Autograd___-and-Interop/Inside-The-Backend-Planner/' | relative_url }}).
