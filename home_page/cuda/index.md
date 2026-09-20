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

Run the maintained numerical and native-boundary suite:

```bash
scripts/lake.sh -R -K cuda=true test
```

Run the CUDA sanitizer suite when changing native kernels:

```bash
scripts/checks/cuda_sanitize_tests.sh --all-tools
```

## What CUDA Covers

The CUDA path is used for supported Float32 tensor operations: elementwise arithmetic, reductions,
matmul/cuBLAS paths, convolution/pooling kernels, shape/view operations, attention kernels, FFT/FNO
support where enabled, and model examples that choose `--device cuda`.

## Determinism

Some CUDA reduction and backward paths use floating-point accumulation. Because Float32 addition is
not associative, atomic accumulation can be schedule-dependent. TorchLean also provides an opt-in
deterministic reductions mode for the covered reduction, gather/scatter, and pooling-backward paths:

```lean
Runtime.Autograd.Cuda.Buffer.setDeterministicReductions true
```

`setDeterministicReductions : Bool → IO Unit` fails when the native runtime did not accept the
setting. Use this IO setter from application code with `import NN.Runtime`.

or:

```bash
TORCHLEAN_CUDA_DETERMINISTIC_REDUCTIONS=1 lake -R -K cuda=true exe torchlean mlp --device cuda
```

The setting covers the reduction, gather/scatter, and pooling-backward paths named in the
[GPU chapter]({{ '/blueprint/Floating-Point-and-Native-Boundaries/From-A-Tensor-Operation-To-A-GPU-Kernel/' | relative_url }}),
which also explains what the CUDA tests establish. For provider selection, VJP ownership, and
assurance policies, read
[Inside the Backend Planner]({{ '/blueprint/Runtime___-Autograd___-and-Interop/Inside-The-Backend-Planner/' | relative_url }}).
