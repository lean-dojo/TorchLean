---
title: CUDA
layout: default
---

# CUDA

LibTorch evaluates supported Float32 tensor operations on an NVIDIA GPU. TorchLean owns the
differentiation tape, chooses the local backward operations, and manages their saved tensors.
Model shapes and graph construction use the same Lean API as CPU execution.

## Build and Run

Select a CUDA-enabled LibTorch SDK and build:

```bash
export TORCHLEAN_LIBTORCH_HOME=/opt/libtorch
scripts/lake.sh -R -K cuda=true build
```

Run a small CUDA example:

```bash
scripts/lake.sh -R -K cuda=true exe torchlean mlp --device cuda --steps 100
```

Run the maintained numerical and native-boundary suite:

```bash
scripts/lake.sh -R -K cuda=true test
```

Run the CUDA sanitizer suite when changing the native adapter:

```bash
scripts/checks/cuda_sanitize_tests.sh --all-tools
```

## What CUDA Covers

The adapter calls ATen for elementwise arithmetic, reductions, matrix products, convolutions,
pooling, shape operations, attention, and FFTs. LibTorch dispatches those calls to implementations
available for the device and request. Model examples select this path with `--device cuda`.
These operations compute values and local gradients without recording a LibTorch autograd graph.

## Determinism

Float32 addition is not associative, so a reduction's evaluation order can affect its result.
Request strict deterministic algorithms through the typed LibTorch controls:

```lean
Runtime.Autograd.Cuda.LibTorch.setDeterministic true
```

`setDeterministic : Bool → IO Unit` checks the native setting and disables convolution
benchmarking. A subsequent operation fails if the SDK cannot meet the request.
Use this IO setter from application code with `import NN.Runtime`.

The same policy can be selected at process startup:

```bash
TORCHLEAN_CUDA_DETERMINISTIC_REDUCTIONS=1 \
  scripts/lake.sh -R -K cuda=true exe torchlean mlp --device cuda
```

Deterministic execution does not establish bitwise agreement with a particular FloatLib reference
or across different SDKs and devices. The
[GPU chapter]({{ '/blueprint/Floating-Point-and-Native-Boundaries/From-A-Tensor-Operation-To-A-GPU-Kernel/' | relative_url }})
explains the numerical contracts and what the CUDA tests establish. For provider selection, VJP ownership, and
assurance policies, read
[Inside the Backend Planner]({{ '/blueprint/Runtime___-Autograd___-and-Interop/Inside-The-Backend-Planner/' | relative_url }}).
