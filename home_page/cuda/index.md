---
title: CUDA
layout: default
---

# CUDA

LibTorch evaluates supported `Float32` and `Float` tensor operations on an NVIDIA GPU. TorchLean owns the
differentiation tape, chooses the local backward operations, and manages their saved tensors.
Model shapes and graph construction use the same Lean API as CPU execution.

## Build and Run

Select a CUDA-enabled LibTorch SDK and build:

```bash
export TORCHLEAN_LIBTORCH_HOME=/opt/libtorch
scripts/lake.sh -Kcuda=true build
```

Run a small CUDA example with a built-in dataset:

```bash
scripts/lake.sh -Kcuda=true exe torchlean quickstart_mlp --device cuda --execution eager --steps 100
```

Run the maintained numerical and native-boundary suite:

```bash
TORCHLEAN_REQUIRE_CUDA=1 scripts/lake.sh -Kcuda=true test
```

Run the CUDA sanitizer suite when changing the native adapter:

```bash
scripts/checks/cuda.sh sanitize --all-tools
```

## What CUDA Covers

The adapter calls ATen for elementwise arithmetic, reductions, matrix products, convolutions,
pooling, shape operations, and FFTs. Lean composes attention from matrix products and softmax,
including its explicit local VJP, and spectral layers from FFT, frequency mixing, and inverse FFT.
The tape owns their saved buffers. LibTorch supplies the numerical primitives without recording
an autograd graph. Model examples select this path with `--device cuda`.

Attention materializes full score and probability matrices, so its memory use is quadratic in
sequence length. It does not select a fused attention implementation.

We can also run our own Lean calculations on GPU, including arithmetic in configured binary
formats such as binary128. Supported calculations record values and gradients on the same tape
without narrowing them. The [custom computations example]({{ '/examples/custom-computations/' | relative_url }})
shows how to try this and which operations we can record. These formats do not extend the dtype
support of LibTorch's model operators.

## Determinism

Floating-point addition is not associative, so a reduction's evaluation order can affect its result.
Request strict deterministic algorithms through the typed LibTorch controls:

```lean
import NN.Runtime

def enableDeterminism : IO Unit :=
  Runtime.Autograd.LibTorch.setDeterministic true
```

`setDeterministic : Bool → IO Unit` checks the native setting and disables convolution
benchmarking. A subsequent operation fails if the SDK cannot meet the request.
Use this IO setter from application code with `import NN.Runtime`.

The same policy can be selected at process startup:

```bash
TORCHLEAN_CUDA_DETERMINISTIC_REDUCTIONS=1 \
  scripts/lake.sh -Kcuda=true exe torchlean quickstart_mlp --device cuda --steps 100
```

Deterministic execution does not establish bitwise agreement with a particular FloatLib reference
or across different SDKs and devices. The
[GPU chapter]({{ '/blueprint/Floating-Point-and-Native-Boundaries/From-A-Tensor-Operation-To-A-GPU-Kernel/' | relative_url }})
explains the numerical contracts and what the CUDA tests establish. For provider selection, VJP ownership, and
assurance policies, read
[Inside the Backend Planner]({{ '/blueprint/Runtime___-Autograd___-and-Interop/Inside-The-Backend-Planner/' | relative_url }}).
