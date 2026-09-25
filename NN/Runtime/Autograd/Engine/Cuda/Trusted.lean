/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

Trusted boundary for CUDA FFI.

Why this file exists:
- TorchLean’s repo policy forbids axioms in general library code.
- The CUDA runtime types are produced by external C/C++ code, so we need a small trusted bridge
  to make them usable in compiled Lean code.

Everything in this module should be treated as part of the "FFI trust base".
-/

module

/-!
# Trusted CUDA Runtime Boundary

This module contains the opaque CUDA buffer type used by the native runtime. Buffers are created by
explicit FFI allocation/copy functions. The nonemptiness witness below is only what Lean needs to
declare extern functions returning `Buffer`; it is not a default CUDA allocation and should not be
used as one.

The rest of this docstring is the map from native translation units to the Lean modules that call
them. DocGen documents Lean modules, not C or CUDA files, so the map lives here, on the module that
is the trust boundary, rather than in a separate source browser.

## Trust boundary

The CUDA backend crosses a C++/LibTorch FFI boundary. Lean does not prove the compiled native
implementation correct. The trusted pieces include LibTorch and ATen, their CUDA libraries,
compiler and runtime, GPU hardware, and Lean's external-object ABI and finalizers.

TorchLean retains its own tape and selected local VJPs. ATen computes tensor values and local
backward operations with autograd recording disabled. Proof-facing kernel specifications,
float32 agreement hypotheses, and graph semantics remain Lean definitions. Runtime regression
and numerical parity tests provide evidence for a particular build and set of inputs.

## Native source groups

- `csrc/libtorch/torchlean_libtorch.h`
  Shared boxed-buffer ABI, size checks, device guards, and the no-autograd call boundary.
  Lean modules: `Cuda.Trusted`, `Cuda.Buffer`, and `Cuda.LibTorch`.

- `csrc/libtorch/runtime.cpp`
  ATen storage ownership, allocation and copies, checkpoint bytes, seeded random values,
  runtime configuration, and allocation telemetry. LibTorch owns the CUDA allocator; logical
  TorchLean payload counters are separate from its allocated and reserved byte counters.

- `csrc/libtorch/elementwise.cpp`
  Pointwise operations, selected local gradients, reductions, losses, and optimizer arithmetic.

- `csrc/libtorch/kernels.cpp`
  Shape operations, indexing, normalization, batched matrix products, FFT, spectral convolution,
  and selective scan through ATen operations. Lean modules: `Cuda.Kernels` and `Cuda.Ops`.

- `csrc/libtorch/conv_pool.cpp`
  Convolution, transpose convolution, pooling, their local backward operations, and the
  output-length checks shared with `Spec`.

- `csrc/libtorch/attention.cpp`
  Attention forward dispatch and matched backward calls using saved forward state.
  TorchLean's tape owns the output buffer that retains this state.

- `csrc/libtorch/blas.cpp`
  The separate binary64 `FloatArray` matrix-multiplication interface, implemented with ATen.
  Lean module: `Cuda.DGemm`. Eager CUDA tape buffers remain binary32.

- `csrc/libtorch/unavailable.c`
  The same symbols for builds without LibTorch. The runtime status is `.notLinked`, IO calls
  return an error, and pure buffer operations abort with a message to rebuild with
  `-K cuda=true`. User CUDA sessions are rejected before any of them runs.

The GPU random stream is evaluated with ATen integer operations and checked against the seeded
contract.

-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace Cuda

/--
Opaque handle to a contiguous float32 CUDA buffer, implemented in `csrc/libtorch/runtime.cpp`.
Builds without `-K cuda=true` cannot create one.
-/
opaque BufferImpl : NonemptyType.{0}

/--
Runtime representation used for native CUDA buffer handles.

The `NonemptyType` wrapper is Lean's standard representation for external resources: it gives
extern declarations a nonempty result type while preserving reference-counting information in
compiled code. The underlying value is still created only by the native buffer constructors.
-/
def Buffer : Type := BufferImpl.val

instance : Nonempty Buffer := BufferImpl.property

end Cuda
end Autograd
end Runtime
