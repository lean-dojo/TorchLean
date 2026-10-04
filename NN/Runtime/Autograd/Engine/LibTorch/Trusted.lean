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
  Lean modules: `LibTorch.Trusted`, `LibTorch.Buffer`, and `LibTorch`.

- `csrc/libtorch/torchlean.cpp`
  Shared implementation of storage ownership, allocation, transfers, seeded random values,
  runtime controls, tensor operations, and explicit backward calls through ATen.
  It implements elementwise operations, reductions, indexing, matrix multiplication,
  normalization, Fourier transforms, scans, and convolution/pooling.
  Lean composes attention in `LibTorch.Ops.Attention` and retains Q/K/V and probability buffers
  on its tape; the native buffer has no attention context.
  LibTorch owns the CUDA allocator; TorchLean's payload counters track logical ownership.
  The `LibTorch.DGemm` interface accepts binary64 host arrays; eager CUDA buffers remain binary32.

- `csrc/libtorch/operations.h`
  Common operation list used to generate LibTorch exports and unavailable-build signatures.

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
namespace LibTorch

/--
Opaque handle to a contiguous float32 buffer on the bridge's device (a CUDA device, or the host),
implemented in `csrc/libtorch/torchlean.cpp`. Builds without `-K cuda=true` cannot create one.
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

end LibTorch
end Autograd
end Runtime
