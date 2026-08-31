/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

Trusted boundary for CUDA FFI.

Why this file exists:
- TorchLean’s repo policy forbids axioms in general library code.
- The CUDA runtime types are produced by external C/CUDA code, so we need a small trusted bridge
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
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace Cuda

/--
Opaque handle to a contiguous float32 buffer (CUDA device memory when built with `-K cuda=true`,
otherwise a CPU stub buffer).

Implementation:
- CUDA: `csrc/cuda/tensor/torchlean_cuda_tensor.cu`
- CPU stub (default `lake build`): `csrc/cuda/tensor/torchlean_cuda_tensor_stub.c`
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

/--
Opaque handle to an immutable, layered 1-D lookup table of float32 samples (a CUDA texture object
over a layered `cudaArray` when built with `-K cuda=true`, otherwise a CPU stub table).

Implementation:
- CUDA: `csrc/cuda/textures/torchlean_cuda_textable.cu`
- CPU stub (default `lake build`): `csrc/cuda/textures/torchlean_cuda_textable_stub.c`
-/
opaque TexTableImpl : NonemptyType.{0}

/--
Runtime representation used for native lookup-table texture handles.

As with `Buffer`, the `NonemptyType` wrapper only gives extern declarations a nonempty result
type; values are created only by the native table constructor.
-/
def TexTable : Type := TexTableImpl.val

instance : Nonempty TexTable := TexTableImpl.property

end Cuda
end Autograd
end Runtime
