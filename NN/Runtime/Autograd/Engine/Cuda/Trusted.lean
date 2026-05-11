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

This module contains the opaque CUDA buffer type and the single inhabitance axiom needed to make the
external FFI type usable from Lean. It is intentionally compact: all declarations here are part of the
TorchLean trusted computing base for CUDA execution.
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
opaque Buffer : Type

-- External code constructs `Buffer` values, so we assume the type is inhabited.
axiom instNonemptyBuffer : Nonempty Buffer
attribute [instance] instNonemptyBuffer

end Cuda
end Autograd
end Runtime
