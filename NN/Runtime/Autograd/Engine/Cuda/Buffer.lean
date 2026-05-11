/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

CUDA buffer primitives (float32).

Implementation:
- CUDA: `csrc/cuda/tensor/torchlean_cuda_tensor.cu`
- CPU stub (default `lake build`): `csrc/cuda/tensor/torchlean_cuda_tensor_stub.c`

These are low-level, runtime-only kernels for the native GPU tape/buffer path.
-/

module

public import NN.Runtime.Autograd.Engine.Cuda.Trusted

@[expose] public section

namespace Runtime
namespace Autograd
namespace Cuda

namespace Buffer

/-!
### Deterministic Reductions Mode

TorchLean's CUDA runtime uses `atomicAdd` in a few kernels to accumulate float32 results. This is
fast, but floating-point addition is non-associative, and CUDA does not fix a global order for the
interleaving of atomic updates. As a result, some kernels can be bit-nondeterministic across runs.

TorchLean therefore exposes an opt-in deterministic mode that replaces those atomic accumulation
paths with fixed-order reductions. This trades performance for reproducibility.

This flag is a *runtime* setting affecting only the CUDA/stub backends; it has no effect on the
pure Lean Spec.
-/

@[extern "torchlean_cuda_set_deterministic_reductions"]
opaque setDeterministicReductionsRaw (on : UInt32) : Unit

@[extern "torchlean_cuda_get_deterministic_reductions_u"]
opaque getDeterministicReductionsRaw (u : UInt32) : UInt32

@[extern "torchlean_cuda_set_deterministic_reductions_checked"]
opaque setDeterministicReductionsCheckedRaw (on : UInt32) : UInt32

/--
Enable/disable deterministic reductions mode and return the observed flag value.

Why this helper exists: the raw setter returns `Unit`, so if you write `let _ := set...` in
Lean, the compiler is free (under pure semantics) to reorder or eliminate that call. The runtime
therefore provides a `*_checked` wrapper that both sets the flag and returns the observed value,
giving us a single call with an explicit return value dependency.
-/
def setDeterministicReductionsChecked (on : Bool) : Bool :=
  setDeterministicReductionsCheckedRaw (if on then 1 else 0) != 0

/-- Enable/disable deterministic reductions mode (see module docstring). -/
def setDeterministicReductions (on : Bool) : Unit :=
  let _ := setDeterministicReductionsChecked on
  ()

/-- Query whether deterministic reductions mode is enabled. -/
def getDeterministicReductions : Bool :=
  getDeterministicReductionsRaw 0 != 0

/-- Create a device buffer by copying from a host `FloatArray` (casts each element to float32). -/
@[extern "torchlean_cuda_buffer_of_float_array"]
opaque ofFloatArray (a : FloatArray) : Buffer

/-- Copy a buffer back to a host `FloatArray` (casts float32 elements to `Float`). -/
@[extern "torchlean_cuda_buffer_to_float_array"]
opaque toFloatArray (b : Buffer) : FloatArray

/-- Number of float32 elements in the buffer. -/
@[extern "torchlean_cuda_buffer_size"]
opaque size (b : Buffer) : UInt32

/--
Release the device allocation held by a buffer, returning `1` when a live allocation was released.

This is a runtime pressure valve for eager training loops that create many short-lived CUDA buffers.
The C finalizer is still safe after an explicit release because the pointer is nulled out.
-/
@[extern "torchlean_cuda_buffer_release"]
opaque release (b : Buffer) : UInt32

/--
Release `scratch` and return `keep`.

This exists for pure CUDA tape code: because the returned buffer is used downstream, Lean cannot
erase the native release call as dead code.
-/
@[extern "torchlean_cuda_buffer_release_then"]
opaque releaseThen (scratch keep : Buffer) : Buffer

/--
Ask the Lean runtime allocator (mimalloc) to collect abandoned/free pages.

This does not change any TorchLean value. It is a pressure valve for long native eager loops where
many short-lived tape closures and external-buffer wrappers are created every step.
-/
@[extern "torchlean_runtime_collect_allocator"]
opaque collectAllocatorRaw (force : UInt32) : UInt32

/-- Collect the native allocator's free pages. -/
def collectAllocator (force : Bool := true) : UInt32 :=
  collectAllocatorRaw (if force then 1 else 0)

/-- Allocate a length-`n` buffer filled with zeros. -/
@[extern "torchlean_cuda_buffer_zeros"]
opaque zeros (n : UInt32) : Buffer

/-- Allocate a length-`n` buffer filled with `v` (host `Float`, cast to float32). -/
@[extern "torchlean_cuda_buffer_full"]
opaque full (n : UInt32) (v : Float) : Buffer

/-!
### Deterministic RNG (device-side)

These are low-level building blocks used by TorchLean's seeded RNG ops (`rand_uniform`,
`bernoulli_mask`) when running on the eager CUDA backend.

They intentionally use the same SplitMix64-style mixing as `TorchLean.Random` so results are
deterministic given `(seed, counter)` and a row-major linear index.
-/

/-- Deterministic `U[0,1)` generator: returns a length-`n` buffer (float32) keyed by `key`. -/
@[extern "torchlean_cuda_buffer_rand_uniform"]
opaque randUniform (n : UInt32) (key : UInt64) : Buffer

/-- Deterministic `{0,1}` mask generator: returns a length-`n` buffer keyed by `key`. -/
@[extern "torchlean_cuda_buffer_bernoulli_mask"]
opaque bernoulliMask (n : UInt32) (keepProb : Float) (key : UInt64) : Buffer

/-- Unary ops. -/
@[extern "torchlean_cuda_buffer_abs"]
opaque abs (b : Buffer) : Buffer

/-- Backward for `abs`: `dx = sign(x) * dLdy` (with `sign(0)=0`). -/
@[extern "torchlean_cuda_buffer_abs_bwd"]
opaque absBwd (x dLdy : Buffer) : Buffer

@[extern "torchlean_cuda_buffer_sqrt"]
opaque sqrt (b : Buffer) : Buffer

/--
Backward for `sqrt`.

Uses the TorchLean convention: `dx = dLdy * (1 / (2*sqrt(x)))` for `x > 0`, else `0`.
-/
@[extern "torchlean_cuda_buffer_sqrt_bwd"]
opaque sqrtBwd (x dLdy : Buffer) : Buffer

@[extern "torchlean_cuda_buffer_exp"]
opaque exp (b : Buffer) : Buffer

@[extern "torchlean_cuda_buffer_log"]
opaque log (b : Buffer) : Buffer

/-- Reciprocal: `1/x`. -/
@[extern "torchlean_cuda_buffer_inv"]
opaque inv (b : Buffer) : Buffer

/-- Clamp each element to `[lo, hi]` (bounds are host `Float`s). -/
@[extern "torchlean_cuda_buffer_clamp"]
opaque clamp (b : Buffer) (lo hi : Float) : Buffer

/--
Backward for `clamp`.

Uses the TorchLean convention: derivative is `1` strictly inside `(lo, hi)`, else `0`.
-/
@[extern "torchlean_cuda_buffer_clamp_bwd"]
opaque clampBwd (x dLdy : Buffer) (lo hi : Float) : Buffer

/-- Binary elementwise ops (sizes must match). -/
@[extern "torchlean_cuda_buffer_max"]
opaque max (a b : Buffer) : Buffer

/--
Backward for `max`, returning `(dA, dB)`.

Tie-breaking follows the spec: when `a = b`, split upstream gradient evenly (`0.5`) between both.
-/
@[extern "torchlean_cuda_buffer_max_bwd"]
opaque maxBwd (a b dLdy : Buffer) : Buffer × Buffer

@[extern "torchlean_cuda_buffer_min"]
opaque min (a b : Buffer) : Buffer

/--
Backward for `min`, returning `(dA, dB)`.

Tie-breaking follows the spec: when `a = b`, split upstream gradient evenly (`0.5`) between both.
-/
@[extern "torchlean_cuda_buffer_min_bwd"]
opaque minBwd (a b dLdy : Buffer) : Buffer × Buffer

/-- Elementwise division. -/
@[extern "torchlean_cuda_buffer_div"]
opaque div (a b : Buffer) : Buffer

/-- Elementwise ReLU. -/
@[extern "torchlean_cuda_buffer_relu"]
opaque relu (b : Buffer) : Buffer

/-- Backward for `relu`: `dx = dLdy` where `x > 0`, else `0`. -/
@[extern "torchlean_cuda_buffer_relu_bwd"]
opaque reluBwd (x dLdy : Buffer) : Buffer

/-- Elementwise addition (sizes must match). -/
@[extern "torchlean_cuda_buffer_add"]
opaque add (a b : Buffer) : Buffer

/-- Elementwise subtraction (sizes must match). -/
@[extern "torchlean_cuda_buffer_sub"]
opaque sub (a b : Buffer) : Buffer

/-- Elementwise multiplication (sizes must match). -/
@[extern "torchlean_cuda_buffer_mul"]
opaque mul (a b : Buffer) : Buffer

/--
Multiply each element by a scalar `c` (host `Float`, cast to float32).

This is a primitive building block for many ops (e.g. scaling gradients).
-/
@[extern "torchlean_cuda_buffer_scale"]
opaque scale (b : Buffer) (c : Float) : Buffer

/-- Device-to-device copy, implemented as a scale-by-one kernel. -/
def copy (b : Buffer) : Buffer :=
  scale b 1.0

/--
Fused multiply-add: `a + c * b` (sizes must match; `c` is a host `Float`, cast to float32).

This is the classic BLAS-style `axpy` primitive and is useful for optimizers and bias-like updates.
-/
@[extern "torchlean_cuda_buffer_axpy"]
opaque axpy (a b : Buffer) (c : Float) : Buffer

/-- Reductions (return a length-1 buffer). -/
@[extern "torchlean_cuda_buffer_reduce_sum"]
opaque reduceSum (b : Buffer) : Buffer

@[extern "torchlean_cuda_buffer_reduce_mean"]
opaque reduceMean (b : Buffer) : Buffer

end Buffer

end Cuda
end Autograd
end Runtime
