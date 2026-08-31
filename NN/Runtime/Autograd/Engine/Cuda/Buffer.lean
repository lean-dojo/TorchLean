/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Engine.Cuda.Trusted

/-!
# CUDA Float32 Buffers

Low-level buffer operations for the native CUDA autograd runtime. CUDA builds use
`csrc/cuda/tensor/torchlean_cuda_tensor.cu`; ordinary CPU builds link the parity implementation in
`csrc/cuda/tensor/torchlean_cuda_tensor_stub.c` so that the same runtime interfaces remain testable.
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace Cuda

namespace Buffer

/-! ### Runtime Availability -/

/-- What implementation sits behind the CUDA FFI symbols in the current process. -/
inductive RuntimeStatus where
  /-- Default non-CUDA builds provide host-memory parity stubs for low-level tests. -/
  | cpuStub
  /-- The project was built with CUDA and at least one CUDA device is visible. -/
  | nativeAvailable
  /-- The project was built with CUDA, but no usable CUDA device is visible. -/
  | nativeUnavailable
  deriving DecidableEq, Repr

@[never_extract, extern "torchlean_cuda_runtime_status"]
opaque runtimeStatusRaw (token : UInt32) : UInt32

/-- Query whether the linked CUDA symbols are native or the CPU parity stubs. -/
def runtimeStatus (token : UInt32 := 0) : RuntimeStatus :=
  match runtimeStatusRaw token with
  | 0 => .cpuStub
  | 1 => .nativeAvailable
  | _ => .nativeUnavailable

/-- Require real CUDA execution for a user-selected CUDA session. -/
def requireNativeRuntime : IO Unit :=
  match runtimeStatus with
  | .nativeAvailable => pure ()
  | .cpuStub =>
      throw <| IO.userError
        "CUDA was requested, but this executable is linked to TorchLean's CPU parity stubs; rebuild and run with `-K cuda=true`"
  | .nativeUnavailable =>
      throw <| IO.userError
        "CUDA was requested and this is a CUDA build, but no usable CUDA device is visible"

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

@[never_extract, extern "torchlean_cuda_set_deterministic_reductions"]
opaque setDeterministicReductionsRaw (on : UInt32) : Unit

@[never_extract, extern "torchlean_cuda_get_deterministic_reductions_u"]
opaque getDeterministicReductionsRaw (u : UInt32) : UInt32

@[never_extract, extern "torchlean_cuda_set_deterministic_reductions_checked"]
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

/-! ### Allocator Telemetry -/

@[never_extract, extern "torchlean_cuda_allocator_live_bytes"]
opaque allocatorLiveBytesRaw (u : UInt32) : UInt64

@[never_extract, extern "torchlean_cuda_allocator_peak_bytes"]
opaque allocatorPeakBytesRaw (u : UInt32) : UInt64

@[never_extract, extern "torchlean_cuda_allocator_alloc_count"]
opaque allocatorAllocCountRaw (u : UInt32) : UInt64

@[never_extract, extern "torchlean_cuda_allocator_free_count"]
opaque allocatorFreeCountRaw (u : UInt32) : UInt64

@[never_extract, extern "torchlean_cuda_wrapper_live_count"]
opaque wrapperLiveCountRaw (u : UInt32) : UInt64

@[never_extract, extern "torchlean_cuda_wrapper_peak_count"]
opaque wrapperPeakCountRaw (u : UInt32) : UInt64

@[never_extract, extern "torchlean_cuda_wrapper_alloc_count"]
opaque wrapperAllocCountRaw (u : UInt32) : UInt64

@[never_extract, extern "torchlean_cuda_wrapper_finalize_count"]
opaque wrapperFinalizeCountRaw (u : UInt32) : UInt64

@[never_extract, extern "torchlean_cuda_allocator_device_free_bytes"]
opaque allocatorDeviceFreeBytesRaw (u : UInt32) : UInt64

@[never_extract, extern "torchlean_cuda_allocator_device_total_bytes"]
opaque allocatorDeviceTotalBytesRaw (u : UInt32) : UInt64

@[never_extract, extern "torchlean_cuda_allocator_cache_bytes"]
opaque allocatorCacheBytesRaw (u : UInt32) : UInt64

@[never_extract, extern "torchlean_cuda_allocator_cache_cap_bytes"]
opaque allocatorCacheCapBytesRaw (u : UInt32) : UInt64

/--
Snapshot of the CUDA buffer allocator.

`liveBytes`/`peakBytes` count device or stub payloads allocated by this runtime layer. The wrapper
counters track the Lean external objects that own those payloads; in a steady workload,
`wrapperAllocCount - wrapperFinalizeCount` should remain bounded. `deviceFreeBytes` and
`deviceTotalBytes` come from `cudaMemGetInfo` in the CUDA build and are `0` in the CPU stub.
Together these fields distinguish payload leaks, wrapper-lifetime leaks, and broader CUDA memory
pressure or fragmentation.

`cacheBytes` is the device memory held in the buffer reuse cache: dropped buffers awaiting reuse,
which are not counted in `liveBytes`. `cacheCapBytes` is the limit selected by
`TORCHLEAN_CUDA_CACHE_CAP_BYTES`; `0` means unbounded. Both fields are `0` in the CPU stub, which
keeps no cache.
-/
structure AllocatorStats where
  liveBytes : UInt64
  peakBytes : UInt64
  allocCount : UInt64
  freeCount : UInt64
  wrapperLiveCount : UInt64
  wrapperPeakCount : UInt64
  wrapperAllocCount : UInt64
  wrapperFinalizeCount : UInt64
  deviceFreeBytes : UInt64
  deviceTotalBytes : UInt64
  cacheBytes : UInt64
  cacheCapBytes : UInt64
deriving Repr

/--
Read the current CUDA allocator counters.

`token` is ignored by the native implementation.  It exists so call sites that sample repeatedly
can pass a changing value (for example, the training step), preventing Lean from treating repeated
FFI reads as identical pure expressions.
-/
def allocatorStatsWithToken (token : UInt32) : IO AllocatorStats := do
  pure
    { liveBytes := allocatorLiveBytesRaw token
      peakBytes := allocatorPeakBytesRaw token
      allocCount := allocatorAllocCountRaw token
      freeCount := allocatorFreeCountRaw token
      wrapperLiveCount := wrapperLiveCountRaw token
      wrapperPeakCount := wrapperPeakCountRaw token
      wrapperAllocCount := wrapperAllocCountRaw token
      wrapperFinalizeCount := wrapperFinalizeCountRaw token
      deviceFreeBytes := allocatorDeviceFreeBytesRaw token
      deviceTotalBytes := allocatorDeviceTotalBytesRaw token
      cacheBytes := allocatorCacheBytesRaw token
      cacheCapBytes := allocatorCacheCapBytesRaw token }

/-- Read the current CUDA allocator counters. Prefer `allocatorStatsWithToken` in repeated loops. -/
def allocatorStats : IO AllocatorStats :=
  allocatorStatsWithToken 0

/-- Format a byte count as MiB for allocator progress messages. -/
def mibString (bytes : UInt64) : String :=
  let mib := (Float.ofNat bytes.toNat) / (1024.0 * 1024.0)
  toString mib ++ " MiB"

/-- One-line allocator report for progress logs. -/
def AllocatorStats.format (s : AllocatorStats) : String :=
  "live=" ++ mibString s.liveBytes ++
  " peak=" ++ mibString s.peakBytes ++
  " allocs=" ++ toString s.allocCount ++
  " frees=" ++ toString s.freeCount ++
  " wrappers_live=" ++ toString s.wrapperLiveCount ++
  " wrappers_peak=" ++ toString s.wrapperPeakCount ++
  " wrappers_alloc=" ++ toString s.wrapperAllocCount ++
  " wrappers_finalized=" ++ toString s.wrapperFinalizeCount ++
  " cuda_free=" ++ mibString s.deviceFreeBytes ++
  " cuda_total=" ++ mibString s.deviceTotalBytes ++
  " cache=" ++ mibString s.cacheBytes ++
  " cache_cap=" ++ (if s.cacheCapBytes == 0 then "unbounded" else mibString s.cacheCapBytes)

/--
Create a device buffer by copying from a host `FloatArray` (casts each element to float32).

This primitive has a pure Lean type, but the native implementation allocates a fresh device buffer.
Runtime code that repeatedly uploads the same host value should prefer `ofFloatArrayIO`, which adds
an IO token so two uploads cannot be collapsed into the same external object after one is released.
-/
@[never_extract, extern "torchlean_cuda_buffer_of_float_array"]
opaque ofFloatArray (a : @& FloatArray) : Buffer

@[never_extract, extern "torchlean_cuda_buffer_of_float_array_with_token"]
opaque ofFloatArrayWithToken (a : @& FloatArray) (token : UInt32) : Buffer

/--
Effectful host-to-device upload.

The token is ignored by C/CUDA. Its purpose is semantic: repeated uploads of the same `FloatArray`
must still allocate distinct device buffers. Without a changing token, Lean can treat the extern as
a pure expression, which is not the ownership model we want for long eager CUDA training loops.
-/
def ofFloatArrayIO (a : @& FloatArray) : IO Buffer := do
  let t ← IO.monoNanosNow
  pure <| ofFloatArrayWithToken a (UInt32.ofNat t)

/-- Copy a buffer back to a host `FloatArray` (casts float32 elements to `Float`). -/
@[never_extract, extern "torchlean_cuda_buffer_to_float_array"]
opaque toFloatArray (b : @& Buffer) : FloatArray

@[never_extract, extern "torchlean_cuda_buffer_to_float_array_io"]
opaque toFloatArrayIO (b : @& Buffer) : IO FloatArray

/--
Copy a buffer to its raw float32 byte representation.

This is primarily used by streaming checkpoints. Unlike `toFloatArrayIO`, it does not widen every
element to Lean `Float`, so a large CUDA parameter can be written without constructing a second
double-precision host array.
-/
@[never_extract, extern "torchlean_cuda_buffer_to_float32_bytes_io"]
opaque toFloat32BytesIO (b : @& Buffer) : IO ByteArray

/-- Upload a raw float32 byte payload to a fresh buffer. -/
@[never_extract, extern "torchlean_cuda_buffer_of_float32_bytes_io"]
opaque ofFloat32BytesIO (bytes : @& ByteArray) : IO Buffer

/-- Encode a host `FloatArray` as raw float32 bytes. -/
@[never_extract, extern "torchlean_float_array_to_float32_bytes"]
opaque floatArrayToFloat32Bytes (values : @& FloatArray) : ByteArray

/-- Decode raw float32 bytes into a host `FloatArray`. -/
@[never_extract, extern "torchlean_float32_bytes_to_float_array"]
opaque float32BytesToFloatArray (bytes : @& ByteArray) : FloatArray

/-- Number of float32 elements in the buffer. -/
@[never_extract, extern "torchlean_cuda_buffer_size"]
opaque size (b : @& Buffer) : UInt32

@[never_extract, extern "torchlean_cuda_buffer_size_with_token"]
opaque sizeWithToken (b : @& Buffer) (token : UInt32) : UInt32

/-- Read a buffer size at a specific point in an `IO` ownership sequence. -/
def sizeIO (b : @& Buffer) : IO UInt32 := do
  let token ← IO.monoNanosNow
  pure <| sizeWithToken b (UInt32.ofNat token)

@[never_extract, extern "torchlean_cuda_buffer_release_with_token"]
opaque releaseWithToken (b : @& Buffer) (token : UInt32) : UInt32

/--
Effectfully release a device allocation owned by a completed runtime scope.

The changing token makes the release depend on the surrounding `IO` sequence. `Buffer` values are
copyable Lean references to one native allocation, so release invalidates every raw alias and Lean's
type system does not establish unique ownership. Callers must enforce that no alias remains usable;
session caches atomically remove their published alias before calling this function. Pure CUDA
formulas that retire an intermediate use `releaseThen`, which threads cleanup through the returned
buffer.
-/
def releaseIO (b : @& Buffer) : IO UInt32 := do
  let token ← IO.monoNanosNow
  pure <| releaseWithToken b (UInt32.ofNat token)

/--
Release `workspace` and return `keep`.

This exists for pure CUDA tape code: because the returned buffer is used downstream, Lean cannot
erase the native release call as dead code.
-/
@[never_extract, extern "torchlean_cuda_buffer_release_then"]
opaque releaseThen (workspace keep : @& Buffer) : Buffer

/--
Release a collection of workspace buffers and return `keep`.

Many CUDA tape formulas create a group of intermediate buffers, then continue with one final result
buffer. Threading cleanup through the result keeps ownership local to the formula and avoids waiting
for external-object finalizers in long training loops.
-/
def releaseManyThen (workspace : Array Buffer) (keep : @& Buffer) : Buffer :=
  workspace.foldr (fun b acc => releaseThen b acc) keep

/--
A CUDA result together with workspace buffers that were needed to compute it.

This is the common ownership shape for eager CUDA formulas.  Some forward computations need
intermediate buffers again during the backward pass, so the tape keeps those buffers on the node
and releases them when the node is retired.  Backward formulas use the same shape when they
recompute a value only to differentiate through it.
-/
structure WithWorkspace where
  value : Buffer
  workspace : Array Buffer := #[]

namespace WithWorkspace

/-- Return `keep` after releasing all workspace buffers owned by this result. -/
def releaseWorkspaceThen (r : WithWorkspace) (keep : @& Buffer) : Buffer :=
  releaseManyThen r.workspace keep

/-- Return `keep` after releasing both the result buffer and its workspace buffers. -/
def releaseAllThen (r : WithWorkspace) (keep : @& Buffer) : Buffer :=
  releaseThen r.value <| releaseManyThen r.workspace keep

end WithWorkspace

/--
Ask the Lean runtime allocator (mimalloc) to collect abandoned/free pages.

This allocator-only operation releases abandoned or free pages without changing TorchLean values.
It is intended for long native eager loops that create many short-lived tape closures and
external-buffer wrappers at every step.

In the CUDA build, `force = true` also releases cached device blocks held by the native buffer pool.
That gives training code a way to trade reuse for returning memory to the CUDA driver at clear
phase boundaries.
-/
@[never_extract, extern "torchlean_runtime_collect_allocator"]
opaque collectAllocatorRaw (force : UInt32) : UInt32

/-- Collect the native allocator's free pages. -/
def collectAllocator (force : Bool := true) : UInt32 :=
  collectAllocatorRaw (if force then 1 else 0)

/-- Allocate a length-`n` buffer filled with zeros. -/
@[never_extract, extern "torchlean_cuda_buffer_zeros"]
opaque zeros (n : UInt32) : Buffer

/-- Allocate a length-`n` buffer filled with `v` (host `Float`, cast to float32). -/
@[never_extract, extern "torchlean_cuda_buffer_full"]
opaque full (n : UInt32) (v : Float) : Buffer

@[never_extract, extern "torchlean_cuda_buffer_full_with_token"]
opaque fullWithToken (n : UInt32) (v : Float) (token : UInt32) : Buffer

/-- Allocate a fresh filled buffer inside a repeated runtime loop. -/
def fullIO (n : UInt32) (v : Float) : IO Buffer := do
  let token ← IO.monoNanosNow
  pure <| fullWithToken n v (UInt32.ofNat token)

/-!
### Deterministic RNG (device-side)

These are low-level building blocks used by TorchLean's seeded RNG ops (`rand_uniform`,
`bernoulli_mask`) when running on the eager CUDA backend.

They use the same SplitMix64-style mixing as `TorchLean.Random` so results are
deterministic given `(seed, counter)` and a row-major linear index.
-/

/-- Deterministic `U[0,1)` generator: returns a length-`n` buffer (float32) keyed by `key`. -/
@[never_extract, extern "torchlean_cuda_buffer_rand_uniform"]
opaque randUniform (n : UInt32) (key : UInt64) : Buffer

/-- Deterministic normal generator using Box-Muller on the device. -/
@[never_extract, extern "torchlean_cuda_buffer_rand_normal"]
opaque randNormal (n : UInt32) (mean std : Float) (key : UInt64) : Buffer

/-- Deterministic `{0,1}` mask generator: returns a length-`n` buffer keyed by `key`. -/
@[never_extract, extern "torchlean_cuda_buffer_bernoulli_mask"]
opaque bernoulliMask (n : UInt32) (keepProb : Float) (key : UInt64) : Buffer

/-- Absolute value applied pointwise to a CUDA buffer. -/
@[never_extract, extern "torchlean_cuda_buffer_abs"]
opaque abs (b : @& Buffer) : Buffer

/-- Backward for `abs`: `dx = sign(x) * dLdy` (with `sign(0)=0`). -/
@[never_extract, extern "torchlean_cuda_buffer_abs_bwd"]
opaque absBwd (x dLdy : @& Buffer) : Buffer

@[never_extract, extern "torchlean_cuda_buffer_sqrt"]
opaque sqrt (b : @& Buffer) : Buffer

/--
Backward for `sqrt`.

Uses the TorchLean convention: `dx = dLdy * (1 / (2*sqrt(x)))` for `x > 0`, else `0`.
-/
@[never_extract, extern "torchlean_cuda_buffer_sqrt_bwd"]
opaque sqrtBwd (x dLdy : @& Buffer) : Buffer

@[never_extract, extern "torchlean_cuda_buffer_exp"]
opaque exp (b : @& Buffer) : Buffer

@[never_extract, extern "torchlean_cuda_buffer_log"]
opaque log (b : @& Buffer) : Buffer

/-- Reciprocal: `1/x`. -/
@[never_extract, extern "torchlean_cuda_buffer_inv"]
opaque inv (b : @& Buffer) : Buffer

/-- Clamp each element to `[lo, hi]` (bounds are host `Float`s). -/
@[never_extract, extern "torchlean_cuda_buffer_clamp"]
opaque clamp (b : @& Buffer) (lo hi : Float) : Buffer

/--
Backward for `clamp`.

Uses the TorchLean convention: derivative is `1` strictly inside `(lo, hi)`, else `0`.
-/
@[never_extract, extern "torchlean_cuda_buffer_clamp_bwd"]
opaque clampBwd (x dLdy : @& Buffer) (lo hi : Float) : Buffer

/-- Pointwise maximum of two equal-length CUDA buffers. -/
@[never_extract, extern "torchlean_cuda_buffer_max"]
opaque max (a b : @& Buffer) : Buffer

/--
Backward for `max`, returning `(dA, dB)`.

Tie-breaking follows the spec: when `a = b`, split upstream gradient evenly (`0.5`) between both.
-/
@[never_extract, extern "torchlean_cuda_buffer_max_bwd"]
opaque maxBwd (a b dLdy : @& Buffer) : Buffer × Buffer

@[never_extract, extern "torchlean_cuda_buffer_min"]
opaque min (a b : @& Buffer) : Buffer

/--
Backward for `min`, returning `(dA, dB)`.

Tie-breaking follows the spec: when `a = b`, split upstream gradient evenly (`0.5`) between both.
-/
@[never_extract, extern "torchlean_cuda_buffer_min_bwd"]
opaque minBwd (a b dLdy : @& Buffer) : Buffer × Buffer

/-- Pointwise division of two equal-length CUDA buffers. -/
@[never_extract, extern "torchlean_cuda_buffer_div"]
opaque div (a b : @& Buffer) : Buffer

/-- Pointwise ReLU activation on a CUDA buffer. -/
@[never_extract, extern "torchlean_cuda_buffer_relu"]
opaque relu (b : @& Buffer) : Buffer

/-- Backward for `relu`: `dx = dLdy` where `x > 0`, else `0`. -/
@[never_extract, extern "torchlean_cuda_buffer_relu_bwd"]
opaque reluBwd (x dLdy : @& Buffer) : Buffer

/-- Tanh-approximate GELU evaluated by one pointwise CUDA kernel. -/
@[never_extract, extern "torchlean_cuda_buffer_gelu"]
opaque gelu (x : @& Buffer) : Buffer

/-- Backward for tanh-approximate GELU using `Activation.geluDerivSpec`. -/
@[never_extract, extern "torchlean_cuda_buffer_gelu_bwd"]
opaque geluBwd (x dLdy : @& Buffer) : Buffer

/-- Elementwise addition (sizes must match). -/
@[never_extract, extern "torchlean_cuda_buffer_add"]
opaque add (a b : @& Buffer) : Buffer

/-- Elementwise subtraction (sizes must match). -/
@[never_extract, extern "torchlean_cuda_buffer_sub"]
opaque sub (a b : @& Buffer) : Buffer

/-- Elementwise multiplication (sizes must match). -/
@[never_extract, extern "torchlean_cuda_buffer_mul"]
opaque mul (a b : @& Buffer) : Buffer

/--
Multiply each element by a scalar `c` (host `Float`, cast to float32).

This is a primitive building block for many ops (e.g. scaling gradients).
-/
@[never_extract, extern "torchlean_cuda_buffer_scale"]
opaque scale (b : @& Buffer) (c : Float) : Buffer

/-- Device-to-device copy, implemented as a scale-by-one kernel. -/
def copy (b : @& Buffer) : Buffer :=
  scale b 1.0

/--
Copy a buffer and release the source after the copy has been produced.

The native operation creates the destination before it retires the source, so the compiler cannot
reorder the two lifetime events. Use this at ownership-transfer boundaries in the sparse CUDA tape.
-/
@[never_extract, extern "torchlean_cuda_buffer_copy_and_release"]
opaque copyAndRelease (b : @& Buffer) : Buffer

/--
Fused multiply-add: `a + c * b` (sizes must match; `c` is a host `Float`, cast to float32).

This is the classic BLAS-style `axpy` primitive and is useful for optimizers and bias-like updates.
-/
@[never_extract, extern "torchlean_cuda_buffer_axpy"]
opaque axpy (a b : @& Buffer) (c : Float) : Buffer

/--
Perform one Adam-family update in a single CUDA pass.

The result is `(parameters, firstMoment, secondMoment)`. Passing `decay = 0` gives Adam; passing
`decay = -(learningRate * weightDecay)` gives AdamW's decoupled parameter decay. The caller
computes the two bias-correction scales from the step counter, exactly as in `Optim.Adam.update`
and `Optim.AdamW.update`.

This primitive changes only the execution plan. TorchLean's optimizer definitions remain the
semantic reference, while this native boundary avoids materializing every intermediate tensor in
the pointwise update.
-/
@[never_extract, extern "torchlean_cuda_buffer_adam_step"]
opaque adamStep
    (parameters gradient firstMoment secondMoment : @& Buffer)
    (beta1 oneMinusBeta1 beta2 oneMinusBeta2 : Float)
    (firstMomentCorrection secondMomentCorrection epsilon : Float)
    (decay updateScale : Float) :
    Buffer × Buffer × Buffer

/--
Scaled product exponential: `exp((c * x) * y)`, a single fused device kernel — one launch and one
result buffer instead of the four elementwise ops (`full c`, two `mul`s, `exp`) of the composed form,
and bit-identical to it (same left-association, same fp32 rounding). `c` is a host `Float` (cast to
float32); `x` and `y` are equal-length buffers.

Domain-neutral: a *scaled product exponential* recurs across the sciences — a Beer–Lambert /
propagation two-way extinction `exp(-2 * κ * ℓ)` in computational electromagnetism and radar/optical
remote sensing, or a Boltzmann-type weight `exp(-β * E * s)`. Fusing the exponential with its scaled
product is the hot inner form in those forward models.
-/
@[never_extract, extern "torchlean_cuda_buffer_scaled_prod_exp"]
opaque scaledProdExp (x y : @& Buffer) (c : Float) : Buffer

/-- Reductions (return a length-1 buffer). -/
@[never_extract, extern "torchlean_cuda_buffer_reduce_sum"]
opaque reduceSum (b : @& Buffer) : Buffer

@[never_extract, extern "torchlean_cuda_buffer_reduce_mean"]
opaque reduceMean (b : @& Buffer) : Buffer

end Buffer

end Cuda
end Autograd
end Runtime
