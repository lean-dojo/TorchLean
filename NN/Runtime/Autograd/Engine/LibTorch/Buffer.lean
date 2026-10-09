/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Engine.LibTorch.Controls
public import NN.Runtime.Autograd.Engine.LibTorch.Trusted
public import NN.Kernel.Cuda.Binary

/-!
# CUDA Buffers

Low-level dtype-preserving operations for the LibTorch CUDA runtime. TorchLean retains its tape and
selected local VJPs; native calls do not record a LibTorch autograd graph. CUDA builds use
`csrc/libtorch/torchlean.cpp`. Builds without LibTorch link `csrc/libtorch/unavailable.c`, which
reports `.notLinked` and fails every buffer operation.
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace LibTorch

namespace Buffer

/-- Scalar representation retained by a live native buffer. Released handles must not be read. -/
@[never_extract, extern "torchlean_cuda_buffer_dtype"]
opaque dtype (b : @& Buffer) : Dtype

/-- Complete binary format retained by an encoded buffer; native buffers return `none`. -/
@[never_extract, extern "torchlean_cuda_buffer_format"]
opaque format? (b : @& Buffer) : Option FloatLib.Floats.Formats.BinaryInterchange.FloatFormat

/-- Upload complete configured words without converting through a native floating-point type. -/
@[never_extract, extern "torchlean_cuda_buffer_of_encoded_io"]
opaque ofEncodedIO (bytes : @& ByteArray)
    (format : @& FloatLib.Floats.Formats.BinaryInterchange.FloatFormat) (width : UInt64) : IO Buffer

/-- Download complete configured words, retaining signed zero and exceptional-value encodings. -/
@[never_extract, extern "torchlean_cuda_buffer_to_encoded_io"]
opaque toEncodedIO (buffer : @& Buffer) : IO ByteArray

@[never_extract, extern "torchlean_kernel_run_encoded"]
private opaque runEncoded (source : @& String) (inputs : @& Array Buffer)
    (format : @& FloatLib.Floats.Formats.BinaryInterchange.FloatFormat)
    (width count : UInt64) : Buffer

@[never_extract, extern "torchlean_cuda_buffer_fail"]
private opaque fail (message : @& String) : Buffer

/-! ### Runtime Availability -/

/-- What implementation sits behind the CUDA FFI symbols in the current process. -/
inductive RuntimeStatus where
  /-- The default build does not link LibTorch; buffer operations fail. -/
  | notLinked
  /-- The project was built with LibTorch CUDA and at least one CUDA device is visible. -/
  | nativeAvailable
  /-- The project was built with LibTorch CUDA, but no usable CUDA device is visible. -/
  | nativeUnavailable
  deriving DecidableEq, Repr

/-- Raw status word from the C layer; `runtimeStatus` decodes it. -/
@[never_extract, extern "torchlean_cuda_runtime_status"]
private opaque runtimeStatusRaw (token : UInt32) : UInt32

/-- Query whether LibTorch is linked and can see a CUDA device. -/
@[no_expose] def runtimeStatus (token : UInt32 := 0) : RuntimeStatus :=
  match runtimeStatusRaw token with
  | 0 => .notLinked
  | 1 => .nativeAvailable
  | _ => .nativeUnavailable

/-- Require real CUDA execution for a user-selected CUDA session. -/
def requireNativeRuntime : IO Unit :=
  match runtimeStatus with
  | .nativeAvailable => pure ()
  | .notLinked =>
      throw <| IO.userError
        "CUDA was requested, but this executable was built without LibTorch; \
          rebuild and run with `-K cuda=true`"
  | .nativeUnavailable =>
      throw <| IO.userError
        "CUDA was requested and this is a CUDA build, but no usable CUDA device is visible"

/-! ### Allocator Telemetry -/

/-- Logical payload bytes currently owned by TorchLean buffer handles. -/
@[never_extract, extern "torchlean_cuda_allocator_live_bytes"]
private opaque allocatorLiveBytesRaw (u : UInt32) : UInt64

/-- High-water mark of `allocatorLiveBytesRaw` since process start. -/
@[never_extract, extern "torchlean_cuda_allocator_peak_bytes"]
private opaque allocatorPeakBytesRaw (u : UInt32) : UInt64

/-- Nonempty buffer payload owners created, independently of native allocator reuse. -/
@[never_extract, extern "torchlean_cuda_allocator_alloc_count"]
private opaque allocatorAllocCountRaw (u : UInt32) : UInt64

/-- Nonempty buffer payload owners retired, independently of native allocator reuse. -/
@[never_extract, extern "torchlean_cuda_allocator_free_count"]
private opaque allocatorFreeCountRaw (u : UInt32) : UInt64

/-- Live external buffer wrappers, including empty or explicitly released wrappers. -/
@[never_extract, extern "torchlean_cuda_wrapper_live_count"]
private opaque wrapperLiveCountRaw (u : UInt32) : UInt64

/-- High-water mark of that live wrapper count. -/
@[never_extract, extern "torchlean_cuda_wrapper_peak_count"]
private opaque wrapperPeakCountRaw (u : UInt32) : UInt64

/-- Wrappers created since process start. -/
@[never_extract, extern "torchlean_cuda_wrapper_alloc_count"]
private opaque wrapperAllocCountRaw (u : UInt32) : UInt64

/-- Wrappers finalized by the Lean garbage collector since process start. -/
@[never_extract, extern "torchlean_cuda_wrapper_finalize_count"]
private opaque wrapperFinalizeCountRaw (u : UInt32) : UInt64

/-- Bytes the device reports as free, as seen by the driver rather than by this allocator. -/
@[never_extract, extern "torchlean_cuda_allocator_device_free_bytes"]
private opaque allocatorDeviceFreeBytesRaw (u : UInt32) : UInt64

/-- Total device memory reported by the driver. -/
@[never_extract, extern "torchlean_cuda_allocator_device_total_bytes"]
private opaque allocatorDeviceTotalBytesRaw (u : UInt32) : UInt64

/-- Selected-device bytes currently allocated by the LibTorch CUDA allocator. -/
@[never_extract, extern "torchlean_libtorch_allocated_bytes"]
private opaque allocatedBytesRaw (token : UInt32) : UInt64

/-- Selected-device bytes currently reserved by the LibTorch CUDA allocator. -/
@[never_extract, extern "torchlean_libtorch_reserved_bytes"]
private opaque reservedBytesRaw (token : UInt32) : UInt64

/-- Peak selected-device allocation since the native allocator's last peak reset. -/
@[never_extract, extern "torchlean_libtorch_peak_allocated_bytes"]
private opaque peakAllocatedBytesRaw (token : UInt32) : UInt64

/-- Peak selected-device reservation since the native allocator's last peak reset. -/
@[never_extract, extern "torchlean_libtorch_peak_reserved_bytes"]
private opaque peakReservedBytesRaw (token : UInt32) : UInt64

/--
Snapshot of TorchLean ownership counters and the LibTorch CUDA allocator.

`liveBytes`/`peakBytes` count logical device payloads owned by TorchLean handles. They are
not physical VRAM usage: shared tensor storage may be counted more than once, and native
temporaries or saved attention state need not have a separate TorchLean handle. `allocCount` and
`freeCount` count those payload lifetimes, not native allocation calls.

The wrapper counters track Lean external buffer objects, including empty and explicitly released
wrappers. All these ownership counters are process-wide. They help distinguish retained Lean
owners from storage managed internally by LibTorch.

`allocatedBytes`, `reservedBytes`, `peakAllocatedBytes`, and `peakReservedBytes` are the native
allocator's counters for the selected CUDA device. Reserved bytes include storage the allocator
retains for reuse; the difference from allocated bytes is not a promise of immediately reclaimable
memory. `deviceFreeBytes` and `deviceTotalBytes` report driver memory for that device. These six
device counters are zero when LibTorch is not linked. Native allocator peaks follow its own reset
lifecycle.

LibTorch owns caching policy. `Runtime.Autograd.LibTorch.setMemoryFraction` configures its
selected-device allocation
limit.

The fields are read separately. A snapshot can include concurrent allocator activity and should
not be treated as an atomic account of every allocation in the process.
-/
structure Memory where
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
  allocatedBytes : UInt64
  reservedBytes : UInt64
  peakAllocatedBytes : UInt64
  peakReservedBytes : UInt64
deriving Repr

/--
Read current memory usage and buffer ownership counters.

Each read is sequenced at this point in `IO`, including repeated calls in a loop. The native reads
stay inside the action rather than constructing a record that Lean could retain from an earlier
call. Applications do not need a step counter or another changing argument to obtain fresh values.
-/
@[no_expose] def memory : IO Memory :=
  IO.lazyPure fun _ =>
    { liveBytes := allocatorLiveBytesRaw 0
      peakBytes := allocatorPeakBytesRaw 0
      allocCount := allocatorAllocCountRaw 0
      freeCount := allocatorFreeCountRaw 0
      wrapperLiveCount := wrapperLiveCountRaw 0
      wrapperPeakCount := wrapperPeakCountRaw 0
      wrapperAllocCount := wrapperAllocCountRaw 0
      wrapperFinalizeCount := wrapperFinalizeCountRaw 0
      deviceFreeBytes := allocatorDeviceFreeBytesRaw 0
      deviceTotalBytes := allocatorDeviceTotalBytesRaw 0
      allocatedBytes := allocatedBytesRaw 0
      reservedBytes := reservedBytesRaw 0
      peakAllocatedBytes := peakAllocatedBytesRaw 0
      peakReservedBytes := peakReservedBytesRaw 0 }

/-- Format a byte count as MiB for allocator progress messages. -/
@[no_expose] private def mibString (bytes : UInt64) : String :=
  let mib := (Float.ofNat bytes.toNat) / (1024.0 * 1024.0)
  toString mib ++ " MiB"

/-- One-line allocator report distinguishing logical payloads from native storage. -/
@[no_expose] def Memory.format (s : Memory) : String :=
  "payload=" ++ mibString s.liveBytes ++
  " payload_peak=" ++ mibString s.peakBytes ++
  " allocs=" ++ toString s.allocCount ++
  " frees=" ++ toString s.freeCount ++
  " wrappers_live=" ++ toString s.wrapperLiveCount ++
  " wrappers_peak=" ++ toString s.wrapperPeakCount ++
  " wrappers_alloc=" ++ toString s.wrapperAllocCount ++
  " wrappers_finalized=" ++ toString s.wrapperFinalizeCount ++
  " cuda_free=" ++ mibString s.deviceFreeBytes ++
  " cuda_total=" ++ mibString s.deviceTotalBytes ++
  " allocated=" ++ mibString s.allocatedBytes ++
  " reserved=" ++ mibString s.reservedBytes ++
  " peak_allocated=" ++ mibString s.peakAllocatedBytes ++
  " peak_reserved=" ++ mibString s.peakReservedBytes

/--
Create a device buffer by copying from a host `FloatArray` in the requested dtype.

This primitive has a pure Lean type, but the native implementation allocates a fresh device buffer.
Runtime code should use `ofFloatArrayIO`: each call allocates a distinct buffer, and ordinary device
exhaustion is returned as an IO error that the caller can handle.
-/
@[never_extract, extern "torchlean_cuda_buffer_of_float_array"]
opaque ofFloatArray (a : @& FloatArray) (dtype : Dtype := .float32) : Buffer

/--
Copy a host `FloatArray` into a fresh device buffer in the requested dtype.

The upload runs at this point in the IO sequence. Repeated calls with the same host array allocate
distinct buffers, so releasing one does not invalidate another. Native allocation failures return
`IO.Error.resourceExhausted`; LibTorch owns the device allocator's retry policy. The host array and
existing device buffers remain owned by the caller.
-/
@[never_extract, extern "torchlean_cuda_buffer_of_float_array_io"]
opaque ofFloatArrayIO (a : @& FloatArray) (dtype : Dtype := .float32) : IO Buffer

/-- Copy a buffer back to a host `FloatArray`, widening binary32 when needed. -/
@[never_extract, extern "torchlean_cuda_buffer_to_float_array"]
opaque toFloatArray (b : @& Buffer) : FloatArray

/-- Download a buffer to the host, widening each element to Lean `Float`. -/
@[never_extract, extern "torchlean_cuda_buffer_to_float_array_io"]
opaque toFloatArrayIO (b : @& Buffer) : IO FloatArray

/--
Copy a buffer to little-endian scalar words. By default its dtype is preserved; an explicit
different dtype requests conversion, as when exporting a legacy binary32 checkpoint.

This is primarily used by streaming checkpoints. Unlike `toFloatArrayIO`, it does not widen every
element to Lean `Float`, so a large CUDA parameter can be written without constructing a second
double-precision host array for binary32 data.
-/
@[never_extract, extern "torchlean_cuda_buffer_to_bytes_io"]
opaque toBytesIO (b : @& Buffer) (dtype : Dtype := Buffer.dtype b) : IO ByteArray

/--
Upload little-endian scalar words to a fresh device buffer.

`source` describes the payload; `dtype` selects storage. They agree by default. A different
source is an explicit conversion, used to load legacy binary32 files into binary64 storage.
Native allocation failures return
`IO.Error.resourceExhausted` before a buffer is returned. The borrowed byte payload remains
available to the caller.
-/
@[never_extract, extern "torchlean_cuda_buffer_of_bytes_io"]
opaque ofBytesIO (bytes : @& ByteArray) (dtype : Dtype := .float32)
    (source : Dtype := dtype) : IO Buffer

/-- Number of scalar elements in the buffer, independently of dtype. -/
@[never_extract, extern "torchlean_cuda_buffer_size"]
opaque size (b : @& Buffer) : UInt32

/-- Evaluate a configured arithmetic expression on resident tape buffers. -/
private def encoded (reference : Buffer) (inputs : Array Buffer)
    (expr : NN.Kernel.Expr Nat [.index] .scalar) : Buffer :=
  match format? reference with
  | none => fail "autograd: configured arithmetic requires a binary format"
  | some format =>
    if inputs.all (fun input => format? input == some format && size input == size reference) then
      match NN.Kernel.Cuda.binarySource format id inputs.size expr with
      | .error message => fail message
      | .ok source => runEncoded source inputs format
          ((NN.Kernel.Precision.binary format).bytes.toUInt64) (size reference).toUInt64
    else fail "autograd: configured arithmetic operands have different formats or sizes"

/-- Element count read at an explicit token; `sizeIO` supplies a changing one. -/
@[never_extract, extern "torchlean_cuda_buffer_size_with_token"]
private opaque sizeWithToken (b : @& Buffer) (token : UInt32) : UInt32

/-- Read a buffer size at a specific point in an `IO` ownership sequence. -/
@[no_expose] def sizeIO (b : @& Buffer) : IO UInt32 := do
  let token ← IO.monoNanosNow
  pure <| sizeWithToken b (UInt32.ofNat token)

/-- Release at an explicit token; the returned word is what keeps the call alive. -/
@[never_extract, extern "torchlean_cuda_buffer_release_with_token"]
private opaque releaseWithToken (b : @& Buffer) (token : UInt32) : UInt32

/--
Effectfully release a device allocation owned by a completed runtime scope.

The changing token makes the release depend on the surrounding `IO` sequence. `Buffer` values are
copyable Lean references to one native allocation, so release invalidates every raw alias and Lean's
type system does not establish unique ownership. Callers must enforce that no alias remains usable;
removing one cache reference is insufficient when a tape still retains the same buffer. Parameter
mirrors and their recorded snapshots therefore use ordinary Lean reference counting. Pure CUDA
formulas that retire an owned intermediate use `releaseThen`, which threads cleanup through the
returned buffer.
-/
@[no_expose] def releaseIO (b : @& Buffer) : IO UInt32 := do
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

/-- Host allocator collection at an explicit token, retaining reusable storage. -/
@[never_extract, extern "torchlean_runtime_collect_allocator"]
private opaque collectAllocatorRaw (token : UInt32) : UInt32

/--
Collect unused host allocator pages while retaining CUDA buffers for reuse.

Training and evaluation call this after retiring a completed tape and its temporary gradients.
LibTorch retains its reusable device cache. Live parameters and optimizer state remain owned
by their sessions.
-/
@[no_expose] def collectGarbage : IO Unit := do
  let collected ← IO.lazyPure fun _ => collectAllocatorRaw 0
  if collected == 0 then
    throw <| IO.userError "CUDA allocator collection failed"

/-- Allocate a length-`n` buffer filled with zeros. -/
@[never_extract, extern "torchlean_cuda_buffer_zeros"]
opaque zeros (n : UInt32) (dtype : Dtype := .float32) : Buffer

/-- Allocate zeros in the reference buffer's complete scalar format. -/
@[never_extract, extern "torchlean_cuda_buffer_zeros_like"]
opaque zerosLike (reference : @& Buffer) (count : UInt32 := size reference) : Buffer

/--
Allocate a fresh zero-filled buffer inside `IO` code.

The allocation runs at this point in the IO sequence. Native allocation failures return
`IO.Error.resourceExhausted`, so a caller can release its temporary buffers and try a smaller
allocation. LibTorch owns the device allocator's retry policy. Existing live buffers remain owned
by their callers, and a failed allocation adds no live buffer to the counters.

The allocating IO constructors share this error boundary. Pure allocation and kernel primitives
have their own native failure policy.
-/
@[never_extract, extern "torchlean_cuda_buffer_zeros_io"]
opaque zerosIO (n : UInt32) (dtype : Dtype := .float32) : IO Buffer

/-- Allocate a length-`n` buffer filled with `v`, rounded in the requested dtype. -/
@[never_extract, extern "torchlean_cuda_buffer_full"]
opaque full (n : UInt32) (v : Float) (dtype : Dtype := .float32) : Buffer

/--
Allocate a fresh length-`n` buffer filled with `v`, rounded in the requested dtype.

Each call owns a distinct buffer. Native allocation failures return `IO.Error.resourceExhausted`,
as in `zerosIO`.
-/
@[never_extract, extern "torchlean_cuda_buffer_full_io"]
opaque fullIO (n : UInt32) (v : Float) (dtype : Dtype := .float32) : IO Buffer

/-!
### Deterministic RNG (device-side)

These are low-level building blocks used by TorchLean's seeded RNG ops (`rand_uniform`,
`bernoulli_mask`) when running on the eager CUDA backend.

They use the same SplitMix64-style mixing as `TorchLean.Random` so results are
deterministic given `(seed, counter)` and a row-major linear index.
-/

/-- Deterministic `U[0,1)` generator in the requested dtype, keyed by `key`. -/
@[never_extract, extern "torchlean_cuda_buffer_rand_uniform"]
opaque randUniform (n : UInt32) (key : UInt64) (dtype : Dtype := .float32) : Buffer

/--
Generate the deterministic values of `randUniform` in a fresh buffer.

The key determines the values; repeated calls still allocate distinct buffers. Native allocation
failures return `IO.Error.resourceExhausted`, as in `zerosIO`.
-/
@[never_extract, extern "torchlean_cuda_buffer_rand_uniform_io"]
opaque randUniformIO (n : UInt32) (key : UInt64) (dtype : Dtype := .float32) : IO Buffer

/-- Deterministic normal generator using Box-Muller on the device. -/
@[never_extract, extern "torchlean_cuda_buffer_rand_normal"]
opaque randNormal (n : UInt32) (mean std : Float) (key : UInt64)
    (dtype : Dtype := .float32) : Buffer

/-- Deterministic `{0,1}` mask generator: returns a length-`n` buffer keyed by `key`. -/
@[never_extract, extern "torchlean_cuda_buffer_bernoulli_mask"]
opaque bernoulliMask (n : UInt32) (keepProb : Float) (key : UInt64)
    (dtype : Dtype := .float32) : Buffer

/--
Generate the deterministic mask of `bernoulliMask` in a fresh buffer.

The probability and key retain the pure primitive's meaning. Each call allocates independently,
with the allocation error boundary of `zerosIO`.
-/
@[never_extract, extern "torchlean_cuda_buffer_bernoulli_mask_io"]
opaque bernoulliMaskIO (n : UInt32) (keepProb : Float) (key : UInt64)
    (dtype : Dtype := .float32) : IO Buffer

/-- Absolute value applied pointwise to a CUDA buffer. -/
@[never_extract, extern "torchlean_cuda_buffer_abs"]
opaque abs (b : @& Buffer) : Buffer

/-- Backward for `abs`: `dx = sign(x) * dLdy` (with `sign(0)=0`). -/
@[never_extract, extern "torchlean_cuda_buffer_abs_bwd"]
opaque absBwd (x dLdy : @& Buffer) : Buffer

/-- Elementwise `sqrt (max x 0)`, matching `Tensor.sqrtSpec`.

Negative inputs and either signed zero return positive zero. NaN inputs remain NaN; the clamp
uses the same ordered comparison as the floating-point maximum in the tensor spec. -/
@[never_extract, extern "torchlean_cuda_buffer_sqrt"]
opaque sqrt (b : @& Buffer) : Buffer

/--
Backward for `sqrt`.

Uses the TorchLean convention: `dx = dLdy * (1 / (2*sqrt(x)))` for `x > 0`, else `0`.
-/
@[never_extract, extern "torchlean_cuda_buffer_sqrt_bwd"]
opaque sqrtBwd (x dLdy : @& Buffer) : Buffer

/-- Elementwise `exp`. -/
@[never_extract, extern "torchlean_cuda_buffer_exp"]
opaque exp (b : @& Buffer) : Buffer

/--
Elementwise sine of angles in radians, retaining the input dtype.

The native implementation uses `at::sin`. The input is borrowed, so the tape can
retain it for the cosine factor in the backward pass.
-/
@[never_extract, extern "torchlean_cuda_buffer_sin"]
opaque sin (b : @& Buffer) : Buffer

/--
Elementwise cosine of angles in radians, retaining the input dtype and borrowing its input.
-/
@[never_extract, extern "torchlean_cuda_buffer_cos"]
opaque cos (b : @& Buffer) : Buffer

/-- Elementwise natural logarithm. -/
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

/-- Elementwise minimum of two buffers. -/
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
private opaque divNative (a b : @& Buffer) : Buffer

/-- Pointwise quotient in the stored format; exceptional values follow that format. -/
@[no_expose] def div (a b : @& Buffer) : Buffer :=
  if (format? a).isSome then
    encoded a #[a, b] (.binary .div (.load 0 (.var .zero)) (.load 1 (.var .zero)))
  else divNative a b

/-- Direct pointwise negation, including the sign of zero. -/
@[never_extract, extern "torchlean_cuda_buffer_neg"]
private opaque negNative (x : @& Buffer) : Buffer

/-- Direct negation, including the format's signed-zero convention. -/
@[no_expose] def neg (x : @& Buffer) : Buffer :=
  if (format? x).isSome then encoded x #[x] (.neg (.load 0 (.var .zero)))
  else negNative x

/--
Keep values where the equal-length mask is nonzero and select zero elsewhere.

Selection discards nonfinite values in blocked entries; multiplication by the mask would not.
A NaN mask is nonzero, matching the numerical backend's boolean conversion.
-/
@[never_extract, extern "torchlean_cuda_buffer_mask"]
opaque mask (values allowed : @& Buffer) : Buffer

/-- Pointwise ReLU activation on a CUDA buffer. -/
@[never_extract, extern "torchlean_cuda_buffer_relu"]
opaque relu (b : @& Buffer) : Buffer

/-- Backward for `relu`: `dx = dLdy` where `x > 0`, else `0`. -/
@[never_extract, extern "torchlean_cuda_buffer_relu_bwd"]
opaque reluBwd (x dLdy : @& Buffer) : Buffer

/--
Elementwise logistic sigmoid through `at::sigmoid`, retaining the input dtype.

The input is borrowed; TorchLean's tape owns the `y * (1 - y)` derivative. Values follow ATen's
rounding and saturation in that dtype, including zero when the negative-tail exponential overflows.
-/
@[never_extract, extern "torchlean_cuda_buffer_sigmoid"]
opaque sigmoid (x : @& Buffer) : Buffer

/--
Elementwise hyperbolic tangent through `at::tanh`, retaining the input dtype.

Direct evaluation avoids cancellation near zero and preserves signed zero. The input is borrowed;
TorchLean's tape owns the `1 - y * y` derivative.
-/
@[never_extract, extern "torchlean_cuda_buffer_tanh"]
opaque tanh (x : @& Buffer) : Buffer

/-- Tanh-approximate GELU evaluated through staged ATen operations.
The arithmetic ordering follows TorchLean's activation specification. -/
@[never_extract, extern "torchlean_cuda_buffer_gelu"]
opaque gelu (x : @& Buffer) : Buffer

/-- Backward for tanh-approximate GELU using `Activation.geluDerivSpec`. -/
@[never_extract, extern "torchlean_cuda_buffer_gelu_bwd"]
opaque geluBwd (x dLdy : @& Buffer) : Buffer

/-- Elementwise addition (sizes must match). -/
@[never_extract, extern "torchlean_cuda_buffer_add"]
private opaque addNative (a b : @& Buffer) : Buffer

/-- Pointwise addition, retaining the complete configured format when present. -/
@[no_expose] def add (a b : @& Buffer) : Buffer :=
  if (format? a).isSome then
    encoded a #[a, b] (.binary .add (.load 0 (.var .zero)) (.load 1 (.var .zero)))
  else addNative a b

/-- Elementwise subtraction (sizes must match). -/
@[never_extract, extern "torchlean_cuda_buffer_sub"]
private opaque subNative (a b : @& Buffer) : Buffer

/-- Pointwise subtraction without implicit precision conversion. -/
@[no_expose] def sub (a b : @& Buffer) : Buffer :=
  if (format? a).isSome then
    encoded a #[a, b] (.binary .sub (.load 0 (.var .zero)) (.load 1 (.var .zero)))
  else subNative a b

/-- Elementwise multiplication (sizes must match). -/
@[never_extract, extern "torchlean_cuda_buffer_mul"]
private opaque mulNative (a b : @& Buffer) : Buffer

/-- Pointwise multiplication without implicit precision conversion. -/
@[no_expose] def mul (a b : @& Buffer) : Buffer :=
  if (format? a).isSome then
    encoded a #[a, b] (.binary .mul (.load 0 (.var .zero)) (.load 1 (.var .zero)))
  else mulNative a b

/--
Multiply each element by a scalar `c`, rounded in the buffer dtype.

This is a primitive building block for many ops (e.g. scaling gradients).
-/
@[never_extract, extern "torchlean_cuda_buffer_scale"]
opaque scale (b : @& Buffer) (c : Float) : Buffer

/-- Device-to-device copy without arithmetic or precision conversion. -/
@[never_extract, extern "torchlean_cuda_buffer_copy"]
opaque copy (b : @& Buffer) : Buffer

/-- Two independently owned copies. Separate identical pure calls can be shared by Lean's
compiler, which is unsafe when one VJP contribution is retired before the other is consumed. -/
@[never_extract, extern "torchlean_cuda_buffer_duplicate"]
opaque duplicate (b : @& Buffer) : Buffer × Buffer

/--
Copy a buffer and release the source after the copy has been produced.

The native operation creates the destination before it retires the source, so the compiler cannot
reorder the two lifetime events. Use this at ownership-transfer boundaries in the sparse CUDA tape.
-/
@[never_extract, extern "torchlean_cuda_buffer_copy_and_release"]
private opaque copyAndReleaseNative (b : @& Buffer) : Buffer

/-- Copy before retiring the source, preserving its scalar format. -/
@[no_expose] def copyAndRelease (b : @& Buffer) : Buffer :=
  if (format? b).isSome then releaseThen b (copy b)
  else copyAndReleaseNative b

/--
Fused multiply-add: `a + c * b` (sizes and dtypes must match; `c` is rounded in that dtype).

This is the classic BLAS-style `axpy` primitive and is useful for optimizers and bias-like updates.
-/
@[never_extract, extern "torchlean_cuda_buffer_axpy"]
opaque axpy (a b : @& Buffer) (c : Float) : Buffer

/--
Perform one Adam-family update through LibTorch tensor operations.

The result is `(parameters, firstMoment, secondMoment)`. Passing `decay = 0` gives Adam; passing
`decay = -(learningRate * weightDecay)` gives AdamW's decoupled parameter decay. The caller
computes the two bias-correction scales from the step counter, exactly as in `Optim.Adam.update`
and `Optim.AdamW.update`.

The caller must validate that `epsilon` remains finite and positive in the parameter dtype.
The native eager optimizer and its checkpoint reader share that validation.

TorchLean's optimizer definitions specify the update. The native bridge evaluates the pointwise
expressions and returns the updated parameter and moment buffers.
-/
@[never_extract, extern "torchlean_cuda_buffer_adam_step"]
opaque adamStep
    (parameters gradient firstMoment secondMoment : @& Buffer)
    (beta1 oneMinusBeta1 beta2 oneMinusBeta2 : Float)
    (firstMomentCorrection secondMomentCorrection epsilon : Float)
    (decay updateScale : Float) :
    Buffer × Buffer × Buffer

/--
Scaled product exponential: `exp((c * x) * y)`.

LibTorch evaluates the two multiplications followed by the exponential. `c` is a host `Float`
rounded in the buffer dtype; `x` and `y` have equal lengths and dtypes.
-/
@[never_extract, extern "torchlean_cuda_buffer_scaled_prod_exp"]
opaque scaledProdExp (x y : @& Buffer) (c : Float) : Buffer

/-- Reductions (return a length-1 buffer). -/
@[never_extract, extern "torchlean_cuda_buffer_reduce_sum"]
opaque reduceSum (b : @& Buffer) : Buffer

/-- Mean of all elements, returned as a one-element buffer. -/
@[never_extract, extern "torchlean_cuda_buffer_reduce_mean"]
opaque reduceMean (b : @& Buffer) : Buffer

end Buffer

end LibTorch
end Autograd
end Runtime
