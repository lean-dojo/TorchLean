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

/-- Raw status word from the C layer; `runtimeStatus` decodes it. -/
@[never_extract, extern "torchlean_cuda_runtime_status"]
private opaque runtimeStatusRaw (token : UInt32) : UInt32

/-- Query whether the linked CUDA symbols are native or the CPU parity stubs. -/
@[no_expose] def runtimeStatus (token : UInt32 := 0) : RuntimeStatus :=
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
        "CUDA was requested, but this executable is linked to TorchLean's CPU parity stubs; \
          rebuild and run with `-K cuda=true`"
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

/-- Read the deterministic-reductions flag. The argument only forces a call. -/
@[never_extract, extern "torchlean_cuda_get_deterministic_reductions_u"]
private opaque getDeterministicReductionsRaw (u : UInt32) : UInt32

/-- Set the flag and return what the runtime observed, so the call has a used result. -/
@[never_extract, extern "torchlean_cuda_set_deterministic_reductions_checked"]
private opaque setDeterministicReductionsCheckedRaw (on : UInt32) : UInt32

/--
Enable/disable deterministic reductions mode as an `IO` action.

The native setter runs at this point in `IO` and reports the resulting setting. Checking that value
both verifies the request and keeps the native effect attached to the action. Throws if the runtime
reports a different setting.
-/
@[no_expose] def setDeterministicReductions (on : Bool) : IO Unit := do
  let observed ← IO.lazyPure fun _ =>
    setDeterministicReductionsCheckedRaw (if on then 1 else 0) != 0
  unless observed == on do
    throw <| IO.userError
      s!"cuda: deterministic reductions flag is {observed} after requesting {on}"

/--
Query whether deterministic reductions mode is enabled.

The read runs inside `IO`, so each call observes the current setting. A pure definition could retain
the value read during module initialization even after `setDeterministicReductions` changes it.
-/
@[no_expose] def getDeterministicReductions : IO Bool :=
  IO.lazyPure fun _ => getDeterministicReductionsRaw 0 != 0

/-! ### Allocator Telemetry -/

/-- Bytes currently handed out by the CUDA allocator. -/
@[never_extract, extern "torchlean_cuda_allocator_live_bytes"]
private opaque allocatorLiveBytesRaw (u : UInt32) : UInt64

/-- High-water mark of `allocatorLiveBytesRaw` since process start. -/
@[never_extract, extern "torchlean_cuda_allocator_peak_bytes"]
private opaque allocatorPeakBytesRaw (u : UInt32) : UInt64

/-- Nonempty buffer payloads handed out, including blocks reused from the cache. -/
@[never_extract, extern "torchlean_cuda_allocator_alloc_count"]
private opaque allocatorAllocCountRaw (u : UInt32) : UInt64

/-- Buffer payloads returned by their owners, including blocks retained for reuse. -/
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

/-- Reclaimable bytes shared by the tensor-buffer and kernel-workspace caches. -/
@[never_extract, extern "torchlean_cuda_allocator_cache_bytes"]
private opaque allocatorCacheBytesRaw (u : UInt32) : UInt64

/-- Configured ceiling on that cache, zero when uncapped. -/
@[never_extract, extern "torchlean_cuda_allocator_cache_cap_bytes"]
private opaque allocatorCacheCapBytesRaw (u : UInt32) : UInt64

/--
Snapshot of the CUDA buffer allocator.

`liveBytes`/`peakBytes` count device or stub payloads allocated by this runtime layer. The wrapper
counters track Lean external buffer objects, including empty wrappers and wrappers whose payloads
were explicitly released. In a steady workload, `wrapperAllocCount - wrapperFinalizeCount` should
remain bounded. `deviceFreeBytes` and `deviceTotalBytes` come from `cudaMemGetInfo` in the CUDA
build and are `0` in the CPU stub. Together these fields distinguish payload leaks, wrapper-lifetime
leaks, and broader CUDA memory pressure or fragmentation.

`allocCount` and `freeCount` count buffer payload lifetimes, including reuse. They do not count
calls to `cudaMalloc` and `cudaFree`. Live kernel workspace is outside these payload counters.

`cacheBytes` counts unused tensor buffers and kernel workspaces retained for reuse. Their combined
budget is `cacheCapBytes`, which defaults to 1 GiB. `TORCHLEAN_CUDA_CACHE_CAP_BYTES` can override
that budget; an explicit `0` selects unbounded caching, and invalid values use the default. Both
cache fields are `0` in the CPU stub, which keeps no cache.

The fields are read separately. A snapshot can include concurrent allocator activity and should
not be treated as an atomic account of every allocation in the process.
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

Each read is sequenced at this point in `IO`, including repeated calls in a loop. The native reads
stay inside the action rather than constructing a record that Lean could retain from an earlier
call. Applications do not need a step counter or another changing argument to obtain fresh values.
-/
@[no_expose] def allocatorStats : IO AllocatorStats :=
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
      cacheBytes := allocatorCacheBytesRaw 0
      cacheCapBytes := allocatorCacheCapBytesRaw 0 }

/-! ### Device identity

A measurement is only comparable if it names the machine it was taken on. The allocator snapshot
answers *how much* device memory there is; these answer *which device* — the fact a benchmark row
needs in order to be filed beside another one. -/

@[never_extract, extern "torchlean_cuda_device_name"]
opaque deviceNameRaw (u : UInt32) : String

@[never_extract, extern "torchlean_cuda_compiled_arch_list"]
opaque compiledArchListRaw (u : UInt32) : String

@[never_extract, extern "torchlean_cuda_device_index"]
opaque deviceIndexRaw (u : UInt32) : UInt32

@[never_extract, extern "torchlean_cuda_device_capability"]
opaque deviceCapabilityRaw (u : UInt32) : UInt32

@[never_extract, extern "torchlean_cuda_device_sm_count"]
opaque deviceSmCountRaw (u : UInt32) : UInt32

@[never_extract, extern "torchlean_cuda_device_clock_khz"]
opaque deviceClockKhzRaw (u : UInt32) : UInt32

@[never_extract, extern "torchlean_cuda_device_mem_clock_khz"]
opaque deviceMemClockKhzRaw (u : UInt32) : UInt32

@[never_extract, extern "torchlean_cuda_device_mem_bus_width"]
opaque deviceMemBusWidthRaw (u : UInt32) : UInt32

@[never_extract, extern "torchlean_cuda_driver_version"]
opaque driverVersionRaw (u : UInt32) : UInt32

@[never_extract, extern "torchlean_cuda_runtime_version"]
opaque runtimeVersionRaw (u : UInt32) : UInt32

/--
Identity of the CUDA device this process is using.

Every field is what `cudaGetDeviceProperties` reports for the current device, except the two
version fields, which come from `cudaDriverGetVersion` / `cudaRuntimeGetVersion`. `capability` is
`major * 10 + minor`, so it reads the way an `-arch=sm_XY` flag is spelled — which is the
comparison that matters, because a binary compiled for a lower architecture still runs here
through PTX JIT and would otherwise report a device that says nothing about the code that ran on
it.

In the CPU stub, and in a CUDA build on a host with no reachable device, `name` is empty and every
scalar is `0`. `deviceInfo` returns `none` in that case rather than a record of zeroes.
-/
structure DeviceInfo where
  name : String
  index : UInt32
  /-- `major * 10 + minor` — `86` is `sm_86`, `120` is `sm_120`. -/
  capability : UInt32
  smCount : UInt32
  clockKhz : UInt32
  memClockKhz : UInt32
  memBusWidthBits : UInt32
  totalBytes : UInt64
  /-- `cudaDriverGetVersion`, e.g. `12060` for 12.6. -/
  driverVersion : UInt32
  /-- `cudaRuntimeGetVersion`, e.g. `12060` for 12.6. -/
  runtimeVersion : UInt32
deriving Repr

/-- Read the current device's identity, or `none` when there is no device to name (the CPU stub, or
a CUDA build on a host where no card answers). -/
def deviceInfo : IO (Option DeviceInfo) := do
  let name := deviceNameRaw 0
  if name.isEmpty then return none
  return some
    { name
      index := deviceIndexRaw 0
      capability := deviceCapabilityRaw 0
      smCount := deviceSmCountRaw 0
      clockKhz := deviceClockKhzRaw 0
      memClockKhz := deviceMemClockKhzRaw 0
      memBusWidthBits := deviceMemBusWidthRaw 0
      totalBytes := allocatorDeviceTotalBytesRaw 0
      driverVersion := driverVersionRaw 0
      runtimeVersion := runtimeVersionRaw 0 }

/-- The `sm_XY` targets this binary was compiled for, as `__CUDA_ARCH_LIST__` spells them
(`"860"`, `"750,860"`). Empty in the CPU stub, and empty in a CUDA build whose compiler did not
define the macro. -/
def compiledArchList : String := compiledArchListRaw 0

/-- Whether this binary holds native code for the device it is running on.

`false` here does not mean broken — PTX JIT will produce code for the device from the embedded
intermediate representation, and it will run. It means the timings are of JIT-compiled code
generated for an architecture the compiler was never told about, which is a different measurement
from the one a reader will assume, and the only place that difference is visible is here.

An empty compiled list is not evidence either way (no device build, or no macro), so it answers
`true` rather than warning about a build it cannot see. -/
def DeviceInfo.runsNatively (d : DeviceInfo) : Bool :=
  compiledArchList.isEmpty ||
    (compiledArchList.splitOn ",").any (fun a => a.trimAscii.toString == toString (d.capability.toNat * 10))

/-- A CUDA version integer (`12060`) as it is written (`12.6`). -/
def versionString (v : UInt32) : String :=
  let n := v.toNat
  s!"{n / 1000}.{(n % 1000) / 10}"

/-- Peak theoretical memory bandwidth in GB/s, from the memory clock and bus width — the roofline
denominator, so that a row reporting achieved GB/s can be read as a fraction without the reader
having to look the card up. Double data rate is assumed (every device this runs on). -/
def DeviceInfo.peakBandwidthGBs (d : DeviceInfo) : Float :=
  (Float.ofNat d.memClockKhz.toNat) * 1.0e3 * 2.0 *
    (Float.ofNat d.memBusWidthBits.toNat / 8.0) / 1.0e9

/-- One-line device identity for a benchmark header. -/
def DeviceInfo.format (d : DeviceInfo) : String :=
  s!"{d.name} (device {d.index}, sm_{d.capability}, {d.smCount} SMs, " ++
  s!"{(Float.ofNat d.totalBytes.toNat) / (1024.0 * 1024.0 * 1024.0)} GiB, " ++
  s!"~{d.peakBandwidthGBs} GB/s peak, " ++
  s!"driver {versionString d.driverVersion}, runtime {versionString d.runtimeVersion})" ++
  (if compiledArchList.isEmpty then ""
   else if d.runsNatively then s!" [compiled for {compiledArchList}]"
   else s!" [WARNING: compiled for {compiledArchList}, NOT sm_{d.capability} — running via PTX JIT]")

/-- Format a byte count as MiB for allocator progress messages. -/
@[no_expose] private def mibString (bytes : UInt64) : String :=
  let mib := (Float.ofNat bytes.toNat) / (1024.0 * 1024.0)
  toString mib ++ " MiB"

/--
One-line allocator report for progress logs.

A zero cache cap is printed as `0`: it means unbounded caching in a CUDA build and no cache in the
CPU stub.
-/
@[no_expose] def AllocatorStats.format (s : AllocatorStats) : String :=
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
  " cache_cap=" ++ (if s.cacheCapBytes == 0 then "0" else mibString s.cacheCapBytes)

/--
Create a device buffer by copying from a host `FloatArray` (casts each element to float32).

This primitive has a pure Lean type, but the native implementation allocates a fresh device buffer.
Runtime code should use `ofFloatArrayIO`: each call allocates a distinct buffer, and ordinary device
exhaustion is returned as an IO error that the caller can handle.
-/
@[never_extract, extern "torchlean_cuda_buffer_of_float_array"]
opaque ofFloatArray (a : @& FloatArray) : Buffer

/--
Copy a host `FloatArray` into a fresh device buffer, rounding each element to float32.

The upload runs at this point in the IO sequence. Repeated calls with the same host array allocate
distinct buffers, so releasing one does not invalidate another. If device allocation fails, the
allocator releases unused cached blocks and retries; a second OOM throws
`IO.Error.resourceExhausted`. The host array and existing device buffers remain owned by the caller.
-/
@[never_extract, extern "torchlean_cuda_buffer_of_float_array_io"]
opaque ofFloatArrayIO (a : @& FloatArray) : IO Buffer

/-- Copy a buffer back to a host `FloatArray` (casts float32 elements to `Float`). -/
@[never_extract, extern "torchlean_cuda_buffer_to_float_array"]
opaque toFloatArray (b : @& Buffer) : FloatArray

/-- Download a buffer to the host, widening each element to Lean `Float`. -/
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

/--
Upload a raw float32 byte payload to a fresh device buffer.

Checkpoint values retain their float32 representation. Device allocation uses the same cache
reclamation and retry as `zerosIO`; ordinary device exhaustion throws `IO.Error.resourceExhausted`
before a buffer is returned. The borrowed byte payload remains available to the caller.
-/
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

/-- Native collection mode: zero keeps reusable CUDA blocks; one drains all unused caches. -/
@[never_extract, extern "torchlean_runtime_collect_allocator"]
private opaque collectAllocatorRaw (force : UInt32) : UInt32

/--
Collect unused host allocator pages while retaining CUDA buffers for reuse.

Training and evaluation call this after retiring a completed tape and its temporary gradients.
The native cache budget bounds retained device memory, so ordinary callers do not need to flush
the cache between updates. Live parameters and optimizer state remain owned by their sessions.
-/
@[no_expose] def collectGarbage : IO Unit := do
  let collected ← IO.lazyPure fun _ => collectAllocatorRaw 0
  if collected == 0 then
    throw <| IO.userError "CUDA allocator collection failed"

/--
Return all unused tensor buffers and kernel workspaces to the CUDA driver.

This explicit operation waits for cached blocks to become safe to free and also asks the host
allocator to release unused pages. It does not release live tensors, parameter mirrors, or optimizer
state. Normal training retains a bounded cache automatically; use this when returning unused memory
to another workload matters more than keeping it for the next operation.

Every invocation performs a fresh collection, including calls after an earlier flush. The raw call
stays inside the `IO` action so repeated requests cannot share a previously computed result.
-/
@[no_expose] def emptyCache : IO Unit := do
  let collected ← IO.lazyPure fun _ => collectAllocatorRaw 1
  if collected == 0 then
    throw <| IO.userError "CUDA cache collection failed"

/-- Allocate a length-`n` buffer filled with zeros. -/
@[never_extract, extern "torchlean_cuda_buffer_zeros"]
opaque zeros (n : UInt32) : Buffer

/--
Allocate a fresh zero-filled buffer inside `IO` code.

The allocation runs at this point in the IO sequence. If device memory is exhausted, the native
allocator first releases unused cached blocks and retries. A second device OOM throws
`IO.Error.resourceExhausted`, so a caller can release its own temporary buffers and try a smaller
allocation. Existing live buffers remain owned by their callers, and a failed allocation adds no
live buffer to the counters.

The allocating IO constructors share this recovery behavior. Pure allocation and kernel primitives
retain their native failure policy. Host allocation failure and errors encountered while flushing
an invalid CUDA context are outside this recovery path.
-/
@[never_extract, extern "torchlean_cuda_buffer_zeros_io"]
opaque zerosIO (n : UInt32) : IO Buffer

/-- Allocate a length-`n` buffer filled with `v` (host `Float`, cast to float32). -/
@[never_extract, extern "torchlean_cuda_buffer_full"]
opaque full (n : UInt32) (v : Float) : Buffer

/--
Allocate a fresh length-`n` buffer filled with `v`, rounded to float32.

Each call owns a distinct buffer. Device allocation follows the cache reclamation and retry used
by `zerosIO`, and ordinary device exhaustion throws `IO.Error.resourceExhausted`.
-/
@[never_extract, extern "torchlean_cuda_buffer_full_io"]
opaque fullIO (n : UInt32) (v : Float) : IO Buffer

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

/--
Generate the deterministic values of `randUniform` in a fresh buffer.

The key determines the values; repeated calls still allocate distinct buffers. Device allocation
uses the recovery behavior of `zerosIO`, including `IO.Error.resourceExhausted` on ordinary OOM.
-/
@[never_extract, extern "torchlean_cuda_buffer_rand_uniform_io"]
opaque randUniformIO (n : UInt32) (key : UInt64) : IO Buffer

/-- Deterministic normal generator using Box-Muller on the device. -/
@[never_extract, extern "torchlean_cuda_buffer_rand_normal"]
opaque randNormal (n : UInt32) (mean std : Float) (key : UInt64) : Buffer

/-- Deterministic `{0,1}` mask generator: returns a length-`n` buffer keyed by `key`. -/
@[never_extract, extern "torchlean_cuda_buffer_bernoulli_mask"]
opaque bernoulliMask (n : UInt32) (keepProb : Float) (key : UInt64) : Buffer

/--
Generate the deterministic mask of `bernoulliMask` in a fresh buffer.

The probability and key retain the pure primitive's meaning. Each call allocates independently,
using the cache reclamation, retry, and ordinary device-OOM error of `zerosIO`.
-/
@[never_extract, extern "torchlean_cuda_buffer_bernoulli_mask_io"]
opaque bernoulliMaskIO (n : UInt32) (keepProb : Float) (key : UInt64) : IO Buffer

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
Elementwise sine of angles in radians, returning a new float32 buffer.

The native implementation applies `sinf` to each entry. The input is borrowed, so the tape can
retain it for the cosine factor in the backward pass.
-/
@[never_extract, extern "torchlean_cuda_buffer_sin"]
opaque sin (b : @& Buffer) : Buffer

/--
Elementwise cosine of angles in radians, returning a new float32 buffer and borrowing its input.
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

/-- Reductions (return a length-1 buffer). -/
@[never_extract, extern "torchlean_cuda_buffer_reduce_sum"]
opaque reduceSum (b : @& Buffer) : Buffer

/-- Mean of all elements, returned as a one-element buffer. -/
@[never_extract, extern "torchlean_cuda_buffer_reduce_mean"]
opaque reduceMean (b : @& Buffer) : Buffer

end Buffer

end Cuda
end Autograd
end Runtime
