/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

/-!
# LibTorch Runtime Controls

Effectful configuration and readback for TorchLean's LibTorch bridge. Configure these process-wide
settings before concurrent runtime work. They are runtime requests, not numerical proof evidence:
IEEE precision and strict determinism do not establish a reference reduction order, correct
rounding of every operation, or FloatLib bit agreement.

The bridge selects its device once, at its first call: a visible CUDA device when the SDK was
built with CUDA support, or the host when the SDK is CPU-only. `TORCHLEAN_LIBTORCH_DEVICE=cpu`
selects the host on any SDK and `TORCHLEAN_LIBTORCH_DEVICE=cuda` insists on a CUDA device; a CUDA
SDK without a visible device never falls back to the host silently. `deviceKind` reports the
choice; the CUDA-only controls below say what they do on the host.

TorchLean owns its tape and selected local VJPs. These controls never enable LibTorch autograd.
Native tensors retain their supported hardware dtypes; this API does not select arbitrary FloatLib
formats. `version` reports the linked native build rather than a version assumed by Lean.

The raw ABI uses setting IDs 0–8: TF32 matmul, TF32 cuDNN convolution, strict determinism, cuDNN
benchmarking, flash SDP, efficient SDP, math SDP, cuDNN SDP, and cuDNN enablement. Public callers
use the typed operations below. Setting and memory-fraction reads, like configuration requests,
return native failures through `IO`. Boolean settings are read back to detect a rejected request.
Without LibTorch linked, reads return zero and configuration requests fail; raw setting reads
also reject unknown IDs.
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace LibTorch

@[never_extract, extern "torchlean_libtorch_version"]
private opaque versionRaw (token : UInt32) : String

@[never_extract, extern "torchlean_libtorch_device_count"]
private opaque deviceCountRaw (token : UInt32) : UInt32

@[never_extract, extern "torchlean_libtorch_get_device"]
private opaque getDeviceRaw (token : UInt32) : UInt32

@[never_extract, extern "torchlean_libtorch_device_kind"]
private opaque deviceKindRaw (token : UInt32) : UInt32

@[never_extract, extern "torchlean_libtorch_set_device"]
private opaque setDeviceRaw (device : UInt32) : IO Unit

@[never_extract, extern "torchlean_libtorch_get_setting"]
private opaque getSettingRaw (id : UInt32) : IO UInt32

@[never_extract, extern "torchlean_libtorch_set_setting"]
private opaque setSettingRaw (id enabled : UInt32) : IO Unit

@[never_extract, extern "torchlean_libtorch_get_memory_fraction"]
private opaque getMemoryFractionRaw (token : UInt32) : IO Float

@[never_extract, extern "torchlean_libtorch_set_memory_fraction"]
private opaque setMemoryFractionRaw (fraction : Float) : IO Unit

@[never_extract, extern "torchlean_libtorch_synchronize"]
private opaque synchronizeRaw (token : UInt32) : IO Unit

@[never_extract, extern "torchlean_libtorch_empty_cache"]
private opaque emptyCacheRaw (token : UInt32) : IO Unit

@[no_expose] private def getSetting (id : UInt32) : IO Bool := do
  let observed ← getSettingRaw id
  match observed with
  | 0 => pure false
  | 1 => pure true
  | _ => throw <| IO.userError s!"LibTorch: setting {id} returned invalid boolean {observed}"

@[no_expose] private def setSetting (id : UInt32) (enabled : Bool) : IO Unit := do
  setSettingRaw id (if enabled then 1 else 0)
  let observed ← getSetting id
  unless observed == enabled do
    throw <| IO.userError
      s!"LibTorch: setting {id} is {observed} after requesting {enabled}"

/-- Native build identification; `"unavailable"` when LibTorch is not linked. -/
@[no_expose] def version : IO String :=
  IO.lazyPure fun _ => versionRaw 0

/-- Number of CUDA devices visible to the linked runtime; zero without LibTorch and zero on a
CPU-only SDK. A visible device is not necessarily the selected one: see `deviceKind`. -/
@[no_expose] def deviceCount : IO UInt32 :=
  IO.lazyPure fun _ => deviceCountRaw 0

/-- The device the bridge runs on, as selected at its first call. -/
inductive DeviceKind where
  /-- A CUDA device. `Buffer.runtimeStatus` says whether one is usable. -/
  | cuda
  /-- The host: a CPU-only SDK, or `TORCHLEAN_LIBTORCH_DEVICE=cpu`. -/
  | host
  deriving DecidableEq, Repr

/-- Which device the bridge selected. Meaningful only when LibTorch is linked: a build without it
reads `.cuda` and `Buffer.runtimeStatus` reads `.notLinked`. -/
@[no_expose] def deviceKind : IO DeviceKind :=
  IO.lazyPure fun _ => if deviceKindRaw 0 == 0 then .host else .cuda

/-- CUDA device index selected by the bridge. This read alone does not establish availability. -/
@[no_expose] def getDevice : IO UInt32 :=
  IO.lazyPure fun _ => getDeviceRaw 0

/--
Select the CUDA device for subsequent bridge work; the host has no index and rejects the call.

The native setter rejects changes while any buffer wrappers remain live, including released or
empty wrappers. Retire those owners before switching devices; the call does not migrate tensors.
-/
@[no_expose] def setDevice (device : UInt32) : IO Unit := do
  let count ← deviceCount
  unless device < count do
    throw <| IO.userError s!"LibTorch: CUDA device {device} is outside the visible count {count}"
  setDeviceRaw device
  let observed ← getDevice
  unless observed == device do
    throw <| IO.userError s!"LibTorch: device is {observed} after requesting {device}"

/-- Float32 input precision for matrix multiplication or cuDNN convolution. -/
inductive Float32Precision where
  /-- Disable the corresponding TF32 permission. This is not a correct-rounding theorem. -/
  | ieee
  /-- Permit TF32; the runtime still selects an eligible implementation. -/
  | tf32
  deriving DecidableEq, Repr

@[no_expose] private def getPrecision (id : UInt32) : IO Float32Precision := do
  let enabled ← getSetting id
  pure <| if enabled then .tf32 else .ieee

@[no_expose] private def setPrecision (id : UInt32) (precision : Float32Precision) : IO Unit :=
  setSetting id (match precision with | .ieee => false | .tf32 => true)

/-- Read the float32 matrix multiplication precision policy. -/
@[no_expose] def getMatmulPrecision : IO Float32Precision :=
  getPrecision 0

/-- Set the float32 matrix multiplication precision policy and check its readback. -/
@[no_expose] def setMatmulPrecision (precision : Float32Precision) : IO Unit :=
  setPrecision 0 precision

/-- Read the float32 cuDNN convolution precision policy. -/
@[no_expose] def getConvPrecision : IO Float32Precision :=
  getPrecision 1

/-- Set the float32 cuDNN convolution precision policy and check its readback. -/
@[no_expose] def setConvPrecision (precision : Float32Precision) : IO Unit :=
  setPrecision 1 precision

/-- Read whether strict deterministic algorithms are enabled, with warning-only mode disabled. -/
@[no_expose] def getDeterministic : IO Bool :=
  getSetting 2

/--
Request strict deterministic execution, or disable that request.

Enabling this requires native errors for unsupported deterministic operations, not warnings,
and disables cuDNN benchmarking. It does not fix a particular reduction tree or promise agreement
across builds or devices. Keep it stable between an attention forward call and its backward call.
-/
@[no_expose] def setDeterministic (enabled : Bool) : IO Unit :=
  setSetting 2 enabled

/-- Read whether cuDNN convolution algorithm benchmarking is enabled. -/
@[no_expose] def getCuDNNBenchmark : IO Bool :=
  getSetting 3

/-- Enable or disable cuDNN convolution benchmarking; enabling is rejected in deterministic mode. -/
@[no_expose] def setCuDNNBenchmark (enabled : Bool) : IO Unit :=
  setSetting 3 enabled

/-- Native scaled-dot-product-attention implementations whose eligibility can be configured. -/
inductive SDPBackend where
  | flash
  | efficient
  | math
  | cuDNN
  deriving DecidableEq, Repr

private def SDPBackend.settingId : SDPBackend → UInt32
  | .flash => 4
  | .efficient => 5
  | .math => 6
  | .cuDNN => 7

/-- Read whether the selected native SDP implementation is permitted. -/
@[no_expose] def getSDPEnabled (backend : SDPBackend) : IO Bool :=
  getSetting backend.settingId

/--
Permit or disable an SDP implementation.

Permission does not guarantee support for a shape, dtype, device, or mask. These choices apply to
the native SDP route. Disabling all
eligible choices causes a native execution error. Configure them before recording attention work.
-/
@[no_expose] def setSDPEnabled (backend : SDPBackend) (enabled : Bool) : IO Unit :=
  setSetting backend.settingId enabled

/-- Read whether cuDNN is enabled in the native context. -/
@[no_expose] def getCuDNNEnabled : IO Bool :=
  getSetting 8

/-- Enable or disable cuDNN in the native context. -/
@[no_expose] def setCuDNNEnabled (enabled : Bool) : IO Unit :=
  setSetting 8 enabled

/--
Read the selected device's allocator memory fraction.

This is a native allocator limit, not the fraction of memory currently free or a cache-only budget.
Returns zero when LibTorch is not linked, and zero on the host, which has no such limit.
-/
@[no_expose] def getMemoryFraction : IO Float := do
  let fraction ← getMemoryFractionRaw 0
  unless fraction.isFinite && 0.0 ≤ fraction && fraction ≤ 1.0 do
    throw <| IO.userError s!"LibTorch: allocator returned invalid memory fraction {fraction}"
  pure fraction

/--
Set the selected device's allocator memory fraction to a finite value in `(0, 1]`.

This configures native allocation policy; it does not release live tensors or reserve memory against
other processes. Read back with `getMemoryFraction`; byte granularity may affect the observed limit.
The host has no such limit and rejects the request.
-/
@[no_expose] def setMemoryFraction (fraction : Float) : IO Unit := do
  unless fraction.isFinite && 0.0 < fraction && fraction ≤ 1.0 do
    throw <| IO.userError "LibTorch: memory fraction must be finite and lie in (0, 1]"
  setMemoryFractionRaw fraction

/-- Wait for the selected CUDA device's work, propagating native errors through `IO`. Host work
is synchronous, so this returns at once there. -/
@[no_expose] def synchronize : IO Unit :=
  synchronizeRaw 0

/--
Release unused native allocator cache blocks.

Live tensors, saved forward state, and library workspaces remain allocated. A cuBLAS workspace can
outlive every TorchLean buffer, so both allocated and reserved bytes may remain after this call.
Use the buffer ownership counters to distinguish those allocations from retained TorchLean owners.
The host has no native cache, so this returns at once there.
-/
@[no_expose] def emptyCache : IO Unit :=
  emptyCacheRaw 0

end LibTorch
end Autograd
end Runtime
