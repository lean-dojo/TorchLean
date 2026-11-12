/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

/-!
# LibTorch CUDA Runtime Controls

Effectful configuration and readback for TorchLean's LibTorch bridge. The settings are process-wide
requests to the native runtime, so configure them before concurrent runtime work. Neither IEEE
precision nor strict determinism establishes a reference reduction order, correct rounding of every
operation, or bit agreement with FloatLib.

TorchLean owns its tape and selected local VJPs, and these controls never enable LibTorch autograd.
They keep every native tensor in its hardware dtype and select no FloatLib format. `version` reports
the release of the torch runtime the process loaded, which may be later than the release the
native build is compiled against.

The raw ABI uses setting IDs 0–8: TF32 matmul, TF32 cuDNN convolution, strict determinism, cuDNN
benchmarking, flash SDP, efficient SDP, math SDP, cuDNN SDP, and cuDNN enablement. Public callers
use the typed operations below. Setting and memory-fraction reads, like configuration requests,
return native failures through `IO`, and boolean settings are read back to detect a rejected
request. Without LibTorch linked, numeric reads return zero, raw setting reads reject IDs above 8
and configuration requests fail.
-/

@[expose] public section

namespace TorchLean
namespace LibTorch

/-- The release of the loaded torch runtime, such as `"2.13.0"`, or `"unavailable"` when LibTorch
is not linked. -/
@[never_extract, extern "torchlean_libtorch_version"]
opaque version : IO String

/-- Number of CUDA devices visible to the linked runtime, or zero without LibTorch. -/
@[never_extract, extern "torchlean_libtorch_device_count"]
opaque deviceCount : IO UInt32

/-- Device index selected by the bridge. This read alone does not establish availability. -/
@[never_extract, extern "torchlean_libtorch_get_device"]
opaque getDevice : IO UInt32

@[never_extract, extern "torchlean_libtorch_set_device"]
private opaque setDeviceRaw (device : UInt32) : IO Unit

@[never_extract, extern "torchlean_libtorch_get_setting"]
private opaque getSettingRaw (id : UInt32) : IO UInt32

@[never_extract, extern "torchlean_libtorch_set_setting"]
private opaque setSettingRaw (id enabled : UInt32) : IO Unit

@[never_extract, extern "torchlean_libtorch_get_memory_fraction"]
private opaque getMemoryFractionRaw : IO Float

@[never_extract, extern "torchlean_libtorch_set_memory_fraction"]
private opaque setMemoryFractionRaw (fraction : Float) : IO Unit

/-- Wait for the selected CUDA device's work, propagating native errors through `IO`. -/
@[never_extract, extern "torchlean_libtorch_synchronize"]
opaque synchronize : IO Unit

/-- Release unused native allocator cache blocks. Live tensors, saved forward state and library
workspaces such as cuBLAS's keep their memory, so allocated and reserved bytes can stay nonzero. -/
@[never_extract, extern "torchlean_libtorch_empty_cache"]
opaque emptyCache : IO Unit

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

/--
Select the device for later bridge work.

The native setter fails while any buffer wrapper is live, released and empty wrappers included,
even when `device` is already selected. Select the device before creating buffers or after every
wrapper is finalized.
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
  /-- Disable the corresponding TF32 permission, keeping float32 inputs at binary32 precision. -/
  | ieee
  /-- Permit TF32. Whether an operation uses it depends on the selected implementation. -/
  | tf32
  deriving DecidableEq, Repr

@[no_expose] private def setPrecision (id : UInt32) (precision : Float32Precision) : IO Unit :=
  setSetting id (match precision with | .ieee => false | .tf32 => true)

/-- Set the float32 matrix multiplication precision policy and check its readback. -/
@[no_expose] def setMatmulPrecision (precision : Float32Precision) : IO Unit :=
  setPrecision 0 precision

/-- Set the float32 cuDNN convolution precision policy and check its readback. -/
@[no_expose] def setConvPrecision (precision : Float32Precision) : IO Unit :=
  setPrecision 1 precision

/-- Read whether strict deterministic mode is on: deterministic algorithms enabled and
warning-only mode off. -/
@[no_expose] def getDeterministic : IO Bool :=
  getSetting 2

/--
Request strict deterministic execution, or withdraw the request.

Enabling it makes ATen fail on any operation without a deterministic implementation, selects
deterministic cuDNN algorithms and turns cuDNN benchmarking off. The flat sum then follows the tree
of `LibTorch.reduceSumTree`, and other reductions follow no particular tree. Results may differ
between builds or devices. Keep the setting fixed between an attention forward call and its
backward call.
-/
@[no_expose] def setDeterministic (enabled : Bool) : IO Unit :=
  setSetting 2 enabled

/-- Read whether cuDNN convolution algorithm benchmarking is enabled. -/
@[no_expose] def getCuDNNBenchmark : IO Bool :=
  getSetting 3

/-- Enable or disable cuDNN convolution benchmarking. Enabling it fails in deterministic mode. -/
@[no_expose] def setCuDNNBenchmark (enabled : Bool) : IO Unit :=
  setSetting 3 enabled

/-- Native scaled-dot-product-attention implementations whose eligibility can be configured. -/
inductive SDPBackend where
  /-- FlashAttention. -/
  | flash
  /-- Memory-efficient attention. -/
  | efficient
  /-- ATen's math implementation, composed of ordinary tensor operations. -/
  | math
  /-- cuDNN attention. -/
  | cuDNN
  deriving DecidableEq, Repr

private def SDPBackend.settingId : SDPBackend → UInt32
  | .flash => 4
  | .efficient => 5
  | .math => 6
  | .cuDNN => 7

/-- Read whether ATen may use the SDP implementation `backend`. -/
@[no_expose] def getSDPEnabled (backend : SDPBackend) : IO Bool :=
  getSetting backend.settingId

/--
Permit or disable one of ATen's SDP implementations.

Permission does not guarantee support for a shape, dtype, device, or mask, and an ATen SDP call
with every eligible implementation disabled fails. TorchLean's attention composes the native
matrix-product and softmax kernels, which these flags do not affect.
-/
@[no_expose] def setSDPEnabled (backend : SDPBackend) (enabled : Bool) : IO Unit :=
  setSetting backend.settingId enabled

/-- Enable or disable cuDNN in the native context. -/
@[no_expose] def setCuDNNEnabled (enabled : Bool) : IO Unit :=
  setSetting 8 enabled

/--
Read the selected device's allocator memory fraction.

The fraction caps the share of the device's total memory that the caching allocator may hold for
this process, live and cached blocks together. Returns zero when LibTorch is not linked.
-/
@[no_expose] def getMemoryFraction : IO Float := do
  let fraction ← getMemoryFractionRaw
  unless fraction.isFinite && 0.0 ≤ fraction && fraction ≤ 1.0 do
    throw <| IO.userError s!"LibTorch: allocator returned invalid memory fraction {fraction}"
  pure fraction

/--
Set the selected device's allocator memory fraction to a finite value in `(0, 1]`.

The fraction caps later allocations by the caching allocator. Live tensors stay allocated, and the
cap reserves no memory against other processes. The limit is a whole number of bytes, so
`getMemoryFraction` can read back a slightly different value.
-/
@[no_expose] def setMemoryFraction (fraction : Float) : IO Unit := do
  unless fraction.isFinite && 0.0 < fraction && fraction ≤ 1.0 do
    throw <| IO.userError "LibTorch: memory fraction must be finite and lie in (0, 1]"
  setMemoryFractionRaw fraction

end LibTorch
end TorchLean
