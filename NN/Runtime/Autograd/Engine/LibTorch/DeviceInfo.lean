/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Engine.LibTorch.Controls

/-!
# LibTorch CUDA Device Identity

Readback of the CUDA device the bridge runs on. A measurement that cannot name the device it was
taken on produces a number nobody can file beside another one. `Buffer.allocatorStats` answers how
much device memory there is; this module answers which device.

Every query takes an explicit device index and reads that device's properties through the SDK's
own per-device cache, so the answer never depends on which device was selected earlier, and a
multi-GPU host never combines fields from two cards in one record. Without LibTorch linked, every
query fails through `IO` like the other runtime controls.

The SDK's packaged GPU architectures are not reported: ATen's C++ API does not expose the list the
Python package reads back, and TorchLean compiles no device code of its own.
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace LibTorch

@[never_extract, extern "torchlean_libtorch_device_name"]
private opaque deviceNameRaw (index : UInt32) : IO String

@[never_extract, extern "torchlean_libtorch_device_capability"]
private opaque deviceCapabilityRaw (index : UInt32) : IO UInt32

@[never_extract, extern "torchlean_libtorch_device_sm_count"]
private opaque deviceSmCountRaw (index : UInt32) : IO UInt32

@[never_extract, extern "torchlean_libtorch_device_clock_khz"]
private opaque deviceClockKhzRaw (index : UInt32) : IO UInt32

@[never_extract, extern "torchlean_libtorch_device_mem_clock_khz"]
private opaque deviceMemClockKhzRaw (index : UInt32) : IO UInt32

@[never_extract, extern "torchlean_libtorch_device_mem_bus_width"]
private opaque deviceMemBusWidthRaw (index : UInt32) : IO UInt32

@[never_extract, extern "torchlean_libtorch_device_total_bytes"]
private opaque deviceTotalBytesRaw (index : UInt32) : IO UInt64

@[never_extract, extern "torchlean_libtorch_driver_version"]
private opaque driverVersionRaw (token : UInt32) : IO UInt32

@[never_extract, extern "torchlean_libtorch_runtime_version"]
private opaque runtimeVersionRaw (token : UInt32) : IO UInt32

/--
Identity of one CUDA device, as the linked SDK reports it.

The device fields come from the SDK's cached `cudaDeviceProp` for that index and from
`cudaDeviceGetAttribute`; the two version fields from `cudaDriverGetVersion` and
`cudaRuntimeGetVersion`. `capability` is `major * 10 + minor`, so it reads the way an `sm_XY`
target is spelled.
-/
structure DeviceInfo where
  /-- The index this record was read for. -/
  index : UInt32
  /-- The device name, for example `NVIDIA RTX A4500`. -/
  name : String
  /-- `major * 10 + minor`: `86` is `sm_86`, `120` is `sm_120`. -/
  capability : UInt32
  /-- Streaming multiprocessor count. -/
  smCount : UInt32
  /-- Core clock in kHz. -/
  clockKhz : UInt32
  /-- Memory clock in kHz. -/
  memClockKhz : UInt32
  /-- Global memory bus width in bits. -/
  memBusWidthBits : UInt32
  /-- Total global memory in bytes. -/
  totalBytes : UInt64
  /-- `cudaDriverGetVersion`, for example `13000` for 13.0. -/
  driverVersion : UInt32
  /-- `cudaRuntimeGetVersion` of the SDK's runtime, for example `12080` for 12.8. -/
  runtimeVersion : UInt32
  deriving Repr

/--
Read the properties of device `index`.

Fails through `IO` when the index is outside the visible device count or LibTorch is not linked.
Reading by explicit index is what keeps the answer independent of the selected device.
-/
@[no_expose] def deviceInfo (index : UInt32) : IO DeviceInfo := do
  let name ← deviceNameRaw index
  let capability ← deviceCapabilityRaw index
  let smCount ← deviceSmCountRaw index
  let clockKhz ← deviceClockKhzRaw index
  let memClockKhz ← deviceMemClockKhzRaw index
  let memBusWidthBits ← deviceMemBusWidthRaw index
  let totalBytes ← deviceTotalBytesRaw index
  let driverVersion ← driverVersionRaw 0
  let runtimeVersion ← runtimeVersionRaw 0
  pure { index, name, capability, smCount, clockKhz, memClockKhz, memBusWidthBits, totalBytes,
         driverVersion, runtimeVersion }

/-- The device the bridge currently selects, as `getDevice` reports it. -/
@[no_expose] def currentDeviceInfo : IO DeviceInfo := do
  deviceInfo (← getDevice)

/-- A CUDA version integer (`12080`) as it is written (`12.8`). -/
def versionString (v : UInt32) : String :=
  let n := v.toNat
  s!"{n / 1000}.{(n % 1000) / 10}"

/--
Peak theoretical memory bandwidth in GB/s from the memory clock and bus width, so a row reporting
achieved GB/s can be read as a fraction of the roofline without looking the card up. Double data
rate is assumed.
-/
def DeviceInfo.peakBandwidthGBs (d : DeviceInfo) : Float :=
  (Float.ofNat d.memClockKhz.toNat) * 1.0e3 * 2.0 *
    (Float.ofNat d.memBusWidthBits.toNat / 8.0) / 1.0e9

/-- One-line device identity for a benchmark header. -/
def DeviceInfo.format (d : DeviceInfo) : String :=
  s!"{d.name} (device {d.index}, sm_{d.capability}, {d.smCount} SMs, " ++
  s!"{(Float.ofNat d.totalBytes.toNat) / (1024.0 * 1024.0 * 1024.0)} GiB, " ++
  s!"~{d.peakBandwidthGBs} GB/s peak, " ++
  s!"driver {versionString d.driverVersion}, runtime {versionString d.runtimeVersion})"

end LibTorch
end Autograd
end Runtime
