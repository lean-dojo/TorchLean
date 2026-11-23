/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.CUDA.LibTorch.Controls

/-!
# LibTorch CUDA Device Identity

Readback of which CUDA device a run used, so that runtime numerical evidence can name the device
it was taken on. `Buffer.Memory.read` answers how much memory the selected device has; `deviceInfo`
answers which device it is.

The record comes from the CUDA driver, like the device count and memory the bridge already
reports: `cuDeviceGetName`, `cuDeviceGetAttribute`, `cuDeviceTotalMem` and `cuDriverGetVersion`.
Each query takes an explicit device index and reads every field of that device in one native
call, so the answer does not depend on which device was selected earlier, and one record never
combines fields of two devices. A read creates no CUDA context. An index outside the visible count
fails through `IO`, and so does every query without LibTorch linked.

Two questions this record does not answer. Which GPU architectures the torch runtime's own kernels
were compiled for is a property of that build, not of the device, and the LibTorch Stable ABI does
not expose it. The kernels TorchLean generates are compiled at run time by NVRTC for the compute
capability of the device they run on, so they need no such list.
-/

@[expose] public section

namespace TorchLean
namespace LibTorch

/-- The device name and, in order, its compute capability, multiprocessor count, core clock in
kHz, memory clock in kHz, memory bus width in bits, total memory in bytes and the driver's CUDA
version. -/
@[never_extract, extern "torchlean_libtorch_device_identity"]
private opaque deviceIdentityRaw (index : UInt32) : IO (String × Array UInt64)

/--
Identity of one CUDA device, as the CUDA driver reports it.

`capability` is `10 * major + minor`, so it reads the way an `sm_XY` target is spelled.
-/
structure DeviceInfo where
  /-- The device index this record was read for. -/
  index : UInt32
  /-- The device name, for example `NVIDIA RTX A4500`. -/
  name : String
  /-- `10 * major + minor`: `86` is `sm_86`, `120` is `sm_120`. -/
  capability : Nat
  /-- Streaming multiprocessor count. -/
  multiprocessors : Nat
  /-- Peak core clock in kHz. -/
  clockKHz : Nat
  /-- Peak memory clock in kHz. -/
  memoryClockKHz : Nat
  /-- Global memory bus width in bits. -/
  memoryBusBits : Nat
  /-- Total global memory in bytes. -/
  totalBytes : Nat
  /-- The CUDA version of the installed driver, for example `13000` for 13.0. -/
  driverVersion : Nat
  deriving Repr, DecidableEq

/--
Read the identity of CUDA device `index`.

Fails through `IO` when `index` is outside the visible device count, which is zero without
LibTorch linked.
-/
@[no_expose] def deviceInfo (index : UInt32) : IO DeviceInfo := do
  let count ← deviceCount
  unless index < count do
    throw <| IO.userError s!"LibTorch: CUDA device {index} is outside the visible count {count}"
  let (name, fields) ← deviceIdentityRaw index
  match fields.toList.map UInt64.toNat with
  | [capability, multiprocessors, clockKHz, memoryClockKHz, memoryBusBits, totalBytes,
      driverVersion] =>
    return { index, name, capability, multiprocessors, clockKHz, memoryClockKHz, memoryBusBits,
             totalBytes, driverVersion }
  | _ =>
    throw <| IO.userError s!"LibTorch: device identity returned {fields.size} fields, not 7"

/-- The identity of the CUDA device the bridge selects, `deviceInfo (← getDevice)`. A bridge that
runs on the host selects no CUDA device, so there this fails, also when a device is visible. -/
@[no_expose] def currentDeviceInfo : IO DeviceInfo := do
  if (← deviceKind) == .host then
    throw <| IO.userError "LibTorch: the bridge runs on the host, so no CUDA device is selected"
  deviceInfo (← getDevice)

namespace DeviceInfo

/--
Peak theoretical memory bandwidth in bytes per second: the memory clock, two transfers per clock,
times the bus width in bytes. A measured bandwidth can then be stated as a fraction of this peak.
-/
def peakBandwidthBytesPerSecond (d : DeviceInfo) : Nat :=
  d.memoryClockKHz * 1000 * 2 * d.memoryBusBits / 8

/-- A CUDA version number as it is written: `12080` is `12.8`. -/
def versionString (version : Nat) : String :=
  s!"{version / 1000}.{version % 1000 / 10}"

/-- `numerator / denominator` rounded to the nearest tenth and written with one decimal. -/
def tenths (numerator denominator : Nat) : String :=
  let t := (numerator * 10 + denominator / 2) / denominator
  s!"{t / 10}.{t % 10}"

/-- One line for a benchmark header, for example
`NVIDIA RTX A4500 (device 0, sm_86, 56 SMs, 19.5 GiB, 640 GB/s peak, driver 13.0)`. -/
def format (d : DeviceInfo) : String :=
  let gigabytes := (d.peakBandwidthBytesPerSecond + 500000000) / 1000000000
  s!"{d.name} (device {d.index}, sm_{d.capability}, {d.multiprocessors} SMs, " ++
    s!"{tenths d.totalBytes (1024 * 1024 * 1024)} GiB, {gigabytes} GB/s peak, " ++
    s!"driver {versionString d.driverVersion})"

end DeviceInfo

end LibTorch
end TorchLean
