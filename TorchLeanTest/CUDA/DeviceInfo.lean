/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.CUDA.LibTorch.Buffer
public import TorchLean.CUDA.LibTorch.DeviceInfo

/-!
# Device Identity

Checks of `TorchLean.LibTorch.deviceInfo`. In every build, the formatting of a fixed record. With
LibTorch linked and a CUDA device visible, every visible index reports itself with a nonempty name
and nonzero capability, multiprocessor count, memory and driver version, two reads of an index
agree, `currentDeviceInfo` reads the selected device, the total memory agrees with
`Buffer.Memory.read`, and the index at the visible count is rejected. When the bridge runs on the
host, the visible devices still report themselves, and `currentDeviceInfo` is rejected. Without
LibTorch, every query is rejected.
-/

@[expose] public section

namespace TorchLean.Test
namespace CUDA
namespace DeviceInfo

open TorchLean.LibTorch

/-- An RTX A4500 as its driver reports it. -/
def sample : LibTorch.DeviceInfo :=
  { index := 0, name := "NVIDIA RTX A4500", capability := 86, multiprocessors := 56,
    clockKHz := 1650000, memoryClockKHz := 8001000, memoryBusBits := 320,
    totalBytes := 20990394368, driverVersion := 13000 }

/-- Fail unless `action` fails with an error containing `fragment`. -/
def expectRejected {α : Type} (what fragment : String) (action : IO α) : IO Unit := do
  let rejected ← try
      let _ ← action
      pure none
    catch e => pure (some e.toString)
  match rejected with
  | none => throw <| IO.userError s!"device identity: {what} was not rejected"
  | some message =>
      unless message.contains fragment do
        throw <| IO.userError s!"device identity: {what} failed with an unexpected error: {message}"

/-- The formatting of `sample`. -/
def runFormat : IO Unit := do
  let expected :=
    "NVIDIA RTX A4500 (device 0, sm_86, 56 SMs, 19.5 GiB, 640 GB/s peak, driver 13.0)"
  unless sample.peakBandwidthBytesPerSecond == 640080000000 do
    throw <| IO.userError
      s!"device identity: peak bandwidth {sample.peakBandwidthBytesPerSecond} B/s"
  unless sample.format == expected do
    throw <| IO.userError s!"device identity: format gave {sample.format}"

/-- The readback of every visible device. -/
def runDevices : IO Unit := do
  let count ← deviceCount
  for i in [0:count.toNat] do
    let index := i.toUInt32
    let d ← deviceInfo index
    unless d.index == index && !d.name.isEmpty && d.capability ≥ 10 && d.multiprocessors > 0 &&
        d.clockKHz > 0 && d.memoryClockKHz > 0 && d.memoryBusBits > 0 && d.totalBytes > 0 &&
        d.driverVersion > 0 do
      throw <| IO.userError s!"device identity: implausible record for device {i}: {repr d}"
    unless (← deviceInfo index) == d do
      throw <| IO.userError s!"device identity: two reads of device {i} disagree"
    IO.println s!"  {d.format}"
  if (← deviceKind) == .host then
    expectRejected "the current device on the host" "the bridge runs on the host"
      currentDeviceInfo
    IO.println "  the current device is rejected on the host"
  else
    let current ← currentDeviceInfo
    unless current == (← deviceInfo (← getDevice)) do
      throw <| IO.userError "device identity: currentDeviceInfo is not the selected device's"
    let memory ← Buffer.Memory.read
    unless memory.deviceTotalBytes.toNat == current.totalBytes do
      throw <| IO.userError
        s!"device identity: {current.totalBytes} total bytes, but Buffer.Memory.read reports \
          {memory.deviceTotalBytes}"
  expectRejected s!"device {count}" "outside the visible count" (deviceInfo count)

/-- Run the checks that the runtime status allows. -/
def run : IO Unit := do
  IO.println "== LibTorch device identity =="
  runFormat
  match (← Buffer.RuntimeStatus.query) with
  | .notLinked =>
      expectRejected "device 0 without LibTorch" "outside the visible count 0" (deviceInfo 0)
      expectRejected "the current device without LibTorch" "outside the visible count 0"
        currentDeviceInfo
      IO.println "  queries are rejected without LibTorch"
  | .nativeAvailable => runDevices
  | .nativeUnavailable =>
      expectRejected "device 0 with no visible device" "outside the visible count 0"
        (deviceInfo 0)
      IO.println "  queries are rejected with no visible device"

end DeviceInfo
end CUDA
end TorchLean.Test
