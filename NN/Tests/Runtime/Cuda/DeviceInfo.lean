/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Engine.LibTorch.Buffer
public import NN.Runtime.Autograd.Engine.LibTorch.DeviceInfo

/-!
# Device identity readback

Checks that every visible device reports itself under its own index, that the current-device
reading agrees with `getDevice`, that an index at the visible count is rejected, and, in a fresh
process, that switching devices changes what `currentDeviceInfo` reports. The switching probe runs
in a child process because `setDevice` refuses while any buffer wrapper is live, and the suite
allocates buffers before the CUDA coverage tests run. Without LibTorch, the queries must fail like
every other runtime control.
-/

@[expose] public section

namespace Tests
namespace Cuda
namespace DeviceInfo

open Runtime.Autograd.LibTorch

def expect (label : String) (ok : Bool) : IO Unit := do
  unless ok do
    throw <| IO.userError s!"device identity check failed: {label}"

/-- Every visible device reports itself under its own index; the current device agrees with
`getDevice`; an index at the visible count is rejected. -/
def runReadback : IO Unit := do
  let count ← deviceCount
  expect "a native-available build sees at least one device" (count > 0)
  for i in [0:count.toNat] do
    let info ← deviceInfo i.toUInt32
    expect s!"device {i} reports its own index" (info.index == i.toUInt32)
    expect s!"device {i} has a name" (!info.name.isEmpty)
    expect s!"device {i} reports a compute capability" (info.capability > 0)
    expect s!"device {i} reports an SM count" (info.smCount > 0)
    expect s!"device {i} reports its memory" (info.totalBytes > 0)
    IO.println s!"  {info.format}"
  let current ← currentDeviceInfo
  expect "current device agrees with getDevice" (current.index == (← getDevice))
  let rejected ←
    try
      discard (deviceInfo count)
      pure false
    catch _ => pure true
  expect "an index at the visible count is rejected" rejected

/--
Switch through every visible device and read the current device back after each switch.

Runs under `TORCHLEAN_LIBTORCH_DEVICE_PROBE=switch` in a fresh process, before any buffer exists.
-/
def runSwitchProbe : IO Unit := do
  Buffer.requireNativeRuntime
  let count ← deviceCount
  let original ← getDevice
  for i in [0:count.toNat] do
    setDevice i.toUInt32
    let info ← currentDeviceInfo
    expect s!"after setDevice {i}, currentDeviceInfo reports {i}" (info.index == i.toUInt32)
    let direct ← deviceInfo i.toUInt32
    expect s!"after setDevice {i}, the name agrees with the direct read" (info.name == direct.name)
  setDevice original
  IO.println s!"  device switching: {count} device(s) report under their own index"

/-- Without LibTorch, identity queries fail like every other control. -/
def runUnavailable : IO Unit := do
  let failed ←
    try
      discard (deviceInfo 0)
      pure false
    catch _ => pure true
  expect "deviceInfo fails without LibTorch" failed
  IO.println "  device identity: queries are rejected without LibTorch"

/-- Entry point called from `NN/Tests/Suite.lean` under every runtime status. -/
def run : IO Unit := do
  IO.println "== LibTorch device identity =="
  match Buffer.runtimeStatus with
  | .notLinked => runUnavailable
  | .nativeUnavailable => pure ()
  | .nativeAvailable =>
      runReadback
      let self : System.FilePath := "/proc/self/exe"
      if !(← self.pathExists) then
        IO.println "  skipped: the device switching probe requires Linux /proc/self/exe"
        return
      let result ← IO.Process.output {
        cmd := self.toString
        args := #[]
        env := #[("TORCHLEAN_LIBTORCH_DEVICE_PROBE", some "switch"),
          ("TORCHLEAN_REQUIRE_CUDA", some "1")]
      }
      if result.exitCode != 0 then
        throw <| IO.userError <|
          s!"device switching probe failed (exit {result.exitCode}):\n" ++
            s!"{result.stdout}\n{result.stderr}"
      IO.print result.stdout

end DeviceInfo
end Cuda
end Tests
