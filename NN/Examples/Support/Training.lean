/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Trainer.Memory
public import NN.Examples.Support.Command

/-!
# Runnable Example Training Support

Logging, runtime, and fixed-sample training helpers used by the runnable model examples.
-/

@[expose] public section

namespace NN.Examples.Support

open TorchLean

/-- Training-log note recording the selected CUDA allocator-reporting cadence. -/
def cudaMemoryNote (runtime : Runtime.Config) (steps requested : Nat) : String :=
  s!"cuda_mem_watch={TorchLean.Trainer.Memory.cadence runtime steps requested}"

/-!
Shared CLI and logging names for the built-in runnable examples.

These are public so examples can stay short and readable. They live under `Support`; ordinary
library code should usually use `Trainer`, `Data`, and `optim` directly.
-/

/-- Standard location for a runnable example's training log under `data/examples`. -/
def trainLogPath (stem : String) : System.FilePath :=
  System.FilePath.mk s!"data/examples/{stem}_trainlog.json"

/-- `device=...` note string used by example logs. -/
def deviceNote (runtime : Runtime.Config) : String :=
  s!"device={runtime.deviceName}"

/-- Example banner with the executable name, a short description, and the selected device. -/
def bannerWithDevice (exeName desc : String) (runtime : Runtime.Config) : String :=
  s!"{exeName}: {desc} (device={runtime.deviceName})"

/--
Two-line example banner: a headline with the selected device, then one detail line.
-/
def bannerWithDeviceDetails
    (exeName desc details : String) (runtime : Runtime.Config) : String :=
  bannerWithDevice exeName desc runtime ++ "\n" ++ details

/-- Fail with a contextual error when an executable model check is false. -/
def check (exeName msg : String) (b : Bool) : IO Unit :=
  unless b do throw <| IO.userError s!"{exeName}: {msg}"

end NN.Examples.Support
