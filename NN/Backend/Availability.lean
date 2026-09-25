/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Backend.Capsule

/-!
# Backend Availability

Build-dependent availability for backend capsules.

`KernelPolicy` says which capsules selection may use. `Availability` says which devices and
providers a build declares. CPU-only and LibTorch CUDA builds share one semantic registry while
exposing different capsule subsets to the planner. An availability
declaration is not runtime discovery; executable paths still probe the linked runtime before
launching work.
-/

@[expose] public section

namespace NN
namespace Backend

/-- Devices and providers declared available to backend planning. -/
structure Availability where
  devices : Array Device := #[.cpu]
  providers : Array Provider := #[.reference, .torchLean]
  deriving Repr

namespace Availability

/-- Whether a capsule can even be considered on this build. -/
def admitsCapsule (a : Availability) (c : KernelCapsule) : Bool :=
  a.devices.contains c.device && a.providers.contains c.provider

/-- Keep only capsules available on this build. -/
def filterCapsules (a : Availability) (capsules : Array KernelCapsule) : Array KernelCapsule :=
  capsules.filter fun c => a.admitsCapsule c

/-- CPU/reference-only availability. -/
def cpu : Availability := {}

/-- CUDA availability with LibTorch primitives and TorchLean compositions of those primitives. -/
def cuda : Availability :=
  { devices := #[.cpu, .cuda]
    providers := #[.reference, .torchLean, .libTorch] }

end Availability

end Backend
end NN
