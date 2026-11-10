/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Backend.Capsule

/-!
# Backend Availability

Build-dependent availability for backend implementations.

`KernelPolicy` says which capsules selection may use, and `Availability` says which devices and
providers a build declares. CPU-only and LibTorch CUDA builds share one semantic registry and
expose different implementations to the planner. A declaration discovers nothing at run time, so
executable paths probe the linked runtime before launching work.
-/

@[expose] public section

namespace TorchLean
namespace Backend

/-- Devices and providers declared available to backend planning. -/
structure Availability where
  /-- Devices available to planning. -/
  devices : Array Device := #[.cpu]
  /-- Providers available to planning. -/
  providers : Array Provider := #[.reference, .torchLean]
  deriving Repr

namespace Availability

/-- Whether this build declares the capsule's device and provider. -/
def admitsCapsule (a : Availability) (c : KernelCapsule) : Bool :=
  a.devices.contains c.device && a.providers.contains c.provider

/-- Whether an implementation's provider and device are available on this build. -/
def admitsImplementation (a : Availability) (impl : Implementation) : Bool :=
  a.devices.contains impl.device && a.providers.contains impl.provider

/-- Keep only implementations available on this build. -/
def filter (a : Availability) (impls : Array Implementation) : Array Implementation :=
  impls.filter a.admitsImplementation

/-- The filtered implementations offer `c` for `op` exactly when `impls` does and `a` admits `c`,
because every offered capsule has its implementation's provider and device. -/
theorem offers_filter {a : Availability} {impls : Array Implementation} {ins outs : List Port}
    {op : Op ins outs} {c : KernelCapsule} :
    Implementation.Offers (a.filter impls) op c ↔
      Implementation.Offers impls op c ∧ a.admitsCapsule c = true := by
  constructor
  · rintro ⟨impl, hmem, hc⟩
    obtain ⟨hmem, havailable⟩ := Array.mem_filter.mp hmem
    refine ⟨⟨impl, hmem, hc⟩, ?_⟩
    simpa [admitsCapsule, admitsImplementation, impl.capsule_provider op c hc,
      impl.capsule_device op c hc] using havailable
  · rintro ⟨⟨impl, hmem, hc⟩, havailable⟩
    refine ⟨impl, Array.mem_filter.mpr ⟨hmem, ?_⟩, hc⟩
    simpa [admitsCapsule, admitsImplementation, impl.capsule_provider op c hc,
      impl.capsule_device op c hc] using havailable

/-- Availability of a CPU-only build: the CPU device with the reference and TorchLean providers. -/
def cpu : Availability := {}

/-- Availability of a LibTorch CUDA build: `cpu` plus the CUDA device and the LibTorch provider. -/
def cuda : Availability :=
  { devices := #[.cpu, .cuda]
    providers := #[.reference, .torchLean, .libTorch] }

end Availability

end Backend
end TorchLean
