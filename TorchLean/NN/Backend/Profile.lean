/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Backend.ContractCheck
public import TorchLean.NN.Backend.Registry

/-!
# Backend Profiles

A `Backend.Profile` bundles the choices that move together: build availability, the
kernel-selection policy and the ordered implementation tables. Downstream APIs pass one profile
around, which keeps device availability, provider preference and the permission to accept
trusted-boundary evidence in step.
-/

@[expose] public section

namespace TorchLean
namespace Backend

/-- A named kernel-selection profile for one device. -/
structure Profile where
  /-- Human-readable profile name used in diagnostics and reports. -/
  name : String
  /-- Device, provider preference, assurance policy, and VJP ownership. -/
  policy : KernelPolicy
  /-- Devices and providers the build declares available to planning. -/
  availability : Availability
  /-- Implementation tables offering the profile's capsules, in preference order. -/
  implementations : Array Implementation := Registry.maintained
  deriving Repr

namespace Profile

/--
The profile `p` with the tables `impls` placed ahead of its own. Availability and the policy are
unchanged.

A table of `impls` replaces the table of `p` with the same name. Duplicate names within `impls`
are rejected when the profile is planned.
-/
def withImplementations (p : Profile) (impls : Array Implementation) : Profile :=
  let names := impls.map (·.name)
  { p with
    implementations :=
      impls ++ p.implementations.filter (fun old => !names.contains old.name) }

/-- Select capsules for registered operators with the profile's implementations, availability and
kernel policy, after `Registry.validate` has checked the implementation names. -/
def planOps (p : Profile) (ops : Array (Σ ins outs, Op ins outs)) :
    Except String KernelPlan := do
  Registry.validate p.implementations
  Backend.planOpsAvailable p.policy p.availability p.implementations ops

/-- The maintained CPU profile `checked_cpu`, which runs the reference table under the `checked`
assurance policy and records VJPs on the TorchLean tape. -/
def checkedCPU : Profile :=
  { name := "checked_cpu"
    policy :=
      { device := .cpu
        provider := .auto
        assurance := .checked
        vjpMode := .torchLeanTape }
    availability := Availability.cpu }

/--
The strict LibTorch CUDA profile `checked_cuda`, with TorchLean tape traversal and selected local
VJPs.

It admits no `trustedExternal` capsule and no trusted-boundary evidence, so an operator whose CUDA
capsule rests on a trusted boundary needs `libTorchCUDA` or another profile whose policy admits
one. No LibTorch autograd graph is recorded.
-/
def checkedCUDA : Profile :=
  { name := "checked_cuda"
    policy :=
      { device := .cuda
        provider := .prefer .libTorch
        assurance := .checked
        vjpMode := .torchLeanTape }
    availability := Availability.cuda }

/-- The default CUDA profile `libtorch_cuda`, which admits the recorded LibTorch trust boundaries.

Runtime guards check the bridge. The capsules without a recorded numerical comparison declare a
trusted boundary, which `checkedCUDA` rejects.
-/
def libTorchCUDA : Profile :=
  { checkedCUDA with
    name := "libtorch_cuda"
    policy := { checkedCUDA.policy with assurance := .external } }

/-- The maintained execution profile of a device: `checkedCPU` for the CPU and `libTorchCUDA` for
CUDA. -/
def maintainedForDevice? : Device → Option Profile
  | .cpu => some checkedCPU
  | .cuda => some libTorchCUDA

/-- Whether this profile registers an implementation table for its selected device. -/
def hasDeviceImplementation (profile : Profile) : Bool :=
  profile.implementations.any fun impl => impl.device == profile.policy.device

end Profile

end Backend
end TorchLean
