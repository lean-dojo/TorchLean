/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Backend.ContractCheck
public import NN.Backend.Registry

/-!
# Backend Profiles

Named backend profiles bundle the choices that should move together: build availability,
kernel-selection policy, and the ordered capsule modules. This gives downstream APIs one object to
pass around instead of separately threading flags such as "CUDA", "LibTorch enabled", and "trusted
external allowed".
-/

@[expose] public section

namespace NN
namespace Backend

/-- A named kernel-selection profile for one device. -/
structure BackendProfile where
  /-- Human-readable profile name used in diagnostics and reports. -/
  name : String
  /-- Device, provider preference, assurance policy, and VJP ownership. -/
  policy : KernelPolicy
  /-- Devices and providers the build declares available to planning. -/
  availability : Availability
  /-- Capsule modules used to construct and validate the profile's planning registry. -/
  capsuleModules : Array Registry.CapsuleModule := Registry.maintainedModules
  deriving Repr

namespace BackendProfile

/--
Extend a profile with operation/provider capsule modules without changing model semantics,
availability, or assurance policy.

A new contribution replaces an existing module with the same name. Duplicate names within `modules`
are still rejected when the profile is planned.
-/
def withCapsuleModules (p : BackendProfile) (modules : Array Registry.CapsuleModule) :
    BackendProfile :=
  let names := modules.map (·.name)
  { p with
    capsuleModules :=
      modules ++ p.capsuleModules.filter (fun old => !names.contains old.name) }

/-- Capsule registry selected by the profile. -/
def registry (p : BackendProfile) : Array KernelCapsule :=
  Registry.flatten p.capsuleModules

/-- Select capsules using the profile registry, availability, and kernel policy. -/
def planOps (p : BackendProfile) (ops : Array BackendOp) : Except String KernelPlan := do
  Registry.validateModules p.capsuleModules
  NN.Backend.planOpsAvailable p.policy p.availability p.registry ops

/-- Select a capsule for every runtime-relevant IR node. -/
def planGraphNodes (p : BackendProfile) (g : NN.IR.Graph) :
    Except String NN.Backend.IR.GraphKernelPlan := do
  Registry.validateModules p.capsuleModules
  NN.Backend.IR.checkedPlanGraph p.policy p.availability p.registry g

/-- Plan, group, and contract-check a graph under the profile. -/
def acceptGraph (p : BackendProfile) (g : NN.IR.Graph) :
    Except String GraphKernelPlanResult := do
  let graphPlan ← p.planGraphNodes g
  pure <| acceptGraphKernelPlan graphPlan p.policy

/-- Maintained portable CPU/reference profile with runtime guards and regression evidence. -/
def checkedCpu : BackendProfile :=
  { name := "checked_cpu"
    policy :=
      { device := .cpu
        provider := .auto
        assurance := .checked
        vjpMode := .torchLeanTape }
    availability := Availability.cpu
    capsuleModules := Registry.maintainedModules }

/-- Checked native CUDA profile. External trusted providers are not admitted. -/
def checkedCuda : BackendProfile :=
  { name := "checked_cuda"
    policy :=
      { device := .cuda
        provider := .prefer .torchLean
        assurance := .checked
        vjpMode := .torchLeanTape }
    availability := Availability.cuda
    capsuleModules := Registry.maintainedModules }

/--
LibTorch forward scaling profile.

LibTorch is allowed to provide selected forward values, but TorchLean still records the graph/tape
boundary and does not hand local backward ownership to LibTorch autograd.
-/
def libTorchForwardCuda : BackendProfile :=
  { name := "libtorch_forward_cuda"
    policy :=
      { device := .cuda
        provider := .prefer .libTorch
        assurance := .external
        vjpMode := .torchLeanTape }
    availability := Availability.cuda (withLibTorch := true)
    capsuleModules := Registry.maintainedModules ++ [Registry.libTorchModule] }

/--
Maintained execution profile for a device, when TorchLean currently provides one.

Other named devices remain expressible through caller-supplied profiles. They are not silently
converted into profiles with empty registries.
-/
def maintainedForDevice? : Device → Option BackendProfile
  | .cpu => some checkedCpu
  | .cuda => some checkedCuda
  | .rocm | .metal | .wasm | .tpu | .trainium | .custom | .external => none

/-- Whether this profile registers at least one capsule for its selected device. -/
def hasDeviceCapsule (profile : BackendProfile) : Bool :=
  profile.registry.any fun capsule =>
    capsule.device == profile.policy.device

end BackendProfile

end Backend
end NN
