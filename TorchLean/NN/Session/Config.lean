/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.LibTorch.Tape
public import TorchLean.NN.Backend.Profile
public import TorchLean.Tensor.SomeTensor

/-!
# Torch Runtime Types

Public handles and options shared by the Torch runtime: the run's `Session.Config` (device, seed and
backend profile), tensor references and parameters. `Session.Config.planOp` selects an operator's
backend capsule through the profile, so this module imports the backend registry, LibTorch CUDA
table included.
-/


@[expose] public section

namespace TorchLean

open TorchLean.Tensor

/--
Configuration of the Torch-style front-end for one run.

PyTorch comparison: these are approximately session/global settings, such as the requested
execution device and the run's seed.
-/
structure Session.Config where
  /-- Device requested for execution. -/
  device : Backend.Device := .cpu
  /--
  Global deterministic seed for runtime randomness.

  TorchLean keeps the semantic core pure and seed-threaded (JAX-style), so user code threads this
  seed into model initialization (per-layer init keys), dataset shuffles and sampling, and
  session-level RNG state such as dropout.

  PyTorch analogue: `torch.manual_seed(seed)`.
  -/
  seed : Nat := 0
  /--
  Optional backend-contract profile selected for this run.

  Ordinary callers select only `device`, and TorchLean then uses the maintained profile for that
  device. Advanced callers can provide a complete profile that chooses provider preference,
  assurance policy, VJP ownership, target availability, and implementation tables together. A
  profile whose device differs from `device` is rejected.
  -/
  backendProfile? : Option Backend.Profile := none
  /-- Print the first accepted selection of each backend capsule name in an eager session. -/
  showBackend : Bool := false
deriving Repr

/- Convenience API for device and backend-contract selection. -/
namespace Session.Config

/-- Select a maintained execution device or explain that it needs a caller-supplied profile. -/
def withDevice (config : Config) (device : Backend.Device) : Except String Config := do
  match Backend.Profile.maintainedForDevice? device with
  | some _ => pure { config with device := device, backendProfile? := none }
  | none =>
      throw (s!"device `{device.cliName}` has no maintained runtime profile; "
        ++ "provide a backend profile with executable capsules")

/-- Select a complete backend profile, including its device and contract policy. -/
def withBackendProfile (config : Config) (profile : Backend.Profile) : Config :=
  { config with device := profile.policy.device, backendProfile? := some profile }

/-- Resolve the backend profile selected by `device` and an optional advanced override. -/
def resolveBackendProfile (config : Config) : Except String Backend.Profile := do
  match config.backendProfile? with
  | some profile =>
      if profile.policy.device == config.device then
        pure profile
      else
        throw (s!"backend profile `{profile.name}` targets `{profile.policy.device.cliName}`, "
          ++ s!"but the runtime requested `{config.device.cliName}`")
  | none =>
      match Backend.Profile.maintainedForDevice? config.device with
      | some profile => pure profile
      | none =>
          throw (s!"device `{config.device.cliName}` has no maintained runtime profile; "
            ++ "provide a backend profile with executable capsules")

/-- Explain why a profile has no implementation table for its selected device. -/
def unsupportedMessage (profile : Backend.Profile) : String :=
  s!"backend profile `{profile.name}` has no implementation for device " ++
    s!"`{profile.policy.device.cliName}`"

/-- Whether the effective runtime device is CUDA. -/
def usesCUDA (config : Config) : Bool :=
  config.device == .cuda

/-- Reject unresolved profiles and profiles with no implementation table for their selected
device. -/
def validateDevice (config : Config) : Except String Unit := do
  let profile ← config.resolveBackendProfile
  if profile.hasDeviceImplementation then
    pure ()
  else
    throw <| unsupportedMessage profile

/-- Validate both the named device and the linked native runtime before executing user code. -/
def validateForExecution (config : Config) : IO Unit := do
  match config.validateDevice with
  | .ok () => pure ()
  | .error msg => throw <| IO.userError msg
  match config.device with
  | .cpu => pure ()
  | .cuda => LibTorch.Buffer.requireNativeRuntime

/-- Effective backend-contract profile. With gradients off (`gradEnabled := false`, as in
inference) the policy's VJP mode is `.none`, because no operation needs a VJP. -/
def effectiveBackendProfile (config : Config) (gradEnabled : Bool := true) :
    Except String Backend.Profile := do
  let profile ← config.resolveBackendProfile
  pure <| if gradEnabled then
    profile
  else
    { profile with policy := { profile.policy with vjpMode := .none } }

/-- Select the backend capsule for one registered operator, attributes included, and check that the
profile's policy accepts it. -/
def planOp (config : Config) {ins outs : List Port} (op : Op ins outs)
    (gradEnabled : Bool := true) : Except String Backend.AcceptedKernel := do
  let profile ← config.effectiveBackendProfile gradEnabled
  let plan ← profile.planOps #[⟨ins, outs, op⟩]
  let tag := (Op.encode ⟨ins, outs, op⟩).tag
  match plan.kernels[0]? with
  | some k =>
      match k.accept profile.policy with
      | .error failures =>
          .error s!"backend profile {profile.name} rejected `{tag}`: {repr failures}"
      | .ok accepted => .ok accepted
  | none =>
      .error s!"backend profile {profile.name} returned no capsule for {tag}"

end Session.Config

end TorchLean

namespace TorchLean

open TorchLean.Tensor

/-- Runtime identity attached to a session-owned handle.

`owner` is a process-unique session id. The stored `generation` distinguishes recording phases
within one session.
-/
structure RefIdentity where
  /-- Process-unique owner id allocated when the session is created. -/
  owner : Nat
  /-- Generation observed when the handle was created. -/
  generation : Nat

instance : Repr RefIdentity where
  reprPrec identity _ := s!"RefIdentity(generation={identity.generation})"

namespace RefIdentity

/-- The next owner id that `freshOwner` hands out, behind a mutex. -/
initialize ownerCounter : Std.Mutex Nat ← Std.Mutex.new 0

/-- Allocate a process-unique owner id, serializing concurrent session creation. -/
def freshOwner : IO Nat :=
  ownerCounter.atomically do
    let owner ← get
    set (owner + 1)
    pure owner

/-- Reject an unowned handle, a foreign session, or an earlier recording phase, in that order. -/
def validate (expectedOwner : Nat) (expectedGeneration : IO.Ref Nat)
    (identity? : Option RefIdentity)
    (kind : String := "reference") : IO Unit := do
  let some identity := identity?
    | throw <| IO.userError s!"torch: {kind} has no session owner"
  unless expectedOwner == identity.owner do
    throw <| IO.userError s!"torch: {kind} belongs to a different session"
  let currentGeneration ← expectedGeneration.get
  if identity.generation != currentGeneration then
    throw <| IO.userError <|
      s!"torch: stale {kind} from generation {identity.generation}; current generation is " ++
        s!"{currentGeneration} (resetTape invalidates recorded references)"

end RefIdentity

/--
Opaque handle to a tensor value in the current execution context.

The identifier refers to a leaf or node in the owning eager tape or typed graph recorder. Runtime
validation checks both owner identity and recording generation before the id is used. The phantom
shape index `s` turns shape mismatches into type errors.
-/
structure TensorRef (α : Type) (s : Shape) where
  /-- Node/leaf identifier in the owning session tape. -/
  id : Nat
  /-- Owning session and recording generation. Only uncommitted internal handles leave it `none`. -/
  identity? : Option RefIdentity := none
deriving Repr

/--
Trainable parameter: a mutable tensor value plus metadata.

PyTorch comparison: analogous to `torch.nn.Parameter`, except the parameter becomes part of the
autograd graph only when a session records it with `use`.
-/
structure Param (α : Type) [Storage α] (s : Shape) where
  /-- Optional user-facing name for logging/debugging. -/
  name : Option String := none
  /-- Host value of the parameter, current whenever `hostSynced` is `true`. -/
  value : IO.Ref (Tensor α s)
  /--
  Optional CUDA-resident mirror of `value`.

  The eager CUDA trainer keeps parameters here between steps, so repeated forward passes reuse the
  device buffer and skip uploading the host tensor. CPU runs and exact or symbolic scalar
  instantiations read the host `value`. When a caller reads parameters after CUDA training, the
  device mirror is copied back to `value`.
  -/
  cudaValue : IO.Ref (Option LibTorch.SomeBuffer)
  /--
  Whether `value` is known to match `cudaValue`.

  A CUDA optimizer step updates only the device mirror and sets this to `false`. Public parameter
  readback copies the mirror to the host and sets it back to `true`.
  -/
  hostSynced : IO.Ref Bool
  /-- Whether this parameter receives accumulated gradients and optimizer updates. -/
  requiresGrad : Bool := true

/-- Allocate fresh host storage with no device mirror and an initially current host value. -/
def Param.create {α : Type} [Storage α] {shape : Shape}
    (initial : Tensor α shape) (name : Option String := none) (requiresGrad : Bool := true) :
    IO (Param α shape) := do
  let value ← IO.mkRef initial
  let cudaValue ← IO.mkRef (none : Option LibTorch.SomeBuffer)
  let hostCurrent ← IO.mkRef true
  pure { name, value, cudaValue, hostSynced := hostCurrent, requiresGrad }

/--
Type-erased parameter wrapper.

Session code stores parameters of different shapes in one `HashMap` keyed by leaf id, which the
optimizer steps read.
-/
structure SomeParam (α : Type) [Storage α] where
  /-- Runtime shape of the erased parameter. -/
  s : Shape
  /-- Whether the underlying parameter receives optimizer updates. -/
  requiresGrad : Bool
  /-- Read the current parameter value with its runtime shape. -/
  get : IO (SomeTensor α)
  /-- Overwrite the current parameter value after checking its runtime shape. -/
  set : SomeTensor α → IO Unit
  /-- Store a CUDA buffer mirror without forcing an immediate host download.

  The supplied buffer can be shared through Lean references, but no other owner may force-release
  it while the parameter or a recorded snapshot uses it. Before a scope-owned tape intermediate is
  installed, it needs an independent copy or a transfer of that scope's release responsibility.
  -/
  setCUDA : LibTorch.SomeBuffer → IO Unit

namespace SomeParam

/-- Remove the parameter's reference to its cached CUDA mirror.

A recorded leaf keeps the buffer that held the parameter's value when `use` was called. Replacing
the parameter must preserve that snapshot, also when another session refers to it. Dropping only the
cache reference lets Lean's external-object finalizer reclaim the allocation once the cache, the
recorded values and the backward closures have all let go of it. Calling `Buffer.releaseIO` here
would invalidate those other references at once.
-/
def releaseCachedCUDAValue {α : Type} [Storage α] {s : Shape} (p : Param α s) : IO Unit :=
  p.cudaValue.set none

/--
Package a typed `Param α s` as a `SomeParam α`, checking shape on `set`.

Generic optimizers and update routines use it to work over heterogeneous parameter packs. `set`
drops the cached CUDA mirror, and `setCUDA` marks the host value stale. `get` reads only the host
value and throws while it is stale (`Param.syncToHost` refreshes it).
-/
def ofParam {α : Type} [Storage α] {s : Shape} (p : Param α s) : SomeParam α :=
  { s := s
    requiresGrad := p.requiresGrad
    get := do
      unless ← p.hostSynced.get do
        throw <| IO.userError "torch: the host value is stale after a CUDA update"
      let v ← p.value.get
      pure (SomeTensor.ofTensor v)
    set := fun v => do
      if h : v.shape = s then
        releaseCachedCUDAValue p
        p.value.set (v.cast h)
        p.hostSynced.set true
      else
        throw <| IO.userError <|
          s!"torch: param update shape mismatch "
            ++ s!"(expected {s}, got {v.shape})"
    setCUDA := fun v => do
      if _h : v.s = s then
        p.cudaValue.set (some { s := s, buf := v.buf })
        p.hostSynced.set false
      else
        throw <| IO.userError <|
          s!"torch: CUDA param update shape mismatch "
            ++ s!"(expected {s}, got {v.s})"
          }

end SomeParam
end TorchLean
