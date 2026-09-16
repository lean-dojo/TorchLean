/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Engine.Cuda.Tape
public import NN.Backend.Profile
import Mathlib.Algebra.Order.Algebra
public import NN.Spec.Core.Tensor.SomeTensor

/-!
# Torch Runtime Types

Public handles and options shared by the Torch runtime. Operation modules import their own
backends, so using a handle does not require the typed-graph recorder or CUDA operation catalog.
-/


@[expose] public section

namespace Runtime
namespace Autograd
namespace Torch

open Spec TorchLean TorchLean.Tensor

/--
Execution mode for the Torch-style front-end.

- `.eager`: execute operations immediately while building a dynamic autograd tape.
- `.typedGraph`: build shape-indexed SSA data instead of using only the dynamic tape. High-level
  trainers build their reusable graph once; the low-level session records graph data during a run.

The second mode changes the representation used for execution. It is not an optimizing compiler,
and graph construction alone does not prove the stored derivative rules correct.

This is not a CUDA Graph selector. CUDA is controlled by `Config.device` on the
eager execution path; CUDA Graph capture/replay will require a distinct persistent-buffer mode.
-/
inductive ExecutionMode where
  | eager
  | typedGraph
deriving Repr, DecidableEq

/--
Configuration controlling the behavior of the Torch-style front-end.

PyTorch comparison: these are approximately session/global settings, such as the default
`requiresGrad` value and requested execution device.
-/
structure Config where
  /-- Choose immediate tape execution or shape-indexed typed graph execution. -/
  execution : ExecutionMode := .eager
  /-- Device requested for execution. -/
  device : NN.Backend.Device := .cpu
  /-- Default `requiresGrad` value for parameters whose constructor omits it. Inputs default to
  false. -/
  requiresGradByDefault : Bool := true
  /--
  Global deterministic seed for runtime randomness.

  TorchLean keeps the semantic core pure and seed-threaded (JAX-style), so this value is the
  runtime seed that user code can thread into:
  - model initialization (per-layer init keys),
  - dataset shuffles / sampling,
  - and session-level RNG state (dropout, etc.).

  PyTorch analogue: `torch.manual_seed(seed)`.
  -/
  seed : Nat := 0
  /--
  Enable gradient tracking for newly recorded leaves.

  Inference helpers set this to `false`: forward values are still materialized so they can be read
  back, but parameter/input leaves are recorded with `requiresGrad = false`. The tape may still
  exist as a runtime value store; this flag controls whether newly recorded leaves participate in
  backward. It does not control whether forward execution may allocate intermediate values.
  -/
  gradEnabled : Bool := true
  /--
  Optional backend-contract profile selected for this run.

  Ordinary callers select only `device`; TorchLean then uses the maintained profile for that
  device. Advanced callers can provide a complete profile to choose provider preference, assurance
  policy, VJP ownership, target availability, and capsule modules together. The profile is rejected
  if its device differs from `device`.
  -/
  backendProfile? : Option NN.Backend.BackendProfile := none
  /-- Print each accepted backend capsule the first time an eager session executes it. -/
  showBackend : Bool := false
deriving Repr

/- Convenience API for device and backend-contract selection. -/
namespace Config

/-- Select a maintained execution device or explain that it needs a caller-supplied profile. -/
def withDevice (config : Config) (device : NN.Backend.Device) : Except String Config := do
  match NN.Backend.BackendProfile.maintainedForDevice? device with
  | some _ => pure { config with device := device, backendProfile? := none }
  | none =>
      throw (s!"device `{device.cliName}` has no maintained runtime profile; "
        ++ "provide a backend profile with executable capsules")

/-- Select a complete backend profile, including its device and contract policy. -/
def withBackendProfile (config : Config) (profile : NN.Backend.BackendProfile) : Config :=
  { config with device := profile.policy.device, backendProfile? := some profile }

/-- Resolve the backend profile selected by `device` and an optional advanced override. -/
def resolveBackendProfile (config : Config) : Except String NN.Backend.BackendProfile := do
  match config.backendProfile? with
  | some profile =>
      if profile.policy.device == config.device then
        pure profile
      else
        throw (s!"backend profile `{profile.name}` targets `{profile.policy.device.cliName}`, "
          ++ s!"but the runtime requested `{config.device.cliName}`")
  | none =>
      match NN.Backend.BackendProfile.maintainedForDevice? config.device with
      | some profile => pure profile
      | none =>
          throw (s!"device `{config.device.cliName}` has no maintained runtime profile; "
            ++ "provide a backend profile with executable capsules")

/-- Explain why a profile has no implementation catalog for its selected device. -/
def unsupportedProfileMessage (profile : NN.Backend.BackendProfile) : String :=
  s!"backend profile `{profile.name}` has no capsule for device `{profile.policy.device.cliName}`"

/-- Whether the effective runtime device is CUDA. -/
def usesCuda (config : Config) : Bool :=
  config.device == .cuda

/-- Reject unresolved profiles and profiles with no registered capsule for their selected device. -/
def validateDevice (config : Config) : Except String Unit := do
  let profile ← config.resolveBackendProfile
  if profile.hasDeviceCapsule then
    pure ()
  else
    throw <| unsupportedProfileMessage profile

/-- Validate both the named device and the linked native runtime before executing user code. -/
def validateForExecution (config : Config) : IO Unit := do
  match config.validateDevice with
  | .ok () => pure ()
  | .error msg => throw <| IO.userError msg
  match config.device with
  | .cpu => pure ()
  | .cuda => Runtime.Autograd.Cuda.Buffer.requireNativeRuntime
  | device =>
      throw <| IO.userError
        s!"device `{device.cliName}` has no linked TorchLean execution runtime"

/-- CLI/log spelling for the effective runtime device. -/
def deviceName (config : Config) : String :=
  config.device.cliName

/-- Effective backend-contract profile after applying inference-only VJP policy. -/
def effectiveBackendProfile (config : Config) : Except String NN.Backend.BackendProfile := do
  let profile ← config.resolveBackendProfile
  pure <| if config.gradEnabled then
    profile
  else
    { profile with policy := { profile.policy with vjpMode := .none } }

/-- Select a backend capsule for one operation under this runtime configuration. -/
def planBackendOp (config : Config) (op : NN.Backend.BackendOp) :
    Except String NN.Backend.AcceptedKernel := do
  let profile ← config.effectiveBackendProfile
  let plan ← profile.planOps #[op]
  match plan.kernels[0]? with
  | some k =>
      match k.accept profile.policy with
      | .error failures =>
          .error s!"backend profile {profile.name} rejected `{op.name}`: {repr failures}"
      | .ok accepted => .ok accepted
  | none =>
      .error s!"backend profile {profile.name} returned no capsule for {op.name}"

end Config

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

initialize ownerCounter : Std.Mutex Nat ← Std.Mutex.new 0

/-- Allocate a process-unique owner id, serializing concurrent session creation. -/
def freshOwner : IO Nat :=
  ownerCounter.atomically do
    let owner ← get
    set (owner + 1)
    pure owner

/-- Reject a handle from another session or an earlier recording phase. -/
def validateAgainst (expectedOwner : Nat) (expectedGeneration : IO.Ref Nat) (identity : RefIdentity)
    (kind : String := "reference") : IO Unit := do
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
shape index `s` makes shape mismatches explicit at compile time.
-/
structure TensorRef (α : Type) (s : Shape) where
  /-- Node/leaf identifier in the owning session tape. -/
  id : Nat
  /-- Owning session and recording generation; absent only on internal, not-yet-committed
  handles. -/
  identity? : Option RefIdentity := none
deriving Repr

/--
Handle to a `Nat` stored in the session's non-differentiable environment.

This is used to model index-like inputs (class labels, gather indices, etc.) which should not
receive gradients.
-/
structure NatRef where
  /-- Index into the session's non-differentiable `Nat` environment. -/
  id : Nat
  /-- Owning session and recording generation. -/
  identity? : Option RefIdentity := none
deriving Repr

/--
Trainable parameter: a mutable tensor value plus metadata.

PyTorch comparison: analogous to `torch.nn.Parameter`, except the parameter becomes part of the
autograd graph only when you `use` it in a particular session/tape.
-/
structure Param (α : Type) [Storage α] (s : Shape) where
  /-- Optional user-facing name for logging/debugging. -/
  name : Option String := none
  /-- Value at the current point. -/
  value : IO.Ref (Tensor α s)
  /--
  Optional CUDA-resident mirror of `value`.

  The eager CUDA trainer uses this as a persistent-parameter cache: repeated forward
  passes can reuse the device buffer instead of uploading the host tensor every step.  The host
  `value` remains the public source for CPU runs and exact/symbolic scalar instantiations.  When a
  caller explicitly reads parameters after CUDA training, the device mirror is copied back here.
  -/
  cudaValue : IO.Ref (Option Runtime.Autograd.Cuda.AnyBuffer)
  /--
  Whether `value` is known to match `cudaValue`.

  CUDA optimizer steps mark this `false` after updating only the device mirror.  Public parameter
  readback synchronizes and flips it back to `true`.
  -/
  hostCurrent : IO.Ref Bool
  /-- Whether this parameter receives accumulated gradients and optimizer updates. -/
  requiresGrad : Bool := true

/-- Allocate fresh host storage with no device mirror and an initially current host value. -/
def Param.Internal.create {α : Type} [Storage α] {shape : Shape}
    (initial : Tensor α shape) (name : Option String := none) (requiresGrad : Bool := true) :
    IO (Param α shape) := do
  let value ← IO.mkRef initial
  let cudaValue ← IO.mkRef (none : Option Runtime.Autograd.Cuda.AnyBuffer)
  let hostCurrent ← IO.mkRef true
  pure { name, value, cudaValue, hostCurrent, requiresGrad }

/--
Type-erased parameter wrapper.

This exists so session code can store heterogeneous parameter shapes in a single `HashMap` keyed
by leaf id (used for SGD updates).
-/
structure AnyParam (α : Type) [Storage α] where
  /-- Runtime shape of the erased parameter. -/
  s : Shape
  /-- Whether the underlying parameter receives optimizer updates. -/
  requiresGrad : Bool
  /-- Read the current parameter value with its runtime shape. -/
  get : IO (Spec.SomeTensor α)
  /-- Overwrite the current parameter value, checking shape at the call site. -/
  set : Spec.SomeTensor α → IO Unit
  /-- Store a CUDA buffer mirror without forcing an immediate host download.

  The supplied buffer can be shared through Lean references, but another owner must not
  force-release it while the parameter or a recorded snapshot still uses it. A scope-owned tape
  intermediate needs an independent copy or a transfer of that scope's release responsibility before
  installation.
  -/
  setCuda : Runtime.Autograd.Cuda.AnyBuffer → IO Unit

namespace AnyParam

/--
Make the result of a native CUDA cleanup call observable in `IO`.

Some CUDA cleanup functions are exposed as pure opaque calls because they are also useful in pure
buffer-building expressions. In executable cleanup paths, we still want the native call to be
sequenced with the surrounding eager-session updates. Branching into `IO` on the returned flag gives
Lean a real dependency on the result without printing anything or changing behavior.
-/
def observeCudaCleanupFlag (released : UInt32) : IO Unit :=
  if released == 0 then
    IO.sleep 0
  else
    pure ()

/-- Remove the parameter's reference to its cached CUDA mirror.

A recorded leaf keeps the buffer that held the parameter's value when `use` was called. Replacing
the parameter must preserve that snapshot, including when another session still refers to it.
Dropping the cache reference lets Lean's external-object finalizer reclaim the allocation once
the cache, recorded values, and backward closures have all stopped referring to it. An explicit
`Buffer.releaseIO` here would invalidate those other references immediately.
-/
def releaseCachedCudaValue {α : Type} [Storage α] {s : Shape} (p : Param α s) : IO Unit :=
  p.cudaValue.set none

/--
Package a typed `Param α s` as an `AnyParam α`, checking shape on `set`.

This is the bridge that allows generic optimizers/update routines to operate over heterogeneous
parameter packs.
-/
def ofParam {α : Type} [Storage α] {s : Shape} (p : Param α s) : AnyParam α :=
  { s := s
    requiresGrad := p.requiresGrad
    get := do
      let v ← p.value.get
      pure (Spec.SomeTensor.ofTensor v)
    set := fun v => do
      if h : v.shape = s then
        releaseCachedCudaValue p
        p.value.set (v.cast h)
        p.hostCurrent.set true
      else
        throw <| IO.userError <|
          s!"torch: param update shape mismatch "
            ++ s!"(expected {Shape.pretty s}, got {Shape.pretty v.shape})"
    setCuda := fun v => do
      if _h : v.s = s then
        p.cudaValue.set (some { s := s, buf := v.buf })
        p.hostCurrent.set false
      else
        throw <| IO.userError <|
          s!"torch: CUDA param update shape mismatch "
            ++ s!"(expected {Shape.pretty s}, got {Shape.pretty v.s})"
          }

end AnyParam
end Torch
end Autograd
end Runtime
