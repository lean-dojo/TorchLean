/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Backend.Types

/-!
# Backend Capsules

A backend capsule is TorchLean's unit of delegation to fast code.

The capsule records the contract TorchLean expects from a foreign implementation: which operation
it implements, what layout and shape conventions are assumed, how refinement of the operation's
canonical value and VJP semantics is justified, and what trust level the planner must account for.
The contract does not prove the implementation. At runtime, a capsule is paired with a typed
handler whose operation, provider, and device must agree with the selected contract.
-/

@[expose] public section

namespace NN
namespace Backend

/--
Reference to a native/FFI symbol used by a backend capsule.

The repository linter checks that `path` exists, that `symbol` occurs in that source file, and
that `buildTarget?`, when present, names a Lake target in `lakefile.lean`.
-/
structure NativeSymbolRef where
  path : String
  symbol : String
  buildTarget? : Option String := none
  deriving DecidableEq, BEq, Repr

/-- Source-level provenance for a contract descriptor. Provenance is not correctness evidence. -/
inductive ContractProvenance where
  | nativeSymbol (ref : NativeSymbolRef)
  deriving DecidableEq, BEq, Repr

/-- Concrete tensor-layout convention named by a backend contract. -/
inductive TensorLayout where
  /-- TorchLean's ordinary typed tensor representation. -/
  | canonicalTensor
  /-- Contiguous flat storage with the last axis varying fastest. -/
  | flatRowMajor
  /-- Contiguous CUDA tensor storage managed by LibTorch. -/
  | libTorchCudaView
  deriving DecidableEq, Repr

/-- The four contract fields carried by every kernel capsule. -/
inductive ContractObligation where
  | shape
  | layout
  | value
  | vjp
  deriving DecidableEq, Repr

/-- A structured backend obligation, independent of how evidence for it is obtained. -/
inductive ContractClaim where
  /-- Inputs and outputs satisfy the shape rule associated with an operation. -/
  | shapeSafety (op : BackendOp)
  /-- Runtime buffers use the declared layout for an operation. -/
  | layoutCompatibility (op : BackendOp) (layout : TensorLayout)
  /-- Forward execution refines the canonical semantics of the operation. -/
  | valueRefinement (op : BackendOp)
  /-- Local backward execution refines the canonical VJP semantics of the operation. -/
  | vjpRefinement (op : BackendOp) (mode : VJPMode)
  /-- The capsule intentionally supplies no reverse-mode implementation. -/
  | vjpUnavailable (op : BackendOp)
  deriving DecidableEq, Repr

/-- How a capsule justifies one part of its contract.

These constructors record engineering evidence and explicit trust boundaries. They do not turn a
foreign implementation into a proved refinement.
-/
inductive ContractEvidence where
  | runtimeGuard (name : String)
  | testSuite (name : String)
  | trustedBoundary (reason : String)
  | notApplicable
  deriving DecidableEq

instance : Repr ContractEvidence where
  reprPrec evidence _ := Std.Format.text <| match evidence with
    | .runtimeGuard name => s!"runtimeGuard({name})"
    | .testSuite name => s!"testSuite({name})"
    | .trustedBoundary reason => s!"trustedBoundary({reason})"
    | .notApplicable => "notApplicable"

namespace AssurancePolicy

/-- Whether the policy admits contract evidence of this kind. -/
def acceptsEvidence (policy : AssurancePolicy) : ContractEvidence → Bool
  | .trustedBoundary _ => policy.allowTrustedExternal
  | .runtimeGuard _ | .testSuite _ | .notApplicable => true

end AssurancePolicy

/-- A structured contract claim together with its evidence and human-readable explanation. -/
structure ContractDescriptor where
  claim : ContractClaim
  summary : String
  evidence : ContractEvidence
  provenance : Array ContractProvenance := #[]
  deriving DecidableEq

instance : Repr ContractDescriptor where
  reprPrec d _ := Std.Format.text s!"ContractDescriptor({repr d.claim}, {repr d.evidence})"

namespace ContractDescriptor

/-- A contract claim enforced by a named runtime guard. -/
def guarded (claim : ContractClaim) (summary guard : String)
    (provenance : Array ContractProvenance := #[]) : ContractDescriptor :=
  { claim, summary, evidence := .runtimeGuard guard, provenance }

/-- A contract claim covered by a named regression suite. -/
def tested (claim : ContractClaim) (summary suite : String)
    (provenance : Array ContractProvenance := #[]) : ContractDescriptor :=
  { claim, summary, evidence := .testSuite suite, provenance }

/-- A contract claim delegated to an explicitly named trusted boundary. -/
def trusted (claim : ContractClaim) (summary reason : String)
    (provenance : Array ContractProvenance := #[]) : ContractDescriptor :=
  { claim, summary, evidence := .trustedBoundary reason, provenance }

/-- Record that a forward-only capsule intentionally has no reverse-mode implementation. -/
def vjpUnavailable (op : BackendOp) (summary : String)
    (provenance : Array ContractProvenance := #[]) : ContractDescriptor :=
  { claim := .vjpUnavailable op
    summary
    evidence := .notApplicable
    provenance }

end ContractDescriptor

/-- Ordering contract for reductions. Different valid orders need not be bitwise equal.

The fixed-left graph certificate propagates a reduction only when the selected capsule promises
the same left fold as the canonical tensor semantics.
-/
inductive ReductionPolicy where
  | fixedLeft
  | implementationDefined
  | notApplicable
  deriving DecidableEq, Repr

namespace ReductionPolicy

/-- Stable report label for reduction order. -/
def label : ReductionPolicy → String
  | .fixedLeft => "fixed-left"
  | .implementationDefined => "implementation-defined"
  | .notApplicable => "n/a"

end ReductionPolicy

/-- Floating-point choices attached to one kernel capsule that numerical certificates consume. -/
structure NumericalPolicy where
  /-- The permitted reduction order; this does not specify every source of numerical variation. -/
  reduction : ReductionPolicy
  deriving DecidableEq, Repr

namespace ContractClaim

/-- Whether a claim has the expected kind and operation for a capsule contract field. -/
def matchesObligation (op : BackendOp) (vjpMode : VJPMode) :
    ContractObligation → ContractClaim → Bool
  | .shape, .shapeSafety claimOp => claimOp == op
  | .layout, .layoutCompatibility claimOp _ => claimOp == op
  | .value, .valueRefinement claimOp => claimOp == op
  | .vjp, .vjpRefinement claimOp claimMode =>
      claimOp == op && claimMode == vjpMode && claimMode != .none
  | .vjp, .vjpUnavailable claimOp => claimOp == op && vjpMode == .none
  | _, _ => false

end ContractClaim

/-- A contract-carrying fast kernel or reference implementation. -/
structure KernelCapsule where
  /-- Name used in selection reports and runtime errors. -/
  name : String
  /-- Backend operation implemented by this capsule. -/
  op : BackendOp
  /-- Provider responsible for the implementation. -/
  provider : Provider
  /-- Device on which the implementation runs. -/
  device : Device
  /-- Assurance level the planner must accept before selection. -/
  trustLevel : TrustLevel
  /-- Whether the capsule supplies forward execution. -/
  supportsForward : Bool := true
  /-- Form of reverse-mode support supplied by the capsule. -/
  vjpMode : VJPMode := .none
  /-- Shape-safety claim and its evidence. -/
  shapeContract : ContractDescriptor
  /-- Tensor-layout claim and its evidence. -/
  layoutContract : ContractDescriptor
  /-- Forward-value refinement claim and its evidence. -/
  valueContract : ContractDescriptor
  /-- Reverse-mode refinement claim and its evidence. -/
  vjpContract : ContractDescriptor
  /-- Floating-point behavior consumed by numerical certificates. -/
  numericalPolicy : NumericalPolicy
  deriving DecidableEq, Repr

/--
An executable implementation for one backend operation.

The result type is local to the call site, so this structure also accommodates operations whose
Lean signatures differ. The capsule argument gives specialized handlers access to the provider and
VJP mode after the common identity checks have succeeded.
-/
structure KernelHandler (β : Type) where
  name : String
  op : BackendOp
  provider : Provider
  device : Device
  execute : KernelCapsule → IO β

/--
A selected capsule paired with a handler for the same operation, provider, and device.

These equalities certify dispatch identity only. Numerical correctness remains exactly as strong as
the capsule's `ContractEvidence`; binding a handler does not turn tests or a trusted boundary into a
proof.
-/
structure ExecutableKernel (β : Type) where
  capsule : KernelCapsule
  handler : KernelHandler β
  operation_matches : handler.op = capsule.op
  provider_matches : handler.provider = capsule.provider
  device_matches : handler.device = capsule.device

namespace KernelHandler

/-- Whether a runtime handler has the identity advertised by a selected capsule. -/
def matchesCapsule {β : Type} (handler : KernelHandler β) (capsule : KernelCapsule) : Bool :=
  handler.op == capsule.op &&
    handler.provider == capsule.provider &&
    handler.device == capsule.device

end KernelHandler

namespace KernelCapsule

/-- Whether each descriptor states the obligation advertised by its field.

Evidence is useful only when it supports the right claim. This guard prevents, for example, a
value-refinement test from being placed in the shape field and then accepted as shape evidence.
-/
def contractsAligned (c : KernelCapsule) : Bool :=
  c.shapeContract.claim.matchesObligation c.op c.vjpMode .shape &&
  c.layoutContract.claim.matchesObligation c.op c.vjpMode .layout &&
  c.valueContract.claim.matchesObligation c.op c.vjpMode .value &&
  c.vjpContract.claim.matchesObligation c.op c.vjpMode .vjp

/-- Compare registration identity only. This ignores contracts and numerical policy;
use full capsule equality when grouping kernels or comparing assurance evidence. -/
def sameIdentity (a b : KernelCapsule) : Bool :=
  a.name == b.name && a.op == b.op && a.provider == b.provider && a.device == b.device

/-- Validate the common part of an eager executor request.

This does not decide how an operation invokes a provider. It prevents every runtime operation from
reimplementing the op, device, and wiring checks before interpreting the capsule locally.
-/
def validateEagerRequest (c : KernelCapsule) (op : BackendOp) (device : Device) :
    Except String Unit := do
  unless c.op == op do
    throw s!"capsule `{c.name}` implements `{c.op.name}`, not `{op.name}`"
  unless c.device == device do
    throw s!"capsule `{c.name}` targets `{c.device.cliName}`, not `{device.cliName}`"

/--
Pair a selected contract with the runtime handler that will execute it.

The returned equalities prevent an executor for one operation or provider from being presented as
another merely because both happen to share a Lean result type.
-/
def bind {β : Type} (c : KernelCapsule) (handler : KernelHandler β) :
    Except String (ExecutableKernel β) := do
  if hop : handler.op = c.op then
    if hprovider : handler.provider = c.provider then
      if hdevice : handler.device = c.device then
        pure
          ({ capsule := c
             handler
             operation_matches := hop
             provider_matches := hprovider
             device_matches := hdevice } : ExecutableKernel β)
      else
        throw <| s!"handler `{handler.name}` targets `{handler.device.cliName}`, but capsule " ++
          s!"`{c.name}` targets `{c.device.cliName}`"
    else
      throw <| s!"handler `{handler.name}` uses provider `{reprStr handler.provider}`, but " ++
        s!"capsule `{c.name}` selects `{reprStr c.provider}`"
  else
    throw <| s!"handler `{handler.name}` implements `{handler.op.name}`, but capsule `{c.name}` " ++
      s!"implements `{c.op.name}`"

/-- Whether the assurance policy admits this capsule's trust level. -/
def allowedBy (policy : KernelPolicy) (c : KernelCapsule) : Bool :=
  policy.assurance.acceptsTrust c.trustLevel

/-- Whether the provider preference admits this capsule. -/
def matchesPreference (policy : KernelPolicy) (c : KernelCapsule) : Bool :=
  match policy.provider with
  | .auto => true
  | .prefer _ => true
  | .only p => p = c.provider

/-- Whether this capsule is available on the selected device. -/
def matchesDevice (policy : KernelPolicy) (c : KernelCapsule) : Bool :=
  policy.device = c.device

/--
Whether the capsule's gradient boundary is compatible with the requested kernel policy.

`none` is inference mode, so any forward-capable capsule is suitable even when it also advertises a
VJP. In `torchLeanTape` mode TorchLean owns the global tape and backward traversal. A capsule may
still implement its local VJP either as TorchLean operations or as a named backend routine;
`backendVJP` requests the latter specifically. The maintained LibTorch routines execute under
disabled gradient recording: neither mode delegates a local or global graph to LibTorch autograd.
-/
def matchesVJP (policy : KernelPolicy) (c : KernelCapsule) : Bool :=
  if policy.vjpMode == .none || !c.op.requiresVJP then
    true
  else
    match policy.vjpMode with
    | .none => true
    | .torchLeanTape => c.vjpMode == .torchLeanTape || c.vjpMode == .backendVJP
    | .backendVJP => c.vjpMode == .backendVJP

/-- Planner-side admissibility predicate for a single capsule. -/
def admissible (policy : KernelPolicy) (c : KernelCapsule) : Bool :=
  c.supportsForward && c.contractsAligned && c.allowedBy policy && c.matchesPreference policy &&
    c.matchesDevice policy && c.matchesVJP policy

end KernelCapsule

namespace ExecutableKernel

/-- Invoke the handler bound to a selected capsule. -/
def run {β : Type} (kernel : ExecutableKernel β) : IO β :=
  kernel.handler.execute kernel.capsule

end ExecutableKernel

/--
Pick an admissible capsule for a typed operation.

An `.only` preference filters the catalog through `admissible`. An `.auto` preference preserves
catalog order. A `.prefer provider` request first searches that provider and then falls back to the
ordinary catalog, so preference does not depend on module registration order.
-/
def chooseCapsuleFor? (policy : KernelPolicy) (op : BackendOp)
    (capsules : Array KernelCapsule) : Option KernelCapsule :=
  let eligible := fun c => c.op == op && c.admissible policy
  match policy.provider with
  | .prefer provider =>
      (capsules.find? fun c => eligible c && c.provider == provider).orElse fun _ =>
        capsules.find? eligible
  | .auto | .only _ =>
      capsules.find? eligible

end Backend
end NN
