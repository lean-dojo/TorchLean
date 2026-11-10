/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Backend
public import TorchLean.Tensor.Op.Codec

/-!
# Backend Capsules

A backend capsule is TorchLean's unit of delegation to fast code.

The capsule records the contract TorchLean expects from a foreign implementation: which registered
operator it implements, attributes included (its `Op.Wire` record), what layout and shape
conventions are assumed, how refinement of the operator's canonical value and VJP semantics is
justified, and what trust level the planner must account for. The contract does not prove the
implementation. An `Implementation` offers at most one capsule per operator, and
`Implementation.ofContracts` builds each from a `KernelContract`, which is a contract without its
operator. The planner chooses among the capsules offered for the requested operator. At runtime, a
selected capsule runs only with an implementation whose own capsule names the same operator,
provider and device (`KernelCapsule.implements`).
-/

@[expose] public section

namespace TorchLean
namespace Backend

/-- Concrete tensor-layout convention named by a backend contract. -/
inductive TensorLayout where
  /-- TorchLean's ordinary typed tensor representation. -/
  | canonicalTensor
  /-- Contiguous CUDA tensor storage managed by LibTorch. -/
  | libTorchCUDAView
  deriving DecidableEq, Repr

/-- The four contract fields carried by every kernel capsule. -/
inductive Contract.Obligation where
  | shape
  | layout
  | value
  | vjp
  deriving DecidableEq, Repr

/-- A structured backend obligation, independent of how evidence for it is obtained. -/
inductive Contract.Claim where
  /-- Inputs and outputs satisfy the shape rule associated with an operation. -/
  | shapeSafety (op : Op.Wire)
  /-- Runtime buffers use the declared layout for an operation. -/
  | layoutCompatibility (op : Op.Wire) (layout : TensorLayout)
  /-- Forward execution refines the canonical semantics of the operation. -/
  | valueRefinement (op : Op.Wire)
  /-- Local backward execution refines the canonical VJP semantics of the operation. -/
  | vjpRefinement (op : Op.Wire) (mode : VJPMode)
  /-- The capsule intentionally supplies no reverse-mode implementation. -/
  | vjpUnavailable (op : Op.Wire)
  deriving DecidableEq, Repr

/-- How a capsule justifies one part of its contract.

The constructors record engineering evidence and trust boundaries with their reasons, and none of
them turns a foreign implementation into a proved refinement.
-/
inductive Contract.Evidence where
  /-- The implementation calls the canonical definitions `names` directly, which records shared
  code, not a separate floating-point accuracy guarantee. Contracts write each name as a
  double-backtick literal, so a name that does not resolve fails to elaborate. -/
  | sharedDefinition (names : List Lean.Name)
  | runtimeGuard (name : String)
  | testSuite (name : String)
  | trustedBoundary (reason : String)
  | notApplicable
  deriving DecidableEq

instance : Repr Contract.Evidence where
  reprPrec evidence _ := Std.Format.text <| match evidence with
    | .sharedDefinition names => s!"sharedDefinition({", ".intercalate (names.map toString)})"
    | .runtimeGuard name => s!"runtimeGuard({name})"
    | .testSuite name => s!"testSuite({name})"
    | .trustedBoundary reason => s!"trustedBoundary({reason})"
    | .notApplicable => "notApplicable"

namespace AssurancePolicy

/-- Whether the policy admits contract evidence of this kind for `claim`. Only a claim that needs
no evidence, `vjpUnavailable`, may rest on `notApplicable`. -/
def acceptsEvidence (policy : AssurancePolicy) (claim : Contract.Claim) :
    Contract.Evidence → Bool
  | .trustedBoundary _ => policy.allowTrustedExternal
  | .sharedDefinition _ | .runtimeGuard _ | .testSuite _ => true
  | .notApplicable => claim matches .vjpUnavailable _

end AssurancePolicy

/-- A structured contract claim with its evidence and a human-readable explanation. A nonempty
`qualification` limits the claim, for instance by noting that values from a vendor math library are
not bit-exact to the Lean reference, and reports print it next to the claim. -/
structure Contract.Descriptor where
  /-- The claim, which names the operator it is about. -/
  claim : Claim
  /-- Human-readable explanation of the claim. -/
  summary : String
  /-- How the capsule justifies the claim. -/
  evidence : Evidence
  /-- A limit on the claim that reports print next to it, empty when there is none. -/
  qualification : String := ""
  deriving DecidableEq

instance : Repr Contract.Descriptor where
  reprPrec d _ := Std.Format.text s!"Contract.Descriptor({repr d.claim}, {repr d.evidence})"

namespace Contract.Descriptor

/-- A contract claim enforced by a named runtime guard. -/
def guarded (claim : Claim) (summary guard : String) : Descriptor :=
  { claim, summary, evidence := .runtimeGuard guard }

/-- A contract claim covered by a named regression suite. -/
def tested (claim : Claim) (summary suite : String) : Descriptor :=
  { claim, summary, evidence := .testSuite suite }

/-- A contract claim delegated to a trusted boundary, recorded with the reason for trusting it. -/
def trusted (claim : Claim) (summary reason : String) : Descriptor :=
  { claim, summary, evidence := .trustedBoundary reason }

/-- Record that a forward-only capsule intentionally has no reverse-mode implementation. -/
def vjpUnavailable (op : Op.Wire) (summary : String) : Descriptor :=
  { claim := .vjpUnavailable op
    summary
    evidence := .notApplicable }

end Contract.Descriptor

/-- Ordering contract for reductions. Different valid orders need not be bitwise equal.

The fixed-left graph certificate propagates a reduction only when the selected capsule promises
the same left fold as the canonical tensor semantics.
-/
inductive ReductionPolicy where
  | fixedLeft
  /-- Preserve fixed-left arithmetic unless its checked arithmetic selects exact replay. -/
  | fixedLeftOrExact
  | implementationDefined
  | notApplicable
  deriving DecidableEq, Repr

namespace ReductionPolicy

/-- Stable report label for reduction order. -/
def label : ReductionPolicy → String
  | .fixedLeft => "fixed-left"
  | .fixedLeftOrExact => "fixed-left-or-exact"
  | .implementationDefined => "implementation-defined"
  | .notApplicable => "n/a"

end ReductionPolicy

/-- Floating-point choices attached to one kernel capsule that numerical certificates consume. -/
structure NumericalPolicy where
  /-- The permitted reduction order. Other sources of numerical variation are left unspecified. -/
  reduction : ReductionPolicy
  deriving DecidableEq, Repr

namespace Contract.Claim

/-- Whether a claim has the kind and operation a capsule contract field expects. A VJP claim must
also fit the capsule's `vjpMode`: `vjpRefinement` for that mode when it is not `none`, and
`vjpUnavailable` when it is. -/
def matchesObligation (op : Op.Wire) (vjpMode : VJPMode) :
    Obligation → Claim → Bool
  | .shape, .shapeSafety claimOp => claimOp == op
  | .layout, .layoutCompatibility claimOp _ => claimOp == op
  | .value, .valueRefinement claimOp => claimOp == op
  | .vjp, .vjpRefinement claimOp claimMode =>
      claimOp == op && claimMode == vjpMode && claimMode != .none
  | .vjp, .vjpUnavailable claimOp => claimOp == op && vjpMode == .none
  | _, _ => false

end Contract.Claim

/-- Whether training through an operator with these inputs needs a local VJP: some input port
carries values, which receive a gradient. Random sources, top-`k` routing and `detach` read no
value port. -/
def requiresVJP (ins : List Port) : Bool :=
  ins.any fun port => port.kind == .value

end Backend
end TorchLean

namespace TorchLean.Op.Wire

/-- Whether training through the operator with this wire record needs a local VJP: the decoded
operator has a value input. A record that does not decode needs one. -/
def requiresVJP (w : Op.Wire) : Bool :=
  match Op.decode w with
  | some ⟨ins, _, _⟩ => Backend.requiresVJP ins
  | none => true

/-- The wire record of a typed operator needs a local VJP exactly when the operator does: this links
`Op.Wire.requiresVJP` to `Backend.requiresVJP`. -/
theorem requiresVJP_encode {ins outs : List Port} (op : Op ins outs) :
    (Op.encode ⟨ins, outs, op⟩).requiresVJP = Backend.requiresVJP ins := by
  simp [requiresVJP, Op.decode_encode]

end TorchLean.Op.Wire

namespace TorchLean
namespace Backend

/-- A contract-carrying fast kernel or reference implementation. -/
structure KernelCapsule where
  /-- Name used in selection reports and runtime errors. -/
  name : String
  /-- The registered operator implemented by this capsule, attributes included. -/
  op : Op.Wire
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
  shapeContract : Contract.Descriptor
  /-- Tensor-layout claim and its evidence. -/
  layoutContract : Contract.Descriptor
  /-- Forward-value refinement claim and its evidence. -/
  valueContract : Contract.Descriptor
  /-- Reverse-mode refinement claim and its evidence. -/
  vjpContract : Contract.Descriptor
  /-- Floating-point behavior consumed by numerical certificates. -/
  numericalPolicy : NumericalPolicy
  deriving DecidableEq, Repr

namespace KernelCapsule

/-- Whether each descriptor states the obligation advertised by its field.

Evidence counts only for the claim it supports, so the guard keeps, for example, a value-refinement
test placed in the shape field from passing as shape evidence.
-/
def contractsAligned (c : KernelCapsule) : Bool :=
  c.shapeContract.claim.matchesObligation c.op c.vjpMode .shape &&
  c.layoutContract.claim.matchesObligation c.op c.vjpMode .layout &&
  c.valueContract.claim.matchesObligation c.op c.vjpMode .value &&
  c.vjpContract.claim.matchesObligation c.op c.vjpMode .vjp

/-- Compare registration identity only: name, operator, provider and device. Trust level, forward
and VJP support, contracts and numerical policy are ignored, so grouping kernels or comparing
assurance evidence needs full capsule equality. -/
def sameIdentity (a b : KernelCapsule) : Bool :=
  a.name == b.name && a.op == b.op && a.provider == b.provider && a.device == b.device

/-- Validate the common part of an eager executor request: the requested operator, attributes
included, and device must be the capsule's.

How an operation invokes its provider is left to that operation, and the shared check saves every
runtime operation from reimplementing it before interpreting the capsule locally.
-/
def validateEagerRequest (c : KernelCapsule) (op : Op.Wire) (device : Device) :
    Except String Unit := do
  unless c.op == op do
    throw s!"capsule `{c.name}` implements `{c.op.tag}`, not `{op.tag}`"
  unless c.device == device do
    throw s!"capsule `{c.name}` targets `{c.device.cliName}`, not `{device.cliName}`"

/-- Whether an implementation whose own capsule is `entry` runs the selected capsule: both name the
same operator with its attributes, provider and device. Eager execution checks it before a LibTorch
CUDA entry runs. The check compares these labels alone, so it neither inspects the implementation
nor strengthens the selected capsule's evidence. -/
def implements (entry selected : KernelCapsule) : Bool :=
  entry.op == selected.op && entry.provider == selected.provider && entry.device == selected.device

/-- Whether the assurance policy admits this capsule's trust level. -/
def allowedBy (policy : KernelPolicy) (c : KernelCapsule) : Bool :=
  policy.assurance.acceptsTrust c.trustLevel

/-- Whether the provider preference admits this capsule. -/
def matchesPreference (policy : KernelPolicy) (c : KernelCapsule) : Bool :=
  match policy.provider with
  | .auto => true
  | .prefer _ => true
  | .only p => p = c.provider

/-- Whether the capsule targets the policy's device. -/
def matchesDevice (policy : KernelPolicy) (c : KernelCapsule) : Bool :=
  policy.device = c.device

/--
Whether the capsule's gradient support is compatible with the policy's `vjpMode`.

`none` is inference mode, where every capsule passes, including one that also advertises a VJP. In
`torchLeanTape` mode TorchLean owns the global tape and backward traversal, and a capsule may
implement its local VJP as TorchLean operations or as a named backend routine. `backendVJP`
accepts only the backend routine. In both gradient modes an operator without a value input needs
no VJP, so any capsule for it passes. The maintained LibTorch routines run with gradient recording
disabled, and neither mode delegates a local or global graph to LibTorch autograd.
-/
def matchesVJP (policy : KernelPolicy) (c : KernelCapsule) : Bool :=
  match policy.vjpMode with
  | .none => true
  | .torchLeanTape =>
      c.vjpMode == .torchLeanTape || c.vjpMode == .backendVJP || !c.op.requiresVJP
  | .backendVJP => c.vjpMode == .backendVJP || !c.op.requiresVJP

/-- Planner-side admissibility predicate for a single capsule. -/
def admissible (policy : KernelPolicy) (c : KernelCapsule) : Bool :=
  c.supportsForward && c.contractsAligned && c.allowedBy policy && c.matchesPreference policy &&
    c.matchesDevice policy && c.matchesVJP policy

end KernelCapsule

/-- One field of a contract without its operator. The field, the operator and the contract's layout
and VJP mode fix the claim, so a note keeps only the explanation, evidence and qualification. -/
structure Contract.Note where
  /-- Human-readable explanation of the field's claim. -/
  summary : String
  /-- How the capsule justifies the field's claim. -/
  evidence : Evidence
  /-- A limit on the field's claim, empty when there is none. -/
  qualification : String := ""
  deriving DecidableEq, Repr

/-- A capsule's contract without the operator it is offered for: its name, trust level, forward and
VJP support, the layout it assumes, the explanation and evidence of its four contract fields, and
its numerical policy. `KernelContract.capsule` builds the capsule of one operator. -/
structure KernelContract where
  /-- Name used in selection reports and runtime errors. -/
  name : String
  /-- Assurance level the planner must accept before selection. -/
  trustLevel : TrustLevel
  /-- Whether the capsule supplies forward execution. -/
  supportsForward : Bool := true
  /-- Form of reverse-mode support supplied by the capsule. With `none` the VJP field claims
  `vjpUnavailable`. -/
  vjpMode : VJPMode := .none
  /-- Tensor layout named by the layout claim. -/
  layout : TensorLayout
  /-- Explanation, evidence and qualification of the shape-safety claim. -/
  shape : Contract.Note
  /-- Explanation, evidence and qualification of the tensor-layout claim. -/
  layoutNote : Contract.Note
  /-- Explanation, evidence and qualification of the forward-value refinement claim. -/
  value : Contract.Note
  /-- Explanation, evidence and qualification of the reverse-mode claim. -/
  vjp : Contract.Note
  /-- Floating-point behavior consumed by numerical certificates. -/
  numericalPolicy : NumericalPolicy
  deriving DecidableEq, Repr

/-- The capsule of `op` under the contract, run by `provider` on `device`. Its identity and each of
its claims name the operator's wire record, and each field states the claim of its obligation
(`vjpUnavailable` when the contract's `vjpMode` is `none`) with its note's summary, evidence and
qualification. -/
def KernelContract.capsule (c : KernelContract) (provider : Provider) (device : Device)
    {ins outs : List Port} (op : Op ins outs) : KernelCapsule :=
  let w := Op.encode ⟨ins, outs, op⟩
  { name := c.name
    op := w
    provider
    device
    trustLevel := c.trustLevel
    supportsForward := c.supportsForward
    vjpMode := c.vjpMode
    shapeContract := { claim := .shapeSafety w, summary := c.shape.summary,
                       evidence := c.shape.evidence, qualification := c.shape.qualification }
    layoutContract := { claim := .layoutCompatibility w c.layout,
                        summary := c.layoutNote.summary, evidence := c.layoutNote.evidence,
                        qualification := c.layoutNote.qualification }
    valueContract := { claim := .valueRefinement w, summary := c.value.summary,
                       evidence := c.value.evidence, qualification := c.value.qualification }
    vjpContract :=
      { claim := match c.vjpMode with
          | .none => .vjpUnavailable w
          | mode => .vjpRefinement w mode
        summary := c.vjp.summary
        evidence := c.vjp.evidence
        qualification := c.vjp.qualification }
    numericalPolicy := c.numericalPolicy }

/-- An implementation table as the planner sees it: the capsule it offers for each registered
operator with its attributes, or `none`. Every offered capsule names the operator it is offered for
and the implementation's provider and device. -/
structure Implementation where
  /-- Name of the table. `Registry.validate` rejects two tables with the same name. -/
  name : String
  /-- Provider of every capsule the table offers. -/
  provider : Provider
  /-- Device of every capsule the table offers. -/
  device : Device
  /-- The capsule offered for an operator with its attributes, or `none`. -/
  capsule? : {ins outs : List Port} → Op ins outs → Option KernelCapsule
  capsule_op : ∀ {ins outs : List Port} (op : Op ins outs) (c : KernelCapsule),
    capsule? op = some c → c.op = Op.encode ⟨ins, outs, op⟩
  capsule_provider : ∀ {ins outs : List Port} (op : Op ins outs) (c : KernelCapsule),
    capsule? op = some c → c.provider = provider
  capsule_device : ∀ {ins outs : List Port} (op : Op ins outs) (c : KernelCapsule),
    capsule? op = some c → c.device = device

instance : Repr Implementation where
  reprPrec i _ := Std.Format.text
    s!"Implementation({i.name}, {reprStr i.provider}, {i.device.cliName})"

namespace Implementation

/-- The implementation offering, for each operator that `contract?` maps to a contract, that
contract's capsule on `provider` and `device`. -/
def ofContracts (name : String) (provider : Provider) (device : Device)
    (contract? : {ins outs : List Port} → Op ins outs → Option KernelContract) :
    Implementation where
  name
  provider
  device
  capsule? op := (contract? op).map (·.capsule provider device op)
  capsule_op op c h := by
    obtain ⟨contract, -, rfl⟩ := Option.map_eq_some_iff.mp h
    rfl
  capsule_provider op c h := by
    obtain ⟨contract, -, rfl⟩ := Option.map_eq_some_iff.mp h
    rfl
  capsule_device op c h := by
    obtain ⟨contract, -, rfl⟩ := Option.map_eq_some_iff.mp h
    rfl

/-- The implementation offering exactly `capsule`, for every operator whose wire record is the
capsule's. -/
def single (name : String) (capsule : KernelCapsule) : Implementation where
  name
  provider := capsule.provider
  device := capsule.device
  capsule? op := if capsule.op = Op.encode ⟨_, _, op⟩ then some capsule else none
  capsule_op op c h := by
    by_cases hop : capsule.op = Op.encode ⟨_, _, op⟩
    · simp only [hop, ↓reduceIte, Option.some.injEq] at h
      exact h ▸ hop
    · simp [hop] at h
  capsule_provider op c h := by
    by_cases hop : capsule.op = Op.encode ⟨_, _, op⟩
    · simp only [hop, ↓reduceIte, Option.some.injEq] at h
      exact h ▸ rfl
    · simp [hop] at h
  capsule_device op c h := by
    by_cases hop : capsule.op = Op.encode ⟨_, _, op⟩
    · simp only [hop, ↓reduceIte, Option.some.injEq] at h
      exact h ▸ rfl
    · simp [hop] at h

/-- `impls` offers `c` for `op`: some implementation's capsule for `op` is `c`. -/
def Offers (impls : Array Implementation) {ins outs : List Port} (op : Op ins outs)
    (c : KernelCapsule) : Prop :=
  ∃ impl ∈ impls, impl.capsule? op = some c

/-- The capsules `impls` offers for `op`, in implementation order. -/
def candidates (impls : Array Implementation) {ins outs : List Port} (op : Op ins outs) :
    Array KernelCapsule :=
  impls.filterMap (·.capsule? op)

/-- A capsule is a candidate for `op` exactly when `impls` offers it for `op`. -/
theorem mem_candidates {impls : Array Implementation} {ins outs : List Port} {op : Op ins outs}
    {c : KernelCapsule} : c ∈ candidates impls op ↔ Offers impls op c := by
  simp [candidates, Offers, Array.mem_filterMap]

end Implementation

/--
Pick an admissible capsule for a registered operator from the capsules the implementations offer
for it, in implementation order.

`.auto` and `.only` return the first admissible candidate, and under `.only` admissibility already
excludes every other provider. `.prefer provider` first searches that provider's admissible
capsules and then falls back to the first admissible candidate, so an admissible capsule of the
preferred provider wins wherever it sits in the implementation order.
-/
def chooseCapsuleFor? (policy : KernelPolicy) {ins outs : List Port} (op : Op ins outs)
    (impls : Array Implementation) : Option KernelCapsule :=
  let capsules := Implementation.candidates impls op
  let eligible := fun c => c.op == Op.encode ⟨ins, outs, op⟩ && c.admissible policy
  match policy.provider with
  | .prefer provider =>
      (capsules.find? fun c => eligible c && c.provider == provider).orElse fun _ =>
        capsules.find? eligible
  | .auto | .only _ =>
      capsules.find? eligible

end Backend
end TorchLean
