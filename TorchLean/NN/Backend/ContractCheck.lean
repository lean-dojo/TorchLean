/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Backend.Grouping

/-!
# Backend Contract Checks

The planner admits a capsule only when it implements the requested operation with a forward pass,
its contract claims match their fields, and its device, provider, gradient mode and trust level
suit the kernel policy. The contract audit then looks inside the selected capsules, where each of
the four descriptors must rest on evidence the assurance policy accepts.

Acceptance repeats both checks at the execution boundary, because `PlannedKernel` is a public
diagnostic record that a caller can assemble by hand. Accepted values carry the successful gate as
a proof field. Graph acceptance derives its grouped audit plan from the graph plan, so a caller
cannot pass in unrelated groups.
-/

@[expose] public section

namespace TorchLean
namespace Backend

/-- One contract obligation of one selected backend kernel. -/
structure ObligationReport where
  /-- Wire record of the kernel's operator. -/
  op : Op.Wire
  /-- Name of the selected capsule. -/
  capsuleName : String
  /-- The contract field the obligation comes from. -/
  obligation : Contract.Obligation
  /-- The claim that the field states. -/
  claim : Contract.Claim
  /-- The evidence the capsule gives for the claim. -/
  evidence : Contract.Evidence
  deriving Repr

/-- Result of checking a kernel plan against an assurance policy. -/
inductive Contract.Check where
  | accepted
  | rejected (reports : Array ObligationReport)
  deriving Repr

namespace PlannedKernel

/-- All contract obligations of a selected kernel. -/
def obligationReports (k : PlannedKernel) : Array ObligationReport :=
  k.contracts.map fun (obligation, descriptor) =>
    { op := k.op
      capsuleName := k.capsule.name
      obligation
      claim := descriptor.claim
      evidence := descriptor.evidence }

end PlannedKernel

namespace KernelPlan

/-- All contract obligations of all selected kernels. -/
def obligationReports (p : KernelPlan) : Array ObligationReport :=
  p.kernels.flatMap PlannedKernel.obligationReports

/-- Obligations whose evidence the policy does not accept. -/
def rejectedReports (policy : AssurancePolicy) (p : KernelPlan) : Array ObligationReport :=
  p.obligationReports.filter fun r => !policy.acceptsEvidence r.claim r.evidence

/-- Check every obligation of a selected plan against an assurance policy. -/
def checkContracts (policy : AssurancePolicy) (p : KernelPlan) : Contract.Check :=
  let rejected := p.rejectedReports policy
  if rejected.isEmpty then .accepted else .rejected rejected

end KernelPlan

namespace GroupedKernelPlan

/-- Check every obligation of a grouped graph plan against an assurance policy. -/
def checkContracts (policy : AssurancePolicy) (p : GroupedKernelPlan) : Contract.Check :=
  p.toKernelPlan.checkContracts policy

end GroupedKernelPlan

/-- One reason a planned kernel cannot cross the execution boundary. -/
inductive AcceptanceFailure where
  /-- The planned operation and the capsule's advertised operation differ. -/
  | operationMismatch (planned capsule : Op.Wire)
  /-- The capsule has no forward implementation. -/
  | forwardUnsupported (op : Op.Wire) (capsuleName : String)
  /-- A contract descriptor states a claim for the wrong field, operator or VJP mode. -/
  | contractsMisaligned (op : Op.Wire) (capsuleName : String)
  /-- The selected assurance policy does not admit the capsule's trust level. -/
  | trustRejected (op : Op.Wire) (capsuleName : String) (trust : TrustLevel)
  /-- The provider policy excludes the capsule. -/
  | providerRejected (op : Op.Wire) (capsuleName : String) (provider : Provider)
  /-- The capsule targets a different device. -/
  | deviceMismatch (op : Op.Wire) (capsuleName : String)
      (requested actual : Device)
  /-- The capsule cannot satisfy the requested gradient mode. -/
  | vjpMismatch (op : Op.Wire) (capsuleName : String)
      (requested actual : VJPMode)
  /-- One contract descriptor rests on evidence rejected by the assurance policy. -/
  | evidenceRejected (report : ObligationReport)
  deriving Repr

namespace PlannedKernel

/-- Whether a planned kernel passes the complete policy and evidence gate: its operator is the
capsule's, the capsule is admissible, and every contract rests on evidence the policy accepts. -/
def acceptable (policy : KernelPolicy) (kernel : PlannedKernel) : Bool :=
  kernel.op == kernel.capsule.op &&
    kernel.capsule.admissible policy &&
    match ({ kernels := #[kernel] } : KernelPlan).checkContracts policy.assurance with
    | .accepted => true
    | .rejected _ => false

/-- Explain every failed part of the complete kernel acceptance gate. -/
def acceptanceFailures (policy : KernelPolicy) (kernel : PlannedKernel) :
    Array AcceptanceFailure := Id.run do
  let mut failures := #[]
  unless kernel.op == kernel.capsule.op do
    failures := failures.push (.operationMismatch kernel.op kernel.capsule.op)
  unless kernel.capsule.supportsForward do
    failures := failures.push (.forwardUnsupported kernel.op kernel.capsule.name)
  unless kernel.capsule.contractsAligned do
    failures := failures.push (.contractsMisaligned kernel.op kernel.capsule.name)
  unless kernel.capsule.allowedBy policy do
    failures := failures.push
      (.trustRejected kernel.op kernel.capsule.name kernel.capsule.trustLevel)
  unless kernel.capsule.matchesPreference policy do
    failures := failures.push
      (.providerRejected kernel.op kernel.capsule.name kernel.capsule.provider)
  unless kernel.capsule.matchesDevice policy do
    failures := failures.push
      (.deviceMismatch kernel.op kernel.capsule.name policy.device kernel.capsule.device)
  unless kernel.capsule.matchesVJP policy do
    failures := failures.push
      (.vjpMismatch kernel.op kernel.capsule.name policy.vjpMode kernel.capsule.vjpMode)
  for report in
      ({ kernels := #[kernel] } : KernelPlan).rejectedReports policy.assurance do
    failures := failures.push (.evidenceRejected report)
  return failures

end PlannedKernel

namespace GroupedKernelPlan

/-- Explain every kernel that fails the complete policy and evidence gate. -/
def acceptanceFailures (policy : KernelPolicy) (plan : GroupedKernelPlan) :
    Array AcceptanceFailure :=
  plan.toKernelPlan.kernels.flatMap (PlannedKernel.acceptanceFailures policy)

/-- Whether every grouped kernel satisfies the complete policy and evidence gate. -/
def acceptable (policy : KernelPolicy) (plan : GroupedKernelPlan) : Bool :=
  plan.toKernelPlan.kernels.all (PlannedKernel.acceptable policy)

end GroupedKernelPlan

/-- One planned operation whose capsule has passed the complete policy and evidence gate. -/
structure AcceptedKernel where
  /-- The operator the kernel implements, attributes included. -/
  op : Op.Wire
  /-- The capsule that carries the kernel's contract claims and evidence. -/
  capsule : KernelCapsule
  /-- The policy the gate checked the capsule against. -/
  policy : KernelPolicy
  /-- Evidence that the complete policy and evidence gate accepted this kernel. -/
  accepted : ({ op, capsule } : PlannedKernel).acceptable policy = true

instance : Repr AcceptedKernel where
  reprPrec k _ := Std.Format.text s!"AcceptedKernel({k.op.tag}, {k.capsule.name})"

/-- Check a planned kernel, returning an `AcceptedKernel` or every failure of the gate. -/
def PlannedKernel.accept (policy : KernelPolicy) (kernel : PlannedKernel) :
    Except (Array AcceptanceFailure) AcceptedKernel :=
  if h : kernel.acceptable policy then
    .ok ⟨kernel.op, kernel.capsule, policy, h⟩
  else
    .error (kernel.acceptanceFailures policy)

/-- `kernel.accept policy = .ok k` exactly when `kernel.acceptable policy = true` and `k` has
`kernel`'s operator and capsule and the policy `policy`. -/
theorem PlannedKernel.accept_eq_ok_iff {policy : KernelPolicy} {kernel : PlannedKernel}
    {k : AcceptedKernel} :
    kernel.accept policy = .ok k ↔
      kernel.acceptable policy = true ∧ k.op = kernel.op ∧ k.capsule = kernel.capsule ∧
        k.policy = policy := by
  unfold PlannedKernel.accept
  split
  · next h =>
    constructor
    · intro hk
      cases hk
      exact ⟨h, rfl, rfl, rfl⟩
    · rintro ⟨-, hop, hcapsule, hpolicy⟩
      cases k
      subst hop hcapsule hpolicy
      rfl
  · next h =>
    simp [h]

/-- A graph kernel plan after grouping and contract checking. -/
structure AcceptedGraphKernelPlan where
  private mk ::
  /-- The node-level plan whose selected contracts the gate checked. -/
  graphPlan : GraphKernelPlan
  /-- Audit groups derived from the node-level plan. -/
  groupedPlan : GroupedKernelPlan
  /-- Grouping cannot substitute a different plan's contract evidence. -/
  grouped_eq : groupedPlan = graphPlan.toCoalescedGroups
  /-- The policy applied to every audit group. -/
  policy : KernelPolicy
  /-- Every group passes the complete policy and evidence gate. -/
  accepted : groupedPlan.acceptable policy = true

namespace AcceptedGraphKernelPlan

/-- Construct an accepted graph plan after every grouped kernel passes the execution gate. -/
@[no_expose] def create (graphPlan : GraphKernelPlan) (policy : KernelPolicy)
    (accepted : graphPlan.toCoalescedGroups.acceptable policy = true) :
    AcceptedGraphKernelPlan :=
  ⟨graphPlan, graphPlan.toCoalescedGroups, rfl, policy, accepted⟩

end AcceptedGraphKernelPlan

instance : Repr AcceptedGraphKernelPlan where
  reprPrec p _ := Std.Format.text s!"AcceptedGraphKernelPlan({repr p.groupedPlan})"

namespace AcceptedGraphKernelPlan

/-- Source IR node ids covered by the accepted grouped plan. -/
def nodeIds (p : AcceptedGraphKernelPlan) : Array Nat :=
  p.groupedPlan.nodeIds

/-- Selected capsule names in accepted group order. -/
def capsuleNames (p : AcceptedGraphKernelPlan) : Array String :=
  p.groupedPlan.capsuleNames

/-- The kernel plan of the accepted groups, one kernel per group. -/
def audit (p : AcceptedGraphKernelPlan) : KernelPlan :=
  p.groupedPlan.toKernelPlan

end AcceptedGraphKernelPlan

/-- Result of planning, grouping, and checking a graph. -/
inductive GraphKernelPlanResult where
  | accepted (plan : AcceptedGraphKernelPlan)
  | rejected (groupedPlan : GroupedKernelPlan) (failures : Array AcceptanceFailure)
  deriving Repr

/--
Derive, check, and expose a grouped graph plan only when every selected kernel passes.

Deriving the groups here keeps the accepted audit tied to the exact graph plan that produced it.
-/
def GraphKernelPlan.accept (graphPlan : GraphKernelPlan)
    (policy : KernelPolicy) : GraphKernelPlanResult :=
  let groupedPlan := graphPlan.toCoalescedGroups
  if h : groupedPlan.acceptable policy then
    .accepted (AcceptedGraphKernelPlan.create graphPlan policy h)
  else
    .rejected groupedPlan (groupedPlan.acceptanceFailures policy)

/-- `GraphKernelPlan.accept graphPlan policy = .accepted p` exactly when
`graphPlan.toCoalescedGroups.acceptable policy = true` and `p` has the graph plan `graphPlan` and
the policy `policy`. -/
theorem GraphKernelPlan.accept_eq_accepted_iff {graphPlan : GraphKernelPlan}
    {policy : KernelPolicy} {p : AcceptedGraphKernelPlan} :
    GraphKernelPlan.accept graphPlan policy = .accepted p ↔
      graphPlan.toCoalescedGroups.acceptable policy = true ∧ p.graphPlan = graphPlan ∧
        p.policy = policy := by
  unfold GraphKernelPlan.accept
  dsimp only
  split
  · next h =>
    constructor
    · intro hp
      cases hp
      exact ⟨h, rfl, rfl⟩
    · rintro ⟨-, hgraph, hpolicy⟩
      obtain ⟨graphPlan', groupedPlan, hgrouped, policy', accepted⟩ := p
      subst hgraph hpolicy hgrouped
      rfl
  · next h =>
    simp [h]

end Backend
end TorchLean
