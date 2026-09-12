/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Backend.Grouping

/-!
# Backend Contract Checks

The planner admits a capsule only when its operation, device, provider, gradient mode, trust level,
and contract claims match the kernel policy. The contract audit then looks inside the selected
capsules: each of the four descriptors must rest on evidence the assurance policy accepts.

Acceptance repeats both checks at the execution boundary. This matters because `PlannedKernel` is
a public diagnostic record and can therefore be assembled directly by a caller. Accepted values
carry the successful gate as a proof field, and graph acceptance derives its grouped audit plan
from the graph plan instead of accepting unrelated groups from the caller.
-/

@[expose] public section

namespace NN
namespace Backend

/-- One contract obligation of one selected backend kernel. -/
structure ObligationReport where
  op : BackendOp
  capsuleName : String
  obligation : ContractObligation
  claim : ContractClaim
  evidence : ContractEvidence
  deriving Repr

/-- Result of checking a plan audit against an assurance policy. -/
inductive ContractCheck where
  | accepted
  | rejected (reports : Array ObligationReport)
  deriving Repr

namespace KernelAudit

/-- All contract obligations of a selected kernel. -/
def obligationReports (a : KernelAudit) : Array ObligationReport :=
  a.contracts.map fun (obligation, descriptor) =>
    { op := a.op
      capsuleName := a.capsuleName
      obligation
      claim := descriptor.claim
      evidence := descriptor.evidence }

end KernelAudit

namespace KernelPlanAudit

/-- All contract obligations of all selected kernels. -/
def obligationReports (a : KernelPlanAudit) : Array ObligationReport :=
  a.kernels.flatMap KernelAudit.obligationReports

/-- Obligations whose evidence the policy does not accept. -/
def rejectedReports (policy : AssurancePolicy) (a : KernelPlanAudit) : Array ObligationReport :=
  a.obligationReports.filter fun r => !policy.acceptsEvidence r.evidence

/-- Check every obligation of a plan audit against an assurance policy. -/
def checkContracts (policy : AssurancePolicy) (a : KernelPlanAudit) : ContractCheck :=
  let rejected := a.rejectedReports policy
  if rejected.isEmpty then .accepted else .rejected rejected

end KernelPlanAudit

namespace KernelPlan

/-- Check every obligation of a selected plan against an assurance policy. -/
def checkContracts (policy : AssurancePolicy) (p : KernelPlan) : ContractCheck :=
  p.audit.checkContracts policy

end KernelPlan

namespace GroupedKernelPlan

/-- Check every obligation of a grouped graph plan against an assurance policy. -/
def checkContracts (policy : AssurancePolicy) (p : GroupedKernelPlan) : ContractCheck :=
  p.toKernelPlan.checkContracts policy

end GroupedKernelPlan

/-- One reason a planned kernel cannot cross the execution boundary. -/
inductive AcceptanceFailure where
  /-- The planned operation and the capsule's advertised operation differ. -/
  | operationMismatch (planned capsule : BackendOp)
  /-- The capsule has no forward implementation. -/
  | forwardUnsupported (op : BackendOp) (capsuleName : String)
  /-- A contract descriptor states a claim for the wrong obligation. -/
  | contractsMisaligned (op : BackendOp) (capsuleName : String)
  /-- The selected assurance policy does not admit the capsule's trust level. -/
  | trustRejected (op : BackendOp) (capsuleName : String) (trust : TrustLevel)
  /-- The provider policy excludes the capsule. -/
  | providerRejected (op : BackendOp) (capsuleName : String) (provider : Provider)
  /-- The capsule targets a different device. -/
  | deviceMismatch (op : BackendOp) (capsuleName : String)
      (requested actual : Device)
  /-- The capsule cannot satisfy the requested gradient mode. -/
  | vjpMismatch (op : BackendOp) (capsuleName : String)
      (requested actual : VJPMode)
  /-- One contract descriptor rests on evidence rejected by the assurance policy. -/
  | evidenceRejected (report : ObligationReport)
  deriving Repr

namespace PlannedKernel

/-- Whether a directly supplied planned kernel satisfies the complete policy and evidence gate. -/
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
      ({ kernels := #[kernel] } : KernelPlan).audit.rejectedReports policy.assurance do
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
  private mk ::
  /-- The operation the kernel implements. -/
  op : BackendOp
  /-- The capsule that carries the kernel's contract claims and evidence. -/
  capsule : KernelCapsule
  /-- The policy the capsule was checked against. -/
  policy : KernelPolicy
  /-- Evidence that the complete policy and evidence gate accepted this kernel. -/
  accepted : ({ op, capsule } : PlannedKernel).acceptable policy = true

namespace AcceptedKernel.Internal

/-- Construct an accepted kernel after the execution-boundary check succeeds. -/
opaque create (op : BackendOp) (capsule : KernelCapsule) (policy : KernelPolicy)
    (accepted : ({ op, capsule } : PlannedKernel).acceptable policy = true) :
    AcceptedKernel :=
  ⟨op, capsule, policy, accepted⟩

end AcceptedKernel.Internal

instance : Repr AcceptedKernel where
  reprPrec k _ := Std.Format.text s!"AcceptedKernel({k.op.name}, {k.capsule.name})"

/-- Check a planned kernel and return a value that an executor can consume only on success. -/
def PlannedKernel.accept (policy : KernelPolicy) (kernel : PlannedKernel) :
    Except (Array AcceptanceFailure) AcceptedKernel :=
  if h : kernel.acceptable policy then
    .ok (AcceptedKernel.Internal.create kernel.op kernel.capsule policy h)
  else
    .error (kernel.acceptanceFailures policy)

/-- A graph kernel plan after grouping and contract checking. -/
structure AcceptedGraphKernelPlan where
  private mk ::
  /-- The node-level plan whose selected contracts were checked. -/
  graphPlan : IR.GraphKernelPlan
  /-- Audit groups derived from the node-level plan. -/
  groupedPlan : GroupedKernelPlan
  /-- Grouping cannot substitute a different plan's contract evidence. -/
  grouped_eq : groupedPlan = graphPlan.toCoalescedGroups
  /-- The policy applied to every audit group. -/
  policy : KernelPolicy
  /-- Every group passes the complete policy and evidence gate. -/
  accepted : groupedPlan.acceptable policy = true

namespace AcceptedGraphKernelPlan.Internal

/-- Construct an accepted graph plan after every grouped kernel passes the execution gate. -/
opaque create (graphPlan : IR.GraphKernelPlan) (policy : KernelPolicy)
    (accepted : graphPlan.toCoalescedGroups.acceptable policy = true) :
    AcceptedGraphKernelPlan :=
  ⟨graphPlan, graphPlan.toCoalescedGroups, rfl, policy, accepted⟩

end AcceptedGraphKernelPlan.Internal

instance : Repr AcceptedGraphKernelPlan where
  reprPrec p _ := Std.Format.text s!"AcceptedGraphKernelPlan({repr p.groupedPlan})"

namespace AcceptedGraphKernelPlan

/-- Source IR node ids covered by the accepted grouped plan. -/
def nodeIds (p : AcceptedGraphKernelPlan) : Array Nat :=
  p.groupedPlan.nodeIds

/-- Selected capsule names in accepted group order. -/
def capsuleNames (p : AcceptedGraphKernelPlan) : Array String :=
  p.groupedPlan.capsuleNames

/-- Audit for the accepted grouped plan. -/
def audit (p : AcceptedGraphKernelPlan) : KernelPlanAudit :=
  p.groupedPlan.audit

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
def acceptGraphKernelPlan (graphPlan : IR.GraphKernelPlan)
    (policy : KernelPolicy) : GraphKernelPlanResult :=
  let groupedPlan := graphPlan.toCoalescedGroups
  if h : groupedPlan.acceptable policy then
    .accepted (AcceptedGraphKernelPlan.Internal.create graphPlan policy h)
  else
    .rejected groupedPlan (groupedPlan.acceptanceFailures policy)

end Backend
end NN
