/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.NumericalCertificate.Contracts
public import TorchLean.NN.Backend.Graph

/-!
# Backend-linked graph numerical certificates

Certificate data for a graph, its link to the backend kernel plan, and the functions that generate,
check and execute certificates. A raw certificate is untrusted: `check` rebuilds its range rows and
kernel-plan audit from the graph and accepts it only when both match.
-/

@[expose] public section

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)

namespace TorchLean
namespace NumericalCertificate

open Backend
open IR
/-! ## Backend-linked graph certificates -/

/-- Untrusted certificate data.

The `audit` field records what kernel selection chose. The checker replans the graph under
`profileName` and compares the complete audit, so an artifact cannot choose its own provider,
trust level or evidence classification.
-/
structure _root_.TorchLean.NumericalCertificate where
  /-- Backend profile the artifact claims to have been planned under. -/
  profileName : String
  /-- Interval assumptions on the graph's external sources. -/
  sources : Array SourceRange
  /-- Claimed range rows, one per node, which the checker replays and compares. -/
  ranges : Array NodeRange
  /-- Claimed kernel-plan audit, which the checker compares with the plan it rebuilds. -/
  audit : KernelPlan

/-- A one-line summary: the profile name, the numbers of source and range rows, and the capsule
names of the audit. -/
instance instReprNumericalCertificate : Repr NumericalCertificate where
  reprPrec certificate _ := Std.Format.text <|
    s!"NumericalCertificate(profile={certificate.profileName}, " ++
      s!"sources={certificate.sources.size}, ranges={certificate.ranges.size}, " ++
      s!"capsules={repr certificate.audit.capsuleNames})"

/-- Result returned after range-trace replay and backend-plan checking.

The checker reconstructs the range rows and proves that the artifact matches the reconstruction.
That the rows enclose the real denotation of the graph is a separate statement, which
`RealEnclosure` carries. -/
structure Checked where
  /-- The graph whose ranges and kernel plan the checker reconstructed. -/
  graph : Graph
  /-- The untrusted artifact supplied to the checker, retained for inspection and serialization. -/
  raw : NumericalCertificate
  /-- Source assumptions whose interval endpoints have been proved finite and ordered. -/
  sources : Array CheckedSourceRange
  /-- The canonical node-by-node range trace reconstructed from the graph. -/
  ranges : Array CheckedNodeRange
  /-- The kernel plan accepted when the checker replanned `graph`. -/
  backendPlan : AcceptedGraphKernelPlan
  /-- Proof that the reconstructed trace matches every range row claimed by `raw`. -/
  ranges_match : sameRangeTrace ranges raw.ranges = true
  /-- Proof that the accepted plan's audit is the one stored in `raw`. -/
  audit_match : backendPlan.audit = raw.audit

/-- A checked certificate prints as its raw artifact `raw`. -/
instance instReprChecked : Repr Checked where
  reprPrec certificate _ := repr certificate.raw

/-- Result of executing the certificate's graph with bit-level binary32 semantics and checking
every intermediate value against a replayed range trace. -/
structure CheckedExecution where
  /-- The checked certificate whose range trace the replay is checked against. -/
  certificate : Checked
  /-- Bit-level binary32 node values, which `executeIEEE32` obtains by replaying the graph. The
  structure holds no equation between them and the graph's denotation. -/
  values : Array (SomeTensor (ExecFloat.Binary 8 23))
  /-- Proof that every replayed value lies in its checked range row. -/
  withinRanges : withinRanges certificate.ranges values = true

/-- Convert an accepted kernel plan and checked range trace into raw certificate data. -/
def ofCheckedTrace (profile : Profile) (sources : Array SourceRange)
    (ranges : Array CheckedNodeRange) (plan : AcceptedGraphKernelPlan) :
    NumericalCertificate :=
  { profileName := profile.name
    sources
    ranges := eraseRangeTrace ranges
    audit := plan.audit }

/-- Obtain an accepted kernel plan or report the acceptance-gate failures. -/
def acceptedPlan (profile : Profile) (graph : Graph) :
    Except String AcceptedGraphKernelPlan := do
  match <- graph.accept profile with
  | .accepted plan => pure plan
  | .rejected _ failures =>
      throw (s!"numerical certificate: backend profile {profile.name} rejected the graph: " ++
        s!"{repr failures}")

/-- Generate a canonical certificate with the range transfers of the graph's operators. -/
def generate (profile : Profile) (graph : Graph) (sources : Array SourceRange) :
    Except String NumericalCertificate := do
  let checkedSources <- checkSources sources
  let plan <- acceptedPlan profile graph
  let ranges <- buildRangeTrace graph checkedSources plan
  pure (ofCheckedTrace profile sources ranges plan)

/-- Check an untrusted certificate against the range transfers of the graph's operators. -/
def check (profile : Profile) (graph : Graph) (raw : NumericalCertificate) :
    Except String Checked := do
  if raw.profileName != profile.name then
    throw (s!"numerical certificate: profile mismatch; artifact names {raw.profileName}, " ++
      s!"checker uses {profile.name}")
  let sources <- checkSources raw.sources
  let plan <- acceptedPlan profile graph
  let ranges <- buildRangeTrace graph sources plan
  if hRanges : sameRangeTrace ranges raw.ranges then
    if hAudit : plan.audit = raw.audit then
      pure
        { graph
          raw
          sources
          ranges
          backendPlan := plan
          ranges_match := hRanges
          audit_match := hAudit }
    else
      throw "numerical certificate: backend audit differs from the plan selected by the checker"
  else
    throw "numerical certificate: node ranges differ from canonical outward-rounded propagation"

/-- Generate a certificate and check it at once. In-process callers and examples go through the
same checker as imported artifacts. -/
def generateChecked (profile : Profile) (graph : Graph) (sources : Array SourceRange) :
    Except String Checked := do
  let raw <- generate profile graph sources
  check profile graph raw

/-- The value of every node of `graph` on `inputs`, in node order: the tensor of scalars each node
produces at its one output, or an error at a node with another output. -/
@[noinline, nospecialize]
def nodeValues {α : Type} [Storage α] [Context α] [Op.Capabilities α]
    (graph : Graph) (payload : Payload α) (inputs : List (Op.Value α)) :
    Except String (Array (SomeTensor α)) := do
  let outputs <- graph.run payload inputs
  outputs.mapIdxM fun id values =>
    match values with
    | [⟨⟨.value, s⟩, x⟩] => pure ⟨s, x⟩
    | [⟨⟨.frozen, s⟩, x⟩] => pure ⟨s, x⟩
    | [⟨⟨.detached, s⟩, x⟩] => pure ⟨s, x⟩
    | _ => throw s!"numerical certificate: node {id} does not produce one tensor of scalars"

/-- Execute the certificate's graph under `ExecFloat.Binary 8 23` and check every intermediate
tensor against its range row.

The replay is a bit-level reference for imported runtime artifacts. The backend audit lists the
capsules and numerical policies selected when the graph is replanned, so it records no runtime
provenance and cannot show that those kernels produced imported values.
-/
def executeIEEE32 (payload : Payload (ExecFloat.Binary 8 23))
    (inputs : List (Op.Value (ExecFloat.Binary 8 23)))
    (certificate : Checked) : Except String CheckedExecution := do
  let values <- nodeValues certificate.graph payload inputs
  if h : withinRanges certificate.ranges values then
    pure { certificate, values, withinRanges := h }
  else
    throw "numerical certificate: IEEE32 graph replay produced a value outside the certified range"

/-- Exact-real execution evidence for the graph stored in a checked certificate.

The numerical checker reconstructs interval transfers, while a semantic proof establishes that the
real graph trace lies in those intervals. Keeping this proof separate prevents successful endpoint
replay from being mistaken for a theorem about an unsupported real operation. -/
structure RealEnclosure (certificate : Checked) where
  /-- Real-valued constants and custom programs used by the graph execution. -/
  payload : Payload Real
  /-- Real-valued graph inputs. -/
  inputs : List (Op.Value Real)
  /-- Complete real-valued node trace, in graph order. -/
  values : Array (SomeTensor Real)
  /-- Evidence that `values` is exactly the graph's execution trace. -/
  denotation : nodeValues certificate.graph payload inputs = .ok values
  /-- Pointwise evidence that every real node value lies in its checked interval. -/
  enclosed : ArraysRelated SomeTensorEnclosed certificate.ranges values

namespace CheckedExecution

/-- Range-checked binary32 values and a proved real enclosure of the same graph give a pointwise
error trace over the whole graph, with each node's interval width as its error budget. The bound
follows from both values lying in the same interval, so it needs no record of which native
kernels ran and gives none. -/
theorem errorTrace (execution : CheckedExecution)
    (exact : RealEnclosure execution.certificate) :
    ErrorTrace execution.certificate.ranges exact.values execution.values :=
  ErrorTrace.of_check exact.enclosed execution.withinRanges

end CheckedExecution

end NumericalCertificate
end TorchLean
