/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.RuntimeApprox.Graph.NumericalCertificate.Contracts
public import NN.Backend.Profile
public import NN.IR.Semantics

/-!
# Backend-linked graph numerical certificates

Certificate data, backend-plan linkage, generation, checking, and checked execution interfaces.
Most users should import `NN.Proofs.RuntimeApprox.Graph.NumericalCertificate`.
-/

@[expose] public section

namespace Proofs
namespace RuntimeApprox
namespace NumericalCertificate

open NN
open NN.Backend
open NN.IR
open Spec TorchLean
open TorchLean.Floats.IEEE754

/-! ## Backend-linked graph certificates -/

/-- Untrusted certificate data.

The audit field records the data selected by kernel selection. The checker replans the graph under
`profileName` and compares the complete audit, so an artifact cannot choose its own provider,
trust level, or evidence classification.
-/
structure GraphNumericalCertificate where
  profileName : String
  registryName : String
  sources : Array SourceRange
  ranges : Array NodeRange
  audit : KernelPlanAudit

instance : Repr GraphNumericalCertificate where
  reprPrec certificate _ := Std.Format.text <|
    s!"GraphNumericalCertificate(profile={certificate.profileName}, " ++
      s!"registry={certificate.registryName}, " ++
      s!"sources={certificate.sources.size}, ranges={certificate.ranges.size}, " ++
      s!"capsules={repr certificate.audit.capsuleNames})"

/-- Result returned after registry replay and backend-plan checking.

The checker reconstructs the range rows and proves that the artifact matches that reconstruction.
This structure does not by itself prove that the rows enclose the graph's real denotation; that
semantic statement is carried separately by `ProvedRealEnclosure`. -/
structure RegistryCheckedCertificate where
  /-- The exact graph whose ranges and kernel plan were reconstructed by the checker. -/
  graph : Graph
  /-- The untrusted artifact supplied to the checker, retained for inspection and serialization. -/
  raw : GraphNumericalCertificate
  /-- Source assumptions whose interval endpoints have been proved finite and ordered. -/
  sources : Array CheckedSourceRange
  /-- The canonical node-by-node range trace reconstructed from the graph. -/
  ranges : Array CheckedNodeRange
  /-- The kernel plan accepted when the checker replanned `graph`. -/
  backendPlan : AcceptedGraphKernelPlan
  /-- Proof that the reconstructed trace matches every range row claimed by `raw`. -/
  rangesMatch : sameRangeTrace ranges raw.ranges = true
  /-- Proof that the accepted plan's audit is the one stored in `raw`. -/
  auditMatch : backendPlan.audit = raw.audit

instance : Repr RegistryCheckedCertificate where
  reprPrec certificate _ := repr certificate.raw

/-- Result of executing the canonical IR with bit-level binary32 semantics and checking every
intermediate value against a registry-replayed range trace. -/
structure RangeCheckedExecution where
  certificate : RegistryCheckedCertificate
  values : Array (Spec.SomeTensor IEEE32Exec)
  withinRanges : executionWithinRanges certificate.ranges values = true

/-- Convert an accepted kernel plan and checked range trace into raw certificate data. -/
def ofCheckedTrace (profile : BackendProfile) (registry : GraphRangeRegistry)
    (sources : Array SourceRange)
    (ranges : Array CheckedNodeRange) (plan : AcceptedGraphKernelPlan) :
    GraphNumericalCertificate :=
  { profileName := profile.name
    registryName := registry.name
    sources
    ranges := eraseRangeTrace ranges
    audit := plan.audit }

/-- Obtain an accepted kernel plan or report the acceptance-gate failures. -/
def acceptedPlan (profile : BackendProfile) (graph : Graph) :
    Except String AcceptedGraphKernelPlan := do
  match <- profile.acceptGraph graph with
  | .accepted plan => pure plan
  | .rejected _ failures =>
      throw (s!"numerical certificate: backend profile {profile.name} rejected the graph: " ++
        s!"{repr failures}")

/-- Generate a canonical certificate using an explicit numerical operation registry. -/
def generateWith (registry : GraphRangeRegistry) (profile : BackendProfile)
    (graph : Graph) (sources : Array SourceRange) :
    Except String GraphNumericalCertificate := do
  let checkedSources <- checkSources sources
  let plan <- acceptedPlan profile graph
  let ranges <- buildRangeTraceWith registry graph checkedSources plan
  pure (ofCheckedTrace profile registry sources ranges plan)

/-- Generate a canonical certificate using TorchLean's built-in numerical contracts. -/
def generate (profile : BackendProfile) (graph : Graph) (sources : Array SourceRange) :
    Except String GraphNumericalCertificate := do
  let registry <- defaultRegistry
  generateWith registry profile graph sources

/-- Check an untrusted certificate with an explicit numerical operation registry. -/
def checkWith (registry : GraphRangeRegistry) (profile : BackendProfile)
    (graph : Graph) (raw : GraphNumericalCertificate) :
    Except String RegistryCheckedCertificate := do
  if raw.profileName != profile.name then
    throw (s!"numerical certificate: profile mismatch; artifact names {raw.profileName}, " ++
      s!"checker uses {profile.name}")
  if raw.registryName != registry.name then
    throw (s!"numerical certificate: registry mismatch; artifact names {raw.registryName}, " ++
      s!"checker uses {registry.name}")
  let sources <- checkSources raw.sources
  let plan <- acceptedPlan profile graph
  let ranges <- buildRangeTraceWith registry graph sources plan
  if hRanges : sameRangeTrace ranges raw.ranges then
    if hAudit : plan.audit = raw.audit then
      pure
        { graph
          raw
          sources
          ranges
          backendPlan := plan
          rangesMatch := hRanges
          auditMatch := hAudit }
    else
      throw "numerical certificate: backend audit differs from the plan selected by the checker"
  else
    throw "numerical certificate: node ranges differ from canonical outward-rounded propagation"

/-- Check an untrusted certificate using TorchLean's built-in numerical contracts. -/
def check (profile : BackendProfile) (graph : Graph) (raw : GraphNumericalCertificate) :
    Except String RegistryCheckedCertificate := do
  let registry <- defaultRegistry
  checkWith registry profile graph raw

/-- Generate and immediately check a certificate. This is convenient for in-process callers and
ensures examples exercise exactly the same checker used for imported artifacts. -/
def generateChecked (profile : BackendProfile) (graph : Graph) (sources : Array SourceRange) :
    Except String RegistryCheckedCertificate := do
  let raw <- generate profile graph sources
  check profile graph raw

/-- Generate and immediately check with one explicit registry. -/
def generateCheckedWith (registry : GraphRangeRegistry) (profile : BackendProfile)
    (graph : Graph) (sources : Array SourceRange) : Except String RegistryCheckedCertificate := do
  let raw <- generateWith registry profile graph sources
  checkWith registry profile graph raw

/-- Execute a graph under `IEEE32Exec` and check all intermediate tensors against the certificate.

This reference replay path gives imported runtime artifacts a bit-level oracle. The backend audit
records the capsules and numerical policies
selected when the graph is replanned. The audit is not runtime provenance and does not prove that
those kernels produced the imported values.
-/
def executeIEEE32 (payload : NN.IR.Payload IEEE32Exec)
    (input : Spec.SomeTensor IEEE32Exec) (certificate : RegistryCheckedCertificate) :
    Except String RangeCheckedExecution := do
  let values <- certificate.graph.denoteAll payload input
  if h : executionWithinRanges certificate.ranges values then
    pure { certificate, values, withinRanges := h }
  else
    throw "numerical certificate: IEEE32 graph replay produced a value outside the certified range"

/-- Exact-real execution evidence for the graph stored in a registry-checked certificate.

The numerical checker reconstructs interval transfers, while a semantic proof establishes that the
real graph trace lies in those intervals. Keeping this proof separate prevents successful endpoint
replay from being mistaken for a theorem about an unsupported real operation. -/
structure ProvedRealEnclosure (certificate : RegistryCheckedCertificate) where
  /-- Real-valued constants and external tensors used by the graph execution. -/
  payload : NN.IR.Payload Real
  /-- Real-valued graph input. -/
  input : Spec.SomeTensor Real
  /-- Complete real-valued node trace, in graph order. -/
  values : Array (Spec.SomeTensor Real)
  /-- Evidence that `values` is exactly the graph's denotational execution trace. -/
  denotation : certificate.graph.denoteAll payload input = .ok values
  /-- Pointwise evidence that every real node value lies in its checked interval. -/
  enclosed : ArraysRelated SomeTensorEnclosed certificate.ranges values

namespace RangeCheckedExecution

/-- Pair a checked IEEE replay with a proved real enclosure trace to obtain a graph-wide,
pointwise error trace. Each node's error budget is the width of its checked outward interval. -/
theorem error_trace (execution : RangeCheckedExecution)
    (exact : ProvedRealEnclosure execution.certificate) :
    ExecutionErrorTrace execution.certificate.ranges exact.values execution.values :=
  execution_error_trace_of_check exact.enclosed execution.withinRanges

end RangeCheckedExecution

end NumericalCertificate
end RuntimeApprox
end Proofs
