/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Backend.Profile

/-!
# Kernel Selection Reports

Human-readable reports for contract-carrying kernel plans. The eager runtime prints the detailed
lines for each capsule the first time it is used when `showBackend` is set, and the runtime API
exposes `BackendProfile.planReport` for command-line inspection.
-/

@[expose] public section

namespace NN
namespace Backend

namespace Provider

/-- Short stable spelling for a backend provider. -/
def label : Provider → String
  | .reference => "reference"
  | .torchLean => "torchlean"
  | .nativeCuda => "native-cuda"
  | .libTorch => "libtorch"
  | .aten => "aten"
  | .mps => "mps"
  | .webGpu => "webgpu"
  | .cuBLAS => "cublas"
  | .cuDNN => "cudnn"
  | .cuFFT => "cufft"
  | .xla => "xla"
  | .neuron => "neuron"
  | .customChip => "custom-chip"
  | .external => "external"

end Provider

namespace TrustLevel

/-- Short stable spelling for a capsule trust level. -/
def label : TrustLevel → String
  | .checked => "checked"
  | .trustedExternal => "trusted-external"

end TrustLevel

namespace AssurancePolicy

/-- Short stable spelling for a profile assurance policy. -/
def label (policy : AssurancePolicy) : String :=
  if policy.allowTrustedExternal then "external" else "checked"

end AssurancePolicy

namespace VJPMode

/-- Short stable spelling for how a capsule handles backward/VJP. -/
def label : VJPMode → String
  | .none => "none"
  | .torchLeanTape => "torchlean-tape"
  | .backendVJP => "backend-vjp"

end VJPMode

namespace TensorLayout

/-- Short stable spelling for a tensor-layout contract. -/
def label : TensorLayout → String
  | .canonicalTensor => "canonical-tensor"
  | .flatRowMajor => "flat-row-major"
  | .libTorchCudaView => "libtorch-cuda-view"

end TensorLayout

namespace ContractObligation

/-- Field name of a capsule contract, as shown in reports. -/
def label : ContractObligation → String
  | .shape => "shape"
  | .layout => "layout"
  | .value => "value"
  | .vjp => "vjp"

end ContractObligation

namespace ContractClaim

/-- Human-readable statement of a structured backend obligation. -/
def label : ContractClaim → String
  | .shapeSafety op => s!"shape safety for {op.name}"
  | .layoutCompatibility op layout =>
      s!"{layout.label} layout compatibility for {op.name}"
  | .valueRefinement op => s!"{op.name} forward refines its TorchLean semantics"
  | .vjpRefinement op mode =>
      s!"{op.name} {mode.label} VJP refines its TorchLean semantics"
  | .vjpUnavailable op => s!"{op.name} has no VJP in this capsule"

end ContractClaim

namespace ContractEvidence

/-- Concise description of the evidence attached to a contract claim. -/
def label : ContractEvidence → String
  | .runtimeGuard name => s!"guarded at runtime by {name}"
  | .testSuite name => s!"covered by test suite {name}"
  | .trustedBoundary reason => s!"trusted boundary: {reason}"
  | .notApplicable => "not applicable"

end ContractEvidence

namespace AcceptanceFailure

/-- Human-readable explanation of why a planned kernel failed the execution-boundary check. -/
def message : AcceptanceFailure → String
  | .operationMismatch planned capsule =>
      s!"planned operation {planned.name} does not match capsule operation {capsule.name}"
  | .forwardUnsupported op capsuleName =>
      s!"{capsuleName} has no forward implementation for {op.name}"
  | .contractsMisaligned op capsuleName =>
      s!"{capsuleName} has misaligned contract claims for {op.name}"
  | .trustRejected op capsuleName trust =>
      s!"{capsuleName} uses rejected trust level {trust.label} for {op.name}"
  | .providerRejected op capsuleName provider =>
      s!"{capsuleName} uses rejected provider {provider.label} for {op.name}"
  | .deviceMismatch op capsuleName requested actual =>
      s!"{capsuleName} targets {actual.cliName}, not requested {requested.cliName}, for {op.name}"
  | .vjpMismatch op capsuleName requested actual =>
      s!"{capsuleName} offers {actual.label}, not requested {requested.label}, for {op.name}"
  | .evidenceRejected report =>
      s!"{report.capsuleName} has rejected {report.obligation.label} evidence for " ++
        s!"{report.op.name}: {report.evidence.label}"

end AcceptanceFailure

namespace ContractDescriptor

/-- One report line for a named contract field. -/
def reportLine (field : ContractObligation) (d : ContractDescriptor) : String :=
  s!"    {field.label}: {d.claim.label}; {d.evidence.label}"

end ContractDescriptor

namespace KernelAudit

/-- One-line summary for a selected backend capsule. -/
def reportLine (a : KernelAudit) : String :=
  s!"  {a.op.name}: {a.capsuleName} " ++
  s!"provider={a.provider.label} trust={a.trustLevel.label} vjp={a.vjpMode.label} " ++
  s!"reduction={a.numericalPolicy.reduction.label}"

/-- Full contract report for a selected backend capsule. -/
def detailedReportLines (a : KernelAudit) : Array String :=
  #[a.reportLine] ++ a.contracts.map fun (field, descriptor) => descriptor.reportLine field

end KernelAudit

namespace KernelPlanAudit

/-- Human-readable contract details for all selected capsules. -/
def detailedReportLines (a : KernelPlanAudit) : Array String :=
  a.kernels.flatMap KernelAudit.detailedReportLines

end KernelPlanAudit

namespace KernelPlan

/-- Human-readable contract details for a selected kernel plan. -/
def detailedReportLines (p : KernelPlan) : Array String :=
  p.audit.detailedReportLines

end KernelPlan

namespace BackendProfile

/-- One-line profile description for logs and interactive choosers. -/
def summary (p : BackendProfile) : String :=
  s!"profile={p.name} device={p.policy.device.cliName} assurance={p.policy.assurance.label} " ++
  s!"vjp={p.policy.vjpMode.label}"

/-- Plan a list of backend ops and format the selected capsules. -/
def planReport (p : BackendProfile) (ops : Array BackendOp) : Except String String := do
  let plan ← p.planOps ops
  let boundary :=
    if plan.hasTrustedExternal then
      "trusted external boundary: " ++ String.intercalate ", " plan.trustedExternalOps.toList
    else
      "trusted external boundary: none"
  pure <| String.intercalate "\n" <|
    (#[p.summary, boundary] ++ plan.detailedReportLines).toList

end BackendProfile

end Backend
end NN
