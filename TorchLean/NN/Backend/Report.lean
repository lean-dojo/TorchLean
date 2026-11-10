/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Backend.Profile

/-!
# Kernel Selection Reports

Human-readable reports for contract-carrying kernel plans. Each provider, trust level, assurance
policy, VJP mode, layout, contract field, claim and evidence kind has a short `label`.
`PlannedKernel.detailedReportLines` gives the lines of one capsule with its four contracts, and
`KernelPlan.detailedReportLines` concatenates them over the kernels of a plan. The eager runtime
prints the lines of the first accepted selection of each capsule name when `showBackend` is set,
and the runtime API exposes `Profile.planReport` for command-line inspection.
-/

@[expose] public section

namespace TorchLean
namespace Backend

namespace Provider

/-- Short stable spelling for a backend provider. -/
def label : Provider → String
  | .reference => "reference"
  | .torchLean => "torchlean"
  | .libTorch => "libtorch"

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
  | .libTorchCUDAView => "libtorch-cuda-view"

end TensorLayout

namespace Contract.Obligation

/-- Field name of a capsule contract, as shown in reports. -/
def label : Obligation → String
  | .shape => "shape"
  | .layout => "layout"
  | .value => "value"
  | .vjp => "vjp"

end Contract.Obligation

namespace Contract.Claim

/-- Human-readable statement of a structured backend obligation. -/
def label : Claim → String
  | .shapeSafety op => s!"shape safety for {op.tag}"
  | .layoutCompatibility op layout =>
      s!"{layout.label} layout compatibility for {op.tag}"
  | .valueRefinement op => s!"{op.tag} forward refines its TorchLean semantics"
  | .vjpRefinement op mode =>
      s!"{op.tag} {mode.label} VJP refines its TorchLean semantics"
  | .vjpUnavailable op => s!"{op.tag} has no VJP in this capsule"

end Contract.Claim

namespace Contract.Evidence

/-- Concise description of the evidence attached to a contract claim. -/
def label : Evidence → String
  | .sharedDefinition [name] => s!"uses canonical definition {name} directly"
  | .sharedDefinition names =>
    s!"uses canonical definitions {", ".intercalate (names.map toString)} directly"
  | .runtimeGuard name => s!"guarded at runtime by {name}"
  | .testSuite name => s!"covered by test suite {name}"
  | .trustedBoundary reason => s!"trusted boundary: {reason}"
  | .notApplicable => "not applicable"

end Contract.Evidence

namespace AcceptanceFailure

/-- Human-readable explanation of why a planned kernel failed the execution-boundary check. -/
def message : AcceptanceFailure → String
  | .operationMismatch planned capsule =>
      s!"planned operation {planned.tag} does not match capsule operation {capsule.tag}"
  | .forwardUnsupported op capsuleName =>
      s!"{capsuleName} has no forward implementation for {op.tag}"
  | .contractsMisaligned op capsuleName =>
      s!"{capsuleName} has misaligned contract claims for {op.tag}"
  | .trustRejected op capsuleName trust =>
      s!"{capsuleName} uses rejected trust level {trust.label} for {op.tag}"
  | .providerRejected op capsuleName provider =>
      s!"{capsuleName} uses rejected provider {provider.label} for {op.tag}"
  | .deviceMismatch op capsuleName requested actual =>
      s!"{capsuleName} targets {actual.cliName}, not requested {requested.cliName}, for {op.tag}"
  | .vjpMismatch op capsuleName requested actual =>
      s!"{capsuleName} offers {actual.label}, not requested {requested.label}, for {op.tag}"
  | .evidenceRejected report =>
      s!"{report.capsuleName} has rejected {report.obligation.label} evidence for " ++
        s!"{report.op.tag}: {report.evidence.label}"

end AcceptanceFailure

namespace Contract.Descriptor

/-- One report line for a named contract field, with the claim's qualification if it has one. -/
def reportLine (field : Obligation) (d : Descriptor) : String :=
  let qualification := if d.qualification.isEmpty then "" else s!" ({d.qualification})"
  s!"    {field.label}: {d.claim.label}{qualification}; {d.evidence.label}"

end Contract.Descriptor

namespace PlannedKernel

/-- One line for a selected capsule: operator tag, capsule name, provider, trust level, VJP mode
and reduction policy. -/
def reportLine (k : PlannedKernel) : String :=
  s!"  {k.op.tag}: {k.capsule.name} " ++
  s!"provider={k.capsule.provider.label} trust={k.capsule.trustLevel.label} " ++
  s!"vjp={k.capsule.vjpMode.label} reduction={k.capsule.numericalPolicy.reduction.label}"

/-- The summary line of a selected capsule followed by one line per contract field. -/
def detailedReportLines (k : PlannedKernel) : Array String :=
  #[k.reportLine] ++ k.contracts.map fun (field, descriptor) => descriptor.reportLine field

end PlannedKernel

namespace KernelPlan

/-- Human-readable contract details for a selected kernel plan. -/
def detailedReportLines (p : KernelPlan) : Array String :=
  p.kernels.flatMap PlannedKernel.detailedReportLines

end KernelPlan

namespace Profile

/-- One-line profile description for logs and interactive choosers. -/
def summary (p : Profile) : String :=
  s!"profile={p.name} device={p.policy.device.cliName} assurance={p.policy.assurance.label} " ++
  s!"vjp={p.policy.vjpMode.label}"

/-- Plan registered operators under the profile and report the result: the profile summary, the
operators whose capsules are `trustedExternal`, then the detailed lines of every capsule.

The `trustedExternal` list does not enumerate every foreign implementation boundary: LibTorch
capsules with recorded comparisons are classified as `checked` and still execute native code. -/
def planReport (p : Profile) (ops : Array (Σ ins outs, Op ins outs)) :
    Except String String := do
  let plan ← p.planOps ops
  let boundary :=
    if plan.hasTrustedExternal then
      "trusted-external capsules: " ++ String.intercalate ", " plan.trustedExternalOps.toList
    else
      "trusted-external capsules: none"
  pure <| String.intercalate "\n" <|
    (#[p.summary, boundary] ++ plan.detailedReportLines).toList

end Profile

end Backend
end TorchLean
