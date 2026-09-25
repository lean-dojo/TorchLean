/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core.Ops
public import NN.Tensor

/-!
# Backend Profile Tests

Regression checks for contract-carrying backend profiles.

These are policy checks, not numerical kernel tests: they make sure backend planning does not
silently cross a trusted boundary or fall back to an unavailable platform provider.
-/

@[expose] public section

namespace NN.Tests.Backend.Profile

open NN.Backend

def expect (tag : String) (ok : Bool) : IO Unit := do
  unless ok do
    throw <| IO.userError s!"backend profile check failed: {tag}"

def expectCapsules (tag : String) (got expected : Array String) : IO Unit := do
  expect tag (got == expected)

def expectOp (tag : String) (kind : NN.IR.OpKind) (expected : Option BackendOp) : IO Unit := do
  expect tag (NN.Backend.IR.op? kind == expected)

def scalarHardMask : NN.IR.HardMask :=
  { shape := Spec.Shape.scalar, allowed := #[true] }

def expectContains (tag needle haystack : String) : IO Unit := do
  expect tag (haystack.contains needle)

def tinyReluGraph : NN.IR.Graph :=
  { nodes := #[
      { id := 0
        parents := #[]
        kind := .input
        outShape := Spec.Shape.scalar },
      { id := 1
        parents := #[0]
        kind := .relu
        outShape := Spec.Shape.scalar },
      { id := 2
        parents := #[1]
        kind := .relu
        outShape := Spec.Shape.scalar }
    ] }

/-- Test-only external capsule used to check availability gating. -/
def externalReluCapsule : KernelCapsule :=
  { Reference.relu with
    name := "external.relu"
    provider := .external
    device := .external
    trustLevel := .checked }

/-- Availability declaring the external device and provider used by `externalReluCapsule`. -/
def externalAvailability : Availability :=
  { devices := #[.cpu, .external], providers := #[.reference, .torchLean, .external] }

/-- Test-only capsule used to check planner forward-support gating. -/
def disabledForwardReluCapsule : KernelCapsule :=
  { Reference.relu with
    name := "reference.relu_disabled"
    supportsForward := false }

/-- A capsule labelled `checked` whose value contract nevertheless delegates to a trusted boundary.
The planner admits it by label; the contract check must still reject it under the checked policy. -/
def trustedValueReluCapsule : KernelCapsule :=
  { Reference.relu with
    name := "reference.relu_trusted_value"
    valueContract :=
      ContractDescriptor.trusted (.valueRefinement .relu)
        "Test-only value contract resting on a trusted boundary."
        "profile-test external implementation" }

def replacementReluCapsule : KernelCapsule :=
  { Reference.relu with
    name := "replacement.relu" }

def malformedReluCapsule : KernelCapsule :=
  { Reference.relu with
    name := "reference.relu_malformed"
    shapeContract := ContractDescriptor.tested
      (.valueRefinement .relu)
      "Deliberately mismatched descriptor for contract-alignment testing."
      "NN.Tests.Backend.Profile" }

def forwardOnlyReluCapsule : KernelCapsule :=
  { Reference.relu with
    name := "reference.relu_forward_only"
    vjpMode := .none
    vjpContract := ContractDescriptor.vjpUnavailable
      .relu "This test capsule intentionally has no VJP." }

/-- Whether a contract check accepted the plan. -/
def isAccepted : ContractCheck → Bool
  | .accepted => true
  | .rejected _ => false

/-- Test capsule whose declared provider does not match the CPU random executor. -/
def mismatchedRandomCapsule : KernelCapsule :=
  { Reference.randUniform with
    name := "torchlean.rand_uniform_mismatched"
    provider := .torchLean }

def planOrThrow (tag : String) (profile : BackendProfile) (ops : Array BackendOp) :
    IO KernelPlan := do
  match profile.planOps ops with
  | .ok plan => pure plan
  | .error msg => throw <| IO.userError s!"{tag}: planning failed: {msg}"

def profileOrThrow (tag : String) (options : Runtime.Autograd.Torch.Config) :
    IO BackendProfile := do
  let profile ← match options.effectiveBackendProfile with
    | .ok profile => pure profile
    | .error msg => throw <| IO.userError s!"{tag}: profile resolution failed: {msg}"
  unless profile.hasDeviceCapsule do
    throw <| IO.userError s!"{tag}: profile `{profile.name}` has no capsule for its device"
  pure profile

def expectPlanningFails (tag : String) (profile : BackendProfile) (ops : Array BackendOp) :
    IO Unit := do
  match profile.planOps ops with
  | .ok plan =>
      throw <| IO.userError
        s!"{tag}: expected planning to fail, got capsules {plan.capsuleNames}"
  | .error _ => pure ()

def acceptedGraphOrThrow (tag : String) (profile : BackendProfile) (g : NN.IR.Graph) :
    IO AcceptedGraphKernelPlan := do
  match profile.acceptGraph g with
  | .ok (.accepted plan) => pure plan
  | .ok (.rejected _ reports) =>
      throw <| IO.userError
        s!"{tag}: expected accepted graph, got rejected contracts {repr reports}"
  | .error msg =>
      throw <| IO.userError s!"{tag}: graph planning failed: {msg}"

def expectLibTorchBindingAccepts (tag : String)
    (options : Runtime.Autograd.Torch.Config) (op : BackendOp) : IO Unit := do
  let base ← Runtime.Autograd.Torch.Internal.EagerSession.new (α := Float)
  let s := { base with options := options }
  let result ← s.executeSelected op
    #[({ name := "profile-test LibTorch CUDA handler"
         op
         provider := .libTorch
         device := .cuda
         execute := fun _ => pure true } : KernelHandler Bool)]
  expect tag result

/-- A handler mismatch must fail before its implementation can run. -/
def expectBindingRejected (tag needle : String) (handler : KernelHandler Bool) : IO Unit := do
  match Reference.relu.bind handler with
  | .ok _ =>
      throw <| IO.userError s!"{tag}: mismatched handler unexpectedly bound"
  | .error message =>
      expectContains tag needle message

def checkHandlerIdentity : IO Unit := do
  let referenceHandler : KernelHandler Bool := {
    name := "reference ReLU test handler"
    op := .relu
    provider := .reference
    device := .cpu
    execute := fun _ => pure true
  }
  match Reference.relu.bind referenceHandler with
  | .ok executable =>
      expect "matching handler executes" (← executable.run)
  | .error message =>
      throw <| IO.userError s!"matching handler did not bind: {message}"
  expectBindingRejected "handler operation mismatch" "implements `add`"
    { referenceHandler with op := .add }
  expectBindingRejected "handler provider mismatch" "uses provider"
    { referenceHandler with provider := .torchLean }
  expectBindingRejected "handler device mismatch" "targets `cuda`"
    { referenceHandler with device := .cuda }

/-- Eager execution must not run a reference implementation under another provider's capsule. -/
def expectRandomProviderRejected : IO Unit := do
  let mismatchedModule : Registry.CapsuleModule :=
    { name := "reference", capsules := #[mismatchedRandomCapsule] }
  let profile := BackendProfile.checkedCpu.withCapsuleModules #[mismatchedModule]
  let options :=
    Runtime.Autograd.Torch.Config.withBackendProfile
      ({} : Runtime.Autograd.Torch.Config) profile
  let session ← Runtime.Autograd.Torch.Internal.EagerSession.new (α := Float)
    options
  try
    let _ ← Runtime.Autograd.Torch.Internal.EagerSession.randUniform
      (s := session) (sh := Spec.Shape.ofList [2]) 7
    throw <| IO.userError "mismatched random provider unexpectedly executed"
  catch e =>
    expectContains "random provider mismatch is rejected"
      "no matching executable handler is linked" e.toString

def expectRuntimeDeviceRejected (tag : String) (device : NN.Backend.Device) :
    IO Unit := do
  let unavailableProfile : BackendProfile :=
    { BackendProfile.checkedCpu with
      name := s!"unavailable_{device.cliName}"
      policy := { BackendProfile.checkedCpu.policy with device := device } }
  try
    let options :=
      Runtime.Autograd.Torch.Config.withBackendProfile
        ({} : Runtime.Autograd.Torch.Config) unavailableProfile
    let _ ← Runtime.Autograd.Torch.Internal.EagerSession.new (α := Float)
      options
    throw <| IO.userError s!"{tag}: expected runtime device rejection"
  catch e =>
    let msg := toString e
    expectContains tag s!"has no capsule for device `{device.cliName}`" msg

/-- Capsule metadata alone cannot make an unsupported device executable. -/
def expectMetadataOnlyDeviceRejected : IO Unit := do
  let metalCapsule := { Reference.relu with device := Device.metal }
  let profile : BackendProfile :=
    { BackendProfile.checkedCpu with
      name := "metadata_only_metal"
      policy := { BackendProfile.checkedCpu.policy with device := .metal }
      capsuleModules := #[{ name := "metal", capsules := #[metalCapsule] }] }
  let options := Runtime.Autograd.Torch.Config.withBackendProfile {} profile
  match options.validateDevice with
  | .error message => throw <| IO.userError s!"invalid metadata test fixture: {message}"
  | .ok () => pure ()
  let failure ← try
    let _ ← Runtime.Autograd.Torch.Internal.EagerSession.new (α := Float) options
    pure none
  catch error => pure (some error.toString)
  match failure with
  | none => throw <| IO.userError "metadata-only Metal profile unexpectedly executed"
  | some message =>
      expectContains "metadata-only device runtime rejection"
        "no linked TorchLean execution runtime" message

/-- User-facing CUDA sessions must agree with the implementation linked behind the CUDA symbols. -/
def expectCudaSessionMatchesRuntime : IO Unit := do
  let options : Runtime.Autograd.Torch.Config :=
    { device := .cuda }
  match Runtime.Autograd.Cuda.Buffer.runtimeStatus with
  | .nativeAvailable =>
      let _ ← Runtime.Autograd.Torch.Internal.EagerSession.new (α := Float) options
      pure ()
  | .notLinked =>
      try
        let _ ← Runtime.Autograd.Torch.Internal.EagerSession.new (α := Float) options
        throw <| IO.userError "build without LibTorch unexpectedly admitted a user CUDA session"
      catch e =>
        expectContains "no-LibTorch CUDA session rejection" "built without LibTorch" e.toString
  | .nativeUnavailable =>
      try
        let _ ← Runtime.Autograd.Torch.Internal.EagerSession.new (α := Float) options
        throw <| IO.userError "CUDA build without a visible device unexpectedly admitted a session"
      catch e =>
        expectContains "unavailable native CUDA session rejection" "no usable CUDA device"
          e.toString

def run : IO Unit := do
  checkHandlerIdentity
  expect "reference scatter-add records accumulation order"
    (Reference.scatterAdd.numericalPolicy.reduction == .fixedLeft)
  expect "LibTorch scatter-add records implementation-defined accumulation order"
    (LibTorch.scatterAdd.numericalPolicy.reduction == .implementationDefined)
  let typedGraphCpuOpts : Runtime.Autograd.Torch.Config :=
    { execution := .typedGraph
      device := .cpu }
  expect "typed graph CPU options retain the public execution and device choices"
    (typedGraphCpuOpts.execution == .typedGraph && typedGraphCpuOpts.device == .cpu)
  let typedGraphCpuProfile ← profileOrThrow "typed graph CPU profile" typedGraphCpuOpts
  expect "execution mode does not alter CPU capsule selection"
    (typedGraphCpuProfile.policy.device == .cpu)
  expect "default registry contract fields are aligned"
    ((Registry.flatten Registry.maintainedModules).all KernelCapsule.contractsAligned)
  expect "CPU availability excludes every LibTorch CUDA primitive"
    ((Availability.cpu.filterCapsules LibTorch.capsules).isEmpty)
  expect "CUDA availability admits every maintained LibTorch primitive"
    ((Availability.cuda.filterCapsules LibTorch.capsules).size == LibTorch.capsules.size)
  let duplicateModule : Registry.CapsuleModule :=
    { name := "duplicate", capsules := #[Reference.relu] }
  let duplicateModuleProfile : BackendProfile :=
    { BackendProfile.checkedCpu with
      capsuleModules := #[duplicateModule, duplicateModule] }
  expectPlanningFails "duplicate capsule module names are rejected"
    duplicateModuleProfile #[.relu]
  let replacementModule : Registry.CapsuleModule :=
    { name := "reference", capsules := #[replacementReluCapsule] }
  let replacementProfile :=
    BackendProfile.checkedCpu.withCapsuleModules #[replacementModule]
  expect "same-name capsule modules are replaced instead of duplicated"
    ((replacementProfile.capsuleModules.filter
      (fun module => module.name == "reference")).size == 1)
  let replaced ← planOrThrow "replacement capsule module" replacementProfile #[.relu]
  expectCapsules "replacement capsule module is selected" replaced.capsuleNames
    #["replacement.relu"]
  let inferenceOpts : Runtime.Autograd.Torch.Config := { gradEnabled := false }
  let inferenceProfile ← profileOrThrow "no-grad default profile" inferenceOpts
  expect "no-grad runtime planning requests no VJP"
    (inferenceProfile.policy.vjpMode == .none)
  let trainingOpts : Runtime.Autograd.Torch.Config := { gradEnabled := true }
  let trainingProfile ← profileOrThrow "training default profile" trainingOpts
  expect "training runtime planning requests the TorchLean tape"
    (trainingProfile.policy.vjpMode == .torchLeanTape)
  expectOp "IR add maps to exact add capsule" .add (some .add)
  expectOp "IR linear maps to exact linear capsule" .linear (some .linear)
  let convConfig : NN.IR.ConvConfig :=
    { spatialRank := 2
      kernel := [3, 3]
      stride := [1, 1]
      padding := [0, 0]
      channelAxis := 0
      inChannels := 1
      outChannels := 1 }
  let poolConfig : NN.IR.WindowConfig :=
    { spatialRank := 2
      kernel := [2, 2]
      stride := [2, 2]
      padding := [0, 0] }
  expectOp "IR convolution maps to the convolution capability"
    (.conv convConfig) (some .conv)
  expectOp "IR max pooling maps to the max-pool capability"
    (.maxPool poolConfig) (some .maxPool)
  expectOp "IR rand uniform maps to exact forward-only capsule"
    (.randUniform 0) (some .randUniform)
  expectOp "IR permute maps to exact permute capsule"
    (.permute #[1, 0]) (some .permute)
  expectOp "IR hard-masked softmax keeps its exact capsule identity"
    (.hardMaskedSoftmax scalarHardMask) (some .hardMaskedSoftmax)
  expectOp "IR input has no backend capsule" .input none

  expect "cpu availability rejects external device capsule"
    (!(Availability.cpu.admitsCapsule externalReluCapsule))
  expect "external availability admits external capsule when provider is available"
    (externalAvailability.admitsCapsule externalReluCapsule)
  match planOpsAvailable
      { device := .external, provider := .only .external }
      externalAvailability
      #[externalReluCapsule]
      #[.relu] with
  | .ok plan =>
      expectCapsules "external target can plan explicit external capsule" plan.capsuleNames
        #["external.relu"]
  | .error msg =>
      throw <| IO.userError s!"external capsule planning failed: {msg}"
  match planOps { device := .cpu } #[disabledForwardReluCapsule] #[.relu] with
  | .ok plan =>
      throw <| IO.userError
        s!"disabled forward capsule unexpectedly planned as {plan.capsuleNames}"
  | .error _ => pure ()
  match planOps
      { device := .cpu, vjpMode := .none }
      #[Reference.relu] #[.relu] with
  | .ok inferencePlan =>
      expectCapsules "inference accepts a forward capsule that also supports VJP"
        inferencePlan.capsuleNames #["reference.relu"]
  | .error msg =>
      throw <| IO.userError s!"inference VJP compatibility test: planning failed: {msg}"
  match planOps { device := .cpu, vjpMode := .torchLeanTape }
      #[forwardOnlyReluCapsule] #[.relu] with
  | .ok forwardOnly =>
      throw <| IO.userError
        s!"forward-only differentiable capsule unexpectedly planned as {forwardOnly.capsuleNames}"
  | .error _ => pure ()
  match planOps { device := .cpu } #[malformedReluCapsule] #[.relu] with
  | .ok malformed =>
      throw <| IO.userError
        s!"malformed capsule unexpectedly planned as {malformed.capsuleNames}"
  | .error _ => pure ()
  match planOps { device := .cpu } #[trustedValueReluCapsule] #[.relu] with
  | .ok trustedValue =>
      expect "checked policy rejects trusted evidence hidden under a checked label"
        (!isAccepted (trustedValue.checkContracts AssurancePolicy.checked))
      expect "external policy accepts trusted evidence"
        (isAccepted (trustedValue.checkContracts AssurancePolicy.external))
      match trustedValue.kernels[0]? with
      | some planned =>
          match planned.accept BackendProfile.checkedCpu.policy with
          | .ok accepted =>
              throw <| IO.userError
                s!"trusted value contract unexpectedly accepted as {repr accepted}"
          | .error failures =>
              expect "rejected contract report names the value obligation"
                (failures.any fun
                  | .evidenceRejected report => report.obligation == .value
                  | _ => false)
      | none => throw <| IO.userError "trusted value plan has no kernel"
  | .error msg =>
      throw <| IO.userError s!"trusted value contract test: planning failed: {msg}"

  let forgedOperation : PlannedKernel :=
    { op := .sigmoid, capsule := Reference.relu }
  match forgedOperation.accept BackendProfile.checkedCpu.policy with
  | .ok accepted =>
      throw <| IO.userError s!"mismatched operation unexpectedly accepted as {repr accepted}"
  | .error failures =>
      expect "acceptance rejects a capsule relabelled as another operation"
        (failures.any fun
          | .operationMismatch .sigmoid .relu => true
          | _ => false)

  let forgedTrust : PlannedKernel :=
    { op := .relu
      capsule := { Reference.relu with trustLevel := .trustedExternal } }
  match forgedTrust.accept BackendProfile.checkedCpu.policy with
  | .ok accepted =>
      throw <| IO.userError s!"relabelled trust boundary unexpectedly accepted as {repr accepted}"
  | .error failures =>
      expect "acceptance rechecks capsule trust instead of trusting the planned record"
        (failures.any fun
          | .trustRejected .relu _ .trustedExternal => true
          | _ => false)

  let acceptedCpuGraph ← acceptedGraphOrThrow "checked cpu graph acceptance"
    BackendProfile.checkedCpu tinyReluGraph
  expectCapsules "checked cpu graph coalesces same relu capsule"
    acceptedCpuGraph.capsuleNames #["reference.relu"]
  expectCapsules "checked cpu graph keeps source node ids"
    (acceptedCpuGraph.nodeIds.map (fun n => toString n)) #["1", "2"]
  expect "checked cpu graph has no trusted external" (!acceptedCpuGraph.audit.hasTrustedExternal)
  expect "checked cpu graph audit has one row per group"
    (acceptedCpuGraph.audit.kernels.size == 1)

  -- Coalescing must not hide a second node's evidence behind the first node's identity.
  let changedCapsules := #[
    { Reference.relu with trustLevel := .trustedExternal },
    { trustedValueReluCapsule with name := Reference.relu.name },
    { malformedReluCapsule with name := Reference.relu.name },
    { Reference.relu with supportsForward := false }]
  for changed in changedCapsules do
    let plan : NN.Backend.IR.GraphKernelPlan :=
      { kernels := #[
          { nodeId := 1, kind := .relu, op := .relu, capsule := Reference.relu },
          { nodeId := 2, kind := .relu, op := .relu, capsule := changed }] }
    expect "different contracts retain separate audit groups"
      (plan.toCoalescedGroups.groups.size == 2)
    match acceptGraphKernelPlan plan BackendProfile.checkedCpu.policy with
    | .accepted _ =>
        throw <| IO.userError "coalescing hid a rejected capsule behind an accepted one"
    | .rejected _ failures =>
        expect "rejected grouped plan explains the failing contract" (!failures.isEmpty)

  let changedReduction : NN.Backend.IR.GraphKernelPlan :=
    { kernels := #[
        { nodeId := 1, kind := .relu, op := .relu, capsule := Reference.relu },
        { nodeId := 2, kind := .relu, op := .relu
          capsule := { Reference.relu with
            numericalPolicy := { reduction := .implementationDefined } } }] }
  expect "different numerical policies retain separate audit groups"
    (changedReduction.toCoalescedGroups.groups.size == 2)

  let exactOps :=
    #[ BackendOp.matmul, .linear, .mseLoss, .add, .sub, .mul, .scale, .abs, .sqrt
    , .clamp, .max, .min, .relu, .gelu, .sigmoid, .tanh
    , .softmax, .hardMaskedSoftmax, .softplus, .exp, .log, .inv, .safeLog, .logSoftmax
    , .sin, .cos, .reduceSum
    , .reduceMean, .randUniform, .bernoulliMask, .reshape, .permute, .broadcast
    , .concat, .slice, .gather, .scatterAdd, .layerNorm, .batchNorm, .conv
    , .convTranspose, .maxPool, .smoothMaxPool, .avgPool ]

  let cudaExactOps := exactOps ++ #[.fftFno, .selectiveScan]
  let exactReferenceCapsules := exactOps.map fun op => s!"reference.{op.name}"
  let exactLibTorchCapsules := cudaExactOps.map fun op => s!"libtorch.{op.name}"
  let profileOps :=
    #[ BackendOp.matmul, .relu, .softmax, .hardMaskedSoftmax, .layerNorm, .batchNorm
    , .conv, .convTranspose, .maxPool, .smoothMaxPool, .avgPool, .mseLoss
    , .scaledDotProductAttention ]

  let cpu ← planOrThrow "checked cpu" BackendProfile.checkedCpu profileOps
  expectCapsules "checked cpu capsule order" cpu.capsuleNames
    #[ "reference.matmul"
    , "reference.relu"
    , "reference.softmax"
    , "reference.hard_masked_softmax"
    , "reference.layer_norm"
    , "reference.batch_norm"
    , "reference.conv"
    , "reference.conv_transpose"
    , "reference.max_pool"
    , "reference.smooth_max_pool"
    , "reference.avg_pool"
    , "reference.mse_loss"
    , "reference.attention"
    ]
  expect "checked cpu has no trusted external" (!cpu.hasTrustedExternal)

  let cpuExact ← planOrThrow "checked cpu exact ops" BackendProfile.checkedCpu exactOps
  expectCapsules "checked cpu exact capsules" cpuExact.capsuleNames exactReferenceCapsules

  let replacementModule : Registry.CapsuleModule :=
    { name := "profile-test-replacement", capsules := #[replacementReluCapsule] }
  let extendedCpu := BackendProfile.checkedCpu.withCapsuleModules #[replacementModule]
  let extendedPlan ← planOrThrow "extended capsule modules" extendedCpu #[.relu, .matmul]
  expectCapsules "extended modules preserve model-independent preference" extendedPlan.capsuleNames
    #["replacement.relu", "reference.matmul"]

  let reportOps := exactOps ++ #[.scaledDotProductAttention]
  match BackendProfile.checkedCpu.planReport reportOps with
  | .ok report =>
      expectContains "checked cpu report names exact add" "add: reference.add" report
      expectContains "checked cpu report names exact reshape" "reshape: reference.reshape" report
      expectContains "checked cpu report names exact batchnorm"
        "batch_norm: reference.batch_norm" report
      expectContains "checked cpu report names exact smooth max pool"
        "smooth_max_pool: reference.smooth_max_pool" report
      expectContains "checked cpu report names exact attention"
        "scaled_dot_product_attention: reference.attention" report
  | .error msg =>
      throw <| IO.userError s!"checked cpu report failed: {msg}"

  let cuda ← planOrThrow "checked cuda" BackendProfile.checkedCuda profileOps
  expectCapsules "checked cuda capsule order" cuda.capsuleNames
    #[ "libtorch.matmul"
    , "libtorch.relu"
    , "libtorch.softmax"
    , "libtorch.hard_masked_softmax"
    , "libtorch.layer_norm"
    , "libtorch.batch_norm"
    , "libtorch.conv"
    , "libtorch.conv_transpose"
    , "libtorch.max_pool"
    , "libtorch.smooth_max_pool"
    , "libtorch.avg_pool"
    , "libtorch.mse_loss"
    , "libtorch.direct_attention"
    ]
  expect "checked CUDA capsules retain the maintained evidence classification"
    (!cuda.hasTrustedExternal)
  expect "checked CUDA contracts are accepted without trusted-boundary evidence"
    (isAccepted (cuda.checkContracts AssurancePolicy.checked))
  expect "checked CUDA retains global TorchLean tape ownership"
    (BackendProfile.checkedCuda.policy.vjpMode == .torchLeanTape)

  let cudaExact ← planOrThrow "checked cuda exact ops" BackendProfile.checkedCuda cudaExactOps
  expectCapsules "checked cuda exact capsules" cudaExact.capsuleNames exactLibTorchCapsules
  expect "every CUDA primitive is dispatched through LibTorch"
    (cudaExact.kernels.all fun kernel => kernel.capsule.provider == .libTorch)
  expect "every CUDA primitive advertises LibTorch tensor storage"
    (cudaExact.kernels.all fun kernel =>
      kernel.capsule.layoutContract.claim ==
        .layoutCompatibility kernel.op .libTorchCudaView)
  match BackendProfile.checkedCuda.planReport (cudaExactOps ++ #[.scaledDotProductAttention]) with
  | .ok report =>
      expectContains "checked cuda report names exact add" "add: libtorch.add" report
      expectContains "checked cuda report names exact max pool"
        "max_pool: libtorch.max_pool" report
      expectContains "checked cuda report names exact batchnorm"
        "batch_norm: libtorch.batch_norm" report
      expectContains "checked cuda report names exact smooth max pool"
        "smooth_max_pool: libtorch.smooth_max_pool" report
      expectContains "checked cuda report names exact attention"
        "scaled_dot_product_attention: libtorch.direct_attention" report
      expectContains "report names the classification rather than denying foreign execution"
        "trusted-external capsules: none" report
  | .error msg =>
      throw <| IO.userError s!"checked cuda report failed: {msg}"

  let directAttention ← planOrThrow "default LibTorch direct attention" BackendProfile.checkedCuda
    #[.scaledDotProductAttention]
  expectCapsules "checked CUDA selects the LibTorch forward and local VJP bridge"
    directAttention.capsuleNames #["libtorch.direct_attention"]
  expect "LibTorch attention supplies its local VJP to the TorchLean tape"
    (directAttention.kernels.all fun kernel => kernel.capsule.vjpMode == .backendVJP)
  expect "backend local VJP satisfies the existing TorchLean tape policy"
    (Attention.libTorchDirectAttention.matchesVJP BackendProfile.checkedCuda.policy)
  expect "LibTorch direct attention has aligned checked contracts"
    (isAccepted (directAttention.checkContracts AssurancePolicy.checked))
  expect "LibTorch direct attention retains the maintained evidence classification"
    (!directAttention.hasTrustedExternal)
  let reorderedLibTorch :=
    { BackendProfile.checkedCuda with
      capsuleModules := BackendProfile.checkedCuda.capsuleModules.reverse.map fun entry =>
        { entry with capsules := entry.capsules.reverse } }
  let reorderedAttention ← planOrThrow "reordered LibTorch registry" reorderedLibTorch
    #[.scaledDotProductAttention]
  expectCapsules "LibTorch preference survives reversed module and capsule order"
    reorderedAttention.capsuleNames directAttention.capsuleNames
  let libTorchOnly : BackendProfile :=
    { BackendProfile.checkedCuda with
      name := "profile_test_libtorch"
      policy := { BackendProfile.checkedCuda.policy with provider := .only .libTorch } }
  let directOps ← planOrThrow "LibTorch attention with surrounding primitives" libTorchOnly
    #[.add, .scaledDotProductAttention, .relu]
  expectCapsules "LibTorch supplies attention and its surrounding primitives"
    directOps.capsuleNames #["libtorch.add", "libtorch.direct_attention", "libtorch.relu"]
  let composedPreferred : BackendProfile :=
    { BackendProfile.checkedCuda with
      name := "profile_test_composed_attention"
      policy := { BackendProfile.checkedCuda.policy with provider := .prefer .torchLean } }
  let composedOps ← planOrThrow "explicit composed attention" composedPreferred
    #[.add, .scaledDotProductAttention, .relu]
  expectCapsules "explicit composition keeps LibTorch surrounding primitives"
    composedOps.capsuleNames #["libtorch.add", "torchlean.composed_attention", "libtorch.relu"]
  expect "explicit composition retains aligned checked contracts"
    (isAccepted (composedOps.checkContracts AssurancePolicy.checked))
  let composedOnly : BackendProfile :=
    { BackendProfile.checkedCuda with
      policy := { BackendProfile.checkedCuda.policy with provider := .only .torchLean } }
  expectPlanningFails "a composed-only provider cannot impersonate a LibTorch primitive"
    composedOnly #[.add]
  match BackendProfile.checkedCuda.planReport #[.scaledDotProductAttention] with
  | .ok report =>
      expectContains "LibTorch profile retains TorchLean tape ownership"
        "vjp=torchlean-tape" report
      expectContains "LibTorch attention report names the selected local VJP"
        "vjp=backend-vjp" report
      expectContains "LibTorch attention report names its capsule"
        "scaled_dot_product_attention: libtorch.direct_attention" report
  | .error msg =>
      throw <| IO.userError s!"LibTorch direct attention report failed: {msg}"

  for (tag, device) in
      [ ("runtime rejects named-but-unimplemented metal", NN.Backend.Device.metal)
      , ("runtime rejects named-but-unimplemented rocm", NN.Backend.Device.rocm)
      , ("runtime rejects named-but-unimplemented wasm", NN.Backend.Device.wasm)
      , ("runtime rejects named-but-unimplemented tpu", NN.Backend.Device.tpu)
      , ("runtime rejects named-but-unimplemented trainium", NN.Backend.Device.trainium)
      , ("runtime rejects named-but-unimplemented custom chip", NN.Backend.Device.custom)
      , ("runtime rejects named-but-unimplemented external", NN.Backend.Device.external)
      ] do
    expectRuntimeDeviceRejected tag device

  expectCudaSessionMatchesRuntime
  expectMetadataOnlyDeviceRejected

  let cpuSession ← Runtime.Autograd.Torch.Internal.EagerSession.new (α := Float)
    ({ device := .cpu } : Runtime.Autograd.Torch.Config)
  let firstRelu ← cpuSession.selectedCapsule .relu
  let secondRelu ← cpuSession.selectedCapsule .relu
  let cpuSelections ← cpuSession.backendSelections
  expect "session reuses the selected capsule for a repeated operation"
    (firstRelu.name == secondRelu.name && cpuSelections.size == 1)
  expectRandomProviderRejected

  let checkedCudaOpts : Runtime.Autograd.Torch.Config :=
    { device := .cuda }
  for op in
      #[ BackendOp.matmul
      , .batchNorm
      , .maxPool
      , .avgPool
      , .smoothMaxPool
      , .scaledDotProductAttention
      ] do
    expectLibTorchBindingAccepts s!"checked cuda runtime binding accepts `{op.name}`"
      checkedCudaOpts op

  IO.println "  backend profiles: ok"

end NN.Tests.Backend.Profile
