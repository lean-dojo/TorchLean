/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Model.Optim
public import NN.Runtime.Autograd.Torch.Core.BackwardOptim
public import NN.Runtime.Autograd.Torch.Core.Trainer.CheckpointSchema
public import NN.Tensor.Internal.Elab.TensorLiteral

/-!
# Parameter Storage Aliases

Equal values, partial sharing, frozen views, changed layouts, and repeated recordings exercise the
storage-identity boundary. Optimizer comparisons use independently allocated reference parameters
with explicitly summed gradients. CUDA checks also round-trip the shared AdamW checkpoint.
-/

public section

namespace NN.Tests.Runtime.ParameterAliases

open TorchLean Runtime.Autograd.Torch Runtime.Autograd.Torch.Internal
open Runtime.Autograd.Torch.Internal.EagerSession

private def check (label : String) (condition : Bool) : IO Unit := do
  unless condition do
    throw <| IO.userError s!"parameter aliases: {label}"

private def rejects (label : String) (action : IO Unit) : IO Unit := do
  let rejected ← try
    action
    pure false
  catch _ => pure true
  check label rejected

private def scalar (value : Float := 2.0) : IO (Param Float []) :=
  Param.Internal.create (Tensor.scalar value)

private def descriptor {shape : Spec.Shape} (p : Param Float shape) : ParameterStorage Float :=
  { shape, value := p.value, cudaValue := p.cudaValue, hostCurrent := p.hostCurrent }

private def bits (p : Param Float []) : IO UInt64 := do
  syncParamCudaToHost p
  pure (← p.value.get).item.toBits

/-- Fixed expected slots distinguish identity from equal values and partial sharing. -/
def checkStorage : IO Unit := do
  let p ← scalar
  let q ← scalar
  let value ← IO.mkRef (Tensor.scalar 2.0)
  let cudaValue ← IO.mkRef (none : Option Runtime.Autograd.Cuda.AnyBuffer)
  let hostCurrent ← IO.mkRef true
  let vector ← Param.Internal.create ([2.0] : Tensor Float [1])
  let entries := #[
    some (descriptor p), some (descriptor q),
    some (descriptor { p with name := some "same storage" }),
    some (descriptor { p with value }),
    some (descriptor { p with cudaValue }),
    some (descriptor { p with hostCurrent }),
    some (descriptor vector), none, some (descriptor p)]
  let expected := #[some 0, some 1, some 0, some 3, some 4, some 5, some 6, none, some 0]
  check "all three storage cells and shape determine aliases"
    ((← ParameterStorage.canonicalIndices entries) == expected)
  setParamHostValue p (Tensor.scalar 19.0)
  check "changing tensor contents preserves storage identity"
    ((← ParameterStorage.canonicalIndices entries) == expected)
  let params : ParamList Float [[], [], [], []] :=
    .cons { p with requiresGrad := false } (.cons p (.cons q (.cons p .nil)))
  check "frozen first occurrence never owns a trainable history"
    ((← ParamList.canonicalSlotIndices params) == #[none, some 1, some 2, some 1])
  let independent : ParamList Float [[], []] := .cons p (.cons q .nil)
  let tied : ParamList Float [[], []] := .cons p (.cons p .nil)
  for _ in [:4] do
    check "retie" ((← ParamList.canonicalSlotIndices tied) == #[some 0, some 0])
    check "untie" ((← ParamList.canonicalSlotIndices independent) == #[some 0, some 1])
  for _ in [:128] do
    let fresh ← scalar
    let other ← scalar
    check "fresh allocations cannot reuse a previous call's alias decisions"
      ((← ParameterStorage.canonicalIndices
        #[some (descriptor fresh), some (descriptor other), some (descriptor fresh)]) ==
        #[some 0, some 1, some 0])
  let independentSchema ← checkpointParameterSchema independent
  check "independent checkpoint keeps version-2 schema"
    (independentSchema.isWellFormed && independentSchema.representatives.isNone)
  let tiedSchema ← checkpointParameterSchema params
  check "checkpoint slots agree with optimizer slots"
    (tiedSchema.isWellFormed &&
      tiedSchema.representatives == some #[none, some 1, some 2, some 1])

/-- Missing descriptors stay separate; malformed descriptors fail before a grouped update. -/
def checkRegistrations : IO Unit := do
  let p ← scalar
  let q ← scalar
  let session ← EagerSession.new (α := Float)
  session.paramsByLeaf.set <|
    (Std.HashMap.emptyWithCapacity : Std.HashMap Nat (AnyParam Float))
      |>.insert 40 (AnyParam.ofParam p)
      |>.insert 10 (AnyParam.ofParam q)
      |>.insert 20 (AnyParam.ofParam p)
      |>.insert 30 (AnyParam.ofParam p)
  session.parameterStorageByLeaf.set <|
    (Std.HashMap.emptyWithCapacity : Std.HashMap Nat (ParameterStorage Float))
      |>.insert 40 (descriptor p) |>.insert 20 (descriptor p)
  let groups ← session.parameterGroups
  check "groups retain recording order and first-leaf optimizer key"
    (groups.map (fun group => (group.id, group.leaves)) ==
      #[(10, #[10]), (20, #[20, 40]), (30, #[30])])
  let vector ← Param.Internal.create ([2.0] : Tensor Float [1])
  session.parameterStorageByLeaf.modify (·.insert 40 (descriptor vector))
  rejects "wrong-shape storage descriptor" (discard session.parameterGroups)
  check "descriptor failure does not mutate parameter" ((← bits p) == (2.0 : Float).toBits)
  session.resetTape

/-- Recorded values survive mutation and updates; cotangents update current storage once. -/
def checkSnapshots (device : NN.Backend.Device) : IO Unit := do
  let p ← scalar 1.0
  let session ← EagerSession.new (α := Float) { device }
  let observer ← EagerSession.new (α := Float) { device }
  let observed ← observer.use p
  let first ← session.use p
  setParamHostValue p (Tensor.scalar 5.0)
  let second ← session.use p
  let groups ← session.parameterGroups
  check "two reads form one storage group"
    (groups.map (·.leaves) == #[#[first.id, second.id]])
  check "first recording remains a snapshot" ((← session.getValue first).item == 1.0)
  check "second recording observes host mutation" ((← session.getValue second).item == 5.0)
  let gradients := #[Spec.SomeTensor.ofTensor (Tensor.scalar (2.0 : Float)),
    Spec.SomeTensor.ofTensor (Tensor.scalar (3.0 : Float))]
  session.sgdStepAll 0.25 gradients
  check "SGD updates current storage once with summed gradients"
    ((← bits p) == (3.75 : Float).toBits)
  check "update preserves first snapshot" ((← session.getValue first).item == 1.0)
  check "update preserves second snapshot" ((← session.getValue second).item == 5.0)
  session.resetTape
  check "reset preserves other session snapshot" ((← observer.getValue observed).item == 1.0)
  observer.resetTape

private def triple (p q : Param Float []) : ParamList Float [[], [], []] :=
  .cons p (.cons q (.cons p .nil))

private def pair (p q : Param Float []) : ParamList Float [[], []] :=
  .cons p (.cons q .nil)

/-- Tied and independently summed updates must retain exactly the same multi-step history. -/
def checkOptimizer
    (make : (shapes : List Spec.Shape) →
      Runtime.Autograd.Model.Optim.Optimizer Float shapes) : IO Unit := do
  let p ← scalar
  let q ← scalar
  let referenceP ← scalar
  let referenceQ ← scalar
  let optimizer := make [[], [], []]
  let referenceOptimizer := make [[], []]
  let mut state ← optimizer.init (triple p q)
  let mut referenceState ← referenceOptimizer.init (pair referenceP referenceQ)
  for step in [:6] do
    let a := (step + 1).toFloat
    let b := (step + 3).toFloat
    let c := (step + 5).toFloat
    let gradients : TensorPack Float [[], [], []] :=
      .cons (Tensor.scalar a) (.cons (Tensor.scalar b) (.cons (Tensor.scalar c) .nil))
    let referenceGradients : TensorPack Float [[], []] :=
      .cons (Tensor.scalar (a + c)) (.cons (Tensor.scalar b) .nil)
    state ← optimizer.step state (triple p q) gradients
    referenceState ← referenceOptimizer.step referenceState
      (pair referenceP referenceQ) referenceGradients
    check "tied optimizer parameter trajectory" ((← bits p) == (← bits referenceP))
    check "independent optimizer parameter trajectory" ((← bits q) == (← bits referenceQ))

/-- Cotangent validation precedes mutation, including cross-shape alias-map corruption. -/
def checkGradientValidation : IO Unit := do
  let p ← scalar
  let q ← scalar
  let session ← EagerSession.new (α := Float)
  discard <| session.use p
  discard <| session.use q
  let malformed := #[Spec.SomeTensor.ofTensor (Tensor.scalar (1.0 : Float)),
    Spec.SomeTensor.ofTensor ([3.0] : Tensor Float [1])]
  rejects "gradient shape mismatch" (session.sgdStepAll 0.1 malformed)
  check "gradient failure preserves first parameter" ((← bits p) == (2.0 : Float).toBits)
  let gradients : TensorPack Float [[], [1]] :=
    .cons (Tensor.scalar 1.0) (.cons ([3.0] : Tensor Float [1]) .nil)
  rejects "cross-shape canonical gradients"
    (discard <| ParamList.canonicalGradients #[some 0, some 0] gradients)
  session.resetTape

private def cudaGradients (a b c : Float) : IO CudaGradMap := do
  let first ← CudaBridge.toAnyBuffer (Tensor.scalar a)
  let second ← CudaBridge.toAnyBuffer (Tensor.scalar b)
  let third ← CudaBridge.toAnyBuffer (Tensor.scalar c)
  pure <| (Std.HashMap.emptyWithCapacity : CudaGradMap)
    |>.insert 0 first |>.insert 1 second |>.insert 2 third

private def recordTriple (session : EagerSession Float) (p q : Param Float []) : IO Unit := do
  session.resetTape
  discard <| session.use p
  discard <| session.use q
  discard <| session.use p

private def firstMomentBytes (state : CudaAdamState) (index : Nat) : IO ByteArray := do
  let some entry := state.get? index
    | throw <| IO.userError "parameter aliases: missing CUDA Adam state"
  Runtime.Autograd.Cuda.Buffer.toFloat32BytesIO entry.m

/-- Separate histories left by a former untied layout are rejected without updating storage. -/
def checkCudaRetie : IO Unit := do
  let p ← scalar
  let q ← scalar
  let session ← EagerSession.new (α := Float) { device := .cuda }
  let config ← IO.mkRef (none : Option CudaAdamConfig)
  let state ← IO.mkRef (Std.HashMap.emptyWithCapacity : CudaAdamState)
  try
    discard <| session.use p
    discard <| session.use q
    let first ← CudaBridge.toAnyBuffer (Tensor.scalar (1.0 : Float))
    let second ← CudaBridge.toAnyBuffer (Tensor.scalar (3.0 : Float))
    let gradients := (Std.HashMap.emptyWithCapacity : CudaGradMap)
      |>.insert 0 first |>.insert 1 second
    withCudaGradMap gradients fun gradients =>
      session.adamWStepAllCudaMap config state 0.1 0.05 0.8 0.9 1e-8 gradients
    let before ← bits p
    let beforeState ← state.get
    let beforeFirst ← firstMomentBytes beforeState 0
    let beforeSecond ← firstMomentBytes beforeState 1
    session.resetTape
    discard <| session.use p
    discard <| session.use p
    let first ← CudaBridge.toAnyBuffer (Tensor.scalar (1.0 : Float))
    let second ← CudaBridge.toAnyBuffer (Tensor.scalar (3.0 : Float))
    let gradients := (Std.HashMap.emptyWithCapacity : CudaGradMap)
      |>.insert 0 first |>.insert 1 second
    withCudaGradMap gradients fun gradients =>
      rejects "retied CUDA layout with separate histories"
        (session.adamWStepAllCudaMap config state 0.1 0.05 0.8 0.9 1e-8 gradients)
    check "rejected retie preserves parameter" ((← bits p) == before)
    let afterState ← state.get
    check "rejected retie preserves state count and steps"
      (afterState.size == 2 && afterState.toList.all (fun entry => entry.2.t == 1))
    check "rejected retie preserves first moment"
      ((← firstMomentBytes afterState 0) == beforeFirst)
    check "rejected retie preserves second moment"
      ((← firstMomentBytes afterState 1) == beforeSecond)
  finally
    session.resetTape
    releaseCudaAdamState (← state.get)

private def cudaStep (session : EagerSession Float)
    (config : IO.Ref (Option CudaAdamConfig)) (state : IO.Ref CudaAdamState) : IO Unit := do
  let gradients ← cudaGradients 1.0 5.0 3.0
  withCudaGradMap gradients fun gradients =>
    session.adamWStepAllCudaMap config state 0.1 0.05 0.8 0.9 1e-8 gradients

private def sameCudaState (left right : CudaAdamState) : IO Bool := do
  unless left.size == right.size do return false
  for (id, entry) in left.toList do
    let some other := right.get? id | return false
    unless entry.t == other.t do return false
    unless (← Runtime.Autograd.Cuda.Buffer.toFloat32BytesIO entry.m) ==
        (← Runtime.Autograd.Cuda.Buffer.toFloat32BytesIO other.m) do return false
    unless (← Runtime.Autograd.Cuda.Buffer.toFloat32BytesIO entry.v) ==
        (← Runtime.Autograd.Cuda.Buffer.toFloat32BytesIO other.v) do return false
  pure true

/-- Tied CUDA AdamW follows an independent run with gradients summed before the optimizer. -/
def checkCudaOptimizer : IO Unit := do
  let p ← scalar
  let q ← scalar
  let referenceP ← scalar
  let referenceQ ← scalar
  let session ← EagerSession.new (α := Float) { device := .cuda }
  let reference ← EagerSession.new (α := Float) { device := .cuda }
  let config ← IO.mkRef (none : Option CudaAdamConfig)
  let state ← IO.mkRef (Std.HashMap.emptyWithCapacity : CudaAdamState)
  let referenceConfig ← IO.mkRef (none : Option CudaAdamConfig)
  let referenceState ← IO.mkRef (Std.HashMap.emptyWithCapacity : CudaAdamState)
  try
    for step in [:6] do
      recordTriple session p q
      reference.resetTape
      discard <| reference.use referenceP
      discard <| reference.use referenceQ
      let a := (step + 1).toFloat
      let b := (step + 3).toFloat
      let c := (step + 5).toFloat
      let gradients ← cudaGradients a b c
      withCudaGradMap gradients fun gradients =>
        session.adamWStepAllCudaMap config state 0.1 0.05 0.8 0.9 1e-8 gradients
      let first ← CudaBridge.toAnyBuffer (Tensor.scalar (a + c))
      let second ← CudaBridge.toAnyBuffer (Tensor.scalar b)
      let referenceGradients := (Std.HashMap.emptyWithCapacity : CudaGradMap)
        |>.insert 0 first |>.insert 1 second
      withCudaGradMap referenceGradients fun gradients =>
        reference.adamWStepAllCudaMap referenceConfig referenceState
          0.1 0.05 0.8 0.9 1e-8 gradients
      check "CUDA tied AdamW trajectory" ((← bits p) == (← bits referenceP))
      check "CUDA independent AdamW trajectory" ((← bits q) == (← bits referenceQ))
      check "CUDA tied AdamW histories"
        (← sameCudaState (← state.get) (← referenceState.get))
  finally
    session.resetTape
    reference.resetTape
    releaseCudaAdamState (← state.get)
    releaseCudaAdamState (← referenceState.get)

/--
Restore tied AdamW state byte-for-byte, reject retied destinations, then match the next update.
-/
def checkCudaCheckpoint : IO Unit := IO.FS.withTempDir fun directory => do
  let p ← scalar
  let q ← scalar
  let session ← EagerSession.new (α := Float) { device := .cuda }
  let config ← IO.mkRef (none : Option CudaAdamConfig)
  let state ← IO.mkRef (Std.HashMap.emptyWithCapacity : CudaAdamState)
  let restoredConfig ← IO.mkRef (none : Option CudaAdamConfig)
  let restoredState ← IO.mkRef (Std.HashMap.emptyWithCapacity : CudaAdamState)
  let resumed ← EagerSession.new (α := Float) { device := .cuda }
  try
    for _ in [:4] do
      recordTriple session p q
      cudaStep session config state
    let currentState ← state.get
    check "CUDA tied parameters have one history per storage"
      (currentState.size == 2 && currentState.contains 0 && currentState.contains 1 &&
        !currentState.contains 2 && currentState.toList.all (fun entry => entry.2.t == 4))
    let schema ← checkpointParameterSchema (triple p q)
    let saved := directory / "adamw.bin"
    writeCudaAdamStateFloat32 saved schema config state
    let pValue ← bits p
    let qValue ← bits q
    let restoredP ← scalar (Float.ofBits pValue)
    let restoredQ ← scalar (Float.ofBits qValue)
    let restoredSchema ← checkpointParameterSchema (triple restoredP restoredQ)
    readCudaAdamStateFloat32 saved restoredSchema restoredConfig restoredState
    check "restored moment bytes" (← sameCudaState (← state.get) (← restoredState.get))
    let wrong : ParamList Float [[], [], []] :=
      .cons restoredP (.cons restoredQ (.cons restoredQ .nil))
    let wrongSchema ← checkpointParameterSchema wrong
    rejects "retied checkpoint destination"
      (readCudaAdamStateFloat32 saved wrongSchema restoredConfig restoredState)
    check "rejected restore preserves existing histories"
      (← sameCudaState (← state.get) (← restoredState.get))
    recordTriple session p q
    recordTriple resumed restoredP restoredQ
    cudaStep session config state
    cudaStep resumed restoredConfig restoredState
    check "checkpoint next tied update" ((← bits p) == (← bits restoredP))
    check "checkpoint next independent update" ((← bits q) == (← bits restoredQ))
    check "checkpoint next moment bytes" (← sameCudaState (← state.get) (← restoredState.get))
    let savedNext := directory / "next.bin"
    let restoredNext := directory / "restored-next.bin"
    writeCudaAdamStateFloat32 savedNext schema config state
    writeCudaAdamStateFloat32 restoredNext restoredSchema restoredConfig restoredState
    check "next checkpoint bytes"
      ((← IO.FS.readBinFile savedNext) == (← IO.FS.readBinFile restoredNext))
    let stochastic := directory / "stochastic.bin"
    let counter ← IO.mkRef 99
    writeCudaAdamStateFloat32 stochastic schema config state (some 7)
    readCudaAdamStateFloat32 stochastic restoredSchema restoredConfig restoredState (some counter)
    check "checkpoint restores the next random invocation" ((← counter.get) == 7)
    rejects "random checkpoint without a destination counter"
      (readCudaAdamStateFloat32 stochastic restoredSchema restoredConfig restoredState)
    rejects "random checkpoint with a retied destination"
      (readCudaAdamStateFloat32 stochastic wrongSchema restoredConfig restoredState (some counter))
    let truncated := directory / "truncated.bin"
    let bytes ← IO.FS.readBinFile stochastic
    IO.FS.writeBinFile truncated (bytes.extract 0 (bytes.size - 1))
    rejects "truncated random checkpoint"
      (readCudaAdamStateFloat32 truncated restoredSchema restoredConfig restoredState
        (some counter))
    check "rejected checkpoint preserves random counter" ((← counter.get) == 7)
    check "rejected random checkpoint preserves moments"
      (← sameCudaState (← state.get) (← restoredState.get))
  finally
    session.resetTape
    resumed.resetTape
    releaseCudaAdamState (← state.get)
    releaseCudaAdamState (← restoredState.get)

def runCpu : IO Unit := do
  checkStorage
  checkRegistrations
  checkSnapshots .cpu
  checkGradientValidation
  checkOptimizer fun _ => Runtime.Autograd.Model.Optim.sgd 0.1
  checkOptimizer fun _ => Runtime.Autograd.Model.Optim.momentumSGD 0.1 0.8
  checkOptimizer fun _ => Runtime.Autograd.Model.Optim.adam 0.1 0.8 0.9 1e-8
  checkOptimizer fun _ => Runtime.Autograd.Model.Optim.adamw 0.1 0.05 0.8 0.9 1e-8
  IO.println "PARAMETER_ALIAS_CPU_PASSED"

def runCuda : IO Unit := do
  Runtime.Autograd.Cuda.Buffer.requireNativeRuntime
  checkSnapshots .cuda
  checkCudaRetie
  checkCudaOptimizer
  checkCudaCheckpoint
  Runtime.Autograd.Cuda.Buffer.collectGarbage
  IO.println "PARAMETER_ALIAS_CUDA_PASSED"

end NN.Tests.Runtime.ParameterAliases
