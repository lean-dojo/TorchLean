/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Engine.Cuda.Ops
public import NN.Runtime.Autograd.Engine.FastKernels
public import NN.Spec.Core.Random
public import NN.Tensor
public import NN.Tests.Runtime.Cuda.Utils

/-!
# CUDA Runtime Stress Tests

Low-level stress coverage that goes beyond the small eager-tape tests:

- reference/deterministic RNG behavior for `randUniform`, `randNormal`, and `bernoulliMask`,
- explicit `Buffer.releaseIO` lifecycle semantics,
- finalization of short-lived external buffer wrappers,
- LibTorch allocation accounting, saved attention context lifetime, and recoverable OOM,
- large-buffer elementwise/reduction checks on direct `Cuda.Buffer` ops,
- extra ATen matmul reference parity checks on rectangular inputs.

CPU parity builds still run the backend-independent checks. Native memory/accounting/OOM probes
explicitly skip CPU parity builds; only the LibTorch CUDA build exercises those contracts.
-/

@[expose] public section

namespace Tests
namespace Cuda
namespace Stress

open Runtime.Autograd
open Runtime.Autograd.Cuda

-- Buffer comparisons live in `Cuda.Utils`; every call below passes its own tolerance, since the
-- RNG prefix checks and the pointwise-pipeline check are not accurate to the same degree.
open Tests.Cuda.Utils (assertFloatArrayEq assertFloatArrayApprox)
open Spec TorchLean
open TorchLean TorchLean.Tensor

def buildFloatArray (n : Nat) (f : Nat → Float) : FloatArray :=
  Id.run do
    let mut out : Array Float := Array.mkEmpty n
    for i in [0:n] do
      out := out.push (f i)
    return FloatArray.mk out

def expectedUniformValue (key : UInt64) (i : Nat) : Float :=
  let z := Spec.Random.splitmix64 (key + UInt64.ofNat i)
  -- This is the cross-backend contract: native CUDA, the CPU stub, and pure Lean all use
  -- `splitmix64(key + i) mod 2^32`, i.e. the low 32 bits.
  let u : Nat := z.toUInt32.toNat
  (u : Float) / (((2 : Nat) ^ 32 : Nat) : Float)

def expectedUniformArray (n : Nat) (key : UInt64) : FloatArray :=
  buildFloatArray n (fun i => expectedUniformValue key i)

def expectedBernoulliArray (n : Nat) (keepProb : Float) (key : UInt64) : FloatArray :=
  buildFloatArray n (fun i =>
    let denom := (2 : Nat) ^ 32
    let draw := Spec.Random.sampleNat key i denom
    (Spec.Random.keepBit (α := Float32) keepProb.toFloat32 draw denom).toFloat)

/-- Probability endpoints remain exact when the largest 32-bit draw rounds to binary32 `1`. -/
def runBernoulliEndpointRegression : IO Unit := do
  let key : UInt64 := 0x25114ed53327345a
  unless Spec.Random.sampleNat key 0 == 4294967295 do
    throw <| IO.userError "bernoulliMask regression key no longer reaches the largest 32-bit draw"

  -- Literal masks check the endpoint contract independently of the Spec implementation.
  -- The interior case contains both kept and dropped entries at the same key.
  let cases : List (Float × Array Float) :=
    [ (0.0, #[0, 0, 0, 0, 0, 0, 0, 0])
    , (1.0, #[1, 1, 1, 1, 1, 1, 1, 1])
    , (0.5, #[0, 0, 1, 0, 0, 1, 1, 1])
    ]
  for (keepProb, expected) in cases do
    let mask ← Buffer.bernoulliMaskIO 8 keepProb key
    let actual ← Buffer.toFloatArrayIO mask
    assertFloatArrayEq s!"bernoulliMask endpoint fixture p={keepProb}"
      actual (FloatArray.mk expected)
    assertFloatArrayEq s!"bernoulliMask Float32 Spec parity p={keepProb}"
      actual (expectedBernoulliArray 8 keepProb key)

def expectedNormalValue (mean std : Float) (key : UInt64) (i : Nat) : Float :=
  let r1 := (Spec.Random.splitmix64 (key + UInt64.ofNat (2 * i))).toUInt32.toNat
  let r2 := (Spec.Random.splitmix64 (key + UInt64.ofNat (2 * i + 1))).toUInt32.toNat
  let u1 := (Float.ofNat r1 + 1.0) / 4294967297.0
  let u2 := Float.ofNat r2 / 4294967296.0
  mean + std * Float.sqrt (-2.0 * Float.log u1) * Float.cos (6.283185307179586 * u2)

def expectedNormalArray (n : Nat) (mean std : Float) (key : UInt64) : FloatArray :=
  buildFloatArray n (expectedNormalValue mean std key)

def assertFloatIsNaN (msg : String) (x : Float) : IO Unit := do
  if !x.isNaN then
    throw <| IO.userError s!"{msg}: expected NaN, got {x}"

def runRngStress : IO Unit := do
  IO.println "== low-level RNG stress =="
  runBernoulliEndpointRegression

  let key : UInt64 := 0x123456789abcdef
  let nSmall : Nat := 64
  let nLarge : Nat := 4096

  -- Exact prefix checks catch low-bits versus high-bits SplitMix64 mismatches between CPU-stub and
  -- CUDA seeded buffers.
  let uSmall := Buffer.toFloatArray (← Buffer.randUniformIO (UInt32.ofNat nSmall) key)
  let uExpected := expectedUniformArray nSmall key
  assertFloatArrayApprox "randUniform exact prefix" uSmall uExpected (tol := 1e-7)

  -- Repeated larger buffers are a cheap stress path for launch coverage and deterministic replay.
  let uLarge1 := Buffer.toFloatArray (← Buffer.randUniformIO (UInt32.ofNat nLarge) key)
  let uLarge2 := Buffer.toFloatArray (← Buffer.randUniformIO (UInt32.ofNat nLarge) key)
  assertFloatArrayEq "randUniform deterministic repeat" uLarge1 uLarge2

  let normalMean : Float := -0.25
  let normalStd : Float := 0.75
  let normalSmall :=
    Buffer.toFloatArray (Buffer.randNormal (UInt32.ofNat nSmall) normalMean normalStd key)
  let normalExpected := expectedNormalArray nSmall normalMean normalStd key
  assertFloatArrayApprox "randNormal reference prefix" normalSmall normalExpected (tol := 5e-5)
  let normalLarge1 :=
    Buffer.toFloatArray (Buffer.randNormal (UInt32.ofNat nLarge) normalMean normalStd key)
  let normalLarge2 :=
    Buffer.toFloatArray (Buffer.randNormal (UInt32.ofNat nLarge) normalMean normalStd key)
  assertFloatArrayEq "randNormal deterministic repeat" normalLarge1 normalLarge2

  let keepProb : Float := 0.35
  let mSmall := Buffer.toFloatArray (← Buffer.bernoulliMaskIO (UInt32.ofNat nSmall) keepProb key)
  let mExpected := expectedBernoulliArray nSmall keepProb key
  assertFloatArrayEq "bernoulliMask exact prefix" mSmall mExpected

  let mLarge1 := Buffer.toFloatArray (← Buffer.bernoulliMaskIO (UInt32.ofNat nLarge) keepProb key)
  let mLarge2 := Buffer.toFloatArray (← Buffer.bernoulliMaskIO (UInt32.ofNat nLarge) keepProb key)
  assertFloatArrayEq "bernoulliMask deterministic repeat" mLarge1 mLarge2

def runReleaseStress : IO Unit := do
  IO.println "== explicit release semantics =="

  let b ← Buffer.fullIO 8 3.25
  -- `releaseIO` is a lifetime hint for long eager loops: success means the allocation was freed and
  -- the wrapper was converted into an empty buffer so the finalizer remains safe.
  let r1 ← Buffer.releaseIO b
  if r1 != 1 then
    throw <| IO.userError s!"release first call: expected 1, got {r1}"
  if Buffer.size b != 0 then
    throw <| IO.userError s!"release size reset: expected 0, got {Buffer.size b}"

@[noinline] def runWrapperLifetimeIteration (i : Nat) : IO Unit := do
  let host := FloatArray.mk #[i.toFloat, 2.0, -3.0, 4.0]
  let a ← Buffer.ofFloatArrayIO host
  let doubled := Buffer.add a a
  let got ← Buffer.toFloatArrayIO doubled
  Utils.assertApprox "wrapper lifetime result" (got.get! 0) (2.0 * i.toFloat) (tol := 1e-3)

def runWrapperLifetimeStress : IO Unit := do
  IO.println "== external buffer wrapper lifetime =="

  let before ← Buffer.allocatorStats
  for i in [0:4096] do
    runWrapperLifetimeIteration i
  let after ← Buffer.allocatorStats

  let allocated := after.wrapperAllocCount - before.wrapperAllocCount
  let finalized := after.wrapperFinalizeCount - before.wrapperFinalizeCount
  if allocated < 8192 then
    throw <| IO.userError
      s!"wrapper lifetime: expected at least 8192 allocations, observed {allocated}"
  if finalized != allocated then
    throw <| IO.userError
      s!"wrapper lifetime: allocated {allocated} wrappers but finalized {finalized}"
  if after.wrapperLiveCount != before.wrapperLiveCount then
    throw <| IO.userError <|
      s!"wrapper lifetime: live wrappers changed from {before.wrapperLiveCount} to " ++
      s!"{after.wrapperLiveCount}"

def runGradientAliasingStress : IO Unit := do
  IO.println "== CUDA tape gradient aliasing regression =="

  let s : Shape := [4]
  let x : Tensor Float s := (Tensor.from #[0.25, -0.50, 0.75, -1.00]).reshape [4] (by dsimp; decide)

  let t0 : Cuda.Tape := Cuda.Tape.empty
  let (t1, xId) := Cuda.Tape.leaf (t := t0) (Utils.tensorToAnyBuffer x) (name := some "x")
  -- `x + x` sends the same upstream gradient to both parents of an add node. This checks
  -- accumulated-gradient aliasing in add nodes.
  let (t2, yId) ← Utils.okOrThrow (Cuda.Tape.add (t := t1) (s := s) xId xId)
  let (t3, outId) ← Utils.okOrThrow (Cuda.Tape.sum (t := t2) (s := s) yId)
  let seed : Cuda.AnyBuffer := { s := Shape.scalar, buf := Buffer.full 1 1.0 }
  let grads ← Utils.okOrThrow (Cuda.Tape.backwardDenseAll (t := t3) outId seed)
  let dx ← Utils.cudaGrad (s := s) grads xId
  let expected : Tensor Float s :=
    (Tensor.from #[2.0, 2.0, 2.0, 2.0]).reshape [4] (by dsimp; decide)
  Utils.assertTensorApprox (s := s) "add backward duplicate-parent gradient" dx expected

/-- Shape-erased CUDA entrypoints must reject malformed native lengths before launching kernels. -/
def runMalformedBufferValidationStress : IO Unit := do
  IO.println "== malformed CUDA buffer validation =="

  let vectorShape : Shape := [4]
  let shortBuffer ← Buffer.fullIO 3 1.0
  let malformed : Cuda.AnyBuffer := { s := vectorShape, buf := shortBuffer }
  match Cuda.AnyBuffer.validate malformed with
  | .ok _ =>
      throw <| IO.userError "AnyBuffer.validate accepted a native buffer shorter than its shape"
  | .error _ => pure ()

  let (malformedTape, malformedId) := Cuda.Tape.empty.leaf malformed
  match Cuda.Tape.sum (t := malformedTape) (s := vectorShape) malformedId with
  | .ok _ =>
      throw <| IO.userError "CUDA tape accepted a malformed leaf buffer"
  | .error _ => pure ()
  discard <| Buffer.releaseIO shortBuffer

  let scalarBuffer ← Buffer.fullIO 1 2.0
  let singleElementValue : Cuda.AnyBuffer := { s := Shape.scalar, buf := scalarBuffer }
  let (scalarTape, scalarId) := Cuda.Tape.empty.leaf singleElementValue
  let badSeedBuffer ← Buffer.fullIO 2 1.0
  let badSeed : Cuda.AnyBuffer := { s := Shape.scalar, buf := badSeedBuffer }
  match Cuda.Tape.backwardDenseAll scalarTape scalarId badSeed with
  | .ok _ =>
      throw <| IO.userError "dense CUDA backward accepted a wrong-length output seed"
  | .error _ => pure ()
  discard <| Buffer.releaseIO badSeedBuffer

  let sparseSeedBuffer ← Buffer.fullIO 2 1.0
  let sparseSeed : Cuda.AnyBuffer := { s := Shape.scalar, buf := sparseSeedBuffer }
  let sparseRejected ←
    try
      discard <| Cuda.Tape.backwardSparse scalarTape scalarId sparseSeed (fun _ => true)
      pure false
    catch _ => pure true
  unless sparseRejected do
    throw <| IO.userError "sparse CUDA backward accepted a wrong-length output seed"
  discard <| Buffer.releaseIO scalarBuffer

  let vectorBuffer ← Buffer.fullIO 4 2.0
  let vectorValue : Cuda.AnyBuffer := { s := vectorShape, buf := vectorBuffer }
  let (vectorTape, _vectorId) := Cuda.Tape.empty.leaf vectorValue
  let shortGradBuffer ← Buffer.fullIO 3 0.0
  let shortGrad : Cuda.AnyBuffer := { s := vectorShape, buf := shortGradBuffer }
  match Cuda.Tape.backwardDenseFrom vectorTape #[shortGrad] with
  | .ok _ =>
      throw <| IO.userError "dense CUDA backward accepted a malformed initial gradient"
  | .error _ => pure ()
  discard <| Buffer.releaseIO shortGradBuffer
  discard <| Buffer.releaseIO vectorBuffer

  let oversized : Cuda.AnyBuffer :=
    { s := [UInt32.size], buf := ← Buffer.zerosIO 0 }
  match Cuda.AnyBuffer.validate oversized with
  | .ok _ =>
      throw <| IO.userError "AnyBuffer.validate accepted a shape whose numel exceeds UInt32"
  | .error _ => pure ()

  let hiddenOversizedAxis : Cuda.AnyBuffer :=
    { s := Shape.ofList [0, UInt32.size], buf := ← Buffer.zerosIO 0 }
  match Cuda.AnyBuffer.validate hiddenOversizedAxis with
  | .ok _ =>
      throw <| IO.userError
        "AnyBuffer.validate accepted an oversized axis hidden by a zero-sized axis"
  | .error _ => pure ()

/-- A disconnected reciprocal at zero must keep explicit CUDA dense gradients finite and zero. -/
def runDisconnectedDenseGradientStress : IO Unit := do
  IO.println "== disconnected CUDA dense gradient =="

  let scalarShape : Shape := Shape.scalar
  let zero : Tensor Float scalarShape := Tensor.scalar 0.0
  let output : Tensor Float scalarShape := Tensor.scalar 3.0
  let t0 : Cuda.Tape := Cuda.Tape.empty
  let (t1, xId) := Cuda.Tape.leaf (t := t0) (Utils.tensorToAnyBuffer zero) (name := some "x")
  let (t2, outId) :=
    Cuda.Tape.leaf (t := t1) (Utils.tensorToAnyBuffer output) (name := some "output")
  let (t3, invId) ← Utils.okOrThrow <|
    Cuda.Tape.inv (t := t2) (s := scalarShape) xId
  let seed : Cuda.AnyBuffer := { s := scalarShape, buf := Buffer.full 1 1.0 }
  let grads ← Utils.okOrThrow <| Cuda.Tape.backwardDenseAll (t := t3) outId seed
  unless grads.size = t3.nodes.size do
    throw <| IO.userError "disconnected CUDA dense gradient: result length mismatch"
  let xGrad := Tensor.item (← Utils.cudaGrad (s := scalarShape) grads xId)
  let invGrad := Tensor.item (← Utils.cudaGrad (s := scalarShape) grads invId)
  unless xGrad.isFinite && xGrad == 0.0 do
    throw <| IO.userError s!"disconnected CUDA reciprocal input: expected finite zero, got {xGrad}"
  unless invGrad.isFinite && invGrad == 0.0 do
    throw <| IO.userError
      s!"disconnected CUDA reciprocal output: expected finite zero, got {invGrad}"

/--
Constant dropout probabilities must not trigger scalar broadcast reductions in backward.
Trainable probabilities retain their local derivative with the sampled mask held fixed.
-/
def runConstantBranchGradientStress : IO Unit := do
  IO.println "== constant branches and trainable dropout probability =="
  let vector : Shape := [4]
  let scalar : Shape := .scalar
  let key : UInt64 := 0x25114ed53327345a
  for probability in #[0.0, 0.5, 1.0] do
    for trainable in #[false, true] do
      let x ← Buffer.ofFloatArrayIO (FloatArray.mk #[1.0, 2.0, 3.0, 4.0])
      let p ← Buffer.fullIO 1 probability
      let mask ← Buffer.bernoulliMaskIO 4 (1.0 - probability) key
      let ones ← Buffer.fullIO 4 1.0
      let (t1, xId) := Cuda.Tape.empty.leaf { s := vector, buf := x }
      let (t2, pId) := t1.leaf { s := scalar, buf := p } (requiresGrad := trainable)
      let (t3, maskId) := t2.leaf { s := vector, buf := mask } (requiresGrad := false)
      let (t4, onesId) := t3.leaf { s := vector, buf := ones } (requiresGrad := false)
      let (t5, broadcastId) ← Utils.okOrThrow <|
        t4.broadcastTo (Shape.CanBroadcastTo.scalarTo vector) pId
      let (t6, retainedId) ← Utils.okOrThrow <| t5.mul (s := vector) broadcastId maskId
      let (t7, denominatorId) ← Utils.okOrThrow <| t6.sub (s := vector) onesId retainedId
      let (t8, inverseId) ← Utils.okOrThrow <| t7.inv (s := vector) denominatorId
      for id in #[broadcastId, retainedId, denominatorId, inverseId] do
        unless (t8.getNode? id).any (fun node => node.requiresGrad == trainable) do
          throw <| IO.userError "dropout probability gradient flag was not propagated"
      -- A failing closure detects accidental traversal independently of the numerical assertions.
      let t8 : Cuda.Tape := if trainable then t8 else
        { nodes := t8.nodes.mapIdx fun id node =>
            if id == broadcastId then
              { node with backward := fun _ => .error "constant probability was differentiated" }
            else node }
      let (t9, maskedId) ← Utils.okOrThrow <| t8.mul (s := vector) xId maskId
      let (t10, yId) ← Utils.okOrThrow <| t9.mul (s := vector) maskedId inverseId
      let (tape, outId) ← Utils.okOrThrow <| t10.sum (s := vector) yId
      let expectedValue := if probability == 0.0 then 10.0 else
        if probability == 0.5 then 6.0 else 0.0
      let expectedInput := FloatArray.mk <| if probability == 0.0 then #[1.0, 1.0, 1.0, 1.0]
        else if probability == 0.5 then #[0.0, 0.0, 2.0, 0.0] else #[0.0, 0.0, 0.0, 0.0]
      let expectedProbability := if probability == 0.0 then 10.0 else
        if probability == 0.5 then 12.0 else 0.0
      let output ← Utils.okOrThrow <| tape.requireValue outId scalar
      assertFloatArrayEq "dropout forward value"
        (← Buffer.toFloatArrayIO output) (FloatArray.mk #[expectedValue])
      let denseSeed : Cuda.AnyBuffer := { s := scalar, buf := ← Buffer.fullIO 1 1.0 }
      let dense ← Utils.okOrThrow <| tape.backwardDenseAll outId denseSeed
      let inputGradient ← Utils.cudaGrad (s := vector) dense xId
      assertFloatArrayEq "dropout dense input gradient"
        (Runtime.Autograd.Cuda.Convert.flattenFloat inputGradient) expectedInput
      let probabilityGradient ← Utils.cudaGrad (s := scalar) dense pId
      unless probabilityGradient.item == (if trainable then expectedProbability else 0.0) do
        throw <| IO.userError "dropout dense probability gradient mismatch"
      for gradient in dense do
        discard <| Buffer.releaseIO gradient.buf
      let sparseSeed : Cuda.AnyBuffer := { s := scalar, buf := ← Buffer.fullIO 1 1.0 }
      let sparse ← tape.backwardSparse outId sparseSeed (fun id => id == xId || id == pId)
      try
        let inputGradient ← match sparse.get? xId with
          | some gradient => pure gradient
          | none => throw <| IO.userError "dropout sparse input gradient missing"
        assertFloatArrayEq "dropout sparse input gradient"
          (← Buffer.toFloatArrayIO inputGradient.buf) expectedInput
        match sparse.get? pId with
        | some gradient =>
            unless trainable do
              throw <| IO.userError "dropout sparse backward retained a constant gradient"
            assertFloatArrayEq "dropout sparse probability gradient"
              (← Buffer.toFloatArrayIO gradient.buf) (FloatArray.mk #[expectedProbability])
        | none =>
            if trainable then
              throw <| IO.userError "dropout sparse probability gradient missing"
      finally
        Cuda.Tape.releaseSparseGrads sparse
        for node in tape.nodes do
          if node.ownsValue then
            discard <| Buffer.releaseIO node.value.buf
          for buffer in node.cleanup do
            discard <| Buffer.releaseIO buffer

def runSparseLifetimeStress : IO Unit := do
  IO.println "== repeated sparse-backward ownership =="

  let before ← Buffer.allocatorStats
  let s : Shape := [4]
  let x : Tensor Float s := (Tensor.from #[0.25, -0.50, 0.75, -1.00]).reshape [4] (by dsimp; decide)
  let t0 : Cuda.Tape := Cuda.Tape.empty
  let (t1, xId) := Cuda.Tape.leaf (t := t0) (Utils.tensorToAnyBuffer x) (name := some "x")
  let (t2, outId) ← Utils.okOrThrow (Cuda.Tape.sum (t := t1) (s := s) xId)

  -- The output cotangent must be allocated afresh on every pass. A pure constant allocation can
  -- be hoisted by Lean and then reused after sparse backward has retired its native storage.
  for pass in [0:8] do
    let seed : Cuda.AnyBuffer := { s := Shape.scalar, buf := ← Buffer.fullIO 1 1.0 }
    let grads ← Cuda.Tape.backwardSparse (t := t2) outId seed (fun id => id == xId)
    let dx ← match grads.get? xId with
      | some dx => pure dx
      | none => throw <| IO.userError s!"sparse backward pass {pass}: missing leaf gradient"
    let dx ← Utils.anyBufferToTensor (s := s) dx
    let expected : Tensor Float s :=
      (Tensor.from #[1.0, 1.0, 1.0, 1.0]).reshape [4] (by dsimp; decide)
    Utils.assertTensorApprox (s := s) s!"sparse backward pass {pass}" dx expected
    Cuda.Tape.releaseSparseGrads grads

  -- This test owns the tape and therefore retires its persistent forward values explicitly.
  for node in t2.nodes do
    discard <| Buffer.releaseIO node.value.buf
  let after ← Buffer.allocatorStats
  if after.liveBytes > before.liveBytes then
    throw <| IO.userError
      s!"sparse backward ownership: live bytes grew from {before.liveBytes} to {after.liveBytes}"

def runLargeBufferStressBody : IO Unit := do
  IO.println "== large buffer elementwise/reduction stress =="

  let n : Nat := 200003
  let aHost := buildFloatArray n (fun i =>
    (((i % 97 : Nat) : Float) / 17.0) - 2.5)
  let bHost := buildFloatArray n (fun i =>
    ((((i * 7 + 3) % 101 : Nat) : Float) / 19.0) - 1.75)

  let aBuf := Buffer.ofFloatArray aHost
  let bBuf := Buffer.ofFloatArray bHost
  -- Run through several direct buffer kernels without involving the autograd tape. This exercises
  -- the low-level launch paths that the small tape tests can miss.
  let added := Buffer.add aBuf bBuf
  let muld := Buffer.mul added aBuf
  let shifted := Buffer.axpy muld bBuf 0.125
  let clamped := Buffer.clamp shifted (-3.5) 4.25
  let relued := Buffer.relu clamped
  let got := Buffer.toFloatArray relued

  let expected := buildFloatArray n (fun i =>
    let a := aHost.get! i
    let b := bHost.get! i
    let y := (a + b) * a + 0.125 * b
    let y := max y (-3.5)
    let y := min y 4.25
    if y > 0.0 then y else 0.0)
  assertFloatArrayApprox "large buffer pointwise pipeline" got expected (tol := 2e-5)

  let previousSettings : Option (Bool × Bool) ←
    match Buffer.runtimeStatus with
    | .nativeAvailable =>
        pure (some (← LibTorch.getDeterministic, ← LibTorch.getCuDNNBenchmark))
    | .cpuStub => pure none
    | .nativeUnavailable =>
        throw <| IO.userError "large buffer stress requires a usable CUDA device"
  -- CPU startup policy is checked by the subprocess runner. Native strict determinism is
  -- requested through LibTorch; the accuracy assertions retain their explicit tolerances.
  try
    if previousSettings.isSome then
      LibTorch.setDeterministic true
    let sumGot := (Buffer.toFloatArray (Buffer.reduceSum relued)).get! 0
    let meanGot := (Buffer.toFloatArray (Buffer.reduceMean relued)).get! 0
    let mut sumExpected : Float := 0.0
    for i in [0:n] do
      sumExpected := sumExpected + expected.get! i
    let meanExpected : Float := sumExpected / (n : Float)

    Utils.assertApprox "large buffer reduceSum" sumGot sumExpected (tol := 0.5)
    Utils.assertApprox "large buffer reduceMean" meanGot meanExpected (tol := 5e-4)
  finally
    if let some (deterministic, benchmark) := previousSettings then
      LibTorch.setDeterministic deterministic
      LibTorch.setCuDNNBenchmark benchmark

  -- The runtime contract for an empty mean is `NaN`; keep that edge case explicit.
  let emptyMean := Buffer.toFloatArray (Buffer.reduceMean (← Buffer.zerosIO 0))
  if emptyMean.size != 1 then
    throw <| IO.userError s!"reduceMean empty size: expected 1, got {emptyMean.size}"
  assertFloatIsNaN "reduceMean empty result" (emptyMean.get! 0)

/--
CPU parity exercises its deterministic reduction path in a fresh process with an explicit startup
policy. Native execution uses LibTorch controls; CPU stubs do not claim to support those controls.
-/
def runLargeBufferStress : IO Unit := do
  match Buffer.runtimeStatus with
  | .cpuStub =>
      if (← IO.getEnv "TORCHLEAN_CPU_REDUCTION_PROBE") == some "1" then
        -- Float32 left-to-right accumulation loses the middle 1; the CPU deterministic
        -- accumulator retains it. This checks the active path rather than just the environment.
        let input ← Buffer.ofFloatArrayIO (FloatArray.mk #[100000000.0, 1.0, -100000000.0])
        let result := Buffer.reduceSum input
        assertFloatArrayEq "CPU deterministic reduction discriminator"
          (← Buffer.toFloatArrayIO result) (FloatArray.mk #[1.0])
        discard <| Buffer.releaseIO result
        discard <| Buffer.releaseIO input
        IO.println "  CPU deterministic reduction coverage; LibTorch controls unavailable"
        runLargeBufferStressBody
      else
        let self ← IO.appPath
        let result ← IO.Process.output {
          cmd := self.toString
          args := #[]
          env := #[("TORCHLEAN_CPU_REDUCTION_PROBE", some "1"),
            ("TORCHLEAN_CUDA_DETERMINISTIC_REDUCTIONS", some "1")]
        }
        if result.exitCode != 0 then
          throw <| IO.userError
            s!"CPU deterministic reduction probe failed (exit {result.exitCode}):\n\
              {result.stdout}\n{result.stderr}"
        IO.print result.stdout
  | .nativeAvailable => runLargeBufferStressBody
  | .nativeUnavailable => Buffer.requireNativeRuntime

def runMatmulStress : IO Unit := do
  IO.println "== ATen matmul reference parity stress =="

  -- Rectangular case: catches row-major/column-major leading-dimension mistakes that square
  -- matrices can accidentally hide.
  let sA1 : Shape := [3, 4]
  let sB1 : Shape := [4, 5]
  let sY1 : Shape := [3, 5]
  let a1 : Tensor Float sA1 :=
    (Tensor.from #[
      0.10, -0.20, 0.30, -0.40,
      0.55, 0.65, -0.75, 0.85,
      -0.15, 0.25, -0.35, 0.45
    ]).reshape [3, 4] (by dsimp; decide)
  let b1 : Tensor Float sB1 :=
    (Tensor.from #[
      0.20, -0.10, 0.05, 0.30, -0.40,
      -0.15, 0.25, -0.35, 0.45, 0.10,
      0.50, -0.60, 0.70, -0.80, 0.90,
      -0.05, 0.15, -0.25, 0.35, -0.45
    ]).reshape [4, 5] (by dsimp; decide)
  let yRef1 := FastKernels.matmulReference (α := Float) (m := 3) (n := 4) (p := 5) a1 b1
  let yFp321 ← IO.ofExcept (FastKernels.Cuda.matmulLibTorch .fp32 (m := 3) (n := 4) (p := 5) a1 b1)
  let yFp641 ← IO.ofExcept (FastKernels.Cuda.matmulLibTorch .fp64 (m := 3) (n := 4) (p := 5) a1 b1)
  Utils.assertTensorApprox (s := sY1) "matmul stress case1 fp32" yFp321 yRef1 (tol := 7e-3)
  Utils.assertTensorApprox (s := sY1) "matmul stress case1 fp64" yFp641 yRef1 (tol := 1e-9)

  -- Dot-product-shaped case: small but asymmetric enough to exercise the degenerate leading
  -- dimensions in the DGEMM bridge.
  let sA2 : Shape := [1, 7]
  let sB2 : Shape := [7, 1]
  let sY2 : Shape := [1, 1]
  let a2 : Tensor Float sA2 :=
    (Tensor.from #[0.25, -0.50, 0.75, -1.00, 1.25, -1.50, 1.75]).reshape [1, 7] (by dsimp; decide)
  let b2 : Tensor Float sB2 :=
    (Tensor.from #[0.10, 0.20, -0.30, 0.40, -0.50, 0.60, -0.70]).reshape [7, 1] (by dsimp; decide)
  let yRef2 := FastKernels.matmulReference (α := Float) (m := 1) (n := 7) (p := 1) a2 b2
  let yFp322 ← IO.ofExcept (FastKernels.Cuda.matmulLibTorch .fp32 (m := 1) (n := 7) (p := 1) a2 b2)
  let yFp642 ← IO.ofExcept (FastKernels.Cuda.matmulLibTorch .fp64 (m := 1) (n := 7) (p := 1) a2 b2)
  Utils.assertTensorApprox (s := sY2) "matmul stress case2 fp32" yFp322 yRef2 (tol := 7e-3)
  Utils.assertTensorApprox (s := sY2) "matmul stress case2 fp64" yFp642 yRef2 (tol := 1e-9)

/-- Check a real device readback, including its length and every uploaded/fill value. -/
def assertMemoryReadback (label : String) (buffer : Buffer) (n : Nat) (value : Float) :
    IO Unit := do
  let actual ← Buffer.toFloatArrayIO buffer
  if actual.size != n then
    throw <| IO.userError s!"{label}: expected {n} elements, got {actual.size}"
  for i in [0:n] do
    Utils.assertApprox s!"{label}[{i}]" (actual.get! i) value (tol := 1e-5)

/-- These inequalities concern native accounting, not a particular cache block size or policy. -/
def assertNativeAccounting (label : String) (stats : Buffer.AllocatorStats) : IO Unit := do
  if stats.allocatedBytes > stats.reservedBytes then
    throw <| IO.userError s!"{label}: allocated bytes exceed reserved bytes"
  if stats.peakAllocatedBytes < stats.allocatedBytes ||
      stats.peakReservedBytes < stats.reservedBytes then
    throw <| IO.userError s!"{label}: native peak is below current usage"
  if stats.peakBytes < stats.liveBytes then
    throw <| IO.userError s!"{label}: logical peak is below live ownership"

/-- Synchronize the selected device before taking a native allocator snapshot. -/
def synchronizedStats : IO Buffer.AllocatorStats := do
  LibTorch.synchronize
  Buffer.allocatorStats

/-- Warm initialization outside the measured ownership interval. -/
@[noinline] def warmMemoryProbe : IO Unit := do
  let buffer ← Buffer.fullIO 4 2.0
  assertMemoryReadback "memory probe warmup" buffer 4 2.0
  discard <| Buffer.releaseIO buffer
  LibTorch.synchronize
  LibTorch.emptyCache

/-- Live allocations survive emptyCache; released payloads leave allocated accounting. -/
def runMemoryAccountingProbe : IO Unit := do
  Buffer.requireNativeRuntime
  IO.println "== LibTorch native memory accounting =="
  warmMemoryProbe
  let before ← synchronizedStats
  let n : UInt32 := 262144 -- 1 MiB of float32 payload per independent handle.
  let count : Nat := 8
  let bytes := UInt64.ofNat (n.toNat * count * 4)
  let mut held : Array Buffer := #[]
  for i in [0:count] do
    let buffer ← Buffer.fullIO n (Float.ofNat (i + 1))
    assertMemoryReadback "held allocation" buffer n.toNat (Float.ofNat (i + 1))
    held := held.push buffer
  let live ← synchronizedStats
  assertNativeAccounting "live" live
  if live.liveBytes != before.liveBytes + bytes then
    throw <| IO.userError "logical payload accounting does not match held handles"
  if live.allocatedBytes < before.allocatedBytes + bytes then
    throw <| IO.userError "native allocated bytes did not include materialized device payloads"

  LibTorch.emptyCache
  let retained ← synchronizedStats
  assertNativeAccounting "live after emptyCache" retained
  if retained.liveBytes != live.liveBytes || retained.allocatedBytes != live.allocatedBytes then
    throw <| IO.userError "emptyCache changed live ownership or native allocated storage"
  if retained.reservedBytes > live.reservedBytes then
    throw <| IO.userError "emptyCache increased native reservation in an idle process"
  for i in [0:held.size] do
    let some buffer := held[i]? |
      throw <| IO.userError "memory accounting: held buffer index is out of bounds"
    assertMemoryReadback "live after emptyCache" buffer n.toNat (Float.ofNat (i + 1))

  for buffer in held do
    if (← Buffer.releaseIO buffer) != 1 then
      throw <| IO.userError "memory accounting: first release did not retire the payload"
  let released ← synchronizedStats
  assertNativeAccounting "released" released
  if released.liveBytes != before.liveBytes || released.allocatedBytes != before.allocatedBytes then
    throw <| IO.userError "released payloads remain live in logical/native allocated accounting"
  -- Keep the empty wrappers alive across the snapshot: wrapper ownership is not VRAM usage.
  for buffer in held do
    if (← Buffer.sizeIO buffer) != 0 || (← Buffer.releaseIO buffer) != 0 then
      throw <| IO.userError "released wrapper is nonempty or release is not idempotent"
  LibTorch.emptyCache
  let emptied ← synchronizedStats
  assertNativeAccounting "released after emptyCache" emptied
  if emptied.allocatedBytes != before.allocatedBytes ||
      emptied.reservedBytes > released.reservedBytes then
    throw <| IO.userError "emptyCache violated native allocation/reservation accounting"
  if emptied.peakAllocatedBytes < live.peakAllocatedBytes ||
      emptied.peakReservedBytes < live.peakReservedBytes then
    throw <| IO.userError "emptyCache reset the allocator peaks"
  -- Driver free bytes can change because of other processes; report rather than assert equality.
  IO.println s!"  before: {before.format}"
  IO.println s!"  held: {live.format}"
  IO.println s!"  released/cache emptied: {emptied.format}"

/--
The forward context must retain Q/K/V after their Lean payload handles are explicitly released.
Readback and backward use that retained storage; both explicit release and finalization of the
output must eventually release the context. Native probabilities/auxiliaries have no Lean handle.
-/
@[noinline] def runAttentionContextIteration (releaseOutput : Bool) : IO Unit := do
  let before ← synchronizedStats
  let n : UInt32 := 64
  let d : UInt32 := 32
  let elements : UInt32 := n * d
  let bytes := UInt64.ofNat (elements.toNat * 4)
  let query ← Buffer.zerosIO elements
  let key ← Buffer.zerosIO elements
  let value ← Buffer.fullIO elements 2.0
  let mask ← Buffer.zerosIO 0
  let outResult ← IO.lazyPure fun _ =>
    Buffer.libTorchAttentionFwd query key value mask 0 1 n d 1.0
  let out ← Utils.okOrThrow outResult
  assertMemoryReadback "attention forward" out elements.toNat 2.0
  for buffer in #[query, key, value, mask] do
    discard <| Buffer.releaseIO buffer
  let retained ← synchronizedStats
  assertNativeAccounting "attention retained context" retained
  if retained.liveBytes != before.liveBytes + bytes then
    throw <| IO.userError "attention input payload handles were not retired"
  if retained.allocatedBytes < before.allocatedBytes + 4 * bytes then
    throw <| IO.userError "attention context did not retain native Q/K/V/output storage"
  LibTorch.emptyCache
  let afterEmpty ← synchronizedStats
  if afterEmpty.allocatedBytes != retained.allocatedBytes then
    throw <| IO.userError "emptyCache released live attention context storage"
  assertMemoryReadback "attention after input release/cache empty" out elements.toNat 2.0

  let upstream ← Buffer.fullIO elements 1.0
  let gradResult ← IO.lazyPure fun _ => Buffer.libTorchAttentionBwd out upstream
  let (dq, dk, dv) ← Utils.okOrThrow gradResult
  -- Uniform attention, constant V, and unit output cotangent give dQ=dK=0, dV=1.
  assertMemoryReadback "attention retained dQ" dq elements.toNat 0.0
  assertMemoryReadback "attention retained dK" dk elements.toNat 0.0
  assertMemoryReadback "attention retained dV" dv elements.toNat 1.0
  for buffer in #[upstream, dq, dk, dv] do
    discard <| Buffer.releaseIO buffer
  if releaseOutput then
    if (← Buffer.releaseIO out) != 1 then
      throw <| IO.userError "attention output release did not retire its payload/context"
  else
    -- Final use keeps out alive until here. The noinline IO scope then finalizes its wrapper.
    if (← Buffer.sizeIO out) != elements then
      throw <| IO.userError "attention output disappeared before finalization"

/-- Saved native attention state is accounted and reclaimed independently of logical handles. -/
def runAttentionMemoryProbe : IO Unit := do
  Buffer.requireNativeRuntime
  IO.println "== LibTorch saved attention context lifetime =="
  -- Select an always-supported float32 provider; do not depend on GPU-specific flash eligibility.
  LibTorch.setSDPEnabled .math true
  LibTorch.setSDPEnabled .flash false
  LibTorch.setSDPEnabled .efficient false
  LibTorch.setSDPEnabled .cuDNN false
  runAttentionContextIteration true
  LibTorch.synchronize
  LibTorch.emptyCache
  let before ← synchronizedStats
  for releaseOutput in [true, false] do
    runAttentionContextIteration releaseOutput
    let after ← synchronizedStats
    assertNativeAccounting "attention context retired" after
    if after.liveBytes != before.liveBytes || after.allocatedBytes != before.allocatedBytes then
      throw <| IO.userError
        s!"attention context leaked after releaseOutput={releaseOutput}: {after.format}"
    LibTorch.emptyCache
    let emptied ← synchronizedStats
    if emptied.allocatedBytes != before.allocatedBytes ||
        emptied.reservedBytes > after.reservedBytes then
      throw <| IO.userError "attention retirement/cache-empty accounting mismatch"
  IO.println "  explicit release and finalization both reclaimed saved context"

/-- A failed native allocation must report the recoverable IO error class, not abort the process. -/
def expectAllocationOOM (label : String) (allocate : IO Buffer) : IO Unit := do
  let exhausted ←
    try
      let unexpected ← allocate
      discard <| Buffer.releaseIO unexpected
      pure false
    catch error =>
      match error with
      | .resourceExhausted .. => pure true
      | _ => throw <| IO.userError s!"{label}: wrong allocation error: {error}"
  unless exhausted do
    throw <| IO.userError s!"{label}: oversized allocation unexpectedly succeeded"

/-- Isolated allocator limit causes a bounded OOM, then smaller allocations recover in-process. -/
def runMemoryOOMProbe : IO Unit := do
  Buffer.requireNativeRuntime
  IO.println "== LibTorch controlled OOM and recovery =="
  warmMemoryProbe
  let survivor ← Buffer.fullIO 1024 3.25
  assertMemoryReadback "OOM survivor before" survivor 1024 3.25
  LibTorch.emptyCache
  let before ← synchronizedStats
  let oldFraction ← LibTorch.getMemoryFraction
  -- This is an allocation limit, not an attempt to exhaust physical VRAM or set a cache budget.
  let limitBytes := before.reservedBytes.toNat + 32 * 1024 * 1024
  if before.deviceTotalBytes.toNat <= 2 * limitBytes then
    throw <| IO.userError "controlled OOM probe needs room for a 32 MiB allocator limit"
  let fraction := Float.ofNat limitBytes / Float.ofNat before.deviceTotalBytes.toNat
  let largeElements := 2 * limitBytes / 4
  if largeElements >= UInt32.size then
    throw <| IO.userError "controlled OOM request exceeds the buffer ABI"
  let n := UInt32.ofNat largeElements
  try
    LibTorch.setMemoryFraction fraction
    let observed ← LibTorch.getMemoryFraction
    Utils.assertApprox "allocator memory fraction readback" observed fraction (tol := 1e-9)
    expectAllocationOOM "zerosIO OOM" (Buffer.zerosIO n)
    expectAllocationOOM "fullIO OOM" (Buffer.fullIO n 1.0)
    -- Upload uses another IO allocation entrypoint and must preserve the same error class.
    let host := FloatArray.mk (Array.replicate largeElements 2.0)
    expectAllocationOOM "ofFloatArrayIO OOM" (Buffer.ofFloatArrayIO host)
    let failed ← synchronizedStats
    if failed.liveBytes != before.liveBytes || failed.allocatedBytes != before.allocatedBytes then
      throw <| IO.userError "failed allocation leaked logical/native allocated storage"
    assertMemoryReadback "OOM survivor after failures" survivor 1024 3.25
    LibTorch.emptyCache
    let recovered ← Buffer.fullIO 1024 7.0
    assertMemoryReadback "smaller allocation after OOM" recovered 1024 7.0
    discard <| Buffer.releaseIO recovered
    let after ← synchronizedStats
    if after.liveBytes != before.liveBytes || after.allocatedBytes != before.allocatedBytes then
      throw <| IO.userError "smaller recovery allocation did not retire cleanly"
  finally
    LibTorch.setMemoryFraction oldFraction
    discard <| Buffer.releaseIO survivor
    LibTorch.synchronize
    LibTorch.emptyCache
  Utils.assertApprox "memory fraction restored" (← LibTorch.getMemoryFraction) oldFraction
    (tol := 1e-9)
  let finalBuffer ← Buffer.fullIO 4096 9.0
  assertMemoryReadback "allocation after restoring limit" finalBuffer 4096 9.0
  discard <| Buffer.releaseIO finalBuffer
  IO.println "  OOM returned resourceExhausted; live inputs and subsequent allocations survived"

/--
Fork fresh processes so accounting excludes earlier suite work and allocator limits cannot affect
other tests. No assertion depends on an exact cached block size or amount returned to the driver.
-/
def runMemoryTests : IO Unit := do
  IO.println "== LibTorch memory accounting and OOM (isolated processes) =="
  match Buffer.runtimeStatus with
  | .cpuStub =>
      let stats ← Buffer.allocatorStats
      if stats.allocatedBytes != 0 || stats.reservedBytes != 0 ||
          stats.peakAllocatedBytes != 0 || stats.peakReservedBytes != 0 ||
          stats.deviceFreeBytes != 0 || stats.deviceTotalBytes != 0 then
        throw <| IO.userError "CPU parity build reported native CUDA memory accounting"
      IO.println "  skipped: CPU parity build has no ATen CUDA memory/accounting/OOM coverage"
      return
  | .nativeUnavailable =>
      throw <| IO.userError "LibTorch memory tests require a usable CUDA device"
  | .nativeAvailable => pure ()
  let self : System.FilePath := "/proc/self/exe"
  if !(← self.pathExists) then
    IO.println "  skipped: isolated memory tests require Linux /proc/self/exe"
    return
  for probe in ["accounting", "attention-context", "oom-recovery"] do
    let result ← IO.Process.output {
      cmd := self.toString
      args := #[]
      env := #[("TORCHLEAN_LIBTORCH_MEMORY_PROBE", some probe),
        ("TORCHLEAN_REQUIRE_CUDA", some "1")]
    }
    if result.exitCode != 0 then
      throw <| IO.userError
        s!"LibTorch {probe} failed (exit {result.exitCode}):\n{result.stdout}\n{result.stderr}"
    IO.print result.stdout

def run : IO Unit := do
  IO.println "=== CUDA runtime stress suite ==="
  runRngStress
  runReleaseStress
  runWrapperLifetimeStress
  runMemoryTests
  runGradientAliasingStress
  runMalformedBufferValidationStress
  runDisconnectedDenseGradientStress
  runConstantBranchGradientStress
  runSparseLifetimeStress
  runLargeBufferStress
  runMatmulStress

end Stress
end Cuda
end Tests
