/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public meta import TorchLean.Tensor.Einops.Elab.Einsum
public meta import TorchLean.Tensor.Einops.Elab.Pack
public meta import TorchLean.Tensor.Literal
public meta import TorchLean.Tensor.Einops.Elab.Transform
public meta import TorchLean.Tensor.Tactic.Einops.Report
public import TorchLean.Tensor.Literal
public import TorchLeanTest.Harness.CUDA
public import TorchLean.NN.Module
public import TorchLean.NN.Builder
public import TorchLean.Tensor.Tactic.Init

/-!
# CUDA Runtime Stress Tests

Low-level stress coverage that goes beyond the small eager-tape tests:

- reference/deterministic RNG behavior for `randUniform` and `bernoulliMask`,
- initial parameters of a seeded module on CUDA, word for word those of the CPU,
- the lifecycle semantics of `Buffer.releaseIO`,
- finalization of short-lived external buffer wrappers,
- LibTorch allocation accounting, saved attention buffer lifetime, and recoverable OOM, each in a
  fresh process,
- rejection of malformed native lengths before kernels launch,
- tape regressions: a parent read twice by an add node, a disconnected reciprocal at zero, masked
  scaling with a constant and with a trainable probability, and live storage across repeated
  sparse backward passes,
- large-buffer elementwise/reduction checks on direct `LibTorch.Buffer` ops,
- ATen matmul in float32 and float64 against `Tensor.matmul` on rectangular inputs.

Only the LibTorch CUDA build runs these checks.
-/

@[expose] public section

namespace TorchLean.Test
namespace CUDA
namespace Stress

open LibTorch Buffer

-- Buffer comparisons come from `Test.Harness`. The RNG prefix checks are exact, and only the
-- pointwise-pipeline check passes a tolerance to `assertFloatArrayApprox`.
open Harness (assertFloatArrayEq assertFloatArrayApprox)
open TorchLean.Tensor

/-- The array `f 0, ..., f (n - 1)`. -/
def buildFloatArray (n : Nat) (f : Nat → Float) : FloatArray :=
  Id.run do
    let mut out : Array Float := Array.mkEmpty n
    for i in [0:n] do
      out := out.push (f i)
    return FloatArray.mk out

/-- The reference uniform draw at index `i` of `key`: the low 32 bits of `splitmix64 (key + i)`
divided by `2^32`. -/
def expectedUniformValue (key : UInt64) (i : Nat) : Float :=
  let z := Random.splitmix64 (key + UInt64.ofNat i)
  -- The cross-backend contract: the LibTorch backend and pure Lean both use
  -- `splitmix64(key + i) mod 2^32`, the low 32 bits.
  let u : Nat := z.toUInt32.toNat
  (u : Float) / (((2 : Nat) ^ 32 : Nat) : Float)

/-- The first `n` reference uniform draws of `key`, rounded to binary32 as a float32 buffer
stores them. -/
def expectedUniformArray (n : Nat) (key : UInt64) : FloatArray :=
  buildFloatArray n (fun i => (expectedUniformValue key i).toFloat32.toFloat)

/-- The first `n` entries of the reference keep mask of `key` with keep probability `keepProb`,
from `Random.keepBit` in binary32. -/
def expectedBernoulliArray (n : Nat) (keepProb : Float) (key : UInt64) : FloatArray :=
  buildFloatArray n (fun i =>
    let denom := (2 : Nat) ^ 32
    let draw := Random.sampleNat key i denom
    (Random.keepBit (α := Float32) keepProb.toFloat32 draw denom).toFloat)

/-- The masks at the probability endpoints `0` and `1` are exact at a key whose first draw is the
largest 32-bit value, `2^32 - 1`, which rounds to binary32 `1` as a fraction of `2^32`. -/
def runBernoulliEndpointRegression : IO Unit := do
  let key : UInt64 := 0x25114ed53327345a
  unless Random.sampleNat key 0 == 4294967295 do
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

/--
Three seeded affine layers whose weights use the Xavier-uniform, normal and uniform
initializers, the last over bounds whose width overflows binary64. The biases use the uniform,
Kaiming-uniform and zeros initializers.
-/
def initializerModel : nn.Sequential [3] [4] :=
  nn.build 17 do
    let first ← nn.linear 3 16 (config := { biasInit := .uniform (-0.25) 0.75 })
    let second ← nn.linear 16 8 (config :=
      { weightInit? := some (.normal (-0.25) 0.75)
        biasInit := .kaimingUniform 16 })
    let third ← nn.linear 8 4 (config :=
      { weightInit? :=
          some (.uniform (-1.7976931348623157e308) 1.7976931348623157e308) })
    pure (nn.compose (nn.compose first second) third)

/-- Append the words `bits x` of every tensor in `pack`, each in row-major order, to `words`. -/
def packWords {α : Type} [Storage α] (bits : α → UInt64) :
    {shapes : List Shape} → TensorPack α shapes → Array UInt64 → Array UInt64
  | [], .nil, words => words
  | _ :: _, .cons tensor rest, words =>
      packWords bits rest (Tensor.foldl (fun acc x => acc.push (bits x)) words tensor)

/-- Check that `initializerModel` instantiated on CUDA with element type `α` has, word for
word, the initial parameters of its CPU instantiation. -/
def checkModuleInitializerParity (name : String) (α : Type) [Storage α] [Context α]
    [FromFloat α] [TensorTransfer α] (bits : α → UInt64) :
    IO Unit := do
  let words (runtime : Session.Config) : IO (Array UInt64) := do
    let module ← nn.Module.instantiate initializerModel runtime α
    pure (packWords bits (← module.state) #[])
  let expected ← words {}
  let actual ← words { device := .cuda }
  unless actual.size == expected.size do
    throw <| IO.userError
      s!"CUDA module initialization ({name}): {actual.size} words, expected {expected.size}"
  for i in [:expected.size] do
    unless actual[i]! == expected[i]! do
      throw <| IO.userError s!"CUDA module initialization ({name})[{i}]: got \
        0x{actual[i]!.toBitVec.toHex}, expected 0x{expected[i]!.toBitVec.toHex}"

/-- Compare the initial parameter words of `initializerModel` on CUDA and on the CPU, for
`Float32` and `Float` parameters. -/
def runInitializerParity : IO Unit := do
  IO.println "== CUDA module initialization parity =="
  -- Without a plan the CUDA parameters are uploads of the CPU values, which would make the
  -- comparison hold for any device sampler.
  unless (nn.Sequential.runtimeInit? initializerModel).isSome do
    throw <| IO.userError "initializerModel has no runtime initialization plan"
  checkModuleInitializerParity "Float32" Float32 fun x => x.toBits.toUInt64
  checkModuleInitializerParity "Float" Float Float.toBits

/-- The Bernoulli endpoint regression, then `randUniform` and `bernoulliMask` buffers of 64 entries
against the reference draws and buffers of 4096 entries against a second draw of the same key. -/
def runRNG : IO Unit := do
  IO.println "== low-level RNG stress =="
  runBernoulliEndpointRegression

  let key : UInt64 := 0x123456789abcdef
  let nSmall : Nat := 64
  let nLarge : Nat := 4096

  -- Prefix comparisons are exact, since each draw `u / 2^32` rounds once to binary32. They catch
  -- low-bits versus high-bits SplitMix64 mismatches between the Lean reference and CUDA seeded
  -- buffers.
  let uSmall := Buffer.toFloatArray (← Buffer.randUniformIO (UInt32.ofNat nSmall) key)
  let uExpected := expectedUniformArray nSmall key
  assertFloatArrayEq "randUniform exact prefix" uSmall uExpected

  -- Repeated larger buffers are a cheap stress path for launch coverage and deterministic replay.
  let uLarge1 := Buffer.toFloatArray (← Buffer.randUniformIO (UInt32.ofNat nLarge) key)
  let uLarge2 := Buffer.toFloatArray (← Buffer.randUniformIO (UInt32.ofNat nLarge) key)
  assertFloatArrayEq "randUniform deterministic repeat" uLarge1 uLarge2

  let keepProb : Float := 0.35
  let mSmall := Buffer.toFloatArray (← Buffer.bernoulliMaskIO (UInt32.ofNat nSmall) keepProb key)
  let mExpected := expectedBernoulliArray nSmall keepProb key
  assertFloatArrayEq "bernoulliMask exact prefix" mSmall mExpected

  let mLarge1 := Buffer.toFloatArray (← Buffer.bernoulliMaskIO (UInt32.ofNat nLarge) keepProb key)
  let mLarge2 := Buffer.toFloatArray (← Buffer.bernoulliMaskIO (UInt32.ofNat nLarge) keepProb key)
  assertFloatArrayEq "bernoulliMask deterministic repeat" mLarge1 mLarge2

/-- A first `Buffer.releaseIO` of a live buffer returns `1` and leaves an empty wrapper. -/
def runRelease : IO Unit := do
  IO.println "== explicit release semantics =="

  let b ← Buffer.fullIO 8 3.25
  -- `releaseIO` is a lifetime hint for long eager loops. The result `1` means it freed the
  -- allocation and turned the wrapper into an empty buffer, so a later finalizer is safe.
  let r1 ← Buffer.releaseIO b
  if r1 != 1 then
    throw <| IO.userError s!"release first call: expected 1, got {r1}"
  if Buffer.size b != 0 then
    throw <| IO.userError s!"release size reset: expected 0, got {Buffer.size b}"

/-- One step of `runWrapperLifetime`: upload a four-entry buffer, add it to itself and check that
the first entry is `2 * i`. -/
@[noinline] def runWrapperLifetimeIteration (i : Nat) : IO Unit := do
  let host := FloatArray.mk #[i.toFloat, 2.0, -3.0, 4.0]
  let a ← Buffer.ofFloatArrayIO host
  let doubled := Buffer.add a a
  let got ← Buffer.toFloatArrayIO doubled
  Harness.assertApprox "wrapper lifetime result" (got.get! 0) (2.0 * i.toFloat) (tol := 1e-3)

/-- 4096 steps of `runWrapperLifetimeIteration` allocate at least 8192 external wrappers, all of
which are finalized, and leave the number of live wrappers unchanged. -/
def runWrapperLifetime : IO Unit := do
  IO.println "== external buffer wrapper lifetime =="

  let before ← Buffer.Memory.read
  for i in [0:4096] do
    runWrapperLifetimeIteration i
  let after ← Buffer.Memory.read

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

/-- On the CUDA tape the gradient of `x` through `sum (x + x)` is `2` in every entry, so the two
contributions of the add node do not alias. -/
def runGradientAliasing : IO Unit := do
  IO.println "== CUDA tape gradient aliasing regression =="

  let s : Shape := [4]
  let x : Tensor Float s := (Tensor.from #[0.25, -0.50, 0.75, -1.00]).reshape [4] (by dsimp; decide)

  let t0 : Tape LibTorch.SomeBuffer := Tape.empty
  let (t1, xId) := Tape.leaf (t := t0) (Harness.tensorToSomeBuffer x) (name := some "x")
  -- `x + x` sends the same upstream gradient to both parents of an add node. The gradient of `x`
  -- must accumulate to `2` in every entry, which catches aliasing between the two contributions.
  let (t2, yId) ← Harness.record t1 "add" (.add (s := s)) (xId, xId, ())
  let (t3, outId) ← Harness.record t2 "sum" (Harness.sumAll s) (yId, ())
  let seed : LibTorch.SomeBuffer := { s := Shape.scalar, buf := ← Buffer.fullIO 1 1.0 }
  let grads ← IO.ofExcept (Tape.backwardDenseAll (t := t3) outId seed)
  let dx ← Harness.cudaGrad (s := s) grads xId
  let expected : Tensor Float s :=
    (Tensor.from #[2.0, 2.0, 2.0, 2.0]).reshape [4] (by dsimp; decide)
  Harness.assertTensorApprox (s := s) "add backward duplicate-parent gradient" dx expected

/-- Shape-erased CUDA entry points must reject malformed native lengths before launching kernels.
`LibTorch.SomeBuffer.validate` rejects a buffer shorter than its shape, an element count beyond
`UInt32`, and an axis beyond `UInt32` that an axis of extent zero hides from the count. On the tape
a sum over the short buffer as a leaf fails, as do dense and sparse backward passes from seeds of
the wrong length and a dense pass from a short initial gradient. -/
def runMalformedBuffers : IO Unit := do
  IO.println "== malformed CUDA buffer validation =="

  let vectorShape : Shape := [4]
  let shortBuffer ← Buffer.fullIO 3 1.0
  let malformed : LibTorch.SomeBuffer := { s := vectorShape, buf := shortBuffer }
  match LibTorch.SomeBuffer.validate malformed with
  | .ok _ =>
      throw <| IO.userError "SomeBuffer.validate accepted a native buffer shorter than its shape"
  | .error _ => pure ()

  let (malformedTape, malformedId)
      := Tape.leaf Tape.empty malformed
  match (← LibTorch.Tape.apply malformedTape "sum" (Harness.sumAll vectorShape) (malformedId, ())
      (pure .float32)).2
  with
  | .ok _ =>
      throw <| IO.userError "CUDA tape accepted a malformed leaf buffer"
  | .error _ => pure ()
  discard <| Buffer.releaseIO shortBuffer

  let scalarBuffer ← Buffer.fullIO 1 2.0
  let singleElementValue : LibTorch.SomeBuffer := { s := Shape.scalar, buf := scalarBuffer }
  let (scalarTape, scalarId)
      := Tape.leaf Tape.empty singleElementValue
  let badSeedBuffer ← Buffer.fullIO 2 1.0
  let badSeed : LibTorch.SomeBuffer := { s := Shape.scalar, buf := badSeedBuffer }
  match Tape.backwardDenseAll scalarTape scalarId badSeed with
  | .ok _ =>
      throw <| IO.userError "dense CUDA backward accepted a wrong-length output seed"
  | .error _ => pure ()
  discard <| Buffer.releaseIO badSeedBuffer

  let sparseSeedBuffer ← Buffer.fullIO 2 1.0
  let sparseSeed : LibTorch.SomeBuffer := { s := Shape.scalar, buf := sparseSeedBuffer }
  let sparseRejected ←
    try
      discard <| LibTorch.Tape.backwardSparse scalarTape scalarId sparseSeed (fun _ => true)
      pure false
    catch _ => pure true
  unless sparseRejected do
    throw <| IO.userError "sparse CUDA backward accepted a wrong-length output seed"
  discard <| Buffer.releaseIO scalarBuffer

  let vectorBuffer ← Buffer.fullIO 4 2.0
  let vectorValue : LibTorch.SomeBuffer := { s := vectorShape, buf := vectorBuffer }
  let (vectorTape, _vectorId)
      := Tape.leaf Tape.empty vectorValue
  let shortGradBuffer ← Buffer.fullIO 3 0.0
  let shortGrad : LibTorch.SomeBuffer := { s := vectorShape, buf := shortGradBuffer }
  match Tape.backwardDenseFrom vectorTape #[shortGrad] with
  | .ok _ =>
      throw <| IO.userError "dense CUDA backward accepted a malformed initial gradient"
  | .error _ => pure ()
  discard <| Buffer.releaseIO shortGradBuffer
  discard <| Buffer.releaseIO vectorBuffer

  let oversized : LibTorch.SomeBuffer :=
    { s := [UInt32.size], buf := ← Buffer.zerosIO 0 }
  match LibTorch.SomeBuffer.validate oversized with
  | .ok _ =>
      throw <| IO.userError "SomeBuffer.validate accepted a shape whose numel exceeds UInt32"
  | .error _ => pure ()

  let hiddenOversizedAxis : LibTorch.SomeBuffer :=
    { s := [0, UInt32.size], buf := ← Buffer.zerosIO 0 }
  match LibTorch.SomeBuffer.validate hiddenOversizedAxis with
  | .ok _ =>
      throw <| IO.userError
        "SomeBuffer.validate accepted an oversized axis hidden by a zero-sized axis"
  | .error _ => pure ()

/-- A reciprocal at zero that the output does not read, and its input, must get dense CUDA
gradients that are finite zeros. -/
def runDisconnectedDenseGradient : IO Unit := do
  IO.println "== disconnected CUDA dense gradient =="

  let scalarShape : Shape := Shape.scalar
  let zero : Tensor Float scalarShape := Tensor.scalar 0.0
  let output : Tensor Float scalarShape := Tensor.scalar 3.0
  let t0 : Tape LibTorch.SomeBuffer := Tape.empty
  let (t1, xId) := Tape.leaf (t := t0) (Harness.tensorToSomeBuffer zero) (name := some "x")
  let (t2, outId) :=
    Tape.leaf (t := t1) (Harness.tensorToSomeBuffer output) (name := some "output")
  let (t3, invId) ← Harness.record t2 "inv" (.reciprocal (s := scalarShape)) (xId, ())
  let seed : LibTorch.SomeBuffer := { s := scalarShape, buf := ← Buffer.fullIO 1 1.0 }
  let grads ← IO.ofExcept <| Tape.backwardDenseAll (t := t3) outId seed
  unless grads.size = t3.nodes.size do
    throw <| IO.userError "disconnected CUDA dense gradient: result length mismatch"
  let xGrad := Tensor.item (← Harness.cudaGrad (s := scalarShape) grads xId)
  let invGrad := Tensor.item (← Harness.cudaGrad (s := scalarShape) grads invId)
  unless xGrad.isFinite && xGrad == 0.0 do
    throw <| IO.userError s!"disconnected CUDA reciprocal input: expected finite zero, got {xGrad}"
  unless invGrad.isFinite && invGrad == 0.0 do
    throw <| IO.userError
      s!"disconnected CUDA reciprocal output: expected finite zero, got {invGrad}"

/--
The masked scaling `x * mask / (1 - p * mask)` built from tape records: a constant probability must
not trigger scalar broadcast reductions in backward, and a trainable one keeps its local derivative
with the sampled mask held fixed.
-/
def runConstantBranchGradient : IO Unit := do
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
      let (t1, xId) :=
        Tape.leaf Tape.empty { s := vector, buf := x : SomeBuffer }
      let (t2, pId) := Tape.leaf t1 { s := scalar, buf := p } (requiresGrad := trainable)
      let (t3, maskId) := Tape.leaf t2 { s := vector, buf := mask } (requiresGrad := false)
      let (t4, onesId) := Tape.leaf t3 { s := vector, buf := ones } (requiresGrad := false)
      let cb := Shape.scalar_canBroadcastTo vector
      let (t5, broadcastId) ← Harness.record t4 "broadcastTo" (.broadcast cb.dims cb.broadcastDims)
        (pId, ())
      let (t6, retainedId) ← Harness.record t5 "mul" (.mul (s := vector)) (broadcastId, maskId, ())
      let (t7, denominatorId) ← Harness.record t6 "sub" (.sub (s := vector))
        (onesId, retainedId, ())
      let (t8, inverseId) ← Harness.record t7 "inv" (.reciprocal (s := vector)) (denominatorId, ())
      for id in #[broadcastId, retainedId, denominatorId, inverseId] do
        unless (t8.getNode? id).any (fun node => node.requiresGrad == trainable) do
          throw <| IO.userError "dropout probability gradient flag was not propagated"
      -- A failing closure detects accidental traversal independently of the numerical assertions.
      let t8 : Tape LibTorch.SomeBuffer
          := if trainable then t8 else
        { nodes := t8.nodes.mapIdx fun id node =>
            if id == broadcastId then
              { node with backward := fun _ => .error "constant probability was differentiated" }
            else node }
      let (t9, maskedId) ← Harness.record t8 "mul" (.mul (s := vector)) (xId, maskId, ())
      let (t10, yId) ← Harness.record t9 "mul" (.mul (s := vector)) (maskedId, inverseId, ())
      let (tape, outId) ← Harness.record t10 "sum" (Harness.sumAll vector) (yId, ())
      let expectedValue := if probability == 0.0 then 10.0 else
        if probability == 0.5 then 6.0 else 0.0
      let expectedInput := FloatArray.mk <| if probability == 0.0 then #[1.0, 1.0, 1.0, 1.0]
        else if probability == 0.5 then #[0.0, 0.0, 2.0, 0.0] else #[0.0, 0.0, 0.0, 0.0]
      let expectedProbability := if probability == 0.0 then 10.0 else
        if probability == 0.5 then 12.0 else 0.0
      let output ← IO.ofExcept <| LibTorch.Tape.requireValue tape outId scalar
      assertFloatArrayEq "dropout forward value"
        (← Buffer.toFloatArrayIO output) (FloatArray.mk #[expectedValue])
      let denseSeed : LibTorch.SomeBuffer := { s := scalar, buf := ← Buffer.fullIO 1 1.0 }
      let dense ← IO.ofExcept <| tape.backwardDenseAll outId denseSeed
      let inputGradient ← Harness.cudaGrad (s := vector) dense xId
      assertFloatArrayEq "dropout dense input gradient"
        (inputGradient.to FloatArray) expectedInput
      let probabilityGradient ← Harness.cudaGrad (s := scalar) dense pId
      unless probabilityGradient.item == (if trainable then expectedProbability else 0.0) do
        throw <| IO.userError "dropout dense probability gradient mismatch"
      for gradient in dense do
        discard <| Buffer.releaseIO gradient.buf
      let sparseSeed : LibTorch.SomeBuffer := { s := scalar, buf := ← Buffer.fullIO 1 1.0 }
      let sparse ← LibTorch.Tape.backwardSparse tape outId sparseSeed
          (fun id => id == xId || id == pId)
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
        LibTorch.Tape.releaseSparseGrads sparse
        for node in tape.nodes do
          if node.ownsValue then
            discard <| Buffer.releaseIO node.value.buf
          for buffer in node.cleanup do
            discard <| Buffer.releaseIO buffer.buf

/-- Eight sparse backward passes through `sum x` on the CUDA tape each give the gradient `1` in
every entry, and the live bytes do not grow once the gradients and the values are released. -/
def runSparseLifetime : IO Unit := do
  IO.println "== repeated sparse-backward ownership =="

  let before ← Buffer.Memory.read
  let s : Shape := [4]
  let x : Tensor Float s := (Tensor.from #[0.25, -0.50, 0.75, -1.00]).reshape [4] (by dsimp; decide)
  let t0 : Tape LibTorch.SomeBuffer := Tape.empty
  let (t1, xId) := Tape.leaf (t := t0) (Harness.tensorToSomeBuffer x) (name := some "x")
  let (t2, outId) ← Harness.record t1 "sum" (Harness.sumAll s) (xId, ())

  -- The output cotangent must be allocated afresh on every pass. A pure constant allocation can
  -- be hoisted by Lean and then reused after sparse backward has retired its native storage.
  for pass in [0:8] do
    let seed : LibTorch.SomeBuffer := { s := Shape.scalar, buf := ← Buffer.fullIO 1 1.0 }
    let grads ← LibTorch.Tape.backwardSparse (t := t2) outId seed (fun id => id == xId)
    let dx ← match grads.get? xId with
      | some dx => pure dx
      | none => throw <| IO.userError s!"sparse backward pass {pass}: missing leaf gradient"
    let dx ← Harness.someBufferToTensor (s := s) dx
    let expected : Tensor Float s :=
      (Tensor.from #[1.0, 1.0, 1.0, 1.0]).reshape [4] (by dsimp; decide)
    Harness.assertTensorApprox (s := s) s!"sparse backward pass {pass}" dx expected
    LibTorch.Tape.releaseSparseGrads grads

  -- The test owns the tape, so it releases the forward values of the tape's nodes itself.
  for node in t2.nodes do
    discard <| Buffer.releaseIO node.value.buf
  let after ← Buffer.Memory.read
  if after.liveBytes > before.liveBytes then
    throw <| IO.userError
      s!"sparse backward ownership: live bytes grew from {before.liveBytes} to {after.liveBytes}"

/-- The pointwise kernels `add`, `mul`, `axpy`, `clamp` and `relu` on buffers of 200003 entries
against the host values, their sum and mean under strict determinism, and the binary32 edge cases
of the empty mean, `divCount`, `mseMean` and `mseSlope`. -/
def runLargeBufferBody : IO Unit := do
  IO.println "== large buffer elementwise/reduction stress =="

  let n : Nat := 200003
  let aHost := buildFloatArray n (fun i =>
    (((i % 97 : Nat) : Float) / 17.0) - 2.5)
  let bHost := buildFloatArray n (fun i =>
    ((((i * 7 + 3) % 101 : Nat) : Float) / 19.0) - 1.75)

  let aBuf ← Buffer.ofFloatArrayIO aHost
  let bBuf ← Buffer.ofFloatArrayIO bHost
  -- Several buffer kernels run here without the autograd tape, on low-level launch paths that the
  -- small tape tests can miss.
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
    let y := Max.max y (-3.5)
    let y := Min.min y 4.25
    if y > 0.0 then y else 0.0)
  assertFloatArrayApprox "large buffer pointwise pipeline" got expected (tol := 2e-5)

  let previousSettings : Option (Bool × Bool) ←
    match (← Buffer.RuntimeStatus.query) with
    | .nativeAvailable =>
        pure (some (← LibTorch.getDeterministic, ←
          LibTorch.getCuDNNBenchmark))
    | .notLinked => pure none
    | .nativeUnavailable =>
        throw <| IO.userError "large buffer stress requires a usable CUDA device"
  -- Strict determinism is requested through LibTorch for the sum and the mean, whose assertions
  -- keep the absolute tolerances `0.5` and `5e-4`, and the `finally` block restores the previous
  -- settings.
  try
    if previousSettings.isSome then
      LibTorch.setDeterministic true
    let sumGot := (Buffer.toFloatArray (Buffer.reduceSum relued)).get! 0
    let meanGot := (Buffer.toFloatArray (Buffer.reduceMean relued)).get! 0
    let mut sumExpected : Float := 0.0
    for i in [0:n] do
      sumExpected := sumExpected + expected.get! i
    let meanExpected : Float := sumExpected / (n : Float)

    Harness.assertApprox "large buffer reduceSum" sumGot sumExpected (tol := 0.5)
    Harness.assertApprox "large buffer reduceMean" meanGot meanExpected (tol := 5e-4)
  finally
    if let some (deterministic, benchmark) := previousSettings then
      LibTorch.setDeterministic deterministic
      LibTorch.setCuDNNBenchmark benchmark

  -- `Op.mean` divides an empty fiber by one: the empty mean is its zero sum.
  let emptyMean := Buffer.toFloatArray (Buffer.reduceMean (← Buffer.zerosIO 0))
  if emptyMean.size != 1 then
    throw <| IO.userError s!"reduceMean empty size: expected 1, got {emptyMean.size}"
  if (emptyMean.get! 0).toBits != 0 then
    throw <| IO.userError s!"reduceMean empty result: expected +0, got {emptyMean.get! 0}"
  -- One division by the count, not a product with a rounded reciprocal: binary32
  -- 147 * 2^-149 / 98 rounds to 2^-148, while 147 * 2^-149 * RN(1 / 98) rounds to 2^-149.
  let tiny ← Buffer.ofFloatArrayIO (FloatArray.mk #[Float.scaleB 147 (-149)]) .float32
  let quotient := (Buffer.toFloatArray (Buffer.divCount tiny 98)).get! 0
  if quotient != Float.scaleB 1 (-148) then
    throw <| IO.userError s!"divCount binary32 147 * 2^-149 / 98: expected 2^-148, got {quotient}"
  -- The mean squared error of binary32 `[2e19, 0, 0, 0]` is `1e38` although its squares overflow.
  -- The slope `2 d / 3` at `d = 3e38` is `2e38` rounded once, where `(d + d) / 3` is infinite, and
  -- a negative zero difference keeps its sign.
  let residuals ← Buffer.ofFloatArrayIO
    (FloatArray.mk #[(2e19 : Float32).toFloat, 0, 0, 0]) .float32
  let mse := (Buffer.toFloatArray (Buffer.mseMean residuals 4)).get! 0
  if mse.toFloat32.toBits != (1e38 : Float32).toBits then
    throw <| IO.userError s!"mseMean binary32 [2e19, 0, 0, 0]: expected 1e38, got {mse}"
  let large ← Buffer.ofFloatArrayIO (FloatArray.mk #[(3e38 : Float32).toFloat, -0.0]) .float32
  let slopes := Buffer.toFloatArray (Buffer.mseSlope large 3)
  if (slopes.get! 0).toFloat32.toBits != (2e38 : Float32).toBits ||
      (slopes.get! 1).toBits != (-0.0 : Float).toBits then
    throw <| IO.userError s!"mseSlope binary32 [3e38, -0] over 3: expected [2e38, -0], got {slopes}"

/-- Large-buffer pointwise and reduction checks under strict LibTorch determinism. -/
def runLargeBuffer : IO Unit := do
  match (← Buffer.RuntimeStatus.query) with
  | .notLinked => IO.println "  skipped: LibTorch is not linked"
  | .nativeAvailable => runLargeBufferBody
  | .nativeUnavailable => Buffer.requireNativeRuntime

/-- ATen matmul in float32 and float64 against `Tensor.matmul` on `[3, 4] × [4, 5]` and on
`[1, 7] × [7, 1]`, to `7e-3` and `1e-9`. -/
def runMatmul : IO Unit := do
  IO.println "== ATen matmul reference parity stress =="

  -- Preserve binary64 uploads and reject dimensions or result sizes outside the ABI contract.
  let matmul (dtype : LibTorch.Dtype) {m n p : Nat}
      (a : Tensor Float [m, n]) (b : Tensor Float [n, p]) :
      Result (Tensor Float [m, p]) := do
    let some rows := Bytes.toUInt32? m | throw SomeBuffer.tooLarge
    let some inner := Bytes.toUInt32? n | throw SomeBuffer.tooLarge
    let some columns := Bytes.toUInt32? p | throw SomeBuffer.tooLarge
    let upload {r c : Nat} (x : Tensor Float [r, c]) :=
      Buffer.ofFloatArray (x.to FloatArray) dtype
    let flat := Buffer.toFloatArray (Buffer.bmm (upload a) (upload b) 1 rows inner columns)
    match (Tensor.from flat).reshape? [m, p] with
    | some tensor => pure tensor
    | none =>
        throw s!"autograd: native matmul: ATen {repr dtype} bmm returned {flat.size} elements, \
          expected {m * p}"

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
  let yRef1 := Tensor.matmul (α := Float) (m := 3) (n := 4) (p := 5) a1 b1
  let yFp321 ← IO.ofExcept
    (matmul .float32 (m := 3) (n := 4) (p := 5) a1 b1)
  let yFp641 ← IO.ofExcept
    (matmul .float64 (m := 3) (n := 4) (p := 5) a1 b1)
  Harness.assertTensorApprox (s := sY1) "matmul stress case1 fp32" yFp321 yRef1 (tol := 7e-3)
  Harness.assertTensorApprox (s := sY1) "matmul stress case1 fp64" yFp641 yRef1 (tol := 1e-9)

  -- Dot-product-shaped case: small but asymmetric enough to exercise the degenerate leading
  -- dimensions of the float32 and float64 products.
  let sA2 : Shape := [1, 7]
  let sB2 : Shape := [7, 1]
  let sY2 : Shape := [1, 1]
  let a2 : Tensor Float sA2 :=
    (Tensor.from #[0.25, -0.50, 0.75, -1.00, 1.25, -1.50, 1.75]).reshape [1, 7] (by dsimp; decide)
  let b2 : Tensor Float sB2 :=
    (Tensor.from #[0.10, 0.20, -0.30, 0.40, -0.50, 0.60, -0.70]).reshape [7, 1] (by dsimp; decide)
  let yRef2 := Tensor.matmul (α := Float) (m := 1) (n := 7) (p := 1) a2 b2
  let yFp322 ← IO.ofExcept
    (matmul .float32 (m := 1) (n := 7) (p := 1) a2 b2)
  let yFp642 ← IO.ofExcept
    (matmul .float64 (m := 1) (n := 7) (p := 1) a2 b2)
  Harness.assertTensorApprox (s := sY2) "matmul stress case2 fp32" yFp322 yRef2 (tol := 7e-3)
  Harness.assertTensorApprox (s := sY2) "matmul stress case2 fp64" yFp642 yRef2 (tol := 1e-9)

/-- Check a real device readback, including its length and every uploaded/fill value. -/
def assertMemoryReadback (label : String) (buffer : Buffer) (n : Nat) (value : Float) :
    IO Unit := do
  let actual ← Buffer.toFloatArrayIO buffer
  if actual.size != n then
    throw <| IO.userError s!"{label}: expected {n} elements, got {actual.size}"
  for i in [0:n] do
    Harness.assertApprox s!"{label}[{i}]" (actual.get! i) value (tol := 1e-5)

/-- Accounting inequalities that hold for any cache block size or policy: allocated bytes at most
reserved bytes, each native peak at least its current value, and the logical peak at least the
live bytes. -/
def assertNativeAccounting (label : String) (stats : Buffer.Memory) : IO Unit := do
  if stats.allocatedBytes > stats.reservedBytes then
    throw <| IO.userError s!"{label}: allocated bytes exceed reserved bytes"
  if stats.peakAllocatedBytes < stats.allocatedBytes ||
      stats.peakReservedBytes < stats.reservedBytes then
    throw <| IO.userError s!"{label}: native peak is below current usage"
  if stats.peakBytes < stats.liveBytes then
    throw <| IO.userError s!"{label}: logical peak is below live ownership"

/-- Synchronize the selected device before taking a native allocator snapshot. -/
def synchronizedStats : IO Buffer.Memory := do
  LibTorch.synchronize
  Buffer.Memory.read

/-- Warm initialization outside the measured ownership interval. -/
@[noinline] def warmMemoryProbe : IO Unit := do
  let buffer ← Buffer.fullIO 4 2.0
  assertMemoryReadback "memory probe warmup" buffer 4 2.0
  discard <| Buffer.releaseIO buffer
  LibTorch.synchronize
  LibTorch.emptyCache

/-- Live allocations survive `emptyCache`, and released payloads leave the logical and native
allocated accounting. -/
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
  -- Other processes can change the driver's free bytes, so the snapshots are printed and no
  -- equality is asserted on them.
  IO.println s!"  before: {before.format}"
  IO.println s!"  held: {live.format}"
  IO.println s!"  released/cache emptied: {emptied.format}"

/--
Lean holds Q, K, V and the saved probabilities as its own buffers. With `releaseSaved` the pass
releases them, which must reclaim their storage and leave the output readable. Otherwise leaving
the scope must finalize every saved buffer.
-/
@[noinline] def runAttentionSavedIteration (releaseSaved : Bool) : IO Unit := do
  let before ← synchronizedStats
  let n : UInt32 := 64
  let d : UInt32 := 32
  let elements : UInt32 := n * d
  let bytes := UInt64.ofNat (elements.toNat * 4)
  let probabilityBytes := UInt64.ofNat (n.toNat * n.toNat * 4)
  let query ← Buffer.zerosIO elements
  let key ← Buffer.zerosIO elements
  let value ← Buffer.fullIO elements 2.0
  let outResult ← IO.lazyPure fun _ =>
    Buffer.attentionForward query key value none 1 n d 1.0
  let (out, probabilities) ← IO.ofExcept outResult
  assertMemoryReadback "attention forward" out elements.toNat 2.0
  assertMemoryReadback "attention saved probabilities" probabilities (n.toNat * n.toNat)
    (1.0 / Float.ofNat n.toNat)
  let retained ← synchronizedStats
  assertNativeAccounting "attention saved buffers" retained
  let savedBytes := 3 * bytes + probabilityBytes
  if retained.liveBytes != before.liveBytes + savedBytes + bytes then
    throw <| IO.userError "attention Q/K/V/probabilities/output payload accounting mismatch"
  if retained.allocatedBytes < before.allocatedBytes + savedBytes + bytes then
    throw <| IO.userError "attention native accounting omitted live saved buffers"
  LibTorch.emptyCache
  let afterEmpty ← synchronizedStats
  if afterEmpty.liveBytes != retained.liveBytes ||
      afterEmpty.allocatedBytes != retained.allocatedBytes then
    throw <| IO.userError "emptyCache released live attention buffers"
  assertMemoryReadback "attention after cache empty" out elements.toNat 2.0

  let upstream ← Buffer.fullIO elements 1.0
  let gradResult ← IO.lazyPure fun _ =>
    Buffer.attentionBackward query key value probabilities upstream 1 n d 1.0
  let (dq, dk, dv) ← IO.ofExcept gradResult
  -- Uniform attention, constant V, and unit output cotangent give dQ=dK=0, dV=1.
  assertMemoryReadback "attention saved dQ" dq elements.toNat 0.0
  assertMemoryReadback "attention saved dK" dk elements.toNat 0.0
  assertMemoryReadback "attention saved dV" dv elements.toNat 1.0
  for buffer in #[upstream, dq, dk, dv] do
    discard <| Buffer.releaseIO buffer
  if releaseSaved then
    let beforeRelease ← synchronizedStats
    for buffer in #[query, key, value, probabilities] do
      if (← Buffer.releaseIO buffer) != 1 then
        throw <| IO.userError "attention saved buffer release did not retire its payload"
    let outputOnly ← synchronizedStats
    assertNativeAccounting "attention output only" outputOnly
    if outputOnly.liveBytes != before.liveBytes + bytes ||
        outputOnly.allocatedBytes + savedBytes != beforeRelease.allocatedBytes then
      throw <| IO.userError "attention output retained hidden saved storage"
    assertMemoryReadback "attention output after saved buffer release" out elements.toNat 2.0
    if (← Buffer.releaseIO out) != 1 then
      throw <| IO.userError "attention output release did not retire its payload"
  else
    -- These final uses retain all buffers until this noinline IO scope finalizes their wrappers.
    for (buffer, size) in #[(query, elements), (key, elements), (value, elements),
        (probabilities, n * n), (out, elements)] do
      if (← Buffer.sizeIO buffer) != size then
        throw <| IO.userError "attention saved buffer disappeared before finalization"

/-- Both a release by `Buffer.releaseIO` and Lean's finalization reclaim the saved attention
buffers, returning the live and allocated bytes to their values before the pass. -/
def runAttentionMemoryProbe : IO Unit := do
  Buffer.requireNativeRuntime
  IO.println "== LibTorch saved attention buffer lifetime =="
  runAttentionSavedIteration true
  LibTorch.synchronize
  LibTorch.emptyCache
  let before ← synchronizedStats
  for releaseSaved in [true, false] do
    runAttentionSavedIteration releaseSaved
    let after ← synchronizedStats
    assertNativeAccounting "attention saved buffers retired" after
    if after.liveBytes != before.liveBytes || after.allocatedBytes != before.allocatedBytes then
      throw <| IO.userError
        s!"attention saved buffers leaked after releaseSaved={releaseSaved}: {after.format}"
    LibTorch.emptyCache
    let emptied ← synchronizedStats
    if emptied.allocatedBytes != before.allocatedBytes ||
        emptied.reservedBytes > after.reservedBytes then
      throw <| IO.userError "attention retirement/cache-empty accounting mismatch"
  IO.println "  explicit release and finalization both reclaimed saved buffers"

/-- A failed native allocation must raise the recoverable IO error `.resourceExhausted`, which the
process survives. -/
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

/-- With an allocator limit 32 MiB above the current reservation, oversized allocations fail
without leaking, an earlier buffer keeps its contents, and smaller allocations succeed in the same
process. The previous memory fraction is restored at the end. -/
def runMemoryOOMProbe : IO Unit := do
  Buffer.requireNativeRuntime
  IO.println "== LibTorch controlled OOM and recovery =="
  warmMemoryProbe
  let survivor ← Buffer.fullIO 1024 3.25
  assertMemoryReadback "OOM survivor before" survivor 1024 3.25
  LibTorch.emptyCache
  let before ← synchronizedStats
  let oldFraction ← LibTorch.getMemoryFraction
  -- The memory fraction sets an allocation limit 32 MiB above the current reservation and below
  -- half the device memory. The oversized requests, twice the limit, fit in the device and fail at
  -- the limit.
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
    Harness.assertApprox "allocator memory fraction readback" observed fraction (tol := 1e-9)
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
  Harness.assertApprox "memory fraction restored" (←
    LibTorch.getMemoryFraction) oldFraction
    (tol := 1e-9)
  let finalBuffer ← Buffer.fullIO 4096 9.0
  assertMemoryReadback "allocation after restoring limit" finalBuffer 4096 9.0
  discard <| Buffer.releaseIO finalBuffer
  IO.println "  OOM returned resourceExhausted; live inputs and subsequent allocations survived"

/--
Run each memory probe (accounting, attention buffers, OOM recovery) in a fresh process of the test
executable, so accounting excludes earlier suite work and allocator limits cannot affect other
tests. No assertion depends on an exact cached block size or amount returned to the driver. A build
without LibTorch must report zero native memory statistics, and without `/proc/self/exe` the probes
are skipped.
-/
def runMemoryTests : IO Unit := do
  IO.println "== LibTorch memory accounting and OOM (isolated processes) =="
  match (← Buffer.RuntimeStatus.query) with
  | .notLinked =>
      let stats ← Buffer.Memory.read
      if stats.allocatedBytes != 0 || stats.reservedBytes != 0 ||
          stats.peakAllocatedBytes != 0 || stats.peakReservedBytes != 0 ||
          stats.deviceFreeBytes != 0 || stats.deviceTotalBytes != 0 then
        throw <| IO.userError "build without LibTorch reported native CUDA memory accounting"
      IO.println "  skipped: LibTorch is not linked"
      return
  | .nativeUnavailable =>
      throw <| IO.userError "LibTorch memory tests require a usable CUDA device"
  | .nativeAvailable => pure ()
  let self : System.FilePath := "/proc/self/exe"
  if !(← self.pathExists) then
    IO.println "  skipped: isolated memory tests require Linux /proc/self/exe"
    return
  for probe in ["accounting", "attention-buffers", "oom-recovery"] do
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

/-- Run every stress check of this file. -/
def run : IO Unit := do
  IO.println "=== CUDA runtime stress suite ==="
  runRNG
  runInitializerParity
  runRelease
  runWrapperLifetime
  runMemoryTests
  runGradientAliasing
  runMalformedBuffers
  runDisconnectedDenseGradient
  runConstantBranchGradient
  runSparseLifetime
  runLargeBuffer
  runMatmul

end Stress
end CUDA
end TorchLean.Test
