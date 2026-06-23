/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Engine.Cuda.Buffer
public import NN.Runtime.Autograd.Engine.Cuda.Ops
public import NN.Runtime.Autograd.Engine.FastKernels
public import NN.Runtime.Autograd.TorchLean.Random
public import NN.Tensor
public import NN.Tests.Runtime.Cuda.Utils

/-!
# CUDA Runtime Stress Tests

Low-level stress coverage that goes beyond the small eager-tape tests:

- reference/deterministic RNG behavior for `randUniform`, `randNormal`, and `bernoulliMask`,
- explicit `Buffer.release` lifecycle semantics,
- finalization of short-lived external buffer wrappers,
- large-buffer elementwise/reduction checks on direct `Cuda.Buffer` ops,
- extra cuBLAS matmul parity checks on rectangular inputs.

These still run without a GPU because the CUDA externs fall back to the CPU stub under the default
build. With `-K cuda=true`, the same tests hit the real CUDA runtime paths.
-/

@[expose] public section

namespace Tests
namespace Cuda
namespace Stress

open Runtime.Autograd
open Runtime.Autograd.Cuda
open Runtime.Autograd.TorchLean
open Spec
open Tensor

def buildFloatArray (n : Nat) (f : Nat → Float) : FloatArray :=
  Id.run do
    let mut out : Array Float := Array.mkEmpty n
    for i in [0:n] do
      out := out.push (f i)
    return FloatArray.mk out

def assertFloatArrayEq (msg : String) (a b : FloatArray) : IO Unit := do
  if a.size != b.size then
    throw <| IO.userError s!"{msg}: size mismatch ({a.size} vs {b.size})"
  for i in [:a.size] do
    let x := a.get! i
    let y := b.get! i
    if x != y then
      throw <| IO.userError s!"{msg}[{i}]: got {x}, expected {y}"

def assertFloatArrayApprox (msg : String) (a b : FloatArray) (tol : Float := 1e-5) : IO Unit := do
  if a.size != b.size then
    throw <| IO.userError s!"{msg}: size mismatch ({a.size} vs {b.size})"
  for i in [:a.size] do
    Utils.assertApprox s!"{msg}[{i}]" (a.get! i) (b.get! i) tol

def expectedUniformValue (key : UInt64) (i : Nat) : Float :=
  let z := Random.splitmix64 (key + UInt64.ofNat i)
  -- This is the cross-backend contract: native CUDA, the CPU stub, and pure Lean all use
  -- `splitmix64(key + i) mod 2^32`, i.e. the low 32 bits.
  let u : Nat := z.toUInt32.toNat
  (u : Float) / (((2 : Nat) ^ 32 : Nat) : Float)

def expectedUniformArray (n : Nat) (key : UInt64) : FloatArray :=
  buildFloatArray n (fun i => expectedUniformValue key i)

def expectedBernoulliArray (n : Nat) (keepProb : Float) (key : UInt64) : FloatArray :=
  buildFloatArray n (fun i =>
    let unitUniform := expectedUniformValue key i
    if keepProb > unitUniform then 1.0 else 0.0)

def expectedNormalValue (mean std : Float) (key : UInt64) (i : Nat) : Float :=
  let r1 := (Random.splitmix64 (key + UInt64.ofNat (2 * i))).toUInt32.toNat
  let r2 := (Random.splitmix64 (key + UInt64.ofNat (2 * i + 1))).toUInt32.toNat
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

  let key : UInt64 := 0x123456789abcdef
  let nSmall : Nat := 64
  let nLarge : Nat := 4096

  -- Exact prefix checks catch low-bits versus high-bits SplitMix64 mismatches between CPU-stub and
  -- CUDA seeded buffers.
  let uSmall := Buffer.toFloatArray (Buffer.randUniform (UInt32.ofNat nSmall) key)
  let uExpected := expectedUniformArray nSmall key
  assertFloatArrayApprox "randUniform exact prefix" uSmall uExpected (tol := 1e-7)

  -- Repeated larger buffers are a cheap stress path for launch coverage and deterministic replay.
  let uLarge1 := Buffer.toFloatArray (Buffer.randUniform (UInt32.ofNat nLarge) key)
  let uLarge2 := Buffer.toFloatArray (Buffer.randUniform (UInt32.ofNat nLarge) key)
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
  let mSmall := Buffer.toFloatArray (Buffer.bernoulliMask (UInt32.ofNat nSmall) keepProb key)
  let mExpected := expectedBernoulliArray nSmall keepProb key
  assertFloatArrayEq "bernoulliMask exact prefix" mSmall mExpected

  let mLarge1 := Buffer.toFloatArray (Buffer.bernoulliMask (UInt32.ofNat nLarge) keepProb key)
  let mLarge2 := Buffer.toFloatArray (Buffer.bernoulliMask (UInt32.ofNat nLarge) keepProb key)
  assertFloatArrayEq "bernoulliMask deterministic repeat" mLarge1 mLarge2

def runReleaseStress : IO Unit := do
  IO.println "== explicit release semantics =="

  let b := Buffer.full 8 3.25
  -- `release` is a lifetime hint for long eager loops: success means the allocation was freed and
  -- the wrapper was converted into an empty buffer so the finalizer remains safe.
  let r1 := Buffer.release b
  if r1 != 1 then
    throw <| IO.userError s!"release first call: expected 1, got {r1}"
  if Buffer.size b != 0 then
    throw <| IO.userError s!"release size reset: expected 0, got {Buffer.size b}"

@[noinline] def runWrapperLifetimeIteration (i : Nat) : IO Unit := do
  let host := FloatArray.mk #[i.toFloat, 2.0, -3.0, 4.0]
  let a ← Buffer.ofFloatArrayIO host
  let doubled := Buffer.add a a
  let got ← Buffer.toFloatArrayIO doubled
  Utils.assertApprox "wrapper lifetime result" (got.get! 0) (2.0 * i.toFloat)

def runWrapperLifetimeStress : IO Unit := do
  IO.println "== external buffer wrapper lifetime =="

  let before ← Buffer.allocatorStatsWithToken 110
  for i in [0:4096] do
    runWrapperLifetimeIteration i
  let after ← Buffer.allocatorStatsWithToken 111

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

  let s : Shape := shape![4]
  let x : Tensor Float s := tensorOfList! [4] [0.25, -0.50, 0.75, -1.00]

  let t0 : Cuda.Tape := Cuda.Tape.empty
  let (t1, xId) := Cuda.Tape.leaf (t := t0) (Utils.tensorToAnyBuffer x) (name := some "x")
  -- `x + x` sends the same upstream gradient to both parents of an add node. This checks
  -- accumulated-gradient aliasing in add nodes.
  let (t2, yId) ← Utils.okOrThrow (Cuda.Tape.add (t := t1) (s := s) xId xId)
  let (t3, outId) ← Utils.okOrThrow (Cuda.Tape.sum (t := t2) (s := s) yId)
  let seed : Cuda.AnyBuffer := { s := Shape.scalar, buf := Buffer.full 1 1.0 }
  let grads ← Utils.okOrThrow (Cuda.Tape.backwardDenseAll (t := t3) outId seed)
  let dx ← Utils.cudaGrad (s := s) grads xId
  let expected : Tensor Float s := tensorOfList! [4] [2.0, 2.0, 2.0, 2.0]
  Utils.assertTensorApprox (s := s) "add backward duplicate-parent gradient" dx expected

def runSparseLifetimeStress : IO Unit := do
  IO.println "== repeated sparse-backward ownership =="

  let before ← Buffer.allocatorStatsWithToken 100
  let s : Shape := shape![4]
  let x : Tensor Float s := tensorOfList! [4] [0.25, -0.50, 0.75, -1.00]
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
    let expected : Tensor Float s := tensorOfList! [4] [1.0, 1.0, 1.0, 1.0]
    Utils.assertTensorApprox (s := s) s!"sparse backward pass {pass}" dx expected
    Cuda.Tape.releaseSparseGrads grads

  -- This test owns the tape and therefore retires its persistent forward values explicitly.
  for node in t2.nodes do
    discard <| Buffer.releaseIO node.value.buf
  let after ← Buffer.allocatorStatsWithToken 101
  if after.liveBytes > before.liveBytes then
    throw <| IO.userError
      s!"sparse backward ownership: live bytes grew from {before.liveBytes} to {after.liveBytes}"

def runLargeBufferStress : IO Unit := do
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

  let prevDet := Buffer.getDeterministicReductions
  -- Force the fixed-order path while comparing against a host accumulation. The fast atomic path is
  -- valid but may differ by normal floating-point associativity noise.
  let observedDet := Buffer.setDeterministicReductionsChecked true
  if !observedDet then
    throw <| IO.userError "failed to enable deterministic reductions for stress test"

  let sumGot := (Buffer.toFloatArray (Buffer.reduceSum relued)).get! 0
  let meanGot := (Buffer.toFloatArray (Buffer.reduceMean relued)).get! 0
  let mut sumExpected : Float := 0.0
  for i in [0:n] do
    sumExpected := sumExpected + expected.get! i
  let meanExpected : Float := sumExpected / (n : Float)

  Utils.assertApprox "large buffer reduceSum" sumGot sumExpected (tol := 0.5)
  Utils.assertApprox "large buffer reduceMean" meanGot meanExpected (tol := 5e-4)

  let _ := Buffer.setDeterministicReductionsChecked prevDet

  -- The runtime contract for an empty mean is `NaN`; keep that edge case explicit.
  let emptyMean := Buffer.toFloatArray (Buffer.reduceMean (Buffer.zeros 0))
  if emptyMean.size != 1 then
    throw <| IO.userError s!"reduceMean empty size: expected 1, got {emptyMean.size}"
  assertFloatIsNaN "reduceMean empty result" (emptyMean.get! 0)

def runMatmulStress : IO Unit := do
  IO.println "== cuBLAS matmul parity stress =="

  -- Rectangular case: catches row-major/column-major leading-dimension mistakes that square
  -- matrices can accidentally hide.
  let sA1 : Shape := shape![3, 4]
  let sB1 : Shape := shape![4, 5]
  let sY1 : Shape := shape![3, 5]
  let a1 : Tensor Float sA1 :=
    tensorOfList! [3, 4] [
      0.10, -0.20, 0.30, -0.40,
      0.55, 0.65, -0.75, 0.85,
      -0.15, 0.25, -0.35, 0.45
    ]
  let b1 : Tensor Float sB1 :=
    tensorOfList! [4, 5] [
      0.20, -0.10, 0.05, 0.30, -0.40,
      -0.15, 0.25, -0.35, 0.45, 0.10,
      0.50, -0.60, 0.70, -0.80, 0.90,
      -0.05, 0.15, -0.25, 0.35, -0.45
    ]
  let yRef1 := FastKernels.matmulForward (α := Float) (m := 3) (n := 4) (p := 5) a1 b1
  let yFp321 := FastKernels.Cuda.matmulForwardcuBLASWith .fp32 (m := 3) (n := 4) (p := 5) a1 b1
  let yFp641 := FastKernels.Cuda.matmulForwardcuBLASWith .fp64 (m := 3) (n := 4) (p := 5) a1 b1
  Utils.assertTensorApprox (s := sY1) "matmul stress case1 fp32" yFp321 yRef1 (tol := 7e-3)
  Utils.assertTensorApprox (s := sY1) "matmul stress case1 fp64" yFp641 yRef1 (tol := 1e-9)

  -- Dot-product-shaped case: small but asymmetric enough to exercise the degenerate leading
  -- dimensions in the DGEMM bridge.
  let sA2 : Shape := shape![1, 7]
  let sB2 : Shape := shape![7, 1]
  let sY2 : Shape := shape![1, 1]
  let a2 : Tensor Float sA2 :=
    tensorOfList! [1, 7] [0.25, -0.50, 0.75, -1.00, 1.25, -1.50, 1.75]
  let b2 : Tensor Float sB2 :=
    tensorOfList! [7, 1] [0.10, 0.20, -0.30, 0.40, -0.50, 0.60, -0.70]
  let yRef2 := FastKernels.matmulForward (α := Float) (m := 1) (n := 7) (p := 1) a2 b2
  let yFp322 := FastKernels.Cuda.matmulForwardcuBLASWith .fp32 (m := 1) (n := 7) (p := 1) a2 b2
  let yFp642 := FastKernels.Cuda.matmulForwardcuBLASWith .fp64 (m := 1) (n := 7) (p := 1) a2 b2
  Utils.assertTensorApprox (s := sY2) "matmul stress case2 fp32" yFp322 yRef2 (tol := 7e-3)
  Utils.assertTensorApprox (s := sY2) "matmul stress case2 fp64" yFp642 yRef2 (tol := 1e-9)

/--
One eager workload step for the leak-bound stress: allocate a short chain of device buffers, then
free every one of them through the explicit `Buffer.release` discipline that long CUDA training
loops use so they do not wait on Lean external-object finalizers.

Returns the number of live allocations actually freed (the sum of the release return codes). Reading
the codes back serves two purposes: it confirms each free found a live allocation, and it keeps Lean
from eliminating the release calls as dead code (each call is otherwise a pure `UInt32`-valued op).
-/
def leakStep (n : UInt32) : IO Nat := do
  let a := Buffer.full n 1.5
  let b := Buffer.full n (-0.5)
  let c := Buffer.add a b
  let d := Buffer.mul c a
  let s := Buffer.reduceSum d
  let freed :=
    Buffer.release a + Buffer.release b + Buffer.release c + Buffer.release d + Buffer.release s
  return freed.toNat

def runLeakBoundStress : IO Unit := do
  IO.println "== allocator leak-bound stress =="

  -- A small eager workload repeated in two equal-length blocks. The release discipline frees every
  -- buffer each step, so a correct runtime keeps the working set flat regardless of step count; a
  -- per-step leak instead grows the working set linearly and fails the cross-block checks below.
  let n : UInt32 := 4096
  let releasesPerStep : Nat := 5
  let block : Nat := 128

  let base ← Buffer.allocatorStats
  IO.println s!"  baseline:        {base.format}"

  let mut freed1 : Nat := 0
  for _ in [0:block] do
    freed1 := freed1 + (← leakStep n)
  let afterK ← Buffer.allocatorStatsWithToken (UInt32.ofNat block)
  IO.println s!"  after {block} steps:  {afterK.format}"

  let mut freed2 : Nat := 0
  for _ in [0:block] do
    freed2 := freed2 + (← leakStep n)
  let after2K ← Buffer.allocatorStatsWithToken (UInt32.ofNat (2 * block))
  IO.println s!"  after {2 * block} steps:  {after2K.format}"

  -- (1) The release discipline fired: each step freed exactly its `releasesPerStep` live
  --     allocations (and Lean did not drop the frees as dead code).
  let expectedFreed := releasesPerStep * block
  if freed1 != expectedFreed then
    throw <| IO.userError s!"leak-bound: first block freed {freed1}, expected {expectedFreed}"
  if freed2 != expectedFreed then
    throw <| IO.userError s!"leak-bound: second block freed {freed2}, expected {expectedFreed}"

  -- (2) The working set does not grow with the number of steps: net live allocations and live bytes
  --     after twice as many steps match the single-block snapshot. This is the leak-bound invariant.
  let netK : UInt64 := afterK.allocCount - afterK.freeCount
  let net2K : UInt64 := after2K.allocCount - after2K.freeCount
  if net2K != netK then
    throw <| IO.userError s!"leak-bound: net live allocations grew with step count ({netK} → {net2K})"
  if after2K.liveBytes != afterK.liveBytes then
    throw <| IO.userError
      s!"leak-bound: live bytes grew with step count ({afterK.liveBytes} → {after2K.liveBytes})"

  -- (3) With every buffer released, the loop returns to the baseline working set — no residual.
  let netBase : UInt64 := base.allocCount - base.freeCount
  if netK != netBase then
    throw <| IO.userError
      s!"leak-bound: live allocations did not return to baseline ({netBase} → {netK})"

def run : IO Unit := do
  IO.println "=== CUDA runtime stress suite ==="
  runRngStress
  runReleaseStress
  runWrapperLifetimeStress
  runLeakBoundStress
  runGradientAliasingStress
  runSparseLifetimeStress
  runLargeBufferStress
  runMatmulStress

end Stress
end Cuda
end Tests
