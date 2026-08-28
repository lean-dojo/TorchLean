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
- explicit `Buffer.releaseIO` lifecycle semantics,
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

  let b ← Buffer.fullIO 8 3.25
  -- `releaseIO` is a lifetime hint for long eager loops: success means the allocation was freed and
  -- the wrapper was converted into an empty buffer so the finalizer remains safe.
  let r1 ← Buffer.releaseIO b
  if r1 != 1 then
    throw <| IO.userError s!"release first call: expected 1, got {r1}"
  if Buffer.size b != 0 then
    throw <| IO.userError s!"release size reset: expected 0, got {Buffer.size b}"

/--
Build `k` distinct length-`n` buffers and force their allocation immediately.

The fill value varies per `(salt, index)` for two reasons: within a call it stops the compiler from
hoisting one shared `full` out of the inner loop, and across calls a distinct `salt` keeps the whole
expression from being treated as loop-invariant (and hoisted out of the *caller's* loop) or CSE'd with
another call site — either of which would allocate the buffers once, outside the arena under test. The
returned element-count total is a forcing witness the caller checks.
-/
def buildArenaScratch (n : UInt32) (k : Nat) (salt : Nat) : Array Buffer × Nat :=
  Id.run do
    let mut held : Array Buffer := Array.mkEmpty k
    for i in [0:k] do
      held := held.push (Buffer.full n (1.0 + Float.ofNat (salt * k + i)))
    let mut touched : Nat := 0
    for b in held do
      touched := touched + (Buffer.size b).toNat
    return (held, touched)

def runArenaStress : IO Unit := do
  IO.println "== cuda arena scope stress =="

  let n : UInt32 := 4096
  let k : Nat := 64
  let blocks : Nat := 4
  let expectTouched := k * n.toNat
  let base ← Buffer.allocatorStats
  IO.println s!"  baseline:        {base.format}"

  -- Reclaim path: `k` buffers are built and held live (never released) inside an arena, kept alive
  -- across the scope exit, and reclaimed anyway. This is the case explicit `release` cannot reach: the
  -- buffers stay GC-reachable for the whole scope, so only the arena can free them.
  for blockIdx in [0:blocks] do
    let before ← Buffer.allocatorStatsWithToken (UInt32.ofNat blockIdx)
    Buffer.arenaEnter
    let (held, touched) := buildArenaScratch n k blockIdx
    if touched != expectTouched then
      throw <| IO.userError s!"arena: scratch build under-allocated ({touched} vs {expectTouched})"
    let inside ← Buffer.allocatorStatsWithToken (UInt32.ofNat (blockIdx + 100))
    if inside.allocCount != before.allocCount + UInt64.ofNat k then
      throw <| IO.userError
        s!"arena: expected {k} in-scope allocations ({before.allocCount} → {inside.allocCount})"
    if inside.freeCount != before.freeCount then
      throw <| IO.userError
        s!"arena: buffers freed before scope exit ({before.freeCount} → {inside.freeCount})"
    -- Keep nothing: every in-scope buffer is reclaimed even though `held` still references them all.
    Buffer.arenaExit #[]
    -- Touch `held` *after* the exit so it stays live across it: this proves the ARENA did the freeing,
    -- not reference-counted finalization (which cannot run while `held` is still referenced).
    let heldGuard := held.size
    let after ← Buffer.allocatorStatsWithToken (UInt32.ofNat (blockIdx + 200))
    if heldGuard != k then
      throw <| IO.userError s!"arena: held guard mismatch ({heldGuard} vs {k})"
    if after.freeCount != before.freeCount + UInt64.ofNat k then
      throw <| IO.userError
        s!"arena: scope exit freed {after.freeCount - before.freeCount}, expected {k} live buffers"
    if after.liveBytes != before.liveBytes then
      throw <| IO.userError
        s!"arena: live bytes not restored at scope exit ({before.liveBytes} → {after.liveBytes})"
  IO.println s!"  reclaimed {k} live in-scope buffers across {blocks} arenas"

  -- Promotion path: a kept buffer survives the scope and stays usable; the rest are reclaimed.
  let before ← Buffer.allocatorStats
  Buffer.arenaEnter
  let keeper := Buffer.full n 2.0
  -- Force `keeper`'s allocation inside the scope (so it is registered and then promoted on exit).
  if Buffer.size keeper != n then
    throw <| IO.userError "arena: keep-path keeper not allocated"
  let (scratch, scratchTouched) := buildArenaScratch n k blocks
  if scratchTouched != expectTouched then
    throw <| IO.userError "arena: keep-path scratch build under-allocated"
  Buffer.arenaExit #[keeper]
  let scratchGuard := scratch.size
  let after ← Buffer.allocatorStats
  if scratchGuard != k then
    throw <| IO.userError "arena: keep-path scratch guard mismatch"
  if after.freeCount != before.freeCount + UInt64.ofNat k then
    throw <| IO.userError
      s!"arena keep: expected {k} frees, got {after.freeCount - before.freeCount}"
  -- `keeper` was promoted out of the scope, so its data is intact: reducing it still sees `2.0`.
  let keptSum := (Buffer.toFloatArray (Buffer.reduceSum keeper)).get! 0
  let expectedSum := 2.0 * Float.ofNat n.toNat
  Utils.assertApprox "arena kept-buffer survives scope" keptSum expectedSum (tol := 1e-1)
  IO.println "  promoted buffer survived its arena"

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

  let s : Shape := [4]
  let x : Tensor Float s := tensorOfArray! [4] #[0.25, -0.50, 0.75, -1.00]

  let t0 : Cuda.Tape := Cuda.Tape.empty
  let (t1, xId) := Cuda.Tape.leaf (t := t0) (Utils.tensorToAnyBuffer x) (name := some "x")
  -- `x + x` sends the same upstream gradient to both parents of an add node. This checks
  -- accumulated-gradient aliasing in add nodes.
  let (t2, yId) ← Utils.okOrThrow (Cuda.Tape.add (t := t1) (s := s) xId xId)
  let (t3, outId) ← Utils.okOrThrow (Cuda.Tape.sum (t := t2) (s := s) yId)
  let seed : Cuda.AnyBuffer := { s := Shape.scalar, buf := Buffer.full 1 1.0 }
  let grads ← Utils.okOrThrow (Cuda.Tape.backwardDenseAll (t := t3) outId seed)
  let dx ← Utils.cudaGrad (s := s) grads xId
  let expected : Tensor Float s := tensorOfArray! [4] #[2.0, 2.0, 2.0, 2.0]
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
    { s := [UInt32.size], buf := Buffer.zeros 0 }
  match Cuda.AnyBuffer.validate oversized with
  | .ok _ =>
      throw <| IO.userError "AnyBuffer.validate accepted a shape whose numel exceeds UInt32"
  | .error _ => pure ()

  let hiddenOversizedAxis : Cuda.AnyBuffer :=
    { s := Shape.ofList [0, UInt32.size], buf := Buffer.zeros 0 }
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
    throw <| IO.userError s!"disconnected CUDA reciprocal output: expected finite zero, got {invGrad}"

def runSparseLifetimeStress : IO Unit := do
  IO.println "== repeated sparse-backward ownership =="

  let before ← Buffer.allocatorStatsWithToken 100
  let s : Shape := [4]
  let x : Tensor Float s := tensorOfArray! [4] #[0.25, -0.50, 0.75, -1.00]
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
    let expected : Tensor Float s := tensorOfArray! [4] #[1.0, 1.0, 1.0, 1.0]
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
  let sA1 : Shape := [3, 4]
  let sB1 : Shape := [4, 5]
  let sY1 : Shape := [3, 5]
  let a1 : Tensor Float sA1 :=
    tensorOfArray! [3, 4] #[
      0.10, -0.20, 0.30, -0.40,
      0.55, 0.65, -0.75, 0.85,
      -0.15, 0.25, -0.35, 0.45
    ]
  let b1 : Tensor Float sB1 :=
    tensorOfArray! [4, 5] #[
      0.20, -0.10, 0.05, 0.30, -0.40,
      -0.15, 0.25, -0.35, 0.45, 0.10,
      0.50, -0.60, 0.70, -0.80, 0.90,
      -0.05, 0.15, -0.25, 0.35, -0.45
    ]
  let yRef1 := FastKernels.matmulReference (α := Float) (m := 3) (n := 4) (p := 5) a1 b1
  let yFp321 := FastKernels.Cuda.matmulCublas .fp32 (m := 3) (n := 4) (p := 5) a1 b1
  let yFp641 := FastKernels.Cuda.matmulCublas .fp64 (m := 3) (n := 4) (p := 5) a1 b1
  Utils.assertTensorApprox (s := sY1) "matmul stress case1 fp32" yFp321 yRef1 (tol := 7e-3)
  Utils.assertTensorApprox (s := sY1) "matmul stress case1 fp64" yFp641 yRef1 (tol := 1e-9)

  -- Dot-product-shaped case: small but asymmetric enough to exercise the degenerate leading
  -- dimensions in the DGEMM bridge.
  let sA2 : Shape := [1, 7]
  let sB2 : Shape := [7, 1]
  let sY2 : Shape := [1, 1]
  let a2 : Tensor Float sA2 :=
    tensorOfArray! [1, 7] #[0.25, -0.50, 0.75, -1.00, 1.25, -1.50, 1.75]
  let b2 : Tensor Float sB2 :=
    tensorOfArray! [7, 1] #[0.10, 0.20, -0.30, 0.40, -0.50, 0.60, -0.70]
  let yRef2 := FastKernels.matmulReference (α := Float) (m := 1) (n := 7) (p := 1) a2 b2
  let yFp322 := FastKernels.Cuda.matmulCublas .fp32 (m := 1) (n := 7) (p := 1) a2 b2
  let yFp642 := FastKernels.Cuda.matmulCublas .fp64 (m := 1) (n := 7) (p := 1) a2 b2
  Utils.assertTensorApprox (s := sY2) "matmul stress case2 fp32" yFp322 yRef2 (tol := 7e-3)
  Utils.assertTensorApprox (s := sY2) "matmul stress case2 fp64" yFp642 yRef2 (tol := 1e-9)

/--
Build `k` freshly-allocated device buffers of length `n`, returned together with the total element
count touched. Reading the sizes back forces the allocations so Lean cannot drop them as dead code;
`salt` varies the fill value between callers so repeated blocks are distinguishable. -/
def buildCacheScratch (n : UInt32) (k : Nat) (salt : Nat) : Array Buffer × Nat :=
  Id.run do
    let mut held : Array Buffer := Array.mkEmpty k
    for i in [0:k] do
      held := held.push (Buffer.full n (1.0 + Float.ofNat (salt * k + i)))
    let mut touched : Nat := 0
    for b in held do
      touched := touched + (Buffer.size b).toNat
    return (held, touched)

/--
Block-cache byte-cap probe, the subject of `runCacheCapTest`. Runs in a forked child so the cap
(`TORCHLEAN_CUDA_CACHE_CAP_BYTES`, read once natively) is fixed before the first cache operation.

The child first checks that the native allocator reports the expected parsed cap. It then allocates
`k` same-size blocks and returns them through `Buffer.release`. The total returned (8 MiB here) far
exceeds the 1 MiB test cap. For a finite cap, the cache must remain below the cap and fill to within
one block of it. For the unbounded control, every returned block must remain cached.

Selected in a forked child by `TORCHLEAN_CUDA_CACHE_PROBE=cache-cap` (see `NN.Tests.run`). -/
def runCacheCapProbe : IO Unit := do
  IO.println "== cuda block-cache byte-cap probe =="
  let expectedCapBytes ←
    match (← IO.getEnv "TORCHLEAN_CUDA_CACHE_EXPECTED_CAP_BYTES").bind (·.toNat?) with
    | some n => pure (UInt64.ofNat n)
    | none => throw <| IO.userError "cache-cap probe: missing expected native cap"
  let n : UInt32 := 65536                              -- 256 KiB per block (float32)
  let blockBytes : UInt64 := UInt64.ofNat (n.toNat * 4)
  let k : Nat := 32                                    -- 8 MiB of returns, far past a 1 MiB cap
  let totalBytes : UInt64 := UInt64.ofNat (n.toNat * 4 * k)
  let pre ← Buffer.allocatorStats
  -- `deviceTotalBytes` comes from `cudaMemGetInfo`: nonzero on the CUDA build, 0 on the CPU stub.
  let onCuda : Bool := pre.deviceTotalBytes != 0
  if !onCuda then
    if pre.cacheBytes != 0 || pre.cacheCapBytes != 0 then
      throw <| IO.userError "cache-cap probe: CPU stub reported CUDA cache state"
    IO.println "  skipped: the CPU stub has no device reuse cache"
    return
  if pre.cacheCapBytes != expectedCapBytes then
    throw <| IO.userError
      s!"cache-cap probe: native cap is {pre.cacheCapBytes}, expected {expectedCapBytes}"
  let capBytes := pre.cacheCapBytes
  -- Fresh child: the cache starts empty, so every block is a real device alloc, not a cache reuse.
  let (held, touched) := buildCacheScratch n k 1
  if touched != k * n.toNat then
    throw <| IO.userError "cache-cap probe: scratch build under-allocated"
  -- Return every block to the cache. Under the cap, returns past the cap free instead of caching.
  let mut freed : Nat := 0
  for b in held do
    freed := freed + (← Buffer.releaseIO b).toNat
  if freed != k then
    throw <| IO.userError s!"cache-cap probe: expected {k} releases, got {freed}"
  let post ← Buffer.allocatorStats
  if post.cacheCapBytes != capBytes then
    throw <| IO.userError "cache-cap probe: native cap changed during the process"
  IO.println s!"  cap={capBytes} returned={totalBytes} cacheBytes={post.cacheBytes}"
  if capBytes == 0 then
    -- Control: no cap, so every returned block stays cached — the growth the cap bounds.
    if post.cacheBytes != totalBytes then
      throw <| IO.userError
        s!"cache-cap probe (control): uncapped cache held {post.cacheBytes}, expected {totalBytes}"
  else
    if post.cacheBytes > capBytes then
      throw <| IO.userError s!"cache-cap probe: cache exceeded cap ({post.cacheBytes} > {capBytes})"
    -- The cap is the binding constraint: the workload exceeds it, yet the cache fills to within one
    -- block of the limit instead of retaining the full 8 MiB.
    if totalBytes ≤ capBytes then
      throw <| IO.userError "cache-cap probe: workload did not exceed the cap (test misconfigured)"
    if post.cacheBytes < blockBytes then
      throw <| IO.userError s!"cache-cap probe: cache did not fill ({post.cacheBytes} < {blockBytes})"
    if post.cacheBytes + blockBytes ≤ capBytes then
      throw <| IO.userError
        s!"cache-cap probe: cache under-filled below the cap ({post.cacheBytes} + {blockBytes} ≤ {capBytes})"
  IO.println "  block-cache byte cap enforced ✓"

/--
Regression test for the device block-cache byte cap. The cap is read once natively, so it must be
fixed before the process's first cache operation; the test therefore forks the suite binary
(`/proc/self/exe`) per configuration (see `runCacheCapProbe`):

* **capped** — `TORCHLEAN_CUDA_CACHE_CAP_BYTES=1048576` bounds an 8 MiB return workload to a 1 MiB
  cache;
* **control** — `TORCHLEAN_CUDA_CACHE_CAP_BYTES=0` (explicitly unbounded), so the same workload
  caches the full 8 MiB (the unbounded growth the cap fixes);
* **malformed** — `TORCHLEAN_CUDA_CACHE_CAP_BYTES=1MiB` is rejected by the strict native parser,
  so the cache stays unbounded and behaves exactly like the control. This pins the rejection: a
  prefix-parsing reader would instead take the leading `1` as a one-byte cap and cache nothing;
* **overflow** — a value past the native word size is likewise rejected rather than truncated or
  saturated, so the cache again behaves like the control.

All four children pass the cap explicitly, so none inherits a stray
`TORCHLEAN_CUDA_CACHE_CAP_BYTES` from the parent environment. In particular, the control child is
pinned to `0` rather than inheriting a cap that would mask the uncapped growth it is meant to
observe.

Each child asserts internally and exits non-zero on failure. The test is CUDA- and Linux-only; the
CPU stub has no reuse cache, and the subprocess launch uses `/proc/self/exe`. -/
def runCacheCapTest : IO Unit := do
  IO.println "== cuda block-cache byte-cap (fork test) =="
  let stats ← Buffer.allocatorStats
  if stats.deviceTotalBytes == 0 then
    IO.println "  skipped: the CPU stub has no device reuse cache"
    return
  let self : System.FilePath := "/proc/self/exe"
  if !(← self.pathExists) then
    IO.println "  skipped: no /proc/self/exe (fork test is Linux-only)"
    return
  -- Always set the cap explicitly in the child's environment so it never inherits the parent's
  -- `TORCHLEAN_CUDA_CACHE_CAP_BYTES`; the control run pins it to "0" (unbounded) rather than unset.
  let fork (cap : String) (expectedCap : UInt64) : IO IO.Process.Output := do
    let env := #[
      ("TORCHLEAN_CUDA_CACHE_PROBE", some "cache-cap"),
      ("TORCHLEAN_CUDA_CACHE_CAP_BYTES", some cap),
      ("TORCHLEAN_CUDA_CACHE_EXPECTED_CAP_BYTES", some (toString expectedCap))
    ]
    IO.Process.output { cmd := self.toString, args := #[], env := env }
  -- capped: a 1 MiB cap bounds 8 MiB of returns.
  let capped ← fork "1048576" 1048576
  if capped.exitCode != 0 then
    throw <| IO.userError
      s!"block-cache cap: capped child failed (exit {capped.exitCode}); stderr:\n{capped.stderr}"
  IO.println "  capped: 8 MiB of returns bounded to a 1 MiB cache ✓"
  -- control: cap explicitly 0 (unbounded), so the same returns all stay cached, the behaviour the
  -- cap exists to bound; the explicit 0 keeps a parent-set cap from masking it.
  let control ← fork "0" 0
  if control.exitCode != 0 then
    throw <| IO.userError
      s!"block-cache cap: control child failed (exit {control.exitCode}); stderr:\n{control.stderr}"
  IO.println "  control: with no cap the full workload is cached, as designed ✓"
  -- malformed: rejected outright by the strict parser, so the cache stays unbounded. A
  -- prefix-parsing reader would take the leading "1" as a one-byte cap and cache nothing; the
  -- child asserts the control (fully cached) outcome instead.
  let malformed ← fork "1MiB" 0
  if malformed.exitCode != 0 then
    throw <| IO.userError
      s!"block-cache cap: malformed-value child failed (exit {malformed.exitCode}); stderr:\n{malformed.stderr}"
  IO.println "  malformed: non-numeric cap rejected, cache stays unbounded ✓"
  -- overflow: past the native word, rejected rather than truncated or saturated.
  let overflow ← fork "99999999999999999999999999" 0
  if overflow.exitCode != 0 then
    throw <| IO.userError
      s!"block-cache cap: overflow-value child failed (exit {overflow.exitCode}); stderr:\n{overflow.stderr}"
  IO.println "  overflow: oversized cap rejected, cache stays unbounded ✓"

/--
Planted use-after-free, the subject of `runArenaDetectorDeathTest`. Allocates two buffers inside an
arena, reclaims **both** at `arenaExit`, then uses them in an op — the same hazard as touching a
`release`d buffer. Because reclaimed buffers have `size == 0`, the bare size check (`0 == 0`) lets this
slip through to a launch on freed memory; only the detector (`TORCHLEAN_ARENA_DEBUG=1`) catches it,
naming the epoch. Run only in a forked child (selected by `TORCHLEAN_ARENA_UAF_PROBE=uaf`): under the
detector it `panic`s (which is why it must be forked, not asserted in-process); with the detector off it
returns a silently-wrong size-0 result — exactly the silent corruption the detector closes. -/
def runArenaUseAfterFreeProbe : IO Unit := do
  IO.println "== cuda arena use-after-free probe =="
  let n : UInt32 := 16
  Buffer.arenaEnter
  let a := Buffer.full n 3.0
  let b := Buffer.full n 5.0
  let forced := Buffer.size a + Buffer.size b          -- force both allocations inside the epoch
  if forced != 2 * n then
    throw <| IO.userError "uaf probe: operands not allocated"
  Buffer.arenaExit #[]                                  -- reclaim BOTH (keep nothing)
  -- `a` and `b` are now reclaimed (size 0). The detector asserts liveness before the size check and
  -- panics here; with the detector off, `add` slips past `0 == 0` and yields a silently-empty buffer.
  let bad := Buffer.add a b
  IO.println s!"  detector OFF: use-after-free slipped through, result size = {Buffer.size bad} (expected 16)"

/--
Valid arena promotion, the negative-case subject of `runArenaDetectorDeathTest`. Promotes one buffer
past the scope and reclaims another, then uses the *promoted* buffer in a **binary** op — which flows
through the same `require_same_size2` choke point the detector guards. A promoted buffer is live
(`arena_freed_depth == 0`), so the detector must not fire and the result must be correct. Selected in a
forked child by `TORCHLEAN_ARENA_UAF_PROBE=valid`. -/
def runArenaValidPromotionProbe : IO Unit := do
  IO.println "== cuda arena valid-promotion probe =="
  let n : UInt32 := 16
  Buffer.arenaEnter
  let keep := Buffer.full n 2.0
  let scratch := Buffer.full n 7.0
  let forced := Buffer.size keep + Buffer.size scratch  -- force both allocations inside the epoch
  if forced != 2 * n then
    throw <| IO.userError "valid-promotion probe: operands not allocated"
  Buffer.arenaExit #[keep]                              -- promote `keep`; reclaim `scratch`
  -- `keep` is promoted (live). A binary op on it exercises the detector's choke point and must pass.
  let ok := Buffer.add keep keep
  let s := (Buffer.toFloatArray (Buffer.reduceSum ok)).get! 0
  if s != 4.0 * Float.ofNat n.toNat then
    throw <| IO.userError s!"valid-promotion probe: wrong result {s} (expected {4.0 * Float.ofNat n.toNat})"
  IO.println s!"  promoted buffer reused in a binary op, result sum = {s}"

/--
Positive + negative regression test for the arena use-after-free detector. A detected UAF must `panic`
(it cannot be caught in-process), so the suite binary is forked in each configuration via
`/proc/self/exe` and its outcome inspected:

* **positive** — `TORCHLEAN_ARENA_DEBUG=1` + the planted UAF ⇒ the child aborts with a
  `use-after-arena-free` panic;
* **negative (no false positive)** — detector on + a *valid* arena promotion
  (`runArenaValidPromotionProbe`, which reuses a promoted buffer in a binary op through the detector's
  own choke point) ⇒ the child exits cleanly, so the detector never fires on a kept buffer;
* **control** — detector off + the planted UAF ⇒ the child exits cleanly (the UAF slips through, the
  silent corruption the detector closes).

The forked children re-enter the suite with `TORCHLEAN_ARENA_UAF_PROBE` set (see `NN.Tests.run`) and so
run only the relevant fragment. Linux-only (uses `/proc/self/exe`); skipped with a note elsewhere. -/
def runArenaDetectorDeathTest : IO Unit := do
  IO.println "== cuda arena use-after-free detector (fork death test) =="
  let self : System.FilePath := "/proc/self/exe"
  if !(← self.pathExists) then
    IO.println "  skipped: no /proc/self/exe (fork death test is Linux-only)"
    return
  let contains (hay needle : String) : Bool := (hay.splitOn needle).length ≥ 2
  let fork (debug : Bool) (mode : String) : IO IO.Process.Output := do
    let env := #[("TORCHLEAN_ARENA_UAF_PROBE", some mode)]
    let env := if debug then env.push ("TORCHLEAN_ARENA_DEBUG", some "1") else env
    IO.Process.output { cmd := self.toString, args := #[], env := env }
  -- positive: the detector aborts the planted use-after-free, naming the hazard.
  let pos ← fork true "uaf"
  if pos.exitCode == 0 then
    throw <| IO.userError "arena UAF detector: detector ON did NOT abort the planted use-after-free"
  if !(contains (pos.stderr ++ pos.stdout) "use-after-arena-free") then
    throw <| IO.userError s!"arena UAF detector: detector ON aborted without the expected message; stderr:\n{pos.stderr}"
  IO.println "  positive: detector ON aborts the planted use-after-free ✓"
  -- negative: a valid promotion under the detector is left untouched (no false positive).
  let neg ← fork true "valid"
  if neg.exitCode != 0 then
    throw <| IO.userError s!"arena UAF detector: false positive on a valid promotion (exit {neg.exitCode}); stderr:\n{neg.stderr}"
  IO.println "  negative: detector ON leaves a valid arena promotion untouched ✓"
  -- control: with the detector off the same use-after-free slips through and the child exits cleanly.
  let off ← fork false "uaf"
  if off.exitCode != 0 then
    throw <| IO.userError s!"arena UAF detector: with the detector off the probe should exit cleanly (exit {off.exitCode})"
  IO.println "  control: detector OFF leaves the use-after-free undetected, as designed ✓"

def run : IO Unit := do
  IO.println "=== CUDA runtime stress suite ==="
  runRngStress
  runReleaseStress
  runArenaStress
  runArenaDetectorDeathTest
  runWrapperLifetimeStress
  runCacheCapTest
  runGradientAliasingStress
  runMalformedBufferValidationStress
  runDisconnectedDenseGradientStress
  runSparseLifetimeStress
  runLargeBufferStress
  runMatmulStress

end Stress
end Cuda
end Tests
