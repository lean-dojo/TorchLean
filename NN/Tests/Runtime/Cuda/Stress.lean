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
public import NN.Entrypoint.Tensor
public import NN.Tests.Runtime.Cuda.Utils

/-!
# CUDA Runtime Stress Tests

Low-level stress coverage that goes beyond the small eager-tape tests:

- exact/deterministic RNG behavior for `randUniform` and `bernoulliMask`,
- explicit `Buffer.release` lifecycle semantics,
- large-buffer elementwise/reduction checks on direct `Cuda.Buffer` ops,
- extra cuBLAS matmul parity checks on rectangular inputs.

These still run without a GPU because the CUDA externs fall back to the CPU stub under the default
build, but when built with `-K cuda=true` they exercise the actual CUDA runtime paths.
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
    let u01 := expectedUniformValue key i
    if keepProb > u01 then 1.0 else 0.0)

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

def runGradientAliasingStress : IO Unit := do
  IO.println "== CUDA tape gradient aliasing regression =="

  let s : Shape := shape![4]
  let x : Tensor Float s := tensorND! [4] [0.25, -0.50, 0.75, -1.00]

  let t0 : Cuda.Tape := Cuda.Tape.empty
  let (t1, xId) := Cuda.Tape.leaf (t := t0) (Utils.tensorToAnyBuffer x) (name := some "x")
  -- `x + x` sends the same upstream gradient to both parents of an add node. This checks
  -- accumulated-gradient aliasing in add nodes.
  let (t2, yId) ← Utils.okOrThrow (Cuda.Tape.add (t := t1) (s := s) xId xId)
  let (t3, outId) ← Utils.okOrThrow (Cuda.Tape.sum (t := t2) (s := s) yId)
  let seed : Cuda.AnyBuffer := { s := Shape.scalar, buf := Buffer.full 1 1.0 }
  let grads ← Utils.okOrThrow (Cuda.Tape.backwardDenseAll (t := t3) outId seed)
  let dx ← Utils.cudaGrad (s := s) grads xId
  let expected : Tensor Float s := tensorND! [4] [2.0, 2.0, 2.0, 2.0]
  Utils.assertTensorApprox (s := s) "add backward duplicate-parent gradient" dx expected

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

  -- Empty mean is intentionally NaN on the runtime path; keep that contract explicit.
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
    tensorND! [3, 4] [
      0.10, -0.20, 0.30, -0.40,
      0.55, 0.65, -0.75, 0.85,
      -0.15, 0.25, -0.35, 0.45
    ]
  let b1 : Tensor Float sB1 :=
    tensorND! [4, 5] [
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
    tensorND! [1, 7] [0.25, -0.50, 0.75, -1.00, 1.25, -1.50, 1.75]
  let b2 : Tensor Float sB2 :=
    tensorND! [7, 1] [0.10, 0.20, -0.30, 0.40, -0.50, 0.60, -0.70]
  let yRef2 := FastKernels.matmulForward (α := Float) (m := 1) (n := 7) (p := 1) a2 b2
  let yFp322 := FastKernels.Cuda.matmulForwardcuBLASWith .fp32 (m := 1) (n := 7) (p := 1) a2 b2
  let yFp642 := FastKernels.Cuda.matmulForwardcuBLASWith .fp64 (m := 1) (n := 7) (p := 1) a2 b2
  Utils.assertTensorApprox (s := sY2) "matmul stress case2 fp32" yFp322 yRef2 (tol := 7e-3)
  Utils.assertTensorApprox (s := sY2) "matmul stress case2 fp64" yFp642 yRef2 (tol := 1e-9)

def run : IO Unit := do
  IO.println "=== CUDA runtime stress suite ==="
  runRngStress
  runReleaseStress
  runGradientAliasingStress
  runLargeBufferStress
  runMatmulStress

end Stress
end Cuda
end Tests
