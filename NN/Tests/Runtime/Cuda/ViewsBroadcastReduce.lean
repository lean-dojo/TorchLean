/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Engine.Core
public import NN.Runtime.Autograd.Engine.Cuda.Ops
public import NN.Entrypoint.Tensor
public import NN.Tests.Runtime.Cuda.Utils

/-!
# CUDA Kernel Coverage: Views, Broadcast, Reduce

Small forward/backward comparisons (CPU tape vs CUDA tape) for:
- `reshape`, `transpose2d`, `swapAdjacentAtDepth`, 3D permutations
- `broadcastTo`
- `reduce_sum`, `reduce_mean`, low-level empty-axis `reduce_max` parity
-/

@[expose] public section

namespace Tests
namespace Cuda
namespace ViewsBroadcastReduce

open Spec
open Tensor
open Runtime.Autograd

def assertFloatArrayAllZero (msg : String) (a : FloatArray) (expectedSize : Nat) : IO Unit := do
  if a.size != expectedSize then
    throw <| IO.userError s!"{msg}: size mismatch ({a.size} vs {expectedSize})"
  for i in [:a.size] do
    let x := a.get! i
    if x != 0.0 then
      throw <| IO.userError s!"{msg}[{i}]: got {x}, expected 0.0"

def runReduceSumAxisEmptyReducedDim : IO Unit := do
  IO.println "== reduce_sum_axis empty reduced dimension =="
  let input := Runtime.Autograd.Cuda.Buffer.zeros 0
  let out := Runtime.Autograd.Cuda.Buffer.reduceSumAxis input #[3, 0, 5] 1
  assertFloatArrayAllZero "reduce_sum_axis dims=[3,0,5], axis=1"
    (Runtime.Autograd.Cuda.Buffer.toFloatArray out) 15

def runReduceMaxAxisEmptyReducedDim : IO Unit := do
  IO.println "== reduce_max_axis empty reduced dimension =="
  let inputAxis0 := Runtime.Autograd.Cuda.Buffer.zeros 0
  let outAxis0 := Runtime.Autograd.Cuda.Buffer.reduceMaxAxis0 inputAxis0 0 5
  assertFloatArrayAllZero "reduce_max_axis0 rows=0 cols=5"
    (Runtime.Autograd.Cuda.Buffer.toFloatArray outAxis0) 5
  let inputAxis1 := Runtime.Autograd.Cuda.Buffer.zeros 0
  let outAxis1 := Runtime.Autograd.Cuda.Buffer.reduceMaxAxis1 inputAxis1 3 0
  assertFloatArrayAllZero "reduce_max_axis1 rows=3 cols=0"
    (Runtime.Autograd.Cuda.Buffer.toFloatArray outAxis1) 3

def run : IO Unit := do
  IO.println "=== CUDA kernel coverage: views/broadcast/reduce ==="

  -- reshape
  IO.println "== reshape =="
  let s1 : Shape := shape![2, 3]
  let s2 : Shape := shape![6]
  let x1 : Tensor Float s1 :=
    tensorND! [2, 3] [
      0.10, -0.20, 0.30,
      0.05,  0.25, -0.15
    ]
  have hSize : Shape.size s1 = Shape.size s2 := by decide
  let t0 : Tape Float := Tape.empty
  let (t1, xId) := Tape.leaf (t := t0) x1 (name := some "x")
  let (t2, yId) ← Utils.okOrThrow (Tape.reshape (α := Float) (t := t1) (s₁ := s1) (s₂ := s2) xId hSize)
  let yCpu ← Utils.cpuValue (s := s2) t2 yId
  let seedCpu : Runtime.AnyTensor Float := AnyTensor.mk (fill (1.0 : Float) s2)
  let gradsCpu ← Utils.okOrThrow (Tape.backwardDenseAll (α := Float) (t := t2) yId seedCpu)
  let dxCpu ← Utils.cpuGrad (s := s1) gradsCpu xId

  let t0c : Runtime.Autograd.Cuda.Tape := Runtime.Autograd.Cuda.Tape.empty
  let (t1c, xIdc) := Runtime.Autograd.Cuda.Tape.leaf (t := t0c) (Utils.tensorToAnyBuffer x1) (name := some "x")
  let (t2c, yIdc) ← Utils.okOrThrow (Runtime.Autograd.Cuda.Tape.reshape (t := t1c) (s₁ := s1) (s₂ := s2) xIdc hSize)
  let yCuda ← Utils.cudaValue (s := s2) t2c yIdc
  let seedCuda : Runtime.Autograd.Cuda.AnyBuffer :=
    { s := s2, buf := Runtime.Autograd.Cuda.Buffer.full (UInt32.ofNat (Shape.size s2)) 1.0 }
  let gradsCuda ← Utils.okOrThrow (Runtime.Autograd.Cuda.Tape.backwardDenseAll (t := t2c) yIdc seedCuda)
  let dxCuda ← Utils.cudaGrad (s := s1) gradsCuda xIdc

  Utils.assertTensorApprox (s := s2) "reshape forward" yCuda yCpu (tol := 2e-3)
  Utils.assertTensorApprox (s := s1) "reshape backward" dxCuda dxCpu (tol := 2e-3)

  -- transpose2d
  IO.println "== transpose2d =="
  let sM : Shape := shape![2, 3]
  let xM : Tensor Float sM :=
    tensorND! [2, 3] [
      1.0, 2.0, 3.0,
      4.0, 5.0, 6.0
    ]
  let t0m : Tape Float := Tape.empty
  let (t1m, xMid) := Tape.leaf (t := t0m) xM (name := some "x")
  let (t2m, yMid) ← Utils.okOrThrow (Tape.transpose2d (α := Float) (t := t1m) (m := 2) (n := 3) xMid)
  let yCpuM ← Utils.cpuValue (s := shape![3, 2]) t2m yMid
  let seedCpuM : Runtime.AnyTensor Float := AnyTensor.mk (fill (1.0 : Float) (shape![3, 2]))
  let gradsCpuM ← Utils.okOrThrow (Tape.backwardDenseAll (α := Float) (t := t2m) yMid seedCpuM)
  let dxCpuM ← Utils.cpuGrad (s := sM) gradsCpuM xMid

  let t0mc : Runtime.Autograd.Cuda.Tape := Runtime.Autograd.Cuda.Tape.empty
  let (t1mc, xMidc) := Runtime.Autograd.Cuda.Tape.leaf (t := t0mc) (Utils.tensorToAnyBuffer xM) (name := some "x")
  let (t2mc, yMidc) ← Utils.okOrThrow (Runtime.Autograd.Cuda.Tape.transpose2d (t := t1mc) (m := 2) (n := 3) xMidc)
  let yCudaM ← Utils.cudaValue (s := shape![3, 2]) t2mc yMidc
  let seedCudaM : Runtime.Autograd.Cuda.AnyBuffer :=
    { s := shape![3, 2], buf := Runtime.Autograd.Cuda.Buffer.full (UInt32.ofNat (Shape.size (shape![3, 2]))) 1.0 }
  let gradsCudaM ← Utils.okOrThrow (Runtime.Autograd.Cuda.Tape.backwardDenseAll (t := t2mc) yMidc seedCudaM)
  let dxCudaM ← Utils.cudaGrad (s := sM) gradsCudaM xMidc

  Utils.assertTensorApprox (s := shape![3, 2]) "transpose2d forward" yCudaM yCpuM (tol := 2e-3)
  Utils.assertTensorApprox (s := sM) "transpose2d backward" dxCudaM dxCpuM (tol := 2e-3)

  -- swapAdjacentAtDepth with unequal adjacent axes.
  IO.println "== swapAdjacentAtDepth unequal axes =="
  let sSwap : Shape := shape![2, 3, 4]
  let sSwapOut : Shape := shape![2, 4, 3]
  let xSwap : Tensor Float sSwap :=
    tensorND! [2, 3, 4] [
      0.0,  1.0,  2.0,  3.0,
      4.0,  5.0,  6.0,  7.0,
      8.0,  9.0, 10.0, 11.0,
      12.0, 13.0, 14.0, 15.0,
      16.0, 17.0, 18.0, 19.0,
      20.0, 21.0, 22.0, 23.0
    ]
  let seedSwap : Tensor Float sSwapOut :=
    tensorND! [2, 4, 3] [
      0.10, 0.20, 0.30,
      0.40, 0.50, 0.60,
      0.70, 0.80, 0.90,
      1.00, 1.10, 1.20,
      1.30, 1.40, 1.50,
      1.60, 1.70, 1.80,
      1.90, 2.00, 2.10,
      2.20, 2.30, 2.40
    ]

  let t0s : Tape Float := Tape.empty
  let (t1s, xSid) := Tape.leaf (t := t0s) xSwap (name := some "x")
  let (t2s, ySid) ← Utils.okOrThrow (Tape.swapAdjacentAtDepth (α := Float) (t := t1s)
    (s := sSwap) 1 xSid)
  let yCpuSwap ← Utils.cpuValue (s := sSwapOut) t2s ySid
  let gradsCpuSwap ← Utils.okOrThrow (Tape.backwardDenseAll (α := Float) (t := t2s) ySid
    ({ s := sSwapOut, t := seedSwap } : Runtime.AnyTensor Float))
  let dxCpuSwap ← Utils.cpuGrad (s := sSwap) gradsCpuSwap xSid

  let t0sc : Runtime.Autograd.Cuda.Tape := Runtime.Autograd.Cuda.Tape.empty
  let (t1sc, xSidc) := Runtime.Autograd.Cuda.Tape.leaf (t := t0sc)
    (Utils.tensorToAnyBuffer xSwap) (name := some "x")
  let (t2sc, ySidc) ← Utils.okOrThrow (Runtime.Autograd.Cuda.Tape.swapAdjacentAtDepth
    (t := t1sc) (s := sSwap) 1 xSidc)
  let yCudaSwap ← Utils.cudaValue (s := sSwapOut) t2sc ySidc
  let gradsCudaSwap ← Utils.okOrThrow (Runtime.Autograd.Cuda.Tape.backwardDenseAll
    (t := t2sc) ySidc { s := sSwapOut, buf := Utils.tensorToBuffer seedSwap })
  let dxCudaSwap ← Utils.cudaGrad (s := sSwap) gradsCudaSwap xSidc

  Utils.assertTensorApprox (s := sSwapOut) "swapAdjacentAtDepth unequal forward"
    yCudaSwap yCpuSwap (tol := 2e-3)
  Utils.assertTensorApprox (s := sSwap) "swapAdjacentAtDepth unequal backward"
    dxCudaSwap dxCpuSwap (tol := 2e-3)

  -- broadcastTo
  IO.println "== broadcastTo =="
  let sB1 : Shape := shape![1, 2]
  let sB2 : Shape := shape![3, 2]
  let cb : Shape.CanBroadcastTo sB1 sB2 := (inferInstance : Shape.BroadcastTo sB1 sB2).proof
  let xB : Tensor Float sB1 := tensorND! [1, 2] [0.25, -0.50]

  let t0b : Tape Float := Tape.empty
  let (t1b, xBid) := Tape.leaf (t := t0b) xB (name := some "x")
  let (t2b, yBid) ← Utils.okOrThrow (Tape.broadcastTo (α := Float) (t := t1b) (s₁ := sB1) (s₂ := sB2) cb xBid)
  let yCpuB ← Utils.cpuValue (s := sB2) t2b yBid
  let seedCpuB : Runtime.AnyTensor Float := AnyTensor.mk (fill (1.0 : Float) sB2)
  let gradsCpuB ← Utils.okOrThrow (Tape.backwardDenseAll (α := Float) (t := t2b) yBid seedCpuB)
  let dxCpuB ← Utils.cpuGrad (s := sB1) gradsCpuB xBid

  let t0bc : Runtime.Autograd.Cuda.Tape := Runtime.Autograd.Cuda.Tape.empty
  let (t1bc, xBidc) := Runtime.Autograd.Cuda.Tape.leaf (t := t0bc) (Utils.tensorToAnyBuffer xB) (name := some "x")
  let (t2bc, yBidc) ← Utils.okOrThrow (Runtime.Autograd.Cuda.Tape.broadcastTo (t := t1bc) (s₁ := sB1) (s₂ := sB2) cb xBidc)
  let yCudaB ← Utils.cudaValue (s := sB2) t2bc yBidc
  let seedCudaB : Runtime.Autograd.Cuda.AnyBuffer :=
    { s := sB2, buf := Runtime.Autograd.Cuda.Buffer.full (UInt32.ofNat (Shape.size sB2)) 1.0 }
  let gradsCudaB ← Utils.okOrThrow (Runtime.Autograd.Cuda.Tape.backwardDenseAll (t := t2bc) yBidc seedCudaB)
  let dxCudaB ← Utils.cudaGrad (s := sB1) gradsCudaB xBidc

  Utils.assertTensorApprox (s := sB2) "broadcastTo forward" yCudaB yCpuB (tol := 2e-3)
  Utils.assertTensorApprox (s := sB1) "broadcastTo backward" dxCudaB dxCpuB (tol := 2e-3)

  -- reduce_sum / reduce_mean
  IO.println "== reduce_sum / reduce_mean =="
  let sR : Shape := shape![2, 2]
  let xR : Tensor Float sR :=
    tensorND! [2, 2] [
      1.0, 2.0,
      3.0, 4.0
    ]
  let axis : Nat := 1
  let sOut := shapeAfterSum sR axis

  let t0r : Tape Float := Tape.empty
  let (t1r, xRid) := Tape.leaf (t := t0r) xR (name := some "x")
  let (t2r, sumId) ← Utils.okOrThrow (Tape.reduceSum (α := Float) (t := t1r) (s := sR) axis xRid)
  let (t3r, meanId) ← Utils.okOrThrow (Tape.reduceMean (α := Float) (t := t2r) (s := sR) axis xRid)

  let yCpuSum ← Utils.cpuValue (s := sOut) t3r sumId
  let seedCpuSum : Runtime.AnyTensor Float := AnyTensor.mk (fill (1.0 : Float) sOut)
  let gradsCpuSum ← Utils.okOrThrow (Tape.backwardDenseAll (α := Float) (t := t3r) sumId seedCpuSum)
  let dxCpuSum ← Utils.cpuGrad (s := sR) gradsCpuSum xRid

  let yCpuMean ← Utils.cpuValue (s := sOut) t3r meanId
  let seedCpuMean : Runtime.AnyTensor Float := AnyTensor.mk (fill (1.0 : Float) sOut)
  let gradsCpuMean ← Utils.okOrThrow (Tape.backwardDenseAll (α := Float) (t := t3r) meanId seedCpuMean)
  let dxCpuMean ← Utils.cpuGrad (s := sR) gradsCpuMean xRid

  let t0rc : Runtime.Autograd.Cuda.Tape := Runtime.Autograd.Cuda.Tape.empty
  let (t1rc, xRidc) := Runtime.Autograd.Cuda.Tape.leaf (t := t0rc) (Utils.tensorToAnyBuffer xR) (name := some "x")
  let (t2rc, sumIdc) ← Utils.okOrThrow (Runtime.Autograd.Cuda.Tape.reduceSum (s := sR) axis (t := t1rc) xRidc)
  let (t3rc, meanIdc) ← Utils.okOrThrow (Runtime.Autograd.Cuda.Tape.reduceMean (s := sR) axis (t := t2rc) xRidc)

  let yCudaSum ← Utils.cudaValue (s := sOut) t3rc sumIdc
  let seedCudaSum : Runtime.Autograd.Cuda.AnyBuffer :=
    { s := sOut, buf := Runtime.Autograd.Cuda.Buffer.full (UInt32.ofNat (Shape.size sOut)) 1.0 }
  let gradsCudaSum ← Utils.okOrThrow (Runtime.Autograd.Cuda.Tape.backwardDenseAll (t := t3rc) sumIdc seedCudaSum)
  let dxCudaSum ← Utils.cudaGrad (s := sR) gradsCudaSum xRidc

  let yCudaMean ← Utils.cudaValue (s := sOut) t3rc meanIdc
  let seedCudaMean : Runtime.Autograd.Cuda.AnyBuffer :=
    { s := sOut, buf := Runtime.Autograd.Cuda.Buffer.full (UInt32.ofNat (Shape.size sOut)) 1.0 }
  let gradsCudaMean ← Utils.okOrThrow (Runtime.Autograd.Cuda.Tape.backwardDenseAll (t := t3rc) meanIdc seedCudaMean)
  let dxCudaMean ← Utils.cudaGrad (s := sR) gradsCudaMean xRidc

  Utils.assertTensorApprox (s := sOut) "reduce_sum forward" yCudaSum yCpuSum (tol := 2e-3)
  Utils.assertTensorApprox (s := sR) "reduce_sum backward" dxCudaSum dxCpuSum (tol := 2e-3)

  Utils.assertTensorApprox (s := sOut) "reduce_mean forward" yCudaMean yCpuMean (tol := 2e-3)
  Utils.assertTensorApprox (s := sR) "reduce_mean backward" dxCudaMean dxCpuMean (tol := 2e-3)
  runReduceSumAxisEmptyReducedDim
  runReduceMaxAxisEmptyReducedDim

end ViewsBroadcastReduce
end Cuda
end Tests
