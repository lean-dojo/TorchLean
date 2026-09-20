/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Engine.Cuda.Ops
public import NN.Tensor
public import NN.Tests.Runtime.Cuda.Utils

/-!
# CUDA Kernel Coverage: Views, Broadcast, Reduce

Small CPU/CUDA tape comparisons for:
- `reshape`, arbitrary adjacent-axis swaps, and permutations
- `broadcastTo`
- `reduce_sum`, `reduce_mean`, low-level empty-axis `reduce_max` parity
-/

@[expose] public section

namespace Tests
namespace Cuda
namespace ViewsBroadcastReduce

open Spec TorchLean
open TorchLean TorchLean.Tensor
open Runtime.Autograd

-- Buffer comparisons and the `floatArray` literal wrapper come from `Cuda.Utils`.
open Tests.Cuda.Utils (floatArray assertFloatArrayApprox)

def assertFloatArrayAllZero (msg : String) (a : FloatArray) (expectedSize : Nat) : IO Unit := do
  if a.size != expectedSize then
    throw <| IO.userError s!"{msg}: size mismatch ({a.size} vs {expectedSize})"
  for i in [:a.size] do
    let x := a.get! i
    if x != 0.0 then
      throw <| IO.userError s!"{msg}[{i}]: got {x}, expected 0.0"

def runRankPolymorphicProductCoverage : IO Unit := do
  IO.println "== rank-polymorphic native product coverage =="
  let x := Runtime.Autograd.Cuda.Buffer.ofFloatArray <| floatArray #[
    0.0, 1.0, 2.0, 3.0, 4.0, 5.0,
    6.0, 7.0, 8.0, 9.0, 10.0, 11.0
  ]
  let b := Runtime.Autograd.Cuda.Buffer.broadcastTo x #[2, 1, 3, 2] #[2, 4, 3, 2] #[1, 2, 3, 4]
  assertFloatArrayApprox "broadcastTo rank-4"
    (Runtime.Autograd.Cuda.Buffer.toFloatArray b)
    (floatArray #[
      0.0, 1.0, 2.0, 3.0, 4.0, 5.0,
      0.0, 1.0, 2.0, 3.0, 4.0, 5.0,
      0.0, 1.0, 2.0, 3.0, 4.0, 5.0,
      0.0, 1.0, 2.0, 3.0, 4.0, 5.0,
      6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
      6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
      6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
      6.0, 7.0, 8.0, 9.0, 10.0, 11.0
    ]) (tol := 1e-5)

  let dOut := Runtime.Autograd.Cuda.Buffer.full 48 1.0
  let reduced :=
    Runtime.Autograd.Cuda.Buffer.reduceFromBroadcastTo dOut #[2, 1, 3, 2] #[2, 4, 3, 2]
      #[1, 2, 3, 4]
  assertFloatArrayApprox "reduceFromBroadcastTo rank-4"
    (Runtime.Autograd.Cuda.Buffer.toFloatArray reduced)
    (floatArray (Array.replicate 12 4.0)) (tol := 1e-5)

  let x24 := Runtime.Autograd.Cuda.Buffer.ofFloatArray <| floatArray #[
    0.0, 1.0, 2.0, 3.0,
    4.0, 5.0, 6.0, 7.0,
    8.0, 9.0, 10.0, 11.0,
    12.0, 13.0, 14.0, 15.0,
    16.0, 17.0, 18.0, 19.0,
    20.0, 21.0, 22.0, 23.0
  ]
  let sumLast := Runtime.Autograd.Cuda.Buffer.reduceSumAxis x24 #[2, 3, 4] 2
  assertFloatArrayApprox "reduceSumAxis three-axis input"
    (Runtime.Autograd.Cuda.Buffer.toFloatArray sumLast)
    (floatArray #[6.0, 22.0, 38.0, 54.0, 70.0, 86.0]) (tol := 1e-5)

  let swapped := Runtime.Autograd.Cuda.Buffer.swapAdjacentAtDepth x24 #[2, 3, 4] 1
  assertFloatArrayApprox "swapAdjacentAtDepth three-axis input"
    (Runtime.Autograd.Cuda.Buffer.toFloatArray swapped)
    (floatArray #[
      0.0, 4.0, 8.0,
      1.0, 5.0, 9.0,
      2.0, 6.0, 10.0,
      3.0, 7.0, 11.0,
      12.0, 16.0, 20.0,
      13.0, 17.0, 21.0,
      14.0, 18.0, 22.0,
      15.0, 19.0, 23.0
    ]) (tol := 1e-5)

def runReduceSumAxisEmptyReducedDim : IO Unit := do
  IO.println "== reduce_sum_axis empty reduced dimension =="
  let input ← Runtime.Autograd.Cuda.Buffer.zerosIO 0
  let out := Runtime.Autograd.Cuda.Buffer.reduceSumAxis input #[3, 0, 5] 1
  assertFloatArrayAllZero "reduce_sum_axis dims=[3,0,5], axis=1"
    (Runtime.Autograd.Cuda.Buffer.toFloatArray out) 15

def runReduceMaxAxisEmptyReducedDim : IO Unit := do
  IO.println "== reduce_max_axis empty reduced dimension =="
  let emptyByColumnInput ← Runtime.Autograd.Cuda.Buffer.zerosIO 0
  let outByColumn := Runtime.Autograd.Cuda.Buffer.reduceMaxByColumn emptyByColumnInput 0 5
  assertFloatArrayAllZero "reduce_max_by_column rows=0 cols=5"
    (Runtime.Autograd.Cuda.Buffer.toFloatArray outByColumn) 5
  let emptyByRowInput ← Runtime.Autograd.Cuda.Buffer.zerosIO 0
  let outByRow := Runtime.Autograd.Cuda.Buffer.reduceMaxByRow emptyByRowInput 3 0
  assertFloatArrayAllZero "reduce_max_by_row rows=3 cols=0"
    (Runtime.Autograd.Cuda.Buffer.toFloatArray outByRow) 3

def run : IO Unit := do
  IO.println "=== CUDA kernel coverage: views/broadcast/reduce ==="

  -- reshape
  IO.println "== reshape =="
  let s1 : Shape := [2, 3]
  let s2 : Shape := [6]
  let x1 : Tensor Float s1 :=
    (Tensor.from #[
      0.10, -0.20, 0.30,
      0.05,  0.25, -0.15
    ]).reshape [2, 3] (by dsimp; decide)
  have hSize : Spec.Shape.size s1 = Spec.Shape.size s2 := by decide
  let t0 : Tape Float := Tape.empty
  let (t1, xId) := Tape.leaf (t := t0) x1 (name := some "x")
  let (t2, yId) ← Utils.okOrThrow
    (Tape.reshape (α := Float) (t := t1) (s₁ := s1) (s₂ := s2) xId hSize)
  let yCpu ← Utils.cpuValue (s := s2) t2 yId
  let seedCpu : Spec.SomeTensor Float := Spec.SomeTensor.ofTensor (Tensor.full s2 (1.0 : Float))
  let gradsCpu ← Utils.okOrThrow (Tape.backwardDenseAll (α := Float) (t := t2) yId seedCpu)
  let dxCpu ← Utils.cpuGrad (s := s1) gradsCpu xId

  let t0c : Runtime.Autograd.Cuda.Tape := Runtime.Autograd.Cuda.Tape.empty
  let (t1c, xIdc) :=
    Runtime.Autograd.Cuda.Tape.leaf (t := t0c) (Utils.tensorToAnyBuffer x1) (name := some "x")
  let (t2c, yIdc) ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.reshape (t := t1c) (s₁ := s1) (s₂ := s2) xIdc hSize)
  let yCuda ← Utils.cudaValue (s := s2) t2c yIdc
  let seedCuda : Runtime.Autograd.Cuda.AnyBuffer :=
    { s := s2, buf := Runtime.Autograd.Cuda.Buffer.full (UInt32.ofNat (Spec.Shape.size s2)) 1.0 }
  let gradsCuda ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.backwardDenseAll (t := t2c) yIdc seedCuda)
  let dxCuda ← Utils.cudaGrad (s := s1) gradsCuda xIdc

  Utils.assertTensorApprox (s := s2) "reshape forward" yCuda yCpu (tol := 2e-3)
  Utils.assertTensorApprox (s := s1) "reshape backward" dxCuda dxCpu (tol := 2e-3)

  -- Matrix transpose is the depth-zero instance of the general adjacent-axis swap.
  IO.println "== transpose axes 0 and 1 =="
  let sM : Shape := [2, 3]
  let xM : Tensor Float sM :=
    (Tensor.from #[
      1.0, 2.0, 3.0,
      4.0, 5.0, 6.0
    ]).reshape [2, 3] (by dsimp; decide)
  let t0m : Tape Float := Tape.empty
  let (t1m, xMid) := Tape.leaf (t := t0m) xM (name := some "x")
  let (t2m, yMid) ← Utils.okOrThrow
    (Tape.swapAdjacentAtDepth (α := Float) (t := t1m) (s := sM) 0 xMid)
  let yCpuM ← Utils.cpuValue (s := [3, 2]) t2m yMid
  let seedCpuM : Spec.SomeTensor Float :=
    Spec.SomeTensor.ofTensor (Tensor.full [3, 2] (1.0 : Float))
  let gradsCpuM ← Utils.okOrThrow (Tape.backwardDenseAll (α := Float) (t := t2m) yMid seedCpuM)
  let dxCpuM ← Utils.cpuGrad (s := sM) gradsCpuM xMid

  let t0mc : Runtime.Autograd.Cuda.Tape := Runtime.Autograd.Cuda.Tape.empty
  let (t1mc, xMidc) :=
    Runtime.Autograd.Cuda.Tape.leaf (t := t0mc) (Utils.tensorToAnyBuffer xM) (name := some "x")
  let (t2mc, yMidc) ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.swapAdjacentAtDepth (t := t1mc) (s := sM) 0 xMidc)
  let yCudaM ← Utils.cudaValue (s := [3, 2]) t2mc yMidc
  let seedCudaM : Runtime.Autograd.Cuda.AnyBuffer :=
    { s := [3, 2],
      buf := Runtime.Autograd.Cuda.Buffer.full (UInt32.ofNat (Spec.Shape.size [3, 2])) 1.0 }
  let gradsCudaM ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.backwardDenseAll (t := t2mc) yMidc seedCudaM)
  let dxCudaM ← Utils.cudaGrad (s := sM) gradsCudaM xMidc

  Utils.assertTensorApprox (s := [3, 2]) "transpose forward" yCudaM yCpuM (tol := 2e-3)
  Utils.assertTensorApprox (s := sM) "transpose backward" dxCudaM dxCpuM (tol := 2e-3)

  -- swapAdjacentAtDepth with unequal adjacent axes.
  IO.println "== swapAdjacentAtDepth unequal axes =="
  let sSwap : Shape := [2, 3, 4]
  let sSwapOut : Shape := [2, 4, 3]
  let xSwap : Tensor Float sSwap :=
    (Tensor.from #[
      0.0,  1.0,  2.0,  3.0,
      4.0,  5.0,  6.0,  7.0,
      8.0,  9.0, 10.0, 11.0,
      12.0, 13.0, 14.0, 15.0,
      16.0, 17.0, 18.0, 19.0,
      20.0, 21.0, 22.0, 23.0
    ]).reshape [2, 3, 4] (by dsimp; decide)
  let seedSwap : Tensor Float sSwapOut :=
    (Tensor.from #[
      0.10, 0.20, 0.30,
      0.40, 0.50, 0.60,
      0.70, 0.80, 0.90,
      1.00, 1.10, 1.20,
      1.30, 1.40, 1.50,
      1.60, 1.70, 1.80,
      1.90, 2.00, 2.10,
      2.20, 2.30, 2.40
    ]).reshape [2, 4, 3] (by dsimp; decide)

  let t0s : Tape Float := Tape.empty
  let (t1s, xSid) := Tape.leaf (t := t0s) xSwap (name := some "x")
  let (t2s, ySid) ← Utils.okOrThrow (Tape.swapAdjacentAtDepth (α := Float) (t := t1s)
    (s := sSwap) 1 xSid)
  let yCpuSwap ← Utils.cpuValue (s := sSwapOut) t2s ySid
  let gradsCpuSwap ← Utils.okOrThrow (Tape.backwardDenseAll (α := Float) (t := t2s) ySid
    (Spec.SomeTensor.ofTensor seedSwap))
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
  let sB1 : Shape := [1, 2]
  let sB2 : Shape := [3, 2]
  let cb : Shape.CanBroadcastTo sB1 sB2 := (inferInstance : Shape.BroadcastTo sB1 sB2).proof
  let xB : Tensor Float sB1 := (Tensor.from #[0.25, -0.50]).reshape [1, 2] (by dsimp; decide)

  let t0b : Tape Float := Tape.empty
  let (t1b, xBid) := Tape.leaf (t := t0b) xB (name := some "x")
  let (t2b, yBid) ← Utils.okOrThrow
    (Tape.broadcastTo (α := Float) (t := t1b) (s₁ := sB1) (s₂ := sB2) cb xBid)
  let yCpuB ← Utils.cpuValue (s := sB2) t2b yBid
  let seedCpuB : Spec.SomeTensor Float := Spec.SomeTensor.ofTensor (Tensor.full sB2 (1.0 : Float))
  let gradsCpuB ← Utils.okOrThrow (Tape.backwardDenseAll (α := Float) (t := t2b) yBid seedCpuB)
  let dxCpuB ← Utils.cpuGrad (s := sB1) gradsCpuB xBid

  let t0bc : Runtime.Autograd.Cuda.Tape := Runtime.Autograd.Cuda.Tape.empty
  let (t1bc, xBidc) :=
    Runtime.Autograd.Cuda.Tape.leaf (t := t0bc) (Utils.tensorToAnyBuffer xB) (name := some "x")
  let (t2bc, yBidc) ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.broadcastTo (t := t1bc) (s₁ := sB1) (s₂ := sB2) cb xBidc)
  let yCudaB ← Utils.cudaValue (s := sB2) t2bc yBidc
  let seedCudaB : Runtime.Autograd.Cuda.AnyBuffer :=
    { s := sB2, buf := Runtime.Autograd.Cuda.Buffer.full (UInt32.ofNat (Spec.Shape.size sB2)) 1.0 }
  let gradsCudaB ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.backwardDenseAll (t := t2bc) yBidc seedCudaB)
  let dxCudaB ← Utils.cudaGrad (s := sB1) gradsCudaB xBidc

  Utils.assertTensorApprox (s := sB2) "broadcastTo forward" yCudaB yCpuB (tol := 2e-3)
  Utils.assertTensorApprox (s := sB1) "broadcastTo backward" dxCudaB dxCpuB (tol := 2e-3)

  -- reduce_sum / reduce_mean
  IO.println "== reduce_sum / reduce_mean =="
  let sR : Shape := [2, 2]
  let xR : Tensor Float sR :=
    (Tensor.from #[
      1.0, 2.0,
      3.0, 4.0
    ]).reshape [2, 2] (by dsimp; decide)
  let axis : Nat := 1
  let sOut := shapeAfterSum sR axis

  let t0r : Tape Float := Tape.empty
  let (t1r, xRid) := Tape.leaf (t := t0r) xR (name := some "x")
  let (t2r, sumId) ← Utils.okOrThrow (Tape.reduceSum (α := Float) (t := t1r) (s := sR) axis xRid)
  let (t3r, meanId) ← Utils.okOrThrow (Tape.reduceMean (α := Float) (t := t2r) (s := sR) axis xRid)

  let yCpuSum ← Utils.cpuValue (s := sOut) t3r sumId
  let seedCpuSum : Spec.SomeTensor Float :=
    Spec.SomeTensor.ofTensor (Tensor.full sOut (1.0 : Float))
  let gradsCpuSum ← Utils.okOrThrow (Tape.backwardDenseAll (α := Float) (t := t3r) sumId seedCpuSum)
  let dxCpuSum ← Utils.cpuGrad (s := sR) gradsCpuSum xRid

  let yCpuMean ← Utils.cpuValue (s := sOut) t3r meanId
  let seedCpuMean : Spec.SomeTensor Float :=
    Spec.SomeTensor.ofTensor (Tensor.full sOut (1.0 : Float))
  let gradsCpuMean ← Utils.okOrThrow
    (Tape.backwardDenseAll (α := Float) (t := t3r) meanId seedCpuMean)
  let dxCpuMean ← Utils.cpuGrad (s := sR) gradsCpuMean xRid

  let t0rc : Runtime.Autograd.Cuda.Tape := Runtime.Autograd.Cuda.Tape.empty
  let (t1rc, xRidc) :=
    Runtime.Autograd.Cuda.Tape.leaf (t := t0rc) (Utils.tensorToAnyBuffer xR) (name := some "x")
  let (t2rc, sumIdc) ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.reduceSum (s := sR) axis (t := t1rc) xRidc)
  let (t3rc, meanIdc) ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.reduceMean (s := sR) axis (t := t2rc) xRidc)

  let yCudaSum ← Utils.cudaValue (s := sOut) t3rc sumIdc
  let seedCudaSum : Runtime.Autograd.Cuda.AnyBuffer :=
    { s := sOut,
      buf := Runtime.Autograd.Cuda.Buffer.full (UInt32.ofNat (Spec.Shape.size sOut)) 1.0 }
  let gradsCudaSum ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.backwardDenseAll (t := t3rc) sumIdc seedCudaSum)
  let dxCudaSum ← Utils.cudaGrad (s := sR) gradsCudaSum xRidc

  let yCudaMean ← Utils.cudaValue (s := sOut) t3rc meanIdc
  let seedCudaMean : Runtime.Autograd.Cuda.AnyBuffer :=
    { s := sOut,
      buf := Runtime.Autograd.Cuda.Buffer.full (UInt32.ofNat (Spec.Shape.size sOut)) 1.0 }
  let gradsCudaMean ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.backwardDenseAll (t := t3rc) meanIdc seedCudaMean)
  let dxCudaMean ← Utils.cudaGrad (s := sR) gradsCudaMean xRidc

  Utils.assertTensorApprox (s := sOut) "reduce_sum forward" yCudaSum yCpuSum (tol := 2e-3)
  Utils.assertTensorApprox (s := sR) "reduce_sum backward" dxCudaSum dxCpuSum (tol := 2e-3)

  Utils.assertTensorApprox (s := sOut) "reduce_mean forward" yCudaMean yCpuMean (tol := 2e-3)
  Utils.assertTensorApprox (s := sR) "reduce_mean backward" dxCudaMean dxCpuMean (tol := 2e-3)
  runRankPolymorphicProductCoverage
  runReduceSumAxisEmptyReducedDim
  runReduceMaxAxisEmptyReducedDim

end ViewsBroadcastReduce
end Cuda
end Tests
