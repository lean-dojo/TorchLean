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
# CUDA Kernel Coverage: Multi-Head Attention

Compares CPU eager tape vs CUDA eager tape for `multi_head_attention` (forward + backward).

We keep the problem small so stub-mode runs quickly and float64/float32 roundoff differences are
limited.
-/

@[expose] public section

namespace Tests
namespace Cuda
namespace Attention

open Spec
open Tensor
open Runtime.Autograd

abbrev n : Nat := 2
abbrev numHeads : Nat := 2
abbrev dModel : Nat := 4
abbrev headDim : Nat := 2

def hN : n ≠ 0 := by decide

abbrev projDim : Nat := numHeads * headDim

def wq : Tensor Float (shape![dModel, projDim]) :=
  tensorND! [dModel, projDim] [
    0.01, 0.02, 0.03, 0.04,
    0.05, 0.06, 0.07, 0.08,
    0.09, 0.10, 0.11, 0.12,
    0.13, 0.14, 0.15, 0.16
  ]

def wk : Tensor Float (shape![dModel, projDim]) :=
  tensorND! [dModel, projDim] [
    0.02, 0.01, 0.04, 0.03,
    0.06, 0.05, 0.08, 0.07,
    0.10, 0.09, 0.12, 0.11,
    0.14, 0.13, 0.16, 0.15
  ]

def wv : Tensor Float (shape![dModel, projDim]) :=
  tensorND! [dModel, projDim] [
    0.03, 0.00, 0.01, 0.02,
    0.00, 0.03, 0.02, 0.01,
    0.01, 0.02, 0.03, 0.00,
    0.02, 0.01, 0.00, 0.03
  ]

def wo : Tensor Float (shape![projDim, dModel]) :=
  tensorND! [projDim, dModel] [
    0.05, 0.00, 0.01, 0.02,
    0.00, 0.05, 0.02, 0.01,
    0.01, 0.02, 0.05, 0.00,
    0.02, 0.01, 0.00, 0.05
  ]

def x : Tensor Float (shape![n, dModel]) :=
  tensorND! [n, dModel] [
    0.10, -0.20, 0.05, 0.30,
    -0.05, 0.25, -0.10, 0.15
  ]

def mask : Tensor Bool (shape![n, n]) :=
  tensorND! [n, n] [
    true,  true,
    false, true
  ]

def run : IO Unit := do
  IO.println "=== CUDA kernel coverage: multi_head_attention ==="

  let outShape : Shape := shape![n, dModel]

  -- CPU tape
  let t0 : Tape Float := Tape.empty
  let (t1, wqId) := Tape.leaf (t := t0) wq (name := some "wq")
  let (t2, wkId) := Tape.leaf (t := t1) wk (name := some "wk")
  let (t3, wvId) := Tape.leaf (t := t2) wv (name := some "wv")
  let (t4, woId) := Tape.leaf (t := t3) wo (name := some "wo")
  let (t5, xId) := Tape.leaf (t := t4) x (name := some "x")
  let (t6, yId) ← Utils.okOrThrow
    (Tape.multiHeadAttention (α := Float) (t := t5)
      (n := n) (numHeads := numHeads) (dModel := dModel) (headDim := headDim)
      (h1 := hN) wqId wkId wvId woId xId (mask := some mask))
  let yCpu ← Utils.cpuValue (s := outShape) t6 yId
  let seedCpu : Runtime.AnyTensor Float := AnyTensor.mk (fill (1.0 : Float) outShape)
  let gradsCpu ← Utils.okOrThrow (Tape.backwardDenseAll (α := Float) (t := t6) yId seedCpu)
  let dxCpu ← Utils.cpuGrad (s := outShape) gradsCpu xId
  let dWqCpu ← Utils.cpuGrad (s := shape![dModel, projDim]) gradsCpu wqId
  let dWkCpu ← Utils.cpuGrad (s := shape![dModel, projDim]) gradsCpu wkId
  let dWvCpu ← Utils.cpuGrad (s := shape![dModel, projDim]) gradsCpu wvId
  let dWoCpu ← Utils.cpuGrad (s := shape![projDim, dModel]) gradsCpu woId

  -- CUDA tape
  let t0c : Runtime.Autograd.Cuda.Tape := Runtime.Autograd.Cuda.Tape.empty
  let (t1c, wqIdc) := Runtime.Autograd.Cuda.Tape.leaf (t := t0c) (Utils.tensorToAnyBuffer wq)
    (name := some "wq")
  let (t2c, wkIdc) := Runtime.Autograd.Cuda.Tape.leaf (t := t1c) (Utils.tensorToAnyBuffer wk)
    (name := some "wk")
  let (t3c, wvIdc) := Runtime.Autograd.Cuda.Tape.leaf (t := t2c) (Utils.tensorToAnyBuffer wv)
    (name := some "wv")
  let (t4c, woIdc) := Runtime.Autograd.Cuda.Tape.leaf (t := t3c) (Utils.tensorToAnyBuffer wo)
    (name := some "wo")
  let (t5c, xIdc) := Runtime.Autograd.Cuda.Tape.leaf (t := t4c) (Utils.tensorToAnyBuffer x)
    (name := some "x")
  let (t6c, yIdc) ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.multiHeadAttention (t := t5c)
      (n := n) (numHeads := numHeads) (dModel := dModel) (headDim := headDim)
      (h1 := hN) wqIdc wkIdc wvIdc woIdc xIdc (mask := some mask))
  let yCuda ← Utils.cudaValue (s := outShape) t6c yIdc
  let seedCuda : Runtime.Autograd.Cuda.AnyBuffer :=
    { s := outShape, buf := Runtime.Autograd.Cuda.Buffer.full (UInt32.ofNat (Shape.size outShape)) 1.0 }
  let gradsCuda ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.backwardDenseAll (t := t6c) yIdc seedCuda)
  let dxCuda ← Utils.cudaGrad (s := outShape) gradsCuda xIdc
  let dWqCuda ← Utils.cudaGrad (s := shape![dModel, projDim]) gradsCuda wqIdc
  let dWkCuda ← Utils.cudaGrad (s := shape![dModel, projDim]) gradsCuda wkIdc
  let dWvCuda ← Utils.cudaGrad (s := shape![dModel, projDim]) gradsCuda wvIdc
  let dWoCuda ← Utils.cudaGrad (s := shape![projDim, dModel]) gradsCuda woIdc

  -- CUDA composed reference path: the same operation through bmm -> mask -> softmax -> bmm.
  -- Keeping this in the test makes the fused native FlashAttention kernels regression-safe.
  let t0s : Runtime.Autograd.Cuda.Tape := Runtime.Autograd.Cuda.Tape.empty
  let (t1s, wqIds) := Runtime.Autograd.Cuda.Tape.leaf (t := t0s) (Utils.tensorToAnyBuffer wq)
    (name := some "wq")
  let (t2s, wkIds) := Runtime.Autograd.Cuda.Tape.leaf (t := t1s) (Utils.tensorToAnyBuffer wk)
    (name := some "wk")
  let (t3s, wvIds) := Runtime.Autograd.Cuda.Tape.leaf (t := t2s) (Utils.tensorToAnyBuffer wv)
    (name := some "wv")
  let (t4s, woIds) := Runtime.Autograd.Cuda.Tape.leaf (t := t3s) (Utils.tensorToAnyBuffer wo)
    (name := some "wo")
  let (t5s, xIds) := Runtime.Autograd.Cuda.Tape.leaf (t := t4s) (Utils.tensorToAnyBuffer x)
    (name := some "x")
  let (t6s, yIds) ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.multiHeadAttention (t := t5s)
      (n := n) (numHeads := numHeads) (dModel := dModel) (headDim := headDim)
      (h1 := hN) wqIds wkIds wvIds woIds xIds (mask := some mask) (useFlash := false))
  let yCudaComposed ← Utils.cudaValue (s := outShape) t6s yIds
  let seedComposed : Runtime.Autograd.Cuda.AnyBuffer :=
    { s := outShape, buf := Runtime.Autograd.Cuda.Buffer.full (UInt32.ofNat (Shape.size outShape)) 1.0 }
  let gradsComposed ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.backwardDenseAll (t := t6s) yIds seedComposed)
  let dxCudaComposed ← Utils.cudaGrad (s := outShape) gradsComposed xIds
  let dWqCudaComposed ← Utils.cudaGrad (s := shape![dModel, projDim]) gradsComposed wqIds
  let dWkCudaComposed ← Utils.cudaGrad (s := shape![dModel, projDim]) gradsComposed wkIds
  let dWvCudaComposed ← Utils.cudaGrad (s := shape![dModel, projDim]) gradsComposed wvIds
  let dWoCudaComposed ← Utils.cudaGrad (s := shape![projDim, dModel]) gradsComposed woIds

  -- Attention is numerically "busy" (exp/softmax + multiple matmuls). Use a slightly looser tol.
  Utils.assertTensorApprox (s := outShape) "flash vs composed mha forward" yCuda yCudaComposed
    (tol := 2e-2)
  Utils.assertTensorApprox (s := outShape) "flash vs composed mha dx" dxCuda dxCudaComposed
    (tol := 2e-2)
  Utils.assertTensorApprox (s := shape![dModel, projDim]) "flash vs composed mha dWq"
    dWqCuda dWqCudaComposed (tol := 2e-2)
  Utils.assertTensorApprox (s := shape![dModel, projDim]) "flash vs composed mha dWk"
    dWkCuda dWkCudaComposed (tol := 2e-2)
  Utils.assertTensorApprox (s := shape![dModel, projDim]) "flash vs composed mha dWv"
    dWvCuda dWvCudaComposed (tol := 2e-2)
  Utils.assertTensorApprox (s := shape![projDim, dModel]) "flash vs composed mha dWo"
    dWoCuda dWoCudaComposed (tol := 2e-2)

  Utils.assertTensorApprox (s := outShape) "mha forward" yCuda yCpu (tol := 2e-2)
  Utils.assertTensorApprox (s := outShape) "mha dx" dxCuda dxCpu (tol := 2e-2)
  Utils.assertTensorApprox (s := shape![dModel, projDim]) "mha dWq" dWqCuda dWqCpu (tol := 2e-2)
  Utils.assertTensorApprox (s := shape![dModel, projDim]) "mha dWk" dWkCuda dWkCpu (tol := 2e-2)
  Utils.assertTensorApprox (s := shape![dModel, projDim]) "mha dWv" dWvCuda dWvCpu (tol := 2e-2)
  Utils.assertTensorApprox (s := shape![projDim, dModel]) "mha dWo" dWoCuda dWoCpu (tol := 2e-2)

end Attention
end Cuda
end Tests
