/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Engine.FastKernels
public import NN.Runtime.Autograd.Engine.Cuda.Ops
public import NN.Tensor
public import NN.Tests.Runtime.Cuda.Utils

/-!
# CUDA Kernel Coverage: Matrix Multiplication

Compares CPU eager tape vs CUDA eager tape for:
- ordinary matrix multiplication,
- matrix multiplication with a batch prefix,
- explicit fast-kernel matmul precision dispatch (`fp32`/`fp64`)
-/

@[expose] public section

namespace Tests
namespace Cuda
namespace Matmul

open Spec TorchLean
open TorchLean TorchLean.Tensor
open Runtime.Autograd

def runMatmul : IO Unit := do
  IO.println "== matmul =="

  let m : Nat := 2
  let n : Nat := 3
  let p : Nat := 2
  let sA : Shape := [m, n]
  let sB : Shape := [n, p]
  let sY : Shape := [m, p]

  let a : Tensor Float sA :=
    (Tensor.from #[
      0.10, 0.20, 0.30,
      -0.10, 0.05, 0.15
    ]).reshape [m, n] (by dsimp; decide)
  let b : Tensor Float sB :=
    (Tensor.from #[
      0.20, -0.10,
      0.00, 0.30,
      -0.20, 0.10
    ]).reshape [n, p] (by dsimp; decide)

  -- CPU
  let t0 : Tape Float := Tape.empty
  let (t1, aId) := Tape.leaf (t := t0) a (name := some "a")
  let (t2, bId) := Tape.leaf (t := t1) b (name := some "b")
  let (t3, yId) ← Utils.okOrThrow
    (Tape.matmul (α := Float) (t := t2) (m := m) (n := n) (p := p) aId bId)
  let yCpu ← Utils.cpuValue (s := sY) t3 yId
  let seedCpu : Spec.SomeTensor Float := Spec.SomeTensor.ofTensor (Tensor.full sY (1.0 : Float))
  let gradsCpu ← Utils.okOrThrow (Tape.backwardDenseAll (α := Float) (t := t3) yId seedCpu)
  let dACpu ← Utils.cpuGrad (s := sA) gradsCpu aId
  let dBCpu ← Utils.cpuGrad (s := sB) gradsCpu bId

  -- CUDA
  let t0c : Runtime.Autograd.Cuda.Tape := Runtime.Autograd.Cuda.Tape.empty
  let (t1c, aIdc) := Runtime.Autograd.Cuda.Tape.leaf (t := t0c) (Utils.tensorToAnyBuffer a)
    (name := some "a")
  let (t2c, bIdc) := Runtime.Autograd.Cuda.Tape.leaf (t := t1c) (Utils.tensorToAnyBuffer b)
    (name := some "b")
  let (t3c, yIdc) ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.Internal.matmul
      (t := t2c) (m := m) (n := n) (p := p) aIdc bIdc)
  let yCuda ← Utils.cudaValue (s := sY) t3c yIdc
  let seedCuda : Runtime.Autograd.Cuda.AnyBuffer :=
    { s := sY, buf := Runtime.Autograd.Cuda.Buffer.full (UInt32.ofNat (Spec.Shape.size sY)) 1.0 }
  let gradsCuda ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.backwardDenseAll (t := t3c) yIdc seedCuda)
  let dACuda ← Utils.cudaGrad (s := sA) gradsCuda aIdc
  let dBCuda ← Utils.cudaGrad (s := sB) gradsCuda bIdc

  Utils.assertTensorApprox (s := sY) "matmul forward" yCuda yCpu (tol := 5e-3)
  Utils.assertTensorApprox (s := sA) "matmul dA" dACuda dACpu (tol := 5e-3)
  Utils.assertTensorApprox (s := sB) "matmul dB" dBCuda dBCpu (tol := 5e-3)

def runBatchedMatmul : IO Unit := do
  IO.println "== batched matmul =="

  let batch : Nat := 2
  let m : Nat := 2
  let n : Nat := 2
  let p : Nat := 2
  let sA : Shape := [batch, m, n]
  let sB : Shape := [batch, n, p]
  let sY : Shape := [batch, m, p]

  let a : Tensor Float sA :=
    (Tensor.from #[
      0.10, 0.20,
      0.30, 0.40,
      -0.10, 0.05,
      0.15, -0.20
    ]).reshape [batch, m, n] (by dsimp; decide)
  let b : Tensor Float sB :=
    (Tensor.from #[
      0.20, 0.10,
      -0.10, 0.30,
      0.05, -0.20,
      0.25, 0.10
    ]).reshape [batch, n, p] (by dsimp; decide)

  -- CPU
  let t0 : Tape Float := Tape.empty
  let (t1, aId) := Tape.leaf (t := t0) a (name := some "a")
  let (t2, bId) := Tape.leaf (t := t1) b (name := some "b")
  let (t3, yId) ← Utils.okOrThrow
    (Tape.matmul (α := Float) (t := t2) (m := m) (n := n) (p := p) aId bId
      (batchA := [batch]) (batchB := [batch]) (batch := [batch]))
  let yCpu ← Utils.cpuValue (s := sY) t3 yId
  let seedCpu : Spec.SomeTensor Float := Spec.SomeTensor.ofTensor (Tensor.full sY (1.0 : Float))
  let gradsCpu ← Utils.okOrThrow (Tape.backwardDenseAll (α := Float) (t := t3) yId seedCpu)
  let dACpu ← Utils.cpuGrad (s := sA) gradsCpu aId
  let dBCpu ← Utils.cpuGrad (s := sB) gradsCpu bId

  -- CUDA
  let t0c : Runtime.Autograd.Cuda.Tape := Runtime.Autograd.Cuda.Tape.empty
  let (t1c, aIdc) := Runtime.Autograd.Cuda.Tape.leaf (t := t0c) (Utils.tensorToAnyBuffer a)
    (name := some "a")
  let (t2c, bIdc) := Runtime.Autograd.Cuda.Tape.leaf (t := t1c) (Utils.tensorToAnyBuffer b)
    (name := some "b")
  let (t3c, yIdc) ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.Internal.matmulFlattened
      (t := t2c) (batch := batch) (m := m) (n := n) (p := p) aIdc bIdc)
  let yCuda ← Utils.cudaValue (s := sY) t3c yIdc
  let seedCuda : Runtime.Autograd.Cuda.AnyBuffer :=
    { s := sY, buf := Runtime.Autograd.Cuda.Buffer.full (UInt32.ofNat (Spec.Shape.size sY)) 1.0 }
  let gradsCuda ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.backwardDenseAll (t := t3c) yIdc seedCuda)
  let dACuda ← Utils.cudaGrad (s := sA) gradsCuda aIdc
  let dBCuda ← Utils.cudaGrad (s := sB) gradsCuda bIdc

  Utils.assertTensorApprox (s := sY) "batched matmul forward" yCuda yCpu (tol := 5e-3)
  Utils.assertTensorApprox (s := sA) "batched matmul dA" dACuda dACpu (tol := 5e-3)
  Utils.assertTensorApprox (s := sB) "batched matmul dB" dBCuda dBCpu (tol := 5e-3)

def runFastMatmulPrecision : IO Unit := do
  IO.println "== fast matmul precision =="

  let m : Nat := 2
  let n : Nat := 3
  let p : Nat := 2
  let sA : Shape := [m, n]
  let sB : Shape := [n, p]
  let sY : Shape := [m, p]

  let a : Tensor Float sA :=
    (Tensor.from #[
      0.10, 0.20, 0.30,
      -0.10, 0.05, 0.15
    ]).reshape [m, n] (by dsimp; decide)
  let b : Tensor Float sB :=
    (Tensor.from #[
      0.20, -0.10,
      0.00, 0.30,
      -0.20, 0.10
    ]).reshape [n, p] (by dsimp; decide)

  let yCpu := FastKernels.matmulReference (α := Float) (m := m) (n := n) (p := p) a b
  let yFp32 ← IO.ofExcept (FastKernels.Cuda.matmulLibTorch .fp32 (m := m) (n := n) (p := p) a b)
  let yFp64 ← IO.ofExcept (FastKernels.Cuda.matmulLibTorch .fp64 (m := m) (n := n) (p := p) a b)

  Utils.assertTensorApprox (s := sY) "fast matmul fp32" yFp32 yCpu (tol := 5e-3)
  Utils.assertTensorApprox (s := sY) "fast matmul fp64" yFp64 yCpu (tol := 1e-9)

def run : IO Unit := do
  IO.println "=== CUDA kernel coverage: matrix multiplication ==="
  runMatmul
  runBatchedMatmul
  runFastMatmulPrecision

end Matmul
end Cuda
end Tests
