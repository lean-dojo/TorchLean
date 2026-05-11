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
# CUDA Kernel Coverage: Gather / Scatter

Compares CPU eager tape vs CUDA eager tape for:
- `gather_vec_nat`
- `gather_rows_nat`
- `scatter_add_vec`
- `scatter_add_row`
-/

@[expose] public section

namespace Tests
namespace Cuda
namespace GatherScatter

open Spec
open Tensor
open Runtime.Autograd

def runGatherVec : IO Unit := do
  IO.println "== gather_vec_nat =="

  let n : Nat := 5
  let k : Nat := 3
  let sX : Shape := shape![n]
  let sY : Shape := shape![k]
  let x : Tensor Float sX :=
    tensorND! [n] [0.10, -0.20, 0.30, 0.40, -0.50]
  let idx : Tensor Nat (shape![k]) :=
    tensorND! [k] [0, 2, 4]

  -- CPU
  let t0 : Tape Float := Tape.empty
  let (t1, xId) := Tape.leaf (t := t0) x (name := some "x")
  let (t2, yId) ← Utils.okOrThrow
    (Tape.gatherVecNat (α := Float) (t := t1) (n := n) (k := k) xId idx)
  let yCpu ← Utils.cpuValue (s := sY) t2 yId
  let seedCpu : Runtime.AnyTensor Float := AnyTensor.mk (fill (1.0 : Float) sY)
  let gradsCpu ← Utils.okOrThrow (Tape.backwardDenseAll (α := Float) (t := t2) yId seedCpu)
  let dxCpu ← Utils.cpuGrad (s := sX) gradsCpu xId

  -- CUDA
  let t0c : Runtime.Autograd.Cuda.Tape := Runtime.Autograd.Cuda.Tape.empty
  let (t1c, xIdc) := Runtime.Autograd.Cuda.Tape.leaf (t := t0c) (Utils.tensorToAnyBuffer x)
    (name := some "x")
  let (t2c, yIdc) ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.gatherVecNat (t := t1c) (n := n) (k := k) xIdc idx)
  let yCuda ← Utils.cudaValue (s := sY) t2c yIdc
  let seedCuda : Runtime.Autograd.Cuda.AnyBuffer :=
    { s := sY, buf := Runtime.Autograd.Cuda.Buffer.full (UInt32.ofNat (Shape.size sY)) 1.0 }
  let gradsCuda ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.backwardDenseAll (t := t2c) yIdc seedCuda)
  let dxCuda ← Utils.cudaGrad (s := sX) gradsCuda xIdc

  Utils.assertTensorApprox (s := sY) "gather_vec forward" yCuda yCpu (tol := 1e-6)
  Utils.assertTensorApprox (s := sX) "gather_vec dx" dxCuda dxCpu (tol := 1e-6)

def runScatterVec : IO Unit := do
  IO.println "== scatter_add_vec =="

  let n : Nat := 5
  let sX : Shape := shape![n]
  let x : Tensor Float sX := tensorND! [n] [1.0, 2.0, 3.0, 4.0, 5.0]
  let v : Tensor Float Shape.scalar := Tensor.scalar 0.7
  let i : Fin n := ⟨2, by decide⟩

  -- CPU
  let t0 : Tape Float := Tape.empty
  let (t1, xId) := Tape.leaf (t := t0) x (name := some "x")
  let (t2, vId) := Tape.leaf (t := t1) v (name := some "v")
  let (t3, yId) ← Utils.okOrThrow
    (Tape.scatterAddVec (α := Float) (t := t2) (n := n) xId vId i)
  let yCpu ← Utils.cpuValue (s := sX) t3 yId
  let seedCpu : Runtime.AnyTensor Float := AnyTensor.mk (fill (1.0 : Float) sX)
  let gradsCpu ← Utils.okOrThrow (Tape.backwardDenseAll (α := Float) (t := t3) yId seedCpu)
  let dxCpu ← Utils.cpuGrad (s := sX) gradsCpu xId
  let dvCpu ← Utils.cpuGrad (s := Shape.scalar) gradsCpu vId

  -- CUDA
  let t0c : Runtime.Autograd.Cuda.Tape := Runtime.Autograd.Cuda.Tape.empty
  let (t1c, xIdc) := Runtime.Autograd.Cuda.Tape.leaf (t := t0c) (Utils.tensorToAnyBuffer x)
    (name := some "x")
  let (t2c, vIdc) := Runtime.Autograd.Cuda.Tape.leaf (t := t1c) (Utils.tensorToAnyBuffer v)
    (name := some "v")
  let (t3c, yIdc) ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.scatterAddVec (t := t2c) (n := n) xIdc vIdc i)
  let yCuda ← Utils.cudaValue (s := sX) t3c yIdc
  let seedCuda : Runtime.Autograd.Cuda.AnyBuffer :=
    { s := sX, buf := Runtime.Autograd.Cuda.Buffer.full (UInt32.ofNat (Shape.size sX)) 1.0 }
  let gradsCuda ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.backwardDenseAll (t := t3c) yIdc seedCuda)
  let dxCuda ← Utils.cudaGrad (s := sX) gradsCuda xIdc
  let dvCuda ← Utils.cudaGrad (s := Shape.scalar) gradsCuda vIdc

  Utils.assertTensorApprox (s := sX) "scatter_add_vec forward" yCuda yCpu (tol := 1e-6)
  Utils.assertTensorApprox (s := sX) "scatter_add_vec dx" dxCuda dxCpu (tol := 1e-6)
  Utils.assertTensorApprox (s := Shape.scalar) "scatter_add_vec dv" dvCuda dvCpu (tol := 1e-6)

def runGatherRows : IO Unit := do
  IO.println "== gather_rows_nat =="

  let rows : Nat := 3
  let cols : Nat := 2
  let k : Nat := 2
  let sX : Shape := shape![rows, cols]
  let sY : Shape := shape![k, cols]
  let x : Tensor Float sX :=
    tensorND! [rows, cols] [
      0.10, 0.20,
      -0.30, 0.40,
      0.50, -0.60
    ]
  let idx : Tensor Nat (shape![k]) :=
    tensorND! [k] [0, 2]

  -- CPU
  let t0 : Tape Float := Tape.empty
  let (t1, xId) := Tape.leaf (t := t0) x (name := some "x")
  let (t2, yId) ← Utils.okOrThrow
    (Tape.gatherRowsNat (α := Float) (t := t1) (rows := rows) (cols := cols) (k := k) xId idx)
  let yCpu ← Utils.cpuValue (s := sY) t2 yId
  let seedCpu : Runtime.AnyTensor Float := AnyTensor.mk (fill (1.0 : Float) sY)
  let gradsCpu ← Utils.okOrThrow (Tape.backwardDenseAll (α := Float) (t := t2) yId seedCpu)
  let dxCpu ← Utils.cpuGrad (s := sX) gradsCpu xId

  -- CUDA
  let t0c : Runtime.Autograd.Cuda.Tape := Runtime.Autograd.Cuda.Tape.empty
  let (t1c, xIdc) := Runtime.Autograd.Cuda.Tape.leaf (t := t0c) (Utils.tensorToAnyBuffer x)
    (name := some "x")
  let (t2c, yIdc) ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.gatherRowsNat (t := t1c) (rows := rows) (cols := cols) (k := k)
      xIdc idx)
  let yCuda ← Utils.cudaValue (s := sY) t2c yIdc
  let seedCuda : Runtime.Autograd.Cuda.AnyBuffer :=
    { s := sY, buf := Runtime.Autograd.Cuda.Buffer.full (UInt32.ofNat (Shape.size sY)) 1.0 }
  let gradsCuda ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.backwardDenseAll (t := t2c) yIdc seedCuda)
  let dxCuda ← Utils.cudaGrad (s := sX) gradsCuda xIdc

  Utils.assertTensorApprox (s := sY) "gather_rows forward" yCuda yCpu (tol := 1e-6)
  Utils.assertTensorApprox (s := sX) "gather_rows dx" dxCuda dxCpu (tol := 1e-6)

def runScatterRow : IO Unit := do
  IO.println "== scatter_add_row =="

  let rows : Nat := 3
  let cols : Nat := 2
  let sX : Shape := shape![rows, cols]
  let x : Tensor Float sX :=
    tensorND! [rows, cols] [
      1.0, 2.0,
      3.0, 4.0,
      5.0, 6.0
    ]
  let v : Tensor Float (shape![cols]) :=
    tensorND! [cols] [0.25, -0.50]
  let i : Fin rows := ⟨1, by decide⟩

  -- CPU
  let t0 : Tape Float := Tape.empty
  let (t1, xId) := Tape.leaf (t := t0) x (name := some "x")
  let (t2, vId) := Tape.leaf (t := t1) v (name := some "v")
  let (t3, yId) ← Utils.okOrThrow
    (Tape.scatterAddRow (α := Float) (t := t2) (rows := rows) (cols := cols) xId vId i)
  let yCpu ← Utils.cpuValue (s := sX) t3 yId
  let seedCpu : Runtime.AnyTensor Float := AnyTensor.mk (fill (1.0 : Float) sX)
  let gradsCpu ← Utils.okOrThrow (Tape.backwardDenseAll (α := Float) (t := t3) yId seedCpu)
  let dxCpu ← Utils.cpuGrad (s := sX) gradsCpu xId
  let dvCpu ← Utils.cpuGrad (s := shape![cols]) gradsCpu vId

  -- CUDA
  let t0c : Runtime.Autograd.Cuda.Tape := Runtime.Autograd.Cuda.Tape.empty
  let (t1c, xIdc) := Runtime.Autograd.Cuda.Tape.leaf (t := t0c) (Utils.tensorToAnyBuffer x)
    (name := some "x")
  let (t2c, vIdc) := Runtime.Autograd.Cuda.Tape.leaf (t := t1c) (Utils.tensorToAnyBuffer v)
    (name := some "v")
  let (t3c, yIdc) ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.scatterAddRow (t := t2c) (rows := rows) (cols := cols) xIdc vIdc i)
  let yCuda ← Utils.cudaValue (s := sX) t3c yIdc
  let seedCuda : Runtime.Autograd.Cuda.AnyBuffer :=
    { s := sX, buf := Runtime.Autograd.Cuda.Buffer.full (UInt32.ofNat (Shape.size sX)) 1.0 }
  let gradsCuda ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.backwardDenseAll (t := t3c) yIdc seedCuda)
  let dxCuda ← Utils.cudaGrad (s := sX) gradsCuda xIdc
  let dvCuda ← Utils.cudaGrad (s := shape![cols]) gradsCuda vIdc

  Utils.assertTensorApprox (s := sX) "scatter_add_row forward" yCuda yCpu (tol := 1e-6)
  Utils.assertTensorApprox (s := sX) "scatter_add_row dx" dxCuda dxCpu (tol := 1e-6)
  Utils.assertTensorApprox (s := shape![cols]) "scatter_add_row dv" dvCuda dvCpu (tol := 1e-6)

def run : IO Unit := do
  IO.println "=== CUDA kernel coverage: gather/scatter ==="
  runGatherVec
  runScatterVec
  runGatherRows
  runScatterRow

end GatherScatter
end Cuda
end Tests
