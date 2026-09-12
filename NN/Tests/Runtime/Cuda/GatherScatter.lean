/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Engine.Core
public import NN.Runtime.Autograd.Engine.Cuda.Ops
public import NN.Tensor
public import NN.Tests.Runtime.Cuda.Utils

/-!
# CUDA Kernel Coverage: Axis Indexing

Compares CPU eager tape vs CUDA eager tape for:
- `indexSelect` on vectors and matrices;
- `scatterAdd` with vector and row slices.
-/

@[expose] public section

namespace Tests
namespace Cuda
namespace GatherScatter

open Spec TorchLean
open TorchLean TorchLean.Tensor
open Runtime.Autograd

def runGatherVec : IO Unit := do
  IO.println "== indexSelect vector =="

  let n : Nat := 5
  let k : Nat := 3
  let sX : Shape := [n]
  let sY : Shape := [k]
  let x : Tensor Float sX :=
    (Tensor.from #[0.10, -0.20, 0.30, 0.40, -0.50]).reshape [n] (by dsimp; decide)
  let indices : Fin k → Fin n :=
    ![⟨0, by decide⟩, ⟨2, by decide⟩, ⟨4, by decide⟩]
  let idx : Tensor (Fin n) [k] := Tensor.ofFn indices

  -- CPU
  let t0 : Tape Float := Tape.empty
  let (t1, xId) := Tape.leaf (t := t0) x (name := some "x")
  let (t2, yId) ← Utils.okOrThrow
    (Tape.indexSelect (α := Float) (s := sX) (t := t1) xId 0 k idx)
  let yCpu ← Utils.cpuValue (s := sY) t2 yId
  let seedCpu : Spec.SomeTensor Float := Spec.SomeTensor.ofTensor (Tensor.full sY (1.0 : Float))
  let gradsCpu ← Utils.okOrThrow (Tape.backwardDenseAll (α := Float) (t := t2) yId seedCpu)
  let dxCpu ← Utils.cpuGrad (s := sX) gradsCpu xId

  -- CUDA
  let t0c : Runtime.Autograd.Cuda.Tape := Runtime.Autograd.Cuda.Tape.empty
  let (t1c, xIdc) := Runtime.Autograd.Cuda.Tape.leaf (t := t0c) (Utils.tensorToAnyBuffer x)
    (name := some "x")
  let (t2c, yIdc) ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.indexSelect (s := sX) (t := t1c) xIdc 0 k idx)
  let yCuda ← Utils.cudaValue (s := sY) t2c yIdc
  let seedCuda : Runtime.Autograd.Cuda.AnyBuffer :=
    { s := sY, buf := Runtime.Autograd.Cuda.Buffer.full (UInt32.ofNat (Spec.Shape.size sY)) 1.0 }
  let gradsCuda ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.backwardDenseAll (t := t2c) yIdc seedCuda)
  let dxCuda ← Utils.cudaGrad (s := sX) gradsCuda xIdc

  Utils.assertTensorApprox (s := sY) "gather_vec forward" yCuda yCpu (tol := 1e-6)
  Utils.assertTensorApprox (s := sX) "gather_vec dx" dxCuda dxCpu (tol := 1e-6)

def runScatterVec : IO Unit := do
  IO.println "== scatter_add_vec =="

  let n : Nat := 5
  let sX : Shape := [n]
  let x : Tensor Float sX := (Tensor.from #[1.0, 2.0, 3.0, 4.0, 5.0]).reshape [n] (by dsimp; decide)
  let v : Tensor Float [1] := (Tensor.from #[0.7]).reshape [1] (by dsimp; decide)
  let i : Fin n := ⟨2, by decide⟩
  let idx : Tensor (Fin n) [1] := Tensor.ofFn (fun _ => i)

  -- CPU
  let t0 : Tape Float := Tape.empty
  let (t1, xId) := Tape.leaf (t := t0) x (name := some "x")
  let (t2, vId) := Tape.leaf (t := t1) v (name := some "v")
  let (t3, yId) ← Utils.okOrThrow
    (Tape.scatterAdd (α := Float) (s := sX) (t := t2) xId vId 0 1 idx)
  let yCpu ← Utils.cpuValue (s := sX) t3 yId
  let seedCpu : Spec.SomeTensor Float := Spec.SomeTensor.ofTensor (Tensor.full sX (1.0 : Float))
  let gradsCpu ← Utils.okOrThrow (Tape.backwardDenseAll (α := Float) (t := t3) yId seedCpu)
  let dxCpu ← Utils.cpuGrad (s := sX) gradsCpu xId
  let dvCpu ← Utils.cpuGrad (s := [1]) gradsCpu vId

  -- CUDA
  let t0c : Runtime.Autograd.Cuda.Tape := Runtime.Autograd.Cuda.Tape.empty
  let (t1c, xIdc) := Runtime.Autograd.Cuda.Tape.leaf (t := t0c) (Utils.tensorToAnyBuffer x)
    (name := some "x")
  let (t2c, vIdc) := Runtime.Autograd.Cuda.Tape.leaf (t := t1c) (Utils.tensorToAnyBuffer v)
    (name := some "v")
  let (t3c, yIdc) ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.scatterAdd (s := sX) (t := t2c) xIdc vIdc 0 1 idx)
  let yCuda ← Utils.cudaValue (s := sX) t3c yIdc
  let seedCuda : Runtime.Autograd.Cuda.AnyBuffer :=
    { s := sX, buf := Runtime.Autograd.Cuda.Buffer.full (UInt32.ofNat (Spec.Shape.size sX)) 1.0 }
  let gradsCuda ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.backwardDenseAll (t := t3c) yIdc seedCuda)
  let dxCuda ← Utils.cudaGrad (s := sX) gradsCuda xIdc
  let dvCuda ← Utils.cudaGrad (s := [1]) gradsCuda vIdc

  Utils.assertTensorApprox (s := sX) "scatter_add_vec forward" yCuda yCpu (tol := 1e-6)
  Utils.assertTensorApprox (s := sX) "scatter_add_vec dx" dxCuda dxCpu (tol := 1e-6)
  Utils.assertTensorApprox (s := [1]) "scatter_add_vec dv" dvCuda dvCpu (tol := 1e-6)

def runGatherRows : IO Unit := do
  IO.println "== indexSelect rows =="

  let rows : Nat := 3
  let cols : Nat := 2
  let k : Nat := 2
  let sX : Shape := [rows, cols]
  let sY : Shape := [k, cols]
  let x : Tensor Float sX :=
    (Tensor.from #[
      0.10, 0.20,
      -0.30, 0.40,
      0.50, -0.60
    ]).reshape [rows, cols] (by dsimp; decide)
  let indices : Fin k → Fin rows := ![⟨0, by decide⟩, ⟨2, by decide⟩]
  let idx : Tensor (Fin rows) [k] := Tensor.ofFn indices

  -- CPU
  let t0 : Tape Float := Tape.empty
  let (t1, xId) := Tape.leaf (t := t0) x (name := some "x")
  let (t2, yId) ← Utils.okOrThrow
    (Tape.indexSelect (α := Float) (s := sX) (t := t1) xId 0 k idx)
  let yCpu ← Utils.cpuValue (s := sY) t2 yId
  let seedCpu : Spec.SomeTensor Float := Spec.SomeTensor.ofTensor (Tensor.full sY (1.0 : Float))
  let gradsCpu ← Utils.okOrThrow (Tape.backwardDenseAll (α := Float) (t := t2) yId seedCpu)
  let dxCpu ← Utils.cpuGrad (s := sX) gradsCpu xId

  -- CUDA
  let t0c : Runtime.Autograd.Cuda.Tape := Runtime.Autograd.Cuda.Tape.empty
  let (t1c, xIdc) := Runtime.Autograd.Cuda.Tape.leaf (t := t0c) (Utils.tensorToAnyBuffer x)
    (name := some "x")
  let (t2c, yIdc) ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.indexSelect (s := sX) (t := t1c) xIdc 0 k idx)
  let yCuda ← Utils.cudaValue (s := sY) t2c yIdc
  let seedCuda : Runtime.Autograd.Cuda.AnyBuffer :=
    { s := sY, buf := Runtime.Autograd.Cuda.Buffer.full (UInt32.ofNat (Spec.Shape.size sY)) 1.0 }
  let gradsCuda ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.backwardDenseAll (t := t2c) yIdc seedCuda)
  let dxCuda ← Utils.cudaGrad (s := sX) gradsCuda xIdc

  Utils.assertTensorApprox (s := sY) "gather_rows forward" yCuda yCpu (tol := 1e-6)
  Utils.assertTensorApprox (s := sX) "gather_rows dx" dxCuda dxCpu (tol := 1e-6)

def runScatterRow : IO Unit := do
  IO.println "== scatter_add_row =="

  let rows : Nat := 3
  let cols : Nat := 2
  let sX : Shape := [rows, cols]
  let x : Tensor Float sX :=
    (Tensor.from #[
      1.0, 2.0,
      3.0, 4.0,
      5.0, 6.0
    ]).reshape [rows, cols] (by dsimp; decide)
  let v : Tensor Float [1, cols] :=
    (Tensor.from #[0.25, -0.50]).reshape [1, cols] (by dsimp; decide)
  let i : Fin rows := ⟨1, by decide⟩
  let idx : Tensor (Fin rows) [1] := Tensor.ofFn (fun _ => i)

  -- CPU
  let t0 : Tape Float := Tape.empty
  let (t1, xId) := Tape.leaf (t := t0) x (name := some "x")
  let (t2, vId) := Tape.leaf (t := t1) v (name := some "v")
  let (t3, yId) ← Utils.okOrThrow
    (Tape.scatterAdd (α := Float) (s := sX) (t := t2) xId vId 0 1 idx)
  let yCpu ← Utils.cpuValue (s := sX) t3 yId
  let seedCpu : Spec.SomeTensor Float := Spec.SomeTensor.ofTensor (Tensor.full sX (1.0 : Float))
  let gradsCpu ← Utils.okOrThrow (Tape.backwardDenseAll (α := Float) (t := t3) yId seedCpu)
  let dxCpu ← Utils.cpuGrad (s := sX) gradsCpu xId
  let dvCpu ← Utils.cpuGrad (s := [1, cols]) gradsCpu vId

  -- CUDA
  let t0c : Runtime.Autograd.Cuda.Tape := Runtime.Autograd.Cuda.Tape.empty
  let (t1c, xIdc) := Runtime.Autograd.Cuda.Tape.leaf (t := t0c) (Utils.tensorToAnyBuffer x)
    (name := some "x")
  let (t2c, vIdc) := Runtime.Autograd.Cuda.Tape.leaf (t := t1c) (Utils.tensorToAnyBuffer v)
    (name := some "v")
  let (t3c, yIdc) ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.scatterAdd (s := sX) (t := t2c) xIdc vIdc 0 1 idx)
  let yCuda ← Utils.cudaValue (s := sX) t3c yIdc
  let seedCuda : Runtime.Autograd.Cuda.AnyBuffer :=
    { s := sX, buf := Runtime.Autograd.Cuda.Buffer.full (UInt32.ofNat (Spec.Shape.size sX)) 1.0 }
  let gradsCuda ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.backwardDenseAll (t := t3c) yIdc seedCuda)
  let dxCuda ← Utils.cudaGrad (s := sX) gradsCuda xIdc
  let dvCuda ← Utils.cudaGrad (s := [1, cols]) gradsCuda vIdc

  Utils.assertTensorApprox (s := sX) "scatter_add_row forward" yCuda yCpu (tol := 1e-6)
  Utils.assertTensorApprox (s := sX) "scatter_add_row dx" dxCuda dxCpu (tol := 1e-6)
  Utils.assertTensorApprox (s := [1, cols]) "scatter_add_row dv" dvCuda dvCpu (tol := 1e-6)

def run : IO Unit := do
  IO.println "=== CUDA kernel coverage: gather/scatter ==="
  runGatherVec
  runScatterVec
  runGatherRows
  runScatterRow

end GatherScatter
end Cuda
end Tests
