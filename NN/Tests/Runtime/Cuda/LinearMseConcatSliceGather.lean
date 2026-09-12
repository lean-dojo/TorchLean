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
# CUDA Kernel Coverage: Linear / Loss / Concat / Slice / Gather

Small forward/backward comparisons (CPU tape vs CUDA tape) for:
- `linear`
- `mse_loss`
- `concat_vectors`
- `slice_leading_axis_range`
- `gather_scalar`, `gather_row`, `gather_scalar_nat_or_zero`
-/

@[expose] public section

namespace Tests
namespace Cuda
namespace LinearMseConcatSliceGather

open Spec TorchLean
open TorchLean TorchLean.Tensor
open Runtime.Autograd

/-- Run CUDA/CPU parity checks for linear, loss, concat, slice, and gather operators. -/
def run : IO Unit := do
  IO.println "=== CUDA kernel coverage: linear/mse/concat/slice/gather ==="

  -- linear + mse_loss (single graph so we exercise both ops in one backward pass)
  IO.println "== linear + mse_loss =="
  let inDim : Nat := 3
  let outDim : Nat := 2
  let sW : Shape := [outDim, inDim]
  let sB : Shape := [outDim]
  let sX : Shape := [inDim]

  let W : Tensor Float sW :=
    (Tensor.from #[
      0.10, -0.20, 0.30,
      -0.05, 0.25, 0.15
    ]).reshape [outDim, inDim] (by dsimp; decide)
  let b : Tensor Float sB := (Tensor.from #[0.01, -0.02]).reshape [outDim] (by dsimp; decide)
  let x : Tensor Float sX := (Tensor.from #[0.50, -0.40, 0.20]).reshape [inDim] (by dsimp; decide)
  let target : Tensor Float sB := (Tensor.from #[0.05, -0.10]).reshape [outDim] (by dsimp; decide)

  -- CPU tape
  let t0 : Tape Float := Tape.empty
  let (t1, wId) := Tape.leaf (t := t0) W (name := some "W")
  let (t2, bId) := Tape.leaf (t := t1) b (name := some "b")
  let (t3, xId) := Tape.leaf (t := t2) x (name := some "x")
  let (t4, yId) ← Utils.okOrThrow
    (Tape.linear (α := Float) (t := t3) (inDim := inDim) (outDim := outDim) wId bId xId)
  let (t5, targetId) := Tape.leaf (t := t4) target (name := some "target")
  let (t6, lossId) ← Utils.okOrThrow (Tape.mseLoss (α := Float) (t := t5) (s := sB) yId targetId)

  let lossCpu ← Utils.cpuValue (s := Shape.scalar) t6 lossId
  let seedCpu : Spec.SomeTensor Float := Spec.SomeTensor.ofTensor (Tensor.scalar 1.0)
  let gradsCpu ← Utils.okOrThrow (Tape.backwardDenseAll (α := Float) (t := t6) lossId seedCpu)
  let dW_cpu ← Utils.cpuGrad (s := sW) gradsCpu wId
  let db_cpu ← Utils.cpuGrad (s := sB) gradsCpu bId
  let dx_cpu ← Utils.cpuGrad (s := sX) gradsCpu xId

  -- CUDA tape
  let t0c : Runtime.Autograd.Cuda.Tape := Runtime.Autograd.Cuda.Tape.empty
  let (t1c, wIdc) := Runtime.Autograd.Cuda.Tape.leaf (t := t0c) (Utils.tensorToAnyBuffer W)
    (name := some "W")
  let (t2c, bIdc) := Runtime.Autograd.Cuda.Tape.leaf (t := t1c) (Utils.tensorToAnyBuffer b)
    (name := some "b")
  let (t3c, xIdc) := Runtime.Autograd.Cuda.Tape.leaf (t := t2c) (Utils.tensorToAnyBuffer x)
    (name := some "x")
  let (t4c, yIdc) ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.linear (t := t3c) (inDim := inDim) (outDim := outDim) wIdc bIdc
      xIdc)
  let (t5c, targetIdc) :=
    Runtime.Autograd.Cuda.Tape.leaf (t := t4c) (Utils.tensorToAnyBuffer target)
    (name := some "target")
  let (t6c, lossIdc) ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.mseLoss (t := t5c) (s := sB) yIdc targetIdc)

  let lossCuda ← Utils.cudaValue (s := Shape.scalar) t6c lossIdc
  let seedCuda : Runtime.Autograd.Cuda.AnyBuffer :=
    { s := Shape.scalar, buf := Runtime.Autograd.Cuda.Buffer.full 1 1.0 }
  let gradsCuda ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.backwardDenseAll (t := t6c) lossIdc seedCuda)
  let dW_cuda ← Utils.cudaGrad (s := sW) gradsCuda wIdc
  let db_cuda ← Utils.cudaGrad (s := sB) gradsCuda bIdc
  let dx_cuda ← Utils.cudaGrad (s := sX) gradsCuda xIdc

  Utils.assertTensorApprox (s := Shape.scalar) "linear+mse loss" lossCuda lossCpu (tol := 2e-3)
  Utils.assertTensorApprox (s := sW) "linear+mse dW" dW_cuda dW_cpu (tol := 2e-3)
  Utils.assertTensorApprox (s := sB) "linear+mse db" db_cuda db_cpu (tol := 2e-3)
  Utils.assertTensorApprox (s := sX) "linear+mse dx" dx_cuda dx_cpu (tol := 2e-3)

  -- concat_vectors + slice_leading_axis_range
  IO.println "== concat_vectors + slice_leading_axis_range =="
  let n : Nat := 2
  let m : Nat := 3
  let sA : Shape := [n]
  let sBv : Shape := [m]
  let sCat : Shape := [n + m]
  let start : Nat := 1
  let len : Nat := 3
  have hSlice : start + len ≤ n + m := by decide

  let a : Tensor Float sA := (Tensor.from #[0.20, -0.10]).reshape [n] (by dsimp; decide)
  let bV : Tensor Float sBv := (Tensor.from #[0.30, 0.05, -0.25]).reshape [m] (by dsimp; decide)

  -- CPU
  let t0s : Tape Float := Tape.empty
  let (t1s, aId) := Tape.leaf (t := t0s) a (name := some "a")
  let (t2s, bId) := Tape.leaf (t := t1s) bV (name := some "b")
  let (t3s, catId) ← Utils.okOrThrow (Tape.concatLeadingAxis (α := Float) (t := t2s)
    (n := n) (m := m) (s := .scalar) aId bId)
  let (t4s, ySliceId) ← Utils.okOrThrow
    (Tape.sliceLeadingAxisRange (α := Float) (t := t3s) (n := n + m) (s := Shape.scalar) catId start
      len hSlice)
  let yCpuSlice ← Utils.cpuValue (s := [len]) t4s ySliceId
  let seedCpuSlice : Spec.SomeTensor Float :=
    Spec.SomeTensor.ofTensor (Tensor.full [len] (1.0 : Float))
  let gradsCpuSlice ← Utils.okOrThrow
    (Tape.backwardDenseAll (α := Float) (t := t4s) ySliceId seedCpuSlice)
  let dA_cpu ← Utils.cpuGrad (s := sA) gradsCpuSlice aId
  let dB_cpu ← Utils.cpuGrad (s := sBv) gradsCpuSlice bId

  -- CUDA
  let t0sc : Runtime.Autograd.Cuda.Tape := Runtime.Autograd.Cuda.Tape.empty
  let (t1sc, aIdc) := Runtime.Autograd.Cuda.Tape.leaf (t := t0sc) (Utils.tensorToAnyBuffer a)
    (name := some "a")
  let (t2sc, bIdc) := Runtime.Autograd.Cuda.Tape.leaf (t := t1sc) (Utils.tensorToAnyBuffer bV)
    (name := some "b")
  let (t3sc, catIdc) ← Utils.okOrThrow (Runtime.Autograd.Cuda.Tape.concatLeadingAxis
    (t := t2sc) (n := n) (m := m) (s := .scalar) aIdc bIdc)
  let (t4sc, ySliceIdc) ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.sliceLeadingAxisRange (t := t3sc) (n := n + m) (s := Shape.scalar)
      catIdc start len hSlice)
  let yCudaSlice ← Utils.cudaValue (s := [len]) t4sc ySliceIdc
  let seedCudaSlice : Runtime.Autograd.Cuda.AnyBuffer :=
    { s := [len]
      , buf := Runtime.Autograd.Cuda.Buffer.full (UInt32.ofNat (Spec.Shape.size [len])) 1.0 }
  let gradsCudaSlice ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.backwardDenseAll (t := t4sc) ySliceIdc seedCudaSlice)
  let dA_cuda ← Utils.cudaGrad (s := sA) gradsCudaSlice aIdc
  let dB_cuda ← Utils.cudaGrad (s := sBv) gradsCudaSlice bIdc

  Utils.assertTensorApprox (s := [len]) "concat+slice forward" yCudaSlice yCpuSlice (tol := 2e-3)
  Utils.assertTensorApprox (s := sA) "concat+slice dA" dA_cuda dA_cpu (tol := 2e-3)
  Utils.assertTensorApprox (s := sBv) "concat+slice dB" dB_cuda dB_cpu (tol := 2e-3)

  -- select
  IO.println "== select =="
  let nG : Nat := 5
  let sG : Shape := [nG]
  let xG : Tensor Float sG :=
    (Tensor.from #[0.10, -0.20, 0.30, 0.05, -0.15]).reshape [nG] (by dsimp; decide)
  let iG : Fin nG := ⟨3, by decide⟩

  -- CPU
  let t0g : Tape Float := Tape.empty
  let (t1g, xGid) := Tape.leaf (t := t0g) xG (name := some "x")
  let (t2g, yGid) ← Utils.okOrThrow
    (Tape.select (α := Float) (s := sG) (t := t1g) xGid 0 iG)
  let yCpuG ← Utils.cpuValue (s := Shape.scalar) t2g yGid
  let seedCpuG : Spec.SomeTensor Float := Spec.SomeTensor.ofTensor (Tensor.scalar 1.0)
  let gradsCpuG ← Utils.okOrThrow (Tape.backwardDenseAll (α := Float) (t := t2g) yGid seedCpuG)
  let dxCpuG ← Utils.cpuGrad (s := sG) gradsCpuG xGid

  -- CUDA
  let t0gc : Runtime.Autograd.Cuda.Tape := Runtime.Autograd.Cuda.Tape.empty
  let (t1gc, xGidc) := Runtime.Autograd.Cuda.Tape.leaf (t := t0gc) (Utils.tensorToAnyBuffer xG)
    (name := some "x")
  let (t2gc, yGidc) ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.select (s := sG) (t := t1gc) xGidc 0 iG)
  let yCudaG ← Utils.cudaValue (s := Shape.scalar) t2gc yGidc
  let seedCudaG : Runtime.Autograd.Cuda.AnyBuffer :=
    { s := Shape.scalar, buf := Runtime.Autograd.Cuda.Buffer.full 1 1.0 }
  let gradsCudaG ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.backwardDenseAll (t := t2gc) yGidc seedCudaG)
  let dxCudaG ← Utils.cudaGrad (s := sG) gradsCudaG xGidc

  Utils.assertTensorApprox (s := Shape.scalar) "gather_scalar forward" yCudaG yCpuG (tol := 2e-3)
  Utils.assertTensorApprox (s := sG) "gather_scalar backward" dxCudaG dxCpuG (tol := 2e-3)

  -- Select a matrix row through the same axis-parametric operation.
  IO.println "== select row =="
  let sRow : Shape := [2]
  let sM : Shape := [3, 2]
  let xM : Tensor Float sM :=
    (Tensor.from #[
      0.10, 0.20,
      -0.30, 0.40,
      0.50, -0.60
    ]).reshape [3, 2] (by dsimp; decide)
  let iRow : Fin (sM.axisSize 0) := ⟨1, by decide⟩

  -- CPU
  let t0r : Tape Float := Tape.empty
  let (t1r, xMid) := Tape.leaf (t := t0r) xM (name := some "x")
  let (t2r, yRowId) ← Utils.okOrThrow
    (Tape.select (α := Float) (s := sM) (t := t1r) xMid 0 iRow)
  let yCpuRow ← Utils.cpuValue (s := sRow) t2r yRowId
  let seedCpuRow : Spec.SomeTensor Float :=
    Spec.SomeTensor.ofTensor (Tensor.full sRow (1.0 : Float))
  let gradsCpuRow ← Utils.okOrThrow
    (Tape.backwardDenseAll (α := Float) (t := t2r) yRowId seedCpuRow)
  let dxCpuRow ← Utils.cpuGrad (s := sM) gradsCpuRow xMid

  -- CUDA
  let t0rc : Runtime.Autograd.Cuda.Tape := Runtime.Autograd.Cuda.Tape.empty
  let (t1rc, xMidc) := Runtime.Autograd.Cuda.Tape.leaf (t := t0rc) (Utils.tensorToAnyBuffer xM)
    (name := some "x")
  let (t2rc, yRowIdc) ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.select (s := sM) (t := t1rc) xMidc 0 iRow)
  let yCudaRow ← Utils.cudaValue (s := sRow) t2rc yRowIdc
  let seedCudaRow : Runtime.Autograd.Cuda.AnyBuffer :=
    { s := sRow
      , buf := Runtime.Autograd.Cuda.Buffer.full (UInt32.ofNat (Spec.Shape.size sRow)) 1.0 }
  let gradsCudaRow ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.backwardDenseAll (t := t2rc) yRowIdc seedCudaRow)
  let dxCudaRow ← Utils.cudaGrad (s := sM) gradsCudaRow xMidc

  Utils.assertTensorApprox (s := sRow) "gather_row forward" yCudaRow yCpuRow (tol := 2e-3)
  Utils.assertTensorApprox (s := sM) "gather_row backward" dxCudaRow dxCpuRow (tol := 2e-3)

end LinearMseConcatSliceGather
end Cuda
end Tests
