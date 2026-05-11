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
# CUDA Kernel Coverage: Linear / Loss / Concat / Slice / Gather

Small forward/backward comparisons (CPU tape vs CUDA tape) for:
- `linear`
- `mse_loss`
- `concat_vectors`
- `slice_range0`
- `gather_scalar`, `gather_row`, `gather_scalar_nat`
-/

@[expose] public section

set_option maxRecDepth 2048

namespace Tests
namespace Cuda
namespace LinearMseConcatSliceGather

open Spec
open Tensor
open Runtime.Autograd

def run : IO Unit := do
  IO.println "=== CUDA kernel coverage: linear/mse/concat/slice/gather ==="

  -- linear + mse_loss (single graph so we exercise both ops in one backward pass)
  IO.println "== linear + mse_loss =="
  let inDim : Nat := 3
  let outDim : Nat := 2
  let sW : Shape := shape![outDim, inDim]
  let sB : Shape := shape![outDim]
  let sX : Shape := shape![inDim]

  let W : Tensor Float sW :=
    tensorND! [outDim, inDim] [
      0.10, -0.20, 0.30,
      -0.05, 0.25, 0.15
    ]
  let b : Tensor Float sB := tensorND! [outDim] [0.01, -0.02]
  let x : Tensor Float sX := tensorND! [inDim] [0.50, -0.40, 0.20]
  let target : Tensor Float sB := tensorND! [outDim] [0.05, -0.10]

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
  let seedCpu : Runtime.AnyTensor Float := AnyTensor.mk (Tensor.scalar 1.0)
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
  let (t5c, targetIdc) := Runtime.Autograd.Cuda.Tape.leaf (t := t4c) (Utils.tensorToAnyBuffer target)
    (name := some "target")
  let (t6c, lossIdc) ← Utils.okOrThrow (Runtime.Autograd.Cuda.Tape.mseLoss (t := t5c) (s := sB) yIdc targetIdc)

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

  -- concat_vectors + slice_range0
  IO.println "== concat_vectors + slice_range0 =="
  let n : Nat := 2
  let m : Nat := 3
  let sA : Shape := shape![n]
  let sBv : Shape := shape![m]
  let sCat : Shape := shape![n + m]
  let start : Nat := 1
  let len : Nat := 3
  have hSlice : len + start ≤ n + m := by decide

  let a : Tensor Float sA := tensorND! [n] [0.20, -0.10]
  let bV : Tensor Float sBv := tensorND! [m] [0.30, 0.05, -0.25]

  -- CPU
  let t0s : Tape Float := Tape.empty
  let (t1s, aId) := Tape.leaf (t := t0s) a (name := some "a")
  let (t2s, bId) := Tape.leaf (t := t1s) bV (name := some "b")
  let (t3s, catId) ← Utils.okOrThrow (Tape.concatVectors (α := Float) (t := t2s) (n := n) (m := m) aId bId)
  let (t4s, ySliceId) ← Utils.okOrThrow (Tape.sliceRange0 (α := Float) (t := t3s) (n := n + m) (s := Shape.scalar) catId start len hSlice)
  let yCpuSlice ← Utils.cpuValue (s := shape![len]) t4s ySliceId
  let seedCpuSlice : Runtime.AnyTensor Float := AnyTensor.mk (fill (1.0 : Float) (shape![len]))
  let gradsCpuSlice ← Utils.okOrThrow (Tape.backwardDenseAll (α := Float) (t := t4s) ySliceId seedCpuSlice)
  let dA_cpu ← Utils.cpuGrad (s := sA) gradsCpuSlice aId
  let dB_cpu ← Utils.cpuGrad (s := sBv) gradsCpuSlice bId

  -- CUDA
  let t0sc : Runtime.Autograd.Cuda.Tape := Runtime.Autograd.Cuda.Tape.empty
  let (t1sc, aIdc) := Runtime.Autograd.Cuda.Tape.leaf (t := t0sc) (Utils.tensorToAnyBuffer a)
    (name := some "a")
  let (t2sc, bIdc) := Runtime.Autograd.Cuda.Tape.leaf (t := t1sc) (Utils.tensorToAnyBuffer bV)
    (name := some "b")
  let (t3sc, catIdc) ← Utils.okOrThrow (Runtime.Autograd.Cuda.Tape.concatVectors (t := t2sc) (n := n) (m := m) aIdc bIdc)
  let (t4sc, ySliceIdc) ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.sliceRange0 (t := t3sc) (n := n + m) (s := Shape.scalar) catIdc start len hSlice)
  let yCudaSlice ← Utils.cudaValue (s := shape![len]) t4sc ySliceIdc
  let seedCudaSlice : Runtime.Autograd.Cuda.AnyBuffer :=
    { s := shape![len]
      , buf := Runtime.Autograd.Cuda.Buffer.full (UInt32.ofNat (Shape.size (shape![len]))) 1.0 }
  let gradsCudaSlice ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.backwardDenseAll (t := t4sc) ySliceIdc seedCudaSlice)
  let dA_cuda ← Utils.cudaGrad (s := sA) gradsCudaSlice aIdc
  let dB_cuda ← Utils.cudaGrad (s := sBv) gradsCudaSlice bIdc

  Utils.assertTensorApprox (s := shape![len]) "concat+slice forward" yCudaSlice yCpuSlice (tol := 2e-3)
  Utils.assertTensorApprox (s := sA) "concat+slice dA" dA_cuda dA_cpu (tol := 2e-3)
  Utils.assertTensorApprox (s := sBv) "concat+slice dB" dB_cuda dB_cpu (tol := 2e-3)

  -- gather_scalar
  IO.println "== gather_scalar =="
  let nG : Nat := 5
  let sG : Shape := shape![nG]
  let xG : Tensor Float sG := tensorND! [nG] [0.10, -0.20, 0.30, 0.05, -0.15]
  let iG : Fin nG := ⟨3, by decide⟩

  -- CPU
  let t0g : Tape Float := Tape.empty
  let (t1g, xGid) := Tape.leaf (t := t0g) xG (name := some "x")
  let (t2g, yGid) ← Utils.okOrThrow (Tape.gatherScalar (α := Float) (t := t1g) (n := nG) xGid iG)
  let yCpuG ← Utils.cpuValue (s := Shape.scalar) t2g yGid
  let seedCpuG : Runtime.AnyTensor Float := AnyTensor.mk (Tensor.scalar 1.0)
  let gradsCpuG ← Utils.okOrThrow (Tape.backwardDenseAll (α := Float) (t := t2g) yGid seedCpuG)
  let dxCpuG ← Utils.cpuGrad (s := sG) gradsCpuG xGid

  -- CUDA
  let t0gc : Runtime.Autograd.Cuda.Tape := Runtime.Autograd.Cuda.Tape.empty
  let (t1gc, xGidc) := Runtime.Autograd.Cuda.Tape.leaf (t := t0gc) (Utils.tensorToAnyBuffer xG)
    (name := some "x")
  let (t2gc, yGidc) ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.gatherScalar (t := t1gc) (n := nG) xGidc iG)
  let yCudaG ← Utils.cudaValue (s := Shape.scalar) t2gc yGidc
  let seedCudaG : Runtime.Autograd.Cuda.AnyBuffer :=
    { s := Shape.scalar, buf := Runtime.Autograd.Cuda.Buffer.full 1 1.0 }
  let gradsCudaG ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.backwardDenseAll (t := t2gc) yGidc seedCudaG)
  let dxCudaG ← Utils.cudaGrad (s := sG) gradsCudaG xGidc

  Utils.assertTensorApprox (s := Shape.scalar) "gather_scalar forward" yCudaG yCpuG (tol := 2e-3)
  Utils.assertTensorApprox (s := sG) "gather_scalar backward" dxCudaG dxCpuG (tol := 2e-3)

  -- gather_scalar_nat (in-range + out-of-range forward)
  IO.println "== gather_scalar_nat =="
  let iNatGood : Nat := 2
  let iNatBad : Nat := 10

  -- CPU (good index)
  let t0gn : Tape Float := Tape.empty
  let (t1gn, xGnid) := Tape.leaf (t := t0gn) xG (name := some "x")
  let (t2gn, yGnid) ← Utils.okOrThrow
    (Tape.gatherScalarNat (α := Float) (t := t1gn) (n := nG) xGnid iNatGood)
  let yCpuGN ← Utils.cpuValue (s := Shape.scalar) t2gn yGnid
  let seedCpuGN : Runtime.AnyTensor Float := AnyTensor.mk (Tensor.scalar 1.0)
  let gradsCpuGN ← Utils.okOrThrow (Tape.backwardDenseAll (α := Float) (t := t2gn) yGnid seedCpuGN)
  let dxCpuGN ← Utils.cpuGrad (s := sG) gradsCpuGN xGnid

  -- CPU (bad index, forward only)
  let (t3gn, yBadId) ← Utils.okOrThrow
    (Tape.gatherScalarNat (α := Float) (t := t2gn) (n := nG) xGnid iNatBad)
  let yCpuBad ← Utils.cpuValue (s := Shape.scalar) t3gn yBadId

  -- CUDA (good index)
  let t0gnc : Runtime.Autograd.Cuda.Tape := Runtime.Autograd.Cuda.Tape.empty
  let (t1gnc, xGnidc) := Runtime.Autograd.Cuda.Tape.leaf (t := t0gnc) (Utils.tensorToAnyBuffer xG)
    (name := some "x")
  let (t2gnc, yGnidc) ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.gatherScalarNat (t := t1gnc) (n := nG) xGnidc iNatGood)
  let yCudaGN ← Utils.cudaValue (s := Shape.scalar) t2gnc yGnidc
  let seedCudaGN : Runtime.Autograd.Cuda.AnyBuffer :=
    { s := Shape.scalar, buf := Runtime.Autograd.Cuda.Buffer.full 1 1.0 }
  let gradsCudaGN ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.backwardDenseAll (t := t2gnc) yGnidc seedCudaGN)
  let dxCudaGN ← Utils.cudaGrad (s := sG) gradsCudaGN xGnidc

  -- CUDA (bad index, forward only)
  let (t3gnc, yBadIdc) ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.gatherScalarNat (t := t2gnc) (n := nG) xGnidc iNatBad)
  let yCudaBad ← Utils.cudaValue (s := Shape.scalar) t3gnc yBadIdc

  Utils.assertTensorApprox (s := Shape.scalar) "gather_scalar_nat good forward" yCudaGN yCpuGN (tol := 2e-3)
  Utils.assertTensorApprox (s := sG) "gather_scalar_nat good backward" dxCudaGN dxCpuGN (tol := 2e-3)
  Utils.assertTensorApprox (s := Shape.scalar) "gather_scalar_nat bad forward" yCudaBad yCpuBad (tol := 2e-3)

  -- gather_row
  IO.println "== gather_row =="
  let sRow : Shape := shape![2]
  let sM : Shape := shape![3, 2]
  let xM : Tensor Float sM :=
    tensorND! [3, 2] [
      0.10, 0.20,
      -0.30, 0.40,
      0.50, -0.60
    ]
  let iRow : Fin 3 := ⟨1, by decide⟩

  -- CPU
  let t0r : Tape Float := Tape.empty
  let (t1r, xMid) := Tape.leaf (t := t0r) xM (name := some "x")
  let (t2r, yRowId) ← Utils.okOrThrow
    (Tape.gatherRow (α := Float) (t := t1r) (rows := 3) (cols := 2) xMid iRow)
  let yCpuRow ← Utils.cpuValue (s := sRow) t2r yRowId
  let seedCpuRow : Runtime.AnyTensor Float := AnyTensor.mk (fill (1.0 : Float) sRow)
  let gradsCpuRow ← Utils.okOrThrow
    (Tape.backwardDenseAll (α := Float) (t := t2r) yRowId seedCpuRow)
  let dxCpuRow ← Utils.cpuGrad (s := sM) gradsCpuRow xMid

  -- CUDA
  let t0rc : Runtime.Autograd.Cuda.Tape := Runtime.Autograd.Cuda.Tape.empty
  let (t1rc, xMidc) := Runtime.Autograd.Cuda.Tape.leaf (t := t0rc) (Utils.tensorToAnyBuffer xM)
    (name := some "x")
  let (t2rc, yRowIdc) ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.gatherRow (t := t1rc) (rows := 3) (cols := 2) xMidc iRow)
  let yCudaRow ← Utils.cudaValue (s := sRow) t2rc yRowIdc
  let seedCudaRow : Runtime.Autograd.Cuda.AnyBuffer :=
    { s := sRow
      , buf := Runtime.Autograd.Cuda.Buffer.full (UInt32.ofNat (Shape.size sRow)) 1.0 }
  let gradsCudaRow ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.backwardDenseAll (t := t2rc) yRowIdc seedCudaRow)
  let dxCudaRow ← Utils.cudaGrad (s := sM) gradsCudaRow xMidc

  Utils.assertTensorApprox (s := sRow) "gather_row forward" yCudaRow yCpuRow (tol := 2e-3)
  Utils.assertTensorApprox (s := sM) "gather_row backward" dxCudaRow dxCpuRow (tol := 2e-3)

end LinearMseConcatSliceGather
end Cuda
end Tests
