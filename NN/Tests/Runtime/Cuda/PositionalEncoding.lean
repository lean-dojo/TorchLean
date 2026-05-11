/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Engine.Core
public import NN.Runtime.Autograd.Engine.Cuda.Ops
public import NN.Spec.Layers.PositionalEncoding
public import NN.Entrypoint.Tensor
public import NN.Tests.Runtime.Cuda.Utils

/-!
# CUDA Kernel Coverage: Positional Encodings

This test exercises:
- sinusoidal positional encoding (constant table + broadcast add),
- RoPE / rotary embeddings (pairwise rotate + broadcasted cos/sin tables).

We compare CPU eager tape vs CUDA eager tape on small shapes.
-/

@[expose] public section

namespace Tests
namespace Cuda
namespace PositionalEncoding

open Spec
open Tensor
open Runtime.Autograd

def run : IO Unit := do
  IO.println "=== CUDA kernel coverage: positional encoding ==="

  -- Sinusoidal PE: `(batch, seqLen, embedDim)`
  IO.println "== sinusoidalPositionalEncoding =="
  let batch : Nat := 2
  let seqLen : Nat := 3
  let embedDim : Nat := 5
  let startPos : Nat := 7
  let sX : Shape := shape![batch, seqLen, embedDim]
  let sPE : Shape := shape![seqLen, embedDim]
  let pe : Tensor Float sPE :=
    Spec.sinusoidalPositionalEncodingSpec (α := Float) seqLen embedDim startPos

  let x : Tensor Float sX :=
    tensorND! [2, 3, 5] [
      0.10, -0.20, 0.30, -0.40, 0.05,
      0.15,  0.25, -0.35, 0.45, -0.55,
      0.60, -0.10, 0.05,  0.20, -0.30,

      -0.05, 0.12, -0.18, 0.24, -0.30,
      0.33, -0.44, 0.55, -0.66, 0.77,
      -0.90, 0.80, -0.70, 0.60, -0.50
    ]

  let cbPE : Shape.CanBroadcastTo sPE sX := (inferInstance : Shape.BroadcastTo sPE sX).proof

  -- CPU tape
  let t0 : Tape Float := Tape.empty
  let (t1, xId) := Tape.leaf (t := t0) x (name := some "x")
  let (t2, peId) := Tape.leaf (t := t1) pe (name := some "pe") (requires_grad := false)
  let (t3, peBId) ← Utils.okOrThrow (Tape.broadcastTo (α := Float) (t := t2) (s₁ := sPE) (s₂ := sX) cbPE peId)
  let (t4, yId) ← Utils.okOrThrow (Tape.add (α := Float) (t := t3) (s := sX) xId peBId)
  let (t5, outId) ← Utils.okOrThrow (Tape.sum (α := Float) (t := t4) (s := sX) yId)
  let outCpu ← Utils.cpuValue (s := Shape.scalar) t5 outId
  let seedCpu : Runtime.AnyTensor Float := AnyTensor.mk (Tensor.scalar 1.0)
  let gradsCpu ← Utils.okOrThrow (Tape.backwardDenseAll (α := Float) (t := t5) outId seedCpu)
  let dxCpu ← Utils.cpuGrad (s := sX) gradsCpu xId

  -- CUDA tape
  let t0c : Runtime.Autograd.Cuda.Tape := Runtime.Autograd.Cuda.Tape.empty
  let (t1c, xIdc) := Runtime.Autograd.Cuda.Tape.leaf (t := t0c) (Utils.tensorToAnyBuffer x) (name := some "x")
  let (t2c, peIdc) := Runtime.Autograd.Cuda.Tape.leaf (t := t1c) (Utils.tensorToAnyBuffer pe) (name := some "pe") (requires_grad := false)
  let (t3c, peBIdc) ← Utils.okOrThrow (Runtime.Autograd.Cuda.Tape.broadcastTo (t := t2c) (s₁ := sPE) (s₂ := sX) cbPE peIdc)
  let (t4c, yIdc) ← Utils.okOrThrow (Runtime.Autograd.Cuda.Tape.add (t := t3c) (s := sX) xIdc peBIdc)
  let (t5c, outIdc) ← Utils.okOrThrow (Runtime.Autograd.Cuda.Tape.sum (t := t4c) (s := sX) yIdc)
  let outCuda ← Utils.cudaValue (s := Shape.scalar) t5c outIdc
  let seedCuda : Runtime.Autograd.Cuda.AnyBuffer :=
    { s := Shape.scalar, buf := Runtime.Autograd.Cuda.Buffer.full 1 1.0 }
  let gradsCuda ← Utils.okOrThrow (Runtime.Autograd.Cuda.Tape.backwardDenseAll (t := t5c) outIdc seedCuda)
  let dxCuda ← Utils.cudaGrad (s := sX) gradsCuda xIdc

  Utils.assertTensorApprox (s := Shape.scalar) "sinusoidal forward" outCuda outCpu (tol := 2e-3)
  Utils.assertTensorApprox (s := sX) "sinusoidal backward dx" dxCuda dxCpu (tol := 2e-3)

  -- RoPE: `(batch, numHeads, seqLen, headDim)`
  IO.println "== rope =="
  let numHeads : Nat := 2
  let headDim : Nat := 4
  let startPosR : Nat := 5
  let sR : Shape := shape![batch, numHeads, seqLen, headDim]
  let sCS : Shape := shape![seqLen, headDim]

  let cosT : Tensor Float sCS :=
    Spec.Tensor.dim (fun (pos : Fin seqLen) =>
      Spec.ropeCosLastdimSpec (α := Float) (startPosR + pos.val) headDim)
  let sinT : Tensor Float sCS :=
    Spec.Tensor.dim (fun (pos : Fin seqLen) =>
      Spec.ropeSinLastdimSpec (α := Float) (startPosR + pos.val) headDim)

  let permIdx : Tensor Nat (shape![headDim]) :=
    Spec.Tensor.dim (fun (j : Fin headDim) =>
      let idx := j.val
      let out : Nat :=
        if idx % 2 = 0 then
          if idx + 1 < headDim then idx + 1 else idx
        else
          idx - 1
      Spec.Tensor.scalar out)

  -- Sign vector as a 1×headDim row (so it broadcasts cleanly to (rowsFold×headDim)).
  let signRow : Tensor Float (shape![1, headDim]) :=
    Spec.Tensor.dim (fun (_ : Fin 1) =>
      Spec.Tensor.dim (fun (j : Fin headDim) =>
        let idx := j.val
        let v : Float := if idx % 2 = 0 ∧ idx + 1 < headDim then (-1.0) else 1.0
        Spec.Tensor.scalar v))

  let xR : Tensor Float sR :=
    tensorND! [2, 2, 3, 4] [
      -- batch0, head0
      0.10, 0.20, 0.30, 0.40,
      -0.15, 0.25, -0.35, 0.45,
      0.05, -0.10, 0.15, -0.20,
      -- batch0, head1
      -0.30, 0.60, -0.90, 1.20,
      0.12, -0.24, 0.36, -0.48,
      0.50, 0.40, 0.30, 0.20,
      -- batch1, head0
      -0.05, 0.10, -0.15, 0.20,
      0.25, -0.30, 0.35, -0.40,
      0.45, 0.50, -0.55, -0.60,
      -- batch1, head1
      0.70, -0.80, 0.90, -1.00,
      1.10, 1.20, -1.30, -1.40,
      -1.50, 1.60, -1.70, 1.80
    ]

  let rowsFold : Nat := batch * numHeads * seqLen
  let sFlat : Shape := shape![rowsFold, headDim]
  let sCS4 : Shape := shape![1, 1, seqLen, headDim]
  let cbCS4 : Shape.CanBroadcastTo sCS4 sR := (inferInstance : Shape.BroadcastTo sCS4 sR).proof
  let cbSign : Shape.CanBroadcastTo (shape![1, headDim]) sFlat :=
    (inferInstance : Shape.BroadcastTo (shape![1, headDim]) sFlat).proof

  -- CPU tape
  let t0r : Tape Float := Tape.empty
  let (t1r, xIdr) := Tape.leaf (t := t0r) xR (name := some "x")
  let (t2r, cosIdr) := Tape.leaf (t := t1r) cosT (name := some "cos") (requires_grad := false)
  let (t3r, sinIdr) := Tape.leaf (t := t2r) sinT (name := some "sin") (requires_grad := false)
  let (t4r, signIdr) := Tape.leaf (t := t3r) signRow (name := some "sign") (requires_grad := false)

  let (t5r, cos4Idr) ← Utils.okOrThrow (Tape.reshape (α := Float) (t := t4r) (s₁ := sCS) (s₂ := sCS4) cosIdr (by simp [sCS, sCS4, Shape.size]))
  let (t6r, sin4Idr) ← Utils.okOrThrow (Tape.reshape (α := Float) (t := t5r) (s₁ := sCS) (s₂ := sCS4) sinIdr (by simp [sCS, sCS4, Shape.size]))
  let (t7r, cosBIdr) ← Utils.okOrThrow (Tape.broadcastTo (α := Float) (t := t6r) (s₁ := sCS4) (s₂ := sR) cbCS4 cos4Idr)
  let (t8r, sinBIdr) ← Utils.okOrThrow (Tape.broadcastTo (α := Float) (t := t7r) (s₁ := sCS4) (s₂ := sR) cbCS4 sin4Idr)
  let (t9r, xCosIdr) ← Utils.okOrThrow (Tape.mul (α := Float) (t := t8r) (s := sR) xIdr cosBIdr)

  -- rotatePairs(x): reshape -> transpose -> gather_rows_nat -> transpose -> mul(sign) -> reshape
  let (t10r, x2dIdr) ← Utils.okOrThrow (Tape.reshape (α := Float) (t := t9r) (s₁ := sR) (s₂ := sFlat) xIdr
    (by simp [sR, sFlat, rowsFold, Shape.size, Nat.mul_assoc]))
  let (t11r, xTIdr) ← Utils.okOrThrow (Tape.transpose2d (α := Float) (t := t10r) (m := rowsFold) (n := headDim) x2dIdr)
  let (t12r, xPermIdr) ← Utils.okOrThrow (Tape.gatherRowsNat (α := Float) (t := t11r) (rows := headDim) (cols := rowsFold) (k := headDim) xTIdr permIdx)
  let (t13r, xBackIdr) ← Utils.okOrThrow (Tape.transpose2d (α := Float) (t := t12r) (m := headDim) (n := rowsFold) xPermIdr)
  let (t14r, signBIdr) ← Utils.okOrThrow (Tape.broadcastTo (α := Float) (t := t13r) (s₁ := shape![1, headDim]) (s₂ := sFlat) cbSign signIdr)
  let (t15r, xRot2dIdr) ← Utils.okOrThrow (Tape.mul (α := Float) (t := t14r) (s := sFlat) xBackIdr signBIdr)
  let (t16r, xRotIdr) ← Utils.okOrThrow (Tape.reshape (α := Float) (t := t15r) (s₁ := sFlat) (s₂ := sR) xRot2dIdr
    (by simp [sR, sFlat, rowsFold, Shape.size, Nat.mul_assoc]))

  let (t17r, rotSinIdr) ← Utils.okOrThrow (Tape.mul (α := Float) (t := t16r) (s := sR) xRotIdr sinBIdr)
  let (t18r, yRIdr) ← Utils.okOrThrow (Tape.add (α := Float) (t := t17r) (s := sR) xCosIdr rotSinIdr)
  let (t19r, outRIdr) ← Utils.okOrThrow (Tape.sum (α := Float) (t := t18r) (s := sR) yRIdr)

  let outRCpu ← Utils.cpuValue (s := Shape.scalar) t19r outRIdr
  let seedRCpu : Runtime.AnyTensor Float := AnyTensor.mk (Tensor.scalar 1.0)
  let gradsRCpu ← Utils.okOrThrow (Tape.backwardDenseAll (α := Float) (t := t19r) outRIdr seedRCpu)
  let dxRCpu ← Utils.cpuGrad (s := sR) gradsRCpu xIdr

  -- CUDA tape
  let t0rc : Runtime.Autograd.Cuda.Tape := Runtime.Autograd.Cuda.Tape.empty
  let (t1rc, xIdrc) := Runtime.Autograd.Cuda.Tape.leaf (t := t0rc) (Utils.tensorToAnyBuffer xR) (name := some "x")
  let (t2rc, cosIdrc) := Runtime.Autograd.Cuda.Tape.leaf (t := t1rc) (Utils.tensorToAnyBuffer cosT) (name := some "cos") (requires_grad := false)
  let (t3rc, sinIdrc) := Runtime.Autograd.Cuda.Tape.leaf (t := t2rc) (Utils.tensorToAnyBuffer sinT) (name := some "sin") (requires_grad := false)
  let (t4rc, signIdrc) := Runtime.Autograd.Cuda.Tape.leaf (t := t3rc) (Utils.tensorToAnyBuffer signRow) (name := some "sign") (requires_grad := false)

  let (t5rc, cos4Idrc) ← Utils.okOrThrow (Runtime.Autograd.Cuda.Tape.reshape (t := t4rc) (s₁ := sCS) (s₂ := sCS4) cosIdrc (by simp [sCS, sCS4, Shape.size]))
  let (t6rc, sin4Idrc) ← Utils.okOrThrow (Runtime.Autograd.Cuda.Tape.reshape (t := t5rc) (s₁ := sCS) (s₂ := sCS4) sinIdrc (by simp [sCS, sCS4, Shape.size]))
  let (t7rc, cosBIdrc) ← Utils.okOrThrow (Runtime.Autograd.Cuda.Tape.broadcastTo (t := t6rc) (s₁ := sCS4) (s₂ := sR) cbCS4 cos4Idrc)
  let (t8rc, sinBIdrc) ← Utils.okOrThrow (Runtime.Autograd.Cuda.Tape.broadcastTo (t := t7rc) (s₁ := sCS4) (s₂ := sR) cbCS4 sin4Idrc)
  let (t9rc, xCosIdrc) ← Utils.okOrThrow (Runtime.Autograd.Cuda.Tape.mul (t := t8rc) (s := sR) xIdrc cosBIdrc)

  let (t10rc, x2dIdrc) ← Utils.okOrThrow (Runtime.Autograd.Cuda.Tape.reshape (t := t9rc) (s₁ := sR) (s₂ := sFlat) xIdrc
    (by simp [sR, sFlat, rowsFold, Shape.size, Nat.mul_assoc]))
  let (t11rc, xTIdrc) ← Utils.okOrThrow (Runtime.Autograd.Cuda.Tape.transpose2d (t := t10rc) (m := rowsFold) (n := headDim) x2dIdrc)
  let (t12rc, xPermIdrc) ← Utils.okOrThrow (Runtime.Autograd.Cuda.Tape.gatherRowsNat (t := t11rc) (rows := headDim) (cols := rowsFold) (k := headDim) xTIdrc permIdx)
  let (t13rc, xBackIdrc) ← Utils.okOrThrow (Runtime.Autograd.Cuda.Tape.transpose2d (t := t12rc) (m := headDim) (n := rowsFold) xPermIdrc)
  let (t14rc, signBIdrc) ← Utils.okOrThrow (Runtime.Autograd.Cuda.Tape.broadcastTo (t := t13rc) (s₁ := shape![1, headDim]) (s₂ := sFlat) cbSign signIdrc)
  let (t15rc, xRot2dIdrc) ← Utils.okOrThrow (Runtime.Autograd.Cuda.Tape.mul (t := t14rc) (s := sFlat) xBackIdrc signBIdrc)
  let (t16rc, xRotIdrc) ← Utils.okOrThrow (Runtime.Autograd.Cuda.Tape.reshape (t := t15rc) (s₁ := sFlat) (s₂ := sR) xRot2dIdrc
    (by simp [sR, sFlat, rowsFold, Shape.size, Nat.mul_assoc]))

  let (t17rc, rotSinIdrc) ← Utils.okOrThrow (Runtime.Autograd.Cuda.Tape.mul (t := t16rc) (s := sR) xRotIdrc sinBIdrc)
  let (t18rc, yRIdrc) ← Utils.okOrThrow (Runtime.Autograd.Cuda.Tape.add (t := t17rc) (s := sR) xCosIdrc rotSinIdrc)
  let (t19rc, outRIdrc) ← Utils.okOrThrow (Runtime.Autograd.Cuda.Tape.sum (t := t18rc) (s := sR) yRIdrc)

  let outRCuda ← Utils.cudaValue (s := Shape.scalar) t19rc outRIdrc
  let seedRCuda : Runtime.Autograd.Cuda.AnyBuffer :=
    { s := Shape.scalar, buf := Runtime.Autograd.Cuda.Buffer.full 1 1.0 }
  let gradsRCuda ← Utils.okOrThrow (Runtime.Autograd.Cuda.Tape.backwardDenseAll (t := t19rc) outRIdrc seedRCuda)
  let dxRCuda ← Utils.cudaGrad (s := sR) gradsRCuda xIdrc

  Utils.assertTensorApprox (s := Shape.scalar) "rope forward" outRCuda outRCpu (tol := 3e-3)
  Utils.assertTensorApprox (s := sR) "rope backward dx" dxRCuda dxRCpu (tol := 3e-3)

end PositionalEncoding
end Cuda
end Tests
