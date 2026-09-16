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
# CUDA Kernel Coverage: LayerNorm

Compares CPU eager tape vs CUDA eager tape for `layer_norm` (forward + backward).
-/

@[expose] public section

namespace Tests
namespace Cuda
namespace LayerNorm

open Spec TorchLean
open TorchLean TorchLean.Tensor
open Runtime.Autograd

abbrev seqLen : Nat := 2
abbrev embedDim : Nat := 4

theorem seqLen_pos : seqLen > 0 := by decide
theorem embedDim_pos : embedDim > 0 := by decide

def x : Tensor Float [seqLen, embedDim] :=
  (Tensor.from #[
    0.10, 0.20, 0.00, -0.10,
    -0.30, 0.50, 0.20, 0.10
  ]).reshape [seqLen, embedDim] (by dsimp; decide)

def gamma : Tensor Float [embedDim] :=
  (Tensor.from #[1.0, 0.9, 1.1, 1.0]).reshape [embedDim] (by dsimp; decide)

def beta : Tensor Float [embedDim] :=
  (Tensor.from #[0.0, 0.1, -0.1, 0.0]).reshape [embedDim] (by dsimp; decide)

def run : IO Unit := do
  IO.println "=== CUDA kernel coverage: layer_norm ==="

  let outShape : Shape := [seqLen, embedDim]

  -- CPU tape
  let t0 : Tape Float := Tape.empty
  let (t1, xId) := Tape.leaf (t := t0) x (name := some "x")
  let (t2, gId) := Tape.leaf (t := t1) gamma (name := some "gamma")
  let (t3, bId) := Tape.leaf (t := t2) beta (name := some "beta")
  let (t4, yId) ← Utils.okOrThrow
    (Tape.layerNorm (α := Float) (t := t3) (seqLen := seqLen) (embedDim := embedDim)
      (h_seq_pos := seqLen_pos) (h_embed_pos := embedDim_pos) xId gId bId)
  let yCpu ← Utils.cpuValue (s := outShape) t4 yId
  let seedCpu : Spec.SomeTensor Float :=
    Spec.SomeTensor.ofTensor (Tensor.full outShape (1.0 : Float))
  let gradsCpu ← Utils.okOrThrow (Tape.backwardDenseAll (α := Float) (t := t4) yId seedCpu)
  let dxCpu ← Utils.cpuGrad (s := outShape) gradsCpu xId
  let dGammaCpu ← Utils.cpuGrad (s := [embedDim]) gradsCpu gId
  let dBetaCpu ← Utils.cpuGrad (s := [embedDim]) gradsCpu bId

  -- CUDA tape
  let t0c : Runtime.Autograd.Cuda.Tape := Runtime.Autograd.Cuda.Tape.empty
  let (t1c, xIdc) := Runtime.Autograd.Cuda.Tape.leaf (t := t0c) (Utils.tensorToAnyBuffer x)
    (name := some "x")
  let (t2c, gIdc) := Runtime.Autograd.Cuda.Tape.leaf (t := t1c) (Utils.tensorToAnyBuffer gamma)
    (name := some "gamma")
  let (t3c, bIdc) := Runtime.Autograd.Cuda.Tape.leaf (t := t2c) (Utils.tensorToAnyBuffer beta)
    (name := some "beta")
  let (t4c, yIdc) ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.layerNorm (t := t3c) (seqLen := seqLen) (embedDim := embedDim)
      (h_seq_pos := seqLen_pos) (h_embed_pos := embedDim_pos) xIdc gIdc bIdc)
  let yCuda ← Utils.cudaValue (s := outShape) t4c yIdc
  let seedCuda : Runtime.Autograd.Cuda.AnyBuffer :=
    { s := outShape,
      buf := Runtime.Autograd.Cuda.Buffer.full (UInt32.ofNat (Spec.Shape.size outShape)) 1.0 }
  let gradsCuda ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.backwardDenseAll (t := t4c) yIdc seedCuda)
  let dxCuda ← Utils.cudaGrad (s := outShape) gradsCuda xIdc
  let dGammaCuda ← Utils.cudaGrad (s := [embedDim]) gradsCuda gIdc
  let dBetaCuda ← Utils.cudaGrad (s := [embedDim]) gradsCuda bIdc

  Utils.assertTensorApprox (s := outShape) "layer_norm forward" yCuda yCpu (tol := 3e-3)
  Utils.assertTensorApprox (s := outShape) "layer_norm dx" dxCuda dxCpu (tol := 3e-3)
  Utils.assertTensorApprox (s := [embedDim]) "layer_norm dgamma" dGammaCuda dGammaCpu (tol := 3e-3)
  Utils.assertTensorApprox (s := [embedDim]) "layer_norm dbeta" dBetaCuda dBetaCpu (tol := 3e-3)

end LayerNorm
end Cuda
end Tests
