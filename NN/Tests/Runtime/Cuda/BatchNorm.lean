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
# CUDA Kernel Coverage: BatchNorm

Compares CPU eager tape and CUDA eager tape for arbitrary-rank spatial BatchNorm.
-/

@[expose] public section

namespace Tests
namespace Cuda
namespace BatchNorm

open Spec TorchLean
open TorchLean TorchLean.Tensor
open Runtime.Autograd

abbrev channels : Nat := 2
abbrev height : Nat := 2
abbrev width : Nat := 2

theorem channels_pos : channels > 0 := by decide
theorem height_pos : height > 0 := by decide
theorem width_pos : width > 0 := by decide

abbrev spatial : Shape := [height, width]

theorem input_wellFormed : (spatial.prependDim channels).wellFormed := by
  exact ⟨channels_pos, ⟨height_pos, ⟨width_pos, trivial⟩⟩⟩

def x : Tensor Float [channels, height, width] :=
  (Tensor.from #[
    -- channel 0
    1.0, 2.0,
    3.0, 4.0,
    -- channel 1
    -0.5, 0.5,
    1.5, -1.0
  ]).reshape [channels, height, width] (by dsimp; decide)

def gamma : Tensor Float [channels] :=
  (Tensor.from #[1.0, 0.5]).reshape [channels] (by dsimp; decide)

def beta : Tensor Float [channels] :=
  (Tensor.from #[0.0, 0.1]).reshape [channels] (by dsimp; decide)

def run : IO Unit := do
  IO.println "=== CUDA kernel coverage: batch_norm ==="

  let outShape : Shape := [channels, height, width]

  -- CPU tape
  let t0 : Tape Float := Tape.empty
  let (t1, xId) := Tape.leaf (t := t0) x (name := some "x")
  let (t2, gId) := Tape.leaf (t := t1) gamma (name := some "gamma")
  let (t3, bId) := Tape.leaf (t := t2) beta (name := some "beta")
  let (t4, yId) ← Utils.okOrThrow
    (Tape.batchNorm (α := Float) (t := t3) (channels := channels) (sSpatial := spatial)
      input_wellFormed xId gId bId)
  let yCpu ← Utils.cpuValue (s := outShape) t4 yId
  let seedCpu : Spec.SomeTensor Float :=
    Spec.SomeTensor.ofTensor (Tensor.full outShape (1.0 : Float))
  let gradsCpu ← Utils.okOrThrow (Tape.backwardDenseAll (α := Float) (t := t4) yId seedCpu)
  let dxCpu ← Utils.cpuGrad (s := outShape) gradsCpu xId
  let dGammaCpu ← Utils.cpuGrad (s := [channels]) gradsCpu gId
  let dBetaCpu ← Utils.cpuGrad (s := [channels]) gradsCpu bId

  -- CUDA tape
  let t0c : Runtime.Autograd.Cuda.Tape := Runtime.Autograd.Cuda.Tape.empty
  let (t1c, xIdc) := Runtime.Autograd.Cuda.Tape.leaf (t := t0c) (Utils.tensorToAnyBuffer x)
    (name := some "x")
  let (t2c, gIdc) := Runtime.Autograd.Cuda.Tape.leaf (t := t1c) (Utils.tensorToAnyBuffer gamma)
    (name := some "gamma")
  let (t3c, bIdc) := Runtime.Autograd.Cuda.Tape.leaf (t := t2c) (Utils.tensorToAnyBuffer beta)
    (name := some "beta")
  let (t4c, yIdc) ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.batchNorm (t := t3c) (channels := channels)
      (spatial := spatial) input_wellFormed xIdc gIdc bIdc)
  let yCuda ← Utils.cudaValue (s := outShape) t4c yIdc
  let seedCuda : Runtime.Autograd.Cuda.AnyBuffer :=
    { s := outShape,
      buf := Runtime.Autograd.Cuda.Buffer.full (UInt32.ofNat (Spec.Shape.size outShape)) 1.0 }
  let gradsCuda ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.backwardDenseAll (t := t4c) yIdc seedCuda)
  let dxCuda ← Utils.cudaGrad (s := outShape) gradsCuda xIdc
  let dGammaCuda ← Utils.cudaGrad (s := [channels]) gradsCuda gIdc
  let dBetaCuda ← Utils.cudaGrad (s := [channels]) gradsCuda bIdc

  Utils.assertTensorApprox (s := outShape) "batchnorm forward" yCuda yCpu (tol := 5e-3)
  Utils.assertTensorApprox (s := outShape) "batchnorm dx" dxCuda dxCpu (tol := 5e-3)
  Utils.assertTensorApprox (s := [channels]) "batchnorm dgamma" dGammaCuda dGammaCpu (tol := 5e-3)
  Utils.assertTensorApprox (s := [channels]) "batchnorm dbeta" dBetaCuda dBetaCpu (tol := 5e-3)

end BatchNorm
end Cuda
end Tests
