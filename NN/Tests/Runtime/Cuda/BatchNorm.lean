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
# CUDA Kernel Coverage: BatchNorm (Channel-First)

Compares CPU eager tape vs CUDA eager tape for `batchnorm_channel_first` (forward + backward).
-/

@[expose] public section

namespace Tests
namespace Cuda
namespace BatchNorm

open Spec
open Tensor
open Runtime.Autograd

abbrev channels : Nat := 2
abbrev height : Nat := 2
abbrev width : Nat := 2

def hC : channels > 0 := by decide
def hH : height > 0 := by decide
def hW : width > 0 := by decide

def x : Tensor Float (shape![channels, height, width]) :=
  tensorND! [channels, height, width] [
    -- channel 0
    1.0, 2.0,
    3.0, 4.0,
    -- channel 1
    -0.5, 0.5,
    1.5, -1.0
  ]

def gamma : Tensor Float (shape![channels]) :=
  tensorND! [channels] [1.0, 0.5]

def beta : Tensor Float (shape![channels]) :=
  tensorND! [channels] [0.0, 0.1]

def run : IO Unit := do
  IO.println "=== CUDA kernel coverage: batchnorm_channel_first ==="

  let outShape : Shape := shape![channels, height, width]

  -- CPU tape
  let t0 : Tape Float := Tape.empty
  let (t1, xId) := Tape.leaf (t := t0) x (name := some "x")
  let (t2, gId) := Tape.leaf (t := t1) gamma (name := some "gamma")
  let (t3, bId) := Tape.leaf (t := t2) beta (name := some "beta")
  let (t4, yId) ← Utils.okOrThrow
    (Tape.batchnormChannelFirst (α := Float) (t := t3) (channels := channels) (height := height)
      (width := width) (h_c := hC) (h_h := hH) (h_w := hW) xId gId bId)
  let yCpu ← Utils.cpuValue (s := outShape) t4 yId
  let seedCpu : Runtime.AnyTensor Float := AnyTensor.mk (fill (1.0 : Float) outShape)
  let gradsCpu ← Utils.okOrThrow (Tape.backwardDenseAll (α := Float) (t := t4) yId seedCpu)
  let dxCpu ← Utils.cpuGrad (s := outShape) gradsCpu xId
  let dGammaCpu ← Utils.cpuGrad (s := shape![channels]) gradsCpu gId
  let dBetaCpu ← Utils.cpuGrad (s := shape![channels]) gradsCpu bId

  -- CUDA tape
  let t0c : Runtime.Autograd.Cuda.Tape := Runtime.Autograd.Cuda.Tape.empty
  let (t1c, xIdc) := Runtime.Autograd.Cuda.Tape.leaf (t := t0c) (Utils.tensorToAnyBuffer x)
    (name := some "x")
  let (t2c, gIdc) := Runtime.Autograd.Cuda.Tape.leaf (t := t1c) (Utils.tensorToAnyBuffer gamma)
    (name := some "gamma")
  let (t3c, bIdc) := Runtime.Autograd.Cuda.Tape.leaf (t := t2c) (Utils.tensorToAnyBuffer beta)
    (name := some "beta")
  let (t4c, yIdc) ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.batchnormChannelFirst (t := t3c) (channels := channels)
      (height := height) (width := width) (h_c := hC) (h_h := hH) (h_w := hW) xIdc gIdc bIdc)
  let yCuda ← Utils.cudaValue (s := outShape) t4c yIdc
  let seedCuda : Runtime.Autograd.Cuda.AnyBuffer :=
    { s := outShape, buf := Runtime.Autograd.Cuda.Buffer.full (UInt32.ofNat (Shape.size outShape)) 1.0 }
  let gradsCuda ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.backwardDenseAll (t := t4c) yIdc seedCuda)
  let dxCuda ← Utils.cudaGrad (s := outShape) gradsCuda xIdc
  let dGammaCuda ← Utils.cudaGrad (s := shape![channels]) gradsCuda gIdc
  let dBetaCuda ← Utils.cudaGrad (s := shape![channels]) gradsCuda bIdc

  Utils.assertTensorApprox (s := outShape) "batchnorm forward" yCuda yCpu (tol := 5e-3)
  Utils.assertTensorApprox (s := outShape) "batchnorm dx" dxCuda dxCpu (tol := 5e-3)
  Utils.assertTensorApprox (s := shape![channels]) "batchnorm dgamma" dGammaCuda dGammaCpu (tol := 5e-3)
  Utils.assertTensorApprox (s := shape![channels]) "batchnorm dbeta" dBetaCuda dBetaCpu (tol := 5e-3)

end BatchNorm
end Cuda
end Tests
