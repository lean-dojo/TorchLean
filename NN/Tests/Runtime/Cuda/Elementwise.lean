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
# CUDA Kernel Coverage: Elementwise Ops

One small composite forward/backward test that exercises the full elementwise surface
(`add/sub/mul/scale/abs/sqrt/clamp/max/min/relu/sigmoid/tanh/softplus/exp/log/inv/safe_log`)
plus `sum`.
-/

@[expose] public section

namespace Tests
namespace Cuda
namespace Elementwise

open Spec
open Tensor
open Runtime.Autograd

def run : IO Unit := do
  IO.println "=== CUDA kernel coverage: elementwise ==="

  let s : Shape := shape![5]
  let a : Tensor Float s := tensorND! [5] [0.10, -0.20, 0.30, -0.15, 0.05]
  let b : Tensor Float s := tensorND! [5] [0.20,  0.10, -0.25, 0.40, -0.05]

  let scaleC : Float := 0.3
  let clampLo : Float := 1e-3
  let clampHi : Float := 10.0
  let eps : Float := 1e-6

  -- CPU tape
  let t0 : Tape Float := Tape.empty
  let (t1, aId) := Tape.leaf (t := t0) a (name := some "a")
  let (t2, bId) := Tape.leaf (t := t1) b (name := some "b")
  let (t3, u1) ← Utils.okOrThrow (Tape.add (α := Float) (t := t2) (s := s) aId bId)
  let (t4, u2) ← Utils.okOrThrow (Tape.scale (α := Float) (t := t3) (s := s) aId scaleC)
  let (t5, u3) ← Utils.okOrThrow (Tape.sub (α := Float) (t := t4) (s := s) u1 u2)
  let (t6, u4) ← Utils.okOrThrow (Tape.mul (α := Float) (t := t5) (s := s) u3 bId)
  let (t7, u5) ← Utils.okOrThrow (Tape.max (α := Float) (t := t6) (s := s) u4 aId)
  let (t8, u6) ← Utils.okOrThrow (Tape.min (α := Float) (t := t7) (s := s) u5 bId)
  let (t9, u7) ← Utils.okOrThrow (Tape.relu (α := Float) (t := t8) (s := s) u6)
  let (t10, u8) ← Utils.okOrThrow (Tape.sigmoid (α := Float) (t := t9) (s := s) u7)
  let (t11, u9) ← Utils.okOrThrow (Tape.tanh (α := Float) (t := t10) (s := s) u8)
  let (t12, u10) ← Utils.okOrThrow (Tape.softplus (α := Float) (t := t11) (s := s) u9)
  let (t13, u11) ← Utils.okOrThrow (Tape.exp (α := Float) (t := t12) (s := s) u10)
  let (t14, u12) ← Utils.okOrThrow (Tape.abs (α := Float) (t := t13) (s := s) u11)
  let (t15, u13) ← Utils.okOrThrow (Tape.clamp (α := Float) (t := t14) (s := s) u12 clampLo clampHi)
  let (t16, u14) ← Utils.okOrThrow (Tape.sqrt (α := Float) (t := t15) (s := s) u13)
  let (t17, u15) ← Utils.okOrThrow (Tape.inv (α := Float) (t := t16) (s := s) u14)
  let (t18, u16) ← Utils.okOrThrow (Tape.log (α := Float) (t := t17) (s := s) u14)
  let (t19, u17) ← Utils.okOrThrow (Tape.safeLog (α := Float) (t := t18) (s := s) u14 (ε := eps))
  let (t20, u18) ← Utils.okOrThrow (Tape.add (α := Float) (t := t19) (s := s) u15 u16)
  let (t21, u19) ← Utils.okOrThrow (Tape.add (α := Float) (t := t20) (s := s) u18 u17)
  let (t22, outId) ← Utils.okOrThrow (Tape.sum (α := Float) (t := t21) (s := s) u19)

  let outCpu ← Utils.cpuValue (s := Shape.scalar) t22 outId
  let seedCpu : Runtime.AnyTensor Float := AnyTensor.mk (Tensor.scalar 1.0)
  let gradsCpu ← Utils.okOrThrow (Tape.backwardDenseAll (α := Float) (t := t22) outId seedCpu)
  let dA_cpu ← Utils.cpuGrad (s := s) gradsCpu aId
  let dB_cpu ← Utils.cpuGrad (s := s) gradsCpu bId

  -- CUDA tape
  let t0c : Runtime.Autograd.Cuda.Tape := Runtime.Autograd.Cuda.Tape.empty
  let (t1c, aIdc) := Runtime.Autograd.Cuda.Tape.leaf (t := t0c) (Utils.tensorToAnyBuffer a) (name := some "a")
  let (t2c, bIdc) := Runtime.Autograd.Cuda.Tape.leaf (t := t1c) (Utils.tensorToAnyBuffer b) (name := some "b")
  let (t3c, u1c) ← Utils.okOrThrow (Runtime.Autograd.Cuda.Tape.add (t := t2c) (s := s) aIdc bIdc)
  let (t4c, u2c) ← Utils.okOrThrow (Runtime.Autograd.Cuda.Tape.scale (t := t3c) (s := s) aIdc scaleC)
  let (t5c, u3c) ← Utils.okOrThrow (Runtime.Autograd.Cuda.Tape.sub (t := t4c) (s := s) u1c u2c)
  let (t6c, u4c) ← Utils.okOrThrow (Runtime.Autograd.Cuda.Tape.mul (t := t5c) (s := s) u3c bIdc)
  let (t7c, u5c) ← Utils.okOrThrow (Runtime.Autograd.Cuda.Tape.max (t := t6c) (s := s) u4c aIdc)
  let (t8c, u6c) ← Utils.okOrThrow (Runtime.Autograd.Cuda.Tape.min (t := t7c) (s := s) u5c bIdc)
  let (t9c, u7c) ← Utils.okOrThrow (Runtime.Autograd.Cuda.Tape.relu (t := t8c) (s := s) u6c)
  let (t10c, u8c) ← Utils.okOrThrow (Runtime.Autograd.Cuda.Tape.sigmoid (t := t9c) (s := s) u7c)
  let (t11c, u9c) ← Utils.okOrThrow (Runtime.Autograd.Cuda.Tape.tanh (t := t10c) (s := s) u8c)
  let (t12c, u10c) ← Utils.okOrThrow (Runtime.Autograd.Cuda.Tape.softplus (t := t11c) (s := s) u9c)
  let (t13c, u11c) ← Utils.okOrThrow (Runtime.Autograd.Cuda.Tape.exp (t := t12c) (s := s) u10c)
  let (t14c, u12c) ← Utils.okOrThrow (Runtime.Autograd.Cuda.Tape.abs (t := t13c) (s := s) u11c)
  let (t15c, u13c) ← Utils.okOrThrow (Runtime.Autograd.Cuda.Tape.clamp (t := t14c) (s := s) u12c clampLo clampHi)
  let (t16c, u14c) ← Utils.okOrThrow (Runtime.Autograd.Cuda.Tape.sqrt (t := t15c) (s := s) u13c)
  let (t17c, u15c) ← Utils.okOrThrow (Runtime.Autograd.Cuda.Tape.inv (t := t16c) (s := s) u14c)
  let (t18c, u16c) ← Utils.okOrThrow (Runtime.Autograd.Cuda.Tape.log (t := t17c) (s := s) u14c)
  let (t19c, u17c) ← Utils.okOrThrow (Runtime.Autograd.Cuda.Tape.safeLog (t := t18c) (s := s) u14c eps)
  let (t20c, u18c) ← Utils.okOrThrow (Runtime.Autograd.Cuda.Tape.add (t := t19c) (s := s) u15c u16c)
  let (t21c, u19c) ← Utils.okOrThrow (Runtime.Autograd.Cuda.Tape.add (t := t20c) (s := s) u18c u17c)
  let (t22c, outIdc) ← Utils.okOrThrow (Runtime.Autograd.Cuda.Tape.sum (t := t21c) (s := s) u19c)

  let outCuda ← Utils.cudaValue (s := Shape.scalar) t22c outIdc
  let seedCuda : Runtime.Autograd.Cuda.AnyBuffer :=
    { s := Shape.scalar, buf := Runtime.Autograd.Cuda.Buffer.full 1 1.0 }
  let gradsCuda ← Utils.okOrThrow (Runtime.Autograd.Cuda.Tape.backwardDenseAll (t := t22c) outIdc seedCuda)
  let dA_cuda ← Utils.cudaGrad (s := s) gradsCuda aIdc
  let dB_cuda ← Utils.cudaGrad (s := s) gradsCuda bIdc

  Utils.assertTensorApprox (s := Shape.scalar) "elementwise forward" outCuda outCpu (tol := 2e-3)
  Utils.assertTensorApprox (s := s) "elementwise backward dA" dA_cuda dA_cpu (tol := 2e-3)
  Utils.assertTensorApprox (s := s) "elementwise backward dB" dB_cuda dB_cpu (tol := 2e-3)

end Elementwise
end Cuda
end Tests
