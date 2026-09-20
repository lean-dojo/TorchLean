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
# CUDA Kernel Coverage: Elementwise Ops

One small composite forward/backward test that exercises the full elementwise surface
(`add/sub/mul/scale/abs/sqrt/clamp/max/min/relu/sigmoid/tanh/gelu/softplus/exp/log/inv/safe_log`)
plus `sum`.
-/

@[expose] public section

namespace Tests
namespace Cuda
namespace Elementwise

open Spec TorchLean
open TorchLean TorchLean.Tensor
open Runtime.Autograd

def run : IO Unit := do
  IO.println "=== CUDA kernel coverage: elementwise ==="

  let s : Shape := [5]
  let a : Tensor Float s :=
    (Tensor.from #[0.10, -0.20, 0.30, -0.15, 0.05]).reshape [5] (by dsimp; decide)
  let b : Tensor Float s :=
    (Tensor.from #[0.20,  0.10, -0.25, 0.40, -0.05]).reshape [5] (by dsimp; decide)

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
  let seedCpu : Spec.SomeTensor Float := Spec.SomeTensor.ofTensor (Tensor.scalar 1.0)
  let gradsCpu ← Utils.okOrThrow (Tape.backwardDenseAll (α := Float) (t := t22) outId seedCpu)
  let dA_cpu ← Utils.cpuGrad (s := s) gradsCpu aId
  let dB_cpu ← Utils.cpuGrad (s := s) gradsCpu bId

  -- CUDA tape
  let t0c : Runtime.Autograd.Cuda.Tape := Runtime.Autograd.Cuda.Tape.empty
  let (t1c, aIdc) :=
    Runtime.Autograd.Cuda.Tape.leaf (t := t0c) (Utils.tensorToAnyBuffer a) (name := some "a")
  let (t2c, bIdc) :=
    Runtime.Autograd.Cuda.Tape.leaf (t := t1c) (Utils.tensorToAnyBuffer b) (name := some "b")
  let (t3c, u1c) ← Utils.okOrThrow (Runtime.Autograd.Cuda.Tape.add (t := t2c) (s := s) aIdc bIdc)
  let (t4c, u2c) ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.scale (t := t3c) (s := s) aIdc scaleC)
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
  let (t15c, u13c) ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.clamp (t := t14c) (s := s) u12c clampLo clampHi)
  let (t16c, u14c) ← Utils.okOrThrow (Runtime.Autograd.Cuda.Tape.sqrt (t := t15c) (s := s) u13c)
  let (t17c, u15c) ← Utils.okOrThrow (Runtime.Autograd.Cuda.Tape.inv (t := t16c) (s := s) u14c)
  let (t18c, u16c) ← Utils.okOrThrow (Runtime.Autograd.Cuda.Tape.log (t := t17c) (s := s) u14c)
  let (t19c, u17c) ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.safeLog (t := t18c) (s := s) u14c eps)
  let (t20c, u18c) ← Utils.okOrThrow (Runtime.Autograd.Cuda.Tape.add (t := t19c) (s := s) u15c u16c)
  let (t21c, u19c) ← Utils.okOrThrow (Runtime.Autograd.Cuda.Tape.add (t := t20c) (s := s) u18c u17c)
  let (t22c, outIdc) ← Utils.okOrThrow (Runtime.Autograd.Cuda.Tape.sum (t := t21c) (s := s) u19c)

  let outCuda ← Utils.cudaValue (s := Shape.scalar) t22c outIdc
  let seedCuda : Runtime.Autograd.Cuda.AnyBuffer :=
    { s := Shape.scalar, buf := Runtime.Autograd.Cuda.Buffer.full 1 1.0 }
  let gradsCuda ← Utils.okOrThrow
    (Runtime.Autograd.Cuda.Tape.backwardDenseAll (t := t22c) outIdc seedCuda)
  let dA_cuda ← Utils.cudaGrad (s := s) gradsCuda aIdc
  let dB_cuda ← Utils.cudaGrad (s := s) gradsCuda bIdc

  Utils.assertTensorApprox (s := Shape.scalar) "elementwise forward" outCuda outCpu (tol := 2e-3)
  Utils.assertTensorApprox (s := s) "elementwise backward dA" dA_cuda dA_cpu (tol := 2e-3)
  Utils.assertTensorApprox (s := s) "elementwise backward dB" dB_cuda dB_cpu (tol := 2e-3)

  -- The former exp(2x) quotient produced infinity divided by infinity at +100.
  let tailShape : Shape := [2]
  let tails : Tensor Float tailShape :=
    (Tensor.from #[100.0, -100.0]).reshape [2] (by dsimp; decide)
  let tailTape0 : Runtime.Autograd.Cuda.Tape := Runtime.Autograd.Cuda.Tape.empty
  let (tailTape1, tailsId) := Runtime.Autograd.Cuda.Tape.leaf
    (t := tailTape0) (Utils.tensorToAnyBuffer tails) (name := some "tanh tails")
  let (tailTape2, tanhId) ← Utils.okOrThrow <|
    Runtime.Autograd.Cuda.Tape.tanh (t := tailTape1) (s := tailShape) tailsId
  let gotTails ← Utils.cudaValue (s := tailShape) tailTape2 tanhId
  let expectedTails : Tensor Float tailShape :=
    (Tensor.from #[1.0, -1.0]).reshape [2] (by dsimp; decide)
  Utils.assertTensorApprox (s := tailShape) "tanh finite tails" gotTails expectedTails

  -- GELU is one semantic tape node and one pointwise kernel in each direction. Check both against
  -- the spec-backed CPU tape over the nonlinear center and saturated tails.
  let geluShape : Shape := [7]
  let geluInput : Tensor Float geluShape :=
    (Tensor.from #[-10.0, -3.0, -1.0, 0.0, 1.0, 3.0, 10.0]).reshape [7] (by dsimp; decide)
  let geluCpu0 : Tape Float := Tape.empty
  let (geluCpu1, geluCpuInputId) :=
    Tape.leaf (t := geluCpu0) geluInput (name := some "gelu input")
  let (geluCpu2, geluCpuOutputId) ← Utils.okOrThrow <|
    Tape.gelu (α := Float) (t := geluCpu1) (s := geluShape) geluCpuInputId
  let geluCpuOutput ← Utils.cpuValue (s := geluShape) geluCpu2 geluCpuOutputId
  let geluSeedCpu : Spec.SomeTensor Float :=
    Spec.SomeTensor.ofTensor (Tensor.full (α := Float) geluShape 1.0)
  let geluCpuGrads ← Utils.okOrThrow <|
    Tape.backwardDenseAll (α := Float) (t := geluCpu2) geluCpuOutputId geluSeedCpu
  let geluCpuGrad ← Utils.cpuGrad (s := geluShape) geluCpuGrads geluCpuInputId

  let geluCuda0 : Runtime.Autograd.Cuda.Tape := Runtime.Autograd.Cuda.Tape.empty
  let (geluCuda1, geluCudaInputId) :=
    Runtime.Autograd.Cuda.Tape.leaf
      (t := geluCuda0) (Utils.tensorToAnyBuffer geluInput) (name := some "gelu input")
  let (geluCuda2, geluCudaOutputId) ← Utils.okOrThrow <|
    Runtime.Autograd.Cuda.Tape.gelu (t := geluCuda1) (s := geluShape) geluCudaInputId
  let geluCudaOutput ← Utils.cudaValue (s := geluShape) geluCuda2 geluCudaOutputId
  let geluSeedCuda : Runtime.Autograd.Cuda.AnyBuffer :=
    { s := geluShape, buf := Runtime.Autograd.Cuda.Buffer.full 7 1.0 }
  let geluCudaGrads ← Utils.okOrThrow <|
    Runtime.Autograd.Cuda.Tape.backwardDenseAll
      (t := geluCuda2) geluCudaOutputId geluSeedCuda
  let geluCudaGrad ← Utils.cudaGrad (s := geluShape) geluCudaGrads geluCudaInputId

  Utils.assertTensorApprox
    (s := geluShape) "fused GELU forward" geluCudaOutput geluCpuOutput (tol := 3e-5)
  Utils.assertTensorApprox
    (s := geluShape) "fused GELU backward" geluCudaGrad geluCpuGrad (tol := 3e-5)

  -- The fused optimizer primitive must preserve the staged AdamW computation it replaces.
  let optimShape : Shape := [4]
  let params : Tensor Float optimShape :=
    (Tensor.from #[1.0, -2.0, 0.5, 4.0]).reshape [4] (by dsimp; decide)
  let gradient : Tensor Float optimShape :=
    (Tensor.from #[0.2, -0.1, 0.4, -0.3]).reshape [4] (by dsimp; decide)
  let firstMoment : Tensor Float optimShape :=
    (Tensor.from #[0.01, -0.02, 0.03, -0.04]).reshape [4] (by dsimp; decide)
  let secondMoment : Tensor Float optimShape :=
    (Tensor.from #[0.2, 0.1, 0.4, 0.3]).reshape [4] (by dsimp; decide)
  let paramsBuf := Utils.tensorToBuffer params
  let gradientBuf := Utils.tensorToBuffer gradient
  let firstMomentBuf := Utils.tensorToBuffer firstMoment
  let secondMomentBuf := Utils.tensorToBuffer secondMoment
  let beta1 : Float := 0.9
  let beta2 : Float := 0.999
  let oneMinusBeta1 : Float := 1.0 - beta1
  let oneMinusBeta2 : Float := 1.0 - beta2
  let firstMomentCorrection : Float := 1.0 / (1.0 - beta1)
  let secondMomentCorrection : Float := 1.0 / (1.0 - beta2)
  let epsilon : Float := 1e-8
  let learningRate : Float := 3e-4
  let weightDecay : Float := 0.1

  let mScaled := Runtime.Autograd.Cuda.Buffer.scale firstMomentBuf beta1
  let expectedM := Runtime.Autograd.Cuda.Buffer.axpy mScaled gradientBuf oneMinusBeta1
  let gradientSquared := Runtime.Autograd.Cuda.Buffer.mul gradientBuf gradientBuf
  let vScaled := Runtime.Autograd.Cuda.Buffer.scale secondMomentBuf beta2
  let expectedV := Runtime.Autograd.Cuda.Buffer.axpy vScaled gradientSquared oneMinusBeta2
  let mHat := Runtime.Autograd.Cuda.Buffer.scale expectedM firstMomentCorrection
  let vHat := Runtime.Autograd.Cuda.Buffer.scale expectedV secondMomentCorrection
  let sqrtVHat := Runtime.Autograd.Cuda.Buffer.sqrt vHat
  let epsilonBuf := Runtime.Autograd.Cuda.Buffer.full 4 epsilon
  let denominator := Runtime.Autograd.Cuda.Buffer.add sqrtVHat epsilonBuf
  let normalizedUpdate := Runtime.Autograd.Cuda.Buffer.div mHat denominator
  let decayedParams :=
    Runtime.Autograd.Cuda.Buffer.axpy paramsBuf paramsBuf (-(learningRate * weightDecay))
  let expectedParams :=
    Runtime.Autograd.Cuda.Buffer.axpy decayedParams normalizedUpdate (-learningRate)
  let (gotParams, gotM, gotV) := Runtime.Autograd.Cuda.Buffer.adamStep
    paramsBuf gradientBuf firstMomentBuf secondMomentBuf
    beta1 oneMinusBeta1 beta2 oneMinusBeta2
    firstMomentCorrection secondMomentCorrection epsilon
    (-(learningRate * weightDecay)) (-learningRate)

  Utils.assertTensorApprox (s := optimShape) "fused AdamW parameters"
    (← Utils.bufferToTensor gotParams) (← Utils.bufferToTensor expectedParams) (tol := 1e-6)
  Utils.assertTensorApprox (s := optimShape) "fused AdamW first moment"
    (← Utils.bufferToTensor gotM) (← Utils.bufferToTensor expectedM) (tol := 1e-6)
  Utils.assertTensorApprox (s := optimShape) "fused AdamW second moment"
    (← Utils.bufferToTensor gotV) (← Utils.bufferToTensor expectedV) (tol := 1e-6)

end Elementwise
end Cuda
end Tests
