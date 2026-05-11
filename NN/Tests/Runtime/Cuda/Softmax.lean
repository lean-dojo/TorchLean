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
# CUDA Kernel Coverage: Softmax

Compares CPU eager tape vs CUDA eager tape on small softmax/log-softmax examples
(forward + backward).
-/

@[expose] public section

namespace Tests
namespace Cuda
namespace Softmax

open Spec
open Tensor
open Runtime.Autograd

def run : IO Unit := do
  IO.println "=== CUDA kernel coverage: softmax ==="

  let s : Shape := shape![2, 3]
  let x : Tensor Float s :=
    tensorND! [2, 3] [
      0.10, -0.20, 0.30,
      0.05,  0.25, -0.15
    ]

  -- CPU tape: y = softmax(x)
  let t0 : Tape Float := Tape.empty
  let (t1, xId) := Tape.leaf (t := t0) x (name := some "x")
  let (t2, yId) ← Utils.okOrThrow (Tape.softmax (α := Float) (t := t1) (s := s) xId)
  let yCpu ← Utils.cpuValue (s := s) t2 yId
  let seedCpu : Runtime.AnyTensor Float := AnyTensor.mk (fill (1.0 : Float) s)
  let gradsCpu ← Utils.okOrThrow (Tape.backwardDenseAll (α := Float) (t := t2) yId seedCpu)
  let dxCpu ← Utils.cpuGrad (s := s) gradsCpu xId

  -- CUDA tape: y = softmax(x)
  let t0c : Runtime.Autograd.Cuda.Tape := Runtime.Autograd.Cuda.Tape.empty
  let (t1c, xIdc) := Runtime.Autograd.Cuda.Tape.leaf (t := t0c) (Utils.tensorToAnyBuffer x)
    (name := some "x")
  let (t2c, yIdc) ← Utils.okOrThrow (Runtime.Autograd.Cuda.Tape.softmax (t := t1c) (s := s) xIdc)
  let yCuda ← Utils.cudaValue (s := s) t2c yIdc
  let seedCuda : Runtime.Autograd.Cuda.AnyBuffer :=
    { s := s, buf := Runtime.Autograd.Cuda.Buffer.full (UInt32.ofNat (Shape.size s)) 1.0 }
  let gradsCuda ← Utils.okOrThrow (Runtime.Autograd.Cuda.Tape.backwardDenseAll (t := t2c) yIdc seedCuda)
  let dxCuda ← Utils.cudaGrad (s := s) gradsCuda xIdc

  -- Compare (float32 vs float64, so use a modest tolerance).
  Utils.assertTensorApprox (s := s) "softmax forward" yCuda yCpu (tol := 2e-3)
  Utils.assertTensorApprox (s := s) "softmax backward" dxCuda dxCpu (tol := 2e-3)

  -- CPU tape: y = stable log_softmax(x)
  let t0Log : Tape Float := Tape.empty
  let (t1Log, xIdLog) := Tape.leaf (t := t0Log) x (name := some "x_log")
  let (t2Log, yIdLog) ← Utils.okOrThrow (Tape.logSoftmax (α := Float) (t := t1Log) (s := s) xIdLog)
  let yLogCpu ← Utils.cpuValue (s := s) t2Log yIdLog
  let seedLogCpu : Runtime.AnyTensor Float := AnyTensor.mk (fill (1.0 : Float) s)
  let gradsLogCpu ← Utils.okOrThrow (Tape.backwardDenseAll (α := Float) (t := t2Log) yIdLog seedLogCpu)
  let dxLogCpu ← Utils.cpuGrad (s := s) gradsLogCpu xIdLog

  -- CUDA tape: y = stable log_softmax(x)
  let t0cLog : Runtime.Autograd.Cuda.Tape := Runtime.Autograd.Cuda.Tape.empty
  let (t1cLog, xIdcLog) := Runtime.Autograd.Cuda.Tape.leaf (t := t0cLog) (Utils.tensorToAnyBuffer x)
    (name := some "x_log")
  let (t2cLog, yIdcLog) ← Utils.okOrThrow (Runtime.Autograd.Cuda.Tape.logSoftmax (t := t1cLog)
    (s := s) xIdcLog)
  let yLogCuda ← Utils.cudaValue (s := s) t2cLog yIdcLog
  let seedLogCuda : Runtime.Autograd.Cuda.AnyBuffer :=
    { s := s, buf := Runtime.Autograd.Cuda.Buffer.full (UInt32.ofNat (Shape.size s)) 1.0 }
  let gradsLogCuda ← Utils.okOrThrow (Runtime.Autograd.Cuda.Tape.backwardDenseAll (t := t2cLog)
    yIdcLog seedLogCuda)
  let dxLogCuda ← Utils.cudaGrad (s := s) gradsLogCuda xIdcLog

  Utils.assertTensorApprox (s := s) "log_softmax forward" yLogCuda yLogCpu (tol := 2e-3)
  Utils.assertTensorApprox (s := s) "log_softmax backward" dxLogCuda dxLogCpu (tol := 2e-3)

end Softmax
end Cuda
end Tests
