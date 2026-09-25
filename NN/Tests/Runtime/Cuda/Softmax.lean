/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Model.Session
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

open Spec TorchLean
open TorchLean TorchLean.Tensor
open Runtime.Autograd

/-- Exercise arbitrary-axis softmax through the public CUDA session, including its VJP. -/
def checkInteriorAxisSession : IO Unit := do
  let s : Shape := [2, 2, 2]
  let x : Tensor Float s :=
    (Tensor.from (#[0, 2, 1, 4, 3, 8, 7, 9] : Array Float)).reshape [2, 2, 2] (by dsimp; decide)
  let upstream : Tensor Float s :=
    (Tensor.from #[1.0, -2.0, 3.0, 4.0, -1.0, 2.0, 5.0, -3.0]).reshape [2, 2, 2] (by dsimp; decide)
  let sess ← Runtime.Autograd.Model.Session.new (α := Float)
    { execution := .eager, device := .cuda }
  let xRef ← Runtime.Autograd.Model.Session.input sess x
    (name := some "interior_axis_input") (requiresGrad := true)
  let yRef ← Runtime.Autograd.Model.Session.softmax sess 1 xRef
  let actual ← Runtime.Autograd.Model.Session.getValue sess yRef
  let gradient ← Runtime.Autograd.Model.Session.vjp sess yRef upstream xRef
  let expected := Activation.softmaxSpec (α := Float) 1 x
  let expectedGradient := Activation.softmaxBackwardSpec (α := Float) 1 x upstream
  Utils.assertTensorApprox (s := s) "softmax interior-axis forward" actual expected (tol := 2e-3)
  Utils.assertTensorApprox (s := s) "softmax interior-axis backward" gradient expectedGradient
    (tol := 2e-3)

def evalSoftmax (device : NN.Backend.Device) (x upstream : Tensor Float [2, 3]) :
    IO (Tensor Float [2, 3] × Tensor Float [2, 3]) := do
  let sess ← Runtime.Autograd.Model.Session.new (α := Float)
    { execution := .eager, device := device }
  let xRef ← Runtime.Autograd.Model.Session.input sess x
    (name := some "softmax_input") (requiresGrad := true)
  let yRef ← Runtime.Autograd.Model.Session.softmax sess 1 xRef
  let y ← Runtime.Autograd.Model.Session.getValue sess yRef
  let gradient ← Runtime.Autograd.Model.Session.vjp sess yRef upstream xRef
  pure (y, gradient)

def evalLogSoftmax (device : NN.Backend.Device)
    (x upstream : Tensor Float [2, 3]) :
    IO (Tensor Float [2, 3] × Tensor Float [2, 3]) := do
  let sess ← Runtime.Autograd.Model.Session.new (α := Float)
    { execution := .eager, device := device }
  let xRef ← Runtime.Autograd.Model.Session.input sess x
    (name := some "log_softmax_input") (requiresGrad := true)
  let yRef ← Runtime.Autograd.Model.Session.logSoftmax sess 1 xRef
  let y ← Runtime.Autograd.Model.Session.getValue sess yRef
  let gradient ← Runtime.Autograd.Model.Session.vjp sess yRef upstream xRef
  pure (y, gradient)

/-- Forward scratch is retired while the output remains usable by repeated backward passes. -/
def checkScratchLifetime (logarithmic : Bool) : IO Unit := do
  let label := if logarithmic then "log_softmax" else "softmax"
  let x : Tensor Float [2, 3] := [[0.1, -0.2, 0.3], [0.05, 0.25, -0.15]]
  let upstream : Tensor Float [2, 3] := [[1.0, -2.0, 0.5], [0.25, 3.0, -1.0]]
  let baseline ← Runtime.Autograd.Cuda.Buffer.allocatorStats
  let input ← Runtime.Autograd.Cuda.Buffer.ofFloatArrayIO
    (Runtime.Autograd.Cuda.Convert.flattenFloat x)
  let (tape, xId) := Runtime.Autograd.Cuda.Tape.empty.leaf { s := [2, 3], buf := input }
  let result ← IO.lazyPure fun _ =>
    if logarithmic then Runtime.Autograd.Cuda.Tape.logSoftmaxLast (s := [2, 3]) tape xId
    else Runtime.Autograd.Cuda.Tape.softmaxLast (s := [2, 3]) tape xId
  let (tape, yId) ← Utils.okOrThrow result
  let forward ← Runtime.Autograd.Cuda.Buffer.allocatorStats
  -- Only the six input and six output elements may remain live.
  unless forward.liveBytes == baseline.liveBytes + 48 do
    throw <| IO.userError
      s!"{label}: retained forward scratch ({forward.liveBytes} live bytes)"
  let expected :=
    if logarithmic then Activation.logSoftmaxBackwardSpec (α := Float) 1
      (Activation.logSoftmaxSpec (α := Float) 1 x) upstream
    else Activation.softmaxBackwardSpec (α := Float) 1 x upstream
  for pass in [0:3] do
    let seed ← Runtime.Autograd.Cuda.Buffer.ofFloatArrayIO
      (Runtime.Autograd.Cuda.Convert.flattenFloat upstream)
    let gradients ← Runtime.Autograd.Cuda.Tape.backwardSparse tape yId
      { s := [2, 3], buf := seed } (fun id => id == xId)
    let some gradient := gradients.get? xId
      | throw <| IO.userError s!"{label}: missing input gradient on pass {pass}"
    Utils.assertTensorApprox s!"{label} repeated backward {pass}"
      (← Utils.anyBufferToTensor (s := [2, 3]) gradient) expected (tol := 2e-3)
    Runtime.Autograd.Cuda.Tape.releaseSparseGrads gradients
    let after ← Runtime.Autograd.Cuda.Buffer.allocatorStats
    unless after.liveBytes == forward.liveBytes do
      throw <| IO.userError s!"{label}: backward retained temporary payloads"
  for node in tape.nodes do
    discard <| Runtime.Autograd.Cuda.Buffer.releaseIO node.value.buf
    for buffer in node.cleanup do
      discard <| Runtime.Autograd.Cuda.Buffer.releaseIO buffer
  let retired ← Runtime.Autograd.Cuda.Buffer.allocatorStats
  unless retired.liveBytes == baseline.liveBytes do
    throw <| IO.userError s!"{label}: tape retirement retained payloads"

def run : IO Unit := do
  IO.println "=== CUDA kernel coverage: softmax ==="

  let s : Shape := [2, 3]
  let x : Tensor Float s :=
    (Tensor.from #[
      0.10, -0.20, 0.30,
      0.05,  0.25, -0.15
    ]).reshape [2, 3] (by dsimp; decide)
  let upstream : Tensor Float s := [[1.0, -2.0, 0.5], [0.25, 3.0, -1.0]]
  let (yCpu, dxCpu) ← evalSoftmax .cpu x upstream
  let (yCuda, dxCuda) ← evalSoftmax .cuda x upstream

  -- Compare (float32 vs float64, so use a modest tolerance).
  Utils.assertTensorApprox (s := s) "softmax forward" yCuda yCpu (tol := 2e-3)
  Utils.assertTensorApprox (s := s) "softmax backward" dxCuda dxCpu (tol := 2e-3)

  let (yLogCpu, dxLogCpu) ← evalLogSoftmax .cpu x upstream
  let (yLogCuda, dxLogCuda) ← evalLogSoftmax .cuda x upstream

  Utils.assertTensorApprox (s := s) "log_softmax forward" yLogCuda yLogCpu (tol := 2e-3)
  Utils.assertTensorApprox (s := s) "log_softmax backward" dxLogCuda dxLogCpu (tol := 2e-3)

  checkScratchLifetime false
  checkScratchLifetime true

  if Runtime.Autograd.Cuda.Buffer.runtimeStatus = .nativeAvailable then
    checkInteriorAxisSession

end Softmax
end Cuda
end Tests
