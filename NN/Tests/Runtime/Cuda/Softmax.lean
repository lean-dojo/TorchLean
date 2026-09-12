/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Model.Session
public import NN.Tensor
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

def run : IO Unit := do
  IO.println "=== CUDA kernel coverage: softmax ==="

  let s : Shape := [2, 3]
  let x : Tensor Float s :=
    (Tensor.from #[
      0.10, -0.20, 0.30,
      0.05,  0.25, -0.15
    ]).reshape [2, 3] (by dsimp; decide)
  let upstream : Tensor Float s := Tensor.full s 1.0
  let (yCpu, dxCpu) ← evalSoftmax .cpu x upstream
  let (yCuda, dxCuda) ← evalSoftmax .cuda x upstream

  -- Compare (float32 vs float64, so use a modest tolerance).
  Utils.assertTensorApprox (s := s) "softmax forward" yCuda yCpu (tol := 2e-3)
  Utils.assertTensorApprox (s := s) "softmax backward" dxCuda dxCpu (tol := 2e-3)

  let (yLogCpu, dxLogCpu) ← evalLogSoftmax .cpu x upstream
  let (yLogCuda, dxLogCuda) ← evalLogSoftmax .cuda x upstream

  Utils.assertTensorApprox (s := s) "log_softmax forward" yLogCuda yLogCpu (tol := 2e-3)
  Utils.assertTensorApprox (s := s) "log_softmax backward" dxLogCuda dxLogCpu (tol := 2e-3)

  if Runtime.Autograd.Cuda.Buffer.runtimeStatus = .nativeAvailable then
    checkInteriorAxisSession

end Softmax
end Cuda
end Tests
