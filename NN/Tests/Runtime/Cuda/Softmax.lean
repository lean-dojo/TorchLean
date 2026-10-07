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

/-- A rejected node or cotangent must not allocate native payloads or buffer wrappers. -/
def Internal.assertRejectedWithoutAllocation {α : Type} (label : String)
    (action : Unit → Except String α) : IO Unit := do
  let before ← LibTorch.Buffer.allocatorStats
  let result ← IO.lazyPure action
  let after ← LibTorch.Buffer.allocatorStats
  match result with
  | .ok _ => throw <| IO.userError s!"{label}: unexpectedly accepted invalid input"
  | .error _ => pure ()
  unless after.liveBytes == before.liveBytes && after.allocCount == before.allocCount &&
      after.wrapperAllocCount == before.wrapperAllocCount do
    throw <| IO.userError s!"{label}: allocated before rejecting invalid input"

/-- Exercise the VJP closure itself, beyond the outer backward seed validation. -/
def Internal.checkWrongUpstream (label : String) (s : Shape) (node : LibTorch.Node) :
    IO Unit := do
  let seed ← LibTorch.Buffer.ofFloatArrayIO (Utils.floatArray (Array.replicate s.size 2.0))
  Internal.assertRejectedWithoutAllocation s!"{label} wrong upstream shape" fun _ =>
    node.backward { s := .dim 1 s, buf := seed }
  discard <| LibTorch.Buffer.releaseIO seed
  let wrongLength ← LibTorch.Buffer.ofFloatArrayIO
    (Utils.floatArray (Array.replicate (s.size + 1) 2.0))
  Internal.assertRejectedWithoutAllocation s!"{label} wrong upstream length" fun _ =>
    node.backward { s := s, buf := wrongLength }
  discard <| LibTorch.Buffer.releaseIO wrongLength

/-- Scalar and empty row operations preserve values, zero VJPs, and repeated-backward lifetime. -/
def Internal.checkDegenerateLast (s : Shape) (logarithmic : Bool) : IO Unit := do
  let label := if logarithmic then "degenerate log_softmax" else "degenerate softmax"
  let applyOp := fun (tape : LibTorch.Tape) (xId : Nat) =>
    if logarithmic then LibTorch.Tape.logSoftmaxLast (s := s) tape xId
    else LibTorch.Tape.softmaxLast (s := s) tape xId
  Internal.assertRejectedWithoutAllocation s!"{label} invalid node" fun _ =>
    applyOp LibTorch.Tape.empty 0
  let baseline ← LibTorch.Buffer.allocatorStats
  let input ← LibTorch.Buffer.ofFloatArrayIO (Utils.floatArray (Array.replicate s.size 7.0))
  let (tape, xId) := LibTorch.Tape.empty.leaf { s := s, buf := input }
  let result ← IO.lazyPure fun _ => applyOp tape xId
  let (tape, yId) ← IO.ofExcept result
  let some node := tape.getNode? yId
    | throw <| IO.userError s!"{label}: missing output node"
  unless node.value.s == s do
    throw <| IO.userError s!"{label}: output shape changed"
  let output ← LibTorch.Buffer.toFloatArrayIO node.value.buf
  let expectedValue := if logarithmic then 0.0 else 1.0
  unless output.size == s.size do
    throw <| IO.userError s!"{label}: output length changed"
  for i in [:output.size] do
    unless output.get! i == expectedValue do
      throw <| IO.userError s!"{label}: unexpected scalar output"
  Internal.checkWrongUpstream label s node
  let forward ← LibTorch.Buffer.allocatorStats
  unless forward.liveBytes == baseline.liveBytes + (8 * s.size).toUInt64 do
    throw <| IO.userError s!"{label}: retained forward payloads"
  for pass in [0:3] do
    let seed ← LibTorch.Buffer.ofFloatArrayIO
      (Utils.floatArray (Array.replicate s.size 3.0))
    let gradients ← LibTorch.Tape.backwardSparse tape yId
      { s := s, buf := seed } (fun id => id == xId)
    let some gradient := gradients.get? xId
      | throw <| IO.userError s!"{label}: missing gradient on pass {pass}"
    Utils.assertTensorApprox s!"{label} backward {pass}"
      (← Utils.anyBufferToTensor (s := s) gradient) (Tensor.full s 0.0) (tol := 0.0)
    LibTorch.Tape.releaseSparseGrads gradients
    let after ← LibTorch.Buffer.allocatorStats
    unless after.liveBytes == forward.liveBytes do
      throw <| IO.userError s!"{label}: backward retained temporary payloads"
  for node in tape.nodes do
    discard <| LibTorch.Buffer.releaseIO node.value.buf
    for buffer in node.cleanup do
      discard <| LibTorch.Buffer.releaseIO buffer
  let retired ← LibTorch.Buffer.allocatorStats
  unless retired.liveBytes == baseline.liveBytes do
    throw <| IO.userError s!"{label}: tape retirement retained payloads"

/-- Forward scratch is retired while the output remains usable by repeated backward passes. -/
def checkScratchLifetime (logarithmic : Bool) : IO Unit := do
  let label := if logarithmic then "log_softmax" else "softmax"
  Internal.assertRejectedWithoutAllocation s!"{label} invalid node" fun _ =>
    if logarithmic then LibTorch.Tape.logSoftmaxLast (s := [2, 3]) LibTorch.Tape.empty 0
    else LibTorch.Tape.softmaxLast (s := [2, 3]) LibTorch.Tape.empty 0
  let x : Tensor Float [2, 3] := [[0.1, -0.2, 0.3], [0.05, 0.25, -0.15]]
  let upstream : Tensor Float [2, 3] := [[1.0, -2.0, 0.5], [0.25, 3.0, -1.0]]
  let baseline ← Runtime.Autograd.LibTorch.Buffer.allocatorStats
  let input ← Runtime.Autograd.LibTorch.Buffer.ofFloatArrayIO
    (Runtime.Autograd.LibTorch.Convert.flattenFloat x)
  let (tape, xId) := Runtime.Autograd.LibTorch.Tape.empty.leaf { s := [2, 3], buf := input }
  let result ← IO.lazyPure fun _ =>
    if logarithmic then Runtime.Autograd.LibTorch.Tape.logSoftmaxLast (s := [2, 3]) tape xId
    else Runtime.Autograd.LibTorch.Tape.softmaxLast (s := [2, 3]) tape xId
  let (tape, yId) ← IO.ofExcept result
  let some node := tape.getNode? yId
    | throw <| IO.userError s!"{label}: missing output node"
  Internal.checkWrongUpstream label [2, 3] node
  let forward ← Runtime.Autograd.LibTorch.Buffer.allocatorStats
  -- Only the six input and six output elements may remain live.
  unless forward.liveBytes == baseline.liveBytes + 48 do
    throw <| IO.userError
      s!"{label}: retained forward scratch ({forward.liveBytes} live bytes)"
  let expected :=
    if logarithmic then Activation.logSoftmaxBackwardSpec (α := Float) 1
      (Activation.logSoftmaxSpec (α := Float) 1 x) upstream
    else Activation.softmaxBackwardSpec (α := Float) 1 x upstream
  for pass in [0:3] do
    let seed ← Runtime.Autograd.LibTorch.Buffer.ofFloatArrayIO
      (Runtime.Autograd.LibTorch.Convert.flattenFloat upstream)
    let gradients ← Runtime.Autograd.LibTorch.Tape.backwardSparse tape yId
      { s := [2, 3], buf := seed } (fun id => id == xId)
    let some gradient := gradients.get? xId
      | throw <| IO.userError s!"{label}: missing input gradient on pass {pass}"
    Utils.assertTensorApprox s!"{label} repeated backward {pass}"
      (← Utils.anyBufferToTensor (s := [2, 3]) gradient) expected (tol := 2e-3)
    Runtime.Autograd.LibTorch.Tape.releaseSparseGrads gradients
    let after ← Runtime.Autograd.LibTorch.Buffer.allocatorStats
    unless after.liveBytes == forward.liveBytes do
      throw <| IO.userError s!"{label}: backward retained temporary payloads"
  for node in tape.nodes do
    discard <| Runtime.Autograd.LibTorch.Buffer.releaseIO node.value.buf
    for buffer in node.cleanup do
      discard <| Runtime.Autograd.LibTorch.Buffer.releaseIO buffer
  let retired ← Runtime.Autograd.LibTorch.Buffer.allocatorStats
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
  for s in (#[Shape.scalar, ([0] : Shape), ([2, 0] : Shape), ([0, 3] : Shape)]) do
    Internal.checkDegenerateLast s false
    Internal.checkDegenerateLast s true

  if Runtime.Autograd.LibTorch.Buffer.runtimeStatus = .nativeAvailable then
    checkInteriorAxisSession

end Softmax
end Cuda
end Tests
