/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Program
public import TorchLean.NN.Session.Optimizer
public import TorchLean.NN.ScalarTrainer.EagerOps
public import TorchLeanTest.Harness.CUDA
public import TorchLean.Tensor.Literal

/-!
# CUDA Kernel Coverage: Softmax

Softmax and log-softmax along the last axis of a `[2, 3]` input have the same values and VJPs on
the CPU and CUDA eager tapes, within `2e-3`. On the CUDA tape, a rejected node or upstream gradient
allocates nothing, the forward pass keeps only its input and output, repeated backward passes free
their temporaries, and empty inputs keep their shape and get zero VJPs. Softmax along the interior
axis of a `[2, 2, 2]` input on a CUDA session is compared with `Tensor.softmax` and
`Tensor.softmaxBackward`.
-/

@[expose] public section

namespace TorchLean.Test
namespace CUDA
namespace Softmax

open TorchLean.Tensor
open LibTorch

/-- Record one operation on a fresh eager session and return its value and its VJP. -/
def valueAndVJP {s : Shape} (device : Backend.Device) (name : String)
    (op : TensorRef Float s → SessionM Float (TensorRef Float s))
    (x upstream : Tensor Float s) : IO (Tensor Float s × Tensor Float s) := do
  let sess ← Session.create (α := Float) { device }
  let xRef ← sess.input x (name := some name) (requiresGrad := true)
  let yRef ← op xRef sess
  let y ← sess.getValue yRef
  let grads ← sess.backwardDenseAll yRef upstream
  pure (y, ← Session.grad grads xRef)

/-- Softmax along the interior axis of a `[2, 2, 2]` input on a CUDA session, value and VJP,
against `Tensor.softmax` and `Tensor.softmaxBackward` within `2e-3`. -/
def checkInteriorAxisSession : IO Unit := do
  let s : Shape := [2, 2, 2]
  let x : Tensor Float s :=
    (Tensor.from (#[0, 2, 1, 4, 3, 8, 7, 9] : Array Float)).reshape [2, 2, 2] (by dsimp; decide)
  let upstream : Tensor Float s :=
    (Tensor.from #[1.0, -2.0, 3.0, 4.0, -1.0, 2.0, 5.0, -3.0]).reshape [2, 2, 2] (by dsimp; decide)
  let (actual, gradient) ← valueAndVJP .cuda "interior_axis_input"
    (fun input => Ops.apply (m := SessionM Float) (α := Float)
      (.softmax (s := s) ⟨1, Shape.AxisInBounds.proof⟩) (input, ())) x upstream
  let expected := Tensor.softmax x ⟨1, by decide⟩
  let expectedGradient := Tensor.softmaxBackward x upstream ⟨1, by decide⟩
  Harness.assertTensorApprox (s := s) "softmax interior-axis forward" actual expected (tol := 2e-3)
  Harness.assertTensorApprox (s := s) "softmax interior-axis backward" gradient expectedGradient
    (tol := 2e-3)

/-- Softmax along axis 1 of a `[2, 3]` input on an eager session on `device`, value and VJP. -/
def eval (device : Backend.Device) (x upstream : Tensor Float [2, 3]) :
    IO (Tensor Float [2, 3] × Tensor Float [2, 3]) :=
  valueAndVJP device "softmax_input"
    (fun input => Ops.apply (m := SessionM Float) (α := Float)
      (.softmax (s := [2, 3]) ⟨1, Shape.AxisInBounds.proof⟩) (input, ())) x upstream

/-- Log-softmax along axis 1 of a `[2, 3]` input on an eager session on `device`, value and
VJP. -/
def evalLog (device : Backend.Device)
    (x upstream : Tensor Float [2, 3]) :
    IO (Tensor Float [2, 3] × Tensor Float [2, 3]) :=
  valueAndVJP device "log_softmax_input"
    (fun input => Ops.apply (m := SessionM Float) (α := Float)
      (.logSoftmax (s := [2, 3]) ⟨1, Shape.AxisInBounds.proof⟩) (input, ()))
    x upstream

/-- Fails unless `action` returns an error and leaves the live bytes and the counts of native
allocations and buffer wrappers unchanged. -/
def assertRejectedNoAlloc {α : Type} (label : String)
    (action : Unit → IO (Except String α)) : IO Unit := do
  let before ← LibTorch.Buffer.Memory.read
  let result ← action ()
  let after ← LibTorch.Buffer.Memory.read
  match result with
  | .ok _ => throw <| IO.userError s!"{label}: unexpectedly accepted invalid input"
  | .error _ => pure ()
  unless after.liveBytes == before.liveBytes && after.allocCount == before.allocCount &&
      after.wrapperAllocCount == before.wrapperAllocCount do
    throw <| IO.userError s!"{label}: allocated before rejecting invalid input"

/-- The VJP closure of `node` itself, without the seed validation of the backward pass around it,
must reject an upstream gradient of the wrong shape and one of the wrong length, allocating
nothing. -/
def checkWrongUpstream (label : String) (s : Shape) (node
    : Tape.Node LibTorch.SomeBuffer) :
    IO Unit := do
  let seed ← LibTorch.Buffer.ofFloatArrayIO (FloatArray.mk (Array.replicate s.size 2.0))
  Softmax.assertRejectedNoAlloc s!"{label} wrong upstream shape" fun _ =>
    IO.lazyPure fun _ => node.backward { s := .dim 1 s, buf := seed }
  discard <| LibTorch.Buffer.releaseIO seed
  let wrongLength ← LibTorch.Buffer.ofFloatArrayIO
    (FloatArray.mk (Array.replicate (s.size + 1) 2.0))
  Softmax.assertRejectedNoAlloc s!"{label} wrong upstream length" fun _ =>
    IO.lazyPure fun _ => node.backward { s := s, buf := wrongLength }
  discard <| LibTorch.Buffer.releaseIO wrongLength

/-- Softmax or log-softmax along `axis` of an empty input of shape `s` on the CUDA tape. An invalid
node and wrong upstream gradients are rejected without allocating, the output keeps the shape `s`,
three backward passes give zero VJPs and free their temporaries, and retiring the tape frees every
payload. -/
def checkDegenerate (s : Shape) (axis : Nat) [Shape.AxisInBounds axis s]
    (logarithmic : Bool) : IO Unit := do
  let label := if logarithmic then "degenerate log_softmax" else "degenerate softmax"
  let dim : Fin s.rank := ⟨axis, Shape.AxisInBounds.proof⟩
  let applyOp := fun (tape : Tape LibTorch.SomeBuffer) (xId
      : Nat) =>
    if logarithmic then
      LibTorch.Tape.apply tape "log_softmax" (.logSoftmax dim) (xId, ()) (pure .float32)
    else LibTorch.Tape.apply tape "softmax" (.softmax dim) (xId, ()) (pure .float32)
  Softmax.assertRejectedNoAlloc s!"{label} invalid node" fun _ => do
    return (← applyOp Tape.empty 0).2
  let baseline ← LibTorch.Buffer.Memory.read
  let input ← LibTorch.Buffer.ofFloatArrayIO (FloatArray.mk (Array.replicate s.size 7.0))
  let (tape, xId) := Tape.leaf Tape.empty
    { s := s, buf := input : LibTorch.SomeBuffer }
  let (tape, result) ← applyOp tape xId
  let (yId, ()) ← IO.ofExcept result
  let some node := tape.getNode? yId
    | throw <| IO.userError s!"{label}: missing output node"
  unless node.value.s == s do
    throw <| IO.userError s!"{label}: output shape changed"
  let output ← LibTorch.Buffer.toFloatArrayIO node.value.buf
  unless output.size == s.size do
    throw <| IO.userError s!"{label}: output length changed"
  Softmax.checkWrongUpstream label s node
  let forward ← LibTorch.Buffer.Memory.read
  unless forward.liveBytes == baseline.liveBytes + (8 * s.size).toUInt64 do
    throw <| IO.userError s!"{label}: retained forward payloads"
  for pass in [0:3] do
    let seed ← LibTorch.Buffer.ofFloatArrayIO
      (FloatArray.mk (Array.replicate s.size 3.0))
    let gradients ← LibTorch.Tape.backwardSparse tape yId
      { s := s, buf := seed } (fun id => id == xId)
    let some gradient := gradients.get? xId
      | throw <| IO.userError s!"{label}: missing gradient on pass {pass}"
    Harness.assertTensorApprox s!"{label} backward {pass}"
      (← Harness.someBufferToTensor (s := s) gradient) (Tensor.full s 0.0) (tol := 0.0)
    LibTorch.Tape.releaseSparseGrads gradients
    let after ← LibTorch.Buffer.Memory.read
    unless after.liveBytes == forward.liveBytes do
      throw <| IO.userError s!"{label}: backward retained temporary payloads"
  for node in tape.nodes do
    discard <| LibTorch.Buffer.releaseIO node.value.buf
    for buffer in node.cleanup do
      discard <| LibTorch.Buffer.releaseIO buffer.buf
  let retired ← LibTorch.Buffer.Memory.read
  unless retired.liveBytes == baseline.liveBytes do
    throw <| IO.userError s!"{label}: tape retirement retained payloads"

/-- Softmax or log-softmax of a `[2, 3]` input on the CUDA tape keeps only its input and output
after the forward pass. Three backward passes then match `Tensor.softmaxBackward` or
`Tensor.logSoftmaxBackward` within `2e-3` and free their temporaries, and retiring the tape frees
every payload. -/
def checkScratchLifetime (logarithmic : Bool) : IO Unit := do
  let label := if logarithmic then "log_softmax" else "softmax"
  assertRejectedNoAlloc s!"{label} invalid node" fun _ => do
    if logarithmic then
      return (← LibTorch.Tape.apply .empty "log_softmax"
        (.logSoftmax (s := [2, 3]) ⟨1, by decide⟩) (0, ()) (pure .float32)).2
    else
      return (← LibTorch.Tape.apply .empty "softmax" (.softmax (s := [2, 3]) ⟨1, by decide⟩)
        (0, ()) (pure .float32)).2
  let x : Tensor Float [2, 3] := [[0.1, -0.2, 0.3], [0.05, 0.25, -0.15]]
  let upstream : Tensor Float [2, 3] := [[1.0, -2.0, 0.5], [0.25, 3.0, -1.0]]
  let baseline ← LibTorch.Buffer.Memory.read
  let input ← LibTorch.Buffer.ofFloatArrayIO (x.to FloatArray)
  let (tape, xId) := Tape.leaf Tape.empty
    { s := [2, 3], buf := input : LibTorch.SomeBuffer }
  let result ←
    if logarithmic then
      LibTorch.Tape.apply tape "log_softmax" (.logSoftmax (s := [2, 3]) ⟨1, by decide⟩) (xId, ())
        (pure .float32)
    else
      LibTorch.Tape.apply tape "softmax" (.softmax (s := [2, 3]) ⟨1, by decide⟩) (xId, ())
        (pure .float32)
  let (tape, result) := result
  let (yId, ()) ← IO.ofExcept result
  let some node := tape.getNode? yId
    | throw <| IO.userError s!"{label}: missing output node"
  checkWrongUpstream label [2, 3] node
  let forward ← LibTorch.Buffer.Memory.read
  -- Only the six input and six output entries, four bytes each, may stay live.
  unless forward.liveBytes == baseline.liveBytes + 48 do
    throw <| IO.userError
      s!"{label}: retained forward scratch ({forward.liveBytes} live bytes)"
  let expected :=
    if logarithmic then
      Tensor.logSoftmaxBackward (Tensor.logSoftmax x ⟨1, Nat.one_lt_two⟩) upstream
        ⟨1, Nat.one_lt_two⟩
    else Tensor.softmaxBackward x upstream ⟨1, Nat.one_lt_two⟩
  for pass in [0:3] do
    let seed ← LibTorch.Buffer.ofFloatArrayIO (upstream.to FloatArray)
    let gradients ← LibTorch.Tape.backwardSparse tape yId
      { s := [2, 3], buf := seed } (fun id => id == xId)
    let some gradient := gradients.get? xId
      | throw <| IO.userError s!"{label}: missing input gradient on pass {pass}"
    Harness.assertTensorApprox s!"{label} repeated backward {pass}"
      (← Harness.someBufferToTensor (s := [2, 3]) gradient) expected (tol := 2e-3)
    LibTorch.Tape.releaseSparseGrads gradients
    let after ← LibTorch.Buffer.Memory.read
    unless after.liveBytes == forward.liveBytes do
      throw <| IO.userError s!"{label}: backward retained temporary payloads"
  for node in tape.nodes do
    discard <| LibTorch.Buffer.releaseIO node.value.buf
    for buffer in node.cleanup do
      discard <| LibTorch.Buffer.releaseIO buffer.buf
  let retired ← LibTorch.Buffer.Memory.read
  unless retired.liveBytes == baseline.liveBytes do
    throw <| IO.userError s!"{label}: tape retirement retained payloads"

/-- Runs the checks of this file: CPU against CUDA values and VJPs of softmax and log-softmax
within `2e-3`, `checkScratchLifetime`, `checkDegenerate` on `[0]`, `[2, 0]` and `[0, 3]`, and
`checkInteriorAxisSession` when the native runtime is available. -/
def run : IO Unit := do
  IO.println "=== CUDA kernel coverage: softmax ==="

  let s : Shape := [2, 3]
  let x : Tensor Float s :=
    (Tensor.from #[
      0.10, -0.20, 0.30,
      0.05,  0.25, -0.15
    ]).reshape [2, 3] (by dsimp; decide)
  let upstream : Tensor Float s := [[1.0, -2.0, 0.5], [0.25, 3.0, -1.0]]
  let (yCpu, dxCpu) ← eval .cpu x upstream
  let (yCuda, dxCuda) ← eval .cuda x upstream

  -- Compare: both sides are binary64, but the CUDA tape runs ATen kernels, so allow a tolerance.
  Harness.assertTensorApprox (s := s) "softmax forward" yCuda yCpu (tol := 2e-3)
  Harness.assertTensorApprox (s := s) "softmax backward" dxCuda dxCpu (tol := 2e-3)

  let (yLogCpu, dxLogCpu) ← evalLog .cpu x upstream
  let (yLogCuda, dxLogCuda) ← evalLog .cuda x upstream

  Harness.assertTensorApprox (s := s) "log_softmax forward" yLogCuda yLogCpu (tol := 2e-3)
  Harness.assertTensorApprox (s := s) "log_softmax backward" dxLogCuda dxLogCpu (tol := 2e-3)

  checkScratchLifetime false
  checkScratchLifetime true
  for logarithmic in [false, true] do
    checkDegenerate [0] 0 logarithmic
    checkDegenerate [2, 0] 1 logarithmic
    checkDegenerate [0, 3] 1 logarithmic

  if (← LibTorch.Buffer.RuntimeStatus.query) = .nativeAvailable then
    checkInteriorAxisSession

end Softmax
end CUDA
end TorchLean.Test
