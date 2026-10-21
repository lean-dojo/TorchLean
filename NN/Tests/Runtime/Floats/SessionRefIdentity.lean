/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core.BackwardOptim
public import NN.Runtime.Autograd.Torch.Core.Ops.Elementwise

/-! Regression checks for session-reference ownership and CUDA cache cleanup. -/

@[expose] public section

open Spec TorchLean
open TorchLean TorchLean.Tensor
open Runtime.Autograd
open Runtime.Autograd.Torch.Internal (EagerSession)

namespace Tests.Floats.SessionRefIdentity

/-- Observe recording and cache counters without evaluating or changing the session. -/
def recordingState (state : EagerSession Float) : IO (Array Nat) := do
  pure #[(← state.tape.get).size,
    ← state.rngCounter.get, ← state.refGeneration.get,
    (← state.selectedBackends.get).size, (← state.paramsByLeaf.get).size]

/-- Rejected references fail before recording, cache changes, or native allocations. -/
def expectReferenceFailure {α : Type} (session : EagerSession Float)
    (label expected : String) (action : IO α) : IO Unit := do
  let before ← recordingState session
  let memoryBefore ← LibTorch.Buffer.memory
  let error? ← try
    let _ ← action
    pure none
  catch error =>
    pure (some error.toString)
  unless error? == some expected do
    throw <| IO.userError s!"{label}: expected {expected}, got {error?}"
  unless (← recordingState session) == before do
    throw <| IO.userError s!"{label}: rejection changed recording or cache state"
  let memoryAfter ← LibTorch.Buffer.memory
  unless memoryAfter.allocCount == memoryBefore.allocCount &&
      memoryAfter.wrapperAllocCount == memoryBefore.wrapperAllocCount do
    throw <| IO.userError s!"{label}: rejection allocated native buffers"

def checkReferences : IO Unit := do
  let first ← EagerSession.new (α := Float)
  let second ← EagerSession.new (α := Float)
  let firstRef ← EagerSession.input first (Tensor.scalar 1.0)
  let secondRef ← EagerSession.input second (Tensor.scalar 2.0)
  let foreign := "torch: tensor reference belongs to a different session"
  let missing := "torch: tensor reference has no session owner"
  expectReferenceFailure second "cross-session tensor op" foreign <|
    EagerSession.add second firstRef secondRef
  let unowned : Torch.TensorRef Float Shape.scalar := { id := 1000000 }
  expectReferenceFailure second "missing owner before foreign operand" missing <|
    EagerSession.add second unowned firstRef
  expectReferenceFailure second "foreign operand before missing owner" foreign <|
    EagerSession.add second firstRef unowned

  EagerSession.resetTape first
  let currentRef ← EagerSession.input first (Tensor.scalar 3.0)
  expectReferenceFailure first "foreign owner before stale generation" foreign <|
    EagerSession.add first secondRef currentRef
  expectReferenceFailure first "post-reset tensor op"
    ("torch: stale tensor reference from generation 0; current generation is 1 " ++
      "(resetTape invalidates recorded references)") <|
    EagerSession.add first firstRef currentRef

/-- Reused output ids must not bypass ownership checks, even before a CUDA device gate. -/
def checkBackwardReferences (device : NN.Backend.Device) : IO Unit := do
  let session ← EagerSession.new (α := Float) { device }
  let foreignSession ← EagerSession.new (α := Float) { device }
  try
    let input ← session.input (Tensor.scalar 2.0) (requiresGrad := true)
    let output ← session.mul input input
    let foreignInput ← foreignSession.input (Tensor.scalar 4.0) (requiresGrad := true)
    let foreignOutput ← foreignSession.mul foreignInput foreignInput
    let unowned : Torch.TensorRef Float Shape.scalar := { id := output.id }
    let actions : Array (String × (Torch.TensorRef Float Shape.scalar → IO Unit)) := #[
      ("backwardDenseAll", fun ref =>
        discard <| session.backwardDenseAll ref (Tensor.scalar 1.0)),
      ("backwardDenseAllCuda", fun ref =>
        discard <| session.backwardDenseAllCuda ref (Tensor.scalar 1.0)),
      ("backwardScalarParamGradsCuda", fun ref =>
        discard <| session.backwardScalarParamGradsCuda ref)]
    for (name, action) in actions do
      expectReferenceFailure session s!"{device.cliName} {name}: foreign output"
        "torch: tensor reference belongs to a different session" (action foreignOutput)
      expectReferenceFailure session s!"{device.cliName} {name}: unowned output"
        "torch: tensor reference has no session owner" (action unowned)

    session.resetTape
    let parameter ← session.param (Tensor.scalar 3.0)
    let current ← session.use parameter
    let square ← session.mul current current
    unless square.id == output.id do
      throw <| IO.userError "backward reference check did not reuse the output id"
    for (name, action) in actions do
      expectReferenceFailure session s!"{device.cliName} {name}: stale output"
        ("torch: stale tensor reference from generation 0; current generation is 1 " ++
          "(resetTape invalidates recorded references)") (action output)
    let gradients ← session.backwardDenseAll square (Tensor.scalar 1.0)
    let gradient ← EagerSession.grad gradients current
    unless gradient.item == 6.0 do
      throw <| IO.userError "valid square backward failed after rejected references"
    if device == .cuda then
      let gradients ← session.backwardScalarParamGradsCuda square
      try
        let some gradient := gradients.get? current.id
          | throw <| IO.userError "valid sparse square backward omitted its parameter gradient"
        let values ← LibTorch.Buffer.toFloatArrayIO gradient.buf
        unless values.size == 1 && values.get! 0 == 6.0 do
          throw <| IO.userError "valid sparse square backward returned the wrong gradient"
      finally
        LibTorch.Tape.releaseSparseGrads gradients
  finally
    session.resetTape
    foreignSession.resetTape

def checkCudaCacheClear : IO Unit := do
  let value ← IO.mkRef (Tensor.scalar 1.0)
  let buffer ← Runtime.Autograd.LibTorch.Buffer.fullIO 1 1.0
  let cudaValue ← IO.mkRef (some { s := Shape.scalar, buf := buffer })
  let hostCurrent ← IO.mkRef true
  let param : Torch.Param Float Shape.scalar :=
    { value, cudaValue, hostCurrent, requiresGrad := true }
  Torch.SomeParam.releaseCachedCudaValue param
  match ← cudaValue.get with
  | none => pure ()
  | some _ => throw <| IO.userError "CUDA cache release did not clear cudaValue"
  Torch.SomeParam.releaseCachedCudaValue param

def run : IO Unit := do
  checkReferences
  checkBackwardReferences .cpu
  if Runtime.Autograd.LibTorch.Buffer.runtimeStatus == .nativeAvailable then
    checkBackwardReferences .cuda
    checkCudaCacheClear

end Tests.Floats.SessionRefIdentity
