/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core.Ops.Elementwise

/-! Regression checks for session-reference ownership and CUDA cache cleanup. -/

@[expose] public section

open Spec TorchLean
open TorchLean TorchLean.Tensor
open Runtime.Autograd
open Runtime.Autograd.Torch.Internal (EagerSession)

namespace Tests.Floats.SessionRefIdentity

def expectFailure {α : Type} (label : String) (action : IO α) : IO Unit := do
  let failed ← try
    let _ ← action
    pure false
  catch _ =>
    pure true
  unless failed do
    throw <| IO.userError s!"{label}: expected failure"

/-- Observe recording and cache counters without evaluating or changing the session. -/
def recordingState (state : EagerSession Float) : IO (Array Nat) := do
  pure #[(← state.tape.get).size,
    ← state.rngCounter.get, ← state.refGeneration.get,
    (← state.selectedBackends.get).size, (← state.paramsByLeaf.get).size]

/-- Rejected references must fail at the intended guard before recording or cache changes. -/
def expectReferenceFailure {α : Type} (session : EagerSession Float)
    (label expected : String) (action : IO α) : IO Unit := do
  let before ← recordingState session
  let error? ← try
    let _ ← action
    pure none
  catch error =>
    pure (some error.toString)
  unless error? == some expected do
    throw <| IO.userError s!"{label}: expected {expected}, got {error?}"
  unless (← recordingState session) == before do
    throw <| IO.userError s!"{label}: rejection changed recording or cache state"

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

def checkCudaCacheClear : IO Unit := do
  let value ← IO.mkRef (Tensor.scalar 1.0)
  let buffer ← Runtime.Autograd.LibTorch.Buffer.fullIO 1 1.0
  let cudaValue ← IO.mkRef (some { s := Shape.scalar, buf := buffer })
  let hostCurrent ← IO.mkRef true
  let param : Torch.Param Float Shape.scalar :=
    { value, cudaValue, hostCurrent, requiresGrad := true }
  Torch.AnyParam.releaseCachedCudaValue param
  match ← cudaValue.get with
  | none => pure ()
  | some _ => throw <| IO.userError "CUDA cache release did not clear cudaValue"
  Torch.AnyParam.releaseCachedCudaValue param

def run : IO Unit := do
  checkReferences
  if Runtime.Autograd.LibTorch.Buffer.runtimeStatus == .nativeAvailable then
    checkCudaCacheClear

end Tests.Floats.SessionRefIdentity
