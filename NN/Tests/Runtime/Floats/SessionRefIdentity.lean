/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Model.Session.Autograd
public import NN.Runtime.Autograd.Model.Session.Ops

/-! Regression checks for session-reference ownership and CUDA cache cleanup. -/

@[expose] public section

open Spec TorchLean
open TorchLean TorchLean.Tensor
open Runtime.Autograd
open Runtime.Autograd.Model

namespace Tests.Floats.SessionRefIdentity

def expectFailure {α : Type} (label : String) (action : IO α) : IO Unit := do
  let failed ← try
    let _ ← action
    pure false
  catch _ =>
    pure true
  unless failed do
    throw <| IO.userError s!"{label}: expected failure"

def checkExecutionMode (execution : Torch.ExecutionMode) : IO Unit := do
  let first ← Session.new (α := Float) (options := { execution := execution })
  let second ← Session.new (α := Float) (options := { execution := execution })
  let firstRef ← Session.input first (Tensor.scalar 1.0)
  let secondRef ← Session.input second (Tensor.scalar 2.0)
  expectFailure "cross-session tensor op" <| Session.add second firstRef secondRef

  let firstNat ← Session.inputNat first 7
  expectFailure "cross-session Nat read" <| Session.getNat second firstNat

  Session.resetTape first
  let currentRef ← Session.input first (Tensor.scalar 3.0)
  expectFailure "post-reset tensor op" <| Session.add first firstRef currentRef
  expectFailure "post-reset Nat read" <| Session.getNat first firstNat

def checkCudaCacheClear : IO Unit := do
  let value ← IO.mkRef (Tensor.scalar 1.0)
  let buffer ← Runtime.Autograd.Cuda.Buffer.fullIO 1 1.0
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
  checkExecutionMode .eager
  checkExecutionMode .typedGraph
  if Runtime.Autograd.Cuda.Buffer.runtimeStatus == .nativeAvailable then
    checkCudaCacheClear
  for device in [NN.Backend.Device.cuda, .metal, .custom] do
    expectFailure "typed graph unsupported device" <|
      Session.new (α := Float) { execution := .typedGraph, device }

end Tests.Floats.SessionRefIdentity
