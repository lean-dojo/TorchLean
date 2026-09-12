/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core.Session.State

/-!
# Session References

Tensor and integer handles carry both a session owner and a recording generation. Check those
before using an id: resetting a tape can reuse the same numeric index.
-/

public section

namespace Runtime.Autograd.Torch.Internal

open Spec TorchLean
namespace EagerSession

/-- Capture the current owner and generation for a newly recorded handle. -/
def currentRefIdentity {α : Type} [Storage α] (s : EagerSession α) : IO RefIdentity := do
  pure { owner := s.refOwner, generation := ← s.refGeneration.get }

/-- Construct a tensor handle owned by the current recording phase. -/
def makeTensorRef {α : Type} [Storage α] {sh : Shape}
    (s : EagerSession α) (id : Nat) : IO (TensorRef α sh) := do
  pure { id, identity? := some (← s.currentRefIdentity) }

/-- Construct a non-differentiable handle owned by the current recording phase. -/
def makeNatRef {α : Type} [Storage α] (s : EagerSession α) (id : Nat) : IO NatRef := do
  pure { id, identity? := some (← s.currentRefIdentity) }

/-- Validate one tensor handle before using its numeric tape id. -/
def validateTensorRef {α : Type} [Storage α] (s : EagerSession α) {sh : Shape}
    (x : TensorRef α sh) : IO Unit := do
  match x.identity? with
  | some identity => identity.validateAgainst s.refOwner s.refGeneration "tensor reference"
  | none => throw <| IO.userError "torch: tensor reference has no session owner"

/-- Validate tensor handles consumed by one operation. -/
def validateRefIdentities {α : Type} [Storage α] (s : EagerSession α)
    (identities : Array (Option RefIdentity)) : IO Unit := do
  for identity? in identities do
    match identity? with
    | some identity => identity.validateAgainst s.refOwner s.refGeneration "tensor reference"
    | none => throw <| IO.userError "torch: tensor reference has no session owner"

/-- Validate one non-differentiable handle before using its environment index. -/
def validateNatRef {α : Type} [Storage α] (s : EagerSession α) (x : NatRef) : IO Unit := do
  match x.identity? with
  | some identity => identity.validateAgainst s.refOwner s.refGeneration "Nat reference"
  | none => throw <| IO.userError "torch: Nat reference has no session owner"

/--
Record a non-differentiable `Nat` input in the session environment.

This supports ops that depend on indices/labels that should not receive gradients.
-/
def inputNat {α : Type} [Storage α] (s : EagerSession α) (v : Nat) : IO NatRef := do
  let values ← s.nats.get
  let id := values.size
  s.nats.set (values.push v)
  s.makeNatRef id

/-- Read a previously recorded `NatRef`. -/
def getNat {α : Type} [Storage α] (s : EagerSession α) (r : NatRef) : IO Nat := do
  s.validateNatRef r
  let values ← s.nats.get
  if h : r.id < values.size then
    pure <| values[r.id]'h
  else
    throw <| IO.userError "torch: invalid nat id"

/-- Overwrite a previously recorded `NatRef`. -/
def setNat {α : Type} [Storage α] (s : EagerSession α) (r : NatRef) (v : Nat) : IO Unit := do
  s.validateNatRef r
  let values ← s.nats.get
  if h : r.id < values.size then
    let i : Fin values.size := ⟨r.id, h⟩
    s.nats.set (values.set i v)
  else
    throw <| IO.userError "torch: invalid nat id"

end EagerSession

end Runtime.Autograd.Torch.Internal
