/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core.Session.State

/-!
# Session References

Tensor handles carry both a session owner and a recording generation. Check those
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

/-- Validate one tensor handle before using its numeric tape id. -/
def validateTensorRef {α : Type} [Storage α] (s : EagerSession α) {sh : Shape}
    (x : TensorRef α sh) : IO Unit :=
  RefIdentity.validate s.refOwner s.refGeneration x.identity? "tensor reference"

/-- Validate tensor handles consumed by one operation. -/
def validateRefIdentities {α : Type} [Storage α] (s : EagerSession α)
    (identities : Array (Option RefIdentity)) : IO Unit := do
  for identity? in identities do
    RefIdentity.validate s.refOwner s.refGeneration identity? "tensor reference"

end EagerSession

end Runtime.Autograd.Torch.Internal
