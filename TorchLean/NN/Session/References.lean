/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Session

/-!
# Session References

Tensor handles carry both a session owner and a recording generation. Operations check both before
they use an id, because resetting a tape can reuse the same numeric index.
-/

public section

namespace TorchLean

namespace Session

/-- Capture the current owner and generation for a newly recorded handle. -/
def refIdentity {α : Type} [Storage α] (s : Session α) : IO RefIdentity := do
  pure { owner := s.refOwner, generation := ← s.refGeneration.get }

/-- Construct a tensor handle owned by the current recording phase. -/
def makeTensorRef {α : Type} [Storage α] {sh : Shape}
    (s : Session α) (id : Nat) : IO (TensorRef α sh) := do
  pure { id, identity? := some (← s.refIdentity) }

/-- Validate one tensor handle before using its numeric tape id. -/
def validateTensorRef {α : Type} [Storage α] (s : Session α) {sh : Shape}
    (x : TensorRef α sh) : IO Unit :=
  RefIdentity.validate s.refOwner s.refGeneration x.identity? "tensor reference"

/-- Validate tensor handles consumed by one operation. -/
def validateRefIdentities {α : Type} [Storage α] (s : Session α)
    (identities : Array (Option RefIdentity)) : IO Unit := do
  for identity? in identities do
    RefIdentity.validate s.refOwner s.refGeneration identity? "tensor reference"

end Session

end TorchLean
