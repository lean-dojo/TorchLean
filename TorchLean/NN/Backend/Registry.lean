/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Backend.Reference
public import TorchLean.NN.LibTorch.Kernel

/-!
# Backend Registry

The implementation tables TorchLean's planner knows, in preference order: the LibTorch CUDA table
first, then the reference table. Each offers a capsule per registered operator, attributes
included. The LibTorch CUDA table offers the capsule of each entry of
`TorchLean.LibTorch.Kernel.ofOp` that carries a contract (every entry but `detach`'s), and the
reference table offers one for each operator with a reference contract.

Model architectures never appear here. They lower to registered operators, and the planner chooses
a capsule for each one. Build availability removes the LibTorch CUDA table from CPU-only profiles,
and implementations a caller supplies pass the same availability and assurance checks.
-/

@[expose] public section

namespace TorchLean
namespace Backend
namespace Registry

/-- The first implementation name that occurs twice in `impls`, if any. -/
def firstDuplicateName? (impls : Array Implementation) : Option String :=
  impls.findSome? fun impl =>
    if 1 < impls.countP (fun candidate => candidate.name == impl.name) then
      some impl.name
    else
      none

/--
Reject an implementation list in which two tables share a name, since the names are what make
implementation order meaningful.

Different providers may implement the same operator. Two offered capsules with the same identity
are caught later, when `planOp` plans their operator.
-/
def validate (impls : Array Implementation) : Except String Unit := do
  if let some name := firstDuplicateName? impls then
    throw s!"duplicate backend implementation `{name}`"

/-- The maintained implementation tables. Adding an architecture leaves this list unchanged, and
only adding a provider extends it. -/
def maintained : Array Implementation :=
  #[LibTorch.implementation, Reference.implementation]

end Registry
end Backend
end TorchLean
