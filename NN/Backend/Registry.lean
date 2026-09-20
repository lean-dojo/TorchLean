/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Backend.NativeCUDA
public import NN.Backend.Reference
public import NN.Backend.LibTorch

/-!
# Backend Registry

Registry of backend capsules known to TorchLean's planner.

Capsules are contributed by operation or provider modules and flattened into one planner catalog.
Model architectures never appear here: they lower to backend operations, and the planner chooses a
capsule for each operation. Optional external modules such as LibTorch are included only when the
caller enables them and chooses an assurance policy that admits them.
-/

@[expose] public section

namespace NN
namespace Backend
namespace Registry

/-- A named, independently maintained contribution to the backend catalog. -/
structure CapsuleModule where
  name : String
  capsules : Array KernelCapsule
  deriving Repr

/-- Flatten capsule modules while preserving module and local preference order. -/
def flatten (modules : Array CapsuleModule) : Array KernelCapsule :=
  modules.flatMap (·.capsules)

/-- First repeated module name, if the registry contains two contributions with the same name. -/
def firstDuplicateModuleName? (modules : Array CapsuleModule) : Option String :=
  modules.findSome? fun module =>
    if 1 < (modules.filter fun candidate => candidate.name == module.name).size then
      some module.name
    else
      none

/-- First repeated capsule identity after flattening, if one exists. -/
def firstDuplicateCapsuleName? (capsules : Array KernelCapsule) : Option String :=
  capsules.findSome? fun capsule =>
    if 1 < (capsules.filter capsule.sameIdentity).size then
      some capsule.name
    else
      none

/--
Validate the identities that make registry ordering meaningful.

Different providers may implement the same operation. What is rejected is registering the same
named module twice or repeating the same capsule name, operation, provider, and device tuple.
-/
def validateModules (modules : Array CapsuleModule) : Except String Unit := do
  if let some name := firstDuplicateModuleName? modules then
    throw s!"duplicate backend capsule module `{name}`"
  if let some name := firstDuplicateCapsuleName? (flatten modules) then
    throw s!"duplicate backend capsule identity `{name}`"

/-- Maintained operation/provider modules. A new architecture does not modify this list; only a new
primitive implementation or provider does. -/
def maintainedModules : Array CapsuleModule :=
  #[ { name := "attention", capsules := Attention.capsules }
  , { name := "native-cuda", capsules := NativeCUDA.capsules }
  , { name := "reference", capsules := Reference.capsules }
  ]

/-- LibTorch's independently maintained provider module. Profiles opt into it by adding this module
to their catalog; provider preference is handled by the kernel policy, so module order
does not encode backend selection. -/
def libTorchModule : CapsuleModule :=
  { name := "libtorch", capsules := LibTorch.capsules }

end Registry
end Backend
end NN
