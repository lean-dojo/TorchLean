/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Numeric.InverseTrig
public import TorchLean.Tensor.Context
public import TorchLean.Tensor.Elementwise

/-!
# Optional capabilities of a scalar type

Some operators need an operation that not every scalar type has: the integral roundings and `fmod`
(`Rounding`), the error function (`Erf`), the inverse trigonometric functions (`InverseTrig`),
the cube root (`Cbrt`), C's `nextafter` (`NextAfter`) and the interchange word of a binary format
(`Bits`, for `bitcast`). `Op.Capability C α` holds an instance of `C α` where synthesis finds one
and `none` otherwise, and `Op.Capabilities α` bundles these six. An operator that needs a missing
capability has no rule at that scalar. `Float` has `Rounding` and `InverseTrig` but no `Erf`,
`Cbrt`, `NextAfter` or `Bits`, while the configured binary formats of
FloatLib have every capability but `Erf`.
-/

@[expose] public section

namespace TorchLean

/-- An instance of `C α` when the scalar type has one. -/
class Op.Capability (C : Type → Type) (α : Type) where
  /-- The instance, or `none`. -/
  get? : Option (C α)

namespace Op.Capability

variable {α : Type}

/-- The low-priority fallback, `none`, for a scalar type without the capability. -/
instance (priority := low) absent {C : Type → Type} : Op.Capability C α := ⟨none⟩

/-- The integral roundings, where the scalar has them. -/
instance rounding [instance_ : Rounding α] : Op.Capability Rounding α := ⟨some instance_⟩

/-- The error function, where the scalar has it. -/
instance erf [instance_ : Erf α] : Op.Capability Erf α := ⟨some instance_⟩

/-- The inverse trigonometric functions, where the scalar has them. -/
instance inverseTrig [instance_ : InverseTrig α] : Op.Capability InverseTrig α := ⟨some instance_⟩

/-- The cube root, where the scalar has it. -/
instance cbrt [instance_ : Cbrt α] : Op.Capability Cbrt α := ⟨some instance_⟩

/-- C's `nextafter`, where the scalar has it. -/
instance nextafter [instance_ : NextAfter α] : Op.Capability NextAfter α := ⟨some instance_⟩

/-- The interchange word, where the scalar has one. -/
instance bits [instance_ : Bits α] : Op.Capability Bits α := ⟨some instance_⟩

end Op.Capability

/-- The optional capabilities `Op.rule?` reads. -/
class Op.Capabilities (α : Type) where
  /-- The integral roundings and `fmod`. -/
  rounding? : Option (Rounding α)
  /-- The error function. -/
  erf? : Option (Erf α)
  /-- The inverse trigonometric functions. -/
  inverseTrig? : Option (InverseTrig α)
  /-- The cube root. -/
  cbrt? : Option (Cbrt α)
  /-- C's `nextafter`. -/
  nextafter? : Option (NextAfter α)
  /-- The interchange word. -/
  bits? : Option (Bits α)

/-- The capabilities a scalar type has, each found by synthesis. -/
instance Op.Capabilities.ofCapability {α : Type} [rounding : Op.Capability Rounding α]
    [erf : Op.Capability Erf α] [inverseTrig : Op.Capability InverseTrig α]
    [cbrt : Op.Capability Cbrt α]
    [nextafter : Op.Capability NextAfter α] [bits : Op.Capability Bits α] : Op.Capabilities α :=
  ⟨rounding.get?, erf.get?, inverseTrig.get?, cbrt.get?, nextafter.get?,
    bits.get?⟩

end TorchLean
