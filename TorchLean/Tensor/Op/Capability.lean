/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Numeric.InverseTrig
public import TorchLean.Tensor.Context
public import TorchLean.Tensor.Elementwise
public import TorchLean.Tensor.Special.Fn
public import TorchLean.Tensor.ComplexFn

/-!
# Optional capabilities of a scalar type

Some operators need an operation that not every scalar type has: the integral roundings and `fmod`
(`Rounding`), the error function (`Erf`), the inverse trigonometric functions (`InverseTrig`),
the cube root (`Cbrt`), C's `nextafter` (`NextAfter`), the interchange word of a binary format
(`Bits`, for `bitcast`), the special functions (`Special`), C's `frexp` and `ldexp` (`Frexp`), the
fused multiply-add (`Fma`) and the complex functions and power (`ComplexFunctions`).
`Op.Capability C α` (declared in `TorchLean.Tensor.Context`, beside most of the capability classes)
holds an instance of `C α` where synthesis finds one and `none` otherwise, and `Op.Capabilities α`
bundles these ten. An operator that needs a missing capability has no rule at that scalar. `Float`
and `Float32` have `Rounding`, `InverseTrig` and `Fma` (core's `Float.fma` and `Float32.fma`) but
no `Erf`, `Cbrt`, `NextAfter`, `Bits`, `Special`, `Frexp` or `ComplexFunctions`, while the
configured binary formats of FloatLib have every capability but `Erf` and `Special`, and `Frexp`
only in an IEEE format of exponent width at least three, so of bias at least two, where every
normalized mantissa is a value, and where every exponent of a finite value fits in an int32, so
never at exponent width 32 or more (`configuredBinaryFrexpCapability`). Only `ℝ` and its dual
numbers have `Special`, and `ℝ` has `Frexp`, `Fma` and `ComplexFunctions`.
-/

@[expose] public section

namespace TorchLean

-- `C α` is a class at every use, though not before `C` is known.
set_option checkBinderAnnotations false in
/-- The instance of `C α`, where synthesis finds one. -/
instance Op.Capability.present {C : Type → Type} {α : Type} [instance_ : C α] :
    Op.Capability C α :=
  ⟨some instance_⟩

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
  /-- The special functions. -/
  special? : Option (Special α)
  /-- C's `frexp` and `ldexp`. -/
  frexp? : Option (Frexp α)
  /-- The fused multiply-add. -/
  fma? : Option (Fma α)
  /-- The complex functions and the complex power. -/
  complex? : Option (ComplexFunctions α)

/-- The capabilities a scalar type has, each found by synthesis. -/
instance Op.Capabilities.ofCapability {α : Type} [rounding : Op.Capability Rounding α]
    [erf : Op.Capability Erf α] [inverseTrig : Op.Capability InverseTrig α]
    [cbrt : Op.Capability Cbrt α]
    [nextafter : Op.Capability NextAfter α] [bits : Op.Capability Bits α]
    [special : Op.Capability Special α] [frexp : Op.Capability Frexp α]
    [fma : Op.Capability Fma α] [complex : Op.Capability ComplexFunctions α] :
    Op.Capabilities α :=
  ⟨rounding.get?, erf.get?, inverseTrig.get?, cbrt.get?, nextafter.get?,
    bits.get?, special.get?, frexp.get?, fma.get?, complex.get?⟩

end TorchLean
