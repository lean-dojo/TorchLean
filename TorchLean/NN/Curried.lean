/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Pack

/-!
# Curried Tensor and Reference Arguments

Convert between shape-indexed packs and curried arguments for tensors or any reference family.
-/

@[expose] public section

namespace TorchLean

open TorchLean.Tensor

/-- A curried function accepting one `Ref s` argument per shape in `ss`. -/
def CurriedRef (Ref : Shape → Type) : List Shape → Type → Type
  | [], β => β
  | s :: ss, β => Ref s → CurriedRef Ref ss β

namespace Curried

/--
Type of a curried function accepting one tensor argument per shape in `ss`.

For example, `Function α [s₁, s₂] β` is `Tensor α s₁ → Tensor α s₂ → β`.
-/
abbrev Function (α : Type) [Storage α] : List Shape → Type → Type :=
  CurriedRef (fun s => Tensor α s)

end Curried

namespace CurriedRef

/-- Uncurry a curried reference function to accept a pack of references. -/
def uncurry {Ref : Shape → Type} {β : Type} : {ss : List Shape} →
    CurriedRef Ref ss β → Pack Ref ss → β
  | [], f, .nil => f
  | _s :: ss, f, .cons x xs => uncurry (ss := ss) (f x) xs

/-- Curry a reference function that consumes a pack of references. -/
def curry {Ref : Shape → Type} {β : Type} : {ss : List Shape} →
    (Pack Ref ss → β) → CurriedRef Ref ss β
  | [], f => f .nil
  | _s :: ss, f => fun x => curry (ss := ss) (fun xs => f (.cons x xs))

/-- Uncurrying the curried form of a function on packs of references gives the function back. -/
@[simp] theorem uncurry_curry {Ref : Shape → Type} {β : Type} {ss : List Shape}
    (f : Pack Ref ss → β) (xs : Pack Ref ss) :
    uncurry (curry f) xs = f xs := by
  induction xs with
  | nil => rfl
  | cons x xs ih => exact ih (fun ys => f (.cons x ys))

end CurriedRef

end TorchLean
