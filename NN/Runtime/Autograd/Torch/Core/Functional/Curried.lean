/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Tensor.Pack

/-!
# Curried Tensor and Reference Arguments

Convert between shape-indexed packs and curried arguments for tensors or any reference family.
-/

@[expose] public section

namespace Runtime.Autograd.Torch

open Spec TorchLean TorchLean.Tensor

namespace Curried

/--
Type of a curried function accepting one tensor argument per shape in `ss`.

For example, `Fn α [s₁, s₂] β` is `Tensor α s₁ → Tensor α s₂ → β`.
-/
def Fn (α : Type) [Storage α] : List Shape → Type → Type
  | [], β => β
  | s :: ss, β => Tensor α s → Fn α ss β

/-- Convert a function on tensor-pack inputs into its curried form. -/
def curry {α : Type} [Storage α] {β : Type} : {ss : List Shape} →
    (TensorPack α ss → β) → Fn α ss β
  | [], f => f .nil
  | _s :: ss, f => fun x => curry (ss := ss) (fun xs => f (.cons x xs))

/-- Convert a curried function into a function on tensor-pack inputs. -/
def uncurry {α : Type} [Storage α] {β : Type} : {ss : List Shape} →
    Fn α ss β → TensorPack α ss → β
  | [], f, .nil => f
  | _s :: ss, f, .cons x xs => uncurry (ss := ss) (f x) xs

end Curried

/-- A pack of references indexed by their shapes, analogous to `TensorPack`. -/
inductive RefList (Ref : Shape → Type) : List Shape → Type where
  | nil : RefList Ref []
  | cons {s : Shape} {ss : List Shape} : Ref s → RefList Ref ss → RefList Ref (s :: ss)

namespace RefList

/-- Append two `RefList`s. -/
def append {Ref : Shape → Type} : {ss₁ ss₂ : List Shape} →
    RefList Ref ss₁ → RefList Ref ss₂ → RefList Ref (ss₁ ++ ss₂)
  | [], _ss₂, .nil, ys => ys
  | _s :: ss₁, ss₂, .cons x xs, ys => .cons x (append (ss₁ := ss₁) (ss₂ := ss₂) xs ys)

/-- Split a `RefList Ref (ss₁ ++ ss₂)` into its left and right parts. -/
def split {Ref : Shape → Type} : {ss₁ ss₂ : List Shape} →
    RefList Ref (ss₁ ++ ss₂) → RefList Ref ss₁ × RefList Ref ss₂
  | [], _ss₂, xs => (.nil, xs)
  | _s :: ss₁, ss₂, .cons x xs =>
      let (left, right) := split (Ref := Ref) (ss₁ := ss₁) (ss₂ := ss₂) xs
      (.cons x left, right)

/-- Split a `RefList Ref (ss ++ [τ])` into its prefix and last element. -/
def splitLast {Ref : Shape → Type} : {ss : List Shape} → {τ : Shape} →
    RefList Ref (ss ++ [τ]) → RefList Ref ss × Ref τ
  | [], _τ, .cons x .nil => (.nil, x)
  | _s :: ss, τ, .cons x xs =>
      let (initial, last) := splitLast (Ref := Ref) (ss := ss) (τ := τ) xs
      (.cons x initial, last)

end RefList

/-- A curried function accepting one `Ref s` argument per shape in `ss`. -/
def CurriedRef (Ref : Shape → Type) : List Shape → Type → Type
  | [], β => β
  | s :: ss, β => Ref s → CurriedRef Ref ss β

namespace CurriedRef

/-- Uncurry a curried reference function to accept a `RefList`. -/
def uncurry {Ref : Shape → Type} {β : Type} : {ss : List Shape} →
    CurriedRef Ref ss β → RefList Ref ss → β
  | [], f, .nil => f
  | _s :: ss, f, .cons x xs => uncurry (ss := ss) (f x) xs

/-- Curry a reference function that consumes a `RefList`. -/
def curry {Ref : Shape → Type} {β : Type} : {ss : List Shape} →
    (RefList Ref ss → β) → CurriedRef Ref ss β
  | [], f => f .nil
  | _s :: ss, f => fun x => curry (ss := ss) (fun xs => f (.cons x xs))

/-- Apply a tensor-valued `CurriedRef` to its shape-indexed tensor pack. -/
def uncurryPack {α β : Type} [Storage α] : {ss : List Shape} →
    CurriedRef (fun s => Tensor α s) ss β → TensorPack α ss → β
  | [], f, .nil => f
  | _s :: ss, f, .cons x xs => uncurryPack (ss := ss) (f x) xs

end CurriedRef

end Runtime.Autograd.Torch
