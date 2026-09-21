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

/-- Splitting concatenated references recovers the original state blocks. -/
@[simp] theorem split_append {Ref : Shape → Type} {ss₁ ss₂ : List Shape}
    (xs : RefList Ref ss₁) (ys : RefList Ref ss₂) :
    split (ss₁ := ss₁) (ss₂ := ss₂) (append xs ys) = (xs, ys) := by
  induction xs with
  | nil => rfl
  | cons x xs ih => simp only [append, split, ih]

/-- The final reference remains separate from the preceding model state. -/
@[simp] theorem splitLast_append {Ref : Shape → Type} {ss : List Shape} {τ : Shape}
    (xs : RefList Ref ss) (x : Ref τ) :
    splitLast (ss := ss) (append xs (.cons x .nil)) = (xs, x) := by
  induction xs with
  | nil => rfl
  | cons y ys ih => simp only [append, splitLast, ih]

/-- Reassembling the two state blocks restores every original reference. -/
@[simp] theorem append_split {Ref : Shape → Type} {ss₁ ss₂ : List Shape}
    (xs : RefList Ref (ss₁ ++ ss₂)) :
    append (split (ss₁ := ss₁) xs).1 (split (ss₁ := ss₁) xs).2 = xs := by
  induction ss₁ with
  | nil => rfl
  | cons s ss ih =>
      cases xs with
      | cons x xs => simp only [split, append, ih]

private theorem cast_cons {Ref : Shape → Type} {s : Shape} {ss ts : List Shape}
    (h : ss = ts) (x : Ref s) (xs : RefList Ref ss) :
    (congrArg (s :: ·) h ▸ RefList.cons x xs) = RefList.cons x (h ▸ xs) := by
  cases h
  rfl

/-- Reassociating state blocks changes only their shape-list witness, not their references. -/
theorem append_assoc {Ref : Shape → Type} {a b c : List Shape}
    (xs : RefList Ref a) (ys : RefList Ref b) (zs : RefList Ref c) :
    (List.append_assoc a b c ▸ append (append xs ys) zs) = append xs (append ys zs) := by
  induction xs with
  | nil => rfl
  | @cons s ss x xs ih =>
      simp only [append]
      rw [cast_cons (List.append_assoc ss b c), ih]

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

/-- Binding the curried arguments preserves the complete reference-list computation. -/
@[simp] theorem uncurry_curry {Ref : Shape → Type} {β : Type} {ss : List Shape}
    (f : RefList Ref ss → β) (xs : RefList Ref ss) :
    uncurry (curry f) xs = f xs := by
  induction xs with
  | nil => rfl
  | cons x xs ih => exact ih (fun ys => f (.cons x ys))

/-- Apply a tensor-valued `CurriedRef` to its shape-indexed tensor pack. -/
def uncurryPack {α β : Type} [Storage α] : {ss : List Shape} →
    CurriedRef (fun s => Tensor α s) ss β → TensorPack α ss → β
  | [], f, .nil => f
  | _s :: ss, f, .cons x xs => uncurryPack (ss := ss) (f x) xs

end CurriedRef

end Runtime.Autograd.Torch
