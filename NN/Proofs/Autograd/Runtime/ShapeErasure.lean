/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Tensor.ShapeErasure
public import Batteries.Lean.Except
public import Mathlib.Algebra.GroupWithZero.Nat

/-!
# Shape Erasure

Bridge lemmas between the proved typed context (`TensorPack`) and the runtime tape representation
(`Spec.SomeTensor` stored in `Array`s).

The conversion preserves order and stores each erased shape in the resulting `SomeTensor`.
-/

@[expose] public section

open Spec TorchLean
open TorchLean TorchLean.Tensor

namespace TorchLean.TensorPack

variable {α : Type} [Storage α]

/--
Every `(pid, tensor)` produced by `toIndexedShapeErasedArray ts start` satisfies
`pid < start + ss.length`.

This is bookkeeping used to prove runtime backward only references earlier nodes.
-/
theorem mem_toIndexedShapeErasedArray_lt :
    {ss : List Shape} → (ts : TensorPack α ss) → (start : Nat) →
      ∀ {pid : Nat} {pg : Spec.SomeTensor α},
        (pid, pg) ∈ toIndexedShapeErasedArray (α := α) (ss := ss) ts start → pid < start + ss.length
  | [], .nil, start, pid, pg, hmem => by
      simp [toIndexedShapeErasedArray] at hmem
  | _ :: ss, .cons _t ts, start, pid, pg, hmem => by
      simp [toIndexedShapeErasedArray] at hmem
      cases hmem with
      | inl h =>
          -- head element: `pid = start`
          have hpid : pid = start := h.1
          cases hpid
          -- After rewriting `(_ :: ss).length`, this is `start < start + Nat.succ ss.length`.
          simp
      | inr h =>
          have := mem_toIndexedShapeErasedArray_lt (ss := ss) ts (start + 1) (pid := pid)
            (pg := pg) h
          -- `start + 1 + ss.length = start + (ss.length + 1)`
          simpa [Nat.add_assoc, Nat.add_left_comm, Nat.add_comm] using this

/--
`toShapeErasedArray` ignores type-level casts of the shape list.
-/
@[simp] theorem toShapeErasedArray_cast {ss₁ ss₂ : List Shape} (h : ss₁ = ss₂)
    (xs : TensorPack α ss₁) :
    toShapeErasedArray (α := α) (ss := ss₂) (TorchLean.TensorPack.cast (α := α) h xs) =
      toShapeErasedArray (α := α) (ss := ss₁) xs := by
  cases h
  simp

/-- `toShapeErasedArray` has the same size as the underlying shape list. -/
@[simp] theorem size_toShapeErasedArray :
    {ss : List Shape} → (xs : TensorPack α ss) →
      (toShapeErasedArray (α := α) (ss := ss) xs).size = ss.length
  | [], .nil => by simp [toShapeErasedArray]
  | _ :: ss, .cons _t ts => by
      simp [toShapeErasedArray, size_toShapeErasedArray (ss := ss) ts, Nat.add_comm]

-- Tell `grind` about the most common cast/length normalization lemmas for these conversions.
attribute [grind =] toShapeErasedArray_cast size_toShapeErasedArray

/-- Recovering a typed pack after an untouched array prefix and shape erasure is lossless. -/
@[simp] theorem ofShapeErasedArray_append_toShapeErasedArray
    : (pref : Array (Spec.SomeTensor α)) → {ss : List Shape} → (xs : TensorPack α ss) →
      ofShapeErasedArray (α := α) (pref ++ toShapeErasedArray (α := α) xs) pref.size =
        .ok xs
  | _, [], .nil => by
      rfl
  | pref, _ :: ss, .cons x xs => by
      have hhead :
          ((pref ++ #[Spec.SomeTensor.ofTensor x]) ++
              toShapeErasedArray (α := α) xs)[pref.size]? =
            some (Spec.SomeTensor.ofTensor x) := by
        rw [Array.getElem?_append_left (by simp)]
        simp
      have ih := ofShapeErasedArray_append_toShapeErasedArray
        (pref ++ #[Spec.SomeTensor.ofTensor x]) (ss := ss) xs
      simp only [toShapeErasedArray, ← Array.append_assoc]
      simp [Array.append_singleton] at ih
      rw [ofShapeErasedArray, hhead]
      simp [ih]

/-- Recovering the expected typed pack immediately after shape erasure is lossless. -/
@[simp] theorem ofShapeErasedArray_toShapeErasedArray
    {ss : List Shape} (xs : TensorPack α ss) :
    ofShapeErasedArray (α := α) (toShapeErasedArray (α := α) xs) = .ok xs := by
  simpa using
    (ofShapeErasedArray_append_toShapeErasedArray (α := α) (pref := #[]) (ss := ss) xs)

/-!
### Shape-erasing conversions

The lemmas below show that these conversions preserve length and order and interact well with
`TorchLean.TensorPack.snoc` and `TorchLean.TensorPack.get`. They are used later to relate runtime
node ids to positions in the typed proof context `Γ ++ ss`.
-/

/--
`toShapeErasedArray` commutes with appending a value.
-/
@[simp] theorem toShapeErasedArray_snoc {ss : List Shape} {τ : Shape} (xs : TensorPack α ss)
    (t : Tensor α τ) :
    toShapeErasedArray (α := α) (ss := ss ++ [τ])
        (TorchLean.TensorPack.snoc (α := α) (ss := ss) (τ := τ) xs t) =
      (toShapeErasedArray (α := α) (ss := ss) xs).push (Spec.SomeTensor.ofTensor t) := by
  induction ss with
  | nil =>
      cases xs
      simp [TorchLean.TensorPack.snoc, toShapeErasedArray]
  | cons s ss ih =>
      cases xs with
      | cons x xs =>
          simp [TorchLean.TensorPack.snoc, toShapeErasedArray, ih (xs := xs)]

/-- `toShapeErasedArray` of a cons context is array cons/append of the head element. -/
theorem toShapeErasedArray_cons {α : Type} [Storage α] {s : Shape} {ss : List Shape}
    (x : Tensor α s) (xs : TensorPack α ss)
  :
    toShapeErasedArray (α := α) (ss := s :: ss) (TensorPack.cons x xs) =
      #[Spec.SomeTensor.ofTensor x] ++ toShapeErasedArray (α := α) (ss := ss) xs := by
  rfl

attribute [grind =] toShapeErasedArray_snoc toShapeErasedArray_cons

/--
Array lookup through `toShapeErasedArray` corresponds to `TorchLean.TensorPack.get` after
erasing the result's shape from its type.

This is the key lemma that lets us connect runtime indexing (`arr[i]`) to proof indexing (`get xs
  i`).
-/
theorem get_toShapeErasedArray {α : Type} [Storage α] :
    {ss : List Shape} → (xs : TensorPack α ss) → (i : Fin ss.length) →
      let arr := toShapeErasedArray (α := α) (ss := ss) xs
      arr[i.1]'(by
        dsimp [arr]
        exact Nat.lt_of_lt_of_eq i.2 (size_toShapeErasedArray (α := α) (ss := ss) xs).symm) =
        Spec.SomeTensor.ofTensor (TorchLean.TensorPack.get (α := α) (ss := ss) xs i)
  | [], .nil, i => by
      cases i with
      | mk val isLt =>
        exact False.elim ((Nat.not_lt_zero val) isLt)
  | _ :: ss, .cons x xs, ⟨0, hi⟩ => by
      -- `0 : Fin (Nat.succ ss.length)` is elaborated with a canonical proof,
      -- which is not definitionally equal to `hi`. Normalize the index so that
      -- `TorchLean.TensorPack.get` reduces by `rfl`.
      have h0 : (0 : Fin (Nat.succ ss.length)) = ⟨0, hi⟩ := by
        ext
        rfl
      cases h0
      simp [toShapeErasedArray]
      rfl
  | _ :: ss, .cons x xs, ⟨Nat.succ i, hi⟩ => by
      have : i < ss.length := Nat.lt_of_succ_lt_succ hi
      simpa [toShapeErasedArray, TorchLean.TensorPack.get] using
        (get_toShapeErasedArray (α := α) (ss := ss) xs ⟨i, this⟩)

end TorchLean.TensorPack
