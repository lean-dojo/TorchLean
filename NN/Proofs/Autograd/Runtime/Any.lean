/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Tape.Algebra.Soundness
public import NN.Runtime.Autograd.Engine.Core

/-!
# Any

Bridge lemmas between the proved typed context (`TList`) and the runtime tape representation
(`Runtime.AnyTensor` stored in `Array`s).

This is part of the explicit trust boundary: the runtime engine operates on erased tensors, while
the proved model keeps shapes at the type level.
-/

@[expose] public section

namespace Proofs
namespace Autograd
namespace Algebra

open Spec
open Tensor

namespace TList

variable {α : Type}

/--
Convert a typed context (`TList`) into a runtime list of erased tensors (`AnyTensor`).

This is part of the “trust boundary” between the proved model (typed shapes) and the runtime engine
(erased shape/value pairs).
-/
def toAnyList : {ss : List Shape} → TList α ss → List (Runtime.AnyTensor α)
  | [], .nil => []
  | _ :: ss, .cons t ts => Runtime.Autograd.AnyTensor.mk t :: toAnyList (ss := ss) ts

/-- `toAnyList` but as an `Array` (the container used by the runtime tape). -/
def toAnyArray {ss : List Shape} (ts : TList α ss) : Array (Runtime.AnyTensor α) :=
  (toAnyList (α := α) (ss := ss) ts).toArray

/--
Like `toAnyList`, but tags each element with the absolute runtime node id it should correspond to,
starting from `start`.
-/
def toIndexedAnyList : {ss : List Shape} → TList α ss → Nat → List (Nat × Runtime.AnyTensor α)
  | [], .nil, _ => []
  | _ :: ss, .cons t ts, i => (i, Runtime.Autograd.AnyTensor.mk t) :: toIndexedAnyList (ss := ss) ts
    (i + 1)

/--
Every `(pid, tensor)` produced by `toIndexedAnyList ts start` satisfies `pid < start + ss.length`.

This is bookkeeping used to prove runtime backward only references earlier nodes.
-/
theorem mem_toIndexedAnyList_lt :
    {ss : List Shape} → (ts : TList α ss) → (start : Nat) →
      ∀ {pid : Nat} {pg : Runtime.AnyTensor α},
        (pid, pg) ∈ toIndexedAnyList (α := α) (ss := ss) ts start → pid < start + ss.length
  | [], .nil, start, pid, pg, hmem => by
      cases hmem
  | _ :: ss, .cons _t ts, start, pid, pg, hmem => by
      simp [toIndexedAnyList] at hmem
      cases hmem with
      | inl h =>
          -- head element: `pid = start`
          have hpid : pid = start := h.1
          cases hpid
          -- After rewriting `(_ :: ss).length`, this is `start < start + Nat.succ ss.length`.
          simp
      | inr h =>
          have := mem_toIndexedAnyList_lt (ss := ss) ts (start + 1) (pid := pid) (pg := pg) h
          -- `start + 1 + ss.length = start + (ss.length + 1)`
          simpa [Nat.add_assoc, Nat.add_left_comm, Nat.add_comm] using this

/--
`toAnyList` ignores type-level casts of the shape list.

This is a convenience lemma for rewriting across reassociation/cast steps in proofs.
-/
@[simp] theorem toAnyList_cast {ss₁ ss₂ : List Shape} (h : ss₁ = ss₂) (xs : TList α ss₁) :
    toAnyList (α := α) (ss := ss₂) (TList.cast (α := α) h xs) = toAnyList (α := α) (ss := ss₁) xs :=
      by
  cases h
  simp [TList.cast]

/--
`toAnyArray` ignores type-level casts of the shape list.

This is a direct corollary of `toAnyList_cast`.
-/
@[simp] theorem toAnyArray_cast {ss₁ ss₂ : List Shape} (h : ss₁ = ss₂) (xs : TList α ss₁) :
    toAnyArray (α := α) (ss := ss₂) (TList.cast (α := α) h xs) = toAnyArray (α := α) (ss := ss₁) xs
      := by
  cases h
  simp [toAnyArray]

/-- `toAnyList` has the same length as the underlying shape list. -/
@[simp] theorem length_toAnyList :
    {ss : List Shape} → (xs : TList α ss) → (toAnyList (α := α) (ss := ss) xs).length = ss.length
  | [], .nil => by simp [toAnyList]
  | _ :: ss, .cons _t ts => by
      simpa [toAnyList] using (congrArg Nat.succ (length_toAnyList (ss := ss) ts))

/-- `toAnyArray` has the same size as the underlying shape list. -/
@[simp] theorem size_toAnyArray :
    {ss : List Shape} → (xs : TList α ss) → (toAnyArray (α := α) (ss := ss) xs).size = ss.length
  | ss, xs => by
      simp [toAnyArray, List.size_toArray, length_toAnyList (ss := ss) xs]

-- Tell `grind` about the most common cast/length normalization lemmas for these conversions.
attribute [grind =] toAnyList_cast toAnyArray_cast length_toAnyList size_toAnyArray

/-!
### Shape-erasing conversions

The lemmas in this section show that these conversions preserve length/order and interact well with
`TList.snoc` and `TList.get`. They’re used later to relate runtime node ids to positions in the
typed proof context `Γ ++ ss`.
-/

/--
`toAnyList` commutes with appending a value: converting `snoc xs t` is `toAnyList xs ++ [t]`.
-/
theorem toAnyList_snoc {ss : List Shape} {τ : Shape} :
    (xs : TList α ss) → (t : Tensor α τ) →
      toAnyList (α := α) (ss := ss ++ [τ]) (TList.snoc (α := α) (ss := ss) (τ := τ) xs t) =
        toAnyList (α := α) (ss := ss) xs ++ [Runtime.Autograd.AnyTensor.mk t] := by
  intro xs t
  induction ss with
  | nil =>
      cases xs
      simp [TList.snoc, toAnyList]
  | cons s ss ih =>
      cases xs with
      | cons x xs =>
          simp [TList.snoc, toAnyList, ih (xs := xs)]

/-- Array-form version of `toAnyList_snoc`. -/
@[simp] theorem toAnyArray_snoc {ss : List Shape} {τ : Shape} (xs : TList α ss) (t : Tensor α τ) :
    toAnyArray (α := α) (ss := ss ++ [τ]) (TList.snoc (α := α) (ss := ss) (τ := τ) xs t) =
      (toAnyArray (α := α) (ss := ss) xs).push (Runtime.Autograd.AnyTensor.mk t) := by
  simp [toAnyArray, toAnyList_snoc (α := α) (ss := ss) (τ := τ) xs t]

/-- `toAnyArray` of a cons context is array cons/append of the head element. -/
theorem toAnyArray_cons {α : Type} {s : Shape} {ss : List Shape} (x : Tensor α s) (xs : TList α ss)
  :
    toAnyArray (α := α) (ss := s :: ss) (TList.cons x xs) =
      #[Runtime.Autograd.AnyTensor.mk x] ++ toAnyArray (α := α) (ss := ss) xs := by
  simp [toAnyArray, toAnyList]

attribute [grind =] toAnyList_snoc toAnyArray_snoc toAnyArray_cons

/--
Array lookup through `toAnyArray` corresponds to `TList.get` (up to the `AnyTensor` wrapper).

This is the key lemma that lets us connect runtime indexing (`arr[i]`) to proof indexing (`get xs
  i`).
-/
theorem get_toAnyArray {α : Type} :
    {ss : List Shape} → (xs : TList α ss) → (i : Fin ss.length) →
      let arr := toAnyArray (α := α) (ss := ss) xs
      arr[i.1]'(by
        dsimp [arr]
        exact Nat.lt_of_lt_of_eq i.2 (size_toAnyArray (α := α) (ss := ss) xs).symm) =
        Runtime.Autograd.AnyTensor.mk (TList.get (α := α) (ss := ss) xs i)
  | [], .nil, i => by
      cases i with
      | mk val isLt =>
        exact False.elim ((Nat.not_lt_zero val) isLt)
  | _ :: ss, .cons x xs, ⟨0, hi⟩ => by
      -- `0 : Fin (Nat.succ ss.length)` is elaborated with a canonical proof,
      -- which is not definitionally equal to `hi`. Normalize the index so that
      -- `TList.get` reduces by `rfl`.
      have h0 : (0 : Fin (Nat.succ ss.length)) = ⟨0, hi⟩ := by
        ext
        rfl
      cases h0
      simp [toAnyArray, toAnyList]
      rfl
  | _ :: ss, .cons x xs, ⟨Nat.succ i, hi⟩ => by
      have : i < ss.length := Nat.lt_of_succ_lt_succ hi
      simpa [toAnyArray, toAnyList, TList.get] using
        (get_toAnyArray (α := α) (ss := ss) xs ⟨i, this⟩)

end TList

end Algebra
end Autograd
end Proofs
