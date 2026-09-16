/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Tensor.Internal.Representation.Fiber.Axis

/-!
# Finite-fiber values

Finite fibers can be summed or collected as multisets. The transport theorems
here compare fibers under equivalences and composition independently of tensor
syntax.
-/

@[expose] public section

namespace TorchLean.Tensor.Internal

open scoped BigOperators

universe u v
namespace Fiber

/-- Every fiber of an equivalence has a unique element. -/
instance equivUnique {ι : Type u} {κ : Type v} (e : ι ≃ κ) (y : κ) :
    Unique (Fiber e y) where
  default := ⟨e.symm y, e.apply_symm_apply y⟩
  uniq x := by
    apply Subtype.ext
    apply e.injective
    exact x.property.trans (e.apply_symm_apply y).symm

/-- The fiber of an equivalence has cardinality one. -/
theorem card_equiv {ι : Type u} {κ : Type v} [Fintype ι]
    [DecidableEq κ] (e : ι ≃ κ) (y : κ) :
    Fintype.card (Fiber e y) = 1 :=
  Fintype.card_unique

/-- Add all values indexed by one finite fiber. -/
def sum {ι : Type u} {κ : Type v} {α : Type*} [Fintype ι] [DecidableEq κ]
    [AddCommMonoid α] (f : ι → κ) (y : κ) (x : ι → α) : α :=
  ∑ i : Fiber f y, x i.1

/--
The multiset of input values in one finite fiber.

Using a multiset records multiplicity while deliberately forgetting
enumeration order. A reduction may therefore change scalar type and need not
be presented as a binary operation with an identity.
-/
def values {ι : Type u} {κ : Type v} {α : Type*}
    [Fintype ι] [DecidableEq κ]
    (f : ι → κ) (outputCoordinate : κ) (input : ι → α) : Multiset α :=
  (Finset.univ : Finset (Fiber f outputCoordinate)).val.map fun inputCoordinate =>
    input inputCoordinate.1

/-- A fiber's value multiset has one entry for every coordinate in the fiber. -/
@[simp] theorem values_card {ι : Type u} {κ : Type v} {α : Type*}
    [Fintype ι] [DecidableEq κ]
    (f : ι → κ) (outputCoordinate : κ) (input : ι → α) :
    (values f outputCoordinate input).card =
      Fintype.card (Fiber f outputCoordinate) := by
  simp [values]

/-- Summing a fiber's value multiset is the corresponding finite coordinate sum. -/
@[simp] theorem values_sum {ι : Type u} {κ : Type v} {R : Type*}
    [Fintype ι] [DecidableEq κ] [AddCommMonoid R]
    (f : ι → κ) (outputCoordinate : κ) (input : ι → R) :
    (values f outputCoordinate input).sum =
      ∑ inputCoordinate : Fiber f outputCoordinate,
        input inputCoordinate.1 := by
  simp [values]

/-- Multiplying a fiber's value multiset is the corresponding finite coordinate product. -/
@[simp] theorem values_prod {ι : Type u} {κ : Type v} {R : Type*}
    [Fintype ι] [DecidableEq κ] [CommMonoid R]
    (f : ι → κ) (outputCoordinate : κ) (input : ι → R) :
    (values f outputCoordinate input).prod =
      ∏ inputCoordinate : Fiber f outputCoordinate,
        input inputCoordinate.1 := by
  simp [values]

/--
Changing the domain of a fiber by an equivalence preserves its multiset of
values.
-/
theorem values_comp_equiv
    {ι : Type u} {κ : Type v} {τ : Type*} {α : Type*}
    [Fintype ι] [Fintype κ] [DecidableEq τ]
    (e : ι ≃ κ) (f : ι → τ) (outputCoordinate : τ) (input : κ → α) :
    values f outputCoordinate (input ∘ e) =
      values (f ∘ e.symm) outputCoordinate input := by
  let fiberEquiv :
      Fiber f outputCoordinate ≃ Fiber (f ∘ e.symm) outputCoordinate :=
    { toFun := fun inputCoordinate =>
        ⟨e inputCoordinate.1, by
          change f (e.symm (e inputCoordinate.1)) = outputCoordinate
          simpa using inputCoordinate.2⟩
      invFun := fun inputCoordinate =>
        ⟨e.symm inputCoordinate.1, inputCoordinate.2⟩
      left_inv := fun inputCoordinate => by
        apply Subtype.ext
        exact e.symm_apply_apply inputCoordinate.1
      right_inv := fun inputCoordinate => by
        apply Subtype.ext
        exact e.apply_symm_apply inputCoordinate.1 }
  unfold values
  calc
    (Finset.univ : Finset (Fiber f outputCoordinate)).val.map
          (fun inputCoordinate => input (e inputCoordinate.1)) =
        (Finset.univ : Finset (Fiber f outputCoordinate)).val.map
          ((fun inputCoordinate => input inputCoordinate.1) ∘ fiberEquiv) := by
      rfl
    _ =
        ((Finset.univ : Finset (Fiber f outputCoordinate)).val.map
          fiberEquiv).map (fun inputCoordinate => input inputCoordinate.1) := by
      rw [Multiset.map_map]
    _ =
        (Finset.univ :
          Finset (Fiber (f ∘ e.symm) outputCoordinate)).val.map
            (fun inputCoordinate => input inputCoordinate.1) := by
      rw [Multiset.map_univ_val_equiv]

/--
Transporting the target of a coordinate map by an equivalence preserves the
corresponding fiber values.
-/
theorem values_target_equiv
    {ι : Type u} {κ : Type v} {τ : Type*} {α : Type*}
    [Fintype ι] [DecidableEq κ] [DecidableEq τ]
    (e : κ ≃ τ) (f : ι → κ) (outputCoordinate : τ) (input : ι → α) :
    values f (e.symm outputCoordinate) input =
      values (e ∘ f) outputCoordinate input := by
  let fiberEquiv :
      Fiber f (e.symm outputCoordinate) ≃ Fiber (e ∘ f) outputCoordinate :=
    { toFun := fun inputCoordinate =>
        ⟨inputCoordinate.1, by
          change e (f inputCoordinate.1) = outputCoordinate
          rw [inputCoordinate.2]
          exact e.apply_symm_apply outputCoordinate⟩
      invFun := fun inputCoordinate =>
        ⟨inputCoordinate.1, by
          apply e.injective
          simpa using inputCoordinate.2⟩
      left_inv := fun inputCoordinate => by
        apply Subtype.ext
        rfl
      right_inv := fun inputCoordinate => by
        apply Subtype.ext
        rfl }
  unfold values
  calc
    (Finset.univ :
        Finset (Fiber f (e.symm outputCoordinate))).val.map
          (fun inputCoordinate => input inputCoordinate.1) =
        (Finset.univ :
          Finset (Fiber f (e.symm outputCoordinate))).val.map
            ((fun inputCoordinate => input inputCoordinate.1) ∘ fiberEquiv) := by
      rfl
    _ =
        ((Finset.univ :
          Finset (Fiber f (e.symm outputCoordinate))).val.map
            fiberEquiv).map (fun inputCoordinate => input inputCoordinate.1) := by
      rw [Multiset.map_map]
    _ =
        (Finset.univ :
          Finset (Fiber (e ∘ f) outputCoordinate)).val.map
            (fun inputCoordinate => input inputCoordinate.1) := by
      rw [Multiset.map_univ_val_equiv]

/-- Renaming a fiber's input coordinates preserves its cardinality. -/
theorem card_comp_equiv
    {ι : Type u} {κ : Type v} {τ : Type*}
    [Fintype ι] [Fintype κ] [DecidableEq τ]
    (e : ι ≃ κ) (f : ι → τ) (outputCoordinate : τ) :
    Fintype.card (Fiber f outputCoordinate) =
      Fintype.card (Fiber (f ∘ e.symm) outputCoordinate) := by
  have hValues :=
    congrArg Multiset.card
      (values_comp_equiv e f outputCoordinate fun _ => ())
  simpa only [values_card] using hValues

/-- Transporting a fiber's target by an equivalence preserves its cardinality. -/
theorem card_target_equiv
    {ι : Type u} {κ : Type v} {τ : Type*}
    [Fintype ι] [DecidableEq κ] [DecidableEq τ]
    (e : κ ≃ τ) (f : ι → κ) (outputCoordinate : τ) :
    Fintype.card (Fiber f (e.symm outputCoordinate)) =
      Fintype.card (Fiber (e ∘ f) outputCoordinate) := by
  have hValues :=
    congrArg Multiset.card
      (values_target_equiv e f outputCoordinate fun _ => ())
  simpa only [values_card] using hValues

end Fiber

end TorchLean.Tensor.Internal
