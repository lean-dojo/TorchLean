/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.CROWN.Conv.Affine
public import TorchLean.Verify.CROWN.BoundOps.Lawful

/-!
# Real box facts for the convolution enclosure

A real box contains a tensor exactly when it bounds every coordinate (`Box.contains_iff_apply`),
and exactly when its flattening contains the flattened tensor (`Box.contains_flattenBox_iff`).
`contains_getAtOrZero` reads the bounds at a list index, and `foldlIndices_rel` carries a relation
through the spatial iterator `foldlIndices`. They are the steps of `ibpConv_sound`, the enclosure
of the grouped convolution by the directed convolution transfer `ibpConv` at a lawful scalar, and
of its flat-box form `Directed.convBox?_encloses`.
-/

public section

namespace TorchLean.CROWN

open Tensor Conv
open scoped BigOperators
open Shape (size_toList_flatten)

/-- A box contains a tensor exactly when every coordinate of the tensor lies between the bounds. -/
theorem Box.contains_iff_apply {s : Shape} (box : Box ℝ s) (input : Tensor ℝ s) :
    Box.Contains box input ↔
      ∀ coordinate : s.Coord,
        box.lo coordinate ≤ input coordinate ∧ input coordinate ≤ box.hi coordinate := by
  induction s with
  | scalar =>
      constructor
      · intro h coordinate
        cases coordinate
        exact h
      · intro h
        exact h PUnit.unit
  | dim n rest ih =>
      constructor
      · intro h coordinate
        rcases coordinate with ⟨head, tail⟩
        simpa only [Tensor.unstack_apply] using
          (ih ⟨box.lo.unstack head, box.hi.unstack head⟩
            (input.unstack head)).mp (h head) tail
      · intro h head
        apply (ih ⟨box.lo.unstack head, box.hi.unstack head⟩
          (input.unstack head)).mpr
        intro tail
        simpa only [Tensor.unstack_apply] using h (head, tail)

/-- Flattening preserves and reflects every component inequality of a real tensor box. -/
theorem Box.contains_flattenBox_iff {s : Shape} (box : Box ℝ s) (input : Tensor ℝ s) :
    Box.Contains (flattenBox box) (Tensor.reshape input [_]
        (Shape.size_vector _).symm) ↔ Box.Contains box input := by
  rw [contains_iff_apply, contains_iff_apply]
  constructor
  · intro h coordinate
    have hc := h ((Shape.Coord.reshapeEquiv (size_toList_flatten s)).symm coordinate)
    simpa only [flattenBox, Tensor.reshape, Rep.reshape_apply_reshapeEquiv,
      Equiv.apply_symm_apply] using hc
  · intro h coordinate
    simpa only [flattenBox, Tensor.reshape, Rep.reshape_apply_reshapeEquiv] using
      h (Shape.Coord.reshapeEquiv (size_toList_flatten s) coordinate)

/-- A tensor in a box lies between the bounds at every index list, as `getAtOrZero` reads them. -/
theorem contains_getAtOrZero {dims : List Nat}
    {box : Box ℝ dims} {input : Tensor ℝ dims}
    (h : Box.Contains box input) (indices : List Nat) :
    getAtOrZero box.lo indices ≤ getAtOrZero input indices ∧
      getAtOrZero input indices ≤ getAtOrZero box.hi indices := by
  induction dims generalizing indices with
  | nil =>
      cases indices with
      | nil => simpa only [getAtOrZero_scalar_nil, Box.Contains] using h
      | cons head tail => simp
  | cons n dims ih =>
      cases indices with
      | nil => simp
      | cons head tail =>
          by_cases hhead : head < n
          · simpa only [getAtOrZero_dim_cons, dite_eq_left hhead] using
              ih (h ⟨head, hhead⟩) tail
          · simp only [getAtOrZero_dim_cons, dite_eq_right hhead, le_refl, and_self]

/-- The spatial iterator preserves a binary relation independently of rank and empty extents. -/
theorem foldlIndices_rel {A B : Type} (relation : A → B → Prop) (dims : List Nat)
    {left : A → List Nat → A} {right : B → List Nat → B}
    (step : ∀ a b i, relation a b → relation (left a i) (right b i))
    {a : A} {b : B} (initial : relation a b) :
    relation (foldlIndices dims a left) (foldlIndices dims b right) := by
  induction dims generalizing left right a b with
  | nil => exact step a b [] initial
  | cons n dims ih =>
      apply List.foldl_rel initial
      intro head _ a b h
      exact ih (fun a b tail hab => step a b (head.val :: tail) hab) h

end TorchLean.CROWN
