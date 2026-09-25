/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Proofs.DirectedIBPBasic
public import NN.Proofs.Tensor.Basic.Core

/-!
# Tensor coordinates for normalization transfers

The graph stores flat coordinate functions. These lemmas pass between those functions and the
typed tensors used by the real normalization specifications.
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Graph.DirectedBackward

open Spec TorchLean TorchLean.Tensor

noncomputable section

/-- Read a graph value in the row-major coordinates of a specified tensor shape. -/
def tensorOfFlatValues (s : Shape) (f : Nat → ℝ) : Tensor ℝ s :=
  TorchLean.Tensor.Internal.Rep.ofFn fun c => f (Shape.Coord.linearize c).val

@[simp] theorem tensorOfFlatValues_apply (s : Shape) (f : Nat → ℝ) (c : s.Coord) :
    tensorOfFlatValues s f c = f (Shape.Coord.linearize c).val := by
  simp only [tensorOfFlatValues, TorchLean.Tensor.Internal.Rep.get_ofFn]

variable {α : Type} [Storage α] [Context α] [BoundOps α] [LawfulBoundOps α]

local notation "value" => LawfulBoundOps.toReal (α := α)

/-- A flattened box encloses a flat graph value exactly when it does so at every tensor
coordinate. -/
theorem rowEncloses_flatten_iff {s : Shape} {lo hi : Tensor α s} {f : Nat → ℝ} :
    RowEncloses { dim := s.size, lo := lo.flattenSpec, hi := hi.flattenSpec } s.size f ↔
      ∀ c : s.Coord, value (lo c) ≤ f (Shape.Coord.linearize c).val ∧
        f (Shape.Coord.linearize c).val ≤ value (hi c) := by
  rw [rowEncloses_iff]
  constructor
  · intro h c
    simpa only [Spec.getScalar_flattenSpec_linearize] using
      h (Shape.Coord.linearize c)
  · intro h i
    have hi := h (Shape.Coord.unlinearize i)
    rw [Shape.Coord.linearize_unlinearize] at hi
    simpa only [← Spec.getScalar_flattenSpec_linearize,
      Shape.Coord.linearize_unlinearize] using hi

omit [Context α] [BoundOps α] [LawfulBoundOps α] in
/-- The checked unflattening operation reads the same flat storage coordinate. -/
theorem ibpUnflatten_apply {s : Shape} {d : Nat} (t : Tensor α [d]) (h : d = s.size)
    (c : s.Coord) :
    ibpUnflatten d t h c = t.getScalar (Fin.cast h.symm (Shape.Coord.linearize c)) := by
  subst d
  change (Tensor.unflattenSpec s t) c = t.getScalar (Shape.Coord.linearize c)
  rw [← Spec.getScalar_flattenSpec_linearize, Tensor.flattenSpec_unflattenSpec]

omit [Context α] [BoundOps α] [LawfulBoundOps α] in
/-- A shape change preserves a constant tensor's fill value. -/
theorem normalization_reshapeSpec_full {s t : Shape} (a : α) (h : s.size = t.size) :
    Tensor.reshapeSpec (Tensor.full s a) h = Tensor.full t a := by
  apply TorchLean.Tensor.Internal.Rep.ext
  intro c
  simp only [Tensor.reshapeSpec, TorchLean.Tensor.Internal.Rep.reshape_apply_coordEquiv,
    Tensor.full_apply]

/-- A flat parent enclosure supplies the coordinates of every checked tensor view. -/
theorem rowEncloses_unflatten {s : Shape} {B : FlatBox α} {f : Nat → ℝ}
    (h : RowEncloses B s.size f) (c : s.Coord) :
    value (ibpUnflatten B.dim B.lo h.1 c) ≤ tensorOfFlatValues s f c ∧
      tensorOfFlatValues s f c ≤ value (ibpUnflatten B.dim B.hi h.1 c) := by
  obtain ⟨d, lo, hi⟩ := B
  have hd := h.1
  change d = s.size at hd
  subst d
  rw [rowEncloses_iff] at h
  simpa only [ibpUnflatten_apply, Fin.cast_refl, id_eq, tensorOfFlatValues_apply] using
    h (Shape.Coord.linearize c)

end

end NN.MLTheory.CROWN.Graph.DirectedBackward
