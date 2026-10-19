/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Proofs.DirectedIBPTensor

/-!
# Tensor coordinates for normalization transfers

The graph stores flat coordinate functions. These lemmas pass between those functions, read as
`realTensor`, and the typed tensors used by the real normalization specifications.
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Graph.DirectedBackward

open Spec TorchLean TorchLean.Tensor

noncomputable section

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

/-- A flat parent enclosure supplies the coordinates of every checked tensor view. This is the
coordinatewise form of `tensorEncloses_ibpUnflatten`. -/
theorem rowEncloses_unflatten {s : Shape} {B : FlatBox α} {f : Nat → ℝ}
    (h : RowEncloses B s.size f) (c : s.Coord) :
    value (ibpUnflatten B.dim B.lo h.1 c) ≤ realTensor s f c ∧
      realTensor s f c ≤ value (ibpUnflatten B.dim B.hi h.1 c) :=
  tensorEncloses_ibpUnflatten h.1 h c

end

end NN.MLTheory.CROWN.Graph.DirectedBackward
