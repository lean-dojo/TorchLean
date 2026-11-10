/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Tensor.Contract
public import TorchLean.Tensor.Layers.Activation

/-!
# `Softmax.maxVec` as a reduction and the channel broadcast

Two layer-level tensor expressions that equal general tensor operators.

* `Softmax.maxVec_eq_reduce` shows that `Softmax.maxVec` is `reduce .max` over the one axis of a
  vector, with the same schedule: the fold starts from the first entry and visits it again. Its
  proof compares the two folds with `foldl_finRange_first_congr`.
* `reshape_broadcast_eq_broadcast` shows that a channel vector repeated along the rows of
  a `[channels, positions]` matrix and reshaped is its broadcast along the first axis.
-/

@[expose] public section

open TorchLean.Shape (rank_eq_length size)

namespace TorchLean

open Tensor

variable {α : Type} [Storage α]

/-- Two folds of `step` over the coordinates of equal shapes, each from its first entry, agree
when their entries agree at coordinates with the same indices. -/
theorem Tensor.foldl_finRange_first_congr {β : Type} (step : β → β → β) {t u : Shape} (h : t = u)
    (f : t.Coord → β) (g : u.Coord → β)
    (hfg : ∀ (c : t.Coord) (c' : u.Coord),
      (∀ k, Shape.Coord.index c k = Shape.Coord.index c' k) → f c = g c')
    (ht : 0 < Shape.size t) (hu : 0 < Shape.size u) :
    (List.finRange (Shape.size t)).foldl
        (fun a i => step a (f (Shape.Coord.unlinearize i)))
        (f (Shape.Coord.unlinearize ⟨0, ht⟩)) =
      (List.finRange (Shape.size u)).foldl
        (fun a i => step a (g (Shape.Coord.unlinearize i)))
        (g (Shape.Coord.unlinearize ⟨0, hu⟩)) := by
  subst h
  have hfg' : f = g := funext fun c => hfg c c fun _ => rfl
  rw [hfg']

/-- `Softmax.maxVec` is the maximum over the one axis of a vector, from its first entry. -/
theorem Softmax.maxVec_eq_reduce [Context α] {n : Nat} (t : Tensor α [n + 1]) :
    maxVec t =
      castShape (reduce .max t [⟨0, by simp [Shape.rank_eq_length]⟩] false
        (fun _ d hd => by
          rw [List.mem_singleton] at hd
          subst hd
          exact Nat.succ_pos n))
        ((Shape.eraseAxes_singleton _ _).trans rfl) := by
  apply eq_castShape_of_index
  intro c c' _
  obtain ⟨⟩ := c
  rw [maxVec, reduce_max_apply]
  simp only [getScalar_eq_apply]
  refine (scalar_apply _).trans ?_
  have hs : 0 < Shape.size [n + 1] := by simp [Shape.size]
  have h0 : t (⟨0, Nat.succ_pos n⟩, PUnit.unit) =
      t ((Shape.Coord.unlinearize (s := [n + 1]) ⟨0, hs⟩).1, PUnit.unit) := by
    congr 2
    apply Fin.ext
    rw [Shape.Coord.unlinearizeFast_eq]
    exact (Nat.zero_div _).symm
  refine (foldl_finRange_size_singleton (n + 1) _ _).trans ?_
  rw [h0]
  refine foldl_finRange_first_congr Context.maximum (t := [n + 1])
    (Shape.selectAxes_singleton [n + 1] ⟨0, by simp [Shape.rank_eq_length]⟩).symm
    (fun r : Shape.Coord [n + 1] => t (r.1, PUnit.unit))
    (fun r => t (Shape.reduceSource [n + 1] [⟨0, by simp [Shape.rank_eq_length]⟩] false c' r))
    ?_ hs _
  intro r r' hrr
  show t _ = t _
  congr 1
  refine Shape.Coord.ext_index (s := [n + 1]) ?_
  intro k hk
  match k, hk with
  | 0, hk =>
    rw [Shape.index_reduceSource, dite_eq_left hk, ite_eq_left (List.mem_singleton_self _),
      Shape.sortedAxes_singleton, List.idxOf_cons_self]
    exact hrr 0
  | k + 1, hk =>
    exact absurd hk (by
      simp only [Shape.rank_eq_length, List.length_cons, List.length_nil]
      omega)

/-- A channel vector repeated along the rows of a `[channels, positions]` matrix and reshaped to
`channels :: sSpatial` is its broadcast along the first axis. -/
theorem Tensor.reshape_broadcast_eq_broadcast {channels : Nat} (sSpatial : Shape)
    (x : Tensor α [channels])
    (h : Shape.size [channels, sSpatial.size] = Shape.size (channels :: sSpatial)) :
    reshape (broadcast x [channels, sSpatial.size] _
        (Shape.eraseIdx_broadcastDims [channels, sSpatial.size] 1)) (channels :: sSpatial) h =
      broadcast x (channels :: sSpatial) _ (Shape.broadcastDims_head channels sSpatial) := by
  apply Rep.ext
  intro o
  rw [reshape_apply]
  erw [broadcast_apply_of_extent_eq _ _ _ (fun i => by
      match i with
      | ⟨0, _⟩ => rfl
      | ⟨k + 1, hk⟩ => exact absurd hk (by simp [Shape.rank_eq_length])),
    broadcast_apply_of_extent_eq _ _ _ (fun i => by
      match i with
      | ⟨0, _⟩ => rfl
      | ⟨k + 1, hk⟩ => exact absurd hk (by simp [Shape.rank_eq_length]))]
  congr 1
  refine Shape.Coord.ext_index (s := [channels]) ?_
  intro k hk
  rw [Shape.Coord.index_ofFun, Shape.Coord.index_ofFun, dite_eq_left hk, dite_eq_left hk]
  match k, hk with
  | 0, hk =>
    obtain ⟨i, j⟩ := o
    have hj := (Shape.Coord.linearize (s := sSpatial) j).isLt
    change ((Shape.Coord.unlinearize (s := [channels, sSpatial.size])
      ⟨(Shape.Coord.linearize (s := channels :: sSpatial) (i, j)).val, _⟩).1).val = i.val
    rw [Shape.Coord.unlinearizeFast_eq]
    change (Shape.Coord.linearize (s := channels :: sSpatial) (i, j)).val /
      Shape.size [sSpatial.size] = i.val
    have h1 : Shape.size [sSpatial.size] = Shape.size sSpatial := by
      simp [Shape.size]
    rw [Shape.Coord.linearize_cons_val (s := sSpatial) i j, h1]
    rw [Nat.add_mul_div_left _ _ (show 0 < Shape.size sSpatial by omega),
      Nat.div_eq_of_lt hj, Nat.zero_add]
  | k + 1, hk =>
    exact absurd hk (by
      simp only [Shape.rank_eq_length, List.length_cons, List.length_nil]
      omega)

end TorchLean
