/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

import all NN.MLTheory.CROWN.Graph.Engine.Base
public import NN.MLTheory.CROWN.Proofs.DirectedIBPTensor

/-!
# Directed broadcasting and axis reductions

Successful transfers enclose the exact broadcast, sum, or mean. The runtime guards decide whether
an axis is available; surviving axes may be empty. Mean denominators are bounded by the directed
count fold, without assuming that a natural-number conversion is exact.
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Graph.DirectedBackward

open Spec TorchLean TorchLean.Tensor
open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Graph.LayerNormDirected
open scoped BigOperators

noncomputable section

variable {α : Type} [Storage α] [Context α] [BoundOps α] [LawfulBoundOps α]

local notation "value" => LawfulBoundOps.toReal (α := α)

/-- The exact sum along `axis`, in the runtime's flat output order. -/
def realAxisSum (s : Shape) (axis : Nat) (f : Nat → ℝ) : Nat → ℝ :=
  tensorValues (Tensor.reduceDim Tensor.sumSpec axis (realTensor s f))

/-- The exact axis sum divided by the selected axis length. -/
def realAxisMean (s : Shape) (axis : Nat) (f : Nat → ℝ) (i : Nat) : ℝ :=
  realAxisSum s axis f i / (s.toList[axis]?.getD 0 : ℝ)

/-- Broadcasting only selects coordinates of the input, at every valid rank and extent. -/
theorem ibpBroadcastTo_encloses {s t : Shape} {B box : FlatBox α} {f : Nat → ℝ}
    (hB : RowEncloses B s.size f) (hb : Shape.CanBroadcastTo s t)
    (hbox : ibpBroadcastTo s t B = some box) :
    RowEncloses box t.size (tensorValues (Tensor.broadcastTo hb (realTensor s f))) := by
  unfold ibpBroadcastTo at hbox
  split at hbox
  next hd =>
    simp only [Option.some.injEq] at hbox
    subst box
    exact ((tensorEncloses_ibpUnflatten hd hB).broadcastTo hb).tensorValues
  next => contradiction

omit [BoundOps α] [LawfulBoundOps α] in
/-- A successful broadcast supplies its compatibility witness. -/
theorem ibpBroadcastTo_canBroadcast {s t : Shape} {B box : FlatBox α}
    (hbox : ibpBroadcastTo s t B = some box) : Shape.CanBroadcastTo s t := by
  unfold ibpBroadcastTo at hbox
  split at hbox
  · split at hbox
    · assumption
    · contradiction
  · contradiction

/-- Directed summation encloses the exact sum over the selected axis. -/
theorem ibpReduceSumAxis_encloses {s : Shape} {axis : Nat} {B box : FlatBox α}
    {f : Nat → ℝ} (hB : RowEncloses B s.size f)
    (hbox : ibpReduceSumAxis axis B s = some box) :
    RowEncloses box (Tensor.shapeAfterSum s axis).size (realAxisSum s axis f) := by
  unfold ibpReduceSumAxis at hbox
  split at hbox
  next hd =>
    cases ha : Shape.nonemptyAxis? axis s with
    | none => simp [ha] at hbox
    | some witness =>
        simp only [ha, Option.some.injEq] at hbox
        subst box
        exact ((tensorEncloses_ibpUnflatten hd hB).reduceSum axis).tensorValues
  next => contradiction

/-- The two directed counters enclose the exact natural length. -/
theorem directed_count_encloses (n : Nat) :
    let count := (List.range n).foldl
      (fun (lo, hi) _ => (BoundOps.addDown lo 1, BoundOps.addUp hi 1)) (0, 0)
    value count.1 ≤ (n : ℝ) ∧ (n : ℝ) ≤ value count.2 := by
  induction n with
  | zero => simp [LawfulBoundOps.toReal_zero (α := α)]
  | succ n ih =>
      simp only [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil]
      constructor
      · refine (LawfulBoundOps.addDown_le _ _).trans ?_
        rw [LawfulBoundOps.toReal_one, Nat.cast_succ]
        exact add_le_add ih.1 le_rfl
      · refine le_trans ?_ (LawfulBoundOps.le_addUp _ _)
        rw [LawfulBoundOps.toReal_one, Nat.cast_succ]
        exact add_le_add ih.2 le_rfl

/-- Dividing each directed axis sum by an enclosing count encloses the exact axis mean. -/
theorem ibpReduceMeanAxis_encloses [NonlinearBoundOps α] [LawfulNonlinearBoundOps α]
    {s : Shape} {axis : Nat} {B box : FlatBox α} {f : Nat → ℝ}
    (hB : RowEncloses B s.size f)
    (hbox : ibpReduceMeanAxis axis B s = some box) :
    RowEncloses box (Tensor.shapeAfterSum s axis).size (realAxisMean s axis f) := by
  unfold ibpReduceMeanAxis at hbox
  obtain ⟨summed, hsum, hbox⟩ := Option.bind_eq_some_iff.mp hbox
  obtain ⟨_, _, hbox⟩ := Option.bind_eq_some_iff.mp hbox
  have hcount := directed_count_encloses (α := α) (s.toList[axis]?.getD 0)
  apply boxUnaryEnclosure?_encloses (F := fun x => x / (s.toList[axis]?.getD 0 : ℝ))
    ?_ (ibpReduceSumAxis_encloses hB hsum) hbox
  intro lo hi outLo outHi x hout hlo hhi
  obtain ⟨_, _, hout⟩ := Option.bind_eq_some_iff.mp hout
  exact LawfulNonlinearBoundOps.divBounds_enclosure
    (checkedFiniteBounds?_bind_eq_some hout) hlo hhi hcount.1 hcount.2

end

end NN.MLTheory.CROWN.Graph.DirectedBackward
