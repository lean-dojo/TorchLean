/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Proofs.DirectedIBPNormalizationTensor
public import NN.Proofs.Autograd.FDeriv.HardMaskedSoftmax
public import NN.Proofs.Autograd.FDeriv.SoftmaxAxis
import all NN.MLTheory.CROWN.Graph.Engine.Base

/-!
# Rounded softmax range transfers

Softmax probabilities are bounded using the real, max-shifted specifications. Axis permutations
preserve these coordinate bounds. The hard-mask transfer additionally detects blocked entries and
rows with exactly one allowed entry.
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Graph.DirectedBackward

open Spec TorchLean TorchLean.Tensor
open _root_.Proofs.Autograd
open scoped BigOperators

noncomputable section

private theorem applyAdjacentSwaps_dim_succ (s : Shape) (n : Nat) (swaps : List Nat) :
    Shape.applyAdjacentSwaps (.dim n s) (swaps.map Nat.succ) =
      .dim n (Shape.applyAdjacentSwaps s swaps) := by
  induction swaps generalizing s with
  | nil => rfl
  | cons depth swaps ih =>
      simpa only [List.map_cons, Shape.applyAdjacentSwaps, Shape.swapAdjacentAtDepth] using
        ih (s.swapAdjacentAtDepth depth)

private theorem move_outer_axis (s : Shape) (n : Nat) :
    Shape.applyAdjacentSwaps (.dim n s) (List.range s.rank) = s.appendDim n := by
  induction s with
  | scalar => rfl
  | dim m s ih =>
      simp only [Shape.rank, List.range_succ_eq_map, Shape.applyAdjacentSwaps,
        Shape.swapAdjacentAtDepth, applyAdjacentSwaps_dim_succ, Shape.appendDim, ih]

private theorem moved_shape_ends_in_axis (s : Shape) (axis : Nat) (haxis : axis < s.rank) :
    ∃ leading : Shape,
      Shape.applyAdjacentSwaps s (Shape.moveAxisToInnermostSwaps s.rank axis) =
      leading.appendDim (s.toList[axis]?.getD 0) := by
  induction s generalizing axis with
  | scalar => simp [Shape.rank] at haxis
  | dim n s ih =>
      cases axis with
      | zero =>
          refine ⟨s, ?_⟩
          simpa [Shape.moveAxisToInnermostSwaps, Shape.rank, Shape.toList] using
            move_outer_axis s n
      | succ axis =>
          have ha : axis < s.rank := by simpa [Shape.rank] using haxis
          obtain ⟨leading, hleading⟩ := ih axis ha
          refine ⟨.dim n leading, ?_⟩
          have hswaps :
              Shape.moveAxisToInnermostSwaps (Shape.dim n s).rank (axis + 1) =
                (Shape.moveAxisToInnermostSwaps s.rank axis).map Nat.succ := by
            have hsub : s.rank + 1 - (axis + 1 + 1) = s.rank - (axis + 1) := by omega
            simp only [Shape.moveAxisToInnermostSwaps, Shape.rank, hsub, List.map_map]
            congr 1
            funext k
            dsimp only [Function.comp_def]
            omega
          rw [hswaps, applyAdjacentSwaps_dim_succ, hleading]
          rfl

private theorem rowWidth_appendDim (s : Shape) (n : Nat) :
    SoftmaxAxis.rowWidth (s.appendDim n) = n := by
  induction s with
  | scalar => rfl
  | dim m s ih =>
      cases s <;> simp_all [Shape.appendDim, SoftmaxAxis.rowWidth]

/-- Moving a valid softmax axis to the end preserves its extent as the row width. -/
theorem softmax_moved_rowWidth (s : Shape) (axis : Nat) (haxis : axis < s.rank) :
    SoftmaxAxis.rowWidth
        (Shape.applyAdjacentSwaps s (Shape.moveAxisToInnermostSwaps s.rank axis)) =
      s.toList[axis]?.getD 0 := by
  obtain ⟨leading, hleading⟩ := moved_shape_ends_in_axis s axis haxis
  rw [hleading, rowWidth_appendDim]

private theorem forall_permuteByAdjacentSwaps {s : Shape} {P : ℝ → Prop}
    (x : Tensor ℝ s) (h : ∀ c, P (x c)) (swaps : List Nat) :
    ∀ c, P (Tensor.permuteByAdjacentSwaps x swaps c) := by
  induction swaps generalizing s with
  | nil => exact h
  | cons depth swaps ih =>
      apply ih
      intro c
      rw [Tensor.swapAdjacentAxes_apply]
      exact h _

private theorem forall_cast_shape {s t : Shape} {P : ℝ → Prop} (hst : s = t)
    (x : Tensor ℝ s) (h : ∀ c, P (x c)) : ∀ c, P ((hst ▸ x) c) := by
  subst t
  exact h

/-- Every coordinate of the actual innermost softmax lies in the closed unit interval. -/
theorem softmaxInnermostSpec_mem_Icc {s : Shape} (x : Tensor ℝ s) (c : s.Coord) :
    Activation.Internal.softmaxInnermostSpec x c ∈ Set.Icc (0 : ℝ) 1 := by
  induction s with
  | scalar =>
      simp [Activation.Internal.softmaxInnermostSpec, Tensor.scalar]
  | dim n s ih =>
      cases s with
      | scalar =>
          cases n with
          | zero => exact Fin.elim0 c.1
          | succ n =>
              simpa only [Activation.Internal.softmaxInnermostSpec,
                Tensor.getScalar_eq_apply] using
                _root_.Proofs.softmax_vec_spec_mem_unitInterval x c.1
      | dim m s =>
          change Tensor.dim
            (fun i => Activation.Internal.softmaxInnermostSpec (x.unstack i)) c ∈ _
          simp only [Tensor.dim, TorchLean.Tensor.Internal.Rep.stack_apply]
          exact ih (x.unstack c.1) c.2

/-- A singleton innermost row has probability one, independently of its logit. -/
theorem softmaxInnermostSpec_eq_one {s : Shape} (x : Tensor ℝ s)
    (hwidth : SoftmaxAxis.rowWidth s = 1) (c : s.Coord) :
    Activation.Internal.softmaxInnermostSpec x c = 1 := by
  rw [SoftmaxAxis.softmaxInnermostSpec_apply]
  have h (n : Nat) (hn : n = 1) (v : Vec n) (i : Fin n) : softmaxVec v i = 1 := by
    subst n
    exact SoftmaxAxis.softmaxVec_one v i
  exact h _ hwidth _ _

/-- The actual real softmax along any valid axis has probability-valued coordinates. -/
theorem softmaxSpec_mem_Icc {s : Shape} (axis : Nat) [Shape.AxisInBounds axis s]
    (x : Tensor ℝ s) (c : s.Coord) :
    Activation.softmaxSpec axis x c ∈ Set.Icc (0 : ℝ) 1 := by
  let swaps := Shape.moveAxisToInnermostSwaps s.rank axis
  let moved := Tensor.permuteByAdjacentSwaps x swaps
  have h := forall_permuteByAdjacentSwaps (Activation.Internal.softmaxInnermostSpec moved)
    (softmaxInnermostSpec_mem_Icc moved) swaps.reverse
  exact forall_cast_shape (Shape.applyAdjacentSwaps_reverse s swaps) _ h c

/-- Softmax on a singleton selected axis is exactly one at every coordinate. -/
theorem softmaxSpec_eq_one {s : Shape} (axis : Nat) [Shape.AxisInBounds axis s]
    (x : Tensor ℝ s) (hwidth : s.toList[axis]?.getD 0 = 1) (c : s.Coord) :
    Activation.softmaxSpec axis x c = 1 := by
  let swaps := Shape.moveAxisToInnermostSwaps s.rank axis
  let moved := Tensor.permuteByAdjacentSwaps x swaps
  have hw : SoftmaxAxis.rowWidth (s.applyAdjacentSwaps swaps) = 1 :=
    (softmax_moved_rowWidth s axis Shape.AxisInBounds.proof).trans hwidth
  have h := forall_permuteByAdjacentSwaps (P := fun z => z = 1)
    (Activation.Internal.softmaxInnermostSpec moved)
    (softmaxInnermostSpec_eq_one moved hw) swaps.reverse
  exact forall_cast_shape (P := fun z => z = 1)
    (Shape.applyAdjacentSwaps_reverse s swaps) _ h c

/-- A blocked logit has exactly zero probability in the real hard-mask specification. -/
theorem hardMaskedSoftmaxVecSpec_eq_zero {n : Nat}
    (x : Tensor ℝ [n]) (allowed : Tensor Bool [n]) (i : Fin n)
    (hi : allowed.getScalar i = false) :
    (Spec.hardMaskedSoftmaxVecSpec x allowed).getScalar i = 0 := by
  rw [HardMaskedSoftmax.getScalar_hardMaskedSoftmaxVecSpec]
  simp [HardMaskedSoftmax.numerator, hi]

/-- Hard-masked probabilities lie in the unit interval, including an entirely blocked row. -/
theorem hardMaskedSoftmaxVecSpec_mem_Icc {n : Nat}
    (x : Tensor ℝ [n]) (allowed : Tensor Bool [n]) (i : Fin n) :
    (Spec.hardMaskedSoftmaxVecSpec x allowed).getScalar i ∈ Set.Icc (0 : ℝ) 1 := by
  classical
  cases hi : allowed.getScalar i with
  | false => simp [hardMaskedSoftmaxVecSpec_eq_zero x allowed i hi]
  | true =>
      rw [HardMaskedSoftmax.getScalar_hardMaskedSoftmaxVecSpec]
      have hn (j : Fin n) :
          0 ≤ HardMaskedSoftmax.numerator allowed (getScalarE x) j := by
        unfold HardMaskedSoftmax.numerator
        split
        · exact (Real.exp_pos _).le
        · exact le_rfl
      have hle :
          HardMaskedSoftmax.numerator allowed (getScalarE x) i ≤
            HardMaskedSoftmax.denominator allowed (getScalarE x) :=
        Finset.single_le_sum (fun j _ => hn j) (Finset.mem_univ i)
      have hp : 0 < HardMaskedSoftmax.denominator allowed (getScalarE x) := by
        apply lt_of_lt_of_le _ hle
        simpa only [HardMaskedSoftmax.numerator, hi, Bool.true_eq, ↓reduceIte] using
          Real.exp_pos (getScalarE x i)
      exact ⟨div_nonneg (hn i) hp.le, (div_le_one hp).mpr hle⟩

/-- The only allowed entry in a hard-mask row has probability one. -/
theorem hardMaskedSoftmaxVecSpec_eq_one {n : Nat}
    (x : Tensor ℝ [n]) (allowed : Tensor Bool [n]) (i : Fin n)
    (hi : allowed.getScalar i = true)
    (hunique : ∀ j, allowed.getScalar j = true → j = i) :
    (Spec.hardMaskedSoftmaxVecSpec x allowed).getScalar i = 1 := by
  classical
  rw [HardMaskedSoftmax.getScalar_hardMaskedSoftmaxVecSpec]
  have hsum : HardMaskedSoftmax.denominator allowed (getScalarE x) =
      HardMaskedSoftmax.numerator allowed (getScalarE x) i := by
    apply Finset.sum_eq_single i
    · intro j _ hji
      have hj : allowed.getScalar j = false := by
        cases h : allowed.getScalar j
        · rfl
        · exact False.elim (hji (hunique j h))
      simp [HardMaskedSoftmax.numerator, hj]
    · simp
  rw [hsum]
  apply div_self
  simp [HardMaskedSoftmax.numerator, hi, Real.exp_ne_zero]

variable {α : Type} [Storage α] [Context α] [BoundOps α] [LawfulBoundOps α]

local notation "value" => LawfulBoundOps.toReal (α := α)

/-- The executable range box encloses the actual real softmax of any input tensor. -/
theorem ibpSoftmaxRange_encloses {s : Shape} (axis : Nat) [Shape.AxisInBounds axis s]
    (x : Tensor ℝ s) {g : Nat → ℝ}
    (hg : ∀ c : s.Coord, g (Shape.Coord.linearize c).val = Activation.softmaxSpec axis x c) :
    RowEncloses (ibpSoftmaxRange (α := α) s axis s.size) s.size g := by
  unfold ibpSoftmaxRange
  dsimp only
  split
  · rename_i hwidth
    rw [rowEncloses_iff]
    intro i
    have h := hg (Shape.Coord.unlinearize i)
    rw [Shape.Coord.linearize_unlinearize,
      softmaxSpec_eq_one axis x hwidth] at h
    simp [Tensor.getScalar_full, LawfulBoundOps.toReal_one, h]
  · rw [rowEncloses_iff]
    intro i
    have h := hg (Shape.Coord.unlinearize i)
    rw [Shape.Coord.linearize_unlinearize] at h
    simpa only [Tensor.getScalar_full, LawfulBoundOps.toReal_zero,
      LawfulBoundOps.toReal_one, h, Set.mem_Icc] using
      softmaxSpec_mem_Icc axis x (Shape.Coord.unlinearize i)

/-- The actual recursive hard-mask transfer encloses its real specification at every rank. -/
theorem ibpHardMaskedSoftmaxLastTensor_encloses {s : Shape}
    (lo hi : Tensor α s) (x : Tensor ℝ s) (allowed : Tensor Bool s) (c : s.Coord) :
    value ((ibpHardMaskedSoftmaxLastTensor lo hi allowed).1 c) ≤
        Spec.hardMaskedSoftmaxSpec x allowed c ∧
      Spec.hardMaskedSoftmaxSpec x allowed c ≤
        value ((ibpHardMaskedSoftmaxLastTensor lo hi allowed).2 c) := by
  induction s with
  | scalar =>
      cases h : allowed.item <;>
        simp [ibpHardMaskedSoftmaxLastTensor, Spec.hardMaskedSoftmaxSpec,
          Tensor.scalar, h, LawfulBoundOps.toReal_zero, LawfulBoundOps.toReal_one]
  | dim n s ih =>
      cases s with
      | scalar =>
          obtain ⟨i, ⟨⟩⟩ := c
          simp only [Spec.hardMaskedSoftmaxSpec, ibpHardMaskedSoftmaxLastTensor,
            Tensor.ofFn_apply]
          have hzero := hardMaskedSoftmaxVecSpec_eq_zero x allowed i
          have hone := hardMaskedSoftmaxVecSpec_eq_one x allowed i
          have hrange := hardMaskedSoftmaxVecSpec_mem_Icc x allowed i
          rw [Tensor.getScalar_eq_apply (hardMaskedSoftmaxVecSpec x allowed) i]
            at hzero hone hrange
          cases ha : allowed.getScalar i with
          | false =>
              simp [hzero ha, LawfulBoundOps.toReal_zero]
          | true =>
              simp only [↓reduceIte, LawfulBoundOps.toReal_one]
              split
              · simpa only [LawfulBoundOps.toReal_zero, Set.mem_Icc] using
                  hrange
              · rename_i hother
                have hunique (j : Fin n) (hj : allowed.getScalar j = true) : j = i := by
                  by_contra hji
                  have hany : (List.finRange n).any
                      (fun k => i != k && allowed.getScalar k) = true := by
                    apply List.any_eq_true.mpr
                    exact ⟨j, List.mem_finRange j, by simp [hj, Ne.symm hji]⟩
                  exact hother hany
                simp [hone ha hunique, LawfulBoundOps.toReal_one]
      | dim m s =>
          change value (Tensor.dim (fun i =>
              (ibpHardMaskedSoftmaxLastTensor (lo.unstack i) (hi.unstack i)
                (allowed.unstack i)).1) c) ≤
              Tensor.dim (fun i => Spec.hardMaskedSoftmaxSpec (x.unstack i)
                (allowed.unstack i)) c ∧
            Tensor.dim (fun i => Spec.hardMaskedSoftmaxSpec (x.unstack i)
              (allowed.unstack i)) c ≤
              value (Tensor.dim (fun i =>
                (ibpHardMaskedSoftmaxLastTensor (lo.unstack i) (hi.unstack i)
                  (allowed.unstack i)).2) c)
          simp only [Tensor.dim, TorchLean.Tensor.Internal.Rep.stack_apply]
          exact ih (lo.unstack c.1) (hi.unstack c.1) (x.unstack c.1)
            (allowed.unstack c.1) c.2

end

end NN.MLTheory.CROWN.Graph.DirectedBackward
