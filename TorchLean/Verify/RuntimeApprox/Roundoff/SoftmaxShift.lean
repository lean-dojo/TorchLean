/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Verify.RuntimeApprox.Roundoff
public import TorchLean.Tensor.ConfiguredBinary
public import TorchLean.Tensor.Layers.Activation
import TorchLean.Verify.RuntimeApprox.Roundoff.Rules.Elementwise -- shake: keep
import TorchLean.Verify.RuntimeApprox.Roundoff.Rules.Tensor -- shake: keep

/-!
# The shift of stable softmax

`TorchLean.Softmax.maxShiftedExpVec` subtracts the maximum of a vector before taking exponentials.
On finite values that maximum is one of the entries, which shifts to an exact zero, and the
correctly rounded exponential of zero is exactly one. Every other exponential is nonnegative, so in
every IEEE format the shifted exponentials of a finite vector sum to at least one. As a `roundoff`
order rule, `sum_maxShiftedExpVec_pos` supplies the positive total that the normalization theorems
of `TorchLean.Verify.RuntimeApprox.Roundoff.SoftmaxReplay` require of their weights, so they apply
to the exponentials that softmax normalizes.

The log-softmax theorems at the end use the same shift. With finite logits,
`TorchLean.LogSoftmax.vec` subtracts the `log1p` of the sum of the other shifted
exponentials, so a finite log-probability is negative when its shifted logit and that sum are finite
and the sum is positive.
-/

public section

namespace TorchLean.Roundoff

open FloatLib.Floats FloatLib.Floats.Formats.BinaryInterchange TorchLean.Tensor

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
  [ExecFloat.ModelCodec plan (Model format) code]

local notation "Value" => ExecFloat (Configured.Family format code plan)

/-- The maximum of a finite vector is finite and, as a real value, one of its entries. -/
theorem exists_toReal_maxVec_eq [Storage Value] {n : Nat} (t : Tensor Value [n + 1])
    (hfinite : ∀ i, ExecFloat.Binary.isFinite (t.getScalar i) = true) :
    ExecFloat.Binary.isFinite (Softmax.maxVec t).item = true ∧
      ∃ k, ExecFloat.Binary.toReal (Softmax.maxVec t).item =
        ExecFloat.Binary.toReal (t.getScalar k) := by
  obtain ⟨finite, hreal⟩ := toReal_foldl_maximum (fun i => t.getScalar i)
    (List.finRange (n + 1)) _ (hfinite ⟨0, Nat.succ_pos n⟩) (fun i _ => hfinite i)
  simp only [Softmax.maxVec, Tensor.item_scalar]
  refine ⟨finite, ?_⟩
  rw [hreal]
  rcases List.foldl_max_eq_init_or_mem (List.finRange (n + 1))
      (fun i => ExecFloat.Binary.toReal (t.getScalar i))
      (ExecFloat.Binary.toReal (t.getScalar ⟨0, Nat.succ_pos n⟩)) with hstart | ⟨k, -, hk⟩
  · exact ⟨_, hstart⟩
  · exact ⟨k, hk⟩

/-- The exponentials of a finite vector shifted by its maximum sum to at least one in every IEEE
format: the entry holding the maximum contributes exactly one. -/
theorem one_le_sum_maxShiftedExpVec [Storage Value] {n : Nat}
    (t : Tensor Value [n + 1]) (hformat : format.isIEEE = true)
    (hfinite : ∀ i, ExecFloat.Binary.isFinite (t.getScalar i) = true) :
    1 ≤ ∑ i, ExecFloat.Binary.toReal ((Softmax.maxShiftedExpVec t).getScalar i) := by
  obtain ⟨maximumFinite, k, hk⟩ := exists_toReal_maxVec_eq t hfinite
  have entry (i : Fin (n + 1)) : (Softmax.maxShiftedExpVec t).getScalar i =
      ExecFloat.Binary.CorrectlyRounded.exp
        (t.getScalar i - (Softmax.maxVec t).item) := by
    simp only [Softmax.maxShiftedExpVec, Tensor.exp, sub_def,
        getScalar_map,
      getScalar_zipWith, getScalar_full]
    exact ExecFloat.Binary.CorrectlyRounded.mathFunctions_exp_of_isIEEE _
  have shift : ExecFloat.Binary.toReal (t.getScalar k - (Softmax.maxVec t).item) = 0 ∧
      ExecFloat.Binary.isFinite (t.getScalar k - (Softmax.maxVec t).item) = true := by
    have range : |ExecFloat.Binary.toReal (t.getScalar k) -
        ExecFloat.Binary.toReal (Softmax.maxVec t).item| ≤
          Model.toReal (Model.posMaxFinite format) := by
      rw [hk, sub_self, abs_zero]
      exact Model.toReal_posMaxFinite_nonneg format
    have finite := Model.isFinite_sub_of_abs_toReal_sub_le_posMaxFinite _ _ hformat (hfinite k)
      maximumFinite range
    change Model.toReal (ExecFloat.Binary.toModel
        (t.getScalar k - (Softmax.maxVec t).item)) = 0 ∧
      Model.isFinite (ExecFloat.Binary.toModel
        (t.getScalar k - (Softmax.maxVec t).item)) = true
    simp only [ExecFloat.Binary.toModel_hSub]
    refine ⟨?_, finite⟩
    rw [Model.toReal_sub_eq_roundAt _ _ hformat finite]
    change Model.roundAt format (ExecFloat.Binary.toReal (t.getScalar k) -
      ExecFloat.Binary.toReal (Softmax.maxVec t).item) = 0
    rw [hk, sub_self, Model.roundAt_zero]
  have pivot : ExecFloat.Binary.toReal ((Softmax.maxShiftedExpVec t).getScalar k) = 1 := by
    rw [entry, ExecFloat.Binary.CorrectlyRounded.toReal_exp_eq_roundAt shift.2 (by
      rw [shift.1, Real.exp_zero]
      exact Model.Transcendentals.CorrectlyRounded.one_lt_overflowThreshold hformat),
      shift.1, Real.exp_zero, Model.roundAt_one]
  rw [← pivot]
  exact Finset.single_le_sum
    (f := fun i => ExecFloat.Binary.toReal ((Softmax.maxShiftedExpVec t).getScalar i))
    (fun i _ => by
      simp only [entry]
      exact ExecFloat.Binary.CorrectlyRounded.toReal_exp_nonneg _) (Finset.mem_univ k)

/-- For a nonempty finite vector in an IEEE format, the shifted exponentials have a positive total,
as the normalization theorems of `TorchLean.Verify.RuntimeApprox.Roundoff.SoftmaxReplay` require. -/
@[roundoff_rule]
theorem sum_maxShiftedExpVec_pos [Storage Value] {n : Nat}
    (t : Tensor Value [n + 1]) (hformat : format.isIEEE = true)
    (hfinite : ∀ i, ExecFloat.Binary.isFinite (t.getScalar i) = true) :
    0 < ∑ i, ExecFloat.Binary.toReal ((Softmax.maxShiftedExpVec t).getScalar i) :=
  one_pos.trans_le (one_le_sum_maxShiftedExpVec t hformat hfinite)

/-! ## Log-softmax -/

section LogSoftmax

open FloatLib.Numerics

/-- With finite logits the pivot is finite, so `LogSoftmax.vec` subtracts
the `log1p` of the sum of the other shifted exponentials. -/
theorem _root_.TorchLean.LogSoftmax.vec_eq_of_isFinite [Storage Value] {n : Nat}
    (t : Tensor Value [n + 1]) (hfinite : ∀ i, ExecFloat.Binary.isFinite (t.getScalar i) = true) :
    LogSoftmax.vec t =
      let pivot : Fin (n + 1) := Softmax.pivot t
      let shift := t - Tensor.full _ (t.getScalar pivot)
      shift - (Tensor.full _ (Context.log1p (Tensor.sum
        (Tensor.ofFn fun j => if j = pivot then 0
          else MathFunctions.exp (shift.getScalar j))))) := by
  simp only [LogSoftmax.vec, isInf_of_isFinite (hfinite _),
    Bool.false_eq_true, ↓reduceIte]

/-- A finite log-probability is negative when the shift of its logit is finite and the sum of the
exponentials other than the pivot's is finite and positive, for every length: the shifted logit is
nonpositive and `log1p` of a positive value is positive. A range condition on the exact shift
gives the finite shift, but a finite shift need not satisfy one. -/
theorem _root_.TorchLean.LogSoftmax.vec_neg_of_tail_pos [Storage Value] {n : Nat}
    (t : Tensor Value [n + 1]) (hformat : format.isIEEE = true)
    (hfinite : ∀ i, ExecFloat.Binary.isFinite (t.getScalar i) = true) (i : Fin (n + 1)) :
    let pivot : Fin (n + 1) := Softmax.pivot t
    let shift := t - Tensor.full _ (t.getScalar pivot)
    let tail : Tensor Value [n + 1] := Tensor.ofFn fun j => if j = pivot then 0
      else MathFunctions.exp (shift.getScalar j)
    ExecFloat.Binary.isFinite (shift.getScalar i) = true →
      ExecFloat.Binary.isFinite (Tensor.sum tail) = true →
      ExecFloat.Binary.isFinite ((LogSoftmax.vec t).getScalar i) = true →
      0 < ExecFloat.Binary.toReal (Tensor.sum tail) →
      ExecFloat.Binary.toReal ((LogSoftmax.vec t).getScalar i) < 0 := by
  rw [LogSoftmax.vec_eq_of_isFinite t hfinite]
  roundoff

/-- A finite log-probability is negative when the shift of its logit and the sum of the
exponentials other than the pivot's are finite and one of those exponentials is positive: a
finite sum of nonnegative values is at least each of them. -/
theorem _root_.TorchLean.LogSoftmax.vec_neg_of_exp_pos [Storage Value] {n : Nat}
    (t : Tensor Value [n + 1]) (hformat : format.isIEEE = true)
    (hfinite : ∀ i, ExecFloat.Binary.isFinite (t.getScalar i) = true) (i k : Fin (n + 1)) :
    let pivot : Fin (n + 1) := Softmax.pivot t
    let shift := t - Tensor.full _ (t.getScalar pivot)
    let tail : Tensor Value [n + 1] := Tensor.ofFn fun j => if j = pivot then 0
      else MathFunctions.exp (shift.getScalar j)
    ExecFloat.Binary.isFinite (shift.getScalar i) = true →
      ExecFloat.Binary.isFinite (Tensor.sum tail) = true →
      ExecFloat.Binary.isFinite ((LogSoftmax.vec t).getScalar i) = true →
      0 < ExecFloat.Binary.toReal (tail.getScalar k) →
      ExecFloat.Binary.toReal ((LogSoftmax.vec t).getScalar i) < 0 := by
  intro pivot shift tail hshift hsum houtput hpositive
  have hnonneg : ∀ j ∈ List.finRange (n + 1), 0 ≤ ExecFloat.Binary.toReal (tail.getScalar j) :=
    fun j _ => by simp only [tail]; roundoff
  have hsumFold := hsum
  rw [Tensor.sum_eq_foldl, foldl_eq_fin_foldl, Fin.foldl_eq_foldl_finRange] at hsumFold
  have hdominates := (ExecFloat.Binary.le_toReal_foldl_add hsumFold hnonneg
    ExecFloat.Binary.toReal_zero.ge).2 k (List.mem_finRange k)
  rw [← Fin.foldl_eq_foldl_finRange, ← foldl_eq_fin_foldl,
    ← Tensor.sum_eq_foldl] at hdominates
  exact LogSoftmax.vec_neg_of_tail_pos t hformat hfinite i hshift hsum houtput
    (hpositive.trans_le hdominates)

end LogSoftmax

end TorchLean.Roundoff
