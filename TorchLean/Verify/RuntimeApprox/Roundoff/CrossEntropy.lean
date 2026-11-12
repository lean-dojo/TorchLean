/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Verify.RuntimeApprox.Roundoff.Softmax
public import TorchLean.Tensor.Layers.Loss
public import TorchLean.Tensor.Layers.Loss.Axis
import TorchLean.NN.Softmax.Correctness

/-!
# The log-probabilities and the hard-target gradient of cross-entropy

Cross-entropy of logits takes its log-probabilities from `CrossEntropy.vectorLogProbs`: the
one-level log-softmax `LogSoftmax.vec` by default, and the two-level normalizer when another class
exists, neither the pivot logit nor the runner-up is infinite, the computed sum of the
exponentials other than the pivot's is zero, and the computed exponential of the log-sum-exp of
the other classes and every entry of the two-level result are finite (`CrossEntropy.twoLevelAt`).
On that branch the `log1p` of the inner sum is `CrossEntropy.log1pSum`, which is that `log1p`
wherever it is finite (`Context.rescale_eq_self`) and has its bounds there
(`Roundoff.mixedError_toReal_rescale`). Over ℝ both branches are the log-softmax
(`CrossEntropy.vectorLogProbs_real`).
`roundoff mixed theorem` infers the error of the combined program for every IEEE format and every
length (`log_probs_error`). The reference is the exact log-softmax of the decoded logits on both
branches (`log_probs_reference`), and the budgets split on the guard `twoLevelAt`, which
`roundoff` keeps as one opaque branch condition. Registered as a rule, the bound covers each
log-probability that the gradient reads.

`roundoff mixed theorem` infers the error of `Tensor.crossEntropyBackward .logits` with mean
reduction from its implementation, for every IEEE format: the log-probabilities of each class
slice, their log-softmax VJP on the negated one-hot target, and the division of each entry by the
slice count. The sum of the one-hot target in the VJP is exact at every length. The inferred
reference follows the schedule. It is the exact gradient, the softmax of the decoded logits minus
the one-hot target (`softmax_sub_oneHot_eq`), and the theorems state their bounds against that.

The division is analyzed as computed: each gradient entry divided by the slice count `B` is the
exact quotient rounded once, with relative error `u` and the underflow error, and no overflow
condition.

The batch form covers the model-facing default, `oneHotCrossEntropy` with mean reduction over
the rows of a `[B, C]` matrix. Its conditions are discharged in bfloat16 for two rows of zero
logits (`batch_two_zeros_bfloat16`), where the kernel evaluates the computed gradient to `-1 / 4`
(`batch_two_zeros_bfloat16_eq`). The tape's cross-entropy node multiplies the gradient by the
upstream seed, which is one at the loss and leaves it exact (`tape_seed_one_exact`).

For finite logits with two or more classes, in every IEEE format and at every length, the computed
log-probabilities are negative wherever the forward computation is finite, provided that
whenever the computed tail is zero, the computed exponential of the log-sum-exp of the other
classes is finite and positive and every entry of the two-level result is finite
(`vectorLogProbs_neg`). The bounds of this file do not show that the two-level branch keeps a
nonzero gradient where the one-level tail underflows. The kernel evaluates that at single inputs
in `TorchLeanExamples.DeepDives.Floats.RoundingBugs`.
-/

public section

namespace TorchLean.Roundoff.CrossEntropy

open FloatLib.Floats FloatLib.Floats.Formats.BinaryInterchange FloatLib.Numerics
open Tensor

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
  [ExecFloat.ModelCodec plan (Model format) code]

local notation "Value" => ExecFloat (Configured.Family format code plan)

attribute [local roundoff_rule] log_softmax_error

/-! ## Log-probabilities -/

/-- The reference `roundoff` infers for `CrossEntropy.vectorLogProbs` on decoded logits `x`, with
pivot `p` and runner-up `q`, is the exact log-softmax `x i - log (∑ j, exp (x j))` on both
branches of `twoLevelAt`. The one-level normalizer `log (1 + ∑_{j ≠ p} exp (x j - x p))` and the
two-level one `log (1 + exp (x q - x p + log (1 + ∑_{j ∉ {p, q}} exp (x j - x q))))`
(`CrossEntropy.log_two_level`) both equal `log (∑ j, exp (x j)) - x p`. -/
theorem log_probs_reference [Storage Value] {n : Nat} (t : Tensor Value [n + 1]) (i : Fin (n + 1)) :
    let p : Fin (n + 1) := Softmax.pivot t
    let q : Fin (n + 1) := CrossEntropy.runnerUp t p
    let x (j : Fin (n + 1)) : ℝ := ExecFloat.Binary.toReal (Tensor.Rep.get t (j, PUnit.unit))
    let one := x i - x p - Real.log (1 + (Tensor.Rep.coordinates [n + 1]).foldl
      (fun sum coordinate => sum + if coordinate.1 = p then 0 else Real.exp (x coordinate.1 - x p))
      0)
    (if CrossEntropy.twoLevelAt t = true then
        x i - x p - Real.log (1 + Real.exp (x q - x p + Real.log (1 +
          (Tensor.Rep.coordinates [n + 1]).foldl (fun sum coordinate =>
            sum + if coordinate.1 = p ∨ coordinate.1 = q then 0
              else Real.exp (x coordinate.1 - x q)) 0)))
      else one) =
      ExecFloat.Binary.toReal (t.getScalar i) -
        Real.log (∑ j, Real.exp (ExecFloat.Binary.toReal (t.getScalar j))) := by
  intro p q x one
  have get (j : Fin (n + 1)) : ExecFloat.Binary.toReal (t.getScalar j) = x j := by
    rw [Tensor.getScalar_eq_apply]
  have hone : one = x i - Real.log (∑ j, Real.exp (x j)) := by
    have hfold : (Tensor.Rep.coordinates [n + 1]).foldl (fun sum coordinate =>
        sum + if coordinate.1 = p then 0 else Real.exp (x coordinate.1 - x p)) 0 =
        ∑ j, if j = p then 0 else Real.exp (x j - x p) :=
      Tensor.Rep.foldl_coordinates_add fun j => if j = p then 0 else Real.exp (x j - x p)
    have htail : 1 + ∑ j, (if j = p then 0 else Real.exp (x j - x p)) =
        ∑ j, Real.exp (x j - x p) := by
      rw [Finset.sum_ite, Finset.sum_const_zero, zero_add, Finset.filter_ne',
        Finset.sum_erase_eq_sub (Finset.mem_univ _), sub_self, Real.exp_zero]
      ring
    have hweight : ∑ j, Real.exp (x j - x p) = (∑ j, Real.exp (x j)) * Real.exp (-x p) := by
      simp_rw [sub_eq_add_neg, Real.exp_add]
      rw [Finset.sum_mul]
    have hS : 0 < ∑ j, Real.exp (x j) :=
      Finset.sum_pos (fun j _ => Real.exp_pos _) ⟨p, Finset.mem_univ p⟩
    change x i - x p - Real.log (1 + _) = _
    rw [hfold, htail, hweight, Real.log_mul hS.ne' (Real.exp_ne_zero _), Real.log_exp]
    ring
  simp only [get]
  by_cases hz : CrossEntropy.twoLevelAt t = true
  · have hq : q ≠ p := CrossEntropy.runnerUp_ne_of_twoLevelAt hz
    rw [ite_eq_left hz, Tensor.Rep.foldl_coordinates_add
      (fun j => if j = p ∨ j = q then 0 else Real.exp (x j - x q)),
      CrossEntropy.log_two_level x hq]
    ring
  · rw [ite_eq_right hz]
    exact hone

/-- The log-probabilities `CrossEntropy.vectorLogProbs` of a vector of finite logits, for every
IEEE format and every length, against the exact log-softmax of the decoded logits
(`log_probs_reference`). The budgets split on the guard `twoLevelAt`. -/
roundoff mixed theorem log_probs_error using log_probs_reference :=
  fun [Storage Value] {n : Nat} (t : Tensor Value [n + 1]) (_hformat : format.isIEEE = true)
    (_hfinite : ∀ i, ExecFloat.Binary.isFinite (t.getScalar i) = true) (i : Fin (n + 1)) =>
    ExecFloat.Binary.toReal ((CrossEntropy.vectorLogProbs t).getScalar i)

attribute [local roundoff_rule] log_probs_error

/-! ## The hard-target gradient -/

/-- The cross-entropy gradient's schedule on decoded logits `x`, with backward pivot `p'` and the
sum `S` of the exponentials `exp (x j)` folded in coordinate order, has the exact gradient
`softmax x i - [i = label]` as its reference: `exp (x j - log S)` is `softmax x j`, and at the
backward pivot the VJP's formula for the pivot entry has the same value. -/
theorem softmax_sub_oneHot_eq {n : Nat} (x : Fin (n + 1) → ℝ) (p' label i : Fin (n + 1)) :
    let S := (Tensor.Rep.coordinates [n + 1]).foldl (fun sum coordinate =>
      sum + Real.exp (x coordinate.1)) 0
    let q (j : Fin (n + 1)) : ℝ := Real.exp (x j - Real.log S)
    (if i = p' then
        -(q p' - 1) * -(if i = label then 1 else 0) - q p' * if label = p' then 0 else -1
      else q i - if i = label then 1 else 0) =
      Real.exp (x i) / ∑ j, Real.exp (x j) - if i = label then 1 else 0 := by
  intro S q
  have hS : S = ∑ j, Real.exp (x j) := Tensor.Rep.foldl_coordinates_add fun j => Real.exp (x j)
  have hq (j : Fin (n + 1)) : q j = Real.exp (x j) / ∑ k, Real.exp (x k) := by
    change Real.exp (x j - Real.log S) = _
    rw [Real.exp_sub, hS,
      Real.exp_log (Finset.sum_pos (fun _ _ => Real.exp_pos _) Finset.univ_nonempty)]
  simp only [hq]
  split_ifs <;> simp_all

/-- `softmax_sub_oneHot_eq` on the decoded logits of a vector: the reference `roundoff` infers for
the cross-entropy gradient is the exact gradient. -/
theorem gradient_reference [Storage Value] {n : Nat} (t : Tensor Value [n + 1])
    (label i : Fin (n + 1)) :
    let p' : Fin (n + 1) := Softmax.pivot (CrossEntropy.vectorLogProbs t)
    let x (j : Fin (n + 1)) : ℝ :=
      ExecFloat.Binary.toReal (Tensor.Rep.get t (j, PUnit.unit))
    let q (j : Fin (n + 1)) : ℝ := Real.exp (x j - Real.log
      ((Tensor.Rep.coordinates [n + 1]).foldl (fun sum coordinate =>
        sum + Real.exp (x coordinate.1)) 0))
    (if i = p' then
        -(q p' - 1) * -(if i = label then 1 else 0) - q p' * if label = p' then 0 else -1
      else q i - if i = label then 1 else 0) =
      Real.exp (ExecFloat.Binary.toReal (t.getScalar i)) /
          ∑ j, Real.exp (ExecFloat.Binary.toReal (t.getScalar j)) -
        if i = label then 1 else 0 := by
  intro p' x q
  simp only [Tensor.getScalar_eq_apply]
  exact softmax_sub_oneHot_eq x p' label i

/-- `softmax_sub_oneHot_eq` on a row of decoded logits of a batch, divided by the slice count: the
reference `roundoff` infers for the batched cross-entropy gradient is the exact gradient divided
by the slice count. -/
theorem batch_gradient_reference [Storage Value] {b n : Nat}
    (logits : Tensor Value [b + 1, n + 1]) (labels : Fin (b + 1) → Fin (n + 1)) (r : Fin (b + 1))
    (k : Fin (n + 1)) :
    let p' : Fin (n + 1) := Softmax.pivot ((Tensor.dim fun row =>
        CrossEntropy.vectorLogProbs (Tensor.unstack logits row)).unstack r)
    let x (j : Fin (n + 1)) : ℝ :=
      ExecFloat.Binary.toReal (Tensor.Rep.get logits (r, j, PUnit.unit))
    let q (j : Fin (n + 1)) : ℝ := Real.exp (x j - Real.log
      ((Tensor.Rep.coordinates [n + 1]).foldl (fun sum coordinate =>
        sum + Real.exp (x coordinate.1)) 0))
    (if k = p' then
        -(q p' - 1) * -(if k = labels r then 1 else 0) - q p' * if labels r = p' then 0 else -1
      else q k - if k = labels r then 1 else 0) /
        (axisMeanDenom [b + 1, n + 1] ⟨1, Shape.AxisInBounds.proof⟩ : ℝ) =
      (Real.exp (ExecFloat.Binary.toReal (Tensor.get2 logits r k)) /
          ∑ j, Real.exp (ExecFloat.Binary.toReal (Tensor.get2 logits r j)) -
        if k = labels r then 1 else 0) /
        axisMeanDenom [b + 1, n + 1] ⟨1, Shape.AxisInBounds.proof⟩ := by
  intro p' x q
  have get (j : Fin (n + 1)) : ExecFloat.Binary.toReal (Tensor.get2 logits r j) = x j := by
    rw [Tensor.get2, Tensor.getScalar_eq_apply, Tensor.unstack_apply]
  simp only [get]
  rw [softmax_sub_oneHot_eq x p' (labels r) k]

/-- The hard-target cross-entropy gradient of a vector of logits, for every IEEE format and every
length. It is one class slice, so the mean scale is exact. Each log-probability is read through
`log_probs_error`. The reference is the exact gradient `exp t_i / ∑ j, exp t_j - [i = label]` on
the decoded logits (`gradient_reference`). -/
roundoff mixed theorem gradient_error using gradient_reference :=
  fun [Storage Value] {n : Nat} (t : Tensor Value [n + 1]) (label : Fin (n + 1))
    (_hformat : format.isIEEE = true)
    (_hfinite : ∀ i, ExecFloat.Binary.isFinite (t.getScalar i) = true) (i : Fin (n + 1)) =>
    ExecFloat.Binary.toReal ((Tensor.crossEntropyBackward .logits t
      (Tensor.ofFn fun j => if j = label then 1 else 0)
      ⟨0, by exact Nat.zero_lt_succ _⟩).getScalar i)

/-- The same gradient for a batch of `B = b + 1` rows of logits with hard labels, classes on the
last axis, each entry divided once by `B`. The reference is the exact gradient
`(exp x_rk / ∑ j, exp x_rj - [k = labels r]) / B` on the decoded logits
(`batch_gradient_reference`). The relative error is `(1 + u) * u + u`, the VJP's
final subtraction and the division each rounded once. -/
roundoff mixed theorem batch_gradient_error using batch_gradient_reference :=
  fun [Storage Value] {b n : Nat} (logits : Tensor Value [b + 1, n + 1])
    (labels : Fin (b + 1) → Fin (n + 1)) (_hformat : format.isIEEE = true)
    (_hfinite : ∀ r k, ExecFloat.Binary.isFinite (Tensor.get2 logits r k) = true)
    (r : Fin (b + 1)) (k : Fin (n + 1)) =>
    ExecFloat.Binary.toReal (Tensor.get2 (Tensor.crossEntropyBackward .logits logits
      (Tensor.dim fun r => Tensor.ofFn fun k => if k = labels r then 1 else 0)
      ⟨1, by exact Nat.one_lt_succ_succ _⟩) r k)

/-- For two rows of two zero logits with labels zero, in bfloat16, the gradient of the first logit
of the first row is finite, with relative error at most `2⁻⁵` and absolute error at most `2⁻⁶`
against `(1 / 2 - 1) / 2 = -1 / 4`. Both rows take the one-level branch of
`CrossEntropy.vectorLogProbs`. The conditions follow from bounds on `log 2` and on two
exponentials of small multiples of it. -/
theorem batch_two_zeros_bfloat16 :
    let logits : Tensor BFloat16 [2, 2] := Tensor.dim fun _ => Tensor.ofFn fun _ => 0
    let target : Tensor BFloat16 [2, 2] :=
      Tensor.dim fun _ => Tensor.ofFn fun k => if k = 0 then 1 else 0
    ExecFloat.Binary.isFinite
        (Tensor.get2 (Tensor.crossEntropyBackward .logits logits target ⟨1,
            by exact Nat.one_lt_succ_succ _⟩) 0 0) = true ∧
      MixedError (ExecFloat.Binary.toReal
          (Tensor.get2 (Tensor.crossEntropyBackward .logits logits target ⟨1,
              by exact Nat.one_lt_succ_succ _⟩) 0 0))
        (-1 / 4) (1 / 2 ^ 5) (1 / 2 ^ 6) := by
  intro logits target
  have hcoordinates : List.finRange (Shape.size [Nat.succ 1]) =
      [⟨0, by decide⟩, ⟨1, by decide⟩] := by decide
  have hfirst : (Shape.Coord.unlinearize (s := [Nat.succ 1]) ⟨0,
      by exact Nat.zero_lt_succ _⟩).1 = 0 := by
    decide
  have hsecond : (Shape.Coord.unlinearize (s := [Nat.succ 1]) ⟨1,
      by exact Nat.one_lt_succ_succ _⟩).1 = 1 := by
    decide
  have hzero (r k : Fin 2) :
      ExecFloat.Binary.toReal (Tensor.Rep.get logits (r, k, PUnit.unit)) = 0 := by
    rw [ExecFloat.Binary.toReal_of_toRat?_eq_some (q := 0) (by revert r k; decide +kernel)]
    norm_num
  have hpivot (r : Fin 2) :
      (Softmax.pivot (Tensor.unstack logits r) : Fin 2) = 0 := by
    revert r; decide +kernel
  have hbackward (r : Fin 2) : (Softmax.pivot ((Tensor.dim fun i =>
        CrossEntropy.vectorLogProbs (Tensor.unstack logits i)).unstack r) :
          Fin 2) = 0 := by
    revert r; decide +kernel
  have hcount : axisMeanDenom [2, 2] ⟨1, Shape.AxisInBounds.proof⟩ = 2 := by decide
  have htwo (r : Fin 2) : CrossEntropy.twoLevelAt (Tensor.unstack logits r) = false := by
    revert r; decide +kernel
  have h := batch_gradient_error logits (fun _ => 0) (by decide)
    (by decide +kernel) 0 0
  have hreference := batch_gradient_reference logits (fun _ => 0) 0 0
  dsimp only at h hreference
  rw [← hreference] at h
  clear hreference
  simp only [hpivot, hbackward, hcount, htwo, Bool.false_eq_true, AllSteps, foldRelative,
    foldAbsolute, List.mixedFoldStep, Tensor.Rep.foldl_eq_fin_foldl, Fin.foldl_eq_foldl_finRange,
    hcoordinates, List.foldl_cons, List.foldl_nil, Tensor.Rep.coordinates, Tensor.Rep.ofFn,
    Tensor.Rep.getFlat_ofFlatFn, id, hfirst, hsecond, hzero, ite_true, true_implies,
    sub_self, abs_zero, mul_zero, Real.exp_zero] at h
  have hlog : |Real.log 2| = Real.log 2 := abs_of_pos (Real.log_pos one_lt_two)
  have hexp : Real.exp (-Real.log 2) = 1 / 2 := by
    rw [Real.exp_neg, Real.exp_log two_pos, one_div]
  norm_num [Model.unitRoundoffAt_eq, Model.underflowErrorAt_eq, FloatFormat.minSubnormalExponent,
    FloatFormat.minNormalExponent, FloatFormat.Encoding.defaultBias, FloatFormat.ieeeBias,
    Model.toReal_posMaxFinite, Model.bpow, Formats.Flocq.bpow_binaryRadix, Model.pow2_eq_two_pow,
    Model.Transcendentals.CorrectlyRounded.cast_overflowThreshold, FloatFormat.maxFiniteFracField,
    FloatFormat.maxNormalExponent, FloatFormat.maxFiniteExpField, FloatFormat.fracMaskNat,
    FloatFormat.Encoding.maxFiniteExponent, ExecFloat.Binary.descriptor, hlog, hexp] at h ⊢
  have hlogTwo := Real.log_two_lt_d9
  have hsmall : Real.exp (1 / 64) ≤ 64 / 63 := by
    have := Real.exp_bound_div_one_sub_of_interval (x := 1 / 64) (by norm_num) (by norm_num)
    norm_num at this ⊢
    exact this
  generalize he₁ : Real.exp _ = e₁ at h
  generalize he₂ : Real.exp _ = e₂ at h
  have he₁pos : 0 < e₁ := he₁ ▸ Real.exp_pos _
  have he₁le : e₁ ≤ 64 / 63 :=
    he₁ ▸ ((Real.exp_le_exp (y := 1 / 64)).mpr (by linarith)).trans hsmall
  have he₂le : e₂ ≤ 3 :=
    he₂ ▸ ((Real.exp_le_exp (y := 1)).mpr (by linarith)).trans (Real.exp_one_lt_d9.le.trans
      (by norm_num))
  have hproduct : Real.log 2 * e₁ ≤ 7 / 10 * (64 / 63) :=
    mul_le_mul (by linarith) he₁le he₁pos.le (by norm_num)
  obtain ⟨finite, bound⟩ := h (by linarith) (by linarith) (by linarith) (by linarith)
    (by linarith) (by linarith)
  exact ⟨finite, ⟨by norm_num, by norm_num, bound.bound.trans (by norm_num; linarith)⟩⟩

/-- The kernel evaluates that gradient entry: it decodes to `-1 / 4`. -/
theorem batch_two_zeros_bfloat16_eq :
    let logits : Tensor BFloat16 [2, 2] := Tensor.dim fun _ => Tensor.ofFn fun _ => 0
    let target : Tensor BFloat16 [2, 2] :=
      Tensor.dim fun _ => Tensor.ofFn fun k => if k = 0 then 1 else 0
    ExecFloat.Binary.toRat?
        (Tensor.get2 (Tensor.crossEntropyBackward .logits logits target ⟨1,
            by exact Nat.one_lt_succ_succ _⟩) 0 0) = some (-1 / 4) := by
  decide +kernel

/-- The tape's cross-entropy node scales the gradient by the upstream seed. At the loss the seed
is one, and the product is exact for every finite gradient. -/
theorem tape_seed_one_exact [Storage Value] {s : Shape} (gradient : Tensor Value s)
    (coordinate : s.Coord) (hformat : format.isIEEE = true)
    (hfinite : ExecFloat.Binary.isFinite (gradient coordinate) = true) :
    MixedError (ExecFloat.Binary.toReal (Tensor.scale gradient 1 coordinate))
      (ExecFloat.Binary.toReal (gradient coordinate)) 0 0 := by
  roundoff

/-! ## Signs of the log-probabilities -/

/-- On the two-level branch, with `others` the computed exponential of the log-sum-exp of the
classes other than the pivot `p`, the log-probability `(t i - t p) - log1p others` is negative
when it, the shifted logit `t i - t p` and `others` are finite and `others` is positive. This
holds for every IEEE format and every length. -/
theorem twoLevel_neg [Storage Value] {n : Nat} (t : Tensor Value [n + 1])
    (hformat : format.isIEEE = true)
    (hfinite : ∀ i, ExecFloat.Binary.isFinite (t.getScalar i) = true) (i : Fin (n + 1)) :
    let p : Fin (n + 1) := Softmax.pivot t
    let q := CrossEntropy.runnerUp t p
    let shift := t - Tensor.full _ (t.getScalar p)
    let inner : Tensor Value [n + 1] := Tensor.ofFn fun j =>
      if j = p ∨ j = q then 0 else MathFunctions.exp (t.getScalar j - t.getScalar q)
    let others := MathFunctions.exp (shift.getScalar q + CrossEntropy.log1pSum inner)
    ExecFloat.Binary.isFinite (shift.getScalar i) = true →
      ExecFloat.Binary.isFinite others = true →
      ExecFloat.Binary.isFinite ((shift - Tensor.full _ (Context.log1p others)).getScalar i) =
        true →
      0 < ExecFloat.Binary.toReal others →
      ExecFloat.Binary.toReal ((shift - Tensor.full _ (Context.log1p others)).getScalar i) < 0 := by
  roundoff

/-- For finite logits with two or more classes, every IEEE format and every length, the computed
log-probability `(CrossEntropy.vectorLogProbs t) i` is negative when it, the shifted logit
`t i - t p` and the computed tail `Σ_{j ≠ p} exp (t j - t p)` are finite, provided that whenever
the computed tail is zero, the computed exponential `others` of the log-sum-exp of the other
classes is finite and positive and every entry of the two-level result `(t j - t p) - log1p others`
is finite. On the one-level branch the failed test `twoLevelAt` then makes the computed tail
positive (`LogSoftmax.vec_neg_of_tail_pos`), and the two-level branch is `twoLevel_neg`. -/
theorem vectorLogProbs_neg [Storage Value] {n : Nat} (t : Tensor Value [n + 1])
    (hformat : format.isIEEE = true)
    (hfinite : ∀ i, ExecFloat.Binary.isFinite (t.getScalar i) = true) (i : Fin (n + 1))
    (hn : n ≠ 0) :
    let p : Fin (n + 1) := Softmax.pivot t
    let q := CrossEntropy.runnerUp t p
    let shift := t - Tensor.full _ (t.getScalar p)
    let tail : Tensor Value [n + 1] := Tensor.ofFn fun j =>
      if j = p then 0 else MathFunctions.exp (shift.getScalar j)
    let inner : Tensor Value [n + 1] := Tensor.ofFn fun j =>
      if j = p ∨ j = q then 0 else MathFunctions.exp (t.getScalar j - t.getScalar q)
    let others := MathFunctions.exp (shift.getScalar q + CrossEntropy.log1pSum inner)
    let twoLevel := shift - Tensor.full _ (Context.log1p others)
    ExecFloat.Binary.isFinite (shift.getScalar i) = true →
      ExecFloat.Binary.isFinite (Tensor.sum tail) = true →
      ExecFloat.Binary.isFinite ((CrossEntropy.vectorLogProbs t).getScalar i) = true →
      ((Tensor.sum tail == 0) = true →
        ExecFloat.Binary.isFinite others = true ∧ 0 < ExecFloat.Binary.toReal others ∧
          ∀ j, ExecFloat.Binary.isFinite (twoLevel.getScalar j) = true) →
      ExecFloat.Binary.toReal ((CrossEntropy.vectorLogProbs t).getScalar i) < 0 := by
  intro p q shift tail inner others twoLevel hshift hsum houtput htwo
  simp only [CrossEntropy.vectorLogProbs] at houtput ⊢
  split_ifs at houtput ⊢ with hz
  · have hbeq : (Tensor.sum tail == 0) = true := by
      unfold CrossEntropy.twoLevelAt at hz
      simp only [Bool.and_eq_true] at hz
      exact hz.1.1.2
    obtain ⟨hothers, hpositive, -⟩ := htwo hbeq
    exact twoLevel_neg t hformat hfinite i hshift hothers houtput hpositive
  · have hnonneg : ∀ j ∈ List.finRange (n + 1), 0 ≤ ExecFloat.Binary.toReal (tail.getScalar j) :=
      fun j _ => by simp only [tail]; roundoff
    have hsumFold := hsum
    rw [Tensor.sum_eq_foldl, foldl_eq_fin_foldl, Fin.foldl_eq_foldl_finRange] at hsumFold
    have hseed := (ExecFloat.Binary.le_toReal_foldl_add hsumFold hnonneg
      ExecFloat.Binary.toReal_zero.ge).1
    rw [← Fin.foldl_eq_foldl_finRange, ← foldl_eq_fin_foldl, ← Tensor.sum_eq_foldl] at hseed
    have hzero : 0 ≤ ExecFloat.Binary.toReal (Tensor.sum tail) :=
      ExecFloat.Binary.toReal_zero.symm.trans_le hseed
    exact LogSoftmax.vec_neg_of_tail_pos t hformat hfinite i hshift hsum houtput
      (lt_of_le_of_ne hzero fun h => hz (by
        have hq : q ≠ p := CrossEntropy.runnerUp_ne t p hn
        have hbeq : (Tensor.sum tail == 0) = true :=
          ExecFloat.Binary.beq_zero_iff.mpr ⟨hsum, h.symm⟩
        obtain ⟨hothers, -, hentries⟩ := htwo hbeq
        have hall : ((List.finRange (n + 1)).all fun j => Context.isFinite (twoLevel.getScalar j)) =
            true :=
          List.all_eq_true.mpr fun j _ => context_isFinite_of_isFinite (hentries j)
        unfold CrossEntropy.twoLevelAt
        dsimp only
        rw [decide_eq_true hq, isInf_of_isFinite (hfinite _), isInf_of_isFinite (hfinite _), hbeq,
          context_isFinite_of_isFinite hothers, hall]
        rfl))

end TorchLean.Roundoff.CrossEntropy
