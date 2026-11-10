/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Verify.RuntimeApprox.Roundoff.Softmax
public import TorchLean.Tensor.Layers.Loss
public import TorchLean.Tensor.Layers.Loss.Axis

/-!
# The hard-target cross-entropy gradient

`roundoff mixed theorem` infers the error of `Tensor.crossEntropyBackward .logits` with mean
reduction from its implementation, for every IEEE format: the log-softmax of each class slice,
its VJP on the negated one-hot target, and the division of each entry by the slice count.
Each slice's log-softmax is read through `log_softmax_error`, registered as a rule, so its
budget and conditions appear once per log-probability the VJP reads. The VJP sums the one-hot
target exactly, at every length. The inferred reference follows the schedule. It is the exact
gradient, the softmax of the decoded logits minus the one-hot target (`softmax_sub_oneHot_eq`),
and the theorems state their bounds against that.

The division is analyzed as computed: each gradient entry divided by the slice count `B` is the
exact quotient rounded once, with relative error `u` and the underflow error, and no overflow
condition.

The batch form covers the model-facing default, `oneHotCrossEntropy` with mean reduction over
the rows of a `[B, C]` matrix. Its conditions are discharged in bfloat16 for two rows of zero
logits (`batch_two_zeros_bfloat16`), where the kernel finds the computed gradient
exactly `-1 / 4` (`batch_two_zeros_bfloat16_eq`). The tape's cross-entropy node
multiplies the gradient by the upstream seed, which is one at the loss and leaves it exact
(`tape_seed_one_exact`).
-/

public section

namespace TorchLean.Roundoff.CrossEntropy

open FloatLib.Floats FloatLib.Floats.Formats.BinaryInterchange FloatLib.Numerics
open Tensor
open Rep

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
  [ExecFloat.ModelCodec plan (Model format) code]

local notation "Value" => ExecFloat (Configured.Family format code plan)

attribute [local roundoff_rule] log_softmax_error

/-- The cross-entropy gradient's schedule on decoded logits `x`, with forward pivot `p` and backward
pivot `p'`, has the exact gradient `softmax x i - [i = label]` as its reference: the
log-normalizer `log (1 + ∑_{j ≠ p} exp (x j - x p))` is `log ∑ j, exp (x j - x p)`, and at the
backward pivot the special formula of the VJP has the same value. -/
theorem softmax_sub_oneHot_eq {n : Nat} (x : Fin (n + 1) → ℝ) (p p' label i : Fin (n + 1)) :
    let tail := (Tensor.Rep.coordinates [n + 1]).foldl (fun sum coordinate =>
      sum + if coordinate.1 = p then 0 else Real.exp (x coordinate.1 - x p)) 0
    let q (j : Fin (n + 1)) : ℝ := Real.exp (x j - x p - Real.log (1 + tail))
    (if i = p' then
        -(q p' - 1) * -(if i = label then 1 else 0) - q p' * if label = p' then 0 else -1
      else q i - if i = label then 1 else 0) =
      Real.exp (x i) / ∑ j, Real.exp (x j) - if i = label then 1 else 0 := by
  intro tail q
  have htail : 1 + tail = ∑ j, Real.exp (x j - x p) := by
    have hfold : tail = ∑ j, if j = p then 0 else Real.exp (x j - x p) :=
      foldl_coordinates_add fun j => if j = p then 0 else Real.exp (x j - x p)
    rw [hfold, Finset.sum_ite, Finset.sum_const_zero, zero_add, Finset.filter_ne',
      Finset.sum_erase_eq_sub (Finset.mem_univ _), sub_self, Real.exp_zero]
    ring
  have hq (j : Fin (n + 1)) : q j = Real.exp (x j) / ∑ k, Real.exp (x k) := by
    have hpos : 0 < ∑ k, Real.exp (x k - x p) :=
      Finset.sum_pos (fun _ _ => Real.exp_pos _) Finset.univ_nonempty
    change Real.exp (x j - x p - Real.log (1 + tail)) = _
    rw [Real.exp_sub, htail, Real.exp_log hpos]
    simp only [Real.exp_sub, ← Finset.sum_div]
    rw [div_div_div_cancel_right₀ (Real.exp_pos _).ne']
  simp only [hq]
  split_ifs <;> simp_all

/-- `softmax_sub_oneHot_eq` on the decoded logits of a vector: the reference `roundoff` infers for
the cross-entropy gradient is the exact gradient. -/
theorem gradient_reference [Storage Value] {n : Nat} (t : Tensor Value [n + 1])
    (label i : Fin (n + 1)) :
    let p : Fin (n + 1) := Softmax.pivot t
    let p' : Fin (n + 1) := Softmax.pivot (LogSoftmax.vec t)
    let x (j : Fin (n + 1)) : ℝ :=
      ExecFloat.Binary.toReal (Tensor.Rep.get t (j, PUnit.unit))
    let q (j : Fin (n + 1)) : ℝ := Real.exp (x j - x p - Real.log
      (1 + (Tensor.Rep.coordinates [n + 1]).foldl (fun sum coordinate =>
        sum + if coordinate.1 = p then 0 else Real.exp (x coordinate.1 - x p)) 0))
    (if i = p' then
        -(q p' - 1) * -(if i = label then 1 else 0) - q p' * if label = p' then 0 else -1
      else q i - if i = label then 1 else 0) =
      Real.exp (ExecFloat.Binary.toReal (t.getScalar i)) /
          ∑ j, Real.exp (ExecFloat.Binary.toReal (t.getScalar j)) -
        if i = label then 1 else 0 := by
  intro p p' x q
  simp only [Tensor.getScalar_eq_apply]
  exact softmax_sub_oneHot_eq x p p' label i

/-- `softmax_sub_oneHot_eq` on a row of decoded logits of a batch, divided by the slice count: the
reference `roundoff` infers for the batched cross-entropy gradient is the exact gradient divided
by the slice count. -/
theorem batch_gradient_reference [Storage Value] {b n : Nat}
    (logits : Tensor Value [b + 1, n + 1]) (labels : Fin (b + 1) → Fin (n + 1)) (r : Fin (b + 1))
    (k : Fin (n + 1)) :
    let p : Fin (n + 1) := Softmax.pivot (Tensor.unstack logits r)
    let p' : Fin (n + 1) := Softmax.pivot ((Tensor.dim fun row =>
        LogSoftmax.vec (Tensor.unstack logits row)).unstack r)
    let x (j : Fin (n + 1)) : ℝ :=
      ExecFloat.Binary.toReal (Tensor.Rep.get logits (r, j, PUnit.unit))
    let q (j : Fin (n + 1)) : ℝ := Real.exp (x j - x p - Real.log
      (1 + (Tensor.Rep.coordinates [n + 1]).foldl (fun sum coordinate =>
        sum + if coordinate.1 = p then 0 else Real.exp (x coordinate.1 - x p)) 0))
    (if k = p' then
        -(q p' - 1) * -(if k = labels r then 1 else 0) - q p' * if labels r = p' then 0 else -1
      else q k - if k = labels r then 1 else 0) /
        (axisMeanDenom [b + 1, n + 1] ⟨1, Shape.AxisInBounds.proof⟩ : ℝ) =
      (Real.exp (ExecFloat.Binary.toReal (Tensor.get2 logits r k)) /
          ∑ j, Real.exp (ExecFloat.Binary.toReal (Tensor.get2 logits r j)) -
        if k = labels r then 1 else 0) /
        axisMeanDenom [b + 1, n + 1] ⟨1, Shape.AxisInBounds.proof⟩ := by
  intro p p' x q
  have get (j : Fin (n + 1)) : ExecFloat.Binary.toReal (Tensor.get2 logits r j) = x j := by
    rw [Tensor.get2, Tensor.getScalar_eq_apply,
      Tensor.unstack_apply]
  simp only [get]
  rw [softmax_sub_oneHot_eq x p p' (labels r) k]

/-- The hard-target cross-entropy gradient of a vector of logits, for every IEEE format and every
length. It is one class slice, so the mean scale is exact. The reference is the exact gradient
`exp t_i / ∑ j, exp t_j - [i = label]` on the decoded logits (`gradient_reference`).
-/
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
roundoff mixed theorem batch_gradient_error using
    batch_gradient_reference :=
  fun [Storage Value] {b n : Nat} (logits : Tensor Value [b + 1, n + 1])
    (labels : Fin (b + 1) → Fin (n + 1)) (_hformat : format.isIEEE = true)
    (_hfinite : ∀ r k, ExecFloat.Binary.isFinite (Tensor.get2 logits r k) = true)
    (r : Fin (b + 1)) (k : Fin (n + 1)) =>
    ExecFloat.Binary.toReal (Tensor.get2 (Tensor.crossEntropyBackward .logits logits
      (Tensor.dim fun r => Tensor.ofFn fun k => if k = labels r then 1 else 0)
      ⟨1, by exact Nat.one_lt_succ_succ _⟩) r k)

/-- For two rows of two zero logits with labels zero, in bfloat16, the gradient of the first logit
of the first row is finite, with relative error at most `2⁻⁵` and absolute error at most `2⁻⁶`
against `(1 / 2 - 1) / 2 = -1 / 4`. The conditions follow from bounds on `log 2` and on two
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
        LogSoftmax.vec (Tensor.unstack logits i)).unstack r) :
          Fin 2) = 0 := by
    revert r; decide +kernel
  have hcount : axisMeanDenom [2, 2] ⟨1, Shape.AxisInBounds.proof⟩ = 2 := by decide
  have h := batch_gradient_error logits (fun _ => 0) (by decide)
    (by decide +kernel) 0 0
  have hreference := batch_gradient_reference logits (fun _ => 0) 0 0
  dsimp only at h hreference
  rw [← hreference] at h
  simp only [hpivot, hbackward, hcount, AllSteps, foldRelative, foldAbsolute, List.mixedFoldStep,
    Tensor.Rep.foldl_eq_fin_foldl, Fin.foldl_eq_foldl_finRange, hcoordinates,
    List.foldl_cons, List.foldl_nil, Tensor.Rep.coordinates, Tensor.Rep.ofFn,
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

/-- The kernel evaluates that gradient entry: it decodes to exactly `-1 / 4`. -/
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

end TorchLean.Roundoff.CrossEntropy
