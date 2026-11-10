/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Verify.RuntimeApprox.Constants
public import TorchLean.Verify.RuntimeApprox.Roundoff.Coverage
public import TorchLean.Verify.Floats.IEEE754.Expressions
public import TorchLean.Verify.RuntimeApprox.NF.Attention
public import TorchLean.Verify.RuntimeApprox.NF.Convolution
public import TorchLean.Verify.RuntimeApprox.NF.EndToEnd
public import TorchLean.Verify.RuntimeApprox.NF.Normalization
public import TorchLean.Verify.RuntimeApprox.NF.Optimizers
public import TorchLean.Verify.RuntimeApprox.NF.ReductionOps
public import TorchLean.Verify.Floats.IEEE754.Reductions
public import TorchLean.Verify.RuntimeApprox.Roundoff.Rules.Autograd -- shake: keep
public import TorchLean.Verify.RuntimeApprox.Roundoff.SoftmaxShift
public import TorchLean.Tensor.ConfiguredBinary
public import TorchLean.Tensor.Layers.Activation
public meta import TorchLean.Tensor.Einops.Elab.Einsum
public meta import TorchLean.Tensor.Einops.Elab.Pack
public meta import TorchLean.Tensor.Literal
public meta import TorchLean.Tensor.Einops.Elab.Transform
public meta import TorchLean.Tensor.Tactic.Einops.Report
public meta import FloatLib.Floats.Formats.BinaryInterchange.Configured.Value.Core
public meta import TorchLean.Tensor.Elementwise
public meta import TorchLean.Tensor.ConfiguredBinary
public import FloatLib.Numerics.Automation.Attributes
public import TorchLean.Tensor.Literal
public import TorchLean.Tensor.Tactic.Init
public import TorchLean.Verify.RuntimeApprox.Roundoff.Rules.Elementwise
public import TorchLean.Verify.RuntimeApprox.Roundoff.Rules.Power
public import TorchLean.Verify.RuntimeApprox.Roundoff.Rules.Tensor
public import TorchLean.Verify.RuntimeApprox.Roundoff.SoftmaxReplay

/-!
# Mixed error of the stable softmax and log-softmax

`roundoff mixed theorem` infers each statement from the computation alone, for every IEEE format
and every length: the finiteness of the computed value, a mixed bound against the exact
computation on the decoded inputs, and the range conditions it needs. The stable softmax of
`TorchLean.Softmax.vec`, the stable log-softmax of
`TorchLean.LogSoftmax.vec` and its VJP `TorchLean.LogSoftmax.vecBackward`
are inferred this way (`softmax_error`, `log_softmax_error`, `log_softmax_backward_error`). The
conditions of the last two are discharged in binary32 and bfloat16: for the log-softmax at two zero
logits, and for the VJP at a confident prediction, where its pivot coordinate keeps the small
gradient `exp y₀ - 1`.

A dot product with a one-hot vector of any length returns the selected entry exactly, or its
negation for a one-hot vector of minus one. The shift that the log-softmax applies before its
exponentials decodes to zero at the pivot, and each of its coordinates is finite and nonpositive
when the differences of the inputs are in range.
-/

public section

namespace TorchLean.Roundoff

open FloatLib.Floats FloatLib.Floats.Formats.BinaryInterchange FloatLib.Numerics
open Tensor.Rep

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
  [ExecFloat.ModelCodec plan (Model format) code]

local notation "Value" => ExecFloat (Configured.Family format code plan)

/-! ## Exact operations -/

/-- A dot product with a one-hot vector is finite and returns the selected entry exactly, for
every length. -/
theorem one_hot_dot_exact [Storage Value] {n : Nat} (v : Tensor Value [n]) (k : Fin n)
    (hformat : format.isIEEE = true)
    (hfinite : ∀ i, ExecFloat.Binary.isFinite (v.getScalar i) = true) :
    ExecFloat.Binary.isFinite (Tensor.dot v (Tensor.ofFn fun i => if i = k then 1 else 0)) =
        true ∧
      MixedError (ExecFloat.Binary.toReal
          (Tensor.dot v (Tensor.ofFn fun i => if i = k then 1 else 0)))
        (ExecFloat.Binary.toReal (v.getScalar k)) 0 0 := by
  roundoff

/-- A one-hot vector of minus one selects the negated entry exactly. -/
theorem one_hot_dot_minus_one_exact [Storage Value] {n : Nat} (v : Tensor Value [n]) (k : Fin n)
    (hformat : format.isIEEE = true)
    (hfinite : ∀ i, ExecFloat.Binary.isFinite (v.getScalar i) = true) :
    MixedError (ExecFloat.Binary.toReal
        (Tensor.dot v (Tensor.ofFn fun i => if i = k then -1 else 0)))
      (-ExecFloat.Binary.toReal (v.getScalar k)) 0 0 := by
  roundoff

/-! ## Stable log-softmax -/

/-- The pivot coordinate of the shift that `LogSoftmax.vec` applies before
its exponentials decodes to zero. -/
theorem log_softmax_shift_pivot [Storage Value] {n : Nat} (t : Tensor Value [n + 1])
    (hformat : format.isIEEE = true)
    (hfinite : ∀ i, ExecFloat.Binary.isFinite (t.getScalar i) = true) :
    let pivot : Fin (n + 1) := Softmax.pivot t
    ExecFloat.Binary.toRat?
      ((t - Tensor.full _ (t.getScalar pivot)).getScalar pivot) =
      some 0 := by
  roundoff

/-- Every coordinate of that shift is finite and nonpositive, given that the differences of
the inputs are in range. -/
theorem log_softmax_shift_nonpositive [Storage Value] {n : Nat} (t : Tensor Value [n + 1])
    (hformat : format.isIEEE = true)
    (hfinite : ∀ i, ExecFloat.Binary.isFinite (t.getScalar i) = true)
    (hrange : ∀ i j, |ExecFloat.Binary.toReal (t.getScalar i) -
      ExecFloat.Binary.toReal (t.getScalar j)| ≤ Model.toReal (Model.posMaxFinite format)) :
    let pivot : Fin (n + 1) := Softmax.pivot t
    ∀ i, ExecFloat.Binary.isFinite
        ((t - Tensor.full _ (t.getScalar pivot)).getScalar i) = true ∧
      ExecFloat.Binary.toReal
        ((t - Tensor.full _ (t.getScalar pivot)).getScalar i) ≤ 0 := by
  roundoff

/-- The stable log-softmax of `LogSoftmax.vec`, for every IEEE format and
every length: the argmax shift, the exponentials of the other entries, their sum, `log1p` and the
final subtraction. Each output is finite and has a mixed error against the exact log-softmax of the
decoded logits. Both terms of the final subtraction are nonpositive, so its relative part weighs
the shift's rounding and the logarithm's error, the sum's accumulated relative error and one
rounding of `log1p`, by their magnitudes, and adds the subtraction's own rounding. The absolute
part comes from the underflow allowances of the sum and of `log1p`. The conditions bound the
shifts, the exponentials and the enclosures of the partial sums, of the logarithm and of the
output. -/
roundoff mixed theorem log_softmax_error :=
  fun [Storage Value] {n : Nat} (t : Tensor Value [n + 1]) (_hformat : format.isIEEE = true)
    (_hfinite : ∀ i, ExecFloat.Binary.isFinite (t.getScalar i) = true) (i : Fin (n + 1)) =>
    ExecFloat.Binary.toReal ((LogSoftmax.vec t).getScalar i)

/-- Apply `log_softmax_error` to the vector `t` of two zero logits and discharge its conditions:
decide the argmax, unroll the two steps of each fold, evaluate the format's constants and bound
`log 2`. The budgets are then weakened to the stated ones. -/
local macro "log_softmax_two_zeros " t:term : tactic => `(tactic| (
  have hpivot : (Softmax.pivot $t :
      Fin 2) = 0 := by decide +kernel
  have hcoordinates : List.finRange (Shape.size [Nat.succ 1]) =
      [⟨0, by decide⟩, ⟨1, by decide⟩] := by decide
  have hfirst : (Shape.Coord.unlinearize (s := [Nat.succ 1]) ⟨0, by decide⟩).1 = 0 := by
    decide
  have hsecond : (Shape.Coord.unlinearize (s := [Nat.succ 1]) ⟨1, by decide⟩).1 = 1 := by
    decide
  have hlog : |Real.log 2| = Real.log 2 := abs_of_pos (Real.log_pos one_lt_two)
  have h := log_softmax_error $t (by decide)
    (fun _ => by simp [Tensor.getScalar_ofFn, ExecFloat.Binary.isFinite_zero]) 0
  dsimp only at h
  simp only [hpivot, Tensor.ofFn_apply, ExecFloat.Binary.toReal_zero, sub_self, abs_zero,
    mul_zero, Real.exp_zero, AllSteps, foldRelative, foldAbsolute, List.mixedFoldStep,
    Tensor.Rep.foldl_eq_fin_foldl, Fin.foldl_eq_foldl_finRange, hcoordinates,
    List.foldl_cons, List.foldl_nil, Tensor.Rep.coordinates, Tensor.Rep.ofFn,
    Tensor.Rep.getFlat_ofFlatFn, id, hfirst, hsecond] at h
  norm_num [Model.unitRoundoffAt_eq, Model.underflowErrorAt_eq, FloatFormat.minSubnormalExponent,
    FloatFormat.minNormalExponent, FloatFormat.Encoding.defaultBias, FloatFormat.ieeeBias,
    Model.toReal_posMaxFinite, Model.bpow, Formats.Flocq.bpow_binaryRadix, Model.pow2_eq_two_pow,
    Model.Transcendentals.CorrectlyRounded.cast_overflowThreshold, FloatFormat.maxFiniteFracField,
    FloatFormat.maxNormalExponent, FloatFormat.maxFiniteExpField, FloatFormat.fracMaskNat,
    FloatFormat.Encoding.maxFiniteExponent, ExecFloat.Binary.descriptor, hlog] at h
  obtain ⟨finite, bound⟩ := h (by linarith [Real.log_two_lt_d9]) (by linarith [Real.log_two_lt_d9])
  exact ⟨finite, bound.mono (by norm_num) (by norm_num)⟩))

/-- In binary32 the log-softmax of two zero logits is finite, with relative error at most
`2⁻²² + 2⁻⁴⁵`, four unit roundoffs and a second-order term, and absolute error at most `2⁻¹⁴⁸`
against `-log 2`. -/
theorem log_softmax_two_zeros_binary32 :
    ExecFloat.Binary.isFinite ((LogSoftmax.vec
        (Tensor.ofFn fun _ => 0 : Tensor Binary32 [2])).getScalar 0) = true ∧
      MixedError (ExecFloat.Binary.toReal ((LogSoftmax.vec
          (Tensor.ofFn fun _ => 0 : Tensor Binary32 [2])).getScalar 0))
        (-Real.log 2) (1 / 2 ^ 22 + 1 / 2 ^ 45) (1 / 2 ^ 148) := by
  log_softmax_two_zeros (Tensor.ofFn fun _ => 0 : Tensor Binary32 [2])

/-- The same in bfloat16: relative error at most `2⁻⁶ + 2⁻¹³` and absolute error at most
`2⁻¹³²`. -/
theorem log_softmax_two_zeros_bfloat16 :
    ExecFloat.Binary.isFinite ((LogSoftmax.vec
        (Tensor.ofFn fun _ => 0 : Tensor BFloat16 [2])).getScalar 0) = true ∧
      MixedError (ExecFloat.Binary.toReal ((LogSoftmax.vec
          (Tensor.ofFn fun _ => 0 : Tensor BFloat16 [2])).getScalar 0))
        (-Real.log 2) (1 / 2 ^ 6 + 1 / 2 ^ 13) (1 / 2 ^ 132) := by
  log_softmax_two_zeros (Tensor.ofFn fun _ => 0 : Tensor BFloat16 [2])

/-! ## Stable softmax -/

/-- The stable softmax of `Softmax.vec`, for every IEEE format and every
length: the maximum, the shifted exponentials and their normalization by
`Softmax.normalizeWeights`. Each output is finite and has a mixed error against the
exact softmax of the decoded logits. The shifted exponentials sum to at least one, so the
normalization theorems of `TorchLean.Verify.RuntimeApprox.Roundoff.SoftmaxReplay` apply with no
condition on their computed total. The conditions bound the shifts and exponentials, require
`(1 + u) ^ (n - 1) < 2` of the sum and keep the relative error of the exponentials' total below
one. -/
roundoff mixed theorem softmax_error :=
  fun [Storage Value] {n : Nat} (t : Tensor Value [n + 1]) (_hformat : format.isIEEE = true)
    (_hfinite : ∀ i, ExecFloat.Binary.isFinite (t.getScalar i) = true) (i : Fin (n + 1)) =>
    ExecFloat.Binary.toReal ((Softmax.vec t).getScalar i)

/-! ## Log-softmax VJP -/

/-- The reference of the log-softmax VJP's schedule is the exact VJP `v_i - exp y_i * ∑ j, v_j` at
every coordinate: at the pivot `p` the special formula
`-(exp y_p - 1) * v_p - exp y_p * ∑_{j ≠ p} v_j` has the same value. The left side is the
reference `roundoff` infers, on the decoded entries. -/
theorem log_softmax_backward_reference [Storage Value] {n : Nat} (y v : Tensor Value [n + 1])
    (i : Fin (n + 1)) :
    let p : Fin (n + 1) := Softmax.pivot y
    let decode (x : Tensor Value [n + 1]) (j : Fin (n + 1)) : ℝ :=
      ExecFloat.Binary.toReal (Tensor.Rep.get x (j, PUnit.unit))
    (if i = p then
        -(Real.exp (decode y p) - 1) * decode v p - Real.exp (decode y p) *
          (Tensor.Rep.coordinates [n + 1]).foldl
            (fun sum coordinate => sum + if coordinate.1 = p then 0 else decode v coordinate.1) 0
      else decode v i - Real.exp (decode y i) *
        (Tensor.Rep.coordinates [n + 1]).foldl
          (fun sum coordinate => sum + decode v coordinate.1) 0) =
      ExecFloat.Binary.toReal (v.getScalar i) -
        Real.exp (ExecFloat.Binary.toReal (y.getScalar i)) *
          ∑ j, ExecFloat.Binary.toReal (v.getScalar j) := by
  intro p decode
  have get (x : Tensor Value [n + 1]) (j : Fin (n + 1)) :
      ExecFloat.Binary.toReal (x.getScalar j) = decode x j := by
    rw [Tensor.getScalar_eq_apply]
  rw [foldl_coordinates_add fun j => if j = p then 0 else decode v j,
    foldl_coordinates_add (decode v)]
  simp only [get]
  split_ifs with h
  · subst h
    rw [Finset.sum_ite, Finset.sum_const_zero, zero_add, Finset.filter_ne',
      Finset.sum_erase_eq_sub (Finset.mem_univ _)]
    ring
  · rfl

/-- The log-softmax VJP `LogSoftmax.vecBackward`, which the specification
and the tape's log-softmax node apply to each row of the stored output `y`, for every IEEE format
and every length. The pivot is the argmax of `y`, chosen anew. At the pivot the VJP computes
`-expm1 y_p * v_p - exp y_p * ∑_{j ≠ p} v_j`, elsewhere `v_i - exp y_i * ∑_j v_j`, rounding each
exponential and product separately. Each output is finite and has a mixed error against the exact
VJP `v_i - exp y_i * ∑_j v_j` on the decoded `y` and `v`, the pivot included
(`log_softmax_backward_reference`): the relative part is the final subtraction's rounding, the
absolute part collects the errors of the products and sums. The conditions bound the
exponentials, the partial sums, the products and the output. -/
roundoff mixed theorem log_softmax_backward_error using log_softmax_backward_reference :=
  fun [Storage Value] {n : Nat} (y v : Tensor Value [n + 1]) (_hformat : format.isIEEE = true)
    (_hy : ∀ i, ExecFloat.Binary.isFinite (y.getScalar i) = true)
    (_hv : ∀ i, ExecFloat.Binary.isFinite (v.getScalar i) = true) (i : Fin (n + 1)) =>
    ExecFloat.Binary.toReal ((LogSoftmax.vecBackward y v).getScalar i)

/-- Apply `log_softmax_backward_error` at the first of three coordinates, where the stored
log-probabilities `y` take their maximum and decode to `y₀`, and the upstream `v` is `[-1, 0, 0]`,
with its reference in the pivot form of `log_softmax_backward_reference`. The conditions follow
from `exp y₀ ≤ 1`. The budgets are then weakened to the stated ones. -/
local macro "log_softmax_backward_first " y₀:term : tactic => `(tactic| (
  intro y v
  have hpivot : (Softmax.pivot y :
      Fin 3) = 0 := by decide +kernel
  have hcoordinates : List.finRange (Shape.size [Nat.succ 2]) =
      [⟨0, by decide⟩, ⟨1, by decide⟩, ⟨2, by decide⟩] := by decide
  have hfirst : (Shape.Coord.unlinearize (s := [Nat.succ 2]) ⟨0, by decide⟩).1 = 0 := by
    decide
  have hsecond : (Shape.Coord.unlinearize (s := [Nat.succ 2]) ⟨1, by decide⟩).1 = 1 := by
    decide
  have hthird : (Shape.Coord.unlinearize (s := [Nat.succ 2]) ⟨2, by decide⟩).1 = 2 := by
    decide
  have hy : ExecFloat.Binary.toReal (Tensor.Rep.get y (0, PUnit.unit)) = ($y₀ : ℝ) := by
    rw [ExecFloat.Binary.toReal_of_toRat?_eq_some (q := $y₀) (by decide +kernel)]
    norm_num
  have hv0 : ExecFloat.Binary.toReal (Tensor.Rep.get v (0, PUnit.unit)) = ((-1 : ℚ) : ℝ) :=
    ExecFloat.Binary.toReal_of_toRat?_eq_some (by decide +kernel)
  have hv1 : ExecFloat.Binary.toReal (Tensor.Rep.get v (1, PUnit.unit)) = ((0 : ℚ) : ℝ) :=
    ExecFloat.Binary.toReal_of_toRat?_eq_some (by decide +kernel)
  have hv2 : ExecFloat.Binary.toReal (Tensor.Rep.get v (2, PUnit.unit)) = ((0 : ℚ) : ℝ) :=
    ExecFloat.Binary.toReal_of_toRat?_eq_some (by decide +kernel)
  have hexp := Real.exp_le_one_iff.mpr (show ($y₀ : ℝ) ≤ 0 by norm_num)
  have hpos := Real.exp_pos ($y₀ : ℝ)
  have h := log_softmax_backward_error y v (by decide) (by decide +kernel) (by decide +kernel) 0
  have hreference := log_softmax_backward_reference y v 0
  dsimp only at h hreference
  rw [← hreference] at h
  simp only [hpivot, AllSteps, foldRelative, foldAbsolute, List.mixedFoldStep,
    Tensor.Rep.foldl_eq_fin_foldl, Fin.foldl_eq_foldl_finRange, hcoordinates,
    List.foldl_cons, List.foldl_nil, Tensor.Rep.coordinates, Tensor.Rep.ofFn,
    Tensor.Rep.getFlat_ofFlatFn, id, hfirst, hsecond, hthird, hy, hv0, hv1, hv2,
    ite_true, true_implies] at h
  generalize Real.exp ($y₀ : ℝ) = exponential at h hexp hpos ⊢
  norm_num [Model.unitRoundoffAt_eq, Model.underflowErrorAt_eq, FloatFormat.minSubnormalExponent,
    FloatFormat.minNormalExponent, FloatFormat.Encoding.defaultBias, FloatFormat.ieeeBias,
    Model.toReal_posMaxFinite, Model.bpow, Formats.Flocq.bpow_binaryRadix, Model.pow2_eq_two_pow,
    Model.Transcendentals.CorrectlyRounded.cast_overflowThreshold, FloatFormat.maxFiniteFracField,
    FloatFormat.maxNormalExponent, FloatFormat.maxFiniteExpField, FloatFormat.fracMaskNat,
    FloatFormat.Encoding.maxFiniteExponent, ExecFloat.Binary.descriptor] at h ⊢
  have habs : |exponential - 1| = 1 - exponential := by
    rw [abs_of_nonpos (by linarith), neg_sub]
  rw [habs] at h
  obtain ⟨finite, bound⟩ := h (by linarith) (by linarith) (by linarith) (by linarith)
  exact ⟨finite, ⟨by norm_num, by norm_num, bound.bound.trans (by rw [habs]; linarith)⟩⟩))

/-- In binary32, the logits `[9, -9, -9]` have stored log-probabilities whose first entry is
`y₀ = -2143429 / 2⁴⁶`. On the upstream `[-1, 0, 0]` the VJP at that pivot is finite, with relative
error at most `2⁻²²` and absolute error at most `2⁻¹⁴⁸` against `exp y₀ - 1`, about `-3 · 10⁻⁸`. -/
theorem log_softmax_backward_binary32 :
    let y := LogSoftmax.vec
      (Tensor.ofFn fun i => if i = 0 then 9 else -9 : Tensor Binary32 [3])
    let v : Tensor Binary32 [3] := Tensor.ofFn fun i => if i = 0 then -1 else 0
    ExecFloat.Binary.isFinite
        ((LogSoftmax.vecBackward y v).getScalar 0) = true ∧
      MixedError (ExecFloat.Binary.toReal
          ((LogSoftmax.vecBackward y v).getScalar 0))
        (Real.exp (-2143429 / 2 ^ 46) - 1) (1 / 2 ^ 22) (1 / 2 ^ 148) := by
  log_softmax_backward_first (-2143429 / 2 ^ 46)

/-- The same in bfloat16 at the logits `[4, -4, -4]`, where `y₀ = -11 / 2¹⁴`: relative error at
most `2⁻⁶` and absolute error at most `2⁻¹³²` against `exp y₀ - 1`. -/
theorem log_softmax_backward_bfloat16 :
    let y := LogSoftmax.vec
      (Tensor.ofFn fun i => if i = 0 then 4 else -4 : Tensor BFloat16 [3])
    let v : Tensor BFloat16 [3] := Tensor.ofFn fun i => if i = 0 then -1 else 0
    ExecFloat.Binary.isFinite
        ((LogSoftmax.vecBackward y v).getScalar 0) = true ∧
      MixedError (ExecFloat.Binary.toReal
          ((LogSoftmax.vecBackward y v).getScalar 0))
        (Real.exp (-11 / 2 ^ 14) - 1) (1 / 2 ^ 6) (1 / 2 ^ 132) := by
  log_softmax_backward_first (-11 / 2 ^ 14)

end TorchLean.Roundoff
