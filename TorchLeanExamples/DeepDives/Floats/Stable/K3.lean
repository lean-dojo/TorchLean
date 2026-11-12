/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.RuntimeApprox.Grid
public import TorchLean.Verify.Stable.Once
public import TorchLeanExamples.DeepDives.Floats.LibraryBugs
import Mathlib.Analysis.Complex.ExponentialBounds


/-!
# The k3 estimator rounded once

TRL's GRPO loss estimates a KL divergence by `exp d - 1 - d` at the log-ratio `d = lr - lp` of
the reference and policy log-probabilities of a token (`LossBugs.KLEstimator`). The estimator is
positive at every nonzero `d`. `k3` is the estimator as a target of one exponential.

In every IEEE binary format, the program `k3.prog` that rounds the estimator once
* has the certificate of `Once.claim` on every pair of inputs (`cert_k3`);
* returns a positive result wherever the estimator lies in the band `(η, Θ)` (`k3_pos`).
In every configured binary format it returns `+0` at finite inputs with `lp = lr` (`k3_zero`).

TRL's formula and its rewrite with `expm1`, as programs of `(lp, lr)` (`libraryK3Prog`,
`expm1K3Prog`), read as the estimator over ℝ. In every IEEE format, `u` being its unit roundoff:
* TRL's `exp (lr - lp) - (lr - lp) - 1` returns `-u` wherever the computed log-ratio lies in
  `(u/2, u)`, if the exponent bias is at least `2` (`libraryK3_eq`), so it is rejected at those
  inputs where the estimator lies in the band (`libraryK3_rejected`);
* the rewrite `expm1 (lr - lp) - (lr - lp)` returns `0` wherever the computed log-ratio is finite
  and below `u` in magnitude (`k3_eq_zero`), so it is rejected at those inputs where the estimator
  lies in the band (`k3_rejected`);
* if the exponent bias is at least `f + 4`, for the fraction width `f`, both are rejected at
  `lp = 0`, `lr = u - u²` (`exists_rejected`).
In binary32 TRL's formula is also rejected at `lp = -1`, `lr = -1 + 2⁻¹²`, where it returns `0`
(`libraryK3_rejected_binary32`).

Away from zero the rewrite is accurate. On every nonempty set of IEEE formats where `0 < δ ≤ 1`,
`0 < θ`, `28 * u / δ + θ < 1`, `(1 + u) / θ ≤ c` and `c * η < e - 2`, it has the certificate of
`k3Claim δ θ c` (`cert_expm1K3`): at the finite inputs with `δ ≤ |lr - lp| ≤ 1` its result is
finite and within the mixed budget `(u, k3Budget)` of the estimator, and where the estimator lies
in the band `(c * η, Θ)` that budget is at most `28 * u / δ + θ` times the estimator. With
`δ = 64 * u`, `θ = 1/4` and `c = 5` the grade is `11/16` on the IEEE formats with `u ≤ 2⁻⁶`
(`cert_expm1K3_coarse`).
-/

@[expose] public section

namespace TorchLean.Examples.Stable

open FloatLib.Floats FloatLib.Floats.Formats.BinaryInterchange
open TorchLean.Stable
open Arith (Val isFinite toReal valid)
open LossBugs.KLEstimator (exact libraryK3)

/-- The k3 estimator `exp d - 1 - d` at `d = lr - lp` of the inputs `(lp, lr)`, the ratio
`(exp d - 1 - d) / 1`. -/
noncomputable def k3 : Target 2 :=
  Target.ofExp (fun v => exact (v 1 - v 0)) Set.univ (fun _ => 1) (fun q => -1 - (q 1 - q 0))
    (fun _ => 0) (fun _ => 1) (fun q => q 1 - q 0) fun q _ => by
      simp only [exact]
      push_cast
      ring

/-- The k3 estimator rounded once has the certificate of `Once.claim` on every pair of inputs in
every IEEE binary format. -/
theorem cert_k3 : Cert k3.prog k3.F {d | valid d} (Once.claim k3.F fun _ => k3.D) := by
  have h1 := Real.exp_one_gt_d9
  have h2 := Real.exp_one_lt_d9
  have hF : k3.F (fun i => ((![0, 1] : Fin 2 → SignType) i : ℝ)) = Real.exp 1 - 2 := by
    norm_num [k3, Target.ofExp, exact]
    ring
  refine k3.computes_prog.cert ![0, 1] (Set.mem_univ _) ?_ ?_ <;>
    rw [hF, abs_of_pos] <;> linarith

/-- In every configured binary format, the k3 estimator rounded once returns `+0` at finite inputs
with `lp = lr`. -/
theorem k3_zero {d : Binary.Desc} {x : Fin 2 → Val d} (hx : ∀ i, isFinite (x i) = true)
    (h : toReal (x 0) = toReal (x 1)) :
    Binary.toModel (k3.prog.float d x) = Model.zero d.format false :=
  k3.prog_zero ⟨hx, Set.mem_univ _⟩ (by
    show exact (toReal (x 1) - toReal (x 0)) = 0
    rw [h, sub_self, exact, Real.exp_zero]
    ring)

/-- The estimator is positive wherever it lies in the band of a binary format. -/
theorem pos_of_band {d : Binary.Desc} {v : Fin 2 → ℝ} (hb : |k3.F v| ∈ Once.band d) :
    0 < k3.F v := by
  have h0 : 0 ≤ k3.F v := by
    show 0 ≤ exact (v 1 - v 0)
    have := Real.add_one_le_exp (v 1 - v 0)
    unfold exact
    linarith
  exact lt_of_le_of_ne h0 (abs_pos.mp ((Model.underflowErrorAt_pos d.format).trans hb.1)).symm

/-- In every IEEE binary format, the k3 estimator rounded once is positive at finite inputs where
the estimator lies in the band `(η, Θ)`. -/
theorem k3_pos {d : Binary.Desc} (hd : valid d) {x : Fin 2 → Val d}
    (hx : ∀ i, isFinite (x i) = true) (hb : |k3.F (toReal ∘ x)| ∈ Once.band d) :
    0 < toReal (k3.prog.float d x) := by
  obtain ⟨-, hs, -⟩ := cert_k3.toBounds.excludes hd ⟨hx, Set.mem_univ _⟩ hb
  have hF := pos_of_band hb
  nlinarith

/-- TRL's estimator `exp (lr - lp) - (lr - lp) - 1` as a program of `(lp, lr)`. -/
noncomputable def libraryK3Prog : Prog Binary.Desc 2 :=
  Prog.ofContext fun x => libraryK3 (x 0) (x 1)

/-- The rewrite `expm1 (lr - lp) - (lr - lp)` as a program of `(lp, lr)`. -/
noncomputable def expm1K3Prog : Prog Binary.Desc 2 :=
  Prog.ofContext fun x => LossBugs.KLEstimator.k3 (x 0) (x 1)

section Rounding

open FloatLib FloatLib.Floats.Formats FloatLib.Floats.Formats.Flocq
open FloatLib.Floats.Formats.BinaryInterchange.Model
open FloatLib.Floats.Formats.BinaryInterchange.Model.Transcendentals.Certified (overflowThreshold)

section Exec

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
    [ExecFloat.ModelCodec plan (Model format) code]

local notation "Value" => ExecFloat (Configured.Family format code plan)

/-- In every IEEE format whose exponent bias is at least `2`, TRL's `exp (lr - lp) - (lr - lp) - 1`
is finite and equal to `-u`, `u` being the unit roundoff, at every `lp`, `lr` whose computed
difference lies in `(u/2, u)`. -/
theorem libraryK3_eq (hformat : format.isIEEE = true) (hbias : 2 ≤ format.exponentBias)
    {lp lr : Value} (h1 : unitRoundoffAt format / 2 < ExecFloat.Binary.toReal (lr - lp))
    (h2 : ExecFloat.Binary.toReal (lr - lp) < unitRoundoffAt format) :
    ExecFloat.Binary.isFinite (libraryK3 lp lr) = true ∧
      ExecFloat.Binary.toReal (libraryK3 lp lr) = -unitRoundoffAt format := by
  have hu0 := unitRoundoffAt_pos format
  have hu := Grid.unitRoundoff_le (fmt := format)
  have hΩ := one_le_toReal_posMaxFinite format
  have hπ := pi_lt_overflowThreshold format hformat
  rw [← Transcendentals.CorrectlyRounded.cast_overflowThreshold] at hπ
  have htf : ExecFloat.Binary.isFinite (lr - lp) = true := by
    by_contra hn
    have h0 : ExecFloat.Binary.toReal (lr - lp) = 0 :=
      Model.toReal_eq_zero_of_isFinite_eq_false (Bool.eq_false_iff.mpr hn)
    linarith
  have hg : genericFormat Numerics.binaryRadix (fexpOf format)
      (ExecFloat.Binary.toReal (lr - lp)) :=
    toReal_genericFormat_of_isFinite _ htf
  have ht : ExecFloat.Binary.toReal (lr - lp) ≤
      unitRoundoffAt format - unitRoundoffAt format ^ 2 := by
    have h : |ExecFloat.Binary.toReal (lr - lp)| ≤
        (1 - unitRoundoffAt format) * unitRoundoffAt format :=
      Grid.abs_le_of_lt (e := -Int.ofNat (format.fracWidth + 1)) hg
        (by rw [abs_of_pos (by linarith)]; exact h2)
    rw [abs_of_pos (by linarith)] at h
    nlinarith
  have hexp : Real.exp (ExecFloat.Binary.toReal (lr - lp)) < overflowThreshold format := by
    have := Real.exp_le_exp.mpr (show ExecFloat.Binary.toReal (lr - lp) ≤ 1 by linarith)
    linarith [Real.exp_one_lt_d9, Real.pi_gt_three]
  have hef := ExecFloat.Binary.CorrectlyRounded.isFinite_exp (budget := {}) htf hexp
  have he : ExecFloat.Binary.toReal (ExecFloat.Binary.CorrectlyRounded.exp (lr - lp)) = 1 := by
    rw [ExecFloat.Binary.CorrectlyRounded.toReal_exp_eq_roundAt (budget := {}) htf hexp]
    exact Grid.roundAt_exp_eq_one (by linarith) ht
  have hsf : ExecFloat.Binary.isFinite
      (ExecFloat.Binary.CorrectlyRounded.exp (lr - lp) - (lr - lp)) = true :=
    ExecFloat.Binary.isFinite_sub_of_mixedError hef htf (.refl _)
      (by rw [he, abs_of_pos (by linarith)]; linarith)
  have hs : ExecFloat.Binary.toReal
      (ExecFloat.Binary.CorrectlyRounded.exp (lr - lp) - (lr - lp)) =
        1 - unitRoundoffAt format := by
    rw [ExecFloat.Binary.toReal_sub_eq_roundAt hsf, he]
    exact Grid.roundAt_eq_one_sub hbias (by linarith) (by linarith)
  have hrf : ExecFloat.Binary.isFinite
      (ExecFloat.Binary.CorrectlyRounded.exp (lr - lp) - (lr - lp) - 1) = true :=
    ExecFloat.Binary.isFinite_sub_of_mixedError hsf (ExecFloat.Binary.isFinite_one hformat)
      (.refl _) (by rw [hs, ExecFloat.Binary.toReal_one hformat, sub_sub_cancel_left, abs_neg,
        abs_of_pos hu0]; linarith)
  unfold libraryK3
  rw [ConfiguredBinary.exp_of_isIEEE hformat]
  refine ⟨hrf, ?_⟩
  rw [ExecFloat.Binary.toReal_sub_eq_roundAt hrf, hs, ExecFloat.Binary.toReal_one hformat,
    sub_sub_cancel_left]
  exact roundAt_of_genericFormat _ (generic_format_neg _ (Grid.format_unitRoundoff hbias))

/-- In every IEEE format, the rewrite `expm1 (lr - lp) - (lr - lp)` is finite and equal to `0` at
every `lp`, `lr` whose computed difference is finite and below the unit roundoff `u` in
magnitude. -/
theorem k3_eq_zero (hformat : format.isIEEE = true) {lp lr : Value}
    (ht : ExecFloat.Binary.isFinite (lr - lp) = true)
    (h : |ExecFloat.Binary.toReal (lr - lp)| < unitRoundoffAt format) :
    ExecFloat.Binary.isFinite (LossBugs.KLEstimator.k3 lp lr) = true ∧
      ExecFloat.Binary.toReal (LossBugs.KLEstimator.k3 lp lr) = 0 := by
  have hu := Grid.unitRoundoff_le (fmt := format)
  have hΩ := one_le_toReal_posMaxFinite format
  have hπ := pi_lt_overflowThreshold format hformat
  rw [← Transcendentals.CorrectlyRounded.cast_overflowThreshold] at hπ
  have hexp : Real.exp (ExecFloat.Binary.toReal (lr - lp)) - 1 < overflowThreshold format := by
    have := Real.exp_le_exp.mpr (show ExecFloat.Binary.toReal (lr - lp) ≤ 1 by
      linarith [le_abs_self (ExecFloat.Binary.toReal (lr - lp))])
    linarith [Real.exp_one_lt_d9, Real.pi_gt_three]
  have hmf := ExecFloat.Binary.CorrectlyRounded.isFinite_expMinus1 (budget := {}) ht hexp
  have hm : ExecFloat.Binary.toReal (ExecFloat.Binary.CorrectlyRounded.expMinus1 (lr - lp)) =
      ExecFloat.Binary.toReal (lr - lp) := by
    rw [ExecFloat.Binary.CorrectlyRounded.toReal_expMinus1_eq_roundAt (budget := {}) ht hexp]
    exact Grid.roundAt_expm1 (toReal_genericFormat_of_isFinite _ ht) h
  have hf : ExecFloat.Binary.isFinite
      (ExecFloat.Binary.CorrectlyRounded.expMinus1 (lr - lp) - (lr - lp)) = true :=
    ExecFloat.Binary.isFinite_sub_of_mixedError hmf ht (.refl _)
      (by rw [hm, sub_self, abs_zero, mul_zero, add_zero]; linarith)
  unfold LossBugs.KLEstimator.k3
  rw [ConfiguredBinary.expm1_of_isIEEE hformat]
  refine ⟨hf, ?_⟩
  rw [ExecFloat.Binary.toReal_sub_eq_roundAt hf, hm, sub_self, roundAt_zero]

end Exec

/-- In every IEEE format, the estimator at a log-ratio `d` in `[0, 1]` with `η < d²/2` lies in
the band `(η, Θ)`. -/
theorem exact_mem_band {f : Binary.Desc} (hf : valid f) {d : ℝ} (h0 : 0 ≤ d) (h1 : d ≤ 1)
    (hη : Model.underflowErrorAt f.format < d ^ 2 / 2) : |exact d| ∈ Once.band f := by
  have hf' : f.format.isIEEE = true := hf
  have hlo := Real.quadratic_le_exp_of_nonneg h0
  have hhi := Real.abs_exp_sub_one_sub_id_le (x := d) (by rw [abs_of_nonneg h0]; exact h1)
  have hΘ := Model.Transcendentals.CorrectlyRounded.one_lt_overflowThreshold hf'
  have hpos : 0 < exact d := by
    unfold exact
    linarith [Model.underflowErrorAt_pos f.format]
  rw [abs_of_pos hpos]
  refine ⟨hη.trans_le (by unfold exact; linarith), ?_⟩
  have hle : exact d ≤ d ^ 2 := (le_abs_self _).trans hhi
  exact hle.trans_lt ((pow_le_one₀ h0 h1).trans_lt hΘ)

/-- In every IEEE format whose exponent bias is at least `2`, TRL's estimator is rejected at
every finite input whose computed log-ratio lies in `(u/2, u)` and whose estimator lies in the
band. -/
theorem libraryK3_rejected {d : Binary.Desc} (hd : valid d) (hbias : 2 ≤ d.format.exponentBias)
    {lp lr : ExecFloat (Configured.Family d.format d.code d.plan)}
    (hp : ExecFloat.Binary.isFinite lp = true) (hr : ExecFloat.Binary.isFinite lr = true)
    (h1 : unitRoundoffAt d.format / 2 < ExecFloat.Binary.toReal (lr - lp))
    (h2 : ExecFloat.Binary.toReal (lr - lp) < unitRoundoffAt d.format)
    (hb : |exact (ExecFloat.Binary.toReal lr - ExecFloat.Binary.toReal lp)| ∈ Once.band d) :
    Rejected libraryK3Prog k3.F d ![lp, lr] := by
  have hb' : |k3.F (Arith.toReal (f := d) ∘ ![lp, lr])| ∈ Once.band d := hb
  have hv : Arith.toReal (f := d) (libraryK3Prog.float d ![lp, lr]) = -unitRoundoffAt d.format :=
    (libraryK3_eq hd hbias h1 h2).2
  have hm : Arith.toReal (f := d) (libraryK3Prog.float d ![lp, lr]) *
      k3.F (Arith.toReal (f := d) ∘ ![lp, lr]) ≤ 0 := by
    rw [hv]
    exact mul_nonpos_of_nonpos_of_nonneg (neg_nonpos.mpr (unitRoundoffAt_pos _).le)
      (pos_of_band hb').le
  exact Rejected.of_mul_nonpos (Fin.forall_fin_two.mpr ⟨hp, hr⟩)
    (LossBugs.KLEstimator.k3_real _ _).1 hb' hm

/-- In every IEEE format, the rewrite `expm1 (lr - lp) - (lr - lp)` is rejected at every finite
input whose computed log-ratio is finite and below `u` in magnitude and whose estimator lies in
the band. -/
theorem k3_rejected {d : Binary.Desc} (hd : valid d)
    {lp lr : ExecFloat (Configured.Family d.format d.code d.plan)}
    (hp : ExecFloat.Binary.isFinite lp = true) (hr : ExecFloat.Binary.isFinite lr = true)
    (ht : ExecFloat.Binary.isFinite (lr - lp) = true)
    (h : |ExecFloat.Binary.toReal (lr - lp)| < unitRoundoffAt d.format)
    (hb : |exact (ExecFloat.Binary.toReal lr - ExecFloat.Binary.toReal lp)| ∈ Once.band d) :
    Rejected expm1K3Prog k3.F d ![lp, lr] := by
  have hb' : |k3.F (Arith.toReal (f := d) ∘ ![lp, lr])| ∈ Once.band d := hb
  have hv : Arith.toReal (f := d) (expm1K3Prog.float d ![lp, lr]) = 0 := (k3_eq_zero hd ht h).2
  have hm : Arith.toReal (f := d) (expm1K3Prog.float d ![lp, lr]) *
      k3.F (Arith.toReal (f := d) ∘ ![lp, lr]) ≤ 0 := by
    rw [hv, zero_mul]
  exact Rejected.of_mul_nonpos (Fin.forall_fin_two.mpr ⟨hp, hr⟩)
    (LossBugs.KLEstimator.k3_real _ _).2 hb' hm

/-- In every IEEE format whose exponent bias is at least `f + 4`, `f` being the fraction width,
there is a finite `lr` of value `u - u²` at which TRL's formula and the rewrite are both rejected
with `lp = 0`. -/
theorem exists_rejected {d : Binary.Desc} (hd : valid d)
    (hbias : d.format.fracWidth + 4 ≤ d.format.exponentBias) :
    ∃ lr : ExecFloat (Configured.Family d.format d.code d.plan),
      ExecFloat.Binary.toReal lr = unitRoundoffAt d.format - unitRoundoffAt d.format ^ 2 ∧
        Rejected libraryK3Prog k3.F d
            ![(0 : ExecFloat (Configured.Family d.format d.code d.plan)), lr] ∧
          Rejected expm1K3Prog k3.F d
            ![(0 : ExecFloat (Configured.Family d.format d.code d.plan)), lr] := by
  have hd' : d.format.isIEEE = true := hd
  have hu0 := unitRoundoffAt_pos d.format
  have hu := Grid.unitRoundoff_le (fmt := d.format)
  have hfe : fexpOf d.format (-Int.ofNat (d.format.fracWidth + 1)) =
      -Int.ofNat (d.format.fracWidth + 1) + -Int.ofNat (d.format.fracWidth + 1) := by
    simp only [fexpOf, fltExp, FloatFormat.minSubnormalExponent, FloatFormat.minNormalExponent,
      Int.ofNat_eq_natCast]
    omega
  have hpred : pred (β := Numerics.binaryRadix) (fexp := fexpOf d.format)
      (unitRoundoffAt d.format) = unitRoundoffAt d.format - unitRoundoffAt d.format ^ 2 := by
    rw [unitRoundoffAt, pred_bpow, hfe, bpow.add_exp, sq]
  have hrep : genericFormat Numerics.binaryRadix (fexpOf d.format)
      (unitRoundoffAt d.format - unitRoundoffAt d.format ^ 2) := by
    rw [← hpred]
    exact generic_format_pred (Grid.format_unitRoundoff (by omega))
  have hy0 : 0 < unitRoundoffAt d.format - unitRoundoffAt d.format ^ 2 := by nlinarith
  have hΘ := Model.Transcendentals.CorrectlyRounded.one_lt_overflowThreshold hd'
  obtain ⟨hfin, hval⟩ := Elementary.roundSigned_eq_roundAt hd'
    (c := fun q : ℚ => cmp (unitRoundoffAt d.format - unitRoundoffAt d.format ^ 2) (q : ℝ))
    (fun _ => rfl) (by rw [abs_of_pos hy0]; nlinarith)
  have he := ExecFloat.Binary.toModel_ofModel (plan := d.plan) (code := d.code)
    (Elementary.roundSigned d.format
      fun q : ℚ => cmp (unitRoundoffAt d.format - unitRoundoffAt d.format ^ 2) (q : ℝ))
  set v : ExecFloat (Configured.Family d.format d.code d.plan) :=
    ExecFloat.Binary.ofModel (plan := d.plan) (code := d.code) (Elementary.roundSigned d.format
      fun q : ℚ => cmp (unitRoundoffAt d.format - unitRoundoffAt d.format ^ 2) (q : ℝ)) with hv
  have hvr : ExecFloat.Binary.toReal v = unitRoundoffAt d.format - unitRoundoffAt d.format ^ 2 := by
    show Model.toReal (ExecFloat.Binary.toModel v) = _
    rw [hv, he, hval, roundAt_of_genericFormat _ hrep]
  have hvf : ExecFloat.Binary.isFinite v = true := by
    show Model.isFinite (ExecFloat.Binary.toModel v) = true
    rw [hv, he]
    exact hfin
  have htf : ExecFloat.Binary.isFinite (v - 0) = true :=
    ExecFloat.Binary.isFinite_sub_of_mixedError hvf ExecFloat.Binary.isFinite_zero (.refl _)
      (by rw [ExecFloat.Binary.toReal_zero, sub_zero, hvr, abs_of_pos hy0]
          nlinarith [one_le_toReal_posMaxFinite d.format])
  have ht : ExecFloat.Binary.toReal (v - 0) =
      unitRoundoffAt d.format - unitRoundoffAt d.format ^ 2 := by
    rw [ExecFloat.Binary.toReal_sub_eq_roundAt htf, ExecFloat.Binary.toReal_zero, sub_zero, hvr,
      roundAt_of_genericFormat _ hrep]
  have hη : Model.underflowErrorAt d.format ≤ unitRoundoffAt d.format ^ 2 / 4 := by
    refine (Model.underflowErrorAt_le_zpow _ (e := -(2 * (d.format.fracWidth : ℤ) + 4)) (by
      simp only [FloatFormat.minSubnormalExponent, FloatFormat.minNormalExponent,
        Int.ofNat_eq_natCast]
      omega)).trans_eq ?_
    have h4 : (2 : ℝ) ^ (-2 : ℤ) = 1 / 4 := by norm_num
    rw [unitRoundoffAt_eq, show -(2 * (d.format.fracWidth : ℤ) + 4) =
      -((d.format.fracWidth : ℤ) + 1) + -((d.format.fracWidth : ℤ) + 1) + -2 by ring,
      zpow_add₀ two_ne_zero, zpow_add₀ two_ne_zero, h4]
    ring
  have hb : |exact (ExecFloat.Binary.toReal v -
      ExecFloat.Binary.toReal (0 : ExecFloat (Configured.Family d.format d.code d.plan)))| ∈
      Once.band d := by
    rw [ExecFloat.Binary.toReal_zero, sub_zero, hvr]
    refine exact_mem_band hd hy0.le (by nlinarith) (hη.trans_lt ?_)
    have h5 : 0 < unitRoundoffAt d.format ^ 2 * ((1 - unitRoundoffAt d.format) ^ 2 / 2 - 1 / 4) :=
      mul_pos (pow_pos hu0 2) (by nlinarith)
    nlinarith
  refine ⟨v, hvr, libraryK3_rejected hd (by omega) ExecFloat.Binary.isFinite_zero hvf
    (by rw [ht]; nlinarith) (by rw [ht]; nlinarith) hb,
    k3_rejected hd ExecFloat.Binary.isFinite_zero hvf htf
      (by rw [ht, abs_of_pos hy0]; nlinarith) hb⟩

/-- In binary32 TRL's estimator is rejected at `lp = -1`, `lr = -1 + 2⁻¹²`, where it decodes to
`0` (`LossBugs.KLEstimator.binary32`). -/
theorem libraryK3_rejected_binary32 :
    Rejected libraryK3Prog k3.F Binary.binary32 ![(-1 : Binary32), (-1 + 1 / 4096 : Binary32)] := by
  have h1 : ExecFloat.Binary.toRat? (-1 : Binary32) = some (-1) := by decide +kernel
  have h2 := LossBugs.KLEstimator.binary32_input
  have hd : ExecFloat.Binary.toReal (-1 + 1 / 4096 : Binary32) -
      ExecFloat.Binary.toReal (-1 : Binary32) = 1 / 4096 := by
    rw [ExecFloat.Binary.toReal_of_toRat?_eq_some h1, ExecFloat.Binary.toReal_of_toRat?_eq_some h2]
    norm_num
  have hb : |k3.F (Arith.toReal (f := Binary.binary32) ∘
      ![(-1 : Binary32), (-1 + 1 / 4096 : Binary32)])| ∈ Once.band Binary.binary32 := by
    show |exact (ExecFloat.Binary.toReal (-1 + 1 / 4096 : Binary32) -
      ExecFloat.Binary.toReal (-1 : Binary32))| ∈ _
    rw [hd]
    exact exact_mem_band FloatFormat.isIEEE_binary32 (by norm_num) (by norm_num)
      ((Model.underflowErrorAt_le_zpow _ (e := -26) (by decide)).trans_lt (by norm_num))
  have hv : Arith.toReal (f := Binary.binary32) (libraryK3Prog.float Binary.binary32
      ![(-1 : Binary32), (-1 + 1 / 4096 : Binary32)]) = 0 :=
    (ExecFloat.Binary.toReal_of_toRat?_eq_some LossBugs.KLEstimator.binary32.1).trans Rat.cast_zero
  refine Rejected.of_mul_nonpos (Fin.forall_fin_two.mpr
    ⟨Model.Algebraic.isFinite_of_toRat?_eq_some h1, Model.Algebraic.isFinite_of_toRat?_eq_some h2⟩)
    (LossBugs.KLEstimator.k3_real _ _).1 hb ?_
  rw [hv, zero_mul]

/-! ## Accuracy of the rewrite away from zero -/

/-- The absolute part of the budget of `k3_error` at `d = v 1 - v 0` in format `f`, that is
`(1 + u) * (b * |exp d - 1| + η + u * |d|)` with
`b = (1 + u) * ((1 + |d|) * exp (u * |d|) * u) + u`. -/
noncomputable def k3Budget (f : Binary.Desc) (v : Fin 2 → ℝ) : ℝ :=
  (1 + unitRoundoffAt f.format) *
    (((1 + unitRoundoffAt f.format) * ((1 + |v 1 - v 0|) *
      Real.exp (unitRoundoffAt f.format * |v 1 - v 0|) * unitRoundoffAt f.format) +
        unitRoundoffAt f.format) * |Real.exp (v 1 - v 0) - 1| + underflowErrorAt f.format +
      unitRoundoffAt f.format * |v 1 - v 0|)

/-- The claim of `cert_expm1K3`: the inputs with `δ ≤ |lr - lp| ≤ 1`, a finite result at every
answer, the band `(c * η, Θ)`, the budget `(u, k3Budget)` and the grade `28 * u / δ + θ`. -/
noncomputable def k3Claim (δ θ c : Binary.Desc → ℝ) : Claim Binary.Desc 2 where
  D f := {v | δ f ≤ |v 1 - v 0| ∧ |v 1 - v 0| ≤ 1}
  fin _ := Set.univ
  band f := Set.Ioo (c f * underflowErrorAt f.format) (overflowThreshold f.format)
  r f _ := unitRoundoffAt f.format
  a := k3Budget
  κ f _ := 28 * unitRoundoffAt f.format / δ f + θ f

/-- With `0 < u ≤ 1/32` and `|d| ≤ 1`, the factor of `|exp d - 1|` in the budget of `k3_error` is
at most `3.13 * u`. -/
private theorem factor_le {U d : ℝ} (hU0 : 0 < U) (hU : U ≤ 1 / 32) (hd : |d| ≤ 1) :
    (1 + U) * ((1 + |d|) * Real.exp (U * |d|) * U) + U ≤ 313 / 100 * U := by
  have ht0 := abs_nonneg d
  have hut : U * |d| ≤ 1 / 32 := by nlinarith
  have hs0 : 0 ≤ U * |d| := by positivity
  have hs2 : (U * |d|) ^ 2 ≤ 1 / 1024 := by nlinarith
  have hE : Real.exp (U * |d|) ≤ 1057 / 1024 := by
    have h := Real.abs_exp_sub_one_sub_id_le (x := U * |d|) (by rw [abs_of_nonneg hs0]; linarith)
    have h' := le_abs_self (Real.exp (U * |d|) - 1 - U * |d|)
    linarith
  have h1 : (1 + U) * (1 + |d|) ≤ 33 / 16 := by nlinarith
  have h2 : (1 + U) * (1 + |d|) * Real.exp (U * |d|) ≤ 33 / 16 * (1057 / 1024) :=
    mul_le_mul h1 hE (Real.exp_pos _).le (by norm_num)
  have h3 : (1 + U) * ((1 + |d|) * Real.exp (U * |d|) * U) =
      (1 + U) * (1 + |d|) * Real.exp (U * |d|) * U := by ring
  rw [h3]
  nlinarith

/-- With `0 < u ≤ 1/32`, `η ≤ u`, `1 ≤ Ω`, `3 < Θ` and `|d| ≤ 1`, the premises of `k3_error`
hold at the log-ratio `d`. -/
private theorem premises {U η Ω Θ d : ℝ} (hU0 : 0 < U) (hU : U ≤ 1 / 32) (hη : η ≤ U)
    (hΩ : 1 ≤ Ω) (hΘ : 3 < Θ) (hd : |d| ≤ 1) :
    |d| ≤ Ω ∧ (1 + (1 + |d|) * Real.exp (U * |d|) * U) * |Real.exp d - 1| < Θ ∧
      |Real.exp d - 1 - d| + (((1 + U) * ((1 + |d|) * Real.exp (U * |d|) * U) + U) *
        |Real.exp d - 1| + η + U * |d|) ≤ Ω := by
  have ht0 := abs_nonneg d
  have hb := factor_le hU0 hU hd
  have hm := Real.abs_exp_sub_one_le hd
  have hm0 := abs_nonneg (Real.exp d - 1)
  have hP0 : 0 ≤ (1 + |d|) * Real.exp (U * |d|) * U := by positivity
  have hP : (1 + |d|) * Real.exp (U * |d|) * U ≤ 7 / 100 := by nlinarith
  have hq := Grid.abs_exp_sub_quadratic_le hd
  have hF : |Real.exp d - 1 - d| ≤ 13 / 18 := by
    have h1 := abs_sub_abs_le_abs_sub (Real.exp d - 1 - d) (d ^ 2 / 2)
    have h2 : |d ^ 2 / 2| ≤ 1 / 2 := by
      rw [abs_of_nonneg (by positivity), ← sq_abs]
      nlinarith
    have h3 : |d| ^ 3 ≤ 1 := pow_le_one₀ ht0 hd
    rw [show Real.exp d - 1 - d - d ^ 2 / 2 = Real.exp d - 1 - d - d ^ 2 / 2 from rfl] at hq
    linarith
  refine ⟨hd.trans hΩ, ?_, ?_⟩
  · have : (1 + (1 + |d|) * Real.exp (U * |d|) * U) * |Real.exp d - 1| ≤ (1 + 7 / 100) * 2 :=
      mul_le_mul (by linarith) (by linarith) hm0 (by linarith)
    linarith
  · have h1 : ((1 + U) * ((1 + |d|) * Real.exp (U * |d|) * U) + U) * |Real.exp d - 1| ≤
        313 / 100 * U * 2 :=
      mul_le_mul hb (by linarith) hm0 (by positivity)
    have h2 : U * |d| ≤ U := by nlinarith
    linarith

/-- With `0 < u ≤ 1/32`, `0 ≤ η`, `0 < δ ≤ |d| ≤ 1`, `0 < θ`, `1 + u ≤ c * θ` and
`c * η < |exp d - 1 - d|`, the budget of `k3_error` is at most
`(28 * u / δ + θ) * |exp d - 1 - d|`. -/
private theorem budget_le {U η δ θ c d : ℝ} (hU0 : 0 < U) (hU : U ≤ 1 / 32) (hη : 0 ≤ η)
    (hδ : 0 < δ) (hδd : δ ≤ |d|) (hd : |d| ≤ 1) (hθ : 0 < θ) (hc : 1 + U ≤ c * θ)
    (hF : c * η < |exact d|) :
    U * |exact d| + (1 + U) * (((1 + U) * ((1 + |d|) * Real.exp (U * |d|) * U) + U) *
      |Real.exp d - 1| + η + U * |d|) ≤ (28 * U / δ + θ) * |exact d| := by
  have ht0 := abs_nonneg d
  have hb := factor_le hU0 hU hd
  have hm := Real.abs_exp_sub_one_le hd
  have hq := Grid.abs_exp_sub_quadratic_le hd
  have hF5 : 5 / 18 * |d| ^ 2 ≤ exact d := by
    have h1 := neg_abs_le (Real.exp d - 1 - d - d ^ 2 / 2)
    have h2 : |d| ^ 3 ≤ |d| ^ 2 := by nlinarith [pow_two_nonneg |d|]
    have h3 : d ^ 2 = |d| ^ 2 := (sq_abs d).symm
    unfold exact
    nlinarith
  have hpos : 0 < exact d := lt_of_lt_of_le (by nlinarith) hF5
  rw [abs_of_pos hpos] at hF ⊢
  have hη' : (1 + U) * η ≤ θ * exact d := by
    calc (1 + U) * η ≤ c * θ * η := mul_le_mul_of_nonneg_right hc hη
      _ = θ * (c * η) := by ring
      _ ≤ θ * exact d := mul_le_mul_of_nonneg_left hF.le hθ.le
  have hq0 : 0 < U / δ := div_pos hU0 hδ
  have hUq : U / δ * δ = U := div_mul_cancel₀ U hδ.ne'
  have h28 : 28 * U / δ = 28 * (U / δ) := mul_div_assoc _ _ _
  rw [h28]
  have h1 : U * exact d ≤ U / δ * exact d := by
    refine mul_le_mul_of_nonneg_right ?_ hpos.le
    rw [le_div_iff₀ hδ]
    nlinarith [hδd.trans hd]
  have h2 : U * |d| ≤ U / δ * |d| ^ 2 := by
    calc U * |d| = U / δ * δ * |d| := by rw [hUq]
      _ ≤ U / δ * |d| * |d| := by gcongr
      _ = U / δ * |d| ^ 2 := by ring
  have h3 : ((1 + U) * ((1 + |d|) * Real.exp (U * |d|) * U) + U) * |Real.exp d - 1| ≤
      313 / 100 * U * (2 * |d|) :=
    mul_le_mul hb hm (abs_nonneg _) (by positivity)
  have h4 : (1 + U) * (313 / 100 * U * (2 * |d|) + U * |d|) ≤
      33 / 32 * (726 / 100) * (U / δ * |d| ^ 2) := by
    have h5 : 313 / 100 * U * (2 * |d|) + U * |d| = 726 / 100 * (U * |d|) := by ring
    rw [h5]
    have h6 : 0 ≤ U * |d| := by positivity
    nlinarith
  have h7 : U / δ * |d| ^ 2 ≤ 18 / 5 * (U / δ * exact d) := by nlinarith
  have h8 : (1 + U) * (((1 + U) * ((1 + |d|) * Real.exp (U * |d|) * U) + U) * |Real.exp d - 1| +
      U * |d|) ≤ (1 + U) * (313 / 100 * U * (2 * |d|) + U * |d|) :=
    mul_le_mul_of_nonneg_left (by linarith) (by linarith)
  have h9 : 0 ≤ U / δ * exact d := by positivity
  nlinarith

/-- The rewrite `expm1 (lr - lp) - (lr - lp)` has the certificate of `k3Claim δ θ c` on every
nonempty set `S` of IEEE formats where `0 < δ ≤ 1`, `0 < θ`, `28 * u / δ + θ < 1`,
`(1 + u) / θ ≤ c` and `c * η < e - 2`. -/
theorem cert_expm1K3 {S : Set Binary.Desc} {δ θ c : Binary.Desc → ℝ} (hS : S.Nonempty)
    (h : ∀ f ∈ S, valid f ∧ 0 < δ f ∧ δ f ≤ 1 ∧ 0 < θ f ∧
      28 * unitRoundoffAt f.format / δ f + θ f < 1 ∧ (1 + unitRoundoffAt f.format) / θ f ≤ c f ∧
        c f * underflowErrorAt f.format < Real.exp 1 - 2) :
    Cert expm1K3Prog k3.F S (k3Claim δ θ c) := by
  have hU : ∀ f ∈ S, unitRoundoffAt f.format ≤ 1 / 32 := fun f hf => by
    obtain ⟨-, hδ, -, hθ, hκ, -, -⟩ := h f hf
    have h16 : unitRoundoffAt f.format < 1 / 16 := by
      have h1 : 28 * unitRoundoffAt f.format / δ f < 1 := by linarith
      rw [div_lt_one hδ] at h1
      linarith [(h f hf).2.2.1]
    rcases le_or_gt 4 f.format.fracWidth with h4 | h4
    · rw [unitRoundoffAt_eq]
      calc (2 : ℝ) ^ (-((f.format.fracWidth : ℤ) + 1)) ≤ 2 ^ (-5 : ℤ) :=
            zpow_le_zpow_right₀ (by norm_num) (by omega)
        _ = 1 / 32 := by norm_num
    · have h5 : (1 : ℝ) / 16 ≤ unitRoundoffAt f.format := by
        rw [unitRoundoffAt_eq]
        calc (1 : ℝ) / 16 = 2 ^ (-4 : ℤ) := by norm_num
          _ ≤ 2 ^ (-((f.format.fracWidth : ℤ) + 1)) := zpow_le_zpow_right₀ (by norm_num) (by omega)
      linarith
  have hc : ∀ f ∈ S, 1 + unitRoundoffAt f.format ≤ c f * θ f := fun f hf => by
    obtain ⟨-, -, -, hθ, -, hc, -⟩ := h f hf
    rwa [div_le_iff₀ hθ] at hc
  refine ⟨⟨fun f _ v _ => (LossBugs.KLEstimator.k3_real (v 0) (v 1)).2, fun f hf y hy => ?_,
    fun _ _ _ _ _ => Set.mem_univ _, fun f hf x hx _ => ?_, fun f hf v hv hb => ?_⟩,
    ⟨hS, fun f hf => (h f hf).1⟩, fun f hf => ?_⟩
  · obtain ⟨-, -, -, hθ, -, -, -⟩ := h f hf
    have h1 : 0 ≤ c f := by
      have := hc f hf
      nlinarith [unitRoundoffAt_pos f.format]
    exact (mul_nonneg h1 (underflowErrorAt_pos f.format).le).trans_lt hy.1
  · have hf' : f.format.isIEEE = true := (h f hf).1
    have hπ := pi_lt_overflowThreshold f.format hf'
    rw [← Transcendentals.CorrectlyRounded.cast_overflowThreshold] at hπ
    exact LossBugs.KLEstimator.k3_error (x 0) (x 1) hf' (hx.1 0) (hx.1 1)
      (premises (unitRoundoffAt_pos _) (hU f hf) Grid.underflowError_le
        (one_le_toReal_posMaxFinite _) (by linarith [Real.pi_gt_three]) hx.2.2)
  · obtain ⟨-, hδ, -, hθ, hκ, -, -⟩ := h f hf
    exact ⟨budget_le (unitRoundoffAt_pos _) (hU f hf) (underflowErrorAt_pos _).le hδ hv.1 hv.2 hθ
      (hc f hf) hb.1, hκ⟩
  · obtain ⟨hv, -, hδ1, -, -, -, hw⟩ := h f hf
    have hx : Arith.toReal ∘ (fun i => Binary.ofSign f ((![0, 1] : Fin 2 → SignType) i)) =
        fun i => (((![0, 1] : Fin 2 → SignType) i : ℝ)) :=
      funext fun i => (Binary.ofSign_spec hv _).2
    have hF : k3.F (fun i => (((![0, 1] : Fin 2 → SignType) i : ℝ))) = Real.exp 1 - 2 := by
      norm_num [k3, Target.ofExp, exact]
      ring
    refine ⟨fun i => Binary.ofSign f ((![0, 1] : Fin 2 → SignType) i),
      ⟨fun i => (Binary.ofSign_spec hv _).1, ?_⟩, ?_⟩
    · rw [hx]
      simpa [k3Claim] using hδ1
    · rw [hx, hF, abs_of_pos (by linarith [Real.exp_one_gt_d9])]
      exact ⟨hw, by linarith [Real.exp_one_lt_d9,
        Model.Transcendentals.CorrectlyRounded.one_lt_overflowThreshold (fmt := f.format) hv]⟩

/-- The rewrite has the certificate of `k3Claim` with `δ = 64 * u`, `θ = 1/4` and `c = 5`, of grade
`28 * u / (64 * u) + 1/4 = 11/16`, on the IEEE formats with `u ≤ 2⁻⁶`. -/
theorem cert_expm1K3_coarse : Cert expm1K3Prog k3.F
    {f | valid f ∧ unitRoundoffAt f.format ≤ 2 ^ (-6 : ℤ)}
    (k3Claim (fun f => 64 * unitRoundoffAt f.format) (fun _ => 1 / 4) fun _ => 5) := by
  refine cert_expm1K3 ⟨Binary.binary32, FloatFormat.isIEEE_binary32, ?_⟩ fun f ⟨hv, hu⟩ => ?_
  · rw [show unitRoundoffAt Binary.binary32.format = 2 ^ (-24 : ℤ) by rw [unitRoundoffAt_eq]; rfl]
    exact zpow_le_zpow_right₀ (by norm_num) (by norm_num)
  have hu0 := unitRoundoffAt_pos f.format
  have hη := Grid.underflowError_le (fmt := f.format)
  have he := Real.exp_one_gt_d9
  rw [show (2 : ℝ) ^ (-6 : ℤ) = 1 / 64 by norm_num] at hu
  refine ⟨hv, by positivity, by linarith, by norm_num, ?_, ?_, by linarith⟩
  · rw [show 28 * unitRoundoffAt f.format / (64 * unitRoundoffAt f.format) = 7 / 16 by
      field_simp; ring]
    norm_num
  · rw [div_le_iff₀ (by norm_num)]
    linarith

end Rounding

end TorchLean.Examples.Stable
