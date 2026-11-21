/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Analysis.Calculus.SmoothSeries
public import Mathlib.Analysis.PSeries
public import TorchLean.Tensor.Special.Gamma

/-!
# The Hurwitz zeta function of real arguments

`torch.special.zeta` and `lax.zeta` compute the Hurwitz zeta function `ζ(s, q) = ∑ₖ (k + q)^(-s)`
(DLMF equation 25.11.1) at real `s` and `q`. `zeta s q` is that series, written by hand. Mathlib's
`hurwitzZeta` takes its shift modulo one, so it agrees with ours only for `q ∈ (0, 1]`, which
`zeta_eq_hurwitzZeta` proves for `s > 1` (in `TorchLean.Tensor.Special.Hurwitz`, so that the
operator table does not import Mathlib's L-series); for a larger `q` the two differ by the first
terms of the series. For `s > 1` the series converges absolutely at every real `q`
(`summable_zeta`), and `zeta` is jointly smooth in `(s, q)` where `s > 1` and `q > 0`
(`contDiffOn_zeta`): past its first term, the series and the series of its partial derivatives of
every order, of the terms `(k + 1 + q) ^ (-(s + j)) (log (k + 1 + q)) ^ i`, converge locally
uniformly. Where it diverges, `zeta` is `tsum`'s junk value `0`; PyTorch 2.13 returns `+∞` at
`s = 1` and a NaN below, which its docstring doesn't state. DLMF defines the series for `ℜ s > 1`
and `q` off the poles `0, -1, -2, …`, with principal powers. For real `s > 1` it is finite and real
at `q > 0`, and at an integer `s` for `q` not an integer, where a term `(k + q)^(-s)` with
`k + q < 0` is the integer power (`Real.rpow_intCast`). At the other real `q`, `Real.rpow` of a
negative base is not the principal power, and `zeta` is a real number that no theorem relates to
the function. The analytically continued function is finite and real at other real points too, as
`ζ(0, q) = 1/2 - q` (DLMF equation 25.11.13); `zeta` does not cover them.

## References

- NIST Digital Library of Mathematical Functions, section 25.11(i), equation 25.11.1, and section
  25.11(v), equation 25.11.13: `https://dlmf.nist.gov/25.11`
- PyTorch 2.13 documentation, `torch.special.zeta`:
  `https://docs.pytorch.org/docs/2.13/special.html#torch.special.zeta`
-/

@[expose] public section

noncomputable section

namespace TorchLean

open Real Set
open scoped ContDiff

/-- The Hurwitz zeta function `ζ(s, q) = ∑ₖ (k + q)^(-s)`, DLMF equation 25.11.1, written by
hand. -/
def zeta (s q : ℝ) : ℝ := ∑' k : ℕ, ((k : ℝ) + q) ^ (-s)

/-- For `s > 1` the series of `zeta s q` converges absolutely at every real `q`, by comparison of
`|(k + q)^(-s)|` with `|k + q|^(-s)` and Mathlib's `Real.summable_one_div_nat_add_rpow`. -/
theorem summable_zeta {s : ℝ} (hs : 1 < s) (q : ℝ) :
    Summable fun k : ℕ => ((k : ℝ) + q) ^ (-s) := by
  refine Summable.of_norm_bounded ((Real.summable_one_div_nat_add_rpow q s).2 hs) fun k => ?_
  rw [Real.norm_eq_abs, one_div, ← Real.rpow_neg (abs_nonneg _)]
  exact Real.abs_rpow_le_abs_rpow _ _

/-- `(k + 1 + q) ^ (-(s + j)) (log (k + 1 + q)) ^ i` at `m = (i, j)` and `p = (s, q)`: the term
`k + 1` of the series of `zeta s q` and its partial derivatives in `(s, q)`. -/
private def zetaTerm (m : ℕ × ℕ) (p : ℝ × ℝ) (k : ℕ) : ℝ :=
  ((k : ℝ) + 1 + p.2) ^ (-(p.1 + m.2)) * log ((k : ℝ) + 1 + p.2) ^ m.1

/-- `zetaTerm m p k` is nonnegative for `p.2 ≥ 0`, where its base is at least `1`. -/
private theorem zetaTerm_nonneg (m : ℕ × ℕ) {p : ℝ × ℝ} (hq : 0 ≤ p.2) (k : ℕ) :
    0 ≤ zetaTerm m p k := by
  have hb : 1 ≤ (k : ℝ) + 1 + p.2 := by linarith [k.cast_nonneg (α := ℝ)]
  exact mul_nonneg (rpow_nonneg (by linarith) _) (pow_nonneg (log_nonneg hb) _)

/-- For `p.2 ≥ 0`, `p.1 ≥ σ` and `0 < δ ≤ σ`, `zetaTerm m p k` is at most a constant times
`(k + 1) ^ (δ - σ)`: `log` is at most a constant times the power `δ`. -/
private theorem zetaTerm_le (m : ℕ × ℕ) {σ δ : ℝ} (hδ : 0 < δ) (hδσ : δ ≤ σ) {p : ℝ × ℝ}
    (hs : σ ≤ p.1) (hq : 0 ≤ p.2) (k : ℕ) :
    zetaTerm m p k ≤ m.1.factorial / δ ^ m.1 * ((k : ℝ) + 1) ^ (δ - σ) := by
  have hk : 1 ≤ (k : ℝ) + 1 := by linarith [k.cast_nonneg (α := ℝ)]
  have hb : (k : ℝ) + 1 ≤ (k : ℝ) + 1 + p.2 := by linarith
  have hb0 : 0 < (k : ℝ) + 1 + p.2 := by linarith
  have hj : (0 : ℝ) ≤ m.2 := m.2.cast_nonneg
  have hlog : log ((k : ℝ) + 1 + p.2) ^ m.1 ≤
      m.1.factorial / δ ^ m.1 * ((k : ℝ) + 1 + p.2) ^ δ := by
    have h := pow_div_factorial_le_exp (δ * log ((k : ℝ) + 1 + p.2))
      (mul_nonneg hδ.le (log_nonneg (hk.trans hb))) m.1
    rw [mul_comm δ, ← rpow_def_of_pos hb0, mul_pow, div_le_iff₀ (by positivity)] at h
    rw [div_mul_eq_mul_div, le_div_iff₀ (by positivity)]
    linarith
  calc zetaTerm m p k
      ≤ ((k : ℝ) + 1 + p.2) ^ (-(p.1 + m.2)) *
          (m.1.factorial / δ ^ m.1 * ((k : ℝ) + 1 + p.2) ^ δ) :=
        mul_le_mul_of_nonneg_left hlog (rpow_nonneg hb0.le _)
    _ = m.1.factorial / δ ^ m.1 * ((k : ℝ) + 1 + p.2) ^ (δ - (p.1 + m.2)) := by
        rw [sub_eq_add_neg, rpow_add hb0]; ring
    _ ≤ m.1.factorial / δ ^ m.1 * ((k : ℝ) + 1) ^ (δ - (p.1 + m.2)) :=
        mul_le_mul_of_nonneg_left (rpow_le_rpow_of_nonpos (by linarith) hb (by linarith))
          (by positivity)
    _ ≤ m.1.factorial / δ ^ m.1 * ((k : ℝ) + 1) ^ (δ - σ) :=
        mul_le_mul_of_nonneg_left (rpow_le_rpow_of_exponent_le hk (by linarith)) (by positivity)

/-- `∑ₖ (k + 1) ^ r` converges for `r < -1`. -/
private theorem summable_succ_rpow {r : ℝ} (hr : r < -1) :
    Summable fun k : ℕ => ((k : ℝ) + 1) ^ r := by
  have h := (summable_nat_add_iff 1).2 (Real.summable_nat_rpow.2 hr)
  push_cast at h
  exact h

/-- For `p.1 > 1` and `p.2 ≥ 0`, the series of `zetaTerm m p` converges. -/
private theorem summable_zetaTerm (m : ℕ × ℕ) {p : ℝ × ℝ} (hs : 1 < p.1) (hq : 0 ≤ p.2) :
    Summable (zetaTerm m p) :=
  Summable.of_nonneg_of_le (zetaTerm_nonneg m hq)
    (zetaTerm_le m (σ := p.1) (δ := (p.1 - 1) / 2) (by linarith) (by linarith) le_rfl hq)
    ((summable_succ_rpow (by linarith)).mul_left _)

/-- For `p.2 ≥ 0`, the derivative of `zetaTerm m p k` in `p` has the components
`-zetaTerm (i + 1, j)` and `-(s + j) zetaTerm (i, j + 1) + i zetaTerm (i - 1, j + 1)`. -/
private theorem hasFDerivAt_zetaTerm (m : ℕ × ℕ) {p : ℝ × ℝ} (hq : 0 ≤ p.2) (k : ℕ) :
    HasFDerivAt (fun p => zetaTerm m p k)
      ((-zetaTerm (m.1 + 1, m.2) p k) • ContinuousLinearMap.fst ℝ ℝ ℝ +
        (-(p.1 + m.2) * zetaTerm (m.1, m.2 + 1) p k + m.1 * zetaTerm (m.1 - 1, m.2 + 1) p k) •
          ContinuousLinearMap.snd ℝ ℝ ℝ) p := by
  have hb0 : 0 < (k : ℝ) + 1 + p.2 := by linarith [k.cast_nonneg (α := ℝ)]
  have hB : HasFDerivAt (fun p : ℝ × ℝ => (k : ℝ) + 1 + p.2) (ContinuousLinearMap.snd ℝ ℝ ℝ) p :=
    (hasFDerivAt_snd).const_add _
  have hE : HasFDerivAt (fun p : ℝ × ℝ => -(p.1 + (m.2 : ℝ))) (-ContinuousLinearMap.fst ℝ ℝ ℝ) p :=
    ((hasFDerivAt_fst).add_const _).neg
  have h := (hB.rpow hE hb0).mul ((hB.log hb0.ne').pow m.1)
  refine h.congr_fderiv (ContinuousLinearMap.ext fun v => ?_)
  have he : -(p.1 + ((m.2 + 1 : ℕ) : ℝ)) = -(p.1 + m.2) - 1 := by push_cast; ring
  simp only [zetaTerm, he, rpow_sub_one hb0.ne']
  simp [pow_succ]
  ring

/-- For `p.1 > 1` and `p.2 > 0`, the series of `zetaTerm m p` is differentiable in `p` term by term:
near `p` the derivatives of its terms are bounded by a convergent series. -/
private theorem hasFDerivAt_zetaSum (m : ℕ × ℕ) {p : ℝ × ℝ} (hp : p ∈ Ioi (1 : ℝ) ×ˢ Ioi (0 : ℝ)) :
    HasFDerivAt (fun p => ∑' k, zetaTerm m p k)
      ((-∑' k, zetaTerm (m.1 + 1, m.2) p k) • ContinuousLinearMap.fst ℝ ℝ ℝ +
        (-(p.1 + m.2) * ∑' k, zetaTerm (m.1, m.2 + 1) p k +
          m.1 * ∑' k, zetaTerm (m.1 - 1, m.2 + 1) p k) • ContinuousLinearMap.snd ℝ ℝ ℝ) p := by
  obtain ⟨hs, hq⟩ := hp
  have hs : 1 < p.1 := hs
  have hq : 0 < p.2 := hq
  set σ := (1 + p.1) / 2
  set δ := (σ - 1) / 2
  have hδ : 0 < δ := by simp only [δ, σ]; linarith
  have hδσ : δ ≤ σ := by simp only [δ, σ]; linarith
  have hr : δ - σ < -1 := by simp only [δ, σ]; linarith
  set C : ℝ := (m.1 + 1).factorial / δ ^ (m.1 + 1) +
    (p.1 + 1 + m.2) * (m.1.factorial / δ ^ m.1) + m.1 * ((m.1 - 1).factorial / δ ^ (m.1 - 1))
  have hV : p ∈ Ioo σ (p.1 + 1) ×ˢ Ioi (0 : ℝ) := ⟨⟨by simp only [σ]; linarith, by linarith⟩, hq⟩
  have key := hasFDerivAt_tsum_of_isPreconnected (u := fun k : ℕ => C * ((k : ℝ) + 1) ^ (δ - σ))
    (f := fun k p => zetaTerm m p k)
    (f' := fun k p => (-zetaTerm (m.1 + 1, m.2) p k) • ContinuousLinearMap.fst ℝ ℝ ℝ +
        (-(p.1 + m.2) * zetaTerm (m.1, m.2 + 1) p k + m.1 * zetaTerm (m.1 - 1, m.2 + 1) p k) •
          ContinuousLinearMap.snd ℝ ℝ ℝ)
    ((summable_succ_rpow hr).mul_left C) (isOpen_Ioo.prod isOpen_Ioi)
    ((convex_Ioo σ (p.1 + 1)).prod (convex_Ioi 0)).isPreconnected
    (fun k x hx => hasFDerivAt_zetaTerm m hx.2.le k)
    (fun k x hx => by
      obtain ⟨⟨hx1, hx2⟩, hx3⟩ := hx
      have hx3 : 0 ≤ x.2 := le_of_lt hx3
      have hj : (0 : ℝ) ≤ m.2 := m.2.cast_nonneg
      have hs0 : 0 ≤ x.1 + m.2 := by linarith
      have e1 := zetaTerm_le (m.1 + 1, m.2) hδ hδσ hx1.le hx3 k
      have e2 := zetaTerm_le (m.1, m.2 + 1) hδ hδσ hx1.le hx3 k
      have e3 := zetaTerm_le (m.1 - 1, m.2 + 1) hδ hδσ hx1.le hx3 k
      have n1 := zetaTerm_nonneg (m.1 + 1, m.2) hx3 k
      have n2 := zetaTerm_nonneg (m.1, m.2 + 1) hx3 k
      have n3 := zetaTerm_nonneg (m.1 - 1, m.2 + 1) hx3 k
      have hw : 0 ≤ ((k : ℝ) + 1) ^ (δ - σ) := rpow_nonneg (by positivity) _
      have h2 := mul_le_mul (by linarith : x.1 + m.2 ≤ p.1 + 1 + m.2) e2 n2 (by linarith)
      have h3 := mul_le_mul_of_nonneg_left e3 (Nat.cast_nonneg (α := ℝ) m.1)
      calc _ ≤ ‖-zetaTerm (m.1 + 1, m.2) x k‖ * ‖ContinuousLinearMap.fst ℝ ℝ ℝ‖ +
            ‖-(x.1 + m.2) * zetaTerm (m.1, m.2 + 1) x k + m.1 * zetaTerm (m.1 - 1, m.2 + 1) x k‖ *
              ‖ContinuousLinearMap.snd ℝ ℝ ℝ‖ :=
            (norm_add_le _ _).trans (add_le_add (ContinuousLinearMap.opNorm_smul_le _ _)
              (ContinuousLinearMap.opNorm_smul_le _ _))
        _ ≤ ‖-zetaTerm (m.1 + 1, m.2) x k‖ +
            ‖-(x.1 + m.2) * zetaTerm (m.1, m.2 + 1) x k + m.1 * zetaTerm (m.1 - 1, m.2 + 1) x k‖ :=
            add_le_add (mul_le_of_le_one_right (norm_nonneg _)
              (ContinuousLinearMap.norm_fst_le ℝ ℝ ℝ))
              (mul_le_of_le_one_right (norm_nonneg _) (ContinuousLinearMap.norm_snd_le ℝ ℝ ℝ))
        _ ≤ zetaTerm (m.1 + 1, m.2) x k + ((x.1 + m.2) * zetaTerm (m.1, m.2 + 1) x k +
            m.1 * zetaTerm (m.1 - 1, m.2 + 1) x k) := by
            rw [norm_neg, Real.norm_of_nonneg n1]
            refine add_le_add le_rfl ((norm_add_le _ _).trans_eq ?_)
            rw [norm_mul, norm_mul, norm_neg, Real.norm_of_nonneg hs0, Real.norm_of_nonneg n2,
              Real.norm_of_nonneg n3, Real.norm_natCast]
        _ ≤ C * ((k : ℝ) + 1) ^ (δ - σ) := by
            simp only [C]
            linarith)
    hV (summable_zetaTerm m hs hq.le) hV
  have s1 := summable_zetaTerm (m.1 + 1, m.2) hs hq.le
  have s2 := summable_zetaTerm (m.1, m.2 + 1) hs hq.le
  have s3 := summable_zetaTerm (m.1 - 1, m.2 + 1) hs hq.le
  convert key using 1
  rw [Summable.tsum_add (s1.neg.smul_const _) (((s2.mul_left _).add (s3.mul_left _)).smul_const _),
    Summable.tsum_smul_const s1.neg, Summable.tsum_smul_const ((s2.mul_left _).add (s3.mul_left _)),
    tsum_neg, Summable.tsum_add (s2.mul_left _) (s3.mul_left _), tsum_mul_left, tsum_mul_left]

/-- The series of `zetaTerm m p` is smooth in `p` where `p.1 > 1` and `p.2 > 0`. -/
private theorem contDiffOn_zetaSum (m : ℕ × ℕ) :
    ContDiffOn ℝ ∞ (fun p : ℝ × ℝ => ∑' k, zetaTerm m p k) (Ioi 1 ×ˢ Ioi 0) :=
  contDiffOn_of_hasFDerivAt (F := ℝ) (isOpen_Ioi.prod isOpen_Ioi)
    (g := fun m p => ∑' k, zetaTerm m p k)
    (g' := fun m p => (-∑' k, zetaTerm (m.1 + 1, m.2) p k) • ContinuousLinearMap.fst ℝ ℝ ℝ +
        (-(p.1 + m.2) * ∑' k, zetaTerm (m.1, m.2 + 1) p k +
          m.1 * ∑' k, zetaTerm (m.1 - 1, m.2 + 1) p k) • ContinuousLinearMap.snd ℝ ℝ ℝ)
    (fun m _ hp => hasFDerivAt_zetaSum m hp)
    (fun _ h _ => ((h _).neg.smul contDiffOn_const).add
      ((((contDiffOn_fst.add contDiffOn_const).neg.mul (h _)).add
        (contDiffOn_const.mul (h _))).smul contDiffOn_const)) m

/-- `zeta` is jointly smooth in `(s, q)` where `s > 1` and `q > 0`. -/
theorem contDiffOn_zeta : ContDiffOn ℝ ∞ (Function.uncurry zeta) (Ioi 1 ×ˢ Ioi 0) := by
  have hA : ContDiffOn ℝ ∞ (fun p : ℝ × ℝ => p.2 ^ (-p.1)) (Ioi 1 ×ˢ Ioi 0) := fun p hp =>
    (ContDiffAt.rpow contDiffAt_snd contDiffAt_fst.neg (ne_of_gt hp.2)).contDiffWithinAt
  refine (hA.add (contDiffOn_zetaSum (0, 0))).congr ?_
  rintro ⟨s, q⟩ ⟨hs, -⟩
  have hs : 1 < s := hs
  simp only [Function.uncurry, zeta]
  rw [(summable_zeta hs q).tsum_eq_zero_add]
  simp [zetaTerm]

end TorchLean
