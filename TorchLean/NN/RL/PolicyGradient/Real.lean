/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.RL.PolicyGradient
public import TorchLean.NN.Softmax
public import TorchLean.Tensor.Elementwise.Real

/-!
# Policy KL Divergences over ℝ

`TorchLean.RL.PolicyGradient.kl` and `klFromLogits` evaluate the divergence through `klTilted`, the
form of Section 4.1 of Amos (2026). This module proves that over `ℝ` the form is exact: with
log-weights `l` whose exponentials `p = e^l` sum to `1` and log-ratios `δ = a - b`,
`klTilted l a b` is `log (∑ p e^δ) - ∑ p δ`. With `l = log p` and `δ = log q - log p` that is
`∑ p (log p - log q)`, the divergence `KL(p ‖ q)`.

## Main Results

- `klTilted_eq_log_sum_exp`: `klTilted l a b = log (∑ e^l e^(a - b)) - ∑ e^l (a - b)` when
  `∑ e^l = 1`.
- `klTilted_log_sub_log`: `klTilted (log p) (log q - log p) 0 = ∑ p (log p - log q)` for positive
  `p`, `q` summing to `1`.
- `kl_eq_sum`: `kl` is the divergence of the guarded probabilities when `0 < epsilon < 1`.
- `klFromLogits_eq_sum`: `klFromLogits` is the divergence of the two softmax policies.

## References

- Amos, "On numerical issues of divergences between nearly equal distributions" (2026),
  arXiv:2611.18511, Section 4.1, "Fixing the forward KL": https://arxiv.org/abs/2611.18511
-/

@[expose] public section

open FloatLib.Numerics (MathFunctions)

namespace TorchLean
namespace RL
namespace PolicyGradient

open TorchLean.Tensor

/-- The real identity behind `klTilted`: for any centering `m` and `c = log (∑ p e^(δ - m))`,
`∑ p (e^s - 1 - s)` at `s = δ - m - c` is `log (∑ p e^δ) - ∑ p δ`. -/
private theorem sum_mul_expm1x_tilted {n : Nat} (p δ : Fin n → ℝ) (hnonneg : ∀ i, 0 ≤ p i)
    (hsum : ∑ i, p i = 1) (m c : ℝ) (hc : c = Real.log (∑ i, p i * Real.exp (δ i - m))) :
    ∑ i, p i * (Real.exp (δ i - m - c) - 1 - (δ i - m - c)) =
      Real.log (∑ i, p i * Real.exp (δ i)) - ∑ i, p i * δ i := by
  have hE : 0 < ∑ i, p i * Real.exp (δ i) := by
    obtain ⟨j, hj⟩ : ∃ j, 0 < p j := by
      by_contra h
      push Not at h
      have : ∑ i, p i ≤ 0 := Finset.sum_nonpos (fun i _ => h i)
      linarith
    have hterm : ∀ i ∈ Finset.univ, 0 ≤ p i * Real.exp (δ i) :=
      fun i _ => mul_nonneg (hnonneg i) (Real.exp_pos _).le
    calc 0 < p j * Real.exp (δ j) := mul_pos hj (Real.exp_pos _)
      _ ≤ ∑ i, p i * Real.exp (δ i) := Finset.single_le_sum hterm (Finset.mem_univ j)
  have hmass : ∑ i, p i * Real.exp (δ i - m) = (∑ i, p i * Real.exp (δ i)) / Real.exp m := by
    rw [Finset.sum_div]
    exact Finset.sum_congr rfl fun i _ => by rw [Real.exp_sub]; ring
  have hc' : c = Real.log (∑ i, p i * Real.exp (δ i)) - m := by
    rw [hc, hmass, Real.log_div hE.ne' (Real.exp_pos m).ne', Real.log_exp]
  have hterms : ∀ i, p i * (Real.exp (δ i - m - c) - 1 - (δ i - m - c)) =
      p i * Real.exp (δ i) / (∑ j, p j * Real.exp (δ j)) - p i - p i * δ i +
        p i * Real.log (∑ j, p j * Real.exp (δ j)) := by
    intro i
    have hs : δ i - m - c = δ i - Real.log (∑ j, p j * Real.exp (δ j)) := by
      rw [hc']
      ring
    rw [hs, Real.exp_sub, Real.exp_log hE]
    ring
  simp only [hterms, Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.sum_div,
    ← Finset.sum_mul, hsum]
  rw [div_self hE.ne']
  ring

/-- Over `ℝ`, `weightedExpm1x l s x quarter` is `e^l * (e^s - 1 - s)` when `x` is `l + s` and
`quarter` is `s / 4`. -/
theorem weightedExpm1x_real (l s x quarter : ℝ) (hx : x = l + s) (hquarter : 4 * quarter = s) :
    weightedExpm1x l s x quarter = Real.exp l * (Real.exp s - 1 - s) := by
  subst hx
  unfold weightedExpm1x
  dsimp only
  split_ifs
  · change Real.exp (l + s) - Real.exp l - Real.exp l * s = _
    rw [Real.exp_add]
    ring
  · change Real.exp (l + s) - Real.exp l - ((Real.exp l * quarter + Real.exp l * quarter) +
      (Real.exp l * quarter + Real.exp l * quarter)) = _
    rw [Real.exp_add, ← hquarter]
    ring
  · rfl

/-- The entries of `Tensor.exp t` over `ℝ`. -/
private theorem getScalar_exp {n : Nat} (t : Tensor ℝ [n]) (i : Fin n) :
    (Tensor.exp t).getScalar i = Real.exp (t.getScalar i) := by
  simp only [Tensor.exp, Tensor.getScalar_map]
  rfl

/-- Over `ℝ`, a shift `t` and the two parts of `logWeightedMass gap e` sum to `log (∑ e^l e^d)` for
log-weights `l` whose exponentials sum to `1`, `gap = ∑ e^l (e^d - 1)`, `e = (d - t) + l`, and
`t = 0` when `gap` is above `-1/2`. -/
theorem logWeightedMass_real {n : Nat} (l d e : Tensor ℝ [n]) (h : 0 < n) (t : ℝ)
    (he : e = (d - Tensor.full _ t) + l)
    (ht : Tensor.sum (Tensor.exp l * Tensor.expm1 d) > -((1 : ℝ) / 2) → t = 0)
    (hsum : ∑ i, Real.exp (l.getScalar i) = 1) :
    t + ((logWeightedMass (Tensor.sum (Tensor.exp l * Tensor.expm1 d)) e h).1 +
      (logWeightedMass (Tensor.sum (Tensor.exp l * Tensor.expm1 d)) e h).2) =
      Real.log (∑ i, Real.exp (l.getScalar i) * Real.exp (d.getScalar i)) := by
  have hE : 0 < ∑ i, Real.exp (l.getScalar i) * Real.exp (d.getScalar i) :=
    Finset.sum_pos (fun i _ => mul_pos (Real.exp_pos _) (Real.exp_pos _))
      ⟨⟨0, h⟩, Finset.mem_univ _⟩
  unfold logWeightedMass
  dsimp only
  split_ifs with hgap
  · rw [ht hgap]
    simp only [zero_add, Tensor.sum_vec, Tensor.getScalar_mul, getScalar_exp, Tensor.expm1,
      Tensor.getScalar_map, Context.expm1_eq_exp_sub_one, Context.log1p_eq_log_one_add]
    congr 1
    simp only [mul_sub, mul_one, Finset.sum_sub_distrib, hsum]
    ring
  · have key : ∀ c u : ℝ,
        c + (u + Real.log (∑ i, Real.exp (d.getScalar i - c + l.getScalar i - u))) =
          Real.log (∑ i, Real.exp (l.getScalar i) * Real.exp (d.getScalar i)) := by
      intro c u
      have hterms : ∀ i, Real.exp (d.getScalar i - c + l.getScalar i - u) =
          Real.exp (l.getScalar i) * Real.exp (d.getScalar i) / Real.exp (c + u) := by
        intro i
        rw [show d.getScalar i - c + l.getScalar i - u =
          (l.getScalar i + d.getScalar i) - (c + u) by ring, Real.exp_sub, Real.exp_add]
      simp only [hterms, ← Finset.sum_div]
      rw [Real.log_div hE.ne' (Real.exp_pos _).ne', Real.log_exp]
      ring
    subst he
    simp only [Tensor.sum_vec, Tensor.sub_def, Tensor.add_def, Tensor.getScalar_zipWith,
      Tensor.getScalar_full, getScalar_exp]
    exact key _ _

/-- The sum of `klTilted` over `ℝ`, for a quarter shift `quarter` with `4 * quarter = δ - m`,
`d = 4 * quarter`, exponents `e = (d - t) + l` and parts `t + (u + r) = log (∑ e^l e^d)`: it is
`log (∑ e^l e^δ) - ∑ e^l δ` when `∑ e^l = 1`. -/
private theorem sum_weightedExpm1x_quarter {n : Nat} (l quarter d e : Tensor ℝ [n + 1])
    (δ : Fin (n + 1) → ℝ) (m t u r : ℝ)
    (hd : ∀ i, d.getScalar i = 4 * quarter.getScalar i)
    (hquarter : ∀ i, 4 * quarter.getScalar i = δ i - m)
    (he : ∀ i, e.getScalar i = d.getScalar i - t + l.getScalar i)
    (hmass : t + (u + r) = Real.log (∑ i, Real.exp (l.getScalar i) * Real.exp (d.getScalar i)))
    (hsum : ∑ i, Real.exp (l.getScalar i) = 1) :
    Tensor.sum (Tensor.ofFn fun i => weightedExpm1x (l.getScalar i) (d.getScalar i - t - u - r)
        (e.getScalar i - u - r) (quarter.getScalar i - t / 4 - u / 4 - r / 4)) =
      Real.log (∑ i, Real.exp (l.getScalar i) * Real.exp (δ i)) -
        ∑ i, Real.exp (l.getScalar i) * δ i := by
  have hdδ : ∀ i, d.getScalar i = δ i - m := fun i => (hd i).trans (hquarter i)
  have hterm : ∀ i, weightedExpm1x (l.getScalar i) (d.getScalar i - t - u - r)
      (e.getScalar i - u - r) (quarter.getScalar i - t / 4 - u / 4 - r / 4) =
        Real.exp (l.getScalar i) *
          (Real.exp (δ i - m - (t + (u + r))) - 1 - (δ i - m - (t + (u + r)))) := by
    intro i
    have hs : d.getScalar i - t - u - r = δ i - m - (t + (u + r)) := by
      rw [hdδ i]
      ring
    rw [weightedExpm1x_real (l.getScalar i) (d.getScalar i - t - u - r) (e.getScalar i - u - r)
      (quarter.getScalar i - t / 4 - u / 4 - r / 4) (by rw [he i]; ring) (by rw [hd i]; ring), hs]
  simp only [Tensor.sum_vec, Tensor.getScalar_ofFn, hterm]
  exact sum_mul_expm1x_tilted (fun i => Real.exp (l.getScalar i)) δ (fun i => (Real.exp_pos _).le)
    hsum m (t + (u + r)) (by rw [hmass]; simp only [hdδ])

/-- Over `ℝ`, `klTilted l a b` is `log (∑ e^l e^(a - b)) - ∑ e^l (a - b)` for log-weights `l`
whose exponentials sum to `1`. -/
theorem klTilted_eq_log_sum_exp {n : Nat} (l a b : Tensor ℝ [n])
    (hsum : ∑ i, Real.exp (l.getScalar i) = 1) :
    klTilted l a b =
      Real.log (∑ i, Real.exp (l.getScalar i) * Real.exp (a.getScalar i - b.getScalar i)) -
        ∑ i, Real.exp (l.getScalar i) * (a.getScalar i - b.getScalar i) := by
  cases n with
  | zero => simp at hsum
  | succ n =>
    have h := Nat.succ_pos n
    let quarter := centered (quarterCentered a h - quarterCentered b h) h
    let d := (quarter + quarter) + (quarter + quarter)
    have hd : ∀ i, d.getScalar i = 4 * quarter.getScalar i := fun i => by
      simp only [d, Tensor.add_def, Tensor.getScalar_zipWith]
      ring
    have hquarter : ∀ i, 4 * quarter.getScalar i = (a.getScalar i - b.getScalar i) -
        4 * ((quarterCentered a h - quarterCentered b h).getScalar
            (Softmax.pivot (quarterCentered a h - quarterCentered b h) h) +
          (a / Tensor.full _ 4).getScalar (Softmax.pivot (a / Tensor.full _ 4) h) -
          (b / Tensor.full _ 4).getScalar (Softmax.pivot (b / Tensor.full _ 4) h)) := fun i => by
      simp only [quarter, centered, quarterCentered, Tensor.sub_def, Tensor.div_def,
        Tensor.getScalar_zipWith, Tensor.getScalar_full]
      ring
    simp only [klTilted, h, ↓reduceDIte]
    let gap := Tensor.sum (Tensor.exp l * Tensor.expm1 d)
    let w := d + l
    let t : ℝ := if gap > -((1 : ℝ) / 2) then 0 else w.getScalar (Softmax.pivot w h)
    let e := (d - Tensor.full _ t) + l
    have he : ∀ i, e.getScalar i = d.getScalar i - t + l.getScalar i := fun i => by
      simp only [e, Tensor.add_def, Tensor.sub_def, Tensor.getScalar_zipWith, Tensor.getScalar_full]
    have ht : gap > -((1 : ℝ) / 2) → t = 0 := fun hg => by simp only [t, hg, ↓reduceIte]
    exact sum_weightedExpm1x_quarter l quarter d e (fun i => a.getScalar i - b.getScalar i) _ t
      (logWeightedMass gap e h).1 (logWeightedMass gap e h).2 hd hquarter he
      (logWeightedMass_real l d e h t rfl ht hsum) hsum

/-- The entries of `Tensor.log t` over `ℝ`. -/
private theorem getScalar_log {n : Nat} (t : Tensor ℝ [n]) (i : Fin n) :
    (Tensor.log t).getScalar i = Real.log (t.getScalar i) := by
  simp only [Tensor.log, Tensor.getScalar_map]
  rfl

/-- With the log-weights `log p`, `a = log q - log p` and `b = 0`, `klTilted` is
`KL(p ‖ q) = ∑ p (log p - log q)` for positive `p` and `q` that each sum to `1`. -/
theorem klTilted_log_sub_log {n : Nat} (p q : Tensor ℝ [n])
    (hp : ∀ i, 0 < p.getScalar i) (hq : ∀ i, 0 < q.getScalar i)
    (hpsum : Tensor.sum p = 1) (hqsum : Tensor.sum q = 1) :
    klTilted (Tensor.log p) (Tensor.log q - Tensor.log p) (Tensor.full _ 0) =
      ∑ i, p.getScalar i * (Real.log (p.getScalar i) - Real.log (q.getScalar i)) := by
  have hw : ∀ i, Real.exp ((Tensor.log p).getScalar i) = p.getScalar i := fun i => by
    rw [getScalar_log, Real.exp_log (hp i)]
  rw [Tensor.sum_vec] at hpsum hqsum
  rw [klTilted_eq_log_sum_exp _ _ _ (by simp only [hw]; exact hpsum)]
  simp only [hw, Tensor.getScalar_full, sub_zero]
  have hδ : ∀ i, (Tensor.log q - Tensor.log p).getScalar i =
      Real.log (q.getScalar i) - Real.log (p.getScalar i) := by
    intro i
    simp only [Tensor.sub_def, Tensor.getScalar_zipWith, getScalar_log]
  have hexp : ∀ i, p.getScalar i * Real.exp ((Tensor.log q - Tensor.log p).getScalar i) =
      q.getScalar i := by
    intro i
    have hpi := (hp i).ne'
    rw [hδ i, Real.exp_sub, Real.exp_log (hq i), Real.exp_log (hp i)]
    field_simp
  simp only [hexp]
  simp only [hδ]
  rw [hqsum, Real.log_one, zero_sub, ← Finset.sum_neg_distrib]
  exact Finset.sum_congr rfl fun i _ => by ring

/-- Every entry of `guardedProbabilities probs epsilon` is `min (max p epsilon) (1 - epsilon)`
divided by the sum of those values. -/
theorem getScalar_guardedProbabilities {n : Nat} (probs : Tensor ℝ [n]) (epsilon : ℝ)
    (i : Fin n) :
    (guardedProbabilities probs epsilon).getScalar i =
      min (max (probs.getScalar i) epsilon) (1 - epsilon) /
        ∑ j, min (max (probs.getScalar j) epsilon) (1 - epsilon) := by
  have hstop : ∀ y : ℝ, Context.stopGradient y = y := fun _ => rfl
  have hclamp : ∀ j, (Tensor.clamp probs (Tensor.full _ epsilon)
      (Tensor.full _ ((1 : ℝ) - epsilon)) .interior).getScalar j =
        min (max (probs.getScalar j) epsilon) (1 - epsilon) := by
    intro j
    simp only [Tensor.getScalar_eq_apply, Tensor.clamp_interior_apply, Tensor.full_apply, hstop,
      ite_self, Context.clamp_real]
  unfold guardedProbabilities
  simp only [Tensor.div_def, Tensor.getScalar_zipWith, Tensor.getScalar_full, Tensor.sum_vec,
    hclamp]

/-- The clamped entry `min (max p epsilon) (1 - epsilon)` is positive when `0 < epsilon < 1`. -/
private theorem guarded_entry_pos {p epsilon : ℝ} (hpos : 0 < epsilon) (hlt : epsilon < 1) :
    0 < min (max p epsilon) (1 - epsilon) :=
  lt_min (lt_max_iff.mpr (Or.inr hpos)) (sub_pos.mpr hlt)

/-- The guarded probabilities are positive when `0 < epsilon < 1`. -/
theorem guardedProbabilities_pos {n : Nat} (probs : Tensor ℝ [n]) {epsilon : ℝ}
    (hpos : 0 < epsilon) (hlt : epsilon < 1) (i : Fin n) :
    0 < (guardedProbabilities probs epsilon).getScalar i := by
  rw [getScalar_guardedProbabilities]
  exact div_pos (guarded_entry_pos hpos hlt)
    (Finset.sum_pos (fun _ _ => guarded_entry_pos hpos hlt) ⟨i, Finset.mem_univ i⟩)

/-- The guarded probabilities of a nonempty vector sum to `1` when `0 < epsilon < 1`. -/
theorem sum_guardedProbabilities {n : Nat} (probs : Tensor ℝ [n + 1]) {epsilon : ℝ}
    (hpos : 0 < epsilon) (hlt : epsilon < 1) :
    Tensor.sum (guardedProbabilities probs epsilon) = 1 := by
  rw [Tensor.sum_vec]
  simp only [getScalar_guardedProbabilities]
  rw [← Finset.sum_div]
  exact div_self
    (Finset.sum_pos (fun _ _ => guarded_entry_pos hpos hlt) Finset.univ_nonempty).ne'

/-- For `0 < epsilon < 1`, `kl oldProbs newProbs epsilon` is `KL(p ‖ q) = ∑ p (log p - log q)` of
the guarded probabilities `p` of `oldProbs` and `q` of `newProbs`. -/
theorem kl_eq_sum {n : Nat} (oldProbs newProbs : Tensor ℝ [n]) {epsilon : ℝ}
    (hpos : 0 < epsilon) (hlt : epsilon < 1) :
    kl oldProbs newProbs epsilon =
      ∑ i, (guardedProbabilities oldProbs epsilon).getScalar i *
        (Real.log ((guardedProbabilities oldProbs epsilon).getScalar i) -
          Real.log ((guardedProbabilities newProbs epsilon).getScalar i)) := by
  cases n with
  | zero => simp [kl, klTilted]
  | succ n =>
    exact klTilted_log_sub_log _ _ (guardedProbabilities_pos oldProbs hpos hlt)
      (guardedProbabilities_pos newProbs hpos hlt) (sum_guardedProbabilities oldProbs hpos hlt)
      (sum_guardedProbabilities newProbs hpos hlt)

/-- Rearranging `∑ p ((a - A) - (b - B))` when the weights `p` sum to `1`. -/
private theorem sum_mul_sub_sub {n : Nat} (p a b : Fin n → ℝ) (A B : ℝ)
    (hsum : ∑ i, p i = 1) :
    B - A - ∑ i, p i * (b i - a i) = ∑ i, p i * ((a i - A) - (b i - B)) := by
  have h : ∀ i, p i * ((a i - A) - (b i - B)) = p i * (B - A) - p i * (b i - a i) :=
    fun i => by ring
  simp only [h, Finset.sum_sub_distrib, ← Finset.sum_mul, hsum, one_mul]

/-- `klFromLogits oldLogits newLogits` is `KL(p ‖ q) = ∑ p (log p - log q)` of the softmax policies
`p` of `oldLogits` and `q` of `newLogits`. -/
theorem klFromLogits_eq_sum {n : Nat} (oldLogits newLogits : Tensor ℝ [n]) :
    klFromLogits oldLogits newLogits =
      ∑ i, (probabilities oldLogits).getScalar i *
        (Real.log ((probabilities oldLogits).getScalar i) -
          Real.log ((probabilities newLogits).getScalar i)) := by
  cases n with
  | zero => simp [klFromLogits, klTilted]
  | succ n =>
    have hvec : ∀ t : Tensor ℝ [n + 1], probabilities t = Softmax.vec t := fun _ => rfl
    have hZ : ∀ t : Tensor ℝ [n + 1], 0 < ∑ j, Real.exp (t.getScalar j) := fun _ =>
      Finset.sum_pos (fun _ _ => Real.exp_pos _) Finset.univ_nonempty
    have hlog : ∀ (t : Tensor ℝ [n + 1]) i, Real.log ((Softmax.vec t).getScalar i) =
        t.getScalar i - Real.log (∑ j, Real.exp (t.getScalar j)) := by
      intro t i
      rw [Softmax.getScalar_vec_eq_exp_div, Real.log_div (Real.exp_pos _).ne' (hZ t).ne',
        Real.log_exp]
    have hw : ∀ i, Real.exp ((Tensor.logSoftmax oldLogits ⟨0, Nat.one_pos⟩).getScalar i) =
        (Softmax.vec oldLogits).getScalar i := by
      intro i
      rw [Tensor.logSoftmax_zero_vec, LogSoftmax.getScalar_vec_eq_sub_log, Real.exp_sub,
        Real.exp_log (hZ _), Softmax.getScalar_vec_eq_exp_div]
    have hpsum := Softmax.sum_vec oldLogits
    rw [Tensor.sum_vec] at hpsum
    have hexp : ∀ i, (Softmax.vec oldLogits).getScalar i *
        Real.exp (newLogits.getScalar i - oldLogits.getScalar i) =
          Real.exp (newLogits.getScalar i) / ∑ j, Real.exp (oldLogits.getScalar j) := by
      intro i
      have hZo := (hZ oldLogits).ne'
      have hexpo := (Real.exp_pos (oldLogits.getScalar i)).ne'
      rw [Softmax.getScalar_vec_eq_exp_div, Real.exp_sub]
      field_simp
    unfold klFromLogits
    rw [klTilted_eq_log_sum_exp _ _ _ (by simp only [hw]; exact hpsum)]
    simp only [hw, hexp, ← Finset.sum_div]
    rw [Real.log_div (hZ newLogits).ne' (hZ oldLogits).ne', hvec, hvec]
    simp only [hlog]
    exact sum_mul_sub_sub (fun i => (Softmax.vec oldLogits).getScalar i)
      (fun i => oldLogits.getScalar i) (fun i => newLogits.getScalar i)
      (Real.log (∑ j, Real.exp (oldLogits.getScalar j)))
      (Real.log (∑ j, Real.exp (newLogits.getScalar j))) hpsum

end PolicyGradient
end RL
end TorchLean
