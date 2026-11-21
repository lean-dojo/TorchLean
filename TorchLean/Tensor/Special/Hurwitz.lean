/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.NumberTheory.LSeries.HurwitzZeta
public import TorchLean.Tensor.Special.Zeta

/-!
# TorchLean's Hurwitz zeta function and Mathlib's

`TorchLean.zeta s q` is the series `∑ₖ (k + q)^(-s)` of DLMF equation 25.11.1. Mathlib's
`hurwitzZeta a s` reduces its shift `a` modulo one, so the two agree only for `q ∈ (0, 1]`;
`zeta_eq_hurwitzZeta` proves that they do there for every `s > 1`. The theorem is apart from
`TorchLean.Tensor.Special.Zeta` so that the operator table does not import Mathlib's L-series.

## References

- NIST Digital Library of Mathematical Functions, section 25.11(i), equation 25.11.1:
  `https://dlmf.nist.gov/25.11`
-/

@[expose] public section

noncomputable section

namespace TorchLean

open Set HurwitzZeta

/-- For `s > 1` and `q ∈ (0, 1]`, `zeta s q` is the real part of Mathlib's `hurwitzZeta q s`. -/
theorem zeta_eq_hurwitzZeta {s q : ℝ} (hs : 1 < s) (hq : q ∈ Ioc (0 : ℝ) 1) :
    zeta s q = (hurwitzZeta q s).re := by
  have hSum := hasSum_hurwitzZeta_of_one_lt_re (a := q) ⟨hq.1.le, hq.2⟩
    (s := s) (by simpa using hs)
  have hRe := Complex.hasSum_re hSum
  refine (HasSum.tsum_eq (hRe.congr_fun fun k => ?_)).symm.trans rfl |>.symm
  have hPos : (0 : ℝ) ≤ (k : ℝ) + q := add_nonneg (Nat.cast_nonneg k) hq.1.le
  simp only [one_div]
  rw [show ((k : ℂ) + (q : ℂ)) = (((k : ℝ) + q : ℝ) : ℂ) by push_cast; ring,
    ← Complex.ofReal_cpow hPos, ← Complex.ofReal_inv, Complex.ofReal_re, Real.rpow_neg hPos]

end TorchLean
