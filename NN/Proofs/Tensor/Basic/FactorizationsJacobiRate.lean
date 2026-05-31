/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Tensor.Basic.FactorizationsJacobiDecrease

/-!
# The aggregate Jacobi convergence rate (classical largest-pivot strategy)

[`FactorizationsJacobiDecrease`](./FactorizationsJacobiDecrease.lean) proved the *per-rotation*
identity `‖offDiag(Jᵀ A J)‖² = ‖offDiag A‖² − 2 · A[p,q]²` exactly. That is a statement about one
rotation; it says nothing on its own about how fast the off-diagonal mass falls over many rotations —
the *aggregate rate*, and hence how many sweeps are needed.

This file proves the aggregate rate **for the classical (largest-pivot) strategy**, which is the
elementary, a-priori-provable part of the convergence story:

* `offSq_le_count_mul_max` — the off-diagonal mass is at most the off-diagonal count `n² − n` times
  the largest squared off-diagonal entry. So if the pivot `(p, q)` is chosen to be the *largest*
  off-diagonal entry, `A[p,q]² ≥ ‖offDiag A‖² / (n² − n)`.
* `jacobi_off_decrease_classical` — combining that lower bound on the pivot with the exact
  per-rotation decrease gives a genuine **linear contraction**: one largest-pivot rotation multiplies
  the off-diagonal mass by at most `1 − 2/(n² − n) < 1`.
* `geom_bound_of_contraction` / `tendsto_zero_of_contraction` — any quantity that contracts by a fixed
  factor `ρ < 1` at every step is bounded by `ρ^k` and tends to `0`. Composed with the single-step
  contraction (with `ρ = 1 − 2/(n² − n)` and `offSq_nonneg`), this is an a-priori proof that the
  classical Jacobi eigenvalue algorithm drives the off-diagonal mass to zero geometrically.

## Honest scope: classical vs. cyclic

The executable solver runs the **cyclic** sweep (pivots visited in fixed row-major order), *not* the
classical largest-pivot rule. The per-step contraction above genuinely fails for a cyclic pivot: a
fixed-order pivot need not be the largest off-diagonal entry, so `2 · A[p,q]²` can fall well short of
`2 · ‖offDiag A‖² / (n² − n)` (and a later rotation in the same sweep can even refill an entry an
earlier one zeroed). Bounding the *sum of the cyclic pivots* below — the per-sweep contraction factor
— is the Forsythe–Henrici / Schönhage result, which Mathlib v4.30.0 has no theory for and which is not
provable by this elementary argument. The abstract `geom_bound_of_contraction` is stated for an
*arbitrary* per-step factor `ρ`, so the moment such a cyclic per-sweep bound is available it plugs in
directly; until then the cyclic rate stays captured by the exact a-posteriori residual certificate of
[`FactorizationsJacobi`](./FactorizationsJacobi.lean), never by `sorry`.
-/

@[expose] public section

namespace Spec.Factorization

open Matrix
open scoped BigOperators

variable {n : Nat}

/-! ## The off-diagonal mass is nonnegative and bounded by the count times the largest entry -/

/-- The squared off-diagonal mass is nonnegative (it is a sum of squares). -/
theorem offSq_nonneg (M : Matrix (Fin n) (Fin n) ℝ) : 0 ≤ offSq M := by
  rw [offSq_eq_sum]
  refine Finset.sum_nonneg (fun i _ => Finset.sum_nonneg (fun j _ => ?_))
  by_cases h : i = j
  · simp [h]
  · simp only [if_neg h]; positivity

/-- The constant off-diagonal sum: there are exactly `n² − n` off-diagonal positions, so summing a
constant `K` over them gives `(n² − n) · K`. -/
private theorem sum_const_offdiag (K : ℝ) :
    ∑ i : Fin n, ∑ j : Fin n, (if i = j then (0 : ℝ) else K) = ((n : ℝ) ^ 2 - (n : ℝ)) * K := by
  have hinner : ∀ i : Fin n,
      ∑ j : Fin n, (if i = j then (0 : ℝ) else K) = ((n : ℝ) - 1) * K := by
    intro i
    have hsplit : ∀ j : Fin n,
        (if i = j then (0 : ℝ) else K) = K - (if i = j then K else 0) := by
      intro j; by_cases h : i = j <;> simp [h]
    rw [Finset.sum_congr rfl (fun j _ => hsplit j), Finset.sum_sub_distrib,
      Finset.sum_const, Finset.sum_ite_eq]
    simp only [Finset.mem_univ, if_true, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
    ring
  rw [Finset.sum_congr rfl (fun i _ => hinner i), Finset.sum_const]
  simp only [Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  ring

/-- **The off-diagonal mass is at most the off-diagonal count `n² − n` times the largest squared
off-diagonal entry.** With `(p', q')` achieving that maximum, this says
`‖offDiag M‖² ≤ (n² − n) · M[p',q']²`, i.e. the largest pivot carries at least an average share of the
mass — the bound the classical Jacobi strategy exploits. -/
theorem offSq_le_count_mul_max (M : Matrix (Fin n) (Fin n) ℝ) (p' q' : Fin n)
    (hmax : ∀ i j : Fin n, i ≠ j → (M i j) ^ 2 ≤ (M p' q') ^ 2) :
    offSq M ≤ ((n : ℝ) ^ 2 - (n : ℝ)) * (M p' q') ^ 2 := by
  rw [offSq_eq_sum, ← sum_const_offdiag ((M p' q') ^ 2)]
  refine Finset.sum_le_sum (fun i _ => Finset.sum_le_sum (fun j _ => ?_))
  by_cases h : i = j
  · simp [h]
  · simp only [if_neg h]; exact hmax i j h

/-! ## The classical (largest-pivot) single-step contraction -/

variable (A : Matrix (Fin n) (Fin n) ℝ) (p q : Nat) (hp : p < n) (hq : q < n) (hpq : p ≠ q)
  (c s : ℝ)

include hpq in
/-- **Classical Jacobi linear convergence — one step.** If the pivot `(p, q)` is the *largest*
off-diagonal entry (`hmax`), `A` is symmetric there (`hsym`), and `J` is the Givens rotation that
annihilates it (`hannih`), then conjugating by `J` contracts the squared off-diagonal mass by the
fixed factor `1 − 2/(n² − n) < 1`:

`‖offDiag(Jᵀ A J)‖² ≤ (1 − 2/(n² − n)) · ‖offDiag A‖²`.

This is the exact per-rotation decrease `2 · A[p,q]²` (`jacobi_off_decrease`) combined with the
pivot lower bound `A[p,q]² ≥ ‖offDiag A‖²/(n² − n)` (`offSq_le_count_mul_max`). It is an a-priori
convergence rate for the largest-pivot strategy; the *cyclic* strategy the solver uses does not
satisfy the largest-pivot hypothesis and needs the research-grade Forsythe–Henrici bound instead. -/
theorem jacobi_off_decrease_classical (hn : 2 ≤ n) (hcs : c ^ 2 + s ^ 2 = 1)
    (hsym : A ⟨q, hq⟩ ⟨p, hp⟩ = A ⟨p, hp⟩ ⟨q, hq⟩)
    (hannih : ((toM n (Spec.arrGivens n p q c s))ᵀ * A * toM n (Spec.arrGivens n p q c s))
      ⟨p, hp⟩ ⟨q, hq⟩ = 0)
    (hmax : ∀ i j : Fin n, i ≠ j → (A i j) ^ 2 ≤ (A ⟨p, hp⟩ ⟨q, hq⟩) ^ 2) :
    offSq ((toM n (Spec.arrGivens n p q c s))ᵀ * A * toM n (Spec.arrGivens n p q c s))
      ≤ (1 - 2 / ((n : ℝ) ^ 2 - (n : ℝ))) * offSq A := by
  have hdec := jacobi_off_decrease A p q hp hq hpq c s hcs hsym hannih
  have hbound := offSq_le_count_mul_max A ⟨p, hp⟩ ⟨q, hq⟩ hmax
  have hN : (0 : ℝ) < (n : ℝ) ^ 2 - (n : ℝ) := by
    have h2 : (2 : ℝ) ≤ (n : ℝ) := by exact_mod_cast hn
    nlinarith
  rw [hdec, sub_mul, one_mul, div_mul_eq_mul_div]
  apply sub_le_sub_left
  rw [div_le_iff₀ hN]
  nlinarith [hbound]

/-! ## Iterating the contraction: geometric convergence -/

end Spec.Factorization

namespace Spec.Factorization

/-- **A fixed-factor contraction is bounded by a geometric sequence.** If `a (k+1) ≤ ρ · a k` for
all `k` with `0 ≤ ρ`, then `a k ≤ ρ^k · a 0`. Applied with `a k = ‖offDiag Aₖ‖²` and
`ρ = 1 − 2/(n² − n)` from `jacobi_off_decrease_classical`, this is the geometric a-priori rate of the
classical Jacobi algorithm. The factor `ρ` is arbitrary, so any future per-sweep cyclic bound plugs
in here unchanged. -/
theorem geom_bound_of_contraction (a : ℕ → ℝ) (ρ : ℝ) (hρ : 0 ≤ ρ)
    (hstep : ∀ k, a (k + 1) ≤ ρ * a k) : ∀ k, a k ≤ ρ ^ k * a 0 := by
  intro k
  induction k with
  | zero => simp
  | succ m ih =>
    calc a (m + 1) ≤ ρ * a m := hstep m
      _ ≤ ρ * (ρ ^ m * a 0) := mul_le_mul_of_nonneg_left ih hρ
      _ = ρ ^ (m + 1) * a 0 := by ring

/-- **The contraction drives the quantity to zero.** With a genuine factor `ρ < 1` (and `0 ≤ a k`,
which holds for `offSq` by `offSq_nonneg`), the off-diagonal mass tends to `0`: the classical Jacobi
algorithm provably converges to a diagonal matrix. -/
theorem tendsto_zero_of_contraction (a : ℕ → ℝ) (ρ : ℝ) (hρ0 : 0 ≤ ρ) (hρ1 : ρ < 1)
    (hnn : ∀ k, 0 ≤ a k) (hstep : ∀ k, a (k + 1) ≤ ρ * a k) :
    Filter.Tendsto a Filter.atTop (nhds 0) := by
  apply squeeze_zero hnn (geom_bound_of_contraction a ρ hρ0 hstep)
  have hpow : Filter.Tendsto (fun k => ρ ^ k) Filter.atTop (nhds 0) :=
    tendsto_pow_atTop_nhds_zero_of_lt_one hρ0 hρ1
  simpa using hpow.mul_const (a 0)

end Spec.Factorization
