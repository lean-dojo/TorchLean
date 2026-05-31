/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Tensor.Basic.FactorizationsJacobi

/-!
# The cyclic Jacobi sweep makes progress (per-rotation off-diagonal decrease)

[`FactorizationsJacobi`](./FactorizationsJacobi.lean) made the residual certificate *unconditional*:
the solver output always satisfies orthogonality and the orthogonal-similarity `A = V · Af · Vᵀ`, so
the reconstruction error equals the off-diagonal mass `‖offDiag(Af)‖²` of the rotated matrix. What
that certificate does **not** say is that the off-diagonal mass actually *goes down*. This file
proves the classical Jacobi progress identity, which is exactly that statement at the level of a
single rotation:

> If a symmetric `A` is conjugated by the Givens rotation that annihilates the pivot `(p, q)`, the
> squared off-diagonal mass drops by exactly `2 · A[p,q]²`.

The two ingredients, both *exact* over `ℝ`:

* `frobSq_orthogonal_conj` — orthogonal similarity preserves the total Frobenius mass
  `‖A‖² = trace(Aᵀ A)`. Combined with `frobSq_eq_diagSq_add_offSq` (`‖A‖² = diag-mass + off-mass`),
  driving the off-diagonal mass down is the *same thing* as driving the diagonal mass up.
* `givens_conj_*` — the explicit entries of `Jᵀ A J` in the rotation plane. A Givens conjugation only
  touches rows/columns `p, q`, so the diagonal mass changes by `A'[p,p]² + A'[q,q]² − A[p,p]² −
  A[q,q]²`, and the `2×2` block algebra (with `c² + s² = 1` and the annihilation `A'[p,q] = 0`) turns
  that into `2 · A[p,q]²`.

The pivot-annihilation is taken as a hypothesis (`hannih`): it is the defining property of the
rotation angle. `givens_conj_pq` gives the explicit value of that pivot entry after the rotation, so
`hannih` is the concrete equation `c·s·A[p,p] + c²·A[p,q] − s²·A[q,p] − c·s·A[q,q] = 0` that the
Golub–Van Loan parameters the code uses are chosen to solve. Scope, as elsewhere in this development:
this is the per-rotation decrease, the exact finite fact behind convergence; the *rate* over a whole
sweep (and hence the number of sweeps needed) remains the research-grade piece Mathlib has no theory
for.
-/

@[expose] public section

namespace Spec.Factorization

open Matrix
open scoped BigOperators

variable {n : Nat}

/-! ## Frobenius mass: total, diagonal, off-diagonal -/

/-- Total squared Frobenius mass `‖M‖² = trace(Mᵀ M) = ∑ᵢⱼ M[i,j]²`. -/
def frobSq (M : Matrix (Fin n) (Fin n) ℝ) : ℝ := (Mᵀ * M).trace

/-- Squared diagonal mass `∑ᵢ M[i,i]²`. -/
def diagSq (M : Matrix (Fin n) (Fin n) ℝ) : ℝ := ∑ i, (M i i) ^ 2

/-- Squared off-diagonal mass `‖offDiag M‖² = trace((offDiag M)ᵀ (offDiag M))`. This is the quantity
the residual certificate equates with the reconstruction error. -/
def offSq (M : Matrix (Fin n) (Fin n) ℝ) : ℝ :=
  ((offDiagonal M)ᵀ * offDiagonal M).trace

/-- `‖M‖²` as the sum of all squared entries. -/
theorem frobSq_eq_sum (M : Matrix (Fin n) (Fin n) ℝ) :
    frobSq M = ∑ i, ∑ j, (M i j) ^ 2 := by
  unfold frobSq
  rw [Matrix.trace]
  simp only [Matrix.diag_apply, Matrix.mul_apply, Matrix.transpose_apply]
  rw [Finset.sum_comm]
  exact Finset.sum_congr rfl (fun i _ => Finset.sum_congr rfl (fun j _ => by ring))

/-- The off-diagonal part has entries `M[i,j]` off the diagonal and `0` on it. -/
theorem offDiagonal_apply (M : Matrix (Fin n) (Fin n) ℝ) (i j : Fin n) :
    offDiagonal M i j = if i = j then 0 else M i j := by
  unfold offDiagonal
  rw [Matrix.sub_apply]
  by_cases h : i = j
  · subst h; simp
  · rw [Matrix.diagonal_apply_ne _ h, sub_zero, if_neg h]

/-- `‖offDiag M‖²` as the sum of squared off-diagonal entries. -/
theorem offSq_eq_sum (M : Matrix (Fin n) (Fin n) ℝ) :
    offSq M = ∑ i, ∑ j, if i = j then 0 else (M i j) ^ 2 := by
  unfold offSq
  rw [Matrix.trace]
  simp only [Matrix.diag_apply, Matrix.mul_apply, Matrix.transpose_apply]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl (fun i _ => Finset.sum_congr rfl (fun j _ => ?_))
  rw [offDiagonal_apply]
  by_cases h : i = j
  · subst h; simp
  · simp only [if_neg h]; ring

/-- **The Frobenius mass splits as diagonal mass plus off-diagonal mass.** -/
theorem frobSq_eq_diagSq_add_offSq (M : Matrix (Fin n) (Fin n) ℝ) :
    frobSq M = diagSq M + offSq M := by
  rw [frobSq_eq_sum, offSq_eq_sum, diagSq, ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl (fun i _ => ?_)
  have hsplit : ∀ j : Fin n,
      (M i j) ^ 2 = (if i = j then (M i j) ^ 2 else 0) + (if i = j then 0 else (M i j) ^ 2) := by
    intro j; by_cases h : i = j <;> simp [h]
  rw [Finset.sum_congr rfl (fun j _ => hsplit j), Finset.sum_add_distrib, Finset.sum_ite_eq]
  simp

/-- **Orthogonal similarity preserves the total Frobenius mass.** Every Jacobi step is such a
similarity, so `‖A‖²` is an exact invariant of the whole run. -/
theorem frobSq_orthogonal_conj {J M : Matrix (Fin n) (Fin n) ℝ} (hJ : Jᵀ * J = 1) :
    frobSq (Jᵀ * M * J) = frobSq M := by
  have hJJ : J * Jᵀ = 1 := mul_eq_one_comm.mp hJ
  unfold frobSq
  have hprod : ((Jᵀ * M * J)ᵀ * (Jᵀ * M * J)) = Jᵀ * (Mᵀ * M) * J := by
    rw [Matrix.transpose_mul, Matrix.transpose_mul, Matrix.transpose_transpose]
    simp only [Matrix.mul_assoc]
    rw [← Matrix.mul_assoc J Jᵀ (M * J), hJJ, Matrix.one_mul]
  rw [hprod, Matrix.trace_mul_comm, ← Matrix.mul_assoc, hJJ, Matrix.one_mul]

/-! ## The `2×2` block algebra

The rotation only mixes rows/columns `p` and `q`, so all the analysis happens in a `2×2` block. The
key fact is that an orthogonal `2×2` conjugation preserves the block's Frobenius mass; we obtain it by
specialising `frobSq_orthogonal_conj` to `Fin 2`. -/

/-- **`2×2` block Frobenius preservation.** Conjugating the block `!![a, b; b', d]` by the orthogonal
rotation `!![c, s; -s, c]` (with `c² + s² = 1`) preserves the sum of squared entries. Proved by
specialising `frobSq_orthogonal_conj` to `Fin 2`. -/
private theorem block_frob (a b b' d c s : ℝ) (hcs : c ^ 2 + s ^ 2 = 1) :
    (c ^ 2 * a - c * s * (b + b') + s ^ 2 * d) ^ 2 + (s ^ 2 * a + c * s * (b + b') + c ^ 2 * d) ^ 2
      + (c * s * a + c ^ 2 * b - s ^ 2 * b' - c * s * d) ^ 2
      + (c * s * a + c ^ 2 * b' - s ^ 2 * b - c * s * d) ^ 2
      = a ^ 2 + b ^ 2 + b' ^ 2 + d ^ 2 := by
  have hR : (!![c, s; -s, c] : Matrix (Fin 2) (Fin 2) ℝ)ᵀ * !![c, s; -s, c] = 1 := by
    ext i j; fin_cases i <;> fin_cases j <;>
      simp [Matrix.mul_apply, Fin.sum_univ_two] <;> nlinarith [hcs]
  have hfrob := frobSq_orthogonal_conj (M := (!![a, b; b', d] : Matrix (Fin 2) (Fin 2) ℝ)) hR
  rw [frobSq_eq_sum, frobSq_eq_sum] at hfrob
  simp [Fin.sum_univ_two, Matrix.mul_apply, Matrix.transpose_apply] at hfrob
  linear_combination hfrob

/-- **The diagonal-mass increase.** Under the rotation parameters (`c² + s² = 1`), symmetry of the
pivot (`b' = b`), and the annihilation equation `c·s·(a − d) + (c² − s²)·b = 0`, the two rotated
diagonal squares exceed the originals by exactly `2 b²`. -/
private theorem block_diag_algebra (a b d c s : ℝ) (hcs : c ^ 2 + s ^ 2 = 1)
    (hann : c * s * (a - d) + (c ^ 2 - s ^ 2) * b = 0) :
    (c ^ 2 * a - 2 * c * s * b + s ^ 2 * d) ^ 2 + (s ^ 2 * a + 2 * c * s * b + c ^ 2 * d) ^ 2
        - a ^ 2 - d ^ 2 = 2 * b ^ 2 := by
  have hbf := block_frob a b b d c s hcs
  have h0 : c * s * a + c ^ 2 * b - s ^ 2 * b - c * s * d = 0 := by linear_combination hann
  linear_combination hbf - 2 * (c * s * a + c ^ 2 * b - s ^ 2 * b - c * s * d) * h0

/-! ## Sum helpers -/

/-- Sum of a function `f` against an indicator supported on the pair `{p', q'}` (with `p' ≠ q'`). -/
private theorem sum_pair (p' q' : Fin n) (hpq : p' ≠ q') (vp vq : ℝ) (f : Fin n → ℝ) :
    ∑ l, (if l = p' then vp else if l = q' then vq else 0) * f l = vp * f p' + vq * f q' := by
  have hterm : ∀ l : Fin n,
      (if l = p' then vp else if l = q' then vq else 0) * f l
        = (if l = p' then vp * f l else 0) + (if l = q' then vq * f l else 0) := by
    intro l
    by_cases hlp : l = p'
    · subst hlp; simp [hpq]
    · by_cases hlq : l = q'
      · subst hlq; simp [hlp]
      · simp [hlp, hlq]
  rw [Finset.sum_congr rfl (fun l _ => hterm l), Finset.sum_add_distrib,
    Finset.sum_ite_eq', Finset.sum_ite_eq']
  simp

/-- A fintype sum of a function supported on the pair `{p', q'}` collapses to the two values. -/
private theorem sum_eq_pair (p' q' : Fin n) (hpq : p' ≠ q') (g : Fin n → ℝ)
    (h0 : ∀ o, o ≠ p' → o ≠ q' → g o = 0) : ∑ o, g o = g p' + g q' := by
  rw [← Finset.sum_pair hpq]
  refine (Finset.sum_subset (Finset.subset_univ _) (fun o _ ho => ?_)).symm
  simp only [Finset.mem_insert, Finset.mem_singleton, not_or] at ho
  exact h0 o ho.1 ho.2

/-! ## Entries of `A · J` and of the conjugation `Jᵀ · A · J`

`J = toM n (arrGivens n p q c s)` has columns supported on `{p, q}` (off `{p, q}` it is the identity),
so multiplying by it only combines columns `p`, `q`. -/

variable (A : Matrix (Fin n) (Fin n) ℝ) (p q : Nat) (hp : p < n) (hq : q < n) (hpq : p ≠ q)
  (c s : ℝ)

include hpq in
private theorem fin_pq_ne : (⟨p, hp⟩ : Fin n) ≠ ⟨q, hq⟩ := fun h => hpq (Fin.ext_iff.mp h)

include hpq in
/-- Column `p` of `A · J`: `c · A[·,p] − s · A[·,q]`. -/
theorem givens_AJ_p (k : Fin n) :
    (A * toM n (Spec.arrGivens n p q c s)) k ⟨p, hp⟩
      = c * A k ⟨p, hp⟩ - s * A k ⟨q, hq⟩ := by
  rw [Matrix.mul_apply,
    Finset.sum_congr rfl (fun l _ => by rw [givens_col_fp p q hp hq hpq c s l, mul_comm]),
    sum_pair ⟨p, hp⟩ ⟨q, hq⟩ (fin_pq_ne p q hp hq hpq) c (-s) (fun l => A k l)]
  ring

include hpq in
/-- Column `q` of `A · J`: `s · A[·,p] + c · A[·,q]`. -/
theorem givens_AJ_q (k : Fin n) :
    (A * toM n (Spec.arrGivens n p q c s)) k ⟨q, hq⟩
      = s * A k ⟨p, hp⟩ + c * A k ⟨q, hq⟩ := by
  rw [Matrix.mul_apply,
    Finset.sum_congr rfl (fun l _ => by rw [givens_col_fq p q hp hq hpq c s l, mul_comm]),
    sum_pair ⟨p, hp⟩ ⟨q, hq⟩ (fin_pq_ne p q hp hq hpq) s c (fun l => A k l)]

/-- Any other column `o ∉ {p, q}` of `A · J` is unchanged. -/
theorem givens_AJ_other (o : Fin n) (hop : o.val ≠ p) (hoq : o.val ≠ q) (k : Fin n) :
    (A * toM n (Spec.arrGivens n p q c s)) k o = A k o := by
  rw [Matrix.mul_apply,
    Finset.sum_congr rfl (fun l _ => by rw [givens_col_other p q c s o l hop hoq])]
  simp

include hpq in
/-- The `(p, p)` entry of the conjugation `Jᵀ · A · J`. -/
theorem givens_conj_pp :
    ((toM n (Spec.arrGivens n p q c s))ᵀ * A * toM n (Spec.arrGivens n p q c s)) ⟨p, hp⟩ ⟨p, hp⟩
      = c ^ 2 * A ⟨p, hp⟩ ⟨p, hp⟩ - c * s * (A ⟨p, hp⟩ ⟨q, hq⟩ + A ⟨q, hq⟩ ⟨p, hp⟩)
        + s ^ 2 * A ⟨q, hq⟩ ⟨q, hq⟩ := by
  rw [Matrix.mul_assoc, Matrix.mul_apply,
    Finset.sum_congr rfl (fun k _ => by
      rw [Matrix.transpose_apply, givens_col_fp p q hp hq hpq c s k,
        givens_AJ_p A p q hp hq hpq c s k]),
    sum_pair ⟨p, hp⟩ ⟨q, hq⟩ (fin_pq_ne p q hp hq hpq) c (-s)
      (fun k => c * A k ⟨p, hp⟩ - s * A k ⟨q, hq⟩)]
  ring

include hpq in
/-- The `(q, q)` entry of the conjugation `Jᵀ · A · J`. -/
theorem givens_conj_qq :
    ((toM n (Spec.arrGivens n p q c s))ᵀ * A * toM n (Spec.arrGivens n p q c s)) ⟨q, hq⟩ ⟨q, hq⟩
      = s ^ 2 * A ⟨p, hp⟩ ⟨p, hp⟩ + c * s * (A ⟨p, hp⟩ ⟨q, hq⟩ + A ⟨q, hq⟩ ⟨p, hp⟩)
        + c ^ 2 * A ⟨q, hq⟩ ⟨q, hq⟩ := by
  rw [Matrix.mul_assoc, Matrix.mul_apply,
    Finset.sum_congr rfl (fun k _ => by
      rw [Matrix.transpose_apply, givens_col_fq p q hp hq hpq c s k,
        givens_AJ_q A p q hp hq hpq c s k]),
    sum_pair ⟨p, hp⟩ ⟨q, hq⟩ (fin_pq_ne p q hp hq hpq) s c
      (fun k => s * A k ⟨p, hp⟩ + c * A k ⟨q, hq⟩)]
  ring

include hpq in
/-- The `(p, q)` entry of the conjugation `Jᵀ · A · J` — the entry the rotation is chosen to
annihilate. -/
theorem givens_conj_pq :
    ((toM n (Spec.arrGivens n p q c s))ᵀ * A * toM n (Spec.arrGivens n p q c s)) ⟨p, hp⟩ ⟨q, hq⟩
      = c * s * A ⟨p, hp⟩ ⟨p, hp⟩ + c ^ 2 * A ⟨p, hp⟩ ⟨q, hq⟩ - s ^ 2 * A ⟨q, hq⟩ ⟨p, hp⟩
        - c * s * A ⟨q, hq⟩ ⟨q, hq⟩ := by
  rw [Matrix.mul_assoc, Matrix.mul_apply,
    Finset.sum_congr rfl (fun k _ => by
      rw [Matrix.transpose_apply, givens_col_fp p q hp hq hpq c s k,
        givens_AJ_q A p q hp hq hpq c s k]),
    sum_pair ⟨p, hp⟩ ⟨q, hq⟩ (fin_pq_ne p q hp hq hpq) c (-s)
      (fun k => s * A k ⟨p, hp⟩ + c * A k ⟨q, hq⟩)]
  ring

/-- Any other diagonal entry `(o, o)` with `o ∉ {p, q}` is unchanged by the conjugation. -/
theorem givens_conj_other (o : Fin n) (hop : o.val ≠ p) (hoq : o.val ≠ q) :
    ((toM n (Spec.arrGivens n p q c s))ᵀ * A * toM n (Spec.arrGivens n p q c s)) o o = A o o := by
  rw [Matrix.mul_assoc, Matrix.mul_apply,
    Finset.sum_congr rfl (fun k _ => by
      rw [Matrix.transpose_apply, givens_col_other p q c s o k hop hoq,
        givens_AJ_other A p q c s o hop hoq k])]
  simp

/-! ## The per-rotation off-diagonal decrease -/

include hpq in
/-- **Per-rotation Jacobi progress.** For a *symmetric pivot* (`A[q,p] = A[p,q]`) and the Givens
rotation that *annihilates* it (`(Jᵀ A J)[p,q] = 0`), conjugating `A` by `J` decreases the squared
off-diagonal mass by exactly `2 · A[p,q]²`:

`‖offDiag(Jᵀ A J)‖² = ‖offDiag A‖² − 2 · A[p,q]²`.

This is the exact finite identity behind Jacobi convergence: each rotation removes `2 · A[p,q]²` of
off-diagonal mass. The *rate* over a sweep (how the pivots are chosen and how fast the total mass
falls) is the research-grade part Mathlib has no theory for. -/
theorem jacobi_off_decrease (hcs : c ^ 2 + s ^ 2 = 1)
    (hsym : A ⟨q, hq⟩ ⟨p, hp⟩ = A ⟨p, hp⟩ ⟨q, hq⟩)
    (hannih : ((toM n (Spec.arrGivens n p q c s))ᵀ * A * toM n (Spec.arrGivens n p q c s))
      ⟨p, hp⟩ ⟨q, hq⟩ = 0) :
    offSq ((toM n (Spec.arrGivens n p q c s))ᵀ * A * toM n (Spec.arrGivens n p q c s))
      = offSq A - 2 * (A ⟨p, hp⟩ ⟨q, hq⟩) ^ 2 := by
  have hpq' : (⟨p, hp⟩ : Fin n) ≠ ⟨q, hq⟩ := fin_pq_ne p q hp hq hpq
  set G := toM n (Spec.arrGivens n p q c s) with hGdef
  have hJ : Gᵀ * G = 1 := givens_orthogonal p q hp hq hpq c s hcs
  have hfrob : frobSq (Gᵀ * A * G) = frobSq A := frobSq_orthogonal_conj hJ
  have hsumP := frobSq_eq_diagSq_add_offSq (Gᵀ * A * G)
  have hsumA := frobSq_eq_diagSq_add_offSq A
  -- The annihilation equation in explicit form (using symmetry).
  have hann : c * s * (A ⟨p, hp⟩ ⟨p, hp⟩ - A ⟨q, hq⟩ ⟨q, hq⟩)
      + (c ^ 2 - s ^ 2) * A ⟨p, hp⟩ ⟨q, hq⟩ = 0 := by
    have hpq0 := givens_conj_pq A p q hp hq hpq c s
    rw [hGdef] at hannih
    rw [hannih] at hpq0
    rw [hsym] at hpq0
    linear_combination -hpq0
  -- The diagonal mass increases by exactly `2 A[p,q]²`.
  have hdiag : diagSq (Gᵀ * A * G) - diagSq A = 2 * (A ⟨p, hp⟩ ⟨q, hq⟩) ^ 2 := by
    unfold diagSq
    rw [← Finset.sum_sub_distrib,
      sum_eq_pair ⟨p, hp⟩ ⟨q, hq⟩ hpq'
        (fun o => (Gᵀ * A * G) o o ^ 2 - A o o ^ 2) ?_]
    · simp only [hGdef, givens_conj_pp A p q hp hq hpq c s, givens_conj_qq A p q hp hq hpq c s,
        hsym]
      have hba := block_diag_algebra (A ⟨p, hp⟩ ⟨p, hp⟩) (A ⟨p, hp⟩ ⟨q, hq⟩) (A ⟨q, hq⟩ ⟨q, hq⟩) c s
        hcs hann
      linear_combination hba
    · intro o hop' hoq'
      have hop : o.val ≠ p := fun h => hop' (Fin.ext h)
      have hoq : o.val ≠ q := fun h => hoq' (Fin.ext h)
      simp only [hGdef, givens_conj_other A p q c s o hop hoq, sub_self]
  linarith [hfrob, hsumP, hsumA, hdiag]

end Spec.Factorization
