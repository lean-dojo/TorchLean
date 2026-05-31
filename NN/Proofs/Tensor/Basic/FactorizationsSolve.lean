/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Tensor.Basic.FactorizationsReconstruction
public import NN.Proofs.Tensor.Basic.FactorizationsOrthonormal
public import Mathlib.Data.Real.StarOrdered

/-!
# The Cholesky linear solve and the kernel-ridge (Tikhonov) solve

[`FactorizationsReconstruction`](./FactorizationsReconstruction.lean) proved that the executable
Cholesky factor satisfies `A = L · Lᵀ` exactly (over `ℝ`, under positive pivots). This file uses that
to verify the *linear solve* built on top of it — forward/back substitution — and hence the
kernel-ridge solve `(K + γ·I)·x = b` that is the numerical heart of CHD `solve_variationnal`.

## Main results

* `triSolveLowerFn_mulVec` / `triSolveUpperFn_mulVec` — forward/back substitution are correct: for a
  lower- (resp. upper-) triangular matrix with nonzero diagonal, the computed vector solves
  `L · y = b` (resp. `U · x = y`) **exactly**. These are finite, non-iterative algorithms, so the
  identity is exact over `ℝ` — no residual/asymptotic caveat.
* `cholSolveFn_mulVec` — composing the two substitutions through a Cholesky factor `L` solves
  `(L · Lᵀ) · x = b` exactly.
* `solveRidgeFn_mulVec` — the kernel-ridge solve: if the Cholesky pivots of `K + γ·I` are positive
  (the success condition), then `solveRidgeFn K γ b` solves `(K + γ·I)·x = b` exactly.
* `choleskyFn_diag_pos_of_posDef` — the **keystone**: a positive-definite matrix has strictly positive
  executable Cholesky pivots (the radicand `A[j,j] − Σ_{k<j} L[j,k]² > 0` at each step), proved via an
  explicit Schur-complement quadratic-form witness.
* `solveRidgeFn_mulVec_of_posSemidef` (and its tensor-level form) — composing the two: for a
  positive-semidefinite kernel `K` and `γ > 0`, `solveRidgeFn K γ b` solves `(K + γ·I)·x = b` exactly,
  with **no pivot hypothesis**. This is the fully discharged verified `solve_variationnal`.

## Method

Each substitution is a `Function.update` fold over the index list (`finRange n` forward, its reverse
for back-substitution). The key observation is that **no induction on the solved values is needed**:
the entry `yᵢ` is *defined* to make row `i` of the equation hold, so unfolding its definition and
using triangularity (the not-yet-visited and structurally-zero terms drop out of the row dot product)
gives `(L · y)ᵢ = bᵢ` directly. Two generic lemmas — `foldl_update_read` (the value written at the
split index) and `foldl_update_stable` (earlier entries are never overwritten) — capture the fold
bookkeeping; `sum_split_lt_eq_gt` performs the `k < i / k = i / k > i` trichotomy on the row sum.
-/

@[expose] public section

namespace Spec.Factorization.Reconstruction

open Matrix
open scoped BigOperators

variable {n : Nat}

/-! ## Generic `Function.update`-fold bookkeeping -/

/-- An update-fold never changes an index it does not visit. -/
theorem foldl_update_not_mem (H : (Fin n → ℝ) → Fin n → ℝ) (l : List (Fin n))
    (init : Fin n → ℝ) {x : Fin n} (hx : x ∉ l) :
    (l.foldl (fun acc j => Function.update acc j (H acc j)) init) x = init x := by
  induction l generalizing init with
  | nil => simp
  | cons a t ih =>
      rw [List.foldl_cons, ih (Function.update init a (H init a))
        (fun h => hx (List.mem_cons_of_mem _ h))]
      have hxa : x ≠ a := by rintro rfl; exact hx (by simp)
      exact Function.update_of_ne hxa _ _

/-- Reading an update-fold over `l₁ ++ i :: l₂` at the split index `i` (not revisited in `l₂`)
returns the step value applied to the `l₁`-prefix state. -/
theorem foldl_update_read (H : (Fin n → ℝ) → Fin n → ℝ) (l₁ l₂ : List (Fin n))
    (init : Fin n → ℝ) {i : Fin n} (hi : i ∉ l₂) :
    ((l₁ ++ i :: l₂).foldl (fun acc j => Function.update acc j (H acc j)) init) i
      = H (l₁.foldl (fun acc j => Function.update acc j (H acc j)) init) i := by
  rw [List.foldl_append, List.foldl_cons, foldl_update_not_mem H l₂ _ hi, Function.update_self]

/-- An update-fold over `l₁ ++ i :: l₂` agrees with its `l₁`-prefix at any index `≠ i` not in `l₂`. -/
theorem foldl_update_stable (H : (Fin n → ℝ) → Fin n → ℝ) (l₁ l₂ : List (Fin n))
    (init : Fin n → ℝ) {i m : Fin n} (hm : m ∉ l₂) (hmi : m ≠ i) :
    ((l₁ ++ i :: l₂).foldl (fun acc j => Function.update acc j (H acc j)) init) m
      = (l₁.foldl (fun acc j => Function.update acc j (H acc j)) init) m := by
  rw [List.foldl_append, List.foldl_cons, foldl_update_not_mem H l₂ _ hm,
    Function.update_of_ne hmi]

/-! ## Splitting a `Fin n` sum at an index -/

/-- Split a sum over `Fin n` into the `k < i`, `k = i`, and `k > i` parts. -/
theorem sum_split_lt_eq_gt (i : Fin n) (f : Fin n → ℝ) :
    (∑ k, f k) = (∑ k, if k.val < i.val then f k else 0) + f i
      + (∑ k, if i.val < k.val then f k else 0) := by
  rw [show f i = ∑ k, (if k = i then f k else 0) by
        rw [Finset.sum_ite_eq' Finset.univ i f]; simp]
  rw [← Finset.sum_add_distrib, ← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro k _
  rcases lt_trichotomy k.val i.val with h | h | h
  · have hne : k ≠ i := fun e => by rw [e] at h; exact lt_irrefl _ h
    rw [if_pos h, if_neg hne, if_neg (by linarith), add_zero, add_zero]
  · have hki : k = i := Fin.ext h
    rw [if_neg (by linarith), if_pos hki, if_neg (by linarith), zero_add, add_zero]
  · have hne : k ≠ i := fun e => by rw [e] at h; exact lt_irrefl _ h
    rw [if_neg (by linarith), if_neg hne, if_pos h, zero_add, zero_add]

/-! ## `finRange` order splits -/

/-- `finRange n` splits at index `i` as the strictly-smaller prefix, `i`, then the strictly-larger
suffix. -/
theorem finRange_split (i : Fin n) :
    List.finRange n
      = (List.finRange n).take i.val ++ i :: (List.finRange n).drop (i.val + 1) := by
  have hlen : i.val < (List.finRange n).length := by rw [List.length_finRange]; exact i.isLt
  conv_lhs => rw [← List.take_append_drop i.val (List.finRange n)]
  congr 1
  rw [List.drop_eq_getElem_cons hlen]
  congr 1
  simp [List.getElem_finRange]

/-! ## Forward substitution solves a lower-triangular system exactly -/

/-- **Forward substitution is correct.** For a lower-triangular `L` (`L i j = 0` when `i < j`) with
nonzero diagonal, `triSolveLowerFn L b` solves `L · y = b` exactly: row `i` of `L · y` is `bᵢ`. -/
theorem triSolveLowerFn_mulVec (L : Fin n → Fin n → ℝ)
    (hlow : ∀ i j, i < j → L i j = 0) (hdiag : ∀ i, L i i ≠ 0) (b : Fin n → ℝ) (i : Fin n) :
    (∑ k, L i k * Spec.triSolveLowerFn L b k) = b i := by
  set H : (Fin n → ℝ) → Fin n → ℝ := fun acc j => (b j - Spec.dotFn (L j) acc) / L j j with hH
  set y := Spec.triSolveLowerFn L b with hy
  set pre := ((List.finRange n).take i.val).foldl
    (fun acc j => Function.update acc j (H acc j)) (fun _ => 0) with hpre
  -- `y` is the update-fold over `finRange n`.
  have hyeq : y = (List.finRange n).foldl (fun acc j => Function.update acc j (H acc j))
      (fun _ => 0) := rfl
  -- `i` is not revisited after its turn, and not in its own prefix.
  have hi₂ : i ∉ (List.finRange n).drop (i.val + 1) := fun hmem => by
    have := mem_drop_finRange hmem; linarith
  have hi₁ : i ∉ (List.finRange n).take i.val := fun hmem => by
    have := mem_take_finRange hmem; exact lt_irrefl _ this
  -- value written at `i`, prefix value at `i`, and stability for `k < i`.
  have hy_i : y i = (b i - Spec.dotFn (L i) pre) / L i i := by
    rw [hyeq]; conv_lhs => rw [finRange_split i]
    rw [foldl_update_read H _ _ _ hi₂]
  have hpre_i : pre i = 0 := by rw [hpre, foldl_update_not_mem H _ _ hi₁]
  have hy_lt : ∀ m : Fin n, m.val < i.val → y m = pre m := by
    intro m hm
    have hm₂ : m ∉ (List.finRange n).drop (i.val + 1) := fun hmem => by
      have := mem_drop_finRange hmem; linarith
    have hmi : m ≠ i := fun e => by rw [e] at hm; exact lt_irrefl _ hm
    rw [hyeq]; conv_lhs => rw [finRange_split i]
    rw [foldl_update_stable H _ _ _ hm₂ hmi]
  -- the row dot product `dotFn (L i) pre` is the masked partial sum over `k < i`.
  have hdot : Spec.dotFn (L i) pre = ∑ k, if k.val < i.val then L i k * pre k else 0 := by
    rw [dotFn_eq_sum, sum_split_lt_eq_gt i (fun k => L i k * pre k)]
    rw [hpre_i, mul_zero]
    rw [show (∑ k, if i.val < k.val then L i k * pre k else 0) = 0 by
          apply Finset.sum_eq_zero; intro k _
          by_cases hk : i.val < k.val
          · rw [if_pos hk, hlow i k (by exact hk), zero_mul]
          · rw [if_neg hk]]
    ring
  -- assemble row `i` of `L · y`.
  rw [sum_split_lt_eq_gt i (fun k => L i k * y k)]
  rw [show (∑ k, if i.val < k.val then L i k * y k else 0) = 0 by
        apply Finset.sum_eq_zero; intro k _
        by_cases hk : i.val < k.val
        · rw [if_pos hk, hlow i k (by exact hk), zero_mul]
        · rw [if_neg hk]]
  rw [show (∑ k, if k.val < i.val then L i k * y k else 0)
        = ∑ k, if k.val < i.val then L i k * pre k else 0 by
        apply Finset.sum_congr rfl; intro k _
        by_cases hk : k.val < i.val
        · rw [if_pos hk, if_pos hk, hy_lt k hk]
        · rw [if_neg hk, if_neg hk]]
  rw [← hdot, hy_i, add_zero]
  have hdi : L i i ≠ 0 := hdiag i
  field_simp
  ring

/-! ## Back substitution solves an upper-triangular system exactly -/

/-- `(finRange n).reverse` splits at index `i` as the strictly-larger block (reversed suffix),
then `i`, then the strictly-smaller block (reversed prefix). -/
theorem finRange_reverse_split (i : Fin n) :
    (List.finRange n).reverse
      = ((List.finRange n).drop (i.val + 1)).reverse
        ++ i :: ((List.finRange n).take i.val).reverse := by
  conv_lhs => rw [finRange_split i]
  rw [List.reverse_append, List.reverse_cons, List.append_assoc, List.singleton_append]

/-- **Back substitution is correct.** For an upper-triangular `U` (`U i j = 0` when `j < i`) with
nonzero diagonal, `triSolveUpperFn U c` solves `U · x = c` exactly: row `i` of `U · x` is `cᵢ`. -/
theorem triSolveUpperFn_mulVec (U : Fin n → Fin n → ℝ)
    (hup : ∀ i j, j < i → U i j = 0) (hdiag : ∀ i, U i i ≠ 0) (c : Fin n → ℝ) (i : Fin n) :
    (∑ k, U i k * Spec.triSolveUpperFn U c k) = c i := by
  set H : (Fin n → ℝ) → Fin n → ℝ := fun acc j => (c j - Spec.dotFn (U j) acc) / U j j with hH
  set y := Spec.triSolveUpperFn U c with hy
  set pre := (((List.finRange n).drop (i.val + 1)).reverse).foldl
    (fun acc j => Function.update acc j (H acc j)) (fun _ => 0) with hpre
  have hyeq : y = ((List.finRange n).reverse).foldl
      (fun acc j => Function.update acc j (H acc j)) (fun _ => 0) := rfl
  have hi₂ : i ∉ ((List.finRange n).take i.val).reverse := fun hmem => by
    rw [List.mem_reverse] at hmem; have := mem_take_finRange hmem; exact lt_irrefl _ this
  have hi₁ : i ∉ ((List.finRange n).drop (i.val + 1)).reverse := fun hmem => by
    rw [List.mem_reverse] at hmem; have := mem_drop_finRange hmem; linarith
  have hy_i : y i = (c i - Spec.dotFn (U i) pre) / U i i := by
    rw [hyeq]; conv_lhs => rw [finRange_reverse_split i]
    rw [foldl_update_read H _ _ _ hi₂]
  have hpre_i : pre i = 0 := by rw [hpre, foldl_update_not_mem H _ _ hi₁]
  have hy_gt : ∀ m : Fin n, i.val < m.val → y m = pre m := by
    intro m hm
    have hm₂ : m ∉ ((List.finRange n).take i.val).reverse := fun hmem => by
      rw [List.mem_reverse] at hmem; have := mem_take_finRange hmem; linarith
    have hmi : m ≠ i := fun e => by rw [e] at hm; exact lt_irrefl _ hm
    rw [hyeq]; conv_lhs => rw [finRange_reverse_split i]
    rw [foldl_update_stable H _ _ _ hm₂ hmi]
  have hdot : Spec.dotFn (U i) pre = ∑ k, if i.val < k.val then U i k * pre k else 0 := by
    rw [dotFn_eq_sum, sum_split_lt_eq_gt i (fun k => U i k * pre k)]
    rw [hpre_i, mul_zero]
    rw [show (∑ k, if k.val < i.val then U i k * pre k else 0) = 0 by
          apply Finset.sum_eq_zero; intro k _
          by_cases hk : k.val < i.val
          · rw [if_pos hk, hup i k (by exact hk), zero_mul]
          · rw [if_neg hk]]
    ring
  rw [sum_split_lt_eq_gt i (fun k => U i k * y k)]
  rw [show (∑ k, if k.val < i.val then U i k * y k else 0) = 0 by
        apply Finset.sum_eq_zero; intro k _
        by_cases hk : k.val < i.val
        · rw [if_pos hk, hup i k (by exact hk), zero_mul]
        · rw [if_neg hk]]
  rw [show (∑ k, if i.val < k.val then U i k * y k else 0)
        = ∑ k, if i.val < k.val then U i k * pre k else 0 by
        apply Finset.sum_congr rfl; intro k _
        by_cases hk : i.val < k.val
        · rw [if_pos hk, if_pos hk, hy_gt k hk]
        · rw [if_neg hk, if_neg hk]]
  rw [← hdot, hy_i]
  have hdi : U i i ≠ 0 := hdiag i
  field_simp
  ring

/-! ## The Cholesky linear solve -/

/-- **Cholesky solve is correct.** For a lower-triangular `L` with nonzero diagonal, the two-pass
substitution `cholSolveFn L b` solves `(L · Lᵀ) · x = b` exactly. -/
theorem cholSolveFn_mulVec (L : Fin n → Fin n → ℝ)
    (hlow : ∀ i j, i < j → L i j = 0) (hdiag : ∀ i, L i i ≠ 0) (b : Fin n → ℝ) :
    (Matrix.of L * (Matrix.of L)ᵀ) *ᵥ (Spec.cholSolveFn L b) = b := by
  set z := Spec.triSolveLowerFn L b with hz
  set U : Fin n → Fin n → ℝ := fun i k => L k i with hU
  have hup : ∀ i j, j < i → U i j = 0 := fun i j hji => hlow j i hji
  have hUdiag : ∀ i, U i i ≠ 0 := fun i => hdiag i
  have hUp : (Matrix.of L)ᵀ *ᵥ (Spec.cholSolveFn L b) = z := by
    funext i
    have hx : Spec.cholSolveFn L b = Spec.triSolveUpperFn U z := rfl
    show (∑ k, ((Matrix.of L)ᵀ i k) * Spec.cholSolveFn L b k) = z i
    simp only [Matrix.transpose_apply, Matrix.of_apply]
    rw [hx]
    exact triSolveUpperFn_mulVec U hup hUdiag z i
  have hLow : (Matrix.of L) *ᵥ z = b := by
    funext i
    show (∑ k, (Matrix.of L i k) * z k) = b i
    simp only [Matrix.of_apply]
    exact triSolveLowerFn_mulVec L hlow hdiag b i
  calc (Matrix.of L * (Matrix.of L)ᵀ) *ᵥ (Spec.cholSolveFn L b)
      = Matrix.of L *ᵥ ((Matrix.of L)ᵀ *ᵥ (Spec.cholSolveFn L b)) := by
        rw [Matrix.mulVec_mulVec]
    _ = Matrix.of L *ᵥ z := by rw [hUp]
    _ = b := hLow

/-! ## The kernel-ridge (Tikhonov) solve -/

/-- **Kernel-ridge solve is correct (conditional on Cholesky success).** If `K` is symmetric and the
Cholesky pivots of `K + γ·I` are positive — exactly the condition under which the SPD Cholesky
succeeds — then `solveRidgeFn K γ b` solves `(K + γ·I)·x = b` exactly. This is the verified core of
CHD `solve_variationnal`; the positive-pivot hypothesis is discharged unconditionally for an SPD
`K + γ·I` (PSD kernel `K`, `γ > 0`) in the companion development. -/
theorem solveRidgeFn_mulVec (K : Fin n → Fin n → ℝ) (γ : ℝ) (b : Fin n → ℝ)
    (hsymm : ∀ i j, K i j = K j i)
    (hpos : ∀ j : Fin n, 0 < Spec.choleskyFn (Spec.addScaledIdFn K γ) j j) :
    (Matrix.of (Spec.addScaledIdFn K γ)) *ᵥ (Spec.solveRidgeFn K γ b) = b := by
  set A := Spec.addScaledIdFn K γ with hA
  have hAsymm : ∀ i j, A i j = A j i := by
    intro i j
    show K i j + (if i = j then γ else 0) = K j i + (if j = i then γ else 0)
    rw [hsymm i j]
    by_cases h : i = j
    · rw [h]
    · rw [if_neg h, if_neg (fun e => h e.symm)]
  obtain ⟨hlowM, hreconM⟩ := isCholesky_of_pos A hAsymm hpos
  have hlow : ∀ i j, i < j → Spec.choleskyFn A i j = 0 := fun i j hij => by
    have := hlowM i j hij; simpa using this
  have hdiag : ∀ i, Spec.choleskyFn A i i ≠ 0 := fun i => ne_of_gt (hpos i)
  have hxeq : Spec.solveRidgeFn K γ b = Spec.cholSolveFn (Spec.choleskyFn A) b := rfl
  rw [hxeq, hreconM]
  exact cholSolveFn_mulVec (Spec.choleskyFn A) hlow hdiag b

/-! ## The regularized matrix `K + γ·I` is symmetric positive-definite

For a positive-semidefinite kernel `K` and regularization `γ > 0`, `K + γ·I` is positive definite —
the precondition under which the Cholesky-based `solveRidgeFn` is the genuine linear solve. Combined
with the keystone below (`choleskyFn_diag_pos_of_posDef`: an SPD matrix has strictly positive
executable Cholesky pivots), this discharges the positive-pivot hypothesis of `solveRidgeFn_mulVec`
unconditionally, giving `solveRidgeFn_mulVec_of_posSemidef`. -/

/-- `Matrix.of (addScaledIdFn K γ) = Matrix.of K + γ • 1`. -/
theorem of_addScaledIdFn (K : Fin n → Fin n → ℝ) (γ : ℝ) :
    Matrix.of (Spec.addScaledIdFn K γ) = Matrix.of K + γ • (1 : Matrix (Fin n) (Fin n) ℝ) := by
  ext i j
  simp only [Matrix.of_apply, Matrix.add_apply, Matrix.smul_apply, Matrix.one_apply,
    Spec.addScaledIdFn, smul_eq_mul]
  by_cases h : i = j <;> simp [h]

/-- **The regularized (ridge) matrix is SPD.** For a PSD kernel `K` and `γ > 0`, `K + γ·I` is positive
definite. This is the precondition that makes the Cholesky ridge solve `solveRidgeFn` well-posed
(its Cholesky factorization exists with positive pivots). -/
theorem posDef_addScaledIdFn {K : Fin n → Fin n → ℝ} (hK : (Matrix.of K).PosSemidef)
    {γ : ℝ} (hγ : 0 < γ) : (Matrix.of (Spec.addScaledIdFn K γ)).PosDef := by
  rw [of_addScaledIdFn]
  exact Matrix.PosDef.posSemidef_add hK (Matrix.PosDef.one.smul hγ)

/-! ## Keystone: a positive-definite matrix has strictly positive Cholesky pivots

The remaining ingredient that makes `solveRidgeFn_mulVec` unconditional for SPD inputs: for a
*positive-definite* `A`, every executable Cholesky pivot is `> 0`. Equivalently, the radicand
`A[j,j] − Σ_{k<j} L[j,k]² > 0` at every step, so the `√` never sees a non-positive argument.

The argument is the classical Schur-complement fact, formalized as an **explicit quadratic-form
witness** (so it needs no matrix inverse). By strong induction on `j`, the leading `j`-block
reconstructs from the pivots below `j` (`choleskyFn_dot_eq_local`). Back-substitution
(`triSolveUpperFn`, already proven correct in this file) produces a vector `z` with `z j = 1` whose
`A`-quadratic form `zᵀ A z` is exactly the radicand; positive-definiteness forces `zᵀ A z > 0`. -/

/-- A double Gram sum collapses to a sum of squares:
`∑ᵢ∑ⱼ zᵢ·(∑ₗ Mᵢₗ Mⱼₗ)·zⱼ = ∑ₗ (∑ᵢ zᵢ Mᵢₗ)²`. (The `A = M·Mᵀ` reconstruction turns the witness
quadratic form into a manifestly nonnegative shape.) -/
theorem double_sum_gram (z : Fin n → ℝ) (M : Fin n → Fin n → ℝ) :
    (∑ i, ∑ j, z i * ((∑ l, M i l * M j l) * z j))
      = ∑ l, (∑ i, z i * M i l) * (∑ i, z i * M i l) := by
  have hexp : (∑ i, ∑ j, z i * ((∑ l, M i l * M j l) * z j))
      = ∑ i, ∑ j, ∑ l, (z i * M i l) * (z j * M j l) := by
    refine Finset.sum_congr rfl (fun i _ => Finset.sum_congr rfl (fun j _ => ?_))
    rw [Finset.sum_mul, Finset.mul_sum]
    exact Finset.sum_congr rfl (fun l _ => by ring)
  rw [hexp,
    show (∑ i, ∑ j, ∑ l, (z i * M i l) * (z j * M j l))
        = ∑ i, ∑ l, ∑ j, (z i * M i l) * (z j * M j l)
      from Finset.sum_congr rfl (fun i _ => Finset.sum_comm),
    Finset.sum_comm]
  refine Finset.sum_congr rfl (fun l _ => ?_)
  rw [Fintype.sum_mul_sum (fun i => z i * M i l) (fun j => z j * M j l)]

/-- **Localized per-entry Cholesky reconstruction.** The proof of `choleskyFn_dot_eq` only uses the
positivity of the *smaller* pivot `L[j,j]`, so for `j ≤ i` the reconstruction `∑ₖ L[i,k]·L[j,k] =
A[i,j]` holds assuming only `0 < L[j,j]` — not global positivity. This is what powers the strong
induction in `choleskyFn_diag_pos_of_posDef`. -/
theorem choleskyFn_dot_eq_local (A : Fin n → Fin n → ℝ) {i j : Fin n}
    (hjpos : 0 < Spec.choleskyFn A j j) (hji : j.val ≤ i.val) :
    (∑ k, Spec.choleskyFn A i k * Spec.choleskyFn A j k) = A i j := by
  set L := Spec.choleskyFn A with hL
  have key : ∀ k : Fin n, L i k * L j k
      = (if k.val < j.val then L i k * L j k else 0) + (if k = j then L i j * L j j else 0) := by
    intro k
    rcases lt_trichotomy k.val j.val with h | h | h
    · have hne : k ≠ j := fun hk => by rw [hk] at h; exact lt_irrefl _ h
      rw [if_pos h, if_neg hne, add_zero]
    · have hkj : k = j := Fin.ext h
      rw [if_neg (by rw [h]; exact lt_irrefl _), if_pos hkj, zero_add, hkj]
    · have hne : k ≠ j := fun hk => by rw [hk] at h; exact lt_irrefl _ h
      rw [if_neg (Nat.not_lt.mpr (le_of_lt h)), if_neg hne, add_zero,
        show L j k = 0 from by rw [hL]; exact Spec.Factorization.choleskyFn_lower_triangular A h,
        mul_zero]
  rw [show (∑ k, L i k * L j k)
      = ∑ k, ((if k.val < j.val then L i k * L j k else 0) + (if k = j then L i j * L j j else 0))
      from Finset.sum_congr rfl (fun k _ => key k),
    Finset.sum_add_distrib, Finset.sum_ite_eq' Finset.univ j (fun _ => L i j * L j j)]
  simp only [Finset.mem_univ, if_true]
  rcases eq_or_lt_of_le hji with heq | hlt
  · have hij' : i = j := Fin.ext heq.symm
    subst hij'
    have hrad : 0 < A i i - (∑ k, if k.val < i.val then L i k * L i k else 0) := by
      have hp := hjpos
      rw [hL, choleskyFn_diag_eq] at hp
      exact Real.sqrt_pos.mp hp
    have hsq : L i i * L i i = A i i - (∑ k, if k.val < i.val then L i k * L i k else 0) := by
      conv_lhs => rw [hL, choleskyFn_diag_eq A i]
      exact Real.mul_self_sqrt hrad.le
    rw [hsq]; ring
  · have hne : L j j ≠ 0 := ne_of_gt hjpos
    have hmul : L i j * L j j
        = A i j - (∑ k, if k.val < j.val then L i k * L j k else 0) := by
      rw [hL, choleskyFn_offdiag_eq A hlt, div_mul_eq_mul_div, mul_div_assoc, div_self hne, mul_one]
    rw [hmul]; ring

/-- **The radicand / Schur keystone.** For a positive-definite `A`, every executable Cholesky pivot is
strictly positive: `0 < L[j,j]`. Hence the SPD Cholesky succeeds and the ridge solve is exact. -/
theorem choleskyFn_diag_pos_of_posDef (A : Fin n → Fin n → ℝ) (hpd : (Matrix.of A).PosDef)
    (m : Fin n) : 0 < Spec.choleskyFn A m m := by
  -- symmetry of `A` from Hermitian-ness
  have hsymm : ∀ i j, A i j = A j i := by
    intro i j
    have h := hpd.1.apply i j
    simp only [Matrix.of_apply, star_trivial] at h
    exact h.symm
  -- strong induction on `m.val`
  suffices H : ∀ N : Nat, ∀ m : Fin n, m.val = N → 0 < Spec.choleskyFn A m m by
    exact H m.val m rfl
  intro N
  induction N using Nat.strong_induction_on with
  | _ N IH =>
    intro m hmN
    have ihpos : ∀ i : Fin n, i.val < m.val → 0 < Spec.choleskyFn A i i := fun i hi =>
      IH i.val (hmN ▸ hi) i rfl
    -- reduce to positivity of the radicand
    rw [choleskyFn_diag_eq A m, Real.sqrt_pos]
    -- localized reconstruction for pairs `≤ m` with at least one index `< m`
    have hAij : ∀ i j : Fin n, i.val ≤ m.val → j.val ≤ m.val → (i.val < m.val ∨ j.val < m.val) →
        (∑ l, Spec.choleskyFn A i l * Spec.choleskyFn A j l) = A i j := by
      intro i j _ _ hor
      rcases le_total j.val i.val with hle | hle
      · have hjm : j.val < m.val := by
          rcases hor with h | h
          · exact lt_of_le_of_lt hle h
          · exact h
        exact choleskyFn_dot_eq_local A (ihpos j hjm) hle
      · have him : i.val < m.val := by
          rcases hor with h | h
          · exact h
          · exact lt_of_le_of_lt hle h
        rw [show (∑ l, Spec.choleskyFn A i l * Spec.choleskyFn A j l)
              = ∑ l, Spec.choleskyFn A j l * Spec.choleskyFn A i l
            from Finset.sum_congr rfl (fun l _ => mul_comm _ _),
          choleskyFn_dot_eq_local A (ihpos i him) hle, hsymm j i]
    -- the back-substitution system solving `(Lₘᵀ) z = −(row m of L)` on the leading block
    set U' : Fin n → Fin n → ℝ := fun l i =>
      if l.val < m.val then (if i.val < m.val then Spec.choleskyFn A i l else 0)
      else (if i = l then 1 else 0) with hU'
    set c : Fin n → ℝ := fun l => if l.val < m.val then -(Spec.choleskyFn A m l) else 0 with hc
    set x' := Spec.triSolveUpperFn U' c with hx'
    set z : Fin n → ℝ := fun i => if i = m then 1 else x' i with hz
    have zm1 : z m = 1 := by simp [hz]
    -- `U'` is upper-triangular with nonzero diagonal
    have hup : ∀ a b : Fin n, b.val < a.val → U' a b = 0 := by
      intro a b hba
      simp only [hU']
      by_cases ha : a.val < m.val
      · rw [if_pos ha]
        by_cases hb : b.val < m.val
        · rw [if_pos hb]; exact Spec.Factorization.choleskyFn_lower_triangular A hba
        · rw [if_neg hb]
      · rw [if_neg ha, if_neg (by intro e; rw [e] at hba; exact lt_irrefl _ hba)]
    have hUdiag : ∀ a : Fin n, U' a a ≠ 0 := by
      intro a
      simp only [hU']
      by_cases ha : a.val < m.val
      · rw [if_pos ha, if_pos ha]; exact ne_of_gt (ihpos a ha)
      · rw [if_neg ha]; simp
    have hsolve : ∀ l : Fin n, (∑ i, U' l i * x' i) = c l := fun l =>
      triSolveUpperFn_mulVec U' hup hUdiag c l
    -- entries `≥ m` of the solve vanish
    have hx'_ge : ∀ l : Fin n, m.val ≤ l.val → x' l = 0 := by
      intro l hl
      have hlm : ¬ l.val < m.val := Nat.not_lt.mpr hl
      have hsum : (∑ i, U' l i * x' i) = x' l := by
        rw [show (∑ i, U' l i * x' i) = ∑ i, (if i = l then x' i else 0) from
          Finset.sum_congr rfl (fun i _ => by
            simp only [hU', if_neg hlm]
            by_cases hi : i = l
            · rw [if_pos hi, if_pos hi, one_mul]
            · rw [if_neg hi, if_neg hi, zero_mul]),
          Finset.sum_ite_eq' Finset.univ l (fun i => x' i)]
        simp
      have hcl : c l = 0 := by simp only [hc, if_neg hlm]
      have := hsolve l
      rw [hsum, hcl] at this
      exact this
    have hz_gt : ∀ i : Fin n, m.val < i.val → z i = 0 := by
      intro i hi
      have hne : i ≠ m := fun e => by rw [e] at hi; exact lt_irrefl _ hi
      simp only [hz, if_neg hne]
      exact hx'_ge i (le_of_lt hi)
    -- the witness annihilates the leading columns of `L`
    have hker : ∀ l : Fin n, l.val < m.val → (∑ i, z i * Spec.choleskyFn A i l) = 0 := by
      intro l hlm
      have hpl : (∑ i, (if i.val < m.val then x' i * Spec.choleskyFn A i l else 0))
          = -(Spec.choleskyFn A m l) := by
        have h := hsolve l
        rw [show c l = -(Spec.choleskyFn A m l) from by simp only [hc, if_pos hlm]] at h
        rw [← h]
        refine Finset.sum_congr rfl (fun i _ => ?_)
        simp only [hU', if_pos hlm]
        by_cases hi : i.val < m.val
        · rw [if_pos hi, if_pos hi, mul_comm]
        · rw [if_neg hi, if_neg hi, zero_mul]
      have tw : ∀ i : Fin n, z i * Spec.choleskyFn A i l
          = (if i.val < m.val then x' i * Spec.choleskyFn A i l else 0)
            + (if i = m then Spec.choleskyFn A m l else 0) := by
        intro i
        rcases lt_trichotomy i.val m.val with hi | hi | hi
        · have hne : i ≠ m := fun e => by rw [e] at hi; exact lt_irrefl _ hi
          rw [if_pos hi, if_neg hne, add_zero]
          simp only [hz, if_neg hne]
        · have him : i = m := Fin.ext hi
          rw [if_neg (by rw [hi]; exact lt_irrefl _), if_pos him, zero_add, him, zm1, one_mul]
        · have hne : i ≠ m := fun e => by rw [e] at hi; exact lt_irrefl _ hi
          rw [if_neg (Nat.not_lt.mpr (le_of_lt hi)), if_neg hne, add_zero,
            show z i = 0 from hz_gt i hi, zero_mul]
      rw [show (∑ i, z i * Spec.choleskyFn A i l)
          = ∑ i, ((if i.val < m.val then x' i * Spec.choleskyFn A i l else 0)
            + (if i = m then Spec.choleskyFn A m l else 0))
          from Finset.sum_congr rfl (fun i _ => tw i),
        Finset.sum_add_distrib, Finset.sum_ite_eq' Finset.univ m (fun _ => Spec.choleskyFn A m l)]
      simp only [Finset.mem_univ, if_true]
      rw [hpl]; ring
    -- value of the column-`l` contraction `∑ᵢ zᵢ L[i,l]`
    have wval : ∀ l : Fin n, (∑ i, z i * Spec.choleskyFn A i l)
        = if l = m then Spec.choleskyFn A m m else 0 := by
      intro l
      rcases lt_trichotomy l.val m.val with hl | hl | hl
      · rw [if_neg (fun e => by rw [e] at hl; exact lt_irrefl _ hl)]
        exact hker l hl
      · have hlm : l = m := Fin.ext hl
        rw [if_pos hlm, hlm]
        have hper : ∀ i : Fin n, z i * Spec.choleskyFn A i m
            = if i = m then Spec.choleskyFn A m m else 0 := by
          intro i
          rcases lt_trichotomy i.val m.val with hi | hi | hi
          · rw [if_neg (fun e => by rw [e] at hi; exact lt_irrefl _ hi),
              Spec.Factorization.choleskyFn_lower_triangular A hi, mul_zero]
          · have him : i = m := Fin.ext hi
            rw [if_pos him, him, zm1, one_mul]
          · rw [if_neg (fun e => by rw [e] at hi; exact lt_irrefl _ hi),
              show z i = 0 from hz_gt i hi, zero_mul]
        rw [Finset.sum_congr rfl (fun i _ => hper i),
          Finset.sum_ite_eq' Finset.univ m (fun _ => Spec.choleskyFn A m m)]
        simp
      · rw [if_neg (fun e => by rw [e] at hl; exact lt_irrefl _ hl)]
        refine Finset.sum_eq_zero (fun i _ => ?_)
        rcases Nat.lt_or_ge i.val l.val with hi | hi
        · rw [Spec.Factorization.choleskyFn_lower_triangular A hi, mul_zero]
        · rw [show z i = 0 from hz_gt i (lt_of_lt_of_le hl hi), zero_mul]
    -- the Gram term `T1 = ∑ₗ (∑ᵢ zᵢ L[i,l])² = L[m,m]²`
    have T1eval : (∑ l, (∑ i, z i * Spec.choleskyFn A i l) * (∑ i, z i * Spec.choleskyFn A i l))
        = Spec.choleskyFn A m m * Spec.choleskyFn A m m := by
      rw [show (∑ l, (∑ i, z i * Spec.choleskyFn A i l) * (∑ i, z i * Spec.choleskyFn A i l))
            = ∑ l, (if l = m then Spec.choleskyFn A m m * Spec.choleskyFn A m m else 0)
          from Finset.sum_congr rfl (fun l _ => by
            rw [wval l]
            by_cases hlm : l = m
            · rw [if_pos hlm, if_pos hlm]
            · rw [if_neg hlm, if_neg hlm, mul_zero]),
        Finset.sum_ite_eq' Finset.univ m (fun _ => Spec.choleskyFn A m m * Spec.choleskyFn A m m)]
      simp
    -- the residual term `T2 = ∑ᵢ∑ⱼ zᵢ (A[i,j] − R[i,j]) zⱼ` reduces to the `(m,m)` entry
    have T2eval : (∑ i, ∑ j, z i
          * ((A i j - ∑ l, Spec.choleskyFn A i l * Spec.choleskyFn A j l) * z j))
        = A m m - ∑ l, Spec.choleskyFn A m l * Spec.choleskyFn A m l := by
      rw [Finset.sum_eq_single m]
      · rw [Finset.sum_eq_single m]
        · rw [zm1]; ring
        · intro j _ hj
          rcases lt_trichotomy j.val m.val with hjm | hjm | hjm
          · rw [hAij m j (le_refl _) (le_of_lt hjm) (Or.inr hjm)]; ring
          · exact absurd (Fin.ext hjm) hj
          · rw [show z j = 0 from hz_gt j hjm, mul_zero, mul_zero]
        · intro h; exact absurd (Finset.mem_univ m) h
      · intro i _ hi
        refine Finset.sum_eq_zero (fun j _ => ?_)
        rcases lt_trichotomy i.val m.val with him | him | him
        · rcases lt_trichotomy j.val m.val with hjm | hjm | hjm
          · rw [hAij i j (le_of_lt him) (le_of_lt hjm) (Or.inl him)]; ring
          · rw [hAij i j (le_of_lt him) (le_of_eq hjm) (Or.inl him)]; ring
          · rw [show z j = 0 from hz_gt j hjm, mul_zero, mul_zero]
        · exact absurd (Fin.ext him) hi
        · rw [show z i = 0 from hz_gt i him, zero_mul]
      · intro h; exact absurd (Finset.mem_univ m) h
    -- splitting the full squared norm of row `m` of `L` (the `> m` part vanishes)
    have Rmm_split : (∑ l, Spec.choleskyFn A m l * Spec.choleskyFn A m l)
        = (∑ k, if k.val < m.val then Spec.choleskyFn A m k * Spec.choleskyFn A m k else 0)
          + Spec.choleskyFn A m m * Spec.choleskyFn A m m := by
      rw [sum_split_lt_eq_gt m (fun l => Spec.choleskyFn A m l * Spec.choleskyFn A m l),
        show (∑ k, if m.val < k.val then Spec.choleskyFn A m k * Spec.choleskyFn A m k else 0) = 0
          from Finset.sum_eq_zero (fun k _ => by
            by_cases hk : m.val < k.val
            · rw [if_pos hk, Spec.Factorization.choleskyFn_lower_triangular A hk, zero_mul]
            · rw [if_neg hk])]
      ring
    -- the witness quadratic form `zᵀ A z` equals the radicand
    have hqf : star z ⬝ᵥ (Matrix.of A *ᵥ z) = ∑ i, ∑ j, z i * (A i j * z j) := by
      show (∑ i, star (z i) * (∑ j, (Matrix.of A) i j * z j)) = _
      refine Finset.sum_congr rfl (fun i _ => ?_)
      rw [star_trivial, Finset.mul_sum]
      exact Finset.sum_congr rfl (fun j _ => by rw [Matrix.of_apply])
    have hQsplit : (∑ i, ∑ j, z i * (A i j * z j))
        = (∑ i, ∑ j, z i * ((∑ l, Spec.choleskyFn A i l * Spec.choleskyFn A j l) * z j))
          + (∑ i, ∑ j, z i
              * ((A i j - ∑ l, Spec.choleskyFn A i l * Spec.choleskyFn A j l) * z j)) := by
      rw [← Finset.sum_add_distrib]
      refine Finset.sum_congr rfl (fun i _ => ?_)
      rw [← Finset.sum_add_distrib]
      exact Finset.sum_congr rfl (fun j _ => by ring)
    have hqf_eq_rad : star z ⬝ᵥ (Matrix.of A *ᵥ z)
        = A m m - ∑ k, if k.val < m.val then Spec.choleskyFn A m k * Spec.choleskyFn A m k else 0 := by
      rw [hqf, hQsplit, double_sum_gram z (Spec.choleskyFn A), T1eval, T2eval, Rmm_split]
      ring
    -- positive-definiteness applied to the nonzero witness finishes it
    have hz_ne : z ≠ 0 := fun h => one_ne_zero (by
      have hzm := congrFun h m; rwa [zm1, Pi.zero_apply] at hzm)
    have hpos := hpd.dotProduct_mulVec_pos hz_ne
    rw [hqf_eq_rad] at hpos
    exact hpos

/-! ## The kernel-ridge solve, unconditional for SPD inputs -/

/-- **Kernel-ridge solve, unconditional for an SPD regularized system.** For a positive-semidefinite
kernel `K` and `γ > 0`, `solveRidgeFn K γ b` solves `(K + γ·I)·x = b` exactly — with *no* pivot
hypothesis. This is the fully discharged verified `solve_variationnal`: the keystone
`choleskyFn_diag_pos_of_posDef` supplies the positive pivots from `posDef_addScaledIdFn`. -/
theorem solveRidgeFn_mulVec_of_posSemidef (K : Fin n → Fin n → ℝ) (γ : ℝ) (b : Fin n → ℝ)
    (hK : (Matrix.of K).PosSemidef) (hγ : 0 < γ) :
    (Matrix.of (Spec.addScaledIdFn K γ)) *ᵥ (Spec.solveRidgeFn K γ b) = b := by
  have hpd : (Matrix.of (Spec.addScaledIdFn K γ)).PosDef := posDef_addScaledIdFn hK hγ
  have hsymm : ∀ i j, K i j = K j i := by
    intro i j
    have h := hK.1.apply i j
    simp only [Matrix.of_apply, star_trivial] at h
    exact h.symm
  exact solveRidgeFn_mulVec K γ b hsymm
    (fun j => choleskyFn_diag_pos_of_posDef (Spec.addScaledIdFn K γ) hpd j)

/-- **Tensor-level kernel-ridge solve, unconditional for SPD inputs.** For a tensor kernel `K` whose
matrix view is positive-semidefinite and `γ > 0`, `solveRidgeSpec K γ b` solves `(K + γ·I)·x = b`
exactly: `(K + γ·I) *ᵥ (solveRidgeSpec K γ b) = b`. -/
theorem solveRidgeSpec_mulVec_of_posSemidef (K : Spec.Tensor ℝ (.dim n (.dim n .scalar))) (γ : ℝ)
    (b : Spec.Tensor ℝ (.dim n .scalar)) (hK : (Matrix.of (Spec.toMatFn K)).PosSemidef) (hγ : 0 < γ) :
    (Matrix.of (Spec.addScaledIdFn (Spec.toMatFn K) γ)) *ᵥ (Spec.toVecFn (Spec.solveRidgeSpec K γ b))
      = Spec.toVecFn b := by
  have hround : Spec.toVecFn (Spec.solveRidgeSpec K γ b)
      = Spec.solveRidgeFn (Spec.toMatFn K) γ (Spec.toVecFn b) := by
    funext i; rfl
  rw [hround]
  exact solveRidgeFn_mulVec_of_posSemidef (Spec.toMatFn K) γ (Spec.toVecFn b) hK hγ
