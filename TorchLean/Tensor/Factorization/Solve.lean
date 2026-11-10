/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Factorization.Orthonormal

/-!
# Exact solves through the Cholesky factor

Over `ℝ` we prove that the solves in `TorchLean.Tensor.Factorization` return
exact solutions: forward and back substitution solve triangular systems with nonzero diagonal, and
`solveRidgeFn K γ b` solves `(K + γ I) x = b` when `K` is symmetric and the Cholesky pivots of
`K + γ I` are positive. These theorems give no floating-point error bound, and the code tests
neither the diagonal nor the pivots at run time.

## Main results

* `mulVec_triSolveLowerFn` and `mulVec_triSolveUpperFn`: `L y = b` and `U x = y` for lower- and
  upper-triangular matrices with nonzero diagonal.
* `mulVec_cholSolveFn`: `x = cholSolveFn L b` satisfies `L (Lᵀ x) = b`.
* `mulVec_solveRidgeFn`: `x = solveRidgeFn K γ b` satisfies `(K + γ I) x = b`, from
  `isCholesky_of_pos` and `mulVec_cholSolveFn`.
* `solveRidge_spec`: the same equation, entry by entry, for `Tensor.solveRidge`.

## Method

Each substitution is one left fold over `List.finRange n` that sets one unknown per step. After `m`
steps the unvisited unknowns are `0` and the equations of the `m` rows already visited hold.
Setting the next unknown adds one term to every row sum: triangularity makes it `0` in the rows
already visited, and the division by the diagonal entry makes the next row's equation hold.
-/

@[expose] public section

namespace TorchLean.Factorization

open Matrix
open scoped BigOperators

variable {n : Nat}

/-- If `P 0 a` holds and step `j` of the fold takes `P j` to `P (j + 1)`, then `P n` holds at the
end of the left fold over `List.finRange n`. -/
private theorem foldl_finRange_induction {σ : Type*} (P : Nat → σ → Prop) (f : σ → Fin n → σ)
    {a : σ} (h0 : P 0 a) (hstep : ∀ (j : Fin n) (x : σ), P j.val x → P (j.val + 1) (f x j)) :
    P n ((List.finRange n).foldl f a) := by
  induction n with
  | zero => exact h0
  | succ m ih =>
    rw [List.finRange_succ_last, List.foldl_append, List.foldl_map, List.foldl_cons,
      List.foldl_nil]
    exact hstep (Fin.last m) _
      (ih (fun x j => f x j.castSucc) (fun j x hx => hstep j.castSucc x hx))

/-- Setting coordinate `r` of `x` to `v` changes `∑ k, c k * x k` by `c r * (v - x r)`. -/
private theorem sum_mul_update (c x : Fin n → ℝ) (r : Fin n) (v : ℝ) :
    ∑ k, c k * Function.update x r v k = ∑ k, c k * x k + c r * (v - x r) := by
  have h : ∀ k, c k * Function.update x r v k
      = c k * x k + if k = r then c r * (v - x r) else 0 := by
    intro k
    by_cases hk : k = r
    · subst hk
      rw [Function.update_self, ite_eq_left rfl]
      ring
    · rw [Function.update_of_ne hk, ite_eq_right hk, add_zero]
  simp only [h, Finset.sum_add_distrib, Finset.sum_ite_eq', Finset.mem_univ, ite_true]

/-- Over `ℝ`, forward substitution solves `L y = b` when `L` is lower-triangular (`L i j = 0` for
`i < j`) with nonzero diagonal: `Matrix.of L *ᵥ triSolveLowerFn L b = b`. -/
theorem mulVec_triSolveLowerFn (L : Fin n → Fin n → ℝ) (hL : ∀ i j, i < j → L i j = 0)
    (hdiag : ∀ i, L i i ≠ 0) (b : Fin n → ℝ) :
    Matrix.of L *ᵥ Factorization.triSolveLowerFn L b = b := by
  have key := foldl_finRange_induction (σ := Fin n → ℝ)
    (fun m acc => (∀ k : Fin n, m ≤ k.val → acc k = 0) ∧
      ∀ i : Fin n, i.val < m → ∑ k, L i k * acc k = b i)
    (fun acc j => Function.update acc j ((b j - Factorization.dotFn (L j) acc) / L j j))
    (a := fun _ => 0) ⟨fun _ _ => rfl, fun i hi => absurd hi (Nat.not_lt_zero _)⟩ ?_
  · funext i
    exact key.2 i i.isLt
  · rintro j acc ⟨hz, heq⟩
    refine ⟨fun k hk => ?_, fun i hi => ?_⟩
    · rw [Function.update_of_ne (fun h => by subst h; omega)]
      exact hz k (by omega)
    · rw [sum_mul_update]
      by_cases hij : i.val < j.val
      · rw [heq i hij, hL i j (Fin.lt_def.mpr hij), zero_mul, add_zero]
      · obtain rfl : i = j := Fin.ext (by omega)
        rw [hz i le_rfl, sub_zero, dotFn_eq_sum, mul_div_cancel₀ _ (hdiag i)]
        ring

/-- Over `ℝ`, back substitution solves `U x = y` when `U` is upper-triangular (`U i j = 0` for
`j < i`) with nonzero diagonal: `Matrix.of U *ᵥ triSolveUpperFn U y = y`. -/
theorem mulVec_triSolveUpperFn (U : Fin n → Fin n → ℝ) (hU : ∀ i j, j < i → U i j = 0)
    (hdiag : ∀ i, U i i ≠ 0) (y : Fin n → ℝ) :
    Matrix.of U *ᵥ Factorization.triSolveUpperFn U y = y := by
  have key := foldl_finRange_induction (σ := Fin n → ℝ)
    (fun m acc => (∀ k : Fin n, k.val + m < n → acc k = 0) ∧
      ∀ i : Fin n, n ≤ i.val + m → ∑ k, U i k * acc k = y i)
    (fun acc j => Function.update acc j.rev
      ((y j.rev - Factorization.dotFn (U j.rev) acc) / U j.rev j.rev))
    (a := fun _ => 0) ⟨fun _ _ => rfl, fun i hi => absurd hi (by omega)⟩ ?_
  · funext i
    unfold Factorization.triSolveUpperFn
    rw [List.finRange_reverse, List.foldl_map]
    exact key.2 i (by omega)
  · rintro j acc ⟨hz, heq⟩
    have hr : j.rev.val + j.val + 1 = n := by rw [Fin.val_rev]; omega
    refine ⟨fun k hk => ?_, fun i hi => ?_⟩
    · rw [Function.update_of_ne (fun h => by subst h; omega)]
      exact hz k (by omega)
    · rw [sum_mul_update]
      by_cases hij : n ≤ i.val + j.val
      · rw [heq i hij, hU i j.rev (Fin.lt_def.mpr (by omega)), zero_mul, add_zero]
      · obtain rfl : i = j.rev := Fin.ext (by omega)
        rw [hz _ (by omega), sub_zero, dotFn_eq_sum, mul_div_cancel₀ _ (hdiag _)]
        ring

/-- Over `ℝ`, `x = cholSolveFn L b` solves `L (Lᵀ x) = b` when `L` is lower-triangular
(`L i j = 0` for `i < j`) with nonzero diagonal. -/
theorem mulVec_cholSolveFn (L : Fin n → Fin n → ℝ) (hL : ∀ i j, i < j → L i j = 0)
    (hdiag : ∀ i, L i i ≠ 0) (b : Fin n → ℝ) :
    (Matrix.of L * (Matrix.of L)ᵀ) *ᵥ Factorization.cholSolveFn L b = b := by
  rw [← Matrix.mulVec_mulVec, show (Matrix.of L)ᵀ = Matrix.of (fun i k => L k i) from rfl]
  unfold Factorization.cholSolveFn
  rw [mulVec_triSolveUpperFn (fun i k => L k i) (fun i j hji => hL j i hji) hdiag,
    mulVec_triSolveLowerFn L hL hdiag]

/-- Over `ℝ`, `x = solveRidgeFn K γ b` solves `(K + γ I) x = b` when `K` is symmetric and every
Cholesky pivot `choleskyFn (addScaledIdFn K γ) j j` is positive. -/
theorem mulVec_solveRidgeFn (K : Fin n → Fin n → ℝ) (γ : ℝ) (hsymm : ∀ i j, K i j = K j i)
    (hpos : ∀ j : Fin n, 0 < Factorization.choleskyFn (Factorization.addScaledIdFn K γ) j j)
    (b : Fin n → ℝ) :
    (Matrix.of K + γ • 1) *ᵥ Factorization.solveRidgeFn K γ b = b := by
  have hA : Matrix.of K + γ • (1 : Matrix (Fin n) (Fin n) ℝ)
      = Matrix.of (Factorization.addScaledIdFn K γ) := by
    ext i j
    rw [Matrix.add_apply, Matrix.smul_apply, Matrix.one_apply, Matrix.of_apply, Matrix.of_apply,
      smul_eq_mul, mul_ite, mul_one, mul_zero]
    rfl
  have hsymmA : ∀ i j, Factorization.addScaledIdFn K γ i j =
      Factorization.addScaledIdFn K γ j i := by
    intro i j
    by_cases h : i = j
    · rw [h]
    · simp only [Factorization.addScaledIdFn, hsymm i j, ite_eq_right h, ite_eq_right (Ne.symm h)]
  obtain ⟨hlow, hmul⟩ := isCholesky_of_pos _ hsymmA hpos
  rw [hA, hmul]
  exact mulVec_cholSolveFn _ hlow (fun j => (hpos j).ne') b

/-- For a symmetric tensor `K` over `ℝ` whose Cholesky pivots
`choleskyFn (addScaledIdFn (get2 K) γ) j j` are all positive, `x = K.solveRidge γ b` solves
`(K + γ I) x = b`: `Σ_j K[i,j] · x[j] + γ · x[i] = b[i]` for every `i`. -/
theorem solveRidge_spec (K : Tensor ℝ [n, n]) (γ : ℝ)
    (hsymm : ∀ i j, Tensor.get2 K i j = Tensor.get2 K j i)
    (hpos : ∀ j : Fin n, 0 <
      Factorization.choleskyFn (Factorization.addScaledIdFn (Tensor.get2 K) γ) j j)
    (b : Tensor ℝ [n]) (i : Fin n) :
    ∑ j, Tensor.get2 K i j * (K.solveRidge γ b).getScalar j + γ * (K.solveRidge γ b).getScalar i
      = b.getScalar i := by
  have hx : ∀ k, (K.solveRidge γ b).getScalar k
      = Factorization.solveRidgeFn (Tensor.get2 K) γ (fun k => b.getScalar k) k :=
    fun k => Tensor.getScalar_ofFn _ k
  have h :=
    congrFun (mulVec_solveRidgeFn (Tensor.get2 K) γ hsymm hpos (fun k => b.getScalar k)) i
  rw [Matrix.add_mulVec, Matrix.smul_mulVec, Matrix.one_mulVec] at h
  simp only [hx]
  exact h

end TorchLean.Factorization
