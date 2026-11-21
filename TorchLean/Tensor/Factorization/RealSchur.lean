/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Factorization.Decomposition

/-!
# Existence of real Schur decompositions

`IsRealSchur A Q T` (`TorchLean.Tensor.Factorization.Decomposition`) says when `(T, Q)` is a real
Schur decomposition of a real `A`, the output of `scipy.linalg.schur` with `output='real'`: `Q`
orthogonal, `A = Q T Qᵀ`, and `T` quasi-upper triangular, with 2 × 2 diagonal blocks of
negative discriminant. Here every real square matrix has one.

## Main results

* `exists_isRealSchur_of_blocks`: an orthogonal `Q₁` for which `Q₁ᵀ A Q₁` is zero below its first
  `k` rows in its first `k` columns, with a quasi-triangular upper left `k × k` block, and a real
  Schur decomposition of every matrix of size `n - k`, give one of `A`.
* `exists_isRealSchur`: every real matrix has a real Schur decomposition, by strong induction on
  its size, splitting off a real eigenvector (`k = 1`) or, when there is none, the plane spanned
  by the real and imaginary parts of a complex eigenvector (`k = 2`, `isRealSchur_two`).
-/

@[expose] public section

namespace TorchLean.Factorization

open Matrix

variable {n : ℕ}

/-- A real Schur decomposition of `A` from an orthogonal `Q₁` for which `B = Q₁ᵀ A Q₁` is zero
below its first `k` rows in its first `k` columns, with its upper left `k × k` block `C`
quasi-triangular (`IsRealSchur C 1 C`), given one of every matrix of size `l = n - k`: one
`(Q₂, T₂)` of `B`'s lower right block gives `Q = Q₁ diag(1, Q₂)` and `T = Qᵀ A Q`, with blocks
`C` and `T₂` on its diagonal and zeros below them. -/
theorem exists_isRealSchur_of_blocks {k l : ℕ} (h : k + l = n) {A Q₁ : Matrix (Fin n) (Fin n) ℝ}
    (hQ₁ : Q₁ᵀ * Q₁ = 1)
    (hB : ∀ i j : Fin n, k ≤ i.val → j.val < k → (Q₁ᵀ * A * Q₁) i j = 0)
    (hC : IsRealSchur ((Q₁ᵀ * A * Q₁).submatrix (Fin.castLE ((Nat.le_add_right k l).trans h.le))
      (Fin.castLE ((Nat.le_add_right k l).trans h.le))) 1
      ((Q₁ᵀ * A * Q₁).submatrix (Fin.castLE ((Nat.le_add_right k l).trans h.le))
        (Fin.castLE ((Nat.le_add_right k l).trans h.le))))
    (hl : ∀ A' : Matrix (Fin l) (Fin l) ℝ, ∃ Q T, IsRealSchur A' Q T) :
    ∃ Q T, IsRealSchur A Q T := by
  -- Split the indices into the first `k` and the last `l` (`e`).
  obtain ⟨e, he⟩ : ∃ e : Fin k ⊕ Fin l ≃ Fin n, e = finSumFinEquiv.trans (finCongr h) :=
    ⟨_, rfl⟩
  have hel : ∀ a, (e (Sum.inl a)).val = a.val := fun a => by simp [he]
  have her : ∀ b, (e (Sum.inr b)).val = k + b.val := fun b => by simp [he]
  have hcl : (Fin.castLE ((Nat.le_add_right k l).trans h.le) : Fin k → Fin n) =
      fun a => e (Sum.inl a) := funext fun a => Fin.ext (by simp [hel])
  obtain ⟨B, hBd⟩ : ∃ B, Q₁ᵀ * A * Q₁ = B := ⟨_, rfl⟩
  rw [hBd, hcl] at hC
  rw [hBd] at hB
  obtain ⟨-, hCH, hC3, -⟩ := hC
  -- Decompose the lower right block of `B`, and let `P` be `diag(1, Q₂)` in the split indices.
  obtain ⟨Q₂, T₂, hQ₂, hH₂, hT₂, hA₂⟩ :=
    hl (B.submatrix (fun b => e (Sum.inr b)) fun b => e (Sum.inr b))
  obtain ⟨P, hP⟩ : ∃ P : Matrix (Fin n) (Fin n) ℝ,
      P = (fromBlocks 1 0 0 Q₂).submatrix e.symm e.symm := ⟨_, rfl⟩
  have hPe : P.submatrix e e = fromBlocks 1 0 0 Q₂ := by
    rw [hP, submatrix_submatrix, Equiv.symm_comp_self, submatrix_id_id]
  have hPP : Pᵀ * P = 1 := by
    rw [hP, transpose_submatrix, submatrix_mul_equiv, fromBlocks_transpose, fromBlocks_multiply]
    simp [hQ₂]
  -- Split, `B` is block upper triangular, so `Pᵀ B P` has blocks `C`, `B₁₂ Q₂`, 0 and `T₂`.
  have hBe : B.submatrix e e = fromBlocks (B.submatrix (fun a => e (Sum.inl a)) fun a =>
      e (Sum.inl a)) (B.submatrix (fun a => e (Sum.inl a)) fun b => e (Sum.inr b)) 0
      (B.submatrix (fun b => e (Sum.inr b)) fun b => e (Sum.inr b)) := by
    ext (a | a) (b | b)
    · rfl
    · rfl
    · exact hB _ _ (by rw [her]; omega) (by rw [hel]; exact b.isLt)
    · rfl
  have hT₂' : Q₂ᵀ * B.submatrix (fun b => e (Sum.inr b)) (fun b => e (Sum.inr b)) * Q₂ = T₂ := by
    rw [hA₂]
    simp only [← Matrix.mul_assoc]
    rw [hQ₂, Matrix.one_mul, Matrix.mul_assoc, hQ₂, Matrix.mul_one]
  have hTe : (Pᵀ * B * P).submatrix e e = fromBlocks (B.submatrix (fun a => e (Sum.inl a))
      fun a => e (Sum.inl a)) (B.submatrix (fun a => e (Sum.inl a)) (fun b => e (Sum.inr b)) * Q₂)
      0 T₂ := by
    rw [← submatrix_mul_equiv _ _ _ e, ← submatrix_mul_equiv _ _ _ e, ← transpose_submatrix, hPe,
      hBe, fromBlocks_transpose, fromBlocks_multiply, fromBlocks_multiply, ← hT₂']
    simp
  have h11 : ∀ a b, (Pᵀ * B * P) (e (Sum.inl a)) (e (Sum.inl b)) = B (e (Sum.inl a))
      (e (Sum.inl b)) := fun a b => by
    simpa using congrFun (congrFun hTe (Sum.inl a)) (Sum.inl b)
  have h21 : ∀ a b, (Pᵀ * B * P) (e (Sum.inr a)) (e (Sum.inl b)) = 0 := fun a b => by
    simpa using congrFun (congrFun hTe (Sum.inr a)) (Sum.inl b)
  have h22 : ∀ a b, (Pᵀ * B * P) (e (Sum.inr a)) (e (Sum.inr b)) = T₂ a b := fun a b => by
    simpa using congrFun (congrFun hTe (Sum.inr a)) (Sum.inr b)
  have hQ : (Q₁ * P)ᵀ * (Q₁ * P) = 1 := by
    rw [transpose_mul, Matrix.mul_assoc, ← Matrix.mul_assoc Q₁ᵀ, hQ₁, Matrix.one_mul, hPP]
  have hT : (Q₁ * P)ᵀ * A * (Q₁ * P) = Pᵀ * B * P := by
    rw [← hBd, transpose_mul]
    simp only [Matrix.mul_assoc]
  -- The band and block conditions hold block by block: the upper left from `C`, the lower right
  -- from `T₂`, and the lower left is zero.
  refine ⟨Q₁ * P, Pᵀ * B * P, hQ, upperBand_one_iff.mp fun i j hij => ?_,
    fun i j hij hne => ?_, ?_⟩
  · obtain ⟨x, rfl⟩ := e.surjective i
    obtain ⟨y, rfl⟩ := e.surjective j
    rcases x with a | a <;> rcases y with b | b
    · rw [h11]
      exact upperBand_one_iff.mpr hCH a b (by rwa [hel, hel] at hij)
    · have := a.isLt
      rw [hel, her] at hij
      omega
    · exact h21 a b
    · rw [h22]
      exact upperBand_one_iff.mpr hH₂ a b (by rw [her, her] at hij; omega)
  · obtain ⟨x, rfl⟩ := e.surjective i
    obtain ⟨y, rfl⟩ := e.surjective j
    rcases x with a | a <;> rcases y with b | b
    · rw [h11] at hne
      obtain ⟨hd, hz⟩ := hC3 a b (by rwa [hel, hel] at hij) hne
      refine ⟨by simpa only [h11, submatrix_apply] using hd, fun l' hl' => ?_⟩
      obtain ⟨z, rfl⟩ := e.surjective l'
      rcases z with c | c
      · rw [h11]
        exact hz c (by rwa [hel, hel] at hl')
      · exact h21 c a
    · have := a.isLt
      rw [hel, her] at hij
      omega
    · exact absurd (h21 a b) hne
    · rw [h22] at hne
      obtain ⟨hd, hz⟩ := hT₂ a b (by rw [her, her] at hij; omega) hne
      refine ⟨by simpa only [h22] using hd, fun l' hl' => ?_⟩
      obtain ⟨z, rfl⟩ := e.surjective l'
      rcases z with c | c
      · have := c.isLt
        rw [her, hel] at hl'
        omega
      · rw [h22]
        exact hz c (by rw [her, her] at hl'; omega)
  · have h := eq_mul_mul_conjTranspose (Q := Q₁ * P)
      (by rw [conjTranspose_eq_transpose_of_trivial]; exact hQ) A
    rw [conjTranspose_eq_transpose_of_trivial] at h
    rw [← hT]
    exact h

/-- Every real matrix has a real Schur decomposition. By strong induction on the size, with
`exists_isRealSchur_of_blocks`: a real eigenvector of unit norm is the first column of an
orthogonal `Q₁` (`exists_unitary_extension`), `k = 1`; without one, an eigenvector `x + i y` of
`A` over `ℂ` (`exists_isEig`) with eigenvalue `a + i b` has `b ≠ 0`, and `A x = a x - b y`,
`A y = b x + a y`, so the first two columns of the `Q₁` of a QR factorization `[x y] = Q₁ R`
(`exists_isQR`) span a plane `A` maps to itself, `k = 2`, on which `Q₁ᵀ A Q₁` is
`R₂ [[a, b], [-b, a]] R₂⁻¹` for `R₂` the invertible upper left block of `R`, with discriminant
`-4 b² < 0`. -/
theorem exists_isRealSchur (A : Matrix (Fin n) (Fin n) ℝ) : ∃ Q T, IsRealSchur A Q T := by
  induction n using Nat.strong_induction_on with
  | _ n ih =>
  rcases Nat.eq_zero_or_pos n with rfl | hn
  · exact ⟨1, A, by simp, upperBand_one_iff.mp fun i => i.elim0, fun i => i.elim0, by simp⟩
  -- A real eigenvector, scaled to unit norm, is the first column of `Q₁`, and `k = 1`.
  by_cases hv : ∃ (v : Fin n → ℝ) (μ : ℝ), v ≠ 0 ∧ A *ᵥ v = μ • v
  · obtain ⟨v, μ, hv0, hAv⟩ := hv
    obtain ⟨c, -, hc⟩ := exists_unit_smul hv0
    obtain ⟨Q₁, hQ₁, hcol⟩ := exists_unitary_extension (s := {⟨0, hn⟩}) (fun _ => c • v)
      (by simpa using hc)
    rw [conjTranspose_eq_transpose_of_trivial] at hQ₁
    have hcol' : ∀ r, Q₁ r ⟨0, hn⟩ = (c • v) r := hcol _ (Set.mem_singleton _)
    have hAQ : ∀ r, (A * Q₁) r ⟨0, hn⟩ = μ * Q₁ r ⟨0, hn⟩ := fun r => by
      have h1 : (A * Q₁) r ⟨0, hn⟩ = (A *ᵥ (c • v)) r := by
        simp only [Matrix.mul_apply, mulVec, dotProduct, hcol']
      rw [h1, mulVec_smul, hAv, hcol']
      simp only [Pi.smul_apply, smul_eq_mul]
      ring
    refine exists_isRealSchur_of_blocks (k := 1) (l := n - 1) (by omega) hQ₁ (fun i j hi hj => ?_)
      ⟨by simp, upperBand_one_iff.mp fun i j h => absurd h (by have := i.isLt; omega),
        fun i j h => absurd h (by have := i.isLt; omega), by simp⟩ (ih (n - 1) (by omega))
    obtain rfl : j = ⟨0, hn⟩ := Fin.ext (by simp; omega)
    have h2 : (Q₁ᵀ * A * Q₁) i ⟨0, hn⟩ = μ * (Q₁ᵀ * Q₁) i ⟨0, hn⟩ := by
      rw [Matrix.mul_assoc, Matrix.mul_apply, Matrix.mul_apply, Finset.mul_sum]
      exact Finset.sum_congr rfl fun r _ => by rw [hAQ]; ring
    rw [h2, hQ₁, one_apply_ne (fun h => by rw [Fin.ext_iff] at h; simp at h; omega), mul_zero]
  -- Otherwise take a complex eigenvector `x + i y` with eigenvalue `a + i b`.
  · obtain ⟨w, V, -, hV, hAV, -⟩ := exists_isEig (A.map ((↑) : ℝ → ℂ))
    have hz : ∀ r, ∑ c, (A r c : ℂ) * V c ⟨0, hn⟩ = V r ⟨0, hn⟩ * w ⟨0, hn⟩ := fun r => by
      have h := congrFun (congrFun hAV r) ⟨0, hn⟩
      rw [mul_diagonal, Matrix.mul_apply] at h
      simpa using h
    have hz0 : ∃ r, V r ⟨0, hn⟩ ≠ 0 := by
      by_contra h
      push Not at h
      have := hV ⟨0, hn⟩
      simp [Matrix.mul_apply, h] at this
    obtain ⟨x, hx⟩ : ∃ x : Fin n → ℝ, x = fun r => (V r ⟨0, hn⟩).re := ⟨_, rfl⟩
    obtain ⟨y, hy⟩ : ∃ y : Fin n → ℝ, y = fun r => (V r ⟨0, hn⟩).im := ⟨_, rfl⟩
    obtain ⟨a, ha⟩ : ∃ a, a = (w ⟨0, hn⟩).re := ⟨_, rfl⟩
    obtain ⟨b, hb⟩ : ∃ b, b = (w ⟨0, hn⟩).im := ⟨_, rfl⟩
    have hAx : A *ᵥ x = a • x - b • y := funext fun r => by
      have h := congrArg Complex.re (hz r)
      rw [Complex.re_sum, Complex.mul_re] at h
      simp only [Complex.re_ofReal_mul] at h
      simp only [mulVec, dotProduct, hx, hy, ha, hb, Pi.sub_apply, Pi.smul_apply, smul_eq_mul]
      rw [h]
      ring
    have hAy : A *ᵥ y = b • x + a • y := funext fun r => by
      have h := congrArg Complex.im (hz r)
      rw [Complex.im_sum, Complex.mul_im] at h
      simp only [Complex.im_ofReal_mul] at h
      simp only [mulVec, dotProduct, hx, hy, ha, hb, Pi.add_apply, Pi.smul_apply, smul_eq_mul]
      rw [h]
      ring
    have hxy : x ≠ 0 ∨ y ≠ 0 := by
      by_contra h
      push Not at h
      obtain ⟨r, hr⟩ := hz0
      exact hr (Complex.ext (by simpa [hx] using congrFun h.1 r)
        (by simpa [hy] using congrFun h.2 r))
    -- With `b = 0`, `x` or `y` would be a real eigenvector.
    have hb0 : b ≠ 0 := by
      rintro rfl
      rcases hxy with h | h
      · exact hv ⟨x, a, h, by rw [hAx, zero_smul, sub_zero]⟩
      · exact hv ⟨y, a, h, by rw [hAy, zero_smul, zero_add]⟩
    have hx0 : x ≠ 0 := by
      rintro rfl
      have hy0 : y = 0 := by
        have h := hAx
        rw [mulVec_zero, smul_zero, zero_sub, eq_comm, neg_eq_zero] at h
        exact (smul_eq_zero.mp h).resolve_left hb0
      rcases hxy with h | h
      · exact h rfl
      · exact h hy0
    -- `n ≥ 2`, since every `1 × 1` matrix has the real eigenvector `1`.
    obtain ⟨n, rfl⟩ : ∃ n', n = n' + 2 := by
      refine ⟨n - 2, ?_⟩
      by_contra hn2
      obtain rfl : n = 1 := by omega
      refine hv ⟨fun _ => 1, A 0 0, fun h => by simpa using congrFun h 0, funext fun r => ?_⟩
      fin_cases r
      simp [mulVec, dotProduct]
    obtain ⟨M, hM⟩ : ∃ M : Matrix (Fin (n + 2)) (Fin 2) ℝ, M = of fun r c => ![x r, y r] c :=
      ⟨_, rfl⟩
    -- `[x y] = Q R` with `R` upper triangular, so `x` and `y` are combinations of the first
    -- two columns of `Q`.
    obtain ⟨Q, R, hQ, hU, -, hMQ⟩ := exists_isQR M
    rw [conjTranspose_eq_transpose_of_trivial] at hQ
    have hxv : x = Q *ᵥ fun t => R t 0 := funext fun r => by
      have := congrFun (congrFun hMQ r) 0
      simp only [hM, of_apply, cons_val_zero] at this
      exact this
    have hyv : y = Q *ᵥ fun t => R t 1 := funext fun r => by
      have := congrFun (congrFun hMQ r) 1
      simp only [hM, of_apply, cons_val_one] at this
      exact this
    have hR0 : ∀ t : Fin (n + 2), t ≠ 0 → R t 0 = 0 := fun t ht => hU t 0 (by
      simp only [Fin.val_zero, add_zero]
      exact Nat.pos_of_ne_zero fun h => ht (Fin.ext (by rw [h, Fin.val_zero])))
    have hR1 : ∀ t : Fin (n + 2), t ≠ 0 ∧ t ≠ 1 → R t 1 = 0 := fun t ⟨h0, h1⟩ => hU t 1 (by
      have h0' : t.val ≠ 0 := fun h => h0 (Fin.ext (by rw [h, Fin.val_zero]))
      have h1' : t.val ≠ 1 := fun h => h1 (Fin.ext (by rw [h, Fin.val_one]))
      simp only [Fin.val_one, add_zero]
      omega)
    have hcol0 : ∀ (N : Matrix (Fin (n + 2)) (Fin (n + 2)) ℝ) r,
        (N *ᵥ fun t => R t 0) r = N r 0 * R 0 0 := fun N r => by
      simp only [mulVec, dotProduct]
      exact Fintype.sum_eq_single 0 fun t ht => by rw [hR0 t ht, mul_zero]
    have hcol1 : ∀ (N : Matrix (Fin (n + 2)) (Fin (n + 2)) ℝ) r,
        (N *ᵥ fun t => R t 1) r = N r 0 * R 0 1 + N r 1 * R 1 1 := fun N r => by
      simp only [mulVec, dotProduct]
      exact Fintype.sum_eq_add 0 1 (by simp) fun t ht => by rw [hR1 t ht, mul_zero]
    have hp : R 0 0 ≠ 0 := fun h0 => hx0 (funext fun r => by
      rw [hxv, hcol0, h0, mul_zero, Pi.zero_apply])
    -- `R 1 1 = 0` would make `y` a multiple of `x`, and `x` a real eigenvector.
    have hs : R 1 1 ≠ 0 := fun h0 => by
      refine hv ⟨x, a - b * (R 0 1 / R 0 0), hx0, ?_⟩
      rw [hAx]
      funext r
      have h1 := congrFun hxv r
      have h2 := congrFun hyv r
      rw [hcol0] at h1
      rw [hcol1, h0, mul_zero, add_zero] at h2
      simp only [Pi.sub_apply, Pi.smul_apply, smul_eq_mul, h1, h2]
      field_simp
    have hr0 : Qᵀ *ᵥ x = fun t => R t 0 := by rw [hxv, mulVec_mulVec, hQ, one_mulVec]
    have hr1 : Qᵀ *ᵥ y = fun t => R t 1 := by rw [hyv, mulVec_mulVec, hQ, one_mulVec]
    -- `B = Qᵀ A Q` acts on the first two columns of `R` as `A` acts on `x` and `y`.
    obtain ⟨B, hBd⟩ : ∃ B, Qᵀ * A * Q = B := ⟨_, rfl⟩
    have hB0 : B *ᵥ (fun t => R t 0) = a • (fun t => R t 0) - b • fun t => R t 1 := by
      rw [← hBd, ← mulVec_mulVec, ← mulVec_mulVec, ← hxv, hAx, mulVec_sub, mulVec_smul,
        mulVec_smul, hr0, hr1]
    have hB1 : B *ᵥ (fun t => R t 1) = b • (fun t => R t 0) + a • fun t => R t 1 := by
      rw [← hBd, ← mulVec_mulVec, ← mulVec_mulVec, ← hyv, hAy, mulVec_add, mulVec_smul,
        mulVec_smul, hr0, hr1]
    have e0 : ∀ i, B i 0 * R 0 0 = a * R i 0 - b * R i 1 := fun i => by
      have := congrFun hB0 i
      rwa [hcol0] at this
    have e1 : ∀ i, B i 0 * R 0 1 + B i 1 * R 1 1 = b * R i 0 + a * R i 1 := fun i => by
      have := congrFun hB1 i
      rwa [hcol1] at this
    have h10 : R 1 0 = 0 := hR0 1 (by simp)
    refine exists_isRealSchur_of_blocks (k := 2) (l := n) (by omega) hQ (fun i j hi hj => ?_) ?_
      (ih n (by omega))
    -- Below the leading `2 × 2` block the first two columns of `B` are zero.
    · rw [hBd]
      have hi0 : i ≠ 0 := fun h => by rw [h] at hi; simp at hi
      have hi1 : i ≠ 1 := fun h => by rw [h] at hi; simp at hi
      have hb0' : B i 0 = 0 := by
        have := e0 i
        rw [hR0 i hi0, hR1 i ⟨hi0, hi1⟩, mul_zero, mul_zero, sub_zero] at this
        exact (mul_eq_zero.mp this).resolve_right hp
      obtain h | h : j.val = 0 ∨ j.val = 1 := by omega
      · rw [show j = 0 from Fin.ext (by simpa using h)]
        exact hb0'
      · rw [show j = 1 from Fin.ext (by simpa using h)]
        have := e1 i
        rw [hR0 i hi0, hR1 i ⟨hi0, hi1⟩, hb0', mul_zero, zero_mul, zero_add, mul_zero,
          add_zero] at this
        exact (mul_eq_zero.mp this).resolve_right hs
    -- The leading block has discriminant `-4 b² < 0`.
    · rw [hBd, eta_fin_two (B.submatrix _ _)]
      have c0 : (Fin.castLE (by omega : 2 ≤ n + 2) (0 : Fin 2) : Fin (n + 2)) = 0 :=
        Fin.ext (by simp)
      have c1 : (Fin.castLE (by omega : 2 ≤ n + 2) (1 : Fin 2) : Fin (n + 2)) = 1 :=
        Fin.ext (by simp)
      simp only [submatrix_apply, c0, c1]
      apply isRealSchur_two
      have f0 := e0 0
      have f1 := e1 0
      have f2 := e0 1
      have f3 := e1 1
      rw [h10] at f2
      rw [h10] at f3
      have g1 : R 0 0 * R 1 1 * (B 0 0 - B 1 1) = -2 * b * R 0 1 * R 1 1 := by
        linear_combination R 1 1 * f0 - R 0 0 * f3 + R 0 1 * f2
      have g2 : R 0 0 * B 0 1 * R 1 1 = b * (R 0 0 ^ 2 + R 0 1 ^ 2) := by
        linear_combination R 0 0 * f1 - R 0 1 * f0
      have g3 : ((B 0 0 - B 1 1) ^ 2 + 4 * B 0 1 * B 1 0) * (R 0 0 * R 1 1) ^ 2 =
          -4 * b ^ 2 * (R 0 0 * R 1 1) ^ 2 := by
        have : ((B 0 0 - B 1 1) ^ 2 + 4 * B 0 1 * B 1 0) * (R 0 0 * R 1 1) ^ 2 =
            (R 0 0 * R 1 1 * (B 0 0 - B 1 1)) ^ 2 +
              4 * (R 0 0 * B 0 1 * R 1 1) * (B 1 0 * R 0 0) * R 1 1 := by ring
        rw [this, g1, g2]
        linear_combination 4 * b * (R 0 0 ^ 2 + R 0 1 ^ 2) * R 1 1 * f2
      rw [mul_right_cancel₀ (pow_ne_zero 2 (mul_ne_zero hp hs)) g3]
      have := lt_of_le_of_ne (sq_nonneg b) (Ne.symm (pow_ne_zero 2 hb0))
      linarith

end TorchLean.Factorization
