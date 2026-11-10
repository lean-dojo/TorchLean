/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Analysis.InnerProductSpace.GramSchmidtOrtho
public import TorchLean.Tensor.Factorization.Reconstruction

/-!
# Orthonormality of the executable Gram–Schmidt `Q` factor (`Qᵀ Q = 1`)

Read the columns of `A` as vectors of `EuclideanSpace ℝ (Fin m)` (`gsCol`). If every diagonal entry
of `R` is positive, the `j`-th executable `Q` column is Mathlib's normalized Gram–Schmidt vector
`gramSchmidtNormed ℝ (gsCol A) j` (`toLp_qCol_eq_gramSchmidtNormed`), so orthonormality follows from
Mathlib's `gramSchmidtNormed_orthonormal'`.

## Main results

* `toLp_qCol_eq_gramSchmidtNormed`: if `0 < rMat A j j` for every `j`, then
  `WithLp.toLp 2 (qCol A k) = gramSchmidtNormed ℝ (gsCol A) k`, by strong induction on `k`.
* `qCol_orthonormal`: `dotFn (qCol A a) (qCol A b) = if a = b then 1 else 0` under positive `R`
  pivots.
* `qMat_transpose_mul_self` and `isQR_of_pos`: the matrix-level `Qᵀ Q = 1`, and the full
  `Factorization.IsQR` predicate for the executable factors, which adds the reconstruction
  `A = Q · R` and the upper-triangular `R` from
  `TorchLean.Tensor.Factorization.Reconstruction`.
* `qrQ_orthonormal`: the tensor-level corollary for the `Q` factor `(Factorization.qr A).1`.

## Method

The bridge rests on three connectors over `ℝ`: `dotFn_eq_inner` (`dotFn` is the inner product of
`EuclideanSpace`), `normFn_eq_norm` (`normFn` is its norm) and `proj_normalize`, which writes the
unnormalized Gram–Schmidt projection term with the normalized vector. The strong induction feeds
the identification of the earlier `Q` columns into `gramSchmidt_def''`, term by term.
-/

@[expose] public section

namespace TorchLean.Factorization

open Matrix
open scoped BigOperators RealInnerProductSpace
open InnerProductSpace

/-! ## Connectors between the executable scalar ops and the Euclidean inner product -/

/-- `dotFn` as a `Finset` sum. -/
theorem dotFn_eq_sum {p : Nat} (u v : Fin p → ℝ) : Factorization.dotFn u v = ∑ i, u i * v i := by
  rw [Factorization.dotFn, Fin.sum_univ_def, List.sum_eq_foldl, List.foldl_map]

/-- The executable dot product is the Euclidean inner product over `ℝ`. -/
theorem dotFn_eq_inner {p : Nat} (u v : Fin p → ℝ) :
    Factorization.dotFn u v
      = ⟪(WithLp.toLp 2 u : EuclideanSpace ℝ (Fin p)), WithLp.toLp 2 v⟫_ℝ := by
  rw [dotFn_eq_sum, PiLp.inner_apply]
  apply Finset.sum_congr rfl
  intro i _
  rw [RCLike.inner_apply', PiLp.toLp_apply, PiLp.toLp_apply]
  simp

/-- The executable Euclidean norm is the `EuclideanSpace` norm over `ℝ`. -/
theorem normFn_eq_norm {p : Nat} (v : Fin p → ℝ) :
    Factorization.normFn v = ‖(WithLp.toLp 2 v : EuclideanSpace ℝ (Fin p))‖ := by
  rw [Factorization.normFn, MathFunctions.sqrt_real, EuclideanSpace.norm_eq]
  congr 1
  rw [dotFn_eq_sum]
  apply Finset.sum_congr rfl
  intro i _
  rw [PiLp.toLp_apply, Real.norm_eq_abs, sq_abs, sq]

/-- The projection of `x` onto `w`, `(⟪w, x⟫ / ‖w‖ ^ 2) • w`, is the projection of `x` along the
normalized vector `‖w‖⁻¹ • w`. No hypothesis on `w` is needed: both sides are `0` when `w = 0`. -/
theorem proj_normalize {F : Type*} [NormedAddCommGroup F] [InnerProductSpace ℝ F] (w x : F) :
    (⟪w, x⟫_ℝ / ‖w‖ ^ 2) • w = ⟪‖w‖⁻¹ • w, x⟫_ℝ • (‖w‖⁻¹ • w) := by
  rw [real_inner_smul_left, smul_smul]
  congr 1
  rw [div_eq_mul_inv, ← inv_pow, sq]
  ring

/-- `gramSchmidtNormed` over `ℝ`, with the scalar coercion removed. -/
theorem gramSchmidtNormed_eq_inv_norm_smul {n : Nat} {F : Type*} [NormedAddCommGroup F]
    [InnerProductSpace ℝ F] (f : Fin n → F) (i : Fin n) :
    gramSchmidtNormed ℝ f i = ‖gramSchmidt ℝ f i‖⁻¹ • gramSchmidt ℝ f i := by
  rw [gramSchmidtNormed]
  norm_num

/-- A masked full sum equals the sum over `Iio`. -/
theorem sum_Iio_eq_sum_ite {n : Nat} (k : Fin n) (h : Fin n → ℝ) :
    ∑ i ∈ Finset.Iio k, h i = ∑ i, if i.val < k.val then h i else 0 := by
  rw [← Finset.sum_filter]
  congr 1
  ext i
  simp only [Finset.mem_Iio, Finset.mem_filter, Finset.mem_univ, true_and, Fin.lt_def]

/-! ## The bridge to Mathlib's `gramSchmidt` -/

section QR

variable {m n : Nat}

/-- Column `j` of `A` as a vector of `EuclideanSpace ℝ (Fin m)`. -/
noncomputable def gsCol (A : Fin m → Fin n → ℝ) (j : Fin n) : EuclideanSpace ℝ (Fin m) :=
  WithLp.toLp 2 (gsA A j)

/-- `gsCol A k` reads as the executable column `gsA A k`. -/
theorem gsCol_apply (A : Fin m → Fin n → ℝ) (k : Fin n) (r : Fin m) :
    gsCol A k r = gsA A k r := rfl

/-- If the `Q` columns before index `k` are Mathlib's normalized Gram–Schmidt vectors (`ih`) and the
pivot `rMat A k k` is positive (`hk`), the orthogonalized vector `gsV` at index `k` is Mathlib's
unnormalized `gramSchmidt` vector. -/
theorem gramSchmidt_eq_toLp_gsV (A : Fin m → Fin n → ℝ) (k : Fin n) (hk : 0 < rMat A k k)
    (ih : ∀ i : Fin n, i.val < k.val →
        (WithLp.toLp 2 (qCol A i) : EuclideanSpace ℝ (Fin m)) = gramSchmidtNormed ℝ (gsCol A) i) :
    gramSchmidt ℝ (gsCol A) k = WithLp.toLp 2 (gsV A (qsPrefix A k) k) := by
  rw [show gramSchmidt ℝ (gsCol A) k
        = gsCol A k - ∑ i ∈ Finset.Iio k,
            (⟪gramSchmidt ℝ (gsCol A) i, gsCol A k⟫_ℝ / ‖gramSchmidt ℝ (gsCol A) i‖ ^ 2)
              • gramSchmidt ℝ (gsCol A) i
      from eq_sub_of_add_eq (gramSchmidt_def'' ℝ (gsCol A) k).symm]
  -- Replace each projection term by the normalized form, then by the executable `Q` column.
  have hproj : ∀ i ∈ Finset.Iio k,
      (⟪gramSchmidt ℝ (gsCol A) i, gsCol A k⟫_ℝ / ‖gramSchmidt ℝ (gsCol A) i‖ ^ 2)
          • gramSchmidt ℝ (gsCol A) i
        = ⟪(WithLp.toLp 2 (qCol A i) : EuclideanSpace ℝ (Fin m)), gsCol A k⟫_ℝ
            • (WithLp.toLp 2 (qCol A i) : EuclideanSpace ℝ (Fin m)) := by
    intro i hi
    have hik : i < k := Finset.mem_Iio.mp hi
    rw [proj_normalize (gramSchmidt ℝ (gsCol A) i) (gsCol A k),
      ← gramSchmidtNormed_eq_inv_norm_smul, ih i hik]
  rw [Finset.sum_congr rfl hproj]
  ext r
  rw [PiLp.sub_apply]
  show gsCol A k r - _ = gsV A (qsPrefix A k) k r
  rw [gsV_eq, gsCol_apply]
  congr 1
  -- The Euclidean `Iio` sum, applied at `r`, equals the executable list projection sum.
  rw [WithLp.ofLp_sum, Finset.sum_apply]
  rw [show ∑ i ∈ Finset.Iio k,
        (WithLp.ofLp (⟪(WithLp.toLp 2 (qCol A i) : EuclideanSpace ℝ (Fin m)), gsCol A k⟫_ℝ
          • (WithLp.toLp 2 (qCol A i) : EuclideanSpace ℝ (Fin m)))) r
      = ∑ i ∈ Finset.Iio k, Factorization.dotFn (qCol A i) (gsA A k) * qCol A i r from by
        apply Finset.sum_congr rfl
        intro i _
        rw [show WithLp.ofLp (⟪(WithLp.toLp 2 (qCol A i) : EuclideanSpace ℝ (Fin m)), gsCol A k⟫_ℝ
              • (WithLp.toLp 2 (qCol A i) : EuclideanSpace ℝ (Fin m))) r
            = ⟪(WithLp.toLp 2 (qCol A i) : EuclideanSpace ℝ (Fin m)), gsCol A k⟫_ℝ • qCol A i r
            from rfl, smul_eq_mul, gsCol, ← dotFn_eq_inner]]
  rw [sum_Iio_eq_sum_ite, qsPrefix_eq_map A k hk, List.map_map, take_map_sum_eq]
  rfl

/-- If every diagonal entry of `R` is positive, the `Q` column at each index `k` is Mathlib's
normalized Gram–Schmidt vector `gramSchmidtNormed ℝ (gsCol A) k`. -/
theorem toLp_qCol_eq_gramSchmidtNormed (A : Fin m → Fin n → ℝ)
    (hrank : ∀ j : Fin n, 0 < rMat A j j) :
    ∀ k : Fin n,
      (WithLp.toLp 2 (qCol A k) : EuclideanSpace ℝ (Fin m)) = gramSchmidtNormed ℝ (gsCol A) k := by
  have main : ∀ N : Nat, ∀ k : Fin n, k.val = N →
      (WithLp.toLp 2 (qCol A k) : EuclideanSpace ℝ (Fin m)) = gramSchmidtNormed ℝ (gsCol A) k := by
    intro N
    induction N using Nat.strong_induction_on with
    | _ N ih =>
      intro k hk
      have IH : ∀ i : Fin n, i.val < k.val →
          (WithLp.toLp 2 (qCol A i) : EuclideanSpace ℝ (Fin m)) = gramSchmidtNormed ℝ (gsCol A) i :=
        fun i hi => ih i.val (hk ▸ hi) i rfl
      have hgsV := gramSchmidt_eq_toLp_gsV A k (hrank k) IH
      rw [gramSchmidtNormed_eq_inv_norm_smul, hgsV]
      ext r
      rw [PiLp.smul_apply, PiLp.toLp_apply, PiLp.toLp_apply, smul_eq_mul]
      show qCol A k r = _
      rw [show qCol A k r = qMat A r k from rfl, qMat_eq A r k (hrank k), qStep_apply,
        ← normFn_eq_norm]
      show gsV A (qsPrefix A k) k r / gsRjj A (qsPrefix A k) k
        = (Factorization.normFn (gsV A (qsPrefix A k) k))⁻¹ * gsV A (qsPrefix A k) k r
      rw [show Factorization.normFn (gsV A (qsPrefix A k) k) = gsRjj A (qsPrefix A k) k from rfl,
        div_eq_mul_inv, mul_comm]
  exact fun k => main k.val k rfl

/-! ## Orthonormality `Qᵀ Q = 1` -/

/-- If every diagonal entry of `R` is positive, each normalized Gram–Schmidt vector of the columns
of `A` is nonzero. -/
theorem gramSchmidtNormed_gsCol_ne_zero (A : Fin m → Fin n → ℝ)
    (hrank : ∀ j : Fin n, 0 < rMat A j j) (j : Fin n) :
    gramSchmidtNormed ℝ (gsCol A) j ≠ 0 := by
  have hpos : 0 < ‖gramSchmidt ℝ (gsCol A) j‖ := by
    have h := (qsPrefix_pivot A j (hrank j)).2.2
    rwa [gramSchmidt_eq_toLp_gsV A j (hrank j)
      (fun i _ => toLp_qCol_eq_gramSchmidtNormed A hrank i), ← normFn_eq_norm]
  rw [gramSchmidtNormed_eq_inv_norm_smul]
  exact smul_ne_zero (inv_ne_zero (ne_of_gt hpos)) (norm_pos_iff.mp hpos)

/-- Under positive `R` pivots, the specification `Q` columns satisfy
`qₐ · q_b = δₐᵦ`. -/
theorem qCol_orthonormal (A : Fin m → Fin n → ℝ) (hrank : ∀ j : Fin n, 0 < rMat A j j)
    (a b : Fin n) :
    Factorization.dotFn (qCol A a) (qCol A b) = if a = b then 1 else 0 := by
  rw [dotFn_eq_inner]
  show ⟪(WithLp.toLp 2 (qCol A a) : EuclideanSpace ℝ (Fin m)), WithLp.toLp 2 (qCol A b)⟫_ℝ = _
  rw [toLp_qCol_eq_gramSchmidtNormed A hrank a, toLp_qCol_eq_gramSchmidtNormed A hrank b]
  have horth := orthonormal_iff_ite.mp (gramSchmidtNormed_orthonormal' (gsCol A))
    ⟨a, gramSchmidtNormed_gsCol_ne_zero A hrank a⟩ ⟨b, gramSchmidtNormed_gsCol_ne_zero A hrank b⟩
  rw [horth]
  simp only [Subtype.mk.injEq]

/-- `Qᵀ Q = 1` for the specification Gram–Schmidt `Q` factor under positive pivots. -/
theorem qMat_transpose_mul_self (A : Fin m → Fin n → ℝ) (hrank : ∀ j : Fin n, 0 < rMat A j j) :
    (Matrix.of (fun i k => qMat A i k))ᵀ * Matrix.of (fun i k => qMat A i k) = 1 := by
  ext a b
  rw [Matrix.mul_apply]
  simp only [Matrix.transpose_apply, Matrix.of_apply, Matrix.one_apply]
  rw [show (∑ i, qMat A i a * qMat A i b) = Factorization.dotFn (qCol A a) (qCol A b) from by
        rw [dotFn_eq_sum]; rfl,
    qCol_orthonormal A hrank a b]

/-- For `A` with positive specification `R`-pivots, the
Gram–Schmidt factors satisfy `Factorization.IsQR`: `Qᵀ Q = 1`, `R` upper-triangular,
and `A = Q · R`. -/
theorem isQR_of_pos (A : Fin m → Fin n → ℝ) (hrank : ∀ j : Fin n, 0 < rMat A j j) :
    Factorization.IsQR (Matrix.of A) (Matrix.of (fun i k => qMat A i k))
      (Matrix.of (fun k j => rMat A k j)) := by
  refine ⟨qMat_transpose_mul_self A hrank, ?_, qr_mul_eq A hrank⟩
  intro i j hji
  show rMat A i j = 0
  exact rMat_upper_triangular A (Fin.lt_def.mp hji)

/-- For a tensor `A` whose `qr` `R` factor has positive diagonal pivots, the columns of the
`Q` factor `(qr A).1` are orthonormal: `Σ_i Q[i,a]·Q[i,b] = δₐᵦ`. -/
theorem qrQ_orthonormal (A : Tensor ℝ [m, n])
    (hrank : ∀ j : Fin n, 0 < Tensor.get2 (Factorization.qr A).2 j j) (a b : Fin n) :
    (∑ i, Tensor.get2 (Factorization.qr A).1 i a * Tensor.get2 (Factorization.qr A).1 i b)
      = if a = b then 1 else 0 := by
  have hQ : ∀ x y, Tensor.get2 (Factorization.qr A).1 x y = qMat (Tensor.get2 A) x y := by
    intro x y
    simp [Factorization.qr, Tensor.get2_matrix, qMat]
  have hR : ∀ x y, Tensor.get2 (Factorization.qr A).2 x y = rMat (Tensor.get2 A) x y := by
    intro x y
    simp [Factorization.qr, Tensor.get2_matrix, rMat]
  simp only [hQ]
  rw [show (∑ i, qMat (Tensor.get2 A) i a * qMat (Tensor.get2 A) i b)
        = Factorization.dotFn (qCol (Tensor.get2 A) a)
          (qCol (Tensor.get2 A) b) from by
        rw [dotFn_eq_sum]; rfl]
  exact qCol_orthonormal (Tensor.get2 A) (fun j => by rw [← hR]; exact hrank j) a b

end QR

end TorchLean.Factorization
