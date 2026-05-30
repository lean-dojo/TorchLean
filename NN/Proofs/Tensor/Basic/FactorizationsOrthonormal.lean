/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Tensor.Basic.FactorizationsReconstruction
public import Mathlib.Analysis.InnerProductSpace.GramSchmidtOrtho
public import Mathlib.Analysis.InnerProductSpace.PiL2

/-!
# Orthonormality of the executable Gram–Schmidt `Q` factor (`Qᵀ Q = 1`)

This file closes the one finite-fold property left open by
[`NN.Proofs.Tensor.Basic.FactorizationsReconstruction`](FactorizationsReconstruction.lean): the
orthonormality of the `Q` factor produced by the executable modified Gram–Schmidt `gramSchmidtFn`.

The strategy is to **unify the executable variant with Mathlib's `gramSchmidt`** rather than re-derive
the orthogonality induction by hand. Reading the columns of `A` as vectors of
`EuclideanSpace ℝ (Fin m)`, the `j`-th executable `Q` column equals Mathlib's `gramSchmidtNormed ℝ`
of the column map (`Qcol_bridge`), so the orthonormality follows from Mathlib's
`gramSchmidtNormed_orthonormal'`.

## Main results

* `Qcol_bridge`: `WithLp.toLp 2 (Qcol A k) = gramSchmidtNormed ℝ (gsCol A) k` — the executable `Q`
  column is Mathlib's normalized Gram–Schmidt vector, proved by strong induction on `k`.
* `Q_orthonormal`: `dotFn (Qcol A a) (Qcol A b) = if a = b then 1 else 0` under positive `R` pivots.
* `QT_mul_Q_eq_one` and `isQR_of_pos`: the matrix-level `Qᵀ Q = 1` and the full
  `Spec.Factorization.IsQR` predicate for the executable factors (combining with the reconstruction
  `A = Q · R` and `R` upper-triangular from the companion file).
* `qrSpec_orthonormal`: the tensor-level corollary.

## Method

The bridge rests on three connectors over `ℝ`: `dotFn = ⟪·,·⟫` and `normFn = ‖·‖` on
`EuclideanSpace ℝ (Fin m)`, and the projection identity `proj_normalize` showing the un-normalized
Gram–Schmidt projection term equals the normalized one. The strong induction feeds the partial
identification of the earlier `Q` columns into `gramSchmidt_def''`, term by term.
-/

@[expose] public section

namespace Spec.Factorization.Reconstruction

open Matrix
open scoped BigOperators RealInnerProductSpace
open InnerProductSpace

/-! ## Connectors between the executable scalar ops and the Euclidean inner product -/

/-- `dotFn` as a `Finset` sum. -/
theorem dotFn_eq_sum {p : Nat} (u v : Fin p → ℝ) : Spec.dotFn u v = ∑ i, u i * v i := by
  unfold Spec.dotFn
  rw [foldl_addf_eq_sum (fun i => u i * v i) (List.finRange p) 0, zero_add,
    ← finsum_eq_finRange_sum (fun i => u i * v i)]

/-- The executable dot product is the Euclidean inner product over `ℝ`. -/
theorem dotFn_eq_inner {p : Nat} (u v : Fin p → ℝ) :
    Spec.dotFn u v
      = ⟪(WithLp.toLp 2 u : EuclideanSpace ℝ (Fin p)), WithLp.toLp 2 v⟫_ℝ := by
  rw [dotFn_eq_sum, PiLp.inner_apply]
  apply Finset.sum_congr rfl
  intro i _
  rw [RCLike.inner_apply', PiLp.toLp_apply, PiLp.toLp_apply]
  simp

/-- The executable Euclidean norm is the `EuclideanSpace` norm over `ℝ`. -/
theorem normFn_eq_norm {p : Nat} (v : Fin p → ℝ) :
    Spec.normFn v = ‖(WithLp.toLp 2 v : EuclideanSpace ℝ (Fin p))‖ := by
  rw [Spec.normFn, mfsqrt_eq, EuclideanSpace.norm_eq]
  congr 1
  rw [dotFn_eq_sum]
  apply Finset.sum_congr rfl
  intro i _
  rw [PiLp.toLp_apply, Real.norm_eq_abs, sq_abs, sq]

/-- The Gram–Schmidt projection term, with the normalized vector pulled out. Holds with no
non-degeneracy hypothesis (both sides vanish when `gramSchmidt = 0`). -/
theorem proj_normalize {F : Type*} [NormedAddCommGroup F] [InnerProductSpace ℝ F] (w x : F) :
    (⟪w, x⟫_ℝ / ‖w‖ ^ 2) • w = ⟪‖w‖⁻¹ • w, x⟫_ℝ • (‖w‖⁻¹ • w) := by
  rw [real_inner_smul_left, smul_smul]
  congr 1
  rw [div_eq_mul_inv, ← inv_pow, sq]
  ring

/-- `gramSchmidtNormed` over `ℝ`, with the scalar coercion removed. -/
theorem gn_eq {n : Nat} {F : Type*} [NormedAddCommGroup F] [InnerProductSpace ℝ F]
    (f : Fin n → F) (i : Fin n) :
    gramSchmidtNormed ℝ f i = ‖gramSchmidt ℝ f i‖⁻¹ • gramSchmidt ℝ f i := by
  rw [gramSchmidtNormed]
  norm_num

/-- A masked full sum equals the sum over `Iio`. -/
theorem sum_Iio_eq_mask {n : Nat} (k : Fin n) (h : Fin n → ℝ) :
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

/-- **Orthogonalized-vector bridge.** Given that the earlier `Q` columns coincide with Mathlib's
normalized Gram–Schmidt vectors, the executable orthogonalized vector `v` at index `k` equals
Mathlib's (un-normalized) `gramSchmidt` vector. -/
theorem gsV_bridge (A : Fin m → Fin n → ℝ) (k : Fin n)
    (ih : ∀ i : Fin n, i.val < k.val →
        (WithLp.toLp 2 (Qcol A i) : EuclideanSpace ℝ (Fin m)) = gramSchmidtNormed ℝ (gsCol A) i) :
    gramSchmidt ℝ (gsCol A) k = WithLp.toLp 2 (gsV A (qsPrefix A k) k) := by
  -- Rewrite Mathlib's vector via the explicit recurrence.
  rw [show gramSchmidt ℝ (gsCol A) k
        = gsCol A k - ∑ i ∈ Finset.Iio k,
            (⟪gramSchmidt ℝ (gsCol A) i, gsCol A k⟫_ℝ / ‖gramSchmidt ℝ (gsCol A) i‖ ^ 2)
              • gramSchmidt ℝ (gsCol A) i
      from eq_sub_of_add_eq (gramSchmidt_def'' ℝ (gsCol A) k).symm]
  -- Replace each projection term by the normalized form, then by the executable `Q` column.
  have hproj : ∀ i ∈ Finset.Iio k,
      (⟪gramSchmidt ℝ (gsCol A) i, gsCol A k⟫_ℝ / ‖gramSchmidt ℝ (gsCol A) i‖ ^ 2)
          • gramSchmidt ℝ (gsCol A) i
        = ⟪(WithLp.toLp 2 (Qcol A i) : EuclideanSpace ℝ (Fin m)), gsCol A k⟫_ℝ
            • (WithLp.toLp 2 (Qcol A i) : EuclideanSpace ℝ (Fin m)) := by
    intro i hi
    have hik : i < k := Finset.mem_Iio.mp hi
    rw [proj_normalize (gramSchmidt ℝ (gsCol A) i) (gsCol A k), ← gn_eq, ih i hik]
  rw [Finset.sum_congr rfl hproj]
  -- Compare entrywise.
  ext r
  rw [PiLp.sub_apply]
  show gsCol A k r - _ = gsV A (qsPrefix A k) k r
  rw [gsV_eq, gsCol_apply]
  congr 1
  -- The Euclidean `Iio` sum, applied at `r`, equals the executable list projection sum.
  rw [WithLp.ofLp_sum, Finset.sum_apply]
  rw [show ∑ i ∈ Finset.Iio k,
        (WithLp.ofLp (⟪(WithLp.toLp 2 (Qcol A i) : EuclideanSpace ℝ (Fin m)), gsCol A k⟫_ℝ
          • (WithLp.toLp 2 (Qcol A i) : EuclideanSpace ℝ (Fin m)))) r
      = ∑ i ∈ Finset.Iio k, Spec.dotFn (Qcol A i) (gsA A k) * Qcol A i r from by
        apply Finset.sum_congr rfl
        intro i _
        rw [show WithLp.ofLp (⟪(WithLp.toLp 2 (Qcol A i) : EuclideanSpace ℝ (Fin m)), gsCol A k⟫_ℝ
              • (WithLp.toLp 2 (Qcol A i) : EuclideanSpace ℝ (Fin m))) r
            = ⟪(WithLp.toLp 2 (Qcol A i) : EuclideanSpace ℝ (Fin m)), gsCol A k⟫_ℝ • Qcol A i r
            from rfl, smul_eq_mul, gsCol, ← dotFn_eq_inner]]
  rw [sum_Iio_eq_mask, qsPrefix_eq_map, List.map_map, take_map_sum_eq]
  rfl

/-- **Normalized-column bridge.** The executable `Q` column at index `k` equals Mathlib's
`gramSchmidtNormed`. Proved by strong induction on `k`, under positive `R` pivots (full column rank). -/
theorem Qcol_bridge (A : Fin m → Fin n → ℝ) (hrank : ∀ j : Fin n, 0 < Rmat A j j) :
    ∀ k : Fin n,
      (WithLp.toLp 2 (Qcol A k) : EuclideanSpace ℝ (Fin m)) = gramSchmidtNormed ℝ (gsCol A) k := by
  have main : ∀ N : Nat, ∀ k : Fin n, k.val = N →
      (WithLp.toLp 2 (Qcol A k) : EuclideanSpace ℝ (Fin m)) = gramSchmidtNormed ℝ (gsCol A) k := by
    intro N
    induction N using Nat.strong_induction_on with
    | _ N ih =>
      intro k hk
      have IH : ∀ i : Fin n, i.val < k.val →
          (WithLp.toLp 2 (Qcol A i) : EuclideanSpace ℝ (Fin m)) = gramSchmidtNormed ℝ (gsCol A) i :=
        fun i hi => ih i.val (hk ▸ hi) i rfl
      have hρpos : 0 < gsRjj A (qsPrefix A k) k := by
        have h := hrank k; rwa [Rmat_eq, rStep_diag] at h
      have hgsV := gsV_bridge A k IH
      rw [gn_eq, hgsV]
      ext r
      rw [PiLp.smul_apply, PiLp.toLp_apply, PiLp.toLp_apply, smul_eq_mul]
      show Qcol A k r = _
      rw [show Qcol A k r = Qmat A r k from rfl, Qmat_eq, qStep_pos A (qsPrefix A k) k hρpos,
        ← normFn_eq_norm]
      show gsV A (qsPrefix A k) k r / gsRjj A (qsPrefix A k) k
        = (Spec.normFn (gsV A (qsPrefix A k) k))⁻¹ * gsV A (qsPrefix A k) k r
      rw [show Spec.normFn (gsV A (qsPrefix A k) k) = gsRjj A (qsPrefix A k) k from rfl,
        div_eq_mul_inv, mul_comm]
  exact fun k => main k.val k rfl

/-! ## Orthonormality `Qᵀ Q = 1` -/

/-- Each normalized Gram–Schmidt vector is non-zero (the pivot is positive). -/
theorem gn_ne_zero (A : Fin m → Fin n → ℝ) (hrank : ∀ j : Fin n, 0 < Rmat A j j) (j : Fin n) :
    gramSchmidtNormed ℝ (gsCol A) j ≠ 0 := by
  have hpos : 0 < ‖gramSchmidt ℝ (gsCol A) j‖ := by
    have h := hrank j
    rw [Rmat_eq, rStep_diag] at h
    rwa [gsV_bridge A j (fun i _ => Qcol_bridge A hrank i), ← normFn_eq_norm]
  rw [gn_eq]
  exact smul_ne_zero (inv_ne_zero (ne_of_gt hpos)) (norm_pos_iff.mp hpos)

/-- **Orthonormality of the executable `Q` columns.** Under positive `R` pivots,
`qₐ · q_b = δₐᵦ`. -/
theorem Q_orthonormal (A : Fin m → Fin n → ℝ) (hrank : ∀ j : Fin n, 0 < Rmat A j j) (a b : Fin n) :
    Spec.dotFn (Qcol A a) (Qcol A b) = if a = b then 1 else 0 := by
  rw [dotFn_eq_inner]
  show ⟪(WithLp.toLp 2 (Qcol A a) : EuclideanSpace ℝ (Fin m)), WithLp.toLp 2 (Qcol A b)⟫_ℝ = _
  rw [Qcol_bridge A hrank a, Qcol_bridge A hrank b]
  have horth := orthonormal_iff_ite.mp (gramSchmidtNormed_orthonormal' (gsCol A))
    ⟨a, gn_ne_zero A hrank a⟩ ⟨b, gn_ne_zero A hrank b⟩
  rw [horth]
  simp only [Subtype.mk.injEq]

/-- **Matrix-level orthonormality.** `Qᵀ Q = 1` for the executable Gram–Schmidt `Q` factor. -/
theorem QT_mul_Q_eq_one (A : Fin m → Fin n → ℝ) (hrank : ∀ j : Fin n, 0 < Rmat A j j) :
    (Matrix.of (fun i k => Qmat A i k))ᵀ * Matrix.of (fun i k => Qmat A i k) = 1 := by
  ext a b
  rw [Matrix.mul_apply]
  simp only [Matrix.transpose_apply, Matrix.of_apply, Matrix.one_apply]
  rw [show (∑ i, Qmat A i a * Qmat A i b) = Spec.dotFn (Qcol A a) (Qcol A b) from by
        rw [dotFn_eq_sum]; rfl,
    Q_orthonormal A hrank a b]

/-- **Full QR specification.** For `A` with positive executable `R`-pivots (full column rank), the
executable Gram–Schmidt factors satisfy `Spec.Factorization.IsQR`: `Qᵀ Q = 1`, `R` upper-triangular,
and `A = Q · R`. -/
theorem isQR_of_pos (A : Fin m → Fin n → ℝ) (hrank : ∀ j : Fin n, 0 < Rmat A j j) :
    Spec.Factorization.IsQR (Matrix.of A) (Matrix.of (fun i k => Qmat A i k))
      (Matrix.of (fun k j => Rmat A k j)) := by
  refine ⟨QT_mul_Q_eq_one A hrank, ?_, qr_mul_eq A hrank⟩
  intro i j hji
  show Rmat A i j = 0
  exact Rmat_upper_triangular A (Fin.lt_def.mp hji)

/-- **Tensor-level orthonormality.** For a tensor `A` with positive `qrRSpec` pivots, the `Q` factor
`qrQSpec A` has orthonormal columns: `Σ_i Q[i,a]·Q[i,b] = δₐᵦ`. -/
theorem qrSpec_orthonormal (A : Spec.Tensor ℝ (.dim m (.dim n .scalar)))
    (hrank : ∀ j : Fin n, 0 < Spec.get2 (Spec.qrRSpec A) j j) (a b : Fin n) :
    (∑ i, Spec.get2 (Spec.qrQSpec A) i a * Spec.get2 (Spec.qrQSpec A) i b)
      = if a = b then 1 else 0 := by
  have hQ : ∀ x y, Spec.get2 (Spec.qrQSpec A) x y = Qmat (Spec.toMatFn A) x y := fun _ _ => rfl
  have hR : ∀ x y, Spec.get2 (Spec.qrRSpec A) x y = Rmat (Spec.toMatFn A) x y := fun _ _ => rfl
  simp only [hQ]
  rw [show (∑ i, Qmat (Spec.toMatFn A) i a * Qmat (Spec.toMatFn A) i b)
        = Spec.dotFn (Qcol (Spec.toMatFn A) a) (Qcol (Spec.toMatFn A) b) from by
        rw [dotFn_eq_sum]; rfl]
  exact Q_orthonormal (Spec.toMatFn A) (fun j => by rw [← hR]; exact hrank j) a b

end QR

end Spec.Factorization.Reconstruction
