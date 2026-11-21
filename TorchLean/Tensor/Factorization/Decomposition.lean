/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Analysis.Complex.Polynomial.Basic
public import Mathlib.Analysis.InnerProductSpace.GramSchmidtOrtho
public import Mathlib.Analysis.Matrix.Spectrum
public import Mathlib.LinearAlgebra.Matrix.IsDiag
public import Mathlib.Data.Fin.Tuple.Sort
public import Mathlib.LinearAlgebra.Matrix.Permutation
public import Mathlib.LinearAlgebra.Matrix.Hessenberg.Defs

/-!
# Matrix decompositions as relations

Each predicate here says when matrices are a valid output of one of the linear algebra
decompositions of PyTorch 2.13 and JAX 0.11.1, over `ℝ` or `ℂ` (`RCLike 𝕜`):

| operation | predicate |
| --- | --- |
| `qr` (`torch.linalg.qr`, `jax.lax.linalg.qr`) | `IsQR`, `IsPivotedQR` |
| `geqrf`, `geqp3` | `IsGeqrf`, `IsGeqp3` |
| `householder_product`, `ormqr` | `householderProduct`, `householderOp` |
| `lu`, `lu_pivots_to_permutation` | `IsLU`, `pivotPerm` |
| `eigh`, `eig` | `IsEigh`, `IsEig` |
| `svd` | `IsSVD` |
| `schur` | `IsSchur`, `IsRealSchur` |
| `hessenberg`, `tridiagonal` | `IsGehrd`, `IsSytrd` (through `IsHessenberg`, `IsTridiagonal`) |
| `tridiagonal_solve` | `IsTridiagonalSolve` |

They are relations, not algorithms: an output satisfies one whichever LAPACK routine computed
it, and where the documentation leaves a choice (the signs of a QR factorization, the phases of
eigenvectors and singular vectors, the permutation of an LU factorization) the predicate leaves
it too. They are mathematical relations, not the exact sets of outputs of a backend: `IsEig`
also leaves free the phase of each eigenvector, which LAPACK's `geev` fixes, so it holds of
outputs that JAX never returns. We wrote each one by hand from the docstrings cited at it.
Only the products of Householder reflectors and the pivot permutation are functions, as their
documentation defines them by a formula.

## Main results

* `conjTranspose_householderQ_mul_self`: a product of reflectors `1 - τ v vᴴ` is unitary when
  each `τ` satisfies `star τ + τ = star τ τ (vᴴ v)`, and `conjTranspose_householderProduct_mul_self`
  gives its first columns orthonormal.
* `exists_isEigh`: every Hermitian matrix has an eigendecomposition with ascending eigenvalues
  (Mathlib's spectral theorem, its eigenvalues sorted), and `IsEigh.eigenvalues_eq`: the
  eigenvalues are unique, being the ascending roots of the characteristic polynomial
  (`IsEigh.sort_roots`, through `sort_roots_prod`).
* `exists_isEig`: every complex matrix has an output of `eig`, its eigenvalues the roots of the
  characteristic polynomial and its eigenvectors of unit norm, and invertible when the matrix is
  diagonalizable (the columns of a diagonalization, scaled to unit norm).
* `exists_isQR`: every matrix has a complete QR factorization, rank-deficient ones included (`Q`
  from Mathlib's Gram-Schmidt basis, `InnerProductSpace.gramSchmidtOrthonormalBasis`, and
  `R = Qᴴ A`), and so, by `IsQR.reduce`, a reduced one.
* `exists_isGeqrf`: every matrix has a `geqrf` factorization (by induction on the number of
  columns, with a Householder reflector from `exists_householder` and `householderQ_cons`),
  and so, applied to the columns of `A` permuted by `P`, a `geqp3` factorization for every `P`.
* `exists_isPivotedQR`: every matrix has a column-pivoted QR factorization, with the moduli of
  `R`'s diagonal non-increasing (by induction on the number of columns, moving first a column of
  largest norm, with a Householder reflector from `exists_householder`; `sum_norm_sq_mulVec`:
  a matrix with orthonormal columns keeps the norm, which bounds the rest of the diagonal).
* `exists_isSchur`: every complex matrix has a Schur decomposition (by induction on its size,
  from `exists_isEig`, `exists_unitary_extension` and `unitary_blockOne`).
* `exists_isGehrd`: every matrix, real or complex, has a `hessenberg` output, with a real first
  subdiagonal (by induction on its size, with a Householder reflector from `exists_householder`,
  `householderQ_cons` and `householderQ_succ`), so a Hessenberg decomposition
  (`exists_isHessenberg`), and `exists_isSytrd`: so every Hermitian matrix has a `tridiagonal`
  output with `lower=True`, a Hermitian Hessenberg matrix being tridiagonal.
* `exists_isLU`: every matrix has an LU factorization with row pivoting, by Gaussian elimination
  (induction on the Schur complement of a nonzero pivot).
* `exists_isSVD`: every matrix has a full singular value decomposition (`exists_isSVD_of_le`,
  from an eigendecomposition of `-(Aᴴ A)` and `exists_unitary_extension`), and so, by
  `IsSVD.reduce`, a reduced one.
* `IsEigh.isEig`, `IsEigh.isSchur`, `IsSchur.isHessenberg`, `IsEigh.isTridiagonal`: an
  eigendecomposition of a Hermitian matrix is an output of `eig`, `schur`, `hessenberg` and
  `tridiagonal`'s relations.
* `IsLU.eq_mul`: JAX's `A[perm] = L U` is PyTorch's `A = P L U` for `P` the permutation matrix
  of `perm⁻¹`.
* Uniqueness up to the freedom PyTorch's warnings name: `IsQR.isQR_iff`, with the
  first `k` columns of `A` linearly independent a QR factorization is unique up to the signs of
  `R`'s diagonal (`exists_signs` for the square case); `IsLU.unique`, with the permutation fixed
  and the first `k` columns independent `L` and `U` are unique (`eq_of_mul_eq_mul`);
  `IsEigh.isEigh_iff`, with distinct eigenvalues the eigenvectors are unique up to a phase each
  (`eq_mul_diagonal`); `IsEig.isEig_iff`, with distinct eigenvalues an output of `eig` is unique
  up to the order of the eigenvalues and a phase for each eigenvector (`IsEig.isUnit_det`,
  `IsEig.submatrix`, `IsEig.mul_diagonal`);
  `IsSVD.singularValues_eq`, the singular values are unique (through `IsSVD.reduce` to the
  reduced shapes and `IsSVD.charpoly`, the characteristic polynomial of `Aᴴ A`), and
  `IsSVD.isSVD_iff`, with distinct positive singular values the reduced singular vectors are
  unique up to a phase for each pair.
* `isTridiagonalSolve_iff`: for an invertible tridiagonal matrix the solution is unique.
* `upperBand_zero_iff`, `upperBand_one_iff`: on square matrices `UpperBand 0` and `UpperBand 1`
  are Mathlib's `Matrix.IsUpperTriangular` and `Matrix.IsUpperHessenberg`, which `IsSchur`,
  `IsRealSchur` and `IsHessenberg` use.
* Instances: `isQR_one`, `isGeqrf_self`, `isGeqp3_one`, `isPivotedQR_one`, `isLU_one`,
  `pivotPerm_castLE`, `isEig_diagonal`, `isSVD_rectDiagonal`, `isRealSchur_two`,
  `isGehrd_self`, `isSytrd_symTridiagonal`.

The factorizations of TorchLean's executable `Factorization.qr` satisfy `IsQR`
(`TorchLean.Tensor.Factorization.Orthonormal`, `isQR_of_pos`), and every real matrix has a real
Schur decomposition (`TorchLean.Tensor.Factorization.RealSchur`, `exists_isRealSchur`). No
operator of TorchLean's operator table is specified by these relations yet: they are the
mathematical statements its decompositions are to be checked against.
-/

@[expose] public section

namespace TorchLean.Factorization

open Matrix

variable {α 𝕜 : Type*} [RCLike 𝕜] {m n k p q r : ℕ}

/-! ## Shapes -/

/-- `R` vanishes below its `b`-th subdiagonal: `R i j = 0` whenever `j + b < i`. `UpperBand 0 R`
says that `R`, of any shape, is upper triangular, and `UpperBand 1 R` that it is upper
Hessenberg; on square matrices these are Mathlib's `IsUpperTriangular` and `IsUpperHessenberg`
(`upperBand_zero_iff`, `upperBand_one_iff`). -/
def UpperBand [Zero α] (b : ℕ) (R : Matrix (Fin k) (Fin n) α) : Prop :=
  ∀ i j, j.val + b < i.val → R i j = 0

/-- The entries of `a` on and above its `b`-th subdiagonal, the others zero. -/
def upperPart [Zero α] (b : ℕ) (a : Matrix (Fin k) (Fin n) α) : Matrix (Fin k) (Fin n) α :=
  of fun i j => if i.val ≤ j.val + b then a i j else 0

/-- The diagonal `R i i`, `i < min k n`, of a matrix of any shape. -/
def rectDiag (R : Matrix (Fin k) (Fin n) α) (i : Fin (min k n)) : α :=
  R (Fin.castLE (min_le_left k n) i) (Fin.castLE (min_le_right k n) i)

/-- The `p × q` matrix with `d i` at `(i, i)` for `i < s` and zeros elsewhere. -/
def rectDiagonal [Zero α] {s : ℕ} (d : Fin s → α) : Matrix (Fin p) (Fin q) α :=
  of fun i j => if h : i.val = j.val ∧ i.val < s then d ⟨i.val, h.2⟩ else 0

/-- `upperPart b a` vanishes below its `b`-th subdiagonal. -/
theorem upperBand_upperPart [Zero α] (b : ℕ) (a : Matrix (Fin k) (Fin n) α) :
    UpperBand b (upperPart b a) :=
  fun i j hij => by
    rw [upperPart, of_apply, ite_eq_right_iff]
    exact fun h => absurd h (by omega)

/-- A matrix that vanishes below its `b`-th subdiagonal is its own `upperPart b`. -/
theorem upperPart_eq_self [Zero α] {b : ℕ} {a : Matrix (Fin k) (Fin n) α} (h : UpperBand b a) :
    upperPart b a = a := by
  ext i j
  simp only [upperPart, of_apply]
  split_ifs with hij
  · rfl
  · exact (h i j (by omega)).symm

/-- On square matrices of size `s`, `rectDiagonal d` is Mathlib's `diagonal d`. -/
theorem rectDiagonal_eq_diagonal [Zero α] {s : ℕ} (d : Fin s → α) :
    (rectDiagonal d : Matrix (Fin s) (Fin s) α) = diagonal d := by
  ext i j
  by_cases hij : i = j
  · subst hij
    simp [rectDiagonal]
  · simp only [rectDiagonal, of_apply, diagonal_apply_ne _ hij]
    exact dite_eq_right fun e => hij (Fin.ext e.1)

/-- `rectDiagonal d` is `diag(d)` embedded in the first `s` rows and columns. -/
theorem rectDiagonal_eq_mul [NonAssocSemiring α] {s : ℕ} (d : Fin s → α) (hp : s ≤ p)
    (hq : s ≤ q) :
    (rectDiagonal d : Matrix (Fin p) (Fin q) α) =
      (1 : Matrix (Fin p) (Fin p) α).submatrix id (Fin.castLE hp) * diagonal d *
        (1 : Matrix (Fin q) (Fin q) α).submatrix (Fin.castLE hq) id := by
  ext i j
  rw [Matrix.mul_apply]
  simp only [mul_diagonal, submatrix_apply, id_eq, Matrix.one_apply, rectDiagonal, of_apply]
  split_ifs with h
  · rw [Fintype.sum_eq_single ⟨i, h.2⟩ fun l hl => ?_]
    · simp [Fin.ext_iff, h.1]
    · have : i ≠ Fin.castLE hp l := fun e => hl (Fin.ext (congrArg Fin.val e).symm)
      simp [this]
  · refine (Finset.sum_eq_zero fun l _ => ?_).symm
    by_cases e1 : i = Fin.castLE hp l
    · have e2 : Fin.castLE hq l ≠ j := fun e2 =>
        h ⟨by rw [e1, ← e2]; rfl, by rw [e1]; exact l.isLt⟩
      simp [e2]
    · simp [e1]

/-- On square matrices, `UpperBand 0` is Mathlib's `IsUpperTriangular`. -/
theorem upperBand_zero_iff [Zero α] {R : Matrix (Fin n) (Fin n) α} :
    UpperBand 0 R ↔ R.IsUpperTriangular :=
  ⟨fun h i j hij => h i j (Fin.lt_def.mp hij), fun h _ _ hij => h (Fin.lt_def.mpr hij)⟩

/-- On square matrices, `UpperBand 1` is Mathlib's `IsUpperHessenberg`. -/
theorem upperBand_one_iff [Zero α] {R : Matrix (Fin n) (Fin n) α} :
    UpperBand 1 R ↔ R.IsUpperHessenberg :=
  isUpperHessenberg_fin_iff.symm

/-- A square factor `T` of `X T` is invertible when `X T` has linearly independent columns
(`mulVec` injective). -/
theorem isUnit_det_of_injective {X : Matrix (Fin m) (Fin k) 𝕜} {T : Matrix (Fin k) (Fin k) 𝕜}
    (h : Function.Injective (X * T).mulVec) : IsUnit T.det :=
  (isUnit_iff_isUnit_det T).mp <| mulVec_injective_iff_isUnit.mp <|
    Function.Injective.of_comp (f := X.mulVec) (by simpa [Function.comp_def, mulVec_mulVec] using h)

/-- `D` is upper triangular when `T = D T'` for an upper triangular `T` and an invertible upper
triangular `T'`. -/
theorem isUpperTriangular_of_eq_mul {T D T' : Matrix (Fin k) (Fin k) 𝕜}
    (hT : T.IsUpperTriangular) (hT' : T'.IsUpperTriangular) (hu : IsUnit T'.det)
    (h : T = D * T') : D.IsUpperTriangular := by
  have := invertibleOfIsUnitDet T' hu
  rw [← mul_nonsing_inv_cancel_right T' D hu, ← h]
  exact hT.mul (blockTriangular_inv_of_blockTriangular hT')

/-- A matrix both upper and lower triangular is diagonal. -/
theorem isDiag_of_triangular [Zero α] {D : Matrix (Fin k) (Fin k) α} (hu : D.IsUpperTriangular)
    (hl : D.IsLowerTriangular) : D.IsDiag := fun _ _ hij =>
  (lt_or_gt_of_ne hij).elim (fun h => hl (OrderDual.toDual_lt_toDual.mpr h)) fun h => hu h

/-! ## QR factorizations -/

/-- `(Q, R)` is a QR factorization of `A`: `Q` has orthonormal columns (`Qᴴ Q = 1`), `R` is upper
triangular with a real diagonal, and `A = Q R`.

We wrote it by hand from the docstrings of `torch.linalg.qr` in PyTorch 2.13
(torch/linalg/__init__.py), "`A = QR` ... and `R` is upper triangular with real diagonal (even in
the complex case)", and of `jax.lax.linalg.qr` in JAX 0.11.1 (jax/_src/lax/linalg.py), "`Q` is a
unitary (orthogonal) matrix, and `R` is an upper-triangular matrix". The modes are values of
`k`: `k = m` for PyTorch's `complete` and JAX's `full_matrices=True` (`Q` square, `R` of `A`'s
shape), `k = min m n` for `reduced` and `full_matrices=False`, and PyTorch's `r` returns the
reduced `R` alone. PyTorch warns that the decomposition "is only unique up to the sign of the
diagonal of `R`", and "only well-defined if the first `k = min(m, n)` columns" of `A` "are linearly
independent"; `IsQR` leaves those signs free. `exists_isQR` gives every `A`, of any rank, a
complete factorization, and `IsQR.reduce` a reduced one. -/
def IsQR (A : Matrix (Fin m) (Fin n) 𝕜) (Q : Matrix (Fin m) (Fin k) 𝕜)
    (R : Matrix (Fin k) (Fin n) 𝕜) : Prop :=
  Qᴴ * Q = 1 ∧ UpperBand 0 R ∧ (∀ i, IsSelfAdjoint (rectDiag R i)) ∧ A = Q * R

/-- A complete QR factorization restricts to a reduced one, on the first `min m n` columns of `Q`
and rows of `R`: the other rows of `R` are below its diagonal, so zero. -/
theorem IsQR.reduce {A : Matrix (Fin m) (Fin n) 𝕜} {Q : Matrix (Fin m) (Fin m) 𝕜}
    {R : Matrix (Fin m) (Fin n) 𝕜} (h : IsQR A Q R) :
    IsQR A (Q.submatrix id (Fin.castLE (min_le_left m n)))
      (R.submatrix (Fin.castLE (min_le_left m n)) id) := by
  refine ⟨?_, fun i j hij => h.2.1 (Fin.castLE _ i) j hij,
    fun i => h.2.2.1 (Fin.castLE (min_le_left _ n) i), ?_⟩
  · rw [conjTranspose_submatrix, ← submatrix_mul _ _ _ _ _ Function.bijective_id, h.1,
      submatrix_one _ (Fin.castLE_injective _)]
  · ext i j
    rw [h.2.2.2, Matrix.mul_apply, Matrix.mul_apply]
    refine (Fintype.sum_of_injective _ (Fin.castLE_injective (min_le_left m n)) _ _
      (fun r hr => ?_) fun t => rfl).symm
    have hr' : ¬ r.val < min m n := fun hl => hr ⟨⟨r, hl⟩, rfl⟩
    rw [Nat.lt_min] at hr'
    have := r.isLt
    have := j.isLt
    rw [h.2.1 r j (by omega), mul_zero]

/-- `(P, Q, R)` is a column-pivoted QR factorization of `A`: `(Q, R)` is a QR factorization of the
columns of `A` permuted by `P` (column `j` of the product is column `P j` of `A`), and the moduli
of the diagonal of `R` do not increase.

We wrote it by hand from the docstring of `jax.lax.linalg.qr` in JAX 0.11.1
(jax/_src/lax/linalg.py) with `pivoting=True`: "compute the column pivoted decomposition
`A[:, P] = Q @ R`, where `P` is chosen such that the diagonal of `R` is non-increasing", the
wording of `scipy.linalg.qr` in SciPy 1.17 (scipy/linalg/_decomp_qr.py). The diagonal's signs are
free, and we read "non-increasing" of its moduli, as SciPy's example for `pivoting=True` checks
it: "`q4, r4, p4 = linalg.qr(a, pivoting=True)`", "`d = np.abs(np.diag(r4))`",
"`np.all(d[1:] <= d[:-1])`" giving `True`. Read of the signed diagonal, JAX's docstring is
contradicted by JAX's own outputs: the pivoted `R` of `[[1, 2], [3, 4]]` has the diagonal
`[-4.472136, 0.4472136]` in binary32. `exists_isPivotedQR` gives every `A` one. -/
def IsPivotedQR (A : Matrix (Fin m) (Fin n) 𝕜) (P : Equiv.Perm (Fin n))
    (Q : Matrix (Fin m) (Fin k) 𝕜) (R : Matrix (Fin k) (Fin n) 𝕜) : Prop :=
  IsQR (A.submatrix id P) Q R ∧ Antitone fun i => ‖rectDiag R i‖

/-- An upper triangular `A` whose diagonal entries are self-adjoint is its own QR decomposition with
`Q = 1` and `R = A`. -/
theorem isQR_one {A : Matrix (Fin m) (Fin n) 𝕜} (hA : UpperBand 0 A)
    (hd : ∀ i, IsSelfAdjoint (rectDiag A i)) : IsQR A 1 A :=
  ⟨by rw [conjTranspose_one, one_mul], hA, hd, (Matrix.one_mul A).symm⟩

/-- An upper triangular `A` whose diagonal entries are self-adjoint and have non-increasing norms is
its own pivoted QR decomposition with `P = 1`, `Q = 1` and `R = A`. -/
theorem isPivotedQR_one {A : Matrix (Fin m) (Fin n) 𝕜} (hA : UpperBand 0 A)
    (hd : ∀ i, IsSelfAdjoint (rectDiag A i)) (hP : Antitone fun i => ‖rectDiag A i‖) :
    IsPivotedQR A 1 1 A :=
  ⟨by rw [Equiv.Perm.coe_one, submatrix_id_id]; exact isQR_one hA hd, hP⟩

/-- Two factorizations `Q T = Q' T'` with orthonormal columns and invertible upper triangular
square factors with real diagonals differ by signs: `Q' = Q S` for a diagonal `S` of `±1`. -/
theorem exists_signs {Q Q' : Matrix (Fin m) (Fin k) 𝕜} {T T' : Matrix (Fin k) (Fin k) 𝕜}
    (hQ : Qᴴ * Q = 1) (hQ' : Q'ᴴ * Q' = 1) (hT : T.IsUpperTriangular)
    (hT' : T'.IsUpperTriangular) (hu : IsUnit T.det) (hu' : IsUnit T'.det)
    (hd : ∀ i, star (T i i) = T i i) (hd' : ∀ i, star (T' i i) = T' i i) (h : Q * T = Q' * T') :
    ∃ s : Fin k → 𝕜, (∀ i, s i = 1 ∨ s i = -1) ∧ Q' = Q * diagonal s := by
  have hTD : T = Qᴴ * Q' * T' := by
    rw [Matrix.mul_assoc, ← h, ← Matrix.mul_assoc, hQ, Matrix.one_mul]
  have hT'D : T' = (Qᴴ * Q')ᴴ * T := by
    rw [conjTranspose_mul, conjTranspose_conjTranspose, Matrix.mul_assoc, h, ← Matrix.mul_assoc,
      hQ', Matrix.one_mul]
  have hD : (Qᴴ * Q').IsDiag := isDiag_of_triangular (isUpperTriangular_of_eq_mul hT hT' hu' hTD)
    fun i j hij => by
      simpa using isUpperTriangular_of_eq_mul hT' hT hu hT'D (OrderDual.toDual_lt_toDual.mp hij)
  obtain ⟨s, hs⟩ : ∃ s, Qᴴ * Q' = diagonal s := ⟨_, hD.diagonal_diag.symm⟩
  rw [hs] at hTD hT'D
  rw [diagonal_conjTranspose] at hT'D
  refine ⟨s, fun i => ?_, ?_⟩
  · have hT0 : T i i ≠ 0 := hT.det_ne_zero_iff.mp hu.ne_zero i
    have hT0' : T' i i ≠ 0 := hT'.det_ne_zero_iff.mp hu'.ne_zero i
    have h1 : T i i = s i * T' i i := by rw [hTD, diagonal_mul]
    have h2 : T' i i = star (s i) * T i i := by simp only [hT'D, diagonal_mul, Pi.star_apply]
    have h3 : star (s i) = s i := by
      have := congrArg star h1
      rw [star_mul', hd i, hd' i] at this
      exact mul_right_cancel₀ hT0' (this.symm.trans h1)
    have h2' : T' i i = s i * T i i := by rw [h2, h3]
    refine mul_self_eq_one_iff.mp (mul_right_cancel₀ hT0 ?_)
    calc s i * s i * T i i = s i * T' i i := by rw [mul_assoc, ← h2']
      _ = T i i := h1.symm
      _ = 1 * T i i := (one_mul _).symm
  · calc Q' = Q' * T' * T'⁻¹ := (mul_nonsing_inv_cancel_right T' Q' hu').symm
      _ = Q * diagonal s := by
        rw [← h, hTD, ← Matrix.mul_assoc, mul_nonsing_inv_cancel_right T' _ hu']

/-- With the first `k` columns of `A` linearly independent, a QR factorization `(Q, R)` of `A` is
unique up to the signs of the diagonal of `R`: `(Q', R')` is one exactly when `Q' = Q S` and
`R' = S R` for a diagonal `S` of `±1`. This is the convention of PyTorch's warning at
`torch.linalg.qr` (quoted at `IsQR`), proved for every `k ≤ n`: the reduced mode `k = min m n`,
and the complete mode when `m ≤ n`. In the complete mode with `m > n` the last `m - n` columns of
`Q` are any orthonormal completion, which `IsQR` leaves free too. -/
theorem IsQR.isQR_iff {A : Matrix (Fin m) (Fin n) 𝕜} {Q Q' : Matrix (Fin m) (Fin k) 𝕜}
    {R R' : Matrix (Fin k) (Fin n) 𝕜} (h : IsQR A Q R) (hk : k ≤ n)
    (hA : LinearIndependent 𝕜 (A.submatrix id (Fin.castLE hk)).col) :
    IsQR A Q' R' ↔ ∃ s : Fin k → 𝕜, (∀ i, s i = 1 ∨ s i = -1) ∧ Q' = Q * diagonal s ∧
      R' = diagonal s * R := by
  have hb : ∀ {Q₀ : Matrix (Fin m) (Fin k) 𝕜} {R₀ : Matrix (Fin k) (Fin n) 𝕜}, IsQR A Q₀ R₀ →
      A.submatrix id (Fin.castLE hk) = Q₀ * R₀.submatrix id (Fin.castLE hk) ∧
      (R₀.submatrix id (Fin.castLE hk)).IsUpperTriangular ∧
      IsUnit (R₀.submatrix id (Fin.castLE hk)).det ∧
      ∀ i, star (R₀.submatrix id (Fin.castLE hk) i i) = R₀.submatrix id (Fin.castLE hk) i i := by
    intro Q₀ R₀ h₀
    have hA₀ : A.submatrix id (Fin.castLE hk) = Q₀ * R₀.submatrix id (Fin.castLE hk) := by
      rw [h₀.2.2.2, submatrix_mul _ _ _ _ _ Function.bijective_id, submatrix_id_id]
    refine ⟨hA₀, upperBand_zero_iff.mp fun i j hij => h₀.2.1 i (Fin.castLE hk j) hij,
      isUnit_det_of_injective (X := Q₀) ?_,
      fun i => h₀.2.2.1 ⟨i, lt_min i.isLt (lt_of_lt_of_le i.isLt hk)⟩⟩
    rw [← hA₀]
    exact mulVec_injective_iff.mpr hA
  have hR : ∀ {Q₀ : Matrix (Fin m) (Fin k) 𝕜} {R₀ : Matrix (Fin k) (Fin n) 𝕜}, IsQR A Q₀ R₀ →
      R₀ = Q₀ᴴ * A := fun h₀ => by rw [h₀.2.2.2, ← Matrix.mul_assoc, h₀.1, Matrix.one_mul]
  have hst : ∀ {s : Fin k → 𝕜}, (∀ i, s i = 1 ∨ s i = -1) → star s = s := fun hs =>
    funext fun i => by rcases hs i with h | h <;> simp [h]
  obtain ⟨hA₁, hT, hu, hd⟩ := hb h
  constructor
  · intro h'
    obtain ⟨hA₁', hT', hu', hd'⟩ := hb h'
    obtain ⟨s, hs, hQ⟩ := exists_signs h.1 h'.1 hT hT' hu hu' hd hd' (hA₁.symm.trans hA₁')
    refine ⟨s, hs, hQ, ?_⟩
    rw [hR h', hQ, conjTranspose_mul, diagonal_conjTranspose, hst hs, Matrix.mul_assoc, ← hR h]
  · rintro ⟨s, hs, rfl, rfl⟩
    have hss : diagonal s * diagonal s = (1 : Matrix (Fin k) (Fin k) 𝕜) := by
      rw [diagonal_mul_diagonal, ← diagonal_one]
      congr 1
      funext i
      rcases hs i with h | h <;> simp [h]
    refine ⟨?_, fun i j hij => ?_, fun i => ?_, ?_⟩
    · rw [conjTranspose_mul, diagonal_conjTranspose, hst hs, Matrix.mul_assoc,
        ← Matrix.mul_assoc Qᴴ, h.1, Matrix.one_mul, hss]
    · rw [diagonal_mul, h.2.1 i j hij, mul_zero]
    · have hi := h.2.2.1 i
      simp only [IsSelfAdjoint, rectDiag, diagonal_mul] at hi ⊢
      rcases hs (Fin.castLE (min_le_left k n) i) with h' | h' <;> simp [h', hi]
    · rw [Matrix.mul_assoc, ← Matrix.mul_assoc (diagonal s), hss, Matrix.one_mul, h.2.2.2]

/-! ## Householder reflectors -/

/-- The elementary reflector `1 - τ v vᴴ`. -/
def reflector (v : Fin m → 𝕜) (τ : 𝕜) : Matrix (Fin m) (Fin m) 𝕜 :=
  1 - τ • vecMulVec v (star v)

/-- The Householder vector that the column `c` stores from row `j`: zero above row `j`, one at
row `j` and `c` below. -/
def householderVec (c : Fin m → 𝕜) (j : ℕ) : Fin m → 𝕜 := fun r =>
  if r.val < j then 0 else if r.val = j then 1 else c r

/-- The product `H₀ H₁ ⋯ H_{k-1}` of the reflectors stored in the first `k` columns of `a`, `H_i`
the reflector of `τ i` and of the Householder vector that column `i` stores from row `i + s`.
`s = 0` is the representation of `geqrf` and `householder_product`, `s = 1` that of `hessenberg`
and of `tridiagonal` with `lower=True`.

We wrote it by hand from the docstring of `torch.linalg.householder_product` in PyTorch 2.13
(torch/linalg/__init__.py), which defines `H_i = I_m - τ_i v_i v_iᴴ` with `v_i` the column `a_i`
with its first `i - 1` entries zeroed and a one at entry `i`, counting from one. -/
def householderQ (a : Matrix (Fin m) (Fin n) 𝕜) (s : ℕ) (τ : Fin k → 𝕜) (h : k ≤ n) :
    Matrix (Fin m) (Fin m) 𝕜 :=
  (List.ofFn fun i => reflector (householderVec (aᵀ (Fin.castLE h i)) (i.val + s)) (τ i)).prod

/-- The first `n` columns of `householderQ a 0 τ hk`: `householder_product(a, τ)`.

We wrote it by hand from the docstrings of `torch.linalg.householder_product` in PyTorch 2.13
(torch/linalg/__init__.py), "computes the first `n` columns of a product of Householder
matrices", which requires `m ≥ n ≥ k`, and of `jax.lax.linalg.householder_product` in JAX 0.11.1
(jax/_src/lax/linalg.py), whose result has the shape of `a`. -/
def householderProduct (a : Matrix (Fin m) (Fin n) 𝕜) (τ : Fin k → 𝕜) (hk : k ≤ n)
    (hn : n ≤ m) : Matrix (Fin m) (Fin n) 𝕜 :=
  (householderQ a 0 τ hk).submatrix id (Fin.castLE hn)

/-- `op(Q)` for `Q = householderQ a 0 τ h`, `op` the conjugate transpose when `transpose` is set.
`ormqr(a, τ, C, left, transpose)` is `householderOp a τ h transpose * C` when `left` is set and
`C * householderOp a τ h transpose` otherwise.

We wrote it by hand from the docstrings of `torch.ormqr` in PyTorch 2.13 (torch/_torch_docs.py),
"If `left` is `True` then `op(Q)` times `C` is computed, otherwise the result is `C` times
`op(Q)`. ... If `transpose` is `True` then `op` is the conjugate transpose operation", and of
`jax.lax.linalg.ormqr` in JAX 0.11.1 (jax/_src/lax/linalg.py), which says the same of `Q @ C`,
`Q^T @ C`, `C @ Q` and `C @ Q^T`, with `Q^H` for complex types. -/
def householderOp (a : Matrix (Fin m) (Fin n) 𝕜) (τ : Fin k → 𝕜) (h : k ≤ n) (transpose : Bool) :
    Matrix (Fin m) (Fin m) 𝕜 :=
  if transpose then (householderQ a 0 τ h)ᴴ else householderQ a 0 τ h

/-- `(a, τ)` is a `geqrf` factorization of `A`: `R` is the upper triangle of `a`, the reflectors
below its diagonal and `τ` give `Q = householderQ a 0 τ`, and `(Q, R)` is a QR factorization of
`A` (with `Q` square).

We wrote it by hand from the docstrings of `torch.geqrf` in PyTorch 2.13 (torch/_torch_docs.py),
"The elements of `R` are stored on and above the diagonal. Elementary reflectors (or Householder
vectors) implicitly defining matrix `Q` are stored below the diagonal", and of `geqrf` in JAX
0.11.1 (jax/_src/lax/linalg.py, where `jax.lax.linalg.qr` calls it; `jax.lax.linalg` does not
export it), "`r` is in the upper triangle of `a`, `q` is represented in the lower triangle of `a`
and in `taus` as elementary Householder reflectors". `exists_isGeqrf` gives every `A` one. -/
def IsGeqrf (A a : Matrix (Fin m) (Fin n) 𝕜) (τ : Fin (min m n) → 𝕜) : Prop :=
  IsQR A (householderQ a 0 τ (min_le_right m n)) (upperPart 0 a)

/-- `(a, P, τ)` is a `geqp3` factorization of `A`: `(a, τ)` is a `geqrf` factorization of the
columns of `A` permuted by `P`.

We wrote it by hand from the docstring of `geqp3` in JAX 0.11.1 (jax/_src/lax/linalg.py, not
exported by `jax.lax.linalg`), "`jpvt` is the column-pivot indices such that `a[:, jpvt] = q @ r`",
which, unlike `qr` with `pivoting=True`, says nothing of the order of the diagonal. The primitive's
`jpvt` is 1-based, as `qr`'s lowering there says ("Convert geqp3's 1-based indices to 0-based
indices by subtracting 1"), so `P j = jpvt[j] - 1`. `exists_isGeqrf` of `A.submatrix id P` gives
every `A` one for every `P`. -/
def IsGeqp3 (A : Matrix (Fin m) (Fin n) 𝕜) (P : Equiv.Perm (Fin n)) (a : Matrix (Fin m) (Fin n) 𝕜)
    (τ : Fin (min m n) → 𝕜) : Prop :=
  IsGeqrf (A.submatrix id P) a τ

/-- The reflector of `τ = 0` is the identity. -/
theorem reflector_zero (v : Fin m → 𝕜) : reflector v 0 = 1 := by
  simp [reflector]

/-- With every `τ i = 0`, `householderQ a s τ h` is the identity. -/
theorem householderQ_zero (a : Matrix (Fin m) (Fin n) 𝕜) (s : ℕ) (h : k ≤ n) :
    householderQ a s (0 : Fin k → 𝕜) h = 1 := by
  unfold householderQ
  refine List.prod_eq_one fun M hM => ?_
  obtain ⟨i, rfl⟩ := List.mem_ofFn.mp hM
  exact reflector_zero _

/-- A reflector `1 - τ v vᴴ` is unitary when `star τ + τ = star τ τ (vᴴ v)`: for real `τ` and
`v ≠ 0`, when `τ = 0` or `τ = 2 / (vᴴ v)`. -/
theorem conjTranspose_reflector_mul_self {v : Fin m → 𝕜} {τ : 𝕜}
    (h : star τ + τ = star τ * τ * (star v ⬝ᵥ v)) : (reflector v τ)ᴴ * reflector v τ = 1 := by
  have hP : (vecMulVec v (star v))ᴴ = vecMulVec v (star v) := by
    rw [conjTranspose_vecMulVec, star_star]
  have hPP : vecMulVec v (star v) * vecMulVec v (star v) =
      (star v ⬝ᵥ v) • vecMulVec v (star v) := by
    rw [vecMulVec_mul_vecMulVec]
    ext i j
    simp only [vecMulVec_apply, Pi.smul_apply, smul_eq_mul, Matrix.smul_apply]
    ring
  have key : star τ • vecMulVec v (star v) * τ • vecMulVec v (star v) =
      (star τ + τ) • vecMulVec v (star v) := by
    rw [Matrix.smul_mul, Matrix.mul_smul, hPP, smul_smul, smul_smul, h, mul_assoc]
  rw [reflector, conjTranspose_sub, conjTranspose_one, conjTranspose_smul, hP]
  simp only [sub_mul, mul_sub, one_mul, mul_one]
  rw [key, add_smul]
  abel

/-- `householderQ a s τ h` is unitary when each of its reflectors satisfies the condition of
`conjTranspose_reflector_mul_self`. -/
theorem conjTranspose_householderQ_mul_self {a : Matrix (Fin m) (Fin n) 𝕜} {s : ℕ}
    {τ : Fin k → 𝕜} {h : k ≤ n}
    (hτ : ∀ i, star (τ i) + τ i = star (τ i) * τ i *
      (star (householderVec (aᵀ (Fin.castLE h i)) (i.val + s)) ⬝ᵥ
        householderVec (aᵀ (Fin.castLE h i)) (i.val + s))) :
    (householderQ a s τ h)ᴴ * householderQ a s τ h = 1 := by
  have hmem : householderQ a s τ h ∈ unitaryGroup (Fin m) 𝕜 := by
    refine list_prod_mem fun M hM => ?_
    obtain ⟨i, rfl⟩ := List.mem_ofFn.mp hM
    rw [mem_unitaryGroup_iff', star_eq_conjTranspose]
    exact conjTranspose_reflector_mul_self (hτ i)
  rw [← star_eq_conjTranspose]
  exact mem_unitaryGroup_iff'.mp hmem

/-- `householderProduct a τ hk hn` has orthonormal columns when each reflector satisfies the
condition of `conjTranspose_reflector_mul_self`. -/
theorem conjTranspose_householderProduct_mul_self {a : Matrix (Fin m) (Fin n) 𝕜}
    {τ : Fin k → 𝕜} {hk : k ≤ n} (hn : n ≤ m)
    (hτ : ∀ i, star (τ i) + τ i = star (τ i) * τ i *
      (star (householderVec (aᵀ (Fin.castLE hk i)) (i.val + 0)) ⬝ᵥ
        householderVec (aᵀ (Fin.castLE hk i)) (i.val + 0))) :
    (householderProduct a τ hk hn)ᴴ * householderProduct a τ hk hn = 1 := by
  rw [householderProduct, conjTranspose_submatrix,
    ← submatrix_mul _ _ _ _ _ Function.bijective_id, conjTranspose_householderQ_mul_self hτ,
    submatrix_one _ (Fin.castLE_injective hn)]

/-- An upper triangular `A` whose diagonal entries are self-adjoint is its own `geqrf` output with
`τ = 0`. -/
theorem isGeqrf_self {A : Matrix (Fin m) (Fin n) 𝕜} (hA : UpperBand 0 A)
    (hd : ∀ i, IsSelfAdjoint (rectDiag A i)) : IsGeqrf A A 0 := by
  rw [IsGeqrf, householderQ_zero, upperPart_eq_self hA]
  exact isQR_one hA hd

/-- An upper triangular `A` whose diagonal entries are self-adjoint is its own `geqp3` output with
`P = 1` and `τ = 0`. -/
theorem isGeqp3_one {A : Matrix (Fin m) (Fin n) 𝕜} (hA : UpperBand 0 A)
    (hd : ∀ i, IsSelfAdjoint (rectDiag A i)) : IsGeqp3 A 1 A 0 := by
  rw [IsGeqp3, Equiv.Perm.coe_one, submatrix_id_id]
  exact isGeqrf_self hA hd

/-! ## LU factorizations -/

/-- `(σ, L, U)` is an LU factorization of `A` with row pivoting: `L` is lower triangular with
ones on its diagonal, `U` is upper triangular, and row `i` of `L U` is row `σ i` of `A`.

We wrote it by hand from the docstrings of `jax.lax.linalg.lu` in JAX 0.11.1
(jax/_src/lax/linalg.py), "`P A = L U` where `P` is a permutation of the rows of `A`, `L` is a
lower-triangular matrix with unit-diagonal elements, and `U` is an upper-triangular matrix", whose
`permutation` output is `σ`, and of `torch.linalg.lu` in PyTorch 2.13 (torch/linalg/__init__.py),
"`A = PLU` ... `L ∈ 𝕂^{m × k}, U ∈ 𝕂^{k × n}` where `k = min(m,n)`" (`IsLU.eq_mul`). JAX's `lu`
output holds `L` below its diagonal and `U` on and above it. PyTorch warns that the factorization
"is almost never unique"; `IsLU` leaves `σ` free, and with it the size of `L`'s entries, and
`IsLU.unique` shows `σ` is the only freedom when the first `min m n` columns of `A` are linearly
independent. `exists_isLU` gives every `A` one, by Gaussian elimination. -/
def IsLU (A : Matrix (Fin m) (Fin n) 𝕜) (σ : Equiv.Perm (Fin m)) (L : Matrix (Fin m) (Fin k) 𝕜)
    (U : Matrix (Fin k) (Fin n) 𝕜) : Prop :=
  UpperBand 0 Lᵀ ∧ (∀ i, rectDiag L i = 1) ∧ UpperBand 0 U ∧ A.submatrix σ id = L * U

/-- The permutation of the row swaps `piv`: starting from the identity, swap the entries at `i`
and `piv i` for `i = 0, …, k - 1`, so that `pivotPerm h piv = swap 0 (piv 0) * ⋯`.

We wrote it by hand from the docstring of `jax.lax.linalg.lu_pivots_to_permutation` in JAX 0.11.1
(jax/_src/lax/linalg.py), "Converts the pivots (row swaps) returned by LU to a permutation", which
requires `permutation_size >= k`, and from its generic lowering there
(`_generic_lu_pivots_to_permutation`), which swaps the entries at `i` and `pivots[i]` of `iota`
in turn. -/
def pivotPerm (h : k ≤ m) (piv : Fin k → Fin m) : Equiv.Perm (Fin m) :=
  (List.ofFn fun i => Equiv.swap (Fin.castLE h i) (piv i)).prod

/-- PyTorch's form `A = P L U`, with `P` the permutation matrix of `σ⁻¹`. -/
theorem IsLU.eq_mul {A : Matrix (Fin m) (Fin n) 𝕜} {σ : Equiv.Perm (Fin m)}
    {L : Matrix (Fin m) (Fin k) 𝕜} {U : Matrix (Fin k) (Fin n) 𝕜} (h : IsLU A σ L U) :
    A = Equiv.Perm.permMatrix 𝕜 σ⁻¹ * (L * U) := by
  rw [← h.2.2.2, Equiv.Perm.permMatrix, PEquiv.toMatrix_toPEquiv_mul, submatrix_submatrix,
    ← Equiv.Perm.coe_mul, mul_inv_cancel, Equiv.Perm.coe_one, Function.comp_id, submatrix_id_id]

/-- An upper triangular `U` of `m` rows is its own LU decomposition with `σ = 1` and `L` the
identity of size `m`. -/
theorem isLU_one {U : Matrix (Fin m) (Fin n) 𝕜} (hU : UpperBand 0 U) :
    IsLU U 1 (1 : Matrix (Fin m) (Fin m) 𝕜) U := by
  refine ⟨fun i j hij => ?_, fun i => by simp [rectDiag], hU, ?_⟩
  · rw [transpose_one]
    exact one_apply_ne (by rintro rfl; omega)
  · rw [Equiv.Perm.coe_one, submatrix_id_id, Matrix.one_mul]

/-- Swapping each row `i < k` with itself, `piv i = i`, gives the identity permutation. -/
theorem pivotPerm_castLE (h : k ≤ m) : pivotPerm h (Fin.castLE h) = 1 := by
  unfold pivotPerm
  refine List.prod_eq_one fun x hx => ?_
  obtain ⟨i, rfl⟩ := List.mem_ofFn.mp hx
  exact (Equiv.swap_self _).trans Equiv.Perm.one_def.symm

/-- The square case of `IsLU.unique`: `L U = L' U'` with `L`, `L'` lower triangular with unit
diagonals and `U`, `U'` upper triangular, `U` invertible, gives `L = L'` and `U = U'`. -/
theorem eq_of_mul_eq_mul {L L' U U' : Matrix (Fin k) (Fin k) 𝕜} (hl : L.IsLowerTriangular)
    (hl' : L'.IsLowerTriangular) (hd : ∀ i, L i i = 1) (hd' : ∀ i, L' i i = 1)
    (hu : U.IsUpperTriangular) (hu' : U'.IsUpperTriangular) (hU : IsUnit U.det)
    (h : L * U = L' * U') : L = L' ∧ U = U' := by
  have hL' : IsUnit L'.det := by
    rw [det_of_isLowerTriangular _ hl']
    simp [hd']
  have := invertibleOfIsUnitDet L' hL'
  have := invertibleOfIsUnitDet U hU
  have hLM : L = L' * (U' * U⁻¹) := by
    rw [← Matrix.mul_assoc, ← h, mul_nonsing_inv_cancel_right U _ hU]
  have hML : U' * U⁻¹ = L'⁻¹ * L := by rw [hLM, nonsing_inv_mul_cancel_left L' _ hL']
  have hD : (U' * U⁻¹).IsDiag := isDiag_of_triangular
    (hu'.mul (blockTriangular_inv_of_blockTriangular hu))
    (hML ▸ (blockTriangular_inv_of_blockTriangular hl').mul hl)
  have h1 : U' * U⁻¹ = 1 := by
    rw [← hD.diagonal_diag, ← diagonal_one]
    congr 1
    funext i
    have := congrFun (congrFun hLM i) i
    rw [← hD.diagonal_diag, mul_diagonal, hd, hd', one_mul] at this
    exact this.symm
  refine ⟨by rw [hLM, h1, Matrix.mul_one], ?_⟩
  calc U = 1 * U := (Matrix.one_mul U).symm
    _ = U' * U⁻¹ * U := by rw [h1]
    _ = U' := nonsing_inv_mul_cancel_right U U' hU

/-- With the first `k` columns of `A` linearly independent, an LU factorization of `A` is unique
once its permutation `σ` is fixed. PyTorch warns at `torch.linalg.lu` that "The LU decomposition
is almost never unique, as often there are different permutation matrices that can yield
different LU decompositions"; the permutation is the only freedom left, for every `k ≤ m, n` and
so for `k = min m n`. -/
theorem IsLU.unique {A : Matrix (Fin m) (Fin n) 𝕜} {σ : Equiv.Perm (Fin m)}
    {L L' : Matrix (Fin m) (Fin k) 𝕜} {U U' : Matrix (Fin k) (Fin n) 𝕜} (h : IsLU A σ L U)
    (h' : IsLU A σ L' U') (hm : k ≤ m) (hn : k ≤ n)
    (hA : LinearIndependent 𝕜 (A.submatrix id (Fin.castLE hn)).col) : L = L' ∧ U = U' := by
  have hb : ∀ {L₀ : Matrix (Fin m) (Fin k) 𝕜} {U₀ : Matrix (Fin k) (Fin n) 𝕜}, IsLU A σ L₀ U₀ →
      (A.submatrix σ id).submatrix id (Fin.castLE hn) = L₀ * U₀.submatrix id (Fin.castLE hn) ∧
      (A.submatrix σ id).submatrix (Fin.castLE hm) id = L₀.submatrix (Fin.castLE hm) id * U₀ ∧
      (L₀.submatrix (Fin.castLE hm) id).IsLowerTriangular ∧ (∀ i, L₀ (Fin.castLE hm i) i = 1) ∧
      (U₀.submatrix id (Fin.castLE hn)).IsUpperTriangular ∧
      IsUnit (U₀.submatrix id (Fin.castLE hn)).det := by
    intro L₀ U₀ h₀
    have hA₀ : (A.submatrix σ id).submatrix id (Fin.castLE hn) =
        L₀ * U₀.submatrix id (Fin.castLE hn) := by
      rw [h₀.2.2.2, submatrix_mul _ _ _ _ _ Function.bijective_id, submatrix_id_id]
    refine ⟨hA₀, by rw [h₀.2.2.2, submatrix_mul _ _ _ _ _ Function.bijective_id, submatrix_id_id],
      fun i j hij => h₀.1 j (Fin.castLE hm i) (OrderDual.toDual_lt_toDual.mp hij),
      fun i => h₀.2.1 ⟨i, lt_min (lt_of_lt_of_le i.isLt hm) i.isLt⟩,
      upperBand_zero_iff.mp fun i j hij => h₀.2.2.1 i (Fin.castLE hn j) hij,
      isUnit_det_of_injective (X := L₀) ?_⟩
    rw [← hA₀]
    intro v v' hv
    refine mulVec_injective_iff.mpr hA (funext fun i => ?_)
    simpa [mulVec, dotProduct] using congrFun hv (σ.symm i)
  obtain ⟨hA₁, hA₂, hl, hd, hu, hU⟩ := hb h
  obtain ⟨hA₁', hA₂', hl', hd', hu', -⟩ := hb h'
  obtain ⟨hL, hU'⟩ := eq_of_mul_eq_mul hl hl' hd hd' hu hu' hU (by
    rw [← submatrix_mul _ _ _ _ _ Function.bijective_id,
      ← submatrix_mul _ _ _ _ _ Function.bijective_id, ← h.2.2.2, ← h'.2.2.2])
  have hLu : IsUnit (L.submatrix (Fin.castLE hm) id).det := by
    rw [det_of_isLowerTriangular _ hl]
    simp [hd]
  have hleft : L * U.submatrix id (Fin.castLE hn) = L' * U.submatrix id (Fin.castLE hn) := by
    rw [← hA₁, hU', ← hA₁']
  have htop : L.submatrix (Fin.castLE hm) id * U = L.submatrix (Fin.castLE hm) id * U' := by
    rw [← hA₂, hL, ← hA₂']
  refine ⟨?_, ?_⟩
  · calc L = L * U.submatrix id (Fin.castLE hn) * (U.submatrix id (Fin.castLE hn))⁻¹ :=
          (mul_nonsing_inv_cancel_right _ L hU).symm
      _ = L' := by rw [hleft, mul_nonsing_inv_cancel_right _ L' hU]
  · calc U = (L.submatrix (Fin.castLE hm) id)⁻¹ * (L.submatrix (Fin.castLE hm) id * U) :=
          (nonsing_inv_mul_cancel_left _ U hLu).symm
      _ = U' := by rw [htop, nonsing_inv_mul_cancel_left _ U' hLu]

/-- Every matrix has an LU factorization with row pivoting, by Gaussian elimination and induction
on the number of columns: for a row `p` whose first entry `a = A p 0` is nonzero (any row when
the first column is zero), `L`'s first column is `A i 0 / a`, `U`'s first row is row `p` of `A`,
and a factorization of the Schur complement `A i j - A i 0 / a * A p j` of the other rows and
columns gives the rest, with `σ = decomposeFin.symm (p, σ')`. LAPACK 3.12.0's `getrf2` takes for
`p` the first row of largest `|A p 0|` in `dgetrf2.f` (`IDAMAX`) and of largest
`|re (A p 0)| + |im (A p 0)|` in `zgetrf2.f` (`IZAMAX`, which compares `DCABS1`), not of largest
modulus; `IsLU` asks for neither. -/
theorem exists_isLU (A : Matrix (Fin m) (Fin n) 𝕜) :
    ∃ (σ : Equiv.Perm (Fin m)) (L : Matrix (Fin m) (Fin (min m n)) 𝕜)
      (U : Matrix (Fin (min m n)) (Fin n) 𝕜), IsLU A σ L U := by
  suffices h : ∀ {m} (A : Matrix (Fin m) (Fin n) 𝕜) k, k = min m n →
      ∃ (σ : Equiv.Perm (Fin m)) (L : Matrix (Fin m) (Fin k) 𝕜) (U : Matrix (Fin k) (Fin n) 𝕜),
        IsLU A σ L U by exact h A _ rfl
  clear A
  induction n with
  | zero =>
    intro m A k hk
    obtain rfl : k = 0 := hk.trans (Nat.min_zero m)
    exact ⟨1, 0, 0, fun i => i.elim0, fun i => absurd i.isLt (by simp), fun i => i.elim0,
      ext fun _ j => j.elim0⟩
  | succ n ih =>
    intro m A k hk
    cases m with
    | zero =>
      obtain rfl : k = 0 := hk.trans (Nat.zero_min _)
      exact ⟨1, 0, 0, fun i => i.elim0, fun i => absurd i.isLt (by simp), fun i => i.elim0,
        ext fun i => i.elim0⟩
    | succ m =>
      obtain rfl : k = min m n + 1 := hk.trans (Nat.succ_min_succ m n)
      obtain ⟨p, hp⟩ : ∃ p, A p 0 = 0 → ∀ i, A i 0 = 0 := by
        by_cases h : ∃ i, A i 0 ≠ 0
        · exact ⟨h.choose, fun e => absurd e h.choose_spec⟩
        · exact ⟨0, fun _ i => not_not.mp fun hi => h ⟨i, hi⟩⟩
      obtain ⟨σ, L, U, hl, hd, hu, hLU⟩ := ih (of fun (i : Fin m) (j : Fin n) =>
        A (Equiv.swap 0 p i.succ) j.succ - A (Equiv.swap 0 p i.succ) 0 / A p 0 * A p j.succ) _ rfl
      refine ⟨Equiv.Perm.decomposeFin.symm (p, σ), of (vecCons (vecCons 1 0) fun i : Fin m =>
        vecCons (A (Equiv.swap 0 p (σ i).succ) 0 / A p 0) (L i)),
        of (vecCons (A p) fun i : Fin (min m n) => vecCons 0 (U i)), fun i j hij => ?_,
        fun i => ?_, fun i j hij => ?_, ?_⟩
      · rw [transpose_apply]
        cases i using Fin.cases with
        | zero => exact absurd hij (by simp)
        | succ i =>
          cases j using Fin.cases with
          | zero => simp
          | succ j =>
            simp only [Fin.val_succ] at hij
            simpa using hl i j (by omega)
      · obtain ⟨v, hv⟩ := i
        obtain ⟨hv1, hv2⟩ := Nat.lt_min.mp hv
        cases v with
        | zero => simp [rectDiag]
        | succ v =>
          have e1 : (Fin.castLE (min_le_left _ _) ⟨v + 1, hv⟩ : Fin (m + 1)) =
              Fin.succ ⟨v, by omega⟩ := rfl
          have e2 : (Fin.castLE (min_le_right _ _) ⟨v + 1, hv⟩ : Fin (min m n + 1)) =
              Fin.succ ⟨v, by omega⟩ := rfl
          rw [rectDiag, e1, e2]
          simp only [of_apply, cons_val_succ]
          exact hd ⟨v, Nat.lt_min.mpr ⟨by omega, by omega⟩⟩
      · cases i using Fin.cases with
        | zero => exact absurd hij (by simp)
        | succ i =>
          cases j using Fin.cases with
          | zero => simp
          | succ j =>
            simp only [Fin.val_succ] at hij
            simpa using hu i j (by omega)
      · ext i j
        rw [Matrix.mul_apply, Fin.sum_univ_succ]
        cases i using Fin.cases with
        | zero => simp
        | succ i =>
          rw [submatrix_apply, Equiv.Perm.decomposeFin_symm_apply_succ]
          cases j using Fin.cases with
          | zero =>
            simp only [of_apply, cons_val_zero, cons_val_succ, mul_zero, Finset.sum_const_zero,
              add_zero, id_eq]
            exact (div_mul_cancel_of_imp fun h => hp h _).symm
          | succ j =>
            have e := congrFun (congrFun hLU i) j
            simp only [submatrix_apply, Matrix.mul_apply, of_apply, id_eq] at e
            simp only [of_apply, cons_val_zero, cons_val_succ, id_eq]
            rw [← e]
            ring

/-! ## Spectral decompositions -/

/-- `(V, w)` is an eigendecomposition of `A`: `V` is unitary, `w` is real and ascending, and
`A = V diag(w) Vᴴ`, so column `i` of `V` is a unit eigenvector of `w i`.

We wrote it by hand from the docstrings of `torch.linalg.eigh` in PyTorch 2.13
(torch/linalg/__init__.py), "`A = Q diag(Λ) Q^H`, `Q ∈ 𝕂^{n × n}, Λ ∈ ℝ^n` ... The eigenvalues
are returned in ascending order", and of `jax.lax.linalg.eigh` in JAX 0.11.1
(jax/_src/lax/linalg.py), "`v[..., :, i]` is the normalized eigenvector corresponding to
eigenvalue `w[..., i]`" with `w` "in ascending order" when `sort_eigenvalues` is set, its
default. The relation is for a Hermitian `A`: PyTorch reads the lower triangle of its input by
default, and JAX's `eigh` passes `(x + xᴴ) / 2` to the primitive by default. `subset_by_index`
is not covered. PyTorch warns that "the eigenvectors of a symmetric matrix are not unique";
`IsEigh.eigenvalues_eq` shows the eigenvalues are, and `IsEigh.isEigh_iff` that with distinct
eigenvalues the eigenvectors are unique up to the phases PyTorch names. -/
def IsEigh (A V : Matrix (Fin n) (Fin n) 𝕜) (w : Fin n → ℝ) : Prop :=
  Vᴴ * V = 1 ∧ Monotone w ∧ A = V * diagonal (fun i => (w i : 𝕜)) * Vᴴ

open Polynomial in
/-- `(w, V)` is an eigendecomposition of a complex `A`: `w` lists the roots of the characteristic
polynomial with their multiplicities, the columns of `V` have unit norm, `A V = V diag(w)`, and
`V` is invertible when `A` is diagonalizable (`A P = P diag(d)` for some invertible `P`).

We wrote it by hand from the docstrings of `torch.linalg.eig` in PyTorch 2.13
(torch/linalg/__init__.py), "`A = V diag(Λ) V^{-1}`", a decomposition that "exists if and only if
`A` is diagonalizable", "If it is not diagonalizable, the returned eigenvalues will be correct but
`A ≠ V diag(Λ)V^{-1}`", and "The returned eigenvectors are normalized to have norm `1`", and of
`jax.lax.linalg.eig` in JAX 0.11.1 (jax/_src/lax/linalg.py), which at `enable_eigvec_derivs`
describes LAPACK `geev`'s normalisation, "each eigenvector has unit 2-norm and its
largest-magnitude component is real", as LAPACK 3.12.0 states it (zgeev.f, dgeev.f): "The computed
eigenvectors are normalized to have Euclidean norm equal to 1 and largest component real" (`IsEig`
leaves that phase free). For a diagonalizable `A` the relation is the decomposition,
`A = V diag(w) V⁻¹`; for any other `A` the documentation promises only the eigenvalues, and the
relation asks only that the columns of `V` be unit eigenvectors. `exists_isEig` gives every complex
`A` an output. A real input is read as its complex image, and JAX's left eigenvectors `vl`,
`vlᴴ A = diag(w) vlᴴ`, are read as `IsEig Aᴴ (star w) vl`. `IsEig.isEig_iff` shows that with
distinct eigenvalues the order of the eigenvalues and the phases of the eigenvectors are the only
freedom. -/
def IsEig (A : Matrix (Fin n) (Fin n) ℂ) (w : Fin n → ℂ) (V : Matrix (Fin n) (Fin n) ℂ) : Prop :=
  A.charpoly = ∏ i, (X - C (w i)) ∧ (∀ j, (Vᴴ * V) j j = 1) ∧ A * V = V * diagonal w ∧
    ∀ (P : Matrix (Fin n) (Fin n) ℂ) (d : Fin n → ℂ), IsUnit P.det → A * P = P * diagonal d →
      IsUnit V.det

/-- `(U, S, Vh)` is a singular value decomposition of `A`: `U` and `Vhᴴ` have orthonormal
columns, `S` is nonnegative and descending, `A = U diag(S) Vh` with `diag(S)` of shape `p × q`,
and `U` has at least `min m n` columns and `Vh` at least `min m n` rows, `min m n ≤ p, q`, so
that every entry of `S` is on the diagonal of `diag(S)`.

We wrote it by hand from the docstrings of `torch.linalg.svd` in PyTorch 2.13
(torch/linalg/__init__.py), "`A = U diag(S) V^H`" with `U ∈ 𝕂^{m × m}, S ∈ ℝ^k, V ∈ 𝕂^{n × n}`
when `full_matrices=True` (`p = m`, `q = n`) and `U ∈ 𝕂^{m × k}`, `V ∈ 𝕂^{n × k}` otherwise
(`p = q = k = min m n`), "The singular values are returned in descending order", and of
`jax.lax.linalg.svd` in JAX 0.11.1 (jax/_src/lax/linalg.py), which returns "the left singular
vectors, the singular values, and the adjoint of the right singular vectors"; both of PyTorch's
shapes have `min m n ≤ p, q`. `exists_isSVD` gives every `A` a full decomposition and
`IsSVD.reduce` a reduced one. PyTorch warns that `U` and `V` are not unique;
`IsSVD.singularValues_eq` shows `S` is, and `IsSVD.isSVD_iff` that with distinct positive singular
values the reduced `U` and `Vh` are unique up to a phase for each pair of singular vectors.
`subset_by_index` is not covered. -/
def IsSVD (A : Matrix (Fin m) (Fin n) 𝕜) (U : Matrix (Fin m) (Fin p) 𝕜) (S : Fin (min m n) → ℝ)
    (Vh : Matrix (Fin q) (Fin n) 𝕜) : Prop :=
  Uᴴ * U = 1 ∧ Vh * Vhᴴ = 1 ∧ (∀ i, 0 ≤ S i) ∧ Antitone S ∧
    A = U * rectDiagonal (fun i => (S i : 𝕜)) * Vh ∧ min m n ≤ p ∧ min m n ≤ q

/-- `(T, Q)` is a Schur decomposition of `A`: `Q` is unitary, `T` is upper triangular, and
`A = Q T Qᴴ`.

We wrote it by hand from the docstring of `jax.lax.linalg.schur` in JAX 0.11.1
(jax/_src/lax/linalg.py), "`A = Q U Q^{-H}`", read as `A = Q U Qᴴ`: the docstring's `Q^{-H}`,
the inverse of `Qᴴ`, is `Q` itself for a unitary `Q`, so its formula read literally is `Q U Q`,
which is not the decomposition JAX returns. `T` is triangular as in SciPy 1.17's
`scipy.linalg.schur` (scipy/linalg/_decomp_schur.py): "T is either upper-triangular, or for real
Schur decomposition (output='real'), quasi-upper triangular". The relation is for a complex
input: `jax.scipy.linalg.schur` (jax/_src/scipy/linalg.py) converts its input to complex before
calling `lax.linalg.schur` when `output='complex'` and passes a real input as it is otherwise,
whose output is the real form, `IsRealSchur`. `exists_isSchur` gives every complex `A` one. -/
def IsSchur (A Q T : Matrix (Fin n) (Fin n) 𝕜) : Prop :=
  Qᴴ * Q = 1 ∧ T.IsUpperTriangular ∧ A = Q * T * Qᴴ

/-- `(T, Q)` is a real Schur decomposition of a real `A`: `Q` is orthogonal, `A = Q T Qᵀ`, and
`T` is upper triangular but for 2 × 2 diagonal blocks of complex conjugate eigenvalues: each
nonzero subdiagonal `T i j` (`i = j + 1`) has the next subdiagonal entry zero, and its block's
discriminant `(T j j - T i i)² + 4 T j i T i j` negative.

We wrote it by hand from the docstring of `scipy.linalg.schur` in SciPy 1.17
(scipy/linalg/_decomp_schur.py): "for real Schur decomposition (output='real'), quasi-upper
triangular. In the quasi-triangular form, 2x2 blocks describing complex-valued eigenvalue pairs
may extrude from the diagonal". `exists_isRealSchur` (`TorchLean.Tensor.Factorization.RealSchur`)
gives every real `A` one. -/
def IsRealSchur (A Q T : Matrix (Fin n) (Fin n) ℝ) : Prop :=
  Qᵀ * Q = 1 ∧ T.IsUpperHessenberg ∧
    (∀ i j : Fin n, j.val + 1 = i.val → T i j ≠ 0 →
      (T j j - T i i) ^ 2 + 4 * T j i * T i j < 0 ∧ ∀ l : Fin n, i.val + 1 = l.val → T l i = 0) ∧
    A = Q * T * Qᵀ

/-- `(H, Q)` is a Hessenberg decomposition of `A`: `Q` is unitary, `H` is upper Hessenberg
(Mathlib's `Matrix.IsUpperHessenberg`), and `A = Q H Qᴴ`. `exists_isHessenberg` gives every `A`
one. -/
def IsHessenberg (A Q H : Matrix (Fin n) (Fin n) 𝕜) : Prop :=
  Qᴴ * Q = 1 ∧ H.IsUpperHessenberg ∧ A = Q * H * Qᴴ

/-- `(a, τ)` is a `hessenberg` output for `A`: `H` is `a` on and above its first subdiagonal, the
reflectors below it and `τ` give `Q = householderQ a 1 τ`, and `(H, Q)` is a Hessenberg
decomposition of `A`.

We wrote it by hand from the docstring of `jax.lax.linalg.hessenberg` in JAX 0.11.1
(jax/_src/lax/linalg.py): "the upper triangle and first subdiagonal of `a` contain the upper
Hessenberg matrix, and the elements below the first subdiagonal contain the Householder reflectors".
The reflectors' storage, one row below the diagonal, is LAPACK's: the primitive's only lowering,
`_hessenberg_cpu_lowering` there, calls `gehrd` with `ilo = 1` and `ihi = n`, and LAPACK 3.12.0
documents `zgehrd` (zgehrd.f, "Further Details"; dgehrd.f with a real `tau` and `v` and `v**T`) as
"`Q = H(ilo) H(ilo+1) . . . H(ihi-1)`. Each `H(i)` has the form `H(i) = I - tau * v * v**H` where
`tau` is a complex scalar, and `v` is a complex vector with `v(1:i) = 0`, `v(i+1) = 1` and
`v(ihi+1:n) = 0`; `v(i+2:ihi)` is stored on exit in `A(i+2:ihi,i)`, and `tau` in `TAU(i)`": counting
from zero, reflector `i` has its one at row `i + 1`, `householderQ`'s offset `s = 1`. We checked the
quotation word for word against zgehrd.f and dgehrd.f at the tag v3.12.0 of
github.com/Reference-LAPACK/lapack. The installed `jax.scipy.linalg.hessenberg`
(jax/_src/scipy/linalg.py), documented as "`A = Q H Q^H`", builds the same `Q`, as
`diag(1, householder_product(a_out[..., 1:, :-1], taus))`, which fixes the offset.
`exists_isGehrd` gives every `A` one. -/
def IsGehrd (A a : Matrix (Fin n) (Fin n) 𝕜) (τ : Fin (n - 1) → 𝕜) : Prop :=
  IsHessenberg A (householderQ a 1 τ (Nat.sub_le n 1)) (upperPart 1 a)

/-- The matrix with diagonal `d`, subdiagonal `dl i` at `(i, i - 1)` and superdiagonal `du i`
at `(i, i + 1)`; `dl 0` and `du (n - 1)` are not used. -/
def tridiagonal [Zero α] (dl d du : Fin n → α) : Matrix (Fin n) (Fin n) α :=
  of fun i j => if i = j then d i else if j.val + 1 = i.val then dl i
    else if i.val + 1 = j.val then du i else 0

/-- The real symmetric tridiagonal matrix with diagonal `d` and off-diagonal `e`. -/
def symTridiagonal (d : Fin n → ℝ) (e : Fin (n - 1) → ℝ) : Matrix (Fin n) (Fin n) 𝕜 :=
  tridiagonal (fun i => if h : 0 < i.val then (e ⟨i.val - 1, by have := i.isLt; omega⟩ : 𝕜)
      else 0) (fun i => (d i : 𝕜))
    (fun i => if h : i.val + 1 < n then (e ⟨i.val, by omega⟩ : 𝕜) else 0)

/-- `(Q, d, e)` is a tridiagonal decomposition of `A`: `Q` is unitary and `A = Q T Qᴴ` for the
real symmetric tridiagonal `T` of `d` and `e`. -/
def IsTridiagonal (A Q : Matrix (Fin n) (Fin n) 𝕜) (d : Fin n → ℝ) (e : Fin (n - 1) → ℝ) :
    Prop :=
  Qᴴ * Q = 1 ∧ A = Q * symTridiagonal d e * Qᴴ

/-- `(a, d, e, τ)` is a `tridiagonal` output for `A` with `lower=True`: the reflectors below the
first subdiagonal of `a` and `τ` give `Q = householderQ a 1 τ`, `(Q, d, e)` is a tridiagonal
decomposition of `A`, and the diagonal and first subdiagonal of `a` are `d` and `e`.

We wrote it by hand from the docstring of `jax.lax.linalg.tridiagonal` in JAX 0.11.1
(jax/_src/lax/linalg.py): "If `lower=True`, the diagonal and first subdiagonal of matrix ... `a`
contain the tridiagonal representation, and elements below the first subdiagonal contain the
elementary Householder reflectors, where additionally `d` contains the diagonal of the matrix and
`e` contains the first subdiagonal". The relation is for a Hermitian `A`, whose upper triangle the
primitive does not read. The reflectors' storage is LAPACK's: the CPU lowering,
`_tridiagonal_cpu_gpu_lowering` there, calls `sytrd` or `hetrd` with `uplo = 'L'` for `lower=True`,
and LAPACK 3.12.0 documents `zhetrd` (zhetrd.f, "Further Details"; dsytrd.f with a real `tau` and
`v` and `v**T`) as "If `UPLO = 'L'`, the matrix `Q` is represented as a product of elementary
reflectors `Q = H(1) H(2) . . . H(n-1)`. Each `H(i)` has the form `H(i) = I - tau * v * v**H` where
`tau` is a complex scalar, and `v` is a complex vector with `v(1:i) = 0` and `v(i+1) = 1`;
`v(i+2:n)` is stored on exit in `A(i+2:n,i)`, and `tau` in `TAU(i)`", the offset `s = 1`, as for
`IsGehrd`. We checked the quotation word for word against zhetrd.f and dsytrd.f at the tag v3.12.0
of github.com/Reference-LAPACK/lapack. No installed JAX code builds `Q` from this output; JAX's
docstring, with the reflectors below the first subdiagonal and `e` on it, is consistent with the
offset but does not fix it. `lower=False` is not covered. `exists_isSytrd` gives every Hermitian `A`
one. -/
def IsSytrd (A a : Matrix (Fin n) (Fin n) 𝕜) (d : Fin n → ℝ) (e : Fin (n - 1) → ℝ)
    (τ : Fin (n - 1) → 𝕜) : Prop :=
  IsTridiagonal A (householderQ a 1 τ (Nat.sub_le n 1)) d e ∧ (∀ i, a i i = (d i : 𝕜)) ∧
    ∀ j : Fin (n - 1), a ⟨j.val + 1, by have := j.isLt; omega⟩ ⟨j.val, by have := j.isLt; omega⟩
      = (e j : 𝕜)

/-- `X` solves the tridiagonal system `A X = B`, `A = tridiagonal dl d du`.

We wrote it by hand from the docstring of `jax.lax.linalg.tridiagonal_solve` in JAX 0.11.1
(jax/_src/lax/linalg.py): "`dl[i] := A[i, i-1]` ... `d[i]  := A[i, i]` ... `du[i] := A[i, i+1]`".
It notes "`dl[0] = 0`", and under `du` "`dl[m - 1] = 0`", evidently meant of `du[m - 1]`;
`tridiagonal` reads neither entry. -/
def IsTridiagonalSolve (dl d du : Fin m → 𝕜) (B X : Matrix (Fin m) (Fin r) 𝕜) : Prop :=
  tridiagonal dl d du * X = B

open Polynomial in
/-- The roots of `∏ i, (X - u i)` are the values of `u`, each as often as it occurs. -/
theorem roots_prod_X_sub_C_univ {K : Type*} [Field K] (u : Fin n → K) :
    (∏ i, (X - C (u i))).roots = Multiset.map u Finset.univ.val := by
  rw [Polynomial.roots_prod]
  · simp
  · simp [Finset.prod_ne_zero_iff, Polynomial.X_sub_C_ne_zero]

/-- A matrix that commutes with `diag(w)` for an injective `w` is diagonal. -/
theorem isDiag_of_diagonal_mul_eq {K : Type*} [Field K] {w : Fin n → K}
    {C : Matrix (Fin n) (Fin n) K} (hw : Function.Injective w)
    (h : diagonal w * C = C * diagonal w) : C.IsDiag := fun i j hij => by
  have := congrFun (congrFun h i) j
  rw [diagonal_mul, mul_diagonal, mul_comm] at this
  by_contra h0
  exact hw.ne hij (mul_left_cancel₀ h0 this)

/-- Eigenvectors of distinct eigenvalues are fixed up to scaling: if `P` is invertible,
`M P = P diag(w)`, `M P' = P' diag(w)` and `w` is injective, then `P' = P diag(c)` for `c` the
diagonal of `P⁻¹ P'`, which commutes with `diag(w)`. -/
theorem eq_mul_diagonal {K : Type*} [Field K] {M P P' : Matrix (Fin n) (Fin n) K}
    {w : Fin n → K} (hw : Function.Injective w) (hP : IsUnit P.det) (h : M * P = P * diagonal w)
    (h' : M * P' = P' * diagonal w) : P' = P * diagonal (P⁻¹ * P').diag := by
  have hP' : P * (P⁻¹ * P') = P' := mul_nonsing_inv_cancel_left P P' hP
  have e : diagonal w * (P⁻¹ * P') = P⁻¹ * P' * diagonal w :=
    calc diagonal w * (P⁻¹ * P') = P⁻¹ * (P * diagonal w) * (P⁻¹ * P') := by
          rw [nonsing_inv_mul_cancel_left P _ hP]
      _ = P⁻¹ * (M * (P * (P⁻¹ * P'))) := by
          rw [← h]
          simp only [Matrix.mul_assoc]
      _ = P⁻¹ * P' * diagonal w := by rw [hP', h', Matrix.mul_assoc]
  rw [(isDiag_of_diagonal_mul_eq hw e).diagonal_diag, hP']

/-- Scaling the columns of `V` by `c` scales the squared norm of column `j` by `star (c j) c j`. -/
theorem conjTranspose_mul_diagonal_apply (V : Matrix (Fin m) (Fin n) 𝕜) (c : Fin n → 𝕜)
    (j : Fin n) :
    ((V * diagonal c)ᴴ * (V * diagonal c)) j j = star (c j) * c j * (Vᴴ * V) j j := by
  rw [conjTranspose_mul, diagonal_conjTranspose, Matrix.mul_assoc, ← Matrix.mul_assoc Vᴴ,
    diagonal_mul, mul_diagonal, Pi.star_apply]
  ring

open scoped InnerProductSpace in
/-- The matrix whose columns are an orthonormal basis of `EuclideanSpace 𝕜 (Fin m)` is unitary:
Mathlib's `OrthonormalBasis.toMatrix_orthonormalBasis_conjTranspose_mul_self` for the standard
basis, `EuclideanSpace.basisFun`. -/
theorem conjTranspose_mul_basis (b : OrthonormalBasis (Fin m) 𝕜 (EuclideanSpace 𝕜 (Fin m))) :
    (of fun i j => b j i)ᴴ * (of fun i j => b j i) = 1 := by
  have e : (of fun i j => b j i) = (EuclideanSpace.basisFun (Fin m) 𝕜).toBasis.toMatrix b := by
    ext i j
    simp [Module.Basis.toMatrix_apply]
  rw [e]
  exact (EuclideanSpace.basisFun (Fin m) 𝕜).toMatrix_orthonormalBasis_conjTranspose_mul_self b

open scoped InnerProductSpace in
/-- Orthonormal columns `u j`, `j ∈ s`, are the columns `j ∈ s` of a unitary matrix: Mathlib's
extension of an orthonormal family to an orthonormal basis of `EuclideanSpace 𝕜 (Fin m)`
(`Orthonormal.exists_orthonormalBasis_extension_of_card_eq`). -/
theorem exists_unitary_extension {s : Set (Fin m)} (u : Fin m → Fin m → 𝕜)
    (hu : ∀ i ∈ s, ∀ j ∈ s, star (u i) ⬝ᵥ u j = if i = j then 1 else 0) :
    ∃ Q : Matrix (Fin m) (Fin m) 𝕜, Qᴴ * Q = 1 ∧ ∀ j ∈ s, ∀ i, Q i j = u j i := by
  have key : ∀ x y : Fin m → 𝕜, ⟪WithLp.toLp 2 x, WithLp.toLp 2 y⟫_𝕜 = star x ⬝ᵥ y :=
    fun x y => by rw [EuclideanSpace.inner_toLp_toLp, dotProduct_comm]
  obtain ⟨b, hb⟩ := Orthonormal.exists_orthonormalBasis_extension_of_card_eq
    (E := EuclideanSpace 𝕜 (Fin m)) (v := fun j => WithLp.toLp 2 (u j)) (s := s)
    (by rw [finrank_euclideanSpace_fin, Fintype.card_fin]) (orthonormal_iff_ite.mpr fun i j => by
      rw [Set.domRestrict_apply, Set.domRestrict_apply, key, hu i i.2 j j.2]
      by_cases hij : i = j
      · rw [ite_eq_left (congrArg Subtype.val hij), ite_eq_left hij]
      · rw [ite_eq_right fun e => hij (Subtype.ext e), ite_eq_right hij])
  exact ⟨of fun i j => b j i, conjTranspose_mul_basis b, fun j hj i => by simp [hb j hj]⟩

/-- `A = Q (Qᴴ A Q) Qᴴ` for a unitary `Q`. -/
theorem eq_mul_mul_conjTranspose {Q : Matrix (Fin n) (Fin n) 𝕜} (hQ : Qᴴ * Q = 1)
    (A : Matrix (Fin n) (Fin n) 𝕜) : A = Q * (Qᴴ * A * Q) * Qᴴ := by
  have hQ' := mul_eq_one_comm.mp hQ
  simp only [← Matrix.mul_assoc]
  rw [hQ', Matrix.one_mul, Matrix.mul_assoc, hQ', Matrix.mul_one]

open scoped ComplexOrder in
/-- A nonzero vector has a multiple of unit norm, `star (c v) ⬝ᵥ c v = 1` for a nonzero `c`. -/
theorem exists_unit_smul {v : Fin m → 𝕜} (hv : v ≠ 0) :
    ∃ c : 𝕜, c ≠ 0 ∧ star (c • v) ⬝ᵥ (c • v) = 1 := by
  obtain ⟨s, hs, hsv⟩ := RCLike.pos_iff_exists_ofReal.mp (dotProduct_star_self_pos_iff.mpr hv)
  have ht : ((Real.sqrt s : ℝ) : 𝕜) ≠ 0 := by exact_mod_cast (Real.sqrt_pos.mpr hs).ne'
  refine ⟨((Real.sqrt s : ℝ) : 𝕜)⁻¹, inv_ne_zero ht, ?_⟩
  rw [star_smul, smul_dotProduct, dotProduct_smul, ← hsv, smul_eq_mul, smul_eq_mul, star_inv₀,
    RCLike.star_def, RCLike.conj_ofReal, ← mul_assoc, ← mul_inv, ← RCLike.ofReal_mul,
    Real.mul_self_sqrt hs.le]
  exact inv_mul_cancel₀ (by exact_mod_cast hs.ne')

/-- `diag(1, M)`: the matrix of size `n + 1` with `1` at `(0, 0)`, `M` on the other rows and
columns, and zeros elsewhere. -/
def blockOne [Semiring α] :
    Matrix (Fin n) (Fin n) α →* Matrix (Fin (n + 1)) (Fin (n + 1)) α where
  toFun M := of (vecCons (vecCons 1 0) fun i => vecCons 0 (M i))
  map_one' := by
    ext i j
    cases i using Fin.cases with
    | zero =>
      cases j using Fin.cases with
      | zero => simp
      | succ j => simp [one_apply_ne (Fin.succ_ne_zero j).symm]
    | succ i =>
      cases j using Fin.cases with
      | zero => simp [one_apply_ne (Fin.succ_ne_zero i)]
      | succ j => simp [Matrix.one_apply]
  map_mul' M N := by
    ext i j
    cases i using Fin.cases <;> cases j using Fin.cases <;>
      simp [Matrix.mul_apply, Fin.sum_univ_succ]

/-- `diag(1, M)ᴴ = diag(1, Mᴴ)`. -/
theorem conjTranspose_blockOne (M : Matrix (Fin n) (Fin n) 𝕜) :
    (blockOne M)ᴴ = blockOne Mᴴ := by
  ext i j
  cases i using Fin.cases <;> cases j using Fin.cases <;> simp [blockOne]

/-- A reflector whose vector starts with a zero is `diag(1, H)` for the reflector `H` of the
vector's other entries. -/
theorem reflector_cons_zero (v : Fin n → 𝕜) (τ : 𝕜) :
    reflector (vecCons 0 v) τ = blockOne (reflector v τ) := by
  ext i j
  simp only [reflector, Matrix.sub_apply, Matrix.smul_apply, vecMulVec_apply, Pi.star_apply]
  cases i using Fin.cases with
  | zero =>
    cases j using Fin.cases with
    | zero => simp [blockOne]
    | succ j => simp [blockOne, one_apply_ne (Fin.succ_ne_zero j).symm]
  | succ i =>
    cases j using Fin.cases with
    | zero => simp [blockOne, one_apply_ne (Fin.succ_ne_zero i)]
    | succ j => simp [blockOne, Matrix.one_apply, vecMulVec_apply]

/-- At offset `s + 1` every Householder vector of `a` is zero at row zero, so the reflectors of
`a` are `diag(1, Q)` for those of `a` without its first row at offset `s`. -/
theorem householderQ_succ (a : Matrix (Fin (m + 1)) (Fin n) 𝕜) (s : ℕ) (τ : Fin k → 𝕜)
    (h : k ≤ n) :
    householderQ a (s + 1) τ h = blockOne (householderQ (a.submatrix Fin.succ id) s τ h) := by
  unfold householderQ
  rw [map_list_prod, List.map_ofFn]
  congr 2
  funext i
  rw [Function.comp_apply, ← reflector_cons_zero]
  congr 1
  ext r
  cases r using Fin.cases with
  | zero =>
    simp only [householderVec, Fin.val_zero, cons_val_zero]
    split_ifs <;> first | rfl | (exfalso; omega)
  | succ r =>
    simp only [householderVec, cons_val_succ, Fin.val_succ, transpose_apply, submatrix_apply, id]
    split_ifs <;> first | rfl | (exfalso; omega)

/-- The reflectors of `a` and `(τ₀, τ)` are the reflector of `τ₀` and of `a`'s column zero, times
`diag(1, Q)` for the reflectors of `a`'s lower right block and `τ` at the same offset `s`. -/
theorem householderQ_cons (a : Matrix (Fin (m + 1)) (Fin (n + 1)) 𝕜) (s : ℕ) (τ₀ : 𝕜)
    (τ : Fin k → 𝕜) (h : k + 1 ≤ n + 1) :
    householderQ a s (vecCons τ₀ τ) h = reflector (householderVec (aᵀ 0) s) τ₀ *
      blockOne (householderQ (a.submatrix Fin.succ Fin.succ) s τ (Nat.le_of_succ_le_succ h)) := by
  unfold householderQ
  rw [List.ofFn_succ, List.prod_cons, map_list_prod, List.map_ofFn]
  congr 1
  · simp
  · congr 2
    funext i
    rw [Function.comp_apply, ← reflector_cons_zero, cons_val_succ]
    congr 1
    ext r
    cases r using Fin.cases with
    | zero =>
      simp only [householderVec, cons_val_zero]
      split_ifs with h1
      all_goals first | rfl | (exfalso; simp only [Fin.val_zero, Fin.val_succ] at h1; omega)
    | succ r =>
      simp only [householderVec, cons_val_succ, Fin.val_succ, transpose_apply, submatrix_apply]
      split_ifs <;> first | rfl | (exfalso; omega)

/-- A unitary `Q₂` of size `n` gives the unitary `blockOne Q₂` of size `n + 1`, which acts on the
entries other than the first as `Q₂` does: on a matrix `M`, in the first column and lower right
block of `(blockOne Q₂)ᴴ M blockOne Q₂`. -/
theorem unitary_blockOne {Q₂ : Matrix (Fin n) (Fin n) 𝕜} (h : Q₂ᴴ * Q₂ = 1) :
    (blockOne Q₂)ᴴ * blockOne Q₂ = 1 ∧
      (∀ (M : Matrix (Fin (n + 1)) (Fin (n + 1)) 𝕜) (i : Fin n),
        ((blockOne Q₂)ᴴ * M * blockOne Q₂) i.succ 0 = (Q₂ᴴ *ᵥ fun r => M r.succ 0) i) ∧
      ∀ (M : Matrix (Fin (n + 1)) (Fin (n + 1)) 𝕜) (i j : Fin n),
        ((blockOne Q₂)ᴴ * M * blockOne Q₂) i.succ j.succ =
          (Q₂ᴴ * M.submatrix Fin.succ Fin.succ * Q₂) i j := by
  have hU : (blockOne Q₂)ᴴ * blockOne Q₂ = 1 := by
    rw [conjTranspose_blockOne, ← map_mul, h, map_one]
  obtain ⟨h00, h0s, hs0, hss⟩ : blockOne Q₂ 0 0 = 1 ∧ (∀ j : Fin n, blockOne Q₂ 0 j.succ = 0) ∧
      (∀ i : Fin n, blockOne Q₂ i.succ 0 = 0) ∧ ∀ i j : Fin n, blockOne Q₂ i.succ j.succ = Q₂ i j :=
    ⟨by simp [blockOne], fun j => by simp [blockOne], fun i => by simp [blockOne],
      fun i j => by simp [blockOne]⟩
  generalize blockOne Q₂ = Q at hU h00 h0s hs0 hss ⊢
  exact ⟨hU,
    fun M i => by
      simp [Matrix.mul_apply, mulVec, dotProduct, Fin.sum_univ_succ, h00, h0s, hs0, hss],
    fun M i j => by simp [Matrix.mul_apply, Fin.sum_univ_succ, h0s, hss]⟩

open InnerProductSpace in
open scoped InnerProductSpace in
/-- Every matrix has a complete QR factorization, of any rank. `Q`'s columns are Mathlib's
Gram-Schmidt orthonormal basis (`gramSchmidtOrthonormalBasis`) built from the columns `f i` of `A`
padded with zero vectors to `m`, and `R = Qᴴ A`. Column `j` of `Q` is orthogonal to the `f i` with
`i < j` (`gramSchmidtOrthonormalBasis_inv_triangular`), so `R` is upper triangular, and `R i i` is
`‖v‖⁻¹ ‖v‖ ^ 2` for the `i`-th Gram-Schmidt vector `v`, or zero, so real. -/
theorem exists_isQR (A : Matrix (Fin m) (Fin n) 𝕜) :
    ∃ (Q : Matrix (Fin m) (Fin m) 𝕜) (R : Matrix (Fin m) (Fin n) 𝕜), IsQR A Q R := by
  obtain ⟨f, hf⟩ : ∃ f : Fin m → EuclideanSpace 𝕜 (Fin m),
      ∀ i, f i = WithLp.toLp 2 fun r => if h : i.val < n then A r ⟨i, h⟩ else 0 :=
    ⟨_, fun _ => rfl⟩
  have hc : Module.finrank 𝕜 (EuclideanSpace 𝕜 (Fin m)) = Fintype.card (Fin m) := by
    rw [finrank_euclideanSpace_fin, Fintype.card_fin]
  obtain ⟨b, hb⟩ : ∃ b, b = gramSchmidtOrthonormalBasis hc f := ⟨_, rfl⟩
  have hR : ∀ i (j : Fin n) (hj : j.val < m),
      ((of fun r j => b j r)ᴴ * A) i j = ⟪b i, f ⟨j, hj⟩⟫_𝕜 := fun i j hj => by
    rw [hf, EuclideanSpace.inner_eq_star_dotProduct, Matrix.mul_apply, dotProduct]
    exact Finset.sum_congr rfl fun r _ => by simp [mul_comm]
  have hgs : ∀ i, ⟪gramSchmidt 𝕜 f i, f i⟫_𝕜 = ⟪gramSchmidt 𝕜 f i, gramSchmidt 𝕜 f i⟫_𝕜 :=
    fun i => by
      conv_lhs => rw [gramSchmidt_def'' 𝕜 f i]
      rw [inner_add_right, inner_sum, Finset.sum_eq_zero fun k hk => by
        rw [inner_smul_right, gramSchmidt_orthogonal 𝕜 f (Finset.mem_Iio.mp hk).ne', mul_zero],
        add_zero]
  have hd : ∀ i, IsSelfAdjoint ⟪b i, f i⟫_𝕜 := fun i => by
    rw [hb]
    by_cases h0 : gramSchmidtNormed 𝕜 f i = 0
    · rw [inner_gramSchmidtOrthonormalBasis_eq_zero hc h0]
      exact IsSelfAdjoint.zero 𝕜
    · rw [gramSchmidtOrthonormalBasis_apply hc h0, gramSchmidtNormed, inner_smul_left, hgs i,
        inner_self_eq_norm_sq_to_K, ← RCLike.ofReal_inv, RCLike.conj_ofReal, ← RCLike.ofReal_pow,
        ← RCLike.ofReal_mul, isSelfAdjoint_iff, RCLike.star_def, RCLike.conj_ofReal]
  refine ⟨of fun r j => b j r, (of fun r j => b j r)ᴴ * A, conjTranspose_mul_basis b,
    fun i j hij => ?_, fun i => ?_, ?_⟩
  · have := i.isLt
    rw [hR i j (by omega), hb]
    exact gramSchmidtOrthonormalBasis_inv_triangular hc f
      (Fin.lt_def.mpr (show j.val < i.val by omega))
  · rw [rectDiag, hR _ (Fin.castLE (min_le_right m n) i) (lt_of_lt_of_le i.2 (min_le_left m n))]
    exact hd _
  · rw [← Matrix.mul_assoc, mul_eq_one_comm.mp (conjTranspose_mul_basis b), Matrix.one_mul]

open scoped ComplexOrder in
/-- Every vector `x` has a unitary reflector `H = 1 - τ v vᴴ` with `v 0 = 1` and `Hᴴ x = β e₀` for
a real `β`, by an algebraic Householder construction: `τ = 0` and `β = 0` for `x = 0`, and
otherwise `β = ‖x‖`, or `β = -‖x‖` when `x 0 = ‖x‖`, so that `β ≠ x 0`,
`v = (x - β e₀) / (x 0 - β)` and `τ = (β - x 0) / β`. For `x = e₀` it gives `β = -1` and
`τ = 2`. It is an existence proof, not LAPACK's `larfg`: LAPACK 3.12.0's zlarfg.f takes
`BETA = -SIGN( DLAPY3( ALPHR, ALPHI, XNORM ), ALPHR )` (dlarfg.f, `DLAPY2( ALPHA, XNORM )`) and
says "If the elements of x are all zero and alpha is real, then tau = 0", choices this construction
does not follow. -/
theorem exists_householder (x : Fin (m + 1) → 𝕜) :
    ∃ (u : Fin m → 𝕜) (τ : 𝕜) (β : ℝ),
      star τ + τ = star τ * τ * (star (vecCons 1 u) ⬝ᵥ vecCons 1 u) ∧
        (reflector (vecCons 1 u) τ)ᴴ *ᵥ x = vecCons (β : 𝕜) 0 := by
  by_cases hx : x = 0
  · refine ⟨0, 0, 0, by simp, ?_⟩
    rw [reflector_zero, conjTranspose_one, one_mulVec, hx]
    ext i
    cases i using Fin.cases <;> simp
  obtain ⟨t, ht, htx⟩ := RCLike.pos_iff_exists_ofReal.mp (dotProduct_star_self_pos_iff.mpr hx)
  have hsβ : ∀ β : ℝ, star (β : 𝕜) = β := fun β => RCLike.conj_ofReal β
  obtain ⟨S, hS⟩ : ∃ S, S = ∑ r : Fin m, star (x r.succ) * x r.succ := ⟨_, rfl⟩
  obtain ⟨β, hβ, hβ2, hαβ⟩ : ∃ β : ℝ, (β : 𝕜) ≠ 0 ∧ (β : 𝕜) ^ 2 = star (x 0) * x 0 + S ∧
      x 0 - β ≠ 0 := by
    have hν : ((Real.sqrt t : ℝ) : 𝕜) ^ 2 = star (x 0) * x 0 + S := by
      rw [← RCLike.ofReal_pow, Real.sq_sqrt ht.le, htx, dotProduct, Fin.sum_univ_succ, hS]
      rfl
    have hν0 : ((Real.sqrt t : ℝ) : 𝕜) ≠ 0 := by exact_mod_cast (Real.sqrt_pos.mpr ht).ne'
    by_cases h : x 0 = (Real.sqrt t : ℝ)
    · refine ⟨-Real.sqrt t, by exact_mod_cast neg_ne_zero.mpr (Real.sqrt_pos.mpr ht).ne', by
        push_cast; rw [neg_sq]; exact hν, ?_⟩
      rw [h]
      push_cast
      rw [sub_neg_eq_add, ← two_mul]
      exact mul_ne_zero two_ne_zero hν0
    · exact ⟨Real.sqrt t, hν0, hν, sub_ne_zero.mpr h⟩
  have hαβ' : star (x 0) - β ≠ 0 := by
    have := star_ne_zero.mpr hαβ
    rwa [star_sub, hsβ] at this
  have hS2 : S = (β : 𝕜) ^ 2 - star (x 0) * x 0 := by rw [hβ2]; ring
  obtain ⟨v, hv⟩ : ∃ v : Fin (m + 1) → 𝕜, v = vecCons 1 fun r => x r.succ / (x 0 - β) :=
    ⟨_, rfl⟩
  have hv0 : v 0 = 1 := by rw [hv, cons_val_zero]
  have hvs : ∀ r : Fin m, v r.succ = x r.succ / (x 0 - β) := fun r => by rw [hv, cons_val_succ]
  have hvv : star v ⬝ᵥ v = 1 + S / ((star (x 0) - β) * (x 0 - β)) := by
    rw [hS, dotProduct, Fin.sum_univ_succ, Finset.sum_div]
    simp only [Pi.star_apply, hv0, hvs, star_one, one_mul, star_div₀, star_sub, hsβ]
    congr 1
    exact Finset.sum_congr rfl fun r _ => div_mul_div_comm _ _ _ _
  have hd : star v ⬝ᵥ x = x 0 + S / (star (x 0) - β) := by
    rw [hS, dotProduct, Fin.sum_univ_succ, Finset.sum_div]
    simp only [Pi.star_apply, hv0, hvs, star_one, one_mul, star_div₀, star_sub, hsβ]
    congr 1
    exact Finset.sum_congr rfl fun r _ => div_mul_eq_mul_div _ _ _
  have hkey : star ((β - x 0) / β) * (star v ⬝ᵥ x) = x 0 - β := by
    rw [hd, hS2, star_div₀, star_sub, hsβ]
    field_simp
    ring
  refine ⟨fun r => x r.succ / (x 0 - β), (β - x 0) / β, β, ?_, ?_⟩ <;> rw [← hv]
  · rw [hvv, hS2, star_div₀, star_sub, hsβ]
    field_simp
    ring
  · rw [reflector, conjTranspose_sub, conjTranspose_one, conjTranspose_smul,
      conjTranspose_vecMulVec, star_star, sub_mulVec, one_mulVec, smul_mulVec, vecMulVec_mulVec]
    ext i
    cases i using Fin.cases with
    | zero =>
      simp only [Pi.sub_apply, Pi.smul_apply, op_smul_eq_mul, smul_eq_mul, hv0, one_mul,
        cons_val_zero]
      rw [hkey]
      ring
    | succ r =>
      simp only [Pi.sub_apply, Pi.smul_apply, op_smul_eq_mul, smul_eq_mul, hvs, cons_val_succ,
        Pi.zero_apply]
      rw [mul_left_comm, hkey, div_mul_cancel₀ _ hαβ, sub_self]

/-- Every matrix has a `geqrf` factorization. By induction on the number of columns: a reflector
`H = 1 - τ₀ v vᴴ` with `v 0 = 1` and `Hᴴ` taking the first column of `A` to `β` times the first
unit vector, `β` real (`exists_householder`), and a factorization `(a', τ')` of the lower right
block of `B = Hᴴ A` give `a` with first column `β` above `v`'s other entries, first row that of
`B`, and `a'` below and to the right of it, and `τ = (τ₀, τ')`. The reflectors of `a` and `τ`
multiply to `H diag(1, Q')` for those `Q'` of `a'` and `τ'` (`householderQ_cons`). -/
theorem exists_isGeqrf (A : Matrix (Fin m) (Fin n) 𝕜) :
    ∃ (a : Matrix (Fin m) (Fin n) 𝕜) (τ : Fin (min m n) → 𝕜), IsGeqrf A a τ := by
  -- Generalize the rows and the reflector count `k = min m n`, which the induction shrinks.
  suffices h : ∀ {m} (A : Matrix (Fin m) (Fin n) 𝕜) k (_ : k = min m n) (h : k ≤ n),
      ∃ (a : Matrix (Fin m) (Fin n) 𝕜) (τ : Fin k → 𝕜),
        IsQR A (householderQ a 0 τ h) (upperPart 0 a) by exact h A _ rfl (min_le_right m n)
  clear A
  induction n with
  | zero =>
    -- Without columns, or below without rows, there is no reflector and `A` is upper triangular.
    intro m A k hk h
    obtain rfl : k = 0 := hk.trans (Nat.min_zero m)
    refine ⟨A, Fin.elim0, ?_⟩
    rw [householderQ, List.ofFn_zero, List.prod_nil, upperPart_eq_self fun _ j => j.elim0]
    exact isQR_one (fun _ j => j.elim0) fun i => absurd i.isLt (by simp)
  | succ n ih =>
    intro m A k hk h
    cases m with
    | zero =>
      obtain rfl : k = 0 := hk.trans (Nat.zero_min _)
      refine ⟨A, Fin.elim0, ?_⟩
      rw [householderQ, List.ofFn_zero, List.prod_nil, upperPart_eq_self fun i _ => i.elim0]
      exact isQR_one (fun i _ => i.elim0) fun i => absurd i.isLt (by simp)
    | succ m =>
      obtain rfl : k = min m n + 1 := hk.trans (Nat.succ_min_succ m n)
      -- Reflect the first column to `β e₀`, then factor the lower right block of `Hᴴ A`.
      obtain ⟨u, τ₀, β, hτ₀, hβ⟩ := exists_householder fun r => A r 0
      have hH := conjTranspose_reflector_mul_self hτ₀
      obtain ⟨B, hB⟩ : ∃ B, B = (reflector (vecCons 1 u) τ₀)ᴴ * A := ⟨_, rfl⟩
      have hB0 : ∀ r, B r 0 = vecCons (β : 𝕜) 0 r := fun r => by
        rw [hB, ← hβ]
        rfl
      obtain ⟨a', τ', hQ', -, hd', hA'⟩ :=
        ih (B.submatrix Fin.succ Fin.succ) _ rfl (min_le_right m n)
      obtain ⟨a, h00, h0s, hs0, hss⟩ : ∃ a : Matrix (Fin (m + 1)) (Fin (n + 1)) 𝕜,
          a 0 0 = β ∧ (∀ j : Fin n, a 0 j.succ = B 0 j.succ) ∧ (∀ i : Fin m, a i.succ 0 = u i) ∧
            ∀ i j, a i.succ j.succ = a' i j :=
        ⟨of (vecCons (vecCons (β : 𝕜) fun j => B 0 j.succ) fun i => vecCons (u i) (a' i)),
          by simp, fun j => by simp, fun i => by simp, fun i j => by simp⟩
      have ha' : a.submatrix Fin.succ Fin.succ = a' := by
        ext i j
        exact hss i j
      have hQ : householderQ a 0 (vecCons τ₀ τ') h =
          reflector (vecCons 1 u) τ₀ * blockOne (householderQ a' 0 τ' (min_le_right m n)) := by
        rw [householderQ_cons, ha']
        congr 2
        ext r
        cases r using Fin.cases with
        | zero => simp [householderVec]
        | succ r => simp [householderVec, hs0]
      -- Orthonormal columns from both factors, a real diagonal from `β` and `a'`, and
      -- `Hᴴ A = diag(1, Q') R`.
      refine ⟨a, vecCons τ₀ τ', ?_⟩
      rw [hQ]
      refine ⟨?_, upperBand_upperPart 0 a, fun i => ?_, ?_⟩
      · rw [conjTranspose_mul, Matrix.mul_assoc, ← Matrix.mul_assoc (reflector (vecCons 1 u) τ₀)ᴴ,
          hH, Matrix.one_mul, conjTranspose_blockOne, ← map_mul, hQ', map_one]
      · obtain ⟨v, hv⟩ := i
        obtain ⟨hv1, hv2⟩ := Nat.lt_min.mp hv
        cases v with
        | zero =>
          have e1 : (Fin.castLE (min_le_left _ _) ⟨0, hv⟩ : Fin (m + 1)) = 0 := rfl
          have e2 : (Fin.castLE (min_le_right _ _) ⟨0, hv⟩ : Fin (n + 1)) = 0 := rfl
          rw [rectDiag, e1, e2]
          simpa [upperPart, h00] using isSelfAdjoint_iff.mpr (RCLike.conj_ofReal (K := 𝕜) β)
        | succ v =>
          have e1 : (Fin.castLE (min_le_left _ _) ⟨v + 1, hv⟩ : Fin (m + 1)) =
              Fin.succ ⟨v, by omega⟩ := rfl
          have e2 : (Fin.castLE (min_le_right _ _) ⟨v + 1, hv⟩ : Fin (n + 1)) =
              Fin.succ ⟨v, by omega⟩ := rfl
          have := hd' ⟨v, Nat.lt_min.mpr ⟨by omega, by omega⟩⟩
          rw [rectDiag, e1, e2, upperPart, of_apply, hss]
          simpa [rectDiag, upperPart] using this
      · have hBR : B = blockOne (householderQ a' 0 τ' (min_le_right m n)) * upperPart 0 a := by
          ext i j
          rw [Matrix.mul_apply, Fin.sum_univ_succ]
          cases i using Fin.cases with
          | zero =>
            cases j using Fin.cases with
            | zero => simp [blockOne, upperPart, h00, hB0]
            | succ j => simp [blockOne, upperPart, h0s]
          | succ i =>
            cases j using Fin.cases with
            | zero => simp [blockOne, upperPart, hB0]
            | succ j =>
              have e := congrFun (congrFun hA' i) j
              simp only [submatrix_apply, Matrix.mul_apply] at e
              simp [blockOne, upperPart, hss, e]
        rw [Matrix.mul_assoc, ← hBR, hB, ← Matrix.mul_assoc, mul_eq_one_comm.mp hH,
          Matrix.one_mul]

/-- A matrix with orthonormal columns keeps the sum of the squared moduli of a vector's
entries. -/
theorem sum_norm_sq_mulVec {U : Matrix (Fin m) (Fin k) 𝕜} (hU : Uᴴ * U = 1) (x : Fin k → 𝕜) :
    ∑ r, ‖(U *ᵥ x) r‖ ^ 2 = ∑ r, ‖x r‖ ^ 2 := by
  have h : ∀ {p} (y : Fin p → 𝕜), ((∑ r, ‖y r‖ ^ 2 : ℝ) : 𝕜) = star y ⬝ᵥ y := fun y => by
    rw [dotProduct]
    push_cast
    exact Finset.sum_congr rfl fun r _ => (RCLike.conj_mul (y r)).symm
  apply (RCLike.ofReal_injective (K := 𝕜))
  rw [h, h, star_mulVec, ← dotProduct_mulVec, mulVec_mulVec, hU, one_mulVec]

/-- Every matrix has a column-pivoted QR factorization, with `Q` square. By induction on the
number of columns, moving first a column `j₀` of `A` of largest norm: a reflector `H` with `Hᴴ`
taking column `j₀` to `β` times the first unit vector, `β` real (`exists_householder`), and a
column-pivoted factorization `(P', Q', R')` of the lower right block of `B = Hᴴ A'`, `A'` being
`A` with columns `0` and `j₀` swapped, give `P` (`P 0 = j₀`, and `P` on the other columns is `P'`
followed by the swap, `Equiv.Perm.decomposeFin`), `Q = H diag(1, Q')` and `R` with first row `β`
followed by `B`'s row zero in `P'`'s order, and `R'` below and to the right of it. `|β|` is the
norm of column `j₀`, and each entry of `R'` has modulus at most the norm of a column of `B`'s
lower right block, which is at most that of a column of `B`, so of `A'` (`sum_norm_sq_mulVec`),
and so at most `|β|`. -/
theorem exists_isPivotedQR (A : Matrix (Fin m) (Fin n) 𝕜) :
    ∃ (P : Equiv.Perm (Fin n)) (Q : Matrix (Fin m) (Fin m) 𝕜) (R : Matrix (Fin m) (Fin n) 𝕜),
      IsPivotedQR A P Q R := by
  -- With no columns or no rows the identity factors `A`.
  induction n generalizing m with
  | zero =>
    exact ⟨1, 1, A, isPivotedQR_one (fun _ j => j.elim0) (fun i => absurd i.isLt (by simp))
      (by intro i; exact absurd i.isLt (by simp))⟩
  | succ n ih =>
    cases m with
    | zero =>
      exact ⟨1, 1, A, isPivotedQR_one (fun i _ => i.elim0) (fun i => absurd i.isLt (by simp))
        (by intro i; exact absurd i.isLt (by simp))⟩
    | succ m =>
      -- Pivot on a column `j₀` of largest norm, which `Hᴴ` takes to `β` times `e₀`.
      obtain ⟨j₀, -, hj₀⟩ := Finset.exists_max_image Finset.univ
        (fun j => ∑ r, ‖A r j‖ ^ 2) Finset.univ_nonempty
      obtain ⟨u, τ₀, β, hτ₀, hβ⟩ := exists_householder fun r => A r j₀
      have hH := conjTranspose_reflector_mul_self hτ₀
      generalize reflector (vecCons 1 u) τ₀ = H at hH hβ
      have hH' : (Hᴴ)ᴴ * Hᴴ = 1 := by
        rw [conjTranspose_conjTranspose]
        exact mul_eq_one_comm.mp hH
      obtain ⟨B, hB⟩ : ∃ B, B = Hᴴ * A.submatrix id (Equiv.swap 0 j₀) := ⟨_, rfl⟩
      have hB0 : ∀ r, B r 0 = vecCons (β : 𝕜) 0 r := fun r => by
        rw [hB, ← hβ]
        simp only [Matrix.mul_apply, mulVec, dotProduct, submatrix_apply, id_eq,
          Equiv.swap_apply_left]
      -- `Hᴴ` keeps column norms, so `|β|` is the norm of column `j₀`.
      have hBcol : ∀ c, ∑ r, ‖B r c‖ ^ 2 = ∑ r, ‖A r (Equiv.swap 0 j₀ c)‖ ^ 2 := fun c => by
        rw [hB]
        exact sum_norm_sq_mulVec hH' fun r => A r (Equiv.swap 0 j₀ c)
      have hβA : ‖(β : 𝕜)‖ ^ 2 = ∑ r, ‖A r j₀‖ ^ 2 := by
        rw [← Equiv.swap_apply_left 0 j₀, ← hBcol, Fin.sum_univ_succ]
        simp [hB0]
      -- Factor the lower right block of `B`.
      obtain ⟨P', Q', R', ⟨hQ', hU', hd', hA'⟩, hanti'⟩ := ih (B.submatrix Fin.succ Fin.succ)
      have hR' : Q'ᴴ * (B.submatrix Fin.succ Fin.succ).submatrix id P' = R' := by
        rw [hA', ← Matrix.mul_assoc, hQ', Matrix.one_mul]
      have hQ'' : (Q'ᴴ)ᴴ * Q'ᴴ = 1 := by
        rw [conjTranspose_conjTranspose]
        exact mul_eq_one_comm.mp hQ'
      -- Every entry of `R'` has modulus at most `|β|`.
      have hcol : ∀ i b, ‖R' i b‖ ^ 2 ≤ ‖(β : 𝕜)‖ ^ 2 := fun i b => by
        calc ‖R' i b‖ ^ 2 ≤ ∑ r, ‖R' r b‖ ^ 2 :=
              Finset.single_le_sum (f := fun r => ‖R' r b‖ ^ 2) (fun _ _ => by positivity)
                (Finset.mem_univ i)
          _ = ∑ r : Fin m, ‖B r.succ (P' b).succ‖ ^ 2 := by
              rw [← hR']
              exact sum_norm_sq_mulVec hQ'' fun r => B r.succ (P' b).succ
          _ ≤ ∑ r, ‖B r (P' b).succ‖ ^ 2 := by
              rw [Fin.sum_univ_succ (f := fun r => ‖B r (P' b).succ‖ ^ 2)]
              exact le_add_of_nonneg_left (by positivity)
          _ ≤ ‖(β : 𝕜)‖ ^ 2 := by
              rw [hBcol, hβA]
              exact hj₀ _ (Finset.mem_univ _)
      obtain ⟨P, hP0, hPs⟩ : ∃ P : Equiv.Perm (Fin (n + 1)),
          P 0 = j₀ ∧ ∀ j, P j.succ = Equiv.swap 0 j₀ (P' j).succ :=
        ⟨Equiv.Perm.decomposeFin.symm (j₀, P'), by simp, fun j => by simp⟩
      obtain ⟨R, h00, h0s, hs0, hss⟩ : ∃ R : Matrix (Fin (m + 1)) (Fin (n + 1)) 𝕜,
          R 0 0 = β ∧ (∀ j : Fin n, R 0 j.succ = B 0 (P' j).succ) ∧
            (∀ i : Fin m, R i.succ 0 = 0) ∧ ∀ i j, R i.succ j.succ = R' i j :=
        ⟨of (vecCons (vecCons (β : 𝕜) fun j => B 0 (P' j).succ) fun i => vecCons 0 (R' i)),
          by simp, fun j => by simp, fun i => by simp, fun i j => by simp⟩
      have hAP0 : ∀ i, (Hᴴ * A.submatrix id P) i 0 = B i 0 := fun i => by
        rw [hB]
        simp only [Matrix.mul_apply, submatrix_apply, id_eq, hP0, Equiv.swap_apply_left]
      have hAPs : ∀ i j, (Hᴴ * A.submatrix id P) i j.succ = B i (P' j).succ := fun i j => by
        rw [hB]
        simp only [Matrix.mul_apply, submatrix_apply, id_eq, hPs]
      -- Past its first entry the diagonal of `R` is that of `R'`.
      have hdiag : ∀ w (hw : w + 1 < min (m + 1) (n + 1)),
          rectDiag R ⟨w + 1, hw⟩ = rectDiag R' ⟨w, by
            obtain ⟨h1, h2⟩ := Nat.lt_min.mp hw
            exact Nat.lt_min.mpr ⟨by omega, by omega⟩⟩ := fun w hw => by
        obtain ⟨h1, h2⟩ := Nat.lt_min.mp hw
        have e1 : (Fin.castLE (min_le_left _ _) ⟨w + 1, hw⟩ : Fin (m + 1)) =
            Fin.succ ⟨w, by omega⟩ := rfl
        have e2 : (Fin.castLE (min_le_right _ _) ⟨w + 1, hw⟩ : Fin (n + 1)) =
            Fin.succ ⟨w, by omega⟩ := rfl
        rw [rectDiag, e1, e2, hss]
        rfl
      -- `Q` is unitary, `R` is upper triangular with a real diagonal, `A P = Q R`, and the
      -- diagonal moduli do not increase.
      refine ⟨P, H * blockOne Q', R, ⟨?_, fun i j hij => ?_, fun i => ?_, ?_⟩, ?_⟩
      · rw [conjTranspose_mul, Matrix.mul_assoc, ← Matrix.mul_assoc Hᴴ, hH, Matrix.one_mul,
          conjTranspose_blockOne, ← map_mul, hQ', map_one]
      · cases i using Fin.cases with
        | zero => exact absurd hij (by simp)
        | succ i =>
          cases j using Fin.cases with
          | zero => exact hs0 i
          | succ j =>
            rw [hss]
            exact hU' i j (by simp only [Fin.val_succ] at hij; omega)
      · obtain ⟨v, hv⟩ := i
        cases v with
        | zero =>
          have e1 : (Fin.castLE (min_le_left _ _) ⟨0, hv⟩ : Fin (m + 1)) = 0 := rfl
          have e2 : (Fin.castLE (min_le_right _ _) ⟨0, hv⟩ : Fin (n + 1)) = 0 := rfl
          rw [rectDiag, e1, e2, h00]
          exact isSelfAdjoint_iff.mpr (RCLike.conj_ofReal β)
        | succ v =>
          rw [hdiag v hv]
          exact hd' _
      · have hBR : Hᴴ * A.submatrix id P = blockOne Q' * R := by
          ext i j
          cases j using Fin.cases with
          | zero =>
            rw [hAP0, Matrix.mul_apply, Fin.sum_univ_succ]
            cases i using Fin.cases with
            | zero => simp [blockOne, h00, hB0]
            | succ i => simp [blockOne, hs0, hB0]
          | succ j =>
            rw [hAPs, Matrix.mul_apply, Fin.sum_univ_succ]
            cases i using Fin.cases with
            | zero => simp [blockOne, h0s]
            | succ i =>
              have e := congrFun (congrFun hA' i) j
              simp only [submatrix_apply, id_eq, Matrix.mul_apply] at e
              simp [blockOne, hss, e]
        rw [Matrix.mul_assoc, ← hBR, ← Matrix.mul_assoc, mul_eq_one_comm.mp hH, Matrix.one_mul]
      · have hle : ∀ i, ‖rectDiag R' i‖ ≤ ‖(β : 𝕜)‖ := fun i => by
          have := hcol (Fin.castLE (min_le_left _ _) i) (Fin.castLE (min_le_right _ _) i)
          rw [rectDiag]
          nlinarith [norm_nonneg (R' (Fin.castLE (min_le_left _ _) i)
            (Fin.castLE (min_le_right _ _) i)), norm_nonneg (β : 𝕜)]
        rintro ⟨vi, hi⟩ ⟨vj, hj⟩ hij
        rw [Fin.mk_le_mk] at hij
        cases vj with
        | zero =>
          obtain rfl : vi = 0 := by omega
          exact le_rfl
        | succ w =>
          show ‖rectDiag R ⟨w + 1, hj⟩‖ ≤ ‖rectDiag R ⟨vi, hi⟩‖
          rw [hdiag w hj]
          cases vi with
          | zero =>
            have e1 : (Fin.castLE (min_le_left _ _) ⟨0, hi⟩ : Fin (m + 1)) = 0 := rfl
            have e2 : (Fin.castLE (min_le_right _ _) ⟨0, hi⟩ : Fin (n + 1)) = 0 := rfl
            have h0 : rectDiag R ⟨0, hi⟩ = β := by rw [rectDiag, e1, e2, h00]
            rw [h0]
            exact hle _
          | succ v =>
            rw [hdiag v hi]
            exact hanti' (Fin.mk_le_mk.mpr (by omega))

/-- Every matrix, real or complex, has a `hessenberg` output, with a real first subdiagonal. By
induction on the size: for `A` of size `n + 2`, a reflector `H = 1 - τ₀ v vᴴ` with `v 0 = 1` and
`Hᴴ` taking the column of `A` below `A 0 0` to `β` times the first unit vector, `β` real
(`exists_householder`), gives `B = diag(1, H)ᴴ A diag(1, H)` with first column `A 0 0, β, 0, …`,
and an output `(a', τ')` for the lower right block of `B`, with reflectors `Q'`, gives `a` with
first row that of `C = diag(1, Q')ᴴ B diag(1, Q')`, `β` and `v`'s other entries below it, and
`a'` below and to the right of it, and `τ = (τ₀, τ')`; the reflectors of `a` and `τ` multiply to
`diag(1, H) diag(1, Q')` (`householderQ_cons`, `householderQ_succ`). Sizes up to one take
`a = A` and no reflector. -/
theorem exists_isGehrd (A : Matrix (Fin n) (Fin n) 𝕜) :
    ∃ (a : Matrix (Fin n) (Fin n) 𝕜) (τ : Fin (n - 1) → 𝕜), IsGehrd A a τ ∧
      ∀ i j : Fin n, i.val = j.val + 1 → star (a i j) = a i j := by
  -- Generalize the reflector count `k = n - 1`, and state the reduction as `Qᴴ A Q`.
  suffices h : ∀ {n} (A : Matrix (Fin n) (Fin n) 𝕜) k (_ : k = n - 1) (h : k ≤ n),
      ∃ (a : Matrix (Fin n) (Fin n) 𝕜) (τ : Fin k → 𝕜),
        (householderQ a 1 τ h)ᴴ * householderQ a 1 τ h = 1 ∧
        (householderQ a 1 τ h)ᴴ * A * householderQ a 1 τ h = upperPart 1 a ∧
        ∀ i j : Fin n, i.val = j.val + 1 → star (a i j) = a i j by
    obtain ⟨a, τ, hQ, hH, hr⟩ := h A _ rfl (Nat.sub_le n 1)
    refine ⟨a, τ, ⟨hQ, upperBand_one_iff.mp (upperBand_upperPart 1 a), ?_⟩, hr⟩
    rw [← hH]
    exact eq_mul_mul_conjTranspose hQ A
  clear A
  intro n
  induction n with
  | zero =>
    -- Sizes 0 and 1 have no reflector and are already Hessenberg.
    intro A k hk h
    obtain rfl : k = 0 := by omega
    exact ⟨A, 0, by rw [householderQ_zero]; simp,
      by rw [householderQ_zero, upperPart_eq_self fun i => i.elim0]; simp, fun i => i.elim0⟩
  | succ n ih =>
    cases n with
    | zero =>
      intro A k hk h
      obtain rfl : k = 0 := by omega
      refine ⟨A, 0, by rw [householderQ_zero]; simp, ?_,
        fun i j hij => absurd hij (by have := i.isLt; omega)⟩
      rw [householderQ_zero, upperPart_eq_self fun i j hij => absurd hij (by have := i.isLt; omega)]
      simp
    | succ n =>
      intro A k hk h
      obtain rfl : k = n + 1 := by omega
      -- Reflect the column below `A 0 0` to `β e₀` on both sides, then reduce the lower right
      -- block of `B`.
      obtain ⟨u, τ₀, β, hτ₀, hβ⟩ := exists_householder fun r : Fin (n + 1) => A r.succ 0
      obtain ⟨hH, hH0, -⟩ := unitary_blockOne (conjTranspose_reflector_mul_self hτ₀)
      obtain ⟨B, hB⟩ : ∃ B, B = (blockOne (reflector (vecCons 1 u) τ₀))ᴴ * A *
          blockOne (reflector (vecCons 1 u) τ₀) := ⟨_, rfl⟩
      have hB0 : (fun r : Fin (n + 1) => B r.succ 0) = vecCons (β : 𝕜) 0 := by
        ext i
        rw [hB, hH0, hβ]
      obtain ⟨a', τ', hQ', hC', hr'⟩ := ih (B.submatrix Fin.succ Fin.succ) n (by omega) (by omega)
      obtain ⟨hU', hc', hs'⟩ := unitary_blockOne hQ'
      obtain ⟨C, hC⟩ : ∃ C, C = (blockOne (householderQ a' 1 τ' (by omega)))ᴴ * B *
          blockOne (householderQ a' 1 τ' (by omega)) := ⟨_, rfl⟩
      obtain ⟨a, h0, hs0, hss⟩ : ∃ a : Matrix (Fin (n + 1 + 1)) (Fin (n + 1 + 1)) 𝕜,
          (∀ j, a 0 j = C 0 j) ∧ (∀ i, a i.succ 0 = vecCons (β : 𝕜) u i) ∧
            ∀ i j, a i.succ j.succ = a' i j :=
        ⟨of (vecCons (C 0) fun i => vecCons (vecCons (β : 𝕜) u i) (a' i)),
          fun j => by simp, fun i => by simp, fun i j => by simp⟩
      have ha' : a.submatrix Fin.succ Fin.succ = a' := by
        ext i j
        exact hss i j
      have hQ : householderQ a 1 (vecCons τ₀ τ') h =
          blockOne (reflector (vecCons 1 u) τ₀) * blockOne (householderQ a' 1 τ' (by omega)) := by
        rw [householderQ_cons, ha', ← reflector_cons_zero]
        congr 2
        ext r
        cases r using Fin.cases with
        | zero => simp [householderVec]
        | succ r =>
          cases r using Fin.cases with
          | zero => simp [householderVec]
          | succ r => simp [householderVec, hs0]
      -- `Q` is unitary. `C` keeps `B`'s first column `A 0 0, β, 0, …` and its lower right block
      -- is reduced as by `a'`. The subdiagonal is `β`, then `a'`'s, all real.
      refine ⟨a, vecCons τ₀ τ', ?_, ?_, ?_⟩
      · rw [hQ, conjTranspose_mul, Matrix.mul_assoc,
          ← Matrix.mul_assoc (blockOne (reflector (vecCons 1 u) τ₀))ᴴ, hH, Matrix.one_mul, hU']
      · have hT : (householderQ a 1 (vecCons τ₀ τ') h)ᴴ * A * householderQ a 1 (vecCons τ₀ τ') h
            = C := by
          rw [hQ, hC, hB, conjTranspose_mul]
          simp only [Matrix.mul_assoc]
        rw [hT]
        ext i j
        cases i using Fin.cases with
        | zero => simp [upperPart, h0]
        | succ i =>
          cases j using Fin.cases with
          | zero =>
            rw [hC, hc', hB0, householderQ_succ a' 0 τ', conjTranspose_blockOne]
            cases i using Fin.cases with
            | zero =>
              simpa [blockOne, mulVec, dotProduct, Fin.sum_univ_succ, upperPart] using (hs0 0).symm
            | succ i => simp [blockOne, mulVec, dotProduct, Fin.sum_univ_succ, upperPart]
          | succ j =>
            rw [hC, hs', hC', ← ha']
            simp [upperPart]
      · intro i j hij
        cases i using Fin.cases with
        | zero => exact absurd hij (by simp)
        | succ i =>
          cases j using Fin.cases with
          | zero =>
            obtain rfl : i = 0 := Fin.ext (by simpa using hij)
            rw [hs0, cons_val_zero]
            exact RCLike.conj_ofReal β
          | succ j =>
            rw [hss]
            exact hr' i j (by simpa using hij)

/-- Every matrix, real or complex, has a Hessenberg decomposition: the reflectors and the upper
part of a `hessenberg` output (`exists_isGehrd`). -/
theorem exists_isHessenberg (A : Matrix (Fin n) (Fin n) 𝕜) : ∃ Q H, IsHessenberg A Q H := by
  obtain ⟨_, _, h, -⟩ := exists_isGehrd A
  exact ⟨_, _, h⟩

/-- Every Hermitian matrix has a `tridiagonal` output with `lower=True`. An output `(a, τ)` of
`hessenberg` with a real first subdiagonal (`exists_isGehrd`) has `upperPart 1 a = Qᴴ A Q`
Hermitian, so tridiagonal with a real diagonal, and `d` and `e` are the real parts of `a`'s
diagonal and first subdiagonal. -/
theorem exists_isSytrd {A : Matrix (Fin n) (Fin n) 𝕜} (hA : A.IsHermitian) :
    ∃ (a : Matrix (Fin n) (Fin n) 𝕜) (d : Fin n → ℝ) (e : Fin (n - 1) → ℝ)
      (τ : Fin (n - 1) → 𝕜), IsSytrd A a d e τ := by
  obtain ⟨a, τ, ⟨hQ, -, hA'⟩, hr⟩ := exists_isGehrd A
  obtain ⟨Q, hQd⟩ : ∃ Q, Q = householderQ a 1 τ (Nat.sub_le n 1) := ⟨_, rfl⟩
  rw [← hQd] at hQ hA'
  have hH : (upperPart 1 a).IsHermitian := by
    have e : Qᴴ * A * Q = upperPart 1 a := by
      rw [hA']
      simp only [Matrix.mul_assoc]
      rw [hQ, Matrix.mul_one, ← Matrix.mul_assoc, hQ, Matrix.one_mul]
    exact e ▸ isHermitian_conjTranspose_mul_mul Q hA
  have hre : ∀ i j : Fin n, star (a i j) = a i j → (RCLike.re (a i j) : 𝕜) = a i j :=
    fun i j h => RCLike.conj_eq_iff_re.mp h
  have hd : ∀ i, star (a i i) = a i i := fun i => by
    simpa [upperPart] using hH.apply i i
  have hT : (symTridiagonal (fun i => RCLike.re (a i i))
      (fun j : Fin (n - 1) => RCLike.re (a ⟨j.val + 1, by omega⟩ ⟨j.val, by omega⟩)) :
        Matrix (Fin n) (Fin n) 𝕜) = upperPart 1 a := by
    ext ⟨i, hi⟩ ⟨j, hj⟩
    have e := hH.apply ⟨i, hi⟩ ⟨j, hj⟩
    simp only [symTridiagonal, tridiagonal, upperPart, of_apply, Fin.mk.injEq] at e ⊢
    by_cases h1 : i = j
    · subst h1
      simpa using (hre _ _ (hd _))
    · by_cases h2 : j + 1 = i
      · subst h2
        simpa [h1] using hre _ _ (hr ⟨j + 1, hi⟩ ⟨j, hj⟩ rfl)
      · by_cases h3 : i + 1 = j
        · subst h3
          have e1 := hr ⟨i + 1, hj⟩ ⟨i, hi⟩ rfl
          simp only [show i + 1 ≤ i + 1 by omega, show i ≤ i + 1 + 1 by omega, ite_true] at e
          simp [h2, hj, ← e, e1, hre _ _ e1]
          intro h; omega
        · simp only [h1, h2, h3, ite_false]
          split_ifs at e ⊢ with h4 h5
          · omega
          · rw [← e, star_zero]
          · rfl
          · rfl
  refine ⟨a, fun i => RCLike.re (a i i),
    fun j => RCLike.re (a ⟨j.val + 1, by omega⟩ ⟨j.val, by omega⟩), τ, ⟨hQd ▸ hQ, ?_⟩,
    fun i => (hre i i (hd i)).symm, fun j => (hre _ _ (hr _ _ rfl)).symm⟩
  rw [hT, ← hQd]
  exact hA'

section Spectral

variable {A V : Matrix (Fin n) (Fin n) 𝕜} {w : Fin n → ℝ}

/-- Every Hermitian matrix has an eigendecomposition: Mathlib's spectral theorem, with its
eigenvalues sorted ascending (`Tuple.sort`) and its eigenvectors permuted alike. -/
theorem exists_isEigh (hA : A.IsHermitian) : ∃ V w, IsEigh A V w := by
  let U : Matrix (Fin n) (Fin n) 𝕜 := hA.eigenvectorUnitary
  let σ := Tuple.sort hA.eigenvalues
  have hU : Uᴴ * U = 1 := by
    rw [← star_eq_conjTranspose]
    exact Unitary.coe_star_mul_self _
  refine ⟨U.submatrix id σ, hA.eigenvalues ∘ σ, ?_, Tuple.monotone_sort _, ?_⟩
  · rw [conjTranspose_submatrix, ← submatrix_mul _ _ _ _ _ Function.bijective_id, hU,
      submatrix_one_equiv]
  · have hd : diagonal (fun i => (((hA.eigenvalues ∘ σ) i : ℝ) : 𝕜)) =
        (diagonal (RCLike.ofReal ∘ hA.eigenvalues)).submatrix σ σ := by
      rw [submatrix_diagonal_equiv]
      rfl
    rw [hd, submatrix_mul_equiv, conjTranspose_submatrix, submatrix_mul_equiv, submatrix_id_id]
    conv_lhs => rw [hA.spectral_theorem]
    rw [Unitary.conjStarAlgAut_apply]
    rfl

/-- The columns of `V` are eigenvectors of `A` with the eigenvalues `w`: `A V = V diag(w)`. -/
theorem IsEigh.mul_eq (h : IsEigh A V w) : A * V = V * diagonal (fun i => (w i : 𝕜)) := by
  rw [h.2.2, Matrix.mul_assoc, h.1, Matrix.mul_one]

/-- A matrix with an eigendecomposition `IsEigh A V w` is Hermitian. -/
theorem IsEigh.isHermitian (h : IsEigh A V w) : A.IsHermitian := by
  have hd : IsSelfAdjoint (fun i => (w i : 𝕜)) := funext fun i => by simp
  rw [h.2.2]
  exact isHermitian_mul_mul_conjTranspose V (isHermitian_diagonal_of_self_adjoint _ hd)

open Polynomial in
/-- The characteristic polynomial of `A` is `∏ i, (X - w i)`. -/
theorem IsEigh.charpoly (h : IsEigh A V w) : A.charpoly = ∏ i, (X - C (w i : 𝕜)) := by
  rw [h.2.2, charpoly_mul_comm, ← mul_assoc, h.1, one_mul, charpoly_diagonal]

open Polynomial in
/-- A monotone `w` lists the roots of `∏ i, (X - w i)` ascending. -/
theorem sort_roots_prod {w : Fin n → ℝ} (hw : Monotone w) :
    ((∏ i, (X - C (w i : 𝕜))).roots.map RCLike.re).sort (· ≤ ·) = List.ofFn w := by
  have hr : (∏ i, (X - C (w i : 𝕜))).roots = Multiset.map (fun i => (w i : 𝕜)) Finset.univ.val :=
    roots_prod_X_sub_C_univ _
  simp_rw [hr, Fin.univ_val_map, Multiset.map_coe, List.map_ofFn, Function.comp_def,
    RCLike.ofReal_re, Multiset.coe_sort]
  apply List.mergeSort_of_pairwise
  simp_rw [decide_eq_true_eq, ← List.sortedLE_iff_pairwise]
  exact hw.sortedLE_ofFn

/-- The eigenvalues are the roots of the characteristic polynomial, listed ascending. -/
theorem IsEigh.sort_roots (h : IsEigh A V w) :
    (A.charpoly.roots.map RCLike.re).sort (· ≤ ·) = List.ofFn w := by
  rw [h.charpoly]
  exact sort_roots_prod h.2.1

/-- The eigenvalues of an eigendecomposition are unique. -/
theorem IsEigh.eigenvalues_eq {V' : Matrix (Fin n) (Fin n) 𝕜} {w' : Fin n → ℝ}
    (h : IsEigh A V w) (h' : IsEigh A V' w') : w = w' :=
  List.ofFn_injective (h.sort_roots.symm.trans h'.sort_roots)

/-- With distinct eigenvalues, an eigendecomposition `(V, w)` of `A` is unique up to the phases
of its eigenvectors: `(V', w')` is one exactly when `w' = w` and `V' = V diag(d)` with
`star (d i) * d i = 1`, so `d i = ±1` over `ℝ`. This is the freedom of PyTorch's warning at
`torch.linalg.eigh`, "multiplying an eigenvector by `-1` in the real case or by `e^{i φ}, φ ∈ ℝ`
in the complex case produces another set of valid eigenvectors of the matrix". -/
theorem IsEigh.isEigh_iff (h : IsEigh A V w) (hw : StrictMono w) {V' : Matrix (Fin n) (Fin n) 𝕜}
    {w' : Fin n → ℝ} :
    IsEigh A V' w' ↔
      w' = w ∧ ∃ d : Fin n → 𝕜, (∀ i, star (d i) * d i = 1) ∧ V' = V * diagonal d := by
  constructor
  · intro h'
    have hww := h.eigenvalues_eq h'
    subst hww
    obtain ⟨c, rfl⟩ : ∃ c, V' = V * diagonal c :=
      ⟨_, eq_mul_diagonal (RCLike.ofReal_injective.comp hw.injective)
        (isUnit_det_of_left_inverse h.1) h.mul_eq h'.mul_eq⟩
    refine ⟨rfl, c, fun i => ?_, rfl⟩
    have := congrFun (congrFun h'.1 i) i
    rwa [conjTranspose_mul_diagonal_apply, h.1, one_apply_eq, mul_one] at this
  · rintro ⟨hw', d, hd, rfl⟩
    rw [hw']
    have hdd : diagonal (star d) * diagonal d = (1 : Matrix (Fin n) (Fin n) 𝕜) := by
      rw [diagonal_mul_diagonal, ← diagonal_one]
      congr 1
      funext i
      exact hd i
    have hW : diagonal d * diagonal (fun i => (w i : 𝕜)) * diagonal (star d) =
        diagonal fun i => (w i : 𝕜) := by
      rw [diagonal_mul_diagonal, diagonal_mul_diagonal]
      congr 1
      funext i
      calc d i * (w i : 𝕜) * star d i = (w i : 𝕜) * (star (d i) * d i) := by
            simp only [Pi.star_apply]
            ring
        _ = w i := by rw [hd i, mul_one]
    refine ⟨?_, h.2.1, ?_⟩
    · rw [conjTranspose_mul, diagonal_conjTranspose, Matrix.mul_assoc, ← Matrix.mul_assoc Vᴴ, h.1,
        Matrix.one_mul, hdd]
    · calc A = V * (diagonal d * diagonal (fun i => (w i : 𝕜)) * diagonal (star d)) * Vᴴ := by
            rw [hW]
            exact h.2.2
        _ = V * diagonal d * diagonal (fun i => (w i : 𝕜)) * (V * diagonal d)ᴴ := by
            rw [conjTranspose_mul, diagonal_conjTranspose]
            simp only [Matrix.mul_assoc]

/-- Over `ℂ`, an eigendecomposition `IsEigh A V w` is an `IsEig A w V`, with `w` cast to `ℂ`. -/
theorem IsEigh.isEig {A V : Matrix (Fin n) (Fin n) ℂ} (h : IsEigh A V w) :
    IsEig A (fun i => (w i : ℂ)) V :=
  ⟨by rw [h.charpoly]; rfl, fun j => by rw [h.1, one_apply_eq], h.mul_eq,
    fun _ _ _ _ => isUnit_det_of_left_inverse h.1⟩

/-- An eigendecomposition `IsEigh A V w` is the Schur decomposition `IsSchur A V (diagonal w)`, `w`
cast to `𝕜`. -/
theorem IsEigh.isSchur (h : IsEigh A V w) : IsSchur A V (diagonal fun i => (w i : 𝕜)) :=
  ⟨h.1, blockTriangular_diagonal _, h.2.2⟩

/-- A Schur decomposition `IsSchur A Q T` is the Hessenberg decomposition `IsHessenberg A Q T`: an
upper triangular `T` is upper Hessenberg. -/
theorem IsSchur.isHessenberg {Q T : Matrix (Fin n) (Fin n) 𝕜} (h : IsSchur A Q T) :
    IsHessenberg A Q T :=
  ⟨h.1, h.2.1.isUpperHessenberg, h.2.2⟩

/-- An eigendecomposition `IsEigh A V w` is the tridiagonal decomposition `IsTridiagonal A V w 0`,
with diagonal `w` and zeros beside it. -/
theorem IsEigh.isTridiagonal (h : IsEigh A V w) : IsTridiagonal A V w 0 := by
  have hT : (symTridiagonal w 0 : Matrix (Fin n) (Fin n) 𝕜) = diagonal fun i => (w i : 𝕜) := by
    ext i j
    simp [symTridiagonal, tridiagonal, diagonal_apply]
  exact ⟨h.1, by rw [hT]; exact h.2.2⟩

end Spectral

section Eig

variable {A V : Matrix (Fin n) (Fin n) ℂ} {w : Fin n → ℂ}

/-- With distinct eigenvalues the eigenvectors are linearly independent (Mathlib's
`Module.End.eigenvectors_linearIndependent'`), so `V` is invertible. -/
theorem IsEig.isUnit_det (h : IsEig A w V) (hw : Function.Injective w) : IsUnit V.det := by
  refine (isUnit_iff_isUnit_det V).mp (linearIndependent_col_iff_isUnit.mp ?_)
  refine Module.End.eigenvectors_linearIndependent' (toLin' A) w hw V.col fun j =>
    Module.End.hasEigenvector_iff.mpr ⟨?_, fun h0 => ?_⟩
  · rw [Module.End.mem_eigenspace_iff, toLin'_apply]
    funext i
    have := congrFun (congrFun h.2.2.1 i) j
    rw [Matrix.mul_apply, mul_diagonal] at this
    simp only [mulVec, dotProduct, col_apply, Pi.smul_apply, smul_eq_mul]
    rw [this, mul_comm]
  · have hz : ∀ i, V i j = 0 := fun i => by simpa using congrFun h0 i
    have := h.2.1 j
    simp [Matrix.mul_apply, hz] at this

/-- Listing the eigenvalues in another order, with the eigenvectors alike, is again an output. -/
theorem IsEig.submatrix (h : IsEig A w V) (σ : Equiv.Perm (Fin n)) :
    IsEig A (w ∘ σ) (V.submatrix id σ) := by
  refine ⟨?_, fun j => ?_, ?_, fun P d hP hAP => ?_⟩
  · rw [h.1]
    exact (Equiv.prod_comp σ fun i => Polynomial.X - Polynomial.C (w i)).symm
  · rw [conjTranspose_submatrix, ← submatrix_mul _ _ _ _ _ Function.bijective_id, submatrix_apply]
    exact h.2.1 (σ j)
  · calc A * V.submatrix id σ = (A * V).submatrix id σ := by
          ext i j
          simp [Matrix.mul_apply]
      _ = V.submatrix id σ * diagonal (w ∘ σ) := by
          rw [h.2.2.1, ← submatrix_diagonal_equiv, submatrix_mul_equiv]
  · rw [det_permute']
    exact (isUnit_iff_ne_zero.mpr (Int.cast_ne_zero.mpr (Units.ne_zero _))).mul
      (h.2.2.2 P d hP hAP)

/-- Multiplying each eigenvector by a unit phase is again an output. -/
theorem IsEig.mul_diagonal (h : IsEig A w V) {d : Fin n → ℂ} (hd : ∀ i, star (d i) * d i = 1) :
    IsEig A w (V * diagonal d) := by
  refine ⟨h.1, fun j => by rw [conjTranspose_mul_diagonal_apply, h.2.1 j, mul_one, hd j], ?_,
    fun P e hP hAP => ?_⟩
  · rw [← Matrix.mul_assoc, h.2.2.1, Matrix.mul_assoc, Matrix.mul_assoc, diagonal_mul_diagonal,
      diagonal_mul_diagonal]
    congr 2
    funext i
    exact mul_comm _ _
  · rw [det_mul, det_diagonal]
    exact (h.2.2.2 P e hP hAP).mul (isUnit_iff_ne_zero.mpr (Finset.prod_ne_zero_iff.mpr
      fun i _ hi => by simpa [hi] using hd i))

/-- With distinct eigenvalues, an output `(w, V)` of `eig` is unique up to the order of the
eigenvalues and the phases of the eigenvectors: `(w', V')` is one exactly when `w' = w ∘ σ` and
`V' = V[:, σ] diag(d)` for a permutation `σ` and `star (d i) * d i = 1`. These are the freedoms
PyTorch's documentation of `torch.linalg.eig` names: "The returned eigenvalues are not
guaranteed to be in any specific order", and "multiplying an eigenvector by ...
`e^{i φ}, φ ∈ ℝ` produces another set of valid eigenvectors of the matrix". -/
theorem IsEig.isEig_iff (h : IsEig A w V) (hw : Function.Injective w) {w' : Fin n → ℂ}
    {V' : Matrix (Fin n) (Fin n) ℂ} :
    IsEig A w' V' ↔ ∃ σ : Equiv.Perm (Fin n), w' = w ∘ σ ∧
      ∃ d : Fin n → ℂ, (∀ i, star (d i) * d i = 1) ∧ V' = V.submatrix id σ * diagonal d := by
  constructor
  · intro h'
    have hm : Multiset.map w' Finset.univ.val = Multiset.map w Finset.univ.val := by
      rw [← roots_prod_X_sub_C_univ w', ← roots_prod_X_sub_C_univ w, ← h'.1, h.1]
    have hnd : (Multiset.map w' Finset.univ.val).Nodup := by
      rw [hm]
      exact Finset.univ.nodup.map hw
    have hw' : Function.Injective w' := fun a b hab =>
      Multiset.inj_on_of_nodup_map hnd a (Finset.mem_univ_val a) b (Finset.mem_univ_val b) hab
    choose f hf using fun j => Multiset.mem_map.mp
      (hm ▸ Multiset.mem_map_of_mem w' (Finset.mem_univ_val j))
    have hfi : Function.Injective f := fun a b hab => hw' (by rw [← (hf a).2, ← (hf b).2, hab])
    obtain ⟨σ, hσ⟩ : ∃ σ : Equiv.Perm (Fin n), w' = w ∘ σ :=
      ⟨Equiv.ofBijective f (Finite.injective_iff_bijective.mp hfi), funext fun j => (hf j).2.symm⟩
    subst hσ
    have hs := h.submatrix σ
    have hi : Function.Injective (w ∘ σ) := hw.comp σ.injective
    obtain ⟨c, rfl⟩ : ∃ c, V' = V.submatrix id σ * diagonal c :=
      ⟨_, eq_mul_diagonal hi (hs.isUnit_det hi) hs.2.2.1 h'.2.2.1⟩
    refine ⟨σ, rfl, c, fun i => ?_, rfl⟩
    have := h'.2.1 i
    rwa [conjTranspose_mul_diagonal_apply, hs.2.1, mul_one] at this
  · rintro ⟨σ, rfl, d, hd, rfl⟩
    exact (h.submatrix σ).mul_diagonal hd

open Polynomial in
/-- Every complex matrix has an output of `eig`. A diagonalizable `A`, `A P = P diag(d)` for an
invertible `P`, has `w = d` and `V` the columns of `P` scaled to unit norm, invertible, and
`A.charpoly = (diagonal d).charpoly`. For any other `A`, the characteristic polynomial splits
over `ℂ`, `w` lists its roots, and column `j` of `V` is an eigenvector of `w j` scaled to unit
norm. -/
theorem exists_isEig (A : Matrix (Fin n) (Fin n) ℂ) : ∃ w V, IsEig A w V := by
  by_cases hA : ∃ (P : Matrix (Fin n) (Fin n) ℂ) (d : Fin n → ℂ), IsUnit P.det ∧
      A * P = P * diagonal d
  · obtain ⟨P, d, hP, hAP⟩ := hA
    choose c hc0 hc using fun j => exists_unit_smul (v := fun i => P i j) fun h0 =>
      not_isUnit_zero (det_eq_zero_of_column_eq_zero j (fun i => congrFun h0 i) ▸ hP)
    refine ⟨d, P * diagonal c, ?_, fun j => ?_, ?_, fun _ _ _ _ => ?_⟩
    · have e : A = P * (diagonal d * P⁻¹) := by
        rw [← Matrix.mul_assoc, ← hAP, Matrix.mul_assoc, mul_nonsing_inv _ hP, Matrix.mul_one]
      rw [e, charpoly_mul_comm, Matrix.mul_assoc, nonsing_inv_mul _ hP, Matrix.mul_one,
        charpoly_diagonal]
    · refine Eq.trans ?_ (hc j)
      rw [conjTranspose_mul_diagonal_apply, star_smul, smul_dotProduct, dotProduct_smul,
        smul_eq_mul, smul_eq_mul, ← mul_assoc]
      rfl
    · rw [← Matrix.mul_assoc, hAP, Matrix.mul_assoc, Matrix.mul_assoc, diagonal_mul_diagonal,
        diagonal_mul_diagonal]
      congr 2
      funext i
      exact mul_comm _ _
    · rw [det_mul, det_diagonal]
      exact hP.mul (isUnit_iff_ne_zero.mpr (Finset.prod_ne_zero_iff.mpr fun j _ => hc0 j))
  have hc : A.charpoly.roots.toList.length = n := by
    rw [Multiset.length_toList, IsAlgClosed.card_roots_eq_natDegree, charpoly_natDegree_eq_dim,
      Fintype.card_fin]
  let w : Fin n → ℂ := fun i => A.charpoly.roots.toList[i.1]'(by rw [hc]; exact i.2)
  have hl : (List.ofFn w : Multiset ℂ) = A.charpoly.roots := by
    rw [← Multiset.coe_toList A.charpoly.roots]
    congr 1
    exact List.ext_getElem (by simp [hc]) fun i _ _ => by simp [w]
  have hu : ∀ j, ∃ u : Fin n → ℂ, star u ⬝ᵥ u = 1 ∧ A *ᵥ u = w j • u := by
    intro j
    have hr : A.charpoly.IsRoot (w j) :=
      (mem_roots (charpoly_monic A).ne_zero).mp (Multiset.mem_toList.mp (List.getElem_mem _))
    obtain ⟨v, hv, hv0⟩ := ((Module.End.hasEigenvalue_iff_isRoot_charpoly (toLin' A) (w j)).mpr
      (by rwa [charpoly_toLin'])).exists_hasEigenvector
    rw [Module.End.mem_eigenspace_iff, toLin'_apply] at hv
    obtain ⟨c, -, hc⟩ := exists_unit_smul hv0
    refine ⟨c • v, hc, ?_⟩
    rw [mulVec_smul, hv]
    exact smul_comm _ _ _
  choose u hu using hu
  refine ⟨w, of fun i j => u j i, ?_, fun j => (hu j).1, ?_,
    fun P d hP hAP => absurd ⟨P, d, hP, hAP⟩ hA⟩
  · conv_lhs => rw [← prod_multiset_X_sub_C_of_monic_of_roots_card_eq (charpoly_monic A)
      IsAlgClosed.card_roots_eq_natDegree, ← hl]
    rw [Multiset.map_coe, Multiset.prod_coe, List.map_ofFn, List.prod_ofFn]
    rfl
  · ext i j
    have := congrFun (hu j).2 i
    simp only [mulVec, dotProduct, Pi.smul_apply, smul_eq_mul] at this
    rw [mul_diagonal, Matrix.mul_apply]
    simp only [of_apply]
    rw [this, mul_comm]

/-- Every complex matrix has a Schur decomposition, by induction on its size: a unit eigenvector
`v` (`exists_isEig`) is the first column of a unitary `Q₁` (`exists_unitary_extension`), so the
first column of `B = Q₁ᴴ A Q₁` is `w 0` times the first unit vector, and a unitary `Q₂`
triangularizing the lower right block of `B` gives `Q = Q₁ diag(1, Q₂)` (`unitary_blockOne`),
with `T = Qᴴ A Q`. -/
theorem exists_isSchur (A : Matrix (Fin n) (Fin n) ℂ) : ∃ Q T, IsSchur A Q T := by
  suffices h : ∀ A : Matrix (Fin n) (Fin n) ℂ, ∃ Q, Qᴴ * Q = 1 ∧ (Qᴴ * A * Q).IsUpperTriangular by
    obtain ⟨Q, hQ, hT⟩ := h A
    exact ⟨Q, _, hQ, hT, eq_mul_mul_conjTranspose hQ A⟩
  clear A
  induction n with
  | zero => exact fun A => ⟨1, by simp, fun i _ _ => i.elim0⟩
  | succ n ih =>
    intro A
    obtain ⟨w, V, hV⟩ := exists_isEig A
    obtain ⟨Q1, hQ1, hc⟩ := exists_unitary_extension (s := {0}) (fun _ r => V r 0)
      fun i hi j hj => by
        rw [Set.mem_singleton_iff.mp hi, Set.mem_singleton_iff.mp hj, ite_eq_left rfl]
        exact hV.2.1 0
    have hc0 : ∀ r, Q1 r 0 = V r 0 := hc 0 (Set.mem_singleton 0)
    have hAQ : ∀ r, (A * Q1) r 0 = Q1 r 0 * w 0 := fun r => by
      have e := congrFun (congrFun hV.2.2.1 r) 0
      rw [Matrix.mul_apply, mul_diagonal] at e
      rw [Matrix.mul_apply, hc0]
      simp_rw [hc0]
      exact e
    obtain ⟨B, hB⟩ : ∃ B, B = Q1ᴴ * A * Q1 := ⟨_, rfl⟩
    have hBs : ∀ k : Fin n, B k.succ 0 = 0 := fun k => by
      rw [hB, Matrix.mul_assoc, Matrix.mul_apply]
      simp_rw [hAQ]
      have e := congrFun (congrFun hQ1 k.succ) 0
      rw [one_apply_ne (Fin.succ_ne_zero k), Matrix.mul_apply] at e
      rw [← mul_zero (w 0), ← e, Finset.mul_sum]
      exact Finset.sum_congr rfl fun r _ => by ring
    obtain ⟨Q2, hQ2, hT2⟩ := ih (B.submatrix Fin.succ Fin.succ)
    obtain ⟨hU3, hc3, hs3⟩ := unitary_blockOne hQ2
    generalize blockOne Q2 = Q3 at hU3 hc3 hs3
    have hT : (Q1 * Q3)ᴴ * A * (Q1 * Q3) = Q3ᴴ * B * Q3 := by
      rw [hB, conjTranspose_mul]
      simp only [Matrix.mul_assoc]
    refine ⟨Q1 * Q3, by rw [conjTranspose_mul, Matrix.mul_assoc, ← Matrix.mul_assoc Q1ᴴ, hQ1,
      Matrix.one_mul, hU3], ?_⟩
    rw [hT]
    intro i j hij
    cases i using Fin.cases with
    | zero => exact absurd hij (Fin.not_lt_zero _)
    | succ i =>
      cases j using Fin.cases with
      | zero =>
        rw [hc3]
        simp [mulVec, dotProduct, hBs]
      | succ j =>
        rw [hs3]
        exact hT2 (Fin.succ_lt_succ_iff.mp hij)

end Eig

/-- The `U` of an SVD has at least `min m n` columns. -/
theorem IsSVD.min_le_cols {A : Matrix (Fin m) (Fin n) 𝕜} {U : Matrix (Fin m) (Fin p) 𝕜}
    {S : Fin (min m n) → ℝ} {Vh : Matrix (Fin q) (Fin n) 𝕜} (h : IsSVD A U S Vh) : min m n ≤ p :=
  h.2.2.2.2.2.1

/-- The `Vh` of an SVD has at least `min m n` rows. -/
theorem IsSVD.min_le_rows {A : Matrix (Fin m) (Fin n) 𝕜} {U : Matrix (Fin m) (Fin p) 𝕜}
    {S : Fin (min m n) → ℝ} {Vh : Matrix (Fin q) (Fin n) 𝕜} (h : IsSVD A U S Vh) : min m n ≤ q :=
  h.2.2.2.2.2.2

/-- An SVD restricts to a reduced one, on the first `min m n` columns of `U` and rows of `Vh`. -/
theorem IsSVD.reduce {A : Matrix (Fin m) (Fin n) 𝕜} {U : Matrix (Fin m) (Fin p) 𝕜}
    {S : Fin (min m n) → ℝ} {Vh : Matrix (Fin q) (Fin n) 𝕜} (h : IsSVD A U S Vh) :
    IsSVD A (U.submatrix id (Fin.castLE h.min_le_cols)) S
      (Vh.submatrix (Fin.castLE h.min_le_rows) id) := by
  have hp := h.min_le_cols
  have hq := h.min_le_rows
  show IsSVD A (U.submatrix id (Fin.castLE hp)) S (Vh.submatrix (Fin.castLE hq) id)
  have e1 : U * (1 : Matrix (Fin p) (Fin p) 𝕜).submatrix id (Fin.castLE hp) =
      U.submatrix id (Fin.castLE hp) := by
    ext a l
    simp [Matrix.mul_apply, Matrix.one_apply]
  have e2 : (1 : Matrix (Fin q) (Fin q) 𝕜).submatrix (Fin.castLE hq) id * Vh =
      Vh.submatrix (Fin.castLE hq) id := by
    ext l b
    simp [Matrix.mul_apply, Matrix.one_apply]
  refine ⟨?_, ?_, h.2.2.1, h.2.2.2.1, ?_, le_rfl, le_rfl⟩
  · rw [conjTranspose_submatrix, ← submatrix_mul _ _ _ _ _ Function.bijective_id, h.1,
      submatrix_one _ (Fin.castLE_injective hp)]
  · rw [conjTranspose_submatrix, ← submatrix_mul _ _ _ _ _ Function.bijective_id, h.2.1,
      submatrix_one _ (Fin.castLE_injective hq)]
  · rw [h.2.2.2.2.1, rectDiagonal_eq_mul _ hp hq, rectDiagonal_eq_diagonal, ← e1, ← e2]
    simp only [Matrix.mul_assoc]

open Polynomial in
/-- For a reduced SVD, the characteristic polynomial of `Aᴴ A` is `X^(n - min m n)` times
`∏ i, (X - S i ^ 2)`. -/
theorem IsSVD.charpoly {A : Matrix (Fin m) (Fin n) 𝕜} {U : Matrix (Fin m) (Fin (min m n)) 𝕜}
    {S : Fin (min m n) → ℝ} {Vh : Matrix (Fin (min m n)) (Fin n) 𝕜} (h : IsSVD A U S Vh) :
    (Aᴴ * A).charpoly = X ^ (n - min m n) * ∏ i, (X - C ((S i ^ 2 : ℝ) : 𝕜)) := by
  have hd : (diagonal fun i => (S i : 𝕜))ᴴ = diagonal fun i => (S i : 𝕜) := by
    rw [diagonal_conjTranspose]
    congr 1
    funext i
    simp
  have hs : diagonal (fun i => ((S i ^ 2 : ℝ) : 𝕜)) =
      diagonal (fun i => (S i : 𝕜)) * diagonal fun i => (S i : 𝕜) := by
    rw [diagonal_mul_diagonal]
    congr 1
    funext i
    push_cast
    ring
  have hA : Aᴴ * A = Vhᴴ * (diagonal (fun i => ((S i ^ 2 : ℝ) : 𝕜)) * Vh) := by
    rw [h.2.2.2.2.1, rectDiagonal_eq_diagonal, conjTranspose_mul, conjTranspose_mul, hd, hs]
    simp only [Matrix.mul_assoc]
    rw [← Matrix.mul_assoc Uᴴ U, h.1, Matrix.one_mul]
  rw [hA, charpoly_mul_comm_of_le _ _ (by simp), Matrix.mul_assoc, h.2.1,
    Matrix.mul_one, charpoly_diagonal, Fintype.card_fin, Fintype.card_fin]

open Polynomial in
/-- The singular values are unique: two SVDs of `A` have the same `S`. PyTorch's warning at
`torch.linalg.svd` is that "The returned tensors `U` and `V` are not unique"; `S`, "returned in
descending order", is determined, being the descending square roots of the roots of `Aᴴ A`'s
characteristic polynomial (`IsSVD.charpoly`, `sort_roots_prod`). -/
theorem IsSVD.singularValues_eq {p' q' : ℕ} {A : Matrix (Fin m) (Fin n) 𝕜}
    {U : Matrix (Fin m) (Fin p) 𝕜} {S : Fin (min m n) → ℝ} {Vh : Matrix (Fin q) (Fin n) 𝕜}
    {U' : Matrix (Fin m) (Fin p') 𝕜} {S' : Fin (min m n) → ℝ} {Vh' : Matrix (Fin q') (Fin n) 𝕜}
    (h : IsSVD A U S Vh) (h' : IsSVD A U' S' Vh') : S = S' := by
  have e : ∀ {S : Fin (min m n) → ℝ}, (∀ i, 0 ≤ S i) → Antitone S →
      ((∏ i, (X - C ((S i ^ 2 : ℝ) : 𝕜))).roots.map RCLike.re).sort (· ≤ ·) =
        List.ofFn fun i => S (Fin.rev i) ^ 2 := by
    intro S h0 hS
    rw [Fintype.prod_equiv Fin.revPerm _ (fun i => X - C ((S (Fin.rev i) ^ 2 : ℝ) : 𝕜))
      fun i => by simp]
    exact sort_roots_prod fun i j hij => pow_le_pow_left₀ (h0 _) (hS (Fin.rev_le_rev.mpr hij)) 2
  have hc := (isRegular_X_pow _).left.eq_iff.mp
    (h.reduce.charpoly.symm.trans h'.reduce.charpoly)
  have e1 := e h.2.2.1 h.2.2.2.1
  rw [hc] at e1
  have := List.ofFn_injective (e1.symm.trans (e h'.2.2.1 h'.2.2.2.1))
  funext i
  have hi := congrFun this (Fin.rev i)
  simp only [Fin.rev_rev] at hi
  exact (pow_left_inj₀ (h.2.2.1 i) (h'.2.2.1 i) two_ne_zero).mp hi

/-- With distinct positive singular values, a reduced singular value decomposition is unique up
to a phase for each pair of singular vectors: `(U', S', Vh')` is one exactly when `S' = S`,
`U' = U diag(d)` and `Vh' = diag(star d) Vh` with `star (d i) * d i = 1`. This is the freedom of
PyTorch's warning at `torch.linalg.svd`, "multiplying any pair of singular vectors `u_k, v_k` by
`-1` in the real case or by `e^{i φ}, φ ∈ ℝ` in the complex case produces another two valid
singular vectors of the matrix". For the full shapes, `IsSVD.reduce` gives the first `min m n`
columns of `U` and rows of `Vh` this freedom; the others, PyTorch notes, "can be arbitrary bases
of the corresponding subspaces". -/
theorem IsSVD.isSVD_iff {A : Matrix (Fin m) (Fin n) 𝕜} {U U' : Matrix (Fin m) (Fin (min m n)) 𝕜}
    {S S' : Fin (min m n) → ℝ} {Vh Vh' : Matrix (Fin (min m n)) (Fin n) 𝕜} (h : IsSVD A U S Vh)
    (hS : StrictAnti S) (h0 : ∀ i, 0 < S i) :
    IsSVD A U' S' Vh' ↔ S' = S ∧ ∃ d : Fin (min m n) → 𝕜, (∀ i, star (d i) * d i = 1) ∧
      U' = U * diagonal d ∧ Vh' = diagonal (star d) * Vh := by
  have hcomm : ∀ a b : Fin (min m n) → 𝕜, diagonal a * diagonal b = diagonal b * diagonal a :=
    fun a b => by
      rw [diagonal_mul_diagonal, diagonal_mul_diagonal]
      congr 1
      funext i
      exact mul_comm _ _
  constructor
  · intro h'
    -- The singular values agree; write both factorizations with `diagonal s`.
    have hSS := h.singularValues_eq h'
    subst hSS
    obtain ⟨s, hs⟩ : ∃ s : Fin (min m n) → 𝕜, s = fun i => (S i : 𝕜) := ⟨_, rfl⟩
    have hA : A = U * diagonal s * Vh := by
      rw [hs, ← rectDiagonal_eq_diagonal]
      exact h.2.2.2.2.1
    have hA' : A = U' * diagonal s * Vh' := by
      rw [hs, ← rectDiagonal_eq_diagonal]
      exact h'.2.2.2.2.1
    have hst : star s = s := by
      rw [hs]
      funext i
      simp
    have hs0 : ∀ i, s i ≠ 0 := fun i => by
      rw [hs]
      exact RCLike.ofReal_ne_zero.mpr (h0 i).ne'
    -- The squares `t` are distinct, since `S` is strictly decreasing and positive.
    obtain ⟨t, ht⟩ : ∃ t : Fin (min m n) → 𝕜, t = fun i => s i * s i := ⟨_, rfl⟩
    have hinj : Function.Injective t := fun i j hij => by
      simp only [ht, hs] at hij
      have hij' : S i * S i = S j * S j := by exact_mod_cast hij
      exact hS.injective ((mul_self_inj (h0 i).le (h0 j).le).mp hij')
    -- Both factorizations give `Aᴴ A = Vhᴴ (diagonal t) Vh`.
    have hG : ∀ {U₁ : Matrix (Fin m) (Fin (min m n)) 𝕜} {Vh₁ : Matrix (Fin (min m n)) (Fin n) 𝕜},
        U₁ᴴ * U₁ = 1 → A = U₁ * diagonal s * Vh₁ → Aᴴ * A = Vh₁ᴴ * diagonal t * Vh₁ := by
      intro U₁ Vh₁ hU₁ hA₁
      rw [hA₁, conjTranspose_mul, conjTranspose_mul, diagonal_conjTranspose, hst, ht,
        ← diagonal_mul_diagonal]
      simp only [Matrix.mul_assoc]
      rw [← Matrix.mul_assoc U₁ᴴ, hU₁, Matrix.one_mul]
    have h1 := hG h.1 hA
    have h2 := hG h'.1 hA'
    -- So `Vh Vh'ᴴ` commutes with `diagonal t`, and distinct entries make it diagonal.
    have e : diagonal t * (Vh * Vh'ᴴ) = Vh * Vh'ᴴ * diagonal t :=
      calc diagonal t * (Vh * Vh'ᴴ) = Vh * (Vhᴴ * diagonal t * Vh) * Vh'ᴴ := by
            simp only [Matrix.mul_assoc]
            rw [← Matrix.mul_assoc Vh Vhᴴ, h.2.1, Matrix.one_mul]
        _ = Vh * (Vh'ᴴ * diagonal t * Vh') * Vh'ᴴ := by rw [← h1, h2]
        _ = Vh * Vh'ᴴ * diagonal t := by
            simp only [Matrix.mul_assoc]
            rw [h'.2.1, Matrix.mul_one]
    obtain ⟨c, hc⟩ : ∃ c, Vh * Vh'ᴴ = diagonal c :=
      ⟨_, (isDiag_of_diagonal_mul_eq hinj e).diagonal_diag.symm⟩
    -- `diagonal t Vh' = diagonal t (diagonal (star c) Vh)`, and the `t i` are nonzero.
    have hV : Vh' = diagonal (star c) * Vh := by
      have hc' : Vh' * Vhᴴ = diagonal (star c) := by
        rw [← diagonal_conjTranspose, ← hc, conjTranspose_mul, conjTranspose_conjTranspose]
      have e' : diagonal t * Vh' = diagonal t * (diagonal (star c) * Vh) :=
        calc diagonal t * Vh' = Vh' * (Vh'ᴴ * diagonal t * Vh') := by
              simp only [Matrix.mul_assoc]
              rw [← Matrix.mul_assoc Vh' Vh'ᴴ, h'.2.1, Matrix.one_mul]
          _ = Vh' * Vhᴴ * diagonal t * Vh := by
              rw [← h2, h1]
              simp only [Matrix.mul_assoc]
          _ = diagonal t * (diagonal (star c) * Vh) := by
              rw [hc', hcomm (star c), Matrix.mul_assoc]
      ext i j
      have := congrFun (congrFun e' i) j
      rw [diagonal_mul, diagonal_mul] at this
      have ht0 : t i ≠ 0 := by
        rw [ht]
        exact mul_ne_zero (hs0 i) (hs0 i)
      exact mul_left_cancel₀ ht0 this
    -- `U' (diagonal s) = A Vh'ᴴ = U (diagonal c) (diagonal s)`, and the `s i` are nonzero.
    have hU : U' = U * diagonal c := by
      have e' : U' * diagonal s = U * diagonal c * diagonal s :=
        calc U' * diagonal s = A * Vh'ᴴ := by
              rw [hA']
              simp only [Matrix.mul_assoc]
              rw [h'.2.1, Matrix.mul_one]
          _ = U * diagonal s * (Vh * Vh'ᴴ) := by
              rw [hA]
              simp only [Matrix.mul_assoc]
          _ = U * diagonal c * diagonal s := by
              rw [hc, Matrix.mul_assoc U (diagonal s), hcomm s c, ← Matrix.mul_assoc]
      ext i j
      have := congrFun (congrFun e' i) j
      rw [mul_diagonal, mul_diagonal] at this
      exact mul_right_cancel₀ (hs0 j) this
    -- `Vh'` has orthonormal rows, so each `c i` has modulus one.
    refine ⟨rfl, c, fun i => ?_, hU, hV⟩
    have := congrFun (congrFun h'.2.1 i) i
    rw [hV, conjTranspose_mul, diagonal_conjTranspose, star_star, Matrix.mul_assoc,
      ← Matrix.mul_assoc Vh, h.2.1, Matrix.one_mul, diagonal_mul_diagonal, one_apply_eq,
      diagonal_apply_eq] at this
    exact this
  -- Conversely, unit phases keep both factors' orthonormality and cancel in the product.
  · rintro ⟨hS', d, hd, rfl, rfl⟩
    rw [hS']
    have hdd : diagonal (star d) * diagonal d = (1 : Matrix (Fin (min m n)) (Fin (min m n)) 𝕜) := by
      rw [diagonal_mul_diagonal, ← diagonal_one]
      congr 1
      funext i
      exact hd i
    refine ⟨?_, ?_, h.2.2.1, h.2.2.2.1, ?_, le_rfl, le_rfl⟩
    · rw [conjTranspose_mul, diagonal_conjTranspose, Matrix.mul_assoc, ← Matrix.mul_assoc Uᴴ, h.1,
        Matrix.one_mul, hdd]
    · rw [conjTranspose_mul, diagonal_conjTranspose, star_star, Matrix.mul_assoc,
        ← Matrix.mul_assoc Vh, h.2.1, Matrix.one_mul, hdd]
    · calc A = U * diagonal (fun i => (S i : 𝕜)) * Vh := by
            rw [← rectDiagonal_eq_diagonal]
            exact h.2.2.2.2.1
        _ = U * (diagonal (fun i => (S i : 𝕜)) * (diagonal d * diagonal (star d))) * Vh := by
            rw [hcomm d, hdd, Matrix.mul_one]
        _ = U * diagonal d * rectDiagonal (fun i => (S i : 𝕜)) * (diagonal (star d) * Vh) := by
            rw [rectDiagonal_eq_diagonal, ← Matrix.mul_assoc (diagonal _) (diagonal d), hcomm _ d]
            simp only [Matrix.mul_assoc]

/-- The conjugate transpose of a singular value decomposition of `A` is one of `Aᴴ`, `Vhᴴ` and
`Uᴴ` exchanging roles. -/
theorem IsSVD.conjTranspose {A : Matrix (Fin m) (Fin n) 𝕜} {U : Matrix (Fin m) (Fin p) 𝕜}
    {S : Fin (min m n) → ℝ} {Vh : Matrix (Fin q) (Fin n) 𝕜} (h : IsSVD A U S Vh) :
    IsSVD Aᴴ Vhᴴ (S ∘ Fin.cast (min_comm n m)) Uᴴ := by
  have hd : (rectDiagonal (fun i => (S i : 𝕜)) : Matrix (Fin p) (Fin q) 𝕜)ᴴ =
      rectDiagonal fun i => ((S ∘ Fin.cast (min_comm n m)) i : 𝕜) := by
    ext i j
    have hmn : min n m = min m n := min_comm n m
    simp only [conjTranspose_apply, rectDiagonal, of_apply, Function.comp_apply]
    split_ifs with h1 h2 h2
    · rw [RCLike.star_def, RCLike.conj_ofReal]
      exact congrArg _ (congrArg S (Fin.ext (by simp only [Fin.val_cast]; omega)))
    · exact absurd ⟨by omega, by omega⟩ h2
    · exact absurd ⟨by omega, by omega⟩ h1
    · exact star_zero _
  refine ⟨by rw [conjTranspose_conjTranspose, h.2.1], by rw [conjTranspose_conjTranspose, h.1],
    fun i => h.2.2.1 _, fun i j hij => h.2.2.2.1 (show Fin.cast (min_comm n m) i ≤
      Fin.cast (min_comm n m) j from hij), ?_, (min_comm n m).trans_le h.min_le_rows,
    (min_comm n m).trans_le h.min_le_cols⟩
  rw [h.2.2.2.2.1, conjTranspose_mul, conjTranspose_mul, hd, Matrix.mul_assoc]

open scoped ComplexOrder in
/-- Every `m × n` matrix with `n ≤ m` has a full singular value decomposition: `Vh` is `Wᴴ` for
an eigendecomposition `W` of `-(Aᴴ A)` (`exists_isEigh`), whose ascending eigenvalues are the
negated squares of the descending singular values, and `U` extends the columns of `A W` of
positive norm, scaled to unit norm (`exists_unitary_extension`). -/
theorem exists_isSVD_of_le (A : Matrix (Fin m) (Fin n) 𝕜) (hnm : n ≤ m) :
    ∃ (U : Matrix (Fin m) (Fin m) 𝕜) (S : Fin (min m n) → ℝ) (Vh : Matrix (Fin n) (Fin n) 𝕜),
      IsSVD A U S Vh := by
  obtain ⟨W, w, hW⟩ := exists_isEigh (isHermitian_conjTranspose_mul_self A).neg
  have hG : ∀ i j, star (fun r => (A * W) r i) ⬝ᵥ (fun r => (A * W) r j) =
      diagonal (fun i => ((-w i : ℝ) : 𝕜)) i j := fun i j => by
    have e : (A * W)ᴴ * (A * W) = diagonal fun i => ((-w i : ℝ) : 𝕜) := by
      rw [conjTranspose_mul, Matrix.mul_assoc, ← Matrix.mul_assoc Aᴴ, ← neg_neg (Aᴴ * A),
        Matrix.neg_mul, hW.mul_eq, Matrix.mul_neg, ← Matrix.mul_assoc, hW.1, Matrix.one_mul,
        diagonal_neg]
      simp
    rw [← e]
    rfl
  have h0 : ∀ j, 0 ≤ -w j := fun j => by
    have h := dotProduct_star_self_nonneg (fun r => (A * W) r j)
    rw [hG, diagonal_apply_eq] at h
    exact RCLike.ofReal_nonneg.mp h
  have hinv : ∀ x : ℝ, 0 < x → (Real.sqrt x)⁻¹ * ((Real.sqrt x)⁻¹ * x) = 1 := fun x hx => by
    rw [← mul_assoc, ← mul_inv, Real.mul_self_sqrt hx.le, inv_mul_cancel₀ hx.ne']
  obtain ⟨U, hU, hUu⟩ := exists_unitary_extension (s := {k | ∃ h : k.val < n, 0 < -w ⟨k, h⟩})
    (fun k => if h : k.val < n then
      (((Real.sqrt (-w ⟨k, h⟩))⁻¹ : ℝ) : 𝕜) • (fun r => (A * W) r ⟨k, h⟩) else 0) (by
      rintro i ⟨hi, hpi⟩ j ⟨hj, -⟩
      simp only [hi, hj, ↓reduceDIte]
      rw [star_smul, smul_dotProduct, dotProduct_smul, hG, smul_eq_mul, smul_eq_mul,
        RCLike.star_def, RCLike.conj_ofReal]
      by_cases hij : i = j
      · subst hij
        rw [diagonal_apply_eq, ite_eq_left rfl]
        exact_mod_cast hinv _ hpi
      · rw [diagonal_apply_ne _ (show (⟨i, hi⟩ : Fin n) ≠ ⟨j, hj⟩ from
          fun e => hij (Fin.ext (Fin.mk.inj_iff.mp e))), ite_eq_right hij, mul_zero, mul_zero])
  obtain ⟨S, hS⟩ : ∃ S : Fin (min m n) → ℝ,
      ∀ i, S i = Real.sqrt (-w (Fin.castLE (min_le_right m n) i)) := ⟨_, fun _ => rfl⟩
  have hAW : A * W = U * (rectDiagonal (fun i => (S i : 𝕜)) : Matrix (Fin m) (Fin n) 𝕜) := by
    ext r j
    have hz : ∀ k, k ≠ Fin.castLE hnm j → U r k * rectDiagonal (fun i => (S i : 𝕜)) k j = 0 :=
      fun k hk => by rw [rectDiagonal, of_apply, dite_eq_right fun h => hk (Fin.ext h.1), mul_zero]
    have hd : rectDiagonal (p := m) (fun i => (S i : 𝕜)) (Fin.castLE hnm j) j =
        (Real.sqrt (-w j) : 𝕜) := by
      rw [rectDiagonal, of_apply, dite_eq_left (show (Fin.castLE hnm j).val = j.val ∧
        (Fin.castLE hnm j).val < min m n from ⟨rfl, lt_min (Fin.castLE hnm j).2 j.2⟩), hS]
      rfl
    conv_rhs => rw [Matrix.mul_apply, Fintype.sum_eq_single _ hz, hd]
    rcases (h0 j).lt_or_eq with hp | hp
    · rw [hUu _ (Set.mem_ofPred.mpr ⟨j.2, hp⟩) r]
      simp only [Fin.val_castLE, Fin.is_lt, ↓reduceDIte, Fin.eta, Pi.smul_apply, smul_eq_mul]
      rw [mul_right_comm, ← RCLike.ofReal_mul, inv_mul_cancel₀ (Real.sqrt_pos.mpr hp).ne',
        RCLike.ofReal_one, one_mul]
    · have hc := hG j j
      simp only [diagonal_apply_eq, ← hp, RCLike.ofReal_zero, dotProduct_star_self_eq_zero] at hc
      rw [← hp, Real.sqrt_zero, RCLike.ofReal_zero, mul_zero]
      exact congrFun hc r
  refine ⟨U, S, Wᴴ, hU, by rw [conjTranspose_conjTranspose, hW.1],
    fun i => by rw [hS]; exact Real.sqrt_nonneg _, fun i j hij => ?_, ?_, min_le_left m n,
    min_le_right m n⟩
  · rw [hS, hS]
    exact Real.sqrt_le_sqrt (neg_le_neg (hW.2.1 (show Fin.castLE (min_le_right m n) i ≤
      Fin.castLE (min_le_right m n) j from hij)))
  · rw [← hAW, Matrix.mul_assoc, mul_eq_one_comm.mp hW.1, Matrix.mul_one]

/-- Every matrix has a full singular value decomposition, and so, by `IsSVD.reduce`, a reduced
one: `exists_isSVD_of_le` when `n ≤ m`, and for `Aᴴ` conjugate transposed (`IsSVD.conjTranspose`)
otherwise. -/
theorem exists_isSVD (A : Matrix (Fin m) (Fin n) 𝕜) :
    ∃ (U : Matrix (Fin m) (Fin m) 𝕜) (S : Fin (min m n) → ℝ) (Vh : Matrix (Fin n) (Fin n) 𝕜),
      IsSVD A U S Vh := by
  rcases le_total n m with h | h
  · exact exists_isSVD_of_le A h
  · obtain ⟨U, S, Vh, hA⟩ := exists_isSVD_of_le Aᴴ h
    have h' := hA.conjTranspose
    rw [conjTranspose_conjTranspose] at h'
    exact ⟨_, _, _, h'⟩

open Polynomial in
/-- `diagonal d` has the eigenvalues `d` with the eigenvectors `1`: `IsEig (diagonal d) d 1`. -/
theorem isEig_diagonal (d : Fin n → ℂ) : IsEig (diagonal d) d 1 :=
  ⟨charpoly_diagonal d, fun j => by simp, by simp, fun _ _ _ _ => by simp⟩

/-- `rectDiagonal S`, for a nonnegative and non-increasing `S`, is its own singular value
decomposition with `U = 1` and `Vh = 1`. -/
theorem isSVD_rectDiagonal {S : Fin (min m n) → ℝ} (h0 : ∀ i, 0 ≤ S i) (hS : Antitone S) :
    IsSVD (rectDiagonal fun i => (S i : 𝕜)) (1 : Matrix (Fin m) (Fin m) 𝕜) S
      (1 : Matrix (Fin n) (Fin n) 𝕜) :=
  ⟨by simp, by simp, h0, hS, by simp, min_le_left m n, min_le_right m n⟩

/-- Every 2 × 2 real matrix with a negative discriminant is its own real Schur form. -/
theorem isRealSchur_two {a b c d : ℝ} (h : (a - d) ^ 2 + 4 * b * c < 0) :
    IsRealSchur !![a, b; c, d] 1 !![a, b; c, d] := by
  refine ⟨by simp, upperBand_one_iff.mp fun i j hij => absurd hij (by have := i.isLt; omega),
    fun i j hij _ => ?_, by simp⟩
  fin_cases i <;> fin_cases j <;> simp at hij
  exact ⟨by simpa using h, fun l hl => absurd hl (by have := l.isLt; simp; omega)⟩

/-- An upper Hessenberg `A` is its own `gehrd` output with `τ = 0`. -/
theorem isGehrd_self {A : Matrix (Fin n) (Fin n) 𝕜} (hA : A.IsUpperHessenberg) :
    IsGehrd A A 0 := by
  rw [IsGehrd, householderQ_zero, upperPart_eq_self (upperBand_one_iff.mpr hA)]
  exact ⟨by simp, hA, by simp⟩

/-- `symTridiagonal d e` is its own `sytrd` output with diagonal `d`, subdiagonal `e` and `τ = 0`.
-/
theorem isSytrd_symTridiagonal (d : Fin n → ℝ) (e : Fin (n - 1) → ℝ) :
    IsSytrd (symTridiagonal d e : Matrix (Fin n) (Fin n) 𝕜) (symTridiagonal d e) d e 0 := by
  refine ⟨?_, fun i => by simp [symTridiagonal, tridiagonal], fun j => ?_⟩
  · rw [householderQ_zero]
    exact ⟨by simp, by simp⟩
  · simp [symTridiagonal, tridiagonal]

/-- For an invertible tridiagonal matrix, the solution of `A X = B` is `A⁻¹ B`. -/
theorem isTridiagonalSolve_iff {dl d du : Fin m → 𝕜} {B X : Matrix (Fin m) (Fin r) 𝕜}
    (hu : IsUnit (tridiagonal dl d du).det) :
    IsTridiagonalSolve dl d du B X ↔ X = (tridiagonal dl d du)⁻¹ * B := by
  constructor
  · intro h
    rw [← h, ← Matrix.mul_assoc, nonsing_inv_mul _ hu, Matrix.one_mul]
  · rintro rfl
    rw [IsTridiagonalSolve, ← Matrix.mul_assoc, mul_nonsing_inv _ hu, Matrix.one_mul]

end TorchLean.Factorization
