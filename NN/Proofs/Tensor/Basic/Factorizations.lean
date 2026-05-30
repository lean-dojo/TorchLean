/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Tensor.Factorizations
public import NN.Proofs.Tensor.Basic.LinearAlgebra
public import Mathlib.Analysis.Matrix.Spectrum
public import Mathlib.Analysis.Matrix.PosDef
public import Mathlib.Analysis.Matrix.HermitianFunctionalCalculus
public import Mathlib.LinearAlgebra.Matrix.PosDef
public import Mathlib.LinearAlgebra.UnitaryGroup
public import Mathlib.LinearAlgebra.Matrix.NonsingularInverse
public import Mathlib.Data.List.GetD

/-!
# Correctness of the matrix factorizations (foundation for CHD)

This file provides the **formal correctness theorems** for the spec-layer factorizations in
[`NN.Spec.Core.Tensor.Factorizations`](../../../Spec/Core/Tensor/Factorizations.lean)
(`choleskySpec`, `qrSpec`, `symEigJacobiSpec`, `svdSpec`). The motivation is
[Computational Hypergraph Discovery](https://github.com/TheoBourdais/ComputationalHypergraphDiscovery):
a Gaussian-process / kernel-ridge method whose numerical core reduces to the **full symmetric
eigendecomposition** of a kernel matrix `K`. CHD's `solve_variationnal`, `find_gamma` and `Z_test`
are all expressed through the eigendecomposition of `K`, so a verified linear-algebra foundation is a
prerequisite for formalizing CHD.

## Architecture (refinement)

* **Specifications** (`IsCholesky`, `IsQR`, `IsSymEig`, `IsSVD`) are `Prop`s on Mathlib
  `Matrix (Fin n) (Fin n) ℝ`. Mathlib's `Matrix m n α` is *definitionally* `m → n → α`, so the
  function representation `Spec.toMatFn` produced by the executable specs bridges for free.
* **Foundation theorems** (this is what CHD consumes) are proved from the *specifications*, independent
  of the executable algorithm, via Mathlib's spectral theorem and continuous functional calculus.
* **Algorithm theorems** connect the executable `Spec.*Fn` defs to the specifications. Proven here:
  the Cholesky factor is lower-triangular (`choleskyFn_lower_triangular`); the Jacobi/SVD routines
  satisfy their *exact* invariants — orthogonal similarity preserves trace/determinant
  (`trace_orthogonal_conj`, `det_orthogonal_conj`), the Givens rotation is orthogonal
  (`givens_normSq`), and the eigendecomposition is exact in the zero-residual limit
  (`isSymEig_of_diagonal`), with the finite-sweep error captured a-posteriori by
  `symEig_frobenius_residual`.

## Scope honesty

`A = V · diag(λ) · Vᵀ` is **not** an exact theorem for the finite-sweep / floating-point Jacobi output;
it is the *target* certified at runtime by the `assertLt` checks in `NN/Examples/Factorization`, and
bounded a-posteriori here by `symEig_frobenius_residual` (residual = off-diagonal mass of `Af`).
Mathlib v4.30.0 has no Jacobi convergence theory and `Float` never diagonalizes exactly, so no
a-priori convergence theorem is possible.

The exact algebraic reconstruction of the executable *finite* factorizations — `A = L · Lᵀ` for
`choleskyFn` (under SPD pivots) and `A = Q · R`, `Qᵀ Q = 1` for `gramSchmidtFn` (under full column
rank) — is the remaining increment: it requires an induction relating the `List.foldl` prefix at step
`j` to the first `j` columns (extending `getD_foldl_finRange`) plus the per-pivot positivity discharge.
The specification-level consequences CHD needs (above) are independent of that algorithmic step.
-/

@[expose] public section

namespace Spec.Factorization

open Matrix
open scoped BigOperators

variable {n : Nat}

/-! ## Specifications

The mathematical meaning of each factorization, as a predicate over real matrices. Over `ℝ`,
`star = id` so `conjTranspose = transpose`; we phrase everything with `ᵀ`.
-/

/-- `L` is a Cholesky factor of `A`: lower-triangular with `A = L · Lᵀ`. -/
def IsCholesky (A L : Matrix (Fin n) (Fin n) ℝ) : Prop :=
  (∀ i j, i < j → L i j = 0) ∧ A = L * Lᵀ

/-- `(Q, R)` is a QR factorization of `A`: `Q` has orthonormal columns, `R` is upper-triangular,
`A = Q · R`. -/
def IsQR {m k : Nat} (A Q : Matrix (Fin m) (Fin k) ℝ) (R : Matrix (Fin k) (Fin k) ℝ) : Prop :=
  Qᵀ * Q = 1 ∧ (∀ i j, j < i → R i j = 0) ∧ A = Q * R

/-- `(Λ, V)` is a symmetric eigendecomposition of `A`: `V` orthogonal, `A = V · diag(Λ) · Vᵀ`. -/
def IsSymEig (A : Matrix (Fin n) (Fin n) ℝ) (Λ : Fin n → ℝ) (V : Matrix (Fin n) (Fin n) ℝ) : Prop :=
  Vᵀ * V = 1 ∧ A = V * Matrix.diagonal Λ * Vᵀ

/-- `(U, σ, V)` is a (thin) SVD of `A`: `U`, `V` have orthonormal columns, `σ ≥ 0`,
`A = U · diag(σ) · Vᵀ`. -/
def IsSVD {m k : Nat} (A U : Matrix (Fin m) (Fin k) ℝ) (σ : Fin k → ℝ)
    (V : Matrix (Fin k) (Fin k) ℝ) : Prop :=
  Uᵀ * U = 1 ∧ Vᵀ * V = 1 ∧ (∀ j, 0 ≤ σ j) ∧ A = U * Matrix.diagonal σ * Vᵀ

/-! ## Foundation theorems consumed by CHD

These follow from the *specification*, not from any particular algorithm. -/

/-- A symmetric eigendecomposition exhibits `A` as Hermitian (here: symmetric, over `ℝ`). -/
theorem IsSymEig.isHermitian {A : Matrix (Fin n) (Fin n) ℝ} {Λ V}
    (h : IsSymEig A Λ V) : A.IsHermitian := by
  obtain ⟨_, hA⟩ := h
  unfold Matrix.IsHermitian
  rw [hA]
  simp [Matrix.mul_assoc]

/-- From a symmetric eigendecomposition, an orthogonal matrix `V` satisfies `V · Vᵀ = 1` as well as
`Vᵀ · V = 1`. -/
theorem IsSymEig.mul_transpose_self {A : Matrix (Fin n) (Fin n) ℝ} {Λ V}
    (h : IsSymEig A Λ V) : V * Vᵀ = 1 :=
  mul_eq_one_comm.mp h.1

/-! ### The kernel-ridge / `solve_variationnal` identity

CHD repeatedly forms `(K + γ I)⁻¹ b`. Diagonalizing `K = V diag(λ) Vᵀ` turns this into a per-eigenvalue
rescaling `V diag(1/(λ+γ)) Vᵀ b`, which is the basis of `solve_variationnal`, `find_gamma` and the
`Z_test`. The identity below is proved purely from orthogonality of `V` (no appeal to Mathlib's own
spectral decomposition), so it holds for *any* eigendecomposition the algorithm returns. -/

/-- Conjugating a diagonal by an orthogonal `V` is inverted by conjugating the entrywise inverse:
`(V · diag(d) · Vᵀ) · (V · diag(d⁻¹) · Vᵀ) = 1` when every `d i ≠ 0`. -/
theorem orthogonal_conj_diagonal_mul_inv {V : Matrix (Fin n) (Fin n) ℝ} (hV : Vᵀ * V = 1)
    {d : Fin n → ℝ} (hd : ∀ i, d i ≠ 0) :
    (V * Matrix.diagonal d * Vᵀ) * (V * Matrix.diagonal (fun i => (d i)⁻¹) * Vᵀ) = 1 := by
  have hdd : (Matrix.diagonal d) * (Matrix.diagonal (fun i => (d i)⁻¹))
      = (1 : Matrix (Fin n) (Fin n) ℝ) := by
    rw [Matrix.diagonal_mul_diagonal]
    rw [show (fun i => d i * (d i)⁻¹) = (fun _ : Fin n => (1 : ℝ)) from
      funext fun i => mul_inv_cancel₀ (hd i)]
    exact Matrix.diagonal_one
  calc
    (V * Matrix.diagonal d * Vᵀ) * (V * Matrix.diagonal (fun i => (d i)⁻¹) * Vᵀ)
        = V * Matrix.diagonal d * (Vᵀ * V) * Matrix.diagonal (fun i => (d i)⁻¹) * Vᵀ := by
          simp [Matrix.mul_assoc]
    _ = V * (Matrix.diagonal d * Matrix.diagonal (fun i => (d i)⁻¹)) * Vᵀ := by
          rw [hV]; simp [Matrix.mul_assoc]
    _ = V * Vᵀ := by rw [hdd, Matrix.mul_one]
    _ = 1 := mul_eq_one_comm.mp hV

/-- `K + γ I` rewritten through the eigendecomposition: `V · diag(λ + γ) · Vᵀ`. -/
theorem IsSymEig.add_smul_eq {A : Matrix (Fin n) (Fin n) ℝ} {Λ V}
    (h : IsSymEig A Λ V) (γ : ℝ) :
    A + γ • (1 : Matrix (Fin n) (Fin n) ℝ)
      = V * Matrix.diagonal (fun i => Λ i + γ) * Vᵀ := by
  obtain ⟨hV, hA⟩ := h
  have hVV : V * Vᵀ = 1 := mul_eq_one_comm.mp hV
  have hsplit : Matrix.diagonal (fun i => Λ i + γ)
      = Matrix.diagonal Λ + γ • (1 : Matrix (Fin n) (Fin n) ℝ) := by
    ext i j
    by_cases hij : i = j <;>
      simp [Matrix.add_apply, Matrix.smul_apply, hij]
  rw [hsplit, hA]
  rw [Matrix.mul_add, Matrix.add_mul]
  congr 1
  rw [Matrix.mul_smul, Matrix.smul_mul, Matrix.mul_one, hVV]

/-- **Regularized inverse / `solve_variationnal`.** For `γ` avoiding `-λᵢ`, the regularized system
`K + γ I` is inverted by per-eigenvalue rescaling: `(K + γ I)⁻¹ = V · diag(1/(λ + γ)) · Vᵀ`. -/
theorem IsSymEig.add_smul_inv {A : Matrix (Fin n) (Fin n) ℝ} {Λ V}
    (h : IsSymEig A Λ V) (γ : ℝ) (hγ : ∀ i, Λ i + γ ≠ 0) :
    (A + γ • (1 : Matrix (Fin n) (Fin n) ℝ))⁻¹
      = V * Matrix.diagonal (fun i => (Λ i + γ)⁻¹) * Vᵀ := by
  apply Matrix.inv_eq_right_inv
  rw [h.add_smul_eq γ]
  exact orthogonal_conj_diagonal_mul_inv h.1 hγ

/-! ### Spectral trace and determinant (used by `find_gamma` / model-evidence terms) -/

/-- `trace K = Σ λᵢ`. -/
theorem IsSymEig.trace_eq {A : Matrix (Fin n) (Fin n) ℝ} {Λ V}
    (h : IsSymEig A Λ V) : A.trace = ∑ i, Λ i := by
  obtain ⟨hV, hA⟩ := h
  rw [hA, Matrix.trace_mul_comm, ← Matrix.mul_assoc, hV, Matrix.one_mul,
    Matrix.trace_diagonal]

/-- `det K = Π λᵢ`. -/
theorem IsSymEig.det_eq {A : Matrix (Fin n) (Fin n) ℝ} {Λ V}
    (h : IsSymEig A Λ V) : A.det = ∏ i, Λ i := by
  obtain ⟨hV, hA⟩ := h
  have hVV : V * Vᵀ = 1 := mul_eq_one_comm.mp hV
  rw [hA, Matrix.det_mul, Matrix.det_mul, Matrix.det_diagonal,
    mul_right_comm, ← Matrix.det_mul, hVV, Matrix.det_one, one_mul]

/-! ### SVD ⟹ eigendecomposition of the Gram matrix

CHD forms the kernel/Gram matrix `K = Aᵀ A` and eigendecomposes it. An SVD of `A` *is* such an
eigendecomposition, with eigenvalues `σᵢ²` and the same orthogonal `V`. -/

/-- The right singular vectors `V` of `A` diagonalize the Gram matrix `Aᵀ A`, with eigenvalues `σᵢ²`. -/
theorem IsSVD.gram_isSymEig {m k : Nat} {A U : Matrix (Fin m) (Fin k) ℝ}
    {σ : Fin k → ℝ} {V} (h : IsSVD A U σ V) :
    IsSymEig (Aᵀ * A) (fun i => σ i ^ 2) V := by
  obtain ⟨hU, hV, _, hA⟩ := h
  refine ⟨hV, ?_⟩
  have hσσ : Matrix.diagonal σ * Matrix.diagonal σ
      = Matrix.diagonal (fun i => σ i ^ 2) := by
    rw [Matrix.diagonal_mul_diagonal]; simp [pow_two]
  rw [hA, Matrix.transpose_mul, Matrix.transpose_mul, Matrix.transpose_transpose,
    Matrix.diagonal_transpose]
  -- V Dᵀ Uᵀ · U D Vᵀ  with Dᵀ = D
  calc
    V * (Matrix.diagonal σ * Uᵀ) * (U * Matrix.diagonal σ * Vᵀ)
        = V * Matrix.diagonal σ * (Uᵀ * U) * Matrix.diagonal σ * Vᵀ := by
          simp [Matrix.mul_assoc]
    _ = V * (Matrix.diagonal σ * Matrix.diagonal σ) * Vᵀ := by
          rw [hU]; simp [Matrix.mul_assoc]
    _ = V * Matrix.diagonal (fun i => σ i ^ 2) * Vᵀ := by rw [hσσ]

/-! ## Tier B — exact structural & invariant facts

These hold *exactly* (no convergence/rounding caveat). The orthogonal-similarity invariants below are
the precise sense in which the Jacobi iteration is faithful: every sweep is an orthogonal similarity
`A ← Jᵀ A J`, so trace, determinant and spectrum are preserved at every step, independent of how far
the off-diagonal has been driven down. -/

/-- Orthogonal similarity preserves the trace: `trace (V · M · Vᵀ) = trace M` when `Vᵀ · V = 1`. -/
theorem trace_orthogonal_conj {V M : Matrix (Fin n) (Fin n) ℝ} (hV : Vᵀ * V = 1) :
    (V * M * Vᵀ).trace = M.trace := by
  rw [Matrix.trace_mul_comm, ← Matrix.mul_assoc, hV, Matrix.one_mul]

/-- Orthogonal similarity preserves the determinant: `det (V · M · Vᵀ) = det M` when `Vᵀ · V = 1`. -/
theorem det_orthogonal_conj {V M : Matrix (Fin n) (Fin n) ℝ} (hV : Vᵀ * V = 1) :
    (V * M * Vᵀ).det = M.det := by
  have hVV : V * Vᵀ = 1 := mul_eq_one_comm.mp hV
  rw [Matrix.det_mul, Matrix.det_mul, mul_right_comm, ← Matrix.det_mul, hVV, Matrix.det_one,
    one_mul]

/-- **Givens rotation is orthogonal.** With `c = 1/√(1+t²)` and `s = t·c` (the parameters
`arrJacobiRotate` uses), the rotation satisfies `c² + s² = 1`, so every Jacobi step is an orthogonal
transformation. -/
theorem givens_normSq (t : ℝ) :
    (1 / Real.sqrt (1 + t ^ 2)) ^ 2 + (t * (1 / Real.sqrt (1 + t ^ 2))) ^ 2 = 1 := by
  have hpos : (0 : ℝ) < 1 + t ^ 2 := by positivity
  have hsqrt : Real.sqrt (1 + t ^ 2) ^ 2 = 1 + t ^ 2 := Real.sq_sqrt hpos.le
  have hne : (1 + t ^ 2) ≠ 0 := ne_of_gt hpos
  have hc2 : (1 / Real.sqrt (1 + t ^ 2)) ^ 2 = 1 / (1 + t ^ 2) := by
    rw [div_pow, one_pow, hsqrt]
  rw [mul_pow, hc2]
  field_simp

/-! ### Fold-indexing for the column-building specs

`choleskyColsFn` and `gramSchmidtFn` build their output with a left fold that appends one column per
index. The lemmas here read off the column produced at a given position, bridging the executable
`List.foldl` form to per-entry reasoning. They are generic over the appended-value function `g`. -/

section FoldSnoc

variable {β : Type _} {ι : Type _}

/-- A left fold that appends one element per input grows the accumulator by `l.length`. -/
private theorem length_foldl_snoc (g : List β → ι → β) (l : List ι) (acc : List β) :
    (l.foldl (fun s a => s ++ [g s a]) acc).length = acc.length + l.length := by
  induction l generalizing acc with
  | nil => simp
  | cons a t ih =>
      rw [List.foldl_cons, ih]
      simp only [List.length_append, List.length_cons, List.length_nil]
      omega

/-- A fold that only appends never changes an index already inside the accumulator. -/
private theorem getD_foldl_snoc_lt (g : List β → ι → β) (d : β) (l : List ι) (acc : List β)
    (k : Nat) (hk : k < acc.length) :
    (l.foldl (fun s a => s ++ [g s a]) acc).getD k d = acc.getD k d := by
  induction l generalizing acc with
  | nil => simp
  | cons a t ih =>
      rw [List.foldl_cons,
        ih (acc ++ [g acc a]) (by rw [List.length_append]; omega),
        List.getD_append _ _ _ _ hk]

/-- The element at position `j` of the snoc-fold over `finRange n` is `g` applied to the fold of the
length-`j` prefix and the index `j`. -/
private theorem getD_foldl_finRange (g : List β → Fin n → β) (d : β) (j : Fin n) :
    ((List.finRange n).foldl (fun s a => s ++ [g s a]) []).getD j.val d
      = g (((List.finRange n).take j.val).foldl (fun s a => s ++ [g s a]) []) j := by
  have hjlen : j.val < (List.finRange n).length := by
    rw [List.length_finRange]; exact j.isLt
  have htake : (List.finRange n).take (j.val + 1)
      = (List.finRange n).take j.val ++ [j] := by
    rw [List.take_succ_eq_append_getElem hjlen]
    congr 1
    simp [List.getElem_finRange]
  have hplen : (((List.finRange n).take j.val).foldl (fun s a => s ++ [g s a]) []).length
      = j.val := by
    rw [length_foldl_snoc, List.length_nil, List.length_take, List.length_finRange, Nat.zero_add,
      Nat.min_eq_left (Nat.le_of_lt j.isLt)]
  calc
    ((List.finRange n).foldl (fun s a => s ++ [g s a]) []).getD j.val d
        = (((List.finRange n).drop (j.val + 1)).foldl (fun s a => s ++ [g s a])
            ((List.finRange n).take (j.val + 1) |>.foldl (fun s a => s ++ [g s a]) [])).getD
              j.val d := by
          conv_lhs => rw [show List.finRange n
            = (List.finRange n).take (j.val + 1) ++ (List.finRange n).drop (j.val + 1) from
            (List.take_append_drop _ _).symm]
          rw [List.foldl_append]
    _ = ((List.finRange n).take (j.val + 1) |>.foldl (fun s a => s ++ [g s a]) []).getD j.val d := by
          apply getD_foldl_snoc_lt
          rw [length_foldl_snoc, List.length_nil, List.length_take, List.length_finRange,
            Nat.zero_add]
          omega
    _ = g (((List.finRange n).take j.val).foldl (fun s a => s ++ [g s a]) []) j := by
          rw [htake, List.foldl_append, List.foldl_cons, List.foldl_nil]
          rw [List.getD_append_right _ _ _ _ (le_of_eq hplen), hplen, Nat.sub_self]
          rfl

end FoldSnoc

/-! ### Cholesky factor is lower-triangular

A structural fact about the executable `choleskyFn`, proved directly from the column fold: the entry
above the diagonal is forced to `0` by the construction. -/

/-- Reading an entry of a matrix tensor built by `ofMatFn` returns the underlying function value. -/
theorem get2_ofMatFn {m k : Nat} (f : Fin m → Fin k → ℝ) (i : Fin m) (j : Fin k) :
    Spec.get2 (Spec.ofMatFn f) i j = f i j := rfl

/-- The executable Cholesky factor is lower-triangular: entries strictly above the diagonal vanish. -/
theorem choleskyFn_lower_triangular (A : Fin n → Fin n → ℝ) {i j : Fin n} (hij : i.val < j.val) :
    Spec.choleskyFn A i j = 0 := by
  unfold Spec.choleskyFn Spec.choleskyColsFn
  rw [getD_foldl_finRange]
  rw [if_pos hij]

/-- Tensor-level statement: the Cholesky factor `choleskySpec A` is lower-triangular. -/
theorem choleskySpec_lower_triangular (A : Spec.Tensor ℝ (.dim n (.dim n .scalar)))
    {i j : Fin n} (hij : i.val < j.val) :
    Spec.get2 (Spec.choleskySpec A) i j = 0 := by
  rw [show Spec.choleskySpec A = Spec.ofMatFn (Spec.choleskyFn (Spec.toMatFn A)) from rfl,
    get2_ofMatFn]
  exact choleskyFn_lower_triangular _ hij

/-! ## Tier D — convergence as an a-posteriori residual certificate

The cyclic Jacobi iteration produces `(Λ, V)` from the rotated matrix `Af = Vᵀ A V` (an *exact*
orthogonal similarity — see `trace_orthogonal_conj`), with `Λ` the diagonal of `Af`. After finitely
many sweeps `Af` is only *approximately* diagonal, so `A = V·diag(Λ)·Vᵀ` does not hold exactly (and
never does in floating point). Mathlib v4.30.0 has no Jacobi convergence theory, so instead of an
*a-priori* convergence proof we give the *a-posteriori* certificate: the reconstruction residual is
exactly the orthogonal conjugation of the off-diagonal part of `Af`, hence its Frobenius mass equals
the off-diagonal mass — which the runtime `assertLt` checks in `NN/Examples/Factorization` bound on
concrete inputs. -/

/-- The off-diagonal part of a matrix (`0` iff the matrix is diagonal). -/
def offDiagonal (M : Matrix (Fin n) (Fin n) ℝ) : Matrix (Fin n) (Fin n) ℝ :=
  M - Matrix.diagonal (fun i => M i i)

/-- **Exact residual identity.** Reconstructing with the diagonal of `Af` leaves exactly the orthogonal
conjugation of `Af`'s off-diagonal part: `A − V·diag(Af)·Vᵀ = V · offDiag(Af) · Vᵀ`. -/
theorem symEig_reconstruction_residual {A V Af : Matrix (Fin n) (Fin n) ℝ}
    (hA : A = V * Af * Vᵀ) :
    A - V * Matrix.diagonal (fun i => Af i i) * Vᵀ = V * offDiagonal Af * Vᵀ := by
  rw [hA, offDiagonal, Matrix.mul_sub, Matrix.sub_mul]

/-- **Frobenius residual certificate.** The squared Frobenius reconstruction error
`‖A − V·diag(Af)·Vᵀ‖²` equals the squared Frobenius off-diagonal mass `‖offDiag(Af)‖²` (expressed as
`trace(Rᵀ R)`), because orthogonal conjugation preserves the Frobenius norm. In particular it is `0`
iff `Af` is diagonal — the exact sense in which "more Jacobi sweeps ⟹ smaller residual". -/
theorem symEig_frobenius_residual {A V Af : Matrix (Fin n) (Fin n) ℝ} (hV : Vᵀ * V = 1)
    (hA : A = V * Af * Vᵀ) :
    ((A - V * Matrix.diagonal (fun i => Af i i) * Vᵀ)ᵀ
        * (A - V * Matrix.diagonal (fun i => Af i i) * Vᵀ)).trace
      = ((offDiagonal Af)ᵀ * offDiagonal Af).trace := by
  rw [symEig_reconstruction_residual hA]
  have hB : (V * offDiagonal Af * Vᵀ)ᵀ = V * (offDiagonal Af)ᵀ * Vᵀ := by
    rw [Matrix.transpose_mul, Matrix.transpose_mul, Matrix.transpose_transpose, Matrix.mul_assoc]
  have key : (V * offDiagonal Af * Vᵀ)ᵀ * (V * offDiagonal Af * Vᵀ)
      = V * ((offDiagonal Af)ᵀ * offDiagonal Af) * Vᵀ := by
    rw [hB]
    calc
      (V * (offDiagonal Af)ᵀ * Vᵀ) * (V * offDiagonal Af * Vᵀ)
          = V * (offDiagonal Af)ᵀ * (Vᵀ * V) * offDiagonal Af * Vᵀ := by simp [Matrix.mul_assoc]
      _ = V * ((offDiagonal Af)ᵀ * offDiagonal Af) * Vᵀ := by rw [hV]; simp [Matrix.mul_assoc]
  rw [key]
  exact trace_orthogonal_conj hV

/-- **Conditional correctness of Jacobi.** When the rotated matrix `Af = Vᵀ A V` is diagonal (zero
residual — the limit the sweeps drive toward), the Jacobi output `(diag Af, V)` is an *exact*
symmetric eigendecomposition `IsSymEig`. Together with `symEig_frobenius_residual` this is the precise
correctness statement: orthogonality and the orthogonal-similarity hold always; full diagonalization
holds exactly in the zero-residual limit. -/
theorem isSymEig_of_diagonal {A V Af : Matrix (Fin n) (Fin n) ℝ} (hV : Vᵀ * V = 1)
    (hA : A = V * Af * Vᵀ) (hdiag : Af = Matrix.diagonal (fun i => Af i i)) :
    IsSymEig A (fun i => Af i i) V :=
  ⟨hV, by rw [hA]; conv_lhs => rw [hdiag]⟩

end Spec.Factorization
