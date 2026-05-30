import VersoManual
import VersoBlueprint

open Verso.Genre Manual

#doc (Manual) "Matrix Factorizations for Kernel Methods" =>
%%%
tag := "matrix-factorizations"
%%%

Kernel and Gaussian-process methods do not reduce to a single forward pass. Their numerical core is a
matrix factorization. The motivating target here is
[Computational Hypergraph Discovery](https://github.com/TheoBourdais/ComputationalHypergraphDiscovery)
(CHD): a Gaussian-process / kernel-ridge method that recovers the dependency structure of a system by
repeatedly solving regularized kernel systems and testing the resulting variances. Every quantity CHD
inspects — the variational solution, the noise/ridge parameter, and the `Z`-test — is a function of the
*full symmetric eigendecomposition* of a kernel matrix `K`.

TorchLean previously had only a power-iteration stub that recovers the *largest* eigenpair. The spec
layer now provides real, shape-indexed reference factorizations in
[`NN.Spec.Core.Tensor.Factorizations`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Spec/Core/Tensor/Factorizations.lean):
Cholesky (`choleskySpec`), QR via modified Gram–Schmidt (`qrSpec`), the full symmetric
eigendecomposition via cyclic Jacobi (`symEigJacobiSpec`), and the SVD (`svdSpec`). The correctness
theorems live in
[`NN.Proofs.Tensor.Basic.Factorizations`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/Tensor/Basic/Factorizations.lean).

# What "verified factorization" can and cannot mean

A subtle but decisive point governs the whole chapter. The executable specs are
`Context`-polymorphic and run over Lean's native `Float` (IEEE binary64). Two of them — Cholesky and
QR — are *finite* constructions, so over the reals they reconstruct their input exactly under the
usual success hypotheses. The other two — the cyclic Jacobi eigensolver and the SVD built on it — are
*iterative*. After a finite number of sweeps the rotated matrix is only approximately diagonal, and in
floating point it is never exactly diagonal. Mathlib v4.30.0 contains no Jacobi convergence theory.

So `A = V · diag(λ) · Vᵀ` is _not_ an a-priori theorem about the floating-point output. The honest
verification therefore splits into three kinds of statement, all proved over `ℝ`:

- *Specification consequences*: facts CHD consumes, proved from a predicate that says "these matrices
  form an eigendecomposition", independent of any algorithm.
- *Exact invariants*: properties the algorithm satisfies on the nose at every step.
- *A-posteriori certificate*: an exact identity bounding the reconstruction residual by the
  off-diagonal mass, with the runtime `assertLt` checks supplying the numeric bound on concrete inputs.

# Specification consequences (the CHD foundation)

The specification predicate is `IsSymEig A Λ V`: an orthogonal `V` (`Vᵀ V = 1`) with
`A = V · diag(Λ) · Vᵀ`. From it the kernel-method facts follow without reference to the solver.

The central one is the regularized inverse behind `solve_variationnal`. CHD repeatedly forms
`(K + γ I)⁻¹ b`; diagonalizing turns this into a per-eigenvalue rescaling:

$$`(K+\gamma I)^{-1} = V\,\operatorname{diag}\!\left(\tfrac{1}{\lambda_i+\gamma}\right) V^\top,
\qquad \gamma \neq -\lambda_i.`

This is `IsSymEig.add_smul_inv`, proved purely from orthogonality of `V` (so it holds for *any*
eigendecomposition the solver returns, not only Mathlib's canonical one). The supporting rewrite
`IsSymEig.add_smul_eq` expresses `K + γI = V · diag(λ + γ) · Vᵀ`, and
`orthogonal_conj_diagonal_mul_inv` is the reusable fact that conjugating a diagonal by an orthogonal
matrix is inverted by conjugating the entrywise inverse.

The scalar summaries used by `find_gamma` and the evidence terms are `IsSymEig.trace_eq`
(`trace K = Σ λᵢ`) and `IsSymEig.det_eq` (`det K = Π λᵢ`). Symmetry itself is `IsSymEig.isHermitian`.

CHD actually builds the Gram matrix `K = Aᵀ A`. `IsSVD.gram_isSymEig` records that an SVD of `A` is
exactly an eigendecomposition of that Gram matrix, with eigenvalues `σᵢ²` and the same orthogonal `V` —
connecting the SVD spec to the eigendecomposition foundation.

# Exact invariants of the algorithms

Some properties hold exactly, with no convergence or rounding caveat, and these pin down the precise
sense in which the iterative solver is faithful.

The cyclic Jacobi iteration applies Givens rotations `J` with `A ← Jᵀ A J` and `V ← V J`. Each `J` is
orthogonal: with `c = 1/\sqrt{1+t^2}` and `s = t c` (the parameters the implementation uses),
`givens_normSq` proves `c² + s² = 1`. Consequently every sweep is an *orthogonal similarity*, and
`trace_orthogonal_conj` and `det_orthogonal_conj` show that the trace and determinant of the running
matrix equal those of the original at every step — the spectrum is preserved exactly, however far the
off-diagonal has been driven down.

For the finite Cholesky construction, `choleskyFn_lower_triangular` (and its tensor-level form
`choleskySpec_lower_triangular`) proves the factor is lower-triangular: entries above the diagonal
vanish by construction. The proof reads the column produced at each position out of the `List.foldl`
that builds the factor, via the reusable indexing lemma `getD_foldl_finRange`.

# The a-posteriori residual certificate

For the iterative routines, the replacement for an impossible a-priori convergence proof is an exact
residual identity. Writing `Af = Vᵀ A V` for the rotated matrix and `Λ` for its diagonal,
`symEig_reconstruction_residual` shows

$$`A - V\,\operatorname{diag}(A_f)\,V^\top \;=\; V\,\operatorname{offDiag}(A_f)\,V^\top,`

so the reconstruction error is exactly the orthogonal conjugation of the off-diagonal part of `Af`.
Because orthogonal conjugation preserves the Frobenius norm, `symEig_frobenius_residual` upgrades this
to an equality of squared Frobenius masses:

$$`\bigl\|A - V\,\operatorname{diag}(A_f)\,V^\top\bigr\|_F^2
   \;=\; \bigl\|\operatorname{offDiag}(A_f)\bigr\|_F^2,`

expressed in Lean as an equality of `trace(Rᵀ R)` terms. The residual is `0` exactly when `Af` is
diagonal, which is the precise meaning of "more Jacobi sweeps shrink the error". And in that
zero-residual limit, `isSymEig_of_diagonal` shows the solver output `(diag Af, V)` is an exact
`IsSymEig` decomposition. The numeric `assertLt` reconstruction checks in
`NN/Examples/Factorization` are concrete instances of this certificate: they bound the off-diagonal
mass on specific matrices.

# What remains

The exact algebraic reconstruction of the *finite* executable factorizations — `A = L · Lᵀ` for the
Cholesky column fold under positive pivots, and `A = Q · R` with `Qᵀ Q = 1` for Gram–Schmidt under
full column rank — is the natural next increment. It needs an induction relating the `List.foldl`
prefix at step `j` to the first `j` produced columns (a strengthening of `getD_foldl_finRange`)
together with the per-pivot positivity discharge from `Matrix.PosDef`. The specification-level facts
the kernel methods rely on are independent of that step, so the CHD foundation is already in place.
