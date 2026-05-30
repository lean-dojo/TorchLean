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

# Exact Cholesky reconstruction

Cholesky is a _finite_ construction, so unlike the iterative routines it admits an exact
reconstruction theorem — no residual, no convergence caveat. In
[`NN.Proofs.Tensor.Basic.FactorizationsReconstruction`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/Tensor/Basic/FactorizationsReconstruction.lean),
`isCholesky_of_pos` proves that for a symmetric `A` whose executable pivots are all positive
(`0 < L[j,j]`, exactly the condition under which the algorithm succeeds over the reals) the factor
`L = choleskyFn A` is a genuine Cholesky factor:

$$`L \text{ lower-triangular} \quad\text{and}\quad A = L\,L^\top.`

The tensor-level corollary `choleskySpec_reconstruction` states the same per entry:
`A[i,j] = Σ_k L[i,k]·L[j,k]`.

The proof turns the executable algorithm — a `List.foldl` that snocs one column per index — into
per-entry algebra. The reusable lemma `getD_foldl_snoc_read` reads the `j`-th column as the step
function applied to the length-`j` prefix; `prefix_eq_map` then identifies that prefix with the first
`j` columns of the final `L`, and `take_map_sum_eq` rewrites the code's `List.foldl` sums as masked
`Finset` partial sums. Lower-triangularity collapses the matrix product to a partial sum plus a single
pivot term, and the positive-pivot hypothesis discharges the two side conditions: `√` of a positive
radicand for the diagonal (`Real.mul_self_sqrt`) and a non-zero divisor for the below-diagonal
entries. Symmetry of `A` extends the lower-triangular reconstruction to the whole matrix.

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

# Exact QR reconstruction

The QR factorization admits the same treatment. `qr_mul_eq` (in the same file) proves that for an
`A` whose executable Gram–Schmidt `R`-pivots are all positive (`0 < R[j,j]`, the full-column-rank
success condition) the factors satisfy

$$`R \text{ upper-triangular} \quad\text{and}\quad A = Q\,R,`

with `qrSpec_reconstruction` the tensor-level corollary. The new wrinkle is that `gramSchmidtFn`
threads a `GSState` that snocs onto _two_ lists at once — the `Q` columns and the `R` columns. Because
the appended values depend only on the `Q`-history, the `Q`-list is itself a single-list snoc-fold
(`gs_proj_qs`, read by `getD_foldl_snoc_read` as for Cholesky), and the `R`-list is the `Q`-prefix
tail `rTail`, read by `gs_fold_split` together with `rTail_getD`. The orthogonalization sum
`v = a − Σ rₖⱼ qₖ`, a fold over `List.zip`, collapses to a single map-fold (`cross_fold_eq`) and then
to a masked `Finset` partial sum, after which the positive-pivot hypothesis cancels the `v / rⱼⱼ`
normalization exactly.

# Orthonormality of the QR factor (`Qᵀ Q = 1`)

The remaining finite-fold property — orthonormality of the `Q` factor, `Qᵀ Q = 1` — is proved in
[`NN.Proofs.Tensor.Basic.FactorizationsOrthonormal`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/Tensor/Basic/FactorizationsOrthonormal.lean)
by *unifying the executable variant with Mathlib's `gramSchmidt`* rather than re-deriving the
orthogonality induction by hand. Reading the columns of `A` as vectors of `EuclideanSpace ℝ (Fin m)`,
`Qcol_bridge` proves by strong induction that the `j`-th executable `Q` column equals Mathlib's
`gramSchmidtNormed ℝ` of the column map. The orthonormality then follows from Mathlib's
`gramSchmidtNormed_orthonormal'`, giving `Q_orthonormal` (`qₐ · q_b = δₐᵦ`), the matrix-level
`QT_mul_Q_eq_one`, and the full `IsQR` predicate `isQR_of_pos` (orthonormal `Q`, upper-triangular `R`,
`A = Q · R`).

The bridge rests on three small connectors over `ℝ`: the executable `dotFn`/`normFn` are the Euclidean
inner product and norm (`dotFn_eq_inner`, `normFn_eq_norm`), and `proj_normalize` shows the
un-normalized Gram–Schmidt projection term equals the normalized one (with no non-degeneracy
hypothesis). The positive-pivot assumption (`0 < R[j,j]`, full column rank) supplies the non-vanishing
of each `gramSchmidt` vector via `gn_ne_zero`. These connectors are stated generally enough to lift
into a future Mathlib matrix-level QR contribution.

# What remains

With Cholesky and QR fully reconstructed (`A = L · Lᵀ`, `A = Q · R`, `Qᵀ Q = 1`), the only properties
not available as a-priori theorems are the *iterative* ones: full diagonalization for the cyclic Jacobi
eigensolver and the SVD built on it. Mathlib v4.30.0 has no Jacobi convergence theory, so those remain
captured by the exact a-posteriori residual certificate above, never by `sorry`. The specification-level
facts the kernel methods rely on are independent of that step, so the CHD foundation is complete.
