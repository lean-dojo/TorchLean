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

# Solving the regularized system: verified `solve_variationnal`

The eigendecomposition route above gives `(K + γI)⁻¹` as an abstract identity. But CHD does not form
inverses; it *solves* the regularized system `(K + γI)·x = b`, and the SPD structure makes the direct
Cholesky route both faster and — crucially for verification — *exact*: because `K + γI` is symmetric
positive-definite, its Cholesky factorization is finite, so the whole solve carries no asymptotic
caveat. This is the second, complementary verified route to `solve_variationnal`, in
[`NN.Proofs.Tensor.Basic.FactorizationsSolve`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/Tensor/Basic/FactorizationsSolve.lean).

The solve is two triangular substitutions. Forward substitution `triSolveLowerFn` and back
substitution `triSolveUpperFn` are *exact*: for a lower- (resp. upper-) triangular matrix with nonzero
diagonal,

$$`(L\,y)_i = b_i \quad\text{and}\quad (U\,x)_i = c_i \qquad\text{for every } i.`

The key observation is that *no induction on the solved values is needed*: the entry `yᵢ` is defined
precisely to make row `i` balance, so unfolding it and using triangularity — the not-yet-visited and
structurally-zero terms drop out of the row dot product — gives the identity directly
(`triSolveLowerFn_mulVec`, `triSolveUpperFn_mulVec`). Each substitution is a `Function.update` fold
over the index list (`finRange n` forward, its reverse for back-substitution); two generic lemmas,
`foldl_update_read` and `foldl_update_stable`, capture the bookkeeping that the value written at index
`i` is never overwritten and earlier values are already in place.

Composing them through a Cholesky factor solves the SPD system exactly (`cholSolveFn_mulVec`):

$$`(L\,L^\top)\,x = b, \qquad x = \texttt{backSolve}\,L^\top\,(\texttt{forwardSolve}\,L\,b).`

Specializing `L` to the Cholesky factor of `K + γI` gives `solveRidgeFn_mulVec`: if the Cholesky
pivots of `K + γI` are positive — the success condition — then `solveRidgeFn K γ b` solves
`(K + γI)·x = b` *exactly*. The `RidgeSolve` example exercises this on a rank-deficient Gram kernel
`K = G·Gᵀ`: with `γ = 0.5` the residual is zero to machine precision, while the *negative control*
`γ = 0` hits a zero pivot on the singular `K` and diverges — regularization is what makes the solve
well-posed.

That success condition is now discharged, so the headline `solveRidgeFn_mulVec_of_posSemidef` is
*unconditional*: for a positive-semidefinite kernel `K` and `γ > 0`, `solveRidgeFn K γ b` solves
`(K + γI)·x = b` exactly with no pivot hypothesis. Two facts combine. First, `posDef_addScaledIdFn`
proves `K + γI` is positive-definite (via `Matrix.PosDef.one`, `Matrix.PosDef.smul`,
`Matrix.PosDef.posSemidef_add`) — genuinely SPD, exactly the regime where Cholesky succeeds. Second,
the *keystone* `choleskyFn_diag_pos_of_posDef` proves that a positive-definite matrix has
*strictly positive* executable Cholesky pivots (equivalently the radicand `A[j,j] − Σ_{k<j} L[j,k]² > 0` at each
step). The proof is the leading-principal Schur-complement fact, formalized as an *explicit
quadratic-form witness* so it needs no matrix inverse: by strong induction on `j`, the leading block
reconstructs from the pivots below `j` (`choleskyFn_dot_eq_local`), and back-substitution — the
`triSolveUpperFn` already proven correct here — produces a vector `z` with `z_j = 1` whose `A`-quadratic
form `zᵀ A z` *equals* the radicand; positive-definiteness (`Matrix.PosDef.dotProduct_mulVec_pos`)
forces `zᵀ A z > 0`. The `RidgeSolve` example also exhibits the keystone directly: the SPD `K + γI` has
all-positive pivots, while the singular `K` has a zero pivot — PosDef is necessary. Nothing here is an
unproved axiom.

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

# Faithfulness of the Jacobi run: orthogonality and orthogonal similarity

The three certificate theorems above are stated *conditionally* — they take the orthogonality
`Vᵀ V = 1` and the orthogonal-similarity identity `A = V · Af · Vᵀ` as hypotheses. Both are
*exact, finite, a-priori* facts about the executable `arrJacobiRun`, needing no convergence theory,
and
[`NN.Proofs.Tensor.Basic.FactorizationsJacobi`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/Tensor/Basic/FactorizationsJacobi.lean)
proves them, discharging the hypotheses for the real solver output.

The development bridges the strict `Array (Array ℝ)` representation the loop runs over to Mathlib
`Matrix` via `toM`, with `toM_matMul`/`toM_tr`/`toM_id` showing the array operations realise the
matrix ones. The single genuinely-new ingredient is `givens_orthogonal`: each rotation
`arrGivens n p q c s` with `c² + s² = 1` is an orthogonal matrix (`Jᵀ J = 1`), proved by reducing the
column dot products to the `c² + s² = 1` identity (`givens_normSq`) for the diagonal blocks and to
orthogonality of distinct standard basis vectors elsewhere. From it, the loop invariant
`JacInv A₀ (A, V) := Vᵀ V = 1 ∧ A₀ = V · A · Vᵀ` is preserved by one rotation (`jacInv_rotate` — the
no-op branch trivially, the rotating branch because conjugating by an orthogonal `J` cancels in
`J Jᵀ = 1`), hence by a whole sweep (`jacInv_sweep`, a `List.foldlRecOn` over `jacobiPairs`) and the
whole run (`jacInv_run`, starting from `(A, I)` where the invariant is immediate).

Specialised to the `symEigJacobiSpec` output, this gives the two premises as theorems with no
hypotheses: `jacobi_orthogonal` (`Vᵀ V = 1`) and `jacobi_similarity` (`A = V · Af · Vᵀ`).
Feeding them into the certificate yields the *unconditional* restatements
`symEigJacobi_reconstruction_residual`, `symEigJacobi_frobenius_residual`, and
`symEigJacobi_isSymEig_of_diagonal`: the residual identity and the zero-residual-limit correctness now
hold for the actual returned `(Λ, V)` outright. So the returned `V` is a genuine orthogonal matrix and
`Af` a genuine orthogonal similarity of the input *regardless of how far the sweeps have converged* —
the only thing the residual certificate still defers to runtime is the *size* of the off-diagonal
mass, never the algebraic faithfulness of the decomposition.

# Per-rotation progress: the off-diagonal mass decreases

Faithfulness says the residual *equals* the off-diagonal mass of `Af`; it does not say that mass ever
goes *down*. The classical Jacobi progress identity, proved in
[`NN.Proofs.Tensor.Basic.FactorizationsJacobiDecrease`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/Tensor/Basic/FactorizationsJacobiDecrease.lean),
is exactly that statement at the level of a single rotation. For a symmetric `A`, conjugating by the
Givens rotation that *annihilates* the pivot `(p, q)` decreases the squared off-diagonal mass by
exactly `2 · A[p,q]²`:

$$`\bigl\|\operatorname{offDiag}(J^\top A J)\bigr\|_F^2 = \bigl\|\operatorname{offDiag} A\bigr\|_F^2 - 2\,A[p,q]^2.`

This is `jacobi_off_decrease`, and it rests on two exact facts. First, *orthogonal similarity
preserves the total Frobenius mass* (`frobSq_orthogonal_conj`): `‖Jᵀ A J‖² = ‖A‖²`, since
`trace((Jᵀ A J)ᵀ (Jᵀ A J)) = trace(Aᵀ A)` after the `J Jᵀ = 1` cancellation. Splitting that total as
diagonal-plus-off-diagonal mass (`frobSq_eq_diagSq_add_offSq`) shows that driving the off-diagonal
down is *the same thing* as driving the diagonal up. Second, the rotation only mixes rows and columns
`p, q`, so the diagonal mass changes by `A'[p,p]² + A'[q,q]² − A[p,p]² − A[q,q]²`; the explicit
conjugation entries (`givens_conj_pp`, `givens_conj_qq`, `givens_conj_pq`, computed from the Givens
columns via the support lemmas) plus the `2×2` block-Frobenius identity — itself just
`frobSq_orthogonal_conj` specialised to `Fin 2` — turn that, under `c² + s² = 1` and the annihilation
`A'[p,q] = 0`, into precisely `2 · A[p,q]²`. The annihilation is the defining equation the
Golub–Van Loan rotation angle solves, and `givens_conj_pq` exhibits the pivot entry whose vanishing
it is. The executable witnesses in
[`NN.Examples.Factorization.JacobiDecrease`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Factorization/JacobiDecrease.lean)
confirm the identity numerically (one rotation takes the off-diagonal mass `6 → 4 = 6 − 2·1²` with
total mass conserved at `35`) and show its hypotheses biting: a wrong-angle rotation misses the
decrease, a non-orthogonal one breaks mass invariance.

# Aggregate rate: linear convergence of the classical strategy

The per-rotation identity removes `2 · A[p,q]²` of off-diagonal mass per step. Turning that into an
*aggregate* rate — a factor by which the mass falls each step, and hence a bound on how many steps are
needed — requires a lower bound on the pivot. For the *classical* strategy, which always annihilates
the *largest* off-diagonal entry, that bound is elementary, and
[`NN.Proofs.Tensor.Basic.FactorizationsJacobiRate`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/Tensor/Basic/FactorizationsJacobiRate.lean)
proves it exactly over `ℝ`. There are `n² − n` off-diagonal positions, so the largest one carries at
least the average share of the mass (`offSq_le_count_mul_max`):

$$`A[p,q]^2 \;\ge\; \frac{\bigl\|\operatorname{offDiag} A\bigr\|_F^2}{n^2 - n}.`

Substituting this into the per-rotation decrease gives a genuine *linear contraction*
(`jacobi_off_decrease_classical`):

$$`\bigl\|\operatorname{offDiag}(J^\top A J)\bigr\|_F^2 \;\le\; \Bigl(1 - \tfrac{2}{n^2 - n}\Bigr)\,\bigl\|\operatorname{offDiag} A\bigr\|_F^2,`

a fixed factor strictly below `1`. A fixed-factor contraction iterates to a geometric bound
(`geom_bound_of_contraction`: `aₖ ≤ ρᵏ · a₀`) and, since `offSq ≥ 0` (`offSq_nonneg`) and the factor
is `< 1`, drives the off-diagonal mass to zero (`tendsto_zero_of_contraction`). So the classical
Jacobi eigenvalue algorithm provably converges, with an a-priori geometric rate. The geometric
machinery is stated for an *arbitrary* per-step factor `ρ`, so it is exactly the slot a future cyclic
per-sweep bound would fill. The executable witnesses in
[`NN.Examples.Factorization.JacobiRate`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Factorization/JacobiRate.lean)
exhibit the contrast on a matrix with one dominant entry (`A[0,1] = 5`): annihilating the largest
pivot collapses the off-diagonal mass `50.04 → 0.04`, far under the guaranteed `33.36`, while
annihilating a tiny pivot `A[0,2] = 0.1` removes only `0.02` and stays *above* the guaranteed bound —
the numerical teeth of the largest-pivot hypothesis.

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

With Cholesky and QR fully reconstructed (`A = L · Lᵀ`, `A = Q · R`, `Qᵀ Q = 1`), the Jacobi run
proved faithful — `V` orthogonal and `A = V · Af · Vᵀ` exactly, so the residual certificate holds
*unconditionally* for the real solver output — the *per-rotation* progress proved exactly (each
annihilating rotation removes `2 · A[p,q]²` of off-diagonal mass), and the *aggregate* rate of the
*classical largest-pivot* strategy proved to be geometric (linear contraction by `1 − 2/(n²−n)`,
iterating to convergence), the one property still not available as an a-priori theorem is the
aggregate rate *for the cyclic ordering the solver actually uses*: that visiting pivots in fixed
row-major order, rather than always the largest, still drives the off-diagonal mass to zero fast
enough that finitely many sweeps suffice. The gap is precise. The classical bound rests on the
largest pivot carrying at least the average share of the mass; a cyclically-chosen pivot need not, so
its single-step decrease can fall arbitrarily short of `2·‖offDiag A‖²/(n²−n)` (and a later rotation
in the same sweep can refill an entry an earlier one zeroed). Summing the per-rotation decrease over a
sweep is exact; what is research-grade is bounding the *sum of the cyclic pivots* below in terms of
the total off-diagonal mass — the Forsythe–Henrici / Schönhage convergence result. Mathlib v4.30.0 has
no cyclic-Jacobi convergence theory, so that cyclic rate remains captured by the exact a-posteriori
residual certificate above — bounded numerically by the `assertLt` checks on concrete inputs — never
by `sorry`; and the geometric machinery (`geom_bound_of_contraction`, `tendsto_zero_of_contraction`)
is stated for an arbitrary per-step factor, ready to consume such a bound the moment it exists.

On the *direct* solve route there is nothing left to do, because it avoids the eigensolver entirely.
The kernel-ridge solve `(K + γI)·x = b` is proved correct *exactly* (via verified forward/back
substitution and Cholesky), the regularized matrix is proved SPD for `γ > 0` (`posDef_addScaledIdFn`),
and the positive-pivot success condition is now discharged from that SPD fact by the keystone
`choleskyFn_diag_pos_of_posDef` (the radicand `A[j,j] − Σ_{k<j} L[j,k]² > 0`, proved via the explicit
Schur-complement quadratic-form witness). Composing them, `solveRidgeFn_mulVec_of_posSemidef` makes the
verified `solve_variationnal` *unconditional* for any positive-semidefinite kernel `K` and `γ > 0`, with
no pivot hypothesis remaining.

Everything else is exact: the algebraic faithfulness of the decomposition (orthogonality, orthogonal
similarity, the residual identity, the per-rotation decrease, the classical-strategy linear rate, and
correctness in the zero-residual limit), the finite Cholesky/QR reconstructions, and the
Cholesky-based regularized solve are proved, and the specification-level facts the kernel methods rely
on are independent of the convergence step, so the CHD foundation is complete.
