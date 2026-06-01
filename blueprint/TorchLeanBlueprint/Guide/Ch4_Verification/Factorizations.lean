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

Two capstones close the solve story. First, the keystone and the reconstruction theorem combine into
`cholesky_posDef`: for *any* positive-definite `A`, the executable `choleskyFn` is — with no pivot,
symmetry, or success hypothesis — a genuine Cholesky factor (`A = L · Lᵀ`, lower-triangular, strictly
positive diagonal). This is the unconditional statement "`choleskyFn` computes the Cholesky
factorization of an SPD matrix". The `RidgeSolve` example exhibits both directions: the SPD `K + γI`
reconstructs to machine precision, while an *indefinite* matrix hits a `√(negative) = NaN` pivot and
fails — positive-definiteness, not mere symmetry, is the hypothesis the capstone needs. (A singular
PSD `K` still reconstructs, with a zero pivot; the zero pivot breaks only the *solve*, which is exactly
the dichotomy the keystone isolates.)

Second, `solveRidgeFn_eq_inv_mulVec` identifies the computed solve with the closed form CHD specifies:

$$`\texttt{solveRidgeFn}\,K\,\gamma\,b \;=\; (K + \gamma I)^{-1} b.`

The solve theorems prove `(K + γI)·x = b`; positive-definiteness makes `K + γI` invertible
(`Matrix.PosDef.isUnit`), so that equation pins `x` down *uniquely* and forces equality with the
inverse — closing the loop to `solve_variationnal`'s `(K + γI)⁻¹ b` *without the algorithm ever forming
an inverse*. The `RidgeSolve` example makes this concrete: solving against each standard basis vector
`eⱼ` produces column `j` of `(K + γI)⁻¹`, and the assembled matrix satisfies
`(K + γI) · (K + γI)⁻¹ = I` to machine precision, every column coming from the verified Cholesky solve.

# The CHD routines: variational solve, `find_gamma`, and `Z_test`

The two solve routes above invert `K + γI`. But CHD's `perform_regression_and_find_gamma`
(`interpolatory.py`) does not stop there: it takes the *eigendecomposition* route — `eigh(K)` once, then
three routines computed from the eigenpairs `(λ, V)`. They share one arithmetic core: the *projected
data* `Pga = Vᵀ ga` and the *shrinkage coefficients* `rᵢ = γ/(λᵢ + γ)`.
[`NN.Proofs.Tensor.Basic.FactorizationsVariational`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/Tensor/Basic/FactorizationsVariational.lean)
identifies what each computes; everything is exact over `ℝ`, proved from the *specification*
`IsSymEig` (so it holds for whatever eigendecomposition the solver returns, asymptotic or not).

The variational solution `solve_variationnal` returns, in eigendecomposition form,
`yb = -V (Pga/(λ+γ))`. `variationalSolveFn_eq_neg_inv_mulVec` proves this *is* the regularized-inverse
solve, directly from `add_smul_inv`:

$$`\texttt{variationalSolveFn}\,\Lambda\,V\,\gamma\,ga \;=\; -\,(K + \gamma I)^{-1} ga.`

So the eigendecomposition route and the Cholesky route compute the *same* `solve_variationnal`:
`variationalSolveFn_eq_neg_solveRidgeFn` proves `variationalSolveFn = -\,\texttt{solveRidgeFn}` for a
positive-semidefinite kernel `K` and `γ > 0` — two independent implementations agreeing on the one
closed form `-(K + γI)⁻¹ ga`. The supporting fact `IsSymEig.eigenvalues_nonneg` (a PSD matrix's
eigenvalues are `≥ 0`, via the congruence `Vᵀ A V` being positive-semidefinite) discharges the
`λᵢ + γ ≠ 0` side condition from `γ > 0`.

`find_gamma` and `Z_test` share a second quantity, the `noise` level. `varNoiseFn_eq_ratio` exhibits it
as a spectral quadratic-form ratio:

$$`\texttt{noise} \;=\; \frac{\sum_i (Pga_i\, r_i)^2}{\sum_i Pga_i^2\, r_i},
\qquad r_i = \frac{\gamma}{\lambda_i + \gamma}.`

`find_gamma` minimises this functional over `γ`; `Z_test` evaluates it on random Gaussian data. The
load-bearing invariant is that the `noise` is a genuine *fraction*: for a PSD spectrum (`λᵢ ≥ 0`) and
`γ > 0`, each coefficient satisfies `0 < rᵢ ≤ 1` (`ridgeCoeffFn_pos`, `ridgeCoeffFn_le_one`), so
`rᵢ² ≤ rᵢ` makes the numerator dominated by the denominator, giving

$$`0 \;\le\; \texttt{noise} \;\le\; 1`

(`varNoiseFn_nonneg`, `varNoiseFn_le_one`). Finally, the `Z_test` statistic depends on the kernel only
through its *spectrum*: replacing the data by `ga = V z` makes `V` cancel, `projFn V (V z) = z`
(`projFn_mulVec_self`), so `varNoiseFn Λ γ (projFn V (V z)) = varNoiseFn Λ γ z`
(`varNoiseFn_projFn_mulVec`). This is the deterministic content of "the `Z_test` null distribution
depends only on the eigenvalues"; the *distributional* step — Gaussian sampling and the 5%/95%
percentiles — is taken up later (*The `Z_test` distributional layer*), where the finite-sample
false-positive rate is bounded and the i.i.d.-Gaussian null law is shown to be a probability measure
on `[0,1]`, leaving only the asymptotic quantile-consistency to runtime.

The `Variational` example confirms all four on a concrete SPD kernel: `(K + γI)·yb = -ga` and
`yb = -\texttt{solveRidgeSpec}` to machine precision, `noise ∈ [0,1]`, and the spectral invariance
`noise(V z) = noise(z)` to machine precision. Its *negative controls* show the hypotheses biting:
feeding the *wrong* eigenvectors (the identity in place of `V`) makes the solve residual large, and
`γ < 0` pushes the `noise` outside `[0,1]` — so the true eigendecomposition and `γ > 0` are both
necessary.

# Building the kernel: the linear mode is positive-semidefinite

Every result above takes the kernel `K` as input *under the hypothesis* that it is positive
-semidefinite — the solve needs `K + γI` to be SPD, the noise bound needs `λᵢ ≥ 0`. But CHD does not
receive `K`; it *builds* it from data (`Modes/kernels.py`). Discharging that standing `PosSemidef`
hypothesis for the kernels CHD actually constructs is the same move as the positive-pivot keystone:
turn an assumed precondition into a theorem.
[`NN.Proofs.Tensor.Basic.FactorizationsKernels`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/Tensor/Basic/FactorizationsKernels.lean)
takes the first and simplest mode. The *linear* kernel is

$$`K[i,j] = 1 + \texttt{scale}\cdot\langle \Phi_i, \Phi_j\rangle,
\qquad K = \mathbf{1}\mathbf{1}^\top + \texttt{scale}\cdot \Phi\,\Phi^\top,`

with `Φ` the column-masked data (`which_dim`). `linearKernelFn_posSemidef` proves this is symmetric
positive-semidefinite whenever `scale ≥ 0`: the all-ones matrix `𝟙𝟙ᵀ` is a rank-one Gram (hence PSD),
`Φ Φᵀ` is a Gram matrix (PSD by `posSemidef_self_mul_conjTranspose`), `scale ≥ 0` keeps the scaling
PSD (`PosSemidef.smul`), and `PosSemidef.add` closes the sum. Symmetry (`linearKernelFn_symm`) is then
a corollary of `PosSemidef.isHermitian`. Composed with the solve development, this makes
`solveRidgeSpec (linearKernelSpec X w scale) γ b` an *unconditional* exact solve for `γ > 0` — no PSD
hypothesis left to assume. The `LinearKernel` example confirms `K = Kᵀ`, the match with the CHD
`LinearMode` formula, all-nonnegative Jacobi eigenvalues (with feature masking preserved), and the
downstream exact ridge solve; its *negative control* takes `scale = -1`, where `𝟙𝟙ᵀ − Φ Φᵀ` is
indefinite and a Jacobi eigenvalue goes negative — so `scale ≥ 0` is necessary.

# Building the kernel: the quadratic mode is positive-semidefinite

The *quadratic* mode (`QuadraticMode.vectorized_kernel`) is the second kernel CHD builds:

$$`K[i,j] = \texttt{scale}\cdot(\alpha + \langle \Phi_i, \Phi_j\rangle)^2 + (1 - \alpha^2\texttt{scale}).`

Squaring and collecting terms makes the PSD structure explicit:

$$`K = \mathbf{1}\mathbf{1}^\top + (2\,\texttt{scale}\,\alpha)\cdot \Phi\,\Phi^\top
       + \texttt{scale}\cdot\bigl(\Phi\,\Phi^\top \odot \Phi\,\Phi^\top\bigr),`

a sum of three PSD pieces: the all-ones Gram, a nonnegative multiple of the data Gram `Φ Φᵀ`, and a
nonnegative multiple of its *Hadamard square* `Φ Φᵀ ⊙ Φ Φᵀ`. The last is PSD by the *Schur product
theorem* `PosSemidef.hadamard` (the Hadamard product of PSD matrices is PSD), which Mathlib v4.30.0
provides. `quadraticKernelFn_posSemidef` assembles the three with `PosSemidef.add`/`PosSemidef.smul`
and proves `K` PSD whenever `scale ≥ 0` *and* `alpha ≥ 0` — both conditions are real: the
`QuadraticKernel` example's two *negative controls* take `alpha = -1` and `scale = -1`, and each makes
a Jacobi eigenvalue go negative. As with the linear mode, this discharges the standing `PosSemidef`
hypothesis, so `solveRidgeSpec (quadraticKernelSpec X w scale alpha) γ b` is an unconditional exact
solve for `γ > 0`, and `quadraticKernelFn_symm` gives symmetry from `PosSemidef.isHermitian`.

# Building the kernel: the Gaussian mode is positive-semidefinite

The third and last mode is the *Gaussian* (fully-nonlinear) kernel. CHD's `GaussianMode` builds, per
feature, the Gaussian `exp(-(X_{i,d}-X_{j,d})^2/2\ell^2)` and takes their masked product:

$$`K[i,j] = \texttt{scale}\cdot\prod_{d} \bigl(1 + w_d\,\exp(-(X_{i,d}-X_{j,d})^2/2\ell^2)\bigr).`

Unlike the linear and quadratic modes, the Gaussian has *no finite algebraic PSD identity* — `exp` is a
genuine limit. The textbook proof is Schoenberg/Bochner, which Mathlib v4.30.0 does not have. But there
is an elementary route that reuses the *same* Schur product theorem, and
`gaussianKernelFn_posSemidef` carries it out. It rests on one genuinely new, independently useful
lemma and three assembly steps:

- *The PSD cone is closed under entrywise limits* (`posSemidef_of_tendsto`): if real PSD matrices `A_k`
  converge entrywise to `B`, then `B` is PSD. The quadratic form `x^\top M x` is a finite polynomial in
  the entries, hence continuous, and `\{y \mid 0 \le y\}` is closed — so `0 \le x^\top A_k x` passes to
  the limit. This is the only piece Mathlib lacked, and it belongs in Mathlib.
- *The entrywise exponential of a PSD matrix is PSD* (`posSemidef_map_exp`): writing
  `\exp\circ G = \sum_k G^{\odot k}/k!` (Hadamard powers), each `G^{\odot k}` is PSD by the Schur
  product theorem, each partial sum is PSD (a finite sum of PSD matrices), and the partial sums converge
  entrywise to `\exp\circ G` (the real exponential series) — so the limit is PSD by the lemma above.
- *A single Gaussian matrix is PSD* (`posSemidef_gaussianCol`): for `c \ge 0`, the matrix
  `\exp(-c\,(y_i-y_j)^2)` factors as the diagonal congruence
  `D\,(\exp\circ(2c\,yy^\top))\,D^\top` with `D = \operatorname{diag}(\exp(-c\,y_i^2))`; the middle
  factor is the entrywise exponential of the (PSD, rank-one) Gram `yy^\top`, and congruence preserves
  PSD.
- *Each feature factor and their product* (`gaussianKernelFn_posSemidef`): `\mathbf{1}\mathbf{1}^\top +
  w_d\cdot\text{Gaussian}_d` is PSD for `w_d \ge 0`, and the product over features is a Hadamard product
  of PSD matrices — PSD by the Schur product theorem again. Scaling by `\texttt{scale} \ge 0` finishes.

So `K` is PSD whenever `scale ≥ 0` and the mask is nonnegative (`w ≥ 0`) — discharging the standing
`PosSemidef` hypothesis for the Gaussian mode, and `gaussianKernelFn_symm` gives symmetry from
`PosSemidef.isHermitian`. The `GaussianKernel` example confirms `K = Kᵀ`, the match with the CHD
`GaussianMode` product formula, all-nonnegative Jacobi eigenvalues (with feature masking preserved),
and the downstream exact ridge solve; its two *negative controls* take `scale = -1` and a *negative
mask weight* `w = [-2,0]` (whose factor `1 - 2\exp(-\Delta^2/2\ell^2)` drives the diagonal below zero),
each producing a negative eigenvalue — so `scale ≥ 0` and `w ≥ 0` are both necessary.

With the linear, quadratic, and Gaussian modes all discharged, *every kernel CHD builds is now
PSD-verified*: there is no `PosSemidef` hypothesis left to assume anywhere in the solve / `find_gamma` /
`Z_test` development.

# The discovery decision layer: turning `noise` into graph structure

Everything above produces *numbers* — a kernel, its eigendecomposition, and the `noise` level
(`varNoiseFn`, proven to be a fraction in `[0,1]`). CHD's outer *discovery loop*
(`decision.py`, `_GraphDiscoveryMain.py`) turns those numbers into the actual hypergraph: which
ancestors a node depends on, through which kernel mode.
[`NN.Proofs.Tensor.Basic.FactorizationsDecision`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/Tensor/Basic/FactorizationsDecision.lean)
formalizes the four deterministic choices the loop makes and proves each one's selection guarantee. They
are comparisons over finite data, so the specs (`Spec.argMinFn`, `Spec.kernelChooserFn`, …) mirror the
Python verbatim and run over `Float`; the proofs are over `ℝ`, bridged from the `Context` order test
(`gtBool`/`ltBool`) to the real `<` by `gtBool_eq_decide`.

The backbone is a single generic fold-selection lemma (`foldl_select`): the running-best `List.foldl`
that both `np.argmin` and `np.argmax` compile to returns a `le`-extremal index over the whole family, for
*any* preorder `le` whose strict part the `Bool` test decides. Instantiating `le := (· ≤ ·)` and
`(· ≥ ·)` gives the two endpoints:

- *Prune the least-activated ancestor.* Each step of `helper_functions.step` drops the candidate of
  smallest *activation* (`min_activation = np.argmin(activations)`); `argMinFn_le` proves the fold returns
  a global minimizer, `argMaxFn_le` the dual.
- *Choose the kernel mode that admits an edge.* `MinNoiseKernelChooser` calls a kernel *valid* when its
  `noise` falls below its `Z_low`, and returns the valid kernel of least `noise`
  (`argmin(np.where(valid, noises, 2))`), or "no ancestor" if none is valid. `kernelChooserFn_eq_some`
  and `kernelChooserFn_eq_none` prove it *sound and complete*: it returns `some s` with `s` itself valid
  and of least `noise` among *all* valid kernels exactly when some kernel is valid, and `none` otherwise.
  The `2` sentinel that suppresses invalid kernels only works because `noise ≤ 1` — which is exactly the
  verified `varNoiseFn_le_one`, threaded in as the hypothesis `hbound`. The bound proved two sections ago
  is what makes the decision correct.
- *Report the pruning iteration of largest `noise` jump.* `MaxIncrementModeChooser` takes the `argmax`
  of the successive `noise` increments (with `1 − noise_last` appended); `modeChooserFn_ge` proves the
  reported iteration has the maximal increment.
- *Stop when every ancestor is pruned.* `allPrunedFn_iff` proves the stopping test
  `np.all(active_modes == 0)` holds iff every entry is zero.

So the loop's structural decisions are not heuristics layered on top of unverified arithmetic: each is a
proved-correct selection over the `noise` statistic whose `[0,1]` range was itself proved. The
`Discovery` example runs all four on concrete data — argmin picks the least-activated ancestor (and not
the most-activated one), the chooser selects the unique valid kernel, takes least noise among two valid
ones, and reports `none` when all are invalid, the mode chooser picks the largest-jump iteration, and the
stopping rule fires only on the all-zero mask — and then closes the stack end-to-end: it builds an SPD
kernel, eigendecomposes it, and runs a `find_gamma`-style sweep that feeds the verified `varNoiseSpec` at
several `γ` straight into `argMinFn`, selecting the least-noise regularization (the smallest `γ`, every
swept noise landing in `[0,1]` as proved).

# The `Z_test`: a null-distribution significance threshold

The kernel chooser of the previous section asks whether the observed `noise` falls below a threshold
`Z_low`. Where does `Z_low` come from? It is not a hand-set constant — it is the *5th percentile of the
null distribution* of the very same `noise` statistic. CHD's `Z_test` (`interpolatory.py`) draws `N`
standard-Gaussian samples, scores each one's `noise` with the same `varNoiseFn`, sorts the `N` values,
and reads off the 5th and 95th percentiles as `Z_low` and `Z_high`. An edge is *significant* — a real
dependency rather than fitting noise — when the observed `noise` falls below `Z_low`, i.e. strictly
inside the lower tail of what random data would produce.

[`NN.Proofs.Tensor.Basic.FactorizationsDecision`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/Tensor/Basic/FactorizationsDecision.lean)
formalizes this statistical layer. The spec `Spec.zLowFn` / `Spec.zHighFn` mirror `Z_test`: the random
draws are an explicit family `samples : Fin N → Fin n → α` (the caller's randomness, exactly as CHD
threads a PRNG key), each is scored by `Spec.sampleNoisesFn` (the *same* `varNoiseFn` again), and the
percentiles are order statistics `Spec.kthSmallestFn` — the `k`-th entry of the list sorted by the
`Context` order. Over `ℝ` that sort key is the real `≤` (`leBool_eq_le`), so Mathlib's
`sortedLE_mergeSort` supplies sortedness and `mergeSort_perm` supplies membership.

The payoff is that the threshold is *well-posed*, and provably so. The keystone is that the `[0,1]`
bound governing the data noise governs *every null sample too* — it is the same `varNoiseFn`. So:

- `sampleNoisesFn_nonneg` / `_le_one` — each of the `N` null noises is a genuine fraction in `[0,1]`,
  directly from `varNoiseFn_nonneg` / `varNoiseFn_le_one`.
- `zLowFn_nonneg` / `zLowFn_le_one` and the `zHighFn` pair — hence each percentile lies in `[0,1]`,
  because an order statistic is one of the sampled values (`kthSmallestFn_mem`).
- `zLowFn_le_zHighFn` — and `Z_low ≤ Z_high`, because a 5th percentile never exceeds a 95th. This is
  *pure order-statistic monotonicity* (`kthSmallestFn_mono`): the underlying list is sorted ascending
  and `⌊0.05 N⌋ ≤ ⌊0.95 N⌋`. The comparison window `Z_low ≤ noise ≤ Z_high` the loop uses (the
  "no anomaly" band of `_GraphDiscoveryMain.py`) is therefore non-degenerate.

Finally `zTest_admits_edge` ties the statistical verdict back to the decision layer: when the observed
`noise` clears `Z_low` (`zSignificantFn = true`), the single-kernel `MinNoiseKernelChooser` admits the
edge — returns `some 0`. The `noise ≤ 1` ceiling that proof needs is, once more, the verified
`varNoiseFn_le_one`. The whole statistical decision thus rests on the one spectral bound proved three
sections ago. The `Discovery` example exhibits the layer end-to-end: it builds the null distribution
from a real eigendecomposition, checks `0 ≤ Z_low ≤ Z_high ≤ 1`, shows data aligned with the *dominant*
eigenvector (smallest shrinkage noise) clears the lower tail and is flagged significant, and confirms a
high noise — and a noise sitting at the upper tail — are both correctly rejected.

# The `Z_test` distributional layer

The section above proved the thresholds *well-posed*; what it deferred was the *distributional*
question — what `Z_low` being "the 5th percentile" actually buys, and what it means that the draws
are Gaussian. `FactorizationsZTest` closes that gap in two honestly-provable halves.

*Finite-sample calibration (counting).* The operational promise of a 5th-percentile threshold is a
bound on its *own* false-positive rate: of the `N` null draws, only a `5%` minority should beat it.
That is exactly true, and exact (not asymptotic): in an ascending-sorted list at most `k` entries lie
strictly below the `k`-th, so — since sorting is a permutation and `List.countP` is permutation-invariant
— at most `⌊N/20⌋` of the null noises fall below `Z_low` (`zLow_null_exceedance_le`) and at most
`N-1-⌊19N/20⌋` rise above `Z_high` (`zHigh_null_exceedance_le`). These rest on the same sortedness
(`List.sortedLE_mergeSort`) that gave `Z_low ≤ Z_high`, now counted rather than compared. The `Discovery`
example makes the numbers concrete: across `N = 20` draws exactly `1` (`= ⌊20/20⌋`) sits below `Z_low` and
`0` above `Z_high`, while the slack `Z_high` admits `19` — a negative control showing the `5%` calibration
is specific to `Z_low`, not an artifact of any threshold.

*The Gaussian null law (measure theory).* CHD draws each null sample i.i.d. standard Gaussian. We model
one draw as `nullGaussian n := Measure.pi (fun _ => gaussianReal 0 1)`, the product of `n` standard
normals on `Fin n → ℝ` — a genuine probability measure. The per-draw statistic `noiseMap` (the same
`varNoiseFn ∘ projFn` the data is scored by, identified with CHD's `sampleNoisesFn` by
`sampleNoisesFn_eq_noiseMap`) is *measurable* (`measurable_noiseMap`: a ratio of finite sums of products
of the draw coordinates), so its pushforward `noiseLaw` is a probability measure
(`IsProbabilityMeasure`). And because every draw's noise lies in `[0,1]` — the verified
`varNoiseFn_nonneg` / `varNoiseFn_le_one`, now lifted to the law — that law is *concentrated on `[0,1]`*:
`noiseLaw_Icc_eq_one` shows it assigns full mass to `[0,1]`. So `Z_low`/`Z_high` are percentiles of a
bona fide `[0,1]`-valued random variable, not of an unconstrained sample.

*The i.i.d. scaffold for the asymptotic step.* `FactorizationsZAsymptotic` takes the first concrete
step toward that asymptotic calibration — the *pointwise* half, which a survey of
`Mathlib.Probability` v4.30.0 shows is in fact assemblable sorry-free (a re-scope from the earlier
"absent from Mathlib" note). It lifts the single null draw to the i.i.d. *sequence*
`nullSeqGaussian n := Measure.infinitePi (fun _ : ℕ => nullGaussian n)` on `ℕ → (Fin n → ℝ)`, and
defines `nullNoise Λ V γ i ω := noiseMap Λ V γ (ω i)` — the same measurable `noiseMap`, read off the
`i`-th coordinate. The coordinate projections are independent under the product measure
(`iIndepFun_infinitePi`) and composing with `noiseMap` preserves it (`nullNoise_iIndepFun`, with the
pairwise corollary `nullNoise_pairwise_indepFun`); each projection is measure-preservingly one
standard-Gaussian draw, so every `nullNoise i` has the *same* law `noiseLaw` (`nullNoise_hasLaw`,
`nullNoise_identDistrib`); and every draw lies in `[0,1]` (`nullNoise_mem_Icc`) hence is integrable
(`integrable_nullNoise`). That is exactly the i.i.d.-bounded-integrable triple — `hint`, `hindep`,
`hident` — that the strong law of large numbers (`strong_law_ae_real`) and the Hoeffding tail consume.
This scaffold is the only genuinely new measure-theory plumbing; the empirical-CDF consistency
(Glivenko–Cantelli via the SLLN) and the per-`t` concentration rate `2 exp(-2 N ε²)` (Hoeffding) are
applications of it.

*Pointwise consistency of the empirical CDF (step b).* The first such application is now proved,
`empCDF_tendsto_cdf`. Fix a threshold `t`. The threshold indicators
`nullBelow Λ V γ t i ω := (Set.Iic t).indicator 1 (nullNoise Λ V γ i ω)` — the events `1{noiseᵢ ≤ t}`
— inherit the scaffold's i.i.d. structure: composing each independent, identically-distributed draw
with the measurable indicator of `Iic t` preserves both (`nullBelow_pairwise_indepFun`,
`nullBelow_identDistrib`), and they are `[0,1]`-valued hence integrable (`integrable_nullBelow`).
Their common mean is pinned by a short `HasLaw.integral_comp` computation that pushes the indicator
through `noiseLaw`: `∫ ω, nullBelow Λ V γ t 0 ω = (noiseLaw Λ V γ).real (Iic t) = cdf (noiseLaw Λ V γ) t`
(`integral_nullBelow_zero`) — so the empirical CDF is literally the Monte-Carlo estimator of the null
CDF. Feeding the `hint`/`hindep`/`hident` triple to `strong_law_ae_real` then yields, almost surely,
`empCDF Λ V γ N t ω → cdf (noiseLaw Λ V γ) t` as `N → ∞`, where
`empCDF Λ V γ N t ω := (∑ i ∈ range N, nullBelow Λ V γ t i ω) / N`. That is the *pointwise*
Glivenko–Cantelli theorem, sorry-free over Mathlib v4.30.0. The executable `Discovery` examples
exercise its computable shadow — the growing-prefix running mean `F̂_N(t)` settling toward the
full-sample estimate of `cdf noiseLaw t`.

*What is honestly left.* What stays genuinely research-grade is the *uniform* Glivenko–Cantelli
(`sup_t |F̂_N - cdf| → 0`) and the full *DKW–Massart* inequality with its sharp constant `2` over
the supremum — both need the bracketing / VC-class chaining Mathlib v4.30.0 lacks — and the
*exchangeability rank rate* `k/(N+1)` for a fresh null draw, which needs a symmetric-group
rank-distribution argument also absent. Those are stated as the open frontier, never stubbed with
`sorry`. The finite-sample false-positive *bound* above is the exact, non-asymptotic statement the
test actually guarantees, and the pointwise scaffold is the sorry-free bridge toward the asymptotic
statement.

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
no pivot hypothesis remaining. The loop to the CHD specification is closed by
`solveRidgeFn_eq_inv_mulVec`, which upgrades the solve identity `(K + γI)·x = b` to the closed form
`x = (K + γI)⁻¹ b` (uniqueness from invertibility), and by `cholesky_posDef`, which states
unconditionally that the executable Cholesky *is* the factorization of any SPD matrix.

Everything else is exact: the algebraic faithfulness of the decomposition (orthogonality, orthogonal
similarity, the residual identity, the per-rotation decrease, the classical-strategy linear rate, and
correctness in the zero-residual limit), the finite Cholesky/QR reconstructions, and the
Cholesky-based regularized solve are proved, and the specification-level facts the kernel methods rely
on are independent of the convergence step. The three concrete CHD routines built on them are now
identified too: the eigendecomposition-form `solve_variationnal` equals `-(K + γI)⁻¹ ga` and agrees
with the Cholesky route, and the `noise`/`find_gamma`-loss/`Z_test` statistic is a spectral ratio
provably in `[0,1]` that depends on the kernel only through its spectrum. The kernel build itself
is now PSD-verified for *all three* CHD modes — linear, quadratic, and Gaussian — so the standing
`PosSemidef` hypothesis is discharged from data, not assumed, even for the fully-nonlinear kernel. And
the *discovery decision layer* on top — the kernel chooser, the activation prune step, the mode
chooser, and the stopping rule — is now proved sound and complete, with the chooser's correctness
resting directly on the verified `noise ≤ 1` bound, so the structural decisions are proved selections
over a statistic whose range was itself proved. The `Z_test` *significance thresholds* are now proved
well-posed too: `Z_low` and `Z_high` are order statistics of the null `noise` distribution, each
inheriting the `[0,1]` bound from the shared `varNoiseFn`, with `Z_low ≤ Z_high` by order-statistic
monotonicity — and the verdict `noise < Z_low` is shown to feed `MinNoiseKernelChooser`. The
*distributional* layer of the `Z_test` is now partly proved too: the threshold's finite-sample
false-positive rate is bounded exactly (`≤ 5%` of the null draws beat `Z_low`,
`zLow_null_exceedance_le`; symmetrically for `Z_high`), and — modelling the draws as i.i.d. standard
Gaussian — the null `noise` law is a genuine probability measure concentrated on `[0,1]`
(`noiseLaw_Icc_eq_one`).

So the CHD foundation is complete, from the kernel build through the regularized solve, the noise
statistic, and the `Z_test` thresholds up to the graph-structure decisions. The remaining open items
are both narrow and deliberately scoped: the cyclic-Jacobi convergence *rate* (captured exactly by the
a-posteriori residual certificate, never by `sorry`), and the *asymptotic* half of the `Z_test` — that
the empirical 5%/95% percentiles converge to the true quantiles of the now-proved null law
(Glivenko–Cantelli / DKW), and that an exchangeable fresh draw is rejected at exactly rank rate
`k/(N+1)`. The *pointwise* part of that asymptotic step is now proved, not merely scaffolded: on the
i.i.d. sequence `FactorizationsZAsymptotic` builds (`nullNoise` an independent,
identically-`noiseLaw`-distributed, `[0,1]`-valued, integrable sequence under
`Measure.infinitePi nullGaussian`), `empCDF_tendsto_cdf` applies the strong law of large numbers to
the bounded indicators `1{noiseᵢ ≤ t}` — whose mean is exactly `cdf noiseLaw t`
(`integral_nullBelow_zero`) — to give almost-sure convergence `F̂_N(t) → cdf noiseLaw t` for every
fixed `t`, the *pointwise* Glivenko–Cantelli theorem, sorry-free. What stays genuinely research-grade
is the *uniform* Glivenko–Cantelli / DKW–Massart sharp
constant (bracketing / VC chaining) and the exchangeability rank rate `k/(N+1)`
(symmetric-group rank distribution) — both absent from `Mathlib.Probability` v4.30.0. One open item is
a proof-only gap on a quantity CHD does not need to *run*; the other is the genuine statistical
frontier, flagged rather than stubbed with `sorry`.
