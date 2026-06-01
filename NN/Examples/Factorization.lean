/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Examples.Factorization.Common
public import NN.Examples.Factorization.Cholesky
public import NN.Examples.Factorization.QR
public import NN.Examples.Factorization.SymEig
public import NN.Examples.Factorization.SVD
public import NN.Examples.Factorization.JacobiDecrease
public import NN.Examples.Factorization.JacobiRate
public import NN.Examples.Factorization.RidgeSolve
public import NN.Examples.Factorization.Variational
public import NN.Examples.Factorization.LinearKernel
public import NN.Examples.Factorization.QuadraticKernel
public import NN.Examples.Factorization.GaussianKernel
public import NN.Examples.Factorization.Discovery

/-!
# Matrix factorization examples

Executable sanity checks for the spec-layer matrix factorizations in
`NN.Spec.Core.Tensor.Factorizations`, designed to corroborate the formal correctness theorems in
`NN.Proofs.Tensor.Basic.{Factorizations, FactorizationsReconstruction, FactorizationsOrthonormal,
FactorizationsJacobi}`. Each check runs through compiled `#eval` assertions, so the build fails if a
factorization misbehaves.

- `Cholesky` — `A = L · Lᵀ`; **negative control**: an indefinite `A` correctly fails (no SPD factor).
- `QR`       — `A = Q · R`, `Qᵀ·Q = I`; **negative control**: a rank-deficient `A` still reconstructs
  but `Qᵀ Q ≠ I`, separating the two guarantees and showing full column rank is needed.
- `SymEig`   — `A = V · diag(λ) · Vᵀ`; orthogonality `Vᵀ V = I` is exact at *any* sweep count (witness
  of the a-priori `jacobi_orthogonal`), diagonalization is asymptotic, and the **exact residual
  certificate** `‖A − V·diag(λ)·Vᵀ‖² = ‖offDiag(VᵀAV)‖²` (`symEigJacobi_frobenius_residual`) is
  verified numerically.
- `SVD`      — `A = U · diag(σ) · Vᵀ`, `Vᵀ V = I`; **negative control**: a permuted `σ` fails to
  reconstruct.
- `JacobiDecrease` — the per-rotation progress identity `‖offDiag(Jᵀ A J)‖² = ‖offDiag A‖² − 2·A[p,q]²`
  (`jacobi_off_decrease`) and Frobenius-mass invariance; **negative controls**: a wrong-angle rotation
  misses the decrease, a non-orthogonal one breaks mass invariance.
- `JacobiRate` — the *aggregate* linear-contraction rate of the classical largest-pivot strategy:
  `‖offDiag(Jᵀ A J)‖² ≤ (1 − 2/(n²−n))·‖offDiag A‖²` (`jacobi_off_decrease_classical`); **negative
  control**: annihilating a non-largest (tiny) pivot misses the guaranteed factor, so the rate is
  specific to the largest-pivot choice.
- `RidgeSolve` — the kernel-ridge (Tikhonov) linear solve `(K + γ·I)·x = b` via Cholesky +
  forward/back substitution (`solveRidgeFn_mulVec_of_posSemidef`, the verified core of CHD
  `solve_variationnal`, now *unconditional* for PSD `K` and `γ > 0`): for a rank-deficient Gram kernel
  `K = G·Gᵀ` and `γ > 0`, `solveRidgeFn` reconstructs `b` to machine precision; **negative control**:
  with `γ = 0` the singular `K` has a zero Cholesky pivot and the solve diverges (`NaN`), so
  regularization is necessary. Also exhibits the **keystone** `choleskyFn_diag_pos_of_posDef`: the SPD
  `K + γ·I` has all-positive Cholesky pivots, while the singular `K` has a zero pivot (PosDef needed);
  and the two **capstones** — `cholesky_posDef` (the SPD Cholesky reconstructs `L·Lᵀ = K + γ·I`
  exactly, while an *indefinite* matrix fails with a `NaN` pivot) and `solveRidgeFn_eq_inv_mulVec` (the
  solve *is* the regularized inverse: its columns assemble into `(K + γ·I)⁻¹` with
  `(K + γ·I)·(K + γ·I)⁻¹ = I`).
- `Variational` — the *eigendecomposition* form of CHD `perform_regression_and_find_gamma`
  (`interpolatory.py`): from `eigh(K)`, the variational solve `yb = -(K + γ·I)⁻¹·ga`, the agreement of
  the eig and Cholesky routes (`variationalSolveFn_eq_neg_solveRidgeFn`), the
  `noise`/`find_gamma`-loss/`Z_test` statistic as a spectral ratio bounded in `[0,1]`
  (`varNoiseFn_nonneg`, `varNoiseFn_le_one`), and `Z_test` spectral invariance
  (`varNoiseFn_projFn_mulVec`); **negative controls**: wrong eigenvectors break the solve, and `γ < 0`
  pushes the noise outside `[0,1]`.
- `LinearKernel` — CHD *builds* the kernel from data (`Modes/kernels.py`); the linear mode is
  `K = 𝟙𝟙ᵀ + scale·Φ·Φᵀ`, proven symmetric positive-semidefinite for `scale ≥ 0`
  (`linearKernelFn_posSemidef`), which discharges the `PosSemidef` hypothesis every solve/`find_gamma`
  theorem assumes. Checks: `K = Kᵀ`, matches the CHD `LinearMode` formula, all Jacobi eigenvalues `≥ 0`
  (masking a feature preserved), and the PSD kernel feeds an exact ridge solve; **negative control**:
  `scale < 0` makes `K` indefinite (a negative eigenvalue appears).
- `QuadraticKernel` — CHD's *quadratic* mode (`Modes/kernels.py`),
  `K = scale·(alpha + Φ·Φᵀ)² + (1 − alpha²·scale) = 𝟙𝟙ᵀ + (2·scale·alpha)·Φ·Φᵀ + scale·(Φ·Φᵀ ⊙ Φ·Φᵀ)`,
  proven symmetric positive-semidefinite for `scale ≥ 0` and `alpha ≥ 0` via the **Schur product
  theorem** on the Hadamard square (`quadraticKernelFn_posSemidef`). Checks mirror the linear mode:
  `K = Kᵀ`, matches the CHD `QuadraticMode` formula, all Jacobi eigenvalues `≥ 0` (masking preserved),
  PSD kernel feeds an exact ridge solve; **negative controls**: both `alpha < 0` and `scale < 0` make
  `K` indefinite, so both bounds are necessary.
- `GaussianKernel` — CHD's *Gaussian* (fully-nonlinear) mode (`Modes/kernels.py`),
  `K = scale·∏_dim (1 + w[dim]·exp(−(X[i,dim]−X[j,dim])²/2l²))`, proven symmetric positive-semidefinite
  for `scale ≥ 0` and a nonnegative mask `w ≥ 0` (`gaussianKernelFn_posSemidef`) — *without*
  Bochner/Schoenberg, via the entrywise-exponential Hadamard-power series (the PSD cone closed under
  limits) and the **Schur product theorem** over features. Checks mirror the other modes: `K = Kᵀ`,
  matches the CHD `GaussianMode` product formula, all Jacobi eigenvalues `≥ 0` (masking preserved), PSD
  kernel feeds an exact ridge solve; **negative controls**: `scale < 0` and a *negative mask weight*
  (`w = [−2,0]`, which drives the diagonal below zero) both make `K` indefinite. With the linear,
  quadratic, and Gaussian modes all discharged, every CHD kernel build is now PSD-verified.
- `Discovery` — CHD's *discovery decision layer* (`decision.py`, `_GraphDiscoveryMain.py`), which turns
  the verified `noise` statistic into graph structure: the activation prune step (`argMinFn`, picks the
  least-activated ancestor), the `MinNoiseKernelChooser` (`kernelChooserFn`, the least-noise valid kernel
  with `noise < Z_low`, or `none`), the `MaxIncrementModeChooser` (`modeChooserFn`, the largest
  `noise`-jump iteration), and the stopping rule (`allPrunedFn`), proved sound/complete in
  `FactorizationsDecision`. Checks: argmin picks the least-activated ancestor (and not the most), the
  chooser selects the unique valid kernel / least noise among valid / `none` when none valid, the mode
  chooser picks the largest-increment iteration, and the stopping rule fires only on the all-zero mask;
  an **end-to-end** block then feeds the verified `varNoiseSpec` at several `γ` into `argMinFn`, a
  `find_gamma` sweep selecting the least-noise regularization (all noises in `[0,1]`); **negative
  controls** confirm the most-activated ancestor and tiny-increment iterations are correctly rejected.
  A closing **`Z_test`** block exercises the statistical layer: the null-distribution thresholds
  `Z_low`/`Z_high` (5th/95th percentiles of the per-sample `noise`) are well-posed
  (`0 ≤ Z_low ≤ Z_high ≤ 1`), data aligned with the dominant eigenvector clears the lower tail
  (`noise < Z_low`, **positive**), and a high noise / a noise at the upper tail are rejected
  (**negative controls**) — feeding `MinNoiseKernelChooser` exactly as in CHD. A final
  **distributional** sub-block checks the *finite-sample calibration* proved in
  `FactorizationsZTest`: across the `N = 20` null draws, at most `⌊N/20⌋ ≈ 5%` fall below `Z_low`
  (`zLow_null_exceedance_le`, here exactly `1/20`) and at most `≈ 5%` rise above `Z_high`
  (`zHigh_null_exceedance_le`, here `0`); a **negative control** confirms the slack `Z_high`
  threshold admits `≈ 95%` of the draws, so the `5%` calibration is specific to `Z_low`. (The
  companion measure-theoretic fact — the i.i.d.-Gaussian null law is a probability measure on
  `[0,1]`, `noiseLaw_Icc_eq_one` — is noncomputable and lives in the proofs.) A closing
  **asymptotic-scaffold** sub-block corroborates `FactorizationsZAsymptotic` (step (a) of the
  asymptotic-calibration plan): the i.i.d. null *sequence* `nullNoise` is proven independent,
  identically distributed with law `noiseLaw`, `[0,1]`-valued and integrable (the SLLN's
  `hint`/`hindep`/`hident`) — noncomputable, so the `#eval`s exercise its **computable shadow**, the
  empirical CDF `F̂_N(t) = #{i<N : noiseᵢ ≤ t}/N`: checks that it is a bona fide CDF (in `[0,1]`,
  monotone, saturating to `1` at the top of the `[0,1]` support, vanishing below `0`), with a
  **negative control** that it is non-degenerate (rises strictly from `0` to `1`, carrying the
  distributional content whose convergence to `cdf noiseLaw` is the next increment, step (b)).
  A final **consistency** sub-block corroborates `empCDF_tendsto_cdf` (step (b)): the empirical CDF
  is the SLLN *running mean* of the bounded i.i.d. indicators `1{noiseᵢ ≤ t}`, whose mean is exactly
  `cdf noiseLaw t` (`integral_nullBelow_zero`), so almost surely `F̂_N(t) → cdf noiseLaw t` (pointwise
  Glivenko–Cantelli). The limit needs `N → ∞`, so the `#eval`s watch the **growing-prefix running
  mean** `F̂_N` settle toward the full-sample estimate: each prefix is a valid `[0,1]` CDF value
  (bounded summands), the limit value `cdf 1 = 1` is attained at every `N`, and a **negative control**
  confirms the estimate genuinely moves with `N` (an early prefix differs from the full sample), so
  the convergence is a real limit being approached rather than a vacuous constant.
  A final **concentration** sub-block corroborates `empCDF_concentration` (step (c)): the
  Dvoretzky–Kiefer–Wolfowitz inequality *at a single point*, `ℙ(|F̂_N(t) − cdf noiseLaw t| ≥ ε) ≤
  2·exp(−2·N·ε²)` with the sharp Hoeffding exponent (the threshold indicators are `[0,1]`-bounded,
  so sub-Gaussian with proxy `1/4`; the one-sided `empCDF_upper_tail`/`empCDF_lower_tail` give
  `exp(−2Nε²)` each, the union the factor `2`). The probability is noncomputable, so the `#eval`s
  exercise the bound's two computable shadows: the tail *function* `2·exp(−2Nε²)` (twice the
  one-sided tail, decreasing in both `N` and `ε`, non-vacuous `< 1` once `2Nε² > ln 2`), and the
  observed deviation it governs — every prefix of `≥ 3` draws keeps `F̂_N` within `ε = 0.3` of the
  full-sample estimate uniformly over thresholds, with a **negative control** that the tiniest
  prefixes (`N = 1, 2`) deviate by `0.5 > ε`, the honest weak-`N` regime where the `2·exp(−2Nε²)`
  bound is still near `2`.
  A final **quantile-transfer** sub-block corroborates `empQuantile_tendsto` (step (d)): inverting the
  CDF convergence into convergence of the empirical *percentiles* the chooser thresholds against.
  Wherever the true CDF strictly straddles a level `p` at its quantile `q` (`StraddlesQuantile`), any
  lower empirical `p`-quantile (`IsLowerQuantile`) converges almost surely to `q`. The limit is
  noncomputable, so the `#eval`s use the **full-sample** quantile `q̂₂₀` as the stand-in for `q` and
  the prefix quantile `q̂_N` as its shadow: the lower `p`-quantile reaches level `p` (`p ≤ F̂₂₀(q̂₂₀)`),
  is monotone in `p` and lands in `[0,1]`, and the empirical median converges (`|q̂_N − q̂₂₀| ≤ 0.02`
  for every prefix of `≥ 3` draws). Two **negative controls** keep it honest: the prefix median
  genuinely moves with `N` (non-vacuous limit), and the convergence is hypothesis-sensitive — the
  `5%`-tail quantile (flatter CDF, sparser straddle) deviates more at `N = 10` than the well-straddled
  median, the empirical signature of `StraddlesQuantile` being a needed hypothesis.

Both **positive** checks (a valid factorization reconstructs to `err ≈ 0`) and **negative controls**
(the same metric reports a large error / `NaN` when a hypothesis is violated) are included, so a
reviewer can see the checks are not vacuous.
-/

@[expose] public section
