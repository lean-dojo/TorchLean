/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Examples.Factorization.Common
meta import NN.Examples.Factorization.Common

/-!
# Example: the CHD discovery decision layer

These checks corroborate `NN.Proofs.Tensor.Basic.FactorizationsDecision`. Once a kernel is built and its
`noise` level computed (`varNoiseSpec`, proven to lie in `[0,1]`), CHD's *discovery loop* turns those
numbers into graph-structure decisions (`decision.py`, `_GraphDiscoveryMain.py`). We exercise the four
deterministic choices the loop makes, each with a positive check and a negative control:

* **prune the least-activated ancestor** — `argMinFn` returns the index of the smallest activation
  (`min_activation = np.argmin(activations)`); the *most*-activated ancestor is correctly **not** chosen;
* **pick the kernel mode that admits an edge** — `kernelChooserFn` (`MinNoiseKernelChooser`) returns the
  valid kernel (`noise < Z_low`) of least `noise`, or `none` when no kernel is valid;
* **report the pruning iteration of largest `noise` jump** — `modeChooserFn` (`MaxIncrementModeChooser`)
  returns the `argmax` of the increments;
* **stop when every ancestor is pruned** — `allPrunedFn` fires on the all-zero mask and not before.

A `find_gamma`-style block then closes the loop end-to-end: it builds an SPD kernel, eigendecomposes
it, and feeds the *verified* `varNoiseSpec` at several `γ` straight into `argMinFn` to select the
regularization with least noise.

A final **`Z_test`** block adds the statistical layer (`interpolatory.py`): the observed `noise` is
judged against the null distribution of the *same* statistic under random data — `Z_low`/`Z_high` are
the 5th/95th percentiles of the per-sample noises. We check the thresholds are well-posed
(`0 ≤ Z_low ≤ Z_high ≤ 1`, each percentile inheriting the verified `noise ∈ [0,1]` bound) and that the
verdict `noise < Z_low` flags a real edge — with a genuine positive (data aligned with the dominant
eigenvector clears the lower tail) and negatives (a high noise, and a noise sitting at the upper tail,
are both correctly rejected). Every decision runs over `Float`, the executable runtime scalar.
-/

@[expose] public section


namespace NN.Examples.Factorization.Discovery

/-- A length-3 `Float` family `Fin 3 → Float` from three entries. -/
def vec3 (a b c : Float) : Fin 3 → Float := fun i => [a, b, c].getD i.val 0.0
/-- A length-4 `Float` family `Fin 4 → Float` from four entries. -/
def vec4 (a b c d : Float) : Fin 4 → Float := fun i => [a, b, c, d].getD i.val 0.0

/-- Build a length-`n` `Float` vector tensor from a list (missing entries `0`). -/
def mkVec {n : Nat} (xs : List Float) : Spec.Tensor Float (.dim n .scalar) :=
  Spec.ofVecFn (fun i => xs.getD i.val 0.0)

/-- Encode a chooser verdict as an `Int`: `-1` for `none` ("no ancestor"), else the chosen index. -/
def chooserCode {m : Nat} (o : Option (Fin m)) : Int :=
  match o with
  | none => -1
  | some i => Int.ofNat i.val

/-- Compiled positive assertion that a `Bool` decision is `true`. -/
def assertTrue (name : String) (b : Bool) : IO Unit :=
  if b then IO.println s!"{name}: OK"
  else throw (IO.userError s!"{name}: FAIL (expected true)")

/-- Compiled negative-control assertion that a `Bool` decision is `false` (the property correctly does
*not* hold). -/
def assertFalse (name : String) (b : Bool) : IO Unit :=
  if b then throw (IO.userError s!"{name}: FAIL (expected false)")
  else IO.println s!"{name}: OK (correctly false)"

/-! ## Pruning: `argMinFn` removes the least-activated ancestor -/

/-- Activations of four candidate ancestors; ancestor 1 is the least-activated. -/
def activations : Fin 4 → Float := vec4 0.8 0.2 0.5 0.9

#eval IO.println s!"activations = {(List.finRange 4).map activations}, \
  argMin = {(Spec.argMinFn activations).val}"

-- Positive — the prune step removes the least-activated ancestor (`argMinFn_le`).
#eval assertTrue "prune picks the least-activated ancestor (argmin = 1)"
  ((Spec.argMinFn activations).val == 1)

-- Negative — it does *not* remove the most-activated ancestor (index 3).
#eval assertFalse "prune does not pick the most-activated ancestor"
  ((Spec.argMinFn activations).val == 3)

/-! ## Kernel chooser: least-noise valid kernel, or `none` -/

/-- Three candidate kernels' `noise` levels and `Z_low` lower bounds. Validity is `noise < Z_low`:
kernel 0 invalid (`0.3 ≥ 0.2`), kernel 1 valid (`0.1 < 0.4`), kernel 2 invalid (`0.5 ≥ 0.1`). -/
def noisesA : Fin 3 → Float := vec3 0.3 0.1 0.5
def ZlowsA : Fin 3 → Float := vec3 0.2 0.4 0.1

#eval IO.println s!"kernel chooser (one valid) -> code {chooserCode (Spec.kernelChooserFn noisesA ZlowsA)}"

-- Positive — exactly kernel 1 is valid, so the chooser admits an edge via kernel 1 (`kernelChooserFn_eq_some`).
#eval assertTrue "kernel chooser selects the unique valid kernel (some 1)"
  (chooserCode (Spec.kernelChooserFn noisesA ZlowsA) == 1)

/-- Two valid kernels (0 and 1); the chooser must take the one of *least* noise (kernel 0, `0.05`). -/
def noisesB : Fin 3 → Float := vec3 0.05 0.1 0.5
def ZlowsB : Fin 3 → Float := vec3 0.2 0.4 0.1

-- Positive — among valid kernels the chooser takes least noise (kernel 0 beats kernel 1).
#eval assertTrue "kernel chooser takes least noise among valid (some 0)"
  (chooserCode (Spec.kernelChooserFn noisesB ZlowsB) == 0)

/-- No kernel is valid (`noise ≥ Z_low` everywhere): the chooser reports "no ancestor". -/
def noisesC : Fin 3 → Float := vec3 0.5 0.6 0.7
def ZlowsC : Fin 3 → Float := vec3 0.1 0.2 0.3

-- Negative — no valid kernel ⟹ no edge (`kernelChooserFn_eq_none`); code `-1`.
#eval assertTrue "kernel chooser reports none when no kernel is valid (code -1)"
  (chooserCode (Spec.kernelChooserFn noisesC ZlowsC) == -1)

/-! ## Mode chooser: the iteration of largest `noise` increment -/

/-- The per-iteration `noise` sequence of a pruning run. The big jump `0.08 → 0.9` is between iterations
1 and 2, so `MaxIncrementModeChooser` reports iteration 1 (increment `0.82`). -/
def noiseSeq : Fin 4 → Float := vec4 0.05 0.08 0.9 0.95

#eval IO.println s!"increments = {(List.finRange 4).map (Spec.modeIncrementFn noiseSeq)}, \
  modeChooser = {(Spec.modeChooserFn noiseSeq).val}"

-- Positive — the mode chooser reports the largest-jump iteration (`modeChooserFn_ge`).
#eval assertTrue "mode chooser picks the largest noise-increment iteration (argmax = 1)"
  ((Spec.modeChooserFn noiseSeq).val == 1)

-- Negative — it does *not* report a tiny-increment iteration (iteration 0, increment 0.03).
#eval assertFalse "mode chooser does not pick a tiny-increment iteration"
  ((Spec.modeChooserFn noiseSeq).val == 0)

/-! ## Stopping rule: fire exactly when all ancestors are pruned -/

-- Positive — the loop stops when every ancestor mode is zero (`allPrunedFn_iff`).
#eval assertTrue "stopping rule fires when all ancestors are pruned"
  (Spec.allPrunedFn (vec3 0.0 0.0 0.0))

-- Negative — it does not fire while an ancestor remains active.
#eval assertFalse "stopping rule does not fire while an ancestor remains"
  (Spec.allPrunedFn (vec3 0.0 1.0 0.0))

/-! ## End-to-end: `find_gamma` feeds the verified `noise` into `argMinFn`

A `find_gamma`-style sweep: build an SPD kernel, eigendecompose it, evaluate the verified
`varNoiseSpec` at several `γ`, and let `argMinFn` pick the regularization of least noise — exactly the
discovery layer consuming the verified statistic. More regularization means more noise, so the smallest
`γ` wins (index 0). -/

/-- A `3 × 3` symmetric positive-definite kernel. -/
def K : Spec.Tensor Float (.dim 3 (.dim 3 .scalar)) :=
  mkMat [[2.0, 0.5, 0.3],
         [0.5, 2.0, 0.4],
         [0.3, 0.4, 2.0]]

/-- Its eigendecomposition `(evals, V)` from the Jacobi solver. -/
def evals : Spec.Tensor Float (.dim 3 .scalar) := (Spec.symEigJacobiSpec K 12).1
def V : Spec.Tensor Float (.dim 3 (.dim 3 .scalar)) := (Spec.symEigJacobiSpec K 12).2

/-- The data vector `ga`. -/
def ga : Spec.Tensor Float (.dim 3 .scalar) := mkVec [1.0, 2.0, 3.0]

/-- The candidate regularizations, increasing. -/
def gammas : Fin 3 → Float := vec3 0.01 0.1 1.0

/-- The verified `noise` at each candidate `γ` (`find_gamma`'s loss). -/
def noiseAt : Fin 3 → Float := fun i => Spec.varNoiseSpec evals V (gammas i) ga

#eval IO.println s!"find_gamma noises = {(List.finRange 3).map noiseAt}, \
  argMin γ index = {(Spec.argMinFn noiseAt).val}"

-- Positive — every swept noise is a genuine fraction in [0,1] (numeric witness of `varNoiseFn_nonneg`/`_le_one`).
#eval assertTrue "find_gamma noises all lie in [0,1]"
  ((List.finRange 3).all (fun i => 0.0 ≤ noiseAt i && noiseAt i ≤ 1.0))

-- Positive — `find_gamma` (argmin of the verified noise) selects the least-regularized γ (index 0).
#eval assertTrue "find_gamma selects least-noise γ via argMinFn (index 0)"
  ((Spec.argMinFn noiseAt).val == 0)

/-! ## `Z_test`: the null-distribution significance thresholds

CHD decides an edge is real by comparing the observed `noise` against the null distribution of the
*same* statistic under random data: draw `N` samples, score each one's `noise`, sort, and read off the
5th/95th percentiles as `Z_low`/`Z_high` (`Z_test` in `interpolatory.py`). An edge is significant when
`noise < Z_low`. These checks corroborate `FactorizationsDecision`: the thresholds are well-posed
(`0 ≤ Z_low ≤ Z_high ≤ 1`, each percentile inheriting the verified `noise ∈ [0,1]` bound) and the
verdict drives `MinNoiseKernelChooser`. -/

/-- An `N = 20` family of pseudo-random null draws `sⱼ ∈ ℝ³` (deterministic, standing in for CHD's
`jax.random.normal` samples). With `N = 20` the percentile indices are `Z_low = ⌊0.05·20⌋ = 1` and
`Z_high = ⌊0.95·20⌋ = 19`. -/
def zSamples : Fin 20 → Fin 3 → Float :=
  fun j i => (Float.ofNat ((j.val * 31 + i.val * 17 + 7) % 23) - 11.0) / 7.0

/-- The regularization at which we run the `Z_test`. -/
def gammaZ : Float := 0.1

/-- `Z_low`: the 5th percentile of the null `noise` distribution from the verified eigendecomposition. -/
def zLow : Float := Spec.zLowFn (Spec.toVecFn evals) (Spec.toMatFn V) gammaZ zSamples
/-- `Z_high`: the 95th percentile of the null `noise` distribution. -/
def zHigh : Float := Spec.zHighFn (Spec.toVecFn evals) (Spec.toMatFn V) gammaZ zSamples

#eval IO.println s!"Z_test null thresholds: Z_low = {zLow}, Z_high = {zHigh}"

-- Positive — the thresholds are ordered (`zLowFn_le_zHighFn`); `leBool` is the very key the sort uses.
#eval assertTrue "Z_low ≤ Z_high (order-statistic monotonicity)" (Spec.leBool zLow zHigh)

-- Positive — both thresholds are genuine fractions in [0,1] (`zLowFn_nonneg`/`_le_one`, `zHighFn_*`).
#eval assertTrue "Z_low and Z_high both lie in [0,1]"
  (Spec.leBool 0.0 zLow && Spec.leBool zLow 1.0 && Spec.leBool 0.0 zHigh && Spec.leBool zHigh 1.0)

/-- The dominant eigen-direction (largest eigenvalue), found by the verified `argMaxFn`. -/
def domIdx : Fin 3 := Spec.argMaxFn (Spec.toVecFn evals)
/-- A "real signal": data aligned with the dominant eigenvector. Its `noise` is exactly the shrinkage
`γ/(λ_dom+γ)` — the smallest shrinkage, so well below the null tail — the kind of edge CHD keeps. -/
def signalGa : Fin 3 → Float := fun i => Spec.toMatFn V i domIdx
/-- The observed `noise` of the signal-aligned data (the verified `varNoiseFn`). -/
def obsSignal : Float := Spec.varNoiseFn (Spec.toVecFn evals) gammaZ (Spec.projFn (Spec.toMatFn V) signalGa)

#eval IO.println s!"signal-aligned noise = {obsSignal}, significant (noise < Z_low)? \
  {Spec.zSignificantFn obsSignal zLow}"

-- Positive — the signal-aligned noise is itself a fraction in [0,1] (witness of `varNoiseFn_*`).
#eval assertTrue "signal-aligned noise lies in [0,1]"
  (Spec.leBool 0.0 obsSignal && Spec.leBool obsSignal 1.0)

-- Positive — end-to-end: data aligned with the dominant eigenvector clears the null's lower tail, so
-- the `Z_test` flags a real edge (`noise < Z_low`).
#eval assertTrue "end-to-end: dominant-direction signal is significant (noise < Z_low)"
  (Spec.zSignificantFn obsSignal zLow)

-- Positive — a clearly-significant edge (noise 0.05 below threshold 0.20) is flagged (`zSignificantFn`).
#eval assertTrue "significant edge: noise 0.05 < Z_low 0.20" (Spec.zSignificantFn 0.05 0.20)

-- Negative — a noise *above* the threshold is correctly not significant.
#eval assertFalse "non-significant: noise 0.50 ≥ Z_low 0.20" (Spec.zSignificantFn 0.50 0.20)

-- Negative — the 95th-percentile value itself is never below the 5th (`zHigh ≥ zLow`), so feeding it as
-- an "observed" noise is correctly judged non-significant — a faithful negative from the real null.
#eval assertFalse "Z_high is not below Z_low (a noise at the upper tail is not significant)"
  (Spec.zSignificantFn zHigh zLow)

-- Positive — the `Z_test` verdict feeds `MinNoiseKernelChooser` (`zTest_admits_edge`): a significant
-- single kernel is admitted as `some 0`.
#eval assertTrue "significant kernel is admitted (chooser → some 0)"
  (chooserCode (Spec.kernelChooserFn (fun _ : Fin 1 => (0.05 : Float)) (fun _ : Fin 1 => 0.20)) == 0)

-- Negative — a non-significant single kernel is rejected (`none`, code -1).
#eval assertTrue "non-significant kernel is rejected (chooser → none, code -1)"
  (chooserCode (Spec.kernelChooserFn (fun _ : Fin 1 => (0.50 : Float)) (fun _ : Fin 1 => 0.20)) == -1)

/-! ### The distributional layer: finite-sample calibration of the thresholds

The `noise` of each null draw, scored by the same functional as the data (`sampleNoisesFn`). The
percentile thresholds carry a *non-asymptotic* false-positive guarantee, proved in
`FactorizationsZTest`: at most `⌊N/20⌋ ≈ 5%` of the `N` draws fall below `Z_low`
(`zLow_null_exceedance_le`) and at most `N-1-⌊19N/20⌋ ≈ 5%` fall above `Z_high`
(`zHigh_null_exceedance_le`). On the measure side, modelling the draws as i.i.d. standard Gaussian
makes the null law a probability measure on `[0,1]` (`noiseLaw_Icc_eq_one`); that part is
noncomputable, so it is exercised by the proofs rather than `#eval`. -/

/-- The per-draw `noise` levels of the `Z_test` null sample (`N = 20` draws). -/
def zNullNoises : Fin 20 → Float :=
  Spec.sampleNoisesFn (Spec.toVecFn evals) (Spec.toMatFn V) gammaZ zSamples

/-- How many of the 20 null draws score strictly below a threshold (the empirical lower-tail count,
using the very `ltBool` comparator the `Z_test` decision uses). -/
def countBelow (thr : Float) : Nat :=
  ((List.finRange 20).filter (fun j => Spec.ltBool (zNullNoises j) thr)).length

/-- How many of the 20 null draws score strictly above a threshold (the empirical upper-tail count). -/
def countAbove (thr : Float) : Nat :=
  ((List.finRange 20).filter (fun j => Spec.ltBool thr (zNullNoises j))).length

#eval IO.println s!"null-draw tail counts: below Z_low = {countBelow zLow} (≤ ⌊20/20⌋ = {Spec.zLowIdx 20}), \
  above Z_high = {countAbove zHigh} (≤ 19 - {Spec.zHighIdx 20} = {20 - 1 - Spec.zHighIdx 20}), \
  below Z_high = {countBelow zHigh}"

-- Positive — `zLow_null_exceedance_le`: at most `⌊N/20⌋` (≈ 5%) of the null draws beat `Z_low`, i.e.
-- the threshold's own empirical false-positive rate is bounded by the 5th-percentile rank.
#eval assertTrue "≤ 5% of null draws fall below Z_low (zLow_null_exceedance_le)"
  (decide (countBelow zLow ≤ Spec.zLowIdx 20))

-- Positive — `zHigh_null_exceedance_le`: at most `N-1-⌊19N/20⌋` (≈ 5%) of the null draws exceed
-- `Z_high`. With `N = 20`, `Z_high` is the top order statistic, so nothing strictly exceeds it.
#eval assertTrue "≤ 5% of null draws rise above Z_high (zHigh_null_exceedance_le)"
  (decide (countAbove zHigh ≤ 20 - 1 - Spec.zHighIdx 20))

-- Negative control — the *slack* upper threshold `Z_high` admits far more than 5% of the null mass
-- below it (≈ 95%), so the 5% lower-tail calibration is specific to `Z_low`, not an artifact of any
-- threshold: a test against `Z_high` would over-reject the null.
#eval assertTrue "Z_high is a slack threshold: > 5% of null draws fall below it (calibration is specific to Z_low)"
  (decide (Spec.zLowIdx 20 < countBelow zHigh))

/-! ### The asymptotic-calibration scaffold (step a): the empirical CDF of the null sample

`FactorizationsZAsymptotic` lifts the single null draw to the i.i.d. *sequence* `nullNoise` under the
product measure `nullSeqGaussian`, proving it independent (`nullNoise_iIndepFun`), identically
distributed with the common law `noiseLaw` (`nullNoise_hasLaw`, `nullNoise_identDistrib`),
`[0,1]`-valued (`nullNoise_mem_Icc`) and integrable (`integrable_nullNoise`) — exactly the three
hypotheses (`hint`/`hindep`/`hident`) the strong law of large numbers consumes. That scaffold is
*noncomputable* (a statement about an infinite product measure), so it cannot be `#eval`'d; what we
exercise here is its **computable shadow**, the empirical CDF of the finite null sample
`F̂_N(t) = #{i < N : noiseᵢ ≤ t} / N`. This is the very object whose almost-sure convergence to
`cdf noiseLaw` *is* the SLLN application (step b of the plan, not yet formalized). At step (a) the
i.i.d. sample alone already gives that `F̂_N` is a bona fide CDF — monotone, valued in `[0,1]`,
saturating to `1` above the support and vanishing below it — which is what we check. -/

/-- Empirical CDF of the `N = 20` null noises at a threshold `t`: the fraction of draws scoring `≤ t`
(using the `leBool` comparator the order statistics already use). The computable shadow of the
noncomputable `empCDF` whose consistency is step (b). -/
def empCdf (t : Float) : Float :=
  (((List.finRange 20).filter (fun j => Spec.leBool (zNullNoises j) t)).length).toFloat / 20.0

#eval IO.println s!"empirical CDF of the null sample: F(0) = {empCdf 0.0}, F(Z_low) = {empCdf zLow}, \
  F(Z_high) = {empCdf zHigh}, F(1) = {empCdf 1.0}"

-- Positive — `F̂` is valued in `[0,1]` at every threshold (it is a fraction of the 20 draws), the
-- finite-sample image of `nullNoise_mem_Icc` / the law `noiseLaw` being a probability measure.
#eval assertTrue "empirical CDF lies in [0,1] across thresholds"
  ([0.0, zLow, zHigh, 0.5, 1.0].all
    (fun t => Spec.leBool 0.0 (empCdf t) && Spec.leBool (empCdf t) 1.0))

-- Positive — `F̂` is monotone nondecreasing: more of the sample falls below a larger threshold.
-- Since `Z_low ≤ Z_high`, `F̂(Z_low) ≤ F̂(Z_high)` — the empirical shadow of `monotone_cdf`.
#eval assertTrue "empirical CDF is monotone: Z_low ≤ Z_high ⇒ F(Z_low) ≤ F(Z_high)"
  (Spec.leBool (empCdf zLow) (empCdf zHigh))

-- Positive — `F̂` saturates to `1`: every null noise lies in `[0,1]` (`nullNoise_mem_Icc`), so all
-- 20 draws score `≤ 1` and the empirical CDF reaches its full mass there.
#eval assertTrue "empirical CDF reaches 1 at t = 1 (all null noises ≤ 1, nullNoise_mem_Icc)"
  (empCdf 1.0 == 1.0)

-- Positive — `F̂` vanishes below the support: no null noise is negative (`nullNoise_mem_Icc`), so
-- none scores `≤` a negative `t`.
#eval assertTrue "empirical CDF is 0 below the support (no null noise < 0)"
  (empCdf (-0.01) == 0.0)

-- Negative control — `F̂` is *not* the constant function: it genuinely rises from `0` to `1` across
-- the support, so it carries the distributional content the i.i.d. scaffold formalizes. A degenerate
-- (point-mass) sample would have a flat-then-jump CDF; a sample with no spread would not separate
-- these thresholds. This is what makes the consistency target of step (b) non-vacuous.
#eval assertTrue "empirical CDF is non-degenerate: F(below support) < F(1) (carries distribution info)"
  (Spec.ltBool (empCdf (-0.01)) (empCdf 1.0))

/-! ### Pointwise consistency of the empirical CDF (step b): the SLLN running mean

`FactorizationsZAsymptotic` now proves `empCDF_tendsto_cdf`: for each threshold `t`, the empirical
CDF `empCDF Λ V γ N t` of the i.i.d. null draws converges *almost surely* to the true CDF
`cdf noiseLaw t` as `N → ∞` — the pointwise Glivenko–Cantelli theorem, via the strong law of large
numbers (`strong_law_ae_real`) applied to the bounded i.i.d. indicators `1{noiseᵢ ≤ t}`. The limit
value is pinned by `integral_nullBelow_zero`: the *mean* of the indicator is exactly
`cdf noiseLaw t`, so the empirical CDF is the Monte-Carlo estimator of the null CDF. Convergence
needs `N → ∞`, so it is not directly `#eval`-able; the computable shadow is the **running mean**
`F̂_N(t) = (1/N)·#{i < N : noiseᵢ ≤ t}` over growing prefixes `N` of the 20-draw sample, which we
watch settle toward the full-sample estimate `empCdf t` of `cdf noiseLaw t`. -/

/-- Running empirical CDF over the first `N ≤ 20` null draws at threshold `t`: the partial mean of the
indicator sequence `1{noiseᵢ ≤ t}` that the strong law averages. As `N → ∞` this is precisely the
quantity `empCDF_tendsto_cdf` sends to `cdf noiseLaw t`; here we watch its growing-`N` prefixes. -/
def empCdfPrefix (N : Nat) (t : Float) : Float :=
  (((List.finRange 20).filter
      (fun j => decide (j.val < N) && Spec.leBool (zNullNoises j) t)).length).toFloat / N.toFloat

#eval IO.println s!"running empirical CDF at t = 0.057 (mid-support) over growing prefixes: \
  F̂_5 = {empCdfPrefix 5 0.057}, F̂_10 = {empCdfPrefix 10 0.057}, F̂_15 = {empCdfPrefix 15 0.057}, \
  F̂_20 = {empCdfPrefix 20 0.057} (→ empCdf 0.057 = {empCdf 0.057}, the estimate of cdf noiseLaw 0.057)"

-- Positive — the full prefix `N = 20` *is* the empirical CDF: `empCDF Λ V γ 20 t` evaluated on this
-- sample. The running mean and the count-fraction `empCdf` coincide at `N = 20` (the shadow of the
-- `empCDF` definition as a normalized indicator sum).
#eval assertTrue "running mean at N = 20 equals the empirical CDF (empCDF is the normalized indicator sum)"
  ([0.0, zLow, 0.5, zHigh, 1.0].all (fun t => empCdfPrefix 20 t == empCdf t))

-- Positive — every running prefix is a valid CDF value in `[0,1]`: a mean of the `[0,1]`-valued
-- indicators stays in `[0,1]` (the `integrable_nullBelow` / boundedness hypothesis feeding the SLLN).
#eval assertTrue "every running prefix mean lies in [0,1] (bounded indicators ⇒ bounded average)"
  ([1, 2, 5, 10, 15, 20].all (fun N =>
    [0.0, zLow, 0.5, zHigh, 1.0].all (fun t =>
      Spec.leBool 0.0 (empCdfPrefix N t) && Spec.leBool (empCdfPrefix N t) 1.0)))

-- Positive — the limit value `cdf noiseLaw 1 = 1` is already attained at *every* finite `N`: all
-- indicators `1{noiseᵢ ≤ 1}` are `1` (`nullNoise_mem_Icc`), so each running mean is exactly `1`. The
-- empirical CDF converges to the saturation endpoint trivially there.
#eval assertTrue "running mean saturates to cdf 1 = 1 at every prefix (all noises ≤ 1)"
  ([1, 2, 5, 10, 15, 20].all (fun N => empCdfPrefix N 1.0 == 1.0))

-- Negative control — consistency is *non-vacuous*: the running estimate genuinely changes with `N`
-- (an early prefix differs from the full sample at some interior threshold), so the convergence
-- `F̂_N → cdf noiseLaw t` is a real limit being approached, not a constant already equal to its limit
-- at `N = 5`. A degenerate (point-mass) sample would make every prefix equal and the SLLN vacuous.
#eval assertTrue "running empirical CDF is non-trivial: an early prefix differs from the full sample"
  ([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, zLow].any (fun t => !(empCdfPrefix 5 t == empCdfPrefix 20 t)))

/-! ### Pointwise finite-sample concentration (step c): the Hoeffding / DKW-at-a-point bound

`empCDF_concentration` proves, for the i.i.d. null sequence, that at any fixed threshold `t`,

  ℙ(|F̂_N(t) − cdf noiseLaw t| ≥ ε) ≤ 2·exp(−2·N·ε²),

the Dvoretzky–Kiefer–Wolfowitz inequality at a single point, with the sharp Hoeffding exponent (the
one-sided `empCDF_upper_tail`/`empCDF_lower_tail` give `exp(−2Nε²)` each; their union the factor `2`).
The probability is noncomputable — a measure on the infinite product of draws — but two computable
shadows make the statement concrete: the tail-bound *function* `2·exp(−2Nε²)`, and the *observed*
deviation of the prefix empirical CDF from the full-sample estimate, which that bound governs. -/

/-- The one-sided Hoeffding tail `exp(−2Nε²)` (`empCDF_upper_tail` / `empCDF_lower_tail`). -/
def oneSidedBound (N : Nat) (ε : Float) : Float := Float.exp (-2.0 * N.toFloat * ε * ε)

/-- The two-sided DKW-at-a-point bound `2·exp(−2Nε²)` (`empCDF_concentration`). -/
def hoeffdingBound (N : Nat) (ε : Float) : Float := 2.0 * oneSidedBound N ε

#eval IO.println s!"Hoeffding two-sided bound 2·exp(−2Nε²): (N=20,ε=0.3) = {hoeffdingBound 20 0.3}, \
  (N=20,ε=0.15) = {hoeffdingBound 20 0.15}, (N=10,ε=0.3) = {hoeffdingBound 10 0.3}, \
  trivial (ε=0) = {hoeffdingBound 20 0.0}"

-- Positive — the two-sided bound is exactly twice the one-sided tail: the union bound assembling
-- `empCDF_concentration` from `empCDF_upper_tail` + `empCDF_lower_tail`, each `exp(−2Nε²)`.
#eval assertTrue "two-sided Hoeffding bound = 2 × one-sided tail (upper + lower)"
  ([(5, 0.1), (10, 0.2), (20, 0.3)].all (fun p => hoeffdingBound p.1 p.2 == 2.0 * oneSidedBound p.1 p.2))

-- Positive — the bound tightens with more samples: doubling `N` shrinks the tail (the `N`-dependence
-- of the `−2Nε²` exponent, i.e. the finite-sample consistency rate).
#eval assertTrue "Hoeffding bound decreases in N (more draws ⇒ sharper concentration)"
  ([0.15, 0.2, 0.3].all (fun ε => Spec.leBool (hoeffdingBound 40 ε) (hoeffdingBound 20 ε)
      && Spec.leBool (hoeffdingBound 20 ε) (hoeffdingBound 10 ε)))

-- Positive — the bound tightens with a looser tolerance `ε` (the `ε²` in the exponent).
#eval assertTrue "Hoeffding bound decreases in ε (larger tolerance ⇒ smaller exceedance probability)"
  ([5, 10, 20].all (fun N => Spec.leBool (hoeffdingBound N 0.4) (hoeffdingBound N 0.2)))

-- Positive — the bound is *non-vacuous* (a genuine probability bound `< 1`) once `2Nε² > ln 2`; at
-- `N = 20`, `ε = 0.3` it is `≈ 0.055`, so the empirical CDF is within 0.3 of the truth w.p. ≥ 0.945.
#eval assertTrue "Hoeffding bound is non-vacuous (< 1) at N = 20, ε = 0.3"
  (Spec.ltBool (hoeffdingBound 20 0.3) 1.0)

-- Negative control — at `ε = 0` the bound is exactly the trivial constant `2` (the vacuous `ℙ ≤ 2`):
-- concentration says nothing without a positive tolerance, so the `ε²` in the exponent does the work.
#eval assertTrue "at ε = 0 the Hoeffding bound is the trivial constant 2 (vacuous without tolerance)"
  (hoeffdingBound 20 0.0 == 2.0)

/-! The bound governs the *observed* fluctuation: as the prefix grows, the empirical CDF `F̂_N`
concentrates around the full-sample estimate. With tolerance `ε = 0.3`, every prefix of `≥ 3` draws
stays within `ε` of the full sample uniformly over the threshold grid, while the tiniest prefixes
(`N = 1, 2`) deviate by `0.5` — exactly the weak-`N` regime where `2·exp(−2Nε²)` is still near `2`. -/

/-- Threshold grid spanning the tight null-noise band `[0.048, 0.062]` plus the `[0,1]` tails. -/
def devGrid : List Float := [-0.01, 0.0, 0.05, 0.055, 0.057, 0.06, 0.062, 0.2, 0.5, 1.0]

/-- Sup-over-the-grid deviation of the prefix-`N` empirical CDF from the full-sample estimate — the
quantity the two-sided bound controls (a computable proxy for `|F̂_N(t) − cdf noiseLaw t|`). -/
def maxDev (N : Nat) : Float :=
  (devGrid.map (fun t => Float.abs (empCdfPrefix N t - empCdf t))).foldl max 0.0

#eval IO.println s!"max |F̂_N − F̂_20| over the grid: N=1 {maxDev 1}, N=2 {maxDev 2}, N=3 {maxDev 3}, \
  N=5 {maxDev 5}, N=10 {maxDev 10}, N=20 {maxDev 20}"

-- Positive — concentration: with enough draws the empirical CDF settles within `ε = 0.3` of the
-- full-sample estimate, uniformly over thresholds (the deviation the two-sided bound governs).
#eval assertTrue "empirical CDF concentrates: max deviation ≤ ε = 0.3 for every prefix of ≥ 3 draws"
  ([3, 5, 10, 15, 20].all (fun N => Spec.leBool (maxDev N) 0.3))

-- Negative control — at the tiniest prefixes (`N = 1, 2`) the empirical CDF still deviates by `0.5 > ε`,
-- so concentration genuinely needs `N` to grow: this is the regime where `2·exp(−2Nε²)` is near `2`,
-- i.e. the bound is honestly vacuous and the empirical CDF has not yet concentrated.
#eval assertTrue "concentration needs N to grow: N = 1, 2 prefixes deviate by > ε = 0.3"
  ([1, 2].all (fun N => Spec.ltBool 0.3 (maxDev N)))

/-! ### Quantile transfer (step d): consistency of the empirical percentiles

`empQuantile_tendsto` inverts steps (b)–(c): wherever the true null CDF is continuous and strictly
increasing through a level `p` at the quantile `q` (`StraddlesQuantile`), *any* lower empirical
`p`-quantile (`IsLowerQuantile`: the CDF is `< p` to its left and `≥ p` to its right) converges almost
surely to `q` as `N → ∞`. This is the honest consistency statement for the 5%/95% percentile
thresholds the `Z_test` chooser uses. The limit `q` is noncomputable (a quantile of the law
`noiseLaw`), so — exactly as for steps (b)/(c) — we exercise it through the **full-sample** quantile
`q̂₂₀` standing in for `q`, and watch the prefix-`N` empirical quantile `q̂_N` settle toward it. -/

/-- The first `N ≤ 20` null noises, as the candidate set for the prefix empirical quantile. -/
def prefixNoises (N : Nat) : List Float :=
  ((List.finRange 20).filter (fun j => decide (j.val < N))).map zNullNoises

/-- The **lower empirical `p`-quantile** of the first `N` null draws: the smallest sampled noise `v`
whose running empirical CDF `F̂_N(v)` reaches `p` (`min`-fold over the qualifying draws, falling back
to `1`). This is the computable shadow of `IsLowerQuantile (empCDF … N) p` — `inf {t | p ≤ F̂_N t}`
for the right-continuous step CDF — the object `empQuantile_tendsto` drives to the true quantile. -/
def empQuantilePrefix (N : Nat) (p : Float) : Float :=
  ((prefixNoises N).filter (fun v => Spec.leBool p (empCdfPrefix N v))).foldl min 1.0

/-- The full-sample (`N = 20`) lower `p`-quantile — the computable stand-in for the true quantile `q`
of `noiseLaw` that `empQuantile_tendsto` sends the prefix quantiles to. -/
def empQuantile20 (p : Float) : Float := empQuantilePrefix 20 p

/-- Deviation of the prefix-`N` lower `p`-quantile from the full-sample limit stand-in `q̂₂₀` — the
computable proxy for `|q̂_N − q|` that `empQuantile_tendsto` drives to `0`. -/
def quantileDev (N : Nat) (p : Float) : Float :=
  Float.abs (empQuantilePrefix N p - empQuantile20 p)

#eval IO.println s!"empirical median (p = 0.5) over growing prefixes: q̂_3 = {empQuantilePrefix 3 0.5}, \
  q̂_5 = {empQuantilePrefix 5 0.5}, q̂_10 = {empQuantilePrefix 10 0.5}, q̂_15 = {empQuantilePrefix 15 0.5}, \
  q̂_20 = {empQuantile20 0.5} (the limit stand-in q)"

#eval IO.println s!"quantile triple at full sample (q̂₂₀): 5% = {empQuantile20 0.05}, 50% = {empQuantile20 0.5}, \
  95% = {empQuantile20 0.95}; median dev at N=10 {quantileDev 10 0.5} vs 5%-tail dev at N=10 {quantileDev 10 0.05}"

-- Positive — `IsLowerQuantile` right-property at the full sample: `p ≤ F̂₂₀(q̂₂₀)`. The lower
-- `p`-quantile genuinely reaches level `p` (here with equality, `p ∈ {0.05, 0.5, 0.95}` being multiples
-- of `1/20`) — the half of `IsLowerQuantile` feeding `empQuantile_tendsto`.
#eval assertTrue "lower p-quantile reaches level p: p ≤ F̂₂₀(q̂₂₀) for p ∈ {0.05, 0.5, 0.95}"
  ([0.05, 0.5, 0.95].all (fun p => Spec.leBool p (empCdf (empQuantile20 p))))

-- Positive — the empirical quantile is monotone in the level `p` (order statistics are nondecreasing):
-- `q̂₂₀(0.05) ≤ q̂₂₀(0.5) ≤ q̂₂₀(0.95)`, the quantile-function shadow of `monotone_cdf` inverted.
#eval assertTrue "empirical quantile is monotone in p: q̂₂₀(5%) ≤ q̂₂₀(50%) ≤ q̂₂₀(95%)"
  (Spec.leBool (empQuantile20 0.05) (empQuantile20 0.5)
    && Spec.leBool (empQuantile20 0.5) (empQuantile20 0.95))

-- Positive — every empirical quantile is a fraction in `[0,1]` (the percentiles live in the null
-- support, `zLowFn_nonneg`/`_le_one` and friends).
#eval assertTrue "empirical quantiles lie in [0,1] for p ∈ {0.05, 0.5, 0.95}"
  ([0.05, 0.5, 0.95].all (fun p =>
    Spec.leBool 0.0 (empQuantile20 p) && Spec.leBool (empQuantile20 p) 1.0))

-- Positive — quantile transfer (consistency): the prefix-`N` empirical median settles toward the
-- full-sample limit, within `0.02` for every prefix of `≥ 3` draws — the computable shadow of
-- `empQuantile_tendsto` (almost-sure `q̂_N → q` at the strictly-straddled median).
#eval assertTrue "empirical median converges: |q̂_N − q̂₂₀| ≤ 0.02 for every prefix of ≥ 3 draws"
  ([3, 5, 10, 15, 20].all (fun N => Spec.leBool (quantileDev N 0.5) 0.02))

-- Negative control — consistency is non-vacuous: the prefix median genuinely *moves* with `N` (some
-- prefix differs from the full-sample limit), so `q̂_N → q̂₂₀` is a real limit being approached, not a
-- value already constant at `N = 3`.
#eval assertTrue "convergence is non-vacuous: some prefix median differs from the full-sample limit"
  ([3, 5, 10, 15].any (fun N => !(empQuantilePrefix N 0.5 == empQuantile20 0.5)))

-- Negative control — the convergence is hypothesis-sensitive: the lower `5%`-tail quantile (where the
-- CDF is flatter and the straddle sparser, fewer draws to pin it) deviates *more* at `N = 10` than the
-- median does — the empirical signature of `StraddlesQuantile` being a genuine, needed hypothesis, not
-- automatic. A flat CDF region (no strict straddle) would defeat consistency entirely.
#eval assertTrue "hypothesis-sensitive: the 5%-tail quantile deviates more at N=10 than the well-straddled median"
  (Spec.ltBool (quantileDev 10 0.5) (quantileDev 10 0.05))

end NN.Examples.Factorization.Discovery
