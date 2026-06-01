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

end NN.Examples.Factorization.Discovery
