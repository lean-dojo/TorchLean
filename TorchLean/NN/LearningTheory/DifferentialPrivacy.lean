/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.MeasureTheory.Measure.ProbabilityMeasure

/-!
# Differential privacy (learning theory)

*Differential privacy* (DP) for TorchLean’s learning-theory layer is defined here in its event-wise
form, with the closure properties a larger construction needs when it composes DP facts:
monotonicity in `δ` and in `ε`, and post-processing (a measurable map of the outputs preserves DP).

The definition is general in two parameters:

- an *adjacency relation* `Adj : α → α → Prop` (e.g. “datasets differ in one example”), and
- a *randomized mechanism* `M : α → ProbabilityMeasure β`.

Then $(\varepsilon,\delta)$-DP is the standard **event-wise** bound

$$
\Pr[M(a)\in S]\leq e^\varepsilon\Pr[M(a')\in S]+\delta
$$

for all adjacent $a\sim a'$ and measurable events $S$.

In Mathlib terms, `M a` is a `ProbabilityMeasure β` and `(M a : Measure β) S` is the probability
of the event `S` as an `ENNReal`. One definition covers discrete and continuous outputs, and
measurability enters as a hypothesis where it is needed (`MeasurableSet S` for an event,
`Measurable f` for post-processing).

## Typical instantiations

- `α` is a dataset type, `Adj` means “replace/remove one example”.
- `β` is some model or statistic released to the outside world.
- `M` is a randomized training procedure / query mechanism.

The stability development in `TorchLean.NN.LearningTheory.Stability` uses datasets
`Tensor Z [n]`, with a `Fin n → Z` view, for a fixed sample size `n`. For DP, `Adj` can then be
defined as replacing one coordinate.

## References

- Dwork, McSherry, Nissim & Smith (2006), “Calibrating Noise to Sensitivity in Private Data
  Analysis”.
- Dwork & Roth (2014), “The Algorithmic Foundations of Differential Privacy” (monograph).
- Post-processing is a standard closure property of DP (see, e.g., Dwork–Roth (2014)).
-/

@[expose] public section

noncomputable section

namespace TorchLean.LearningTheory

open MeasureTheory

variable {α β γ : Type}

/-! ## Mechanisms -/

/--
A randomized mechanism from inputs `α` to outputs `β`.

We use `ProbabilityMeasure β` so probability mass is total ($\mu(\mathrm{univ})=1$) and so that
post-processing can be phrased using `ProbabilityMeasure.map` (pushforward along a function of the
output).
-/
abbrev Mechanism (α β : Type) [MeasurableSpace β] : Type :=
  α → ProbabilityMeasure β

/-! ## Differential privacy -/

/--
$(\varepsilon,\delta)$-differential privacy with respect to an adjacency relation `Adj`, in the
standard “event-wise” form: for all adjacent inputs $a\sim a'$ and all measurable events $S$,

$$
\Pr[M(a)\in S]\leq e^\varepsilon\Pr[M(a')\in S]+\delta.
$$

Probabilities are the measures `(M a : Measure β) S`, so the inequality lives in `ENNReal`. The
predicate imposes no symmetry on `Adj`, no nonnegativity condition on `ε` and no upper bound on
`δ`, and callers add the adjacency and budget restrictions their application needs.
-/
def DifferentialPrivacy (Adj : α → α → Prop) [MeasurableSpace β]
    (M : Mechanism α β) (ε : ℝ) (δ : ENNReal) : Prop :=
  ∀ a a', Adj a a' →
    ∀ S : Set β, MeasurableSet S →
      (M a : Measure β) S ≤ (ENNReal.ofReal (Real.exp ε)) * (M a' : Measure β) S + δ

/-- A common special case: $\delta=0$ (“pure DP”). -/
abbrev PureDP (Adj : α → α → Prop) [MeasurableSpace β]
    (M : Mechanism α β) (ε : ℝ) : Prop :=
  DifferentialPrivacy (α := α) (β := β) Adj M ε 0

/--
$\delta$-monotonicity: if a mechanism is $(\varepsilon,\delta_1)$-DP and
$\delta_1\leq\delta_2$, then it is also $(\varepsilon,\delta_2)$-DP.

A DP proof with a “clean” bound can then be reused under a looser $\delta$, for example after
taking a `sup` or adding a slack term.
-/
theorem DifferentialPrivacy.mono_delta {Adj : α → α → Prop} [MeasurableSpace β]
    {M : Mechanism α β} {ε : ℝ} {δ₁ δ₂ : ENNReal} (hδ : δ₁ ≤ δ₂) :
    DifferentialPrivacy (α := α) (β := β) Adj M ε δ₁ →
      DifferentialPrivacy (α := α) (β := β) Adj M ε δ₂ := by
  intro hdp a a' hadj S hS
  refine le_trans (hdp a a' hadj S hS) ?_
  -- Only the additive slack `δ` changes.
  gcongr

/--
$\varepsilon$-monotonicity: privacy at a smaller $\varepsilon$ implies privacy at a larger one.

With `mono_delta`, the predicate is monotone in the whole budget
$(\varepsilon,\delta)$. A pipeline can then advertise one budget for a step proved private at a
tighter one, which comes up whenever several mechanisms are composed and their budgets are
rounded to a common value.

The proof needs only that `Real.exp`, `ENNReal.ofReal` and multiplication on `ENNReal` are
monotone. We record it as a lemma so callers do not repeat that rewriting inline.
-/
theorem DifferentialPrivacy.mono_eps {Adj : α → α → Prop} [MeasurableSpace β]
    {M : Mechanism α β} {ε₁ ε₂ : ℝ} {δ : ENNReal} (hε : ε₁ ≤ ε₂) :
    DifferentialPrivacy (α := α) (β := β) Adj M ε₁ δ →
      DifferentialPrivacy (α := α) (β := β) Adj M ε₂ δ := by
  intro hdp a a' hadj S hS
  refine le_trans (hdp a a' hadj S hS) ?_
  -- `gcongr` steps through the sum, the `ENNReal` product, `ENNReal.ofReal` and `Real.exp` to `hε`.
  gcongr

/-! ## Post-processing -/

/--
Post-process a mechanism by applying a function to its output.

If `M` is DP, then so is `f ∘ M` for any measurable `f` that does *not* look at the private input.
DP folklore calls this **post-processing**, and it is one of the main reasons DP composes well
with downstream pipelines.

Formally, `postprocess M f` is the pushforward `(M a).map f` for each input `a`. Mathlib's
`ProbabilityMeasure.map` is total (an `f` that is not almost everywhere measurable yields a junk
Dirac mass), so the measurability hypothesis sits on `DifferentialPrivacy.postprocess`, the one
place that uses it.
-/
def Mechanism.postprocess [MeasurableSpace β] [MeasurableSpace γ] (M : Mechanism α β) (f : β → γ) :
    Mechanism α γ :=
  fun a => (M a).map f

/--
Post-processing theorem: measurable mappings of outputs preserve DP.

The proof is the standard one. The probability of an event `S` under the mapped output is the
probability of the preimage `f ⁻¹' S` under the original output, and DP for `M` applies to the
measurable set `f ⁻¹' S`.
-/
theorem DifferentialPrivacy.postprocess {Adj : α → α → Prop} [MeasurableSpace β]
    [MeasurableSpace γ] {M : Mechanism α β} {ε : ℝ} {δ : ENNReal} {f : β → γ}
    (hf : Measurable f) :
    DifferentialPrivacy (α := α) (β := β) Adj M ε δ →
      DifferentialPrivacy (α := α) (β := γ) Adj (Mechanism.postprocess M f) ε δ := by
  intro hdp a a' hadj S hS
  -- Reduce the event on the post-processed output to a preimage event on the original output.
  change ((M a).map f : Measure γ) S ≤
    (ENNReal.ofReal (Real.exp ε)) * ((M a').map f : Measure γ) S + δ
  rw [(M a).map_apply' hf.aemeasurable hS, (M a').map_apply' hf.aemeasurable hS]
  exact hdp a a' hadj (f ⁻¹' S) (hf hS)

end TorchLean.LearningTheory
