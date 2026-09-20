/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.MeasureTheory.Measure.ProbabilityMeasure

/-!
# Differential privacy (learning theory)

This file introduces a small, reusable vocabulary for *differential privacy* (DP) in TorchLean’s
learning-theory layer.

We isolate the core event-wise definition of differential privacy and the closure properties that
show up when DP is connected to learning theory, stability, and verification pipelines.

We keep the definition general by parameterizing over:

- an *adjacency relation* `Adj : α → α → Prop` (e.g. “datasets differ in one example”), and
- a *randomized mechanism* `M : α → ProbabilityMeasure β`.

Then $(\varepsilon,\delta)$-DP is the standard **event-wise** bound:

$$
\Pr[M(a)\in S]\leq e^\varepsilon\Pr[M(a')\in S]+\delta
$$

for all adjacent $a\sim a'$ and measurable events $S$.

We phrase this in mathlib using `ProbabilityMeasure` and `Measure`:

- `M a` is a `ProbabilityMeasure β`, and
- `(M a : Measure β) S` is the probability (as an `ENNReal`) of the event `S`.

This makes the definition work uniformly for discrete and continuous outputs, while keeping the
measurability side-conditions explicit.

We also include a couple of basic structural lemmas that are easy to reuse downstream (and that
are often needed to compose DP facts through a larger construction):

- monotonicity in `δ` and in `ε`, and
- post-processing (measurable mapping of outputs preserves DP).

## Typical instantiations

- `α` is a dataset type, `Adj` means “replace/remove one example”.
- `β` is some model or statistic released to the outside world.
- `M` is a randomized training procedure / query mechanism.

In this repository, the stability development (see `NN.MLTheory.LearningTheory.Stability.Core`) uses
datasets `Tensor Z [n]`, with a `Fin n → Z` view, for a fixed sample size `n`. For DP, one can
define `Adj` in terms of replacing one coordinate.

## References

- Dwork, McSherry, Nissim & Smith (2006), “Calibrating Noise to Sensitivity in Private Data
  Analysis”.
- Dwork & Roth (2014), “The Algorithmic Foundations of Differential Privacy” (monograph).
- Post-processing is a standard closure property of DP; see, e.g., Dwork–Roth (2014).
-/

@[expose] public section


noncomputable section

namespace NN.MLTheory.LearningTheory

open scoped BigOperators

open MeasureTheory

variable {α β γ : Type}

/-! ## Mechanisms -/

/--
A randomized mechanism from inputs `α` to outputs `β`.

We use `ProbabilityMeasure β` so probability mass is total ($\mu(\mathrm{univ})=1$) and so that
post-processing can be phrased using `ProbabilityMeasure.map` (pushforward along a measurable
function).
-/
abbrev Mechanism (α β : Type) [MeasurableSpace β] : Type :=
  α → ProbabilityMeasure β

/-! ## Differential privacy -/

/--
$(\varepsilon,\delta)$-differential privacy with respect to an adjacency relation `Adj`.

This is the standard “event-wise” definition:

for all adjacent inputs $a\sim a'$ and all measurable events $S$,

$$
\Pr[M(a)\in S]\leq e^\varepsilon\Pr[M(a')\in S]+\delta.
$$

We write probabilities as measures `(M a : Measure β) S` so the inequality lives in `ENNReal`.
The predicate imposes no symmetry on `Adj`, no nonnegativity condition on `ε`, and no upper bound
on `δ`; callers supply the adjacency and budget restrictions needed for their application.
-/
def DifferentialPrivacy (Adj : α → α → Prop) [MeasurableSpace β]
    (M : Mechanism α β) (ε : ℝ) (δ : ENNReal) : Prop :=
  ∀ a a', Adj a a' →
    ∀ S : Set β, MeasurableSet S →
      (M a : Measure β) S ≤ (ENNReal.ofReal (Real.exp ε)) * (M a' : Measure β) S + δ

/-! A common special case: $\delta=0$ (“pure DP”). -/
abbrev PureDP (Adj : α → α → Prop) [MeasurableSpace β]
    (M : Mechanism α β) (ε : ℝ) : Prop :=
  DifferentialPrivacy (α := α) (β := β) Adj M ε 0

/--
$\delta$-monotonicity: if a mechanism is $(\varepsilon,\delta_1)$-DP and
$\delta_1\leq\delta_2$, then it is also $(\varepsilon,\delta_2)$-DP.

This small lemma is useful when you:

- prove DP with a “clean” bound, then
- want to reuse it under a slightly looser $\delta$ (e.g. after taking a `sup`, or adding a slack
  term).
-/
theorem differentialPrivacy_mono_delta {Adj : α → α → Prop} [MeasurableSpace β]
    {M : Mechanism α β} {ε : ℝ} {δ₁ δ₂ : ENNReal} (hδ : δ₁ ≤ δ₂) :
    DifferentialPrivacy (α := α) (β := β) Adj M ε δ₁ →
      DifferentialPrivacy (α := α) (β := β) Adj M ε δ₂ := by
  intro hdp a a' hadj S hS
  have h := hdp a a' hadj S hS
  exact le_trans h (by
    -- `add_le_add_left` produces the inequality with `δ` on the left; rewrite by commutativity.
    simpa [add_comm, add_left_comm, add_assoc] using
      add_le_add_left hδ ((ENNReal.ofReal (Real.exp ε)) * (M a' : Measure β) S))

/--
$\varepsilon$-monotonicity: privacy at a smaller $\varepsilon$ implies privacy at a larger one.

Together with `differentialPrivacy_mono_delta` this says the predicate is monotone in the whole
budget $(\varepsilon,\delta)$. That is what lets a pipeline advertise one budget for a step that was
actually proved private at a tighter one, which happens constantly once several mechanisms are
composed and their budgets are rounded to a common value.

The mathematical content is only monotonicity of `Real.exp` together with monotonicity of
multiplication on `ENNReal`; we record it as a lemma so callers do not repeat that rewriting inline.
-/
theorem differentialPrivacy_mono_eps {Adj : α → α → Prop} [MeasurableSpace β]
    {M : Mechanism α β} {ε₁ ε₂ : ℝ} {δ : ENNReal} (hε : ε₁ ≤ ε₂) :
    DifferentialPrivacy (α := α) (β := β) Adj M ε₁ δ →
      DifferentialPrivacy (α := α) (β := β) Adj M ε₂ δ := by
  intro hdp a a' hadj S hS
  refine le_trans (hdp a a' hadj S hS) ?_
  -- `gcongr` does both monotonicity steps: `Real.exp` in the exponent, then the `ENNReal` product.
  gcongr

/-! ## Post-processing -/

/--
Post-process a mechanism by applying a measurable function to its output.

In DP folklore: if `M` is DP, then so is `f ∘ M` for any (measurable) `f` that does *not* look at
the private input. This is called **post-processing** and is one of the core reasons DP composes
well with downstream pipelines.

Formally, `postprocess M f` is the pushforward measure `(M a).map f` for each input `a`.
-/
def postprocess [MeasurableSpace β] [MeasurableSpace γ]
    (M : Mechanism α β) (f : β → γ) (hf : Measurable f) : Mechanism α γ :=
  fun a => ⟨(M a : Measure β).map f,
    (Measure.isProbabilityMeasure_map_iff hf.aemeasurable).mpr inferInstance⟩

/--
Post-processing theorem: measurable mappings of outputs preserve DP.

Proof idea (the standard one):

- the probability of an event `S` under the mapped output is the probability of the preimage
  `f ⁻¹' S` under the original output;
- apply DP for `M` to the measurable set `f ⁻¹' S`.
-/
theorem differentialPrivacy_postprocess {Adj : α → α → Prop} [MeasurableSpace β] [MeasurableSpace γ]
    {M : Mechanism α β} {ε : ℝ} {δ : ENNReal} {f : β → γ} (hf : Measurable f) :
    DifferentialPrivacy (α := α) (β := β) Adj M ε δ →
      DifferentialPrivacy (α := α) (β := γ) Adj (postprocess (α := α) (β := β) (γ := γ) M f hf) ε δ
        := by
  intro hdp a a' hadj S hS
  -- Reduce the event on the post-processed output to a preimage event on the original output.
  change ((M a).map f : Measure γ) S ≤
    (ENNReal.ofReal (Real.exp ε)) * ((M a').map f : Measure γ) S + δ
  rw [(M a).map_apply' hf.aemeasurable hS, (M a').map_apply' hf.aemeasurable hS]
  exact hdp a a' hadj (f ⁻¹' S) (hf hS)

end NN.MLTheory.LearningTheory
