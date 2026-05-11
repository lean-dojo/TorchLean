/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Analysis.SpecialFunctions.Exp
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

Then `(ε, δ)`-DP is the standard **event-wise** bound:

`P[M a ∈ S] ≤ exp(ε) * P[M a' ∈ S] + δ` for all adjacent `a ~ a'` and measurable events `S`.

We phrase this in mathlib using `ProbabilityMeasure` and `Measure`:

- `M a` is a `ProbabilityMeasure β`, and
- `(M a : Measure β) S` is the probability (as an `ENNReal`) of the event `S`.

This makes the definition work uniformly for discrete and continuous outputs, while keeping the
measurability side-conditions explicit.

We also include a couple of basic structural lemmas that are easy to reuse downstream (and that
are often needed to compose DP facts through a larger construction):

- monotonicity in `δ`, and
- post-processing (measurable mapping of outputs preserves DP).

## Typical instantiations

- `α` is a dataset type, `Adj` means “replace/remove one example”.
- `β` is some model or statistic released to the outside world.
- `M` is a randomized training procedure / query mechanism.

In this repository, the stability development (see `NN.MLTheory.LearningTheory.Stability.Core`) uses
datasets of the form `Fin n → Z` for a fixed “sample size” `n`. For DP, one often uses a similar
encoding and defines `Adj` in terms of replacing one coordinate.

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

We use `ProbabilityMeasure β` so probability mass is total (`μ univ = 1`) and so that
post-processing can be phrased using `ProbabilityMeasure.map` (pushforward along a measurable
function).
-/
abbrev Mechanism (α β : Type) [MeasurableSpace β] : Type :=
  α → ProbabilityMeasure β

/-! ## Differential privacy -/

/--
`(ε, δ)`-differential privacy with respect to an adjacency relation `Adj`.

This is the standard “event-wise” definition:

for all adjacent inputs `a ~ a'` and all measurable events `S`,

`P[M a ∈ S] ≤ exp(ε) * P[M a' ∈ S] + δ`.

We write probabilities as measures `(M a : Measure β) S` so the inequality lives in `ENNReal`.
-/
def DifferentialPrivacy (Adj : α → α → Prop) [MeasurableSpace β]
    (M : Mechanism α β) (ε : ℝ) (δ : ENNReal) : Prop :=
  ∀ a a', Adj a a' →
    ∀ S : Set β, MeasurableSet S →
      (M a : Measure β) S ≤ (ENNReal.ofReal (Real.exp ε)) * (M a' : Measure β) S + δ

/-! A common special case: `δ = 0` (“pure DP”). -/
abbrev PureDP (Adj : α → α → Prop) [MeasurableSpace β]
    (M : Mechanism α β) (ε : ℝ) : Prop :=
  DifferentialPrivacy (α := α) (β := β) Adj M ε 0

/--
`δ`-monotonicity: if a mechanism is `(ε, δ₁)`-DP and `δ₁ ≤ δ₂`, then it is also `(ε, δ₂)`-DP.

This small lemma is useful when you:

- prove DP with a “clean” bound, then
- want to reuse it under a slightly looser `δ` (e.g. after taking a `sup`, or adding a slack term).
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
  fun a => (M a).map hf.aemeasurable

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
  have hpre : MeasurableSet (f ⁻¹' S) := hf hS
  have h := hdp a a' hadj (f ⁻¹' S) hpre
  -- Rewrite both sides via `ProbabilityMeasure.map_apply'` (ENNReal-level).
  simpa [postprocess, ProbabilityMeasure.map_apply', hS, hpre, hf] using h

end NN.MLTheory.LearningTheory
