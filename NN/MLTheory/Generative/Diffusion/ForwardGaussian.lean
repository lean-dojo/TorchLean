/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Probability.Distributions.Gaussian.Multivariate

/-!
# Diffusion forward process: Gaussian law

We use this file as the mathlib-backed anchor for diffusion probability theory.

It formalizes a standard fact used implicitly throughout diffusion models:

If $Z$ is standard Gaussian in a finite-dimensional Euclidean space $E$, then

$x_t = c_0 x_0 + c_1 Z$

is Gaussian for any fixed $x_0 \in E$ and scalar coefficients $c_0,c_1 \in \mathbb{R}$.

In the DDPM/VP setting, the usual coefficients are:

- $c_0 = \sqrt{\bar{\alpha}_t}$
- $c_1 = \sqrt{1-\bar{\alpha}_t}$

At the spec layer (`NN.Spec.Generative.Diffusion.ForwardProcess`), we treat the noise
$\varepsilon$ as an explicit tensor input. This file provides the probability-theory side: when
that noise is sampled from `stdGaussian`, the resulting distribution is Gaussian.

The result gives the exact law-level fact used by VP/DDPM forward processes: affine noising of a
fixed data point by standard Gaussian noise produces another Gaussian probability measure. We keep
the statement at this level because it is the reusable primitive needed by ELBO or SDE developments.

References:
- Ho, Jain, and Abbeel, "Denoising Diffusion Probabilistic Models", NeurIPS 2020.
- Song et al., "Score-Based Generative Modeling through Stochastic Differential Equations", ICLR
  2021.
-/

@[expose] public section

noncomputable section

namespace NN.MLTheory.Generative.Diffusion

open MeasureTheory ProbabilityTheory

variable {ι : Type*} [Fintype ι]

local notation "E" => EuclideanSpace ℝ ι

/--
Forward noising measure in a finite-dimensional Euclidean space:

$x \mapsto c_0 x_0 + c_1 x$, where $x$ follows the standard Gaussian law.

We define it as a composition of:
- a linear map $x \mapsto c_1 x$, and
- a translation $y \mapsto y + c_0 x_0$,

so that Gaussian-closure lemmas in mathlib apply directly.
-/
def forwardGaussian (c0 c1 : ℝ) (x0 : E) : Measure E :=
  let μ : Measure E := stdGaussian E
  ((μ.map (c1 • (ContinuousLinearMap.id ℝ E))).map (fun y => y + c0 • x0))

instance (c0 c1 : ℝ) (x0 : E) : IsProbabilityMeasure (forwardGaussian (ι := ι) c0 c1 x0) := by
  -- We use that `stdGaussian` is a probability measure and measurable push-forwards preserve mass.
  unfold forwardGaussian
  infer_instance

/-- Affine noising of a fixed point has a Gaussian law. This closure theorem does not identify
the marginal of a separately defined multistep diffusion chain. -/
theorem forwardGaussian_isGaussian (c0 c1 : ℝ) (x0 : E) :
    IsGaussian (forwardGaussian (ι := ι) c0 c1 x0) := by
  -- Gaussian laws are closed under continuous linear maps and translations.
  let ν : Measure E := (stdGaussian E).map (c1 • (ContinuousLinearMap.id ℝ E))
  have : IsGaussian ν := by
    dsimp [ν]
    exact isGaussian_map_of_measurable (μ := stdGaussian E)
      (L := c1 • (ContinuousLinearMap.id ℝ E)) (by fun_prop)
  change IsGaussian (ν.map (fun y : E => y + c0 • x0))
  infer_instance

end NN.MLTheory.Generative.Diffusion
