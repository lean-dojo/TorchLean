/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.MeasureTheory.Measure.Typeclasses.Probability
public import Mathlib.Probability.Distributions.Gaussian.Multivariate

/-!
# Diffusion forward process: Gaussian law

We use this file as the mathlib-backed anchor for diffusion probability theory.

It formalizes a standard fact used implicitly throughout diffusion models:

If `Z` is standard Gaussian in a finite-dimensional Euclidean space `E`, then

`x_t := c0 • x0 + c1 • Z`

is Gaussian for any fixed `x0 : E` and scalar coefficients `c0,c1 : ℝ`.

In the DDPM/VP setting, the usual coefficients are:

- `c0 = sqrt(ᾱ_t)`
- `c1 = sqrt(1-ᾱ_t)`

At the spec layer (`NN.Spec.Generative.Diffusion.ForwardProcess`), we treat the noise `ε` as an
explicit tensor input. This file provides the probability-theory side: when that noise is sampled
from `stdGaussian`, the resulting distribution is Gaussian.

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

`x ↦ c0 • x0 + c1 • x`, where `x ~ stdGaussian`.

We define it as a composition of:
- a linear map (`x ↦ c1 • x`), and
- a translation (`y ↦ y + c0 • x0`),

so that Gaussian-closure lemmas in mathlib apply directly.
-/
def forwardGaussian (c0 c1 : ℝ) (x0 : E) : Measure E :=
  let μ : Measure E := stdGaussian E
  ((μ.map (c1 • (ContinuousLinearMap.id ℝ E))).map (fun y => y + c0 • x0))

instance (c0 c1 : ℝ) (x0 : E) : IsProbabilityMeasure (forwardGaussian (ι := ι) c0 c1 x0) := by
  -- We use that `stdGaussian` is a probability measure and measurable push-forwards preserve mass.
  let ν : Measure E := (stdGaussian E).map (c1 • (ContinuousLinearMap.id ℝ E))
  haveI : IsProbabilityMeasure ν := by
    dsimp [ν]
    exact Measure.isProbabilityMeasure_map (μ := stdGaussian E)
      (f := (c1 • (ContinuousLinearMap.id ℝ E) : E → E)) (by fun_prop)
  change IsProbabilityMeasure (ν.map (fun y : E => y + c0 • x0))
  exact Measure.isProbabilityMeasure_map (μ := ν) (f := fun y : E => y + c0 • x0) (by fun_prop)

theorem forwardGaussian_isGaussian (c0 c1 : ℝ) (x0 : E) :
    IsGaussian (forwardGaussian (ι := ι) c0 c1 x0) := by
  -- Gaussian laws are closed under continuous linear maps and translations.
  let ν : Measure E := (stdGaussian E).map (c1 • (ContinuousLinearMap.id ℝ E))
  haveI : IsGaussian ν := by
    dsimp [ν]
    exact isGaussian_map_of_measurable (μ := stdGaussian E)
      (L := c1 • (ContinuousLinearMap.id ℝ E)) (by fun_prop)
  change IsGaussian (ν.map (fun y : E => y + c0 • x0))
  infer_instance

end NN.MLTheory.Generative.Diffusion
