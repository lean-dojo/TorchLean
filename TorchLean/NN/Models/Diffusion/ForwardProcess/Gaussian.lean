/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Probability.Distributions.Gaussian.Multivariate
-- Fernique's theorem (`IsGaussian.integrable_id`) is only needed inside the proof of
-- `integral_id_forwardNoising`, so it stays a private import: no statement here mentions it.
import Mathlib.Probability.Distributions.Gaussian.Fernique

/-!
# Diffusion forward process: Gaussian noising

The forward (noising) step of a diffusion model is an affine pushforward of the standard Gaussian
measure on a finite-dimensional real inner product space `E` with its Borel $\sigma$-algebra.
`forwardNoising a b x` is the law of `x' = a • x + b • z` with `z ∼ stdGaussian E`, and
`forwardKernel a b` is the Markov kernel `x ↦ forwardNoising a b x`
(`ProbabilityTheory.IsMarkovKernel`). The law is Gaussian (`ProbabilityTheory.IsGaussian`), hence a
probability measure, and the step has mean `a • x` (`integral_id_forwardNoising`) and isotropic
noise of scale `b` (`variance_dual_forwardNoising`).

`IsGaussian` says that the law is Gaussian without saying which Gaussian it is. Anything that
reasons about a DDPM noise schedule needs the parameters, so the first moment and the dual form of
the covariance are recorded here, and callers need not repeat the pushforward computation.

In the notation of Ho, Jain and Abbeel (2020), `forwardKernel (√(1 - β_t)) (√β_t)` is the
one-step kernel of their eq. (2), and `forwardNoising (√ᾱ_t) (√(1 - ᾱ_t)) x₀` is the closed form
of eq. (4). So `integral_id_forwardNoising` and `variance_dual_forwardNoising` give that closed
form its mean `√ᾱ_t • x₀` and, once `ᾱ_t ≤ 1`, variance `1 - ᾱ_t` along every unit direction.

Reference for the schedule these parameters feed:
Jonathan Ho, Ajay Jain, and Pieter Abbeel (2020), *Denoising Diffusion Probabilistic Models*,
NeurIPS 2020, arXiv:2006.11239, eqs. (2) and (4).
-/

@[expose] public section

namespace TorchLean.Diffusion

open MeasureTheory ProbabilityTheory
open scoped ProbabilityTheory

noncomputable section

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [FiniteDimensional ℝ E]
    [MeasurableSpace E] [BorelSpace E]

/-- The law of one forward noising step `x' = a • x + b • z` of a diffusion model, with
`z ∼ stdGaussian E`.

The clean state `x` is scaled by `a`, and isotropic Gaussian noise scaled by `b` is added, as in
the forward process of DDPM-style models. The schedule that chooses `a` and `b` belongs to the
model spec, and the theorems here hold for every `a` and `b`.
-/
noncomputable
def forwardNoising (a b : ℝ) (x : E) : Measure E :=
  ((ProbabilityTheory.stdGaussian E).map (b • (ContinuousLinearMap.id ℝ E))).map
    (fun y ↦ a • x + y)

/--
`forwardNoising` is the pushforward of the standard Gaussian by the affine map
`z ↦ a • x + b • z`.

The definition pushes forward in two stages, through a linear map and then a translation, so that
typeclass inference finds the Gaussian instance. Downstream proofs usually want the single map.
-/
@[simp]
theorem forwardNoising_eq_map (a b : ℝ) (x : E) :
    forwardNoising (E := E) a b x =
      (ProbabilityTheory.stdGaussian E).map (fun z ↦ a • x + b • z) := by
  unfold forwardNoising
  rw [Measure.map_map (μ := ProbabilityTheory.stdGaussian E)
      (g := fun y : E ↦ a • x + y)
      (f := (b • (ContinuousLinearMap.id ℝ E) : E →L[ℝ] E))
      (by fun_prop) (by fun_prop)]
  apply Measure.map_congr
  filter_upwards with z
  simp [Function.comp]

/-- Affine images of a finite-dimensional standard Gaussian are Gaussian. -/
instance (a b : ℝ) (x : E) : ProbabilityTheory.IsGaussian (forwardNoising (E := E) a b x) := by
  unfold forwardNoising
  infer_instance

/-- Every forward-noising measure has total mass one. -/
instance (a b : ℝ) (x : E) : IsProbabilityMeasure (forwardNoising (E := E) a b x) := by
  infer_instance

/-- Total mass of the forward-noising measure, as a named theorem for downstream references.
`measure_univ` already proves it through the `IsProbabilityMeasure` instance, so it is not a
`simp` lemma. -/
theorem forwardNoising_univ (a b : ℝ) (x : E) : forwardNoising (E := E) a b x Set.univ = 1 :=
  measure_univ

/--
The mean of one forward-noising step is the scaled clean state:

`E[x'] = a • x`.

The noise term contributes nothing because the standard Gaussian is centred
(`ProbabilityTheory.integral_id_stdGaussian`), so all that survives the pushforward is the constant
`a • x`. Splitting the integral of the sum needs the identity to be integrable under a Gaussian
measure, a consequence of Fernique's theorem that Mathlib exposes as
`ProbabilityTheory.IsGaussian.integrable_id`.
-/
theorem integral_id_forwardNoising (a b : ℝ) (x : E) :
    ∫ y, y ∂(forwardNoising (E := E) a b x) = a • x := by
  have hNoiseIntegrable : Integrable (fun z : E => b • z) (ProbabilityTheory.stdGaussian E) :=
    (ProbabilityTheory.IsGaussian.integrable_id
      (μ := ProbabilityTheory.stdGaussian E)).smul b
  have hNoiseMean : ∫ z : E, b • z ∂(ProbabilityTheory.stdGaussian E) = 0 := by
    rw [integral_smul, ProbabilityTheory.integral_id_stdGaussian, smul_zero]
  rw [forwardNoising_eq_map, integral_map (by fun_prop) (by fun_prop),
    integral_add (integrable_const _) hNoiseIntegrable, hNoiseMean, add_zero]
  simp

/--
Every continuous linear functional `L` of a forward-noising step has variance `b ^ 2 * ‖L‖ ^ 2`.

The statement is the dual form of "the covariance operator is `b ^ 2` times the identity". It is
stated against `StrongDual` because Mathlib's multivariate Gaussian API is built on that form
(`ProbabilityTheory.variance_dual_stdGaussian`) and because a proof about a single coordinate of the
noised state consumes it. Taking `L` to be the inner product with a unit vector gives variance
`b ^ 2` along that direction, and since the value does not depend on the direction, the noise is
isotropic.

The right side has no `a` and no `x`, because translating by the constant `a • x` moves the mean
and leaves the spread alone.
-/
theorem variance_dual_forwardNoising (a b : ℝ) (x : E) (L : StrongDual ℝ E) :
    Var[L; forwardNoising (E := E) a b x] = b ^ 2 * ‖L‖ ^ 2 := by
  rw [forwardNoising_eq_map,
    variance_map L.continuous.aemeasurable (Measurable.aemeasurable (by fun_prop))]
  have hAffine : L ∘ (fun z : E => a • x + b • z) = fun z : E => L (a • x) + b * L z := by
    ext z; simp
  rw [hAffine, variance_const_add (by fun_prop), variance_const_mul,
    ProbabilityTheory.variance_dual_stdGaussian]

/-- Forward noising kernel for a diffusion step, as a Markov kernel. -/
noncomputable
def forwardKernel (a b : ℝ) : Kernel E E :=
  Kernel.map (Kernel.id ×ₖ Kernel.const E (ProbabilityTheory.stdGaussian E))
    (fun p : E × E ↦ a • p.1 + b • p.2)

/--
The forward diffusion transition is a Markov kernel.

With the instance, later statements compose diffusion steps as kernels and need not carry the
measurability and probability-mass facts by hand.
-/
instance (a b : ℝ) : IsMarkovKernel (forwardKernel (E := E) a b) := by
  classical
  refine Kernel.IsMarkovKernel.map
    (κ := (Kernel.id ×ₖ Kernel.const E (ProbabilityTheory.stdGaussian E)))
    (f := fun p : E × E ↦ a • p.1 + b • p.2) (by fun_prop)

/--
At state `x` the kernel is the forward-noising measure at `x`.

The kernel is built from `id × const stdGaussian` so that it fits Mathlib's kernel composition,
and the theorem connects that construction to the noising formula of `forwardNoising`.
-/
theorem forwardKernel_apply (a b : ℝ) (x : E) :
    forwardKernel (E := E) a b x = forwardNoising (E := E) a b x := by
  classical
  have hg : Measurable (fun p : E × E ↦ a • p.1 + b • p.2) := by fun_prop
  have hmk : Measurable (Prod.mk x : E → E × E) := by fun_prop
  have hf' : Measurable (((b • (ContinuousLinearMap.id ℝ E) : E →L[ℝ] E) : E → E)) := by
    fun_prop
  have hh : Measurable (fun y : E ↦ a • x + y) := by fun_prop
  unfold forwardKernel forwardNoising
  simp [Kernel.map_apply, hg, Kernel.prod_apply, Kernel.id_apply, Kernel.const_apply,
    Measure.dirac_prod]
  rw [Measure.map_map hg hmk]
  calc
    Measure.map ((fun p : E × E ↦ a • p.1 + b • p.2) ∘ Prod.mk x)
          (ProbabilityTheory.stdGaussian E) =
        Measure.map
          ((fun y : E ↦ a • x + y) ∘
            ((b • (ContinuousLinearMap.id ℝ E) : E →L[ℝ] E) : E → E))
          (ProbabilityTheory.stdGaussian E) := by
      apply Measure.map_congr
      filter_upwards with z
      simp [Function.comp]
    _ = Measure.map (fun y : E ↦ a • x + y)
          (Measure.map
            ((b • (ContinuousLinearMap.id ℝ E) : E →L[ℝ] E) : E → E)
            (ProbabilityTheory.stdGaussian E)) :=
      (Measure.map_map hh hf').symm

/-- Each transition distribution of the forward kernel is Gaussian. -/
theorem isGaussian_forwardKernel (a b : ℝ) (x : E) :
    ProbabilityTheory.IsGaussian (forwardKernel (E := E) a b x) := by
  simpa [forwardKernel_apply (E := E) a b x] using
    (inferInstance : ProbabilityTheory.IsGaussian (forwardNoising (E := E) a b x))

/-- The transition of the forward kernel at `x` has mean `a • x`. -/
theorem integral_id_forwardKernel (a b : ℝ) (x : E) :
    ∫ y, y ∂(forwardKernel (E := E) a b x) = a • x := by
  rw [forwardKernel_apply]
  exact integral_id_forwardNoising a b x

/-- The transition of the forward kernel at `x` is isotropic with noise scale `b`: every
continuous linear functional `L` has variance `b ^ 2 * ‖L‖ ^ 2`. -/
theorem variance_dual_forwardKernel (a b : ℝ) (x : E) (L : StrongDual ℝ E) :
    Var[L; forwardKernel (E := E) a b x] = b ^ 2 * ‖L‖ ^ 2 := by
  rw [forwardKernel_apply]
  exact variance_dual_forwardNoising a b x L

end

end TorchLean.Diffusion
