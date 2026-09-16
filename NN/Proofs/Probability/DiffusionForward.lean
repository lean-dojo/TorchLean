/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Probability.Distributions.Gaussian.Multivariate
-- Fernique's theorem (`IsGaussian.integrable_id`) is only needed inside the proof of
-- `integral_id_forwardNoising`, so it stays a private import: none of our statements mention it.
import Mathlib.Probability.Distributions.Gaussian.Fernique

/-!
# Diffusion forward process: Gaussian noising

This file gives a small, Mathlib-backed formalization of the *forward* (noising) step used in
diffusion models, expressed as an affine pushforward of the standard Gaussian measure.

We work in a finite-dimensional real inner product space `E` equipped with its Borel
$\sigma$-algebra.

Main definitions:
* `forwardNoising a b x` : the measure of `x' = a • x + b • z` with `z ∼ stdGaussian E`.
* `forwardKernel a b` : the associated Markov kernel `x ↦ forwardNoising a b x`.

Main facts:
* `forwardNoising` is Gaussian (`ProbabilityTheory.IsGaussian`), hence a probability measure.
* `forwardKernel` is a Markov kernel (`ProbabilityTheory.IsMarkovKernel`).
* the step has mean `a • x` (`integral_id_forwardNoising`) and isotropic noise of scale `b`
  (`variance_dual_forwardNoising`).

A note on why the moments get their own theorems: `IsGaussian` says the law is Gaussian, not *which*
Gaussian it is. Anything that reasons about a DDPM noise schedule needs the parameters, so we record
the first moment and the dual form of the covariance here instead of asking every caller to redo the
same pushforward computation. Reference for the schedule these parameters feed:
Ho, Jain, and Abbeel, *Denoising Diffusion Probabilistic Models*, NeurIPS 2020.
-/

@[expose] public section

namespace NN.Proofs.Probability

open MeasureTheory ProbabilityTheory
open scoped ProbabilityTheory

noncomputable section

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [FiniteDimensional ℝ E]
    [MeasurableSpace E] [BorelSpace E]

/-- Forward noising measure for a diffusion step:
`x' = a • x + b • z` with `z ∼ stdGaussian E`.

This is the measure-level analogue of the forward process used in DDPM-style diffusion models:
the current clean state `x` is scaled by `a`, and isotropic Gaussian noise is scaled by `b` and
added. The exact schedule that chooses `a` and `b` belongs to the model spec; this theorem layer
only needs the affine-Gaussian kernel shape.
-/
noncomputable
def forwardNoising (a b : ℝ) (x : E) : Measure E :=
  ((ProbabilityTheory.stdGaussian E).map (b • (ContinuousLinearMap.id ℝ E))).map
    (fun y ↦ a • x + y)

/--
The two-step definition of `forwardNoising` is equal to one direct affine pushforward.

The definition is written in stages so typeclass inference can see a Gaussian pushforward through
a linear map followed by translation; this lemma is the cleaner formula downstream proofs usually
want.
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

/-- The explicit total-mass theorem for the forward-noising measure. -/
@[simp]
theorem forwardNoising_univ (a b : ℝ) (x : E) : forwardNoising (E := E) a b x Set.univ = 1 := by
  exact measure_univ

/--
The mean of one forward-noising step is the scaled clean state:

`E[x'] = a • x`.

The noise term contributes nothing because the standard Gaussian is centred
(`ProbabilityTheory.integral_id_stdGaussian`), so all that survives the pushforward is the constant
`a • x`. Integrability of the identity under a Gaussian measure comes from Fernique's theorem, which
Mathlib exposes as `ProbabilityTheory.IsGaussian.integrable_id`; we need it to split the integral of
the sum.
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
Every continuous linear functional of a forward-noising step has variance `b ^ 2 * ‖L‖ ^ 2`.

This is the dual form of "the covariance operator is `b ^ 2` times the identity". We state it
against `StrongDual` rather than as a covariance matrix for two reasons: it is the form Mathlib's
multivariate Gaussian API is built on (`ProbabilityTheory.variance_dual_stdGaussian`), and it is
what a proof about a single coordinate of the noised state actually consumes. Taking `L` to be the
inner product with a unit vector gives variance `b ^ 2` along that direction, and the absence of any
dependence on the direction is exactly the isotropy claim.

Note that `a` and `x` do not appear on the right: scaling and translating by a constant shifts the
mean and leaves the spread alone.
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

This packages measurability and probability-mass obligations so later verification statements can
compose diffusion steps as kernels rather than manually carrying measure facts.
-/
instance (a b : ℝ) : IsMarkovKernel (forwardKernel (E := E) a b) := by
  classical
  refine Kernel.IsMarkovKernel.map
    (κ := (Kernel.id ×ₖ Kernel.const E (ProbabilityTheory.stdGaussian E)))
    (f := fun p : E × E ↦ a • p.1 + b • p.2) (by fun_prop)

/--
Applying the kernel at state `x` recovers exactly the forward-noising measure at `x`.

The kernel is built from `id × const stdGaussian` so it fits Mathlib kernel
composition; this theorem reconnects that construction to the simpler noising formula.
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

/-- The transition at `x` has mean `a • x`, read off the kernel rather than the measure. -/
theorem integral_id_forwardKernel (a b : ℝ) (x : E) :
    ∫ y, y ∂(forwardKernel (E := E) a b x) = a • x := by
  rw [forwardKernel_apply]
  exact integral_id_forwardNoising a b x

/-- The transition at `x` is isotropic with noise scale `b`, read off the kernel. -/
theorem variance_dual_forwardKernel (a b : ℝ) (x : E) (L : StrongDual ℝ E) :
    Var[L; forwardKernel (E := E) a b x] = b ^ 2 * ‖L‖ ^ 2 := by
  rw [forwardKernel_apply]
  exact variance_dual_forwardNoising a b x L

end

end NN.Proofs.Probability
