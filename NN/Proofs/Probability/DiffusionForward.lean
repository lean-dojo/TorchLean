/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.MeasureTheory.Measure.Prod
public import Mathlib.MeasureTheory.Measure.Typeclasses.Probability
public import Mathlib.Probability.Distributions.Gaussian.Multivariate
public import Mathlib.Probability.Kernel.Composition.Prod

/-!
# Diffusion forward process: Gaussian noising

This file gives a small, Mathlib-backed formalization of the *forward* (noising) step used in
diffusion models, expressed as an affine pushforward of the standard Gaussian measure.

We work in a finite-dimensional real inner product space `E` equipped with its Borel σ-algebra.

Main definitions:
* `forwardNoising a b x` : the measure of `x' = a • x + b • z` with `z ∼ stdGaussian E`.
* `forwardKernel a b` : the associated Markov kernel `x ↦ forwardNoising a b x`.

Main facts:
* `forwardNoising` is Gaussian (`ProbabilityTheory.IsGaussian`), hence a probability measure.
* `forwardKernel` is a Markov kernel (`ProbabilityTheory.IsMarkovKernel`).
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

The definition is written in stages so typeclass inference can see a Gaussian pushforward through a
linear map followed by translation; this lemma is the cleaner formula downstream proofs usually want.
-/
@[simp]
lemma forwardNoising_eq_map (a b : ℝ) (x : E) :
    forwardNoising (E := E) a b x =
      (ProbabilityTheory.stdGaussian E).map (fun z ↦ a • x + b • z) := by
  unfold forwardNoising
  simpa [Function.comp] using
    (Measure.map_map (μ := ProbabilityTheory.stdGaussian E)
      (g := fun y : E ↦ a • x + y)
      (f := (b • (ContinuousLinearMap.id ℝ E) : E →L[ℝ] E))
      (by fun_prop) (by fun_prop))

/-- Affine images of a finite-dimensional standard Gaussian are Gaussian. -/
instance (a b : ℝ) (x : E) : ProbabilityTheory.IsGaussian (forwardNoising (E := E) a b x) := by
  unfold forwardNoising
  infer_instance

/-- Every forward-noising measure has total mass one. -/
instance (a b : ℝ) (x : E) : IsProbabilityMeasure (forwardNoising (E := E) a b x) := by
  infer_instance

/-- The explicit total-mass theorem for the forward-noising measure. -/
@[simp]
lemma forwardNoising_univ (a b : ℝ) (x : E) : forwardNoising (E := E) a b x Set.univ = 1 := by
  simpa using (measure_univ : forwardNoising (E := E) a b x Set.univ = 1)

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
lemma forwardKernel_apply (a b : ℝ) (x : E) :
    forwardKernel (E := E) a b x = forwardNoising (E := E) a b x := by
  classical
  have hg : Measurable (fun p : E × E ↦ a • p.1 + b • p.2) := by fun_prop
  have hmk : Measurable (Prod.mk x : E → E × E) := by fun_prop
  have hf' : Measurable (b • (id : E → E)) := by fun_prop
  have hh : Measurable (fun y : E ↦ a • x + y) := by fun_prop
  unfold forwardKernel forwardNoising
  simp [Kernel.map_apply, hg, Kernel.prod_apply, Kernel.id_apply, Kernel.const_apply,
    Measure.dirac_prod]
  rw [Measure.map_map hg hmk]
  rw [Measure.map_map hh hf']
  rfl

/-- Each transition distribution of the forward kernel is Gaussian. -/
lemma isGaussian_forwardKernel (a b : ℝ) (x : E) :
    ProbabilityTheory.IsGaussian (forwardKernel (E := E) a b x) := by
  simpa [forwardKernel_apply (E := E) a b x] using
    (inferInstance : ProbabilityTheory.IsGaussian (forwardNoising (E := E) a b x))

end

end NN.Proofs.Probability
