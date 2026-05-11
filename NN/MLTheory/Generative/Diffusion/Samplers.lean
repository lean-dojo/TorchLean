/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Generative.Diffusion
public import NN.Spec.Dynamics.System
public import NN.MLTheory.LearningTheory.Robustness.Spec
public import NN.Proofs.Analysis.Lipschitz

import Mathlib.Data.List.FinRange

/-!
# Diffusion sampler theorems

This file collects the fully proved sampler facts that sit between TorchLean's executable diffusion
specs and the stability/verification APIs used elsewhere in the library.

We separate sampler mechanics from measure-level claims. ELBO optimality, reverse-SDE/PF-ODE
marginal equivalence, and consistency distillation require additional probability and regularity
assumptions; the theorems here prove the sampler-side facts that downstream proofs and verifiers
can reuse directly:

- schedule boundary values,
- zero-step sampler behavior, and
- the link between sampler adapters and `DynamicalSystem` transitions,
- L2 stability of explicit Euler probability-flow updates.

References:
- Ho, Jain, and Abbeel, "Denoising Diffusion Probabilistic Models", NeurIPS 2020.
- Song et al., "Score-Based Generative Modeling through Stochastic Differential Equations", ICLR
  2021.
- Hairer, Nørsett, and Wanner, *Solving Ordinary Differential Equations I*, for the Euler stability
  estimate used below.
-/

@[expose] public section

namespace NN.MLTheory.Generative.Diffusion

open _root_.Spec
open _root_.Spec.Tensor
open _root_.Generative.Diffusion

variable {α : Type} [Context α]
variable {T : Nat} {s : Shape}

/-- The VP schedule convention is `ᾱ₀ = 1`. -/
@[simp] theorem alphaBar_zero (sched : VPSchedule α T) :
    sched.alphaBar ⟨0, Nat.succ_pos T⟩ = 1 := by
  rfl

/-- With no reverse steps, DDPM returns its initial terminal-noise state. -/
@[simp] theorem ddpmSample_zero_steps (sched : VPSchedule α 0) (model : EpsModel α s)
    (x_T : Tensor α s) (noise : Fin 0 → Tensor α s) :
    ddpmSample (α := α) (T := 0) (s := s) sched model x_T noise = x_T := by
  simp [ddpmSample]

/-- With no reverse steps, deterministic DDIM returns its initial terminal-noise state. -/
@[simp] theorem ddimSample_zero_steps (sched : VPSchedule α 0) (model : EpsModel α s)
    (x_T : Tensor α s) :
    ddimSample (α := α) (T := 0) (s := s) sched model x_T = x_T := by
  simp [ddimSample]

/-- With zero Euler steps, the probability-flow sampler returns the initial state. -/
@[simp] theorem pfOdeSampleEuler_zero_steps (sch : VPLinearSchedule α) (model : EpsModel α s)
    (x1 : Tensor α s) :
    pfOdeSampleEuler (α := α) (s := s) sch model 0 x1 = x1 := by
  rfl

/-- The DDIM system adapter is definitionally the corresponding DDIM step. -/
@[simp] theorem ddimStepSystem_eq_step (sched : VPSchedule SpecScalar T)
    (model : EpsModel SpecScalar s) (k : Fin T) (x : SpecTensor s) :
    (ddimStepSystem (T := T) (s := s) sched model k).step x =
      ddimStep (α := SpecScalar) (T := T) (s := s) sched model k x := by
  rfl

/-- The probability-flow Euler system adapter is definitionally the Euler update. -/
@[simp] theorem pfOdeEulerSystem_eq_step (sch : VPLinearSchedule SpecScalar)
    (model : EpsModel SpecScalar s) (t dt : SpecScalar) (x : SpecTensor s) :
    (pfOdeEulerSystem (s := s) sch model t dt).step x =
      eulerStep (α := SpecScalar) (s := s)
        (pfOdeRhs (α := SpecScalar) (s := s) sch model) x t dt := by
  rfl

/-! ## Quantitative Euler stability for probability-flow samplers -/

/--
Subtraction algebra for two explicit Euler updates.

The identity

`(x + dt • fx) - (y + dt • fy) = (x - y) + dt • (fx - fy)`

is the tensor-level algebraic core behind stability and Lipschitz proofs for ODE samplers. We keep
it private because users should usually consume the norm and Lipschitz theorems below.
-/
private theorem sub_add_scaled_eq {s : Shape} (x y fx fy : Tensor ℝ s) (dt : ℝ) :
    (x + scaleSpec fx dt).subSpec (y + scaleSpec fy dt) =
      addSpec (subSpec x y) (scaleSpec (subSpec fx fy) dt) := by
  induction s with
  | scalar =>
      cases x with | scalar x0 =>
      cases y with | scalar y0 =>
      cases fx with | scalar fx0 =>
      cases fy with | scalar fy0 =>
      change Tensor.scalar ((x0 + fx0 * dt) - (y0 + fy0 * dt)) =
        Tensor.scalar ((x0 - y0) + ((fx0 - fy0) * dt))
      congr
      ring
  | dim n inner ih =>
      cases x with | dim xs =>
      cases y with | dim ys =>
      cases fx with | dim fxs =>
      cases fy with | dim fys =>
      simp [subSpec, addSpec, scaleSpec, map2Spec, mapSpec]
      apply congrArg Tensor.dim
      funext i
      exact ih (xs i) (ys i) (fxs i) (fys i)

/--
One explicit Euler step is stable in L2 up to the current state separation plus the RHS separation.

For an ODE `x' = f(x,t)`, the Euler update is `E(x)=x+dt f(x,t)`. This theorem proves the standard
numerical-analysis estimate

`‖E(x)-E(y)‖₂ ≤ ‖x-y‖₂ + |dt| ‖f(x,t)-f(y,t)‖₂`.

This proof uses TorchLean's real tensor norm library: the algebraic Euler-difference identity above,
the L2 triangle inequality, and L2 homogeneity of scalar multiplication. It is the bridge from the
executable PF-ODE sampler to quantitative stability/verification arguments.

Reference: this is the elementary one-step stability estimate underlying explicit Euler analyses;
see Hairer, Nørsett, and Wanner, *Solving Ordinary Differential Equations I*.
-/
theorem eulerStep_l2_distance_bound
    (f : Tensor ℝ s → ℝ → Tensor ℝ s) (t dt : ℝ) (x y : Tensor ℝ s) :
    NN.MLTheory.Robustness.Spec.tensorDistance (α := ℝ)
      (fun {s} => Proofs.tensorL2Norm (s := s))
      (eulerStep (α := ℝ) (s := s) f x t dt)
      (eulerStep (α := ℝ) (s := s) f y t dt)
      ≤ NN.MLTheory.Robustness.Spec.tensorDistance (α := ℝ)
          (fun {s} => Proofs.tensorL2Norm (s := s)) x y +
        |dt| * NN.MLTheory.Robustness.Spec.tensorDistance (α := ℝ)
          (fun {s} => Proofs.tensorL2Norm (s := s)) (f x t) (f y t) := by
  simp only [NN.MLTheory.Robustness.Spec.tensorDistance,
    NN.MLTheory.Robustness.Spec.tensor_distance_tensor_sub_eq_sub_spec, eulerStep]
  rw [sub_add_scaled_eq]
  calc
    Proofs.tensorL2Norm (addSpec (subSpec x y) (scaleSpec (subSpec (f x t) (f y t)) dt))
        ≤ Proofs.tensorL2Norm (subSpec x y) +
          Proofs.tensorL2Norm (scaleSpec (subSpec (f x t) (f y t)) dt) :=
            Proofs.tensor_l2_norm_triangle (subSpec x y)
              (scaleSpec (subSpec (f x t) (f y t)) dt)
    _ = Proofs.tensorL2Norm (subSpec x y) +
        |dt| * Proofs.tensorL2Norm (subSpec (f x t) (f y t)) := by
          rw [Proofs.tensor_l2_norm_scale]

/--
If the ODE right-hand side is `L`-Lipschitz in L2 at a fixed time, then one Euler step is
`(1 + |dt| L)`-Lipschitz in L2.

This is a quantitative sampler theorem: once we prove or import a Lipschitz bound for the neural
vector field, this theorem converts it into a certified bound for the actual update used by the
sampler.
-/
theorem eulerStep_l2_lipschitz_of_rhs_lipschitz
    (f : Tensor ℝ s → ℝ → Tensor ℝ s) (t dt L : ℝ)
    (h : NN.MLTheory.Robustness.Spec.isLipschitzContinuous (α := ℝ)
      (fun x : Tensor ℝ s => f x t)
      (fun {s} => Proofs.tensorL2Norm (s := s))
      (fun {s} => Proofs.tensorL2Norm (s := s)) L) :
    NN.MLTheory.Robustness.Spec.isLipschitzContinuous (α := ℝ)
      (fun x : Tensor ℝ s => eulerStep (α := ℝ) (s := s) f x t dt)
      (fun {s} => Proofs.tensorL2Norm (s := s))
      (fun {s} => Proofs.tensorL2Norm (s := s)) (1 + |dt| * L) := by
  intro x y
  have hstep := eulerStep_l2_distance_bound (s := s) f t dt x y
  have hf := h x y
  have hscaled :
      |dt| * NN.MLTheory.Robustness.Spec.tensorDistance (α := ℝ)
          (fun {s} => Proofs.tensorL2Norm (s := s)) (f x t) (f y t) ≤
        |dt| * (L * NN.MLTheory.Robustness.Spec.tensorDistance (α := ℝ)
          (fun {s} => Proofs.tensorL2Norm (s := s)) x y) := by
    exact mul_le_mul_of_nonneg_left hf (abs_nonneg dt)
  calc
    NN.MLTheory.Robustness.Spec.tensorDistance (α := ℝ)
        (fun {s} => Proofs.tensorL2Norm (s := s))
        (eulerStep (α := ℝ) (s := s) f x t dt)
        (eulerStep (α := ℝ) (s := s) f y t dt)
        ≤ NN.MLTheory.Robustness.Spec.tensorDistance (α := ℝ)
            (fun {s} => Proofs.tensorL2Norm (s := s)) x y +
          |dt| * NN.MLTheory.Robustness.Spec.tensorDistance (α := ℝ)
            (fun {s} => Proofs.tensorL2Norm (s := s)) (f x t) (f y t) := hstep
    _ ≤ NN.MLTheory.Robustness.Spec.tensorDistance (α := ℝ)
          (fun {s} => Proofs.tensorL2Norm (s := s)) x y +
        |dt| * (L * NN.MLTheory.Robustness.Spec.tensorDistance (α := ℝ)
          (fun {s} => Proofs.tensorL2Norm (s := s)) x y) := by
          linarith
    _ = (1 + |dt| * L) * NN.MLTheory.Robustness.Spec.tensorDistance (α := ℝ)
          (fun {s} => Proofs.tensorL2Norm (s := s)) x y := by
          ring

/--
Probability-flow Euler systems inherit the concrete L2 Euler-step Lipschitz bound.

For the PF-ODE vector field `pfOdeRhs sch model`, any certified `L`-Lipschitz bound on the RHS at
time `t` yields a `(1 + |dt| L)` Lipschitz bound on the `DynamicalSystem.step` used by TorchLean
trajectories. This is the theorem downstream PF-ODE certificates should target first.
-/
theorem pfOdeEulerSystem_l2_lipschitz_of_rhs_lipschitz
    (sch : VPLinearSchedule ℝ) (model : EpsModel ℝ s) (t dt L : ℝ)
    (h : NN.MLTheory.Robustness.Spec.isLipschitzContinuous (α := ℝ)
      (fun x : Tensor ℝ s => pfOdeRhs (α := ℝ) (s := s) sch model x t)
      (fun {s} => Proofs.tensorL2Norm (s := s))
      (fun {s} => Proofs.tensorL2Norm (s := s)) L) :
    NN.MLTheory.Robustness.Spec.isLipschitzContinuous (α := ℝ)
      (pfOdeEulerSystem (s := s) sch model t dt).step
      (fun {s} => Proofs.tensorL2Norm (s := s))
      (fun {s} => Proofs.tensorL2Norm (s := s)) (1 + |dt| * L) := by
  simpa [pfOdeEulerSystem] using
    (eulerStep_l2_lipschitz_of_rhs_lipschitz
      (s := s) (f := pfOdeRhs (α := ℝ) (s := s) sch model) (t := t) (dt := dt) (L := L) h)

/-! ## Lipschitz and contraction transport through sampler adapters -/

/--
If the underlying DDIM update is `L`-Lipschitz, then the `DynamicalSystem` wrapper around that
update is also `L`-Lipschitz.

This is the compositional bridge we want at the API boundary: once a concrete DDIM step bound is
available, the same bound is immediately available through the common dynamics API used elsewhere in
TorchLean.
-/
theorem ddimStepSystem_lipschitz_of_step_lipschitz
    (sched : VPSchedule SpecScalar T) (model : EpsModel SpecScalar s) (k : Fin T)
    (norm : ∀ {s : Shape}, SpecTensor s → SpecScalar) (L : SpecScalar)
    (h : NN.MLTheory.Robustness.Spec.isLipschitzContinuous
      (fun x : SpecTensor s => ddimStep (α := SpecScalar) (T := T) (s := s) sched model k x)
      norm norm L) :
    NN.MLTheory.Robustness.Spec.isLipschitzContinuous
      (ddimStepSystem (T := T) (s := s) sched model k).step norm norm L := by
  simpa [ddimStepSystem]

/--
If the underlying probability-flow Euler update is `L`-Lipschitz, then its `DynamicalSystem`
adapter is `L`-Lipschitz.

For quantitative PF-ODE certification, this is the bridge from an ODE-step bound (for example via
IBP/CROWN on the vector field) to the reusable `trajectory` and `iterate` definitions in
`NN.Spec.Dynamics.System`.
-/
theorem pfOdeEulerSystem_lipschitz_of_step_lipschitz
    (sch : VPLinearSchedule SpecScalar) (model : EpsModel SpecScalar s) (t dt : SpecScalar)
    (norm : ∀ {s : Shape}, SpecTensor s → SpecScalar) (L : SpecScalar)
    (h : NN.MLTheory.Robustness.Spec.isLipschitzContinuous
      (fun x : SpecTensor s =>
        eulerStep (α := SpecScalar) (s := s)
          (pfOdeRhs (α := SpecScalar) (s := s) sch model) x t dt)
      norm norm L) :
    NN.MLTheory.Robustness.Spec.isLipschitzContinuous
      (pfOdeEulerSystem (s := s) sch model t dt).step norm norm L := by
  simpa [pfOdeEulerSystem]

/--
A contractive DDIM update remains contractive after packaging it as a `DynamicalSystem`.

This is the formal hook for fixed-point and stability arguments about deterministic diffusion
samplers: once a concrete DDIM step bound is proved, the dynamics-level contraction predicate follows
without re-opening the sampler definition.
-/
theorem ddimStepSystem_contracts_of_step_contracts
    (sched : VPSchedule SpecScalar T) (model : EpsModel SpecScalar s) (k : Fin T)
    (norm : ∀ {s : Shape}, SpecTensor s → SpecScalar) (factor : SpecScalar)
    (h : NN.MLTheory.Robustness.Spec.isContractive (α := SpecScalar)
      (fun x : SpecTensor s => ddimStep (α := SpecScalar) (T := T) (s := s) sched model k x)
      norm factor) :
    NN.Spec.Dynamics.isContractive (ddimStepSystem (T := T) (s := s) sched model k)
      norm factor := by
  simpa [NN.Spec.Dynamics.isContractive, ddimStepSystem] using h

/--
A contractive PF-ODE Euler update remains contractive after packaging it as a `DynamicalSystem`.

This keeps the verification-facing statement compositional: prove the Euler map contracts under the
chosen norm, then reuse the dynamics API for iterated sampler trajectories.
-/
theorem pfOdeEulerSystem_contracts_of_step_contracts
    (sch : VPLinearSchedule SpecScalar) (model : EpsModel SpecScalar s) (t dt : SpecScalar)
    (norm : ∀ {s : Shape}, SpecTensor s → SpecScalar) (factor : SpecScalar)
    (h : NN.MLTheory.Robustness.Spec.isContractive (α := SpecScalar)
      (fun x : SpecTensor s =>
        eulerStep (α := SpecScalar) (s := s)
          (pfOdeRhs (α := SpecScalar) (s := s) sch model) x t dt)
      norm factor) :
    NN.Spec.Dynamics.isContractive (pfOdeEulerSystem (s := s) sch model t dt)
      norm factor := by
  simpa [NN.Spec.Dynamics.isContractive, pfOdeEulerSystem] using h

end NN.MLTheory.Generative.Diffusion
