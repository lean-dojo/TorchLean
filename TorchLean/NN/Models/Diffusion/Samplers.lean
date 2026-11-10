/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Lipschitz.TensorNorm
public import TorchLean.NN.Models.Diffusion.ImageDDIM
public import TorchLean.NN.Models.Diffusion.PFODE
public import TorchLean.NN.Models.Diffusion.ReverseDDPM

/-!
# Diffusion sampler theorems

The theorems here sit between TorchLean's executable diffusion specs and its stability and
verification APIs. They cover the schedule's boundary value, the zero-step behavior of the DDPM,
DDIM and probability-flow samplers, the L2 stability of one explicit Euler probability-flow step,
and Lipschitz and contraction bounds carried through the `Dynamics.System` adapters.

We separate sampler mechanics from measure-level claims. ELBO optimality, reverse-SDE/PF-ODE
marginal equivalence, and consistency distillation need more probability and regularity
assumptions. The theorems here prove the sampler-side facts that downstream proofs and verifiers
can reuse:

- schedule boundary values,
- zero-step sampler behavior, and
- L2 stability of explicit Euler probability-flow updates.

The DDIM facts concern `ImageDDIM.sample` and its step, which the runnable diffusion example runs.
`ImageDDIM.stepSystem` packages one such step as a real-valued `Dynamics.System`. We define it
here so that the runtime, which imports the image specification, does not import
`TorchLean.Tensor.Context.Real`. The probability-flow adapter and its `rfl` fact sit next to
`eulerStep` in `TorchLean.NN.Models.Diffusion.PFODE`.

All of these are proved by hand from the definitions, the zero-step facts by `rfl` or `simp`.
With `steps = 1000` each Euler step has `|dt| = 0.001`, so `eulerStep_l2_lipschitz` turns an
`L`-Lipschitz drift into a `(1 + 0.001 L)`-Lipschitz step.

References:
- Jonathan Ho, Ajay Jain, and Pieter Abbeel (2020), "Denoising Diffusion Probabilistic Models",
  NeurIPS 2020, arXiv:2006.11239.
- Yang Song et al. (2021), "Score-Based Generative Modeling through Stochastic Differential
  Equations", ICLR 2021, arXiv:2011.13456.
- E. Hairer, S. P. Nørsett, and G. Wanner (1993), *Solving Ordinary Differential Equations I*,
  Springer Series in Computational Mathematics, doi:10.1007/978-3-540-78862-1, for the Euler
  stability estimate used below.
-/

@[expose] public section

namespace TorchLean.Diffusion

open _root_.TorchLean.Tensor

variable {α : Type} [Storage α] [Context α]
variable {T : Nat} {s : Shape}

/-- The VP schedule convention is $\bar{\alpha}_0 = 1$. -/
@[simp] theorem VPSchedule.alphaBar_zero (sched : VPSchedule α T) :
    sched.alphaBar ⟨0, Nat.succ_pos T⟩ = 1 := by
  rfl

/-- With no reverse steps, DDPM returns its initial terminal-noise state. -/
@[simp] theorem ddpmSample_zero_steps (sched : VPSchedule α 0) (model : EpsModel α s)
    (x_T : Tensor α s) (noise : Fin 0 → Tensor α s) :
    ddpmSample (α := α) (T := 0) (s := s) sched model x_T noise = x_T := by
  simp [ddpmSample]

/-- With an empty coefficient table, `ImageDDIM.sample` returns `pure initial` from every
start. -/
@[simp] theorem ImageDDIM.sample_zero_steps {m : Type → Type} [Monad m] (denominatorFloor : α)
    (coefficients : Tensor α [0]) (predict : Fin 0 → Tensor α s → m (Tensor α s))
    (initial : Tensor α s) (postprocess : Tensor α s → Tensor α s) (start : Option (Fin 0)) :
    ImageDDIM.sample denominatorFloor coefficients predict initial postprocess start =
      pure initial := by
  cases start with
  | none => simp [ImageDDIM.sample]
  | some index => exact index.elim0

/-- With zero Euler steps, the probability-flow sampler returns the initial state. -/
@[simp] theorem pfODESampleEuler_zero_steps (sch : VPLinearSchedule α) (model : EpsModel α s)
    (x1 : Tensor α s) :
    pfODESampleEuler (α := α) (s := s) sch model 0 x1 = x1 := by
  rfl

/-! ## Quantitative Euler stability for probability-flow samplers -/

/--
The difference of two explicit Euler updates:

$$
(x+\Delta t\,f_x)-(y+\Delta t\,f_y)=(x-y)+\Delta t\,(f_x-f_y)
$$

The stability and Lipschitz proofs for ODE samplers below start from this identity. We keep it
private because callers want the norm and Lipschitz bounds.
-/
private theorem sub_add_scaled_eq {s : Shape} (x y fx fy : Tensor ℝ s) (dt : ℝ) :
    (x + Tensor.scale fx dt) - (y + Tensor.scale fy dt) =
      (x - y) + Tensor.scale (fx - fy) dt := by
  apply Tensor.Rep.ext
  intro coordinate
  simp [sub_def, add_def, Tensor.scale]
  ring_nf

/--
For an ODE $x'=f(x,t)$ with Euler update $E(x)=x+\Delta t\,f(x,t)$, the L2 distance of two updates
is at most the distance of their states plus $|\Delta t|$ times the distance of their right-hand
sides:

$$
\lVert E(x)-E(y)\rVert_2
\leq \lVert x-y\rVert_2 +
  |\Delta t|\,\lVert f(x,t)-f(y,t)\rVert_2.
$$

The proof is the Euler-difference identity above followed by the triangle inequality `norm_add_le`
and `norm_smul` for the Euclidean norm `‖·‖` on real tensors. It is the bridge from the executable
PF-ODE sampler to quantitative stability/verification arguments.

Reference: this is the elementary one-step stability estimate underlying explicit Euler analyses, as
in Hairer, Nørsett, and Wanner, *Solving Ordinary Differential Equations I*.
-/
theorem eulerStep_l2_distance_le
    (f : Tensor ℝ s → ℝ → Tensor ℝ s) (t dt : ℝ) (x y : Tensor ℝ s) :
    Robustness.distance (α := ℝ)
      (‖·‖)
      (eulerStep (α := ℝ) (s := s) f x t dt)
      (eulerStep (α := ℝ) (s := s) f y t dt)
      ≤ Robustness.distance (α := ℝ)
          (‖·‖) x y +
        |dt| * Robustness.distance (α := ℝ)
          (‖·‖) (f x t) (f y t) := by
  simp only [Robustness.distance,
    eulerStep]
  rw [sub_add_scaled_eq]
  calc
    ‖(x - y) + Tensor.scale (f x t - f y t) dt‖
        ≤ ‖x - y‖ + ‖Tensor.scale (f x t - f y t) dt‖ :=
            norm_add_le (x - y) (Tensor.scale (f x t - f y t) dt)
    _ = ‖x - y‖ + |dt| * ‖f x t - f y t‖ := by
          rw [Tensor.scale_eq_smul, norm_smul, Real.norm_eq_abs]

/--
If the ODE right-hand side is $L$-Lipschitz in L2 at a fixed time, then one Euler step is
$(1+|\Delta t|L)$-Lipschitz in L2.

A Lipschitz bound for the neural vector field, proved or imported, thus becomes a certified bound
on the update the sampler runs.
-/
theorem eulerStep_l2_lipschitz
    (f : Tensor ℝ s → ℝ → Tensor ℝ s) (t dt L : ℝ)
    (h : Robustness.IsLipschitzContinuous (α := ℝ)
      (fun x : Tensor ℝ s => f x t)
      (‖·‖)
      (‖·‖) L) :
    Robustness.IsLipschitzContinuous (α := ℝ)
      (fun x : Tensor ℝ s => eulerStep (α := ℝ) (s := s) f x t dt)
      (‖·‖)
      (‖·‖) (1 + |dt| * L) := by
  intro x y
  have hstep := eulerStep_l2_distance_le (s := s) f t dt x y
  have hf := h x y
  have hscaled :
      |dt| * Robustness.distance (α := ℝ)
          (‖·‖) (f x t) (f y t) ≤
        |dt| * (L * Robustness.distance (α := ℝ)
          (‖·‖) x y) := by
    exact mul_le_mul_of_nonneg_left hf (abs_nonneg dt)
  calc
    Robustness.distance (α := ℝ)
        (‖·‖)
        (eulerStep (α := ℝ) (s := s) f x t dt)
        (eulerStep (α := ℝ) (s := s) f y t dt)
        ≤ Robustness.distance (α := ℝ)
            (‖·‖) x y +
          |dt| * Robustness.distance (α := ℝ)
            (‖·‖) (f x t) (f y t) := hstep
    _ ≤ Robustness.distance (α := ℝ)
          (‖·‖) x y +
        |dt| * (L * Robustness.distance (α := ℝ)
          (‖·‖) x y) := by
          linarith
    _ = (1 + |dt| * L) * Robustness.distance (α := ℝ)
          (‖·‖) x y := by
          ring

/--
If the PF-ODE vector field `pfODERhs sch model` is $L$-Lipschitz in L2 at time $t$, then the step
of `pfODEEulerSystem sch model t dt` is $(1+|\Delta t|L)$-Lipschitz in L2.

A downstream PF-ODE certificate should target this theorem first.
-/
theorem pfODEEulerSystem_l2_lipschitz
    (sch : VPLinearSchedule ℝ) (model : EpsModel ℝ s) (t dt L : ℝ)
    (h : Robustness.IsLipschitzContinuous (α := ℝ)
      (fun x : Tensor ℝ s => pfODERhs (α := ℝ) (s := s) sch model x t)
      (‖·‖)
      (‖·‖) L) :
    Robustness.IsLipschitzContinuous (α := ℝ)
      (pfODEEulerSystem (s := s) sch model t dt).step
      (‖·‖)
      (‖·‖) (1 + |dt| * L) := by
  simpa [pfODEEulerSystem] using
    (eulerStep_l2_lipschitz
      (s := s) (f := pfODERhs (α := ℝ) (s := s) sch model) (t := t) (dt := dt) (L := L) h)

/-! ## Lipschitz and contraction transport through sampler adapters -/

/-- The image DDIM update at noisy state `index` with a pure noise predictor, as a
`Dynamics.System`: the step `ImageDDIM.sample` takes at `index`, run in `Id`. -/
noncomputable def ImageDDIM.stepSystem (denominatorFloor : ℝ)
    (coefficients : Tensor ℝ [T]) (predict : Fin T → Tensor ℝ s → Tensor ℝ s)
    (index : Fin T) (postprocess : Tensor ℝ s → Tensor ℝ s) :
    Dynamics.System s where
  step := fun x => ImageDDIM.step (m := Id) denominatorFloor coefficients predict index x
    postprocess

/-- `ImageDDIM.stepSystem` steps by one `ImageDDIM.step` in `Id`. -/
@[simp] theorem ImageDDIM.stepSystem_step (denominatorFloor : ℝ)
    (coefficients : Tensor ℝ [T]) (predict : Fin T → Tensor ℝ s → Tensor ℝ s)
    (index : Fin T) (postprocess : Tensor ℝ s → Tensor ℝ s) (x : Tensor ℝ s) :
    (ImageDDIM.stepSystem denominatorFloor coefficients predict index postprocess).step x =
      ImageDDIM.step (m := Id) denominatorFloor coefficients predict index x postprocess := by
  rfl

/-- If the image DDIM update at `index` is `L`-Lipschitz from `norm` to `norm`, then so is the
step of `ImageDDIM.stepSystem` at `index`. -/
theorem ImageDDIM.stepSystem_lipschitz_of_step_lipschitz (denominatorFloor : ℝ)
    (coefficients : Tensor ℝ [T]) (predict : Fin T → Tensor ℝ s → Tensor ℝ s)
    (index : Fin T) (postprocess : Tensor ℝ s → Tensor ℝ s)
    (norm : ∀ {s : Shape}, Tensor ℝ s → ℝ) (L : ℝ)
    (h : Robustness.IsLipschitzContinuous
      (fun x : Tensor ℝ s =>
        ImageDDIM.step (m := Id) denominatorFloor coefficients predict index x postprocess)
      norm norm L) :
    Robustness.IsLipschitzContinuous
      (ImageDDIM.stepSystem denominatorFloor coefficients predict index postprocess).step
      norm norm L := by
  simpa [ImageDDIM.stepSystem]

/--
If the probability-flow Euler update is `L`-Lipschitz for `norm`, so is the step of its
`Dynamics.System` adapter `pfODEEulerSystem sch model t dt`.

For quantitative PF-ODE certification, this is the bridge from an ODE-step bound (for example via
IBP/CROWN on the vector field) to the iterates `step^[n]` of the `Dynamics` system.
-/
theorem pfODEEulerSystem_lipschitz
    (sch : VPLinearSchedule ℝ) (model : EpsModel ℝ s) (t dt : ℝ)
    (norm : ∀ {s : Shape}, Tensor ℝ s → ℝ) (L : ℝ)
    (h : Robustness.IsLipschitzContinuous
      (fun x : Tensor ℝ s =>
        eulerStep (α := ℝ) (s := s)
          (pfODERhs (α := ℝ) (s := s) sch model) x t dt)
      norm norm L) :
    Robustness.IsLipschitzContinuous
      (pfODEEulerSystem (s := s) sch model t dt).step norm norm L := by
  simpa [pfODEEulerSystem]

/-- If the image DDIM update at `index` contracts by `factor` under `norm`, then
`ImageDDIM.stepSystem` at `index` is contractive with that factor. -/
theorem ImageDDIM.stepSystem_contracts_of_step_contracts (denominatorFloor : ℝ)
    (coefficients : Tensor ℝ [T]) (predict : Fin T → Tensor ℝ s → Tensor ℝ s)
    (index : Fin T) (postprocess : Tensor ℝ s → Tensor ℝ s)
    (norm : ∀ {s : Shape}, Tensor ℝ s → ℝ) (factor : ℝ)
    (h : Robustness.IsContractive (α := ℝ)
      (fun x : Tensor ℝ s =>
        ImageDDIM.step (m := Id) denominatorFloor coefficients predict index x postprocess)
      norm factor) :
    Robustness.IsContractive (α := ℝ)
      (ImageDDIM.stepSystem denominatorFloor coefficients predict index postprocess).step
      norm factor := by
  simpa [ImageDDIM.stepSystem]

/--
If the PF-ODE Euler update contracts by `factor` for `norm`, so does the step of its
`Dynamics.System` adapter `pfODEEulerSystem sch model t dt`.

To bound iterated sampler trajectories, prove that the Euler map contracts under the chosen norm
and continue with the dynamics API.
-/
theorem pfODEEulerSystem_contracts
    (sch : VPLinearSchedule ℝ) (model : EpsModel ℝ s) (t dt : ℝ)
    (norm : ∀ {s : Shape}, Tensor ℝ s → ℝ) (factor : ℝ)
    (h : Robustness.IsContractive (α := ℝ)
      (fun x : Tensor ℝ s =>
        eulerStep (α := ℝ) (s := s)
          (pfODERhs (α := ℝ) (s := s) sch model) x t dt)
      norm factor) :
    Robustness.IsContractive (α := ℝ) (pfODEEulerSystem (s := s) sch model t dt).step
      norm factor := by
  simpa [pfODEEulerSystem]

end TorchLean.Diffusion
