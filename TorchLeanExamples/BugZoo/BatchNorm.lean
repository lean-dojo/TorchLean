/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Loss.Cosine
public import TorchLean.Tensor.Elementwise.Real
public meta import TorchLean.Tensor.Einops.Elab.Einsum
public meta import TorchLean.Tensor.Einops.Elab.Pack
public meta import TorchLean.Tensor.Literal
public meta import TorchLean.Tensor.Einops.Elab.Transform
public meta import TorchLean.Tensor.Tactic.Einops.Report
public import TorchLean.Tensor.Literal
public import TorchLean.Tensor.Tactic.Init

/-!
# BugZoo: normalization state and BatchNorm contracts

BatchNorm is a small formula with a surprisingly large bug surface. Cross-backend testing found
real library bugs around normalization formulas and backend conventions, among them epsilon
placement in BatchNorm. Model-generation testing found BatchNormalization failures with wrong
moving statistics and NaN-producing outputs.

References:
- Pham et al., "CRADLE: Cross-Backend Validation to Detect and Localize Bugs in Deep Learning
  Libraries", ICSE 2019.
- Wang et al., "Deep Learning Library Testing via Effective Model Generation", ISSTA 2020.
- Ioffe and Szegedy, "Batch Normalization: Accelerating Deep Network Training by Reducing Internal
  Covariate Shift", ICML 2015.

TorchLean answers this class of bug in two places. The spec writes the formula out, so epsilon
placement can be read in Lean. Inference-time running statistics are arguments of the spec, so the
train/eval state boundary is part of the checked object and never ambient mutable framework state.
-/

@[expose] public section

open FloatLib.Numerics (MathFunctions)

namespace TorchLean.Examples.BugZoo

noncomputable section

/--
Scalar BatchNorm with epsilon outside the square root, the bug that cross-backend testing
reported. Moving epsilon changes the formula from

$$
\frac{x-\mu}{\sqrt{\sigma^2+\varepsilon}}
$$

to

$$
\frac{x-\mu}{\sqrt{\sigma^2}+\varepsilon}.
$$

`Examples.BugZoo.BatchNorm.outsideSqrt_ne_insideSqrt` separates the two at zero variance and epsilon
four.
-/
def BatchNorm.outsideSqrt (x mean variance gamma beta epsilon : ℝ) : ℝ :=
  ((x - mean) / (Real.sqrt variance + epsilon)) * gamma + beta

/--
The intended scalar BatchNorm expression, which adds epsilon to the variance before the square
root, as the TorchLean normalization spec does.
-/
def BatchNorm.insideSqrt (x mean variance gamma beta epsilon : ℝ) : ℝ :=
  ((x - mean) / Real.sqrt (variance + epsilon)) * gamma + beta

/--
With `x = 1`, zero mean, variance and bias, unit scale and epsilon four, the misplaced epsilon gives
`1/4` where the intended formula gives `1/2`.
-/
theorem BatchNorm.outsideSqrt_ne_insideSqrt :
    outsideSqrt 1 0 0 1 0 4 ≠ insideSqrt 1 0 0 1 0 4 := by
  have hsqrt : Real.sqrt 4 = 2 := by
    rw [show (4 : ℝ) = 2 ^ 2 by norm_num, Real.sqrt_sq (by norm_num)]
  simp [outsideSqrt, insideSqrt, hsqrt]

/-- Batch normalization along `dim` of the tensor with every entry `x`, with running mean `mean`,
running variance `variance`, scale `gamma` and bias `beta` in every channel, is a constant tensor.
Its entry is `(x - mean) / sqrt (max (max variance 0 + epsilon) 0) * gamma + beta`. -/
theorem BatchNorm.normalize_scalar {s : Shape} (dim : Fin s.rank)
    (x mean variance gamma beta epsilon : ℝ) :
    Tensor.batchNorm (Tensor.full s x) dim (Tensor.full _ gamma) (Tensor.full _ beta)
        (.running (Tensor.full _ mean) (Tensor.full _ variance)) epsilon =
      Tensor.full s
        ((x - mean) / MathFunctions.sqrt (max (max variance 0 + epsilon) 0) * gamma + beta) := by
  apply Tensor.Rep.ext
  intro c
  rw [Tensor.batchNorm_running_apply]
  simp [Context.maximum_real]

/--
Running statistics, part of the BatchNorm inference contract.

The running mean and variance are arguments to the spec, so a reader can see which state a result
uses. Their type says nothing about whether they are current or came from the intended training
run, and that provenance is a separate obligation.
-/
structure RunningStats (channels : Nat) where
  /-- Inference-time running mean, usually accumulated during training. -/
  mean : Tensor ℝ [channels]
  /-- Inference-time running variance, clamped at zero by the spec before normalization. -/
  variance : Tensor ℝ [channels]

/--
With fixed running statistics, inference-time BatchNorm is one affine map: there are `scale` and
`bias` tensors such that `Tensor.batchNorm` in `.running` mode sends every input `x` to
`x * scale + bias`.

The witnesses depend only on the running statistics, the affine parameters and epsilon. Folding
inference-time normalization into a neighbouring layer needs this uniform form.
-/
theorem RunningStats.batchNorm_affine
    {channels : Nat} {sSpatial : Shape}
    (stats : RunningStats channels)
    (gamma : Tensor ℝ [channels])
    (beta : Tensor ℝ [channels])
    (epsilon : ℝ := normalizationEpsilon) :
    ∃ scale bias : Tensor ℝ (Shape.dim channels sSpatial),
      ∀ x : Tensor ℝ (Shape.dim channels sSpatial),
        Tensor.batchNorm x ⟨0, Nat.succ_pos _⟩ gamma beta (.running stats.mean stats.variance)
            epsilon =
          (x * scale) + bias := by
  let s : Shape := Shape.dim channels sSpatial
  let runningVar :=
    Tensor.maximum stats.variance (Tensor.full [channels] 0)
  let spread := fun t : Tensor ℝ [channels] =>
    Tensor.broadcast t s _ (Shape.broadcastDims_axis s ⟨0, Nat.succ_pos _⟩)
  let mean_b := spread stats.mean
  let var_b := spread runningVar
  let gamma_b := spread gamma
  let beta_b := spread beta
  let std :=
    Tensor.sqrt
      (var_b + Tensor.full s epsilon) .clampNegative
  refine
    ⟨gamma_b / std,
      beta_b -
        (mean_b * (gamma_b / std)),
      ?_⟩
  intro x
  simpa [s, spread, runningVar, mean_b, var_b, gamma_b, beta_b, std]
    using
      Tensor.batchNorm_running_eq_mul_add
        (x := x)
        (runningMean := stats.mean)
        (runningVar := stats.variance)
        (gamma := gamma)
        (beta := beta)
        (epsilon := epsilon)
end

end TorchLean.Examples.BugZoo
