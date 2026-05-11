/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Analysis.Normalization

/-!
# BugZoo: normalization state and BatchNorm contracts

BatchNorm is a small formula with a surprisingly large bug surface. Cross-backend testing work
found real library bugs around normalization formulas and backend conventions, including epsilon
placement in BatchNorm. Model-generation testing also found BatchNormalization failures involving
wrong moving statistics and NaN-producing outputs.

References:
- Pham et al., "CRADLE: Cross-Backend Validation to Detect and Localize Bugs in Deep Learning
  Libraries", ICSE 2019.
- Wang et al., "Deep Learning Library Testing via Effective Model Generation", ISSTA 2020.
- Ioffe and Szegedy, "Batch Normalization: Accelerating Deep Network Training by Reducing Internal
  Covariate Shift", ICML 2015.

TorchLean addresses this class in two layers:
- the spec formula is explicit, so epsilon placement is not hidden in backend code;
- inference-time running statistics are explicit inputs, so train/eval state boundaries become part
  of the checked object rather than ambient mutable framework state.
-/

@[expose] public section

namespace NN.Examples.BugZoo.NormalizationState

noncomputable section

/--
The buggy BatchNorm pattern reported by cross-backend testing is easy to state:
putting epsilon outside the square root changes the formula from

`(x - μ) / sqrt(σ² + ε)`

to

`(x - μ) / (sqrt(σ²) + ε)`.

We keep this definition as the "bad PyTorch-like code" analogue for documentation and regression
tests. TorchLean's actual `normalizeCore` does not use this expression.
-/
def wrongEpsilonOutsideSqrt (x mean variance gamma beta epsilon : ℝ) : ℝ :=
  ((x - mean) / (Real.sqrt variance + epsilon)) * gamma + beta

/--
The intended scalar BatchNorm expression. This mirrors the public TorchLean normalization spec:
epsilon is added to the variance before the square root.
-/
def correctEpsilonInsideSqrt (x mean variance gamma beta epsilon : ℝ) : ℝ :=
  ((x - mean) / Real.sqrt (variance + epsilon)) * gamma + beta

/--
Spec-level BatchNorm uses epsilon inside the variance term.

There is one extra implementation detail worth making explicit: `sqrtSpec` is total, so it computes
`sqrt (max (variance + epsilon) 0)`. On the usual BatchNorm path, variance is nonnegative and
epsilon is positive, so this is the same mathematical formula as `sqrt (variance + epsilon)`.
-/
theorem normalizeCore_scalar_uses_variance_plus_epsilon
    (x mean variance gamma beta epsilon : ℝ) :
    Spec.normalizeCore
        (s := .scalar)
        (s_mean := .scalar)
        (s_var := .scalar)
        (s_gamma := .scalar)
        (s_beta := .scalar)
        (epsilon := epsilon)
        (x := Spec.Tensor.scalar x)
        (mean := Spec.Tensor.scalar mean)
        (variance := Spec.Tensor.scalar variance)
        (gamma := Spec.Tensor.scalar gamma)
        (beta := Spec.Tensor.scalar beta)
        (cb_mean := Spec.Shape.CanBroadcastTo.scalar_to_any .scalar)
        (cb_var := Spec.Shape.CanBroadcastTo.scalar_to_any .scalar)
        (cb_gamma := Spec.Shape.CanBroadcastTo.scalar_to_any .scalar)
        (cb_beta := Spec.Shape.CanBroadcastTo.scalar_to_any .scalar)
      =
    Spec.Tensor.scalar
      (((x - mean) / MathFunctions.sqrt (Max.max (variance + epsilon) 0)) * gamma + beta) := by
  simp [Spec.normalizeCore, Spec.Tensor.broadcastTo, Spec.Tensor.addSpec, Spec.Tensor.subSpec,
    Spec.Tensor.mulSpec, Spec.Tensor.divSpec, Spec.Tensor.sqrtSpec, Spec.Tensor.mapSpec,
    Spec.Tensor.map2Spec, Spec.fill, Spec.replicate]

/--
Running statistics are part of the BatchNorm inference contract.

This is the boundary that catches a common state bug: using stale or unintended moving statistics
cannot be invisible inside TorchLean, because the exact `runningMean` and `runningVar` tensors are
arguments to the spec.
-/
structure RunningStats (channels : Nat) where
  /-- Inference-time running mean, usually learned/updated during training. -/
  mean : Spec.Tensor ℝ (.dim channels .scalar)
  /-- Inference-time running variance, clamped by the spec before normalization. -/
  variance : Spec.Tensor ℝ (.dim channels .scalar)

/-- Evaluation-time BatchNorm with state packaged as an explicit value. -/
def batchNormEvalWithStats {channels : Nat} {sSpatial : Spec.Shape}
    (x : Spec.Tensor ℝ (.dim channels sSpatial))
    (stats : RunningStats channels)
    (gamma : Spec.Tensor ℝ (.dim channels .scalar))
    (beta : Spec.Tensor ℝ (.dim channels .scalar))
    (epsilon : ℝ := Numbers.epsilon) :
    Spec.Tensor ℝ (.dim channels sSpatial) :=
  Spec.batchNormInference
    (x := x)
    (runningMean := stats.mean)
    (runningVar := stats.variance)
    (gamma := gamma)
    (beta := beta)
    (epsilon := epsilon)

/-- The packaged-state wrapper is exactly the public inference-time BatchNorm spec. -/
theorem batchNormEvalWithStats_unfolds {channels : Nat} {sSpatial : Spec.Shape}
    (x : Spec.Tensor ℝ (.dim channels sSpatial))
    (stats : RunningStats channels)
    (gamma : Spec.Tensor ℝ (.dim channels .scalar))
    (beta : Spec.Tensor ℝ (.dim channels .scalar))
    (epsilon : ℝ := Numbers.epsilon) :
    batchNormEvalWithStats x stats gamma beta epsilon =
      Spec.batchNormInference
        (x := x)
        (runningMean := stats.mean)
        (runningVar := stats.variance)
        (gamma := gamma)
        (beta := beta)
        (epsilon := epsilon) := by
  rfl

/--
Inference-time BatchNorm is affine once running statistics are fixed.

This re-exports the analysis theorem used by optimizers and verifiers: the checked stateful
inference spec can be folded into a pointwise affine map, but only after the running statistics are
made explicit.
-/
theorem batchNormEvalWithStats_is_affine
    {channels : Nat} {sSpatial : Spec.Shape}
    (x : Spec.Tensor ℝ (.dim channels sSpatial))
    (stats : RunningStats channels)
    (gamma : Spec.Tensor ℝ (.dim channels .scalar))
    (beta : Spec.Tensor ℝ (.dim channels .scalar))
    (epsilon : ℝ := Numbers.epsilon) :
    ∃ scale bias : Spec.Tensor ℝ (.dim channels sSpatial),
      batchNormEvalWithStats x stats gamma beta epsilon =
        Spec.Tensor.addSpec (Spec.Tensor.mulSpec x scale) bias := by
  let s : Spec.Shape := .dim channels sSpatial
  let cb : Spec.Shape.CanBroadcastTo (.dim channels .scalar) s := by
    apply Spec.Shape.CanBroadcastTo.dim_eq
    exact Spec.Shape.CanBroadcastTo.scalar_to_any sSpatial
  let runningVar := Spec.Tensor.maxSpec stats.variance (Spec.fill 0 (.dim channels .scalar))
  let mean_b := Spec.Tensor.broadcastTo cb stats.mean
  let var_b := Spec.Tensor.broadcastTo cb runningVar
  let gamma_b := Spec.Tensor.broadcastTo cb gamma
  let beta_b := Spec.Tensor.broadcastTo cb beta
  let std := Spec.Tensor.sqrtSpec (Spec.Tensor.addSpec var_b (Spec.fill epsilon s))
  refine
    ⟨Spec.Tensor.divSpec gamma_b std,
      Spec.Tensor.subSpec beta_b (Spec.Tensor.mulSpec mean_b (Spec.Tensor.divSpec gamma_b std)),
      ?_⟩
  simpa [batchNormEvalWithStats, s, cb, runningVar, mean_b, var_b, gamma_b, beta_b, std]
    using
      Proofs.Normalization.batchNorm_inference_eq_mul_add
        (x := x)
        (runningMean := stats.mean)
        (runningVar := stats.variance)
        (gamma := gamma)
        (beta := beta)
        (epsilon := epsilon)

end

end NN.Examples.BugZoo.NormalizationState
