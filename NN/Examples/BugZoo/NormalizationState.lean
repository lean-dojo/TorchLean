/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Analysis.Normalization
public import NN.Tensor

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

open TorchLean

namespace NN.Examples.BugZoo.NormalizationState

noncomputable section

/--
The buggy BatchNorm pattern reported by cross-backend testing is easy to state:
putting epsilon outside the square root changes the formula from

$$
\frac{x-\mu}{\sqrt{\sigma^2+\varepsilon}}
$$

to

$$
\frac{x-\mu}{\sqrt{\sigma^2}+\varepsilon}.
$$

The separating example below uses zero variance and epsilon four.
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
At variance zero and epsilon four, the misplaced epsilon divides by four instead of two.
-/
theorem wrongEpsilonOutsideSqrt_ne_correctEpsilonInsideSqrt :
    wrongEpsilonOutsideSqrt 1 0 0 1 0 4 ≠ correctEpsilonInsideSqrt 1 0 0 1 0 4 := by
  have hsqrt : Real.sqrt 4 = 2 := by
    rw [show (4 : ℝ) = 2 ^ 2 by norm_num, Real.sqrt_sq (by norm_num)]
  simp [wrongEpsilonOutsideSqrt, correctEpsilonInsideSqrt, hsqrt]

/--
Spec-level BatchNorm uses epsilon inside the variance term.

There is one extra implementation detail worth making explicit: `sqrtSpec` is total, so it computes
$\sqrt{\max(\mathrm{variance}+\varepsilon,0)}$. On the usual BatchNorm path, variance is nonnegative
and epsilon is positive, so this is the same mathematical formula as
$\sqrt{\mathrm{variance}+\varepsilon}$.
-/
theorem normalizeCore_scalar_uses_variance_plus_epsilon
    (x mean variance gamma beta epsilon : ℝ) :
    Spec.normalizeCore
        (epsilon := epsilon)
        (x := Tensor.full [] x)
        (mean := Tensor.full [] mean)
        (variance := Tensor.full [] variance)
        (gamma := Tensor.full [] gamma)
        (beta := Tensor.full [] beta)
        (cbMean := Spec.Shape.CanBroadcastTo.refl [])
        (cbVar := Spec.Shape.CanBroadcastTo.refl [])
        (cbGamma := Spec.Shape.CanBroadcastTo.refl [])
        (cbBeta := Spec.Shape.CanBroadcastTo.refl [])
      =
    Tensor.full []
      (((x - mean) / MathFunctions.sqrt (Max.max (variance + epsilon) 0)) * gamma + beta) := by
  apply Tensor.ext_scalar
  simp [Spec.normalizeCore, Tensor.addSpec,
    Tensor.subSpec, Tensor.mulSpec, Tensor.divSpec,
    Tensor.sqrtSpec]

/--
Running statistics are part of the BatchNorm inference contract.

The exact running mean and variance are arguments to the spec, so a reviewer can identify which
state the result uses. Their shape does not establish that they are current or came from the
intended training run; that provenance remains a separate obligation.
-/
structure RunningStats (channels : Nat) where
  /-- Inference-time running mean, usually learned/updated during training. -/
  mean : Tensor ℝ [channels]
  /-- Inference-time running variance, clamped by the spec before normalization. -/
  variance : Tensor ℝ [channels]

/-- Evaluation-time BatchNorm with state packaged as an explicit value. -/
def batchNormEvalWithStats {channels : Nat} {sSpatial : Spec.Shape}
    (x : Tensor ℝ (sSpatial.prependDim channels))
    (stats : RunningStats channels)
    (gamma : Tensor ℝ [channels])
    (beta : Tensor ℝ [channels])
    (epsilon : ℝ := TorchLean.normalizationEpsilon) :
    Tensor ℝ (sSpatial.prependDim channels) :=
  Spec.batchNormInference
    (x := x)
    (runningMean := stats.mean)
    (runningVar := stats.variance)
    (gamma := gamma)
    (beta := beta)
    (epsilon := epsilon)

/-- The packaged-state wrapper is exactly the public inference-time BatchNorm spec. -/
theorem batchNormEvalWithStats_unfolds {channels : Nat} {sSpatial : Spec.Shape}
    (x : Tensor ℝ (sSpatial.prependDim channels))
    (stats : RunningStats channels)
    (gamma : Tensor ℝ [channels])
    (beta : Tensor ℝ [channels])
    (epsilon : ℝ := TorchLean.normalizationEpsilon) :
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
Fixed BatchNorm running statistics determine one scale and bias that work for every input.

The witnesses depend only on the running statistics, affine parameters, and epsilon. This is the
uniform affine representation needed when folding inference-time normalization into another layer.
-/
theorem batchNormEvalWithStats_affine
    {channels : Nat} {sSpatial : Spec.Shape}
    (stats : RunningStats channels)
    (gamma : Tensor ℝ [channels])
    (beta : Tensor ℝ [channels])
    (epsilon : ℝ := TorchLean.normalizationEpsilon) :
    ∃ scale bias : Tensor ℝ (sSpatial.prependDim channels),
      ∀ x : Tensor ℝ (sSpatial.prependDim channels),
        batchNormEvalWithStats x stats gamma beta epsilon =
          Tensor.addSpec (Tensor.mulSpec x scale) bias := by
  let s : Spec.Shape := sSpatial.prependDim channels
  let runningVar :=
    Tensor.maxSpec stats.variance (Tensor.full [channels] 0)
  let mean_b := Spec.broadcastChannel sSpatial stats.mean
  let var_b := Spec.broadcastChannel sSpatial runningVar
  let gamma_b := Spec.broadcastChannel sSpatial gamma
  let beta_b := Spec.broadcastChannel sSpatial beta
  let std :=
    Tensor.sqrtSpec
      (Tensor.addSpec var_b (Tensor.full s epsilon))
  refine
    ⟨Tensor.divSpec gamma_b std,
      Tensor.subSpec beta_b
        (Tensor.mulSpec mean_b (Tensor.divSpec gamma_b std)),
      ?_⟩
  intro x
  simpa [batchNormEvalWithStats, s, runningVar, mean_b, var_b, gamma_b, beta_b, std]
    using
      Proofs.Normalization.batchNorm_inference_eq_mul_add
        (x := x)
        (runningMean := stats.mean)
        (runningVar := stats.variance)
        (gamma := gamma)
        (beta := beta)
        (epsilon := epsilon)


/-- Specialize the shared affine representation to one input. -/
theorem batchNormEvalWithStats_is_affine
    {channels : Nat} {sSpatial : Spec.Shape}
    (x : Tensor ℝ (sSpatial.prependDim channels))
    (stats : RunningStats channels)
    (gamma : Tensor ℝ [channels])
    (beta : Tensor ℝ [channels])
    (epsilon : ℝ := TorchLean.normalizationEpsilon) :
    ∃ scale bias : Tensor ℝ (sSpatial.prependDim channels),
      batchNormEvalWithStats x stats gamma beta epsilon =
        Tensor.addSpec (Tensor.mulSpec x scale) bias := by
  obtain ⟨scale, bias, h⟩ := batchNormEvalWithStats_affine (sSpatial := sSpatial)
    stats gamma beta epsilon
  exact ⟨scale, bias, h x⟩

end

end NN.Examples.BugZoo.NormalizationState
