/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Normalization
import NN.Proofs.Tensor.Basic
import Mathlib.Tactic.Ring

/-!
# Normalization analysis properties

This file records theorem-level properties of TorchLean's normalization specs. The executable and
spec definitions live in `NN.Spec.Layers.Normalization`; this file belongs under
`NN.Proofs.Analysis` because it proves algebraic facts about those definitions over `ℝ`.

Current focus:
- **BatchNorm inference is affine** in the input `x` when the running statistics are fixed.

That fact matters because inference-time BatchNorm can be folded into a preceding/following affine
layer for verification, bound propagation, and simplification. Training-mode BatchNorm has
batch-dependent statistics and is intentionally not claimed by this theorem.
-/

@[expose] public section

namespace Proofs

open Spec
open Tensor

noncomputable section

namespace Normalization

/-!
## BatchNorm inference is affine

At inference time, BatchNorm uses fixed running mean/variance. The resulting function is affine:

`y = x ⊙ scale + bias`.
-/

/--
Scalar algebra behind inference-time BatchNorm folding.

For fixed mean `μ`, scale `γ`, shift `β`, and standard deviation `std`, the expression

`((x - μ) / std) * γ + β`

is affine in `x`, with multiplicative coefficient `γ / std` and bias
`β - μ * (γ / std)`.
-/
private lemma batchNorm_inference_affine_scalar (x μ γ β std : ℝ) :
    ((x - μ) / std) * γ + β = x * (γ / std) + (β - μ * (γ / std)) := by
  -- Treat `γ / std` as an atom, so we can use `ring` on the remaining algebra.
  set t : ℝ := γ / std
  have hrewrite : ((x - μ) / std) * γ = (x - μ) * t := by
    calc
      ((x - μ) / std) * γ = ((x - μ) * γ) / std := by
        simp [div_mul_eq_mul_div]
      _ = (x - μ) * (γ / std) := by
        simp [mul_div_assoc]
      _ = (x - μ) * t := by
        simp [t]
  -- Now `ring` closes: `(x - μ) * t + β = x * t + (β - μ * t)`.
  simpa [hrewrite, t] using (by ring : (x - μ) * t + β = x * t + (β - μ * t))

/--
Shape-generic tensor version of `batchNorm_inference_affine_scalar`.

All operations here are pointwise tensor ops, so the proof is structural induction on the tensor
shape. This private theorem handles the algebra after BatchNorm parameters have already been
broadcast to the input shape.
-/
private theorem batchNorm_inference_affine_tensor {s : Shape}
    (x mean gamma beta std : Tensor ℝ s) :
    Tensor.addSpec (Tensor.mulSpec (Tensor.divSpec (Tensor.subSpec x mean) std) gamma) beta =
      Tensor.addSpec (Tensor.mulSpec x (Tensor.divSpec gamma std))
        (Tensor.subSpec beta (Tensor.mulSpec mean (Tensor.divSpec gamma std))) := by
  induction s with
  | scalar =>
      cases x
      cases mean
      cases gamma
      cases beta
      cases std
      simp [Tensor.addSpec, Tensor.subSpec, Tensor.mulSpec, Tensor.divSpec, Tensor.map2Spec,
        batchNorm_inference_affine_scalar]
  | dim n s ih =>
      cases x with
      | dim fx =>
        cases mean with
        | dim fmean =>
          cases gamma with
          | dim fgamma =>
            cases beta with
            | dim fbeta =>
              cases std with
              | dim fstd =>
                apply congrArg Tensor.dim
                funext i
                exact ih (x := fx i) (mean := fmean i) (gamma := fgamma i) (beta := fbeta i)
                  (std := fstd i)

/--
Inference-time BatchNorm is affine in the input `x`.

This is the public theorem users want for verification and graph simplification:

`batchNormInference x runningMean runningVar gamma beta epsilon`

is definitionally equal to a pointwise affine map

`x * (gamma / std) + (beta - mean * (gamma / std))`

after broadcasting channel parameters to the input shape and clamping the running variance exactly
as the spec does.
-/
theorem batchNorm_inference_eq_mul_add
    {channels : Nat} {sSpatial : Shape}
    (x : Tensor ℝ (.dim channels sSpatial))
    (runningMean : Tensor ℝ (.dim channels .scalar))
    (runningVar : Tensor ℝ (.dim channels .scalar))
    (gamma : Tensor ℝ (.dim channels .scalar))
    (beta : Tensor ℝ (.dim channels .scalar))
    (epsilon : ℝ := Numbers.epsilon) :
    Spec.batchNormInference (α := ℝ) (channels := channels) (sSpatial := sSpatial)
        x runningMean runningVar gamma beta epsilon
      =
    let s : Shape := .dim channels sSpatial
    let cb : Shape.CanBroadcastTo (.dim channels .scalar) s := by
      apply Shape.CanBroadcastTo.dim_eq
      exact Shape.CanBroadcastTo.scalar_to_any sSpatial
    let runningVar := Tensor.maxSpec runningVar (Tensor.fill 0 (.dim channels .scalar))
    let mean_b := Tensor.broadcastTo cb runningMean
    let var_b := Tensor.broadcastTo cb runningVar
    let gamma_b := Tensor.broadcastTo cb gamma
    let beta_b := Tensor.broadcastTo cb beta
    let std := Tensor.sqrtSpec (Tensor.addSpec var_b (Tensor.fill epsilon s))
    Tensor.addSpec (Tensor.mulSpec x (Tensor.divSpec gamma_b std))
      (Tensor.subSpec beta_b (Tensor.mulSpec mean_b (Tensor.divSpec gamma_b std))) := by
  -- Unfold to the shared `normalize_core` form, then apply the shape-generic affine identity.
  simp [Spec.batchNormInference, Spec.normalizeCore, batchNorm_inference_affine_tensor]

end Normalization

end

end Proofs
