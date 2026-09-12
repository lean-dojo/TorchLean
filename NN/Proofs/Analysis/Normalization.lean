/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Normalization.BatchNorm
public import NN.Spec.Core.Context.Real
import Mathlib.Algebra.Order.Algebra
import Mathlib.Analysis.SpecialFunctions.Pow.NNReal
import Mathlib.Data.Sym.Sym2.Init
import Mathlib.Tactic.NormNum.GCD

/-!
# Normalization analysis properties

This file records theorem-level properties of TorchLean's normalization specs. The executable and
spec definitions live in `NN.Spec.Layers.Normalization`; this file belongs under
`NN.Proofs.Analysis` because it proves algebraic facts about those definitions over `ℝ`.

Current focus:
- **BatchNorm inference is affine** in the input `x` when the running statistics are fixed.

That fact matters because inference-time BatchNorm can be folded into a preceding/following affine
layer for verification, bound propagation, and simplification. Training-mode BatchNorm has
batch-dependent statistics and is not claimed by this theorem.
-/

@[expose] public section

namespace Proofs

open Spec TorchLean
open TorchLean TorchLean.Tensor

noncomputable section

namespace Normalization

/-!
## BatchNorm inference is affine

At inference time, BatchNorm uses fixed running mean/variance. The resulting function is affine:

$y=x\odot\operatorname{scale}+\operatorname{bias}$.
-/

/--
Scalar algebra behind inference-time BatchNorm folding.

For fixed mean $\mu$, scale $\gamma$, shift $\beta$, and standard deviation
$\operatorname{std}$, the expression

$\frac{x-\mu}{\operatorname{std}}\gamma+\beta$

is affine in $x$, with multiplicative coefficient $\gamma/\operatorname{std}$ and bias
$\beta-\mu\gamma/\operatorname{std}$.
-/
private theorem batchNorm_inference_affine_scalar (x μ γ β std : ℝ) :
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
  apply TorchLean.Tensor.Internal.Rep.ext
  intro coordinate
  simpa [Tensor.addSpec, Tensor.subSpec, Tensor.mulSpec, Tensor.divSpec, Tensor.map2Spec] using
    batchNorm_inference_affine_scalar
      (x coordinate) (mean coordinate) (gamma coordinate) (beta coordinate) (std coordinate)

/--
Inference-time BatchNorm is affine in the input `x`.

This is the public theorem users want for verification and graph simplification:

`batchNormInference x runningMean runningVar gamma beta epsilon`

is definitionally equal to a pointwise affine map

$x\,\frac{\gamma}{\operatorname{std}}
+\left(\beta-\mu\frac{\gamma}{\operatorname{std}}\right)$

after broadcasting channel parameters to the input shape and clamping the running variance exactly
as the spec does.
-/
theorem batchNorm_inference_eq_mul_add
    {channels : Nat} {sSpatial : Shape}
    (x : Tensor ℝ (.dim channels sSpatial))
    (runningMean : Tensor ℝ [channels])
    (runningVar : Tensor ℝ [channels])
    (gamma : Tensor ℝ [channels])
    (beta : Tensor ℝ [channels])
    (epsilon : ℝ := TorchLean.normalizationEpsilon) :
    Spec.batchNormInference (α := ℝ) (channels := channels) (sSpatial := sSpatial)
        x runningMean runningVar gamma beta epsilon
      =
    let s : Shape := .dim channels sSpatial
    let runningVar := Tensor.maxSpec runningVar (Tensor.full (.dim channels .scalar) 0)
    let mean_b := Spec.broadcastChannel sSpatial runningMean
    let var_b := Spec.broadcastChannel sSpatial runningVar
    let gamma_b := Spec.broadcastChannel sSpatial gamma
    let beta_b := Spec.broadcastChannel sSpatial beta
    let std := Tensor.sqrtSpec (Tensor.addSpec var_b (Tensor.full s epsilon))
    Tensor.addSpec (Tensor.mulSpec x (Tensor.divSpec gamma_b std))
      (Tensor.subSpec beta_b (Tensor.mulSpec mean_b (Tensor.divSpec gamma_b std))) := by
  simp [Spec.batchNormInference, batchNorm_inference_affine_tensor]

end Normalization

end

end Proofs
