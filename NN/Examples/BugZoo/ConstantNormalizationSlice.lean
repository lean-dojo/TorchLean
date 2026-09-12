/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Normalization
public import NN.Tensor
public import NN.Spec.Core.Context.Real

/-!
# BugZoo: constant normalization slices

Normalization layers should have a boring answer on a constant slice. If every value in the slice
being normalized is the same finite value `x`, then the slice mean is `x`, the variance is zero, and
the normalized activations are zero.

For affine normalization layers this gives the contract:

$$
\operatorname{normalize}([x,x,\ldots])=\beta.
$$

The scale/weight gradient for that slice is also zero, because it is multiplied by the normalized
activation. This applies to the mathematical core behind LayerNorm, GroupNorm, InstanceNorm, and
BatchNorm; those layers differ mainly in which axes define the slice.

Run `python3 scripts/verification/normalization_contract_probe.py --device cpu` to measure forward
and backward residuals from PyTorch normalization kernels on constant tensors.

The theorems use real arithmetic with totalized division and square root. They assume the supplied
mean and variance already equal `x` and zero; they do not prove that a floating-point statistics
kernel computes those values exactly on a constant slice.
-/

@[expose] public section

open TorchLean

namespace NN.Examples.BugZoo.ConstantNormalizationSlice

/--
TorchLean's scalar normalization core sends a constant normalized slice to the affine bias.

This is the pointwise representative of the GroupNorm/InstanceNorm/BatchNorm constant-slice
contract: once the slice statistics are $\mathrm{mean}=x$ and $\mathrm{variance}=0$, the normalized
contribution is zero and only `beta` remains.
-/
theorem constant_slice_normalizeCore_outputs_bias (x gamma beta epsilon : ℝ) :
    Spec.normalizeCore
        (epsilon := epsilon)
        (x := Tensor.full [] x)
        (mean := Tensor.full [] x)
        (variance := Tensor.full [] 0)
        (gamma := Tensor.full [] gamma)
        (beta := Tensor.full [] beta)
        (cbMean := Spec.Shape.CanBroadcastTo.refl [])
        (cbVar := Spec.Shape.CanBroadcastTo.refl [])
        (cbGamma := Spec.Shape.CanBroadcastTo.refl [])
        (cbBeta := Spec.Shape.CanBroadcastTo.refl [])
      = Tensor.full [] beta := by
  apply Tensor.ext_scalar
  simp [Spec.normalizeCore, Tensor.full, Tensor.addSpec,
    Tensor.subSpec, Tensor.mulSpec, Tensor.divSpec,
    Tensor.sqrtSpec]

/-- The scale gradient contribution from a constant normalized slice is zero. -/
theorem constant_slice_scale_grad_zero (dy x epsilon : ℝ) :
    dy * ((x - x) / MathFunctions.sqrt (Max.max (0 + epsilon) 0)) = 0 := by
  simp

end NN.Examples.BugZoo.ConstantNormalizationSlice
