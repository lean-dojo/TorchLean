/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Normalization

/-!
# BugZoo: constant normalization slices

Normalization layers should have a boring answer on a constant slice. If every value in the slice
being normalized is the same finite value `x`, then the slice mean is `x`, the variance is zero, and
the normalized activations are zero.

For affine normalization layers this gives the contract:

```text
normalize([x, x, ...]) = beta
```

The scale/weight gradient for that slice is also zero, because it is multiplied by the normalized
activation. This applies to the mathematical core behind LayerNorm, GroupNorm, InstanceNorm, and
BatchNorm; those layers differ mainly in which axes define the slice.

The paired Python reproducer checks this contract against PyTorch normalization kernels on large
constant tensors.
-/

@[expose] public section

namespace NN.Examples.BugZoo.ConstantNormalizationSlice

/--
TorchLean's scalar normalization core sends a constant normalized slice to the affine bias.

This is the pointwise representative of the GroupNorm/InstanceNorm/BatchNorm constant-slice
contract: once the slice statistics are `mean = x` and `variance = 0`, the normalized contribution is
zero and only `beta` remains.
-/
theorem constant_slice_normalizeCore_outputs_bias (x gamma beta epsilon : ℝ) :
    Spec.normalizeCore
        (s := .scalar)
        (s_mean := .scalar)
        (s_var := .scalar)
        (s_gamma := .scalar)
        (s_beta := .scalar)
        (epsilon := epsilon)
        (x := Spec.Tensor.scalar x)
        (mean := Spec.Tensor.scalar x)
        (variance := Spec.Tensor.scalar 0)
        (gamma := Spec.Tensor.scalar gamma)
        (beta := Spec.Tensor.scalar beta)
        (cb_mean := Spec.Shape.CanBroadcastTo.scalar_to_any .scalar)
        (cb_var := Spec.Shape.CanBroadcastTo.scalar_to_any .scalar)
        (cb_gamma := Spec.Shape.CanBroadcastTo.scalar_to_any .scalar)
        (cb_beta := Spec.Shape.CanBroadcastTo.scalar_to_any .scalar)
      = Spec.Tensor.scalar beta := by
  simp [Spec.normalizeCore, Spec.Tensor.broadcastTo, Spec.Tensor.addSpec, Spec.Tensor.subSpec,
    Spec.Tensor.mulSpec, Spec.Tensor.divSpec, Spec.Tensor.sqrtSpec, Spec.Tensor.mapSpec,
    Spec.Tensor.map2Spec, Spec.fill, Spec.replicate]

/-- The scale gradient contribution from a constant normalized slice is zero. -/
theorem constant_slice_scale_grad_zero (dy x epsilon : ℝ) :
    dy * ((x - x) / MathFunctions.sqrt (Max.max (0 + epsilon) 0)) = 0 := by
  simp

end NN.Examples.BugZoo.ConstantNormalizationSlice
