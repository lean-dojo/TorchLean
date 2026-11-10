/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public meta import TorchLean.Tensor.Einops.Elab.Einsum
public meta import TorchLean.Tensor.Einops.Elab.Pack
public meta import TorchLean.Tensor.Literal
public meta import TorchLean.Tensor.Einops.Elab.Transform
public meta import TorchLean.Tensor.Tactic.Einops.Report
public import TorchLean.Tensor.Literal
public import TorchLean.Tensor.Context.Real
public import TorchLean.Tensor.Layers.Normalization
public import TorchLean.Tensor.Tactic.Init
public import TorchLeanExamples.BugZoo.BatchNorm

/-!
# BugZoo: constant normalization slices

Normalization layers should have a boring answer on a constant slice. If every value in the slice
being normalized is the same finite value `x`, then the slice mean is `x`, the variance is zero, and
the normalized activations are zero.

For affine normalization layers this gives the contract:

$$
\operatorname{normalize}([x,x,\ldots])=\beta.
$$

The scale (weight) gradient for that slice is zero too, because it is multiplied by the normalized
activation. The same mathematical core sits behind LayerNorm, GroupNorm, InstanceNorm and
BatchNorm, which differ mainly in the axes that define the slice.

The theorems use real arithmetic with totalized division and square root, and they take the mean
and variance as inputs equal to `x` and zero. A floating-point kernel that computes those
statistics on a constant slice is outside their scope.
-/

@[expose] public section

namespace TorchLean.Examples.BugZoo.ConstantSlice

/-- Batch normalization along `dim` of the tensor with every entry `x`, with running mean `x`,
running variance zero, scale `gamma` and bias `beta` in every channel, has every entry `beta`. -/
theorem normalize_const {s : Shape} (dim : Fin s.rank)
    (x gamma beta epsilon : ℝ) :
    Tensor.batchNorm (Tensor.full s x) dim (Tensor.full _ gamma) (Tensor.full _ beta)
        (.running (Tensor.full _ x) (Tensor.full _ 0)) epsilon = Tensor.full s beta := by
  rw [BatchNorm.normalize_scalar]
  simp

/-- The reverse rule of batch normalization along `dim` at the tensor with every entry `x`, with
running mean `x` and running variance zero, gives a zero scale gradient for every scale `gamma`,
epsilon and output gradient `dy`. -/
theorem scaleGrad_eq_zero {s : Shape} (dim : Fin s.rank) (x epsilon : ℝ)
    (gamma : Tensor ℝ [s.extent dim]) (dy : Tensor ℝ s) :
    (BatchNorm.runningBackward (Tensor.full s x) dim gamma (Tensor.full _ x) (Tensor.full _ 0)
      epsilon dy).scaleGradient = 0 := by
  apply Tensor.Rep.ext
  intro o
  simp only [BatchNorm.runningBackward]
  rw [Tensor.reduce_sum_otherAxes_apply, Tensor.Rep.zero_apply]
  refine Finset.sum_eq_zero fun c _ => ite_eq_right_iff.mpr fun _ => ?_
  simp only [Tensor.sub_def, Tensor.div_def, Tensor.mul_def, Tensor.zipWith_apply,
    Tensor.full_apply]
  rw [Tensor.broadcast_axis_apply, Tensor.getScalar_eq_apply, Tensor.full_apply]
  simp

end TorchLean.Examples.BugZoo.ConstantSlice
