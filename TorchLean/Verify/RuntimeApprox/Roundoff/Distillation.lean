/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.NN.VJPOp
public import TorchLean.Tensor.Elementwise
public import TorchLean.Tensor.Movement
public import TorchLean.Tensor.Layers.Dropout
public import TorchLean.Tensor.Layers.Embedding
public import TorchLean.Tensor.Layers.Linear
public import TorchLean.Tensor.Layers.Loss
public import TorchLean.Tensor.Layers.Softmax
public import TorchLean.Tensor.ConfiguredBinary
public meta import TorchLean.Tensor.Einops.Elab.Einsum
public meta import TorchLean.Tensor.Einops.Elab.Pack
public meta import TorchLean.Tensor.Literal
public meta import TorchLean.Tensor.Einops.Elab.Transform
public meta import TorchLean.Tensor.Tactic.Einops.Report
public import FloatLib.Numerics.Enclosure.Elementary.Runtime
public import TorchLean.Tensor.Context.Real
public import FloatLib.Floats.Formats.BinaryInterchange.Analysis.StandardModel
public import FloatLib.Floats.Formats.BinaryInterchange.Configured.Real
public import FloatLib.Numerics.MixedError.Basic
public import TorchLean.Verify.RuntimeApprox.Constants
public import TorchLean.Verify.RuntimeApprox.Roundoff.Coverage
public import TorchLean.Verify.RuntimeApprox.NF.CROWN
public import TorchLean.Verify.NumericalCertificate
public import TorchLean.Verify.Floats.IEEE754.Expressions
public import TorchLean.Verify.RuntimeApprox.NF.Attention
public import TorchLean.Verify.RuntimeApprox.NF.Convolution
public import TorchLean.Verify.RuntimeApprox.NF.EndToEnd
public import TorchLean.Verify.RuntimeApprox.NF.Normalization
public import TorchLean.Verify.RuntimeApprox.NF.Optimizers
public import TorchLean.Verify.RuntimeApprox.NF.ReductionOps
public import TorchLean.Verify.Floats.IEEE754.Reductions
public import TorchLean.Verify.RuntimeApprox.Roundoff.Rules.Autograd -- shake: keep
public import TorchLean.Verify.RuntimeApprox.TransformerBlock
public import TorchLean.Verify.RuntimeApprox.Roundoff.SoftmaxShift
public import TorchLean.Tensor.Layers.Loss.Axis
public import TorchLean.NN.Models.GradientBoostedTrees
public import FloatLib.Numerics.Automation.Attributes
public import TorchLean.Tensor.Literal
public import TorchLean.Tensor.Tactic.Init
public import TorchLean.Verify.RuntimeApprox.Roundoff.Rules.Movement
public import TorchLean.Verify.RuntimeApprox.Roundoff.Rules.Tensor

/-!
# Distillation gradient at equal and nearby logits

`TorchLean.Tensor.distillLossBackward` centers the difference of student and teacher logits under
the teacher weights, takes its log-normalizer with `expm1` and `log1p`, and subtracts the sum of the
weighted differences. At equal logits each of these steps returns a zero: a finite value minus
itself is zero, a finite value times zero is zero, a sum of zeros is zero, and the correctly rounded
`expm1` and `log1p` return a zero unchanged.

`roundoff` proves the whole gradient zero at equal logits, for every IEEE format, every shape and
every axis, when the logits and the softmax weights are finite
(`distillLossBackward_self`). The rounded weights need not sum to one. The loss is
not zero at equal logits. `TorchLean.Tensor.distillLoss` is the cross-entropy against the teacher
distribution, which at equal logits is the entropy of that distribution: at two equal logits it
decodes to a value near `log 2`.

At nearby logits `roundoff mixed theorem` infers, for every IEEE format and every length, the errors
of the centering sum, of the log-normalizer and of the near case of the weighted difference. The
weights enter as stored values and need not sum to one, so the argument of the log-normalizer can be
negative. Its conditions keep one plus that argument positive.
-/

public section

namespace TorchLean.Roundoff

open FloatLib.Floats FloatLib.Floats.Formats.BinaryInterchange FloatLib.Numerics

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
  [ExecFloat.ModelCodec plan (Model format) code]

local notation "Value" => ExecFloat (Configured.Family format code plan)

/-! ## Equal logits -/

/-- A value that decodes to zero, divided by a positive count, is that value: it decodes to zero.
The pullback divides each entry by the slice count. -/
theorem toRat?_divNat_eq_zero {x : Value} {n : ℕ} (hn : (0 : ℝ) < n)
    (hx : ExecFloat.Binary.toRat? x = some 0) :
    ExecFloat.Binary.toRat? (Context.divNat x n) = some 0 := by
  rw [divNat_eq_of_toRat? (Nat.cast_pos.mp hn).ne' hx]
  simpa using hx

/-- For identical student and teacher logits `x`, every coordinate of `Tensor.distillLossBackward`
decodes to zero, for every IEEE format, every shape and every axis, when the logits and the
softmax weights are finite. -/
theorem distillLossBackward_self [Storage Value] {s : Shape} (dim : Fin s.rank)
    (x : Tensor Value s) (hformat : format.isIEEE = true)
    (hfinite : ∀ c, ExecFloat.Binary.isFinite (x c) = true)
    (hweights : ∀ c, ExecFloat.Binary.isFinite ((Tensor.softmax x dim) c) = true)
    (c : Shape.Coord s) :
    ExecFloat.Binary.toRat? (Tensor.distillLossBackward dim x x c) = some 0 := by
  unfold Tensor.distillLossBackward
  simp only [Tensor.map_apply]
  refine toRat?_divNat_eq_zero (by exact_mod_cast Tensor.axisMeanDenom_pos s dim) ?_
  roundoff

/-- At two equal logits the binary32 loss decodes to `1453635 / 2 ^ 21`, about `0.6931472`: near
`log 2`, the entropy of the uniform teacher distribution, and not zero. -/
theorem distillLoss_zeros_binary32 :
    ExecFloat.Binary.toRat? (Tensor.distillLoss ⟨0, Shape.AxisInBounds.proof⟩
      (Tensor.ofFn fun _ => 0 : Tensor Binary32 [2])
      (Tensor.ofFn fun _ => 0)) = some (1453635 / 2 ^ 21) := by
  decide +kernel

/-- The same in bfloat16, where the loss decodes to `177 / 256`, about `0.6914`. -/
theorem distillLoss_zeros_bfloat16 :
    ExecFloat.Binary.toRat? (Tensor.distillLoss ⟨0, Shape.AxisInBounds.proof⟩
      (Tensor.ofFn fun _ => 0 : Tensor BFloat16 [2])
      (Tensor.ofFn fun _ => 0)) = some (177 / 2 ^ 8) := by
  decide +kernel

/-! ## Nearby logits -/

/-- The centering sum `∑ j, p j * (s j - t j)` of `Tensor.distillLossBackward` on a vector, with
stored weights `p`. Its terms can have either sign, so each addition carries the error
accumulated so far as an absolute error. -/
roundoff mixed theorem centering_sum_error :=
  fun [Storage Value] {n : Nat} (p s t : Tensor Value [n + 1]) (_hformat : format.isIEEE = true)
    (_hp : ∀ i, ExecFloat.Binary.isFinite (p.getScalar i) = true)
    (_hs : ∀ i, ExecFloat.Binary.isFinite (s.getScalar i) = true)
    (_ht : ∀ i, ExecFloat.Binary.isFinite (t.getScalar i) = true) =>
    ExecFloat.Binary.toReal (Tensor.sum (p * (s - t)))

/-- The log-normalizer `log (1 + ∑ j, p j * (exp (d j) - 1))` of the centered differences `d`,
with stored weights `p`. The weights need not sum to one, so the argument of `log1p` can be
negative. The analysis applies the logarithm's rule with a margin on the exact side: the error of
the sum is divided by the lower end `1 + x₀ - (r * |x₀| + a)` of the enclosure of one plus the
computed sum, where `x₀` is the exact sum and `(r, a)` its error. The conditions keep that margin
positive and assume that the computed sum is finite. -/
roundoff mixed theorem log_normalizer_error :=
  fun [Storage Value] {n : Nat} (p d : Tensor Value [n + 1]) (_hformat : format.isIEEE = true)
    (_hp : ∀ i, ExecFloat.Binary.isFinite (p.getScalar i) = true)
    (_hd : ∀ i, ExecFloat.Binary.isFinite (d.getScalar i) = true) =>
    ExecFloat.Binary.toReal (Context.log1p
      (Tensor.sum (p * Tensor.map Context.expm1 d)))

/-- The near case of the weighted difference: `w * (exp r - 1)` when the computed log-ratio `r` is
below one in magnitude, and zero otherwise. Its conditions concern the near case only. -/
roundoff mixed theorem near_weighted_difference_error :=
  fun (w r : Value) (_hformat : format.isIEEE = true)
    (_hw : ExecFloat.Binary.isFinite w = true) (_hr : ExecFloat.Binary.isFinite r = true) =>
    ExecFloat.Binary.toReal
      (if MathFunctions.abs r < 1 then w * Context.expm1 r else 0)

end TorchLean.Roundoff
