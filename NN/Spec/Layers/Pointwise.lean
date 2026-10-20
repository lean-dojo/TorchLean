/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Activation

/-!
# Meanings of pointwise operators

`Spec.PointwiseOp.meaning` gives every operator of `Spec.PointwiseOp` its meaning at a scalar type
with a `Context`. Negation is the dictionary's negation. The tanh GELU is the dictionary's entry
when it has one and `Activation.Math.geluSpec` otherwise, so every dictionary supports both.
FloatLib binary arithmetic has an entry that follows PyTorch's kernel step by step. The exact
GELU is `x * 0.5 * (1 + erf (x * √½))`, PyTorch's order of operations, and needs the dictionary's
`erf`. `clamp_min` is `clampBelow`, a comparison and a choice, so every dictionary supports it. The
remaining operators are the dictionary's `Context.pointwise` entries; a dictionary that omits one
leaves it unsupported.

## References

- PyTorch GELU kernels, `aten/src/ATen/native/cpu/Gelu.h`: `https://github.com/pytorch/pytorch`
- Dan Hendrycks and Kevin Gimpel, "Gaussian Error Linear Units (GELUs)", arXiv:1606.08415
-/

@[expose] public section

namespace Spec.PointwiseOp

variable {α : Type} [TorchLean.Storage α] [Context α]

/-- The exact GELU `x * 0.5 * (1 + erf (x * √½))` from an error function, in PyTorch's order. -/
def geluErf (erf : α → Except String α) (x : α) : Except String α := do
  let half : α := Rat.cast (1 / 2 : Rat)
  let value ← erf (x * MathFunctions.sqrt half)
  pure (x * half * (1 + value))

/--
PyTorch's `clamp_min` with a scalar bound: `bound` where `x < bound`, otherwise `x`, and the bound
where it is a NaN, which `bound == bound` detects. A NaN `x` is returned, and the sign of a zero `x`
is kept: eager PyTorch 2.13 returns `-0.0` for `clamp_min(-0.0, 0)` and a NaN for a NaN bound, in
binary32, binary64, bfloat16 and binary16 and in both of its loops. Over an ordered field this is
`max x bound`.
-/
def clampBelow (x bound : α) : α :=
  if bound > x then bound else if bound == bound then x else bound

/-- The meaning of a pointwise operator at a scalar type, or `none` where the type lacks one. -/
def meaning : (op : PointwiseOp) → Option (PointwiseFn α op.arity)
  | .neg => some (PointwiseFn.unary fun x => .ok (-x))
  | .gelu .tanh => some <| (Context.pointwise (α := α) (.gelu .tanh)).getD
      (PointwiseFn.unary fun x => .ok (Activation.Math.geluSpec x))
  | .gelu .none =>
      (Context.pointwise (α := α) .erf).map fun erf =>
        PointwiseFn.unary (geluErf fun x => erf fun _ => x)
  | .rsqrt => Context.pointwise .rsqrt
  | .erf => Context.pointwise .erf
  | .div rounding => Context.pointwise (.div rounding)
  | .pow => Context.pointwise .pow
  | .clampMin => some (PointwiseFn.binary fun x bound => .ok (clampBelow x bound))

/-- Negation is supported by every dictionary. -/
@[simp] theorem meaning_neg :
    meaning (α := α) .neg = some (PointwiseFn.unary fun x => .ok (-x)) := rfl

/-- A dictionary without a tanh GELU entry gives it the meaning `Activation.Math.geluSpec`. -/
theorem meaning_gelu_tanh (h : Context.pointwise (α := α) (.gelu .tanh) = none) :
    meaning (α := α) (.gelu .tanh) =
      some (PointwiseFn.unary fun x => .ok (Activation.Math.geluSpec x)) := by
  simp only [meaning, h]
  rfl

end Spec.PointwiseOp
