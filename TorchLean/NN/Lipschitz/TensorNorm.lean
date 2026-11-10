/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Robustness
public import TorchLean.Tensor.Euclidean
public import TorchLean.Tensor.LinearAlgebra.Norm

/-!
# Real tensor norm facts

This file ties TorchLean's tensor operations to the Euclidean structure of `Tensor ℝ s`, with
Mathlib's norm `‖·‖`, distance `dist` and inner product `⟪·, ·⟫_ℝ` from
`TorchLean.Tensor.Euclidean`. The Lipschitz bounds of
`TorchLean.NN.Lipschitz.Network` are stated with them, so Mathlib's lemmas apply
directly.

`full_zero_eq_zero` and `scale_eq_smul` identify the zero-filled tensor and `Tensor.scale` with the
additive zero and the module action. `Robustness.linfNorm_le_norm` shows that on real vectors
the L∞ norm `linfNorm` of `TorchLean.Robustness` is at most `‖·‖`, so an L2 Lipschitz
bound also bounds the largest change of any output coordinate. For `y = [3, -4]` that's `4 ≤ 5`.

The Euclidean norm of a tensor is PyTorch's `torch.linalg.vector_norm` at its defaults `ord=2` and
`dim=None`, which flatten the tensor first:
https://pytorch.org/docs/stable/generated/torch.linalg.vector_norm.html
-/

@[expose] public section

open FloatLib.Numerics (MathFunctions)

namespace TorchLean

open TorchLean.Tensor

/-! ## Bridges to the inner product space structure -/

/-- The zero-filled tensor is the zero of the additive group. -/
theorem Tensor.full_zero_eq_zero (s : Shape) : Tensor.full s (0 : ℝ) = 0 :=
  rfl

/-- Scaling by a scalar is the module action. -/
theorem Tensor.scale_eq_smul {s : Shape} (t : Tensor ℝ s) (c : ℝ) : Tensor.scale t c = c • t := by
  apply Tensor.Rep.ext
  intro i
  simp only [Tensor.scale, Tensor.map_apply,
    Tensor.Rep.smul_apply, smul_eq_mul, mul_comm]

/-! ## L∞ and Euclidean norms -/

/-- For a real vector `y`, the L∞ norm `linfNorm y` is at most the Euclidean norm `‖y‖`. -/
theorem Robustness.linfNorm_le_norm {n : Nat} (y : Tensor ℝ [n]) :
    Robustness.linfNorm (α := ℝ) y ≤ ‖y‖ := by
  classical
  have hval :
      ∀ i : Fin n,
        Robustness.linfNorm (α := ℝ) (y.unstack i) ≤ ‖y‖ := by
    intro i
    have hi : |y (i, PUnit.unit)| ≤ ‖y‖ := Tensor.abs_apply_le_norm y (i, PUnit.unit)
    simpa [Robustness.linfNorm, MathFunctions.abs, Tensor.item] using hi
  simpa [Robustness.linfNorm] using
    List.foldl_max_le_of_le (List.finRange n)
      (fun i => Robustness.linfNorm (α := ℝ) (y.unstack i)) (norm_nonneg y)
      (fun i _ => hval i)

end TorchLean
