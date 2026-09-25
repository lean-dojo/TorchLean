/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Models.Mlp
public import NN.Proofs.Analysis.Lipschitz.Network

/-!
# MLP robustness: basic analytic lemmas

This file proves L2 Lipschitz bounds for spec-level linear layers and two-layer ReLU MLPs, with
the Frobenius norm of each weight matrix as the layer constant. The general tensor-norm facts come
from `NN.Proofs.Analysis.Lipschitz`.

## References

- PyTorch ReLU: https://pytorch.org/docs/stable/generated/torch.nn.functional.relu.html
-/

@[expose] public section

namespace NN.MLTheory.Proofs

open _root_.Spec _root_.TorchLean
open _root_.TorchLean.Tensor
open scoped BigOperators

/-- Frobenius-norm Lipschitz bound for a linear layer's weight matrix. -/
noncomputable def linearLayerFrobeniusBound {inDim outDim : ℕ}
    (layer : Spec.LinearSpec ℝ inDim outDim) : ℝ :=
  Proofs.matrixFrobeniusNorm layer.weights

/-- Adding the same bias to two linear outputs does not change their L2 distance. -/
private theorem tensorL2Dist_linearSpec_eq_matVecMulSpec {inDim outDim : ℕ}
    (layer : Spec.LinearSpec ℝ inDim outDim) (x y : Tensor ℝ [inDim]) :
    Proofs.tensorL2Dist (Spec.linearSpec layer x) (Spec.linearSpec layer y) =
      Proofs.tensorL2Dist (matVecMulSpec layer.weights x) (matVecMulSpec layer.weights y) := by
  unfold Proofs.tensorL2Dist Spec.linearSpec
  rw [Spec.sub_spec_bias_cancel]

/-- The Frobenius constant is nonnegative. -/
theorem linearLayerFrobeniusBound_nonneg {inDim outDim : ℕ}
    (layer : Spec.LinearSpec ℝ inDim outDim) : 0 ≤ linearLayerFrobeniusBound layer := by
  simp [linearLayerFrobeniusBound, Proofs.matrixFrobeniusNorm, Real.sqrt_nonneg]

/-- The Frobenius constant is positive for a nonzero weight matrix. -/
theorem linearLayerFrobeniusBound_pos {inDim outDim : ℕ}
    (layer : Spec.LinearSpec ℝ inDim outDim)
    (h_weights_nonzero : layer.weights ≠ Tensor.full _ (0 : ℝ)) :
    0 < linearLayerFrobeniusBound layer := by
  let L := linearLayerFrobeniusBound layer
  have hL_nonneg : 0 ≤ L := linearLayerFrobeniusBound_nonneg layer
  have hL_ne : L ≠ 0 := by
    intro hL0
    have hterm_nonneg : ∀ i ∈ (Finset.univ : Finset (Fin outDim)),
        0 ≤ Spec.tensorNormSquared (Spec.get layer.weights i) :=
      fun i _ => Spec.tensor_norm_squared_nonneg (tensor := Spec.get layer.weights i)
    have hsum0 : (∑ i : Fin outDim, Spec.tensorNormSquared (Spec.get layer.weights i)) = 0 :=
      (Real.sqrt_eq_zero (Finset.sum_nonneg hterm_nonneg)).1
        (by simpa [L, linearLayerFrobeniusBound, Proofs.matrixFrobeniusNorm] using hL0)
    apply h_weights_nonzero
    apply Spec.matrix_ext
    intro i j
    have hi0 := (Finset.sum_eq_zero_iff_of_nonneg hterm_nonneg).1 hsum0 i (Finset.mem_univ _)
    have hrow := (Spec.tensor_norm_squared_zero_iff (tensor := Spec.get layer.weights i)).1 hi0
    simpa [Spec.get2] using congrArg (fun row : Tensor ℝ [inDim] => row.getScalar j) hrow
  exact lt_of_le_of_ne hL_nonneg (Ne.symm hL_ne)

/--
A linear layer is Lipschitz in L2 with the Frobenius norm of its weights as a valid, possibly
loose, constant. The bias cancels in the difference of two outputs.
-/
theorem linear_layer_lipschitz_bound {inDim outDim : ℕ}
    (layer : Spec.LinearSpec ℝ inDim outDim) (x y : Tensor ℝ [inDim]) :
    Proofs.tensorL2Dist (Spec.linearSpec layer x) (Spec.linearSpec layer y) ≤
      linearLayerFrobeniusBound layer * Proofs.tensorL2Dist x y := by
  rw [tensorL2Dist_linearSpec_eq_matVecMulSpec]
  exact Proofs.linear_op_norm_bound layer.weights x y

/-- ReLU is 1-Lipschitz in the L2 distance, so activations never amplify an input perturbation. -/
theorem relu_activation_lipschitz {n : ℕ} (x y : Tensor ℝ [n]) :
    Proofs.tensorL2Dist (Activation.reluSpec x) (Activation.reluSpec y) ≤ Proofs.tensorL2Dist
      x y := by
  -- This follows directly from the existing relu_lipschitz_general theorem
  exact Proofs.relu_lipschitz_general x y

/-- A two-layer ReLU MLP is Lipschitz in L2, with the product of the two Frobenius constants.

The constant is the naive product of layer norms, which is what makes it cheap: it needs no
information about the input region. That is also why it is loose compared to the CROWN bounds in
`NN.MLTheory.CROWN`, and the contrast is the reason both developments are kept. -/
theorem mlp_lipschitz_frobenius {inDim hidDim outDim : ℕ}
    (l1 : Spec.LinearSpec ℝ inDim hidDim)
    (l2 : Spec.LinearSpec ℝ hidDim outDim) (x y : Tensor ℝ [inDim]) :
    Proofs.tensorL2Dist (Examples.mlpForward l1 l2 x) (Examples.mlpForward l1 l2 y) ≤
      (linearLayerFrobeniusBound l2 * linearLayerFrobeniusBound l1) * Proofs.tensorL2Dist x y := by
  have hL2 := linearLayerFrobeniusBound_nonneg l2
  unfold Examples.mlpForward
  calc Proofs.tensorL2Dist (Spec.linearSpec l2 (Activation.reluSpec (Spec.linearSpec l1 x)))
        (Spec.linearSpec l2 (Activation.reluSpec (Spec.linearSpec l1 y)))
      ≤ linearLayerFrobeniusBound l2 * Proofs.tensorL2Dist
          (Activation.reluSpec (Spec.linearSpec l1 x))
          (Activation.reluSpec (Spec.linearSpec l1 y)) := linear_layer_lipschitz_bound l2 _ _
    _ ≤ linearLayerFrobeniusBound l2 *
          Proofs.tensorL2Dist (Spec.linearSpec l1 x) (Spec.linearSpec l1 y) :=
        mul_le_mul_of_nonneg_left (relu_activation_lipschitz _ _) hL2
    _ ≤ linearLayerFrobeniusBound l2 * (linearLayerFrobeniusBound l1 * Proofs.tensorL2Dist x y) :=
        mul_le_mul_of_nonneg_left (linear_layer_lipschitz_bound l1 x y) hL2
    _ = _ := by ring

/--
Repackage `mlp_lipschitz_frobenius` as a robustness-spec `isLipschitzContinuous` fact with a
positive constant. The nonzero-weight hypotheses only make the Frobenius product positive.

This is the form expected by the certified-robustness lemmas in
`NN.MLTheory.Proofs.Verification.Robustness.LipschitzCertified`.
-/
theorem mlp_is_lipschitz_continuous_l2 {inDim hidDim outDim : ℕ}
    (l1 : Spec.LinearSpec ℝ inDim hidDim)
    (l2 : Spec.LinearSpec ℝ hidDim outDim)
    (h1_nonzero : l1.weights ≠ Tensor.full _ (0 : ℝ))
    (h2_nonzero : l2.weights ≠ Tensor.full _ (0 : ℝ)) :
    ∃ L : ℝ, L > 0 ∧
      NN.MLTheory.Robustness.Spec.isLipschitzContinuous
        (f := fun x => Examples.mlpForward l1 l2 x)
        (norm₁ := Proofs.tensorL2Norm)
        (norm₂ := Proofs.tensorL2Norm)
        L := by
  refine ⟨_, mul_pos (linearLayerFrobeniusBound_pos l2 h2_nonzero)
    (linearLayerFrobeniusBound_pos l1 h1_nonzero), ?_⟩
  intro x y
  simpa [NN.MLTheory.Robustness.Spec.tensorDistance, Proofs.tensorL2Dist] using
    mlp_lipschitz_frobenius l1 l2 x y

end NN.MLTheory.Proofs
/-!
Robustness statements for MLPs (ML theory layer).

This file collects robustness definitions and theorems specialized to multi-layer perceptrons,
used as a bridge between learning-theory specifications and concrete model classes.
-/
