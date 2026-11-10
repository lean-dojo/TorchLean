/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Models.MLP
public import TorchLean.NN.Lipschitz.Network
public import TorchLean.Tensor.Contract

/-!
# MLP robustness: basic analytic lemmas

Spec-level linear layers and two-layer ReLU MLPs satisfy L2 Lipschitz bounds with the Frobenius
norm of each weight matrix as the layer constant. Distances and norms are Mathlib's `dist` and
`‖·‖` on real tensors, and for a weight matrix `W` the Frobenius norm is `‖W‖`.

## References

- PyTorch ReLU: https://pytorch.org/docs/stable/generated/torch.nn.functional.relu.html
-/

@[expose] public section

namespace TorchLean

open _root_.TorchLean.Tensor

/-- Adding the same bias to two linear outputs does not change their L2 distance. -/
private theorem Linear.dist_eq_contract {inDim outDim : ℕ}
    (layer : Linear ℝ inDim outDim) (x y : Tensor ℝ [inDim]) :
    dist (forward layer x) (forward layer y) =
      dist (Tensor.contract layer.weights x _ _ [outDim] .matVec)
          (Tensor.contract layer.weights y _ _ [outDim] .matVec) := by
  unfold forward
  rw [dist_eq_norm, dist_eq_norm, add_sub_add_right_eq_sub]

/-- The Frobenius norm of a nonzero weight matrix is positive. -/
theorem Linear.norm_weights_pos {inDim outDim : ℕ}
    (layer : Linear ℝ inDim outDim)
    (h_weights_nonzero : layer.weights ≠ Tensor.full _ (0 : ℝ)) :
    0 < ‖layer.weights‖ := by
  rw [Tensor.full_zero_eq_zero] at h_weights_nonzero
  exact norm_pos_iff.2 h_weights_nonzero

/--
A linear layer is Lipschitz in L2 with the Frobenius norm of its weights as constant, an upper
bound on the spectral norm. The bias cancels in the difference of two outputs.
-/
theorem Linear.dist_forward_le {inDim outDim : ℕ}
    (layer : Linear ℝ inDim outDim) (x y : Tensor ℝ [inDim]) :
    dist (forward layer x) (forward layer y) ≤ ‖layer.weights‖ * dist x y := by
  rw [dist_eq_contract]
  exact Tensor.l2Dist_matVec_le layer.weights x y

/-- A two-layer ReLU MLP is Lipschitz in L2, with the product of the Frobenius norms of its two
weight matrices as constant.

The proof chains the two layer bounds through the $1$-Lipschitz ReLU. The product of layer norms
needs no information about the input region, which makes it cheap and usually looser than the
region-dependent CROWN bounds in `CROWN`. -/
theorem MLP.forward_dist_le {inDim hidDim outDim : ℕ}
    (l1 : Linear ℝ inDim hidDim)
    (l2 : Linear ℝ hidDim outDim) (x y : Tensor ℝ [inDim]) :
    dist (forward l1 l2 x) (forward l1 l2 y) ≤
      (‖l2.weights‖ * ‖l1.weights‖) * dist x y := by
  have hL2 := norm_nonneg l2.weights
  unfold forward
  calc dist (Linear.forward l2 (Tensor.relu (Linear.forward l1 x)))
        (Linear.forward l2 (Tensor.relu (Linear.forward l1 y)))
      ≤ ‖l2.weights‖ * dist
          (Tensor.relu (Linear.forward l1 x))
          (Tensor.relu (Linear.forward l1 y)) := Linear.dist_forward_le l2 _ _
    _ ≤ ‖l2.weights‖ * dist (Linear.forward l1 x) (Linear.forward l1 y) :=
        mul_le_mul_of_nonneg_left (Tensor.l2Dist_relu_le _ _) hL2
    _ ≤ ‖l2.weights‖ * (‖l1.weights‖ * dist x y) :=
        mul_le_mul_of_nonneg_left (Linear.dist_forward_le l1 x y) hL2
    _ = _ := by ring

/--
A two-layer ReLU MLP is `Robustness.IsLipschitzContinuous` in L2, with the product of the
Frobenius norms of its two weight matrices as constant (`forward_dist_le`).

The statement matches the Lipschitz hypothesis of the certified-robustness lemmas in
`TorchLean.Verify.Robustness.LipschitzCertified`.
-/
theorem MLP.forward_lipschitz {inDim hidDim outDim : ℕ}
    (l1 : Linear ℝ inDim hidDim)
    (l2 : Linear ℝ hidDim outDim) :
    Robustness.IsLipschitzContinuous
      (f := fun x => forward l1 l2 x)
      (norm₁ := (‖·‖))
      (norm₂ := (‖·‖))
      (‖l2.weights‖ * ‖l1.weights‖) := by
  intro x y
  simpa [Robustness.distance, dist_eq_norm] using
    forward_dist_le l1 l2 x y

end TorchLean
