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

This file collects small real-analysis facts used by robustness statements for spec-level MLPs.

Contents:
- ReLU (written as $\max(0,x)$) is $1$-Lipschitz on $\mathbb{R}$.
- Lipschitz bounds compose under function composition.
- A few tensor-norm helper lemmas that connect to the more general library in
  `NN.Proofs.Analysis.Lipschitz`.

## References

- PyTorch ReLU: https://pytorch.org/docs/stable/generated/torch.nn.functional.relu.html
-/

@[expose] public section

namespace NN.MLTheory.Proofs

open _root_.Spec _root_.TorchLean
open _root_.TorchLean.Tensor
open scoped BigOperators

/-! ## Basic Lipschitz lemmas -/

/--
Fundamental property: ReLU satisfies
$|\operatorname{ReLU}(x)-\operatorname{ReLU}(y)|\le|x-y|$.
-/
theorem relu_lipschitz_bound : ∀ x y : ℝ, |max 0 x - max 0 y| ≤ |x - y| := by
  intro x y
  by_cases hx : 0 ≤ x
  · by_cases hy : 0 ≤ y
    · -- x ≥ 0, y ≥ 0: max 0 acts as identity
      simp [max_eq_right hx, max_eq_right hy]
    · -- x ≥ 0, y < 0
      have hy_neg : y < 0 := not_le.mp hy
      have h_nonneg : 0 ≤ x := hx
      have h_nonneg_neg_y : 0 ≤ -y := le_of_lt (neg_pos.mpr hy_neg)
      have h_xy_nonneg : 0 ≤ x - y := by
        simpa [sub_eq_add_neg] using add_nonneg h_nonneg h_nonneg_neg_y
      -- After simp, goal reduces to x ≤ x - y
      have hx_le : x ≤ x - y := by
        simpa [sub_eq_add_neg] using add_le_add_left h_nonneg_neg_y x
      simpa [max_eq_right hx, max_eq_left (le_of_lt hy_neg), abs_of_nonneg h_nonneg, abs_of_nonneg
        h_xy_nonneg]
        using hx_le
  · by_cases hy : 0 ≤ y
    · -- x < 0, y ≥ 0 (symmetric)
      have hx_neg : x < 0 := not_le.mp hx
      have h_nonneg : 0 ≤ y := hy
      have h_nonneg_neg_x : 0 ≤ -x := le_of_lt (neg_pos.mpr hx_neg)
      have h_yx_nonneg : 0 ≤ y - x := by
        simpa [sub_eq_add_neg] using add_nonneg h_nonneg h_nonneg_neg_x
      -- After simp, goal reduces to y ≤ y - x
      have hy_le : y ≤ y - x := by
        simpa [sub_eq_add_neg] using add_le_add_left h_nonneg_neg_x y
      -- Strengthen to y ≤ |x - y|
      have hy_le2 : y ≤ |x - y| := by
        have h1 : y ≤ |y - x| := by
          simpa [abs_of_nonneg h_yx_nonneg] using hy_le
        simpa [abs_sub_comm] using h1
      -- Now simplify left-hand side to y
      simpa [max_eq_left (le_of_lt hx_neg), max_eq_right hy, sub_eq_add_neg, abs_neg,
        abs_of_nonneg h_nonneg] using hy_le2
    · -- x < 0, y < 0: both clamp to 0
      have hx_neg : x < 0 := not_le.mp hx
      have hy_neg : y < 0 := not_le.mp hy
      simp [max_eq_left (le_of_lt hx_neg), max_eq_left (le_of_lt hy_neg)]

/--
Composition property for Lipschitz functions
-/
theorem comp_lipschitz (f g : ℝ → ℝ)
    (h_f : ∀ x y, |f x - f y| ≤ |x - y|)
    (h_g : ∀ x y, |g x - g y| ≤ |x - y|) :
    ∀ x y, |(g ∘ f) x - (g ∘ f) y| ≤ |x - y| := by
  intro x y
  calc |(g ∘ f) x - (g ∘ f) y|
    = |g (f x) - g (f y)| := rfl
    _ ≤ |f x - f y| := h_g (f x) (f y)
    _ ≤ |x - y| := h_f x y

/--
Main result: ReLU composition preserves unit Lipschitz property
-/
theorem relu_comp_preserves (f : ℝ → ℝ) (h : ∀ x y, |f x - f y| ≤ |x - y|) :
    ∀ x y, |max 0 (f x) - max 0 (f y)| ≤ |x - y| := by
  exact comp_lipschitz f (fun t => max 0 t) h relu_lipschitz_bound

/-! ## Tensor helpers -/

/-- Definition: A non-zero tensor has at least one non-zero entry -/
def tensorNonzeroHasNonzeroEntry {m n : ℕ} (W : Tensor ℝ [m, n]) : Prop :=
  W ≠ Tensor.full (.dim m (.dim n .scalar)) (0 : ℝ) →
  ∃ i : Fin m, ∃ j : Fin n, get2 W i j ≠ 0

/-- A non-zero tensor has positive L2 norm -/
theorem tensor_l2_norm_pos_of_ne_zero {s : Shape} (t : Tensor ℝ s) :
  t ≠ Tensor.full s (0 : ℝ) → 0 < Proofs.tensorL2Norm t := by
  intro h_ne_zero
  have h_norm_ne_zero : Proofs.tensorL2Norm t ≠ 0 := by
    intro h_eq
    rw [Proofs.tensor_l2_norm_zero_iff] at h_eq
    exact h_ne_zero h_eq
  exact lt_of_le_of_ne (Proofs.tensor_l2_norm_nonneg t) (by simpa using h_norm_ne_zero.symm)

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

/--
Linear layers are Lipschitz continuous with the Frobenius norm as a valid, possibly loose, bound.
-/
theorem linear_layer_lipschitz_bound {inDim outDim : ℕ}
    (layer : Spec.LinearSpec ℝ inDim outDim)
    (h_weights_nonzero : layer.weights ≠ Tensor.full _ (0 : ℝ)) :
    ∃ L : ℝ, L > 0 ∧ ∀ x y : Tensor ℝ [inDim],
      Proofs.tensorL2Dist (Spec.linearSpec layer x) (Spec.linearSpec layer y) ≤
      L * Proofs.tensorL2Dist x y := by

  let L := linearLayerFrobeniusBound layer

  use L
  constructor
  · -- `matrix_op_norm` is nonnegative and only zero on the zero matrix.
    have hL_nonneg : 0 ≤ L := by
      simp [L, linearLayerFrobeniusBound, Proofs.matrixFrobeniusNorm, Real.sqrt_nonneg]

    have hL_ne : L ≠ 0 := by
      intro hL0
      -- Unfold `L` to a `sqrt` statement.
      have hsqrt0 :
          Real.sqrt (∑ i : Fin outDim, Spec.tensorNormSquared (Spec.get layer.weights i)) = 0 :=
            by
        simpa [L, linearLayerFrobeniusBound, Proofs.matrixFrobeniusNorm] using hL0

      -- The sum is nonnegative (each term is a squared norm).
      have hsum_nonneg :
          0 ≤ ∑ i : Fin outDim, Spec.tensorNormSquared (Spec.get layer.weights i) := by
        have : 0 ≤ ∑ i ∈ (Finset.univ : Finset (Fin outDim)),
              Spec.tensorNormSquared (Spec.get layer.weights i) := by
          refine Finset.sum_nonneg ?_
          intro i _
          exact Spec.tensor_norm_squared_nonneg (tensor := Spec.get layer.weights i)
        simpa using this

      have hsum0 :
          (∑ i : Fin outDim, Spec.tensorNormSquared (Spec.get layer.weights i)) = 0 :=
        (Real.sqrt_eq_zero hsum_nonneg).1 hsqrt0

      -- Turn `sum = 0` into per-row `tensor_norm_squared = 0`.
      have hsum0_mem :
          (∑ i ∈ (Finset.univ : Finset (Fin outDim)),
              Spec.tensorNormSquared (Spec.get layer.weights i)) = 0 := by
        simpa using hsum0

      have hterm_nonneg :
          ∀ i ∈ (Finset.univ : Finset (Fin outDim)),
            0 ≤ Spec.tensorNormSquared (Spec.get layer.weights i) := by
        intro i _
        exact Spec.tensor_norm_squared_nonneg (tensor := Spec.get layer.weights i)

      have hterms0_mem :
          ∀ i ∈ (Finset.univ : Finset (Fin outDim)),
            Spec.tensorNormSquared (Spec.get layer.weights i) = 0 := by
        have hiff :=
          (Finset.sum_eq_zero_iff_of_nonneg (s := (Finset.univ : Finset (Fin outDim)))
            (f := fun i : Fin outDim => Spec.tensorNormSquared (Spec.get layer.weights i))
              hterm_nonneg)
        exact (hiff.1 hsum0_mem)

      have hrows0 :
          ∀ i : Fin outDim,
            Spec.get layer.weights i = Tensor.full (.dim inDim .scalar) (0 : ℝ) := by
        intro i
        have hi0 : Spec.tensorNormSquared (Spec.get layer.weights i) = 0 :=
          hterms0_mem i (by simp)
        exact (Spec.tensor_norm_squared_zero_iff (tensor := Spec.get layer.weights i)).1 hi0

      have hw0 : layer.weights = Tensor.full (.dim outDim (.dim inDim .scalar)) (0 : ℝ) := by
        apply Spec.matrix_ext
        intro i j
        have hi := congrArg (fun row : Tensor ℝ [inDim] => row.getScalar j) (hrows0 i)
        simpa [Spec.get2] using hi

      exact h_weights_nonzero hw0

    exact lt_of_le_of_ne hL_nonneg (Ne.symm hL_ne)

  intro x y
  -- Need to handle the linear layer structure: linear_spec layer x = mat_vec_mul_spec layer.weights
  -- x + layer.bias
  -- Since bias cancels out in the difference, we can use the matrix operation bound
  have h_linear_eq : Proofs.tensorL2Dist (Spec.linearSpec layer x) (Spec.linearSpec layer y) =
    Proofs.tensorL2Dist (matVecMulSpec layer.weights x) (matVecMulSpec layer.weights y) :=
    tensorL2Dist_linearSpec_eq_matVecMulSpec layer x y

  rw [h_linear_eq]
  exact Proofs.linear_op_norm_bound layer.weights x y

/-- ReLU is 1-Lipschitz in the L2 distance, so activations never amplify an input perturbation. -/
theorem relu_activation_lipschitz {n : ℕ} (x y : Tensor ℝ [n]) :
    Proofs.tensorL2Dist (Activation.reluSpec x) (Activation.reluSpec y) ≤ Proofs.tensorL2Dist
      x y := by
  -- This follows directly from the existing relu_lipschitz_general theorem
  exact Proofs.relu_lipschitz_general x y

/-- A two-layer MLP is Lipschitz, with a positive constant obtained as the product of the layer
constants.

The constant here is the naive product of operator norms, which is what makes it cheap: it needs no
information about the input region. That is also why it is loose compared to the CROWN bounds in
`NN.MLTheory.CROWN`, and the contrast is the reason both developments are kept. -/
theorem mlp_lipschitz_complete_analysis {inDim hidDim outDim : ℕ}
    (l1 : Spec.LinearSpec ℝ inDim hidDim)
    (l2 : Spec.LinearSpec ℝ hidDim outDim)
    (h1_nonzero : l1.weights ≠ Tensor.full _ (0 : ℝ))
    (h2_nonzero : l2.weights ≠ Tensor.full _ (0 : ℝ)) :
    ∃ lipschitzConstant : ℝ, lipschitzConstant > 0 ∧
      ∀ x y : Tensor ℝ [inDim],
        Proofs.tensorL2Dist (Examples.mlpForward l1 l2 x) (Examples.mlpForward l1 l2 y) ≤
        lipschitzConstant * Proofs.tensorL2Dist x y := by

  -- Get Lipschitz constants for each layer
  obtain ⟨L1, h1_pos, h1_bound⟩ := linear_layer_lipschitz_bound l1 h1_nonzero
  obtain ⟨L2, h2_pos, h2_bound⟩ := linear_layer_lipschitz_bound l2 h2_nonzero

  -- The Lipschitz constant is the product L1 * L2
  let lipschitzConstant := L1 * L2

  use lipschitzConstant
  constructor
  · -- Need to prove L1 * L2 > 0
    exact mul_pos h1_pos h2_pos

  intro x y
  -- Unfold MLP forward pass: mlp_forward l1 l2 x = l2(ReLU(l1(x)))
  unfold Examples.mlpForward

  -- Apply composition theorem for Lipschitz functions
  -- First, establish that linear_spec is equivalent to mat_vec_mul + bias for distance purposes
  have linear_equiv_1 : Proofs.tensorL2Dist (Spec.linearSpec l1 x) (Spec.linearSpec l1 y) =
    Proofs.tensorL2Dist (matVecMulSpec l1.weights x) (matVecMulSpec l1.weights y) := by
    exact tensorL2Dist_linearSpec_eq_matVecMulSpec l1 x y

  have linear_equiv_2_pre : ∀ a b, Proofs.tensorL2Dist (Spec.linearSpec l2 a) (Spec.linearSpec
    l2 b) =
    Proofs.tensorL2Dist (matVecMulSpec l2.weights a) (matVecMulSpec l2.weights b) := by
    intro a b
    exact tensorL2Dist_linearSpec_eq_matVecMulSpec l2 a b

  have h1' : Proofs.tensorL2Dist (Spec.linearSpec l1 x) (Spec.linearSpec l1 y) ≤ L1 *
    Proofs.tensorL2Dist x y :=
    h1_bound x y

  have h2 : Proofs.tensorL2Dist (Activation.reluSpec (Spec.linearSpec l1 x))
                           (Activation.reluSpec (Spec.linearSpec l1 y)) ≤
            Proofs.tensorL2Dist (Spec.linearSpec l1 x) (Spec.linearSpec l1 y) :=
    Proofs.relu_lipschitz_general (Spec.linearSpec l1 x) (Spec.linearSpec l1 y)

  have h3 : Proofs.tensorL2Dist (Spec.linearSpec l2 (Activation.reluSpec (Spec.linearSpec l1
    x)))
                           (Spec.linearSpec l2 (Activation.reluSpec (Spec.linearSpec l1 y))) ≤
            L2 * Proofs.tensorL2Dist (Activation.reluSpec (Spec.linearSpec l1 x))
                                (Activation.reluSpec (Spec.linearSpec l1 y)) :=
    h2_bound _ _

  -- First establish non-negativity of operator norms from the positivity
  have L1_nonneg : 0 ≤ L1 := le_of_lt h1_pos
  have L2_nonneg : 0 ≤ L2 := le_of_lt h2_pos

  -- Chain the inequalities
  calc Proofs.tensorL2Dist (Spec.linearSpec l2 (Activation.reluSpec (Spec.linearSpec l1 x)))
                      (Spec.linearSpec l2 (Activation.reluSpec (Spec.linearSpec l1 y)))
    ≤ L2 * Proofs.tensorL2Dist (Activation.reluSpec (Spec.linearSpec l1 x))
                          (Activation.reluSpec (Spec.linearSpec l1 y)) := h3
    _ ≤ L2 * Proofs.tensorL2Dist (Spec.linearSpec l1 x) (Spec.linearSpec l1 y) := by
        exact mul_le_mul_of_nonneg_left h2 L2_nonneg
    _ ≤ L2 * (L1 * Proofs.tensorL2Dist x y) := by
        exact mul_le_mul_of_nonneg_left h1' L2_nonneg
    _ = (L2 * L1) * Proofs.tensorL2Dist x y := by ring
    _ = lipschitzConstant * Proofs.tensorL2Dist x y := by ring

/--
Repackage `mlp_lipschitz_complete_analysis` as a robustness-spec `isLipschitzContinuous` fact.

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
  obtain ⟨L, hLpos, hLip⟩ := mlp_lipschitz_complete_analysis (l1 := l1) (l2 := l2)
    h1_nonzero h2_nonzero
  refine ⟨L, hLpos, ?_⟩
  intro x y
  -- Rewrite robustness-spec distances into the `Proofs.tensor_l2_dist` form used by `hLip`.
  simpa [NN.MLTheory.Robustness.Spec.tensorDistance, Proofs.tensorL2Dist] using hLip x y

/-- Bound output drift on an input ball. This theorem does not by itself preserve a class label. -/
theorem mlp_output_drift_bound_on_ball {inDim hidDim outDim : ℕ}
    (l1 : Spec.LinearSpec ℝ inDim hidDim)
    (l2 : Spec.LinearSpec ℝ hidDim outDim)
    (h1_nonzero : l1.weights ≠ Tensor.full _ (0 : ℝ))
    (h2_nonzero : l2.weights ≠ Tensor.full _ (0 : ℝ))
    (x₀ : Tensor ℝ [inDim]) (perturbationRadius : ℝ) :
    perturbationRadius > 0 →
    ∃ robustnessGuarantee : ℝ, robustnessGuarantee > 0 ∧
      ∀ x : Tensor ℝ [inDim],
        Proofs.tensorL2Dist x₀ x ≤ perturbationRadius →
        Proofs.tensorL2Dist (Examples.mlpForward l1 l2 x₀) (Examples.mlpForward l1 l2 x) ≤
        robustnessGuarantee * perturbationRadius := by

  intro h_radius_pos
  obtain ⟨L, h_L_pos, h_network_lipschitz⟩ := mlp_lipschitz_complete_analysis l1 l2 h1_nonzero
    h2_nonzero
  use L, h_L_pos
  intro x h_perturbation

  calc Proofs.tensorL2Dist (Examples.mlpForward l1 l2 x₀) (Examples.mlpForward l1 l2 x)
    ≤ L * Proofs.tensorL2Dist x₀ x := h_network_lipschitz x₀ x
    _ ≤ L * perturbationRadius := mul_le_mul_of_nonneg_left h_perturbation (le_of_lt h_L_pos)

end NN.MLTheory.Proofs
/-!
Robustness statements for MLPs (ML theory layer).

This file collects robustness definitions and theorems specialized to multi-layer perceptrons,
used as a bridge between learning-theory specifications and concrete model classes.
-/
