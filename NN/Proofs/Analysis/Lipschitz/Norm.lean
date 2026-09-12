/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.LearningTheory.Robustness.Spec
public import NN.Proofs.Tensor.Euclidean
public import NN.Proofs.Tensor.Basic.BoundsNorms
public import NN.Spec.Core.Tensor -- shake: keep

/-!
# Real-valued tensor norm facts

This module owns the proof-oriented `Tensor` norm and distance definitions over `ℝ`, together with
their algebraic properties. Neural-network Lipschitz bounds build on these facts in
`NN.Proofs.Analysis.Lipschitz.Network`.

## Scope and conventions
- Everything here is **spec-level** and **real-valued** (`ℝ`), so we can freely use Mathlib's
  analysis and order theory.
- `Tensor ℝ s` is an `InnerProductSpace ℝ` (see `NN.Proofs.Tensor.Euclidean`). The bridge lemmas
  `tensorL2Norm_eq_norm`, `tensorL2Dist_eq_dist`, and `dot_eq_inner` identify the historical
  `tensorL2Norm`, `tensorL2Dist`, and `Spec.dot` with `‖·‖`, `dist`, and `⟪·, ·⟫_ℝ`; every norm
  inequality below is then a direct instance of the Mathlib theorem.
- `NN.MLTheory.Robustness.Spec` also has scalar-polymorphic norm definitions for runtime and
  verification statements. This file does **not** duplicate that API surface; it proves real-valued
  theorems and includes bridge lemmas where those polymorphic specs need theorem-level support.

## PyTorch correspondence / citations
- $\ell_2$/$\ell_1$/$\ell_\infty$ norms correspond to PyTorch's `torch.linalg.*_norm` /
  `torch.linalg.norm` APIs.
  https://pytorch.org/docs/stable/generated/torch.linalg.vector_norm.html
  https://pytorch.org/docs/stable/generated/torch.linalg.norm.html

Import `NN.Proofs.Analysis.Lipschitz` for both the norm foundation and network bounds, or this
module when only the norm theory is needed.
-/

@[expose] public section

namespace Proofs

open Spec TorchLean
open TorchLean TorchLean.Tensor
open scoped BigOperators RealInnerProductSpace

open Spec (dot tensorNormSquared tensor_norm_squared_nonneg tensor_norm_squared_zero_iff dot_comm)

/-! ## Tensor norms and distance functions -/

/--
$\ell_2$ norm (Euclidean norm) for tensors.
Fundamental for measuring tensor magnitudes and distances.
-/
noncomputable def tensorL2Norm {s : Shape} (t : Tensor ℝ s) : ℝ :=
  Real.sqrt (tensorNormSquared t)

/--
$\ell_\infty$ norm (maximum norm) for tensors.
Important for uniform convergence and pointwise bounds.
-/
noncomputable def tensorLInftyNorm {s : Shape} (t : Tensor ℝ s) : ℝ :=
  foldlSpec (fun acc x => max acc (|x|)) (0 : ℝ) t

/--
$\ell_1$ norm (Manhattan norm) for tensors.
Useful for sparsity-inducing regularization.
-/
noncomputable def tensorL1Norm {s : Shape} (t : Tensor ℝ s) : ℝ :=
  sumSpec (absSpec t)

/--
Distance function based on the $\ell_2$ norm.
-/
noncomputable def tensorL2Dist {s : Shape} (x y : Tensor ℝ s) : ℝ :=
  tensorL2Norm (subSpec x y)

/-- Distance function based on the $\ell_\infty$ norm. -/
noncomputable def tensorLInftyDist {s : Shape} (x y : Tensor ℝ s) : ℝ :=
  tensorLInftyNorm (subSpec x y)

/-! ## Bridges to the inner product space structure -/

/-- The zero-filled tensor is the zero of the additive group. -/
theorem full_zero_eq_zero (s : Shape) : Tensor.full s (0 : ℝ) = 0 :=
  rfl

/-- Scaling by a scalar is the module action. -/
theorem scaleSpec_eq_smul {s : Shape} (t : Tensor ℝ s) (c : ℝ) : scaleSpec t c = c • t := by
  apply TorchLean.Tensor.Internal.Rep.ext
  intro i
  simp only [scaleSpec, mapSpec, Tensor.map, TorchLean.Tensor.Internal.Rep.map_apply,
    TorchLean.Tensor.Internal.Rep.smul_apply, smul_eq_mul, mul_comm]

/-- The squared norm is the coordinate sum of squares. -/
theorem tensorNormSquared_eq_sum {s : Shape} (t : Tensor ℝ s) :
    tensorNormSquared t = ∑ i, t i ^ 2 := by
  rw [tensorNormSquared, dot, sum_spec_eq_coord_sum]
  simp only [mulSpec, map2Spec_apply, sq]

/-- `Spec.dot` is the Euclidean inner product. -/
theorem dot_eq_inner {s : Shape} (x y : Tensor ℝ s) : dot x y = ⟪x, y⟫ := by
  rw [dot, sum_spec_eq_coord_sum, Tensor.inner_eq_sum]
  simp only [mulSpec, map2Spec_apply]

/-- `tensorNormSquared` is the squared Euclidean norm. -/
theorem tensorNormSquared_eq_norm_sq {s : Shape} (t : Tensor ℝ s) :
    tensorNormSquared t = ‖t‖ ^ 2 := by
  rw [tensorNormSquared_eq_sum, Tensor.norm_sq_eq_sum]

/-- `tensorL2Norm` is the Euclidean norm. -/
theorem tensorL2Norm_eq_norm {s : Shape} (t : Tensor ℝ s) : tensorL2Norm t = ‖t‖ := by
  rw [tensorL2Norm, tensorNormSquared_eq_norm_sq, Real.sqrt_sq (norm_nonneg t)]

/-- `tensorL2Dist` is the Euclidean distance. -/
theorem tensorL2Dist_eq_dist {s : Shape} (x y : Tensor ℝ s) : tensorL2Dist x y = dist x y := by
  rw [tensorL2Dist, tensorL2Norm_eq_norm, subSpec_eq_sub, dist_eq_norm]

/-!
## Cross-library norm facts

`NN.MLTheory.Robustness.Spec` defines a scalar-polymorphic `tensorLinfNorm`. In this file we work
over $\mathbb R$ and often use `tensorL2Norm`. The key inequality
$\lVert v\rVert_\infty\le\lVert v\rVert_2$ is what lets $\ell_2$-based
Lipschitz proofs feed directly into the $\ell_\infty$-robustness lemmas.
-/

/--
For a real vector-valued tensor, the $\ell_\infty$ norm from
`NN.MLTheory.Robustness.Spec` is bounded by the $\ell_2$ norm from this file:

$\lVert v\rVert_\infty\le\lVert v\rVert_2$.
-/
theorem tensor_linf_norm_le_tensor_l2_norm {n : Nat} (y : Tensor ℝ [n]) :
    NN.MLTheory.Robustness.Spec.tensorLinfNorm (α := ℝ) y ≤ tensorL2Norm y := by
  classical
  have hval :
      ∀ i : Fin n,
        NN.MLTheory.Robustness.Spec.tensorLinfNorm (α := ℝ) (y.unstack i) ≤ tensorL2Norm y := by
    intro i
    have hi : |y (i, PUnit.unit)| ≤ ‖y‖ := Tensor.abs_apply_le_norm y (i, PUnit.unit)
    rw [tensorL2Norm_eq_norm]
    simpa [NN.MLTheory.Robustness.Spec.tensorLinfNorm, MathFunctions.abs, Tensor.item,
      Tensor.unstack] using hi
  have h0 : (0 : ℝ) ≤ tensorL2Norm y := by
    rw [tensorL2Norm_eq_norm]
    exact norm_nonneg y
  simpa [NN.MLTheory.Robustness.Spec.tensorLinfNorm] using
    List.foldl_max_le_of_le (List.finRange n)
      (fun i => NN.MLTheory.Robustness.Spec.tensorLinfNorm (α := ℝ) (y.unstack i)) h0
      (fun i _ => hval i)

/-! ## Basic norm properties used throughout the Lipschitz development -/

/--
The $\ell_2$ norm is nonnegative.
-/
theorem tensor_l2_norm_nonneg {s : Shape} (t : Tensor ℝ s) :
    tensorL2Norm t ≥ (0 : ℝ) := by
  rw [tensorL2Norm_eq_norm]
  exact norm_nonneg t

/--
The $\ell_2$ norm is zero if and only if the tensor is zero.
-/
theorem tensor_l2_norm_zero_iff {s : Shape} (t : Tensor ℝ s) :
    tensorL2Norm t = (0 : ℝ) ↔ t = Tensor.full s (0 : ℝ) := by
  rw [tensorL2Norm_eq_norm, full_zero_eq_zero, norm_eq_zero]

/--
Basic lemma: dot product with zero tensor is zero.
-/
theorem dot_zero_right {s : Shape} (x : Tensor ℝ s) :
    dot x (Tensor.full s (0 : ℝ)) = (0 : ℝ) := by
  rw [dot_eq_inner, full_zero_eq_zero, inner_zero_right]

/--
Bilinearity of dot product over addition (distributive property).
-/
theorem dot_add_add {s : Shape} (x y : Tensor ℝ s) :
    dot (addSpec x y) (addSpec x y) =
      dot x x + 2 * dot x y + dot y y := by
  simp only [dot_eq_inner, addSpec_eq_add, inner_add_left, inner_add_right, real_inner_comm x y]
  ring

/-- Bilinearity of the dot product:
$\operatorname{dot}(x+ty,x+ty)=\lVert x\rVert^2+2t\langle x,y\rangle+t^2\lVert y\rVert^2$. -/
theorem dot_quadratic_expand {s : Shape} (x y : Tensor ℝ s) (t : ℝ) :
    dot (addSpec x (scaleSpec y t)) (addSpec x (scaleSpec y t)) =
      dot x x + 2 * t * dot x y + t^2 * dot y y := by
  simp only [dot_eq_inner, addSpec_eq_add, scaleSpec_eq_smul, inner_add_left, inner_add_right,
    real_inner_smul_left, real_inner_smul_right, real_inner_comm x y]
  ring

/--
Cauchy-Schwarz inequality for tensors.
For any tensors $x$ and $y$,
$|\langle x,y\rangle|\le\lVert x\rVert\,\lVert y\rVert$.
This is a fundamental inequality in inner product spaces.
-/
theorem tensor_cauchy_schwarz {s : Shape} (x y : Tensor ℝ s) :
    |dot x y| ≤ tensorL2Norm x * tensorL2Norm y := by
  rw [dot_eq_inner, tensorL2Norm_eq_norm, tensorL2Norm_eq_norm]
  exact abs_real_inner_le_norm x y

/--
Triangle inequality for the $\ell_2$ norm.
-/
theorem tensor_l2_norm_triangle {s : Shape} (x y : Tensor ℝ s) :
    tensorL2Norm (addSpec x y) ≤ tensorL2Norm x + tensorL2Norm y := by
  rw [addSpec_eq_add, tensorL2Norm_eq_norm, tensorL2Norm_eq_norm, tensorL2Norm_eq_norm]
  exact norm_add_le x y

/--
Homogeneity of the $\ell_2$ norm.
-/
theorem tensor_l2_norm_scale {s : Shape} (t : Tensor ℝ s) (c : ℝ) :
    tensorL2Norm (scaleSpec t c) = |c| * tensorL2Norm t := by
  rw [scaleSpec_eq_smul, tensorL2Norm_eq_norm, tensorL2Norm_eq_norm, norm_smul, Real.norm_eq_abs]

/-- `tensorL2Dist` is symmetric. -/
theorem tensor_l2_dist_comm {s : Shape} (x y : Tensor ℝ s) :
    tensorL2Dist x y = tensorL2Dist y x := by
  rw [tensorL2Dist_eq_dist, tensorL2Dist_eq_dist, dist_comm]

/-- Triangle inequality for `tensorL2Dist`. -/
theorem tensor_l2_dist_triangle {s : Shape} (x y z : Tensor ℝ s) :
    tensorL2Dist x z ≤ tensorL2Dist x y + tensorL2Dist y z := by
  rw [tensorL2Dist_eq_dist, tensorL2Dist_eq_dist, tensorL2Dist_eq_dist]
  exact dist_triangle x y z

end Proofs
