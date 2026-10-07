/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Analysis.Lipschitz.Network

/-!
# Tensor-shape induction and lifting lemmas

The scalar and stacked views of a tensor support induction on its shape. These views describe the
mathematical components; the tensor itself uses contiguous row-major storage, not a nested tree.

`tensor_induction_principle` and `binary_tensor_induction` lift componentwise arguments to arbitrary
shapes. The remaining results give stacked L2 norm bounds, activation bounds and composition of
maps carrying a proved mathlib `LipschitzWith` constant.
-/

@[expose] public section


namespace Proofs

open Spec TorchLean TorchLean.Tensor
open Shape
open scoped BigOperators

/--
Structural induction on tensors by their `Shape`.

Informally: to prove `P t` for all tensors `t`, it suffices to prove it for scalars, and to prove
that it is preserved when we build a higher-dimensional tensor `Tensor.dim f` from its components.
-/
theorem tensor_induction_principle
  (P : ∀ {s : Shape}, Tensor ℝ s → Prop)
  (base : ∀ x : ℝ, P (Tensor.scalar x))
  (step : ∀ {n : Nat} {s : Shape} (f : Fin n → Tensor ℝ s),
    (∀ i : Fin n, P (f i)) → P (Tensor.dim f))
  : ∀ {s : Shape} (t : Tensor ℝ s), P t := by
  intro s t
  induction s with
  | scalar =>
    rw [← Tensor.scalar_item t]
    exact base t.item
  | dim n s ih =>
    rw [← Tensor.dim_unstack t]
    apply step
    intro i
    exact ih (t.unstack i)

/--
Structural induction for *binary* tensor predicates.

Informally: to prove `P t₁ t₂` for all tensors of the same shape, it suffices to prove it for
scalar pairs, and to prove it componentwise for `Tensor.dim f`/`Tensor.dim g`.
-/
theorem binary_tensor_induction
  (P : ∀ {s : Shape}, Tensor ℝ s → Tensor ℝ s → Prop)
  (base : ∀ x y : ℝ, P (Tensor.scalar x) (Tensor.scalar y))
  (step : ∀ {n : Nat} {s : Shape} (f g : Fin n → Tensor ℝ s),
    (∀ i : Fin n, P (f i) (g i)) → P (Tensor.dim f) (Tensor.dim g))
  : ∀ {s : Shape} (t₁ t₂ : Tensor ℝ s), P t₁ t₂ := by
  intro s t₁ t₂
  induction s with
  | scalar =>
    rw [← Tensor.scalar_item t₁, ← Tensor.scalar_item t₂]
    exact base t₁.item t₂.item
  | dim n s ih =>
    rw [← Tensor.dim_unstack t₁, ← Tensor.dim_unstack t₂]
    apply step
    intro i
    exact ih (t₁.unstack i) (t₂.unstack i)

/--
The squared $\ell_2$ norm of a concatenation is the sum of the squared $\ell_2$ norms.

Informally: `Tensor.dim f` is a “stack/concat along the outer dimension”. The Euclidean norm
satisfies
$\left\lVert\operatorname{concat}_i f_i\right\rVert_2^2
=\sum_i\lVert f_i\rVert_2^2$.
-/
theorem l2_norm_concatenation {n : Nat} {s : Shape}
  (f : Fin n → Tensor ℝ s) :
  (tensorL2Norm (Tensor.dim f))^2 =
  (List.finRange n).foldl (fun acc i => acc + (tensorL2Norm (f i))^2) 0 := by
  classical
  calc
    (tensorL2Norm (Tensor.dim f))^2 = tensorNormSquared (Tensor.dim f) := sq_tensorL2Norm _
    _ = (Finset.univ : Finset (Fin n)).sum (fun i => tensorNormSquared (f i)) := by
      calc
        tensorNormSquared (Tensor.dim f) = sumSpec (Tensor.dim (fun i => mulSpec (f i) (f i)))
          := by
            simp [tensorNormSquared, Spec.dot, TorchLean.Tensor.mulSpec]
        _ = (Finset.univ : Finset (Fin n)).sum (fun i => sumSpec (mulSpec (f i) (f i))) := by
          -- `sum_spec_dim` (`NN/Proofs/Tensor/Basic/Folds.lean`) is the outer-fold-to-sum step.
          simpa [Spec.get] using
            (Spec.sum_spec_dim (t := Tensor.dim (fun i => mulSpec (f i) (f i))))
        _ = (Finset.univ : Finset (Fin n)).sum (fun i => tensorNormSquared (f i)) := by
          refine Finset.sum_congr rfl ?_
          intro i _
          rfl
    _ = (Finset.univ : Finset (Fin n)).sum (fun i => (tensorL2Norm (f i))^2) :=
      Finset.sum_congr rfl fun i _ => (sq_tensorL2Norm (f i)).symm
    _ = (List.finRange n).foldl (fun acc i => acc + (tensorL2Norm (f i))^2) 0 :=
      (List.finRange_foldl_add_eq_finset_sum fun i : Fin n => (tensorL2Norm (f i))^2).symm

/--
Component-wise bounds extend to full tensors.
Key principle for lifting scalar bounds to tensor bounds.
-/
theorem componentwise_bound_extension {n : Nat} {s : Shape}
  (f g : Fin n → Tensor ℝ s) (C : ℝ)
  (h : ∀ i : Fin n, tensorL2Norm (f i) ≤ C * tensorL2Norm (g i)) :
  tensorL2Norm (Tensor.dim f) ≤ C * tensorL2Norm (Tensor.dim g) := by
  classical
  -- The squared norm of a stacked tensor is the sum of the squared component norms.
  have hSf : (tensorL2Norm (Tensor.dim f))^2 = ∑ i, (tensorL2Norm (f i))^2 := by
    rw [l2_norm_concatenation]
    exact List.finRange_foldl_add_eq_finset_sum fun i => (tensorL2Norm (f i))^2
  have hSg : (tensorL2Norm (Tensor.dim g))^2 = ∑ i, (tensorL2Norm (g i))^2 := by
    rw [l2_norm_concatenation]
    exact List.finRange_foldl_add_eq_finset_sum fun i => (tensorL2Norm (g i))^2
  by_cases hC : 0 ≤ C
  · -- Compare squares; the right-hand side is nonnegative.
    have hsquared :
        (tensorL2Norm (Tensor.dim f))^2 ≤ (C * tensorL2Norm (Tensor.dim g))^2 := by
      rw [mul_pow, hSf, hSg, Finset.mul_sum]
      refine Finset.sum_le_sum fun i _ => ?_
      rw [← mul_pow]
      simpa only [sq] using mul_self_le_mul_self (tensor_l2_norm_nonneg (f i)) (h i)
    exact le_of_sq_le_sq hsquared (mul_nonneg hC (tensor_l2_norm_nonneg _))
  · -- If `C < 0`, every component norm vanishes, hence so do both stacked norms.
    have hCneg : C < 0 := lt_of_not_ge hC
    have hg_norm0 : ∀ i : Fin n, tensorL2Norm (g i) = 0 := by
      intro i
      have hCg_nonneg : 0 ≤ C * tensorL2Norm (g i) :=
        le_trans (tensor_l2_norm_nonneg (f i)) (h i)
      have hCg_nonpos : C * tensorL2Norm (g i) ≤ 0 :=
        mul_nonpos_of_nonpos_of_nonneg hCneg.le (tensor_l2_norm_nonneg (g i))
      rcases mul_eq_zero.mp (le_antisymm hCg_nonpos hCg_nonneg) with hC0 | hg0
      · exact (hCneg.ne hC0).elim
      · exact hg0
    have hf_norm0 : ∀ i : Fin n, tensorL2Norm (f i) = 0 := by
      intro i
      have hf_le0 : tensorL2Norm (f i) ≤ 0 := by simpa [hg_norm0 i] using h i
      exact le_antisymm hf_le0 (tensor_l2_norm_nonneg (f i))
    have nf0 : tensorL2Norm (Tensor.dim f) = 0 :=
      (pow_eq_zero_iff two_ne_zero).1
        (hSf.trans (Finset.sum_eq_zero fun i _ => by simp [hf_norm0 i]))
    have ng0 : tensorL2Norm (Tensor.dim g) = 0 :=
      (pow_eq_zero_iff two_ne_zero).1
        (hSg.trans (Finset.sum_eq_zero fun i _ => by simp [hg_norm0 i]))
    simp [nf0, ng0]

/--
ReLU preserves non-negativity inductively over all dimensions.

PyTorch analogue: `relu` is defined pointwise as $\max(x,0)$, so its outputs are always
$\ge 0$.
https://pytorch.org/docs/stable/generated/torch.nn.functional.relu.html
-/
theorem relu_nonneg_inductive {s : Shape} (t : Tensor ℝ s) :
  ∀ indices : List Nat,
  match getSpec (Activation.reluSpec t) indices with
  | some x => x ≥ 0
  | none => True := by
  apply tensor_induction_principle
    (P := fun {s} t =>
      ∀ indices : List Nat,
        match getSpec (Activation.reluSpec t) indices with
        | some x => x ≥ 0
        | none => True)
    (t := t)
  · -- Base case: scalar
    intro x indices
    cases indices with
    | nil =>
      -- ReLU(x) = max x 0, so it is always nonnegative.
      simp [Activation.reluSpec, Activation.Math.reluSpec_eq_max]
    | cons _ _ => simp [Activation.reluSpec]
  · -- Inductive case
    intro n s f ih indices
    simp [Activation.reluSpec]
    cases indices with
    | nil =>
      simp
    | cons head tail =>
        simp
        by_cases h : head < n
        · simpa [Activation.reluSpec, h] using ih ⟨head, h⟩ tail
        · simp [h]

/--
Sigmoid output bounds extend inductively.
Shows $0<\sigma(x)<1$ for all tensor components.

PyTorch analogue: `torch.sigmoid` maps reals to the open interval (0, 1) pointwise.
https://pytorch.org/docs/stable/generated/torch.sigmoid.html
-/
theorem sigmoid_bounds_inductive {s : Shape} (t : Tensor ℝ s) :
  ∀ indices : List Nat,
  match getSpec (Tensor.map (fun x => 1 / (1 + Real.exp (-x))) t) indices with
  | some y => 0 < y ∧ y < 1
  | none => True := by
  refine tensor_induction_principle
    (P := fun {s} t =>
      ∀ indices : List Nat,
        match getSpec (Tensor.map (fun x => 1 / (1 + Real.exp (-x))) t) indices with
        | some y => 0 < y ∧ y < 1
        | none => True)
    (t := t) ?_ ?_
  · -- Base case: scalar
    intro x indices
    cases indices with
    | nil =>
      simp
      have hden_pos : 0 < (1 + Real.exp (-x)) := by
        have : 0 < Real.exp (-x) := by simpa using Real.exp_pos (-x)
        linarith
      have hden_lt : (1 : ℝ) < (1 + Real.exp (-x)) := by
        have : 0 < Real.exp (-x) := by simpa using Real.exp_pos (-x)
        linarith
      constructor
      · exact hden_pos
      ·
        have : (1 : ℝ) / (1 + Real.exp (-x)) < 1 := (div_lt_one hden_pos).2 hden_lt
        simpa [one_div] using this
    | cons _ _ =>
      simp
  · -- Inductive case
    intro n s f ih indices
    cases indices with
    | nil =>
      simp
    | cons head tail =>
      rw [map_dim, get_spec_dim_cons]
      by_cases h : head < n
      · simp only [h, dite_true]
        simpa using ih ⟨head, h⟩ tail
      · simp [h]

/--
A tensor map packaged with a proved Lipschitz constant.

The Lipschitz bound is Mathlib's `LipschitzWith` for the Euclidean metric on `Tensor ℝ s`;
`LipschitzLayer.dist_le` restates it in terms of `tensorL2Dist`.
-/
structure LipschitzLayer (s : Shape) where
  /-- The layer's forward map. -/
  forward : Tensor ℝ s → Tensor ℝ s
  /-- A global Lipschitz constant for `forward`. -/
  constant : NNReal
  /-- The Lipschitz bound witnessed by `constant`. -/
  lipschitz : LipschitzWith constant forward

/-- Lipschitz constants are nonnegative. -/
theorem LipschitzLayer.constant_nonneg {s : Shape} (layer : LipschitzLayer s) :
    (0 : ℝ) ≤ layer.constant :=
  layer.constant.coe_nonneg

/-- The `tensorL2Dist` bound witnessed by `constant`. -/
theorem LipschitzLayer.dist_le {s : Shape} (layer : LipschitzLayer s) (x y : Tensor ℝ s) :
    tensorL2Dist (layer.forward x) (layer.forward y) ≤ layer.constant * tensorL2Dist x y :=
  layer.lipschitz.tensorL2Dist_le x y

/-- Apply a runtime-sized sequence of shape-preserving layers from left to right. -/
def composeFunctions {s : Shape} (layers : Array (LipschitzLayer s))
    (x : Tensor ℝ s) : Tensor ℝ s :=
  layers.foldl (fun value layer => layer.forward value) x

/-- Product of the Lipschitz constants attached to a layer sequence. -/
def composedLipschitzConstant {s : Shape} (layers : Array (LipschitzLayer s)) : NNReal :=
  layers.foldr (fun layer bound => layer.constant * bound) 1

/-- The composition of proved Lipschitz layers is Lipschitz with the product constant. -/
theorem composeFunctions_lipschitzWith {s : Shape} (layers : Array (LipschitzLayer s)) :
    LipschitzWith (composedLipschitzConstant layers) (composeFunctions layers) := by
  unfold composedLipschitzConstant composeFunctions
  simp only [← Array.foldr_toList, ← Array.foldl_toList]
  generalize layers.toList = items
  induction items with
  | nil => exact LipschitzWith.id
  | cons layer layers ih =>
      simpa only [List.foldr_cons, List.foldl_cons, mul_comm, Function.comp_def] using
        ih.comp layer.lipschitz

/-- The composition of proved Lipschitz layers is Lipschitz with the product bound. -/
theorem nested_lipschitz_composition {s : Shape} (layers : Array (LipschitzLayer s))
    (x y : Tensor ℝ s) :
    tensorL2Dist (composeFunctions layers x) (composeFunctions layers y) ≤
      composedLipschitzConstant layers * tensorL2Dist x y :=
  (composeFunctions_lipschitzWith layers).tensorL2Dist_le x y

end Proofs
