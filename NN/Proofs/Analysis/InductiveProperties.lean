/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Analysis.SpecialFunctions.Exp
public import Mathlib.Data.Fin.Basic
public import Mathlib.Data.Nat.Basic
public import NN.Proofs.Analysis.Lipschitz
public import NN.Proofs.Tensor.Basic
public import NN.Spec.Core.Context
public import NN.Spec.Core.Shape
public import NN.Spec.Core.Tensor
public import NN.Spec.Core.TensorOps

/-!
# Tensor-shape induction and lifting lemmas

This file collects reusable *proof patterns* for reasoning about `Tensor` by structural induction
on its `Shape` (i.e. the nested `scalar`/`dim` structure), plus a few higher-level lifting lemmas
that are easiest to state once the Lipschitz/norm library is available.

## Why this exists
Many lemmas in TorchLean are naturally phrased as “for all shapes / for all dimensions …”.
Rather than re-proving the same induction scaffolding (or writing deeply nested `cases`/`induction`
blocks) throughout the repo, we keep a few canonical lemmas here.

- `tensor_induction_principle` for predicates `P : Tensor ℝ s → Prop`,
- `binary_tensor_induction` for predicates `P : Tensor ℝ s → Tensor ℝ s → Prop`.

These are especially useful when proving algebraic properties of `*_spec` tensor operations, or
norm/metric bounds that are proved “componentwise” and then lifted to the whole tensor.

Why this is not under `NN/Spec`: `NN/Spec` should define the mathematical objects and operations.
The induction principles below are theorem/proof conveniences about those objects, so they belong
under `NN/Proofs`.

## References
- This is standard structural induction on an inductive family; no external paper is required.
  The main nontrivial detail is simply that TorchLean encodes tensors as a tree indexed by `Shape`,
  rather than (say) a flat array with a runtime `shape`.
-/

@[expose] public section


namespace Proofs

open Spec
open Tensor
open Shape

-- ====================================================================
-- DIMENSIONAL INDUCTION PATTERNS
-- ====================================================================

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
    cases t with | scalar x => exact base x
  | dim n s ih =>
    cases t with | dim f =>
    apply step
    intro i
    exact ih (f i)

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
    cases t₁ with | scalar x =>
    cases t₂ with | scalar y =>
    exact base x y
  | dim n s ih =>
    cases t₁ with | dim f =>
    cases t₂ with | dim g =>
    apply step
    intro i
    exact ih (f i) (g i)

-- ====================================================================
-- NORM PRESERVATION UNDER DIMENSIONAL SCALING
-- ====================================================================

/--
Squared L2 norm of a concatenation is the sum of squared L2 norms.

Informally: `Tensor.dim f` is a “stack/concat along the outer dimension”. The Euclidean norm
satisfies `‖concat_i f i‖₂² = ∑ i, ‖f i‖₂²`.
-/
theorem l2_norm_concatenation {n : Nat} {s : Shape}
  (f : Fin n → Tensor ℝ s) :
  (tensorL2Norm (Tensor.dim f))^2 =
  (List.finRange n).foldl (fun acc i => acc + (tensorL2Norm (f i))^2) 0 := by
  classical
  have l2_sq : ∀ {s : Shape} (t : Tensor ℝ s), (tensorL2Norm t)^2 = tensorNormSquared t := by
    intro s t
    simp [tensorL2Norm, Real.sq_sqrt (tensor_norm_squared_nonneg2 (t := t))]
  calc
    (tensorL2Norm (Tensor.dim f))^2 = tensorNormSquared (Tensor.dim f) := l2_sq (t := Tensor.dim
      f)
    _ = (Finset.univ : Finset (Fin n)).sum (fun i => tensorNormSquared (f i)) := by
      calc
        tensorNormSquared (Tensor.dim f) = sumSpec (Tensor.dim (fun i => mulSpec (f i) (f i)))
          := by rfl
        _ = (Finset.univ : Finset (Fin n)).sum (fun i => sumSpec (mulSpec (f i) (f i))) := by
          -- Use the canonical lemma from `NN/Proofs/Tensor/Basic.lean` instead of duplicating the
          -- outer-fold-to-`Finset.sum` proof here.
          simpa [Spec.get] using
            (Spec.sum_spec_dim (t := Tensor.dim (fun i => mulSpec (f i) (f i))))
        _ = (Finset.univ : Finset (Fin n)).sum (fun i => tensorNormSquared (f i)) := by
          refine Finset.sum_congr rfl ?_
          intro i _
          rfl
    _ = (Finset.univ : Finset (Fin n)).sum (fun i => (tensorL2Norm (f i))^2) := by
      refine Finset.sum_congr rfl ?_
      intro i _
      simpa using (l2_sq (t := f i)).symm
    _ = (List.finRange n).foldl (fun acc i => acc + (tensorL2Norm (f i))^2) 0 := by
      simpa using
        (Spec.finRange_foldl_add_eq_finset_sum (f := fun i : Fin n => (tensorL2Norm (f
          i))^2)).symm

/--
Component-wise bounds extend to full tensors.
Key principle for lifting scalar bounds to tensor bounds.
-/
theorem componentwise_bound_extension {n : Nat} {s : Shape}
  (f g : Fin n → Tensor ℝ s) (C : ℝ)
  (h : ∀ i : Fin n, tensorL2Norm (f i) ≤ C * tensorL2Norm (g i)) :
  tensorL2Norm (Tensor.dim f) ≤ C * tensorL2Norm (Tensor.dim g) := by
  classical
  by_cases hC : 0 ≤ C
  · -- Compare squares and use `le_of_sq_le_sq` (RHS nonnegative).
    have hSf :
        (tensorL2Norm (Tensor.dim f))^2 =
          (Finset.univ : Finset (Fin n)).sum (fun i => (tensorL2Norm (f i))^2) := by
      calc
        (tensorL2Norm (Tensor.dim f))^2 =
            (List.finRange n).foldl (fun acc i => acc + (tensorL2Norm (f i))^2) 0 := by
              simpa using (l2_norm_concatenation (f := f))
        _ = (Finset.univ : Finset (Fin n)).sum (fun i => (tensorL2Norm (f i))^2) := by
          simpa using
            (Spec.finRange_foldl_add_eq_finset_sum (f := fun i : Fin n => (tensorL2Norm (f i))^2))
    have hSg :
        (tensorL2Norm (Tensor.dim g))^2 =
          (Finset.univ : Finset (Fin n)).sum (fun i => (tensorL2Norm (g i))^2) := by
      calc
        (tensorL2Norm (Tensor.dim g))^2 =
            (List.finRange n).foldl (fun acc i => acc + (tensorL2Norm (g i))^2) 0 := by
              simpa using (l2_norm_concatenation (f := g))
        _ = (Finset.univ : Finset (Fin n)).sum (fun i => (tensorL2Norm (g i))^2) := by
          simpa using
            (Spec.finRange_foldl_add_eq_finset_sum (f := fun i : Fin n => (tensorL2Norm (g i))^2))

    have h_term :
        ∀ i : Fin n, (tensorL2Norm (f i))^2 ≤ C^2 * (tensorL2Norm (g i))^2 := by
      intro i
      have hi := h i
      have hf_nonneg : 0 ≤ tensorL2Norm (f i) := by
        simpa [ge_iff_le] using tensor_l2_norm_nonneg (t := f i)
      have hg_nonneg : 0 ≤ tensorL2Norm (g i) := by
        simpa [ge_iff_le] using tensor_l2_norm_nonneg (t := g i)
      have hCg_nonneg : 0 ≤ C * tensorL2Norm (g i) := mul_nonneg hC hg_nonneg
      have hsq :
          (tensorL2Norm (f i))^2 ≤ (C * tensorL2Norm (g i))^2 := by
        have hmul :
            tensorL2Norm (f i) * tensorL2Norm (f i) ≤
              (C * tensorL2Norm (g i)) * (C * tensorL2Norm (g i)) :=
          mul_le_mul hi hi hf_nonneg hCg_nonneg
        simpa [pow_two] using hmul
      simpa [mul_pow] using hsq

    have hsquared :
        (tensorL2Norm (Tensor.dim f))^2 ≤ (C * tensorL2Norm (Tensor.dim g))^2 := by
      -- Convert to `Finset` sums and bound termwise.
      rw [hSf]
      simp [mul_pow]
      rw [hSg]
      have hsum :
          (Finset.univ : Finset (Fin n)).sum (fun i => (tensorL2Norm (f i))^2) ≤
            (Finset.univ : Finset (Fin n)).sum (fun i => C^2 * (tensorL2Norm (g i))^2) := by
        refine Finset.sum_le_sum ?_
        intro i _
        exact h_term i
      calc
        (Finset.univ : Finset (Fin n)).sum (fun i => (tensorL2Norm (f i))^2)
            ≤ (Finset.univ : Finset (Fin n)).sum (fun i => C^2 * (tensorL2Norm (g i))^2) := hsum
        _ = C^2 * (Finset.univ : Finset (Fin n)).sum (fun i => (tensorL2Norm (g i))^2) := by
          simpa using
            (Finset.mul_sum (s := (Finset.univ : Finset (Fin n)))
              (f := fun i : Fin n => (tensorL2Norm (g i))^2) (a := C^2)).symm

    have hR_nonneg : 0 ≤ C * tensorL2Norm (Tensor.dim g) := by
      have hg_nonneg : 0 ≤ tensorL2Norm (Tensor.dim g) := by
        simpa [ge_iff_le] using tensor_l2_norm_nonneg (t := Tensor.dim g)
      exact mul_nonneg hC hg_nonneg
    exact le_of_sq_le_sq hsquared hR_nonneg

  · -- If `C < 0`, the hypotheses force both sides to be zero.
    have hCneg : C < 0 := lt_of_not_ge hC

    have hg_norm0 : ∀ i : Fin n, tensorL2Norm (g i) = 0 := by
      intro i
      have hf_nonneg : 0 ≤ tensorL2Norm (f i) := by
        simpa [ge_iff_le] using tensor_l2_norm_nonneg (t := f i)
      have hCg_nonneg : 0 ≤ C * tensorL2Norm (g i) := le_trans hf_nonneg (h i)
      have hg_nonneg : 0 ≤ tensorL2Norm (g i) := by
        simpa [ge_iff_le] using tensor_l2_norm_nonneg (t := g i)
      have hCg_nonpos : C * tensorL2Norm (g i) ≤ 0 :=
        mul_nonpos_of_nonpos_of_nonneg (le_of_lt hCneg) hg_nonneg
      have hCg0 : C * tensorL2Norm (g i) = 0 := le_antisymm hCg_nonpos hCg_nonneg
      have hCne : C ≠ 0 := ne_of_lt hCneg
      rcases (mul_eq_zero.mp hCg0) with hC0 | hg0
      · exact (hCne hC0).elim
      · exact hg0

    have hf_norm0 : ∀ i : Fin n, tensorL2Norm (f i) = 0 := by
      intro i
      have hg0 := hg_norm0 i
      have hf_nonneg : 0 ≤ tensorL2Norm (f i) := by
        simpa [ge_iff_le] using tensor_l2_norm_nonneg (t := f i)
      have hf_le0 : tensorL2Norm (f i) ≤ 0 := by simpa [hg0] using (h i)
      exact le_antisymm hf_le0 hf_nonneg

    have hg0 : ∀ i : Fin n, g i = fill (0 : ℝ) s := by
      intro i
      exact (tensor_l2_norm_zero_iff (t := g i)).1 (hg_norm0 i)
    have hf0 : ∀ i : Fin n, f i = fill (0 : ℝ) s := by
      intro i
      exact (tensor_l2_norm_zero_iff (t := f i)).1 (hf_norm0 i)

    have hg_dim : Tensor.dim g = fill (0 : ℝ) (.dim n s) := by
      have : g = (fun _ : Fin n => fill (0 : ℝ) s) := by
        funext i
        exact hg0 i
      simp [this, fill]

    have hf_dim : Tensor.dim f = fill (0 : ℝ) (.dim n s) := by
      have : f = (fun _ : Fin n => fill (0 : ℝ) s) := by
        funext i
        exact hf0 i
      simp [this, fill]

    have nf0 : tensorL2Norm (Tensor.dim f) = 0 :=
      (tensor_l2_norm_zero_iff (t := Tensor.dim f)).2 hf_dim
    have ng0 : tensorL2Norm (Tensor.dim g) = 0 :=
      (tensor_l2_norm_zero_iff (t := Tensor.dim g)).2 hg_dim
    simp [nf0, ng0]

-- ====================================================================
-- ACTIVATION FUNCTION INDUCTIVE ANALYSIS
-- ====================================================================

/--
ReLU preserves non-negativity inductively over all dimensions.

PyTorch analogue: `relu` is defined pointwise as `max(x, 0)`, so its outputs are always ≥ 0.
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
      simp [Activation.reluSpec, Activation.Math.reluSpec, mapSpec]
    | cons _ _ => simp [Activation.reluSpec, Activation.Math.reluSpec, mapSpec]
  · -- Inductive case
    intro n s f ih indices
    simp [Activation.reluSpec, mapSpec]
    cases indices with
    | nil =>
      simp
    | cons head tail =>
      simp
      by_cases h : head < n
      · simpa [h] using ih ⟨head, h⟩ tail
      · simp [h]

/--
Sigmoid output bounds extend inductively.
Shows 0 < σ(x) < 1 for all tensor components.

PyTorch analogue: `torch.sigmoid` maps reals to the open interval (0, 1) pointwise.
https://pytorch.org/docs/stable/generated/torch.sigmoid.html
-/
theorem sigmoid_bounds_inductive {s : Shape} (t : Tensor ℝ s) :
  ∀ indices : List Nat,
  match getSpec (mapSpec (fun x => 1 / (1 + Real.exp (-x))) t) indices with
  | some y => 0 < y ∧ y < 1
  | none => True := by
  refine tensor_induction_principle
    (P := fun {s} t =>
      ∀ indices : List Nat,
        match getSpec (mapSpec (fun x => 1 / (1 + Real.exp (-x))) t) indices with
        | some y => 0 < y ∧ y < 1
        | none => True)
    (t := t) ?_ ?_
  · -- Base case: scalar
    intro x indices
    cases indices with
    | nil =>
      simp [mapSpec]
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
      simp [mapSpec]
  · -- Inductive case
    intro n s f ih indices
    cases indices with
    | nil =>
      simp [mapSpec]
    | cons head tail =>
      simp [mapSpec]
      by_cases h : head < n
      · simpa [h] using ih ⟨head, h⟩ tail
      · simp [h]

-- ====================================================================
-- LINEAR TRANSFORMATION INDUCTIVE PROPERTIES
-- ====================================================================

/--
Matrix-vector multiplication dimension consistency.
Proves output dimensions are correct regardless of input tensor structure.
-/
theorem matvec_dimension_consistency {m n : Nat}
  (A : Tensor ℝ (.dim m (.dim n .scalar)))
  (x : Tensor ℝ (.dim n .scalar)) :
  shapeOf (matVecMulSpec A x) = .dim m .scalar := by
  exact shapeOf_eq_shape (matVecMulSpec A x)

/--
Linear transformation preserves tensor structure inductively.
Shows that linearity holds component-wise across all dimensions.
-/
theorem linear_structure_preservation {m n : Nat}
  (A : Tensor ℝ (.dim m (.dim n .scalar))) :
  ∀ (x y : Tensor ℝ (.dim n .scalar)) (a b : ℝ),
  matVecMulSpec A (addSpec (scaleSpec x a) (scaleSpec y b)) =
  addSpec (scaleSpec (matVecMulSpec A x) a) (scaleSpec (matVecMulSpec A y) b) := by
  intro x y a b
  simpa using (Spec.mat_vec_linear_combination (W := A) (x := x) (y := y) (a := a) (b := b))

-- ====================================================================
-- COMPOSITION INDUCTIVE THEOREMS
-- ====================================================================

/--
Compose a list of functions into a single function by folding left.
This is the forward semantics of a sequential neural network.
-/
def composeFunctions {s : Shape} :
    List (Tensor ℝ s → Tensor ℝ s) →
    Tensor ℝ s → Tensor ℝ s :=
  fun fs x => fs.foldl (fun acc f => f acc) x

/--
Nested function composition preserves Lipschitz constants inductively.
This shows that a deep network is Lipschitz with constant equal to
the product of its layers’ Lipschitz constants.
-/
theorem nested_lipschitz_composition {s : Shape}
  (functions : List (Tensor ℝ s → Tensor ℝ s))
  (constants : List ℝ)
  (h_len : functions.length = constants.length)
  (h_nonneg : ∀ j : Fin constants.length, 0 ≤ constants.get j) -- needed for `mul_le_mul_*`
  (h_lipschitz : ∀ i : Fin functions.length, ∀ x y,
    tensorL2Dist (functions.get i x) (functions.get i y) ≤
      constants.get (Fin.cast (h_len) i) * tensorL2Dist x y) :
  ∀ x y, tensorL2Dist (composeFunctions functions x) (composeFunctions functions y) ≤
    constants.foldl (· * ·) 1 * tensorL2Dist x y := by
  -- A small helper: nonnegativity of a list product from pointwise nonnegativity (via `get`).
  have prod_nonneg_of_get :
      ∀ (l : List ℝ), (∀ j : Fin l.length, 0 ≤ l.get j) → 0 ≤ l.prod := by
    intro l h
    induction l with
    | nil =>
      simp
    | cons a l ih =>
      have ha : 0 ≤ a := by
        simpa using h ⟨0, by simp⟩
      have htail : ∀ j : Fin l.length, 0 ≤ l.get j := by
        intro j
        simpa using h (Fin.succ j)
      have hl : 0 ≤ l.prod := ih htail
      simpa [List.prod_cons] using mul_nonneg ha hl

  induction functions generalizing constants with
  | nil =>
    intro x y
    cases constants with
    | nil =>
      simp [composeFunctions]
    | cons c cs =>
      -- impossible: constants nonempty but functions empty
      simp at h_len
  | cons f fs ih =>
    intro x y
    cases constants with
    | nil =>
      -- impossible: functions nonempty but constants empty
      simp at h_len
    | cons c cs =>
      -- Reduce the goal to the tail-composition applied to `f x`/`f y`.
      simp [composeFunctions]
      have h_len_tail : fs.length = cs.length := by
        exact Nat.succ_inj.mp (by simpa using h_len)
      have hc_nonneg : 0 ≤ c := by
        simpa using h_nonneg ⟨0, by simp⟩
      have h_nonneg_tail : ∀ j : Fin cs.length, 0 ≤ cs.get j := by
        intro j
        simpa using h_nonneg (Fin.succ j)

      have h_lipschitz_tail :
          ∀ i : Fin fs.length, ∀ x y,
            tensorL2Dist (fs.get i x) (fs.get i y) ≤
              cs.get (Fin.cast h_len_tail i) * tensorL2Dist x y := by
        intro i x y
        have hcast_succ :
            Fin.cast h_len (Fin.succ i) = Fin.succ (Fin.cast h_len_tail i) := by
          ext
          rfl
        simpa [hcast_succ] using (h_lipschitz (Fin.succ i) x y)

      have tail_bound :
          tensorL2Dist (composeFunctions fs (f x)) (composeFunctions fs (f y)) ≤
            cs.foldl (· * ·) 1 * tensorL2Dist (f x) (f y) := by
        exact ih cs h_len_tail h_nonneg_tail h_lipschitz_tail (f x) (f y)

      have head_bound : tensorL2Dist (f x) (f y) ≤ c * tensorL2Dist x y := by
        simpa using h_lipschitz ⟨0, by simp⟩ x y

      have cs_nonneg : 0 ≤ cs.foldl (· * ·) 1 := by
        -- `cs.foldl (·*·) 1 = cs.prod` and products of nonneg terms are nonneg.
        have : 0 ≤ cs.prod := prod_nonneg_of_get cs h_nonneg_tail
        simpa [List.prod_eq_foldl] using this

      calc
        tensorL2Dist (composeFunctions fs (f x)) (composeFunctions fs (f y))
            ≤ cs.foldl (· * ·) 1 * tensorL2Dist (f x) (f y) := tail_bound
        _ ≤ cs.foldl (· * ·) 1 * (c * tensorL2Dist x y) := by
          exact mul_le_mul_of_nonneg_left head_bound cs_nonneg
        _ = (c :: cs).foldl (· * ·) 1 * tensorL2Dist x y := by
          -- Convert folds to products and reorder.
          rw [← List.prod_eq_foldl (l := cs), ← List.prod_eq_foldl (l := c :: cs)]
          simp [List.prod_cons, mul_assoc, mul_left_comm]
        _ = cs.foldl (· * ·) c * tensorL2Dist x y := by
          simp [List.foldl]

end Proofs
