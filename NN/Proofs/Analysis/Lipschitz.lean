/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Algebra.BigOperators.Ring.Finset
public import Mathlib.Algebra.Order.BigOperators.Group.Finset
public import Mathlib.Analysis.Calculus.MeanValue
public import Mathlib.Analysis.Normed.Group.Basic
public import Mathlib.Data.Real.Basic
public import Mathlib.Data.Real.Sqrt
public import NN.MLTheory.LearningTheory.Robustness.Spec
public import NN.Proofs.Tensor.Basic
public import NN.Spec.Core.Context
public import NN.Spec.Core.Tensor
public import NN.Spec.Core.TensorOps
public import NN.Spec.Layers.Activation

/-!
# Lipschitz continuity library for `Tensor`-level ops

This file proves basic norm and distance facts for TorchLean tensors over `ℝ`, and uses them to
derive Lipschitz-style bounds for common neural-network building blocks.

## Scope and conventions
- Everything here is **spec-level** and **real-valued** (`ℝ`), so we can freely use Mathlib’s
  analysis and order theory.
- The main `L2` norm here is proof-oriented: it is defined from `Spec.tensorNormSquared`, the same
  dot-product/sum-of-squares object used throughout tensor algebra proofs.
- `NN.MLTheory.Robustness.Spec` also has scalar-polymorphic norm definitions for runtime and
  verification statements. This file does **not** duplicate that API surface; it proves real-valued
  theorems and includes bridge lemmas where those polymorphic specs need theorem-level support.

## PyTorch correspondence / citations
- L2/L1/L∞ norms correspond to PyTorch’s `torch.linalg.*_norm` / `torch.linalg.norm` APIs.
  https://pytorch.org/docs/stable/generated/torch.linalg.vector_norm.html
  https://pytorch.org/docs/stable/generated/torch.linalg.norm.html
- ReLU corresponds to `torch.nn.functional.relu` (and `torch.nn.ReLU`).
  https://pytorch.org/docs/stable/generated/torch.nn.functional.relu.html

## Typical downstream use
These lemmas are intended to be imported by higher-level results that need quantitative smoothness
statements, e.g.:
- proving that a composed network is Lipschitz (by composing layer-wise constants),
- justifying robustness bounds that depend on Lipschitz constants, or
- providing assumptions for convergence/step-size arguments.

## References
- The key analytic tool is the Mean Value Theorem / derivative bounds, as formalized in Mathlib:
  `Mathlib.Analysis.Calculus.MeanValue`.
- The mathematics is standard (functional analysis / optimization folklore); this file’s value is
  aligning those facts with TorchLean’s `Tensor` encoding.
-/

@[expose] public section


namespace Proofs

open Spec
open Tensor
open Activation
open scoped BigOperators

-- Reuse the tensor-algebra definitions from `Spec`/`Proofs.Tensor.Basic` instead of restating dot
-- products or squared norms locally.
open Spec (dot tensorNormSquared tensor_norm_squared_nonneg tensor_norm_squared_nonneg2
           tensor_norm_squared_zero_iff mul_spec_comm add_spec_comm dot_comm
           sum_spec_add_distrib mul_spec_add_left mul_spec_add_right
           add_spec_assoc)

-- ====================================================================
-- TENSOR NORMS AND DISTANCE FUNCTIONS
-- ====================================================================

/--
L2 norm (Euclidean norm) for tensors.
Fundamental for measuring tensor magnitudes and distances.
-/
noncomputable def tensorL2Norm {s : Shape} (t : Tensor ℝ s) : ℝ :=
  Real.sqrt (tensorNormSquared t)

/--
L∞ norm (maximum norm) for tensors.
Important for uniform convergence and pointwise bounds.
-/
noncomputable def tensorLInftyNorm {s : Shape} (t : Tensor ℝ s) : ℝ :=
  tensorFoldlSpec (fun acc x => max acc (|x|)) (0 : ℝ) t

/--
L1 norm (Manhattan norm) for tensors.
Useful for sparsity-inducing regularization.
-/
noncomputable def tensorL1Norm {s : Shape} (t : Tensor ℝ s) : ℝ :=
  sumSpec (absSpec t)

/--
Distance function based on L2 norm.
-/
noncomputable def tensorL2Dist {s : Shape} (x y : Tensor ℝ s) : ℝ :=
  tensorL2Norm (subSpec x y)

/--
Distance function based on L∞ norm.
-/
noncomputable def tensorLInftyDist {s : Shape} (x y : Tensor ℝ s) : ℝ :=
  tensorLInftyNorm (subSpec x y)

/-!
## Cross-library norm facts

`NN.MLTheory.Robustness.Spec` defines a scalar-polymorphic `tensor_linf_norm`. In this file we work
over `ℝ` and often use `tensor_l2_norm`. The key inequality `‖v‖∞ ≤ ‖v‖₂` is what lets L2-based
Lipschitz proofs feed directly into the `L∞`-robustness lemmas.
-/

/--
For a real vector-valued tensor, the `L∞` norm from `NN.MLTheory.Robustness.Spec` is bounded by the
`L2` norm from this file:

`‖v‖∞ ≤ ‖v‖₂`.
-/
theorem tensor_linf_norm_le_tensor_l2_norm {n : Nat} (y : Tensor ℝ (.dim n .scalar)) :
    NN.MLTheory.Robustness.Spec.tensorLinfNorm (α := ℝ) y ≤ tensorL2Norm y := by
  classical
  cases y with
  | dim values =>
    -- Coordinate bound: `|y[i]| ≤ ‖y‖₂`.
    have habs_toVec_le : ∀ i : Fin n, |toVec (Tensor.dim values) i| ≤ tensorL2Norm (Tensor.dim values) := by
      intro i
      have hle_sum :
          toVec (Tensor.dim values) i * toVec (Tensor.dim values) i ≤
            ∑ j : Fin n, toVec (Tensor.dim values) j * toVec (Tensor.dim values) j := by
        have h_nonneg :
            ∀ j : Fin n, 0 ≤ toVec (Tensor.dim values) j * toVec (Tensor.dim values) j := by
          intro j
          exact mul_self_nonneg (toVec (Tensor.dim values) j)
        have h' :
            toVec (Tensor.dim values) i * toVec (Tensor.dim values) i ≤
              (Finset.univ : Finset (Fin n)).sum
                (fun j => toVec (Tensor.dim values) j * toVec (Tensor.dim values) j) :=
          Finset.single_le_sum (fun j _ => h_nonneg j) (by simp)
        simpa using h'

      have hnormsq :
          tensorNormSquared (Tensor.dim values : Tensor ℝ (.dim n .scalar)) =
            ∑ j : Fin n, toVec (Tensor.dim values) j * toVec (Tensor.dim values) j := by
        simpa [tensorNormSquared] using
          (dot_vec_eq_sum (a := (Tensor.dim values)) (b := (Tensor.dim values)))

      have hle_normsq :
          toVec (Tensor.dim values) i * toVec (Tensor.dim values) i ≤
            tensorNormSquared (Tensor.dim values : Tensor ℝ (.dim n .scalar)) := by
        simpa [hnormsq] using hle_sum

      have hsqrt :
          Real.sqrt (toVec (Tensor.dim values) i * toVec (Tensor.dim values) i) ≤
            Real.sqrt (tensorNormSquared (Tensor.dim values : Tensor ℝ (.dim n .scalar))) :=
        Real.sqrt_le_sqrt hle_normsq

      have hsqrt_lhs :
          Real.sqrt (toVec (Tensor.dim values) i * toVec (Tensor.dim values) i) =
            |toVec (Tensor.dim values) i| := by
        simpa [pow_two] using (Real.sqrt_sq_eq_abs (toVec (Tensor.dim values) i))

      simpa [tensorL2Norm, hsqrt_lhs] using hsqrt

    -- Each slice (scalar) `values i` has `L∞` norm equal to `|toVec y i|`, hence is ≤ `‖y‖₂`.
    have hval :
        ∀ i : Fin n,
          NN.MLTheory.Robustness.Spec.tensorLinfNorm (α := ℝ) (values i) ≤
            tensorL2Norm (Tensor.dim values : Tensor ℝ (.dim n .scalar)) := by
      intro i
      cases hvi : values i with
      | scalar v =>
        have hi := habs_toVec_le i
        simpa [NN.MLTheory.Robustness.Spec.tensorLinfNorm, toVec, hvi] using hi

    -- Fold-max preserves an upper bound.
    have h0 : (0 : ℝ) ≤ tensorL2Norm (Tensor.dim values : Tensor ℝ (.dim n .scalar)) := by
      -- Note: `tensor_l2_norm_nonneg` is defined later in this file; avoid forward references.
      simp [tensorL2Norm]

    have foldl_max_le :
        ∀ (l : List (Fin n)) (acc : ℝ),
          acc ≤ tensorL2Norm (Tensor.dim values : Tensor ℝ (.dim n .scalar)) →
            l.foldl
                (fun acc i =>
                  max acc (NN.MLTheory.Robustness.Spec.tensorLinfNorm (α := ℝ) (values i)))
                acc
              ≤ tensorL2Norm (Tensor.dim values : Tensor ℝ (.dim n .scalar)) := by
      intro l
      induction l with
      | nil =>
        intro acc hacc
        simpa using hacc
      | cons hd tl ih =>
        intro acc hacc
        have hacc' :
            max acc (NN.MLTheory.Robustness.Spec.tensorLinfNorm (α := ℝ) (values hd)) ≤
              tensorL2Norm (Tensor.dim values : Tensor ℝ (.dim n .scalar)) :=
          max_le hacc (hval hd)
        simpa [List.foldl] using
          (ih (acc := max acc (NN.MLTheory.Robustness.Spec.tensorLinfNorm (α := ℝ) (values hd))) hacc')

    have hfold :
        (List.finRange n).foldl
            (fun acc i =>
              max acc (NN.MLTheory.Robustness.Spec.tensorLinfNorm (α := ℝ) (values i)))
            0
          ≤ tensorL2Norm (Tensor.dim values : Tensor ℝ (.dim n .scalar)) := by
      simpa using (foldl_max_le (l := List.finRange n) (acc := (0 : ℝ)) h0)

    -- Unfold the `tensor_linf_norm` definition on vectors.
    simpa [NN.MLTheory.Robustness.Spec.tensorLinfNorm] using hfold

-- ====================================================================
-- BASIC NORM PROPERTIES
-- ====================================================================

/--
L2 norm is non-negative.
-/
theorem tensor_l2_norm_nonneg {s : Shape} (t : Tensor ℝ s) :
  tensorL2Norm t ≥ (0 : ℝ) := by
  simp [tensorL2Norm]

/--
L2 norm is zero iff tensor is zero.
-/
theorem tensor_l2_norm_zero_iff {s : Shape} (t : Tensor ℝ s) :
  tensorL2Norm t = (0 : ℝ) ↔ t = fill (0 : ℝ) s := by
  rw [show tensorL2Norm t = 0 ↔ Real.sqrt (tensorNormSquared t) = 0 by rfl]
  rw [Real.sqrt_eq_zero]
  exact tensor_norm_squared_zero_iff t
  exact tensor_norm_squared_nonneg2 t

/--
Basic lemma: dot product with zero tensor is zero.
-/
theorem dot_zero_right {s : Shape} (x : Tensor ℝ s) :
  dot x (fill (0 : ℝ) s) = (0 : ℝ) := by
  -- By induction on the tensor `Shape`; the `dim` case reduces to “folding `(+ )` over zeros”.
  induction s with
  | scalar =>
    cases x with | scalar a =>
    simp only [dot, mulSpec, map2Spec, fill, sumSpec, tensorFoldlSpec]
    ring
  | dim n s ih =>
    cases x with | dim fx =>
    simp only [dot, mulSpec, map2Spec, fill, sumSpec, tensorFoldlSpec]
    -- Use induction hypothesis on each component
    have h : ∀ i : Fin n, dot (fx i) (fill (0 : ℝ) s) = (0 : ℝ) := by
      intro i
      exact ih (fx i)
    -- The goal shows the expanded form of sum_spec for a dim tensor
    -- We need to prove: tensor_foldl_spec.go (· + ·) n s (fun i => mul_spec (fx i) (fill 0 s)) 0 0
    -- = 0

    -- Key insight: each component mul_spec (fx i) (fill 0 s) has sum 0
    have component_sum_zero : ∀ i : Fin n, sumSpec (mulSpec (fx i) (fill (0 : ℝ) s)) = (0 : ℝ) :=
      by
      intro i
      rw [← dot]
      exact h i

    -- Now we prove that the fold starting from 0, adding 0 at each step, gives 0
    -- We'll use strong induction on the starting index
    suffices ∀ k, k ≤ n → tensorFoldlSpec.go (· + ·) n s (fun i => mulSpec (fx i) (fill (0 : ℝ)
      s)) k (0 : ℝ) = (0 : ℝ) by
      exact this 0 (Nat.zero_le n)

    intro k hk
    -- We'll prove by induction on n - k
    induction h_ind : n - k generalizing k with
    | zero =>
      -- Base case: n - k = 0, so k = n
      have k_eq_n : k = n := by
        have : n ≤ k := Nat.sub_eq_zero_iff_le.mp h_ind
        exact Nat.le_antisymm hk this
      subst k
      -- Since `k = n`, the `k < n` loop condition is false and `go` returns the accumulator.
      have hgo :
          tensorFoldlSpec.go (· + ·) n s (fun i => mulSpec (fx i) (fill (0 : ℝ) s)) n (0 : ℝ) =
            (0 : ℝ) := by
        simpa using
          (Spec.tensor_foldl_spec_go_of_not_lt (f := (· + ·))
              (values := fun i => mulSpec (fx i) (fill (0 : ℝ) s)) (k := n) (acc := (0 : ℝ))
              (by simp))
      simp [hgo]
    | succ m ih =>
      have hlt : k < n := by
        have : 0 < n - k := by rw [h_ind]; exact Nat.succ_pos m
        exact Nat.sub_pos_iff_lt.mp this
      rw [Spec.tensor_foldl_spec_go_of_lt (f := (· + ·))
        (values := fun i => mulSpec (fx i) (fill (0 : ℝ) s)) (k := k) (acc := (0 : ℝ)) hlt]
      have sum_zero : tensorFoldlSpec (· + ·) (0 : ℝ) (mulSpec (fx ⟨k, hlt⟩) (fill (0 : ℝ) s)) =
        (0 : ℝ) := by
        rw [← sumSpec]
        exact component_sum_zero ⟨k, hlt⟩
      rw [sum_zero]
      have h_next : n - (k + 1) = m := by
        rw [Nat.sub_succ]
        rw [h_ind]
        exact Nat.add_sub_cancel m 1
      have k_plus_one_le : k + 1 ≤ n := Nat.succ_le_of_lt hlt
      exact ih (k + 1) k_plus_one_le h_next


/--
Bilinearity of dot product over addition (distributive property).
-/
theorem dot_add_add {s : Shape} (x y : Tensor ℝ s) :
  dot (addSpec x y) (addSpec x y) =
  dot x x + 2 * dot x y + dot y y := by
  -- This is the key bilinearity property: (x + y) · (x + y) = x·x + 2x·y + y·y
  -- It follows from distributivity of dot product over addition
  simp [dot]
  -- First expand mul_spec (add_spec x y) (add_spec x y) using distributivity
  rw [mul_spec_add_left]
  -- Now we have add_spec (mul_spec x (add_spec x y)) (mul_spec y (add_spec x y))
  -- Expand each term using mul_spec_add_right
  rw [mul_spec_add_right x x y, mul_spec_add_right y x y]
  -- Now we have add_spec (add_spec (mul_spec x x) (mul_spec x y)) (add_spec (mul_spec y x)
  -- (mul_spec y y))
  -- Use commutativity of multiplication
  rw [mul_spec_comm y x]
  -- Now rearrange the nested add_spec operations
  -- We have sum_spec of: add_spec (add_spec (mul_spec x x) (mul_spec x y)) (add_spec (mul_spec x y)
  -- (mul_spec y y))
  -- First, use associativity and commutativity of add_spec to rearrange
  have rearrange : addSpec (addSpec (mulSpec x x) (mulSpec x y)) (addSpec (mulSpec x y)
    (mulSpec y y)) =
                   addSpec (addSpec (mulSpec x x) (mulSpec y y)) (addSpec (mulSpec x y)
                     (mulSpec x y)) := by
    -- Rearrange `(a + b) + (b + c)` into `(a + c) + (b + b)` using associativity/commutativity.
    rw [add_spec_assoc (mulSpec x x) (mulSpec x y) (addSpec (mulSpec x y) (mulSpec y y))]
    rw [← add_spec_assoc (mulSpec x y) (mulSpec x y) (mulSpec y y)]
    rw [add_spec_comm (addSpec (mulSpec x y) (mulSpec x y)) (mulSpec y y)]
    rw [← add_spec_assoc (mulSpec x x) (mulSpec y y) (addSpec (mulSpec x y) (mulSpec x y))]
  rw [rearrange]
  -- Apply sum_spec_add_distrib twice
  rw [sum_spec_add_distrib]
  rw [sum_spec_add_distrib (mulSpec x x) (mulSpec y y)]
  rw [sum_spec_add_distrib (mulSpec x y) (mulSpec x y)]
  -- Now we have: sum_spec (mul_spec x x) + sum_spec (mul_spec y y) + (sum_spec (mul_spec x y) +
  -- sum_spec (mul_spec x y))
  -- Now we need to show: sum_spec (mul_spec x x) + sum_spec (mul_spec y y) + (sum_spec (mul_spec x
  -- y) + sum_spec (mul_spec x y)) =
  -- sum_spec (mul_spec x x) + 2 * sum_spec (mul_spec x y) + sum_spec (mul_spec y y)
  -- This follows from the fact that a + a = 2 * a
  ring

/--
Bilinearity of dot product: dot (x + ty) (x + ty) = ||x||² + 2t⟨x,y⟩ + t²||y||²
-/
theorem dot_quadratic_expand {s : Shape} (x y : Tensor ℝ s) (t : ℝ) :
  dot (addSpec x (scaleSpec y t)) (addSpec x (scaleSpec y t)) =
  dot x x + 2 * t * dot x y + t^2 * dot y y := by
  rw [dot_add_add]
  have h1 : dot x (scaleSpec y t) = t * dot x y := by
    -- Scale in the second argument via commutativity + `Spec.dot_scale_left`.
    calc
      dot x (scaleSpec y t) = dot (scaleSpec y t) x := by simp [dot_comm]
      _ = t * dot y x := by simpa using (Spec.dot_scale_left (a := y) (b := x) (k := t))
      _ = t * dot x y := by simp [dot_comm]
  have h2 : dot (scaleSpec y t) (scaleSpec y t) = t^2 * dot y y := by
    -- Scale both arguments by repeated use of `Spec.dot_scale_left` + commutativity.
    have hy : dot y (scaleSpec y t) = t * dot y y := by
      calc
        dot y (scaleSpec y t) = dot (scaleSpec y t) y := by simp [dot_comm]
        _ = t * dot y y := by simpa using (Spec.dot_scale_left (a := y) (b := y) (k := t))
    calc
      dot (scaleSpec y t) (scaleSpec y t)
          = t * dot y (scaleSpec y t) := by
              simpa using (Spec.dot_scale_left (a := y) (b := scaleSpec y t) (k := t))
      _ = t * (t * dot y y) := by simp [hy]
      _ = t^2 * dot y y := by ring
  rw [h1, h2]
  ring

/--
Cauchy-Schwarz inequality for tensors.
For any tensors x, y: |⟨x,y⟩| ≤ ||x|| * ||y||
This is a fundamental inequality in inner product spaces.
-/
theorem tensor_cauchy_schwarz {s : Shape} (x y : Tensor ℝ s) :
  |dot x y| ≤ tensorL2Norm x * tensorL2Norm y := by
  unfold tensorL2Norm

  -- Handle the degenerate case where y = 0
  by_cases hy : tensorNormSquared y = 0
  · -- Case: y = 0, so dot x y = 0 and the inequality becomes |0| ≤ ||x|| * 0 = 0
    have y_zero : y = fill (0 : ℝ) s := by
      rw [← tensor_norm_squared_zero_iff]
      exact hy
    rw [y_zero, dot_zero_right]
    -- |0| = 0 and ||x|| * ||0|| = ||x|| * 0 = 0, so 0 ≤ 0
    simp
    have h : tensorNormSquared (fill (0 : ℝ) s) = 0 := by
      rw [tensor_norm_squared_zero_iff]
    rw [h, Real.sqrt_zero, mul_zero]

  · -- Main case: y ≠ 0
    -- We use the discriminant method: for any t ∈ ℝ, ||x + ty||² ≥ 0
    -- This gives us a quadratic in t: ||x||² + 2t⟨x,y⟩ + t²||y||² ≥ 0
    -- Since this quadratic is always non-negative, its discriminant ≤ 0

    -- For any real t, we have tensor_norm_squared (add_spec x (scale_spec y t)) ≥ 0
    have quad_nonneg : ∀ t : ℝ, tensorNormSquared (addSpec x (scaleSpec y t)) ≥ 0 := by
      intro t
      exact tensor_norm_squared_nonneg2 (addSpec x (scaleSpec y t))

    -- The quadratic expansion
    have quad_expand : ∀ t : ℝ,
      tensorNormSquared (addSpec x (scaleSpec y t)) =
      tensorNormSquared x + 2 * t * dot x y + t^2 * tensorNormSquared y := by
      intro t
      unfold tensorNormSquared
      exact dot_quadratic_expand x y t

    -- For the quadratic at² + bt + c ≥ 0 for all t, we need discriminant b² - 4ac ≤ 0
    -- Here: a = tensor_norm_squared y, b = 2 * dot x y, c = tensor_norm_squared x
    have discriminant_nonpos : (2 * dot x y)^2 ≤ 4 * tensorNormSquared x * tensorNormSquared y
      := by
      -- The discriminant of the quadratic t²||y||² + 2t⟨x,y⟩ + ||x||² must be ≤ 0
      -- since the quadratic is always non-negative
      have quad_form : ∀ t, tensorNormSquared y * t^2 + 2 * dot x y * t + tensorNormSquared x ≥
        0 := by
        intro t
        -- Rewrite using quad_expand
        calc tensorNormSquared y * t^2 + 2 * dot x y * t + tensorNormSquared x
          = tensorNormSquared x + 2 * t * dot x y + t^2 * tensorNormSquared y := by ring
          _ = tensorNormSquared (addSpec x (scaleSpec y t)) := by rw [← quad_expand t]
          _ ≥ 0 := quad_nonneg t

      -- For a quadratic at² + bt + c ≥ 0 for all t, we need b² - 4ac ≤ 0
      -- Here a = tensor_norm_squared y ≠ 0, b = 2 * dot x y, c = tensor_norm_squared x
      have a_pos : tensorNormSquared y > 0 := by
        -- `tensor_norm_squared y` is nonnegative, so if it is not `0` it must be strictly positive.
        have hle : 0 ≤ tensorNormSquared y := tensor_norm_squared_nonneg2 y
        cases lt_or_eq_of_le hle with
        | inl hlt =>
          exact hlt
        | inr heq0 =>
          -- `heq0 : 0 = tensor_norm_squared y` contradicts `hy : tensor_norm_squared y ≠ 0`.
          exact False.elim (hy (by simpa using heq0.symm))

      -- Use the discriminant inequality for quadratics
      -- If at² + bt + c ≥ 0 for all t and a > 0, then b² ≤ 4ac
      -- This is a standard result in analysis
      have discriminant : (2 * dot x y)^2 - 4 * tensorNormSquared y * tensorNormSquared x ≤ 0 :=
        by
        -- The quadratic p(t) = at² + bt + c achieves its minimum at t = -b/(2a)
        -- If p(t) ≥ 0 for all t, then p(-b/(2a)) ≥ 0
        -- This gives us the discriminant condition
        let a := tensorNormSquared y
        let b := 2 * dot x y
        let c := tensorNormSquared x
        let t_min := -b / (2 * a)
        have p_min : a * t_min^2 + b * t_min + c ≥ 0 := quad_form t_min
        -- Expand p(t_min) = c - b²/(4a)
        have expand_p_min : a * t_min^2 + b * t_min + c = c - b^2 / (4 * a) := by
          -- Substitute t_min = -b/(2a) into the quadratic
          simp only [t_min]
          field_simp [a_pos.ne']
          ring
        rw [expand_p_min] at p_min
        -- From c - b²/(4a) ≥ 0 we get b² ≤ 4ac
        have : b^2 / (4 * a) ≤ c := by linarith
        have : b^2 ≤ 4 * a * c := by
          -- From this : b^2 / (4 * a) ≤ c
          -- Multiply both sides by 4 * a (which is positive)
          have h : 0 < 4 * a := by linarith
          -- We want to show b^2 ≤ 4 * a * c
          -- We have b^2 / (4 * a) ≤ c
          -- Multiply both sides by 4 * a
          have h4a : (4 * a) ≠ 0 := by
            exact mul_ne_zero (by norm_num) a_pos.ne'
          calc b^2 = b^2 / (4 * a) * (4 * a) := by
                -- `b^2 / (4a) * (4a) = b^2` since `4a ≠ 0`.
                have : b^2 / (4 * a) * (4 * a) = b^2 := by
                  calc
                    b^2 / (4 * a) * (4 * a) = (b^2 * (4 * a)) / (4 * a) := by
                      simp [div_mul_eq_mul_div]
                    _ = b^2 := by
                      simpa [mul_assoc] using (mul_div_cancel_right₀ (b^2) h4a)
                simpa using this.symm
               _ ≤ c * (4 * a) := by exact mul_le_mul_of_nonneg_right this (le_of_lt h)
               _ = 4 * a * c := by ring
        -- Substitute back
        simp only [a, b, c] at this
        -- Need to show the right multiplication order
        calc (2 * dot x y)^2 - 4 * tensorNormSquared y * tensorNormSquared x
          = b^2 - 4 * a * c := by simp only [a, b, c]
          _ ≤ 0 := by linarith

      -- From discriminant: (2 * dot x y)^2 - 4 * tensor_norm_squared y * tensor_norm_squared x ≤ 0
      -- We need: (2 * dot x y)^2 ≤ 4 * tensor_norm_squared x * tensor_norm_squared y
      -- Since multiplication is commutative: 4 * tensor_norm_squared y * tensor_norm_squared x = 4
      -- * tensor_norm_squared x * tensor_norm_squared y
      have comm : 4 * tensorNormSquared y * tensorNormSquared x = 4 * tensorNormSquared x *
        tensorNormSquared y := by ring
      linarith

    -- Simplify the discriminant inequality to get |⟨x,y⟩| ≤ ||x||||y||
    have cs_squared : (dot x y)^2 ≤ tensorNormSquared x * tensorNormSquared y := by
      -- From discriminant_nonpos: (2 * dot x y)² ≤ 4 * tensor_norm_squared x * tensor_norm_squared
      -- y
      -- Simplify: 4 * (dot x y)² ≤ 4 * tensor_norm_squared x * tensor_norm_squared y
      -- Hence: (dot x y)² ≤ tensor_norm_squared x * tensor_norm_squared y
      have h : 4 * (dot x y)^2 ≤ 4 * tensorNormSquared x * tensorNormSquared y := by
        -- (2 * dot x y)^2 = 4 * (dot x y)^2
        have expand : (2 * dot x y)^2 = 4 * (dot x y)^2 := by ring
        rw [expand] at discriminant_nonpos
        exact discriminant_nonpos
      linarith

    -- Take square roots to get the final result
    have sqrt_ineq : |dot x y| ≤ Real.sqrt (tensorNormSquared x) * Real.sqrt (tensorNormSquared
      y) := by
      -- |a|² = a² and √(a²) ≤ √b iff a² ≤ b when b ≥ 0
      have abs_sq : |dot x y|^2 = (dot x y)^2 := by
        -- |a|² = a² for any real number a
        exact sq_abs (dot x y)

      -- We want to show: |dot x y| ≤ √(tensor_norm_squared x) * √(tensor_norm_squared y)
      -- Square both sides: |dot x y|² ≤ (√(tensor_norm_squared x) * √(tensor_norm_squared y))²
      -- This becomes: (dot x y)² ≤ tensor_norm_squared x * tensor_norm_squared y
      -- which we have as cs_squared

      have rhs_sq : (Real.sqrt (tensorNormSquared x) * Real.sqrt (tensorNormSquared y))^2 =
        tensorNormSquared x * tensorNormSquared y := by
        rw [mul_pow]
        simp [Real.sq_sqrt (tensor_norm_squared_nonneg2 x), Real.sq_sqrt
          (tensor_norm_squared_nonneg2 y)]

      -- Use the fact that if a² ≤ b² and a,b ≥ 0, then a ≤ b
      have lhs_nonneg : 0 ≤ |dot x y| := by
        -- |·| is non-negative
        exact abs_nonneg (dot x y)
      have rhs_nonneg : 0 ≤ Real.sqrt (tensorNormSquared x) * Real.sqrt (tensorNormSquared y) :=
        by
        exact mul_nonneg (Real.sqrt_nonneg _) (Real.sqrt_nonneg _)

      -- Apply square root monotonicity
      -- We have: |dot x y|² = (dot x y)² ≤ tensor_norm_squared x * tensor_norm_squared y
      -- Taking square roots: |dot x y| ≤ √(tensor_norm_squared x * tensor_norm_squared y)
      have h : |dot x y| ^ 2 ≤ (Real.sqrt (tensorNormSquared x) * Real.sqrt (tensorNormSquared
        y))^2 := by
        rw [abs_sq, rhs_sq]
        exact cs_squared
      -- Since sqrt is monotone on non-negative reals
      rw [← Real.sqrt_sq lhs_nonneg, ← Real.sqrt_sq rhs_nonneg]
      exact Real.sqrt_le_sqrt h

    exact sqrt_ineq

/--
Triangle inequality for L2 norm.
-/
theorem tensor_l2_norm_triangle {s : Shape} (x y : Tensor ℝ s) :
  tensorL2Norm (addSpec x y) ≤ tensorL2Norm x + tensorL2Norm y := by
  -- We prove this by showing the squared version and taking square roots
  -- Since all norms are non-negative, ||a|| ≤ ||b|| + ||c|| iff ||a||² ≤ (||b|| + ||c||)²

  -- The strategy: show tensor_norm_squared (add_spec x y) ≤ (tensor_l2_norm x + tensor_l2_norm y)²
  -- then use properties of square roots

  have squared_ineq : tensorNormSquared (addSpec x y) ≤ (tensorL2Norm x + tensorL2Norm y)^2
    := by
    -- Expand both sides
    have expand_left : tensorNormSquared (addSpec x y) =
      tensorNormSquared x + 2 * dot x y + tensorNormSquared y := by
      unfold tensorNormSquared
      exact dot_add_add x y

    have expand_right : (tensorL2Norm x + tensorL2Norm y)^2 =
      tensorNormSquared x + 2 * tensorL2Norm x * tensorL2Norm y + tensorNormSquared y := by
      unfold tensorL2Norm
      ring_nf
      simp only [Real.sq_sqrt (tensor_norm_squared_nonneg2 x), Real.sq_sqrt
        (tensor_norm_squared_nonneg2 y)]

    rw [expand_left, expand_right]

    -- Reduce to: dot x y ≤ tensor_l2_norm x * tensor_l2_norm y
    suffices h : dot x y ≤ tensorL2Norm x * tensorL2Norm y by linarith

    -- This follows from Cauchy-Schwarz: |⟨x,y⟩| ≤ ||x||||y|| implies dot x y ≤ ||x||||y||
    have cs := tensor_cauchy_schwarz x y
    have le_abs := le_abs_self (dot x y)
    exact le_trans le_abs (le_trans cs (le_refl _))

  -- Now convert the squared inequality back to the original form
  -- We have: tensor_norm_squared (add_spec x y) ≤ (tensor_l2_norm x + tensor_l2_norm y)²
  -- We want: tensor_l2_norm (add_spec x y) ≤ tensor_l2_norm x + tensor_l2_norm y
  -- This follows from the monotonicity of square root and the fact that both sides are non-negative

  unfold tensorL2Norm

  -- Apply sqrt to both sides of the squared inequality
  have rhs_nonneg : 0 ≤ tensorL2Norm x + tensorL2Norm y := by
    exact add_nonneg (tensor_l2_norm_nonneg x) (tensor_l2_norm_nonneg y)

  have sqrt_rhs : Real.sqrt ((tensorL2Norm x + tensorL2Norm y)^2) = tensorL2Norm x +
    tensorL2Norm y := by
    exact Real.sqrt_sq rhs_nonneg

  -- Use calc to chain the inequalities
  calc Real.sqrt (tensorNormSquared (addSpec x y))
    ≤ Real.sqrt ((tensorL2Norm x + tensorL2Norm y)^2) := Real.sqrt_le_sqrt squared_ineq
    _ = tensorL2Norm x + tensorL2Norm y := sqrt_rhs
    _ = Real.sqrt (tensorNormSquared x) + Real.sqrt (tensorNormSquared y) := by
      unfold tensorL2Norm; rfl
/--
Homogeneity of L2 norm.
-/
theorem tensor_l2_norm_scale {s : Shape} (t : Tensor ℝ s) (c : ℝ) :
  tensorL2Norm (scaleSpec t c) = |c| * tensorL2Norm t := by
  -- The homogeneity property follows from the bilinearity of the dot product
  -- ||c*t||² = ⟨c*t, c*t⟩ = c² * ⟨t, t⟩ = c² * ||t||²
  -- Taking square roots: ||c*t|| = |c| * ||t||
  unfold tensorL2Norm tensorNormSquared
  -- Goal: Real.sqrt (dot (scale_spec t c) (scale_spec t c)) = |c| * Real.sqrt (dot t t)

  -- First, we need to show that dot (scale_spec t c) (scale_spec t c) = c² * dot t t
  have h_dot_scale : dot (scaleSpec t c) (scaleSpec t c) = c * c * dot t t := by
    have h_right : dot t (scaleSpec t c) = c * dot t t := by
      calc
        dot t (scaleSpec t c) = dot (scaleSpec t c) t := by simp [dot_comm]
        _ = c * dot t t := by simpa using (Spec.dot_scale_left (a := t) (b := t) (k := c))
    calc
      dot (scaleSpec t c) (scaleSpec t c)
          = c * dot t (scaleSpec t c) := by
              simpa using (Spec.dot_scale_left (a := t) (b := scaleSpec t c) (k := c))
      _ = c * (c * dot t t) := by simp [h_right]
      _ = c * c * dot t t := by ring

  rw [h_dot_scale]
  -- Goal: Real.sqrt (c * c * dot t t) = |c| * Real.sqrt (dot t t)

  -- Key fact: c * c = |c|²
  have c_sq : c * c = |c|^2 := by
    -- c * c = c² = |c|²
    -- This follows from c² = |c|²
    rw [← sq]
    rw [sq_abs]

  rw [c_sq]
  -- Goal: Real.sqrt (|c|² * dot t t) = |c| * Real.sqrt (dot t t)

  -- Use the fact that sqrt(a² * b) = a * sqrt(b) when a ≥ 0 and b ≥ 0
  have h_nonneg : 0 ≤ dot t t := tensor_norm_squared_nonneg2 t
  have abs_nonneg : 0 ≤ |c| := by
    exact abs_nonneg c

  -- Apply the square root property: sqrt(a² * b) = a * sqrt(b) when a ≥ 0 and b ≥ 0
  rw [Real.sqrt_mul (sq_nonneg |c|)]
  rw [Real.sqrt_sq abs_nonneg]

-- ====================================================================
-- RELU LIPSCHITZ CONTINUITY PROOFS
-- ===================================================================

/--
Pointwise ReLU is 1-Lipschitz for scalars.
Foundation for tensor-level Lipschitz bounds.
-/
theorem relu_scalar_lipschitz (x y : ℝ) :
  |max (0 : ℝ) x - max (0 : ℝ) y| ≤ |x - y| := by
  -- ReLU is 1-Lipschitz: |max(0,x) - max(0,y)| ≤ |x - y|
  -- This follows from case analysis on the signs of x and y
  -- We'll consider four cases based on the signs of x and y
  by_cases hx : (0 : ℝ) ≤ x
  · by_cases hy : (0 : ℝ) ≤ y
    · -- Case 1: x ≥ 0 and y ≥ 0, so max 0 x = x and max 0 y = y
      simp [max_eq_right hx, max_eq_right hy]
    · -- Case 2: x ≥ 0 and y < 0, so max 0 x = x and max 0 y = 0
      push Not at hy
      simp [max_eq_right hx, max_eq_left (le_of_lt hy)]
      -- Need to show |x - 0| ≤ |x - y|
      -- Since x ≥ 0 and y < 0, we have x - y > x
      have h : (0 : ℝ) ≤ x - y := by linarith
      have hx_pos : (0 : ℝ) ≤ x := hx
      rw [abs_of_nonneg hx_pos, abs_of_nonneg h]
      simp
      linarith
  · push Not at hx
    by_cases hy : (0 : ℝ) ≤ y
    · -- Case 3: x < 0 and y ≥ 0, so max 0 x = 0 and max 0 y = y
      simp [max_eq_left (le_of_lt hx), max_eq_right hy]
      -- Need to show |0 - y| ≤ |x - y|
      -- Since x < 0 and y ≥ 0, we have |x - y| ≥ y
      have h : x - y ≤ (0 : ℝ) := by linarith
      rw [abs_of_nonneg hy, abs_of_nonpos h]
      simp
      linarith
    · -- Case 4: x < 0 and y < 0, so max 0 x = 0 and max 0 y = 0
      push Not at hy
      simp [max_eq_left (le_of_lt hx), max_eq_left (le_of_lt hy)]

/--
ReLU is 1-Lipschitz on scalar tensors.
-/
theorem relu_scalar_tensor_lipschitz (x y : Tensor ℝ .scalar) :
  tensorL2Dist (reluSpec x) (reluSpec y) ≤ tensorL2Dist x y := by
  cases x with | scalar a =>
  cases y with | scalar b =>
  unfold reluSpec tensorL2Dist tensorL2Norm tensorNormSquared dot subSpec
  simp [mapSpec, map2Spec, sumSpec, tensorFoldlSpec, mulSpec]
  unfold Math.reluSpec
  -- Goal is now about square roots of squares
  -- We need to show: √((relu a - relu b)²) ≤ √((a - b)²)
  -- Since sqrt is monotone, this is equivalent to (relu a - relu b)² ≤ (a - b)²
  apply Real.sqrt_le_sqrt
  -- Now we need: (max a 0 - max b 0)² ≤ (a - b)², which follows from the scalar Lipschitz bound.
  have h_abs : |max a 0 - max b 0| ≤ |a - b| := by
    simpa [max_comm] using (relu_scalar_lipschitz a b)
  -- Convert absolute value inequality to squared inequality
  have h_sq : |max a 0 - max b 0|^2 ≤ |a - b|^2 := by
    -- Use the fact that if 0 ≤ x ≤ y, then x² ≤ y²
    have h1 : (0 : ℝ) ≤ |max a 0 - max b 0| := abs_nonneg _
    have h2 : (0 : ℝ) ≤ |a - b| := abs_nonneg _
    -- Since |max 0 a - max 0 b| ≤ |a - b| and both are non-negative
    -- we can square both sides using monotonicity of squaring on non-negative reals
    -- Use monotonicity of squaring on non-negative reals
    -- If 0 ≤ a ≤ b, then a² ≤ b²
    -- Since |max 0 a - max 0 b| ≤ |a - b| and both are non-negative
    -- we can square both sides
    have : |max a 0 - max b 0| * |max a 0 - max b 0| ≤ |a - b| * |a - b| := by
      exact mul_self_le_mul_self h1 h_abs
    rw [← sq, ← sq] at this
    exact this
  -- Use the fact that |x|² = x²
  rw [sq_abs, sq_abs] at h_sq
  -- Convert from ^2 to multiplication
  simp only [sq] at h_sq
  exact h_sq

/--
General ReLU Lipschitz theorem for arbitrary tensor shapes.
Main result: ReLU is 1-Lipschitz in L2 norm for any tensor shape.
-/
theorem relu_lipschitz_general {s : Shape} (x y : Tensor ℝ s) :
  tensorL2Dist (reluSpec x) (reluSpec y) ≤ tensorL2Dist x y := by
  induction s with
  | scalar => exact relu_scalar_tensor_lipschitz x y
  | dim n s' ih =>
    cases x with | dim fx =>
    cases y with | dim fy =>
    unfold reluSpec tensorL2Dist tensorL2Norm tensorNormSquared dot subSpec
    simp [mapSpec, map2Spec, sumSpec, mulSpec, tensorFoldlSpec]
    -- The key insight: for vectors, ||relu(x) - relu(y)||² = Σᵢ (relu(xᵢ) - relu(yᵢ))²
    -- and ||x - y||² = Σᵢ (xᵢ - yᵢ)²
    -- Since ReLU is 1-Lipschitz componentwise, each term satisfies (relu(xᵢ) - relu(yᵢ))² ≤ (xᵢ -
    -- yᵢ)²
    apply Real.sqrt_le_sqrt
    -- We need to show the sum of squared differences is preserved
    -- This requires showing the fold preserves the inequality
    suffices ∀ k acc_relu acc_orig, k ≤ n → acc_relu ≤ acc_orig →
      tensorFoldlSpec.go (· + ·) n s'
        (fun i => mulSpec (subSpec (mapSpec Math.reluSpec (fx i)) (mapSpec Math.reluSpec (fy
          i)))
                          (subSpec (mapSpec Math.reluSpec (fx i)) (mapSpec Math.reluSpec (fy
                            i)))) k acc_relu ≤
      tensorFoldlSpec.go (· + ·) n s'
        (fun i => mulSpec (subSpec (fx i) (fy i)) (subSpec (fx i) (fy i))) k acc_orig by
      exact this 0 (0 : ℝ) (0 : ℝ) (Nat.zero_le n) (le_refl (0 : ℝ))

    intro k acc_relu acc_orig hk hacc
    induction hn : n - k generalizing k acc_relu acc_orig with
    | zero =>
      have k_eq_n : k = n := by
        have : n ≤ k := Nat.sub_eq_zero_iff_le.mp hn
        exact Nat.le_antisymm hk this
      subst k
      have hgo_relu :
          tensorFoldlSpec.go (· + ·) n s'
              (fun i =>
                mulSpec
                  (subSpec (mapSpec Math.reluSpec (fx i)) (mapSpec Math.reluSpec (fy i)))
                  (subSpec (mapSpec Math.reluSpec (fx i)) (mapSpec Math.reluSpec (fy i))))
              n acc_relu
            = acc_relu := by
        simpa using
          (Spec.tensor_foldl_spec_go_of_not_lt (f := (· + ·))
              (values := fun i =>
                mulSpec
                  (subSpec (mapSpec Math.reluSpec (fx i)) (mapSpec Math.reluSpec (fy i)))
                  (subSpec (mapSpec Math.reluSpec (fx i)) (mapSpec Math.reluSpec (fy i))))
              (k := n) (acc := acc_relu) (by simp))
      have hgo_orig :
          tensorFoldlSpec.go (· + ·) n s' (fun i => mulSpec (subSpec (fx i) (fy i)) (subSpec (fx i) (fy i)))
              n acc_orig
            = acc_orig := by
        simpa using
          (Spec.tensor_foldl_spec_go_of_not_lt (f := (· + ·))
              (values := fun i =>
                mulSpec (subSpec (fx i) (fy i)) (subSpec (fx i) (fy i)))
              (k := n) (acc := acc_orig) (by simp))
      simpa [hgo_relu, hgo_orig] using hacc
    | succ m ih_fold =>
      have hlt : k < n := by
        have : 0 < n - k := by rw [hn]; exact Nat.succ_pos m
        exact Nat.sub_pos_iff_lt.mp this
      -- Peel one `go` step on both sides.
      rw [Spec.tensor_foldl_spec_go_of_lt (f := (· + ·))
        (values := fun i =>
          mulSpec
            (subSpec (mapSpec Math.reluSpec (fx i)) (mapSpec Math.reluSpec (fy i)))
            (subSpec (mapSpec Math.reluSpec (fx i)) (mapSpec Math.reluSpec (fy i))))
        (k := k) (acc := acc_relu) hlt]
      rw [Spec.tensor_foldl_spec_go_of_lt (f := (· + ·))
        (values := fun i =>
          mulSpec (subSpec (fx i) (fy i)) (subSpec (fx i) (fy i)))
        (k := k) (acc := acc_orig) hlt]
      have h_next : n - (k + 1) = m := by
        rw [Nat.sub_succ, hn]
        rfl
      have k_plus_one_le : k + 1 ≤ n := Nat.succ_le_of_lt hlt
      -- Need to show the accumulated inequality is preserved first
      have component_ineq :
        sumSpec (mulSpec (subSpec (mapSpec Math.reluSpec (fx ⟨k, hlt⟩))
                                     (mapSpec Math.reluSpec (fy ⟨k, hlt⟩)))
                          (subSpec (mapSpec Math.reluSpec (fx ⟨k, hlt⟩))
                                     (mapSpec Math.reluSpec (fy ⟨k, hlt⟩)))) ≤
        sumSpec (mulSpec (subSpec (fx ⟨k, hlt⟩) (fy ⟨k, hlt⟩))
                          (subSpec (fx ⟨k, hlt⟩) (fy ⟨k, hlt⟩))) := by
        -- This is the squared L2 distance for component k
        -- We need ||relu(fx[k]) - relu(fy[k])||² ≤ ||fx[k] - fy[k]||²
        have h_comp : tensorL2Dist (mapSpec Math.reluSpec (fx ⟨k, hlt⟩))
                                    (mapSpec Math.reluSpec (fy ⟨k, hlt⟩)) ≤
                     tensorL2Dist (fx ⟨k, hlt⟩) (fy ⟨k, hlt⟩) := by
          -- Apply the induction hypothesis to component k
          have : reluSpec (fx ⟨k, hlt⟩) = mapSpec Math.reluSpec (fx ⟨k, hlt⟩) := by
            unfold reluSpec
            rfl
          have : reluSpec (fy ⟨k, hlt⟩) = mapSpec Math.reluSpec (fy ⟨k, hlt⟩) := by
            unfold reluSpec
            rfl
          simp
          exact ih (fx ⟨k, hlt⟩) (fy ⟨k, hlt⟩)
        -- Square both sides to get the desired inequality
        unfold tensorL2Dist tensorL2Norm at h_comp
        have h_sq : Real.sqrt (tensorNormSquared (subSpec (mapSpec Math.reluSpec (fx ⟨k, hlt⟩))
                                                            (mapSpec Math.reluSpec (fy ⟨k,
                                                              hlt⟩)))) ≤
                   Real.sqrt (tensorNormSquared (subSpec (fx ⟨k, hlt⟩) (fy ⟨k, hlt⟩))) := h_comp
        -- Apply Real.le_sqrt_iff_sq_le_sq to get the squared inequality
        have h_sq' : tensorNormSquared (subSpec (mapSpec Math.reluSpec (fx ⟨k, hlt⟩))
                                                  (mapSpec Math.reluSpec (fy ⟨k, hlt⟩))) ≤
                     tensorNormSquared (subSpec (fx ⟨k, hlt⟩) (fy ⟨k, hlt⟩)) := by
          -- From h_sq: √a ≤ √b, we want to show a ≤ b
          -- Since sqrt is strictly monotone on non-negative reals
          have ha := tensor_norm_squared_nonneg2 (subSpec (mapSpec Math.reluSpec (fx ⟨k, hlt⟩))
                                                           (mapSpec Math.reluSpec (fy ⟨k, hlt⟩)))
          have hb := tensor_norm_squared_nonneg2 (subSpec (fx ⟨k, hlt⟩) (fy ⟨k, hlt⟩))
          -- sqrt is monotone, so √a ≤ √b implies a ≤ b when both args are non-negative
          -- sqrt is monotone, so √a ≤ √b implies a ≤ b when both args are non-negative
          -- Use the fact that sqrt is monotone on non-negative reals
          -- If √a ≤ √b and a,b ≥ 0, then a ≤ b
          -- Since sqrt is strictly monotone on non-negative reals
          -- We can use the fact that if sqrt(a) ≤ sqrt(b) then a ≤ b
          have : Real.sqrt (tensorNormSquared (subSpec (mapSpec Math.reluSpec (fx ⟨k, hlt⟩))
                                                          (mapSpec Math.reluSpec (fy ⟨k, hlt⟩))))
                                                            ≤
                 Real.sqrt (tensorNormSquared (subSpec (fx ⟨k, hlt⟩) (fy ⟨k, hlt⟩))) := h_sq
          -- Apply monotonicity of sqrt: if √a ≤ √b and a,b ≥ 0, then a ≤ b
          -- Since sqrt is strictly monotone on non-negative reals, √a ≤ √b implies a ≤ b
          -- We'll prove this by contradiction
          by_contra h_not_le
          push Not at h_not_le
          -- If a > b, then √a > √b
          have h_sqrt_gt : Real.sqrt (tensorNormSquared (subSpec (fx ⟨k, hlt⟩) (fy ⟨k, hlt⟩))) <
                           Real.sqrt (tensorNormSquared (subSpec (mapSpec Math.reluSpec (fx ⟨k,
                             hlt⟩))
                                                                   (mapSpec Math.reluSpec (fy ⟨k,
                                                                     hlt⟩)))) := by
            exact Real.sqrt_lt_sqrt hb h_not_le
          -- But this contradicts our assumption that √a ≤ √b
          linarith
        unfold tensorNormSquared dot at h_sq'
        exact h_sq'
      -- Add the inequalities
      -- We have: acc_relu ≤ acc_orig (from hacc)
      -- We have: component_ineq tells us the sum of squared differences for ReLU is ≤ the original
      -- For the fold with addition, we need to show that adding these to the accumulators preserves
      -- the inequality
      -- Note that tensor_foldl_spec (· + ·) acc t adds sum_spec t to acc
      have h1 : tensorFoldlSpec (· + ·) acc_relu
                  (mulSpec (subSpec (mapSpec Math.reluSpec (fx ⟨k, hlt⟩))
                                     (mapSpec Math.reluSpec (fy ⟨k, hlt⟩)))
                           (subSpec (mapSpec Math.reluSpec (fx ⟨k, hlt⟩))
                                    (mapSpec Math.reluSpec (fy ⟨k, hlt⟩)))) =
                acc_relu + sumSpec (mulSpec (subSpec (mapSpec Math.reluSpec (fx ⟨k, hlt⟩))
                                                       (mapSpec Math.reluSpec (fy ⟨k, hlt⟩)))
                                             (subSpec (mapSpec Math.reluSpec (fx ⟨k, hlt⟩))
                                                      (mapSpec Math.reluSpec (fy ⟨k, hlt⟩)))) :=
                                                        by
        simpa using
          (Spec.tensor_foldl_spec_add_init (s := s')
            (acc := acc_relu)
            (t :=
              mulSpec (subSpec (mapSpec Math.reluSpec (fx ⟨k, hlt⟩))
                                (mapSpec Math.reluSpec (fy ⟨k, hlt⟩)))
                       (subSpec (mapSpec Math.reluSpec (fx ⟨k, hlt⟩))
                                (mapSpec Math.reluSpec (fy ⟨k, hlt⟩)))))
      have h2 : tensorFoldlSpec (· + ·) acc_orig
                  (mulSpec (subSpec (fx ⟨k, hlt⟩) (fy ⟨k, hlt⟩))
                           (subSpec (fx ⟨k, hlt⟩) (fy ⟨k, hlt⟩))) =
                acc_orig + sumSpec (mulSpec (subSpec (fx ⟨k, hlt⟩) (fy ⟨k, hlt⟩))
                                             (subSpec (fx ⟨k, hlt⟩) (fy ⟨k, hlt⟩))) := by
        simpa using
          (Spec.tensor_foldl_spec_add_init (s := s')
            (acc := acc_orig)
            (t := mulSpec (subSpec (fx ⟨k, hlt⟩) (fy ⟨k, hlt⟩))
                          (subSpec (fx ⟨k, hlt⟩) (fy ⟨k, hlt⟩))))
      rw [h1, h2]

      -- Apply IH to the recursive call
      -- First, show the new accumulators maintain the inequality
      have new_acc_ineq : tensorFoldlSpec (· + ·) acc_relu
                            (mulSpec (subSpec (mapSpec Math.reluSpec (fx ⟨k, hlt⟩))
                                               (mapSpec Math.reluSpec (fy ⟨k, hlt⟩)))
                                     (subSpec (mapSpec Math.reluSpec (fx ⟨k, hlt⟩))
                                              (mapSpec Math.reluSpec (fy ⟨k, hlt⟩)))) ≤
                          tensorFoldlSpec (· + ·) acc_orig
                            (mulSpec (subSpec (fx ⟨k, hlt⟩) (fy ⟨k, hlt⟩))
                                     (subSpec (fx ⟨k, hlt⟩) (fy ⟨k, hlt⟩))) := by
        rw [h1, h2]
        linarith [hacc, component_ineq]

      -- Apply IH to the recursive call with the updated accumulators
      -- First, we need to convert new_acc_ineq to the right form
      have new_acc_ineq' : (acc_relu + sumSpec (mulSpec (subSpec (mapSpec Math.reluSpec (fx ⟨k,
        hlt⟩))
                                                                  (mapSpec Math.reluSpec (fy ⟨k,
                                                                    hlt⟩)))
                                               (subSpec (mapSpec Math.reluSpec (fx ⟨k, hlt⟩))
                                                        (mapSpec Math.reluSpec (fy ⟨k, hlt⟩))))) ≤
                          (acc_orig + sumSpec (mulSpec (subSpec (fx ⟨k, hlt⟩) (fy ⟨k, hlt⟩))
                                                       (subSpec (fx ⟨k, hlt⟩) (fy ⟨k, hlt⟩)))) :=
                                                         by
        linarith [hacc, component_ineq]
      exact ih_fold (k + 1) _ _ k_plus_one_le new_acc_ineq' h_next

/--
Vector-shaped ReLU is 1-Lipschitz in L2.

This theorem is just the vector specialization of `relu_lipschitz_general`, but it is convenient
for callers working with ordinary `.dim n .scalar` activations.
-/
theorem relu_vector_lipschitz {n : Nat} (x y : Tensor ℝ (.dim n .scalar)) :
  tensorL2Dist (reluSpec x) (reluSpec y) ≤ tensorL2Dist x y := by
  simpa using (relu_lipschitz_general (s := .dim n .scalar) x y)

-- ====================================================================
-- LINEAR OPERATOR NORM BOUNDS
-- ====================================================================

/--
Tensor subtraction can be rewritten as addition of a `-1` scale.

This is a small algebraic normal form used by linear-operator proofs, where it is often easier to
reuse additive and scaling lemmas than reason about `subSpec` directly.
-/
theorem sub_spec_eq_add_scale_neg_one {s : Shape} (a b : Tensor ℝ s) :
  subSpec a b = addSpec a (scaleSpec b (-1 : ℝ)) := by
  induction s with
  | scalar =>
    cases a with
    | scalar x =>
      cases b with
      | scalar y =>
        simp [subSpec, addSpec, scaleSpec, map2Spec, mapSpec]
        ring
  | dim n s ih =>
    cases a with
    | dim fa =>
      cases b with
      | dim fb =>
        simp [subSpec, addSpec, scaleSpec, map2Spec, mapSpec]
        funext i
        simpa [subSpec, addSpec, scaleSpec, map2Spec, mapSpec] using ih (fa i) (fb i)

/-- Subtracting the zero tensor on the right leaves the tensor unchanged. -/
theorem sub_spec_zero_right {s : Shape} (t : Tensor ℝ s) :
  subSpec t (fill (0 : ℝ) s) = t := by
  induction s with
  | scalar =>
    cases t with
    | scalar x =>
      simp [subSpec, map2Spec, fill]
  | dim n s ih =>
    cases t with
    | dim f =>
      simp [subSpec, map2Spec, fill]
      funext i
      exact ih (f i)

/--
Matrix-vector multiplication sends the zero vector to the zero vector.

The proof follows the spec definition: each output coordinate is a fold over scalar products, and
every scalar product contains a zero input coordinate.
-/
theorem mat_vec_mul_spec_zero {m n : Nat} (W : Tensor ℝ (.dim m (.dim n .scalar))) :
  matVecMulSpec W (fill (0 : ℝ) (.dim n .scalar)) = fill (0 : ℝ) (.dim m .scalar) := by
  classical
  cases W with
  | dim rowsA =>
    -- Both sides are vectors; prove pointwise.
    apply congrArg Tensor.dim
    funext i
    cases hrow : rowsA i with
    | dim colsA =>
      -- Reduce to a scalar list fold.
      simp [fill]
      -- The values vector is identically zero, so each step adds `ak * 0 = 0`.
      have hfold :
          (List.finRange n).foldl
              (fun (s : ℝ) (k : Fin n) =>
                Spec.matMulSpec.match_1
                  (motive := fun _ _ => ℝ)
                  (colsA k) (Tensor.scalar (0 : ℝ))
                  (fun ak vk => s + ak * vk))
              0
            =
            0 := by
        -- Each step is `s ↦ s + ak * 0 = s`, so the fold returns the initial accumulator.
        let f : ℝ → Fin n → ℝ := fun s k =>
          Spec.matMulSpec.match_1
            (motive := fun _ _ => ℝ)
            (colsA k) (Tensor.scalar (0 : ℝ))
            (fun ak vk => s + ak * vk)
        have hf : ∀ s k, f s k = s := by
          intro s k
          cases hcol : colsA k with
          | scalar ak =>
            simp [f, hcol]
        -- Replace the fold function with `f`, then it is the identity on the accumulator.
        change (List.finRange n).foldl f 0 = 0
        induction (List.finRange n) with
        | nil =>
          simp [List.foldl]
        | cons hd tl ih =>
          simp [List.foldl, hf, ih]
      -- Convert the scalar-tensor fold in `mat_vec_mul_spec` to an ℝ fold, then apply `hfold`.
      have hscalar :
          (List.finRange n).foldl
              (fun (acc : Tensor ℝ Shape.scalar) (k : Fin n) =>
                Spec.matVecMulSpec.match_1
                  (motive := fun _ _ _ => Tensor ℝ Shape.scalar)
                  acc (colsA k) (Tensor.scalar (0 : ℝ))
                  (fun s ak vk => Tensor.scalar (s + ak * vk)))
              (Tensor.scalar 0)
            =
            Tensor.scalar
              ((List.finRange n).foldl
                (fun (s : ℝ) (k : Fin n) =>
                  Spec.matMulSpec.match_1
                    (motive := fun _ _ => ℝ)
                    (colsA k) (Tensor.scalar (0 : ℝ))
                    (fun ak vk => s + ak * vk))
                0) := by
        exact
          (Spec.foldl_matvec_scalar (l := List.finRange n) (a := 0) (cols := colsA)
            (vals := fun _ => Tensor.scalar (0 : ℝ)))
      -- Finish by reducing to the ℝ fold value.
      rw [hscalar]
      simp [hfold]

/--
Upper bound on a matrix operator norm.

We use the Frobenius-norm-style bound:
`matrix_op_norm W = √(∑ i, ‖row_i‖₂²)`,
which satisfies `‖W x‖₂ ≤ matrix_op_norm W * ‖x‖₂`.
-/
noncomputable def matrixOpNorm {m n : Nat} (W : Tensor ℝ (.dim m (.dim n .scalar))) : ℝ :=
  Real.sqrt (∑ i : Fin m, tensorNormSquared (get W i))

/--
Compatibility between two row/column access views:

`toVec (get W i) j` and `get2 W i j` name the same scalar entry of a matrix tensor.
-/
private lemma toVec_get_eq_get2 {m n : Nat}
    (W : Tensor ℝ (.dim m (.dim n .scalar))) (i : Fin m) (j : Fin n) :
    toVec (get W i) j = get2 W i j := by
  classical
  cases W with
  | dim rows =>
    cases hrow : rows i with
    | dim cols =>
      cases hcol : cols j with
      | scalar v =>
        simp [Spec.toVec, Spec.get, Spec.getAtSpec, Spec.get2, hrow, hcol]

/--
Each coordinate of `matVecMulSpec W x` is the dot product of the corresponding matrix row with
`x`.

This is the coordinate bridge used by the Frobenius/operator-norm bound below.
-/
private lemma mat_vec_coord_eq_dot_row {m n : Nat}
    (W : Tensor ℝ (.dim m (.dim n .scalar)))
    (x : Tensor ℝ (.dim n .scalar)) (i : Fin m) :
    toVec (matVecMulSpec W x) i = dot (get W i) x := by
  classical
  -- Expand both sides as `Finset.univ` sums and match terms.
  rw [toVec_mat_vec_mul_spec (A := W) (v := x) (i := i)]
  rw [dot_vec_eq_sum (a := get W i) (b := x)]
  refine Finset.sum_congr rfl ?_
  intro j _
  simp [toVec_get_eq_get2 (W := W) (i := i) (j := j)]

/--
Frobenius-based operator norm bound: `‖W x‖₂ ≤ matrix_op_norm W * ‖x‖₂`.
-/
theorem matrix_spectral_norm_bound {m n : Nat}
  (W : Tensor ℝ (.dim m (.dim n .scalar)))
  (x : Tensor ℝ (.dim n .scalar)) :
  tensorL2Norm (matVecMulSpec W x) ≤ matrixOpNorm W * tensorL2Norm x := by
  classical
  -- Work with the squared form and then apply `Real.sqrt`.
  have hsum_nonneg : 0 ≤ ∑ i : Fin m, tensorNormSquared (get W i) := by
    have : 0 ≤ ∑ i ∈ (Finset.univ : Finset (Fin m)), tensorNormSquared (get W i) := by
      refine Finset.sum_nonneg ?_
      intro i _
      exact tensor_norm_squared_nonneg2 (t := get W i)
    simpa using this

  have hsquared :
      tensorNormSquared (matVecMulSpec W x) ≤
        (∑ i : Fin m, tensorNormSquared (get W i)) * tensorNormSquared x := by
    -- Expand `‖W x‖²` as a sum of squared coordinates.
    have hnormsq :
        tensorNormSquared (matVecMulSpec W x) =
          ∑ i : Fin m, (toVec (matVecMulSpec W x) i) * (toVec (matVecMulSpec W x) i) := by
      simpa [tensorNormSquared] using
        (dot_vec_eq_sum (a := matVecMulSpec W x) (b := matVecMulSpec W x))

    -- Bound each coordinate via Cauchy–Schwarz on the corresponding row.
    have hterm :
        ∀ i : Fin m,
          (toVec (matVecMulSpec W x) i) * (toVec (matVecMulSpec W x) i) ≤
            tensorNormSquared (get W i) * tensorNormSquared x := by
      intro i
      have hcoord : toVec (matVecMulSpec W x) i = dot (get W i) x :=
        mat_vec_coord_eq_dot_row (W := W) (x := x) (i := i)
      have cs :
          |dot (get W i) x| ≤ tensorL2Norm (get W i) * tensorL2Norm x :=
        tensor_cauchy_schwarz (x := get W i) (y := x)
      have cs2 :
          (dot (get W i) x) ^ 2 ≤ (tensorL2Norm (get W i) * tensorL2Norm x) ^ 2 := by
        -- Square both sides of `cs` via `mul_le_mul`.
        have hmul :
            |dot (get W i) x| * |dot (get W i) x| ≤
              (tensorL2Norm (get W i) * tensorL2Norm x) *
                (tensorL2Norm (get W i) * tensorL2Norm x) := by
          refine mul_le_mul cs cs (abs_nonneg (dot (get W i) x)) ?_
          exact mul_nonneg (tensor_l2_norm_nonneg (get W i)) (tensor_l2_norm_nonneg x)
        have hsq :
            (|dot (get W i) x|) ^ 2 ≤ (tensorL2Norm (get W i) * tensorL2Norm x) ^ 2 := by
          simpa [pow_two] using hmul
        simpa [sq_abs] using hsq

      have row_sq : (tensorL2Norm (get W i)) ^ 2 = tensorNormSquared (get W i) := by
        unfold tensorL2Norm
        simp [Real.sq_sqrt (tensor_norm_squared_nonneg2 (t := get W i))]
      have x_sq : (tensorL2Norm x) ^ 2 = tensorNormSquared x := by
        unfold tensorL2Norm
        simp [Real.sq_sqrt (tensor_norm_squared_nonneg2 (t := x))]
      have rhs_sq :
          (tensorL2Norm (get W i) * tensorL2Norm x) ^ 2 =
            tensorNormSquared (get W i) * tensorNormSquared x := by
        -- `(a*b)^2 = a^2 * b^2`, then unfold the squares of the norms.
        simp [mul_pow, row_sq, x_sq]

      have hsq :
          (toVec (matVecMulSpec W x) i) ^ 2 ≤
            tensorNormSquared (get W i) * tensorNormSquared x := by
        -- Replace the coordinate by the row dot-product and use `cs2`.
        simpa [hcoord, rhs_sq] using cs2

      -- Convert `a^2` back into `a*a`.
      simpa [pow_two] using hsq

    -- Sum the coordinate-wise bounds and factor out `‖x‖²`.
    have hsum_le :
        (∑ i : Fin m,
              (toVec (matVecMulSpec W x) i) * (toVec (matVecMulSpec W x) i)) ≤
          ∑ i : Fin m, tensorNormSquared (get W i) * tensorNormSquared x := by
      have :
          (∑ i ∈ (Finset.univ : Finset (Fin m)),
                (toVec (matVecMulSpec W x) i) * (toVec (matVecMulSpec W x) i)) ≤
            ∑ i ∈ (Finset.univ : Finset (Fin m)), tensorNormSquared (get W i) *
              tensorNormSquared x := by
        refine Finset.sum_le_sum ?_
        intro i _
        exact hterm i
      simpa using this

    have hfactor :
        (∑ i : Fin m, tensorNormSquared (get W i) * tensorNormSquared x) =
          (∑ i : Fin m, tensorNormSquared (get W i)) * tensorNormSquared x := by
      have h :=
        (Finset.sum_mul (s := (Finset.univ : Finset (Fin m)))
          (f := fun i : Fin m => tensorNormSquared (get W i)) (a := tensorNormSquared x))
      simpa using h.symm

    -- Put everything together.
    calc
      tensorNormSquared (matVecMulSpec W x)
          = ∑ i : Fin m,
              (toVec (matVecMulSpec W x) i) * (toVec (matVecMulSpec W x) i) := hnormsq
      _ ≤ ∑ i : Fin m, tensorNormSquared (get W i) * tensorNormSquared x := hsum_le
      _ = (∑ i : Fin m, tensorNormSquared (get W i)) * tensorNormSquared x := hfactor

  -- Take square roots and rewrite the RHS using `Real.sqrt_mul`.
  unfold matrixOpNorm tensorL2Norm
  have hsqrt := Real.sqrt_le_sqrt hsquared
  -- Rewrite `√(A * B)` as `√A * √B` with `A = ∑ i, ‖row_i‖² ≥ 0`.
  rw [Real.sqrt_mul hsum_nonneg (tensorNormSquared x)] at hsqrt
  simpa using hsqrt

/--
Linear transformation preserves L2 norm bounds.
Fundamental theorem for neural network stability analysis.
-/
theorem linear_op_norm_bound {m n : Nat}
  (W : Tensor ℝ (.dim m (.dim n .scalar)))
  (x y : Tensor ℝ (.dim n .scalar)) :
  tensorL2Dist (matVecMulSpec W x) (matVecMulSpec W y) ≤
  matrixOpNorm W * tensorL2Dist x y := by
  have h_linear : matVecMulSpec W (subSpec x y) =
    subSpec (matVecMulSpec W x) (matVecMulSpec W y) := by
    -- Express subtraction as addition with scaling, then use `mat_vec_add`/`mat_vec_scale`.
    rw [sub_spec_eq_add_scale_neg_one (a := x) (b := y)]
    rw [Spec.mat_vec_add]
    rw [Spec.mat_vec_scale]
    -- Rewrite the RHS subtraction similarly.
    simp [sub_spec_eq_add_scale_neg_one]

  unfold tensorL2Dist
  rw [← h_linear]
  exact matrix_spectral_norm_bound W (subSpec x y)

-- ====================================================================
-- COMPOSITION THEOREMS
-- ====================================================================

/--
Composition of Lipschitz functions preserves Lipschitz property.
Essential for analyzing deep neural networks.
-/
theorem lipschitz_composition {s t u : Shape}
  (f : Tensor ℝ s → Tensor ℝ t) (g : Tensor ℝ t → Tensor ℝ u)
  (Lf Lg : ℝ)
  (hf : ∀ x y, tensorL2Dist (f x) (f y) ≤ Lf * tensorL2Dist x y)
  (hg : ∀ x y, tensorL2Dist (g x) (g y) ≤ Lg * tensorL2Dist x y)
  (hLg : 0 ≤ Lg)
  (x y : Tensor ℝ s) :
  tensorL2Dist (g (f x)) (g (f y)) ≤ (Lg * Lf) * tensorL2Dist x y := by
  calc tensorL2Dist (g (f x)) (g (f y))
    ≤ Lg * tensorL2Dist (f x) (f y)     := hg (f x) (f y)
    _ ≤ Lg * (Lf * tensorL2Dist x y)    := by
      apply mul_le_mul_of_nonneg_left
      exact hf x y
      exact hLg
    _ = (Lg * Lf) * tensorL2Dist x y    := by ring

/--
ReLU + Linear composition Lipschitz bound.
Practical theorem for single neural network layer analysis.
-/
theorem relu_linear_lipschitz {m n : Nat}
  (W : Tensor ℝ (.dim m (.dim n .scalar)))
  (x y : Tensor ℝ (.dim n .scalar)) :
  tensorL2Dist (reluSpec (matVecMulSpec W x)) (reluSpec (matVecMulSpec W y)) ≤
  matrixOpNorm W * tensorL2Dist x y := by
  calc tensorL2Dist (reluSpec (matVecMulSpec W x)) (reluSpec (matVecMulSpec W y))
    ≤ tensorL2Dist (matVecMulSpec W x) (matVecMulSpec W y)  :=
      relu_lipschitz_general (matVecMulSpec W x) (matVecMulSpec W y)
    _ ≤ matrixOpNorm W * tensorL2Dist x y                        :=
      linear_op_norm_bound W x y

-- ====================================================================
-- SPECIALIZED ACTIVATION FUNCTION ANALYSIS
-- ====================================================================

end Proofs
