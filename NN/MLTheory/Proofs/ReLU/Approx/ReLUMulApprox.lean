/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Data.Fin.Tuple.Basic
public import NN.MLTheory.Proofs.ReLU.Bridge.ReLUMlpBridge
import Mathlib.Tactic.Ring

/-!
# Approximating multiplication with a 2-layer ReLU MLP (2D box)

This file gives a constructive, fully proved approximation result:
on `[-M,M]²`, the function `(x₀,x₁) ↦ x₀ * x₁` can be uniformly approximated by a
  single-hidden-layer
ReLU MLP on `Tensor ℝ (.dim 2 .scalar)`.
-/

@[expose] public section


namespace NN.MLTheory.Proofs.ReLUMulApprox

open _root_.Spec
open Examples

open NN.MLTheory.Proofs.UniversalApproximation
open NN.MLTheory.Proofs.ReLUMlpBridge

/-- `TensorVec` specialized to the 2D (rank-2) tensor-vector shape. -/
abbrev TensorVec2 : Type := TensorVec 2

/-- First coordinate projection `x ↦ x0` for `TensorVec2`. -/
noncomputable def x0 (x : TensorVec2) : ℝ := toVec x ⟨0, by decide⟩

/-- Second coordinate projection `x ↦ x1` for `TensorVec2`. -/
noncomputable def x1 (x : TensorVec2) : ℝ := toVec x ⟨1, by decide⟩

/-- The closed box domain `[-M,M] × [-M,M]` inside `TensorVec2`. -/
noncomputable def box (M : ℝ) : Set TensorVec2 :=
  fun x => x0 x ∈ Set.Icc (-M) M ∧ x1 x ∈ Set.Icc (-M) M

/-- The target multiplication map `x ↦ x0 * x1`. -/
noncomputable def mulFun (x : TensorVec2) : ℝ := x0 x * x1 x

/-- Ridge direction with `dot wPlus x = x0 + x1`. -/
noncomputable def wPlus : Fin 2 → ℝ := fun _ => 1

/-- Ridge direction with `dot wMinus x = x0 - x1`. -/
noncomputable def wMinus : Fin 2 → ℝ := fun i => if i.1 = 0 then 1 else (-1 : ℝ)

/-- Evaluate the ridge `wPlus`: `dot wPlus x = x0 + x1`. -/
lemma dot_wPlus (x : TensorVec2) : dot wPlus x = x0 x + x1 x := by
  classical
  -- Expand the `Fin 2` sum explicitly.
  simp [dot, wPlus, x0, x1, Fin.sum_univ_two]

/-- Evaluate the ridge `wMinus`: `dot wMinus x = x0 - x1`. -/
lemma dot_wMinus (x : TensorVec2) : dot wMinus x = x0 x - x1 x := by
  classical
  simp [dot, wMinus, x0, x1, Fin.sum_univ_two, sub_eq_add_neg]

/-- Algebraic identity expressing multiplication via a difference of squares. -/
lemma mul_identity (x y : ℝ) : x * y = ((x + y) * (x + y) - (x - y) * (x - y)) / 4 := by
  ring

/-- Unpack the defining bounds of membership in `box M`. -/
lemma box_bounds {M : ℝ} (_hM : 0 ≤ M) {x : TensorVec2} (hx : x ∈ box M) :
    x0 x ∈ Set.Icc (-M) M ∧ x1 x ∈ Set.Icc (-M) M := hx

/-- If `x ∈ box M`, then the ridge input `x0 + x1` lies in `[-2M, 2M]`. -/
lemma sum_mem_Icc {M : ℝ} (_hM : 0 ≤ M) {x : TensorVec2} (hx : x ∈ box M) :
    dot wPlus x ∈ Set.Icc (-2*M) (2*M) := by
  have hx0 := hx.1
  have hx1 := hx.2
  have hx0l : -M ≤ x0 x := hx0.1
  have hx0u : x0 x ≤ M := hx0.2
  have hx1l : -M ≤ x1 x := hx1.1
  have hx1u : x1 x ≤ M := hx1.2
  -- bounds on sum
  have hl : -(2*M) ≤ x0 x + x1 x := by linarith
  have hu : x0 x + x1 x ≤ 2*M := by linarith
  simpa [dot_wPlus] using And.intro hl hu

/-- If `x ∈ box M`, then the ridge input `x0 - x1` lies in `[-2M, 2M]`. -/
lemma diff_mem_Icc {M : ℝ} (_hM : 0 ≤ M) {x : TensorVec2} (hx : x ∈ box M) :
    dot wMinus x ∈ Set.Icc (-2*M) (2*M) := by
  have hx0 := hx.1
  have hx1 := hx.2
  have hx0l : -M ≤ x0 x := hx0.1
  have hx0u : x0 x ≤ M := hx0.2
  have hx1l : -M ≤ x1 x := hx1.1
  have hx1u : x1 x ≤ M := hx1.2
  have hl : -(2*M) ≤ x0 x - x1 x := by linarith
  have hu : x0 x - x1 x ≤ 2*M := by linarith
  simpa [dot_wMinus] using And.intro hl hu

/-- Lipschitz bound for `square` on `[-R,R]`: `|x^2 - y^2| ≤ (2R) * |x - y|`. -/
lemma square_lipschitz_Icc {R : ℝ} (_hR : 0 ≤ R) :
    ∀ x ∈ Set.Icc (-R) R, ∀ y ∈ Set.Icc (-R) R, |(x*x) - (y*y)| ≤ (2*R) * |x - y| := by
  intro x hx y hy
  have hxabs : |x| ≤ R := by
    have hx' : -R ≤ x ∧ x ≤ R := by simpa [Set.Icc] using hx
    exact (abs_le).2 hx'
  have hyabs : |y| ≤ R := by
    have hy' : -R ≤ y ∧ y ≤ R := by simpa [Set.Icc] using hy
    exact (abs_le).2 hy'
  -- Factor and bound by `|x-y| * |x+y|` and then `|x+y| ≤ |x|+|y| ≤ 2R`.
  have hfactor : x*x - y*y = (x - y) * (x + y) := by ring
  calc
    |x*x - y*y| = |(x - y) * (x + y)| := by simp [hfactor]
    _ = |x - y| * |x + y| := by simp [abs_mul]
    _ ≤ |x - y| * (|x| + |y|) := by
      have : |x + y| ≤ |x| + |y| := by simpa using abs_add_le x y
      exact mul_le_mul_of_nonneg_left this (abs_nonneg (x - y))
    _ ≤ |x - y| * (R + R) := by
      have hsum : |x| + |y| ≤ R + R := add_le_add hxabs hyabs
      exact mul_le_mul_of_nonneg_left hsum (abs_nonneg (x - y))
    _ = (2*R) * |x - y| := by ring_nf

-- ---------------------------------------------------------------------------
-- Tensor helpers: concatenating hidden units
-- ---------------------------------------------------------------------------

/--
Concatenate tensors along the leading dimension.

In this file, this is used to append the hidden-unit vectors of two subnetworks.
-/
noncomputable def appendDim {α : Type} {m n : Nat} {s : Shape}
    (a : Tensor α (.dim m s)) (b : Tensor α (.dim n s)) : Tensor α (.dim (m+n) s) :=
  match a, b with
  | .dim fa, .dim fb => .dim (Fin.append fa fb)

/-- Append two first-layer linear specs by appending their weight and bias tensors. -/
noncomputable def appendLinearSpec
    {inDim m n : Nat} (a : LinearSpec ℝ inDim m) (b : LinearSpec ℝ inDim n) :
    LinearSpec ℝ inDim (m+n) :=
  { weights := appendDim a.weights b.weights
    bias := appendDim a.bias b.bias }

/-- Extract the `j`-th entry from a `1 × n` tensor interpreted as a row matrix. -/
noncomputable def mat1Get {n : Nat} (A : Tensor ℝ (.dim 1 (.dim n .scalar))) (j : Fin n) : ℝ :=
  match A with
  | .dim rows =>
    match rows ⟨0, by decide⟩ with
    | .dim cols => (cols j).toScalar

/-- `mat1_get` agrees with the `matrixMN` constructor. -/
lemma mat1_get_matrixMN {n : Nat} (f : Fin 1 → Fin n → ℝ) (j : Fin n) :
    mat1Get (matrixMN 1 n (fun i j => f i j)) j = f 0 j := by
  simp [mat1Get, matrixMN, Tensor.toScalar]

/--
Combine two scalar-output linear specs into one scalar-output spec on an appended hidden layer.

If the appended hidden vector is `[z_a; z_b]`, the resulting output layer computes
`γ + α*out_a(z_a) + β*out_b(z_b)`.
-/
noncomputable def combineOutput
    {m n : Nat} (α β γ : ℝ) (a : LinearSpec ℝ m 1) (b : LinearSpec ℝ n 1) :
    LinearSpec ℝ (m+n) 1 :=
  let c : Fin (m+n) → ℝ :=
    Fin.addCases (fun j : Fin m => α * mat1Get a.weights j) (fun j : Fin n => β * mat1Get
      b.weights j)
  { weights := matrixMN 1 (m+n) (fun _ j => c j)
    bias := vectorN 1 (fun _ => γ + α * extractScalarOutput a.bias + β * extractScalarOutput
      b.bias) }

/-- Reading the left component from an appended hidden vector. -/
lemma vec_get_append_left {m n : Nat} (a : Tensor ℝ (.dim m .scalar)) (b : Tensor ℝ (.dim n
  .scalar)) (i : Fin m) :
    vecGet (appendDim a b) (Fin.castAdd n i) = vecGet a i := by
  cases a with
  | dim fa =>
    cases b with
    | dim fb =>
      simp [appendDim, vecGet, Fin.append]

/-- Reading the right component from an appended hidden vector. -/
lemma vec_get_append_right {m n : Nat} (a : Tensor ℝ (.dim m .scalar)) (b : Tensor ℝ (.dim n
  .scalar)) (i : Fin n) :
    vecGet (appendDim a b) (Fin.natAdd m i) = vecGet b i := by
  cases a with
  | dim fa =>
    cases b with
    | dim fb =>
      simp [appendDim, vecGet, Fin.append]

/-- Pointwise behavior of the ReLU activation on tensor-vectors. -/
lemma vec_get_relu {n : Nat} (z : Tensor ℝ (.dim n .scalar)) (i : Fin n) :
    vecGet (Activation.reluSpec (α := ℝ) (s := .dim n .scalar) z) i = relu (vecGet z i) := by
  cases z with
  | dim f =>
    cases hfi : f i with
    | scalar r =>
      simp [Activation.reluSpec, Spec.Tensor.mapSpec, vecGet, relu, Activation.Math.reluSpec,
        hfi, Tensor.toScalar]

/-- Matrix-vector multiplication for a `1 × n` matrix produces a single scalar coordinate. -/
lemma mat_vec_mul_spec_oneRow {n : Nat} (A : Tensor ℝ (.dim 1 (.dim n .scalar))) (v : Tensor ℝ (.dim
  n .scalar)) :
    Spec.matVecMulSpec A v =
      Tensor.dim (fun _ : Fin 1 => Tensor.scalar (∑ j : Fin n, mat1Get A j * vecGet v j)) := by
  classical
  -- Put `A` and `v` into the canonical `matrixMN` / `Tensor.dim (Tensor.scalar ·)` forms,
  -- then use the general matrix×vector lemma from `relu_mlp_bridge.lean`.
  let c : Fin 1 → Fin n → ℝ := fun _ j => mat1Get A j
  let vfun : Fin n → ℝ := fun j => vecGet v j
  have hA : A = matrixMN 1 n c := by
    cases A with
    | dim rows =>
      apply congrArg Tensor.dim
      funext i
      fin_cases i
      -- Reduce to pointwise equality of the unique row.
      cases hrow : rows ⟨0, by decide⟩ with
      | dim cols =>
        have hrow0 : rows 0 = Tensor.dim cols := by
          simpa using hrow
        -- Now show the row entries match `Tensor.scalar (mat1_get ...)`.
        -- `mat1_get` unfolds to `Tensor.toScalar (cols j)`.
        apply congrArg Tensor.dim
        funext j
        cases hcol : cols j with
        | scalar r =>
          simp [c, mat1Get, hrow0, Tensor.toScalar, hcol]
  have hv : v = Tensor.dim (fun j => Tensor.scalar (vfun j)) := by
    cases v with
    | dim valuesV =>
      apply congrArg Tensor.dim
      funext j
      cases hvj : valuesV j with
      | scalar r =>
        simp [vfun, vecGet, Tensor.toScalar, hvj]
  -- Rewrite and apply the general lemma.
  rw [hA, hv]
  simpa [c, vfun] using
    (mat_vec_mul_spec_matrixMN_vector (m := 1) (n := n) (c := c) (v := vfun))

/--
Expand `mlp_eval_nd` into “bias + sum over hidden units” form.

This is the main normalization lemma used to prove that `appendLinearSpec` together with
`combineOutput` implements affine combinations of subnetworks.
-/
lemma mlp_eval_nd_eq_bias_sum
    {inDim hidDim : Nat} (l1 : LinearSpec ℝ inDim hidDim) (l2 : LinearSpec ℝ hidDim 1)
    (x : Tensor ℝ (.dim inDim .scalar)) :
    mlpEvalNd (n := inDim) (hidDim := hidDim) l1 l2 x =
      extractScalarOutput l2.bias
        + ∑ j : Fin hidDim, (mat1Get l2.weights j) * relu (vecGet (Spec.linearSpec (α := ℝ) l1 x)
          j) := by
  classical
  unfold mlpEvalNd
  rw [mlp_forward_eq_linear_relu_linear (n := inDim) (hidDim := hidDim) (l1 := l1) (l2 := l2) (x :=
    x)]
  dsimp
  have hmv :
      Spec.matVecMulSpec l2.weights
          (Activation.reluSpec (α := ℝ) (s := .dim hidDim .scalar) (Spec.linearSpec (α := ℝ) l1
            x)) =
        Tensor.dim (fun _ : Fin 1 =>
          Tensor.scalar (∑ j : Fin hidDim,
            mat1Get l2.weights j *
              vecGet (Activation.reluSpec (α := ℝ) (s := .dim hidDim .scalar) (Spec.linearSpec (α
                := ℝ) l1 x)) j)) := by
    simpa using mat_vec_mul_spec_oneRow (A := l2.weights)
      (v := Activation.reluSpec (α := ℝ) (s := .dim hidDim .scalar) (Spec.linearSpec (α := ℝ) l1
        x))
  cases hbias : l2.bias with
  | dim fb =>
    -- `Spec.linear_spec` unfolds to a `Tensor.map2_spec` term; rewrite `hmv` to match that form.
    have hmv' : Spec.matVecMulSpec l2.weights
        (Activation.reluSpec (α := ℝ) (s := .dim hidDim .scalar)
          (Tensor.map2Spec (fun x1 x2 ↦ x1 + x2) (Spec.matVecMulSpec l1.weights x) l1.bias))
        =
        Tensor.dim (fun _ : Fin 1 =>
          Tensor.scalar (∑ j : Fin hidDim,
            mat1Get l2.weights j *
              vecGet
                (Activation.reluSpec (α := ℝ) (s := .dim hidDim .scalar)
                  (Tensor.map2Spec (fun x1 x2 ↦ x1 + x2) (Spec.matVecMulSpec l1.weights x)
                    l1.bias))
                j)) := by
      simpa [Spec.linearSpec] using hmv
    -- Use the mat-vec sum form, then compute the single coordinate.
    cases hfb0 : fb 0 with
    | scalar b0 =>
      simp [Spec.linearSpec, hmv', Spec.Tensor.addSpec, Spec.Tensor.map2Spec,
        extractScalarOutput,
        vec_get_relu, hbias, hfb0, add_comm, Tensor.toScalar]

/-- Selecting the left block of a linear spec appended via `appendLinearSpec`. -/
lemma vec_get_linear_spec_append_left
    {inDim m n : Nat} (l1a : LinearSpec ℝ inDim m) (l1b : LinearSpec ℝ inDim n)
    (x : Tensor ℝ (.dim inDim .scalar)) (i : Fin m) :
    vecGet (Spec.linearSpec (α := ℝ) (appendLinearSpec (inDim := inDim) l1a l1b) x) (Fin.castAdd n
      i)
      =
    vecGet (Spec.linearSpec (α := ℝ) l1a x) i := by
  classical
  -- Unfold down to pointwise evaluation; `Fin.castAdd` selects the left half of the appended layer.
  cases l1a with
  | mk wa ba =>
    cases l1b with
    | mk wb bb =>
      cases wa with
      | dim waF =>
        cases wb with
        | dim wbF =>
          cases ba with
          | dim baF =>
            cases bb with
            | dim bbF =>
              cases x with
              | dim xv =>
                simp [appendLinearSpec, appendDim, Spec.linearSpec, Spec.Tensor.addSpec,
                  Spec.Tensor.map2Spec,
                  Spec.matVecMulSpec, vecGet, Fin.append, Fin.addCases,
                  Tensor.toScalar]

/-- Selecting the right block of a linear spec appended via `appendLinearSpec`. -/
lemma vec_get_linear_spec_append_right
    {inDim m n : Nat} (l1a : LinearSpec ℝ inDim m) (l1b : LinearSpec ℝ inDim n)
    (x : Tensor ℝ (.dim inDim .scalar)) (i : Fin n) :
    vecGet (Spec.linearSpec (α := ℝ) (appendLinearSpec (inDim := inDim) l1a l1b) x) (Fin.natAdd m
      i)
      =
    vecGet (Spec.linearSpec (α := ℝ) l1b x) i := by
  classical
  cases l1a with
  | mk wa ba =>
    cases l1b with
    | mk wb bb =>
      cases wa with
      | dim waF =>
        cases wb with
        | dim wbF =>
          cases ba with
          | dim baF =>
            cases bb with
            | dim bbF =>
              cases x with
              | dim xv =>
                simp [appendLinearSpec, appendDim, Spec.linearSpec, Spec.Tensor.addSpec,
                  Spec.Tensor.map2Spec,
                  Spec.matVecMulSpec, vecGet, Fin.append, Fin.addCases,
                  Tensor.toScalar]

/--
Appending hidden units and wiring the output with `combineOutput` yields an affine combination.

Concretely, the combined network computes:
`γ + α * net_a(x) + β * net_b(x)`.
-/
theorem mlp_eval_append_linear
    {inDim m n : Nat} (l1a : LinearSpec ℝ inDim m) (l1b : LinearSpec ℝ inDim n)
    (l2a : LinearSpec ℝ m 1) (l2b : LinearSpec ℝ n 1)
    (α β γ : ℝ) (x : Tensor ℝ (.dim inDim .scalar)) :
    mlpEvalNd (n := inDim) (hidDim := m+n)
        (appendLinearSpec (inDim := inDim) l1a l1b)
        (combineOutput (m := m) (n := n) α β γ l2a l2b) x
      =
    γ + α * mlpEvalNd (n := inDim) (hidDim := m) l1a l2a x
      + β * mlpEvalNd (n := inDim) (hidDim := n) l1b l2b x := by
  classical
  -- This lemma is a “network algebra” fact:
  -- appending hidden units + picking an output layer via `combineOutput` implements an affine
  -- combination of two subnetworks’ scalar outputs.
  -- Expand all three evaluations into “bias + sum of hidden units”.
  rw [mlp_eval_nd_eq_bias_sum (l1 := appendLinearSpec (inDim := inDim) l1a l1b)
        (l2 := combineOutput (m := m) (n := n) α β γ l2a l2b) (x := x)]
  rw [mlp_eval_nd_eq_bias_sum (l1 := l1a) (l2 := l2a) (x := x)]
  rw [mlp_eval_nd_eq_bias_sum (l1 := l1b) (l2 := l2b) (x := x)]
  -- Split the combined sum over `Fin (m+n)` into left/right parts.
  -- The combined weights are `Fin.addCases` and the combined hidden pre-activations come from
  -- `appendLinearSpec`.
  have hsplit :
      (∑ j : Fin (m + n),
          mat1Get (combineOutput (m := m) (n := n) α β γ l2a l2b).weights j *
            relu (vecGet (Spec.linearSpec (α := ℝ) (appendLinearSpec (inDim := inDim) l1a l1b) x)
              j))
        =
      (∑ j : Fin m,
          (α * mat1Get l2a.weights j) *
            relu (vecGet (Spec.linearSpec (α := ℝ) l1a x) j))
      +
      (∑ j : Fin n,
          (β * mat1Get l2b.weights j) *
            relu (vecGet (Spec.linearSpec (α := ℝ) l1b x) j)) := by
    classical
    -- Use `Fin.sum_univ_add` and then simplify the `Fin.addCases` selectors.
    have hsum :=
      (Fin.sum_univ_add (a := m) (b := n)
        (f := fun j : Fin (m+n) =>
          mat1Get (combineOutput (m := m) (n := n) α β γ l2a l2b).weights j *
            relu (vecGet (Spec.linearSpec (α := ℝ) (appendLinearSpec (inDim := inDim) l1a l1b) x)
              j)))
    -- Rewrite the `castAdd` / `natAdd` branches using the selector lemmas above.
    -- `combineOutput` uses `Fin.addCases` in its weights.
    simpa [combineOutput, mat1_get_matrixMN, relu,
      vec_get_linear_spec_append_left (l1a := l1a) (l1b := l1b) (x := x),
      vec_get_linear_spec_append_right (l1a := l1a) (l1b := l1b) (x := x),
      Fin.addCases_left, Fin.addCases_right, mul_assoc, mul_left_comm, mul_comm] using hsum
  -- Factor out the scalars `α`/`β` from the two sums.
  let sumA : ℝ :=
    ∑ j : Fin m,
      mat1Get l2a.weights j * relu (vecGet (Spec.linearSpec (α := ℝ) l1a x) j)
  let sumB : ℝ :=
    ∑ j : Fin n,
      mat1Get l2b.weights j * relu (vecGet (Spec.linearSpec (α := ℝ) l1b x) j)
  have hsumA :
      (∑ j : Fin m,
          (α * mat1Get l2a.weights j) * relu (vecGet (Spec.linearSpec (α := ℝ) l1a x) j))
        = α * sumA := by
    classical
    -- `∑ (α * t_j) = α * ∑ t_j` over `Fin m`.
    simpa [sumA, mul_assoc, mul_left_comm, mul_comm] using
      (Finset.mul_sum α (s := (Finset.univ : Finset (Fin m)))
          (f := fun j : Fin m =>
            mat1Get l2a.weights j * relu (vecGet (Spec.linearSpec (α := ℝ) l1a x) j))).symm
  have hsumB :
      (∑ j : Fin n,
          (β * mat1Get l2b.weights j) * relu (vecGet (Spec.linearSpec (α := ℝ) l1b x) j))
        = β * sumB := by
    classical
    simpa [sumB, mul_assoc, mul_left_comm, mul_comm] using
      (Finset.mul_sum β (s := (Finset.univ : Finset (Fin n)))
          (f := fun j : Fin n =>
            mat1Get l2b.weights j * relu (vecGet (Spec.linearSpec (α := ℝ) l1b x) j))).symm
  -- Finish by normalizing the combined output bias and collecting the two hidden sums.
  have hbias :
      extractScalarOutput (combineOutput (m := m) (n := n) α β γ l2a l2b).bias
        = γ + α * extractScalarOutput l2a.bias + β * extractScalarOutput l2b.bias := by
    simp [combineOutput, extractScalarOutput, vectorN, Tensor.toScalar]
  rw [hbias, hsplit, hsumA, hsumB]
  simp [sumA, sumB, mul_add, add_assoc, add_left_comm]

-- ---------------------------------------------------------------------------
-- Main theorem: multiplication approximation on `[-M,M]²`
-- ---------------------------------------------------------------------------

/--
Uniform approximation of multiplication on `[-M,M]^2` by a single-hidden-layer ReLU MLP.

The construction follows the classical reduction
`x*y = ((x+y)^2 - (x-y)^2) / 4`, combined with a 1D ReLU approximator for `square` on `[-2M,2M]`
that is lifted along the ridge directions `wPlus` and `wMinus`.
-/
theorem relu_mul_universal_approximation_box
    {M : ℝ} (hM : 0 < M) :
    ∀ ε > 0, ∃ (hidDim : ℕ) (l1 : LinearSpec ℝ 2 hidDim) (l2 : LinearSpec ℝ hidDim 1),
      ∀ x ∈ box M, |mulFun x - mlpEvalNd (n := 2) (hidDim := hidDim) l1 l2 x| < ε := by
  intro ε hε
  have hM0 : 0 ≤ M := le_of_lt hM
  -- Step 1: approximate `square` on `[-2M,2M]` with error `δ = 2ε`.
  let δ : ℝ := 2*ε
  have hδ : 0 < δ := by nlinarith
  have h_ab : (-2*M) < (2*M) := by nlinarith
  have hL : 0 < (4*M) := by nlinarith
  have h_lip :
      ∀ x ∈ Set.Icc (-2*M) (2*M), ∀ y ∈ Set.Icc (-2*M) (2*M),
        |(x*x) - (y*y)| ≤ (4*M) * |x - y| := by
    intro x hx y hy
    -- Apply `square_lipschitz_Icc` with `R = 2M`.
    have h :=
      square_lipschitz_Icc (R := 2*M) (by nlinarith [hM0]) x (by simpa using hx) y (by simpa using
        hy)
    -- `2*(2M) = 4M`
    convert h using 1; ring
  rcases relu_universal_approximation_Icc (f := fun u => u*u) (a := -2*M) (b := 2*M) (L := 4*M)
      h_ab hL h_lip δ hδ with ⟨hidSq, l1Sq, l2Sq, hSq⟩
  -- Step 2: lift to `u = x0+x1` and `u = x0-x1`.
  let l1Plus : LinearSpec ℝ 2 hidSq := liftLayer1From1d (n := 2) l1Sq wPlus 0
  let l1Minus : LinearSpec ℝ 2 hidSq := liftLayer1From1d (n := 2) l1Sq wMinus 0
  -- Step 3: combine the two lifted square nets to form a product approximator.
  let l1Prod : LinearSpec ℝ 2 (hidSq + hidSq) := appendLinearSpec (inDim := 2) l1Plus l1Minus
  let l2Prod : LinearSpec ℝ (hidSq + hidSq) 1 :=
    combineOutput (m := hidSq) (n := hidSq) (α := (1/4 : ℝ)) (β := (-1/4 : ℝ)) (γ := 0) l2Sq l2Sq
  refine ⟨hidSq + hidSq, l1Prod, l2Prod, ?_⟩
  intro x hx
  -- Abbreviate the two ridge inputs.
  have hx_plus : dot wPlus x ∈ Set.Icc (-2*M) (2*M) := sum_mem_Icc (M := M) hM0 hx
  have hx_minus : dot wMinus x ∈ Set.Icc (-2*M) (2*M) := diff_mem_Icc (M := M) hM0 hx
  -- Use the lifted equality to rewrite the lifted nets as 1D evaluations.
  have hplus_eval :
      mlpEvalNd (n := 2) (hidDim := hidSq) l1Plus l2Sq x =
        mlpEval1d hidSq l1Sq l2Sq (dot wPlus x) := by
    simpa [l1Plus, add_comm] using
      (mlp_eval_lift_from_1d (n := 2) (hidDim := hidSq) l1Sq l2Sq wPlus 0 x)
  have hminus_eval :
      mlpEvalNd (n := 2) (hidDim := hidSq) l1Minus l2Sq x =
        mlpEval1d hidSq l1Sq l2Sq (dot wMinus x) := by
    simpa [l1Minus, add_comm] using
      (mlp_eval_lift_from_1d (n := 2) (hidDim := hidSq) l1Sq l2Sq wMinus 0 x)
  -- Evaluate the combined network as a linear combination of the two lifted nets.
  have hcomb :
      mlpEvalNd (n := 2) (hidDim := hidSq + hidSq) l1Prod l2Prod x
        =
      (1/4 : ℝ) * mlpEvalNd (n := 2) (hidDim := hidSq) l1Plus l2Sq x
        + (-1/4 : ℝ) * mlpEvalNd (n := 2) (hidDim := hidSq) l1Minus l2Sq x := by
    -- `mlp_eval_append_linear` gives the general linear combination form.
    have := mlp_eval_append_linear (inDim := 2) (m := hidSq) (n := hidSq)
      (l1a := l1Plus) (l1b := l1Minus) (l2a := l2Sq) (l2b := l2Sq)
      (α := (1/4 : ℝ)) (β := (-1/4 : ℝ)) (γ := 0) (x := x)
    simpa [l1Prod, l2Prod, add_assoc, add_left_comm, add_comm] using this
  -- Apply the square approximation bounds.
  have hsq_plus : |(dot wPlus x) * (dot wPlus x) - mlpEval1d hidSq l1Sq l2Sq (dot wPlus x)| < δ :=
    hSq (dot wPlus x) hx_plus
  have hsq_minus : |(dot wMinus x) * (dot wMinus x) - mlpEval1d hidSq l1Sq l2Sq (dot wMinus x)| <
    δ :=
    hSq (dot wMinus x) hx_minus
  -- Now finish via `xy = ((x+y)^2 - (x-y)^2)/4` and triangle inequality.
  -- Expand `mulFun` and rewrite the network output using `hcomb` and the lift equalities.
  have hmul : mulFun x = ((dot wPlus x) * (dot wPlus x) - (dot wMinus x) * (dot wMinus x)) / 4 := by
    -- Convert to scalar coordinates and use the algebraic identity.
    have := mul_identity (x0 x) (x1 x)
    -- Rewrite `x+y` / `x-y` as `dot` expressions.
    simpa [mulFun, dot_wPlus, dot_wMinus, sub_eq_add_neg, add_comm, add_left_comm, add_assoc] using
      this
  -- Rewrite goal into an error between the true formula and the approximated formula.
  -- Use `δ = 2ε` so the final bound is `< ε`.
  have : |mulFun x - mlpEvalNd (n := 2) (hidDim := hidSq + hidSq) l1Prod l2Prod x| < ε := by
    -- Replace `mulFun` and the network output by the “difference of squares” forms.
    rw [hmul, hcomb, hplus_eval, hminus_eval]
    -- Let `e₁,e₂` be the square approximation errors.
    set e1 := (dot wPlus x) * (dot wPlus x) - mlpEval1d hidSq l1Sq l2Sq (dot wPlus x) with he1
    set e2 := (dot wMinus x) * (dot wMinus x) - mlpEval1d hidSq l1Sq l2Sq (dot wMinus x) with he2
    -- Reduce to bounding `|(e1 - e2)/4|`.
    have hrew :
        ((dot wPlus x) * (dot wPlus x) - (dot wMinus x) * (dot wMinus x)) / 4
          - ((1 / 4 : ℝ) * mlpEval1d hidSq l1Sq l2Sq (dot wPlus x) +
              (-1 / 4 : ℝ) * mlpEval1d hidSq l1Sq l2Sq (dot wMinus x))
        =
        (e1 - e2) / 4 := by
      -- Pure ring arithmetic.
      subst e1 e2
      ring
    -- Convert to abs and apply triangle inequality.
    have htri : |e1 - e2| ≤ |e1| + |e2| := by
      simpa [sub_eq_add_neg, abs_neg] using (abs_add_le e1 (-e2))
    have habs : |(e1 - e2) / 4| = |e1 - e2| / 4 := by
      simp [abs_div]
    -- Use the square approximation bounds to show `|e1|<δ` and `|e2|<δ`.
    have he1lt : |e1| < δ := by simpa [he1] using hsq_plus
    have he2lt : |e2| < δ := by simpa [he2] using hsq_minus
    -- Combine.
    have hsumlt : |e1| + |e2| < 2*δ := by linarith
    have : |(e1 - e2) / 4| < ε := by
      -- `|(e1-e2)/4| = |e1-e2|/4 ≤ (|e1|+|e2|)/4 < (2δ)/4 = ε` since `δ=2ε`.
      have hle : |e1 - e2| / 4 ≤ (|e1| + |e2|) / 4 := by
        have := div_le_div_of_nonneg_right htri (by norm_num : (0:ℝ) ≤ 4)
        simpa [div_eq_mul_inv, mul_assoc, mul_left_comm, mul_comm] using this
      have hlt : (|e1| + |e2|) / 4 < ε := by
        -- `(|e1|+|e2|) < 2δ` and `2δ/4 = ε`.
        have h' : (|e1| + |e2|) / 4 < (2*δ) / 4 :=
          div_lt_div_of_pos_right hsumlt (by norm_num : (0:ℝ) < 4)
        have hEq : (2*δ) / 4 = ε := by
          simp [δ]
          ring
        exact lt_of_lt_of_eq h' hEq
      exact lt_of_le_of_lt (by simpa [habs] using hle) hlt
    -- Return to the original abs goal.
    -- `hrew` turns the raw difference into `(e1-e2)/4`.
    -- Then `abs` agrees with `|·|`.
    have : |((dot wPlus x) * (dot wPlus x) - (dot wMinus x) * (dot wMinus x)) / 4
          - ((1 / 4 : ℝ) * mlpEval1d hidSq l1Sq l2Sq (dot wPlus x) +
              (-1 / 4 : ℝ) * mlpEval1d hidSq l1Sq l2Sq (dot wMinus x))| < ε := by
      -- rewrite to `(e1-e2)/4`
      have hrew' :
          ((dot wPlus x) * (dot wPlus x) - (dot wMinus x) * (dot wMinus x)) / 4
              - ((4 : ℝ)⁻¹ * mlpEval1d hidSq l1Sq l2Sq (dot wPlus x) +
                  (-1 / 4 : ℝ) * mlpEval1d hidSq l1Sq l2Sq (dot wMinus x))
            =
          (e1 - e2) / 4 := by
        simpa [one_div] using hrew
      -- avoid `simp [inv_eq_one_div]` (loops with `one_div`)
      simpa [hrew'] using this
    simpa [sub_eq_add_neg, add_assoc, add_comm, add_left_comm] using this
  exact this

end NN.MLTheory.Proofs.ReLUMulApprox
