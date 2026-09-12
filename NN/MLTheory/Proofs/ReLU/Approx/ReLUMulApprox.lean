/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.Proofs.ReLU.Bridge.ReLUMlpBridge

/-!
# Approximating multiplication with a 2-layer ReLU MLP (2D box)

This file gives a constructive, fully proved approximation result:
on $[-M,M]^2$, the function $(x_0,x_1)\mapsto x_0x_1$ can be uniformly approximated by a
  single-hidden-layer
ReLU MLP on `Tensor ℝ [2]`.
-/

@[expose] public section


namespace NN.MLTheory.Proofs.ReLUMulApprox

open _root_.Spec _root_.TorchLean
open Examples

open NN.MLTheory.Proofs.UniversalApproximation
open NN.MLTheory.Proofs.ReLUMlpBridge

/-- First coordinate projection from a rank-one tensor with two entries. -/
noncomputable def firstCoordinate (x : Tensor ℝ [2]) : ℝ :=
  TorchLean.Tensor.getScalar x ⟨0, by decide⟩

/-- Second coordinate projection from a rank-one tensor with two entries. -/
noncomputable def secondCoordinate (x : Tensor ℝ [2]) : ℝ :=
  TorchLean.Tensor.getScalar x ⟨1, by decide⟩

/-- The closed box domain $[-M,M]\times[-M,M]$. -/
noncomputable def box (M : ℝ) : Set (Tensor ℝ [2]) :=
  fun x => firstCoordinate x ∈ Set.Icc (-M) M ∧ secondCoordinate x ∈ Set.Icc (-M) M

/-- The target multiplication map: multiply the two coordinates. -/
noncomputable def mulFun (x : Tensor ℝ [2]) : ℝ :=
  firstCoordinate x * secondCoordinate x

/-- Ridge direction with `dot wPlus x` equal to the sum of the two coordinates. -/
noncomputable def wPlus : Fin 2 → ℝ := fun _ => 1

/-- Ridge direction with `dot wMinus x` equal to the first coordinate minus the second. -/
noncomputable def wMinus : Fin 2 → ℝ := fun i => if i.1 = 0 then 1 else (-1 : ℝ)

/-- Evaluate the ridge `wPlus`: it sums the two coordinates. -/
theorem dot_wPlus (x : Tensor ℝ [2]) : dot wPlus x = firstCoordinate x + secondCoordinate x := by
  classical
  -- Expand the `Fin 2` sum explicitly.
  simp [ReLUMlpBridge.dot, wPlus, firstCoordinate, secondCoordinate, Fin.sum_univ_two]

/-- Evaluate the ridge `wMinus`: $\operatorname{dot}(w_-,x)=x_0-x_1$. -/
theorem dot_wMinus (x : Tensor ℝ [2]) : dot wMinus x = firstCoordinate x - secondCoordinate x := by
  classical
  simp [ReLUMlpBridge.dot, wMinus, firstCoordinate, secondCoordinate, Fin.sum_univ_two,
    sub_eq_add_neg]

/-- Algebraic identity expressing multiplication via a difference of squares. -/
theorem mul_identity (x y : ℝ) : x * y = ((x + y) * (x + y) - (x - y) * (x - y)) / 4 := by
  ring

/-- Unpack the defining bounds of membership in `box M`. -/
theorem box_bounds {M : ℝ} (_hM : 0 ≤ M) {x : Tensor ℝ [2]} (hx : x ∈ box M) :
    firstCoordinate x ∈ Set.Icc (-M) M ∧ secondCoordinate x ∈ Set.Icc (-M) M := hx

/-- If $x\in\operatorname{box}(M)$, then $x_0+x_1\in[-2M,2M]$. -/
theorem sum_mem_Icc {M : ℝ} (_hM : 0 ≤ M) {x : Tensor ℝ [2]} (hx : x ∈ box M) :
    dot wPlus x ∈ Set.Icc (-2*M) (2*M) := by
  have hx0 := hx.1
  have hx1 := hx.2
  have hx0l : -M ≤ firstCoordinate x := hx0.1
  have hx0u : firstCoordinate x ≤ M := hx0.2
  have hx1l : -M ≤ secondCoordinate x := hx1.1
  have hx1u : secondCoordinate x ≤ M := hx1.2
  -- bounds on sum
  have hl : -(2*M) ≤ firstCoordinate x + secondCoordinate x := by linarith
  have hu : firstCoordinate x + secondCoordinate x ≤ 2*M := by linarith
  simpa [dot_wPlus] using And.intro hl hu

/-- If $x\in\operatorname{box}(M)$, then $x_0-x_1\in[-2M,2M]$. -/
theorem diff_mem_Icc {M : ℝ} (_hM : 0 ≤ M) {x : Tensor ℝ [2]} (hx : x ∈ box M) :
    dot wMinus x ∈ Set.Icc (-2*M) (2*M) := by
  have hx0 := hx.1
  have hx1 := hx.2
  have hx0l : -M ≤ firstCoordinate x := hx0.1
  have hx0u : firstCoordinate x ≤ M := hx0.2
  have hx1l : -M ≤ secondCoordinate x := hx1.1
  have hx1u : secondCoordinate x ≤ M := hx1.2
  have hl : -(2*M) ≤ firstCoordinate x - secondCoordinate x := by linarith
  have hu : firstCoordinate x - secondCoordinate x ≤ 2*M := by linarith
  simpa [dot_wMinus] using And.intro hl hu

/-- Lipschitz bound for `square` on $[-R,R]$: $|x^2-y^2|\leq 2R|x-y|$. -/
theorem square_lipschitz_Icc {R : ℝ} (_hR : 0 ≤ R) :
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
noncomputable def appendDim {α : Type} [TorchLean.Storage α] {m n : Nat} {s : Shape}
    (a : Tensor α (.dim m s)) (b : Tensor α (.dim n s)) : Tensor α (.dim (m+n) s) :=
  Tensor.dim (Fin.append a.unstack b.unstack)

/-- Append two first-layer linear specs by appending their weight and bias tensors. -/
noncomputable def appendLinearSpec
    {inDim m n : Nat} (a : LinearSpec ℝ inDim m) (b : LinearSpec ℝ inDim n) :
    LinearSpec ℝ inDim (m+n) :=
  { weights := appendDim a.weights b.weights
    bias := appendDim a.bias b.bias }

/-- Extract the `j`-th entry from a `1 × n` tensor interpreted as a row matrix. -/
noncomputable def mat1Get {n : Nat} (A : Tensor ℝ [1, n]) (j : Fin n) : ℝ :=
  Spec.get2 A ⟨0, by decide⟩ j

/-- `mat1Get` agrees with the `Tensor.matrix` constructor. -/
theorem singleRowMatrix_get_matrix {n : Nat} (f : Fin 1 → Fin n → ℝ) (j : Fin n) :
    mat1Get (Tensor.matrix (m := 1) (n := n) f) j = f 0 j := by
  simp [mat1Get, Tensor.matrix, Spec.get2]

/--
Combine two scalar-output linear specs into one scalar-output spec on an appended hidden layer.

If the appended hidden vector is $[z_a;z_b]$, the resulting output layer computes
$\gamma+\alpha\,\mathrm{out}_a(z_a)+\beta\,\mathrm{out}_b(z_b)$.
-/
noncomputable def combineOutput
    {m n : Nat} (α β γ : ℝ) (a : LinearSpec ℝ m 1) (b : LinearSpec ℝ n 1) :
    LinearSpec ℝ (m+n) 1 :=
  let c : Fin (m+n) → ℝ :=
    Fin.addCases (fun j : Fin m => α * mat1Get a.weights j) (fun j : Fin n => β * mat1Get
      b.weights j)
  { weights := Tensor.matrix (m := 1) (n := m + n) (fun _ j => c j)
    bias := Tensor.ofFn (n := 1)
      (fun _ => γ + α * extractScalarOutput a.bias + β * extractScalarOutput b.bias) }

/-- Reading the left component from an appended hidden vector. -/
theorem getScalar_append_left {m n : Nat} (a : Tensor ℝ [m]) (b : Tensor ℝ [n]) (i : Fin m) :
    TorchLean.Tensor.getScalar (appendDim a b) (Fin.castAdd n i) =
      TorchLean.Tensor.getScalar a i := by
  simp [appendDim, TorchLean.Tensor.getScalar, Spec.get, Fin.append]

/-- Reading the right component from an appended hidden vector. -/
theorem getScalar_append_right {m n : Nat} (a : Tensor ℝ [m]) (b : Tensor ℝ [n]) (i : Fin n) :
    TorchLean.Tensor.getScalar (appendDim a b) (Fin.natAdd m i) =
      TorchLean.Tensor.getScalar b i := by
  simp [appendDim, TorchLean.Tensor.getScalar, Spec.get, Fin.append]

/-- Pointwise behavior of the ReLU activation on tensor-vectors. -/
theorem getScalar_relu {n : Nat} (z : Tensor ℝ [n]) (i : Fin n) :
    TorchLean.Tensor.getScalar (Activation.reluSpec (α := ℝ) (s := .dim n .scalar) z) i =
      relu (TorchLean.Tensor.getScalar z i) := by
  simp [Activation.reluSpec, relu]

/-- Matrix-vector multiplication for a `1 × n` matrix produces a single scalar coordinate. -/
theorem mat_vec_mul_spec_oneRow {n : Nat} (A : Tensor ℝ [1, n]) (v : Tensor ℝ [n]) :
    Spec.matVecMulSpec A v =
      Tensor.dim (fun _ : Fin 1 =>
        Tensor.scalar (∑ j : Fin n, mat1Get A j * TorchLean.Tensor.getScalar v j)) := by
  classical
  apply TorchLean.Tensor.ext_vector
  intro i
  fin_cases i
  simp [Spec.getScalar_mat_vec_mul_spec, mat1Get]

/--
Expand `mlp_eval_nd` into “bias + sum over hidden units” form.

This is the main normalization lemma used to prove that `appendLinearSpec` together with
`combineOutput` implements affine combinations of subnetworks.
-/
theorem mlp_eval_nd_eq_bias_sum
    {inDim hidDim : Nat} (l1 : LinearSpec ℝ inDim hidDim) (l2 : LinearSpec ℝ hidDim 1)
    (x : Tensor ℝ [inDim]) :
    mlpEval (n := inDim) (hidDim := hidDim) l1 l2 x =
      extractScalarOutput l2.bias
        + ∑ j : Fin hidDim, (mat1Get l2.weights j) *
            relu (TorchLean.Tensor.getScalar (Spec.linearSpec (α := ℝ) l1 x) j) := by
  classical
  unfold mlpEval
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
              TorchLean.Tensor.getScalar
                (Activation.reluSpec (α := ℝ) (s := .dim hidDim .scalar)
                  (Spec.linearSpec (α := ℝ) l1 x)) j)) := by
    simpa using mat_vec_mul_spec_oneRow (A := l2.weights)
      (v := Activation.reluSpec (α := ℝ) (s := .dim hidDim .scalar) (Spec.linearSpec (α := ℝ) l1
        x))
  change extractScalarOutput
      (TorchLean.Tensor.addSpec
        (Spec.matVecMulSpec l2.weights
          (Activation.reluSpec (Spec.linearSpec (α := ℝ) l1 x)))
        l2.bias) = _
  rw [hmv]
  unfold extractScalarOutput
  rw [congrFun (Spec.getScalar_add_spec _ _) ⟨0, by decide⟩]
  simp [getScalar_relu, add_comm]

/-- Selecting the left block of a linear spec appended via `appendLinearSpec`. -/
theorem getScalar_linear_spec_append_left
    {inDim m n : Nat} (l1a : LinearSpec ℝ inDim m) (l1b : LinearSpec ℝ inDim n)
    (x : Tensor ℝ [inDim]) (i : Fin m) :
    TorchLean.Tensor.getScalar
        (Spec.linearSpec (α := ℝ) (appendLinearSpec (inDim := inDim) l1a l1b) x) (Fin.castAdd n i)
      =
    TorchLean.Tensor.getScalar (Spec.linearSpec (α := ℝ) l1a x) i := by
  classical
  unfold Spec.linearSpec
  rw [congrFun (Spec.getScalar_add_spec _ _) (Fin.castAdd n i)]
  rw [congrFun (Spec.getScalar_add_spec _ _) i]
  rw [Spec.getScalar_mat_vec_mul_spec, Spec.getScalar_mat_vec_mul_spec]
  simp [appendLinearSpec, appendDim, Spec.get2, Spec.get, TorchLean.Tensor.getScalar,
    Fin.append]

/-- Selecting the right block of a linear spec appended via `appendLinearSpec`. -/
theorem getScalar_linear_spec_append_right
    {inDim m n : Nat} (l1a : LinearSpec ℝ inDim m) (l1b : LinearSpec ℝ inDim n)
    (x : Tensor ℝ [inDim]) (i : Fin n) :
    TorchLean.Tensor.getScalar
        (Spec.linearSpec (α := ℝ) (appendLinearSpec (inDim := inDim) l1a l1b) x) (Fin.natAdd m i)
      =
    TorchLean.Tensor.getScalar (Spec.linearSpec (α := ℝ) l1b x) i := by
  classical
  unfold Spec.linearSpec
  rw [congrFun (Spec.getScalar_add_spec _ _) (Fin.natAdd m i)]
  rw [congrFun (Spec.getScalar_add_spec _ _) i]
  rw [Spec.getScalar_mat_vec_mul_spec, Spec.getScalar_mat_vec_mul_spec]
  simp [appendLinearSpec, appendDim, Spec.get2, Spec.get, TorchLean.Tensor.getScalar,
    Fin.append]

/--
Appending hidden units and wiring the output with `combineOutput` yields an affine combination.

Concretely, the combined network computes:
$\gamma+\alpha\,\mathrm{net}_a(x)+\beta\,\mathrm{net}_b(x)$.
-/
theorem mlp_eval_append_linear
    {inDim m n : Nat} (l1a : LinearSpec ℝ inDim m) (l1b : LinearSpec ℝ inDim n)
    (l2a : LinearSpec ℝ m 1) (l2b : LinearSpec ℝ n 1)
    (α β γ : ℝ) (x : Tensor ℝ [inDim]) :
    mlpEval (n := inDim) (hidDim := m+n)
        (appendLinearSpec (inDim := inDim) l1a l1b)
        (combineOutput (m := m) (n := n) α β γ l2a l2b) x
      =
    γ + α * mlpEval (n := inDim) (hidDim := m) l1a l2a x
      + β * mlpEval (n := inDim) (hidDim := n) l1b l2b x := by
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
            relu (TorchLean.Tensor.getScalar
              (Spec.linearSpec (α := ℝ) (appendLinearSpec (inDim := inDim) l1a l1b) x) j))
        =
      (∑ j : Fin m,
          (α * mat1Get l2a.weights j) *
            relu (TorchLean.Tensor.getScalar (Spec.linearSpec (α := ℝ) l1a x) j))
      +
      (∑ j : Fin n,
          (β * mat1Get l2b.weights j) *
            relu (TorchLean.Tensor.getScalar (Spec.linearSpec (α := ℝ) l1b x) j)) := by
    classical
    -- Use `Fin.sum_univ_add` and then simplify the `Fin.addCases` selectors.
    have hsum :=
      (Fin.sum_univ_add (a := m) (b := n)
        (f := fun j : Fin (m+n) =>
          mat1Get (combineOutput (m := m) (n := n) α β γ l2a l2b).weights j *
            relu (TorchLean.Tensor.getScalar
              (Spec.linearSpec (α := ℝ) (appendLinearSpec (inDim := inDim) l1a l1b) x) j)))
    -- Rewrite the `castAdd` / `natAdd` branches using the selector lemmas above.
    -- `combineOutput` uses `Fin.addCases` in its weights.
    simpa [combineOutput, singleRowMatrix_get_matrix, relu,
      getScalar_linear_spec_append_left (l1a := l1a) (l1b := l1b) (x := x),
      getScalar_linear_spec_append_right (l1a := l1a) (l1b := l1b) (x := x),
      Fin.addCases_left, Fin.addCases_right, mul_assoc, mul_left_comm, mul_comm] using hsum
  -- Factor out the scalars `α`/`β` from the two sums.
  let sumA : ℝ :=
    ∑ j : Fin m,
      mat1Get l2a.weights j * relu (TorchLean.Tensor.getScalar (Spec.linearSpec (α := ℝ) l1a x) j)
  let sumB : ℝ :=
    ∑ j : Fin n,
      mat1Get l2b.weights j * relu (TorchLean.Tensor.getScalar (Spec.linearSpec (α := ℝ) l1b x) j)
  have hsumA :
      (∑ j : Fin m,
          (α * mat1Get l2a.weights j) *
            relu (TorchLean.Tensor.getScalar (Spec.linearSpec (α := ℝ) l1a x) j))
        = α * sumA := by
    classical
    -- `∑ (α * t_j) = α * ∑ t_j` over `Fin m`.
    simpa [sumA, mul_assoc, mul_left_comm, mul_comm] using
      (Finset.mul_sum α (s := (Finset.univ : Finset (Fin m)))
          (f := fun j : Fin m =>
            mat1Get l2a.weights j *
              relu (TorchLean.Tensor.getScalar (Spec.linearSpec (α := ℝ) l1a x) j))).symm
  have hsumB :
      (∑ j : Fin n,
          (β * mat1Get l2b.weights j) *
            relu (TorchLean.Tensor.getScalar (Spec.linearSpec (α := ℝ) l1b x) j))
        = β * sumB := by
    classical
    simpa [sumB, mul_assoc, mul_left_comm, mul_comm] using
      (Finset.mul_sum β (s := (Finset.univ : Finset (Fin n)))
          (f := fun j : Fin n =>
            mat1Get l2b.weights j *
              relu (TorchLean.Tensor.getScalar (Spec.linearSpec (α := ℝ) l1b x) j))).symm
  -- Finish by normalizing the combined output bias and collecting the two hidden sums.
  have hbias :
      extractScalarOutput (combineOutput (m := m) (n := n) α β γ l2a l2b).bias
        = γ + α * extractScalarOutput l2a.bias + β * extractScalarOutput l2b.bias := by
    simp [combineOutput, extractScalarOutput, Tensor.ofFn]
  rw [hbias, hsplit, hsumA, hsumB]
  simp [sumA, sumB, mul_add, add_assoc, add_left_comm]

-- ---------------------------------------------------------------------------
-- Main theorem: multiplication approximation on `[-M,M]²`
-- ---------------------------------------------------------------------------

/--
Uniform approximation of multiplication on $[-M,M]^2$ by a single-hidden-layer ReLU MLP.

The construction follows the classical reduction
$xy=((x+y)^2-(x-y)^2)/4$, combined with a one-dimensional ReLU approximator for `square` on
$[-2M,2M]$
that is lifted along the ridge directions `wPlus` and `wMinus`.
-/
theorem relu_mul_universal_approximation_box
    {M : ℝ} (hM : 0 < M) :
    ∀ ε > 0, ∃ (hidDim : ℕ) (l1 : LinearSpec ℝ 2 hidDim) (l2 : LinearSpec ℝ hidDim 1),
      ∀ x ∈ box M, |mulFun x - mlpEval (n := 2) (hidDim := hidDim) l1 l2 x| < ε := by
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
  -- Step 2: lift to `u = firstCoordinate+secondCoordinate` and
  -- `u = firstCoordinate-secondCoordinate`.
  let l1Plus : LinearSpec ℝ 2 hidSq := liftScalarLayer1 (n := 2) l1Sq wPlus 0
  let l1Minus : LinearSpec ℝ 2 hidSq := liftScalarLayer1 (n := 2) l1Sq wMinus 0
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
      mlpEval (n := 2) (hidDim := hidSq) l1Plus l2Sq x =
        mlpEvalScalar hidSq l1Sq l2Sq (dot wPlus x) := by
    simpa [l1Plus, add_comm] using
      (mlp_eval_lift_from_scalar (n := 2) (hidDim := hidSq) l1Sq l2Sq wPlus 0 x)
  have hminus_eval :
      mlpEval (n := 2) (hidDim := hidSq) l1Minus l2Sq x =
        mlpEvalScalar hidSq l1Sq l2Sq (dot wMinus x) := by
    simpa [l1Minus, add_comm] using
      (mlp_eval_lift_from_scalar (n := 2) (hidDim := hidSq) l1Sq l2Sq wMinus 0 x)
  -- Evaluate the combined network as a linear combination of the two lifted nets.
  have hcomb :
      mlpEval (n := 2) (hidDim := hidSq + hidSq) l1Prod l2Prod x
        =
      (1/4 : ℝ) * mlpEval (n := 2) (hidDim := hidSq) l1Plus l2Sq x
        + (-1/4 : ℝ) * mlpEval (n := 2) (hidDim := hidSq) l1Minus l2Sq x := by
    -- `mlp_eval_append_linear` gives the general linear combination form.
    have := mlp_eval_append_linear (inDim := 2) (m := hidSq) (n := hidSq)
      (l1a := l1Plus) (l1b := l1Minus) (l2a := l2Sq) (l2b := l2Sq)
      (α := (1/4 : ℝ)) (β := (-1/4 : ℝ)) (γ := 0) (x := x)
    simpa [l1Prod, l2Prod, add_assoc, add_left_comm, add_comm] using this
  -- Apply the square approximation bounds.
  have hsq_plus :
      |(dot wPlus x) * (dot wPlus x) - mlpEvalScalar hidSq l1Sq l2Sq (dot wPlus x)| < δ :=
    hSq (dot wPlus x) hx_plus
  have hsq_minus :
      |(dot wMinus x) * (dot wMinus x) - mlpEvalScalar hidSq l1Sq l2Sq (dot wMinus x)| < δ :=
    hSq (dot wMinus x) hx_minus
  -- Now finish via `xy = ((x+y)^2 - (x-y)^2)/4` and triangle inequality.
  -- Expand `mulFun` and rewrite the network output using `hcomb` and the lift equalities.
  have hmul : mulFun x = ((dot wPlus x) * (dot wPlus x) - (dot wMinus x) * (dot wMinus x)) / 4 := by
    -- Convert to scalar coordinates and use the algebraic identity.
    have := mul_identity (firstCoordinate x) (secondCoordinate x)
    -- Rewrite `x+y` / `x-y` as `dot` expressions.
    simpa [mulFun, dot_wPlus, dot_wMinus, sub_eq_add_neg, add_comm, add_left_comm, add_assoc] using
      this
  -- Rewrite goal into an error between the true formula and the approximated formula.
  -- Use `δ = 2ε` so the final bound is `< ε`.
  have : |mulFun x - mlpEval (n := 2) (hidDim := hidSq + hidSq) l1Prod l2Prod x| < ε := by
    -- Replace `mulFun` and the network output by the “difference of squares” forms.
    rw [hmul, hcomb, hplus_eval, hminus_eval]
    -- Let `e₁,e₂` be the square approximation errors.
    set e1 := (dot wPlus x) * (dot wPlus x) - mlpEvalScalar hidSq l1Sq l2Sq (dot wPlus x) with he1
    set e2 := (dot wMinus x) * (dot wMinus x) - mlpEvalScalar hidSq l1Sq l2Sq (dot wMinus x)
      with he2
    -- Reduce to bounding `|(e1 - e2)/4|`.
    have hrew :
        ((dot wPlus x) * (dot wPlus x) - (dot wMinus x) * (dot wMinus x)) / 4
          - ((1 / 4 : ℝ) * mlpEvalScalar hidSq l1Sq l2Sq (dot wPlus x) +
              (-1 / 4 : ℝ) * mlpEvalScalar hidSq l1Sq l2Sq (dot wMinus x))
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
          - ((1 / 4 : ℝ) * mlpEvalScalar hidSq l1Sq l2Sq (dot wPlus x) +
              (-1 / 4 : ℝ) * mlpEvalScalar hidSq l1Sq l2Sq (dot wMinus x))| < ε := by
      -- rewrite to `(e1-e2)/4`
      have hrew' :
          ((dot wPlus x) * (dot wPlus x) - (dot wMinus x) * (dot wMinus x)) / 4
              - ((4 : ℝ)⁻¹ * mlpEvalScalar hidSq l1Sq l2Sq (dot wPlus x) +
                  (-1 / 4 : ℝ) * mlpEvalScalar hidSq l1Sq l2Sq (dot wMinus x))
            =
          (e1 - e2) / 4 := by
        simpa [one_div] using hrew
      -- avoid `simp [inv_eq_one_div]` (loops with `one_div`)
      simpa [hrew'] using this
    simpa [sub_eq_add_neg, add_assoc, add_comm, add_left_comm] using this
  exact this

end NN.MLTheory.Proofs.ReLUMulApprox
