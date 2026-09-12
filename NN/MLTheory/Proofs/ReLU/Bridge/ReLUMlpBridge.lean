/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.Proofs.Approximation.Universal.UniversalApproximation
public import NN.Spec.Core.Tensor -- shake: keep

/-!
# Bridging Scalar ReLU MLPs to Tensor Inputs

This file is a first “bridge step” between:

- the constructive scalar ReLU approximation theorem in `UniversalApproximation.lean`, and
- tensor inputs `Tensor ℝ [n]` used throughout TorchLean.

What is proved here (fully proved):

1. **Exact representability of affine maps** $x\mapsto w\mathbin{\cdot}x+b$ by a two-layer ReLU
   MLP of width $2$, using
   $\operatorname{ReLU}(u)-\operatorname{ReLU}(-u)=u$.
2. **Ridge lifting**: any scalar-input 2-layer ReLU MLP can be lifted to a tensor input via
   $u=w\mathbin{\cdot}x+c$, by scaling each first-layer weight by $w$ and adjusting biases
   accordingly.

What is *not* proved here: the full classical multivariate universal approximation theorem for ReLU
MLPs. That requires substantially more formalization (e.g. piecewise-linear approximation machinery
or a functional-analytic Cybenko/Leshno style proof).
-/

@[expose] public section


namespace NN.MLTheory.Proofs.ReLUMlpBridge

open _root_.Spec _root_.TorchLean
open Examples

open NN.MLTheory.Proofs.UniversalApproximation

/-- Rewrapping a rank-one tensor by `Tensor.dim` preserves every coordinate. -/
theorem getScalar_dim_getScalar {n : Nat} (x : Tensor ℝ [n]) :
    (fun i => (Tensor.dim (fun j : Fin n => Tensor.scalar (x.getScalar j))).getScalar i) =
      fun i => x.getScalar i := by
  funext j
  simp

/-- Dot product $w\mathbin{\cdot}x$ for coordinate weights `w` and a rank-one tensor `x`. -/
noncomputable def dot {n : Nat} (w : Fin n → ℝ) (x : Tensor ℝ [n]) : ℝ :=
  ∑ j : Fin n, w j * TorchLean.Tensor.getScalar x j

/-- Evaluate a single-hidden-layer ReLU MLP on a tensor input and return the scalar output. -/
noncomputable def mlpEval {n hidDim : Nat}
    (l1 : LinearSpec ℝ n hidDim) (l2 : LinearSpec ℝ hidDim 1)
    (x : Tensor ℝ [n]) : ℝ :=
  extractScalarOutput (Examples.mlpForward l1 l2 x)

/--
The identity $\operatorname{ReLU}(u)-\operatorname{ReLU}(-u)=u$, used to represent affine maps
exactly with ReLU.
-/
theorem relu_sub_relu_neg (u : ℝ) : relu u - relu (-u) = u := by
  by_cases h : 0 ≤ u
  · have hneg : -u ≤ 0 := by linarith
    simp [relu, Activation.Math.reluSpec_eq_max, max_eq_left h, max_eq_right hneg]
  · have hu : u ≤ 0 := le_of_not_ge h
    have hneg : 0 ≤ -u := by linarith
    -- In this branch, `relu u = 0` and `relu (-u) = -u`.
    simp [relu, Activation.Math.reluSpec_eq_max, max_eq_right hu, max_eq_left hneg]

/--
Unfold `mlpForward` as
$\operatorname{linear}\circ\operatorname{ReLU}\circ\operatorname{linear}$.

This lemma is used as the standard normalization step in “network algebra” proofs.
-/
theorem mlp_forward_eq_linear_relu_linear
    {n hidDim : Nat}
    (l1 : LinearSpec ℝ n hidDim) (l2 : LinearSpec ℝ hidDim 1)
    (x : Tensor ℝ [n]) :
    Examples.mlpForward l1 l2 x =
      let z1 := Spec.linearSpec (α := ℝ) l1 x
      let a1 := Activation.reluSpec z1
      Spec.linearSpec (α := ℝ) l2 a1 := by
  simpa [Examples.mlpForward] using (Examples.mlp_spec_forward_eq (α := ℝ) l1 l2 x)

/-- Extract the unique entry from row `i` of an `(m×1)` tensor interpreted as a matrix. -/
noncomputable def matDim1Get {m : Nat} (A : Tensor ℝ [m, 1]) (i : Fin m) : ℝ :=
  Spec.get2 A i ⟨0, by decide⟩

/-- Specialized matrix-vector multiplication when the input is a scalar (dimension `1`). -/
theorem mat_vec_mul_spec_dim1 {m : Nat} (A : Tensor ℝ [m, 1]) (x : ℝ) :
    Spec.matVecMulSpec A (Tensor.singleton x) =
      Tensor.dim (fun i : Fin m => Tensor.scalar (matDim1Get A i * x)) := by
  classical
  apply Tensor.ext_vector
  intro i
  simp [Spec.getScalar_mat_vec_mul_spec, matDim1Get, Tensor.singleton]

/--
General matrix-vector multiplication for `Tensor.matrix` and a vector written as `Tensor.dim`.

This generalizes the one-row dot-product lemma from `UniversalApproximation.lean` to arbitrary `m`.
-/
theorem mat_vec_mul_spec_matrix_vector
    (m n : ℕ) (c : Fin m → Fin n → ℝ) (v : Fin n → ℝ) :
    Spec.matVecMulSpec (Tensor.matrix (m := m) (n := n) c)
        (Tensor.dim (fun j => Tensor.scalar (v j))) =
      Tensor.dim (fun i : Fin m => Tensor.scalar (∑ j : Fin n, c i j * v j)) := by
  classical
  apply Tensor.ext_vector
  intro i
  simp [Spec.getScalar_mat_vec_mul_spec, Tensor.matrix, Spec.get2]

/--
First layer for exact affine representability.

Given an affine form $u(x)=w\mathbin{\cdot}x+b$, this layer outputs $[u(x),-u(x)]$.
-/
noncomputable def affineIdLayer1 {n : Nat} (w : Fin n → ℝ) (b : ℝ) : LinearSpec ℝ n 2 :=
  { weights := Tensor.matrix (m := 2) (n := n) (fun i j => if i.1 = 0 then w j else -w j)
    bias := Tensor.ofFn (n := 2) (fun i => if i.1 = 0 then b else -b) }

/--
Second layer for exact affine representability.

With hidden activations
$[\operatorname{ReLU}(u),\operatorname{ReLU}(-u)]$, this output layer computes
$\operatorname{ReLU}(u)-\operatorname{ReLU}(-u)=u$.
-/
noncomputable def affineIdLayer2 : LinearSpec ℝ 2 1 :=
  { weights := Tensor.matrix (m := 1) (n := 2)
      (fun _ j => if j.1 = 0 then (1 : ℝ) else (-1 : ℝ))
    bias := Tensor.ofFn (n := 1) (fun _ => (0 : ℝ)) }

/-- Standard basis vector $e_i\in\mathbb{R}^n$. -/
noncomputable def stdBasis {n : Nat} (i : Fin n) : Fin n → ℝ :=
  fun j => if j = i then (1 : ℝ) else 0

/-- $\operatorname{dot}(e_i,x)=x_i$ for the standard basis `stdBasis`. -/
theorem dot_stdBasis {n : Nat} (i : Fin n) (x : Tensor ℝ [n]) :
    dot (stdBasis (n := n) i) x = TorchLean.Tensor.getScalar x i := by
  classical
  -- `simp` evaluates the `Finset` sum with the `if j = i` selector.
  simp [dot, stdBasis]

/--
Exact representability of affine maps by a 2-layer ReLU MLP (width 2).

This is the core bridge lemma that turns scalar affine forms $w\mathbin{\cdot}x+b$ into MLP
evaluations.
-/
theorem mlp_eval_affine_id {n : Nat} (w : Fin n → ℝ) (b : ℝ) (x : Tensor ℝ [n]) :
    mlpEval (n := n) (hidDim := 2) (affineIdLayer1 (n := n) w b) affineIdLayer2 x =
      dot w x + b := by
  classical
  -- Unfold evaluation and rewrite the MLP as `linear ∘ relu ∘ linear`.
  unfold mlpEval
  rw [mlp_forward_eq_linear_relu_linear (n := n) (hidDim := 2)
        (l1 := affineIdLayer1 (n := n) w b) (l2 := affineIdLayer2) (x := x)]
  -- Compute the first linear layer output: `[u, -u]` where `u = dot w x + b`.
  have hz1 :
      Spec.linearSpec (α := ℝ) (affineIdLayer1 (n := n) w b) x =
        Tensor.dim (fun i : Fin 2 =>
          Tensor.scalar (if i.1 = 0 then (dot w x + b) else -(dot w x + b))) := by
    unfold Spec.linearSpec affineIdLayer1
    apply Tensor.ext_vector
    intro i
    rw [congrFun (Spec.getScalar_add_spec _ _) i]
    rw [Spec.getScalar_mat_vec_mul_spec]
    fin_cases i <;>
      simp [Tensor.matrix, Spec.get2, Tensor.ofFn, dot, add_comm]
  -- Apply ReLU pointwise.
  have ha1 :
      Activation.reluSpec (α := ℝ) (s := .dim 2 .scalar)
          (Spec.linearSpec (α := ℝ) (affineIdLayer1 (n := n) w b) x) =
        Tensor.dim (fun i : Fin 2 =>
          Tensor.scalar (if i.1 = 0 then relu (dot w x + b) else relu (-b + -dot w x))) := by
    rw [hz1]
    apply Tensor.ext_vector
    intro i
    fin_cases i <;>
      simp [Activation.reluSpec, TorchLean.Tensor.mapSpec, relu, Activation.Math.reluSpec_eq_max]
  -- Compute the output layer and extract the scalar.
  have hy :
      Spec.linearSpec (α := ℝ) affineIdLayer2
          (Activation.reluSpec (α := ℝ) (s := .dim 2 .scalar)
            (Spec.linearSpec (α := ℝ) (affineIdLayer1 (n := n) w b) x)) =
        Tensor.dim (fun _ : Fin 1 => Tensor.scalar (dot w x + b)) := by
    rw [ha1]
    dsimp only [Spec.linearSpec, affineIdLayer2]
    apply Tensor.ext_vector
    intro i
    rw [congrFun (Spec.getScalar_add_spec _ _) i]
    rw [Spec.getScalar_mat_vec_mul_spec]
    fin_cases i
    have h := relu_sub_relu_neg (u := dot w x + b)
    simpa [Tensor.matrix, Spec.get2, Tensor.ofFn, sub_eq_add_neg, neg_add, add_assoc,
      add_comm, add_left_comm, mul_assoc] using h
  -- Finish: `extract_scalar_output` picks the unique element of `Fin 1`.
  simp [extractScalarOutput, hy]

/-- Exact representability of coordinate projections $x\mapsto x_i$ by a width-$2$ ReLU MLP. -/
theorem mlp_eval_coord {n : Nat} (i : Fin n) (x : Tensor ℝ [n]) :
    mlpEval (n := n) (hidDim := 2) (affineIdLayer1 (n := n) (stdBasis (n := n) i) 0)
        affineIdLayer2 x =
      TorchLean.Tensor.getScalar x i := by
  simpa [dot_stdBasis] using (mlp_eval_affine_id (n := n) (w := stdBasis (n := n) i) (b := (0 : ℝ))
    x)

/-!
## Ridge lifting

Given a scalar-input MLP `(l1,l2)` and an affine scalar map
$u=w\mathbin{\cdot}x+c$, we build a tensor-input MLP whose pre-activations match the scalar-input
pre-activations at $u$. This lets you reuse any scalar approximation
result for functions of one affine form (“ridge functions”).
-/

/--
Lift a scalar-input first layer to a tensor-input first layer along a ridge direction.

Given a first layer that expects a scalar $u\in\mathbb{R}$, this constructs a tensor-input layer
that feeds it $u=w\mathbin{\cdot}x+c$.
-/
noncomputable def liftScalarLayer1
    {n hidDim : Nat} (l1 : LinearSpec ℝ 1 hidDim) (w : Fin n → ℝ) (c : ℝ) : LinearSpec ℝ n hidDim :=
  { weights := Tensor.matrix (m := hidDim) (n := n)
      (fun i j => matDim1Get l1.weights i * w j)
    bias := Tensor.ofFn (n := hidDim)
      (fun i => matDim1Get l1.weights i * c + TorchLean.Tensor.getScalar l1.bias i) }

/--
Lifting lemma: the lifted tensor-input MLP agrees with the scalar-input MLP evaluated at
$w\mathbin{\cdot}x+c$.
-/
theorem mlp_eval_lift_from_scalar
    {n hidDim : Nat} (l1 : LinearSpec ℝ 1 hidDim) (l2 : LinearSpec ℝ hidDim 1)
    (w : Fin n → ℝ) (c : ℝ) (x : Tensor ℝ [n]) :
    mlpEval (n := n) (hidDim := hidDim) (liftScalarLayer1 (n := n) l1 w c) l2 x =
      mlpEvalScalar hidDim l1 l2 (dot w x + c) := by
  classical
  -- Expand both sides to `mlp_forward`, then use the `linear ∘ relu ∘ linear` form.
  unfold mlpEval mlpEvalScalar
  rw [mlp_forward_eq_linear_relu_linear (n := n) (hidDim := hidDim)
        (l1 := liftScalarLayer1 (n := n) l1 w c) (l2 := l2) (x := x)]
  rw [UniversalApproximation.mlp_forward_eq_linear_relu_linear (hidDim := hidDim)
        (l1 := l1) (l2 := l2) (x := Tensor.singleton (dot w x + c))]
  -- Show the first linear layers agree coordinatewise.
  have hz1 :
      Spec.linearSpec (α := ℝ) (liftScalarLayer1 (n := n) l1 w c) x =
        Spec.linearSpec (α := ℝ) l1 (Tensor.singleton (dot w x + c)) := by
    unfold Spec.linearSpec liftScalarLayer1
    apply Tensor.ext_vector
    intro i
    repeat' rw [congrFun (Spec.getScalar_add_spec _ _) i]
    repeat' rw [Spec.getScalar_mat_vec_mul_spec]
    simp [matDim1Get, Tensor.matrix, Spec.get2, Tensor.singleton, Tensor.ofFn, dot,
      Finset.mul_sum, mul_add, mul_assoc, mul_left_comm, mul_comm, add_assoc, add_comm]
  -- With `z1` equal, everything else is definitional.
  simp [hz1, extractScalarOutput, Activation.reluSpec]

end NN.MLTheory.Proofs.ReLUMlpBridge
