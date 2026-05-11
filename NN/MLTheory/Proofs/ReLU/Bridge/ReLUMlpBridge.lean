/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Data.Real.Basic
public import NN.MLTheory.Proofs.Approximation.Universal.UniversalApproximation
public import NN.Spec.Core.Tensor
public import NN.Spec.Layers.Activation
public import NN.Spec.Layers.Linear
public import NN.Spec.Models.Mlp
import Mathlib.Tactic.Linarith

/-!
# Bridging 1D ReLU MLPs to Tensor inputs (ridge functions)

This file is a first “bridge step” between:

- the constructive 1D ReLU approximation theorem in `universal_approximation.lean`, and
- nD Tensor inputs `Tensor ℝ (.dim n .scalar)` used throughout TorchLean.

What is proved here (fully proved):

1. **Exact representability of affine maps** `x ↦ w⋅x + b` by a 2-layer ReLU MLP (width 2),
   using the identity `relu(u) - relu(-u) = u`.
2. **Ridge lifting**: any 1D 2-layer ReLU MLP can be lifted to an nD Tensor input via
   `u = w⋅x + c`, by scaling each first-layer weight by `w` and adjusting biases accordingly.

What is *not* proved here: the full classical nD universal approximation theorem for ReLU MLPs.
That requires substantially more formalization (e.g. piecewise-linear approximation machinery or a
functional-analytic Cybenko/Leshno style proof).
-/

@[expose] public section


namespace NN.MLTheory.Proofs.ReLUMlpBridge

open _root_.Spec
open Examples

open NN.MLTheory.Proofs.UniversalApproximation

/-- `Tensor ℝ (.dim n .scalar)` viewed as an `n`-vector of real scalars. -/
abbrev TensorVec (n : Nat) : Type := Tensor ℝ (.dim n .scalar)

/-- View a `TensorVec n` as a function `Fin n → ℝ` via `Tensor.dimScalarEquiv`. -/
noncomputable def toVec {n : Nat} (x : TensorVec n) : Fin n → ℝ :=
  (Tensor.dimScalarEquiv (α := ℝ) n).toFun x

/-- Rewrapping a vector by `Tensor.dim` preserves the underlying coordinate function `toVec`. -/
lemma toVec_dim_toVec {n : Nat} (x : TensorVec n) :
    toVec (Tensor.dim (fun j : Fin n => Tensor.scalar (toVec x j))) = toVec x := by
  funext j
  cases x with
  | dim f =>
    simp [toVec, Tensor.dimScalarEquiv, Tensor.toScalar]

/-- Dot product `w ⋅ x` for a weight function `w : Fin n → ℝ` and `x : TensorVec n`. -/
noncomputable def dot {n : Nat} (w : Fin n → ℝ) (x : TensorVec n) : ℝ :=
  ∑ j : Fin n, w j * toVec x j

/-- Evaluate a single-hidden-layer ReLU MLP on a tensor input and return the scalar output. -/
noncomputable def mlpEvalNd {n hidDim : Nat}
    (l1 : LinearSpec ℝ n hidDim) (l2 : LinearSpec ℝ hidDim 1) (x : TensorVec n) : ℝ :=
  extractScalarOutput (Examples.mlpForward l1 l2 x)

/-- Identity `relu u - relu (-u) = u`, used to represent affine maps exactly with ReLU. -/
lemma relu_sub_relu_neg (u : ℝ) : relu u - relu (-u) = u := by
  by_cases h : 0 ≤ u
  · have hneg : -u ≤ 0 := by linarith
    simp [relu, Activation.Math.reluSpec, max_eq_left h, max_eq_right hneg]
  · have hu : u ≤ 0 := le_of_not_ge h
    have hneg : 0 ≤ -u := by linarith
    -- In this branch, `relu u = 0` and `relu (-u) = -u`.
    simp [relu, Activation.Math.reluSpec, max_eq_right hu, max_eq_left hneg]

/--
Unfold `mlp_forward` as `linear ∘ relu ∘ linear`.

This lemma is used as the standard normalization step in “network algebra” proofs.
-/
lemma mlp_forward_eq_linear_relu_linear
    {n hidDim : Nat}
    (l1 : LinearSpec ℝ n hidDim) (l2 : LinearSpec ℝ hidDim 1) (x : TensorVec n) :
    Examples.mlpForward l1 l2 x =
      let z1 := Spec.linearSpec (α := ℝ) l1 x
      let a1 := Activation.reluSpec z1
      Spec.linearSpec (α := ℝ) l2 a1 := by
  simpa [Examples.mlpForward] using (Examples.mlp_spec_forward_eq (α := ℝ) l1 l2 x)

/-- Extract the unique entry from row `i` of an `(m×1)` tensor interpreted as a matrix. -/
noncomputable def matDim1Get {m : Nat} (A : Tensor ℝ (.dim m (.dim 1 .scalar))) (i : Fin m) : ℝ :=
  match A with
  | .dim rows =>
    match rows i with
    | .dim cols => Tensor.toScalar (cols ⟨0, by decide⟩)

/-- Extract the `i`-th entry of a vector-shaped tensor. -/
noncomputable def vecGet {m : Nat} (v : Tensor ℝ (.dim m .scalar)) (i : Fin m) : ℝ :=
  match v with
  | .dim f => Tensor.toScalar (f i)

/-- Specialized matrix-vector multiplication when the input is a scalar (dimension `1`). -/
lemma mat_vec_mul_spec_dim1 {m : Nat} (A : Tensor ℝ (.dim m (.dim 1 .scalar))) (x : ℝ) :
    Spec.matVecMulSpec A (Tensor.singleton x) =
      Tensor.dim (fun i : Fin m => Tensor.scalar (matDim1Get A i * x)) := by
  classical
  -- The shape forces `A` to be a `Tensor.dim`.
  cases A with
  | dim rows =>
    apply congrArg Tensor.dim
    funext i
    -- Unfold the unique dot-product over `Fin 1`.
    have hfin : (List.finRange 1) = [⟨0, by decide⟩] := by
      simp [List.finRange_succ]
    -- Reduce to the only row element `A[i,0]`.
    cases hrow : rows i with
    | dim cols =>
      cases hcol : cols ⟨0, by decide⟩ with
      | scalar ak =>
        have hcol0 : cols 0 = Tensor.scalar ak := by
          simpa using hcol
        simp [matDim1Get, hfin, hrow, hcol0, List.foldl, Tensor.toScalar]

/--
General matrix-vector multiplication for `matrixMN` and a vector written as `Tensor.dim`.

This generalizes the 1-row dot-product lemma from `universal_approximation.lean` to arbitrary `m`.
-/
lemma mat_vec_mul_spec_matrixMN_vector
    (m n : ℕ) (c : Fin m → Fin n → ℝ) (v : Fin n → ℝ) :
    Spec.matVecMulSpec (matrixMN m n c) (Tensor.dim (fun j => Tensor.scalar (v j))) =
      Tensor.dim (fun i : Fin m => Tensor.scalar (∑ j : Fin n, c i j * v j)) := by
  classical
  cases m with
  | zero =>
    -- Both sides are length-0 vectors.
    apply congrArg Tensor.dim
    funext i
    exact (Fin.elim0 i)
  | succ m =>
    apply congrArg Tensor.dim
    funext i
    -- Compute the dot product for row `i`.
    simp []
    -- Convert the fold to the scalar-accumulator form used in `finRange_foldl_add_scalar`.
    have hfold :
        List.foldl
            (fun (acc : Tensor ℝ .scalar) (k : Fin n) =>
              Spec.matVecMulSpec.match_1 (α := ℝ) (motive := fun _ _ _ => Tensor ℝ .scalar)
                acc (Tensor.scalar (c i k)) (Tensor.scalar (v k))
                (fun s ak vk => Tensor.scalar (s + ak * vk)))
            (Tensor.scalar 0) (List.finRange n) =
          List.foldl
            (fun (acc : Tensor ℝ .scalar) (k : Fin n) =>
              match acc with
              | Tensor.scalar s => Tensor.scalar (s + c i k * v k))
            (Tensor.scalar 0) (List.finRange n) := by
      refine List.foldl_ext
          (f := fun (acc : Tensor ℝ .scalar) (k : Fin n) =>
            Spec.matVecMulSpec.match_1 (α := ℝ) (motive := fun _ _ _ => Tensor ℝ .scalar)
              acc (Tensor.scalar (c i k)) (Tensor.scalar (v k))
              (fun s ak vk => Tensor.scalar (s + ak * vk)))
          (g := fun (acc : Tensor ℝ .scalar) (k : Fin n) =>
            match acc with
            | Tensor.scalar s => Tensor.scalar (s + c i k * v k))
          (a := Tensor.scalar 0) ?_
      intro acc k _
      cases acc
      simp
    -- Finish using the fold lemma from the 1D ReLU approximation file.
    exact hfold.trans (finRange_foldl_add_scalar n (fun j : Fin n => c i j * v j))

/--
First layer for exact affine representability.

Given an affine form `u(x) = w ⋅ x + b`, this layer outputs `[u(x), -u(x)]`.
-/
noncomputable def affineIdLayer1 {n : Nat} (w : Fin n → ℝ) (b : ℝ) : LinearSpec ℝ n 2 :=
  { weights := matrixMN 2 n (fun i j => if i.1 = 0 then w j else -w j)
    bias := vectorN 2 (fun i => if i.1 = 0 then b else -b) }

/--
Second layer for exact affine representability.

With hidden activations `[relu(u), relu(-u)]`, this output layer computes `relu(u) - relu(-u) = u`.
-/
noncomputable def affineIdLayer2 : LinearSpec ℝ 2 1 :=
  { weights := matrixMN 1 2 (fun _ j => if j.1 = 0 then (1 : ℝ) else (-1 : ℝ))
    bias := vectorN 1 (fun _ => (0 : ℝ)) }

/-- Standard basis vector `e_i : Fin n → ℝ`. -/
noncomputable def stdBasis {n : Nat} (i : Fin n) : Fin n → ℝ :=
  fun j => if j = i then (1 : ℝ) else 0

/-- `dot e_i x = x_i` for the standard basis `stdBasis`. -/
lemma dot_stdBasis {n : Nat} (i : Fin n) (x : TensorVec n) :
    dot (stdBasis (n := n) i) x = toVec x i := by
  classical
  -- `simp` evaluates the `Finset` sum with the `if j = i` selector.
  simp [dot, stdBasis]

/--
Exact representability of affine maps by a 2-layer ReLU MLP (width 2).

This is the core “bridge lemma” that turns scalar affine forms `w ⋅ x + b` into MLP evaluations.
-/
theorem mlp_eval_affine_id {n : Nat} (w : Fin n → ℝ) (b : ℝ) (x : TensorVec n) :
    mlpEvalNd (n := n) (hidDim := 2) (affineIdLayer1 (n := n) w b) affineIdLayer2 x =
      dot w x + b := by
  classical
  -- Unfold evaluation and rewrite the MLP as `linear ∘ relu ∘ linear`.
  unfold mlpEvalNd
  rw [mlp_forward_eq_linear_relu_linear (n := n) (hidDim := 2)
        (l1 := affineIdLayer1 (n := n) w b) (l2 := affineIdLayer2) (x := x)]
  -- Work with `x` as an explicit vector of scalars.
  have hx :
      x = Tensor.dim (fun j : Fin n => Tensor.scalar (toVec x j)) := by
    cases x with
    | dim f =>
      apply congrArg Tensor.dim
      funext j
      cases fj : f j with
      | scalar r =>
        simp [toVec, Tensor.dimScalarEquiv, fj, Tensor.toScalar]
  -- Compute the first linear layer output: `[u, -u]` where `u = dot w x + b`.
  have hz1 :
      Spec.linearSpec (α := ℝ) (affineIdLayer1 (n := n) w b) x =
        Tensor.dim (fun i : Fin 2 =>
          Tensor.scalar (if i.1 = 0 then (dot w x + b) else -(dot w x + b))) := by
    -- Unfold `linear_spec` and use the generalized dot-product lemma.
    unfold Spec.linearSpec affineIdLayer1
    -- Rewrite `x` to the canonical `Tensor.dim` form so `mat_vec_mul_spec_matrixMN_vector` applies.
    rw [hx]
    have hmv :
        Spec.matVecMulSpec
            (matrixMN 2 n (fun i j => if i.1 = 0 then w j else -w j))
            (Tensor.dim (fun j : Fin n => Tensor.scalar (toVec x j))) =
          Tensor.dim (fun i : Fin 2 =>
            Tensor.scalar (∑ j : Fin n, (if i.1 = 0 then w j else -w j) * toVec x j)) := by
      simpa using
        (mat_vec_mul_spec_matrixMN_vector (m := 2) (n := n)
          (c := fun i j => if i.1 = 0 then w j else -w j) (v := toVec x))
    rw [hmv]
    -- Now add the bias and simplify each coordinate.
    -- `simp` reduces the tensor addition to a pointwise scalar goal.
    apply congrArg Tensor.dim
    funext i
    -- Split on the `Fin 2` index (`i=0` / `i=1`).
    fin_cases i <;>
      simp [Spec.Tensor.map2Spec, dot, toVec_dim_toVec,
        add_comm]
  -- Apply ReLU pointwise.
  have ha1 :
      Activation.reluSpec (α := ℝ) (s := .dim 2 .scalar)
          (Spec.linearSpec (α := ℝ) (affineIdLayer1 (n := n) w b) x) =
        Tensor.dim (fun i : Fin 2 =>
          Tensor.scalar (if i.1 = 0 then relu (dot w x + b) else relu (-b + -dot w x))) := by
    -- Rewrite the pre-activation tensor into a `Tensor.dim`, then unfold `relu_spec` as a map.
    rw [hz1]
    -- `relu_spec` maps elementwise, so we can prove the result pointwise over `Fin 2`.
    simp [Activation.reluSpec, Spec.Tensor.mapSpec]
    funext i
    fin_cases i <;> simp [relu, Activation.Math.reluSpec]
  -- Compute the output layer and extract the scalar.
  have hy :
      Spec.linearSpec (α := ℝ) affineIdLayer2
          (Activation.reluSpec (α := ℝ) (s := .dim 2 .scalar)
            (Spec.linearSpec (α := ℝ) (affineIdLayer1 (n := n) w b) x)) =
        Tensor.dim (fun _ : Fin 1 => Tensor.scalar (dot w x + b)) := by
    rw [ha1]
    unfold Spec.linearSpec affineIdLayer2
    -- Dot product in the final linear layer: `[1, -1] · [relu u, relu (-u)]`.
    have hmv2 :
        Spec.matVecMulSpec (matrixMN 1 2 (fun _ j => if j.1 = 0 then (1 : ℝ) else (-1 : ℝ)))
            (Tensor.dim (fun j : Fin 2 =>
              Tensor.scalar (if j.1 = 0 then relu (dot w x + b) else relu (-b + -dot w x)))) =
          Tensor.dim (fun _ : Fin 1 =>
            Tensor.scalar ((1 : ℝ) * relu (dot w x + b) + (-1 : ℝ) * relu (-b + -dot w x))) := by
      -- Use the 1-row lemma from the 1D ReLU approximation file.
      have := UniversalApproximation.mat_vec_mul_spec_matrixMN_vector 2
        (c := fun j : Fin 2 => if j.1 = 0 then (1 : ℝ) else (-1 : ℝ))
        (v := fun j : Fin 2 => if j.1 = 0 then relu (dot w x + b) else relu (-b + -dot w x))
      simpa [Tensor.dim] using this
    rw [hmv2]
    -- Add bias 0 and simplify using `relu(u) - relu(-u) = u`.
    apply congrArg Tensor.dim
    funext i
    fin_cases i
    -- The scalar goal is exactly `relu u - relu (-u) = u` (after rewriting `a - b` as `a + -b`).
    have h := relu_sub_relu_neg (u := dot w x + b)
    simpa [vectorN_eq_dim, Spec.Tensor.addSpec, Spec.Tensor.map2Spec, sub_eq_add_neg, relu,
      Tensor.toScalar,
      neg_add, add_assoc, add_comm, add_left_comm, mul_assoc] using h
  -- Finish: `extract_scalar_output` picks the unique element of `Fin 1`.
  simp [extractScalarOutput, hy, Tensor.toScalar]

/-- Exact representability of coordinate projections `x ↦ x_i` by a width-2 ReLU MLP. -/
theorem mlp_eval_coord {n : Nat} (i : Fin n) (x : TensorVec n) :
    mlpEvalNd (n := n) (hidDim := 2) (affineIdLayer1 (n := n) (stdBasis (n := n) i) 0)
        affineIdLayer2 x =
      toVec x i := by
  simpa [dot_stdBasis] using (mlp_eval_affine_id (n := n) (w := stdBasis (n := n) i) (b := (0 : ℝ))
    x)

/-!
## Ridge lifting

Given a 1D MLP `(l1,l2)` and an affine scalar map `u = w⋅x + c`, we build an nD MLP whose
pre-activations match the 1D pre-activations at `u`. This lets you reuse any 1D approximation
result for functions of one affine form (“ridge functions”).
-/

/--
Lift a 1D first-layer spec to an nD first-layer spec along a ridge direction.

Given a scalar 1D first layer that expects input `u : ℝ`, this constructs an nD first layer that
feeds `u = w ⋅ x + c`.
-/
noncomputable def liftLayer1From1d
    {n hidDim : Nat} (l1 : LinearSpec ℝ 1 hidDim) (w : Fin n → ℝ) (c : ℝ) : LinearSpec ℝ n hidDim :=
  { weights := matrixMN hidDim n (fun i j => matDim1Get l1.weights i * w j)
    bias := vectorN hidDim (fun i => matDim1Get l1.weights i * c + vecGet l1.bias i) }

/-- Lifting lemma: the lifted nD MLP agrees with the 1D MLP evaluated at `dot w x + c`. -/
theorem mlp_eval_lift_from_1d
    {n hidDim : Nat} (l1 : LinearSpec ℝ 1 hidDim) (l2 : LinearSpec ℝ hidDim 1)
    (w : Fin n → ℝ) (c : ℝ) (x : TensorVec n) :
    mlpEvalNd (n := n) (hidDim := hidDim) (liftLayer1From1d (n := n) l1 w c) l2 x =
      mlpEval1d hidDim l1 l2 (dot w x + c) := by
  classical
  -- Expand both sides to `mlp_forward`, then use the `linear ∘ relu ∘ linear` form.
  unfold mlpEvalNd mlpEval1d
  rw [mlp_forward_eq_linear_relu_linear (n := n) (hidDim := hidDim)
        (l1 := liftLayer1From1d (n := n) l1 w c) (l2 := l2) (x := x)]
  rw [UniversalApproximation.mlp_forward_eq_linear_relu_linear (hidDim := hidDim)
        (l1 := l1) (l2 := l2) (x := Tensor.singleton (dot w x + c))]
  -- Show the first linear layers agree coordinatewise.
  have hx :
      x = Tensor.dim (fun j : Fin n => Tensor.scalar (toVec x j)) := by
    cases x with
    | dim f =>
      apply congrArg Tensor.dim
      funext j
      cases fj : f j with
      | scalar r =>
        simp [toVec, Tensor.dimScalarEquiv, fj, Tensor.toScalar]
  have hz1 :
      Spec.linearSpec (α := ℝ) (liftLayer1From1d (n := n) l1 w c) x =
        Spec.linearSpec (α := ℝ) l1 (Tensor.singleton (dot w x + c)) := by
    -- Compute both sides as explicit vectors in `Fin hidDim → ℝ`.
    have hleft :
        Spec.linearSpec (α := ℝ) (liftLayer1From1d (n := n) l1 w c) x =
          Tensor.dim (fun i : Fin hidDim =>
            Tensor.scalar (matDim1Get l1.weights i * (dot w x) + (matDim1Get l1.weights i * c +
              vecGet l1.bias i))) := by
      unfold Spec.linearSpec liftLayer1From1d
      rw [hx]
      have hmv :
          Spec.matVecMulSpec
              (matrixMN hidDim n (fun i j => matDim1Get l1.weights i * w j))
              (Tensor.dim (fun j : Fin n => Tensor.scalar (toVec x j))) =
            Tensor.dim (fun i : Fin hidDim =>
              Tensor.scalar (∑ j : Fin n, (matDim1Get l1.weights i * w j) * toVec x j)) := by
        simpa using
          (mat_vec_mul_spec_matrixMN_vector (m := hidDim) (n := n)
            (c := fun i j => matDim1Get l1.weights i * w j) (v := toVec x))
      rw [hmv]
      -- Add bias and factor out the constant scalar weight.
      simp [Spec.Tensor.addSpec, Spec.Tensor.map2Spec, vectorN_eq_dim, dot, toVec_dim_toVec,
        Finset.mul_sum,
        mul_assoc, mul_left_comm, mul_comm, add_assoc, add_comm]
    -- Right side: because the input is a singleton, mat-vec reduces to `a_i * u`.
    have hright :
        Spec.linearSpec (α := ℝ) l1 (Tensor.singleton (dot w x + c)) =
          Tensor.dim (fun i : Fin hidDim =>
            Tensor.scalar (matDim1Get l1.weights i * (dot w x + c) + vecGet l1.bias i)) := by
      unfold Spec.linearSpec
      have hmv1 :
          Spec.matVecMulSpec l1.weights (Tensor.singleton (dot w x + c)) =
            Tensor.dim (fun i : Fin hidDim => Tensor.scalar (matDim1Get l1.weights i * (dot w x +
              c))) := by
        simpa using (mat_vec_mul_spec_dim1 (A := l1.weights) (x := dot w x + c))
      rw [hmv1]
      cases hbias : l1.bias with
      | dim fbias =>
        -- Reduce tensor addition to pointwise scalar addition.
        apply congrArg Tensor.dim
        funext i
        cases hbi : fbias i with
        | scalar bi =>
          simp [Spec.Tensor.map2Spec, vecGet, hbi, Tensor.toScalar]
    -- Rewrite both sides to explicit `Tensor.dim` forms and compare pointwise.
    rw [hleft, hright]
    apply congrArg Tensor.dim
    funext i
    -- Scalar arithmetic.
    simp [mul_add, add_assoc, add_left_comm, add_comm]
  -- With `z1` equal, everything else is definitional.
  simp [hz1, extractScalarOutput, Activation.reluSpec]

end NN.MLTheory.Proofs.ReLUMlpBridge
