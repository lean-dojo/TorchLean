/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.UniversalApproximation
public import TorchLean.Tensor.Contract

/-!
# Bridging Scalar ReLU MLPs to Tensor Inputs

The constructive ReLU approximation theorem of
`TorchLean.NN.UniversalApproximation` takes a scalar input. This
file connects it to the tensor inputs `Tensor ℝ [n]` used throughout TorchLean with two results:

1. **Exact representability of affine maps** $x\mapsto w\mathbin{\cdot}x+b$ by a two-layer ReLU
   MLP of width $2$, using
   $\operatorname{ReLU}(u)-\operatorname{ReLU}(-u)=u$.
2. **Ridge lifting**: any scalar-input 2-layer ReLU MLP can be lifted to a tensor input via
   $u=w\mathbin{\cdot}x+c$, by scaling each first-layer weight by $w$ and adjusting biases
   accordingly.

`TorchLean.NN.UniversalApproximation.CompactSet` combines these lemmas with Stone–Weierstrass
to prove the multivariate approximation theorem on compact sets.
-/

@[expose] public section


namespace TorchLean.UniversalApproximation


open MLP
open Activation (relu)

/-- Evaluate a single-hidden-layer ReLU MLP on a tensor input and return the scalar output. -/
def mlpEval {α : Type} [Storage α] [Context α] {n hidDim : Nat}
    (l1 : Linear α n hidDim) (l2 : Linear α hidDim 1)
    (x : Tensor α [n]) : α :=
  extractScalarOutput (MLP.forward l1 l2 x)

/--
The identity $\operatorname{ReLU}(u)-\operatorname{ReLU}(-u)=u$, used to represent affine maps
exactly with ReLU.
-/
theorem relu_sub_relu_neg (u : ℝ) : relu u - relu (-u) = u := by
  simp only [Activation.relu_eq_max]
  exact max_zero_sub_max_neg_zero_eq_self u

/--
First layer of the affine representation: for $u(x)=w\mathbin{\cdot}x+b$ it outputs
$[u(x),-u(x)]$.
-/
noncomputable def affineIdLayer1 {n : Nat} (w : Fin n → ℝ) (b : ℝ) : Linear ℝ n 2 :=
  { weights := Tensor.matrix (m := 2) (n := n) (fun i j => if i.1 = 0 then w j else -w j)
    bias := Tensor.ofFn (n := 2) (fun i => if i.1 = 0 then b else -b) }

/--
Output layer of the affine representation: on the hidden activations
$[\operatorname{ReLU}(u),\operatorname{ReLU}(-u)]$ it returns
$\operatorname{ReLU}(u)-\operatorname{ReLU}(-u)=u$.
-/
noncomputable def affineIdLayer2 : Linear ℝ 2 1 :=
  { weights := Tensor.matrix (m := 1) (n := 2)
      (fun _ j => if j.1 = 0 then (1 : ℝ) else (-1 : ℝ))
    bias := Tensor.ofFn (n := 1) (fun _ => (0 : ℝ)) }

/--
Every affine map $x\mapsto w\mathbin{\cdot}x+b$ is exactly a width-2, 2-layer ReLU MLP:
`affineIdLayer1 w b` followed by `affineIdLayer2`.

The compact-set theorem uses it to turn affine forms into MLP evaluations.
-/
theorem mlpEval_affineIdLayer {n : Nat} (w : Fin n → ℝ) (b : ℝ) (x : Tensor ℝ [n]) :
    mlpEval (n := n) (hidDim := 2) (affineIdLayer1 (n := n) w b) affineIdLayer2 x =
      w ⬝ᵥ x.getScalar + b := by
  classical
  -- Unfold evaluation and rewrite the MLP as `linear ∘ relu ∘ linear`.
  unfold mlpEval
  rw [forward_def (n := n) (hidDim := 2)
        (l1 := affineIdLayer1 (n := n) w b) (l2 := affineIdLayer2) (x := x)]
  -- Compute the first linear layer output: `[u, -u]` where `u = w ⬝ᵥ x.getScalar + b`.
  have hz1 :
      Linear.forward (α := ℝ) (affineIdLayer1 (n := n) w b) x =
        Tensor.dim (fun i : Fin 2 =>
          Tensor.scalar
            (if i.1 = 0 then (w ⬝ᵥ x.getScalar + b) else -(w ⬝ᵥ x.getScalar + b))) := by
    unfold Linear.forward affineIdLayer1
    apply Tensor.ext_vector
    intro i
    rw [congrFun (Tensor.getScalar_add _ _) i]
    rw [Tensor.getScalar_contract_matVec_eq_sum]
    fin_cases i <;>
      simp [Tensor.matrix, Tensor.get2, Tensor.ofFn, dotProduct, add_comm]
  -- After ReLU the hidden layer holds `relu u` and `relu (-u)`.
  have ha1 :
      Tensor.relu (α := ℝ) (s := .dim 2 .scalar)
          (Linear.forward (α := ℝ) (affineIdLayer1 (n := n) w b) x) =
        Tensor.dim (fun i : Fin 2 =>
          Tensor.scalar (if i.1 = 0 then relu (w ⬝ᵥ x.getScalar + b)
            else relu (-b + -(w ⬝ᵥ x.getScalar)))) := by
    rw [hz1]
    apply Tensor.ext_vector
    intro i
    fin_cases i <;>
      simp [Tensor.relu, Activation.relu_eq_max]
  -- The output layer returns `relu u - relu (-u)`, which is `u` by `relu_sub_relu_neg`.
  have hy :
      Linear.forward (α := ℝ) affineIdLayer2
          (Tensor.relu (α := ℝ) (s := .dim 2 .scalar)
            (Linear.forward (α := ℝ) (affineIdLayer1 (n := n) w b) x)) =
        Tensor.dim (fun _ : Fin 1 => Tensor.scalar (w ⬝ᵥ x.getScalar + b)) := by
    rw [ha1]
    dsimp only [Linear.forward, affineIdLayer2]
    apply Tensor.ext_vector
    intro i
    rw [congrFun (Tensor.getScalar_add _ _) i]
    rw [Tensor.getScalar_contract_matVec_eq_sum]
    fin_cases i
    have h := relu_sub_relu_neg (u := w ⬝ᵥ x.getScalar + b)
    simpa [Tensor.matrix, Tensor.get2, Tensor.ofFn, sub_eq_add_neg, neg_add, add_assoc,
      add_comm, add_left_comm, mul_assoc] using h
  -- Finish: `extractScalarOutput` picks the unique element of `Fin 1`.
  simp [extractScalarOutput, hy]

/-- Exact representability of coordinate projections $x\mapsto x_i$ by a width-$2$ ReLU MLP. -/
theorem mlpEval_coord {n : Nat} (i : Fin n) (x : Tensor ℝ [n]) :
    mlpEval (n := n) (hidDim := 2) (affineIdLayer1 (n := n) (Pi.single i 1) 0)
        affineIdLayer2 x =
      Tensor.getScalar x i := by
  rw [mlpEval_affineIdLayer, single_one_dotProduct, add_zero]

/-!
## Ridge lifting

Given a scalar-input MLP `(l1,l2)` and an affine scalar map
$u=w\mathbin{\cdot}x+c$, the lifted tensor-input MLP has the pre-activations of the scalar-input
one at $u$, so any scalar approximation result carries over to functions of one affine form
(“ridge functions”).
-/

/--
Lift a scalar-input first layer to a tensor-input first layer along a ridge direction.

Row `i` of the lifted weights is the scalar weight of `l1` at `i` times `w`, and the bias absorbs
that weight times `c`, so at input `x` the lifted layer computes what `l1` computes at
$u=w\mathbin{\cdot}x+c$.
-/
noncomputable def liftScalarLayer1
    {n hidDim : Nat} (l1 : Linear ℝ 1 hidDim) (w : Fin n → ℝ) (c : ℝ) : Linear ℝ n hidDim :=
  { weights := Tensor.matrix (m := hidDim) (n := n)
      (fun i j => Tensor.get2 l1.weights i 0 * w j)
    bias := Tensor.ofFn (n := hidDim)
      (fun i => Tensor.get2 l1.weights i 0 * c + Tensor.getScalar l1.bias i) }

/--
Lifting lemma: the lifted tensor-input MLP agrees with the scalar-input MLP evaluated at
$w\mathbin{\cdot}x+c$.
-/
theorem mlpEval_liftScalarLayer
    {n hidDim : Nat} (l1 : Linear ℝ 1 hidDim) (l2 : Linear ℝ hidDim 1)
    (w : Fin n → ℝ) (c : ℝ) (x : Tensor ℝ [n]) :
    mlpEval (n := n) (hidDim := hidDim) (liftScalarLayer1 (n := n) l1 w c) l2 x =
      mlpEvalScalar hidDim l1 l2 (w ⬝ᵥ x.getScalar + c) := by
  classical
  -- Expand both sides to `forward`, then use the `linear ∘ relu ∘ linear` form.
  unfold mlpEval mlpEvalScalar
  rw [forward_def (n := n) (hidDim := hidDim)
        (l1 := liftScalarLayer1 (n := n) l1 w c) (l2 := l2) (x := x)]
  rw [MLP.forward_def (hidDim := hidDim)
        (l1 := l1) (l2 := l2) (x := Tensor.singleton (w ⬝ᵥ x.getScalar + c))]
  -- Show the first linear layers agree coordinatewise.
  have hz1 :
      Linear.forward (α := ℝ) (liftScalarLayer1 (n := n) l1 w c) x =
        Linear.forward (α := ℝ) l1 (Tensor.singleton (w ⬝ᵥ x.getScalar + c)) := by
    unfold Linear.forward liftScalarLayer1
    apply Tensor.ext_vector
    intro i
    repeat' rw [congrFun (Tensor.getScalar_add _ _) i]
    repeat' rw [Tensor.getScalar_contract_matVec_eq_sum]
    simp [Tensor.matrix, Tensor.get2, Tensor.singleton, Tensor.ofFn, dotProduct,
      Finset.mul_sum, mul_add, mul_assoc, mul_left_comm, mul_comm, add_assoc, add_comm]
  -- Once the first layers agree (`hz1`), the rest of the network matches by unfolding.
  simp [hz1, extractScalarOutput, Tensor.relu]

end TorchLean.UniversalApproximation
