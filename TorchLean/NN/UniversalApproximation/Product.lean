/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.UniversalApproximation.TensorInput
public import TorchLean.Tensor.Contract

/-!
# Approximating multiplication with a 2-layer ReLU MLP (2D box)

On the box $[-M,M]^2$ with $M>0$, a single-hidden-layer ReLU MLP on `Tensor ℝ [2]` approximates
$(x_0,x_1)\mapsto x_0x_1$ uniformly to any accuracy (`relu_mul_box`). The
proof is constructive. It lifts one ReLU approximator of $u\mapsto u^2$, taken from the
one-dimensional theorem, along two ridge directions and joins the two copies with a small network
algebra (`appendLinear`, `combineOutput`).
-/

@[expose] public section


namespace TorchLean.UniversalApproximation


open MLP
open Activation (relu)

/-- The box `[-M,M]^n` as a subset of `Tensor ℝ [n]`. -/
noncomputable def boxN (n : Nat) (M : ℝ) : Set (Tensor ℝ [n]) :=
  fun x => ∀ i : Fin n, Tensor.getScalar x i ∈ Set.Icc (-M) M

/-- The target multiplication map: multiply the two coordinates. -/
noncomputable def mulFun (x : Tensor ℝ [2]) : ℝ :=
  x.getScalar 0 * x.getScalar 1

/-- Ridge direction with `wPlus ⬝ᵥ x.getScalar` equal to the sum of the two coordinates. -/
noncomputable def wPlus : Fin 2 → ℝ := fun _ => 1

/-- Ridge direction with `wMinus ⬝ᵥ x.getScalar` equal to the first coordinate minus the second. -/
noncomputable def wMinus : Fin 2 → ℝ := fun i => if i.1 = 0 then 1 else (-1 : ℝ)

/-- `wPlus ⬝ᵥ x.getScalar` is the sum of the two coordinates of `x`. -/
theorem dot_wPlus (x : Tensor ℝ [2]) :
    wPlus ⬝ᵥ x.getScalar = x.getScalar 0 + x.getScalar 1 := by
  classical
  simp [dotProduct, wPlus, Fin.sum_univ_two]

/-- `wMinus ⬝ᵥ x.getScalar` is the first coordinate of `x` minus the second, $x_0-x_1$. -/
theorem dot_wMinus (x : Tensor ℝ [2]) :
    wMinus ⬝ᵥ x.getScalar = x.getScalar 0 - x.getScalar 1 := by
  classical
  simp [dotProduct, wMinus, Fin.sum_univ_two, sub_eq_add_neg]

/-- Multiplication as a difference of squares: $xy=((x+y)^2-(x-y)^2)/4$. -/
theorem mul_identity (x y : ℝ) : x * y = ((x + y) * (x + y) - (x - y) * (x - y)) / 4 := by
  ring

/-- If `x ∈ boxN 2 M`, then $x_0+x_1\in[-2M,2M]$. -/
theorem dot_wPlus_mem_Icc {M : ℝ} {x : Tensor ℝ [2]} (hx : x ∈ boxN 2 M) :
    wPlus ⬝ᵥ x.getScalar ∈ Set.Icc (-2*M) (2*M) := by
  have hx0 : x.getScalar 0 ∈ Set.Icc (-M) M := hx 0
  have hx1 : x.getScalar 1 ∈ Set.Icc (-M) M := hx 1
  rw [dot_wPlus]
  exact ⟨by linarith [hx0.1, hx1.1], by linarith [hx0.2, hx1.2]⟩

/-- If `x ∈ boxN 2 M`, then $x_0-x_1\in[-2M,2M]$. -/
theorem dot_wMinus_mem_Icc {M : ℝ} {x : Tensor ℝ [2]} (hx : x ∈ boxN 2 M) :
    wMinus ⬝ᵥ x.getScalar ∈ Set.Icc (-2*M) (2*M) := by
  have hx0 : x.getScalar 0 ∈ Set.Icc (-M) M := hx 0
  have hx1 : x.getScalar 1 ∈ Set.Icc (-M) M := hx 1
  rw [dot_wMinus]
  exact ⟨by linarith [hx0.1, hx1.2], by linarith [hx0.2, hx1.1]⟩

-- ---------------------------------------------------------------------------
-- Tensor helpers: concatenating hidden units
-- ---------------------------------------------------------------------------

/--
Concatenate two tensors along the leading dimension, used here to append the hidden units of two
subnetworks.
-/
noncomputable def appendDim {α : Type} [Storage α] {m n : Nat} {s : Shape}
    (a : Tensor α (.dim m s)) (b : Tensor α (.dim n s)) : Tensor α (.dim (m+n) s) :=
  Tensor.dim (Fin.append a.unstack b.unstack)

/-- Append two first-layer linear specs by appending their weight and bias tensors. -/
noncomputable def appendLinear
    {inDim m n : Nat} (a : Linear ℝ inDim m) (b : Linear ℝ inDim n) :
    Linear ℝ inDim (m+n) :=
  { weights := appendDim a.weights b.weights
    bias := appendDim a.bias b.bias }

/--
Combine two scalar-output linear specs into one scalar-output spec on an appended hidden layer.

If the appended hidden vector is $[z_a;z_b]$, the resulting output layer computes
$\gamma+\alpha\,\mathrm{out}_a(z_a)+\beta\,\mathrm{out}_b(z_b)$.
-/
noncomputable def combineOutput
    {m n : Nat} (α β γ : ℝ) (a : Linear ℝ m 1) (b : Linear ℝ n 1) :
    Linear ℝ (m+n) 1 :=
  let c : Fin (m+n) → ℝ :=
    Fin.addCases (fun j : Fin m => α * Tensor.get2 a.weights 0 j)
      (fun j : Fin n => β * Tensor.get2 b.weights 0 j)
  { weights := Tensor.matrix (m := 1) (n := m + n) (fun _ j => c j)
    bias := Tensor.ofFn (n := 1)
      (fun _ => γ + α * extractScalarOutput a.bias + β * extractScalarOutput b.bias) }

/-- A `1 × n` matrix times `v` is a one-entry vector holding the dot product of the row with
`v`. -/
theorem contract_matVec_one_row {n : Nat} (A : Tensor ℝ [1, n]) (v : Tensor ℝ [n]) :
    Tensor.contract A v _ _ [1] .matVec =
      Tensor.dim (fun _ : Fin 1 =>
        Tensor.scalar (∑ j : Fin n, Tensor.get2 A 0 j * Tensor.getScalar v j)) := by
  classical
  apply Tensor.ext_vector
  intro i
  fin_cases i
  simp [Tensor.getScalar_contract_matVec_eq_sum]

/--
`mlpEval` equals the output bias plus $\sum_j w_j\operatorname{ReLU}(z_j)$, where $w$ is the
output row and $z$ is the first layer's output.

`mlpEval_appendLinear` uses this form to show that `appendLinear` and `combineOutput` build
affine combinations of subnetworks.
-/
theorem mlpEval_eq_bias_sum
    {inDim hidDim : Nat} (l1 : Linear ℝ inDim hidDim) (l2 : Linear ℝ hidDim 1)
    (x : Tensor ℝ [inDim]) :
    mlpEval (n := inDim) (hidDim := hidDim) l1 l2 x =
      extractScalarOutput l2.bias
        + ∑ j : Fin hidDim, (Tensor.get2 l2.weights 0 j) *
            relu (Tensor.getScalar (Linear.forward (α := ℝ) l1 x) j) := by
  classical
  unfold mlpEval
  rw [forward_def (n := inDim) (hidDim := hidDim) (l1 := l1) (l2 := l2) (x :=
    x)]
  dsimp
  have hmv :
      Tensor.contract l2.weights
          (Tensor.relu (α := ℝ) (s := .dim hidDim .scalar) (Linear.forward (α := ℝ) l1
            x)) _ _ [1] .matVec =
        Tensor.dim (fun _ : Fin 1 =>
          Tensor.scalar (∑ j : Fin hidDim,
            Tensor.get2 l2.weights 0 j *
              Tensor.getScalar
                (Tensor.relu (α := ℝ) (s := .dim hidDim .scalar)
                  (Linear.forward (α := ℝ) l1 x)) j)) := by
    simpa using contract_matVec_one_row (A := l2.weights)
      (v := Tensor.relu (α := ℝ) (s := .dim hidDim .scalar) (Linear.forward (α := ℝ) l1
        x))
  change extractScalarOutput
      ((Tensor.contract l2.weights
          (Tensor.relu (Linear.forward (α := ℝ) l1 x)) _ _ [1] .matVec) +
        l2.bias) = _
  rw [hmv]
  unfold extractScalarOutput
  rw [congrFun (Tensor.getScalar_add _ _) ⟨0, by decide⟩]
  simp [Tensor.relu, add_comm]

/-- The first `m` outputs of `appendLinear l1a l1b` are the outputs of `l1a`. -/
theorem appendLinear_left
    {inDim m n : Nat} (l1a : Linear ℝ inDim m) (l1b : Linear ℝ inDim n)
    (x : Tensor ℝ [inDim]) (i : Fin m) :
    Tensor.getScalar
        (Linear.forward (α := ℝ) (appendLinear (inDim := inDim) l1a l1b) x) (Fin.castAdd n i)
      =
    Tensor.getScalar (Linear.forward (α := ℝ) l1a x) i := by
  classical
  unfold Linear.forward
  rw [congrFun (Tensor.getScalar_add _ _) (Fin.castAdd n i)]
  rw [congrFun (Tensor.getScalar_add _ _) i]
  rw [Tensor.getScalar_contract_matVec_eq_sum,
    Tensor.getScalar_contract_matVec_eq_sum]
  simp [appendLinear, appendDim, Tensor.get2, Tensor.getScalar,
    Fin.append]

/-- The last `n` outputs of `appendLinear l1a l1b` are the outputs of `l1b`. -/
theorem appendLinear_right
    {inDim m n : Nat} (l1a : Linear ℝ inDim m) (l1b : Linear ℝ inDim n)
    (x : Tensor ℝ [inDim]) (i : Fin n) :
    Tensor.getScalar
        (Linear.forward (α := ℝ) (appendLinear (inDim := inDim) l1a l1b) x) (Fin.natAdd m i)
      =
    Tensor.getScalar (Linear.forward (α := ℝ) l1b x) i := by
  classical
  unfold Linear.forward
  rw [congrFun (Tensor.getScalar_add _ _) (Fin.natAdd m i)]
  rw [congrFun (Tensor.getScalar_add _ _) i]
  rw [Tensor.getScalar_contract_matVec_eq_sum,
    Tensor.getScalar_contract_matVec_eq_sum]
  simp [appendLinear, appendDim, Tensor.get2, Tensor.getScalar,
    Fin.append]

/--
Appending first layers with `appendLinear` and wiring the output with `combineOutput` gives
the affine combination $\gamma+\alpha\,\mathrm{net}_a(x)+\beta\,\mathrm{net}_b(x)$ of the two
networks.
-/
theorem mlpEval_appendLinear
    {inDim m n : Nat} (l1a : Linear ℝ inDim m) (l1b : Linear ℝ inDim n)
    (l2a : Linear ℝ m 1) (l2b : Linear ℝ n 1)
    (α β γ : ℝ) (x : Tensor ℝ [inDim]) :
    mlpEval (n := inDim) (hidDim := m+n)
        (appendLinear (inDim := inDim) l1a l1b)
        (combineOutput (m := m) (n := n) α β γ l2a l2b) x
      =
    γ + α * mlpEval (n := inDim) (hidDim := m) l1a l2a x
      + β * mlpEval (n := inDim) (hidDim := n) l1b l2b x := by
  classical
  -- Expand all three evaluations into bias plus a sum over hidden units.
  rw [mlpEval_eq_bias_sum (l1 := appendLinear (inDim := inDim) l1a l1b)
        (l2 := combineOutput (m := m) (n := n) α β γ l2a l2b) (x := x)]
  rw [mlpEval_eq_bias_sum (l1 := l1a) (l2 := l2a) (x := x)]
  rw [mlpEval_eq_bias_sum (l1 := l1b) (l2 := l2b) (x := x)]
  -- Split the combined sum over `Fin (m+n)` into left/right parts.
  -- The combined weights are `Fin.addCases` and the combined hidden pre-activations come from
  -- `appendLinear`.
  have hsplit :
      (∑ j : Fin (m + n),
          Tensor.get2 (combineOutput (m := m) (n := n) α β γ l2a l2b).weights 0 j *
            relu (Tensor.getScalar
              (Linear.forward (α := ℝ) (appendLinear (inDim := inDim) l1a l1b) x) j))
        =
      (∑ j : Fin m,
          (α * Tensor.get2 l2a.weights 0 j) *
            relu (Tensor.getScalar (Linear.forward (α := ℝ) l1a x) j))
      +
      (∑ j : Fin n,
          (β * Tensor.get2 l2b.weights 0 j) *
            relu (Tensor.getScalar (Linear.forward (α := ℝ) l1b x) j)) := by
    classical
    -- Use `Fin.sum_univ_add` and then simplify the `Fin.addCases` selectors.
    have hsum :=
      (Fin.sum_univ_add (a := m) (b := n)
        (f := fun j : Fin (m+n) =>
          Tensor.get2 (combineOutput (m := m) (n := n) α β γ l2a l2b).weights 0 j *
            relu (Tensor.getScalar
              (Linear.forward (α := ℝ) (appendLinear (inDim := inDim) l1a l1b) x) j)))
    -- Resolve the `castAdd` and `natAdd` branches with the selector lemmas above.
    simpa [combineOutput, Tensor.matrix, Tensor.get2,
      appendLinear_left (l1a := l1a) (l1b := l1b) (x := x),
      appendLinear_right (l1a := l1a) (l1b := l1b) (x := x),
      Fin.addCases_left, Fin.addCases_right, mul_assoc, mul_left_comm, mul_comm] using hsum
  -- Factor out the scalars `α`/`β` from the two sums.
  let sumA : ℝ :=
    ∑ j : Fin m,
      Tensor.get2 l2a.weights 0 j *
        relu (Tensor.getScalar (Linear.forward (α := ℝ) l1a x) j)
  let sumB : ℝ :=
    ∑ j : Fin n,
      Tensor.get2 l2b.weights 0 j *
        relu (Tensor.getScalar (Linear.forward (α := ℝ) l1b x) j)
  have hsumA :
      (∑ j : Fin m,
          (α * Tensor.get2 l2a.weights 0 j) *
            relu (Tensor.getScalar (Linear.forward (α := ℝ) l1a x) j))
        = α * sumA := by
    classical
    -- `∑ (α * t_j) = α * ∑ t_j` over `Fin m`.
    simpa [sumA, mul_assoc, mul_left_comm, mul_comm] using
      (Finset.mul_sum α (s := (Finset.univ : Finset (Fin m)))
          (f := fun j : Fin m =>
            Tensor.get2 l2a.weights 0 j *
              relu (Tensor.getScalar (Linear.forward (α := ℝ) l1a x) j))).symm
  have hsumB :
      (∑ j : Fin n,
          (β * Tensor.get2 l2b.weights 0 j) *
            relu (Tensor.getScalar (Linear.forward (α := ℝ) l1b x) j))
        = β * sumB := by
    classical
    simpa [sumB, mul_assoc, mul_left_comm, mul_comm] using
      (Finset.mul_sum β (s := (Finset.univ : Finset (Fin n)))
          (f := fun j : Fin n =>
            Tensor.get2 l2b.weights 0 j *
              relu (Tensor.getScalar (Linear.forward (α := ℝ) l1b x) j))).symm
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
For $M>0$ and every $\varepsilon>0$, a single-hidden-layer ReLU MLP approximates multiplication
within $\varepsilon$ everywhere on the box $[-M,M]^2$.

The construction follows the classical reduction $xy=((x+y)^2-(x-y)^2)/4$. A one-dimensional
ReLU approximator of $u\mapsto u^2$ on $[-2M,2M]$, accurate to $2\varepsilon$, is lifted along
the ridge directions `wPlus` and `wMinus`, and the factor $1/4$ brings the combined error below
$\varepsilon$.
-/
theorem relu_mul_box
    {M : ℝ} (hM : 0 < M) :
    ∀ ε > 0, ∃ (hidDim : ℕ) (l1 : Linear ℝ 2 hidDim) (l2 : Linear ℝ hidDim 1),
      ∀ x ∈ boxN 2 M, |mulFun x - mlpEval (n := 2) (hidDim := hidDim) l1 l2 x| < ε := by
  intro ε hε
  -- Step 1: approximate `u ↦ u * u` on `[-2M,2M]` with error `2ε`.
  have h_ab : (-2*M) < (2*M) := by nlinarith
  have hL : 0 < (4*M) := by nlinarith
  have h_lip :
      ∀ x ∈ Set.Icc (-2*M) (2*M), ∀ y ∈ Set.Icc (-2*M) (2*M),
        |(x*x) - (y*y)| ≤ (4*M) * |x - y| := by
    intro x hx y hy
    have hmax : max |x| |y| ≤ 2 * M :=
      max_le (abs_le.2 ⟨by linarith [hx.1], hx.2⟩) (abs_le.2 ⟨by linarith [hy.1], hy.2⟩)
    have h : |x ^ 2 - y ^ 2| ≤ |x - y| * 2 * max |x| |y| := by
      simpa using abs_pow_sub_pow_le x y 2
    calc |(x*x) - (y*y)| = |x ^ 2 - y ^ 2| := by rw [sq, sq]
      _ ≤ |x - y| * 2 * max |x| |y| := h
      _ ≤ |x - y| * 2 * (2 * M) := mul_le_mul_of_nonneg_left hmax (by positivity)
      _ = (4*M) * |x - y| := by ring
  rcases relu_Icc (f := fun u => u*u) (a := -2*M) (b := 2*M) (L := 4*M)
      h_ab hL h_lip (2 * ε) (by linarith) with ⟨hidSq, l1Sq, l2Sq, hSq⟩
  -- Step 2: lift to `u = x₀ + x₁` and `u = x₀ - x₁`.
  let l1Plus : Linear ℝ 2 hidSq := liftScalarLayer1 (n := 2) l1Sq wPlus 0
  let l1Minus : Linear ℝ 2 hidSq := liftScalarLayer1 (n := 2) l1Sq wMinus 0
  -- Step 3: combine the two lifted square nets to form a product approximator.
  let l1Prod : Linear ℝ 2 (hidSq + hidSq) := appendLinear (inDim := 2) l1Plus l1Minus
  let l2Prod : Linear ℝ (hidSq + hidSq) 1 :=
    combineOutput (m := hidSq) (n := hidSq) (α := (1/4 : ℝ)) (β := (-1/4 : ℝ)) (γ := 0) l2Sq l2Sq
  refine ⟨hidSq + hidSq, l1Prod, l2Prod, ?_⟩
  intro x hx
  -- Both ridge inputs lie in `[-2M,2M]`, where the 1D approximation holds.
  have hx_plus : wPlus ⬝ᵥ x.getScalar ∈ Set.Icc (-2*M) (2*M) := dot_wPlus_mem_Icc hx
  have hx_minus : wMinus ⬝ᵥ x.getScalar ∈ Set.Icc (-2*M) (2*M) := dot_wMinus_mem_Icc hx
  -- Use the lifted equality to rewrite the lifted nets as 1D evaluations.
  have hplus_eval :
      mlpEval (n := 2) (hidDim := hidSq) l1Plus l2Sq x =
        mlpEvalScalar hidSq l1Sq l2Sq (wPlus ⬝ᵥ x.getScalar) := by
    simpa [l1Plus, add_comm] using
      (mlpEval_liftScalarLayer (n := 2) (hidDim := hidSq) l1Sq l2Sq wPlus 0 x)
  have hminus_eval :
      mlpEval (n := 2) (hidDim := hidSq) l1Minus l2Sq x =
        mlpEvalScalar hidSq l1Sq l2Sq (wMinus ⬝ᵥ x.getScalar) := by
    simpa [l1Minus, add_comm] using
      (mlpEval_liftScalarLayer (n := 2) (hidDim := hidSq) l1Sq l2Sq wMinus 0 x)
  -- Evaluate the combined network as a linear combination of the two lifted nets.
  have hcomb :
      mlpEval (n := 2) (hidDim := hidSq + hidSq) l1Prod l2Prod x
        =
      (1/4 : ℝ) * mlpEval (n := 2) (hidDim := hidSq) l1Plus l2Sq x
        + (-1/4 : ℝ) * mlpEval (n := 2) (hidDim := hidSq) l1Minus l2Sq x := by
    -- `mlpEval_appendLinear` gives the general linear combination form.
    have := mlpEval_appendLinear (inDim := 2) (m := hidSq) (n := hidSq)
      (l1a := l1Plus) (l1b := l1Minus) (l2a := l2Sq) (l2b := l2Sq)
      (α := (1/4 : ℝ)) (β := (-1/4 : ℝ)) (γ := 0) (x := x)
    simpa [l1Prod, l2Prod, add_assoc, add_left_comm, add_comm] using this
  -- Apply the square approximation bounds.
  have hsq_plus :
      |(wPlus ⬝ᵥ x.getScalar) * (wPlus ⬝ᵥ x.getScalar) -
          mlpEvalScalar hidSq l1Sq l2Sq (wPlus ⬝ᵥ x.getScalar)| < 2 * ε :=
    hSq (wPlus ⬝ᵥ x.getScalar) hx_plus
  have hsq_minus :
      |(wMinus ⬝ᵥ x.getScalar) * (wMinus ⬝ᵥ x.getScalar) -
          mlpEvalScalar hidSq l1Sq l2Sq (wMinus ⬝ᵥ x.getScalar)| < 2 * ε :=
    hSq (wMinus ⬝ᵥ x.getScalar) hx_minus
  -- Finish via `xy = ((x+y)^2 - (x-y)^2)/4` and the triangle inequality.
  have hmul : mulFun x = ((wPlus ⬝ᵥ x.getScalar) * (wPlus ⬝ᵥ x.getScalar) -
      (wMinus ⬝ᵥ x.getScalar) * (wMinus ⬝ᵥ x.getScalar)) / 4 := by
    -- Rewrite the dot products as coordinate sums and use the algebraic identity.
    rw [dot_wPlus, dot_wMinus]
    exact mul_identity (x.getScalar 0) (x.getScalar 1)
  -- Replace `mulFun` and the network output by the “difference of squares” forms.
  rw [hmul, hcomb, hplus_eval, hminus_eval]
  -- Name the two square-approximation errors `e1` and `e2`.
  set e1 := (wPlus ⬝ᵥ x.getScalar) * (wPlus ⬝ᵥ x.getScalar) -
    mlpEvalScalar hidSq l1Sq l2Sq (wPlus ⬝ᵥ x.getScalar) with he1
  set e2 := (wMinus ⬝ᵥ x.getScalar) * (wMinus ⬝ᵥ x.getScalar) -
    mlpEvalScalar hidSq l1Sq l2Sq (wMinus ⬝ᵥ x.getScalar) with he2
  -- The remaining difference is `(e1 - e2)/4`.
  have hrew :
      ((wPlus ⬝ᵥ x.getScalar) * (wPlus ⬝ᵥ x.getScalar) -
          (wMinus ⬝ᵥ x.getScalar) * (wMinus ⬝ᵥ x.getScalar)) / 4
        - ((1 / 4 : ℝ) * mlpEvalScalar hidSq l1Sq l2Sq (wPlus ⬝ᵥ x.getScalar) +
            (-1 / 4 : ℝ) * mlpEvalScalar hidSq l1Sq l2Sq (wMinus ⬝ᵥ x.getScalar))
      =
      (e1 - e2) / 4 := by
    subst e1 e2
    ring
  have habs : |(e1 - e2) / 4| = |e1 - e2| / 4 := by
    simp [abs_div]
  have he1lt : |e1| < 2 * ε := by simpa [he1] using hsq_plus
  have he2lt : |e2| < 2 * ε := by simpa [he2] using hsq_minus
  -- `|e1 - e2| ≤ |e1| + |e2| < 4ε`, and dividing by `4` gives the claim.
  rw [hrew, habs]
  linarith [abs_sub e1 e2]

end TorchLean.UniversalApproximation
