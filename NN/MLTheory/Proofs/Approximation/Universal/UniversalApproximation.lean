/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Algebra.Order.Floor.Semiring
public import Mathlib.Data.Real.Basic
public import NN.Spec.Core.Tensor
public import NN.Runtime.Context
public import NN.Spec.Layers.Activation
public import NN.Spec.Layers.Linear
public import NN.Spec.Models.Mlp
import Mathlib.Tactic.Linarith

/-!
# Universal approximation (1D, constructive)

On a compact interval `I = [a,b]`, any Lipschitz function `f : ℝ → ℝ` can be uniformly approximated
by a single-hidden-layer ReLU network (a 2-layer MLP).

This file formalizes the classic constructive proof strategy:

- approximate `f` by a polygonal function on a uniform grid,
- express that polygonal function as an affine term plus a finite sum of hinges `relu(x - tᵢ)`,
- package the hinge representation as TorchLean's spec-level 2-layer MLP (`NN.Spec.Models.Mlp`).

## Main result

- `relu_universal_approximation_Icc`: existence of a 2-layer ReLU MLP approximator on `Set.Icc a b`.

## References

- Leshno, Lin, Pinkus, Schocken (1993), *Multilayer feedforward networks with a nonpolynomial
  activation function can approximate any function*.
- Yarotsky (2017), *Error bounds for approximations with deep ReLU networks*.
- Pinkus (1999), *Approximation theory of the MLP model in neural networks*.
-/

@[expose] public section

namespace NN.MLTheory.Proofs.UniversalApproximation

  open _root_.Spec
  open _root_.Spec.Tensor
  open Examples

  /-- Shorthand for `relu` in this development, using TorchLean’s spec semantics. -/
  abbrev relu (x : ℝ) : ℝ := Activation.Math.reluSpec x

  /-- If the knot `t` is to the left of `x`, the hinge `relu (x - t)` equals `x - t`. -/
  lemma relu_sub_eq_of_le {x t : ℝ} (h : t ≤ x) : relu (x - t) = x - t := by
    have : 0 ≤ x - t := sub_nonneg.mpr h
    simp [relu, Activation.Math.reluSpec, max_eq_left this]

  /-- If `x` is to the left of the knot `t`, the hinge `relu (x - t)` is zero. -/
  lemma relu_sub_eq_zero_of_le {x t : ℝ} (h : x ≤ t) : relu (x - t) = 0 := by
    have : x - t ≤ 0 := sub_nonpos.mpr h
    simp [relu, Activation.Math.reluSpec, max_eq_right this]

/-- Extract scalar from a length-1 tensor. -/
def extractScalarOutput (t : Tensor ℝ (.dim 1 .scalar)) : ℝ :=
  match t with
  | .dim f => toScalar (f ⟨0, by norm_num⟩)

/-- Evaluate a 2-layer ReLU MLP on a scalar input. -/
noncomputable def mlpEval1d (hidDim : ℕ)
    (l1 : LinearSpec ℝ 1 hidDim) (l2 : LinearSpec ℝ hidDim 1) (x : ℝ) : ℝ :=
  extractScalarOutput (Examples.mlpForward l1 l2 (Tensor.singleton x))

/-- TorchLean's MLP forward pass is exactly `linear ∘ relu ∘ linear`. -/
lemma mlp_forward_eq_linear_relu_linear {hidDim : ℕ}
    (l1 : LinearSpec ℝ 1 hidDim) (l2 : LinearSpec ℝ hidDim 1) (x : Tensor ℝ (.dim 1 .scalar)) :
    Examples.mlpForward l1 l2 x =
      let z1 := Spec.linearSpec (α := ℝ) l1 x
      let a1 := Activation.reluSpec z1
      Spec.linearSpec (α := ℝ) l2 a1 := by
  simpa [Examples.mlpForward] using (Examples.mlp_spec_forward_eq (α := ℝ) l1 l2 x)

/--
Move a scalar initial accumulator out of a left fold that only adds terms.

This is bookkeeping for converting TorchLean's list-fold tensor semantics into Mathlib finite
sums.
-/
lemma foldl_add_init {α : Type} (l : List α) (f : α → ℝ) (a : ℝ) :
    l.foldl (fun acc x => acc + f x) a = a + l.foldl (fun acc x => acc + f x) 0 := by
  induction l generalizing a with
  | nil =>
    simp
  | cons x xs ih =>
    -- Expand both sides one step and use the induction hypothesis twice.
    -- `foldl` on a cons updates the accumulator once and recurses.
    simp [List.foldl]
    have h1 := ih (a := a + f x)
    have h2 := ih (a := f x)
    -- Rewrite using `h1`/`h2` and reassociate.
    -- After rewriting, both sides become `a + f x + foldl ... 0 xs`.
    simp [h1, h2, add_assoc]

/-- Convert the `List.finRange` fold used in `matVecMulSpec` into a `Finset.univ` sum. -/
lemma finRange_foldl_add (n : ℕ) (f : Fin n → ℝ) :
    (List.finRange n).foldl (fun acc i => acc + f i) 0 = ∑ i : Fin n, f i := by
  classical
  induction n with
  | zero =>
    simp
  | succ n ih =>
    -- Unfold `finRange (n+1)` and compute `foldl` step-by-step.
    -- Then reduce to the induction hypothesis on the tail.
    have :
        (List.finRange (n + 1)).foldl (fun acc i => acc + f i) 0 =
          f 0 + (List.finRange n).foldl (fun acc i => acc + f i.succ) 0 := by
      -- `finRange (n+1) = 0 :: (finRange n).map Fin.succ`
      -- and `foldl` over a `map` composes the element function.
      calc
        (List.finRange (n + 1)).foldl (fun acc i => acc + f i) 0
            = ((0 : Fin (n + 1)) :: (List.finRange n).map Fin.succ).foldl (fun acc i => acc + f i) 0
              := by
              simp [List.finRange_succ]
        _ = ((List.finRange n).map Fin.succ).foldl (fun acc i => acc + f i) (f 0) := by
              simp [List.foldl]
        _ = (List.finRange n).foldl (fun acc i => acc + f i.succ) (f 0) := by
              simp [List.foldl_map]
        _ = f 0 + (List.finRange n).foldl (fun acc i => acc + f i.succ) 0 := by
              exact foldl_add_init (List.finRange n) (fun i : Fin n => f i.succ) (f 0)
    -- Apply IH to the shifted function.
    rw [this, ih (fun i : Fin n => f i.succ), Fin.sum_univ_succ]

/-- `vectorN` is the dependent-tensor vector constructor expanded pointwise. -/
lemma vectorN_eq_dim {α : Type} [Zero α] (n : ℕ) (f : Fin n → α) :
    vectorN n f = Tensor.dim (fun i : Fin n => Tensor.scalar (f i)) := by
  cases n with
  | zero =>
    apply congrArg Tensor.dim
    funext i
    exact (Fin.elim0 i)
  | succ n =>
    simp [vectorN]

/-- First real hinge layer: hidden unit `i` computes `x - tᵢ` before ReLU. -/
noncomputable def hingeLayer1 (n : ℕ) (t : Fin n → ℝ) : LinearSpec ℝ 1 n :=
  { weights := matrixMN n 1 (fun _ _ => (1 : ℝ))
    bias := vectorN n (fun i => -t i) }

/-- Second real hinge layer: sum hidden activations with coefficients `cᵢ` and bias `b`. -/
noncomputable def hingeLayer2 (n : ℕ) (c : Fin n → ℝ) (b : ℝ) : LinearSpec ℝ n 1 :=
  { weights := matrixMN 1 n (fun _ j => c j)
    bias := vectorN 1 (fun _ => b) }

/-- Real hinge network `b + Σᵢ cᵢ ReLU(x - tᵢ)`. -/
noncomputable def hingeFun (n : ℕ) (t : Fin n → ℝ) (c : Fin n → ℝ) (b x : ℝ) : ℝ :=
  b + ∑ i : Fin n, c i * relu (x - t i)

/-- Fold lemma matching the scalar tensor accumulator used in `matVecMulSpec`. -/
lemma finRange_foldl_add_scalar (n : ℕ) (f : Fin n → ℝ) :
    (List.finRange n).foldl
        (fun (acc : Tensor ℝ .scalar) (k : Fin n) =>
          match acc with
          | Tensor.scalar s => Tensor.scalar (s + f k))
        (Tensor.scalar 0)
      =
      Tensor.scalar (∑ k : Fin n, f k) := by
  classical
  -- First, reduce the tensor fold to a scalar fold.
  have h_scalar :
      ∀ s0 : ℝ,
        (List.finRange n).foldl
            (fun (acc : Tensor ℝ .scalar) (k : Fin n) =>
              match acc with
              | Tensor.scalar s => Tensor.scalar (s + f k))
            (Tensor.scalar s0)
          =
          Tensor.scalar ((List.finRange n).foldl (fun (s : ℝ) (k : Fin n) => s + f k) s0) := by
    intro s0
    induction (List.finRange n) generalizing s0 with
    | nil =>
      simp
    | cons k ks ih =>
      simp [List.foldl, ih]
  -- Then use `finRange_foldl_add` on the scalar fold.
  rw [h_scalar 0]
  congr 1
  simpa using (finRange_foldl_add n f)

/-- Matrix-vector multiply for a one-row matrix is the expected finite dot product. -/
lemma mat_vec_mul_spec_matrixMN_vector (n : ℕ) (c v : Fin n → ℝ) :
    matVecMulSpec (matrixMN 1 n (fun _ j => c j))
        (Tensor.dim (fun j => Tensor.scalar (v j))) =
      Tensor.dim (fun _ => Tensor.scalar (∑ j : Fin n, c j * v j)) := by
  classical
  -- Reduce to pointwise equality on the unique index of `Fin 1`, then compute the dot product.
  apply congrArg Tensor.dim
  funext i
  fin_cases i
  -- Unfold everything; the goal becomes a `List.foldl` over `Fin n`.
  simp
  -- Convert the fold step (which is an aux matcher generated inside `mat_vec_mul_spec`) into the
  -- scalar-accumulator form, then apply `finRange_foldl_add_scalar`.
  have hfold :
      List.foldl
          (fun (acc : Tensor ℝ .scalar) (k : Fin n) =>
            Spec.matVecMulSpec.match_1 (α := ℝ) (motive := fun _ _ _ => Tensor ℝ .scalar)
              acc (Tensor.scalar (c k)) (Tensor.scalar (v k))
              (fun s ak vk => Tensor.scalar (s + ak * vk)))
          (Tensor.scalar 0) (List.finRange n) =
        List.foldl
          (fun (acc : Tensor ℝ .scalar) (k : Fin n) =>
            match acc with
            | Tensor.scalar s => Tensor.scalar (s + c k * v k))
          (Tensor.scalar 0) (List.finRange n) := by
    refine List.foldl_ext
        (f := fun (acc : Tensor ℝ .scalar) (k : Fin n) =>
          Spec.matVecMulSpec.match_1 (α := ℝ) (motive := fun _ _ _ => Tensor ℝ .scalar)
            acc (Tensor.scalar (c k)) (Tensor.scalar (v k))
            (fun s ak vk => Tensor.scalar (s + ak * vk)))
        (g := fun (acc : Tensor ℝ .scalar) (k : Fin n) =>
          match acc with
          | Tensor.scalar s => Tensor.scalar (s + c k * v k))
        (a := Tensor.scalar 0) ?_
    intro acc k _
    cases acc
    simp
  exact hfold.trans (finRange_foldl_add_scalar n (fun j : Fin n => c j * v j))

/-- Matrix-vector multiply by the all-ones column extracts the scalar input into every hidden unit. -/
lemma mat_vec_mul_spec_matrixMN_singleton (n : ℕ) (x : ℝ) :
    matVecMulSpec (matrixMN n 1 (fun _ _ => (1 : ℝ))) (Tensor.singleton x) =
      Tensor.dim (fun _ : Fin n => Tensor.scalar x) := by
  classical
  apply congrArg Tensor.dim
  funext i
  -- Unfold and compute the unique dot product over `Fin 1`.
  simp [List.finRange_succ, List.foldl]

/--
The explicit two-layer network built from `hingeLayer1` and `hingeLayer2` computes `hingeFun`.

This is the main semantic bridge from the approximation-theory hinge representation to TorchLean's
spec-level MLP model.
-/
lemma mlp_eval_1d_hinge (n : ℕ) (t : Fin n → ℝ) (c : Fin n → ℝ) (b x : ℝ) :
    mlpEval1d n (hingeLayer1 n t) (hingeLayer2 n c b) x = hingeFun n t c b x := by
  classical
  -- Unfold the definition of `mlp_eval_1d` and rewrite `mlp_forward`.
  unfold mlpEval1d hingeFun extractScalarOutput
  -- Avoid `simp` here (it unfolds too much under matchers); rewrite `mlp_forward` explicitly.
  rw [mlp_forward_eq_linear_relu_linear (l1 := hingeLayer1 n t) (l2 := hingeLayer2 n c b) (x :=
    Tensor.singleton x)]
  -- Reduce the let-bindings introduced by `mlp_forward_eq_linear_relu_linear`.
  dsimp
  -- Compute the first linear layer: `x ↦ (x - t i)`.
  have hz1 :
      Spec.linearSpec (α := ℝ) (hingeLayer1 n t) (Tensor.singleton x) =
        Tensor.dim (fun i : Fin n => Tensor.scalar (x - t i)) := by
    unfold hingeLayer1 Spec.linearSpec
    -- `mat_vec_mul_spec` yields the constant vector `x`; then add the bias `-t`.
    rw [mat_vec_mul_spec_matrixMN_singleton]
    simp [vectorN_eq_dim, addSpec, Tensor.map2Spec, sub_eq_add_neg]
  -- Apply ReLU pointwise.
  have ha1 :
      Activation.reluSpec (α := ℝ) (s := .dim n .scalar)
          (Spec.linearSpec (α := ℝ) (hingeLayer1 n t) (Tensor.singleton x)) =
        Tensor.dim (fun i : Fin n => Tensor.scalar (relu (x - t i))) := by
    simp [hz1, Activation.reluSpec, Tensor.mapSpec, relu]
  -- Compute the second linear layer as a sum of hinges.
  have hy :
      Spec.linearSpec (α := ℝ) (hingeLayer2 n c b)
          (Activation.reluSpec (α := ℝ) (s := .dim n .scalar)
            (Spec.linearSpec (α := ℝ) (hingeLayer1 n t) (Tensor.singleton x))) =
        Tensor.dim (fun _ : Fin 1 => Tensor.scalar ((∑ i : Fin n, c i * relu (x - t i)) + b)) := by
    -- Rewrite the activation output first, so unfolding `Spec.linear_spec` doesn't unfold the inner
    -- layer.
    rw [ha1]
    unfold hingeLayer2 Spec.linearSpec
    -- `mat_vec_mul_spec_matrixMN_vector` gives the dot product as a `Finset.univ` sum.
    have hmv :
        matVecMulSpec (matrixMN 1 n (fun _ j => c j))
            (Tensor.dim (fun j => Tensor.scalar (relu (x - t j)))) =
          Tensor.dim (fun _ : Fin 1 => Tensor.scalar (∑ j : Fin n, c j * relu (x - t j))) := by
      simpa using mat_vec_mul_spec_matrixMN_vector n c (fun j => relu (x - t j))
    rw [hmv]
    simp [vectorN_eq_dim, addSpec, Tensor.map2Spec, add_comm]
  -- Extract the scalar output (the unique element of `Fin 1`) and reorder `b + sum`.
  -- `Fin 1` has a unique element, so `fin_cases` reduces the extracted component.
  simp [hy, Tensor.toScalar, add_comm]

/--
1D Universal Approximation (ReLU, one hidden layer).

This is the classic constructive proof:
Lipschitz continuity on `[a,b]` + uniform partition + piecewise-linear interpolation,
then represent the interpolant as a finite linear combination of hinges `relu(x - t_i)`.
-/
  theorem relu_universal_approximation_Icc_hinge {f : ℝ → ℝ} {a b L : ℝ}
      (h_ab : a < b) (hL : 0 < L)
      (h_lip : ∀ x ∈ Set.Icc a b, ∀ y ∈ Set.Icc a b, |f x - f y| ≤ L * |x - y|) :
      ∀ ε > 0, ∃ (hidDim : ℕ) (t : Fin hidDim → ℝ) (c : Fin hidDim → ℝ),
        ∀ x ∈ Set.Icc a b, |f x - hingeFun hidDim t c (f a) x| < ε := by
    intro ε hε
    classical
    have hba : 0 < b - a := sub_pos.mpr h_ab
    have hεhalf : 0 < ε / 2 := by nlinarith
    have hprod : 0 < L * (b - a) := by nlinarith [hL, hba]
    have hε' : 0 < (ε / 2) / (L * (b - a)) := div_pos hεhalf hprod
    rcases exists_nat_one_div_lt hε' with ⟨n, hn⟩
    let N : ℕ := n + 1
    have hNpos_nat : 0 < N := Nat.succ_pos n
    have hNpos : 0 < (N : ℝ) := by exact_mod_cast hNpos_nat
    have hNne : (N : ℝ) ≠ 0 := ne_of_gt hNpos
    let δ : ℝ := (b - a) / (N : ℝ)
    have hδpos : 0 < δ := div_pos hba hNpos
    have hδnonneg : 0 ≤ δ := le_of_lt hδpos
    have hmesh : L * δ < ε / 2 := by
      have hn' :
          (L * (b - a)) * (1 / ((n : ℝ) + 1)) <
            (L * (b - a)) * ((ε / 2) / (L * (b - a))) :=
        mul_lt_mul_of_pos_left hn hprod
      have hnonzero : (L * (b - a)) ≠ 0 := ne_of_gt hprod
      have hright : (L * (b - a)) * ((ε / 2) / (L * (b - a))) = ε / 2 := by
        calc
          (L * (b - a)) * ((ε / 2) / (L * (b - a))) =
              (L * (b - a)) * (ε / 2) / (L * (b - a)) := by
            simp [mul_div_assoc']
          _ = ε / 2 := by
            simpa using (mul_div_cancel_left₀ (ε / 2) hnonzero)
      have hleft : (L * (b - a)) * (1 / ((n : ℝ) + 1)) = (L * (b - a)) / ((n : ℝ) + 1) := by
        simpa using (mul_one_div (L * (b - a)) ((n : ℝ) + 1))
      have hmesh' : L * (b - a) / ((n : ℝ) + 1) < ε / 2 := by
        simpa [hleft, hright] using hn'
      have hmesh'' : L * (b - a) / (N : ℝ) < ε / 2 := by
        simpa [N, Nat.cast_add, Nat.cast_one] using hmesh'
      simpa [δ, mul_div_assoc'] using hmesh''
    have h2mesh : 2 * L * δ < ε := by nlinarith [hmesh]

    let grid : ℕ → ℝ := fun k => a + (k : ℝ) * δ
    have hgrid0 : grid 0 = a := by simp [grid]
    have hgridN : grid N = b := by
      calc
        grid N = a + (N : ℝ) * δ := by simp [grid]
        _ = a + (N : ℝ) * (b - a) / (N : ℝ) := by simp [δ, mul_div_assoc']
        _ = a + (b - a) := by simp [mul_div_cancel_left₀, hNne]
        _ = b := by ring

    let mNat : ℕ → ℝ := fun k => (f (grid (k + 1)) - f (grid k)) / δ
    let cNat : ℕ → ℝ
      | 0 => mNat 0
      | k + 1 => mNat (k + 1) - mNat k
    let g : ℝ → ℝ := fun x => f a + ∑ i ∈ Finset.range N, cNat i * relu (x - grid i)

    have prefix_sum_cNat_eq_mNat : ∀ k : ℕ, (∑ i ∈ Finset.range (k + 1), cNat i) = mNat k := by
      intro k
      induction k with
      | zero =>
        simp [cNat, mNat]
      | succ k ih =>
        calc
          (∑ i ∈ Finset.range (k + 2), cNat i)
              = (∑ i ∈ Finset.range (k + 1), cNat i) + cNat (k + 1) := by
                simpa using (Finset.sum_range_succ (f := fun i => cNat i) (n := k + 1))
          _ = mNat k + cNat (k + 1) := by simp [ih]
          _ = mNat k + (mNat (k + 1) - mNat k) := by simp [cNat]
          _ = mNat (k + 1) := by ring

    have grid_mono : Monotone grid := by
      intro m n hmn
      dsimp [grid]
      have : (m : ℝ) ≤ (n : ℝ) := by exact_mod_cast hmn
      nlinarith

    have g_affine_on_segment :
        ∀ {k : ℕ}, k + 1 ≤ N → ∀ {x : ℝ}, grid k ≤ x → x ≤ grid (k + 1) →
          g x = g (grid k) + mNat k * (x - grid k) := by
      intro k hkN x hx0 hx1
      classical
      let F : ℕ → ℝ := fun i => cNat i * relu (x - grid i)
      let G : ℕ → ℝ := fun i => cNat i * relu (grid k - grid i)
      have hsub : Finset.range (k + 1) ⊆ Finset.range N := by
        intro i hi
        have hi' : i < k + 1 := Finset.mem_range.mp hi
        have : i < N := lt_of_lt_of_le hi' hkN
        exact Finset.mem_range.mpr this
      have hFzero : ∀ i ∈ Finset.range N, i ∉ Finset.range (k + 1) → F i = 0 := by
        intro i hiN hik
        have hik' : k + 1 ≤ i := by
          have : ¬ i < k + 1 := by
            exact fun hlt => hik (Finset.mem_range.mpr hlt)
          exact Nat.le_of_not_gt this
        have hgi : grid (k + 1) ≤ grid i := grid_mono hik'
        have hxle : x ≤ grid i := le_trans hx1 hgi
        simp [F, relu_sub_eq_zero_of_le (x := x) (t := grid i) hxle]
      have hGzero : ∀ i ∈ Finset.range N, i ∉ Finset.range (k + 1) → G i = 0 := by
        intro i hiN hik
        have hik' : k + 1 ≤ i := by
          have : ¬ i < k + 1 := by
            exact fun hlt => hik (Finset.mem_range.mpr hlt)
          exact Nat.le_of_not_gt this
        have hgi : grid k ≤ grid i := by
          have : k ≤ i := le_trans (Nat.le_succ k) hik'
          exact grid_mono this
        simp [G, relu_sub_eq_zero_of_le (x := grid k) (t := grid i) hgi]
      have sumF : (∑ i ∈ Finset.range N, F i) = (∑ i ∈ Finset.range (k + 1), F i) := by
        symm
        exact Finset.sum_subset hsub hFzero
      have sumG : (∑ i ∈ Finset.range N, G i) = (∑ i ∈ Finset.range (k + 1), G i) := by
        symm
        exact Finset.sum_subset hsub hGzero
      have gx : g x = f a + ∑ i ∈ Finset.range (k + 1), F i := by
        simp [g, F, sumF]
      have gk : g (grid k) = f a + ∑ i ∈ Finset.range (k + 1), G i := by
        simp [g, G, sumG]
      have hF' : ∀ i ∈ Finset.range (k + 1), F i = cNat i * (x - grid i) := by
        intro i hi
        have hi' : i ≤ k := Nat.le_of_lt_succ (Finset.mem_range.mp hi)
        have hgi : grid i ≤ grid k := grid_mono hi'
        have : grid i ≤ x := le_trans hgi hx0
        simp [F, relu_sub_eq_of_le (x := x) (t := grid i) this]
      have hG' : ∀ i ∈ Finset.range (k + 1), G i = cNat i * (grid k - grid i) := by
        intro i hi
        have hi' : i ≤ k := Nat.le_of_lt_succ (Finset.mem_range.mp hi)
        have : grid i ≤ grid k := grid_mono hi'
        simp [G, relu_sub_eq_of_le (x := grid k) (t := grid i) this]
      have gx' : g x = f a + ∑ i ∈ Finset.range (k + 1), cNat i * (x - grid i) := by
        refine gx.trans ?_
        congr 1
        exact Finset.sum_congr rfl hF'
      have gk' : g (grid k) = f a + ∑ i ∈ Finset.range (k + 1), cNat i * (grid k - grid i) := by
        refine gk.trans ?_
        congr 1
        exact Finset.sum_congr rfl hG'
      have hdiff :
          (∑ i ∈ Finset.range (k + 1), cNat i * (x - grid i)) -
            (∑ i ∈ Finset.range (k + 1), cNat i * (grid k - grid i)) =
              (∑ i ∈ Finset.range (k + 1), cNat i) * (x - grid k) := by
        calc
          (∑ i ∈ Finset.range (k + 1), cNat i * (x - grid i)) -
              (∑ i ∈ Finset.range (k + 1), cNat i * (grid k - grid i)) =
              ∑ i ∈ Finset.range (k + 1),
                (cNat i * (x - grid i) - cNat i * (grid k - grid i)) := by
                exact
                  (Finset.sum_sub_distrib (s := Finset.range (k + 1))
                    (f := fun i => cNat i * (x - grid i))
                    (g := fun i => cNat i * (grid k - grid i))).symm
          _ = ∑ i ∈ Finset.range (k + 1), cNat i * (x - grid k) := by
                apply Finset.sum_congr rfl
                intro i hi
                have hmul :
                    cNat i * (x - grid i) - cNat i * (grid k - grid i) =
                      cNat i * ((x - grid i) - (grid k - grid i)) := by
                  simpa using (mul_sub (cNat i) (x - grid i) (grid k - grid i)).symm
                have hinner : (x - grid i) - (grid k - grid i) = x - grid k := by ring
                simp [hmul, hinner]
          _ = (∑ i ∈ Finset.range (k + 1), cNat i) * (x - grid k) := by
                simpa using
                  (Finset.sum_mul (s := Finset.range (k + 1))
                    (f := fun i => cNat i)
                    (a := (x - grid k))).symm
      have hsub' : g x - g (grid k) = mNat k * (x - grid k) := by
        calc
          g x - g (grid k) =
              (f a + ∑ i ∈ Finset.range (k + 1), cNat i * (x - grid i)) -
                (f a + ∑ i ∈ Finset.range (k + 1), cNat i * (grid k - grid i)) := by
                simp [gx', gk']
          _ =
              (∑ i ∈ Finset.range (k + 1), cNat i * (x - grid i)) -
                (∑ i ∈ Finset.range (k + 1), cNat i * (grid k - grid i)) := by ring
          _ = (∑ i ∈ Finset.range (k + 1), cNat i) * (x - grid k) := hdiff
          _ = mNat k * (x - grid k) := by simp [prefix_sum_cNat_eq_mNat k]
      linarith

    have g_grid_eq_f_grid : ∀ k : ℕ, k ≤ N → g (grid k) = f (grid k) := by
      intro k hk
      induction k with
      | zero =>
        -- g(a) = f(a)
        have ha_le : ∀ i ∈ Finset.range N, a ≤ grid i := by
          intro i hi
          have : grid 0 ≤ grid i := grid_mono (Nat.zero_le i)
          simpa [hgrid0] using this
        have hsum :
            (∑ i ∈ Finset.range N, cNat i * relu (a - grid i)) = 0 := by
          refine Finset.sum_eq_zero ?_
          intro i hi
          have : a ≤ grid i := ha_le i hi
          simp [relu_sub_eq_zero_of_le (x := a) (t := grid i) this]
        simp [g, hgrid0, hsum]
      | succ k ih =>
        have hkN : k + 1 ≤ N := hk
        have hk_le : k ≤ N := le_trans (Nat.le_succ k) hkN
        have hx0 : grid k ≤ grid (k + 1) := grid_mono (Nat.le_succ k)
        have haff := g_affine_on_segment (k := k) hkN (x := grid (k + 1)) hx0 le_rfl
        have hstep : grid (k + 1) - grid k = δ := by
          dsimp [grid]
          simp [Nat.cast_add, Nat.cast_one, add_assoc, add_comm, mul_add, sub_eq_add_neg, mul_comm]
        have hm : mNat k * δ = f (grid (k + 1)) - f (grid k) := by
          have hδne : δ ≠ 0 := ne_of_gt hδpos
          dsimp [mNat]
          field_simp [hδne]
        have hk_eq : g (grid k) = f (grid k) := ih hk_le
        -- rewrite and finish
        have : g (grid (k + 1)) = f (grid (k + 1)) := by
          -- from affine step
          calc
            g (grid (k + 1)) = g (grid k) + mNat k * (grid (k + 1) - grid k) := haff
            _ = f (grid k) + mNat k * δ := by simp [hk_eq, hstep]
            _ = f (grid k) + (f (grid (k + 1)) - f (grid k)) := by simp [hm]
            _ = f (grid (k + 1)) := by ring
        simpa using this

    -- Build the network corresponding to the polygonal interpolant.
    let t : Fin N → ℝ := fun i => grid i.1
    let c : Fin N → ℝ := fun i => cNat i.1
    refine ⟨N, t, c, ?_⟩
    intro x hx
    have hhinge : hingeFun N t c (f a) x = g x := by
      -- hinge_fun sums over `Fin N`; use `Fin.sum_univ_eq_sum_range` to convert to `Finset.range
      -- N`.
      unfold hingeFun g t c
      congr 1
      simpa using
        (Fin.sum_univ_eq_sum_range
          (f := fun i : ℕ => cNat i * relu (x - grid i)) (n := N))
    -- Prove the approximation bound on `[a,b]`.
    have hxle_gridN : x ≤ grid N := by simpa [hgridN] using hx.2
    have hexists : ∃ k : ℕ, x ≤ grid k := ⟨N, hxle_gridN⟩
    -- Two cases: x = a (`Nat.find hexists = 0`) or x lies in some segment `[grid k, grid (k+1)]`.
    cases hjcases : Nat.find hexists with
    | zero =>
      have hx_eq : x = a := by
        have hxle : x ≤ grid 0 := by
          simpa [hjcases] using Nat.find_spec hexists
        have hxle' : x ≤ a := by simpa [hgrid0] using hxle
        exact le_antisymm hxle' hx.1
      have hg_a : g a = f a := by
        have := g_grid_eq_f_grid 0 (Nat.zero_le N)
        simpa [hgrid0] using this
      have : |f x - g x| < ε := by
        -- at `x = a`, `g a = f a`
        simpa [hx_eq, hg_a, abs_zero] using hε
      simpa [hhinge] using this
    | succ k =>
      have hj : x ≤ grid (k + 1) := by
        simpa [hjcases] using Nat.find_spec hexists
      have hkN : k + 1 ≤ N := by
        have hmin : Nat.find hexists ≤ N := Nat.find_min' hexists hxle_gridN
        simpa [hjcases] using hmin
      have hx_not_le_prev : ¬ x ≤ grid k := by
        intro hxle
        have hmin : Nat.find hexists ≤ k := Nat.find_min' hexists hxle
        rw [hjcases] at hmin
        exact Nat.not_succ_le_self k hmin
      have hx0 : grid k ≤ x := le_of_lt (lt_of_not_ge hx_not_le_prev)
      have hx1 : x ≤ grid (k + 1) := hj
      have hk_le : k ≤ N := le_trans (Nat.le_succ k) hkN
      have hgk : g (grid k) = f (grid k) := g_grid_eq_f_grid k hk_le
      have haff : g x = g (grid k) + mNat k * (x - grid k) :=
        g_affine_on_segment (k := k) hkN (x := x) hx0 hx1
      have hfx : |f x - g x| < ε := by
        -- bound via Lipschitz: |f x - g x| ≤ 2 L δ < ε
        have hgridk_mem : grid k ∈ Set.Icc a b := by
          have ha : a ≤ grid k := by
            have : grid 0 ≤ grid k := grid_mono (Nat.zero_le k)
            simpa [hgrid0] using this
          have hb : grid k ≤ b := by
            have : grid k ≤ grid N := grid_mono hk_le
            simpa [hgridN] using this
          exact ⟨ha, hb⟩
        have hgridkp1_mem : grid (k + 1) ∈ Set.Icc a b := by
          have ha : a ≤ grid (k + 1) := by
            have : grid 0 ≤ grid (k + 1) := grid_mono (Nat.zero_le (k + 1))
            simpa [hgrid0] using this
          have hb : grid (k + 1) ≤ b := by
            have : grid (k + 1) ≤ grid N := grid_mono hkN
            simpa [hgridN] using this
          exact ⟨ha, hb⟩
        have hx_dist : |x - grid k| ≤ δ := by
          have hx0' : 0 ≤ x - grid k := sub_nonneg.mpr hx0
          have hxle : x - grid k ≤ δ := by
            have : x - grid k ≤ grid (k + 1) - grid k := sub_le_sub_right hx1 (grid k)
            have hstep : grid (k + 1) - grid k = δ := by
              dsimp [grid]
              simp [Nat.cast_add, Nat.cast_one, add_assoc, add_comm, mul_add, sub_eq_add_neg,
                mul_comm]
            simpa [hstep] using this
          have habs : |x - grid k| = x - grid k := abs_of_nonneg hx0'
          simpa [habs] using hxle
        have h1 : |f x - f (grid k)| ≤ L * δ := by
          have := h_lip x hx (grid k) hgridk_mem
          exact le_trans this (mul_le_mul_of_nonneg_left hx_dist hL.le)
        have hmδ : mNat k * δ = f (grid (k + 1)) - f (grid k) := by
          have hδne : δ ≠ 0 := ne_of_gt hδpos
          dsimp [mNat]
          field_simp [hδne]
        have h2 : |f (grid k) - g x| ≤ L * δ := by
          -- g x is between the endpoints; bound by the endpoint difference.
          have habs' : |f (grid k) - g x| = |mNat k * (x - grid k)| := by
            have hgx : g x = f (grid k) + mNat k * (x - grid k) := by
              simpa [hgk] using haff
            have hs : f (grid k) - g x = -(mNat k * (x - grid k)) := by
              simp [hgx]
            have : |f (grid k) - g x| = |-(mNat k * (x - grid k))| := by
              simpa using congrArg abs hs
            simpa [abs_neg] using this
          -- use |x-grid k| ≤ δ to bound
          have hmul_le : |mNat k * (x - grid k)| ≤ |mNat k| * δ := by
            have := mul_le_mul_of_nonneg_left hx_dist (abs_nonneg (mNat k))
            simpa [abs_mul] using this
          have hmul_eq : |mNat k| * δ = |mNat k * δ| := by
            simp [abs_mul, abs_of_nonneg hδnonneg, mul_comm]
          have hendpoint : |mNat k * δ| ≤ L * δ := by
            -- from Lipschitz on grid points
            have hdiff := h_lip (grid (k + 1)) hgridkp1_mem (grid k) hgridk_mem
            have hstep' : grid (k + 1) - grid k = δ := by
              dsimp [grid]
              simp [Nat.cast_add, Nat.cast_one, add_assoc, add_comm, mul_add, sub_eq_add_neg,
                mul_comm]
            have hnonneg : 0 ≤ grid (k + 1) - grid k := by
              have : grid k ≤ grid (k + 1) := grid_mono (Nat.le_succ k)
              exact sub_nonneg.mpr this
            have hstep : |grid (k + 1) - grid k| = δ := by
              calc
                |grid (k + 1) - grid k| = grid (k + 1) - grid k := abs_of_nonneg hnonneg
                _ = δ := hstep'
            -- rewrite |mNat k * δ| as |f(grid(k+1)) - f(grid k)|
            have : |mNat k * δ| = |f (grid (k + 1)) - f (grid k)| := by simp [hmδ]
            -- combine
            simpa [this, hstep] using hdiff
          -- combine bounds
          have hbound : |mNat k * (x - grid k)| ≤ |mNat k * δ| := by
            exact le_trans hmul_le (le_of_eq hmul_eq)
          have hfinal : |mNat k * (x - grid k)| ≤ L * δ := le_trans hbound hendpoint
          simpa [habs'] using hfinal
        have : |f x - g x| ≤ 2 * L * δ := by
          have htri : |f x - g x| ≤ |f x - f (grid k)| + |f (grid k) - g x| := by
            have htri0 :
                |(f x - f (grid k)) + (f (grid k) - g x)| ≤
                  |f x - f (grid k)| + |f (grid k) - g x| :=
              abs_add_le (f x - f (grid k)) (f (grid k) - g x)
            have hrew' : (f x - f (grid k)) + (f (grid k) - g x) = f x - g x := by ring
            simpa [hrew'] using htri0
          have hsum : |f x - f (grid k)| + |f (grid k) - g x| ≤ L * δ + L * δ :=
            add_le_add h1 (by simpa [abs_sub_comm] using h2)
          have hsum' : L * δ + L * δ = 2 * L * δ := by ring
          exact le_trans htri (by simpa [hsum'] using hsum)
        exact lt_of_le_of_lt this h2mesh
      simpa [hhinge] using hfx

/--
1D Universal Approximation (ReLU, one hidden layer), stated as an existence theorem for a 2-layer
  MLP.

This is a wrapper around `relu_universal_approximation_Icc_hinge` that instantiates the linear
  layers
as the explicit hinge construction.
-/
theorem relu_universal_approximation_Icc {f : ℝ → ℝ} {a b L : ℝ}
    (h_ab : a < b) (hL : 0 < L)
    (h_lip : ∀ x ∈ Set.Icc a b, ∀ y ∈ Set.Icc a b, |f x - f y| ≤ L * |x - y|) :
    ∀ ε > 0, ∃ (hidDim : ℕ) (l1 : LinearSpec ℝ 1 hidDim) (l2 : LinearSpec ℝ hidDim 1),
      ∀ x ∈ Set.Icc a b, |f x - mlpEval1d hidDim l1 l2 x| < ε := by
  intro ε hε
  classical
  rcases
      relu_universal_approximation_Icc_hinge (f := f) (a := a) (b := b) (L := L)
        h_ab hL h_lip ε hε with
    ⟨hidDim, t, c, happx⟩
  refine ⟨hidDim, hingeLayer1 hidDim t, hingeLayer2 hidDim c (f a), ?_⟩
  intro x hx
  have hnet :
      mlpEval1d hidDim (hingeLayer1 hidDim t) (hingeLayer2 hidDim c (f a)) x =
        hingeFun hidDim t c (f a) x := by
    simpa using (mlp_eval_1d_hinge hidDim t c (f a) x)
  simpa [hnet] using happx x hx

  end NN.MLTheory.Proofs.UniversalApproximation
