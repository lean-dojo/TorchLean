/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.FDeriv.Elementwise

/-!
# Softmax

Fréchet-derivative facts for **axis softmax** on Euclidean vectors. These statements over
$\mathbb R$ justify the attention-style row softmax nodes (`Vec n → Vec n`) in the tape and DAG
autograd proofs.

## References
- Baydin et al., *Automatic Differentiation in Machine Learning: a Survey* (JMLR 2018).
- The Matrix Cookbook (softmax Jacobian identities / vector calculus conventions).
- PyTorch docs, followed for naming and behavior (no theorem depends on them):
  https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html
-/

@[expose] public section


namespace TorchLean

open scoped BigOperators

noncomputable section

/--
`sumExp x = ∑ᵢ exp(xᵢ)`, the normalizing denominator of softmax.
-/
def Vec.sumExp {n : Nat} (x : Vec n) : ℝ :=
  ∑ i : Fin n, Real.exp (x i)

/-- `sumExp x` is strictly positive when the index type is nonempty. -/
theorem Vec.sumExp_pos {n : Nat} (x : Vec (Nat.succ n)) : 0 < sumExp (n := Nat.succ n) x := by
  simpa [sumExp] using Finset.sum_pos (fun i _ => Real.exp_pos (x i)) Finset.univ_nonempty

/-- `sumExp x ≠ 0` for `n = succ _`, a corollary of `Vec.sumExp_pos`. -/
theorem Vec.sumExp_ne_zero {n : Nat} (x : Vec (Nat.succ n)) : sumExp (n := Nat.succ n) x ≠ 0 :=
  ne_of_gt (sumExp_pos (n := n) x)

/-- The coordinate exponential `x ↦ exp (x j)` has derivative `dx ↦ exp (x j) * dx j`. -/
theorem Vec.hasFDerivAt_exp_apply {n : Nat} (x : Vec n) (j : Fin n) :
    HasFDerivAt (fun x : Vec n => Real.exp (x j))
      (Real.exp (x j) • EuclideanSpace.proj (𝕜 := ℝ) j) x := by
  simpa only [EuclideanSpace.coe_proj] using
    ((EuclideanSpace.proj (𝕜 := ℝ) j).hasFDerivAt (x := x)).exp

/-- `sumExp` has derivative `dx ↦ ∑ j, exp (x j) * dx j`. -/
theorem Vec.hasFDerivAt_sumExp {n : Nat} (x : Vec n) :
    HasFDerivAt (sumExp (n := n))
      (∑ j : Fin n, Real.exp (x j) • EuclideanSpace.proj (𝕜 := ℝ) j) x := by
  change HasFDerivAt (fun y : Vec n => ∑ j : Fin n, Real.exp (y j))
    (∑ j : Fin n, Real.exp (x j) • EuclideanSpace.proj (𝕜 := ℝ) j) x
  exact HasFDerivAt.fun_sum (u := (Finset.univ : Finset (Fin n)))
    (fun j _ => hasFDerivAt_exp_apply x j)

/--
Softmax on Euclidean vectors: `softmax x i = exp(xᵢ) / sumExp x` for `n = succ _`, and the
identity on the trivial space when `n = 0`.
-/
def Vec.softmax : {n : Nat} → Vec n → Vec n
  | 0, x => x
  | Nat.succ _, x => WithLp.toLp 2 fun i => Real.exp (x i) / sumExp x

/--
Coordinate `i` of the softmax derivative at `x`, as a continuous linear map: the functional
`dx ↦ yᵢ * dxᵢ - yᵢ * ⟪y, dx⟫`, where `y = softmax x`.
-/
def Vec.softmaxDerivCoord {n : Nat} (x : Vec n) (i : Fin n) : Vec n →L[ℝ] ℝ :=
  let y := softmax (n := n) x
  (EuclideanSpace.proj (𝕜 := ℝ) i).smulRight (y i) - (innerSL ℝ y).smulRight (y i)

/-- The full Fréchet derivative of `softmax` at `x`, packaged as a CLM `Vec n →L Vec n`. -/
def Vec.softmaxDerivCLM {n : Nat} (x : Vec n) : Vec n →L[ℝ] Vec n :=
  (EuclideanSpace.equiv (Fin n) ℝ).symm.toContinuousLinearMap.comp <|
    ContinuousLinearMap.pi (fun i : Fin n => softmaxDerivCoord (n := n) x i)

/--
Closed-form JVP (directional derivative) for softmax.

If `y = softmax x` and `s = ⟪y, dx⟫`, then `(softmaxJVP x dx)ᵢ = yᵢ * (dxᵢ - s)`.
-/
def Vec.softmaxJVP : {n : Nat} → Vec n → Vec n → Vec n
  | 0, _x, dx => dx
  | Nat.succ n, x, dx =>
      let y := softmax (n := Nat.succ n) x
      let s : ℝ := inner ℝ y dx
      WithLp.toLp 2 fun i => y i * (dx i - s)

/-- The closed-form JVP `softmaxJVP` agrees with the CLM derivative `softmaxDerivCLM`. -/
theorem Vec.softmaxJVP_eq_deriv {n : Nat} (x dx : Vec n) :
    softmaxJVP (n := n) x dx = (softmaxDerivCLM (n := n) x) dx := by
  classical
  cases n with
  | zero =>
      ext i
      exact i.elim0
  | succ n =>
      ext i
      have hR :
          ((softmaxDerivCLM (n := Nat.succ n) x) dx) i =
            (softmaxDerivCoord (n := Nat.succ n) x i) dx := by
        simp [softmaxDerivCLM]
      rw [hR]
      simp [softmaxJVP, softmaxDerivCoord, inner_eq_sum_mul,
        ContinuousLinearMap.smulRight_apply, sub_eq_add_neg, mul_add, mul_comm]

/--
The softmax Jacobian, applied through `softmaxJVP`, is self-adjoint for the Euclidean inner
product, so the VJP can reuse the JVP formula.
-/
theorem Vec.inner_softmaxJVP_comm {n : Nat} (x dx δ : Vec n) :
    inner ℝ (softmaxJVP (n := n) x dx) δ = inner ℝ dx (softmaxJVP (n := n) x δ) := by
  cases n with
  | zero =>
      -- In `Vec 0` every vector is `0`, so both sides are `0`.
      simp [softmaxJVP]
  | succ n =>
      rw [inner_eq_sum_mul, inner_eq_sum_mul]
      simp only [softmaxJVP, PiLp.toLp_apply, inner_eq_sum_mul]
      set y := softmax x
      -- Both sides expand to `∑ yᵢ dxᵢ δᵢ - (∑ yⱼ dxⱼ) (∑ yᵢ δᵢ)`.
      have hleft :
          (∑ i, y i * (dx i - ∑ j, y j * dx j) * δ i) =
            (∑ i, y i * dx i * δ i) - (∑ j, y j * dx j) * ∑ i, y i * δ i := by
        rw [Finset.mul_sum, ← Finset.sum_sub_distrib]
        apply Finset.sum_congr rfl
        intro i _
        ring
      have hright :
          (∑ i, dx i * (y i * (δ i - ∑ j, y j * δ j))) =
            (∑ i, y i * dx i * δ i) - (∑ j, y j * δ j) * ∑ i, y i * dx i := by
        rw [Finset.mul_sum, ← Finset.sum_sub_distrib]
        apply Finset.sum_congr rfl
        intro i _
        ring
      rw [hleft, hright]
      ring

/-- Softmax is Fréchet-differentiable everywhere, with derivative `softmaxDerivCLM`. -/
theorem Vec.hasFDerivAt_softmax {n : Nat} (x : Vec n) :
    HasFDerivAt (softmax (n := n)) (softmaxDerivCLM (n := n) x) x := by
  cases n with
  | zero =>
      -- `softmax` is the identity on the trivial space, and all CLMs coincide.
      have hD : softmaxDerivCLM (n := 0) x = (1 : (Vec 0) →L[ℝ] (Vec 0)) := by
        ext dx i
        exact i.elim0
      rw [hD]
      change HasFDerivAt (fun x : Vec 0 => x) (1 : (Vec 0) →L[ℝ] (Vec 0)) x
      exact ((1 : (Vec 0) →L[ℝ] (Vec 0)).hasFDerivAt (x := x))
  | succ n =>
      -- Each coordinate is the quotient `exp (x i) / sumExp x`. Differentiate it as a product
      -- with the reciprocal of the positive denominator, then assemble the coordinates.
      have hsum_ne : sumExp (n := Nat.succ n) x ≠ 0 := sumExp_ne_zero x
      have hinv := (hasFDerivAt_inv (𝕜 := ℝ) hsum_ne).comp x (hasFDerivAt_sumExp x)
      have hcoord (i : Fin (Nat.succ n)) :
          HasFDerivAt (fun y : Vec (Nat.succ n) => softmax (n := Nat.succ n) y i)
            (softmaxDerivCoord (n := Nat.succ n) x i) x := by
        have hmul := (hasFDerivAt_exp_apply x i).mul hinv
        have hmap :
            Real.exp (x i) •
                ((ContinuousLinearMap.toSpanSingleton ℝ (-(sumExp x ^ 2)⁻¹)).comp
                  (∑ j : Fin (Nat.succ n), Real.exp (x j) • EuclideanSpace.proj (𝕜 := ℝ) j)) +
              (sumExp x)⁻¹ • (Real.exp (x i) • EuclideanSpace.proj (𝕜 := ℝ) i) =
            softmaxDerivCoord (n := Nat.succ n) x i := by
          ext dx
          -- Pull the common denominator out of the dot product before comparing coefficients.
          have hsum :
              (∑ j : Fin (Nat.succ n), Real.exp (x j) / sumExp x * dx j) =
                (∑ j : Fin (Nat.succ n), Real.exp (x j) * dx j) / sumExp x := by
            rw [div_eq_mul_inv, Finset.sum_mul]
            exact Finset.sum_congr rfl fun j _ => by ring
          simp only [_root_.add_apply, smul_apply, ContinuousLinearMap.comp_apply,
            ContinuousLinearMap.toSpanSingleton_apply, sub_apply, sum_apply,
            EuclideanSpace.coe_proj, innerSL_apply_apply, inner_eq_sum_mul,
            ContinuousLinearMap.smulRight_apply, smul_eq_mul, softmaxDerivCoord, softmax,
            PiLp.toLp_apply]
          rw [hsum]
          ring
        have hfun :
            (fun y : Vec (Nat.succ n) => softmax (n := Nat.succ n) y i) =
              (fun y : Vec (Nat.succ n) => Real.exp (y i)) *
                ((fun s : ℝ => s⁻¹) ∘ sumExp (n := Nat.succ n)) := by
          funext y
          simp only [softmax, PiLp.toLp_apply, div_eq_mul_inv, Pi.mul_apply,
            Function.comp_apply]
        exact (hmul.congr_of_eventuallyEq hfun.eventuallyEq).congr_fderiv hmap
      have hpi := (hasFDerivAt_pi (𝕜 := ℝ)
        (φ := fun i (y : Vec (Nat.succ n)) => softmax (n := Nat.succ n) y i)
        (φ' := fun i => softmaxDerivCoord (n := Nat.succ n) x i) (x := x)).2 hcoord
      have hfun :
          softmax (n := Nat.succ n) =
            (EuclideanSpace.equiv (Fin (Nat.succ n)) ℝ).symm ∘
              (fun (y : Vec (Nat.succ n)) i => softmax (n := Nat.succ n) y i) := by
        funext y
        ext i
        rfl
      have hcomp :=
        (EuclideanSpace.equiv (Fin (Nat.succ n)) ℝ).symm.toContinuousLinearMap.hasFDerivAt.comp
          x hpi
      exact hcomp.congr_of_eventuallyEq hfun.eventuallyEq

end
end TorchLean
