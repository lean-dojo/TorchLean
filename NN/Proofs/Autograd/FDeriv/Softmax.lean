/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.FDeriv.Elementwise

public import Mathlib.Analysis.Calculus.Deriv.Inv
public import Mathlib.Analysis.Calculus.FDeriv.Add
public import Mathlib.Analysis.Calculus.FDeriv.Bilinear
public import Mathlib.Analysis.Calculus.FDeriv.Comp
public import Mathlib.Analysis.InnerProductSpace.Calculus

/-!
# Softmax

Fréchet-derivative facts for **axis softmax** on Euclidean vectors.

This is the analytic (ℝ) ingredient used to justify attention-style row softmax nodes
(`Vec n → Vec n`) in the tape/DAG autograd proofs.

## References
- Baydin et al., *Automatic Differentiation in Machine Learning: a Survey* (JMLR 2018).
- The Matrix Cookbook (softmax Jacobian identities / vector calculus conventions).
- PyTorch docs for naming/behavior alignment (not used for theorems):
  https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html
-/

@[expose] public section


namespace Proofs
namespace Autograd

open scoped BigOperators

noncomputable section

/--
Package a coordinate function `Fin n → ℝ` as a Euclidean vector `Vec n`.

This is just `(EuclideanSpace.equiv …).symm`, but it is convenient to name in analytic proofs.
-/
def softmaxVecOfFun {n : Nat} (f : Fin n → ℝ) : Vec n :=
  (EuclideanSpace.equiv (𝕜 := ℝ) (ι := Fin n)).symm f

@[simp] lemma softmaxVecOfFun_apply {n : Nat} (f : Fin n → ℝ) (i : Fin n) :
    softmaxVecOfFun (n := n) f i = f i := by
  simp [softmaxVecOfFun]

/--
`sumExp x = ∑ᵢ exp(xᵢ)`.

This is the normalizing denominator in softmax.
-/
def sumExp {n : Nat} (x : Vec n) : ℝ :=
  ∑ i : Fin n, Real.exp (x i)

/--
Softmax on Euclidean vectors.

For `n = succ _`:
`softmaxVec x i = exp(xᵢ) / sumExp x`.

The `n = 0` branch is the identity on the trivial space.
-/
def softmaxVec : {n : Nat} → Vec n → Vec n
  | 0, x => x
  | Nat.succ n, x => softmaxVecOfFun (n := Nat.succ n) fun i => Real.exp (x i) / sumExp x

/--
The dot-product functional `x ↦ ∑ᵢ yᵢ * xᵢ`, packaged as a continuous linear map.

This is used to express the softmax Jacobian in a clean coordinate-free way.
-/
def dotCLM {n : Nat} (y : Vec n) : Vec n →L[ℝ] ℝ :=
  ∑ j : Fin n, (evalCLM (n := n) j).smulRight (y j)

@[simp] lemma dotCLM_apply {n : Nat} (y x : Vec n) :
    dotCLM (n := n) y x = ∑ j : Fin n, y j * x j := by
  classical
  simp [dotCLM, evalCLM_apply, ContinuousLinearMap.smulRight_apply, mul_comm]

/--
The `i`th output coordinate of the softmax derivative at `x`, as a continuous linear map.

If `y = softmaxVec x`, then this is the linear functional:
`dx ↦ yᵢ * dxᵢ - yᵢ * ⟪y, dx⟫`.
-/
def softmaxDerivCoord {n : Nat} (x : Vec n) (i : Fin n) : Vec n →L[ℝ] ℝ :=
  let y := softmaxVec (n := n) x
  (evalCLM (n := n) i).smulRight (y i) - (dotCLM (n := n) y).smulRight (y i)

/-- The full Fréchet derivative of `softmaxVec` at `x`, packaged as a CLM `Vec n →L Vec n`. -/
def softmaxDerivCLM {n : Nat} (x : Vec n) : Vec n →L[ℝ] Vec n :=
  (euclideanEquiv n).symm.toContinuousLinearMap.comp <|
    ContinuousLinearMap.pi (fun i : Fin n => softmaxDerivCoord (n := n) x i)

@[simp] lemma pi_apply_vec {n : Nat} (f : Fin n → Vec n →L[ℝ] ℝ) (x : Vec n) (i : Fin n) :
    (ContinuousLinearMap.pi f x) i = f i x := by
  rfl

/--
Closed-form JVP (directional derivative) for softmax.

If `y = softmaxVec x` and `s = ⟪y, dx⟫`, then `(softmaxJvp x dx)ᵢ = yᵢ * (dxᵢ - s)`.
-/
def softmaxJvp : {n : Nat} → Vec n → Vec n → Vec n
  | 0, _x, dx => dx
  | Nat.succ n, x, dx =>
      let y := softmaxVec (n := Nat.succ n) x
      let s : ℝ := dotCLM (n := Nat.succ n) y dx
      softmaxVecOfFun (n := Nat.succ n) fun i => y i * (dx i - s)

/-- The closed-form JVP `softmaxJvp` agrees with the CLM derivative `softmaxDerivCLM`. -/
theorem softmaxJvp_eq_deriv {n : Nat} (x dx : Vec n) :
    softmaxJvp (n := n) x dx = (softmaxDerivCLM (n := n) x) dx := by
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
        simp [softmaxDerivCLM, euclideanEquiv]
      rw [hR]
      simp [softmaxJvp, softmaxDerivCoord, dotCLM_apply, evalCLM_apply,
        ContinuousLinearMap.add_apply, ContinuousLinearMap.smulRight_apply,
        sub_eq_add_neg, mul_add, mul_comm]

/--
Self-adjointness identity for the softmax Jacobian in this inner-product encoding.

This lemma is used to show the VJP can be expressed by reusing the JVP formula.
-/
theorem inner_softmaxJvp_comm {n : Nat} (x dx δ : Vec n) :
    inner ℝ (softmaxJvp (n := n) x dx) δ = inner ℝ dx (softmaxJvp (n := n) x δ) := by
  classical
  cases n with
  | zero =>
      -- all vectors are `0`, so both sides are `0`
      simp [softmaxJvp]
  | succ n =>
      let y : Vec (Nat.succ n) := softmaxVec (n := Nat.succ n) x
      let sdx : ℝ := dotCLM (n := Nat.succ n) y dx
      let sδ : ℝ := dotCLM (n := Nat.succ n) y δ
      have hsdx : sdx = ∑ i : Fin (Nat.succ n), y i * dx i := by
        simp [sdx, dotCLM_apply, mul_comm]
      have hsδ : sδ = ∑ i : Fin (Nat.succ n), y i * δ i := by
        simp [sδ, dotCLM_apply, mul_comm]
      calc
        inner ℝ (softmaxJvp (n := Nat.succ n) x dx) δ
            = ∑ i : Fin (Nat.succ n), (softmaxJvp (n := Nat.succ n) x dx i) * δ i := by
                simp [inner_eq_sum_mul]
        _ = ∑ i : Fin (Nat.succ n), (y i * (dx i - sdx)) * δ i := by
              simp [softmaxJvp, y, sdx]
        _ =
            (∑ i : Fin (Nat.succ n), (y i * dx i) * δ i) - sdx * (∑ i : Fin (Nat.succ n), y i * δ i)
              := by
              -- expand and factor the constant `sdx`
              have hsplit :
                  (∑ i : Fin (Nat.succ n), (y i * (dx i - sdx)) * δ i)
                    =
                  (∑ i : Fin (Nat.succ n), (y i * dx i) * δ i) -
                    (∑ i : Fin (Nat.succ n), (y i * sdx) * δ i) := by
                -- pointwise `ring_nf`, then use `sum_sub_distrib`
                calc
                  (∑ i : Fin (Nat.succ n), (y i * (dx i - sdx)) * δ i)
                      =
                      ∑ i : Fin (Nat.succ n), ((y i * dx i) * δ i - (y i * sdx) * δ i) := by
                        refine Finset.sum_congr rfl ?_
                        intro i _hi
                        ring_nf
                  _ =
                      (∑ i : Fin (Nat.succ n), (y i * dx i) * δ i) -
                        (∑ i : Fin (Nat.succ n), (y i * sdx) * δ i) := by
                        simp [Finset.sum_sub_distrib]
              have hfactor :
                  (∑ i : Fin (Nat.succ n), (y i * sdx) * δ i) = sdx * (∑ i : Fin (Nat.succ n), y i *
                    δ i) := by
                calc
                  (∑ i : Fin (Nat.succ n), (y i * sdx) * δ i)
                      = ∑ i : Fin (Nat.succ n), sdx * (y i * δ i) := by
                          refine Finset.sum_congr rfl ?_
                          intro i _hi
                          ring_nf
                  _ = sdx * (∑ i : Fin (Nat.succ n), y i * δ i) := by
                        simp [Finset.mul_sum]
              simpa [hfactor] using hsplit
        _ = (∑ i : Fin (Nat.succ n), (y i * dx i) * δ i) - sdx * sδ := by
              simp [hsδ]
        _ = (∑ i : Fin (Nat.succ n), (y i * dx i) * δ i) - sδ * sdx := by
              ring_nf
        _ =
            (∑ i : Fin (Nat.succ n), (y i * dx i) * δ i) -
              sδ * (∑ i : Fin (Nat.succ n), y i * dx i) := by
              simp [hsdx]
        _ = ∑ i : Fin (Nat.succ n), dx i * (y i * (δ i - sδ)) := by
              -- reverse the previous calculation for the RHS form
              have hsplit :
                  (∑ i : Fin (Nat.succ n), dx i * (y i * (δ i - sδ)))
                    =
                  (∑ i : Fin (Nat.succ n), (y i * dx i) * δ i) - sδ * (∑ i : Fin (Nat.succ n), y i *
                    dx i) := by
                -- expand and factor the constant `sδ`
                have hsplit0 :
                    (∑ i : Fin (Nat.succ n), dx i * (y i * (δ i - sδ)))
                      =
                    (∑ i : Fin (Nat.succ n), (y i * dx i) * δ i) -
                      (∑ i : Fin (Nat.succ n), (y i * dx i) * sδ) := by
                  calc
                    (∑ i : Fin (Nat.succ n), dx i * (y i * (δ i - sδ)))
                        =
                        ∑ i : Fin (Nat.succ n), ((y i * dx i) * δ i - (y i * dx i) * sδ) := by
                          refine Finset.sum_congr rfl ?_
                          intro i _hi
                          ring_nf
                    _ =
                        (∑ i : Fin (Nat.succ n), (y i * dx i) * δ i) -
                          (∑ i : Fin (Nat.succ n), (y i * dx i) * sδ) := by
                          simp [Finset.sum_sub_distrib]
                have hfactor0 :
                    (∑ i : Fin (Nat.succ n), (y i * dx i) * sδ) = sδ * (∑ i : Fin (Nat.succ n), y i
                      * dx i) := by
                  calc
                    (∑ i : Fin (Nat.succ n), (y i * dx i) * sδ)
                        = ∑ i : Fin (Nat.succ n), sδ * (y i * dx i) := by
                            refine Finset.sum_congr rfl ?_
                            intro i _hi
                            ring_nf
                    _ = sδ * (∑ i : Fin (Nat.succ n), y i * dx i) := by
                          simp [Finset.mul_sum]
                simpa [hfactor0] using hsplit0
              simpa using hsplit.symm
        _ = ∑ i : Fin (Nat.succ n), dx i * (softmaxJvp (n := Nat.succ n) x δ i) := by
              simp [softmaxJvp, y, sδ, mul_comm]
        _ = inner ℝ dx (softmaxJvp (n := Nat.succ n) x δ) := by
              simp [inner_eq_sum_mul]

/-- Softmax is Fréchet-differentiable everywhere, with derivative `softmaxDerivCLM`. -/
theorem hasFDerivAt_softmaxVec {n : Nat} (x : Vec n) :
    HasFDerivAt (softmaxVec (n := n)) (softmaxDerivCLM (n := n) x) x := by
  classical
  cases n with
  | zero =>
      -- `softmaxVec` is the identity on the trivial space, and all CLMs coincide.
      have hD : softmaxDerivCLM (n := 0) x = (1 : (Vec 0) →L[ℝ] (Vec 0)) := by
        ext dx i
        exact i.elim0
      simpa [softmaxVec, hD] using ((1 : (Vec 0) →L[ℝ] (Vec 0)).hasFDerivAt (x := x))
  | succ n =>
      -- Coordinatewise proof: `softmaxVec x i = exp(x i) * (sumExp x)⁻¹`.
      have hsum_ne : sumExp (n := Nat.succ n) x ≠ 0 := by
        have hpos : 0 < sumExp (n := Nat.succ n) x := by
          -- sum of strictly positive terms over a nonempty index set
          have hterm : ∀ i : Fin (Nat.succ n), 0 < Real.exp (x i) := fun i => Real.exp_pos (x i)
          simpa [sumExp] using Finset.sum_pos (fun i _ => hterm i) (Finset.univ_nonempty)
        exact ne_of_gt hpos

      -- Derivative of `sumExp`.
      let sumDeriv : Vec (Nat.succ n) →L[ℝ] ℝ :=
        ∑ j : Fin (Nat.succ n), (evalCLM (n := Nat.succ n) j).smulRight (Real.exp (x j))
      have hsumF :
          HasFDerivAt (sumExp (n := Nat.succ n)) sumDeriv x := by
        -- `sumExp` is a finite sum of `x ↦ exp(x j)`.
        have hcoord :
            ∀ j : Fin (Nat.succ n),
              HasFDerivAt (fun x : Vec (Nat.succ n) => Real.exp (x j))
                ((evalCLM (n := Nat.succ n) j).smulRight (Real.exp (x j))) x := by
          intro j
          have hexp : HasDerivAt Real.exp (Real.exp (x j)) (x j) := Real.hasDerivAt_exp (x j)
          have hexpF :
              HasFDerivAt Real.exp
                (ContinuousLinearMap.smulRight (M₁ := ℝ) (M₂ := ℝ) (R := ℝ) (S := ℝ)
                  (1 : ℝ →L[ℝ] ℝ) (Real.exp (x j))) (x j) :=
            hexp.hasFDerivAt
          have happly :
              HasFDerivAt (fun x : Vec (Nat.succ n) => x j) (evalCLM (n := Nat.succ n) j) x := by
            simpa using ((evalCLM (n := Nat.succ n) j).hasFDerivAt (x := x))
          have hcomp := hexpF.comp x happly
          -- simplify the composed CLM
          have hlin :
              (ContinuousLinearMap.smulRight (M₁ := ℝ) (M₂ := ℝ) (R := ℝ) (S := ℝ)
                  (1 : ℝ →L[ℝ] ℝ) (Real.exp (x j))).comp (evalCLM (n := Nat.succ n) j)
                =
              (evalCLM (n := Nat.succ n) j).smulRight (Real.exp (x j)) := by
            ext dx
            simp [ContinuousLinearMap.smulRight_apply]
          exact hcomp.congr_fderiv hlin
        have hsum :=
          (HasFDerivAt.sum (u := (Finset.univ : Finset (Fin (Nat.succ n))))
            (A := fun j : Fin (Nat.succ n) => fun x : Vec (Nat.succ n) => Real.exp (x j))
            (A' := fun j : Fin (Nat.succ n) => (evalCLM (n := Nat.succ n) j).smulRight (Real.exp (x
              j)))
            (x := x))
            (by
              intro j _hj
              simpa using (hcoord j))
        -- transport the goal function from "sum of coordinate functions" to `sumExp`
        have hEq :
            (∑ j : Fin (Nat.succ n), (fun x : Vec (Nat.succ n) => Real.exp (x j)))
              =
            (sumExp (n := Nat.succ n)) := by
          funext x
          simp [sumExp]
        -- rewrite the summed derivative map to `sumDeriv`
        have hsum' :
            HasFDerivAt (sumExp (n := Nat.succ n)) (∑ j : Fin (Nat.succ n),
                (evalCLM (n := Nat.succ n) j).smulRight (Real.exp (x j))) x :=
          hsum.congr_of_eventuallyEq hEq.symm.eventuallyEq
        simpa [sumDeriv] using hsum'

      -- Derivative of the inverse `x ↦ (sumExp x)⁻¹`.
      have hinv :
          HasFDerivAt (fun x : Vec (Nat.succ n) => (sumExp (n := Nat.succ n) x)⁻¹)
            ((ContinuousLinearMap.smulRight (1 : ℝ →L[ℝ] ℝ) (-(sumExp (n := Nat.succ n) x ^
              2)⁻¹)).comp sumDeriv) x := by
        have hinv0 := (hasFDerivAt_inv (𝕜 := ℝ) (x := sumExp (n := Nat.succ n) x) hsum_ne)
        exact hinv0.comp x hsumF

      -- Multiply `exp(x i)` by `(sumExp x)⁻¹` coordinatewise, then assemble via `hasFDerivAt_pi`.
      have hcoord_soft :
          ∀ i : Fin (Nat.succ n),
            HasFDerivAt (fun x : Vec (Nat.succ n) => softmaxVec (n := Nat.succ n) x i)
              (softmaxDerivCoord (n := Nat.succ n) x i) x := by
        intro i
        let u : Vec (Nat.succ n) → ℝ := fun x => Real.exp (x i)
        let u' : Vec (Nat.succ n) →L[ℝ] ℝ :=
          (evalCLM (n := Nat.succ n) i).smulRight (Real.exp (x i))
        have hu : HasFDerivAt u u' x := by
          -- same proof as above for a fixed coordinate
          have hexp : HasDerivAt Real.exp (Real.exp (x i)) (x i) := Real.hasDerivAt_exp (x i)
          have hexpF :
              HasFDerivAt Real.exp
                (ContinuousLinearMap.smulRight (M₁ := ℝ) (M₂ := ℝ) (R := ℝ) (S := ℝ)
                  (1 : ℝ →L[ℝ] ℝ) (Real.exp (x i))) (x i) :=
            hexp.hasFDerivAt
          have happly :
              HasFDerivAt (fun x : Vec (Nat.succ n) => x i) (evalCLM (n := Nat.succ n) i) x := by
            simpa using ((evalCLM (n := Nat.succ n) i).hasFDerivAt (x := x))
          have hcomp := hexpF.comp x happly
          have hlin :
              (ContinuousLinearMap.smulRight (M₁ := ℝ) (M₂ := ℝ) (R := ℝ) (S := ℝ)
                  (1 : ℝ →L[ℝ] ℝ) (Real.exp (x i))).comp (evalCLM (n := Nat.succ n) i)
                =
              u' := by
            ext dx
            simp [u', ContinuousLinearMap.smulRight_apply]
          exact hcomp.congr_fderiv hlin

        let v : Vec (Nat.succ n) → ℝ := fun x => (sumExp (n := Nat.succ n) x)⁻¹
        let v' : Vec (Nat.succ n) →L[ℝ] ℝ :=
          (ContinuousLinearMap.smulRight (1 : ℝ →L[ℝ] ℝ) (-(sumExp (n := Nat.succ n) x ^ 2)⁻¹)).comp
            sumDeriv

        let B := (ContinuousLinearMap.mul ℝ ℝ)
        have hmul :=
          ContinuousLinearMap.hasFDerivAt_of_bilinear (B := B) (hf := hu) (hg := hinv)
        -- rewrite `u*v` to the softmax coordinate and the derivative to `softmaxDerivCoord`.
        have hfun :
            (fun x : Vec (Nat.succ n) => (B (u x)) (v x)) =
              (fun x : Vec (Nat.succ n) => softmaxVec (n := Nat.succ n) x i) := by
          funext x
          simp [B, softmaxVec, sumExp, u, v, ContinuousLinearMap.mul_apply', div_eq_mul_inv]
        -- now show the derivative CLM matches our closed form by ext on directions
        have hderiv :
            (B.precompR (Vec (Nat.succ n)) (u x) v' + B.precompL (Vec (Nat.succ n)) u' (v x))
              =
            softmaxDerivCoord (n := Nat.succ n) x i := by
          ext dx
          -- expand everything and keep powers factored (`inv_pow` avoids expanding squares)
          simp [softmaxDerivCoord, dotCLM, softmaxVec, sumExp, u, v, u', v', sumDeriv,
            B, ContinuousLinearMap.mul_apply', ContinuousLinearMap.precompR_apply,
            ContinuousLinearMap.precompL_apply, ContinuousLinearMap.add_apply,
            ContinuousLinearMap.comp_apply,
            ContinuousLinearMap.smulRight_apply, evalCLM_apply, div_eq_mul_inv,
            mul_assoc, mul_left_comm, mul_comm, add_comm,
            sub_eq_add_neg]
          -- remaining goal is just factoring the constant `((∑ exp)⁻¹)^2` through the finite sum
          classical
          set a : ℝ := (∑ i : Fin (Nat.succ n), Real.exp (x i))⁻¹ with ha
          -- rewrite powers and associate/commute multiplications
          simp (config := { failIfUnchanged := false })
            [ha.symm, pow_two] at *

          -- Normalize the two sums so we can use `Finset.sum_mul` to factor out constants on the
          -- right.
          have hnorm₂ :
              (∑ x_1 : Fin (Nat.succ n), Real.exp (x x_1) * (dx x_1 * (a * a)))
                =
              ∑ x_1 : Fin (Nat.succ n), (Real.exp (x x_1) * dx x_1) * (a * a) := by
            classical
            refine Finset.sum_congr rfl ?_
            intro x_1 _hx_1
            ac_rfl

          have hnorm₁ :
              (∑ x_1 : Fin (Nat.succ n), Real.exp (x x_1) * (dx x_1 * a))
                =
              ∑ x_1 : Fin (Nat.succ n), (Real.exp (x x_1) * dx x_1) * a := by
            classical
            refine Finset.sum_congr rfl ?_
            intro x_1 _hx_1
            ac_rfl

          have hsum_mul₂ :
              (∑ x_1 : Fin (Nat.succ n), (Real.exp (x x_1) * dx x_1) * (a * a))
                =
              (∑ x_1 : Fin (Nat.succ n), Real.exp (x x_1) * dx x_1) * (a * a) := by
            simpa [mul_assoc, mul_left_comm, mul_comm] using
              (Finset.sum_mul (s := (Finset.univ : Finset (Fin (Nat.succ n))))
                (f := fun x_1 : Fin (Nat.succ n) => Real.exp (x x_1) * dx x_1)
                (a := a * a)).symm

          have hsum_mul₁ :
              (∑ x_1 : Fin (Nat.succ n), (Real.exp (x x_1) * dx x_1) * a)
                =
              (∑ x_1 : Fin (Nat.succ n), Real.exp (x x_1) * dx x_1) * a := by
            simpa [mul_assoc, mul_left_comm, mul_comm] using
              (Finset.sum_mul (s := (Finset.univ : Finset (Fin (Nat.succ n))))
                (f := fun x_1 : Fin (Nat.succ n) => Real.exp (x x_1) * dx x_1)
                (a := a)).symm

          -- Finish by factoring, then commuting multiplications in ℝ.
          calc
            (∑ x_1 : Fin (Nat.succ n), Real.exp (x x_1) * (dx x_1 * (a * a)))
                = (∑ x_1 : Fin (Nat.succ n), (Real.exp (x x_1) * dx x_1) * (a * a)) := hnorm₂
            _ = (∑ x_1 : Fin (Nat.succ n), Real.exp (x x_1) * dx x_1) * (a * a) := hsum_mul₂
            _ = a * ((∑ x_1 : Fin (Nat.succ n), Real.exp (x x_1) * dx x_1) * a) := by
                  simp [mul_left_comm]
            _ = a * (∑ x_1 : Fin (Nat.succ n), Real.exp (x x_1) * (dx x_1 * a)) := by
                  -- rewrite the inner `(...)*a` back into a sum form
                  rw [← hsum_mul₁]
                  rw [hnorm₁]
        -- apply the congruence results
        refine (hmul.congr_of_eventuallyEq hfun.eventuallyEq).congr_fderiv hderiv

      -- assemble into vector-valued derivative
      -- First: prove the derivative of the coordinate function `x ↦ (fun i => softmaxVec x i)`.
      have hpi :
          HasFDerivAt
            (fun x : Vec (Nat.succ n) => (fun i : Fin (Nat.succ n) => softmaxVec (n := Nat.succ n) x
              i))
            (ContinuousLinearMap.pi (fun i : Fin (Nat.succ n) => softmaxDerivCoord (n := Nat.succ n)
              x i)) x := by
        refine (hasFDerivAt_pi (𝕜 := ℝ)
          (φ := fun i : Fin (Nat.succ n) => fun x : Vec (Nat.succ n) => softmaxVec (n := Nat.succ n)
            x i)
          (φ' := fun i : Fin (Nat.succ n) => softmaxDerivCoord (n := Nat.succ n) x i)
          (x := x)).2 ?_
        intro i
        simpa using (hcoord_soft i)

      -- Second: convert the `Fin n → ℝ` statement into a `Vec n` statement via the linear isometry
      -- `(e _).symm`.
      have hcomp :
          HasFDerivAt (fun x : Vec (Nat.succ n) => (euclideanEquiv (Nat.succ n)).symm fun i =>
            softmaxVec (n := Nat.succ n) x i)
            ((euclideanEquiv (Nat.succ n)).symm.toContinuousLinearMap.comp
              (ContinuousLinearMap.pi (fun i : Fin (Nat.succ n) => softmaxDerivCoord (n := Nat.succ
                n) x i))) x :=
        (((euclideanEquiv (Nat.succ n)).symm.hasFDerivAt (x := fun i => softmaxVec (n := Nat.succ n)
          x i)).comp x hpi)

      -- Finally, simplify the LHS to `softmaxVec` and the derivative to `softmaxDerivCLM`.
      simpa [softmaxVec, softmaxDerivCLM, softmaxVecOfFun, euclideanEquiv] using hcomp

end
end Autograd
end Proofs
