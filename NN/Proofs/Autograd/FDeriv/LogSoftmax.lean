/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.FDeriv.Softmax

public import Mathlib.Analysis.Calculus.FDeriv.Add
public import Mathlib.Analysis.Calculus.FDeriv.Comp
public import Mathlib.Analysis.Calculus.FDeriv.Pi
public import Mathlib.Analysis.InnerProductSpace.Calculus

/-!
# LogSoftmax

Fréchet-derivative facts for **log-softmax** on Euclidean vectors.

This is the analytic (ℝ) ingredient used to justify `log_softmax` nodes
(`Vec n → Vec n`) in the tape/DAG autograd proofs.

## References
- Baydin et al., *Automatic Differentiation in Machine Learning: a Survey* (JMLR 2018).
- PyTorch docs for naming/behavior alignment (not used for theorems):
  https://pytorch.org/docs/stable/generated/torch.nn.functional.log_softmax.html
-/

@[expose] public section


namespace Proofs
namespace Autograd

open scoped BigOperators

noncomputable section

/-- `sumExp x` is strictly positive when the index type is nonempty. -/
lemma sumExp_pos {n : Nat} (x : Vec (Nat.succ n)) : 0 < sumExp (n := Nat.succ n) x := by
  have hterm : ∀ i : Fin (Nat.succ n), 0 < Real.exp (x i) := fun i => Real.exp_pos (x i)
  simpa [sumExp] using Finset.sum_pos (fun i _ => hterm i) (Finset.univ_nonempty)

/-- Convenience corollary: `sumExp x ≠ 0` (for `n = succ _`). -/
lemma sumExp_ne_zero {n : Nat} (x : Vec (Nat.succ n)) : sumExp (n := Nat.succ n) x ≠ 0 :=
  ne_of_gt (sumExp_pos (n := n) x)

/--
Log-softmax on Euclidean vectors.

For `n = succ _`:
`logSoftmaxVec x i = xᵢ - log(sumExp x)`.

The `n = 0` branch is the identity on the trivial space.
-/
def logSoftmaxVec : {n : Nat} → Vec n → Vec n
  | 0, x => x
  | Nat.succ n, x =>
      softmaxVecOfFun (n := Nat.succ n) fun i => x i - Real.log (sumExp (n := Nat.succ n) x)

/--
The `i`th output coordinate of the log-softmax derivative at `x` (for `n = succ _`).

If `y = softmaxVec x`, then this is the linear functional `dx ↦ dxᵢ - ⟪y, dx⟫`.
-/
def logSoftmaxDerivCoord {n : Nat} (x : Vec (Nat.succ n)) (i : Fin (Nat.succ n)) :
    Vec (Nat.succ n) →L[ℝ] ℝ :=
  let y := softmaxVec (n := Nat.succ n) x
  evalCLM (n := Nat.succ n) i - dotCLM (n := Nat.succ n) y

/-- The full Fréchet derivative of `logSoftmaxVec` at `x`, packaged as a CLM. -/
def logSoftmaxDerivCLM : {n : Nat} → Vec n → Vec n →L[ℝ] Vec n
  | 0, _x => (1 : (Vec 0) →L[ℝ] (Vec 0))
  | Nat.succ n, x =>
      (euclideanEquiv (Nat.succ n)).symm.toContinuousLinearMap.comp <|
        ContinuousLinearMap.pi (fun i : Fin (Nat.succ n) => logSoftmaxDerivCoord (n := n) x i)

/--
Closed-form JVP (directional derivative) for log-softmax.

For `n = succ _`, if `y = softmaxVec x` and `s = ⟪y, dx⟫`, then `(logSoftmaxJvp x dx)ᵢ = dxᵢ - s`.
-/
def logSoftmaxJvp : {n : Nat} → Vec n → Vec n → Vec n
  | 0, _x, dx => dx
  | Nat.succ n, x, dx =>
      let y := softmaxVec (n := Nat.succ n) x
      let s : ℝ := dotCLM (n := Nat.succ n) y dx
      softmaxVecOfFun (n := Nat.succ n) fun i => dx i - s

/--
Closed-form VJP for log-softmax (transpose-Jacobian product).

For `n = succ _`, if `y = softmaxVec x` and `t = ∑ᵢ δᵢ`, then `(logSoftmaxVjp x δ)ᵢ = δᵢ - yᵢ * t`.
-/
def logSoftmaxVjp : {n : Nat} → Vec n → Vec n → Vec n
  | 0, _x, δ => δ
  | Nat.succ n, x, δ =>
      let y := softmaxVec (n := Nat.succ n) x
      let t : ℝ := ∑ i : Fin (Nat.succ n), δ i
      softmaxVecOfFun (n := Nat.succ n) fun i => δ i - y i * t

/-- The closed-form JVP `logSoftmaxJvp` agrees with the CLM derivative `logSoftmaxDerivCLM`. -/
theorem logSoftmaxJvp_eq_deriv {n : Nat} (x dx : Vec n) :
    logSoftmaxJvp (n := n) x dx = (logSoftmaxDerivCLM (n := n) x) dx := by
  classical
  cases n with
  | zero =>
      ext i
      exact i.elim0
  | succ n =>
      ext i
      have hR :
          ((logSoftmaxDerivCLM (n := Nat.succ n) x) dx) i =
            (logSoftmaxDerivCoord (n := n) x i) dx := by
        simp [logSoftmaxDerivCLM, logSoftmaxDerivCoord, euclideanEquiv]
      rw [hR]
      simp [logSoftmaxJvp, logSoftmaxDerivCoord, dotCLM_apply, evalCLM_apply,
        ContinuousLinearMap.add_apply,
        sub_eq_add_neg, mul_comm]

/--
Adjointness identity: the log-softmax JVP and VJP are adjoint w.r.t. the Euclidean inner product.

This is the analytic statement that justifies using `logSoftmaxVjp` as backward.
-/
theorem inner_logSoftmaxJvp_vjp {n : Nat} (x dx δ : Vec n) :
    inner ℝ (logSoftmaxJvp (n := n) x dx) δ = inner ℝ dx (logSoftmaxVjp (n := n) x δ) := by
  classical
  cases n with
  | zero =>
      simp [logSoftmaxJvp, logSoftmaxVjp]
  | succ n =>
      let y : Vec (Nat.succ n) := softmaxVec (n := Nat.succ n) x
      let sdx : ℝ := dotCLM (n := Nat.succ n) y dx
      let tδ : ℝ := ∑ i : Fin (Nat.succ n), δ i
      have hsdx : sdx = ∑ i : Fin (Nat.succ n), y i * dx i := by
        simp [sdx, dotCLM_apply, mul_comm]
      calc
        inner ℝ (logSoftmaxJvp (n := Nat.succ n) x dx) δ
            = ∑ i : Fin (Nat.succ n), (logSoftmaxJvp (n := Nat.succ n) x dx i) * δ i := by
                simp [inner_eq_sum_mul]
        _ = ∑ i : Fin (Nat.succ n), (dx i - sdx) * δ i := by
              simp [logSoftmaxJvp, y, sdx]
        _ =
            (∑ i : Fin (Nat.succ n), dx i * δ i) - sdx * (∑ i : Fin (Nat.succ n), δ i) := by
              have hsplit :
                  (∑ i : Fin (Nat.succ n), (dx i - sdx) * δ i)
                    =
                  (∑ i : Fin (Nat.succ n), dx i * δ i) - (∑ i : Fin (Nat.succ n), sdx * δ i) := by
                calc
                  (∑ i : Fin (Nat.succ n), (dx i - sdx) * δ i)
                      =
                      ∑ i : Fin (Nat.succ n), (dx i * δ i - sdx * δ i) := by
                        refine Finset.sum_congr rfl ?_
                        intro i _hi
                        ring_nf
                  _ =
                      (∑ i : Fin (Nat.succ n), dx i * δ i) - (∑ i : Fin (Nat.succ n), sdx * δ i) :=
                        by
                        simp [Finset.sum_sub_distrib]
              have hfactor : (∑ i : Fin (Nat.succ n), sdx * δ i) = sdx * (∑ i : Fin (Nat.succ n), δ
                i) := by
                simp [Finset.mul_sum]
              simpa [hfactor] using hsplit
        _ = (∑ i : Fin (Nat.succ n), dx i * δ i) - sdx * tδ := by
              simp [tδ]
        _ = (∑ i : Fin (Nat.succ n), dx i * δ i) - tδ * sdx := by
              ring_nf
        _ =
            (∑ i : Fin (Nat.succ n), dx i * δ i) -
              tδ * (∑ i : Fin (Nat.succ n), y i * dx i) := by
              simp [hsdx]
        _ = ∑ i : Fin (Nat.succ n), dx i * (δ i - y i * tδ) := by
              have hsplit :
                  (∑ i : Fin (Nat.succ n), dx i * (δ i - y i * tδ))
                    =
                  (∑ i : Fin (Nat.succ n), dx i * δ i) - tδ * (∑ i : Fin (Nat.succ n), y i * dx i)
                    := by
                have hsplit0 :
                    (∑ i : Fin (Nat.succ n), dx i * (δ i - y i * tδ))
                      =
                    (∑ i : Fin (Nat.succ n), dx i * δ i) -
                      (∑ i : Fin (Nat.succ n), (y i * dx i) * tδ) := by
                  calc
                    (∑ i : Fin (Nat.succ n), dx i * (δ i - y i * tδ))
                        =
                        ∑ i : Fin (Nat.succ n), (dx i * δ i - dx i * (y i * tδ)) := by
                          refine Finset.sum_congr rfl ?_
                          intro i _hi
                          ring_nf
                    _ =
                        (∑ i : Fin (Nat.succ n), dx i * δ i) -
                          (∑ i : Fin (Nat.succ n), dx i * (y i * tδ)) := by
                          simp [Finset.sum_sub_distrib]
                    _ =
                        (∑ i : Fin (Nat.succ n), dx i * δ i) -
                          (∑ i : Fin (Nat.succ n), (y i * dx i) * tδ) := by
                          refine congrArg (fun z => (∑ i : Fin (Nat.succ n), dx i * δ i) - z) ?_
                          refine Finset.sum_congr rfl ?_
                          intro i _hi
                          ring_nf
                have hfactor0 :
                    (∑ i : Fin (Nat.succ n), (y i * dx i) * tδ) = tδ * (∑ i : Fin (Nat.succ n), y i
                      * dx i) := by
                  simp [Finset.mul_sum, mul_comm]
                simpa [hfactor0] using hsplit0
              simpa using hsplit.symm
        _ = ∑ i : Fin (Nat.succ n), dx i * (logSoftmaxVjp (n := Nat.succ n) x δ i) := by
              simp [logSoftmaxVjp, y, tδ, mul_comm]
        _ = inner ℝ dx (logSoftmaxVjp (n := Nat.succ n) x δ) := by
              simp [inner_eq_sum_mul]

/-- Log-softmax is Fréchet-differentiable everywhere, with derivative `logSoftmaxDerivCLM`. -/
theorem hasFDerivAt_logSoftmaxVec {n : Nat} (x : Vec n) :
    HasFDerivAt (logSoftmaxVec (n := n)) (logSoftmaxDerivCLM (n := n) x) x := by
  classical
  cases n with
  | zero =>
      have hD : logSoftmaxDerivCLM (n := 0) x = (1 : (Vec 0) →L[ℝ] (Vec 0)) := rfl
      simpa [logSoftmaxVec, hD] using ((1 : (Vec 0) →L[ℝ] (Vec 0)).hasFDerivAt (x := x))
  | succ n =>
      -- Derivative of `sumExp`.
      let sumDeriv : Vec (Nat.succ n) →L[ℝ] ℝ :=
        ∑ j : Fin (Nat.succ n), (evalCLM (n := Nat.succ n) j).smulRight (Real.exp (x j))
      have hsumF : HasFDerivAt (sumExp (n := Nat.succ n)) sumDeriv x := by
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
        have hEq :
            (∑ j : Fin (Nat.succ n), (fun x : Vec (Nat.succ n) => Real.exp (x j)))
              =
            (sumExp (n := Nat.succ n)) := by
          funext x
          simp [sumExp]
        have hsum' :
            HasFDerivAt (sumExp (n := Nat.succ n)) (∑ j : Fin (Nat.succ n),
                (evalCLM (n := Nat.succ n) j).smulRight (Real.exp (x j))) x :=
          hsum.congr_of_eventuallyEq hEq.symm.eventuallyEq
        simpa [sumDeriv] using hsum'

      -- Derivative of `x ↦ log(sumExp x)`.
      let y : Vec (Nat.succ n) := softmaxVec (n := Nat.succ n) x
      let logDeriv : Vec (Nat.succ n) →L[ℝ] ℝ :=
        (dotCLM (n := Nat.succ n) y)
      have hlog :
          HasFDerivAt (fun x : Vec (Nat.succ n) => Real.log (sumExp (n := Nat.succ n) x)) logDeriv x
            := by
        have hsum_ne : sumExp (n := Nat.succ n) x ≠ 0 := sumExp_ne_zero (n := n) x
        have hlog0 : HasDerivAt Real.log ((sumExp (n := Nat.succ n) x)⁻¹) (sumExp (n := Nat.succ n)
          x) :=
          Real.hasDerivAt_log hsum_ne
        have hlogF :
            HasFDerivAt Real.log
              (ContinuousLinearMap.smulRight (M₁ := ℝ) (M₂ := ℝ) (R := ℝ) (S := ℝ)
                (1 : ℝ →L[ℝ] ℝ) ((sumExp (n := Nat.succ n) x)⁻¹)) (sumExp (n := Nat.succ n) x) :=
          hlog0.hasFDerivAt
        have hcomp := hlogF.comp x hsumF
        -- simplify the composed derivative to `dotCLM y`
        have hlin :
            (ContinuousLinearMap.smulRight (M₁ := ℝ) (M₂ := ℝ) (R := ℝ) (S := ℝ)
                (1 : ℝ →L[ℝ] ℝ) ((sumExp (n := Nat.succ n) x)⁻¹)).comp sumDeriv
              =
            logDeriv := by
          ext dx
          -- `(∑ exp(xj) * dxj) / sumExp = ∑ softmax(xj) * dxj`
          simp [logDeriv, y, sumDeriv, dotCLM_apply, softmaxVec, sumExp, div_eq_mul_inv,
            ContinuousLinearMap.comp_apply, ContinuousLinearMap.smulRight_apply,
            evalCLM_apply, mul_assoc, mul_left_comm, mul_comm]
        exact hcomp.congr_fderiv hlin

      -- Assemble coordinatewise via `hasFDerivAt_pi`, then transport through `euclideanEquiv`.
      have hcoord :
          ∀ i : Fin (Nat.succ n),
            HasFDerivAt (fun x : Vec (Nat.succ n) => logSoftmaxVec (n := Nat.succ n) x i)
              (logSoftmaxDerivCoord (n := n) x i) x := by
        intro i
        have hid : HasFDerivAt (fun x : Vec (Nat.succ n) => x i) (evalCLM (n := Nat.succ n) i) x :=
          by
          simpa using ((evalCLM (n := Nat.succ n) i).hasFDerivAt (x := x))
        have hsub := hid.sub hlog
        have hfun :
            (fun x : Vec (Nat.succ n) => x i - Real.log (sumExp (n := Nat.succ n) x))
              =
            (fun x : Vec (Nat.succ n) => logSoftmaxVec (n := Nat.succ n) x i) := by
          funext x'
          simp [logSoftmaxVec, softmaxVecOfFun]
        -- derivative already matches by definition
        simpa [logSoftmaxDerivCoord, hfun] using hsub

      have hFun :
          HasFDerivAt (fun x : Vec (Nat.succ n) => fun i : Fin (Nat.succ n) => logSoftmaxVec (n :=
            Nat.succ n) x i)
            (ContinuousLinearMap.pi (fun i : Fin (Nat.succ n) => logSoftmaxDerivCoord (n := n) x i))
              x := by
        refine (hasFDerivAt_pi (𝕜 := ℝ)
          (φ := fun i : Fin (Nat.succ n) => fun x : Vec (Nat.succ n) => logSoftmaxVec (n := Nat.succ
            n) x i)
          (φ' := fun i : Fin (Nat.succ n) => logSoftmaxDerivCoord (n := n) x i)
          (x := x)).2 ?_
        intro i
        simpa using hcoord i
      have he' :
          HasFDerivAt (fun g : Fin (Nat.succ n) → ℝ => (euclideanEquiv (Nat.succ n)).symm g)
            ((euclideanEquiv (Nat.succ n)).symm.toContinuousLinearMap)
            (fun i : Fin (Nat.succ n) => logSoftmaxVec (n := Nat.succ n) x i) :=
        (ContinuousLinearMap.hasFDerivAt ((euclideanEquiv (Nat.succ n)).symm.toContinuousLinearMap))
      have hcomp := he'.comp x hFun
      simpa [logSoftmaxVec, logSoftmaxDerivCLM, euclideanEquiv, ContinuousLinearMap.comp_apply]
        using hcomp

end
end Autograd
end Proofs
