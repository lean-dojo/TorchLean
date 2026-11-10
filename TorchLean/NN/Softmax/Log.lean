/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Softmax.FDeriv

/-!
# LogSoftmax

Fréchet derivatives of log-softmax on Euclidean vectors. These statements over $\mathbb R$ justify
the `logSoftmax` nodes (`Vec n → Vec n`) in the tape and DAG autograd proofs: the closed-form JVP
is the derivative, and the closed-form VJP is its adjoint.

## References
- Baydin et al., *Automatic Differentiation in Machine Learning: a Survey* (JMLR 2018).
- PyTorch docs, followed for naming and behavior (no theorem depends on them):
  https://pytorch.org/docs/stable/generated/torch.nn.functional.log_softmax.html
-/

@[expose] public section


namespace TorchLean

open scoped BigOperators

noncomputable section

/--
Log-softmax on Euclidean vectors: `logSoftmax x i = xᵢ - log(sumExp x)` for `n = succ _`, and
the identity on the trivial space when `n = 0`.
-/
def Vec.logSoftmax : {n : Nat} → Vec n → Vec n
  | 0, x => x
  | Nat.succ n, x =>
      WithLp.toLp 2 fun i => x i - Real.log (sumExp (n := Nat.succ n) x)

/--
Coordinate `i` of the log-softmax derivative at `x` (for `n = succ _`): the linear functional
`dx ↦ dxᵢ - ⟪y, dx⟫`, where `y = softmax x`.
-/
def Vec.logSoftmaxDerivCoord {n : Nat} (x : Vec (Nat.succ n)) (i : Fin (Nat.succ n)) :
    Vec (Nat.succ n) →L[ℝ] ℝ :=
  let y := softmax (n := Nat.succ n) x
  EuclideanSpace.proj (𝕜 := ℝ) i - innerSL ℝ y

/-- Fréchet derivative of `logSoftmax` at `x` as a continuous linear map: the coordinates
`Vec.logSoftmaxDerivCoord x i` for `n = succ _`, and the identity when `n = 0`. -/
def Vec.logSoftmaxDerivCLM : {n : Nat} → Vec n → Vec n →L[ℝ] Vec n
  | 0, _x => (1 : (Vec 0) →L[ℝ] (Vec 0))
  | Nat.succ n, x =>
      (EuclideanSpace.equiv (Fin (Nat.succ n)) ℝ).symm.toContinuousLinearMap.comp <|
        ContinuousLinearMap.pi (fun i : Fin (Nat.succ n) => logSoftmaxDerivCoord (n := n) x i)

/--
Closed-form JVP (directional derivative) for log-softmax.

For `n = succ _`, if `y = softmax x` and `s = ⟪y, dx⟫`, then `(logSoftmaxJVP x dx)ᵢ = dxᵢ - s`.
-/
def Vec.logSoftmaxJVP : {n : Nat} → Vec n → Vec n → Vec n
  | 0, _x, dx => dx
  | Nat.succ n, x, dx =>
      let y := softmax (n := Nat.succ n) x
      let s : ℝ := inner ℝ y dx
      WithLp.toLp 2 fun i => dx i - s

/--
Closed-form VJP for log-softmax (transpose-Jacobian product).

For `n = succ _`, if `y = softmax x` and `t = ∑ᵢ δᵢ`, then `(logSoftmaxVJP x δ)ᵢ = δᵢ - yᵢ * t`.
-/
def Vec.logSoftmaxVJP : {n : Nat} → Vec n → Vec n → Vec n
  | 0, _x, δ => δ
  | Nat.succ n, x, δ =>
      let y := softmax (n := Nat.succ n) x
      let t : ℝ := ∑ i : Fin (Nat.succ n), δ i
      WithLp.toLp 2 fun i => δ i - y i * t

/-- The closed-form JVP `logSoftmaxJVP` agrees with the CLM derivative `logSoftmaxDerivCLM`. -/
theorem Vec.logSoftmaxJVP_eq_deriv {n : Nat} (x dx : Vec n) :
    logSoftmaxJVP (n := n) x dx = (logSoftmaxDerivCLM (n := n) x) dx := by
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
        simp [logSoftmaxDerivCLM, logSoftmaxDerivCoord]
      rw [hR]
      simp [logSoftmaxJVP, logSoftmaxDerivCoord, inner_eq_sum_mul, sub_eq_add_neg, mul_comm]

/--
The log-softmax JVP and VJP are adjoint for the Euclidean inner product. With
`Vec.logSoftmaxJVP_eq_deriv`, this identity justifies `logSoftmaxVJP` as the backward rule.
-/
theorem Vec.inner_logSoftmaxJVP_vjp {n : Nat} (x dx δ : Vec n) :
    inner ℝ (logSoftmaxJVP (n := n) x dx) δ = inner ℝ dx (logSoftmaxVJP (n := n) x δ) := by
  classical
  cases n with
  | zero =>
      simp [logSoftmaxJVP, logSoftmaxVJP]
  | succ n =>
      let y : Vec (Nat.succ n) := softmax (n := Nat.succ n) x
      have hjvp (i : Fin (Nat.succ n)) :
          logSoftmaxJVP (n := Nat.succ n) x dx i = dx i - ∑ j, y j * dx j := by
        simp [logSoftmaxJVP, y, inner_eq_sum_mul]
      have hvjp (i : Fin (Nat.succ n)) :
          logSoftmaxVJP (n := Nat.succ n) x δ i = δ i - y i * ∑ j, δ j := by
        simp [logSoftmaxVJP, y]
      rw [inner_eq_sum_mul, inner_eq_sum_mul]
      simp only [hjvp, hvjp]
      -- Both sides equal `∑ᵢ dxᵢ δᵢ - (∑ⱼ yⱼ dxⱼ) (∑ⱼ δⱼ)`.
      have hleft :
          (∑ i, (dx i - ∑ j, y j * dx j) * δ i) =
            (∑ i, dx i * δ i) - (∑ j, y j * dx j) * ∑ i, δ i := by
        rw [Finset.mul_sum, ← Finset.sum_sub_distrib]
        exact Finset.sum_congr rfl fun i _ => by ring
      have hright :
          (∑ i, dx i * (δ i - y i * ∑ j, δ j)) =
            (∑ i, dx i * δ i) - (∑ j, δ j) * ∑ i, y i * dx i := by
        rw [Finset.mul_sum, ← Finset.sum_sub_distrib]
        exact Finset.sum_congr rfl fun i _ => by ring
      rw [hleft, hright]
      ring

/-- Log-softmax is Fréchet-differentiable everywhere, with derivative `logSoftmaxDerivCLM`. -/
theorem Vec.hasFDerivAt_logSoftmax {n : Nat} (x : Vec n) :
    HasFDerivAt (logSoftmax (n := n)) (logSoftmaxDerivCLM (n := n) x) x := by
  classical
  cases n with
  | zero =>
      have hD : logSoftmaxDerivCLM (n := 0) x = (1 : (Vec 0) →L[ℝ] (Vec 0)) := rfl
      rw [hD]
      change HasFDerivAt (fun x : Vec 0 => x) (1 : (Vec 0) →L[ℝ] (Vec 0)) x
      exact ((1 : (Vec 0) →L[ℝ] (Vec 0)).hasFDerivAt (x := x))
  | succ n =>
      -- Derivative of `sumExp`, a finite sum of the coordinate exponentials `x ↦ exp (x j)`.
      let sumDeriv : Vec (Nat.succ n) →L[ℝ] ℝ :=
        ∑ j : Fin (Nat.succ n), Real.exp (x j) • EuclideanSpace.proj (𝕜 := ℝ) j
      have hsumF : HasFDerivAt (sumExp (n := Nat.succ n)) sumDeriv x :=
        hasFDerivAt_sumExp x

      -- `log (Vec.sumExp x)` has derivative `dx ↦ ⟪Vec.softmax x, dx⟫`.
      let y : Vec (Nat.succ n) := softmax (n := Nat.succ n) x
      let logDeriv : Vec (Nat.succ n) →L[ℝ] ℝ := innerSL ℝ y
      have hlog :
          HasFDerivAt (fun x : Vec (Nat.succ n) => Real.log (sumExp (n := Nat.succ n) x)) logDeriv x
            := by
        -- `(∑ⱼ exp(xⱼ) dxⱼ) / sumExp x = ∑ⱼ softmax(x)ⱼ dxⱼ`.
        have hlin : (sumExp (n := Nat.succ n) x)⁻¹ • sumDeriv = logDeriv := by
          ext dx
          simp [logDeriv, y, sumDeriv, inner_eq_sum_mul, softmax, sumExp, div_eq_mul_inv,
            Finset.mul_sum, mul_assoc, mul_left_comm, mul_comm]
        exact (hsumF.log (sumExp_ne_zero (n := n) x)).congr_fderiv hlin

      -- Assemble coordinatewise via `hasFDerivAt_pi`, then transport through
      -- `EuclideanSpace.equiv`.
      have hcoord :
          ∀ i : Fin (Nat.succ n),
            HasFDerivAt (fun x : Vec (Nat.succ n) => (logSoftmax (n := Nat.succ n) x).ofLp i)
              (logSoftmaxDerivCoord (n := n) x i) x := by
        intro i
        have hid :
            HasFDerivAt (fun x : Vec (Nat.succ n) => x i) (EuclideanSpace.proj (𝕜 := ℝ) i) x :=
          (EuclideanSpace.proj (𝕜 := ℝ) i).hasFDerivAt
        have hsub := hid.sub hlog
        have hfun :
            ((fun x : Vec (Nat.succ n) => x.ofLp i) -
              fun x : Vec (Nat.succ n) => Real.log (sumExp (n := Nat.succ n) x))
              =
            (fun x : Vec (Nat.succ n) => (logSoftmax (n := Nat.succ n) x).ofLp i) := by
          funext x'
          simp [logSoftmax]
        -- After the rewrite, `hsub` matches the goal up to unfolding `Vec.logSoftmaxDerivCoord`.
        rw [hfun] at hsub
        simpa [logSoftmaxDerivCoord, logDeriv, y] using hsub

      have hFun :
          HasFDerivAt (fun x : Vec (Nat.succ n) => fun i : Fin (Nat.succ n) =>
            (logSoftmax (n := Nat.succ n) x).ofLp i)
            (ContinuousLinearMap.pi (fun i : Fin (Nat.succ n) => logSoftmaxDerivCoord (n := n) x i))
              x := by
        refine (hasFDerivAt_pi (𝕜 := ℝ)
          (φ := fun i : Fin (Nat.succ n) => fun x : Vec (Nat.succ n) =>
            (logSoftmax (n := Nat.succ n) x).ofLp i)
          (φ' := fun i : Fin (Nat.succ n) => logSoftmaxDerivCoord (n := n) x i)
          (x := x)).2 ?_
        intro i
        simpa using hcoord i
      have he' :
          HasFDerivAt
            (fun g : Fin (Nat.succ n) → ℝ => (EuclideanSpace.equiv (Fin (Nat.succ n)) ℝ).symm g)
            ((EuclideanSpace.equiv (Fin (Nat.succ n)) ℝ).symm.toContinuousLinearMap)
            (fun i : Fin (Nat.succ n) => (logSoftmax (n := Nat.succ n) x).ofLp i) :=
        (ContinuousLinearMap.hasFDerivAt
          ((EuclideanSpace.equiv (Fin (Nat.succ n)) ℝ).symm.toContinuousLinearMap))
      have hcomp := he'.comp x hFun
      change HasFDerivAt
        (fun x : Vec (Nat.succ n) => WithLp.toLp 2 fun i : Fin (Nat.succ n) =>
          x.ofLp i - Real.log (sumExp (n := Nat.succ n) x))
        ((EuclideanSpace.equiv (Fin (Nat.succ n)) ℝ).symm.toContinuousLinearMap.comp
          (ContinuousLinearMap.pi (fun i : Fin (Nat.succ n) => logSoftmaxDerivCoord (n := n) x i)))
        x
      simpa [logSoftmax, logSoftmaxDerivCLM, Function.comp_def,
        ContinuousLinearMap.comp_apply]
        using hcomp

end
end TorchLean
