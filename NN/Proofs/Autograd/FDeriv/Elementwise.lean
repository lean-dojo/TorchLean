/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.FDeriv.OpSpec

public import Mathlib.Analysis.Calculus.FDeriv.Pi

/-!
# Elementwise

Elementwise (`map`) Fréchet-derivative facts for Euclidean vectors.

This is the missing bridge for turning the scalar calculus lemmas in
`NN/Proofs/Gradients/Activation.lean` into `OpSpecFDerivCorrect` instances for
vector-valued ops (sigmoid/tanh/softplus/…).
-/

@[expose] public section


namespace Proofs
namespace Autograd

open Spec
open Tensor
open scoped BigOperators

noncomputable section

-- ---------------------------------------------------------------------------
-- Generic coordinatewise calculus (`Vec n → Vec n`)
-- ---------------------------------------------------------------------------

/--
Apply a scalar function `f : ℝ → ℝ` coordinatewise to a vector.

This is the Euclidean-space analogue of the tensor-level `map_spec`.
-/
def elemwiseVec {n : Nat} (f : ℝ → ℝ) : Vec n → Vec n :=
  fun x => (Proofs.Autograd.euclideanEquiv n).symm fun i => f (x i)

/-- Coordinate evaluation as a continuous linear map on `Vec n`. -/
def evalCLM {n : Nat} (i : Fin n) : Vec n →L[ℝ] ℝ := by
  classical
  let fLin : Vec n →ₗ[ℝ] ℝ :=
    { toFun := fun x => x i
      map_add' := by
        intro x y
        simp
      map_smul' := by
        intro a x
        simp }
  refine { toLinearMap := fLin, cont := ?_ }
  exact LinearMap.continuous_of_finiteDimensional (f := fLin)

@[simp] lemma evalCLM_apply {n : Nat} (i : Fin n) (x : Vec n) : evalCLM (n := n) i x = x i := rfl

/--
The derivative candidate for `elemwiseVec f` at a point `x`, built from a proposed scalar derivative
  `f'`.

Concretely: `(elemwiseDerivCLM f' x) dx` has coordinates `i ↦ f'(xᵢ) * dxᵢ`.
-/
def elemwiseDerivCLM {n : Nat} (f' : ℝ → ℝ) (x : Vec n) : Vec n →L[ℝ] Vec n :=
  (Proofs.Autograd.euclideanEquiv n).symm.toContinuousLinearMap.comp <|
    ContinuousLinearMap.pi (fun i : Fin n =>
      ContinuousLinearMap.smulRight (M₁ := Vec n) (M₂ := ℝ) (R := ℝ) (S := ℝ)
        (evalCLM (n := n) i) (f' (x i)))

/--
If `f` is differentiable everywhere with derivative `f'`, then `elemwiseVec f` is Fréchet
differentiable everywhere with derivative `elemwiseDerivCLM f'`.
-/
theorem hasFDerivAt_elemwiseVec {n : Nat} {f f' : ℝ → ℝ} (x : Vec n)
    (hf : ∀ z, HasDerivAt f (f' z) z) :
    HasFDerivAt (elemwiseVec (n := n) f) (elemwiseDerivCLM (n := n) f' x) x := by
  classical
  have hcoord :
      ∀ i : Fin n,
        HasFDerivAt (fun x : Vec n => f (x i))
          (ContinuousLinearMap.smulRight (M₁ := Vec n) (M₂ := ℝ) (R := ℝ) (S := ℝ)
            (evalCLM (n := n) i) (f' (x i))) x := by
    intro i
    have hf_i : HasDerivAt f (f' (x i)) (x i) := hf (x i)
    have hfF :
        HasFDerivAt f
          (ContinuousLinearMap.smulRight (M₁ := ℝ) (M₂ := ℝ) (R := ℝ) (S := ℝ)
            (1 : ℝ →L[ℝ] ℝ) (f' (x i))) (x i) :=
      hf_i.hasFDerivAt
    have happly :
        HasFDerivAt (fun x : Vec n => x i) (evalCLM (n := n) i) x := by
      simpa using ((evalCLM (n := n) i).hasFDerivAt (x := x))
    have hcomp := hfF.comp x happly
    have hlin :
        (ContinuousLinearMap.smulRight (M₁ := ℝ) (M₂ := ℝ) (R := ℝ) (S := ℝ)
            (1 : ℝ →L[ℝ] ℝ) (f' (x i))).comp (evalCLM (n := n) i)
          =
        ContinuousLinearMap.smulRight (M₁ := Vec n) (M₂ := ℝ) (R := ℝ) (S := ℝ)
          (evalCLM (n := n) i) (f' (x i)) := by
      ext dx
      simp [ContinuousLinearMap.smulRight_apply]
    exact hcomp.congr_fderiv hlin

  -- First prove the derivative as a map into `Fin n → ℝ`, then transport through `e n`.symm.
  have hFun :
      HasFDerivAt (fun x : Vec n => fun i : Fin n => f (x i))
        (ContinuousLinearMap.pi (fun i : Fin n =>
          ContinuousLinearMap.smulRight (M₁ := Vec n) (M₂ := ℝ) (R := ℝ) (S := ℝ)
            (evalCLM (n := n) i) (f' (x i)))) x := by
    refine (hasFDerivAt_pi (𝕜 := ℝ)
        (φ := fun i : Fin n => fun x : Vec n => f (x i))
        (φ' := fun i : Fin n =>
          ContinuousLinearMap.smulRight (M₁ := Vec n) (M₂ := ℝ) (R := ℝ) (S := ℝ)
            (evalCLM (n := n) i) (f' (x i)))
        (x := x)).2 ?_
    intro i
    simpa using hcoord i
  have he' :
      HasFDerivAt (fun g : Fin n → ℝ => (Proofs.Autograd.euclideanEquiv n).symm g)
        ((Proofs.Autograd.euclideanEquiv n).symm.toContinuousLinearMap)
        (fun i : Fin n => f (x i)) :=
    (ContinuousLinearMap.hasFDerivAt (Proofs.Autograd.euclideanEquiv n).symm.toContinuousLinearMap)
  have hcomp := he'.comp x hFun
  simpa [elemwiseVec, elemwiseDerivCLM, Proofs.Autograd.euclideanEquiv,
    ContinuousLinearMap.comp_apply] using hcomp

/--
Pointwise (at `x`) version of `hasFDerivAt_elemwiseVec`.

This is useful when the scalar `HasDerivAt` facts are only available at the coordinates of `x`.
-/
theorem hasFDerivAt_elemwiseVec_at {n : Nat} {f f' : ℝ → ℝ} (x : Vec n)
    (hf : ∀ i : Fin n, HasDerivAt f (f' (x i)) (x i)) :
    HasFDerivAt (elemwiseVec (n := n) f) (elemwiseDerivCLM (n := n) f' x) x := by
  classical
  have hcoord :
      ∀ i : Fin n,
        HasFDerivAt (fun x : Vec n => f (x i))
          (ContinuousLinearMap.smulRight (M₁ := Vec n) (M₂ := ℝ) (R := ℝ) (S := ℝ)
            (evalCLM (n := n) i) (f' (x i))) x := by
    intro i
    have hf_i : HasDerivAt f (f' (x i)) (x i) := hf i
    have hfF :
        HasFDerivAt f
          (ContinuousLinearMap.smulRight (M₁ := ℝ) (M₂ := ℝ) (R := ℝ) (S := ℝ)
            (1 : ℝ →L[ℝ] ℝ) (f' (x i))) (x i) :=
      hf_i.hasFDerivAt
    have happly : HasFDerivAt (fun x : Vec n => x i) (evalCLM (n := n) i) x := by
      simpa using ((evalCLM (n := n) i).hasFDerivAt (x := x))
    have hcomp := hfF.comp x happly
    have hlin :
        (ContinuousLinearMap.smulRight (M₁ := ℝ) (M₂ := ℝ) (R := ℝ) (S := ℝ)
            (1 : ℝ →L[ℝ] ℝ) (f' (x i))).comp (evalCLM (n := n) i)
          =
        ContinuousLinearMap.smulRight (M₁ := Vec n) (M₂ := ℝ) (R := ℝ) (S := ℝ)
          (evalCLM (n := n) i) (f' (x i)) := by
      ext dx
      simp [ContinuousLinearMap.smulRight_apply]
    exact hcomp.congr_fderiv hlin

  have hFun :
      HasFDerivAt (fun x : Vec n => fun i : Fin n => f (x i))
        (ContinuousLinearMap.pi (fun i : Fin n =>
          ContinuousLinearMap.smulRight (M₁ := Vec n) (M₂ := ℝ) (R := ℝ) (S := ℝ)
            (evalCLM (n := n) i) (f' (x i)))) x := by
    refine (hasFDerivAt_pi (𝕜 := ℝ)
        (φ := fun i : Fin n => fun x : Vec n => f (x i))
        (φ' := fun i : Fin n =>
          ContinuousLinearMap.smulRight (M₁ := Vec n) (M₂ := ℝ) (R := ℝ) (S := ℝ)
            (evalCLM (n := n) i) (f' (x i)))
        (x := x)).2 ?_
    intro i
    simpa using hcoord i
  have he' :
      HasFDerivAt (fun g : Fin n → ℝ => (Proofs.Autograd.euclideanEquiv n).symm g)
        ((Proofs.Autograd.euclideanEquiv n).symm.toContinuousLinearMap)
        (fun i : Fin n => f (x i)) :=
    (ContinuousLinearMap.hasFDerivAt (Proofs.Autograd.euclideanEquiv n).symm.toContinuousLinearMap)
  have hcomp := he'.comp x hFun
  simpa [elemwiseVec, elemwiseDerivCLM, Proofs.Autograd.euclideanEquiv,
    ContinuousLinearMap.comp_apply] using hcomp

/--
Evaluation lemma: converting an elementwise-mapped tensor back to coordinates agrees with applying
`f` to the corresponding Euclidean coordinate.
-/
@[simp] lemma toVec_map_spec_ofVecE {n : Nat} (f : ℝ → ℝ) (xV : Vec n) (i : Fin n) :
    Spec.toVec (mapSpec (s := .dim n .scalar) f (ofVecE xV)) i = f (xV i) := by
  -- `toVecE_map_spec` + `toVecE_ofVecE` and then evaluate at `i`.
  have h :=
    congrArg (fun v : Vec n => v.ofLp i) (toVecE_map_spec (n := n) f (t := ofVecE xV))
  -- Left: use `toVecE_ofLp`. Right: `ofLp` of `e.symm` is just function evaluation.
  simpa [toVecE_ofLp, toVecE_ofVecE, ofVecE, Proofs.Autograd.euclideanEquiv] using h

-- ---------------------------------------------------------------------------
-- `OpSpecFDerivCorrect` instances for common elementwise ops
-- ---------------------------------------------------------------------------

namespace OpSpecFDerivCorrect

/-- `exp` as an `OpSpecFDerivCorrect` instance (elementwise `Real.exp`). -/
def exp {n : Nat} : OpSpecFDerivCorrect n n :=
{
  correct := expCorrect (s := .dim n .scalar)
  deriv := fun xV => elemwiseDerivCLM (n := n) (fun z => Real.exp z) xV
  hasFDerivAt := by
    intro xV
    have h :=
      hasFDerivAt_elemwiseVec (n := n) (x := xV) (f := fun z => Real.exp z) (f' := fun z => Real.exp
        z)
        (fun z => Real.hasDerivAt_exp z)
    -- Rewrite the forward function through `toVecE/map_spec`.
    simpa [OpSpecFDerivCorrect.forwardVec, expCorrect, Spec.expOp, Spec.liftElementwise,
      expSpec, elemwiseVec, elemwiseDerivCLM, toVecE_map_spec] using h
  jvp_eq := by
    intro xV dxV
    ext i
    -- LHS: unfold the JVP definition.
    have hL :
        toVecE ((expCorrect (s := .dim n .scalar)).jvp (ofVecE xV) (ofVecE dxV)) i
          =
        dxV i * Spec.toVec (mapSpec (s := .dim n .scalar) MathFunctions.exp (ofVecE xV)) i := by
      simp [expCorrect, Spec.expOp, expSpec,
        toVecE, ofVecE, Spec.toVec_ofVec, Spec.toVec_mul_spec]
    -- simplify the `map_spec` term.
    have hMap :
        Spec.toVec (mapSpec (s := .dim n .scalar) MathFunctions.exp (ofVecE xV)) i =
          MathFunctions.exp (xV i) := by
      simp
    -- RHS: apply the derivative CLM at coordinate `i`.
    have hR :
        (elemwiseDerivCLM (n := n) (fun z => Real.exp z) xV) dxV i = dxV i * Real.exp (xV i) := by
      rfl
    -- Combine.
    simpa [hMap] using (hL.trans hR.symm)
}

/-- `square` as an `OpSpecFDerivCorrect` instance (elementwise `x ↦ x^2`). -/
def square {n : Nat} : OpSpecFDerivCorrect n n :=
{
  correct := squareCorrect (s := .dim n .scalar)
  deriv := fun xV => elemwiseDerivCLM (n := n) (fun z => (Numbers.two : ℝ) * z) xV
  hasFDerivAt := by
    intro xV
    have h :=
      hasFDerivAt_elemwiseVec (n := n) (x := xV)
        (f := fun z : ℝ => z * z) (f' := fun z => (Numbers.two : ℝ) * z)
        (fun z => Proofs.square_deriv_correct (x := z))
    simpa [OpSpecFDerivCorrect.forwardVec, squareCorrect, Spec.squareOp,
      squareSpec, elemwiseVec, elemwiseDerivCLM, toVecE_map_spec, toVecE, ofVecE] using h
  jvp_eq := by
    intro xV dxV
    ext i
    have hL :
        toVecE ((squareCorrect (s := .dim n .scalar)).jvp (ofVecE xV) (ofVecE dxV)) i
          =
        dxV i *
          Spec.toVec (mulSpec (fill (Numbers.two : ℝ) (.dim n .scalar)) (ofVecE xV)) i := by
      simp [squareCorrect, Spec.squareOp, toVecE, ofVecE, Spec.toVec_ofVec,
        Spec.toVec_mul_spec]
    have hMap :
        Spec.toVec (mulSpec (fill (Numbers.two : ℝ) (.dim n .scalar)) (ofVecE xV)) i =
          (Numbers.two : ℝ) * xV i := by
      calc
        Spec.toVec (mulSpec (fill (Numbers.two : ℝ) (.dim n .scalar)) (ofVecE xV)) i
            =
          Spec.toVec (fill (Numbers.two : ℝ) (.dim n .scalar)) i *
            Spec.toVec (ofVecE xV) i := by
              exact Spec.toVec_mul_spec (a := fill (Numbers.two : ℝ) (.dim n .scalar))
                (b := ofVecE xV) (i := i)
        _ = (Numbers.two : ℝ) * xV i := by
              have hFill :
                  Spec.toVec (fill (Numbers.two : ℝ) (.dim n .scalar)) i =
                    (Numbers.two : ℝ) := by
                simp [fill, Spec.toVec]
              have hX : Spec.toVec (ofVecE xV) i = xV i := by
                simp [ofVecE]
              rw [hFill, hX]
    have hR :
        (elemwiseDerivCLM (n := n) (fun z => (Numbers.two : ℝ) * z) xV) dxV i =
          dxV i * ((Numbers.two : ℝ) * xV i) := by
      rfl
    simpa [hMap] using (hL.trans hR.symm)
}

/-- `sinh` as an `OpSpecFDerivCorrect` instance (elementwise). -/
def sinh {n : Nat} : OpSpecFDerivCorrect n n :=
{
  correct := sinhCorrect (s := .dim n .scalar)
  deriv := fun xV => elemwiseDerivCLM (n := n) (fun z => Real.cosh z) xV
  hasFDerivAt := by
    intro xV
    have h :=
      hasFDerivAt_elemwiseVec (n := n) (x := xV)
        (f := fun z => Activation.Math.sinhSpec z) (f' := fun z => Activation.Math.sinhDerivSpec z)
        (fun z => Proofs.sinh_deriv_correct (x := z))
    simpa [OpSpecFDerivCorrect.forwardVec, sinhCorrect, Spec.sinhOp, Spec.liftElementwise,
      sinhSpec, elemwiseVec, elemwiseDerivCLM, toVecE_map_spec, toVecE, ofVecE,
      Activation.Math.sinhDerivSpec, mathfunc_cosh_eq_rcosh] using h
  jvp_eq := by
    intro xV dxV
    ext i
    have hL :
        toVecE ((sinhCorrect (s := .dim n .scalar)).jvp (ofVecE xV) (ofVecE dxV)) i
          =
        dxV i * Spec.toVec (coshSpec (s := .dim n .scalar) (ofVecE xV)) i := by
      simp [sinhCorrect, Spec.sinhOp, coshSpec,
        toVecE, ofVecE, Spec.toVec_ofVec, Spec.toVec_mul_spec]
    have hMap :
        Spec.toVec (coshSpec (s := .dim n .scalar) (ofVecE xV)) i = Real.cosh (xV i) := by
      simp [coshSpec, mathfunc_cosh_eq_rcosh]
    have hR :
        (elemwiseDerivCLM (n := n) (fun z => Real.cosh z) xV) dxV i =
          dxV i * Real.cosh (xV i) := by
      rfl
    simpa [hMap] using (hL.trans hR.symm)
}

/-- `cosh` as an `OpSpecFDerivCorrect` instance (elementwise). -/
def cosh {n : Nat} : OpSpecFDerivCorrect n n :=
{
  correct := coshCorrect (s := .dim n .scalar)
  deriv := fun xV => elemwiseDerivCLM (n := n) (fun z => Real.sinh z) xV
  hasFDerivAt := by
    intro xV
    have h :=
      hasFDerivAt_elemwiseVec (n := n) (x := xV)
        (f := fun z => Activation.Math.coshSpec z) (f' := fun z => Activation.Math.coshDerivSpec z)
        (fun z => Proofs.cosh_deriv_correct (x := z))
    simpa [OpSpecFDerivCorrect.forwardVec, coshCorrect, Spec.coshOp, Spec.liftElementwise,
      coshSpec, elemwiseVec, elemwiseDerivCLM, toVecE_map_spec, toVecE, ofVecE,
      Activation.Math.coshDerivSpec, mathfunc_sinh_eq_rsinh] using h
  jvp_eq := by
    intro xV dxV
    ext i
    have hL :
        toVecE ((coshCorrect (s := .dim n .scalar)).jvp (ofVecE xV) (ofVecE dxV)) i
          =
        dxV i * Spec.toVec (sinhSpec (s := .dim n .scalar) (ofVecE xV)) i := by
      simp [coshCorrect, Spec.coshOp, sinhSpec,
        toVecE, ofVecE, Spec.toVec_ofVec, Spec.toVec_mul_spec]
    have hMap :
        Spec.toVec (sinhSpec (s := .dim n .scalar) (ofVecE xV)) i = Real.sinh (xV i) := by
      simp [sinhSpec, mathfunc_sinh_eq_rsinh]
    have hR :
        (elemwiseDerivCLM (n := n) (fun z => Real.sinh z) xV) dxV i =
          dxV i * Real.sinh (xV i) := by
      rfl
    simpa [hMap] using (hL.trans hR.symm)
}

/-- `tanh` as an `OpSpecFDerivCorrect` instance (elementwise). -/
def tanh {n : Nat} : OpSpecFDerivCorrect n n :=
{
  correct := tanhCorrect (s := .dim n .scalar)
  deriv := fun xV => elemwiseDerivCLM (n := n) Activation.Math.tanhDerivSpec xV
  hasFDerivAt := by
    intro xV
    have h :=
      hasFDerivAt_elemwiseVec (n := n) (x := xV)
        (f := fun z => Activation.Math.tanhSpec z) (f' := fun z => Activation.Math.tanhDerivSpec
          z)
        (fun z => Proofs.tanh_deriv_correct (x := z))
    simpa [OpSpecFDerivCorrect.forwardVec, tanhCorrect, Spec.tanhOp, Spec.liftElementwise,
      elemwiseVec, elemwiseDerivCLM, toVecE_map_spec, toVecE, ofVecE] using h
  jvp_eq := by
    intro xV dxV
    ext i
    have hL :
        toVecE ((tanhCorrect (s := .dim n .scalar)).jvp (ofVecE xV) (ofVecE dxV)) i
          =
        dxV i * Spec.toVec (mapSpec (s := .dim n .scalar) Activation.Math.tanhDerivSpec (ofVecE
          xV)) i := by
      simp [tanhCorrect, Spec.tanhOp, Spec.liftElementwise,
        Activation.tanhDerivSpec, toVecE, ofVecE, Spec.toVec_ofVec, Spec.toVec_mul_spec]
    have hMap :
        Spec.toVec (mapSpec (s := .dim n .scalar) Activation.Math.tanhDerivSpec (ofVecE xV)) i
          =
        Activation.Math.tanhDerivSpec (xV i) := by
      simp
    have hR :
        (elemwiseDerivCLM (n := n) Activation.Math.tanhDerivSpec xV) dxV i =
          dxV i * Activation.Math.tanhDerivSpec (xV i) := by
      rfl
    simpa [hMap] using (hL.trans hR.symm)
}

/-- `sigmoid` as an `OpSpecFDerivCorrect` instance (elementwise). -/
def sigmoid {n : Nat} : OpSpecFDerivCorrect n n :=
{
  correct := sigmoidCorrect (s := .dim n .scalar)
  deriv := fun xV => elemwiseDerivCLM (n := n) Activation.Math.sigmoidDerivSpec xV
  hasFDerivAt := by
    intro xV
    have h :=
      hasFDerivAt_elemwiseVec (n := n) (x := xV)
        (f := fun z => Activation.Math.sigmoidSpec z) (f' := fun z =>
          Activation.Math.sigmoidDerivSpec z)
        (fun z => Proofs.sigmoid_deriv_correct (x := z))
    simpa [OpSpecFDerivCorrect.forwardVec, sigmoidCorrect, Spec.sigmoidOp, Spec.liftElementwise,
      elemwiseVec, elemwiseDerivCLM, toVecE_map_spec, toVecE, ofVecE] using h
  jvp_eq := by
    intro xV dxV
    ext i
    have hL :
        toVecE ((sigmoidCorrect (s := .dim n .scalar)).jvp (ofVecE xV) (ofVecE dxV)) i
          =
        dxV i *
          Spec.toVec (mapSpec (s := .dim n .scalar) Activation.Math.sigmoidDerivSpec (ofVecE xV))
            i := by
      simp [sigmoidCorrect, Spec.sigmoidOp, Spec.liftElementwise,
        Activation.sigmoidDerivSpec, toVecE, ofVecE, Spec.toVec_ofVec, Spec.toVec_mul_spec]
    have hMap :
        Spec.toVec (mapSpec (s := .dim n .scalar) Activation.Math.sigmoidDerivSpec (ofVecE xV)) i
          =
        Activation.Math.sigmoidDerivSpec (xV i) := by
      simp
    have hR :
        (elemwiseDerivCLM (n := n) Activation.Math.sigmoidDerivSpec xV) dxV i =
          dxV i * Activation.Math.sigmoidDerivSpec (xV i) := by
      rfl
    simpa [hMap] using (hL.trans hR.symm)
}

/-- `softplus` as an `OpSpecFDerivCorrect` instance (elementwise). -/
def softplus {n : Nat} : OpSpecFDerivCorrect n n :=
{
  correct := softplusCorrect (s := .dim n .scalar)
  deriv := fun xV => elemwiseDerivCLM (n := n) Activation.Math.softplusDerivSpec xV
  hasFDerivAt := by
    intro xV
    have h :=
      hasFDerivAt_elemwiseVec (n := n) (x := xV)
        (f := fun z => Activation.Math.softplusSpec z) (f' := fun z =>
          Activation.Math.softplusDerivSpec z)
        (fun z => Proofs.softplus_deriv_correct (x := z))
    simpa [OpSpecFDerivCorrect.forwardVec, softplusCorrect, Spec.softplusOp,
      Spec.liftElementwise,
      elemwiseVec, elemwiseDerivCLM, toVecE_map_spec, toVecE, ofVecE] using h
  jvp_eq := by
    intro xV dxV
    ext i
    have hL :
        toVecE ((softplusCorrect (s := .dim n .scalar)).jvp (ofVecE xV) (ofVecE dxV)) i
          =
        dxV i *
          Spec.toVec (mapSpec (s := .dim n .scalar) Activation.Math.softplusDerivSpec (ofVecE
            xV)) i := by
      simp [softplusCorrect, Spec.softplusOp, Spec.liftElementwise,
        Activation.softplusDerivSpec, toVecE, ofVecE, Spec.toVec_ofVec, Spec.toVec_mul_spec]
    have hMap :
        Spec.toVec (mapSpec (s := .dim n .scalar) Activation.Math.softplusDerivSpec (ofVecE xV))
          i
          =
        Activation.Math.softplusDerivSpec (xV i) := by
      simp
    have hR :
        (elemwiseDerivCLM (n := n) Activation.Math.softplusDerivSpec xV) dxV i =
          dxV i * Activation.Math.softplusDerivSpec (xV i) := by
      rfl
    simpa [hMap] using (hL.trans hR.symm)
}

/-- SiLU as an `OpSpecFDerivCorrect` instance (elementwise). -/
def silu {n : Nat} : OpSpecFDerivCorrect n n :=
{
  correct := siluCorrect (s := .dim n .scalar)
  deriv := fun xV => elemwiseDerivCLM (n := n) Activation.Math.swishDerivSpec xV
  hasFDerivAt := by
    intro xV
    have h :=
      hasFDerivAt_elemwiseVec (n := n) (x := xV)
        (f := fun z => Activation.Math.swishSpec z) (f' := fun z =>
          Activation.Math.swishDerivSpec z)
        (fun z => Proofs.silu_deriv_correct (x := z))
    simpa [OpSpecFDerivCorrect.forwardVec, siluCorrect, Spec.swishOp,
      elemwiseVec, elemwiseDerivCLM, toVecE_map_spec, toVecE, ofVecE] using h
  jvp_eq := by
    intro xV dxV
    ext i
    have hL :
        toVecE ((siluCorrect (s := .dim n .scalar)).jvp (ofVecE xV) (ofVecE dxV)) i
          =
        dxV i *
          Spec.toVec (mapSpec (s := .dim n .scalar) Activation.Math.swishDerivSpec (ofVecE xV))
            i := by
      simp [siluCorrect, Spec.swishOp,
        Activation.swishDerivSpec, toVecE, ofVecE, Spec.toVec_ofVec, Spec.toVec_mul_spec]
    have hMap :
        Spec.toVec (mapSpec (s := .dim n .scalar) Activation.Math.swishDerivSpec (ofVecE xV)) i
          =
        Activation.Math.swishDerivSpec (xV i) := by
      simp
    have hR :
        (elemwiseDerivCLM (n := n) Activation.Math.swishDerivSpec xV) dxV i =
          dxV i * Activation.Math.swishDerivSpec (xV i) := by
      rfl
    simpa [hMap] using (hL.trans hR.symm)
}

/-- Tanh-approximate GELU as an `OpSpecFDerivCorrect` instance (elementwise). -/
def gelu {n : Nat} : OpSpecFDerivCorrect n n :=
{
  correct := geluCorrect (s := .dim n .scalar)
  deriv := fun xV => elemwiseDerivCLM (n := n) Activation.Math.geluDerivSpec xV
  hasFDerivAt := by
    intro xV
    have h :=
      hasFDerivAt_elemwiseVec (n := n) (x := xV)
        (f := fun z => Activation.Math.geluSpec z) (f' := fun z =>
          Activation.Math.geluDerivSpec z)
        (fun z => Proofs.gelu_deriv_correct (x := z))
    simpa [OpSpecFDerivCorrect.forwardVec, geluCorrect, Spec.geluOp,
      elemwiseVec, elemwiseDerivCLM, toVecE_map_spec, toVecE, ofVecE] using h
  jvp_eq := by
    intro xV dxV
    ext i
    have hL :
        toVecE ((geluCorrect (s := .dim n .scalar)).jvp (ofVecE xV) (ofVecE dxV)) i
          =
        dxV i *
          Spec.toVec (mapSpec (s := .dim n .scalar) Activation.Math.geluDerivSpec (ofVecE xV))
            i := by
      simp [geluCorrect, Spec.geluOp,
        Activation.geluDerivSpec, toVecE, ofVecE, Spec.toVec_ofVec, Spec.toVec_mul_spec]
    have hMap :
        Spec.toVec (mapSpec (s := .dim n .scalar) Activation.Math.geluDerivSpec (ofVecE xV)) i
          =
        Activation.Math.geluDerivSpec (xV i) := by
      simp
    have hR :
        (elemwiseDerivCLM (n := n) Activation.Math.geluDerivSpec xV) dxV i =
          dxV i * Activation.Math.geluDerivSpec (xV i) := by
      rfl
    simpa [hMap] using (hL.trans hR.symm)
}

/--
`safe_log` as an `OpSpecFDerivCorrect` instance (elementwise), assuming `ε > 0`.

This is the differentiable calculus fact; the corresponding dot-level VJP correctness lives in
`NN.Proofs.Autograd.Core.RealCorrectness`.
-/
def safeLog {n : Nat} (ε : ℝ) (hε : 0 < ε) : OpSpecFDerivCorrect n n :=
{
  correct := safeLogCorrect (s := .dim n .scalar) ε
  deriv := fun xV => elemwiseDerivCLM (n := n) (fun z => Activation.Math.safeLogDerivSpec z ε) xV
  hasFDerivAt := by
    intro xV
    have h :=
      hasFDerivAt_elemwiseVec (n := n) (x := xV)
        (f := fun z => Activation.Math.safeLogSpec z ε)
        (f' := fun z => Activation.Math.safeLogDerivSpec z ε)
        (fun z => Proofs.safe_log_deriv_correct (x := z) (ε := ε) hε)
    simpa [OpSpecFDerivCorrect.forwardVec, safeLogCorrect, Spec.safeLogOp,
      Spec.liftElementwise,
      elemwiseVec, elemwiseDerivCLM, toVecE_map_spec, toVecE, ofVecE] using h
  jvp_eq := by
    intro xV dxV
    ext i
    have hL :
        toVecE ((safeLogCorrect (s := .dim n .scalar) ε).jvp (ofVecE xV) (ofVecE dxV)) i
          =
        dxV i *
          Spec.toVec (mapSpec (s := .dim n .scalar) (fun x => Activation.Math.safeLogDerivSpec x
            ε) (ofVecE xV)) i := by
      simp [safeLogCorrect, Spec.safeLogOp, Spec.liftElementwise,
        Activation.safeLogDerivSpec, toVecE, ofVecE, Spec.toVec_ofVec, Spec.toVec_mul_spec]
    have hMap :
        Spec.toVec (mapSpec (s := .dim n .scalar) (fun x => Activation.Math.safeLogDerivSpec x
          ε) (ofVecE xV)) i
          =
        Activation.Math.safeLogDerivSpec (xV i) ε := by
      simp
    have hR :
        (elemwiseDerivCLM (n := n) (fun x => Activation.Math.safeLogDerivSpec x ε) xV) dxV i =
          dxV i * Activation.Math.safeLogDerivSpec (xV i) ε := by
      rfl
    simpa [hMap] using (hL.trans hR.symm)
}

/--
`smooth_abs` as an `OpSpecFDerivCorrect` instance (elementwise), assuming `ε > 0`.

This is a differentiable approximation to `abs`.
-/
def smoothAbs {n : Nat} (ε : ℝ) (hε : 0 < ε) : OpSpecFDerivCorrect n n :=
{
  correct := smoothAbsCorrect (s := .dim n .scalar) ε
  deriv := fun xV => elemwiseDerivCLM (n := n) (fun z => Activation.Math.smoothAbsDerivSpec z ε)
    xV
  hasFDerivAt := by
    intro xV
    have h :=
      hasFDerivAt_elemwiseVec (n := n) (x := xV)
        (f := fun z => Activation.Math.smoothAbsSpec z ε)
        (f' := fun z => Activation.Math.smoothAbsDerivSpec z ε)
        (fun z => Proofs.smooth_abs_deriv_correct (x := z) (ε := ε) hε)
    simpa [OpSpecFDerivCorrect.forwardVec, smoothAbsCorrect, Spec.smoothAbsOp,
      Spec.liftElementwise,
      elemwiseVec, elemwiseDerivCLM, toVecE_map_spec, toVecE, ofVecE] using h
  jvp_eq := by
    intro xV dxV
    ext i
    have hL :
        toVecE ((smoothAbsCorrect (s := .dim n .scalar) ε).jvp (ofVecE xV) (ofVecE dxV)) i
          =
        dxV i *
          Spec.toVec (mapSpec (s := .dim n .scalar) (fun x => Activation.Math.smoothAbsDerivSpec
            x ε) (ofVecE xV)) i := by
      simp [smoothAbsCorrect, Spec.smoothAbsOp, Spec.liftElementwise,
        Activation.smoothAbsDerivSpec, toVecE, ofVecE, Spec.toVec_ofVec, Spec.toVec_mul_spec]
    have hMap :
        Spec.toVec (mapSpec (s := .dim n .scalar) (fun x => Activation.Math.smoothAbsDerivSpec x
          ε) (ofVecE xV)) i
          =
        Activation.Math.smoothAbsDerivSpec (xV i) ε := by
      simp
    have hR :
        (elemwiseDerivCLM (n := n) (fun x => Activation.Math.smoothAbsDerivSpec x ε) xV) dxV i =
          dxV i * Activation.Math.smoothAbsDerivSpec (xV i) ε := by
      rfl
    simpa [hMap] using (hL.trans hR.symm)
}

end OpSpecFDerivCorrect

end
end Autograd
end Proofs
