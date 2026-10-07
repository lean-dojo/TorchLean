/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.FDeriv.OpSpec

/-!
# Elementwise

`OpSpecFDerivCorrect` instances for elementwise (`map`) ops on Euclidean vectors.

The generic coordinatewise calculus (`elemwiseVec`, `elemwiseDerivCLM`,
`hasFDerivAt_elemwiseVec_at`) lives in `NN.Proofs.Autograd.FDeriv.Core`. This file turns the
scalar calculus lemmas in `NN/Proofs/Gradients/Activation.lean` into `OpSpecFDerivCorrect`
instances for vector-valued ops (sigmoid/tanh/softplus/…).
-/

@[expose] public section

namespace Proofs
namespace Autograd

open Spec TorchLean
open TorchLean TorchLean.Tensor
open scoped BigOperators

noncomputable section

/--
Evaluation lemma: converting an elementwise-mapped tensor back to coordinates agrees with applying
`f` to the corresponding Euclidean coordinate.
-/
@[simp] theorem getScalar_map_spec_ofFnE {n : Nat} (f : ℝ → ℝ) (xV : Vec n) (i : Fin n) :
    TorchLean.Tensor.getScalar (Tensor.map f (ofFnE xV)) i = f (xV i) := by
  simp [ofFnE]

/-- Vectorized form of the previous lemma: mapping `f` over a tensor is `elemwiseVec f` on vectors.

This is the statement that lets an elementwise operation's derivative be proved once on
`EuclideanSpace` and then transported to every tensor of vector shape. -/
@[simp] theorem getScalarE_map_spec_ofFnE_eq_elemwiseVec {n : Nat} (f : ℝ → ℝ) (xV : Vec n) :
    getScalarE (Tensor.map (shape := .dim n .scalar) f (ofFnE xV)) =
      elemwiseVec (n := n) f xV := by
  ext i
  simp [elemwiseVec]

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
    have hfun :
        (fun xV : Vec n => getScalarE ((expCorrect (s := .dim n .scalar)).op.forward (ofFnE xV))) =
          elemwiseVec (n := n) (fun z => Real.exp z) := by
      funext xV
      ext i
      simp [expCorrect, Spec.expOp, expSpec, elemwiseVec, mathfunc_exp_eq_rexp]
    rw [hfun]
    exact h
  jvp_eq := by
    intro xV dxV
    ext i
    -- LHS: unfold the JVP definition.
    have hL :
        getScalarE ((expCorrect (s := .dim n .scalar)).jvp (ofFnE xV) (ofFnE dxV)) i
          =
        dxV i *
          TorchLean.Tensor.getScalar
            (Tensor.map (shape := .dim n .scalar) MathFunctions.exp (ofFnE xV)) i := by
      simp [expCorrect, Spec.expOp, expSpec,
        getScalarE, ofFnE, TorchLean.Tensor.getScalar_ofFn, Spec.getScalar_mul_spec]
    -- simplify the `map_spec` term.
    have hMap :
        TorchLean.Tensor.getScalar (Tensor.map MathFunctions.exp (ofFnE xV)) i =
          MathFunctions.exp (xV i) := by
      simp
    -- RHS: apply the derivative CLM at coordinate `i`.
    have hR :
        (elemwiseDerivCLM (n := n) (fun z => Real.exp z) xV) dxV i = dxV i * Real.exp (xV i) := by
      rfl
    -- Combine.
    rw [hMap] at hL
    exact hL.trans hR.symm
}

/-- `square` as an `OpSpecFDerivCorrect` instance (elementwise `x ↦ x^2`). -/
def square {n : Nat} : OpSpecFDerivCorrect n n :=
{
  correct := squareCorrect (s := .dim n .scalar)
  deriv := fun xV => elemwiseDerivCLM (n := n) (fun z => (2 : ℝ) * z) xV
  hasFDerivAt := by
    intro xV
    have h :=
      hasFDerivAt_elemwiseVec (n := n) (x := xV)
        (f := fun z : ℝ => z * z) (f' := fun z => (2 : ℝ) * z)
        (fun z => Proofs.square_deriv_correct (x := z))
    have hfun :
        (fun xV : Vec n =>
            getScalarE ((squareCorrect (s := .dim n .scalar)).op.forward (ofFnE xV))) =
          elemwiseVec (n := n) (fun z : ℝ => z * z) := by
      funext xV
      ext i
      simp [squareCorrect, Spec.squareOp, squareSpec, elemwiseVec]
    rw [hfun]
    exact h
  jvp_eq := by
    intro xV dxV
    ext i
    have hL :
        getScalarE ((squareCorrect (s := .dim n .scalar)).jvp (ofFnE xV) (ofFnE dxV)) i
          =
        dxV i *
          TorchLean.Tensor.getScalar
            (mulSpec (Tensor.full (.dim n .scalar) (2 : ℝ)) (ofFnE xV)) i := by
      simp [squareCorrect, Spec.squareOp, getScalarE, ofFnE, TorchLean.Tensor.getScalar_ofFn,
        Spec.getScalar_mul_spec]
    have hMap :
        TorchLean.Tensor.getScalar
            (mulSpec (Tensor.full (.dim n .scalar) (2 : ℝ)) (ofFnE xV)) i =
          (2 : ℝ) * xV i := by
      calc
        TorchLean.Tensor.getScalar
              (mulSpec (Tensor.full (.dim n .scalar) (2 : ℝ)) (ofFnE xV)) i =
          TorchLean.Tensor.getScalar (Tensor.full (.dim n .scalar) (2 : ℝ)) i *
            TorchLean.Tensor.getScalar (ofFnE xV) i := by
              exact Spec.getScalar_mul_spec (a := Tensor.full (.dim n .scalar) (2 : ℝ))
                (b := ofFnE xV) (i := i)
        _ = (2 : ℝ) * xV i := by
              have hFill :
                  TorchLean.Tensor.getScalar (Tensor.full (.dim n .scalar) (2 : ℝ)) i =
                    (2 : ℝ) := by
                exact TorchLean.Tensor.getScalar_full n (2 : ℝ) i
              have hX : TorchLean.Tensor.getScalar (ofFnE xV) i = xV i := by
                simp [ofFnE]
              rw [hFill, hX]
    have hR :
        (elemwiseDerivCLM (n := n) (fun z => (2 : ℝ) * z) xV) dxV i =
          dxV i * ((2 : ℝ) * xV i) := by
      rfl
    rw [hMap] at hL
    exact hL.trans hR.symm
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
    have hfun :
        (fun xV : Vec n =>
            getScalarE ((sinhCorrect (s := .dim n .scalar)).op.forward (ofFnE xV))) =
          elemwiseVec (n := n) (fun z => Activation.Math.sinhSpec z) := by
      funext xV
      ext i
      simp [sinhCorrect, Spec.sinhOp, sinhSpec, Activation.Math.sinhSpec, elemwiseVec,
        mathfunc_sinh_eq_rsinh]
    rw [hfun]
    exact h
  jvp_eq := by
    intro xV dxV
    ext i
    have hL :
        getScalarE ((sinhCorrect (s := .dim n .scalar)).jvp (ofFnE xV) (ofFnE dxV)) i
          =
        dxV i * TorchLean.Tensor.getScalar (coshSpec (s := .dim n .scalar) (ofFnE xV)) i := by
      simp [sinhCorrect, Spec.sinhOp, coshSpec,
        getScalarE, ofFnE, TorchLean.Tensor.getScalar_ofFn, Spec.getScalar_mul_spec]
    have hMap :
        TorchLean.Tensor.getScalar (coshSpec (s := .dim n .scalar) (ofFnE xV)) i
          = Real.cosh (xV i) := by
      simp [coshSpec, mathfunc_cosh_eq_rcosh]
    have hR :
        (elemwiseDerivCLM (n := n) (fun z => Real.cosh z) xV) dxV i =
          dxV i * Real.cosh (xV i) := by
      rfl
    rw [hMap] at hL
    exact hL.trans hR.symm
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
    have hfun :
        (fun xV : Vec n =>
            getScalarE ((coshCorrect (s := .dim n .scalar)).op.forward (ofFnE xV))) =
          elemwiseVec (n := n) (fun z => Activation.Math.coshSpec z) := by
      funext xV
      ext i
      simp [coshCorrect, Spec.coshOp, coshSpec, Activation.Math.coshSpec, elemwiseVec,
        mathfunc_cosh_eq_rcosh]
    rw [hfun]
    exact h
  jvp_eq := by
    intro xV dxV
    ext i
    have hL :
        getScalarE ((coshCorrect (s := .dim n .scalar)).jvp (ofFnE xV) (ofFnE dxV)) i
          =
        dxV i * TorchLean.Tensor.getScalar (sinhSpec (s := .dim n .scalar) (ofFnE xV)) i := by
      simp [coshCorrect, Spec.coshOp, sinhSpec,
        getScalarE, ofFnE, TorchLean.Tensor.getScalar_ofFn, Spec.getScalar_mul_spec]
    have hMap :
        TorchLean.Tensor.getScalar (sinhSpec (s := .dim n .scalar) (ofFnE xV)) i
          = Real.sinh (xV i) := by
      simp [sinhSpec, mathfunc_sinh_eq_rsinh]
    have hR :
        (elemwiseDerivCLM (n := n) (fun z => Real.sinh z) xV) dxV i =
          dxV i * Real.sinh (xV i) := by
      rfl
    rw [hMap] at hL
    exact hL.trans hR.symm
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
    have hfun :
        (fun xV : Vec n =>
            getScalarE ((tanhCorrect (s := .dim n .scalar)).op.forward (ofFnE xV))) =
          elemwiseVec (n := n) (fun z => Activation.Math.tanhSpec z) := by
      funext xV
      ext i
      simp [tanhCorrect, Spec.tanhOp, elemwiseVec]
    rw [hfun]
    exact h
  jvp_eq := by
    intro xV dxV
    ext i
    have hL :
        getScalarE ((tanhCorrect (s := .dim n .scalar)).jvp (ofFnE xV) (ofFnE dxV)) i
          =
        dxV i *
          TorchLean.Tensor.getScalar
            (Tensor.map (shape := .dim n .scalar) Activation.Math.tanhDerivSpec (ofFnE xV)) i := by
      simp [tanhCorrect, Spec.tanhOp,
        Activation.tanhDerivSpec, getScalarE, ofFnE,
        TorchLean.Tensor.getScalar_ofFn, Spec.getScalar_mul_spec]
    have hMap :
        TorchLean.Tensor.getScalar
            (Tensor.map (shape := .dim n .scalar) Activation.Math.tanhDerivSpec (ofFnE xV)) i
          =
        Activation.Math.tanhDerivSpec (xV i) := by
      simp
    have hR :
        (elemwiseDerivCLM (n := n) Activation.Math.tanhDerivSpec xV) dxV i =
          dxV i * Activation.Math.tanhDerivSpec (xV i) := by
      rfl
    rw [hMap] at hL
    exact hL.trans hR.symm
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
    have hfun :
        (fun xV : Vec n =>
            getScalarE ((sigmoidCorrect (s := .dim n .scalar)).op.forward (ofFnE xV))) =
          elemwiseVec (n := n) (fun z => Activation.Math.sigmoidSpec z) := by
      funext xV
      ext i
      simp [sigmoidCorrect, Spec.sigmoidOp, elemwiseVec]
    rw [hfun]
    exact h
  jvp_eq := by
    intro xV dxV
    ext i
    have hL :
        getScalarE ((sigmoidCorrect (s := .dim n .scalar)).jvp (ofFnE xV) (ofFnE dxV)) i
          =
        dxV i *
          TorchLean.Tensor.getScalar
            (Tensor.map Activation.Math.sigmoidDerivSpec (ofFnE xV)) i := by
      simp [sigmoidCorrect, Spec.sigmoidOp,
        Activation.sigmoidDerivSpec, getScalarE, ofFnE,
        TorchLean.Tensor.getScalar_ofFn, Spec.getScalar_mul_spec]
    have hMap :
        TorchLean.Tensor.getScalar
            (Tensor.map (shape := .dim n .scalar) Activation.Math.sigmoidDerivSpec (ofFnE xV)) i
          =
        Activation.Math.sigmoidDerivSpec (xV i) := by
      simp
    have hR :
        (elemwiseDerivCLM (n := n) Activation.Math.sigmoidDerivSpec xV) dxV i =
          dxV i * Activation.Math.sigmoidDerivSpec (xV i) := by
      rfl
    rw [hMap] at hL
    exact hL.trans hR.symm
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
    have hfun :
        (fun xV : Vec n =>
            getScalarE ((softplusCorrect (s := .dim n .scalar)).op.forward (ofFnE xV))) =
          elemwiseVec (n := n) (fun z => Activation.Math.softplusSpec z) := by
      funext xV
      ext i
      simp [softplusCorrect, Spec.softplusOp, elemwiseVec]
    rw [hfun]
    exact h
  jvp_eq := by
    intro xV dxV
    ext i
    have hL :
        getScalarE ((softplusCorrect (s := .dim n .scalar)).jvp (ofFnE xV) (ofFnE dxV)) i
          =
        dxV i *
          TorchLean.Tensor.getScalar
            (Tensor.map Activation.Math.softplusDerivSpec (ofFnE xV)) i := by
      simp [softplusCorrect, Spec.softplusOp,
        Activation.softplusDerivSpec, getScalarE, ofFnE,
        TorchLean.Tensor.getScalar_ofFn, Spec.getScalar_mul_spec]
    have hMap :
        TorchLean.Tensor.getScalar
            (Tensor.map (shape := .dim n .scalar) Activation.Math.softplusDerivSpec (ofFnE xV)) i
          =
        Activation.Math.softplusDerivSpec (xV i) := by
      simp
    have hR :
        (elemwiseDerivCLM (n := n) Activation.Math.softplusDerivSpec xV) dxV i =
          dxV i * Activation.Math.softplusDerivSpec (xV i) := by
      rfl
    rw [hMap] at hL
    exact hL.trans hR.symm
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
    have hfun :
        (fun xV : Vec n =>
            getScalarE ((siluCorrect (s := .dim n .scalar)).op.forward (ofFnE xV))) =
          elemwiseVec (n := n) (fun z => Activation.Math.swishSpec z) := by
      funext xV
      ext i
      simp [siluCorrect, Spec.siluOp, Activation.swishSpec, elemwiseVec]
    rw [hfun]
    exact h
  jvp_eq := by
    intro xV dxV
    ext i
    have hL :
        getScalarE ((siluCorrect (s := .dim n .scalar)).jvp (ofFnE xV) (ofFnE dxV)) i
          =
        dxV i *
          TorchLean.Tensor.getScalar
            (Tensor.map (shape := .dim n .scalar) Activation.Math.swishDerivSpec (ofFnE xV)) i := by
      simp [siluCorrect, Spec.siluOp,
        Activation.swishDerivSpec, getScalarE, ofFnE,
        TorchLean.Tensor.getScalar_ofFn, Spec.getScalar_mul_spec]
    have hMap :
        TorchLean.Tensor.getScalar
            (Tensor.map (shape := .dim n .scalar) Activation.Math.swishDerivSpec (ofFnE xV)) i
          =
        Activation.Math.swishDerivSpec (xV i) := by
      simp
    have hR :
        (elemwiseDerivCLM (n := n) Activation.Math.swishDerivSpec xV) dxV i =
          dxV i * Activation.Math.swishDerivSpec (xV i) := by
      rfl
    rw [hMap] at hL
    exact hL.trans hR.symm
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
    have hfun :
        (fun xV : Vec n =>
            getScalarE ((geluCorrect (s := .dim n .scalar)).op.forward (ofFnE xV))) =
          elemwiseVec (n := n) (fun z => Activation.Math.geluSpec z) := by
      funext xV
      ext i
      simp [geluCorrect, Spec.geluOp, Activation.geluSpec, elemwiseVec]
    rw [hfun]
    exact h
  jvp_eq := by
    intro xV dxV
    ext i
    have hL :
        getScalarE ((geluCorrect (s := .dim n .scalar)).jvp (ofFnE xV) (ofFnE dxV)) i
          =
        dxV i *
          TorchLean.Tensor.getScalar
            (Tensor.map (shape := .dim n .scalar) Activation.Math.geluDerivSpec (ofFnE xV)) i := by
      simp [geluCorrect, Spec.geluOp,
        Activation.geluDerivSpec, getScalarE, ofFnE,
        TorchLean.Tensor.getScalar_ofFn, Spec.getScalar_mul_spec]
    have hMap :
        TorchLean.Tensor.getScalar
            (Tensor.map (shape := .dim n .scalar) Activation.Math.geluDerivSpec (ofFnE xV)) i
          =
        Activation.Math.geluDerivSpec (xV i) := by
      simp
    have hR :
        (elemwiseDerivCLM (n := n) Activation.Math.geluDerivSpec xV) dxV i =
          dxV i * Activation.Math.geluDerivSpec (xV i) := by
      rfl
    rw [hMap] at hL
    exact hL.trans hR.symm
}

/--
`safeLog` as an `OpSpecFDerivCorrect` instance (elementwise), assuming `ε > 0`.

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
    have hfun :
        (fun xV : Vec n =>
            getScalarE ((safeLogCorrect (s := .dim n .scalar) ε).op.forward (ofFnE xV))) =
          elemwiseVec (n := n) (fun z => Activation.Math.safeLogSpec z ε) := by
      funext xV
      ext i
      simp [safeLogCorrect, Spec.safeLogOp, elemwiseVec]
    rw [hfun]
    exact h
  jvp_eq := by
    intro xV dxV
    ext i
    have hL :
        getScalarE ((safeLogCorrect (s := .dim n .scalar) ε).jvp (ofFnE xV) (ofFnE dxV)) i
          =
        dxV i *
          TorchLean.Tensor.getScalar
            (Tensor.map (shape := .dim n .scalar) (fun x => Activation.Math.safeLogDerivSpec x ε)
              (ofFnE xV)) i := by
      simp [safeLogCorrect, Spec.safeLogOp,
        Activation.safeLogDerivSpec, getScalarE, ofFnE,
        TorchLean.Tensor.getScalar_ofFn, Spec.getScalar_mul_spec]
    have hMap :
        TorchLean.Tensor.getScalar
            (Tensor.map (shape := .dim n .scalar) (fun x => Activation.Math.safeLogDerivSpec x ε)
              (ofFnE xV)) i
          =
        Activation.Math.safeLogDerivSpec (xV i) ε := by
      simp
    have hR :
        (elemwiseDerivCLM (n := n) (fun x => Activation.Math.safeLogDerivSpec x ε) xV) dxV i =
          dxV i * Activation.Math.safeLogDerivSpec (xV i) ε := by
      rfl
    rw [hMap] at hL
    exact hL.trans hR.symm
}

/--
`smoothAbs` as an `OpSpecFDerivCorrect` instance (elementwise), assuming `ε > 0`.

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
    have hfun :
        (fun xV : Vec n =>
            getScalarE ((smoothAbsCorrect (s := .dim n .scalar) ε).op.forward (ofFnE xV))) =
          elemwiseVec (n := n) (fun z => Activation.Math.smoothAbsSpec z ε) := by
      funext xV
      ext i
      simp [smoothAbsCorrect, Spec.smoothAbsOp, elemwiseVec]
    rw [hfun]
    exact h
  jvp_eq := by
    intro xV dxV
    ext i
    have hL :
        getScalarE ((smoothAbsCorrect (s := .dim n .scalar) ε).jvp (ofFnE xV) (ofFnE dxV)) i
          =
        dxV i *
          TorchLean.Tensor.getScalar
            (Tensor.map (shape := .dim n .scalar) (fun x => Activation.Math.smoothAbsDerivSpec x ε)
              (ofFnE xV)) i := by
      simp [smoothAbsCorrect, Spec.smoothAbsOp,
        Activation.smoothAbsDerivSpec, getScalarE, ofFnE,
        TorchLean.Tensor.getScalar_ofFn, Spec.getScalar_mul_spec]
    have hMap :
        TorchLean.Tensor.getScalar
            (Tensor.map (shape := .dim n .scalar) (fun x => Activation.Math.smoothAbsDerivSpec x ε)
              (ofFnE xV)) i
          =
        Activation.Math.smoothAbsDerivSpec (xV i) ε := by
      simp
    have hR :
        (elemwiseDerivCLM (n := n) (fun x => Activation.Math.smoothAbsDerivSpec x ε) xV) dxV i =
          dxV i * Activation.Math.smoothAbsDerivSpec (xV i) ε := by
      rfl
    rw [hMap] at hL
    exact hL.trans hR.symm
}

end OpSpecFDerivCorrect

end
end Autograd
end Proofs
