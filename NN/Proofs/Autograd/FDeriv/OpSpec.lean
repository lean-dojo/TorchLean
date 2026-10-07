/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.FDeriv.Core


/-!
# OpSpec

Generic analytic (`HasFDerivAt`/`fderiv`) soundness for **composed** `Spec.OpSpec`s.

`NN.Proofs.Autograd.FDeriv.Core` proves the first end-to-end instance (a 2-layer MLP) by:
1. proving `OpSpecCorrect` (dot/JVP/VJP adjointness), and
2. identifying the JVP with the Fréchet derivative.

This file packages (2) as an extra field and shows it is **closed under `OpSpecCorrect.compose`**.

Result: once primitive ops have analytic JVP facts, any sequential `OpSpec` graph built by
composition gets the theorem:

`backward x δ = VJP[forward, x] δ` (after converting tensors ↔ Euclidean vectors).
-/

@[expose] public section


namespace Proofs
namespace Autograd

open Spec _root_.TorchLean
open _root_.TorchLean _root_.TorchLean.Tensor

open scoped BigOperators
open scoped _root_.Autograd

noncomputable section

/--
A proved `OpSpec` (`OpSpecCorrect`) together with the analytic fact that its JVP is `fderiv`.

This is the “bridge object” that upgrades dot-level correctness (JVP/VJP adjointness) into an
actual `HasFDerivAt` statement about the forward function on `Vec n`.

PyTorch analogy: this corresponds to saying “the local backward rule is the transpose Jacobian of
the true derivative” for a primitive op, so that composing ops yields correct global backward.
-/
structure OpSpecFDerivCorrect (inDim outDim : Nat) where
  /-- The op together with its JVP and the VJP/JVP adjointness proof. -/
  correct : OpSpecCorrect (.dim inDim .scalar) (.dim outDim .scalar)
  /-- The Frechet derivative of the forward map at each point, as a continuous linear map. -/
  deriv : Vec inDim → Vec inDim →L[ℝ] Vec outDim
  /-- The forward map on `Vec inDim` is differentiable everywhere with derivative `deriv`. -/
  hasFDerivAt :
      ∀ xV : Vec inDim,
        HasFDerivAt
          (fun xV : Vec inDim => getScalarE (correct.op.forward (ofFnE xV)))
          (deriv xV) xV
  /-- The JVP of the op agrees with `deriv` applied to the tangent vector. -/
  jvp_eq :
      ∀ xV dxV : Vec inDim,
        getScalarE (correct.jvp (ofFnE xV) (ofFnE dxV)) = (deriv xV) dxV

namespace OpSpecFDerivCorrect

/-- The induced forward function on Euclidean vectors. -/
def forwardVec {inDim outDim : Nat} (C : OpSpecFDerivCorrect inDim outDim) : Vec inDim → Vec outDim
  :=
  fun xV => getScalarE (C.correct.op.forward (ofFnE xV))

/--
Main analytic soundness statement for a single `OpSpecFDerivCorrect`:

`backward x δ` is the adjoint of the Fréchet derivative of the forward map, applied to `δ`.

This is the analytic justification for reverse-mode: it says the implemented VJP is the true
Jacobian-transpose product.
-/
theorem backward_eq_adjoint_fderiv {inDim outDim : Nat} (C : OpSpecFDerivCorrect inDim outDim)
    (x : Tensor ℝ [inDim]) (δ : Tensor ℝ [outDim]) :
    getScalarE (C.correct.op.backward x δ) =
      VJP[C.forwardVec, getScalarE x] (getScalarE δ) := by
  change getScalarE (C.correct.op.backward x δ) =
    (fderiv ℝ C.forwardVec (getScalarE x)).adjoint (getScalarE δ)
  have hf : HasFDerivAt C.forwardVec (C.deriv (getScalarE x)) (getScalarE x) :=
    C.hasFDerivAt (getScalarE x)
  rw [hf.fderiv]
  apply ext_inner_left ℝ
  intro dxV
  rw [ContinuousLinearMap.adjoint_inner_right]
  have hdot := C.correct.correct (x := x) (dx := ofFnE dxV) (δ := δ)
  have hjvp :
      getScalarE (C.correct.jvp x (ofFnE dxV)) = C.deriv (getScalarE x) dxV := by
    simpa only [ofFnE_getScalarE] using C.jvp_eq (getScalarE x) dxV
  simpa only [dot_eq_inner_vec, hjvp, getScalarE_ofFnE] using hdot.symm

/--
Composition preserves analytic correctness (chain rule).

If `f` and `g` each have a correct `fderiv` identification of their JVP, then `g ∘ f` does too.
This is the key closure property used to scale from primitive ops to sequential models.
-/
def compose {inDim midDim outDim : Nat}
    (f : OpSpecFDerivCorrect inDim midDim) (g : OpSpecFDerivCorrect midDim outDim) :
    OpSpecFDerivCorrect inDim outDim :=
{
  correct := OpSpecCorrect.compose f.correct g.correct
  deriv := fun xV => (g.deriv (f.forwardVec xV)).comp (f.deriv xV)
  hasFDerivAt := by
    intro xV
    -- Use the chain rule in Euclidean space and then rewrite the forward function.
    have hf : HasFDerivAt (f.forwardVec) (f.deriv xV) xV := by
      change HasFDerivAt
        (fun xV : Vec inDim => getScalarE (f.correct.op.forward (ofFnE xV))) (f.deriv xV) xV
      exact f.hasFDerivAt xV
    have hg : HasFDerivAt (g.forwardVec) (g.deriv (f.forwardVec xV)) (f.forwardVec xV) := by
      change HasFDerivAt
        (fun xV : Vec midDim => getScalarE (g.correct.op.forward (ofFnE xV)))
        (g.deriv (f.forwardVec xV)) (f.forwardVec xV)
      exact g.hasFDerivAt (f.forwardVec xV)
    have hcomp : HasFDerivAt (fun xV => g.forwardVec (f.forwardVec xV))
        ((g.deriv (f.forwardVec xV)).comp (f.deriv xV)) xV := hg.comp xV hf
    -- The composed `OpSpecCorrect` forward is definitionally `g ∘ f` up to `ofFnE/getScalarE`
    -- roundtrips.
    simpa [OpSpecFDerivCorrect.forwardVec, OpSpecCorrect.compose, Spec.OpSpec.compose] using hcomp
  jvp_eq := by
    intro xV dxV
    -- Expand the composed JVP and rewrite inputs/outputs through `ofFnE/getScalarE`.
    have h_fwd : f.correct.op.forward (ofFnE xV) = ofFnE (f.forwardVec xV) := by
      -- `ofFnE (getScalarE t) = t`.
      simp [OpSpecFDerivCorrect.forwardVec]
    have h_jvp :
        f.correct.jvp (ofFnE xV) (ofFnE dxV) = ofFnE ((f.deriv xV) dxV) := by
      -- Apply `ofFnE` to the analytic JVP equality.
      have hv := congrArg (ofFnE (n := midDim)) (f.jvp_eq xV dxV)
      -- Peel off the `ofFnE ∘ getScalarE` roundtrip on the left explicitly.
      calc
        f.correct.jvp (ofFnE xV) (ofFnE dxV)
            = ofFnE (n := midDim) (getScalarE (f.correct.jvp (ofFnE xV) (ofFnE dxV))) := by
                simp
        _ = ofFnE (n := midDim) ((f.deriv xV) dxV) := hv
    -- Now use `g.jvp_eq` at the intermediate point.
    -- (The cast `h_fwd`/`h_jvp` makes the arguments match `ofFnE` form.)
    -- Expand the composed JVP and rewrite to the `ofFnE` form expected by `g.jvp_eq`.
    calc
      getScalarE ((OpSpecCorrect.compose f.correct g.correct).jvp (ofFnE xV) (ofFnE dxV))
          = getScalarE (g.correct.jvp (f.correct.op.forward (ofFnE xV))
              (f.correct.jvp (ofFnE xV) (ofFnE dxV))) := by
                rfl
      _ = getScalarE (g.correct.jvp (ofFnE (f.forwardVec xV)) (ofFnE ((f.deriv xV) dxV))) := by
            simp [h_fwd, h_jvp]
      _ = (g.deriv (f.forwardVec xV)) ((f.deriv xV) dxV) := by
            simpa using (g.jvp_eq (f.forwardVec xV) ((f.deriv xV) dxV))
      _ = ((g.deriv (f.forwardVec xV)).comp (f.deriv xV)) dxV := by
            simp [ContinuousLinearMap.comp_apply]
}

end OpSpecFDerivCorrect

end
end Autograd
end Proofs
