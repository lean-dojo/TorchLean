/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.FDeriv.Core

public import Mathlib.Analysis.InnerProductSpace.Calculus

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

open Spec
open Tensor

open scoped BigOperators
open scoped _root_.Autograd

noncomputable section

/-!
## Basic tensor/vector roundtrip

Most analytic statements here are written in Euclidean space (`Vec n`) because Mathlib’s `fderiv`
and adjoint API lives there. The following lemma just re-exports the `ofVecE/toVecE` roundtrip in a
form that is convenient for rewriting.
-/

@[simp] lemma ofVecE_toVec {n : Nat} (t : Tensor ℝ (.dim n .scalar)) :
    ofVecE (n := n) (toVecE t) = t := by
  simp

/--
A proved `OpSpec` (`OpSpecCorrect`) together with the analytic fact that its JVP is `fderiv`.

This is the “bridge object” that upgrades dot-level correctness (JVP/VJP adjointness) into an
actual `HasFDerivAt` statement about the forward function on `Vec n`.

PyTorch analogy: this corresponds to saying “the local backward rule is the transpose Jacobian of
the true derivative” for a primitive op, so that composing ops yields correct global backward.
-/
structure OpSpecFDerivCorrect (inDim outDim : Nat) where
  /-- correct. -/
  correct : OpSpecCorrect (.dim inDim .scalar) (.dim outDim .scalar)
  /-- deriv. -/
  deriv : Vec inDim → Vec inDim →L[ℝ] Vec outDim
  /-- has FDeriv At. -/
  hasFDerivAt :
      ∀ xV : Vec inDim,
        HasFDerivAt
          (fun xV : Vec inDim => toVecE (correct.op.forward (ofVecE xV)))
          (deriv xV) xV
  /-- jvp eq. -/
  jvp_eq :
      ∀ xV dxV : Vec inDim,
        toVecE (correct.jvp (ofVecE xV) (ofVecE dxV)) = (deriv xV) dxV

namespace OpSpecFDerivCorrect

/-- The induced forward function on Euclidean vectors. -/
def forwardVec {inDim outDim : Nat} (C : OpSpecFDerivCorrect inDim outDim) : Vec inDim → Vec outDim
  :=
  fun xV => toVecE (C.correct.op.forward (ofVecE xV))

/--
Main analytic soundness statement for a single `OpSpecFDerivCorrect`:

`backward x δ` is the adjoint of the Fréchet derivative of the forward map, applied to `δ`.

This is the analytic justification for reverse-mode: it says the implemented VJP is the true
Jacobian-transpose product.
-/
theorem backward_eq_adjoint_fderiv {inDim outDim : Nat} (C : OpSpecFDerivCorrect inDim outDim)
    (x : Tensor ℝ (.dim inDim .scalar)) (δ : Tensor ℝ (.dim outDim .scalar)) :
    toVecE (C.correct.op.backward x δ) =
      VJP[C.forwardVec, toVecE x] (toVecE δ) := by
  classical
  -- Reduce to the `x = ofVecE xV` case.
  let xV : Vec inDim := toVecE x
  have hx : x = ofVecE (n := inDim) xV := by
    simp [xV]
  -- Prove the statement at `xV` and then rewrite.
  have h_ofVec :
      toVecE (C.correct.op.backward (ofVecE xV) δ) =
        VJP[C.forwardVec, xV] (toVecE δ) := by
    -- Use the dot-level correctness to characterize the backward cotangent via inner products.
    have hf : HasFDerivAt (C.forwardVec) (C.deriv xV) xV := by
      simpa [OpSpecFDerivCorrect.forwardVec] using (C.hasFDerivAt xV)
    have hfderiv : fderiv ℝ (C.forwardVec) xV = C.deriv xV := by
      simpa using hf.fderiv

    have hinner :
        ∀ dxV : Vec inDim,
          inner ℝ ((C.deriv xV) dxV) (toVecE δ) =
            inner ℝ dxV (toVecE (C.correct.op.backward (ofVecE xV) δ)) := by
      intro dxV
      have hdot := C.correct.correct (x := ofVecE xV) (dx := ofVecE dxV) (δ := δ)
      -- Convert `dot` to `inner` and rewrite the JVP via `jvp_eq`.
      have hinner' :
          inner ℝ (toVecE (C.correct.jvp (ofVecE xV) (ofVecE dxV))) (toVecE δ) =
            inner ℝ (toVecE (ofVecE dxV)) (toVecE (C.correct.op.backward (ofVecE xV) δ)) := by
        simpa [dot_eq_inner_vec] using hdot
      -- Replace the JVP with the analytic derivative and simplify `toVecE (ofVecE dxV)`.
      have hinner'' := hinner'
      -- Rewrite the JVP term using the analytic identification.
      rw [C.jvp_eq xV dxV] at hinner''
      simpa using hinner''

    -- Identify the unique element satisfying the adjointness law.
    let A : Vec inDim →L[ℝ] Vec outDim := C.deriv xV
    let u : Vec inDim := toVecE (C.correct.op.backward (ofVecE xV) δ)
    let v : Vec inDim := A.adjoint (toVecE δ)
    have hforall : ∀ dxV : Vec inDim, inner ℝ dxV u = inner ℝ dxV v := by
      intro dxV
      -- Both sides equal `⟪A dxV, δ⟫`.
      calc
        inner ℝ dxV u
            = inner ℝ ((C.deriv xV) dxV) (toVecE δ) := by
                simpa [u] using (hinner (dxV := dxV)).symm
        _ = inner ℝ dxV (A.adjoint (toVecE δ)) := by
              simpa [A] using
                (ContinuousLinearMap.adjoint_inner_right (A := A) (x := dxV) (y := toVecE δ)).symm
        _ = inner ℝ dxV v := by simp [v]

    have h0 : inner ℝ (u - v) (u - v) = 0 := by
      have hEq := hforall (dxV := (u - v))
      have : inner ℝ (u - v) u - inner ℝ (u - v) v = 0 := by
        simpa [sub_eq_zero] using congrArg (fun t => t - inner ℝ (u - v) v) hEq
      have hinnerSub :
          inner ℝ (u - v) (u - v) = inner ℝ (u - v) u - inner ℝ (u - v) v := by
        simpa using (inner_sub_right (x := u - v) (y := u) (z := v))
      exact hinnerSub.trans this
    have huv : u - v = 0 := (inner_self_eq_zero (𝕜 := ℝ) (x := (u - v))).1 h0
    have huv' : u = v := sub_eq_zero.mp huv

    -- Rewrite `v` using `fderiv` and finish.
    calc
      toVecE (C.correct.op.backward (ofVecE xV) δ) = v := by simpa [u] using huv'
      _ = (fderiv ℝ (C.forwardVec) xV).adjoint (toVecE δ) := by
            simp [v, A, hfderiv]

  -- Rewrite `x` to `ofVecE xV` everywhere.
  rw [hx]
  -- `toVecE (ofVecE xV) = xV`.
  simpa using h_ofVec

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
      simpa [OpSpecFDerivCorrect.forwardVec] using (f.hasFDerivAt xV)
    have hg : HasFDerivAt (g.forwardVec) (g.deriv (f.forwardVec xV)) (f.forwardVec xV) := by
      simpa [OpSpecFDerivCorrect.forwardVec] using (g.hasFDerivAt (f.forwardVec xV))
    have hcomp : HasFDerivAt (fun xV => g.forwardVec (f.forwardVec xV))
        ((g.deriv (f.forwardVec xV)).comp (f.deriv xV)) xV := hg.comp xV hf
    -- The composed `OpSpecCorrect` forward is definitionally `g ∘ f` up to `ofVecE/toVecE`
    -- roundtrips.
    simpa [OpSpecFDerivCorrect.forwardVec, OpSpecCorrect.compose, Spec.OpSpec.compose] using hcomp
  jvp_eq := by
    intro xV dxV
    -- Expand the composed JVP and rewrite inputs/outputs through `ofVecE/toVecE`.
    have h_fwd : f.correct.op.forward (ofVecE xV) = ofVecE (f.forwardVec xV) := by
      -- `ofVecE (toVecE t) = t`.
      simp [OpSpecFDerivCorrect.forwardVec]
    have h_jvp :
        f.correct.jvp (ofVecE xV) (ofVecE dxV) = ofVecE ((f.deriv xV) dxV) := by
      -- Apply `ofVecE` to the analytic JVP equality.
      have hv := congrArg (ofVecE (n := midDim)) (f.jvp_eq xV dxV)
      -- Peel off the `ofVecE ∘ toVecE` roundtrip on the left explicitly.
      calc
        f.correct.jvp (ofVecE xV) (ofVecE dxV)
            = ofVecE (n := midDim) (toVecE (f.correct.jvp (ofVecE xV) (ofVecE dxV))) := by
                simp
        _ = ofVecE (n := midDim) ((f.deriv xV) dxV) := hv
    -- Now use `g.jvp_eq` at the intermediate point.
    -- (The cast `h_fwd`/`h_jvp` makes the arguments match `ofVecE` form.)
    -- Expand the composed JVP and rewrite to the `ofVecE` form expected by `g.jvp_eq`.
    calc
      toVecE ((OpSpecCorrect.compose f.correct g.correct).jvp (ofVecE xV) (ofVecE dxV))
          = toVecE (g.correct.jvp (f.correct.op.forward (ofVecE xV))
              (f.correct.jvp (ofVecE xV) (ofVecE dxV))) := by
                rfl
      _ = toVecE (g.correct.jvp (ofVecE (f.forwardVec xV)) (ofVecE ((f.deriv xV) dxV))) := by
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
