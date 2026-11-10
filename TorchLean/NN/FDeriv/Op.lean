/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.FDeriv


/-!
# Analytic soundness of composed VJPOps

Analytic (`HasFDerivAt`/`fderiv`) soundness for composed `VJPOp`s. The first end-to-end
instance, the two-layer MLP in `TorchLean.NN.FDeriv`, takes two steps: it
proves `AdjointOp` (JVP/VJP adjointness for the tensor dot product) and then identifies the JVP
with the Fréchet derivative. `FDerivOp` adds the second step as extra fields, and
`FDerivOp.compose` proves those fields for `AdjointOp.compose`.

Once every primitive op carries its analytic JVP fact, a sequential `VJPOp` chain built by
composition satisfies `backward x δ = (fderiv ℝ forward x).adjoint δ`, after converting between
tensors and Euclidean vectors.
-/

@[expose] public section


namespace TorchLean

open _root_.TorchLean.Tensor

open scoped BigOperators

noncomputable section

/--
A proved `VJPOp` (`AdjointOp`) together with the analytic fact that its JVP is the Fréchet
derivative of its forward map on `Vec inDim`. Dot-level adjointness alone relates two formulas to
each other, and the extra fields tie them to a `HasFDerivAt` statement.

PyTorch analogy: the local backward rule of a primitive op is the transposed Jacobian of the true
derivative, so composing ops gives the correct global backward.
-/
structure FDerivOp (inDim outDim : Nat) where
  /-- The op together with its JVP and the VJP/JVP adjointness proof. -/
  toAdjointOp : Tape.AdjointOp ℝ (.dim inDim .scalar) (.dim outDim .scalar)
  /-- Fréchet derivative of the forward map at each point, as a continuous linear map. -/
  deriv : Vec inDim → Vec inDim →L[ℝ] Vec outDim
  /-- The forward map on `Vec inDim` is differentiable everywhere with derivative `deriv`. -/
  hasFDerivAt :
      ∀ xV : Vec inDim,
        HasFDerivAt
          (fun xV : Vec inDim => getScalarE (toAdjointOp.op.forward (ofFnE xV)))
          (deriv xV) xV
  /-- The JVP of the op agrees with `deriv` applied to the tangent vector. -/
  jvp_eq :
      ∀ xV dxV : Vec inDim,
        getScalarE (toAdjointOp.jvp (ofFnE xV) (ofFnE dxV)) = (deriv xV) dxV

namespace FDerivOp

/-- Forward map read on Euclidean vectors: `xV ↦ Tensor.getScalarE (forward (Tensor.ofFnE xV))`. -/
def forwardVec {inDim outDim : Nat} (C : FDerivOp inDim outDim) : Vec inDim → Vec outDim
  :=
  fun xV => getScalarE (C.toAdjointOp.op.forward (ofFnE xV))

/--
For an `FDerivOp` op, `backward x δ` is the adjoint of the Fréchet derivative of the
forward map at `x`, applied to `δ`. The implemented VJP is therefore the true Jacobian-transpose
product, which is what reverse mode needs.
-/
theorem backward_eq_fderiv {inDim outDim : Nat} (C : FDerivOp inDim outDim)
    (x : Tensor ℝ [inDim]) (δ : Tensor ℝ [outDim]) :
    getScalarE (C.toAdjointOp.op.backward x δ) =
      (fderiv ℝ C.forwardVec (getScalarE x)).adjoint (getScalarE δ) := by
  have hf : HasFDerivAt C.forwardVec (C.deriv (getScalarE x)) (getScalarE x) :=
    C.hasFDerivAt (getScalarE x)
  rw [hf.fderiv]
  apply ext_inner_left ℝ
  intro dxV
  rw [ContinuousLinearMap.adjoint_inner_right]
  have hdot := C.toAdjointOp.isAdjointPair (x := x) (dx := ofFnE dxV) (δ := δ)
  have hjvp :
      getScalarE (C.toAdjointOp.jvp x (ofFnE dxV)) = C.deriv (getScalarE x) dxV := by
    simpa only [ofFnE_getScalarE] using C.jvp_eq (getScalarE x) dxV
  simpa only [dot_eq_inner_getScalarE, hjvp, getScalarE_ofFnE] using hdot.symm

/--
Composition preserves analytic correctness by the chain rule: if the JVPs of `f` and `g` are
their Fréchet derivatives, so is the JVP of `g ∘ f`, with derivative `g.deriv (f x) ∘ f.deriv x`.
Closure under composition takes the result from primitive ops to sequential models.
-/
def compose {inDim midDim outDim : Nat}
    (f : FDerivOp inDim midDim) (g : FDerivOp midDim outDim) :
    FDerivOp inDim outDim :=
{
  toAdjointOp := Tape.AdjointOp.compose f.toAdjointOp g.toAdjointOp
  deriv := fun xV => (g.deriv (f.forwardVec xV)).comp (f.deriv xV)
  hasFDerivAt := by
    intro xV
    -- Use the chain rule in Euclidean space and then rewrite the forward function.
    have hf : HasFDerivAt (f.forwardVec) (f.deriv xV) xV := by
      change HasFDerivAt
        (fun xV : Vec inDim => getScalarE (f.toAdjointOp.op.forward (ofFnE xV))) (f.deriv xV) xV
      exact f.hasFDerivAt xV
    have hg : HasFDerivAt (g.forwardVec) (g.deriv (f.forwardVec xV)) (f.forwardVec xV) := by
      change HasFDerivAt
        (fun xV : Vec midDim => getScalarE (g.toAdjointOp.op.forward (ofFnE xV)))
        (g.deriv (f.forwardVec xV)) (f.forwardVec xV)
      exact g.hasFDerivAt (f.forwardVec xV)
    have hcomp : HasFDerivAt (fun xV => g.forwardVec (f.forwardVec xV))
        ((g.deriv (f.forwardVec xV)).comp (f.deriv xV)) xV := hg.comp xV hf
    -- The composed certificate's forward map is definitionally `g ∘ f` up to
    -- `ofFnE/getScalarE` roundtrips.
    simpa [FDerivOp.forwardVec, Tape.AdjointOp.compose, VJPOp.compose] using
      hcomp
  jvp_eq := by
    intro xV dxV
    -- First put the forward value and the JVP of `f` into `ofFnE` form.
    have h_fwd : f.toAdjointOp.op.forward (ofFnE xV) = ofFnE (f.forwardVec xV) := by
      -- `ofFnE (getScalarE t) = t`.
      simp [FDerivOp.forwardVec]
    have h_jvp :
        f.toAdjointOp.jvp (ofFnE xV) (ofFnE dxV) = ofFnE ((f.deriv xV) dxV) := by
      have hv := congrArg (ofFnE (n := midDim)) (f.jvp_eq xV dxV)
      -- Insert the `ofFnE ∘ getScalarE` roundtrip on the left so that `hv` applies.
      calc
        f.toAdjointOp.jvp (ofFnE xV) (ofFnE dxV)
            = ofFnE (n := midDim) (getScalarE (f.toAdjointOp.jvp (ofFnE xV) (ofFnE dxV))) := by
                simp
        _ = ofFnE (n := midDim) ((f.deriv xV) dxV) := hv
    -- Unfold the composed JVP, rewrite with `h_fwd` and `h_jvp` into the `ofFnE` form that
    -- `g.jvp_eq` expects, and apply it at the intermediate point.
    calc
      getScalarE ((Tape.AdjointOp.compose f.toAdjointOp g.toAdjointOp).jvp (ofFnE xV) (ofFnE dxV))
          = getScalarE (g.toAdjointOp.jvp (f.toAdjointOp.op.forward (ofFnE xV))
              (f.toAdjointOp.jvp (ofFnE xV) (ofFnE dxV))) := by
                rfl
      _ = getScalarE (g.toAdjointOp.jvp (ofFnE (f.forwardVec xV)) (ofFnE ((f.deriv xV) dxV))) := by
            simp [h_fwd, h_jvp]
      _ = (g.deriv (f.forwardVec xV)) ((f.deriv xV) dxV) := by
            simpa using (g.jvp_eq (f.forwardVec xV) ((f.deriv xV) dxV))
      _ = ((g.deriv (f.forwardVec xV)).comp (f.deriv xV)) dxV := by
            simp [ContinuousLinearMap.comp_apply]
}

end FDerivOp

end
end TorchLean
