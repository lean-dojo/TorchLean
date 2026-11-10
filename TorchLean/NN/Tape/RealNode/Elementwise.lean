/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Tape.RealNode.Context
public import TorchLean.NN.FDeriv.Elementwise

/-!
# Elementwise tape nodes

The node `elemwise idx f f'` applies a scalar function `f` to every coordinate of a context entry,
and its JVP and VJP multiply by `f'` coordinatewise. Its certificate `elemwiseFDeriv` needs only
the scalar fact `HasDerivAt f (f' z) z` at every `z`, and `elemwiseFDerivAt` needs it at each
coordinate of one basepoint. `unaryOp` turns a certified vector operation, an `FDerivOp`, into a
node that reads one slot, and `linear` is `unaryOp` applied to the certificate of an affine layer.
We take the named pointwise operators (ReLU, sigmoid, tanh, the square root and the others) from
the operator arms of `Tape.LawfulNode.ofArm` and their `op_cert` certificates.
-/

@[expose] public section

open FloatLib.Numerics (MathFunctions)
open TorchLean.Shape (size size_cons)

namespace TorchLean

open TorchLean.Tensor

noncomputable section

open scoped BigOperators

namespace Tape.RealNode

open CtxVec
open Vec (elemwiseDerivCLM)

/-- `CtxVec.get` specialized to vector shapes. -/
def _root_.TorchLean.CtxVec.getVec {Γ : List Shape} {n : Nat} (idx : Idx Γ (.dim n .scalar))
    (x : CtxVec Γ) : Vec n :=
  LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ
    (finCongr (by simp : Shape.size (Shape.dim n .scalar) = n))
    (CtxVec.get (Γ := Γ) (s := .dim n .scalar) idx x)

/-- `CtxVec.getCLM` specialized to vector shapes `.dim n .scalar`. -/
def _root_.TorchLean.CtxVec.getVecCLM {Γ : List Shape} {n : Nat} (idx : Idx Γ (.dim n .scalar)) :
    CtxVec Γ →L[ℝ] Vec n :=
  (LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ (finCongr (by simp :
      Shape.size (Shape.dim n .scalar) = n))).toContinuousLinearEquiv.toContinuousLinearMap.comp
    (CtxVec.getCLM (Γ := Γ) (s := .dim n .scalar) idx)

/-- The vector-shaped context lookup agrees with `getVec`, size cast included. -/
@[simp] theorem _root_.TorchLean.CtxVec.getVecCLM_apply {Γ : List Shape} {n : Nat}
    (idx : Idx Γ (.dim n .scalar)) (x : CtxVec Γ) :
    getVecCLM (Γ := Γ) (n := n) idx x = getVec (Γ := Γ) (n := n) idx x := by
  -- Here and below, simp leaves out `size_cons`. It would rewrite the sizes inside `Fin` types
  -- before the cast and `ofFn` lemmas match.
  simp [getVecCLM, getVec, CtxVec.getCLM_apply, -Shape.size_cons]

/-- Inject a `Vec n` into a vectorized context at `idx` (fills other blocks with zeros). -/
def _root_.TorchLean.CtxVec.singleVec {Γ : List Shape} {n : Nat} (idx : Idx Γ (.dim n .scalar))
    (v : Vec n) : CtxVec Γ :=
  CtxVec.single (Γ := Γ) (s := .dim n .scalar) idx
    (LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ
      (finCongr (by simp : Shape.size (Shape.dim n .scalar) = n).symm) v)

/-- Reading a vector slot is adjoint to writing it:
`⟪x, CtxVec.singleVec idx v⟫ = ⟪CtxVec.getVec idx x, v⟫`.

A node that reads its input from slot `idx` sends its cotangent back through `CtxVec.singleVec idx`,
and this equation is the adjointness step in the VJP proofs of `unaryOp` and the matrix nodes. -/
@[simp] theorem _root_.TorchLean.CtxVec.inner_getVec_singleVec {Γ : List Shape} {n : Nat}
    (idx : Idx Γ (.dim n .scalar)) (x : CtxVec Γ) (v : Vec n) :
    inner ℝ x (singleVec (Γ := Γ) (n := n) idx v) = inner ℝ (getVec (Γ := Γ) (n := n) idx x) v := by
  rw [singleVec, CtxVec.inner_get_single, getVec, LinearIsometryEquiv.inner_map_eq_flip,
    LinearIsometryEquiv.piLpCongrLeft_symm, finCongr_symm]

-- ---------------------------------------------------------------------------
-- Generic elementwise nodes on flattened tensors
-- ---------------------------------------------------------------------------

/-- `elemwise idx f f'` applies `f` to every coordinate of the context entry at `idx`. Its JVP and
VJP multiply each coordinate of the tangent or cotangent by `f'` at the matching input. -/
def elemwise {Γ : List Shape} {s : Shape} (idx : Idx Γ s) (f f' : ℝ → ℝ) :
    LawfulNode (α := ℝ) Unit Γ s :=
  let n : Nat := Shape.size s
  LawfulNode.ofFn (Γ := Γ) (τ := s)
    (f := fun xV =>
      WithLp.toLp 2 (fun i : Fin n => f (CtxVec.get (Γ := Γ) (s := s) idx xV i)))
    (jvp := fun xV dxV =>
      WithLp.toLp 2 (fun i : Fin n =>
        (CtxVec.get (Γ := Γ) (s := s) idx dxV i) * f' (CtxVec.get (Γ := Γ) (s := s) idx xV i)))
    (vjp := fun xV δ =>
      CtxVec.single (Γ := Γ) (s := s) idx
        (WithLp.toLp 2 (fun i : Fin n => δ i * f' (CtxVec.get (Γ := Γ) (s := s) idx xV i))))
    (correct_inner := by
      intro xV dxV δ
      classical
      -- First use the context adjointness, then expand both inners.
      simp [CtxVec.inner_get_single]
      simp [Vec.inner_eq_sum_mul, mul_assoc, mul_comm])

/-- If `f` has derivative `f' xᵢ` at every coordinate `xᵢ` of the entry at `idx`, the `elemwise`
node has a Fréchet derivative at `xV` and its JVP is that derivative. -/
def elemwiseFDerivAt {Γ : List Shape} {s : Shape} (idx : Idx Γ s) (f f' : ℝ → ℝ) (xV : CtxVec Γ)
    (hf : ∀ i : Fin (Shape.size s),
      HasDerivAt f (f' (CtxVec.get (Γ := Γ) (s := s) idx xV i)) (CtxVec.get (Γ := Γ) (s := s) idx xV
        i)) :
    FDerivCorrectAt (elemwise (Γ := Γ) (s := s) idx f f').toNodeData () xV := by
  classical
  let n : Nat := Shape.size s
  refine
    { deriv :=
        (elemwiseDerivCLM (n := n) f' (CtxVec.get (Γ := Γ) (s := s) idx xV)).comp
          (CtxVec.getCLM (Γ := Γ) (s := s) idx)
      hasFDerivAt := ?_
      jvp_eq := ?_ }
  · have hget :
        HasFDerivAt (fun x : CtxVec Γ => CtxVec.get (Γ := Γ) (s := s) idx x)
          (CtxVec.getCLM (Γ := Γ) (s := s) idx) xV :=
      CtxVec.hasFDerivAt_get idx xV
    have helem :
        HasFDerivAt (Vec.elemwise (n := n) f) (elemwiseDerivCLM (n := n) f' (CtxVec.get (Γ := Γ) (s
          := s) idx xV))
          (CtxVec.get (Γ := Γ) (s := s) idx xV) :=
      Vec.hasFDerivAt_elemwise (n := n) (x := CtxVec.get (Γ := Γ) (s := s) idx xV) (f := f) (f' :=
        f') hf
    have hcomp := helem.comp xV hget
    have hforward :
        (Vec.elemwise (n := n) f ∘ fun x : CtxVec Γ => CtxVec.get (Γ := Γ) (s := s) idx x)
          =
        (fun xV : CtxVec Γ =>
          WithLp.toLp 2 fun i => f ((CtxVec.get (Γ := Γ) (s := s) idx xV).ofLp i)) := by
      funext x
      ext i
      simp [Vec.elemwise]
    rw [hforward] at hcomp
    simpa [elemwise, LawfulNode.forwardVec_ofFn, Vec.elemwise, n,
      ContinuousLinearMap.comp_apply] using
      hcomp
  · intro dxV
    ext i
    simp [elemwise, LawfulNode.jvpVec_ofFn, elemwiseDerivCLM, ContinuousLinearMap.comp_apply, n,
      CtxVec.getCLM_apply]

/-- If `f` has derivative `f' z` at every `z`, the `elemwise` node is `FDerivCorrect`, built
from `elemwiseFDerivAt` at each basepoint. -/
def elemwiseFDeriv {Γ : List Shape} {s : Shape} (idx : Idx Γ s) (f f' : ℝ → ℝ)
    (hf : ∀ z, HasDerivAt f (f' z) z) :
    FDerivCorrect (elemwise (Γ := Γ) (s := s) idx f f').toNodeData () where
  deriv xV :=
    (elemwiseDerivCLM (n := Shape.size s) f' (CtxVec.get (Γ := Γ) (s := s) idx xV)).comp
      (CtxVec.getCLM (Γ := Γ) (s := s) idx)
  hasFDerivAt xV := (elemwiseFDerivAt (Γ := Γ) (s := s) idx f f' xV fun _ => hf _).hasFDerivAt
  jvp_eq xV := (elemwiseFDerivAt (Γ := Γ) (s := s) idx f f' xV fun _ => hf _).jvp_eq

/-- At a positive `x`, the clamped square root `y ↦ sqrt (max y 0)` has derivative
`1 / (2 * sqrt x)`. -/
theorem hasDerivAt_sqrt_max_of_pos {x : ℝ} (hx : 0 < x) :
    HasDerivAt (fun y : ℝ => Real.sqrt (max y 0)) (1 / (2 * Real.sqrt x)) x := by
  -- Near a positive `x` the clamp is inactive, so the function agrees with `Real.sqrt`.
  have heq :
      (fun y : ℝ => Real.sqrt (max y 0)) =ᶠ[nhds x] fun y : ℝ => Real.sqrt y := by
    filter_upwards [Ioi_mem_nhds hx] with y (hy : 0 < y)
    simp [max_eq_left hy.le]
  have hs : HasDerivAt (fun y : ℝ => Real.sqrt y) (1 / (2 * Real.sqrt x)) x := by
    -- `Real.hasDerivAt_sqrt` expects `x ≠ 0`.
    simpa using (Real.hasDerivAt_sqrt (ne_of_gt hx))
  -- transport across the local equality `max y 0 = y` near positive `x`
  exact hs.congr_of_eventuallyEq heq

/-- Tape node that applies a certified vector operation `C` to the slot `idx`, with the JVP and
backward pass of `C` as its own. -/
def unaryOp {Γ : List Shape} {inDim outDim : Nat}
    (idx : Idx Γ (.dim inDim .scalar))
    (C : FDerivOp inDim outDim) : LawfulNode (α := ℝ) Unit Γ (.dim outDim .scalar) :=
  let hOut : Shape.size (Shape.dim outDim .scalar) = outDim := by simp
  LawfulNode.ofFn (Γ := Γ) (τ := .dim outDim .scalar)
    (f := fun ctxV =>
      LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ (finCongr hOut.symm)
        (C.forwardVec ((getVecCLM (Γ := Γ) (n := inDim) idx) ctxV)))
    (jvp := fun ctxV dctxV =>
      LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ (finCongr hOut.symm)
        (getScalarE (C.toAdjointOp.jvp (ofFnE ((getVecCLM (Γ := Γ) (n := inDim) idx) ctxV))
          (ofFnE ((getVecCLM (Γ := Γ) (n := inDim) idx) dctxV)))))
    (vjp := fun ctxV δV =>
      let δV' : Vec outDim := LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ (finCongr hOut) δV
      singleVec (Γ := Γ) (n := inDim) idx
        (getScalarE (C.toAdjointOp.op.backward (ofFnE ((getVecCLM (Γ := Γ) (n := inDim) idx) ctxV))
          (ofFnE δV'))))
    (correct_inner := by
      intro ctxV dctxV δV
      rw [LinearIsometryEquiv.inner_map_eq_flip, LinearIsometryEquiv.piLpCongrLeft_symm,
        finCongr_symm, inner_getVec_singleVec]
      have h := C.toAdjointOp.isAdjointPair
        (x := ofFnE (getVec (Γ := Γ) (n := inDim) idx ctxV))
        (dx := ofFnE (getVec (Γ := Γ) (n := inDim) idx dctxV))
        (δ := ofFnE (LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ (finCongr hOut) δV))
      simpa [dot_eq_inner_getScalarE, getScalarE_ofFnE, getVecCLM_apply] using h)

/-- `FDerivCorrect` for `unaryOp`: the derivative at `xV` is `C.deriv` at the slot value,
after the projection `CtxVec.getVecCLM idx`. -/
def unaryOpFDeriv {Γ : List Shape} {inDim outDim : Nat}
    (idx : Idx Γ (.dim inDim .scalar))
    (C : FDerivOp inDim outDim) :
    FDerivCorrect (unaryOp (Γ := Γ) (inDim := inDim) (outDim := outDim) idx C).toNodeData () :=
{ deriv := fun xV =>
    let hOut : Shape.size (Shape.dim outDim .scalar) = outDim := by simp
    (LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ
        (finCongr hOut.symm)).toContinuousLinearEquiv.toContinuousLinearMap.comp
      ((C.deriv ((getVecCLM (Γ := Γ) (n := inDim) idx) xV)).comp (getVecCLM (Γ := Γ) (n := inDim)
        idx))
  hasFDerivAt := by
    intro xV
    let hOut : Shape.size (Shape.dim outDim .scalar) = outDim := by simp
    -- The slot projection is linear, so it is its own derivative.
    have hproj :
        HasFDerivAt (fun xV : CtxVec Γ => (getVecCLM (Γ := Γ) (n := inDim) idx) xV)
          (getVecCLM (Γ := Γ) (n := inDim) idx) xV :=
      (getVecCLM (Γ := Γ) (n := inDim) idx).hasFDerivAt (x := xV)
    have hC :
        HasFDerivAt (fun x : Vec inDim => C.forwardVec x)
          (C.deriv ((getVecCLM (Γ := Γ) (n := inDim) idx) xV))
          ((getVecCLM (Γ := Γ) (n := inDim) idx) xV) := by
      simpa [FDerivOp.forwardVec] using (C.hasFDerivAt ((getVecCLM (Γ := Γ) (n := inDim)
        idx) xV))
    have hcomp : HasFDerivAt (fun xV : CtxVec Γ => C.forwardVec ((getVecCLM (Γ := Γ) (n := inDim)
      idx) xV))
        ((C.deriv ((getVecCLM (Γ := Γ) (n := inDim) idx) xV)).comp (getVecCLM (Γ := Γ) (n := inDim)
          idx)) xV :=
      (hC.comp xV hproj)
    -- The output size cast is linear as well.
    have hcast :
        HasFDerivAt
          (fun y : Vec outDim => LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ (finCongr hOut.symm) y)
          (LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ
            (finCongr hOut.symm)).toContinuousLinearEquiv.toContinuousLinearMap
          (C.forwardVec ((getVecCLM (Γ := Γ) (n := inDim) idx) xV)) :=
      (LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ
        (finCongr hOut.symm)).toContinuousLinearEquiv.toContinuousLinearMap.hasFDerivAt
    have hfinal := hcast.comp xV hcomp
    -- `unaryOp.forwardVec` is definitional to this composition (after unfolding casts).
    have htarget :
        (unaryOp (Γ := Γ) (inDim := inDim) (outDim := outDim) idx C).forwardVec () =
          (fun ctxV : CtxVec Γ =>
            LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ (finCongr hOut.symm)
              (C.forwardVec ((getVecCLM (Γ := Γ) (n := inDim) idx) ctxV))) := by
      funext ctxV
      simp [unaryOp, LawfulNode.forwardVec_ofFn, -Shape.size_cons]
    rw [htarget]
    simpa [Function.comp_def, -Shape.size_cons] using hfinal
  jvp_eq := by
    intro xV dxV
    let hOut : Shape.size (Shape.dim outDim .scalar) = outDim := by simp
    have hjvp :=
      C.jvp_eq
        (xV := (getVecCLM (Γ := Γ) (n := inDim) idx) xV)
        (dxV := (getVecCLM (Γ := Γ) (n := inDim) idx) dxV)
    -- apply the output cast to both sides
    have hjvp' := congrArg (LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ (finCongr hOut.symm)) hjvp
    -- unfold `unaryOp` and simplify
    simpa [unaryOp, getVecCLM_apply, getVec, hOut, ContinuousLinearMap.comp_apply,
      -Shape.size_cons] using hjvp' }

/-- Linear layer as a single tape node (fixed weights/bias in the `Linear`). -/
def linear {Γ : List Shape} {inDim outDim : Nat}
    (x : Idx Γ (.dim inDim .scalar)) (m : Linear ℝ inDim outDim) :
    LawfulNode (α := ℝ) Unit Γ (.dim outDim .scalar) :=
  unaryOp (Γ := Γ) (inDim := inDim) (outDim := outDim) x (FDerivOp.linear m)

/-- `FDerivCorrect` for `linear`: the node derivative is the spec's `VJPOp` derivative. -/
def linearFDeriv {Γ : List Shape} {inDim outDim : Nat}
    (x : Idx Γ (.dim inDim .scalar)) (m : Linear ℝ inDim outDim) :
    FDerivCorrect (linear (Γ := Γ) (inDim := inDim) (outDim := outDim) x m).toNodeData () :=
  unaryOpFDeriv (Γ := Γ) (inDim := inDim) (outDim := outDim) x (FDerivOp.linear m)

end Tape.RealNode

end

end TorchLean
