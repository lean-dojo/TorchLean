/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.FDeriv.Elementwise
public import NN.Proofs.Autograd.FDeriv.LogSoftmax
public import NN.Proofs.Autograd.FDeriv.Softmax
public import NN.Proofs.Autograd.Tape.Core.FDeriv

public import Mathlib.Analysis.Calculus.Deriv.Abs
public import Mathlib.Analysis.Calculus.FDeriv.Add
public import Mathlib.Analysis.Calculus.FDeriv.Bilinear
public import Mathlib.Analysis.Calculus.FDeriv.Comp
public import Mathlib.Analysis.Calculus.FDeriv.Linear
public import Mathlib.Analysis.Calculus.FDeriv.Mul
public import Mathlib.Analysis.InnerProductSpace.Calculus
public import Mathlib.Analysis.SpecialFunctions.Sqrt
public import Mathlib.Data.Fintype.BigOperators

/-!
# Tape-node context primitives

This module contains the low-level vectorized context operations used by the tape-node proof library:
block projections, one-hot cotangent injections, and the bridge from generic `OpSpecFDerivCorrect`
witnesses to `NodeFDerivCorrect` nodes.
-/

@[expose] public section

namespace Proofs
namespace Autograd

open Spec
open Tensor

noncomputable section

open scoped BigOperators

open scoped BigOperators

@[simp] lemma piLpContinuousLinearEquiv2_symm_apply {n : Nat} (f : Fin n → ℝ) (i : Fin n) :
    ((PiLp.continuousLinearEquiv 2 ℝ (fun _ : Fin n => ℝ)).symm f) i = f i := by
  simp

@[simp] lemma piLpContinuousLinearEquiv2_symm_clm_apply {n : Nat} (f : Fin n → ℝ) (i : Fin n) :
    (((PiLp.continuousLinearEquiv 2 ℝ (fun _ : Fin n => ℝ)).symm.toContinuousLinearMap) f) i = f i
      := by
  simp

@[simp] lemma piLpContinuousLinearEquiv2_symm_clm_apply_ofLp {n : Nat} (f : Fin n → ℝ) (i : Fin n) :
    (((PiLp.continuousLinearEquiv 2 ℝ (fun _ : Fin n => ℝ)).symm.toContinuousLinearMap) f).ofLp i =
      f i := by
  simp

@[simp] lemma euclideanEquiv_symm_ofLp {n : Nat} (f : Fin n → ℝ) (i : Fin n) :
    ((EuclideanSpace.equiv (𝕜 := ℝ) (ι := Fin n)).symm f).ofLp i = f i := by
  simp [EuclideanSpace.equiv]

@[simp] lemma inner_scalarVec_left (a : ℝ) (δ : Vec (Shape.size Shape.scalar)) :
    inner ℝ (vecOfFun (n := Shape.size Shape.scalar) fun _ => a) δ = a * δ.ofLp ⟨0, by simp
      [Shape.size]⟩ := by
  calc
    inner ℝ (vecOfFun (n := Shape.size Shape.scalar) fun _ => a) δ
        =
      ∑ i : Fin (Shape.size Shape.scalar), (vecOfFun (n := Shape.size Shape.scalar) (fun _ =>
        a)).ofLp i * δ.ofLp i := by
          simpa using
            (inner_eq_sum_mul
              (x := vecOfFun (n := Shape.size Shape.scalar) (fun _ => a))
              (y := δ))
    _ = a * δ.ofLp ⟨0, by simp [Shape.size]⟩ := by
          simp [Shape.size]

-- We use `inner_append` (proved once in `NN/Proofs/Autograd/Tape/Core/FDeriv.lean`) to split
-- inner products on concatenated Euclidean vectors `appendVec a b`.

-- ---------------------------------------------------------------------------
-- Context slicing on `CtxVec`
-- ---------------------------------------------------------------------------

namespace CtxVec

/--
Raw projection from a vectorized context onto the `i`th block.

This is the underlying block-splitting operation; `CtxVec.get` below wraps it with an `Idx Γ s`
that also remembers the expected shape.
-/
def getRaw : {Γ : List Shape} → (i : Fin Γ.length) → CtxVec Γ → Vec (Shape.size (Γ.get i))
  | [], i, _ => nomatch i
  | s :: ss, ⟨0, _⟩, v => vecOfFun (n := Shape.size s) fun j => v (Fin.castAdd (ctxSize ss) j)
  | s :: ss, ⟨Nat.succ k, hk⟩, v =>
      getRaw (Γ := ss) ⟨k, Nat.lt_of_succ_lt_succ hk⟩
        (vecOfFun (n := ctxSize ss) fun j => v (Fin.natAdd (Shape.size s) j))

/--
Raw injection into a vectorized context: place `v` into block `i`, fill others with zeros.

This is the adjoint of `getRaw` with respect to the Euclidean inner product (proved below).
-/
def singleRaw : {Γ : List Shape} → (i : Fin Γ.length) → Vec (Shape.size (Γ.get i)) → CtxVec Γ
  | [], i, _ => nomatch i
  | s :: ss, ⟨0, _⟩, v =>
      appendVec (m := Shape.size s) (n := ctxSize ss) v (vecOfFun (n := ctxSize ss) fun _ => (0 :
        ℝ))
  | s :: ss, ⟨Nat.succ k, hk⟩, v =>
      appendVec (m := Shape.size s) (n := ctxSize ss)
        (vecOfFun (n := Shape.size s) fun _ => (0 : ℝ))
        (singleRaw (Γ := ss) ⟨k, Nat.lt_of_succ_lt_succ hk⟩ v)

/--
Adjointness of raw projection/injection: `⟪x, singleRaw i v⟫ = ⟪getRaw i x, v⟫`.

This is the vectorized counterpart of the “one-hot cotangent” principle used in tape soundness.
-/
theorem inner_getRaw_singleRaw :
    ∀ {Γ : List Shape} (i : Fin Γ.length) (x : CtxVec Γ) (v : Vec (Shape.size (Γ.get i))),
      inner ℝ x (singleRaw (Γ := Γ) i v) = inner ℝ (getRaw (Γ := Γ) i x) v := by
  intro Γ
  induction Γ with
  | nil =>
      intro i
      exact (nomatch i)
  | cons s ss ih =>
      intro i x v
      classical
      -- decompose `x` as `(head, tail)` and use `inner_append`
      let head : Vec (Shape.size s) := vecOfFun (n := Shape.size s) fun j => x (Fin.castAdd (ctxSize
        ss) j)
      let tail : Vec (ctxSize ss) := vecOfFun (n := ctxSize ss) fun j => x (Fin.natAdd (Shape.size
        s) j)
      have hx : appendVec (m := Shape.size s) (n := ctxSize ss) head tail = x := by
        ext j
        have := congrArg (fun f : Fin (Shape.size s + ctxSize ss) → ℝ => f j)
          (Fin.append_castAdd_natAdd (f := x) (m := Shape.size s) (n := ctxSize ss))
        simpa [appendVec, head, tail] using this
      cases i using Fin.cases with
      | zero =>
          have hinner :=
            inner_append (m := Shape.size s) (n := ctxSize ss) (a := head) (b := tail)
              (c := v) (d := vecOfFun (n := ctxSize ss) fun _ => (0 : ℝ))
          have htail0 : inner ℝ tail (vecOfFun (n := ctxSize ss) fun _ => (0 : ℝ)) = 0 := by
            exact (inner_zero_right (𝕜 := ℝ) (x := tail))
          have h' : inner ℝ x (appendVec (m := Shape.size s) (n := ctxSize ss) v (vecOfFun (n :=
            ctxSize ss) fun _ => (0 : ℝ))) =
              inner ℝ head v := by
            -- rewrite `x` as an append and simplify away the tail/zero inner product
            simpa [hx, htail0, add_assoc] using hinner
          simpa [CtxVec.singleRaw, CtxVec.getRaw, head, hx] using h'
      | succ k =>
          have hinner :=
            inner_append (m := Shape.size s) (n := ctxSize ss)
              (a := head) (b := tail) (c := vecOfFun (n := Shape.size s) fun _ => (0 : ℝ))
              (d := singleRaw (Γ := ss) k v)
          have hhead0 : inner ℝ head (vecOfFun (n := Shape.size s) fun _ => (0 : ℝ)) = 0 := by
            exact (inner_zero_right (𝕜 := ℝ) (x := head))
          have htail := ih (i := k) (x := tail) (v := v)
          -- rewrite `x` as an append and use IH on the tail term
          have h' :
              inner ℝ x (appendVec (m := Shape.size s) (n := ctxSize ss)
                (vecOfFun (n := Shape.size s) fun _ => (0 : ℝ)) (singleRaw (Γ := ss) k v))
                =
              inner ℝ (getRaw (Γ := ss) k tail) v := by
            -- start from `inner_append`, drop the head/zero term, then apply IH
            simpa [hx, hhead0, htail, add_assoc, add_left_comm, add_comm] using hinner
          -- `getRaw`/`singleRaw` at `succ` are definitional on the tail
          simpa [CtxVec.singleRaw, CtxVec.getRaw, tail, hx] using h'

/-- Project the block specified by `idx : Idx Γ s` out of a vectorized context. -/
def get {Γ : List Shape} {s : Shape} (idx : Idx Γ s) (x : CtxVec Γ) : Vec (Shape.size s) :=
  castVec (congrArg Shape.size idx.h) (getRaw (Γ := Γ) idx.i x)

/-- Inject a block into a vectorized context at `idx`, filling other blocks with zeros. -/
def single {Γ : List Shape} {s : Shape} (idx : Idx Γ s) (v : Vec (Shape.size s)) : CtxVec Γ :=
  singleRaw (Γ := Γ) idx.i (castVec (congrArg Shape.size idx.h).symm v)

/-- Adjointness of `get`/`single`: `⟪x, single idx v⟫ = ⟪get idx x, v⟫`. -/
theorem inner_get_single {Γ : List Shape} {s : Shape} (idx : Idx Γ s)
    (x : CtxVec Γ) (v : Vec (Shape.size s)) :
    inner ℝ x (single (Γ := Γ) (s := s) idx v) = inner ℝ (get (Γ := Γ) (s := s) idx x) v := by
  classical
  -- unfold `get`/`single` and reduce to the raw statement + cast isometries
  let hsz : Shape.size (Γ.get idx.i) = Shape.size s := congrArg Shape.size idx.h
  -- use the raw lemma, then cancel the casts on both sides
  have hraw :=
    inner_getRaw_singleRaw (Γ := Γ) idx.i x (castVec hsz.symm v)
  -- rewrite RHS using cast-isometry
  have hcastR :
      inner ℝ (castVec hsz (getRaw (Γ := Γ) idx.i x)) v =
        inner ℝ (getRaw (Γ := Γ) idx.i x) (castVec hsz.symm v) := by
    -- reduce to the isometry lemma `inner_castVec_castVec`
    have hv : castVec hsz (castVec hsz.symm v) = v := by
      simp
    calc
      inner ℝ (castVec hsz (getRaw (Γ := Γ) idx.i x)) v
          = inner ℝ (castVec hsz (getRaw (Γ := Γ) idx.i x)) (castVec hsz (castVec hsz.symm v)) := by
              simp [hv]
      _ = inner ℝ (getRaw (Γ := Γ) idx.i x) (castVec hsz.symm v) := by
            simpa using
              (inner_castVec_castVec (h := hsz) (x := getRaw (Γ := Γ) idx.i x) (y := castVec
                hsz.symm v))
  simpa [get, single, hcastR] using hraw

/-- Continuous linear map extracting the head block of a nonempty vectorized context. -/
def headCLM {s : Shape} {ss : List Shape} : CtxVec (s :: ss) →L[ℝ] Vec (Shape.size s) := by
  classical
  let fLin : CtxVec (s :: ss) →ₗ[ℝ] Vec (Shape.size s) :=
    { toFun := fun x => vecOfFun (n := Shape.size s) fun j => x (Fin.castAdd (ctxSize ss) j)
      map_add' := by
        intro x y
        ext j
        simp
      map_smul' := by
        intro a x
        ext j
        simp [smul_eq_mul] }
  refine ⟨fLin, ?_⟩
  exact LinearMap.continuous_of_finiteDimensional (f := fLin)

@[simp] lemma headCLM_apply {s : Shape} {ss : List Shape} (x : CtxVec (s :: ss)) (j : Fin
  (Shape.size s)) :
    headCLM (s := s) (ss := ss) x j = x (Fin.castAdd (ctxSize ss) j) := by
  simp [headCLM]

/-- Continuous linear map extracting the tail blocks of a nonempty vectorized context. -/
def tailCLM {s : Shape} {ss : List Shape} : CtxVec (s :: ss) →L[ℝ] CtxVec ss := by
  classical
  let fLin : CtxVec (s :: ss) →ₗ[ℝ] CtxVec ss :=
    { toFun := fun x => vecOfFun (n := ctxSize ss) fun j => x (Fin.natAdd (Shape.size s) j)
      map_add' := by
        intro x y
        ext j
        simp
      map_smul' := by
        intro a x
        ext j
        simp [smul_eq_mul] }
  refine ⟨fLin, ?_⟩
  exact LinearMap.continuous_of_finiteDimensional (f := fLin)

@[simp] lemma tailCLM_apply {s : Shape} {ss : List Shape} (x : CtxVec (s :: ss)) (j : Fin (ctxSize
  ss)) :
    tailCLM (s := s) (ss := ss) x j = x (Fin.natAdd (Shape.size s) j) := by
  simp [tailCLM]

/-- `getRaw` packaged as a continuous linear map (constructed by recursion with
  `headCLM`/`tailCLM`). -/
def getCLMRaw : {Γ : List Shape} → (i : Fin Γ.length) → CtxVec Γ →L[ℝ] Vec (Shape.size (Γ.get i))
  | [], i => nomatch i
  | s :: ss, ⟨0, _⟩ => headCLM (s := s) (ss := ss)
  | s :: ss, ⟨Nat.succ k, hk⟩ =>
      (getCLMRaw (Γ := ss) ⟨k, Nat.lt_of_succ_lt_succ hk⟩).comp (tailCLM (s := s) (ss := ss))

@[simp] lemma getCLMRaw_apply {Γ : List Shape} (i : Fin Γ.length) (x : CtxVec Γ) :
    getCLMRaw (Γ := Γ) i x = getRaw (Γ := Γ) i x := by
  induction Γ with
  | nil =>
      exact (nomatch i)
  | cons s ss ih =>
      cases i with
      | mk val isLt =>
          cases val with
          | zero =>
              ext j
              simp [getCLMRaw, getRaw]
              change headCLM (s := s) (ss := ss) x j = x (Fin.castAdd (ctxSize ss) j)
              exact headCLM_apply (s := s) (ss := ss) x j
          | succ k =>
              -- peel one dimension and apply IH to the tail context
              let iTail : Fin ss.length := ⟨k, Nat.lt_of_succ_lt_succ isLt⟩
              have hrec := ih (i := iTail) (x := tailCLM (s := s) (ss := ss) x)
              ext j
              have := congrArg (fun v : Vec (Shape.size (ss.get iTail)) => v j) hrec
              simpa [getCLMRaw, getRaw, tailCLM, tailCLM_apply, iTail] using this

/-- `get` packaged as a continuous linear map. -/
def getCLM {Γ : List Shape} {s : Shape} (idx : Idx Γ s) : CtxVec Γ →L[ℝ] Vec (Shape.size s) :=
  (Graph.castCLM (h := congrArg Shape.size idx.h)).comp (getCLMRaw (Γ := Γ) idx.i)

@[simp] lemma getCLM_apply {Γ : List Shape} {s : Shape} (idx : Idx Γ s) (x : CtxVec Γ) :
    getCLM (Γ := Γ) (s := s) idx x = get (Γ := Γ) (s := s) idx x := by
  -- unfold and reduce to the `getCLMRaw_apply` lemma under `castVec`
  simp [getCLM, get, Graph.castCLM]
  exact congrArg (castVec (congrArg Shape.size idx.h)) (getCLMRaw_apply (Γ := Γ) (i := idx.i) (x :=
    x))

end CtxVec

-- ---------------------------------------------------------------------------
-- Nodes defined on `CtxVec` (so `forwardVec`/`jvpVec`/`vjpVec` are definitional)
-- ---------------------------------------------------------------------------

namespace Node

/-!
Nodes in this file are authored directly on the vectorized context `CtxVec`.

This is the most convenient authoring style for analytic proofs: `forwardVec`/`jvpVec`/`vjpVec`
are definitional, and the correctness obligation is an inner-product identity on Euclidean
vectors.
-/

/--
Convenience constructor: build a tape `Node` from vector-level forward/JVP/VJP plus adjointness.

The `correct_inner` field is exactly the local VJP/JVP law:
`⟪jvp x dx, δ⟫ = ⟪dx, vjp x δ⟫`.
-/
def ofVec {Γ : List Shape} {τ : Shape}
    (f : CtxVec Γ → Vec (Shape.size τ))
    (jvp : CtxVec Γ → CtxVec Γ → Vec (Shape.size τ))
    (vjp : CtxVec Γ → Vec (Shape.size τ) → CtxVec Γ)
    (correct_inner :
      ∀ (x dx : CtxVec Γ) (δ : Vec (Shape.size τ)),
        inner ℝ (jvp x dx) δ = inner ℝ dx (vjp x δ)) :
    Node Γ τ :=
{ forward := fun ctx => ofVecT (s := τ) (f (flattenCtx (Γ := Γ) ctx))
  jvp := fun ctx dctx => ofVecT (s := τ) (jvp (flattenCtx (Γ := Γ) ctx) (flattenCtx (Γ := Γ) dctx))
  vjp := fun ctx δ => unflattenCtx (Γ := Γ) (vjp (flattenCtx (Γ := Γ) ctx) (toVecT (t := δ)))
  correct := by
    intro ctx dctx δ
    let xV : CtxVec Γ := flattenCtx (Γ := Γ) ctx
    let dxV : CtxVec Γ := flattenCtx (Γ := Γ) dctx
    let δV : Vec (Shape.size τ) := toVecT (t := δ)
    have hL :
        dot (ofVecT (s := τ) (jvp xV dxV)) δ = inner ℝ (jvp xV dxV) δV := by
      simpa [δV] using (dot_eq_inner_toVecT (a := ofVecT (s := τ) (jvp xV dxV)) (b := δ))
    have hR :
        TList.dotList (ss := Γ) dctx (unflattenCtx (Γ := Γ) (vjp xV δV)) =
          inner ℝ dxV (vjp xV δV) := by
      simpa [dxV] using
        (dotList_eq_inner_flattenCtx (Γ := Γ) (x := dctx) (y := unflattenCtx (Γ := Γ) (vjp xV δV)))
    have hinner := correct_inner xV dxV δV
    simpa [xV, dxV, δV, hL, hR, toVecT_ofVecT, flattenCtx_unflattenCtx] using hinner
}

@[simp] lemma forwardVec_ofVec {Γ : List Shape} {τ : Shape}
    (f) (jvp) (vjp) (h) :
    (Node.forwardVec (Γ := Γ) (τ := τ) (ofVec (Γ := Γ) (τ := τ) f jvp vjp h)) = f := by
  funext xV
  simp [Node.forwardVec, ofVec]

@[simp] lemma jvpVec_ofVec {Γ : List Shape} {τ : Shape}
    (f) (jvp) (vjp) (h) :
    (Node.jvpVec (Γ := Γ) (τ := τ) (ofVec (Γ := Γ) (τ := τ) f jvp vjp h)) = jvp := by
  funext xV dxV
  simp [Node.jvpVec, ofVec]

@[simp] lemma vjpVec_ofVec {Γ : List Shape} {τ : Shape}
    (f) (jvp) (vjp) (h) :
    (Node.vjpVec (Γ := Γ) (τ := τ) (ofVec (Γ := Γ) (τ := τ) f jvp vjp h)) = vjp := by
  funext xV δV
  simp [Node.vjpVec, ofVec]

end Node

-- ---------------------------------------------------------------------------
-- Turning `OpSpecFDerivCorrect` into tape nodes at a context index
-- ---------------------------------------------------------------------------

namespace OpSpecFDerivCorrect

open scoped BigOperators

/--
`OpSpecFDerivCorrect` instance for a linear layer.

This is the analytic correctness lemma behind the tape node constructors: it identifies the JVP
with the Fréchet derivative (a matrix multiplication) for `linear_spec`.

PyTorch analogue: `torch.nn.Linear` forward map is affine, derivative is constant.
https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
-/
def linear {inDim outDim : Nat} (m : Spec.LinearSpec ℝ inDim outDim) :
    OpSpecFDerivCorrect inDim outDim :=
{ correct := linearCorrect (inDim := inDim) (outDim := outDim) m
  deriv := fun _xV =>
    matCLM (m := outDim) (n := inDim) (tensorToMatrix (m := outDim) (n := inDim) m.weights)
  hasFDerivAt := by
    intro xV
    -- The forward map is an affine function on Euclidean vectors.
    have hAffine :
        (fun xV : Vec inDim =>
            toVecE ((linearCorrect (inDim := inDim) (outDim := outDim) m).op.forward (ofVecE xV)))
          =
        affine (inDim := inDim) (outDim := outDim)
          (tensorToMatrix (m := outDim) (n := inDim) m.weights) (toVecE m.bias) := by
      funext xV
      simpa [linearCorrect, Spec.linearOp] using
        (toVecE_linear_spec (inDim := inDim) (outDim := outDim) m (x := ofVecE xV))
    have h :=
      hasFDerivAt_affine (inDim := inDim) (outDim := outDim)
        (W := tensorToMatrix (m := outDim) (n := inDim) m.weights) (b := toVecE m.bias) (x := xV)
    -- rewrite the goal function from `affine` to the `OpSpec` forward
    exact (hAffine.symm ▸ h)
  jvp_eq := by
    intro xV dxV
    -- The JVP is the linear part of `linear_spec` (bias drops out), so this is just
    -- `matCLM` applied to `dxV`.
    simpa [linearCorrect, Spec.linearOp, OpSpecCorrect.jvp, toVecE_ofVecE] using
      (toVecE_mat_vec_mul_spec (m := outDim) (n := inDim) m.weights (ofVecE dxV))
}

end OpSpecFDerivCorrect

end

end Autograd
end Proofs
