/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Tape.RealGraph.FDeriv
public import TorchLean.NN.FDeriv.Op
public import TorchLean.Tensor.Contract

/-!
# Tape-node context primitives

Low-level operations on vectorized contexts for the tape-node proof library: block projections,
one-hot cotangent injections and the node constructor `LawfulNode.ofFn` for maps on `CtxVec`. The
file also gives the `FDerivOp` witness for a linear layer, which `RealNode.unaryOp` in
`TorchLean.NN.Tape.RealNode.Elementwise` turns into a tape node.
-/

@[expose] public section

open TorchLean.Shape (size)

namespace TorchLean

open TorchLean.Tensor
open Rep (dot)

noncomputable section

/-- Inner product against a one-dimensional constant vector collapses to a single product.

Scalar tensors vectorize to `Vec 1`, and the lemma reduces the adjointness statement for a scalar
node from a sum over `Finset.univ` to one ordinary product. -/
@[simp] theorem Vec.inner_ofFun_scalar_left (a : ℝ) (δ : Vec (Shape.size Shape.scalar)) :
    inner ℝ (WithLp.toLp 2 fun _ => a) δ = a * δ.ofLp ⟨0, by simp⟩ := by
  rw [inner_eq_sum_mul]
  simp only
  rw [show Finset.univ = {(⟨0, by simp⟩ :
      Fin (Shape.size Shape.scalar))} by
    apply Finset.eq_singleton_iff_unique_mem.2
    constructor
    · exact Finset.mem_univ _
    · intro b _
      exact Fin.eq_of_val_eq (Nat.eq_zero_of_le_zero (Nat.le_of_lt_succ b.isLt))]
  exact Finset.sum_singleton _ _

-- ---------------------------------------------------------------------------
-- Context slicing on `CtxVec`
-- ---------------------------------------------------------------------------

namespace CtxVec

/-- Project a vectorized context onto the block at list position `i`.

`CtxVec.get` is the shape-indexed interface, which takes an `Idx Γ s` and transports the result to
the statically known shape `s`.
-/
def getBlock : {Γ : List Shape} → (i : Fin Γ.length) → CtxVec Γ → Vec (Shape.size (Γ.get i))
  | [], i, _ => nomatch i
  | _ :: ss, ⟨0, _⟩, v => WithLp.toLp 2 fun j => v (Fin.castAdd (ctxSize ss) j)
  | s :: ss, ⟨Nat.succ k, hk⟩, v =>
      getBlock (Γ := ss) ⟨k, Nat.lt_of_succ_lt_succ hk⟩
        (WithLp.toLp 2 fun j => v (Fin.natAdd (Shape.size s) j))

/-- Project the block specified by `idx : Idx Γ s` out of a vectorized context. -/
def get {Γ : List Shape} {s : Shape} (idx : Idx Γ s) (x : CtxVec Γ) : Vec (Shape.size s) :=
  LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ (finCongr (congrArg Shape.size idx.h))
    (getBlock (Γ := Γ) idx.i x)

/-- Selecting a block after vectorization agrees with selecting the original tensor. -/
theorem getBlock_toVec {Γ : List Shape} (xs : TensorPack ℝ Γ) (i : Fin Γ.length) :
    getBlock i (TensorPack.toVec xs) = toVec (xs.get i) := by
  induction xs with
  | nil => exact nomatch i
  | cons x xs ih =>
    obtain ⟨i, hi⟩ := i
    cases i with
    | zero =>
      rw [Pack.get_cons_zero]
      apply PiLp.ext
      intro j
      simp [getBlock, TensorPack.toVec_cons]
    | succ i =>
      rw [Pack.get_cons_succ]
      change getBlock ⟨i, Nat.lt_of_succ_lt_succ hi⟩ (WithLp.toLp 2 fun j =>
        (TensorPack.toVec (.cons x xs)) (Fin.natAdd _ j)) = _
      rw [← ih ⟨i, Nat.lt_of_succ_lt_succ hi⟩]
      congr 1
      apply PiLp.ext
      intro j
      simp [TensorPack.toVec_cons]

/-- Typed tensor selection commutes with vectorization, including its shape cast. -/
theorem get_toVec {Γ : List Shape} {s : Shape}
    (idx : Idx Γ s) (xs : TensorPack ℝ Γ) :
    get idx (TensorPack.toVec xs) = toVec (TensorPack.getIdx xs idx) := by
  obtain ⟨i, rfl⟩ := idx
  exact getBlock_toVec xs i

/-- The vectorization of `TensorPack.single`: the block `v` at `idx` and zeros in every
other block. -/
def single {Γ : List Shape} {s : Shape} (idx : Idx Γ s) (v : Vec (Shape.size s)) : CtxVec Γ :=
  TensorPack.toVec (TensorPack.single idx (ofVec v))

/-- Adjointness of `get`/`single`: `⟪x, single idx v⟫ = ⟪get idx x, v⟫`. -/
theorem inner_get_single {Γ : List Shape} {s : Shape} (idx : Idx Γ s)
    (x : CtxVec Γ) (v : Vec (Shape.size s)) :
    inner ℝ x (single (Γ := Γ) (s := s) idx v) = inner ℝ (get (Γ := Γ) (s := s) idx x) v := by
  rw [single, ← TensorPack.toVec_ofVec x, ← TensorPack.realDotList_eq_inner_toVec,
    TensorPack.dotList_single, dot_eq_inner_toVec, get_toVec,
    toVec_ofVec]

/-- We bundle `getBlock` as a continuous linear map: block `0` is the first component of
`EuclideanSpace.finAddEquivProd`, and block `k + 1` is block `k` of its second component. -/
def getBlockCLM :
    {Γ : List Shape} → (i : Fin Γ.length) → CtxVec Γ →L[ℝ] Vec (Shape.size (Γ.get i))
  | [], i => nomatch i
  | s :: ss, ⟨0, _⟩ =>
      (ContinuousLinearMap.fst ℝ (Vec (Shape.size s)) (CtxVec ss)).comp
        (EuclideanSpace.finAddEquivProd (𝕜 := ℝ)).toContinuousLinearMap
  | s :: ss, ⟨Nat.succ k, hk⟩ =>
      (getBlockCLM (Γ := ss) ⟨k, Nat.lt_of_succ_lt_succ hk⟩).comp
        ((ContinuousLinearMap.snd ℝ (Vec (Shape.size s)) (CtxVec ss)).comp
          (EuclideanSpace.finAddEquivProd (𝕜 := ℝ)).toContinuousLinearMap)

/-- The bundled `getBlockCLM i` agrees with `getBlock i` on every vectorized context. -/
@[simp] theorem getBlockCLM_apply {Γ : List Shape} (i : Fin Γ.length) (x : CtxVec Γ) :
    getBlockCLM (Γ := Γ) i x = getBlock (Γ := Γ) i x := by
  induction Γ with
  | nil =>
      exact (nomatch i)
  | cons s ss ih =>
      cases i with
      | mk val isLt =>
          cases val with
          | zero =>
              rfl
          | succ k =>
              -- We apply the induction hypothesis to the tail context.
              exact ih (i := ⟨k, Nat.lt_of_succ_lt_succ isLt⟩)
                (x := (EuclideanSpace.finAddEquivProd (𝕜 := ℝ) (n := Shape.size s)
                  (m := ctxSize ss) x).2)

/-- `get` packaged as a continuous linear map. -/
def getCLM {Γ : List Shape} {s : Shape} (idx : Idx Γ s) : CtxVec Γ →L[ℝ] Vec (Shape.size s) :=
  ContinuousLinearMap.comp
    (LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ
      (finCongr (congrArg Shape.size idx.h))).toContinuousLinearEquiv.toContinuousLinearMap
    (getBlockCLM (Γ := Γ) idx.i)

/-- `getCLM` applied to a context agrees with `get`. -/
@[simp] theorem getCLM_apply {Γ : List Shape} {s : Shape} (idx : Idx Γ s) (x : CtxVec Γ) :
    getCLM (Γ := Γ) (s := s) idx x = get (Γ := Γ) (s := s) idx x := by
  -- We reduce to `getBlockCLM_apply` under the coordinate cast.
  exact congrArg
    (LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ (finCongr (congrArg Shape.size idx.h)))
    (getBlockCLM_apply (Γ := Γ) (i := idx.i) (x := x))

/-- The adjoint of block projection is block injection. -/
theorem adjoint_getCLM {Γ : List Shape} {s : Shape} (idx : Idx Γ s)
    (v : Vec (Shape.size s)) :
    (CtxVec.getCLM idx).adjoint v = CtxVec.single idx v := by
  apply ext_inner_left ℝ
  intro x
  rw [ContinuousLinearMap.adjoint_inner_right, CtxVec.getCLM_apply, CtxVec.inner_get_single]

/-- Reading a context slot is differentiable, with derivative the bundled projection. -/
theorem hasFDerivAt_get {Γ : List Shape} {s : Shape} (idx : Idx Γ s) (x : CtxVec Γ) :
    HasFDerivAt (fun y : CtxVec Γ => get idx y) (getCLM idx) x :=
  (getCLM idx).hasFDerivAt.congr_of_eventuallyEq
    (Filter.Eventually.of_forall fun y => (getCLM_apply idx y).symm)

end CtxVec

-- ---------------------------------------------------------------------------
-- Nodes defined on `CtxVec` (`forwardVec`/`jvpVec`/`vjpVec` reduce by `simp`)
-- ---------------------------------------------------------------------------

namespace Tape

namespace LawfulNode

/-!
Nodes built with `ofFn` are authored on the vectorized context `CtxVec`.

That style suits analytic proofs: the simp lemmas `forwardVec_ofFn`, `jvpVec_ofFn` and
`vjpVec_ofFn` return the supplied maps, and the correctness obligation is an inner-product identity
on Euclidean vectors.
-/

/--
Build a tape `LawfulNode` from vector-level forward, JVP and VJP maps and a proof of their
adjointness. The node ignores its environment `d : Δ`.

The argument `correct_inner` is the local JVP/VJP law `⟪jvp x dx, δ⟫ = ⟪dx, vjp x δ⟫`.
-/
def ofFn {Δ : Type} {Γ : List Shape} {τ : Shape}
    (f : CtxVec Γ → Vec (Shape.size τ))
    (jvp : CtxVec Γ → CtxVec Γ → Vec (Shape.size τ))
    (vjp : CtxVec Γ → Vec (Shape.size τ) → CtxVec Γ)
    (correct_inner :
      ∀ (x dx : CtxVec Γ) (δ : Vec (Shape.size τ)),
        inner ℝ (jvp x dx) δ = inner ℝ dx (vjp x δ)) :
    LawfulNode (α := ℝ) Δ Γ τ :=
{ forward := fun ctx _ => ofVec (s := τ) (f (TensorPack.toVec (Γ := Γ) ctx))
  jvp := fun ctx dctx _ =>
    ofVec (s := τ) (jvp (TensorPack.toVec (Γ := Γ) ctx) (TensorPack.toVec (Γ := Γ) dctx))
  vjp := fun ctx _ δ =>
    TensorPack.ofVec (Γ := Γ) (vjp (TensorPack.toVec (Γ := Γ) ctx) (toVec (t := δ)))
  jvp_vjp_adjoint := by
    intro ctx dctx _ δ
    let xV : CtxVec Γ := TensorPack.toVec (Γ := Γ) ctx
    let dxV : CtxVec Γ := TensorPack.toVec (Γ := Γ) dctx
    let δV : Vec (Shape.size τ) := toVec (t := δ)
    have hL :
        Rep.dot (ofVec (s := τ) (jvp xV dxV)) δ = inner ℝ (jvp xV dxV) δV := by
      simpa [δV] using (dot_eq_inner_toVec (a := ofVec (s := τ) (jvp xV dxV)) (b := δ))
    have hR :
        TensorPack.dotList (ss := Γ) dctx (TensorPack.ofVec (Γ := Γ) (vjp xV δV)) =
          inner ℝ dxV (vjp xV δV) := by
      simpa [dxV] using
        (TensorPack.realDotList_eq_inner_toVec (Γ := Γ) (x := dctx)
          (y := TensorPack.ofVec (Γ := Γ) (vjp xV δV)))
    have hinner := correct_inner xV dxV δV
    simpa [xV, dxV, δV, hL, hR, toVec_ofVec, TensorPack.toVec_ofVec] using hinner
}

/-- A node built by `ofFn` has the given forward map. -/
@[simp] theorem forwardVec_ofFn {Δ : Type} {Γ : List Shape} {τ : Shape}
    (f) (jvp) (vjp) (h) (d : Δ) :
    (ofFn (Δ := Δ) (Γ := Γ) (τ := τ) f jvp vjp h).forwardVec d = f := by
  funext xV
  simp [NodeData.forwardVec, ofFn]

/-- A node built by `ofFn` has the given forward-mode derivative. -/
@[simp] theorem jvpVec_ofFn {Δ : Type} {Γ : List Shape} {τ : Shape}
    (f) (jvp) (vjp) (h) (d : Δ) :
    (ofFn (Δ := Δ) (Γ := Γ) (τ := τ) f jvp vjp h).jvpVec d = jvp := by
  funext xV dxV
  simp [NodeData.jvpVec, ofFn]

/-- A node built by `ofFn` has the given reverse-mode derivative. Together the three projection
lemmas mean a caller never has to unfold `ofFn`, only supply the soundness argument `h` once. -/
@[simp] theorem vjpVec_ofFn {Δ : Type} {Γ : List Shape} {τ : Shape}
    (f) (jvp) (vjp) (h) (d : Δ) :
    (ofFn (Δ := Δ) (Γ := Γ) (τ := τ) f jvp vjp h).vjpVec d = vjp := by
  funext xV δV
  simp [NodeData.vjpVec, ofFn]

end LawfulNode

end Tape

-- ---------------------------------------------------------------------------
-- `FDerivOp` witness for a linear layer
-- ---------------------------------------------------------------------------

namespace FDerivOp

/--
`FDerivOp` witness for a linear layer.

It identifies the JVP of `Linear.forward` with its Fréchet derivative, the constant matrix
multiplication by the weights. `Tape.RealNode.linear` and its certificate
`Tape.RealNode.linearFDeriv` are built from it through `unaryOp` and `unaryOpFDeriv`.

PyTorch analogue: the `torch.nn.Linear` forward map is affine, so its derivative is constant.
https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
-/
def linear {inDim outDim : Nat} (m : Linear ℝ inDim outDim) :
    FDerivOp inDim outDim :=
{ toAdjointOp := Tape.AdjointOp.linear (inDim := inDim) (outDim := outDim) m
  deriv := fun _xV =>
    (Matrix.toEuclideanLin (Rep.matrixEquiv ℝ _ _ m.weights)).toContinuousLinearMap
  hasFDerivAt := by
    intro xV
    -- The forward map is an affine function on Euclidean vectors.
    have hAffine :
        (fun xV : Vec inDim =>
            getScalarE
              ((Tape.AdjointOp.linear (inDim := inDim) (outDim := outDim) m).op.forward
                (ofFnE xV)))
          =
        Vec.affine (inDim := inDim) (outDim := outDim)
          (Rep.matrixEquiv ℝ outDim inDim m.weights) (getScalarE m.bias) := by
      funext xV
      simpa [Tape.AdjointOp.linear] using
        (Linear.getScalarE_forward (inDim := inDim) (outDim := outDim) m (x := ofFnE xV))
    have h :=
      Vec.hasFDerivAt_affine (inDim := inDim) (outDim := outDim)
        (W := Rep.matrixEquiv ℝ outDim inDim m.weights) (b := getScalarE m.bias)
        (x := xV)
    -- rewrite the goal function from `affine` to the `VJPOp` forward
    exact (hAffine.symm ▸ h)
  jvp_eq := by
    intro xV dxV
    -- The JVP is the linear part of `Linear.forward` (bias drops out), so this is just
    -- `Matrix.toEuclideanLin` of the weights applied to `dxV`.
    simpa [Tape.AdjointOp.linear, Tape.AdjointOp.jvp, getScalarE_ofFnE] using
      (getScalarE_contract_matVec (m := outDim) (n := inDim) m.weights (ofFnE dxV))
}

end FDerivOp

end

end TorchLean
