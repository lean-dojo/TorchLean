/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Tape.Core.FDeriv
public import NN.Proofs.Autograd.FDeriv.OpSpec

/-!
# Tape-node context primitives

This module contains the low-level vectorized context operations used by the tape-node proof
library: block projections, one-hot cotangent injections, and the bridge from generic
`OpSpecFDerivCorrect` witnesses to `NodeFDerivCorrect` nodes.
-/

@[expose] public section

namespace Proofs
namespace Autograd

open Spec TorchLean
open TorchLean TorchLean.Tensor

noncomputable section

open scoped BigOperators

/-!
### Coordinate lemmas for the Euclidean identification

Mathlib's `EuclideanSpace` is `PiLp 2`, a type synonym carrying a `WithLp` wrapper. The four lemmas
below are the plumbing that lets us forget the wrapper: each says that reading coordinate `i` of a
vector built from `f` gives `f i`, for the various shapes the wrapper takes. They are boring on
purpose, and having them as `simp` lemmas is what keeps the real proofs in this file readable.
-/

/-- Coordinates of the inverse `PiLp` equivalence are the values of the underlying function. -/
@[simp] theorem piLpContinuousLinearEquiv2_symm_apply {n : Nat} (f : Fin n → ℝ) (i : Fin n) :
    ((PiLp.continuousLinearEquiv 2 ℝ (fun _ : Fin n => ℝ)).symm f) i = f i := by
  simp

/-- The same, applied through the bundled continuous linear map. -/
@[simp] theorem piLpContinuousLinearEquiv2_symm_clm_apply {n : Nat} (f : Fin n → ℝ) (i : Fin n) :
    (((PiLp.continuousLinearEquiv 2 ℝ (fun _ : Fin n => ℝ)).symm.toContinuousLinearMap) f) i = f i
      := by
  simp

/-- The same again, with the `ofLp` projection made explicit. -/
@[simp] theorem piLpContinuousLinearEquiv2_symm_clm_apply_ofLp {n : Nat} (f : Fin n → ℝ)
    (i : Fin n) :
    (((PiLp.continuousLinearEquiv 2 ℝ (fun _ : Fin n => ℝ)).symm.toContinuousLinearMap) f).ofLp i =
      f i := by
  simp

/-- And once more for `EuclideanSpace.equiv`, the spelling used by `vecOfFun`. -/
@[simp] theorem euclideanEquiv_symm_ofLp {n : Nat} (f : Fin n → ℝ) (i : Fin n) :
    ((EuclideanSpace.equiv (𝕜 := ℝ) (ι := Fin n)).symm f).ofLp i = f i := by
  simp [EuclideanSpace.equiv]

/-- Inner product against a one-dimensional constant vector collapses to a single product.

Scalar tensors vectorize to `Vec 1`, so this is the lemma that turns the adjointness statement for a
scalar node into ordinary multiplication instead of a sum over `Finset.univ`. -/
@[simp] theorem inner_scalarVec_left (a : ℝ) (δ : Vec (Spec.Shape.size Shape.scalar)) :
    inner ℝ (vecOfFun (n := Spec.Shape.size Shape.scalar) fun _ => a) δ = a * δ.ofLp ⟨0, by simp
      [Spec.Shape.size]⟩ := by
  rw [inner_eq_sum_mul]
  simp only [vecOfFun_ofLp]
  rw [show Finset.univ = {(⟨0, by simp [Spec.Shape.size]⟩ :
      Fin (Spec.Shape.size Shape.scalar))} by
    apply Finset.eq_singleton_iff_unique_mem.2
    constructor
    · simp
    · intro b _
      exact Fin.eq_of_val_eq (Nat.eq_zero_of_le_zero (Nat.le_of_lt_succ b.isLt))]
  simp

-- We use `inner_append` (proved once in `NN/Proofs/Autograd/Tape/Core/FDeriv.lean`) to split
-- inner products on concatenated Euclidean vectors `appendVec a b`.

-- ---------------------------------------------------------------------------
-- Context slicing on `CtxVec`
-- ---------------------------------------------------------------------------

namespace CtxVec

/-- Project a vectorized context onto the block at list position `i`.

`CtxVec.get` is the shape-indexed interface; it accepts an `Idx Γ s` and transports the result to
the statically known shape `s`.
-/
def getBlock : {Γ : List Shape} → (i : Fin Γ.length) → CtxVec Γ → Vec (Spec.Shape.size (Γ.get i))
  | [], i, _ => nomatch i
  | s :: ss, ⟨0, _⟩, v => vecOfFun (n := Spec.Shape.size s) fun j => v (Fin.castAdd (ctxSize ss) j)
  | s :: ss, ⟨Nat.succ k, hk⟩, v =>
      getBlock (Γ := ss) ⟨k, Nat.lt_of_succ_lt_succ hk⟩
        (vecOfFun (n := ctxSize ss) fun j => v (Fin.natAdd (Spec.Shape.size s) j))

/-- Inject `v` into block `i` of a vectorized context and fill every other block with zeros.

This is the adjoint of `getBlock` with respect to the Euclidean inner product.
-/
def singleBlock : {Γ : List Shape} → (i : Fin Γ.length) → Vec (Spec.Shape.size (Γ.get i)) → CtxVec Γ
  | [], i, _ => nomatch i
  | s :: ss, ⟨0, _⟩, v =>
      appendVec (m := Spec.Shape.size s) (n := ctxSize ss) v
        (vecOfFun (n := ctxSize ss) fun _ => (0 : ℝ))
  | s :: ss, ⟨Nat.succ k, hk⟩, v =>
      appendVec (m := Spec.Shape.size s) (n := ctxSize ss)
        (vecOfFun (n := Spec.Shape.size s) fun _ => (0 : ℝ))
        (singleBlock (Γ := ss) ⟨k, Nat.lt_of_succ_lt_succ hk⟩ v)

/-- Adjointness of block projection and injection.

This is the vectorized counterpart of the one-hot cotangent principle used in tape soundness.
-/
theorem inner_getBlock_singleBlock :
    ∀ {Γ : List Shape} (i : Fin Γ.length) (x : CtxVec Γ) (v : Vec (Spec.Shape.size (Γ.get i))),
      inner ℝ x (singleBlock (Γ := Γ) i v) = inner ℝ (getBlock (Γ := Γ) i x) v := by
  intro Γ
  induction Γ with
  | nil =>
      intro i
      exact (nomatch i)
  | cons s ss ih =>
      intro i x v
      classical
      -- decompose `x` as `(head, tail)` and use `inner_append`
      let head : Vec (Spec.Shape.size s) :=
        vecOfFun (n := Spec.Shape.size s) fun j => x (Fin.castAdd (ctxSize ss) j)
      let tail : Vec (ctxSize ss) :=
        vecOfFun (n := ctxSize ss) fun j => x (Fin.natAdd (Spec.Shape.size s) j)
      have hx : appendVec (m := Spec.Shape.size s) (n := ctxSize ss) head tail = x := by
        ext j
        have := congrArg (fun f : Fin (Spec.Shape.size s + ctxSize ss) → ℝ => f j)
          (Fin.append_castAdd_natAdd (f := x) (m := Spec.Shape.size s) (n := ctxSize ss))
        change
          Fin.append
              (fun j : Fin (Spec.Shape.size s) => x.ofLp (Fin.castAdd (ctxSize ss) j))
              (fun j : Fin (ctxSize ss) => x.ofLp (Fin.natAdd (Spec.Shape.size s) j)) j =
            x.ofLp j
        simpa using this
      cases i using Fin.cases with
      | zero =>
          have hinner :=
            inner_append (m := Spec.Shape.size s) (n := ctxSize ss) (a := head) (b := tail)
              (c := v) (d := vecOfFun (n := ctxSize ss) fun _ => (0 : ℝ))
          have htail0 : inner ℝ tail (vecOfFun (n := ctxSize ss) fun _ => (0 : ℝ)) = 0 := by
            exact (inner_zero_right (𝕜 := ℝ) (x := tail))
          have h' : inner ℝ x (appendVec (m := Spec.Shape.size s) (n := ctxSize ss) v
              (vecOfFun (n := ctxSize ss) fun _ => (0 : ℝ))) =
              inner ℝ head v := by
            -- rewrite `x` as an append and simplify away the tail/zero inner product
            calc
              inner ℝ x (appendVec (m := Spec.Shape.size s) (n := ctxSize ss) v
                  (vecOfFun (n := ctxSize ss) fun _ => (0 : ℝ)))
                  =
                inner ℝ (appendVec (m := Spec.Shape.size s) (n := ctxSize ss) head tail)
                  (appendVec (m := Spec.Shape.size s) (n := ctxSize ss) v
                    (vecOfFun (n := ctxSize ss) fun _ => (0 : ℝ))) := by
                    rw [← hx]
                    rfl
              _ = inner ℝ head v + inner ℝ tail (vecOfFun (n := ctxSize ss) fun _ => (0 : ℝ)) :=
                    hinner
              _ = inner ℝ head v := by
                    rw [htail0, add_zero]
          change
            inner ℝ x (appendVec (m := Spec.Shape.size s) (n := ctxSize ss) v
              (vecOfFun (n := ctxSize ss) fun _ => (0 : ℝ))) =
              inner ℝ head v
          exact h'
      | succ k =>
          have hinner :=
            inner_append (m := Spec.Shape.size s) (n := ctxSize ss)
              (a := head) (b := tail) (c := vecOfFun (n := Spec.Shape.size s) fun _ => (0 : ℝ))
              (d := singleBlock (Γ := ss) k v)
          have hhead0 : inner ℝ head (vecOfFun (n := Spec.Shape.size s) fun _ => (0 : ℝ)) = 0 := by
            exact (inner_zero_right (𝕜 := ℝ) (x := head))
          have htail := ih (i := k) (x := tail) (v := v)
          -- rewrite `x` as an append and use IH on the tail term
          have h' :
              inner ℝ x (appendVec (m := Spec.Shape.size s) (n := ctxSize ss)
                (vecOfFun (n := Spec.Shape.size s) fun _ => (0 : ℝ)) (singleBlock (Γ := ss) k v))
                =
              inner ℝ (getBlock (Γ := ss) k tail) v := by
            -- start from `inner_append`, drop the head/zero term, then apply IH
            calc
              inner ℝ x (appendVec (m := Spec.Shape.size s) (n := ctxSize ss)
                  (vecOfFun (n := Spec.Shape.size s) fun _ => (0 : ℝ)) (singleBlock (Γ := ss) k v))
                  =
                inner ℝ (appendVec (m := Spec.Shape.size s) (n := ctxSize ss) head tail)
                  (appendVec (m := Spec.Shape.size s) (n := ctxSize ss)
                    (vecOfFun (n := Spec.Shape.size s) fun _ => (0 : ℝ))
                    (singleBlock (Γ := ss) k v)) := by
                    rw [← hx]
                    rfl
              _ = inner ℝ head (vecOfFun (n := Spec.Shape.size s) fun _ => (0 : ℝ)) +
                    inner ℝ tail (singleBlock (Γ := ss) k v) := hinner
              _ = inner ℝ (getBlock (Γ := ss) k tail) v := by
                    rw [hhead0, zero_add, htail]
          -- `getBlock`/`singleBlock` at `succ` are definitional on the tail
          change
            inner ℝ x (appendVec (m := Spec.Shape.size s) (n := ctxSize ss)
              (vecOfFun (n := Spec.Shape.size s) fun _ => (0 : ℝ)) (singleBlock (Γ := ss) k v)) =
              inner ℝ (getBlock (Γ := ss) k tail) v
          exact h'

/-- Project the block specified by `idx : Idx Γ s` out of a vectorized context. -/
def get {Γ : List Shape} {s : Shape} (idx : Idx Γ s) (x : CtxVec Γ) : Vec (Spec.Shape.size s) :=
  castVec (congrArg Spec.Shape.size idx.h) (getBlock (Γ := Γ) idx.i x)

/-- Inject a block into a vectorized context at `idx`, filling other blocks with zeros. -/
def single {Γ : List Shape} {s : Shape} (idx : Idx Γ s) (v : Vec (Spec.Shape.size s)) : CtxVec Γ :=
  singleBlock (Γ := Γ) idx.i (castVec (congrArg Spec.Shape.size idx.h).symm v)

/-- Adjointness of `get`/`single`: `⟪x, single idx v⟫ = ⟪get idx x, v⟫`. -/
theorem inner_get_single {Γ : List Shape} {s : Shape} (idx : Idx Γ s)
    (x : CtxVec Γ) (v : Vec (Spec.Shape.size s)) :
    inner ℝ x (single (Γ := Γ) (s := s) idx v) = inner ℝ (get (Γ := Γ) (s := s) idx x) v := by
  classical
  -- unfold `get`/`single` and reduce to the raw statement + cast isometries
  let hsz : Spec.Shape.size (Γ.get idx.i) = Spec.Shape.size s := congrArg Spec.Shape.size idx.h
  -- use the raw lemma, then cancel the casts on both sides
  have hraw :=
    inner_getBlock_singleBlock (Γ := Γ) idx.i x (castVec hsz.symm v)
  -- rewrite RHS using cast-isometry
  have hcastR :
      inner ℝ (castVec hsz (getBlock (Γ := Γ) idx.i x)) v =
        inner ℝ (getBlock (Γ := Γ) idx.i x) (castVec hsz.symm v) := by
    -- reduce to the isometry lemma `inner_castVec_castVec`
    have hv : castVec hsz (castVec hsz.symm v) = v := by
      simp
    calc
      inner ℝ (castVec hsz (getBlock (Γ := Γ) idx.i x)) v
          = inner ℝ (castVec hsz (getBlock (Γ := Γ) idx.i x))
              (castVec hsz (castVec hsz.symm v)) := by
              simp [hv]
      _ = inner ℝ (getBlock (Γ := Γ) idx.i x) (castVec hsz.symm v) := by
            simpa using
              (inner_castVec_castVec (h := hsz) (x := getBlock (Γ := Γ) idx.i x) (y := castVec
                hsz.symm v))
  simpa [get, single, hcastR] using hraw

/-- Continuous linear map extracting the head block of a nonempty vectorized context. -/
def headCLM {s : Shape} {ss : List Shape} : CtxVec (s :: ss) →L[ℝ] Vec (Spec.Shape.size s) := by
  classical
  let fLin : CtxVec (s :: ss) →ₗ[ℝ] Vec (Spec.Shape.size s) :=
    { toFun := fun x => vecOfFun (n := Spec.Shape.size s) fun j => x (Fin.castAdd (ctxSize ss) j)
      map_add' := by
        intro x y
        ext j
        change x.ofLp _ + y.ofLp _ = _
        rfl
      map_smul' := by
        intro a x
        ext j
        change a * x.ofLp _ = _
        rfl }
  refine ⟨fLin, ?_⟩
  exact LinearMap.continuous_of_finiteDimensional (f := fLin)

/-- `headCLM` reads the leading block of coordinates, exactly as `flattenCtx_cons` lays them out. -/
@[simp] theorem headCLM_apply {s : Shape} {ss : List Shape} (x : CtxVec (s :: ss)) (j : Fin
  (Spec.Shape.size s)) :
    headCLM (s := s) (ss := ss) x j = x (Fin.castAdd (ctxSize ss) j) := by
  simp [headCLM]

/-- Continuous linear map extracting the tail blocks of a nonempty vectorized context. -/
def tailCLM {s : Shape} {ss : List Shape} : CtxVec (s :: ss) →L[ℝ] CtxVec ss := by
  classical
  let fLin : CtxVec (s :: ss) →ₗ[ℝ] CtxVec ss :=
    { toFun := fun x => vecOfFun (n := ctxSize ss) fun j => x (Fin.natAdd (Spec.Shape.size s) j)
      map_add' := by
        intro x y
        ext j
        change x.ofLp _ + y.ofLp _ = _
        rfl
      map_smul' := by
        intro a x
        ext j
        change a * x.ofLp _ = _
        rfl }
  refine ⟨fLin, ?_⟩
  exact LinearMap.continuous_of_finiteDimensional (f := fLin)

/-- `tailCLM` reads the trailing blocks. -/
@[simp] theorem tailCLM_apply {s : Shape} {ss : List Shape} (x : CtxVec (s :: ss)) (j : Fin (ctxSize
  ss)) :
    tailCLM (s := s) (ss := ss) x j = x (Fin.natAdd (Spec.Shape.size s) j) := by
  simp [tailCLM]

/-- `getBlock` as a continuous linear map, constructed recursively from `headCLM` and `tailCLM`. -/
def getBlockCLM :
    {Γ : List Shape} → (i : Fin Γ.length) → CtxVec Γ →L[ℝ] Vec (Spec.Shape.size (Γ.get i))
  | [], i => nomatch i
  | s :: ss, ⟨0, _⟩ => headCLM (s := s) (ss := ss)
  | s :: ss, ⟨Nat.succ k, hk⟩ =>
      (getBlockCLM (Γ := ss) ⟨k, Nat.lt_of_succ_lt_succ hk⟩).comp (tailCLM (s := s) (ss := ss))

/-- The bundled `getBlockCLM` computes the same thing as the plain `getBlock`.

This is the payoff of building it recursively from `headCLM` and `tailCLM`: block selection comes
out
continuous and linear by construction, so nothing downstream has to prove it again. -/
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
              ext j
              simp [getBlockCLM, getBlock]
          | succ k =>
              -- peel one dimension and apply IH to the tail context
              let iTail : Fin ss.length := ⟨k, Nat.lt_of_succ_lt_succ isLt⟩
              have hrec := ih (i := iTail) (x := tailCLM (s := s) (ss := ss) x)
              ext j
              have := congrArg (fun v : Vec (Spec.Shape.size (ss.get iTail)) => v j) hrec
              change
                ((getBlockCLM (Γ := ss) iTail)
                    (tailCLM (s := s) (ss := ss) x)).ofLp j =
                  (getBlock (Γ := ss) iTail
                    (vecOfFun (n := ctxSize ss) fun j =>
                      x.ofLp (Fin.natAdd (Spec.Shape.size s) j))).ofLp j
              simpa [getBlockCLM, getBlock, tailCLM, tailCLM_apply, iTail] using this

/-- `get` packaged as a continuous linear map. -/
def getCLM {Γ : List Shape} {s : Shape} (idx : Idx Γ s) : CtxVec Γ →L[ℝ] Vec (Spec.Shape.size s) :=
  (Graph.castCLM (h := congrArg Spec.Shape.size idx.h)).comp (getBlockCLM (Γ := Γ) idx.i)

/-- And the shape-indexed `getCLM` agrees with `get`. -/
@[simp] theorem getCLM_apply {Γ : List Shape} {s : Shape} (idx : Idx Γ s) (x : CtxVec Γ) :
    getCLM (Γ := Γ) (s := s) idx x = get (Γ := Γ) (s := s) idx x := by
  -- Unfold and reduce to `getBlockCLM_apply` under `castVec`.
  simp [getCLM, get, Graph.castCLM]
  exact congrArg (castVec (congrArg Spec.Shape.size idx.h))
    (getBlockCLM_apply (Γ := Γ) (i := idx.i) (x := x))

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
def ofFn {Γ : List Shape} {τ : Shape}
    (f : CtxVec Γ → Vec (Spec.Shape.size τ))
    (jvp : CtxVec Γ → CtxVec Γ → Vec (Spec.Shape.size τ))
    (vjp : CtxVec Γ → Vec (Spec.Shape.size τ) → CtxVec Γ)
    (correct_inner :
      ∀ (x dx : CtxVec Γ) (δ : Vec (Spec.Shape.size τ)),
        inner ℝ (jvp x dx) δ = inner ℝ dx (vjp x δ)) :
    Node Γ τ :=
{ forward := fun ctx => vecToTensor (s := τ) (f (flattenCtx (Γ := Γ) ctx))
  jvp := fun ctx dctx =>
    vecToTensor (s := τ) (jvp (flattenCtx (Γ := Γ) ctx) (flattenCtx (Γ := Γ) dctx))
  vjp := fun ctx δ => unflattenCtx (Γ := Γ) (vjp (flattenCtx (Γ := Γ) ctx) (tensorToVec (t := δ)))
  correct := by
    intro ctx dctx δ
    let xV : CtxVec Γ := flattenCtx (Γ := Γ) ctx
    let dxV : CtxVec Γ := flattenCtx (Γ := Γ) dctx
    let δV : Vec (Spec.Shape.size τ) := tensorToVec (t := δ)
    have hL :
        dot (vecToTensor (s := τ) (jvp xV dxV)) δ = inner ℝ (jvp xV dxV) δV := by
      simpa [δV] using (dot_eq_inner_tensorToVec (a := vecToTensor (s := τ) (jvp xV dxV)) (b := δ))
    have hR :
        TensorPack.dotList (ss := Γ) dctx (unflattenCtx (Γ := Γ) (vjp xV δV)) =
          inner ℝ dxV (vjp xV δV) := by
      simpa [dxV] using
        (dotList_eq_inner_flattenCtx (Γ := Γ) (x := dctx) (y := unflattenCtx (Γ := Γ) (vjp xV δV)))
    have hinner := correct_inner xV dxV δV
    simpa [xV, dxV, δV, hL, hR, tensorToVec_vecToTensor, flattenCtx_unflattenCtx] using hinner
}

/-- A node built by `ofFn` has the given forward map. -/
@[simp] theorem forwardVec_ofFn {Γ : List Shape} {τ : Shape}
    (f) (jvp) (vjp) (h) :
    (Node.forwardVec (Γ := Γ) (τ := τ) (ofFn (Γ := Γ) (τ := τ) f jvp vjp h)) = f := by
  funext xV
  simp [Node.forwardVec, ofFn]

/-- A node built by `ofFn` has the given forward-mode derivative. -/
@[simp] theorem jvpVec_ofFn {Γ : List Shape} {τ : Shape}
    (f) (jvp) (vjp) (h) :
    (Node.jvpVec (Γ := Γ) (τ := τ) (ofFn (Γ := Γ) (τ := τ) f jvp vjp h)) = jvp := by
  funext xV dxV
  simp [Node.jvpVec, ofFn]

/-- A node built by `ofFn` has the given reverse-mode derivative. Together the three projection
lemmas mean a caller never has to unfold `ofFn`, only supply the soundness argument `h` once. -/
@[simp] theorem vjpVec_ofFn {Γ : List Shape} {τ : Shape}
    (f) (jvp) (vjp) (h) :
    (Node.vjpVec (Γ := Γ) (τ := τ) (ofFn (Γ := Γ) (τ := τ) f jvp vjp h)) = vjp := by
  funext xV δV
  simp [Node.vjpVec, ofFn]

end Node

-- ---------------------------------------------------------------------------
-- Turning `OpSpecFDerivCorrect` into tape nodes at a context index
-- ---------------------------------------------------------------------------

namespace OpSpecFDerivCorrect

open scoped BigOperators

/--
`OpSpecFDerivCorrect` instance for a linear layer.

This is the analytic correctness lemma behind the tape node constructors: it identifies the JVP
with the Fréchet derivative (a matrix multiplication) for `linearSpec`.

PyTorch analogue: the `torch.nn.Linear` forward map is affine, so its derivative is constant.
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
            getScalarE
              ((linearCorrect (inDim := inDim) (outDim := outDim) m).op.forward (ofFnE xV)))
          =
        affine (inDim := inDim) (outDim := outDim)
          (tensorToMatrix (m := outDim) (n := inDim) m.weights) (getScalarE m.bias) := by
      funext xV
      simpa [linearCorrect, Spec.linearOp] using
        (getScalarE_linear_spec (inDim := inDim) (outDim := outDim) m (x := ofFnE xV))
    have h :=
      hasFDerivAt_affine (inDim := inDim) (outDim := outDim)
        (W := tensorToMatrix (m := outDim) (n := inDim) m.weights) (b := getScalarE m.bias)
        (x := xV)
    -- rewrite the goal function from `affine` to the `OpSpec` forward
    exact (hAffine.symm ▸ h)
  jvp_eq := by
    intro xV dxV
    -- The JVP is the linear part of `linear_spec` (bias drops out), so this is just
    -- `matCLM` applied to `dxV`.
    simpa [linearCorrect, Spec.linearOp, OpSpecCorrect.jvp, getScalarE_ofFnE] using
      (getScalarE_mat_vec_mul_spec (m := outDim) (n := inDim) m.weights (ofFnE dxV))
}

end OpSpecFDerivCorrect

end

end Autograd
end Proofs
