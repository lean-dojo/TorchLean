/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Tape.Nodes.Context

/-!
# Shape

Additional analytic (`HasFDerivAt`) tape nodes for **shape permutations**.

These nodes are linear/isometric and are useful for models that do explicit reshaping and
dimension permutations (e.g. Multi-Head Attention head splitting/combining).
-/

@[expose] public section


namespace Proofs
namespace Autograd

open Spec
open Tensor

noncomputable section

open scoped BigOperators

-- ---------------------------------------------------------------------------
-- Nodes
-- ---------------------------------------------------------------------------

namespace TapeNodes

namespace ShapeOps

/-- Move `castVec` across the left argument of an inner product. -/
public lemma inner_castVec_left {n m : Nat} (h : n = m) (x : Vec n) (y : Vec m) :
    inner ℝ (castVec h x) y = inner ℝ x (castVec h.symm y) := by
  -- Insert the cancelling cast on `y` and use `inner_castVec_castVec`.
  have hy : castVec h (castVec h.symm y) = y := by
    simp
  calc
    inner ℝ (castVec h x) y
        = inner ℝ (castVec h x) (castVec h (castVec h.symm y)) := by simp [hy]
    _ = inner ℝ x (castVec h.symm y) := by
          simpa using (inner_castVec_castVec (h := h) (x := x) (y := castVec h.symm y))

/-- `castVec` is proof-irrelevant in its equality argument. -/
public lemma castVec_proof_irrel {n m : Nat} (h₁ h₂ : n = m) (v : Vec n) :
    castVec h₁ v = castVec h₂ v := by
  have : h₁ = h₂ := Subsingleton.elim _ _
  cases this
  rfl

/-!
`reshape` is linear: on vectors it is just a type cast along `Shape.size` equality.
We implement it as a `Node` to keep the DAG theorem applicable.
-/

/--
`reshape` node: reinterpret the same underlying coordinates as a different shape.

This is only definable when `Shape.size s₁ = Shape.size s₂`; at the vector level it is a cast.

PyTorch analogue: `view`/`reshape` operations that do not change the total number of elements.
https://pytorch.org/docs/stable/tensor_view.html
-/
def reshape {Γ : List Shape} {s₁ s₂ : Shape}
    (idx : Idx Γ s₁) (h : Shape.size s₁ = Shape.size s₂) : Node Γ s₂ :=
  Node.ofVec (Γ := Γ) (τ := s₂)
    (f := fun xV => castVec h (CtxVec.get (Γ := Γ) (s := s₁) idx xV))
    (jvp := fun _xV dxV => castVec h (CtxVec.get (Γ := Γ) (s := s₁) idx dxV))
    (vjp := fun _xV δV => CtxVec.single (Γ := Γ) (s := s₁) idx (castVec h.symm δV))
    (correct_inner := by
      intro _xV dxV δV
      classical
      -- Reduce to the `get/single` adjointness plus the cast-isometry lemma.
      have hget := CtxVec.inner_get_single (Γ := Γ) (s := s₁) idx dxV (castVec h.symm δV)
      -- Move the cast across the left inner product.
      simpa [inner_castVec_left (h := h)] using hget.symm)

/-- `NodeFDerivCorrect` for `reshape` (it is linear/isometric). -/
def reshapeFderiv {Γ : List Shape} {s₁ s₂ : Shape}
    (idx : Idx Γ s₁) (h : Shape.size s₁ = Shape.size s₂) :
    NodeFDerivCorrect (reshape (Γ := Γ) (s₁ := s₁) (s₂ := s₂) idx h) := by
  classical
  let Rlin : Vec (Shape.size s₁) →L[ℝ] Vec (Shape.size s₂) := Graph.castCLM (h := h)
  refine
    { deriv := fun _xV => Rlin.comp (CtxVec.getCLM (Γ := Γ) (s := s₁) idx)
      hasFDerivAt := ?_
      jvp_eq := ?_ }
  · intro xV
    have hCLM := (Rlin.comp (CtxVec.getCLM (Γ := Γ) (s := s₁) idx)).hasFDerivAt (x := xV)
    have hfun :
        (Node.forwardVec (Γ := Γ) (τ := s₂) (reshape (Γ := Γ) (s₁ := s₁) (s₂ := s₂) idx h)) =
          fun x : CtxVec Γ => (Rlin.comp (CtxVec.getCLM (Γ := Γ) (s := s₁) idx)) x := by
      funext x
      simp [reshape, Node.forwardVec_ofVec, Rlin, ContinuousLinearMap.comp_apply,
        CtxVec.getCLM_apply, Graph.castCLM]
    exact hCLM.congr_of_eventuallyEq hfun.eventuallyEq
  · intro _xV dxV
    simp [reshape, Node.jvpVec_ofVec, Rlin, ContinuousLinearMap.comp_apply, CtxVec.getCLM_apply,
      Graph.castCLM]

/-!
`flatten` is a specialization of `reshape` to the canonical vector shape
`(.dim (Shape.size s) .scalar)`.
-/

/--
`flatten` node: specialization of `reshape` to the canonical vector shape `(.dim (Shape.size s)
  .scalar)`.

PyTorch analogue: `flatten` when applied to a contiguous tensor.
https://pytorch.org/docs/stable/generated/torch.flatten.html
-/
def flatten {Γ : List Shape} {s : Shape} (idx : Idx Γ s) :
    Node Γ (.dim (Shape.size s) .scalar) :=
  reshape (Γ := Γ) (s₁ := s) (s₂ := .dim (Shape.size s) .scalar) idx (by simp [Shape.size])

/-- `NodeFDerivCorrect` for `flatten`. -/
def flattenFderiv {Γ : List Shape} {s : Shape} (idx : Idx Γ s) :
    NodeFDerivCorrect (flatten (Γ := Γ) (s := s) idx) :=
  reshapeFderiv (Γ := Γ) (s₁ := s) (s₂ := .dim (Shape.size s) .scalar) idx (by simp [Shape.size])

-- ---------------------------------------------------------------------------
-- Generic coordinate reindexing (`Vec n` ↔ `Vec m`) via a `Fin` equivalence
-- ---------------------------------------------------------------------------

/-- Reindex a vector along a `Fin` equivalence (coordinate permutation/renaming). -/
public def reindexVec {n m : Nat} (e : Fin n ≃ Fin m) : Vec n → Vec m :=
  fun v => vecOfFun (n := m) fun i => v (e.symm i)

/-- The linear map induced by `reindexVec`. -/
public def reindexLin {n m : Nat} (e : Fin n ≃ Fin m) : Vec n →L[ℝ] Vec m := by
  classical
  let fLin : Vec n →ₗ[ℝ] Vec m :=
    { toFun := reindexVec (n := n) (m := m) e
      map_add' := by
        intro x y
        ext i
        simp [reindexVec]
      map_smul' := by
        intro r x
        ext i
        simp [reindexVec, smul_eq_mul] }
  refine ⟨fLin, ?_⟩
  exact LinearMap.continuous_of_finiteDimensional (f := fLin)

@[simp] public lemma reindexLin_apply {n m : Nat} (e : Fin n ≃ Fin m) (v : Vec n) :
    reindexLin (n := n) (m := m) e v = reindexVec (n := n) (m := m) e v := by
  rfl

/-- Move `reindexVec` across the left argument of an inner product. -/
public lemma inner_reindex_left {n m : Nat} (e : Fin n ≃ Fin m) (x : Vec n) (y : Vec m) :
    inner ℝ (reindexVec (n := n) (m := m) e x) y = inner ℝ x (reindexVec (n := m) (m := n) e.symm y)
      := by
  classical
  have hs :
      (∑ i : Fin m, x (e.symm i) * y i) = ∑ j : Fin n, x j * y (e j) := by
    refine (Fintype.sum_equiv (e := e.symm)
      (f := fun i : Fin m => x (e.symm i) * y i)
      (g := fun j : Fin n => x j * y (e j)) ?_)
    intro i
    have hy : y (e (e.symm i)) = y i := by
      simp
    -- `f i = g (e.symm i)`
    simp [hy]
  calc
    inner ℝ (reindexVec (n := n) (m := m) e x) y
        = ∑ i : Fin m, x (e.symm i) * y i := by
            simp [reindexVec, inner_eq_sum_mul]
    _ = ∑ j : Fin n, x j * y (e j) := hs
    _ = inner ℝ x (reindexVec (n := m) (m := n) e.symm y) := by
            simp [reindexVec, inner_eq_sum_mul]

-- ---------------------------------------------------------------------------
-- 3D swap of the first two axes: `.dim m (.dim n rest)` ↦ `.dim n (.dim m rest)`
-- ---------------------------------------------------------------------------

/-- Underlying coordinate permutation for swapping the first two axes of a 3D tensor. -/
public def swapFirstTwoEquiv (m n k : Nat) : Fin (m * (n * k)) ≃ Fin (n * (m * k)) :=
  let e_m_nk : (Fin m × Fin (n * k)) ≃ Fin (m * (n * k)) := finProdFinEquiv
  let e_n_k : (Fin n × Fin k) ≃ Fin (n * k) := finProdFinEquiv
  let e_m_k : (Fin m × Fin k) ≃ Fin (m * k) := finProdFinEquiv
  let e_n_mk : (Fin n × Fin (m * k)) ≃ Fin (n * (m * k)) := finProdFinEquiv
  e_m_nk.symm
    |>.trans (Equiv.prodCongrRight (fun _ : Fin m => e_n_k.symm))
    |>.trans (Equiv.prodAssoc (Fin m) (Fin n) (Fin k)).symm
    |>.trans (Equiv.prodCongrLeft (fun _ : Fin k => Equiv.prodComm (Fin m) (Fin n)))
    |>.trans (Equiv.prodAssoc (Fin n) (Fin m) (Fin k))
    |>.trans (Equiv.prodCongrRight (fun _ : Fin n => e_m_k))
    |>.trans e_n_mk

/--
Swap the first two axes of a 3D tensor shape: `.dim m (.dim n rest) ↦ .dim n (.dim m rest)`.

This is implemented as a coordinate permutation (a linear isometry).

PyTorch analogue: `transpose(0, 1)` on a 3D tensor.
https://pytorch.org/docs/stable/generated/torch.transpose.html
-/
def swapFirstTwo3d {Γ : List Shape} {m n : Nat} {rest : Shape}
    (idx : Idx Γ (.dim m (.dim n rest))) :
    Node Γ (.dim n (.dim m rest)) :=
  let sIn : Shape := .dim m (.dim n rest)
  let sOut : Shape := .dim n (.dim m rest)
  let k : Nat := Shape.size rest
  let e : Fin (Shape.size sIn) ≃ Fin (Shape.size sOut) := swapFirstTwoEquiv m n k
  Node.ofVec (Γ := Γ) (τ := sOut)
    (f := fun xV =>
      reindexVec (n := Shape.size sIn) (m := Shape.size sOut) e
        (CtxVec.get (Γ := Γ) (s := sIn) idx xV))
    (jvp := fun _xV dxV =>
      reindexVec (n := Shape.size sIn) (m := Shape.size sOut) e
        (CtxVec.get (Γ := Γ) (s := sIn) idx dxV))
    (vjp := fun _xV δV =>
      CtxVec.single (Γ := Γ) (s := sIn) idx
        (reindexVec (n := Shape.size sOut) (m := Shape.size sIn) e.symm δV))
    (correct_inner := by
      intro _xV dxV δV
      classical
      have hctx :=
        CtxVec.inner_get_single (Γ := Γ) (s := sIn) idx dxV
          (reindexVec (n := Shape.size sOut) (m := Shape.size sIn) e.symm δV)
      have hperm :=
        inner_reindex_left (e := e)
          (x := CtxVec.get (Γ := Γ) (s := sIn) idx dxV)
          (y := δV)
      exact hperm.trans hctx.symm)

/-- `NodeFDerivCorrect` for `swap_first_two3d` (linear coordinate permutation). -/
def swapFirstTwo3dFderiv {Γ : List Shape} {m n : Nat} {rest : Shape}
    (idx : Idx Γ (.dim m (.dim n rest))) :
    NodeFDerivCorrect (swapFirstTwo3d (Γ := Γ) (m := m) (n := n) (rest := rest) idx) := by
  classical
  let sIn : Shape := .dim m (.dim n rest)
  let sOut : Shape := .dim n (.dim m rest)
  let k : Nat := Shape.size rest
  let e : Fin (Shape.size sIn) ≃ Fin (Shape.size sOut) := swapFirstTwoEquiv m n k
  let P : Vec (Shape.size sIn) →L[ℝ] Vec (Shape.size sOut) :=
    reindexLin (n := Shape.size sIn) (m := Shape.size sOut) e
  let D : CtxVec Γ →L[ℝ] Vec (Shape.size sOut) :=
    P.comp (CtxVec.getCLM (Γ := Γ) (s := sIn) idx)
  refine
    { deriv := fun _ => D
      hasFDerivAt := ?_
      jvp_eq := ?_ }
  · intro xV
    have hD : HasFDerivAt (fun x : CtxVec Γ => D x) D xV := D.hasFDerivAt (x := xV)
    have hEq :
        (Node.forwardVec (Γ := Γ) (τ := sOut)
            (swapFirstTwo3d (Γ := Γ) (m := m) (n := n) (rest := rest) idx))
          =
        fun x : CtxVec Γ => D x := by
      funext x
      ext i
      simp [swapFirstTwo3d, D, P, k, e, Node.forwardVec_ofVec, CtxVec.getCLM_apply,
        ContinuousLinearMap.comp_apply]
    exact hD.congr_of_eventuallyEq hEq.eventuallyEq
  · intro xV dxV
    ext i
    -- Unfold the RHS into a pointwise application of `reindexLin`, then rewrite to `reindexVec`.
    simp [swapFirstTwo3d, D, P, k, e, Node.jvpVec_ofVec]

-- ---------------------------------------------------------------------------
-- 3D transpose of the last two axes: `.dim a (.dim b (.dim c .scalar)) ↦ .dim a (.dim c (.dim b
-- .scalar))`
-- ---------------------------------------------------------------------------

/-- Underlying coordinate permutation for transposing the last two axes of a 3D tensor. -/
public def transposeLastTwoEquiv (a b c : Nat) :
    Fin (a * (b * (c * 1))) ≃ Fin (a * (c * (b * 1))) :=
  -- Work with the definitional sizes coming from `Shape.size`: `c` contributes a `* 1` from the
  -- trailing `.scalar`.
  let swapBC : Fin (b * (c * 1)) ≃ Fin (c * (b * 1)) :=
    let e_b_c1 : (Fin b × Fin (c * 1)) ≃ Fin (b * (c * 1)) := finProdFinEquiv
    let e_c_1 : (Fin c × Fin 1) ≃ Fin (c * 1) := finProdFinEquiv
    let e_b_1 : (Fin b × Fin 1) ≃ Fin (b * 1) := finProdFinEquiv
    let e_c_b1 : (Fin c × Fin (b * 1)) ≃ Fin (c * (b * 1)) := finProdFinEquiv
    e_b_c1.symm
      |>.trans (Equiv.prodCongrRight (fun _ : Fin b => e_c_1.symm))
      |>.trans (Equiv.prodAssoc (Fin b) (Fin c) (Fin 1)).symm
      |>.trans (Equiv.prodCongr (Equiv.prodComm (Fin b) (Fin c)) (Equiv.refl (Fin 1)))
      |>.trans (Equiv.prodAssoc (Fin c) (Fin b) (Fin 1))
      |>.trans (Equiv.prodCongrRight (fun _ : Fin c => e_b_1))
      |>.trans e_c_b1
  let e_a_bc1 : (Fin a × Fin (b * (c * 1))) ≃ Fin (a * (b * (c * 1))) := finProdFinEquiv
  let e_a_cb1 : (Fin a × Fin (c * (b * 1))) ≃ Fin (a * (c * (b * 1))) := finProdFinEquiv
  e_a_bc1.symm
    |>.trans (Equiv.prodCongrRight (fun _ : Fin a => swapBC))
    |>.trans e_a_cb1

/--
Transpose the last two axes of a 3D tensor: `.dim a (.dim b (.dim c .scalar)) ↦ .dim a (.dim c (.dim
  b .scalar))`.

This is another coordinate permutation used in attention (switching `K` to `Kᵀ` while keeping
  head/batch axes).
-/
def transpose3dLastTwo {Γ : List Shape} {a b c : Nat}
    (idx : Idx Γ (.dim a (.dim b (.dim c .scalar)))) :
    Node Γ (.dim a (.dim c (.dim b .scalar))) :=
  let sIn : Shape := .dim a (.dim b (.dim c .scalar))
  let sOut : Shape := .dim a (.dim c (.dim b .scalar))
  let e : Fin (Shape.size sIn) ≃ Fin (Shape.size sOut) := transposeLastTwoEquiv a b c
  Node.ofVec (Γ := Γ) (τ := sOut)
    (f := fun xV =>
      reindexVec (n := Shape.size sIn) (m := Shape.size sOut) e
        (CtxVec.get (Γ := Γ) (s := sIn) idx xV))
    (jvp := fun _xV dxV =>
      reindexVec (n := Shape.size sIn) (m := Shape.size sOut) e
        (CtxVec.get (Γ := Γ) (s := sIn) idx dxV))
    (vjp := fun _xV δV =>
      CtxVec.single (Γ := Γ) (s := sIn) idx
        (reindexVec (n := Shape.size sOut) (m := Shape.size sIn) e.symm δV))
    (correct_inner := by
      intro _xV dxV δV
      classical
      have hctx :=
        CtxVec.inner_get_single (Γ := Γ) (s := sIn) idx dxV
          (reindexVec (n := Shape.size sOut) (m := Shape.size sIn) e.symm δV)
      -- `inner_reindex_left` is exactly the isometry/adjointness property for the permutation.
      have hperm :=
        inner_reindex_left (e := e)
          (x := CtxVec.get (Γ := Γ) (s := sIn) idx dxV)
          (y := δV)
      -- Combine with `get/single` adjointness in the context.
      exact hperm.trans hctx.symm)

/-- `NodeFDerivCorrect` for `transpose3d_last_two` (linear coordinate permutation). -/
def transpose3dLastTwoFderiv {Γ : List Shape} {a b c : Nat}
    (idx : Idx Γ (.dim a (.dim b (.dim c .scalar)))) :
    NodeFDerivCorrect (transpose3dLastTwo (Γ := Γ) (a := a) (b := b) (c := c) idx) := by
  classical
  let sIn : Shape := .dim a (.dim b (.dim c .scalar))
  let sOut : Shape := .dim a (.dim c (.dim b .scalar))
  let e : Fin (Shape.size sIn) ≃ Fin (Shape.size sOut) := transposeLastTwoEquiv a b c
  let P : Vec (Shape.size sIn) →L[ℝ] Vec (Shape.size sOut) :=
    reindexLin (n := Shape.size sIn) (m := Shape.size sOut) e
  let D : CtxVec Γ →L[ℝ] Vec (Shape.size sOut) :=
    P.comp (CtxVec.getCLM (Γ := Γ) (s := sIn) idx)
  refine
    { deriv := fun _ => D
      hasFDerivAt := ?_
      jvp_eq := ?_ }
  · intro xV
    have hD : HasFDerivAt (fun x : CtxVec Γ => D x) D xV := D.hasFDerivAt (x := xV)
    have hEq :
        (Node.forwardVec (Γ := Γ) (τ := sOut)
            (transpose3dLastTwo (Γ := Γ) (a := a) (b := b) (c := c) idx))
          =
        fun x : CtxVec Γ => D x := by
      funext x
      simp [transpose3dLastTwo, D, P, Node.forwardVec_ofVec, CtxVec.getCLM_apply,
        ContinuousLinearMap.comp_apply, e]
    exact hD.congr_of_eventuallyEq hEq.eventuallyEq
  · intro xV dxV
    ext i
    simp [transpose3dLastTwo, D, P, Node.jvpVec_ofVec]
    simp [e]

end ShapeOps

end TapeNodes

end
end Autograd
end Proofs
