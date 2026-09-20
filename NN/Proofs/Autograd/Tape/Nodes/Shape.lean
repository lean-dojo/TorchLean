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

open Spec TorchLean
open TorchLean TorchLean.Tensor

noncomputable section

open scoped BigOperators

-- Shape-changing tape nodes and their saved-tensor bookkeeping.

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
`reshape` is linear: on vectors it is just a type cast along `Spec.Shape.size` equality.
We implement it as a `Node` to keep the DAG theorem applicable.
-/

/--
`reshape` node: reinterpret the same underlying coordinates as a different shape.

This is only definable when `Spec.Shape.size s₁ = Spec.Shape.size s₂`; at the vector level it is
a cast.

PyTorch analogue: `view`/`reshape` operations that do not change the total number of elements.
https://pytorch.org/docs/stable/tensor_view.html
-/
def reshape {Γ : List Shape} {s₁ s₂ : Shape}
    (idx : Idx Γ s₁) (h : Spec.Shape.size s₁ = Spec.Shape.size s₂) : Node Γ s₂ :=
  Node.ofFn (Γ := Γ) (τ := s₂)
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
    (idx : Idx Γ s₁) (h : Spec.Shape.size s₁ = Spec.Shape.size s₂) :
    NodeFDerivCorrect (reshape (Γ := Γ) (s₁ := s₁) (s₂ := s₂) idx h) := by
  classical
  let Rlin : Vec (Spec.Shape.size s₁) →L[ℝ] Vec (Spec.Shape.size s₂) := Graph.castCLM (h := h)
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
      simp [reshape, Node.forwardVec_ofFn, Rlin, ContinuousLinearMap.comp_apply,
        CtxVec.getCLM_apply, Graph.castCLM]
    exact hCLM.congr_of_eventuallyEq hfun.eventuallyEq
  · intro _xV dxV
    simp [reshape, Node.jvpVec_ofFn, Rlin, ContinuousLinearMap.comp_apply, CtxVec.getCLM_apply,
      Graph.castCLM]

/-!
`flatten` is a specialization of `reshape` to the canonical vector shape
`(.dim (Spec.Shape.size s) .scalar)`.
-/

/--
`flatten` node: specialization of `reshape` to the canonical vector shape `(.dim (Spec.Shape.size s)
  .scalar)`.

PyTorch analogue: `flatten` when applied to a contiguous tensor.
https://pytorch.org/docs/stable/generated/torch.flatten.html
-/
def flatten {Γ : List Shape} {s : Shape} (idx : Idx Γ s) :
    Node Γ (.dim (Spec.Shape.size s) .scalar) :=
  reshape (Γ := Γ) (s₁ := s) (s₂ := .dim (Spec.Shape.size s) .scalar) idx
    (by simp [Spec.Shape.size])

/-- `NodeFDerivCorrect` for `flatten`. -/
def flattenFderiv {Γ : List Shape} {s : Shape} (idx : Idx Γ s) :
    NodeFDerivCorrect (flatten (Γ := Γ) (s := s) idx) :=
  reshapeFderiv (Γ := Γ) (s₁ := s) (s₂ := .dim (Spec.Shape.size s) .scalar) idx
    (by simp [Spec.Shape.size])

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

/-- Coordinate equivalence induced by exchanging two adjacent blocks. -/
public def swapAdjacentEquiv (outer inner tail : Nat) :
    Fin (outer * (inner * tail)) ≃ Fin (inner * (outer * tail)) :=
  let e_m_nk : (Fin outer × Fin (inner * tail)) ≃ Fin (outer * (inner * tail)) :=
    finProdFinEquiv
  let e_n_k : (Fin inner × Fin tail) ≃ Fin (inner * tail) := finProdFinEquiv
  let e_m_k : (Fin outer × Fin tail) ≃ Fin (outer * tail) := finProdFinEquiv
  let e_n_mk : (Fin inner × Fin (outer * tail)) ≃ Fin (inner * (outer * tail)) :=
    finProdFinEquiv
  e_m_nk.symm
    |>.trans (Equiv.prodCongrRight (fun _ : Fin outer => e_n_k.symm))
    |>.trans (Equiv.prodAssoc (Fin outer) (Fin inner) (Fin tail)).symm
    |>.trans
      (Equiv.prodCongrLeft (fun _ : Fin tail => Equiv.prodComm (Fin outer) (Fin inner)))
    |>.trans (Equiv.prodAssoc (Fin inner) (Fin outer) (Fin tail))
    |>.trans (Equiv.prodCongrRight (fun _ : Fin inner => e_m_k))
    |>.trans e_n_mk

/--
Lift a coordinate equivalence pointwise under a leading axis.
-/
public def mapOuterEquiv (leading : Nat) {source target : Nat}
    (e : Fin source ≃ Fin target) : Fin (leading * source) ≃ Fin (leading * target) :=
  let sourceProd : (Fin leading × Fin source) ≃ Fin (leading * source) := finProdFinEquiv
  let targetProd : (Fin leading × Fin target) ≃ Fin (leading * target) := finProdFinEquiv
  sourceProd.symm
    |>.trans (Equiv.prodCongrRight (fun _ : Fin leading => e))
    |>.trans targetProd

/--
Reindex a tensor node by any coordinate equivalence. This is the proof-level primitive behind
arbitrary axis permutations; its JVP uses the same permutation and its VJP uses the inverse.
-/
def reindex {Γ : List Shape} {source target : Shape}
    (idx : Idx Γ source)
    (e : Fin (Spec.Shape.size source) ≃ Fin (Spec.Shape.size target)) :
    Node Γ target :=
  Node.ofFn (Γ := Γ) (τ := target)
    (f := fun xV =>
      reindexVec (n := Spec.Shape.size source) (m := Spec.Shape.size target) e
        (CtxVec.get (Γ := Γ) (s := source) idx xV))
    (jvp := fun _xV dxV =>
      reindexVec (n := Spec.Shape.size source) (m := Spec.Shape.size target) e
        (CtxVec.get (Γ := Γ) (s := source) idx dxV))
    (vjp := fun _xV δV =>
      CtxVec.single (Γ := Γ) (s := source) idx
        (reindexVec (n := Spec.Shape.size target) (m := Spec.Shape.size source) e.symm δV))
    (correct_inner := by
      intro _xV dxV δV
      classical
      have hctx :=
        CtxVec.inner_get_single (Γ := Γ) (s := source) idx dxV
          (reindexVec (n := Spec.Shape.size target) (m := Spec.Shape.size source) e.symm δV)
      have hperm :=
        inner_reindex_left (e := e)
          (x := CtxVec.get (Γ := Γ) (s := source) idx dxV)
          (y := δV)
      exact hperm.trans hctx.symm)

/-- `NodeFDerivCorrect` for an arbitrary coordinate reindexing. -/
def reindexFDeriv {Γ : List Shape} {source target : Shape}
    (idx : Idx Γ source)
    (e : Fin (Spec.Shape.size source) ≃ Fin (Spec.Shape.size target)) :
    NodeFDerivCorrect (reindex (Γ := Γ) (source := source) (target := target) idx e) := by
  classical
  let P : Vec (Spec.Shape.size source) →L[ℝ] Vec (Spec.Shape.size target) :=
    reindexLin (n := Spec.Shape.size source) (m := Spec.Shape.size target) e
  let D : CtxVec Γ →L[ℝ] Vec (Spec.Shape.size target) :=
    P.comp (CtxVec.getCLM (Γ := Γ) (s := source) idx)
  refine
    { deriv := fun _ => D
      hasFDerivAt := ?_
      jvp_eq := ?_ }
  · intro xV
    have hD : HasFDerivAt (fun x : CtxVec Γ => D x) D xV := D.hasFDerivAt (x := xV)
    have hEq :
        (Node.forwardVec (Γ := Γ) (τ := target)
            (reindex (Γ := Γ) (source := source) (target := target) idx e))
          =
        fun x : CtxVec Γ => D x := by
      funext x
      simp only [reindex, Node.forwardVec_ofFn, D, P,
        ContinuousLinearMap.comp_apply, CtxVec.getCLM_apply]
      exact (reindexLin_apply e _).symm
    exact hD.congr_of_eventuallyEq hEq.eventuallyEq
  · intro xV dxV
    simp only [reindex, Node.jvpVec_ofFn, D, P,
      ContinuousLinearMap.comp_apply, CtxVec.getCLM_apply]
    exact (reindexLin_apply e _).symm

end ShapeOps

end TapeNodes

end
end Autograd
end Proofs
