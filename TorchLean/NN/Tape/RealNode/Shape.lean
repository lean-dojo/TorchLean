/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Tape.RealNode.Context

/-!
# Shape

Tape nodes that move coordinates without changing their values. `reshape` casts a tensor to
another shape of the same size, and `reindex` permutes coordinates along any `Fin` equivalence.
Both are linear isometries, so each gets a global `RealNode.FDerivCorrect` certificate whose
derivative is the same linear map at every point. Multi-head attention splits and merges heads
with `reindex` and the block permutations `swapAdjacentEquiv` and `mapOuterEquiv`. The lemmas
`toVec_reshape` and `toVec_permute_adjacentSwap_zero` carry `Tensor.toVec` through
`Tensor.reshape`, as a size cast, and through the exchange of the two leading axes, as the
reindexing by `swapAdjacentEquiv`.
-/

@[expose] public section

open TorchLean.Shape (size)

namespace TorchLean

open TorchLean.Tensor

noncomputable section

open scoped BigOperators

/-- Transporting a vector along a length equality reindexes its Euclidean view. -/
theorem getScalarE_cast {a b : Nat} (h : a = b) (v : Tensor ℝ [a]) (k : Fin b) :
    (getScalarE (h ▸ v)) k = getScalarE v (Fin.cast h.symm k) := by
  subst h
  rfl

/-- `toVec` of a reshaped tensor is a reindexing of `toVec`. -/
theorem toVec_reshape_apply {s₁ s₂ : Shape} (t : Tensor ℝ s₁) (h : s₁.size = s₂.size)
    (k : Fin (Shape.size s₂)) :
    toVec (t := Tensor.reshape t _ h) k = toVec (t := t) (Fin.cast h.symm k) := by
  simp only [toVec, reshape_vector_reshape]
  exact getScalarE_cast h _ k

/-- Vectorizing a reshaped tensor casts the vectorization along the size equality. -/
theorem toVec_reshape {s₁ s₂ : Shape} (t : Tensor ℝ s₁) (h : s₁.size = s₂.size) :
    toVec (Tensor.reshape t s₂ h) =
      LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ (finCongr h) (toVec t) := by
  ext k
  rw [toVec_reshape_apply, LinearIsometryEquiv.piLpCongrLeft_apply, Equiv.piCongrLeft'_apply,
    finCongr_symm_apply]

/-- The vectorization of a tensor with a leading axis lists the vectorized slices in order. -/
theorem toVec_finProdFinEquiv {n : Nat} {s : Shape} (T : Tensor ℝ (.dim n s)) (i : Fin n)
    (k : Fin (Shape.size s)) :
    toVec T (finProdFinEquiv (i, k)) = toVec (Tensor.unstack T i) k := by
  conv_lhs => rw [← Tensor.dim_unstack T]
  exact toVec_dim_apply _ (i, k)

-- Shape-changing tape nodes and their saved-tensor bookkeeping.

namespace Tape.RealNode

/-!
`reshape` is linear: on flattened vectors it is a cast along the equality of `Shape.size`.
We implement it as a `LawfulNode` to keep the DAG theorem applicable.
-/

/--
`reshape` node: reinterpret the same underlying coordinates as a different shape.

The node takes a proof of `Shape.size s₁ = Shape.size s₂`, and on flattened vectors it
is the cast along that equality.

PyTorch analogue: `view`/`reshape` operations that do not change the total number of elements.
https://pytorch.org/docs/stable/tensor_view.html
-/
def reshape {Γ : List Shape} {s₁ s₂ : Shape}
    (idx : Idx Γ s₁) (h : Shape.size s₁ = Shape.size s₂) :
    LawfulNode (α := ℝ) Unit Γ s₂ :=
  LawfulNode.ofFn (Γ := Γ) (τ := s₂)
    (f := fun xV =>
      LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ (finCongr h) (CtxVec.get (Γ := Γ) (s := s₁) idx xV))
    (jvp := fun _xV dxV =>
      LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ (finCongr h) (CtxVec.get (Γ := Γ) (s := s₁) idx dxV))
    (vjp := fun _xV δV =>
      CtxVec.single (Γ := Γ) (s := s₁) idx
        (LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ (finCongr h.symm) δV))
    (correct_inner := by
      intro _xV dxV δV
      classical
      -- Reduce to the `get/single` adjointness plus the cast-isometry lemma.
      have hget := CtxVec.inner_get_single (Γ := Γ) (s := s₁) idx dxV
        (LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ (finCongr h.symm) δV)
      -- Move the cast across the left inner product.
      rw [LinearIsometryEquiv.inner_map_eq_flip, LinearIsometryEquiv.piLpCongrLeft_symm,
        finCongr_symm]
      exact hget.symm)

/-- `FDerivCorrect` for `reshape`, whose derivative at every point is the same cast. -/
def reshapeFDeriv {Γ : List Shape} {s₁ s₂ : Shape}
    (idx : Idx Γ s₁) (h : Shape.size s₁ = Shape.size s₂) :
    FDerivCorrect (reshape (Γ := Γ) (s₁ := s₁) (s₂ := s₂) idx h).toNodeData () := by
  classical
  let Rlin : Vec (Shape.size s₁) →L[ℝ] Vec (Shape.size s₂) :=
    (LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ
      (finCongr h)).toContinuousLinearEquiv.toContinuousLinearMap
  refine
    { deriv := fun _xV => Rlin.comp (CtxVec.getCLM (Γ := Γ) (s := s₁) idx)
      hasFDerivAt := ?_
      jvp_eq := ?_ }
  · intro xV
    have hCLM := (Rlin.comp (CtxVec.getCLM (Γ := Γ) (s := s₁) idx)).hasFDerivAt (x := xV)
    have hfun :
        (NodeData.forwardVec (Γ := Γ) (τ := s₂)
          (reshape (Γ := Γ) (s₁ := s₁) (s₂ := s₂) idx h).toNodeData ()) =
          fun x : CtxVec Γ => (Rlin.comp (CtxVec.getCLM (Γ := Γ) (s := s₁) idx)) x := by
      funext x
      simp [reshape, LawfulNode.forwardVec_ofFn, Rlin, ContinuousLinearMap.comp_apply,
        CtxVec.getCLM_apply]
    exact hCLM.congr_of_eventuallyEq hfun.eventuallyEq
  · intro _xV dxV
    simp [reshape, LawfulNode.jvpVec_ofFn, Rlin, ContinuousLinearMap.comp_apply,
      CtxVec.getCLM_apply]

-- ---------------------------------------------------------------------------
-- Generic coordinate reindexing (`Vec n` ↔ `Vec m`) via a `Fin` equivalence
-- ---------------------------------------------------------------------------

/-- Flat-index equivalence that swaps the first two axes of a row-major `[outer, inner, tail]`
layout, giving `[inner, outer, tail]`. -/
def swapAdjacentEquiv (outer inner tail : Nat) :
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

/-- `swapAdjacentEquiv` sends the row-major coordinate `(i, j, k)` to `(j, i, k)`. -/
@[simp] theorem swapAdjacentEquiv_apply (outer inner tail : Nat) (i : Fin outer) (j : Fin inner)
    (k : Fin tail) :
    swapAdjacentEquiv outer inner tail (finProdFinEquiv (i, finProdFinEquiv (j, k))) =
      finProdFinEquiv (j, finProdFinEquiv (i, k)) := by
  simp [swapAdjacentEquiv]

/-- Exchanging two adjacent blocks back undoes the exchange. -/
@[simp] theorem swapAdjacentEquiv_symm (outer inner tail : Nat) :
    (swapAdjacentEquiv outer inner tail).symm = swapAdjacentEquiv inner outer tail := by
  refine Equiv.ext fun x => ?_
  obtain ⟨⟨j, y⟩, rfl⟩ := (finProdFinEquiv (m := inner) (n := outer * tail)).surjective x
  obtain ⟨⟨i, k⟩, rfl⟩ := (finProdFinEquiv (m := outer) (n := tail)).surjective y
  rw [Equiv.symm_apply_eq, swapAdjacentEquiv_apply, swapAdjacentEquiv_apply]

/-- Exchanging the two leading axes of a tensor reindexes its vectorization by
`swapAdjacentEquiv`. -/
theorem toVec_permute_adjacentSwap_zero {a b : Nat} {s : Shape}
    (T : Tensor ℝ (.dim a (.dim b s)))
    (hout : Shape.dim b (.dim a s) =
      Shape.permute (.dim a (.dim b s)) (Shape.adjacentSwap (.dim a (.dim b s)) 0)) :
    toVec (permute T (Shape.adjacentSwap (.dim a (.dim b s)) 0) (.dim b (.dim a s)) hout) =
      LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ (swapAdjacentEquiv a b (Shape.size s))
        (toVec T) := by
  ext x
  obtain ⟨⟨j, y⟩, rfl⟩ :=
    (finProdFinEquiv (m := b) (n := Shape.size (Shape.dim a s))).surjective x
  obtain ⟨⟨i, k⟩, rfl⟩ := (finProdFinEquiv (m := a) (n := Shape.size s)).surjective y
  refine (toVec_finProdFinEquiv _ j _).trans
    ((toVec_finProdFinEquiv _ i k).trans ?_)
  show _ = toVec T ((swapAdjacentEquiv a b (Shape.size s)).symm _)
  rw [swapAdjacentEquiv_symm]
  refine Eq.trans ?_ (congrArg (WithLp.ofLp (toVec T))
    (swapAdjacentEquiv_apply b a (Shape.size s) j i k)).symm
  rw [permute_adjacentSwap_zero]
  simp only [Tensor.unstack_dim]
  exact ((toVec_finProdFinEquiv T i _).trans (toVec_finProdFinEquiv _ j k)).symm

/--
`mapOuterEquiv leading e` applies `e` to the trailing index of a `[leading, source]` layout and
leaves the leading index alone.
-/
def mapOuterEquiv (leading : Nat) {source target : Nat}
    (e : Fin source ≃ Fin target) : Fin (leading * source) ≃ Fin (leading * target) :=
  let sourceProd : (Fin leading × Fin source) ≃ Fin (leading * source) := finProdFinEquiv
  let targetProd : (Fin leading × Fin target) ≃ Fin (leading * target) := finProdFinEquiv
  sourceProd.symm
    |>.trans (Equiv.prodCongrRight (fun _ : Fin leading => e))
    |>.trans targetProd

/-- `mapOuterEquiv` keeps the leading coordinate and applies `e` to the rest. -/
@[simp] theorem mapOuterEquiv_apply (leading : Nat) {source target : Nat}
    (e : Fin source ≃ Fin target) (i : Fin leading) (j : Fin source) :
    mapOuterEquiv leading e (finProdFinEquiv (i, j)) = finProdFinEquiv (i, e j) := by
  simp [mapOuterEquiv]

/--
Reindex a tensor node by any coordinate equivalence `e`. Every axis permutation is a case of this
node, with JVP reindexing by `e` and VJP by `e.symm`.
-/
def reindex {Γ : List Shape} {source target : Shape}
    (idx : Idx Γ source)
    (e : Fin (Shape.size source) ≃ Fin (Shape.size target)) :
    LawfulNode (α := ℝ) Unit Γ target :=
  LawfulNode.ofFn (Γ := Γ) (τ := target)
    (f := fun xV =>
      LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ e
        (CtxVec.get (Γ := Γ) (s := source) idx xV))
    (jvp := fun _xV dxV =>
      LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ e
        (CtxVec.get (Γ := Γ) (s := source) idx dxV))
    (vjp := fun _xV δV =>
      CtxVec.single (Γ := Γ) (s := source) idx
        ((LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ e).symm δV))
    (correct_inner := by
      intro _xV dxV δV
      classical
      have hctx :=
        CtxVec.inner_get_single (Γ := Γ) (s := source) idx dxV
          ((LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ e).symm δV)
      have hperm :=
        LinearIsometryEquiv.inner_map_eq_flip (LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ e)
          (CtxVec.get (Γ := Γ) (s := source) idx dxV) δV
      exact hperm.trans hctx.symm)

/-- `FDerivCorrect` for an arbitrary coordinate reindexing. -/
def reindexFDeriv {Γ : List Shape} {source target : Shape}
    (idx : Idx Γ source)
    (e : Fin (Shape.size source) ≃ Fin (Shape.size target)) :
    FDerivCorrect
      (reindex (Γ := Γ) (source := source) (target := target) idx e).toNodeData () := by
  classical
  let P : Vec (Shape.size source) →L[ℝ] Vec (Shape.size target) :=
    (LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ e).toContinuousLinearEquiv.toContinuousLinearMap
  let D : CtxVec Γ →L[ℝ] Vec (Shape.size target) :=
    P.comp (CtxVec.getCLM (Γ := Γ) (s := source) idx)
  refine
    { deriv := fun _ => D
      hasFDerivAt := ?_
      jvp_eq := ?_ }
  · intro xV
    have hD : HasFDerivAt (fun x : CtxVec Γ => D x) D xV := D.hasFDerivAt (x := xV)
    have hEq :
        (NodeData.forwardVec (Γ := Γ) (τ := target)
            (reindex (Γ := Γ) (source := source) (target := target) idx e).toNodeData ())
          =
        fun x : CtxVec Γ => D x := by
      funext x
      simp only [reindex, LawfulNode.forwardVec_ofFn, D, P,
        ContinuousLinearMap.comp_apply, CtxVec.getCLM_apply]
      exact rfl
    exact hD.congr_of_eventuallyEq hEq.eventuallyEq
  · intro xV dxV
    simp only [reindex, LawfulNode.jvpVec_ofFn, D, P,
      ContinuousLinearMap.comp_apply, CtxVec.getCLM_apply]
    exact rfl

end Tape.RealNode

end
end TorchLean
