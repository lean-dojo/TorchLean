/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Tape.RealNode.Shape

/-!
# Reduction and shape tape nodes

The two nodes in this file are linear: `sum` adds all entries of a tensor into a scalar, and
`concat` joins two tensors along the leading axis. Each certificate therefore uses one constant
derivative, and each VJP is the adjoint of the forward map. A broadcast along
`Shape.CanBroadcastTo` is the node `affine idx A 0` with `A = Broadcast.toCLM cb`. A sum along one
axis takes for `A` the adjoint of the repeat map `Broadcast.afterSumCLM`, and a mean over an axis
of length `n` scales that adjoint by `1/n`. The certificate `affineFDeriv` covers all three.
-/

@[expose] public section

open TorchLean.Shape (size size_nil)

namespace TorchLean

open Shape TorchLean.Tensor

noncomputable section

open scoped BigOperators

namespace Tape.RealNode

open Vec

-- ---------------------------------------------------------------------------
-- Reduction: sum to a scalar
-- ---------------------------------------------------------------------------

/-- Continuous linear map embedding a scalar into the 1D scalar-vector representation. -/
def _root_.TorchLean.Vec.scalarCLM : ℝ →L[ℝ] Vec (Shape.size Shape.scalar) := by
  classical
  let fLin : ℝ →ₗ[ℝ] Vec (Shape.size Shape.scalar) :=
    { toFun := fun a => WithLp.toLp 2 fun _ => a
      -- simp leaves out `size_nil`, which would rewrite `Fin (Shape.size Shape.scalar)` to
      -- `Fin 1` in the index types before the vector lemmas match.
      map_add' := by intro a b; ext i; simp [-Shape.size_nil]
      map_smul' := by intro r a; ext i; simp [-Shape.size_nil] }
  refine ⟨fLin, ?_⟩
  exact LinearMap.continuous_of_finiteDimensional (f := fLin)

/-- The one coordinate of a scalar vector is the scalar. -/
@[simp] theorem _root_.TorchLean.Vec.scalarCLM_apply (a : ℝ) (i : Fin (Shape.size Shape.scalar)) :
    scalarCLM a i = a := rfl

/-- Continuous linear map summing the entries of a vector: `x ↦ ∑ i, x i`. -/
def _root_.TorchLean.Vec.sumCLM (n : Nat) : Vec n →L[ℝ] ℝ := by
  classical
  let fLin : Vec n →ₗ[ℝ] ℝ :=
    { toFun := fun x => ∑ i : Fin n, x i
      map_add' := by
        intro x y
        simp [Finset.sum_add_distrib]
      map_smul' := by
        intro r x
        simp [Finset.mul_sum] }
  refine ⟨fLin, ?_⟩
  exact LinearMap.continuous_of_finiteDimensional (f := fLin)

/-- Evaluation lemma for `sumCLM`. -/
theorem _root_.TorchLean.Vec.sumCLM_apply {n : Nat} (x : Vec n) :
    sumCLM (n := n) x = ∑ i : Fin n, x i := rfl

/-- Sum all entries of a context tensor into a scalar tensor. -/
def sum {Γ : List Shape} {s : Shape} (idx : Idx Γ s) : LawfulNode (α := ℝ) Unit Γ Shape.scalar :=
  LawfulNode.ofFn (Γ := Γ) (τ := Shape.scalar)
    (f := fun x =>
      WithLp.toLp 2 fun _ : Fin (Shape.size Shape.scalar) =>
        (sumCLM (n := Shape.size s)) (CtxVec.get (Γ := Γ) (s := s) idx x))
    (jvp := fun _x dx =>
      WithLp.toLp 2 fun _ : Fin (Shape.size Shape.scalar) =>
        (sumCLM (n := Shape.size s)) (CtxVec.get (Γ := Γ) (s := s) idx dx))
    (vjp := fun _x δ =>
      let i0 : Fin (Shape.size Shape.scalar) := ⟨0, by simp⟩
      CtxVec.single (Γ := Γ) (s := s) idx
        (WithLp.toLp 2 fun _ : Fin (Shape.size s) => δ i0))
    (correct_inner := by
      intro _x dx δ
      classical
      let i0 : Fin (Shape.size Shape.scalar) := ⟨0, by simp⟩
      let δ0 : ℝ := δ i0
      have hctx :
          inner ℝ dx (CtxVec.single (Γ := Γ) (s := s) idx
              (WithLp.toLp 2 fun _ => δ0)) =
            inner ℝ (CtxVec.get (Γ := Γ) (s := s) idx dx)
              (WithLp.toLp 2 fun _ => δ0) := by
        simpa using
          (CtxVec.inner_get_single (Γ := Γ) (s := s) idx dx
            (WithLp.toLp 2 fun _ => δ0))
      -- expand both inner products into coordinate sums
      -- LHS (Vec 1): a single coordinate
      have hL :
          inner ℝ
              (WithLp.toLp 2
                fun _ : Fin (Shape.size Shape.scalar) =>
                  (sumCLM (n := Shape.size s)) (CtxVec.get (Γ := Γ) (s := s) idx dx))
              δ
            =
          (sumCLM (n := Shape.size s)) (CtxVec.get (Γ := Γ) (s := s) idx dx) * δ0 := by
        convert
          inner_ofFun_scalar_left
            (a := (sumCLM (n := Shape.size s)) (CtxVec.get (Γ := Γ) (s := s) idx dx))
            (δ := δ) using 1
      have hR :
          inner ℝ (CtxVec.get (Γ := Γ) (s := s) idx dx)
              (WithLp.toLp 2 fun _ => δ0)
            =
          (sumCLM (n := Shape.size s)) (CtxVec.get (Γ := Γ) (s := s) idx dx) * δ0 := by
        classical
        -- Expand `inner` to `∑ j, dx_j * δ0`, pull out the constant, and recognise `sumCLM`.
        have hsum :
            (∑ j : Fin (Shape.size s), CtxVec.get (Γ := Γ) (s := s) idx dx j * δ0)
              =
            (∑ j : Fin (Shape.size s), CtxVec.get (Γ := Γ) (s := s) idx dx j) * δ0 := by
          -- `∑ j, f j * a = (∑ j, f j) * a`
          simpa using
            (Finset.sum_mul (s := Finset.univ) (f := fun j : Fin (Shape.size s) =>
                CtxVec.get (Γ := Γ) (s := s) idx dx j) (a := δ0)).symm
        -- rewrite the LHS via `inner_eq_sum_mul` then apply `hsum` and `sumCLM_apply`
        calc
          inner ℝ (CtxVec.get (Γ := Γ) (s := s) idx dx)
              (WithLp.toLp 2 fun _ => δ0)
              = ∑ j : Fin (Shape.size s), CtxVec.get (Γ := Γ) (s := s) idx dx j * δ0 := by
                  simp [inner_eq_sum_mul]
          _ = (∑ j : Fin (Shape.size s), CtxVec.get (Γ := Γ) (s := s) idx dx j) * δ0 := hsum
          _ = (sumCLM (n := Shape.size s)) (CtxVec.get (Γ := Γ) (s := s) idx dx) * δ0 := by
                simp [sumCLM_apply]
      calc
        inner ℝ
            (WithLp.toLp 2
              fun _ : Fin (Shape.size Shape.scalar) =>
                (sumCLM (n := Shape.size s)) (CtxVec.get (Γ := Γ) (s := s) idx dx))
            δ
            = (sumCLM (n := Shape.size s)) (CtxVec.get (Γ := Γ) (s := s) idx dx) * δ0 := hL
        _ = inner ℝ (CtxVec.get (Γ := Γ) (s := s) idx dx)
              (WithLp.toLp 2 fun _ => δ0) := by
              simpa using hR.symm
        _ = inner ℝ dx (CtxVec.single (Γ := Γ) (s := s) idx
              (WithLp.toLp 2 fun _ => δ0)) := by
              simpa using hctx.symm )

/-- `FDerivCorrect` for `sum`. Its derivative is the constant map that reads the slot,
adds up its coordinates and embeds the total as a scalar tensor. -/
def sumFDeriv {Γ : List Shape} {s : Shape} (idx : Idx Γ s) :
    FDerivCorrect (sum (Γ := Γ) (s := s) idx).toNodeData () :=
{ deriv := fun _ =>
    scalarCLM.comp
      ((sumCLM (n := Shape.size s)).comp (CtxVec.getCLM (Γ := Γ) (s := s) idx))
  hasFDerivAt := by
    intro xV
    let D :=
      scalarCLM.comp
        ((sumCLM (n := Shape.size s)).comp (CtxVec.getCLM (Γ := Γ) (s := s) idx))
    have hD : HasFDerivAt (fun x : CtxVec Γ => D x) D xV :=
      D.hasFDerivAt (x := xV)
    -- rewrite the forward function of the `sum` node to this CLM (pointwise)
    have hEq :
        (NodeData.forwardVec (Γ := Γ) (τ := Shape.scalar)
          (sum (Γ := Γ) (s := s) idx).toNodeData ()) = fun x : CtxVec Γ => D x := by
      funext x
      simp only [sum, LawfulNode.forwardVec_ofFn, D, ContinuousLinearMap.comp_apply,
        CtxVec.getCLM_apply]
      apply PiLp.ext
      intro i
      simp only [scalarCLM_apply]
    exact hD.congr_of_eventuallyEq hEq.eventuallyEq
  jvp_eq := by
    intro xV dxV
    simp only [sum, LawfulNode.jvpVec_ofFn, ContinuousLinearMap.comp_apply, CtxVec.getCLM_apply]
    apply PiLp.ext
    intro i
    simp only [scalarCLM_apply] }

-- ---------------------------------------------------------------------------
-- Broadcast and repeat-along-an-axis maps
-- ---------------------------------------------------------------------------

namespace Broadcast

open scoped BigOperators

/-- Compute the source index in `s₁` that corresponds to a target index in `s₂` under broadcasting.

The recursion is on the shapes. The broadcast relation only rules out the impossible cases and
shows that a source extent differing from the target extent is one. -/
def toIndex :
    {s₁ s₂ : Shape} → Shape.CanBroadcastTo s₁ s₂ →
      Fin (Shape.size s₂) → Fin (Shape.size s₁)
  | .scalar, .scalar, _, _ => ⟨0, by simp⟩
  | .dim _ _, .scalar, h, _ => absurd h Shape.not_canBroadcastTo_dim_scalar
  | .scalar, .dim n s₂, h, j =>
      toIndex (Shape.canBroadcastTo_scalar_dim.mp h)
        (j.modNat (m := n) (n := Shape.size s₂))
  | .dim m s₁, .dim n s₂, h, j =>
      if hRank : s₁.rank = s₂.rank then
        have hHead := ((Shape.canBroadcastTo_dim_dim_of_rank_eq hRank).mp h).1
        have tail := ((Shape.canBroadcastTo_dim_dim_of_rank_eq hRank).mp h).2
        let jInner : Fin (Shape.size s₂) := j.modNat (m := n) (n := Shape.size s₂)
        let iOuter : Fin m :=
          if hExtent : m = n then
            Fin.cast hExtent.symm (j.divNat (m := n) (n := Shape.size s₂))
          else
            Fin.cast (hHead.resolve_left hExtent).symm (0 : Fin 1)
        finProdFinEquiv (iOuter, toIndex (s₁ := s₁) (s₂ := s₂) tail jInner)
      else
        toIndex ((Shape.canBroadcastTo_dim_dim_of_rank_ne hRank).mp h)
          (j.modNat (m := n) (n := Shape.size s₂))

/-- Broadcast a vector `Vec (size s₁)` into `Vec (size s₂)` using the `CanBroadcastTo` index map. -/
def toVec {s₁ s₂ : Shape} (cb : Shape.CanBroadcastTo s₁ s₂) :
    Vec (Shape.size s₁) → Vec (Shape.size s₂) :=
  fun v =>
    WithLp.toLp 2 fun j =>
      v (toIndex (s₁ := s₁) (s₂ := s₂) cb j)

/-- Continuous-linear-map form of `toVec`. -/
def toCLM {s₁ s₂ : Shape} (cb : Shape.CanBroadcastTo s₁ s₂) :
    Vec (Shape.size s₁) →L[ℝ] Vec (Shape.size s₂) := by
  classical
  let fLin : Vec (Shape.size s₁) →ₗ[ℝ] Vec (Shape.size s₂) :=
    { toFun := toVec (s₁ := s₁) (s₂ := s₂) cb
      map_add' := by
        intro x y
        ext i
        simp [toVec]
      map_smul' := by
        intro r x
        ext i
        simp [toVec, smul_eq_mul] }
  refine ⟨fLin, ?_⟩
  exact LinearMap.continuous_of_finiteDimensional (f := fLin)

/-- The bundled broadcast map computes `toVec`. Broadcasting is linear, and the node
`affine idx (toCLM cb) 0` broadcasts the slot `idx`. -/
@[simp] theorem toCLM_apply {s₁ s₂ : Shape} (cb : Shape.CanBroadcastTo s₁ s₂)
    (v : Vec (Shape.size s₁)) :
    toCLM (s₁ := s₁) (s₂ := s₂) cb v = toVec (s₁ := s₁) (s₂ := s₂) cb v := rfl

/-- Source index obtained by deleting coordinate `axis` from an index into `s`. -/
def afterSumIndex : (s : Shape) → (axis : Nat) →
    Fin (Shape.size s) → Fin (Shape.size (s.eraseIdx axis))
  | .scalar, _, _ => ⟨0, by simp⟩
  | .dim n inner, 0, j => j.modNat (m := n) (n := Shape.size inner)
  | .dim n inner, Nat.succ axis, j =>
      let jOuter : Fin n := j.divNat (m := n) (n := Shape.size inner)
      let jInner : Fin (Shape.size inner) :=
        j.modNat (m := n) (n := Shape.size inner)
      finProdFinEquiv (jOuter, afterSumIndex inner axis jInner)

/-- Repeat a flattened reduced tensor along the axis removed by `List.eraseIdx`. -/
def afterSumVec (s : Shape) (axis : Nat) :
    Vec (Shape.size (s.eraseIdx axis)) → Vec (Shape.size s) :=
  fun v => WithLp.toLp 2 fun j => v (afterSumIndex s axis j)

/-- Continuous-linear-map form of `afterSumVec`. -/
def afterSumCLM (s : Shape) (axis : Nat) :
    Vec (Shape.size (s.eraseIdx axis)) →L[ℝ] Vec (Shape.size s) := by
  classical
  let fLin : Vec (Shape.size (s.eraseIdx axis)) →ₗ[ℝ] Vec (Shape.size s) :=
    { toFun := afterSumVec s axis
      map_add' := by
        intro x y
        ext i
        simp [afterSumVec]
      map_smul' := by
        intro r x
        ext i
        simp [afterSumVec, smul_eq_mul] }
  refine ⟨fLin, ?_⟩
  exact LinearMap.continuous_of_finiteDimensional (f := fLin)

/-- The bundled repeat-along-the-summed-axis map computes `afterSumVec`.

Summation over `axis` is the adjoint of this map, and the node `affine idx A 0` with `A` that
adjoint sums the slot `idx` along `axis`. Its backward pass is therefore a broadcast, since each
input coordinate feeds exactly one sum and gets back that sum's cotangent. -/
@[simp] theorem afterSumCLM_apply (s : Shape) (axis : Nat)
    (v : Vec (Shape.size (s.eraseIdx axis))) :
    afterSumCLM s axis v = afterSumVec s axis v := rfl

end Broadcast

-- ---------------------------------------------------------------------------
-- Concatenation
-- ---------------------------------------------------------------------------

/-- Concatenate two tensors along the leading axis by appending their flattened vectors.
Its VJP splits the cotangent at the boundary and sends each part back to its own input. -/
def concat {Γ : List Shape} {n m : Nat} {s : Shape}
    (a : Idx Γ (.dim n s)) (b : Idx Γ (.dim m s)) :
    LawfulNode (α := ℝ) Unit Γ (.dim (n + m) s) :=
  let hsz :
      Shape.size (Shape.dim n s) + Shape.size (Shape.dim m s) =
        Shape.size (Shape.dim (n + m) s) := by
        simp [Nat.add_mul]
  LawfulNode.ofFn (Γ := Γ) (τ := .dim (n + m) s)
    (f := fun xV =>
      LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ (finCongr hsz)
        ((EuclideanSpace.finAddEquivProd (𝕜 := ℝ)).symm
          (CtxVec.get (Γ := Γ) (s := .dim n s) a xV, CtxVec.get (Γ := Γ) (s := .dim m s) b xV)))
    (jvp := fun _xV dxV =>
      LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ (finCongr hsz)
        ((EuclideanSpace.finAddEquivProd (𝕜 := ℝ)).symm
          (CtxVec.get (Γ := Γ) (s := .dim n s) a dxV, CtxVec.get (Γ := Γ) (s := .dim m s) b dxV)))
    (vjp := fun _xV δV =>
      let δ' : Vec (Shape.size (Shape.dim n s) + Shape.size (Shape.dim m s)) :=
        LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ (finCongr hsz.symm) δV
      let δL : Vec (Shape.size (Shape.dim n s)) := (EuclideanSpace.finAddEquivProd (𝕜 := ℝ) δ').1
      let δR : Vec (Shape.size (Shape.dim m s)) := (EuclideanSpace.finAddEquivProd (𝕜 := ℝ) δ').2
      CtxVec.single (Γ := Γ) (s := .dim n s) a δL + CtxVec.single (Γ := Γ) (s := .dim m s) b δR)
    (correct_inner := by
      intro _xV dxV δV
      classical
      let hsz :
          Shape.size (Shape.dim n s) + Shape.size (Shape.dim m s) =
            Shape.size (Shape.dim (n + m) s) := by
            simp [Nat.add_mul]
      let da : Vec (Shape.size (Shape.dim n s)) := CtxVec.get (Γ := Γ) (s := .dim n s) a dxV
      let db : Vec (Shape.size (Shape.dim m s)) := CtxVec.get (Γ := Γ) (s := .dim m s) b dxV
      let δ' : Vec (Shape.size (Shape.dim n s) + Shape.size (Shape.dim m s)) :=
        LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ (finCongr hsz.symm) δV
      let δL : Vec (Shape.size (Shape.dim n s)) := (EuclideanSpace.finAddEquivProd (𝕜 := ℝ) δ').1
      let δR : Vec (Shape.size (Shape.dim m s)) := (EuclideanSpace.finAddEquivProd (𝕜 := ℝ) δ').2
      have hδ' : (EuclideanSpace.finAddEquivProd (𝕜 := ℝ)).symm (δL, δR) = δ' :=
        (EuclideanSpace.finAddEquivProd (𝕜 := ℝ)).symm_apply_apply δ'
      have hadd :
          inner ℝ ((EuclideanSpace.finAddEquivProd (𝕜 := ℝ)).symm (da, db)) δ' =
            inner ℝ da δL + inner ℝ db δR := by
        rw [← hδ']
        exact inner_append (a := da) (c := δL) (b := db) (d := δR)
      have hadjA := (CtxVec.inner_get_single (Γ := Γ) (s := .dim n s) a dxV δL).symm
      have hadjB := (CtxVec.inner_get_single (Γ := Γ) (s := .dim m s) b dxV δR).symm
      have hcast :
          inner ℝ
              (LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ (finCongr hsz)
                ((EuclideanSpace.finAddEquivProd (𝕜 := ℝ)).symm (da, db)))
              δV
            =
          inner ℝ ((EuclideanSpace.finAddEquivProd (𝕜 := ℝ)).symm (da, db)) δ' := by
        -- move the cast to the right argument
        rw [LinearIsometryEquiv.inner_map_eq_flip, LinearIsometryEquiv.piLpCongrLeft_symm,
          finCongr_symm]
      calc
        inner ℝ
            (LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ (finCongr hsz)
              ((EuclideanSpace.finAddEquivProd (𝕜 := ℝ)).symm (da, db)))
            δV
            =
          inner ℝ ((EuclideanSpace.finAddEquivProd (𝕜 := ℝ)).symm (da, db)) δ' := hcast
        _ = inner ℝ da δL + inner ℝ db δR := hadd
        _ = inner ℝ dxV (CtxVec.single (Γ := Γ) (s := .dim n s) a δL) +
              inner ℝ dxV (CtxVec.single (Γ := Γ) (s := .dim m s) b δR) := by
              calc
                inner ℝ da δL + inner ℝ db δR
                    =
                  inner ℝ dxV (CtxVec.single (Γ := Γ) (s := .dim n s) a δL) + inner ℝ db δR := by
                    simpa [hadjA]
                _ =
                  inner ℝ dxV (CtxVec.single (Γ := Γ) (s := .dim n s) a δL) +
                    inner ℝ dxV (CtxVec.single (Γ := Γ) (s := .dim m s) b δR) := by
                    simpa [hadjB]
        _ = inner ℝ dxV (CtxVec.single (Γ := Γ) (s := .dim n s) a δL + CtxVec.single (Γ := Γ) (s :=
          .dim m s) b δR) := by
              simp [inner_add_right]
    )

/-- `FDerivCorrect` for `concat` (concat is linear). -/
def concatFDeriv {Γ : List Shape} {n m : Nat} {s : Shape}
    (a : Idx Γ (.dim n s)) (b : Idx Γ (.dim m s)) :
    FDerivCorrect (concat (Γ := Γ) (n := n) (m := m) (s := s) a b).toNodeData () := by
  classical
  let szA : Nat := Shape.size (Shape.dim n s)
  let szB : Nat := Shape.size (Shape.dim m s)
  let hsz : szA + szB = Shape.size (Shape.dim (n + m) s) := by
    simp [szA, szB, Nat.add_mul]
  let Dcast : Vec (szA + szB) →L[ℝ] Vec (Shape.size (Shape.dim (n + m) s)) :=
    (LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ
      (finCongr hsz)).toContinuousLinearEquiv.toContinuousLinearMap
  let Dapp : (Vec szA × Vec szB) →L[ℝ] Vec (szA + szB) :=
    (EuclideanSpace.finAddEquivProd (𝕜 := ℝ)).symm.toContinuousLinearMap
  let Dpair : CtxVec Γ →L[ℝ] (Vec szA × Vec szB) :=
    ContinuousLinearMap.prod (CtxVec.getCLM (Γ := Γ) (s := .dim n s) a) (CtxVec.getCLM (Γ := Γ) (s
      := .dim m s) b)
  let D : CtxVec Γ →L[ℝ] Vec (Shape.size (Shape.dim (n + m) s)) := Dcast.comp (Dapp.comp Dpair)
  refine
    { deriv := fun _ => D
      hasFDerivAt := ?_
      jvp_eq := ?_ }
  · intro xV
    have hD : HasFDerivAt (fun x : CtxVec Γ => D x) D xV := D.hasFDerivAt (x := xV)
    have hEq :
        (NodeData.forwardVec (Γ := Γ) (τ := .dim (n + m) s)
          (concat (Γ := Γ) (n := n) (m := m) (s := s) a b).toNodeData ())
          =
        fun x : CtxVec Γ => D x := by
      funext x
      -- Unfold and normalize casts/append.
      simp only [concat, LawfulNode.forwardVec_ofFn, D, Dcast, Dapp, Dpair,
        ContinuousLinearMap.comp_apply, ContinuousLinearMap.prod_apply, CtxVec.getCLM_apply,
        szA, szB, ContinuousLinearEquiv.coe_coe, LinearIsometryEquiv.coe_toContinuousLinearEquiv]
    exact hD.congr_of_eventuallyEq hEq.eventuallyEq
  · intro xV dxV
    -- `concat` is linear, so its JVP matches the (constant) derivative.
    simp only [concat, LawfulNode.jvpVec_ofFn, D, Dcast, Dapp, Dpair,
      ContinuousLinearMap.comp_apply, ContinuousLinearMap.prod_apply, CtxVec.getCLM_apply,
      szA, szB, ContinuousLinearEquiv.coe_coe, LinearIsometryEquiv.coe_toContinuousLinearEquiv]


end Tape.RealNode

end

end TorchLean
