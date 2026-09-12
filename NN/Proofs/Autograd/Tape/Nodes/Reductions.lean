/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Tape.Nodes.Shape

/-!
# Reduction and shape tape nodes

Scalar sums, broadcast-to, reduce-sum, reduce-mean, concatenation, and the linear shape adapters
used by larger graph proofs.
-/

@[expose] public section

namespace Proofs
namespace Autograd

open Spec TorchLean
open TorchLean TorchLean.Tensor

noncomputable section

open scoped BigOperators

namespace TapeNodes

-- ---------------------------------------------------------------------------
-- Reduction: sum to a scalar
-- ---------------------------------------------------------------------------

/-- Continuous linear map embedding a scalar into the 1D scalar-vector representation. -/
def vecScalarCLM : ℝ →L[ℝ] Vec (Spec.Shape.size Shape.scalar) := by
  classical
  let fLin : ℝ →ₗ[ℝ] Vec (Spec.Shape.size Shape.scalar) :=
    { toFun := fun a => vecOfFun (n := Spec.Shape.size Shape.scalar) fun _ => a
      map_add' := by intro a b; ext i; simp [vecOfFun]
      map_smul' := by intro r a; ext i; simp [vecOfFun] }
  refine ⟨fLin, ?_⟩
  exact LinearMap.continuous_of_finiteDimensional (f := fLin)

/-- The one coordinate of a scalar vector is the scalar. -/
@[simp] theorem vecScalarCLM_apply (a : ℝ) (i : Fin (Spec.Shape.size Shape.scalar)) :
    vecScalarCLM a i = a := rfl

/-- Same through the `WithLp` wrapper. -/
@[simp] theorem vecScalarCLM_ofLp (a : ℝ) (i : Fin (Spec.Shape.size Shape.scalar)) :
    (vecScalarCLM a).ofLp i = a := rfl

/-- Continuous linear map summing the entries of a vector: `x ↦ ∑ i, x i`. -/
def sumCLM (n : Nat) : Vec n →L[ℝ] ℝ := by
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
theorem sumCLM_apply {n : Nat} (x : Vec n) :
    sumCLM (n := n) x = ∑ i : Fin n, x i := rfl

/-- Sum all entries of a context tensor into a scalar tensor. -/
def sum {Γ : List Shape} {s : Shape} (idx : Idx Γ s) : Node Γ Shape.scalar :=
  Node.ofFn (Γ := Γ) (τ := Shape.scalar)
    (f := fun x =>
      vecOfFun (n := Spec.Shape.size Shape.scalar) fun _ : Fin (Spec.Shape.size Shape.scalar) =>
        (sumCLM (n := Spec.Shape.size s)) (CtxVec.get (Γ := Γ) (s := s) idx x))
    (jvp := fun _x dx =>
      vecOfFun (n := Spec.Shape.size Shape.scalar) fun _ : Fin (Spec.Shape.size Shape.scalar) =>
        (sumCLM (n := Spec.Shape.size s)) (CtxVec.get (Γ := Γ) (s := s) idx dx))
    (vjp := fun _x δ =>
      let i0 : Fin (Spec.Shape.size Shape.scalar) := ⟨0, by simp [Spec.Shape.size]⟩
      CtxVec.single (Γ := Γ) (s := s) idx
        (vecOfFun (n := Spec.Shape.size s) fun _ : Fin (Spec.Shape.size s) => δ i0))
    (correct_inner := by
      intro _x dx δ
      classical
      let i0 : Fin (Spec.Shape.size Shape.scalar) := ⟨0, by simp [Spec.Shape.size]⟩
      let δ0 : ℝ := δ i0
      have hctx :
          inner ℝ dx (CtxVec.single (Γ := Γ) (s := s) idx
              (vecOfFun (n := Spec.Shape.size s) fun _ => δ0)) =
            inner ℝ (CtxVec.get (Γ := Γ) (s := s) idx dx)
              (vecOfFun (n := Spec.Shape.size s) fun _ => δ0) := by
        simpa using
          (CtxVec.inner_get_single (Γ := Γ) (s := s) idx dx
            (vecOfFun (n := Spec.Shape.size s) fun _ => δ0))
      -- expand both inner products into coordinate sums
      -- LHS (Vec 1): a single coordinate
      have hL :
          inner ℝ
              (vecOfFun (n := Spec.Shape.size Shape.scalar)
                fun _ : Fin (Spec.Shape.size Shape.scalar) =>
                  (sumCLM (n := Spec.Shape.size s)) (CtxVec.get (Γ := Γ) (s := s) idx dx))
              δ
            =
          (sumCLM (n := Spec.Shape.size s)) (CtxVec.get (Γ := Γ) (s := s) idx dx) * δ0 := by
        convert
          inner_scalarVec_left
            (a := (sumCLM (n := Spec.Shape.size s)) (CtxVec.get (Γ := Γ) (s := s) idx dx))
            (δ := δ) using 1
      have hR :
          inner ℝ (CtxVec.get (Γ := Γ) (s := s) idx dx)
              (vecOfFun (n := Spec.Shape.size s) fun _ => δ0)
            =
          (sumCLM (n := Spec.Shape.size s)) (CtxVec.get (Γ := Γ) (s := s) idx dx) * δ0 := by
        classical
        -- expand `inner` and pull out the constant factor
        -- `inner` expands to `∑ i, (CtxVec.get .. dx i) * δ0`
        -- and `sumCLM` is the coordinate sum.
        -- Expand `inner` and pull out the constant factor.
        have hsum :
            (∑ j : Fin (Spec.Shape.size s), CtxVec.get (Γ := Γ) (s := s) idx dx j * δ0)
              =
            (∑ j : Fin (Spec.Shape.size s), CtxVec.get (Γ := Γ) (s := s) idx dx j) * δ0 := by
          -- `∑ j, f j * a = (∑ j, f j) * a`
          simpa using
            (Finset.sum_mul (s := Finset.univ) (f := fun j : Fin (Spec.Shape.size s) =>
                CtxVec.get (Γ := Γ) (s := s) idx dx j) (a := δ0)).symm
        -- rewrite the LHS via `inner_eq_sum_mul` then apply `hsum` and `sumCLM_apply`
        calc
          inner ℝ (CtxVec.get (Γ := Γ) (s := s) idx dx)
              (vecOfFun (n := Spec.Shape.size s) fun _ => δ0)
              = ∑ j : Fin (Spec.Shape.size s), CtxVec.get (Γ := Γ) (s := s) idx dx j * δ0 := by
                  simp [inner_eq_sum_mul]
          _ = (∑ j : Fin (Spec.Shape.size s), CtxVec.get (Γ := Γ) (s := s) idx dx j) * δ0 := hsum
          _ = (sumCLM (n := Spec.Shape.size s)) (CtxVec.get (Γ := Γ) (s := s) idx dx) * δ0 := by
                simp [sumCLM_apply]
      -- combine
      calc
        inner ℝ
            (vecOfFun (n := Spec.Shape.size Shape.scalar)
              fun _ : Fin (Spec.Shape.size Shape.scalar) =>
                (sumCLM (n := Spec.Shape.size s)) (CtxVec.get (Γ := Γ) (s := s) idx dx))
            δ
            = (sumCLM (n := Spec.Shape.size s)) (CtxVec.get (Γ := Γ) (s := s) idx dx) * δ0 := hL
        _ = inner ℝ (CtxVec.get (Γ := Γ) (s := s) idx dx)
              (vecOfFun (n := Spec.Shape.size s) fun _ => δ0) := by
              simpa using hR.symm
        _ = inner ℝ dx (CtxVec.single (Γ := Γ) (s := s) idx
              (vecOfFun (n := Spec.Shape.size s) fun _ => δ0)) := by
              simpa using hctx.symm )

/-- `NodeFDerivCorrect` for `sum`: derivative is the composite of context projection and coordinate
  sum. -/
def sumFderiv {Γ : List Shape} {s : Shape} (idx : Idx Γ s) :
    NodeFDerivCorrect (sum (Γ := Γ) (s := s) idx) :=
{ deriv := fun _ =>
    vecScalarCLM.comp
      ((sumCLM (n := Spec.Shape.size s)).comp (CtxVec.getCLM (Γ := Γ) (s := s) idx))
  hasFDerivAt := by
    intro xV
    let D :=
      vecScalarCLM.comp
        ((sumCLM (n := Spec.Shape.size s)).comp (CtxVec.getCLM (Γ := Γ) (s := s) idx))
    have hD : HasFDerivAt (fun x : CtxVec Γ => D x) D xV :=
      D.hasFDerivAt (x := xV)
    -- rewrite the forward function of the `sum` node to this CLM (pointwise)
    have hEq :
        (Node.forwardVec (Γ := Γ) (τ := Shape.scalar) (sum (Γ := Γ) (s := s) idx)) = fun x : CtxVec
          Γ => D x := by
      funext x
      simp only [sum, Node.forwardVec_ofFn, D, ContinuousLinearMap.comp_apply,
        CtxVec.getCLM_apply]
      apply PiLp.ext
      intro i
      simp only [vecOfFun_ofLp, vecScalarCLM_ofLp]
    exact hD.congr_of_eventuallyEq hEq.eventuallyEq
  jvp_eq := by
    intro xV dxV
    simp only [sum, Node.jvpVec_ofFn, ContinuousLinearMap.comp_apply, CtxVec.getCLM_apply]
    apply PiLp.ext
    intro i
    simp only [vecOfFun_ofLp, vecScalarCLM_ofLp] }

-- ---------------------------------------------------------------------------
-- Shape ops: broadcast, reductions, concat, losses
-- ---------------------------------------------------------------------------

namespace Broadcast

open scoped BigOperators

/-- Compute the source index in `s₁` that corresponds to a target index in `s₂` under broadcasting.

The recursion is on the shapes. The broadcast relation only rules out the impossible cases and
shows that a source extent differing from the target extent is one. -/
def broadcastToIndex :
    {s₁ s₂ : Shape} → Shape.CanBroadcastTo s₁ s₂ →
      Fin (Spec.Shape.size s₂) → Fin (Spec.Shape.size s₁)
  | .scalar, .scalar, _, _ => ⟨0, by simp [Spec.Shape.size]⟩
  | .dim _ _, .scalar, h, _ => absurd h Shape.not_canBroadcastTo_dim_scalar
  | .scalar, .dim n s₂, h, j =>
      broadcastToIndex (Shape.canBroadcastTo_scalar_dim.mp h)
        (j.modNat (m := n) (n := Spec.Shape.size s₂))
  | .dim m s₁, .dim n s₂, h, j =>
      if hRank : s₁.rank = s₂.rank then
        have hHead := ((Shape.canBroadcastTo_dim_dim_of_rank_eq hRank).mp h).1
        have tail := ((Shape.canBroadcastTo_dim_dim_of_rank_eq hRank).mp h).2
        let jInner : Fin (Spec.Shape.size s₂) := j.modNat (m := n) (n := Spec.Shape.size s₂)
        let iOuter : Fin m :=
          if hExtent : m = n then
            Fin.cast hExtent.symm (j.divNat (m := n) (n := Spec.Shape.size s₂))
          else
            Fin.cast (hHead.resolve_left hExtent).symm (0 : Fin 1)
        finProdFinEquiv (iOuter, broadcastToIndex (s₁ := s₁) (s₂ := s₂) tail jInner)
      else
        broadcastToIndex ((Shape.canBroadcastTo_dim_dim_of_rank_ne hRank).mp h)
          (j.modNat (m := n) (n := Spec.Shape.size s₂))

/-- Broadcast a vector `Vec (size s₁)` into `Vec (size s₂)` using the `CanBroadcastTo` index map. -/
def broadcastToVec {s₁ s₂ : Shape} (cb : Shape.CanBroadcastTo s₁ s₂) :
    Vec (Spec.Shape.size s₁) → Vec (Spec.Shape.size s₂) :=
  fun v =>
    vecOfFun (n := Spec.Shape.size s₂) fun j =>
      v (broadcastToIndex (s₁ := s₁) (s₂ := s₂) cb j)

/-- Continuous-linear-map form of `broadcastToVec`. -/
def broadcastToCLM {s₁ s₂ : Shape} (cb : Shape.CanBroadcastTo s₁ s₂) :
    Vec (Spec.Shape.size s₁) →L[ℝ] Vec (Spec.Shape.size s₂) := by
  classical
  let fLin : Vec (Spec.Shape.size s₁) →ₗ[ℝ] Vec (Spec.Shape.size s₂) :=
    { toFun := broadcastToVec (s₁ := s₁) (s₂ := s₂) cb
      map_add' := by
        intro x y
        ext i
        simp [broadcastToVec, vecOfFun]
      map_smul' := by
        intro r x
        ext i
        simp [broadcastToVec, smul_eq_mul, vecOfFun] }
  refine ⟨fLin, ?_⟩
  exact LinearMap.continuous_of_finiteDimensional (f := fLin)

/-- The bundled broadcast map computes `broadcastToVec`. Broadcasting is linear, so nothing about
the
shape relation `cb` needs to reappear in the derivative. -/
@[simp] theorem broadcastToCLM_apply {s₁ s₂ : Shape} (cb : Shape.CanBroadcastTo s₁ s₂) (v : Vec
  (Spec.Shape.size s₁)) :
    broadcastToCLM (s₁ := s₁) (s₂ := s₂) cb v = broadcastToVec (s₁ := s₁) (s₂ := s₂) cb v := rfl

/-- Source index obtained by deleting coordinate `axis` from an index into `s`. -/
def afterSumIndex : (s : Shape) → (axis : Nat) →
    Fin (Spec.Shape.size s) → Fin (Spec.Shape.size (shapeAfterSum s axis))
  | .scalar, _, _ => ⟨0, by simp [Spec.Shape.size]⟩
  | .dim n inner, 0, j => j.modNat (m := n) (n := Spec.Shape.size inner)
  | .dim n inner, Nat.succ axis, j =>
      let jOuter : Fin n := j.divNat (m := n) (n := Spec.Shape.size inner)
      let jInner : Fin (Spec.Shape.size inner) :=
        j.modNat (m := n) (n := Spec.Shape.size inner)
      finProdFinEquiv (jOuter, afterSumIndex inner axis jInner)

/-- Repeat a flattened reduced tensor along the axis removed by `shapeAfterSum`. -/
def afterSumVec (s : Shape) (axis : Nat) :
    Vec (Spec.Shape.size (shapeAfterSum s axis)) → Vec (Spec.Shape.size s) :=
  fun v => vecOfFun (n := Spec.Shape.size s) fun j => v (afterSumIndex s axis j)

/-- Continuous-linear-map form of `afterSumVec`. -/
def afterSumCLM (s : Shape) (axis : Nat) :
    Vec (Spec.Shape.size (shapeAfterSum s axis)) →L[ℝ] Vec (Spec.Shape.size s) := by
  classical
  let fLin : Vec (Spec.Shape.size (shapeAfterSum s axis)) →ₗ[ℝ] Vec (Spec.Shape.size s) :=
    { toFun := afterSumVec s axis
      map_add' := by
        intro x y
        ext i
        simp [afterSumVec, vecOfFun]
      map_smul' := by
        intro r x
        ext i
        simp [afterSumVec, smul_eq_mul, vecOfFun] }
  refine ⟨fLin, ?_⟩
  exact LinearMap.continuous_of_finiteDimensional (f := fLin)

/-- The bundled repeat-along-the-summed-axis map computes `afterSumVec`.

This map is the adjoint of summation over `axis`, which is why the backward pass for a reduction is
a
broadcast: each output coordinate contributed to exactly one sum. -/
@[simp] theorem afterSumCLM_apply (s : Shape) (axis : Nat)
    (v : Vec (Spec.Shape.size (shapeAfterSum s axis))) :
    afterSumCLM s axis v = afterSumVec s axis v := rfl

end Broadcast

/-- General shape broadcast node `s₁ → s₂` (linear). -/
def broadcastTo {Γ : List Shape} {s₁ s₂ : Shape} (idx : Idx Γ s₁) (cb : Shape.CanBroadcastTo s₁ s₂)
  :
    Node Γ s₂ :=
  Node.ofFn (Γ := Γ) (τ := s₂)
    (f := fun xV => Broadcast.broadcastToVec (s₁ := s₁) (s₂ := s₂) cb (CtxVec.get (Γ := Γ) (s := s₁)
      idx xV))
    (jvp := fun _xV dxV =>
      Broadcast.broadcastToVec (s₁ := s₁) (s₂ := s₂) cb (CtxVec.get (Γ := Γ) (s := s₁) idx dxV))
    (vjp := fun _xV δV =>
      CtxVec.single (Γ := Γ) (s := s₁) idx ((Broadcast.broadcastToCLM (s₁ := s₁) (s₂ := s₂)
        cb).adjoint δV))
    (correct_inner := by
      intro _xV dxV δV
      classical
      have hctx :=
        (CtxVec.inner_get_single (Γ := Γ) (s := s₁) idx dxV
          ((Broadcast.broadcastToCLM (s₁ := s₁) (s₂ := s₂) cb).adjoint δV))
      have hadj :
          inner ℝ
              (Broadcast.broadcastToVec (s₁ := s₁) (s₂ := s₂) cb (CtxVec.get (Γ := Γ) (s := s₁) idx
                dxV))
              δV
            =
          inner ℝ (CtxVec.get (Γ := Γ) (s := s₁) idx dxV)
              ((Broadcast.broadcastToCLM (s₁ := s₁) (s₂ := s₂) cb).adjoint δV) := by
        -- `⟪A dx, δ⟫ = ⟪dx, A† δ⟫`
        simpa [Broadcast.broadcastToCLM_apply] using
          (ContinuousLinearMap.adjoint_inner_right
              (A := Broadcast.broadcastToCLM (s₁ := s₁) (s₂ := s₂) cb)
              (x := CtxVec.get (Γ := Γ) (s := s₁) idx dxV) (y := δV)).symm
      exact hadj.trans hctx.symm)

/-- `NodeFDerivCorrect` for `broadcastTo` (broadcasting is linear). -/
def broadcastToFderiv {Γ : List Shape} {s₁ s₂ : Shape} (idx : Idx Γ s₁) (cb : Shape.CanBroadcastTo
  s₁ s₂) :
    NodeFDerivCorrect (broadcastTo (Γ := Γ) (s₁ := s₁) (s₂ := s₂) idx cb) :=
{ deriv := fun _ =>
    (Broadcast.broadcastToCLM (s₁ := s₁) (s₂ := s₂) cb).comp (CtxVec.getCLM (Γ := Γ) (s := s₁) idx)
  hasFDerivAt := by
    intro xV
    let D :=
      (Broadcast.broadcastToCLM (s₁ := s₁) (s₂ := s₂) cb).comp (CtxVec.getCLM (Γ := Γ) (s := s₁)
        idx)
    have hD : HasFDerivAt (fun x : CtxVec Γ => D x) D xV := D.hasFDerivAt (x := xV)
    have hEq :
        (Node.forwardVec (Γ := Γ) (τ := s₂) (broadcastTo (Γ := Γ) (s₁ := s₁) (s₂ := s₂) idx cb))
          =
        fun x : CtxVec Γ => D x := by
      funext x
      simp [broadcastTo, Node.forwardVec_ofFn, D, Broadcast.broadcastToCLM_apply,
        CtxVec.getCLM_apply,
        ContinuousLinearMap.comp_apply]
    exact hD.congr_of_eventuallyEq hEq.eventuallyEq
  jvp_eq := by
    intro xV dxV
    simp [broadcastTo, Node.jvpVec_ofFn, Broadcast.broadcastToCLM_apply, CtxVec.getCLM_apply,
      ContinuousLinearMap.comp_apply] }

/-- Sum reduction along `axis` (linear; adjoint is broadcast back). -/
def reduceSum {Γ : List Shape} {s : Shape} (axis : Nat)
    [_valid : Shape.HasNonemptyAxis axis s] [_wf : Shape.WellFormed s]
    (idx : Idx Γ s) : Node Γ (shapeAfterSum s axis) :=
  let B := Broadcast.afterSumCLM s axis
  Node.ofFn (Γ := Γ) (τ := shapeAfterSum s axis)
    (f := fun xV => (B.adjoint) (CtxVec.get (Γ := Γ) (s := s) idx xV))
    (jvp := fun _xV dxV => (B.adjoint) (CtxVec.get (Γ := Γ) (s := s) idx dxV))
    (vjp := fun _xV δV => CtxVec.single (Γ := Γ) (s := s) idx (B δV))
    (correct_inner := by
      intro _xV dxV δV
      classical
      have hctx := CtxVec.inner_get_single (Γ := Γ) (s := s) idx dxV (B δV)
      have hadj :
          inner ℝ ((B.adjoint) (CtxVec.get (Γ := Γ) (s := s) idx dxV)) δV
            =
          inner ℝ (CtxVec.get (Γ := Γ) (s := s) idx dxV) (B δV) := by
        -- Here `A := B†`, so `A† = B`.
        simpa using
          (ContinuousLinearMap.adjoint_inner_right (A := B.adjoint)
              (x := CtxVec.get (Γ := Γ) (s := s) idx dxV) (y := δV)).symm
      exact hadj.trans hctx.symm)

/-- `NodeFDerivCorrect` for `reduceSum`. -/
def reduceSumFderiv {Γ : List Shape} {s : Shape} (axis : Nat) [valid : Shape.HasNonemptyAxis axis
  s] [wf : Shape.WellFormed s]
    (idx : Idx Γ s) :
    NodeFDerivCorrect (reduceSum (Γ := Γ) (s := s) axis idx) :=
by
  classical
  let B := Broadcast.afterSumCLM s axis
  let D : CtxVec Γ →L[ℝ] Vec (Spec.Shape.size (shapeAfterSum s axis)) :=
    (B.adjoint).comp (CtxVec.getCLM (Γ := Γ) (s := s) idx)
  refine
    { deriv := fun _ => D
      hasFDerivAt := ?_
      jvp_eq := ?_ }
  · intro xV
    have hD : HasFDerivAt (fun x : CtxVec Γ => D x) D xV := D.hasFDerivAt (x := xV)
    have hEq :
        (Node.forwardVec (Γ := Γ) (τ := shapeAfterSum s axis) (reduceSum (Γ := Γ) (s := s) axis
          idx))
          =
        fun x : CtxVec Γ => D x := by
      funext x
      simp [reduceSum, Node.forwardVec_ofFn, D, B, CtxVec.getCLM_apply,
        ContinuousLinearMap.comp_apply]
    exact hD.congr_of_eventuallyEq hEq.eventuallyEq
  · intro xV dxV
    simp [reduceSum, Node.jvpVec_ofFn, D, B, CtxVec.getCLM_apply,
      ContinuousLinearMap.comp_apply]

/-- Mean reduction along `axis` (linear; adjoint is broadcast+scale). -/
def reduceMean {Γ : List Shape} {s : Shape} (axis : Nat)
    [valid : Shape.HasNonemptyAxis axis s] [_wf : Shape.WellFormed s]
    (idx : Idx Γ s) : Node Γ (shapeAfterSum s axis) :=
  let B := Broadcast.afterSumCLM s axis
  letI : Shape.AxisInBounds axis s := valid.proof.toAxisInBounds
  let denomNat : Nat := Shape.axisSize s axis
  let c : ℝ := (1 : ℝ) / (denomNat : ℝ)
  Node.ofFn (Γ := Γ) (τ := shapeAfterSum s axis)
    (f := fun xV => c • ((B.adjoint) (CtxVec.get (Γ := Γ) (s := s) idx xV)))
    (jvp := fun _xV dxV => c • ((B.adjoint) (CtxVec.get (Γ := Γ) (s := s) idx dxV)))
    (vjp := fun _xV δV => CtxVec.single (Γ := Γ) (s := s) idx (c • (B δV)))
    (correct_inner := by
      intro _xV dxV δV
      classical
      have hctx := CtxVec.inner_get_single (Γ := Γ) (s := s) idx dxV (c • (B δV))
      have h0 :
          inner ℝ ((B.adjoint) (CtxVec.get (Γ := Γ) (s := s) idx dxV)) δV
            =
          inner ℝ (CtxVec.get (Γ := Γ) (s := s) idx dxV) (B δV) := by
        simpa using
          (ContinuousLinearMap.adjoint_inner_right (A := B.adjoint)
              (x := CtxVec.get (Γ := Γ) (s := s) idx dxV) (y := δV)).symm
      have hadj :
          inner ℝ (c • ((B.adjoint) (CtxVec.get (Γ := Γ) (s := s) idx dxV))) δV
            =
          inner ℝ (CtxVec.get (Γ := Γ) (s := s) idx dxV) (c • (B δV)) := by
        -- `⟪c • u, v⟫ = c * ⟪u, v⟫ = ⟪u, c • v⟫`
        simp [inner_smul_left, inner_smul_right, h0]
      exact hadj.trans hctx.symm)

/-- `NodeFDerivCorrect` for `reduceMean`. -/
def reduceMeanFderiv {Γ : List Shape} {s : Shape} (axis : Nat) [valid : Shape.HasNonemptyAxis axis
  s] [wf : Shape.WellFormed s]
    (idx : Idx Γ s) :
    NodeFDerivCorrect (reduceMean (Γ := Γ) (s := s) axis idx) :=
by
  classical
  let B := Broadcast.afterSumCLM s axis
  letI : Shape.AxisInBounds axis s := valid.proof.toAxisInBounds
  let denomNat : Nat := Shape.axisSize s axis
  let c : ℝ := (1 : ℝ) / (denomNat : ℝ)
  let D : CtxVec Γ →L[ℝ] Vec (Spec.Shape.size (shapeAfterSum s axis)) :=
    (c • (B.adjoint)).comp (CtxVec.getCLM (Γ := Γ) (s := s) idx)
  refine
    { deriv := fun _ => D
      hasFDerivAt := ?_
      jvp_eq := ?_ }
  · intro xV
    have hD : HasFDerivAt (fun x : CtxVec Γ => D x) D xV := D.hasFDerivAt (x := xV)
    have hEq :
        (Node.forwardVec (Γ := Γ) (τ := shapeAfterSum s axis) (reduceMean (Γ := Γ) (s := s) axis
          idx))
          =
        fun x : CtxVec Γ => D x := by
      funext x
      simp [reduceMean, Node.forwardVec_ofFn, D, B, c, denomNat, CtxVec.getCLM_apply]
    exact hD.congr_of_eventuallyEq hEq.eventuallyEq
  · intro xV dxV
    simp [reduceMean, Node.jvpVec_ofFn, D, B, c, denomNat, CtxVec.getCLM_apply]

-- ---------------------------------------------------------------------------
-- Concatenation
-- ---------------------------------------------------------------------------

/-- Take the left `m` entries of a length `m+n` vector. -/
def takeLeftVec {m n : Nat} (v : Vec (m + n)) : Vec m :=
  vecOfFun (n := m) fun i : Fin m => v (Fin.castAdd n i)

/-- Take the right `n` entries of a length `m+n` vector. -/
def takeRightVec {m n : Nat} (v : Vec (m + n)) : Vec n :=
  vecOfFun (n := n) fun i : Fin n => v (Fin.natAdd m i)

/-- Splitting then appending recovers the original vector: `append (takeLeft v) (takeRight v) = v`.
  -/
private theorem append_takeLeft_takeRight {m n : Nat} (v : Vec (m + n)) :
    appendVec (m := m) (n := n) (takeLeftVec (m := m) (n := n) v) (takeRightVec (m := m) (n := n) v)
      = v := by
  classical
  ext i
  cases i using Fin.addCases <;>
    simp [appendVec, takeLeftVec, takeRightVec, vecOfFun, Fin.append, Fin.addCases]

/-- Concatenate two tensors along dimension 0 (dim-0 concat), using flattened vectors internally. -/
def concatLeadingAxis {Γ : List Shape} {n m : Nat} {s : Shape}
    (a : Idx Γ (.dim n s)) (b : Idx Γ (.dim m s)) :
    Node Γ (.dim (n + m) s) :=
  let hsz :
      Spec.Shape.size (.dim n s) + Spec.Shape.size (.dim m s) =
        Spec.Shape.size (.dim (n + m) s) := by
        simp [Spec.Shape.size, Nat.add_mul]
  Node.ofFn (Γ := Γ) (τ := .dim (n + m) s)
    (f := fun xV =>
      castVec hsz (appendVec (m := Spec.Shape.size (.dim n s)) (n := Spec.Shape.size (.dim m s))
        (CtxVec.get (Γ := Γ) (s := .dim n s) a xV)
        (CtxVec.get (Γ := Γ) (s := .dim m s) b xV)))
    (jvp := fun _xV dxV =>
      castVec hsz (appendVec (m := Spec.Shape.size (.dim n s)) (n := Spec.Shape.size (.dim m s))
        (CtxVec.get (Γ := Γ) (s := .dim n s) a dxV)
        (CtxVec.get (Γ := Γ) (s := .dim m s) b dxV)))
    (vjp := fun _xV δV =>
      let δ' : Vec (Spec.Shape.size (.dim n s) + Spec.Shape.size (.dim m s)) := castVec hsz.symm δV
      let δL : Vec (Spec.Shape.size (.dim n s)) :=
        takeLeftVec (m := Spec.Shape.size (.dim n s)) (n := Spec.Shape.size (.dim m s)) δ'
      let δR : Vec (Spec.Shape.size (.dim m s)) :=
        takeRightVec (m := Spec.Shape.size (.dim n s)) (n := Spec.Shape.size (.dim m s)) δ'
      CtxVec.single (Γ := Γ) (s := .dim n s) a δL + CtxVec.single (Γ := Γ) (s := .dim m s) b δR)
    (correct_inner := by
      intro _xV dxV δV
      classical
      let hsz :
          Spec.Shape.size (.dim n s) + Spec.Shape.size (.dim m s) =
            Spec.Shape.size (.dim (n + m) s) := by
            simp [Spec.Shape.size, Nat.add_mul]
      let da : Vec (Spec.Shape.size (.dim n s)) := CtxVec.get (Γ := Γ) (s := .dim n s) a dxV
      let db : Vec (Spec.Shape.size (.dim m s)) := CtxVec.get (Γ := Γ) (s := .dim m s) b dxV
      let δ' : Vec (Spec.Shape.size (.dim n s) + Spec.Shape.size (.dim m s)) := castVec hsz.symm δV
      let δL : Vec (Spec.Shape.size (.dim n s)) :=
        takeLeftVec (m := Spec.Shape.size (.dim n s)) (n := Spec.Shape.size (.dim m s)) δ'
      let δR : Vec (Spec.Shape.size (.dim m s)) :=
        takeRightVec (m := Spec.Shape.size (.dim n s)) (n := Spec.Shape.size (.dim m s)) δ'
      have hδ' :
          appendVec (m := Spec.Shape.size (.dim n s)) (n := Spec.Shape.size (.dim m s)) δL δR
            = δ' :=
        append_takeLeft_takeRight
          (m := Spec.Shape.size (.dim n s)) (n := Spec.Shape.size (.dim m s)) δ'
      have hadd :
          inner ℝ
              (appendVec (m := Spec.Shape.size (.dim n s)) (n := Spec.Shape.size (.dim m s)) da db)
              δ'
            =
          inner ℝ da δL + inner ℝ db δR := by
        simpa [hδ'] using
          (inner_append (m := Spec.Shape.size (.dim n s)) (n := Spec.Shape.size (.dim m s))
            (a := da) (b := db) (c := δL) (d := δR))
      have hadjA := (CtxVec.inner_get_single (Γ := Γ) (s := .dim n s) a dxV δL).symm
      have hadjB := (CtxVec.inner_get_single (Γ := Γ) (s := .dim m s) b dxV δR).symm
      have hcast :
          inner ℝ
              (castVec hsz
                (appendVec (m := Spec.Shape.size (.dim n s)) (n := Spec.Shape.size (.dim m s))
                  da db))
              δV
            =
          inner ℝ
              (appendVec (m := Spec.Shape.size (.dim n s)) (n := Spec.Shape.size (.dim m s)) da db)
              δ' := by
        -- move the cast to the right argument
        simpa [δ'] using
          (inner_castVec_castVec (h := hsz)
            (x := appendVec (m := Spec.Shape.size (.dim n s)) (n := Spec.Shape.size (.dim m s))
              da db)
            (y := δ'))
      calc
        inner ℝ
            (castVec hsz
              (appendVec (m := Spec.Shape.size (.dim n s)) (n := Spec.Shape.size (.dim m s))
                da db))
            δV
            =
          inner ℝ
              (appendVec (m := Spec.Shape.size (.dim n s)) (n := Spec.Shape.size (.dim m s)) da db)
              δ' := hcast
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

/-- `NodeFDerivCorrect` for `concatLeadingAxis` (concat is linear). -/
def concatLeadingAxisFderiv {Γ : List Shape} {n m : Nat} {s : Shape}
    (a : Idx Γ (.dim n s)) (b : Idx Γ (.dim m s)) :
    NodeFDerivCorrect (concatLeadingAxis (Γ := Γ) (n := n) (m := m) (s := s) a b) := by
  classical
  let szA : Nat := Spec.Shape.size (.dim n s)
  let szB : Nat := Spec.Shape.size (.dim m s)
  let hsz : szA + szB = Spec.Shape.size (.dim (n + m) s) := by
    simp [szA, szB, Spec.Shape.size, Nat.add_mul]
  let Dcast : Vec (szA + szB) →L[ℝ] Vec (Spec.Shape.size (.dim (n + m) s)) :=
    Graph.castCLM (h := hsz)
  let Dapp : (Vec szA × Vec szB) →L[ℝ] Vec (szA + szB) := by
    classical
    let fLin : (Vec szA × Vec szB) →ₗ[ℝ] Vec (szA + szB) :=
      { toFun := fun p => appendVec (m := szA) (n := szB) p.1 p.2
        map_add' := by
          intro p q
          ext i
          cases i using Fin.addCases <;>
            simp [appendVec, Fin.append, Fin.addCases]
        map_smul' := by
          intro r p
          ext i
          cases i using Fin.addCases <;>
            simp [appendVec, Fin.append, Fin.addCases, Prod.smul_fst, Prod.smul_snd] }
    exact ⟨fLin, LinearMap.continuous_of_finiteDimensional (f := fLin)⟩
  let Dpair : CtxVec Γ →L[ℝ] (Vec szA × Vec szB) :=
    ContinuousLinearMap.prod (CtxVec.getCLM (Γ := Γ) (s := .dim n s) a) (CtxVec.getCLM (Γ := Γ) (s
      := .dim m s) b)
  let D : CtxVec Γ →L[ℝ] Vec (Spec.Shape.size (.dim (n + m) s)) := Dcast.comp (Dapp.comp Dpair)
  refine
    { deriv := fun _ => D
      hasFDerivAt := ?_
      jvp_eq := ?_ }
  · intro xV
    have hD : HasFDerivAt (fun x : CtxVec Γ => D x) D xV := D.hasFDerivAt (x := xV)
    have hEq :
        (Node.forwardVec (Γ := Γ) (τ := .dim (n + m) s)
          (concatLeadingAxis (Γ := Γ) (n := n) (m := m) (s := s) a b))
          =
        fun x : CtxVec Γ => D x := by
      funext x
      -- Unfold and normalize casts/append.
      simp [concatLeadingAxis, Node.forwardVec_ofFn, D, Dcast, Dapp, Dpair,
        Graph.castCLM, ContinuousLinearMap.comp_apply, ContinuousLinearMap.prod_apply,
        CtxVec.getCLM_apply, hsz, szA, szB, ShapeOps.castVec_proof_irrel]
    exact hD.congr_of_eventuallyEq hEq.eventuallyEq
  · intro xV dxV
    -- `concat_leading_axis` is linear, so its JVP matches the (constant) derivative.
    simp [concatLeadingAxis, Node.jvpVec_ofFn, D, Dcast, Dapp, Dpair,
      Graph.castCLM, ContinuousLinearMap.comp_apply, ContinuousLinearMap.prod_apply,
      CtxVec.getCLM_apply, hsz, szA, szB, ShapeOps.castVec_proof_irrel]


end TapeNodes

end

end Autograd
end Proofs
