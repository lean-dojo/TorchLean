/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Tape.Nodes.Softmax

/-!
# Reduction and shape tape nodes

Scalar sums, broadcast-to, reduce-sum, reduce-mean, concatenation, and the linear shape adapters used
by larger graph proofs.
-/

@[expose] public section

namespace Proofs
namespace Autograd

open Spec
open Tensor

noncomputable section

open scoped BigOperators

namespace TapeNodes

-- ---------------------------------------------------------------------------
-- Reduction: sum to a scalar
-- ---------------------------------------------------------------------------

/-- Continuous linear map embedding a scalar into the 1D scalar-vector representation. -/
def vecScalarCLM : ℝ →L[ℝ] Vec (Shape.size Shape.scalar) := by
  classical
  let fLin : ℝ →ₗ[ℝ] Vec (Shape.size Shape.scalar) :=
    { toFun := fun a => vecOfFun (n := Shape.size Shape.scalar) fun _ => a
      map_add' := by intro a b; ext i; simp [vecOfFun]
      map_smul' := by intro r a; ext i; simp [vecOfFun] }
  refine ⟨fLin, ?_⟩
  exact LinearMap.continuous_of_finiteDimensional (f := fLin)

@[simp] lemma vecScalarCLM_apply (a : ℝ) (i : Fin (Shape.size Shape.scalar)) :
    vecScalarCLM a i = a := rfl

@[simp] lemma vecScalarCLM_ofLp (a : ℝ) (i : Fin (Shape.size Shape.scalar)) :
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
lemma sumCLM_apply {n : Nat} (x : Vec n) :
    sumCLM (n := n) x = ∑ i : Fin n, x i := rfl

/-- Sum all entries of a context tensor into a scalar tensor. -/
def sum {Γ : List Shape} {s : Shape} (idx : Idx Γ s) : Node Γ Shape.scalar :=
  Node.ofVec (Γ := Γ) (τ := Shape.scalar)
    (f := fun x =>
      vecOfFun (n := Shape.size Shape.scalar) fun _ : Fin (Shape.size Shape.scalar) =>
        (sumCLM (n := Shape.size s)) (CtxVec.get (Γ := Γ) (s := s) idx x))
    (jvp := fun _x dx =>
      vecOfFun (n := Shape.size Shape.scalar) fun _ : Fin (Shape.size Shape.scalar) =>
        (sumCLM (n := Shape.size s)) (CtxVec.get (Γ := Γ) (s := s) idx dx))
    (vjp := fun _x δ =>
      let i0 : Fin (Shape.size Shape.scalar) := ⟨0, by simp [Shape.size]⟩
      CtxVec.single (Γ := Γ) (s := s) idx (vecOfFun (n := Shape.size s) fun _ : Fin (Shape.size s)
        => δ i0))
    (correct_inner := by
      intro _x dx δ
      classical
      let i0 : Fin (Shape.size Shape.scalar) := ⟨0, by simp [Shape.size]⟩
      let δ0 : ℝ := δ i0
      have hctx :
          inner ℝ dx (CtxVec.single (Γ := Γ) (s := s) idx (vecOfFun (n := Shape.size s) fun _ =>
            δ0)) =
            inner ℝ (CtxVec.get (Γ := Γ) (s := s) idx dx) (vecOfFun (n := Shape.size s) fun _ => δ0)
              := by
        simpa using
          (CtxVec.inner_get_single (Γ := Γ) (s := s) idx dx (vecOfFun (n := Shape.size s) fun _ =>
            δ0))
      -- expand both inner products into coordinate sums
      -- LHS (Vec 1): a single coordinate
      have hL :
          inner ℝ
              (vecOfFun (n := Shape.size Shape.scalar) fun _ : Fin (Shape.size Shape.scalar) =>
                (sumCLM (n := Shape.size s)) (CtxVec.get (Γ := Γ) (s := s) idx dx))
              δ
            =
          (sumCLM (n := Shape.size s)) (CtxVec.get (Γ := Γ) (s := s) idx dx) * δ0 := by
        convert
          inner_scalarVec_left
            (a := (sumCLM (n := Shape.size s)) (CtxVec.get (Γ := Γ) (s := s) idx dx))
            (δ := δ) using 1
      have hR :
          inner ℝ (CtxVec.get (Γ := Γ) (s := s) idx dx) (vecOfFun (n := Shape.size s) fun _ => δ0)
            =
          (sumCLM (n := Shape.size s)) (CtxVec.get (Γ := Γ) (s := s) idx dx) * δ0 := by
        classical
        -- expand `inner` and pull out the constant factor
        -- `inner` expands to `∑ i, (CtxVec.get .. dx i) * δ0`
        -- and `sumCLM` is the coordinate sum.
        -- Expand `inner` and pull out the constant factor.
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
          inner ℝ (CtxVec.get (Γ := Γ) (s := s) idx dx) (vecOfFun (n := Shape.size s) fun _ => δ0)
              = ∑ j : Fin (Shape.size s), CtxVec.get (Γ := Γ) (s := s) idx dx j * δ0 := by
                  simp [inner_eq_sum_mul]
          _ = (∑ j : Fin (Shape.size s), CtxVec.get (Γ := Γ) (s := s) idx dx j) * δ0 := hsum
          _ = (sumCLM (n := Shape.size s)) (CtxVec.get (Γ := Γ) (s := s) idx dx) * δ0 := by
                simp [sumCLM_apply]
      -- combine
      calc
        inner ℝ
            (vecOfFun (n := Shape.size Shape.scalar) fun _ : Fin (Shape.size Shape.scalar) =>
              (sumCLM (n := Shape.size s)) (CtxVec.get (Γ := Γ) (s := s) idx dx))
            δ
            = (sumCLM (n := Shape.size s)) (CtxVec.get (Γ := Γ) (s := s) idx dx) * δ0 := hL
        _ = inner ℝ (CtxVec.get (Γ := Γ) (s := s) idx dx) (vecOfFun (n := Shape.size s) fun _ => δ0)
          := by
              simpa using hR.symm
        _ = inner ℝ dx (CtxVec.single (Γ := Γ) (s := s) idx (vecOfFun (n := Shape.size s) fun _ =>
          δ0)) := by
              simpa using hctx.symm )

/-- `NodeFDerivCorrect` for `sum`: derivative is the composite of context projection and coordinate
  sum. -/
def sumFderiv {Γ : List Shape} {s : Shape} (idx : Idx Γ s) :
    NodeFDerivCorrect (sum (Γ := Γ) (s := s) idx) :=
{ deriv := fun _ =>
    vecScalarCLM.comp ((sumCLM (n := Shape.size s)).comp (CtxVec.getCLM (Γ := Γ) (s := s) idx))
  hasFDerivAt := by
    intro xV
    let D :=
      vecScalarCLM.comp ((sumCLM (n := Shape.size s)).comp (CtxVec.getCLM (Γ := Γ) (s := s) idx))
    have hD : HasFDerivAt (fun x : CtxVec Γ => D x) D xV :=
      D.hasFDerivAt (x := xV)
    -- rewrite the forward function of the `sum` node to this CLM (pointwise)
    have hEq :
        (Node.forwardVec (Γ := Γ) (τ := Shape.scalar) (sum (Γ := Γ) (s := s) idx)) = fun x : CtxVec
          Γ => D x := by
      funext x
      ext i
      change (Node.forwardVec (Γ := Γ) (τ := Shape.scalar) (sum (Γ := Γ) (s := s) idx) x).ofLp i =
        (D x).ofLp i
      -- unfold both sides down to scalars
      have hL :
          (Node.forwardVec (Γ := Γ) (τ := Shape.scalar) (sum (Γ := Γ) (s := s) idx) x).ofLp i
            =
          (sumCLM (n := Shape.size s)) (CtxVec.get (Γ := Γ) (s := s) idx x) := by
        simp [sum, Node.forwardVec_ofVec, Shape.size]
      have hR :
          (D x).ofLp i =
            (sumCLM (n := Shape.size s)) (CtxVec.get (Γ := Γ) (s := s) idx x) := by
        change
            (vecScalarCLM (((sumCLM (n := Shape.size s)).comp (CtxVec.getCLM (Γ := Γ) (s := s) idx))
              x)).ofLp i
              =
            (sumCLM (n := Shape.size s)) (CtxVec.get (Γ := Γ) (s := s) idx x)
        simpa [ContinuousLinearMap.comp_apply, CtxVec.getCLM_apply, Function.comp, Shape.size] using
          (vecScalarCLM_ofLp (a := (sumCLM (n := Shape.size s))
            (CtxVec.get (Γ := Γ) (s := s) idx x)) (i := i))
      exact hL.trans hR.symm
    exact hD.congr_of_eventuallyEq hEq.eventuallyEq
  jvp_eq := by
    intro xV dxV
    ext i
    change (Node.jvpVec (Γ := Γ) (τ := Shape.scalar) (sum (Γ := Γ) (s := s) idx) xV dxV).ofLp i =
      (vecScalarCLM.comp ((sumCLM (n := Shape.size s)).comp (CtxVec.getCLM (Γ := Γ) (s := s) idx))
        dxV).ofLp i
    have hL :
        (Node.jvpVec (Γ := Γ) (τ := Shape.scalar) (sum (Γ := Γ) (s := s) idx) xV dxV).ofLp i
          =
        (sumCLM (n := Shape.size s)) (CtxVec.get (Γ := Γ) (s := s) idx dxV) := by
      simp [sum, Node.jvpVec_ofVec, Shape.size]
    have hR :
        (vecScalarCLM.comp ((sumCLM (n := Shape.size s)).comp (CtxVec.getCLM (Γ := Γ) (s := s) idx))
          dxV).ofLp i
          =
        (sumCLM (n := Shape.size s)) (CtxVec.get (Γ := Γ) (s := s) idx dxV) := by
      change
          (vecScalarCLM (((sumCLM (n := Shape.size s)).comp (CtxVec.getCLM (Γ := Γ) (s := s) idx))
            dxV)).ofLp i
            =
          (sumCLM (n := Shape.size s)) (CtxVec.get (Γ := Γ) (s := s) idx dxV)
      simpa [ContinuousLinearMap.comp_apply, CtxVec.getCLM_apply, Function.comp, Shape.size] using
        (vecScalarCLM_ofLp (a := (sumCLM (n := Shape.size s))
          (CtxVec.get (Γ := Γ) (s := s) idx dxV)) (i := i))
    exact hL.trans hR.symm }

-- ---------------------------------------------------------------------------
-- Shape ops: broadcast, reductions, concat, losses
-- ---------------------------------------------------------------------------

namespace Broadcast

open scoped BigOperators

/-- Compute the source index in `s₁` that corresponds to a target index in `s₂` under broadcasting.
  -/
def broadcastToIndex :
    {s₁ s₂ : Shape} → Shape.CanBroadcastTo s₁ s₂ → Fin (Shape.size s₂) → Fin (Shape.size s₁)
  | .scalar, s₂, Shape.CanBroadcastTo.scalar_to_any _, _ => ⟨0, by simp [Shape.size]⟩
  | .dim n s₁, .dim _ s₂, Shape.CanBroadcastTo.dim_eq tail, j =>
      let jOuter : Fin n := j.divNat (m := n) (n := Shape.size s₂)
      let jInner : Fin (Shape.size s₂) := j.modNat (m := n) (n := Shape.size s₂)
      finProdFinEquiv (jOuter, broadcastToIndex (s₁ := s₁) (s₂ := s₂) tail jInner)
  | .dim 1 s₁, .dim n s₂, Shape.CanBroadcastTo.dim_1_to_n tail, j =>
      let jInner : Fin (Shape.size s₂) := j.modNat (m := n) (n := Shape.size s₂)
      let z : Fin 1 := ⟨0, by simp⟩
      finProdFinEquiv (z, broadcastToIndex (s₁ := s₁) (s₂ := s₂) tail jInner)
  | s₁, .dim n s₂, Shape.CanBroadcastTo.expand_dims tail, j =>
      let jInner : Fin (Shape.size s₂) := j.modNat (m := n) (n := Shape.size s₂)
      broadcastToIndex (s₁ := s₁) (s₂ := s₂) tail jInner

/-- Broadcast a vector `Vec (size s₁)` into `Vec (size s₂)` using the `CanBroadcastTo` index map. -/
def broadcastToVec {s₁ s₂ : Shape} (cb : Shape.CanBroadcastTo s₁ s₂) :
    Vec (Shape.size s₁) → Vec (Shape.size s₂) :=
  fun v =>
    vecOfFun (n := Shape.size s₂) fun j =>
      v (broadcastToIndex (s₁ := s₁) (s₂ := s₂) cb j)

/-- Continuous-linear-map form of `broadcastToVec`. -/
def broadcastToCLM {s₁ s₂ : Shape} (cb : Shape.CanBroadcastTo s₁ s₂) :
    Vec (Shape.size s₁) →L[ℝ] Vec (Shape.size s₂) := by
  classical
  let fLin : Vec (Shape.size s₁) →ₗ[ℝ] Vec (Shape.size s₂) :=
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

@[simp] lemma broadcastToCLM_apply {s₁ s₂ : Shape} (cb : Shape.CanBroadcastTo s₁ s₂) (v : Vec
  (Shape.size s₁)) :
    broadcastToCLM (s₁ := s₁) (s₂ := s₂) cb v = broadcastToVec (s₁ := s₁) (s₂ := s₂) cb v := rfl

end Broadcast

/-- General shape broadcast node `s₁ → s₂` (linear). -/
def broadcastTo {Γ : List Shape} {s₁ s₂ : Shape} (idx : Idx Γ s₁) (cb : Shape.CanBroadcastTo s₁ s₂)
  :
    Node Γ s₂ :=
  Node.ofVec (Γ := Γ) (τ := s₂)
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
      simp [broadcastTo, Node.forwardVec_ofVec, D, Broadcast.broadcastToCLM_apply,
        CtxVec.getCLM_apply,
        ContinuousLinearMap.comp_apply]
    exact hD.congr_of_eventuallyEq hEq.eventuallyEq
  jvp_eq := by
    intro xV dxV
    simp [broadcastTo, Node.jvpVec_ofVec, Broadcast.broadcastToCLM_apply, CtxVec.getCLM_apply,
      ContinuousLinearMap.comp_apply] }

/-- Sum reduction along `axis` (linear; adjoint is broadcast back). -/
def reduceSum {Γ : List Shape} {s : Shape} (axis : Nat) [valid : Shape.valid_axis_inst axis s] [wf
  : Shape.WellFormed s]
    (idx : Idx Γ s) : Node Γ (shapeAfterSum s axis) :=
  let cb := shapeAfterSumBroadcastBack (s := s) axis valid wf
  let B := Broadcast.broadcastToCLM (s₁ := shapeAfterSum s axis) (s₂ := s) cb
  Node.ofVec (Γ := Γ) (τ := shapeAfterSum s axis)
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

/-- `NodeFDerivCorrect` for `reduce_sum`. -/
def reduceSumFderiv {Γ : List Shape} {s : Shape} (axis : Nat) [valid : Shape.valid_axis_inst axis
  s] [wf : Shape.WellFormed s]
    (idx : Idx Γ s) :
    NodeFDerivCorrect (reduceSum (Γ := Γ) (s := s) axis idx) :=
by
  classical
  let cb := shapeAfterSumBroadcastBack (s := s) axis valid wf
  let B := Broadcast.broadcastToCLM (s₁ := shapeAfterSum s axis) (s₂ := s) cb
  let D : CtxVec Γ →L[ℝ] Vec (Shape.size (shapeAfterSum s axis)) :=
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
      simp [reduceSum, Node.forwardVec_ofVec, D, B, cb, CtxVec.getCLM_apply,
        ContinuousLinearMap.comp_apply]
    exact hD.congr_of_eventuallyEq hEq.eventuallyEq
  · intro xV dxV
    simp [reduceSum, Node.jvpVec_ofVec, D, B, cb, CtxVec.getCLM_apply,
      ContinuousLinearMap.comp_apply]

/-- Mean reduction along `axis` (linear; adjoint is broadcast+scale). -/
def reduceMean {Γ : List Shape} {s : Shape} (axis : Nat) [valid : Shape.valid_axis_inst axis s] [wf
  : Shape.WellFormed s]
    (idx : Idx Γ s) : Node Γ (shapeAfterSum s axis) :=
  let cb := shapeAfterSumBroadcastBack (s := s) axis valid wf
  let B := Broadcast.broadcastToCLM (s₁ := shapeAfterSum s axis) (s₂ := s) cb
  let denomNat : Nat :=
    match getDimSize s axis with
    | some n => n
    | none => 1
  let c : ℝ := (1 : ℝ) / (denomNat : ℝ)
  Node.ofVec (Γ := Γ) (τ := shapeAfterSum s axis)
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

/-- `NodeFDerivCorrect` for `reduce_mean`. -/
def reduceMeanFderiv {Γ : List Shape} {s : Shape} (axis : Nat) [valid : Shape.valid_axis_inst axis
  s] [wf : Shape.WellFormed s]
    (idx : Idx Γ s) :
    NodeFDerivCorrect (reduceMean (Γ := Γ) (s := s) axis idx) :=
by
  classical
  let cb := shapeAfterSumBroadcastBack (s := s) axis valid wf
  let B := Broadcast.broadcastToCLM (s₁ := shapeAfterSum s axis) (s₂ := s) cb
  let denomNat : Nat :=
    match getDimSize s axis with
    | some n => n
    | none => 1
  let c : ℝ := (1 : ℝ) / (denomNat : ℝ)
  let D : CtxVec Γ →L[ℝ] Vec (Shape.size (shapeAfterSum s axis)) :=
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
      simp [reduceMean, Node.forwardVec_ofVec, D, B, cb, c, denomNat, CtxVec.getCLM_apply]
    exact hD.congr_of_eventuallyEq hEq.eventuallyEq
  · intro xV dxV
    simp [reduceMean, Node.jvpVec_ofVec, D, B, cb, c, denomNat, CtxVec.getCLM_apply]

-- ---------------------------------------------------------------------------
-- Concat (binary specializations used by the runtime engine)
-- ---------------------------------------------------------------------------

/-- `castVec` does not depend on the particular proof of `n = m` (proof irrelevance). -/
lemma castVec_proof_irrel {n m : Nat} (h₁ h₂ : n = m) (v : Vec n) :
    castVec h₁ v = castVec h₂ v := by
  have : h₁ = h₂ := Subsingleton.elim _ _
  cases this
  rfl

/-- Take the left `m` entries of a length `m+n` vector. -/
def takeLeftVec {m n : Nat} (v : Vec (m + n)) : Vec m :=
  vecOfFun (n := m) fun i : Fin m => v (Fin.castAdd n i)

/-- Take the right `n` entries of a length `m+n` vector. -/
def takeRightVec {m n : Nat} (v : Vec (m + n)) : Vec n :=
  vecOfFun (n := n) fun i : Fin n => v (Fin.natAdd m i)

/-- Splitting then appending recovers the original vector: `append (takeLeft v) (takeRight v) = v`.
  -/
private lemma append_takeLeft_takeRight {m n : Nat} (v : Vec (m + n)) :
    appendVec (m := m) (n := n) (takeLeftVec (m := m) (n := n) v) (takeRightVec (m := m) (n := n) v)
      = v := by
  classical
  ext i
  cases i using Fin.addCases <;>
    simp [appendVec, takeLeftVec, takeRightVec, vecOfFun, Fin.append, Fin.addCases]

  /-- Concatenate two context vectors into a single `(n+m)`-vector node (dim-0 concat). -/
  def concatVectors {Γ : List Shape} {n m : Nat}
      (a : Idx Γ (.dim n .scalar)) (b : Idx Γ (.dim m .scalar)) :
      Node Γ (.dim (n + m) .scalar) :=
    let hsz : Shape.size (.dim (n + m) .scalar) = n + m := by
      simp [Shape.size]
    Node.ofVec (Γ := Γ) (τ := .dim (n + m) .scalar)
      (f := fun xV =>
        castVec hsz.symm <|
          appendVec (m := n) (n := m) (getVec (Γ := Γ) (n := n) a xV) (getVec (Γ := Γ) (n := m) b
            xV))
      (jvp := fun _xV dxV =>
        castVec hsz.symm <|
          appendVec (m := n) (n := m) (getVec (Γ := Γ) (n := n) a dxV) (getVec (Γ := Γ) (n := m) b
            dxV))
      (vjp := fun _xV δV =>
        let δ' : Vec (n + m) := castVec hsz δV
        let δL : Vec n := takeLeftVec (m := n) (n := m) δ'
        let δR : Vec m := takeRightVec (m := n) (n := m) δ'
        singleVec (Γ := Γ) (n := n) a δL + singleVec (Γ := Γ) (n := m) b δR)
      (correct_inner := by
        intro _xV dxV δV
        classical
        let hsz : Shape.size (.dim (n + m) .scalar) = n + m := by
          simp [Shape.size]
        let da : Vec n := getVec (Γ := Γ) (n := n) a dxV
        let db : Vec m := getVec (Γ := Γ) (n := m) b dxV
        let δ' : Vec (n + m) := castVec hsz δV
        let δL : Vec n := takeLeftVec (m := n) (n := m) δ'
        let δR : Vec m := takeRightVec (m := n) (n := m) δ'
        have hδ : appendVec (m := n) (n := m) δL δR = δ' :=
          append_takeLeft_takeRight (m := n) (n := m) δ'
        have hcast :
            inner ℝ (castVec hsz.symm (appendVec (m := n) (n := m) da db)) δV
              =
            inner ℝ (appendVec (m := n) (n := m) da db) δ' := by
          -- rewrite `δV` as a cast of `δ'` and use the cast-isometry lemma
          simpa [δ'] using
            (inner_castVec_castVec (h := hsz.symm)
              (x := appendVec (m := n) (n := m) da db) (y := δ'))
        have hadd :
            inner ℝ (appendVec (m := n) (n := m) da db) δ' = inner ℝ da δL + inner ℝ db δR := by
          simpa [hδ] using
            (inner_append (m := n) (n := m) (a := da) (b := db) (c := δL) (d := δR))
        have hctxA := (inner_getVec_singleVec (Γ := Γ) (n := n) a dxV δL)
        have hctxB := (inner_getVec_singleVec (Γ := Γ) (n := m) b dxV δR)
        calc
          inner ℝ (castVec hsz.symm (appendVec (m := n) (n := m) da db)) δV
              = inner ℝ (appendVec (m := n) (n := m) da db) δ' := hcast
          _ = inner ℝ da δL + inner ℝ db δR := hadd
          _ = inner ℝ dxV (singleVec (Γ := Γ) (n := n) a δL) + inner ℝ dxV (singleVec (Γ := Γ) (n :=
            m) b δR) := by
                simp [hctxA, hctxB, da, db]
          _ = inner ℝ dxV (singleVec (Γ := Γ) (n := n) a δL + singleVec (Γ := Γ) (n := m) b δR) :=
            by
                simp [inner_add_right])

  /-- `NodeFDerivCorrect` for `concat_vectors`. -/
  def concatVectorsFderiv {Γ : List Shape} {n m : Nat}
      (a : Idx Γ (.dim n .scalar)) (b : Idx Γ (.dim m .scalar)) :
      NodeFDerivCorrect (concatVectors (Γ := Γ) (n := n) (m := m) a b) :=
  by
    classical
    let hsz : Shape.size (.dim (n + m) .scalar) = n + m := by
      simp [Shape.size]
    let D : CtxVec Γ →L[ℝ] Vec (Shape.size (.dim (n + m) .scalar)) := by
      let f0 : CtxVec Γ →ₗ[ℝ] Vec (n + m) :=
        { toFun := fun xV =>
            appendVec (m := n) (n := m) (getVec (Γ := Γ) (n := n) a xV) (getVec (Γ := Γ) (n := m) b
              xV)
          map_add' := by
            intro x y
            have hA :
                getVec (Γ := Γ) (n := n) a (x + y) =
                  getVec (Γ := Γ) (n := n) a x + getVec (Γ := Γ) (n := n) a y := by
              simpa [getVecCLM_apply] using (getVecCLM (Γ := Γ) (n := n) a).map_add x y
            have hB :
                getVec (Γ := Γ) (n := m) b (x + y) =
                  getVec (Γ := Γ) (n := m) b x + getVec (Γ := Γ) (n := m) b y := by
              simpa [getVecCLM_apply] using (getVecCLM (Γ := Γ) (n := m) b).map_add x y
            ext i
            cases i using Fin.addCases <;>
              simp [appendVec, vecOfFun, Fin.append, Fin.addCases, Pi.add_apply, hA, hB]
          map_smul' := by
            intro r x
            have hA :
                getVec (Γ := Γ) (n := n) a (r • x) = r • getVec (Γ := Γ) (n := n) a x := by
              calc
                getVec (Γ := Γ) (n := n) a (r • x)
                    =
                  getVecCLM (Γ := Γ) (n := n) a (r • x) := (getVecCLM_apply (Γ := Γ) (n := n) a (r •
                    x)).symm
                _ = r • getVecCLM (Γ := Γ) (n := n) a x := (getVecCLM (Γ := Γ) (n := n) a).map_smul
                  r x
                _ = r • getVec (Γ := Γ) (n := n) a x :=
                  congrArg (fun v => r • v) (getVecCLM_apply (Γ := Γ) (n := n) a x)
            have hB :
                getVec (Γ := Γ) (n := m) b (r • x) = r • getVec (Γ := Γ) (n := m) b x := by
              calc
                getVec (Γ := Γ) (n := m) b (r • x)
                    =
                  getVecCLM (Γ := Γ) (n := m) b (r • x) := (getVecCLM_apply (Γ := Γ) (n := m) b (r •
                    x)).symm
                _ = r • getVecCLM (Γ := Γ) (n := m) b x := (getVecCLM (Γ := Γ) (n := m) b).map_smul
                  r x
                _ = r • getVec (Γ := Γ) (n := m) b x :=
                  congrArg (fun v => r • v) (getVecCLM_apply (Γ := Γ) (n := m) b x)
            ext i
            cases i using Fin.addCases <;>
              simp [appendVec, vecOfFun, Fin.append, Fin.addCases, Pi.smul_apply, smul_eq_mul, hA,
                hB] }
      let fLin : CtxVec Γ →ₗ[ℝ] Vec (Shape.size (.dim (n + m) .scalar)) :=
        { toFun := fun xV => castVec hsz.symm (f0 xV)
          map_add' := by
            intro x y
            simp [castVec_add]
          map_smul' := by
            intro r x
            simp [castVec_smul]}
      exact ⟨fLin, LinearMap.continuous_of_finiteDimensional (f := fLin)⟩
    refine
      { deriv := fun _ => D
        hasFDerivAt := ?_
        jvp_eq := ?_ }
    · intro xV
      have hD : HasFDerivAt (fun x : CtxVec Γ => D x) D xV := D.hasFDerivAt (x := xV)
      have hEq :
          (Node.forwardVec (Γ := Γ) (τ := .dim (n + m) .scalar) (concatVectors (Γ := Γ) (n := n) (m
            := m) a b))
            =
          fun x : CtxVec Γ => D x := by
        funext x
        simp [concatVectors, Node.forwardVec_ofVec, D]
      exact hD.congr_of_eventuallyEq hEq.eventuallyEq
    · intro xV dxV
      simp [concatVectors, Node.jvpVec_ofVec, D]

/-- Concatenate two tensors along dimension 0 (dim-0 concat), using flattened vectors internally. -/
def concatDim0 {Γ : List Shape} {n m : Nat} {s : Shape}
    (a : Idx Γ (.dim n s)) (b : Idx Γ (.dim m s)) :
    Node Γ (.dim (n + m) s) :=
  let hsz :
      Shape.size (.dim n s) + Shape.size (.dim m s) = Shape.size (.dim (n + m) s) := by
        simp [Shape.size, Nat.add_mul]
  Node.ofVec (Γ := Γ) (τ := .dim (n + m) s)
    (f := fun xV =>
      castVec hsz (appendVec (m := Shape.size (.dim n s)) (n := Shape.size (.dim m s))
        (CtxVec.get (Γ := Γ) (s := .dim n s) a xV)
        (CtxVec.get (Γ := Γ) (s := .dim m s) b xV)))
    (jvp := fun _xV dxV =>
      castVec hsz (appendVec (m := Shape.size (.dim n s)) (n := Shape.size (.dim m s))
        (CtxVec.get (Γ := Γ) (s := .dim n s) a dxV)
        (CtxVec.get (Γ := Γ) (s := .dim m s) b dxV)))
    (vjp := fun _xV δV =>
      let δ' : Vec (Shape.size (.dim n s) + Shape.size (.dim m s)) := castVec hsz.symm δV
      let δL : Vec (Shape.size (.dim n s)) := takeLeftVec (m := Shape.size (.dim n s)) (n :=
        Shape.size (.dim m s)) δ'
      let δR : Vec (Shape.size (.dim m s)) := takeRightVec (m := Shape.size (.dim n s)) (n :=
        Shape.size (.dim m s)) δ'
      CtxVec.single (Γ := Γ) (s := .dim n s) a δL + CtxVec.single (Γ := Γ) (s := .dim m s) b δR)
    (correct_inner := by
      intro _xV dxV δV
      classical
      let hsz :
          Shape.size (.dim n s) + Shape.size (.dim m s) = Shape.size (.dim (n + m) s) := by
            simp [Shape.size, Nat.add_mul]
      let da : Vec (Shape.size (.dim n s)) := CtxVec.get (Γ := Γ) (s := .dim n s) a dxV
      let db : Vec (Shape.size (.dim m s)) := CtxVec.get (Γ := Γ) (s := .dim m s) b dxV
      let δ' : Vec (Shape.size (.dim n s) + Shape.size (.dim m s)) := castVec hsz.symm δV
      let δL : Vec (Shape.size (.dim n s)) := takeLeftVec (m := Shape.size (.dim n s)) (n :=
        Shape.size (.dim m s)) δ'
      let δR : Vec (Shape.size (.dim m s)) := takeRightVec (m := Shape.size (.dim n s)) (n :=
        Shape.size (.dim m s)) δ'
      have hδ' : appendVec (m := Shape.size (.dim n s)) (n := Shape.size (.dim m s)) δL δR = δ' :=
        append_takeLeft_takeRight (m := Shape.size (.dim n s)) (n := Shape.size (.dim m s)) δ'
      have hadd :
          inner ℝ (appendVec (m := Shape.size (.dim n s)) (n := Shape.size (.dim m s)) da db) δ'
            =
          inner ℝ da δL + inner ℝ db δR := by
        simpa [hδ'] using
          (inner_append (m := Shape.size (.dim n s)) (n := Shape.size (.dim m s))
            (a := da) (b := db) (c := δL) (d := δR))
      have hadjA := (CtxVec.inner_get_single (Γ := Γ) (s := .dim n s) a dxV δL).symm
      have hadjB := (CtxVec.inner_get_single (Γ := Γ) (s := .dim m s) b dxV δR).symm
      have hcast :
          inner ℝ (castVec hsz (appendVec (m := Shape.size (.dim n s)) (n := Shape.size (.dim m s))
            da db)) δV
            =
          inner ℝ (appendVec (m := Shape.size (.dim n s)) (n := Shape.size (.dim m s)) da db) δ' :=
            by
        -- move the cast to the right argument
        simpa [δ'] using
          (inner_castVec_castVec (h := hsz) (x := appendVec (m := Shape.size (.dim n s)) (n :=
            Shape.size (.dim m s)) da db)
            (y := δ'))
      calc
        inner ℝ (castVec hsz (appendVec (m := Shape.size (.dim n s)) (n := Shape.size (.dim m s)) da
          db)) δV
            =
          inner ℝ (appendVec (m := Shape.size (.dim n s)) (n := Shape.size (.dim m s)) da db) δ' :=
            hcast
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

/-- `NodeFDerivCorrect` for `concat_dim0` (concat is linear). -/
def concatDim0Fderiv {Γ : List Shape} {n m : Nat} {s : Shape}
    (a : Idx Γ (.dim n s)) (b : Idx Γ (.dim m s)) :
    NodeFDerivCorrect (concatDim0 (Γ := Γ) (n := n) (m := m) (s := s) a b) := by
  classical
  let szA : Nat := Shape.size (.dim n s)
  let szB : Nat := Shape.size (.dim m s)
  let hsz : szA + szB = Shape.size (.dim (n + m) s) := by
    simp [szA, szB, Shape.size, Nat.add_mul]
  let Dcast : Vec (szA + szB) →L[ℝ] Vec (Shape.size (.dim (n + m) s)) := Graph.castCLM (h := hsz)
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
  let D : CtxVec Γ →L[ℝ] Vec (Shape.size (.dim (n + m) s)) := Dcast.comp (Dapp.comp Dpair)
  refine
    { deriv := fun _ => D
      hasFDerivAt := ?_
      jvp_eq := ?_ }
  · intro xV
    have hD : HasFDerivAt (fun x : CtxVec Γ => D x) D xV := D.hasFDerivAt (x := xV)
    have hEq :
        (Node.forwardVec (Γ := Γ) (τ := .dim (n + m) s) (concatDim0 (Γ := Γ) (n := n) (m := m) (s
          := s) a b))
          =
        fun x : CtxVec Γ => D x := by
      funext x
      -- Unfold and normalize casts/append.
      simp [concatDim0, Node.forwardVec_ofVec, D, Dcast, Dapp, Dpair,
        Graph.castCLM, ContinuousLinearMap.comp_apply, ContinuousLinearMap.prod_apply,
        CtxVec.getCLM_apply, hsz, szA, szB, castVec_proof_irrel]
    exact hD.congr_of_eventuallyEq hEq.eventuallyEq
  · intro xV dxV
    -- `concat_dim0` is linear, so its JVP matches the (constant) derivative.
    simp [concatDim0, Node.jvpVec_ofVec, D, Dcast, Dapp, Dpair,
      Graph.castCLM, ContinuousLinearMap.comp_apply, ContinuousLinearMap.prod_apply,
      CtxVec.getCLM_apply, hsz, szA, szB, castVec_proof_irrel]


end TapeNodes

end

end Autograd
end Proofs
