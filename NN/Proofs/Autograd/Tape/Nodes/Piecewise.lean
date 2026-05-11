/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Tape.Nodes.Losses

/-!
# Piecewise tape nodes

Piecewise smooth min/max nodes and pointwise differentiability facts under strict branch-selection
hypotheses.
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
-- Pointwise nondifferentiable ops: max/min (elementwise, differentiable away from ties)
-- ---------------------------------------------------------------------------

/-- Elementwise `max` node. Differentiable at points where `a ≠ b` coordinatewise. -/
def maxElem {Γ : List Shape} {s : Shape} (a b : Idx Γ s) : Node Γ s :=
  Node.ofVec (Γ := Γ) (τ := s)
    (f := fun xV =>
      vecOfFun (n := Shape.size s) fun i =>
        max (CtxVec.get (Γ := Γ) (s := s) a xV i) (CtxVec.get (Γ := Γ) (s := s) b xV i))
    (jvp := fun xV dxV =>
      vecOfFun (n := Shape.size s) fun i =>
        let xa := CtxVec.get (Γ := Γ) (s := s) a xV i
        let xb := CtxVec.get (Γ := Γ) (s := s) b xV i
        let da := CtxVec.get (Γ := Γ) (s := s) a dxV i
        let db := CtxVec.get (Γ := Γ) (s := s) b dxV i
        if xa > xb then da else db)
    (vjp := fun xV δV =>
      let vA : Vec (Shape.size s) :=
        vecOfFun (n := Shape.size s) fun i =>
          let xa := CtxVec.get (Γ := Γ) (s := s) a xV i
          let xb := CtxVec.get (Γ := Γ) (s := s) b xV i
          if xa > xb then δV i else 0
      let vB : Vec (Shape.size s) :=
        vecOfFun (n := Shape.size s) fun i =>
          let xa := CtxVec.get (Γ := Γ) (s := s) a xV i
          let xb := CtxVec.get (Γ := Γ) (s := s) b xV i
          if xa > xb then 0 else δV i
      CtxVec.single (Γ := Γ) (s := s) a vA + CtxVec.single (Γ := Γ) (s := s) b vB)
    (correct_inner := by
      intro xV dxV δV
      classical
      -- First eliminate `CtxVec.single` via `inner_get_single`, then expand the remaining vector
      -- inners.
      simp [inner_add_right, CtxVec.inner_get_single]
      simp [inner_eq_sum_mul, vecOfFun, mul_comm]
      rw [← Finset.sum_add_distrib]
      refine Finset.sum_congr rfl ?_
      intro i _
      by_cases h : (CtxVec.get (Γ := Γ) (s := s) b xV).ofLp i < (CtxVec.get (Γ := Γ) (s := s) a
        xV).ofLp i
      · simp [h]
      · simp [h])

/-- Elementwise `min` node. Differentiable at points where `a ≠ b` coordinatewise. -/
def minElem {Γ : List Shape} {s : Shape} (a b : Idx Γ s) : Node Γ s :=
  Node.ofVec (Γ := Γ) (τ := s)
    (f := fun xV =>
      vecOfFun (n := Shape.size s) fun i =>
        min (CtxVec.get (Γ := Γ) (s := s) a xV i) (CtxVec.get (Γ := Γ) (s := s) b xV i))
    (jvp := fun xV dxV =>
      vecOfFun (n := Shape.size s) fun i =>
        let xa := CtxVec.get (Γ := Γ) (s := s) a xV i
        let xb := CtxVec.get (Γ := Γ) (s := s) b xV i
        let da := CtxVec.get (Γ := Γ) (s := s) a dxV i
        let db := CtxVec.get (Γ := Γ) (s := s) b dxV i
        if xa < xb then da else db)
    (vjp := fun xV δV =>
      let vA : Vec (Shape.size s) :=
        vecOfFun (n := Shape.size s) fun i =>
          let xa := CtxVec.get (Γ := Γ) (s := s) a xV i
          let xb := CtxVec.get (Γ := Γ) (s := s) b xV i
          if xa < xb then δV i else 0
      let vB : Vec (Shape.size s) :=
        vecOfFun (n := Shape.size s) fun i =>
          let xa := CtxVec.get (Γ := Γ) (s := s) a xV i
          let xb := CtxVec.get (Γ := Γ) (s := s) b xV i
          if xa < xb then 0 else δV i
      CtxVec.single (Γ := Γ) (s := s) a vA + CtxVec.single (Γ := Γ) (s := s) b vB)
    (correct_inner := by
      intro xV dxV δV
      classical
      simp [inner_add_right, CtxVec.inner_get_single]
      simp [inner_eq_sum_mul, vecOfFun, mul_comm]
      rw [← Finset.sum_add_distrib]
      refine Finset.sum_congr rfl ?_
      intro i _
      by_cases h : (CtxVec.get (Γ := Γ) (s := s) a xV).ofLp i < (CtxVec.get (Γ := Γ) (s := s) b
        xV).ofLp i
      · simp [h]
      · simp [h])

/-- Derivative of `max` at points where the `f` branch strictly dominates (`f xV > g xV`). -/
lemma hasFDerivAt_max_of_lt {Γ : List Shape} {f g : CtxVec Γ → ℝ} {f' g' : CtxVec Γ →L[ℝ] ℝ}
    {xV : CtxVec Γ} (hf : HasFDerivAt f f' xV) (hg : HasFDerivAt g g' xV) (hfg : f xV > g xV) :
    HasFDerivAt (fun x => max (f x) (g x)) f' xV := by
  have hcont : ContinuousAt (fun x : CtxVec Γ => f x - g x) xV := hf.continuousAt.sub
    hg.continuousAt
  have hmem : Set.Ioi (0 : ℝ) ∈ nhds (f xV - g xV) := isOpen_Ioi.mem_nhds (sub_pos.mpr hfg)
  have hpos : ∀ᶠ x in nhds xV, f x > g x := by
    have := (hcont.preimage_mem_nhds hmem)
    filter_upwards [this] with x hx
    have : f x - g x > 0 := hx
    exact sub_pos.mp this
  have heq :
      (fun x => max (f x) (g x)) =ᶠ[nhds xV] fun x => f x := by
    filter_upwards [hpos] with x hx
    simp [max_eq_left (le_of_lt hx)]
  exact hf.congr_of_eventuallyEq heq

/-- Derivative of `min` at points where the `f` branch strictly dominates (`f xV < g xV`). -/
lemma hasFDerivAt_min_of_lt {Γ : List Shape} {f g : CtxVec Γ → ℝ} {f' g' : CtxVec Γ →L[ℝ] ℝ}
    {xV : CtxVec Γ} (hf : HasFDerivAt f f' xV) (hg : HasFDerivAt g g' xV) (hfg : f xV < g xV) :
    HasFDerivAt (fun x => min (f x) (g x)) f' xV := by
  have hcont : ContinuousAt (fun x : CtxVec Γ => g x - f x) xV := hg.continuousAt.sub
    hf.continuousAt
  have hmem : Set.Ioi (0 : ℝ) ∈ nhds (g xV - f xV) := isOpen_Ioi.mem_nhds (sub_pos.mpr hfg)
  have hpos : ∀ᶠ x in nhds xV, f x < g x := by
    have := (hcont.preimage_mem_nhds hmem)
    filter_upwards [this] with x hx
    have : g x - f x > 0 := hx
    exact sub_pos.mp this
  have heq :
      (fun x => min (f x) (g x)) =ᶠ[nhds xV] fun x => f x := by
    filter_upwards [hpos] with x hx
    simp [min_eq_left (le_of_lt hx)]
  exact hf.congr_of_eventuallyEq heq

/-- Pointwise `NodeFDerivCorrectAt` for `max_elem`, assuming there are no ties. -/
def maxElemFderivAt {Γ : List Shape} {s : Shape} (a b : Idx Γ s) (xV : CtxVec Γ)
    (hneq : ∀ i : Fin (Shape.size s), CtxVec.get (Γ := Γ) (s := s) a xV i ≠ CtxVec.get (Γ := Γ) (s
      := s) b xV i) :
    NodeFDerivCorrectAt (maxElem (Γ := Γ) (s := s) a b) xV := by
  classical
  let aCLM : CtxVec Γ →L[ℝ] Vec (Shape.size s) := CtxVec.getCLM (Γ := Γ) (s := s) a
  let bCLM : CtxVec Γ →L[ℝ] Vec (Shape.size s) := CtxVec.getCLM (Γ := Γ) (s := s) b
  have ha0 : HasFDerivAt (fun x : CtxVec Γ => aCLM x) aCLM xV := aCLM.hasFDerivAt (x := xV)
  have hb0 : HasFDerivAt (fun x : CtxVec Γ => bCLM x) bCLM xV := bCLM.hasFDerivAt (x := xV)
  -- coordinatewise derivative via branch stability
  have hcoord :
      ∀ i : Fin (Shape.size s),
        HasFDerivAt (fun x : CtxVec Γ => max (aCLM x i) (bCLM x i))
          (if aCLM xV i > bCLM xV i then (evalCLM (n := Shape.size s) i).comp aCLM
           else (evalCLM (n := Shape.size s) i).comp bCLM) xV := by
    intro i
    have hne : aCLM xV i ≠ bCLM xV i := by
      simpa [aCLM, bCLM, CtxVec.getCLM_apply] using hneq i
    -- coordinate projections are linear, so their derivatives are just `evalCLM ∘ getCLM`
    have ha_i :
        HasFDerivAt (fun x : CtxVec Γ => aCLM x i) ((evalCLM (n := Shape.size s) i).comp aCLM) xV :=
          by
      have houter :
          HasFDerivAt (fun v : Vec (Shape.size s) => (evalCLM (n := Shape.size s) i) v) (evalCLM (n
            := Shape.size s) i)
            (aCLM xV) :=
        (evalCLM (n := Shape.size s) i).hasFDerivAt (x := aCLM xV)
      have hcomp := houter.comp xV ha0
      simpa [ContinuousLinearMap.comp_apply, evalCLM_apply] using hcomp
    have hb_i :
        HasFDerivAt (fun x : CtxVec Γ => bCLM x i) ((evalCLM (n := Shape.size s) i).comp bCLM) xV :=
          by
      have houter :
          HasFDerivAt (fun v : Vec (Shape.size s) => (evalCLM (n := Shape.size s) i) v) (evalCLM (n
            := Shape.size s) i)
            (bCLM xV) :=
        (evalCLM (n := Shape.size s) i).hasFDerivAt (x := bCLM xV)
      have hcomp := houter.comp xV hb0
      simpa [ContinuousLinearMap.comp_apply, evalCLM_apply] using hcomp
    have hlt : aCLM xV i < bCLM xV i ∨ aCLM xV i > bCLM xV i := lt_or_gt_of_ne hne
    cases hlt with
    | inr hgt =>
        simpa [hgt, if_pos hgt] using (hasFDerivAt_max_of_lt ha_i hb_i hgt)
    | inl hlt =>
        have hn : ¬ aCLM xV i > bCLM xV i := (le_of_lt hlt).not_gt
        have hgt : bCLM xV i > aCLM xV i := hlt
        have h' :
            HasFDerivAt (fun x : CtxVec Γ => max (bCLM x i) (aCLM x i))
              ((evalCLM (n := Shape.size s) i).comp bCLM) xV :=
          hasFDerivAt_max_of_lt hb_i ha_i hgt
        -- rewrite back (and match the `if` branch)
        simpa [hn, if_neg hn, max_comm] using h'
  -- assemble
  refine
      { deriv :=
          (euclideanEquiv (Shape.size s)).symm.toContinuousLinearMap.comp <|
            ContinuousLinearMap.pi fun i : Fin (Shape.size s) =>
              if aCLM xV i > bCLM xV i then (evalCLM (n := Shape.size s) i).comp aCLM
              else (evalCLM (n := Shape.size s) i).comp bCLM
        hasFDerivAt := ?_
        jvp_eq := ?_ }
  · -- package coordinate statements, then transport `Fin n → ℝ` to `Vec n`
    have hpi :
        HasFDerivAt
          (fun x : CtxVec Γ => fun i : Fin (Shape.size s) => max (aCLM x i) (bCLM x i))
          (ContinuousLinearMap.pi fun i : Fin (Shape.size s) =>
            if aCLM xV i > bCLM xV i then (evalCLM (n := Shape.size s) i).comp aCLM
            else (evalCLM (n := Shape.size s) i).comp bCLM)
          xV := by
      refine (hasFDerivAt_pi (𝕜 := ℝ)
        (φ := fun i : Fin (Shape.size s) => fun x : CtxVec Γ => max (aCLM x i) (bCLM x i))
        (φ' := fun i : Fin (Shape.size s) =>
          if aCLM xV i > bCLM xV i then (evalCLM (n := Shape.size s) i).comp aCLM
          else (evalCLM (n := Shape.size s) i).comp bCLM)
        (x := xV)).2 ?_
      intro i
      simpa using hcoord i
    have he' :
        HasFDerivAt (fun g : Fin (Shape.size s) → ℝ => (euclideanEquiv (Shape.size s)).symm g)
          ((euclideanEquiv (Shape.size s)).symm.toContinuousLinearMap)
          (fun i : Fin (Shape.size s) => max (aCLM xV i) (bCLM xV i)) :=
      (ContinuousLinearMap.hasFDerivAt (euclideanEquiv (Shape.size s)).symm.toContinuousLinearMap)
    have hcomp := he'.comp xV hpi
    have hEq :
        (Node.forwardVec (Γ := Γ) (τ := s) (maxElem (Γ := Γ) (s := s) a b)) =
          (fun x : CtxVec Γ =>
            vecOfFun (n := Shape.size s) fun i : Fin (Shape.size s) => max (aCLM x i) (bCLM x i)) :=
              by
      funext x
      ext i
      simp [maxElem, Node.forwardVec_ofVec, vecOfFun, aCLM, bCLM]
    exact hcomp.congr_of_eventuallyEq hEq.eventuallyEq
  · intro dxV
    ext i
    have hne : CtxVec.get (Γ := Γ) (s := s) a xV i ≠ CtxVec.get (Γ := Γ) (s := s) b xV i := hneq i
    have hlt : CtxVec.get (Γ := Γ) (s := s) a xV i < CtxVec.get (Γ := Γ) (s := s) b xV i ∨
        CtxVec.get (Γ := Γ) (s := s) a xV i > CtxVec.get (Γ := Γ) (s := s) b xV i := lt_or_gt_of_ne
          hne
    cases hlt with
    | inr hgt =>
        simp [maxElem, Node.jvpVec_ofVec, hgt, vecOfFun, aCLM, bCLM, CtxVec.getCLM_apply,
          ContinuousLinearMap.comp_apply, evalCLM_apply]
    | inl hlt =>
        have hn : ¬ CtxVec.get (Γ := Γ) (s := s) a xV i > CtxVec.get (Γ := Γ) (s := s) b xV i :=
          (le_of_lt hlt).not_gt
        have hgt : CtxVec.get (Γ := Γ) (s := s) b xV i > CtxVec.get (Γ := Γ) (s := s) a xV i := hlt
        simp [maxElem, Node.jvpVec_ofVec, hn, vecOfFun, aCLM, bCLM, CtxVec.getCLM_apply,
          ContinuousLinearMap.comp_apply, evalCLM_apply]

/-- Pointwise `NodeFDerivCorrectAt` for `min_elem`, assuming there are no ties. -/
def minElemFderivAt {Γ : List Shape} {s : Shape} (a b : Idx Γ s) (xV : CtxVec Γ)
    (hneq : ∀ i : Fin (Shape.size s), CtxVec.get (Γ := Γ) (s := s) a xV i ≠ CtxVec.get (Γ := Γ) (s
      := s) b xV i) :
    NodeFDerivCorrectAt (minElem (Γ := Γ) (s := s) a b) xV := by
  classical
  let n : Nat := Shape.size s
  let aCLM : CtxVec Γ →L[ℝ] Vec n := CtxVec.getCLM (Γ := Γ) (s := s) a
  let bCLM : CtxVec Γ →L[ℝ] Vec n := CtxVec.getCLM (Γ := Γ) (s := s) b
  have ha0 : HasFDerivAt (fun x : CtxVec Γ => aCLM x) aCLM xV := aCLM.hasFDerivAt (x := xV)
  have hb0 : HasFDerivAt (fun x : CtxVec Γ => bCLM x) bCLM xV := bCLM.hasFDerivAt (x := xV)
  have hcoord :
      ∀ i : Fin n,
        HasFDerivAt (fun x : CtxVec Γ => min (aCLM x i) (bCLM x i))
          (if aCLM xV i < bCLM xV i then (evalCLM (n := n) i).comp aCLM else (evalCLM (n := n)
            i).comp bCLM) xV := by
    intro i
    have haCoord : aCLM xV i = CtxVec.get (Γ := Γ) (s := s) a xV i := by
      dsimp [aCLM]
      exact congrArg (fun v : Vec n => v i) (CtxVec.getCLM_apply (Γ := Γ) (s := s) a xV)
    have hbCoord : bCLM xV i = CtxVec.get (Γ := Γ) (s := s) b xV i := by
      dsimp [bCLM]
      exact congrArg (fun v : Vec n => v i) (CtxVec.getCLM_apply (Γ := Γ) (s := s) b xV)
    have ha_i :
        HasFDerivAt (fun x : CtxVec Γ => aCLM x i) ((evalCLM (n := n) i).comp aCLM) xV := by
      have houter : HasFDerivAt (fun v : Vec n => (evalCLM (n := n) i) v) (evalCLM (n := n) i) (aCLM
        xV) :=
        (evalCLM (n := n) i).hasFDerivAt (x := aCLM xV)
      have hcomp := houter.comp xV ha0
      simpa [ContinuousLinearMap.comp_apply, evalCLM_apply] using hcomp
    have hb_i :
        HasFDerivAt (fun x : CtxVec Γ => bCLM x i) ((evalCLM (n := n) i).comp bCLM) xV := by
      have houter : HasFDerivAt (fun v : Vec n => (evalCLM (n := n) i) v) (evalCLM (n := n) i) (bCLM
        xV) :=
        (evalCLM (n := n) i).hasFDerivAt (x := bCLM xV)
      have hcomp := houter.comp xV hb0
      simpa [ContinuousLinearMap.comp_apply, evalCLM_apply] using hcomp
    have hcmp :
        CtxVec.get (Γ := Γ) (s := s) a xV i < CtxVec.get (Γ := Γ) (s := s) b xV i ∨
          CtxVec.get (Γ := Γ) (s := s) a xV i > CtxVec.get (Γ := Γ) (s := s) b xV i :=
      lt_or_gt_of_ne (hneq i)
    cases hcmp with
    | inl hlt0 =>
        have hlt : aCLM xV i < bCLM xV i := by simpa [haCoord, hbCoord] using hlt0
        simpa [hlt, if_pos hlt] using
          (hasFDerivAt_min_of_lt (f := fun x : CtxVec Γ => aCLM x i) (g := fun x : CtxVec Γ => bCLM
            x i) ha_i hb_i hlt)
    | inr hgt0 =>
        have hn0 : ¬ CtxVec.get (Γ := Γ) (s := s) a xV i < CtxVec.get (Γ := Γ) (s := s) b xV i :=
          not_lt_of_gt hgt0
        have hn : ¬ aCLM xV i < bCLM xV i := by simpa [haCoord, hbCoord] using hn0
        have hlt : bCLM xV i < aCLM xV i := by simpa [haCoord, hbCoord] using (show CtxVec.get (Γ :=
          Γ) (s := s) b xV i < CtxVec.get (Γ := Γ) (s := s) a xV i from hgt0)
        have h' :
            HasFDerivAt (fun x : CtxVec Γ => min (bCLM x i) (aCLM x i))
              ((evalCLM (n := n) i).comp bCLM) xV :=
          hasFDerivAt_min_of_lt (f := fun x : CtxVec Γ => bCLM x i) (g := fun x : CtxVec Γ => aCLM x
            i) hb_i ha_i hlt
        simpa [hn, if_neg hn, min_comm] using h'
  refine
    { deriv :=
        (euclideanEquiv n).symm.toContinuousLinearMap.comp <|
          ContinuousLinearMap.pi fun i : Fin n =>
            if aCLM xV i < bCLM xV i then (evalCLM (n := n) i).comp aCLM else (evalCLM (n := n)
              i).comp bCLM
      hasFDerivAt := ?_
      jvp_eq := ?_ }
  · have hpi :
        HasFDerivAt
          (fun x : CtxVec Γ => fun i : Fin n => min (aCLM x i) (bCLM x i))
          (ContinuousLinearMap.pi fun i : Fin n =>
            if aCLM xV i < bCLM xV i then (evalCLM (n := n) i).comp aCLM else (evalCLM (n := n)
              i).comp bCLM)
          xV := by
      refine (hasFDerivAt_pi (𝕜 := ℝ)
        (φ := fun i : Fin n => fun x : CtxVec Γ => min (aCLM x i) (bCLM x i))
        (φ' := fun i : Fin n =>
          if aCLM xV i < bCLM xV i then (evalCLM (n := n) i).comp aCLM else (evalCLM (n := n)
            i).comp bCLM)
        (x := xV)).2 ?_
      intro i
      simpa using hcoord i
    have he' :
        HasFDerivAt (fun g : Fin n → ℝ => (euclideanEquiv n).symm g) ((euclideanEquiv
          n).symm.toContinuousLinearMap)
          (fun i : Fin n => min (aCLM xV i) (bCLM xV i)) :=
      (ContinuousLinearMap.hasFDerivAt (euclideanEquiv n).symm.toContinuousLinearMap)
    have hcomp := he'.comp xV hpi
    have hEq :
        (Node.forwardVec (Γ := Γ) (τ := s) (minElem (Γ := Γ) (s := s) a b)) =
          (fun x : CtxVec Γ =>
            vecOfFun (n := n) fun i : Fin n => min (aCLM x i) (bCLM x i)) := by
      funext x
      ext i
      have haCoord : aCLM x i = CtxVec.get (Γ := Γ) (s := s) a x i := by
        dsimp [aCLM]
        exact congrArg (fun v : Vec n => v i) (CtxVec.getCLM_apply (Γ := Γ) (s := s) a x)
      have hbCoord : bCLM x i = CtxVec.get (Γ := Γ) (s := s) b x i := by
        dsimp [bCLM]
        exact congrArg (fun v : Vec n => v i) (CtxVec.getCLM_apply (Γ := Γ) (s := s) b x)
      simp [minElem, Node.forwardVec_ofVec, vecOfFun, aCLM, bCLM, haCoord, hbCoord]
    exact hcomp.congr_of_eventuallyEq hEq.eventuallyEq
  · intro dxV
    ext i
    have hne : CtxVec.get (Γ := Γ) (s := s) a xV i ≠ CtxVec.get (Γ := Γ) (s := s) b xV i := hneq i
    have hlt : CtxVec.get (Γ := Γ) (s := s) a xV i < CtxVec.get (Γ := Γ) (s := s) b xV i ∨
        CtxVec.get (Γ := Γ) (s := s) a xV i > CtxVec.get (Γ := Γ) (s := s) b xV i := lt_or_gt_of_ne
          hne
    cases hlt with
    | inl hlt =>
        have haCoord : (aCLM xV).ofLp i = (CtxVec.get (Γ := Γ) (s := s) a xV).ofLp i := by
          dsimp [aCLM]
          exact getCLM_apply_ofLp (Γ := Γ) (s := s) a xV i
        have hbCoord : (bCLM xV).ofLp i = (CtxVec.get (Γ := Γ) (s := s) b xV).ofLp i := by
          dsimp [bCLM]
          exact getCLM_apply_ofLp (Γ := Γ) (s := s) b xV i
        have hltCLM : (aCLM xV).ofLp i < (bCLM xV).ofLp i := by
          simpa [haCoord, hbCoord] using hlt
        have hR :
            ((((euclideanEquiv n).symm.toContinuousLinearMap).comp
                    (ContinuousLinearMap.pi fun j : Fin n =>
                      if (aCLM xV).ofLp j < (bCLM xV).ofLp j then (evalCLM (n := n) j).comp aCLM
                      else (evalCLM (n := n) j).comp bCLM)) dxV).ofLp i
              =
              ((evalCLM (n := n) i).comp aCLM) dxV := by
          simp [ContinuousLinearMap.comp_apply, hltCLM]
        have hGet :
            (CtxVec.get (Γ := Γ) (s := s) a dxV).ofLp i = ((evalCLM (n := n) i).comp aCLM) dxV := by
          have hGet' :
              (aCLM dxV).ofLp i = (CtxVec.get (Γ := Γ) (s := s) a dxV).ofLp i := by
            dsimp [aCLM]
            exact getCLM_apply_ofLp (Γ := Γ) (s := s) a dxV i
          calc
            (CtxVec.get (Γ := Γ) (s := s) a dxV).ofLp i
                = (aCLM dxV).ofLp i := by simpa using hGet'.symm
            _   = ((evalCLM (n := n) i).comp aCLM) dxV := by
                  simp [ContinuousLinearMap.comp_apply, evalCLM_apply]
        -- goal is a coordinate identity; finish by rewriting the RHS via `hR`.
        have hjvp :
            ((minElem (Γ := Γ) (s := s) a b).jvpVec xV dxV).ofLp i =
              (CtxVec.get (Γ := Γ) (s := s) a dxV).ofLp i := by
          simp [minElem, Node.jvpVec_ofVec, hlt]
        exact hjvp.trans (hGet.trans hR.symm)
    | inr hgt =>
        have hn : ¬ CtxVec.get (Γ := Γ) (s := s) a xV i < CtxVec.get (Γ := Γ) (s := s) b xV i :=
          not_lt_of_gt hgt
        have haCoord : (aCLM xV).ofLp i = (CtxVec.get (Γ := Γ) (s := s) a xV).ofLp i := by
          dsimp [aCLM]
          exact getCLM_apply_ofLp (Γ := Γ) (s := s) a xV i
        have hbCoord : (bCLM xV).ofLp i = (CtxVec.get (Γ := Γ) (s := s) b xV).ofLp i := by
          dsimp [bCLM]
          exact getCLM_apply_ofLp (Γ := Γ) (s := s) b xV i
        have hnCLM : ¬ (aCLM xV).ofLp i < (bCLM xV).ofLp i := by
          simpa [haCoord, hbCoord] using hn
        have hR :
            ((((euclideanEquiv n).symm.toContinuousLinearMap).comp
                    (ContinuousLinearMap.pi fun j : Fin n =>
                      if (aCLM xV).ofLp j < (bCLM xV).ofLp j then (evalCLM (n := n) j).comp aCLM
                      else (evalCLM (n := n) j).comp bCLM)) dxV).ofLp i
              =
              ((evalCLM (n := n) i).comp bCLM) dxV := by
          simp [ContinuousLinearMap.comp_apply, hnCLM]
        have hGet :
            (CtxVec.get (Γ := Γ) (s := s) b dxV).ofLp i = ((evalCLM (n := n) i).comp bCLM) dxV := by
          have hGet' :
              (bCLM dxV).ofLp i = (CtxVec.get (Γ := Γ) (s := s) b dxV).ofLp i := by
            dsimp [bCLM]
            exact getCLM_apply_ofLp (Γ := Γ) (s := s) b dxV i
          calc
            (CtxVec.get (Γ := Γ) (s := s) b dxV).ofLp i
                = (bCLM dxV).ofLp i := by simpa using hGet'.symm
            _   = ((evalCLM (n := n) i).comp bCLM) dxV := by
                  simp [ContinuousLinearMap.comp_apply, evalCLM_apply]
        have hjvp :
            ((minElem (Γ := Γ) (s := s) a b).jvpVec xV dxV).ofLp i =
              (CtxVec.get (Γ := Γ) (s := s) b dxV).ofLp i := by
          simp [minElem, Node.jvpVec_ofVec, hn]
        exact hjvp.trans (hGet.trans hR.symm)


end TapeNodes

end

end Autograd
end Proofs
