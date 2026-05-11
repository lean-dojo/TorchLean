/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Data.NNReal.Defs
public import NN.Proofs.RuntimeApprox.Scale.ScaleApprox

/-!
# ForwardScale

Forward scale propagation over SSA/DAG graphs.

This optional module mirrors `NN.Proofs.RuntimeApprox.Graph.ForwardApprox`, but for *scale bounds*
(nonnegative bounds on `linf_norm`) rather than eps error bounds.

PyTorch analogue: users often track activation magnitudes (for normalization, stability, or
analysis). Here we make that tracking explicit and compositional at the proof level.
-/

@[expose] public section


namespace Proofs
namespace RuntimeApprox

open Spec
open NN.MLTheory.Robustness.Spec
open scoped NNReal

noncomputable section

variable {α : Type}

/-- A forward node augmented with a scale bound function and a soundness lemma. -/
structure FwdNodeScale (toSpec : α → SpecScalar) (Γ : List Shape) (τ : Shape) extends
    FwdNode (α := α) toSpec Γ τ where
  scaleBound : BList Γ → TList α Γ → ℝ≥0
  scaleSound : ∀ (ctxS : TList SpecScalar Γ) (ctxR : TList α Γ) (epsCtx : EList Γ) (bCtx : BList Γ),
      approxCtx (α := α) toSpec ctxS ctxR epsCtx →
      scaleCtx (α := α) toSpec ctxS ctxR bCtx →
        scaleT (α := α) (toSpec := toSpec) (forwardSpec ctxS) (forwardRuntime ctxR) (scaleBound bCtx
          ctxR)

/-- Forward graph with scale-aware nodes. -/
inductive FwdGraphScale (toSpec : α → SpecScalar) (Γ : List Shape) : List Shape → Type where
  | nil : FwdGraphScale toSpec Γ []
  | snoc {ss : List Shape} {τ : Shape} :
      FwdGraphScale toSpec Γ ss →
      FwdNodeScale (α := α) toSpec (Γ := Γ ++ ss) τ →
      FwdGraphScale toSpec Γ (ss ++ [τ])

namespace FwdGraphScale

variable {toSpec : α → SpecScalar}

/-- Forget the scale annotations on nodes, producing an ordinary `FwdGraph`. -/
def toFwdGraph {Γ : List Shape} {ss : List Shape} :
    FwdGraphScale (α := α) toSpec Γ ss → FwdGraph (α := α) toSpec Γ ss
  | .nil => .nil
  | .snoc g node => .snoc (toFwdGraph g) node.toFwdNode

/-- Evaluate the forward pass on spec values, returning the extended context `Γ ++ ss`. -/
def evalSpec {Γ : List Shape} {ss : List Shape} (g : FwdGraphScale (α := α) toSpec Γ ss)
    (x : TList SpecScalar Γ) : TList SpecScalar (Γ ++ ss) :=
  FwdGraph.evalSpec (α := α) (toSpec := toSpec) (Γ := Γ) (ss := ss) (toFwdGraph (α := α) g) x

/-- Evaluate the forward pass on runtime values, returning the extended context `Γ ++ ss`. -/
def evalRuntime {Γ : List Shape} {ss : List Shape} (g : FwdGraphScale (α := α) toSpec Γ ss)
    (x : TList α Γ) : TList α (Γ ++ ss) :=
  FwdGraph.evalRuntime (α := α) (toSpec := toSpec) (Γ := Γ) (ss := ss) (toFwdGraph (α := α) g) x

/-- Forward-pass error bounds for all intermediate nodes, computed from input bounds `epsIn`. -/
def evalBounds {Γ : List Shape} {ss : List Shape} (g : FwdGraphScale (α := α) toSpec Γ ss)
    (epsIn : EList Γ) (xR : TList α Γ) : EList (Γ ++ ss) :=
  FwdGraph.evalBounds (α := α) (toSpec := toSpec) (Γ := Γ) (ss := ss) (toFwdGraph (α := α) g) epsIn
    xR

/-- Forward-pass scale bounds for all intermediate nodes, computed from input bounds `bIn`. -/
def evalScales {Γ : List Shape} {ss : List Shape} (g : FwdGraphScale (α := α) toSpec Γ ss)
    (bIn : BList Γ) (xR : TList α Γ) : BList (Γ ++ ss) :=
  match g with
  | .nil =>
      let h : Γ = Γ ++ [] := (List.append_nil Γ).symm
      BList.cast (ss₁ := Γ) (ss₂ := Γ ++ []) h bIn
  | .snoc (ss := ssPrev) (τ := τ) g node =>
      let bPrev := evalScales (Γ := Γ) (ss := ssPrev) g bIn xR
      let ctxR := evalRuntime (Γ := Γ) (ss := ssPrev) g xR
      let b := node.scaleBound bPrev ctxR
      let hAssoc : (Γ ++ ssPrev) ++ [τ] = Γ ++ (ssPrev ++ [τ]) := List.append_assoc Γ ssPrev [τ]
      BList.cast (ss₁ := (Γ ++ ssPrev) ++ [τ]) (ss₂ := Γ ++ (ssPrev ++ [τ])) hAssoc
        (BList.snoc (ss := Γ ++ ssPrev) (τ := τ) bPrev b)

theorem eval_scale {Γ : List Shape} {ss : List Shape} (g : FwdGraphScale (α := α) toSpec Γ ss) :
    ∀ (xS : TList SpecScalar Γ) (xR : TList α Γ) (epsIn : EList Γ) (bIn : BList Γ),
      approxCtx (α := α) toSpec xS xR epsIn →
      scaleCtx (α := α) toSpec xS xR bIn →
        scaleCtx (α := α) toSpec
          (evalSpec (α := α) (toSpec := toSpec) (Γ := Γ) (ss := ss) g xS)
          (evalRuntime (α := α) (toSpec := toSpec) (Γ := Γ) (ss := ss) g xR)
          (evalScales (α := α) (toSpec := toSpec) (Γ := Γ) (ss := ss) g bIn xR) := by
  intro xS xR epsIn bIn hε hB
  induction g generalizing xS xR epsIn bIn with
  | nil =>
      -- `eval*` are casts along `Γ = Γ ++ []`.
      simpa [evalSpec, evalRuntime, evalScales] using
        (scaleCtx_cast (α := α) (toSpec := toSpec) (h := (List.append_nil Γ).symm) hB)
  | snoc g node ih =>
      rename_i ssPrev τ
      -- IH gives scale for the previous context.
      have hBPrev :
          scaleCtx (α := α) toSpec
            (evalSpec (α := α) (toSpec := toSpec) (Γ := Γ) (ss := ssPrev) g xS)
            (evalRuntime (α := α) (toSpec := toSpec) (Γ := Γ) (ss := ssPrev) g xR)
            (evalScales (α := α) (toSpec := toSpec) (Γ := Γ) (ss := ssPrev) g bIn xR) :=
        ih xS xR epsIn bIn hε hB

      let ctxS := evalSpec (α := α) (toSpec := toSpec) (Γ := Γ) (ss := ssPrev) g xS
      let ctxR := evalRuntime (α := α) (toSpec := toSpec) (Γ := Γ) (ss := ssPrev) g xR
      let epsPrev := evalBounds (α := α) (toSpec := toSpec) (Γ := Γ) (ss := ssPrev) g epsIn xR
      let bPrev := evalScales (α := α) (toSpec := toSpec) (Γ := Γ) (ss := ssPrev) g bIn xR

      -- Approx for the node context `Γ ++ ssPrev`.
      have hεPrev :
          approxCtx (α := α) toSpec ctxS ctxR epsPrev :=
        FwdGraph.eval_approx (α := α) (toSpec := toSpec) (Γ := Γ) (ss := ssPrev)
          (toFwdGraph (α := α) g) xS xR epsIn hε

      have hb :
          scaleT (α := α) (toSpec := toSpec)
            (node.forwardSpec ctxS)
            (node.forwardRuntime ctxR)
            (node.scaleBound bPrev ctxR) :=
        node.scaleSound ctxS ctxR epsPrev bPrev (by simpa [ctxS, ctxR, epsPrev] using hεPrev)
          (by simpa [ctxS, ctxR, bPrev] using hBPrev)

      -- Extend the context scale predicate with the new node output.
      have hSnoc :
          scaleCtx (α := α) toSpec
            (TList.snoc (α := SpecScalar) (ss := Γ ++ ssPrev) ctxS (node.forwardSpec ctxS))
            (TList.snoc (α := α) (ss := Γ ++ ssPrev) ctxR (node.forwardRuntime ctxR))
            (BList.snoc (ss := Γ ++ ssPrev) (τ := τ) bPrev (node.scaleBound bPrev ctxR)) :=
        scaleCtx_snoc (α := α) (toSpec := toSpec) (hx := by simpa [ctxS, ctxR, bPrev] using hBPrev)
          hb

      -- Cast to match the `Γ ++ (ssPrev ++ [τ])` shape.
      simpa [evalSpec, evalRuntime, evalScales, ctxS, ctxR, epsPrev, bPrev, List.append_assoc] using
        (scaleCtx_cast (α := α) (toSpec := toSpec) (h := List.append_assoc Γ ssPrev [τ]) hSnoc)

end FwdGraphScale

end

end RuntimeApprox
end Proofs
