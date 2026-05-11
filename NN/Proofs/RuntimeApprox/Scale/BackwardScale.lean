/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Data.NNReal.Defs
public import NN.Proofs.RuntimeApprox.Graph.BackwardApprox
public import NN.Proofs.RuntimeApprox.Scale.ForwardScale

/-!
# BackwardScale

Backward (reverse-mode) scale propagation.

This optional module mirrors `NN.Proofs.RuntimeApprox.Graph.BackwardApprox`, but for *scale bounds*
(nonnegative bounds on `linf_norm`) rather than eps error bounds.

Use it alongside the backward approximation graph when you want to derive abs+rel tolerances for
gradients/cotangents from both an eps error bound and a propagated magnitude bound.
-/

@[expose] public section


namespace Proofs
namespace RuntimeApprox

open Spec
open NN.MLTheory.Robustness.Spec
open scoped NNReal

noncomputable section

variable {α : Type}

/-- Soundness condition for accumulating scale bounds under addition in a context. -/
def AddScaleSound (toSpec : α → SpecScalar) [Add α]
    (addBound : {Δ : List Shape} → BList Δ → BList Δ → TList α Δ → TList α Δ → BList Δ) : Prop :=
  ∀ {Δ : List Shape},
    ∀ xS yS : TList SpecScalar Δ,
    ∀ xR yR : TList α Δ,
    ∀ bX bY : BList Δ,
      scaleCtx (α := α) toSpec xS xR bX →
      scaleCtx (α := α) toSpec yS yR bY →
        scaleCtx (α := α) toSpec (TList.add (α := SpecScalar) xS yS) (TList.add (α := α) xR yR)
          (addBound bX bY xR yR)

/-- A reverse node augmented with forward+VJP scale bounds. -/
structure RevNodeScale (toSpec : α → SpecScalar) (Γ : List Shape) (τ : Shape) extends
    RevNode (α := α) toSpec Γ τ where
  fwdScaleBound : BList Γ → TList α Γ → ℝ≥0
  fwdScaleSound : ∀ (ctxS : TList SpecScalar Γ) (ctxR : TList α Γ) (epsCtx : EList Γ) (bCtx : BList
    Γ),
      approxCtx (α := α) toSpec ctxS ctxR epsCtx →
      scaleCtx (α := α) toSpec ctxS ctxR bCtx →
        scaleT (α := α) (toSpec := toSpec) (forwardSpec ctxS) (forwardRuntime ctxR) (fwdScaleBound
          bCtx ctxR)
  vjpScaleBound : BList Γ → TList α Γ → ℝ≥0 → Tensor α τ → BList Γ
  vjpScaleSound : ∀ (ctxS : TList SpecScalar Γ) (ctxR : TList α Γ) (epsCtx : EList Γ) (bCtx : BList
    Γ)
      (δS : SpecTensor τ) (δR : Tensor α τ) (bδ : ℝ≥0),
      approxCtx (α := α) toSpec ctxS ctxR epsCtx →
      scaleCtx (α := α) toSpec ctxS ctxR bCtx →
      scaleT (α := α) (toSpec := toSpec) δS δR bδ →
        scaleCtx (α := α) toSpec (vjpSpec ctxS δS) (vjpRuntime ctxR δR) (vjpScaleBound bCtx ctxR bδ
          δR)

/-- Reverse-mode graph with scale-aware nodes. -/
inductive RevGraphScale (toSpec : α → SpecScalar) (Γ : List Shape) : List Shape → Type where
  | nil : RevGraphScale toSpec Γ []
  | snoc {ss : List Shape} {τ : Shape} :
      RevGraphScale toSpec Γ ss →
      RevNodeScale (α := α) toSpec (Γ := Γ ++ ss) τ →
      RevGraphScale toSpec Γ (ss ++ [τ])

namespace RevGraphScale

variable {toSpec : α → SpecScalar}

/-- Forget the scale annotations on nodes, producing an ordinary `RevGraph`. -/
def toRevGraph {Γ : List Shape} {ss : List Shape} :
    RevGraphScale (α := α) toSpec Γ ss → RevGraph (α := α) toSpec Γ ss
  | .nil => .nil
  | .snoc g node => .snoc (toRevGraph g) node.toRevNode

/-- Convert a `RevGraphScale` into a `FwdGraphScale` by dropping the reverse-mode payload. -/
def toFwdGraphScale {Γ : List Shape} {ss : List Shape} :
    RevGraphScale (α := α) toSpec Γ ss → FwdGraphScale (α := α) toSpec Γ ss
  | .nil => .nil
  | .snoc g node =>
      .snoc (toFwdGraphScale g)
        { toFwdNode := node.toFwdNode
          scaleBound := node.fwdScaleBound
          scaleSound := node.fwdScaleSound }

theorem toFwdGraph_toFwdGraphScale_eq {Γ : List Shape} {ss : List Shape}
    (g : RevGraphScale (α := α) toSpec Γ ss) :
    FwdGraphScale.toFwdGraph (toFwdGraphScale (α := α) (toSpec := toSpec) g) =
      RevGraph.toFwdGraph (toRevGraph (α := α) (toSpec := toSpec) g) := by
  induction g with
  | nil => rfl
  | snoc g node ih =>
      simp [toFwdGraphScale, toRevGraph, FwdGraphScale.toFwdGraph, RevGraph.toFwdGraph, ih]

/-- Evaluate the forward pass on spec values, returning the extended context `Γ ++ ss`. -/
def evalSpec {Γ : List Shape} {ss : List Shape} (g : RevGraphScale (α := α) toSpec Γ ss) (x : TList
  SpecScalar Γ) :
    TList SpecScalar (Γ ++ ss) :=
  FwdGraphScale.evalSpec (α := α) (toSpec := toSpec) (Γ := Γ) (ss := ss) (toFwdGraphScale (α := α)
    g) x

/-- Evaluate the forward pass on runtime values, returning the extended context `Γ ++ ss`. -/
def evalRuntime {Γ : List Shape} {ss : List Shape} (g : RevGraphScale (α := α) toSpec Γ ss) (x :
  TList α Γ) :
    TList α (Γ ++ ss) :=
  FwdGraphScale.evalRuntime (α := α) (toSpec := toSpec) (Γ := Γ) (ss := ss) (toFwdGraphScale (α :=
    α) g) x

@[simp] theorem evalSpec_eq_rev {Γ : List Shape} {ss : List Shape}
    (g : RevGraphScale (α := α) toSpec Γ ss) (x : TList SpecScalar Γ) :
    evalSpec (α := α) (toSpec := toSpec) (Γ := Γ) (ss := ss) g x =
      RevGraph.evalSpec (α := α) (toSpec := toSpec) (Γ := Γ) (ss := ss) (toRevGraph (α := α) g) x :=
        by
  -- `RevGraph.evalSpec` is `FwdGraph.evalSpec` of `RevGraph.toFwdGraph`.
  simpa [evalSpec, RevGraph.evalSpec, FwdGraphScale.evalSpec] using congrArg
    (fun fg => FwdGraph.evalSpec (α := α) (toSpec := toSpec) (Γ := Γ) (ss := ss) fg x)
    (toFwdGraph_toFwdGraphScale_eq (α := α) (toSpec := toSpec) (Γ := Γ) (ss := ss) g)

@[simp] theorem evalRuntime_eq_rev {Γ : List Shape} {ss : List Shape}
    (g : RevGraphScale (α := α) toSpec Γ ss) (x : TList α Γ) :
    evalRuntime (α := α) (toSpec := toSpec) (Γ := Γ) (ss := ss) g x =
      RevGraph.evalRuntime (α := α) (toSpec := toSpec) (Γ := Γ) (ss := ss) (toRevGraph (α := α) g) x
        := by
  simpa [evalRuntime, RevGraph.evalRuntime, FwdGraphScale.evalRuntime] using congrArg
    (fun fg => FwdGraph.evalRuntime (α := α) (toSpec := toSpec) (Γ := Γ) (ss := ss) fg x)
    (toFwdGraph_toFwdGraphScale_eq (α := α) (toSpec := toSpec) (Γ := Γ) (ss := ss) g)

/-- Forward-pass error bounds for all intermediate nodes, computed from input bounds `epsIn`. -/
def evalBounds {Γ : List Shape} {ss : List Shape} (g : RevGraphScale (α := α) toSpec Γ ss)
    (epsIn : EList Γ) (xR : TList α Γ) : EList (Γ ++ ss) :=
  RevGraph.evalBounds (α := α) (toSpec := toSpec) (Γ := Γ) (ss := ss) (toRevGraph (α := α) g) epsIn
    xR

/-- Forward-pass scale bounds for all intermediate nodes, computed from input bounds `bIn`. -/
def evalScales {Γ : List Shape} {ss : List Shape} (g : RevGraphScale (α := α) toSpec Γ ss)
    (bIn : BList Γ) (xR : TList α Γ) : BList (Γ ++ ss) :=
  FwdGraphScale.evalScales (α := α) (toSpec := toSpec) (Γ := Γ) (ss := ss) (toFwdGraphScale (α := α)
    g) bIn xR

theorem eval_scale {Γ : List Shape} {ss : List Shape} (g : RevGraphScale (α := α) toSpec Γ ss) :
    ∀ (xS : TList SpecScalar Γ) (xR : TList α Γ) (epsIn : EList Γ) (bIn : BList Γ),
      approxCtx (α := α) toSpec xS xR epsIn →
      scaleCtx (α := α) toSpec xS xR bIn →
        scaleCtx (α := α) toSpec
          (evalSpec (α := α) (toSpec := toSpec) (Γ := Γ) (ss := ss) g xS)
          (evalRuntime (α := α) (toSpec := toSpec) (Γ := Γ) (ss := ss) g xR)
          (evalScales (α := α) (toSpec := toSpec) (Γ := Γ) (ss := ss) g bIn xR) :=
  FwdGraphScale.eval_scale (α := α) (toSpec := toSpec) (Γ := Γ) (ss := ss) (toFwdGraphScale (α := α)
    g)

-- ---------------------------------------------------------------------------
-- Backprop scale bounds (analogous to `backpropBounds`)
-- ---------------------------------------------------------------------------

/-- Backpropagate scale bounds through a `RevGraphScale`, analogous to `RevGraph.backpropRuntime`.
  -/
def backpropScales {Γ : List Shape} {ss : List Shape} (g : RevGraphScale (α := α) toSpec Γ ss)
    [Add α]
    (bIn : BList Γ) (xR : TList α Γ) (bSeed : BList (Γ ++ ss)) (seedR : TList α (Γ ++ ss))
    (addBound : {Δ : List Shape} → BList Δ → BList Δ → TList α Δ → TList α Δ → BList Δ) : BList Γ :=
  match g with
  | .nil =>
      BList.cast (ss₁ := Γ ++ []) (ss₂ := Γ) (List.append_nil Γ) bSeed
  | .snoc (ss := ssPrev) (τ := τ) g node =>
      let assoc : (Γ ++ ssPrev) ++ [τ] = Γ ++ (ssPrev ++ [τ]) := List.append_assoc Γ ssPrev [τ]
      let seed' : TList α ((Γ ++ ssPrev) ++ [τ]) :=
        TList.cast (α := α) (ss₁ := Γ ++ (ssPrev ++ [τ])) (ss₂ := (Γ ++ ssPrev) ++ [τ]) assoc.symm
          seedR
      let bSeed' : BList ((Γ ++ ssPrev) ++ [τ]) :=
        BList.cast (ss₁ := Γ ++ (ssPrev ++ [τ])) (ss₂ := (Γ ++ ssPrev) ++ [τ]) assoc.symm bSeed
      let bSeedPrev : BList (Γ ++ ssPrev) := (BList.unsnoc (ss := Γ ++ ssPrev) (τ := τ) bSeed').1
      let bSeedOut : ℝ≥0 := (BList.unsnoc (ss := Γ ++ ssPrev) (τ := τ) bSeed').2
      let seedPrev : TList α (Γ ++ ssPrev) := (TList.unsnoc (α := α) (ss := Γ ++ ssPrev) (τ := τ)
        seed').1
      let seedOut : Tensor α τ := (TList.unsnoc (α := α) (ss := Γ ++ ssPrev) (τ := τ) seed').2
      let ctxR := evalRuntime (α := α) (toSpec := toSpec) (Γ := Γ) (ss := ssPrev) g xR
      let bCtx := evalScales (α := α) (toSpec := toSpec) (Γ := Γ) (ss := ssPrev) g bIn xR
      let contrib := node.vjpRuntime ctxR seedOut
      let bContrib := node.vjpScaleBound bCtx ctxR bSeedOut seedOut
      let seedPrev' := TList.add (α := α) seedPrev contrib
      let bSeedPrev' := addBound (Δ := Γ ++ ssPrev) bSeedPrev bContrib seedPrev contrib
      backpropScales g bIn xR bSeedPrev' seedPrev' addBound

theorem backprop_scale {Γ : List Shape} {ss : List Shape} (g : RevGraphScale (α := α) toSpec Γ ss)
    [Add α]
    (addBound : {Δ : List Shape} → BList Δ → BList Δ → TList α Δ → TList α Δ → BList Δ)
    (addSound : AddScaleSound (α := α) toSpec addBound) :
    ∀ (xS : TList SpecScalar Γ) (xR : TList α Γ) (epsIn : EList Γ) (bIn : BList Γ)
      (seedS : TList SpecScalar (Γ ++ ss)) (seedR : TList α (Γ ++ ss)) (bSeed : BList (Γ ++ ss)),
      approxCtx (α := α) toSpec xS xR epsIn →
      scaleCtx (α := α) toSpec xS xR bIn →
      scaleCtx (α := α) toSpec seedS seedR bSeed →
        scaleCtx (α := α) toSpec
          (RevGraph.backpropSpec (toSpec := toSpec) (Γ := Γ) (ss := ss) (toRevGraph (α := α) g) xS
            seedS)
          (RevGraph.backpropRuntime (toSpec := toSpec) (Γ := Γ) (ss := ss) (toRevGraph (α := α) g)
            xR seedR)
          (backpropScales (α := α) (toSpec := toSpec) (Γ := Γ) (ss := ss) g bIn xR bSeed seedR
            addBound) := by
  intro xS xR epsIn bIn seedS seedR bSeed hx hB hinSeed
  revert xS xR epsIn bIn seedS seedR bSeed hx hB hinSeed
  induction g with
  | nil =>
      intro xS xR epsIn bIn seedS seedR bSeed hx hB hinSeed
      -- backprop is just a cast along `Γ ++ [] = Γ`.
      simpa [RevGraph.backpropSpec, RevGraph.backpropRuntime, backpropScales,
        RevGraphScale.toRevGraph] using
        (scaleCtx_cast (α := α) (toSpec := toSpec) (h := (List.append_nil Γ)) hinSeed)
  | snoc g node ih =>
      intro xS xR epsIn bIn seedS seedR bSeed hx hB hinSeed
      rename_i ssPrev τ

      -- forward scale for the node context `Γ ++ ssPrev`
      have hctxScale :
          scaleCtx (α := α) toSpec
            (evalSpec (α := α) (toSpec := toSpec) (Γ := Γ) (ss := ssPrev) g xS)
            (evalRuntime (α := α) (toSpec := toSpec) (Γ := Γ) (ss := ssPrev) g xR)
            (evalScales (α := α) (toSpec := toSpec) (Γ := Γ) (ss := ssPrev) g bIn xR) :=
        eval_scale (α := α) (toSpec := toSpec) (Γ := Γ) (ss := ssPrev) g xS xR epsIn bIn hx hB

      -- forward approx for the node context `Γ ++ ssPrev` (needed for `vjpScaleSound`).
      have hctxApprox :
          approxCtx (α := α) toSpec
            (RevGraph.evalSpec (α := α) (toSpec := toSpec) (Γ := Γ) (ss := ssPrev) (toRevGraph (α :=
              α) g) xS)
            (RevGraph.evalRuntime (α := α) (toSpec := toSpec) (Γ := Γ) (ss := ssPrev) (toRevGraph (α
              := α) g) xR)
            (RevGraph.evalBounds (α := α) (toSpec := toSpec) (Γ := Γ) (ss := ssPrev) (toRevGraph (α
              := α) g) epsIn xR) :=
        RevGraph.eval_approx (α := α) (toSpec := toSpec) (Γ := Γ) (ss := ssPrev) (toRevGraph (α :=
          α) g) xS xR epsIn hx

      -- Cast seed to `(Γ ++ ssPrev) ++ [τ]`, then split.
      let assoc : (Γ ++ ssPrev) ++ [τ] = Γ ++ (ssPrev ++ [τ]) := List.append_assoc Γ ssPrev [τ]
      let seedS' : TList SpecScalar ((Γ ++ ssPrev) ++ [τ]) :=
        TList.cast (α := SpecScalar) (ss₁ := Γ ++ (ssPrev ++ [τ])) (ss₂ := (Γ ++ ssPrev) ++ [τ])
          assoc.symm seedS
      let seedR' : TList α ((Γ ++ ssPrev) ++ [τ]) :=
        TList.cast (α := α) (ss₁ := Γ ++ (ssPrev ++ [τ])) (ss₂ := (Γ ++ ssPrev) ++ [τ]) assoc.symm
          seedR
      let bSeed' : BList ((Γ ++ ssPrev) ++ [τ]) :=
        BList.cast (ss₁ := Γ ++ (ssPrev ++ [τ])) (ss₂ := (Γ ++ ssPrev) ++ [τ]) assoc.symm bSeed

      have hseed' : scaleCtx (α := α) toSpec seedS' seedR' bSeed' := by
        simpa [seedS', seedR', bSeed'] using
          (scaleCtx_cast (α := α) (toSpec := toSpec) (h := assoc.symm) hinSeed)

      let seedPrevS : TList SpecScalar (Γ ++ ssPrev) :=
        (TList.unsnoc (α := SpecScalar) (ss := Γ ++ ssPrev) (τ := τ) seedS').1
      let seedOutS : SpecTensor τ :=
        (TList.unsnoc (α := SpecScalar) (ss := Γ ++ ssPrev) (τ := τ) seedS').2
      let seedPrevR : TList α (Γ ++ ssPrev) :=
        (TList.unsnoc (α := α) (ss := Γ ++ ssPrev) (τ := τ) seedR').1
      let seedOutR : Tensor α τ :=
        (TList.unsnoc (α := α) (ss := Γ ++ ssPrev) (τ := τ) seedR').2
      let bSeedPrev : BList (Γ ++ ssPrev) :=
        (BList.unsnoc (ss := Γ ++ ssPrev) (τ := τ) bSeed').1
      let bSeedOut : ℝ≥0 :=
        (BList.unsnoc (ss := Γ ++ ssPrev) (τ := τ) bSeed').2

      have hseedSplit :
          scaleCtx (α := α) toSpec seedPrevS seedPrevR bSeedPrev ∧
            scaleT (α := α) (toSpec := toSpec) seedOutS seedOutR bSeedOut := by
        simpa [seedPrevS, seedPrevR, bSeedPrev, seedOutS, seedOutR, bSeedOut] using
          (scaleCtx_unsnoc (α := α) (toSpec := toSpec) (ss := Γ ++ ssPrev) (τ := τ)
            (xS := seedS') (xR := seedR') (bs := bSeed') hseed')

      have hseedPrev : scaleCtx (α := α) toSpec seedPrevS seedPrevR bSeedPrev := hseedSplit.1
      have hseedOut : scaleT (α := α) (toSpec := toSpec) seedOutS seedOutR bSeedOut := hseedSplit.2

      let ctxS := evalSpec (α := α) (toSpec := toSpec) (Γ := Γ) (ss := ssPrev) g xS
      let ctxR := evalRuntime (α := α) (toSpec := toSpec) (Γ := Γ) (ss := ssPrev) g xR
      let epsCtx := evalBounds (α := α) (toSpec := toSpec) (Γ := Γ) (ss := ssPrev) g epsIn xR
      let bCtx := evalScales (α := α) (toSpec := toSpec) (Γ := Γ) (ss := ssPrev) g bIn xR

      have hctxApprox' : approxCtx (α := α) toSpec ctxS ctxR epsCtx := by
        simpa [ctxS, ctxR, epsCtx, evalBounds] using hctxApprox

      have hcontrib :
          scaleCtx (α := α) toSpec (node.vjpSpec ctxS seedOutS) (node.vjpRuntime ctxR seedOutR)
            (node.vjpScaleBound bCtx ctxR bSeedOut seedOutR) :=
        node.vjpScaleSound ctxS ctxR epsCtx bCtx seedOutS seedOutR bSeedOut
          hctxApprox'
          (by simpa [ctxS, ctxR, bCtx] using hctxScale)
          (by simpa [seedOutS, seedOutR, bSeedOut] using hseedOut)

      -- Add the contribution into the previous seed, and update the scale bound via `addSound`.
      let seedPrevS' := TList.add (α := SpecScalar) seedPrevS (node.vjpSpec ctxS seedOutS)
      let seedPrevR' := TList.add (α := α) seedPrevR (node.vjpRuntime ctxR seedOutR)
      let bSeedPrev' := addBound bSeedPrev (node.vjpScaleBound bCtx ctxR bSeedOut seedOutR)
        seedPrevR (node.vjpRuntime ctxR seedOutR)

      have hseedPrev' : scaleCtx (α := α) toSpec seedPrevS' seedPrevR' bSeedPrev' :=
        addSound seedPrevS (node.vjpSpec ctxS seedOutS) seedPrevR (node.vjpRuntime ctxR seedOutR)
          bSeedPrev (node.vjpScaleBound bCtx ctxR bSeedOut seedOutR) hseedPrev hcontrib

      -- Recurse.
      have := ih xS xR epsIn bIn seedPrevS' seedPrevR' bSeedPrev' hx hB hseedPrev'
      simpa [RevGraphScale.toRevGraph, RevGraph.backpropSpec, RevGraph.backpropRuntime,
        backpropScales,
        assoc, ctxS, ctxR, epsCtx, bCtx, seedS', seedR', bSeed', seedPrevS, seedPrevR, bSeedPrev,
        seedOutS, seedOutR, bSeedOut, seedPrevS', seedPrevR', bSeedPrev', evalBounds, evalScales,
          TList.add] using this

end RevGraphScale

end

end RuntimeApprox
end Proofs
