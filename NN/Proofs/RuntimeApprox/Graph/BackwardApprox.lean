/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.RuntimeApprox.Graph.ForwardApprox
public import NN.Spec.Core.TensorOps

/-!
# BackwardApprox

Reverse-mode (backward) runtime→spec approximation framework.

This is the backward analogue of `NN/Proofs/RuntimeApprox/Graph/ForwardApprox.lean`.

It models a tape/SSA-style DAG where each node provides:
- a forward map (spec + runtime) with an explicit forward error bound (as in `FwdNode`);
- a local VJP (reverse-mode) map (spec + runtime) with an explicit VJP error bound.

The global theorem `RevGraph.backprop_approx` composes these local bounds over the whole graph,
analogously to `FwdGraph.eval_approx`.

Note: reverse-mode accumulation uses context-wise addition; the corresponding bound/soundness
for accumulation is an explicit parameter to the theorem (`addBound`, `addSound`).

## What you get
- `RevGraph.eval_approx`: a forward approximation theorem for the runtime/spec node contexts.
- `RevGraph.backprop_approx`: an end-to-end approximation theorem for reverse-mode accumulation
  (VJP-based backprop on the whole graph).

## Reading guide
1. `RevNode` / `RevGraph`: nodes carry both forward and VJP approximation data.
2. `RevGraph.eval_*`: reuse the forward theory by forgetting VJP data.
3. `RevGraph.backprop*`: executable reverse-mode accumulation + its explicit bound propagation.
4. `RevGraph.backprop_approx`: composes node-local VJP bounds and accumulation bounds over the
   whole graph (induction over the snoc-list).

## PyTorch correspondence / citations
This mirrors the structure of reverse-mode AD in PyTorch Autograd: a forward pass produces
intermediate values, and a backward pass propagates cotangents/gradients in reverse topological
order while accumulating contributions at shared nodes.
https://pytorch.org/docs/stable/autograd.html
-/

@[expose] public section


namespace Proofs
namespace RuntimeApprox

open Spec
open Tensor
open NN.MLTheory.Robustness.Spec

noncomputable section

namespace EList

/-- Componentwise addition of error lists (used when accumulating cotangent/gradient bounds). -/
def add : {ss : List Shape} → EList ss → EList ss → EList ss
  | [], .nil, .nil => .nil
  | _ :: ss, .cons a as, .cons b bs => .cons (a + b) (add (ss := ss) as bs)

end EList

variable {α : Type}

-- ---------------------------------------------------------------------------
-- Reverse nodes and graphs
-- ---------------------------------------------------------------------------

/-- A node with both forward and VJP approximation data. -/
structure RevNode (toSpec : α → SpecScalar) (Γ : List Shape) (τ : Shape) extends
    FwdNode (α := α) toSpec Γ τ where
  /-- Spec-level VJP: maps a context and an output cotangent into a context cotangent. -/
  vjpSpec : TList SpecScalar Γ → SpecTensor τ → TList SpecScalar Γ
  /-- Runtime VJP: same shape-level function on the runtime side. -/
  vjpRuntime : TList α Γ → Tensor α τ → TList α Γ
  /-- Explicit bound transformer for VJP: pushes bounds backward through this node. -/
  vjpBound : EList Γ → TList α Γ → SpecScalar → Tensor α τ → EList Γ
  /-- Soundness of the VJP bound: if context + output cotangent are approximated, so is the VJP. -/
  vjpSound : ∀ (ctxS : TList SpecScalar Γ) (ctxR : TList α Γ) (epsCtx : EList Γ)
      (δS : SpecTensor τ) (δR : Tensor α τ) (epsδ : SpecScalar),
      approxCtx (α := α) toSpec ctxS ctxR epsCtx →
      approxT (α := α) (toSpec := toSpec) δS δR epsδ →
        approxCtx (α := α) toSpec (vjpSpec ctxS δS) (vjpRuntime ctxR δR) (vjpBound epsCtx ctxR epsδ
          δR)

/--
Tape/SSA DAG with reverse-mode metadata (nodes appended in topological order).

Compared to `FwdGraph`, each node additionally carries a local VJP rule (`RevNode.vjp*`) plus an
explicit error transformer `vjpBound` and its soundness lemma `vjpSound`.
-/
inductive RevGraph (toSpec : α → SpecScalar) (Γ : List Shape) : List Shape → Type where
  | nil : RevGraph toSpec Γ []
  | snoc {ss : List Shape} {τ : Shape} :
      RevGraph toSpec Γ ss →
      RevNode (α := α) toSpec (Γ := Γ ++ ss) τ →
      RevGraph toSpec Γ (ss ++ [τ])

namespace RevGraph

variable {toSpec : α → SpecScalar}

/--
Forget the VJP metadata of a `RevGraph`, yielding a `FwdGraph` with the same node ordering.

This lets us reuse the forward approximation theorem (`FwdGraph.eval_approx`) for free.
-/
def toFwdGraph {Γ : List Shape} {ss : List Shape} :
    RevGraph (α := α) toSpec Γ ss → FwdGraph (α := α) toSpec Γ ss
  | .nil => .nil
  | .snoc g node => .snoc (toFwdGraph g) node.toFwdNode

/-- Spec-level evaluation of a `RevGraph` (delegates to `toFwdGraph`). -/
def evalSpec {Γ : List Shape} {ss : List Shape} (g : RevGraph (α := α) toSpec Γ ss) (x : TList
  SpecScalar Γ) :
    TList SpecScalar (Γ ++ ss) :=
  FwdGraph.evalSpec (α := α) (toSpec := toSpec) (Γ := Γ) (ss := ss) (toFwdGraph (α := α) g) x

/-- Runtime-level evaluation of a `RevGraph` (delegates to `toFwdGraph`). -/
def evalRuntime {Γ : List Shape} {ss : List Shape} (g : RevGraph (α := α) toSpec Γ ss) (x : TList α
  Γ) :
    TList α (Γ ++ ss) :=
  FwdGraph.evalRuntime (α := α) (toSpec := toSpec) (Γ := Γ) (ss := ss) (toFwdGraph (α := α) g) x

/-- Forward bound propagation for a `RevGraph` (delegates to `toFwdGraph`). -/
def evalBounds {Γ : List Shape} {ss : List Shape} (g : RevGraph (α := α) toSpec Γ ss)
    (epsIn : EList Γ) (xR : TList α Γ) : EList (Γ ++ ss) :=
  FwdGraph.evalBounds (α := α) (toSpec := toSpec) (Γ := Γ) (ss := ss) (toFwdGraph (α := α) g) epsIn
    xR

/--
Forward approximation theorem for `RevGraph` (just `FwdGraph.eval_approx` via `toFwdGraph`).
-/
theorem eval_approx {Γ : List Shape} {ss : List Shape} (g : RevGraph (α := α) toSpec Γ ss) :
    ∀ (xS : TList SpecScalar Γ) (xR : TList α Γ) (epsIn : EList Γ),
      approxCtx (α := α) toSpec xS xR epsIn →
        approxCtx (α := α) toSpec
          (evalSpec (α := α) (toSpec := toSpec) (Γ := Γ) (ss := ss) g xS)
          (evalRuntime (α := α) (toSpec := toSpec) (Γ := Γ) (ss := ss) g xR)
          (evalBounds (α := α) (toSpec := toSpec) (Γ := Γ) (ss := ss) g epsIn xR) :=
  FwdGraph.eval_approx (α := α) (toSpec := toSpec) (Γ := Γ) (ss := ss) (toFwdGraph (α := α) g)

-- ---------------------------------------------------------------------------
-- Reverse-mode accumulation semantics + bound composition
-- ---------------------------------------------------------------------------

/--
Spec-level reverse-mode evaluation.

Informally: run the forward pass to get node contexts, then traverse nodes in reverse topological
order, applying each node’s VJP and accumulating the resulting context-cotangent into the running
seed.
-/
def backpropSpec {Γ : List Shape} {ss : List Shape} (g : RevGraph (α := α) toSpec Γ ss)
    (x : TList SpecScalar Γ) (seed : TList SpecScalar (Γ ++ ss)) : TList SpecScalar Γ :=
  match g with
  | .nil =>
      TList.cast (α := SpecScalar) (ss₁ := Γ ++ []) (ss₂ := Γ) (List.append_nil Γ) seed
  | .snoc (ss := ssPrev) (τ := τ) g node =>
      let assoc : (Γ ++ ssPrev) ++ [τ] = Γ ++ (ssPrev ++ [τ]) := List.append_assoc Γ ssPrev [τ]
      let seed' : TList SpecScalar ((Γ ++ ssPrev) ++ [τ]) :=
        TList.cast (α := SpecScalar) (ss₁ := Γ ++ (ssPrev ++ [τ])) (ss₂ := (Γ ++ ssPrev) ++ [τ])
          assoc.symm seed
      let seedPrev : TList SpecScalar (Γ ++ ssPrev) := (TList.unsnoc (α := SpecScalar) (ss := Γ ++
        ssPrev) (τ := τ) seed').1
      let seedOut : SpecTensor τ := (TList.unsnoc (α := SpecScalar) (ss := Γ ++ ssPrev) (τ := τ)
        seed').2
      let ctx := evalSpec (α := α) (toSpec := toSpec) (Γ := Γ) (ss := ssPrev) g x
      let contrib := node.vjpSpec ctx seedOut
      let seedPrev' := TList.add (α := SpecScalar) seedPrev contrib
      backpropSpec g x seedPrev'

/--
Runtime-level reverse-mode accumulation.

This mirrors `backpropSpec`, but uses runtime tensors and relies on an `Add α` instance to
accumulate contributions at shared nodes.
-/
def backpropRuntime {Γ : List Shape} {ss : List Shape} (g : RevGraph (α := α) toSpec Γ ss)
    [Add α]
    (x : TList α Γ) (seed : TList α (Γ ++ ss)) : TList α Γ :=
  match g with
  | .nil =>
      TList.cast (α := α) (ss₁ := Γ ++ []) (ss₂ := Γ) (List.append_nil Γ) seed
  | .snoc (ss := ssPrev) (τ := τ) g node =>
      let assoc : (Γ ++ ssPrev) ++ [τ] = Γ ++ (ssPrev ++ [τ]) := List.append_assoc Γ ssPrev [τ]
      let seed' : TList α ((Γ ++ ssPrev) ++ [τ]) :=
        TList.cast (α := α) (ss₁ := Γ ++ (ssPrev ++ [τ])) (ss₂ := (Γ ++ ssPrev) ++ [τ]) assoc.symm
          seed
      let seedPrev : TList α (Γ ++ ssPrev) := (TList.unsnoc (α := α) (ss := Γ ++ ssPrev) (τ := τ)
        seed').1
      let seedOut : Tensor α τ := (TList.unsnoc (α := α) (ss := Γ ++ ssPrev) (τ := τ) seed').2
      let ctx := evalRuntime (α := α) (toSpec := toSpec) (Γ := Γ) (ss := ssPrev) g x
      let contrib := node.vjpRuntime ctx seedOut
      let seedPrev' := TList.add (α := α) seedPrev contrib
      backpropRuntime g x seedPrev'

/--
Backward error propagation for `backprop*`.

This is parameterized by:
- `addBound`: how to compute the (per-entry) error list for context-wise addition.

`addBound` may depend on the runtime addends, to account for rounding models.
-/
def backpropBounds {Γ : List Shape} {ss : List Shape} (g : RevGraph (α := α) toSpec Γ ss)
    [Add α]
    (epsIn : EList Γ) (xR : TList α Γ)
    (epsSeed : EList (Γ ++ ss)) (seedR : TList α (Γ ++ ss))
    (addBound : {Δ : List Shape} → EList Δ → EList Δ → TList α Δ → TList α Δ → EList Δ) : EList Γ :=
  match g with
  | .nil =>
      EList.cast (ss₁ := Γ ++ []) (ss₂ := Γ) (List.append_nil Γ) epsSeed
  | .snoc (ss := ssPrev) (τ := τ) g node =>
      let assoc : (Γ ++ ssPrev) ++ [τ] = Γ ++ (ssPrev ++ [τ]) := List.append_assoc Γ ssPrev [τ]
      let epsSeed' : EList ((Γ ++ ssPrev) ++ [τ]) :=
        EList.cast (ss₁ := Γ ++ (ssPrev ++ [τ])) (ss₂ := (Γ ++ ssPrev) ++ [τ]) assoc.symm epsSeed
      let seed' : TList α ((Γ ++ ssPrev) ++ [τ]) :=
        TList.cast (α := α) (ss₁ := Γ ++ (ssPrev ++ [τ])) (ss₂ := (Γ ++ ssPrev) ++ [τ]) assoc.symm
          seedR
      let epsSeedPrev : EList (Γ ++ ssPrev) := (EList.unsnoc (ss := Γ ++ ssPrev) (τ := τ)
        epsSeed').1
      let epsSeedOut : SpecScalar := (EList.unsnoc (ss := Γ ++ ssPrev) (τ := τ) epsSeed').2
      let seedPrev : TList α (Γ ++ ssPrev) := (TList.unsnoc (α := α) (ss := Γ ++ ssPrev) (τ := τ)
        seed').1
      let seedOut : Tensor α τ := (TList.unsnoc (α := α) (ss := Γ ++ ssPrev) (τ := τ) seed').2
      let ctxR := evalRuntime (α := α) (toSpec := toSpec) (Γ := Γ) (ss := ssPrev) g xR
      let epsCtx := evalBounds (α := α) (toSpec := toSpec) (Γ := Γ) (ss := ssPrev) g epsIn xR
      let contrib := node.vjpRuntime ctxR seedOut
      let epsContrib := node.vjpBound epsCtx ctxR epsSeedOut seedOut
      let seedPrev' := TList.add (α := α) seedPrev contrib
      let epsSeedPrev' := addBound (Δ := Γ ++ ssPrev) epsSeedPrev epsContrib seedPrev contrib
      backpropBounds g epsIn xR epsSeedPrev' seedPrev' addBound

/--
End-to-end reverse-mode approximation theorem for `RevGraph.backprop*`.

Informally:
assume (1) the runtime inputs `xR` approximate the spec inputs `xS` with bounds `epsIn`, and
(2) the runtime seed cotangents `seedR` approximate the spec seeds `seedS` with bounds `epsSeed`.
Then the *whole* backprop result `backpropRuntime g xR seedR` approximates the spec backprop result
`backpropSpec g xS seedS`, with an explicit bound computed by `backpropBounds`.

The only "extra" ingredient beyond per-node VJP approximation is how we accumulate contributions:
`addBound` describes how addition affects error bounds, and `addSound` is the theorem justifying it
(e.g. for exact reals it is trivial; for rounding models it carries the rounding-error analysis).
-/
theorem backprop_approx {Γ : List Shape} {ss : List Shape} (g : RevGraph (α := α) toSpec Γ ss)
    [Add α]
    (addBound : {Δ : List Shape} → EList Δ → EList Δ → TList α Δ → TList α Δ → EList Δ)
    (addSound : ∀ {Δ : List Shape} (xS yS : TList SpecScalar Δ) (xR yR : TList α Δ)
      (epsx epsy : EList Δ),
      approxCtx (α := α) toSpec xS xR epsx →
      approxCtx (α := α) toSpec yS yR epsy →
        approxCtx (α := α) toSpec (TList.add (α := SpecScalar) xS yS) (TList.add (α := α) xR yR)
          (addBound epsx epsy xR yR)) :
    ∀ (xS : TList SpecScalar Γ) (xR : TList α Γ) (epsIn : EList Γ)
      (seedS : TList SpecScalar (Γ ++ ss)) (seedR : TList α (Γ ++ ss)) (epsSeed : EList (Γ ++ ss)),
      approxCtx (α := α) toSpec xS xR epsIn →
      approxCtx (α := α) toSpec seedS seedR epsSeed →
        approxCtx (α := α) toSpec
          (backpropSpec g xS seedS)
          (backpropRuntime g xR seedR)
          (backpropBounds g epsIn xR epsSeed seedR addBound) := by
  intro xS xR epsIn seedS seedR epsSeed hx hseed
  induction g with
  | nil =>
      -- backprop is just a cast along `Γ ++ [] = Γ`.
      simpa [backpropSpec, backpropRuntime, backpropBounds] using
        (approxCtx_cast (α := α) (toSpec := toSpec) (h := (List.append_nil Γ)) (xS := seedS) (xR :=
          seedR) (eps := epsSeed) hseed)
  | snoc g node ih =>
      rename_i ssPrev τ
      -- forward approx for the node context `Γ ++ ssPrev`
      have hctx :
          approxCtx (α := α) toSpec
            (evalSpec (α := α) (toSpec := toSpec) (Γ := Γ) (ss := ssPrev) g xS)
            (evalRuntime (α := α) (toSpec := toSpec) (Γ := Γ) (ss := ssPrev) g xR)
            (evalBounds (α := α) (toSpec := toSpec) (Γ := Γ) (ss := ssPrev) g epsIn xR) :=
        (eval_approx (α := α) (toSpec := toSpec) (Γ := Γ) (ss := ssPrev) g xS xR epsIn hx)

      -- Cast seed to `(Γ ++ ssPrev) ++ [τ]`, then split.
      let assoc : (Γ ++ ssPrev) ++ [τ] = Γ ++ (ssPrev ++ [τ]) := List.append_assoc Γ ssPrev [τ]
      let seedS' : TList SpecScalar ((Γ ++ ssPrev) ++ [τ]) :=
        TList.cast (α := SpecScalar) (ss₁ := Γ ++ (ssPrev ++ [τ])) (ss₂ := (Γ ++ ssPrev) ++ [τ])
          assoc.symm seedS
      let seedR' : TList α ((Γ ++ ssPrev) ++ [τ]) :=
        TList.cast (α := α) (ss₁ := Γ ++ (ssPrev ++ [τ])) (ss₂ := (Γ ++ ssPrev) ++ [τ]) assoc.symm
          seedR
      let epsSeed' : EList ((Γ ++ ssPrev) ++ [τ]) :=
        EList.cast (ss₁ := Γ ++ (ssPrev ++ [τ])) (ss₂ := (Γ ++ ssPrev) ++ [τ]) assoc.symm epsSeed

      have hseed' :
          approxCtx (α := α) toSpec seedS' seedR' epsSeed' := by
        simpa [seedS', seedR', epsSeed'] using
          (approxCtx_cast (α := α) (toSpec := toSpec) (h := assoc.symm) (xS := seedS) (xR := seedR)
            (eps := epsSeed) hseed)

      let seedPrevS : TList SpecScalar (Γ ++ ssPrev) :=
        (TList.unsnoc (α := SpecScalar) (ss := Γ ++ ssPrev) (τ := τ) seedS').1
      let seedOutS : SpecTensor τ :=
        (TList.unsnoc (α := SpecScalar) (ss := Γ ++ ssPrev) (τ := τ) seedS').2
      let seedPrevR : TList α (Γ ++ ssPrev) :=
        (TList.unsnoc (α := α) (ss := Γ ++ ssPrev) (τ := τ) seedR').1
      let seedOutR : Tensor α τ :=
        (TList.unsnoc (α := α) (ss := Γ ++ ssPrev) (τ := τ) seedR').2
      let epsSeedPrev : EList (Γ ++ ssPrev) :=
        (EList.unsnoc (ss := Γ ++ ssPrev) (τ := τ) epsSeed').1
      let epsSeedOut : SpecScalar :=
        (EList.unsnoc (ss := Γ ++ ssPrev) (τ := τ) epsSeed').2

      have hseedSplit :
          approxCtx (α := α) toSpec seedPrevS seedPrevR epsSeedPrev ∧
            approxT (α := α) (toSpec := toSpec) seedOutS seedOutR epsSeedOut := by
        simpa [seedPrevS, seedPrevR, epsSeedPrev, seedOutS, seedOutR, epsSeedOut] using
          (approxCtx_unsnoc (α := α) (toSpec := toSpec) (ss := Γ ++ ssPrev) (τ := τ)
            (xS := seedS') (xR := seedR') (eps := epsSeed') hseed')

      have hseedPrev :
          approxCtx (α := α) toSpec seedPrevS seedPrevR epsSeedPrev :=
        hseedSplit.1

      have hseedOut :
          approxT (α := α) (toSpec := toSpec) seedOutS seedOutR epsSeedOut :=
        hseedSplit.2

      -- local VJP approximation
      let ctxS := evalSpec (α := α) (toSpec := toSpec) (Γ := Γ) (ss := ssPrev) g xS
      let ctxR := evalRuntime (α := α) (toSpec := toSpec) (Γ := Γ) (ss := ssPrev) g xR
      let epsCtx := evalBounds (α := α) (toSpec := toSpec) (Γ := Γ) (ss := ssPrev) g epsIn xR
      let contribS := node.vjpSpec ctxS seedOutS
      let contribR := node.vjpRuntime ctxR seedOutR
      let epsContrib := node.vjpBound epsCtx ctxR epsSeedOut seedOutR

      have hcontrib :
          approxCtx (α := α) toSpec contribS contribR epsContrib :=
        node.vjpSound ctxS ctxR epsCtx seedOutS seedOutR epsSeedOut hctx hseedOut

      -- accumulate into the seed prefix
      have hseedPrev' :
          approxCtx (α := α) toSpec
            (TList.add (α := SpecScalar) seedPrevS contribS)
            (TList.add (α := α) seedPrevR contribR)
            (addBound (Δ := Γ ++ ssPrev) epsSeedPrev epsContrib seedPrevR contribR) :=
        addSound (Δ := Γ ++ ssPrev) seedPrevS contribS seedPrevR contribR epsSeedPrev epsContrib
          hseedPrev hcontrib

      -- finish by IH
      simpa [backpropSpec, backpropRuntime, backpropBounds, assoc, seedS', seedR', epsSeed',
        seedPrevS, seedPrevR, seedOutS, seedOutR, epsSeedPrev, epsSeedOut, ctxS, ctxR, epsCtx,
          contribS, contribR, epsContrib]
        using
          ih
            (seedS := TList.add (α := SpecScalar) seedPrevS contribS)
            (seedR := TList.add (α := α) seedPrevR contribR)
            (epsSeed := addBound (Δ := Γ ++ ssPrev) epsSeedPrev epsContrib seedPrevR contribR)
            hseedPrev'

end RevGraph

end

end RuntimeApprox
end Proofs
