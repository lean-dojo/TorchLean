/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.RuntimeApprox.Graph.Forward

/-!
# Backward graph approximation

Error bounds for reverse-mode evaluation of runtime graphs against their spec, the backward
counterpart of `TorchLean/Verify/RuntimeApprox/Graph/Forward.lean`. A `RevGraph` is a tape of
`RevNode`s in topological order. Each node carries a forward map with its error bound, as a
`FwdNode` does, and spec and runtime VJP maps with the bound transformer `vjpBound` and its
soundness lemma `vjp_sound`.

`RevGraph.eval_approx` reuses the forward theorem through `toFwdGraph`, which forgets the VJP data.
`RevGraph.backprop_approx` composes the local VJP bounds and the accumulation bounds over the whole
graph by induction, as `FwdGraph.eval_approx` does for the forward pass. Accumulation adds context
cotangents, so the theorem takes the bound of that addition and its soundness as parameters
(`addBound`, `addSound`).
-/

@[expose] public section

namespace TorchLean

open TorchLean.Tensor

noncomputable section

variable {α : Type}

-- Reverse-mode graph nodes carrying forward and VJP approximation data.

/-- A `FwdNode` extended with spec and runtime VJP maps and an error bound for the runtime VJP.

`vjp_sound` relates the two supplied reverse maps to each other. Derivative correctness of either
map needs a separate node contract. -/
structure RevNode (toSpec : α → ℝ) (Γ : List Shape) (τ : Shape) extends
    FwdNode (α := α) toSpec Γ τ where
  /-- Spec-level VJP: maps a context and an output cotangent into a context cotangent. -/
  vjpReal : TensorPack ℝ Γ → Tensor ℝ τ → TensorPack ℝ Γ
  /-- Runtime VJP: same shape-level function on the runtime side. -/
  vjpRuntime : TensorPack α Γ → Tensor α τ → TensorPack α Γ
  /-- Bounds on the context cotangent, computed from the context bounds, the output-cotangent bound
  and the runtime context and cotangent. -/
  vjpBound : EList Γ → TensorPack α Γ → ℝ → Tensor α τ → EList Γ
  /-- Soundness of `vjpBound`: if the context and the output cotangent are approximated, the runtime
  VJP approximates the spec VJP within `vjpBound`. -/
  vjp_sound : ∀ (ctxS : TensorPack ℝ Γ) (ctxR : TensorPack α Γ)
      (epsCtx : EList Γ) (δS : Tensor ℝ τ) (δR : Tensor α τ) (epsδ : ℝ),
      ApproxCtx (α := α) toSpec ctxS ctxR epsCtx →
      ApproxTensor (α := α) (toSpec := toSpec) δS δR epsδ →
        ApproxCtx (α := α) toSpec (vjpReal ctxS δS) (vjpRuntime ctxR δR) (vjpBound epsCtx ctxR epsδ
          δR)

/--
A tape (SSA graph) of `RevNode`s appended in topological order, built like a `FwdGraph` from nodes
that also carry VJP data.
-/
inductive RevGraph (toSpec : α → ℝ) (Γ : List Shape) : List Shape → Type where
  /-- The tape with no node. -/
  | nil : RevGraph toSpec Γ []
  /-- A tape followed by one more node over its inputs and values. -/
  | snoc {ss : List Shape} {τ : Shape} :
      RevGraph toSpec Γ ss →
      RevNode (α := α) toSpec (Γ := Γ ++ ss) τ →
      RevGraph toSpec Γ (ss ++ [τ])

namespace RevGraph

variable {toSpec : α → ℝ}

/--
Forget the VJP data of a `RevGraph`, giving the `FwdGraph` with the same nodes in the same order,
to which `FwdGraph.eval_approx` applies.
-/
def toFwdGraph {Γ : List Shape} {ss : List Shape} :
    RevGraph (α := α) toSpec Γ ss → FwdGraph (α := α) toSpec Γ ss
  | .nil => .nil
  | .snoc g node => .snoc (toFwdGraph g) node.toFwdNode

/-- Spec-level evaluation of a `RevGraph` (delegates to `toFwdGraph`). -/
def evalReal {Γ : List Shape} {ss : List Shape} (g : RevGraph (α := α) toSpec Γ ss)
    (x : TensorPack ℝ Γ) :
    TensorPack ℝ (Γ ++ ss) :=
  FwdGraph.evalReal (α := α) (toSpec := toSpec) (Γ := Γ) (ss := ss) (toFwdGraph (α := α) g) x

/-- Runtime-level evaluation of a `RevGraph` (delegates to `toFwdGraph`). -/
def evalRuntime {Γ : List Shape} {ss : List Shape} (g : RevGraph (α := α) toSpec Γ ss)
    (x : TensorPack α Γ) :
    TensorPack α (Γ ++ ss) :=
  FwdGraph.evalRuntime (α := α) (toSpec := toSpec) (Γ := Γ) (ss := ss) (toFwdGraph (α := α) g) x

/-- Forward bound propagation for a `RevGraph` (delegates to `toFwdGraph`). -/
def evalBounds {Γ : List Shape} {ss : List Shape} (g : RevGraph (α := α) toSpec Γ ss)
    (epsIn : EList Γ) (xR : TensorPack α Γ) : EList (Γ ++ ss) :=
  FwdGraph.evalBounds (α := α) (toSpec := toSpec) (Γ := Γ) (ss := ss) (toFwdGraph (α := α) g) epsIn
    xR

/--
Forward approximation for `RevGraph`: `FwdGraph.eval_approx` applied to `toFwdGraph g`.
-/
theorem eval_approx {Γ : List Shape} {ss : List Shape} (g : RevGraph (α := α) toSpec Γ ss) :
    ∀ (xS : TensorPack ℝ Γ) (xR : TensorPack α Γ) (epsIn : EList Γ),
      ApproxCtx (α := α) toSpec xS xR epsIn →
        ApproxCtx (α := α) toSpec
          (evalReal (α := α) (toSpec := toSpec) (Γ := Γ) (ss := ss) g xS)
          (evalRuntime (α := α) (toSpec := toSpec) (Γ := Γ) (ss := ss) g xR)
          (evalBounds (α := α) (toSpec := toSpec) (Γ := Γ) (ss := ss) g epsIn xR) :=
  FwdGraph.eval_approx (α := α) (toSpec := toSpec) (Γ := Γ) (ss := ss) (toFwdGraph (α := α) g)

-- ---------------------------------------------------------------------------
-- Reverse-mode accumulation semantics + bound composition
-- ---------------------------------------------------------------------------

/--
Spec-level reverse-mode accumulation, from a seed cotangent for every entry of `Γ ++ ss` to a
cotangent of the inputs `Γ`. Nodes are visited from last to first. Each node's VJP maps its output
cotangent, at the context that `evalReal` computes, to a cotangent of that context, which is added
to the seed of the earlier entries.
-/
def backpropReal {Γ : List Shape} {ss : List Shape} (g : RevGraph (α := α) toSpec Γ ss)
    (x : TensorPack ℝ Γ) (seed : TensorPack ℝ (Γ ++ ss)) :
    TensorPack ℝ Γ :=
  match g with
  | .nil =>
      Pack.cast (ss₁ := Γ ++ []) (ss₂ := Γ)
        (List.append_nil Γ) seed
  | .snoc (ss := ssPrev) (τ := τ) g node =>
      let assoc : (Γ ++ ssPrev) ++ [τ] = Γ ++ (ssPrev ++ [τ]) := List.append_assoc Γ ssPrev [τ]
      let seed' : TensorPack ℝ ((Γ ++ ssPrev) ++ [τ]) :=
        Pack.cast (ss₁ := Γ ++ (ssPrev ++ [τ]))
          (ss₂ := (Γ ++ ssPrev) ++ [τ]) assoc.symm seed
      let seedPrev : TensorPack ℝ (Γ ++ ssPrev) :=
        (Pack.unsnoc (ss := Γ ++ ssPrev) (τ := τ) seed').1
      let seedOut : Tensor ℝ τ :=
        (Pack.unsnoc (ss := Γ ++ ssPrev) (τ := τ) seed').2
      let ctx := evalReal (α := α) (toSpec := toSpec) (Γ := Γ) (ss := ssPrev) g x
      let contrib := node.vjpReal ctx seedOut
      let seedPrev' := TensorPack.add (α := ℝ) seedPrev contrib
      backpropReal g x seedPrev'

/--
Runtime-level reverse-mode accumulation: `backpropReal` on runtime tensors, with the contributions
at shared nodes summed by the `Add α` instance.
-/
def backpropRuntime {Γ : List Shape} {ss : List Shape} (g : RevGraph (α := α) toSpec Γ ss)
    [Add α]
    (x : TensorPack α Γ) (seed : TensorPack α (Γ ++ ss)) :
    TensorPack α Γ :=
  match g with
  | .nil =>
      Pack.cast (ss₁ := Γ ++ []) (ss₂ := Γ) (List.append_nil Γ) seed
  | .snoc (ss := ssPrev) (τ := τ) g node =>
      let assoc : (Γ ++ ssPrev) ++ [τ] = Γ ++ (ssPrev ++ [τ]) := List.append_assoc Γ ssPrev [τ]
      let seed' : TensorPack α ((Γ ++ ssPrev) ++ [τ]) :=
        Pack.cast (ss₁ := Γ ++ (ssPrev ++ [τ]))
          (ss₂ := (Γ ++ ssPrev) ++ [τ]) assoc.symm seed
      let seedPrev : TensorPack α (Γ ++ ssPrev) :=
        (Pack.unsnoc (ss := Γ ++ ssPrev) (τ := τ) seed').1
      let seedOut : Tensor α τ :=
        (Pack.unsnoc (ss := Γ ++ ssPrev) (τ := τ) seed').2
      let ctx := evalRuntime (α := α) (toSpec := toSpec) (Γ := Γ) (ss := ssPrev) g x
      let contrib := node.vjpRuntime ctx seedOut
      let seedPrev' := TensorPack.add (α := α) seedPrev contrib
      backpropRuntime g x seedPrev'

/--
Error bounds for `backpropRuntime` against `backpropReal`, propagated in the same order. The
parameter `addBound` gives the bounds of a context-wise sum from the bounds of the two addends. It
also reads the runtime addends, which a rounding model needs to bound the rounding of the sum.
-/
def backpropBounds {Γ : List Shape} {ss : List Shape} (g : RevGraph (α := α) toSpec Γ ss)
    [Add α]
    (epsIn : EList Γ) (xR : TensorPack α Γ)
    (epsSeed : EList (Γ ++ ss)) (seedR : TensorPack α (Γ ++ ss))
    (addBound : {Δ : List Shape} → EList Δ → EList Δ →
      TensorPack α Δ → TensorPack α Δ → EList Δ) : EList Γ :=
  match g with
  | .nil =>
      Pack.cast (ss₁ := Γ ++ []) (ss₂ := Γ) (List.append_nil Γ) epsSeed
  | .snoc (ss := ssPrev) (τ := τ) g node =>
      let assoc : (Γ ++ ssPrev) ++ [τ] = Γ ++ (ssPrev ++ [τ]) := List.append_assoc Γ ssPrev [τ]
      let epsSeed' : EList ((Γ ++ ssPrev) ++ [τ]) :=
        Pack.cast (ss₁ := Γ ++ (ssPrev ++ [τ])) (ss₂ := (Γ ++ ssPrev) ++ [τ]) assoc.symm epsSeed
      let seed' : TensorPack α ((Γ ++ ssPrev) ++ [τ]) :=
        Pack.cast (ss₁ := Γ ++ (ssPrev ++ [τ]))
          (ss₂ := (Γ ++ ssPrev) ++ [τ]) assoc.symm seedR
      let epsSeedPrev : EList (Γ ++ ssPrev) :=
        (Pack.unsnoc (ss := Γ ++ ssPrev) (τ := τ) epsSeed').1
      let epsSeedOut : ℝ := (Pack.unsnoc (ss := Γ ++ ssPrev) (τ := τ) epsSeed').2
      let seedPrev : TensorPack α (Γ ++ ssPrev) :=
        (Pack.unsnoc (ss := Γ ++ ssPrev) (τ := τ) seed').1
      let seedOut : Tensor α τ :=
        (Pack.unsnoc (ss := Γ ++ ssPrev) (τ := τ) seed').2
      let ctxR := evalRuntime (α := α) (toSpec := toSpec) (Γ := Γ) (ss := ssPrev) g xR
      let epsCtx := evalBounds (α := α) (toSpec := toSpec) (Γ := Γ) (ss := ssPrev) g epsIn xR
      let contrib := node.vjpRuntime ctxR seedOut
      let epsContrib := node.vjpBound epsCtx ctxR epsSeedOut seedOut
      let seedPrev' := TensorPack.add (α := α) seedPrev contrib
      let epsSeedPrev' := addBound (Δ := Γ ++ ssPrev) epsSeedPrev epsContrib seedPrev contrib
      backpropBounds g epsIn xR epsSeedPrev' seedPrev' addBound

/--
End-to-end reverse-mode approximation for `RevGraph`: if the runtime inputs `xR` approximate the
spec inputs `xS` with bounds `epsIn` and the runtime seeds `seedR` approximate the spec seeds
`seedS` with bounds `epsSeed`, then `backpropRuntime g xR seedR` approximates
`backpropReal g xS seedS` with the bounds `backpropBounds g epsIn xR epsSeed seedR addBound`.

Beyond the local VJP bounds, the proof needs one rule for accumulation: `addBound` bounds a
context-wise sum and `addSound` proves that bound. For exact reals `addSound` is elementary, and
for rounding models it carries the rounding-error analysis of the addition.
-/
theorem backprop_approx {Γ : List Shape} {ss : List Shape} (g : RevGraph (α := α) toSpec Γ ss)
    [Add α]
    (addBound : {Δ : List Shape} → EList Δ → EList Δ →
      TensorPack α Δ → TensorPack α Δ → EList Δ)
    (addSound : ∀ {Δ : List Shape} (xS yS : TensorPack ℝ Δ)
      (xR yR : TensorPack α Δ) (epsx epsy : EList Δ),
      ApproxCtx (α := α) toSpec xS xR epsx →
      ApproxCtx (α := α) toSpec yS yR epsy →
        ApproxCtx (α := α) toSpec (TensorPack.add (α := ℝ) xS yS)
          (TensorPack.add (α := α) xR yR) (addBound epsx epsy xR yR)) :
    ∀ (xS : TensorPack ℝ Γ) (xR : TensorPack α Γ) (epsIn : EList Γ)
      (seedS : TensorPack ℝ (Γ ++ ss)) (seedR : TensorPack α (Γ ++ ss))
      (epsSeed : EList (Γ ++ ss)),
      ApproxCtx (α := α) toSpec xS xR epsIn →
      ApproxCtx (α := α) toSpec seedS seedR epsSeed →
        ApproxCtx (α := α) toSpec
          (backpropReal g xS seedS)
          (backpropRuntime g xR seedR)
          (backpropBounds g epsIn xR epsSeed seedR addBound) := by
  intro xS xR epsIn seedS seedR epsSeed hx hseed
  induction g with
  | nil =>
      -- With no nodes, backprop is a cast along `Γ ++ [] = Γ`.
      simpa [backpropReal, backpropRuntime, backpropBounds] using
        (ApproxCtx.cast (α := α) (toSpec := toSpec) (h := (List.append_nil Γ)) (xS := seedS) (xR :=
          seedR) (eps := epsSeed) hseed)
  | snoc g node ih =>
      rename_i ssPrev τ
      -- forward approx for the node context `Γ ++ ssPrev`
      have hctx :
          ApproxCtx (α := α) toSpec
            (evalReal (α := α) (toSpec := toSpec) (Γ := Γ) (ss := ssPrev) g xS)
            (evalRuntime (α := α) (toSpec := toSpec) (Γ := Γ) (ss := ssPrev) g xR)
            (evalBounds (α := α) (toSpec := toSpec) (Γ := Γ) (ss := ssPrev) g epsIn xR) :=
        (eval_approx (α := α) (toSpec := toSpec) (Γ := Γ) (ss := ssPrev) g xS xR epsIn hx)

      -- Cast seed to `(Γ ++ ssPrev) ++ [τ]`, then split.
      let assoc : (Γ ++ ssPrev) ++ [τ] = Γ ++ (ssPrev ++ [τ]) := List.append_assoc Γ ssPrev [τ]
      let seedS' : TensorPack ℝ ((Γ ++ ssPrev) ++ [τ]) :=
        Pack.cast (ss₁ := Γ ++ (ssPrev ++ [τ]))
          (ss₂ := (Γ ++ ssPrev) ++ [τ]) assoc.symm seedS
      let seedR' : TensorPack α ((Γ ++ ssPrev) ++ [τ]) :=
        Pack.cast (ss₁ := Γ ++ (ssPrev ++ [τ]))
          (ss₂ := (Γ ++ ssPrev) ++ [τ]) assoc.symm seedR
      let epsSeed' : EList ((Γ ++ ssPrev) ++ [τ]) :=
        Pack.cast (ss₁ := Γ ++ (ssPrev ++ [τ])) (ss₂ := (Γ ++ ssPrev) ++ [τ]) assoc.symm epsSeed

      have hseed' :
          ApproxCtx (α := α) toSpec seedS' seedR' epsSeed' := by
        simpa [seedS', seedR', epsSeed'] using
          (ApproxCtx.cast (α := α) (toSpec := toSpec) (h := assoc.symm) (xS := seedS) (xR := seedR)
            (eps := epsSeed) hseed)

      let seedPrevS : TensorPack ℝ (Γ ++ ssPrev) :=
        (Pack.unsnoc (ss := Γ ++ ssPrev) (τ := τ) seedS').1
      let seedOutS : Tensor ℝ τ :=
        (Pack.unsnoc (ss := Γ ++ ssPrev) (τ := τ) seedS').2
      let seedPrevR : TensorPack α (Γ ++ ssPrev) :=
        (Pack.unsnoc (ss := Γ ++ ssPrev) (τ := τ) seedR').1
      let seedOutR : Tensor α τ :=
        (Pack.unsnoc (ss := Γ ++ ssPrev) (τ := τ) seedR').2
      let epsSeedPrev : EList (Γ ++ ssPrev) :=
        (Pack.unsnoc (ss := Γ ++ ssPrev) (τ := τ) epsSeed').1
      let epsSeedOut : ℝ :=
        (Pack.unsnoc (ss := Γ ++ ssPrev) (τ := τ) epsSeed').2

      have hseedSplit :
          ApproxCtx (α := α) toSpec seedPrevS seedPrevR epsSeedPrev ∧
            ApproxTensor (α := α) (toSpec := toSpec) seedOutS seedOutR epsSeedOut := by
        simpa [seedPrevS, seedPrevR, epsSeedPrev, seedOutS, seedOutR, epsSeedOut] using
          (ApproxCtx.unsnoc (α := α) (toSpec := toSpec) (ss := Γ ++ ssPrev) (τ := τ)
            (xS := seedS') (xR := seedR') (eps := epsSeed') hseed')

      have hseedPrev :
          ApproxCtx (α := α) toSpec seedPrevS seedPrevR epsSeedPrev :=
        hseedSplit.1

      have hseedOut :
          ApproxTensor (α := α) (toSpec := toSpec) seedOutS seedOutR epsSeedOut :=
        hseedSplit.2

      -- local VJP approximation
      let ctxS := evalReal (α := α) (toSpec := toSpec) (Γ := Γ) (ss := ssPrev) g xS
      let ctxR := evalRuntime (α := α) (toSpec := toSpec) (Γ := Γ) (ss := ssPrev) g xR
      let epsCtx := evalBounds (α := α) (toSpec := toSpec) (Γ := Γ) (ss := ssPrev) g epsIn xR
      let contribS := node.vjpReal ctxS seedOutS
      let contribR := node.vjpRuntime ctxR seedOutR
      let epsContrib := node.vjpBound epsCtx ctxR epsSeedOut seedOutR

      have hcontrib :
          ApproxCtx (α := α) toSpec contribS contribR epsContrib :=
        node.vjp_sound ctxS ctxR epsCtx seedOutS seedOutR epsSeedOut hctx hseedOut

      -- accumulate into the seed prefix
      have hseedPrev' :
          ApproxCtx (α := α) toSpec
            (TensorPack.add (α := ℝ) seedPrevS contribS)
            (TensorPack.add (α := α) seedPrevR contribR)
            (addBound (Δ := Γ ++ ssPrev) epsSeedPrev epsContrib seedPrevR contribR) :=
        addSound (Δ := Γ ++ ssPrev) seedPrevS contribS seedPrevR contribR epsSeedPrev epsContrib
          hseedPrev hcontrib

      simpa [backpropReal, backpropRuntime, backpropBounds, assoc, seedS', seedR', epsSeed',
        seedPrevS, seedPrevR, seedOutS, seedOutR, epsSeedPrev, epsSeedOut, ctxS, ctxR, epsCtx,
          contribS, contribR, epsContrib]
        using
          ih
            (seedS := TensorPack.add (α := ℝ) seedPrevS contribS)
            (seedR := TensorPack.add (α := α) seedPrevR contribR)
            (epsSeed := addBound (Δ := Γ ++ ssPrev) epsSeedPrev epsContrib seedPrevR contribR)
            hseedPrev'

end RevGraph

end

end TorchLean
