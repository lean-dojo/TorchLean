/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Tape.Algebra.Soundness
public import NN.Proofs.RuntimeApprox.Core.SpecApprox

/-!
# ForwardApprox

Forward (runtime→spec) approximation framework.

This file is **backend-agnostic**: it proves that approximation bounds compose over a
tape/SSA-style graph, assuming each node provides a local forward approximation lemma.

It is intended to be instantiated by proof-relevant runtimes such as rounding models
(`NF` / `neural_round`). Lean's builtin `Float` is treated as trusted (see
`NN/Runtime/Scalar.lean`).

## What you get
- `FwdGraph.eval_approx`: an end-to-end theorem saying that if the runtime input context is within
  an explicit per-entry error budget of the spec input context, then runtime graph evaluation
  is within a computable propagated error budget of the spec evaluation.

## Reading guide
1. `TList` and `EList`: heterogeneous runtime/spec contexts and aligned error vectors.
2. `approxT` and `approxCtx`: the approximation predicates for a single tensor and a whole context.
3. `Idx`: a typed index into a context (so graph nodes can refer to earlier values safely).
4. `FwdNode` / `FwdGraph`: local approximation lemmas and their composition over a snoc-list DAG.

## PyTorch correspondence / citations
This is conceptually similar to the “graph of ops” view behind PyTorch Autograd (and tooling like
`torch.fx`), except that our graph nodes carry *proof-relevant* approximation bounds that can be
composed into an end-to-end theorem.
https://pytorch.org/docs/stable/autograd.html
https://pytorch.org/docs/stable/fx.html
-/

@[expose] public section


namespace Proofs
namespace RuntimeApprox

open Spec
open NN.MLTheory.Robustness.Spec

noncomputable section

-- ---------------------------------------------------------------------------
-- Heterogeneous contexts (tensors indexed by a list of shapes)
-- ---------------------------------------------------------------------------

/-
We reuse the canonical `TList` implementation from the tape-style autograd algebra layer.
This avoids duplicating the “heterogeneous context indexed by `List Shape`” encoding in multiple
proof subsystems.
-/
/--
A heterogeneous context (one tensor per shape), indexed by a `List Shape`.

This is just an alias of the canonical tape/autograd-algebra `TList` so that all the helper
operations (`get`, `cast`, `snoc`, `unsnoc`, `add`, ...) are shared across proof subsystems.
-/
abbrev TList (α : Type) (ss : List Shape) : Type :=
  Proofs.Autograd.Algebra.TList α ss

namespace TList

-- Re-export the most-used context operations so downstream files can stay in the `RuntimeApprox`
-- namespace without qualifying everything with `Proofs.Autograd.Algebra`.
export Proofs.Autograd.Algebra.TList (get cast cast_rfl cast_cast snoc unsnoc zero add)

end TList

-- ---------------------------------------------------------------------------
-- Error vectors aligned with contexts
-- ---------------------------------------------------------------------------

/-- Scalar error bounds aligned with a context shape list. -/
inductive EList : List Shape → Type where
  | nil : EList []
  | cons {s : Shape} {ss : List Shape} : ℝ → EList ss → EList (s :: ss)

namespace EList

/-- Cast an error list along an equality of shape lists. -/
def cast {ss₁ ss₂ : List Shape} (h : ss₁ = ss₂) (xs : EList ss₁) : EList ss₂ :=
  Eq.mp (congrArg EList h) xs

/-- Casting along `rfl` is the identity. -/
@[simp] theorem cast_rfl {ss : List Shape} (xs : EList ss) :
    cast (ss₁ := ss) (ss₂ := ss) rfl xs = xs := by
  rfl

/-- `EList.cast` composes as expected. -/
@[simp] theorem cast_cast {ss₁ ss₂ ss₃ : List Shape} (h₁ : ss₁ = ss₂) (h₂ : ss₂ = ss₃) (xs : EList
  ss₁) :
    cast h₂ (cast h₁ xs) = cast (h₁.trans h₂) xs := by
  cases h₁
  cases h₂
  rfl

/-- Append a scalar error bound to the end of an error list. -/
def snoc {τ : Shape} : {ss : List Shape} → EList ss → ℝ → EList (ss ++ [τ])
  | [], .nil, e => .cons e .nil
  | _ :: ss, .cons x xs, e => .cons x (snoc (ss := ss) xs e)

/-- Split an error list aligned with `ss ++ [τ]` into its prefix and last scalar. -/
def unsnoc {τ : Shape} : {ss : List Shape} → EList (ss ++ [τ]) → EList ss × ℝ
  | [], .cons e .nil => (.nil, e)
  | _ :: ss, .cons x xs =>
      let (ys, last) := unsnoc (ss := ss) (τ := τ) xs
      (.cons x ys, last)

/-- `unsnoc` is a left inverse of `snoc`. -/
@[simp] theorem unsnoc_snoc {ss : List Shape} {τ : Shape} (xs : EList ss) (e : ℝ) :
    unsnoc (ss := ss) (τ := τ) (snoc (ss := ss) (τ := τ) xs e) = (xs, e) := by
  induction ss with
  | nil =>
      cases xs
      simp [snoc, unsnoc]
  | cons s ss ih =>
      cases xs with
      | cons x xt =>
          simp [snoc, unsnoc, ih]

/-- `snoc` is a left inverse of `unsnoc`. -/
@[simp] theorem snoc_unsnoc {ss : List Shape} {τ : Shape} (xs : EList (ss ++ [τ])) :
    snoc (ss := ss) (τ := τ)
        (unsnoc (ss := ss) (τ := τ) xs).1
        (unsnoc (ss := ss) (τ := τ) xs).2
      = xs := by
  induction ss with
  | nil =>
      cases xs with
      | cons e xs =>
          cases xs with
          | nil =>
              simp [unsnoc, snoc]
  | cons s ss ih =>
      cases xs with
      | cons x xs =>
          have hrec := ih (xs := xs)
          simp [unsnoc, snoc, hrec]

/-- Get the `i`th scalar bound from an error list. -/
def get : {ss : List Shape} → EList ss → (i : Fin ss.length) → ℝ
  | [], .nil, i => nomatch i
  | _ :: _, .cons x _xs, ⟨0, _⟩ => x
  | _ :: ss, .cons _x xs, ⟨Nat.succ i, hi⟩ =>
      get (ss := ss) xs ⟨i, Nat.lt_of_succ_lt_succ hi⟩

end EList

-- ---------------------------------------------------------------------------
-- Approximation predicates
-- ---------------------------------------------------------------------------

variable {α : Type}

/-- Tensor-level approximation under a `toSpec : α → ℝ` mapping. -/
def approxT {s : Shape} (toSpec : α → SpecScalar) (spec : SpecTensor s) (runtime : Tensor α s) (eps
  : SpecScalar) : Prop :=
  approxWith (α := α) (toSpec := toSpec) (norm := linfNorm) spec runtime eps

/--
Scoped notation for tensor approximation (`approxT`).

```lean
open scoped RuntimeApprox
spec ≈ᵀ[toSpec] runtime : eps
```
-/
scoped[RuntimeApprox] notation:50 spec " ≈ᵀ[" toSpec "] " runtime " : " eps =>
  Proofs.RuntimeApprox.approxT (toSpec := toSpec) spec runtime eps

/--
Monotonicity of `approxTTol`: if you only loosen tolerances, an approximation stays valid.
-/
lemma approxTTol_mono {s : Shape} {toSpec : α → SpecScalar}
    {spec : SpecTensor s} {runtime : Tensor α s} {tol₁ tol₂ : ApproxTol}
    (habs : tol₁.abs ≤ tol₂.abs) (hrel : tol₁.rel ≤ tol₂.rel) (hslack : tol₁.slack ≤ tol₂.slack)
    (h : approxTTol (toSpec := toSpec) spec runtime tol₁) :
    approxTTol (toSpec := toSpec) spec runtime tol₂ :=
  approx_with_tol_mono (toSpec := toSpec) (norm := linfNorm)
    (spec := spec) (runtime := runtime) habs hrel hslack h

/--
`approxTTol` specialized to an absolute-only tolerance is equivalent to `approxT`.

This is mostly a convenience lemma for switching between the "tolerance" API and the plain
`eps : ℝ` API.
-/
lemma approxTTol_absOnly_iff {s : Shape} {toSpec : α → SpecScalar}
    {spec : SpecTensor s} {runtime : Tensor α s} {eps : ℝ} (heps : 0 ≤ eps) :
    approxTTol (toSpec := toSpec) spec runtime (ApproxTol.absOnly eps) ↔
      approxT (toSpec := toSpec) spec runtime eps := by
  -- Reduce to the `approx_with` lemma.
  simpa [approxT, approxTTol] using
    (approx_with_tol_absOnly_iff (toSpec := toSpec) (norm := linfNorm)
      (spec := spec) (runtime := runtime) heps)

/-- Context-level approximation with a per-entry error list. -/
def approxCtx (toSpec : α → SpecScalar) : {ss : List Shape} →
    TList SpecScalar ss → TList α ss → EList ss → Prop
  | [], .nil, .nil, .nil => True
  | _ :: ss, .cons x xs, .cons y ys, .cons e es =>
      approxT (α := α) (toSpec := toSpec) x y e ∧ approxCtx (ss := ss) toSpec xs ys es

/--
Scoped notation for `approxCtx`.

This keeps end-to-end forward-soundness theorems readable:

```lean
open scoped RuntimeApprox
-- `ΓS` is an approximate view of `ΓR` with per-entry bounds `eps`:
ΓS ≈ᶜ[toSpec] ΓR : eps
```

The notation is scoped to avoid polluting unrelated code.
-/
scoped[RuntimeApprox] notation:50 ΓS " ≈ᶜ[" toSpec "] " ΓR " : " eps =>
  Proofs.RuntimeApprox.approxCtx (toSpec := toSpec) ΓS ΓR eps

/--
Transport a context approximation across an equality of shape lists.

This is used any time we need to reassociate `Γ ++ ss` type indices (casts are unavoidable in this
`List Shape`-indexed encoding).
-/
lemma approxCtx_cast {toSpec : α → SpecScalar} {ss₁ ss₂ : List Shape} (h : ss₁ = ss₂)
    {xS : TList SpecScalar ss₁} {xR : TList α ss₁} {eps : EList ss₁} :
    approxCtx (α := α) toSpec xS xR eps →
      approxCtx (α := α) toSpec
        (TList.cast (α := SpecScalar) (ss₁ := ss₁) (ss₂ := ss₂) h xS)
        (TList.cast (α := α) (ss₁ := ss₁) (ss₂ := ss₂) h xR)
        (EList.cast (ss₁ := ss₁) (ss₂ := ss₂) h eps) := by
  cases h
  simp

/--
Extend a context approximation by appending one more approximated tensor.

This is the core "composition" step used when evaluating a snoc-graph: if the previous context is
approximated, and the new node output is approximated with some bound `e`, then the extended context
is approximated with the extended error list.
-/
lemma approxCtx_snoc {toSpec : α → SpecScalar} {ss : List Shape} {τ : Shape}
    {xS : TList SpecScalar ss} {xR : TList α ss} {eps : EList ss}
    (hx : approxCtx (α := α) toSpec xS xR eps)
    {yS : SpecTensor τ} {yR : Tensor α τ} {e : SpecScalar}
    (hy : approxT (α := α) (toSpec := toSpec) yS yR e) :
    approxCtx (α := α) toSpec
      (TList.snoc (α := SpecScalar) (ss := ss) xS yS)
      (TList.snoc (α := α) (ss := ss) xR yR)
      (EList.snoc (ss := ss) (τ := τ) eps e) := by
  induction ss with
  | nil =>
      cases xS
      cases xR
      cases eps
      simpa [TList.snoc, EList.snoc, approxCtx] using And.intro hy True.intro
  | cons s ss ih =>
      cases xS with
      | cons xSh xSt =>
          cases xR with
          | cons xRh xRt =>
              cases eps with
              | cons eh et =>
                  have hx' : approxCtx (α := α) toSpec xSt xRt et := hx.2
                  have ih' := ih hx'
                  -- peel off the head entry
                  exact And.intro hx.1 ih'

/-- Extract a single entry approximation from `approxCtx`. -/
lemma approxCtx_get {toSpec : α → SpecScalar} {Γ : List Shape}
    {xS : TList SpecScalar Γ} {xR : TList α Γ} {eps : EList Γ}
    (h : approxCtx (α := α) toSpec xS xR eps) (i : Fin Γ.length) :
    approxT (α := α) (toSpec := toSpec)
      (TList.get (α := SpecScalar) xS i)
      (TList.get (α := α) xR i)
      (EList.get eps i) := by
  induction Γ with
  | nil =>
      cases i with
      | mk val isLt =>
          exact False.elim ((Nat.not_lt_zero val) isLt)
  | cons s0 Γ ih =>
      cases xS with
      | cons xSh xSt =>
          cases xR with
          | cons xRh xRt =>
              cases eps with
              | cons eh et =>
                  cases i with
                  | mk iVal hiVal =>
                      cases iVal with
                      | zero =>
                          simpa [TList.get, EList.get] using h.1
                      | succ j =>
                          have := ih (xS := xSt) (xR := xRt) (eps := et) h.2
                            ⟨j, Nat.lt_of_succ_lt_succ hiVal⟩
                          simpa [TList.get, EList.get] using this

/--
`approxCtx_get` expressed in terms of `approxTTol` with an absolute-only tolerance.

Many downstream theorems are stated using a tolerance record (`ApproxTol`) rather than a bare
`eps : ℝ`. For absolute-only bounds, this lemma gives the bridge.
-/
lemma approxCtx_get_tolAbsOnly {toSpec : α → SpecScalar} {Γ : List Shape}
    {xS : TList SpecScalar Γ} {xR : TList α Γ} {eps : EList Γ}
    (h : approxCtx (α := α) toSpec xS xR eps) (i : Fin Γ.length) :
    approxTTol (α := α) (toSpec := toSpec)
      (TList.get (α := SpecScalar) xS i)
      (TList.get (α := α) xR i)
      (ApproxTol.absOnly (EList.get eps i)) := by
  have hi :
      approxT (α := α) (toSpec := toSpec)
        (TList.get (α := SpecScalar) xS i)
        (TList.get (α := α) xR i)
        (EList.get eps i) :=
    approxCtx_get (α := α) (toSpec := toSpec) (xS := xS) (xR := xR) (eps := eps) h i
  have : approxWith (α := α) (toSpec := toSpec) (norm := linfNorm)
      (TList.get (α := SpecScalar) xS i)
      (TList.get (α := α) xR i)
      (EList.get eps i) := by
    simpa [approxT] using hi
  simpa using
    (approxT_to_approxTTol_absOnly (toSpec := toSpec)
      (spec := (TList.get (α := SpecScalar) xS i))
      (runtime := (TList.get (α := α) xR i))
      (eps := (EList.get eps i)) this)

/--
Split a context approximation for `ss ++ [τ]` into:
- a prefix context approximation for `ss`, and
- a single-tensor approximation for the last entry of shape `τ`.
-/
lemma approxCtx_unsnoc {toSpec : α → SpecScalar} {ss : List Shape} {τ : Shape}
    {xS : TList SpecScalar (ss ++ [τ])} {xR : TList α (ss ++ [τ])} {eps : EList (ss ++ [τ])} :
    approxCtx (α := α) toSpec xS xR eps →
      approxCtx (α := α) toSpec
          (TList.unsnoc (α := SpecScalar) (ss := ss) (τ := τ) xS).1
          (TList.unsnoc (α := α) (ss := ss) (τ := τ) xR).1
          (EList.unsnoc (ss := ss) (τ := τ) eps).1
        ∧
      approxT (α := α) (toSpec := toSpec)
          (TList.unsnoc (α := SpecScalar) (ss := ss) (τ := τ) xS).2
          (TList.unsnoc (α := α) (ss := ss) (τ := τ) xR).2
          (EList.unsnoc (ss := ss) (τ := τ) eps).2 := by
  intro h
  induction ss with
  | nil =>
      cases xS with
      | cons tS xsS =>
          cases xsS with
          | nil =>
              cases xR with
              | cons tR xsR =>
                  cases xsR with
                  | nil =>
                      cases eps with
                      | cons e es =>
                          cases es with
                          | nil =>
                              refine And.intro ?_ ?_
                              · simp [TList.unsnoc, EList.unsnoc, approxCtx]
                              · simpa [TList.unsnoc, EList.unsnoc, approxCtx] using h.1
  | cons s ss ih =>
      cases xS with
      | cons xSh xSt =>
          cases xR with
          | cons xRh xRt =>
              cases eps with
              | cons eh et =>
                  have hx : approxT (α := α) (toSpec := toSpec) xSh xRh eh := h.1
                  have ht : approxCtx (α := α) toSpec xSt xRt et := h.2
                  have ih' := ih (xS := xSt) (xR := xRt) (eps := et) ht
                  refine And.intro ?_ ?_
                  · simpa [TList.unsnoc, EList.unsnoc, approxCtx] using And.intro hx ih'.1
                  · simpa [TList.unsnoc, EList.unsnoc] using ih'.2

-- ---------------------------------------------------------------------------
-- Typed indexing into contexts (for building graphs)
-- ---------------------------------------------------------------------------

/-- An index into a heterogeneous context, carrying a proof of the expected shape. -/
structure Idx (Γ : List Shape) (s : Shape) where
  /-- i. -/
  i : Fin Γ.length
  /-- h. -/
  h : Γ.get i = s

/-- Typed lookup from a heterogeneous context `TList α Γ` using an index `Idx Γ s`. -/
def getIdx {α : Type} {Γ : List Shape} {s : Shape} (xs : TList α Γ) (idx : Idx Γ s) : Tensor α s :=
  Spec.tensorCast (α := α) (t := s) idx.h (TList.get (α := α) xs idx.i)

/-- Lookup the epsilon entry associated to an index `Idx Γ s`. -/
def getIdxEps {Γ : List Shape} {s : Shape} (es : EList Γ) (idx : Idx Γ s) : ℝ :=
  EList.get es idx.i

/--
Context approximation implies approximation of any indexed entry.

Informally: if every tensor in the runtime context is close to its spec counterpart (with an
aligned error list `eps`), then reading any entry `idx : Idx Γ s` yields an `approxT` fact with the
corresponding scalar bound `getIdxEps eps idx`.
-/
lemma approxCtx_getIdx {toSpec : α → SpecScalar} {Γ : List Shape} {s : Shape}
    {xS : TList SpecScalar Γ} {xR : TList α Γ} {eps : EList Γ}
    (h : approxCtx (α := α) toSpec xS xR eps) (idx : Idx Γ s) :
    approxT (α := α) (toSpec := toSpec)
      (getIdx (α := SpecScalar) xS idx)
      (getIdx (α := α) xR idx)
      (getIdxEps (Γ := Γ) (s := s) eps idx) := by
  cases idx with
  | mk i hshape =>
      have h' := approxCtx_get (α := α) (toSpec := toSpec) (xS := xS) (xR := xR) (eps := eps) h i
      cases hshape
      simpa [Idx, getIdx, getIdxEps] using h'

-- ---------------------------------------------------------------------------
-- A forward-only tape/SSA graph with local approximation lemmas
-- ---------------------------------------------------------------------------

/--
A single SSA/DAG node with a *local* forward approximation lemma.

Fields:
- `forwardSpec` / `forwardRuntime`: the spec vs runtime semantics of the node.
- `bound`: computes an explicit scalar error bound for the node’s output, given current context
  bounds and the runtime context (to allow data-dependent bounds).
- `sound`: the “local theorem” that justifies `bound`.

Informally: if the whole input context is approximated (`approxCtx`), then this node’s output is
approximated (`approxT`) with error at most `bound`.
-/
structure FwdNode (toSpec : α → SpecScalar) (Γ : List Shape) (τ : Shape) where
  /-- forward Spec. -/
  forwardSpec : TList SpecScalar Γ → SpecTensor τ
  /-- forward Runtime. -/
  forwardRuntime : TList α Γ → Tensor α τ
  /-- bound. -/
  bound : EList Γ → TList α Γ → SpecScalar
  /-- sound. -/
  sound : ∀ (xS : TList SpecScalar Γ) (xR : TList α Γ) (eps : EList Γ),
      approxCtx (α := α) toSpec xS xR eps →
        approxT (α := α) (toSpec := toSpec) (forwardSpec xS) (forwardRuntime xR) (bound eps xR)

/--
Forward-only SSA/DAG graph (nodes appended in topological order).

The type parameter `ss : List Shape` tracks the shapes of intermediate values produced by the
graph; evaluating a graph returns an extended context of shape list `Γ ++ ss`.
-/
inductive FwdGraph (toSpec : α → SpecScalar) (Γ : List Shape) : List Shape → Type where
  | nil : FwdGraph toSpec Γ []
  | snoc {ss : List Shape} {τ : Shape} :
      FwdGraph toSpec Γ ss → FwdNode (α := α) toSpec (Γ := Γ ++ ss) τ → FwdGraph toSpec Γ (ss ++
        [τ])

namespace FwdGraph

variable {toSpec : α → SpecScalar}

/--
Evaluate a forward graph in the **spec** semantics.

Result type: an extended context `Γ ++ ss` containing the original inputs and all intermediate
values produced by the graph.
-/
def evalSpec {Γ : List Shape} {ss : List Shape} (g : FwdGraph (α := α) toSpec Γ ss) (x : TList
  SpecScalar Γ) :
    TList SpecScalar (Γ ++ ss) :=
  match g with
  | .nil =>
      let h : Γ = Γ ++ [] := (List.append_nil Γ).symm
      TList.cast (α := SpecScalar) (ss₁ := Γ) (ss₂ := Γ ++ []) h x
  | .snoc (ss := ssPrev) (τ := τ) g node =>
      let ctx := evalSpec (Γ := Γ) (ss := ssPrev) g x
      let y := node.forwardSpec ctx
      let hAssoc : (Γ ++ ssPrev) ++ [τ] = Γ ++ (ssPrev ++ [τ]) := List.append_assoc Γ ssPrev [τ]
      TList.cast (α := SpecScalar) (ss₁ := (Γ ++ ssPrev) ++ [τ]) (ss₂ := Γ ++ (ssPrev ++ [τ]))
        hAssoc
        (TList.snoc (α := SpecScalar) (ss := Γ ++ ssPrev) ctx y)

/--
Evaluate a forward graph in the **runtime** semantics.

This mirrors `evalSpec`, but uses the backend `α` tensors and the node runtime closures.
-/
def evalRuntime {Γ : List Shape} {ss : List Shape} (g : FwdGraph (α := α) toSpec Γ ss) (x : TList α
  Γ) :
    TList α (Γ ++ ss) :=
  match g with
  | .nil =>
      let h : Γ = Γ ++ [] := (List.append_nil Γ).symm
      TList.cast (α := α) (ss₁ := Γ) (ss₂ := Γ ++ []) h x
  | .snoc (ss := ssPrev) (τ := τ) g node =>
      let ctx := evalRuntime (Γ := Γ) (ss := ssPrev) g x
      let y := node.forwardRuntime ctx
      let hAssoc : (Γ ++ ssPrev) ++ [τ] = Γ ++ (ssPrev ++ [τ]) := List.append_assoc Γ ssPrev [τ]
      TList.cast (α := α) (ss₁ := (Γ ++ ssPrev) ++ [τ]) (ss₂ := Γ ++ (ssPrev ++ [τ])) hAssoc
        (TList.snoc (α := α) (ss := Γ ++ ssPrev) ctx y)

/--
Propagate an input error list `epsIn` through the whole graph, producing output bounds for
`Γ ++ ss`.

Each node can compute its own output bound from the current context bounds and the runtime context;
`evalBounds` just composes those local transformers over the snoc-list DAG.
-/
def evalBounds {Γ : List Shape} {ss : List Shape} (g : FwdGraph (α := α) toSpec Γ ss)
    (epsIn : EList Γ) (xR : TList α Γ) : EList (Γ ++ ss) :=
  match g with
  | .nil =>
      let h : Γ = Γ ++ [] := (List.append_nil Γ).symm
      EList.cast (ss₁ := Γ) (ss₂ := Γ ++ []) h epsIn
  | .snoc (ss := ssPrev) (τ := τ) g node =>
      let epsPrev := evalBounds (Γ := Γ) (ss := ssPrev) g epsIn xR
      let ctxR := evalRuntime (Γ := Γ) (ss := ssPrev) g xR
      let e := node.bound epsPrev ctxR
      let hAssoc : (Γ ++ ssPrev) ++ [τ] = Γ ++ (ssPrev ++ [τ]) := List.append_assoc Γ ssPrev [τ]
      EList.cast (ss₁ := (Γ ++ ssPrev) ++ [τ]) (ss₂ := Γ ++ (ssPrev ++ [τ])) hAssoc
        (EList.snoc (ss := Γ ++ ssPrev) (τ := τ) epsPrev e)

/--
End-to-end forward approximation theorem for `FwdGraph`.

Informally:
assume every input tensor in the runtime context `xR` is within the provided per-entry bounds
`epsIn` of the corresponding spec tensor in `xS`. Then evaluating the whole graph preserves that
approximation relation, with output bounds given by `evalBounds`.

Proof idea: induction over the snoc-list graph; at each step, apply the node's local bound/soundness
lemma (`FwdNode.sound`) and then extend the context approximation via `approxCtx_snoc`.
-/
theorem eval_approx {Γ : List Shape} {ss : List Shape} (g : FwdGraph (α := α) toSpec Γ ss) :
    ∀ (xS : TList SpecScalar Γ) (xR : TList α Γ) (epsIn : EList Γ),
      approxCtx (α := α) toSpec xS xR epsIn →
        approxCtx (α := α) toSpec
          (evalSpec (Γ := Γ) (ss := ss) g xS)
          (evalRuntime (Γ := Γ) (ss := ss) g xR)
          (evalBounds (Γ := Γ) (ss := ss) g epsIn xR) := by
  intro xS xR epsIn hIn
  induction g generalizing xS xR epsIn with
  | nil =>
      -- `eval*` are casts along `Γ = Γ ++ []`.
      simpa [evalSpec, evalRuntime, evalBounds] using
        (approxCtx_cast (α := α) (toSpec := toSpec) (h := (List.append_nil Γ).symm) hIn)
  | snoc g node ih =>
      rename_i ssPrev τ
      -- IH gives approximation for the previous context.
      have hPrev :
          approxCtx (α := α) toSpec
            (evalSpec (Γ := Γ) (ss := ssPrev) g xS)
            (evalRuntime (Γ := Γ) (ss := ssPrev) g xR)
            (evalBounds (Γ := Γ) (ss := ssPrev) g epsIn xR) :=
        ih xS xR epsIn hIn

      let ctxS := evalSpec (Γ := Γ) (ss := ssPrev) g xS
      let ctxR := evalRuntime (Γ := Γ) (ss := ssPrev) g xR
      let epsPrev := evalBounds (Γ := Γ) (ss := ssPrev) g epsIn xR

      have hy :
          approxT (α := α) (toSpec := toSpec)
            (node.forwardSpec ctxS)
            (node.forwardRuntime ctxR)
            (node.bound epsPrev ctxR) :=
        node.sound ctxS ctxR epsPrev (by simpa [ctxS, ctxR, epsPrev] using hPrev)

      -- Extend the context approximation with the new node output.
      have hSnoc :
          approxCtx (α := α) toSpec
            (TList.snoc (α := SpecScalar) (ss := Γ ++ ssPrev) ctxS (node.forwardSpec ctxS))
            (TList.snoc (α := α) (ss := Γ ++ ssPrev) ctxR (node.forwardRuntime ctxR))
            (EList.snoc (ss := Γ ++ ssPrev) (τ := τ) epsPrev (node.bound epsPrev ctxR)) :=
        approxCtx_snoc (α := α) (toSpec := toSpec) (hx := by simpa [ctxS, ctxR, epsPrev] using
          hPrev) hy

      -- Cast to match the `Γ ++ (ssPrev ++ [τ])` shape.
      simpa [evalSpec, evalRuntime, evalBounds, ctxS, ctxR, epsPrev, List.append_assoc] using
        (approxCtx_cast (α := α) (toSpec := toSpec) (h := List.append_assoc Γ ssPrev [τ]) hSnoc)

end FwdGraph

end

end RuntimeApprox
end Proofs
