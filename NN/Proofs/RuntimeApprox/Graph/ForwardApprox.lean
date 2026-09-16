/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.RuntimeApprox.Core.SpecApprox
public import NN.Tensor.Pack
-- `Idx` and `getIdx`: one definition of a typed context index, shared with the tape proofs.
public import NN.Proofs.Autograd.Tape.Util.Idx
public import NN.Proofs.Autograd.Tape.Algebra.Soundness -- shake: keep

/-!
# ForwardApprox

Forward (runtime→spec) approximation framework.

This file is **backend-agnostic**: it proves that approximation bounds compose over a
tape/SSA-style graph, assuming each node provides a local forward approximation lemma.

It is intended to be instantiated by rounding models such as the noncomputable
`FloatLib.Floats.Formats.Flocq.NF`, whose arithmetic uses `Flocq.round`. Lean gives builtin `Float`
a logical model, but connecting that model and its runtime implementation to these approximation
bounds requires separate per-operation proofs.

## What you get
- `FwdGraph.eval_approx`: an end-to-end theorem saying that if the runtime input context is within
  an explicit per-entry error budget of the spec input context, then runtime graph evaluation
  is within a computable propagated error budget of the spec evaluation.

## Reading guide
1. `Autograd.Algebra.TensorPack` and `EList`: heterogeneous contexts and aligned error vectors.
2. `approxTensor` and `approxCtx`: the approximation predicates for a single tensor and a whole
   context.
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

open Spec TorchLean
open NN.MLTheory.Robustness.Spec
open Proofs.Autograd.Algebra

noncomputable section

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

/-- Erase shape indices while retaining bounds in context order.

This is the reporting boundary used by architecture-independent numerical traces. Proofs continue
to use `EList`; UI and serialization code receives an array only after propagation has
finished.
-/
def toArray : {ss : List Shape} → EList ss → Array ℝ
  | [], .nil => #[]
  | _ :: _, .cons error errors => #[error] ++ toArray errors

/-- Flattening an error list to an array keeps one entry per shape. -/
@[simp] theorem size_toArray {ss : List Shape} (errors : EList ss) :
    errors.toArray.size = ss.length := by
  induction errors with
  | nil => rfl
  | cons _ _ ih => simp [toArray, ih, Nat.add_comm]

end EList

-- Tensor and context approximation predicates for forward runtime graphs.

variable {α : Type} [TorchLean.Storage α]

/-- Tensor-level approximation under a `toSpec : α → ℝ` mapping. -/
def approxTensor {s : Shape} (toSpec : α → SpecScalar) (spec : SpecTensor s) (runtime : Tensor α s)
    (eps : SpecScalar) : Prop :=
  approxWith (α := α) (toSpec := toSpec) (norm := linfNorm) spec runtime eps

/--
Scoped notation for tensor approximation (`approxTensor`).

```lean
open scoped RuntimeApprox
spec ≈ᵀ[toSpec] runtime : eps
```
-/
scoped[RuntimeApprox] notation:50 spec " ≈ᵀ[" toSpec "] " runtime " : " eps =>
  Proofs.RuntimeApprox.approxTensor (toSpec := toSpec) spec runtime eps

/--
Monotonicity of `approxTensorWithTol`: if you only loosen tolerances, an approximation stays valid.
-/
theorem approxTensorWithTol_mono {s : Shape} {toSpec : α → SpecScalar}
    {spec : SpecTensor s} {runtime : Tensor α s} {tol₁ tol₂ : ApproxTol}
    (habs : tol₁.abs ≤ tol₂.abs) (hrel : tol₁.rel ≤ tol₂.rel) (hslack : tol₁.slack ≤ tol₂.slack)
    (h : approxTensorWithTol (toSpec := toSpec) spec runtime tol₁) :
    approxTensorWithTol (toSpec := toSpec) spec runtime tol₂ :=
  approx_with_tol_mono (toSpec := toSpec) (norm := linfNorm)
    (spec := spec) (runtime := runtime) habs hrel hslack h

/--
`approxTensorWithTol` specialized to an absolute-only tolerance is equivalent to `approxTensor`.

This is mostly a convenience lemma for switching between the "tolerance" API and the plain
`eps : ℝ` API.
-/
theorem approxTensorWithTol_absOnly_iff {s : Shape} {toSpec : α → SpecScalar}
    {spec : SpecTensor s} {runtime : Tensor α s} {eps : ℝ} (heps : 0 ≤ eps) :
    approxTensorWithTol (toSpec := toSpec) spec runtime (ApproxTol.absOnly eps) ↔
      approxTensor (toSpec := toSpec) spec runtime eps := by
  -- Reduce to the `approx_with` lemma.
  simpa [approxTensor, approxTensorWithTol] using
    (approx_with_tol_absOnly_iff (toSpec := toSpec) (norm := linfNorm)
      (spec := spec) (runtime := runtime) heps)

/-- Context-level approximation with a per-entry error list. -/
def approxCtx (toSpec : α → SpecScalar) : {ss : List Shape} →
    TorchLean.TensorPack SpecScalar ss → TorchLean.TensorPack α ss → EList ss → Prop
  | [], .nil, .nil, .nil => True
  | _ :: ss, .cons x xs, .cons y ys, .cons e es =>
      approxTensor (α := α) (toSpec := toSpec) x y e ∧ approxCtx (ss := ss) toSpec xs ys es

/--
Scoped notation for `approxCtx`.

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
theorem approxCtx_cast {toSpec : α → SpecScalar} {ss₁ ss₂ : List Shape} (h : ss₁ = ss₂)
    {xS : TorchLean.TensorPack SpecScalar ss₁} {xR : TorchLean.TensorPack α ss₁} {eps : EList ss₁} :
    approxCtx (α := α) toSpec xS xR eps →
      approxCtx (α := α) toSpec
        (TorchLean.TensorPack.cast (α := SpecScalar) (ss₁ := ss₁) (ss₂ := ss₂) h xS)
        (TorchLean.TensorPack.cast (α := α) (ss₁ := ss₁) (ss₂ := ss₂) h xR)
        (EList.cast (ss₁ := ss₁) (ss₂ := ss₂) h eps) := by
  cases h
  simp

/--
Extend a context approximation by appending one more approximated tensor.

This is the core "composition" step used when evaluating a snoc-graph: if the previous context is
approximated, and the new node output is approximated with some bound `e`, then the extended context
is approximated with the extended error list.
-/
theorem approxCtx_snoc {toSpec : α → SpecScalar} {ss : List Shape} {τ : Shape}
    {xS : TorchLean.TensorPack SpecScalar ss} {xR : TorchLean.TensorPack α ss} {eps : EList ss}
    (hx : approxCtx (α := α) toSpec xS xR eps)
    {yS : SpecTensor τ} {yR : Tensor α τ} {e : SpecScalar}
    (hy : approxTensor (α := α) (toSpec := toSpec) yS yR e) :
    approxCtx (α := α) toSpec
      (TorchLean.TensorPack.snoc (α := SpecScalar) (ss := ss) xS yS)
      (TorchLean.TensorPack.snoc (α := α) (ss := ss) xR yR)
      (EList.snoc (ss := ss) (τ := τ) eps e) := by
  induction ss with
  | nil =>
      cases xS
      cases xR
      cases eps
      simpa [TorchLean.TensorPack.snoc, EList.snoc, approxCtx] using And.intro hy True.intro
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
theorem approxCtx_get {toSpec : α → SpecScalar} {Γ : List Shape}
    {xS : TorchLean.TensorPack SpecScalar Γ} {xR : TorchLean.TensorPack α Γ} {eps : EList Γ}
    (h : approxCtx (α := α) toSpec xS xR eps) (i : Fin Γ.length) :
    approxTensor (α := α) (toSpec := toSpec)
      (TorchLean.TensorPack.get (α := SpecScalar) xS i)
      (TorchLean.TensorPack.get (α := α) xR i)
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
                          change approxTensor toSpec xSh xRh eh
                          exact h.1
                      | succ j =>
                          have := ih (xS := xSt) (xR := xRt) (eps := et) h.2
                            ⟨j, Nat.lt_of_succ_lt_succ hiVal⟩
                          simpa [TorchLean.TensorPack.get, EList.get] using this

/--
`approxCtx_get` expressed in terms of `approxTensorWithTol` with an absolute-only tolerance.

Many downstream theorems are stated using a tolerance record (`ApproxTol`) rather than a bare
`eps : ℝ`. For absolute-only bounds, this lemma gives the bridge.
-/
theorem approxCtx_get_tolAbsOnly {toSpec : α → SpecScalar} {Γ : List Shape}
    {xS : TorchLean.TensorPack SpecScalar Γ} {xR : TorchLean.TensorPack α Γ} {eps : EList Γ}
    (h : approxCtx (α := α) toSpec xS xR eps) (i : Fin Γ.length) :
    approxTensorWithTol (α := α) (toSpec := toSpec)
      (TorchLean.TensorPack.get (α := SpecScalar) xS i)
      (TorchLean.TensorPack.get (α := α) xR i)
      (ApproxTol.absOnly (EList.get eps i)) := by
  have hi :
      approxTensor (α := α) (toSpec := toSpec)
        (TorchLean.TensorPack.get (α := SpecScalar) xS i)
        (TorchLean.TensorPack.get (α := α) xR i)
        (EList.get eps i) :=
    approxCtx_get (α := α) (toSpec := toSpec) (xS := xS) (xR := xR) (eps := eps) h i
  have : approxWith (α := α) (toSpec := toSpec) (norm := linfNorm)
      (TorchLean.TensorPack.get (α := SpecScalar) xS i)
      (TorchLean.TensorPack.get (α := α) xR i)
      (EList.get eps i) := by
    simpa [approxTensor] using hi
  simpa using
    (approxTensor_to_approxTensorWithTol_absOnly (toSpec := toSpec)
      (spec := (TorchLean.TensorPack.get (α := SpecScalar) xS i))
      (runtime := (TorchLean.TensorPack.get (α := α) xR i))
      (eps := (EList.get eps i)) this)

/--
Split a context approximation for `ss ++ [τ]` into:
- a prefix context approximation for `ss`, and
- a single-tensor approximation for the last entry of shape `τ`.
-/
theorem approxCtx_unsnoc {toSpec : α → SpecScalar} {ss : List Shape} {τ : Shape}
    {xS : TorchLean.TensorPack SpecScalar (ss ++ [τ])} {xR : TorchLean.TensorPack α (ss ++ [τ])}
    {eps : EList (ss ++ [τ])} :
    approxCtx (α := α) toSpec xS xR eps →
      approxCtx (α := α) toSpec
          (TorchLean.TensorPack.unsnoc (α := SpecScalar) (ss := ss) (τ := τ) xS).1
          (TorchLean.TensorPack.unsnoc (α := α) (ss := ss) (τ := τ) xR).1
          (EList.unsnoc (ss := ss) (τ := τ) eps).1
        ∧
      approxTensor (α := α) (toSpec := toSpec)
          (TorchLean.TensorPack.unsnoc (α := SpecScalar) (ss := ss) (τ := τ) xS).2
          (TorchLean.TensorPack.unsnoc (α := α) (ss := ss) (τ := τ) xR).2
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
                              · simp [TorchLean.TensorPack.unsnoc, EList.unsnoc, approxCtx]
                              · simpa [TorchLean.TensorPack.unsnoc, EList.unsnoc, approxCtx]
                                  using h.1
  | cons s ss ih =>
      cases xS with
      | cons xSh xSt =>
          cases xR with
          | cons xRh xRt =>
              cases eps with
              | cons eh et =>
                  have hx : approxTensor (α := α) (toSpec := toSpec) xSh xRh eh := h.1
                  have ht : approxCtx (α := α) toSpec xSt xRt et := h.2
                  have ih' := ih (xS := xSt) (xR := xRt) (eps := et) ht
                  refine And.intro ?_ ?_
                  · simpa [TorchLean.TensorPack.unsnoc, EList.unsnoc, approxCtx]
                      using And.intro hx ih'.1
                  · simpa [TorchLean.TensorPack.unsnoc, EList.unsnoc] using ih'.2

-- ---------------------------------------------------------------------------
-- Typed indexing into contexts (for building graphs)
-- ---------------------------------------------------------------------------

/-- Lookup the epsilon entry associated to an index `Idx Γ s`. -/
def getIdxEps {Γ : List Shape} {s : Shape} (es : EList Γ) (idx : Idx Γ s) : ℝ :=
  EList.get es idx.i

/--
Context approximation implies approximation of any indexed entry.

Informally: if every tensor in the runtime context is close to its spec counterpart (with an
aligned error list `eps`), then reading any entry `idx : Idx Γ s` yields an `approxTensor` fact
with the corresponding scalar bound `getIdxEps eps idx`.
-/
theorem approxCtx_getIdx {toSpec : α → SpecScalar} {Γ : List Shape} {s : Shape}
    {xS : TorchLean.TensorPack SpecScalar Γ} {xR : TorchLean.TensorPack α Γ} {eps : EList Γ}
    (h : approxCtx (α := α) toSpec xS xR eps) (idx : Idx Γ s) :
    approxTensor (α := α) (toSpec := toSpec)
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
approximated (`approxTensor`) with error at most `bound`.
-/
structure FwdNode (toSpec : α → SpecScalar) (Γ : List Shape) (τ : Shape) where
  /-- Specification-level semantics of this node. -/
  forwardSpec : TorchLean.TensorPack SpecScalar Γ → SpecTensor τ
  /-- Runtime semantics of this node. -/
  forwardRuntime : TorchLean.TensorPack α Γ → Tensor α τ
  /-- Error bound computed from the current context bounds and runtime values. -/
  bound : EList Γ → TorchLean.TensorPack α Γ → SpecScalar
  /-- Local approximation theorem for this node. -/
  sound : ∀ (xS : TorchLean.TensorPack SpecScalar Γ) (xR : TorchLean.TensorPack α Γ)
      (eps : EList Γ),
      approxCtx (α := α) toSpec xS xR eps →
        approxTensor (α := α) (toSpec := toSpec) (forwardSpec xS) (forwardRuntime xR) (bound eps xR)

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
def evalSpec {Γ : List Shape} {ss : List Shape} (g : FwdGraph (α := α) toSpec Γ ss)
    (x : TorchLean.TensorPack SpecScalar Γ) :
    TorchLean.TensorPack SpecScalar (Γ ++ ss) :=
  match g with
  | .nil =>
      let h : Γ = Γ ++ [] := (List.append_nil Γ).symm
      TorchLean.TensorPack.cast (α := SpecScalar) (ss₁ := Γ) (ss₂ := Γ ++ []) h x
  | .snoc (ss := ssPrev) (τ := τ) g node =>
      let ctx := evalSpec (Γ := Γ) (ss := ssPrev) g x
      let y := node.forwardSpec ctx
      let hAssoc : (Γ ++ ssPrev) ++ [τ] = Γ ++ (ssPrev ++ [τ]) := List.append_assoc Γ ssPrev [τ]
      TorchLean.TensorPack.cast (α := SpecScalar) (ss₁ := (Γ ++ ssPrev) ++ [τ])
        (ss₂ := Γ ++ (ssPrev ++ [τ])) hAssoc
        (TorchLean.TensorPack.snoc (α := SpecScalar) (ss := Γ ++ ssPrev) ctx y)

/--
Evaluate a forward graph in the **runtime** semantics.

This mirrors `evalSpec`, but uses the backend `α` tensors and the node runtime closures.
-/
def evalRuntime {Γ : List Shape} {ss : List Shape} (g : FwdGraph (α := α) toSpec Γ ss)
    (x : TorchLean.TensorPack α Γ) :
    TorchLean.TensorPack α (Γ ++ ss) :=
  match g with
  | .nil =>
      let h : Γ = Γ ++ [] := (List.append_nil Γ).symm
      TorchLean.TensorPack.cast (α := α) (ss₁ := Γ) (ss₂ := Γ ++ []) h x
  | .snoc (ss := ssPrev) (τ := τ) g node =>
      let ctx := evalRuntime (Γ := Γ) (ss := ssPrev) g x
      let y := node.forwardRuntime ctx
      let hAssoc : (Γ ++ ssPrev) ++ [τ] = Γ ++ (ssPrev ++ [τ]) := List.append_assoc Γ ssPrev [τ]
      TorchLean.TensorPack.cast (α := α) (ss₁ := (Γ ++ ssPrev) ++ [τ])
        (ss₂ := Γ ++ (ssPrev ++ [τ])) hAssoc
        (TorchLean.TensorPack.snoc (α := α) (ss := Γ ++ ssPrev) ctx y)

/--
Propagate an input error list `epsIn` through the whole graph, producing output bounds for
`Γ ++ ss`.

Each node can compute its own output bound from the current context bounds and the runtime context;
`evalBounds` just composes those local transformers over the snoc-list DAG.
-/
def evalBounds {Γ : List Shape} {ss : List Shape} (g : FwdGraph (α := α) toSpec Γ ss)
    (epsIn : EList Γ) (xR : TorchLean.TensorPack α Γ) : EList (Γ ++ ss) :=
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
theorem (`FwdNode.sound`) and then extend the context approximation via `approxCtx_snoc`.
-/
theorem eval_approx {Γ : List Shape} {ss : List Shape} (g : FwdGraph (α := α) toSpec Γ ss) :
    ∀ (xS : TorchLean.TensorPack SpecScalar Γ) (xR : TorchLean.TensorPack α Γ) (epsIn : EList Γ),
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
          approxTensor (α := α) (toSpec := toSpec)
            (node.forwardSpec ctxS)
            (node.forwardRuntime ctxR)
            (node.bound epsPrev ctxR) :=
        node.sound ctxS ctxR epsPrev (by simpa [ctxS, ctxR, epsPrev] using hPrev)

      -- Extend the context approximation with the new node output.
      have hSnoc :
          approxCtx (α := α) toSpec
            (TorchLean.TensorPack.snoc (α := SpecScalar) (ss := Γ ++ ssPrev) ctxS
              (node.forwardSpec ctxS))
            (TorchLean.TensorPack.snoc (α := α) (ss := Γ ++ ssPrev) ctxR (node.forwardRuntime ctxR))
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
