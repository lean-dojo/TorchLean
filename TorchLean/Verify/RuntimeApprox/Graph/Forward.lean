/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.RuntimeApprox
-- `Idx` and `getIdx`: one definition of a typed context index, shared with the tape proofs.
public import TorchLean.NN.Tape.Graph -- shake: keep

/-!
# Forward graph approximation

`FwdGraph.eval_approx` composes the local bounds that `FwdNode.sound` supplies. Each node receives
a heterogeneous context of tensors and an aligned `EList` of scalar error bounds. The shape list
indexes the context by the shapes of its entries, and the tensors hold the data.

The framework does not depend on the backend and admits noncomputable rounding models such as
FloatLib's `NF`, so a propagated bound need not be an executable numerical calculation.
Instantiating the node contracts with native arithmetic takes a separate correspondence proof for
each operation.
-/

@[expose] public section

namespace TorchLean

open Robustness

noncomputable section

-- ---------------------------------------------------------------------------
-- Error vectors aligned with contexts
-- ---------------------------------------------------------------------------

/-- Scalar error bounds aligned with a context shape list. -/
abbrev EList : List Shape → Type := Pack fun _ : Shape => ℝ

-- Tensor and context approximation predicates for forward runtime graphs.

variable {α : Type} [Storage α]

/-- `ApproxTensor toSpec spec runtime eps`: `runtime`, read through `toSpec`, is within `eps` of
`spec` in the `L∞` norm. -/
def ApproxTensor {s : Shape} (toSpec : α → ℝ) (spec : Tensor ℝ s) (runtime : Tensor α s)
    (eps : ℝ) : Prop :=
  ApproxWith (α := α) (toSpec := toSpec) (norm := linfNorm) spec runtime eps

/--
Scoped notation for tensor approximation (`ApproxTensor`).

```lean
open scoped TorchLean.ApproxTensor
spec ≈ᵀ[toSpec] runtime : eps
```
-/
scoped[TorchLean.ApproxTensor] notation:50 spec " ≈ᵀ[" toSpec "] " runtime " : " eps =>
  ApproxTensor (toSpec := toSpec) spec runtime eps

/-- Entrywise approximation of contexts: at each position the runtime tensor approximates the spec
tensor (`ApproxTensor`) with the bound at that position of the `EList`. -/
def ApproxCtx (toSpec : α → ℝ) : {ss : List Shape} →
    TensorPack ℝ ss → TensorPack α ss → EList ss → Prop
  | [], .nil, .nil, .nil => True
  | _ :: ss, .cons x xs, .cons y ys, .cons e es =>
      ApproxTensor (α := α) (toSpec := toSpec) x y e ∧ ApproxCtx (ss := ss) toSpec xs ys es

/--
Scoped notation for `ApproxCtx`.

```lean
open scoped TorchLean.ApproxTensor
-- `ΓS` is an approximate view of `ΓR` with per-entry bounds `eps`:
ΓS ≈ᶜ[toSpec] ΓR : eps
```

The notation is scoped to avoid polluting unrelated code.
-/
scoped[TorchLean.ApproxTensor] notation:50 ΓS " ≈ᶜ[" toSpec "] " ΓR " : " eps =>
  ApproxCtx (toSpec := toSpec) ΓS ΓR eps

/--
Transport a context approximation across an equality of shape lists. Reassociating the `Γ ++ ss`
indices needs it, since casts are unavoidable in this `List Shape`-indexed encoding.
-/
theorem ApproxCtx.cast {toSpec : α → ℝ} {ss₁ ss₂ : List Shape} (h : ss₁ = ss₂)
    {xS : TensorPack ℝ ss₁} {xR : TensorPack α ss₁} {eps : EList ss₁} :
    ApproxCtx (α := α) toSpec xS xR eps →
      ApproxCtx (α := α) toSpec
        (Pack.cast (ss₁ := ss₁) (ss₂ := ss₂) h xS)
        (Pack.cast (ss₁ := ss₁) (ss₂ := ss₂) h xR)
        (Pack.cast (ss₁ := ss₁) (ss₂ := ss₂) h eps) := by
  cases h
  simp

/--
If a context is approximated with bounds `eps` and a tensor with bound `e`, then the context
extended by that tensor is approximated with `eps` extended by `e`. The induction step of
`eval_approx` uses it to add each node's output.
-/
theorem ApproxCtx.snoc {toSpec : α → ℝ} {ss : List Shape} {τ : Shape}
    {xS : TensorPack ℝ ss} {xR : TensorPack α ss} {eps : EList ss}
    (hx : ApproxCtx (α := α) toSpec xS xR eps)
    {yS : Tensor ℝ τ} {yR : Tensor α τ} {e : ℝ}
    (hy : ApproxTensor (α := α) (toSpec := toSpec) yS yR e) :
    ApproxCtx (α := α) toSpec
      (Pack.snoc (ss := ss) xS yS)
      (Pack.snoc (ss := ss) xR yR)
      (Pack.snoc (ss := ss) (τ := τ) eps e) := by
  induction ss with
  | nil =>
      cases xS
      cases xR
      cases eps
      simpa [Pack.snoc, ApproxCtx] using And.intro hy True.intro
  | cons s ss ih =>
      cases xS with
      | cons xSh xSt =>
          cases xR with
          | cons xRh xRt =>
              cases eps with
              | cons eh et =>
                  have hx' : ApproxCtx (α := α) toSpec xSt xRt et := hx.2
                  have ih' := ih hx'
                  exact And.intro hx.1 ih'

/-- If `ApproxCtx` holds with bounds `eps`, the `i`th runtime entry approximates the `i`th spec
entry with bound `Pack.get eps i`. -/
theorem ApproxCtx.get {toSpec : α → ℝ} {Γ : List Shape}
    {xS : TensorPack ℝ Γ} {xR : TensorPack α Γ} {eps : EList Γ}
    (h : ApproxCtx (α := α) toSpec xS xR eps) (i : Fin Γ.length) :
    ApproxTensor (α := α) (toSpec := toSpec)
      (Pack.get xS i)
      (Pack.get xR i)
      (Pack.get eps i) := by
  induction Γ with
  | nil => exact i.elim0
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
                          change ApproxTensor toSpec xSh xRh eh
                          exact h.1
                      | succ j =>
                          have := ih (xS := xSt) (xR := xRt) (eps := et) h.2
                            ⟨j, Nat.lt_of_succ_lt_succ hiVal⟩
                          simpa [Pack.get] using this

/--
`get` restated with the absolute-only tolerance `ApproxTol.absOnly (Pack.get eps i)`, as an
`ApproxTensorWithTol` bound.
-/
theorem ApproxCtx.get_tolAbsOnly {toSpec : α → ℝ} {Γ : List Shape}
    {xS : TensorPack ℝ Γ} {xR : TensorPack α Γ} {eps : EList Γ}
    (h : ApproxCtx (α := α) toSpec xS xR eps) (i : Fin Γ.length) :
    ApproxTensorWithTol (α := α) (toSpec := toSpec)
      (Pack.get xS i)
      (Pack.get xR i)
      (ApproxTol.absOnly (Pack.get eps i)) :=
  ApproxWith.approxTensorWithTol_absOnly (toSpec := toSpec) (Pack.get eps i)
    (get (α := α) (toSpec := toSpec) h i)

/--
Split a context approximation for `ss ++ [τ]` into:
- a prefix context approximation for `ss`, and
- a single-tensor approximation for the last entry of shape `τ`.
-/
theorem ApproxCtx.unsnoc {toSpec : α → ℝ} {ss : List Shape} {τ : Shape}
    {xS : TensorPack ℝ (ss ++ [τ])} {xR : TensorPack α (ss ++ [τ])}
    {eps : EList (ss ++ [τ])} :
    ApproxCtx (α := α) toSpec xS xR eps →
      ApproxCtx (α := α) toSpec
          (Pack.unsnoc (ss := ss) (τ := τ) xS).1
          (Pack.unsnoc (ss := ss) (τ := τ) xR).1
          (Pack.unsnoc (ss := ss) (τ := τ) eps).1
        ∧
      ApproxTensor (α := α) (toSpec := toSpec)
          (Pack.unsnoc (ss := ss) (τ := τ) xS).2
          (Pack.unsnoc (ss := ss) (τ := τ) xR).2
          (Pack.unsnoc (ss := ss) (τ := τ) eps).2 := by
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
                              · simp [Pack.unsnoc, ApproxCtx]
                              · simpa [Pack.unsnoc, ApproxCtx]
                                  using h.1
  | cons s ss ih =>
      cases xS with
      | cons xSh xSt =>
          cases xR with
          | cons xRh xRt =>
              cases eps with
              | cons eh et =>
                  have hx : ApproxTensor (α := α) (toSpec := toSpec) xSh xRh eh := h.1
                  have ht : ApproxCtx (α := α) toSpec xSt xRt et := h.2
                  have ih' := ih (xS := xSt) (xR := xRt) (eps := et) ht
                  refine And.intro ?_ ?_
                  · simpa [Pack.unsnoc, ApproxCtx]
                      using And.intro hx ih'.1
                  · simpa [Pack.unsnoc] using ih'.2

-- ---------------------------------------------------------------------------
-- Typed indexing into contexts (for building graphs)
-- ---------------------------------------------------------------------------

/-- The error bound at a typed index `Idx Γ s`. -/
def EList.getIdx {Γ : List Shape} {s : Shape} (es : EList Γ) (idx : Idx Γ s) : ℝ :=
  Pack.get es idx.i

/-- Read an indexed tensor approximation with its aligned scalar error bound. -/
theorem ApproxCtx.getIdx {toSpec : α → ℝ} {Γ : List Shape} {s : Shape}
    {xS : TensorPack ℝ Γ} {xR : TensorPack α Γ} {eps : EList Γ}
    (h : ApproxCtx (α := α) toSpec xS xR eps) (idx : Idx Γ s) :
    ApproxTensor (α := α) (toSpec := toSpec)
      (TensorPack.getIdx (α := ℝ) xS idx)
      (TensorPack.getIdx (α := α) xR idx)
      (EList.getIdx (Γ := Γ) (s := s) eps idx) := by
  cases idx with
  | mk i hshape =>
      have h' := get (α := α) (toSpec := toSpec) (xS := xS) (xR := xR) (eps := eps) h i
      cases hshape
      simpa [Idx, TensorPack.getIdx, EList.getIdx] using h'

-- ---------------------------------------------------------------------------
-- A forward-only tape/SSA graph with local approximation lemmas
-- ---------------------------------------------------------------------------

/--
A node of a forward SSA graph, with its spec and runtime semantics, a scalar error bound for its
output and the local theorem `sound` behind that bound. If the input context is approximated with
bounds `eps` (`ApproxCtx`), the output is approximated with error at most `bound eps xR`
(`ApproxTensor`). The bound reads the runtime context, so it can depend on the data.
-/
structure FwdNode (toSpec : α → ℝ) (Γ : List Shape) (τ : Shape) where
  /-- Specification-level semantics of this node. -/
  forwardReal : TensorPack ℝ Γ → Tensor ℝ τ
  /-- Runtime semantics of this node. -/
  forwardRuntime : TensorPack α Γ → Tensor α τ
  /-- Error bound computed from the current context bounds and runtime values. -/
  bound : EList Γ → TensorPack α Γ → ℝ
  /-- Local approximation theorem for this node. -/
  sound : ∀ (xS : TensorPack ℝ Γ) (xR : TensorPack α Γ)
      (eps : EList Γ),
      ApproxCtx (α := α) toSpec xS xR eps →
        ApproxTensor (α := α) (toSpec := toSpec) (forwardReal xS) (forwardRuntime xR) (bound eps xR)

/--
A forward SSA graph over the inputs `Γ`, with nodes appended in topological order. The index
`ss : List Shape` lists the shapes of the values the nodes produce, so evaluating the graph returns
a context of shape list `Γ ++ ss`.
-/
inductive FwdGraph (toSpec : α → ℝ) (Γ : List Shape) : List Shape → Type where
  /-- The graph with no node. -/
  | nil : FwdGraph toSpec Γ []
  /-- A graph followed by one more node over its inputs and values. -/
  | snoc {ss : List Shape} {τ : Shape} :
      FwdGraph toSpec Γ ss → FwdNode (α := α) toSpec (Γ := Γ ++ ss) τ → FwdGraph toSpec Γ (ss ++
        [τ])

namespace FwdGraph

variable {toSpec : α → ℝ}

/--
Evaluate a forward graph in the spec semantics. The result is the context `Γ ++ ss`: the inputs
followed by every value the graph produces.
-/
def evalReal {Γ : List Shape} {ss : List Shape} (g : FwdGraph (α := α) toSpec Γ ss)
    (x : TensorPack ℝ Γ) :
    TensorPack ℝ (Γ ++ ss) :=
  match g with
  | .nil =>
      let h : Γ = Γ ++ [] := (List.append_nil Γ).symm
      Pack.cast (ss₁ := Γ) (ss₂ := Γ ++ []) h x
  | .snoc (ss := ssPrev) (τ := τ) g node =>
      let ctx := evalReal (Γ := Γ) (ss := ssPrev) g x
      let y := node.forwardReal ctx
      let hAssoc : (Γ ++ ssPrev) ++ [τ] = Γ ++ (ssPrev ++ [τ]) := List.append_assoc Γ ssPrev [τ]
      Pack.cast (ss₁ := (Γ ++ ssPrev) ++ [τ])
        (ss₂ := Γ ++ (ssPrev ++ [τ])) hAssoc
        (Pack.snoc (ss := Γ ++ ssPrev) ctx y)

/--
Evaluate a forward graph in the runtime semantics, as `evalReal` does but on the backend's `α`
tensors with each node's `forwardRuntime`.
-/
def evalRuntime {Γ : List Shape} {ss : List Shape} (g : FwdGraph (α := α) toSpec Γ ss)
    (x : TensorPack α Γ) :
    TensorPack α (Γ ++ ss) :=
  match g with
  | .nil =>
      let h : Γ = Γ ++ [] := (List.append_nil Γ).symm
      Pack.cast (ss₁ := Γ) (ss₂ := Γ ++ []) h x
  | .snoc (ss := ssPrev) (τ := τ) g node =>
      let ctx := evalRuntime (Γ := Γ) (ss := ssPrev) g x
      let y := node.forwardRuntime ctx
      let hAssoc : (Γ ++ ssPrev) ++ [τ] = Γ ++ (ssPrev ++ [τ]) := List.append_assoc Γ ssPrev [τ]
      Pack.cast (ss₁ := (Γ ++ ssPrev) ++ [τ])
        (ss₂ := Γ ++ (ssPrev ++ [τ])) hAssoc
        (Pack.snoc (ss := Γ ++ ssPrev) ctx y)

/--
Propagate an input error list `epsIn` through the whole graph, producing bounds for `Γ ++ ss`.
Each node computes its output bound from the current bounds and the runtime context, and
`evalBounds` composes these local transformers along the graph.
-/
def evalBounds {Γ : List Shape} {ss : List Shape} (g : FwdGraph (α := α) toSpec Γ ss)
    (epsIn : EList Γ) (xR : TensorPack α Γ) : EList (Γ ++ ss) :=
  match g with
  | .nil =>
      let h : Γ = Γ ++ [] := (List.append_nil Γ).symm
      Pack.cast (ss₁ := Γ) (ss₂ := Γ ++ []) h epsIn
  | .snoc (ss := ssPrev) (τ := τ) g node =>
      let epsPrev := evalBounds (Γ := Γ) (ss := ssPrev) g epsIn xR
      let ctxR := evalRuntime (Γ := Γ) (ss := ssPrev) g xR
      let e := node.bound epsPrev ctxR
      let hAssoc : (Γ ++ ssPrev) ++ [τ] = Γ ++ (ssPrev ++ [τ]) := List.append_assoc Γ ssPrev [τ]
      Pack.cast (ss₁ := (Γ ++ ssPrev) ++ [τ]) (ss₂ := Γ ++ (ssPrev ++ [τ])) hAssoc
        (Pack.snoc (ss := Γ ++ ssPrev) (τ := τ) epsPrev e)

/--
End-to-end forward approximation for `FwdGraph`: if every input tensor in `xR` is within its
bound in `epsIn` of the matching spec tensor in `xS`, then the evaluated runtime context is within
the bounds `evalBounds g epsIn xR` of the evaluated spec context.

The proof is an induction over the graph. Each step applies the node's local theorem
`FwdNode.sound` and extends the context approximation with `ApproxCtx.snoc`.
-/
theorem eval_approx {Γ : List Shape} {ss : List Shape} (g : FwdGraph (α := α) toSpec Γ ss) :
    ∀ (xS : TensorPack ℝ Γ) (xR : TensorPack α Γ) (epsIn : EList Γ),
      ApproxCtx (α := α) toSpec xS xR epsIn →
        ApproxCtx (α := α) toSpec
          (evalReal (Γ := Γ) (ss := ss) g xS)
          (evalRuntime (Γ := Γ) (ss := ss) g xR)
          (evalBounds (Γ := Γ) (ss := ss) g epsIn xR) := by
  intro xS xR epsIn hIn
  induction g generalizing xS xR epsIn with
  | nil =>
      -- With no nodes, `evalReal`, `evalRuntime` and `evalBounds` are casts along `Γ = Γ ++ []`.
      simpa [evalReal, evalRuntime, evalBounds] using
        (ApproxCtx.cast (α := α) (toSpec := toSpec) (h := (List.append_nil Γ).symm) hIn)
  | snoc g node ih =>
      rename_i ssPrev τ
      have hPrev :
          ApproxCtx (α := α) toSpec
            (evalReal (Γ := Γ) (ss := ssPrev) g xS)
            (evalRuntime (Γ := Γ) (ss := ssPrev) g xR)
            (evalBounds (Γ := Γ) (ss := ssPrev) g epsIn xR) :=
        ih xS xR epsIn hIn

      let ctxS := evalReal (Γ := Γ) (ss := ssPrev) g xS
      let ctxR := evalRuntime (Γ := Γ) (ss := ssPrev) g xR
      let epsPrev := evalBounds (Γ := Γ) (ss := ssPrev) g epsIn xR

      have hy :
          ApproxTensor (α := α) (toSpec := toSpec)
            (node.forwardReal ctxS)
            (node.forwardRuntime ctxR)
            (node.bound epsPrev ctxR) :=
        node.sound ctxS ctxR epsPrev (by simpa [ctxS, ctxR, epsPrev] using hPrev)

      -- Extend the context approximation with the node's output.
      have hSnoc :
          ApproxCtx (α := α) toSpec
            (Pack.snoc (ss := Γ ++ ssPrev) ctxS
              (node.forwardReal ctxS))
            (Pack.snoc (ss := Γ ++ ssPrev) ctxR (node.forwardRuntime ctxR))
            (Pack.snoc (ss := Γ ++ ssPrev) (τ := τ) epsPrev (node.bound epsPrev ctxR)) :=
        ApproxCtx.snoc (α := α) (toSpec := toSpec) (hx := by simpa [ctxS, ctxR, epsPrev] using
          hPrev) hy

      -- Cast to match the `Γ ++ (ssPrev ++ [τ])` shape.
      simpa [evalReal, evalRuntime, evalBounds, ctxS, ctxR, epsPrev, List.append_assoc] using
        (ApproxCtx.cast (α := α) (toSpec := toSpec) (h := List.append_assoc Γ ssPrev [τ]) hSnoc)

end FwdGraph

end

end TorchLean
