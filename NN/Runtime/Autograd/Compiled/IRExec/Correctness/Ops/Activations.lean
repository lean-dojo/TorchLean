/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Compiled.IRExec.Correctness.SemanticEquivalenceCommon

set_option linter.unusedSimpArgs false
set_option linter.unnecessarySimpa false

/-!
# Activation Operators

Semantic-preservation lemmas for unary activation operators in the IR -> compiled runtime bridge.

Each lemma mirrors the corresponding branch in `Correctness/SemanticEquivalence.lean`, but is
named and factored out so we can keep the main semantic equivalence proof focused on graph
traversal rather than repeating parent-list and typed-index boilerplate.

Build note: these proofs can be slower than the operators look. The activation itself is simple;
the proof cost comes from checking the singleton-parent contract, recovering a typed index from the
IR parent id, and showing that the dynamically evaluated `DVal` is the same value as the compiled
node output. The shared unary-operator skeleton keeps each activation
only supplies its tensor function.
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace Compiled

open Spec
open Tensor
open Proofs.Autograd.Algebra
open NN.IR
open IRExec

/-- Semantic-preservation lemma for `.relu` lowering. -/
theorem buildFrom_denoteAllFrom_relu
    {α : Type} [Context α] [DecidableEq Shape]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : GraphData α Unit [inShape] ss) (i : Nat) (st' : State α inShape)
    (x : Tensor α inShape) (n : NN.IR.Node)
    (hN : g.getNode i = .ok n) (hk : n.kind = .relu) (hi : i < g.nodes.size)
    (hBuild :
      buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
        (i := i) (st := (⟨ss, gd⟩ : State α inShape)) = .ok st')
    (ih :
      ∀ (st1 : State α inShape),
        buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
          (i := i + 1) st1 = .ok st' →
        NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
          (input := NN.IR.DVal.mk (α := α) inShape x)
          (i := i + 1) (vals := denoteAllState (α := α) inShape st1 x) =
          .ok (denoteAllState (α := α) inShape st' x)) :
    NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
      (input := NN.IR.DVal.mk (α := α) inShape x)
      (i := i) (vals := denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x) =
      .ok (denoteAllState (α := α) inShape st' x) := by
  let vals0 : Array (NN.IR.DVal α) :=
    denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x
  let ctx : TList α ([inShape] ++ ss) :=
    GraphData.eval (α := α) (Δ := Unit) (Γ := [inShape]) (ss := ss) gd (.cons x .nil) ()
  let input : NN.IR.DVal α := NN.IR.DVal.mk (α := α) inShape x

  unfold buildFrom at hBuild
  simp (config := { failIfUnchanged := false }) [hi, hN, hk] at hBuild
  cases hp : n.parents with
  | nil =>
      simp [hp] at hBuild
      try cases hBuild
  | cons pId ps =>
      cases ps with
      | cons _ _ =>
          simp [hp] at hBuild
          try cases hBuild
      | nil =>
          cases hIdx : mkIdx (inShape := inShape) (ss := ss) pId n.outShape with
          | error msg =>
              simp [hp, hIdx] at hBuild
              try cases hBuild
          | ok ip =>
              simp [hp, hIdx] at hBuild
              let nodeData : NodeData α Unit ([inShape] ++ ss) n.outShape :=
                mkFwdNode (α := α) (Γ := [inShape] ++ ss) (τ := n.outShape) (fun ctx =>
                  Activation.reluSpec (α := α) (getIdx (α := α) (xs := ctx) ip))
              let st1 : State α inShape := ⟨ss ++ [n.outShape], .snoc (ss := ss) gd nodeData⟩
              have hRec :
                  buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
                      (i := i + 1) st1 =
                    .ok st' := by
                simpa [st1, nodeData] using hBuild
              have hTail := ih st1 hRec
              have hGet :
                  vals0[pId]! =
                    NN.IR.DVal.mk (α := α) n.outShape (getIdx (α := α) (xs := ctx) ip) := by
                simpa [vals0, ctx] using
                  (denoteAllState_get_mkIdx (inShape := inShape) (ss := ss)
                    (gd := gd) (x := x) (pid := pId) (s := n.outShape) (idx := ip) hIdx)
              have hEval :
                  NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
                      (input := input) (vals := vals0) (i := i) =
                    .ok (NN.IR.DVal.mk (α := α) n.outShape (nodeData.forward ctx ())) := by
                have hExp :
                    NN.IR.Graph.expectShape (α := α) (expected := n.outShape) vals0[pId]! =
                      .ok (getIdx (α := α) (xs := ctx) ip) := by
                  simpa [hGet, NN.IR.DVal.mk] using
                    (Graph.expectShape_sigma (α := α) (s := n.outShape)
                      (t := getIdx (α := α) (xs := ctx) ip))
                simp [NN.IR.Graph.evalAt, hN, hk, hp, hExp,
                  nodeData, mkFwdNode, NN.IR.DVal.shape, NN.IR.DVal.tensor, NN.IR.DVal.mk,
                  throw_eq_error, Except.instMonad, Except.bind, Except.pure]
              exact buildFrom_denoteAllFrom_unary_exact (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (hi := hi)
                (τ := n.outShape) (nodeData := nodeData) hTail hEval

/-- Semantic-preservation lemma for `.tanh` lowering. -/
theorem buildFrom_denoteAllFrom_tanh
    {α : Type} [Context α] [DecidableEq Shape]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : GraphData α Unit [inShape] ss) (i : Nat) (st' : State α inShape)
    (x : Tensor α inShape) (n : NN.IR.Node)
    (hN : g.getNode i = .ok n) (hk : n.kind = .tanh) (hi : i < g.nodes.size)
    (hBuild :
      buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
        (i := i) (st := (⟨ss, gd⟩ : State α inShape)) = .ok st')
    (ih :
      ∀ (st1 : State α inShape),
        buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
          (i := i + 1) st1 = .ok st' →
        NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
          (input := NN.IR.DVal.mk (α := α) inShape x)
          (i := i + 1) (vals := denoteAllState (α := α) inShape st1 x) =
          .ok (denoteAllState (α := α) inShape st' x)) :
    NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
      (input := NN.IR.DVal.mk (α := α) inShape x)
      (i := i) (vals := denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x) =
      .ok (denoteAllState (α := α) inShape st' x) := by
  let vals0 : Array (NN.IR.DVal α) :=
    denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x
  let ctx : TList α ([inShape] ++ ss) :=
    GraphData.eval (α := α) (Δ := Unit) (Γ := [inShape]) (ss := ss) gd (.cons x .nil) ()
  let input : NN.IR.DVal α := NN.IR.DVal.mk (α := α) inShape x

  unfold buildFrom at hBuild
  simp (config := { failIfUnchanged := false }) [hi, hN, hk] at hBuild
  cases hp : n.parents with
  | nil =>
      simp [hp] at hBuild
      try cases hBuild
  | cons pId ps =>
      cases ps with
      | cons _ _ =>
          simp [hp] at hBuild
          try cases hBuild
      | nil =>
          cases hIdx : mkIdx (inShape := inShape) (ss := ss) pId n.outShape with
          | error msg =>
              simp [hp, hIdx] at hBuild
              try cases hBuild
          | ok ip =>
              simp [hp, hIdx] at hBuild
              let nodeData : NodeData α Unit ([inShape] ++ ss) n.outShape :=
                mkFwdNode (α := α) (Γ := [inShape] ++ ss) (τ := n.outShape) (fun ctx =>
                  Activation.tanhSpec (α := α) (getIdx (α := α) (xs := ctx) ip))
              let st1 : State α inShape := ⟨ss ++ [n.outShape], .snoc (ss := ss) gd nodeData⟩
              have hRec :
                  buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
                      (i := i + 1) st1 =
                    .ok st' := by
                simpa [st1, nodeData] using hBuild
              have hTail := ih st1 hRec
              have hGet :
                  vals0[pId]! =
                    NN.IR.DVal.mk (α := α) n.outShape (getIdx (α := α) (xs := ctx) ip) := by
                simpa [vals0, ctx] using
                  (denoteAllState_get_mkIdx (inShape := inShape) (ss := ss)
                    (gd := gd) (x := x) (pid := pId) (s := n.outShape) (idx := ip) hIdx)
              have hEval :
                  NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
                      (input := input) (vals := vals0) (i := i) =
                    .ok (NN.IR.DVal.mk (α := α) n.outShape (nodeData.forward ctx ())) := by
                simp [NN.IR.Graph.evalAt, hN, hk, hp, hGet, Graph.expectShape_mk,
                  nodeData, mkFwdNode, NN.IR.DVal.shape, NN.IR.DVal.tensor, NN.IR.DVal.mk,
                  throw_eq_error, Except.instMonad, Except.bind, Except.pure]
              exact buildFrom_denoteAllFrom_unary_exact (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (hi := hi)
                (τ := n.outShape) (nodeData := nodeData) hTail hEval

/-- Semantic-preservation lemma for `.sigmoid` lowering. -/
theorem buildFrom_denoteAllFrom_sigmoid
    {α : Type} [Context α] [DecidableEq Shape]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : GraphData α Unit [inShape] ss) (i : Nat) (st' : State α inShape)
    (x : Tensor α inShape) (n : NN.IR.Node)
    (hN : g.getNode i = .ok n) (hk : n.kind = .sigmoid) (hi : i < g.nodes.size)
    (hBuild :
      buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
        (i := i) (st := (⟨ss, gd⟩ : State α inShape)) = .ok st')
    (ih :
      ∀ (st1 : State α inShape),
        buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
          (i := i + 1) st1 = .ok st' →
        NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
          (input := NN.IR.DVal.mk (α := α) inShape x)
          (i := i + 1) (vals := denoteAllState (α := α) inShape st1 x) =
          .ok (denoteAllState (α := α) inShape st' x)) :
    NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
      (input := NN.IR.DVal.mk (α := α) inShape x)
      (i := i) (vals := denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x) =
      .ok (denoteAllState (α := α) inShape st' x) := by
  let vals0 : Array (NN.IR.DVal α) :=
    denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x
  let ctx : TList α ([inShape] ++ ss) :=
    GraphData.eval (α := α) (Δ := Unit) (Γ := [inShape]) (ss := ss) gd (.cons x .nil) ()
  let input : NN.IR.DVal α := NN.IR.DVal.mk (α := α) inShape x

  unfold buildFrom at hBuild
  simp (config := { failIfUnchanged := false }) [hi, hN, hk] at hBuild
  cases hp : n.parents with
  | nil =>
      simp [hp] at hBuild
      try cases hBuild
  | cons pId ps =>
      cases ps with
      | cons _ _ =>
          simp [hp] at hBuild
          try cases hBuild
      | nil =>
          cases hIdx : mkIdx (inShape := inShape) (ss := ss) pId n.outShape with
          | error msg =>
              simp [hp, hIdx] at hBuild
              try cases hBuild
          | ok ip =>
              simp [hp, hIdx] at hBuild
              let nodeData : NodeData α Unit ([inShape] ++ ss) n.outShape :=
                mkFwdNode (α := α) (Γ := [inShape] ++ ss) (τ := n.outShape) (fun ctx =>
                  Activation.sigmoidSpec (α := α) (getIdx (α := α) (xs := ctx) ip))
              let st1 : State α inShape := ⟨ss ++ [n.outShape], .snoc (ss := ss) gd nodeData⟩
              have hRec :
                  buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
                      (i := i + 1) st1 =
                    .ok st' := by
                simpa [st1, nodeData] using hBuild
              have hTail := ih st1 hRec
              have hGet :
                  vals0[pId]! =
                    NN.IR.DVal.mk (α := α) n.outShape (getIdx (α := α) (xs := ctx) ip) := by
                simpa [vals0, ctx] using
                  (denoteAllState_get_mkIdx (inShape := inShape) (ss := ss)
                    (gd := gd) (x := x) (pid := pId) (s := n.outShape) (idx := ip) hIdx)
              have hEval :
                  NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
                      (input := input) (vals := vals0) (i := i) =
                    .ok (NN.IR.DVal.mk (α := α) n.outShape (nodeData.forward ctx ())) := by
                simp [NN.IR.Graph.evalAt, hN, hk, hp, hGet, Graph.expectShape_mk,
                  nodeData, mkFwdNode, NN.IR.DVal.shape, NN.IR.DVal.tensor, NN.IR.DVal.mk,
                  throw_eq_error, Except.instMonad, Except.bind, Except.pure]
              exact buildFrom_denoteAllFrom_unary_exact (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (hi := hi)
                (τ := n.outShape) (nodeData := nodeData) hTail hEval

/-- Semantic-preservation lemma for `.exp` lowering. -/
theorem buildFrom_denoteAllFrom_exp
    {α : Type} [Context α] [DecidableEq Shape]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : GraphData α Unit [inShape] ss) (i : Nat) (st' : State α inShape)
    (x : Tensor α inShape) (n : NN.IR.Node)
    (hN : g.getNode i = .ok n) (hk : n.kind = .exp) (hi : i < g.nodes.size)
    (hBuild :
      buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
        (i := i) (st := (⟨ss, gd⟩ : State α inShape)) = .ok st')
    (ih :
      ∀ (st1 : State α inShape),
        buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
          (i := i + 1) st1 = .ok st' →
        NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
          (input := NN.IR.DVal.mk (α := α) inShape x)
          (i := i + 1) (vals := denoteAllState (α := α) inShape st1 x) =
          .ok (denoteAllState (α := α) inShape st' x)) :
    NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
      (input := NN.IR.DVal.mk (α := α) inShape x)
      (i := i) (vals := denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x) =
      .ok (denoteAllState (α := α) inShape st' x) := by
  let vals0 : Array (NN.IR.DVal α) :=
    denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x
  let ctx : TList α ([inShape] ++ ss) :=
    GraphData.eval (α := α) (Δ := Unit) (Γ := [inShape]) (ss := ss) gd (.cons x .nil) ()
  let input : NN.IR.DVal α := NN.IR.DVal.mk (α := α) inShape x

  unfold buildFrom at hBuild
  simp (config := { failIfUnchanged := false }) [hi, hN, hk] at hBuild
  cases hp : n.parents with
  | nil =>
      simp [hp] at hBuild
      try cases hBuild
  | cons pId ps =>
      cases ps with
      | cons _ _ =>
          simp [hp] at hBuild
          try cases hBuild
      | nil =>
          cases hIdx : mkIdx (inShape := inShape) (ss := ss) pId n.outShape with
          | error msg =>
              simp [hp, hIdx] at hBuild
              try cases hBuild
          | ok ip =>
              simp [hp, hIdx] at hBuild
              let nodeData : NodeData α Unit ([inShape] ++ ss) n.outShape :=
                mkFwdNode (α := α) (Γ := [inShape] ++ ss) (τ := n.outShape) (fun ctx =>
                  Tensor.expSpec (α := α) (getIdx (α := α) (xs := ctx) ip))
              let st1 : State α inShape := ⟨ss ++ [n.outShape], .snoc (ss := ss) gd nodeData⟩
              have hRec :
                  buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
                      (i := i + 1) st1 =
                    .ok st' := by
                simpa [st1, nodeData] using hBuild
              have hTail := ih st1 hRec
              have hGet :
                  vals0[pId]! =
                    NN.IR.DVal.mk (α := α) n.outShape (getIdx (α := α) (xs := ctx) ip) := by
                simpa [vals0, ctx] using
                  (denoteAllState_get_mkIdx (inShape := inShape) (ss := ss)
                    (gd := gd) (x := x) (pid := pId) (s := n.outShape) (idx := ip) hIdx)
              have hEval :
                  NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
                      (input := input) (vals := vals0) (i := i) =
                    .ok (NN.IR.DVal.mk (α := α) n.outShape (nodeData.forward ctx ())) := by
                simp [NN.IR.Graph.evalAt, hN, hk, hp, hGet, Graph.expectShape_mk,
                  nodeData, mkFwdNode, NN.IR.DVal.shape, NN.IR.DVal.tensor, NN.IR.DVal.mk,
                  throw_eq_error, Except.instMonad, Except.bind, Except.pure]
              exact buildFrom_denoteAllFrom_unary_exact (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (hi := hi)
                (τ := n.outShape) (nodeData := nodeData) hTail hEval

/-- Semantic-preservation lemma for `.log` lowering. -/
theorem buildFrom_denoteAllFrom_log
    {α : Type} [Context α] [DecidableEq Shape]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : GraphData α Unit [inShape] ss) (i : Nat) (st' : State α inShape)
    (x : Tensor α inShape) (n : NN.IR.Node)
    (hN : g.getNode i = .ok n) (hk : n.kind = .log) (hi : i < g.nodes.size)
    (hBuild :
      buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
        (i := i) (st := (⟨ss, gd⟩ : State α inShape)) = .ok st')
    (ih :
      ∀ (st1 : State α inShape),
        buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
          (i := i + 1) st1 = .ok st' →
        NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
          (input := NN.IR.DVal.mk (α := α) inShape x)
          (i := i + 1) (vals := denoteAllState (α := α) inShape st1 x) =
          .ok (denoteAllState (α := α) inShape st' x)) :
    NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
      (input := NN.IR.DVal.mk (α := α) inShape x)
      (i := i) (vals := denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x) =
      .ok (denoteAllState (α := α) inShape st' x) := by
  let vals0 : Array (NN.IR.DVal α) :=
    denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x
  let ctx : TList α ([inShape] ++ ss) :=
    GraphData.eval (α := α) (Δ := Unit) (Γ := [inShape]) (ss := ss) gd (.cons x .nil) ()
  let input : NN.IR.DVal α := NN.IR.DVal.mk (α := α) inShape x

  unfold buildFrom at hBuild
  simp (config := { failIfUnchanged := false }) [hi, hN, hk] at hBuild
  cases hp : n.parents with
  | nil =>
      simp [hp] at hBuild
      try cases hBuild
  | cons pId ps =>
      cases ps with
      | cons _ _ =>
          simp [hp] at hBuild
          try cases hBuild
      | nil =>
          cases hIdx : mkIdx (inShape := inShape) (ss := ss) pId n.outShape with
          | error msg =>
              simp [hp, hIdx] at hBuild
              try cases hBuild
          | ok ip =>
              simp [hp, hIdx] at hBuild
              let nodeData : NodeData α Unit ([inShape] ++ ss) n.outShape :=
                mkFwdNode (α := α) (Γ := [inShape] ++ ss) (τ := n.outShape) (fun ctx =>
                  let x := getIdx (α := α) (xs := ctx) ip
                  -- `IRExec.buildFrom` inserts a runtime-domain check for raw `log` to match
                  -- the eager autograd engine behavior. In Lean, `panic!` reduces to
                  -- `Inhabited.default`, so this becomes a checked `if` with a default fallback.
                  if Tensor.allSpec (α := α) (s := n.outShape) (fun v => decide (0 < v)) x = true then
                    Tensor.logSpec (α := α) x
                  else
                    (Inhabited.default : Tensor α n.outShape))
              let st1 : State α inShape := ⟨ss ++ [n.outShape], .snoc (ss := ss) gd nodeData⟩
              have hRec :
                  buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
                      (i := i + 1) st1 =
                    .ok st' := by
                simpa [st1, nodeData] using hBuild
              have hTail := ih st1 hRec
              have hGet :
                  vals0[pId]! =
                    NN.IR.DVal.mk (α := α) n.outShape (getIdx (α := α) (xs := ctx) ip) := by
                simpa [vals0, ctx] using
                  (denoteAllState_get_mkIdx (inShape := inShape) (ss := ss)
                    (gd := gd) (x := x) (pid := pId) (s := n.outShape) (idx := ip) hIdx)
              have hEval :
                  NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
                      (input := input) (vals := vals0) (i := i) =
                    .ok (NN.IR.DVal.mk (α := α) n.outShape (nodeData.forward ctx ())) := by
                simp [NN.IR.Graph.evalAt, hN, hk, hp, hGet, Graph.expectShape_mk,
                  nodeData, mkFwdNode, NN.IR.DVal.shape, NN.IR.DVal.tensor, NN.IR.DVal.mk,
                  throw_eq_error, Except.instMonad, Except.bind, Except.pure]
              exact buildFrom_denoteAllFrom_unary_exact (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (hi := hi)
                (τ := n.outShape) (nodeData := nodeData) hTail hEval

/-- Semantic-preservation lemma for `.sin` lowering. -/
theorem buildFrom_denoteAllFrom_sin
    {α : Type} [Context α] [DecidableEq Shape]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : GraphData α Unit [inShape] ss) (i : Nat) (st' : State α inShape)
    (x : Tensor α inShape) (n : NN.IR.Node)
    (hN : g.getNode i = .ok n) (hk : n.kind = .sin) (hi : i < g.nodes.size)
    (hBuild :
      buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
        (i := i) (st := (⟨ss, gd⟩ : State α inShape)) = .ok st')
    (ih :
      ∀ (st1 : State α inShape),
        buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
          (i := i + 1) st1 = .ok st' →
        NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
          (input := NN.IR.DVal.mk (α := α) inShape x)
          (i := i + 1) (vals := denoteAllState (α := α) inShape st1 x) =
          .ok (denoteAllState (α := α) inShape st' x)) :
    NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
      (input := NN.IR.DVal.mk (α := α) inShape x)
      (i := i) (vals := denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x) =
      .ok (denoteAllState (α := α) inShape st' x) := by
  let vals0 : Array (NN.IR.DVal α) :=
    denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x
  let ctx : TList α ([inShape] ++ ss) :=
    GraphData.eval (α := α) (Δ := Unit) (Γ := [inShape]) (ss := ss) gd (.cons x .nil) ()
  let input : NN.IR.DVal α := NN.IR.DVal.mk (α := α) inShape x

  unfold buildFrom at hBuild
  simp (config := { failIfUnchanged := false }) [hi, hN, hk] at hBuild
  cases hp : n.parents with
  | nil =>
      simp [hp] at hBuild
      try cases hBuild
  | cons pId ps =>
      cases ps with
      | cons _ _ =>
          simp [hp] at hBuild
          try cases hBuild
      | nil =>
          cases hIdx : mkIdx (inShape := inShape) (ss := ss) pId n.outShape with
          | error msg =>
              simp [hp, hIdx] at hBuild
              try cases hBuild
          | ok ip =>
              simp [hp, hIdx] at hBuild
              let nodeData : NodeData α Unit ([inShape] ++ ss) n.outShape :=
                mkFwdNode (α := α) (Γ := [inShape] ++ ss) (τ := n.outShape) (fun ctx =>
                  Tensor.mapSpec (α := α) (s := n.outShape) (fun x => MathFunctions.sin x)
                    (getIdx (α := α) (xs := ctx) ip))
              let st1 : State α inShape := ⟨ss ++ [n.outShape], .snoc (ss := ss) gd nodeData⟩
              have hRec :
                  buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
                      (i := i + 1) st1 =
                    .ok st' := by
                simpa [st1, nodeData] using hBuild
              have hTail := ih st1 hRec
              have hGet :
                  vals0[pId]! =
                    NN.IR.DVal.mk (α := α) n.outShape (getIdx (α := α) (xs := ctx) ip) := by
                simpa [vals0, ctx] using
                  (denoteAllState_get_mkIdx (inShape := inShape) (ss := ss)
                    (gd := gd) (x := x) (pid := pId) (s := n.outShape) (idx := ip) hIdx)
              have hEval :
                  NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
                      (input := input) (vals := vals0) (i := i) =
                    .ok (NN.IR.DVal.mk (α := α) n.outShape (nodeData.forward ctx ())) := by
                simp [NN.IR.Graph.evalAt, hN, hk, hp, hGet, Graph.expectShape_mk,
                  nodeData, mkFwdNode, NN.IR.DVal.shape, NN.IR.DVal.tensor, NN.IR.DVal.mk,
                  throw_eq_error, Except.instMonad, Except.bind, Except.pure]
              exact buildFrom_denoteAllFrom_unary_exact (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (hi := hi)
                (τ := n.outShape) (nodeData := nodeData) hTail hEval

/-- Semantic-preservation lemma for `.cos` lowering. -/
theorem buildFrom_denoteAllFrom_cos
    {α : Type} [Context α] [DecidableEq Shape]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : GraphData α Unit [inShape] ss) (i : Nat) (st' : State α inShape)
    (x : Tensor α inShape) (n : NN.IR.Node)
    (hN : g.getNode i = .ok n) (hk : n.kind = .cos) (hi : i < g.nodes.size)
    (hBuild :
      buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
        (i := i) (st := (⟨ss, gd⟩ : State α inShape)) = .ok st')
    (ih :
      ∀ (st1 : State α inShape),
        buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
          (i := i + 1) st1 = .ok st' →
        NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
          (input := NN.IR.DVal.mk (α := α) inShape x)
          (i := i + 1) (vals := denoteAllState (α := α) inShape st1 x) =
          .ok (denoteAllState (α := α) inShape st' x)) :
    NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
      (input := NN.IR.DVal.mk (α := α) inShape x)
      (i := i) (vals := denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x) =
      .ok (denoteAllState (α := α) inShape st' x) := by
  let vals0 : Array (NN.IR.DVal α) :=
    denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x
  let ctx : TList α ([inShape] ++ ss) :=
    GraphData.eval (α := α) (Δ := Unit) (Γ := [inShape]) (ss := ss) gd (.cons x .nil) ()
  let input : NN.IR.DVal α := NN.IR.DVal.mk (α := α) inShape x

  unfold buildFrom at hBuild
  simp (config := { failIfUnchanged := false }) [hi, hN, hk] at hBuild
  cases hp : n.parents with
  | nil =>
      simp [hp] at hBuild
      try cases hBuild
  | cons pId ps =>
      cases ps with
      | cons _ _ =>
          simp [hp] at hBuild
          try cases hBuild
      | nil =>
          cases hIdx : mkIdx (inShape := inShape) (ss := ss) pId n.outShape with
          | error msg =>
              simp [hp, hIdx] at hBuild
              try cases hBuild
          | ok ip =>
              simp [hp, hIdx] at hBuild
              let nodeData : NodeData α Unit ([inShape] ++ ss) n.outShape :=
                mkFwdNode (α := α) (Γ := [inShape] ++ ss) (τ := n.outShape) (fun ctx =>
                  Tensor.mapSpec (α := α) (s := n.outShape) (fun x => MathFunctions.cos x)
                    (getIdx (α := α) (xs := ctx) ip))
              let st1 : State α inShape := ⟨ss ++ [n.outShape], .snoc (ss := ss) gd nodeData⟩
              have hRec :
                  buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
                      (i := i + 1) st1 =
                    .ok st' := by
                simpa [st1, nodeData] using hBuild
              have hTail := ih st1 hRec
              have hGet :
                  vals0[pId]! =
                    NN.IR.DVal.mk (α := α) n.outShape (getIdx (α := α) (xs := ctx) ip) := by
                simpa [vals0, ctx] using
                  (denoteAllState_get_mkIdx (inShape := inShape) (ss := ss)
                    (gd := gd) (x := x) (pid := pId) (s := n.outShape) (idx := ip) hIdx)
              have hEval :
                  NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
                      (input := input) (vals := vals0) (i := i) =
                    .ok (NN.IR.DVal.mk (α := α) n.outShape (nodeData.forward ctx ())) := by
                simp [NN.IR.Graph.evalAt, hN, hk, hp, hGet, Graph.expectShape_mk,
                  nodeData, mkFwdNode, NN.IR.DVal.shape, NN.IR.DVal.tensor, NN.IR.DVal.mk,
                  throw_eq_error, Except.instMonad, Except.bind, Except.pure]
              exact buildFrom_denoteAllFrom_unary_exact (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (hi := hi)
                (τ := n.outShape) (nodeData := nodeData) hTail hEval

/--
Semantic-preservation lemma for `.softmax axis` lowering.

Implementation note: TorchLean's compiled softmax operator supports the last axis.
This is reflected by an explicit guard in `buildFrom`:
`axis + 1 = Shape.rank outShape` (equivalently, `axis = rank-1`).
-/
theorem buildFrom_denoteAllFrom_softmax
    {α : Type} [Context α] [DecidableEq Shape]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : GraphData α Unit [inShape] ss) (i : Nat) (st' : State α inShape)
    (x : Tensor α inShape) (n : NN.IR.Node)
    (axis : Nat)
    (hN : g.getNode i = .ok n) (hk : n.kind = .softmax axis) (hi : i < g.nodes.size)
    (hBuild :
      buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
        (i := i) (st := (⟨ss, gd⟩ : State α inShape)) = .ok st')
    (ih :
      ∀ (st1 : State α inShape),
        buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
          (i := i + 1) st1 = .ok st' →
        NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
          (input := NN.IR.DVal.mk (α := α) inShape x)
          (i := i + 1) (vals := denoteAllState (α := α) inShape st1 x) =
          .ok (denoteAllState (α := α) inShape st' x)) :
    NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
      (input := NN.IR.DVal.mk (α := α) inShape x)
      (i := i) (vals := denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x) =
      .ok (denoteAllState (α := α) inShape st' x) := by
  let vals0 : Array (NN.IR.DVal α) :=
    denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x
  let ctx : TList α ([inShape] ++ ss) :=
    GraphData.eval (α := α) (Δ := Unit) (Γ := [inShape]) (ss := ss) gd (.cons x .nil) ()
  let input : NN.IR.DVal α := NN.IR.DVal.mk (α := α) inShape x

  unfold buildFrom at hBuild
  simp (config := { failIfUnchanged := false }) [hi, hN, hk] at hBuild
  cases hp : n.parents with
  | nil =>
      simp [hp] at hBuild
      try cases hBuild
  | cons pId ps =>
      cases ps with
      | cons _ _ =>
          simp [hp] at hBuild
          try cases hBuild
      | nil =>
          cases hAxis : OpContracts.checkLastAxis "softmax" axis n.outShape with
          | error msg =>
              simp [hp, hAxis] at hBuild
              try cases hBuild
          | ok _ =>
              simp (config := { failIfUnchanged := false }) [hp, hAxis] at hBuild
              cases hIdx : mkIdx (inShape := inShape) (ss := ss) pId n.outShape with
              | error msg =>
                  simp [hp, hIdx] at hBuild
                  try cases hBuild
              | ok ip =>
                  simp [hp, hIdx] at hBuild
                  have hAxisLast : axis + 1 = Shape.rank n.outShape := by
                    have hAxis' := hAxis
                    unfold OpContracts.checkLastAxis at hAxis'
                    -- `checkLastAxis` succeeds only if its internal `if` is taken.
                    cases hAxisValid : OpContracts.checkAxisValid axis n.outShape <;>
                      simp [hAxisValid] at hAxis'
                    · by_cases h : axis + 1 = Shape.rank n.outShape
                      · exact h
                      · simp [h] at hAxis'
                  have hAxisLt : axis < Shape.rank n.outShape := by
                    have : axis < axis + 1 := Nat.lt_succ_self axis
                    simpa [hAxisLast] using this

                  let nodeData : NodeData α Unit ([inShape] ++ ss) n.outShape :=
                    mkFwdNode (α := α) (Γ := [inShape] ++ ss) (τ := n.outShape) (fun ctx =>
                      Activation.softmaxSpec (α := α) (getIdx (α := α) (xs := ctx) ip))
                  let st1 : State α inShape := ⟨ss ++ [n.outShape], .snoc (ss := ss) gd nodeData⟩
                  have hRec :
                      buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
                        (i := i + 1) st1 = .ok st' := by
                    simpa [st1, nodeData] using hBuild
                  have hGet :
                      vals0[pId]! =
                        NN.IR.DVal.mk (α := α) n.outShape (getIdx (α := α) (xs := ctx) ip) := by
                    simpa [vals0, ctx] using
                      (denoteAllState_get_mkIdx (inShape := inShape) (ss := ss)
                        (gd := gd) (x := x) (pid := pId) (s := n.outShape) (idx := ip) hIdx)
                  have hExp :
                      NN.IR.Graph.expectShape (α := α) (expected := n.outShape) vals0[pId]! =
                        .ok (getIdx (α := α) (xs := ctx) ip) := by
                    simpa [hGet, NN.IR.DVal.mk] using
                      (Graph.expectShape_sigma (α := α) (s := n.outShape)
                        (t := getIdx (α := α) (xs := ctx) ip))
                  have hEval :
                      NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
                          (input := input) (vals := vals0) (i := i) =
                        .ok (NN.IR.DVal.mk (α := α) n.outShape (nodeData.forward ctx ())) := by
                    -- `checkLastAxis` success implies `checkAxisValid` success and last-axis equality.
                    simp [NN.IR.Graph.evalAt, hN, hk, hp, hExp, throw_eq_error,
                      OpContracts.checkAxisValid, hAxisLt, hAxisLast, Pure.pure, Except.pure, nodeData,
                      mkFwdNode]
                  have hStep :
                      denoteAllState (α := α) inShape st1 x =
                        vals0.push (NN.IR.DVal.mk (α := α) n.outShape (nodeData.forward ctx ())) := by
                    simpa [vals0, st1, nodeData, ctx] using
                      (denoteAllState_snoc (α := α) (inShape := inShape) (ss := ss) (τ := n.outShape)
                        (gd := gd) (nodeData := nodeData) (x := x))
                  have hTail := ih st1 hRec
                  exact buildFrom_denoteAllFrom_finish (α := α) (g := g) (payload := payload)
                    (i := i) (x := x) (hi := hi) (τ := n.outShape)
                    (nodeData := nodeData) (st1 := st1) (st' := st')
                    (ctx := ctx) (vals0 := vals0) (input := input) hTail hEval hStep

end Compiled
end Autograd
end Runtime
