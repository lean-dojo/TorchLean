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
# Elementwise Operators

Semantic-preservation lemmas for same-shape binary elementwise operators in the IR -> compiled
runtime bridge.

The operators in this file all share the same compiler shape:

- two parent ids,
- both parents typed at the declared output shape,
- one compiled `GraphData` node whose `forward` closure calls the corresponding tensor spec op.

Factoring these cases out keeps the recursive semantic-equivalence theorem focused on graph
traversal rather than on repeating parent-list and typed-index boilerplate for every elementwise op.

Build note: elementwise proofs spend most of their time on the shared two-parent shape discipline,
not on addition or multiplication themselves. Each branch must rule out bad parent lists, recover
typed indices for both parents, and match the compiled output against `NN.IR.Graph.evalAt`.
The shared two-parent pattern belongs in a helper lemma so these lemmas state only the
operator-specific tensor equation.
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

/-- Semantic-preservation lemma for `.add` lowering. -/
theorem buildFrom_denoteAllFrom_add
    {α : Type} [Context α] [DecidableEq Shape]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : GraphData α Unit [inShape] ss) (i : Nat) (st' : State α inShape)
    (x : Tensor α inShape) (n : NN.IR.Node)
    (hN : g.getNode i = .ok n) (hk : n.kind = .add) (hi : i < g.nodes.size)
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
  | cons aId rest =>
      cases rest with
      | cons bId rest2 =>
          cases rest2 with
          | nil =>
              cases hIa : mkIdx (inShape := inShape) (ss := ss) aId n.outShape with
              | error msg =>
                  simp [hp, hIa] at hBuild
                  try cases hBuild
              | ok ia =>
                  cases hIb : mkIdx (inShape := inShape) (ss := ss) bId n.outShape with
                  | error msg =>
                      simp [hp, hIa, hIb] at hBuild
                      try cases hBuild
                  | ok ib =>
                      simp [hp, hIa, hIb] at hBuild
                      let nodeData : NodeData α Unit ([inShape] ++ ss) n.outShape :=
                        mkFwdNode (α := α) (Γ := [inShape] ++ ss) (τ := n.outShape) (fun ctx =>
                          Tensor.addSpec (α := α)
                            (getIdx (α := α) (xs := ctx) ia)
                            (getIdx (α := α) (xs := ctx) ib))
                      let st1 : State α inShape :=
                        ⟨ss ++ [n.outShape], .snoc (ss := ss) gd nodeData⟩
                      have hRec :
                          buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
                            (i := i + 1) st1 = .ok st' := by
                        simpa [st1, nodeData] using hBuild
                      have hTail := ih st1 hRec
                      have hGetA :
                          vals0[aId]! =
                            NN.IR.DVal.mk (α := α) n.outShape
                              (getIdx (α := α) (xs := ctx) ia) := by
                        simpa [vals0, ctx] using
                          (denoteAllState_get_mkIdx (inShape := inShape) (ss := ss)
                            (gd := gd) (x := x) (pid := aId) (s := n.outShape) (idx := ia) hIa)
                      have hGetB :
                          vals0[bId]! =
                            NN.IR.DVal.mk (α := α) n.outShape
                              (getIdx (α := α) (xs := ctx) ib) := by
                        simpa [vals0, ctx] using
                          (denoteAllState_get_mkIdx (inShape := inShape) (ss := ss)
                            (gd := gd) (x := x) (pid := bId) (s := n.outShape) (idx := ib) hIb)
                      have hEval :
                          NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
                              (input := input) (vals := vals0) (i := i) =
                            .ok (NN.IR.DVal.mk (α := α) n.outShape (nodeData.forward ctx ())) := by
                        simp [NN.IR.Graph.evalAt, hN, hk, hp, hGetA, hGetB,
                          Graph.expectShape_mk, NN.IR.DVal.shape, NN.IR.DVal.tensor,
                          NN.IR.DVal.mk, Tensor.eqRec_proof_irrel, nodeData, mkFwdNode,
                          throw_eq_error, Except.instMonad, Except.bind, Except.pure]
                      exact buildFrom_denoteAllFrom_binary_exact (α := α) (g := g)
                        (payload := payload) (gd := gd) (i := i) (st' := st') (x := x) (hi := hi)
                        (τ := n.outShape) (nodeData := nodeData) hTail hEval
          | cons _ _ =>
              simp [hp, throw_eq_error, Except.instMonad, Except.bind, Except.pure] at hBuild
              try cases hBuild
      | nil =>
          simp [hp, throw_eq_error, Except.instMonad, Except.bind, Except.pure] at hBuild
          try cases hBuild
  | nil =>
      simp [hp, throw_eq_error, Except.instMonad, Except.bind, Except.pure] at hBuild
      try cases hBuild

/-- Semantic-preservation lemma for `.sub` lowering. -/
theorem buildFrom_denoteAllFrom_sub
    {α : Type} [Context α] [DecidableEq Shape]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : GraphData α Unit [inShape] ss) (i : Nat) (st' : State α inShape)
    (x : Tensor α inShape) (n : NN.IR.Node)
    (hN : g.getNode i = .ok n) (hk : n.kind = .sub) (hi : i < g.nodes.size)
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
  | cons aId rest =>
      cases rest with
      | cons bId rest2 =>
          cases rest2 with
          | nil =>
              cases hIa : mkIdx (inShape := inShape) (ss := ss) aId n.outShape with
              | error msg =>
                  simp [hp, hIa] at hBuild
                  try cases hBuild
              | ok ia =>
                  cases hIb : mkIdx (inShape := inShape) (ss := ss) bId n.outShape with
                  | error msg =>
                      simp [hp, hIa, hIb] at hBuild
                      try cases hBuild
                  | ok ib =>
                      simp [hp, hIa, hIb] at hBuild
                      let nodeData : NodeData α Unit ([inShape] ++ ss) n.outShape :=
                        mkFwdNode (α := α) (Γ := [inShape] ++ ss) (τ := n.outShape) (fun ctx =>
                          Tensor.subSpec (α := α)
                            (getIdx (α := α) (xs := ctx) ia)
                            (getIdx (α := α) (xs := ctx) ib))
                      let st1 : State α inShape :=
                        ⟨ss ++ [n.outShape], .snoc (ss := ss) gd nodeData⟩
                      have hRec :
                          buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
                            (i := i + 1) st1 = .ok st' := by
                        simpa [st1, nodeData] using hBuild
                      have hTail := ih st1 hRec
                      have hGetA :
                          vals0[aId]! =
                            NN.IR.DVal.mk (α := α) n.outShape
                              (getIdx (α := α) (xs := ctx) ia) := by
                        simpa [vals0, ctx] using
                          (denoteAllState_get_mkIdx (inShape := inShape) (ss := ss)
                            (gd := gd) (x := x) (pid := aId) (s := n.outShape) (idx := ia) hIa)
                      have hGetB :
                          vals0[bId]! =
                            NN.IR.DVal.mk (α := α) n.outShape
                              (getIdx (α := α) (xs := ctx) ib) := by
                        simpa [vals0, ctx] using
                          (denoteAllState_get_mkIdx (inShape := inShape) (ss := ss)
                            (gd := gd) (x := x) (pid := bId) (s := n.outShape) (idx := ib) hIb)
                      have hEval :
                          NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
                              (input := input) (vals := vals0) (i := i) =
                            .ok (NN.IR.DVal.mk (α := α) n.outShape (nodeData.forward ctx ())) := by
                        simp [NN.IR.Graph.evalAt, hN, hk, hp, hGetA, hGetB,
                          Graph.expectShape_mk, NN.IR.DVal.shape, NN.IR.DVal.tensor,
                          NN.IR.DVal.mk, nodeData, mkFwdNode, throw_eq_error,
                          Except.instMonad, Except.bind, Except.pure]
                      exact buildFrom_denoteAllFrom_binary_exact (α := α) (g := g)
                        (payload := payload) (gd := gd) (i := i) (st' := st') (x := x) (hi := hi)
                        (τ := n.outShape) (nodeData := nodeData) hTail hEval
          | cons _ _ =>
              simp [hp, throw_eq_error, Except.instMonad, Except.bind, Except.pure] at hBuild
              try cases hBuild
      | nil =>
          simp [hp, throw_eq_error, Except.instMonad, Except.bind, Except.pure] at hBuild
          try cases hBuild
  | nil =>
      simp [hp, throw_eq_error, Except.instMonad, Except.bind, Except.pure] at hBuild
      try cases hBuild

/-- Semantic-preservation lemma for `.mul_elem` lowering. -/
theorem buildFrom_denoteAllFrom_mul_elem
    {α : Type} [Context α] [DecidableEq Shape]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : GraphData α Unit [inShape] ss) (i : Nat) (st' : State α inShape)
    (x : Tensor α inShape) (n : NN.IR.Node)
    (hN : g.getNode i = .ok n) (hk : n.kind = .mul_elem) (hi : i < g.nodes.size)
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
  | cons aId rest =>
      cases rest with
      | cons bId rest2 =>
          cases rest2 with
          | nil =>
              cases hIa : mkIdx (inShape := inShape) (ss := ss) aId n.outShape with
              | error msg =>
                  simp [hp, hIa] at hBuild
                  try cases hBuild
              | ok ia =>
                  cases hIb : mkIdx (inShape := inShape) (ss := ss) bId n.outShape with
                  | error msg =>
                      simp [hp, hIa, hIb] at hBuild
                      try cases hBuild
                  | ok ib =>
                      simp [hp, hIa, hIb] at hBuild
                      let nodeData : NodeData α Unit ([inShape] ++ ss) n.outShape :=
                        mkFwdNode (α := α) (Γ := [inShape] ++ ss) (τ := n.outShape) (fun ctx =>
                          Tensor.mulSpec (α := α)
                            (getIdx (α := α) (xs := ctx) ia)
                            (getIdx (α := α) (xs := ctx) ib))
                      let st1 : State α inShape :=
                        ⟨ss ++ [n.outShape], .snoc (ss := ss) gd nodeData⟩
                      have hRec :
                          buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
                            (i := i + 1) st1 = .ok st' := by
                        simpa [st1, nodeData] using hBuild
                      have hTail := ih st1 hRec
                      have hGetA :
                          vals0[aId]! =
                            NN.IR.DVal.mk (α := α) n.outShape
                              (getIdx (α := α) (xs := ctx) ia) := by
                        simpa [vals0, ctx] using
                          (denoteAllState_get_mkIdx (inShape := inShape) (ss := ss)
                            (gd := gd) (x := x) (pid := aId) (s := n.outShape) (idx := ia) hIa)
                      have hGetB :
                          vals0[bId]! =
                            NN.IR.DVal.mk (α := α) n.outShape
                              (getIdx (α := α) (xs := ctx) ib) := by
                        simpa [vals0, ctx] using
                          (denoteAllState_get_mkIdx (inShape := inShape) (ss := ss)
                            (gd := gd) (x := x) (pid := bId) (s := n.outShape) (idx := ib) hIb)
                      have hEval :
                          NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
                              (input := input) (vals := vals0) (i := i) =
                            .ok (NN.IR.DVal.mk (α := α) n.outShape (nodeData.forward ctx ())) := by
                        simp [NN.IR.Graph.evalAt, hN, hk, hp, hGetA, hGetB,
                          Graph.expectShape_mk, NN.IR.DVal.shape, NN.IR.DVal.tensor,
                          NN.IR.DVal.mk, nodeData, mkFwdNode, throw_eq_error,
                          Except.instMonad, Except.bind, Except.pure]
                      exact buildFrom_denoteAllFrom_binary_exact (α := α) (g := g)
                        (payload := payload) (gd := gd) (i := i) (st' := st') (x := x) (hi := hi)
                        (τ := n.outShape) (nodeData := nodeData) hTail hEval
          | cons _ _ =>
              simp [hp] at hBuild
              try cases hBuild
      | nil =>
          simp [hp] at hBuild
          try cases hBuild
  | nil =>
      simp [hp] at hBuild
      try cases hBuild

/-- Semantic-preservation lemma for `.maxElem` lowering. -/
theorem buildFrom_denoteAllFrom_max_elem
    {α : Type} [Context α] [DecidableEq Shape]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : GraphData α Unit [inShape] ss) (i : Nat) (st' : State α inShape)
    (x : Tensor α inShape) (n : NN.IR.Node)
    (hN : g.getNode i = .ok n) (hk : n.kind = .maxElem) (hi : i < g.nodes.size)
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
  | cons aId rest =>
      cases rest with
      | cons bId rest2 =>
          cases rest2 with
          | nil =>
              cases hIa : mkIdx (inShape := inShape) (ss := ss) aId n.outShape with
              | error msg =>
                  simp [hp, hIa] at hBuild
                  try cases hBuild
              | ok ia =>
                  cases hIb : mkIdx (inShape := inShape) (ss := ss) bId n.outShape with
                  | error msg =>
                      simp [hp, hIa, hIb] at hBuild
                      try cases hBuild
                  | ok ib =>
                      simp [hp, hIa, hIb] at hBuild
                      let nodeData : NodeData α Unit ([inShape] ++ ss) n.outShape :=
                        mkFwdNode (α := α) (Γ := [inShape] ++ ss) (τ := n.outShape) (fun ctx =>
                          Tensor.maxSpec (α := α)
                            (getIdx (α := α) (xs := ctx) ia)
                            (getIdx (α := α) (xs := ctx) ib))
                      let st1 : State α inShape :=
                        ⟨ss ++ [n.outShape], .snoc (ss := ss) gd nodeData⟩
                      have hRec :
                          buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
                            (i := i + 1) st1 = .ok st' := by
                        simpa [st1, nodeData] using hBuild
                      have hTail := ih st1 hRec
                      have hGetA :
                          vals0[aId]! =
                            NN.IR.DVal.mk (α := α) n.outShape
                              (getIdx (α := α) (xs := ctx) ia) := by
                        simpa [vals0, ctx] using
                          (denoteAllState_get_mkIdx (inShape := inShape) (ss := ss)
                            (gd := gd) (x := x) (pid := aId) (s := n.outShape) (idx := ia) hIa)
                      have hGetB :
                          vals0[bId]! =
                            NN.IR.DVal.mk (α := α) n.outShape
                              (getIdx (α := α) (xs := ctx) ib) := by
                        simpa [vals0, ctx] using
                          (denoteAllState_get_mkIdx (inShape := inShape) (ss := ss)
                            (gd := gd) (x := x) (pid := bId) (s := n.outShape) (idx := ib) hIb)
                      have hEval :
                          NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
                              (input := input) (vals := vals0) (i := i) =
                            .ok (NN.IR.DVal.mk (α := α) n.outShape (nodeData.forward ctx ())) := by
                        simp [NN.IR.Graph.evalAt, hN, hk, hp, hGetA, hGetB,
                          Graph.expectShape_mk, NN.IR.DVal.shape, NN.IR.DVal.tensor,
                          NN.IR.DVal.mk, nodeData, mkFwdNode, throw_eq_error,
                          Except.instMonad, Except.bind, Except.pure]
                      exact buildFrom_denoteAllFrom_binary_exact (α := α) (g := g)
                        (payload := payload) (gd := gd) (i := i) (st' := st') (x := x) (hi := hi)
                        (τ := n.outShape) (nodeData := nodeData) hTail hEval
          | cons _ _ =>
              simp [hp] at hBuild
              try cases hBuild
      | nil =>
          simp [hp] at hBuild
          try cases hBuild
  | nil =>
      simp [hp] at hBuild
      try cases hBuild

/-- Semantic-preservation lemma for `.minElem` lowering. -/
theorem buildFrom_denoteAllFrom_min_elem
    {α : Type} [Context α] [DecidableEq Shape]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : GraphData α Unit [inShape] ss) (i : Nat) (st' : State α inShape)
    (x : Tensor α inShape) (n : NN.IR.Node)
    (hN : g.getNode i = .ok n) (hk : n.kind = .minElem) (hi : i < g.nodes.size)
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
  | cons aId rest =>
      cases rest with
      | cons bId rest2 =>
          cases rest2 with
          | nil =>
              cases hIa : mkIdx (inShape := inShape) (ss := ss) aId n.outShape with
              | error msg =>
                  simp [hp, hIa] at hBuild
                  try cases hBuild
              | ok ia =>
                  cases hIb : mkIdx (inShape := inShape) (ss := ss) bId n.outShape with
                  | error msg =>
                      simp [hp, hIa, hIb] at hBuild
                      try cases hBuild
                  | ok ib =>
                      simp [hp, hIa, hIb] at hBuild
                      let nodeData : NodeData α Unit ([inShape] ++ ss) n.outShape :=
                        mkFwdNode (α := α) (Γ := [inShape] ++ ss) (τ := n.outShape) (fun ctx =>
                          Tensor.minSpec (α := α)
                            (getIdx (α := α) (xs := ctx) ia)
                            (getIdx (α := α) (xs := ctx) ib))
                      let st1 : State α inShape :=
                        ⟨ss ++ [n.outShape], .snoc (ss := ss) gd nodeData⟩
                      have hRec :
                          buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
                            (i := i + 1) st1 = .ok st' := by
                        simpa [st1, nodeData] using hBuild
                      have hTail := ih st1 hRec
                      have hGetA :
                          vals0[aId]! =
                            NN.IR.DVal.mk (α := α) n.outShape
                              (getIdx (α := α) (xs := ctx) ia) := by
                        simpa [vals0, ctx] using
                          (denoteAllState_get_mkIdx (inShape := inShape) (ss := ss)
                            (gd := gd) (x := x) (pid := aId) (s := n.outShape) (idx := ia) hIa)
                      have hGetB :
                          vals0[bId]! =
                            NN.IR.DVal.mk (α := α) n.outShape
                              (getIdx (α := α) (xs := ctx) ib) := by
                        simpa [vals0, ctx] using
                          (denoteAllState_get_mkIdx (inShape := inShape) (ss := ss)
                            (gd := gd) (x := x) (pid := bId) (s := n.outShape) (idx := ib) hIb)
                      have hEval :
                          NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
                              (input := input) (vals := vals0) (i := i) =
                            .ok (NN.IR.DVal.mk (α := α) n.outShape (nodeData.forward ctx ())) := by
                        simp [NN.IR.Graph.evalAt, hN, hk, hp, hGetA, hGetB,
                          Graph.expectShape_mk, NN.IR.DVal.shape, NN.IR.DVal.tensor,
                          NN.IR.DVal.mk, nodeData, mkFwdNode, throw_eq_error,
                          Except.instMonad, Except.bind, Except.pure]
                      exact buildFrom_denoteAllFrom_binary_exact (α := α) (g := g)
                        (payload := payload) (gd := gd) (i := i) (st' := st') (x := x) (hi := hi)
                        (τ := n.outShape) (nodeData := nodeData) hTail hEval
          | cons _ _ =>
              simp [hp] at hBuild
              try cases hBuild
      | nil =>
          simp [hp] at hBuild
          try cases hBuild
  | nil =>
      simp [hp] at hBuild
      try cases hBuild

end Compiled
end Autograd
end Runtime
