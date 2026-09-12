/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.IRExec.Correctness.SemanticEquivalenceCommon

/-!
# Elementwise Operators

Semantic-preservation lemmas for same-shape binary elementwise operators in the IR -> lowered
runtime bridge.

The operators in this file all share the same lowering pass shape:

- two parent ids,
- both parents typed at the declared output shape,
- one lowered `ForwardNode` whose `eval` closure calls the corresponding tensor specification.

Factoring these cases out keeps the recursive semantic-equivalence theorem focused on graph
traversal rather than on repeating parent-list and typed-index boilerplate for every elementwise op.

Build note: elementwise proofs spend most of their time on the shared two-parent shape discipline,
not on addition or multiplication themselves. Each branch must rule out bad parent lists, recover
typed indices for both parents, and match the forward-graph output against `NN.IR.Graph.evalAt`.
The shared two-parent pattern belongs in a helper lemma so these lemmas state only the
operator-specific tensor equation.
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace IRExec

open Spec TorchLean
open Proofs.Autograd.Algebra
open NN.IR
open Internal
-- Typed context indices come from `NN.Proofs.Autograd.Tape.Util.Idx`, the one place
-- `Idx` and `getIdx` are defined.
open Proofs (Idx getIdx)

/-- Semantic-preservation lemma for `.add` lowering. -/
theorem buildFrom_denoteAllFrom_add
    {α : Type} [TorchLean.Storage α] [Context α]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : ForwardData α [inShape] ss) (i : Nat) (st' : State α inShape)
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
          (input := Spec.SomeTensor.mk (α := α) inShape x)
          (i := i + 1) (vals := denoteAllState (α := α) inShape st1 x) =
          .ok (denoteAllState (α := α) inShape st' x)) :
    NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
      (input := Spec.SomeTensor.mk (α := α) inShape x)
      (i := i) (vals := denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x) =
      .ok (denoteAllState (α := α) inShape st' x) := by
  let vals0 : Array (Spec.SomeTensor α) :=
    denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x
  let ctx : TorchLean.TensorPack α ([inShape] ++ ss) :=
    ForwardData.eval (α := α) (Γ := [inShape]) (ss := ss) gd (.cons x .nil)
  let input : Spec.SomeTensor α := Spec.SomeTensor.mk (α := α) inShape x

  unfold buildFrom at hBuild
  simp (config := { failIfUnchanged := false }) [hi, hN, hk, lowerAdd] at hBuild
  cases hp : binaryParents? n.parents with
  | some parentIds =>
              rcases parentIds with ⟨aId, bId⟩
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
                      let nodeData : ForwardNode α ([inShape] ++ ss) n.outShape :=
                        mkForwardNode (α := α) (Γ := [inShape] ++ ss) (τ := n.outShape) (fun ctx =>
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
                          vals0[aId]? = some (Spec.SomeTensor.mk (α := α) n.outShape
                              (getIdx (α := α) (xs := ctx) ia)) := by
                        simpa [vals0, ctx] using
                          (denoteAllState_get_mkIdx? (inShape := inShape) (ss := ss)
                            (gd := gd) (x := x) (pid := aId) (s := n.outShape) (idx := ia) hIa)
                      have hGetB :
                          vals0[bId]? = some (Spec.SomeTensor.mk (α := α) n.outShape
                              (getIdx (α := α) (xs := ctx) ib)) := by
                        simpa [vals0, ctx] using
                          (denoteAllState_get_mkIdx? (inShape := inShape) (ss := ss)
                            (gd := gd) (x := x) (pid := bId) (s := n.outShape) (idx := ib) hIb)
                      have hParentIds :
                          NN.IR.Graph.binaryParentIds i n = .ok (aId, bId) :=
                        binaryParentIds_eq_ok_of_binaryParents_eq_some i aId bId n hp
                      have hEval :
                          NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
                              (input := input) (vals := vals0) (i := i) =
                            .ok (Spec.SomeTensor.mk (α := α) n.outShape (nodeData.eval ctx)) := by
                        simp [NN.IR.Graph.evalAt, NN.IR.Graph.evalNode,
                          NN.IR.Graph.normalizeNodeOutput, hN, hk, hParentIds, hGetA, hGetB,
                          nodeData, mkForwardNode, throw_eq_error,
                          Pure.pure, Except.pure]
                      exact buildFrom_denoteAllFrom_nodeData_exact (α := α) (g := g)
                        (payload := payload) (gd := gd) (i := i) (st' := st') (x := x) (hi := hi)
                        (τ := n.outShape) (nodeData := nodeData) hTail hEval
  | none =>
      simp [hp, throw_eq_error] at hBuild
      try cases hBuild

/-- Semantic-preservation lemma for `.safeLog` lowering. -/
theorem buildFrom_denoteAllFrom_safeLog
    {α : Type} [TorchLean.Storage α] [Context α]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : ForwardData α [inShape] ss) (i : Nat) (st' : State α inShape)
    (x : Tensor α inShape) (n : NN.IR.Node)
    (hN : g.getNode i = .ok n) (hk : n.kind = .safeLog) (hi : i < g.nodes.size)
    (hBuild :
      buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
        (i := i) (st := (⟨ss, gd⟩ : State α inShape)) = .ok st')
    (ih :
      ∀ (st1 : State α inShape),
        buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
          (i := i + 1) st1 = .ok st' →
        NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
          (input := Spec.SomeTensor.mk (α := α) inShape x)
          (i := i + 1) (vals := denoteAllState (α := α) inShape st1 x) =
          .ok (denoteAllState (α := α) inShape st' x)) :
    NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
      (input := Spec.SomeTensor.mk (α := α) inShape x)
      (i := i) (vals := denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x) =
      .ok (denoteAllState (α := α) inShape st' x) := by
  let vals0 : Array (Spec.SomeTensor α) :=
    denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x
  let ctx : TorchLean.TensorPack α ([inShape] ++ ss) :=
    ForwardData.eval (α := α) (Γ := [inShape]) (ss := ss) gd (.cons x .nil)
  let input : Spec.SomeTensor α := Spec.SomeTensor.mk (α := α) inShape x

  unfold buildFrom at hBuild
  simp (config := { failIfUnchanged := false }) [hi, hN, hk, lowerSafeLog] at hBuild
  cases hp : binaryParents? n.parents with
  | some parentIds =>
              rcases parentIds with ⟨aId, bId⟩
              cases hIa : mkIdx (inShape := inShape) (ss := ss) aId n.outShape with
              | error msg =>
                  simp [hp, hIa] at hBuild
                  try cases hBuild
              | ok ia =>
                  cases hIb : mkIdx (inShape := inShape) (ss := ss) bId .scalar with
                  | error msg =>
                      simp [hp, hIa, hIb] at hBuild
                      try cases hBuild
                  | ok ib =>
                      simp [hp, hIa, hIb] at hBuild
                      let nodeData : ForwardNode α ([inShape] ++ ss) n.outShape :=
                        mkForwardNode (α := α) (Γ := [inShape] ++ ss) (τ := n.outShape) (fun ctx =>
                          Activation.safeLogSpec (α := α)
                            (getIdx (α := α) (xs := ctx) ia)
                            (getIdx (α := α) (xs := ctx) ib).item)
                      let st1 : State α inShape :=
                        ⟨ss ++ [n.outShape], .snoc (ss := ss) gd nodeData⟩
                      have hRec :
                          buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
                            (i := i + 1) st1 = .ok st' := by
                        simpa [st1, nodeData] using hBuild
                      have hTail := ih st1 hRec
                      have hGetA :
                          vals0[aId]? = some (Spec.SomeTensor.mk (α := α) n.outShape
                              (getIdx (α := α) (xs := ctx) ia)) := by
                        simpa [vals0, ctx] using
                          (denoteAllState_get_mkIdx? (inShape := inShape) (ss := ss)
                            (gd := gd) (x := x) (pid := aId) (s := n.outShape) (idx := ia) hIa)
                      have hGetB :
                          vals0[bId]? = some (Spec.SomeTensor.mk (α := α) .scalar
                              (getIdx (α := α) (xs := ctx) ib)) := by
                        simpa [vals0, ctx] using
                          (denoteAllState_get_mkIdx? (inShape := inShape) (ss := ss)
                            (gd := gd) (x := x) (pid := bId) (s := .scalar) (idx := ib) hIb)
                      have hParentIds :
                          NN.IR.Graph.binaryParentIds i n = .ok (aId, bId) :=
                        binaryParentIds_eq_ok_of_binaryParents_eq_some i aId bId n hp
                      have hEval :
                          NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
                              (input := input) (vals := vals0) (i := i) =
                            .ok (Spec.SomeTensor.mk (α := α) n.outShape (nodeData.eval ctx)) := by
                        simp [NN.IR.Graph.evalAt, NN.IR.Graph.evalNode,
                          NN.IR.Graph.normalizeNodeOutput, hN, hk, hParentIds, hGetA, hGetB,
                          nodeData, mkForwardNode, throw_eq_error,
                          Pure.pure, Except.pure]
                      exact buildFrom_denoteAllFrom_nodeData_exact (α := α) (g := g)
                        (payload := payload) (gd := gd) (i := i) (st' := st') (x := x) (hi := hi)
                        (τ := n.outShape) (nodeData := nodeData) hTail hEval
  | none =>
      simp [hp, throw_eq_error] at hBuild
      try cases hBuild

/-- Semantic-preservation lemma for `.sub` lowering. -/
theorem buildFrom_denoteAllFrom_sub
    {α : Type} [TorchLean.Storage α] [Context α]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : ForwardData α [inShape] ss) (i : Nat) (st' : State α inShape)
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
          (input := Spec.SomeTensor.mk (α := α) inShape x)
          (i := i + 1) (vals := denoteAllState (α := α) inShape st1 x) =
          .ok (denoteAllState (α := α) inShape st' x)) :
    NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
      (input := Spec.SomeTensor.mk (α := α) inShape x)
      (i := i) (vals := denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x) =
      .ok (denoteAllState (α := α) inShape st' x) := by
  let vals0 : Array (Spec.SomeTensor α) :=
    denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x
  let ctx : TorchLean.TensorPack α ([inShape] ++ ss) :=
    ForwardData.eval (α := α) (Γ := [inShape]) (ss := ss) gd (.cons x .nil)
  let input : Spec.SomeTensor α := Spec.SomeTensor.mk (α := α) inShape x

  unfold buildFrom at hBuild
  simp (config := { failIfUnchanged := false }) [hi, hN, hk, lowerSub] at hBuild
  cases hp : binaryParents? n.parents with
  | some parentIds =>
              rcases parentIds with ⟨aId, bId⟩
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
                      let nodeData : ForwardNode α ([inShape] ++ ss) n.outShape :=
                        mkForwardNode (α := α) (Γ := [inShape] ++ ss) (τ := n.outShape) (fun ctx =>
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
                          vals0[aId]? = some (Spec.SomeTensor.mk (α := α) n.outShape
                              (getIdx (α := α) (xs := ctx) ia)) := by
                        simpa [vals0, ctx] using
                          (denoteAllState_get_mkIdx? (inShape := inShape) (ss := ss)
                            (gd := gd) (x := x) (pid := aId) (s := n.outShape) (idx := ia) hIa)
                      have hGetB :
                          vals0[bId]? = some (Spec.SomeTensor.mk (α := α) n.outShape
                              (getIdx (α := α) (xs := ctx) ib)) := by
                        simpa [vals0, ctx] using
                          (denoteAllState_get_mkIdx? (inShape := inShape) (ss := ss)
                            (gd := gd) (x := x) (pid := bId) (s := n.outShape) (idx := ib) hIb)
                      have hParentIds :
                          NN.IR.Graph.binaryParentIds i n = .ok (aId, bId) :=
                        binaryParentIds_eq_ok_of_binaryParents_eq_some i aId bId n hp
                      have hEval :
                          NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
                              (input := input) (vals := vals0) (i := i) =
                            .ok (Spec.SomeTensor.mk (α := α) n.outShape (nodeData.eval ctx)) := by
                        simp [NN.IR.Graph.evalAt, NN.IR.Graph.evalNode,
                          NN.IR.Graph.normalizeNodeOutput, hN, hk, hParentIds, hGetA, hGetB,
                          nodeData, mkForwardNode, throw_eq_error,
                          Pure.pure, Except.pure]
                      exact buildFrom_denoteAllFrom_nodeData_exact (α := α) (g := g)
                        (payload := payload) (gd := gd) (i := i) (st' := st') (x := x) (hi := hi)
                        (τ := n.outShape) (nodeData := nodeData) hTail hEval
  | none =>
      simp [hp, throw_eq_error] at hBuild
      try cases hBuild

/-- Semantic-preservation lemma for `.mulElem` lowering. -/
theorem buildFrom_denoteAllFrom_mulElem
    {α : Type} [TorchLean.Storage α] [Context α]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : ForwardData α [inShape] ss) (i : Nat) (st' : State α inShape)
    (x : Tensor α inShape) (n : NN.IR.Node)
    (hN : g.getNode i = .ok n) (hk : n.kind = .mulElem) (hi : i < g.nodes.size)
    (hBuild :
      buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
        (i := i) (st := (⟨ss, gd⟩ : State α inShape)) = .ok st')
    (ih :
      ∀ (st1 : State α inShape),
        buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
          (i := i + 1) st1 = .ok st' →
        NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
          (input := Spec.SomeTensor.mk (α := α) inShape x)
          (i := i + 1) (vals := denoteAllState (α := α) inShape st1 x) =
          .ok (denoteAllState (α := α) inShape st' x)) :
    NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
      (input := Spec.SomeTensor.mk (α := α) inShape x)
      (i := i) (vals := denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x) =
      .ok (denoteAllState (α := α) inShape st' x) := by
  let vals0 : Array (Spec.SomeTensor α) :=
    denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x
  let ctx : TorchLean.TensorPack α ([inShape] ++ ss) :=
    ForwardData.eval (α := α) (Γ := [inShape]) (ss := ss) gd (.cons x .nil)
  let input : Spec.SomeTensor α := Spec.SomeTensor.mk (α := α) inShape x

  unfold buildFrom at hBuild
  simp (config := { failIfUnchanged := false }) [hi, hN, hk, lowerMulElem] at hBuild
  cases hp : binaryParents? n.parents with
  | some parentIds =>
              rcases parentIds with ⟨aId, bId⟩
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
                      let nodeData : ForwardNode α ([inShape] ++ ss) n.outShape :=
                        mkForwardNode (α := α) (Γ := [inShape] ++ ss) (τ := n.outShape) (fun ctx =>
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
                          vals0[aId]? = some (Spec.SomeTensor.mk (α := α) n.outShape
                              (getIdx (α := α) (xs := ctx) ia)) := by
                        simpa [vals0, ctx] using
                          (denoteAllState_get_mkIdx? (inShape := inShape) (ss := ss)
                            (gd := gd) (x := x) (pid := aId) (s := n.outShape) (idx := ia) hIa)
                      have hGetB :
                          vals0[bId]? = some (Spec.SomeTensor.mk (α := α) n.outShape
                              (getIdx (α := α) (xs := ctx) ib)) := by
                        simpa [vals0, ctx] using
                          (denoteAllState_get_mkIdx? (inShape := inShape) (ss := ss)
                            (gd := gd) (x := x) (pid := bId) (s := n.outShape) (idx := ib) hIb)
                      have hParentIds :
                          NN.IR.Graph.binaryParentIds i n = .ok (aId, bId) :=
                        binaryParentIds_eq_ok_of_binaryParents_eq_some i aId bId n hp
                      have hEval :
                          NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
                              (input := input) (vals := vals0) (i := i) =
                            .ok (Spec.SomeTensor.mk (α := α) n.outShape (nodeData.eval ctx)) := by
                        simp [NN.IR.Graph.evalAt, NN.IR.Graph.evalNode,
                          NN.IR.Graph.normalizeNodeOutput, hN, hk, hParentIds, hGetA, hGetB,
                          nodeData, mkForwardNode, throw_eq_error,
                          Pure.pure, Except.pure]
                      exact buildFrom_denoteAllFrom_nodeData_exact (α := α) (g := g)
                        (payload := payload) (gd := gd) (i := i) (st' := st') (x := x) (hi := hi)
                        (τ := n.outShape) (nodeData := nodeData) hTail hEval
  | none =>
      simp [hp] at hBuild
      try cases hBuild

/-- Semantic-preservation lemma for `.maxElem` lowering. -/
theorem buildFrom_denoteAllFrom_max_elem
    {α : Type} [TorchLean.Storage α] [Context α]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : ForwardData α [inShape] ss) (i : Nat) (st' : State α inShape)
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
          (input := Spec.SomeTensor.mk (α := α) inShape x)
          (i := i + 1) (vals := denoteAllState (α := α) inShape st1 x) =
          .ok (denoteAllState (α := α) inShape st' x)) :
    NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
      (input := Spec.SomeTensor.mk (α := α) inShape x)
      (i := i) (vals := denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x) =
      .ok (denoteAllState (α := α) inShape st' x) := by
  let vals0 : Array (Spec.SomeTensor α) :=
    denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x
  let ctx : TorchLean.TensorPack α ([inShape] ++ ss) :=
    ForwardData.eval (α := α) (Γ := [inShape]) (ss := ss) gd (.cons x .nil)
  let input : Spec.SomeTensor α := Spec.SomeTensor.mk (α := α) inShape x

  unfold buildFrom at hBuild
  simp (config := { failIfUnchanged := false }) [hi, hN, hk, lowerMaxElem] at hBuild
  cases hp : binaryParents? n.parents with
  | some parentIds =>
              rcases parentIds with ⟨aId, bId⟩
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
                      let nodeData : ForwardNode α ([inShape] ++ ss) n.outShape :=
                        mkForwardNode (α := α) (Γ := [inShape] ++ ss) (τ := n.outShape) (fun ctx =>
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
                          vals0[aId]? = some (Spec.SomeTensor.mk (α := α) n.outShape
                              (getIdx (α := α) (xs := ctx) ia)) := by
                        simpa [vals0, ctx] using
                          (denoteAllState_get_mkIdx? (inShape := inShape) (ss := ss)
                            (gd := gd) (x := x) (pid := aId) (s := n.outShape) (idx := ia) hIa)
                      have hGetB :
                          vals0[bId]? = some (Spec.SomeTensor.mk (α := α) n.outShape
                              (getIdx (α := α) (xs := ctx) ib)) := by
                        simpa [vals0, ctx] using
                          (denoteAllState_get_mkIdx? (inShape := inShape) (ss := ss)
                            (gd := gd) (x := x) (pid := bId) (s := n.outShape) (idx := ib) hIb)
                      have hParentIds :
                          NN.IR.Graph.binaryParentIds i n = .ok (aId, bId) :=
                        binaryParentIds_eq_ok_of_binaryParents_eq_some i aId bId n hp
                      have hEval :
                          NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
                              (input := input) (vals := vals0) (i := i) =
                            .ok (Spec.SomeTensor.mk (α := α) n.outShape (nodeData.eval ctx)) := by
                        simp [NN.IR.Graph.evalAt, NN.IR.Graph.evalNode,
                          NN.IR.Graph.normalizeNodeOutput, hN, hk, hParentIds, hGetA, hGetB,
                          nodeData, mkForwardNode, throw_eq_error,
                          Pure.pure, Except.pure]
                      exact buildFrom_denoteAllFrom_nodeData_exact (α := α) (g := g)
                        (payload := payload) (gd := gd) (i := i) (st' := st') (x := x) (hi := hi)
                        (τ := n.outShape) (nodeData := nodeData) hTail hEval
  | none =>
      simp [hp] at hBuild
      try cases hBuild

/-- Semantic-preservation lemma for `.minElem` lowering. -/
theorem buildFrom_denoteAllFrom_min_elem
    {α : Type} [TorchLean.Storage α] [Context α]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : ForwardData α [inShape] ss) (i : Nat) (st' : State α inShape)
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
          (input := Spec.SomeTensor.mk (α := α) inShape x)
          (i := i + 1) (vals := denoteAllState (α := α) inShape st1 x) =
          .ok (denoteAllState (α := α) inShape st' x)) :
    NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
      (input := Spec.SomeTensor.mk (α := α) inShape x)
      (i := i) (vals := denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x) =
      .ok (denoteAllState (α := α) inShape st' x) := by
  let vals0 : Array (Spec.SomeTensor α) :=
    denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x
  let ctx : TorchLean.TensorPack α ([inShape] ++ ss) :=
    ForwardData.eval (α := α) (Γ := [inShape]) (ss := ss) gd (.cons x .nil)
  let input : Spec.SomeTensor α := Spec.SomeTensor.mk (α := α) inShape x

  unfold buildFrom at hBuild
  simp (config := { failIfUnchanged := false }) [hi, hN, hk, lowerMinElem] at hBuild
  cases hp : binaryParents? n.parents with
  | some parentIds =>
              rcases parentIds with ⟨aId, bId⟩
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
                      let nodeData : ForwardNode α ([inShape] ++ ss) n.outShape :=
                        mkForwardNode (α := α) (Γ := [inShape] ++ ss) (τ := n.outShape) (fun ctx =>
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
                          vals0[aId]? = some (Spec.SomeTensor.mk (α := α) n.outShape
                              (getIdx (α := α) (xs := ctx) ia)) := by
                        simpa [vals0, ctx] using
                          (denoteAllState_get_mkIdx? (inShape := inShape) (ss := ss)
                            (gd := gd) (x := x) (pid := aId) (s := n.outShape) (idx := ia) hIa)
                      have hGetB :
                          vals0[bId]? = some (Spec.SomeTensor.mk (α := α) n.outShape
                              (getIdx (α := α) (xs := ctx) ib)) := by
                        simpa [vals0, ctx] using
                          (denoteAllState_get_mkIdx? (inShape := inShape) (ss := ss)
                            (gd := gd) (x := x) (pid := bId) (s := n.outShape) (idx := ib) hIb)
                      have hParentIds :
                          NN.IR.Graph.binaryParentIds i n = .ok (aId, bId) :=
                        binaryParentIds_eq_ok_of_binaryParents_eq_some i aId bId n hp
                      have hEval :
                          NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
                              (input := input) (vals := vals0) (i := i) =
                            .ok (Spec.SomeTensor.mk (α := α) n.outShape (nodeData.eval ctx)) := by
                        simp [NN.IR.Graph.evalAt, NN.IR.Graph.evalNode,
                          NN.IR.Graph.normalizeNodeOutput, hN, hk, hParentIds, hGetA, hGetB,
                          nodeData, mkForwardNode, throw_eq_error,
                          Pure.pure, Except.pure]
                      exact buildFrom_denoteAllFrom_nodeData_exact (α := α) (g := g)
                        (payload := payload) (gd := gd) (i := i) (st' := st') (x := x) (hi := hi)
                        (τ := n.outShape) (nodeData := nodeData) hTail hEval
  | none =>
      simp [hp] at hBuild
      try cases hBuild

end IRExec
end Autograd
end Runtime
