/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.IRExec.Correctness.SemanticEquivalenceCommon

/-!
# Unary Elementwise Operators

Semantic-preservation lemmas for single-parent, same-shape elementwise operators in the
IR-to-forward-graph runtime bridge.

The proof pattern is deliberately explicit: we check the one-parent contract, recover the typed
parent index, build the forward-graph closure, prove that IR evaluation produces the same value,
and then hand the tail of the graph to the shared semantic-equivalence finishing lemma. Keeping
these branches named avoids a single monolithic recursive proof and gives reviewers stable theorem
names for each primitive operator.

Build note: the unary branches are proof-heavy for the same reason as activations: Lean checks the
runtime shape cast, parent lookup, `Except` failure branches, and final `Spec.SomeTensor` equality
all in one goal. The common unary skeleton belongs in `SemanticEquivalenceCommon`, keeping this
file as a short list of operator instances.
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

/-- Semantic-preservation lemma for `.abs` lowering. -/
theorem buildFrom_denoteAllFrom_abs
    {α : Type} [TorchLean.Storage α] [Context α]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : ForwardData α [inShape] ss) (i : Nat) (st' : State α inShape)
    (x : Tensor α inShape) (n : NN.IR.Node)
    (hN : g.getNode i = .ok n) (hk : n.kind = .abs) (hi : i < g.nodes.size)
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
  simp (config := { failIfUnchanged := false }) [hi, hN, hk, lowerAbs] at hBuild
  cases hp : unaryParent? n.parents with
  | none =>
      simp [hp] at hBuild
      try cases hBuild
  | some pId =>
          cases hIdx : mkIdx (inShape := inShape) (ss := ss) pId n.outShape with
          | error msg =>
              simp [hp, hIdx] at hBuild
              try cases hBuild
          | ok ip =>
              simp [hp, hIdx] at hBuild
              let nodeData : ForwardNode α ([inShape] ++ ss) n.outShape :=
                mkForwardNode (α := α) (Γ := [inShape] ++ ss) (τ := n.outShape) (fun ctx =>
                  Tensor.absSpec (α := α) (getIdx (α := α) (xs := ctx) ip))
              let st1 : State α inShape :=
                ⟨ss ++ [n.outShape], .snoc (ss := ss) gd nodeData⟩
              have hRec :
                  buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
                    (i := i + 1) st1 = .ok st' := by
                simpa [st1, nodeData] using hBuild
              have hTail := ih st1 hRec
              have hGet :
                  vals0[pId]? =
                    some (Spec.SomeTensor.mk (α := α) n.outShape
                      (getIdx (α := α) (xs := ctx) ip)) := by
                simpa [vals0, ctx] using
                  (denoteAllState_get_mkIdx? (inShape := inShape) (ss := ss)
                    (gd := gd) (x := x) (pid := pId) (s := n.outShape) (idx := ip) hIdx)
              have hEval :
                  NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
                      (input := input) (vals := vals0) (i := i) =
                    .ok (Spec.SomeTensor.mk (α := α) n.outShape (nodeData.eval ctx)) := by
                simp [NN.IR.Graph.evalAt, NN.IR.Graph.evalNode, NN.IR.Graph.normalizeNodeOutput,
                  hN, hk, hp, hGet, nodeData, mkForwardNode, throw_eq_error, Pure.pure, Except.pure]
              exact buildFrom_denoteAllFrom_nodeData_exact (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (hi := hi)
                (τ := n.outShape) (nodeData := nodeData) hTail hEval

/-- Semantic-preservation lemma for `.sqrt` lowering. -/
theorem buildFrom_denoteAllFrom_sqrt
    {α : Type} [TorchLean.Storage α] [Context α]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : ForwardData α [inShape] ss) (i : Nat) (st' : State α inShape)
    (x : Tensor α inShape) (n : NN.IR.Node)
    (hN : g.getNode i = .ok n) (hk : n.kind = .sqrt) (hi : i < g.nodes.size)
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
  simp (config := { failIfUnchanged := false }) [hi, hN, hk, lowerSqrt] at hBuild
  cases hp : unaryParent? n.parents with
  | none =>
      simp [hp] at hBuild
      try cases hBuild
  | some pId =>
          cases hIdx : mkIdx (inShape := inShape) (ss := ss) pId n.outShape with
          | error msg =>
              simp [hp, hIdx] at hBuild
              try cases hBuild
          | ok ip =>
              simp [hp, hIdx] at hBuild
              let nodeData : ForwardNode α ([inShape] ++ ss) n.outShape :=
                mkForwardNode (α := α) (Γ := [inShape] ++ ss) (τ := n.outShape) (fun ctx =>
                  Tensor.sqrtSpec (α := α) (getIdx (α := α) (xs := ctx) ip))
              let st1 : State α inShape :=
                ⟨ss ++ [n.outShape], .snoc (ss := ss) gd nodeData⟩
              have hRec :
                  buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
                    (i := i + 1) st1 = .ok st' := by
                simpa [st1, nodeData] using hBuild
              have hTail := ih st1 hRec
              have hGet :
                  vals0[pId]? =
                    some (Spec.SomeTensor.mk (α := α) n.outShape
                      (getIdx (α := α) (xs := ctx) ip)) := by
                simpa [vals0, ctx] using
                  (denoteAllState_get_mkIdx? (inShape := inShape) (ss := ss)
                    (gd := gd) (x := x) (pid := pId) (s := n.outShape) (idx := ip) hIdx)
              have hEval :
                  NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
                      (input := input) (vals := vals0) (i := i) =
                    .ok (Spec.SomeTensor.mk (α := α) n.outShape (nodeData.eval ctx)) := by
                simp [NN.IR.Graph.evalAt, NN.IR.Graph.evalNode, NN.IR.Graph.normalizeNodeOutput,
                  hN, hk, hp, hGet, nodeData, mkForwardNode, throw_eq_error, Pure.pure, Except.pure]
              exact buildFrom_denoteAllFrom_nodeData_exact (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (hi := hi)
                (τ := n.outShape) (nodeData := nodeData) hTail hEval

/-- Semantic-preservation lemma for `.inv` lowering. -/
theorem buildFrom_denoteAllFrom_inv
    {α : Type} [TorchLean.Storage α] [Context α]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : ForwardData α [inShape] ss) (i : Nat) (st' : State α inShape)
    (x : Tensor α inShape) (n : NN.IR.Node)
    (hN : g.getNode i = .ok n) (hk : n.kind = .inv) (hi : i < g.nodes.size)
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
  simp (config := { failIfUnchanged := false }) [hi, hN, hk, lowerInv] at hBuild
  cases hp : unaryParent? n.parents with
  | none =>
      simp [hp] at hBuild
      try cases hBuild
  | some pId =>
          cases hIdx : mkIdx (inShape := inShape) (ss := ss) pId n.outShape with
          | error msg =>
              simp [hp, hIdx] at hBuild
              try cases hBuild
          | ok ip =>
              simp [hp, hIdx] at hBuild
              let nodeData : ForwardNode α ([inShape] ++ ss) n.outShape :=
                mkForwardNode (α := α) (Γ := [inShape] ++ ss) (τ := n.outShape) (fun ctx =>
                  Tensor.invSpec (α := α) (getIdx (α := α) (xs := ctx) ip))
              let st1 : State α inShape :=
                ⟨ss ++ [n.outShape], .snoc (ss := ss) gd nodeData⟩
              have hRec :
                  buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
                    (i := i + 1) st1 = .ok st' := by
                simpa [st1, nodeData] using hBuild
              have hTail := ih st1 hRec
              have hGet :
                  vals0[pId]? =
                    some (Spec.SomeTensor.mk (α := α) n.outShape
                      (getIdx (α := α) (xs := ctx) ip)) := by
                simpa [vals0, ctx] using
                  (denoteAllState_get_mkIdx? (inShape := inShape) (ss := ss)
                    (gd := gd) (x := x) (pid := pId) (s := n.outShape) (idx := ip) hIdx)
              have hEval :
                  NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
                      (input := input) (vals := vals0) (i := i) =
                    .ok (Spec.SomeTensor.mk (α := α) n.outShape (nodeData.eval ctx)) := by
                simp [NN.IR.Graph.evalAt, NN.IR.Graph.evalNode, NN.IR.Graph.normalizeNodeOutput,
                  hN, hk, hp, hGet, nodeData, mkForwardNode, throw_eq_error, Pure.pure, Except.pure]
              exact buildFrom_denoteAllFrom_nodeData_exact (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (hi := hi)
                (τ := n.outShape) (nodeData := nodeData) hTail hEval

end IRExec
end Autograd
end Runtime
