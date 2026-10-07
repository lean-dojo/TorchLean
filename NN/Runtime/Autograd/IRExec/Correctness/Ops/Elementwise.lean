/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.IRExec.Correctness.Common

/-!
# Elementwise Operators

Semantic preservation for binary elementwise operations. The shared
`buildFrom_denoteAllFrom_binary` lemma validates the left parent before the right and completes
the recursive graph step. Add, subtract, multiply, maximum, and minimum use the output shape
for both inputs. `safeLog` uses that shape for its value and a scalar shape for epsilon.
Each theorem below supplies the operator's tensor equation.
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace IRExec

open Spec TorchLean
open Proofs.Autograd.Algebra
open NN.IR
open Internal
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
  refine buildFrom_denoteAllFrom_binary g payload gd i st' x n hN hi
    "add" n.outShape (Tensor.addSpec) ?_ ?_ hBuild ih
  · simp [loweringContext, hk, lowerAdd]
  · intro aId bId left right hp hGetA hGetB
    have hParents := binaryParentIds_eq_ok_of_binaryParents_eq_some i aId bId n hp
    simp [NN.IR.Graph.evalAt, NN.IR.Graph.evalNode, NN.IR.Graph.normalizeNodeOutput,
      hN, hk, hParents, hGetA, hGetB, throw_eq_error, Pure.pure, Except.pure]

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
  refine buildFrom_denoteAllFrom_binary g payload gd i st' x n hN hi
    "safe_log" .scalar (fun value epsilon => Activation.safeLogSpec value epsilon.item)
    ?_ ?_ hBuild ih
  · simp [loweringContext, hk, lowerSafeLog]
  · intro aId bId left right hp hGetA hGetB
    have hParents := binaryParentIds_eq_ok_of_binaryParents_eq_some i aId bId n hp
    simp [NN.IR.Graph.evalAt, NN.IR.Graph.evalNode, NN.IR.Graph.normalizeNodeOutput,
      hN, hk, hParents, hGetA, hGetB, throw_eq_error, Pure.pure, Except.pure]

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
  refine buildFrom_denoteAllFrom_binary g payload gd i st' x n hN hi
    "sub" n.outShape (Tensor.subSpec) ?_ ?_ hBuild ih
  · simp [loweringContext, hk, lowerSub]
  · intro aId bId left right hp hGetA hGetB
    have hParents := binaryParentIds_eq_ok_of_binaryParents_eq_some i aId bId n hp
    simp [NN.IR.Graph.evalAt, NN.IR.Graph.evalNode, NN.IR.Graph.normalizeNodeOutput,
      hN, hk, hParents, hGetA, hGetB, throw_eq_error, Pure.pure, Except.pure]

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
  refine buildFrom_denoteAllFrom_binary g payload gd i st' x n hN hi
    "mul_elem" n.outShape (Tensor.mulSpec) ?_ ?_ hBuild ih
  · simp [loweringContext, hk, lowerMulElem]
  · intro aId bId left right hp hGetA hGetB
    have hParents := binaryParentIds_eq_ok_of_binaryParents_eq_some i aId bId n hp
    simp [NN.IR.Graph.evalAt, NN.IR.Graph.evalNode, NN.IR.Graph.normalizeNodeOutput,
      hN, hk, hParents, hGetA, hGetB, throw_eq_error, Pure.pure, Except.pure]

/-- Semantic-preservation lemma for `.maxElem` lowering. -/
theorem buildFrom_denoteAllFrom_maxElem
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
  refine buildFrom_denoteAllFrom_binary g payload gd i st' x n hN hi
    "max_elem" n.outShape (Tensor.maxSpec) ?_ ?_ hBuild ih
  · simp [loweringContext, hk, lowerMaxElem]
  · intro aId bId left right hp hGetA hGetB
    have hParents := binaryParentIds_eq_ok_of_binaryParents_eq_some i aId bId n hp
    simp [NN.IR.Graph.evalAt, NN.IR.Graph.evalNode, NN.IR.Graph.normalizeNodeOutput,
      hN, hk, hParents, hGetA, hGetB, throw_eq_error, Pure.pure, Except.pure]

/-- Semantic-preservation lemma for `.minElem` lowering. -/
theorem buildFrom_denoteAllFrom_minElem
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
  refine buildFrom_denoteAllFrom_binary g payload gd i st' x n hN hi
    "min_elem" n.outShape (Tensor.minSpec) ?_ ?_ hBuild ih
  · simp [loweringContext, hk, lowerMinElem]
  · intro aId bId left right hp hGetA hGetB
    have hParents := binaryParentIds_eq_ok_of_binaryParents_eq_some i aId bId n hp
    simp [NN.IR.Graph.evalAt, NN.IR.Graph.evalNode, NN.IR.Graph.normalizeNodeOutput,
      hN, hk, hParents, hGetA, hGetB, throw_eq_error, Pure.pure, Except.pure]

end IRExec
end Autograd
end Runtime
