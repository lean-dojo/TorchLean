/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.IRExec.Correctness.SemanticEquivalenceCommon

/-!
# Structural Nodes

Structural correctness facts for IR nodes that do not lower to ordinary executable operators.

The first node of the source IR graph is the distinguished input, but the recursive `buildFrom` loop
starts after that input node. If `buildFrom` ever encounters another `.input` node while lowering
the tail, successful lowering is impossible. We keep that fact as a named theorem so the
top-level semantic-equivalence proof can dispatch to it directly.

The `.detach` case checks the parent shape and applies `Tensor.detachSpec`, which preserves primal
values and removes scalar differentiation metadata. This matters when the scalar carrier is a dual
number: retaining its tangent would let later operations differentiate through the detached value.
Lowering applies detachment before transporting the result to the declared shape; IR evaluation
checks that shape first. The proof accounts for this transport while preserving the same scalar
operation on both sides.
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

/-- The recursive lowering pass cannot successfully lower an `.input` node in the graph tail. -/
theorem buildFrom_denoteAllFrom_input_impossible
    {α : Type} [TorchLean.Storage α] [Context α]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : ForwardData α [inShape] ss) (i : Nat) (st' : State α inShape)
    (x : Tensor α inShape) (n : NN.IR.Node)
    (hN : g.getNode i = .ok n) (hk : n.kind = .input) (hi : i < g.nodes.size)
    (hBuild :
      buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
        (i := i) (st := (⟨ss, gd⟩ : State α inShape)) = .ok st') :
    NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
      (input := Spec.SomeTensor.mk (α := α) inShape x)
      (i := i) (vals := denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x) =
      .ok (denoteAllState (α := α) inShape st' x) := by
  have : False := by
    unfold buildFrom at hBuild
    simp [hi, hN, hk, lowerInput, throw_eq_error] at hBuild
  cases this

/-- Semantic-preservation lemma for `.detach` lowering. -/
theorem buildFrom_denoteAllFrom_detach
    {α : Type} [TorchLean.Storage α] [Context α]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : ForwardData α [inShape] ss) (i : Nat) (st' : State α inShape)
    (x : Tensor α inShape) (n : NN.IR.Node)
    (hN : g.getNode i = .ok n) (hk : n.kind = .detach) (hi : i < g.nodes.size)
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
  simp [hi, hN] at hBuild
  simp (config := { failIfUnchanged := false }) [hk, lowerDetach] at hBuild
  cases hp : unaryParent? n.parents with
  | none =>
      simp [hp] at hBuild
      try cases hBuild
  | some pId =>
          cases hP : g.getNode pId with
          | error msg =>
              simp [hp, hP] at hBuild
              try cases hBuild
          | ok pNode =>
              simp (config := { failIfUnchanged := false }) [hp, hP] at hBuild
              cases hIdx : mkIdx (inShape := inShape) (ss := ss) pId pNode.outShape with
              | error msg =>
                  simp [hIdx] at hBuild
                  try cases hBuild
              | ok ip =>
                  simp [hIdx] at hBuild
                  by_cases hOut : pNode.outShape = n.outShape
                  · simp [hOut] at hBuild
                    let nodeData : ForwardNode α ([inShape] ++ ss) n.outShape :=
                      mkForwardNode (α := α) (Γ := [inShape] ++ ss) (τ := n.outShape) (fun ctx =>
                        hOut ▸ Tensor.detachSpec (getIdx (α := α) (xs := ctx) ip))
                    let st1 : State α inShape :=
                      ⟨ss ++ [n.outShape], .snoc (ss := ss) gd nodeData⟩
                    have hRec :
                        buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
                          (i := i + 1) st1 = .ok st' := by
                      simpa [st1, nodeData] using hBuild

                    have hGet :
                        vals0[pId]? = some (Spec.SomeTensor.mk (α := α) pNode.outShape
                            (getIdx (α := α) (xs := ctx) ip)) := by
                      simpa [vals0, ctx] using
                        (denoteAllState_get_mkIdx? (inShape := inShape) (ss := ss)
                          (gd := gd) (x := x) (pid := pId) (s := pNode.outShape) (idx := ip) hIdx)

                    have hEval :
                        NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
                            (input := input) (vals := vals0) (i := i) =
                          .ok (Spec.SomeTensor.mk (α := α) n.outShape (nodeData.eval ctx)) := by
                      -- Shape transport leaves the scalar operation unchanged. This reconciles
                      -- detachment before the cast in lowering with detachment after it in the IR.
                      have hDetachCast {source target : Shape} (h : source = target)
                          (value : Tensor α source) :
                          Tensor.detachSpec (h ▸ value) = h ▸ Tensor.detachSpec value := by
                        subst target
                        rfl
                      have hExpect :
                          NN.IR.Graph.expectShape (α := α) (expected := n.outShape)
                              (Spec.SomeTensor.mk (α := α) pNode.outShape
                                (getIdx (α := α) (xs := ctx) ip)) =
                            .ok (hOut ▸ getIdx (α := α) (xs := ctx) ip) := by
                        -- `expectShape` is a dependent `if` on shape equality. We take the
                        -- successful branch explicitly and then normalize the cast proof using
                        -- proof-irrelevance for tensor transports.
                        by_cases hEq : pNode.outShape = n.outShape
                        · have hCast :
                            (hEq ▸ getIdx (α := α) (xs := ctx) ip) =
                              (hOut ▸ getIdx (α := α) (xs := ctx) ip) := by
                            simp
                          -- Reduce `expectShape` using `hEq`, then rewrite casts using `hCast`.
                          -- We finish by normalizing `pure` to `.ok` explicitly to avoid
                          -- depending on simp's unfolding heuristics for typeclass methods.
                          have hOk :
                              (pure (hEq ▸ getIdx (α := α) (xs := ctx) ip) :
                                  Except String (Tensor α n.outShape)) =
                                .ok (hOut ▸ getIdx (α := α) (xs := ctx) ip) := by
                            -- `pure` for `Except` is definitional `.ok`, so this is just a cast
                            -- proof-irrelevance step.
                            change (.ok (hEq ▸ getIdx (α := α) (xs := ctx) ip) :
                                Except String (Tensor α n.outShape)) =
                              .ok (hOut ▸ getIdx (α := α) (xs := ctx) ip)
                            simp [hCast]
                          simp [NN.IR.Graph.expectShape, hEq, hOk]
                        · cases (hEq hOut)
                      simp [NN.IR.Graph.evalAt, NN.IR.Graph.evalNode,
                        NN.IR.Graph.normalizeNodeOutput, hN, hk, hp, hGet, hExpect,
                        nodeData, mkForwardNode, hDetachCast, throw_eq_error,
                        Pure.pure, Except.pure]

                    have hTail := ih st1 hRec
                    have hEvalForTail :
                        NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
                            (input := Spec.SomeTensor.mk (α := α) inShape x)
                            (vals := denoteAllState (α := α) inShape
                              (st := (⟨ss, gd⟩ : State α inShape)) x)
                            (i := i) =
                          .ok (Spec.SomeTensor.mk (α := α) n.outShape
                            (nodeData.eval
                              (ForwardData.eval (α := α) (Γ := [inShape])
                                (ss := ss) gd (.cons x .nil)))) := by
                      change
                        NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
                            (input := input) (vals := vals0) (i := i) =
                          .ok (Spec.SomeTensor.mk (α := α) n.outShape
                            (nodeData.eval ctx))
                      exact hEval
                    exact buildFrom_denoteAllFrom_nodeData_exact (α := α) (g := g)
                      (payload := payload)
                      (gd := gd) (i := i) (st' := st') (x := x) (hi := hi)
                      (τ := n.outShape) (nodeData := nodeData) hTail hEvalForTail
                  · simp [hOut] at hBuild
                    try cases hBuild

end IRExec
end Autograd
end Runtime
