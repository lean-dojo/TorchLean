/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.IRExec.Correctness.Common

/-!
# Constants

Constant-node correctness for the IR-to-forward-executor lowering.

The IR `.const s` node reads an external payload entry and appends that value to the execution
table. This file proves that the lowered `ForwardNode` produced by `buildFrom` appends exactly
the same tensor as the IR denotational evaluator.

Keeping this proof outside the recursive semantic-equivalence theorem matters for readability:
the top-level theorem should read as a dispatcher over named semantic cases, not as a long script
that re-proves every operator branch inline.

Build note: `.const` is semantically straightforward but proof-intensive because the value arrives
through the payload table and must be reconstructed as a shape-indexed forward node. Named payload
lookup facts keep this proof focused instead of turning it into a long `simp` script.

## Main definitions

- `buildFrom_denoteAllFrom_const`: semantic-preservation step for `.const`.
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace IRExec

open Spec TorchLean
open Proofs.Autograd.Algebra
open NN.IR
open Internal

/-- Semantic-preservation lemma for `.const s` lowering. -/
theorem buildFrom_denoteAllFrom_const
    {α : Type} [TorchLean.Storage α] [Context α]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : ForwardData α [inShape] ss) (i : Nat) (st' : State α inShape)
    (x : Tensor α inShape) (n : NN.IR.Node)
    (s : Shape)
    (hN : g.getNode i = .ok n) (hk : n.kind = .const s) (hi : i < g.nodes.size)
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
  simp (config := { failIfUnchanged := false }) [hk, lowerConst] at hBuild
  cases hT : NN.IR.Graph.evalConst (α := α) (payload := payload) (id := n.id) (s := s) with
  | error msg =>
      simp [hT] at hBuild
  | ok t =>
      simp (config := { failIfUnchanged := false }) [hT] at hBuild
      by_cases hOut : s = n.outShape
      · simp [hOut] at hBuild
        let nodeData : ForwardNode α ([inShape] ++ ss) n.outShape :=
          mkForwardNode (α := α) (Γ := [inShape] ++ ss) (τ := n.outShape) (fun _ctx => hOut ▸ t)
        let st1 : State α inShape := ⟨ss ++ [n.outShape], .snoc (ss := ss) gd nodeData⟩
        have hRec :
            buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
              (i := i + 1) st1 = .ok st' := by
          simpa [st1, nodeData] using hBuild
        have hEval :
            NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
                (input := input) (vals := vals0) (i := i) =
              .ok (Spec.SomeTensor.mk (α := α) n.outShape (nodeData.eval ctx)) := by
          cases hOut
          simp [NN.IR.Graph.evalAt, NN.IR.Graph.evalNode, NN.IR.Graph.normalizeNodeOutput, hN, hk,
            hT, input, throw, throwThe, MonadExceptOf.throw, nodeData]
          try rfl
        have hStep :
            denoteAllState (α := α) inShape st1 x =
              vals0.push (Spec.SomeTensor.mk (α := α) n.outShape (nodeData.eval ctx)) := by
          simpa [vals0, st1, ctx] using
            (denoteAllState_snoc (α := α) (inShape := inShape) (ss := ss) (τ := n.outShape)
              (gd := gd) (nodeData := nodeData) (x := x))
        have hTail := ih st1 hRec
        exact buildFrom_denoteAllFrom_finish (α := α) (g := g) (payload := payload)
          (i := i) (x := x) (hi := hi) (τ := n.outShape)
          (nodeData := nodeData) (st1 := st1) (st' := st')
          (ctx := ctx) (vals0 := vals0) (input := input) hTail hEval hStep
      · exact False.elim <| by
          exact throw_bind_ne_ok (h := by simpa [hOut] using hBuild)

end IRExec
end Autograd
end Runtime
