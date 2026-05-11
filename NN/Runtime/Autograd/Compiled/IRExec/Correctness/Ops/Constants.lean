/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Compiled.IRExec.Correctness.Common

set_option linter.unusedSimpArgs false
set_option linter.unnecessarySimpa false

/-!
# Constants

Constant-node correctness for the IR -> compiled runtime bridge.

The IR `.const s` node reads an external payload entry and appends that value to the execution
table. This file proves that the compiled `GraphData` node produced by `buildFrom` appends exactly
the same tensor as the IR denotational evaluator.

Keeping this proof outside the recursive semantic-equivalence theorem matters for readability:
the top-level theorem should read as a dispatcher over named semantic cases, not as a long script
that re-proves every operator branch inline.

Build note: `.const` is semantically straightforward but proof-intensive because the value arrives
through the payload table and must be re-packed as a typed compiled node. Named payload lookup
facts keep this proof focused instead of turning it into a long `simp` script.

## Main definitions

- `buildFrom_denoteAllFrom_const`: semantic-preservation step for `.const`.
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

/-- Semantic-preservation lemma for `.const s` lowering. -/
theorem buildFrom_denoteAllFrom_const
    {α : Type} [Context α] [DecidableEq Shape]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : GraphData α Unit [inShape] ss) (i : Nat) (st' : State α inShape)
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
  simp [hi, hN] at hBuild
  simp (config := { failIfUnchanged := false }) [hk] at hBuild
  cases hT : NN.IR.Graph.evalConst (α := α) (payload := payload) (id := n.id) (s := s) with
  | error msg =>
      exact False.elim <| by
        simpa [hT] using hBuild
  | ok t =>
      simp (config := { failIfUnchanged := false }) [hT] at hBuild
      by_cases hOut : s = n.outShape
      · simp [hOut] at hBuild
        let nodeData : NodeData α Unit ([inShape] ++ ss) n.outShape :=
          mkFwdNode (α := α) (Γ := [inShape] ++ ss) (τ := n.outShape) (fun _ctx => hOut ▸ t)
        let st1 : State α inShape := ⟨ss ++ [n.outShape], .snoc (ss := ss) gd nodeData⟩
        have hRec :
            buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
              (i := i + 1) st1 = .ok st' := by
          simpa [st1, nodeData] using hBuild
        have hEval :
            NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
                (input := input) (vals := vals0) (i := i) =
              .ok (NN.IR.DVal.mk (α := α) n.outShape (nodeData.forward ctx ())) := by
          cases hOut
          have hShape :
              (NN.IR.DVal.mk (α := α) n.outShape t).shape = n.outShape := rfl
          simp [NN.IR.Graph.evalAt, hN, hk, hT, input, hShape, throw, throwThe,
            MonadExceptOf.throw, Except.instMonad, Except.bind, Except.pure, nodeData]
          try rfl
        have hStep :
            denoteAllState (α := α) inShape st1 x =
              vals0.push (NN.IR.DVal.mk (α := α) n.outShape (nodeData.forward ctx ())) := by
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

end Compiled
end Autograd
end Runtime
