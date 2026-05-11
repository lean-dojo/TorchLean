/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Compiled.IRExec.Correctness.Common
public import NN.Runtime.Autograd.Compiled.IRExec.Correctness.SemanticEquivalenceCommon

set_option linter.unusedSimpArgs false
set_option linter.unnecessarySimpa false

/-!
# Structural Nodes

Structural correctness facts for IR nodes that do not lower to ordinary executable operators.

The first node of a compiled graph is the distinguished input, but the recursive `buildFrom` loop
starts after that input node. If `buildFrom` ever encounters another `.input` node while compiling
the tail, successful compilation is impossible. We keep that fact as a named theorem so the
top-level semantic-equivalence proof can dispatch to it directly.

This file also contains the correctness lemma for `.detach`. Although `.detach` appears in the IR
as a node, it does not change the tensor value at the spec layer, and the compiled lowering is just
the identity function with a shape check.

Build note: structural nodes look simple, but they sit on the boundary between graph control flow
and tensor semantics. The `.input` case is an impossible-success proof, while `.detach` is a value
identity proof wrapped in parent and shape checks. These boundary cases should stay
small and separate from numeric operator proofs.
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

/-- The recursive compiler cannot successfully compile an `.input` node in the graph tail. -/
theorem buildFrom_denoteAllFrom_input_impossible
    {α : Type} [Context α] [DecidableEq Shape]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : GraphData α Unit [inShape] ss) (i : Nat) (st' : State α inShape)
    (x : Tensor α inShape) (n : NN.IR.Node)
    (hN : g.getNode i = .ok n) (hk : n.kind = .input) (hi : i < g.nodes.size)
    (hBuild :
      buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
        (i := i) (st := (⟨ss, gd⟩ : State α inShape)) = .ok st') :
    NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
      (input := NN.IR.DVal.mk (α := α) inShape x)
      (i := i) (vals := denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x) =
      .ok (denoteAllState (α := α) inShape st' x) := by
  have : False := by
    unfold buildFrom at hBuild
    simpa [hi, hN, hk, throw_eq_error, Except.instMonad, Except.bind, Except.pure] using hBuild
  cases this

/-- Semantic-preservation lemma for `.detach` lowering. -/
theorem buildFrom_denoteAllFrom_detach
    {α : Type} [Context α] [DecidableEq Shape]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : GraphData α Unit [inShape] ss) (i : Nat) (st' : State α inShape)
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
                    let nodeData : NodeData α Unit ([inShape] ++ ss) n.outShape :=
                      mkFwdNode (α := α) (Γ := [inShape] ++ ss) (τ := n.outShape) (fun ctx =>
                        hOut ▸ (getIdx (α := α) (xs := ctx) ip))
                    let st1 : State α inShape :=
                      ⟨ss ++ [n.outShape], .snoc (ss := ss) gd nodeData⟩
                    have hRec :
                        buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
                          (i := i + 1) st1 = .ok st' := by
                      simpa [st1, nodeData] using hBuild

                    have hGet :
                        vals0[pId]! =
                          NN.IR.DVal.mk (α := α) pNode.outShape
                            (getIdx (α := α) (xs := ctx) ip) := by
                      simpa [vals0, ctx] using
                        (denoteAllState_get_mkIdx (inShape := inShape) (ss := ss)
                          (gd := gd) (x := x) (pid := pId) (s := pNode.outShape) (idx := ip) hIdx)

                    have hEval :
                        NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
                            (input := input) (vals := vals0) (i := i) =
                          .ok (NN.IR.DVal.mk (α := α) n.outShape (nodeData.forward ctx ())) := by
                      let pV : NN.IR.DVal α := vals0[pId]!
                      have hPV :
                          pV =
                            NN.IR.DVal.mk (α := α) pNode.outShape
                              (getIdx (α := α) (xs := ctx) ip) := by
                        simpa [pV] using hGet
                      have hExpect :
                          NN.IR.Graph.expectShape (α := α) (expected := n.outShape) pV =
                            .ok (hOut ▸ getIdx (α := α) (xs := ctx) ip) := by
                        rw [hPV]
                        -- `expectShape` is a dependent `if` on shape equality. We take the
                        -- successful branch explicitly and then normalize the cast proof using
                        -- proof-irrelevance for tensor transports.
                        by_cases hEq : pNode.outShape = n.outShape
                        · have hCast :
                            (hEq ▸ getIdx (α := α) (xs := ctx) ip) =
                              (hOut ▸ getIdx (α := α) (xs := ctx) ip) := by
                            simpa using
                              (Tensor.eqRec_proof_irrel
                                (t := getIdx (α := α) (xs := ctx) ip) (p := hEq) (q := hOut))
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
                            simpa [hCast]
                          simpa [NN.IR.Graph.expectShape, NN.IR.DVal.shape, NN.IR.DVal.tensor,
                            NN.IR.DVal.mk, hEq, throw_eq_error, Except.instMonad, Except.bind,
                            Except.pure, hOk]
                        · cases (hEq hOut)
                      simp [NN.IR.Graph.evalAt, hN, hk, hp, pV, hExpect, hOut,
                        nodeData, mkFwdNode, NN.IR.DVal.shape, NN.IR.DVal.tensor, NN.IR.DVal.mk,
                        throw_eq_error, Except.instMonad, Except.bind, Except.pure]

                    have hTail := ih st1 hRec
                    exact buildFrom_denoteAllFrom_unary_exact (α := α) (g := g) (payload := payload)
                      (gd := gd) (i := i) (st' := st') (x := x) (hi := hi)
                      (τ := n.outShape) (nodeData := nodeData) hTail (by simpa using hEval)
                  · simp [hOut] at hBuild
                    try cases hBuild

end Compiled
end Autograd
end Runtime
