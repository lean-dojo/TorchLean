/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.IRExec.Correctness.Common

/-!
# Random

Correctness lemmas for random IR nodes in the IR-to-forward-executor lowering.

These lemmas keep the end-to-end semantic equivalence proof in `Correctness.SemanticEquivalence`
small: the top-level proof can dispatch to branch theorems, while this file checks branch-specific
lowering pass and evaluator behavior.

Build note: the random operators are deterministic in the semantics once the seed and node id are
fixed. The proof still has to show that the lowering pass and IR evaluator derive the same key,
append a value of the same dependent shape, and continue with the same tail graph. Seed/key helper
lemmas keep additional deterministic random primitives mechanical.

## Main definitions

- `buildFrom_denoteAllFrom_rand_uniform`
- `buildFrom_denoteAllFrom_bernoulli_mask`
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

/-- Correctness lemma for `.randUniform seed` lowering. -/
theorem buildFrom_denoteAllFrom_rand_uniform
    {α : Type} [TorchLean.Storage α] [Context α]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : ForwardData α [inShape] ss) (i : Nat) (st' : State α inShape)
    (x : Tensor α inShape) (n : NN.IR.Node)
    (seed : Nat)
    (hN : g.getNode i = .ok n) (hk : n.kind = .randUniform seed) (hi : i < g.nodes.size)
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
  simp (config := { failIfUnchanged := false }) [hk, lowerRandUniform] at hBuild
  cases hp : n.parents.isEmpty with
  | true =>
      simp [hp] at hBuild
      have hParents : n.parents = #[] := by
        simpa using hp
      let key := Spec.Random.keyOf seed i
      let t : Tensor α n.outShape := Spec.Random.uniform (α := α)
        key (s := n.outShape)
      let nodeData : ForwardNode α ([inShape] ++ ss) n.outShape :=
        mkForwardNode (α := α) (Γ := [inShape] ++ ss) (τ := n.outShape) (fun _ctx => t)
      let st1 : State α inShape := ⟨ss ++ [n.outShape], .snoc (ss := ss) gd nodeData⟩
      have hRec :
          buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
            (i := i + 1) st1 = .ok st' := by
        simpa [st1, nodeData] using hBuild
      have hEval :
          NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
              (input := input) (vals := vals0) (i := i) =
            .ok (Spec.SomeTensor.mk (α := α) n.outShape (nodeData.eval ctx)) := by
        simp [NN.IR.Graph.evalAt, NN.IR.Graph.evalNode, NN.IR.Graph.normalizeNodeOutput,
          hN, hk, hParents, nodeData, mkForwardNode,
          key, t,
          throw_eq_error]
        rfl
      have hStep :
          denoteAllState (α := α) inShape st1 x =
            vals0.push
              (Spec.SomeTensor.mk (α := α) n.outShape (nodeData.eval ctx)) := by
        simpa [vals0, st1, nodeData, ctx] using
          (denoteAllState_snoc (α := α) (inShape := inShape) (ss := ss)
            (τ := n.outShape) (gd := gd) (nodeData := nodeData) (x := x))
      have hTail := ih st1 hRec
      exact buildFrom_denoteAllFrom_finish (α := α) (g := g) (payload := payload)
        (i := i) (x := x) (hi := hi) (τ := n.outShape)
        (nodeData := nodeData) (st1 := st1) (st' := st')
        (ctx := ctx) (vals0 := vals0) (input := input) hTail hEval hStep
  | false =>
      exact False.elim <| throw_bind_ne_ok (by simpa [hp] using hBuild)

/-- Correctness lemma for `.bernoulliMask seed` lowering. -/
theorem buildFrom_denoteAllFrom_bernoulli_mask
    {α : Type} [TorchLean.Storage α] [Context α]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : ForwardData α [inShape] ss) (i : Nat) (st' : State α inShape)
    (x : Tensor α inShape) (n : NN.IR.Node)
    (seed : Nat)
    (hN : g.getNode i = .ok n) (hk : n.kind = .bernoulliMask seed) (hi : i < g.nodes.size)
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
  simp (config := { failIfUnchanged := false }) [hk, lowerBernoulliMask] at hBuild
  cases hp : unaryParent? n.parents with
  | none =>
      simp [hp] at hBuild
      try cases hBuild
  | some pId =>
          cases hIdx : mkIdx (inShape := inShape) (ss := ss) pId Shape.scalar with
          | error msg =>
              simp [hp, hIdx] at hBuild
              try cases hBuild
          | ok ip =>
              simp [hp, hIdx] at hBuild
              let key := Spec.Random.keyOf seed i
              let nodeData : ForwardNode α ([inShape] ++ ss) n.outShape :=
                mkForwardNode (α := α) (Γ := [inShape] ++ ss) (τ := n.outShape) (fun ctx =>
                  let kpT := readTensor (α := α) (xs := ctx) ip
                  let kp : α := kpT.item
                  Spec.Random.mask (α := α) key kp (s := n.outShape))
              let st1 : State α inShape := ⟨ss ++ [n.outShape], .snoc (ss := ss) gd nodeData⟩
              have hRec :
                  buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
                      (i := i + 1) st1 =
                    .ok st' := by
                change
                  buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
                      (i := i + 1)
                      ⟨ss ++ [n.outShape],
                        ForwardData.snoc (α := α) (Γ := [inShape]) (ss := ss)
                          (τ := n.outShape) gd
                          (mkForwardNode (α := α) (Γ := [inShape] ++ ss) (τ := n.outShape)
                            (fun ctx =>
                              Spec.Random.mask (α := α)
                                (Spec.Random.keyOf seed i)
                                (readTensor (α := α) (xs := ctx) ip).item
                                (s := n.outShape)))⟩ =
                    .ok st'
                exact hBuild
              have hGet :
                  vals0[pId]? =
                    some (Spec.SomeTensor.mk (α := α) Shape.scalar
                      (getIdx (α := α) (xs := ctx) ip)) := by
                simpa [vals0, ctx] using
                  (denoteAllState_get_mkIdx? (inShape := inShape) (ss := ss)
                    (gd := gd) (x := x) (pid := pId) (s := Shape.scalar) (idx := ip) hIdx)
              have hEval :
                  NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
                      (input := input) (vals := vals0) (i := i) =
                    .ok (Spec.SomeTensor.mk (α := α) n.outShape (nodeData.eval ctx)) := by
                simp [NN.IR.Graph.evalAt, NN.IR.Graph.evalNode,
                  NN.IR.Graph.normalizeNodeOutput, hN, hk, hp, hGet, nodeData,
                  mkForwardNode, throw_eq_error, key]
                rfl
              have hStep :
                  denoteAllState (α := α) inShape st1 x =
                    vals0.push
                      (Spec.SomeTensor.mk (α := α) n.outShape (nodeData.eval ctx)) := by
                simpa [vals0, st1, nodeData, ctx] using
                  (denoteAllState_snoc (α := α) (inShape := inShape) (ss := ss)
                    (τ := n.outShape) (gd := gd) (nodeData := nodeData) (x := x))
              have hTail := ih st1 hRec
              exact buildFrom_denoteAllFrom_finish (α := α) (g := g) (payload := payload)
                (i := i) (x := x) (hi := hi) (τ := n.outShape)
                (nodeData := nodeData) (st1 := st1) (st' := st')
                (ctx := ctx) (vals0 := vals0) (input := input) hTail hEval hStep

end IRExec
end Autograd
end Runtime
