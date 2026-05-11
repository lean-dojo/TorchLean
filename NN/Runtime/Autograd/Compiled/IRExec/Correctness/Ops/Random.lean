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
# Random

Correctness lemmas for random IR nodes in the IR -> compiled runtime bridge.

These lemmas keep the end-to-end semantic equivalence proof in `Correctness.SemanticEquivalence` small: the top-level proof
can dispatch to branch theorems, while this file checks branch-specific compiler and evaluator
behavior.

Build note: the random operators are deterministic in the semantics once the seed and node id are
fixed. The proof still has to show that the compiler and IR evaluator derive the same key, append a
value of the same dependent shape, and continue with the same tail graph. Seed/key helper lemmas
keep additional deterministic random primitives mechanical.

## Main definitions

- `buildFrom_denoteAllFrom_rand_uniform`
- `buildFrom_denoteAllFrom_bernoulli_mask`
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

set_option maxHeartbeats 1200000 in
/-- Correctness lemma for `.randUniform seed` lowering. -/
theorem buildFrom_denoteAllFrom_rand_uniform
    {α : Type} [Context α] [DecidableEq Shape]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : GraphData α Unit [inShape] ss) (i : Nat) (st' : State α inShape)
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
      let key := Runtime.Autograd.TorchLean.Random.keyOf seed i
      let t : Tensor α n.outShape := Runtime.Autograd.TorchLean.Random.uniform (α := α)
        key (s := n.outShape)
      let nodeData : NodeData α Unit ([inShape] ++ ss) n.outShape :=
        mkFwdNode (α := α) (Γ := [inShape] ++ ss) (τ := n.outShape) (fun _ctx => t)
      let st1 : State α inShape := ⟨ss ++ [n.outShape], .snoc (ss := ss) gd nodeData⟩
      have hRec :
          buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
            (i := i + 1) st1 = .ok st' := by
        simpa [st1, nodeData] using hBuild
      have hEval :
          NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
              (input := input) (vals := vals0) (i := i) =
            .ok (NN.IR.DVal.mk (α := α) n.outShape (nodeData.forward ctx ())) := by
        simp [NN.IR.Graph.evalAt, hN, hk, hp, nodeData, mkFwdNode,
          NN.IR.DVal.shape, NN.IR.DVal.tensor, NN.IR.DVal.mk,
          key, t,
          throw_eq_error,
          Except.instMonad, Except.bind, Except.pure]
        rfl
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
  | cons _ _ =>
      exact False.elim <| throw_bind_ne_ok (by simpa [hp] using hBuild)

set_option maxHeartbeats 1200000 in
/-- Correctness lemma for `.bernoulliMask seed` lowering. -/
theorem buildFrom_denoteAllFrom_bernoulli_mask
    {α : Type} [Context α] [DecidableEq Shape]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : GraphData α Unit [inShape] ss) (i : Nat) (st' : State α inShape)
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
          cases hIdx : mkIdx (inShape := inShape) (ss := ss) pId Shape.scalar with
          | error msg =>
              simp [hp, hIdx] at hBuild
              try cases hBuild
          | ok ip =>
              simp [hp, hIdx] at hBuild
              let key := Runtime.Autograd.TorchLean.Random.keyOf seed i
              let nodeData : NodeData α Unit ([inShape] ++ ss) n.outShape :=
                mkFwdNode (α := α) (Γ := [inShape] ++ ss) (τ := n.outShape) (fun ctx =>
                  let kpT := getIdx (α := α) (xs := ctx) ip
                  let kp : α := match kpT with | Tensor.scalar v => v
                  Runtime.Autograd.TorchLean.Random.mask (α := α) key kp (s := n.outShape))
              let st1 : State α inShape := ⟨ss ++ [n.outShape], .snoc (ss := ss) gd nodeData⟩
              have hRec :
                  buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
                      (i := i + 1) st1 =
                    .ok st' := by
                simpa [st1, nodeData] using hBuild
              have hGet :
                  vals0[pId]! =
                    NN.IR.DVal.mk (α := α) Shape.scalar (getIdx (α := α) (xs := ctx) ip) := by
                simpa [vals0, ctx] using
                  (denoteAllState_get_mkIdx (inShape := inShape) (ss := ss)
                    (gd := gd) (x := x) (pid := pId) (s := Shape.scalar) (idx := ip) hIdx)
              have hEval :
                  NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
                      (input := input) (vals := vals0) (i := i) =
                    .ok (NN.IR.DVal.mk (α := α) n.outShape (nodeData.forward ctx ())) := by
                cases hkp : getIdx (α := α) (xs := ctx) ip with
                | scalar keepProb =>
                    set pV : NN.IR.DVal α := vals0[pId]! with hpV
                    have hPV0 :
                        pV =
                          NN.IR.DVal.mk (α := α) Shape.scalar
                            (getIdx (α := α) (xs := ctx) ip) := by
                      exact Eq.trans hpV hGet
                    have hPV :
                        pV = NN.IR.DVal.mk (α := α) Shape.scalar (Tensor.scalar keepProb) := by
                      -- Rewrite the parent scalar tensor explicitly to avoid simp-orientation
                      -- fragility.
                      simpa using (hPV0.trans (by simpa [hkp]))
                    simp [NN.IR.Graph.evalAt, hN, hk, hp, hkp,
                      nodeData, mkFwdNode, NN.IR.DVal.shape, NN.IR.DVal.tensor,
                        NN.IR.DVal.mk,
                      throw_eq_error,
                      Except.instMonad, Except.bind, Except.pure, key]
                    rw [← hpV]
                    rw [hPV]
                    have hMatch :
                        (match getIdx (α := α) (xs := ctx) ip with
                        | Tensor.scalar v => v) =
                          keepProb := by
                      have h := congrArg
                        (fun t : Tensor α Shape.scalar => match t with | Tensor.scalar v => v) hkp
                      simpa using h
                    have hMask :
                        Runtime.Autograd.TorchLean.Random.mask (α := α)
                            (Runtime.Autograd.TorchLean.Random.keyOf seed i) keepProb
                            (s := n.outShape) =
                          Runtime.Autograd.TorchLean.Random.mask (α := α)
                            (Runtime.Autograd.TorchLean.Random.keyOf seed i)
                            (match getIdx (α := α) (xs := ctx) ip with
                            | Tensor.scalar v => v)
                            (s := n.outShape) := by
                      have h := congrArg
                        (fun kp : α =>
                          Runtime.Autograd.TorchLean.Random.mask (α := α)
                            (Runtime.Autograd.TorchLean.Random.keyOf seed i) kp (s := n.outShape))
                        hMatch.symm
                      simpa using h
                    -- Turn the goal into equality of the underlying tensors, then apply `hMask`.
                    simp [hMask]
                    rfl
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
