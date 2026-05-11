/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Compiled.IRExec.Correctness.Common

/-!
# Linear Algebra

Linear-algebra correctness lemmas for the IR → compiled runtime bridge.

This file proves the forward-correctness step for compiling a `.matmul` IR node into a single SSA
node in the compiled `GraphData`. Concretely, it shows that:

* if `buildFrom` successfully compiles a `.matmul` node at position `i`, and
* we compare the IR evaluator `NN.IR.Graph.denoteAllFrom` against the compiled evaluator
  `denoteAllState`,

then the value appended by the IR evaluator at step `i` is definitionally the same tensor as the
value produced by the compiled node's `forward`.

We handle the two shape cases supported by TorchLean's current matmul compiler:

* matrix multiply (2D): `Spec.mat_mul_spec`,
* batched matrix multiply (3D): `Tensor.bmm_spec`.

This module is about *semantic correctness*. Performance backends (e.g. external BLAS libraries,
kernel fusion, etc.) are a separate lowering layer and are not involved here.

The shape rules match the public PyTorch APIs for matrix multiply and batched matrix multiply:

* `torch.matmul`: https://pytorch.org/docs/stable/generated/torch.matmul.html
* `torch.bmm`: https://pytorch.org/docs/stable/generated/torch.bmm.html

## Main definitions

- `buildFrom_denoteAllFrom_matmul_mm_success`: correctness step for 2D matmul.
- `buildFrom_denoteAllFrom_matmul_bmm_success`: correctness step for batched matmul.

## Implementation notes

- We explicitly split 2D and 3D cases, which makes shape distinctions visible instead of hiding them
  behind one very generic theorem.
- The theorem shape follows compiler control flow so failed typing/shape branches collapse quickly.
- Matmul proofs can be slow because the compiler has to distinguish 2D matrix multiplication from
  batched 3D multiplication while preserving exact type-level shapes. The
  shape-dispatch lemmas separate from the semantic equality proof.

## Tags

matmul, bmm, correctness, ir, runtime
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

/--
Correctness lemma for the `.matmul` compilation step in the 2D matrix-multiply case.

This is used when the parent shapes match `Spec.mat_mul_spec` (no batch dimension), and gives the
exact equality needed to hand off to the tail-induction hypothesis.
-/
theorem buildFrom_denoteAllFrom_matmul_mm_success
    {α : Type} [Context α] [DecidableEq Shape]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : GraphData α Unit [inShape] ss) (i : Nat) (st' : State α inShape)
    (x : Tensor α inShape) (n : NN.IR.Node)
    (hN : g.getNode i = .ok n) (hk : n.kind = .matmul) (hi : i < g.nodes.size)
    (ih :
      ∀ (st1 : State α inShape),
        buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
          (i := i + 1) st1 = .ok st' →
        NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
          (input := NN.IR.DVal.mk (α := α) inShape x)
          (i := i + 1) (vals := denoteAllState (α := α) inShape st1 x) =
          .ok (denoteAllState (α := α) inShape st' x))
    (aId bId a0 a1 b1 : Nat)
    (hp : n.parents = [aId, bId])
    (ia : Idx ([inShape] ++ ss) (.dim a0 (.dim a1 .scalar)))
    (hIa : mkIdx (inShape := inShape) (ss := ss) aId
      (.dim a0 (.dim a1 .scalar)) = .ok ia)
    (ib : Idx ([inShape] ++ ss) (.dim a1 (.dim b1 .scalar)))
    (hIb : mkIdx (inShape := inShape) (ss := ss) bId
      (.dim a1 (.dim b1 .scalar)) = .ok ib)
    (hOut : (.dim a0 (.dim b1 .scalar)) = n.outShape)
    (hBuildNext :
      buildFrom (α := α) (g := g) (payload := payload)
        (inShape := inShape) (i := i + 1)
        (st := (⟨ss ++ [n.outShape],
          .snoc (ss := ss) gd
            (mkFwdNode (α := α) (Γ := [inShape] ++ ss) (τ := n.outShape) (fun ctx =>
              let aT := getIdx (α := α) (xs := ctx) ia
              let bT := getIdx (α := α) (xs := ctx) ib
              let y : Tensor α (.dim a0 (.dim b1 .scalar)) :=
                Spec.matMulSpec (α := α) (m := a0) (n := a1) (p := b1) aT bT
              hOut ▸ y))⟩ : State α inShape)) = .ok st') :
    NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
      (input := NN.IR.DVal.mk (α := α) inShape x)
      (i := i) (vals := denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x) =
      .ok (denoteAllState (α := α) inShape st' x) := by
  let vals0 : Array (NN.IR.DVal α) :=
    denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x
  let ctx : TList α ([inShape] ++ ss) :=
    GraphData.eval (α := α) (Δ := Unit) (Γ := [inShape]) (ss := ss) gd (.cons x .nil) ()
  let input : NN.IR.DVal α := NN.IR.DVal.mk (α := α) inShape x
  let expected : Shape := .dim a0 (.dim b1 .scalar)
  let nodeData : NodeData α Unit ([inShape] ++ ss) n.outShape :=
    mkFwdNode (α := α) (Γ := [inShape] ++ ss) (τ := n.outShape) (fun ctx =>
      let aT := getIdx (α := α) (xs := ctx) ia
      let bT := getIdx (α := α) (xs := ctx) ib
      let y : Tensor α expected :=
        Spec.matMulSpec (α := α) (m := a0) (n := a1) (p := b1) aT bT
      hOut ▸ y)
  have hRec :
      buildFrom (α := α) (g := g) (payload := payload)
        (inShape := inShape) (i := i + 1)
        (st := (⟨ss ++ [n.outShape], .snoc (ss := ss) gd nodeData⟩ : State α inShape)) = .ok st' :=
          by
    simpa [expected, nodeData] using hBuildNext
  have hGetA :
      vals0[aId]! =
        NN.IR.DVal.mk (α := α) (.dim a0 (.dim a1 .scalar))
          (getIdx (α := α) (xs := ctx) ia) := by
    simpa [vals0, ctx] using
      (denoteAllState_get_mkIdx (inShape := inShape) (ss := ss)
        (gd := gd) (x := x) (pid := aId)
        (s := .dim a0 (.dim a1 .scalar)) (idx := ia) hIa)
  have hGetB :
      vals0[bId]! =
        NN.IR.DVal.mk (α := α) (.dim a1 (.dim b1 .scalar))
          (getIdx (α := α) (xs := ctx) ib) := by
    simpa [vals0, ctx] using
      (denoteAllState_get_mkIdx (inShape := inShape) (ss := ss)
        (gd := gd) (x := x) (pid := bId)
        (s := .dim a1 (.dim b1 .scalar)) (idx := ib) hIb)
  have hEval :
      NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
        (input := input) (vals := vals0) (i := i) =
        .ok (NN.IR.DVal.mk (α := α) n.outShape (nodeData.forward ctx ())) := by
    simpa [nodeData, mkFwdNode] using
      (evalAt_matmul_mm_ok (α := α)
        (g := g) (payload := payload) (input := input) (vals := vals0)
        (i := i) (n := n) (aId := aId) (bId := bId)
        (m := a0) (nDim := a1) (p := b1)
        (aT := getIdx (α := α) (xs := ctx) ia)
        (bT := getIdx (α := α) (xs := ctx) ib)
        hN hk hp hGetA hGetB hOut)
  have hStep :
      denoteAllState (α := α) inShape
        (st := (⟨ss ++ [n.outShape], .snoc (ss := ss) gd nodeData⟩ : State α inShape)) x =
          vals0.push (NN.IR.DVal.mk (α := α) n.outShape (nodeData.forward ctx ())) := by
    simpa [vals0, nodeData, ctx] using
      (denoteAllState_snoc (α := α) (inShape := inShape)
        (ss := ss) (τ := n.outShape) (gd := gd)
        (nodeData := nodeData) (x := x))
  have hTail := ih ⟨ss ++ [n.outShape], .snoc (ss := ss) gd nodeData⟩ hRec
  exact buildFrom_denoteAllFrom_finish (α := α) (g := g) (payload := payload)
    (i := i) (x := x) (hi := hi) (τ := n.outShape)
    (nodeData := nodeData)
    (st1 := ⟨ss ++ [n.outShape], .snoc (ss := ss) gd nodeData⟩)
    (st' := st') (ctx := ctx) (vals0 := vals0) (input := input)
    hTail hEval hStep

/--
Correctness lemma for the `.matmul` compilation step in the batched-matmul (`bmm`) case.

This is used when the parent shapes match `Tensor.bmm_spec` with an explicit batch dimension, and
again yields the exact one-step equality consumed by the semantic equivalence skeleton.
-/
theorem buildFrom_denoteAllFrom_matmul_bmm_success
    {α : Type} [Context α] [DecidableEq Shape]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : GraphData α Unit [inShape] ss) (i : Nat) (st' : State α inShape)
    (x : Tensor α inShape) (n : NN.IR.Node)
    (hN : g.getNode i = .ok n) (hk : n.kind = .matmul) (hi : i < g.nodes.size)
    (ih :
      ∀ (st1 : State α inShape),
        buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
          (i := i + 1) st1 = .ok st' →
        NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
          (input := NN.IR.DVal.mk (α := α) inShape x)
          (i := i + 1) (vals := denoteAllState (α := α) inShape st1 x) =
          .ok (denoteAllState (α := α) inShape st' x))
    (aId bId a0 a1 a2 bP : Nat)
    (hp : n.parents = [aId, bId])
    (ia : Idx ([inShape] ++ ss) (.dim a0 (.dim a1 (.dim a2 .scalar))))
    (hIa : mkIdx (inShape := inShape) (ss := ss) aId
      (.dim a0 (.dim a1 (.dim a2 .scalar))) = .ok ia)
    (ib : Idx ([inShape] ++ ss) (.dim a0 (.dim a2 (.dim bP .scalar))))
    (hIb : mkIdx (inShape := inShape) (ss := ss) bId
      (.dim a0 (.dim a2 (.dim bP .scalar))) = .ok ib)
    (hOut : (.dim a0 (.dim a1 (.dim bP .scalar))) = n.outShape)
    (hBuildNext :
      buildFrom (α := α) (g := g) (payload := payload)
        (inShape := inShape) (i := i + 1)
        (st := (⟨ss ++ [n.outShape],
          .snoc (ss := ss) gd
            (mkFwdNode (α := α) (Γ := [inShape] ++ ss) (τ := n.outShape) (fun ctx =>
              let aT := getIdx (α := α) (xs := ctx) ia
              let bT := getIdx (α := α) (xs := ctx) ib
              let y : Tensor α (.dim a0 (.dim a1 (.dim bP .scalar))) :=
                Tensor.bmmSpec (α := α) (batch := a0) (m := a1) (n := a2) (p := bP) aT bT
              hOut ▸ y))⟩ : State α inShape)) = .ok st')
    :
    NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
      (input := NN.IR.DVal.mk (α := α) inShape x)
      (i := i) (vals := denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x) =
      .ok (denoteAllState (α := α) inShape st' x) := by
  let vals0 : Array (NN.IR.DVal α) :=
    denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x
  let ctx : TList α ([inShape] ++ ss) :=
    GraphData.eval (α := α) (Δ := Unit) (Γ := [inShape]) (ss := ss) gd (.cons x .nil) ()
  let input : NN.IR.DVal α := NN.IR.DVal.mk (α := α) inShape x
  let expected : Shape := .dim a0 (.dim a1 (.dim bP .scalar))
  let nodeData : NodeData α Unit ([inShape] ++ ss) n.outShape :=
    mkFwdNode (α := α) (Γ := [inShape] ++ ss) (τ := n.outShape) (fun ctx =>
      let aT := getIdx (α := α) (xs := ctx) ia
      let bT := getIdx (α := α) (xs := ctx) ib
      let y : Tensor α expected :=
        Tensor.bmmSpec (α := α) (batch := a0) (m := a1) (n := a2) (p := bP) aT bT
      hOut ▸ y)
  have hRec :
      buildFrom (α := α) (g := g) (payload := payload)
        (inShape := inShape) (i := i + 1)
        (st := (⟨ss ++ [n.outShape], .snoc (ss := ss) gd nodeData⟩ : State α inShape)) = .ok st' :=
          by
    simpa [expected, nodeData] using hBuildNext
  have hGetA :
      vals0[aId]! =
        NN.IR.DVal.mk (α := α) (.dim a0 (.dim a1 (.dim a2 .scalar)))
          (getIdx (α := α) (xs := ctx) ia) := by
    simpa [vals0, ctx] using
      (denoteAllState_get_mkIdx (inShape := inShape) (ss := ss)
        (gd := gd) (x := x) (pid := aId)
        (s := .dim a0 (.dim a1 (.dim a2 .scalar))) (idx := ia) hIa)
  have hGetB :
      vals0[bId]! =
        NN.IR.DVal.mk (α := α) (.dim a0 (.dim a2 (.dim bP .scalar)))
          (getIdx (α := α) (xs := ctx) ib) := by
    simpa [vals0, ctx] using
      (denoteAllState_get_mkIdx (inShape := inShape) (ss := ss)
        (gd := gd) (x := x) (pid := bId)
        (s := .dim a0 (.dim a2 (.dim bP .scalar))) (idx := ib) hIb)
  have hEval :
      NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
        (input := input) (vals := vals0) (i := i) =
        .ok (NN.IR.DVal.mk (α := α) n.outShape (nodeData.forward ctx ())) := by
    simpa [nodeData, mkFwdNode] using
      (evalAt_matmul_bmm_ok (α := α)
        (g := g) (payload := payload) (input := input) (vals := vals0)
        (i := i) (n := n) (aId := aId) (bId := bId)
        (batch := a0) (m := a1) (nDim := a2) (p := bP)
        (aT := getIdx (α := α) (xs := ctx) ia)
        (bT := getIdx (α := α) (xs := ctx) ib)
        hN hk hp hGetA hGetB hOut)
  have hStep :
      denoteAllState (α := α) inShape
        (st := (⟨ss ++ [n.outShape], .snoc (ss := ss) gd nodeData⟩ : State α inShape)) x =
          vals0.push (NN.IR.DVal.mk (α := α) n.outShape (nodeData.forward ctx ())) := by
    simpa [vals0, nodeData, ctx] using
      (denoteAllState_snoc (α := α) (inShape := inShape)
        (ss := ss) (τ := n.outShape) (gd := gd)
        (nodeData := nodeData) (x := x))
  have hTail := ih ⟨ss ++ [n.outShape], .snoc (ss := ss) gd nodeData⟩ hRec
  exact buildFrom_denoteAllFrom_finish (α := α) (g := g) (payload := payload)
    (i := i) (x := x) (hi := hi) (τ := n.outShape)
    (nodeData := nodeData)
    (st1 := ⟨ss ++ [n.outShape], .snoc (ss := ss) gd nodeData⟩)
    (st' := st') (ctx := ctx) (vals0 := vals0) (input := input)
    hTail hEval hStep

/--
Correctness lemma for the `.matmul` node compiler.

This is a dispatcher that selects the appropriate specialized lemma:
- `buildFrom_denoteAllFrom_matmul_mm_success` for 2D matmul, or
- `buildFrom_denoteAllFrom_matmul_bmm_success` for batched matmul.
-/
theorem buildFrom_denoteAllFrom_matmul
    {α : Type} [Context α] [DecidableEq Shape]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : GraphData α Unit [inShape] ss) (i : Nat) (st' : State α inShape)
    (x : Tensor α inShape) (n : NN.IR.Node)
    (hN : g.getNode i = .ok n) (hk : n.kind = .matmul) (hi : i < g.nodes.size)
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
      cases hBuild
  | cons aId rest =>
      cases rest with
      | nil =>
          simp [hp] at hBuild
          cases hBuild
      | cons bId rest2 =>
          cases rest2 with
          | cons _ _ =>
              simp [hp] at hBuild
              cases hBuild
          | nil =>
              cases hA : g.getNode aId with
              | error msg =>
                  simp [hp, hA] at hBuild
              | ok aNode =>
                  cases hB : g.getNode bId with
                  | error msg =>
                      simp [hp, hA, hB] at hBuild
                  | ok bNode =>
                      simp (config := { failIfUnchanged := false }) [hp, hA, hB] at hBuild
                      cases hAS : aNode.outShape with
                      | scalar =>
                          simp (config := { failIfUnchanged := false }) [hAS] at hBuild
                          try cases hBuild
                      | dim a0 aTail =>
                          cases aTail with
                          | scalar =>
                              simp (config := { failIfUnchanged := false }) [hAS] at hBuild
                              try cases hBuild
                          | dim a1 aTail2 =>
                              cases aTail2 with
                              | scalar =>
                                  cases hBS : bNode.outShape with
                                  | scalar =>
                                      simp (config := { failIfUnchanged := false })
                                        [hAS, hBS] at hBuild
                                      try cases hBuild
                                  | dim b0 bTail =>
                                      cases bTail with
                                      | scalar =>
                                          simp (config := { failIfUnchanged := false }) [hAS, hBS]
                                            at hBuild
                                          try cases hBuild
                                      | dim b1 bTail2 =>
                                          cases bTail2 with
                                          | dim _ _ =>
                                              simp (config := { failIfUnchanged := false }) [hAS,
                                                hBS] at hBuild
                                              try cases hBuild
                                          | scalar =>
                                              by_cases hn : a1 = b0
                                              ·
                                                cases hn
                                                cases hIa :
                                                    mkIdx (inShape := inShape) (ss := ss) aId
                                                      (.dim a0 (.dim a1 .scalar)) with
                                                | error msg =>
                                                    simp [hAS, hBS, hIa] at hBuild
                                                | ok ia =>
                                                    cases hIb :
                                                        mkIdx (inShape := inShape) (ss := ss) bId
                                                          (.dim a1 (.dim b1 .scalar)) with
                                                    | error msg =>
                                                        simp [hAS, hBS, hIa, hIb] at hBuild
                                                    | ok ib =>
                                                        let expected : Shape :=
                                                          .dim a0 (.dim b1 .scalar)
                                                        by_cases hOut : expected = n.outShape
                                                        ·
                                                          have hBuildNext := by
                                                            simpa
                                                                [hAS, hBS, hIa, hIb, expected,
                                                                  hOut]
                                                                using hBuild
                                                          exact
                                                            buildFrom_denoteAllFrom_matmul_mm_success
                                                              (α := α)
                                                              (g := g) (payload := payload)
                                                              (gd := gd) (i := i) (st' := st')
                                                              (x := x) (n := n) hN hk hi ih
                                                              (aId := aId) (bId := bId)
                                                              (a0 := a0) (a1 := a1) (b1 := b1)
                                                              hp (ia := ia) hIa (ib := ib) hIb
                                                              hOut hBuildNext
                                                        ·
                                                          exact False.elim <|
                                                            throw_bind_ne_ok (by
                                                              simpa [hAS, hBS, hIa, hIb, expected, hOut] using hBuild)
                                              ·
                                                exact False.elim <|
                                                  throw_bind_ne_ok (by simpa [hAS, hBS, hn] using hBuild)
                              | dim a2 aTail3 =>
                                  cases aTail3 with
                                  | dim _ _ =>
                                      simp (config := { failIfUnchanged := false }) [hAS] at hBuild
                                      try cases hBuild
                                  | scalar =>
                                      cases hBS : bNode.outShape with
                                      | scalar =>
                                          simp (config := { failIfUnchanged := false }) [hAS, hBS]
                                            at hBuild
                                          try cases hBuild
                                      | dim bBatch bTail =>
                                          cases bTail with
                                          | scalar =>
                                              simp (config := { failIfUnchanged := false }) [hAS,
                                                hBS] at hBuild
                                              try cases hBuild
                                          | dim bN bTail2 =>
                                              cases bTail2 with
                                              | scalar =>
                                                  simp (config := { failIfUnchanged := false })
                                                    [hAS, hBS] at hBuild
                                                  try cases hBuild
                                              | dim bP bTail3 =>
                                                  cases bTail3 with
                                                  | dim _ _ =>
                                                      simp (config := { failIfUnchanged := false })
                                                        [hAS, hBS] at hBuild
                                                      try cases hBuild
                                                  | scalar =>
                                                      by_cases hb : a0 = bBatch
                                                      ·
                                                        by_cases hn : a2 = bN
                                                        ·
                                                          cases hb
                                                          cases hn
                                                          cases hIa :
                                                              mkIdx (inShape := inShape) (ss := ss)
                                                                aId
                                                                (.dim a0 (.dim a1 (.dim a2
                                                                  .scalar))) with
                                                          | error msg =>
                                                              simp [hAS, hBS, hIa] at hBuild
                                                          | ok ia =>
                                                              cases hIb :
                                                                  mkIdx (inShape := inShape) (ss :=
                                                                    ss) bId
                                                                    (.dim a0 (.dim a2 (.dim bP
                                                                      .scalar))) with
                                                              | error msg =>
                                                                  simp [hAS, hBS, hIa, hIb] at hBuild
                                                              | ok ib =>
                                                                  let expected : Shape :=
                                                                    .dim a0
                                                                      (.dim a1 (.dim bP .scalar))
                                                                  by_cases hOut :
                                                                      expected = n.outShape
                                                                  ·
                                                                    have hBuildNext := by
                                                                      simpa
                                                                          [hAS, hBS, hIa, hIb,
                                                                            expected, hOut]
                                                                          using hBuild
                                                                    exact
                                                                      buildFrom_denoteAllFrom_matmul_bmm_success
                                                                        (α := α)
                                                                        (g := g)
                                                                        (payload := payload)
                                                                        (gd := gd) (i := i)
                                                                        (st' := st')
                                                                        (x := x) (n := n) hN hk hi
                                                                        ih
                                                                        (aId := aId) (bId := bId)
                                                                        (a0 := a0) (a1 := a1)
                                                                        (a2 := a2) (bP := bP)
                                                                        hp (ia := ia) hIa (ib := ib)
                                                                        hIb hOut hBuildNext
                                                                  · exact False.elim <|
                                                                      throw_bind_ne_ok (by
                                                                        simpa [hAS, hBS, hIa, hIb, expected, hOut]
                                                                          using hBuild)
                                                        ·
                                                          exact False.elim <|
                                                            throw_bind_ne_ok (by
                                                              simpa [hAS, hBS, hb, hn] using hBuild)
                                                      ·
                                                        exact False.elim <|
                                                          throw_bind_ne_ok (by simpa [hAS, hBS, hb] using hBuild)



end Compiled
end Autograd
end Runtime
