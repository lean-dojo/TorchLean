/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.IRExec.Correctness.Common
public import NN.Runtime.Autograd.IRExec.Correctness.SemanticEquivalenceCommon

/-!
# Semantic Equivalence (Op Cases)

This module contains semantic-preservation lemmas for IR node kinds handled
inline in the main recursive theorem [`Correctness.SemanticEquivalence`].

Why split these out?

1. **Lowering performance:** `SemanticEquivalence.lean` is a large mutually-dependent proof
   script; extracting the heaviest branches into separate theorems makes elaboration more
   incremental and keeps error messages local to the relevant operator case.
2. **Auditability:** these cases are part of the lowering pass/denotation contract. Giving them
   named theorems makes it easier to see which IR fragments are covered.

The proofs follow the same pattern as the per-operator modules under `Correctness/Ops/`:

* unfold `buildFrom` and mirror its runtime checks,
* construct the lowered `nodeData` forward closure,
* show that `NN.IR.Graph.evalAt` produces the same dynamic value,
* finish with the shared `buildFrom_denoteAllFrom_finish` lemma for the tail.

Build note: these proofs are slow because each branch normalizes both the lowering pass and the IR
evaluator, then proves that the resulting dynamic value agrees with a shape-indexed forward node.
Shape casts and `Except` error paths are the main source of proof noise. The local linter scopes in
this file mark the current proof-engineering boundary: the proof is checked, but the simplification
scripts still deserve a pass with more helper lemmas.

More cases should live in `Correctness/Ops/*`, with repeated parent and cast facts packaged as
lemmas from `SemanticEquivalenceCommon`.
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

/--
Semantic-preservation lemma for `.linear` lowering (payload-backed affine map).

The parent may carry any leading shape before its final `inDim` axis; both the lowering and the IR
evaluator apply `NN.IR.Graph.linearLeading` over that leading shape.
-/
theorem buildFrom_denoteAllFrom_linear
    {α : Type} [TorchLean.Storage α] [Context α]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : ForwardData α [inShape] ss) (i : Nat) (st' : State α inShape)
    (x : Tensor α inShape) (n : NN.IR.Node)
    (hN : g.getNode i = .ok n) (hk : n.kind = .linear) (hi : i < g.nodes.size)
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
  simp (config := { failIfUnchanged := false }) [hk, lowerLinear] at hBuild
  cases hp : unaryParent? n.parents with
  | none =>
      simp [hp] at hBuild; try cases hBuild
  | some xId =>
  cases hLin : payload.linear? n.id with
  | none =>
      simp [hp, hLin] at hBuild; try cases hBuild
  | some p =>
  cases hX : g.getNode xId with
  | error msg => simp [hp, hLin, hX] at hBuild
  | ok xNode =>
  simp (config := { failIfUnchanged := false }) [hp, hLin, hX] at hBuild
  cases hIdx : mkIdx (inShape := inShape) (ss := ss) xId xNode.outShape with
  | error msg =>
      simp [hIdx] at hBuild
  | ok ix =>
  simp (config := { failIfUnchanged := false }) [hIdx] at hBuild
  let leading : Shape := Shape.ofList xNode.outShape.toList.dropLast
  let expectedIn : Shape := leading.concat [p.inDim]
  let expectedOut : Shape := leading.concat [p.outDim]
  -- Both guards are dependent `if`s whose conditions mention the parent shape on both sides. Split
  -- them rather than using them as rewrite rules: `by_cases` on the outer shape leaves instance
  -- search for `Decidable` stuck on the let-bound shapes.
  split at hBuild
  · rename_i hIn
    split at hBuild
    · rename_i hOut
      let nodeData : ForwardNode α ([inShape] ++ ss) n.outShape :=
        mkForwardNode (α := α) (Γ := [inShape] ++ ss) (τ := n.outShape) (fun ctx =>
          let xIn : Tensor α expectedIn :=
            Tensor.castShape (getIdx (α := α) (xs := ctx) ix) hIn
          let y : Tensor α expectedOut := NN.IR.Graph.linearLeading leading p.W p.b xIn
          Tensor.castShape y hOut)
      let st1 : State α inShape := ⟨ss ++ [n.outShape], .snoc (ss := ss) gd nodeData⟩
      have hRec :
          buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
            (i := i + 1) st1 = .ok st' := by
        simpa [st1, nodeData, leading, expectedIn, expectedOut, Pure.pure, Except.pure]
          using hBuild
      have hGet :
          vals0[xId]? = some
            (Spec.SomeTensor.mk (α := α) xNode.outShape (getIdx (α := α) (xs := ctx) ix)) := by
        simpa [vals0, ctx] using
          (denoteAllState_get_mkIdx? (inShape := inShape) (ss := ss)
            (gd := gd) (x := x) (pid := xId) (s := xNode.outShape) (idx := ix) hIdx)
      have hExpect :
          NN.IR.Graph.expectShape (α := α) (expected := expectedIn)
              (Spec.SomeTensor.mk (α := α) xNode.outShape (getIdx (α := α) (xs := ctx) ix)) =
            .ok ((getIdx (α := α) (xs := ctx) ix).castShape hIn) :=
        Graph.expectShape_mk_of_eq hIn _
      have hEval :
          NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
              (input := input) (vals := vals0) (i := i) =
            .ok (Spec.SomeTensor.mk (α := α) n.outShape (nodeData.eval ctx)) := by
        -- Split into two stages so `simp` does not explore unrelated `evalAt` branches.
        simp [NN.IR.Graph.evalAt, NN.IR.Graph.evalNode,
          NN.IR.Graph.normalizeNodeOutput, hN, hk, hp, hGet]
        -- `evalLinear` recovers the leading shape from the parent value and checks the symmetric
        -- condition `n.outShape = expectedOut`.
        have hOut' : n.outShape = expectedOut := hOut.symm
        simp [NN.IR.Graph.evalLinear, hLin, expectedIn, expectedOut, leading, hExpect, hOut',
          nodeData, Tensor.eqRec_eq_cast_shape, Pure.pure, Except.pure]
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
    · exact False.elim <| throw_bind_ne_ok hBuild
  · exact False.elim <| throw_bind_ne_ok hBuild

/-- Semantic-preservation lemma for `.reshape inS outS` lowering. -/
theorem buildFrom_denoteAllFrom_reshape
    {α : Type} [TorchLean.Storage α] [Context α]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : ForwardData α [inShape] ss) (i : Nat) (st' : State α inShape)
    (x : Tensor α inShape) (n : NN.IR.Node)
    (inS outS : Shape)
    (hN : g.getNode i = .ok n) (hk : n.kind = .reshape inS outS) (hi : i < g.nodes.size)
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
  simp (config := { failIfUnchanged := false }) [hk, lowerReshape] at hBuild
  cases hp : unaryParent? n.parents with
  | none =>
      simp [hp] at hBuild; try cases hBuild
  | some pId =>
          cases hIdx : mkIdx (inShape := inShape) (ss := ss) pId inS with
          | error msg =>
              -- The parent id/shape check fails, so `buildFrom` cannot return `.ok _`.
              simp [Bind.bind, Except.bind, hp, hIdx] at hBuild
          | ok ip =>
              simp [Bind.bind, Except.bind, hp, hIdx] at hBuild
              by_cases hNumel : Spec.Shape.size inS = Spec.Shape.size outS
              ·
                simp [hNumel] at hBuild
                by_cases hOut : outS = n.outShape
                ·
                  simp [hOut] at hBuild
                  let nodeData : ForwardNode α ([inShape] ++ ss) n.outShape :=
                    mkForwardNode (α := α) (Γ := [inShape] ++ ss) (τ := n.outShape) (fun ctx =>
                      let x := getIdx (α := α) (xs := ctx) ip
                      hOut ▸ Tensor.reshapeSpec (α := α) (source := inS) (target := outS) x
                        hNumel)
                  let st1 : State α inShape := ⟨ss ++ [n.outShape], .snoc (ss := ss) gd nodeData⟩
                  have hRec :
                      buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
                        (i := i + 1) st1 = .ok st' := by
                    simpa [st1, nodeData, Pure.pure, Except.pure] using hBuild
                  have hGet :
                      vals0[pId]? =
                        some (Spec.SomeTensor.mk (α := α) inS
                          (getIdx (α := α) (xs := ctx) ip)) := by
                    simpa [vals0, ctx] using
                      (denoteAllState_get_mkIdx? (inShape := inShape) (ss := ss)
                        (gd := gd) (x := x) (pid := pId) (s := inS) (idx := ip) hIdx)
                  have hEval :
                      NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
                          (input := input) (vals := vals0) (i := i) =
                        .ok (Spec.SomeTensor.mk (α := α) n.outShape (nodeData.eval ctx)) := by
                    -- Stage simp so we only normalize the `.reshape` branch (and its one
                    -- `expectShape`).
                    simp [NN.IR.Graph.evalAt, NN.IR.Graph.evalNode,
                      NN.IR.Graph.normalizeNodeOutput, hN, hk, hp, hGet]
                    simp [hNumel, Pure.pure, Except.pure]
                    cases hOut
                    simp [nodeData]
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
                ·
                  simp [hOut] at hBuild
                  try cases hBuild
              ·
                simp [hNumel] at hBuild
                try cases hBuild

/-- Semantic-preservation lemma for `.flatten s` lowering. -/
theorem buildFrom_denoteAllFrom_flatten
    {α : Type} [TorchLean.Storage α] [Context α]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : ForwardData α [inShape] ss) (i : Nat) (st' : State α inShape)
    (x : Tensor α inShape) (n : NN.IR.Node)
    (s : Shape)
    (hN : g.getNode i = .ok n) (hk : n.kind = .flatten s) (hi : i < g.nodes.size)
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
  simp (config := { failIfUnchanged := false }) [hk, lowerFlatten] at hBuild
  cases hp : unaryParent? n.parents with
  | none =>
      simp [hp] at hBuild; try cases hBuild
  | some pId =>
          cases hIdx : mkIdx (inShape := inShape) (ss := ss) pId s with
          | error msg =>
              -- The parent id/shape check fails, so `buildFrom` cannot return `.ok _`.
              simp [Bind.bind, Except.bind, hp, hIdx] at hBuild
          | ok ip =>
              simp [Bind.bind, Except.bind, hp, hIdx] at hBuild
              let expected : Shape := .dim (Spec.Shape.size s) .scalar
              by_cases hOut : expected = n.outShape
              ·
                simp [expected, hOut] at hBuild
                let nodeData : ForwardNode α ([inShape] ++ ss) n.outShape :=
                  mkForwardNode (α := α) (Γ := [inShape] ++ ss) (τ := n.outShape) (fun ctx =>
                    let x := getIdx (α := α) (xs := ctx) ip
                    let y : Tensor α expected := Tensor.flattenSpec (α := α) (shape := s) x
                    hOut ▸ y)
                let st1 : State α inShape := ⟨ss ++ [n.outShape], .snoc (ss := ss) gd nodeData⟩
                have hRec :
                    buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
                      (i := i + 1) st1 = .ok st' := by
                  simpa [st1, nodeData, Pure.pure, Except.pure] using hBuild
                have hGet :
                    vals0[pId]? =
                      some (Spec.SomeTensor.mk (α := α) s
                        (getIdx (α := α) (xs := ctx) ip)) := by
                  simpa [vals0, ctx] using
                    (denoteAllState_get_mkIdx? (inShape := inShape) (ss := ss)
                      (gd := gd) (x := x) (pid := pId) (s := s) (idx := ip) hIdx)
                have hEval :
                    NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
                        (input := input) (vals := vals0) (i := i) =
                      .ok (Spec.SomeTensor.mk (α := α) n.outShape (nodeData.eval ctx)) := by
                  -- Stage simp so we only normalize the `.flatten` branch (and its one
                  -- `expectShape`).
                  simp [NN.IR.Graph.evalAt, NN.IR.Graph.evalNode,
                    NN.IR.Graph.normalizeNodeOutput, hN, hk, hp, hGet]
                  -- `evalAt` performs a final produced-shape check against `n.outShape`.
                  rw [dif_pos hOut]
                  simp [nodeData, Pure.pure, Except.pure]
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
              ·
                simp [expected, hOut] at hBuild
                try cases hBuild

end IRExec
end Autograd
end Runtime
