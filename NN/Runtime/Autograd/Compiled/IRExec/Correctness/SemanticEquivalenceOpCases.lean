/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Compiled.IRExec.Correctness.Common
public import NN.Runtime.Autograd.Compiled.IRExec.Correctness.SemanticEquivalenceCommon

/-!
# Semantic Equivalence (Op Cases)

This module contains semantic-preservation lemmas for IR node kinds handled
inline in the main recursive theorem [`Correctness.SemanticEquivalence`].

Why split these out?

1. **Compilation performance:** `SemanticEquivalence.lean` is a large mutually-dependent proof
   script; extracting the heaviest branches into separate theorems makes elaboration more
   incremental and keeps error messages local to the relevant operator case.
2. **Auditability:** these cases are part of the compiler/denotation contract. Giving them named
   theorems makes it easier to see which IR fragments are covered.

The proofs follow the same pattern as the per-operator modules under `Correctness/Ops/`:

* unfold `buildFrom` and mirror its runtime checks,
* construct the compiled `nodeData` forward closure,
* show that `NN.IR.Graph.evalAt` produces the same dynamic value,
* finish with the shared `buildFrom_denoteAllFrom_finish` lemma for the tail.

Build note: these proofs are slow because each branch normalizes both the compiler and the IR
evaluator, then proves that the resulting dynamic value agrees with a typed compiled node. Shape
casts and `Except` error paths are the main source of proof noise. The local linter scopes in this
file mark the current proof-engineering boundary: the proof is checked, but the simplification
scripts still deserve a pass with more helper lemmas.

More cases should live in `Correctness/Ops/*`, with repeated parent and cast facts packaged as
lemmas from `SemanticEquivalenceCommon`.
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

set_option linter.unnecessarySimpa false in
set_option linter.unusedSimpArgs false in
-- `NN.IR.Graph.evalAt` is a large match; these op-case proofs sometimes require a larger
-- heartbeat budget for `simp` to normalize only the relevant branch.
set_option maxHeartbeats 1200000 in
/-- Semantic-preservation lemma for `.linear` lowering (payload-backed affine map). -/
theorem buildFrom_denoteAllFrom_linear
    {α : Type} [Context α] [DecidableEq Shape]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : GraphData α Unit [inShape] ss) (i : Nat) (st' : State α inShape)
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
      simp [hp] at hBuild; try cases hBuild
  | cons xId ps =>
      cases ps with
      | cons _ _ =>
          simp [hp] at hBuild; try cases hBuild
      | nil =>
          cases hLin : payload.linear? n.id with
          | none =>
              simp [hp, hLin] at hBuild; try cases hBuild
          | some p =>
              simp [hp, hLin] at hBuild
              let expectedIn : Shape := .dim p.inDim .scalar
              let expectedOut : Shape := .dim p.outDim .scalar
              cases hIdx :
                  -- Keep the same syntactic shape argument as the `buildFrom` branch so `simp [hIdx]`
                  -- can fire.
                  mkIdx (inShape := inShape) (ss := ss) xId (.dim p.inDim .scalar) with
              | error msg =>
                  have : False := by
                    -- If the parent id/shape check fails, `buildFrom` returns `.error _`, contradicting `.ok`.
                    simpa [Bind.bind, Except.bind, hIdx] using hBuild
                  cases this
              | ok ix =>
                  simp (config := { failIfUnchanged := false })
                    [Bind.bind, Except.bind, hIdx] at hBuild
                  by_cases hOut : expectedOut = n.outShape
                  ·
                    simp [expectedOut, hOut] at hBuild
                    let nodeData : NodeData α Unit ([inShape] ++ ss) n.outShape :=
                      mkFwdNode (α := α) (Γ := [inShape] ++ ss) (τ := n.outShape) (fun ctx =>
                        let x := getIdx (α := α) (xs := ctx) ix
                        let y : Tensor α expectedOut :=
                          Tensor.addSpec (α := α)
                            (Spec.matVecMulSpec (α := α) (m := p.outDim) (n := p.inDim) p.W x) p.b
                        hOut ▸ y)
                    let st1 : State α inShape := ⟨ss ++ [n.outShape], .snoc (ss := ss) gd nodeData⟩
                    have hRec :
                        buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
                          (i := i + 1) st1 = .ok st' := by
                      simpa [st1, nodeData] using hBuild
                    have hGet :
                        vals0[xId]! =
                          NN.IR.DVal.mk (α := α) expectedIn (getIdx (α := α) (xs := ctx) ix) := by
                      simpa [vals0, ctx, expectedIn] using
                        (denoteAllState_get_mkIdx (inShape := inShape) (ss := ss)
                          (gd := gd) (x := x) (pid := xId) (s := expectedIn) (idx := ix) hIdx)
                    have hEval :
                        NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
                            (input := input) (vals := vals0) (i := i) =
                          .ok (NN.IR.DVal.mk (α := α) n.outShape (nodeData.forward ctx ())) := by
                      -- Split into two stages so `simp` doesn't explore unrelated `evalAt` branches.
                      simp [NN.IR.Graph.evalAt, hN, hk, hp]
                      -- `buildFrom` checks `expectedOut = n.outShape`, but `evalLinear` checks the symmetric
                      -- condition `n.outShape = expectedOut`; record it explicitly so `simp` can reduce the `if`.
                      have hOut' : n.outShape = expectedOut := hOut.symm
                      -- `expectShape` returns a transported `vals0[xId]!.snd`; record that transport once.
                      have hCastX :=
                        dval_snd_cast_of_eq_mk (α := α) (v := vals0[xId]!)
                          (s := expectedIn) (t := getIdx (α := α) (xs := ctx) ix) hGet
                      simp [NN.IR.Graph.evalLinear, hLin, hGet, NN.IR.Graph.expectShape,
                        expectedIn, expectedOut, hOut', nodeData, Pure.pure, Except.pure, hCastX]
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
                  · simp [expectedOut, hOut] at hBuild; try cases hBuild

set_option linter.unnecessarySimpa false in
set_option linter.unusedSimpArgs false in
set_option maxHeartbeats 1200000 in
/-- Semantic-preservation lemma for `.conv2d …` lowering (payload-backed convolution). -/
theorem buildFrom_denoteAllFrom_conv2d
    {α : Type} [Context α] [DecidableEq Shape]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : GraphData α Unit [inShape] ss) (i : Nat) (st' : State α inShape)
    (x : Tensor α inShape) (n : NN.IR.Node)
    (inC outC kH kW stride padding : Nat)
    (hN : g.getNode i = .ok n)
    (hk : n.kind = .conv2d inC outC kH kW stride padding)
    (hi : i < g.nodes.size)
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
      simp [hp] at hBuild; try cases hBuild
  | cons xId ps =>
      cases ps with
      | cons _ _ =>
          simp [hp] at hBuild; try cases hBuild
      | nil =>
          cases hConv : payload.conv2d? n.id with
          | none =>
              simp [hp, hConv] at hBuild; try cases hBuild
          | some cfg =>
              simp [hp, hConv] at hBuild
              let expectedIn : Shape := .dim cfg.inC (.dim cfg.inH (.dim cfg.inW .scalar))
              -- Use the same shape argument as the `buildFrom` branch so `simp [hIdx]` can fire.
              cases hIdx : mkIdx (inShape := inShape) (ss := ss) xId expectedIn with
              | error msg =>
                  -- If the parent id/shape check fails, `buildFrom` returns `.error _`, contradicting `.ok`.
                  have hIdx0 :
                      mkIdx (inShape := inShape) (ss := ss) xId
                          (.dim cfg.inC (.dim cfg.inH (.dim cfg.inW .scalar))) =
                        .error msg := by
                    simpa [expectedIn] using hIdx
                  have hBuild' : Except.error msg = (Except.ok st') := by
                    -- The `mkIdx` bind reduces to `.error _`, so `buildFrom` cannot return `.ok _`.
                    simpa [Bind.bind, Except.bind, hIdx0] using hBuild
                  cases hBuild'
              | ok ix =>
                  have hIdx0 :
                      mkIdx (inShape := inShape) (ss := ss) xId
                          (.dim cfg.inC (.dim cfg.inH (.dim cfg.inW .scalar))) =
                        .ok ix := by
                    simpa [expectedIn] using hIdx
                  simp (config := { failIfUnchanged := false }) [Bind.bind, Except.bind, hIdx0] at hBuild
                  let expected : Shape :=
                    .dim cfg.outC
                      (.dim ((cfg.inH + 2 * cfg.padding - cfg.kH) / cfg.stride + 1)
                        (.dim ((cfg.inW + 2 * cfg.padding - cfg.kW) / cfg.stride + 1) .scalar))
                  by_cases hOut : expected = n.outShape
                  ·
                    simp (config := { failIfUnchanged := false }) [expected, hOut] at hBuild
                    let nodeData : NodeData α Unit ([inShape] ++ ss) n.outShape :=
                      mkFwdNode (α := α) (Γ := [inShape] ++ ss) (τ := n.outShape) (fun ctx =>
                        let x := getIdx (α := α) (xs := ctx) ix
                        let y : Tensor α expected :=
                          Spec.conv2dSpec (α := α) (layer := cfg.spec) (input := x)
                        hOut ▸ y)
                    let st1 : State α inShape := ⟨ss ++ [n.outShape], .snoc (ss := ss) gd nodeData⟩
                    have hRec :
                        buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
                          (i := i + 1) st1 = .ok st' := by
                      simpa [st1, nodeData] using hBuild
                    have hGet :
                        vals0[xId]! =
                          NN.IR.DVal.mk (α := α) expectedIn (getIdx (α := α) (xs := ctx) ix) := by
                      simpa [vals0, ctx, expectedIn] using
                        (denoteAllState_get_mkIdx (inShape := inShape) (ss := ss)
                          (gd := gd) (x := x) (pid := xId) (s := expectedIn) (idx := ix) hIdx)
                    have hEval :
                        NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
                            (input := input) (vals := vals0) (i := i) =
                          .ok (NN.IR.DVal.mk (α := α) n.outShape (nodeData.forward ctx ())) := by
                      -- Stage simplification so we only normalize the `.conv2d` evaluator branch.
                      simp [NN.IR.Graph.evalAt, hN, hk, hp, throw_eq_error]
                      -- Reduce the payload lookup and parent-shape check inside `evalConv2D`.
                      have hCastX :=
                        dval_snd_cast_of_eq_mk (α := α) (v := vals0[xId]!)
                          (s := expectedIn) (t := getIdx (α := α) (xs := ctx) ix) hGet
                      simp [NN.IR.Graph.evalConv2D, hConv, NN.IR.Graph.expectShape, hGet, hCastX]
                      -- After inlining `evalConv2D`, `evalAt` checks the computed `outShape` against the
                      -- declared `n.outShape`. Use `hOut` to collapse the mismatch branch.
                      have hOutBool : (expected != n.outShape) = false :=
                        shape_bne_eq_false_of_eq (s := expected) (t := n.outShape) hOut
                      -- First eliminate the boolean `!=` guard, then normalize the remaining shape cast.
                      simp (config := { failIfUnchanged := false }) [hOutBool]
                      -- `simp` above unfolded the computed shape expression; restate our equalities in that
                      -- form to make the final produced-shape check reduce.
                      have hOut0 :
                          Shape.dim cfg.outC
                              (Shape.dim ((cfg.inH + 2 * cfg.padding - cfg.kH) / cfg.stride + 1)
                                (Shape.dim ((cfg.inW + 2 * cfg.padding - cfg.kW) / cfg.stride + 1)
                                  Shape.scalar)) =
                            n.outShape := by
                        simpa [expected] using hOut
                      have hOutBool0 :
                          (Shape.dim cfg.outC
                                  (Shape.dim ((cfg.inH + 2 * cfg.padding - cfg.kH) / cfg.stride + 1)
                                    (Shape.dim ((cfg.inW + 2 * cfg.padding - cfg.kW) / cfg.stride + 1)
                                      Shape.scalar)) !=
                                n.outShape) =
                              false := by
                        simpa [expected] using hOutBool
                      simp (config := { failIfUnchanged := false }) [hOutBool0]
                      rw [dif_pos hOut0]
                      simp [nodeData, expected]
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
                  ·
                    simp (config := { failIfUnchanged := false }) [expected, hOut] at hBuild
                    try cases hBuild

set_option linter.unnecessarySimpa false in
set_option linter.unusedSimpArgs false in
/-- Semantic-preservation lemma for `.reshape inS outS` lowering. -/
theorem buildFrom_denoteAllFrom_reshape
    {α : Type} [Context α] [DecidableEq Shape]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : GraphData α Unit [inShape] ss) (i : Nat) (st' : State α inShape)
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
      simp [hp] at hBuild; try cases hBuild
  | cons pId ps =>
      cases ps with
      | cons _ _ =>
          simp [hp] at hBuild; try cases hBuild
      | nil =>
          cases hIdx : mkIdx (inShape := inShape) (ss := ss) pId inS with
          | error msg =>
              have hBuild' : Except.error msg = (Except.ok st') := by
                -- The parent id/shape check fails, so `buildFrom` cannot return `.ok _`.
                simpa [Bind.bind, Except.bind, hp, hIdx] using hBuild
              cases hBuild'
          | ok ip =>
              simp [Bind.bind, Except.bind, hp, hIdx] at hBuild
              by_cases hNumel : Shape.size inS = Shape.size outS
              ·
                simp [hNumel] at hBuild
                by_cases hOut : outS = n.outShape
                ·
                  simp [hOut] at hBuild
                  let nodeData : NodeData α Unit ([inShape] ++ ss) n.outShape :=
                    mkFwdNode (α := α) (Γ := [inShape] ++ ss) (τ := n.outShape) (fun ctx =>
                      let x := getIdx (α := α) (xs := ctx) ip
                      hOut ▸ Tensor.reshapeSpec (α := α) (s₁ := inS) (s₂ := outS) x hNumel)
                  let st1 : State α inShape := ⟨ss ++ [n.outShape], .snoc (ss := ss) gd nodeData⟩
                  have hRec :
                      buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
                        (i := i + 1) st1 = .ok st' := by
                    simpa [st1, nodeData] using hBuild
                  have hGet :
                      vals0[pId]! =
                        NN.IR.DVal.mk (α := α) inS (getIdx (α := α) (xs := ctx) ip) := by
                    simpa [vals0, ctx] using
                      (denoteAllState_get_mkIdx (inShape := inShape) (ss := ss)
                        (gd := gd) (x := x) (pid := pId) (s := inS) (idx := ip) hIdx)
                  have hEval :
                      NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
                          (input := input) (vals := vals0) (i := i) =
                        .ok (NN.IR.DVal.mk (α := α) n.outShape (nodeData.forward ctx ())) := by
                    -- Stage simp so we only normalize the `.reshape` branch (and its one `expectShape`).
                    simp [NN.IR.Graph.evalAt, hN, hk, hp]
                    have hCastP :=
                      dval_snd_cast_of_eq_mk (α := α) (v := vals0[pId]!)
                        (s := inS) (t := getIdx (α := α) (xs := ctx) ip) hGet
                    simp [hGet, NN.IR.Graph.expectShape, hCastP, hNumel, Pure.pure, Except.pure]
                    cases hOut
                    simp [nodeData, NN.IR.DVal.mk, Pure.pure, Except.pure]
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
                ·
                  simp [hOut] at hBuild
                  try cases hBuild
              ·
                simp [hNumel] at hBuild
                try cases hBuild

set_option linter.unnecessarySimpa false in
set_option linter.unusedSimpArgs false in
/-- Semantic-preservation lemma for `.flatten s` lowering. -/
theorem buildFrom_denoteAllFrom_flatten
    {α : Type} [Context α] [DecidableEq Shape]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : GraphData α Unit [inShape] ss) (i : Nat) (st' : State α inShape)
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
      simp [hp] at hBuild; try cases hBuild
  | cons pId ps =>
      cases ps with
      | cons _ _ =>
          simp [hp] at hBuild; try cases hBuild
      | nil =>
          cases hIdx : mkIdx (inShape := inShape) (ss := ss) pId s with
          | error msg =>
              have hBuild' : Except.error msg = (Except.ok st') := by
                -- The parent id/shape check fails, so `buildFrom` cannot return `.ok _`.
                simpa [Bind.bind, Except.bind, hp, hIdx] using hBuild
              cases hBuild'
          | ok ip =>
              simp [Bind.bind, Except.bind, hp, hIdx] at hBuild
              let expected : Shape := .dim (Shape.size s) .scalar
              by_cases hOut : expected = n.outShape
              ·
                simp [expected, hOut] at hBuild
                let nodeData : NodeData α Unit ([inShape] ++ ss) n.outShape :=
                  mkFwdNode (α := α) (Γ := [inShape] ++ ss) (τ := n.outShape) (fun ctx =>
                    let x := getIdx (α := α) (xs := ctx) ip
                    let y : Tensor α expected := Tensor.flattenSpec (α := α) (s := s) x
                    hOut ▸ y)
                let st1 : State α inShape := ⟨ss ++ [n.outShape], .snoc (ss := ss) gd nodeData⟩
                have hRec :
                    buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
                      (i := i + 1) st1 = .ok st' := by
                  simpa [st1, nodeData] using hBuild
                have hGet :
                    vals0[pId]! =
                      NN.IR.DVal.mk (α := α) s (getIdx (α := α) (xs := ctx) ip) := by
                  simpa [vals0, ctx] using
                    (denoteAllState_get_mkIdx (inShape := inShape) (ss := ss)
                      (gd := gd) (x := x) (pid := pId) (s := s) (idx := ip) hIdx)
                have hEval :
                    NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
                        (input := input) (vals := vals0) (i := i) =
                      .ok (NN.IR.DVal.mk (α := α) n.outShape (nodeData.forward ctx ())) := by
                  -- Stage simp so we only normalize the `.flatten` branch (and its one `expectShape`).
                  simp [NN.IR.Graph.evalAt, hN, hk, hp]
                  have hCastP :=
                    dval_snd_cast_of_eq_mk (α := α) (v := vals0[pId]!)
                      (s := s) (t := getIdx (α := α) (xs := ctx) ip) hGet
                  simp [hGet, NN.IR.Graph.expectShape, hCastP, expected]
                  -- `evalAt` performs a final produced-shape check against `n.outShape`.
                  rw [dif_pos hOut]
                  simp [nodeData, NN.IR.DVal.mk, Pure.pure, Except.pure]
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
              ·
                simp [expected, hOut] at hBuild
                try cases hBuild

set_option linter.unnecessarySimpa false in
set_option linter.unusedSimpArgs false in
/-- The `.concat axis` IR node is not compiled by `buildFrom`, so successful compilation is impossible. -/
theorem buildFrom_denoteAllFrom_concat_impossible
    {α : Type} [Context α] [DecidableEq Shape]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : GraphData α Unit [inShape] ss) (i : Nat) (st' : State α inShape)
    (x : Tensor α inShape) (n : NN.IR.Node)
    (axis : Nat)
    (hN : g.getNode i = .ok n) (hk : n.kind = .concat axis) (hi : i < g.nodes.size)
    (hBuild :
      buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
        (i := i) (st := (⟨ss, gd⟩ : State α inShape)) = .ok st') :
    NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
      (input := NN.IR.DVal.mk (α := α) inShape x)
      (i := i) (vals := denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x) =
      .ok (denoteAllState (α := α) inShape st' x) := by
  -- `buildFrom` does not implement `.concat`, so the successful compilation hypothesis is contradictory.
  unfold buildFrom at hBuild
  -- The `.concat` branch reduces to `throw`, so it cannot return `.ok st'`.
  have hBuild' := hBuild
  simp [hi, hN, hk, throw_eq_error, Except.instMonad, Except.bind, Except.pure] at hBuild'
  cases hBuild'

set_option linter.unnecessarySimpa false in
set_option linter.unusedSimpArgs false in
/-- Semantic-preservation lemma for `.swap_first_two` lowering. -/
theorem buildFrom_denoteAllFrom_swap_first_two
    {α : Type} [Context α] [DecidableEq Shape]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : GraphData α Unit [inShape] ss) (i : Nat) (st' : State α inShape)
    (x : Tensor α inShape) (n : NN.IR.Node)
    (hN : g.getNode i = .ok n) (hk : n.kind = .swap_first_two) (hi : i < g.nodes.size)
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
      simp [hp] at hBuild; try cases hBuild
  | cons pId ps =>
      cases ps with
      | cons _ _ =>
          simp [hp] at hBuild; try cases hBuild
      | nil =>
          -- Reduce the parent-list match inside `buildFrom` now that we are in the 1-parent branch.
          simp [hp] at hBuild
          cases hτ : n.outShape with
          | scalar =>
              -- The compiler rejects rank-0 `outShape`, so successful compilation is impossible.
              have : False := by
                simpa [hτ, throw_eq_error, Except.instMonad, Except.bind, Except.pure] using hBuild
              exact False.elim this
          | dim nDim sTail =>
              cases sTail with
              | scalar =>
                  -- The compiler rejects rank-1 `outShape`, so successful compilation is impossible.
                  have : False := by
                    simpa [hτ, throw_eq_error, Except.instMonad, Except.bind, Except.pure] using hBuild
                  exact False.elim this
              | dim m rest =>
                  -- Rewrite `n.outShape` to a constructor so the `buildFrom`-internal `match hτ : n.outShape` reduces.
                  rw [hτ] at hBuild
                  simp (config := { failIfUnchanged := false }) at hBuild
                  let expectedIn : Shape := .dim m (.dim nDim rest)
                  cases hIdx : mkIdx (inShape := inShape) (ss := ss) pId expectedIn with
                  | error msg =>
                      -- If the parent id/shape check fails, `buildFrom` returns `.error _`, contradicting `.ok`.
                      have : False := by
                        have hBuild' := hBuild
                        simp [Bind.bind, Except.bind, hIdx] at hBuild'
                        cases hBuild'
                      exact False.elim this
                  | ok ip =>
                      simp [Bind.bind, Except.bind, hIdx] at hBuild
                      let nodeData : NodeData α Unit ([inShape] ++ ss) n.outShape :=
                        mkFwdNode (α := α) (Γ := [inShape] ++ ss) (τ := n.outShape) (fun ctx =>
                          let y : Tensor α (.dim nDim (.dim m rest)) :=
                            Tensor.swapFirstTwoSpec (α := α) (m := m) (n := nDim) (s := rest)
                              (getIdx (α := α) (xs := ctx) ip)
                          Tensor.castShape y hτ.symm)
                      let st1 : State α inShape := ⟨ss ++ [n.outShape], .snoc (ss := ss) gd nodeData⟩
                      have hRec :
                          buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
                            (i := i + 1) st1 = .ok st' := by
                        simpa [st1, nodeData] using hBuild
                      have hGet :
                          vals0[pId]! =
                            NN.IR.DVal.mk (α := α) expectedIn (getIdx (α := α) (xs := ctx) ip) := by
                        simpa [vals0, ctx, expectedIn] using
                          (denoteAllState_get_mkIdx (inShape := inShape) (ss := ss)
                            (gd := gd) (x := x) (pid := pId) (s := expectedIn) (idx := ip) hIdx)
                      have hEval :
                          NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
                              (input := input) (vals := vals0) (i := i) =
                            .ok (NN.IR.DVal.mk (α := α) n.outShape (nodeData.forward ctx ())) := by
                        -- Stage the simp calls to keep normalization localized to the `.swap_first_two` branch.
                        simp [NN.IR.Graph.evalAt, hN, hk, hp]
                        have hCastP :=
                          dval_snd_cast_of_eq_mk (α := α) (v := vals0[pId]!)
                            (s := expectedIn) (t := getIdx (α := α) (xs := ctx) ip) hGet
                        simp [hτ, hGet, NN.IR.Graph.expectShape, expectedIn, nodeData, Pure.pure, Except.pure,
                          hCastP, Tensor.eqRec_eq_cast_shape]
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

set_option linter.unnecessarySimpa false in
set_option linter.unusedSimpArgs false in
set_option maxHeartbeats 1200000 in
/-- Semantic-preservation lemma for `.transpose3dLastTwo` lowering. -/
theorem buildFrom_denoteAllFrom_transpose3dLastTwo
    {α : Type} [Context α] [DecidableEq Shape]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : GraphData α Unit [inShape] ss) (i : Nat) (st' : State α inShape)
    (x : Tensor α inShape) (n : NN.IR.Node)
    (hN : g.getNode i = .ok n) (hk : n.kind = .transpose3dLastTwo) (hi : i < g.nodes.size)
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
      simp [hp] at hBuild; try cases hBuild
  | cons pId ps =>
      cases ps with
      | cons _ _ =>
          simp [hp] at hBuild; try cases hBuild
      | nil =>
          -- Reduce the parent-list match inside `buildFrom` now that we are in the 1-parent branch.
          simp [hp] at hBuild
          cases hτ : n.outShape with
          | scalar =>
              have : False := by
                simpa [hτ, throw_eq_error, Except.instMonad, Except.bind, Except.pure] using hBuild
              exact False.elim this
          | dim a sTail =>
              cases sTail with
              | scalar =>
                  have : False := by
                    simpa [hτ, throw_eq_error, Except.instMonad, Except.bind, Except.pure] using hBuild
                  exact False.elim this
              | dim c sTail2 =>
                  cases sTail2 with
                  | scalar =>
                      have : False := by
                        simpa [hτ, throw_eq_error, Except.instMonad, Except.bind, Except.pure] using hBuild
                      exact False.elim this
                  | dim b sTail3 =>
                      cases sTail3 with
                      | dim _ _ =>
                          have : False := by
                            simpa [hτ, throw_eq_error, Except.instMonad, Except.bind, Except.pure] using hBuild
                          exact False.elim this
                      | scalar =>
                          simp (config := { failIfUnchanged := false }) [hτ] at hBuild
                          let expectedIn : Shape := .dim a (.dim b (.dim c .scalar))
                          cases hIdx : mkIdx (inShape := inShape) (ss := ss) pId expectedIn with
                          | error msg =>
                              -- If the parent id/shape check fails, `buildFrom` returns `.error _`, contradicting `.ok`.
                              have : False := by
                                have hBuild' := hBuild
                                simp [Bind.bind, Except.bind, hIdx] at hBuild'
                                cases hBuild'
                              exact False.elim this
                          | ok ip =>
                              simp [Bind.bind, Except.bind, hIdx] at hBuild
                              let nodeData : NodeData α Unit ([inShape] ++ ss) n.outShape :=
                                mkFwdNode (α := α) (Γ := [inShape] ++ ss) (τ := n.outShape) (fun ctx =>
                                  let y : Tensor α (.dim a (.dim c (.dim b .scalar))) :=
                                    Tensor.transpose3DLastTwoSpec (α := α) (a := a) (b := b) (c := c)
                                      (getIdx (α := α) (xs := ctx) ip)
                                  Tensor.castShape y hτ.symm)
                              let st1 : State α inShape :=
                                ⟨ss ++ [n.outShape], .snoc (ss := ss) gd nodeData⟩
                              have hRec :
                                  buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
                                      (i := i + 1) st1 = .ok st' := by
                                simpa [st1, nodeData] using hBuild
                              have hGet :
                                  vals0[pId]! =
                                    NN.IR.DVal.mk (α := α) expectedIn (getIdx (α := α) (xs := ctx) ip) := by
                                simpa [vals0, ctx, expectedIn] using
                                  (denoteAllState_get_mkIdx (inShape := inShape) (ss := ss)
                                    (gd := gd) (x := x) (pid := pId) (s := expectedIn) (idx := ip) hIdx)
                              have hEval :
                                  NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
                                      (input := input) (vals := vals0) (i := i) =
                                    .ok (NN.IR.DVal.mk (α := α) n.outShape (nodeData.forward ctx ())) := by
                                -- Stage simp so we only normalize the `.transpose3dLastTwo` branch.
                                simp [NN.IR.Graph.evalAt, hN, hk, hp]
                                have hCastP :=
                                  dval_snd_cast_of_eq_mk (α := α) (v := vals0[pId]!)
                                    (s := expectedIn) (t := getIdx (α := α) (xs := ctx) ip) hGet
                                simp [hτ, throw_eq_error, hGet, expectedIn, NN.IR.Graph.expectShape,
                                  nodeData, Pure.pure, Except.pure, hCastP, Tensor.eqRec_eq_cast_shape]
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
