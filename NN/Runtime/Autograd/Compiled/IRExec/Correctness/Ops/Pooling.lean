/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Compiled.IRExec.Correctness.Common

/-!
# Pooling

Pooling correctness lemmas for the IR → compiled runtime bridge.

TorchLean models pooling on rank-3 image tensors `C × H × W` (and batched variants upstream in the
runtime), matching the usual PyTorch max-pooling convention:
`torch.nn.functional.max_pool2d` / `torch.nn.MaxPool2d`.

This file proves the forward-correctness statement for the compiler path that lowers pooling IR
nodes into a single SSA node whose `forward` computes the corresponding spec-level pooling
operation. Concretely, successful compilation at graph index
`i` implies that the IR evaluator `NN.IR.Graph.denoteAllFrom` and the compiled evaluator
`denoteAllState` stay in semantic equivalence.

References:

* PyTorch functional max-pool docs:
  <https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.max_pool2d>
* PyTorch module max-pool docs:
  <https://docs.pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html>

## Main definitions

- `buildFrom_denoteAllFrom_max_pool2d`: correctness step for unpadded max-pool lowering.
- `buildFrom_denoteAllFrom_max_pool2d_pad`: correctness step for padded max-pool lowering.
- `buildFrom_denoteAllFrom_avg_pool2d`: correctness step for unpadded average-pool lowering.
- `buildFrom_denoteAllFrom_avg_pool2d_pad`: correctness step for padded average-pool lowering.

## Implementation notes

- The proof follows compiler control flow closely (shape checks, parent checks, guard conditions);
  this one-to-one structure keeps maintenance direct when lowering rules evolve.
- Impossible branches are discharged early via `throw_bind_ne_ok`, which keeps the success path
  readable.
- Pooling proofs build slowly because the output height and width are computed from kernel, stride,
  and padding parameters, then reflected in dependent tensor shapes. Focused helper lemmas should isolate
  those output-shape arithmetic facts so the semantic proof only compares the compiled pooling call
  with the IR pooling evaluator.

## Tags

pool2d, correctness, ir, runtime, semantic equivalence
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

/-- Correctness lemma for the `.max_pool2d` node compiler (no padding). -/
theorem buildFrom_denoteAllFrom_max_pool2d
    {α : Type} [Context α] [DecidableEq Shape]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : GraphData α Unit [inShape] ss) (i : Nat) (st' : State α inShape)
    (x : Tensor α inShape) (n : NN.IR.Node)
    (kH kW stride : Nat)
    (hN : g.getNode i = .ok n) (hk : n.kind = .maxPool2d kH kW stride) (hi : i < g.nodes.size)
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
      exact False.elim <| throw_bind_ne_ok (by simpa [hp] using hBuild)
  | cons pId ps =>
      cases ps with
      | cons _ _ =>
          exact False.elim <| throw_bind_ne_ok (by simpa [hp] using hBuild)
      | nil =>
          cases hP : g.getNode pId with
          | error msg =>
              have hFalse : False := by
                simp [hp, hP] at hBuild
              cases hFalse
          | ok pNode =>
              simp [hp, hP] at hBuild
              cases hS : pNode.outShape with
              | scalar =>
                  exact False.elim <| throw_bind_ne_ok (by simpa [hS] using hBuild)
              | dim inC sTail =>
                  cases sTail with
                  | scalar =>
                      exact False.elim <| throw_bind_ne_ok (by simpa [hS] using hBuild)
                  | dim inH sTail2 =>
                      cases sTail2 with
                      | scalar =>
                          exact False.elim <| throw_bind_ne_ok (by simpa [hS] using hBuild)
                      | dim inW sTail3 =>
                          cases sTail3 with
                          | dim _ _ =>
                              exact False.elim <| throw_bind_ne_ok (by simpa [hS] using hBuild)
                          | scalar =>
                              by_cases hkH0 : kH = 0
                              · exact False.elim <| throw_bind_ne_ok (by simpa [hS, hkH0] using
                                hBuild)
                              · by_cases hkW0 : kW = 0
                                · exact False.elim <| throw_bind_ne_ok (by simpa [hS, hkH0, hkW0]
                                  using hBuild)
                                · by_cases hs : stride = 0
                                  · exact False.elim <| throw_bind_ne_ok (by simpa [hS, hkH0, hkW0, hs]
                                    using hBuild)
                                  ·
                                    let sIn : Shape := .dim inC (.dim inH (.dim inW .scalar))
                                    cases hIdx :
                                        mkIdx (inShape := inShape) (ss := ss) pId sIn with
                                    | error msg =>
                                        have hFalse : False := by
                                          simp [hS, hkH0, hkW0, hs, sIn, hIdx] at hBuild
                                        cases hFalse
                                    | ok ip =>
                                        simp [hS, hkH0, hkW0, hs, sIn, hIdx] at hBuild
                                        let expected : Shape :=
                                          Spec.pool2dMultiOutShape inC inH inW kH kW stride
                                        by_cases hOut : expected = n.outShape
                                        · simp [expected, hOut] at hBuild
                                          let layer : Spec.MaxPool2DSpec kH kW stride hkH0 hkW0 hs := {}
                                          let nodeData : NodeData α Unit ([inShape] ++ ss) n.outShape
                                            :=
                                            mkFwdNode (α := α) (Γ := [inShape] ++ ss) (τ :=
                                              n.outShape) (fun ctx =>
                                              let xCHW := getIdx (α := α) (xs := ctx) ip
                                              let y : Tensor α expected :=
                                                Spec.maxPool2dMultiSpec (α := α) (kH := kH) (kW :=
                                                  kW)
                                                  (inH := inH) (inW := inW) (inC := inC) (stride :=
                                                    stride)
                                                  (layer := layer) (input := xCHW)
                                              hOut ▸ y)
                                          let st1 : State α inShape :=
                                            ⟨ss ++ [n.outShape], .snoc (ss := ss) gd nodeData⟩
                                          have hRec :
                                              buildFrom (α := α) (g := g) (payload := payload)
                                                (inShape := inShape)
                                                  (i := i + 1) st1 = .ok st' := by
                                            simp [st1, nodeData]
                                            exact hBuild
                                          have hGet :
                                              vals0[pId]! =
                                                NN.IR.DVal.mk (α := α) sIn
                                                  (getIdx (α := α) (xs := ctx) ip) := by
                                            simpa [vals0, ctx, sIn] using
                                              (denoteAllState_get_mkIdx (inShape := inShape) (ss :=
                                                ss)
                                                (gd := gd) (x := x) (pid := pId) (s := sIn) (idx :=
                                                  ip) hIdx)
                                          have hEval :
                                              NN.IR.Graph.evalAt (α := α) (g := g) (payload :=
                                                payload)
                                                  (input := input) (vals := vals0) (i := i) =
                                                .ok (NN.IR.DVal.mk (α := α) n.outShape
                                                  (nodeData.forward ctx ())) := by
                                            have hShapeV : vals0[pId]!.shape = sIn := by
                                              simpa [NN.IR.DVal.shape, NN.IR.DVal.mk] using
                                                congrArg (fun v => NN.IR.DVal.shape (α := α) v) hGet
                                            have hFst : vals0[pId]!.fst = sIn := by
                                              -- `DVal.shape` is definitional `Sigma.fst`, but
                                              -- `evalAt` matches on `.fst`.
                                              simpa [NN.IR.DVal.shape] using hShapeV
                                            have hExpSIn :
                                                NN.IR.Graph.expectShape (α := α) (expected := sIn)
                                                  vals0[pId]! =
                                                  .ok (getIdx (α := α) (xs := ctx) ip) := by
                                              rw [hGet]
                                              exact (Graph.expectShape_sigma (α := α) (s := sIn)
                                                (t := getIdx (α := α) (xs := ctx) ip))
                                            have hExp :
                                                NN.IR.Graph.expectShape (α := α)
                                                    (expected := Shape.dim inC (Shape.dim inH
                                                      (Shape.dim inW Shape.scalar)))
                                                    vals0[pId]! =
                                                  .ok (getIdx (α := α) (xs := ctx) ip) := by
                                              simpa [sIn] using hExpSIn
                                            simp [NN.IR.Graph.evalAt, hN, hk, hp, hExp,
                                              nodeData, layer,
                                              hFst, sIn, expected, hkH0, hkW0, hs, hOut]
                                          have hStep :
                                              denoteAllState (α := α) inShape st1 x =
                                                vals0.push (NN.IR.DVal.mk (α := α) n.outShape
                                                  (nodeData.forward ctx ())) := by
                                            simpa [vals0, st1, nodeData, ctx] using
                                              (denoteAllState_snoc (α := α) (inShape := inShape) (ss
                                                := ss) (τ := n.outShape)
                                                (gd := gd) (nodeData := nodeData) (x := x))
                                          have hTail := ih st1 hRec
                                          exact buildFrom_denoteAllFrom_finish (α := α) (g := g)
                                            (payload := payload)
                                            (i := i) (x := x) (hi := hi) (τ := n.outShape)
                                            (nodeData := nodeData) (st1 := st1) (st' := st')
                                            (ctx := ctx) (vals0 := vals0) (input := input) hTail hEval
                                              hStep
                                        · exact False.elim <|
                                            throw_bind_ne_ok (by simpa [expected, hOut, hs] using hBuild)

/-- Correctness lemma for the `.max_pool2d_pad` node compiler (explicit padding). -/
theorem buildFrom_denoteAllFrom_max_pool2d_pad
    {α : Type} [Context α] [DecidableEq Shape]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : GraphData α Unit [inShape] ss) (i : Nat) (st' : State α inShape)
    (x : Tensor α inShape) (n : NN.IR.Node)
    (kH kW stride padding : Nat)
    (hN : g.getNode i = .ok n) (hk : n.kind = .maxPool2dPad kH kW stride padding) (hi : i <
      g.nodes.size)
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
      exact False.elim <| throw_bind_ne_ok (by simpa [hp] using hBuild)
  | cons pId ps =>
      cases ps with
      | cons _ _ =>
          exact False.elim <| throw_bind_ne_ok (by simpa [hp] using hBuild)
      | nil =>
          cases hP : g.getNode pId with
          | error msg =>
              have hFalse : False := by
                simp [hp, hP] at hBuild
              cases hFalse
          | ok pNode =>
              simp [hp, hP] at hBuild
              cases hS : pNode.outShape with
              | scalar =>
                  exact False.elim <| throw_bind_ne_ok (by simpa [hS] using hBuild)
              | dim inC sTail =>
                  cases sTail with
                  | scalar =>
                      exact False.elim <| throw_bind_ne_ok (by simpa [hS] using hBuild)
                  | dim inH sTail2 =>
                      cases sTail2 with
                      | scalar =>
                          exact False.elim <| throw_bind_ne_ok (by simpa [hS] using hBuild)
                      | dim inW sTail3 =>
                          cases sTail3 with
                          | dim _ _ =>
                              exact False.elim <| throw_bind_ne_ok (by simpa [hS] using hBuild)
                          | scalar =>
                              by_cases hkH0 : kH = 0
                              · exact False.elim <| throw_bind_ne_ok (by simpa [hS, hkH0] using
                                hBuild)
                              · by_cases hkW0 : kW = 0
                                · exact False.elim <| throw_bind_ne_ok (by simpa [hS, hkH0, hkW0]
                                  using hBuild)
                                · by_cases hs : stride = 0
                                  · exact False.elim <| throw_bind_ne_ok (by simpa [hS, hkH0, hkW0, hs]
                                    using hBuild)
                                  ·
                                    let sIn : Shape := .dim inC (.dim inH (.dim inW .scalar))
                                    cases hIdx :
                                        mkIdx (inShape := inShape) (ss := ss) pId sIn with
                                    | error msg =>
                                        have hFalse : False := by
                                          simp [hS, hkH0, hkW0, hs, sIn, hIdx] at hBuild
                                        cases hFalse
                                    | ok ip =>
                                        simp [hS, hkH0, hkW0, hs, sIn, hIdx] at hBuild
                                        let expected : Shape :=
                                          Spec.pool2dMultiOutShapePad inC inH inW kH kW stride
                                            padding
                                        by_cases hOut : expected = n.outShape
                                        · simp [expected, hOut] at hBuild
                                          let layer : Spec.MaxPool2DSpec kH kW stride hkH0 hkW0 hs := {}
                                          let nodeData : NodeData α Unit ([inShape] ++ ss) n.outShape
                                            :=
                                            mkFwdNode (α := α) (Γ := [inShape] ++ ss) (τ :=
                                              n.outShape) (fun ctx =>
                                              let xCHW := getIdx (α := α) (xs := ctx) ip
                                              let y : Tensor α expected :=
                                                Spec.maxPool2dMultiSpecPad (α := α) (kH := kH) (kW
                                                  := kW)
                                                  (inH := inH) (inW := inW) (inC := inC) (stride :=
                                                    stride)
                                                  (padding := padding) (layer := layer) (input :=
                                                    xCHW)
                                              hOut ▸ y)
                                          let st1 : State α inShape :=
                                            ⟨ss ++ [n.outShape], .snoc (ss := ss) gd nodeData⟩
                                          have hRec :
                                              buildFrom (α := α) (g := g) (payload := payload)
                                                (inShape := inShape)
                                                  (i := i + 1) st1 = .ok st' := by
                                            simp [st1, nodeData]
                                            exact hBuild
                                          have hGet :
                                              vals0[pId]! =
                                                NN.IR.DVal.mk (α := α) sIn
                                                  (getIdx (α := α) (xs := ctx) ip) := by
                                            simpa [vals0, ctx, sIn] using
                                              (denoteAllState_get_mkIdx (inShape := inShape) (ss :=
                                                ss)
                                                (gd := gd) (x := x) (pid := pId) (s := sIn) (idx :=
                                                  ip) hIdx)
                                          have hEval :
                                              NN.IR.Graph.evalAt (α := α) (g := g) (payload :=
                                                payload)
                                                  (input := input) (vals := vals0) (i := i) =
                                                .ok (NN.IR.DVal.mk (α := α) n.outShape
                                                  (nodeData.forward ctx ())) := by
                                            have hShapeV : vals0[pId]!.shape = sIn := by
                                              simpa [NN.IR.DVal.shape, NN.IR.DVal.mk] using
                                                congrArg (fun v => NN.IR.DVal.shape (α := α) v) hGet
                                            have hFst : vals0[pId]!.fst = sIn := by
                                              simpa [NN.IR.DVal.shape] using hShapeV
                                            have hExpSIn :
                                                NN.IR.Graph.expectShape (α := α) (expected := sIn)
                                                  vals0[pId]! =
                                                  .ok (getIdx (α := α) (xs := ctx) ip) := by
                                              rw [hGet]
                                              exact (Graph.expectShape_sigma (α := α) (s := sIn)
                                                (t := getIdx (α := α) (xs := ctx) ip))
                                            have hExp :
                                                NN.IR.Graph.expectShape (α := α)
                                                    (expected := Shape.dim inC (Shape.dim inH
                                                      (Shape.dim inW Shape.scalar)))
                                                    vals0[pId]! =
                                                  .ok (getIdx (α := α) (xs := ctx) ip) := by
                                              simpa [sIn] using hExpSIn
                                            simp [NN.IR.Graph.evalAt, hN, hk, hp, hExp,
                                              nodeData, layer,
                                              hFst, sIn, expected, hkH0, hkW0, hs, hOut]
                                          have hStep :
                                              denoteAllState (α := α) inShape st1 x =
                                                vals0.push (NN.IR.DVal.mk (α := α) n.outShape
                                                  (nodeData.forward ctx ())) := by
                                            simpa [vals0, st1, nodeData, ctx] using
                                              (denoteAllState_snoc (α := α) (inShape := inShape) (ss
                                                := ss) (τ := n.outShape)
                                                (gd := gd) (nodeData := nodeData) (x := x))
                                          have hTail := ih st1 hRec
                                          exact buildFrom_denoteAllFrom_finish (α := α) (g := g)
                                            (payload := payload)
                                            (i := i) (x := x) (hi := hi) (τ := n.outShape)
                                            (nodeData := nodeData) (st1 := st1) (st' := st')
                                            (ctx := ctx) (vals0 := vals0) (input := input) hTail hEval
                                              hStep
                                        · exact False.elim <|
                                            throw_bind_ne_ok (by simpa [expected, hOut, hs] using hBuild)

/-- Correctness lemma for the `.avg_pool2d` node compiler (no padding). -/
theorem buildFrom_denoteAllFrom_avg_pool2d
    {α : Type} [Context α] [DecidableEq Shape]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : GraphData α Unit [inShape] ss) (i : Nat) (st' : State α inShape)
    (x : Tensor α inShape) (n : NN.IR.Node)
    (kH kW stride : Nat)
    (hN : g.getNode i = .ok n) (hk : n.kind = .avgPool2d kH kW stride) (hi : i < g.nodes.size)
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
      exact False.elim <| throw_bind_ne_ok (by simpa [hp] using hBuild)
  | cons pId ps =>
      cases ps with
      | cons _ _ =>
          exact False.elim <| throw_bind_ne_ok (by simpa [hp] using hBuild)
      | nil =>
          cases hP : g.getNode pId with
          | error msg =>
              have hFalse : False := by
                simp [hp, hP] at hBuild
              cases hFalse
          | ok pNode =>
              simp [hp, hP] at hBuild
              cases hS : pNode.outShape with
              | scalar =>
                  exact False.elim <| throw_bind_ne_ok (by simpa [hS] using hBuild)
              | dim inC sTail =>
                  cases sTail with
                  | scalar =>
                      exact False.elim <| throw_bind_ne_ok (by simpa [hS] using hBuild)
                  | dim inH sTail2 =>
                      cases sTail2 with
                      | scalar =>
                          exact False.elim <| throw_bind_ne_ok (by simpa [hS] using hBuild)
                      | dim inW sTail3 =>
                          cases sTail3 with
                          | dim _ _ =>
                              exact False.elim <| throw_bind_ne_ok (by simpa [hS] using hBuild)
                          | scalar =>
                              by_cases hkH0 : kH = 0
                              · exact False.elim <| throw_bind_ne_ok (by simpa [hS, hkH0] using
                                hBuild)
                              · by_cases hkW0 : kW = 0
                                · exact False.elim <| throw_bind_ne_ok (by simpa [hS, hkH0, hkW0]
                                  using hBuild)
                                · by_cases hs : stride = 0
                                  · exact False.elim <| throw_bind_ne_ok (by simpa [hS, hkH0, hkW0, hs]
                                    using hBuild)
                                  ·
                                    let sIn : Shape := .dim inC (.dim inH (.dim inW .scalar))
                                    cases hIdx :
                                        mkIdx (inShape := inShape) (ss := ss) pId sIn with
                                    | error msg =>
                                        have hFalse : False := by
                                          simp [hS, hkH0, hkW0, hs, sIn, hIdx] at hBuild
                                        cases hFalse
                                    | ok ip =>
                                        simp [hS, hkH0, hkW0, hs, sIn, hIdx] at hBuild
                                        let expected : Shape :=
                                          Spec.pool2dMultiOutShape inC inH inW kH kW stride
                                        by_cases hOut : expected = n.outShape
                                        · simp [expected, hOut] at hBuild
                                          let layer : Spec.AvgPool2DSpec kH kW stride hkH0 hkW0 hs := {}
                                          let nodeData : NodeData α Unit ([inShape] ++ ss) n.outShape
                                            :=
                                            mkFwdNode (α := α) (Γ := [inShape] ++ ss) (τ :=
                                              n.outShape) (fun ctx =>
                                              let xCHW := getIdx (α := α) (xs := ctx) ip
                                              let y : Tensor α expected :=
                                                Spec.avgPool2dMultiSpec (α := α) (kH := kH) (kW :=
                                                  kW)
                                                  (inH := inH) (inW := inW) (inC := inC) (stride :=
                                                    stride)
                                                  (h1 := hkH0) (h2 := hkW0) (layer := layer) (input :=
                                                    xCHW)
                                              hOut ▸ y)
                                          let st1 : State α inShape :=
                                            ⟨ss ++ [n.outShape], .snoc (ss := ss) gd nodeData⟩
                                          have hRec :
                                              buildFrom (α := α) (g := g) (payload := payload)
                                                (inShape := inShape)
                                                  (i := i + 1) st1 = .ok st' := by
                                            simp [st1, nodeData]
                                            exact hBuild
                                          have hGet :
                                              vals0[pId]! =
                                                NN.IR.DVal.mk (α := α) sIn
                                                  (getIdx (α := α) (xs := ctx) ip) := by
                                            simpa [vals0, ctx, sIn] using
                                              (denoteAllState_get_mkIdx (inShape := inShape) (ss :=
                                                ss)
                                                (gd := gd) (x := x) (pid := pId) (s := sIn) (idx :=
                                                  ip) hIdx)
                                          have hEval :
                                              NN.IR.Graph.evalAt (α := α) (g := g) (payload :=
                                                payload)
                                                  (input := input) (vals := vals0) (i := i) =
                                                .ok (NN.IR.DVal.mk (α := α) n.outShape
                                                  (nodeData.forward ctx ())) := by
                                            have hShapeV : vals0[pId]!.shape = sIn := by
                                              simpa [NN.IR.DVal.shape, NN.IR.DVal.mk] using
                                                congrArg (fun v => NN.IR.DVal.shape (α := α) v) hGet
                                            have hFst : vals0[pId]!.fst = sIn := by
                                              simpa [NN.IR.DVal.shape] using hShapeV
                                            have hExpSIn :
                                                NN.IR.Graph.expectShape (α := α) (expected := sIn)
                                                  vals0[pId]! =
                                                  .ok (getIdx (α := α) (xs := ctx) ip) := by
                                              rw [hGet]
                                              exact (Graph.expectShape_sigma (α := α) (s := sIn)
                                                (t := getIdx (α := α) (xs := ctx) ip))
                                            have hExp :
                                                NN.IR.Graph.expectShape (α := α)
                                                    (expected := Shape.dim inC (Shape.dim inH
                                                      (Shape.dim inW Shape.scalar)))
                                                    vals0[pId]! =
                                                  .ok (getIdx (α := α) (xs := ctx) ip) := by
                                              simpa [sIn] using hExpSIn
                                            simp [NN.IR.Graph.evalAt, hN, hk, hp, hExp,
                                              nodeData, layer,
                                              hFst, sIn, expected, hkH0, hkW0, hs, hOut]
                                          have hStep :
                                              denoteAllState (α := α) inShape st1 x =
                                                vals0.push (NN.IR.DVal.mk (α := α) n.outShape
                                                  (nodeData.forward ctx ())) := by
                                            simpa [vals0, st1, nodeData, ctx] using
                                              (denoteAllState_snoc (α := α) (inShape := inShape) (ss
                                                := ss) (τ := n.outShape)
                                                (gd := gd) (nodeData := nodeData) (x := x))
                                          have hTail := ih st1 hRec
                                          exact buildFrom_denoteAllFrom_finish (α := α) (g := g)
                                            (payload := payload)
                                            (i := i) (x := x) (hi := hi) (τ := n.outShape)
                                            (nodeData := nodeData) (st1 := st1) (st' := st')
                                            (ctx := ctx) (vals0 := vals0) (input := input) hTail hEval
                                              hStep
                                        · exact False.elim <|
                                            throw_bind_ne_ok (by simpa [expected, hOut, hs] using hBuild)

/-- Correctness lemma for the `.avg_pool2d_pad` node compiler (explicit padding). -/
theorem buildFrom_denoteAllFrom_avg_pool2d_pad
    {α : Type} [Context α] [DecidableEq Shape]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : GraphData α Unit [inShape] ss) (i : Nat) (st' : State α inShape)
    (x : Tensor α inShape) (n : NN.IR.Node)
    (kH kW stride padding : Nat)
    (hN : g.getNode i = .ok n) (hk : n.kind = .avgPool2dPad kH kW stride padding) (hi : i <
      g.nodes.size)
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
      exact False.elim <| throw_bind_ne_ok (by simpa [hp] using hBuild)
  | cons pId ps =>
      cases ps with
      | cons _ _ =>
          exact False.elim <| throw_bind_ne_ok (by simpa [hp] using hBuild)
      | nil =>
          cases hP : g.getNode pId with
          | error msg =>
              have hFalse : False := by
                simp [hp, hP] at hBuild
              cases hFalse
          | ok pNode =>
              simp [hp, hP] at hBuild
              cases hS : pNode.outShape with
              | scalar =>
                  exact False.elim <| throw_bind_ne_ok (by simpa [hS] using hBuild)
              | dim inC sTail =>
                  cases sTail with
                  | scalar =>
                      exact False.elim <| throw_bind_ne_ok (by simpa [hS] using hBuild)
                  | dim inH sTail2 =>
                      cases sTail2 with
                      | scalar =>
                          exact False.elim <| throw_bind_ne_ok (by simpa [hS] using hBuild)
                      | dim inW sTail3 =>
                          cases sTail3 with
                          | dim _ _ =>
                              exact False.elim <| throw_bind_ne_ok (by simpa [hS] using hBuild)
                          | scalar =>
                              by_cases hkH0 : kH = 0
                              · exact False.elim <| throw_bind_ne_ok (by simpa [hS, hkH0] using
                                hBuild)
                              · by_cases hkW0 : kW = 0
                                · exact False.elim <| throw_bind_ne_ok (by simpa [hS, hkH0, hkW0]
                                  using hBuild)
                                · by_cases hs : stride = 0
                                  · exact False.elim <| throw_bind_ne_ok (by simpa [hS, hkH0, hkW0, hs]
                                    using hBuild)
                                  ·
                                    let sIn : Shape := .dim inC (.dim inH (.dim inW .scalar))
                                    cases hIdx :
                                        mkIdx (inShape := inShape) (ss := ss) pId sIn with
                                    | error msg =>
                                        have hFalse : False := by
                                          simp [hS, hkH0, hkW0, hs, sIn, hIdx] at hBuild
                                        cases hFalse
                                    | ok ip =>
                                        simp [hS, hkH0, hkW0, hs, sIn, hIdx] at hBuild
                                        let expected : Shape :=
                                          Spec.pool2dMultiOutShapePad inC inH inW kH kW stride
                                            padding
                                        by_cases hOut : expected = n.outShape
                                        · simp [expected, hOut] at hBuild
                                          let layer : Spec.AvgPool2DSpec kH kW stride hkH0 hkW0 hs := {}
                                          let nodeData : NodeData α Unit ([inShape] ++ ss) n.outShape
                                            :=
                                            mkFwdNode (α := α) (Γ := [inShape] ++ ss) (τ :=
                                              n.outShape) (fun ctx =>
                                              let xCHW := getIdx (α := α) (xs := ctx) ip
                                              let y : Tensor α expected :=
                                                Spec.avgPool2dMultiSpecPad (α := α) (kH := kH) (kW
                                                  := kW)
                                                  (inH := inH) (inW := inW) (inC := inC) (stride :=
                                                    stride)
                                                  (padding := padding) (h1 := hkH0) (h2 := hkW0)
                                                    (layer := layer)
                                                  (input := xCHW)
                                              hOut ▸ y)
                                          let st1 : State α inShape :=
                                            ⟨ss ++ [n.outShape], .snoc (ss := ss) gd nodeData⟩
                                          have hRec :
                                              buildFrom (α := α) (g := g) (payload := payload)
                                                (inShape := inShape)
                                                  (i := i + 1) st1 = .ok st' := by
                                            simp [st1, nodeData]
                                            exact hBuild
                                          have hGet :
                                              vals0[pId]! =
                                                NN.IR.DVal.mk (α := α) sIn
                                                  (getIdx (α := α) (xs := ctx) ip) := by
                                            simpa [vals0, ctx, sIn] using
                                              (denoteAllState_get_mkIdx (inShape := inShape) (ss :=
                                                ss)
                                                (gd := gd) (x := x) (pid := pId) (s := sIn) (idx :=
                                                  ip) hIdx)
                                          have hEval :
                                              NN.IR.Graph.evalAt (α := α) (g := g) (payload :=
                                                payload)
                                                  (input := input) (vals := vals0) (i := i) =
                                                .ok (NN.IR.DVal.mk (α := α) n.outShape
                                                  (nodeData.forward ctx ())) := by
                                            have hShapeV : vals0[pId]!.shape = sIn := by
                                              simpa [NN.IR.DVal.shape, NN.IR.DVal.mk] using
                                                congrArg (fun v => NN.IR.DVal.shape (α := α) v) hGet
                                            have hFst : vals0[pId]!.fst = sIn := by
                                              simpa [NN.IR.DVal.shape] using hShapeV
                                            have hExpSIn :
                                                NN.IR.Graph.expectShape (α := α) (expected := sIn)
                                                  vals0[pId]! =
                                                  .ok (getIdx (α := α) (xs := ctx) ip) := by
                                              rw [hGet]
                                              exact (Graph.expectShape_sigma (α := α) (s := sIn)
                                                (t := getIdx (α := α) (xs := ctx) ip))
                                            have hExp :
                                                NN.IR.Graph.expectShape (α := α)
                                                    (expected := Shape.dim inC (Shape.dim inH
                                                      (Shape.dim inW Shape.scalar)))
                                                    vals0[pId]! =
                                                  .ok (getIdx (α := α) (xs := ctx) ip) := by
                                              simpa [sIn] using hExpSIn
                                            simp [NN.IR.Graph.evalAt, hN, hk, hp, hExp,
                                              nodeData, layer,
                                              hFst, sIn, expected, hkH0, hkW0, hs, hOut]
                                          have hStep :
                                              denoteAllState (α := α) inShape st1 x =
                                                vals0.push (NN.IR.DVal.mk (α := α) n.outShape
                                                  (nodeData.forward ctx ())) := by
                                            simpa [vals0, st1, nodeData, ctx] using
                                              (denoteAllState_snoc (α := α) (inShape := inShape) (ss
                                                := ss) (τ := n.outShape)
                                                (gd := gd) (nodeData := nodeData) (x := x))
                                          have hTail := ih st1 hRec
                                          exact buildFrom_denoteAllFrom_finish (α := α) (g := g)
                                            (payload := payload)
                                            (i := i) (x := x) (hi := hi) (τ := n.outShape)
                                            (nodeData := nodeData) (st1 := st1) (st' := st')
                                            (ctx := ctx) (vals0 := vals0) (input := input) hTail hEval
                                              hStep
                                        · exact False.elim <|
                                            throw_bind_ne_ok (by simpa [expected, hOut, hs] using hBuild)



end Compiled
end Autograd
end Runtime
