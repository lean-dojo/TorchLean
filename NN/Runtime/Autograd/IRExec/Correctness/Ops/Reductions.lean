/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.IRExec.Correctness.Common

/-!
# Reductions and Broadcasting

Correctness lemmas for IR nodes whose primary behavior is broadcasting or reduction:

* `broadcastTo s₁ s₂` (explicit broadcasting, used to keep elementwise ops simple),
* `reduceSum axis` and `reduceMean axis` (single-axis reductions),
* `sum` (full reduction to a scalar).

Sum and mean share the axis-validation proof but keep their distinct denotations. Axis reductions
require the selected dimension to be nonempty; other dimensions may be empty. The full sum accepts
empty tensors and folds scalar leaves in row-major order from zero. No associativity or native
reduction-order assumption is used. `buildFrom_denoteAllFrom_nodeData_exact` handles the remaining
graph after each operation's output has been identified.
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace IRExec

open Spec TorchLean
open Proofs.Autograd.Algebra
open NN.IR
open Internal
open Proofs (Idx getIdx)

/-- Correctness lemma for `.broadcastTo s₁ s₂` lowering. -/
theorem buildFrom_denoteAllFrom_broadcastTo
    {α : Type} [TorchLean.Storage α] [Context α]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : ForwardData α [inShape] ss) (i : Nat) (st' : State α inShape)
    (x : Tensor α inShape) (n : NN.IR.Node)
    (s₁ s₂ : Shape)
    (hN : g.getNode i = .ok n) (hk : n.kind = .broadcastTo s₁ s₂) (hi : i < g.nodes.size)
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
  simp (config := { failIfUnchanged := false }) [hk, lowerBroadcastTo] at hBuild
  cases hp : unaryParent? n.parents with
  | none =>
      simp [hp] at hBuild; try cases hBuild
  | some pId =>
          cases hIdx : mkIdx (inShape := inShape) (ss := ss) pId s₁ with
          | error msg =>
              simp [hp, hIdx] at hBuild; try cases hBuild
          | ok ip =>
              simp [hp, hIdx] at hBuild
              by_cases hCan : Spec.Shape.CanBroadcastTo s₁ s₂
              swap
              · simp [hCan] at hBuild; try cases hBuild
              ·
                  simp [hCan] at hBuild
                  by_cases hOut : s₂ = n.outShape
                  ·
                    simp [hOut] at hBuild
                    let nodeData : ForwardNode α ([inShape] ++ ss) n.outShape :=
                      mkForwardNode (α := α) (Γ := [inShape] ++ ss) (τ := n.outShape)
                        (fun ctx =>
                          let x := ctx.read ip
                          hOut ▸ Tensor.broadcastTo (α := α) (s₁ := s₁) (s₂ := s₂) hCan x)
                    let st1 : State α inShape :=
                      ⟨ss ++ [n.outShape], .snoc (ss := ss) gd nodeData⟩
                    have hRec :
                        buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
                          (i := i + 1) st1 = .ok st' := by
                      simpa [st1, nodeData] using hBuild
                    have hGet :
                        vals0[pId]? = some (Spec.SomeTensor.mk (α := α) s₁
                            (getIdx (α := α) (xs := ctx) ip)) := by
                      simpa [vals0, ctx] using
                        (denoteAllState_get_mkIdx? (inShape := inShape) (ss := ss)
                          (gd := gd) (x := x) (pid := pId) (s := s₁) (idx := ip) hIdx)
                    have hEval :
                        NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
                            (input := input) (vals := vals0) (i := i) =
                          .ok
                            (Spec.SomeTensor.mk (α := α) n.outShape (nodeData.eval ctx)) := by
                      cases hOut
                      simp [NN.IR.Graph.evalAt, NN.IR.Graph.evalNode,
                        NN.IR.Graph.normalizeNodeOutput, hN, hk, hp, hGet, hCan,
                        nodeData, mkForwardNode, throw_eq_error,
                        Pure.pure, Except.pure]
                    have hTail := ih st1 hRec
                    exact buildFrom_denoteAllFrom_nodeData_exact (α := α) (g := g)
                      (payload := payload) (gd := gd) (i := i) (st' := st') (x := x) (hi := hi)
                      (τ := n.outShape) (nodeData := nodeData) hTail hEval
                  ·
                    simp [hOut] at hBuild
                    try cases hBuild

/-- The two axis reductions have the same validation proof and retain their own denotation. -/
theorem buildFrom_denoteAllFrom_axisReduction
    {α : Type} [TorchLean.Storage α] [Context α]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : ForwardData α [inShape] ss) (i : Nat) (st' : State α inShape)
    (x : Tensor α inShape) (n : NN.IR.Node)
    (operation : AxisReductionKind) (axis : Nat)
    (hN : g.getNode i = .ok n) (hk : n.kind = operation.toOpKind axis) (hi : i < g.nodes.size)
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

  have hLower : lowerNode (loweringContext g payload inShape ss i n) =
      lowerAxisReduction (loweringContext g payload inShape ss i n) operation axis := by
    cases operation <;> simp [loweringContext, AxisReductionKind.toOpKind, hk,
      lowerReduceSum, lowerReduceMean]
  unfold buildFrom at hBuild
  simp only [hi, ↓reduceDIte, hN, Except.ok_bind] at hBuild
  rw [hLower] at hBuild
  cases hp : unaryParent? n.parents with
  | none =>
      simp [hp] at hBuild; try cases hBuild
  | some pId =>
          cases hP : g.getNode pId with
          | error msg =>
              simp [hp, hP] at hBuild; try cases hBuild
          | ok pNode =>
              simp [hp, hP] at hBuild
              let s := pNode.outShape
              cases hIdx : mkIdx (inShape := inShape) (ss := ss) pId s with
              | error msg =>
                  simp [s, hIdx] at hBuild
              | ok ip =>
                  simp (config := { failIfUnchanged := false }) [s, hIdx] at hBuild
                  cases hAxis : Spec.Shape.nonemptyAxis? (axis := axis) s with
                  | none =>
                      exact False.elim <| throw_bind_ne_ok (h := (by simpa [s, hAxis] using hBuild))
                  | some hAxisPf =>
                      simp (config := { failIfUnchanged := false }) [s, hAxis] at hBuild
                      let hRed := hAxisPf.down
                      let expected : Shape := TorchLean.Tensor.shapeAfterSum s axis
                      by_cases hOut : expected = n.outShape
                      ·
                        have hCond :
                            TorchLean.Tensor.shapeAfterSum pNode.outShape axis = n.outShape := by
                          simpa [expected, s] using hOut
                        simp [hCond] at hBuild
                        let nodeData : ForwardNode α ([inShape] ++ ss) n.outShape :=
                          mkForwardNode (α := α) (Γ := [inShape] ++ ss) (τ := n.outShape)
                            (fun ctx =>
                              let x := ctx.read ip
                              let y : Tensor α expected :=
                                operation.denote axis x hRed
                              hOut ▸ y)
                        let st1 : State α inShape :=
                          ⟨ss ++ [n.outShape], .snoc (ss := ss) gd nodeData⟩
                        have hRec :
                            buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
                              (i := i + 1) st1 = .ok st' := by
                          simpa [st1, nodeData] using hBuild
                        have hGet :
                            vals0[pId]? = some (Spec.SomeTensor.mk (α := α) s
                                (getIdx (α := α) (xs := ctx) ip)) := by
                          simpa [vals0, ctx] using
                            (denoteAllState_get_mkIdx? (inShape := inShape) (ss := ss)
                              (gd := gd) (x := x) (pid := pId) (s := s) (idx := ip) hIdx)
                        have hEval :
                            NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
                                (input := input) (vals := vals0) (i := i) =
                              .ok (Spec.SomeTensor.mk (α := α) n.outShape
                                (nodeData.eval ctx)) := by
                          simpa [nodeData, mkForwardNode] using
                            (evalAt_axisReduction_ok (α := α) operation (g := g)
                              (payload := payload)
                              (input := input) (vals := vals0) (i := i) (n := n)
                              (pId := pId) (axis := axis) (s := s)
                              (pT := getIdx (α := α) (xs := ctx) ip) (hAxisPf := hAxisPf)
                              (hN := hN) (hk := hk) (hp := hp) (hGet := hGet) (hAxis := hAxis)
                              (hOut := hOut))
                        have hTail := ih st1 hRec
                        exact buildFrom_denoteAllFrom_nodeData_exact (α := α) (g := g)
                          (payload := payload) (gd := gd) (i := i) (st' := st') (x := x) (hi := hi)
                          (τ := n.outShape) (nodeData := nodeData) hTail hEval
                      ·
                        have hCondNe :
                            ¬TorchLean.Tensor.shapeAfterSum pNode.outShape axis = n.outShape := by
                          intro hEq
                          apply hOut
                          simpa [expected, s] using hEq
                        simp [hCondNe] at hBuild
                        try cases hBuild

/-- Correctness lemma for `.reduceSum axis` lowering. -/
theorem buildFrom_denoteAllFrom_reduceSum
    {α : Type} [TorchLean.Storage α] [Context α]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : ForwardData α [inShape] ss) (i : Nat) (st' : State α inShape)
    (x : Tensor α inShape) (n : NN.IR.Node)
    (axis : Nat)
    (hN : g.getNode i = .ok n) (hk : n.kind = .reduceSum axis) (hi : i < g.nodes.size)
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
  exact buildFrom_denoteAllFrom_axisReduction g payload gd i st' x n .sum axis
    hN hk hi hBuild ih

/-- Correctness lemma for `.reduceMean axis` lowering. -/
theorem buildFrom_denoteAllFrom_reduceMean
    {α : Type} [TorchLean.Storage α] [Context α]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : ForwardData α [inShape] ss) (i : Nat) (st' : State α inShape)
    (x : Tensor α inShape) (n : NN.IR.Node)
    (axis : Nat)
    (hN : g.getNode i = .ok n) (hk : n.kind = .reduceMean axis) (hi : i < g.nodes.size)
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
  exact buildFrom_denoteAllFrom_axisReduction g payload gd i st' x n .mean axis
    hN hk hi hBuild ih

/-- Correctness lemma for `.sum` lowering (sum-reduction to scalar). -/
theorem buildFrom_denoteAllFrom_sum
    {α : Type} [TorchLean.Storage α] [Context α]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : ForwardData α [inShape] ss) (i : Nat) (st' : State α inShape)
    (x : Tensor α inShape) (n : NN.IR.Node)
    (hN : g.getNode i = .ok n) (hk : n.kind = .sum) (hi : i < g.nodes.size)
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
  simp (config := { failIfUnchanged := false }) [hk, lowerSum] at hBuild
  cases hp : unaryParent? n.parents with
  | none =>
      simp [hp] at hBuild; try cases hBuild
  | some pId =>
          cases hP : g.getNode pId with
          | error msg =>
              simp [hp, hP] at hBuild; try cases hBuild
          | ok pNode =>
              simp [hp, hP] at hBuild
              let s := pNode.outShape
              cases hIdx : mkIdx (inShape := inShape) (ss := ss) pId s with
              | error msg =>
                  simp [s, hIdx] at hBuild
              | ok ip =>
                  simp (config := { failIfUnchanged := false }) [s, hIdx] at hBuild
                  by_cases hOut : Shape.scalar = n.outShape
                  ·
                    simp [hOut] at hBuild
                    let nodeData : ForwardNode α ([inShape] ++ ss) n.outShape :=
                      mkForwardNode (α := α) (Γ := [inShape] ++ ss) (τ := n.outShape)
                        (fun ctx =>
                          let x := ctx.read ip
                          hOut ▸ Tensor.scalar (Tensor.sumSpec (α := α) x))
                    let st1 : State α inShape :=
                      ⟨ss ++ [n.outShape], .snoc (ss := ss) gd nodeData⟩
                    have hRec :
                        buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
                          (i := i + 1) st1 = .ok st' := by
                      simpa [st1, nodeData] using hBuild
                    have hGet :
                        vals0[pId]? = some (Spec.SomeTensor.mk (α := α) s
                            (getIdx (α := α) (xs := ctx) ip)) := by
                      simpa [vals0, ctx] using
                        (denoteAllState_get_mkIdx? (inShape := inShape) (ss := ss)
                          (gd := gd) (x := x) (pid := pId) (s := s) (idx := ip) hIdx)
                    have hEval :
                        NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
                            (input := input) (vals := vals0) (i := i) =
                          .ok (Spec.SomeTensor.mk (α := α) n.outShape
                            (nodeData.eval ctx)) := by
                      simp [NN.IR.Graph.evalAt, NN.IR.Graph.evalNode,
                        NN.IR.Graph.normalizeNodeOutput, hN, hk, hp, hGet,
                        hOut, nodeData, mkForwardNode]
                      rfl
                    have hTail := ih st1 hRec
                    exact buildFrom_denoteAllFrom_nodeData_exact (α := α) (g := g)
                      (payload := payload) (gd := gd) (i := i) (st' := st') (x := x) (hi := hi)
                      (τ := n.outShape) (nodeData := nodeData) hTail hEval
                  ·
                    -- `simp` normalizes the guard to `n.outShape = []` (via `List.nil_eq`), so the
                    -- `dite_eq_right` witness has to be stated in that orientation too.
                    exact False.elim <|
                      throw_bind_ne_ok (h := (by simpa [dite_eq_right (Ne.symm hOut)] using hBuild))

end IRExec
end Autograd
end Runtime
