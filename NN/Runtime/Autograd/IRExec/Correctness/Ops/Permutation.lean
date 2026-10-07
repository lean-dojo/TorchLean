/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.IRExec.Correctness.SemanticEquivalenceCommon

/-!
# Permutation Operators

Semantic-preservation lemmas for permutation-style operators in the IR-to-forward-graph bridge.

Permutation and transpose both lower to adjacent-axis swaps. `permuteSomeTensor_eq_swapAxes`
identifies the shape-erased evaluator's swap fold with the typed `swapAxes` calculation. Transpose
first checks its two axes and constructs the corresponding permutation. Both proofs retain the
computed-shape and declared-output checks before composing with the rest of the graph.
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

/-- Semantic-preservation lemma for `.permute perm` lowering. -/
theorem buildFrom_denoteAllFrom_permute
    {α : Type} [TorchLean.Storage α] [Context α]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : ForwardData α [inShape] ss) (i : Nat) (st' : State α inShape)
    (x : Tensor α inShape) (n : NN.IR.Node) (perm : Array Nat)
    (hN : g.getNode i = .ok n) (hk : n.kind = .permute perm) (hi : i < g.nodes.size)
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
  simp (config := { failIfUnchanged := false }) [hi, hN, hk, lowerPermute] at hBuild
  cases hp : unaryParent? n.parents with
  | none =>
      simp [hp, throw_eq_error] at hBuild
      try cases hBuild
  | some pId =>
          cases hP : g.getNode pId with
          | error msg =>
              simp [hp, hP, throw_eq_error] at hBuild
              try cases hBuild
          | ok pNode =>
              simp (config := { failIfUnchanged := false }) [hp, hP] at hBuild
              cases hIdx : mkIdx (inShape := inShape) (ss := ss) pId pNode.outShape with
              | error msg =>
                  simp [hIdx] at hBuild
                  try cases hBuild
              | ok ip =>
                  simp [hIdx] at hBuild
                  cases hPerm : Spec.Shape.permute? pNode.outShape perm.toList with
                  | none =>
                      simp [hPerm] at hBuild
                      try cases hBuild
                  | some expected =>
                      simp [hPerm] at hBuild
                      cases hSwaps :
                          NN.IR.Graph.swapDepthsForPerm perm
                              (Spec.Shape.rank pNode.outShape) with
                      | error msg =>
                          simp [hSwaps] at hBuild
                          try cases hBuild
                      | ok swaps =>
                          simp [hSwaps] at hBuild
                          let sFinal : Shape := swapShape pNode.outShape swaps
                          by_cases hFinal : sFinal = expected
                          · simp [sFinal, hFinal, throw_eq_error] at hBuild
                            by_cases hOut : expected = n.outShape
                            · simp [hOut] at hBuild
                              let nodeData : ForwardNode α ([inShape] ++ ss) n.outShape :=
                                mkForwardNode (α := α) (Γ := [inShape] ++ ss) (τ := n.outShape)
                                  (fun ctx =>
                                    let x := ctx.read ip
                                    let y : Tensor α sFinal :=
                                      swapAxes (α := α) (s := pNode.outShape)
                                        (swaps := swaps) x
                                    let yExpected : Tensor α expected :=
                                      Tensor.castShape y hFinal
                                    Tensor.castShape yExpected hOut)
                              let st1 : State α inShape :=
                                ⟨ss ++ [n.outShape], .snoc (ss := ss) gd nodeData⟩
                              have hRec :
                                  buildFrom (α := α) (g := g) (payload := payload)
                                      (inShape := inShape) (i := i + 1) st1 =
                                    .ok st' := by
                                simpa [st1, nodeData] using hBuild
                              have hTail := ih st1 hRec
                              have hGet :
                                  vals0[pId]? = some (Spec.SomeTensor.mk (α := α) pNode.outShape
                                      (getIdx (α := α) (xs := ctx) ip)) := by
                                simpa [vals0, ctx] using
                                  (denoteAllState_get_mkIdx? (inShape := inShape) (ss := ss)
                                    (gd := gd) (x := x) (pid := pId) (s := pNode.outShape)
                                    (idx := ip)
                                    hIdx)
                              have hEval :
                                  NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
                                      (input := input) (vals := vals0) (i := i) =
                                    .ok (Spec.SomeTensor.mk (α := α) n.outShape
                                      (nodeData.eval ctx)) := by
                                have hPermute :
                                    NN.IR.Graph.permuteSomeTensor (α := α)
                                        (v := Spec.SomeTensor.mk (α := α) pNode.outShape
                                          (getIdx (α := α) (xs := ctx) ip)) perm =
                                      .ok
                                        (Spec.SomeTensor.mk (α := α)
                                          (swapShape pNode.outShape swaps)
                                          (swapAxes (α := α) (s := pNode.outShape)
                                            (swaps := swaps)
                                            (getIdx (α := α) (xs := ctx) ip))) := by
                                  simpa using
                                    (permuteSomeTensor_eq_swapAxes (α := α)
                                      (t := getIdx (α := α) (xs := ctx) ip) (perm := perm)
                                      (expected := expected) (swaps := swaps) hPerm (by
                                        simpa [NN.IR.Graph.swapDepthsForPerm] using hSwaps))
                                have hShape :
                                    swapShape pNode.outShape swaps = n.outShape := by
                                  simpa [sFinal] using (hFinal.trans hOut)
                                simp (config := { failIfUnchanged := false })
                                  [NN.IR.Graph.evalAt, NN.IR.Graph.evalNode,
                                    NN.IR.Graph.normalizeNodeOutput,
                                    hN, hk, hp, hGet, throw_eq_error]
                                erw [hPermute]
                                simp (config := { failIfUnchanged := false })
                                -- Transport the computed swap shape to the declared output.
                                rw [dite_eq_left hShape]
                                simp [nodeData, sFinal, mkForwardNode,
                                  Tensor.eqRec_eq_cast_shape, Tensor.cast_shape_trans]
                                change Except.ok _ = Except.ok _
                                congr 2
                              exact buildFrom_denoteAllFrom_nodeData_exact (α := α) (g := g)
                                (payload := payload)
                                (gd := gd) (i := i) (st' := st') (x := x) (hi := hi)
                                (τ := n.outShape) (nodeData := nodeData) hTail hEval
                            · simp [hOut] at hBuild
                              try cases hBuild
                          · simp [sFinal, hFinal, throw_eq_error] at hBuild
                            try cases hBuild

/-- Semantic preservation for transposing any two axes of a tensor. -/
theorem buildFrom_denoteAllFrom_transpose
    {α : Type} [TorchLean.Storage α] [Context α]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : ForwardData α [inShape] ss) (i : Nat) (st' : State α inShape)
    (x : Tensor α inShape) (n : NN.IR.Node) (axis₁ axis₂ : Nat)
    (hN : g.getNode i = .ok n) (hk : n.kind = .transpose axis₁ axis₂) (hi : i < g.nodes.size)
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
  simp (config := { failIfUnchanged := false }) [hi, hN, hk, lowerTranspose] at hBuild
  cases hp : unaryParent? n.parents with
  | none =>
      simp [hp, throw_eq_error] at hBuild
      try cases hBuild
  | some pId =>
          cases hP : g.getNode pId with
          | error msg =>
              simp [hp, hP, throw_eq_error] at hBuild
              try cases hBuild
          | ok pNode =>
              simp (config := { failIfUnchanged := false }) [hp, hP] at hBuild
              cases hIdx : mkIdx (inShape := inShape) (ss := ss) pId pNode.outShape with
              | error msg =>
                  simp [hIdx] at hBuild
                  try cases hBuild
              | ok ip =>
                  simp [hIdx] at hBuild
                  cases hTranspose :
                      OpContracts.transposePerm pNode.outShape.rank axis₁ axis₂ with
                  | error msg =>
                      simp [hTranspose, throw_eq_error] at hBuild
                      try cases hBuild
                  | ok perm =>
                      simp [hTranspose] at hBuild
                      cases hPerm : Spec.Shape.permute? pNode.outShape perm.toList with
                      | none =>
                          simp [hPerm] at hBuild
                          try cases hBuild
                      | some expected =>
                          simp [hPerm] at hBuild
                          cases hSwaps :
                              NN.IR.Graph.swapDepthsForPerm perm
                                  (Spec.Shape.rank pNode.outShape) with
                          | error msg =>
                              simp [hSwaps] at hBuild
                              try cases hBuild
                          | ok swaps =>
                              simp [hSwaps] at hBuild
                              let sFinal : Shape := swapShape pNode.outShape swaps
                              by_cases hFinal : sFinal = expected
                              · simp [sFinal, hFinal, throw_eq_error] at hBuild
                                by_cases hOut : expected = n.outShape
                                · simp [hOut] at hBuild
                                  let nodeData : ForwardNode α ([inShape] ++ ss) n.outShape :=
                                    mkForwardNode (α := α) (Γ := [inShape] ++ ss) (τ := n.outShape)
                                      (fun ctx =>
                                        let x := ctx.read ip
                                        let y : Tensor α sFinal :=
                                          swapAxes (α := α) (s := pNode.outShape)
                                            (swaps := swaps) x
                                        let yExpected : Tensor α expected :=
                                          Tensor.castShape y hFinal
                                        Tensor.castShape yExpected hOut)
                                  let st1 : State α inShape :=
                                    ⟨ss ++ [n.outShape], .snoc (ss := ss) gd nodeData⟩
                                  have hRec :
                                      buildFrom (α := α) (g := g) (payload := payload)
                                          (inShape := inShape) (i := i + 1) st1 =
                                        .ok st' := by
                                    simpa [st1, nodeData] using hBuild
                                  have hTail := ih st1 hRec
                                  have hGet :
                                      vals0[pId]? = some (Spec.SomeTensor.mk (α := α) pNode.outShape
                                          (getIdx (α := α) (xs := ctx) ip)) := by
                                    simpa [vals0, ctx] using
                                      (denoteAllState_get_mkIdx? (inShape := inShape) (ss := ss)
                                        (gd := gd) (x := x) (pid := pId) (s := pNode.outShape)
                                        (idx := ip)
                                        hIdx)
                                  have hEval :
                                      NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
                                          (input := input) (vals := vals0) (i := i) =
                                        .ok (Spec.SomeTensor.mk (α := α) n.outShape
                                          (nodeData.eval ctx)) := by
                                    have hPermute :
                                        NN.IR.Graph.permuteSomeTensor (α := α)
                                            (v := Spec.SomeTensor.mk (α := α) pNode.outShape
                                              (getIdx (α := α) (xs := ctx) ip)) perm =
                                          .ok
                                            (Spec.SomeTensor.mk (α := α)
                                              (swapShape pNode.outShape swaps)
                                              (swapAxes (α := α) (s := pNode.outShape)
                                                (swaps := swaps)
                                                (getIdx (α := α) (xs := ctx) ip))) := by
                                      simpa using
                                        (permuteSomeTensor_eq_swapAxes (α := α)
                                          (t := getIdx (α := α) (xs := ctx) ip) (perm := perm)
                                          (expected := expected) (swaps := swaps) hPerm (by
                                            simpa [NN.IR.Graph.swapDepthsForPerm] using hSwaps))
                                    have hShape :
                                        swapShape pNode.outShape swaps = n.outShape := by
                                      simpa [sFinal] using (hFinal.trans hOut)
                                    simp (config := { failIfUnchanged := false })
                                      [NN.IR.Graph.evalAt, NN.IR.Graph.evalNode,
                                        NN.IR.Graph.normalizeNodeOutput,
                                        hN, hk, hp, hGet, hTranspose, throw_eq_error]
                                    erw [hPermute]
                                    simp (config := { failIfUnchanged := false })
                                    rw [Graph.expectShape_mk_of_eq hShape]
                                    simp [nodeData, sFinal, mkForwardNode,
                                      Tensor.cast_shape_trans]
                                  exact buildFrom_denoteAllFrom_nodeData_exact (α := α) (g := g)
                                    (payload := payload)
                                    (gd := gd) (i := i) (st' := st') (x := x) (hi := hi)
                                    (τ := n.outShape) (nodeData := nodeData) hTail hEval
                                · simp [hOut] at hBuild
                                  try cases hBuild
                              · simp [sFinal, hFinal, throw_eq_error] at hBuild
                                try cases hBuild

end IRExec
end Autograd
end Runtime
