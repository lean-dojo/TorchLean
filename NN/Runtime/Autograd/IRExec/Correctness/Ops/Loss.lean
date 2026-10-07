/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.IRExec.Correctness.Common

/-!
# Loss

Successful `.mseLoss` lowering appends the same scalar loss as the IR evaluator. Both operands
must have the same shape. The loss averages over all elements and is zero for an empty tensor,
using `Tensor.meanDenominator` to avoid division by zero.

## Main definitions

- `buildFrom_denoteAllFrom_mseLoss`: correctness step for `.mseLoss` lowering.

## References

- [Mean squared error (concept overview)](https://en.wikipedia.org/wiki/Mean_squared_error)

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

/-- Correctness lemma for the `.mseLoss` node lowering pass. -/
theorem buildFrom_denoteAllFrom_mseLoss
    {α : Type} [TorchLean.Storage α] [Context α]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : ForwardData α [inShape] ss) (i : Nat) (st' : State α inShape)
    (x : Tensor α inShape) (n : NN.IR.Node)
    (hN : g.getNode i = .ok n) (hk : n.kind = .mseLoss) (hi : i < g.nodes.size)
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
  rcases n with ⟨nId, nParents, nKind, nOutShape⟩
  have hkKind : nKind = .mseLoss := by
    simpa using hk
  subst nKind
  -- Select the loss branch before splitting its dependent shape guards.
  have hBuild0 := hBuild
  unfold buildFrom at hBuild0
  simp (config := { failIfUnchanged := false }) only
    [hi, hN, Except.ok_bind]
    at hBuild0
  cases hp : binaryParents? nParents with
  | none =>
      exact False.elim <| throw_bind_ne_ok (h := (by simpa [hp, lowerMseLoss] using hBuild0))
  | some parentIds =>
      rcases parentIds with ⟨yId, tId⟩
      simp (config := { failIfUnchanged := false }) [hp, lowerMseLoss] at hBuild0
      cases hY : g.getNode yId with
      | error msg =>
          simp [hY] at hBuild0
      | ok yNode =>
          cases hT : g.getNode tId with
          | error msg =>
              simp [hY, hT] at hBuild0
          | ok tNode =>
              have hBuild1 := hBuild0
              simp (config := { failIfUnchanged := false }) only
                [hY, hT, Except.ok_bind]
                at hBuild1
              by_cases hShape : yNode.outShape = tNode.outShape
              · have hBuild2 := hBuild1
                simp (config := { failIfUnchanged := false }) only [hShape] at hBuild2
                by_cases hOut : Shape.scalar = nOutShape
                · have hBuild3 := hBuild2
                  simp (config := { failIfUnchanged := false }) only [hOut] at hBuild3
                  let s : Shape := yNode.outShape
                  cases hIy : mkIdx (inShape := inShape) (ss := ss) yId s with
                  | error msg =>
                      simp [s, hIy] at hBuild3
                  | ok iy =>
                      cases hIt : mkIdx (inShape := inShape) (ss := ss) tId s with
                      | error msg =>
                          simp [s, hIy, hIt] at hBuild3
                      | ok it =>
                          have hBuild4 := hBuild3
                          simp (config := { failIfUnchanged := false }) only
                            [s, hIy, hIt, Except.ok_bind]
                            at hBuild4
                          let nodeData : ForwardNode α ([inShape] ++ ss) nOutShape :=
                            mkForwardNode (α := α) (Γ := [inShape] ++ ss) (τ := nOutShape) (fun
                              ctx =>
                              let yhat := ctx.read iy
                              let target := ctx.read it
                              let diff := Tensor.subSpec (α := α) yhat target
                              let sq := Tensor.mulSpec (α := α) diff diff
                              let total : α := Tensor.sumSpec (α := α) sq
                              let y0 : Tensor α .scalar :=
                                Tensor.scalar (total / (↑(TorchLean.Tensor.meanDenominator s) : α))
                              Tensor.castShape y0 hOut)
                          let st1 : State α inShape :=
                            ⟨ss ++ [nOutShape], .snoc (ss := ss) gd nodeData⟩
                          have hs : tNode.outShape = s := by
                            simpa [s] using hShape.symm
                          have hRec :
                              buildFrom (α := α) (g := g) (payload := payload) (inShape :=
                                inShape)
                                  (i := i + 1) st1 = .ok st' := by
                            simpa [st1, nodeData, hs, Tensor.cast_shape_proof_irrel] using hBuild4
                          have hGetY :
                              vals0[yId]? = some (Spec.SomeTensor.mk (α := α) s
                                (getIdx (α := α) (xs := ctx) iy)) := by
                            simpa [vals0, ctx, s] using
                              (denoteAllState_get_mkIdx? (inShape := inShape) (ss := ss)
                                (gd := gd) (x := x) (pid := yId) (s := s) (idx := iy) hIy)
                          have hGetT :
                              vals0[tId]? = some (Spec.SomeTensor.mk (α := α) s
                                (getIdx (α := α) (xs := ctx) it)) := by
                            simpa [vals0, ctx, s] using
                              (denoteAllState_get_mkIdx? (inShape := inShape) (ss := ss)
                                (gd := gd) (x := x) (pid := tId) (s := s) (idx := it) hIt)
                          have hParentIds :
                              NN.IR.Graph.binaryParentIds i
                                  { id := nId, parents := nParents, kind := .mseLoss,
                                    outShape := nOutShape } = .ok (yId, tId) :=
                            binaryParentIds_eq_ok_of_binaryParents_eq_some i yId tId
                              { id := nId, parents := nParents, kind := .mseLoss,
                                outShape := nOutShape } hp
                          have hEval :
                              NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
                                  (input := input) (vals := vals0) (i := i) =
                                .ok (Spec.SomeTensor.mk (α := α) nOutShape
                                  (nodeData.eval ctx)) := by
                            -- Reconcile the evaluator's equality transport with `castShape`.
                            unfold NN.IR.Graph.evalAt NN.IR.Graph.evalNode
                            simp (config := { failIfUnchanged := false })
                              [hN, hParentIds, hGetY, hGetT,
                                NN.IR.Graph.mseLossSomeTensor_mk,
                                hOut, nodeData, mkForwardNode,
                                NN.IR.Graph.normalizeNodeOutput,
                                Tensor.eqRec_eq_cast_shape,
                                Tensor.cast_shape_proof_irrel]
                            congr 1
                          have hTail := ih st1 hRec
                          exact buildFrom_denoteAllFrom_nodeData_exact (α := α) (g := g)
                            (payload := payload)
                            (gd := gd) (i := i) (st' := st') (x := x) (hi := hi)
                            (τ := nOutShape) (nodeData := nodeData) hTail hEval
                · -- `simp` normalizes the guard to `nOutShape = []` (via `List.nil_eq`), so the
                  -- `dite_eq_right` witness has to be stated in that orientation too.
                  exact False.elim <|
                    throw_bind_ne_ok (h := (by simpa [dite_eq_right (Ne.symm hOut)] using hBuild2))
              · exact False.elim <|
                  throw_bind_ne_ok (h := (by simpa [ite_eq_right hShape] using hBuild1))

end IRExec
end Autograd
end Runtime
