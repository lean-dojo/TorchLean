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
# Loss

Loss-function correctness lemmas for the IR → compiled runtime bridge.

The IR node kind `.mse_loss` is compiled into an SSA node whose `forward` computes the
specification-level mean squared error loss.

This file proves the forward-correctness lemma for that compilation step: on successful compilation
at position `i`, the IR evaluator `NN.IR.Graph.denoteAllFrom` and the compiled evaluator
`denoteAllState` append the same result.

This is a structural correctness statement: it is not about generalization, training convergence,
or statistical properties of MSE; it simply connects the IR semantics to the compiled runtime node.

## Main definitions

- `buildFrom_denoteAllFrom_mse_loss`: correctness step for `.mse_loss` lowering.

## Implementation notes

- We keep this theorem in a dedicated file because it is heavier than most per-op steps.
- The proof structure follows the compiler's guard sequence; this keeps dependent shape checks
  aligned with compiler guards.
- This file can build slowly because MSE touches two parents, a scalar output shape, and a sequence
  of compiler guards. Repeated guard eliminations belong in focused helper lemmas, leaving the
  theorem focused on the loss equation itself.

## References

- [Mean squared error (concept overview)](https://en.wikipedia.org/wiki/Mean_squared_error)

## Tags

mse-loss, correctness, ir, runtime, semantic equivalence
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
/-- Correctness lemma for the `.mse_loss` node compiler. -/
theorem buildFrom_denoteAllFrom_mse_loss
    {α : Type} [Context α] [DecidableEq Shape]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : GraphData α Unit [inShape] ss) (i : Nat) (st' : State α inShape)
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
  rcases n with ⟨nId, nParents, nKind, nOutShape⟩
  have hkKind : nKind = .mseLoss := by
    simpa using hk
  subst nKind
  -- Pre-simplify `buildFrom` once so we don't repeatedly whnf the huge op-table in every branch.
  have hBuild0 := hBuild
  unfold buildFrom at hBuild0
  -- Keep simp very focused: unfolding `buildFrom` introduces a large `match` over op-kinds, and we
  -- only want to reduce the control-flow forced by `hi`, `hN`, and the monad bind structure.
  simp (config := { failIfUnchanged := false }) only
    [hi, hN, Except.ok_bind, Except.error_bind, Except.bind_ok, Except.bind_error, Except.pure]
    at hBuild0
  cases nParents with
  | nil =>
      exact False.elim <| throw_bind_ne_ok (h := (by simpa using hBuild0))
  | cons yId rest =>
      cases rest with
      | nil =>
          exact False.elim <| throw_bind_ne_ok (h := (by simpa using hBuild0))
      | cons tId rest2 =>
          cases rest2 with
          | cons _ _ =>
              exact False.elim <| throw_bind_ne_ok (h := (by simpa using hBuild0))
          | nil =>
              cases hY : g.getNode yId with
              | error msg =>
                  have : False := by
                    simpa [hY] using hBuild0
                  cases this
              | ok yNode =>
                  cases hT : g.getNode tId with
                  | error msg =>
                      have : False := by
                        simpa [hY, hT] using hBuild0
                      cases this
                  | ok tNode =>
                      have hBuild1 := hBuild0
                      -- Keep simp focused; `buildFrom` has a large op table, and default simp
                      -- search does unnecessary work here.
                      simp (config := { failIfUnchanged := false }) only
                        [hY, hT, Except.ok_bind, Except.error_bind, Except.bind_ok, Except.bind_error,
                          Except.pure]
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
                              have : False := by
                                simpa [s, hIy] using hBuild3
                              cases this
                          | ok iy =>
                              cases hIt : mkIdx (inShape := inShape) (ss := ss) tId s with
                              | error msg =>
                                  have : False := by
                                    simpa [s, hIy, hIt] using hBuild3
                                  cases this
                              | ok it =>
                                  have hBuild4 := hBuild3
                                  simp (config := { failIfUnchanged := false }) only
                                    [s, hIy, hIt, Except.ok_bind, Except.error_bind, Except.bind_ok,
                                      Except.bind_error, Except.pure]
                                    at hBuild4
                                  let nodeData : NodeData α Unit ([inShape] ++ ss) nOutShape :=
                                    mkFwdNode (α := α) (Γ := [inShape] ++ ss) (τ := nOutShape) (fun
                                      ctx =>
                                      let yhat := getIdx (α := α) (xs := ctx) iy
                                      let target := getIdx (α := α) (xs := ctx) it
                                      let diff := Tensor.subSpec (α := α) yhat target
                                      let sq := Tensor.mulSpec (α := α) diff diff
                                      let total : α := Tensor.sumSpec (α := α) sq
                                      let y0 : Tensor α Shape.scalar :=
                                        Tensor.scalar (total / (↑(Shape.size s) : α))
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
                                      vals0[yId]! =
                                        NN.IR.DVal.mk (α := α) s (getIdx (α := α) (xs := ctx) iy) :=
                                          by
                                    simpa [vals0, ctx, s] using
                                      (denoteAllState_get_mkIdx (inShape := inShape) (ss := ss)
                                        (gd := gd) (x := x) (pid := yId) (s := s) (idx := iy) hIy)
                                  have hGetT :
                                      vals0[tId]! =
                                        NN.IR.DVal.mk (α := α) s (getIdx (α := α) (xs := ctx) it) :=
                                          by
                                    simpa [vals0, ctx, s] using
                                      (denoteAllState_get_mkIdx (inShape := inShape) (ss := ss)
                                        (gd := gd) (x := x) (pid := tId) (s := s) (idx := it) hIt)
                                  have hEval :
                                      NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
                                          (input := input) (vals := vals0) (i := i) =
                                        .ok (NN.IR.DVal.mk (α := α) nOutShape (nodeData.forward ctx
                                          ())) := by
                                    -- `evalAt` normalizes using `Eq.rec` casts, while the compiled
                                    -- `forward` closure uses `Tensor.cast_shape`.
                                    --
                                    -- Reduce the node fetch first so the large `OpKind` match
                                    -- collapses to the `.mse_loss` branch.
                                    unfold NN.IR.Graph.evalAt
                                    simp (config := { failIfUnchanged := false }) [hN]
                                    have hMSE :
                                        NN.IR.Graph.mseLossDVal (α := α) i vals0[yId]! vals0[tId]! =
                                          .ok (NN.IR.DVal.mk (α := α) Shape.scalar
                                            (Tensor.scalar
                                              ((((getIdx (α := α) (xs := ctx) iy).subSpec
                                                    (getIdx (α := α) (xs := ctx) it)).mulSpec
                                                  ((getIdx (α := α) (xs := ctx) iy).subSpec
                                                    (getIdx (α := α) (xs := ctx) it))).sumSpec /
                                                (↑(Shape.size s) : α)))) := by
                                      rw [hGetY, hGetT]
                                      exact NN.IR.Graph.mseLossDVal_mk (α := α) i
                                        (getIdx (α := α) (xs := ctx) iy)
                                        (getIdx (α := α) (xs := ctx) it)
                                    simp (config := { failIfUnchanged := false })
                                      [hMSE, hOut, nodeData, mkFwdNode,
                                        Tensor.eqRec_eq_cast_shape, Tensor.cast_shape_self,
                                        Tensor.cast_shape_proof_irrel,
                                        NN.IR.DVal.shape, NN.IR.DVal.tensor, NN.IR.DVal.mk]
                                    congr 1
                                  have hStep :
                                      denoteAllState (α := α) inShape st1 x =
                                        vals0.push (NN.IR.DVal.mk (α := α) nOutShape
                                          (nodeData.forward ctx ())) := by
                                    simpa [vals0, st1, nodeData, ctx] using
                                      (denoteAllState_snoc (α := α) (inShape := inShape) (ss := ss)
                                        (τ := nOutShape) (gd := gd) (nodeData := nodeData) (x := x))
                                  have hTail := ih st1 hRec
                                  exact buildFrom_denoteAllFrom_finish (α := α) (g := g) (payload :=
                                    payload)
                                    (i := i) (x := x) (hi := hi) (τ := nOutShape)
                                    (nodeData := nodeData) (st1 := st1) (st' := st')
                                    (ctx := ctx) (vals0 := vals0) (input := input) hTail hEval hStep
                        · exact False.elim <| throw_bind_ne_ok (h := (by simpa [dif_neg hOut] using hBuild2))
                      · exact False.elim <| throw_bind_ne_ok (h := (by simpa [if_neg hShape] using hBuild1))

end Compiled
end Autograd
end Runtime
