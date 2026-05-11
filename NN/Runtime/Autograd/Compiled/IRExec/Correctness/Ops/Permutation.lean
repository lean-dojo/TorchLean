/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Compiled.IRExec.Correctness.SemanticEquivalenceCommon

set_option linter.unusedSimpArgs false
set_option linter.unnecessarySimpa false

/-!
# Permutation Operators

Semantic-preservation lemmas for permutation-style operators in the IR-to-compiled-runtime bridge.

This file covers the IR `.permute perm` node kind. The compiler lowers a permutation to a
swap-depth program (a list of axis swaps). The proof below mirrors that lowering: it validates the
permutation witness and computed swaps, constructs the compiled forward closure in terms of
`applySwapsTensor`, then rewrites the IR evaluator's `permuteDVal` to the same swap-based
implementation via `permuteDVal_eq_applySwapsTensor`.

Build note: permutation proofs are slow because a permutation changes both tensor data and the type
level shape. The proof therefore has to relate a dynamic IR permutation to the compiled swap program
while keeping casts proof-irrelevant. The swap-program lemmas should stay small and
avoid redoing permutation arithmetic inside the compiler branch proof.
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

/-- Semantic-preservation lemma for `.permute perm` lowering. -/
theorem buildFrom_denoteAllFrom_permute
    {α : Type} [Context α] [DecidableEq Shape]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : GraphData α Unit [inShape] ss) (i : Nat) (st' : State α inShape)
    (x : Tensor α inShape) (n : NN.IR.Node) (perm : List Nat)
    (hN : g.getNode i = .ok n) (hk : n.kind = .permute perm) (hi : i < g.nodes.size)
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
  simp (config := { failIfUnchanged := false }) [hi, hN, hk] at hBuild
  -- Parents must be a singleton list.
  cases hp : n.parents with
  | nil =>
      simp [hp, throw_eq_error, Except.instMonad, Except.bind, Except.pure] at hBuild
      try cases hBuild
  | cons pId ps =>
      cases ps with
      | cons _ _ =>
          simp [hp, throw_eq_error, Except.instMonad, Except.bind, Except.pure] at hBuild
          try cases hBuild
      | nil =>
          -- `getNode pId` must succeed.
          cases hP : g.getNode pId with
          | error msg =>
              simp [hp, hP, throw_eq_error, Except.instMonad, Except.bind, Except.pure] at hBuild
              try cases hBuild
          | ok pNode =>
              simp (config := { failIfUnchanged := false }) [hp, hP] at hBuild
              -- Parent index must typecheck.
              cases hIdx : mkIdx (inShape := inShape) (ss := ss) pId pNode.outShape with
              | error msg =>
                  simp [hIdx] at hBuild
                  try cases hBuild
              | ok ip =>
                  simp [hIdx] at hBuild
                  -- Permutation must be valid and swap computation must succeed.
                  cases hPerm : Spec.Shape.permute? pNode.outShape perm with
                  | none =>
                      simp [hPerm] at hBuild
                      try cases hBuild
                  | some expected =>
                      simp [hPerm] at hBuild
                      cases hSwaps :
                          NN.IR.Graph.swapDepthsForPerm perm (Shape.rank pNode.outShape) with
                      | error msg =>
                          simp [hSwaps] at hBuild
                          try cases hBuild
                      | ok swaps =>
                          simp [hSwaps] at hBuild
                          let sFinal : Shape := swapShapeBySwaps pNode.outShape swaps
                          by_cases hFinal : sFinal = expected
                          · simp [sFinal, hFinal] at hBuild
                            by_cases hOut : expected = n.outShape
                            · simp [hOut] at hBuild
                              let nodeData : NodeData α Unit ([inShape] ++ ss) n.outShape :=
                                mkFwdNode (α := α) (Γ := [inShape] ++ ss) (τ := n.outShape) (fun ctx =>
                                  let x := getIdx (α := α) (xs := ctx) ip
                                  let y : Tensor α sFinal :=
                                    applySwapsTensor (α := α) (s := pNode.outShape) (swaps := swaps) x
                                  let yExpected : Tensor α expected := Tensor.castShape y hFinal
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
                                  vals0[pId]! =
                                    NN.IR.DVal.mk (α := α) pNode.outShape
                                      (getIdx (α := α) (xs := ctx) ip) := by
                                simpa [vals0, ctx] using
                                  (denoteAllState_get_mkIdx (inShape := inShape) (ss := ss)
                                    (gd := gd) (x := x) (pid := pId) (s := pNode.outShape) (idx := ip)
                                    hIdx)
                              have hEval :
                                  NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
                                      (input := input) (vals := vals0) (i := i) =
                                    .ok (NN.IR.DVal.mk (α := α) n.outShape (nodeData.forward ctx ())) := by
                                -- `evalAt` uses `permuteDVal`; rewrite it to swaps + `applySwapsTensor`.
                                have hPermute :
                                    NN.IR.Graph.permuteDVal (α := α)
                                        (v := NN.IR.DVal.mk (α := α) pNode.outShape
                                          (getIdx (α := α) (xs := ctx) ip)) perm =
                                      .ok
                                        (NN.IR.DVal.mk (α := α) (swapShapeBySwaps pNode.outShape swaps)
                                          (applySwapsTensor (α := α) (s := pNode.outShape) (swaps := swaps)
                                            (getIdx (α := α) (xs := ctx) ip))) := by
                                  simpa using
                                    (permuteDVal_eq_applySwapsTensor (α := α)
                                      (t := getIdx (α := α) (xs := ctx) ip) (perm := perm)
                                      (expected := expected) (swaps := swaps) hPerm (by
                                        -- `simp` normalizes `.ok`/`.error` to `Except.ok`/`Except.error`.
                                        simpa [NN.IR.Graph.swapDepthsForPerm] using hSwaps))
                                -- Now simplify `evalAt` with the computed permutation and parent lookup.
                                have hShape : swapShapeBySwaps pNode.outShape swaps = n.outShape := by
                                  simpa [sFinal] using (hFinal.trans hOut)
                                -- Rewrite the `permuteDVal` call to `.ok _` so the monadic bind reduces.
                                have hPermute' :
                                    NN.IR.Graph.permuteDVal (α := α)
                                        (v := (⟨pNode.outShape, getIdx (α := α) (xs := ctx) ip⟩ : NN.IR.DVal α))
                                        perm =
                                      .ok
                                        (NN.IR.DVal.mk (α := α) (swapShapeBySwaps pNode.outShape swaps)
                                          (applySwapsTensor (α := α) (s := pNode.outShape) (swaps := swaps)
                                            (getIdx (α := α) (xs := ctx) ip))) := by
                                  simpa [NN.IR.DVal.mk] using hPermute
                                -- `simp`/`rw` are syntax-sensitive; package `hPermute'` in the implicit-argument
                                -- form that actually appears inside the `evalAt` do-block.
                                have hPermute0 :
                                    NN.IR.Graph.permuteDVal (α := α) (v := (⟨pNode.outShape, getIdx ctx ip⟩ : NN.IR.DVal α))
                                        perm =
                                      .ok
                                        (NN.IR.DVal.mk (α := α) (swapShapeBySwaps pNode.outShape swaps)
                                          (applySwapsTensor (α := α) (s := pNode.outShape) (swaps := swaps)
                                            (getIdx ctx ip))) := by
                                  simpa [NN.IR.DVal.mk] using hPermute'
                                -- Expand `evalAt` to the permute branch, rewrite `permuteDVal` by `hPermute'`,
                                -- then discharge the dependent shape check via `hShape`.
                                simp (config := { failIfUnchanged := false })
                                  [NN.IR.Graph.evalAt, hN, hk, hp, hGet, throw_eq_error]
                                -- Rewrite the `permuteDVal` call to its computed `.ok` value, reduce the
                                -- `Except` do-block, and select the success branch using `hShape`.
                                erw [hPermute0]
                                simp (config := { failIfUnchanged := false })
                                  [Except.instMonad, Except.bind, Except.pure]
                                -- The remaining conditional is a dependent `if` (`dite`); `dif_pos` picks the
                                -- success branch and carries the proof `hShape` into the cast.
                                rw [dif_pos hShape]
                                simp [nodeData, sFinal, mkFwdNode, Tensor.cast_shape_trans,
                                  Tensor.eqRec_eq_cast_shape, Tensor.castShape, throw_eq_error]
                                -- Close the remaining `pure _ = .ok _` goal by unfolding the `Except` instance.
                                with_unfolding_all rfl
                              exact buildFrom_denoteAllFrom_unary_exact (α := α) (g := g) (payload := payload)
                                (gd := gd) (i := i) (st' := st') (x := x) (hi := hi)
                                (τ := n.outShape) (nodeData := nodeData) hTail hEval
                            · simp [hOut] at hBuild
                              try cases hBuild
                          · simp [sFinal, hFinal] at hBuild
                            try cases hBuild

end Compiled
end Autograd
end Runtime
