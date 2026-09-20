/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.Builtin.Proved.Correctness.Eval.Denote
public import NN.Verification.Builtin.Proved.Correctness.Eval.Return

/-!
# Lowered Forward Evaluation: End-to-End Correctness
-/

@[expose] public section

namespace NN.Verification.Builtin.Proved

open Spec TorchLean
open TorchLean.Tensor
open NN.IR

namespace Correctness

open NN.Verification.Builtin

  /--
  **Main lowering correctness theorem (verified forward fragment).**

  In words: lowering a first-order forward program `p` into the verifier IR and then
  evaluating the lowered graph yields the same output as directly evaluating `p` with
  `evalForward`.
  -/
  theorem runForwardIR_eq_evalForward
    {α : Type} [TorchLean.Storage α] [Context α]
    {paramShapes : List Shape} {inShape outShape : Shape}
    (p : ForwardProgram α paramShapes inShape outShape)
    (params : TorchLean.TensorPack α paramShapes)
    (x : Tensor α inShape) :
    runForwardIR (α := α) (inShape := inShape) (outShape := outShape)
        (c := lowerForwardProgramToIR (α := α) (paramShapes := paramShapes) (inShape := inShape)
          (outShape := outShape) p params)
        x
      =
    evalForward (α := α) (paramShapes := paramShapes) (inShape := inShape) (outShape := outShape) p
      params x := by
    classical
    -- Evaluate `lowerForwardProgramToIR` via the IR semantics, and rewrite it to the DSL evaluator.
    let inputVal : Spec.SomeTensor α := Spec.SomeTensor.mk (α := α) inShape x
    let inputNode : NN.IR.Node := { id := 0, parents := #[], kind := .input, outShape := inShape }
    let c0 : NN.Verification.Builtin.LoweredIR α :=
      { graph := { nodes := #[inputNode] }, ps := {}, inputId := 0, outputId := 0 }
    have hWF :
        (lowerForwardProgramToIR (α := α) (paramShapes := paramShapes) (inShape := inShape)
          (outShape := outShape) p params).graph.wellFormed = true :=
      lowerForwardProgramToIR_wellFormed (α := α) (paramShapes := paramShapes) (inShape := inShape)
        (outShape := outShape) p params
    -- Unfold the lowered evaluator down to the IR `denoteAllFrom` suffix, then apply the
    -- correctness lemma.
    -- The input node is always id=0, so we start the suffix evaluation at `i=1` with
    -- `vals=[inputVal]`.
    have hDenote :
        (NN.IR.Graph.denoteAllFrom (α := α)
            (g := (lowerForwardProgramToIR (α := α) (paramShapes := paramShapes)
              (inShape := inShape) (outShape := outShape) p params).graph)
            (payload := payloadOfParamStore (α := α)
              (lowerForwardProgramToIR (α := α) (paramShapes := paramShapes) (inShape := inShape)
                (outShape := outShape) p params).ps)
            (input := inputVal)
            (i := 1) (vals := #[inputVal]))
          =
        evalForwardLetChainVals (α := α) (paramShapes := paramShapes) (inShape := inShape)
          (ss := []) (out := outShape) p params #[inputVal] := by
      -- `lowerForwardProgramToIR` is `lowerForwardLetChain` starting from `c0`, so apply the lemma
      -- at `c=c0` and `vals=[inputVal]`.
      have hSize0 : (#[inputVal] : Array (Spec.SomeTensor α)).size = c0.graph.nodes.size := by
        simp [c0]
      have hShapes0 :
          shapesOfVals (α := α) (#[inputVal] : Array (Spec.SomeTensor α)) = Ctx inShape [] := by
        simp [shapesOfVals, Ctx, inputVal]
      simpa [lowerForwardProgramToIR, c0, inputNode, inputVal, Spec.SomeTensor.mk] using
        denoteAllFrom_lowerForwardLetChain_eq_evalForwardLetChainVals (α := α)
          (paramShapes := paramShapes) (inShape := inShape) (ss := []) (out := outShape)
        (g := p) (params := params) (c := c0) (x := x) (vals := #[inputVal]) hSize0 hShapes0
    -- Unfold both front-end evaluators, then rewrite both sides to a shared
    -- `evalForwardLetChainVals` computation.
    have hOutId :
        (lowerForwardProgramToIR (α := α) (paramShapes := paramShapes) (inShape := inShape)
          (outShape := outShape) p params).outputId
          =
        (outputIndex (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := [])
          (out := outShape) p).id := by
      simpa [lowerForwardProgramToIR] using
        lowerForwardLetChain_outputId_eq_outputIndex_id (α := α) (paramShapes := paramShapes)
          (inShape := inShape) (ss := []) (out := outShape) p params c0

    -- Rewrite `Graph.denoteAll` to start at `i=1` with the already-evaluated input.
    have hDenoteAll0 :
        NN.IR.Graph.denoteAll (α := α)
            (g := (lowerForwardProgramToIR (α := α) (paramShapes := paramShapes)
              (inShape := inShape) (outShape := outShape) p params).graph)
            (payload := payloadOfParamStore (α := α)
              (lowerForwardProgramToIR (α := α) (paramShapes := paramShapes) (inShape := inShape)
                (outShape := outShape) p params).ps)
            (input := inputVal)
          =
        NN.IR.Graph.denoteAllFrom (α := α)
            (g := (lowerForwardProgramToIR (α := α) (paramShapes := paramShapes)
              (inShape := inShape) (outShape := outShape) p params).graph)
            (payload := payloadOfParamStore (α := α)
              (lowerForwardProgramToIR (α := α) (paramShapes := paramShapes) (inShape := inShape)
                (outShape := outShape) p params).ps)
            (input := inputVal) (i := 1) (vals := #[inputVal]) := by
      -- `denoteAll` runs `denoteAllFrom` from `i=0` with `vals=[]`.
      simp (config := { zeta := false }) [NN.IR.Graph.denoteAll, hWF]
      -- Unfold one step at `i=0`: the `input` node deterministically yields `inputVal`.
      have h0 :
          (0 : Nat) <
            (lowerForwardProgramToIR (α := α) (paramShapes := paramShapes) (inShape := inShape)
              (outShape := outShape)
                p params).graph.nodes.size := by
        have h0c : (0 : Nat) < c0.graph.nodes.size := by
          simp [c0]
        have hLe :
            c0.graph.nodes.size ≤
              (lowerForwardProgramToIR (α := α) (paramShapes := paramShapes) (inShape := inShape)
                (outShape := outShape)
                  p params).graph.nodes.size := by
          simpa [lowerForwardProgramToIR] using
            lowerForwardLetChain_nodesSize_le (α := α) (paramShapes := paramShapes)
              (inShape := inShape) (ss := []) (out := outShape) (g := p) (params := params)
                (c := c0)
        exact Nat.lt_of_lt_of_le h0c hLe
      have hGet0 :
          NN.IR.Graph.getNode
              (g := (lowerForwardProgramToIR (α := α) (paramShapes := paramShapes)
                (inShape := inShape) (outShape := outShape)
                  p params).graph)
              0
            =
          pure inputNode := by
        have hi : (0 : Nat) < c0.graph.nodes.size := by
          simp [c0]
        -- `lowerForwardProgramToIR` is `lowerForwardLetChain` starting from `c0`; indices below
        -- `c0`'s size are preserved.
        simpa [lowerForwardProgramToIR, c0, inputNode, NN.IR.Graph.getNode, NN.IR.Graph.getNode?,
          BEq.beq] using
          lowerForwardLetChain_getNode_lt (α := α) (paramShapes := paramShapes) (inShape := inShape)
            (ss := []) (out := outShape) (g := p) (params := params) (c := c0) (i := 0) (hi := hi)
      have hEval0 :
          NN.IR.Graph.evalAt (α := α)
            (g := (lowerForwardProgramToIR (α := α) (paramShapes := paramShapes)
              (inShape := inShape) (outShape := outShape)
                p params).graph)
            (payload := payloadOfParamStore (α := α)
              (lowerForwardProgramToIR (α := α) (paramShapes := paramShapes) (inShape := inShape)
                (outShape := outShape) p params).ps)
            (input := inputVal) (vals := #[]) (i := 0)
          =
          Except.ok inputVal := by
        -- `getNode 0` returns the input node, and the `.input` branch deterministically returns
        -- `inputVal`.
        simp [NN.IR.Graph.evalAt, NN.IR.Graph.evalNode, NN.IR.Graph.normalizeNodeOutput, hGet0,
          inputNode, inputVal, NN.IR.Graph.expectShape,
          Bind.bind, Pure.pure, Except.pure, Except.bind]
      rw [NN.IR.Graph.denoteAllFrom.eq_1, dite_eq_left h0]
      rw [hEval0]
      simp
      have hPushEq : (#[].push inputVal : Array (Spec.SomeTensor α)) = #[inputVal] := by
        rfl
      exact congrArg
        (fun vals =>
          NN.IR.Graph.denoteAllFrom (α := α)
            (g := (lowerForwardProgramToIR (α := α) (paramShapes := paramShapes)
              (inShape := inShape) (outShape := outShape) p params).graph)
            (payload := payloadOfParamStore (α := α)
              (lowerForwardProgramToIR (α := α) (paramShapes := paramShapes) (inShape := inShape)
                (outShape := outShape) p params).ps)
            (input := inputVal) (i := 1) (vals := vals))
        hPushEq

    -- Now both `runForwardIR` and `evalForward` can be expressed via `evalForwardLetChainVals`.
    -- We finish by case-splitting on `evalForwardLetChainVals` and simplifying the shared
    -- `if`/lookup logic.
    -- (The "out of bounds" / "shape mismatch" branches are unreachable under the well-typedness
    -- invariants we establish.)
    have hEvalForward :
        evalForward (α := α) (paramShapes := paramShapes) (inShape := inShape)
          (outShape := outShape)
          p params x
          =
        (do
          let vals' ←
            evalForwardLetChainVals (α := α) (paramShapes := paramShapes) (inShape := inShape)
              (ss := []) (out := outShape) p params #[inputVal]
          let v : Spec.SomeTensor α ← getValue? vals' ((outputIndex (α := α)
            (paramShapes := paramShapes) (inShape := inShape) (ss := []) (out := outShape) p).id)
          if h : v.shape = outShape then
            pure (h ▸ v.tensor)
          else
            throw s!"TorchLeanVerified: expected shape {repr outShape}, got {repr v.shape}") := by
      simpa [evalForward, inputVal] using
        (evalForwardLetChain_eq_evalForwardLetChainVals_outputIndex (α := α)
          (paramShapes := paramShapes) (inShape := inShape)
          (ss := []) (out := outShape) (g := p) (params := params) (vals := #[inputVal]))

    -- Rewrite the RHS, and unfold the lowered evaluator down to the same `evalForwardLetChainVals`.
    rw [hEvalForward]
    rw [NN.Verification.Builtin.runForwardIR, NN.IR.Graph.denote, hOutId]

    -- The remaining LHS still mentions IR `denoteAll`; rewrite it using the established equalities.
    have hDenoteAll0' :
        NN.IR.Graph.denoteAll (α := α)
            (g := (lowerForwardProgramToIR (α := α) (paramShapes := paramShapes)
              (inShape := inShape) (outShape := outShape) p params).graph)
            (payload := payloadOfParamStore (α := α)
              (lowerForwardProgramToIR (α := α) (paramShapes := paramShapes) (inShape := inShape)
                (outShape := outShape) p params).ps)
            (input := Spec.SomeTensor.mk (α := α) inShape x)
          =
        NN.IR.Graph.denoteAllFrom (α := α)
            (g := (lowerForwardProgramToIR (α := α) (paramShapes := paramShapes)
              (inShape := inShape) (outShape := outShape) p params).graph)
            (payload := payloadOfParamStore (α := α)
              (lowerForwardProgramToIR (α := α) (paramShapes := paramShapes) (inShape := inShape)
                (outShape := outShape) p params).ps)
            (input := Spec.SomeTensor.mk (α := α) inShape x) (i := 1)
            (vals := #[Spec.SomeTensor.mk (α := α) inShape x]) := by
      simpa [inputVal] using hDenoteAll0
    have hDenote' :
        NN.IR.Graph.denoteAllFrom (α := α)
            (g := (lowerForwardProgramToIR (α := α) (paramShapes := paramShapes)
              (inShape := inShape) (outShape := outShape) p params).graph)
            (payload := payloadOfParamStore (α := α)
              (lowerForwardProgramToIR (α := α) (paramShapes := paramShapes) (inShape := inShape)
                (outShape := outShape) p params).ps)
            (input := Spec.SomeTensor.mk (α := α) inShape x) (i := 1)
            (vals := #[Spec.SomeTensor.mk (α := α) inShape x])
          =
        evalForwardLetChainVals (α := α) (paramShapes := paramShapes) (inShape := inShape)
          (ss := []) (out := outShape) p params
          #[Spec.SomeTensor.mk (α := α) inShape x] := by
      simpa [inputVal, Except.bind, Except.pure, Pure.pure] using hDenote
    -- Rewrite `Graph.denoteAll` → `denoteAllFrom` → `evalForwardLetChainVals` on the lowered side.
    rw [hDenoteAll0', hDenote']

    -- Now both sides bind the same `evalForwardLetChainVals`; split on that result.
    cases hVals :
        evalForwardLetChainVals (α := α) (paramShapes := paramShapes) (inShape := inShape)
          (ss := []) (out := outShape) p params #[inputVal] with
    | error e =>
        -- Both sides are `Except.error e` because the first monadic bind fails.
        simp [Bind.bind, Except.bind]
    | ok vals' =>
        -- Establish that the output index is in-bounds and has the expected shape.
        have hShapes' :
            shapesOfVals (α := α) vals' = Ctx inShape (finalShapes (α := α)
              (paramShapes := paramShapes) (inShape := inShape)
              (ss := []) (out := outShape) p) := by
          exact evalForwardLetChainVals_shapes_of_hShapes (α := α) (paramShapes := paramShapes)
            (inShape := inShape)
            (ss := []) (out := outShape) (g := p) (params := params) (vals := #[inputVal]) (vals' :=
              vals')
            (hShapes := by simp [shapesOfVals, Ctx, inputVal]) (hOk := by simp [hVals])
        let outIdx := outputIndex (α := α) (paramShapes := paramShapes) (inShape := inShape)
          (ss := []) (out := outShape) p
        let outVal := packedAt vals' outIdx hShapes'
        have hOutSome : vals'[outIdx.id]? = some outVal := by
          simpa [outVal] using getElem?_eq_some_packedAt vals' outIdx hShapes'
        have hOutShape : outVal.shape = outShape := by
          exact packedAt_shape vals' outIdx hShapes'
        simp [getValue?, outIdx, hOutSome, hOutShape, Bind.bind, Except.bind,
          Pure.pure, Except.pure]

end Correctness

end NN.Verification.Builtin.Proved
