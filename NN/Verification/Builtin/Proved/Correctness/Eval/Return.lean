/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.Builtin.Proved.Correctness.Eval.NodeShape

/-!
# Lowered Forward Evaluation: Return Value Shape
-/

@[expose] public section

namespace NN.Verification.Builtin.Proved

open Spec TorchLean
open TorchLean.Tensor
open NN.IR

namespace Correctness

open NN.Verification.Builtin
-- Typed context indices come from `NN.Proofs.Autograd.Tape.Util.Idx`, the one place
-- `Idx` and `getIdx` are defined.
open Proofs (Idx getIdx)

  /-!
  Helper functions for the final "lowered forward = DSL forward" theorem.

  - `finalShapes g` is the list of available value shapes at the point where `g` returns.
    (It is the `ss` parameter of the `.ret` constructor reached by running through `.let1`.)
  - `outputIndex g` is the return index of `g`, but expressed at the `finalShapes g` context.
  -/

  /-- Shape context available after evaluating every binding in a forward let-chain. -/
  def finalShapes
      {α : Type} [TorchLean.Storage α]
      {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {out : Shape} :
      ForwardLetChain α paramShapes inShape ss out → List Shape
    | .ret _y => ss
    | .let1 _node gNext => finalShapes gNext

  /--
  Return index of a forward let-chain, expressed in the *final* context.

  As we traverse `.let1` nodes, the local context `ss` grows; this function returns the output
  index at the end of the chain (`finalShapes g`), so it can be used with the final `vals` array
  produced by `evalForwardLetChainVals`.
  -/
  def outputIndex
      {α : Type} [TorchLean.Storage α]
      {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {out : Shape} :
      (g : ForwardLetChain α paramShapes inShape ss out) → Idx (Ctx inShape (finalShapes g)) out
    | .ret y => y
    | .let1 _node gNext => outputIndex gNext

  /--
  The lowered graph's `outputId` agrees with the return index `outputIndex` of the source
  let-chain. The lowering pass records exactly the node index returned by the `.ret` case after
  threading through the `.let1` chain.
  -/
  theorem lowerForwardLetChain_outputId_eq_outputIndex_id
      {α : Type} [TorchLean.Storage α] [Context α]
      {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {out : Shape}
      (g : ForwardLetChain α paramShapes inShape ss out)
      (params : TorchLean.TensorPack α paramShapes)
      (c : NN.Verification.Builtin.LoweredIR α) :
      (lowerForwardLetChain (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss)
          (out := out) g params c).outputId
        =
      (outputIndex (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss)
        (out := out) g).id := by
    classical
    induction g generalizing c with
    | ret y =>
        simp [lowerForwardLetChain, outputIndex]
        rfl
    | @let1 ss₀ mid₀ out₀ node gNext ih =>
        simp [lowerForwardLetChain, outputIndex, ih]
        rfl

  /--
  `evalForwardLetChain` is `evalForwardLetChainVals` followed by selecting the return index
  `outputIndex`.

  This isolates “evaluate all SSA values” from “pick the output tensor”, which is useful in the
    final
  correctness statement.
  -/
  theorem evalForwardLetChain_eq_evalForwardLetChainVals_outputIndex
      {α : Type} [TorchLean.Storage α] [Context α]
      {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {out : Shape}
      (g : ForwardLetChain α paramShapes inShape ss out)
      (params : TorchLean.TensorPack α paramShapes)
      (vals : Array (Spec.SomeTensor α)) :
      evalForwardLetChain (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss)
        (out := out) g params vals
        =
      (do
        let vals' ←
          evalForwardLetChainVals (α := α) (paramShapes := paramShapes) (inShape := inShape)
            (ss := ss) (out := out)
            g params vals
        let v : Spec.SomeTensor α ←
          getValue? vals' ((outputIndex (α := α) (paramShapes := paramShapes)
            (inShape := inShape) (ss := ss) (out := out) g).id)
        if h : v.shape = out then
          pure (h ▸ v.tensor)
        else
          throw s!"TorchLeanVerified: expected shape {repr out}, got {repr v.shape}") := by
    classical
    induction g generalizing vals with
    | ret y =>
        -- Definitional: `evalForwardLetChainVals (.ret _) = pure vals`.
        rfl
    | @let1 ss₀ mid₀ out₀ node gNext ih =>
        cases hNode :
            evalNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss₀) (out :=
              mid₀)
              node params vals with
        | error e =>
            -- Short-circuiting on `Except.error` makes both sides definitional.
            simp [evalForwardLetChain, evalForwardLetChainVals, outputIndex, hNode]
            rfl
        | ok vOut =>
            have hIH := ih (vals := vals.push vOut)
            -- Reduce the outer `evalNode` bind and then apply the IH on the extended `vals`.
            simp [evalForwardLetChain, evalForwardLetChainVals, outputIndex, hNode, Pure.pure,
              Except.pure, Except.bind, bind, pure]
            cases hVals :
                evalForwardLetChainVals (α := α) (paramShapes := paramShapes) (inShape := inShape)
                  (ss := ss₀ ++ [mid₀]) (out := out₀) gNext params (vals.push vOut) with
            | error e =>
                simp [hVals] at hIH ⊢
                exact hIH
            | ok valsNext =>
                cases hGet :
                    getValue? valsNext
                      ((outputIndex (α := α) (paramShapes := paramShapes) (inShape := inShape)
                        (ss := ss₀ ++ [mid₀]) (out := out₀) gNext).id) with
                | error e =>
                    simp [hVals] at hIH ⊢
                    exact hIH
                | ok v =>
                    by_cases hShape : v.shape = out₀
                    · simp [hVals] at hIH ⊢
                      exact hIH
                    · simp [hVals] at hIH ⊢
                      exact hIH

  /--
  Shape-invariant for `evalForwardLetChainVals`.

  If the input value array has shapes `Ctx inShape ss`, then the result array has shapes
  `Ctx inShape (finalShapes g)` at the return point.
  -/
  theorem evalForwardLetChainVals_shapes_of_hShapes
      {α : Type} [TorchLean.Storage α] [Context α]
      {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {out : Shape}
      (g : ForwardLetChain α paramShapes inShape ss out)
      (params : TorchLean.TensorPack α paramShapes)
      (vals vals' : Array (Spec.SomeTensor α))
      (hShapes : shapesOfVals (α := α) vals = Ctx inShape ss)
      (hOk :
        evalForwardLetChainVals (α := α) (paramShapes := paramShapes) (inShape := inShape)
          (ss := ss) (out := out) g params vals =
          Except.ok vals') :
      shapesOfVals (α := α) vals' = Ctx inShape (finalShapes g) := by
    classical
    induction g generalizing vals vals' with
    | ret y =>
        simp [evalForwardLetChainVals] at hOk
        cases hOk
        simpa [finalShapes]
    | @let1 ss₀ mid₀ out₀ node gNext ih =>
        -- Unfold once and split on `evalNode`.
        cases hNode :
            evalNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss₀) (out :=
              mid₀)
              node params vals with
        | error e =>
            -- impossible: `hOk` claims the whole computation returned `ok`.
            simp [evalForwardLetChainVals, hNode] at hOk
            cases hOk
        | ok vOut =>
            have hvOutShape : vOut.1 = mid₀ :=
              evalNode_ok_shape_of_hShapes (α := α) (paramShapes := paramShapes) (inShape :=
                inShape)
                (ss := ss₀) (out := mid₀) node params vals hShapes (v := vOut) (by simp [hNode])
            have hShapes' : shapesOfVals (α := α) (vals.push vOut) = Ctx inShape (ss₀ ++ [mid₀]) :=
              by
              calc
                shapesOfVals (α := α) (vals.push vOut)
                    = shapesOfVals (α := α) vals ++ [vOut.1] :=
                      shapesOfVals_push (α := α) (vals := vals) (v := vOut)
                _ = Ctx inShape ss₀ ++ [vOut.1] := by simp [hShapes]
                _ = Ctx inShape (ss₀ ++ [mid₀]) := by simp [Ctx, hvOutShape, List.cons_append]
            have hOk' :
                evalForwardLetChainVals (α := α) (paramShapes := paramShapes) (inShape := inShape)
                    (ss := ss₀ ++ [mid₀]) (out := out₀)
                    gNext params (vals.push vOut)
                  =
                Except.ok vals' := by
              simpa [evalForwardLetChainVals, hNode, Pure.pure, Except.pure, Except.bind,
                Except.instMonad, bind, pure] using hOk
            -- Apply IH to the suffix.
            simpa [finalShapes, evalForwardLetChainVals, hNode] using
              ih (vals := vals.push vOut) (vals' := vals') (hShapes := hShapes') (hOk := hOk')
end Correctness

end NN.Verification.Builtin.Proved
