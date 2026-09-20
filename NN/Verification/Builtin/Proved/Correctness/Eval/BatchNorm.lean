/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.Builtin.Proved.Correctness.Eval.PayloadOps

/-!
# BatchNorm IR Evaluation

The IR represents inference-time BatchNorm by a channel axis and a channel count. This file states
the payload contract for that representation without fixing the tensor rank or choosing a layout.
After shape inference identifies the channel axis, evaluation decomposes the input into leading
axes, the channel axis, and trailing axes, then applies `Spec.batchNormInference` independently to
each leading slice.
-/

@[expose] public section

namespace NN.Verification.Builtin.Proved

open Spec TorchLean
open NN.IR

namespace Correctness.IRStep

/--
Evaluation of inference-time BatchNorm at an arbitrary channel axis.

The conditional on the right is the checked cast from the dynamically shaped IR value to the
typed tensor expected by `Spec.batchNormInference`. For every shape accepted by
`inferBatchNormEvalOutShape`, this equality records the complete payload-backed computation.
-/
theorem evalBatchNorm_eq
    {α : Type} [TorchLean.Storage α] [Context α]
    (id channelAxis channels : Nat)
    (params : BatchNormEvalParams α)
    (x : Spec.SomeTensor α)
    (leading spatial : Shape)
    (hInfer : OpContracts.inferBatchNormEvalOutShape channelAxis channels x.shape = .ok x.shape)
    (hChannels : params.c = channels)
    (hLeading : Shape.ofList (x.shape.toList.take channelAxis) = leading)
    (hSpatial : Shape.ofList (x.shape.toList.drop (channelAxis + 1)) = spatial)
    (hInput : x.shape = leading.concat (.dim params.c spatial)) :
    Graph.evalBatchNorm (α := α)
        (payload := singletonBatchNormEvalPayload (α := α) id params)
        id channelAxis channels x =
      .ok (Spec.SomeTensor.ofTensor <|
        Tensor.mapLeading leading
          (fun sample => Spec.batchNormInference sample params.mean params.var params.gamma
            params.beta params.eps)
          (hInput ▸ x.tensor)) := by
  -- Substitute the derived leading and spatial shapes instead of rewriting with them. The
  -- evaluator's shape check sits inside a dependent cast, so abstracting an occurrence of `spatial`
  -- has no type-correct motive.
  subst hLeading
  subst hSpatial
  unfold Graph.evalBatchNorm
  rw [hInfer]
  simp only [Bind.bind, Except.bind]
  simp only [singletonBatchNormEvalPayload, singletonAt_self]
  simp [hChannels]
  split
  · congr
  · contradiction

/-- A missing BatchNorm payload is rejected before normalization is evaluated. -/
theorem evalBatchNorm_missing_payload
    {α : Type} [TorchLean.Storage α] [Context α]
    (payload : Payload α) (id channelAxis channels : Nat)
    (x : Spec.SomeTensor α)
    (hInfer : OpContracts.inferBatchNormEvalOutShape channelAxis channels x.shape = .ok x.shape)
    (hMissing : payload.batchNormEval? id = none) :
    Graph.evalBatchNorm (α := α) (payload := payload) id channelAxis channels x =
      .error s!"IR eval: missing batch_norm_eval payload for node {id}" := by
  simp [Graph.evalBatchNorm, hInfer, hMissing]
  rfl

end Correctness.IRStep

end NN.Verification.Builtin.Proved
