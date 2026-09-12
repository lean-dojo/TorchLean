/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.Builtin.Proved.Correctness.Eval.Core

/-!
# Transpose IR Evaluation

Local semantics for swapping any two tensor axes. The graph operation is rank-polymorphic; axis
validity and the resulting shape are checked by `OpContracts.inferTransposeOutShape`.
-/

@[expose] public section

namespace NN.Verification.Builtin.Proved

open Spec TorchLean
open NN.IR

namespace Correctness
namespace IRStep

/-- Local IR semantics for swapping two arbitrary axes. -/
theorem evalAt_transpose_eq
    {α : Type} [TorchLean.Storage α] [Context α]
    {inputShape outputShape : Shape}
    (axis₁ axis₂ : Nat)
    (input : Tensor α inputShape)
    (perm : Array Nat)
    (output : Spec.SomeTensor α)
    (hPerm : OpContracts.transposePerm inputShape.rank axis₁ axis₂ = .ok perm)
    (hEval : Graph.permuteSomeTensor (α := α)
      (Spec.SomeTensor.mk (α := α) inputShape input) perm = .ok output)
    (hShape : output.shape = outputShape) :
    Graph.evalAt (α := α)
        (g := unaryGraphOut (.transpose axis₁ axis₂) inputShape outputShape)
        (payload := {})
        (input := Spec.SomeTensor.mk (α := α) inputShape input)
        (vals := #[Spec.SomeTensor.mk (α := α) inputShape input]) (i := 1)
      = Except.ok
          (Spec.SomeTensor.mk (α := α) outputShape (hShape ▸ output.tensor)) := by
  have hExpect :
      Graph.expectShape (α := α) (expected := outputShape) output =
        Except.ok (hShape ▸ output.tensor) := by
    cases output with
    | mk shape tensor =>
      change shape = outputShape at hShape
      subst outputShape
      simp [Graph.expectShape, Pure.pure, Except.pure]
  simp [Graph.evalAt, Graph.evalNode, Graph.normalizeNodeOutput, unaryGraphOut, unaryNodeOut,
    Graph.getNode, Graph.getNode?, Graph.unaryParentId, unaryParent?, hPerm, hEval, hExpect,
    Bind.bind, Except.bind, Pure.pure, Except.pure]

end IRStep
end Correctness
end NN.Verification.Builtin.Proved
