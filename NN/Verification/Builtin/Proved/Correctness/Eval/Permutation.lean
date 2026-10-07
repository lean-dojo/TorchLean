/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.Builtin.Proved.Correctness.Eval.Core

/-!
# Permutation IR Evaluation

Local semantics for axis permutation. The theorem assumes that the shared dynamic interpreter,
`Graph.permuteSomeTensor`, succeeded and returned the declared output shape; it then proves that
the complete IR node returns that value. Softmax's axis handling follows its tensor specification
directly and is proved separately in `Eval.Softmax`.
-/

@[expose] public section

namespace NN.Verification.Builtin.Proved

open Spec TorchLean
open TorchLean.Tensor
open NN.IR

namespace Correctness

namespace IRStep

/-- Local IR semantics for `permute`, using the shared dynamic-value permutation interpreter. -/
theorem evalAt_permute_eq
    {α : Type} [TorchLean.Storage α] [Context α]
    {s out : Shape} (perm : Array Nat) (x : Tensor α s) (vOut : Spec.SomeTensor α)
    (hPerm :
      Graph.permuteSomeTensor (α := α) (v := Spec.SomeTensor.mk (α := α) s x) perm = .ok vOut)
    (hShape : vOut.shape = out) :
    Graph.evalAt (α := α) (g := unaryGraphOut (.permute perm) s out)
        (payload := {})
        (input := Spec.SomeTensor.mk (α := α) s x)
        (vals := #[Spec.SomeTensor.mk (α := α) s x]) (i := 1)
      =
      Except.ok (Spec.SomeTensor.mk (α := α) out (hShape ▸ vOut.tensor)) := by
  simp [Graph.evalAt, Graph.evalNode, Graph.normalizeNodeOutput, unaryGraphOut, unaryNodeOut,
    Graph.getNode,
    Graph.unaryParentId, unaryParent?, Bind.bind, Except.bind, Pure.pure, Except.pure]
  rw [hPerm]
  simp [hShape]

end IRStep

end Correctness

end NN.Verification.Builtin.Proved
