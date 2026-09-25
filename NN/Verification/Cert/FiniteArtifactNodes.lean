/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.IR.Semantics
public import NN.Spec.Core.Context.Real

/-! Real IR node equations for the finite vector linear/ReLU fragment. -/

public section

namespace NN.Verification.Cert.FiniteArtifact

open _root_.Spec TorchLean TorchLean.Tensor

/-- The real IR evaluator returns a vector input at its declared shape. -/
theorem evalNode_input {n : Nat} (payload : NN.IR.Payload ℝ) (x : Tensor ℝ [n])
    (values : Array (SomeTensor ℝ)) :
    NN.IR.Graph.evalNode payload ⟨[n], x⟩ values 0 ⟨0, #[], .input, [n]⟩ =
      .ok ⟨[n], x⟩ := by
  simp [NN.IR.Graph.evalNode, NN.IR.Graph.expectShape]

/-- A unary vector linear node applies the payload's exact-real affine map. -/
theorem evalNode_linear {n m : Nat} (payload : NN.IR.Payload ℝ) (input : SomeTensor ℝ)
    (values : Array (SomeTensor ℝ)) (id parent : Nat) (x : Tensor ℝ [n])
    (weights : Tensor ℝ [m, n]) (bias : Tensor ℝ [m])
    (hp : values[parent]? = some ⟨[n], x⟩)
    (hw : payload.linear? id = some ⟨m, n, weights, bias⟩) :
    NN.IR.Graph.evalNode payload input values id ⟨id, #[parent], .linear, [m]⟩ =
      .ok ⟨[m], Tensor.addSpec (matVecMulSpec weights x) bias⟩ := by
  simp [NN.IR.Graph.evalNode, NN.IR.Graph.unaryParentId, NN.IR.unaryParent?, hp,
    NN.IR.Graph.evalLinear, hw, NN.IR.Graph.expectShape, NN.IR.Graph.linearLeading,
    Shape.ofList, Shape.toList, Shape.concat,
    Pure.pure, Except.pure, Bind.bind, Except.bind]
  rfl

/-- A unary vector ReLU node uses the same real activation as the decoded program. -/
theorem evalNode_relu {n : Nat} (payload : NN.IR.Payload ℝ) (input : SomeTensor ℝ)
    (values : Array (SomeTensor ℝ)) (id parent : Nat) (x : Tensor ℝ [n])
    (hp : values[parent]? = some ⟨[n], x⟩) :
    NN.IR.Graph.evalNode payload input values id ⟨id, #[parent], .relu, [n]⟩ =
      .ok ⟨[n], Activation.reluSpec x⟩ := by
  simp [NN.IR.Graph.evalNode, NN.IR.Graph.unaryParentId, NN.IR.unaryParent?, hp,
    NN.IR.Graph.expectShape]

end NN.Verification.Cert.FiniteArtifact
