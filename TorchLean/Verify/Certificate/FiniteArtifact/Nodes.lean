/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.IR.Eval
public import TorchLean.Tensor.Op.Table.Elementwise
public import TorchLean.Tensor.Op.Table.Linear
public import TorchLean.Tensor.Context.Real

/-! Real operator-graph node equations for the finite vector linear/ReLU fragment. -/

public section

namespace TorchLean.Certificate.FiniteArtifact

open TorchLean.Tensor
open IR (Node Ref Payload)

/-- An input node returns its vector graph input at its declared shape. -/
theorem evalNode_input {n : Nat} (payload : Payload ℝ) (x : Tensor ℝ [n])
    (earlier : Array (List (Op.Value ℝ))) (id : Nat) :
    (Node.input 0 (.value [n])).eval payload [⟨.value [n], x⟩] earlier id =
      .ok [⟨.value [n], x⟩] := by
  simp [Node.eval, Port.read_self, pure, Except.pure, bind, Except.bind]

/-- A vector linear node whose references read `weights`, `bias` and `x` applies the exact-real
affine map. -/
theorem evalNode_linear {n m : Nat} (payload : Payload ℝ) (inputs : List (Op.Value ℝ))
    (earlier : Array (List (Op.Value ℝ))) (id : Nat) (rW rb rx : Ref)
    (weights : Tensor ℝ [m, n]) (bias : Tensor ℝ [m]) (x : Tensor ℝ [n])
    (hw : rW.read earlier = .ok ⟨.value [m, n], weights⟩)
    (hb : rb.read earlier = .ok ⟨.value [m], bias⟩)
    (hx : rx.read earlier = .ok ⟨.value [n], x⟩) :
    (Node.op ⟨_, _, Op.linear (lead := []) (inDim := n) (outDim := m)⟩
        #[rW, rb, rx]).eval payload inputs earlier id =
      .ok [⟨.value [m], Tensor.contract weights x _ _ [m] .matVec + bias⟩] :=
  IR.evalOp_eq _ rfl rfl Op.rule?_linear (payload := payload) (id := id)
    (values := [⟨.value [m, n], weights⟩, ⟨.value [m], bias⟩, ⟨.value [n], x⟩])
    (xs := (weights, bias, x, ()))
    (by simp [hw, hb, hx, pure, Except.pure, bind, Except.bind])
    (by simp [Values.ofErased, Port.read_self, bind, Except.bind, pure, Except.pure])

/-- A vector ReLU node whose reference reads `x` uses the same real activation as the decoded
program. -/
theorem evalNode_relu {n : Nat} (payload : Payload ℝ) (inputs : List (Op.Value ℝ))
    (earlier : Array (List (Op.Value ℝ))) (id : Nat) (r : Ref) (x : Tensor ℝ [n])
    (hx : r.read earlier = .ok ⟨.value [n], x⟩) :
    (Node.op ⟨_, _, Op.relu (s := [n])⟩ #[r]).eval payload inputs earlier id =
      .ok [⟨.value [n], Tensor.relu x⟩] :=
  IR.Node.eval_unary _ rfl rfl Op.rule?_relu hx

end TorchLean.Certificate.FiniteArtifact
