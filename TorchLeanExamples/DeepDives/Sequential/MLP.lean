/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Models.MLP
public import TorchLean.Tensor.Context.Rational
public import TorchLean.NN.Sequential
public import TorchLean.NN.Layer.Activation
public import TorchLean.NN.Layer.Recurrent

/-!
# Closed MLP Model Facts

`model` is the MLP as an `nn.Sequential`. `eval_model` proves at every scalar that it evaluates to
`linear ∘ ReLU ∘ linear`, and `mlpOpspec_backward` that the input gradient `vjpOp` composes
from the operator specifications equals the input gradient of the hand-written `backward`.
-/

@[expose] public section

namespace TorchLean.Examples.Sequential.MLP

open TorchLean.Tensor
open TorchLean.MLP
open nn

/-- `Linear(inDim → hidDim)`, `ReLU`, `Linear(hidDim → outDim)` as an `nn.Sequential`, with weight
seeds 0 and 1 for the two linear layers. -/
def model (inDim hidDim outDim : Nat) : nn.Sequential [inDim] [outDim] :=
  (nn.Sequential.fromLayer (Layer.linear inDim hidDim 0)).comp
    ((nn.Sequential.fromLayer (Layer.op .relu)).comp
      (nn.Sequential.fromLayer (Layer.linear hidDim outDim 1)))

/-- At every scalar, the MLP `model` evaluated with the weights and biases of `l1` and `l2` is
`linear l2 ∘ ReLU ∘ linear l1`. -/
theorem eval_model {α : Type} [Storage α] [Context α] {inDim hidDim outDim : Nat}
    (l1 : Linear α inDim hidDim) (l2 : Linear α hidDim outDim)
    (x : Tensor α [inDim]) :
    (model inDim hidDim outDim).eval
        (.cons l1.weights (.cons l1.bias (.cons l2.weights (.cons l2.bias .nil)))) x =
      let z1 := Linear.forward l1 x
      let a1 := Tensor.relu z1
      Linear.forward l2 a1 := rfl

/-- The initial parameters of `model` are the two linear layers' initial states, weight seeds 0 and
1, in parameter order. -/
theorem initState_model (inDim hidDim outDim : Nat) :
    nn.Sequential.initState (model inDim hidDim outDim) =
      Pack.append (ss₁ := [[hidDim, inDim], [hidDim]])
        (ss₂ := [[outDim, hidDim], [outDim]])
        (Layer.linear inDim hidDim (weightSeed := 0)).initState
        (Layer.linear hidDim outDim (weightSeed := 1)).initState := rfl

/-- At every scalar, the backward pass that `vjpOp` composes from the operator VJPs is the input
gradient, the last component, of the hand-written `backward`. -/
theorem mlpOpspec_backward {α : Type} [Storage α] [Context α] {inDim hidDim outDim : Nat}
    (l1 : Linear α inDim hidDim) (l2 : Linear α hidDim outDim)
    (x : Tensor α [inDim]) (g : Tensor α [outDim]) :
    (vjpOp l1 l2).backward x g = (backward l1 l2 x g).2.2.2.2 := by
  rfl

end TorchLean.Examples.Sequential.MLP
