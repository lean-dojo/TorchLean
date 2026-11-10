/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Sequential.Composition
public import TorchLean.NN.Layer.Activation
public import TorchLean.NN.Layer.Recurrent

/-!
# A linear classifier on any feature model

`classifier features o` flattens the output of an arbitrary `nn.Sequential` feature model and
applies a linear head with `o` outputs. `eval_classifier` proves, for every scalar with `Storage`
and `Context`, that its evaluation is `TorchLean.Linear.forward` of the head on the flattened
feature output, with the features' state followed by the head's weight and bias.
-/

@[expose] public section

open scoped TorchLean.nn.Sequential

namespace TorchLean.nn.Sequential

/-- A linear classifier with `outputSize` outputs on the flattened output of `features`. -/
def classifier {σ φ : Shape} (features : nn.Sequential σ φ) (outputSize : Nat)
    (weightSeed : Nat := 0) : nn.Sequential σ [outputSize] :=
  features >>> .fromLayer (Layer.op (.reshape (Shape.size_vector φ.size).symm)) >>>
    .fromLayer (Layer.linear φ.size outputSize weightSeed)

/-- For every scalar with `Storage` and `Context`, the classifier evaluates to `Linear.forward` of
the head on the flattened output of `features`, when its state lists the features' state and then
the head's weight and bias. -/
theorem eval_classifier {α : Type} [Storage α] [Context α] {σ φ : Shape} {outputSize : Nat}
    (weightSeed : Nat) (features : nn.Sequential σ φ)
    (state : TensorPack α (nn.Sequential.stateShapes features))
    (head : Linear α φ.size outputSize) (x : Tensor α σ) :
    (classifier features outputSize weightSeed).eval
        ((nn.Sequential.stateShapes_comp features _).symm ▸
          state.append (ss₂ := [[outputSize, φ.size], [outputSize]])
            (.cons head.weights (.cons head.bias .nil))) x =
      Linear.forward head
        (Tensor.reshape (features.eval state x) [_] (Shape.size_vector _).symm) :=
  nn.Sequential.eval_comp features
    (nn.Sequential.fromLayer (Layer.op (.reshape (Shape.size_vector φ.size).symm) rfl) >>>
      nn.Sequential.fromLayer (Layer.linear φ.size outputSize weightSeed))
    state (.cons head.weights (.cons head.bias .nil)) x

end TorchLean.nn.Sequential
