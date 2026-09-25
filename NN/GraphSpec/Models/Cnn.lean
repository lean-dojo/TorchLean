/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.GraphSpec.Primitives.Spatial
public import NN.GraphSpec.Chain.Primitives
public import NN.GraphSpec.Chain.Semantics

/-!
# GraphSpec Convolutional Classifier

A classifier takes a typed feature chain, flattens its output, and applies a linear head.
The feature chain determines its own depth, channel widths, spatial geometry, and parameter
shapes. Its stages can use different convolution and pooling configurations.
-/

@[expose] public section

namespace NN.GraphSpec.Models

open Spec TorchLean

/--
Attach a linear classifier to an arbitrary feature chain.

Parameters remain in feature-chain order, followed by the head's weight and bias. An identity
feature chain gives a linear classifier on the flattened input. The general chain-to-DAG
conversion applies to the resulting classifier without a model-specific wrapper.
-/
def cnn {parameters : List Shape} {input featureShape : Shape}
    (features : Chain parameters input featureShape) (outputSize : Nat) :
    Chain (parameters ++ [[outputSize, featureShape.size], [outputSize]]) input [outputSize] :=
  features >>> Chain.flatten featureShape >>> Chain.linear featureShape.size outputSize

/-- The classifier applies its head to the flattened feature output, preserving parameter order. -/
theorem cnn_interp {α : Type} [Storage α] [Context α]
    {parameters : List Shape} {input featureShape : Shape} {outputSize : Nat}
    (features : Chain parameters input featureShape)
    (state : TensorPack α parameters)
    (head : Spec.LinearSpec α featureShape.size outputSize) (x : Tensor α input) :
    Interp.spec (cnn features outputSize)
        (state.append (.cons head.weights (.cons head.bias .nil))) x =
      Spec.linearSpec head (Tensor.flattenSpec (Interp.spec features state x)) := by
  simp only [cnn, Interp.spec, TensorPack.split_append]
  rfl

end NN.GraphSpec.Models
