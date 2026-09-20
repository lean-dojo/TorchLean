/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.TensorReductionShape.LinearAlgebra
-- `timeDistributedLinearBackward` threads the parameter gradients across a sequence axis with
-- `Spec.Sequence.mapAccum`.
public import NN.Spec.Core.Sequence

/-!
# Linear layer (spec layer)

This file defines a fully‑connected layer and its gradients:

- forward: `y = W x + b`
- backward: ∂L/∂W, ∂L/∂b, ∂L/∂x

Definitions are purely functional and shape‑indexed, suitable for both proofs and reuse by
autograd wrappers in `NN/Spec/Autograd`.
-/

@[expose] public section


open TorchLean

namespace Spec

open TorchLean TorchLean.Tensor

variable {α : Type} [TorchLean.Storage α]
  [Add α] [Mul α] [Zero α]

/--
Linear layer specification (pure, shape-indexed).

This is the spec-level analogue of PyTorch `torch.nn.Linear` / `torch.nn.functional.linear`:
- `weights` has shape `[outDim, inDim]`,
- `bias` has shape `[outDim]`.
-/
structure LinearSpec (α : Type) [TorchLean.Storage α]
    (inDim outDim : Nat) where
  /-- Weight matrix with rows indexed by output features. -/
  weights : Tensor α [outDim, inDim]
  /-- Bias vector added to each output feature. -/
  bias    : Tensor α [outDim]

/--
Unbatched forward pass: `y = W x + b`.

PyTorch analogue: `torch.nn.functional.linear`.
-/
def linearSpec {inDim outDim : Nat}
  (m : LinearSpec α inDim outDim)
  (input : Tensor α [inDim]) :
  Tensor α [outDim] :=
  addSpec (matVecMulSpec m.weights input) m.bias

/--
Gradient w.r.t. weights: `∂L/∂W = (∂L/∂y) ⊗ x` (outer product).

This is the standard linear-layer backward formula for `y = W x + b`.
-/
def linearWeightsDerivSpec {inDim outDim : Nat}
  (input : Tensor α [inDim])
  (gradOutput : Tensor α [outDim]) :
  Tensor α [outDim, inDim] :=
  outerProductSpec gradOutput input

/--
Gradient w.r.t. bias: `∂L/∂b = ∂L/∂y`.

Since `y = W x + b`, the Jacobian of `y` w.r.t. `b` is the identity.
-/
def linearBiasDerivSpec {inDim outDim : Nat}
  (_dW : Tensor α [outDim, inDim])
  (gradOutput : Tensor α [outDim])
  (_input : Tensor α [inDim]) :
  Tensor α [outDim] := gradOutput

/--
Gradient w.r.t. input: `∂L/∂x = Wᵀ (∂L/∂y)`.

This is the standard "matmul by the transpose" rule for `y = W x + b`.
-/
def linearInputDerivSpec {inDim outDim : Nat}
  (weights : Tensor α [outDim, inDim])
  (gradOutput : Tensor α [outDim]) :
  Tensor α [inDim] :=
  vecMatMulSpec gradOutput weights

/--
Gradients for the parameters of a linear layer `y = W x + b`.

`LinearSpec` stores `weights` and `bias`, so this bundle names its fields the same way with a
`Gradient` suffix. The LSTM output head in `NN/Spec/Module/LstmModels.lean` and the seq2seq decoder
projection in `NN/Spec/Models/Seq2seq.lean` each used to declare a private copy of exactly this
record, which meant a gradient produced by one model could not be read by code written against the
other. One shared record fixes that.

PyTorch analogue: `(layer.weight.grad, layer.bias.grad)` for `torch.nn.Linear`.
-/
structure LinearParameterGradients (α : Type) [TorchLean.Storage α] (inDim outDim : Nat) where
  /-- Gradient with respect to the weight matrix `W`. -/
  weightGradient : Tensor α [outDim, inDim]
  /-- Gradient with respect to the bias vector `b`. -/
  biasGradient : Tensor α [outDim]
deriving Repr

/--
Everything a linear backward pass produces: the two parameter gradients, plus the gradient that
keeps travelling backwards into the layer's input.

`inputShape` is a parameter because the same rule serves the unbatched case (`[inDim]`) and the
batched one (`leading.appendDim inDim`). The three fields are spelled out rather than inherited from
`LinearParameterGradients`: a structure that extends another prints its parent as a nested
`toLinearParameterGradients := ...`, and these records get `#eval`'d in the guide, where a flat line
is what a reader wants to compare against the formula. `LinearGradients.parameters` recovers the
parameter pair for model-level gradient records.

Convolution returns its gradients the same way, as `Spec.ConvGradients`.
-/
structure LinearGradients (α : Type) [TorchLean.Storage α] (inDim outDim : Nat)
    (inputShape : Shape) where
  /-- Gradient with respect to the weight matrix `W`. -/
  weightGradient : Tensor α [outDim, inDim]
  /-- Gradient with respect to the bias vector `b`. -/
  biasGradient : Tensor α [outDim]
  /-- Gradient with respect to the layer input. -/
  inputGradient : Tensor α inputShape
deriving Repr

/-- Forget the input gradient and keep the two parameter gradients, which is what a model-level
gradient record stores for a layer whose input gradient has already been consumed. -/
def LinearGradients.parameters {inDim outDim : Nat} {inputShape : Shape}
  (gradients : LinearGradients α inDim outDim inputShape) :
  LinearParameterGradients α inDim outDim :=
  { weightGradient := gradients.weightGradient, biasGradient := gradients.biasGradient }

/--
Linear derivatives over any nonempty leading shape.

The leading axes are flattened only while accumulating the parameter gradients:
- `d_weights = (gradOutputᵀ) · input`,
- `d_bias = sum(gradOutput)` over every leading coordinate,
- `d_input = gradOutput · weights`.
-/
def linearDerivSpec [Inhabited α] {leading : Shape} {inDim outDim : Nat}
  (hLeading : 0 < Shape.size leading)
  (weights : Tensor α [outDim, inDim])
  (input : Tensor α (leading.appendDim inDim))
  (gradOutput : Tensor α (leading.appendDim outDim)) :
  LinearGradients α inDim outDim (leading.appendDim inDim) :=
  let inputFlat : Tensor α [Shape.size leading, inDim] :=
    reshapeSpec input (by simp [Shape.size_appendDim, Shape.size])
  let gradOutputFlat : Tensor α [Shape.size leading, outDim] :=
    reshapeSpec gradOutput (by simp [Shape.size_appendDim, Shape.size])
  let hSamples : Shape.NonemptyAxis 0 [Shape.size leading, outDim] := by
    obtain ⟨sampleCount, hSampleCount⟩ :=
      Nat.exists_eq_succ_of_ne_zero (Nat.ne_of_gt hLeading)
    rw [hSampleCount]
    exact .zero
  let dWeights := matMulSpec (swapAdjacentAxes gradOutputFlat 0) inputFlat
  let dBias := reduceSum 0 gradOutputFlat hSamples
  let dInputFlat := matMulSpec gradOutputFlat weights
  let dInput := reshapeSpec dInputFlat (by simp [Shape.size_appendDim, Shape.size])
  { weightGradient := dWeights, biasGradient := dBias, inputGradient := dInput }

/--
Complete unbatched backward pass for a linear layer.

Returns ∂L/∂W, ∂L/∂b and ∂L/∂x given the layer params, input `x`, and output gradient `∂L/∂y`.
-/
def linearBackwardSpec {inDim outDim : Nat}
  (layer : LinearSpec α inDim outDim)
  (input : Tensor α [inDim])
  (gradOutput : Tensor α [outDim]) :
  LinearGradients α inDim outDim [inDim] :=
  let d_weights := linearWeightsDerivSpec input gradOutput
  let d_bias := linearBiasDerivSpec d_weights gradOutput input
  let d_input := linearInputDerivSpec layer.weights gradOutput
  { weightGradient := d_weights, biasGradient := d_bias, inputGradient := d_input }

/--
Backward pass for a linear layer applied at every position of a sequence.

The layer is shared across positions, so its parameter gradients accumulate over the sequence while
each position gets its own input gradient. This is the rule behind both the LSTM output head
(`Spec.Lstm.Model.backward`) and the seq2seq decoder projection
(`Spec.Seq2SeqDecoderSpec.backwardTeacherForcing`); those two files each carried a byte-identical
copy of it under different binder names until the record above gave them a common vocabulary.

PyTorch analogue: backprop through `nn.Linear` applied inside a loop over timesteps, where
`weight.grad` sums the per-step contributions.
-/
def timeDistributedLinearBackward {seqLen inDim outDim : Nat}
  (layer : LinearSpec α inDim outDim)
  (inputs : Tensor α [seqLen, inDim])
  (gradOutputs : Tensor α [seqLen, outDim]) :
  LinearGradients α inDim outDim [seqLen, inDim] :=
  let step (i : Fin seqLen) (acc : LinearParameterGradients α inDim outDim) :=
    let stepGrads := linearBackwardSpec layer (get inputs i) (get gradOutputs i)
    ({ weightGradient := addSpec acc.weightGradient stepGrads.weightGradient
       biasGradient := addSpec acc.biasGradient stepGrads.biasGradient },
     stepGrads.inputGradient)
  let init : LinearParameterGradients α inDim outDim :=
    { weightGradient := Tensor.full ([outDim, inDim]) 0
      biasGradient := Tensor.full ([outDim]) 0 }
  let (parameterGradients, inputGradients) := Sequence.mapAccum seqLen init step
  { weightGradient := parameterGradients.weightGradient
    biasGradient := parameterGradients.biasGradient
    inputGradient := Tensor.dim inputGradients.getScalar }

/--
Accumulate two weight gradients by addition.

This is a small helper used by batching/training code.
-/
def linearGradientAccumulateSpec {inDim outDim : Nat}
  (grad1 : Tensor α [outDim, inDim])
  (grad2 : Tensor α [outDim, inDim]) :
  Tensor α [outDim, inDim] :=
  addSpec grad1 grad2

/-- Scale a weight gradient by a scalar factor (e.g. learning-rate adjustment). -/
def linearGradientScaleSpec {inDim outDim : Nat}
  (grad : Tensor α [outDim, inDim])
  (scaleFactor : α) :
  Tensor α [outDim, inDim] :=
  scaleSpec grad scaleFactor

end Spec
