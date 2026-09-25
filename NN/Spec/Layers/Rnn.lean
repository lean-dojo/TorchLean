/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Sequence
public import NN.Spec.Layers.Activation
public import NN.Spec.Core.TensorReductionShape.ConcatSlice

/-!
# RNN (spec layer)

Defines a vanilla RNN cell and sequence semantics, along with BPTT-style gradients.

This is the recurrent core that TorchLean builds on:

- a single-step cell (`rnnCellSpec`),
- an explicit unrolling over time (`rnnSequenceSpec`),
- and a reverse-time VJP (`rnnSequenceBackwardSpec`).

PyTorch analogy:

- `rnnCellSpec` corresponds to `torch.nn.RNNCell` with `nonlinearity="tanh"`.
- `rnnSequenceSpec` corresponds to `torch.nn.RNN` unrolled over `seqLen`.

## References

- Elman, "Finding Structure in Time" (1990): https://crl.ucsd.edu/~elman/Papers/fsit.pdf
- PyTorch `RNNCell`: https://docs.pytorch.org/docs/stable/generated/torch.nn.RNNCell.html
- PyTorch `RNN`: https://docs.pytorch.org/docs/stable/generated/torch.nn.RNN.html
-/

@[expose] public section


open TorchLean

namespace Spec

open TorchLean TorchLean.Tensor
open Activation

variable {α : Type} [TorchLean.Storage α] [Context α]

/-!
## Recurrent tensor shapes

Recurrent vectors and matrices use ordinary rank-one and rank-two tensors. Sequences are
time-major, with `seqLen` as the outermost axis, because that layout follows the recursive
definitions and proofs directly.
-/

/--
RNN cell parameters.

We use a single weight matrix applied to a concatenated vector `[x_t; h_{t-1}]`:

`h_t = tanh(W [x_t; h_{t-1}] + b)`.

This is equivalent to the common split-parameter form:

`h_t = tanh(W_ih x_t + W_hh h_{t-1} + b)`,

just packaged to reuse the same tensor primitives elsewhere in TorchLean.
-/
structure RNNSpec (α : Type) [TorchLean.Storage α]
    (inputSize hiddenSize : Nat) where
  /-- Combined input-to-hidden and hidden-to-hidden weight matrix. -/
  weights : Tensor α [hiddenSize, inputSize + hiddenSize]
  /-- Hidden-state bias vector. -/
  bias    : Tensor α [hiddenSize]

/--
Single RNN cell forward pass.

Math:
`h_t = tanh(W [x_t; h_{t-1}] + b)`.

PyTorch analogy: `RNNCell(input, hidden)` with `tanh` nonlinearity.
-/
def rnnCellSpec {inputSize hiddenSize : Nat}
  (rnn : RNNSpec α inputSize hiddenSize)
  (input : Tensor α [inputSize])
  (hidden : Tensor α [hiddenSize]) :
  Tensor α [hiddenSize] :=
  -- Concatenate input and hidden state
  let concat := concatAxisSpec .scalar input hidden
  -- Apply linear transformation: Wx + b
  let linearOut := addSpec (matVecMulSpec rnn.weights concat) rnn.bias
  -- Apply tanh activation
  tanhSpec linearOut

-- ============================================================================
-- Backpropagation (BPTT)
-- ============================================================================

/--
Parameter gradients for an `RNNSpec` cell.

The cell holds a single weight matrix applied to `[x_t; h_{t-1}]` plus a bias, so this pair is the
whole parameter gradient. Encoder and decoder cells use the same record.

PyTorch analogue: `(cell.weight_ih.grad, cell.weight_hh.grad)` fused into one matrix, plus the bias
gradient.
-/
structure RNNParameterGradients (α : Type) [TorchLean.Storage α] (inputSize hiddenSize : Nat) where
  /-- Gradient of the fused input/hidden weight matrix. -/
  weightGradient : Tensor α [hiddenSize, inputSize + hiddenSize]
  /-- Gradient of the bias. -/
  biasGradient : Tensor α [hiddenSize]
deriving Repr

/-- All-zero parameter gradients: the starting point for accumulation over a sequence. -/
def RNNParameterGradients.zero {inputSize hiddenSize : Nat} :
  RNNParameterGradients α inputSize hiddenSize :=
  { weightGradient := Tensor.full ([hiddenSize, inputSize + hiddenSize]) 0
    biasGradient := Tensor.full ([hiddenSize]) 0 }

/-- Add two parameter gradient bundles, which is what one BPTT step contributes. -/
def RNNParameterGradients.add {inputSize hiddenSize : Nat}
  (left right : RNNParameterGradients α inputSize hiddenSize) :
  RNNParameterGradients α inputSize hiddenSize :=
  { weightGradient := addSpec left.weightGradient right.weightGradient
    biasGradient := addSpec left.biasGradient right.biasGradient }

/-- Everything one RNN cell step sends backwards. -/
structure RNNCellGradients (α : Type) [TorchLean.Storage α] (inputSize hiddenSize : Nat) where
  /-- Gradients for the cell parameters. -/
  parameters : RNNParameterGradients α inputSize hiddenSize
  /-- Gradient with respect to the step input `x_t`. -/
  input : Tensor α [inputSize]
  /-- Gradient with respect to the incoming hidden state `h_{t-1}`. -/
  previousHidden : Tensor α [hiddenSize]

/-- Result of backpropagation through time for an RNN: parameter gradients summed over the
sequence, one input gradient per timestep, and the gradient for the hidden state fed in at
`t = 0`. -/
structure RNNSequenceGradients (α : Type) [TorchLean.Storage α]
    (seqLen inputSize hiddenSize : Nat) where
  /-- Parameter gradients accumulated over every timestep. -/
  parameters : RNNParameterGradients α inputSize hiddenSize
  /-- Gradient with respect to the input sequence. -/
  inputs : Tensor α [seqLen, inputSize]
  /-- Gradient with respect to the initial hidden state. -/
  initialHidden : Tensor α [hiddenSize]

-- Single RNN cell backward pass.
-- Forward: h_t = tanh(W @ [x_t; h_{t-1}] + b)
/--
Backward/VJP for a single RNN cell.

Inputs:
- `x_t`, `h_{t-1}`,
- the cached forward output `h_t` (so we can write `tanh'` in terms of `h_t`),
- an upstream gradient `dL/dh_t`.

Outputs:
- an `RNNCellGradients` record with `dL/dx_t`, `dL/dh_{t-1}`, and the parameter gradients.
-/
def rnnCellBackwardSpec {inputSize hiddenSize : Nat}
  (rnn : RNNSpec α inputSize hiddenSize)
  (input : Tensor α [inputSize])
  (prevHidden : Tensor α [hiddenSize])
  (hidden : Tensor α [hiddenSize])
  (gradHidden : Tensor α [hiddenSize]) :
  RNNCellGradients α inputSize hiddenSize :=
  let concat := concatAxisSpec .scalar input prevHidden

  -- tanh'(z) = 1 - tanh(z)^2, and tanh(z) = hidden
  let tanhDeriv := subSpec (Tensor.full (.dim hiddenSize .scalar) 1) (mulSpec hidden hidden)
  let gradPreact := mulSpec gradHidden tanhDeriv

  let gradWeights := outerProductSpec gradPreact concat
  let gradBias := gradPreact

  -- dConcat = gradPreactᵀ * W  (shape: inputSize + hiddenSize)
  let gradConcat := vecMatMulSpec gradPreact rnn.weights
  let gradInput := sliceRangeSpec gradConcat 0 inputSize (by simp)
  let gradPrevHidden := sliceRangeSpec gradConcat inputSize hiddenSize (by simp)

  { parameters := { weightGradient := gradWeights, biasGradient := gradBias }
    input := gradInput
    previousHidden := gradPrevHidden }

-- RNN sequence forward pass: processes a sequence of inputs
/--
Unroll an RNN over `seqLen` steps (time-major).

Returns the sequence of hidden states `[h_0, ..., h_{seqLen-1}]`.
-/
def rnnSequenceSpec {seqLen inputSize hiddenSize : Nat}
  (rnn : RNNSpec α inputSize hiddenSize)
  (inputs : Tensor α [seqLen, inputSize])
  (initialHidden : Tensor α [hiddenSize]) :
  Tensor α [seqLen, hiddenSize] :=
  let (_, outputs) := Sequence.mapAccum seqLen initialHidden fun i previous =>
    let hidden := rnnCellSpec rnn (get inputs i) previous
    (hidden, hidden)
  Tensor.dim outputs.getScalar

/-- Batched RNN forward pass (maps `rnnSequenceSpec` over the batch dimension). -/
def rnnBatchedSpec {batchSize seqLen inputSize hiddenSize : Nat}
  (rnn : RNNSpec α inputSize hiddenSize)
  (inputs : Tensor α [batchSize, seqLen, inputSize])
  (initialHidden : Tensor α [batchSize, hiddenSize]) :
  Tensor α [batchSize, seqLen, hiddenSize] :=
  Tensor.dim (fun b =>
    rnnSequenceSpec rnn (Tensor.unstack inputs b) (Tensor.unstack initialHidden b))

/--
Gradient w.r.t. weights from a full unroll, given per-step preactivation gradients.

This helper is for analyses that already have preactivation gradients. It assumes:
- the initial hidden state is `0`, and
- `gradOutputs[t]` is already `dL/dz_t` (preactivation gradient).

For end-to-end BPTT from `dL/dh_t`, prefer `rnnSequenceBackwardSpec`.
-/
def rnnWeightsDerivSpec {seqLen inputSize hiddenSize : Nat}
  (inputs : Tensor α [seqLen, inputSize])
  (hiddens : Tensor α [seqLen, hiddenSize])
  (gradOutputs : Tensor α [seqLen, hiddenSize]) :
  Tensor α [hiddenSize, inputSize + hiddenSize] :=
  -- Assumes initial hidden state is 0 (matches the default module wrappers).
  -- Assumes `gradOutputs` is the preactivation gradient at each timestep.
  -- For full BPTT from post-activation gradients, use `rnnSequenceBackwardSpec`.
  let rec accumulate_grads (t : Nat) (acc : Tensor α [hiddenSize, inputSize + hiddenSize]) :
      Tensor α [hiddenSize, inputSize + hiddenSize] :=
    if h : t < seqLen then
      let inputT := get inputs ⟨t, h⟩
      let hiddenPrev :=
        if ht : t > 0 then
          have h_pred : t - 1 < t := by
            simpa [Nat.pred_eq_sub_one] using Nat.pred_lt (Nat.ne_of_gt ht)
          have h_t' : t - 1 < seqLen := lt_trans h_pred h
          get hiddens ⟨t - 1, h_t'⟩
        else
          Tensor.full (.dim hiddenSize .scalar) 0
      let gradPreactT := get gradOutputs ⟨t, h⟩
      let concatT := concatAxisSpec .scalar inputT hiddenPrev
      let gradWT := outerProductSpec gradPreactT concatT
      accumulate_grads (t + 1) (addSpec acc gradWT)
    else
      acc
  accumulate_grads 0 (Tensor.full (.dim hiddenSize (.dim (inputSize + hiddenSize) .scalar)) 0)

/--
Gradient w.r.t. bias from per-step preactivation gradients.

This is `sum_t dL/dz_t` over the sequence dimension.
-/
def rnnBiasDerivSpec {seqLen hiddenSize : Nat}
  (gradOutputs : Tensor α [seqLen, hiddenSize])
  (h : seqLen ≠ 0) :
  Tensor α [hiddenSize] :=
  -- Assumes `gradOutputs` is already the preactivation gradient.
  -- For full RNN backprop, prefer `rnnSequenceBackwardSpec`.
  reduceSum 0 gradOutputs (Shape.hasNonemptyAxisZeroOfNe h).proof

/--
Full BPTT backward pass through an RNN sequence.

This is the spec-level version of what PyTorch autograd computes for `nn.RNN` when unrolled:

- we walk time in reverse,
- accumulate parameter gradients,
- and compute gradients for each input step plus the initial hidden state.

### Diagram: forward unroll + BPTT (vanilla RNN)

One step (forward):

```
x_t        h_{t-1}
 |            |
 +---- concat ----+
                 |
             z_t = W · [x_t; h_{t-1}] + b
                 |
             h_t = tanh(z_t)
```

Unrolled over time (forward):

```
h_-1 = h0

x0 -> [cell] -> h0 -> [cell] -> h1 -> ... -> [cell] -> h_{T-1}
        ^          ^                       ^
      uses h_-1  uses h0                 uses h_{T-2}
```

Backprop through time (reverse):

At each time step we combine two sources of gradient for `h_t`:

- the gradient coming from the loss that touches `h_t` directly (`gradHiddens[t]`),
- plus the gradient flowing "from the future" through the recurrence (`dHidden_next`).

Then we push `total_grad` through the single-step VJP (`rnnCellBackwardSpec`), producing:

- `dInput_t` and `dHidden_prev`,
- and parameter gradients which are accumulated across time.
-/

def rnnSequenceBackwardSpec {seqLen inputSize hiddenSize : Nat}
  (rnn : RNNSpec α inputSize hiddenSize)
  (inputs : Tensor α [seqLen, inputSize])
  (initialHidden : Tensor α [hiddenSize])
  (hiddens : Tensor α [seqLen, hiddenSize])
  (gradHiddens : Tensor α [seqLen, hiddenSize]) :
  RNNSequenceGradients α seqLen inputSize hiddenSize :=

  let initial : Tensor α [hiddenSize] × RNNParameterGradients α inputSize hiddenSize :=
    (Tensor.full ([hiddenSize]) 0, RNNParameterGradients.zero)
  let (result, dInputs) := Sequence.mapAccumRight seqLen initial fun index state =>
    let (dHiddenNext, accumulated) := state
    let input := get inputs index
    let hidden := get hiddens index
    let previous :=
      if h : index.val > 0 then
        have hp : index.val - 1 < seqLen := by grind
        get hiddens ⟨index.val - 1, hp⟩
      else
        initialHidden
    let totalGradient := addSpec (get gradHiddens index) dHiddenNext
    let step := rnnCellBackwardSpec rnn input previous hidden totalGradient
    ((step.previousHidden, accumulated.add step.parameters), step.input)
  let (dInitialHidden, parameterGradients) := result
  { parameters := parameterGradients
    inputs := Tensor.dim dInputs.getScalar
    initialHidden := dInitialHidden }

end Spec
