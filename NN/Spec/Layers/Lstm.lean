/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Sequence
public import NN.Spec.Core.TensorReductionShape.ConcatSlice
public import NN.Spec.Layers.Activation

/-!
# LSTM (spec layer)

TorchLean provides a small LSTM specification that is:

- explicit about shapes (so common dimension mistakes are caught early),
- explicit about the gate math (so gradients are inspectable and proofs can refer to the equations),
- close in spirit to the way PyTorch documents `nn.LSTMCell` / `nn.LSTM`.

## References (math + PyTorch behavior)

- Hochreiter, Schmidhuber, "Long Short-Term Memory" (Neural Computation, 1997).
  Free PDF: http://www.bioinf.jku.at/publications/older/2604.pdf
- PyTorch `LSTMCell` equations:
  https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTMCell.html
- PyTorch `LSTM` equations: https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html

## Notes on parameterization

Many libraries expose two matrices per gate (`W_ih` and `W_hh`) and add them.
In this spec we use a single matrix applied to a concatenated vector `[x_t; h_{t-1}]`.
It's the same computation, just packaged to reuse TorchLean's tensor building blocks.
-/

@[expose] public section


open TorchLean

namespace Spec

open TorchLean TorchLean.Tensor
open Activation

variable {α : Type} [TorchLean.Storage α] [Context α]

/-- Parameters for an LSTM cell, with one `(hiddenSize × (inputSize + hiddenSize))` matrix per gate.

This corresponds to the usual `(W_ih, W_hh)` parameterization in libraries like PyTorch, but we
package it as a single matrix applied to `[x_t; h_{t-1}]` to reuse TorchLean's tensor building
blocks.
-/
structure LSTMSpec (α : Type) [TorchLean.Storage α]
    (inputSize hiddenSize : Nat) where
  /-- Forget-gate weights for `f_t = sigmoid(W_f [x_t; h_{t-1}] + b_f)`. -/
  forgetWeight : Tensor α [hiddenSize, inputSize + hiddenSize]
  /-- Forget-gate bias. -/
  forgetBias : Tensor α [hiddenSize]
  /-- Input-gate weights for `i_t = sigmoid(W_i [x_t; h_{t-1}] + b_i)`. -/
  inputWeight : Tensor α [hiddenSize, inputSize + hiddenSize]
  /-- Input-gate bias. -/
  inputBias : Tensor α [hiddenSize]
  /-- Candidate/cell-proposal weights for `g_t = tanh(W_g [x_t; h_{t-1}] + b_g)`. -/
  candidateWeight : Tensor α [hiddenSize, inputSize + hiddenSize]
  /-- Candidate/cell-proposal bias. -/
  candidateBias : Tensor α [hiddenSize]
  /-- Output-gate weights for `o_t = sigmoid(W_o [x_t; h_{t-1}] + b_o)`. -/
  outputWeight : Tensor α [hiddenSize, inputSize + hiddenSize]
  /-- Output-gate bias. -/
  outputBias : Tensor α [hiddenSize]

/-- LSTM recurrent state: hidden vector `h_t` and cell vector `c_t`. -/
structure LSTMState (α : Type) [TorchLean.Storage α] (hiddenSize : Nat) where
  /-- Exposed hidden state `h_t`. -/
  hidden : Tensor α [hiddenSize]  -- h_t
  /-- Internal memory/cell state `c_t`. -/
  cell   : Tensor α [hiddenSize]  -- c_t

/--
Forward pass for one LSTM cell that also returns the gate activations.

This is the spec analogue of the "saved tensors" that a runtime will keep for backward.
-/
def lstmCellSpecWithIntermediates {inputSize hiddenSize : Nat}
  (lstm : LSTMSpec α inputSize hiddenSize)
  (input : Tensor α [inputSize])
  (prevState : LSTMState α hiddenSize) :
  (LSTMState α hiddenSize ×                -- new state (h_t, c_t)
   Tensor α [hiddenSize] ×             -- forget gate f_t
   Tensor α [hiddenSize] ×             -- input gate i_t
   Tensor α [hiddenSize] ×             -- candidate g_t
   Tensor α [hiddenSize]) :=           -- output gate o_t
  -- We follow the standard LSTM equations (same layout as in the PyTorch docs):
  --
  --   f_t = sigmoid(W_f [x_t; h_{t-1}] + b_f)      (forget gate)
  --   i_t = sigmoid(W_i [x_t; h_{t-1}] + b_i)      (input gate)
  --   g_t = tanh   (W_g [x_t; h_{t-1}] + b_g)      (candidate / cell proposal)
  --   o_t = sigmoid(W_o [x_t; h_{t-1}] + b_o)      (output gate)
  --   c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t              (cell state update)
  --   h_t = o_t ⊙ tanh(c_t)                        (exposed hidden state)
  --
  -- The `cell` component is what lets information persist over long ranges.
  let concat := concatAxisSpec .scalar input prevState.hidden
  let f := sigmoidSpec (addSpec (matVecMulSpec lstm.forgetWeight concat) lstm.forgetBias)
  let i := sigmoidSpec (addSpec (matVecMulSpec lstm.inputWeight concat) lstm.inputBias)
  let g := tanhSpec (addSpec (matVecMulSpec lstm.candidateWeight concat) lstm.candidateBias)
  let o := sigmoidSpec (addSpec (matVecMulSpec lstm.outputWeight concat) lstm.outputBias)
  let c := addSpec (mulSpec f prevState.cell) (mulSpec i g)
  let h := mulSpec o (tanhSpec c)
  (⟨h, c⟩, f, i, g, o)

/-- One LSTM cell step: update `(h_{t-1}, c_{t-1})` given `x_t` and parameters.

This is the state component of `lstmCellSpecWithIntermediates`, so the two cannot drift apart. -/
def lstmCellSpec {inputSize hiddenSize : Nat}
  (lstm : LSTMSpec α inputSize hiddenSize)
  (input : Tensor α [inputSize])
  (prevState : LSTMState α hiddenSize) :
  LSTMState α hiddenSize :=
  (lstmCellSpecWithIntermediates lstm input prevState).1

/-- Run an LSTM cell over a length-`seqLen` input sequence, returning outputs and final state. -/
def lstmSequenceSpec {seqLen inputSize hiddenSize : Nat}
  (lstm : LSTMSpec α inputSize hiddenSize)
  (inputs : Tensor α [seqLen, inputSize])
  (initialState : LSTMState α hiddenSize) :
  (Tensor α [seqLen, hiddenSize] × LSTMState α hiddenSize) :=
  let (finalState, outputs) := Sequence.mapAccum seqLen initialState fun i previous =>
    let state := lstmCellSpec lstm (get inputs i) previous
    (state, state.hidden)
  (Tensor.dim outputs.getScalar, finalState)

-- ============================================================================
-- Backpropagation (BPTT)
-- ============================================================================

/--
Gate-wise parameter gradients for an LSTM cell.

`LSTMSpec` keeps one weight matrix per gate, each applied to the concatenation `[x_t; h_{t-1}]`, so
every weight gradient has shape `[hiddenSize, inputSize + hiddenSize]` and every bias gradient has
shape `[hiddenSize]`. Named fields distinguish the four gate gradients despite their identical
shapes.

PyTorch analogue: the `.grad` fields of `nn.LSTMCell.weight_ih`, `weight_hh` and their biases, with
the input and hidden blocks kept in one matrix here rather than two.
-/
structure LSTMGateGradients (α : Type) [TorchLean.Storage α] (inputSize hiddenSize : Nat) where
  /-- Gradient of the forget-gate weight matrix. -/
  forgetWeight : Tensor α [hiddenSize, inputSize + hiddenSize]
  /-- Gradient of the forget-gate bias. -/
  forgetBias : Tensor α [hiddenSize]
  /-- Gradient of the input-gate weight matrix. -/
  inputWeight : Tensor α [hiddenSize, inputSize + hiddenSize]
  /-- Gradient of the input-gate bias. -/
  inputBias : Tensor α [hiddenSize]
  /-- Gradient of the candidate-state weight matrix. -/
  candidateWeight : Tensor α [hiddenSize, inputSize + hiddenSize]
  /-- Gradient of the candidate-state bias. -/
  candidateBias : Tensor α [hiddenSize]
  /-- Gradient of the output-gate weight matrix. -/
  outputWeight : Tensor α [hiddenSize, inputSize + hiddenSize]
  /-- Gradient of the output-gate bias. -/
  outputBias : Tensor α [hiddenSize]
deriving Repr

/-- All-zero gate gradients: the starting point for accumulation over a sequence. -/
def LSTMGateGradients.zero {inputSize hiddenSize : Nat} :
  LSTMGateGradients α inputSize hiddenSize :=
  let weight : Tensor α [hiddenSize, inputSize + hiddenSize] :=
    Tensor.full ([hiddenSize, inputSize + hiddenSize]) 0
  let bias : Tensor α [hiddenSize] := Tensor.full ([hiddenSize]) 0
  { forgetWeight := weight, forgetBias := bias
    inputWeight := weight, inputBias := bias
    candidateWeight := weight, candidateBias := bias
    outputWeight := weight, outputBias := bias }

/-- Add two gate gradient bundles gate by gate, which is what one BPTT step contributes. -/
def LSTMGateGradients.add {inputSize hiddenSize : Nat}
  (left right : LSTMGateGradients α inputSize hiddenSize) :
  LSTMGateGradients α inputSize hiddenSize :=
  { forgetWeight := addSpec left.forgetWeight right.forgetWeight
    forgetBias := addSpec left.forgetBias right.forgetBias
    inputWeight := addSpec left.inputWeight right.inputWeight
    inputBias := addSpec left.inputBias right.inputBias
    candidateWeight := addSpec left.candidateWeight right.candidateWeight
    candidateBias := addSpec left.candidateBias right.candidateBias
    outputWeight := addSpec left.outputWeight right.outputWeight
    outputBias := addSpec left.outputBias right.outputBias }

/-- Everything one LSTM cell step sends backwards: gate parameter gradients, the input gradient,
and the gradient for the state that arrived from the previous step. -/
structure LSTMCellGradients (α : Type) [TorchLean.Storage α] (inputSize hiddenSize : Nat) where
  /-- Gradients for the four gate parameter blocks. -/
  gates : LSTMGateGradients α inputSize hiddenSize
  /-- Gradient with respect to the step input `x_t`. -/
  input : Tensor α [inputSize]
  /-- Gradient with respect to the incoming state `(h_{t-1}, c_{t-1})`. -/
  previousState : LSTMState α hiddenSize

/-- Result of backpropagation through time: gate gradients summed over the sequence, one input
gradient per timestep, and the gradient for the state fed in at `t = 0`. -/
structure LSTMSequenceGradients (α : Type) [TorchLean.Storage α]
    (seqLen inputSize hiddenSize : Nat) where
  /-- Gate parameter gradients accumulated over every timestep. -/
  gates : LSTMGateGradients α inputSize hiddenSize
  /-- Gradient with respect to the input sequence. -/
  inputs : Tensor α [seqLen, inputSize]
  /-- Gradient with respect to the initial state. -/
  initialState : LSTMState α hiddenSize

-- Single LSTM cell backward pass.
/--
Backward pass (VJP) for a single LSTM cell.

Inputs:
- parameters `lstm`,
- inputs `x_t`, previous state `(h_{t-1}, c_{t-1})`, and current state `(h_t, c_t)`,
- the gate activations from the forward pass,
- upstream gradients for both `h_t` and `c_t`.

Outputs, as an `LSTMCellGradients` record: gradients w.r.t. `x_t` and the previous state, plus one
gradient per gate parameter tensor.

This is the quantity computed by PyTorch autograd for an `nn.LSTMCell` unrolled in time.
-/
def lstmCellBackwardSpec {inputSize hiddenSize : Nat}
  (lstm : LSTMSpec α inputSize hiddenSize)
  (input : Tensor α [inputSize])
  (prevState : LSTMState α hiddenSize)
  (state : LSTMState α hiddenSize)
  (forgetGate : Tensor α [hiddenSize])
  (inputGate : Tensor α [hiddenSize])
  (candidate : Tensor α [hiddenSize])
  (outputGate : Tensor α [hiddenSize])
  (gradHidden : Tensor α [hiddenSize])
  (gradCell : Tensor α [hiddenSize]) :
  LSTMCellGradients α inputSize hiddenSize :=

  let concat := concatAxisSpec .scalar input prevState.hidden

  let tanhC := tanhSpec state.cell
  let tanhCDeriv := subSpec (Tensor.full (.dim hiddenSize .scalar) 1) (mulSpec tanhC tanhC)

  -- h = o ⊙ tanh(c)
  let dO := mulSpec gradHidden tanhC
  let dCFromH := mulSpec (mulSpec gradHidden outputGate) tanhCDeriv
  let dC := addSpec gradCell dCFromH

  -- c = f ⊙ c_prev + i ⊙ g
  let dF := mulSpec dC prevState.cell
  let dI := mulSpec dC candidate
  let dG := mulSpec dC inputGate
  let dCPrev := mulSpec dC forgetGate

  -- preactivation gradients
  let dFPre := mulSpec dF (Activation.sigmoidOutputDerivSpec forgetGate)
  let dIPre := mulSpec dI (Activation.sigmoidOutputDerivSpec inputGate)
  let dOPre := mulSpec dO (Activation.sigmoidOutputDerivSpec outputGate)
  let dGPre :=
    let tanhDeriv :=
      subSpec (Tensor.full (.dim hiddenSize .scalar) 1) (mulSpec candidate candidate)
    mulSpec dG tanhDeriv

  let dWf := outerProductSpec dFPre concat
  let dbf := dFPre
  let dWi := outerProductSpec dIPre concat
  let dbi := dIPre
  let dWc := outerProductSpec dGPre concat
  let dbc := dGPre
  let dWo := outerProductSpec dOPre concat
  let dbo := dOPre

  let dConcatF := vecMatMulSpec dFPre lstm.forgetWeight
  let dConcatI := vecMatMulSpec dIPre lstm.inputWeight
  let dConcatC := vecMatMulSpec dGPre lstm.candidateWeight
  let dConcatO := vecMatMulSpec dOPre lstm.outputWeight
  let dConcat := addSpec (addSpec dConcatF dConcatI) (addSpec dConcatC dConcatO)

  let dInput := sliceRangeSpec dConcat 0 inputSize (by simp)
  let dPrevHidden := sliceRangeSpec dConcat inputSize hiddenSize (by simp)

  { gates :=
      { forgetWeight := dWf, forgetBias := dbf
        inputWeight := dWi, inputBias := dbi
        candidateWeight := dWc, candidateBias := dbc
        outputWeight := dWo, outputBias := dbo }
    input := dInput
    previousState := ⟨dPrevHidden, dCPrev⟩ }

-- Full BPTT backward pass through an LSTM sequence.
-- Recomputes intermediate gates/states internally to avoid requiring a "tape" argument.
/--
Backprop through time (BPTT) for the whole sequence.

This function recomputes and stores the forward intermediates (gates and states) internally, then
walks time backward accumulating parameter gradients and input gradients. This matches the usual
PyTorch training structure, with the save-vs-recompute choice made explicit.
-/
def lstmSequenceBackwardSpec {seqLen inputSize hiddenSize : Nat}
  (lstm : LSTMSpec α inputSize hiddenSize)
  (inputs : Tensor α [seqLen, inputSize])
  (initialState : LSTMState α hiddenSize)
  (gradHiddens : Tensor α [seqLen, hiddenSize]) :
  LSTMSequenceGradients α seqLen inputSize hiddenSize :=

  let (_, saved) := Sequence.mapAccum seqLen initialState fun index state =>
    let (next, forget, input, candidate, output) :=
      lstmCellSpecWithIntermediates lstm (get inputs index) state
    (next, (next, forget, input, candidate, output))

  let zeroHidden := Tensor.full ([hiddenSize]) 0
  let initial : LSTMState α hiddenSize × LSTMGateGradients α inputSize hiddenSize :=
    (⟨zeroHidden, zeroHidden⟩, LSTMGateGradients.zero)
  let (result, dInputs) := Sequence.mapAccumRight seqLen initial fun index state =>
    let (dNext, accumulated) := state
    let (current, forget, inputGate, candidate, output) := saved.getScalar index
    let previous :=
      if h : index.val > 0 then
        have hp : index.val - 1 < seqLen := by grind
        let (previous, _, _, _, _) := saved.getScalar ⟨index.val - 1, hp⟩
        previous
      else
        initialState
    let totalHidden := addSpec (get gradHiddens index) dNext.hidden
    let step :=
      lstmCellBackwardSpec lstm (get inputs index) previous current forget inputGate candidate
        output totalHidden dNext.cell
    ((step.previousState, accumulated.add step.gates), step.input)
  let (dInitialState, gateGradients) := result
  { gates := gateGradients
    inputs := Tensor.dim dInputs.getScalar
    initialState := dInitialState }

end Spec
