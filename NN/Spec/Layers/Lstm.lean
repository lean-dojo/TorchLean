/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Rnn

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


namespace Spec

open Tensor
open Activation

variable {α : Type} [Context α]

/-- Parameters for an LSTM cell, with one `(hiddenSize × (inputSize + hiddenSize))` matrix per gate.

This corresponds to the usual `(W_ih, W_hh)` parameterization in libraries like PyTorch, but we
package it as a single matrix applied to `[x_t; h_{t-1}]` to reuse TorchLean's tensor building
blocks.
-/
structure LSTMSpec (α : Type) (inputSize hiddenSize : Nat) where
  /-- Forget-gate weights for `f_t = sigmoid(W_f [x_t; h_{t-1}] + b_f)`. -/
  forget_weights : WeightMatrix α hiddenSize (inputSize + hiddenSize)
  /-- Forget-gate bias. -/
  forget_bias    : HiddenVector α hiddenSize
  /-- Input-gate weights for `i_t = sigmoid(W_i [x_t; h_{t-1}] + b_i)`. -/
  input_weights  : WeightMatrix α hiddenSize (inputSize + hiddenSize)
  /-- Input-gate bias. -/
  input_bias     : HiddenVector α hiddenSize
  /-- Candidate/cell-proposal weights for `g_t = tanh(W_g [x_t; h_{t-1}] + b_g)`. -/
  candidate_weights : WeightMatrix α hiddenSize (inputSize + hiddenSize)
  /-- Candidate/cell-proposal bias. -/
  candidate_bias    : HiddenVector α hiddenSize
  /-- Output-gate weights for `o_t = sigmoid(W_o [x_t; h_{t-1}] + b_o)`. -/
  output_weights : WeightMatrix α hiddenSize (inputSize + hiddenSize)
  /-- Output-gate bias. -/
  output_bias    : HiddenVector α hiddenSize

/-- LSTM recurrent state: hidden vector `h_t` and cell vector `c_t`. -/
structure LSTMState (α : Type) (hiddenSize : Nat) where
  /-- Exposed hidden state `h_t`. -/
  hidden : HiddenVector α hiddenSize  -- h_t
  /-- Internal memory/cell state `c_t`. -/
  cell   : HiddenVector α hiddenSize  -- c_t

/-- One LSTM cell step: update `(h_{t-1}, c_{t-1})` given `x_t` and parameters. -/
def lstmCellSpec {inputSize hiddenSize : Nat}
  (lstm : LSTMSpec α inputSize hiddenSize)
  (input : InputVector α inputSize)
  (prev_state : LSTMState α hiddenSize) :
  LSTMState α hiddenSize :=
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
  let concat := concatVectorsSpec input prev_state.hidden

  -- Forget gate: f_t = σ(W_f @ [x_t; h_{t-1}] + b_f)
  let forget_gate := sigmoidSpec (addSpec (matVecMulSpec lstm.forget_weights concat)
    lstm.forget_bias)

  -- Input gate: i_t = σ(W_i @ [x_t; h_{t-1}] + b_i)
  let input_gate := sigmoidSpec (addSpec (matVecMulSpec lstm.input_weights concat)
    lstm.input_bias)

  -- Candidate values: ĉ_t = tanh(W_c @ [x_t; h_{t-1}] + b_c)
  let candidate := tanhSpec (addSpec (matVecMulSpec lstm.candidate_weights concat)
    lstm.candidate_bias)

  -- Output gate: o_t = σ(W_o @ [x_t; h_{t-1}] + b_o)
  let output_gate := sigmoidSpec (addSpec (matVecMulSpec lstm.output_weights concat)
    lstm.output_bias)

  -- Cell state: c_t = f_t ⊙ c_{t-1} + i_t ⊙ ĉ_t
  let new_cell := addSpec (mulSpec forget_gate prev_state.cell) (mulSpec input_gate candidate)

  -- Hidden state: h_t = o_t ⊙ tanh(c_t)
  let new_hidden := mulSpec output_gate (tanhSpec new_cell)

  ⟨new_hidden, new_cell⟩

/-- Run an LSTM cell over a length-`seqLen` input sequence, returning outputs and final state. -/
def lstmSequenceSpec {seqLen inputSize hiddenSize : Nat}
  (lstm : LSTMSpec α inputSize hiddenSize)
  (inputs : SequenceTensor α seqLen (.dim inputSize .scalar))
  (initial_state : LSTMState α hiddenSize) :
  (SequenceTensor α seqLen (.dim hiddenSize .scalar) × LSTMState α hiddenSize) :=
  -- Spec semantics: run the cell over time, carrying `(h_t, c_t)` forward.
  -- A runtime can implement the same semantics with a tight loop and its own caching strategy.
  let rec process_sequence (t : Nat) (prev_state : LSTMState α hiddenSize)
    : (LSTMState α hiddenSize × List (HiddenVector α hiddenSize)) :=
    if h : t < seqLen then
      let input_t := getAtSpec inputs ⟨t, h⟩
      let state_t := lstmCellSpec lstm input_t prev_state
      let (final_state, rest_outputs) := process_sequence (t + 1) state_t
      (final_state, state_t.hidden :: rest_outputs)
    else
      (prev_state, [])

  let (final_state, outputs_rev) := process_sequence 0 initial_state
  let outputs := outputs_rev.reverse
  -- Convert list to tensor
  let output_tensor := match outputs with
  | [] =>
      -- Convention for `seqLen = 0`: there are no outputs, and the eliminator gives us a
      -- function `Fin 0 -> _` anyway.
      Tensor.dim (fun _ => initial_state.hidden)
  | h :: _ => Tensor.dim (fun i => outputs.getD i.val h)

  (output_tensor, final_state)

/-- Batched wrapper around `lstmSequenceSpec` (runs one sequence per batch element). -/
def lstmBatchedSpec {batchSize seqLen inputSize hiddenSize : Nat}
  (lstm : LSTMSpec α inputSize hiddenSize)
  (inputs : BatchedTensor α batchSize (.dim seqLen (.dim inputSize .scalar)))
  (initial_hiddens : BatchedTensor α batchSize (.dim hiddenSize .scalar)) :
  (BatchedTensor α batchSize (.dim seqLen (.dim hiddenSize .scalar)) ×
   BatchedTensor α batchSize (.dim hiddenSize .scalar)) :=
  match inputs, initial_hiddens with
  | .dim batch_inputs, .dim batch_hidden =>
    -- In PyTorch you typically pass both `h_0` and `c_0`. Here we take only `h_0` and set `c_0 =
    -- 0`.
    let batch_cell := Tensor.dim (fun _ => fill 0 (.dim hiddenSize .scalar))

    -- compute per-batch results
    let outputs := Tensor.dim (fun b =>
      let initial_state : LSTMState α hiddenSize :=
        { hidden := batch_hidden b, cell := getAtSpec batch_cell b }
      (lstmSequenceSpec lstm (batch_inputs b) initial_state).1)

    let final_hiddens := Tensor.dim (fun b =>
      let initial_state : LSTMState α hiddenSize :=
        { hidden := batch_hidden b, cell := getAtSpec batch_cell b }
      (lstmSequenceSpec lstm (batch_inputs b) initial_state).2.hidden)

    (outputs, final_hiddens)

-- ============================================================================
-- Backpropagation (BPTT)
-- ============================================================================

/--
Forward pass for one LSTM cell that also returns the gate activations.

This is the spec analogue of the "saved tensors" that a runtime will keep for backward.
-/
def lstmCellSpecWithIntermediates {inputSize hiddenSize : Nat}
  (lstm : LSTMSpec α inputSize hiddenSize)
  (input : InputVector α inputSize)
  (prev_state : LSTMState α hiddenSize) :
  (LSTMState α hiddenSize ×                -- new state (h_t, c_t)
   HiddenVector α hiddenSize ×             -- forget gate f_t
   HiddenVector α hiddenSize ×             -- input gate i_t
   HiddenVector α hiddenSize ×             -- candidate g_t
   HiddenVector α hiddenSize) :=           -- output gate o_t
  let concat := concatVectorsSpec input prev_state.hidden
  let f := sigmoidSpec (addSpec (matVecMulSpec lstm.forget_weights concat) lstm.forget_bias)
  let i := sigmoidSpec (addSpec (matVecMulSpec lstm.input_weights concat) lstm.input_bias)
  let g := tanhSpec (addSpec (matVecMulSpec lstm.candidate_weights concat) lstm.candidate_bias)
  let o := sigmoidSpec (addSpec (matVecMulSpec lstm.output_weights concat) lstm.output_bias)
  let c := addSpec (mulSpec f prev_state.cell) (mulSpec i g)
  let h := mulSpec o (tanhSpec c)
  (⟨h, c⟩, f, i, g, o)

-- Single LSTM cell backward pass.
-- Returns:
--   dX_t, dPrevState, (dWf, dbf, dWi, dbi, dWc, dbc, dWo, dbo)
/--
Backward pass (VJP) for a single LSTM cell.

Inputs:
- parameters `lstm`,
- inputs `x_t`, previous state `(h_{t-1}, c_{t-1})`, and current state `(h_t, c_t)`,
- the gate activations from the forward pass,
- upstream gradients for both `h_t` and `c_t`.

Outputs:
- gradients w.r.t. `x_t` and the previous state,
- plus gradients for each parameter tensor.

PyTorch mental model: this is what `autograd` computes for `nn.LSTMCell` when unrolled in time.
-/
def lstmCellBackwardSpec {inputSize hiddenSize : Nat}
  (lstm : LSTMSpec α inputSize hiddenSize)
  (input : InputVector α inputSize)
  (prev_state : LSTMState α hiddenSize)
  (state : LSTMState α hiddenSize)
  (forget_gate : HiddenVector α hiddenSize)
  (input_gate : HiddenVector α hiddenSize)
  (candidate : HiddenVector α hiddenSize)
  (output_gate : HiddenVector α hiddenSize)
  (grad_hidden : HiddenVector α hiddenSize)
  (grad_cell : HiddenVector α hiddenSize) :
  ( InputVector α inputSize × LSTMState α hiddenSize ×
    WeightMatrix α hiddenSize (inputSize + hiddenSize) × HiddenVector α hiddenSize ×
    WeightMatrix α hiddenSize (inputSize + hiddenSize) × HiddenVector α hiddenSize ×
    WeightMatrix α hiddenSize (inputSize + hiddenSize) × HiddenVector α hiddenSize ×
    WeightMatrix α hiddenSize (inputSize + hiddenSize) × HiddenVector α hiddenSize ) :=

  let concat := concatVectorsSpec input prev_state.hidden

  let tanh_c := tanhSpec state.cell
  let tanh_c_deriv := subSpec (fill 1 (.dim hiddenSize .scalar)) (mulSpec tanh_c tanh_c)

  -- h = o ⊙ tanh(c)
  let dO := mulSpec grad_hidden tanh_c
  let dC_from_h := mulSpec (mulSpec grad_hidden output_gate) tanh_c_deriv
  let dC := addSpec grad_cell dC_from_h

  -- c = f ⊙ c_prev + i ⊙ g
  let dF := mulSpec dC prev_state.cell
  let dI := mulSpec dC candidate
  let dG := mulSpec dC input_gate
  let dC_prev := mulSpec dC forget_gate

  -- preactivation gradients
  let dF_pre := mulSpec dF (Activation.sigmoidOutputDerivSpec forget_gate)
  let dI_pre := mulSpec dI (Activation.sigmoidOutputDerivSpec input_gate)
  let dO_pre := mulSpec dO (Activation.sigmoidOutputDerivSpec output_gate)
  let dG_pre :=
    let tanh_deriv := subSpec (fill 1 (.dim hiddenSize .scalar)) (mulSpec candidate candidate)
    mulSpec dG tanh_deriv

  let dWf := outerProductSpec dF_pre concat
  let dbf := dF_pre
  let dWi := outerProductSpec dI_pre concat
  let dbi := dI_pre
  let dWc := outerProductSpec dG_pre concat
  let dbc := dG_pre
  let dWo := outerProductSpec dO_pre concat
  let dbo := dO_pre

  let dConcat_f := vecMatMulSpec dF_pre lstm.forget_weights
  let dConcat_i := vecMatMulSpec dI_pre lstm.input_weights
  let dConcat_c := vecMatMulSpec dG_pre lstm.candidate_weights
  let dConcat_o := vecMatMulSpec dO_pre lstm.output_weights
  let dConcat := addSpec (addSpec dConcat_f dConcat_i) (addSpec dConcat_c dConcat_o)

  let dInput := sliceVectorSpec dConcat 0 inputSize (by simp)
  let dPrevHidden := sliceVectorSpec dConcat inputSize hiddenSize (by simp)

  ( dInput, ⟨dPrevHidden, dC_prev⟩
  , dWf, dbf, dWi, dbi, dWc, dbc, dWo, dbo )

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
  (inputs : SequenceTensor α seqLen (.dim inputSize .scalar))
  (initial_state : LSTMState α hiddenSize)
  (grad_hiddens : SequenceTensor α seqLen (.dim hiddenSize .scalar)) :
  ( WeightMatrix α hiddenSize (inputSize + hiddenSize) × HiddenVector α hiddenSize ×  -- dWf, dbf
    WeightMatrix α hiddenSize (inputSize + hiddenSize) × HiddenVector α hiddenSize ×  -- dWi, dbi
    WeightMatrix α hiddenSize (inputSize + hiddenSize) × HiddenVector α hiddenSize ×  -- dWc, dbc
    WeightMatrix α hiddenSize (inputSize + hiddenSize) × HiddenVector α hiddenSize ×  -- dWo, dbo
    SequenceTensor α seqLen (.dim inputSize .scalar) ×                                -- dInputs
    LSTMState α hiddenSize ) :=
    -- dInitialState

  -- Forward pass with intermediates.
  let rec forward_collect (t : Nat) (st : LSTMState α hiddenSize) :
      (LSTMState α hiddenSize ×
        List (LSTMState α hiddenSize) ×
        List (HiddenVector α hiddenSize) ×
        List (HiddenVector α hiddenSize) ×
        List (HiddenVector α hiddenSize) ×
        List (HiddenVector α hiddenSize)) :=
    if h : t < seqLen then
      let input_t := getAtSpec inputs ⟨t, h⟩
      let (st', f, i, g, o) := lstmCellSpecWithIntermediates lstm input_t st
      let (st_final, states, fs, is, gs, os) := forward_collect (t + 1) st'
      (st_final, st' :: states, f :: fs, i :: is, g :: gs, o :: os)
    else
      (st, [], [], [], [], [])

  let (_final, states_rev, fs_rev, is_rev, gs_rev, os_rev) := forward_collect 0 initial_state
  let states := states_rev.reverse
  let fs := fs_rev.reverse
  let is := is_rev.reverse
  let gs := gs_rev.reverse
  let os := os_rev.reverse

  let state_seq : List (LSTMState α hiddenSize) := states
  let f_seq := fs
  let i_seq := is
  let g_seq := gs
  let o_seq := os

  let rec backward_step (t : Nat) (_h_t : t ≤ seqLen)
      (dH_next : HiddenVector α hiddenSize)
      (dC_next : HiddenVector α hiddenSize)
      (acc_inputs : List (InputVector α inputSize))
      (accWf : WeightMatrix α hiddenSize (inputSize + hiddenSize)) (accbf : HiddenVector α
        hiddenSize)
      (accWi : WeightMatrix α hiddenSize (inputSize + hiddenSize)) (accbi : HiddenVector α
        hiddenSize)
      (accWc : WeightMatrix α hiddenSize (inputSize + hiddenSize)) (accbc : HiddenVector α
        hiddenSize)
      (accWo : WeightMatrix α hiddenSize (inputSize + hiddenSize)) (accbo : HiddenVector α
        hiddenSize) :
      (List (InputVector α inputSize) × HiddenVector α hiddenSize × HiddenVector α hiddenSize ×
        WeightMatrix α hiddenSize (inputSize + hiddenSize) × HiddenVector α hiddenSize ×
        WeightMatrix α hiddenSize (inputSize + hiddenSize) × HiddenVector α hiddenSize ×
        WeightMatrix α hiddenSize (inputSize + hiddenSize) × HiddenVector α hiddenSize ×
        WeightMatrix α hiddenSize (inputSize + hiddenSize) × HiddenVector α hiddenSize) :=
    if ht : t > 0 then
      let time_idx := t - 1
      have h_pred : t - 1 < t := by
        simpa [Nat.pred_eq_sub_one] using Nat.pred_lt (Nat.ne_of_gt ht)
      have h_time : time_idx < seqLen := lt_of_lt_of_le h_pred _h_t
      let input_t := getAtSpec inputs ⟨time_idx, h_time⟩
      let grad_h_t := getAtSpec grad_hiddens ⟨time_idx, h_time⟩
      let total_dH := addSpec grad_h_t dH_next

      let st_t := state_seq.getD time_idx initial_state
      let prev_st :=
        if hprev : time_idx > 0 then
          state_seq.getD (time_idx - 1) initial_state
        else
          initial_state

      let f_t := f_seq.getD time_idx (fill 0 (.dim hiddenSize .scalar))
      let i_t := i_seq.getD time_idx (fill 0 (.dim hiddenSize .scalar))
      let g_t := g_seq.getD time_idx (fill 0 (.dim hiddenSize .scalar))
      let o_t := o_seq.getD time_idx (fill 0 (.dim hiddenSize .scalar))

      let (dInput_t, dPrevState, dWf_t, dbf_t, dWi_t, dbi_t, dWc_t, dbc_t, dWo_t, dbo_t) :=
        lstmCellBackwardSpec lstm input_t prev_st st_t f_t i_t g_t o_t total_dH dC_next

      have h_t' : t - 1 ≤ seqLen := le_trans (Nat.sub_le t 1) _h_t
      backward_step (t - 1) h_t' dPrevState.hidden dPrevState.cell (dInput_t :: acc_inputs)
        (addSpec accWf dWf_t) (addSpec accbf dbf_t)
        (addSpec accWi dWi_t) (addSpec accbi dbi_t)
        (addSpec accWc dWc_t) (addSpec accbc dbc_t)
        (addSpec accWo dWo_t) (addSpec accbo dbo_t)
    else
      (acc_inputs, dH_next, dC_next, accWf, accbf, accWi, accbi, accWc, accbc, accWo, accbo)

  let (dInputs_list, dH0, dC0, dWf, dbf, dWi, dbi, dWc, dbc, dWo, dbo) :=
    backward_step seqLen le_rfl (fill 0 (.dim hiddenSize .scalar)) (fill 0 (.dim hiddenSize
      .scalar)) []
      (fill 0 (.dim hiddenSize (.dim (inputSize + hiddenSize) .scalar))) (fill 0 (.dim hiddenSize
        .scalar))
      (fill 0 (.dim hiddenSize (.dim (inputSize + hiddenSize) .scalar))) (fill 0 (.dim hiddenSize
        .scalar))
      (fill 0 (.dim hiddenSize (.dim (inputSize + hiddenSize) .scalar))) (fill 0 (.dim hiddenSize
        .scalar))
      (fill 0 (.dim hiddenSize (.dim (inputSize + hiddenSize) .scalar))) (fill 0 (.dim hiddenSize
        .scalar))

  let dInputs :=
    match dInputs_list with
    | [] => fill 0 (.dim seqLen (.dim inputSize .scalar))
    | h :: _ => Tensor.dim (fun i => dInputs_list.getD i.val h)

  (dWf, dbf, dWi, dbi, dWc, dbc, dWo, dbo, dInputs, ⟨dH0, dC0⟩)

end Spec
