/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Rnn

/-!
# GRU (spec layer)

TorchLean provides a small GRU specification that is:

- explicit about shapes (so dimension mistakes are caught early),
- explicit about the math (so we can reason about it and differentiate it),
- close in spirit to PyTorch's `nn.GRUCell` / `nn.GRU` documentation.

## References (math + PyTorch behavior)

- Cho et al.,
  "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation"
  (EMNLP 2014): https://aclanthology.org/D14-1179/ (PDF: https://aclanthology.org/D14-1179.pdf)
- Chung et al., "Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling"
  (2014):
  https://arxiv.org/abs/1412.3555
- PyTorch `GRUCell` equations: https://docs.pytorch.org/docs/stable/generated/torch.nn.GRUCell.html
- PyTorch `GRU` equations:
  https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.rnn.GRU.html

## Notes on parameterization

The GRU equations are often written with separate matrices `W_*` for the input and `U_*` for the
hidden state. In this spec we use a single matrix per gate applied to a concatenated vector
`[x_t; h_{t-1}]` (or `[x_t; r_t ⊙ h_{t-1}]` for the candidate). This is the same idea, just packaged
in a way that reuses the tensor building blocks already present in the spec layer.

One small place where libraries differ is the candidate equation: some implementations apply the
reset gate before the hidden-state linear map (as in Cho et al.), while others apply it after a
hidden-state linear map (as in the PyTorch docs). This file follows the former, because it matches
the original GRU equations and stays close to the "concatenate then multiply" style used elsewhere
in TorchLean's spec layer.
-/

@[expose] public section


namespace Spec

open Tensor
open Activation

variable {α : Type} [Context α]

-- GRU cell specification: separate weights for reset, update, and new gates
-- Each gate has weights [hidden_size, input_size + hidden_size] and bias [hidden_size]
/--
Parameters for a single GRU cell.

This is the spec-level analogue of PyTorch `torch.nn.GRUCell` parameters, using a concatenated
input `[x_t; h_{t-1}]` (shape `inputSize + hiddenSize`) for the reset/update gates and
`[x_t; r_t ⊙ h_{t-1}]` for the candidate gate.

Shapes:
- each `*_weights` is `[hiddenSize, inputSize + hiddenSize]`,
- each `*_bias` is `[hiddenSize]`.
-/
structure GRUSpec (α : Type) (inputSize hiddenSize : Nat) where
  /-- Reset-gate weights for `r_t = sigmoid(W_r [x_t; h_{t-1}] + b_r)`. -/
  reset_weights : Tensor α (.dim hiddenSize (.dim (inputSize + hiddenSize) .scalar))
  /-- Reset-gate bias. -/
  reset_bias    : Tensor α (.dim hiddenSize .scalar)
  /-- Update-gate weights for `z_t = sigmoid(W_z [x_t; h_{t-1}] + b_z)`. -/
  update_weights : Tensor α (.dim hiddenSize (.dim (inputSize + hiddenSize) .scalar))
  /-- Update-gate bias. -/
  update_bias    : Tensor α (.dim hiddenSize .scalar)
  /-- Candidate-state weights for `n_t = tanh(W_n [x_t; r_t ⊙ h_{t-1}] + b_n)`. -/
  new_weights   : Tensor α (.dim hiddenSize (.dim (inputSize + hiddenSize) .scalar))
  /-- Candidate-state bias. -/
  new_bias      : Tensor α (.dim hiddenSize .scalar)

-- Single GRU cell forward pass
/--
Forward pass for a single GRU cell.

Given input `x_t` and previous hidden state `h_{t-1}`, compute the next hidden state `h_t` using
the standard GRU equations.

PyTorch analogue: `torch.nn.GRUCell` forward (see module header links).
-/
def gruCellSpec {inputSize hiddenSize : Nat}
  (gru : GRUSpec α inputSize hiddenSize)
  (input : Tensor α (.dim inputSize .scalar))
  (prev_hidden : Tensor α (.dim hiddenSize .scalar)) :
  Tensor α (.dim hiddenSize .scalar) :=
  -- We follow the textbook GRU layout:
  --
  --   r_t = sigmoid(W_r [x_t; h_{t-1}] + b_r)
  --   z_t = sigmoid(W_z [x_t; h_{t-1}] + b_z)
  --   n_t = tanh   (W_n [x_t; r_t ⊙ h_{t-1}] + b_n)
  --   h_t = (1 - z_t) ⊙ n_t + z_t ⊙ h_{t-1}
  --
  -- This is the same update rule written in the PyTorch docs for `GRUCell`.
  let concat := concatVectorsSpec input prev_hidden

  -- Reset gate.
  let reset_gate := sigmoidSpec (addSpec (matVecMulSpec gru.reset_weights concat)
    gru.reset_bias)

  -- Update gate.
  let update_gate := sigmoidSpec (addSpec (matVecMulSpec gru.update_weights concat)
    gru.update_bias)

  -- The reset gate decides what portion of the previous state is used in the candidate update.
  let reset_hidden := mulSpec reset_gate prev_hidden

  -- Candidate uses `[x_t; r_t ⊙ h_{t-1}]`.
  let reset_concat := concatVectorsSpec input reset_hidden

  -- Candidate (sometimes called `n_t` or `h~_t` in the literature).
  let new_candidate := tanhSpec (addSpec (matVecMulSpec gru.new_weights reset_concat)
    gru.new_bias)

  -- Final hidden state (PyTorch's convention):
  --   h_t = (1 - z_t) ⊙ n_t + z_t ⊙ h_{t-1}.
  let one_minus_update := subSpec (fill 1 (.dim hiddenSize .scalar)) update_gate
  let new_contribution := mulSpec one_minus_update new_candidate
  let hidden_contribution := mulSpec update_gate prev_hidden
  addSpec new_contribution hidden_contribution

-- GRU sequence forward pass: processes a sequence of inputs
/--
Unroll a GRU over `seqLen` timesteps (time-major).

This returns the sequence of hidden states `[h_0, ..., h_{seqLen-1}]`. It is a pure spec-level
definition of semantics; an efficient runtime is free to implement the same behavior with loops and
caching.

PyTorch analogue: `torch.nn.GRU` run on a time-major input (or `batch_first=false`), returning the
output sequence.
-/
def gruSequenceSpec {seqLen inputSize hiddenSize : Nat}
  (gru : GRUSpec α inputSize hiddenSize)
  (inputs : Tensor α (.dim seqLen (.dim inputSize .scalar)))
  (initial_hidden : Tensor α (.dim hiddenSize .scalar)) :
  Tensor α (.dim seqLen (.dim hiddenSize .scalar)) :=
  -- This is a "spec-level" definition: it states the semantics of a GRU unrolled over time.
  -- A runtime can implement the same behavior with an efficient loop and appropriate caching.
  let rec process_sequence (t : Nat) (prev_hidden : Tensor α (.dim hiddenSize .scalar))
    : (Tensor α (.dim hiddenSize .scalar) × List (Tensor α (.dim hiddenSize .scalar))) :=
    if h : t < seqLen then
      let input_t := getAtSpec inputs ⟨t, h⟩
      let hidden_t := gruCellSpec gru input_t prev_hidden
      let (final_hidden, rest_outputs) := process_sequence (t + 1) hidden_t
      (final_hidden, hidden_t :: rest_outputs)
    else
      (prev_hidden, [])

  let (_, outputs_rev) := process_sequence 0 initial_hidden
  let outputs := outputs_rev.reverse
  -- Convert list to tensor
  match outputs with
  | [] => Tensor.dim (fun _ => initial_hidden) -- should not happen if seqLen > 0
  | h :: _ => Tensor.dim (fun i => outputs.getD i.val h)

-- GRU cell forward pass that returns all intermediate values for BPTT
/--
GRU cell forward pass that also returns cached intermediates for BPTT.

This computes the same next hidden state as `gru_cell_spec`, but additionally returns:
- `reset_gate` (`r_t`),
- `update_gate` (`z_t`),
- `new_candidate` (`n_t`), and
- `reset_hidden` (`r_t ⊙ h_{t-1}`).

These are exactly the quantities commonly saved by a reverse-mode implementation (PyTorch-style
autograd) to compute gradients efficiently in the backward pass.
-/
def gruCellSpecWithIntermediates {inputSize hiddenSize : Nat}
  (gru : GRUSpec α inputSize hiddenSize)
  (input : Tensor α (.dim inputSize .scalar))
  (prev_hidden : Tensor α (.dim hiddenSize .scalar)) :
  (Tensor α (.dim hiddenSize .scalar) ×  -- new_hidden
   Tensor α (.dim hiddenSize .scalar) ×  -- reset_gate
   Tensor α (.dim hiddenSize .scalar) ×  -- update_gate
   Tensor α (.dim hiddenSize .scalar) ×  -- new_candidate
   Tensor α (.dim hiddenSize .scalar)) := -- reset_hidden
  -- Same computation as `gru_cell_spec`, but we also return the gate activations and the candidate.
  -- Those values are what a "tape" would store for a standard BPTT implementation.
  let concat := concatVectorsSpec input prev_hidden

  -- Reset gate: r_t = σ(W_r @ [x_t; h_{t-1}] + b_r)
  let reset_gate := sigmoidSpec (addSpec (matVecMulSpec gru.reset_weights concat)
    gru.reset_bias)

  -- Update gate: z_t = σ(W_z @ [x_t; h_{t-1}] + b_z)
  let update_gate := sigmoidSpec (addSpec (matVecMulSpec gru.update_weights concat)
    gru.update_bias)

  -- Reset hidden state: h_reset = r_t ⊙ h_{t-1}
  let reset_hidden := mulSpec reset_gate prev_hidden

  -- Concatenate input with reset hidden state
  let reset_concat := concatVectorsSpec input reset_hidden

  -- New hidden state candidate: ĥ_t = tanh(W_h @ [x_t; r_t ⊙ h_{t-1}] + b_h)
  let new_candidate := tanhSpec (addSpec (matVecMulSpec gru.new_weights reset_concat)
    gru.new_bias)

  -- Final hidden state follows the same convention as `gru_cell_spec`.
  let one_minus_update := subSpec (fill 1 (.dim hiddenSize .scalar)) update_gate
  let new_contribution := mulSpec one_minus_update new_candidate
  let hidden_contribution := mulSpec update_gate prev_hidden
  let new_hidden := addSpec new_contribution hidden_contribution

  (new_hidden, reset_gate, update_gate, new_candidate, reset_hidden)

/--
Run a GRU forward pass while collecting the per-timestep intermediates needed for BPTT.

This is the "spec-level" analogue of what frameworks do internally:

- the forward pass produces `h_t`,
- and it also saves gate activations (`r_t`, `z_t`) and the candidate (`n_t`) for the backward pass.

The returned tensors are all time-major (`seqLen` first) to match the rest of the spec layer.
-/
def gruExtractIntermediateValues {seqLen inputSize hiddenSize : Nat}
  (gru : GRUSpec α inputSize hiddenSize)
  (inputs : Tensor α (.dim seqLen (.dim inputSize .scalar)))
  (initial_hidden : Tensor α (.dim hiddenSize .scalar)) :
  (Tensor α (.dim seqLen (.dim hiddenSize .scalar)) ×  -- hidden_states
   Tensor α (.dim seqLen (.dim hiddenSize .scalar)) ×  -- reset_gates
   Tensor α (.dim seqLen (.dim hiddenSize .scalar)) ×  -- update_gates
   Tensor α (.dim seqLen (.dim hiddenSize .scalar)) ×  -- new_candidates
   Tensor α (.dim seqLen (.dim hiddenSize .scalar))) := -- reset_hiddens
  -- Process sequence with intermediate value storage
  let rec process_sequence (t : Nat) (prev_hidden : Tensor α (.dim hiddenSize .scalar))
    : (Tensor α (.dim hiddenSize .scalar) ×
       List (Tensor α (.dim hiddenSize .scalar)) ×  -- hidden_states
       List (Tensor α (.dim hiddenSize .scalar)) ×  -- reset_gates
       List (Tensor α (.dim hiddenSize .scalar)) ×  -- update_gates
       List (Tensor α (.dim hiddenSize .scalar)) ×  -- new_candidates
       List (Tensor α (.dim hiddenSize .scalar))) := -- reset_hiddens
    if h : t < seqLen then
      let input_t := getAtSpec inputs ⟨t, h⟩
      let (hidden_t, reset_gate_t, update_gate_t, new_candidate_t, reset_hidden_t) :=
        gruCellSpecWithIntermediates gru input_t prev_hidden
      let (final_hidden, rest_hiddens, rest_resets, rest_updates, rest_candidates,
        rest_reset_hiddens) :=
        process_sequence (t + 1) hidden_t
      (final_hidden,
       hidden_t :: rest_hiddens,
       reset_gate_t :: rest_resets,
       update_gate_t :: rest_updates,
       new_candidate_t :: rest_candidates,
       reset_hidden_t :: rest_reset_hiddens)
    else
      (prev_hidden, [], [], [], [], [])

  let (_, hiddens_rev, resets_rev, updates_rev, candidates_rev, reset_hiddens_rev) :=
    process_sequence 0 initial_hidden

  let hiddens := hiddens_rev.reverse
  let resets := resets_rev.reverse
  let updates := updates_rev.reverse
  let candidates := candidates_rev.reverse
  let reset_hiddens := reset_hiddens_rev.reverse

  -- Convert lists to tensors
  let hidden_states := match hiddens with
  | [] => Tensor.dim (fun _ => initial_hidden)
  | h :: _ => Tensor.dim (fun i => hiddens.getD i.val h)

  let reset_gates := match resets with
  | [] => Tensor.dim (fun _ => fill 0 (.dim hiddenSize .scalar))
  | h :: _ => Tensor.dim (fun i => resets.getD i.val h)

  let update_gates := match updates with
  | [] => Tensor.dim (fun _ => fill 0 (.dim hiddenSize .scalar))
  | h :: _ => Tensor.dim (fun i => updates.getD i.val h)

  let new_candidates := match candidates with
  | [] => Tensor.dim (fun _ => fill 0 (.dim hiddenSize .scalar))
  | h :: _ => Tensor.dim (fun i => candidates.getD i.val h)

  let reset_hiddens := match reset_hiddens with
  | [] => Tensor.dim (fun _ => fill 0 (.dim hiddenSize .scalar))
  | h :: _ => Tensor.dim (fun i => reset_hiddens.getD i.val h)

  (hidden_states, reset_gates, update_gates, new_candidates, reset_hiddens)

-- Batched GRU sequence forward pass
/--
Batched GRU forward pass (map `gruSequenceSpec` over the batch dimension).

This is a simple spec-level definition for semantics, not an optimized kernel.
PyTorch analogue: `torch.nn.GRU` on a batched input tensor.
-/
def gruBatchedSpec {batchSize seqLen inputSize hiddenSize : Nat}
  (gru : GRUSpec α inputSize hiddenSize)
  (inputs : Tensor α (.dim batchSize (.dim seqLen (.dim inputSize .scalar))))
  (initial_hidden : Tensor α (.dim batchSize (.dim hiddenSize .scalar))) :
  Tensor α (.dim batchSize (.dim seqLen (.dim hiddenSize .scalar))) :=
  -- This is a simple "map over the batch dimension".
  -- It matches the semantics of a batched GRU, but it is not an optimized runtime kernel.
  match inputs, initial_hidden with
  | Tensor.dim batch_inputs, Tensor.dim batch_hidden =>
    Tensor.dim (fun b =>
      gruSequenceSpec gru (batch_inputs b) (batch_hidden b))

-- Gradient computations for GRU

-- Gradient w.r.t. reset gate weights
/--
Reference gradient for reset-gate weights via the generic RNN weight-gradient helper.

This uses `rnn_weights_deriv_spec` on the concatenated inputs/hidden states. It is a convenient
building block, but the more explicit `*_bptt_spec` helpers below show the time-unrolled
accumulation form.
-/
def gruResetWeightsDerivSpec {seqLen inputSize hiddenSize : Nat}
  (inputs : Tensor α (.dim seqLen (.dim inputSize .scalar)))
  (hiddens : Tensor α (.dim seqLen (.dim hiddenSize .scalar)))
  (grad_reset : Tensor α (.dim seqLen (.dim hiddenSize .scalar))) :
  Tensor α (.dim hiddenSize (.dim (inputSize + hiddenSize) .scalar)) :=
  rnnWeightsDerivSpec inputs hiddens grad_reset

-- Gradient w.r.t. update gate weights
/-- Reference gradient for update-gate weights (via `rnn_weights_deriv_spec`). -/
def gruUpdateWeightsDerivSpec {seqLen inputSize hiddenSize : Nat}
  (inputs : Tensor α (.dim seqLen (.dim inputSize .scalar)))
  (hiddens : Tensor α (.dim seqLen (.dim hiddenSize .scalar)))
  (grad_update : Tensor α (.dim seqLen (.dim hiddenSize .scalar))) :
  Tensor α (.dim hiddenSize (.dim (inputSize + hiddenSize) .scalar)) :=
  rnnWeightsDerivSpec inputs hiddens grad_update

-- Gradient w.r.t. new gate weights
/--
Reference gradient for candidate ("new") gate weights (via `rnn_weights_deriv_spec`).

Note the second sequence argument is `reset_hiddens = r_t ⊙ h_{t-1}`.
-/
def gruNewWeightsDerivSpec {seqLen inputSize hiddenSize : Nat}
  (inputs : Tensor α (.dim seqLen (.dim inputSize .scalar)))
  (reset_hiddens : Tensor α (.dim seqLen (.dim hiddenSize .scalar))) -- r_t ⊙ h_{t-1}
  (grad_new : Tensor α (.dim seqLen (.dim hiddenSize .scalar))) :
  Tensor α (.dim hiddenSize (.dim (inputSize + hiddenSize) .scalar)) :=
  rnnWeightsDerivSpec inputs reset_hiddens grad_new

-- Gradient w.r.t. biases (sum over sequence length)
/--
Bias gradient by summing per-timestep gradients over the time axis.

This is the spec-level analogue of the common "sum across batch/time" reduction used for bias
gradients. The `seqLen ≠ 0` hypothesis is exactly what makes axis `0` a valid reduction axis for
`reduce_sum_auto` in the shape-indexed tensor API.
-/
def gruBiasDerivSpec {seqLen hiddenSize : Nat}
  (grad_outputs : Tensor α (.dim seqLen (.dim hiddenSize .scalar)))
  (h : seqLen ≠ 0) :
  Tensor α (.dim hiddenSize .scalar) :=
  -- `reduce_sum_auto` wants a proof that reducing along axis 0 is valid.
  -- The `seqLen ≠ 0` side condition is exactly what makes axis 0 meaningful here.
  letI : Shape.valid_axis_inst 0 (Shape.dim seqLen (Shape.dim hiddenSize Shape.scalar)) :=
    Shape.validAxisInstZeroAlt h
  reduceSumAuto 0 grad_outputs

-- Gradient w.r.t. reset gate weights with proper BPTT
/--
Reset-gate weight gradient by explicit time-unrolled accumulation (BPTT-style).

This computes `Σ_t (dL/dr_t) ⊗ [x_t; h_{t-1}]`, where `⊗` is an outer product.
-/
def gruResetWeightsDerivBpttSpec {seqLen inputSize hiddenSize : Nat}
  (inputs : Tensor α (.dim seqLen (.dim inputSize .scalar)))
  (hiddens : Tensor α (.dim seqLen (.dim hiddenSize .scalar)))
  (_reset_gates : Tensor α (.dim seqLen (.dim hiddenSize .scalar)))
  (grad_reset_gates : Tensor α (.dim seqLen (.dim hiddenSize .scalar))) :
  Tensor α (.dim hiddenSize (.dim (inputSize + hiddenSize) .scalar)) :=
  -- Accumulate gradients over time steps
  let rec accumulate_grads (t : Nat) (acc : Tensor α (.dim hiddenSize (.dim (inputSize + hiddenSize)
    .scalar))) :
    Tensor α (.dim hiddenSize (.dim (inputSize + hiddenSize) .scalar)) :=
    if h : t < seqLen then
      let input_t := getAtSpec inputs ⟨t, h⟩
      let hidden_prev :=
        if ht : t > 0 then
          have ht0 : t ≠ 0 := Nat.ne_of_gt ht
          have htPred : t - 1 < t := by
            simpa [Nat.pred_eq_sub_one] using Nat.pred_lt ht0
          have htPrev : t - 1 < seqLen := lt_trans htPred h
          getAtSpec hiddens ⟨t - 1, htPrev⟩
        else
          fill 0 (.dim hiddenSize .scalar)
      let concat_t := concatVectorsSpec input_t hidden_prev
      let grad_reset_t := getAtSpec grad_reset_gates ⟨t, h⟩
      let grad_w_t := outerProductSpec grad_reset_t concat_t
      accumulate_grads (t + 1) (addSpec acc grad_w_t)
    else acc
  accumulate_grads 0 (fill 0 (.dim hiddenSize (.dim (inputSize + hiddenSize) .scalar)))

-- Gradient w.r.t. update gate weights with proper BPTT
/--
Update-gate weight gradient by explicit time-unrolled accumulation (BPTT-style).

This computes `Σ_t (dL/dz_t) ⊗ [x_t; h_{t-1}]`.
-/
def gruUpdateWeightsDerivBpttSpec {seqLen inputSize hiddenSize : Nat}
  (inputs : Tensor α (.dim seqLen (.dim inputSize .scalar)))
  (hiddens : Tensor α (.dim seqLen (.dim hiddenSize .scalar)))
  (grad_update_gates : Tensor α (.dim seqLen (.dim hiddenSize .scalar))) :
  Tensor α (.dim hiddenSize (.dim (inputSize + hiddenSize) .scalar)) :=
  -- Accumulate gradients over time steps
  let rec accumulate_grads (t : Nat) (acc : Tensor α (.dim hiddenSize (.dim (inputSize + hiddenSize)
    .scalar))) :
    Tensor α (.dim hiddenSize (.dim (inputSize + hiddenSize) .scalar)) :=
    if h : t < seqLen then
      let input_t := getAtSpec inputs ⟨t, h⟩
      let hidden_prev :=
        if ht : t > 0 then
          have ht0 : t ≠ 0 := Nat.ne_of_gt ht
          have htPred : t - 1 < t := by
            simpa [Nat.pred_eq_sub_one] using Nat.pred_lt ht0
          have htPrev : t - 1 < seqLen := lt_trans htPred h
          getAtSpec hiddens ⟨t - 1, htPrev⟩
        else
          fill 0 (.dim hiddenSize .scalar)
      let concat_t := concatVectorsSpec input_t hidden_prev
      let grad_update_t := getAtSpec grad_update_gates ⟨t, h⟩
      let grad_w_t := outerProductSpec grad_update_t concat_t
      accumulate_grads (t + 1) (addSpec acc grad_w_t)
    else acc
  accumulate_grads 0 (fill 0 (.dim hiddenSize (.dim (inputSize + hiddenSize) .scalar)))

-- Gradient w.r.t. new gate weights with proper BPTT
/--
Candidate-gate weight gradient by explicit time-unrolled accumulation (BPTT-style).

This computes `Σ_t (dL/dn_t) ⊗ [x_t; r_t ⊙ h_{t-1}]`.
-/
def gruNewWeightsDerivBpttSpec {seqLen inputSize hiddenSize : Nat}
  (inputs : Tensor α (.dim seqLen (.dim inputSize .scalar)))
  (reset_hiddens : Tensor α (.dim seqLen (.dim hiddenSize .scalar))) -- r_t ⊙ h_{t-1}
  (grad_new_candidates : Tensor α (.dim seqLen (.dim hiddenSize .scalar))) :
  Tensor α (.dim hiddenSize (.dim (inputSize + hiddenSize) .scalar)) :=
  -- Accumulate gradients over time steps
  let rec accumulate_grads (t : Nat) (acc : Tensor α (.dim hiddenSize (.dim (inputSize + hiddenSize)
    .scalar))) :
    Tensor α (.dim hiddenSize (.dim (inputSize + hiddenSize) .scalar)) :=
    if h : t < seqLen then
      let input_t := getAtSpec inputs ⟨t, h⟩
      let reset_hidden_t := getAtSpec reset_hiddens ⟨t, h⟩
      let concat_t := concatVectorsSpec input_t reset_hidden_t
      let grad_new_t := getAtSpec grad_new_candidates ⟨t, h⟩
      let grad_w_t := outerProductSpec grad_new_t concat_t
      accumulate_grads (t + 1) (addSpec acc grad_w_t)
    else acc
  accumulate_grads 0 (fill 0 (.dim hiddenSize (.dim (inputSize + hiddenSize) .scalar)))

/--
Backward (VJP) for a single GRU cell.

Inputs:
- the cell parameters `gru`,
- the current input `x_t`,
- the previous hidden state `h_{t-1}`,
- an upstream gradient `dL/dh_t`,
- and the forward intermediates (`r_t`, `z_t`, `n_t`) that a typical BPTT implementation would
  cache.

Outputs:
- gradients w.r.t. the input and previous hidden state,
- plus gradients for each parameter tensor (weights and biases).

This is written to match the forward equations in `gru_cell_spec`. It is not an optimized kernel;
it is a precise spec for what gradients *should* be.
-/
def gruCellBackwardFullSpec {inputSize hiddenSize : Nat}
  (gru : GRUSpec α inputSize hiddenSize)
  (input : Tensor α (.dim inputSize .scalar))
  (prev_hidden : Tensor α (.dim hiddenSize .scalar))
  (grad_output : Tensor α (.dim hiddenSize .scalar))
  (reset_gate : Tensor α (.dim hiddenSize .scalar))
  (update_gate : Tensor α (.dim hiddenSize .scalar))
  (new_candidate : Tensor α (.dim hiddenSize .scalar)) :
  ( Tensor α (.dim inputSize .scalar) ×                     -- dInput
    Tensor α (.dim hiddenSize .scalar) ×                    -- dPrevHidden
    Tensor α (.dim hiddenSize (.dim (inputSize + hiddenSize) .scalar)) ×  -- dResetW
    Tensor α (.dim hiddenSize .scalar) ×                    -- dResetB
    Tensor α (.dim hiddenSize (.dim (inputSize + hiddenSize) .scalar)) ×  -- dUpdateW
    Tensor α (.dim hiddenSize .scalar) ×                    -- dUpdateB
    Tensor α (.dim hiddenSize (.dim (inputSize + hiddenSize) .scalar)) ×  -- dNewW
    Tensor α (.dim hiddenSize .scalar)                      -- dNewB
  ) :=
  let concat := concatVectorsSpec input prev_hidden

  -- Start from the output equation:
  --   h = (1 - z) ⊙ n + z ⊙ h_prev
  --
  -- This yields three immediate partials:
  --   d n      = d h ⊙ (1 - z)
  --   d z      = d h ⊙ (h_prev - n)
  --   d h_prev (direct) = d h ⊙ z
  let one_minus_z := subSpec (fill 1 (.dim hiddenSize .scalar)) update_gate
  let dHtilde := mulSpec grad_output one_minus_z
  let dZ := mulSpec grad_output (subSpec prev_hidden new_candidate)
  let dPrev_direct := mulSpec grad_output update_gate

  -- tanh preactivation derivative using output new_candidate = tanh(pre_h)
  let dPre_h := mulSpec dHtilde (subSpec (fill 1 (.dim hiddenSize .scalar)) (mulSpec
    new_candidate new_candidate))

  -- h_reset = r ⊙ h_prev, reset_concat = [x; h_reset]
  let reset_hidden := mulSpec reset_gate prev_hidden
  let reset_concat := concatVectorsSpec input reset_hidden

  -- New gate grads.
  let dNewW := outerProductSpec dPre_h reset_concat
  let dNewB := dPre_h
  let dResetConcat := vecMatMulSpec dPre_h gru.new_weights
  let dX_from_h := sliceVectorSpec dResetConcat 0 inputSize (by
    simp)
  let dHreset := sliceVectorSpec dResetConcat inputSize hiddenSize (by
    simp)

  -- Backprop through reset_hidden = r ⊙ h_prev.
  let dR_from_reset := mulSpec dHreset prev_hidden
  let dPrev_from_reset := mulSpec dHreset reset_gate

  -- Reset gate grads: r = sigmoid(pre_r)
  let dPre_r := mulSpec dR_from_reset (Activation.sigmoidOutputDerivSpec reset_gate)
  let dResetW := outerProductSpec dPre_r concat
  let dResetB := dPre_r
  let dConcat_from_r := vecMatMulSpec dPre_r gru.reset_weights
  let dX_from_r := sliceVectorSpec dConcat_from_r 0 inputSize (by
    simp)
  let dPrev_from_r := sliceVectorSpec dConcat_from_r inputSize hiddenSize (by
    simp)

  -- Update gate grads: z = sigmoid(pre_z)
  let dPre_z := mulSpec dZ (Activation.sigmoidOutputDerivSpec update_gate)
  let dUpdateW := outerProductSpec dPre_z concat
  let dUpdateB := dPre_z
  let dConcat_from_z := vecMatMulSpec dPre_z gru.update_weights
  let dX_from_z := sliceVectorSpec dConcat_from_z 0 inputSize (by
    simp)
  let dPrev_from_z := sliceVectorSpec dConcat_from_z inputSize hiddenSize (by
    simp)

  let dInput := addSpec (addSpec dX_from_h dX_from_r) dX_from_z
  let dPrevHidden := addSpec (addSpec (addSpec dPrev_direct dPrev_from_reset) dPrev_from_r)
    dPrev_from_z

  (dInput, dPrevHidden, dResetW, dResetB, dUpdateW, dUpdateB, dNewW, dNewB)

/--
Reverse-mode backprop through an unrolled GRU over `seqLen` steps (BPTT).

This function consumes the same intermediates produced by `gru_extract_intermediate_values`:
per-timestep gate activations and candidates. This mirrors the PyTorch mental model: the forward
pass produces a sequence of hidden states and saves what it needs; the backward pass walks time
in reverse and accumulates gradients.
-/
def gruSequenceBackwardFullSpec {seqLen inputSize hiddenSize : Nat}
  (gru : GRUSpec α inputSize hiddenSize)
  (inputs : Tensor α (.dim seqLen (.dim inputSize .scalar)))
  (hiddens : Tensor α (.dim seqLen (.dim hiddenSize .scalar)))
  (grad_outputs : Tensor α (.dim seqLen (.dim hiddenSize .scalar)))
  (reset_gates : Tensor α (.dim seqLen (.dim hiddenSize .scalar)))
  (update_gates : Tensor α (.dim seqLen (.dim hiddenSize .scalar)))
  (new_candidates : Tensor α (.dim seqLen (.dim hiddenSize .scalar)))
  (initial_hidden : Tensor α (.dim hiddenSize .scalar) := fill 0 (.dim hiddenSize .scalar)) :
  ( Tensor α (.dim hiddenSize (.dim (inputSize + hiddenSize) .scalar)) ×  -- dResetW
    Tensor α (.dim hiddenSize .scalar) ×                                  -- dResetB
    Tensor α (.dim hiddenSize (.dim (inputSize + hiddenSize) .scalar)) ×  -- dUpdateW
    Tensor α (.dim hiddenSize .scalar) ×                                  -- dUpdateB
    Tensor α (.dim hiddenSize (.dim (inputSize + hiddenSize) .scalar)) ×  -- dNewW
    Tensor α (.dim hiddenSize .scalar) ×                                  -- dNewB
    Tensor α (.dim seqLen (.dim inputSize .scalar)) ×                     -- dInputs
    Tensor α (.dim hiddenSize .scalar)                                    -- dInitialHidden
  ) :=

  let rec backward_step (t : Nat) (_h_t : t ≤ seqLen)
      (dHidden_next : Tensor α (.dim hiddenSize .scalar))
      (acc_inputs : List (Tensor α (.dim inputSize .scalar)))
      (dResetW : Tensor α (.dim hiddenSize (.dim (inputSize + hiddenSize) .scalar)))
      (dResetB : Tensor α (.dim hiddenSize .scalar))
      (dUpdateW : Tensor α (.dim hiddenSize (.dim (inputSize + hiddenSize) .scalar)))
      (dUpdateB : Tensor α (.dim hiddenSize .scalar))
      (dNewW : Tensor α (.dim hiddenSize (.dim (inputSize + hiddenSize) .scalar)))
      (dNewB : Tensor α (.dim hiddenSize .scalar)) :
      ( List (Tensor α (.dim inputSize .scalar)) × Tensor α (.dim hiddenSize .scalar) ×
        Tensor α (.dim hiddenSize (.dim (inputSize + hiddenSize) .scalar)) × Tensor α (.dim
          hiddenSize .scalar) ×
        Tensor α (.dim hiddenSize (.dim (inputSize + hiddenSize) .scalar)) × Tensor α (.dim
          hiddenSize .scalar) ×
        Tensor α (.dim hiddenSize (.dim (inputSize + hiddenSize) .scalar)) × Tensor α (.dim
          hiddenSize .scalar) ) :=
    if ht : t > 0 then
      let time_idx := t - 1
      have h_time : time_idx < seqLen := by
        have ht0 : t ≠ 0 := Nat.ne_of_gt ht
        have htPred : t - 1 < t := by
          simpa [Nat.pred_eq_sub_one] using Nat.pred_lt ht0
        have : t - 1 < seqLen := Nat.lt_of_lt_of_le htPred _h_t
        simpa [time_idx] using this
      let input_t := getAtSpec inputs ⟨time_idx, h_time⟩
      let prev_hidden :=
        if hprev : time_idx > 0 then
          have hp : time_idx - 1 < seqLen := by
            have hprev0 : time_idx ≠ 0 := Nat.ne_of_gt hprev
            have hPred : time_idx - 1 < time_idx := by
              simpa [Nat.pred_eq_sub_one] using Nat.pred_lt hprev0
            exact lt_trans hPred h_time
          getAtSpec hiddens ⟨time_idx - 1, hp⟩
        else
          initial_hidden

      let grad_out_t := getAtSpec grad_outputs ⟨time_idx, h_time⟩
      let total_grad := addSpec grad_out_t dHidden_next

      let reset_gate_t := getAtSpec reset_gates ⟨time_idx, h_time⟩
      let update_gate_t := getAtSpec update_gates ⟨time_idx, h_time⟩
      let new_candidate_t := getAtSpec new_candidates ⟨time_idx, h_time⟩

      let (dInput_t, dPrevHidden, dResetW_t, dResetB_t, dUpdateW_t, dUpdateB_t, dNewW_t, dNewB_t) :=
        gruCellBackwardFullSpec gru input_t prev_hidden total_grad reset_gate_t update_gate_t
          new_candidate_t

      have h_t' : t - 1 ≤ seqLen := le_trans (Nat.sub_le t 1) _h_t
      backward_step (t - 1) h_t' dPrevHidden (dInput_t :: acc_inputs)
        (addSpec dResetW dResetW_t) (addSpec dResetB dResetB_t)
        (addSpec dUpdateW dUpdateW_t) (addSpec dUpdateB dUpdateB_t)
        (addSpec dNewW dNewW_t) (addSpec dNewB dNewB_t)
    else
      (acc_inputs, dHidden_next, dResetW, dResetB, dUpdateW, dUpdateB, dNewW, dNewB)

  let (dInputs_list, dInitialHidden, dResetW, dResetB, dUpdateW, dUpdateB, dNewW, dNewB) :=
    backward_step seqLen (by simp) (fill 0 (.dim hiddenSize .scalar)) []
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

  (dResetW, dResetB, dUpdateW, dUpdateB, dNewW, dNewB, dInputs, dInitialHidden)

/--
Convenience wrapper: return only `(dInputs, dInitialHidden)` from `gruSequenceBackwardFullSpec`.

This is useful when you only need gradients w.r.t. the input sequence and initial hidden state, and
not the full parameter-gradient bundle.
-/
def gruSequenceBackwardSpec {seqLen inputSize hiddenSize : Nat}
  (gru : GRUSpec α inputSize hiddenSize)
  (inputs : Tensor α (.dim seqLen (.dim inputSize .scalar)))
  (hiddens : Tensor α (.dim seqLen (.dim hiddenSize .scalar)))
  (grad_outputs : Tensor α (.dim seqLen (.dim hiddenSize .scalar)))
  (reset_gates : Tensor α (.dim seqLen (.dim hiddenSize .scalar)))
  (update_gates : Tensor α (.dim seqLen (.dim hiddenSize .scalar)))
  (new_candidates : Tensor α (.dim seqLen (.dim hiddenSize .scalar)))
  (initial_hidden : Tensor α (.dim hiddenSize .scalar) := fill 0 (.dim hiddenSize .scalar)) :
  (Tensor α (.dim seqLen (.dim inputSize .scalar)) × Tensor α (.dim hiddenSize .scalar)) :=
  let (_, _, _, _, _, _, dInputs, dInitialHidden) :=
    gruSequenceBackwardFullSpec gru inputs hiddens grad_outputs reset_gates update_gates
      new_candidates initial_hidden
  (dInputs, dInitialHidden)

end Spec
