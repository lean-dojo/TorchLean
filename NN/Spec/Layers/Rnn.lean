/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Activation

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


namespace Spec

open Tensor
open Activation

variable {α : Type} [Context α]

/-!
## Common shape aliases

We use these aliases pervasively in the spec layer so that signatures read like the math:

- `InputVector α inputSize` is a length-`inputSize` vector,
- `HiddenVector α hiddenSize` is a length-`hiddenSize` vector,
- `WeightMatrix α hiddenSize inputSize` is a `(hiddenSize × inputSize)` matrix,
- `SequenceTensor α seqLen s` is a time-major sequence of length `seqLen`.

PyTorch note: `torch.nn.RNN` can be configured as batch-first or time-first. In the spec layer we
standardize on time-major (`seqLen` outermost) because it matches recursive definitions and proofs.
-/
/-- Shape alias: length-`inputSize` input vector. -/
abbrev InputVector (α : Type) (inputSize : Nat) := Tensor α (.dim inputSize .scalar)
/-- Shape alias: length-`hiddenSize` hidden-state vector. -/
abbrev HiddenVector (α : Type) (hiddenSize : Nat) := Tensor α (.dim hiddenSize .scalar)
/-- Shape alias: `(hiddenSize × inputSize)` dense weight matrix. -/
abbrev WeightMatrix (α : Type) (hiddenSize inputSize : Nat) := Tensor α (.dim hiddenSize (.dim
  inputSize .scalar))
/-- Shape alias: time-major sequence of length `seqLen` with per-step shape `shape`. -/
abbrev SequenceTensor (α : Type) (seqLen : Nat) (shape : Shape) := Tensor α (.dim seqLen shape)
/-- Shape alias: batch of `batchSize` tensors with per-item shape `shape`. -/
abbrev BatchedTensor (α : Type) (batchSize : Nat) (shape : Shape) := Tensor α (.dim batchSize shape)

/--
RNN cell parameters.

We use a single weight matrix applied to a concatenated vector `[x_t; h_{t-1}]`:

`h_t = tanh(W [x_t; h_{t-1}] + b)`.

This is equivalent to the common split-parameter form:

`h_t = tanh(W_ih x_t + W_hh h_{t-1} + b)`,

just packaged to reuse the same tensor primitives elsewhere in TorchLean.
-/
structure RNNSpec (α : Type) (inputSize hiddenSize : Nat) where
  /-- weights. -/
  weights : WeightMatrix α hiddenSize (inputSize + hiddenSize)
  /-- bias. -/
  bias    : HiddenVector α hiddenSize

/--
Single RNN cell forward pass.

Math:
`h_t = tanh(W [x_t; h_{t-1}] + b)`.

PyTorch analogy: `RNNCell(input, hidden)` with `tanh` nonlinearity.
-/
def rnnCellSpec {inputSize hiddenSize : Nat}
  (rnn : RNNSpec α inputSize hiddenSize)
  (input : InputVector α inputSize)
  (hidden : HiddenVector α hiddenSize) :
  HiddenVector α hiddenSize :=
  -- Concatenate input and hidden state
  let concat := concatVectorsSpec input hidden
  -- Apply linear transformation: Wx + b
  let linear_out := addSpec (matVecMulSpec rnn.weights concat) rnn.bias
  -- Apply tanh activation
  tanhSpec linear_out

-- ============================================================================
-- Backpropagation (BPTT)
-- ============================================================================

-- Single RNN cell backward pass.
-- Forward: h_t = tanh(W @ [x_t; h_{t-1}] + b)
-- Backward returns:
--   dX_t, dH_{t-1}, dW, db
/--
Backward/VJP for a single RNN cell.

Inputs:
- `x_t`, `h_{t-1}`,
- the cached forward output `h_t` (so we can write `tanh'` in terms of `h_t`),
- an upstream gradient `dL/dh_t`.

Outputs:
- `dL/dx_t`, `dL/dh_{t-1}`, and parameter gradients `(dL/dW, dL/db)`.
-/
def rnnCellBackwardSpec {inputSize hiddenSize : Nat}
  (rnn : RNNSpec α inputSize hiddenSize)
  (input : InputVector α inputSize)
  (prev_hidden : HiddenVector α hiddenSize)
  (hidden : HiddenVector α hiddenSize)
  (grad_hidden : HiddenVector α hiddenSize) :
  (InputVector α inputSize × HiddenVector α hiddenSize ×
    WeightMatrix α hiddenSize (inputSize + hiddenSize) × HiddenVector α hiddenSize) :=
  let concat := concatVectorsSpec input prev_hidden

  -- tanh'(z) = 1 - tanh(z)^2, and tanh(z) = hidden
  let tanh_deriv := subSpec (fill 1 (.dim hiddenSize .scalar)) (mulSpec hidden hidden)
  let grad_preact := mulSpec grad_hidden tanh_deriv

  let grad_weights := outerProductSpec grad_preact concat
  let grad_bias := grad_preact

  -- dConcat = grad_preactᵀ * W  (shape: inputSize + hiddenSize)
  let grad_concat := vecMatMulSpec grad_preact rnn.weights
  let grad_input := sliceVectorSpec grad_concat 0 inputSize (by simp)
  let grad_prev_hidden := sliceVectorSpec grad_concat inputSize hiddenSize (by simp)

  (grad_input, grad_prev_hidden, grad_weights, grad_bias)

-- RNN sequence forward pass: processes a sequence of inputs
/--
Unroll an RNN over `seqLen` steps (time-major).

Returns the sequence of hidden states `[h_0, ..., h_{seqLen-1}]`.
-/
def rnnSequenceSpec {seqLen inputSize hiddenSize : Nat}
  (rnn : RNNSpec α inputSize hiddenSize)
  (inputs : SequenceTensor α seqLen (.dim inputSize .scalar))
  (initial_hidden : HiddenVector α hiddenSize) :
  SequenceTensor α seqLen (.dim hiddenSize .scalar) :=
  let rec process_sequence (t : Nat) (prev_hidden : HiddenVector α hiddenSize)
    : (HiddenVector α hiddenSize × List (HiddenVector α hiddenSize)) :=
    if h : t < seqLen then
      let input_t := getAtSpec inputs ⟨t, h⟩
      let hidden_t := rnnCellSpec rnn input_t prev_hidden
      let (final_hidden, rest_outputs) := process_sequence (t + 1) hidden_t
      (final_hidden, hidden_t :: rest_outputs)
    else
      (prev_hidden, [])

  let (_, outputs_rev) := process_sequence 0 initial_hidden
  let outputs := outputs_rev.reverse
  -- Convert list to tensor
  match outputs with
  | [] =>
      -- Convention for `seqLen = 0`: there are no outputs, and the eliminator gives us a
      -- function `Fin 0 -> _` anyway.
      Tensor.dim (fun _ => initial_hidden)
  | h :: _ => Tensor.dim (fun i => outputs.getD i.val h)

/-- Batched RNN forward pass (maps `rnnSequenceSpec` over the batch dimension). -/
def rnnBatchedSpec {batchSize seqLen inputSize hiddenSize : Nat}
  (rnn : RNNSpec α inputSize hiddenSize)
  (inputs : BatchedTensor α batchSize (.dim seqLen (.dim inputSize .scalar)))
  (initial_hidden : BatchedTensor α batchSize (.dim hiddenSize .scalar)) :
  BatchedTensor α batchSize (.dim seqLen (.dim hiddenSize .scalar)) :=
  match inputs, initial_hidden with
  | Tensor.dim batch_inputs, Tensor.dim batch_hidden =>
    Tensor.dim (fun b =>
      rnnSequenceSpec rnn (batch_inputs b) (batch_hidden b))

/--
Gradient w.r.t. weights from a full unroll, given per-step preactivation gradients.

This is a lightweight helper for analyses that already have preactivation gradients. It assumes:
- the initial hidden state is `0`, and
- `grad_outputs[t]` is already `dL/dz_t` (preactivation gradient).

For end-to-end BPTT from `dL/dh_t`, prefer `rnnSequenceBackwardSpec`.
-/
def rnnWeightsDerivSpec {seqLen inputSize hiddenSize : Nat}
  (inputs : SequenceTensor α seqLen (.dim inputSize .scalar))
  (hiddens : SequenceTensor α seqLen (.dim hiddenSize .scalar))
  (grad_outputs : SequenceTensor α seqLen (.dim hiddenSize .scalar)) :
  WeightMatrix α hiddenSize (inputSize + hiddenSize) :=
  -- Assumes initial hidden state is 0 (matches the default module wrappers).
  -- Assumes `grad_outputs` is the preactivation gradient at each timestep.
  -- For full BPTT from post-activation gradients, use `rnnSequenceBackwardSpec`.
  let rec accumulate_grads (t : Nat) (acc : WeightMatrix α hiddenSize (inputSize + hiddenSize)) :
      WeightMatrix α hiddenSize (inputSize + hiddenSize) :=
    if h : t < seqLen then
      let input_t := getAtSpec inputs ⟨t, h⟩
      let hidden_prev :=
        if ht : t > 0 then
          have h_pred : t - 1 < t := by
            simpa [Nat.pred_eq_sub_one] using Nat.pred_lt (Nat.ne_of_gt ht)
          have h_t' : t - 1 < seqLen := lt_trans h_pred h
          getAtSpec hiddens ⟨t - 1, h_t'⟩
        else
          fill 0 (.dim hiddenSize .scalar)
      let grad_preact_t := getAtSpec grad_outputs ⟨t, h⟩
      let concat_t := concatVectorsSpec input_t hidden_prev
      let grad_w_t := outerProductSpec grad_preact_t concat_t
      accumulate_grads (t + 1) (addSpec acc grad_w_t)
    else
      acc
  accumulate_grads 0 (fill 0 (.dim hiddenSize (.dim (inputSize + hiddenSize) .scalar)))

/--
Gradient w.r.t. bias from per-step preactivation gradients.

This is `sum_t dL/dz_t` over the sequence dimension.
-/
def rnnBiasDerivSpec {seqLen hiddenSize : Nat}
  (grad_outputs : SequenceTensor α seqLen (.dim hiddenSize .scalar))
  (h : seqLen ≠ 0) :
  HiddenVector α hiddenSize :=
  -- Assumes `grad_outputs` is already the preactivation gradient.
  -- For full RNN backprop, prefer `rnnSequenceBackwardSpec`.
  letI : Shape.valid_axis_inst 0 (Shape.dim seqLen (Shape.dim hiddenSize Shape.scalar)) :=
    Shape.validAxisInstZeroAlt h
  reduceSumAuto 0 grad_outputs

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

- the gradient coming from the loss that touches `h_t` directly (`grad_hiddens[t]`),
- plus the gradient flowing "from the future" through the recurrence (`dHidden_next`).

Then we push `total_grad` through the single-step VJP (`rnn_cell_backward_spec`), producing:

- `dInput_t` and `dHidden_prev`,
- and parameter gradients `dW_t`, `db_t` which are accumulated across time.
-/

def rnnSequenceBackwardSpec {seqLen inputSize hiddenSize : Nat}
  (rnn : RNNSpec α inputSize hiddenSize)
  (inputs : SequenceTensor α seqLen (.dim inputSize .scalar))
  (initial_hidden : HiddenVector α hiddenSize)
  (hiddens : SequenceTensor α seqLen (.dim hiddenSize .scalar))
  (grad_hiddens : SequenceTensor α seqLen (.dim hiddenSize .scalar)) :
  ( WeightMatrix α hiddenSize (inputSize + hiddenSize) ×  -- dW
    HiddenVector α hiddenSize ×                            -- db
    SequenceTensor α seqLen (.dim inputSize .scalar) ×     -- dInputs
    HiddenVector α hiddenSize ) :=                          -- dInitialHidden

  let rec backward_step (t : Nat)
      (h_t : t ≤ seqLen)
      (dHidden_next : HiddenVector α hiddenSize)
      (acc_inputs : List (InputVector α inputSize))
      (acc_W : WeightMatrix α hiddenSize (inputSize + hiddenSize))
      (acc_b : HiddenVector α hiddenSize) :
      (List (InputVector α inputSize) × HiddenVector α hiddenSize ×
        WeightMatrix α hiddenSize (inputSize + hiddenSize) × HiddenVector α hiddenSize) :=
    if ht : t > 0 then
      let time_idx := t - 1
      have h_time : time_idx < seqLen := by
        have ht0 : 0 < t := ht
        have h1t : 1 ≤ t := Nat.succ_le_of_lt ht0
        have htime_lt_t : time_idx < t := by
          simpa [time_idx] using (Nat.sub_lt_self (by decide : 0 < 1) h1t)
        exact lt_of_lt_of_le htime_lt_t h_t
      let input_t := getAtSpec inputs ⟨time_idx, h_time⟩
      let hidden_t := getAtSpec hiddens ⟨time_idx, h_time⟩
      let prev_hidden :=
        if hprev : time_idx > 0 then
          have hp : time_idx - 1 < seqLen := by
            have hprev0 : 0 < time_idx := hprev
            have h1ti : 1 ≤ time_idx := Nat.succ_le_of_lt hprev0
            have hpred_lt : time_idx - 1 < time_idx := by
              simpa using (Nat.sub_lt_self (by decide : 0 < 1) h1ti)
            exact lt_trans hpred_lt h_time
          getAtSpec hiddens ⟨time_idx - 1, hp⟩
        else
          initial_hidden
      let grad_hidden_t := getAtSpec grad_hiddens ⟨time_idx, h_time⟩
      let total_grad := addSpec grad_hidden_t dHidden_next

      let (dInput_t, dHidden_prev, dW_t, db_t) :=
        rnnCellBackwardSpec rnn input_t prev_hidden hidden_t total_grad

      backward_step (t - 1) (by
        have : t - 1 ≤ t := Nat.sub_le _ _
        exact le_trans this h_t) dHidden_prev (dInput_t :: acc_inputs)
        (addSpec acc_W dW_t) (addSpec acc_b db_t)
    else
      (acc_inputs, dHidden_next, acc_W, acc_b)

  let (dInputs_list, dInitialHidden, dW, db) :=
    backward_step seqLen le_rfl (fill 0 (.dim hiddenSize .scalar)) []
      (fill 0 (.dim hiddenSize (.dim (inputSize + hiddenSize) .scalar)))
      (fill 0 (.dim hiddenSize .scalar))

  let dInputs :=
    match dInputs_list with
    | [] => fill 0 (.dim seqLen (.dim inputSize .scalar))
    | h :: _ => Tensor.dim (fun i => dInputs_list.getD i.val h)

  (dW, db, dInputs, dInitialHidden)

end Spec
