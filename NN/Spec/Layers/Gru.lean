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
- explicit about which candidate equation is used.

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

The GRU equations are often written with separate matrices $W_\bullet$ for the input and
$U_\bullet$ for the hidden state. The legacy spec uses a single matrix per gate applied to a
concatenated vector $[x_t;h_{t-1}]$ (or $[x_t;r_t\odot h_{t-1}]$ for the candidate). This is the
same idea, just packaged in a way that reuses the tensor building blocks already present in the
spec layer.

The legacy `GRUSpec` applies the reset before the hidden-state linear map, as in Cho et al.
`GRUResetAfterSpec` applies it to the recurrent affine output and retains both bias vectors.
Use that second specification for PyTorch parameters; the two candidate equations are different
functions for general recurrent matrices, so changing tensor layout cannot convert between them.
-/

@[expose] public section


open TorchLean

namespace Spec

open TorchLean TorchLean.Tensor
open Activation

variable {α : Type} [TorchLean.Storage α] [Context α]

/--
Where the reset gate acts in a GRU candidate.

The original cell resets the hidden vector before multiplying by its recurrent matrix. PyTorch
resets the recurrent affine output instead. The distinction matters for a non-diagonal matrix
and for a nonzero recurrent candidate bias, so it belongs to the model configuration.
-/
inductive GRUConvention where
  | resetBefore
  | resetAfter
deriving Repr, DecidableEq

/--
Reset-after GRU parameters, in PyTorch's packed reset/update/candidate row order.

Rows `0 .. hiddenSize` belong to reset, the next block to update, and the last block to the
candidate. Input and recurrent weights use `[output, input]` layout. Both bias vectors remain
independent parameters: adding them would lose the candidate's reset-gated recurrent bias and
would change how an optimizer updates even the reset and update gates.
-/
structure GRUResetAfterSpec (α : Type) [TorchLean.Storage α]
    (inputSize hiddenSize : Nat) where
  /-- PyTorch `weight_ih`, with reset, update, and candidate rows. -/
  inputWeight : Tensor α [3 * hiddenSize, inputSize]
  /-- PyTorch `weight_hh`, in the same gate order. -/
  hiddenWeight : Tensor α [3 * hiddenSize, hiddenSize]
  /-- PyTorch `bias_ih`; the candidate part is added outside the reset gate. -/
  inputBias : Tensor α [3 * hiddenSize]
  /-- PyTorch `bias_hh`; the candidate part is multiplied by the reset gate. -/
  hiddenBias : Tensor α [3 * hiddenSize]

/--
Import a single PyTorch GRU cell's tensors without transposing, merging biases, or changing gates.

The shape indices check the packed row count. A checkpoint's layer and direction selection is
the caller's responsibility; these four tensors describe one cell.
-/
def GRUResetAfterSpec.ofPyTorch {inputSize hiddenSize : Nat}
    (weightIH : Tensor α [3 * hiddenSize, inputSize])
    (weightHH : Tensor α [3 * hiddenSize, hiddenSize])
    (biasIH biasHH : Tensor α [3 * hiddenSize]) :
    GRUResetAfterSpec α inputSize hiddenSize :=
  ⟨weightIH, weightHH, biasIH, biasHH⟩

/--
One reset-after step, with an explicit previous hidden state.

We first compute both packed affine maps, then split their gate blocks. In the candidate,
`reset * hiddenCandidate` includes the recurrent bias because it is already part of that affine
map. Keeping this order is what makes copied PyTorch parameters describe the same recurrence.
-/
def gruResetAfterCellSpec {inputSize hiddenSize : Nat}
    (gru : GRUResetAfterSpec α inputSize hiddenSize)
    (input : Tensor α [inputSize]) (prevHidden : Tensor α [hiddenSize]) :
    Tensor α [hiddenSize] :=
  let inputGates := addSpec (matVecMulSpec gru.inputWeight input) gru.inputBias
  let hiddenGates := addSpec (matVecMulSpec gru.hiddenWeight prevHidden) gru.hiddenBias
  let gate := fun (values : Tensor α [3 * hiddenSize]) (index : Fin 3) =>
    sliceRangeSpec values (index.val * hiddenSize) hiddenSize (by
      simpa [Nat.add_mul] using
        Nat.mul_le_mul_right hiddenSize (Nat.succ_le_of_lt index.isLt))
  let reset := sigmoidSpec (addSpec (gate inputGates 0) (gate hiddenGates 0))
  let update := sigmoidSpec (addSpec (gate inputGates 1) (gate hiddenGates 1))
  let candidate := tanhSpec (addSpec (gate inputGates 2) (mulSpec reset (gate hiddenGates 2)))
  addSpec
    (mulSpec (subSpec (Tensor.full [hiddenSize] 1) update) candidate)
    (mulSpec update prevHidden)

/-- Unroll the reset-after cell from the supplied initial state, returning every hidden state. -/
def gruResetAfterSequenceSpec {seqLen inputSize hiddenSize : Nat}
    (gru : GRUResetAfterSpec α inputSize hiddenSize)
    (inputs : Tensor α [seqLen, inputSize]) (initialHidden : Tensor α [hiddenSize]) :
    Tensor α [seqLen, hiddenSize] :=
  let (_, outputs) := Sequence.mapAccum seqLen initialHidden fun i previous =>
    let hidden := gruResetAfterCellSpec gru (get inputs i) previous
    (hidden, hidden)
  Tensor.dim outputs.getScalar

-- GRU cell specification: separate weights for reset, update, and new gates
-- Each gate has weights [hidden_size, input_size + hidden_size] and bias [hidden_size]
/--
Parameters for a single GRU cell.

This is the original concatenated GRU parameterization, using $[x_t;h_{t-1}]$ (shape
`inputSize + hiddenSize`) for the reset/update gates and
$[x_t;r_t\odot h_{t-1}]$ for the candidate gate.

Shapes:

- each gate weight is `[hiddenSize, inputSize + hiddenSize]`,
- each gate bias is `[hiddenSize]`.
-/
structure GRUSpec (α : Type) [TorchLean.Storage α]
    (inputSize hiddenSize : Nat) where
  /-- Reset-gate weights for
  $r_t=\operatorname{sigmoid}(W_r[x_t;h_{t-1}]+b_r)$. -/
  resetWeight : Tensor α [hiddenSize, inputSize + hiddenSize]
  /-- Reset-gate bias. -/
  resetBias : Tensor α [hiddenSize]
  /-- Update-gate weights for
  $z_t=\operatorname{sigmoid}(W_z[x_t;h_{t-1}]+b_z)$. -/
  updateWeight : Tensor α [hiddenSize, inputSize + hiddenSize]
  /-- Update-gate bias. -/
  updateBias : Tensor α [hiddenSize]
  /-- Candidate-state weights for
  $n_t=\tanh(W_n[x_t;r_t\odot h_{t-1}]+b_n)$. -/
  candidateWeight : Tensor α [hiddenSize, inputSize + hiddenSize]
  /-- Candidate-state bias. -/
  candidateBias : Tensor α [hiddenSize]

/--
Forward pass for a single GRU cell.

Given input $x_t$ and previous hidden state $h_{t-1}$, compute the next hidden state $h_t$ using
the standard GRU equations.

This is not PyTorch's reset-after candidate parameterization; see the module note above.
-/
def gruCellSpec {inputSize hiddenSize : Nat}
  (gru : GRUSpec α inputSize hiddenSize)
  (input : Tensor α [inputSize])
  (prevHidden : Tensor α [hiddenSize]) :
  Tensor α [hiddenSize] :=
  -- We follow the textbook GRU layout:
  --
  --   r_t = sigmoid(W_r [x_t; h_{t-1}] + b_r)
  --   z_t = sigmoid(W_z [x_t; h_{t-1}] + b_z)
  --   n_t = tanh   (W_n [x_t; r_t ⊙ h_{t-1}] + b_n)
  --   h_t = (1 - z_t) ⊙ n_t + z_t ⊙ h_{t-1}
  --
  -- PyTorch instead resets the hidden affine contribution after its matrix multiplication.
  let concat := concatAxisSpec .scalar input prevHidden

  -- Reset gate.
  let resetGate := sigmoidSpec (addSpec (matVecMulSpec gru.resetWeight concat)
    gru.resetBias)

  -- Update gate.
  let updateGate := sigmoidSpec (addSpec (matVecMulSpec gru.updateWeight concat)
    gru.updateBias)

  -- The reset gate decides what portion of the previous state is used in the candidate update.
  let reset_hidden := mulSpec resetGate prevHidden

  -- Candidate uses `[x_t; r_t ⊙ h_{t-1}]`.
  let resetConcat := concatAxisSpec .scalar input reset_hidden

  -- Candidate (sometimes called `n_t` or `h~_t` in the literature).
  let newCandidate := tanhSpec (addSpec (matVecMulSpec gru.candidateWeight resetConcat)
    gru.candidateBias)

  -- Final hidden state:
  --   h_t = (1 - z_t) ⊙ n_t + z_t ⊙ h_{t-1}.
  let oneMinusUpdate := subSpec (Tensor.full (.dim hiddenSize .scalar) 1) updateGate
  let newContribution := mulSpec oneMinusUpdate newCandidate
  let hiddenContribution := mulSpec updateGate prevHidden
  addSpec newContribution hiddenContribution

-- GRU sequence forward pass: processes a sequence of inputs
/--
Unroll a GRU over `seqLen` timesteps (time-major).

This returns the sequence of hidden states $[h_0,\ldots,h_{\mathtt{seqLen}-1}]$. It is a pure
spec-level
definition of semantics; an efficient runtime is free to implement the same behavior with loops and
caching.

The input is time-major and the result contains every hidden state. The candidate semantics remain
the Cho-style equations of `gruCellSpec`.
-/
def gruSequenceSpec {seqLen inputSize hiddenSize : Nat}
  (gru : GRUSpec α inputSize hiddenSize)
  (inputs : Tensor α [seqLen, inputSize])
  (initialHidden : Tensor α [hiddenSize]) :
  Tensor α [seqLen, hiddenSize] :=
  let (_, outputs) := Sequence.mapAccum seqLen initialHidden fun i previous =>
    let hidden := gruCellSpec gru (get inputs i) previous
    (hidden, hidden)
  Tensor.dim outputs.getScalar

-- GRU cell forward pass that returns all intermediate values for BPTT
/--
GRU cell forward pass that also returns cached intermediates for BPTT.

This computes the same next hidden state as `gruCellSpec`, but additionally returns:

- `resetGate` ($r_t$),
- `updateGate` ($z_t$),
- `newCandidate` ($n_t$), and
- `reset_hidden` ($r_t\odot h_{t-1}$).

These are exactly the quantities commonly saved by a reverse-mode implementation (PyTorch-style
autograd) to compute gradients efficiently in the backward pass.
-/
def gruCellSpecWithIntermediates {inputSize hiddenSize : Nat}
  (gru : GRUSpec α inputSize hiddenSize)
  (input : Tensor α [inputSize])
  (prevHidden : Tensor α [hiddenSize]) :
  (Tensor α [hiddenSize] ×  -- newHidden
   Tensor α [hiddenSize] ×  -- resetGate
   Tensor α [hiddenSize] ×  -- updateGate
   Tensor α [hiddenSize] ×  -- newCandidate
   Tensor α [hiddenSize]) := -- reset_hidden
  -- Same computation as `gruCellSpec`, but we also return the gate activations and the candidate.
  -- Those values are what a "tape" would store for a standard BPTT implementation.
  let concat := concatAxisSpec .scalar input prevHidden

  -- Reset gate: r_t = σ(W_r @ [x_t; h_{t-1}] + b_r)
  let resetGate := sigmoidSpec (addSpec (matVecMulSpec gru.resetWeight concat)
    gru.resetBias)

  -- Update gate: z_t = σ(W_z @ [x_t; h_{t-1}] + b_z)
  let updateGate := sigmoidSpec (addSpec (matVecMulSpec gru.updateWeight concat)
    gru.updateBias)

  -- Reset hidden state: h_reset = r_t ⊙ h_{t-1}
  let reset_hidden := mulSpec resetGate prevHidden

  -- Concatenate input with reset hidden state
  let resetConcat := concatAxisSpec .scalar input reset_hidden

  -- New hidden state candidate: ĥ_t = tanh(W_h @ [x_t; r_t ⊙ h_{t-1}] + b_h)
  let newCandidate := tanhSpec (addSpec (matVecMulSpec gru.candidateWeight resetConcat)
    gru.candidateBias)

  -- Final hidden state follows the same convention as `gruCellSpec`.
  let oneMinusUpdate := subSpec (Tensor.full (.dim hiddenSize .scalar) 1) updateGate
  let newContribution := mulSpec oneMinusUpdate newCandidate
  let hiddenContribution := mulSpec updateGate prevHidden
  let newHidden := addSpec newContribution hiddenContribution

  (newHidden, resetGate, updateGate, newCandidate, reset_hidden)

/--
Run a GRU forward pass while collecting the per-timestep intermediates needed for BPTT.

This is the "spec-level" analogue of what frameworks do internally:

- the forward pass produces $h_t$,
- and it also saves gate activations $r_t$, $z_t$, and candidate $n_t$ for the backward pass.

The returned tensors are all time-major (`seqLen` first) to match the rest of the spec layer.
-/
def gruExtractIntermediateValues {seqLen inputSize hiddenSize : Nat}
  (gru : GRUSpec α inputSize hiddenSize)
  (inputs : Tensor α [seqLen, inputSize])
  (initialHidden : Tensor α [hiddenSize]) :
  (Tensor α [seqLen, hiddenSize] ×  -- hiddenStates
   Tensor α [seqLen, hiddenSize] ×  -- resetGates
   Tensor α [seqLen, hiddenSize] ×  -- updateGates
   Tensor α [seqLen, hiddenSize] ×  -- newCandidates
   Tensor α [seqLen, hiddenSize]) := -- resetHiddens
  let (_, saved) := Sequence.mapAccum seqLen initialHidden fun i previous =>
    let (hidden, reset, update, candidate, resetHidden) :=
      gruCellSpecWithIntermediates gru (get inputs i) previous
    (hidden, (hidden, reset, update, candidate, resetHidden))

  let hiddenStates := Tensor.dim fun i =>
    let (hidden, _, _, _, _) := saved.getScalar i
    hidden
  let resetGates := Tensor.dim fun i =>
    let (_, reset, _, _, _) := saved.getScalar i
    reset
  let updateGates := Tensor.dim fun i =>
    let (_, _, update, _, _) := saved.getScalar i
    update
  let newCandidates := Tensor.dim fun i =>
    let (_, _, _, candidate, _) := saved.getScalar i
    candidate
  let resetHiddens := Tensor.dim fun i =>
    let (_, _, _, _, resetHidden) := saved.getScalar i
    resetHidden

  (hiddenStates, resetGates, updateGates, newCandidates, resetHiddens)

-- Batched GRU sequence forward pass
/--
Batched GRU forward pass (map `gruSequenceSpec` over the batch dimension).

This is a simple spec-level definition for semantics, not an optimized kernel. It maps the same
Cho-style cell over a batch; it is not a `torch.nn.GRU` checkpoint format.
-/
def gruBatchedSpec {batchSize seqLen inputSize hiddenSize : Nat}
  (gru : GRUSpec α inputSize hiddenSize)
  (inputs : Tensor α [batchSize, seqLen, inputSize])
  (initialHidden : Tensor α [batchSize, hiddenSize]) :
  Tensor α [batchSize, seqLen, hiddenSize] :=
  -- This is a simple "map over the batch dimension".
  -- It matches the semantics of a batched GRU, but it is not an optimized runtime kernel.
  Tensor.dim (fun batch =>
    gruSequenceSpec gru (Tensor.unstack inputs batch)
      (Tensor.unstack initialHidden batch))

-- Gradient computations for GRU

-- Gradient w.r.t. reset gate weights
/--
Reference gradient for reset-gate weights via the generic RNN weight-gradient helper.

This uses `rnnWeightsDerivSpec` on the concatenated inputs/hidden states. It is a convenient
building block, but the more explicit BPTT helpers below show the time-unrolled
accumulation form.
-/
def gruResetWeightsDerivSpec {seqLen inputSize hiddenSize : Nat}
  (inputs : Tensor α [seqLen, inputSize])
  (hiddens : Tensor α [seqLen, hiddenSize])
  (gradReset : Tensor α [seqLen, hiddenSize]) :
  Tensor α [hiddenSize, inputSize + hiddenSize] :=
  rnnWeightsDerivSpec inputs hiddens gradReset

-- Gradient w.r.t. update gate weights
/-- Reference gradient for update-gate weights (via `rnnWeightsDerivSpec`). -/
def gruUpdateWeightsDerivSpec {seqLen inputSize hiddenSize : Nat}
  (inputs : Tensor α [seqLen, inputSize])
  (hiddens : Tensor α [seqLen, hiddenSize])
  (gradUpdate : Tensor α [seqLen, hiddenSize]) :
  Tensor α [hiddenSize, inputSize + hiddenSize] :=
  rnnWeightsDerivSpec inputs hiddens gradUpdate

-- Gradient w.r.t. new gate weights
/--
Reference gradient for candidate ("new") gate weights (via `rnnWeightsDerivSpec`).

The second sequence argument satisfies
$\mathtt{reset\_hiddens}_t=r_t\odot h_{t-1}$.
-/
def gruNewWeightsDerivSpec {seqLen inputSize hiddenSize : Nat}
  (inputs : Tensor α [seqLen, inputSize])
  (resetHiddens : Tensor α [seqLen, hiddenSize]) -- r_t ⊙ h_{t-1}
  (gradNew : Tensor α [seqLen, hiddenSize]) :
  Tensor α [hiddenSize, inputSize + hiddenSize] :=
  rnnWeightsDerivSpec inputs resetHiddens gradNew

-- Gradient w.r.t. biases (sum over sequence length)
/--
Bias gradient by summing per-timestep gradients over the time axis.

This is the spec-level analogue of the common "sum across batch/time" reduction used for bias
gradients. The `seqLen ≠ 0` hypothesis is exactly what makes axis `0` a valid reduction axis.
-/
def gruBiasDerivSpec {seqLen hiddenSize : Nat}
  (gradOutputs : Tensor α [seqLen, hiddenSize])
  (h : seqLen ≠ 0) :
  Tensor α [hiddenSize] :=
  reduceSum 0 gradOutputs (Shape.hasNonemptyAxisZeroOfNe h).proof

-- Gradient w.r.t. reset gate weights with proper BPTT
/--
Reset-gate weight gradient by explicit time-unrolled accumulation (BPTT-style).

This computes
$$
\sum_t \frac{\partial L}{\partial r_t}\otimes[x_t;h_{t-1}],
$$
where $\otimes$ is an outer product.
-/
def gruResetWeightsDerivBpttSpec {seqLen inputSize hiddenSize : Nat}
  (inputs : Tensor α [seqLen, inputSize])
  (hiddens : Tensor α [seqLen, hiddenSize])
  (_reset_gates : Tensor α [seqLen, hiddenSize])
  (gradResetGates : Tensor α [seqLen, hiddenSize]) :
  Tensor α [hiddenSize, inputSize + hiddenSize] :=
  -- Accumulate gradients over time steps
  let rec accumulate_grads (t : Nat) (acc : Tensor α [hiddenSize, inputSize + hiddenSize]) :
    Tensor α [hiddenSize, inputSize + hiddenSize] :=
    if h : t < seqLen then
      let inputT := get inputs ⟨t, h⟩
      let hiddenPrev :=
        if ht : t > 0 then
          have ht0 : t ≠ 0 := Nat.ne_of_gt ht
          have htPred : t - 1 < t := by
            simpa [Nat.pred_eq_sub_one] using Nat.pred_lt ht0
          have htPrev : t - 1 < seqLen := lt_trans htPred h
          get hiddens ⟨t - 1, htPrev⟩
        else
          Tensor.full (.dim hiddenSize .scalar) 0
      let concatT := concatAxisSpec .scalar inputT hiddenPrev
      let gradResetT := get gradResetGates ⟨t, h⟩
      let gradWT := outerProductSpec gradResetT concatT
      accumulate_grads (t + 1) (addSpec acc gradWT)
    else acc
  accumulate_grads 0 (Tensor.full (.dim hiddenSize (.dim (inputSize + hiddenSize) .scalar)) 0)

-- Gradient w.r.t. update gate weights with proper BPTT
/--
Update-gate weight gradient by explicit time-unrolled accumulation (BPTT-style).

This computes
$$
\sum_t \frac{\partial L}{\partial z_t}\otimes[x_t;h_{t-1}].
$$
-/
def gruUpdateWeightsDerivBpttSpec {seqLen inputSize hiddenSize : Nat}
  (inputs : Tensor α [seqLen, inputSize])
  (hiddens : Tensor α [seqLen, hiddenSize])
  (gradUpdateGates : Tensor α [seqLen, hiddenSize]) :
  Tensor α [hiddenSize, inputSize + hiddenSize] :=
  -- Accumulate gradients over time steps
  let rec accumulate_grads (t : Nat) (acc : Tensor α [hiddenSize, inputSize + hiddenSize]) :
    Tensor α [hiddenSize, inputSize + hiddenSize] :=
    if h : t < seqLen then
      let inputT := get inputs ⟨t, h⟩
      let hiddenPrev :=
        if ht : t > 0 then
          have ht0 : t ≠ 0 := Nat.ne_of_gt ht
          have htPred : t - 1 < t := by
            simpa [Nat.pred_eq_sub_one] using Nat.pred_lt ht0
          have htPrev : t - 1 < seqLen := lt_trans htPred h
          get hiddens ⟨t - 1, htPrev⟩
        else
          Tensor.full (.dim hiddenSize .scalar) 0
      let concatT := concatAxisSpec .scalar inputT hiddenPrev
      let gradUpdateT := get gradUpdateGates ⟨t, h⟩
      let gradWT := outerProductSpec gradUpdateT concatT
      accumulate_grads (t + 1) (addSpec acc gradWT)
    else acc
  accumulate_grads 0 (Tensor.full (.dim hiddenSize (.dim (inputSize + hiddenSize) .scalar)) 0)

-- Gradient w.r.t. new gate weights with proper BPTT
/--
Candidate-gate weight gradient by explicit time-unrolled accumulation (BPTT-style).

This computes
$$
\sum_t \frac{\partial L}{\partial n_t}\otimes[x_t;r_t\odot h_{t-1}].
$$
-/
def gruNewWeightsDerivBpttSpec {seqLen inputSize hiddenSize : Nat}
  (inputs : Tensor α [seqLen, inputSize])
  (resetHiddens : Tensor α [seqLen, hiddenSize]) -- r_t ⊙ h_{t-1}
  (gradNewCandidates : Tensor α [seqLen, hiddenSize]) :
  Tensor α [hiddenSize, inputSize + hiddenSize] :=
  -- Accumulate gradients over time steps
  let rec accumulate_grads (t : Nat) (acc : Tensor α [hiddenSize, inputSize + hiddenSize]) :
    Tensor α [hiddenSize, inputSize + hiddenSize] :=
    if h : t < seqLen then
      let inputT := get inputs ⟨t, h⟩
      let resetHiddenT := get resetHiddens ⟨t, h⟩
      let concatT := concatAxisSpec .scalar inputT resetHiddenT
      let gradNewT := get gradNewCandidates ⟨t, h⟩
      let gradWT := outerProductSpec gradNewT concatT
      accumulate_grads (t + 1) (addSpec acc gradWT)
    else acc
  accumulate_grads 0 (Tensor.full (.dim hiddenSize (.dim (inputSize + hiddenSize) .scalar)) 0)

/--
Backward (VJP) for a single GRU cell.

Inputs:

- the cell parameters `gru`,
- the current input $x_t$,
- the previous hidden state $h_{t-1}$,
- an upstream gradient $\partial L/\partial h_t$,
- and the forward intermediates $r_t$, $z_t$, and $n_t$ that a typical BPTT implementation would
  cache.

Outputs:

- gradients w.r.t. the input and previous hidden state,
- plus gradients for each parameter tensor (weights and biases).

This is written to match the forward equations in `gruCellSpec`. It is not an optimized kernel;
it is a precise spec for what gradients *should* be.
-/
def gruCellBackwardFullSpec {inputSize hiddenSize : Nat}
  (gru : GRUSpec α inputSize hiddenSize)
  (input : Tensor α [inputSize])
  (prevHidden : Tensor α [hiddenSize])
  (gradOutput : Tensor α [hiddenSize])
  (resetGate : Tensor α [hiddenSize])
  (updateGate : Tensor α [hiddenSize])
  (newCandidate : Tensor α [hiddenSize]) :
  ( Tensor α [inputSize] ×                     -- dInput
    Tensor α [hiddenSize] ×                    -- dPrevHidden
    Tensor α [hiddenSize, inputSize + hiddenSize] ×  -- dResetW
    Tensor α [hiddenSize] ×                    -- dResetB
    Tensor α [hiddenSize, inputSize + hiddenSize] ×  -- dUpdateW
    Tensor α [hiddenSize] ×                    -- dUpdateB
    Tensor α [hiddenSize, inputSize + hiddenSize] ×  -- dNewW
    Tensor α [hiddenSize]                      -- dNewB
  ) :=
  let concat := concatAxisSpec .scalar input prevHidden

  -- Start from the output equation:
  --   h = (1 - z) ⊙ n + z ⊙ h_prev
  --
  -- This yields three immediate partials:
  --   d n      = d h ⊙ (1 - z)
  --   d z      = d h ⊙ (h_prev - n)
  --   d h_prev (direct) = d h ⊙ z
  let one_minus_z := subSpec (Tensor.full (.dim hiddenSize .scalar) 1) updateGate
  let dHtilde := mulSpec gradOutput one_minus_z
  let dZ := mulSpec gradOutput (subSpec prevHidden newCandidate)
  let dPrevDirect := mulSpec gradOutput updateGate

  -- tanh preactivation derivative using output newCandidate = tanh(pre_h)
  let dPreH := mulSpec dHtilde (subSpec (Tensor.full (.dim hiddenSize .scalar) 1) (mulSpec
    newCandidate newCandidate))

  -- h_reset = r ⊙ h_prev, resetConcat = [x; h_reset]
  let reset_hidden := mulSpec resetGate prevHidden
  let resetConcat := concatAxisSpec .scalar input reset_hidden

  -- New gate grads.
  let dNewW := outerProductSpec dPreH resetConcat
  let dNewB := dPreH
  let dResetConcat := vecMatMulSpec dPreH gru.candidateWeight
  let dXFromH := sliceRangeSpec dResetConcat 0 inputSize (by
    simp)
  let dHreset := sliceRangeSpec dResetConcat inputSize hiddenSize (by
    simp)

  -- Backprop through reset_hidden = r ⊙ h_prev.
  let dRFromReset := mulSpec dHreset prevHidden
  let dPrevFromReset := mulSpec dHreset resetGate

  -- Reset gate grads: r = sigmoid(pre_r)
  let dPreR := mulSpec dRFromReset (Activation.sigmoidOutputDerivSpec resetGate)
  let dResetW := outerProductSpec dPreR concat
  let dResetB := dPreR
  let dConcatFromR := vecMatMulSpec dPreR gru.resetWeight
  let dXFromR := sliceRangeSpec dConcatFromR 0 inputSize (by
    simp)
  let dPrevFromR := sliceRangeSpec dConcatFromR inputSize hiddenSize (by
    simp)

  -- Update gate grads: z = sigmoid(pre_z)
  let dPreZ := mulSpec dZ (Activation.sigmoidOutputDerivSpec updateGate)
  let dUpdateW := outerProductSpec dPreZ concat
  let dUpdateB := dPreZ
  let dConcatFromZ := vecMatMulSpec dPreZ gru.updateWeight
  let dXFromZ := sliceRangeSpec dConcatFromZ 0 inputSize (by
    simp)
  let dPrevFromZ := sliceRangeSpec dConcatFromZ inputSize hiddenSize (by
    simp)

  let dInput := addSpec (addSpec dXFromH dXFromR) dXFromZ
  let dPrevHidden := addSpec (addSpec (addSpec dPrevDirect dPrevFromReset) dPrevFromR)
    dPrevFromZ

  (dInput, dPrevHidden, dResetW, dResetB, dUpdateW, dUpdateB, dNewW, dNewB)

/--
Reverse-mode backprop through an unrolled GRU over `seqLen` steps (BPTT).

This function consumes the same intermediates produced by `gruExtractIntermediateValues`:
per-timestep gate activations and candidates. The backward pass walks time in reverse and
accumulates gradients for the Cho-style forward equation.
-/
def gruSequenceBackwardFullSpec {seqLen inputSize hiddenSize : Nat}
  (gru : GRUSpec α inputSize hiddenSize)
  (inputs : Tensor α [seqLen, inputSize])
  (hiddens : Tensor α [seqLen, hiddenSize])
  (gradOutputs : Tensor α [seqLen, hiddenSize])
  (resetGates : Tensor α [seqLen, hiddenSize])
  (updateGates : Tensor α [seqLen, hiddenSize])
  (newCandidates : Tensor α [seqLen, hiddenSize])
  (initialHidden : Tensor α [hiddenSize] := Tensor.full [hiddenSize] 0) :
  ( Tensor α [hiddenSize, inputSize + hiddenSize] ×  -- dResetW
    Tensor α [hiddenSize] ×                                  -- dResetB
    Tensor α [hiddenSize, inputSize + hiddenSize] ×  -- dUpdateW
    Tensor α [hiddenSize] ×                                  -- dUpdateB
    Tensor α [hiddenSize, inputSize + hiddenSize] ×  -- dNewW
    Tensor α [hiddenSize] ×                                  -- dNewB
    Tensor α [seqLen, inputSize] ×                     -- dInputs
    Tensor α [hiddenSize]                                    -- dInitialHidden
  ) :=

  let zeroHidden := Tensor.full (.dim hiddenSize .scalar) 0
  let zeroWeights := Tensor.full (.dim hiddenSize (.dim (inputSize + hiddenSize) .scalar)) 0
  let initial :=
    (zeroHidden, zeroWeights, zeroHidden, zeroWeights, zeroHidden, zeroWeights, zeroHidden)
  let (result, dInputs) := Sequence.mapAccumRight seqLen initial fun index state =>
    let (dHiddenNext, resetWeights, resetBias, updateWeights, updateBias, newWeights, newBias) :=
      state
    let input := get inputs index
    let previous :=
      if h : index.val > 0 then
        have hp : index.val - 1 < seqLen := by grind
        get hiddens ⟨index.val - 1, hp⟩
      else
        initialHidden
    let totalGradient := addSpec (get gradOutputs index) dHiddenNext
    let (dInput, dHidden, dResetWeights, dResetBias, dUpdateWeights, dUpdateBias,
        dNewWeights, dNewBias) :=
      gruCellBackwardFullSpec gru input previous totalGradient (get resetGates index)
        (get updateGates index) (get newCandidates index)
    ((dHidden, addSpec resetWeights dResetWeights, addSpec resetBias dResetBias,
      addSpec updateWeights dUpdateWeights, addSpec updateBias dUpdateBias,
      addSpec newWeights dNewWeights, addSpec newBias dNewBias), dInput)
  let (dInitialHidden, dResetWeights, dResetBias, dUpdateWeights, dUpdateBias, dNewWeights,
      dNewBias) := result
  (dResetWeights, dResetBias, dUpdateWeights, dUpdateBias, dNewWeights, dNewBias,
    Tensor.dim dInputs.getScalar, dInitialHidden)

/--
Return the input-sequence and initial-hidden gradients from `gruSequenceBackwardFullSpec`.

The full backward pass also returns parameter gradients. This projection records the common contract
used by callers that only propagate gradients to the preceding recurrent computation.
-/
def gruSequenceBackwardSpec {seqLen inputSize hiddenSize : Nat}
  (gru : GRUSpec α inputSize hiddenSize)
  (inputs : Tensor α [seqLen, inputSize])
  (hiddens : Tensor α [seqLen, hiddenSize])
  (gradOutputs : Tensor α [seqLen, hiddenSize])
  (resetGates : Tensor α [seqLen, hiddenSize])
  (updateGates : Tensor α [seqLen, hiddenSize])
  (newCandidates : Tensor α [seqLen, hiddenSize])
  (initialHidden : Tensor α [hiddenSize] := Tensor.full [hiddenSize] 0) :
  (Tensor α [seqLen, inputSize] × Tensor α [hiddenSize]) :=
  let (_, _, _, _, _, _, dInputs, dInitialHidden) :=
    gruSequenceBackwardFullSpec gru inputs hiddens gradOutputs resetGates updateGates
      newCandidates initialHidden
  (dInputs, dInitialHidden)

end Spec
