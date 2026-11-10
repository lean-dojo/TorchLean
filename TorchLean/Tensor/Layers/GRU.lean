/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Movement
public import TorchLean.Tensor.Layers.RNN
public import TorchLean.Tensor.Contract

/-!
# GRU (spec layer)

Gated recurrent unit cells and their unrolled sequences, in both conventions for the reset gate:
the reset-before cell of Cho et al. (`GRU`, `cell`, `GRU.sequence`) and PyTorch's
reset-after cell (`GRU.ResetAfter`, `ResetAfter.cell`, `ResetAfter.sequence`).
Shapes are part of the parameter types, so a dimension mismatch is a type error.

`GRU.cell` is Cho et al.'s eqs. (5) to (8) plus a bias per gate (`b_r`, `b_z`, `b_n`), which their
equations don't have. We transcribed it by hand. `GRU.ResetAfter.cell` is the cell of
`torch.nn.GRUCell` and `torch.nn.GRU` (PyTorch 2.8 docs), blocks in their `r`, `z`, `n` order.
Both cells keep the orientation of Cho et al.'s eq. (7) and torch's `h' = (1 - z) ⊙ n + z ⊙ h`,
so an update gate near one keeps the previous state. Chung et al.'s eq. (5) swaps `z` and `1 - z`.
`nn.Layer.gruCell_resetBefore_eq` and `nn.Layer.gruCell_resetAfter_eq`
(`TorchLean/NN/Layer/Recurrent/Correctness.lean`) prove that the module layer's cell returns these
values under the pure interpreter. We found no PyTorch test for either cell.

## References (math + PyTorch behavior)

- Cho et al.,
  "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation",
  EMNLP 2014, pp. 1724-1734, DOI 10.3115/v1/D14-1179, Sec. 2.3, eqs. (5) to (8):
  https://aclanthology.org/D14-1179/ (PDF: https://aclanthology.org/D14-1179.pdf)
- Chung et al., "Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling"
  (2014, NIPS 2014 Deep Learning and Representation Learning Workshop), Sec. 3.2, eq. (5):
  https://arxiv.org/abs/1412.3555
- PyTorch 2.8 `GRUCell` equations: https://docs.pytorch.org/docs/2.8/generated/torch.nn.GRUCell.html
- PyTorch 2.8 `GRU` equations: https://docs.pytorch.org/docs/2.8/generated/torch.nn.GRU.html

## Notes on parameterization

The GRU equations are often written with separate matrices $W_\bullet$ for the input and $U_\bullet$
for the hidden state. `GRU` uses one `Linear` per gate applied to a concatenated vector
$[x_t;h_{t-1}]$ (or $[x_t;r_t\odot h_{t-1}]$ for the candidate), whose weight holds $W_\bullet$ and
$U_\bullet$ side by side, so the cell reuses `Linear.forward` from the spec layer.

`GRU.ResetAfter` keeps PyTorch's two bias vectors, and PyTorch parameters load into it
unchanged. The two candidate equations are different functions for general recurrent matrices, so
no change of tensor layout converts one into the other.

Both conventions evaluate the complementary update gate $1 - z_t$ as `sigmoid (-updatePre)`, so the
candidate term and its gradient can be nonzero after $z_t$ has rounded to one.
-/

@[expose] public section


open TorchLean.Shape (size)

namespace TorchLean

open TorchLean.Tensor

variable {α : Type} [Storage α] [Context α]

/--
Where the reset gate acts in a GRU candidate. `resetBefore` multiplies the hidden vector by the
reset gate before the recurrent matrix, as in Cho et al., and `resetAfter` multiplies the recurrent
affine output, as PyTorch does.

The two differ for a non-diagonal matrix and for a nonzero recurrent candidate bias, so the choice
belongs to the model configuration.
-/
inductive GRU.Convention where
  /-- The reset gate multiplies the hidden vector before the recurrent matrix (Cho et al.). -/
  | resetBefore
  /-- The reset gate multiplies the recurrent affine output (PyTorch). -/
  | resetAfter
deriving Repr, DecidableEq

/--
Reset-after GRU parameters, in PyTorch's packed reset/update/candidate row order.

Rows `0 .. hiddenSize` belong to reset, the next block to update, and the last block to the
candidate. Input and recurrent weights use `[output, input]` layout. The two bias vectors are
separate parameters: adding them would lose the candidate's reset-gated recurrent bias and would
change how an optimizer updates even the reset and update gates.

The four tensors describe one cell and are copied from PyTorch without transposing, merging biases
or reordering gates. Picking the layer and direction out of a checkpoint is up to the caller.
-/
structure GRU.ResetAfter (α : Type) [Storage α]
    (inputSize hiddenSize : Nat) where
  /-- PyTorch `weight_ih`, with reset, update, and candidate rows. -/
  inputWeight : Tensor α [3 * hiddenSize, inputSize]
  /-- PyTorch `weight_hh`, in the same gate order. -/
  hiddenWeight : Tensor α [3 * hiddenSize, hiddenSize]
  /-- PyTorch `bias_ih`, whose candidate block is added outside the reset gate. -/
  inputBias : Tensor α [3 * hiddenSize]
  /-- PyTorch `bias_hh`, whose candidate block is multiplied by the reset gate. -/
  hiddenBias : Tensor α [3 * hiddenSize]

/--
One reset-after GRU step from the previous hidden state `prevHidden`, with the PyTorch candidate
$n_t=\tanh(W_{in}x_t+b_{in}+r_t\odot(W_{hn}h_{t-1}+b_{hn}))$.

We first compute both packed affine maps and then split their gate blocks, so the recurrent bias
$b_{hn}$ sits inside the product `reset * gate hiddenGates 2`. That order is what makes copied
PyTorch parameters describe the same recurrence.
-/
def GRU.ResetAfter.cell {inputSize hiddenSize : Nat}
    (gru : ResetAfter α inputSize hiddenSize)
    (input : Tensor α [inputSize]) (prevHidden : Tensor α [hiddenSize]) :
    Tensor α [hiddenSize] :=
  let inputGates := Tensor.contract gru.inputWeight input _ _ [3 *
      hiddenSize] .matVec + gru.inputBias
  let hiddenGates := Tensor.contract gru.hiddenWeight prevHidden _ _ [3 *
      hiddenSize] .matVec + gru.hiddenBias
  let gate := fun (values : Tensor α [3 * hiddenSize]) (index : Fin 3) =>
    let start := index.val * hiddenSize
    have hRange : start + hiddenSize ≤ 3 * hiddenSize := by
      simpa [start, Nat.add_mul] using
        Nat.mul_le_mul_right hiddenSize (Nat.succ_le_of_lt index.isLt)
    Tensor.reshape
      (Tensor.slice values [⟨0, by simp⟩] [start] [start + hiddenSize] [1]
        (Shape.sliceValid_singleton _ ⟨0, by simp⟩ start hiddenSize hRange))
      (.dim hiddenSize _) (congrArg Shape.size (Shape.slice_singleton _ _ _ _))
  let reset := Tensor.sigmoid (gate inputGates 0 + gate hiddenGates 0)
  let updatePre := gate inputGates 1 + gate hiddenGates 1
  let update := Tensor.sigmoid updatePre
  let complement := Tensor.sigmoid (Tensor.map (fun x => -x) updatePre)
  let candidate := Tensor.tanh (gate inputGates 2 + (reset * gate hiddenGates 2))
  (complement * candidate) +
    (update * prevHidden)

/-- Unroll the reset-after cell from the supplied initial state, returning every hidden state and
the final one (`initialHidden` for an empty sequence). -/
def GRU.ResetAfter.sequence {seqLen inputSize hiddenSize : Nat}
    (gru : ResetAfter α inputSize hiddenSize)
    (inputs : Tensor α [seqLen, inputSize]) (initialHidden : Tensor α [hiddenSize]) :
    Tensor α [seqLen, hiddenSize] × Tensor α [hiddenSize] :=
  let (finalHidden, outputs) := Sequence.mapAccum seqLen initialHidden fun i previous =>
    let hidden := cell gru (Tensor.unstack inputs i) previous
    (hidden, hidden)
  (Tensor.dim outputs.getScalar, finalHidden)

/--
Parameters for a single reset-before GRU cell in concatenated form: one `Linear` per gate,
applied to $[x_t;h_{t-1}]$ (length `inputSize + hiddenSize`) for the reset and update gates and to
$[x_t;r_t\odot h_{t-1}]$ for the candidate. Each gate weight is
`[hiddenSize, inputSize + hiddenSize]` and each gate bias `[hiddenSize]`.
-/
structure GRU (α : Type) [Storage α]
    (inputSize hiddenSize : Nat) where
  /-- Reset gate, $r_t=\operatorname{sigmoid}(W_r[x_t;h_{t-1}]+b_r)$. -/
  reset : Linear α (inputSize + hiddenSize) hiddenSize
  /-- Update gate, $z_t=\operatorname{sigmoid}(W_z[x_t;h_{t-1}]+b_z)$. -/
  update : Linear α (inputSize + hiddenSize) hiddenSize
  /-- Candidate state, $n_t=\tanh(W_n[x_t;r_t\odot h_{t-1}]+b_n)$. -/
  candidate : Linear α (inputSize + hiddenSize) hiddenSize

/--
One reset-before GRU step: the next hidden state $h_t$ from the input $x_t$ and the previous state
$h_{t-1}$, with the reset gate applied to $h_{t-1}$ before the candidate's linear map.

For PyTorch parameters use `GRU.ResetAfter.cell`.
-/
def GRU.cell {inputSize hiddenSize : Nat}
  (gru : GRU α inputSize hiddenSize)
  (input : Tensor α [inputSize])
  (prevHidden : Tensor α [hiddenSize]) :
  Tensor α [hiddenSize] :=
  -- The textbook GRU layout:
  --
  --   `r_t = sigmoid(W_r [x_t; h_{t-1}] + b_r)`
  --   `z_t = sigmoid(W_z [x_t; h_{t-1}] + b_z)`
  --   `n_t = tanh   (W_n [x_t; r_t ⊙ h_{t-1}] + b_n)`
  --   `h_t = (1 - z_t) ⊙ n_t + z_t ⊙ h_{t-1}`
  --
  -- PyTorch resets the hidden affine term after its matrix multiplication.
  let concat := Tensor.cat (s := [0]) ⟨0, by simp⟩ ![inputSize, hiddenSize]
      (Fin.cons input (Fin.cons prevHidden finZeroElim)) [inputSize + hiddenSize]

  let resetGate := Tensor.sigmoid (Linear.forward gru.reset concat)

  let updatePre := Linear.forward gru.update concat
  let updateGate := Tensor.sigmoid updatePre

  -- The candidate's input `[x_t; r_t ⊙ h_{t-1}]`.
  let resetConcat := Tensor.cat (s := [0]) ⟨0, by simp⟩ ![inputSize, hiddenSize]
      (Fin.cons input (Fin.cons (resetGate * prevHidden) finZeroElim)) [inputSize + hiddenSize]

  let newCandidate := Tensor.tanh (Linear.forward gru.candidate resetConcat)

  -- `1 - z_t` as `sigmoid (-updatePre)` can be nonzero after `z_t` has rounded to one.
  let oneMinusUpdate := Tensor.sigmoid (Tensor.map (fun x => -x) updatePre)
  (oneMinusUpdate * newCandidate) + (updateGate * prevHidden)

/--
Unroll a reset-before GRU over `seqLen` timesteps (time-major) from `initialHidden`.

The result is the sequence of hidden states $[h_0,\ldots,h_{\mathtt{seqLen}-1}]$ and the final
hidden state, which is `initialHidden` for an empty sequence.
-/
def GRU.sequence {seqLen inputSize hiddenSize : Nat}
  (gru : GRU α inputSize hiddenSize)
  (inputs : Tensor α [seqLen, inputSize])
  (initialHidden : Tensor α [hiddenSize]) :
  Tensor α [seqLen, hiddenSize] × Tensor α [hiddenSize] :=
  let (finalHidden, outputs) := Sequence.mapAccum seqLen initialHidden fun i previous =>
    let hidden := cell gru (Tensor.unstack inputs i) previous
    (hidden, hidden)
  (Tensor.dim outputs.getScalar, finalHidden)

end TorchLean
