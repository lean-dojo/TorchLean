/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Layer
public import TorchLean.Tensor.Layers.GRU

/-!
# TorchLean NN: Linear and Recurrent Layers

The recurrent cells `rnnCell`, `lstmCell` and `gruCell` take their parameters and the previous state
as arguments. `Recurrent.scan` runs a cell over a sequence from an initial state and returns the
outputs and the final state. The layers `rnn`, `gru`, `gruResetAfter` and `lstm` run the same cells
from zero and return every hidden state. `TorchLean.NN.Layer.Recurrent.Correctness` proves that
under the pure interpreter the cells are the Spec cells and `Recurrent.scan` is the Spec sequence
recursion.
-/

@[expose] public section

namespace TorchLean
namespace nn.Layer

open TorchLean.Tensor
open Ops

/-! ## Convenience constructors (layers) -/

/--
Stack `count` consecutive one-row tensors starting at `start` along the leading axis.

Halves are joined pairwise, so each row is copied once per level of a balanced tree and the whole
stack costs `O(count * log count)` row copies. A full-output copy per row would cost
`O(count ^ 2)`.
-/
def stackRows {α : Type} [Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {rows : Nat} {tail : Shape}
    (row : Fin rows → Ref (m := m) (α := α) (Shape.dim 1 tail)) :
    (start count : Nat) → start + count ≤ rows → 0 < count →
      m (Ref (m := m) (α := α) (Shape.dim count tail))
  | start, 1, hRange, _ => pure (row ⟨start, by omega⟩)
  | start, count + 2, hRange, _ => do
      let half := (count + 2) / 2
      let left ← stackRows row start half (by omega) (by omega)
      let right ← stackRows row (start + half) (count + 2 - half) (by omega) (by omega)
      let joined ← Ops.apply (m := m) (α := α)
          (.cat (s := .dim 0 tail) ⟨0, by simp⟩ [half, count + 2 - half]) (left, right, ())
      have hCount : half + (count + 2 - half) = count + 2 := by omega
      pure (hCount ▸ joined)
  termination_by _ count => count

/-- Run `step` at each leading index from `index` to `rows - 1` in order, push each output onto
`acc` as a one-row tensor, and return all `rows` rows with the final state. -/
def collectRows {α : Type} [Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {State : Type} {rows : Nat} {tail : Shape}
    (step : Fin rows → State → m (State × Ref (m := m) (α := α) tail)) :
    (index : Nat) → index ≤ rows → State →
      (acc : Array (Ref (m := m) (α := α) (Shape.dim 1 tail))) → acc.size = index →
      m ({ result : Array (Ref (m := m) (α := α) (Shape.dim 1 tail)) // result.size = rows } ×
        State)
  | index, hIndex, state, acc, hAcc =>
      if hLt : index < rows then do
        let (next, value) ← step ⟨index, hLt⟩ state
        let row ← Ops.apply (m := m) (α := α)
            (.reshape (s := tail) (t := Shape.dim 1 tail) (by simp)) (value, ())
        collectRows step (index + 1) hLt next (acc.push row) (by rw [Array.size_push, hAcc])
      else
        pure (⟨acc, by omega⟩, state)
  termination_by index => rows - index

/--
Run `step` for each leading index in order from `initial`, and return the step outputs stacked
along a leading axis of length `rows` together with the final state. With no rows the outputs are
a zero tensor and the state is `initial`.
-/
def scanRows {α : Type} [Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {State : Type} {rows : Nat} {tail : Shape} (initial : State)
    (step : Fin rows → State → m (State × Ref (m := m) (α := α) tail)) :
    m (Ref (m := m) (α := α) (Shape.dim rows tail) × State) := do
  let (stepRows, final) ← collectRows (m := m) (α := α) step 0 (Nat.zero_le rows) initial #[] rfl
  if hRows : 0 < rows then
    let stacked ← stackRows (m := m) (α := α) (rows := rows)
      (fun index => stepRows.val[index.val]'(by rw [stepRows.property]; exact index.isLt))
      0 rows (by omega) hRows
    pure (stacked, final)
  else
    let empty ← Ops.const (m := m) (α := α)
      (Tensor.zeros (α := α) (Shape.dim rows tail))
    pure (empty, final)

/-- Validate one positive architectural dimension. -/
def requirePositive (kind field : String) (value : Nat) : Except String Unit := do
  if value = 0 then
    throw s!"{kind}: {field} must be positive"

/-- Require positive sequence length, input width and hidden width for a recurrent layer. -/
def validateRecurrent
    (kind : String) (sequenceLength inputWidth hiddenWidth : Nat) :
    Except String Unit := do
  requirePositive kind "sequence length" sequenceLength
  requirePositive kind "input width" inputWidth
  requirePositive kind "hidden width" hiddenWidth

/--
Fully-connected affine layer on vectors: $y=Wx+b$.

Parameters:
- `W : (outputWidth × inputWidth)` with Xavier uniform initialization under `weightSeed`,
- `b : (outputWidth)` initialized to zeros.

PyTorch analogy: `torch.nn.Linear(inputWidth, outputWidth)`.
-/
def linear (inputWidth outputWidth : Nat) (weightSeed : Nat := 0) :
    Layer ([inputWidth]) ([outputWidth]) :=
  let WShape : Shape := [outputWidth, inputWidth]
  let bShape : Shape := [outputWidth]
  let w0 : Tensor Float WShape :=
    Init.xavierUniform (outputWidth := outputWidth) (inputWidth := inputWidth)
      (seed := weightSeed)
  let b0 : Tensor Float bShape := Tensor.zeros (α := Float) bShape
  { kind := s!"Linear({inputWidth}, {outputWidth})"
    stateShapes := [WShape, bShape]
    initState := .cons w0 (.cons b0 .nil)
    runtimeInit :=
      some (.cons (.scheme (.xavierUniform inputWidth outputWidth) weightSeed)
        (.cons (.scheme .zeros) .nil))
    requiresGrad := #[true, true]
    validateConfig := do
      requirePositive "Linear" "input width" inputWidth
      requirePositive "Linear" "output width" outputWidth
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun w b x =>
          Ops.apply (m := m) (α := α)
              (.linear (lead := []) (inDim := inputWidth) (outDim := outputWidth)) (w, b, x, ())
  }

/-! ## Recurrent cells with explicit state -/

/-- The parameter shapes of an Elman RNN cell: the weight on `[x; h]` and the bias, in the order
of `Linear`. -/
def rnnCellShapes (inputWidth hiddenWidth : Nat) : List Shape :=
  [[hiddenWidth, inputWidth + hiddenWidth], [hiddenWidth]]

/-- The parameter shapes of an LSTM cell: a weight on `[x; h]` and a bias for the forget, input,
candidate and output gates, in the order of `TorchLean.LSTM`. -/
def lstmCellShapes (inputWidth hiddenWidth : Nat) : List Shape :=
  [[hiddenWidth, inputWidth + hiddenWidth], [hiddenWidth],
    [hiddenWidth, inputWidth + hiddenWidth], [hiddenWidth],
    [hiddenWidth, inputWidth + hiddenWidth], [hiddenWidth],
    [hiddenWidth, inputWidth + hiddenWidth], [hiddenWidth]]

/-- The parameter shapes of a GRU cell. Reset-before has a weight on `[x; h]` and a bias for the
reset, update and candidate gates (`GRU`). Reset-after has PyTorch's packed `weight_ih`,
`weight_hh`, `bias_ih` and `bias_hh` (`GRU.ResetAfter`). -/
def gruCellShapes : GRU.Convention → (inputWidth hiddenWidth : Nat) → List Shape
  | .resetBefore, inputWidth, hiddenWidth =>
      [[hiddenWidth, inputWidth + hiddenWidth], [hiddenWidth],
        [hiddenWidth, inputWidth + hiddenWidth], [hiddenWidth],
        [hiddenWidth, inputWidth + hiddenWidth], [hiddenWidth]]
  | .resetAfter, inputWidth, hiddenWidth =>
      [[3 * hiddenWidth, inputWidth], [3 * hiddenWidth, hiddenWidth], [3 * hiddenWidth],
        [3 * hiddenWidth]]

/-- One Elman RNN step from the previous hidden state: `tanh (W [x; h] + b)`. -/
def rnnCell {α : Type} [Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)] {inputWidth hiddenWidth : Nat}
    (params : Pack (Ref (m := m) (α := α)) (rnnCellShapes inputWidth hiddenWidth))
    (input : Ref (m := m) (α := α) [inputWidth])
    (previous : Ref (m := m) (α := α) [hiddenWidth]) : m (Ref (m := m) (α := α) [hiddenWidth]) :=
  match params with
  | .cons weight (.cons bias .nil) => do
      let concat ← Ops.apply (m := m) (α := α)
          (.cat (s := .dim 0 .scalar) ⟨0, by simp⟩ [inputWidth, hiddenWidth])
          (input, previous, ())
      let pre ← Ops.apply (m := m) (α := α)
          (.linear (lead := []) (inDim := inputWidth + hiddenWidth) (outDim := hiddenWidth))
          (weight, bias, concat, ())
      Ops.apply (m := m) (α := α) (.tanh (s := [hiddenWidth])) (pre, ())

/-- One LSTM step from the previous `(hidden, cell)` state to the next one, with the gate
equations of `TorchLean.LSTM.cell`. -/
def lstmCell {α : Type} [Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)] {inputWidth hiddenWidth : Nat}
    (params : Pack (Ref (m := m) (α := α)) (lstmCellShapes inputWidth hiddenWidth))
    (input : Ref (m := m) (α := α) [inputWidth])
    (previous : Ref (m := m) (α := α) [hiddenWidth] × Ref (m := m) (α := α) [hiddenWidth]) :
    m (Ref (m := m) (α := α) [hiddenWidth] × Ref (m := m) (α := α) [hiddenWidth]) :=
  match params with
  | .cons wF (.cons bF (.cons wI (.cons bI (.cons wC (.cons bC (.cons wO (.cons bO .nil))))))) => do
      let (hPrev, cPrev) := previous
      let concat ← Ops.apply (m := m) (α := α)
          (.cat (s := .dim 0 .scalar) ⟨0, by simp⟩ [inputWidth, hiddenWidth]) (input, hPrev, ())
      let f_pre ← Ops.apply (m := m) (α := α)
          (.linear (lead := []) (inDim := inputWidth + hiddenWidth) (outDim := hiddenWidth))
          (wF, bF, concat, ())
      let f ← Ops.apply (m := m) (α := α)
          (.sigmoid (s := [hiddenWidth])) (f_pre, ())
      let i_pre ← Ops.apply (m := m) (α := α)
          (.linear (lead := []) (inDim := inputWidth + hiddenWidth) (outDim := hiddenWidth))
          (wI, bI, concat, ())
      let i ← Ops.apply (m := m) (α := α)
          (.sigmoid (s := [hiddenWidth])) (i_pre, ())
      let g_pre ← Ops.apply (m := m) (α := α)
          (.linear (lead := []) (inDim := inputWidth + hiddenWidth) (outDim := hiddenWidth))
          (wC, bC, concat, ())
      let g ← Ops.apply (m := m) (α := α) (.tanh (s := [hiddenWidth]))
          (g_pre, ())
      let o_pre ← Ops.apply (m := m) (α := α)
          (.linear (lead := []) (inDim := inputWidth + hiddenWidth) (outDim := hiddenWidth))
          (wO, bO, concat, ())
      let o ← Ops.apply (m := m) (α := α)
          (.sigmoid (s := [hiddenWidth])) (o_pre, ())
      let fc ← Ops.apply (m := m) (α := α) (.mul (s := [hiddenWidth]))
          (f, cPrev, ())
      let ig ← Ops.apply (m := m) (α := α) (.mul (s := [hiddenWidth]))
          (i, g, ())
      let c_t ← Ops.apply (m := m) (α := α) (.add (s := [hiddenWidth]))
          (fc, ig, ())
      let tanhC ← Ops.apply (m := m) (α := α)
          (.tanh (s := [hiddenWidth])) (c_t, ())
      let h_t ← Ops.apply (m := m) (α := α) (.mul (s := [hiddenWidth]))
          (o, tanhC, ())
      pure (h_t, c_t)

/--
One GRU step from the previous hidden state under `convention`.

Reset-before is Cho et al.'s cell, `n = tanh (W_n [x; r * h] + b_n)`. Reset-after is PyTorch's,
which multiplies the whole recurrent candidate block, recurrent bias included, by the reset gate
(`GRU.ResetAfter`). Both compute `1 - z` as `sigmoid (-z_pre)`, which subtracts no rounded
gate from one but needs `neg`. An interpreter that refuses `neg`, such as the verifier recorder,
therefore refuses the cell.
-/
def gruCell {α : Type} [Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)] (convention : GRU.Convention)
    {inputWidth hiddenWidth : Nat}
    (params : Pack (Ref (m := m) (α := α)) (gruCellShapes convention inputWidth hiddenWidth))
    (input : Ref (m := m) (α := α) [inputWidth])
    (previous : Ref (m := m) (α := α) [hiddenWidth]) : m (Ref (m := m) (α := α) [hiddenWidth]) :=
  match convention, params with
  | .resetBefore,
      .cons wReset (.cons bReset (.cons wUpdate (.cons bUpdate (.cons wNew (.cons bNew .nil))))) =>
      do
      let concat ← Ops.apply (m := m) (α := α)
          (.cat (s := .dim 0 .scalar) ⟨0, by simp⟩ [inputWidth, hiddenWidth])
          (input, previous, ())
      let r_pre ← Ops.apply (m := m) (α := α)
          (.linear (lead := []) (inDim := inputWidth + hiddenWidth) (outDim := hiddenWidth))
          (wReset, bReset, concat, ())
      let r ← Ops.apply (m := m) (α := α)
          (.sigmoid (s := [hiddenWidth])) (r_pre, ())
      let z_pre ← Ops.apply (m := m) (α := α)
          (.linear (lead := []) (inDim := inputWidth + hiddenWidth) (outDim := hiddenWidth))
          (wUpdate, bUpdate, concat, ())
      let z ← Ops.apply (m := m) (α := α)
          (.sigmoid (s := [hiddenWidth])) (z_pre, ())
      let r_hPrev ← Ops.apply (m := m) (α := α)
          (.mul (s := [hiddenWidth])) (r, previous, ())
      let concat2 ← Ops.apply (m := m) (α := α)
          (.cat (s := .dim 0 .scalar) ⟨0, by simp⟩ [inputWidth, hiddenWidth])
          (input, r_hPrev, ())
      let n_pre ← Ops.apply (m := m) (α := α)
          (.linear (lead := []) (inDim := inputWidth + hiddenWidth) (outDim := hiddenWidth))
          (wNew, bNew, concat2, ())
      let n ← Ops.apply (m := m) (α := α) (.tanh (s := [hiddenWidth]))
          (n_pre, ())
      let complementPre ← Ops.apply (m := m) (α := α)
          (.neg (s := [hiddenWidth])) (z_pre, ())
      let oneMinusZ ← Ops.apply (m := m) (α := α)
          (.sigmoid (s := [hiddenWidth])) (complementPre, ())
      let newContrib ← Ops.apply (m := m) (α := α)
          (.mul (s := [hiddenWidth])) (oneMinusZ, n, ())
      let hiddenContrib ← Ops.apply (m := m) (α := α)
          (.mul (s := [hiddenWidth])) (z, previous, ())
      Ops.apply (m := m) (α := α) (.add (s := [hiddenWidth]))
          (newContrib, hiddenContrib, ())
  | .resetAfter, .cons inputWeight (.cons hiddenWeight (.cons inputBias (.cons hiddenBias .nil))) =>
      do
      let inputGates ← Ops.apply (m := m) (α := α)
          (.linear (lead := []) (inDim := inputWidth) (outDim := 3 * hiddenWidth))
          (inputWeight, inputBias, input, ())
      let hiddenGates ← Ops.apply (m := m) (α := α)
          (.linear (lead := []) (inDim := hiddenWidth) (outDim := 3 * hiddenWidth))
          (hiddenWeight, hiddenBias, previous, ())
      let gate := fun (values : Ref (m := m) (α := α) [3 * hiddenWidth]) (index : Fin 3) =>
        (Ref.cast
              (Shape.slice_singleton (.dim _ .scalar) ⟨0, by simp⟩ (index.val * hiddenWidth)
                hiddenWidth) <$> Ops.apply (m := m) (α := α)
              (.slice (s := .dim _ .scalar) [⟨0, by simp⟩] [index.val * hiddenWidth]
                [index.val * hiddenWidth + hiddenWidth] [1]
                (Shape.sliceValid_singleton _ ⟨0, by simp⟩ (index.val * hiddenWidth) hiddenWidth
                  (by
                    show _ ≤ 3 * hiddenWidth
                    simpa [Nat.add_mul] using
                      Nat.mul_le_mul_right hiddenWidth (Nat.succ_le_of_lt index.isLt))))
              (values, ()))
      let inputReset ← gate inputGates 0
      let hiddenReset ← gate hiddenGates 0
      let resetPre ← Ops.apply (m := m) (α := α)
          (.add (s := [hiddenWidth])) (inputReset, hiddenReset, ())
      let reset ← Ops.apply (m := m) (α := α)
          (.sigmoid (s := [hiddenWidth])) (resetPre, ())
      let inputUpdate ← gate inputGates 1
      let hiddenUpdate ← gate hiddenGates 1
      let updatePre ← Ops.apply (m := m) (α := α)
          (.add (s := [hiddenWidth])) (inputUpdate, hiddenUpdate, ())
      let update ← Ops.apply (m := m) (α := α)
          (.sigmoid (s := [hiddenWidth])) (updatePre, ())
      let inputCandidate ← gate inputGates 2
      let hiddenCandidate ← gate hiddenGates 2
      let resetCandidate ← Ops.apply (m := m) (α := α)
          (.mul (s := [hiddenWidth])) (reset, hiddenCandidate, ())
      let candidatePre ← Ops.apply (m := m) (α := α)
          (.add (s := [hiddenWidth])) (inputCandidate, resetCandidate, ())
      let candidate ← Ops.apply (m := m) (α := α)
          (.tanh (s := [hiddenWidth])) (candidatePre, ())
      let complementPre ← Ops.apply (m := m) (α := α)
          (.neg (s := [hiddenWidth])) (updatePre, ())
      let oneMinusUpdate ← Ops.apply (m := m) (α := α)
          (.sigmoid (s := [hiddenWidth])) (complementPre, ())
      let candidateContribution ← Ops.apply (m := m) (α := α)
          (.mul (s := [hiddenWidth])) (oneMinusUpdate, candidate, ())
      let previousContribution ← Ops.apply (m := m) (α := α)
          (.mul (s := [hiddenWidth])) (update, previous, ())
      Ops.apply (m := m) (α := α) (.add (s := [hiddenWidth]))
          (candidateContribution, previousContribution, ())

/--
Run `step` over the leading axis of `inputs` from `initial`: the step outputs stacked along that
axis, and the final state. An empty sequence gives a zero tensor and `initial`.
-/
def _root_.TorchLean.Recurrent.scan {α : Type} [Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)] {State : Type} {length : Nat}
    {σ τ : Shape} (step : Ref (m := m) (α := α) σ → State → m (State × Ref (m := m) (α := α) τ))
    (initial : State) (inputs : Ref (m := m) (α := α) (Shape.dim length σ)) :
    m (Ref (m := m) (α := α) (Shape.dim length τ) × State) :=
  scanRows (m := m) (α := α) initial fun t state => do
    let input ← Ops.apply (m := m) (α := α)
      (.indexAt (s := Shape.dim length σ) ⟨0, Shape.AxisInBounds.proof⟩ t) (inputs, ())
    step input state

/--
Vanilla RNN layer (time-major sequence, no batch axis).

Semantics:
$$
h_t=\tanh\!\left(W[x_t;h_{t-1}]+b\right),
\qquad h_{-1}=0.
$$

The layer unrolls `sequenceLength` steps of `nn.Layer.rnnCell`, which is built from TorchLean
operators, so it runs on the CPU and CUDA backends. The weight uses Xavier uniform initialization
under `weightSeed` and the bias starts at zero.

PyTorch analogy: `torch.nn.RNN(inputWidth, hiddenWidth, nonlinearity="tanh")` with
`batch_first=False`, specialized to a single batch element.
Docs: https://docs.pytorch.org/docs/stable/generated/torch.nn.RNN.html
-/
def rnn (sequenceLength inputWidth hiddenWidth : Nat) (weightSeed : Nat := 0) :
    Layer ([sequenceLength, inputWidth]) ([sequenceLength, hiddenWidth]) :=
  let WShape : Shape := [hiddenWidth, inputWidth + hiddenWidth]
  let bShape : Shape := [hiddenWidth]
  let w0 : Tensor Float WShape :=
    Init.xavierUniform (outputWidth := hiddenWidth)
      (inputWidth := inputWidth + hiddenWidth) (seed := weightSeed)
  let b0 : Tensor Float bShape := Tensor.zeros (α := Float) bShape
  { kind := s!"RNN({inputWidth}, {hiddenWidth})"
    stateShapes := [WShape, bShape]
    initState := .cons w0 (.cons b0 .nil)
    runtimeInit := some (.cons
      (.scheme (.xavierUniform (inputWidth + hiddenWidth) hiddenWidth) weightSeed)
      (.cons (.scheme .zeros) .nil))
    requiresGrad := #[true, true]
    validateConfig :=
      validateRecurrent "RNN" sequenceLength inputWidth hiddenWidth
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun w b xs => show m (Ref (m := m) (α := α) [sequenceLength, hiddenWidth]) from do
          let h0T : Tensor α [hiddenWidth] :=
            Tensor.full (α := α) ([hiddenWidth]) (0 : α)
          let h0 ← Ops.const (m := m) (α := α) (s := [hiddenWidth]) h0T
          Prod.fst <$> scanRows (m := m) (α := α) h0 (fun t hPrev => do
            let x_t ← Ops.apply (m := m) (α := α)
                (.indexAt (s := [sequenceLength, inputWidth]) ⟨0, Shape.AxisInBounds.proof⟩ t)
                (xs, ())
            let h_t ← rnnCell (m := m) (α := α) (.cons w (.cons b .nil)) x_t hPrev
            pure (h_t, h_t))
  }

/--
GRU layer (time-major sequence, no batch axis).

The Cho-style reset-before recurrence starts with $h_{-1}=0$ and computes

$$
\begin{aligned}
r_t &= \operatorname{sigmoid}(W_r[x_t;h_{t-1}]+b_r),\\
z_t &= \operatorname{sigmoid}(W_z[x_t;h_{t-1}]+b_z),\\
n_t &= \tanh\!\left(W_{nx}x_t+W_{nh}(r_t\odot h_{t-1})+b_n\right),\\
h_t &= (1-z_t)\odot n_t+z_t\odot h_{t-1}.
\end{aligned}
$$

The candidate matrix is $W_n=[W_{nx}\;W_{nh}]$: the reset multiplies the previous hidden state
before the hidden columns of this matrix are applied. PyTorch's GRU applies the reset after the
recurrent product:

$$
n_t = \tanh\!\left(W_{nx}x_t+b_{nx}+r_t\odot(W_{nh}h_{t-1}+b_{nh})\right).
$$

In that equation the reset multiplies the recurrent affine output, including the recurrent
candidate bias $b_{nh}$. A general matrix does not commute with elementwise reset multiplication,
and $r_t\odot b_{nh}$ is not a constant bias. Concatenating or reordering checkpoint tensors and
adding their biases therefore cannot generally turn a PyTorch GRU into this cell.

The six trainable tensors are stored as `wReset, bReset, wUpdate, bUpdate, wNew, bNew`. Every weight
has shape `[hiddenWidth, inputWidth + hiddenWidth]`, with input columns first and hidden columns
second, and every bias has shape `[hiddenWidth]`. PyTorch packs reset, update and candidate rows
into separate input and recurrent matrices, with a separate bias vector for each matrix. Its reset
and update bias pairs may be summed to reproduce those gates' forward equations. The candidate
bias and reset placement require the different recurrence shown above. Even where two biases can
be merged for forward evaluation, training one merged bias differs from updating two independent
bias parameters.

Weights use Xavier uniform initialization with the three supplied seeds, and biases start at zero.
Each forward call unrolls `[sequenceLength, inputWidth]` into `[sequenceLength, hiddenWidth]` and
starts from a fresh zero hidden state. The result contains every hidden state, including the last
one as its final row. The layer accepts no initial hidden state, returns no separate final state,
and carries no hidden state between calls. `gruCell .resetBefore` with `Recurrent.scan` runs the
same step from a given state and returns the final state. The public batched wrapper applies this
core to each sequence independently, with shared parameters.
-/
def gru (sequenceLength inputWidth hiddenWidth : Nat)
    (resetWeightSeed updateWeightSeed candidateWeightSeed : Nat := 0) :
    Layer ([sequenceLength, inputWidth]) ([sequenceLength, hiddenWidth]) :=
  let WShape : Shape := [hiddenWidth, inputWidth + hiddenWidth]
  let bShape : Shape := [hiddenWidth]
  let wReset0 : Tensor Float WShape := Init.xavierUniform
    (outputWidth := hiddenWidth) (inputWidth := inputWidth + hiddenWidth)
    (seed := resetWeightSeed)
  let bReset0 : Tensor Float bShape := Tensor.zeros (α := Float) bShape
  let wUpdate0 : Tensor Float WShape := Init.xavierUniform
    (outputWidth := hiddenWidth) (inputWidth := inputWidth + hiddenWidth)
    (seed := updateWeightSeed)
  let bUpdate0 : Tensor Float bShape := Tensor.zeros (α := Float) bShape
  let wNew0 : Tensor Float WShape := Init.xavierUniform
    (outputWidth := hiddenWidth) (inputWidth := inputWidth + hiddenWidth)
    (seed := candidateWeightSeed)
  let bNew0 : Tensor Float bShape := Tensor.zeros (α := Float) bShape
  { kind := s!"ChoGRU({inputWidth}, {hiddenWidth})"
    stateShapes := [WShape, bShape, WShape, bShape, WShape, bShape]
    initState :=
      .cons wReset0 (.cons bReset0 (.cons wUpdate0
        (.cons bUpdate0 (.cons wNew0 (.cons bNew0 .nil)))))
    runtimeInit := some <|
      .cons (.scheme (.xavierUniform (inputWidth + hiddenWidth) hiddenWidth) resetWeightSeed) <|
      .cons (.scheme .zeros) <|
      .cons (.scheme (.xavierUniform (inputWidth + hiddenWidth) hiddenWidth) updateWeightSeed) <|
      .cons (.scheme .zeros) <|
      .cons (.scheme (.xavierUniform (inputWidth + hiddenWidth) hiddenWidth) candidateWeightSeed) <|
      .cons (.scheme .zeros) .nil
    requiresGrad := #[true, true, true, true, true, true]
    validateConfig :=
      validateRecurrent "GRU" sequenceLength inputWidth hiddenWidth
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun wReset bReset wUpdate bUpdate wNew bNew xs =>
          show m (Ref (m := m) (α := α) [sequenceLength, hiddenWidth]) from do
          let h0T : Tensor α [hiddenWidth] :=
            Tensor.full (α := α) ([hiddenWidth]) (0 : α)
          let h0 ← Ops.const (m := m) (α := α) (s := [hiddenWidth]) h0T
          Prod.fst <$> scanRows (m := m) (α := α) h0 (fun t hPrev => do
            let x_t ← Ops.apply (m := m) (α := α)
                (.indexAt (s := [sequenceLength, inputWidth]) ⟨0, Shape.AxisInBounds.proof⟩ t)
                (xs, ())
            let h_t ← gruCell (m := m) (α := α) .resetBefore
              (.cons wReset (.cons bReset (.cons wUpdate (.cons bUpdate (.cons wNew
                (.cons bNew .nil)))))) x_t hPrev
            pure (h_t, h_t))
  }

/--
GRU sequence layer using the reset-after convention.

State order is `weight_ih, weight_hh, bias_ih, bias_hh`, with reset/update/candidate rows in each
tensor. The fields of `GRU.ResetAfter` come in the same order, so copied tensors need no
gate permutation or bias merging. Both weights use Xavier uniform initialization, under
`inputWeightSeed` and `hiddenWeightSeed`, and both biases start at zero. Each call starts from a
zero hidden state and returns every hidden state. `gruCell .resetAfter` with `Recurrent.scan` runs
the same step from a supplied state.
-/
def gruResetAfter (sequenceLength inputWidth hiddenWidth : Nat)
    (inputWeightSeed hiddenWeightSeed : Nat := 0) :
    Layer [sequenceLength, inputWidth] [sequenceLength, hiddenWidth] :=
  { kind := s!"GRU(resetAfter, {inputWidth}, {hiddenWidth})"
    stateShapes :=
      [[3 * hiddenWidth, inputWidth], [3 * hiddenWidth, hiddenWidth],
        [3 * hiddenWidth], [3 * hiddenWidth]]
    initState :=
      .cons (Init.xavierUniform
        (inputWidth := inputWidth) (outputWidth := 3 * hiddenWidth) (seed := inputWeightSeed)) <|
      .cons (Init.xavierUniform
        (inputWidth := hiddenWidth) (outputWidth := 3 * hiddenWidth) (seed := hiddenWeightSeed)) <|
      .cons (Tensor.zeros (α := Float) [3 * hiddenWidth]) <|
      .cons (Tensor.zeros (α := Float) [3 * hiddenWidth]) .nil
    runtimeInit := some <|
      .cons (.scheme (.xavierUniform inputWidth (3 * hiddenWidth)) inputWeightSeed) <|
      .cons (.scheme (.xavierUniform hiddenWidth (3 * hiddenWidth)) hiddenWeightSeed) <|
      .cons (.scheme .zeros) <| .cons (.scheme .zeros) .nil
    requiresGrad := #[true, true, true, true]
    validateConfig :=
      validateRecurrent "GRU" sequenceLength inputWidth hiddenWidth
    forward := fun _ {α} _ _ => fun {m} _ _ =>
      fun inputWeight hiddenWeight inputBias hiddenBias inputs =>
        show m (Ref (m := m) (α := α) [sequenceLength, hiddenWidth]) from do
        let initial ← Ops.const (m := m) (α := α) (Tensor.zeros (α := α) [hiddenWidth])
        Prod.fst <$> scanRows (m := m) (α := α) initial fun time previous => do
            let input ← Ops.apply (m := m) (α := α)
                (.indexAt (s := [sequenceLength, inputWidth]) ⟨0, Shape.AxisInBounds.proof⟩ time)
                (inputs, ())
            let hidden ← gruCell (m := m) (α := α) .resetAfter
              (.cons inputWeight (.cons hiddenWeight (.cons inputBias (.cons hiddenBias .nil))))
              input previous
            pure (hidden, hidden) }

/--
LSTM layer (time-major sequence, no batch axis).

The layer unrolls `sequenceLength` steps of `nn.Layer.lstmCell`, an LSTM step with the standard four
gates, from $(h_{-1},c_{-1})=(0,0)$, and returns every hidden state. The four gate weights use
Xavier uniform initialization, one seed each, and the biases start at zero.

PyTorch analogy: `torch.nn.LSTM(inputWidth, hiddenWidth)` with `batch_first=False`, specialized to a
single batch element.
Docs: https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html
-/
def lstm (sequenceLength inputWidth hiddenWidth : Nat)
    (forgetWeightSeed inputWeightSeed candidateWeightSeed outputWeightSeed : Nat := 0) :
    Layer ([sequenceLength, inputWidth]) ([sequenceLength, hiddenWidth]) :=
  let WShape : Shape := [hiddenWidth, inputWidth + hiddenWidth]
  let bShape : Shape := [hiddenWidth]
  let wF0 : Tensor Float WShape := Init.xavierUniform
    (outputWidth := hiddenWidth) (inputWidth := inputWidth + hiddenWidth)
    (seed := forgetWeightSeed)
  let bF0 : Tensor Float bShape := Tensor.zeros (α := Float) bShape
  let wI0 : Tensor Float WShape := Init.xavierUniform
    (outputWidth := hiddenWidth) (inputWidth := inputWidth + hiddenWidth)
    (seed := inputWeightSeed)
  let bI0 : Tensor Float bShape := Tensor.zeros (α := Float) bShape
  let wC0 : Tensor Float WShape := Init.xavierUniform
    (outputWidth := hiddenWidth) (inputWidth := inputWidth + hiddenWidth)
    (seed := candidateWeightSeed)
  let bC0 : Tensor Float bShape := Tensor.zeros (α := Float) bShape
  let wO0 : Tensor Float WShape := Init.xavierUniform
    (outputWidth := hiddenWidth) (inputWidth := inputWidth + hiddenWidth)
    (seed := outputWeightSeed)
  let bO0 : Tensor Float bShape := Tensor.zeros (α := Float) bShape
  { kind := s!"LSTM({inputWidth}, {hiddenWidth})"
    stateShapes := [WShape, bShape, WShape, bShape, WShape, bShape, WShape, bShape]
    initState :=
      .cons wF0 (.cons bF0 (.cons wI0 (.cons bI0
        (.cons wC0 (.cons bC0 (.cons wO0 (.cons bO0 .nil)))))))
    runtimeInit := some <|
      .cons (.scheme (.xavierUniform (inputWidth + hiddenWidth) hiddenWidth) forgetWeightSeed) <|
      .cons (.scheme .zeros) <|
      .cons (.scheme (.xavierUniform (inputWidth + hiddenWidth) hiddenWidth) inputWeightSeed) <|
      .cons (.scheme .zeros) <|
      .cons (.scheme (.xavierUniform (inputWidth + hiddenWidth) hiddenWidth) candidateWeightSeed) <|
      .cons (.scheme .zeros) <|
      .cons (.scheme (.xavierUniform (inputWidth + hiddenWidth) hiddenWidth) outputWeightSeed) <|
      .cons (.scheme .zeros) .nil
    requiresGrad := #[true, true, true, true, true, true, true, true]
    validateConfig :=
      validateRecurrent "LSTM" sequenceLength inputWidth hiddenWidth
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun wF bF wI bI wC bC wO bO xs =>
          show m (Ref (m := m) (α := α) [sequenceLength, hiddenWidth]) from do
          let h0T : Tensor α [hiddenWidth] :=
            Tensor.full (α := α) ([hiddenWidth]) (0 : α)
          let h0 ← Ops.const (m := m) (α := α) (s := [hiddenWidth]) h0T
          let c0 ← Ops.const (m := m) (α := α) (s := [hiddenWidth]) h0T
          Prod.fst <$> scanRows (m := m) (α := α) (h0, c0) (fun t st => do
            let (hPrev, cPrev) := st
            let x_t ← Ops.apply (m := m) (α := α)
                (.indexAt (s := [sequenceLength, inputWidth]) ⟨0, Shape.AxisInBounds.proof⟩ t)
                (xs, ())
            let next ← lstmCell (m := m) (α := α)
              (.cons wF (.cons bF (.cons wI (.cons bI (.cons wC (.cons bC (.cons wO
                (.cons bO .nil)))))))) x_t (hPrev, cPrev)
            pure (next, next.1))
  }

end nn.Layer
end TorchLean
