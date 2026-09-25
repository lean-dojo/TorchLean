/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Model.Layers.Core
public import NN.Spec.Layers.Gru

/-!
# TorchLean NN: Linear and Recurrent Layers
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace Model

open Spec TorchLean
open TorchLean TorchLean.Tensor
open Proofs.Autograd.Algebra

namespace Layers

/-! ## Convenience constructors (layers) -/

namespace Internal

/-- Write one leading-axis slice through the general `scatterAdd` operation. -/
def writeLeading {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {rows : Nat} {tail : Shape}
    (base : Ref (m := m) (α := α) (tail.prependDim rows))
    (value : Ref (m := m) (α := α) tail) (index : Fin rows) :
    m (Ref (m := m) (α := α) (tail.prependDim rows)) := do
  let source ← Runtime.Autograd.Model.reshape (m := m) (α := α)
    (s₁ := tail) (s₂ := tail.prependDim 1) value (by simp [Shape.size])
  let indices : Tensor (Fin rows) [1] := Tensor.ofFn fun _ => index
  Runtime.Autograd.Model.scatterAdd (m := m) (α := α) (s := tail.prependDim rows) 0 1
    base source (Runtime.Autograd.Torch.dataConst (m := m) (α := α) indices)

/--
Stack `count` consecutive one-row tensors starting at `start` along the leading axis.

Halves are joined pairwise, so each row is copied once per level of a balanced tree and the whole
stack costs `O(count * log count)` copies instead of one full-output copy per row.
-/
def stackRows {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {rows : Nat} {tail : Shape}
    (row : Fin rows → Ref (m := m) (α := α) (tail.prependDim 1)) :
    (start count : Nat) → start + count ≤ rows → 0 < count →
      m (Ref (m := m) (α := α) (tail.prependDim count))
  | start, 1, hRange, _ => pure (row ⟨start, by omega⟩)
  | start, count + 2, hRange, _ => do
      let half := (count + 2) / 2
      let left ← stackRows row start half (by omega) (by omega)
      let right ← stackRows row (start + half) (count + 2 - half) (by omega) (by omega)
      let joined ← Runtime.Autograd.Model.concatLeadingAxis (m := m) (α := α)
        (nDim := half) (mDim := count + 2 - half) (s := tail) left right
      have hCount : half + (count + 2 - half) = count + 2 := by omega
      pure (hCount ▸ joined)
  termination_by _ count => count

/-- Run `step` for each leading index in order and keep every reshaped result. -/
def collectRows {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {State : Type} {rows : Nat} {tail : Shape}
    (step : Fin rows → State → m (State × Ref (m := m) (α := α) tail)) :
    (index : Nat) → index ≤ rows → State →
      (acc : Array (Ref (m := m) (α := α) (tail.prependDim 1))) → acc.size = index →
      m { result : Array (Ref (m := m) (α := α) (tail.prependDim 1)) // result.size = rows }
  | index, hIndex, state, acc, hAcc =>
      if hLt : index < rows then do
        let (next, value) ← step ⟨index, hLt⟩ state
        let row ← Runtime.Autograd.Model.reshape (m := m) (α := α)
          (s₁ := tail) (s₂ := tail.prependDim 1) value (by simp [Shape.size])
        collectRows step (index + 1) hLt next (acc.push row) (by rw [Array.size_push, hAcc])
      else
        pure ⟨acc, by omega⟩
  termination_by index => rows - index

/--
Unroll a recurrence over the leading axis and add the stacked step outputs to `base`.

Row `t` of the result is `base[t] + value_t`, where `value_t` is the output of step `t`. This is
the tensor that writing each step through `writeLeading` produces, built with one stack and one
addition instead of one full-output scatter per step.
-/
def unrollLeading {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {State : Type} {rows : Nat} {tail : Shape}
    (base : Ref (m := m) (α := α) (tail.prependDim rows)) (initial : State)
    (step : Fin rows → State → m (State × Ref (m := m) (α := α) tail)) :
    m (Ref (m := m) (α := α) (tail.prependDim rows)) := do
  let stepRows ← collectRows (m := m) (α := α) step 0 (Nat.zero_le rows) initial #[] rfl
  if hRows : 0 < rows then
    let stacked ← stackRows (m := m) (α := α) (rows := rows)
      (fun index => stepRows.val[index.val]'(by rw [stepRows.property]; exact index.isLt))
      0 rows (by omega) hRows
    Runtime.Autograd.Model.add (m := m) (α := α) base stacked
  else
    pure base

/-- Validate one positive architectural dimension. -/
def requirePositive (kind field : String) (value : Nat) : Except String Unit := do
  if value = 0 then
    throw s!"{kind}: {field} must be positive"

/-- Validate the dimensions shared by recurrent layers. -/
def validateRecurrentDimensions
    (kind : String) (sequenceLength inputWidth hiddenWidth : Nat) :
    Except String Unit := do
  requirePositive kind "sequence length" sequenceLength
  requirePositive kind "input width" inputWidth
  requirePositive kind "hidden width" hiddenWidth

end Internal

/--
Fully-connected affine layer on vectors: $y=Wx+b$.

Parameters:
- `W : (outputWidth × inputWidth)` initialized with Xavier initialization,
- `b : (outputWidth)` initialized to zeros.

PyTorch analogy: `torch.nn.Linear(inputWidth, outputWidth)`.
-/
def linear (inputWidth outputWidth : Nat) (weightSeed : Nat := 0) :
    Layer ([inputWidth]) ([outputWidth]) :=
  let WShape : Shape := [outputWidth, inputWidth]
  let bShape : Shape := [outputWidth]
  let w0 : Tensor Float WShape :=
    Torch.Init.xavierUniform (outDim := outputWidth) (inDim := inputWidth) (seed := weightSeed)
  let b0 : Tensor Float bShape := Tensor.zeros (α := Float) bShape
  { kind := s!"Linear({inputWidth}, {outputWidth})"
    stateShapes := [WShape, bShape]
    initState := .cons w0 (.cons b0 .nil)
    runtimeInit :=
      some (.cons (.xavierUniform inputWidth outputWidth weightSeed) (.cons .zeros .nil))
    requiresGrad := #[true, true]
    validateConfig := do
      Internal.requirePositive "Linear" "input width" inputWidth
      Internal.requirePositive "Linear" "output width" outputWidth
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun w b x =>
          Runtime.Autograd.Torch.linear
            (m := m) (α := α) (inDim := inputWidth) (outDim := outputWidth) w b x
  }

/--
Vanilla RNN layer (time-major sequence, no batch axis).

Semantics:
$$
h_t=\tanh\!\left(W[x_t;h_{t-1}]+b\right),
\qquad h_{-1}=0.
$$

This is implemented by unrolling a fixed number of steps (`sequenceLength`) using existing
TorchLean ops, so it works on both CPU and CUDA backends.

PyTorch analogy: `torch.nn.RNN(inputWidth, hiddenWidth, nonlinearity="tanh")` with
`batch_first=false`, specialized to a single batch element.
Docs: https://docs.pytorch.org/docs/stable/generated/torch.nn.RNN.html
-/
def rnn (sequenceLength inputWidth hiddenWidth : Nat) (weightSeed : Nat := 0) :
    Layer ([sequenceLength, inputWidth]) ([sequenceLength, hiddenWidth]) :=
  let WShape : Shape := [hiddenWidth, inputWidth + hiddenWidth]
  let bShape : Shape := [hiddenWidth]
  let w0 : Tensor Float WShape :=
    Torch.Init.xavierUniform (outDim := hiddenWidth) (inDim := inputWidth + hiddenWidth)
      (seed := weightSeed)
  let b0 : Tensor Float bShape := Tensor.zeros (α := Float) bShape
  { kind := s!"RNN({inputWidth}, {hiddenWidth})"
    stateShapes := [WShape, bShape]
    initState := .cons w0 (.cons b0 .nil)
    runtimeInit := some (.cons (.xavierUniform (inputWidth + hiddenWidth) hiddenWidth weightSeed)
      (.cons .zeros .nil))
    requiresGrad := #[true, true]
    validateConfig :=
      Internal.validateRecurrentDimensions "RNN" sequenceLength inputWidth hiddenWidth
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun w b xs => show m (Ref ([sequenceLength, hiddenWidth])) from do
          let h0T : Tensor α [hiddenWidth] :=
            Tensor.full (α := α) ([hiddenWidth]) (0 : α)
          let out0T : Tensor α [sequenceLength, hiddenWidth] :=
            Tensor.full (α := α) ([sequenceLength, hiddenWidth]) (0 : α)
          let h0 ← Runtime.Autograd.Model.const (m := m) (α := α) (s := [hiddenWidth]) h0T
          let out0 ← Runtime.Autograd.Model.const (m := m) (α := α)
            (s := [sequenceLength, hiddenWidth]) out0T
          Internal.unrollLeading (m := m) (α := α) out0 h0 (fun t hPrev => do
            let x_t ← Runtime.Autograd.Model.select (m := m) (α := α)
              (s := [sequenceLength, inputWidth]) 0 xs t
            let concat ← Runtime.Autograd.Model.concatLeadingAxis (m := m) (α := α) (s := .scalar)
              (nDim := inputWidth) (mDim := hiddenWidth) x_t hPrev
            let pre ← Runtime.Autograd.Torch.linear (m := m) (α := α)
              (inDim := inputWidth + hiddenWidth) (outDim := hiddenWidth)
              w b concat
            let h_t ← Runtime.Autograd.Model.tanh (m := m) (α := α) (s := [hiddenWidth]) pre
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
before the hidden columns of this matrix are applied. PyTorch uses reset-after instead:

$$
n_t = \tanh\!\left(W_{nx}x_t+b_{nx}+r_t\odot(W_{nh}h_{t-1}+b_{nh})\right).
$$

In that equation the reset multiplies the recurrent affine output, including the recurrent
candidate bias $b_{nh}$. A general matrix does not commute with elementwise reset multiplication,
and $r_t\odot b_{nh}$ is not a constant bias. Concatenating or reordering checkpoint tensors and
adding their biases therefore cannot generally turn a PyTorch GRU into this cell.

The six trainable tensors are stored as `wReset, bReset, wUpdate, bUpdate, wNew, bNew`. Every weight
has shape `[hiddenWidth, inputWidth + hiddenWidth]`, with input columns first and hidden columns
second; every bias has shape `[hiddenWidth]`. PyTorch packs reset, update and candidate rows into
separate input and recurrent matrices, with a separate bias vector for each matrix. Its reset and
update bias pairs may be summed to reproduce those gates' forward equations. The candidate bias
and reset placement require the different recurrence shown above. Even where two biases can be
merged for forward evaluation, training one merged bias differs from updating two independent
bias parameters.

Weights use Xavier uniform initialization with the three supplied seeds; biases start at zero.
Each forward call unrolls `[sequenceLength, inputWidth]` into `[sequenceLength, hiddenWidth]` and
starts from a fresh zero hidden state. The result contains every hidden state, including the last
one as its final row. The layer accepts no initial hidden state, returns no separate final state,
and carries no hidden state between calls. The public batched wrapper applies this same core
independently to each sequence with shared parameters.
-/
def gru (sequenceLength inputWidth hiddenWidth : Nat)
    (resetWeightSeed updateWeightSeed candidateWeightSeed : Nat := 0) :
    Layer ([sequenceLength, inputWidth]) ([sequenceLength, hiddenWidth]) :=
  let WShape : Shape := [hiddenWidth, inputWidth + hiddenWidth]
  let bShape : Shape := [hiddenWidth]
  let wReset0 : Tensor Float WShape := Torch.Init.xavierUniform
    (outDim := hiddenWidth) (inDim := inputWidth + hiddenWidth) (seed := resetWeightSeed)
  let bReset0 : Tensor Float bShape := Tensor.zeros (α := Float) bShape
  let wUpdate0 : Tensor Float WShape := Torch.Init.xavierUniform
    (outDim := hiddenWidth) (inDim := inputWidth + hiddenWidth) (seed := updateWeightSeed)
  let bUpdate0 : Tensor Float bShape := Tensor.zeros (α := Float) bShape
  let wNew0 : Tensor Float WShape := Torch.Init.xavierUniform
    (outDim := hiddenWidth) (inDim := inputWidth + hiddenWidth) (seed := candidateWeightSeed)
  let bNew0 : Tensor Float bShape := Tensor.zeros (α := Float) bShape
  { kind := s!"ChoGRU({inputWidth}, {hiddenWidth})"
    stateShapes := [WShape, bShape, WShape, bShape, WShape, bShape]
    initState :=
      .cons wReset0 (.cons bReset0 (.cons wUpdate0
        (.cons bUpdate0 (.cons wNew0 (.cons bNew0 .nil)))))
    runtimeInit := some <|
      .cons (.xavierUniform (inputWidth + hiddenWidth) hiddenWidth resetWeightSeed) <|
      .cons .zeros <|
      .cons (.xavierUniform (inputWidth + hiddenWidth) hiddenWidth updateWeightSeed) <|
      .cons .zeros <|
      .cons (.xavierUniform (inputWidth + hiddenWidth) hiddenWidth candidateWeightSeed) <|
      .cons .zeros .nil
    requiresGrad := #[true, true, true, true, true, true]
    validateConfig :=
      Internal.validateRecurrentDimensions "GRU" sequenceLength inputWidth hiddenWidth
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun wReset bReset wUpdate bUpdate wNew bNew xs =>
          show m (Ref ([sequenceLength, hiddenWidth])) from do
          let h0T : Tensor α [hiddenWidth] :=
            Tensor.full (α := α) ([hiddenWidth]) (0 : α)
          let out0T : Tensor α [sequenceLength, hiddenWidth] :=
            Tensor.full (α := α) ([sequenceLength, hiddenWidth]) (0 : α)
          let onesT : Tensor α [hiddenWidth] :=
            Tensor.full (α := α) ([hiddenWidth]) (1 : α)
          let h0 ← Runtime.Autograd.Model.const (m := m) (α := α) (s := [hiddenWidth]) h0T
          let out0 ← Runtime.Autograd.Model.const (m := m) (α := α)
            (s := [sequenceLength, hiddenWidth]) out0T
          let ones ← Runtime.Autograd.Model.const (m := m) (α := α) (s := [hiddenWidth]) onesT
          Internal.unrollLeading (m := m) (α := α) out0 h0 (fun t hPrev => do
            let x_t ← Runtime.Autograd.Model.select (m := m) (α := α)
              (s := [sequenceLength, inputWidth]) 0 xs t
            let concat ← Runtime.Autograd.Model.concatLeadingAxis (m := m) (α := α) (s := .scalar)
              (nDim := inputWidth) (mDim := hiddenWidth) x_t hPrev
            let r_pre ← Runtime.Autograd.Torch.linear (m := m) (α := α)
              (inDim := inputWidth + hiddenWidth) (outDim := hiddenWidth)
              wReset bReset concat
            let r ← Runtime.Autograd.Model.sigmoid (m := m) (α := α) (s := [hiddenWidth]) r_pre
            let z_pre ← Runtime.Autograd.Torch.linear (m := m) (α := α)
              (inDim := inputWidth + hiddenWidth) (outDim := hiddenWidth)
              wUpdate bUpdate concat
            let z ← Runtime.Autograd.Model.sigmoid (m := m) (α := α) (s := [hiddenWidth]) z_pre
            let r_hPrev ← Runtime.Autograd.Model.mul (m := m) (α := α) (s := [hiddenWidth]) r hPrev
            let concat2 ← Runtime.Autograd.Model.concatLeadingAxis (m := m) (α := α) (s := .scalar)
              (nDim := inputWidth) (mDim := hiddenWidth) x_t r_hPrev
            let n_pre ← Runtime.Autograd.Torch.linear (m := m) (α := α)
              (inDim := inputWidth + hiddenWidth) (outDim := hiddenWidth)
              wNew bNew concat2
            let n ← Runtime.Autograd.Model.tanh (m := m) (α := α) (s := [hiddenWidth]) n_pre
            let oneMinusZ ← Runtime.Autograd.Model.sub (m := m) (α := α) (s := [hiddenWidth]) ones z
            let newContrib ← Runtime.Autograd.Model.mul (m := m) (α := α) (s := [hiddenWidth])
              oneMinusZ n
            let hiddenContrib ← Runtime.Autograd.Model.mul (m := m) (α := α) (s := [hiddenWidth])
              z hPrev
            let h_t ← Runtime.Autograd.Model.add (m := m) (α := α) (s := [hiddenWidth])
              newContrib hiddenContrib
            pure (h_t, h_t))
  }

/--
One differentiable reset-after GRU step with PyTorch's four packed parameter tensors.

The input and hidden affine maps each produce reset, update, and candidate blocks. We apply the
reset to the whole recurrent candidate block, including its bias. All four parameter tensors and
the previous hidden state remain ordinary autograd references, so the same program supports
copied-parameter evaluation, reverse-mode gradients, and differentiation through several steps.
-/
def gruResetAfterCell {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {inputWidth hiddenWidth : Nat}
    (inputWeight : Ref (m := m) (α := α) [3 * hiddenWidth, inputWidth])
    (hiddenWeight : Ref (m := m) (α := α) [3 * hiddenWidth, hiddenWidth])
    (inputBias hiddenBias : Ref (m := m) (α := α) [3 * hiddenWidth])
    (input : Ref (m := m) (α := α) [inputWidth])
    (previous : Ref (m := m) (α := α) [hiddenWidth]) :
    m (Ref (m := m) (α := α) [hiddenWidth]) := do
  let inputGates ← Runtime.Autograd.Torch.linear (m := m) (α := α)
    (inDim := inputWidth) (outDim := 3 * hiddenWidth)
    inputWeight inputBias input
  let hiddenGates ← Runtime.Autograd.Torch.linear (m := m) (α := α)
    (inDim := hiddenWidth) (outDim := 3 * hiddenWidth)
    hiddenWeight hiddenBias previous
  let gate := fun (values : Ref (m := m) (α := α) [3 * hiddenWidth]) (index : Fin 3) =>
    sliceLeadingAxisRange (m := m) (α := α) (s := .scalar)
      (index.val * hiddenWidth) hiddenWidth (by
        simpa [Nat.add_mul] using
          Nat.mul_le_mul_right hiddenWidth (Nat.succ_le_of_lt index.isLt)) values
  let inputReset ← gate inputGates 0
  let hiddenReset ← gate hiddenGates 0
  let resetPre ← add (m := m) (α := α) (s := [hiddenWidth]) inputReset hiddenReset
  let reset ← sigmoid (m := m) (α := α) (s := [hiddenWidth]) resetPre
  let inputUpdate ← gate inputGates 1
  let hiddenUpdate ← gate hiddenGates 1
  let updatePre ← add (m := m) (α := α) (s := [hiddenWidth]) inputUpdate hiddenUpdate
  let update ← sigmoid (m := m) (α := α) (s := [hiddenWidth]) updatePre
  let inputCandidate ← gate inputGates 2
  let hiddenCandidate ← gate hiddenGates 2
  let resetCandidate ← mul (m := m) (α := α) (s := [hiddenWidth]) reset hiddenCandidate
  let candidatePre ← add (m := m) (α := α) (s := [hiddenWidth]) inputCandidate resetCandidate
  let candidate ← tanh (m := m) (α := α) (s := [hiddenWidth]) candidatePre
  let ones ← const (m := m) (α := α) (Tensor.full [hiddenWidth] (1 : α))
  let oneMinusUpdate ← sub (m := m) (α := α) (s := [hiddenWidth]) ones update
  let candidateContribution ← mul (m := m) (α := α) (s := [hiddenWidth]) oneMinusUpdate candidate
  let previousContribution ← mul (m := m) (α := α) (s := [hiddenWidth]) update previous
  add (m := m) (α := α) (s := [hiddenWidth]) candidateContribution previousContribution

/--
GRU sequence layer using the reset-after convention.

State order is `weight_ih, weight_hh, bias_ih, bias_hh`, with reset/update/candidate rows in each
tensor. This is also the order accepted by `Spec.GRUResetAfterSpec.ofPyTorch`; copied tensors need
no gate permutation or bias merging. Each call starts at zero and returns every hidden state.
Use `gruResetAfterCell` when the initial state is supplied by another part of the model.
-/
def gruResetAfter (sequenceLength inputWidth hiddenWidth : Nat)
    (inputWeightSeed hiddenWeightSeed : Nat := 0) :
    Layer [sequenceLength, inputWidth] [sequenceLength, hiddenWidth] :=
  { kind := s!"GRU(resetAfter, {inputWidth}, {hiddenWidth})"
    stateShapes :=
      [[3 * hiddenWidth, inputWidth], [3 * hiddenWidth, hiddenWidth],
        [3 * hiddenWidth], [3 * hiddenWidth]]
    initState :=
      .cons (Torch.Init.xavierUniform
        (inDim := inputWidth) (outDim := 3 * hiddenWidth) (seed := inputWeightSeed)) <|
      .cons (Torch.Init.xavierUniform
        (inDim := hiddenWidth) (outDim := 3 * hiddenWidth) (seed := hiddenWeightSeed)) <|
      .cons (Tensor.zeros (α := Float) [3 * hiddenWidth]) <|
      .cons (Tensor.zeros (α := Float) [3 * hiddenWidth]) .nil
    runtimeInit := some <|
      .cons (.xavierUniform inputWidth (3 * hiddenWidth) inputWeightSeed) <|
      .cons (.xavierUniform hiddenWidth (3 * hiddenWidth) hiddenWeightSeed) <|
      .cons .zeros <| .cons .zeros .nil
    requiresGrad := #[true, true, true, true]
    validateConfig :=
      Internal.validateRecurrentDimensions "GRU" sequenceLength inputWidth hiddenWidth
    forward := fun _ {α} _ _ => fun {m} _ _ =>
      fun inputWeight hiddenWeight inputBias hiddenBias inputs =>
        show m (Ref (m := m) (α := α) [sequenceLength, hiddenWidth]) from do
        let initial ← const (m := m) (α := α) (Tensor.zeros (α := α) [hiddenWidth])
        let output ← const (m := m) (α := α)
          (Tensor.zeros (α := α) [sequenceLength, hiddenWidth])
        Internal.unrollLeading (m := m) (α := α) output initial fun time previous => do
            let input ← select (m := m) (α := α) (s := [sequenceLength, inputWidth])
              0 inputs time
            let hidden ← gruResetAfterCell (m := m) (α := α)
              (inputWidth := inputWidth) (hiddenWidth := hiddenWidth)
              inputWeight hiddenWeight inputBias hiddenBias input previous
            pure (hidden, hidden) }

/--
LSTM layer (time-major sequence, no batch axis).

This is an unrolled LSTM using the standard four gates, with
$(h_{-1},c_{-1})=(0,0)$.

PyTorch analogy: `torch.nn.LSTM(inputWidth, hiddenWidth)` with `batch_first=false`, specialized to a
single batch element.
Docs: https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html
-/
def lstm (sequenceLength inputWidth hiddenWidth : Nat)
    (forgetWeightSeed inputWeightSeed candidateWeightSeed outputWeightSeed : Nat := 0) :
    Layer ([sequenceLength, inputWidth]) ([sequenceLength, hiddenWidth]) :=
  let WShape : Shape := [hiddenWidth, inputWidth + hiddenWidth]
  let bShape : Shape := [hiddenWidth]
  let wF0 : Tensor Float WShape := Torch.Init.xavierUniform
    (outDim := hiddenWidth) (inDim := inputWidth + hiddenWidth) (seed := forgetWeightSeed)
  let bF0 : Tensor Float bShape := Tensor.zeros (α := Float) bShape
  let wI0 : Tensor Float WShape := Torch.Init.xavierUniform
    (outDim := hiddenWidth) (inDim := inputWidth + hiddenWidth) (seed := inputWeightSeed)
  let bI0 : Tensor Float bShape := Tensor.zeros (α := Float) bShape
  let wC0 : Tensor Float WShape := Torch.Init.xavierUniform
    (outDim := hiddenWidth) (inDim := inputWidth + hiddenWidth) (seed := candidateWeightSeed)
  let bC0 : Tensor Float bShape := Tensor.zeros (α := Float) bShape
  let wO0 : Tensor Float WShape := Torch.Init.xavierUniform
    (outDim := hiddenWidth) (inDim := inputWidth + hiddenWidth) (seed := outputWeightSeed)
  let bO0 : Tensor Float bShape := Tensor.zeros (α := Float) bShape
  { kind := s!"LSTM({inputWidth}, {hiddenWidth})"
    stateShapes := [WShape, bShape, WShape, bShape, WShape, bShape, WShape, bShape]
    initState :=
      .cons wF0 (.cons bF0 (.cons wI0 (.cons bI0
        (.cons wC0 (.cons bC0 (.cons wO0 (.cons bO0 .nil)))))))
    runtimeInit := some <|
      .cons (.xavierUniform (inputWidth + hiddenWidth) hiddenWidth forgetWeightSeed) <|
      .cons .zeros <|
      .cons (.xavierUniform (inputWidth + hiddenWidth) hiddenWidth inputWeightSeed) <|
      .cons .zeros <|
      .cons (.xavierUniform (inputWidth + hiddenWidth) hiddenWidth candidateWeightSeed) <|
      .cons .zeros <|
      .cons (.xavierUniform (inputWidth + hiddenWidth) hiddenWidth outputWeightSeed) <|
      .cons .zeros .nil
    requiresGrad := #[true, true, true, true, true, true, true, true]
    validateConfig :=
      Internal.validateRecurrentDimensions "LSTM" sequenceLength inputWidth hiddenWidth
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun wF bF wI bI wC bC wO bO xs =>
          show m (Ref ([sequenceLength, hiddenWidth])) from do
          let h0T : Tensor α [hiddenWidth] :=
            Tensor.full (α := α) ([hiddenWidth]) (0 : α)
          let out0T : Tensor α [sequenceLength, hiddenWidth] :=
            Tensor.full (α := α) ([sequenceLength, hiddenWidth]) (0 : α)
          let h0 ← Runtime.Autograd.Model.const (m := m) (α := α) (s := [hiddenWidth]) h0T
          let c0 ← Runtime.Autograd.Model.const (m := m) (α := α) (s := [hiddenWidth]) h0T
          let out0 ← Runtime.Autograd.Model.const (m := m) (α := α)
            (s := [sequenceLength, hiddenWidth]) out0T
          Internal.unrollLeading (m := m) (α := α) out0 (h0, c0) (fun t st => do
            let (hPrev, cPrev) := st
            let x_t ← Runtime.Autograd.Model.select (m := m) (α := α)
              (s := [sequenceLength, inputWidth]) 0 xs t
            let concat ← Runtime.Autograd.Model.concatLeadingAxis (m := m) (α := α) (s := .scalar)
              (nDim := inputWidth) (mDim := hiddenWidth) x_t hPrev
            let f_pre ← Runtime.Autograd.Torch.linear (m := m) (α := α)
              (inDim := inputWidth + hiddenWidth) (outDim := hiddenWidth)
              wF bF concat
            let f ← Runtime.Autograd.Model.sigmoid (m := m) (α := α) (s := [hiddenWidth]) f_pre
            let i_pre ← Runtime.Autograd.Torch.linear (m := m) (α := α)
              (inDim := inputWidth + hiddenWidth) (outDim := hiddenWidth)
              wI bI concat
            let i ← Runtime.Autograd.Model.sigmoid (m := m) (α := α) (s := [hiddenWidth]) i_pre
            let g_pre ← Runtime.Autograd.Torch.linear (m := m) (α := α)
              (inDim := inputWidth + hiddenWidth) (outDim := hiddenWidth)
              wC bC concat
            let g ← Runtime.Autograd.Model.tanh (m := m) (α := α) (s := [hiddenWidth]) g_pre
            let o_pre ← Runtime.Autograd.Torch.linear (m := m) (α := α)
              (inDim := inputWidth + hiddenWidth) (outDim := hiddenWidth)
              wO bO concat
            let o ← Runtime.Autograd.Model.sigmoid (m := m) (α := α) (s := [hiddenWidth]) o_pre
            let fc ← Runtime.Autograd.Model.mul (m := m) (α := α) (s := [hiddenWidth]) f cPrev
            let ig ← Runtime.Autograd.Model.mul (m := m) (α := α) (s := [hiddenWidth]) i g
            let c_t ← Runtime.Autograd.Model.add (m := m) (α := α) (s := [hiddenWidth]) fc ig
            let tanhC ← Runtime.Autograd.Model.tanh (m := m) (α := α) (s := [hiddenWidth]) c_t
            let h_t ← Runtime.Autograd.Model.mul (m := m) (α := α) (s := [hiddenWidth]) o tanhC
            pure ((h_t, c_t), h_t))
  }
end Layers

end Model
end Autograd
end Runtime
