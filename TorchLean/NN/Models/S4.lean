/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.StateSpace
public import TorchLean.Tensor.Contract

/-!
# Diagonal S4-style state-space layer

TorchLean's diagonal recurrent SSM layer in the S4 family runs the state-space recurrence of
S4-style models

`h_{t+1} = A h_t + B x_t`,
`y_t     = C h_{t+1} + D x_t`,

with elementwise (diagonal) `A`, `B`, `C` and `D`. Here `x_t` is the token after the input
projection `inProj`, and `outProj` maps `y_t` to the output width.

We use the diagonal form because it shares the selective-scan core of the Mamba-style models and
admits direct recurrence proofs. As a linear time-invariant recurrence it also has the
convolutional form of the S4 paper, which this module does not build.

The step is the recurrent view of Gu, Goel and Ré (2022), eq. (3), coded by hand. The readout
`C h_{t+1}` reads the updated state, as their `y_k = C x_k` does, and `D x_t` is the skip term of
their eq. (1). Our `A` and `B` are the discrete `Ā` and `B̄` themselves. S4 computes those from a
continuous `(A, B)` and a step size `Δ` by the bilinear method, starts `A` at the HiPPO matrix of
eq. (2) and keeps an `N`-dimensional state in each of its `H` copies (Section 3.4). Each of our
`stateDim` channels is such a copy with `N = 1`, fed by `inProj` and mixed by `outProj`.

Reference: Albert Gu, Karan Goel, and Christopher Ré (2022), "Efficiently Modeling Long Sequences
with Structured State Spaces", ICLR 2022, arXiv:2111.00396, eqs. (1) to (5), Section 3.4.

## Implementation status

No API builder implements this layer, and the `nn.*` builders have no S4 layer.
`S4.runArray_causal` in
`TorchLean/NN/StateSpace/Causality.lean` proves that extending the input stream
preserves the earlier outputs. Its proof rests on the scan algebra in
`TorchLean/NN/StateSpace/Scan.lean`. No TorchLeanTest test runs the layer.
-/

@[expose] public section

namespace TorchLean

open Dynamics

/-- Parameters for a diagonal S4-style sequence layer. -/
structure S4 (α : Type) [Storage α]
    (inputDim stateDim outputDim : Nat) where
  /-- Input projection from token features to the SSM state channels. -/
  inProj : Tensor α [inputDim, stateDim]
  /-- Output projection from the SSM state channels to output features. -/
  outProj : Tensor α [stateDim, outputDim]
  /-- Diagonal recurrent state-space core. -/
  ssm : DiagonalSSM α stateDim

namespace S4

variable {α : Type} [Storage α] [Add α] [Mul α] [Zero α]
variable {inputDim stateDim outputDim : Nat}

/-- One recurrent S4-style token step, returning `(new_state, output)`. -/
def step (m : S4 α inputDim stateDim outputDim)
    (h : Tensor α [stateDim])
    (x : Tensor α [inputDim]) :
    Tensor α [stateDim] × Tensor α [outputDim] :=
  let xState := Tensor.contract x m.inProj _ _ [stateDim] .vecMat
  let h' := m.ssm.step h xState
  (h', Tensor.contract (m.ssm.readout h' xState) m.outProj _ _ [outputDim] .vecMat)

/-- Run an array of tokens through the recurrent layer. -/
def runArray (m : S4 α inputDim stateDim outputDim)
    (h0 : Tensor α [stateDim])
    (xs : Array (Tensor α [inputDim])) :
    Tensor α [stateDim] × Array (Tensor α [outputDim]) :=
  StateSpace.scanArray m.step h0 xs

/-- An empty token sequence leaves the S4 hidden state untouched and emits nothing. -/
@[simp] theorem runArray_empty (m : S4 α inputDim stateDim outputDim)
    (h0 : Tensor α [stateDim]) :
    m.runArray h0 #[] = (h0, #[]) := by
  rfl

/-- A recurrent S4 pass emits one output token per input token. -/
@[simp] theorem runArray_outputs_size (m : S4 α inputDim stateDim outputDim)
    (h0 : Tensor α [stateDim])
    (xs : Array (Tensor α [inputDim])) :
    (m.runArray h0 xs).2.size = xs.size := by
  exact StateSpace.scanArray_outputs_size m.step h0 xs

end S4

end TorchLean
