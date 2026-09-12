/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Tensor.Linalg
public import NN.Spec.Dynamics.StateSpace

/-!
# Diagonal S4-style state-space layer

This module provides TorchLean's diagonal recurrent SSM layer in the S4 family.  It exposes the
state-space recurrence used by S4-style models:

`h_{t+1} = A h_t + B x_t`,
`y_t     = C h_{t+1} + D x_t`.

The diagonal form is intentional: it shares the selective-scan core used by Mamba-style models,
admits direct recurrence proofs, and can be connected to convolutional S4 kernels through a
separate structured-kernel layer.

Reference: Gu, Goel, Ré. "Efficiently Modeling Long Sequences with Structured State Spaces",
ICLR 2022.

## Implementation status

No API builder implements this layer; the `nn.*` builders have no S4 layer. The relating theorem is
`diagonalS4_runArray_append_outputs_prefix` in `NN/MLTheory/Proofs/StateSpace/MambaCausality.lean`,
which proves that extending the input stream preserves earlier outputs; it rests on the scan
algebra in `NN/MLTheory/Proofs/StateSpace/Scan.lean`.
-/

@[expose] public section

namespace Models

open Spec TorchLean
open Spec.Dynamics

/-- Parameters for a diagonal S4-style sequence layer. -/
structure DiagonalS4Spec (α : Type) [TorchLean.Storage α]
    (inputDim stateDim outputDim : Nat) where
  /-- Input projection from token/features into SSM state channels. -/
  inProj : Tensor α [inputDim, stateDim]
  /-- Output projection from SSM channels to token/features. -/
  outProj : Tensor α [stateDim, outputDim]
  /-- Diagonal recurrent state-space core. -/
  ssm : DiagonalSSM α stateDim

namespace DiagonalS4Spec

variable {α : Type} [TorchLean.Storage α] [Add α] [Mul α] [Zero α]
variable {inputDim stateDim outputDim : Nat}

/-- Project an input token into state channels. -/
def projectInput (m : DiagonalS4Spec α inputDim stateDim outputDim)
    (x : Tensor α [inputDim]) : Tensor α [stateDim] :=
  vecMatMulSpec x m.inProj

/-- Project state channels to output channels. -/
def projectOutput (m : DiagonalS4Spec α inputDim stateDim outputDim)
    (h : Tensor α [stateDim]) : Tensor α [outputDim] :=
  vecMatMulSpec h m.outProj

/-- One recurrent S4-style token step, returning `(new_state, output)`. -/
def step (m : DiagonalS4Spec α inputDim stateDim outputDim)
    (h : Tensor α [stateDim])
    (x : Tensor α [inputDim]) :
    Tensor α [stateDim] × Tensor α [outputDim] :=
  let xState := m.projectInput x
  let h' := m.ssm.step h xState
  (h', m.projectOutput (m.ssm.readout h' xState))

/-- Run an array of tokens through the recurrent layer. -/
def runArray (m : DiagonalS4Spec α inputDim stateDim outputDim)
    (h0 : Tensor α [stateDim])
    (xs : Array (Tensor α [inputDim])) :
    Tensor α [stateDim] × Array (Tensor α [outputDim]) :=
  Spec.scanArray m.step h0 xs

/-- An empty token sequence leaves the S4 hidden state untouched and emits nothing. -/
@[simp] theorem runArray_empty (m : DiagonalS4Spec α inputDim stateDim outputDim)
    (h0 : Tensor α [stateDim]) :
    m.runArray h0 #[] = (h0, #[]) := by
  rfl

/-- A recurrent S4 pass emits one output token per input token. -/
@[simp] theorem runArray_outputs_size (m : DiagonalS4Spec α inputDim stateDim outputDim)
    (h0 : Tensor α [stateDim])
    (xs : Array (Tensor α [inputDim])) :
    (m.runArray h0 xs).2.size = xs.size := by
  exact Spec.scanArray_outputs_size m.step h0 xs

end DiagonalS4Spec

end Models
