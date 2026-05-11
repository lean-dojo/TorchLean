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
-/

@[expose] public section

namespace Models

open Spec
open NN.Spec.Dynamics

/-- Parameters for a diagonal S4-style sequence layer. -/
structure DiagonalS4Spec (α : Type) (inputDim stateDim outputDim : Nat) where
  /-- Input projection from token/features into SSM state channels. -/
  inProj : Tensor α (.dim inputDim (.dim stateDim .scalar))
  /-- Output projection from SSM channels to token/features. -/
  outProj : Tensor α (.dim stateDim (.dim outputDim .scalar))
  /-- Diagonal recurrent state-space core. -/
  ssm : DiagonalSSM α stateDim

namespace DiagonalS4Spec

variable {α : Type} [Add α] [Mul α] [Zero α]
variable {inputDim stateDim outputDim : Nat}

/-- Project an input token into state channels. -/
def projectInput (m : DiagonalS4Spec α inputDim stateDim outputDim)
    (x : Tensor α (.dim inputDim .scalar)) : Tensor α (.dim stateDim .scalar) :=
  vecMatMulSpec x m.inProj

/-- Project state channels to output channels. -/
def projectOutput (m : DiagonalS4Spec α inputDim stateDim outputDim)
    (h : Tensor α (.dim stateDim .scalar)) : Tensor α (.dim outputDim .scalar) :=
  vecMatMulSpec h m.outProj

/-- One recurrent S4-style token step, returning `(new_state, output)`. -/
def step (m : DiagonalS4Spec α inputDim stateDim outputDim)
    (h : Tensor α (.dim stateDim .scalar))
    (x : Tensor α (.dim inputDim .scalar)) :
    Tensor α (.dim stateDim .scalar) × Tensor α (.dim outputDim .scalar) :=
  let xState := m.projectInput x
  let h' := m.ssm.step h xState
  (h', m.projectOutput (m.ssm.readout h' xState))

/-- Run a list of tokens through the recurrent layer. -/
def runList (m : DiagonalS4Spec α inputDim stateDim outputDim)
    (h0 : Tensor α (.dim stateDim .scalar)) :
    List (Tensor α (.dim inputDim .scalar)) →
    Tensor α (.dim stateDim .scalar) × List (Tensor α (.dim outputDim .scalar))
  | [] => (h0, [])
  | x :: xs =>
      let (h1, y) := m.step h0 x
      let (hN, ys) := m.runList h1 xs
      (hN, y :: ys)

@[simp] theorem runList_nil (m : DiagonalS4Spec α inputDim stateDim outputDim)
    (h0 : Tensor α (.dim stateDim .scalar)) :
    m.runList h0 [] = (h0, []) := by
  rfl

@[simp] theorem runList_cons (m : DiagonalS4Spec α inputDim stateDim outputDim)
    (h0 : Tensor α (.dim stateDim .scalar))
    (x : Tensor α (.dim inputDim .scalar))
    (xs : List (Tensor α (.dim inputDim .scalar))) :
    m.runList h0 (x :: xs) =
      let (h1, y) := m.step h0 x
      let (hN, ys) := m.runList h1 xs
      (hN, y :: ys) := by
  rfl

/-- A recurrent S4 pass emits one output token per input token. -/
theorem runList_outputs_length (m : DiagonalS4Spec α inputDim stateDim outputDim)
    (h0 : Tensor α (.dim stateDim .scalar))
    (xs : List (Tensor α (.dim inputDim .scalar))) :
    (m.runList h0 xs).2.length = xs.length := by
  induction xs generalizing h0 with
  | nil =>
      simp
  | cons x rest ih =>
      simp [runList_cons, ih]

end DiagonalS4Spec

end Models
