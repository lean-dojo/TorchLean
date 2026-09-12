/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.TensorOps
public import NN.Spec.Core.Tensor -- shake: keep

/-!
# Selective scan specs

This file contains the small proof layer core behind state-space sequence models such as S4 and
Mamba.

The key observation, used by Mamba's hardware-aware parallel scan, is that each per-token recurrent
update can be viewed as an affine map

`h ↦ A_t h + b_t`.

Affine maps compose associatively.  A recurrent scan can therefore be implemented either by a
left-to-right recurrence or by a parallel prefix scan over affine summaries.  The scalar definitions
below are kept compact so that `NN/MLTheory/Proofs/StateSpace/Scan.lean` can prove the algebra
without depending on a particular runtime backend.  The diagonal tensor definitions are the direct
TorchLean spec analogue used by the model and CUDA contracts.

References:
- Gu, Goel, Ré. "Efficiently Modeling Long Sequences with Structured State Spaces" (S4), ICLR 2022.
- Gu, Dao. "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", COLM 2024.
- Dao, Gu. "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured
  State Space Duality" (Mamba-2), ICML 2024.
-/

@[expose] public section

open TorchLean

namespace Spec

/--
Run a stateful step over an array, appending each emitted value to `initialOutputs`.

The initial output buffer makes this suitable for chunked execution without changing the state
transition being specified.
-/
def scanArrayFrom {State Input Output : Type}
    (step : State → Input → State × Output) (initial : State) (initialOutputs : Array Output)
    (xs : Array Input) : State × Array Output :=
  xs.foldl
    (fun (state, outputs) input =>
      let (nextState, output) := step state input
      (nextState, outputs.push output))
    (initial, initialOutputs)

/-- Run a stateful step over an array and return the final state and emitted values. -/
def scanArray {State Input Output : Type}
    (step : State → Input → State × Output) (initial : State) (xs : Array Input) :
    State × Array Output :=
  scanArrayFrom step initial #[] xs

/-- Scanning an empty array returns the initial state and no outputs. -/
@[simp] theorem scanArray_empty {State Input Output : Type}
    (step : State → Input → State × Output) (initial : State) :
    scanArray step initial (#[] : Array Input) = (initial, (#[] : Array Output)) := by
  rfl

/-- A stateful scan emits exactly one value for each input. -/
@[simp] theorem scanArray_outputs_size {State Input Output : Type}
    (step : State → Input → State × Output) (initial : State) (xs : Array Input) :
    (scanArray step initial xs).2.size = xs.size := by
  unfold scanArray scanArrayFrom
  let f : (State × Array Output) → Input → State × Array Output :=
    fun (state, outputs) input =>
      let (nextState, output) := step state input
      (nextState, outputs.push output)
  change (Array.foldl f (initial, #[]) xs).2.size = xs.size
  refine Array.foldl_induction (as := xs) (init := (initial, #[])) (f := f)
    (motive := fun i result => result.2.size = i) ?_ ?_
  · rfl
  · intro i result h
    rcases step result.1 xs[i] with ⟨nextState, output⟩
    simp [f, h]

/-- A scalar affine transition `h ↦ a*h + b`. -/
structure ScalarAffineTransition (α : Type) where
  /-- Linear multiplier. In diagonal SSMs this is one channel of the discretized state matrix. -/
  a : α
  /-- Additive input contribution for the current token. -/
  b : α
deriving Repr

namespace ScalarAffineTransition

variable {α : Type}

/-- Apply a scalar affine transition. -/
def apply [Mul α] [Add α] (tr : ScalarAffineTransition α) (h : α) : α :=
  tr.a * h + tr.b

/-- Identity affine transition. -/
def id [One α] [Zero α] : ScalarAffineTransition α :=
  { a := 1, b := 0 }

/--
Compose two affine transitions.

`compose t₂ t₁` means "first apply `t₁`, then apply `t₂`".
-/
def compose [Mul α] [Add α] (t₂ t₁ : ScalarAffineTransition α) :
    ScalarAffineTransition α :=
  { a := t₂.a * t₁.a
    b := t₂.a * t₁.b + t₂.b }

end ScalarAffineTransition

/-- Sequentially run scalar affine transitions from an initial state. -/
def runScalarAffine {α : Type} [Mul α] [Add α] (h0 : α)
    (transitions : Array (ScalarAffineTransition α)) : α :=
  (scanArray (fun state transition =>
    let nextState := transition.apply state
    (nextState, nextState)) h0 transitions).1

/--
Summarize an array of transitions as one affine transition.

This is the algebraic payload used by parallel selective scan: prefix summaries can be produced by
any associative scan algorithm, and applying the summary to `h0` is equivalent to recurrence.
-/
def summarizeScalarAffine {α : Type} [Semiring α]
    (transitions : Array (ScalarAffineTransition α)) : ScalarAffineTransition α :=
  transitions.foldr
    (fun transition summary => ScalarAffineTransition.compose summary transition)
    ScalarAffineTransition.id

/-- Return every recurrent state after each scalar affine transition. -/
def scalarAffineScan {α : Type} [Mul α] [Add α] (h0 : α) :
    Array (ScalarAffineTransition α) → Array α :=
  fun transitions => (scanArray (fun state transition =>
    let nextState := transition.apply state
    (nextState, nextState)) h0 transitions).2

/-- The scalar affine scan has one state per transition. -/
@[simp] theorem scalarAffineScan_size {α : Type} [Mul α] [Add α] (h0 : α)
    (transitions : Array (ScalarAffineTransition α)) :
    (scalarAffineScan h0 transitions).size = transitions.size := by
  exact scanArray_outputs_size _ h0 transitions

/-- A diagonal vector affine transition `h ↦ a ⊙ h + b`. -/
structure DiagonalTransition (α : Type) [TorchLean.Storage α] (stateDim : Nat) where
  /-- Elementwise recurrent multiplier. -/
  a : Tensor α [stateDim]
  /-- Elementwise additive token contribution. -/
  b : Tensor α [stateDim]

namespace DiagonalTransition

variable {α : Type} [TorchLean.Storage α] [Add α] [Mul α] {stateDim : Nat}

/-- Apply one diagonal affine state update. -/
def apply (tr : DiagonalTransition α stateDim)
    (h : Tensor α [stateDim]) : Tensor α [stateDim] :=
  Tensor.addSpec (Tensor.mulSpec tr.a h) tr.b

/--
Compose diagonal affine transitions channelwise.

The order is the same as `ScalarAffineTransition.compose`: `compose t₂ t₁` is first `t₁`, then `t₂`.
-/
def compose (t₂ t₁ : DiagonalTransition α stateDim) : DiagonalTransition α stateDim :=
  { a := Tensor.mulSpec t₂.a t₁.a
    b := Tensor.addSpec (Tensor.mulSpec t₂.a t₁.b) t₂.b }

end DiagonalTransition

/-- Sequentially run diagonal transitions and return the final state. -/
def runDiagonalTransitions {α : Type} [TorchLean.Storage α]
    [Add α] [Mul α] {stateDim : Nat}
    (h0 : Tensor α [stateDim])
    (transitions : Array (DiagonalTransition α stateDim)) :
    Tensor α [stateDim] :=
  (scanArray (fun state transition =>
    let nextState := transition.apply state
    (nextState, nextState)) h0 transitions).1

/-- Return every hidden state from a diagonal selective scan. -/
def diagonalSelectiveScan {α : Type} [TorchLean.Storage α]
    [Add α] [Mul α] {stateDim : Nat}
    (h0 : Tensor α [stateDim]) :
    Array (DiagonalTransition α stateDim) → Array (Tensor α [stateDim]) :=
  fun transitions => (scanArray (fun state transition =>
    let nextState := transition.apply state
    (nextState, nextState)) h0 transitions).2

/-- The diagonal selective scan has one state per transition. -/
@[simp] theorem diagonalSelectiveScan_size {α : Type} [TorchLean.Storage α]
    [Add α] [Mul α] {stateDim : Nat}
    (h0 : Tensor α [stateDim])
    (transitions : Array (DiagonalTransition α stateDim)) :
    (diagonalSelectiveScan h0 transitions).size = transitions.size := by
  exact scanArray_outputs_size _ h0 transitions

end Spec
