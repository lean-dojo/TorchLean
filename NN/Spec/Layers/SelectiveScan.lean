/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Tensor
public import NN.Spec.Core.TensorOps

/-!
# Selective scan specs

This file contains the small proof-facing core behind state-space sequence models such as S4 and
Mamba.

The key observation, used by Mamba's hardware-aware parallel scan, is that each per-token recurrent
update can be viewed as an affine map

`h ↦ A_t h + b_t`.

Affine maps compose associatively.  A recurrent scan can therefore be implemented either by a
left-to-right recurrence or by a parallel prefix scan over affine summaries.  The scalar definitions
below are intentionally compact so that `NN/MLTheory/Proofs/StateSpace/Scan.lean` can prove the algebra
without depending on a particular runtime backend.  The diagonal tensor definitions are the direct
TorchLean spec analogue used by the model and CUDA contracts.

References:
- Gu, Goel, Ré. "Efficiently Modeling Long Sequences with Structured State Spaces" (S4), ICLR 2022.
- Gu, Dao. "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", COLM 2024.
- Dao, Gu. "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured
  State Space Duality" (Mamba-2), ICML 2024.
-/

@[expose] public section

namespace Spec

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

/-- Sequentially run a list of scalar affine transitions from an initial state. -/
def runScalarAffine {α : Type} [Mul α] [Add α] (h0 : α) : List (ScalarAffineTransition α) → α
  | [] => h0
  | tr :: rest => runScalarAffine (tr.apply h0) rest

/--
Summarize a transition list as one affine transition.

This is the algebraic payload used by parallel selective scan: prefix summaries can be produced by
any associative scan algorithm, and applying the summary to `h0` is equivalent to recurrence.
-/
def summarizeScalarAffine {α : Type} [Semiring α] : List (ScalarAffineTransition α) →
    ScalarAffineTransition α
  | [] => ScalarAffineTransition.id
  | tr :: rest => ScalarAffineTransition.compose (summarizeScalarAffine rest) tr

/-- Return every recurrent state after each scalar affine transition. -/
def scalarAffineScan {α : Type} [Mul α] [Add α] (h0 : α) :
    List (ScalarAffineTransition α) → List α
  | [] => []
  | tr :: rest =>
      let h1 := tr.apply h0
      h1 :: scalarAffineScan h1 rest

/-- A diagonal vector affine transition `h ↦ a ⊙ h + b`. -/
structure DiagonalTransition (α : Type) (stateDim : Nat) where
  /-- Elementwise recurrent multiplier. -/
  a : Tensor α (.dim stateDim .scalar)
  /-- Elementwise additive token contribution. -/
  b : Tensor α (.dim stateDim .scalar)

namespace DiagonalTransition

variable {α : Type} [Add α] [Mul α] {stateDim : Nat}

/-- Apply one diagonal affine state update. -/
def apply (tr : DiagonalTransition α stateDim)
    (h : Tensor α (.dim stateDim .scalar)) : Tensor α (.dim stateDim .scalar) :=
  tr.a * h + tr.b

/--
Compose diagonal affine transitions channelwise.

The order is the same as `ScalarAffineTransition.compose`: `compose t₂ t₁` is first `t₁`, then `t₂`.
-/
def compose (t₂ t₁ : DiagonalTransition α stateDim) : DiagonalTransition α stateDim :=
  { a := t₂.a * t₁.a
    b := t₂.a * t₁.b + t₂.b }

end DiagonalTransition

/-- Sequentially run diagonal transitions and return the final state. -/
def runDiagonalTransitions {α : Type} [Add α] [Mul α] {stateDim : Nat}
    (h0 : Tensor α (.dim stateDim .scalar)) : List (DiagonalTransition α stateDim) →
    Tensor α (.dim stateDim .scalar)
  | [] => h0
  | tr :: rest => runDiagonalTransitions (tr.apply h0) rest

/-- Return every hidden state from a diagonal selective scan. -/
def diagonalSelectiveScan {α : Type} [Add α] [Mul α] {stateDim : Nat}
    (h0 : Tensor α (.dim stateDim .scalar)) :
    List (DiagonalTransition α stateDim) → List (Tensor α (.dim stateDim .scalar))
  | [] => []
  | tr :: rest =>
      let h1 := tr.apply h0
      h1 :: diagonalSelectiveScan h1 rest

end Spec
