/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.Optimization.Muon.Core
public import NN.Tactic.Verify

/-!
# Muon Step Certificates

This module connects a backend's orthogonalization contract to the direction, state, and parameter
values produced by one executable Muon update. The generic checked-backend theorems are the public
proof interface; concrete QR and Newton-Schulz backends instantiate them in their own modules.
-/

@[expose] public section

namespace Optim

open Spec TorchLean
open TorchLean TorchLean.Tensor

namespace Muon

variable {α : Type} [TorchLean.Storage α] [Context α]

/--
Evidence that `direction` is exactly the output of an orthogonalizer on `buffer` and has column
Gram matrix $I$.
-/
structure ExactCertifiedDirection {m n : Nat}
    (orthogonalizer : Orthogonalizer α (.dim m (.dim n .scalar)))
    (buffer direction : MatrixTensor α m n) : Prop where
  /-- The certified direction is the backend output. -/
  direction_eq : direction = orthogonalizer.apply buffer
  /-- The certified direction has exact column Gram $I$. -/
  exact_column_gram : HasExactColumnGram direction

/--
Evidence that `direction` is exactly the output of an orthogonalizer on `buffer` and has an
entrywise column-Gram residual bounded by $\varepsilon$.
-/
structure ApproxCertifiedDirection {m n : Nat} (eps : α)
    (orthogonalizer : Orthogonalizer α (.dim m (.dim n .scalar)))
    (buffer direction : MatrixTensor α m n) : Prop where
  /-- The certified direction is the backend output. -/
  direction_eq : direction = orthogonalizer.apply buffer
  /-- The certified direction satisfies the requested residual bound. -/
  approx_column_gram : HasApproxColumnGram eps direction

/--
Certificate for one exact Muon update: the fresh momentum buffer is orthogonalized, the new state
stores that buffer, and the parameters move along the certified direction.
-/
structure ExactCertifiedStep {m n : Nat}
    (state : State α (.dim m (.dim n .scalar)))
    (parameters gradients direction : MatrixTensor α m n) : Prop where
  /-- Certificate for the direction computed from the fresh momentum buffer. -/
  direction_cert :
    ExactCertifiedDirection state.orthogonalizer
      (update state parameters gradients).optimizerState.momentumBuffer direction
  /-- Muon changes only the momentum buffer in its optimizer state. -/
  state_eq :
    (update state parameters gradients).optimizerState =
      { state with
          momentumBuffer := updateMomentumBuffer state.momentumBuffer state.momentum gradients }
  /-- Parameter equation for the certified update direction. -/
  parameters_eq :
    (update state parameters gradients).parameters =
      subSpec parameters (scaleSpec direction state.learningRate)

/-- The residual-bounded counterpart of `ExactCertifiedStep`. -/
structure ApproxCertifiedStep {m n : Nat} (eps : α)
    (state : State α (.dim m (.dim n .scalar)))
    (parameters gradients direction : MatrixTensor α m n) : Prop where
  /-- Certificate for the direction computed from the fresh momentum buffer. -/
  direction_cert :
    ApproxCertifiedDirection eps state.orthogonalizer
      (update state parameters gradients).optimizerState.momentumBuffer direction
  /-- Muon changes only the momentum buffer in its optimizer state. -/
  state_eq :
    (update state parameters gradients).optimizerState =
      { state with
          momentumBuffer := updateMomentumBuffer state.momentumBuffer state.momentum gradients }
  /-- Parameter equation for the certified update direction. -/
  parameters_eq :
    (update state parameters gradients).parameters =
      subSpec parameters (scaleSpec direction state.learningRate)

/-- A local exact backend fact for the fresh buffer produces a certified Muon step. -/
theorem exactCertifiedStep_of_buffer {m n : Nat}
    (state : State α (.dim m (.dim n .scalar)))
    (parameters gradients : MatrixTensor α m n)
    (horth : ExactOrthogonalizesBuffer state.orthogonalizer
      (update state parameters gradients).optimizerState.momentumBuffer) :
    ∃ direction : MatrixTensor α m n, ExactCertifiedStep state parameters gradients direction := by
  let direction :=
    state.orthogonalizer.apply (update state parameters gradients).optimizerState.momentumBuffer
  refine ⟨direction, ⟨⟨rfl, horth⟩, rfl, ?_⟩⟩
  rfl

/-- A local residual bound for the fresh buffer produces a certified Muon step. -/
theorem approxCertifiedStep_of_buffer {m n : Nat} {eps : α}
    (state : State α (.dim m (.dim n .scalar)))
    (parameters gradients : MatrixTensor α m n)
    (horth :
      ApproxOrthogonalizesBuffer eps state.orthogonalizer
        (update state parameters gradients).optimizerState.momentumBuffer) :
    ∃ direction : MatrixTensor α m n,
      ApproxCertifiedStep eps state parameters gradients direction := by
  let direction :=
    state.orthogonalizer.apply (update state parameters gradients).optimizerState.momentumBuffer
  refine ⟨direction, ⟨⟨rfl, horth⟩, rfl, ?_⟩⟩
  rfl

/--
A checked exact backend certifies the concrete direction and equations of one Muon update whenever
its success predicate holds on the fresh momentum buffer.
-/
@[verify] theorem exactCertifiedStep_of_checkedBackend {m n : Nat}
    (backend : CheckedExactOrthogonalizer α m n)
    (learningRate momentum : α) (momentumBuffer parameters gradients : MatrixTensor α m n)
    (hsuccess :
      backend.Success
        (update
          ({ learningRate := learningRate, momentum := momentum, momentumBuffer := momentumBuffer,
             orthogonalizer := backend.orthogonalizer } :
            State α (.dim m (.dim n .scalar)))
          parameters gradients).optimizerState.momentumBuffer) :
    ∃ direction : MatrixTensor α m n,
      ExactCertifiedStep
        ({ learningRate := learningRate, momentum := momentum, momentumBuffer := momentumBuffer,
           orthogonalizer := backend.orthogonalizer } :
          State α (.dim m (.dim n .scalar)))
        parameters gradients direction := by
  let state : State α (.dim m (.dim n .scalar)) :=
    { learningRate := learningRate, momentum := momentum, momentumBuffer := momentumBuffer,
      orthogonalizer := backend.orthogonalizer }
  exact exactCertifiedStep_of_buffer state parameters gradients
    (backend.certified (update state parameters gradients).optimizerState.momentumBuffer hsuccess)

/--
A checked approximate backend certifies one Muon update whenever its success predicate establishes
the requested Gram-residual bound on the fresh momentum buffer.
-/
@[verify] theorem approxCertifiedStep_of_checkedBackend {m n : Nat} {eps : α}
    (backend : CheckedApproxOrthogonalizer α m n eps)
    (learningRate momentum : α) (momentumBuffer parameters gradients : MatrixTensor α m n)
    (hsuccess :
      backend.Success
        (update
          ({ learningRate := learningRate, momentum := momentum, momentumBuffer := momentumBuffer,
             orthogonalizer := backend.orthogonalizer } :
            State α (.dim m (.dim n .scalar)))
          parameters gradients).optimizerState.momentumBuffer) :
    ∃ direction : MatrixTensor α m n,
      ApproxCertifiedStep eps
        ({ learningRate := learningRate, momentum := momentum, momentumBuffer := momentumBuffer,
           orthogonalizer := backend.orthogonalizer } :
          State α (.dim m (.dim n .scalar)))
        parameters gradients direction := by
  let state : State α (.dim m (.dim n .scalar)) :=
    { learningRate := learningRate, momentum := momentum, momentumBuffer := momentumBuffer,
      orthogonalizer := backend.orthogonalizer }
  exact approxCertifiedStep_of_buffer state parameters gradients
    (backend.certified (update state parameters gradients).optimizerState.momentumBuffer hsuccess)

/-- A checked exact backend gives $Q^\mathsf{T}Q=I$ for the direction used by an update. -/
theorem checkedBackend_updateDirection_hasExactColumnGram {m n : Nat}
    (backend : CheckedExactOrthogonalizer α m n)
    (learningRate momentum : α) (momentumBuffer parameters gradients : MatrixTensor α m n)
    (hsuccess :
      backend.Success
        (update
          ({ learningRate := learningRate, momentum := momentum, momentumBuffer := momentumBuffer,
             orthogonalizer := backend.orthogonalizer } :
            State α (.dim m (.dim n .scalar)))
          parameters gradients).optimizerState.momentumBuffer) :
    HasExactColumnGram
      (backend.orthogonalizer.apply
        (update
          ({ learningRate := learningRate, momentum := momentum, momentumBuffer := momentumBuffer,
             orthogonalizer := backend.orthogonalizer } :
            State α (.dim m (.dim n .scalar)))
          parameters gradients).optimizerState.momentumBuffer) := by
  exact backend.certified _ hsuccess

/-- A checked approximate backend gives its residual bound for the direction used by an update. -/
theorem checkedBackend_updateDirection_hasApproxColumnGram {m n : Nat} {eps : α}
    (backend : CheckedApproxOrthogonalizer α m n eps)
    (learningRate momentum : α) (momentumBuffer parameters gradients : MatrixTensor α m n)
    (hsuccess :
      backend.Success
        (update
          ({ learningRate := learningRate, momentum := momentum, momentumBuffer := momentumBuffer,
             orthogonalizer := backend.orthogonalizer } :
            State α (.dim m (.dim n .scalar)))
          parameters gradients).optimizerState.momentumBuffer) :
    HasApproxColumnGram eps
      (backend.orthogonalizer.apply
        (update
          ({ learningRate := learningRate, momentum := momentum, momentumBuffer := momentumBuffer,
             orthogonalizer := backend.orthogonalizer } :
            State α (.dim m (.dim n .scalar)))
          parameters gradients).optimizerState.momentumBuffer) := by
  exact backend.certified _ hsuccess

/-- Extract exact column orthogonality from a certified step. -/
theorem ExactCertifiedStep.hasExactColumnGram {m n : Nat}
    {state : State α (.dim m (.dim n .scalar))}
    {parameters gradients direction : MatrixTensor α m n}
    (cert : ExactCertifiedStep state parameters gradients direction) :
    HasExactColumnGram direction :=
  cert.direction_cert.exact_column_gram

/-- Extract the Gram-residual bound from an approximate certified step. -/
theorem ApproxCertifiedStep.hasApproxColumnGram {m n : Nat} {eps : α}
    {state : State α (.dim m (.dim n .scalar))}
    {parameters gradients direction : MatrixTensor α m n}
    (cert : ApproxCertifiedStep eps state parameters gradients direction) :
    HasApproxColumnGram eps direction :=
  cert.direction_cert.approx_column_gram

end Muon

end Optim
