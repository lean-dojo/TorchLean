/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Optimizer.Muon
public import TorchLean.Tensor.Tactic.Verify

/-!
# Muon Step Certificates

A backend's orthogonalization contract on the fresh momentum buffer gives a certificate for one
executable Muon update, covering the direction it uses, the optimizer state it stores and the
parameters it returns. The checked-backend theorems are the public proof interface, and the QR and
Newton-Schulz modules instantiate them for their own backends.
-/

@[expose] public section

namespace TorchLean

open TorchLean.Tensor

namespace Muon

open Optimizer (updateMomentumBuffer)

variable {α : Type} [Storage α] [Context α]

/--
Evidence that `direction` is the orthogonalizer's output on `buffer` and has column Gram matrix $I$.
-/
structure ExactCertifiedDirection {m n : Nat}
    (orthogonalizer : Orthogonalizer α (.dim m (.dim n .scalar)))
    (buffer direction : MatrixTensor α m n) : Prop where
  /-- The certified direction is the backend output. -/
  direction_eq : direction = orthogonalizer.apply buffer
  /-- The certified direction has exact column Gram $I$. -/
  hasExactColumnGram : HasExactColumnGram direction

/--
Evidence that `direction` is the orthogonalizer's output on `buffer` and that every entry of its
column-Gram residual is at most $\varepsilon$ in absolute value.
-/
structure ApproxCertifiedDirection {m n : Nat} (eps : α)
    (orthogonalizer : Orthogonalizer α (.dim m (.dim n .scalar)))
    (buffer direction : MatrixTensor α m n) : Prop where
  /-- The certified direction is the backend output. -/
  direction_eq : direction = orthogonalizer.apply buffer
  /-- The certified direction satisfies the requested residual bound. -/
  hasApproxColumnGram : HasApproxColumnGram eps direction

/--
Certificate for one exact Muon update: `direction` is the certified orthogonalization of the fresh
momentum buffer, the returned state stores that buffer, and the parameters move along `direction`.
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
  /-- The update subtracts `direction`, scaled by the learning rate, from the parameters. -/
  parameters_eq :
    (update state parameters gradients).parameters =
      parameters - Tensor.scale direction state.lr

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
  /-- The update subtracts `direction`, scaled by the learning rate, from the parameters. -/
  parameters_eq :
    (update state parameters gradients).parameters =
      parameters - Tensor.scale direction state.lr

/--
If the state's orthogonalizer exactly orthogonalizes the fresh momentum buffer, the Muon update has
an exact certified step.
-/
theorem exactCertifiedStep_of_buffer {m n : Nat}
    (state : State α (.dim m (.dim n .scalar)))
    (parameters gradients : MatrixTensor α m n)
    (horth : ExactOrthogonalizesBuffer state.orthogonalizer
      (update state parameters gradients).optimizerState.momentumBuffer) :
    ∃ direction : MatrixTensor α m n, ExactCertifiedStep state parameters gradients direction :=
  ⟨state.orthogonalizer.apply (update state parameters gradients).optimizerState.momentumBuffer,
    ⟨rfl, horth⟩, rfl, rfl⟩

/--
If the state's orthogonalizer meets the residual bound on the fresh momentum buffer, the Muon update
has an approximate certified step.
-/
theorem approxCertifiedStep_of_buffer {m n : Nat} {eps : α}
    (state : State α (.dim m (.dim n .scalar)))
    (parameters gradients : MatrixTensor α m n)
    (horth :
      ApproxOrthogonalizesBuffer eps state.orthogonalizer
        (update state parameters gradients).optimizerState.momentumBuffer) :
    ∃ direction : MatrixTensor α m n,
      ApproxCertifiedStep eps state parameters gradients direction :=
  ⟨state.orthogonalizer.apply (update state parameters gradients).optimizerState.momentumBuffer,
    ⟨rfl, horth⟩, rfl, rfl⟩

/--
If a checked exact backend's success predicate holds on the fresh momentum buffer, the Muon update
that uses this backend has an exact certified step.
-/
@[verify] theorem exactCertifiedStep_of_success {m n : Nat}
    (backend : CheckedExactOrthogonalizer α m n)
    (lr momentum : α) (momentumBuffer parameters gradients : MatrixTensor α m n)
    (hsuccess :
      backend.Success
        (update
          ({ lr := lr, momentum := momentum, momentumBuffer := momentumBuffer,
             orthogonalizer := backend.orthogonalizer } :
            State α (.dim m (.dim n .scalar)))
          parameters gradients).optimizerState.momentumBuffer) :
    ∃ direction : MatrixTensor α m n,
      ExactCertifiedStep
        ({ lr := lr, momentum := momentum, momentumBuffer := momentumBuffer,
           orthogonalizer := backend.orthogonalizer } :
          State α (.dim m (.dim n .scalar)))
        parameters gradients direction := by
  let state : State α (.dim m (.dim n .scalar)) :=
    { lr := lr, momentum := momentum, momentumBuffer := momentumBuffer,
      orthogonalizer := backend.orthogonalizer }
  exact exactCertifiedStep_of_buffer state parameters gradients
    (backend.certified (update state parameters gradients).optimizerState.momentumBuffer hsuccess)

/--
If a checked approximate backend's success predicate holds on the fresh momentum buffer, the Muon
update that uses this backend has a certified step with Gram residual at most $\varepsilon$.
-/
@[verify] theorem approxCertifiedStep_of_success {m n : Nat} {eps : α}
    (backend : CheckedApproxOrthogonalizer α m n eps)
    (lr momentum : α) (momentumBuffer parameters gradients : MatrixTensor α m n)
    (hsuccess :
      backend.Success
        (update
          ({ lr := lr, momentum := momentum, momentumBuffer := momentumBuffer,
             orthogonalizer := backend.orthogonalizer } :
            State α (.dim m (.dim n .scalar)))
          parameters gradients).optimizerState.momentumBuffer) :
    ∃ direction : MatrixTensor α m n,
      ApproxCertifiedStep eps
        ({ lr := lr, momentum := momentum, momentumBuffer := momentumBuffer,
           orthogonalizer := backend.orthogonalizer } :
          State α (.dim m (.dim n .scalar)))
        parameters gradients direction := by
  let state : State α (.dim m (.dim n .scalar)) :=
    { lr := lr, momentum := momentum, momentumBuffer := momentumBuffer,
      orthogonalizer := backend.orthogonalizer }
  exact approxCertifiedStep_of_buffer state parameters gradients
    (backend.certified (update state parameters gradients).optimizerState.momentumBuffer hsuccess)

/--
If its success predicate holds on the fresh momentum buffer, a checked exact backend gives
$Q^\mathsf{T}Q=I$ for the direction the update uses.
-/
theorem update_hasExactColumnGram {m n : Nat}
    (backend : CheckedExactOrthogonalizer α m n)
    (lr momentum : α) (momentumBuffer parameters gradients : MatrixTensor α m n)
    (hsuccess :
      backend.Success
        (update
          ({ lr := lr, momentum := momentum, momentumBuffer := momentumBuffer,
             orthogonalizer := backend.orthogonalizer } :
            State α (.dim m (.dim n .scalar)))
          parameters gradients).optimizerState.momentumBuffer) :
    HasExactColumnGram
      (backend.orthogonalizer.apply
        (update
          ({ lr := lr, momentum := momentum, momentumBuffer := momentumBuffer,
             orthogonalizer := backend.orthogonalizer } :
            State α (.dim m (.dim n .scalar)))
          parameters gradients).optimizerState.momentumBuffer) :=
  backend.certified _ hsuccess

/--
If its success predicate holds on the fresh momentum buffer, a checked approximate backend gives its
residual bound for the direction the update uses.
-/
theorem update_hasApproxColumnGram {m n : Nat} {eps : α}
    (backend : CheckedApproxOrthogonalizer α m n eps)
    (lr momentum : α) (momentumBuffer parameters gradients : MatrixTensor α m n)
    (hsuccess :
      backend.Success
        (update
          ({ lr := lr, momentum := momentum, momentumBuffer := momentumBuffer,
             orthogonalizer := backend.orthogonalizer } :
            State α (.dim m (.dim n .scalar)))
          parameters gradients).optimizerState.momentumBuffer) :
    HasApproxColumnGram eps
      (backend.orthogonalizer.apply
        (update
          ({ lr := lr, momentum := momentum, momentumBuffer := momentumBuffer,
             orthogonalizer := backend.orthogonalizer } :
            State α (.dim m (.dim n .scalar)))
          parameters gradients).optimizerState.momentumBuffer) :=
  backend.certified _ hsuccess

/-- Extract exact column orthogonality from a certified step. -/
theorem ExactCertifiedStep.hasExactColumnGram {m n : Nat}
    {state : State α (.dim m (.dim n .scalar))}
    {parameters gradients direction : MatrixTensor α m n}
    (cert : ExactCertifiedStep state parameters gradients direction) :
    HasExactColumnGram direction :=
  cert.direction_cert.hasExactColumnGram

/-- Extract the Gram-residual bound from an approximate certified step. -/
theorem ApproxCertifiedStep.hasApproxColumnGram {m n : Nat} {eps : α}
    {state : State α (.dim m (.dim n .scalar))}
    {parameters gradients direction : MatrixTensor α m n}
    (cert : ApproxCertifiedStep eps state parameters gradients direction) :
    HasApproxColumnGram eps direction :=
  cert.direction_cert.hasApproxColumnGram

end Muon

end TorchLean
