/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Analysis.SpecialFunctions.Pow.NNReal
public import Mathlib.Data.Sym.Sym2.Init
import Mathlib.Tactic.NormNum.GCD
public import NN.Runtime.Optim.Optimizers
public import NN.Spec.Core.TensorReductionShape.LinearAlgebra

/-!
# Muon Orthogonalization Contracts

Muon is more than momentum with a different name: after the momentum buffer is updated, the
parameter direction is supposed to be an orthogonalized matrix direction.  The runtime optimizer
therefore takes an explicit orthogonalizer backend, while this file records the proof contract that
such a backend should satisfy.

There are two useful levels:

* `ExactMatrixOrthogonalizer` says the backend returns a matrix $Q$ with $Q^\mathsf{T}Q=I$.
* `ApproxMatrixOrthogonalizer eps` says the Gram residual $Q^\mathsf{T}Q-I$ is entrywise bounded
  by $\varepsilon$.

The theorems below connect those contracts to the executable Muon step: if the backend satisfies one
of these contracts, the direction used in the parameter update is certified at the same level.
-/

@[expose] public section

namespace Optim

open Spec TorchLean
open TorchLean TorchLean.Tensor

namespace Muon

variable {α : Type} [TorchLean.Storage α] [Context α]

/-- A matrix-shaped TorchLean tensor. -/
abbrev MatrixTensor (α : Type) [TorchLean.Storage α] (m n : Nat) :=
  Tensor α [m, n]

/-- The column Gram matrix $Q^\mathsf{T}Q$. -/
def columnGram {m n : Nat} (Q : MatrixTensor α m n) :
    MatrixTensor α n n :=
  matMulSpec (TorchLean.Tensor.swapAdjacentAxes Q 0) Q

/-- Exact column orthogonality for a matrix-shaped update direction. -/
def HasExactColumnGram {m n : Nat} (Q : MatrixTensor α m n) : Prop :=
  columnGram Q = identityTensorSpec n

/-- A backend exactly orthogonalizes one specified momentum buffer. -/
def ExactOrthogonalizesBuffer {m n : Nat}
    (orthogonalizer : Orthogonalizer α (.dim m (.dim n .scalar)))
    (buffer : MatrixTensor α m n) : Prop :=
  HasExactColumnGram (orthogonalizer.apply buffer)

/-- Residual matrix $Q^\mathsf{T}Q-I$, used for approximate orthogonalization certificates. -/
def columnGramResidual {m n : Nat} (Q : MatrixTensor α m n) :
    MatrixTensor α n n :=
  subSpec (columnGram Q) (identityTensorSpec n)

/--
Entrywise approximate column orthogonality.

For an exact backend use `HasExactColumnGram`. For Newton-Schulz or CUDA implementations, this is
the certificate shape we want the backend to establish or export: every entry of
$Q^\mathsf{T}Q-I$ is bounded by $\varepsilon$.
-/
def HasApproxColumnGram {m n : Nat} (eps : α) (Q : MatrixTensor α m n) : Prop :=
  ∀ i : Fin n, ∀ j : Fin n,
    MathFunctions.abs (get2 (columnGramResidual Q) i j) ≤ eps

/--
A backend approximately orthogonalizes one specified momentum buffer.

This is also the lightest sound checker boundary we can put on an approximate backend: after it
returns a direction, prove or check that the direction's Gram residual is bounded by
$\varepsilon$, and nothing more. `residualCheckedApproxOrthogonalizer` uses it exactly that way.
-/
def ApproxOrthogonalizesBuffer {m n : Nat} (eps : α)
    (orthogonalizer : Orthogonalizer α (.dim m (.dim n .scalar)))
    (buffer : MatrixTensor α m n) : Prop :=
  HasApproxColumnGram eps (orthogonalizer.apply buffer)

/--
An exact matrix Muon orthogonalizer maps every momentum buffer to a direction whose columns have
Gram matrix $I$.
-/
def ExactMatrixOrthogonalizer {m n : Nat}
    (orthogonalizer : Orthogonalizer α (.dim m (.dim n .scalar))) : Prop :=
  ∀ buffer : MatrixTensor α m n, HasExactColumnGram (orthogonalizer.apply buffer)

/--
An approximate matrix Muon orthogonalizer maps every momentum buffer to a direction whose Gram
residual is entrywise bounded by $\varepsilon$.
-/
def ApproxMatrixOrthogonalizer {m n : Nat} (eps : α)
    (orthogonalizer : Orthogonalizer α (.dim m (.dim n .scalar))) : Prop :=
  ∀ buffer : MatrixTensor α m n, HasApproxColumnGram eps (orthogonalizer.apply buffer)

/-! ## Certified backend records -/

/--
Unconditionally certified exact Muon backend.

Use this when the orthogonalizer is known to return an exact $Q^\mathsf{T}Q=I$ direction for every
buffer of a fixed matrix shape.
-/
structure ExactCertifiedOrthogonalizer (α : Type) [TorchLean.Storage α] [Context α]
    (m n : Nat) where
  /-- Executable orthogonalizer used by Muon. -/
  orthogonalizer : Orthogonalizer α (.dim m (.dim n .scalar))
  /-- The backend returns an exactly orthogonalized direction for every buffer. -/
  certified : ExactMatrixOrthogonalizer orthogonalizer

/--
Unconditionally certified approximate Muon backend.

Use this when the orthogonalizer is known to return a direction whose Gram residual is bounded by
$\varepsilon$ for every buffer of a fixed matrix shape.
-/
structure ApproxCertifiedOrthogonalizer (α : Type) [TorchLean.Storage α] [Context α]
    (m n : Nat) (eps : α) where
  /-- Executable orthogonalizer used by Muon. -/
  orthogonalizer : Orthogonalizer α (.dim m (.dim n .scalar))
  /-- The backend returns an approximately orthogonalized direction for every buffer. -/
  certified : ApproxMatrixOrthogonalizer eps orthogonalizer

/--
Checked exact Muon backend with a per-buffer success predicate.

This is the practical interface for algorithms whose correctness has preconditions on the matrix
being orthogonalized. The QR backend below is the first instance: its success predicate is positive
executable `R` pivots.
-/
structure CheckedExactOrthogonalizer (α : Type) [TorchLean.Storage α] [Context α]
    (m n : Nat) where
  /-- Executable orthogonalizer used by Muon. -/
  orthogonalizer : Orthogonalizer α (.dim m (.dim n .scalar))
  /-- Backend-specific success/certification condition for an input buffer. -/
  Success : MatrixTensor α m n → Prop
  /-- Whenever `Success buffer` holds, the backend exactly orthogonalizes that buffer. -/
  certified : ∀ buffer : MatrixTensor α m n,
    Success buffer → ExactOrthogonalizesBuffer orthogonalizer buffer

/--
Checked approximate Muon backend with a per-buffer success predicate.

This is the intended proof shape for Newton-Schulz/CUDA-style backends: the kernel may be fast and
approximate, but the exported proof or checker must establish `Success buffer`, which then gives the
entrywise $Q^\mathsf{T}Q-I$ bound.
-/
structure CheckedApproxOrthogonalizer (α : Type) [TorchLean.Storage α] [Context α]
    (m n : Nat) (eps : α) where
  /-- Executable orthogonalizer used by Muon. -/
  orthogonalizer : Orthogonalizer α (.dim m (.dim n .scalar))
  /-- Backend-specific success/certification condition for an input buffer. -/
  Success : MatrixTensor α m n → Prop
  /-- Whenever `Success buffer` holds, the backend approximately orthogonalizes that buffer. -/
  certified : ∀ buffer : MatrixTensor α m n,
    Success buffer → ApproxOrthogonalizesBuffer eps orthogonalizer buffer

end Muon

end Optim
