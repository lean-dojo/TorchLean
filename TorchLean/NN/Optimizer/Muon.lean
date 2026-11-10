/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Optimizer.Update
public import TorchLean.Tensor.AdjacentSwap
public import TorchLean.Tensor.Contract

/-!
# Muon Orthogonalization Contracts

Muon updates a momentum buffer and then moves the parameters along an orthogonalized version of
that buffer. The runtime optimizer takes the orthogonalizer as a backend argument, and this module
states what a backend has to deliver on one momentum buffer, at two levels:

* `ExactOrthogonalizesBuffer` says the backend returns a matrix $Q$ with $Q^\mathsf{T}Q=I$.
* `ApproxOrthogonalizesBuffer eps` says the Gram residual $Q^\mathsf{T}Q-I$ is entrywise bounded
  by $\varepsilon$.

`CheckedExactOrthogonalizer` and `CheckedApproxOrthogonalizer` package a backend with the
precondition under which it meets one of them.

`TorchLean.NN.Optimizer.Muon.Certificates` connects those contracts to the executable
Muon step: if the backend satisfies one of them on the fresh momentum buffer, the direction used in
the parameter update is certified at the same level. Concrete backends live in
`TorchLean.NN.Optimizer.Muon.QR` (exact, under positive QR pivots) and
`TorchLean.NN.Optimizer.Muon.NewtonSchulz` (residual-checked, or exact at a fixed
point).

We wrote these contracts by hand. Jordan et al. (2024), "Muon: An optimizer for hidden layers in
neural networks", https://kellerjordan.github.io/posts/muon/, define the ideal direction
$\mathrm{Ortho}(G)$ as the matrix nearest $G$ in Frobenius norm with orthonormal rows or columns,
which is $UV^\mathsf{T}$ for the SVD $USV^\mathsf{T}$. Ours ask less, since any $Q$ with
orthonormal columns passes. They are also column-oriented, so over ℝ a wide buffer ($m<n$) never
meets the exact contract: there $Q^\mathsf{T}Q$ has rank at most $m$.
-/

@[expose] public section

open FloatLib.Numerics (MathFunctions)

namespace TorchLean

open Tensor

namespace Muon

variable {α : Type} [Storage α] [Context α]

/-- A matrix-shaped TorchLean tensor. -/
abbrev MatrixTensor (α : Type) [Storage α] (m n : Nat) :=
  Tensor α [m, n]

/-- The column Gram matrix $Q^\mathsf{T}Q$. -/
def columnGram {m n : Nat} (Q : MatrixTensor α m n) :
    MatrixTensor α n n :=
  Tensor.matmul (Tensor.permute Q (Shape.adjacentSwap [m, n] 0) [n, m]) Q

/-- Exact column orthogonality of a matrix-shaped update direction: $Q^\mathsf{T}Q=I$. -/
def HasExactColumnGram {m n : Nat} (Q : MatrixTensor α m n) : Prop :=
  columnGram Q = Tensor.identity n

/-- A backend exactly orthogonalizes one momentum buffer: its output $Q$ has $Q^\mathsf{T}Q=I$. -/
def ExactOrthogonalizesBuffer {m n : Nat}
    (orthogonalizer : Orthogonalizer α (.dim m (.dim n .scalar)))
    (buffer : MatrixTensor α m n) : Prop :=
  HasExactColumnGram (orthogonalizer.apply buffer)

/-- Residual matrix $Q^\mathsf{T}Q-I$, used for approximate orthogonalization certificates. -/
def columnGramResidual {m n : Nat} (Q : MatrixTensor α m n) :
    MatrixTensor α n n :=
  columnGram Q - Tensor.identity n

/--
Entrywise approximate column orthogonality: every entry of $Q^\mathsf{T}Q-I$ has absolute value at
most $\varepsilon$.

An approximate backend, such as the residual-checked Newton-Schulz iteration of
`TorchLean.NN.Optimizer.Muon.NewtonSchulz`, certifies a statement of this shape. An
exact backend uses `HasExactColumnGram`.
-/
def HasApproxColumnGram {m n : Nat} (eps : α) (Q : MatrixTensor α m n) : Prop :=
  ∀ i : Fin n, ∀ j : Fin n,
    MathFunctions.abs (get2 (columnGramResidual Q) i j) ≤ eps

/--
A backend approximately orthogonalizes a given momentum buffer: the Gram residual of its output is
entrywise bounded by $\varepsilon$.

The predicate looks only at the returned direction, which makes it the lightest sound check we can
put on an approximate backend. `Orthogonalizer.residualChecked` in
`TorchLean.NN.Optimizer.Muon.NewtonSchulz` uses it as its success predicate.
-/
def ApproxOrthogonalizesBuffer {m n : Nat} (eps : α)
    (orthogonalizer : Orthogonalizer α (.dim m (.dim n .scalar)))
    (buffer : MatrixTensor α m n) : Prop :=
  HasApproxColumnGram eps (orthogonalizer.apply buffer)

/-! ## Certified backend records -/

/--
A Muon orthogonalizer with a success predicate on buffers: for every `buffer` with
`Success buffer`, the returned direction $Q$ satisfies $Q^\mathsf{T}Q=I$
(`ExactOrthogonalizesBuffer`).
-/
structure CheckedExactOrthogonalizer (α : Type) [Storage α] [Context α]
    (m n : Nat) where
  /-- The orthogonalizer that Muon applies to the momentum buffer. -/
  orthogonalizer : Orthogonalizer α (.dim m (.dim n .scalar))
  /-- Backend-specific condition on an input buffer under which the output is certified. -/
  Success : MatrixTensor α m n → Prop
  /-- Whenever `Success buffer` holds, the backend exactly orthogonalizes that buffer. -/
  certified : ∀ buffer : MatrixTensor α m n,
    Success buffer → ExactOrthogonalizesBuffer orthogonalizer buffer

/--
A Muon orthogonalizer with a success predicate on buffers: for every `buffer` with
`Success buffer`, every entry of $Q^\mathsf{T}Q-I$ for the returned direction $Q$ is at most `eps`
in absolute value (`ApproxOrthogonalizesBuffer`).
-/
structure CheckedApproxOrthogonalizer (α : Type) [Storage α] [Context α]
    (m n : Nat) (eps : α) where
  /-- The orthogonalizer that Muon applies to the momentum buffer. -/
  orthogonalizer : Orthogonalizer α (.dim m (.dim n .scalar))
  /-- Backend-specific condition on an input buffer under which the output is certified. -/
  Success : MatrixTensor α m n → Prop
  /-- Whenever `Success buffer` holds, the backend approximately orthogonalizes that buffer. -/
  certified : ∀ buffer : MatrixTensor α m n,
    Success buffer → ApproxOrthogonalizesBuffer eps orthogonalizer buffer

end Muon

end TorchLean
