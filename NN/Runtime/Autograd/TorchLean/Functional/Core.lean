/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.TorchLean.Backend

import Mathlib.Algebra.Order.Algebra

@[expose] public section


namespace Runtime
namespace Autograd
namespace TorchLean

open Spec
open Tensor

namespace F

/-!
# Functional Core

Small functional helpers built from the primitive `TorchLean.Ops` surface.

These definitions are shared by eager and compiled execution, so they stay close to the primitive
operation names: elementwise helpers, broadcasting, embedding lookup, reductions, and seeded RNG.
-/

/-! ## Elementwise helpers -/

/-- Safe list indexing helper used in the dynamic (`String`-parsed) einsum/permute code paths. -/
def listGet? {β : Type} (xs : List β) (i : Nat) : Option β :=
  match xs.drop i with
  | [] => none
  | x :: _ => some x

/--
Elementwise square: `x ↦ x * x`.

PyTorch analogue: `torch.square`.
-/
def square {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {s : Shape} (x : RefTy (m := m) (α := α) s) : m (RefTy (m := m) (α := α) s) :=
  mul (m := m) (α := α) (s := s) x x

/-! ## Elementwise transcendentals (scientific forward-model ops)

These lift the primitive `Ops.{exp,log}` and the scalar-affine
`Ops.scale`/`Ops.const` into the functional surface, so geophysical / scientific
forward models — which lean on `exp`/`log` of an affine argument rather than the
NN-flavoured `relu`/`square`/`softmax` ops — can be written directly as a pure
`Function1.Fn` and differentiated by the autograd engine. Each wraps a primitive
that already carries a registered backward, so reverse-mode `jacrev`/`grad`
works through them unchanged.

PyTorch analogues: `torch.exp`, `torch.log`, and `c·x` / `c·x + k` via
`torch.mul`/`torch.add` against scalars. -/

/-- Elementwise exponential `x ↦ eˣ`.  PyTorch: `torch.exp`. -/
def exp {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {s : Shape} (x : RefTy (m := m) (α := α) s) : m (RefTy (m := m) (α := α) s) :=
  _root_.Runtime.Autograd.Torch.exp (m := m) (α := α) (s := s) x

/-- Elementwise natural log `x ↦ ln x`.  PyTorch: `torch.log`.

Domain: for real-valued reasoning, assume positive inputs — this is the real
natural log only on `x > 0`.  At the runtime/`Float` boundary, nonpositive inputs
follow backend behavior (e.g. `nan` / `-inf`) rather than a safe total
real-valued log. -/
def log {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {s : Shape} (x : RefTy (m := m) (α := α) s) : m (RefTy (m := m) (α := α) s) :=
  _root_.Runtime.Autograd.Torch.log (m := m) (α := α) (s := s) x

/-- Multiply by a compile-time-or-runtime constant scalar `c`: `x ↦ c · x`.
A re-export of the primitive `Ops.scale` onto the functional surface (it powers
`mean`, but was not itself exposed as `nn.functional.*`). -/
def scale {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {s : Shape} (x : RefTy (m := m) (α := α) s) (c : α) : m (RefTy (m := m) (α := α) s) :=
  _root_.Runtime.Autograd.Torch.scale (m := m) (α := α) (s := s) x c

/-- Add a constant scalar `c` to every element: `x ↦ x + c`.  Builds the
constant via `Ops.const` at scalar shape and broadcasts it to `s` (same pattern
as the dropout keep-probability broadcast). -/
def shift {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {s : Shape} (x : RefTy (m := m) (α := α) s) (c : α) : m (RefTy (m := m) (α := α) s) := do
  let cs ← _root_.Runtime.Autograd.Torch.const (m := m) (α := α) (s := Shape.scalar)
    (Tensor.scalar c)
  let cb ← _root_.Runtime.Autograd.Torch.broadcastTo (m := m) (α := α)
    (s₁ := Shape.scalar) (s₂ := s) (Shape.CanBroadcastTo.scalar_to_any s) cs
  _root_.Runtime.Autograd.Torch.add (m := m) (α := α) (s := s) x cb

/-- Scalar affine map `x ↦ c · x + k`.  The single most common building block of
linearised physical forward models (e.g. the SMAP-NISAR AVS surface/vegetation
terms).  Composes `scale` then `shift`. -/
def affine {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {s : Shape} (x : RefTy (m := m) (α := α) s) (c k : α) : m (RefTy (m := m) (α := α) s) := do
  let sx ← scale (m := m) (α := α) (s := s) x c
  shift (m := m) (α := α) (s := s) sx k

/-! ## Checkpointing (semantics-first identity wrapper) -/

/--
Checkpoint wrapper for API parity with PyTorch-style memory-saving patterns.

In this codebase, checkpointing is a semantic identity wrapper (`checkpoint f x = f x`). Backends
that implement recomputation can refine this hook without changing the mathematical meaning.
-/
def checkpoint {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {s t : Shape}
    (f : RefTy (m := m) (α := α) s → m (RefTy (m := m) (α := α) t))
    (x : RefTy (m := m) (α := α) s) :
    m (RefTy (m := m) (α := α) t) :=
  f x

/-! ## Detach / stop-grad -/

/-- Stop-gradient boundary (forward identity). -/
def detach {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {s : Shape} (x : RefTy (m := m) (α := α) s) : m (RefTy (m := m) (α := α) s) :=
  _root_.Runtime.Autograd.Torch.detach (m := m) (α := α) (s := s) x

/-- Alias for `detach`. -/
def stopGrad {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {s : Shape} (x : RefTy (m := m) (α := α) s) : m (RefTy (m := m) (α := α) s) :=
  detach (m := m) (α := α) (s := s) x

/-! ## Broadcasting helpers -/

/--
Broadcasting add: compute `x + y` after broadcasting both inputs to the target shape `t`.

PyTorch analogue: `torch.add` (broadcasting semantics).
-/
def addB {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {s₁ s₂ t : Shape} [Shape.BroadcastTo s₁ t] [Shape.BroadcastTo s₂ t]
    (x : RefTy (m := m) (α := α) s₁) (y : RefTy (m := m) (α := α) s₂) :
    m (RefTy (m := m) (α := α) t) := do
  let xb ← broadcastTo (m := m) (α := α) (s₁ := s₁) (s₂ := t) Shape.BroadcastTo.proof x
  let yb ← broadcastTo (m := m) (α := α) (s₁ := s₂) (s₂ := t) Shape.BroadcastTo.proof y
  add (m := m) (α := α) (s := t) xb yb

/--
Broadcasting multiply: compute `x * y` after broadcasting both inputs to the target shape `t`.

PyTorch analogue: `torch.mul` (broadcasting semantics).
-/
def mulB {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {s₁ s₂ t : Shape} [Shape.BroadcastTo s₁ t] [Shape.BroadcastTo s₂ t]
    (x : RefTy (m := m) (α := α) s₁) (y : RefTy (m := m) (α := α) s₂) :
    m (RefTy (m := m) (α := α) t) := do
  let xb ← broadcastTo (m := m) (α := α) (s₁ := s₁) (s₂ := t) Shape.BroadcastTo.proof x
  let yb ← broadcastTo (m := m) (α := α) (s₁ := s₂) (s₂ := t) Shape.BroadcastTo.proof y
  mul (m := m) (α := α) (s := t) xb yb

/-! ## Indexing helpers -/

/--
Embedding lookup (gather one row of an embedding table).

Given `w : vocab × dim`, return `w[idx] : dim`.

PyTorch analogue: `torch.nn.functional.embedding` for a single index.
-/
def embedding {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {vocab dim : Nat}
    (w : RefTy (m := m) (α := α) (.dim vocab (.dim dim .scalar)))
    (idx : Fin vocab) :
    m (RefTy (m := m) (α := α) (.dim dim .scalar)) :=
  gatherRow (m := m) (α := α) (rows := vocab) (cols := dim) w idx

/--
Embedding lookup for a vector of token ids.

This is the indexed version of the public one-hot embedding layer: instead of multiplying a
`k × vocab` one-hot matrix by the embedding table, gather the `k` rows directly from
`w : vocab × dim`.
-/
def embeddingRowsNat {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {vocab dim k : Nat}
    (w : RefTy (m := m) (α := α) (.dim vocab (.dim dim .scalar)))
    (idx : Tensor Nat (.dim k .scalar)) :
    m (RefTy (m := m) (α := α) (.dim k (.dim dim .scalar))) :=
  gatherRowsNat (m := m) (α := α) (rows := vocab) (cols := dim) (k := k) w idx

/--
Embedding lookup for a flattened `(batch × seqLen)` token-id tensor.

The index tensor is kept flat because that is how datasets and CUDA gather kernels naturally store
token ids.  The result is reshaped back to `(batch, seqLen, dim)` after the row gather.
-/
def embeddingBatchSeqNat {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {vocab dim batch seqLen : Nat}
    (w : RefTy (m := m) (α := α) (.dim vocab (.dim dim .scalar)))
    (idx : Tensor Nat (.dim (batch * seqLen) .scalar)) :
    m (RefTy (m := m) (α := α) (.dim batch (.dim seqLen (.dim dim .scalar)))) := do
  let gathered ← embeddingRowsNat (m := m) (α := α)
    (vocab := vocab) (dim := dim) (k := batch * seqLen) w idx
  reshape (m := m) (α := α)
    (s₁ := .dim (batch * seqLen) (.dim dim .scalar))
    (s₂ := .dim batch (.dim seqLen (.dim dim .scalar)))
    gathered (by
      simp [Shape.size, Nat.mul_assoc])

/-- Read float-encoded token ids as a `Tensor Nat` index vector (non-differentiable). -/
def floatVecToNatTensor {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)] {k : Nat}
    (x : RefTy (m := m) (α := α) (.dim k .scalar)) :
    m (Tensor Nat (.dim k .scalar)) :=
  _root_.Runtime.Autograd.Torch.floatVecToNatTensor (m := m) (α := α) (k := k) x

/-! ## Reductions -/

/--
Mean reduction: `mean(x) = sum(x) / numel(x)`.

PyTorch analogue: `torch.mean`.
-/
def mean {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {s : Shape} (x : RefTy (m := m) (α := α) s) : m (RefTy (m := m) (α := α) Shape.scalar) := do
  let total ← sum (m := m) (α := α) (s := s) x
  -- `sum` returns a scalar tensor; scale by `1 / numel` to get a mean.
  scale (m := m) (α := α) (s := Shape.scalar) total (1 / (Shape.size s : α))

/-! ## Seeded RNG helpers -/

/-- Deterministic `U[0,1)` tensor generator (seeded). -/
def randUniform {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {s : Shape} (seed : Nat) : m (RefTy (m := m) (α := α) s) :=
  _root_.Runtime.Autograd.Torch.randUniform (m := m) (α := α) (s := s) seed

/-- Deterministic `{0,1}` mask generator (seeded) with scalar keep-probability input. -/
def bernoulliMask {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {s : Shape} (keepProb : RefTy (m := m) (α := α) Shape.scalar) (seed : Nat) :
    m (RefTy (m := m) (α := α) s) :=
  _root_.Runtime.Autograd.Torch.bernoulliMask (m := m) (α := α) (s := s) keepProb seed

/--
Seeded dropout implemented as `x * mask / keepProb` where `mask ∈ {0,1}` is sampled from a
deterministic PRNG keyed by `seed`.
-/
def dropoutSeeded {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {s : Shape} (x : RefTy (m := m) (α := α) s) (p : α) (seed : Nat) (training : Bool := true) :
    m (RefTy (m := m) (α := α) s) := do
  if !training then
    pure x
  else
    let keepProb : α := (1 : α) - p
    let kpRef ← const (m := m) (α := α) (s := Shape.scalar) (Tensor.scalar keepProb)
    let mask ← bernoulliMask (m := m) (α := α) (s := s) kpRef seed
    let masked ← mul (m := m) (α := α) (s := s) x mask
    let invKp ← inv (m := m) (α := α) (s := Shape.scalar) kpRef
    let invKpB ←
      broadcastTo (m := m) (α := α) (s₁ := Shape.scalar) (s₂ := s)
        (Shape.CanBroadcastTo.scalar_to_any s) invKp
    mul (m := m) (α := α) (s := s) masked invKpB

/--
Seeded dropout where the probability is supplied as a scalar tensor ref.

This is useful in model builders where the layer definition stores `p` as data, so polymorphic model
code does not need a separate `Float → α` cast at the call site.
-/
def dropoutRefSeeded {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {s : Shape}
    (x : RefTy (m := m) (α := α) s)
    (p : RefTy (m := m) (α := α) Shape.scalar)
    (seed : Nat) (training : Bool := true) :
    m (RefTy (m := m) (α := α) s) := do
  if !training then
    pure x
  else
    let one ← const (m := m) (α := α) (s := Shape.scalar) (Tensor.scalar (1 : α))
    let keepProb ← sub (m := m) (α := α) (s := Shape.scalar) one p
    let mask ← bernoulliMask (m := m) (α := α) (s := s) keepProb seed
    let masked ← mul (m := m) (α := α) (s := s) x mask
    let invKp ← inv (m := m) (α := α) (s := Shape.scalar) keepProb
    let invKpB ←
      broadcastTo (m := m) (α := α) (s₁ := Shape.scalar) (s₂ := s)
        (Shape.CanBroadcastTo.scalar_to_any s) invKp
    mul (m := m) (α := α) (s := s) masked invKpB
end F
end TorchLean
end Autograd
end Runtime
