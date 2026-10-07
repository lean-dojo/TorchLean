/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Model.Program

@[expose] public section


namespace Runtime
namespace Autograd
namespace Model

open Spec TorchLean
open TorchLean TorchLean.Tensor

namespace F

/-!
# Functional Core

Small functional helpers built from the primitive `Runtime.Autograd.Torch.Ops` API.

These definitions are shared by eager and typed graph execution, so they stay close to the primitive
operation names: elementwise helpers, broadcasting, embedding lookup, reductions, and seeded RNG.
-/

/-! ## Elementwise helpers -/

/--
Elementwise square: $x\mapsto x^2$.

PyTorch analogue: `torch.square`.
-/
def square {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {s : Shape} (x : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) s) :=
  mul (m := m) (α := α) (s := s) x x

/-! ## Elementwise transcendentals for scientific forward models

Scientific forward models often use affine terms together with `exp`, `log`, `sin`, or `cos`.
These helpers expose the corresponding primitives through `nn.functional`, so the
forward equation can be written once as a pure function and differentiated
by the autograd engine. Each exported primitive has a registered backward
rule, so reverse-mode `jacrev` and `grad` work through the expression.

PyTorch analogues: `torch.exp`, `torch.log`, `torch.sin`, `torch.cos`, and `c·x` / `c·x + k` via
`torch.mul`/`torch.add` against scalars. -/

/-!
For real-valued reasoning, `log` requires positive inputs. The eager CPU tape and IR evaluator
reject nonpositive inputs; raw typed graph closures can panic, and CUDA uses the native operation.
Use `safeLog` when a model needs an epsilon-protected log-like operation.
-/
export Runtime.Autograd.Torch (exp sin cos log scale)

/-- Add a constant scalar $c$ to every element: $x\mapsto x+c$. Builds the
constant via `Ops.const` at scalar shape and broadcasts it to `s` (same pattern
as the dropout keep-probability broadcast). -/
def shift {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {s : Shape} (x : Ref (m := m) (α := α) s) (c : α) : m (Ref (m := m) (α := α) s) := do
  let cs ← Runtime.Autograd.Torch.const (m := m) (α := α) (s := Shape.scalar)
    (Tensor.scalar c)
  let cb ← Runtime.Autograd.Torch.broadcastTo (m := m) (α := α)
    (s₁ := Shape.scalar) (s₂ := s) (Shape.CanBroadcastTo.scalarTo s) cs
  Runtime.Autograd.Torch.add (m := m) (α := α) (s := s) x cb

/-- Scalar affine map $x\mapsto xc+k$.

This is a common building block in physical forward models, including the
SMAP-NISAR AVS surface and vegetation terms. It composes `scale` and `shift`. -/
def affine {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {s : Shape} (x : Ref (m := m) (α := α) s) (c k : α) : m (Ref (m := m) (α := α) s) := do
  let sx ← scale (m := m) (α := α) (s := s) x c
  shift (m := m) (α := α) (s := s) sx k

/-! ## Detach -/

export Runtime.Autograd.Torch (detach)

/-! ## Broadcasting helpers -/

/-- Avoid recording a broadcast node when the input already has the target shape. -/
def Internal.broadcastUnlessSame {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {s t : Shape} [Shape.BroadcastTo s t]
    (x : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) t) :=
  if h : s = t then
    pure (h ▸ x)
  else
    broadcastTo (m := m) (α := α) (s₁ := s) (s₂ := t) Shape.BroadcastTo.proof x

/--
Broadcasting add: compute `x + y` after broadcasting both inputs to the target shape `t`.

PyTorch analogue: `torch.add` (broadcasting semantics).
-/
def addB {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {s₁ s₂ t : Shape} [Shape.BroadcastTo s₁ t] [Shape.BroadcastTo s₂ t]
    (x : Ref (m := m) (α := α) s₁) (y : Ref (m := m) (α := α) s₂) :
    m (Ref (m := m) (α := α) t) := do
  let xb ← Internal.broadcastUnlessSame (m := m) (α := α) (s := s₁) (t := t) x
  let yb ← Internal.broadcastUnlessSame (m := m) (α := α) (s := s₂) (t := t) y
  add (m := m) (α := α) (s := t) xb yb

/--
Broadcasting multiply: compute `x * y` after broadcasting both inputs to the target shape `t`.

PyTorch analogue: `torch.mul` (broadcasting semantics).
-/
def mulB {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {s₁ s₂ t : Shape} [Shape.BroadcastTo s₁ t] [Shape.BroadcastTo s₂ t]
    (x : Ref (m := m) (α := α) s₁) (y : Ref (m := m) (α := α) s₂) :
    m (Ref (m := m) (α := α) t) := do
  let xb ← Internal.broadcastUnlessSame (m := m) (α := α) (s := s₁) (t := t) x
  let yb ← Internal.broadcastUnlessSame (m := m) (α := α) (s := s₂) (t := t) y
  mul (m := m) (α := α) (s := t) xb yb

/-! ## Indexing helpers -/

/--
Embedding lookup for an arbitrary tensor of bounded token ids.

The indexing primitive operates on a flat vector of indices. This wrapper flattens any input shape,
selects the corresponding rows, and restores the original axes with the embedding dimension
appended. The element type `Fin vocabularySize` makes an out-of-range token unrepresentable.
-/
def embedding {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {vocabularySize embeddingWidth : Nat} {s : Shape}
    (weight : Ref (m := m) (α := α) [vocabularySize, embeddingWidth])
    (indices : Runtime.Autograd.Torch.DataRef
      (m := m) (α := α) (Fin vocabularySize) s) :
    m (Ref (m := m) (α := α) (s.appendDim embeddingWidth)) := do
  let flatIds := Runtime.Autograd.Torch.mapData (m := m) (α := α)
    (fun x => TorchLean.Tensor.reshapeSpec
      (source := s) (target := [s.size]) x (by simp [Shape.size])) indices
  let gathered ← indexSelect (m := m) (α := α)
    (s := [vocabularySize, embeddingWidth]) 0 s.size weight flatIds
  reshape (m := m) (α := α)
    (s₁ := [s.size, embeddingWidth])
    (s₂ := s.appendDim embeddingWidth) gathered
    (by simp [Shape.size_appendDim, Shape.size])

/-! ## Reductions -/

/-- Mean over all elements, scaling the sum by the reciprocal of the element count.

An empty tensor uses denominator one and returns its zero sum, unlike PyTorch's NaN convention.
The reciprocal is computed before multiplication; this need not round like direct division.
-/
def mean {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
  {s : Shape} (x : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) Shape.scalar) := do
  let total ← sum (m := m) (α := α) (s := s) x
  let denom : Nat := Tensor.meanDenominator s
  scale (m := m) (α := α) (s := Shape.scalar) total (1 / (denom : α))

/-! ## Seeded RNG helpers -/

export Runtime.Autograd.Torch (randUniform bernoulliMask)

/--
Seeded dropout implemented as $x\odot\mathtt{mask}/\mathtt{keepProb}$, where
$\mathtt{mask}\in\{0,1\}$ is sampled from a
deterministic PRNG keyed by `seed`.
-/
def dropoutSeeded {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {s : Shape} (x : Ref (m := m) (α := α) s) (p : α) (seed : Nat) (training : Bool := true) :
    m (Ref (m := m) (α := α) s) := do
  if !training || p == 0 then
    pure x
  else if p == 1 then
    scale (m := m) (α := α) (s := s) x 0
  else
    let keepProb : α := (1 : α) - p
    let kpRef ← const (m := m) (α := α) (s := Shape.scalar) (Tensor.scalar keepProb)
    let mask ← bernoulliMask (m := m) (α := α) (s := s) kpRef seed
    let masked ← mul (m := m) (α := α) (s := s) x mask
    let invKp ← inv (m := m) (α := α) (s := Shape.scalar) kpRef
    let invKpB ←
      broadcastTo (m := m) (α := α) (s₁ := Shape.scalar) (s₂ := s)
        (Shape.CanBroadcastTo.scalarTo s) invKp
    mul (m := m) (α := α) (s := s) masked invKpB

/--
Seeded dropout where the probability is supplied as a scalar tensor ref.

Model builders can store `p` as tensor data and pass it through the same interface as the input.
For finite inputs and `0 ≤ p ≤ 1`, a retained entry is scaled by `1 / (1 - p)` and a
dropped entry is zero.
The seeded mask is held fixed during differentiation.

The denominator is `1 - p * mask`: it equals `1 - p` at retained entries and `1` at dropped
entries. At `p = 1`, every entry is dropped, avoiding a reciprocal at zero. With finite
intermediates, the forward value and gradients are zero. Multiplying by a zero mask does not
suppress a NaN or infinity. Evaluation mode returns the input directly.
-/
def dropoutRefSeeded {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {s : Shape}
    (x : Ref (m := m) (α := α) s)
    (p : Ref (m := m) (α := α) Shape.scalar)
    (seed : Nat) (training : Bool := true) :
    m (Ref (m := m) (α := α) s) := do
  if !training then
    pure x
  else
    let one ← const (m := m) (α := α) (s := Shape.scalar) (Tensor.scalar (1 : α))
    let keepProb ← sub (m := m) (α := α) (s := Shape.scalar) one p
    let mask ← bernoulliMask (m := m) (α := α) (s := s) keepProb seed
    let masked ← mul (m := m) (α := α) (s := s) x mask
    let probability ←
      broadcastTo (m := m) (α := α) (s₁ := Shape.scalar) (s₂ := s)
        (Shape.CanBroadcastTo.scalarTo s) p
    let retainedProbability ← mul (m := m) (α := α) (s := s) probability mask
    let ones ←
      broadcastTo (m := m) (α := α) (s₁ := Shape.scalar) (s₂ := s)
        (Shape.CanBroadcastTo.scalarTo s) one
    let denominator ← sub (m := m) (α := α) (s := s) ones retainedProbability
    let inverse ← inv (m := m) (α := α) (s := s) denominator
    mul (m := m) (α := α) (s := s) masked inverse
end F
end Model
end Autograd
end Runtime
