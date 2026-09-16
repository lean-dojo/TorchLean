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
    {s : Shape} (x : RefTy (m := m) (α := α) s) : m (RefTy (m := m) (α := α) s) :=
  mul (m := m) (α := α) (s := s) x x

/-! ## Elementwise transcendentals for scientific forward models

Scientific forward models often use affine terms together with `exp`, `log`, `sin`, or `cos`.
These helpers expose the corresponding primitives through `nn.functional`, so the
forward equation can be written once as a pure `Function.Fn` and differentiated
by the autograd engine. Each helper wraps a primitive with a registered backward
rule, so reverse-mode `jacrev` and `grad` work through the expression.

PyTorch analogues: `torch.exp`, `torch.log`, `torch.sin`, `torch.cos`, and `c·x` / `c·x + k` via
`torch.mul`/`torch.add` against scalars. -/

/-- Elementwise exponential $x\mapsto e^x$. PyTorch: `torch.exp`. -/
def exp {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {s : Shape} (x : RefTy (m := m) (α := α) s) : m (RefTy (m := m) (α := α) s) :=
  Runtime.Autograd.Torch.exp (m := m) (α := α) (s := s) x

/--
Elementwise sine of angles in radians, differentiable through eager and typed graph execution.
-/
def sin {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {s : Shape} (x : RefTy (m := m) (α := α) s) : m (RefTy (m := m) (α := α) s) :=
  Runtime.Autograd.Torch.sin (m := m) (α := α) (s := s) x

/-- Elementwise cosine of angles in radians, with derivative `-sin(x)`. -/
def cos {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {s : Shape} (x : RefTy (m := m) (α := α) s) : m (RefTy (m := m) (α := α) s) :=
  Runtime.Autograd.Torch.cos (m := m) (α := α) (s := s) x

/-- Elementwise natural log $x\mapsto\log x$. PyTorch: `torch.log`.

Domain: for real-valued reasoning, assume positive inputs. This is the real
natural log only on $x>0$. TorchLean's eager CPU tape, IR evaluator, and proved
forward-fragment evaluator reject nonpositive inputs explicitly; typed graph
closures hit a runtime panic on a bad raw-log domain, and CUDA follows the native
buffer operation. Use `safeLog` when the model needs a total epsilon-protected
log-like operation. -/
def log {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {s : Shape} (x : RefTy (m := m) (α := α) s) : m (RefTy (m := m) (α := α) s) :=
  Runtime.Autograd.Torch.log (m := m) (α := α) (s := s) x

/-- Multiply by a scalar $c$: $x\mapsto cx$.
A re-export of the primitive `Ops.scale` through the functional API. `Ops.scale`
already powers `mean`; this definition gives users the direct
`nn.functional.*` name too. -/
def scale {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {s : Shape} (x : RefTy (m := m) (α := α) s) (c : α) : m (RefTy (m := m) (α := α) s) :=
  Runtime.Autograd.Torch.scale (m := m) (α := α) (s := s) x c

/-- Add a constant scalar $c$ to every element: $x\mapsto x+c$. Builds the
constant via `Ops.const` at scalar shape and broadcasts it to `s` (same pattern
as the dropout keep-probability broadcast). -/
def shift {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {s : Shape} (x : RefTy (m := m) (α := α) s) (c : α) : m (RefTy (m := m) (α := α) s) := do
  let cs ← Runtime.Autograd.Torch.const (m := m) (α := α) (s := Shape.scalar)
    (Tensor.scalar c)
  let cb ← Runtime.Autograd.Torch.broadcastTo (m := m) (α := α)
    (s₁ := Shape.scalar) (s₂ := s) (Shape.CanBroadcastTo.scalarTo s) cs
  Runtime.Autograd.Torch.add (m := m) (α := α) (s := s) x cb

/-- Scalar affine map $x\mapsto cx+k$.

This is a common building block in physical forward models, including the
SMAP-NISAR AVS surface and vegetation terms. It composes `scale` and `shift`. -/
def affine {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {s : Shape} (x : RefTy (m := m) (α := α) s) (c k : α) : m (RefTy (m := m) (α := α) s) := do
  let sx ← scale (m := m) (α := α) (s := s) x c
  shift (m := m) (α := α) (s := s) sx k

/-! ## Checkpointing (semantics-first identity wrapper) -/

/--
Checkpoint wrapper matching PyTorch's memory saving pattern.

In this codebase, checkpointing is a semantic identity wrapper
($\operatorname{checkpoint}(f,x)=f(x)$). Backends
that implement recomputation can refine this hook without changing the mathematical meaning.
-/
def checkpoint {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {s t : Shape}
    (f : RefTy (m := m) (α := α) s → m (RefTy (m := m) (α := α) t))
    (x : RefTy (m := m) (α := α) s) :
    m (RefTy (m := m) (α := α) t) :=
  f x

/-! ## Detach -/

/-- Stop-gradient boundary (forward identity). -/
def detach {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {s : Shape} (x : RefTy (m := m) (α := α) s) : m (RefTy (m := m) (α := α) s) :=
  Runtime.Autograd.Torch.detach (m := m) (α := α) (s := s) x

/-! ## Broadcasting helpers -/

/--
Broadcasting add: compute `x + y` after broadcasting both inputs to the target shape `t`.

PyTorch analogue: `torch.add` (broadcasting semantics).
-/
def addB {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {s₁ s₂ t : Shape} [Shape.BroadcastTo s₁ t] [Shape.BroadcastTo s₂ t]
    (x : RefTy (m := m) (α := α) s₁) (y : RefTy (m := m) (α := α) s₂) :
    m (RefTy (m := m) (α := α) t) := do
  let xb ←
    if h : s₁ = t then
      pure (h ▸ x)
    else
      broadcastTo (m := m) (α := α) (s₁ := s₁) (s₂ := t) Shape.BroadcastTo.proof x
  let yb ←
    if h : s₂ = t then
      pure (h ▸ y)
    else
      broadcastTo (m := m) (α := α) (s₁ := s₂) (s₂ := t) Shape.BroadcastTo.proof y
  add (m := m) (α := α) (s := t) xb yb

/--
Broadcasting multiply: compute `x * y` after broadcasting both inputs to the target shape `t`.

PyTorch analogue: `torch.mul` (broadcasting semantics).
-/
def mulB {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {s₁ s₂ t : Shape} [Shape.BroadcastTo s₁ t] [Shape.BroadcastTo s₂ t]
    (x : RefTy (m := m) (α := α) s₁) (y : RefTy (m := m) (α := α) s₂) :
    m (RefTy (m := m) (α := α) t) := do
  let xb ←
    if h : s₁ = t then
      pure (h ▸ x)
    else
      broadcastTo (m := m) (α := α) (s₁ := s₁) (s₂ := t) Shape.BroadcastTo.proof x
  let yb ←
    if h : s₂ = t then
      pure (h ▸ y)
    else
      broadcastTo (m := m) (α := α) (s₁ := s₂) (s₂ := t) Shape.BroadcastTo.proof y
  mul (m := m) (α := α) (s := t) xb yb

/-! ## Indexing helpers -/

namespace Internal

/-- Embedding lookup on an already flat vector of token ids: row select along axis `0`.

The public `embedding` reshapes down to this case and back, so all the interesting work happens
here and the wrapper only moves axes around. -/
def embeddingFlat {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {vocabularySize embeddingWidth count : Nat}
    (weight : RefTy (m := m) (α := α) [vocabularySize, embeddingWidth])
    (indices : Runtime.Autograd.Torch.DataRef
      (m := m) (α := α) (Fin vocabularySize) [count]) :
    m (RefTy (m := m) (α := α) [count, embeddingWidth]) :=
  indexSelect (m := m) (α := α)
    (s := [vocabularySize, embeddingWidth]) 0 count weight indices

end Internal

/--
Embedding lookup for an arbitrary tensor of bounded token ids.

The indexing primitive operates on a flat vector of indices. This wrapper flattens any input shape,
selects the corresponding rows, and restores the original axes with the embedding dimension
appended. The element type `Fin vocabularySize` makes an out-of-range token unrepresentable.
-/
def embedding {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {vocabularySize embeddingWidth : Nat} {s : Shape}
    (weight : RefTy (m := m) (α := α) [vocabularySize, embeddingWidth])
    (indices : Runtime.Autograd.Torch.DataRef
      (m := m) (α := α) (Fin vocabularySize) s) :
    m (RefTy (m := m) (α := α) (s.appendDim embeddingWidth)) := do
  let flatIds := Runtime.Autograd.Torch.mapData (m := m) (α := α)
    (fun x => TorchLean.Tensor.reshapeSpec
      (source := s) (target := [s.size]) x (by simp [Shape.size])) indices
  let gathered ← Internal.embeddingFlat (m := m) (α := α)
    (vocabularySize := vocabularySize) (embeddingWidth := embeddingWidth)
    (count := s.size) weight flatIds
  reshape (m := m) (α := α)
    (s₁ := [s.size, embeddingWidth])
    (s₂ := s.appendDim embeddingWidth) gathered
    (by simp [Shape.size_appendDim, Shape.size])

/-! ## Reductions -/

/--
Mean reduction:
$\operatorname{mean}(x)=\operatorname{sum}(x)/\operatorname{numel}(x)$.

PyTorch analogue: `torch.mean`.
-/
def mean {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
  {s : Shape} (x : RefTy (m := m) (α := α) s) : m (RefTy (m := m) (α := α) Shape.scalar) := do
  let total ← sum (m := m) (α := α) (s := s) x
  -- `sum` returns a scalar tensor; scale by `1 / numel` to get a mean.
  let denom : Nat := if Spec.Shape.size s = 0 then 1 else Spec.Shape.size s
  scale (m := m) (α := α) (s := Shape.scalar) total (1 / (denom : α))

/-! ## Seeded RNG helpers -/

/-- Deterministic `U[0,1)` tensor generator (seeded). -/
def randUniform {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {s : Shape} (seed : Nat) : m (RefTy (m := m) (α := α) s) :=
  Runtime.Autograd.Torch.randUniform (m := m) (α := α) (s := s) seed

/-- Deterministic `{0,1}` mask generator (seeded) with scalar keep-probability input. -/
def bernoulliMask {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {s : Shape} (keepProb : RefTy (m := m) (α := α) Shape.scalar) (seed : Nat) :
    m (RefTy (m := m) (α := α) s) :=
  Runtime.Autograd.Torch.bernoulliMask (m := m) (α := α) (s := s) keepProb seed

/--
Seeded dropout implemented as $x\odot\mathtt{mask}/\mathtt{keepProb}$, where
$\mathtt{mask}\in\{0,1\}$ is sampled from a
deterministic PRNG keyed by `seed`.
-/
def dropoutSeeded {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {s : Shape} (x : RefTy (m := m) (α := α) s) (p : α) (seed : Nat) (training : Bool := true) :
    m (RefTy (m := m) (α := α) s) := do
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
For `0 ≤ p ≤ 1`, a retained entry is scaled by `1 / (1 - p)` and a dropped entry is zero.
The seeded mask is held fixed during differentiation.

The denominator is `1 - p * mask`: it equals `1 - p` at retained entries and `1` at dropped
entries. At `p = 1`, every entry is dropped, so the forward value and gradients are zero without
evaluating a reciprocal at zero. Evaluation mode returns the input directly.
-/
def dropoutRefSeeded {α : Type} [TorchLean.Storage α] [Context α]
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
    let probability ←
      broadcastTo (m := m) (α := α) (s₁ := Shape.scalar) (s₂ := s)
        (Shape.CanBroadcastTo.scalarTo s) p
    let retainedProbability ← mul (m := m) (α := α) (s := s) probability mask
    let ones ← const (m := m) (α := α) (s := s) (Tensor.full s (1 : α))
    let denominator ← sub (m := m) (α := α) (s := s) ones retainedProbability
    let inverse ← inv (m := m) (α := α) (s := s) denominator
    mul (m := m) (α := α) (s := s) masked inverse
end F
end Model
end Autograd
end Runtime
