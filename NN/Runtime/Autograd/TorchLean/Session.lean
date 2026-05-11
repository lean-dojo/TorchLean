/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.LinkedSession

import Mathlib.Algebra.Order.Algebra

/-!
# Session

TorchLean unified imperative session.

## Mental Model

A `Session α` is TorchLean's runtime analogue of a PyTorch "training loop environment". It:
- owns a collection of *leaf tensors* (parameters and inputs),
- records a forward computation as a tape / dataflow graph,
- can run reverse-mode AD to produce gradients for all leaves, and
- can apply simple optimizer steps (e.g. SGD) in a session-style workflow.

TorchLean exposes a **single API** with two execution backends selected at construction time:
- `.eager`: a tape-backed runtime session (imperative autograd tape; great for debugging and quick
  demos),
- `.compiled`: a proof-linked session that also records a (proved) IR graph while you build the tape
  and
  then executes via `Runtime.Autograd.Torch.Internal.SessionIR`.

Importantly, *the user-facing `Session` API is the same*: each op dispatches through `Session.impl`.

## Typical Training Loop (PyTorch Analogy)

Think of the following mapping (roughly):
- `Session.param` ~ create a `torch.nn.Parameter` (and later include it in a `state_dict`-like
  bundle).
- `Session.use` ~ "read" a parameter as a tensor in the current forward graph.
- `Session.input` ~ add a leaf tensor input (like feeding a batch tensor into the forward pass).
- `Session.resetTape` ~ start a fresh forward graph (closest in spirit to `optimizer.zero_grad()` +
  new forward).
- `Session.backwardScalarDenseAll` ~ `loss.backward()` (but returns gradients explicitly as an
  array).
- `Session.sgdStepAll` ~ `optimizer.step()` (dense helper; higher-level training lives in
  `NN.API.*`).
- `Session.detach` ~ `tensor.detach()` (cut the gradient edge at a value).

TorchLean does *not* store mutable `.grad` fields on each tensor ref; instead, gradients are
  returned
explicitly (see `grad`, `vjp`, and the `backward*DenseAll` functions).

## Non-Differentiable Inputs (`NatRef`)

For labels/indices, we keep a separate non-differentiable channel (`NatRef` and `NatVecRef`), used
  by
gather/indexing ops. This mirrors the practical reality that targets are often integer tensors in
PyTorch and should not require embedding into `α`.

## Deterministic RNG (Session-Level)

`RngState` provides explicit, deterministic RNG state (closer to JAX PRNG keys than a global RNG).
`freshSeedIO` is a convenience for sampling an initial seed at the IO boundary, while the *core*
semantics remains seed-threaded and replayable.

## Connection To TorchLean IR / Graph Execution

In the `.compiled` backend, the session records an IR graph while you build the tape; that IR is the
artifact that can be linked to proofs/verifiers. Execution remains compatible with the same
tape-level semantics, and the concrete graph object can be inspected or exported through the
compiled session.

Practical note: the current `.compiled` implementation expects all leaves (tensor inputs/parameters
and `NatRef`s) to be created before any op nodes are recorded. For portability, allocate leaves and
initialize/split RNG up-front, then build the forward graph.

### PyTorch References

- `torch.autograd`: https://pytorch.org/docs/stable/autograd.html
- Tensor hooks (conceptual analogue of `backwardDenseAllWithHook`):
  https://pytorch.org/docs/stable/generated/torch.Tensor.register_hook.html

### AD References

This code follows the classic "tape / Wengert list" view of reverse-mode AD:
- Andreas Griewank and Andrea Walther, *Evaluating Derivatives*, 2nd ed., 2008.
- Seppo Linnainmaa, 1970 (reverse accumulation; precursor to modern backprop/autograd).
-/

@[expose] public section


namespace Runtime
namespace Autograd
namespace TorchLean

open Spec
open Tensor

/--
Eager-only session wrapper.

This is a thin wrapper around the internal tape-backed session
`Runtime.Autograd.Torch.Internal.EagerSession`. Users normally interact with the unified `Session`
API; this type exists to support backend dispatch (`SessionImpl.eager`).
-/
structure EagerSession (α : Type) where
  /-- inner. -/
  inner : _root_.Runtime.Autograd.Torch.Internal.EagerSession α

namespace EagerSession

/--
Create a new eager (tape-backed) session.

This corresponds to the `.eager` backend of `Session.new`.
-/
def new {α : Type} (opts : _root_.Runtime.Autograd.Torch.Options := {}) : IO (EagerSession α) := do
  let inner ← _root_.Runtime.Autograd.Torch.Internal.EagerSession.new (α := α) (opts := opts)
  pure { inner := inner }

/-- Reset the eager autograd tape / graph-building state. -/
def resetTape {α : Type} (s : EagerSession α) : IO Unit := do
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.resetTape (α := α) s.inner

/--
Create a learnable parameter owned by this session.

PyTorch analogy: creating a `torch.nn.Parameter` during module initialization.
-/
def param {α : Type} (s : EagerSession α) {sh : Shape}
  (init : Tensor α sh) (name : Option String := none) (requiresGrad : Option Bool := none) :
  IO (_root_.Runtime.Autograd.Torch.Param α sh) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.param (α := α) (sh := sh) s.inner
    init (name := name) (requiresGrad := requiresGrad)

/--
Use a parameter inside the current forward graph.

PyTorch analogy: reading a parameter in `forward` (it becomes part of the autograd graph).
-/
def use {α : Type} (s : EagerSession α) {sh : Shape} [DecidableEq Shape]
  [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
  (p : _root_.Runtime.Autograd.Torch.Param α sh) : IO (_root_.Runtime.Autograd.Torch.TensorRef α sh)
    :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.use (α := α) (sh := sh) s.inner p

/--
Add a tensor input leaf to the current graph.

`requiresGrad` controls whether this input is recorded as a differentiable leaf.
-/
def input {α : Type} (s : EagerSession α) {sh : Shape} [DecidableEq Shape]
  [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
  (v : Tensor α sh) (name : Option String := none) (requiresGrad : Bool := false) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α sh) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.input (α := α) (sh := sh) s.inner
    v (name := name) (requiresGrad := requiresGrad)

/--
Add a non-differentiable `Nat` leaf to the session.

Used for labels/indices and gather-style ops.
-/
def inputNat {α : Type} (s : EagerSession α) (v : Nat) : IO (_root_.Runtime.Autograd.Torch.NatRef)
  :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.inputNat (α := α) s.inner v

/-- Read a `NatRef` value. -/
def getNat {α : Type} (s : EagerSession α) (r : _root_.Runtime.Autograd.Torch.NatRef) : IO Nat :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.getNat (α := α) s.inner r

/-- Mutate a `NatRef` value. -/
def setNat {α : Type} (s : EagerSession α) (r : _root_.Runtime.Autograd.Torch.NatRef) (v : Nat) : IO
  Unit :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.setNat (α := α) s.inner r v

/-- Add a non-differentiable vector-of-`Nat` leaf. -/
def inputNatVec {α : Type} {k : Nat} (s : EagerSession α) (v : Tensor Nat (.dim k .scalar)) :
    IO (_root_.Runtime.Autograd.Torch.NatVecRef k) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.inputNatVec (α := α) (k := k) s.inner v

/-- Read back a `NatVecRef` value. -/
def getNatVec {α : Type} {k : Nat} (s : EagerSession α) (r : _root_.Runtime.Autograd.Torch.NatVecRef
  k) :
    IO (Tensor Nat (.dim k .scalar)) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.getNatVec (α := α) (k := k) s.inner r

/-- Mutate a `NatVecRef` value. -/
def setNatVec {α : Type} {k : Nat} (s : EagerSession α) (r : _root_.Runtime.Autograd.Torch.NatVecRef
  k)
    (v : Tensor Nat (.dim k .scalar)) : IO Unit :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.setNatVec (α := α) (k := k) s.inner r v

/--
Insert a constant tensor into the current graph.

PyTorch analogy: using a tensor literal/constant in the forward pass (as a leaf constant node).
-/
def const {α : Type} (s : EagerSession α) {sh : Shape} [DecidableEq Shape]
  [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
  (v : Tensor α sh) (name : Option String := none) : IO (_root_.Runtime.Autograd.Torch.TensorRef α
    sh) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.const (α := α) (sh := sh) s.inner v (name :=
    name)

/-- Read the concrete value for a tensor ref (for logging/debugging). -/
def getValue {α : Type} (s : EagerSession α) {sh : Shape} [DecidableEq Shape]
  [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
  (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) : IO (Tensor α sh) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.getValue (α := α) (sh := sh) s.inner x

/--
Detach a tensor ref from the tape (stop gradient flow through it).

PyTorch analogy: `x.detach()`.
-/
def detach {α : Type} (s : EagerSession α) {sh : Shape} [Context α] [DecidableEq Shape]
    [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) :
    IO (_root_.Runtime.Autograd.Torch.TensorRef α sh) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.detach (α := α) (sh := sh) s.inner x

/-- Elementwise addition on tensor refs (eager backend). -/
def add {α : Type} (s : EagerSession α) [Add α] [DecidableEq Shape] {sh : Shape}
  (a b : _root_.Runtime.Autograd.Torch.TensorRef α sh) : IO (_root_.Runtime.Autograd.Torch.TensorRef
    α sh) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.add (α := α) (sh := sh) s.inner a b

/-- Elementwise subtraction on tensor refs (eager backend). -/
def sub {α : Type} (s : EagerSession α) [Sub α] [Zero α] [DecidableEq Shape] {sh : Shape}
  (a b : _root_.Runtime.Autograd.Torch.TensorRef α sh) : IO (_root_.Runtime.Autograd.Torch.TensorRef
    α sh) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.sub (α := α) (sh := sh) s.inner a b

/-- Elementwise multiplication on tensor refs (eager backend). -/
def mul {α : Type} (s : EagerSession α) [Mul α] [DecidableEq Shape] {sh : Shape}
  (a b : _root_.Runtime.Autograd.Torch.TensorRef α sh) : IO (_root_.Runtime.Autograd.Torch.TensorRef
    α sh) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.mul (α := α) (sh := sh) s.inner a b

/-- Elementwise scaling by a scalar constant `c` (eager backend). -/
def scale {α : Type} (s : EagerSession α) [Mul α] [DecidableEq Shape] {sh : Shape}
  [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
  (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) (c : α) : IO
    (_root_.Runtime.Autograd.Torch.TensorRef α sh) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.scale (α := α) (sh := sh) s.inner x c

/-- Elementwise absolute value (eager backend). -/
def abs {α : Type} (s : EagerSession α) [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  [DecidableEq Shape]
  {sh : Shape} (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α sh) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.abs (α := α) (sh := sh) s.inner x

/-- Elementwise square root (eager backend). -/
def sqrt {α : Type} (s : EagerSession α) [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  [DecidableEq Shape]
  {sh : Shape} (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α sh) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.sqrt (α := α) (sh := sh) s.inner x

/-- Elementwise clamp to `[minVal, maxVal]` (eager backend). -/
def clamp {α : Type} (s : EagerSession α) [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  [DecidableEq Shape]
  [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
  {sh : Shape} (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) (minVal maxVal : α) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α sh) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.clamp (α := α) (sh := sh) s.inner x minVal
    maxVal

/-- Elementwise maximum (eager backend). -/
def max {α : Type} (s : EagerSession α) [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  [DecidableEq Shape]
  {sh : Shape} (a b : _root_.Runtime.Autograd.Torch.TensorRef α sh) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α sh) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.max (α := α) (sh := sh) s.inner a b

/-- Elementwise minimum (eager backend). -/
def min {α : Type} (s : EagerSession α) [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  [DecidableEq Shape]
  {sh : Shape} (a b : _root_.Runtime.Autograd.Torch.TensorRef α sh) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α sh) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.min (α := α) (sh := sh) s.inner a b

/--
Matrix multiplication (2D) on tensor refs (eager backend).

PyTorch analogy: `torch.matmul` on rank-2 tensors (or the `@` operator).
-/
def matmul {α : Type} (s : EagerSession α) [Context α] [DecidableEq Shape]
  {m n p : Nat}
  (a : _root_.Runtime.Autograd.Torch.TensorRef α (.dim m (.dim n .scalar)))
  (b : _root_.Runtime.Autograd.Torch.TensorRef α (.dim n (.dim p .scalar))) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (.dim m (.dim p .scalar))) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.matmul (α := α) s.inner (m := m) (n := n) (p
    := p) a b

/--
Batched matrix multiplication (3D) on tensor refs (eager backend).

PyTorch analogy: `torch.bmm`.
-/
def bmm {α : Type} (s : EagerSession α) [Add α] [Mul α] [Zero α] [DecidableEq Shape]
  {batch m n p : Nat}
  (a : _root_.Runtime.Autograd.Torch.TensorRef α (.dim batch (.dim m (.dim n .scalar))))
  (b : _root_.Runtime.Autograd.Torch.TensorRef α (.dim batch (.dim n (.dim p .scalar)))) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (.dim batch (.dim m (.dim p .scalar)))) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.bmm (α := α) s.inner (batch := batch) (m := m)
    (n := n) (p := p) a b

/--
Concatenate two vectors along the only dimension (eager backend).

PyTorch analogy: `torch.cat([a, b], dim=0)` for 1D tensors.
-/
def concatVectors {α : Type} (s : EagerSession α) [Context α] [DecidableEq Shape]
  {n m : Nat}
  (a : _root_.Runtime.Autograd.Torch.TensorRef α (.dim n .scalar))
  (b : _root_.Runtime.Autograd.Torch.TensorRef α (.dim m .scalar)) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (.dim (n + m) .scalar)) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.concatVectors (α := α) s.inner (n := n) (m :=
    m) a b

/--
Concatenate along the outermost dimension (dimension 0) (eager backend).

PyTorch analogy: `torch.cat([a, b], dim=0)`.
-/
def concatDim0 {α : Type} (s : EagerSession α) [Context α] [DecidableEq Shape]
  {n m : Nat} {sh : Shape}
  (a : _root_.Runtime.Autograd.Torch.TensorRef α (.dim n sh))
  (b : _root_.Runtime.Autograd.Torch.TensorRef α (.dim m sh)) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (.dim (n + m) sh)) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.concatDim0 (α := α) s.inner (n := n) (m := m)
    (sh := sh) a b

/--
Slice a contiguous `[start, start+len)` range from dimension 0 (eager backend).

PyTorch analogy: `x[start:start+len]` for the first dimension.
-/
def sliceRange0 {α : Type} (s : EagerSession α) [Zero α] [DecidableEq Shape]
  {n : Nat} {sh : Shape}
  (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim n sh)) (start len : Nat) (h : len + start ≤
    n) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (.dim len sh)) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.sliceRange0 (α := α) s.inner (n := n) (sh :=
    sh) x start len h

/--
2D max pooling on a CHW tensor (eager backend).

PyTorch analogy: `torch.nn.functional.max_pool2d` (channel-first layout).
-/
def maxPool2d {α : Type} (s : EagerSession α) [Context α] [DecidableEq Shape]
  {kH kW inH inW inC stride : Nat} {h1 : kH ≠ 0} {h2 : kW ≠ 0}
  (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim inC (.dim inH (.dim inW .scalar)))) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α
    (.dim inC (.dim ((inH - kH) / stride + 1) (.dim ((inW - kW) / stride + 1) .scalar)))) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.maxPool2d (α := α) s.inner
    (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride)
    (h1 := h1) (h2 := h2) x

/-- Alias for `max_pool2d` (PyTorch-style shorthand). -/
abbrev maxPool {α : Type} := maxPool2d (α := α)

/--
Smooth max pooling (softmax-like pooling) on a CHW tensor (eager backend).

This is a differentiable surrogate for max pooling parameterized by `beta`.
-/
def smoothMaxPool2d {α : Type} (s : EagerSession α) [Context α] [DecidableEq Shape]
  [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
  {kH kW inH inW inC stride : Nat} {h1 : kH ≠ 0} {h2 : kW ≠ 0}
  (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim inC (.dim inH (.dim inW .scalar)))) (beta :
    α) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α
    (.dim inC (.dim ((inH - kH) / stride + 1) (.dim ((inW - kW) / stride + 1) .scalar)))) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.smoothMaxPool2d (α := α) s.inner
    (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride)
    (h1 := h1) (h2 := h2) x beta

/-- Alias for `smooth_max_pool2d` (PyTorch-style shorthand). -/
abbrev smoothMaxPool {α : Type} := smoothMaxPool2d (α := α)

/--
2D average pooling on a CHW tensor (eager backend).

PyTorch analogy: `torch.nn.functional.avg_pool2d` (channel-first layout).
-/
def avgPool2d {α : Type} (s : EagerSession α) [Context α] [DecidableEq Shape]
  {kH kW inH inW inC stride : Nat} (h1 : kH ≠ 0) (h2 : kW ≠ 0)
  (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim inC (.dim inH (.dim inW .scalar)))) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α
    (.dim inC (.dim ((inH - kH) / stride + 1) (.dim ((inW - kW) / stride + 1) .scalar)))) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.avgPool2d (α := α) s.inner
    (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride)
    h1 h2 x

/-- Alias for `avg_pool2d` (PyTorch-style shorthand). -/
abbrev avgPool {α : Type} := avgPool2d (α := α)

/-- Elementwise ReLU activation (eager backend). -/
def relu {α : Type} (s : EagerSession α)
  [Mul α] [Zero α] [Max α] [One α] [LT α]
  [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {sh : Shape} (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) : IO
    (_root_.Runtime.Autograd.Torch.TensorRef α sh) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.relu (α := α) (sh := sh) s.inner x

/-- Elementwise sigmoid activation (eager backend). -/
def sigmoid {α : Type} (s : EagerSession α) [Context α] [DecidableEq Shape]
  {sh : Shape} (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) : IO
    (_root_.Runtime.Autograd.Torch.TensorRef α sh) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.sigmoid (α := α) (sh := sh) s.inner x

/-- Elementwise tanh activation (eager backend). -/
def tanh {α : Type} (s : EagerSession α) [Context α] [DecidableEq Shape]
  {sh : Shape} (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) : IO
    (_root_.Runtime.Autograd.Torch.TensorRef α sh) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.tanh (α := α) (sh := sh) s.inner x

/-- Elementwise softmax activation (eager backend). -/
def softmax {α : Type} (s : EagerSession α) [Context α] [DecidableEq Shape]
  {sh : Shape} (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) : IO
    (_root_.Runtime.Autograd.Torch.TensorRef α sh) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.softmax (α := α) (sh := sh) s.inner x

/-- Stable log-softmax along the last axis (eager backend). -/
def logSoftmax {α : Type} (s : EagerSession α) [Context α] [DecidableEq Shape]
  {sh : Shape} (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) : IO
    (_root_.Runtime.Autograd.Torch.TensorRef α sh) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.logSoftmax (α := α) (sh := sh) s.inner x

/-- Elementwise softplus activation (eager backend). -/
def softplus {α : Type} (s : EagerSession α) [Context α] [DecidableEq Shape]
  {sh : Shape} (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) : IO
    (_root_.Runtime.Autograd.Torch.TensorRef α sh) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.softplus (α := α) (sh := sh) s.inner x

/-- Elementwise exponential (eager backend). -/
def exp {α : Type} (s : EagerSession α) [Context α] [DecidableEq Shape]
  {sh : Shape} (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) : IO
    (_root_.Runtime.Autograd.Torch.TensorRef α sh) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.exp (α := α) (sh := sh) s.inner x

/-- Elementwise logarithm (eager backend). -/
def log {α : Type} (s : EagerSession α) [Context α] [DecidableEq Shape]
  {sh : Shape} (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) : IO
    (_root_.Runtime.Autograd.Torch.TensorRef α sh) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.log (α := α) (sh := sh) s.inner x

/-- Elementwise `safe_log` activation (`log(softplus(x) + ε)`) (eager backend). -/
def safeLog {α : Type} (s : EagerSession α) [Context α] [DecidableEq Shape]
  [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
  {sh : Shape} (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) (ε : α := Numbers.epsilon) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α sh) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.safeLog (α := α) (sh := sh) s.inner x (ε :=
    ε)

/-- Sum-reduce a tensor to a scalar (eager backend). -/
def sum {α : Type} (s : EagerSession α) [Add α] [Zero α] [DecidableEq Shape]
  {sh : Shape} (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α Shape.scalar) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.sum (α := α) (sh := sh) s.inner x

/-- Flatten a tensor into a 1D vector (eager backend). -/
def flatten {α : Type} (s : EagerSession α) [Inhabited α] [DecidableEq Shape] {sh : Shape}
  (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (.dim (Shape.size sh) .scalar)) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.flatten (α := α) (sh := sh) s.inner x

/--
Reshape a tensor, given a proof that the total number of elements is preserved (eager backend).

PyTorch analogy: `x.reshape(...)` when the element count matches.
-/
def reshape {α : Type} (s : EagerSession α) [Inhabited α] [DecidableEq Shape] {sh1 sh2 : Shape}
  (x : _root_.Runtime.Autograd.Torch.TensorRef α sh1) (h : Shape.size sh1 = Shape.size sh2) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α sh2) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.reshape (α := α) (sh1 := sh1) (sh2 := sh2)
    s.inner x h

/-- Transpose a 2D matrix (eager backend). -/
def transpose2d {α : Type} (s : EagerSession α) [Zero α] [DecidableEq Shape] {m n : Nat}
  (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim m (.dim n .scalar))) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (.dim n (.dim m .scalar))) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.transpose2d (α := α) (m := m) (n := n) s.inner
    x

/-- Permute a 3D tensor by moving the first dimension to the last (eager backend). -/
def transpose3dFirstToLast {α : Type} (s : EagerSession α) [Zero α] [DecidableEq Shape] {a b c :
  Nat}
  (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim a (.dim b (.dim c .scalar)))) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (.dim b (.dim c (.dim a .scalar)))) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.transpose3dFirstToLast (α := α)
    (a := a) (b := b) (c := c) s.inner x

/-- Permute a 3D tensor by moving the last dimension to the first (eager backend). -/
def transpose3dLastToFirst {α : Type} (s : EagerSession α) [Zero α] [DecidableEq Shape] {a b c :
  Nat}
  (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim a (.dim b (.dim c .scalar)))) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (.dim c (.dim a (.dim b .scalar)))) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.transpose3dLastToFirst (α := α)
    (a := a) (b := b) (c := c) s.inner x

/-- Swap the last two axes of a 3D tensor (eager backend). -/
def transpose3dLastTwo {α : Type} (s : EagerSession α) [Zero α] [DecidableEq Shape] {a b c : Nat}
  (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim a (.dim b (.dim c .scalar)))) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (.dim a (.dim c (.dim b .scalar)))) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.transpose3dLastTwo (α := α)
    (a := a) (b := b) (c := c) s.inner x

/--
Generic "swap adjacent axes" view operation (eager backend).

This is a shape-driven permutation helper used in some attention/transformer code.
-/
def swapAdjacentAtDepth {α : Type} (s : EagerSession α) [Context α] [DecidableEq Shape] {sh : Shape}
  (depth : Nat) (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (sh.swapAdjacentAtDepth depth)) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.swapAdjacentAtDepth (α := α) (sh := sh)
    s.inner depth x

/-- Broadcast a tensor to a larger shape (eager backend). -/
def broadcastTo {α : Type} (s : EagerSession α) [Inhabited α] [Add α] [Zero α] [DecidableEq Shape]
  {sh1 sh2 : Shape} (cb : Shape.CanBroadcastTo sh1 sh2) (x : _root_.Runtime.Autograd.Torch.TensorRef
    α sh1) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α sh2) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.broadcastTo (α := α) (sh1 := sh1) (sh2 := sh2)
    s.inner cb x

/-- Reduce-sum along an axis (eager backend). -/
def reduceSum {α : Type} (s : EagerSession α) [Add α] [Zero α] [Inhabited α] [DecidableEq Shape]
  {sh : Shape} (axis : Nat) [valid : Shape.valid_axis_inst axis sh] [wf : Shape.WellFormed sh]
  (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (shapeAfterSum sh axis)) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.reduceSum (α := α) (sh := sh) s.inner axis x

/-- Reduce-mean along an axis (eager backend). -/
def reduceMean {α : Type} (s : EagerSession α) [Context α] [Inhabited α] [DecidableEq Shape]
  {sh : Shape} (axis : Nat) [valid : Shape.valid_axis_inst axis sh] [wf : Shape.WellFormed sh]
  (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (shapeAfterSum sh axis)) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.reduceMean (α := α) (sh := sh) s.inner axis x

/-- Gather a single scalar from a vector at a `Fin` index (eager backend). -/
def gatherScalar {α : Type} (s : EagerSession α) [Zero α] [DecidableEq Shape]
  {n : Nat} (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim n .scalar)) (i : Fin n) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α Shape.scalar) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.gatherScalar (α := α) (n := n) s.inner x i

/-- Gather a row from a matrix at a `Fin` index (eager backend). -/
def gatherRow {α : Type} (s : EagerSession α) [Zero α] [DecidableEq Shape]
  {rows cols : Nat} (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim rows (.dim cols .scalar)))
    (i : Fin rows) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (.dim cols .scalar)) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.gatherRow (α := α) (rows := rows) (cols :=
    cols) s.inner x i

/-- Gather a scalar from a vector using a `NatRef` index (eager backend). -/
def gatherScalarRef {α : Type} (s : EagerSession α) [Zero α] [DecidableEq Shape]
  {n : Nat} (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim n .scalar)) (i :
    _root_.Runtime.Autograd.Torch.NatRef) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α Shape.scalar) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.gatherScalarRef (α := α) (n := n) s.inner x
    i

/-- Gather a row from a matrix using a `NatRef` index (eager backend). -/
def gatherRowRef {α : Type} (s : EagerSession α) [Zero α] [DecidableEq Shape]
  [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
  {rows cols : Nat} (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim rows (.dim cols .scalar)))
    (i : _root_.Runtime.Autograd.Torch.NatRef) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (.dim cols .scalar)) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.gatherRowRef (α := α) (rows := rows) (cols
    := cols) s.inner x i

/-- Gather a scalar using a raw `Nat` index (eager backend). -/
def gatherScalarNat {α : Type} (s : EagerSession α) [Zero α] [DecidableEq Shape]
  {n : Nat} (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim n .scalar)) (i : Nat) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α Shape.scalar) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.gatherScalarNat (α := α) (n := n) s.inner x
    i

/--
Gather a vector of entries from a vector using an index tensor (eager backend).

PyTorch analogy: `x[idx]` where `idx` is an integer tensor (1D).
-/
def gatherVecNat {α : Type} (s : EagerSession α) [Add α] [Zero α] [DecidableEq Shape]
  {n k : Nat} (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim n .scalar)) (idx : Tensor Nat
    (.dim k .scalar)) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (.dim k .scalar)) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.gatherVecNat (α := α) (n := n) (k := k)
    s.inner x idx

/-- Gather multiple rows from a matrix using an index tensor (eager backend). -/
def gatherRowsNat {α : Type} (s : EagerSession α) [Add α] [Zero α] [DecidableEq Shape]
  {rows cols k : Nat} (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim rows (.dim cols
    .scalar)))
  (idx : Tensor Nat (.dim k .scalar)) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (.dim k (.dim cols .scalar))) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.gatherRowsNat (α := α)
    (rows := rows) (cols := cols) (k := k) s.inner x idx

/-- `gather_vec_nat`, but the indices are provided as a `NatVecRef` leaf (eager backend). -/
def gatherVecRef {α : Type} (s : EagerSession α) [Add α] [Zero α] [DecidableEq Shape]
  {n k : Nat} (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim n .scalar))
  (idx : _root_.Runtime.Autograd.Torch.NatVecRef k) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (.dim k .scalar)) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.gatherVecRef (α := α) (n := n) (k := k)
    s.inner x idx

/-- `gather_rows_nat`, but the indices are provided as a `NatVecRef` leaf (eager backend). -/
def gatherRowsRef {α : Type} (s : EagerSession α) [Add α] [Zero α] [DecidableEq Shape]
  {rows cols k : Nat} (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim rows (.dim cols
    .scalar)))
  (idx : _root_.Runtime.Autograd.Torch.NatVecRef k) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (.dim k (.dim cols .scalar))) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.gatherRowsRef (α := α) (rows := rows) (cols
    := cols) (k := k) s.inner x idx

/--
Scatter-add into a vector at a `Fin` index (eager backend).

PyTorch analogy: `x.index_add_(dim=0, index=[i], source=v)` for a single index.
-/
def scatterAddVec {α : Type} (s : EagerSession α) [Add α] [Zero α] [DecidableEq Shape]
  {n : Nat} (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim n .scalar))
  (v : _root_.Runtime.Autograd.Torch.TensorRef α Shape.scalar) (i : Fin n) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (.dim n .scalar)) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.scatterAddVec (α := α) (n := n) s.inner x v
    i

/-- Scatter-add into a matrix row at a `Fin` index (eager backend). -/
def scatterAddRow {α : Type} (s : EagerSession α) [Add α] [Zero α] [DecidableEq Shape]
  {rows cols : Nat}
  (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim rows (.dim cols .scalar)))
  (v : _root_.Runtime.Autograd.Torch.TensorRef α (.dim cols .scalar)) (i : Fin rows) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (.dim rows (.dim cols .scalar))) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.scatterAddRow (α := α) (rows := rows) (cols
    := cols) s.inner x v i

/--
Fully-connected (affine) layer on vectors: `y = w·x + b` (eager backend).

PyTorch analogue: `torch.nn.functional.linear` (with weight shape `(outDim, inDim)`).
-/
def linear {α : Type} (s : EagerSession α) [Inhabited α] [Add α] [Mul α] [Zero α] [DecidableEq
  Shape] [_root_.Runtime.Autograd.FastKernels.FastMatmul α]
  {inDim outDim : Nat}
  (w : _root_.Runtime.Autograd.Torch.TensorRef α (.dim outDim (.dim inDim .scalar)))
  (b : _root_.Runtime.Autograd.Torch.TensorRef α (.dim outDim .scalar))
  (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim inDim .scalar)) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (.dim outDim .scalar)) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.linear (α := α) (inDim := inDim) (outDim :=
    outDim)
    s.inner w b x

/--
Mean squared error loss returning a scalar (eager backend).

PyTorch analogue: `torch.nn.functional.mse_loss(..., reduction='mean')`.
-/
def mseLoss {α : Type} (s : EagerSession α)
  [Inhabited α] [Add α] [Sub α] [Mul α] [Div α] [Zero α] [One α] [Coe Nat α] [DecidableEq Shape]
  [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
  {sh : Shape}
  (yhat target : _root_.Runtime.Autograd.Torch.TensorRef α sh) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α Shape.scalar) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.mseLoss (α := α) (sh := sh) s.inner yhat
    target

/--
LayerNorm over a `seqLen × embedDim` tensor (eager backend).

PyTorch analogue: `torch.nn.LayerNorm(embedDim)` applied per token.
-/
def layerNorm {α : Type} (s : EagerSession α) [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {seqLen embedDim : Nat} (h_seq_pos : seqLen > 0) (h_embed_pos : embedDim > 0)
  (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim seqLen (.dim embedDim .scalar)))
  (gamma : _root_.Runtime.Autograd.Torch.TensorRef α (.dim embedDim .scalar))
  (beta : _root_.Runtime.Autograd.Torch.TensorRef α (.dim embedDim .scalar)) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (.dim seqLen (.dim embedDim .scalar))) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.layerNorm (α := α)
    (seqLen := seqLen) (embedDim := embedDim) (h_seq_pos := h_seq_pos) (h_embed_pos := h_embed_pos)
    s.inner x gamma beta

/--
BatchNorm over a CHW tensor (eager backend).

PyTorch analogue: `torch.nn.BatchNorm2d` (channel-first layout).
-/
def batchnormChannelFirst {α : Type} (s : EagerSession α) [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {channels height width : Nat} (h_c : channels > 0) (h_h : height > 0) (h_w : width > 0)
  (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim channels (.dim height (.dim width .scalar))))
  (gamma : _root_.Runtime.Autograd.Torch.TensorRef α (.dim channels .scalar))
  (beta : _root_.Runtime.Autograd.Torch.TensorRef α (.dim channels .scalar)) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (.dim channels (.dim height (.dim width .scalar))))
    :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.batchnormChannelFirst (α := α)
    (channels := channels) (height := height) (width := width) (h_c := h_c) (h_h := h_h) (h_w :=
      h_w)
    s.inner x gamma beta

/--
N-D convolution over a channels-first tensor `(inC, spatial...)` (eager backend).

This is the generic counterpart to `conv2d`.

PyTorch analogue: `torch.nn.functional.conv{d}d` specialized to a single sample.
-/
def conv {α : Type} (s : EagerSession α) [Context α]
  [DecidableEq Shape]
  {d inC outC : Nat}
  {kernel stride padding : Vector Nat d}
  {inSpatial : Vector Nat d}
  {hInC : inC ≠ 0} {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
  (w : _root_.Runtime.Autograd.Torch.TensorRef α (Shape.ofList (outC :: inC :: kernel.toList)))
  (b : _root_.Runtime.Autograd.Torch.TensorRef α (.dim outC .scalar))
  (x : _root_.Runtime.Autograd.Torch.TensorRef α (Shape.ofList (inC :: inSpatial.toList))) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α
    (Shape.ofList (outC :: (Spec.convOutSpatial inSpatial kernel stride padding).toList))) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.conv (α := α)
    (d := d) (inC := inC) (outC := outC)
    (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial)
    (hInC := hInC) (hKernel := hKernel)
    s.inner w b x

/--
N-D transpose convolution over a channels-first tensor `(inC, spatial...)` (eager backend).

PyTorch analogue: `torch.nn.functional.conv_transpose{d}d` specialized to a single sample.
-/
def convTranspose {α : Type} (s : EagerSession α) [Context α]
  [DecidableEq Shape]
  {d inC outC : Nat}
  {kernel stride padding : Vector Nat d}
  {inSpatial : Vector Nat d}
  {hInC : inC ≠ 0} {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
  (w : _root_.Runtime.Autograd.Torch.TensorRef α (Shape.ofList (inC :: outC :: kernel.toList)))
  (b : _root_.Runtime.Autograd.Torch.TensorRef α (.dim outC .scalar))
  (x : _root_.Runtime.Autograd.Torch.TensorRef α (Shape.ofList (inC :: inSpatial.toList))) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α
    (Shape.ofList (outC :: (Spec.convTransposeOutSpatial inSpatial kernel stride padding).toList)))
    :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.convTranspose (α := α)
    (d := d) (inC := inC) (outC := outC)
    (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial)
    (hInC := hInC) (hKernel := hKernel)
    s.inner w b x

/--
2D convolution over a CHW tensor (eager backend).

PyTorch analogue: `torch.nn.functional.conv2d` (channel-first layout).
-/
def conv2d {α : Type} (s : EagerSession α) [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {inC outC kH kW stride padding inH inW : Nat}
  {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
  (kernel : _root_.Runtime.Autograd.Torch.TensorRef α (.dim outC (.dim inC (.dim kH (.dim kW
    .scalar)))))
  (bias : _root_.Runtime.Autograd.Torch.TensorRef α (.dim outC .scalar))
  (input : _root_.Runtime.Autograd.Torch.TensorRef α (.dim inC (.dim inH (.dim inW .scalar)))) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α
    (.dim outC (.dim ((inH + 2 * padding - kH) / stride + 1) (.dim ((inW + 2 * padding - kW) /
      stride + 1) .scalar)))) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.conv2d (α := α)
    (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding)
    (inH := inH) (inW := inW) (h1 := h1) (h2 := h2) (h3 := h3)
    s.inner kernel bias input

/--
2D transpose convolution over a CHW tensor (eager backend).

PyTorch analogue: `torch.nn.functional.conv_transpose2d` (channel-first layout).
-/
def convTranspose2d {α : Type} (s : EagerSession α) [Context α]
  [DecidableEq Shape]
  {inC outC kH kW stride padding inH inW : Nat}
  {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
  (kernel : _root_.Runtime.Autograd.Torch.TensorRef α (.dim inC (.dim outC (.dim kH (.dim kW
    .scalar)))))
  (bias : _root_.Runtime.Autograd.Torch.TensorRef α (.dim outC .scalar))
  (input : _root_.Runtime.Autograd.Torch.TensorRef α (.dim inC (.dim inH (.dim inW .scalar)))) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α
    (.dim outC (.dim ((inH - 1) * stride - 2 * padding + kH)
      (.dim ((inW - 1) * stride - 2 * padding + kW) .scalar)))) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.convTranspose2d (α := α)
    (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding)
    (inH := inH) (inW := inW) (h1 := h1) (h2 := h2) (h3 := h3)
    s.inner kernel bias input

/-- Alias for `conv2d` (compat shorthand). -/
abbrev conv2dCompat {α : Type} := conv2d (α := α)

/--
Multi-head self-attention (eager backend).

This is the eager-backend implementation used by the transformer examples (roughly analogous to
`torch.nn.MultiheadAttention` in self-attention mode).
-/
def multiHeadAttention {α : Type} (s : EagerSession α) [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {n numHeads dModel headDim : Nat} (h1 : n ≠ 0)
  (wq : _root_.Runtime.Autograd.Torch.TensorRef α (.dim dModel (.dim (numHeads * headDim) .scalar)))
  (wk : _root_.Runtime.Autograd.Torch.TensorRef α (.dim dModel (.dim (numHeads * headDim) .scalar)))
  (wv : _root_.Runtime.Autograd.Torch.TensorRef α (.dim dModel (.dim (numHeads * headDim) .scalar)))
  (wo : _root_.Runtime.Autograd.Torch.TensorRef α (.dim (numHeads * headDim) (.dim dModel .scalar)))
  (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim n (.dim dModel .scalar)))
  (mask : Option (Tensor Bool (.dim n (.dim n .scalar))) := none) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (.dim n (.dim dModel .scalar))) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.multiHeadAttention (α := α)
    (n := n) (numHeads := numHeads) (dModel := dModel) (headDim := headDim) (h1 := h1)
    s.inner wq wk wv wo x (mask := mask)

/--
Run a backward pass and return dense gradients for all leaves (eager backend).

See the unified version `Session.backwardDenseAll` for the public API.
-/
def backwardDenseAll {α : Type} (s : EagerSession α) [Add α] [Zero α] [DecidableEq Shape]
  [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
  {sh : Shape} (out : _root_.Runtime.Autograd.Torch.TensorRef α sh) (seed : Tensor α sh) :
  IO (Array (Runtime.AnyTensor α)) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.backwardDenseAll (α := α) (sh := sh) s.inner
    out seed

/-- Backward pass specialized to scalar losses (seed is implicitly `1`) (eager backend). -/
def backwardScalarDenseAll {α : Type} (s : EagerSession α) [Add α] [Zero α] [One α] [DecidableEq
  Shape]
  [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
  (loss : _root_.Runtime.Autograd.Torch.TensorRef α Shape.scalar) :
  IO (Array (Runtime.AnyTensor α)) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.backwardScalarDenseAll (α := α) s.inner loss

/--
Apply an SGD step to all learnable parameters given a dense gradient array (eager backend).

PyTorch analogy: `optimizer.step()` for an SGD optimizer, with gradients supplied explicitly.
-/
def sgdStepAll {α : Type} (s : EagerSession α)
  [Sub α] [Mul α] [Add α] [Zero α] [DecidableEq Shape]
  [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
  (lr : α) (grads : Array (Runtime.AnyTensor α)) : IO Unit :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.sgdStepAll (α := α) s.inner lr grads

end EagerSession

/--
Implementation choice for a `Session`.

This is an internal sum type used to dispatch each operation to either:
- the eager tape-backed runtime (`.eager`), or
- the proof-linked compiled runtime (`.compiled`).
-/
inductive SessionImpl (α : Type) where
  | eager (s : EagerSession α)
  | compiled (s : _root_.Runtime.Autograd.Torch.Internal.SessionIR α)

/--
Unified imperative session: choose `.eager` vs `.compiled` at construction via `opts.backend`.

This is the recommended "one interface" for:
- training/debugging (eager),
- verification-friendly execution (compiled/proof-linked),
without users having to learn two different Session APIs.
-/
structure Session (α : Type) where
  /-- opts. -/
  opts : _root_.Runtime.Autograd.Torch.Options
  /-- impl. -/
  impl : SessionImpl α

namespace Session

/--
Create a new unified session.

The backend is selected by `opts.backend`:
- `.eager` builds a tape-backed runtime session, and
- `.compiled` builds a proof-linked compiled session.
-/
def new {α : Type} (opts : _root_.Runtime.Autograd.Torch.Options := {}) : IO (Session α) := do
  match opts.backend with
  | .eager =>
      let s ← EagerSession.new (α := α) (opts := opts)
      pure { opts := opts, impl := .eager s }
  | .compiled =>
      let s ← _root_.Runtime.Autograd.Torch.Internal.SessionIR.new (α := α) (opts := opts)
      pure { opts := opts, impl := .compiled s }

/-- Reset the autograd tape / graph-building state. -/
def resetTape {α : Type} (s : Session α) : IO Unit := do
  match s.impl with
  | .eager sess => EagerSession.resetTape (α := α) sess
  | .compiled sess => _root_.Runtime.Autograd.Torch.Internal.SessionIR.resetTape (α := α) sess

/--
Create a learnable parameter (a leaf tensor) owned by the session.

PyTorch analogue: `torch.nn.Parameter` (conceptually), created inside a module/init and later used
in forward passes.
-/
def param {α : Type} (s : Session α) {sh : Shape}
  (init : Tensor α sh) (name : Option String := none) (requiresGrad : Option Bool := none) :
  IO (_root_.Runtime.Autograd.Torch.Param α sh) := do
  match s.impl with
  | .eager sess =>
      EagerSession.param (α := α) sess (sh := sh)
        init (name := name) (requiresGrad := requiresGrad)
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.param (α := α) sess (sh := sh)
        init (name := name) (requiresGrad := requiresGrad)

/--
Use a parameter inside the current tape/graph.

This returns a `TensorRef` that can be passed to ops to build a forward graph.
-/
def use {α : Type} (s : Session α) {sh : Shape} [DecidableEq Shape]
  [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
  (p : _root_.Runtime.Autograd.Torch.Param α sh) : IO (_root_.Runtime.Autograd.Torch.TensorRef α sh)
    := do
  match s.impl with
  | .eager sess => EagerSession.use (α := α) sess (sh := sh) p
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.use (α := α) sess (sh := sh) p

/--
Add an input tensor to the current tape/graph.

Inputs are leaf tensors that may or may not require gradients (controlled by `requiresGrad`).
-/
def input {α : Type} (s : Session α) {sh : Shape} [DecidableEq Shape]
  [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
  (v : Tensor α sh) (name : Option String := none) (requiresGrad : Bool := false) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α sh) := do
  match s.impl with
  | .eager sess =>
      EagerSession.input (α := α) sess (sh := sh) v (name := name) (requiresGrad := requiresGrad)
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.input (α := α) sess (sh := sh)
        v (name := name) (requiresGrad := requiresGrad)

/--
Add a non-differentiable `Nat` input to the session.

This is used for labels/indices (e.g. classification targets, gather indices) without forcing a
numeric embedding into `α`.
-/
def inputNat {α : Type} (s : Session α) (v : Nat) : IO (_root_.Runtime.Autograd.Torch.NatRef) := do
  match s.impl with
  | .eager sess => EagerSession.inputNat (α := α) sess v
  | .compiled sess => _root_.Runtime.Autograd.Torch.Internal.SessionIR.inputNat (α := α) sess v

/-- Read back a `NatRef` value. -/
def getNat {α : Type} (s : Session α) (r : _root_.Runtime.Autograd.Torch.NatRef) : IO Nat := do
  match s.impl with
  | .eager sess => EagerSession.getNat (α := α) sess r
  | .compiled sess => _root_.Runtime.Autograd.Torch.Internal.SessionIR.getNat (α := α) sess r

/-- Mutate a `NatRef` value. -/
def setNat {α : Type} (s : Session α) (r : _root_.Runtime.Autograd.Torch.NatRef) (v : Nat) : IO Unit
  := do
  match s.impl with
  | .eager sess => EagerSession.setNat (α := α) sess r v
  | .compiled sess => _root_.Runtime.Autograd.Torch.Internal.SessionIR.setNat (α := α) sess r v

/--
Add a non-differentiable vector-of-`Nat` input leaf.

This is convenient for batched indices (e.g. gather a batch of rows) without embedding indices into
  `α`.
-/
def inputNatVec {α : Type} {k : Nat} (s : Session α) (v : Tensor Nat (.dim k .scalar)) :
    IO (_root_.Runtime.Autograd.Torch.NatVecRef k) := do
  match s.impl with
  | .eager sess => EagerSession.inputNatVec (α := α) (k := k) sess v
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.inputNatVec (α := α) (k := k) sess v

/-- Read back a `NatVecRef` value. -/
def getNatVec {α : Type} {k : Nat} (s : Session α) (r : _root_.Runtime.Autograd.Torch.NatVecRef k) :
    IO (Tensor Nat (.dim k .scalar)) := do
  match s.impl with
  | .eager sess => EagerSession.getNatVec (α := α) (k := k) sess r
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.getNatVec (α := α) (k := k) sess r

/-- Mutate a `NatVecRef` value. -/
def setNatVec {α : Type} {k : Nat} (s : Session α) (r : _root_.Runtime.Autograd.Torch.NatVecRef k)
    (v : Tensor Nat (.dim k .scalar)) : IO Unit := do
  match s.impl with
  | .eager sess => EagerSession.setNatVec (α := α) (k := k) sess r v
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.setNatVec (α := α) (k := k) sess r v

/-! ### Deterministic RNG state (Session-level) -/

/--
Deterministic RNG state stored inside a session.

We model RNG state explicitly using two non-differentiable leaves:
- `seed`: the current seed value
- `counter`: a monotone counter used to derive fresh keys

PyTorch analogy: explicit `torch.manual_seed` + per-op counter, but represented as explicit state.
-/
structure RngState where
  /-- Random seed. -/
  seed : _root_.Runtime.Autograd.Torch.NatRef
  /-- counter. -/
  counter : _root_.Runtime.Autograd.Torch.NatRef

/--
Initialize an `RngState` from a concrete seed.

This allocates two `NatRef`s in the session (`seed` and `counter`) and initializes `counter` to 0.
-/
def initRng {α : Type} (s : Session α) (seed : Nat) : IO RngState := do
  let seedRef ← inputNat (α := α) s seed
  let counterRef ← inputNat (α := α) s 0
  pure { seed := seedRef, counter := counterRef }

/-- Draw a fresh seed from `IO` (best-effort entropy). -/
def freshSeedIO : IO Nat := do
  -- We use `IO.rand` for practicality/ergonomics; this is *not* part of the semantic core.
  -- The semantic model remains seed-threaded deterministic RNG: this just chooses an initial seed.
  IO.rand 0 (Nat.pow 2 63 - 1)

/--
Initialize a deterministic RNG state by sampling an initial seed from `IO`.

This is the recommended "PyTorch-like ergonomics, JAX-like semantics" bridge:
- you get a convenient source of entropy at the boundary,
- but the *core* semantics remains deterministic and replayable given the chosen seed.
-/
def initRngFromIO {α : Type} (s : Session α) : IO RngState := do
  initRng (α := α) s (← freshSeedIO)

/-!
Practical note: in the proof-linked `.compiled` backend, the current session implementation
requires that all leaves (tensor inputs/parameters and `NatRef`s) are created before any op nodes.
So for maximum portability, initialize and split RNG states *up-front* before building a graph.
-/

/--
Split an RNG stream into a fresh child stream (deterministic).

This is useful for isolating submodules (e.g. separate dropout sites) without sharing RNG state.
-/
def splitRng {α : Type} (s : Session α) (rng : RngState) : IO RngState := do
  let seedNat ← getNat (α := α) s rng.seed
  let ctrNat ← getNat (α := α) s rng.counter
  -- Derive two fresh seeds deterministically.
  let childSeed := Random.nextSeed seedNat ctrNat
  let parentSeed := Random.nextSeed childSeed (ctrNat + 1)
  setNat (α := α) s rng.seed parentSeed
  setNat (α := α) s rng.counter (ctrNat + 2)
  initRng (α := α) s childSeed

/--
Insert a constant tensor into the current graph.

PyTorch analogy: using a tensor constant/literal in `forward`.
-/
def const {α : Type} (s : Session α) {sh : Shape} [Zero α] [DecidableEq Shape]
  [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
  (v : Tensor α sh) (name : Option String := none) : IO (_root_.Runtime.Autograd.Torch.TensorRef α
    sh) := do
  match s.impl with
  | .eager sess => EagerSession.const (α := α) sess (sh := sh) v (name := name)
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.const (α := α) sess (sh := sh) v
        (name := name)

/-- Read the concrete value for a tensor ref (for logging/debugging). -/
def getValue {α : Type} (s : Session α) {sh : Shape} [DecidableEq Shape]
  [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
  (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) : IO (Tensor α sh) := do
  match s.impl with
  | .eager sess => EagerSession.getValue (α := α) sess (sh := sh) x
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.getValue (α := α) sess (sh := sh) x

/--
Detach a tensor ref from the graph (stop gradient flow through it).

PyTorch analogy: `x.detach()`.
-/
def detach {α : Type} (s : Session α) [Context α] {sh : Shape} [DecidableEq Shape]
    [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) :
    IO (_root_.Runtime.Autograd.Torch.TensorRef α sh) := do
  match s.impl with
  | .eager sess => EagerSession.detach (α := α) sess (sh := sh) x
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.detach (α := α) sess (sh := sh) x

/-- Elementwise addition (dispatches to eager vs compiled backend). -/
def add {α : Type} (s : Session α) [Add α] [Zero α] [DecidableEq Shape] {sh : Shape}
  (a b : _root_.Runtime.Autograd.Torch.TensorRef α sh) : IO (_root_.Runtime.Autograd.Torch.TensorRef
    α sh) := do
  match s.impl with
  | .eager sess => EagerSession.add (α := α) sess (sh := sh) a b
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.add (α := α) sess (sh := sh) a b

/-- Elementwise subtraction (dispatches to eager vs compiled backend). -/
def sub {α : Type} (s : Session α) [Sub α] [Add α] [Zero α] [DecidableEq Shape] {sh : Shape}
  (a b : _root_.Runtime.Autograd.Torch.TensorRef α sh) : IO (_root_.Runtime.Autograd.Torch.TensorRef
    α sh) := do
  match s.impl with
  | .eager sess => EagerSession.sub (α := α) sess (sh := sh) a b
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.sub (α := α) sess (sh := sh) a b

/-- Elementwise multiplication (dispatches to eager vs compiled backend). -/
def mul {α : Type} (s : Session α) [Mul α] [Add α] [Zero α] [DecidableEq Shape] {sh : Shape}
  (a b : _root_.Runtime.Autograd.Torch.TensorRef α sh) : IO (_root_.Runtime.Autograd.Torch.TensorRef
    α sh) := do
  match s.impl with
  | .eager sess => EagerSession.mul (α := α) sess (sh := sh) a b
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.mul (α := α) sess (sh := sh) a b

/--
Scale a tensor by a scalar constant `c` (elementwise).

PyTorch analogy: `x * c` or `torch.mul(x, c)`.
-/
def scale {α : Type} (s : Session α) [Mul α] [Add α] [Zero α] [DecidableEq Shape]
  [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α] {sh : Shape}
  (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) (c : α) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α sh) := do
  match s.impl with
  | .eager sess => EagerSession.scale (α := α) sess (sh := sh) x c
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.scale (α := α) sess (sh := sh) x c

/--
Dropout implemented as a Session-level derived op.

In training mode, this records:

`y = x * mask / (1 - p)`

where `mask : {0,1}` is generated by the Torch primitive `bernoulli_mask`.

## RNG Semantics (JAX-style / functional RNG)

Randomness is a deterministic function of explicit session state:
- we read `(seed, counter)` from the session-level `RngState`,
- derive a per-call `opSeed` deterministically, and
- advance the `RngState` (update `seed`, increment `counter`).

Important: the `bernoulli_mask` op also mixes in a backend-internal counter (roughly: current tape
size / node index). So even with the same `RngState`, changing the surrounding graph structure can
change the exact samples. This is still fully deterministic for a fixed graph.

In evaluation mode (`train=false`), this is the identity.

`p` is expected to satisfy `0 ≤ p < 1`; we throw if `keepProb = 1 - p` is not strictly positive.
-/
def dropout {α : Type} [Context α] [DecidableEq Shape]
    [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    (s : Session α) (rng : RngState) {sh : Shape}
    (x : _root_.Runtime.Autograd.Torch.TensorRef α sh)
    (p : α) (train : Bool := true) :
    IO (_root_.Runtime.Autograd.Torch.TensorRef α sh) := do
  if !train then
    pure x
  else
    let keepProb : α := (1 : α) - p
    have : Decidable (keepProb > (0 : α)) := (Context.decidable_gt) keepProb 0
    if _h : keepProb > (0 : α) then
      let seedNat ← getNat (α := α) s rng.seed
      let ctrNat ← getNat (α := α) s rng.counter
      -- Seed is derived deterministically from the explicit `(seed, counter)` state.
      -- Note: `Torch.bernoulli_mask` also mixes in an internal per-graph counter (tape size / node
      -- index), so graph structure can affect the exact samples even with the same `RngState`.
      let opSeed := Random.nextSeed seedNat ctrNat
      setNat (α := α) s rng.seed opSeed
      setNat (α := α) s rng.counter (ctrNat + 1)

      let keepProbRef ← const (α := α) s (sh := Shape.scalar) (Tensor.scalar keepProb)
      let maskRef ←
        match s.impl with
        | .eager sess =>
            _root_.Runtime.Autograd.Torch.Internal.EagerSession.bernoulliMask (α := α) sess.inner
              (sh := sh) keepProbRef opSeed
        | .compiled sess =>
            _root_.Runtime.Autograd.Torch.Internal.SessionIR.commitGraphM (α := α) sess
              (β := _root_.Runtime.Autograd.Torch.TensorRef α sh) (fun {Γ} {ss} xv nat g => do
                let (v, st') ← _root_.Runtime.Autograd.Torch.Internal.SessionIR.runGraphM (α := α)
                  (Γ := Γ)
                  (Runtime.Autograd.Compiled.GraphM.bernoulliMask (α := α) (Γ := Γ) (s := sh)
                    { id := keepProbRef.id } (seed := opSeed))
                  ss g
                let ⟨ss', g'⟩ := st'
                let st1 : _root_.Runtime.Autograd.Torch.Internal.SessionIRState α :=
                  { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
                pure ({ id := v.id }, st1))

      let y ← mul (α := α) s (sh := sh) x maskRef
      let invKeep : α := (1 : α) / keepProb
      scale (α := α) s (sh := sh) y invKeep
    else
      throw <| IO.userError "dropout: expected (1 - p) > 0"

/-- Elementwise absolute value (dispatches to eager vs compiled backend). -/
def abs {α : Type} (s : Session α) [Context α] [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq
  Shape]
  {sh : Shape} (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α sh) := do
  match s.impl with
  | .eager sess => EagerSession.abs (α := α) sess (sh := sh) x
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.abs (α := α) sess (sh := sh) x

/-- Elementwise square root (dispatches to eager vs compiled backend). -/
def sqrt {α : Type} (s : Session α) [Context α] [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq
  Shape]
  {sh : Shape} (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α sh) := do
  match s.impl with
  | .eager sess => EagerSession.sqrt (α := α) sess (sh := sh) x
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.sqrt (α := α) sess (sh := sh) x

/-- Elementwise clamp to `[minVal, maxVal]` (dispatches to eager vs compiled backend). -/
def clamp {α : Type} (s : Session α) [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  [DecidableEq Shape]
  [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
  {sh : Shape} (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) (minVal maxVal : α) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α sh) := do
  match s.impl with
  | .eager sess => EagerSession.clamp (α := α) sess (sh := sh) x minVal maxVal
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.clamp (α := α) sess
        (sh := sh) x minVal maxVal

/-- Elementwise maximum (dispatches to eager vs compiled backend). -/
def max {α : Type} (s : Session α) [Context α] [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq
  Shape]
  {sh : Shape} (a b : _root_.Runtime.Autograd.Torch.TensorRef α sh) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α sh) := do
  match s.impl with
  | .eager sess => EagerSession.max (α := α) sess (sh := sh) a b
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.max (α := α) sess (sh := sh) a b

/-- Elementwise minimum (dispatches to eager vs compiled backend). -/
def min {α : Type} (s : Session α) [Context α] [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq
  Shape]
  {sh : Shape} (a b : _root_.Runtime.Autograd.Torch.TensorRef α sh) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α sh) := do
  match s.impl with
  | .eager sess => EagerSession.min (α := α) sess (sh := sh) a b
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.min (α := α) sess (sh := sh) a b

/--
Matrix multiplication (rank-2 tensors).

PyTorch analogy: `torch.matmul` / `@` for matrices.
-/
def matmul {α : Type} (s : Session α) [Context α] [DecidableEq Shape]
  {m n p : Nat}
  (a : _root_.Runtime.Autograd.Torch.TensorRef α (.dim m (.dim n .scalar)))
  (b : _root_.Runtime.Autograd.Torch.TensorRef α (.dim n (.dim p .scalar))) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (.dim m (.dim p .scalar))) := do
  match s.impl with
  | .eager sess => EagerSession.matmul (α := α) sess (m := m) (n := n) (p := p) a b
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.matmul (α := α) sess
        (m := m) (n := n) (p := p) a b

/--
Batched matrix multiplication (rank-3 tensors).

PyTorch analogy: `torch.bmm`.
-/
def bmm {α : Type} (s : Session α) [Add α] [Mul α] [Zero α] [DecidableEq Shape]
  {batch m n p : Nat}
  (a : _root_.Runtime.Autograd.Torch.TensorRef α (.dim batch (.dim m (.dim n .scalar))))
  (b : _root_.Runtime.Autograd.Torch.TensorRef α (.dim batch (.dim n (.dim p .scalar)))) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (.dim batch (.dim m (.dim p .scalar)))) := do
  match s.impl with
  | .eager sess => EagerSession.bmm (α := α) sess (batch := batch) (m := m) (n := n) (p := p) a b
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.bmm (α := α) sess (batch := batch) (m := m)
        (n := n) (p := p) a b

/-- Concatenate two vectors along dimension 0 (dispatches to eager vs compiled backend). -/
def concatVectors {α : Type} (s : Session α) [Context α] [DecidableEq Shape]
  {n m : Nat}
  (a : _root_.Runtime.Autograd.Torch.TensorRef α (.dim n .scalar))
  (b : _root_.Runtime.Autograd.Torch.TensorRef α (.dim m .scalar)) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (.dim (n + m) .scalar)) := do
  match s.impl with
  | .eager sess => EagerSession.concatVectors (α := α) sess (n := n) (m := m) a b
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.concatVectors (α := α) sess
        (n := n) (m := m) a b

/-- Concatenate along the outermost dimension (dimension 0) (dispatches to eager vs compiled
  backend). -/
def concatDim0 {α : Type} (s : Session α) [Context α] [DecidableEq Shape]
  {n m : Nat} {sh : Shape}
  (a : _root_.Runtime.Autograd.Torch.TensorRef α (.dim n sh))
  (b : _root_.Runtime.Autograd.Torch.TensorRef α (.dim m sh)) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (.dim (n + m) sh)) := do
  match s.impl with
  | .eager sess => EagerSession.concatDim0 (α := α) sess (n := n) (m := m) (sh := sh) a b
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.concatDim0 (α := α) sess (n := n) (m := m)
        (sh := sh) a b

/--
Slice a contiguous `[start, start+len)` range from dimension 0.

PyTorch analogy: `x[start:start+len]` for the first dimension.
-/
def sliceRange0 {α : Type} (s : Session α) [Zero α] [DecidableEq Shape]
  {n : Nat} {sh : Shape}
  (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim n sh)) (start len : Nat) (h : len + start ≤
    n) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (.dim len sh)) := do
  match s.impl with
  | .eager sess => EagerSession.sliceRange0 (α := α) sess (n := n) (sh := sh) x start len h
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.sliceRange0 (α := α) sess
        (n := n) (sh := sh) x start len h

/--
2D max pooling on a CHW tensor.

PyTorch analogy: `torch.nn.functional.max_pool2d` (channel-first layout).
-/
def maxPool2d {α : Type} (s : Session α) [Context α] [DecidableEq Shape]
  {kH kW inH inW inC stride : Nat} {h1 : kH ≠ 0} {h2 : kW ≠ 0}
  (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim inC (.dim inH (.dim inW .scalar)))) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α
    (.dim inC (.dim ((inH - kH) / stride + 1) (.dim ((inW - kW) / stride + 1) .scalar)))) := do
  match s.impl with
  | .eager sess =>
      EagerSession.maxPool2d (α := α) sess (kH := kH) (kW := kW) (inH := inH) (inW := inW)
        (inC := inC) (stride := stride) (h1 := h1) (h2 := h2) x
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.maxPool2d (α := α) sess
        (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride)
        (h1 := h1) (h2 := h2) x

/-- Alias for `max_pool2d` (PyTorch-style shorthand). -/
abbrev maxPool {α : Type} := maxPool2d (α := α)

/--
Smooth max pooling (differentiable surrogate for max pooling) on a CHW tensor.

This is parameterized by `beta` (larger values behave more like true max pooling).
-/
def smoothMaxPool2d {α : Type} (s : Session α) [Context α] [DecidableEq Shape]
  [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
  {kH kW inH inW inC stride : Nat} {h1 : kH ≠ 0} {h2 : kW ≠ 0}
  (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim inC (.dim inH (.dim inW .scalar)))) (beta :
    α) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α
    (.dim inC (.dim ((inH - kH) / stride + 1) (.dim ((inW - kW) / stride + 1) .scalar)))) := do
  match s.impl with
  | .eager sess =>
      EagerSession.smoothMaxPool2d (α := α) sess (kH := kH) (kW := kW) (inH := inH) (inW := inW)
        (inC := inC) (stride := stride) (h1 := h1) (h2 := h2) x beta
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.smoothMaxPool2d (α := α) sess
        (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride)
        (h1 := h1) (h2 := h2) x beta

/-- Alias for `smooth_max_pool2d` (PyTorch-style shorthand). -/
abbrev smoothMaxPool {α : Type} := smoothMaxPool2d (α := α)

/--
2D average pooling on a CHW tensor.

PyTorch analogy: `torch.nn.functional.avg_pool2d` (channel-first layout).
-/
def avgPool2d {α : Type} (s : Session α) [Context α] [DecidableEq Shape]
  {kH kW inH inW inC stride : Nat} (h1 : kH ≠ 0) (h2 : kW ≠ 0)
  (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim inC (.dim inH (.dim inW .scalar)))) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α
    (.dim inC (.dim ((inH - kH) / stride + 1) (.dim ((inW - kW) / stride + 1) .scalar)))) := do
  match s.impl with
  | .eager sess =>
      EagerSession.avgPool2d (α := α) sess (kH := kH) (kW := kW) (inH := inH) (inW := inW)
        (inC := inC) (stride := stride) h1 h2 x
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.avgPool2d (α := α) sess
        (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride)
        h1 h2 x

/-- Alias for `avg_pool2d` (PyTorch-style shorthand). -/
abbrev avgPool {α : Type} := avgPool2d (α := α)

/--
Rectified Linear Unit (ReLU) activation.

This is a pointwise nonlinearity, `relu(x) = max(x, 0)`, recorded as part of the session’s autograd
graph.

PyTorch analogy: `torch.relu(x)` / `torch.nn.functional.relu(x)`.
-/
def relu {α : Type} (s : Session α)
  [Mul α] [Add α] [Zero α] [Max α] [One α] [LT α]
  [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {sh : Shape} (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α sh) := do
  match s.impl with
  | .eager sess => EagerSession.relu (α := α) sess (sh := sh) x
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.relu (α := α) sess (sh := sh) x

/--
Sigmoid (logistic) activation, applied pointwise.

PyTorch analogy: `torch.sigmoid(x)`.
-/
def sigmoid {α : Type} (s : Session α) [Context α] [DecidableEq Shape]
  {sh : Shape} (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α sh) := do
  match s.impl with
  | .eager sess => EagerSession.sigmoid (α := α) sess (sh := sh) x
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.sigmoid (α := α) sess (sh := sh) x

/--
Hyperbolic tangent activation, applied pointwise.

PyTorch analogy: `torch.tanh(x)`.
-/
def tanh {α : Type} (s : Session α) [Context α] [DecidableEq Shape]
  {sh : Shape} (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α sh) := do
  match s.impl with
  | .eager sess => EagerSession.tanh (α := α) sess (sh := sh) x
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.tanh (α := α) sess (sh := sh) x

/--
Softmax along the last axis (recursing over outer dimensions), shape-preserving.

This matches the spec-layer `Activation.softmax_spec` and uses a standard VJP implementation in the
backend (so we do not materialize an explicit Jacobian).

PyTorch analogy: `torch.softmax(x, dim=-1)`.
-/
def softmax {α : Type} (s : Session α) [Context α] [DecidableEq Shape]
  {sh : Shape} (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α sh) := do
  match s.impl with
  | .eager sess => EagerSession.softmax (α := α) sess (sh := sh) x
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.softmax (α := α) sess (sh := sh) x

/--
Stable log-softmax along the last axis.

PyTorch analogy: `torch.nn.functional.log_softmax(x, dim=-1)`.
-/
def logSoftmax {α : Type} (s : Session α) [Context α] [DecidableEq Shape]
  {sh : Shape} (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α sh) := do
  match s.impl with
  | .eager sess => EagerSession.logSoftmax (α := α) sess (sh := sh) x
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.logSoftmax (α := α) sess (sh := sh) x

/--
Softplus activation, applied pointwise: `softplus(x) = log(1 + exp(x))`.

PyTorch analogy: `torch.nn.functional.softplus(x)`.
-/
def softplus {α : Type} (s : Session α) [Context α] [DecidableEq Shape]
  {sh : Shape} (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α sh) := do
  match s.impl with
  | .eager sess => EagerSession.softplus (α := α) sess (sh := sh) x
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.softplus (α := α) sess (sh := sh) x

/--
Elementwise exponential.

PyTorch analogy: `torch.exp(x)`.
-/
def exp {α : Type} (s : Session α) [Context α] [DecidableEq Shape]
  {sh : Shape} (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α sh) := do
  match s.impl with
  | .eager sess => EagerSession.exp (α := α) sess (sh := sh) x
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.exp (α := α) sess (sh := sh) x

/--
Elementwise natural logarithm.

PyTorch analogy: `torch.log(x)`.

If you need a total (always-defined) "log-like" surrogate without positivity side conditions, see
`safe_log`.
-/
def log {α : Type} (s : Session α) [Context α] [DecidableEq Shape]
  {sh : Shape} (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α sh) := do
  match s.impl with
  | .eager sess => EagerSession.log (α := α) sess (sh := sh) x
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.log (α := α) sess (sh := sh) x

/--
Elementwise "safe log" surrogate: `safe_log(x; ε) = log(softplus(x) + ε)`.

We use this when we want something log-like but would rather not carry side conditions about inputs
being strictly positive.

PyTorch analogy: `torch.log(torch.nn.functional.softplus(x) + eps)`.
-/
def safeLog {α : Type} (s : Session α) [Context α] [DecidableEq Shape]
  [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
  {sh : Shape} (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) (ε : α := Numbers.epsilon) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α sh) := do
  match s.impl with
  | .eager sess => EagerSession.safeLog (α := α) sess (sh := sh) x (ε := ε)
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.safeLog (α := α) sess (sh := sh) x (ε := ε)

/--
Sum-reduce all elements of a tensor to a scalar.

PyTorch analogy: `x.sum()` (with no `dim` argument).
-/
def sum {α : Type} (s : Session α) [Context α] [DecidableEq Shape] {sh : Shape}
  (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α Shape.scalar) := do
  match s.impl with
  | .eager sess =>
      EagerSession.sum (α := α) sess (sh := sh) x
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.sum (α := α) sess (sh := sh) x

/--
Flatten a tensor to a 1D vector of length `Shape.size sh`.

PyTorch analogy: `torch.flatten(x)` or `x.reshape(-1)`.
-/
def flatten {α : Type} (s : Session α) [Inhabited α] [Zero α] [DecidableEq Shape] {sh : Shape}
  (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (.dim (Shape.size sh) .scalar)) := do
  match s.impl with
  | .eager sess => EagerSession.flatten (α := α) sess (sh := sh) x
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.flatten (α := α) sess (sh := sh) x

/--
Reshape a tensor without changing the number of elements.

The proof `h : Shape.size sh1 = Shape.size sh2` plays the role of PyTorch’s runtime check performed
by `reshape`/`view`.

PyTorch analogy: `x.reshape(new_shape)` (when the element count matches).
-/
def reshape {α : Type} (s : Session α) [Inhabited α] [Zero α] [DecidableEq Shape] {sh1 sh2 : Shape}
  (x : _root_.Runtime.Autograd.Torch.TensorRef α sh1) (h : Shape.size sh1 = Shape.size sh2) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α sh2) := do
  match s.impl with
  | .eager sess => EagerSession.reshape (α := α) sess (sh1 := sh1) (sh2 := sh2) x h
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.reshape (α := α) sess
        (sh1 := sh1) (sh2 := sh2) x h

/--
Transpose a rank-2 tensor (matrix transpose): `m×n → n×m`.

PyTorch analogy: `x.transpose(0, 1)` (or `x.T` for 2D tensors).
-/
def transpose2d {α : Type} (s : Session α) [Zero α] [DecidableEq Shape] {m n : Nat}
  (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim m (.dim n .scalar))) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (.dim n (.dim m .scalar))) := do
  match s.impl with
  | .eager sess => EagerSession.transpose2d (α := α) sess (m := m) (n := n) x
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.transpose2d (α := α) sess (m := m) (n := n) x

/--
Permute a rank-3 tensor by moving the first axis to the end: `(a, b, c) → (b, c, a)`.

PyTorch analogy: `x.permute(1, 2, 0)`.
-/
def transpose3dFirstToLast {α : Type} (s : Session α) [Zero α] [DecidableEq Shape] {a b c : Nat}
  (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim a (.dim b (.dim c .scalar)))) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (.dim b (.dim c (.dim a .scalar)))) := do
  match s.impl with
  | .eager sess => EagerSession.transpose3dFirstToLast (α := α) sess (a := a) (b := b) (c := c) x
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.transpose3dFirstToLast (α := α) sess
        (a := a) (b := b) (c := c) x

/--
Permute a rank-3 tensor by moving the last axis to the front: `(a, b, c) → (c, a, b)`.

PyTorch analogy: `x.permute(2, 0, 1)`.
-/
def transpose3dLastToFirst {α : Type} (s : Session α) [Zero α] [DecidableEq Shape] {a b c : Nat}
  (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim a (.dim b (.dim c .scalar)))) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (.dim c (.dim a (.dim b .scalar)))) := do
  match s.impl with
  | .eager sess => EagerSession.transpose3dLastToFirst (α := α) sess (a := a) (b := b) (c := c) x
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.transpose3dLastToFirst (α := α) sess
        (a := a) (b := b) (c := c) x

/--
Swap the last two axes of a rank-3 tensor: `(a, b, c) → (a, c, b)`.

PyTorch analogy: `x.permute(0, 2, 1)`.
-/
def transpose3dLastTwo {α : Type} (s : Session α) [Zero α] [DecidableEq Shape] {a b c : Nat}
  (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim a (.dim b (.dim c .scalar)))) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (.dim a (.dim c (.dim b .scalar)))) := do
  match s.impl with
  | .eager sess => EagerSession.transpose3dLastTwo (α := α) sess (a := a) (b := b) (c := c) x
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.transpose3dLastTwo (α := α) sess (a := a)
        (b := b) (c := c) x

/--
Generic "swap adjacent axes" view operation.

This is a shape-driven permutation helper used in some attention/transformer code.
-/
def swapAdjacentAtDepth {α : Type} (s : Session α) [Context α] [DecidableEq Shape] {sh : Shape}
  (depth : Nat) (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (sh.swapAdjacentAtDepth depth)) := do
  match s.impl with
  | .eager sess => EagerSession.swapAdjacentAtDepth (α := α) sess (sh := sh) depth x
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.swapAdjacentAtDepth (α := α) sess
        (sh := sh) depth x

/-- Broadcast a tensor to a larger shape (dispatches to eager vs compiled backend). -/
def broadcastTo {α : Type} (s : Session α) [Inhabited α] [Add α] [Zero α] [DecidableEq Shape]
  {sh1 sh2 : Shape} (cb : Shape.CanBroadcastTo sh1 sh2) (x : _root_.Runtime.Autograd.Torch.TensorRef
    α sh1) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α sh2) := do
  match s.impl with
  | .eager sess => EagerSession.broadcastTo (α := α) sess (sh1 := sh1) (sh2 := sh2) cb x
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.broadcastTo (α := α) sess (sh1 := sh1)
        (sh2 := sh2) cb x

/-- Reduce-sum along an axis (dispatches to eager vs compiled backend). -/
def reduceSum {α : Type} (s : Session α) [Add α] [Zero α] [Inhabited α] [DecidableEq Shape]
  {sh : Shape} (axis : Nat) [valid : Shape.valid_axis_inst axis sh] [wf : Shape.WellFormed sh]
  (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (shapeAfterSum sh axis)) := do
  match s.impl with
  | .eager sess => EagerSession.reduceSum (α := α) sess (sh := sh) axis x
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.reduceSum (α := α) sess (sh := sh) axis x

/-- Reduce-mean along an axis (dispatches to eager vs compiled backend). -/
def reduceMean {α : Type} (s : Session α) [Context α] [Inhabited α] [DecidableEq Shape]
  {sh : Shape} (axis : Nat) [valid : Shape.valid_axis_inst axis sh] [wf : Shape.WellFormed sh]
  (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (shapeAfterSum sh axis)) := do
  match s.impl with
  | .eager sess => EagerSession.reduceMean (α := α) sess (sh := sh) axis x
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.reduceMean (α := α) sess (sh := sh) axis x

/-- Gather a scalar from a vector at a `Fin` index (dispatches to eager vs compiled backend). -/
def gatherScalar {α : Type} (s : Session α) [Zero α] [DecidableEq Shape]
  {n : Nat} (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim n .scalar)) (i : Fin n) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α Shape.scalar) := do
  match s.impl with
  | .eager sess => EagerSession.gatherScalar (α := α) sess (n := n) x i
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.gatherScalar (α := α) sess (n := n) x i

/-- Gather a row from a matrix at a `Fin` index (dispatches to eager vs compiled backend). -/
def gatherRow {α : Type} (s : Session α) [Zero α] [DecidableEq Shape]
  {rows cols : Nat} (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim rows (.dim cols .scalar)))
    (i : Fin rows) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (.dim cols .scalar)) := do
  match s.impl with
  | .eager sess => EagerSession.gatherRow (α := α) sess (rows := rows) (cols := cols) x i
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.gatherRow (α := α) sess (rows := rows)
        (cols := cols) x i

/-- Gather a scalar from a vector using a `NatRef` index (dispatches to eager vs compiled backend).
  -/
def gatherScalarRef {α : Type} (s : Session α) [Zero α] [DecidableEq Shape]
  {n : Nat} (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim n .scalar)) (i :
    _root_.Runtime.Autograd.Torch.NatRef) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α Shape.scalar) := do
  match s.impl with
  | .eager sess => EagerSession.gatherScalarRef (α := α) sess (n := n) x i
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.gatherScalarRef (α := α) sess (n := n) x i

/-- Gather a row from a matrix using a `NatRef` index (dispatches to eager vs compiled backend). -/
def gatherRowRef {α : Type} (s : Session α) [Zero α] [DecidableEq Shape]
  [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
  {rows cols : Nat} (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim rows (.dim cols .scalar)))
    (i : _root_.Runtime.Autograd.Torch.NatRef) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (.dim cols .scalar)) := do
  match s.impl with
  | .eager sess => EagerSession.gatherRowRef (α := α) sess (rows := rows) (cols := cols) x i
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.gatherRowRef (α := α) sess (rows := rows)
        (cols := cols) x i

/-- Gather a scalar using a raw `Nat` index (dispatches to eager vs compiled backend). -/
def gatherScalarNat {α : Type} (s : Session α) [Zero α] [DecidableEq Shape]
  {n : Nat} (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim n .scalar)) (i : Nat) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α Shape.scalar) := do
  match s.impl with
  | .eager sess => EagerSession.gatherScalarNat (α := α) sess (n := n) x i
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.gatherScalarNat (α := α) sess (n := n) x i

/-- Gather a vector of entries using an index tensor (dispatches to eager vs compiled backend). -/
def gatherVecNat {α : Type} (s : Session α) [Add α] [Zero α] [DecidableEq Shape]
  {n k : Nat} (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim n .scalar)) (idx : Tensor Nat
    (.dim k .scalar)) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (.dim k .scalar)) := do
  match s.impl with
  | .eager sess => EagerSession.gatherVecNat (α := α) sess (n := n) (k := k) x idx
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.gatherVecNat (α := α) sess
        (n := n) (k := k) x idx

/-- Gather multiple rows using an index tensor (dispatches to eager vs compiled backend). -/
def gatherRowsNat {α : Type} (s : Session α) [Add α] [Zero α] [DecidableEq Shape]
  {rows cols k : Nat} (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim rows (.dim cols
    .scalar)))
  (idx : Tensor Nat (.dim k .scalar)) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (.dim k (.dim cols .scalar))) := do
  match s.impl with
  | .eager sess =>
      EagerSession.gatherRowsNat (α := α) sess (rows := rows) (cols := cols) (k := k) x idx
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.gatherRowsNat (α := α) sess
        (rows := rows) (cols := cols) (k := k) x idx

/-- `gather_vec_nat`, but indices are stored in a `NatVecRef` leaf (dispatches to eager vs compiled
  backend). -/
def gatherVecRef {α : Type} (s : Session α) [Add α] [Zero α] [DecidableEq Shape]
  {n k : Nat} (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim n .scalar))
  (idx : _root_.Runtime.Autograd.Torch.NatVecRef k) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (.dim k .scalar)) := do
  match s.impl with
  | .eager sess => EagerSession.gatherVecRef (α := α) sess (n := n) (k := k) x idx
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.gatherVecRef (α := α) sess
        (n := n) (k := k) x idx

/-- `gather_rows_nat`, but indices are stored in a `NatVecRef` leaf (dispatches to eager vs compiled
  backend). -/
def gatherRowsRef {α : Type} (s : Session α) [Add α] [Zero α] [DecidableEq Shape]
  {rows cols k : Nat} (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim rows (.dim cols
    .scalar)))
  (idx : _root_.Runtime.Autograd.Torch.NatVecRef k) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (.dim k (.dim cols .scalar))) := do
  match s.impl with
  | .eager sess =>
      EagerSession.gatherRowsRef (α := α) sess (rows := rows) (cols := cols) (k := k) x idx
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.gatherRowsRef (α := α) sess (rows := rows)
        (cols := cols) (k := k) x idx

/-- Scatter-add into a vector at a `Fin` index (dispatches to eager vs compiled backend). -/
def scatterAddVec {α : Type} (s : Session α) [Add α] [Zero α] [DecidableEq Shape]
  {n : Nat} (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim n .scalar))
  (v : _root_.Runtime.Autograd.Torch.TensorRef α Shape.scalar) (i : Fin n) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (.dim n .scalar)) := do
  match s.impl with
  | .eager sess => EagerSession.scatterAddVec (α := α) sess (n := n) x v i
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.scatterAddVec (α := α) sess (n := n) x v i

/-- Scatter-add into a matrix row at a `Fin` index (dispatches to eager vs compiled backend). -/
def scatterAddRow {α : Type} (s : Session α) [Add α] [Zero α] [DecidableEq Shape]
  {rows cols : Nat}
  (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim rows (.dim cols .scalar)))
  (v : _root_.Runtime.Autograd.Torch.TensorRef α (.dim cols .scalar)) (i : Fin rows) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (.dim rows (.dim cols .scalar))) := do
  match s.impl with
  | .eager sess => EagerSession.scatterAddRow (α := α) sess (rows := rows) (cols := cols) x v i
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.scatterAddRow (α := α) sess (rows := rows)
        (cols := cols) x v i

/--
Fully-connected (affine) layer on vectors: `y = w·x + b`.

PyTorch analogue: `torch.nn.functional.linear` (weight shape `(outDim, inDim)`).
-/
def linear {α : Type} (s : Session α) [Inhabited α] [Add α] [Mul α] [Zero α] [DecidableEq Shape]
  [_root_.Runtime.Autograd.FastKernels.FastMatmul α]
  {inDim outDim : Nat}
  (w : _root_.Runtime.Autograd.Torch.TensorRef α (.dim outDim (.dim inDim .scalar)))
  (b : _root_.Runtime.Autograd.Torch.TensorRef α (.dim outDim .scalar))
  (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim inDim .scalar)) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (.dim outDim .scalar)) := do
  match s.impl with
  | .eager sess =>
      EagerSession.linear (α := α) sess (inDim := inDim) (outDim := outDim) w b x
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.linear (α := α) sess
        (inDim := inDim) (outDim := outDim) w b x

/--
Mean squared error loss returning a scalar.

PyTorch analogue: `torch.nn.functional.mse_loss(..., reduction='mean')`.
-/
def mseLoss {α : Type} (s : Session α)
  [Inhabited α] [Add α] [Sub α] [Mul α] [Div α] [Zero α] [One α] [Coe Nat α] [DecidableEq Shape]
  [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
  {sh : Shape}
  (yhat target : _root_.Runtime.Autograd.Torch.TensorRef α sh) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α Shape.scalar) := do
  match s.impl with
  | .eager sess => EagerSession.mseLoss (α := α) sess (sh := sh) yhat target
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.mseLoss (α := α) sess (sh := sh) yhat target

/--
LayerNorm over a `seqLen × embedDim` tensor.

PyTorch analogue: `torch.nn.LayerNorm(embedDim)` applied per token.
-/
def layerNorm {α : Type} (s : Session α) [Context α] [DecidableEq Shape]
  {seqLen embedDim : Nat} (h_seq_pos : seqLen > 0) (h_embed_pos : embedDim > 0)
  (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim seqLen (.dim embedDim .scalar)))
  (gamma : _root_.Runtime.Autograd.Torch.TensorRef α (.dim embedDim .scalar))
  (beta : _root_.Runtime.Autograd.Torch.TensorRef α (.dim embedDim .scalar)) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (.dim seqLen (.dim embedDim .scalar))) := do
  match s.impl with
  | .eager sess =>
      EagerSession.layerNorm (α := α) sess
        (seqLen := seqLen) (embedDim := embedDim) (h_seq_pos := h_seq_pos) (h_embed_pos :=
          h_embed_pos)
        x gamma beta
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.layerNorm (α := α) sess
        (seqLen := seqLen) (embedDim := embedDim) (h_seq_pos := h_seq_pos) (h_embed_pos :=
          h_embed_pos)
        x gamma beta

/--
BatchNorm over a CHW tensor (channel-first).

PyTorch analogue: `torch.nn.BatchNorm2d` (in channel-first layout).
-/
def batchnormChannelFirst {α : Type} (s : Session α) [Context α] [DecidableEq Shape]
  {channels height width : Nat} (h_c : channels > 0) (h_h : height > 0) (h_w : width > 0)
  (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim channels (.dim height (.dim width .scalar))))
  (gamma : _root_.Runtime.Autograd.Torch.TensorRef α (.dim channels .scalar))
  (beta : _root_.Runtime.Autograd.Torch.TensorRef α (.dim channels .scalar)) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (.dim channels (.dim height (.dim width .scalar))))
    := do
  match s.impl with
  | .eager sess =>
      EagerSession.batchnormChannelFirst (α := α) sess
        (channels := channels) (height := height) (width := width) (h_c := h_c) (h_h := h_h) (h_w :=
          h_w)
        x gamma beta
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.batchnormChannelFirst (α := α) sess
        (channels := channels) (height := height) (width := width) (h_c := h_c) (h_h := h_h) (h_w :=
          h_w)
        x gamma beta

/--
N-D convolution over a channels-first tensor `(inC, spatial...)`.

This is the generic counterpart to `conv2d`.

PyTorch analogue: `torch.nn.functional.conv{d}d` specialized to a single sample.
-/
def conv {α : Type} (s : Session α) [Context α] [DecidableEq Shape]
  {d inC outC : Nat}
  {kernel stride padding : Vector Nat d}
  {inSpatial : Vector Nat d}
  (hInC : inC ≠ 0) (hKernel : ∀ i : Fin d, kernel.get i ≠ 0)
  (w : _root_.Runtime.Autograd.Torch.TensorRef α (Shape.ofList (outC :: inC :: kernel.toList)))
  (b : _root_.Runtime.Autograd.Torch.TensorRef α (.dim outC .scalar))
  (x : _root_.Runtime.Autograd.Torch.TensorRef α (Shape.ofList (inC :: inSpatial.toList))) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α
    (Shape.ofList (outC :: (Spec.convOutSpatial inSpatial kernel stride padding).toList))) := do
  match s.impl with
  | .eager sess =>
      EagerSession.conv (α := α) sess
        (d := d) (inC := inC) (outC := outC)
        (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial)
        (hInC := hInC) (hKernel := hKernel)
        w b x
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.conv (α := α) sess
        (d := d) (inC := inC) (outC := outC)
        (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial)
        (hInC := hInC) (hKernel := hKernel)
        w b x

/--
N-D transpose convolution over a channels-first tensor `(inC, spatial...)`.

PyTorch analogue: `torch.nn.functional.conv_transpose{d}d` specialized to a single sample.
-/
def convTranspose {α : Type} (s : Session α) [Context α] [DecidableEq Shape]
  {d inC outC : Nat}
  {kernel stride padding : Vector Nat d}
  {inSpatial : Vector Nat d}
  (hInC : inC ≠ 0) (hKernel : ∀ i : Fin d, kernel.get i ≠ 0)
  (w : _root_.Runtime.Autograd.Torch.TensorRef α (Shape.ofList (inC :: outC :: kernel.toList)))
  (b : _root_.Runtime.Autograd.Torch.TensorRef α (.dim outC .scalar))
  (x : _root_.Runtime.Autograd.Torch.TensorRef α (Shape.ofList (inC :: inSpatial.toList))) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α
    (Shape.ofList (outC :: (Spec.convTransposeOutSpatial inSpatial kernel stride padding).toList)))
    := do
  match s.impl with
  | .eager sess =>
      EagerSession.convTranspose (α := α) sess
        (d := d) (inC := inC) (outC := outC)
        (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial)
        (hInC := hInC) (hKernel := hKernel)
        w b x
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.convTranspose (α := α) sess
        (d := d) (inC := inC) (outC := outC)
        (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial)
        (hInC := hInC) (hKernel := hKernel)
        w b x

/--
2D convolution over a CHW tensor.

PyTorch analogue: `torch.nn.functional.conv2d` (channel-first layout).
-/
def conv2d {α : Type} (s : Session α) [Context α] [DecidableEq Shape]
  {inC outC kH kW stride padding inH inW : Nat}
  (h1 : inC ≠ 0) (h2 : kH ≠ 0) (h3 : kW ≠ 0)
  (kernel : _root_.Runtime.Autograd.Torch.TensorRef α (.dim outC (.dim inC (.dim kH (.dim kW
    .scalar)))))
  (bias : _root_.Runtime.Autograd.Torch.TensorRef α (.dim outC .scalar))
  (input : _root_.Runtime.Autograd.Torch.TensorRef α (.dim inC (.dim inH (.dim inW .scalar)))) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α
    (.dim outC (.dim ((inH + 2 * padding - kH) / stride + 1)
      (.dim ((inW + 2 * padding - kW) / stride + 1) .scalar)))) := do
  match s.impl with
  | .eager sess =>
      EagerSession.conv2d (α := α) sess
        (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding)
        (inH := inH) (inW := inW) (h1 := h1) (h2 := h2) (h3 := h3)
        kernel bias input
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.conv2d (α := α) sess
        (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding)
        (inH := inH) (inW := inW) (h1 := h1) (h2 := h2) (h3 := h3)
        kernel bias input

/--
2D transpose convolution over a CHW tensor.

PyTorch analogue: `torch.nn.functional.conv_transpose2d` (channel-first layout).
-/
def convTranspose2d {α : Type} (s : Session α) [Context α] [DecidableEq Shape]
  {inC outC kH kW stride padding inH inW : Nat}
  (h1 : inC ≠ 0) (h2 : kH ≠ 0) (h3 : kW ≠ 0)
  (kernel : _root_.Runtime.Autograd.Torch.TensorRef α (.dim inC (.dim outC (.dim kH (.dim kW
    .scalar)))))
  (bias : _root_.Runtime.Autograd.Torch.TensorRef α (.dim outC .scalar))
  (input : _root_.Runtime.Autograd.Torch.TensorRef α (.dim inC (.dim inH (.dim inW .scalar)))) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α
    (.dim outC (.dim ((inH - 1) * stride - 2 * padding + kH)
      (.dim ((inW - 1) * stride - 2 * padding + kW) .scalar)))) := do
  match s.impl with
  | .eager sess =>
      EagerSession.convTranspose2d (α := α) sess
        (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding)
        (inH := inH) (inW := inW) (h1 := h1) (h2 := h2) (h3 := h3)
        kernel bias input
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.convTranspose2d (α := α) sess
        (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding)
        (inH := inH) (inW := inW) (h1 := h1) (h2 := h2) (h3 := h3)
        kernel bias input

/-- Alias for `conv2d` (compat shorthand). -/
abbrev conv2dCompat := @conv2d

/--
Multi-head self-attention (single sequence, single batch).

This is a convenience op used by the transformer examples; it corresponds roughly to the forward
pass of `torch.nn.MultiheadAttention` in "self-attention" mode.
-/
def multiHeadAttention {α : Type} (s : Session α) [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {n numHeads dModel headDim : Nat} (h1 : n ≠ 0)
  (wq : _root_.Runtime.Autograd.Torch.TensorRef α (.dim dModel (.dim (numHeads * headDim) .scalar)))
  (wk : _root_.Runtime.Autograd.Torch.TensorRef α (.dim dModel (.dim (numHeads * headDim) .scalar)))
  (wv : _root_.Runtime.Autograd.Torch.TensorRef α (.dim dModel (.dim (numHeads * headDim) .scalar)))
  (wo : _root_.Runtime.Autograd.Torch.TensorRef α (.dim (numHeads * headDim) (.dim dModel .scalar)))
  (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim n (.dim dModel .scalar)))
  (mask : Option (Tensor Bool (.dim n (.dim n .scalar))) := none) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (.dim n (.dim dModel .scalar))) := do
  match s.impl with
  | .eager sess =>
      EagerSession.multiHeadAttention (α := α) sess
        (n := n) (numHeads := numHeads) (dModel := dModel) (headDim := headDim) (h1 := h1)
        wq wk wv wo x (mask := mask)
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.multiHeadAttention (α := α) sess
        (n := n) (numHeads := numHeads) (dModel := dModel) (headDim := headDim) (h1 := h1)
        wq wk wv wo x (mask := mask)

/--
Run a backward pass and return a dense array of gradients for *all* leaf tensors.

This is the TorchLean analogue of calling `backward()` and then reading `.grad` for every leaf,
but in an explicit "dense array" form.
-/
def backwardDenseAll {α : Type} (s : Session α) [Add α] [Zero α] [DecidableEq Shape]
  [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
  {sh : Shape} (out : _root_.Runtime.Autograd.Torch.TensorRef α sh) (seed : Tensor α sh) :
  IO (Array (_root_.Runtime.AnyTensor α)) := do
  match s.impl with
  | .eager sess =>
      EagerSession.backwardDenseAll (α := α) sess (sh := sh) out seed
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.backwardDenseAll (α := α) sess (sh := sh) out
        seed

namespace Internal

/--
Apply a gradient hook pointwise to a dense gradient array.

Invariant: the hook must preserve each gradient tensor's shape; we check this and throw if it
changes.
-/
def applyGradHook {α : Type}
    (grads : Array (_root_.Runtime.AnyTensor α))
    (hook : Nat → _root_.Runtime.AnyTensor α → IO (_root_.Runtime.AnyTensor α)) :
    IO (Array (_root_.Runtime.AnyTensor α)) := do
  let mut out : Array (_root_.Runtime.AnyTensor α) := #[]
  for i in List.finRange grads.size do
    let g := grads[i]
    let g' ← hook i.1 g
    if h : g'.s = g.s then
      out := out.push { g' with s := g.s, t := Tensor.castShape g'.t h }
    else
      throw <| IO.userError <|
        s!"torchlean: grad hook changed shape at id={i.1} (expected {Shape.pretty g.s}, got "
          ++ s!"{Shape.pretty g'.s})"
  pure out

end Internal

/--
Backward pass with an optional gradient hook applied to the *dense* gradient array.

This is a runtime utility (similar in spirit to PyTorch hooks), not part of the proof semantics.
-/
def backwardDenseAllWithHook {α : Type} (s : Session α) [Add α] [Zero α] [DecidableEq Shape]
  [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    {sh : Shape} (out : _root_.Runtime.Autograd.Torch.TensorRef α sh) (seed : Tensor α sh)
    (hook : Nat → _root_.Runtime.AnyTensor α → IO (_root_.Runtime.AnyTensor α)) :
    IO (Array (_root_.Runtime.AnyTensor α)) := do
  Internal.applyGradHook (α := α) (grads := (← backwardDenseAll (α := α) s (sh := sh) out seed))
    hook

/-- Backward pass for a scalar loss, returning the dense gradient array (seed is implicitly `1`). -/
def backwardScalarDenseAll {α : Type} (s : Session α) [Add α] [Zero α] [One α] [DecidableEq Shape]
  [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
  (loss : _root_.Runtime.Autograd.Torch.TensorRef α Shape.scalar) :
  IO (Array (_root_.Runtime.AnyTensor α)) := do
  match s.impl with
  | .eager sess => EagerSession.backwardScalarDenseAll (α := α) sess loss
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.backwardScalarDenseAll (α := α) sess loss

/-- `backwardScalarDenseAll` with a per-leaf gradient hook applied. -/
def backwardScalarDenseAllWithHook {α : Type} (s : Session α) [Add α] [Zero α] [One α] [DecidableEq
  Shape]
  [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    (loss : _root_.Runtime.Autograd.Torch.TensorRef α Shape.scalar)
    (hook : Nat → _root_.Runtime.AnyTensor α → IO (_root_.Runtime.AnyTensor α)) :
    IO (Array (_root_.Runtime.AnyTensor α)) := do
  Internal.applyGradHook (α := α) (grads := (← backwardScalarDenseAll (α := α) s loss)) hook

/--
Extract the gradient for a particular tensor ref from a dense gradient array.

This is the TorchLean analogue of reading `x.grad` (but without mutation).
-/
def grad {α : Type} {sh : Shape} [DecidableEq Shape]
  (grads : Array (_root_.Runtime.AnyTensor α)) (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) :
  IO (Tensor α sh) := do
  let gAny ← match grads[x.id]? with
    | some g => pure g
    | none => throw <| IO.userError "torchlean: gradient array out of bounds"
  if h : gAny.s = sh then
    pure (Tensor.castShape gAny.t h)
  else
    throw <| IO.userError
      s!"torchlean: grad shape mismatch (expected {Shape.pretty sh}, got {Shape.pretty gAny.s})"

/-- Vector-Jacobian product: `vjp(out, seed)[x]`. -/
def vjp {α : Type} (s : Session α) [Add α] [Zero α] [DecidableEq Shape]
  [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    {shOut shX : Shape}
    (out : _root_.Runtime.Autograd.Torch.TensorRef α shOut)
    (seed : Tensor α shOut)
    (x : _root_.Runtime.Autograd.Torch.TensorRef α shX) :
    IO (Tensor α shX) := do
  let grads ← backwardDenseAll (α := α) s (sh := shOut) out seed
  grad (α := α) (sh := shX) grads x

/-- Scalar-loss VJP with implicit seed `1`: `∇_x loss`. -/
def vjpScalar {α : Type} (s : Session α) [Add α] [Zero α] [One α] [DecidableEq Shape]
  [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    {shX : Shape}
    (loss : _root_.Runtime.Autograd.Torch.TensorRef α Shape.scalar)
    (x : _root_.Runtime.Autograd.Torch.TensorRef α shX) :
    IO (Tensor α shX) := do
  let grads ← backwardScalarDenseAll (α := α) s loss
  grad (α := α) (sh := shX) grads x

/-! ## Forward-mode: JVP -/

/--
Jacobian-vector product for a single leaf (compiled backend only).

For eager sessions, use the compiled backend if you need JVPs.
-/
def jvpLeaf {α : Type} (s : Session α) [Zero α] [DecidableEq Shape]
    {shOut shX : Shape}
    (out : _root_.Runtime.Autograd.Torch.TensorRef α shOut)
    (x : _root_.Runtime.Autograd.Torch.TensorRef α shX)
    (dx : Tensor α shX) :
    IO (Tensor α shOut) := do
  match s.impl with
  | .eager _ =>
      throw <| IO.userError "torchlean: jvpLeaf is only supported for compiled sessions"
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.jvpLeaf (α := α) sess
        (shOut := shOut) (shX := shX) out x dx

/-- Scalar-loss JVP for a single leaf (compiled backend only). -/
def jvpScalarLeaf {α : Type} (s : Session α) [Zero α] [DecidableEq Shape]
    (loss : _root_.Runtime.Autograd.Torch.TensorRef α Shape.scalar)
    {shX : Shape} (x : _root_.Runtime.Autograd.Torch.TensorRef α shX) (dx : Tensor α shX) :
    IO α := do
  match s.impl with
  | .eager _ =>
      throw <| IO.userError "torchlean: jvpScalarLeaf is only supported for compiled sessions"
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.jvpScalarLeaf (α := α) sess loss x dx

/-! ## Forward-mode: dense JVP (compiled backend only) -/

/--
Jacobian-vector product with explicit tangents for all *leaf* tensors.

`dxs[i]` is the tangent for leaf `i` (same indexing as `grad`/`backwardDenseAll`).
-/
def jvpDenseAll {α : Type} (s : Session α) [Zero α] [DecidableEq Shape]
    {shOut : Shape}
    (out : _root_.Runtime.Autograd.Torch.TensorRef α shOut)
    (dxs : Array (_root_.Runtime.AnyTensor α)) :
    IO (Tensor α shOut) := do
  match s.impl with
  | .eager _ =>
      throw <| IO.userError "torchlean: jvpDenseAll is only supported for compiled sessions"
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.jvpDenseAll (α := α) sess (sh := shOut) out
        dxs

/--
Apply a dense SGD step to all learnable parameters.

This is an optimizer helper used by examples; for a higher-level API see
  `NN.API.TorchLean.Trainer`.
-/
def sgdStepAll {α : Type} (s : Session α)
  [Sub α] [Mul α] [Add α] [Zero α] [DecidableEq Shape]
  [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
  (lr : α) (grads : Array (_root_.Runtime.AnyTensor α)) : IO Unit := do
  match s.impl with
  | .eager sess => EagerSession.sgdStepAll (α := α) sess lr grads
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.sgdStepAll (α := α) sess lr grads

/-- Reset the tape, then run one fresh graph-building action. -/
def withFreshTape {α β : Type} (s : Session α) (act : IO β) : IO β := do
  resetTape (α := α) s
  act

/--
Build one scalar-loss graph, run backward on it, and apply a dense SGD step.

This is the small "session-style training step" helper intended for demos and simple imperative
workflows that are lower-level than `API.TorchLean.Trainer`, but should still avoid manual
`resetTape` / `backwardScalarDenseAll` / `sgdStepAll` wiring.
-/
def sgdStepScalarGraph {α : Type} (s : Session α)
    [Sub α] [Mul α] [Add α] [Zero α] [One α] [DecidableEq Shape]
    [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    (lr : α)
    (buildLoss : IO (_root_.Runtime.Autograd.Torch.TensorRef α Shape.scalar)) :
    IO α :=
  withFreshTape (α := α) s do
    let loss ← buildLoss
    let lossT ← getValue (α := α) s (sh := Shape.scalar) loss
    let grads ← backwardScalarDenseAll (α := α) s loss
    sgdStepAll (α := α) s lr grads
    pure (Tensor.toScalar lossT)

/--
Apply a dense SGD step to all parameters after transforming gradients with a user hook.

The `hook` is applied to each gradient tensor (given its leaf index) and can implement common
training tricks like gradient clipping, normalization, or noise injection.

PyTorch analogy:
- tensor hooks (`Tensor.register_hook`) on gradients, or
- manually postprocessing gradients before calling `optimizer.step()`.
-/
def sgdStepAllWithHook {α : Type} (s : Session α)
    [Sub α] [Mul α] [Add α] [Zero α] [DecidableEq Shape]
    [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    (lr : α) (grads : Array (_root_.Runtime.AnyTensor α))
    (hook : Nat → _root_.Runtime.AnyTensor α → IO (_root_.Runtime.AnyTensor α)) : IO Unit := do
  sgdStepAll (α := α) s lr (← Internal.applyGradHook (α := α) grads hook)

end Session

end TorchLean
end Autograd
end Runtime
