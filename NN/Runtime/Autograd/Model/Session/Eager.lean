/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Model.Functional.ShapeOps
import Mathlib.Algebra.Order.Algebra
public import NN.Runtime.Autograd.Torch.Core.Trainer.EagerOps
public import NN.Runtime.Autograd.Torch.Core.BackwardOptim

/-!
# Session

TorchLean unified imperative session.

## Session state

A `Session α` is TorchLean's runtime analogue of a PyTorch "training loop environment". It:
- owns a collection of *leaf tensors* (parameters and inputs),
- records an eager computation on a tape or builds typed SSA graph data,
- can run reverse-mode AD to produce gradients for all leaves, and
- can apply simple optimizer steps (e.g. SGD) in a session-style workflow.

TorchLean exposes a **single API** with two execution modes selected at construction time:
- `.eager`: a tape-backed runtime session (imperative autograd tape; useful for debugging and
  interactive examples),
- `.typedGraph`: a session that records shape-indexed graph data while building the runtime
  tape, then executes via `Runtime.Autograd.Torch.Internal.TypedGraphSession`.

Both modes use the same `Session` API; each operation dispatches through `Session.state`.

## Typical Training Loop (PyTorch Analogy)

Think of the following mapping (approximately):
- `Session.param` ~ create a `torch.nn.Parameter` (and later include it in a `state_dict`-like
  bundle).
- `Session.use` ~ read a parameter as a tensor in the current recording phase.
- `Session.input` ~ add a leaf tensor input (like feeding a batch tensor into the forward pass).
- `Session.resetTape` ~ start a fresh recording phase (closest in spirit to
  `optimizer.zero_grad()` + new forward).
- `Session.backwardScalarDenseAll` ~ `loss.backward()` (but returns gradients explicitly as an
  array).
- `Session.sgdStepAll` ~ `optimizer.step()` (dense helper; higher-level training lives in
  `NN.API.*`).
- `Session.detach` ~ `tensor.detach()` (cut the gradient edge at a value).

TorchLean does *not* store mutable `.grad` fields on each tensor ref; instead, gradients are
  returned
explicitly (see `grad`, `vjp`, and the `backward*DenseAll` functions).

## Non-Differentiable State (`NatRef`)

`NatRef` stores the seed and counter used by the explicit random stream. Non-differentiable tensor
inputs use the element-polymorphic data-input channel rather than a second session tensor type.

## Deterministic RNG (Session-Level)

`RngState` provides explicit, deterministic RNG state (closer to JAX PRNG keys than a global RNG).
`freshSeedIO` is a convenience for sampling an initial seed at the IO boundary, while the *core*
semantics remains seed-threaded and replayable.

## Connection To TorchLean IR / Graph Execution

In `.typedGraph` execution, the session records executable `GraphData` while building the tape.
Each call to `resetTape` starts a new recording phase. Callers that need one reusable artifact
should use `Runtime.Autograd.Torch.TypedGraph` directly; the high-level scalar trainer uses that
artifact and records its loss graph once.

The type-level context checks graph shapes. Correctness of a stored JVP or VJP is a separate claim,
proved only for operations connected to the proof-carrying `Proofs.Autograd.Algebra.Node` layer.

Practical note: the current `.typedGraph` implementation expects all leaves (tensor
inputs/parameters and `NatRef`s) to be created before any op nodes are recorded. For portability,
allocate leaves and initialize/split RNG up-front, then build the typed graph.

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
namespace Model

open Spec TorchLean
open TorchLean TorchLean.Tensor

/--
Eager-only session wrapper.

This is the public eager-session record backed by the internal tape session
`Runtime.Autograd.Torch.Internal.EagerSession`. Users normally interact with the unified `Session`
API; this type exists to support execution-mode dispatch (`SessionState.eager`).
-/
structure EagerSession (α : Type) [TorchLean.Storage α] where
  /-- The internal tape session being wrapped. The extra layer exists so the execution-mode
  dispatch in `SessionState` has a public type to name. -/
  inner : Runtime.Autograd.Torch.Internal.EagerSession α

namespace EagerSession

/--
Create a new eager (tape-backed) session.

This corresponds to the `.eager` execution mode of `Session.new`.
-/
def new {α : Type} [TorchLean.Storage α] (options : Runtime.Autograd.Torch.Config := {}) :
  IO (EagerSession α) := do
  let inner ← Runtime.Autograd.Torch.Internal.EagerSession.new (α := α) (options := options)
  pure { inner := inner }

/-- Reset the eager autograd tape and begin a fresh recording phase. -/
def resetTape {α : Type} [TorchLean.Storage α] (s : EagerSession α) : IO Unit := do
  Runtime.Autograd.Torch.Internal.EagerSession.resetTape (α := α) s.inner

/--
Create a learnable parameter owned by this session.

PyTorch analogy: creating a `torch.nn.Parameter` during module initialization.
-/
def param {α : Type} [TorchLean.Storage α] (s : EagerSession α) {sh : Shape}
  (init : Tensor α sh) (name : Option String := none) (requiresGrad : Option Bool := none) :
  IO (Runtime.Autograd.Torch.Param α sh) :=
  Runtime.Autograd.Torch.Internal.EagerSession.param (α := α) (sh := sh) s.inner
    init (name := name) (requiresGrad := requiresGrad)

/--
Use a parameter in the current eager recording.

PyTorch analogy: reading a parameter in `forward` (it becomes part of the autograd graph).
-/
def use {α : Type} [TorchLean.Storage α] (s : EagerSession α) {sh : Shape}
  [Runtime.Autograd.Torch.TensorTransfer α]
  (p : Runtime.Autograd.Torch.Param α sh) : IO (Runtime.Autograd.Torch.TensorRef α sh)
    :=
  Runtime.Autograd.Torch.Internal.EagerSession.use (α := α) (sh := sh) s.inner p

/--
Add a tensor input leaf to the current graph.

`requiresGrad` controls whether this input is recorded as a differentiable leaf.
-/
def input {α : Type} [TorchLean.Storage α] (s : EagerSession α) {sh : Shape}
  [Runtime.Autograd.Torch.TensorTransfer α]
  (v : Tensor α sh) (name : Option String := none) (requiresGrad : Bool := false) :
  IO (Runtime.Autograd.Torch.TensorRef α sh) :=
  Runtime.Autograd.Torch.Internal.EagerSession.input (α := α) (sh := sh) s.inner
    v (name := name) (requiresGrad := requiresGrad)

/--
Add a non-differentiable `Nat` leaf to the session.

Used for labels/indices and gather-style ops.
-/
def inputNat {α : Type} [TorchLean.Storage α] (s : EagerSession α) (v : Nat) :
  IO (Runtime.Autograd.Torch.NatRef) :=
  Runtime.Autograd.Torch.Internal.EagerSession.inputNat (α := α) s.inner v

/-- Read a `NatRef` value. -/
def getNat {α : Type} [TorchLean.Storage α] (s : EagerSession α)
  (r : Runtime.Autograd.Torch.NatRef) : IO Nat :=
  Runtime.Autograd.Torch.Internal.EagerSession.getNat (α := α) s.inner r

/-- Mutate a `NatRef` value. -/
def setNat {α : Type} [TorchLean.Storage α] (s : EagerSession α) (r : Runtime.Autograd.Torch.NatRef)
  (v : Nat) : IO Unit :=
  Runtime.Autograd.Torch.Internal.EagerSession.setNat (α := α) s.inner r v

/--
Insert a constant tensor into the current graph.

PyTorch analogy: using a tensor literal/constant in the forward pass (as a leaf constant node).
-/
def const {α : Type} [TorchLean.Storage α] (s : EagerSession α) {sh : Shape}
  [Runtime.Autograd.Torch.TensorTransfer α]
  (v : Tensor α sh) (name : Option String := none) : IO (Runtime.Autograd.Torch.TensorRef α
    sh) :=
  Runtime.Autograd.Torch.Internal.EagerSession.const (α := α) (sh := sh) s.inner v (name :=
    name)

/-- Read the concrete value for a tensor ref (for logging/debugging). -/
def getValue {α : Type} [TorchLean.Storage α] (s : EagerSession α) {sh : Shape}
  [Runtime.Autograd.Torch.TensorTransfer α]
  (x : Runtime.Autograd.Torch.TensorRef α sh) : IO (Tensor α sh) :=
  Runtime.Autograd.Torch.Internal.EagerSession.getValue (α := α) (sh := sh) s.inner x

/--
Detach a tensor ref from the tape (stop gradient flow through it).

PyTorch analogy: `x.detach()`.
-/
def detach {α : Type} [TorchLean.Storage α] (s : EagerSession α) {sh : Shape} [Context α]
    [Runtime.Autograd.Torch.TensorTransfer α]
    (x : Runtime.Autograd.Torch.TensorRef α sh) :
    IO (Runtime.Autograd.Torch.TensorRef α sh) :=
  Runtime.Autograd.Torch.Internal.EagerSession.detach (α := α) (sh := sh) s.inner x

/-- Elementwise addition on tensor refs (eager execution path). -/
def add {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Add α] {sh : Shape}
  (a b : Runtime.Autograd.Torch.TensorRef α sh) : IO (Runtime.Autograd.Torch.TensorRef
    α sh) :=
  Runtime.Autograd.Torch.Internal.EagerSession.add (α := α) (sh := sh) s.inner a b

/-- Elementwise subtraction on tensor refs (eager execution path). -/
def sub {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Sub α] [Zero α] {sh : Shape}
  (a b : Runtime.Autograd.Torch.TensorRef α sh) : IO (Runtime.Autograd.Torch.TensorRef
    α sh) :=
  Runtime.Autograd.Torch.Internal.EagerSession.sub (α := α) (sh := sh) s.inner a b

/-- Elementwise multiplication on tensor refs (eager execution path). -/
def mul {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Mul α] {sh : Shape}
  (a b : Runtime.Autograd.Torch.TensorRef α sh) : IO (Runtime.Autograd.Torch.TensorRef
    α sh) :=
  Runtime.Autograd.Torch.Internal.EagerSession.mul (α := α) (sh := sh) s.inner a b

/-- Elementwise scaling by a scalar constant `c` (eager execution path). -/
def scale {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Mul α] {sh : Shape}
  [Runtime.Autograd.Torch.TensorTransfer α]
  (x : Runtime.Autograd.Torch.TensorRef α sh) (c : α) : IO
    (Runtime.Autograd.Torch.TensorRef α sh) :=
  Runtime.Autograd.Torch.Internal.EagerSession.scale (α := α) (sh := sh) s.inner x c

/-- Elementwise absolute value (eager execution path). -/
def abs {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)]
  {sh : Shape} (x : Runtime.Autograd.Torch.TensorRef α sh) :
  IO (Runtime.Autograd.Torch.TensorRef α sh) :=
  Runtime.Autograd.Torch.Internal.EagerSession.abs (α := α) (sh := sh) s.inner x

/-- Elementwise square root (eager execution path). -/
def sqrt {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)]
  {sh : Shape} (x : Runtime.Autograd.Torch.TensorRef α sh) :
  IO (Runtime.Autograd.Torch.TensorRef α sh) :=
  Runtime.Autograd.Torch.Internal.EagerSession.sqrt (α := α) (sh := sh) s.inner x

/-- Elementwise clamp to `[minVal, maxVal]` (eager execution path). -/
def clamp {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)]
  [Runtime.Autograd.Torch.TensorTransfer α]
  {sh : Shape} (x : Runtime.Autograd.Torch.TensorRef α sh) (minVal maxVal : α) :
  IO (Runtime.Autograd.Torch.TensorRef α sh) :=
  Runtime.Autograd.Torch.Internal.EagerSession.clamp (α := α) (sh := sh) s.inner x minVal
    maxVal

/-- Elementwise maximum (eager execution path). -/
def max {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)]
  {sh : Shape} (a b : Runtime.Autograd.Torch.TensorRef α sh) :
  IO (Runtime.Autograd.Torch.TensorRef α sh) :=
  Runtime.Autograd.Torch.Internal.EagerSession.max (α := α) (sh := sh) s.inner a b

/-- Elementwise minimum (eager execution path). -/
def min {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)]
  {sh : Shape} (a b : Runtime.Autograd.Torch.TensorRef α sh) :
  IO (Runtime.Autograd.Torch.TensorRef α sh) :=
  Runtime.Autograd.Torch.Internal.EagerSession.min (α := α) (sh := sh) s.inner a b

/-- Matrix multiplication with broadcasted batch prefixes (eager execution path). -/
def matmul {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Context α]
  {batchA batchB batch : Shape} {m n p : Nat}
  [broadcastA : Shape.BroadcastTo batchA batch]
  [broadcastB : Shape.BroadcastTo batchB batch]
  (a : Runtime.Autograd.Torch.TensorRef α (batchA.concat [m, n]))
  (b : Runtime.Autograd.Torch.TensorRef α (batchB.concat [n, p])) :
  IO (Runtime.Autograd.Torch.TensorRef α (batch.concat [m, p])) :=
  Runtime.Autograd.Torch.Internal.EagerSession.matmul (α := α) s.inner
    (batchA := batchA) (batchB := batchB) (batch := batch)
    (m := m) (n := n) (p := p) a b

/--
Concatenate along the outermost dimension (dimension 0) (eager execution path).

PyTorch analogy: `torch.cat([a, b], dim=0)`.
-/
def concatLeadingAxis {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Context α]
  {n m : Nat} {sh : Shape}
  (a : Runtime.Autograd.Torch.TensorRef α (.dim n sh))
  (b : Runtime.Autograd.Torch.TensorRef α (.dim m sh)) :
  IO (Runtime.Autograd.Torch.TensorRef α (.dim (n + m) sh)) :=
  Runtime.Autograd.Torch.Internal.EagerSession.concatLeadingAxis (α := α) s.inner (n := n) (m := m)
    (sh := sh) a b

/--
Slice a contiguous `[start, start+len)` range from dimension 0 (eager execution path).

PyTorch analogy: `x[start:start+len]` for the first dimension.
-/
def sliceLeadingAxisRange {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Zero α]
  {n : Nat} {sh : Shape}
  (x : Runtime.Autograd.Torch.TensorRef α (.dim n sh)) (start len : Nat) (h : start + len ≤
    n) :
  IO (Runtime.Autograd.Torch.TensorRef α (.dim len sh)) :=
  Runtime.Autograd.Torch.Internal.EagerSession.sliceLeadingAxisRange (α := α) s.inner (n := n)
    (sh := sh) x start len h

/-- Apply max pooling over an arbitrary number of spatial axes. -/
def maxPool {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Context α]
    {d channels : Nat} {spatial kernel stride padding : TorchLean.Tensor Nat [d]}
    (x : Runtime.Autograd.Torch.TensorRef α
      (Shape.ofList (channels :: spatial.to (List Nat)))) :
    IO (Runtime.Autograd.Torch.TensorRef α
      (Shape.ofList
        (channels ::
          (Spec.poolOutSpatialPad spatial kernel stride padding).to (List Nat)))) :=
  Runtime.Autograd.Torch.Internal.EagerSession.maxPool (α := α) s.inner
    (d := d) (C := channels) (inSpatial := spatial)
    (kernel := kernel) (stride := stride) (padding := padding) x

/-- Apply smooth max pooling over an arbitrary number of spatial axes. -/
def smoothMaxPool {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Context α] [DecidableEq α]
    [Runtime.Autograd.Torch.TensorTransfer α]
    {d channels : Nat} {spatial kernel stride padding : TorchLean.Tensor Nat [d]}
    (x : Runtime.Autograd.Torch.TensorRef α
      (Shape.ofList (channels :: spatial.to (List Nat)))) (beta : α) :
    IO (Runtime.Autograd.Torch.TensorRef α
      (Shape.ofList
        (channels ::
          (Spec.poolOutSpatialPad spatial kernel stride padding).to (List Nat)))) :=
  Runtime.Autograd.Torch.Internal.EagerSession.smoothMaxPool (α := α) s.inner
    (d := d) (C := channels) (inSpatial := spatial)
    (kernel := kernel) (stride := stride) (padding := padding) x beta

/-- Apply average pooling over an arbitrary number of spatial axes. -/
def avgPool {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Context α]
    {d channels : Nat} {spatial kernel stride padding : TorchLean.Tensor Nat [d]}
    (x : Runtime.Autograd.Torch.TensorRef α
      (Shape.ofList (channels :: spatial.to (List Nat)))) :
    IO (Runtime.Autograd.Torch.TensorRef α
      (Shape.ofList
        (channels ::
          (Spec.poolOutSpatialPad spatial kernel stride padding).to (List Nat)))) :=
  Runtime.Autograd.Torch.Internal.EagerSession.avgPool (α := α) s.inner
    (d := d) (C := channels) (inSpatial := spatial)
    (kernel := kernel) (stride := stride) (padding := padding) x

/-- Elementwise ReLU activation (eager execution path). -/
def relu {α : Type} [TorchLean.Storage α] (s : EagerSession α)
  [Mul α] [Zero α] [Max α] [BEq α] [One α] [LT α]
  [DecidableRel ((· > ·) : α → α → Prop)]
  {sh : Shape} (x : Runtime.Autograd.Torch.TensorRef α sh) : IO
    (Runtime.Autograd.Torch.TensorRef α sh) :=
  Runtime.Autograd.Torch.Internal.EagerSession.relu (α := α) (sh := sh) s.inner x

/-- Elementwise sigmoid activation (eager execution path). -/
def sigmoid {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Context α]
  {sh : Shape} (x : Runtime.Autograd.Torch.TensorRef α sh) : IO
    (Runtime.Autograd.Torch.TensorRef α sh) :=
  Runtime.Autograd.Torch.Internal.EagerSession.sigmoid (α := α) (sh := sh) s.inner x

/-- Elementwise tanh activation (eager execution path). -/
def tanh {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Context α]
  {sh : Shape} (x : Runtime.Autograd.Torch.TensorRef α sh) : IO
    (Runtime.Autograd.Torch.TensorRef α sh) :=
  Runtime.Autograd.Torch.Internal.EagerSession.tanh (α := α) (sh := sh) s.inner x

/-- Softmax along an explicitly selected tensor dimension (eager execution path). -/
def softmax {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Context α]
    {sh : Shape} (axis : Nat) [Shape.AxisInBounds axis sh]
    (x : Runtime.Autograd.Torch.TensorRef α sh) :
    IO (Runtime.Autograd.Torch.TensorRef α sh) :=
  let action : Runtime.Autograd.Torch.Internal.EagerM α
      (Runtime.Autograd.Torch.TensorRef α sh) :=
    F.softmax (m := Runtime.Autograd.Torch.Internal.EagerM α)
      (α := α) (s := sh) axis x
  action s.inner

/-- Stable log-softmax along an explicitly selected tensor dimension (eager execution path). -/
def logSoftmax {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Context α]
    {sh : Shape} (axis : Nat) [Shape.AxisInBounds axis sh]
    (x : Runtime.Autograd.Torch.TensorRef α sh) :
    IO (Runtime.Autograd.Torch.TensorRef α sh) :=
  let action : Runtime.Autograd.Torch.Internal.EagerM α
      (Runtime.Autograd.Torch.TensorRef α sh) :=
    F.logSoftmax (m := Runtime.Autograd.Torch.Internal.EagerM α)
      (α := α) (s := sh) axis x
  action s.inner

/-- Elementwise softplus activation (eager execution path). -/
def softplus {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Context α]
  {sh : Shape} (x : Runtime.Autograd.Torch.TensorRef α sh) : IO
    (Runtime.Autograd.Torch.TensorRef α sh) :=
  Runtime.Autograd.Torch.Internal.EagerSession.softplus (α := α) (sh := sh) s.inner x

/-- Elementwise exponential (eager execution path). -/
def exp {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Context α]
  {sh : Shape} (x : Runtime.Autograd.Torch.TensorRef α sh) : IO
    (Runtime.Autograd.Torch.TensorRef α sh) :=
  Runtime.Autograd.Torch.Internal.EagerSession.exp (α := α) (sh := sh) s.inner x

/-- Elementwise sine of angles in radians, recorded on the eager session's tape. -/
def sin {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Context α]
    {sh : Shape} (x : Runtime.Autograd.Torch.TensorRef α sh) :
    IO (Runtime.Autograd.Torch.TensorRef α sh) :=
  Runtime.Autograd.Torch.Internal.EagerSession.sin (α := α) (sh := sh) s.inner x

/-- Elementwise cosine with the eager tape's `-sin(x) * dLdy` backward rule. -/
def cos {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Context α]
    {sh : Shape} (x : Runtime.Autograd.Torch.TensorRef α sh) :
    IO (Runtime.Autograd.Torch.TensorRef α sh) :=
  Runtime.Autograd.Torch.Internal.EagerSession.cos (α := α) (sh := sh) s.inner x

/-- Elementwise logarithm (eager execution path). -/
def log {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Context α]
  {sh : Shape} (x : Runtime.Autograd.Torch.TensorRef α sh) : IO
    (Runtime.Autograd.Torch.TensorRef α sh) :=
  Runtime.Autograd.Torch.Internal.EagerSession.log (α := α) (sh := sh) s.inner x

/-- Elementwise `safeLog` activation (`log(softplus(x) + ε)`) (eager execution path). -/
def safeLog {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Context α]
  [Runtime.Autograd.Torch.TensorTransfer α]
  {sh : Shape} (x : Runtime.Autograd.Torch.TensorRef α sh) (ε : α := Context.defaultEpsilon) :
  IO (Runtime.Autograd.Torch.TensorRef α sh) :=
  Runtime.Autograd.Torch.Internal.EagerSession.safeLog (α := α) (sh := sh) s.inner x (ε :=
    ε)

/-- Sum-reduce a tensor to a scalar (eager execution path). -/
def sum {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Add α] [Zero α]
  {sh : Shape} (x : Runtime.Autograd.Torch.TensorRef α sh) :
  IO (Runtime.Autograd.Torch.TensorRef α Shape.scalar) :=
  Runtime.Autograd.Torch.Internal.EagerSession.sum (α := α) (sh := sh) s.inner x

/-- Flatten a tensor into a 1D vector (eager execution path). -/
def flatten {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Inhabited α] {sh : Shape}
  (x : Runtime.Autograd.Torch.TensorRef α sh) :
  IO (Runtime.Autograd.Torch.TensorRef α [Spec.Shape.size sh]) :=
  Runtime.Autograd.Torch.Internal.EagerSession.flatten (α := α) (sh := sh) s.inner x

/--
Reshape a tensor, given a proof that the total number of elements is preserved (eager execution
path).

PyTorch analogy: `x.reshape(...)` when the element count matches.
-/
def reshape {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Inhabited α] {sh1 sh2 : Shape}
  (x : Runtime.Autograd.Torch.TensorRef α sh1) (h : Spec.Shape.size sh1 = Spec.Shape.size sh2) :
  IO (Runtime.Autograd.Torch.TensorRef α sh2) :=
  Runtime.Autograd.Torch.Internal.EagerSession.reshape (α := α) (sh1 := sh1) (sh2 := sh2)
    s.inner x h

/--
Generic "swap adjacent axes" view operation (eager execution path).

This is a shape-driven permutation helper used in some attention/transformer code.
-/
def swapAdjacentAtDepth {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Context α]
  {sh : Shape} (depth : Nat) (x : Runtime.Autograd.Torch.TensorRef α sh) :
  IO (Runtime.Autograd.Torch.TensorRef α (sh.swapAdjacentAtDepth depth)) :=
  Runtime.Autograd.Torch.Internal.EagerSession.swapAdjacentAtDepth (α := α) (sh := sh)
    s.inner depth x

/-- Broadcast a tensor to a larger shape (eager execution path). -/
def broadcastTo {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Inhabited α] [Add α] [Zero α]
  {sh1 sh2 : Shape} (cb : Shape.CanBroadcastTo sh1 sh2) (x : Runtime.Autograd.Torch.TensorRef
    α sh1) :
  IO (Runtime.Autograd.Torch.TensorRef α sh2) :=
  Runtime.Autograd.Torch.Internal.EagerSession.broadcastTo (α := α) (sh1 := sh1) (sh2 := sh2)
    s.inner cb x

/-- Reduce-sum along an axis (eager execution path). -/
def reduceSum {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Add α] [Zero α] [Inhabited α]
  {sh : Shape} (axis : Nat) (x : Runtime.Autograd.Torch.TensorRef α sh)
  [valid : Shape.HasNonemptyAxis axis sh] [wf : Shape.WellFormed sh] :
  IO (Runtime.Autograd.Torch.TensorRef α (shapeAfterSum sh axis)) :=
  Runtime.Autograd.Torch.Internal.EagerSession.reduceSum (α := α) (sh := sh) s.inner axis x

/-- Reduce-mean along an axis (eager execution path). -/
def reduceMean {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Context α]
  {sh : Shape} (axis : Nat) (x : Runtime.Autograd.Torch.TensorRef α sh)
  [valid : Shape.HasNonemptyAxis axis sh] [wf : Shape.WellFormed sh] :
  IO (Runtime.Autograd.Torch.TensorRef α (shapeAfterSum sh axis)) :=
  Runtime.Autograd.Torch.Internal.EagerSession.reduceMean (α := α) (sh := sh) s.inner axis x

/-- Select one bounded coordinate from an arbitrary tensor axis. -/
def select {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Zero α]
    {shape : Shape} (axis : Nat) (x : Runtime.Autograd.Torch.TensorRef α shape)
    [Shape.AxisInBounds axis shape]
    (index : Fin (Shape.axisSize shape axis)) :
    IO (Runtime.Autograd.Torch.TensorRef α (shape.eraseAxis axis)) :=
  Runtime.Autograd.Torch.Internal.EagerSession.select (α := α) s.inner axis x index

/-- Select several bounded coordinates from an arbitrary tensor axis. -/
def indexSelect {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Add α] [Zero α]
    {shape : Shape} (axis count : Nat) (x : Runtime.Autograd.Torch.TensorRef α shape)
    [Shape.AxisInBounds axis shape]
    (indices : Tensor (Fin (Shape.axisSize shape axis)) [count]) :
    IO (Runtime.Autograd.Torch.TensorRef α (shape.replaceAxis axis count)) :=
  Runtime.Autograd.Torch.Internal.EagerSession.indexSelect (α := α) s.inner axis count
    x indices

/-- Add source slices into an arbitrary tensor axis at bounded coordinates. -/
def scatterAdd {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Add α] [Zero α]
    {shape : Shape} (axis count : Nat) (base : Runtime.Autograd.Torch.TensorRef α shape)
    [Shape.AxisInBounds axis shape]
    (source : Runtime.Autograd.Torch.TensorRef α (shape.replaceAxis axis count))
    (indices : Tensor (Fin (Shape.axisSize shape axis)) [count]) :
    IO (Runtime.Autograd.Torch.TensorRef α shape) :=
  Runtime.Autograd.Torch.Internal.EagerSession.scatterAdd (α := α) s.inner axis count
    base source indices

/--
Fully-connected (affine) layer on vectors: `y = w·x + b` (eager execution path).

PyTorch analogue: `torch.nn.functional.linear` (with weight shape `(outDim, inDim)`).
-/
def linear {α : Type} [TorchLean.Storage α] (s : EagerSession α)
  [Inhabited α] [Add α] [Mul α] [Zero α]
  {inDim outDim : Nat}
  (w : Runtime.Autograd.Torch.TensorRef α [outDim, inDim])
  (b : Runtime.Autograd.Torch.TensorRef α [outDim])
  (x : Runtime.Autograd.Torch.TensorRef α [inDim]) :
  IO (Runtime.Autograd.Torch.TensorRef α [outDim]) :=
  Runtime.Autograd.Torch.Internal.EagerSession.linear (α := α) (inDim := inDim) (outDim :=
    outDim)
    s.inner w b x

/--
Mean squared error loss returning a scalar (eager execution path).

PyTorch analogue: `torch.nn.functional.mse_loss(..., reduction='mean')`.
-/
def mseLoss {α : Type} [TorchLean.Storage α] (s : EagerSession α)
  [Inhabited α] [Add α] [Sub α] [Mul α] [Div α] [Zero α] [One α] [NatCast α]
  [Runtime.Autograd.Torch.TensorTransfer α]
  {sh : Shape}
  (yhat target : Runtime.Autograd.Torch.TensorRef α sh) :
  IO (Runtime.Autograd.Torch.TensorRef α Shape.scalar) :=
  Runtime.Autograd.Torch.Internal.EagerSession.mseLoss (α := α) (sh := sh) s.inner yhat
    target

/--
LayerNorm over a `seqLen × embedDim` tensor (eager execution path).

PyTorch analogue: `torch.nn.LayerNorm(embedDim)` applied per token.
-/
def layerNorm {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Context α]
  [Runtime.Autograd.Torch.TensorTransfer α]
  [DecidableRel ((· > ·) : α → α → Prop)]
  {seqLen embedDim : Nat} (h_seq_pos : seqLen > 0) (h_embed_pos : embedDim > 0)
  (x : Runtime.Autograd.Torch.TensorRef α [seqLen, embedDim])
  (gamma : Runtime.Autograd.Torch.TensorRef α [embedDim])
  (beta : Runtime.Autograd.Torch.TensorRef α [embedDim])
  (epsilon : α := TorchLean.normalizationEpsilon) :
  IO (Runtime.Autograd.Torch.TensorRef α [seqLen, embedDim]) :=
  Runtime.Autograd.Torch.Internal.EagerSession.layerNorm (α := α)
    (seqLen := seqLen) (embedDim := embedDim) (h_seq_pos := h_seq_pos) (h_embed_pos := h_embed_pos)
    s.inner x gamma beta (epsilon := epsilon)

/-- Batch normalization over every spatial axis of a channel-first tensor. -/
def batchNorm {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Context α]
    [Runtime.Autograd.Torch.TensorTransfer α]
    [DecidableRel ((· > ·) : α → α → Prop)]
    {channels : Nat} {sSpatial : Shape}
    (hWellFormed : (Shape.dim channels sSpatial).wellFormed)
  (x : Runtime.Autograd.Torch.TensorRef α (.dim channels sSpatial))
  (gamma : Runtime.Autograd.Torch.TensorRef α [channels])
  (beta : Runtime.Autograd.Torch.TensorRef α [channels])
  (epsilon : α := TorchLean.normalizationEpsilon) :
  IO (Runtime.Autograd.Torch.TensorRef α (.dim channels sSpatial)) :=
  Runtime.Autograd.Torch.Internal.EagerSession.batchNorm (α := α)
    (channels := channels) (sSpatial := sSpatial) s.inner hWellFormed x gamma beta
    (epsilon := epsilon)

/--
N-D convolution over a channels-first tensor `(inC, spatial...)` (eager execution path).

PyTorch analogue: `torch.nn.functional.conv{d}d` specialized to a single sample.
-/
def conv {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Context α]
  {d inC outC : Nat}
  {kernel stride padding : TorchLean.Tensor Nat [d]}
  {inSpatial : TorchLean.Tensor Nat [d]}
  (w : Runtime.Autograd.Torch.TensorRef α
    (Shape.ofList (outC :: inC :: kernel.to (List Nat))))
  (b : Runtime.Autograd.Torch.TensorRef α [outC])
  (x : Runtime.Autograd.Torch.TensorRef α
    (Shape.ofList (inC :: inSpatial.to (List Nat)))) :
  IO (Runtime.Autograd.Torch.TensorRef α
    (Shape.ofList
      (outC :: (Spec.convOutSpatial inSpatial kernel stride padding).to (List Nat)))) :=
  Runtime.Autograd.Torch.Internal.EagerSession.conv (α := α)
    (d := d) (inC := inC) (outC := outC)
    (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial)
    s.inner w b x

/--
N-D transpose convolution over a channels-first tensor `(inC, spatial...)` (eager execution path).

PyTorch analogue: `torch.nn.functional.conv_transpose{d}d` specialized to a single sample.
-/
def convTranspose {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Context α]
  {d inC outC : Nat}
  {kernel stride padding : TorchLean.Tensor Nat [d]}
  {inSpatial : TorchLean.Tensor Nat [d]}
  (w : Runtime.Autograd.Torch.TensorRef α
    (Shape.ofList (inC :: outC :: kernel.to (List Nat))))
  (b : Runtime.Autograd.Torch.TensorRef α [outC])
  (x : Runtime.Autograd.Torch.TensorRef α
    (Shape.ofList (inC :: inSpatial.to (List Nat)))) :
  IO (Runtime.Autograd.Torch.TensorRef α
    (Shape.ofList
      (outC :: (Spec.convTransposeOutSpatial inSpatial kernel stride padding).to (List Nat))))
    :=
  Runtime.Autograd.Torch.Internal.EagerSession.convTranspose (α := α)
    (d := d) (inC := inC) (outC := outC)
    (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial)
    s.inner w b x

/--
Multi-head self-attention (eager execution path).

This is the eager implementation used by the transformer examples (approximately analogous to
`torch.nn.MultiheadAttention` in self-attention mode).
-/
def multiHeadAttention {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)]
  {n numHeads dModel headDim : Nat} (h1 : n ≠ 0)
  (wq : Runtime.Autograd.Torch.TensorRef α [dModel, numHeads * headDim])
  (wk : Runtime.Autograd.Torch.TensorRef α [dModel, numHeads * headDim])
  (wv : Runtime.Autograd.Torch.TensorRef α [dModel, numHeads * headDim])
  (wo : Runtime.Autograd.Torch.TensorRef α [numHeads * headDim, dModel])
  (x : Runtime.Autograd.Torch.TensorRef α [n, dModel])
  (mask : Option (Tensor Bool [n, n]) := none) :
  IO (Runtime.Autograd.Torch.TensorRef α [n, dModel]) :=
  Runtime.Autograd.Torch.Internal.EagerSession.multiHeadAttention (α := α)
    (n := n) (numHeads := numHeads) (dModel := dModel) (headDim := headDim) (h1 := h1)
    s.inner wq wk wv wo x (mask := mask)

/--
Run a backward pass and return dense gradients for all leaves (eager execution path).

See the unified version `Session.backwardDenseAll` for the public API.
-/
def backwardDenseAll {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Add α] [Zero α]
  [Runtime.Autograd.Torch.TensorTransfer α]
  {sh : Shape} (out : Runtime.Autograd.Torch.TensorRef α sh) (seed : Tensor α sh) :
  IO (Array (Spec.SomeTensor α)) := do
  s.inner.validateTensorRef out
  Runtime.Autograd.Torch.Internal.EagerSession.backwardDenseAll (α := α) (sh := sh) s.inner
    out seed

/-- Backward pass specialized to scalar losses (seed is implicitly `1`) (eager execution path). -/
def backwardScalarDenseAll {α : Type} [TorchLean.Storage α] (s : EagerSession α)
  [Add α] [Zero α] [One α]
  [Runtime.Autograd.Torch.TensorTransfer α]
  (loss : Runtime.Autograd.Torch.TensorRef α Shape.scalar) :
  IO (Array (Spec.SomeTensor α)) := do
  s.inner.validateTensorRef loss
  Runtime.Autograd.Torch.Internal.EagerSession.backwardScalarDenseAll (α := α) s.inner loss

/--
Apply an SGD step to all learnable parameters given a dense gradient array (eager execution path).

PyTorch analogy: `optimizer.step()` for an SGD optimizer, with gradients supplied explicitly.
-/
def sgdStepAll {α : Type} [TorchLean.Storage α] (s : EagerSession α)
  [Sub α] [Mul α] [Add α] [Zero α]
  [Runtime.Autograd.Torch.TensorTransfer α]
  (lr : α) (grads : Array (Spec.SomeTensor α)) : IO Unit :=
  Runtime.Autograd.Torch.Internal.EagerSession.sgdStepAll (α := α) s.inner lr grads

end EagerSession

end Model
end Autograd
end Runtime
