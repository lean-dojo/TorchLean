/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Engine.Core.ActivationsLoss
public import NN.Runtime.Autograd.Engine.Core.Backward
public import NN.Runtime.Autograd.Engine.Core.ConvPool
public import NN.Runtime.Autograd.Engine.Core.Elementwise
public import NN.Runtime.Autograd.Engine.Core.Linear
public import NN.Runtime.Autograd.Engine.Core.Neural

/-!
# TapeM

Tape-building convenience API.

The core autograd runtime (`Runtime.Autograd.Tape`) is pure and explicitly threaded:
each op returns an updated tape plus the new node id. This makes the engine easy to reason
about and convenient for proofs, but it can feel verbose in user code.

`Runtime.Autograd.TapeM` is a small `StateT` wrapper that threads the tape implicitly,
closer to the "define ops; then call backward" ergonomics users expect from frameworks
like PyTorch.

For training scripts and tests, see `NN.Runtime.Autograd.Train`, which provides dataset and
optimizer helpers, and `NN.Runtime.Autograd.Torch.ScalarTrainer`, which provides packed adapters
for reading scalar losses, extracting typed gradients, and applying SGD updates.

## Main declarations

- `NN.Runtime.Autograd.Engine.Core` contains the pure tape and low-level node constructors.
- `TapeM.run` returns the result and final tape; `StateT.run'` returns only the result.
- The op wrappers below (`add`, `linear`, `conv`, etc.) mirror the `Tape` namespace while
  threading state implicitly.
-/

@[expose] public section


namespace Runtime
namespace Autograd

open Spec TorchLean
open TorchLean TorchLean.Tensor

/--
A convenient tape-builder monad.

`TapeM α β` is `StateT (Tape α) Result β`: a pure tape threaded implicitly with errors reported
via `Except String`. This mirrors the common eager style of building a computation and then calling
`backward`, similar to PyTorch's imperative API, but remains purely functional.
-/
abbrev TapeM (α : Type) [TorchLean.Storage α] : Type → Type :=
  StateT (Tape α) Result

namespace TapeM

variable {α β : Type} [TorchLean.Storage α]

/-- Run a `TapeM` computation from an initial tape, returning both the result and the final tape. -/
def run (t : Tape α) (m : TapeM α β) : Result (β × Tape α) :=
  StateT.run m t

/-- Adapt the tape's `(state, id)` result to the `StateT` `(id, state)` convention. -/
@[inline] def Internal.record (op : Tape α → Result (Tape α × Nat)) : TapeM α Nat :=
  fun t => do
    let (t', id) ← op t
    pure (id, t')

/--
Create a leaf node holding a concrete tensor value.

A leaf is the "input tensor" analogue: it has no parents. Setting `requiresGrad := true`
corresponds to PyTorch tensors created with `requires_grad=True`.
-/
def leaf {s : Shape}
  (value : Tensor α s) (name : Option String := none) (requiresGrad : Bool := true) :
  TapeM α Nat :=
  StateT.modifyGet fun t =>
    (Tape.leaf (t := t) value (name := name) (requiresGrad := requiresGrad)).swap

/-- StateT wrapper around `Tape.add`. PyTorch comparison: `torch.add(a, b)`. -/
def add {α : Type} [TorchLean.Storage α] [Add α] {s : Shape}
  (aId bId : Nat) : TapeM α Nat :=
  Internal.record fun t => Tape.add (t := t) (s := s) aId bId

/-- StateT wrapper around `Tape.sub`. PyTorch comparison: `torch.sub(a, b)`. -/
def sub {α : Type} [TorchLean.Storage α] [Sub α] [Zero α] {s : Shape}
  (aId bId : Nat) : TapeM α Nat :=
  Internal.record fun t => Tape.sub (t := t) (s := s) aId bId

/-- StateT wrapper around `Tape.mul`. PyTorch comparison: `torch.mul(a, b)`. -/
def mul {α : Type} [TorchLean.Storage α] [Mul α] {s : Shape}
  (aId bId : Nat) : TapeM α Nat :=
  Internal.record fun t => Tape.mul (t := t) (s := s) aId bId

/-- StateT wrapper around `Tape.div`. PyTorch comparison: `torch.div(a, b)` / `a / b`. -/
def div {α : Type} [TorchLean.Storage α] [Div α] [Mul α] [Sub α] [Zero α] {s : Shape}
  (aId bId : Nat) : TapeM α Nat :=
  Internal.record fun t => Tape.div (t := t) (s := s) aId bId

/-- StateT wrapper around `Tape.scale`. PyTorch comparison: `c * x` / `torch.mul(x, c)`. -/
def scale {α : Type} [TorchLean.Storage α] [Mul α] {s : Shape}
  (xId : Nat) (c : α) : TapeM α Nat :=
  Internal.record fun t => Tape.scale (t := t) (s := s) xId c

/-- StateT wrapper around `Tape.abs`. PyTorch comparison: `torch.abs(x)`. -/
def abs {α : Type} [TorchLean.Storage α] [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  {s : Shape} (xId : Nat) : TapeM α Nat :=
  Internal.record fun t => Tape.abs (t := t) (s := s) xId

/-- StateT wrapper around `Tape.sqrt`. PyTorch comparison: `torch.sqrt(x)`. -/
def sqrt {α : Type} [TorchLean.Storage α] [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  {s : Shape} (xId : Nat) : TapeM α Nat :=
  Internal.record fun t => Tape.sqrt (t := t) (s := s) xId

/-- StateT wrapper around `Tape.clamp`. PyTorch comparison: `torch.clamp(x, min, max)`. -/
def clamp {α : Type} [TorchLean.Storage α] [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  {s : Shape} (xId : Nat) (minVal maxVal : α) : TapeM α Nat :=
  Internal.record fun t => Tape.clamp (t := t) (s := s) xId minVal maxVal

/-- StateT wrapper around `Tape.max`. PyTorch comparison: `torch.maximum(a, b)`. -/
def max {α : Type} [TorchLean.Storage α] [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  {s : Shape} (aId bId : Nat) : TapeM α Nat :=
  Internal.record fun t => Tape.max (t := t) (s := s) aId bId

/-- StateT wrapper around `Tape.min`. PyTorch comparison: `torch.minimum(a, b)`. -/
def min {α : Type} [TorchLean.Storage α] [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  {s : Shape} (aId bId : Nat) : TapeM α Nat :=
  Internal.record fun t => Tape.min (t := t) (s := s) aId bId

/-- StateT wrapper around `Tape.relu`. PyTorch comparison: `torch.nn.functional.relu(x)`. -/
def relu {α : Type} [TorchLean.Storage α]
  [Mul α] [Zero α] [Max α] [BEq α] [One α] [LT α]
  [DecidableRel ((· > ·) : α → α → Prop)]
  {s : Shape} (xId : Nat) : TapeM α Nat :=
  Internal.record fun t => Tape.relu (t := t) (s := s) xId

/-- StateT wrapper around `Tape.linear`. PyTorch comparison: `torch.nn.functional.linear`. -/
def linear {α : Type} [TorchLean.Storage α] [Add α] [Mul α] [Zero α]
  {inDim outDim : Nat} (wId bId xId : Nat) : TapeM α Nat :=
  Internal.record fun t => Tape.linear (t := t) (inDim := inDim) (outDim := outDim) wId bId xId

/-- Record matrix multiplication with broadcasted batch prefixes.
The empty-prefix defaults give ordinary matrix multiplication. -/
def matmul {α : Type} [TorchLean.Storage α] [Add α] [Mul α] [Zero α]
    {m n p : Nat} (aId bId : Nat)
    (batchA : Shape := .scalar) (batchB : Shape := .scalar) (batch : Shape := .scalar)
    [Shape.BroadcastTo batchA batch] [Shape.BroadcastTo batchB batch] : TapeM α Nat :=
  Internal.record fun t =>
    Tape.matmul (t := t) (m := m) (n := n) (p := p) aId bId batchA batchB batch

/-- State wrapper around arbitrary-rank `Tape.conv`. -/
def conv {α : Type} [TorchLean.Storage α] [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
    {d inC outC : Nat}
    {kernel stride padding inSpatial : TorchLean.Tensor Nat [d]}
    (kernelId biasId inputId : Nat) (name : String := "conv") : TapeM α Nat :=
  Internal.record fun t => Tape.conv (t := t) (d := d) (inC := inC) (outC := outC)
    (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial)
    kernelId biasId inputId (name := name)

/--
StateT wrapper around `Tape.convTranspose`.

PyTorch comparison: `torch.nn.functional.conv_transpose{d}d` specialized to a single sample
(no batch axis).
-/
def convTranspose {α : Type} [TorchLean.Storage α] [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)]
  {d inC outC : Nat}
  {kernel stride padding : TorchLean.Tensor Nat [d]}
  {inSpatial : TorchLean.Tensor Nat [d]}
  (kernelId biasId inputId : Nat) (name : String := "conv_transpose") : TapeM α Nat :=
  Internal.record fun t => Tape.convTranspose (t := t)
    (d := d) (inC := inC) (outC := outC)
    (kernel := kernel) (stride := stride) (padding := padding)
    (inSpatial := inSpatial) kernelId biasId inputId (name := name)

/-- State wrapper around arbitrary-rank `Tape.maxPool`. -/
def maxPool {α : Type} [TorchLean.Storage α] [Context α]
    {d C : Nat} {inSpatial kernel stride padding : TorchLean.Tensor Nat [d]}
    (xId : Nat) : TapeM α Nat :=
  Internal.record fun t => Tape.maxPool (t := t) (d := d) (C := C)
    (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding) xId

/-- State wrapper around arbitrary-rank `Tape.smoothMaxPool`. -/
def smoothMaxPool {α : Type} [TorchLean.Storage α] [Context α] [DecidableEq α]
    {d C : Nat} {inSpatial kernel stride padding : TorchLean.Tensor Nat [d]}
    (xId : Nat) (beta : α) : TapeM α Nat :=
  Internal.record fun t => Tape.smoothMaxPool (t := t) (d := d) (C := C)
    (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding) xId beta

/-- State wrapper around arbitrary-rank `Tape.avgPool`. -/
def avgPool {α : Type} [TorchLean.Storage α] [Context α]
    {d C : Nat} {inSpatial kernel stride padding : TorchLean.Tensor Nat [d]}
    (xId : Nat) : TapeM α Nat :=
  Internal.record fun t => Tape.avgPool (t := t) (d := d) (C := C)
    (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding) xId

/-- StateT wrapper around `Tape.layerNorm`. PyTorch comparison: `torch.nn.LayerNorm`. -/
def layerNorm {α : Type} [TorchLean.Storage α] [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  {seqLen embedDim : Nat} (h_seq_pos : seqLen > 0) (h_embed_pos : embedDim > 0)
  (xId gammaId betaId : Nat)
  (epsilon : α := TorchLean.normalizationEpsilon) : TapeM α Nat :=
  Internal.record fun t => Tape.layerNorm (t := t)
    (seqLen := seqLen) (embedDim := embedDim) (h_seq_pos := h_seq_pos) (h_embed_pos := h_embed_pos)
    xId gammaId betaId (epsilon := epsilon)

/-- State wrapper around batch normalization over an arbitrary spatial shape. -/
def batchNorm {α : Type} [TorchLean.Storage α] [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
    {channels : Nat} {sSpatial : Shape}
    (hWellFormed : (Shape.dim channels sSpatial).wellFormed)
    (xId gammaId betaId : Nat)
    (epsilon : α := TorchLean.normalizationEpsilon) : TapeM α Nat :=
  Internal.record fun t => Tape.batchNorm (t := t)
    (channels := channels) (sSpatial := sSpatial) hWellFormed xId gammaId betaId
    (epsilon := epsilon)

/-- StateT wrapper around `Tape.attention`. PyTorch comparison:
  `torch.nn.MultiheadAttention` / scaled dot-product attention. -/
def attention {α : Type} [TorchLean.Storage α] [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)]
  {n numHeads dModel headDim : Nat} (h1 : n ≠ 0)
  (wqId wkId wvId woId xId : Nat)
  (mask : Option (Tensor Bool [n, n]) := none) : TapeM α Nat :=
  Internal.record fun t => Tape.attention (t := t)
    (n := n) (numHeads := numHeads) (dModel := dModel) (headDim := headDim) (h1 := h1)
    wqId wkId wvId woId xId mask

/-- StateT wrapper around `Tape.mseLoss`. PyTorch comparison: `torch.nn.functional.mse_loss`. -/
def mseLoss {α : Type} [TorchLean.Storage α]
  [Add α] [Sub α] [Mul α] [Div α] [Zero α] [One α] [NatCast α]
  {s : Shape} (yhatId targetId : Nat) : TapeM α Nat :=
  Internal.record fun t => Tape.mseLoss (t := t) (s := s) yhatId targetId

/-- StateT wrapper around `Tape.sigmoid`. PyTorch comparison: `torch.sigmoid`. -/
def sigmoid {α : Type} [TorchLean.Storage α] [Context α]
  {s : Shape} (xId : Nat) : TapeM α Nat :=
  Internal.record fun t => Tape.sigmoid (t := t) (s := s) xId

/-- StateT wrapper around `Tape.tanh`. PyTorch comparison: `torch.tanh`. -/
def tanh {α : Type} [TorchLean.Storage α] [Context α]
  {s : Shape} (xId : Nat) : TapeM α Nat :=
  Internal.record fun t => Tape.tanh (t := t) (s := s) xId

/-- StateT wrapper around `Tape.softmaxLast`. PyTorch comparison: `torch.softmax(x,
  dim=-1)`. -/
def softmaxLast {α : Type} [TorchLean.Storage α] [Context α]
  {s : Shape} (xId : Nat) : TapeM α Nat :=
  Internal.record fun t => Tape.softmaxLast (t := t) (s := s) xId

/-- StateT wrapper around `Tape.softplus`. PyTorch comparison: `torch.nn.functional.softplus`. -/
def softplus {α : Type} [TorchLean.Storage α] [Context α]
  {s : Shape} (xId : Nat) : TapeM α Nat :=
  Internal.record fun t => Tape.softplus (t := t) (s := s) xId

/-- StateT wrapper around `Tape.exp`. PyTorch comparison: `torch.exp`. -/
def exp {α : Type} [TorchLean.Storage α] [Context α]
  {s : Shape} (xId : Nat) : TapeM α Nat :=
  Internal.record fun t => Tape.exp (t := t) (s := s) xId

/-- Record elementwise sine in the current tape; inputs are angles in radians. -/
def sin {α : Type} [TorchLean.Storage α] [Context α]
    {s : Shape} (xId : Nat) : TapeM α Nat :=
  Internal.record fun t => Tape.sin (t := t) (s := s) xId

/-- Record elementwise cosine in the current tape with derivative `-sin(x)`. -/
def cos {α : Type} [TorchLean.Storage α] [Context α]
    {s : Shape} (xId : Nat) : TapeM α Nat :=
  Internal.record fun t => Tape.cos (t := t) (s := s) xId

/-- StateT wrapper around `Tape.log`. PyTorch comparison: `torch.log`. -/
def log {α : Type} [TorchLean.Storage α] [Context α]
  {s : Shape} (xId : Nat) : TapeM α Nat :=
  Internal.record fun t => Tape.log (t := t) (s := s) xId

/-- StateT wrapper around `Tape.inv`. PyTorch comparison: `torch.reciprocal`. -/
def inv {α : Type} [TorchLean.Storage α] [Context α]
  {s : Shape} (xId : Nat) : TapeM α Nat :=
  Internal.record fun t => Tape.inv (t := t) (s := s) xId

/-- StateT wrapper around `Tape.safeLog` (a numerically-stable `log`). -/
def safeLog {α : Type} [TorchLean.Storage α] [Context α]
  {s : Shape} (xId : Nat) (ε : α := Context.defaultEpsilon) : TapeM α Nat :=
  Internal.record fun t => Tape.safeLog (t := t) (s := s) xId ε

/-- StateT wrapper around `Tape.sum`. PyTorch comparison: `torch.sum`. -/
def sum {α : Type} [TorchLean.Storage α] [Add α] [Zero α]
  {s : Shape} (xId : Nat) : TapeM α Nat :=
  Internal.record fun t => Tape.sum (t := t) (s := s) xId

/--
 Run reverse-mode autodiff from a scalar output and return accumulated gradients.

 This calls `Tape.backwardScalar` on the current tape and returns a `HashMap` from node ids to
 gradient tensors.
 -/
def backwardScalar {α : Type} [TorchLean.Storage α] [Add α] [One α]
  (outId : Nat) : TapeM α (Std.HashMap Nat (Spec.SomeTensor α)) := do
  let t ← get
  liftM (Tape.backwardScalar (t := t) outId)

end TapeM
end Autograd
end Runtime
