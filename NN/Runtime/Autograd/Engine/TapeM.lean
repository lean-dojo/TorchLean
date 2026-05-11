/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Engine.Core

/-!
# TapeM

Tape-building convenience API.

The core autograd runtime (`Runtime.Autograd.Tape`) is pure and explicitly threaded:
each op returns an updated tape plus the new node id. This makes the engine easy to reason
about and convenient for proofs, but it can feel verbose in user code.

`Runtime.Autograd.TapeM` is a small `StateT` wrapper that threads the tape implicitly,
closer to the "define ops; then call backward" ergonomics users expect from frameworks
like PyTorch.

For training scripts/tests, also see `NN.Runtime.Autograd.Utils` which provides small helpers
for common patterns (reading scalar losses, extracting typed grads, simple SGD loops).

## Reading map

- `NN.Runtime.Autograd.Engine.Core` contains the pure tape and low-level node constructors.
- `TapeM.run` / `TapeM.eval` / `TapeM.exec` are the main control-flow wrappers.
- The op wrappers below (`add`, `linear`, `conv2d`, etc.) mirror the `Tape` namespace while
  threading state implicitly.
-/

@[expose] public section


namespace Runtime
namespace Autograd

open Spec
open Tensor

/--
A convenient tape-builder monad.

`TapeM α β` is `StateT (Tape α) Result β`: a pure tape threaded implicitly with errors reported
via `Except String`. This mirrors the common eager style of building a computation and then calling
`backward`, similar to PyTorch's imperative API, but remains purely functional.
-/
abbrev TapeM (α : Type) : Type → Type :=
  StateT (Tape α) Result

namespace TapeM

variable {α β : Type}

/-- Run a `TapeM` computation from an initial tape, returning both the result and the final tape. -/
def run (t : Tape α) (m : TapeM α β) : Result (β × Tape α) :=
  StateT.run m t

/-- Evaluate a `TapeM` computation, discarding the final tape. -/
def eval (t : Tape α) (m : TapeM α β) : Result β := do
  let (a, _) ← run t m
  pure a

/-- Execute a `TapeM` computation, discarding the produced value and returning the final tape. -/
def exec (t : Tape α) (m : TapeM α β) : Result (Tape α) := do
  let (_, t') ← run t m
  pure t'

/-- Get the current tape state. -/
def getTape : TapeM α (Tape α) :=
  get

/-- Replace the current tape state. -/
def setTape (t : Tape α) : TapeM α Unit :=
  set t

/--
Create a leaf node holding a concrete tensor value.

A leaf is the "input tensor" analogue: it has no parents. Setting `requires_grad := true`
corresponds to PyTorch tensors created with `requires_grad=True`.
-/
def leaf {s : Shape}
  (value : Tensor α s) (name : Option String := none) (requires_grad : Bool := true) :
  TapeM α Nat := do
  let t ← get
  let (t', id) := Tape.leaf (t := t) value (name := name) (requires_grad := requires_grad)
  set t'
  pure id

/-- StateT wrapper around `Tape.add`. PyTorch comparison: `torch.add(a, b)`. -/
def add {α : Type} [Add α] [DecidableEq Shape] {s : Shape}
  (aId bId : Nat) : TapeM α Nat := do
  let t ← get
  let (t', id) ← liftM (Tape.add (t := t) (s := s) aId bId)
  set t'
  pure id

/-- StateT wrapper around `Tape.sub`. PyTorch comparison: `torch.sub(a, b)`. -/
def sub {α : Type} [Sub α] [Zero α] [DecidableEq Shape] {s : Shape}
  (aId bId : Nat) : TapeM α Nat := do
  let t ← get
  let (t', id) ← liftM (Tape.sub (t := t) (s := s) aId bId)
  set t'
  pure id

/-- StateT wrapper around `Tape.mul`. PyTorch comparison: `torch.mul(a, b)`. -/
def mul {α : Type} [Mul α] [DecidableEq Shape] {s : Shape}
  (aId bId : Nat) : TapeM α Nat := do
  let t ← get
  let (t', id) ← liftM (Tape.mul (t := t) (s := s) aId bId)
  set t'
  pure id

/-- StateT wrapper around `Tape.scale`. PyTorch comparison: `c * x` / `torch.mul(x, c)`. -/
def scale {α : Type} [Mul α] [DecidableEq Shape] {s : Shape}
  (xId : Nat) (c : α) : TapeM α Nat := do
  let t ← get
  let (t', id) ← liftM (Tape.scale (t := t) (s := s) xId c)
  set t'
  pure id

/-- StateT wrapper around `Tape.abs`. PyTorch comparison: `torch.abs(x)`. -/
def abs {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {s : Shape} (xId : Nat) : TapeM α Nat := do
  let t ← get
  let (t', id) ← liftM (Tape.abs (t := t) (s := s) xId)
  set t'
  pure id

/-- StateT wrapper around `Tape.sqrt`. PyTorch comparison: `torch.sqrt(x)`. -/
def sqrt {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {s : Shape} (xId : Nat) : TapeM α Nat := do
  let t ← get
  let (t', id) ← liftM (Tape.sqrt (t := t) (s := s) xId)
  set t'
  pure id

/-- StateT wrapper around `Tape.clamp`. PyTorch comparison: `torch.clamp(x, min, max)`. -/
def clamp {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {s : Shape} (xId : Nat) (minVal maxVal : α) : TapeM α Nat := do
  let t ← get
  let (t', id) ← liftM (Tape.clamp (t := t) (s := s) xId minVal maxVal)
  set t'
  pure id

/-- StateT wrapper around `Tape.max`. PyTorch comparison: `torch.maximum(a, b)`. -/
def max {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {s : Shape} (aId bId : Nat) : TapeM α Nat := do
  let t ← get
  let (t', id) ← liftM (Tape.max (t := t) (s := s) aId bId)
  set t'
  pure id

/-- StateT wrapper around `Tape.min`. PyTorch comparison: `torch.minimum(a, b)`. -/
def min {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {s : Shape} (aId bId : Nat) : TapeM α Nat := do
  let t ← get
  let (t', id) ← liftM (Tape.min (t := t) (s := s) aId bId)
  set t'
  pure id

/-- StateT wrapper around `Tape.relu`. PyTorch comparison: `torch.nn.functional.relu(x)`. -/
def relu {α : Type}
  [Mul α] [Zero α] [Max α] [One α] [LT α]
  [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {s : Shape} (xId : Nat) : TapeM α Nat := do
  let t ← get
  let (t', id) ← liftM (Tape.relu (t := t) (s := s) xId)
  set t'
  pure id

/-- StateT wrapper around `Tape.linear`. PyTorch comparison: `torch.nn.functional.linear`. -/
def linear {α : Type} [Add α] [Mul α] [Zero α] [DecidableEq Shape]
  {inDim outDim : Nat} (wId bId xId : Nat) : TapeM α Nat := do
  let t ← get
  let (t', id) ← liftM (Tape.linear (t := t) (inDim := inDim) (outDim := outDim) wId bId xId)
  set t'
  pure id

/-- StateT wrapper around `Tape.matmul`. PyTorch comparison: `torch.matmul(a, b)`. -/
def matmul {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {m n p : Nat} (aId bId : Nat) : TapeM α Nat := do
  let t ← get
  let (t', id) ← liftM (Tape.matmul (t := t) (m := m) (n := n) (p := p) aId bId)
  set t'
  pure id

/-- StateT wrapper around `Tape.concat_vectors`. PyTorch comparison: `torch.cat([a,b], dim=0)` for
  vectors. -/
def concatVectors {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq
  Shape]
  {n m : Nat} (aId bId : Nat) : TapeM α Nat := do
  let t ← get
  let (t', id) ← liftM (Tape.concatVectors (t := t) (n := n) (m := m) aId bId)
  set t'
  pure id

/--
StateT wrapper around `Tape.conv2d`.

PyTorch comparison: `torch.nn.functional.conv2d` (this codebase uses a single-image specialization;
see `Tape.conv2d` for the exact shape conventions).
-/
def conv2d {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {inC outC kH kW stride padding inH inW : Nat}
  {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
  (kernelId biasId inputId : Nat) : TapeM α Nat := do
  let t ← get
  let (t', id) ← liftM (Tape.conv2d (t := t)
    (inC := inC) (outC := outC) (kH := kH) (kW := kW)
    (stride := stride) (padding := padding) (inH := inH) (inW := inW)
    (h1 := h1) (h2 := h2) (h3 := h3) kernelId biasId inputId)
  set t'
  pure id

/--
StateT wrapper around `Tape.conv_transpose`.

PyTorch comparison: `torch.nn.functional.conv_transpose{d}d` specialized to a single sample
(no batch axis).
-/
def convTranspose {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  [DecidableEq Shape]
  {d inC outC : Nat}
  {kernel stride padding : Vector Nat d}
  {inSpatial : Vector Nat d}
  (kernelId biasId inputId : Nat) (name : String := "conv_transpose") : TapeM α Nat := do
  let t ← get
  let (t', id) ← liftM (Tape.convTranspose (t := t)
    (d := d) (inC := inC) (outC := outC)
    (kernel := kernel) (stride := stride) (padding := padding)
    (inSpatial := inSpatial) kernelId biasId inputId (name := name))
  set t'
  pure id

/--
StateT wrapper around `Tape.conv_transpose2d`.

PyTorch comparison: `torch.nn.functional.conv_transpose2d` (single-image specialization; see
`Tape.conv_transpose2d` for exact shape conventions).
-/
def convTranspose2d {α : Type} [Context α] [DecidableEq Shape]
  {inC outC kH kW stride padding inH inW : Nat}
  {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
  (kernelId biasId inputId : Nat) : TapeM α Nat := do
  let t ← get
  let (t', id) ← liftM (Tape.convTranspose2d (t := t)
    (inC := inC) (outC := outC) (kH := kH) (kW := kW)
    (stride := stride) (padding := padding) (inH := inH) (inW := inW)
    (h1 := h1) (h2 := h2) (h3 := h3) kernelId biasId inputId)
  set t'
  pure id

/-- StateT wrapper around `Tape.max_pool2d`. PyTorch comparison: `torch.nn.functional.max_pool2d`.
  -/
def maxPool2d {α : Type} [Context α] [DecidableEq Shape]
  {kH kW inH inW inC stride : Nat} {h1 : kH ≠ 0} {h2 : kW ≠ 0}
  (xId : Nat) : TapeM α Nat := do
  let t ← get
  let (t', id) ← liftM (Tape.maxPool2d (t := t)
    (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride)
    (h1 := h1) (h2 := h2) xId)
  set t'
  pure id

/-- StateT wrapper around `Tape.max_pool2d_pad`. PyTorch comparison:
  `torch.nn.functional.max_pool2d` with padding. -/
def maxPool2dPad {α : Type} [Context α] [DecidableEq Shape]
  {kH kW inH inW inC stride padding : Nat} {h1 : kH ≠ 0} {h2 : kW ≠ 0}
  (xId : Nat) : TapeM α Nat := do
  let t ← get
  let (t', id) ← liftM (Tape.maxPool2dPad (t := t)
    (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride) (padding :=
      padding)
    (h1 := h1) (h2 := h2) xId)
  set t'
  pure id

/--
 StateT wrapper around `Tape.smooth_max_pool2d`.

 This is a differentiable (soft) approximation to max-pooling controlled by `beta`.
 -/
def smoothMaxPool2d {α : Type} [Context α] [DecidableEq Shape]
  {kH kW inH inW inC stride : Nat} {h1 : kH ≠ 0} {h2 : kW ≠ 0}
  (xId : Nat) (beta : α) : TapeM α Nat := do
  let t ← get
  let (t', id) ← liftM (Tape.smoothMaxPool2d (t := t)
    (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride)
    (h1 := h1) (h2 := h2) xId beta)
  set t'
  pure id

/-- StateT wrapper around `Tape.avg_pool2d`. PyTorch comparison: `torch.nn.functional.avg_pool2d`.
  -/
def avgPool2d {α : Type} [Context α] [DecidableEq Shape]
  {kH kW inH inW inC stride : Nat} (h1 : kH ≠ 0) (h2 : kW ≠ 0)
  (xId : Nat) : TapeM α Nat := do
  let t ← get
  let (t', id) ← liftM (Tape.avgPool2d (t := t)
    (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride)
    (h1 := h1) (h2 := h2) xId)
  set t'
  pure id

/-- StateT wrapper around `Tape.avg_pool2d_pad`. PyTorch comparison:
  `torch.nn.functional.avg_pool2d` with padding. -/
def avgPool2dPad {α : Type} [Context α] [DecidableEq Shape]
  {kH kW inH inW inC stride padding : Nat} (h1 : kH ≠ 0) (h2 : kW ≠ 0)
  (xId : Nat) : TapeM α Nat := do
  let t ← get
  let (t', id) ← liftM (Tape.avgPool2dPad (t := t)
    (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride) (padding :=
      padding)
    (h1 := h1) (h2 := h2) xId)
  set t'
  pure id

/-- StateT wrapper around `Tape.layer_norm`. PyTorch comparison: `torch.nn.LayerNorm`. -/
def layerNorm {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {seqLen embedDim : Nat} (h_seq_pos : seqLen > 0) (h_embed_pos : embedDim > 0)
  (xId gammaId betaId : Nat) : TapeM α Nat := do
  let t ← get
  let (t', id) ← liftM (Tape.layerNorm (t := t)
    (seqLen := seqLen) (embedDim := embedDim) (h_seq_pos := h_seq_pos) (h_embed_pos := h_embed_pos)
    xId gammaId betaId)
  set t'
  pure id

/-- StateT wrapper around `Tape.batchnorm_channel_first`. PyTorch comparison: `torch.nn.BatchNorm2d`
  in channel-first layout. -/
def batchnormChannelFirst {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  [DecidableEq Shape]
  {channels height width : Nat}
  (h_c : channels > 0) (h_h : height > 0) (h_w : width > 0)
  (xId gammaId betaId : Nat) : TapeM α Nat := do
  let t ← get
  let (t', id) ← liftM (Tape.batchnormChannelFirst (t := t)
    (channels := channels) (height := height) (width := width)
    (h_c := h_c) (h_h := h_h) (h_w := h_w) xId gammaId betaId)
  set t'
  pure id

/-- StateT wrapper around `Tape.multi_head_attention`. PyTorch comparison:
  `torch.nn.MultiheadAttention` / scaled dot-product attention. -/
def multiHeadAttention {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq
  Shape]
  {n numHeads dModel headDim : Nat} (h1 : n ≠ 0)
  (wqId wkId wvId woId xId : Nat)
  (mask : Option (Tensor Bool (.dim n (.dim n .scalar))) := none) : TapeM α Nat := do
  let t ← get
  let (t', id) ← liftM (Tape.multiHeadAttention (t := t)
    (n := n) (numHeads := numHeads) (dModel := dModel) (headDim := headDim) (h1 := h1)
    wqId wkId wvId woId xId mask)
  set t'
  pure id

/-- StateT wrapper around `Tape.mse_loss`. PyTorch comparison: `torch.nn.functional.mse_loss`. -/
def mseLoss {α : Type}
  [Add α] [Sub α] [Mul α] [Div α] [Zero α] [One α] [Coe Nat α] [DecidableEq Shape]
  {s : Shape} (yhatId targetId : Nat) : TapeM α Nat := do
  let t ← get
  let (t', id) ← liftM (Tape.mseLoss (t := t) (s := s) yhatId targetId)
  set t'
  pure id

/-- StateT wrapper around `Tape.sigmoid`. PyTorch comparison: `torch.sigmoid`. -/
def sigmoid {α : Type} [Context α] [DecidableEq Shape]
  {s : Shape} (xId : Nat) : TapeM α Nat := do
  let t ← get
  let (t', id) ← liftM (Tape.sigmoid (t := t) (s := s) xId)
  set t'
  pure id

/-- StateT wrapper around `Tape.tanh`. PyTorch comparison: `torch.tanh`. -/
def tanh {α : Type} [Context α] [DecidableEq Shape]
  {s : Shape} (xId : Nat) : TapeM α Nat := do
  let t ← get
  let (t', id) ← liftM (Tape.tanh (t := t) (s := s) xId)
  set t'
  pure id

/-- StateT wrapper around `Tape.softmax` (last-axis). PyTorch comparison: `torch.softmax(x,
  dim=-1)`. -/
def softmax {α : Type} [Context α] [DecidableEq Shape]
  {s : Shape} (xId : Nat) : TapeM α Nat := do
  let t ← get
  let (t', id) ← liftM (Tape.softmax (t := t) (s := s) xId)
  set t'
  pure id

/-- StateT wrapper around `Tape.softplus`. PyTorch comparison: `torch.nn.functional.softplus`. -/
def softplus {α : Type} [Context α] [DecidableEq Shape]
  {s : Shape} (xId : Nat) : TapeM α Nat := do
  let t ← get
  let (t', id) ← liftM (Tape.softplus (t := t) (s := s) xId)
  set t'
  pure id

/-- StateT wrapper around `Tape.exp`. PyTorch comparison: `torch.exp`. -/
def exp {α : Type} [Context α] [DecidableEq Shape]
  {s : Shape} (xId : Nat) : TapeM α Nat := do
  let t ← get
  let (t', id) ← liftM (Tape.exp (t := t) (s := s) xId)
  set t'
  pure id

/-- StateT wrapper around `Tape.log`. PyTorch comparison: `torch.log`. -/
def log {α : Type} [Context α] [DecidableEq Shape]
  {s : Shape} (xId : Nat) : TapeM α Nat := do
  let t ← get
  let (t', id) ← liftM (Tape.log (t := t) (s := s) xId)
  set t'
  pure id

/-- StateT wrapper around `Tape.inv`. PyTorch comparison: `torch.reciprocal`. -/
def inv {α : Type} [Context α] [DecidableEq Shape]
  {s : Shape} (xId : Nat) : TapeM α Nat := do
  let t ← get
  let (t', id) ← liftM (Tape.inv (t := t) (s := s) xId)
  set t'
  pure id

/-- StateT wrapper around `Tape.safe_log` (a numerically-stable `log`). -/
def safeLog {α : Type} [Context α] [DecidableEq Shape]
  {s : Shape} (xId : Nat) (ε : α := Numbers.epsilon) : TapeM α Nat := do
  let t ← get
  let (t', id) ← liftM (Tape.safeLog (t := t) (s := s) xId ε)
  set t'
  pure id

/-- StateT wrapper around `Tape.sum`. PyTorch comparison: `torch.sum`. -/
def sum {α : Type} [Add α] [Zero α] [DecidableEq Shape]
  {s : Shape} (xId : Nat) : TapeM α Nat := do
  let t ← get
  let (t', id) ← liftM (Tape.sum (t := t) (s := s) xId)
  set t'
  pure id

/--
 Run reverse-mode autodiff from a scalar output and return accumulated gradients.

 This calls `Tape.backwardScalar` on the current tape and returns a `HashMap` from node ids to
 gradient tensors.
 -/
def backwardScalar {α : Type} [Add α] [One α] [DecidableEq Shape]
  (outId : Nat) : TapeM α (Std.HashMap Nat (Runtime.AnyTensor α)) := do
  let t ← get
  liftM (Tape.backwardScalar (t := t) outId)

end TapeM
end Autograd
end Runtime
