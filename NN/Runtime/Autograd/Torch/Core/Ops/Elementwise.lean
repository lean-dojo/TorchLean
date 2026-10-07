/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core.Ops.Dispatch
public import NN.Runtime.Autograd.Engine.Core.ActivationsLoss
public import NN.Runtime.Autograd.Engine.Core.Elementwise
public import NN.Runtime.Autograd.Engine.LibTorch.Ops.Elementwise
public import NN.Runtime.Autograd.Engine.LibTorch.Ops.NormSoftmax

/-!
# Eager Tensor Operations

PyTorch-style tensor operations backed by the eager CPU/CUDA tapes. These wrappers record runtime
nodes, dispatch CUDA kernels when requested, and preserve the typed `TensorRef` surface.
-/


@[expose] public section

namespace Runtime
namespace Autograd
namespace Torch

open Spec TorchLean TorchLean.Tensor

namespace Internal

namespace EagerSession

/-! ## Elementwise operations -/

/-- Record elementwise addition `a + b`. PyTorch: `torch.add`. -/
def add {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Add α]
  {sh : Shape}
  (a b : TensorRef α sh) : IO (TensorRef α sh) := do
  let cpu := do
    s.recordCpu fun t0 => keepTapeOnError t0 <|
      Runtime.Autograd.Tape.add (t := t0) (s := sh) a.id b.id
  let cuda := do
    s.recordCuda fun t0 => keepTapeOnError t0 <|
      Runtime.Autograd.LibTorch.Tape.add (t := t0) (s := sh) a.id b.id
  executeRecorded (α := α) s .add #[a.identity?, b.identity?] cpu cuda

/-- Record elementwise subtraction `a - b`. PyTorch: `torch.sub`. -/
def sub {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Sub α] [Zero α] {sh : Shape}
  (a b : TensorRef α sh) : IO (TensorRef α sh) := do
  let cpu := do
    s.recordCpu fun t0 => keepTapeOnError t0 <|
      Runtime.Autograd.Tape.sub (t := t0) (s := sh) a.id b.id
  let cuda := do
    s.recordCuda fun t0 => keepTapeOnError t0 <|
      Runtime.Autograd.LibTorch.Tape.sub (t := t0) (s := sh) a.id b.id
  executeRecorded (α := α) s .sub #[a.identity?, b.identity?] cpu cuda

/-- Record elementwise multiplication `a * b`. PyTorch: `torch.mul`. -/
def mul {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Mul α]
  {sh : Shape}
  (a b : TensorRef α sh) : IO (TensorRef α sh) := do
  let cpu := do
    s.recordCpu fun t0 => keepTapeOnError t0 <|
      Runtime.Autograd.Tape.mul (t := t0) (s := sh) a.id b.id
  let cuda := do
    s.recordCuda fun t0 => keepTapeOnError t0 <|
      Runtime.Autograd.LibTorch.Tape.mul (t := t0) (s := sh) a.id b.id
  executeRecorded (α := α) s .mul #[a.identity?, b.identity?] cpu cuda

/-- Record scaling by a scalar constant. PyTorch: `x * c`. -/
def scale {α : Type} [TorchLean.Storage α] [TensorTransfer α] (s : EagerSession α) [Mul α]
  {sh : Shape}
  (x : TensorRef α sh) (c : α) : IO (TensorRef α sh) := do
  let cpu := do
    s.recordCpu fun t0 => keepTapeOnError t0 <|
      Runtime.Autograd.Tape.scale (t := t0) (s := sh) x.id c
  let cuda := do
    let cF ← TensorTransfer.toFloat (α := α) c
    s.recordCuda fun t0 => keepTapeOnError t0 <|
      Runtime.Autograd.LibTorch.Tape.scale (t := t0) (s := sh) x.id cF
  executeRecorded (α := α) s .scale #[x.identity?] cpu cuda

/-- Record elementwise absolute value. PyTorch: `torch.abs`. -/
def abs {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)]
  {sh : Shape} (x : TensorRef α sh) : IO (TensorRef α sh) := do
  let cpu := do
    s.recordCpu fun t0 => keepTapeOnError t0 <|
      Runtime.Autograd.Tape.abs (t := t0) (s := sh) x.id
  let cuda := do
    s.recordCuda fun t0 => keepTapeOnError t0 <|
      Runtime.Autograd.LibTorch.Tape.abs (t := t0) (s := sh) x.id
  executeRecorded (α := α) s .abs #[x.identity?] cpu cuda

/-- Record elementwise square root. PyTorch: `torch.sqrt`. -/
def sqrt {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)]
  {sh : Shape} (x : TensorRef α sh) : IO (TensorRef α sh) := do
  let cpu := do
    s.recordCpu fun t0 => keepTapeOnError t0 <|
      Runtime.Autograd.Tape.sqrt (t := t0) (s := sh) x.id
  let cuda := do
    s.recordCuda fun t0 => keepTapeOnError t0 <|
      Runtime.Autograd.LibTorch.Tape.sqrt (t := t0) (s := sh) x.id
  executeRecorded (α := α) s .sqrt #[x.identity?] cpu cuda

/-- Record elementwise clamp to `[minVal,maxVal]`. PyTorch: `torch.clamp`. -/
def clamp {α : Type} [TorchLean.Storage α] [TensorTransfer α] (s : EagerSession α) [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)]
  {sh : Shape} (x : TensorRef α sh) (minVal maxVal : α) : IO (TensorRef α sh) := do
  let cpu := do
    s.recordCpu fun t0 => keepTapeOnError t0 <|
      Runtime.Autograd.Tape.clamp (t := t0) (s := sh) x.id minVal maxVal
  let cuda := do
    let lo ← TensorTransfer.toFloat (α := α) minVal
    let hi ← TensorTransfer.toFloat (α := α) maxVal
    s.recordCuda fun t0 => keepTapeOnError t0 <|
      Runtime.Autograd.LibTorch.Tape.clamp (t := t0) (s := sh) x.id lo hi
  executeRecorded (α := α) s .clamp #[x.identity?] cpu cuda

/-- Record elementwise maximum. PyTorch: `torch.maximum`. -/
def max {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)]
  {sh : Shape} (a b : TensorRef α sh) : IO (TensorRef α sh) := do
  let cpu := do
    s.recordCpu fun t0 => keepTapeOnError t0 <|
      Runtime.Autograd.Tape.max (t := t0) (s := sh) a.id b.id
  let cuda := do
    s.recordCuda fun t0 => keepTapeOnError t0 <|
      Runtime.Autograd.LibTorch.Tape.max (t := t0) (s := sh) a.id b.id
  executeRecorded (α := α) s .max #[a.identity?, b.identity?] cpu cuda

/-- Record elementwise minimum. PyTorch: `torch.minimum`. -/
def min {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)]
  {sh : Shape} (a b : TensorRef α sh) : IO (TensorRef α sh) := do
  let cpu := do
    s.recordCpu fun t0 => keepTapeOnError t0 <|
      Runtime.Autograd.Tape.min (t := t0) (s := sh) a.id b.id
  let cuda := do
    s.recordCuda fun t0 => keepTapeOnError t0 <|
      Runtime.Autograd.LibTorch.Tape.min (t := t0) (s := sh) a.id b.id
  executeRecorded (α := α) s .min #[a.identity?, b.identity?] cpu cuda

/-- Record elementwise ReLU. PyTorch: `torch.relu` / `torch.nn.functional.relu`. -/
def relu {α : Type} [TorchLean.Storage α] (s : EagerSession α)
  [Mul α] [Zero α] [Max α] [BEq α] [One α] [LT α]
  [DecidableRel ((· > ·) : α → α → Prop)]
  {sh : Shape} (x : TensorRef α sh) : IO (TensorRef α sh) := do
  let cpu := do
    s.recordCpu fun t0 => keepTapeOnError t0 <|
      Runtime.Autograd.Tape.relu (t := t0) (s := sh) x.id
  let cuda := do
    s.recordCuda fun t0 => keepTapeOnError t0 <|
      Runtime.Autograd.LibTorch.Tape.relu (t := t0) (s := sh) x.id
  executeRecorded (α := α) s .relu #[x.identity?] cpu cuda

/-- Record elementwise sigmoid. PyTorch: `torch.sigmoid`. -/
def sigmoid {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Context α]
  {sh : Shape} (x : TensorRef α sh) : IO (TensorRef α sh) := do
  let cpu := do
    s.recordCpu fun t0 => keepTapeOnError t0 <|
      Runtime.Autograd.Tape.sigmoid (t := t0) (s := sh) x.id
  let cuda := do
    s.recordCuda fun t0 => keepTapeOnError t0 <|
      Runtime.Autograd.LibTorch.Tape.sigmoid (t := t0) (s := sh) x.id
  executeRecorded (α := α) s .sigmoid #[x.identity?] cpu cuda

/-- Record elementwise tanh. PyTorch: `torch.tanh`. -/
def tanh {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Context α]
  {sh : Shape} (x : TensorRef α sh) : IO (TensorRef α sh) := do
  let cpu := do
    s.recordCpu fun t0 => keepTapeOnError t0 <|
      Runtime.Autograd.Tape.tanh (t := t0) (s := sh) x.id
  let cuda := do
    s.recordCuda fun t0 => keepTapeOnError t0 <|
      Runtime.Autograd.LibTorch.Tape.tanh (t := t0) (s := sh) x.id
  executeRecorded (α := α) s .tanh #[x.identity?] cpu cuda

/-- Record tanh-approximate GELU as one tape operation. -/
def gelu {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Context α]
  {sh : Shape} (x : TensorRef α sh) : IO (TensorRef α sh) := do
  let cpu := do
    s.recordCpu fun t0 => keepTapeOnError t0 <|
      Runtime.Autograd.Tape.gelu (t := t0) (s := sh) x.id
  let cuda := do
    s.recordCuda fun t0 => keepTapeOnError t0 <|
      Runtime.Autograd.LibTorch.Tape.gelu (t := t0) (s := sh) x.id
  executeRecorded (α := α) s .gelu #[x.identity?] cpu cuda

/--
Record shape-preserving softmax over the last axis.

PyTorch comparison: `torch.softmax(x, dim=-1)`.
-/
def softmaxLast {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Context α]
  {sh : Shape} (x : TensorRef α sh) : IO (TensorRef α sh) := do
  let cpu := do
    s.recordCpu fun t0 => keepTapeOnError t0 <|
      Runtime.Autograd.Tape.softmaxLast (t := t0) (s := sh) x.id
  let cuda := do
    s.recordCuda fun t0 => keepTapeOnError t0 <|
      Runtime.Autograd.LibTorch.Tape.softmaxLast (t := t0) (s := sh) x.id
  executeRecorded (α := α) s .softmax #[x.identity?] cpu cuda

/--
Record stable log-softmax (shape-preserving, last-axis convention).

PyTorch comparison: `torch.nn.functional.log_softmax(x, dim=-1)`.
-/
def logSoftmaxLast {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Context α]
  {sh : Shape} (x : TensorRef α sh) : IO (TensorRef α sh) := do
  let cpu := do
    s.recordCpu fun t0 => keepTapeOnError t0 <|
      Runtime.Autograd.Tape.logSoftmaxLast (t := t0) (s := sh) x.id
  let cuda := do
    s.recordCuda fun t0 => keepTapeOnError t0 <|
      Runtime.Autograd.LibTorch.Tape.logSoftmaxLast (t := t0) (s := sh) x.id
  executeRecorded (α := α) s .logSoftmax #[x.identity?] cpu cuda

/-- Record elementwise softplus. PyTorch: `torch.nn.functional.softplus`. -/
def softplus {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Context α]
  {sh : Shape} (x : TensorRef α sh) : IO (TensorRef α sh) := do
  let cpu := do
    s.recordCpu fun t0 => keepTapeOnError t0 <|
      Runtime.Autograd.Tape.softplus (t := t0) (s := sh) x.id
  let cuda := do
    s.recordCuda fun t0 => keepTapeOnError t0 <|
      Runtime.Autograd.LibTorch.Tape.softplus (t := t0) (s := sh) x.id
  executeRecorded (α := α) s .softplus #[x.identity?] cpu cuda

/-- Record elementwise exponential. PyTorch: `torch.exp`. -/
def exp {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Context α]
  {sh : Shape} (x : TensorRef α sh) : IO (TensorRef α sh) := do
  let cpu := do
    s.recordCpu fun t0 => keepTapeOnError t0 <|
      Runtime.Autograd.Tape.exp (t := t0) (s := sh) x.id
  let cuda := do
    s.recordCuda fun t0 => keepTapeOnError t0 <|
      Runtime.Autograd.LibTorch.Tape.exp (t := t0) (s := sh) x.id
  executeRecorded (α := α) s .exp #[x.identity?] cpu cuda

/-- Record elementwise sine and dispatch its value and VJP through the selected backend. -/
def sin {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Context α]
    {sh : Shape} (x : TensorRef α sh) : IO (TensorRef α sh) := do
  let cpu := do
    s.recordCpu fun t0 => keepTapeOnError t0 <|
      Runtime.Autograd.Tape.sin (t := t0) (s := sh) x.id
  let cuda := do
    s.recordCuda fun t0 => keepTapeOnError t0 <|
      Runtime.Autograd.LibTorch.Tape.sin (t := t0) (s := sh) x.id
  executeRecorded (α := α) s .sin #[x.identity?] cpu cuda

/-- Record elementwise cosine with the selected backend's `-sin(x) * dLdy` backward rule. -/
def cos {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Context α]
    {sh : Shape} (x : TensorRef α sh) : IO (TensorRef α sh) := do
  let cpu := do
    s.recordCpu fun t0 => keepTapeOnError t0 <|
      Runtime.Autograd.Tape.cos (t := t0) (s := sh) x.id
  let cuda := do
    s.recordCuda fun t0 => keepTapeOnError t0 <|
      Runtime.Autograd.LibTorch.Tape.cos (t := t0) (s := sh) x.id
  executeRecorded (α := α) s .cos #[x.identity?] cpu cuda

/-- Record elementwise log. PyTorch: `torch.log`. -/
def log {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Context α]
  {sh : Shape} (x : TensorRef α sh) : IO (TensorRef α sh) := do
  let cpu := do
    s.recordCpu fun t0 => keepTapeOnError t0 <|
      Runtime.Autograd.Tape.log (t := t0) (s := sh) x.id
  let cuda := do
    s.recordCuda fun t0 => keepTapeOnError t0 <|
      Runtime.Autograd.LibTorch.Tape.log (t := t0) (s := sh) x.id
  executeRecorded (α := α) s .log #[x.identity?] cpu cuda

/-- Record elementwise inverse `1/x`. PyTorch: `torch.reciprocal`. -/
def inv {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Context α]
  {sh : Shape} (x : TensorRef α sh) : IO (TensorRef α sh) := do
  let cpu := do
    s.recordCpu fun t0 => keepTapeOnError t0 <|
      Runtime.Autograd.Tape.inv (t := t0) (s := sh) x.id
  let cuda := do
    s.recordCuda fun t0 => keepTapeOnError t0 <|
      Runtime.Autograd.LibTorch.Tape.inv (t := t0) (s := sh) x.id
  executeRecorded (α := α) s .inv #[x.identity?] cpu cuda

/--
Record `log(softplus(x) + ε)` and its derivative.

Backward multiplies the incoming gradient by `sigmoid(x) / (softplus(x) + ε)`. CUDA converts
`ε` to binary32; choose a value that stays positive after conversion if softplus can round to
zero. Neither backend checks positivity here or guarantees finite results for nonfinite inputs.
-/
def safeLog {α : Type} [TorchLean.Storage α] [TensorTransfer α] (s : EagerSession α)
  [Context α]
  {sh : Shape} (x : TensorRef α sh) (ε : α := Context.defaultEpsilon) : IO (TensorRef α sh) := do
  let cpu := do
    s.recordCpu fun t0 => keepTapeOnError t0 <|
      Runtime.Autograd.Tape.safeLog (t := t0) (s := sh) x.id (ε := ε)
  let cuda := do
    let epsF ← TensorTransfer.toFloat (α := α) ε
    s.recordCuda fun t0 => keepTapeOnError t0 <|
      Runtime.Autograd.LibTorch.Tape.safeLog (t := t0) (s := sh) x.id epsF
  executeRecorded (α := α) s .safeLog #[x.identity?] cpu cuda

end EagerSession

end Internal
end Torch
end Autograd
end Runtime
