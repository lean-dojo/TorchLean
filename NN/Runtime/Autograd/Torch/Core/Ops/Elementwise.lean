/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core.Ops.Dispatch
public import NN.Runtime.Autograd.Engine.Core.ActivationsLoss
public import NN.Runtime.Autograd.Engine.Core.Elementwise
public import NN.Runtime.Autograd.Engine.Cuda.Ops.Elementwise
public import NN.Runtime.Autograd.Engine.Cuda.Ops.NormSoftmax

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
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.add (t := t0) (s := sh) a.id b.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.add (t := t0) (s := sh) a.id b.id
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s .add #[a.identity?, b.identity?] cpu cuda

/-- Record elementwise subtraction `a - b`. PyTorch: `torch.sub`. -/
def sub {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Sub α] [Zero α] {sh : Shape}
  (a b : TensorRef α sh) : IO (TensorRef α sh) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.sub (t := t0) (s := sh) a.id b.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.sub (t := t0) (s := sh) a.id b.id
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s .sub #[a.identity?, b.identity?] cpu cuda

/-- Record elementwise multiplication `a * b`. PyTorch: `torch.mul`. -/
def mul {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Mul α]
  {sh : Shape}
  (a b : TensorRef α sh) : IO (TensorRef α sh) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.mul (t := t0) (s := sh) a.id b.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.mul (t := t0) (s := sh) a.id b.id
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s .mul #[a.identity?, b.identity?] cpu cuda

/-- Record scaling by a scalar constant. PyTorch: `x * c`. -/
def scale {α : Type} [TorchLean.Storage α] [TensorTransfer α] (s : EagerSession α) [Mul α]
  {sh : Shape}
  (x : TensorRef α sh) (c : α) : IO (TensorRef α sh) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.scale (t := t0) (s := sh) x.id c)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let cF ← TensorTransfer.toFloat (α := α) c
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.scale (t := t0) (s := sh) x.id cF
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s .scale #[x.identity?] cpu cuda

/-- Record elementwise absolute value. PyTorch: `torch.abs`. -/
def abs {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)]
  {sh : Shape} (x : TensorRef α sh) : IO (TensorRef α sh) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.abs (t := t0) (s := sh) x.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.abs (t := t0) (s := sh) x.id
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s .abs #[x.identity?] cpu cuda

/-- Record elementwise square root. PyTorch: `torch.sqrt`. -/
def sqrt {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)]
  {sh : Shape} (x : TensorRef α sh) : IO (TensorRef α sh) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.sqrt (t := t0) (s := sh) x.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.sqrt (t := t0) (s := sh) x.id
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s .sqrt #[x.identity?] cpu cuda

/-- Record elementwise clamp to `[minVal,maxVal]`. PyTorch: `torch.clamp`. -/
def clamp {α : Type} [TorchLean.Storage α] [TensorTransfer α] (s : EagerSession α) [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)]
  {sh : Shape} (x : TensorRef α sh) (minVal maxVal : α) : IO (TensorRef α sh) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.clamp (t := t0) (s := sh) x.id minVal maxVal)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let lo ← TensorTransfer.toFloat (α := α) minVal
    let hi ← TensorTransfer.toFloat (α := α) maxVal
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.clamp (t := t0) (s := sh) x.id lo hi
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s .clamp #[x.identity?] cpu cuda

/-- Record elementwise maximum. PyTorch: `torch.maximum`. -/
def max {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)]
  {sh : Shape} (a b : TensorRef α sh) : IO (TensorRef α sh) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.max (t := t0) (s := sh) a.id b.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.max (t := t0) (s := sh) a.id b.id
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s .max #[a.identity?, b.identity?] cpu cuda

/-- Record elementwise minimum. PyTorch: `torch.minimum`. -/
def min {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)]
  {sh : Shape} (a b : TensorRef α sh) : IO (TensorRef α sh) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.min (t := t0) (s := sh) a.id b.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.min (t := t0) (s := sh) a.id b.id
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s .min #[a.identity?, b.identity?] cpu cuda

/-- Record elementwise ReLU. PyTorch: `torch.relu` / `torch.nn.functional.relu`. -/
def relu {α : Type} [TorchLean.Storage α] (s : EagerSession α)
  [Mul α] [Zero α] [Max α] [BEq α] [One α] [LT α]
  [DecidableRel ((· > ·) : α → α → Prop)]
  {sh : Shape} (x : TensorRef α sh) : IO (TensorRef α sh) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.relu (t := t0) (s := sh) x.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.relu (t := t0) (s := sh) x.id
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s .relu #[x.identity?] cpu cuda

/-- Record elementwise sigmoid. PyTorch: `torch.sigmoid`. -/
def sigmoid {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Context α]
  {sh : Shape} (x : TensorRef α sh) : IO (TensorRef α sh) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.sigmoid (t := t0) (s := sh) x.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Cuda.Tape.sigmoid (t := t0) (s := sh) x.id)
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s .sigmoid #[x.identity?] cpu cuda

/-- Record elementwise tanh. PyTorch: `torch.tanh`. -/
def tanh {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Context α]
  {sh : Shape} (x : TensorRef α sh) : IO (TensorRef α sh) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.tanh (t := t0) (s := sh) x.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Cuda.Tape.tanh (t := t0) (s := sh) x.id)
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s .tanh #[x.identity?] cpu cuda

/-- Record tanh-approximate GELU as one tape operation. -/
def gelu {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Context α]
  {sh : Shape} (x : TensorRef α sh) : IO (TensorRef α sh) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.gelu (t := t0) (s := sh) x.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Cuda.Tape.gelu (t := t0) (s := sh) x.id)
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s .gelu #[x.identity?] cpu cuda

/--
Record softmax (shape-preserving).

PyTorch comparison: `torch.softmax(x, dim=...)` (dimension convention is chosen by the underlying
  tape op).
-/
def softmaxLast {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Context α]
  {sh : Shape} (x : TensorRef α sh) : IO (TensorRef α sh) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.softmaxLast (t := t0) (s := sh) x.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Cuda.Tape.softmaxLast (t := t0) (s := sh) x.id)
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s .softmax #[x.identity?] cpu cuda

/--
Record stable log-softmax (shape-preserving, last-axis convention).

PyTorch comparison: `torch.nn.functional.log_softmax(x, dim=-1)`.
-/
def logSoftmaxLast {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Context α]
  {sh : Shape} (x : TensorRef α sh) : IO (TensorRef α sh) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.logSoftmaxLast (t := t0) (s := sh) x.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Cuda.Tape.logSoftmaxLast (t := t0) (s := sh) x.id)
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s .logSoftmax #[x.identity?] cpu cuda

/-- Record elementwise softplus. PyTorch: `torch.nn.functional.softplus`. -/
def softplus {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Context α]
  {sh : Shape} (x : TensorRef α sh) : IO (TensorRef α sh) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.softplus (t := t0) (s := sh) x.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Cuda.Tape.softplus (t := t0) (s := sh) x.id)
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s .softplus #[x.identity?] cpu cuda

/-- Record elementwise exponential. PyTorch: `torch.exp`. -/
def exp {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Context α]
  {sh : Shape} (x : TensorRef α sh) : IO (TensorRef α sh) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.exp (t := t0) (s := sh) x.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.exp (t := t0) (s := sh) x.id
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s .exp #[x.identity?] cpu cuda

/-- Record elementwise sine and dispatch its value and VJP through the selected backend. -/
def sin {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Context α]
    {sh : Shape} (x : TensorRef α sh) : IO (TensorRef α sh) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.sin (t := t0) (s := sh) x.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Cuda.Tape.sin (t := t0) (s := sh) x.id)
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s .sin #[x.identity?] cpu cuda

/-- Record elementwise cosine with the selected backend's `-sin(x) * dLdy` backward rule. -/
def cos {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Context α]
    {sh : Shape} (x : TensorRef α sh) : IO (TensorRef α sh) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.cos (t := t0) (s := sh) x.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Cuda.Tape.cos (t := t0) (s := sh) x.id)
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s .cos #[x.identity?] cpu cuda

/-- Record elementwise log. PyTorch: `torch.log`. -/
def log {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Context α]
  {sh : Shape} (x : TensorRef α sh) : IO (TensorRef α sh) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.log (t := t0) (s := sh) x.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.log (t := t0) (s := sh) x.id
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s .log #[x.identity?] cpu cuda

/-- Record elementwise inverse `1/x`. PyTorch: `torch.reciprocal`. -/
def inv {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Context α]
  {sh : Shape} (x : TensorRef α sh) : IO (TensorRef α sh) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.inv (t := t0) (s := sh) x.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.inv (t := t0) (s := sh) x.id
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s .inv #[x.identity?] cpu cuda

/--
Record `log(softplus(x) + ε)` and its derivative.

The CPU and CUDA nodes use the same softplus transformation and the same `ε`. Backward
multiplies the incoming gradient by `sigmoid(x) / (softplus(x) + ε)`. Choose positive `ε` to
keep the logarithm's argument positive even when softplus rounds to zero.
-/
def safeLog {α : Type} [TorchLean.Storage α] [TensorTransfer α] (s : EagerSession α)
  [Context α]
  {sh : Shape} (x : TensorRef α sh) (ε : α := Context.defaultEpsilon) : IO (TensorRef α sh) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.safeLog (t := t0) (s := sh) x.id (ε := ε))
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let epsF ← TensorTransfer.toFloat (α := α) ε
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.safeLog (t := t0) (s := sh) x.id epsF
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s .safeLog #[x.identity?] cpu cuda

end EagerSession

end Internal
end Torch
end Autograd
end Runtime
