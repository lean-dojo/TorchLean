/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core.Session

/-!
# Eager Tensor Operations

PyTorch-style tensor operations backed by the eager CPU/CUDA tapes. These wrappers record runtime
nodes, dispatch CUDA kernels when requested, and preserve the typed `TensorRef` surface.
-/


@[expose] public section

namespace Runtime
namespace Autograd
namespace Torch

open Spec
open Tensor
open Proofs.Autograd.Algebra

namespace Internal

namespace EagerSession

/-!
## Tensor ops (eager tape wrappers)

The following definitions are the eager front-end for `Runtime.Autograd.Tape.*` primitives. Each one:
- reads the current tape from `s.tape`,
- appends a new node/leaf via a `Tape.*` constructor,
- writes the updated tape back, and
- returns a fresh `TensorRef` pointing to the new node id.

PyTorch comparison: this is the standard eager autograd mechanism (a dynamic tape of ops).
-/

/--
Dispatch an eager op with optional CUDA support.

When `Options.device = .cuda`, any op whose CUDA implementation returns `none` will throw.

TorchLean's CUDA eager mode has no per-op CPU fallback: either the op is supported by CUDA, or it
errors immediately.
-/
def dispatchCudaOpt {α β : Type} (s : EagerSession α) (opName : String)
    (cpu : IO β) (cuda : IO (Option β)) : IO β := do
  if Options.device s.opts == .cuda then
    let r? ← cuda
    match r? with
    | some r => pure r
    | none =>
        throw <| IO.userError s!"torch: cuda: `{opName}` is unsupported by the eager CUDA backend"
  else
    cpu

/-- Record elementwise addition `a + b`. PyTorch: `torch.add`. -/
def add {α : Type} (s : EagerSession α) [Add α] [DecidableEq Shape] {sh : Shape}
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
  dispatchCudaOpt (α := α) s "add" cpu cuda

/-- Record elementwise subtraction `a - b`. PyTorch: `torch.sub`. -/
def sub {α : Type} (s : EagerSession α) [Sub α] [Zero α] [DecidableEq Shape] {sh : Shape}
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
  dispatchCudaOpt (α := α) s "sub" cpu cuda

/-- Record elementwise multiplication `a * b`. PyTorch: `torch.mul`. -/
def mul {α : Type} (s : EagerSession α) [Mul α] [DecidableEq Shape] {sh : Shape}
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
  dispatchCudaOpt (α := α) s "mul" cpu cuda

/-- Record scaling by a scalar constant. PyTorch: `x * c`. -/
def scale {α : Type} [CudaBridge.TensorConv α] (s : EagerSession α) [Mul α] [DecidableEq Shape]
  {sh : Shape}
  (x : TensorRef α sh) (c : α) : IO (TensorRef α sh) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.scale (t := t0) (s := sh) x.id c)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let cF ← CudaBridge.TensorConv.toFloat (α := α) c
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.scale (t := t0) (s := sh) x.id cF
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "scale" cpu cuda

/-- Record elementwise absolute value. PyTorch: `torch.abs`. -/
def abs {α : Type} (s : EagerSession α) [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  [DecidableEq Shape]
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
  dispatchCudaOpt (α := α) s "abs" cpu cuda

/-- Record elementwise square root. PyTorch: `torch.sqrt`. -/
def sqrt {α : Type} (s : EagerSession α) [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  [DecidableEq Shape]
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
  dispatchCudaOpt (α := α) s "sqrt" cpu cuda

/-- Record elementwise clamp to `[minVal,maxVal]`. PyTorch: `torch.clamp`. -/
def clamp {α : Type} [CudaBridge.TensorConv α] (s : EagerSession α) [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {sh : Shape} (x : TensorRef α sh) (minVal maxVal : α) : IO (TensorRef α sh) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.clamp (t := t0) (s := sh) x.id minVal maxVal)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let lo ← CudaBridge.TensorConv.toFloat (α := α) minVal
    let hi ← CudaBridge.TensorConv.toFloat (α := α) maxVal
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.clamp (t := t0) (s := sh) x.id lo hi
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "clamp" cpu cuda

/-- Record elementwise maximum. PyTorch: `torch.maximum`. -/
def max {α : Type} (s : EagerSession α) [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  [DecidableEq Shape]
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
  dispatchCudaOpt (α := α) s "max" cpu cuda

/-- Record elementwise minimum. PyTorch: `torch.minimum`. -/
def min {α : Type} (s : EagerSession α) [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  [DecidableEq Shape]
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
  dispatchCudaOpt (α := α) s "min" cpu cuda

/-- Record elementwise ReLU. PyTorch: `torch.relu` / `torch.nn.functional.relu`. -/
def relu {α : Type} (s : EagerSession α)
  [Mul α] [Zero α] [Max α] [One α] [LT α]
  [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
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
  dispatchCudaOpt (α := α) s "relu" cpu cuda

/-- Record elementwise sigmoid. PyTorch: `torch.sigmoid`. -/
def sigmoid {α : Type} (s : EagerSession α) [Context α] [DecidableEq Shape]
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
  dispatchCudaOpt (α := α) s "sigmoid" cpu cuda

/-- Record elementwise tanh. PyTorch: `torch.tanh`. -/
def tanh {α : Type} (s : EagerSession α) [Context α] [DecidableEq Shape]
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
  dispatchCudaOpt (α := α) s "tanh" cpu cuda

/--
Record softmax (shape-preserving).

PyTorch comparison: `torch.softmax(x, dim=...)` (dimension convention is chosen by the underlying
  tape op).
-/
def softmax {α : Type} (s : EagerSession α) [Context α] [DecidableEq Shape]
  {sh : Shape} (x : TensorRef α sh) : IO (TensorRef α sh) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.softmax (t := t0) (s := sh) x.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Cuda.Tape.softmax (t := t0) (s := sh) x.id)
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "softmax" cpu cuda

/--
Record stable log-softmax (shape-preserving, last-axis convention).

PyTorch comparison: `torch.nn.functional.log_softmax(x, dim=-1)`.
-/
def logSoftmax {α : Type} (s : EagerSession α) [Context α] [DecidableEq Shape]
  {sh : Shape} (x : TensorRef α sh) : IO (TensorRef α sh) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.logSoftmax (t := t0) (s := sh) x.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Cuda.Tape.logSoftmax (t := t0) (s := sh) x.id)
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "log_softmax" cpu cuda

/-- Record elementwise softplus. PyTorch: `torch.nn.functional.softplus`. -/
def softplus {α : Type} (s : EagerSession α) [Context α] [DecidableEq Shape]
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
  dispatchCudaOpt (α := α) s "softplus" cpu cuda

/-- Record elementwise exponential. PyTorch: `torch.exp`. -/
def exp {α : Type} (s : EagerSession α) [Context α] [DecidableEq Shape]
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
  dispatchCudaOpt (α := α) s "exp" cpu cuda

/-- Record elementwise log. PyTorch: `torch.log`. -/
def log {α : Type} (s : EagerSession α) [Context α] [DecidableEq Shape]
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
  dispatchCudaOpt (α := α) s "log" cpu cuda

/-- Record elementwise inverse `1/x`. PyTorch: `torch.reciprocal`. -/
def inv {α : Type} (s : EagerSession α) [Context α] [DecidableEq Shape]
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
  dispatchCudaOpt (α := α) s "inv" cpu cuda

/--
Record elementwise log with epsilon guard.

PyTorch comparison: `torch.log(torch.clamp(x, min=ε))`.
-/
def safeLog {α : Type} [CudaBridge.TensorConv α] (s : EagerSession α) [Context α] [DecidableEq Shape]
  {sh : Shape} (x : TensorRef α sh) (ε : α := Numbers.epsilon) : IO (TensorRef α sh) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.safeLog (t := t0) (s := sh) x.id (ε := ε))
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let epsF ← CudaBridge.TensorConv.toFloat (α := α) ε
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.safeLog (t := t0) (s := sh) x.id epsF
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "safe_log" cpu cuda

/-- Sum-reduce all elements to a scalar. PyTorch: `x.sum()`. -/
def sum {α : Type} (s : EagerSession α) [Add α] [Zero α] [DecidableEq Shape]
  {sh : Shape} (x : TensorRef α sh) : IO (TensorRef α Shape.scalar) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.sum (t := t0) (s := sh) x.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.sum (t := t0) (s := sh) x.id
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "sum" cpu cuda

/-- Flatten a tensor to a 1D vector. PyTorch: `torch.flatten`. -/
def flatten {α : Type} (s : EagerSession α) [Inhabited α] [DecidableEq Shape] {sh : Shape}
  (x : TensorRef α sh) : IO (TensorRef α (.dim (Shape.size sh) .scalar)) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.flatten (t := t0) (s := sh) x.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.flatten (t := t0) (s := sh) x.id
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "flatten" cpu cuda

/--
Reshape a tensor while preserving total number of elements.

PyTorch comparison: `torch.reshape` / `view` (when valid).
-/
def reshape {α : Type} (s : EagerSession α) [Inhabited α] [DecidableEq Shape] {sh1 sh2 : Shape}
  (x : TensorRef α sh1) (h : Shape.size sh1 = Shape.size sh2) : IO (TensorRef α sh2) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.reshape (t := t0) (s₁ := sh1) (s₂ := sh2) x.id h)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.reshape (t := t0) (s₁ := sh1) (s₂ := sh2) x.id h
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "reshape" cpu cuda

/-- Transpose a 2D matrix. PyTorch: `x.t()` / `x.transpose(0,1)`. -/
def transpose2d {α : Type} (s : EagerSession α) [DecidableEq Shape] {m n : Nat}
  (x : TensorRef α (.dim m (.dim n .scalar))) : IO (TensorRef α (.dim n (.dim m .scalar))) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.transpose2d (t := t0) (m := m) (n := n) x.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.transpose2d (t := t0) (m := m) (n := n) x.id
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "transpose2d" cpu cuda

/-- Swap two adjacent axes at a given depth. PyTorch analogue: `x.transpose(dim, dim+1)`. -/
def swapAdjacentAtDepth {α : Type} (s : EagerSession α) [DecidableEq Shape] {sh : Shape}
  (depth : Nat) (x : TensorRef α sh) : IO (TensorRef α (sh.swapAdjacentAtDepth depth)) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.swapAdjacentAtDepth (t := t0) (s := sh) depth
      x.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.swapAdjacentAtDepth (t := t0) (s := sh) depth x.id
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "swapAdjacentAtDepth" cpu cuda

/-- Permute a 3D tensor `(a,b,c) → (b,c,a)`. PyTorch: `x.permute(1,2,0)`. -/
def transpose3dFirstToLast {α : Type} (s : EagerSession α) [DecidableEq Shape] {a b c : Nat}
  (x : TensorRef α (.dim a (.dim b (.dim c .scalar)))) :
  IO (TensorRef α (.dim b (.dim c (.dim a .scalar)))) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.transpose3dFirstToLast (t := t0) (a := a) (b :=
      b) (c := c) x.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.transpose3dFirstToLast (t := t0) (a := a) (b := b) (c := c) x.id
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "transpose3d_first_to_last" cpu cuda

/-- Permute a 3D tensor `(a,b,c) → (c,a,b)`. PyTorch: `x.permute(2,0,1)`. -/
def transpose3dLastToFirst {α : Type} (s : EagerSession α) [DecidableEq Shape] {a b c : Nat}
  (x : TensorRef α (.dim a (.dim b (.dim c .scalar)))) :
  IO (TensorRef α (.dim c (.dim a (.dim b .scalar)))) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.transpose3dLastToFirst (t := t0) (a := a) (b :=
      b) (c := c) x.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.transpose3dLastToFirst (t := t0) (a := a) (b := b) (c := c) x.id
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "transpose3d_last_to_first" cpu cuda

/-- Swap the last two axes of a 3D tensor `(a,b,c) → (a,c,b)`. PyTorch: `x.transpose(1,2)`. -/
def transpose3dLastTwo {α : Type} (s : EagerSession α) [DecidableEq Shape] {a b c : Nat}
  (x : TensorRef α (.dim a (.dim b (.dim c .scalar)))) :
  IO (TensorRef α (.dim a (.dim c (.dim b .scalar)))) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.transpose3dLastTwo (t := t0) (a := a) (b := b)
      (c := c) x.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.transpose3dLastTwo (t := t0) (a := a) (b := b) (c := c) x.id
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "transpose3d_last_two" cpu cuda

/-- Broadcast a tensor to a larger shape. PyTorch: implicit broadcasting / `expand`. -/
def broadcastTo {α : Type} (s : EagerSession α) [Inhabited α] [Add α] [Zero α] [DecidableEq Shape]
  {sh1 sh2 : Shape} (cb : Shape.CanBroadcastTo sh1 sh2) (x : TensorRef α sh1) : IO (TensorRef α sh2)
    := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.broadcastTo (α := α) (t := t0) (s₁ := sh1) (s₂ :=
      sh2) cb x.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.broadcastTo (t := t0) (s₁ := sh1) (s₂ := sh2) cb x.id
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "broadcastTo" cpu cuda

/-- Sum-reduce along `axis`. PyTorch: `torch.sum(x, dim=axis)`. -/
def reduceSum {α : Type} (s : EagerSession α) [Add α] [Zero α] [Inhabited α] [DecidableEq Shape]
  {sh : Shape} (axis : Nat) [valid : Shape.valid_axis_inst axis sh] [wf : Shape.WellFormed sh]
  (x : TensorRef α sh) : IO (TensorRef α (shapeAfterSum sh axis)) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.reduceSum (t := t0) (s := sh) axis x.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.reduceSum (s := sh) axis (t := t0) x.id
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "reduce_sum" cpu cuda

/-- Mean-reduce along `axis`. PyTorch: `torch.mean(x, dim=axis)`. -/
def reduceMean {α : Type} (s : EagerSession α) [Context α] [DecidableEq Shape]
  {sh : Shape} (axis : Nat) [valid : Shape.valid_axis_inst axis sh] [wf : Shape.WellFormed sh]
  (x : TensorRef α sh) : IO (TensorRef α (shapeAfterSum sh axis)) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.reduceMean (t := t0) (s := sh) axis x.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.reduceMean (s := sh) axis (t := t0) x.id
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "reduce_mean" cpu cuda

/-- Gather a scalar from a 1D vector with a `Fin n` index. PyTorch: `x[i]`. -/
def gatherScalar {α : Type} (s : EagerSession α) [Zero α] [DecidableEq Shape]
  {n : Nat} (x : TensorRef α (.dim n .scalar)) (i : Fin n) : IO (TensorRef α Shape.scalar) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.gatherScalar (t := t0) (n := n) x.id i)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.gatherScalar (t := t0) (n := n) x.id i
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "gather_scalar" cpu cuda

/-- Gather a row from a 2D tensor with a `Fin rows` index. PyTorch: `x[i]` for 2D tensors. -/
def gatherRow {α : Type} (s : EagerSession α) [Zero α] [DecidableEq Shape]
  {rows cols : Nat} (x : TensorRef α (.dim rows (.dim cols .scalar))) (i : Fin rows) :
  IO (TensorRef α (.dim cols .scalar)) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.gatherRow (t := t0) (rows := rows) (cols := cols)
      x.id i)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.gatherRow (t := t0) (rows := rows) (cols := cols) x.id i
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "gather_row" cpu cuda

/-- Gather a scalar from a 1D vector with a raw `Nat` index (totalized by the tape op). -/
def gatherScalarNat {α : Type} (s : EagerSession α) [Zero α] [DecidableEq Shape]
  {n : Nat} (x : TensorRef α (.dim n .scalar)) (i : Nat) : IO (TensorRef α Shape.scalar) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.gatherScalarNat (t := t0) (n := n) x.id i)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.gatherScalarNat (t := t0) (n := n) x.id i
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "gather_scalar_nat" cpu cuda

/-- Dynamic gather scalar using an index stored in `NatRef`. -/
def gatherScalarRef {α : Type} (s : EagerSession α) [Zero α] [DecidableEq Shape]
  {n : Nat} (x : TensorRef α (.dim n .scalar)) (i : NatRef) : IO (TensorRef α Shape.scalar) := do
  let idx ← getNat (α := α) s i
  gatherScalarNat (α := α) s (n := n) x idx

/-- Dynamic gather row using an index stored in `NatRef` (out-of-range gives a zero row). -/
def gatherRowRef {α : Type} [CudaBridge.TensorConv α] (s : EagerSession α) [Zero α] [DecidableEq Shape]
  {rows cols : Nat} (x : TensorRef α (.dim rows (.dim cols .scalar))) (i : NatRef) :
  IO (TensorRef α (.dim cols .scalar)) := do
  let idx ← getNat (α := α) s i
  if h : idx < rows then
    gatherRow (α := α) s (rows := rows) (cols := cols) x ⟨idx, h⟩
  else
    -- total: out-of-bounds labels map to a zero row
    const (α := α) s (sh := .dim cols .scalar) (fill (0 : α) (.dim cols .scalar)) (name := none)

/-- Gather `k` scalars using an explicit index tensor. PyTorch analogue: `gather` / advanced
  indexing. -/
def gatherVecNat {α : Type} (s : EagerSession α) [Add α] [Zero α] [DecidableEq Shape]
  {n k : Nat} (x : TensorRef α (.dim n .scalar)) (idx : Tensor Nat (.dim k .scalar)) :
  IO (TensorRef α (.dim k .scalar)) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.gatherVecNat (t := t0) (n := n) (k := k) x.id
      idx)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.gatherVecNat (t := t0) (n := n) (k := k) x.id idx
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "gather_vec_nat" cpu cuda

/-- Gather `k` rows using an explicit index tensor. PyTorch: `index_select(dim=0, index=...)`. -/
def gatherRowsNat {α : Type} (s : EagerSession α) [Add α] [Zero α] [DecidableEq Shape]
  {rows cols k : Nat} (x : TensorRef α (.dim rows (.dim cols .scalar))) (idx : Tensor Nat (.dim k
    .scalar)) :
  IO (TensorRef α (.dim k (.dim cols .scalar))) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.gatherRowsNat (t := t0) (rows := rows) (cols :=
      cols) (k := k) x.id idx)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.gatherRowsNat (t := t0) (rows := rows) (cols := cols) (k := k) x.id idx
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "gather_rows_nat" cpu cuda

/--
Read a float input vector and return the corresponding `Tensor Nat` index vector.

Non-differentiable: used by token-id language-model losses that accept float-encoded ids as inputs.
-/
def floatVecToNatTensor {α : Type} (s : EagerSession α) [CudaBridge.TensorConv α] [DecidableEq Shape]
    {k : Nat} (x : TensorRef α (.dim k .scalar)) : IO (Tensor Nat (.dim k .scalar)) := do
  let v ← getValue (α := α) s (sh := .dim k .scalar) x
  match v with
  | .dim f =>
      let ns ← (List.finRange k).mapM (fun i => do
        match f i with
        | .scalar fl => do
            let ff ← CudaBridge.TensorConv.toFloat (α := α) fl
            pure (UInt64.toNat (Float.floor ff).toUInt64))
      pure <|
        Tensor.dim (fun i => Tensor.scalar (ns.getD i.val 0))

/-- Gather `k` scalars using indices stored in the nat-environment (`NatVecRef`). -/
def gatherVecRef {α : Type} (s : EagerSession α) [Add α] [Zero α] [DecidableEq Shape]
  {n k : Nat} (x : TensorRef α (.dim n .scalar)) (idx : NatVecRef k) :
  IO (TensorRef α (.dim k .scalar)) := do
  let it ← getNatVec (α := α) (k := k) s idx
  gatherVecNat (α := α) s (n := n) (k := k) x it

/-- Gather `k` rows using indices stored in the nat-environment (`NatVecRef`). -/
def gatherRowsRef {α : Type} (s : EagerSession α) [Add α] [Zero α] [DecidableEq Shape]
  {rows cols k : Nat} (x : TensorRef α (.dim rows (.dim cols .scalar))) (idx : NatVecRef k) :
  IO (TensorRef α (.dim k (.dim cols .scalar))) := do
  let it ← getNatVec (α := α) (k := k) s idx
  gatherRowsNat (α := α) s (rows := rows) (cols := cols) (k := k) x it

/-- Scatter-add into a vector: return a copy of `x` with `x[i] += v`. -/
def scatterAddVec {α : Type} (s : EagerSession α) [Add α] [Zero α] [DecidableEq Shape]
  {n : Nat} (x : TensorRef α (.dim n .scalar)) (v : TensorRef α Shape.scalar) (i : Fin n) :
  IO (TensorRef α (.dim n .scalar)) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.scatterAddVec (t := t0) (n := n) x.id v.id i)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.scatterAddVec (t := t0) (n := n) x.id v.id i
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "scatter_add_vec" cpu cuda

/-- Scatter-add into a matrix row: return a copy of `x` with `x[i,:] += v`. -/
def scatterAddRow {α : Type} (s : EagerSession α) [Add α] [Zero α] [DecidableEq Shape]
  {rows cols : Nat}
  (x : TensorRef α (.dim rows (.dim cols .scalar))) (v : TensorRef α (.dim cols .scalar)) (i : Fin
    rows) :
  IO (TensorRef α (.dim rows (.dim cols .scalar))) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.scatterAddRow (t := t0) (rows := rows) (cols :=
      cols) x.id v.id i)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.scatterAddRow (t := t0) (rows := rows) (cols := cols) x.id v.id i
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "scatter_add_row" cpu cuda

/--
Fully-connected linear layer `y = w x + b` (matvec).

If `opts.fastKernels` is enabled, uses a runtime-only fast kernel implementation.
PyTorch comparison: `torch.nn.functional.linear(x, weight=w, bias=b)`.
-/
def linear {α : Type} (s : EagerSession α) [Inhabited α] [Add α] [Mul α] [Zero α] [DecidableEq
  Shape] [Runtime.Autograd.FastKernels.FastMatmul α]
  {inDim outDim : Nat}
  (w : TensorRef α (.dim outDim (.dim inDim .scalar)))
  (b : TensorRef α (.dim outDim .scalar))
  (x : TensorRef α (.dim inDim .scalar)) : IO (TensorRef α (.dim outDim .scalar)) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ←
      if s.opts.fastKernels then
        okOrThrow (Runtime.Autograd.Tape.linearFast (useGpu := s.opts.useGpu) (t := t0)
          (gpuPrecision := s.opts.fastGpuMatmulPrecision)
          (inDim := inDim) (outDim := outDim) w.id b.id x.id)
      else
        okOrThrow (Runtime.Autograd.Tape.linear (t := t0) (inDim := inDim) (outDim := outDim) w.id
          b.id x.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.linear (t := t0) (outDim := outDim) (inDim := inDim) w.id b.id x.id
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "linear" cpu cuda

/--
Mean-squared-error loss returning a scalar.

If `opts.fastKernels` is enabled, uses a runtime-only fast kernel implementation.
PyTorch comparison: `torch.nn.functional.mse_loss(..., reduction=\"mean\")`.
-/
def mseLoss {α : Type} [CudaBridge.TensorConv α] (s : EagerSession α)
  [Inhabited α] [Add α] [Sub α] [Mul α] [Div α] [Zero α] [One α] [Coe Nat α] [DecidableEq Shape]
  {sh : Shape} (yhat target : TensorRef α sh) : IO (TensorRef α Shape.scalar) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ←
      if s.opts.fastKernels then
        okOrThrow (Runtime.Autograd.Tape.mseLossFast (t := t0) (s := sh) yhat.id target.id)
      else
        okOrThrow (Runtime.Autograd.Tape.mseLoss (t := t0) (s := sh) yhat.id target.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.mseLoss (t := t0) (s := sh) yhat.id target.id
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "mse_loss" cpu cuda

/-- Layer normalization over embedding dimension. PyTorch: `nn.LayerNorm` / `functional.layer_norm`.
  -/
def layerNorm {α : Type} (s : EagerSession α) [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {seqLen embedDim : Nat} (h_seq_pos : seqLen > 0) (h_embed_pos : embedDim > 0)
  (x : TensorRef α (.dim seqLen (.dim embedDim .scalar)))
  (gamma : TensorRef α (.dim embedDim .scalar))
  (beta : TensorRef α (.dim embedDim .scalar)) : IO (TensorRef α (.dim seqLen (.dim embedDim
    .scalar))) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.layerNorm (t := t0)
      (seqLen := seqLen) (embedDim := embedDim) (h_seq_pos := h_seq_pos) (h_embed_pos := h_embed_pos)
      x.id gamma.id beta.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Cuda.Tape.layerNorm (t := t0)
      (seqLen := seqLen) (embedDim := embedDim) (h_seq_pos := h_seq_pos) (h_embed_pos := h_embed_pos)
      x.id gamma.id beta.id)
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "layer_norm" cpu cuda

/-- BatchNorm for channel-first images `(C,H,W)` (no batch axis). PyTorch: `nn.BatchNorm2d`
  (conceptually). -/
def batchnormChannelFirst {α : Type} (s : EagerSession α) [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {channels height width : Nat} (h_c : channels > 0) (h_h : height > 0) (h_w : width > 0)
  (x : TensorRef α (.dim channels (.dim height (.dim width .scalar))))
  (gamma : TensorRef α (.dim channels .scalar))
  (beta : TensorRef α (.dim channels .scalar)) : IO (TensorRef α (.dim channels (.dim height (.dim
    width .scalar)))) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.batchnormChannelFirst (t := t0)
      (channels := channels) (height := height) (width := width) (h_c := h_c) (h_h := h_h) (h_w :=
        h_w)
      x.id gamma.id beta.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Cuda.Tape.batchnormChannelFirst (t := t0)
      (channels := channels) (height := height) (width := width) (h_c := h_c) (h_h := h_h)
      (h_w := h_w)
      x.id gamma.id beta.id)
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "batchnorm_channel_first" cpu cuda

/-- Multi-head self-attention (typed, proof-friendly). PyTorch: `nn.MultiheadAttention`
  (conceptually). -/
def multiHeadAttention {α : Type} (s : EagerSession α) [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {n numHeads dModel headDim : Nat} (h1 : n ≠ 0)
  (wq : TensorRef α (.dim dModel (.dim (numHeads * headDim) .scalar)))
  (wk : TensorRef α (.dim dModel (.dim (numHeads * headDim) .scalar)))
  (wv : TensorRef α (.dim dModel (.dim (numHeads * headDim) .scalar)))
  (wo : TensorRef α (.dim (numHeads * headDim) (.dim dModel .scalar)))
  (x : TensorRef α (.dim n (.dim dModel .scalar)))
  (mask : Option (Tensor Bool (.dim n (.dim n .scalar))) := none) :
  IO (TensorRef α (.dim n (.dim dModel .scalar))) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.multiHeadAttention (t := t0)
      (n := n) (numHeads := numHeads) (dModel := dModel) (headDim := headDim) (h1 := h1)
      wq.id wk.id wv.id wo.id x.id mask)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Cuda.Tape.multiHeadAttention (t := t0)
      (n := n) (numHeads := numHeads) (dModel := dModel) (headDim := headDim) (h1 := h1)
      wq.id wk.id wv.id wo.id x.id (mask := mask) (useFlash := s.opts.fastKernels))
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "multi_head_attention" cpu cuda

/--
N-D convolution for channels-first tensors `(inC, spatial...)` (no batch axis).

This is the generic counterpart to `conv2d`.
PyTorch comparison: `torch.nn.functional.conv{d}d` specialized to a single sample.
-/
def conv {α : Type} (s : EagerSession α) [Context α]
  [DecidableEq Shape]
  {d inC outC : Nat}
  {kernel stride padding : Vector Nat d}
  {inSpatial : Vector Nat d}
  {hInC : inC ≠ 0} {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
  (w : TensorRef α (Shape.ofList (outC :: inC :: kernel.toList)))
  (b : TensorRef α (.dim outC .scalar))
  (x : TensorRef α (Shape.ofList (inC :: inSpatial.toList))) :
  IO (TensorRef α
    (Shape.ofList (outC :: (Spec.convOutSpatial inSpatial kernel stride padding).toList))) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.conv (t := t0)
      (d := d) (inC := inC) (outC := outC)
      (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial)
      w.id b.id x.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Cuda.Tape.conv (t := t0)
      (d := d) (inC := inC) (outC := outC)
      (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial)
      w.id b.id x.id
      (hInC := hInC) (hKernel := hKernel))
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "conv" cpu cuda

/-- 2D convolution for channel-first images `(C,H,W)` (no batch axis). PyTorch:
  `torch.nn.functional.conv2d`. -/
def conv2d {α : Type} (s : EagerSession α) [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {inC outC kH kW stride padding inH inW : Nat}
  {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
  (kernel : TensorRef α (.dim outC (.dim inC (.dim kH (.dim kW .scalar)))))
  (bias : TensorRef α (.dim outC .scalar))
  (input : TensorRef α (.dim inC (.dim inH (.dim inW .scalar)))) :
  IO (TensorRef α (.dim outC (.dim ((inH + 2 * padding - kH) / stride + 1)
    (.dim ((inW + 2 * padding - kW) / stride + 1) .scalar)))) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.conv2d (t := t0)
      (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding)
      (inH := inH) (inW := inW) (h1 := h1) (h2 := h2) (h3 := h3)
      kernel.id bias.id input.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.conv2d (t := t0)
        (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding)
        (inH := inH) (inW := inW)
        (h1 := h1) (h2 := h2) (h3 := h3)
        kernel.id bias.id input.id
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "conv2d" cpu cuda

/--
N-D transpose convolution for channels-first tensors `(inC, spatial...)` (no batch axis).

This is the generic counterpart to `conv_transpose2d`.
PyTorch comparison: `torch.nn.functional.conv_transpose{d}d` specialized to a single sample.
-/
def convTranspose {α : Type} (s : EagerSession α) [Context α]
  [DecidableEq Shape]
  {d inC outC : Nat}
  {kernel stride padding : Vector Nat d}
  {inSpatial : Vector Nat d}
  {hInC : inC ≠ 0} {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
  (w : TensorRef α (Shape.ofList (inC :: outC :: kernel.toList)))
  (b : TensorRef α (.dim outC .scalar))
  (x : TensorRef α (Shape.ofList (inC :: inSpatial.toList))) :
  IO (TensorRef α
    (Shape.ofList (outC :: (Spec.convTransposeOutSpatial inSpatial kernel stride padding).toList)))
    := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.convTranspose (t := t0)
      (d := d) (inC := inC) (outC := outC)
      (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial)
      w.id b.id x.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Cuda.Tape.convTranspose (t := t0)
      (d := d) (inC := inC) (outC := outC)
      (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial)
      w.id b.id x.id
      (hInC := hInC) (hKernel := hKernel))
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "conv_transpose" cpu cuda

/-- 2D transpose convolution for channel-first images `(C,H,W)` (no batch axis). PyTorch:
  `torch.nn.functional.conv_transpose2d`. -/
def convTranspose2d {α : Type} (s : EagerSession α) [Context α]
  [DecidableEq Shape]
  {inC outC kH kW stride padding inH inW : Nat}
  {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
  (kernel : TensorRef α (.dim inC (.dim outC (.dim kH (.dim kW .scalar)))))
  (bias : TensorRef α (.dim outC .scalar))
  (input : TensorRef α (.dim inC (.dim inH (.dim inW .scalar)))) :
  IO (TensorRef α (.dim outC (.dim ((inH - 1) * stride - 2 * padding + kH)
    (.dim ((inW - 1) * stride - 2 * padding + kW) .scalar)))) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.convTranspose2d (t := t0)
      (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding)
      (inH := inH) (inW := inW) (h1 := h1) (h2 := h2) (h3 := h3)
      kernel.id bias.id input.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.convTranspose2d (t := t0)
        (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding)
        (inH := inH) (inW := inW)
        (h1 := h1) (h2 := h2) (h3 := h3)
        kernel.id bias.id input.id
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "conv_transpose2d" cpu cuda

/-- 2D matrix multiplication. PyTorch: `torch.matmul` for 2D tensors. -/
def matmul {α : Type} (s : EagerSession α) [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {m n p : Nat}
  (a : TensorRef α (.dim m (.dim n .scalar)))
  (b : TensorRef α (.dim n (.dim p .scalar))) :
  IO (TensorRef α (.dim m (.dim p .scalar))) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ←
      if s.opts.fastKernels then
        okOrThrow (Runtime.Autograd.Tape.matmulFast (useGpu := s.opts.useGpu)
          (gpuPrecision := s.opts.fastGpuMatmulPrecision)
          (t := t0) (m := m) (n := n) (p := p) a.id b.id)
      else
        okOrThrow (Runtime.Autograd.Tape.matmul (t := t0) (m := m) (n := n) (p := p) a.id b.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.matmul (t := t0) (m := m) (n := n) (p := p) a.id b.id
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "matmul" cpu cuda

/-- Batched matrix multiplication. PyTorch: `torch.bmm`. -/
def bmm {α : Type} (s : EagerSession α) [Add α] [Mul α] [Zero α] [DecidableEq Shape]
  {batch m n p : Nat}
  (a : TensorRef α (.dim batch (.dim m (.dim n .scalar))))
  (b : TensorRef α (.dim batch (.dim n (.dim p .scalar)))) :
  IO (TensorRef α (.dim batch (.dim m (.dim p .scalar)))) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.bmm (α := α) (t := t0) (batch := batch) (m := m)
      (n := n) (p := p) a.id b.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.bmm (t := t0) (batch := batch) (m := m) (n := n) (p := p) a.id b.id
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "bmm" cpu cuda

/-- Concatenate two vectors along dim 0. PyTorch: `torch.cat([a,b], dim=0)`. -/
def concatVectors {α : Type} (s : EagerSession α) [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {n m : Nat}
  (a : TensorRef α (.dim n .scalar))
  (b : TensorRef α (.dim m .scalar)) :
  IO (TensorRef α (.dim (n + m) .scalar)) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.concatVectors (t := t0) (n := n) (m := m) a.id
      b.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.concatVectors (t := t0) (n := n) (m := m) a.id b.id
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "concat_vectors" cpu cuda

/-- Concatenate along dim 0 for tensors with leading dimension. PyTorch: `torch.cat(..., dim=0)`. -/
def concatDim0 {α : Type} (s : EagerSession α) [DecidableEq Shape]
  {n m : Nat} {sh : Shape}
  (a : TensorRef α (.dim n sh))
  (b : TensorRef α (.dim m sh)) :
  IO (TensorRef α (.dim (n + m) sh)) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.concatDim0 (α := α) (t := t0) (n := n) (m := m)
      (s := sh) a.id b.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.concatDim0 (t := t0) (n := n) (m := m) (s := sh) a.id b.id
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "concat_dim0" cpu cuda

/-- Slice along dim 0: `x[start:start+len]`. PyTorch: standard slicing. -/
def sliceRange0 {α : Type} (s : EagerSession α) [Zero α] [DecidableEq Shape]
  {n : Nat} {sh : Shape}
  (x : TensorRef α (.dim n sh)) (start len : Nat) (h : len + start ≤ n) :
  IO (TensorRef α (.dim len sh)) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.sliceRange0 (α := α) (t := t0) (n := n) (s := sh)
      x.id start len h)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.sliceRange0 (t := t0) (n := n) (s := sh) x.id start len h
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "slice_range0" cpu cuda

/--
N-D max pooling for channels-first tensors `(C, spatial...)` (no batch axis).

PyTorch comparison: `torch.nn.functional.max_pool1d` / `max_pool2d` / `max_pool3d` depending on the
spatial rank `d`.
-/
def maxPool {α : Type} (s : EagerSession α) [Context α] [DecidableEq Shape]
  {d C : Nat} {inSpatial kernel stride padding : Vector Nat d}
  {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
  (x : TensorRef α (Shape.ofList (C :: inSpatial.toList))) :
  IO (TensorRef α
    (Shape.ofList (C :: (Spec.poolOutSpatialPad inSpatial kernel stride padding).toList))) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.maxPool (t := t0)
      (d := d) (C := C)
      (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
      (hKernel := hKernel) x.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Cuda.Tape.maxPool (t := t0)
      (d := d) (C := C)
      (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
      (hKernel := hKernel) x.id)
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "max_pool" cpu cuda

/--
N-D average pooling for channels-first tensors `(C, spatial...)` (no batch axis).

PyTorch comparison: `torch.nn.functional.avg_pool1d` / `avg_pool2d` / `avg_pool3d` depending on the
spatial rank `d`.
-/
def avgPool {α : Type} (s : EagerSession α) [Context α] [DecidableEq Shape]
  {d C : Nat} {inSpatial kernel stride padding : Vector Nat d}
  (hKernel : ∀ i : Fin d, kernel.get i ≠ 0)
  (x : TensorRef α (Shape.ofList (C :: inSpatial.toList))) :
  IO (TensorRef α
    (Shape.ofList (C :: (Spec.poolOutSpatialPad inSpatial kernel stride padding).toList))) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.avgPool (t := t0)
      (d := d) (C := C)
      (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
      hKernel x.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Cuda.Tape.avgPool (t := t0)
      (d := d) (C := C)
      (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
      hKernel x.id)
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "avg_pool" cpu cuda

/--
N-D smooth max pooling (log-sum-exp surrogate) for channels-first tensors `(C, spatial...)`.

This is a differentiable approximation to max pooling; PyTorch does not expose it as a single
primitive, but it can be emulated with `logsumexp` over local windows.
-/
def smoothMaxPool {α : Type} [CudaBridge.TensorConv α] (s : EagerSession α) [Context α]
  [DecidableEq Shape]
  {d C : Nat} {inSpatial kernel stride padding : Vector Nat d}
  {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
  (x : TensorRef α (Shape.ofList (C :: inSpatial.toList))) (beta : α) :
  IO (TensorRef α
    (Shape.ofList (C :: (Spec.poolOutSpatialPad inSpatial kernel stride padding).toList))) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.smoothMaxPool (t := t0)
      (d := d) (C := C)
      (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
      (hKernel := hKernel) x.id beta)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let betaF ← CudaBridge.TensorConv.toFloat (α := α) beta
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Cuda.Tape.smoothMaxPool (t := t0)
      (d := d) (C := C)
      (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
      (hKernel := hKernel) x.id betaF)
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "smooth_max_pool" cpu cuda

/-- 2D max-pooling (no batch axis). PyTorch: `torch.nn.functional.max_pool2d`. -/
def maxPool2d {α : Type} (s : EagerSession α) [Context α] [DecidableEq Shape]
  {kH kW inH inW inC stride : Nat} {h1 : kH ≠ 0} {h2 : kW ≠ 0}
  (x : TensorRef α (.dim inC (.dim inH (.dim inW .scalar)))) :
  IO (TensorRef α (.dim inC (.dim ((inH - kH) / stride + 1) (.dim ((inW - kW) / stride + 1)
    .scalar)))) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.maxPool2d (t := t0)
      (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride)
      (h1 := h1) (h2 := h2) x.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let inCU32 : UInt32 := UInt32.ofNat inC
    let inHU32 : UInt32 := UInt32.ofNat inH
    let inWU32 : UInt32 := UInt32.ofNat inW
    let kHU32 : UInt32 := UInt32.ofNat kH
    let kWU32 : UInt32 := UInt32.ofNat kW
    let strideU32 : UInt32 := UInt32.ofNat stride
    let paddingU32 : UInt32 := 0
    let outSh : Shape :=
      .dim inC (.dim ((inH - kH) / stride + 1) (.dim ((inW - kW) / stride + 1) .scalar))
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.unary (t := t0) "max_pool2d" x.id
        (.dim inC (.dim inH (.dim inW .scalar))) outSh
        (forward := fun xBuf =>
          Runtime.Autograd.Cuda.torchleanMaxPool2dFwdCuda xBuf inCU32 inHU32 inWU32 kHU32 kWU32 strideU32 paddingU32)
        (backward := fun xBuf dLdy =>
          Runtime.Autograd.Cuda.torchleanMaxPool2dBwdCuda xBuf dLdy inCU32 inHU32 inWU32 kHU32 kWU32 strideU32 paddingU32)
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "max_pool2d" cpu cuda

/-- 2D max-pooling with padding (no batch axis). PyTorch: `max_pool2d(..., padding=...)`. -/
def maxPool2dPad {α : Type} (s : EagerSession α) [Context α] [DecidableEq Shape]
  {kH kW inH inW inC stride padding : Nat} {h1 : kH ≠ 0} {h2 : kW ≠ 0}
  (x : TensorRef α (.dim inC (.dim inH (.dim inW .scalar)))) :
  IO (TensorRef α (.dim inC (.dim ((inH + 2 * padding - kH) / stride + 1)
    (.dim ((inW + 2 * padding - kW) / stride + 1) .scalar)))) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.maxPool2dPad (t := t0)
      (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride) (padding :=
        padding)
      (h1 := h1) (h2 := h2) x.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let inCU32 : UInt32 := UInt32.ofNat inC
    let inHU32 : UInt32 := UInt32.ofNat inH
    let inWU32 : UInt32 := UInt32.ofNat inW
    let kHU32 : UInt32 := UInt32.ofNat kH
    let kWU32 : UInt32 := UInt32.ofNat kW
    let strideU32 : UInt32 := UInt32.ofNat stride
    let paddingU32 : UInt32 := UInt32.ofNat padding
    let outSh : Shape :=
      .dim inC (.dim ((inH + 2 * padding - kH) / stride + 1)
        (.dim ((inW + 2 * padding - kW) / stride + 1) .scalar))
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.unary (t := t0) "max_pool2d_pad" x.id
        (.dim inC (.dim inH (.dim inW .scalar))) outSh
        (forward := fun xBuf =>
          Runtime.Autograd.Cuda.torchleanMaxPool2dFwdCuda xBuf inCU32 inHU32 inWU32 kHU32 kWU32 strideU32 paddingU32)
        (backward := fun xBuf dLdy =>
          Runtime.Autograd.Cuda.torchleanMaxPool2dBwdCuda xBuf dLdy inCU32 inHU32 inWU32 kHU32 kWU32 strideU32 paddingU32)
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "max_pool2d_pad" cpu cuda

/-- Alias for `max_pool2d_pad` (PyTorch-style shorthand). -/
abbrev maxPoolPad {α : Type} := maxPool2dPad (α := α)

/-- Smooth max-pooling (softmax pooling). Not a standard PyTorch primitive; see
  `Torch.LinkedSession.smooth_max_pool2d`. -/
def smoothMaxPool2d {α : Type} [CudaBridge.TensorConv α] (s : EagerSession α) [Context α]
  [DecidableEq Shape]
  {kH kW inH inW inC stride : Nat} {h1 : kH ≠ 0} {h2 : kW ≠ 0}
  (x : TensorRef α (.dim inC (.dim inH (.dim inW .scalar)))) (beta : α) :
  IO (TensorRef α (.dim inC (.dim ((inH - kH) / stride + 1) (.dim ((inW - kW) / stride + 1)
    .scalar)))) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.smoothMaxPool2d (t := t0)
      (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride)
      (h1 := h1) (h2 := h2) x.id beta)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let betaF ← CudaBridge.TensorConv.toFloat (α := α) beta
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.smoothMaxPool2d (t := t0)
        (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride)
        (h1 := h1) (h2 := h2) x.id betaF
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "smooth_max_pool2d" cpu cuda

/-- 2D average-pooling (no batch axis). PyTorch: `torch.nn.functional.avg_pool2d`. -/
def avgPool2d {α : Type} (s : EagerSession α) [Context α] [DecidableEq Shape]
  {kH kW inH inW inC stride : Nat} (h1 : kH ≠ 0) (h2 : kW ≠ 0)
  (x : TensorRef α (.dim inC (.dim inH (.dim inW .scalar)))) :
  IO (TensorRef α (.dim inC (.dim ((inH - kH) / stride + 1) (.dim ((inW - kW) / stride + 1)
    .scalar)))) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.avgPool2d (t := t0)
      (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride)
      (h1 := h1) (h2 := h2) x.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Cuda.Tape.avgPool2d (t := t0)
      (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride)
      h1 h2 x.id)
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "avg_pool2d" cpu cuda

/-- 2D average-pooling with padding (no batch axis). PyTorch: `avg_pool2d(..., padding=...)`. -/
def avgPool2dPad {α : Type} (s : EagerSession α) [Context α] [DecidableEq Shape]
  {kH kW inH inW inC stride padding : Nat} (h1 : kH ≠ 0) (h2 : kW ≠ 0)
  (x : TensorRef α (.dim inC (.dim inH (.dim inW .scalar)))) :
  IO (TensorRef α (.dim inC (.dim ((inH + 2 * padding - kH) / stride + 1)
    (.dim ((inW + 2 * padding - kW) / stride + 1) .scalar)))) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.avgPool2dPad (t := t0)
      (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride) (padding :=
        padding)
      (h1 := h1) (h2 := h2) x.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Cuda.Tape.avgPool2dPad (t := t0)
      (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride) (padding :=
        padding)
      h1 h2 x.id)
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s "avg_pool2d_pad" cpu cuda

/-- Alias for `avg_pool2d_pad` (PyTorch-style shorthand). -/
abbrev avgPoolPad {α : Type} := avgPool2dPad (α := α)

end EagerSession

end Internal
end Torch
end Autograd
end Runtime
