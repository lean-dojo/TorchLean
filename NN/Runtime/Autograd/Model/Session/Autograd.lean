/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Model.Session.Types
public import NN.Runtime.Autograd.Torch.TypedGraphSession.Autograd

/-!
Session-level autograd operations.

This module exposes backward and gradient-readback helpers for session tensors while preserving the
host/CUDA synchronization invariants maintained by the runtime.
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace Model

open Spec
open Runtime.Autograd.Torch.Internal (EagerSession)
open TorchLean TorchLean.Tensor

namespace Session

/--
Run a backward pass and return a dense array of gradients for inputs and recorded nodes.

The seed has the selected output's shape. For a scalar loss, pass `Tensor.scalar 1`.
-/
def backwardDenseAll {α : Type} [TorchLean.Storage α] (s : Session α) [Add α] [Zero α]
  [Runtime.Autograd.Torch.TensorTransfer α]
  {sh : Shape} (out : Runtime.Autograd.Torch.TensorRef α sh) (seed : Tensor α sh) :
  IO (Array (Spec.SomeTensor α)) := do
  match s.state with
  | .eager sess =>
      sess.validateTensorRef out
      EagerSession.backwardDenseAll (α := α) sess (sh := sh) out seed
  | .typedGraph sess =>
      sess.validateTensorRef out
      Runtime.Autograd.Torch.Internal.TypedGraphSession.backwardDenseAll (α := α) sess (sh := sh)
        out seed

/--
Extract the gradient for a particular tensor ref from a dense gradient array.

This is the non-mutating counterpart of reading `x.grad`.
-/
def grad {α : Type} [TorchLean.Storage α] (s : Session α) {sh : Shape}
  (grads : Array (Spec.SomeTensor α)) (x : Runtime.Autograd.Torch.TensorRef α sh) :
  IO (Tensor α sh) := do
  match s.state with
  | .eager sess => sess.validateTensorRef x
  | .typedGraph sess => sess.validateTensorRef x
  let gAny ← match grads[x.id]? with
    | some g => pure g
    | none => throw <| IO.userError "torchlean: gradient array out of bounds"
  if h : gAny.shape = sh then
    pure (gAny.cast h)
  else
    throw <| IO.userError
      s!"torchlean: grad shape mismatch (expected {Shape.pretty sh}, got {Shape.pretty gAny.shape})"

/-- Vector-Jacobian product: `vjp(out, seed)[x]`. -/
def vjp {α : Type} [TorchLean.Storage α] (s : Session α) [Add α] [Zero α]
  [Runtime.Autograd.Torch.TensorTransfer α]
    {shOut shX : Shape}
    (out : Runtime.Autograd.Torch.TensorRef α shOut)
    (seed : Tensor α shOut)
    (x : Runtime.Autograd.Torch.TensorRef α shX) :
    IO (Tensor α shX) := do
  let grads ← backwardDenseAll (α := α) s (sh := shOut) out seed
  grad (α := α) s (sh := shX) grads x

/-! ## Forward-mode: JVP -/

/--
Jacobian-vector product for a single leaf (typed graph execution only).

For eager sessions, use the typed graph execution if you need JVPs.
-/
def jvpLeaf {α : Type} [TorchLean.Storage α] (s : Session α) [Zero α]
    {shOut shX : Shape}
    (out : Runtime.Autograd.Torch.TensorRef α shOut)
    (x : Runtime.Autograd.Torch.TensorRef α shX)
    (dx : Tensor α shX) :
    IO (Tensor α shOut) := do
  match s.state with
  | .eager _ =>
      throw <| IO.userError "torchlean: jvpLeaf is only supported for typed graph sessions"
  | .typedGraph sess =>
      Runtime.Autograd.Torch.Internal.TypedGraphSession.jvpLeaf (α := α) sess
        (shOut := shOut) (shX := shX) out x dx

/-! ## Forward-mode: dense JVP (typed graph execution only) -/

/--
Jacobian-vector product with explicit tangents for all *leaf* tensors.

`dxs[i]` is the tangent for leaf `i` (same indexing as `grad`/`backwardDenseAll`).
-/
def jvpDenseAll {α : Type} [TorchLean.Storage α] (s : Session α)
    {shOut : Shape}
    (out : Runtime.Autograd.Torch.TensorRef α shOut)
    (dxs : Array (Spec.SomeTensor α)) :
    IO (Tensor α shOut) := do
  match s.state with
  | .eager _ =>
      throw <| IO.userError "torchlean: jvpDenseAll is only supported for typed graph sessions"
  | .typedGraph sess =>
      Runtime.Autograd.Torch.Internal.TypedGraphSession.jvpDenseAll (α := α) sess (sh := shOut) out
        dxs

/--
Apply a dense SGD step to all learnable parameters.

This is an optimizer helper used by examples; for a higher-level API see
  `TorchLean.Trainer` and `TorchLean.Trainer.Session`.
-/
def sgdStepAll {α : Type} [TorchLean.Storage α] (s : Session α)
  [Sub α] [Mul α] [Add α] [Zero α]
  [Runtime.Autograd.Torch.TensorTransfer α]
  (lr : α) (grads : Array (Spec.SomeTensor α)) : IO Unit := do
  match s.state with
  | .eager sess => EagerSession.sgdStepAll (α := α) sess lr grads
  | .typedGraph sess =>
      Runtime.Autograd.Torch.Internal.TypedGraphSession.sgdStepAll (α := α) sess lr grads

end Session

end Model
end Autograd
end Runtime
