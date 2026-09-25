/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core.Session.Backend
public import NN.Runtime.Autograd.Torch.Core.Session.Recording
public import NN.Spec.Core.Random

/-!
# Eager Random Tensors

A session counter gives repeated seeded calls distinct keys, including calls after a tape reset.
CPU and CUDA dispatch use the same key schedule.
-/

public section

namespace Runtime.Autograd.Torch.Internal

open Spec TorchLean

namespace EagerSession

/-- Advance once per random operation, including operations after a tape reset. -/
private def nextRandomKey {α : Type} [Storage α]
    (session : EagerSession α) (seed : Nat) : IO UInt64 := do
  let invocation ← session.rngCounter.get
  session.rngCounter.set (invocation + 1)
  pure <| Spec.Random.keyOf seed invocation

/-- Record a seeded uniform tensor using the session's next random key. -/
def randUniform {α : Type} [TorchLean.Storage α] [Context α] [TensorTransfer α]
    (s : EagerSession α) {sh : Shape}
    (seed : Nat) (name : Option String := none) : IO (TensorRef α sh) := do
  let cuda := do
    let key ← nextRandomKey s seed
    let tape ← s.cudaTape.get
    let elementCount ← Runtime.Autograd.okOrThrow <|
      Runtime.Autograd.Cuda.AnyBuffer.natToU32Checked (Spec.Shape.size sh)
    let buf := Runtime.Autograd.Cuda.Buffer.randUniform elementCount key
    let any : Runtime.Autograd.Cuda.AnyBuffer := { s := sh, buf := buf }
    let (nextTape, id) :=
      Runtime.Autograd.Cuda.Tape.leaf (t := tape) (value := any) (name := name)
        (requiresGrad := false)
    s.cudaTape.set nextTape
    s.makeTensorRef id
  let cpu := do
    let key ← nextRandomKey s seed
    let v : Tensor α sh := Spec.Random.uniform (α := α) key (s := sh)
    const (α := α) s (sh := sh) v (name := name)
  s.executeReferenceOrLibTorch .randUniform cpu cuda

/-- Deterministic `{0,1}` mask generator (seeded) with a scalar keep-probability input. -/
def bernoulliMask {α : Type} [TorchLean.Storage α] [Context α] [TensorTransfer α]
    (s : EagerSession α) {sh : Shape}
    (keepProb : TensorRef α Shape.scalar) (seed : Nat) (name : Option String := none) :
    IO (TensorRef α sh) := do
  let probabilityTensor ← getValue (α := α) s (sh := Shape.scalar) keepProb
  let probability : α := Tensor.item probabilityTensor
  let cuda := do
    let key ← nextRandomKey s seed
    let tape ← s.cudaTape.get
    let elementCount ← Runtime.Autograd.okOrThrow <|
      Runtime.Autograd.Cuda.AnyBuffer.natToU32Checked (Spec.Shape.size sh)
    let probabilityFloat ← TensorTransfer.toFloat (α := α) probability
    let buf := Runtime.Autograd.Cuda.Buffer.bernoulliMask elementCount probabilityFloat key
    let any : Runtime.Autograd.Cuda.AnyBuffer := { s := sh, buf := buf }
    let (nextTape, id) :=
      Runtime.Autograd.Cuda.Tape.leaf (t := tape) (value := any) (name := name)
        (requiresGrad := false)
    s.cudaTape.set nextTape
    s.makeTensorRef id
  let cpu := do
    let key ← nextRandomKey s seed
    let v : Tensor α sh := Spec.Random.mask (α := α) key probability (s := sh)
    const (α := α) s (sh := sh) v (name := name)
  s.executeReferenceOrLibTorch .bernoulliMask cpu cuda

end EagerSession

end Runtime.Autograd.Torch.Internal
