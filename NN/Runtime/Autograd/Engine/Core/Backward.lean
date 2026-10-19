/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/


module

public import NN.Runtime.Autograd.Engine.Core.Base

@[expose] public section

namespace Runtime
namespace Autograd

open Spec TorchLean
open TorchLean TorchLean.Tensor

namespace Tape

/-!
## CPU gradient maps

The reverse sweeps themselves live with the tape in `NN.Runtime.Autograd.Engine.Tape`. These
helpers turn a CPU sweep's result into the sparse map that training code reads.
-/

/--
Convert the optional dense gradient array returned by `backwardDense` into a sparse `HashMap`.

Only entries that are present (`some (some g)`) are kept. The result records exactly the nodes
reached by reverse-mode propagation.
-/
def denseToHashMap {α : Type} [TorchLean.Storage α]
  (grads : Array (Option (Spec.SomeTensor α))) :
  Std.HashMap Nat (Spec.SomeTensor α) :=
  (List.range grads.size).foldl (fun acc id =>
    match grads[id]? with
    | some (some g) => acc.insert id g
    | _ => acc
  ) (Std.HashMap.emptyWithCapacity)

/--
Reverse-mode backpropagation returning a `HashMap` of only the nodes that received gradients.

This is the sparse public form of `backwardDense`: it computes dense gradients first, then drops
nodes that did not receive a gradient.
-/
def backward {α : Type} [TorchLean.Storage α] [Add α]
  (t : Tape (Spec.SomeTensor α)) (outId : Nat) (seed : Spec.SomeTensor α) :
  Result (Std.HashMap Nat (Spec.SomeTensor α)) := do
  let dense ← backwardDense (t := t) outId seed
  pure (denseToHashMap dense)

/--
Backpropagate from a scalar output with seed gradient `1`.

PyTorch analogy: `loss.backward()` when `loss` is a scalar.
-/
def backwardScalar {α : Type} [TorchLean.Storage α] [Add α] [One α]
  (t : Tape (Spec.SomeTensor α)) (outId : Nat) : Result (Std.HashMap Nat (Spec.SomeTensor α)) :=
  backward (t:=t) outId (Spec.SomeTensor.ofTensor (Tensor.scalar (1 : α)))

end Tape
end Autograd
end Runtime
