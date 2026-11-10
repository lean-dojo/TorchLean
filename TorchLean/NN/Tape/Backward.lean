/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/


module

public import TorchLean.NN.Tape.CPU

@[expose] public section

namespace TorchLean

open TorchLean.Tensor

namespace Tape

/-!
## CPU gradient maps

The reverse sweeps live with the tape in `TorchLean.NN.Tape`, and the
helpers here turn a CPU sweep's result into the sparse map that training code reads.
-/

/--
Convert the optional dense gradient array returned by `backwardDense` into a sparse `HashMap`.

Only entries that are present (`some (some g)`) are kept. The result records exactly the nodes
reached by reverse-mode propagation.
-/
def denseToHashMap {α : Type} [Storage α]
  (grads : Array (Option (SomeTensor α))) :
  Std.HashMap Nat (SomeTensor α) :=
  (List.range grads.size).foldl (fun acc id =>
    match grads[id]? with
    | some (some g) => acc.insert id g
    | _ => acc
  ) (Std.HashMap.emptyWithCapacity)

/--
Reverse-mode backpropagation returning a `HashMap` of only the nodes that received gradients.

The sparse public form of `backwardDense`: the dense sweep runs first, and `denseToHashMap` then
drops the nodes that received no gradient.
-/
def backward {α : Type} [Storage α] [Add α]
  (t : Tape (SomeTensor α)) (outId : Nat) (seed : SomeTensor α) :
  Result (Std.HashMap Nat (SomeTensor α)) := do
  let dense ← backwardDense (t := t) outId seed
  pure (denseToHashMap dense)

/--
Backpropagate from a scalar output with seed gradient `1`.

PyTorch analogy: `loss.backward()` when `loss` is a scalar.
-/
def backwardScalar {α : Type} [Storage α] [Add α] [One α]
  (t : Tape (SomeTensor α)) (outId : Nat) : Result (Std.HashMap Nat (SomeTensor α)) :=
  backward (t:=t) outId (SomeTensor.ofTensor (Tensor.scalar (1 : α)))

end Tape
end TorchLean
