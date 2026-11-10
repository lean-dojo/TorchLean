/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Ops
public import TorchLean.NN.TypedGraph.GraphM.Apply

/-!
# Graph Operations

Interpret `Ops` programs as typed graph nodes. The recorded graphs serve evaluation, the autograd
transforms and their proofs. Training does not use this instance, because each trainer step records
an eager tape. Its `apply` is `GraphM.apply` and its `applyRandom` is `GraphM.applyRandom`.
`atomic` is the identity, since a failed recording discards its builder state.
-/

@[expose] public section

namespace TorchLean

open TorchLean.Tensor

/--
`Ops` instance for the typed graph builder monad `GraphM`.

Each primitive records a typed SSA node, which runs only when the graph is evaluated.
`TypedGraph.GraphM` builds the graph data, and
`TorchLean.TypedGraph` packages it for repeated execution.
-/
instance {α Δ : Type} [Storage α] [Context α] {Γ : List Shape} :
    Ops (TypedGraph.GraphM.MWith α Δ Γ) α where
  Ref := fun s => TypedGraph.GraphM.Var s
  DataRef := fun β _ s => Δ → Tensor β s
  dataConst := fun x _ => x
  mapData := fun f x d => f (x d)
  observe := fun _ => throw "autograd: input-dependent scalar control requires an eager tape"
  updateBuffers? := some fun refs input update => do
    let readState (getValue : ∀ {s : Shape}, Nat → IO (Tensor α s)) :=
      let rec read : {ss : List Shape} →
          Pack TypedGraph.GraphM.Var ss → IO (TensorPack α ss)
        | [], .nil => pure .nil
        | _ :: _, .cons ref rest => do
            pure (.cons (← getValue ref.id) (← read rest))
      read refs
    let observer : TypedGraph.GraphM.BufferUpdate α := fun getState getValue => do
      let values ← update (← readState getState) (← getValue input.id)
      let rec writes : {ss : List Shape} →
          Pack TypedGraph.GraphM.Var ss → TensorPack α ss →
          Array (Nat × SomeTensor α) →
          Array (Nat × SomeTensor α)
        | [], .nil, .nil, acc => acc
        | _ :: _, .cons ref rest, .cons value values, acc =>
            writes rest values (acc.push (ref.id, SomeTensor.ofTensor value))
      pure (writes refs values #[])
    modify fun state => { state with bufferUpdates := state.bufferUpdates.push observer }
  apply := fun {_ _ _} op [supported : Op.Supported α op] args =>
    TypedGraph.GraphM.apply (α := α) (Δ := Δ) (Γ := Γ) (supported := supported)
      op args
  applyRandom := fun {_ _ _} op [supported : Op.Supported α op] seed args =>
    TypedGraph.GraphM.applyRandom (α := α) (Δ := Δ) (Γ := Γ)
      (supported := supported) op seed args
  atomic := id
  const := fun {s} t => TypedGraph.GraphM.const (α := α) (Γ := Γ) (s := s) t

end TorchLean
