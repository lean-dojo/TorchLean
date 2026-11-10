/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Batch
public import TorchLean.NN.Ops
public import TorchLean.NN.Session.Apply

/-!
# Eager Operations

Run an `Ops` program on the tape of an eager session. The instance lives here so evaluation and
recording can use it without importing trainer construction.
-/

@[expose] public section

namespace TorchLean

open TorchLean.Tensor

/--
Monad used for the eager `Ops` instance: read an `Session α` and execute in `IO`.

An `Ops` program in this monad runs each operation as it is called, on the session's mutable tape.
-/
abbrev SessionM (α : Type) [Storage α] :=
  ReaderT (Session α) IO

/--
`Ops` instance for the eager Torch-style runtime. Each operator is recorded at once on the hidden
mutable tape of the `Session` it reads: `detach` by `Session.detach`, every other
operator by `Session.apply` under its selected capsule.

`updateBuffers?` writes the updated values back to the parameters that do not require gradients.
On a CUDA session, `applyNative?` runs `Op.rfft`, `Op.irfft` and `Op.selectiveScan` with
`Session.apply` and declines every other operator.
-/
instance instOpsSessionM {α : Type} [Storage α] [Context α] [TensorTransfer α] :
    Ops (SessionM α) α where
  Ref := fun s => TensorRef α s
  DataRef := fun β _ s => Tensor β s
  dataConst := fun x => x
  mapData := fun f x => f x
  observe := fun x session => do return (← session.getValue x).item
  updateBuffers? := some fun refs input update session => do
    let parameters ← session.paramsByLeaf.get
    let storages ← session.parameterStorageByLeaf.get
    let rec read : {ss : List Shape} → Pack (TensorRef α) ss → IO (TensorPack α ss)
      | [], .nil => pure .nil
      | shape :: _, .cons ref rest => do
          session.validateTensorRef ref
          let value ←
            match parameters[ref.id]?, storages[ref.id]? with
            | some parameter, some storage => do
                if ← storage.hostSynced.get then
                  let value ← parameter.get
                  if h : value.shape = shape then pure (value.cast h)
                  else throw <| IO.userError "buffer update: state shape mismatch"
                else session.getValue ref
            | _, _ => session.getValue ref
          pure (.cons value (← read rest))
    let values ← update (← read refs) (← session.getValue input)
    let rec write : {ss : List Shape} → Pack (TensorRef α) ss → TensorPack α ss → IO Unit
      | [], .nil, .nil => pure ()
      | _ :: _, .cons ref rest, .cons value values => do
          if let some parameter := parameters[ref.id]? then
            unless parameter.requiresGrad do
              parameter.set (SomeTensor.ofTensor value)
          write rest values
    write refs values
  apply := fun {ins rest s} op [supported : Op.Supported α op] args sess =>
    match ins, rest, op, args with
    | _, _, .detach, (x, ()) =>
      letI := supported.caps
      Session.detach (α := α) sess (sh := s) x
        (name := some (Op.encode ⟨_, _, Op.detach (s := s)⟩).tag)
    | ins, _, op, args => do
        let args ← Session.Args.ofPorts sess ins args
        letI := supported.caps
        sess.apply (Op.encode ⟨ins, _, op⟩).tag op args
  applyRandom := fun {ins _ _} op [supported : Op.Supported α op] seed args sess => do
    let some withKey := Op.withKey? op |
      throw <| IO.userError s!"torch: {(Op.encode ⟨ins, _, op⟩).tag} takes no random key"
    let args ← Session.Args.ofPorts sess ins args
    letI := supported.caps
    sess.applyRandom (Op.encode ⟨ins, _, op⟩).tag withKey seed args
  applyNative? := some fun {ins _ _} op [supported : Op.Supported α op] args sess => do
    let native : Bool := match op with
      | .rfft .. | .irfft .. | .selectiveScan .. => true
      | _ => false
    if !sess.options.usesCUDA || !native then return none
    let args ← Session.Args.ofPorts sess ins args
    letI := supported.caps
    some <$> sess.apply (Op.encode ⟨ins, _, op⟩).tag op args
  atomic := fun act sess => Session.atomic sess (act sess)
  const := fun {s} t => fun sess => Session.const (α := α) sess (sh := s) t

end TorchLean
