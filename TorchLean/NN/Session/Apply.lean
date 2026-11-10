/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Tape.Apply
public import TorchLean.Tensor.Op.Args
public import TorchLean.NN.Session.Backend
public import TorchLean.NN.Session.Recording
public import TorchLean.NN.Session.Random

/-!
# Recording a registered operator in an eager session

`Session.apply` records one registered operator on the session's tape: on the CPU tape with
`Tape.apply` and the operator's rule in `Op.rule?`, on the CUDA tape with `LibTorch.Tape.apply` and
the operator's entry in `LibTorch.Kernel.ofOp`. The session's device chose the tape when the session
was created. The session's profile selects the backend capsule of the operator, and the capsule must
name the provider and device of that tape. `atomic` records several operators as one
operation. `detach` records `Op.detach`, a reference that receives no gradient, on either tape
without a capsule selection.

PyTorch comparison: this is the standard eager autograd mechanism (a dynamic tape of ops).
-/

@[expose] public section

namespace TorchLean

namespace Session

/-- An argument of a registered operator in a session: a tensor reference for a value or detached
port, the tensor itself for a frozen, index, Boolean or integer port. -/
abbrev Arg (α : Type) [Storage α] : Port → Type :=
  Port.Slot (TensorRef α) (fun s => Tensor α s) (TensorRef α)

/-- The arguments of a list of ports, as nested pairs ending in `Unit`. -/
abbrev Args (α : Type) [Storage α] : List Port → Type := List.TProd (Arg α)

namespace Args

variable {α : Type} [Storage α]

/-- The identities of the tensor references, in port order. -/
def identities : (ps : List Port) → Args α ps → Array (Option RefIdentity)
  | [], () => #[]
  | ⟨.value, _⟩ :: ps, (x, xs) => #[x.identity?] ++ identities ps xs
  | ⟨.frozen, _⟩ :: ps, (_, xs) => identities ps xs
  | ⟨.detached, _⟩ :: ps, (x, xs) => #[x.identity?] ++ identities ps xs
  | ⟨.index _, _⟩ :: ps, (_, xs) => identities ps xs
  | ⟨.bool, _⟩ :: ps, (_, xs) => identities ps xs
  | ⟨.integer _ _, _⟩ :: ps, (_, xs) => identities ps xs

/-- The parents on the CPU tape. -/
def cpu : (ps : List Port) → Args α ps → Tape.Args α ps
  | [], () => ()
  | ⟨.value, _⟩ :: ps, (x, xs) => (x.id, cpu ps xs)
  | ⟨.frozen, _⟩ :: ps, (x, xs) => (x, cpu ps xs)
  | ⟨.detached, _⟩ :: ps, (x, xs) => (x.id, cpu ps xs)
  | ⟨.index _, _⟩ :: ps, (x, xs) => (x, cpu ps xs)
  | ⟨.bool, _⟩ :: ps, (x, xs) => (x, cpu ps xs)
  | ⟨.integer _ _, _⟩ :: ps, (x, xs) => (x, cpu ps xs)

/-- A frozen tensor as host `Float`. A scalar goes through `TensorTransfer.toFloat`, the conversion
used for the scalar attributes of native operations, and every other shape through
`TensorTransfer.toFloatTensor`. -/
def frozenFloat [TensorTransfer α] : (s : Shape) → Tensor α s → IO (Tensor Float s)
  | .scalar, x => do pure (Tensor.scalar (← TensorTransfer.toFloat x.item))
  | .dim _ _, x => TensorTransfer.toFloatTensor x

/-- The parents on the CUDA tape, the frozen tensors converted in port order. -/
def cuda [TensorTransfer α] : (ps : List Port) → Args α ps → IO (LibTorch.Args ps)
  | [], () => pure ()
  | ⟨.value, _⟩ :: ps, (x, xs) => do pure (x.id, ← cuda ps xs)
  | ⟨.frozen, s⟩ :: ps, (x, xs) => do
      let x ← frozenFloat s x
      pure (x, ← cuda ps xs)
  | ⟨.detached, _⟩ :: ps, (x, xs) => do pure (x.id, ← cuda ps xs)
  | ⟨.index _, _⟩ :: ps, (x, xs) => do pure (x, ← cuda ps xs)
  | ⟨.bool, _⟩ :: ps, (x, xs) => do pure (x, ← cuda ps xs)
  | ⟨.integer _ _, _⟩ :: ps, (x, xs) => do pure (x, ← cuda ps xs)

/-- The session's arguments of a model's call: a frozen reference is read with `getValue`, in port
order, and every other argument is passed as it is. -/
def ofPorts [TensorTransfer α] (session : Session α) (ps : List Port)
    (args : PortArgs α (TensorRef α) (fun β _ s => Tensor β s) ps) : IO (Args α ps) :=
  PortArgs.mapFrozen (m := IO) (F := fun s => Tensor α s) ps args
    fun | .data x => pure x | .ref x => session.getValue x

end Args

/-- Record the registered operator `op`, with one value output, on the session's CPU tape with
`Tape.apply` under the node name `name`. A failure leaves the tape unchanged. -/
def recordApply {α : Type} [Storage α] [Context α] [Op.Capabilities α]
    (s : Session α) {sh : Shape} {ins rest : List Port} (name : String)
    (op : Op ins (.value sh :: rest)) (args : Tape.Args α ins) :
    IO Nat :=
  s.recordCPU fun t => keepTapeOnError t <|
    (Tape.apply t name op args).map fun (t, id, _) => (t, id)

/-- `recordCUDA` for a recording in `IO` that returns the tape it records on. The tape leaves the
session's reference while `record` runs, so a recording that keeps no second reference appends in
place. The returned tape is stored back and a failure in the result is rethrown. `record` must not
throw, because the session would be left holding the empty tape put in its place. -/
def recordCUDAIO {α : Type} [Storage α] (s : Session α)
    (record : Tape LibTorch.SomeBuffer →
      IO (Tape LibTorch.SomeBuffer × Result Nat)) : IO Nat := do
  let taken ← s.tape.modifyGet fun
    | .cuda t => (some t, .cuda Tape.empty)
    | other => (none, other)
  let some tape := taken |
    throw <| IO.userError "torch: this session records on the CPU, not on CUDA"
  let (tape, result) ← record tape
  s.setCUDATape tape
  IO.ofExcept result

/-- Whether the selected `capsule` for the operator named `tag` runs on the CUDA table (`true`, a
LibTorch CUDA capsule) or on the reference CPU rules (`false`). A capsule of any other provider or
device has no handler linked in this build, and the error lists the two that are. -/
def runsOnCUDA (tag : String) (capsule : Backend.KernelCapsule) : Except String Bool :=
  if capsule.provider == .reference && capsule.device == .cpu then pure false
  else if capsule.provider == .libTorch && capsule.device == .cuda then pure true
  else throw <|
    s!"torch: `{tag}` selected capsule `{capsule.name}` " ++
      s!"({reprStr capsule.provider}/{capsule.device.cliName}), but no matching executable " ++
      s!"handler is linked; available handlers: " ++
      s!"TorchLean reference CPU ({reprStr Backend.Provider.reference}/" ++
      s!"{Backend.Device.cpu.cliName}), " ++
      s!"LibTorch CUDA executor ({reprStr Backend.Provider.libTorch}/" ++
      s!"{Backend.Device.cuda.cliName})"

/-- The CUDA entry of `op` for its selected LibTorch `capsule`. It fails when the tape holds buffers
of a configured scalar format that the entry does not run, or when `op` has no entry or its entry
has no capsule. It also fails when the entry's own capsule does not implement the selected one
(`implements`). -/
def cudaEntry {α : Type} [Storage α] (s : Session α) {ins outs : List Port}
    (capsule : Backend.KernelCapsule) (op : Op ins outs) :
    IO (LibTorch.Kernel ins outs) := do
  let tag := (Op.encode ⟨ins, outs, op⟩).tag
  let tape ← s.cudaTape
  let configured :=
    tape.nodes[0]?.any fun node => (LibTorch.Buffer.format? node.value.buf).isSome
  let entry := LibTorch.Kernel.ofOp op
  if configured && !entry.any (·.configuredFormats) then
    throw <| IO.userError
      s!"autograd: configured GPU arithmetic does not support `{tag}`; use CPU"
  let some kernel := entry |
    throw <| IO.userError s!"torch: cuda: `{tag}` is unsupported by LibTorch"
  let some contract := kernel.contract? |
    throw <| IO.userError s!"torch: cuda: the entry of `{tag}` has no capsule to select"
  let own := contract.capsule .libTorch .cuda op
  unless own.implements capsule do
    throw <| IO.userError <|
      s!"torch: invalid executable backend binding: entry `{own.name}` " ++
        s!"implements `{own.op.tag}`, but capsule `{capsule.name}` implements `{capsule.op.tag}`"
  pure kernel

/-- The checks `apply` makes before it records `op` under its selected `capsule`: `runsOnCUDA` and,
on the CUDA table, `cudaEntry`. A program of several recordings runs them for its operator first,
so an unsupported operator fails before anything is recorded. -/
def checkImplementation {α : Type} [Storage α] (s : Session α) {ins outs : List Port}
    (capsule : Backend.KernelCapsule) (op : Op ins outs) : IO Unit := do
  if ← IO.ofExcept (runsOnCUDA (Op.encode ⟨ins, outs, op⟩).tag capsule) then
    discard <| s.cudaEntry capsule op

/-- Record `op` under its selected `capsule` after `checks`, which run once the implementation is
known and before anything is recorded: the CUDA entry after `cudaEntry`'s checks, or the operator's
rule on the CPU tape. We state the two programs in
`Session.recordSelected_of_libTorch_cuda` and
`Session.recordSelected_of_reference_cpu`. -/
def recordSelected {α : Type} [Storage α] [Context α] [Op.Capabilities α] [TensorTransfer α]
    (s : Session α) {ins : List Port} {sh : Shape} (name : String)
    {rest : List Port} (capsule : Backend.KernelCapsule) (op : Op ins (.value sh :: rest))
    (args : Args α ins)
    (checks : IO Unit) : IO (TensorRef α sh) := do
  let id ←
    if ← IO.ofExcept (runsOnCUDA (Op.encode ⟨ins, _, op⟩).tag capsule) then do
      let kernel ← s.cudaEntry capsule op
      checks
      let parents ← Args.cuda ins args
      s.recordCUDAIO fun tape => do
        let (tape, result) ← kernel.record tape name (TensorTransfer.dtype (α := α)) parents
        pure (tape, result.map fun (id, _) => id)
    else do
      checks
      s.recordApply name op (Args.cpu ins args)
  pure { id, identity? := some (← s.refIdentity) }

/-- Record the registered operator `op`, with one value output, under the node label `name`, and
return a reference to the new node stamped with the session's current owner and generation. The
references are validated first, then the session's profile selects the capsule of `op`. A reference
CPU capsule records `Tape.apply` on the CPU tape, with the operator's rule. A LibTorch CUDA capsule
records the operator's entry on the CUDA tape after `cudaEntry`'s checks. With no profile override,
`Session.apply_of_cpu` and
`Session.apply_of_cuda` state the program on the CPU and on
CUDA. A capsule of another provider or device is rejected (`runsOnCUDA`). A failure leaves the tape
unchanged. -/
def apply {α : Type} [Storage α] [Context α] [Op.Capabilities α] [TensorTransfer α]
    (s : Session α) {ins : List Port} {sh : Shape} (name : String)
    {rest : List Port} (op : Op ins (.value sh :: rest)) (args : Args α ins) :
    IO (TensorRef α sh) := do
  s.validateRefIdentities (Args.identities ins args)
  let capsule ← s.selectedCapsule op
  s.recordSelected name capsule op args (pure ())

/-- `apply` for the random operator `operator key`, where `key` is the session's next key for
`seed`. The selection and the checks before recording see that key, and the session's counter
advances only once they pass. A rejected reference, selection or implementation takes no key. A
failure after those checks, in argument conversion or in the recording, has already taken it. -/
def applyRandom {α : Type} [Storage α] [Context α] [Op.Capabilities α] [TensorTransfer α]
    (s : Session α) {ins : List Port} {sh : Shape} (name : String)
    {rest : List Port} (operator : UInt64 → Op ins (.value sh :: rest)) (seed : Nat)
    (args : Args α ins) :
    IO (TensorRef α sh) := do
  s.validateRefIdentities (Args.identities ins args)
  let invocation ← s.rngCounter.get
  let op := operator (Random.keyOf seed invocation)
  if s.options.usesCUDA then
    -- A CUDA session over a CPU tape is rejected before the selection.
    let _ ← s.cudaTape
  let capsule ← s.selectedCapsule op
  s.recordSelected name capsule op args s.takeRandomKey

/--
Keep the primal value and record a reference that receives no gradient.

The node is `Op.detach`, recorded without a capsule selection and labeled `name`, or `detach` when
`name` is `none`. On the CPU tape it runs the rule in `Op.rule?`, whose forward `Tensor.detach` also
clears the differentiation metadata of carriers that have it. On CUDA it runs the entry in
`LibTorch.Kernel.ofOp`, which borrows the stored buffer.
-/
def detach {α : Type} [Storage α] [Context α] [Op.Capabilities α] [TensorTransfer α]
    (s : Session α) {sh : Shape}
    (x : TensorRef α sh) (name : Option String := none) : IO (TensorRef α sh) := do
  s.validateTensorRef x
  if s.options.usesCUDA then
    let some stored := (← s.cudaTape).getValue? x.id |
      throw <| IO.userError "torch: detach: invalid tensor id (missing CUDA value)"
    unless stored.s = sh do
      throw <| IO.userError <|
        s!"torch: detach: shape mismatch (expected {sh}, got {stored.s})"
    let id ← s.recordCUDAIO fun tape => do
      let (tape, result) ← LibTorch.Tape.apply tape (name.getD "detach") (.detach (s := sh))
        (x.id, ()) (TensorTransfer.dtype (α := α))
      pure (tape, result.map fun (id, ()) => id)
    s.makeTensorRef id
  else
    let id ← s.recordApply (name.getD "detach") (.detach (s := sh)) (x.id, ())
    s.makeTensorRef id

/-- Run `program`, several recordings on the session's tape, as one operation. When it throws, the
tape it started from is restored and the exception rethrown, so the program keeps all of its nodes
or none. Only the append-only tape rolls back: selections and their reports, random keys,
parameters, reference generations and released buffers keep what `program` did to them, so
`program` should only record. -/
def atomic {α β : Type} [Storage α] (s : Session α) (program : IO β) : IO β := do
  let start ← s.tape.get
  try program
  catch error =>
    s.tape.set start
    throw error

end Session

end TorchLean
