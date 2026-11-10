/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.NN.IR.Eval
public import TorchLean.NN.LibTorch.Kernel
import TorchLean.CUDA.Kernel.Launch
import TorchLean.NN.TensorTransfer

/-!
# Resident execution of IR graphs

`TorchLean.IR.Graph.runBuffers` runs a graph of registered operators (`TorchLean.IR.Graph`) on
resident binary32 buffers. Each operator node runs its operator's entry in `LibTorch.Kernel.ofOp`,
the implementation the CUDA tape records with, on a tape of its own whose leaves are the node's
value and detached inputs, and the tape is dropped afterwards. An entry takes value and detached
inputs as buffers and frozen, index, Boolean and integer inputs as host tensors (`IR.Resident`), so
index tensors and masks flow between nodes and every output of a node is kept. A custom operation
runs its checked NVRTC program, an input node reads its supplied value and a constant node places
its payload constant at its port.

Before any node is launched, the graph passes `Graph.check`, the runner requires a native CUDA
runtime, and every node is checked. Each input is found, accepted by its node's port and, in a
buffer, of the port's length and binary32. Each constant is read at its port. Each custom program
is found and native binary32, has one reference per input shape and has its CUDA source generated.
Each other operator has an entry, and each buffered output shape fits the `UInt32` ABI. An entry's
own recording checks (folded sizes, spatial ranks) and NVRTC compilation run when the node runs,
after earlier nodes. Every buffer a node returns has its port's length and is binary32, checked
before later nodes read it.

Softmax runs the tape's entry: the axis moved to the last place, the row maximum subtracted, the
exponentials summed and divided. Like the other pure native primitives, it aborts on a native
exception (the convolution's forward and the uploads of running statistics return their native
failures as errors). Matrix products run the tape's program of broadcasts, views and one batched
product.

The runner is forward only. It keeps no differentiation tape of the graph, runs no backward pass
and derives no VJP for a custom operation, and no theorem here covers native execution. Buffers
and their view aliases are reference counted, and the runner releases no borrowed input or view
alias by hand.
-/

public section

namespace TorchLean.IR

open LibTorch

/-- The binary32 tensor `tensor` on a resident buffer, as its little-endian binary32 words. -/
private def Resident.upload {s : Shape} (tensor : Tensor Float32 s) : IO Buffer := do
  Buffer.ofBytesIO (← IO.ofExcept (Kernel.instScalarFloat32.pack tensor)) .float32

end TorchLean.IR

/-! ## Graphs of registered operators -/

namespace TorchLean.IR

open LibTorch

/-- A value of the resident runner at port `p`, as the CUDA entries take it: a binary32 device
buffer at a value or detached port, the host tensor at a frozen, index, Boolean or integer port. -/
abbrev Resident.Carrier : Port → Type :=
  Port.Slot (fun _ => Buffer) (fun s => Tensor Float s) (fun _ => Buffer)

/-- A value of the resident runner with its port. -/
abbrev Resident := Σ p : Port, Resident.Carrier p

/-- Whether the values of port `p` live in device buffers. -/
private def Resident.buffered (p : Port) : Bool :=
  match p.kind with
  | .value | .detached => true
  | _ => false

/-- A value's buffer checked against its port: its length is the port shape's size and fits the
`UInt32` ABI, and its dtype is binary32. `what` names the value in the message, and host values
need no check. -/
private def Resident.checkBuffer (what : String) : Resident → Except String Unit
  | ⟨⟨.value, s⟩, buf⟩ | ⟨⟨.detached, s⟩, buf⟩ => do
      discard <| (SomeBuffer.validate { s, buf }).mapError fun message => s!"{what}: {message}"
      unless Buffer.dtype buf == .float32 do throw s!"{what} dtype must be binary32"
  | _ => pure ()

/-- Value `v` read at port `p`, as `Port.read` reads an operator's input: the same port, a value
buffer where a detached port of its shape is expected, or a value buffer read back to the host
where a frozen port of its shape is expected. -/
private def Resident.read : (p : Port) → Resident → IO (Resident.Carrier p)
  | ⟨.frozen, s⟩, ⟨⟨.value, t⟩, buf⟩ => do
      unless t = s do throw (IO.userError s!"IR native: a value of shape {t} where {s} is expected")
      let some host := (Tensor.from (← Buffer.toFloatArrayIO buf)).reshape? s |
        throw (IO.userError s!"IR native: a buffer that is not of shape {s}")
      pure host
  | ⟨.detached, s⟩, ⟨⟨.value, t⟩, buf⟩ =>
      if t = s then pure buf
      else throw (IO.userError s!"IR native: a value of shape {t} where {s} is expected")
  | p, ⟨q, x⟩ =>
      if h : q = p then pure (h ▸ x)
      else throw (IO.userError s!"IR native: a value at port {repr q} where {repr p} is expected")

/-- A payload constant at port `p` as a resident value: uploaded at a value or detached port,
converted to `Float` (exactly) at a frozen port, the tensor itself at an index, Boolean or
integer port. -/
private def Resident.place : (p : Port) → p.Carrier Float32 → IO (Resident.Carrier p)
  | ⟨.value, _⟩, x | ⟨.detached, _⟩, x => IR.Resident.upload x
  | ⟨.frozen, _⟩, x => TensorTransfer.toFloatTensor x
  | ⟨.index _, _⟩, x | ⟨.bool, _⟩, x | ⟨.integer _ _, _⟩, x => pure x

/-- The value at reference `r` among the outputs of earlier nodes. -/
private def Resident.readRef (values : Array (List Resident)) (r : Ref) : IO Resident :=
  match values[r.node]? >>= (·[r.out]?) with
  | some v => pure v
  | none => throw (IO.userError s!"IR native: no output {r.out} of node {r.node}")

/-- The entry arguments of the input ports `ps` from the values `vs`, one per port, on the node's
tape: a value or detached input is a leaf of its buffer, a frozen, index, Boolean or integer input
the host tensor. -/
private def Resident.bind : (ps : List Port) → List Resident → Tape SomeBuffer →
    IO (Tape SomeBuffer × Args ps)
  | [], [], tape => pure (tape, ())
  | ⟨.value, s⟩ :: ps, v :: vs, tape => do
      let buf ← read ⟨.value, s⟩ v
      let (tape, id) := tape.leaf ⟨s, buf⟩ (requiresGrad := false)
      let (tape, rest) ← bind ps vs tape
      pure (tape, (id, rest))
  | ⟨.detached, s⟩ :: ps, v :: vs, tape => do
      let buf ← read ⟨.detached, s⟩ v
      let (tape, id) := tape.leaf ⟨s, buf⟩ (requiresGrad := false)
      let (tape, rest) ← bind ps vs tape
      pure (tape, (id, rest))
  | ⟨.frozen, s⟩ :: ps, v :: vs, tape => do
      let host ← read ⟨.frozen, s⟩ v
      let (tape, rest) ← bind ps vs tape
      pure (tape, (host, rest))
  | ⟨.index bounds, s⟩ :: ps, v :: vs, tape => do
      let host ← read ⟨.index bounds, s⟩ v
      let (tape, rest) ← bind ps vs tape
      pure (tape, (host, rest))
  | ⟨.bool, s⟩ :: ps, v :: vs, tape => do
      let host ← read ⟨.bool, s⟩ v
      let (tape, rest) ← bind ps vs tape
      pure (tape, (host, rest))
  | ⟨.integer signed width, s⟩ :: ps, v :: vs, tape => do
      let host ← read ⟨.integer signed width, s⟩ v
      let (tape, rest) ← bind ps vs tape
      pure (tape, (host, rest))
  | _, _, _ => throw (IO.userError "IR native: one input per port is expected")

/-- The buffer of a value output `id` of the node's tape, whose recorded shape must be `s`. -/
private def Resident.output (tape : Tape SomeBuffer) (s : Shape) (id : Nat) :
    Except String Buffer := do
  let some value := tape.getValue? id | throw s!"no value at tape node {id}"
  unless value.s = s do throw s!"a value of shape {value.s} where {s} is declared"
  pure value.buf

/-- The outputs of an entry at the ports `ps` as resident values: the buffer of each value or
detached output's tape node, the host tensor of any other output. -/
private def Resident.collect (tape : Tape SomeBuffer) : (ps : List Port) → Args ps →
    Except String (List Resident)
  | [], () => pure []
  | ⟨.value, s⟩ :: ps, (id, ys) => do
      pure (⟨⟨.value, s⟩, ← output tape s id⟩ :: (← collect tape ps ys))
  | ⟨.detached, s⟩ :: ps, (id, ys) => do
      pure (⟨⟨.detached, s⟩, ← output tape s id⟩ :: (← collect tape ps ys))
  | ⟨.frozen, s⟩ :: ps, (y, ys) => do pure (⟨⟨.frozen, s⟩, y⟩ :: (← collect tape ps ys))
  | ⟨.index bounds, s⟩ :: ps, (y, ys) => do
      pure (⟨⟨.index bounds, s⟩, y⟩ :: (← collect tape ps ys))
  | ⟨.bool, s⟩ :: ps, (y, ys) => do pure (⟨⟨.bool, s⟩, y⟩ :: (← collect tape ps ys))
  | ⟨.integer signed width, s⟩ :: ps, (y, ys) => do
      pure (⟨⟨.integer signed width, s⟩, y⟩ :: (← collect tape ps ys))

/-- The action of the custom operation at node `id` reading `refs`: the payload's checked program,
compiled by NVRTC and run on one input buffer per input shape. -/
private def Resident.custom (payload : Payload Float32) (id : Nat) (refs : Array Ref)
    (name : String) (shapes : List Shape) (out : Shape) :
    Except String (Array (List Resident) → IO (List Resident)) := do
  let some program := payload.custom? id | throw s!"custom {name}: missing checked body"
  let .native .binary32 := program.scalar.precision |
    throw s!"custom {name}: resident buffers require native binary32"
  unless refs.size = shapes.length do
    throw s!"custom {name}: expected {shapes.length} inputs"
  let bits := fun x => (program.scalar.toBits x).toNat.toUInt64
  let _ ← (CUDA.emit .binary32 bits refs.size program.expression).mapError
    fun message => s!"custom {name}: {message}"
  pure fun values => do
    let inputs ← (refs.toList.zip shapes).mapM fun (r, s) => do
      read ⟨.value, s⟩ (← readRef values r)
    let buf ← Kernel.runBuffers .binary32 bits program.expression inputs.toArray
      out.size.toUInt64
    pure [⟨⟨.value, out⟩, buf⟩]

/-- The action of node `n` at id `id`, checked before any node runs: an input is supplied, a
constant placed from the payload, a custom operation runs its checked program (its CUDA source
generated here, compiled by NVRTC when it runs) and any other operator its entry in
`LibTorch.Kernel.ofOp` on a tape of its own, dropped afterwards. -/
private def Resident.prepare (payload : Payload Float32) (inputs : Array Resident) (id : Nat)
    (n : Node) :
    Except String (Array (List Resident) → IO (List Resident)) := do
  for p in n.outs do
    if buffered p then
      discard <| (SomeBuffer.numelU32 p.shape).mapError fun message =>
        s!"IR native: node {id}: {message}"
  match n with
  | .input index p =>
      let some v := inputs[index]? | throw s!"IR native: node {id}: missing input {index}"
      unless p.accepts v.1 do
        throw s!"IR native: node {id}: input {index} at port {repr v.1} where {repr p} is expected"
      (checkBuffer s!"input {index}" v).mapError fun message => s!"IR native: node {id}: {message}"
      pure fun _ => do pure [⟨p, ← read p v⟩]
  | .const p =>
      let some v := payload.const? id | throw s!"IR native: node {id}: missing constant"
      let x ← (p.read v).mapError fun message => s!"IR native: node {id}: {message}"
      pure fun _ => do pure [⟨p, ← place p x⟩]
  | .op ⟨_, _, .custom name shapes out⟩ refs =>
      (custom payload id refs name shapes out).mapError fun message =>
        s!"IR native: node {id}: {message}"
  | .op ⟨ins, outs, op⟩ refs =>
      let tag := (Op.encode ⟨ins, outs, op⟩).tag
      let some entry := LibTorch.Kernel.ofOp op |
        throw s!"IR native: node {id}: no CUDA entry for {tag}"
      pure fun values => do
        let (tape, args) ← bind ins (← refs.toList.mapM (readRef values)) .empty
        let (tape, result) ← entry.record tape tag (pure .float32) args
        let ys ← IO.ofExcept (result.mapError fun message => s!"IR native: node {id}: {message}")
        IO.ofExcept ((collect tape outs ys).mapError fun message =>
          s!"IR native: node {id}: {message}")

/-- Execute the graph on resident binary32 buffers, returning every output of every node. The
checks the module documentation lists run before the first node, and every buffer a node returns
is checked against its port before later nodes read it. -/
def Graph.runBuffers (g : Graph) (payload : Payload Float32) (inputs : Array Resident) :
    IO (Array (List Resident)) := do
  IO.ofExcept g.check
  Buffer.requireNativeRuntime
  let mut actions : Array (Array (List Resident) → IO (List Resident)) := #[]
  for (n, id) in g.nodes.zipIdx do
    actions := actions.push (← IO.ofExcept (Resident.prepare payload inputs id n))
  let mut values : Array (List Resident) := #[]
  for (action, id) in actions.zipIdx do
    let outputs ← action values
    for (v, k) in outputs.zipIdx do
      IO.ofExcept ((Resident.checkBuffer s!"output {k}" v).mapError fun message =>
        s!"IR native: node {id}: {message}")
    values := values.push outputs
  pure values

end TorchLean.IR
