/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Json
public import TorchLean.NN.IR.Artifact

/-!
# Loading capture documents

The PyTorch and JAX captures write one JSON document per computation. Its `"artifact"` is the
typed artifact of format `torchlean.ir` (`TorchLean.IR.Artifact`: the program's steps, outputs and
scalars, and the interchange words of each constant). `"arguments"` and `"reference_outputs"`
hold tensors by element type, shape and complete words, and `"frontend"` names the capture that
wrote it. The words keep every bit of a value, so loading never goes through JSON numbers.

`load` decodes the artifact and loads it under the reference runtime with no registered custom
operations or calls (`TorchLean.IR.Artifact.load`), and `load_ok` proves that the program it returns
passes `Program.check .reference` and its payload passes `Payload.check`. Then `load` checks the
document's `"declared_types"`, where the PyTorch and JAX captures write one entry per value they
typed: the step output holding it and the element type and shape the capture recorded.
`declaredTypes` reads the entries, and `Program.checkDeclared` requires the program's type at each
listed output to have that element type and shape, so a lowering that changes a listed value's type
fails the load. Only the listed values are checked, so an empty list passes. A document that names
a `"frontend"` must have `"declared_types"`, and one that names none and lacks it declares no types.
`load_declared` proves that a loaded program gives every declared value its declared element type
and shape. `loadArguments` reads the arguments under a runtime, `loadValue` one tensor, and `words`
gives the element type, shape and words of a value back.
-/

@[expose] public section

namespace TorchLean.IR

open Lean
open FloatLib.Floats.Formats.BinaryInterchange

/-- One entry of a capture document's `"declared_types"`: the capture's internal node `node`, made
for the framework value `origin`, whose value the program holds at the step output `ref`, with the
element type and shape that the capture recorded for it. -/
structure Declared where
  /-- The capture's internal node. -/
  node : Nat
  /-- The step output that holds the node's value. -/
  ref : Ref
  /-- The framework operation or variable the node is made for. -/
  origin : String
  /-- The element type the capture recorded. -/
  dtype : DType
  /-- The shape the capture recorded. -/
  shape : Shape

/-- The entry of the JSON object with exactly the keys `node`, `ref` (`[step, output]`), `origin`,
`source`, `dtype` and `shape`. `source` says where the capture took the type from (both captures
write `capture`: the framework's own metadata), and no check reads it. -/
def Declared.ofJson (json : Json) : Except String Declared := do
  let ctx := "declared_types"
  let o ← Json.exactObject ctx ["node", "ref", "origin", "source", "dtype", "shape"] json
  pure { node := ← Json.natField ctx "node" o, ref := ← Ref.ofJson (← Json.field ctx "ref" o),
         origin := ← Json.string s!"{ctx}.origin" (← Json.field ctx "origin" o),
         dtype := ← DType.fromJson (← Json.field ctx "dtype" o),
         shape := ← Json.shapeField ctx "shape" o }

/-- The `"declared_types"` of a capture document. A document without them declares none if it
names no `"frontend"` (a raw IR document) and is refused if it names one. -/
def declaredTypes (document : Json) : Except String (List Declared) :=
  match document.getObjVal? "declared_types", document.getObjVal? "frontend" with
  | .ok json, _ => do pure (← (← Json.array "declared_types" json).mapM Declared.ofJson).toList
  | .error _, .ok _ => throw "IR types: the capture declares no value types (declared_types)"
  | .error _, .error _ => pure []

/-- A typed port as the type checks print it: `float32 [2, 1]`, or the bounds and shape of an index
port. -/
def Program.Typed.describe (t : Program.Typed) : String :=
  match t.port.kind, t.dtype? with
  | .index bounds, _ => s!"index tuples below {bounds} {t.port.shape}"
  | _, some d => s!"{d} {t.port.shape}"
  | _, none => s!"a value without a scalar {t.port.shape}"

/-- Whether the program gives the value of `e` its declared type: the output `e.ref` exists, with
element type `e.dtype` and shape `e.shape`. -/
def Program.declares (p : Program) (e : Declared) : Prop :=
  ∃ t, p.typedBefore p.nodes.size e.ref = some t ∧ t.dtype? = some e.dtype ∧ t.port.shape = e.shape

/-- The check that the program gives the value of `e` its declared type (`Program.declares`). -/
def Program.checkType (p : Program) (e : Declared) : Except String Unit :=
  match p.typedBefore p.nodes.size e.ref with
  | none => throw s!"IR types: node {e.node} ({e.origin}): no step output {e.ref.out} of \
      step {e.ref.node}"
  | some t =>
      if t.dtype? = some e.dtype ∧ t.port.shape = e.shape then pure ()
      else throw s!"IR types: node {e.node} ({e.origin}, step {e.ref.node} output {e.ref.out}): \
        the IR computes {t.describe}, the capture declares {e.dtype} {e.shape}"

/-- The check that the program gives every value of `declared` its declared type, first failure
first. -/
def Program.checkDeclared (p : Program) (declared : List Declared) : Except String Unit :=
  declared.forM p.checkType

/-- A passing `checkType` means the program gives the value its declared type. -/
theorem Program.checkType_ok {p : Program} {e : Declared} (h : p.checkType e = .ok ()) :
    p.declares e := by
  unfold checkType at h
  split at h
  · cases h
  · rename_i t ht
    split at h
    · rename_i hd
      exact ⟨t, ht, hd⟩
    · cases h

/-- A passing `checkDeclared` means the program gives every declared value its declared type. -/
theorem Program.checkDeclared_ok {p : Program} :
    ∀ {declared : List Declared}, p.checkDeclared declared = .ok () → ∀ e ∈ declared, p.declares e
  | [], _, e, he => by cases he
  | d :: rest, h, e, he => by
      unfold checkDeclared at h
      simp only [List.forM] at h
      obtain ⟨⟨⟩, hd, h⟩ := IR.bind_eq_ok.1 h
      rcases List.mem_cons.1 he with rfl | he
      · exact checkType_ok hd
      · exact checkDeclared_ok h e he

/-- The program and payload of a capture document's artifact under the reference runtime, with no
registered custom operations or calls, once the program gives every value of the document's
`declaredTypes` its declared type (`Program.checkDeclared`). A document without an `"artifact"`
field is refused. -/
def load (document : Json) :
    Except String (IR.Program × IR.Program.Payload .reference) := do
  let some artifact := (document.getObjVal? "artifact").toOption
    | throw "IR: the document has no artifact"
  let (p, P) ← (← IR.Artifact.ofJson artifact).load .reference ⟨fun _ => none, fun _ => none⟩
  p.checkDeclared (← declaredTypes document)
  pure (p, P)

/-- A loaded document's artifact loads to its program and payload, and the program gives every
value of the document's `declaredTypes` its declared type. -/
theorem load_eq_ok {document : Json} {p : IR.Program}
    {P : IR.Program.Payload .reference} (h : load document = .ok (p, P)) :
    ∃ artifact a declared, document.getObjVal? "artifact" = .ok artifact ∧
      IR.Artifact.ofJson artifact = .ok a ∧
      a.load .reference ⟨fun _ => none, fun _ => none⟩ = .ok (p, P) ∧
      declaredTypes document = .ok declared ∧ p.checkDeclared declared = .ok () := by
  unfold load at h
  split at h
  · rename_i artifact hartifact
    obtain ⟨a, ha, h⟩ := IR.bind_eq_ok.1 h
    obtain ⟨⟨q, Q⟩, hl, h⟩ := IR.bind_eq_ok.1 h
    obtain ⟨declared, hd, h⟩ := IR.bind_eq_ok.1 h
    obtain ⟨⟨⟩, hc, h⟩ := IR.bind_eq_ok.1 h
    cases h
    refine ⟨artifact, a, declared, ?_, ha, hl, hd, hc⟩
    cases hj : document.getObjVal? "artifact" <;> simp_all [Except.toOption]
  · cases h

/-- The program of a loaded document passes `Program.check .reference`, and its payload passes
`Payload.check`. -/
theorem load_ok {document : Json} {p : IR.Program}
    {P : IR.Program.Payload .reference} (h : load document = .ok (p, P)) :
    p.check .reference = .ok () ∧ P.check p = .ok () := by
  obtain ⟨_, _, _, -, -, hl, -⟩ := load_eq_ok h
  exact (IR.Artifact.load_ok hl).2

/-- The program of a loaded document gives every value of the document's `declaredTypes` its
declared element type and shape. -/
theorem load_declared {document : Json} {p : IR.Program}
    {P : IR.Program.Payload .reference} (h : load document = .ok (p, P)) :
    ∃ declared, declaredTypes document = .ok declared ∧ ∀ e ∈ declared, p.declares e := by
  obtain ⟨_, _, declared, -, -, -, hd, hc⟩ := load_eq_ok h
  exact ⟨declared, hd, Program.checkDeclared_ok hc⟩

/-- A tensor under the runtime `R` from its declared element type, shape and complete interchange
words, decoded at the type of that element type and shape (`Program.Typed.decode`). -/
def loadValue (R : IR.Program.Machine) (value : Json) :
    Except String (IR.Program.Value R) := do
  let dtype ← DType.fromJson (← value.getObjVal? "dtype")
  let shape ← Json.shape "tensor.shape" (← value.getObjVal? "shape")
  let words ← Json.nats "tensor.bits" (← value.getObjVal? "bits")
  let some typed := IR.Program.Typed.ofDType? dtype shape
    | throw s!"IR import: no port carries {dtype}"
  pure ⟨typed, ← typed.decode R words⟩

/-- The arguments of a capture document under the runtime `R`, in their positional order. -/
def loadArguments (R : IR.Program.Machine) (document : Json) :
    Except String (Array (IR.Program.Value R)) := do
  (← (← document.getObjVal? "arguments").getArr?).mapM (loadValue R)

/-- The element type, shape and complete words of a value under the reference runtime: its
entries (`Program.Value.entries`) written by `Reference.write`, so a binary value keeps its
signed zeros and NaN payloads and an integer value its two's-complement words. -/
def words (v : IR.Program.Value .reference) :
    Except String (DType × Shape × Array Nat) := do
  let ⟨d, s, y⟩ ← v.entries
  let entries := @Tensor.Rep.data _
    (IR.Program.Machine.reference.entryStorage d) s y
  pure (d, s, ← entries.mapM fun x =>
    Reference.write d (_root_.cast (IR.Program.entry_reference d) x))

end TorchLean.IR
