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
passes `Program.check .reference` and its payload passes `Payload.check`. `loadArguments` reads
the arguments under a runtime, `loadValue` one tensor, and `words` gives the element type, shape
and words of a value back.
-/

@[expose] public section

namespace TorchLean.IR

open Lean
open FloatLib.Floats.Formats.BinaryInterchange

/-- The program and payload of a capture document's artifact under the reference runtime, with no
registered custom operations or calls. A document without an `"artifact"` field is refused. -/
def load (document : Json) :
    Except String (IR.Program × IR.Program.Payload .reference) := do
  let some artifact := (document.getObjVal? "artifact").toOption
    | throw "IR: the document has no artifact"
  (← IR.Artifact.ofJson artifact).load .reference ⟨fun _ => none, fun _ => none⟩

/-- The program of a loaded document passes `Program.check .reference`, and its payload passes
`Payload.check`. -/
theorem load_ok {document : Json} {p : IR.Program}
    {P : IR.Program.Payload .reference} (h : load document = .ok (p, P)) :
    p.check .reference = .ok () ∧ P.check p = .ok () := by
  unfold load at h
  split at h
  · obtain ⟨_, -, h⟩ := IR.bind_eq_ok.1 h
    exact (IR.Artifact.load_ok h).2
  · cases h

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
