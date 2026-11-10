/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.IR.Loaded

/-!
# Jaxpr capture documents

`python/torchlean/jaxpr.py` lowers a captured JAX computation to a capture document whose
`"frontend"` is `"jaxpr"`. It holds the typed artifact of format `torchlean.ir` with its
arguments and reference outputs, and beside them the XLA metadata of the JAX equation behind each
step, which `TorchLean.Verify.Capture` reads.

`load` checks the frontend marker and loads the document with `TorchLean.IR.load`, and `load_ok`
proves that the program it returns passes `Program.check .reference` and its payload passes
`Payload.check`.

The adapter and XLA lie outside the proof boundary. An accepted program passes these checks, but
nothing certifies it as a translation of its JAX source.
-/

@[expose] public section

namespace TorchLean.JAX

open Lean

/-- Check that a document's frontend marker is `jaxpr`. -/
def checkFrontend (document : Json) : Except String Unit := do
  let frontend ← (← document.getObjVal? "frontend").getStr?
  unless frontend == "jaxpr" do
    throw s!"Jaxpr import: expected frontend `jaxpr`, got `{frontend}`"

/-- The program and payload of a Jaxpr capture document under the reference runtime
(`IR.load`), once its frontend marker is checked. -/
def load (document : Json) :
    Except String (IR.Program × IR.Program.Payload .reference) := do
  checkFrontend document
  IR.load document

/-- The program of a loaded Jaxpr document passes `Program.check .reference`, and its payload
passes `Payload.check`. -/
theorem load_ok {document : Json} {p : IR.Program}
    {P : IR.Program.Payload .reference} (h : load document = .ok (p, P)) :
    p.check .reference = .ok () ∧ P.check p = .ok () := by
  obtain ⟨_, -, h⟩ := IR.bind_eq_ok.1 h
  exact IR.load_ok h

end TorchLean.JAX
