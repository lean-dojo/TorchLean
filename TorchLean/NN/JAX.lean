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
`"frontend"` is `"jaxpr"`. It holds the typed artifact of format `torchlean.ir`, a flow artifact
(`TorchLean.IR.FlowArtifact`) where the computation holds a `cond` or `while` that the capture does
not decide, and that gives a value the computation returns, or a loop past the importer's count of
iterations, with its arguments and reference outputs, and beside an artifact without a flow the XLA
metadata of the JAX equation behind each step, which `TorchLean.Verify.Capture` reads.

`load` checks the frontend marker and loads the document with `TorchLean.IR.loadFlow`, and
`load_ok` proves that the program of each block of the flow it returns passes
`Program.check .reference` and its payload passes `Payload.check`. An artifact without control flow
is the flow of its one block.

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

/-- The flow of programs and payloads of a Jaxpr capture document under the reference runtime,
and the fuel of its loops (`IR.loadFlow`), once its frontend marker is checked. -/
def load (document : Json) :
    Except String (IR.Flow (IR.Program × IR.Program.Payload .reference) × Nat) := do
  checkFrontend document
  IR.loadFlow document

/-- The program of each block of a loaded Jaxpr document passes `Program.check .reference`, and its
payload passes `Payload.check`. -/
theorem load_ok {document : Json} {flow : IR.Flow (IR.Program × IR.Program.Payload .reference)}
    {fuel : Nat} (h : load document = .ok (flow, fuel)) :
    ∀ b ∈ flow.blocks, b.1.check .reference = .ok () ∧ b.2.check b.1 = .ok () := by
  obtain ⟨_, -, h⟩ := IR.bind_eq_ok.1 h
  exact (IR.loadFlow_ok h).1

end TorchLean.JAX
