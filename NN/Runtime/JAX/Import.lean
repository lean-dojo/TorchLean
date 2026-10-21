/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.PyTorch.Import.TorchExport
public import NN.Runtime.IR

/-!
# Jaxpr graph import

`NN/Runtime/JAX/jaxpr.py` lowers captured JAX computations to the shared graph schema.
The artifact lists the positional tensor inputs in `input_ids`.
We reuse the existing graph parser and shape checker. `parseProgram` imports
per-node types and explicit casts; `parseBinaryPayload` loads exact binary words.
`parseGraph` retains the scalar-polymorphic path and rejects mixed precision.

`parsePayload` loads finite constants into native `Float` for numerical comparisons.
That evaluator uses binary64 arithmetic; it does not reproduce a captured binary16,
bfloat16, or binary32 rounding schedule. The graph can also be studied under its
real-valued semantics, with the assumptions required by each operation.

The adapter and XLA are outside the proof boundary. Accepted graphs are shape
checked, not certified translations of their source programs.
-/

@[expose] public section

namespace Import.JAX

open Lean

namespace Internal

/-- Check the frontend marker before using the shared IR readers. -/
def checkFrontend (artifact : Json) : Except String Unit := do
  let frontend ← (← artifact.getObjVal? "frontend").getStr?
  unless frontend == "jaxpr" do
    throw s!"Jaxpr import: expected frontend `jaxpr`, got `{frontend}`"

end Internal

/-- Import a Jaxpr with per-node precision and explicit binary conversions. -/
def parseProgram (artifact : Json) : Except String NN.IR.Program := do
  Internal.checkFrontend artifact
  Import.IR.parseProgram artifact

/-- Load binary constants without converting their values through native binary64. -/
def parseBinaryPayload (artifact : Json) :
    Except String (NN.IR.Interpreter.Payload NN.IR.Interpreter.binary) := do
  Internal.checkFrontend artifact
  Import.IR.parsePayload NN.IR.Interpreter.binary (fun dtype shape words => do
    let .binary format := dtype
      | throw s!"binary IR: no representation for {dtype}"
    NN.IR.Interpreter.Value.binary format shape words) artifact

namespace Internal

/-- Check that dtype metadata does not claim a cast the homogeneous IR cannot represent. -/
def checkPrecision (artifact : Json) : Except String Unit := do
  checkFrontend artifact
  let dtype ← (← artifact.getObjVal? "dtype").getStr?
  unless #["float16", "bfloat16", "float32", "float64"].contains dtype do
    throw s!"Jaxpr import: unsupported element type `{dtype}`"
  let nodes ← (← artifact.getObjVal? "nodes").getArr?
  for node in nodes do
    let nodeDtype ← (← node.getObjVal? "dtype").getStr?
    unless nodeDtype == dtype do
      throw <| s!"Jaxpr import: mixed precision `{dtype}` / `{nodeDtype}` " ++
        "needs dtype-changing IR nodes"

end Internal

/-- Import a Jaxpr artifact through the shared structural and shape validators. -/
def parseGraph (artifact : Json) : Except String PyTorch.TorchExport.CapturedGraph := do
  Internal.checkPrecision artifact
  PyTorch.TorchExport.parseGraph artifact

/-- Load constants for the native binary64 diagnostic evaluator, not source-dtype replay. -/
def parsePayload (artifact : Json) : Except String (NN.IR.Payload Float) := do
  Internal.checkPrecision artifact
  PyTorch.TorchExport.parsePayload artifact

/-- An accepted Jaxpr artifact satisfies the same shape invariant as other imported graphs. -/
theorem parseGraph_wellShaped {artifact : Json} {graph : PyTorch.TorchExport.CapturedGraph}
    (h : parseGraph artifact = .ok graph) : graph.graph.WellShaped := by
  simp only [parseGraph, Bind.bind, Except.bind] at h
  cases hPrecision : Internal.checkPrecision artifact with
  | error message => simp [hPrecision] at h
  | ok _ =>
      exact PyTorch.TorchExport.parseGraph_wellShaped (by simpa [hPrecision] using h)

end Import.JAX
