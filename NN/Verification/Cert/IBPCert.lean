/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.PyTorch.Import.Core
public import NN.Verification.Util.Json
public import NN.Verification.Util.Tensor
public import NN.MLTheory.CROWN.Graph.Engine.Refinement

/-!
# IBPCert

IBP “output bounds” certificate checking.

This is the simplest certificate checker in the repo: given a graph `g` and parameters `ps`, we:
1. recompute IBP bounds in Lean (`runIBP`), and
2. compare the final output interval `[lo,hi]` against a Python-exported JSON certificate.

This is primarily used by the small LiRPA-style artifacts under
`NN/Examples/Verification/LiRPA/*`.

Certificate shape:

```json
{ "result": { "lo": [...], "hi": [...] } }
```

Notes on trust boundaries:
- The JSON is an *untrusted* artifact; we only accept it if Lean recomputation agrees.
- Each serialized interval must contain the interval recomputed by TorchLean. An inward endpoint is
  rejected; an outward endpoint is allowed.
- This checker validates an exported artifact against Lean execution. The theorem-backed path is
  separate: use `NN.Verification` when you need a Lean theorem connecting checker
  hypotheses to semantic enclosure.

References (informal):
- IBP: Gowal et al. (2018).
- CROWN/LiRPA param-store model: see `NN.MLTheory.CROWN.*`.
-/

@[expose] public section


namespace NN.Verification.IBPCert

open NN.MLTheory.CROWN.Graph
open NN.MLTheory.CROWN
open NN.Verification.Util
open Spec TorchLean
open TorchLean.Tensor
open Lean
open Json
open Import.PyTorch
open NN.Verification.Json

/--
Run IBP on `(g, ps)` and compare the output box at `outId` against the JSON certificate at `path`.

Returns `true` iff the serialized bounds contain the recomputed bounds componentwise.
On mismatch, prints both Lean and JSON bounds for debugging.

`refinement := some (inputId, splitBudget)` optionally subdivides the chosen graph input before
comparing bounds. Every branch is included; the default keeps the original single-pass behavior.
-/
def check (g : Graph) (ps : ParamStore Float) (outId : Nat) (path : String)
    (refinement : Option (Nat × Nat) := none) : IO Bool := do
  unless validIBPInputs g ps do
    throw <| IO.userError "Lean IBP received missing, malformed, or invalid input bounds"
  let some outNode := g.nodes[outId]?
    | throw <| IO.userError "Lean IBP output node is missing"
  let result :=
    match refinement with
    | none => NN.MLTheory.CROWN.Graph.outputBox? (runIBP (α := Float) g ps) outId
    | some (inputId, budget) =>
      match refinedIBPOutput? g ps inputId outId budget with
      | some box => .ok box
      | none => .error
        "refinement produced no output box; check input selection and supported transfers"
  let outB ←
    match result with
    | .ok outB => pure outB
    | .error msg => throw <| IO.userError s!"Lean IBP produced no output box: {msg}"

  unless outB.dim == outNode.outShape.size && Refinement.valid outB do
    throw <| IO.userError
      "Lean IBP produced invalid output bounds (unordered or reversed endpoints)"

  let topObj ← readJsonObjectFile path
  let resultObj ← expectFieldObject topObj "result" "top-level"
  let loJ ← expectField resultObj "lo" "result"
  let hiJ ← expectField resultObj "hi" "result"

  let n := outB.dim
  let some loVec := parseFloatVec n loJ | throw <| IO.userError "Missing/invalid result.lo"
  let some hiVec := parseFloatVec n hiJ | throw <| IO.userError "Missing/invalid result.hi"
  unless (List.finRange n).all (fun i => (loVec i).isFinite && (hiVec i).isFinite) do
    throw <| IO.userError "Invalid result bounds: every value must be finite"

  let claimedLo := TorchLean.Tensor.dim (fun i => TorchLean.Tensor.scalar (loVec i))
  let claimedHi := TorchLean.Tensor.dim (fun i => TorchLean.Tensor.scalar (hiVec i))
  let okLo := NN.Verification.Util.Tensor.boundsOrdered claimedLo outB.lo
  let okHi := NN.Verification.Util.Tensor.boundsOrdered outB.hi claimedHi
  if okLo && okHi then
    IO.println "IBP certificate verified: serialized bounds enclose Lean recomputation."
    pure true
  else
    IO.println "Mismatch between Python and Lean IBP bounds."
    let pyLoStr :=
      String.intercalate ", " ((List.finRange n).map (fun i => toString (loVec i)))
    let pyHiStr :=
      String.intercalate ", " ((List.finRange n).map (fun i => toString (hiVec i)))
    IO.println s!"Lean lo: {Spec.pretty outB.lo}"
    IO.println s!"Py   lo: [{pyLoStr}]"
    IO.println s!"Lean hi: {Spec.pretty outB.hi}"
    IO.println s!"Py   hi: [{pyHiStr}]"
    for i in List.finRange n do
      let leanLo := outB.lo.getScalar i
      let leanHi := outB.hi.getScalar i
      unless loVec i ≤ leanLo && leanHi ≤ hiVec i do
        -- Decimal pretty-printing can hide the adjacent binary64 values that failed containment.
        IO.println (s!"coordinate {i.val} binary64 bits: " ++
          s!"Lean lo={leanLo.toBits}, Py lo={(loVec i).toBits}, " ++
          s!"Lean hi={leanHi.toBits}, Py hi={(hiVec i).toBits}")
    pure false

/--
Run `check` and raise a readable error on mismatch.

Verification examples use this entrypoint when the surrounding artifact should fail loudly rather
than returning a Boolean that a caller might ignore.
-/
def checkOrThrow (g : Graph) (ps : ParamStore Float) (outId : Nat) (path : String)
    (refinement : Option (Nat × Nat) := none) : IO Unit := do
  let ok ← check g ps outId path refinement
  if !ok then
    throw <| IO.userError s!"IBP certificate mismatch: {path}"

end NN.Verification.IBPCert
