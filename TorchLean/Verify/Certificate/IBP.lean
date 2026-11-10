/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.PyTorch.Import
public import TorchLean.NN.Json
public import TorchLean.Verify.Tensor
public import TorchLean.Verify.Certificate.DecimalJSON
public import TorchLean.Verify.DecimalRounding
public import TorchLean.Verify.CROWN.Op.Exec
public import TorchLean.Verify.Certificate.Report

/-!
# IBP output-bound certificates

Checking IBP output-bound certificates.

Given a graph of registered operators `G` with its payload `P` and a box at each graph input, the
checker recomputes IBP bounds in Lean (`TorchLean.CROWN.Op.runIBP`). The box `[lo,hi]` of one
graph output is then compared with a JSON certificate exported from Python.

Each node's box comes from the boxes of its parents, as the coordinatewise minimum and maximum of
Eq. (5) in Gowal et al. (2018) do. Lean runs this at `Float`, where `instBoundOpsFloat` moves each
endpoint sum, difference and product one binary64 value outward. At a scalar with a
`LawfulNonlinearBoundOps` instance (`ℝ` and `FP32` have one, `Float` does not),
`TorchLean.CROWN.Op.runIBP_encloses` proves, for a box `xB` on graph input `0`, that each box
encloses its node's value in a successful real run whose input `0` lies in `xB`.

The `ibp-cert` example (`TorchLeanExamples/Verification/IBPCert/MLP.lean`) checks its bundled
certificate this way.

Certificate shape:

```json
{ "result": { "lo": [...], "hi": [...] } }
```

Notes on trust boundaries:
- The JSON is an *untrusted* artifact and is accepted only if it agrees with Lean's
  recomputation.
- Each serialized interval must contain the interval recomputed by TorchLean. Each decimal endpoint
  is rounded once into binary64, the lower endpoint toward `-∞` and the upper toward `+∞`, the
  rounded value must be finite, and containment is checked exactly on those values. An endpoint
  inward of Lean's by one binary64 ulp or more is rejected, and one that is closer reads as Lean's
  value.
- Acceptance compares bounds with Lean's execution and on its own proves no enclosure over the real
  inputs. `Certificate.CROWNQuery.acceptsText_sound` proves an output property for its separate
  exact-rational dense/ReLU format, and `Certificate.FiniteArtifact.accepts_graph_sound` covers
  finite binary32 affine artifacts with extra exact dominance checks.

References (informal):
- IBP: Gowal et al., "On the Effectiveness of Interval Bound Propagation for Training Verifiably
  Robust Models", 2018, arXiv:1810.12715, Sec. 3, Eq. (5).
- IBP on operator graphs: see `TorchLean.CROWN.Op.runIBP`.
-/

@[expose] public section


namespace TorchLean.Certificate.IBP

open CROWN
open TorchLean.Tensor
open Lean
open Lean.Json
open Tensor (boundsOrdered)

/--
Run IBP on `(G, P)` from the boxes `inputs` (box `k` at graph input `k`) and compare the box of
graph output `output` with the JSON certificate at `path`. The report accepts the certificate iff
the serialized bounds contain the recomputed bounds componentwise. On a mismatch its lines give
Lean's and the JSON's bounds and the binary64 bits of each coordinate that fails containment.
A graph that fails `Graph.check`, input boxes that fail `Op.validInputs`, a missing output box, a
malformed certificate and an endpoint that rounds to an infinity raise an error.

`refinement := some (k, budget)` bounds the output with `Graph.Refinement.bound`, which splits the
box of graph input `k` at most `budget` times and keeps every branch. With `none`, IBP runs once.
-/
def check (G : IR.Graph) (P : IR.Payload Float) (inputs : List (FlatBox Float))
    (output : Nat) (path : String) (refinement : Option (Nat × Nat) := none) : IO Report := do
  if let .error e := G.check then
    throw <| IO.userError s!"Lean IBP graph rejected: {e}"
  unless Op.validInputs G inputs do
    throw <| IO.userError "Lean IBP received missing, malformed, or invalid input bounds"
  let some port := G.outputs[output]?.bind fun r => (G.nodes[r.node]?).bind (·.outs[r.out]?)
    | throw <| IO.userError "Lean IBP output node is missing"
  let run (xs : List (FlatBox Float)) : Except String (FlatBox Float) :=
    Op.outputBox? G (Op.runIBP G P (Op.inputSeeds G xs)) output
  let result :=
    match refinement with
    | none => run inputs
    | some (k, budget) =>
      let enclose (box : FlatBox Float) : Option (FlatBox Float) :=
        match run (inputs.set k box) with
        | .ok out =>
          if out.dim == port.shape.size && boundsOrdered out.lo out.hi then some out else none
        | .error _ => none
      match inputs[k]?.bind fun box => Graph.Refinement.bound enclose box budget with
      | some box => .ok box
      | none => .error
        "refinement produced no output box; check input selection and supported transfers"
  let outB ←
    match result with
    | .ok outB => pure outB
    | .error msg => throw <| IO.userError s!"Lean IBP produced no output box: {msg}"

  unless outB.dim == port.shape.size && boundsOrdered outB.lo outB.hi do
    throw <| IO.userError
      "Lean IBP produced invalid output bounds (unordered or reversed endpoints)"

  let top ← DecimalJson.readObjectFile path
  let resultV ← IO.ofExcept <| top.field "result" "top-level"
  let loJ ← IO.ofExcept <| resultV.field "lo" "result"
  let hiJ ← IO.ofExcept <| resultV.field "hi" "result"

  let n := outB.dim
  let some loNums := loJ.decimals? n
    | throw <| IO.userError "Missing/invalid result.lo"
  let some hiNums := hiJ.decimals? n
    | throw <| IO.userError "Missing/invalid result.hi"
  let loVec i := DecimalRounding.float .down (loNums i)
  let hiVec i := DecimalRounding.float .up (hiNums i)
  unless (List.finRange n).all (fun i => (loVec i).isFinite && (hiVec i).isFinite) do
    throw <| IO.userError "Invalid result bounds: every value must be finite"

  let claimedLo := Tensor.dim (fun i => Tensor.scalar (loVec i))
  let claimedHi := Tensor.dim (fun i => Tensor.scalar (hiVec i))
  let okLo := boundsOrdered claimedLo outB.lo
  let okHi := boundsOrdered outB.hi claimedHi
  if okLo && okHi then
    return { lines := #["IBP certificate verified: serialized bounds enclose Lean recomputation."] }
  let pyLoStr :=
    String.intercalate ", " ((List.finRange n).map (fun i => toString (loVec i)))
  let pyHiStr :=
    String.intercalate ", " ((List.finRange n).map (fun i => toString (hiVec i)))
  -- Decimal pretty-printing can hide the adjacent binary64 values that failed containment.
  let bits := (List.finRange n).filterMap fun i =>
    let leanLo := outB.lo.getScalar i
    let leanHi := outB.hi.getScalar i
    if loVec i ≤ leanLo && leanHi ≤ hiVec i then none
    else some (s!"coordinate {i.val} binary64 bits: " ++
      s!"Lean lo={leanLo.toBits}, Py lo={(loVec i).toBits}, " ++
      s!"Lean hi={leanHi.toBits}, Py hi={(hiVec i).toBits}")
  return {
    lines := #["Mismatch between Python and Lean IBP bounds.",
      s!"Lean lo: {Tensor.pretty outB.lo}", s!"Py   lo: [{pyLoStr}]",
      s!"Lean hi: {Tensor.pretty outB.hi}", s!"Py   hi: [{pyHiStr}]"] ++ bits.toArray
    rejection := some s!"IBP certificate mismatch: {path}" }

end TorchLean.Certificate.IBP
