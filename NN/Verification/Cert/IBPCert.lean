/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Graph
public import NN.Runtime.PyTorch.Import.Core
public import NN.Spec.Core.Utils
public import NN.Verification.Util.FloatApprox
public import NN.Verification.Util.Json

/-!
# IBPCert

IBP “output bounds” certificate checking.

This is the simplest certificate checker in the repo: given a graph `g` and parameters `ps`, we:
1. recompute IBP bounds in Lean (`runIBP`), and
2. compare the final output interval `[lo,hi]` against a Python-exported JSON certificate.

This is primarily used by the small LiRPA-style fixtures under
`NN/Examples/Verification/LiRPA/*`.

Certificate shape:

```json
{ "result": { "lo": [...], "hi": [...] } }
```

Notes on trust boundaries:
- The JSON is an *untrusted* artifact; we only accept it if Lean recomputation agrees.
- Agreement is checked with a tolerance because the JSON uses decimal serialization.
- This checker validates an exported artifact against Lean execution. The theorem-backed path is
  separate: use `NN.Entrypoint.Verification` when you need a Lean theorem connecting checker
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
open _root_.Spec
open _root_.Spec.Tensor
open Lean
open Json
open Import.PyTorch
open NN.Verification.Json

/--
Run IBP on `(g, ps)` and compare the output box at `outId` against the JSON certificate at `path`.

Returns `true` iff bounds match componentwise (within tolerance).
On mismatch, prints both Lean and JSON bounds for debugging.
-/
def check (g : Graph) (ps : ParamStore Float) (outId : Nat) (path : String) (tol : Float := 1e-5) :
    IO Bool := do
  let boxes := runIBP (α := Float) g ps
  let some outB := boxes[outId]! |
    throw <| IO.userError s!"Lean IBP produced no output box at node {outId}"

  let topObj ← readJsonObjectFile path
  let resultObj ← expectFieldObj topObj "result" "top-level"
  let loJ ← expectField resultObj "lo" "result"
  let hiJ ← expectField resultObj "hi" "result"

  let n := outB.dim
  let some loVec := parseFloatVec n loJ | throw <| IO.userError "Missing/invalid result.lo"
  let some hiVec := parseFloatVec n hiJ | throw <| IO.userError "Missing/invalid result.hi"

  match outB.lo, outB.hi with
  | .dim vlo, .dim vhi =>
      let okLo :=
        (List.finRange n).all (fun i =>
          match vlo i with
          | .scalar v => approxEq v (loVec i) (tol := tol))
      let okHi :=
        (List.finRange n).all (fun i =>
          match vhi i with
          | .scalar v => approxEq v (hiVec i) (tol := tol))
      if okLo && okHi then
        IO.println "IBP certificate verified: Python and Lean agree."
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
        pure false

/--
Convenience wrapper for examples: run `check` and raise a readable error on mismatch.

This keeps the verification examples focused on their fixture graphs and parameter stores rather
than repeating the same “if !ok then throw” boilerplate everywhere.
-/
def checkOrThrow (g : Graph) (ps : ParamStore Float) (outId : Nat) (path : String)
    (tol : Float := 1e-5) : IO Unit := do
  let ok ← check g ps outId path (tol := tol)
  if !ok then
    throw <| IO.userError s!"IBP certificate mismatch: {path}"

end NN.Verification.IBPCert
