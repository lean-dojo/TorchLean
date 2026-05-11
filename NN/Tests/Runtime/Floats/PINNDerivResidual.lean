/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.PINN.Core

/-!
# PINNDerivResidual

Derivative residual regression test for the PINN certificate pipeline.

This checks that derivative-based residual bounds stored in the Python-produced certificate
match Lean's `runDeriv2D` bounds pointwise (within a small tolerance).

This module intentionally does **not** define a top-level `main` so it can be imported by other
test runners (e.g. `NN.Tests.Suite`) without a `main` name collision.
-/

@[expose] public section


open NN.Verification.PINN
open NN.MLTheory.CROWN.Graph
open Spec
open Tensor
open Lean

namespace Tests
namespace Floats
namespace PinnDerivResidual

/-- Test that derivative-based residuals from a Python certificate match Lean's `runDeriv2D` per
  point. -/
def run : IO Unit := do
  -- Kept as a file path (not an import) so this test can validate the Python-side pipeline.
  let path := "NN/Examples/Verification/PINN/pinn_cert.json"
  let jsonStr ← IO.FS.readFile path
  let j ← match Lean.Json.parse jsonStr with
    | Except.ok j => pure j
    | Except.error msg => throw <| IO.userError s!"Bad JSON: {msg}"

  let res := parseCert j
  match res with
  | .error msg => throw <| IO.userError s!"Bad Cert JSON: {msg}"
  | .ok (cfg, _, residPairsDeriv, _uTriples) => do
    let g := buildGraph
    let basePs := seedParamsFloat
    let residDerivA := residPairsDeriv.toArray
    let tol := 1e-5
    for i in List.finRange cfg.nPts do
      let x := Tensor.vecGet cfg.pts i
      let ps := seedInputFloat basePs x cfg.eps
      let boxes := runIBP (α:=Float) g ps
      let d1 := runDeriv1D (α:=Float) g ps boxes
      let d2 := runDeriv2D (α:=Float) g ps boxes d1
      let some d2B := d2[5]! | throw <| IO.userError "No d2 box at output"
      let d2lo := Spec.Tensor.sumSpec d2B.lo
      let d2hi := Spec.Tensor.sumSpec d2B.hi
      let (pyLo, pyHi) := residDerivA[i.1]!
      if ¬approxEq d2lo pyLo tol ∨ ¬approxEq d2hi pyHi tol then
        throw <| IO.userError
          s!"Derivative residual mismatch at x={x}: Lean [{d2lo},{d2hi}] vs Py [{pyLo},{pyHi}]"
    IO.println "Derivative residuals match between Python and Lean (within tolerance)."

end PinnDerivResidual
end Floats
end Tests
