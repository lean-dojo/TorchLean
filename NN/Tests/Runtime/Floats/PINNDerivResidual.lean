/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.PINN.Core

/-!
# PINNDerivResidual

Derivative-residual regression test for the PINN certificate pipeline.

This checks that Lean's `runScalarSecondDerivative` enclosure contains each derivative-residual
interval stored in the Python-produced certificate. Equality is neither required nor expected:
the executable CROWN pass may use a coarser sound relaxation than the certificate producer.

This module has no top-level `main`, so it can be imported by test runners
(e.g. `NN.Tests.Suite`) without a `main` name collision.
-/

@[expose] public section


open NN.Verification.PINN
open NN.MLTheory.CROWN.Graph
open Spec TorchLean
open TorchLean TorchLean.Tensor
open Lean

namespace Tests
namespace Floats
namespace PinnDerivResidual

/-- Check that Lean encloses every derivative-residual interval supplied by the certificate. -/
def run : IO Unit := do
  -- Kept as a file path (not an import) so this test can validate the Python-side pipeline.
  let path := "NN/Examples/Verification/PINN/pinn_cert.json"
  let jsonStr ← IO.FS.readFile path
  let j ← match Lean.Json.parse jsonStr with
    | Except.ok j => pure j
    | Except.error msg => throw <| IO.userError s!"Bad JSON: {msg}"

  let res := parseCertificate j
  match res with
  | .error msg => throw <| IO.userError s!"Bad Cert JSON: {msg}"
  | .ok certificate => do
    let config := certificate.config
    let g := buildReferenceGraph 1
    let basePs : ParamStore Float := referenceParams 1
    let tol := 1e-5
    for i in Array.finRange config.pointCount do
      let x := Tensor.getScalar config.points i
      let center : TorchLean.Tensor Float [1] :=
        TorchLean.Tensor.ofFn fun _ => x
      let ps := seedInput basePs center config.radius
      let boxes := runIBP (α:=Float) g ps
      let d1 := runScalarDerivative (α:=Float) g ps boxes
      let d2 := runScalarSecondDerivative (α:=Float) g ps boxes d1
      let some d2B := d2[5]! | throw <| IO.userError "No d2 box at output"
      let d2lo := TorchLean.Tensor.sumSpec d2B.lo
      let d2hi := TorchLean.Tensor.sumSpec d2B.hi
      -- NaN makes both containment comparisons false, while an infinite endpoint can pass
      -- without giving this fixture a finite enclosure. Check the computed bounds first.
      if d2lo.isNaN || d2lo.isInf || d2hi.isNaN || d2hi.isInf then
        throw <| IO.userError
          s!"Non-finite derivative residual at x={x}: Lean [{d2lo},{d2hi}]"
      let artifactBounds ←
        match certificate.derivativeResidualBounds[i.1]? with
        | some bounds => pure bounds
        | none =>
            throw <| IO.userError <|
              s!"PINN certificate derivative residual missing index {i.1} " ++
              s!"(size={certificate.derivativeResidualBounds.size})"
      if d2lo > artifactBounds.lower + tol ∨ d2hi + tol < artifactBounds.upper then
        throw <| IO.userError <|
          s!"Derivative residual is not enclosed at x={x}: " ++
          s!"Lean [{d2lo},{d2hi}], " ++
          s!"certificate [{artifactBounds.lower},{artifactBounds.upper}]"
    IO.println "Lean encloses every derivative-residual interval in the certificate."

end PinnDerivResidual
end Floats
end Tests
