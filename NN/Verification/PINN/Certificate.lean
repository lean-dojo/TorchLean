/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.PINN.Core
public import NN.Verification.Util.FloatApprox
public import NN.Verification.PINN.PdeParse

/-!
# PINN Certificate

PINN certificate checker (recompute-and-compare).

This module is the executable checker for the PINN certificate workflow:
- parse a JSON certificate produced by Python,
- rebuild the same CROWN graph and seed the same input boxes,
- recompute IBP + derivative bounds in Lean, and
- compare the resulting residual intervals against the exported values.

It is conservative by design: it validates the export/import path and interval computations, rather
than trying to be a fully featured PDE verifier.

References / context:
- PINNs: Raissi et al. (2019), "Physics-informed neural networks" (JCP)
- CROWN/LiRPA background (for the bound propagation machinery): `https://arxiv.org/abs/1811.00866`

Export (Python):
`python3.12 scripts/verification/pinn/export_pinn_cert.py`

Run (Lean):
`lake exe verify -- pinn-cert [NN/Examples/Verification/PINN/pinn_cert.json]`
-/

@[expose] public section


namespace NN.Verification.PINN.Certificate

open NN.Verification.PINN
open NN.Verification.PINN.PdeAst
open NN.Verification.PINN.PdeParse
open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Graph
open Spec TorchLean
open TorchLean.Tensor
open Lean
open Json

/-- Bundled PINN certificate sample used by `lake exe verify -- pinn-cert`. -/
def defaultCertPath : String :=
  "NN/Examples/Verification/PINN/pinn_cert.json"

/-- Reject a certificate interval that differs from Lean's recomputed interval. -/
def requireApproxInterval (ctx : String)
    (leanBounds artifactBounds : FloatInterval) : IO Unit :=
  if Util.approxEq leanBounds.lower artifactBounds.lower (tol := certTol) &&
      Util.approxEq leanBounds.upper artifactBounds.upper (tol := certTol) then
    pure ()
  else
    throw <| IO.userError
      s!"{ctx}: Lean {repr leanBounds} differs from certificate {repr artifactBounds}"

/-- IO entry that reads the cert, recomputes bounds, and prints comparisons. -/
def verifyCert (path : String) : IO Unit := do
  let j ← NN.Verification.Json.readJsonFile path
  match parseCertificate j with
  | .error msg => throw <| IO.userError s!"Bad Cert JSON: {msg}"
  | .ok certificate => do
    let config := certificate.config
    let g := buildReferenceGraph 1
    let outId := g.nodes.size - 1
    let basePs : ParamStore Float := referenceParams 1
    for i in Array.finRange config.pointCount do
      let x := Tensor.getScalar config.points i
      let xs := #[x - config.spacing, x, x + config.spacing]
      let mut solutionTriplet : Array FloatInterval := #[]
      let mut derivativeTriplet : Array FloatInterval := #[]
      let mut secondDerivativeTriplet : Array FloatInterval := #[]
      for xi in xs do
        let center : TorchLean.Tensor Float [1] :=
          TorchLean.Tensor.dim fun _ => TorchLean.Tensor.scalar xi
        let ps := seedInput basePs center config.radius
        let boxes := NN.MLTheory.CROWN.Graph.runIBP (α:=Float) g ps
        let outB ←
          match NN.MLTheory.CROWN.Graph.outputBox? boxes outId with
          | .ok outB => pure outB
          | .error msg => throw <| IO.userError s!"PINN IBP failed: {msg}"
        let loVal := TorchLean.Tensor.sumSpec outB.lo
        let hiVal := TorchLean.Tensor.sumSpec outB.hi
        solutionTriplet := solutionTriplet.push { lower := loVal, upper := hiVal }
        let dboxes := NN.MLTheory.CROWN.Graph.runScalarDerivative (α:=Float) g ps boxes
        let dB ←
          match NN.MLTheory.CROWN.Graph.outputBox? dboxes outId with
          | .ok dB => pure dB
          | .error msg => throw <| IO.userError s!"PINN first-derivative propagation failed: {msg}"
        let dlo := TorchLean.Tensor.sumSpec dB.lo
        let dhi := TorchLean.Tensor.sumSpec dB.hi
        derivativeTriplet := derivativeTriplet.push { lower := dlo, upper := dhi }
        let d2boxes :=
          NN.MLTheory.CROWN.Graph.runScalarSecondDerivative (α := Float) g ps boxes dboxes
        let d2B ←
          match NN.MLTheory.CROWN.Graph.outputBox? d2boxes outId with
          | .ok d2B => pure d2B
          | .error msg => throw <| IO.userError s!"PINN second-derivative propagation failed: {msg}"
        let d2lo := TorchLean.Tensor.sumSpec d2B.lo
        let d2hi := TorchLean.Tensor.sumSpec d2B.hi
        secondDerivativeTriplet :=
          secondDerivativeTriplet.push { lower := d2lo, upper := d2hi }
      match solutionTriplet[0]?, solutionTriplet[1]?, solutionTriplet[2]? with
      | some previous, some center, some next =>
        let artifactBounds ←
          match certificate.solutionBounds[i.1]? with
          | some entry => pure entry
          | none =>
              throw <| IO.userError <|
                s!"PINN certificate u_bounds missing index {i.1} " ++
                s!"(size={certificate.solutionBounds.size})"
        if !Util.approxEq x artifactBounds.point (tol := certTol) then
          throw <| IO.userError <|
            s!"PINN certificate point mismatch at index {i.1}: " ++
            s!"Lean {x}, certificate {artifactBounds.point}"
        requireApproxInterval s!"u(x-h) mismatch at x={x}" previous artifactBounds.previous
        requireApproxInterval s!"u(x) mismatch at x={x}" center artifactBounds.center
        requireApproxInterval s!"u(x+h) mismatch at x={x}" next artifactBounds.next
        let residual := finiteDifferenceResidual previous center next config.spacing
        let artifactResidual ←
          match certificate.residualBounds[i.1]? with
          | some bounds => pure bounds
          | none =>
              throw <| IO.userError <|
                s!"PINN certificate residual missing index {i.1} " ++
                s!"(size={certificate.residualBounds.size})"
        requireApproxInterval
          s!"finite-difference residual mismatch at x={x}" residual artifactResidual
        let derivativeArtifact ←
          match certificate.derivativeResidualBounds[i.1]? with
          | some bounds => pure bounds
          | none =>
              throw <| IO.userError <|
                s!"PINN certificate derivative residual missing index {i.1} " ++
                s!"(size={certificate.derivativeResidualBounds.size})"
        -- Compute and print residual bounds from the PDE specification via the parser/AST.
        -- We support a small DSL: u, ux, uxx, uy, uyy, +, -, *, scaling constants, parentheses, and
        -- powers by ^n.
        let env : String → Option Float := fun _ => none
        -- identifiers map, can be extended to constants
        let pdeParsed ←
          match parseExpr env config.pde with
          | .ok e => pure e
          | .error msg => throw <| IO.userError s!"PINN PDE parse failed: {msg}"
        -- Build primitive bounds at the central point x using computed intervals
        let prims ←
          match solutionTriplet[1]?, derivativeTriplet[1]?, secondDerivativeTriplet[1]? with
          | some solution, some derivative, some secondDerivative =>
            pure
              { u := some (solution.lower, solution.upper)
                duX := some (derivative.lower, derivative.upper)
                duY := none
                d2uX := some (secondDerivative.lower, secondDerivative.upper)
                d2uY := none }
          | _, _, _ => throw <| IO.userError "PINN derivative samples are incomplete"
        let pdeResidual ←
          match eval prims pdeParsed with
          | some residual => pure residual
          | none =>
              throw <| IO.userError
                (s!"PINN PDE '{config.pde}' evaluation failed " ++
                  "because required primitives are missing")
        let (residualLower, residualUpper) := pdeResidual
        requireApproxInterval s!"derivative residual mismatch at x={x}"
          { lower := residualLower, upper := residualUpper } derivativeArtifact
        IO.println
          s!"Residual R(x) from PDE '{config.pde}': [{residualLower},{residualUpper}]"
        match derivativeTriplet[0]?, derivativeTriplet[1]?, derivativeTriplet[2]? with
        | some previousDerivative, some centerDerivative, some nextDerivative =>
          IO.println <|
            s!"u'(x-h)∈[{previousDerivative.lower},{previousDerivative.upper}], " ++
            s!"u'(x)∈[{centerDerivative.lower},{centerDerivative.upper}], " ++
            s!"u'(x+h)∈[{nextDerivative.lower},{nextDerivative.upper}]"
        | _, _, _ => pure ()
        match secondDerivativeTriplet[0]?, secondDerivativeTriplet[1]?,
            secondDerivativeTriplet[2]? with
        | some previousSecond, some centerSecond, some nextSecond =>
          IO.println <|
            s!"u''(x-h)∈[{previousSecond.lower},{previousSecond.upper}], " ++
            s!"u''(x)∈[{centerSecond.lower},{centerSecond.upper}], " ++
            s!"u''(x+h)∈[{nextSecond.lower},{nextSecond.upper}]"
        | _, _, _ => pure ()
      | _, _, _ => throw <| IO.userError "unexpected number of PINN stencil samples"
    IO.println "PINN artifact replay matched Lean's recomputed residual bounds."

end NN.Verification.PINN.Certificate
