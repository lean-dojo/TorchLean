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

Acceptance means that the exported endpoints match the recomputation within `certTol`. This is
not an exact-real residual enclosure theorem or a bound on PDE solution error.

This checker always uses `referenceParams 1`. The JSON's `model` and `pinn.domain` fields are
producer metadata, not validated inputs; acceptance does not authenticate their weights or domain.
Use the weight-loading CLI for a trained model.

References / context:
- PINNs: Raissi et al. (2019), "Physics-informed neural networks" (JCP)
- CROWN/LiRPA background (for the bound propagation machinery): `https://arxiv.org/abs/1811.00866`

Export (Python):
`python3 scripts/verification/pinn/export_pinn_cert.py`

Run (Lean):
`scripts/lake.sh exe verify -- pinn-cert [NN/Examples/Verification/PINN/pinn_cert.json]`
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

/-- Bundled PINN certificate sample used by `scripts/lake.sh exe verify -- pinn-cert`. -/
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
    let outId := SequentialPINNArch.graphOutputId g
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
        solutionTriplet := solutionTriplet.push
          (← IO.ofExcept (outputInterval "PINN IBP failed" boxes outId))
        let dboxes := NN.MLTheory.CROWN.Graph.runScalarDerivative (α:=Float) g ps boxes
        derivativeTriplet := derivativeTriplet.push
          (← IO.ofExcept
            (outputInterval "PINN first-derivative propagation failed" dboxes outId))
        let d2boxes :=
          NN.MLTheory.CROWN.Graph.runSecondDirectionalDerivative (α := Float) g ps boxes dboxes
        secondDerivativeTriplet := secondDerivativeTriplet.push
          (← IO.ofExcept
            (outputInterval "PINN second-derivative propagation failed" d2boxes outId))
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
        -- Recompute the residual from the PDE expression (grammar in `PdeParse`). The certificate
        -- format defines no named constants, so the identifier environment is empty.
        let pdeParsed ←
          match parseExpr (fun _ => none) config.pde with
          | .ok e => pure e
          | .error msg => throw <| IO.userError s!"PINN PDE parse failed: {msg}"
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
