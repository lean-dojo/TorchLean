/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLeanExamples.Verification.PINN
public import TorchLean.Verify.Numeric.ApproxEq
public import TorchLeanExamples.Verification.PINN.PDEParse
public import TorchLeanExamples.Verification.PINN.PyTorch.Model

/-!
# PINN Certificate

The executable recompute-and-compare checker of the PINN certificate workflow. It parses a JSON
certificate produced by Python, rebuilds the same graph of registered operators and the same input
boxes, recomputes the IBP and derivative bounds in Lean, and compares the resulting bounds and
residual intervals with the exported values.

The graph is the network of the certificate's `model` state dict, which must have one input and
one output, and every sample point must lie in `pinn.domain`. `parseModel_eq_ok` and
`checkDomain_sound` prove what these two input checks accept.

Acceptance means that every exported endpoint matches the recomputation within the absolute
tolerance `certTol`. This is a numerical tolerance check: it gives no exact-real residual
enclosure and no bound on the error of a PDE solution.

References / context:
- PINNs: Raissi et al. (2019), "Physics-informed neural networks" (JCP)
- CROWN/LiRPA background (for the bound propagation machinery): `https://arxiv.org/abs/1811.00866`

Export (Python):
`python3 TorchLeanExamples/Verification/PINN/export_pinn_cert.py`

Run (Lean):
`scripts/lake.sh exe verify -- pinn-cert [TorchLeanExamples/Verification/PINN/pinn_cert.json]`
-/

@[expose] public section


namespace TorchLean.Examples.PINN.Certificate

open Derivatives
open Expr
open CROWN
open TorchLean.Tensor
open Lean
open Lean.Json

/-- Bundled PINN certificate sample used by `scripts/lake.sh exe verify -- pinn-cert`. -/
def defaultPath : String :=
  "TorchLeanExamples/Verification/PINN/pinn_cert.json"

/-- Reject a certificate interval unless each endpoint lies within `certTol` of the matching
endpoint of Lean's recomputed interval and all four endpoints are finite. -/
def requireApproxInterval (ctx : String)
    (leanBounds artifactBounds : FloatLib.Numerics.Interval Float) : IO Unit :=
  if Numeric.approxEq leanBounds.lo artifactBounds.lo (tol := certTol) &&
      Numeric.approxEq leanBounds.hi artifactBounds.hi (tol := certTol) then
    pure ()
  else
    throw <| IO.userError
      s!"{ctx}: Lean {repr leanBounds} differs from certificate {repr artifactBounds}"

/-- The certificate's network: the state dict under `model`, with one input and one output. -/
def parseModel (j : Json) : Except String Checkpoint := do
  let model ←
    Json.object "PINN certificate" j >>= Json.field "PINN certificate" "model"
  let some state := Checkpoint.ofJson? model
    | throw "PINN certificate.model: expected `layers.<i>.weight` and `layers.<i>.bias` \
        entries of finite numbers with chained widths and a supported `meta.activation`"
  if state.arch.inputDim = 1 ∧ state.arch.outputDim = 1 then
    pure state
  else
    throw s!"PINN certificate.model: expected one input and one output, got \
      {state.arch.inputDim} and {state.arch.outputDim}"

/-- `parseModel` returns exactly the state dicts under `model` that load and have one input and
one output. -/
theorem parseModel_eq_ok {j : Json} {state : Checkpoint} :
    parseModel j = .ok state ↔
      ∃ model, (Json.object "PINN certificate" j >>=
          Json.field "PINN certificate" "model") = .ok model ∧
        Checkpoint.ofJson? model = some state ∧
        state.arch.inputDim = 1 ∧ state.arch.outputDim = 1 := by
  unfold parseModel
  rcases Json.object "PINN certificate" j >>=
      Json.field "PINN certificate" "model" with e | model
  · simp [bind, Except.bind]
  simp only [bind, Except.bind, Except.ok.injEq, exists_eq_left']
  rcases Checkpoint.ofJson? model with _ | s
  · simp [throw, throwThe, MonadExceptOf.throw]
  simp only [Option.some.injEq]
  by_cases hd : s.arch.inputDim = 1 ∧ s.arch.outputDim = 1
  · simp only [hd, and_self, ↓reduceIte, pure, Except.pure, Except.ok.injEq]
    constructor
    · rintro rfl; exact ⟨rfl, hd⟩
    · rintro ⟨rfl, -⟩; rfl
  · simp only [hd, ↓reduceIte, throw, throwThe, MonadExceptOf.throw, reduceCtorEq, false_iff,
      not_and]
    rintro rfl h1 h2; exact hd ⟨h1, h2⟩

/-- Whether every sample point of `config` lies in `[lo, hi]`. -/
def pointsWithin (lo hi : Float) (config : CertificateConfig) : Bool :=
  (List.finRange config.pointCount).all fun i =>
    lo ≤ Tensor.getScalar config.points i && Tensor.getScalar config.points i ≤ hi

/-- `pointsWithin` holds exactly when `lo ≤ x ≤ hi` for every sample point `x`. -/
theorem pointsWithin_eq_true {lo hi : Float} {config : CertificateConfig} :
    pointsWithin lo hi config = true ↔
      ∀ i : Fin config.pointCount,
        lo ≤ Tensor.getScalar config.points i ∧ Tensor.getScalar config.points i ≤ hi := by
  simp [pointsWithin, List.all_eq_true]

/-- Read `pinn.domain = [lo, hi]`, require `lo ≤ hi` and every sample point of `config` in
`[lo, hi]`, and return `(lo, hi)`. -/
def checkDomain (j : Json) (config : CertificateConfig) : Except String (Float × Float) := do
  let po ← Json.object "PINN certificate" j >>=
    Json.field "PINN certificate" "pinn" >>= Json.object "PINN certificate.pinn"
  let ctx := "PINN certificate.pinn.domain"
  let bounds ← Json.field "PINN certificate.pinn" "domain" po >>=
    Json.parseFiniteFloatArray "PINN certificate.pinn.domain"
  let [lo, hi] := bounds.toList
    | throw s!"{ctx}: expected [lo, hi], got {bounds.size} entries"
  unless lo ≤ hi do throw s!"{ctx}: lower endpoint {lo} exceeds upper endpoint {hi}"
  unless pointsWithin lo hi config do
    throw s!"{ctx}: a sample point lies outside [{lo}, {hi}]"
  pure (lo, hi)

/-- When `checkDomain` returns `(lo, hi)`, these are the two entries of `pinn.domain`,
`lo ≤ hi`, and every sample point lies in `[lo, hi]`. -/
theorem checkDomain_sound {j : Json} {config : CertificateConfig} {lo hi : Float}
    (h : checkDomain j config = .ok (lo, hi)) :
    (∃ po, (Json.object "PINN certificate" j >>=
        Json.field "PINN certificate" "pinn" >>=
        Json.object "PINN certificate.pinn") = .ok po ∧
      (Json.field "PINN certificate.pinn" "domain" po >>=
        Json.parseFiniteFloatArray "PINN certificate.pinn.domain") = .ok #[lo, hi]) ∧
    lo ≤ hi ∧ ∀ i : Fin config.pointCount,
      lo ≤ Tensor.getScalar config.points i ∧ Tensor.getScalar config.points i ≤ hi := by
  unfold checkDomain at h
  rcases hp : Json.object "PINN certificate" j >>=
      Json.field "PINN certificate" "pinn" >>=
      Json.object "PINN certificate.pinn" with e | po
  · rw [hp] at h
    simp [bind, Except.bind] at h
  rcases hb : Json.field "PINN certificate.pinn" "domain" po >>=
      Json.parseFiniteFloatArray "PINN certificate.pinn.domain" with e | bounds
  · rw [hp] at h
    simp only [bind, Except.bind] at h hb
    simp [hb] at h
  have hd := hb
  rw [hp] at h
  simp only [bind, Except.bind] at h hd
  rw [hd] at h
  simp only at h
  split at h
  · rename_i lo' hi' hl
    have hb' : bounds = #[lo', hi'] := by rw [← hl, Array.toArray_toList]
    by_cases h1 : lo' ≤ hi'
    · by_cases h2 : pointsWithin lo' hi' config = true
      · simp [h1, h2, pure, Except.pure] at h
        obtain ⟨rfl, rfl⟩ := h
        exact ⟨⟨po, rfl, hb' ▸ hb⟩, h1, pointsWithin_eq_true.mp h2⟩
      · simp [h1, h2, throw, throwThe, MonadExceptOf.throw] at h
    · simp [h1, throw, throwThe, MonadExceptOf.throw] at h
  · simp [throw, throwThe, MonadExceptOf.throw] at h

/-- Replay the certificate at `path`: recompute its bounds and residuals with the network of its
`model` state dict, print them, and throw on the first value that differs from the certificate. -/
def verify (path : String) : IO Unit := do
  let j ← Json.readFile path
  match parseCertificate j with
  | .error msg => throw <| IO.userError s!"Bad Cert JSON: {msg}"
  | .ok certificate => do
    let config := certificate.config
    let state ← IO.ofExcept (parseModel j)
    let _ ← IO.ofExcept (checkDomain j config)
    let (G, P) ← IO.ofExcept (Checkpoint.model (α := Float) id state)
    let D ← IO.ofExcept (ofGraph G P)
    unless D.hasSecond do
      throw <| IO.userError "PINN second-derivative propagation failed: no second tangent graph"
    -- The derivative along the only input coordinate: the direction `1`.
    let direction : FlatBox Float := FlatBox.ofTensor (Tensor.full [1] 1)
    for i in Array.finRange config.pointCount do
      let x := Tensor.getScalar config.points i
      let xs := #[x - config.spacing, x, x + config.spacing]
      let mut solutionTriplet : Array (FloatLib.Numerics.Interval Float) := #[]
      let mut derivativeTriplet : Array (FloatLib.Numerics.Interval Float) := #[]
      let mut secondDerivativeTriplet : Array (FloatLib.Numerics.Interval Float) := #[]
      for xi in xs do
        let center : Tensor Float [1] :=
          Tensor.dim fun _ => Tensor.scalar xi
        let xB := FlatBox.lInfBall center config.radius
        let boxes := Op.runIBP D.graph D.payload (Op.inputSeeds D.graph [xB, direction])
        solutionTriplet := solutionTriplet.push
          (← IO.ofExcept (outputInterval "PINN IBP failed" D.graph boxes 0))
        derivativeTriplet := derivativeTriplet.push
          (← IO.ofExcept
            (outputInterval "PINN first-derivative propagation failed" D.graph boxes 1))
        secondDerivativeTriplet := secondDerivativeTriplet.push
          (← IO.ofExcept
            (outputInterval "PINN second-derivative propagation failed" D.graph boxes 3))
      match solutionTriplet[0]?, solutionTriplet[1]?, solutionTriplet[2]? with
      | some previous, some center, some next =>
        let artifactBounds ←
          match certificate.solutionBounds[i.1]? with
          | some entry => pure entry
          | none =>
              throw <| IO.userError <|
                s!"PINN certificate u_bounds missing index {i.1} " ++
                s!"(size={certificate.solutionBounds.size})"
        if !Numeric.approxEq x artifactBounds.point (tol := certTol) then
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
        -- Recompute the residual from the PDE expression (the grammar of `parseExpr`). The
        -- certificate format defines no named constants, so the identifier environment is empty.
        let pdeParsed ←
          match parseExpr (fun _ => none) config.pde with
          | .ok e => pure e
          | .error msg => throw <| IO.userError s!"PINN PDE parse failed: {msg}"
        let prims ←
          match solutionTriplet[1]?, derivativeTriplet[1]?, secondDerivativeTriplet[1]? with
          | some solution, some derivative, some secondDerivative =>
            pure
              { u := some (solution.lo, solution.hi)
                duX := some (derivative.lo, derivative.hi)
                duY := none
                d2uX := some (secondDerivative.lo, secondDerivative.hi)
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
          { lo := residualLower, hi := residualUpper } derivativeArtifact
        IO.println
          s!"Residual R(x) from PDE '{config.pde}': [{residualLower},{residualUpper}]"
        match derivativeTriplet[0]?, derivativeTriplet[1]?, derivativeTriplet[2]? with
        | some previousDerivative, some centerDerivative, some nextDerivative =>
          IO.println <|
            s!"u'(x-h)∈[{previousDerivative.lo},{previousDerivative.hi}], " ++
            s!"u'(x)∈[{centerDerivative.lo},{centerDerivative.hi}], " ++
            s!"u'(x+h)∈[{nextDerivative.lo},{nextDerivative.hi}]"
        | _, _, _ => pure ()
        match secondDerivativeTriplet[0]?, secondDerivativeTriplet[1]?,
            secondDerivativeTriplet[2]? with
        | some previousSecond, some centerSecond, some nextSecond =>
          IO.println <|
            s!"u''(x-h)∈[{previousSecond.lo},{previousSecond.hi}], " ++
            s!"u''(x)∈[{centerSecond.lo},{centerSecond.hi}], " ++
            s!"u''(x+h)∈[{nextSecond.lo},{nextSecond.hi}]"
        | _, _, _ => pure ()
      | _, _, _ => throw <| IO.userError "unexpected number of PINN stencil samples"
    IO.println "PINN artifact replay matched Lean's recomputed residual bounds."

end TorchLean.Examples.PINN.Certificate
