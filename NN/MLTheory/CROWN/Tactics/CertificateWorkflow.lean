/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Lean
public import Lean.Data.Json
public import Lean.Elab.Exception
public import Lean.Elab.Tactic
public import Lean.Elab.Tactic.Basic
public import Lean.Log
public meta import NN.Verification.Util.Json
public import Std.Data.HashMap

/-!
# CROWN certificate workflow tools

Tactics and helpers for loading external CROWN/IBP-style certificates (typically produced by
Python tooling), parsing JSON payloads, and inspecting producer claims.

We keep these tactics deliberately workflow-oriented. They help us inspect certificates and wire
external producers into Lean, but the trust boundary remains explicit: loading a certificate does
not prove its semantic validity. A theorem that depends on an external verifier must state the
required bound hypothesis or discharge it through a checked TorchLean graph certificate.

References:
- Zhang et al., "Efficient Neural Network Robustness Certification with General Activation
  Functions" (CROWN), NeurIPS 2018.
- Xu et al., "Automatic Perturbation Analysis for Scalable Certified Robustness and Beyond"
  (auto_LiRPA), NeurIPS 2020.
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Tactics.CertificateWorkflow

open Lean Elab Tactic Meta Term
open NN.Verification.Json

/-! ## Certificate data structures -/

/-- Parsed CROWN/Lyapunov certificate loaded from an external JSON producer. -/
structure CrownCert where
  /-- Producer-reported verification method, such as `"ibp"` or `"crown"`. -/
  method : String
  /-- Optional dynamics tag for closed-loop or Lyapunov workflows. -/
  dynamics : Option String := none
  /-- Flattened input dimension. -/
  inputDim : Nat
  /-- Lower endpoints of the input box. -/
  inputLo : Array Float
  /-- Upper endpoints of the input box. -/
  inputHi : Array Float
  /-- Lower bound for the Lyapunov candidate `V`. -/
  vLower : Float
  /-- Upper bound for the Lyapunov candidate `V`. -/
  vUpper : Float
  /-- Lower endpoints for gradient bounds, when supplied by the producer. -/
  gradLo : Array Float
  /-- Upper endpoints for gradient bounds, when supplied by the producer. -/
  gradHi : Array Float
  /-- Lower bound for the orbital derivative `Vdot`. -/
  derivativeLower : Float
  /-- Upper bound for the orbital derivative `Vdot`. -/
  derivativeUpper : Float
  /-- Whether the certificate claims `V` is positive on the region. -/
  vPositive : Bool
  /-- Whether the certificate claims `Vdot` is negative on the region. -/
  derivativeNegative : Bool
  /-- Overall producer-reported Lyapunov verification flag. -/
  lyapunovVerified : Bool
  deriving Repr, Inhabited

meta section

/-! ## JSON parsing -/

/-- Parse a Bool field when present; otherwise derive it from a fallback condition. -/
def parseBoolFieldOr (o : Json) (key : String) (fallback : Bool) : Except String Bool :=
  match o.getObjVal? key with
  | .error _ => .ok fallback
  | .ok (.bool b) => .ok b
  | .ok value => .error s!"Expected Boolean field `{key}`, got {value}"

/-- Validate the numeric and dimensional invariants shared by supported certificate schemas. -/
def CrownCert.validate (cert : CrownCert) : Except String Unit := do
  if cert.inputLo.size != cert.inputDim || cert.inputHi.size != cert.inputDim then
    let message := s!"Input region has dimension {cert.inputDim}, but its endpoint arrays have " ++
      s!"lengths {cert.inputLo.size} and {cert.inputHi.size}"
    throw message
  for i in [0:cert.inputDim] do
    let lo := cert.inputLo[i]!
    let hi := cert.inputHi[i]!
    unless lo.isFinite && hi.isFinite && lo ≤ hi do
      throw s!"Invalid input interval at coordinate {i}: [{lo}, {hi}]"
  unless cert.vLower.isFinite && cert.vUpper.isFinite && cert.vLower ≤ cert.vUpper do
    throw s!"Invalid V interval: [{cert.vLower}, {cert.vUpper}]"
  unless cert.derivativeLower.isFinite && cert.derivativeUpper.isFinite &&
      cert.derivativeLower ≤ cert.derivativeUpper do
    throw s!"Invalid Vdot interval: [{cert.derivativeLower}, {cert.derivativeUpper}]"
  unless cert.gradLo.size = cert.gradHi.size do
    throw <| s!"Gradient endpoint arrays have different lengths: " ++
      s!"{cert.gradLo.size} and {cert.gradHi.size}"
  if cert.gradLo.size != 0 && cert.gradLo.size != cert.inputDim then
    throw s!"Gradient interval has dimension {cert.gradLo.size}; expected {cert.inputDim}"
  for i in [0:cert.gradLo.size] do
    let lo := cert.gradLo[i]!
    let hi := cert.gradHi[i]!
    unless lo.isFinite && hi.isFinite && lo ≤ hi do
      throw s!"Invalid gradient interval at coordinate {i}: [{lo}, {hi}]"
  if cert.vPositive && !(cert.vLower > 0.0) then
    throw s!"Certificate reports V > 0, but its lower bound is {cert.vLower}"
  if cert.derivativeNegative && !(cert.derivativeUpper < 0.0) then
    throw s!"Certificate reports Vdot < 0, but its upper bound is {cert.derivativeUpper}"
  if cert.lyapunovVerified && !(cert.vPositive && cert.derivativeNegative) then
    throw "Certificate reports Lyapunov verification without both sign claims"

/-- Parse the input-region block used by CROWN certificates. -/
def parseInputRegion (j : Json) : Except String NN.Verification.Json.BoxRegion := do
  let inputLike ←
    match j.getObjVal? "input" with
    | .ok input => .ok input
    | .error _ => j.getObjVal? "region"
  parseBoxRegion "input" inputLike

/-- Parse a certificate from JSON, accepting supported closed-loop schemas. -/
def parseCertificate (j : Json) : Except String CrownCert := do
  let method ← j.getObjValAs? String "method"
  let dynamics := j.getObjValAs? String "dynamics" |>.toOption

  let region ← parseInputRegion j
  let inputDim := region.dim
  let inputLo := region.lo
  let inputHi := region.hi

  -- Parse V bounds
  let vBounds ← j.getObjVal? "V_bounds"
  let vLower ← vBounds.getObjVal? "lo" >>= parseFiniteFloat "V_bounds.lo"
  let vUpper ← vBounds.getObjVal? "hi" >>= parseFiniteFloat "V_bounds.hi"
  let vPositive ←
    parseBoolFieldOr vBounds "guaranteed_positive" (vLower > 0.0)

  -- Parse gradient bounds (optional; accept supported key names)
  let (gradLo, gradHi) ← do
    match j.getObjVal? "gradient_bounds" with
    | .ok gradBounds =>
      let loJson ← gradBounds.getObjVal? "lo"
      let hiJson ← gradBounds.getObjVal? "hi"
      let lo ← parseFiniteFloatArray "gradient_bounds.lo" loJson
      let hi ← parseFiniteFloatArray "gradient_bounds.hi" hiJson
      pure (lo, hi)
    | .error _ =>
      match j.getObjVal? "grad_bounds" with
      | .ok gradBounds =>
        let loJson ← gradBounds.getObjVal? "lo"
        let hiJson ← gradBounds.getObjVal? "hi"
        let lo ← parseFiniteFloatArray "grad_bounds.lo" loJson
        let hi ← parseFiniteFloatArray "grad_bounds.hi" hiJson
        pure (lo, hi)
      | .error _ =>
        pure (#[], #[])

  -- Parse Vdot bounds
  let vdotBounds ← j.getObjVal? "Vdot_bounds"
  let derivativeLower ← vdotBounds.getObjVal? "lo" >>= parseFiniteFloat "Vdot_bounds.lo"
  let derivativeUpper ← vdotBounds.getObjVal? "hi" >>= parseFiniteFloat "Vdot_bounds.hi"
  let derivativeNegative ←
    parseBoolFieldOr vdotBounds "guaranteed_negative" (derivativeUpper < 0.0)

  -- Parse verification result from either schema, or derive it from the bounds if omitted.
  let lyapunovVerified ← do
    match j.getObjVal? "verification_result" with
    | .ok result => result.getObjValAs? Bool "lyapunov_verified"
    | .error _ =>
      match j.getObjVal? "verification" with
      | .ok result =>
        parseBoolFieldOr result "lyapunov_verified" (vPositive && derivativeNegative)
      | .error _ => .ok (vPositive && derivativeNegative)

  let cert : CrownCert := {
    method, dynamics, inputDim, inputLo, inputHi,
    vLower, vUpper, gradLo, gradHi, derivativeLower, derivativeUpper,
    vPositive, derivativeNegative, lyapunovVerified
  }
  cert.validate
  return cert

/-- Load and parse a CROWN/Lyapunov certificate from a JSON file. -/
def loadCertificate (path : System.FilePath) : IO CrownCert := do
  let contents ← IO.FS.readFile path
  match Json.parse contents with
  | .error msg => throw <| IO.userError s!"JSON parse error: {msg}"
  | .ok j =>
    match parseCertificate j with
    | .error msg => throw <| IO.userError s!"Certificate parse error: {msg}"
    | .ok cert => return cert

/-!
# Python execution
-/

/-- Run the external Python CROWN producer and write its JSON certificate output. -/
def runPythonCrown (networkPath : String) (inputBox : String)
    (dynamics : String) (outputPath : String) : IO Unit := do
  let args := #[
    "NN/MLTheory/CROWN/Tactics/crown_verifier.py",
    "verify",
    "--model", networkPath,
    "--region", inputBox,
    "--dynamics", dynamics,
    "--output", outputPath,
    "--format", "json"
  ]
  let result ← IO.Process.output {
    cmd := "python3"
    args := args
    cwd := some "."
  }
  if result.exitCode != 0 then
    throw <| IO.userError s!"Python CROWN failed:\n{result.stderr}"

/-!
# Certificate inspection
-/

/-- `inspect_crown_certificate` loads a certificate and reports its producer claims.

Usage:
  inspect_crown_certificate "path/to/cert.json"

The tactic deliberately leaves the current goal unchanged. JSON parsing and producer-reported
booleans are not proofs of the corresponding real-semantic enclosure. -/
syntax (name := inspectCrownCertificate) "inspect_crown_certificate" str : tactic

@[tactic inspectCrownCertificate]
meta def evalInspectCrownCertificate : Tactic := fun stx => do
  match stx with
  | `(tactic| inspect_crown_certificate $pathStx:str) => do
    let path := pathStx.getString
    let cert ← loadCertificate path

    logInfo m!"Loaded CROWN certificate from {path}"
    logInfo m!"  V(x) ∈ [{cert.vLower}, {cert.vUpper}]"
    logInfo m!"  V̇(x) ∈ [{cert.derivativeLower}, {cert.derivativeUpper}]"
    logInfo m!"  producer reports V > 0: {cert.vPositive}"
    logInfo m!"  producer reports V̇ < 0: {cert.derivativeNegative}"
    logInfo m!"  producer reports overall success: {cert.lyapunovVerified}"
    logInfo m!"The current goal was not changed; use a checked certificate theorem to prove it."

  | _ => throwUnsupportedSyntax

/-- `run_crown_producer` runs Python and reports the resulting producer claims.

Usage:
  run_crown_producer "network.json" "[lo,hi]x[lo,hi]" "dynamics" "output.json"

This command does not close the current goal. The resulting JSON still needs a checked semantic
validity proof before it can support a theorem. -/
syntax (name := runCrownProducer) "run_crown_producer" str str str str : tactic

@[tactic runCrownProducer]
meta def evalRunCrownProducer : Tactic := fun stx => do
  match stx with
  | `(tactic| run_crown_producer $netPath:str $inputBox:str $dynamics:str $outPath:str) => do
    let networkPath := netPath.getString
    let box := inputBox.getString
    let dyn := dynamics.getString
    let outputPath := outPath.getString

    logInfo m!"Running the external CROWN producer"
    logInfo m!"  Network: {networkPath}"
    logInfo m!"  Input box: {box}"
    logInfo m!"  Dynamics: {dyn}"

    runPythonCrown networkPath box dyn outputPath

    logInfo m!"  Certificate written to: {outputPath}"

    let cert ← loadCertificate outputPath

    logInfo m!"Loaded the producer output"
    logInfo m!"  V(x) ∈ [{cert.vLower}, {cert.vUpper}]"
    logInfo m!"  V̇(x) ∈ [{cert.derivativeLower}, {cert.derivativeUpper}]"

    logInfo m!"  producer reports overall success: {cert.lyapunovVerified}"
    logInfo m!"The current goal was not changed; prove certificate validity with a checked theorem."

  | _ => throwUnsupportedSyntax

/-- `summarize_crown_certificate` prints the contents of a producer certificate.

Usage:
  summarize_crown_certificate "path/to/cert.json"

This is a diagnostic command. A `[PASS]` label means that the corresponding finite JSON values
satisfy the stated comparison; it is not a proof that the bounds enclose the network. -/
syntax (name := summarizeCrownCertificate) "summarize_crown_certificate" str : tactic

@[tactic summarizeCrownCertificate]
meta def evalSummarizeCrownCertificate : Tactic := fun stx => do
  match stx with
  | `(tactic| summarize_crown_certificate $pathStx:str) => do
    let path := pathStx.getString
    let cert ← loadCertificate path

    logInfo m!"╔══════════════════════════════════════════════════════════════╗"
    logInfo m!"║                 CROWN producer summary                       ║"
    logInfo m!"╠══════════════════════════════════════════════════════════════╣"
    logInfo m!"║ Certificate: {path}"
    logInfo m!"║ Method: {cert.method}"
    logInfo m!"║ Input dimension: {cert.inputDim}"
    logInfo m!"║ Input region: {cert.inputLo} to {cert.inputHi}"
    logInfo m!"╠══════════════════════════════════════════════════════════════╣"
    logInfo m!"║ V(x) bounds:  [{cert.vLower}, {cert.vUpper}]"
    logInfo m!"║ V̇(x) bounds: [{cert.derivativeLower}, {cert.derivativeUpper}]"
    logInfo m!"╠══════════════════════════════════════════════════════════════╣"
    if cert.vPositive then
      logInfo m!"║ [PASS] producer lower bound is positive ({cert.vLower})"
    else
      logInfo m!"║ [FAIL] producer lower bound is not positive ({cert.vLower})"
    if cert.derivativeNegative then
      logInfo m!"║ [PASS] producer derivative upper bound is negative ({cert.derivativeUpper})"
    else
      logInfo m!"║ [FAIL] producer derivative upper bound is not negative ({cert.derivativeUpper})"
    logInfo m!"╠══════════════════════════════════════════════════════════════╣"
    if cert.lyapunovVerified then
      logInfo m!"║         PRODUCER REPORTS BOTH CONDITIONS                     ║"
    else
      logInfo m!"║         PRODUCER DOES NOT REPORT BOTH CONDITIONS             ║"
    logInfo m!"╚══════════════════════════════════════════════════════════════╝"

  | _ => throwUnsupportedSyntax

end

end NN.MLTheory.CROWN.Tactics.CertificateWorkflow

/-!
# External certificate workflow

These tactics are for *workflow support* (loading JSON certificates and showing diagnostics).
To turn a certificate into a Lean theorem, prove that its bounds enclose the specified network on
the specified region, then derive the desired consequence such as the Lyapunov inequalities.

Trust boundary: loading a report does not verify the external tool or its bounds.
-/
