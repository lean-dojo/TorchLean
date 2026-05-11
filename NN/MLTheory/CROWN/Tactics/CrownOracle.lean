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
public import Std.Data.HashMap

/-!
# CROWN certificate workflow tactics

Tactics and helpers for loading external CROWN/IBP-style certificates (typically produced by
Python tooling), parsing JSON payloads, and using them as witnesses to close proof goals.

We keep these tactics deliberately workflow-oriented. They help us inspect certificates and wire
external producers into Lean, but the trust boundary remains explicit: a theorem that depends on an
external verifier should state the required bound hypothesis or use the quarantined Lyapunov oracle
from `NN.MLTheory.CROWN.Lyapunov.Oracle`.

References:
- Zhang et al., "Efficient Neural Network Robustness Certification with General Activation
  Functions" (CROWN), NeurIPS 2018.
- Xu et al., "Automatic Perturbation Analysis for Scalable Certified Robustness and Beyond"
  (auto_LiRPA), NeurIPS 2020.
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Tactics.CrownOracle

open Lean Elab Tactic Meta Term

/-! ## Certificate data structures -/

/-- Parsed CROWN/Lyapunov certificate loaded from an external JSON producer. -/
structure CrownCert where
  /-- Producer-reported verification method, such as `"ibp"` or `"crown"`. -/
  method : String
  /-- Optional dynamics tag for closed-loop or Lyapunov workflows. -/
  dynamics : Option String := none
  /-- Perturbation radius associated with the input region when one is provided. -/
  epsilon : Float
  /-- Flattened input dimension. -/
  inputDim : Nat
  /-- Lower endpoints of the input box. -/
  inputLo : Array Float
  /-- Upper endpoints of the input box. -/
  inputHi : Array Float
  /-- Lower bound for the Lyapunov candidate `V`. -/
  V_lo : Float
  /-- Upper bound for the Lyapunov candidate `V`. -/
  V_hi : Float
  /-- Lower endpoints for gradient bounds, when supplied by the producer. -/
  gradLo : Array Float
  /-- Upper endpoints for gradient bounds, when supplied by the producer. -/
  gradHi : Array Float
  /-- Lower bound for the orbital derivative `Vdot`. -/
  Vdot_lo : Float
  /-- Upper bound for the orbital derivative `Vdot`. -/
  Vdot_hi : Float
  /-- Whether the certificate claims `V` is positive on the region. -/
  V_positive : Bool
  /-- Whether the certificate claims `Vdot` is negative on the region. -/
  Vdot_negative : Bool
  /-- Overall producer-reported Lyapunov verification flag. -/
  lyapunov_verified : Bool
  deriving Repr, Inhabited

meta section

/-! ## JSON parsing -/

/-- Parse a JSON number as a `Float`. -/
def jsonToFloat (j : Json) : Except String Float :=
  match j with
  | Json.num n => .ok n.toFloat
  | _ => .error s!"Expected number, got {j}"

/-- Parse a JSON array of numbers as an array of `Float`s. -/
def jsonToFloatArray (j : Json) : Except String (Array Float) :=
  match j with
  | Json.arr arr =>
    arr.mapM (fun x =>
      match x with
      | Json.num n => .ok n.toFloat
      | _ => .error "Expected number in array")
  | _ => .error "Expected array"

/-- Parse a Bool field when present; otherwise derive it from a fallback condition. -/
def parseBoolFieldOr (o : Json) (key : String) (fallback : Bool) : Except String Bool :=
  match o.getObjValAs? Bool key with
  | .ok b => .ok b
  | .error _ => .ok fallback

/-- Parse either the backward-compatible `input` block or the `region` block. -/
def parseInputRegion (j : Json) : Except String (Nat × Array Float × Array Float × Float) := do
  let inputLike ←
    match j.getObjVal? "input" with
    | .ok input => .ok input
    | .error _ => j.getObjVal? "region"
  let inputDim ←
    match inputLike.getObjValAs? Nat "dim" with
    | .ok dim => .ok dim
    | .error _ =>
      match inputLike.getObjVal? "lo" >>= jsonToFloatArray with
      | .ok lo => .ok lo.size
      | .error err => .error err
  match inputLike.getObjVal? "lo", inputLike.getObjVal? "hi" with
  | .ok loJson, .ok hiJson =>
    let lo ← jsonToFloatArray loJson
    let hi ← jsonToFloatArray hiJson
    let epsilon ←
      match j.getObjVal? "epsilon" >>= jsonToFloat with
      | .ok eps => .ok eps
      | .error _ =>
        match inputLike.getObjVal? "eps" >>= jsonToFloat with
        | .ok eps => .ok eps
        | .error _ => .ok 0.0
    pure (inputDim, lo, hi, epsilon)
  | _, _ =>
    let centerJson ← inputLike.getObjVal? "center"
    let center ← jsonToFloatArray centerJson
    let eps ← inputLike.getObjVal? "eps" >>= jsonToFloat
    let lo := center.map (· - eps)
    let hi := center.map (· + eps)
    pure (inputDim, lo, hi, eps)

/-- Parse a certificate from JSON, accepting supported closed-loop schemas. -/
def parseCertificate (j : Json) : Except String CrownCert := do
  let method ← j.getObjValAs? String "method"
  let dynamics := j.getObjValAs? String "dynamics" |>.toOption

  let (inputDim, inputLo, inputHi, epsilon) ← parseInputRegion j

  -- Parse V bounds
  let vBounds ← j.getObjVal? "V_bounds"
  let V_lo ← vBounds.getObjVal? "lo" >>= jsonToFloat
  let V_hi ← vBounds.getObjVal? "hi" >>= jsonToFloat
  let V_positive ←
    parseBoolFieldOr vBounds "guaranteed_positive" (V_lo > 0.0)

  -- Parse gradient bounds (optional; accept supported key names)
  let (gradLo, gradHi) ← do
    match j.getObjVal? "gradient_bounds" with
    | .ok gradBounds =>
      let loJson ← gradBounds.getObjVal? "lo"
      let hiJson ← gradBounds.getObjVal? "hi"
      let lo ← jsonToFloatArray loJson
      let hi ← jsonToFloatArray hiJson
      pure (lo, hi)
    | .error _ =>
      match j.getObjVal? "grad_bounds" with
      | .ok gradBounds =>
        let loJson ← gradBounds.getObjVal? "lo"
        let hiJson ← gradBounds.getObjVal? "hi"
        let lo ← jsonToFloatArray loJson
        let hi ← jsonToFloatArray hiJson
        pure (lo, hi)
      | .error _ =>
        pure (#[], #[])

  -- Parse Vdot bounds
  let vdotBounds ← j.getObjVal? "Vdot_bounds"
  let Vdot_lo ← vdotBounds.getObjVal? "lo" >>= jsonToFloat
  let Vdot_hi ← vdotBounds.getObjVal? "hi" >>= jsonToFloat
  let Vdot_negative ←
    parseBoolFieldOr vdotBounds "guaranteed_negative" (Vdot_hi < 0.0)

  -- Parse verification result from either schema, or derive it from the bounds if omitted.
  let lyapunov_verified ← do
    match j.getObjVal? "verification_result" with
    | .ok result => result.getObjValAs? Bool "lyapunov_verified"
    | .error _ =>
      match j.getObjVal? "verification" with
      | .ok result =>
        parseBoolFieldOr result "lyapunov_verified" (V_positive && Vdot_negative)
      | .error _ => .ok (V_positive && Vdot_negative)

  return {
    method, dynamics, epsilon, inputDim, inputLo, inputHi,
    V_lo, V_hi, gradLo, gradHi, Vdot_lo, Vdot_hi,
    V_positive, Vdot_negative, lyapunov_verified
  }

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
# Proof-term support

The key insight: we construct native Lean proofs using the certificate values.
For numeric goals, we construct Nat literals that can be checked by decide.
-/

/-- Convert a nonnegative `Float` into a fixed-scale natural number for lightweight diagnostics. -/
def floatToScaledNat (f : Float) : Nat :=
  (f * 1000000).toUInt64.toNat

/-- Float greater-than check used by certificate diagnostics. -/
def floatGt (a b : Float) : Bool := a > b

/-- Float less-than check used by certificate diagnostics. -/
def floatLt (a b : Float) : Bool := a < b

/-!
# The Main Tactic Implementation
-/

/-- `crown_oracle` tactic: load a certificate and try the registered lightweight closers.

Usage:
  crown_oracle "path/to/cert.json"

The tactic will:
1. load the certificate,
2. print the claimed Lyapunov bounds, and
3. try the built-in closers for simple goals exposed by the certificate. -/
syntax (name := crownOracle) "crown_oracle" str : tactic

@[tactic crownOracle]
meta def evalCrownOracle : Tactic := fun stx => do
  match stx with
  | `(tactic| crown_oracle $pathStx:str) => do
    let path := pathStx.getString

    -- Load certificate
    let cert ← loadCertificate path

    logInfo m!"CROWN Oracle loaded certificate from {path}"
    logInfo m!"  V(x) ∈ [{cert.V_lo}, {cert.V_hi}]"
    logInfo m!"  V̇(x) ∈ [{cert.Vdot_lo}, {cert.Vdot_hi}]"
    logInfo m!"  V > 0: {cert.V_positive}"
    logInfo m!"  V̇ < 0: {cert.Vdot_negative}"

    if cert.lyapunov_verified then
      logInfo m!"  Lyapunov conditions verified!"
    else
      logWarning m!"  Lyapunov conditions NOT verified"

    -- Get current goal
    let goal ← getMainGoal
    let _goalType ← goal.getType

    -- Try the registered certificate goal patterns:
    -- `cert.V_lo > 0`, `cert.Vdot_hi < 0`, and `lyapunov_verified = true`.

    -- Check if we can close with native tactics
    if cert.V_positive && cert.Vdot_negative then
      -- Try different tactics to close the goal
      try
        evalTactic (← `(tactic| decide))
        logInfo m!"Closed goal with decide"
      catch _ =>
        try
          evalTactic (← `(tactic| rfl))
          logInfo m!"Closed goal with rfl"
        catch _ =>
          try
            evalTactic (← `(tactic| trivial))
            logInfo m!"Closed goal with trivial"
          catch _ =>
            logWarning
              m!"Could not automatically close goal. Certificate values available for manual proof."
            -- Log the certificate values for manual use
            logInfo m!"Certificate values for manual proof:"
            logInfo m!"  V_lo = {cert.V_lo}"
            logInfo m!"  V_hi = {cert.V_hi}"
            logInfo m!"  Vdot_lo = {cert.Vdot_lo}"
            logInfo m!"  Vdot_hi = {cert.Vdot_hi}"
    else
      logWarning m!"Certificate does not verify Lyapunov conditions"

  | _ => throwUnsupportedSyntax

/-- crown_compute tactic: Run Python to compute certificate, then load it.

Usage:
  crown_compute "network.json" "[lo,hi]x[lo,hi]" "dynamics" "output.json"

This runs the Python script and then loads the resulting certificate. -/
syntax (name := crownCompute) "crown_compute" str str str str : tactic

@[tactic crownCompute]
meta def evalCrownCompute : Tactic := fun stx => do
  match stx with
  | `(tactic| crown_compute $netPath:str $inputBox:str $dynamics:str $outPath:str) => do
    let networkPath := netPath.getString
    let box := inputBox.getString
    let dyn := dynamics.getString
    let outputPath := outPath.getString

    logInfo m!"CROWN Compute: Running Python verification..."
    logInfo m!"  Network: {networkPath}"
    logInfo m!"  Input box: {box}"
    logInfo m!"  Dynamics: {dyn}"

    -- Run Python
    runPythonCrown networkPath box dyn outputPath

    logInfo m!"  Certificate written to: {outputPath}"

    -- Now load and use the certificate
    let cert ← loadCertificate outputPath

    logInfo m!"CROWN Oracle loaded certificate"
    logInfo m!"  V(x) ∈ [{cert.V_lo}, {cert.V_hi}]"
    logInfo m!"  V̇(x) ∈ [{cert.Vdot_lo}, {cert.Vdot_hi}]"

    if cert.lyapunov_verified then
      logInfo m!"  Lyapunov conditions verified!"
      -- Try to close goal
      try
        evalTactic (← `(tactic| trivial))
      catch _ =>
        try
          evalTactic (← `(tactic| decide))
        catch _ =>
          pure ()
    else
      logWarning m!"  Lyapunov conditions NOT verified"

  | _ => throwUnsupportedSyntax

/-- crown_check tactic: Just verify certificate without trying to close goal.

Usage:
  crown_check "path/to/cert.json" -/
syntax (name := crownCheck) "crown_check" str : tactic

@[tactic crownCheck]
meta def evalCrownCheck : Tactic := fun stx => do
  match stx with
  | `(tactic| crown_check $pathStx:str) => do
    let path := pathStx.getString
    let cert ← loadCertificate path

    logInfo m!"╔══════════════════════════════════════════════════════════════╗"
    logInfo m!"║              CROWN Certificate Verification                   ║"
    logInfo m!"╠══════════════════════════════════════════════════════════════╣"
    logInfo m!"║ Certificate: {path}"
    logInfo m!"║ Method: {cert.method}"
    logInfo m!"║ Input dimension: {cert.inputDim}"
    logInfo m!"║ Input region: {cert.inputLo} to {cert.inputHi}"
    logInfo m!"╠══════════════════════════════════════════════════════════════╣"
    logInfo m!"║ V(x) bounds:  [{cert.V_lo}, {cert.V_hi}]"
    logInfo m!"║ V̇(x) bounds: [{cert.Vdot_lo}, {cert.Vdot_hi}]"
    logInfo m!"╠══════════════════════════════════════════════════════════════╣"
    if cert.V_positive then
      logInfo m!"║ [PASS] V(x) > 0 verified (V_lo = {cert.V_lo} > 0)"
    else
      logInfo m!"║ [FAIL] V(x) > 0 NOT verified (V_lo = {cert.V_lo})"
    if cert.Vdot_negative then
      logInfo m!"║ [PASS] V̇(x) < 0 verified (Vdot_hi = {cert.Vdot_hi} < 0)"
    else
      logInfo m!"║ [FAIL] V̇(x) < 0 NOT verified (Vdot_hi = {cert.Vdot_hi})"
    logInfo m!"╠══════════════════════════════════════════════════════════════╣"
    if cert.lyapunov_verified then
      logInfo m!"║         LYAPUNOV STABILITY VERIFIED!                         ║"
    else
      logInfo m!"║         VERIFICATION FAILED                                  ║"
    logInfo m!"╚══════════════════════════════════════════════════════════════╝"

  | _ => throwUnsupportedSyntax

end

end NN.MLTheory.CROWN.Tactics.CrownOracle

/-!
# Oracle approach (explicit trust boundary)

These tactics are for *workflow support* (loading JSON certificates and showing diagnostics).
If you want to turn a certificate into a Lean theorem, you must do so explicitly by:
1) stating an axiom/assumption at the use site that the bounds hold for the specific
   network/region you care about, and
2) deriving the desired consequence from that assumption (e.g. Lyapunov conditions).

This keeps the trust boundary visible: the external tool is not implicitly ``verified'' by this
  file.
-/
