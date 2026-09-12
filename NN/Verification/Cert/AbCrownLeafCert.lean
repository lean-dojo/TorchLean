/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.Util.Tensor
public import NN.Verification.Util.Json
public import NN.Verification.Util.FloatApprox
public import NN.API.CLI.Parser

/-!
# AbCrown Leaf Artifact

Alpha-beta-CROWN (AbCrown) leaf-artifact checker.

This module checks a small TorchLean JSON schema (`abcrown_leaf_artifact_v0_1`). Vanilla
alpha-beta-CROWN does not emit this TorchLean schema directly; use
`scripts/verification/abcrown/export_leaf_artifact.py` to convert terminal leaf/domain data from an
external verifier into the checked schema.

The checker does **not** run bound propagation itself. It validates only the finite claims present
in the artifact:
- each leaf input box is nested inside the declared root input box, and
- each leaf contains a witness that refutes the unsafe threshold
  ($\mathrm{lb}[i]>\mathrm{threshold}[i]$ for some $i$).

This is useful for:
- regression testing JSON export/import paths, and
- reviewer-friendly validation of the leaf data that TorchLean actually checks.

References:
- beta-CROWN paper (NeurIPS 2021): `https://arxiv.org/abs/2103.06624`
- alpha-beta-CROWN implementation: `https://github.com/Verified-Intelligence/alpha-beta-CROWN`

Run:
`lake exe verify -- abcrown-leaf [path/to/artifact.json]`
-/

@[expose] public section


namespace NN.Verification.Cert.AbCrownLeafCert

open Lean
open Data
open NN.Verification.Json

/-- Bundled sample alpha-beta-CROWN-style leaf artifact. -/
def defaultArtifactPath : String :=
  "NN/Examples/Verification/AbCrown/sample_abcrown_leaf_artifact_v0_1.json"

/--
Parse and validate a `abcrown_leaf_artifact_v0_1` JSON artifact.

Structural problems with the document itself (wrong format tag, missing fields, inconsistent
dimensions) throw `IO.userError` immediately. A leaf that parses but fails one of the three accepted
predicates is counted, and the reason is printed for that leaf before the summary line, because the
three failure modes call for different responses: a leaf outside the root box means the exporter
built the wrong region, a failed prune inequality means the producer's bound does not clear the
threshold, and a stale `witness_margin` means the document disagrees with itself.
-/
def checkAbCrownLeafArtifact (path : String) : IO Unit := do
  let topObj ← readJsonObjectFile path
  expectFormat topObj "abcrown_leaf_artifact_v0_1"
  let inputDim ← expectFieldNat topObj "input_dim" "top-level"

  let rootObj ← expectFieldObject topObj "root" "top-level"
  let root ← fromExcept <| parseEndpointBoxRegion "root" rootObj
  if root.dim ≠ inputDim then
    throw <| IO.userError
      s!"root dimension mismatch: input_dim={inputDim}, endpoints={root.dim}"

  let rootLo ← NN.Verification.Util.Tensor.requireVecOfArray "root.lo" inputDim root.lo
  let rootHi ← NN.Verification.Util.Tensor.requireVecOfArray "root.hi" inputDim root.hi

  let leaves ← expectFieldArray topObj "leaves" "top-level"
  if leaves.isEmpty then
    throw <| IO.userError "invalid leaf artifact: leaves must be nonempty"

  let mut okCount := 0
  let mut badCount := 0
  let mut leafIdx := 0
  for leaf in leaves do
    let leafObj ← expectObject leaf "leaf"
    let region ← fromExcept <| parseEndpointBoxRegion "leaf" leafObj
    let lb ← expectFieldFiniteFloatArray leafObj "lb" "leaf"
    let thr ← expectFieldFiniteFloatArray leafObj "threshold" "leaf"
    if region.dim ≠ inputDim then
      throw <| IO.userError
        s!"leaf dimension mismatch: input_dim={inputDim}, endpoints={region.dim}"
    if lb.size ≠ thr.size then
      throw <| IO.userError
        s!"leaf lower-bound/threshold length mismatch: lb={lb.size}, threshold={thr.size}"

    let lo ← NN.Verification.Util.Tensor.requireVecOfArray "leaf.lo" inputDim region.lo
    let hi ← NN.Verification.Util.Tensor.requireVecOfArray "leaf.hi" inputDim region.hi
    let outputDim := lb.size
    let lb ← NN.Verification.Util.Tensor.requireVecOfArray "leaf.lb" outputDim lb
    let thr ← NN.Verification.Util.Tensor.requireVecOfArray "leaf.threshold" outputDim thr
    let within := NN.Verification.Util.Tensor.boxWithin rootLo rootHi lo hi
    let witnessIdx? ← optionalFieldNat? leafObj "witness_idx" "leaf"
    let witnessMargin? ← optionalFieldFiniteFloat? leafObj "witness_margin" "leaf"
    let verified :=
      match witnessIdx? with
      | some wi => NN.Verification.Util.Tensor.refutesThresholdAt lb thr wi
      | none => NN.Verification.Util.Tensor.refutesThreshold lb thr
    -- The margin this leaf should have reported, when it names a witness index that is in range.
    -- Keeping it as a value rather than folding it into the comparison lets the failure message
    -- quote both numbers, which is the difference between a diagnostic and a verdict.
    let actualMargin? : Option Float :=
      match witnessIdx? with
      | some wi =>
          if h : wi < outputDim then
            some (lb.getScalar ⟨wi, h⟩ - thr.getScalar ⟨wi, h⟩)
          else none
      | none => none
    let marginMatches :=
      match witnessMargin?, actualMargin? with
      | some claimedMargin, some actualMargin =>
          NN.Verification.Util.approxEq actualMargin claimedMargin (tol := 1e-6)
      | some _, none => false
      | none, _ => true
    if within && verified && marginMatches then
      okCount := okCount + 1
    else
      badCount := badCount + 1
      let mut reasons : Array String := #[]
      unless within do
        reasons := reasons.push "box escapes the root region"
      unless verified do
        reasons := reasons.push <|
          match witnessIdx? with
          | some wi => s!"witness index {wi} does not satisfy lb > threshold"
          | none => "no coordinate satisfies lb > threshold"
      unless marginMatches do
        reasons := reasons.push <|
          match witnessMargin?, actualMargin? with
          | some claimed, some actual =>
              s!"witness_margin {claimed} disagrees with lb - threshold = {actual}"
          | some claimed, none =>
              s!"witness_margin {claimed} names no witness index in range"
          | none, _ => "witness margin bookkeeping is inconsistent"
      IO.println
        s!"[artifact] leaf {leafIdx} rejected: {String.intercalate "; " reasons.toList}"
    leafIdx := leafIdx + 1

  IO.println s!"[artifact] Checked {leaves.size} leaves: ok={okCount}, bad={badCount}"
  if badCount > 0 then
    throw <| IO.userError s!"Artifact failed checks for {badCount} leaves"

/--
CLI entry point: `lake exe verify -- abcrown-leaf [artifact.json]`.

If no path is provided, checks a small bundled sample artifact under
`NN/Examples/Verification/AbCrown/`.
-/
def run (args : List String) : IO Unit := do
  let usage :=
    String.intercalate "\n" [
      "Usage:",
      "  lake exe verify -- abcrown-leaf [<path/to/artifact.json>]",
      "",
      "If no path is provided, runs a small bundled sample artifact:",
      s!"  {defaultArtifactPath}"
    ]

  if TorchLean.CLI.hasHelp args then
    IO.println usage
    return

  let args := TorchLean.CLI.dropDashDash args
  let (path, rest) ←
    match TorchLean.CLI.takePositional args (default := defaultArtifactPath) with
    | .ok result => pure result
    | .error e => throw <| IO.userError s!"{e}\n\n{usage}"
  match TorchLean.CLI.checkNoArgs rest with
  | .ok () => pure ()
  | .error e => throw <| IO.userError s!"{e}\n\n{usage}"
  checkAbCrownLeafArtifact path

end NN.Verification.Cert.AbCrownLeafCert
