/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.Tensor
public import TorchLean.Verify.Certificate.DecimalJSON
public import TorchLean.Verify.JSON
public import TorchLean.Verify.Numeric.ApproxEq
public import TorchLean.Tensor.Context.Rational
public import TorchLean.Verify.Certificate.Report

/-!
# AbCrown Leaf Artifact

A checker for alpha-beta-CROWN (AbCrown) leaf artifacts in a small TorchLean JSON schema,
`abcrown_leaf_artifact_v0_1`. The upstream alpha-beta-CROWN tool does not emit this schema, so
`scripts/verification/abcrown/export_leaf_artifact.py` converts the terminal leaf and domain data
of an external verifier into it.

The checker runs no bound propagation. It validates only the finite claims present in the
artifact:
- each leaf input box is nested inside the declared root input box,
- the leaf boxes together cover the root box, and
- each leaf contains a witness that refutes the unsafe threshold
  ($\mathrm{lb}[i]>\mathrm{threshold}[i]$ for some $i$).

Nesting, cover and the prune test compare the exact rationals that the JSON decimals denote. A leaf
ending at `1.0000000000000001` therefore escapes a root ending at `1`, although both decimals round
to the binary64 `1`. A `witness_margin` must lie within `1 / 1000000` of `lb - threshold` at its
witness index.

The lower bounds `lb` are the producer's claims and are not recomputed, so a passing artifact is
consistent, not verified. The report says so.

A leaf is an input box with its claimed `lb` and `threshold`, and it carries none of beta-CROWN's
per-neuron split constraints, so the cover check is about splits of the input domain. Schema and
checks are both hand-written.

This is useful for:
- regression testing JSON export/import paths, and
- reviewer-friendly validation of the leaf data that TorchLean actually checks.

References:
- beta-CROWN: Wang, Zhang, Xu, Lin, Jana, Hsieh and Kolter, "Beta-CROWN: Efficient Bound
  Propagation with Per-neuron Split Constraints for Complete and Incomplete Neural Network
  Robustness Verification", NeurIPS 2021, arXiv:2103.06624: `https://arxiv.org/abs/2103.06624`
- alpha-beta-CROWN implementation: `https://github.com/Verified-Intelligence/alpha-beta-CROWN`

Run:
`lake exe verify -- abcrown-leaf [path/to/artifact.json]`
-/

@[expose] public section


namespace TorchLean.Certificate.AbCROWNLeaf

open Lean
open TorchLean.Json BoxRegion
open Numeric (approxEq)
open Tensor (requireArray boxWithin)
open Tensor (refutesThreshold refutesThresholdAt)
-- The box, threshold and margin checks compare through the scoped exact `Context ℚ`.
open scoped RationalAlgebraic

/--
Check that the closed leaf boxes cover the closed root box, assuming every leaf is inside the root.

Along each coordinate, the root and leaf endpoints cut the root into a grid of closed cells. Every
leaf is a union of cells, so the leaves cover the root exactly when each cell lies inside one leaf.
Coordinates on which every leaf spans the whole root are skipped, and a grid of more than
`maxCells` cells is refused with an error.
-/
def leavesCoverRoot (rootLo rootHi : Array Rat) (leaves : Array (Array Rat × Array Rat))
    (maxCells : Nat := 1000000) : Except String Bool := do
  let dim := rootLo.size
  -- For each split coordinate: its index and the closed intervals between breakpoints.
  let mut axes : Array (Nat × Array (Rat × Rat)) := #[]
  let mut cells := 1
  for d in [:dim] do
    if leaves.all fun leaf => leaf.1[d]! == rootLo[d]! && leaf.2[d]! == rootHi[d]! then
      continue
    -- The sorted distinct endpoints of the root and the leaves along coordinate `d`.
    let pts := (leaves.foldl (init := #[rootLo[d]!, rootHi[d]!]) fun acc leaf =>
      (acc.push leaf.1[d]!).push leaf.2[d]!).qsort (· < ·) |>.eraseReps
    let intervals :=
      if pts.size = 1 then #[(pts[0]!, pts[0]!)]
      else (Array.range (pts.size - 1)).map fun j => (pts[j]!, pts[j + 1]!)
    cells := cells * intervals.size
    if cells > maxCells then
      throw s!"coverage grid exceeds {maxCells} cells; split the artifact or check it elsewhere"
    axes := axes.push (d, intervals)
  let inside (cell : Array (Rat × Rat)) (leaf : Array Rat × Array Rat) : Bool :=
    (List.range axes.size).all fun k =>
      let d := axes[k]!.1
      leaf.1[d]! ≤ cell[k]!.1 && cell[k]!.2 ≤ leaf.2[d]!
  for index in [:cells] do
    let mut rest := index
    let mut cell : Array (Rat × Rat) := #[]
    for (_, intervals) in axes do
      cell := cell.push intervals[rest % intervals.size]!
      rest := rest / intervals.size
    unless leaves.any (inside cell) do
      return false
  return true

/--
Parse and validate an `abcrown_leaf_artifact_v0_1` JSON artifact.

The report accepts the artifact iff every leaf lies in the root box and clears its threshold, every
`witness_margin` is within `1 / 1000000` of `lb - threshold` at the leaf's `witness_idx`, and
`leavesCoverRoot` finds that the leaves cover the root box. Structural problems with the document
itself (wrong format tag, missing fields, inconsistent dimensions, no leaves) throw `IO.userError`
immediately. A leaf that parses but fails one of the three leaf predicates is counted, and the
report gives its reasons in a line before the summary line. The three failure modes call for
different responses. A leaf outside the root box means the exporter built the wrong region, a failed
prune inequality means the producer's bound does not clear the threshold, and a stale
`witness_margin` means the document disagrees with itself.
-/
def check (path : String) : IO Report := do
  let top ← DecimalJson.readObjectFile path
  IO.ofExcept <| top.field "format" "top-level" >>= (·.string "top-level.format") >>=
    (expectFormat · "abcrown_leaf_artifact_v0_1")
  let inputDim ← IO.ofExcept <|
    top.field "input_dim" "top-level" >>= (·.nat "top-level.input_dim")

  let root ← IO.ofExcept <| top.field "root" "top-level" >>= parseEndpoints "root"
  if root.dim ≠ inputDim then
    throw <| IO.userError
      s!"root dimension mismatch: input_dim={inputDim}, endpoints={root.dim}"

  let rootLo ← requireArray "root.lo" [inputDim] root.lo
  let rootHi ← requireArray "root.hi" [inputDim] root.hi

  let leaves ← IO.ofExcept <| top.field "leaves" "top-level" >>= (·.array "top-level.leaves")
  if leaves.isEmpty then
    throw <| IO.userError "invalid leaf artifact: leaves must be nonempty"

  let mut lines : Array String := #[]
  let mut okCount := 0
  let mut badCount := 0
  let mut leafIdx := 0
  let mut boxes : Array (Array Rat × Array Rat) := #[]
  for leaf in leaves do
    let region ← IO.ofExcept <| parseEndpoints "leaf" leaf
    let lb ← IO.ofExcept <| leaf.field "lb" "leaf" >>= parseDecimalRatArray "leaf.lb"
    let thr ← IO.ofExcept <|
      leaf.field "threshold" "leaf" >>= parseDecimalRatArray "leaf.threshold"
    if region.dim ≠ inputDim then
      throw <| IO.userError
        s!"leaf dimension mismatch: input_dim={inputDim}, endpoints={region.dim}"
    if lb.size ≠ thr.size then
      throw <| IO.userError
        s!"leaf lower-bound/threshold length mismatch: lb={lb.size}, threshold={thr.size}"

    boxes := boxes.push (region.lo, region.hi)
    let lo ← requireArray "leaf.lo" [inputDim] region.lo
    let hi ← requireArray "leaf.hi" [inputDim] region.hi
    let outputDim := lb.size
    let lb ← requireArray "leaf.lb" [outputDim] lb
    let thr ← requireArray "leaf.threshold" [outputDim] thr
    let within := boxWithin rootLo rootHi lo hi
    let witnessIdx? ← IO.ofExcept <|
      leaf.optionalField? "witness_idx" "leaf" >>= (·.mapM (·.nat "leaf.witness_idx"))
    let witnessMargin? ← IO.ofExcept <| leaf.optionalField? "witness_margin" "leaf" >>=
      (·.mapM (parseDecimalRat "leaf.witness_margin"))
    let verified :=
      match witnessIdx? with
      | some wi => refutesThresholdAt lb thr wi
      | none => refutesThreshold lb thr
    -- Retain the computed margin so a mismatch can report both values.
    let actualMargin? : Option Rat :=
      match witnessIdx? with
      | some wi =>
          if h : wi < outputDim then
            some (lb.getScalar ⟨wi, h⟩ - thr.getScalar ⟨wi, h⟩)
          else none
      | none => none
    let marginMatches :=
      match witnessMargin?, actualMargin? with
      | some claimedMargin, some actualMargin =>
          approxEq actualMargin claimedMargin (tol := 1 / 1000000)
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
      lines := lines.push
        s!"[artifact] leaf {leafIdx} rejected: {String.intercalate "; " reasons.toList}"
    leafIdx := leafIdx + 1

  lines := lines.push s!"[artifact] Checked {leaves.size} leaves: ok={okCount}, bad={badCount}"
  if badCount > 0 then
    return { lines, rejection := some s!"Artifact failed checks for {badCount} leaves" }
  match leavesCoverRoot root.lo root.hi boxes with
  | .error e => return { lines, rejection := some e }
  | .ok false =>
    return { lines, rejection := some "Artifact failed: the leaf boxes do not cover the root box" }
  | .ok true =>
    return { lines := lines ++ #[
      "[artifact] consistent: the leaves cover the root and every leaf clears its threshold.",
      "[artifact] The lower bounds are the producer's claims; " ++
        "TorchLean did not recompute them."] }

end TorchLean.Certificate.AbCROWNLeaf
