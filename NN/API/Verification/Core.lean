/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module
public import NN.Tensor


/-!
# Verification

The public vocabulary used by trained-model verification. Ordinary code calls
`trained.verify center (radius := r) (norm := .inf)` and adds named choices only when needed.
`Report` retains those choices with the returned bounds; explicit graph lowering is intentionally
separate in `NN.API.Verification.Lowering`.
-/

@[expose] public section

namespace TorchLean

namespace Verification

/-- Bound-propagation algorithm used to construct a verification report. -/
inductive Algorithm where
  /-- Fast interval bound propagation. -/
  | ibp
  /-- Forward affine CROWN bounds. -/
  | crown
  /--
  Fixed-relaxation Alpha-Beta-CROWN replay. Stable ReLU phases are inferred from IBP; this does
  not run an external branch-and-bound optimizer.
  -/
  | alphaBetaCrown
  deriving DecidableEq, Repr

namespace Algorithm

/-- Human-readable method name used in reports. -/
def name : Algorithm → String
  | .ibp => "IBP"
  | .crown => "CROWN"
  | .alphaBetaCrown => "Alpha-Beta-CROWN"

instance : ToString Algorithm where
  toString := name

end Algorithm

/--
Norm of the input region requested for verification.

The current native CROWN path implements `.inf`; the other cases remain part of the request
language so adding a non-box region does not require another public API.
-/
inductive Norm where
  /-- L1 input ball. -/
  | one
  /-- L2 input ball. -/
  | two
  /-- L-infinity input ball. -/
  | inf
  deriving DecidableEq, Repr

namespace Norm

/-- Human-readable norm name used in reports and diagnostics. -/
def name : Norm → String
  | .one => "L1"
  | .two => "L2"
  | .inf => "Linf"

instance : ToString Norm where
  toString := name

end Norm

/--
Property evaluated from output bounds.

`.bounds` returns the complete output enclosure. `.topLabel label` additionally asks whether one
flattened output stays strictly above every competing output.
-/
inductive Property where
  /-- Return the output enclosure without an additional assertion. -/
  | bounds
  /-- Check that `label` remains the unique largest flattened output. -/
  | topLabel (label : Nat)
  deriving DecidableEq, Repr

namespace Property

/-- Human-readable property description used in reports. -/
def name : Property → String
  | .bounds => "output bounds"
  | .topLabel label => s!"top label {label}"

instance : ToString Property where
  toString := name

end Property

namespace Internal

/-- Reject an invalid input-region radius before verification begins. -/
def validateRadius (radius : Float) : Except String Unit := do
  unless radius.isFinite && 0.0 ≤ radius do
    throw s!"verification radius must be finite and nonnegative, got {radius}"

end Internal

/-- Componentwise lower and upper bounds for a flattened model output. -/
structure Bounds where
  /-- Shared number of flattened output components. -/
  size : Nat
  /-- Componentwise lower output bounds. -/
  lower : Tensor Float [size]
  /-- Componentwise upper output bounds. -/
  upper : Tensor Float [size]
  deriving Repr

namespace Bounds

/-- Check the basic consistency needed by public reporting helpers. -/
def validate (bounds : Bounds) : Except String Unit := do
  unless (List.finRange bounds.size).all fun index =>
      bounds.lower[index].isFinite && bounds.upper[index].isFinite do
    throw "verification returned a non-finite output bound"
  unless (List.finRange bounds.size).all fun index =>
      bounds.lower[index] ≤ bounds.upper[index] do
    throw "verification returned an output interval with lower bound above upper bound"

/-- Largest output value other than `label`. -/
def maxOther? {n : Nat} (values : Tensor Float [n]) (label : Nat) : Option Float :=
  (Tensor.foldl
    (fun (state : Nat × Option Float) value =>
      let best := if state.1 == label then state.2 else
        match state.2 with
        | none => some value
        | some current => some (max current value)
      (state.1 + 1, best))
    (0, none) values).2

end Bounds

/-- Result of evaluating a requested output property from valid bounds. -/
inductive Result where
  /-- The report contains an output enclosure but no additional assertion. -/
  | bounds
  /-- A top-label assertion with its certified lower margin. -/
  | topLabel (label : Nat) (margin : Float) (certified : Bool)
  deriving Repr

namespace Result

/-- One-line rendering of a verification result. -/
def summary : Result → String
  | .bounds => "output bounds"
  | .topLabel label margin certified =>
      s!"label={label} margin={margin} certified={certified}"

instance : ToString Result where
  toString := summary

/-- Evaluate a public property against componentwise output bounds. -/
def fromBounds (property : Property) (bounds : Bounds) : Except String Result := do
  bounds.validate
  match property with
  | .bounds => pure .bounds
  | .topLabel label =>
      let index : Fin bounds.size ←
        if h : label < bounds.size then pure ⟨label, h⟩
        else throw s!"class label {label} is out of bounds for {bounds.size} outputs"
      let competitor ←
        match Bounds.maxOther? bounds.upper label with
        | some value => pure value
        | none => throw "top-label verification requires at least two output classes"
      let margin := bounds.lower[index] - competitor
      pure (.topLabel label margin (margin > 0.0))

end Result

/-- Complete report returned by `trained.verify`. -/
structure Report where
  /-- Radius of the checked input region around the supplied center tensor. -/
  radius : Float
  /-- Norm used to define the checked input region. -/
  norm : Norm
  /-- Output property evaluated from the output enclosure. -/
  property : Property
  /-- Bound-propagation algorithm that produced the enclosure. -/
  algorithm : Algorithm
  /-- Output enclosure produced by the selected algorithm. -/
  bounds : Bounds
  /-- Chosen property evaluated from the output enclosure. -/
  result : Result
  deriving Repr

namespace Report

/-- Build a report from verified output bounds and named verification choices. -/
def fromBounds (radius : Float) (bounds : Bounds)
    (norm : Norm := .inf)
    (property : Property := .bounds)
    (algorithm : Algorithm := .alphaBetaCrown) :
    Except String Report := do
  Internal.validateRadius radius
  let result ← Result.fromBounds property bounds
  pure { radius, norm, property, algorithm, bounds, result }

/-- One-line human-readable verification summary. -/
def summary (report : Report) : String :=
  let header :=
    s!"{report.algorithm} norm={report.norm} radius={report.radius} " ++
      s!"lower={reprStr report.bounds.lower} upper={reprStr report.bounds.upper}"
  match report.result with
  | .bounds => header
  | .topLabel label margin certified =>
      header ++ s!" label={label} margin={margin} certified={certified}"

/-- Print the concise verification summary. -/
def printSummary (report : Report) : IO Unit :=
  IO.println report.summary

instance : ToString Report where
  toString := summary

end Report

end Verification


end TorchLean
