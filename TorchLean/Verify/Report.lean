/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module
public import TorchLean.Verify.CROWN.FlatBox
public meta import TorchLean.Tensor.Einops.Elab.Einsum
public meta import TorchLean.Tensor.Einops.Elab.Pack
public meta import TorchLean.Tensor.Literal
public meta import TorchLean.Tensor.Einops.Elab.Transform
public meta import TorchLean.Tensor.Tactic.Einops.Report
public import TorchLean.Tensor.Reduce.Fold
public import TorchLean.Verify.Tensor


/-!
# Verification requests and reports

The vocabulary of trained-model verification: the bound algorithm, the output property and the
`Report` that `trained.verify center (radius := r)` returns. The input region is the L∞ ball of
radius `r` around `center`, and the report keeps the choices with the bounds they produced.
-/

@[expose] public section

namespace TorchLean

namespace Verification

/-- Bound-propagation algorithm used to construct a verification report. -/
inductive Algorithm where
  /-- Fast interval bound propagation. -/
  | ibp
  /-- CROWN output bounds: forward affine on exact backends, directed backward on rounded ones. -/
  | crown
  /-- Fixed-relaxation alpha-CROWN with the ReLU phases IBP proves, without beta splits or a
  branch-and-bound search, printed as `Alpha-CROWN (IBP phases)`. On a rounded scalar it runs
  directed CROWN, and the report says `CROWN`. -/
  | alphaCROWN
  deriving DecidableEq, Repr

namespace Algorithm

/-- Human-readable method name used in reports. -/
def name : Algorithm → String
  | .ibp => "IBP"
  | .crown => "CROWN"
  | .alphaCROWN => "Alpha-CROWN (IBP phases)"

/-- An algorithm prints as its report name, `Algorithm.name`. -/
instance : ToString Algorithm where
  toString := name

end Algorithm

/--
Output property checked against the computed bounds.

`outputBounds` asks for the enclosure and nothing more. `topLabel label` asks for a certificate
that one class wins everywhere on the input region.
-/
inductive Property where
  /-- Return the output enclosure without an additional assertion. -/
  | outputBounds
  /-- Check that flattened output `label` stays strictly above every other output. -/
  | topLabel (label : Nat)
  deriving DecidableEq, Repr

namespace Property

/-- Human-readable property description used in reports. -/
def name : Property → String
  | .outputBounds => "output bounds"
  | .topLabel label => s!"top label {label}"

/-- A property prints as its description, `Property.name`. -/
instance : ToString Property where
  toString := name

end Property

/-- Reject a radius that is not finite or is negative before verification begins. -/
def validateRadius (radius : Float) : Except String Unit := do
  unless radius.isFinite && 0.0 ≤ radius do
    throw s!"verification radius must be finite and nonnegative, got {radius}"

/-- Reject an input-region center with a NaN or infinite entry before verification begins. -/
def validateCenter {σ : Shape} (center : Tensor Float σ) : Except String Unit := do
  unless Tensor.foldl (fun finite value => finite && value.isFinite) true center do
    throw "verification center must have finite entries"

/-- Result of evaluating a requested output property from valid bounds. -/
inductive Result where
  /-- The report contains an output enclosure but no additional assertion. -/
  | bounds
  /--
  Top-label outcome. `margin` is the lower bound of output `label` minus the largest upper bound
  among the other outputs, and `certified` says whether it is positive. The binary64 difference
  can overflow to infinity even when both endpoints are finite.
  -/
  | topLabel (label : Nat) (margin : Float) (certified : Bool)
  deriving Repr

namespace Result

/-- One-line rendering of a verification result. -/
def summary : Result → String
  | .bounds => "output bounds"
  | .topLabel label margin certified =>
      s!"label={label} margin={margin} certified={certified}"

/-- A result prints as its one-line rendering, `Result.summary`. -/
instance : ToString Result where
  toString := summary

/-- Evaluate `property` against componentwise output bounds, after checking that every endpoint is
finite and that `lo ≤ hi` in every coordinate. -/
def fromBounds (property : Property) (bounds : CROWN.FlatBox Float) :
    Except String Result := do
  unless (List.finRange bounds.dim).all fun index =>
      bounds.lo[index].isFinite && bounds.hi[index].isFinite do
    throw "verification returned a non-finite output bound"
  unless Tensor.boundsOrdered bounds.lo bounds.hi do
    throw "verification returned an output interval with lower bound above upper bound"
  match property with
  | .outputBounds => pure .bounds
  | .topLabel label =>
      let index : Fin bounds.dim ←
        if h : label < bounds.dim then pure ⟨label, h⟩
        else throw s!"class label {label} is out of bounds for {bounds.dim} outputs"
      let competitor ←
        match bounds.maxOther? label with
        | some value => pure value
        | none => throw "top-label verification requires at least two output classes"
      let margin := bounds.lo[index] - competitor
      pure (.topLabel label margin (margin > 0.0))

end Result

/-- Complete report returned by `trained.verify`. -/
structure Report where
  /-- Radius of the checked input region around the supplied center tensor. -/
  radius : Float
  /-- Output property evaluated from the output enclosure. -/
  property : Property
  /-- Bound-propagation algorithm that produced the enclosure. -/
  algorithm : Algorithm
  /-- Output enclosure produced by the selected algorithm. -/
  bounds : CROWN.FlatBox Float
  /-- Chosen property evaluated from the output enclosure. -/
  result : Result
  deriving Repr

namespace Report

/--
Build a report after checking that the radius is finite and nonnegative, that the bounds are
finite with `lo ≤ hi`, and that the property can be evaluated.

That the bounds enclose the model's outputs is the caller's guarantee. These run-time checks look
at the numbers alone and cannot establish it.
-/
def fromBounds (radius : Float) (bounds : CROWN.FlatBox Float)
    (property : Property := .outputBounds)
    (algorithm : Algorithm := .crown) :
    Except String Report := do
  validateRadius radius
  let result ← Result.fromBounds property bounds
  pure { radius, property, algorithm, bounds, result }

/-- One-line human-readable verification summary. -/
def summary (report : Report) : String :=
  let header :=
    s!"{report.algorithm} radius={report.radius} " ++
      s!"lower={reprStr report.bounds.lo} upper={reprStr report.bounds.hi}"
  match report.result with
  | .bounds => header
  | .topLabel .. => s!"{header} {report.result.summary}"

/-- Print the concise verification summary. -/
def printSummary (report : Report) : IO Unit :=
  IO.println report.summary

/-- A report prints as its one-line summary, `Report.summary`. -/
instance : ToString Report where
  toString := summary

end Report

end Verification

end TorchLean
