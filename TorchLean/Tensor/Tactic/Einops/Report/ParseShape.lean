/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public meta import TorchLean.Tensor.Tactic.Einops.Report.Decode
public import TorchLean.Tensor.Tactic.Einops.Report.Decode

/-!
# Parse-shape reports

The `einops?` report for a concrete `parse_shape` check, with the ordered axis bindings that the
checker certifies.
-/

public meta section

namespace TorchLean.Einops.Report

open Lean Lean.Elab Tactic Meta

/-- Decode one reflected `(axis name, length)` parse-shape binding. -/
def decodeNameLength (expression : Expr) :
    MetaM (Option (String × Nat)) := do
  let expression ← whnfAll expression
  let some fields :=
      constructorFields? expression ``Prod.mk 4
    | return none
  let some name ← decodeStringLiteral fields[2]!
    | return none
  let some length ← decodeNatLiteral fields[3]!
    | return none
  return some (name, length)

/-- Describe one source axis that parse-shape ellipsis expansion keeps: named, wildcard, literal,
unit or an unexpanded ellipsis. -/
def decodeLocatedParseShapeAxis (expression : Expr) :
    MetaM (Option String) := do
  let expression ← whnfAll expression
  let some fields :=
      constructorFields? expression ``Einops.Located.mk 3
    | return none
  let axis ← whnfAll fields[1]!
  if axis.isAppOfArity ``Einops.Axis.named 1 then
    let some name ← decodeStringLiteral axis.getAppArgs[0]!
      | return none
    if name = "_" then
      return some "wildcard `_`"
    else
      return some s!"named axis `{name}`"
  else if axis.isAppOfArity ``Einops.Axis.anonymous 2 then
    let some value ← decodeNatLiteral axis.getAppArgs[0]!
      | return none
    return some s!"literal axis `{value}`"
  else if axis.isAppOfArity ``Einops.Axis.unit 0 then
    return some "unit axis `1`"
  else if axis.isAppOfArity ``Einops.Axis.ellipsis 0 then
    return some "unexpanded ellipsis"
  else
    return none

/-- Describe one parse-shape position after ellipsis expansion. -/
def decodeExpandedParseShapeAxis (expression : Expr) :
    MetaM (Option String) := do
  let expression ← whnfAll expression
  if expression.isAppOfArity ``Option.none 1 then
    return some "ellipsis-expanded wildcard"
  let some fields :=
      constructorFields? expression ``Option.some 2
    | return none
  let semanticAxesExpression ←
    mkAppM ``Einops.CompositeAxis.semanticAxes #[fields[1]!]
  let some semanticAxes ←
      decodeExprList semanticAxesExpression decodeLocatedParseShapeAxis
    | return none
  match semanticAxes with
  | [] => return some "unit axis `1`"
  | [axis] => return some axis
  | _ => return some "unsupported composite axis"

/-- Render parse-shape bindings in the public result order. -/
def formatBindings (bindings : List (String × Nat)) : String :=
  "[" ++ String.intercalate ", "
    (bindings.map fun binding => s!"(\"{binding.1}\", {binding.2})") ++ "]"

/--
Report a concrete structural parse-shape check whose checker reduces to `Except.ok`, and return
`none` otherwise. No scalar type is passed because the operation inspects static shape metadata
only.
-/
def concreteParseShape
    (pattern inputShapeExpression : Expr) : MetaM (Option String) := do
  let some inputShape ← decodeNatList inputShapeExpression
    | return none
  let pattern ← whnfAll pattern
  let some patternFields :=
      constructorFields? pattern ``Einops.Expression.mk 2
    | return none
  let sourceAxisCountExpression ←
    mkAppM ``List.length #[patternFields[0]!]
  let some sourceAxisCount ← decodeNatLiteral sourceAxisCountExpression
    | return none
  let ellipsisCountExpression ←
    mkAppM ``Einops.Expression.ellipsisCount #[pattern]
  let some ellipsisCount ← decodeNatLiteral ellipsisCountExpression
    | return none
  let expandedAxesExpression ←
    mkAppM ``ParseShape.expandAxes #[
      pattern, Lean.toExpr inputShape.length]
  let some expandedAxes ←
      decodeExprList expandedAxesExpression decodeExpandedParseShapeAxis
    | return none
  let checkedResult ←
    mkAppM ``ParseShape.check #[pattern, inputShapeExpression]
  let checkedResult ← whnfAll checkedResult
  let some resultFields :=
      constructorFields? checkedResult ``Except.ok 3
    | return none
  let some bindings ←
      decodeExprList resultFields[2]! decodeNameLength
    | return none
  let inputShapeType ← inferredType inputShapeExpression
  let checkerResultType ← inferredType checkedResult
  let publicResultType ← formatType (mkConst ``SupplementaryLengths)
  let typeEntries := [
    ("Structural shape metadata", inputShapeType),
    ("Checker result", checkerResultType),
    ("Public return metadata", publicResultType),
    ("Scalar type", "arbitrary and erased before this metadata check")]
  let positionChecks :=
    (expandedAxes.zip inputShape).mapIdx fun index (axis, dimension) =>
      s!"Position {index}: {axis} matches physical dimension {dimension}."
  let ellipsisRank :=
    if ellipsisCount = 0 then 0
    else inputShape.length - (sourceAxisCount - 1)
  let obligations :=
    ["Grammar: composite axes are absent, at most one ellipsis occurs, and \
        an ellipsis is not parenthesized.",
     s!"Rank: {sourceAxisCount} source positions with {ellipsisCount} \
        ellipsis expand by {ellipsisRank} positions to physical rank \
        {inputShape.length}."] ++
      positionChecks ++
    [s!"Ordered bindings: {formatBindings bindings}.",
     "Returned names are duplicate-free; wildcard, literal, unit, and \
        ellipsis-expanded positions produce no binding.",
     "Every returned length is the exact physical dimension paired with its \
        named pattern position."]
  let logicalStages :=
    ["Parse the metadata pattern and expand its optional ellipsis to the \
        structural tensor rank.",
     "Validate every expanded position against its physical dimension, \
        including literal and unit-length requirements.",
     "Filter named non-wildcard positions in source order to produce the \
        public metadata list."]
  let workEstimate :=
    [s!"Structural positions checked: {inputShape.length}.",
     s!"Returned metadata bindings: {bindings.length}.",
     "Scalar reads: 0.",
     "Output buffers: 0.",
     "Only the returned metadata list is constructed."]
  return some <| String.intercalate "\n" <|
    ["parse_shape"] ++
      typeCheckSection typeEntries ++
      [s!"  Structural witness shape: {inputShape}",
       s!"  Expanded positions: \
          {String.intercalate ", " expandedAxes}"] ++
      formatSection "Discharged obligations" obligations ++
      ["  Verified logical stages:"] ++
      numberedStageLines 1 logicalStages ++
      formatSection "Generated execution strategy"
        ["metadata-only construction; tensor elements are never inspected."] ++
      formatSection "Shape-derived work estimate" workEstimate ++
      formatSection "Correctness"
        [s!"{Lowering.reportName ``ParseShape.check_axes_match} certifies lockstep \
            dimension agreement.",
         s!"{Lowering.reportName ``ParseShape.check_ordered_bindings} certifies source \
            order.",
         s!"{Lowering.reportName ``ParseShape.check_names_nodup} certifies unique \
            returned names.",
         s!"{Lowering.reportName ``ParseShape.check_dimension_agreement} certifies every \
            returned name-length pair."] ++
      formatSection "Performance note"
        ["parse_shape has no tensor data kernel; its runtime result contains \
            shape metadata only.",
         "`einops?` inspects certificates and never evaluates user terms."]

end TorchLean.Einops.Report
