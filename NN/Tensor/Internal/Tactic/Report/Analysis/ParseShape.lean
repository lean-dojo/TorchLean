/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public meta import NN.Tensor.Internal.Tactic.Report.Analysis.Common
public import NN.Tensor.Internal.Tactic.Report.Analysis.Common
import Mathlib.Algebra.Order.Field.Basic

/-!
# Parse-shape reports

This module reports the ordered axis bindings certified by `parse_shape`.
-/

public meta section

namespace TorchLean.Tensor.Internal.Report.Impl

open Lean Elab Tactic Meta

/-- Decode one reflected `(axis name, length)` parse-shape binding. -/
def decodeNameLength (expression : Expr) :
    MetaM (Option (String × Nat)) := do
  let expression ← reportWhnf expression
  let some fields :=
      constructorFields? expression ``Prod.mk 4
    | return none
  let some name ← decodeStringLiteral fields[2]!
    | return none
  let some length ← decodeNatLiteral fields[3]!
    | return none
  return some (name, length)

/-- Decode one source axis retained by parse-shape ellipsis expansion. -/
def decodeLocatedParseShapeAxis (expression : Expr) :
    MetaM (Option String) := do
  let expression ← reportWhnf expression
  let some fields :=
      constructorFields? expression ``TorchLean.Tensor.Internal.Syntax.Located.mk 3
    | return none
  let axis ← reportWhnf fields[1]!
  if axis.isAppOfArity ``TorchLean.Tensor.Internal.Syntax.Axis.named 1 then
    let some name ← decodeStringLiteral axis.getAppArgs[0]!
      | return none
    if name = "_" then
      return some "wildcard `_`"
    else
      return some s!"named axis `{name}`"
  else if axis.isAppOfArity ``TorchLean.Tensor.Internal.Syntax.Axis.anonymous 2 then
    let some value ← decodeNatLiteral axis.getAppArgs[0]!
      | return none
    return some s!"literal axis `{value}`"
  else if axis.isAppOfArity ``TorchLean.Tensor.Internal.Syntax.Axis.unit 0 then
    return some "unit axis `1`"
  else if axis.isAppOfArity ``TorchLean.Tensor.Internal.Syntax.Axis.ellipsis 0 then
    return some "unexpanded ellipsis"
  else
    return none

/-- Describe one parse-shape position after ellipsis expansion. -/
def decodeExpandedParseShapeAxis (expression : Expr) :
    MetaM (Option String) := do
  let expression ← reportWhnf expression
  if expression.isAppOfArity ``Option.none 1 then
    return some "ellipsis-expanded wildcard"
  let some fields :=
      constructorFields? expression ``Option.some 2
    | return none
  let semanticAxesExpression ←
    mkAppM ``TorchLean.Tensor.Internal.Syntax.CompositeAxis.semanticAxes #[fields[1]!]
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
Decode the structural parse-shape check. The tensor scalar term is absent
because this operation inspects only static shape metadata.
-/
def concreteParseShapeReport
    (pattern inputShapeExpression : Expr) : MetaM (Option String) := do
  let some inputShape ← decodeNatList inputShapeExpression
    | return none
  let pattern ← reportWhnf pattern
  let some patternFields :=
      constructorFields? pattern ``TorchLean.Tensor.Internal.Syntax.Expression.mk 2
    | return none
  let sourceAxisCountExpression ←
    mkAppM ``List.length #[patternFields[0]!]
  let some sourceAxisCount ← decodeNatLiteral sourceAxisCountExpression
    | return none
  let ellipsisCountExpression ←
    mkAppM ``TorchLean.Tensor.Internal.Syntax.Expression.ellipsisCount #[pattern]
  let some ellipsisCount ← decodeNatLiteral ellipsisCountExpression
    | return none
  let expandedAxesExpression ←
    mkAppM ``Check.ParseShape.expandAxes #[
      pattern, Lean.toExpr inputShape.length]
  let some expandedAxes ←
      decodeExprList expandedAxesExpression decodeExpandedParseShapeAxis
    | return none
  let checkedResult ←
    mkAppM ``Check.checkParseShape #[pattern, inputShapeExpression]
  let checkedResult ← reportWhnf checkedResult
  let some resultFields :=
      constructorFields? checkedResult ``Except.ok 3
    | return none
  let some bindings ←
      decodeExprList resultFields[2]! decodeNameLength
    | return none
  let inputShapeType ← inferredType inputShapeExpression
  let checkerResultType ← inferredType checkedResult
  let publicResultType ← formatType (mkConst ``Check.SupplementaryLengths)
  let typeEntries := [
    ("Structural shape metadata", inputShapeType),
    ("Checker result", checkerResultType),
    ("Public return metadata", publicResultType),
    ("Rep scalar type", "arbitrary and erased before this metadata check")]
  let positionChecks :=
    ((List.range inputShape.length).zip
      (expandedAxes.zip inputShape)).map fun position =>
        s!"Position {position.1}: {position.2.1} matches physical \
          dimension {position.2.2}."
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
     "Rep scalar reads: 0.",
     "Rep output buffers: 0.",
     "Only the returned metadata list is constructed."]
  return some <| String.intercalate "\n" <|
    ["parse_shape"] ++
      typeCheckSection typeEntries ++
      [s!"  Structural witness shape: {formatShape inputShape}",
       s!"  Expanded positions: \
          {String.intercalate ", " expandedAxes}"] ++
      reportSection "Discharged obligations" obligations ++
      ["  Verified logical stages:"] ++
      numberedStageLines 1 logicalStages ++
      reportSection "Generated execution strategy"
        ["metadata-only construction; tensor elements are never inspected."] ++
      reportSection "Shape-derived work estimate" workEstimate ++
      reportSection "Correctness"
        ["Check.checkParseShape_axes_match certifies lockstep dimension \
            agreement.",
         "Check.checkParseShape_ordered_bindings certifies source order.",
         "Check.checkParseShape_names_nodup certifies unique returned names.",
         "Check.checkParseShape_dimension_agreement certifies every returned \
            name-length pair."] ++
      reportSection "Performance note"
        ["parse_shape has no tensor data kernel; its runtime result contains \
            shape metadata only.",
         "`einops?` inspects certificates and never evaluates user terms."]

end TorchLean.Tensor.Internal.Report.Impl
