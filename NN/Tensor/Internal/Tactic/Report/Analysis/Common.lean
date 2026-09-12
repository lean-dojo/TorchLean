/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public meta import NN.Tensor.Internal.Tactic.Proof
public meta import NN.Tensor.Internal.Elab.Einsum.ParallelOutput -- shake: keep
public import NN.Tensor.Internal.Check.ParseShape -- shake: keep

/-!
# Shared report decoding

This module provides the reflection, formatting, and axis-analysis utilities
used by the operation-specific `einops?` reports.
-/

public meta section

namespace TorchLean.Tensor.Internal.Report.Impl

open Lean Elab Tactic Meta

/-!
`instantiateOuterLets` is not defined here. It is `Tensor.Internal.instantiateOuterLets` from
`NN/Tensor/Internal/Tactic/Proof.lean`, which this module already imports, and which had the
identical four-line body.
`Report.Impl` sits inside `Tensor.Internal`, so the uses below find it by walking outward.
-/

/-- Normalize reflected report data after instantiating pending metavariables. -/
def reportWhnf (expression : Expr) : MetaM Expr := do
  withTransparency .all <| whnf (← instantiateMVars expression)

/--
Expose the fields of an expected reflected constructor after removing
generated outer lets and metadata.
-/
def constructorFields? (expression : Expr)
    (constructorName : Name) (fieldCount : Nat) : Option (Array Expr) :=
  let expression := (instantiateOuterLets expression).consumeMData
  if expression.isAppOfArity constructorName fieldCount then
    some expression.getAppArgs
  else
    none

/-- Decode a reflected string only when kernel normalization exposes a literal. -/
def decodeStringLiteral (expression : Expr) :
    MetaM (Option String) := do
  match ← reportWhnf expression with
  | .lit (.strVal value) => return some value
  | _ => return none

/-- Decode a reflected natural number only when it reduces to a literal. -/
def decodeNatLiteral (expression : Expr) : MetaM (Option Nat) := do
  getNatValue? (← reportWhnf expression)

/-- Decode a reflected list with an element decoder, failing atomically. -/
partial def decodeExprList {α : Type}
    (expression : Expr) (decodeItem : Expr → MetaM (Option α)) :
    MetaM (Option (List α)) := do
  let expression ← reportWhnf expression
  if expression.isAppOfArity ``List.nil 1 then
    return some []
  else if expression.isAppOfArity ``List.cons 3 then
    let arguments := expression.getAppArgs
    let some item ← decodeItem arguments[1]!
      | return none
    let some remaining ← decodeExprList arguments[2]! decodeItem
      | return none
    return some (item :: remaining)
  else
    return none

/-- Decode a fully concrete reflected shape or list of natural numbers. -/
def decodeNatList (expression : Expr) :
    MetaM (Option (List Nat)) :=
  decodeExprList expression decodeNatLiteral

/-- Decode a transformation axis together with its user-facing description. -/
def decodeTransformAxis (expression : Expr) :
    MetaM (Option (Expr × String)) := do
  let expression ← reportWhnf expression
  if expression.isAppOfArity ``Check.AxisId.named 1 then
    let some name ← decodeStringLiteral expression.getAppArgs[0]!
      | return none
    return some (expression, name)
  else if expression.isAppOfArity ``Check.AxisId.anonymous 2 then
    let some value ← decodeNatLiteral expression.getAppArgs[0]!
      | return none
    return some (expression, s!"literal {value}")
  else if expression.isAppOfArity ``Check.AxisId.ellipsis 1 then
    let some index ← decodeNatLiteral expression.getAppArgs[0]!
      | return none
    return some (expression, s!"ellipsis[{index}]")
  else
    return none

/-- Decode an einsum axis together with its user-facing description. -/
def decodeEinsumAxis (expression : Expr) :
    MetaM (Option (Expr × String)) := do
  let expression ← reportWhnf expression
  if expression.isAppOfArity ``Check.EinsumAxis.named 1 then
    let some name ← decodeStringLiteral expression.getAppArgs[0]!
      | return none
    return some (expression, name)
  else if expression.isAppOfArity ``Check.EinsumAxis.ellipsis 1 then
    let some index ← decodeNatLiteral expression.getAppArgs[0]!
      | return none
    return some (expression, s!"ellipsis[{index}]")
  else
    return none

/-- Decode a reflected einsum axis to its concrete checker value. -/
def decodeEinsumAxisValue (expression : Expr) :
    MetaM (Option Check.EinsumAxis) := do
  let expression ← reportWhnf expression
  if expression.isAppOfArity ``Check.EinsumAxis.named 1 then
    let some name ← decodeStringLiteral expression.getAppArgs[0]!
      | return none
    return some (.named name)
  else if expression.isAppOfArity ``Check.EinsumAxis.ellipsis 1 then
    let some index ← decodeNatLiteral expression.getAppArgs[0]!
      | return none
    return some (.ellipsis index)
  else
    return none

/-- Render one concrete shape in tensor notation. -/
def formatShape (shape : List Nat) : String :=
  "[" ++ String.intercalate ", " (shape.map toString) ++ "]"

/-- Render a heterogeneous family of concrete shapes. -/
def formatShapes (shapes : List (List Nat)) : String :=
  "[" ++ String.intercalate ", " (shapes.map formatShape) ++ "]"

/-- Compute the number of scalar entries described by a concrete shape. -/
def shapeSize : List Nat → Nat
  | [] => 1
  | dimension :: shape => dimension * shapeSize shape

/-- Pretty-print an elaborated type using the current local context. -/
def formatType (typeExpression : Expr) : MetaM String := do
  return (← ppExpr (← instantiateMVars typeExpression)).pretty

/-- Infer and pretty-print the type of an elaborated expression. -/
def inferredType (expression : Expr) : MetaM String := do
  formatType (← inferType expression)

/-- Test whether a reflected term contains an application of a named constant. -/
def containsConstant (expression : Expr) (name : Name) : Bool :=
  (expression.find? fun subterm =>
    match subterm.getAppFn with
    | .const candidate _ => candidate == name
    | _ => false).isSome

/-- Pretty-print one concrete tensor type without evaluating a tensor value. -/
def concreteTensorType (scalarType : Expr)
    (shape : List Nat) : MetaM String := do
  formatType (← mkAppM ``Rep #[scalarType, Lean.toExpr shape])

/-- Format one uniformly indented report section. -/
def reportSection (title : String) (lines : List String) :
    List String :=
  s!"  {title}:" :: lines.map fun line => s!"    {line}"

/-- Format checked input and output types. -/
def typeCheckSection
    (entries : List (String × String)) : List String :=
  reportSection "Type checks" <|
    entries.map fun entry => s!"{entry.1}: {entry.2}"

/-- Explain how static estimates relate to actual runtime measurements. -/
def performanceFooter : List String :=
  reportSection "Performance note"
    ["Work estimates use only static shapes; `einops?` never executes or \
        times user terms.",
     "Native benchmark:",
     "  lake build performanceComparison",
     "  ./.lake/build/bin/performanceComparison field 0"]

/-- Render a flat logical-axis list. -/
def formatAxes (axes : List (Expr × String)) : String :=
  "[" ++ String.intercalate ", " (axes.map Prod.snd) ++ "]"

/-- Render one elementary-axis group using einops parentheses. -/
def formatAxisGroup (group : List (Expr × String)) : String :=
  match group with
  | [] => "()"
  | [axis] => axis.2
  | _ => "(" ++ String.intercalate " " (group.map Prod.snd) ++ ")"

/-- Render a physical shape's complete sequence of logical-axis groups. -/
def formatAxisGroups
    (groups : List (List (Expr × String))) : String :=
  if groups.isEmpty then
    "(scalar)"
  else
    String.intercalate " " (groups.map formatAxisGroup)

/-- Preserve the first occurrence of each reflected axis expression. -/
def uniqueAxes
    (axes : List (Expr × String)) : List (Expr × String) :=
  axes.foldl
    (fun result axis =>
      if result.any fun existing => existing.1 == axis.1 then
        result
      else
        result.concat axis)
    []

/-- Select axes absent from another reflected axis list. -/
def axesNotIn (axes others : List (Expr × String)) :
    List (Expr × String) :=
  axes.filter fun axis =>
    !(others.any fun other => other.1 == axis.1)

/-- Select the axes repeated within one operand. -/
def repeatedAxes
    (axes : List (Expr × String)) : List (Expr × String) :=
  uniqueAxes <| axes.filter fun axis =>
    (axes.filter fun candidate => candidate.1 == axis.1).length > 1

/-- Look up the concrete length associated with a reflected axis expression. -/
def axisLength? (lengths : List (Expr × Nat))
    (axis : Expr) : Option Nat :=
  (lengths.find? fun item => item.1 == axis).map Prod.snd

/-- Resolve an ordered axis list to a concrete shape when every length is known. -/
def axisShape? (axes : List (Expr × String))
    (lengths : List (Expr × Nat)) : Option (List Nat) :=
  axes.mapM fun axis => axisLength? lengths axis.1

/-- Describe singleton dimensions that expand to non-singleton logical axes. -/
def broadcastDescriptions (shape : List Nat)
    (axes : List (Expr × String)) (lengths : List (Expr × Nat)) :
    List String :=
  (shape.zip axes).filterMap fun (dimension, axis) =>
    match axisLength? lengths axis.1 with
    | some logicalLength =>
        if dimension == 1 && logicalLength != 1 then
          some s!"{axis.2}: 1 -> {logicalLength}"
        else
          none
    | none => none

/-- Number human-readable lowering stages from the supplied starting index. -/
def numberedStageLines : Nat → List String → List String
  | _, [] => []
  | number, stage :: stages =>
      s!"    {number}. {stage}" ::
        numberedStageLines (number + 1) stages


end TorchLean.Tensor.Internal.Report.Impl
