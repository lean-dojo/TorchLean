/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public meta import TorchLean.Tensor.Tactic.Einops
public meta import TorchLean.Tensor.Einops.Elab.Output.Parallel -- shake: keep
public import TorchLean.Tensor.Einops.Check.ParseShape -- shake: keep
public import TorchLean.Tensor.Tactic.Einops

/-!
# Shared report decoding

Reflection decoders, formatters and axis analysis for the operation-specific `einops?` reports.
Each decoder reduces with `whnfAll` and returns `none` unless it reaches a literal or a known
constructor.
-/

public meta section

namespace TorchLean.Einops.Report

open Lean Lean.Elab Tactic Meta

/-- Reduce reflected report data to weak head normal form at full transparency, after
instantiating pending metavariables. -/
def whnfAll (expression : Expr) : MetaM Expr := do
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

/-- Decode a reflected string when `whnfAll` reduces it to a literal. -/
def decodeStringLiteral (expression : Expr) :
    MetaM (Option String) := do
  match ← whnfAll expression with
  | .lit (.strVal value) => return some value
  | _ => return none

/-- Decode a reflected natural number only when it reduces to a literal. -/
def decodeNatLiteral (expression : Expr) : MetaM (Option Nat) := do
  getNatValue? (← whnfAll expression)

/-- Decode a reflected list element by element, returning `none` if the spine or any element fails
to decode. -/
partial def decodeExprList {α : Type}
    (expression : Expr) (decodeItem : Expr → MetaM (Option α)) :
    MetaM (Option (List α)) := do
  let expression ← whnfAll expression
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
  let expression ← whnfAll expression
  if expression.isAppOfArity ``AxisId.named 1 then
    let some name ← decodeStringLiteral expression.getAppArgs[0]!
      | return none
    return some (expression, name)
  else if expression.isAppOfArity ``AxisId.anonymous 2 then
    let some value ← decodeNatLiteral expression.getAppArgs[0]!
      | return none
    return some (expression, s!"literal {value}")
  else if expression.isAppOfArity ``AxisId.ellipsis 1 then
    let some index ← decodeNatLiteral expression.getAppArgs[0]!
      | return none
    return some (expression, s!"ellipsis[{index}]")
  else
    return none

/-- Decode a reflected einsum axis to its concrete checker value. -/
def decodeEinsumAxisValue (expression : Expr) :
    MetaM (Option EinsumAxis) := do
  let expression ← whnfAll expression
  if expression.isAppOfArity ``EinsumAxis.named 1 then
    let some name ← decodeStringLiteral expression.getAppArgs[0]!
      | return none
    return some (.named name)
  else if expression.isAppOfArity ``EinsumAxis.ellipsis 1 then
    let some index ← decodeNatLiteral expression.getAppArgs[0]!
      | return none
    return some (.ellipsis index)
  else
    return none

/-- Decode an einsum axis together with its user-facing description. -/
def decodeEinsumAxis (expression : Expr) :
    MetaM (Option (Expr × String)) := do
  let expression ← whnfAll expression
  let some axis ← decodeEinsumAxisValue expression
    | return none
  let description :=
    match axis with
    | .named name => name
    | .ellipsis index => s!"ellipsis[{index}]"
  return some (expression, description)

/-- Pretty-print an elaborated type using the current local context. -/
def formatType (typeExpression : Expr) : MetaM String := do
  return (← ppExpr (← instantiateMVars typeExpression)).pretty

/-- Infer and pretty-print the type of an elaborated expression. -/
def inferredType (expression : Expr) : MetaM String := do
  formatType (← inferType expression)

/-- Pretty-print one concrete tensor type without evaluating a tensor value. -/
def concreteTensorType (scalarType : Expr)
    (shape : List Nat) : MetaM String := do
  formatType (← mkAppM ``Tensor.Rep #[scalarType, Lean.toExpr shape])

/-- Format one uniformly indented report section. -/
def formatSection (title : String) (lines : List String) :
    List String :=
  s!"  {title}:" :: lines.map fun line => s!"    {line}"

/-- Format checked input and output types. -/
def typeCheckSection
    (entries : List (String × String)) : List String :=
  formatSection "Type checks" <|
    entries.map fun entry => s!"{entry.1}: {entry.2}"

/-- Footer section stating that work estimates use static shapes only and that `einops?` never
executes or times user terms. -/
def performanceFooter : List String :=
  formatSection "Performance note"
    ["Work estimates use only static shapes; `einops?` never executes or \
        times user terms."]

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

/-- Remove repeated axes, keeping the first occurrence of each reflected axis expression. -/
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

/-- Resolve an ordered axis list to a concrete shape when every length is known. -/
def axisShape? (axes : List (Expr × String))
    (lengths : List (Expr × Nat)) : Option (List Nat) :=
  axes.mapM fun axis => lengths.lookup axis.1

/-- Describe singleton dimensions that expand to non-singleton logical axes. -/
def broadcastDescriptions (shape : List Nat)
    (axes : List (Expr × String)) (lengths : List (Expr × Nat)) :
    List String :=
  (shape.zip axes).filterMap fun (dimension, axis) =>
    match lengths.lookup axis.1 with
    | some logicalLength =>
        if dimension == 1 && logicalLength != 1 then
          some s!"{axis.2}: 1 -> {logicalLength}"
        else
          none
    | none => none

/-- Number human-readable lowering stages from the supplied starting index. -/
def numberedStageLines (start : Nat) (stages : List String) : List String :=
  stages.mapIdx fun index stage => s!"    {start + index}. {stage}"


end TorchLean.Einops.Report
