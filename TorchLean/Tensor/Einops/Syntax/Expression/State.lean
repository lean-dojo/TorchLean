/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Tensor.Einops.Syntax.Expression.Config
public import TorchLean.Tensor.Einops.Syntax.Lexer

/-!
# Expression token state

This module owns duplicate detection, group state, and token decoding.
-/

@[expose] public section

namespace TorchLean.Einops

/-- Duplicate-detection identity for a named axis or the single ellipsis marker. -/
inductive AxisKey where
  /-- Duplicate-detection key for a user-named axis. -/
  | named (name : String)
  /-- Duplicate-detection key shared by all ellipsis occurrences. -/
  | ellipsis
deriving DecidableEq

/-- Accumulated grouping and duplicate-detection state for one expression. -/
structure ExpressionState where
  /-- Axis identities already accepted in this expression. -/
  seen : List AxisKey := []
  /-- The open parenthesized group, if any: the span of its `(` and its axes in reverse order. -/
  group : Option (Span × List (Located Axis)) := none
  /-- Completed physical axes in reverse source order. -/
  axesRev : List CompositeAxis := []

/-- Construct a source-located duplicate-axis diagnostic. -/
def Syntax.duplicateDiagnostic (span : Span) : Diagnostic ParseError :=
  { code := .duplicateAxis
    message := "axis occurs more than once in an indexing expression"
    span }

/--
Add `key` to `seen`. A second ellipsis always fails, and a repeated name fails unless
`config.allowDuplicates` holds or the name is `_` with `config.allowUnderscore` set.
-/
def Syntax.registerKey (config : ExpressionConfig) (key : AxisKey)
    (span : Span) (seen : List AxisKey) : Result (List AxisKey) :=
  let repeatedUnderscore :=
    config.allowUnderscore && key == .named "_"
  if key == .ellipsis && key ∈ seen then
    .error
      { code := .duplicateEllipsis
        message := "an expression may contain only one ellipsis"
        span }
  else if key ∈ seen && !config.allowDuplicates && !repeatedUnderscore then
    .error (duplicateDiagnostic span)
  else
    .ok (key :: seen)

/-- Successful duplicate registration is independent of the diagnostic source span. -/
theorem Syntax.registerKey_eq_ok_at_span (config : ExpressionConfig)
    (key : AxisKey) (sourceSpan targetSpan : Span)
    (seen nextSeen : List AxisKey)
    (hRegister :
      registerKey config key sourceSpan seen = .ok nextSeen) :
    registerKey config key targetSpan seen = .ok nextSeen := by
  unfold registerKey at hRegister ⊢
  split <;> simp_all
  split <;> simp_all

/--
Whether `name` breaks the einops underscore rule by starting or ending with `_`. The name `_`
itself passes when `config.allowUnderscore` is set.
-/
def Syntax.hasForbiddenUnderscore (config : ExpressionConfig) (name : String) : Bool :=
  if config.allowUnderscore && name == "_" then
    false
  else
    name.startsWith "_" || name.endsWith "_"

/--
Decode one word token as a named, anonymous or unit axis.

A decimal word gives an anonymous axis of that length, except that value 1 gives the unit axis
and value 0 is rejected. Any other word must pass `policy.isIdentifier` and the underscore rule,
and is registered for duplicate detection.
-/
def Syntax.decodeWord (policy : IdentifierPolicy) (config : ExpressionConfig)
    (token : Token) (state : ExpressionState) :
    Result (Located Axis × ExpressionState) := do
  let .word text := token.value
    | .error
        { code := .unexpectedToken
          message := "expected an axis identifier"
          span := token.span }
  if policy.isDecimal text then
    let some value := policy.toNat? text
      | .error
          { code := .invalidAnonymousAxis
            message := "anonymous axis is not a natural number"
            span := token.span }
    if value == 0 then
      .error
        { code := .invalidAnonymousAxis
          message := "anonymous axes must have positive length"
          span := token.span }
    else if value == 1 then
      .ok (⟨.unit, token.span⟩, state)
    else
      let axis := Axis.anonymous value token.span.offset
      .ok (⟨axis, token.span⟩, state)
  else if !policy.isIdentifier text || hasForbiddenUnderscore config text then
    .error
      { code := .invalidIdentifier
        message := "invalid axis identifier"
        span := token.span }
  else
    let seen ← registerKey config (.named text) token.span state.seen
    .ok (⟨.named text, token.span⟩, { state with seen })

/-- Decode and register a bare ellipsis token. -/
def Syntax.decodeEllipsis (config : ExpressionConfig) (token : Token)
    (state : ExpressionState) : Result (Located Axis × ExpressionState) := do
  let seen ← registerKey config .ellipsis token.span state.seen
  .ok (⟨.ellipsis, token.span⟩, { state with seen })

/-- Append an axis to the open group, or as its own top-level physical axis if none is open. -/
def Syntax.pushAxis (axis : Located Axis) (state : ExpressionState) : ExpressionState :=
  match state.group with
  | none =>
      { state with
        axesRev :=
          { axes := [axis], parenthesized := false, span := axis.span } :: state.axesRev }
  | some (openSpan, axesRev) =>
      { state with group := some (openSpan, axis :: axesRev) }

end TorchLean.Einops
