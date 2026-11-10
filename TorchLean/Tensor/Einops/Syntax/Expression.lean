/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Tensor.Einops.Syntax.Expression.State

/-!
# Expression token parsing

`Expression.parseTokens` reads one expression from a lexed token stream. It allows one level of
parentheses, decodes each word as a named, anonymous or unit axis, and rejects arrows, commas
and stars. Tensor dimensions and operation-specific constraints are checked later, against the
parsed tree.
-/

@[expose] public section

namespace TorchLean.Einops

open Syntax (Result)

section

open Syntax (pushAxis)

/--
Parse the tokens of one indexing expression, carrying the open group, the completed axes and the
axis keys seen so far in an `Einops.ExpressionState`.
-/
def Expression.parseTokens (policy : IdentifierPolicy)
    (config : ExpressionConfig) (expressionSpan : Span) :
    List Token → ExpressionState → Result Expression
  | [], state =>
      match state.group with
      | some (openSpan, _) =>
          .error
            { code := .unbalancedParenthesis
              message := "unclosed parenthesis in indexing expression"
              span := Span.join openSpan (Span.point expressionSpan.stop) }
      | none =>
          .ok { axes := state.axesRev.reverse, span := expressionSpan }
  | token :: rest, state => do
      match token.value with
      | .word _ =>
          let (axis, state) ← Syntax.decodeWord policy config token state
          parseTokens policy config expressionSpan rest (pushAxis axis state)
      | .ellipsis =>
          let (axis, state) ← Syntax.decodeEllipsis config token state
          parseTokens policy config expressionSpan rest (pushAxis axis state)
      | .leftParen =>
          match state.group with
          | some _ =>
              .error
                { code := .nestedParenthesis
                  message := "axis composition is one-level"
                  span := token.span }
          | none =>
              parseTokens policy config expressionSpan rest
                { state with group := some (token.span, []) }
      | .rightParen =>
          match state.group with
          | none =>
              .error
                { code := .unbalancedParenthesis
                  message := "closing parenthesis has no matching opening parenthesis"
                  span := token.span }
          | some (openSpan, axesRev) =>
              let composite :=
                { axes := axesRev.reverse
                  parenthesized := true
                  span := Span.join openSpan token.span }
              parseTokens policy config expressionSpan rest
                { state with group := none, axesRev := composite :: state.axesRev }
      | _ =>
          .error
            { code := .unexpectedToken
              message := "unexpected punctuation in indexing expression"
              span := token.span }

/-- The span of the whole `source`, starting at 0 and counted in Unicode scalar values. -/
def Syntax.sourceSpan (source : String) : Span :=
  ⟨0, source.toList.length⟩

end

/-- Lex and parse one indexing expression, such as a `parse_shape` pattern, under `config`. -/
def Expression.parse (source : String)
    (config := ExpressionConfig.transformation)
    (policy := IdentifierPolicy.python) : Result Expression := do
  let tokens ← Syntax.lex source policy .transformation
  parseTokens policy config (Syntax.sourceSpan source) tokens {}

end TorchLean.Einops
