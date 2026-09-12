/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Tensor.Internal.Syntax.Span

/-!
# Parser diagnostics

Diagnostics carry a stable machine-readable code, a human-readable message,
and the smallest useful source span.
-/

@[expose] public section

namespace TorchLean.Tensor.Internal.Syntax

universe u

/-- Stable categories for syntax errors. -/
inductive DiagnosticCode where
  /-- The lexer encountered a character outside the einops token alphabet. -/
  | unknownCharacter
  /-- A pattern that requires `->` did not contain one. -/
  | missingArrow
  /-- A pattern contained more than one `->`. -/
  | duplicateArrow
  /-- A token was not valid at its position in the current grammar. -/
  | unexpectedToken
  /-- An opening or closing parenthesis had no matching partner. -/
  | unbalancedParenthesis
  /-- A parenthesized axis group contained another parenthesized group. -/
  | nestedParenthesis
  /-- A word did not satisfy the selected identifier policy. -/
  | invalidIdentifier
  /-- A named axis occurred more than once where uniqueness is required. -/
  | duplicateAxis
  /-- A numeric axis was zero or could not be decoded as a natural number. -/
  | invalidAnonymousAxis
  /-- An expression contained more than one ellipsis. -/
  | duplicateEllipsis
  /-- A pack pattern did not contain its required `*` axis. -/
  | missingPackAxis
  /-- A pack pattern contained more than one `*` axis. -/
  | duplicatePackAxis
deriving Repr, DecidableEq, BEq

/-- A structured syntax error. -/
structure Diagnostic where
  /-- Stable machine-readable category of the parse failure. -/
  code : DiagnosticCode
  /-- Human-readable explanation of the malformed pattern. -/
  message : String
  /-- Smallest useful source range responsible for the failure. -/
  span : Span
deriving Repr, DecidableEq

/-- The result type used by the syntax front end. -/
abbrev Result (α : Type u) := Except Diagnostic α

end TorchLean.Tensor.Internal.Syntax
