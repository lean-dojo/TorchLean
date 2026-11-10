/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Tensor.Einops.Syntax.Span

/-!
# Parser diagnostics

Diagnostics carry a stable machine-readable code, a human-readable message,
and the smallest useful source span. Parsing and shape checking share the
structure and differ only in their code enums.
-/

@[expose] public section

namespace TorchLean.Einops

universe u

/-- Stable categories for syntax errors. -/
inductive ParseError where
  /-- The lexer finds a character outside the einops token alphabet. -/
  | unknownCharacter
  /-- A pattern that requires `->` has none. -/
  | missingArrow
  /-- A pattern contains more than one `->`. -/
  | duplicateArrow
  /-- A token is not valid at its position in the current grammar. -/
  | unexpectedToken
  /-- An opening or closing parenthesis has no matching partner. -/
  | unbalancedParenthesis
  /-- A parenthesized axis group contains another parenthesized group. -/
  | nestedParenthesis
  /-- A word fails the selected identifier policy. -/
  | invalidIdentifier
  /-- A named axis occurs more than once where uniqueness is required. -/
  | duplicateAxis
  /-- A numeric axis is zero or does not decode to a natural number. -/
  | invalidAnonymousAxis
  /-- An expression contains more than one ellipsis. -/
  | duplicateEllipsis
  /-- A pack pattern lacks its required `*` axis. -/
  | missingPackAxis
  /-- A pack pattern contains more than one `*` axis. -/
  | duplicatePackAxis
deriving Repr, DecidableEq, BEq

/-- A structured front-end error whose category is drawn from `Code`. -/
structure Diagnostic (Code : Type) where
  /-- Stable machine-readable category of the failure. -/
  code : Code
  /-- Human-readable explanation of the malformed pattern or failed obligation. -/
  message : String
  /-- Smallest useful source range responsible for the failure. -/
  span : Span
deriving Repr, DecidableEq

/-- The result type used by the syntax front end. -/
abbrev Syntax.Result (α : Type u) := Except (Diagnostic ParseError) α

end TorchLean.Einops
