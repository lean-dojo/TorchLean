/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Tensor.Einops.Syntax.Expression
public import TorchLean.Tensor.Einops.Syntax.Expression.Split -- shake: keep

/-!
# Pack-pattern parsing

This module parses the fixed axes surrounding a unique packed `*` axis. Fixed
axes must be distinct names. Numeric axes, underscores, groups, and ellipses are
not part of this grammar.
The pack checker handles tensor shapes and unpacked component sizes.
-/

@[expose] public section

namespace TorchLean.Einops.PackPattern

open Syntax

/--
Parse the fixed axis names before and after the one packed `*`.

The accumulators hold the names before the star (reversed), the star's span once it has been
read, the names after it (reversed) and every name seen so far, for duplicate detection.
-/
def parseTokens (policy : IdentifierPolicy) (whole : Span) :
    List Token → List (Located String) → Option Span →
      List (Located String) → List String → Result PackPattern
  | [], _, none, _, _ =>
      .error
        { code := .missingPackAxis
          message := "pack pattern must contain exactly one '*'"
          span := Span.point whole.stop }
  | [], beforeRev, some packed, afterRev, _ =>
      .ok
        { before := beforeRev.reverse
          packed
          after := afterRev.reverse
          span := whole }
  | token :: rest, beforeRev, packed, afterRev, seen => do
      match token.value with
      | .star =>
          match packed with
          | some _ =>
              .error
                { code := .duplicatePackAxis
                  message := "pack pattern contains more than one '*'"
                  span := token.span }
          | none =>
              parseTokens policy whole rest beforeRev (some token.span) afterRev seen
      | .word name =>
          if !policy.isIdentifier name ||
              name.startsWith "_" || name.endsWith "_" then
            .error
              { code := .invalidIdentifier
                message := "invalid axis identifier in pack pattern"
                span := token.span }
          else if name ∈ seen then
            .error (duplicateDiagnostic token.span)
          else
            let located := Located.mk name token.span
            match packed with
            | none =>
                parseTokens policy whole rest (located :: beforeRev) packed afterRev
                  (name :: seen)
            | some _ =>
                parseTokens policy whole rest beforeRev packed (located :: afterRev)
                  (name :: seen)
      | _ =>
          .error
            { code := .unexpectedToken
              message := "pack patterns contain only axis names and one '*'"
              span := token.span }

/-- Lex and parse a `pack` or `unpack` pattern. -/
def parse (source : String)
    (policy := IdentifierPolicy.python) : Result PackPattern := do
  let whole := sourceSpan source
  let tokens ← lex source policy .pack
  parseTokens policy whole tokens [] none [] []

end TorchLean.Einops.PackPattern
