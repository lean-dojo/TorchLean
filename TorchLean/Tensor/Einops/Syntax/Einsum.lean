/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Tensor.Einops.Syntax.Expression
public import TorchLean.Tensor.Einops.Syntax.Expression.Split -- shake: keep

/-!
# Einsum-pattern parsing

This module parses comma-separated operands and the output expression. Empty
operand expressions represent scalar tensors, so leading, adjacent, and trailing
commas preserve empty operands.
Dimension compatibility and the supported axis forms belong to the einsum checker.
-/

@[expose] public section

namespace TorchLean.Einops

open Syntax EinsumPattern Expression

/--
Split a token list at its commas into groups paired with their start offsets (`start` for the
first, the end of the preceding comma for the others), keeping empty groups because each one is a
scalar operand. The two accumulators hold the current group and the finished groups, both reversed.
-/
def Syntax.splitCommas (start : Nat) :
    List Token → List Token → List (Nat × List Token) → List (Nat × List Token)
  | [], currentRev, groupsRev => ((start, currentRev.reverse) :: groupsRev).reverse
  | token :: rest, currentRev, groupsRev =>
      match token.value with
      | .comma => splitCommas token.span.stop rest [] ((start, currentRev.reverse) :: groupsRev)
      | _ => splitCommas start rest (token :: currentRev) groupsRev

/-- The span from the first token's start to the last token's end, or `fallback` for `[]`. -/
def Syntax.tokenListSpan (fallback : Span) : List Token → Span
  | [] => fallback
  | first :: rest =>
      match rest.getLast? with
      | none => first.span
      | some last => Span.join first.span last.span

/-- Parse comma-separated einsum operands in source order. -/
def EinsumPattern.parseInputs (policy : IdentifierPolicy) :
    List (Nat × List Token) → List Expression → Result (List Expression)
  | [], parsedRev => .ok parsedRev.reverse
  | (start, tokens) :: rest, parsedRev => do
      let span := tokenListSpan (Span.point start) tokens
      let parsed ← parseTokens policy .einsumInput span tokens {}
      parseInputs policy rest (parsed :: parsedRev)

/-- Parse an einsum pattern with one or more comma-separated input expressions. -/
def EinsumPattern.parse (source : String)
    (policy := IdentifierPolicy.python) : Result EinsumPattern := do
  let whole := sourceSpan source
  let tokens ← lex source policy .transformation
  let (inputTokens, arrow, outputTokens) ← splitArrow (Span.point whole.stop) tokens []
  let inputs ← parseInputs policy (splitCommas 0 inputTokens [] []) []
  let outputSpan := Span.between arrow.span.stop whole.stop
  let output ← parseTokens policy .parseShape outputSpan outputTokens {}
  .ok { inputs, arrow := arrow.span, output, span := whole }

end TorchLean.Einops
