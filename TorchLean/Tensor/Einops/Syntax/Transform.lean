/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Tensor.Einops.Syntax.Expression
public import TorchLean.Tensor.Einops.Syntax.Expression.Split -- shake: keep

/-!
# Transformation-pattern parsing

This module parses `rearrange`, `repeat`, and `reduce` patterns.
-/

@[expose] public section

namespace TorchLean.Einops.TransformPattern

open Expression Syntax

/-- Parse the shared pattern language of `rearrange`, `repeat`, and `reduce`. -/
def parse (source : String)
    (policy := IdentifierPolicy.python) : Result TransformPattern := do
  let whole := sourceSpan source
  let tokens ← lex source policy .transformation
  let (leftTokens, arrow, rightTokens) ← splitArrow (Span.point whole.stop) tokens []
  let leftSpan := Span.between whole.offset arrow.span.offset
  let rightSpan := Span.between arrow.span.stop whole.stop
  let left ← parseTokens policy .transformation leftSpan leftTokens {}
  let right ← parseTokens policy .transformation rightSpan rightTokens {}
  .ok { left, arrow := arrow.span, right, span := whole }

end TorchLean.Einops.TransformPattern
