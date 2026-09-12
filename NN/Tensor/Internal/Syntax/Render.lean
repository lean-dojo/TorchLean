/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Tensor.Internal.Syntax.Ast
public import NN.Tensor.Internal.Syntax.Lexer

/-!
# Canonical rendering of einops patterns

These functions serialize the existing syntax trees; there is no parallel
pretty-printing AST. Rendering deliberately omits source spans and the source
offset used to distinguish anonymous-axis occurrences. It preserves every
syntactic choice that affects checking: axis order, parentheses, unit axes,
ellipses, pack boundaries, and einsum operand boundaries.

Canonical output uses ASCII decimal notation for anonymous dimensions,
retains named axes exactly (including Unicode), and follows the spacing rules
of `TokenKind.renderSequence`.
-/

@[expose] public section

namespace TorchLean.Tensor.Internal.Syntax

namespace Axis

/-- Convert one elementary axis to its canonical lexer token. -/
def tokenKind : Axis → TokenKind
  | .named name => .word name
  | .anonymous value _ => .word (toString value)
  | .unit => .word "1"
  | .ellipsis => .ellipsis

/-- Render one elementary axis without source-location metadata. -/
def render (axis : Axis) : String :=
  axis.tokenKind.render

end Axis

namespace CompositeAxis

/-- Canonical token sequence for one top-level or parenthesized axis. -/
def tokenKinds (axis : CompositeAxis) : List TokenKind :=
  let body := axis.axes.map fun item => item.value.tokenKind
  if axis.parenthesized then
    .leftParen :: body ++ [.rightParen]
  else
    body

/--
Render one top-level axis, retaining parentheses even for empty or unary
groups because grouping is part of the parsed syntax.
-/
def render (axis : CompositeAxis) : String :=
  TokenKind.renderSequence axis.tokenKinds

end CompositeAxis

namespace Expression

/-- Canonical lexer tokens for an expression, in axis order. -/
def tokenKinds (expression : Expression) : List TokenKind :=
  expression.axes.flatMap CompositeAxis.tokenKinds

/-- Render one expression with exactly one space between top-level axes. -/
def render (expression : Expression) : String :=
  TokenKind.renderSequence expression.tokenKinds

end Expression

namespace TransformPattern

/-- Render a rearrange, repeat, or reduce pattern in canonical form. -/
def render (pattern : TransformPattern) : String :=
  TokenKind.renderSequence <|
    pattern.left.tokenKinds ++ [TokenKind.arrow] ++
      pattern.right.tokenKinds

end TransformPattern

namespace PackPattern

/-- Render the unique packed segment together with its fixed surrounding axes. -/
def render (pattern : PackPattern) : String :=
  TokenKind.renderSequence <|
    pattern.before.map (fun axis => TokenKind.word axis.value) ++
      [TokenKind.star] ++
      pattern.after.map (fun axis => TokenKind.word axis.value)

end PackPattern

namespace EinsumPattern

/-- Render all einsum operands and the requested output expression. -/
def render (pattern : EinsumPattern) : String :=
  ((pattern.inputs.map Expression.tokenKinds).intersperse [.comma]).flatten ++
    [.arrow] ++ pattern.output.tokenKinds
  |> TokenKind.renderSequence

end EinsumPattern

end TorchLean.Tensor.Internal.Syntax
