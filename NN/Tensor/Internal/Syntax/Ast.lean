/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Tensor.Internal.Syntax.Span

/-!
# Lossless einops syntax trees

The AST preserves grouping, unit axes, anonymous-axis occurrences, and source
locations. This is more informative than the reference implementation's
internal `ParsedExpression`, which erases unit axes while parsing.
-/

@[expose] public section

namespace TorchLean.Tensor.Internal.Syntax

/--
An elementary axis in a transformation expression.

Anonymous axes carry an occurrence number because two textual occurrences of
the same numeral denote distinct axes in einops.
-/
inductive Axis where
  /-- A logical axis identified by the user's name. -/
  | named (name : String)
  /-- A numeric axis, distinguished by its occurrence in the source pattern. -/
  | anonymous (value occurrence : Nat)
  /-- A length-one axis written as `1`. -/
  | unit
  /-- The `...` axis that stands for zero or more physical dimensions. -/
  | ellipsis
deriving Repr, DecidableEq

/-- One top-level axis or parenthesized composition. -/
structure CompositeAxis where
  /-- Elementary axes written in this top-level component. -/
  axes : List (Located Axis)
  /-- Whether the component was explicitly enclosed in parentheses. -/
  parenthesized : Bool
  /-- Source range covering the complete component. -/
  span : Span
deriving Repr, DecidableEq

namespace CompositeAxis

/-- Unit axes disappear from the elementary-axis sequence. -/
def semanticAxes (axis : CompositeAxis) : List (Located Axis) :=
  axis.axes.filter fun item => item.value != .unit

/-- Whether this composition contains an ellipsis. -/
def hasEllipsis (axis : CompositeAxis) : Bool :=
  axis.axes.any fun item => item.value == .ellipsis

end CompositeAxis

/-- One side of a transformation or einsum pattern. -/
structure Expression where
  /-- Top-level physical axes in source order. -/
  axes : List CompositeAxis
  /-- Source range covering the complete expression. -/
  span : Span
deriving Repr, DecidableEq

namespace Expression

/-- Elementary axes after removing syntactic unit axes. -/
def flatAxes (expression : Expression) : List (Located Axis) :=
  expression.axes.flatMap CompositeAxis.semanticAxes

/-- Named axes in source order, including repeated occurrences when allowed. -/
def namedAxes (expression : Expression) : List (Located String) :=
  expression.flatAxes.filterMap fun axis =>
    match axis.value with
    | .named name => some ⟨name, axis.span⟩
    | _ => none

/-- The number of ellipses in an expression. -/
def ellipsisCount (expression : Expression) : Nat :=
  expression.flatAxes.countP fun axis => axis.value == .ellipsis

/-- Whether an ellipsis occurs inside parentheses. -/
def hasParenthesizedEllipsis (expression : Expression) : Bool :=
  expression.axes.any fun axis => axis.parenthesized && axis.hasEllipsis

/-- Whether an expression contains a genuine composite axis. -/
def hasCompositeAxes (expression : Expression) : Bool :=
  expression.axes.any fun axis => axis.semanticAxes.length > 1

end Expression

/-- The shared syntax of `rearrange`, `repeat`, and `reduce`. -/
structure TransformPattern where
  /-- Input-side expression. -/
  left : Expression
  /-- Source range of the `->` token. -/
  arrow : Span
  /-- Output-side expression. -/
  right : Expression
  /-- Source range covering the complete transformation pattern. -/
  span : Span
deriving Repr, DecidableEq

/-- The separate grammar used by `pack` and `unpack`. -/
structure PackPattern where
  /-- Fixed named axes before the packed `*` region. -/
  before : List (Located String)
  /-- Source range of the unique `*` token. -/
  packed : Span
  /-- Fixed named axes after the packed `*` region. -/
  after : List (Located String)
  /-- Source range covering the complete packing pattern. -/
  span : Span
deriving Repr, DecidableEq

/-- The multi-input grammar used by `einsum`. -/
structure EinsumPattern where
  /-- Comma-separated operand expressions in source order. -/
  inputs : List Expression
  /-- Source range of the `->` token. -/
  arrow : Span
  /-- Output expression following the arrow. -/
  output : Expression
  /-- Source range covering the complete einsum pattern. -/
  span : Span
deriving Repr, DecidableEq

end TorchLean.Tensor.Internal.Syntax
