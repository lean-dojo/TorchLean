/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module
public import TorchLean.Tensor.Einops.Syntax.AST -- shake: keep

/-!
# Expression parser configuration

Transformation patterns, `parse_shape` and einsum differ in two rules: whether the name `_` is
allowed and whether an axis name may repeat. Each language hands its choice to the expression
parser as one `ExpressionConfig` value.
-/

@[expose] public section

namespace TorchLean.Einops

/-- Duplicate and underscore rules for one expression. -/
structure ExpressionConfig where
  /-- Whether the name `_` is accepted, including repeated occurrences. -/
  allowUnderscore : Bool := false
  /-- Whether a named axis may occur more than once. At most one ellipsis is allowed either way. -/
  allowDuplicates : Bool := false

namespace ExpressionConfig

/-- Rules for transformation patterns: `_` is rejected and no axis name may repeat. -/
def transformation : ExpressionConfig :=
  {}

/--
Rules for `parse_shape` and einsum output expressions, which accept `_` (as a dimension to skip
in `parse_shape` and as an ordinary label in einsum).
-/
def parseShape : ExpressionConfig :=
  { allowUnderscore := true }

/-- Rules for einsum inputs: `_` is accepted and a repeated label denotes a diagonal. -/
def einsumInput : ExpressionConfig :=
  { allowUnderscore := true, allowDuplicates := true }

end ExpressionConfig

end TorchLean.Einops
