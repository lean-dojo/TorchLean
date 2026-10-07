/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module
public import NN.Tensor.Internal.Syntax.Ast -- shake: keep

/-!
# Expression parser configuration

Each public pattern language selects duplicate-axis and ignored-axis rules
through one explicit configuration value.
-/

@[expose] public section

namespace TorchLean.Tensor.Internal.Syntax

/-- Duplicate and underscore rules for one expression. -/
structure ExpressionConfig where
  /-- Whether the name `_` is accepted, including repeated occurrences. -/
  allowUnderscore : Bool := false
  /-- Whether named axes may repeat; an ellipsis must still occur at most once. -/
  allowDuplicates : Bool := false

namespace ExpressionConfig

/-- Rules used by transformation patterns. -/
def transformation : ExpressionConfig :=
  {}

/-- Accept `_` in `parse_shape` and einsum output expressions; only `parse_shape` ignores it. -/
def parseShape : ExpressionConfig :=
  { allowUnderscore := true }

/-- Rules used by an einsum input expression. -/
def einsumInput : ExpressionConfig :=
  { allowUnderscore := true, allowDuplicates := true }

end ExpressionConfig

end TorchLean.Tensor.Internal.Syntax
