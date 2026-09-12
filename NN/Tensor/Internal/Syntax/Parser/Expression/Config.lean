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
  /-- Whether `_` is accepted as an ignored axis name. -/
  allowUnderscore : Bool := false
  /-- Whether a named axis may occur more than once in the expression. -/
  allowDuplicates : Bool := false

namespace ExpressionConfig

/-- Rules used by transformation patterns. -/
def transformation : ExpressionConfig :=
  {}

/-- Rules used by `parse_shape`. -/
def parseShape : ExpressionConfig :=
  { allowUnderscore := true }

/-- Rules used by an einsum input expression. -/
def einsumInput : ExpressionConfig :=
  { allowUnderscore := true, allowDuplicates := true }

/-- Rules used by an einsum output expression. -/
def einsumOutput : ExpressionConfig :=
  { allowUnderscore := true }

end ExpressionConfig

end TorchLean.Tensor.Internal.Syntax
