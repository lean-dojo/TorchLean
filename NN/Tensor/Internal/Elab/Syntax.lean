/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module
public import Mathlib.Init -- shake: keep


/-!
# Literal tensor-pattern term syntax

This module contains only the user-facing term grammar. Elaborator
implementations live in the other `TorchLean.Tensor.Internal.Elab` modules, so tools that
inspect or extend the syntax do not need to import compiler machinery.

The term forms are scoped syntax in the canonical `TorchLean.Tensor` namespace. Write
`open TorchLean.Tensor` (or `open scoped TorchLean.Tensor`) to activate the
`rearrange`, `expand`, `reduce`, `einsum`, `pack`, `unpack`, and `parse_shape`
keywords. Without that `open`, these words stay ordinary identifiers, so a
file that only imports the tensor library keeps `pack`, `reduce`, and
`expand` available as names.
-/

public meta section

namespace TorchLean.Tensor.Internal.Elab

/--
One elementary-axis length supplied to an einops pattern.

Lean identifiers provide the concise common form. A string literal names any
Python-compatible axis, including names outside Lean's identifier grammar and
Python keywords.
-/
declare_syntax_cat einopsAxisLength

/-- Assign an axis length using a Lean identifier. -/
syntax ident " := " term : einopsAxisLength

/-- Assign an axis length using an arbitrary string name. -/
syntax str " := " term : einopsAxisLength

/--
A built-in or named reducer, or a parenthesized total multiset aggregate.
-/
declare_syntax_cat einopsReduction

/-- Select a built-in reducer or an unparenthesized named aggregate. -/
syntax ident : einopsReduction

/-- Supply a total multiset aggregate as a reduction term. -/
syntax "(" term ")" : einopsReduction

end TorchLean.Tensor.Internal.Elab

namespace TorchLean.Tensor

/--
Rearrange a tensor with a literal einops pattern checked against its static
  shape. The optional `with` clause supplies lengths needed to split composite
  input dimensions. Active after `open TorchLean.Tensor`.
-/
scoped syntax (name := rearrangeStx)
  "rearrange " term:arg str
    (" with " einopsAxisLength,*)? : term

/--
Expand a tensor along new axes with a literal einops pattern checked against
its static shape. This is the einops `repeat` operation under a name that
does not collide with Lean's `repeat`. New output axes require lengths in the
  optional `with` clause. Active after `open TorchLean.Tensor`.
-/
scoped syntax (name := expandStx)
  "expand " term:arg str
    (" with " einopsAxisLength,*)? : term

/--
Reduce a tensor with a literal einops pattern. The named reduction follows
`by`; accepted built-ins are `sum`, `prod`, `mean`, `min`, `max`, `any`, and
`all`. Other identifiers and parenthesized terms are elaborated as total
multiset aggregates and may change the output scalar type. Active after
  `open TorchLean.Tensor`.
-/
scoped syntax (name := reduceStx)
  "reduce " term:arg str " by " einopsReduction
    (" with " einopsAxisLength,*)? : term

/--
Contract one or more tensors with a literal, statically checked einsum
pattern. Operand tensors may have different shapes and registered scalar
types; multi-input scalar promotion is automatic. Active after
  `open TorchLean.Tensor`.
-/
scoped syntax (name := einsumStx)
  "einsum " term:arg,+ str : term

/--
Pack one or more tensors with a literal packing pattern. Registered scalar
types are promoted automatically. The result keeps the packed tensor and the
certified star shape of each input in one named value. Active after
  `open TorchLean.Tensor`.
-/
scoped syntax (name := packStx)
  "pack " term:arg,+ str : term

/--
Recover the component tensors from a named result returned by `pack`. Active
  after `open TorchLean.Tensor`.
-/
scoped syntax (name := unpackStx)
  "unpack " term:arg str : term

/--
Match a literal parse-shape expression against a statically shaped tensor.
The result lists each named axis and its length in pattern order; wildcards
and ellipsis-expanded dimensions are omitted. Active after
  `open TorchLean.Tensor`.
-/
scoped syntax (name := parseShapeStx)
  "parse_shape " term:arg str : term

end TorchLean.Tensor
