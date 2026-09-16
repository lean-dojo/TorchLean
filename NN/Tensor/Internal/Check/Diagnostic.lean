/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Tensor.Internal.Syntax.Diagnostic -- shake: keep

/-!
# Shape-checking diagnostics

Shape checking is separate from parsing, but uses the same source spans.
-/

@[expose] public section

namespace TorchLean.Tensor.Internal.Check

universe u

/-- Stable categories for concrete shape-checking failures. -/
inductive DiagnosticCode where
  /-- The number of physical dimensions disagreed with the pattern rank. -/
  | rankMismatch
  /-- An ellipsis was placed or expanded in a way forbidden by the operation. -/
  | invalidEllipsis
  /-- A rearrangement used a numeric axis, which cannot identify input data. -/
  | anonymousAxisInRearrange
  /-- The input and output patterns referred to different logical axes. -/
  | axisMismatch
  /-- The caller supplied more than one length for the same named axis. -/
  | duplicateAxisLength
  /-- The caller supplied a named length that the pattern does not use. -/
  | unusedAxisLength
  /-- A composite input dimension did not determine one of its axis lengths. -/
  | cannotInferAxis
  /-- A known axis length disagreed with the corresponding tensor dimension. -/
  | dimensionMismatch
  /-- An output-only axis had no length supplied by the caller. -/
  | missingAxisLength
  /-- The number of einsum tensors disagreed with the number of input expressions. -/
  | einsumOperandCountMismatch
  /-- An einsum expression used grouping or an axis form outside its surface grammar. -/
  | unsupportedEinsumAxis
  /-- An einsum output named an axis absent from every input. -/
  | unknownEinsumOutputAxis
  /-- An einsum output named the same logical axis more than once. -/
  | duplicateEinsumOutputAxis
  /-- The reference-compatible einsum front end exceeded its 52-label limit. -/
  | tooManyEinsumAxes
  /-- Repeated occurrences of an einsum label in one operand had unequal lengths. -/
  | repeatedEinsumDimensionMismatch
  /-- Non-singleton occurrences of an einsum axis had incompatible lengths. -/
  | incompatibleEinsumBroadcast
  /-- A `parse_shape` expression used a grouped physical axis. -/
  | compositeParseShapeAxis
  /-- A `parse_shape` expression bound the same name more than once. -/
  | duplicateParseShapeAxis
  /-- `pack` was asked to combine no tensors. -/
  | emptyPackInput
  /-- A tensor rank was too small for the fixed axes of a pack pattern. -/
  | packRankMismatch
  /-- Corresponding fixed axes of packed tensors had different lengths. -/
  | packDimensionMismatch
  /-- A packed tensor rank disagreed with the rank implied by its metadata. -/
  | unpackRankMismatch
  /-- Unpack metadata did not describe a valid concrete packed shape. -/
  | invalidPackedShape
  /-- An unpack shape requested inference for more than one dimension. -/
  | multipleInferredDimensions
  /-- The packed extent did not uniquely determine an inferred dimension. -/
  | cannotInferPackedDimension
  /-- Unpack metadata did not multiply to the packed tensor dimension. -/
  | packedAxisMismatch
  /-- An implementation invariant failed after earlier checks had established it. -/
  | internalInvariant
deriving Repr, DecidableEq, BEq

/-- A structured concrete-checking error. -/
structure Diagnostic where
  /-- Stable machine-readable category of the checking failure. -/
  code : DiagnosticCode
  /-- Human-readable explanation of the failed shape obligation. -/
  message : String
  /-- Smallest useful range of the pattern responsible for the failure. -/
  span : Syntax.Span
deriving Repr, DecidableEq

/-- The result type used by concrete checkers. -/
abbrev Result (α : Type u) := Except Diagnostic α

end TorchLean.Tensor.Internal.Check
