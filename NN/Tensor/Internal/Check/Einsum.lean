/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Tensor.Internal.Check.Diagnostic
public import NN.Tensor.Internal.Representation.Shape
public import NN.Tensor.Internal.Syntax.Parser -- shake: keep

/-!
# Einsum plans and executable checking

This module defines the semantic data carried by a checked einsum and checks
named-axis patterns against finite tensor shapes. The executable checker
implements the shape behavior of einops v0.8.2:

* an input ellipsis expands to the dimensions not named explicitly;
* ellipsis dimensions are right-aligned between operands;
* dimensions attached to the same global label broadcast when either is one;
* repeated labels inside one operand must have equal dimensions and denote a
  diagonal; and
* output labels must occur in an input.

The mathematical semantics accepts any finite number of named labels. The
pinned Python implementation caps distinct labels at 52 because it encodes them
as ASCII letters, but that ceiling belongs to the encoding rather than to
einsum, so nothing here enforces it.

The solitary name `_` is an ordinary einsum label. It is not the skipped-axis
wildcard used by `parse_shape`.

`CheckedEinsum` is the only einsum plan structure. It stores the parsed
pattern, input shapes, and one total logical-axis length function. Its
invariants prove that each physical dimension is either that resolved length
or a singleton, and that every resolved length is supplied by an occurrence
of the axis unless all occurrences are singleton. This representation supports
symbolic `Nat` dimensions without adding a parallel symbolic plan.

## Reference

The surface restrictions follow einops v0.8.2, pinned at commit
`8e911db71f2e693a0c434b041180388c685ed06f`. Its 52-label limit comes from
`einops.einops._compactify_pattern_for_einsum`, which encodes labels as ASCII
letters; it is not an einsum law. Dimension broadcasting is made explicit here
because the Python front end delegates it to tensor backends.
-/

public section

namespace TorchLean.Tensor.Internal.Check

open TorchLean.Tensor.Internal.Syntax

/--
The identity of one logical einsum dimension after ellipsis expansion.

Named labels preserve the user's full identifier. Ellipsis slots are numbered
from left to right in the common right-aligned ellipsis shape.
-/
inductive EinsumAxis where
  /-- A logical dimension identified by the label written in the pattern. -/
  | named (name : String)
  /-- One slot of the common right-aligned ellipsis dimensions. -/
  | ellipsis (index : Nat)
deriving Repr, @[expose] BEq, @[expose] ReflBEq,
  @[expose] LawfulBEq, @[expose] DecidableEq

attribute [implicit_reducible]
  instBEqEinsumAxis instDecidableEqEinsumAxis

namespace EinsumAxis

/-- A readable label used in concrete shape diagnostics. -/
def description : EinsumAxis → String
  | .named name => s!"axis '{name}'"
  | .ellipsis index => s!"ellipsis dimension {index}"

end EinsumAxis

/--
Whether an einsum expression uses only the v0.8.2-supported axis forms.

Empty expressions represent scalar tensors and are accepted. Each nonempty
top-level position must contain exactly one named label or one bare ellipsis.
Parentheses around a single named label are harmless, as in the reference
parser, but genuine compositions, unit axes, and numeric axes are rejected.
-/
@[expose, implicit_reducible]
def einsumExpressionSupported (expression : Expression) : Bool :=
  decide (expression.ellipsisCount ≤ 1) &&
    expression.axes.all fun composite =>
      match composite.semanticAxes with
      | [axis] =>
          match axis.value with
          | .named _ => true
          | .ellipsis => !composite.parenthesized
          | .anonymous _ _ | .unit => false
      | _ => false

/-- Whether every input and the output use the supported einsum surface. -/
@[expose, implicit_reducible]
def einsumPatternSupported (pattern : EinsumPattern) : Bool :=
  pattern.inputs.all einsumExpressionSupported &&
    einsumExpressionSupported pattern.output

/-- The number of non-ellipsis tensor dimensions written in an expression. -/
@[expose, implicit_reducible]
def fixedEinsumRank (expression : Expression) : Nat :=
  expression.axes.length - expression.ellipsisCount

/--
The number of physical dimensions captured by one input ellipsis.

Rank checking proves that the subtraction is exact. For an expression without
an ellipsis the value is zero.
-/
@[expose, implicit_reducible]
def inputEllipsisRank (expression : Expression) (inputShape : Shape) : Nat :=
  if expression.ellipsisCount = 0 then
    0
  else
    inputShape.length - fixedEinsumRank expression

/--
The common ellipsis rank of an einsum call.

Each operand contributes its own captured rank. Taking their maximum creates
the target to which shorter ellipses are right-aligned.
-/
@[expose, implicit_reducible]
def einsumEllipsisRank (pattern : EinsumPattern) (inputShapes : List Shape) : Nat :=
  ((pattern.inputs.zip inputShapes).map fun input =>
      inputEllipsisRank input.1 input.2).foldl Nat.max 0

/-- Common ellipsis slots occupied by an operand with the given local rank. -/
@[expose, implicit_reducible]
def expandedEllipsisAxes (commonRank operandRank : Nat) : List EinsumAxis :=
  (List.range operandRank).map fun index =>
    .ellipsis (commonRank - operandRank + index)

/-- Expanding an ellipsis never introduces duplicate logical axes. -/
theorem expandedEllipsisAxes_nodup (commonRank operandRank : Nat) :
    (expandedEllipsisAxes commonRank operandRank).Nodup := by
  apply List.nodup_range.map
  intro firstIndex secondIndex hEqual
  exact Nat.add_left_cancel (EinsumAxis.ellipsis.inj hEqual)

/--
Expand one supported expression to logical labels.

`ellipsisRank` is the local captured rank for an input and the common rank for
the output. Unsupported composites contribute no labels; checked patterns
prove that this fallback is unreachable.
-/
@[expose, implicit_reducible]
def expandEinsumExpression (commonRank ellipsisRank : Nat)
    (expression : Expression) : List EinsumAxis :=
  expression.axes.flatMap fun composite =>
    match composite.semanticAxes with
    | [axis] =>
        match axis.value with
        | .named name => [.named name]
        | .ellipsis => expandedEllipsisAxes commonRank ellipsisRank
        | .anonymous _ _ | .unit => []
    | _ => []

/-- Expanded logical labels for every input operand. -/
@[expose, implicit_reducible]
def einsumInputAxes (pattern : EinsumPattern)
    (inputShapes : List Shape) : List (List EinsumAxis) :=
  let commonRank := einsumEllipsisRank pattern inputShapes
  (pattern.inputs.zip inputShapes).map fun input =>
    expandEinsumExpression commonRank
      (inputEllipsisRank input.1 input.2) input.1

/-- Expanded logical labels of the output tensor. -/
@[expose, implicit_reducible]
def einsumOutputAxes (pattern : EinsumPattern)
    (inputShapes : List Shape) : List EinsumAxis :=
  let commonRank := einsumEllipsisRank pattern inputShapes
  expandEinsumExpression commonRank commonRank pattern.output

/--
All logical labels in first-occurrence order.

Repeated input labels are collapsed here, but remain present in each
operand's expanded list where they impose diagonal indexing.
-/
@[expose, implicit_reducible]
def einsumGlobalAxes (pattern : EinsumPattern)
    (inputShapes : List Shape) : List EinsumAxis :=
  (einsumInputAxes pattern inputShapes).flatten.eraseDups

/-- Every physical input dimension paired with its expanded logical label. -/
@[expose, implicit_reducible]
def einsumAxisOccurrences (pattern : EinsumPattern)
    (inputShapes : List Shape) : List (EinsumAxis × Nat) :=
  ((einsumInputAxes pattern inputShapes).zip inputShapes).flatMap fun input =>
    input.1.zip input.2

/--
The broadcast dimension of one logical label.

The first non-singleton occurrence determines the result; if every occurrence
is singleton, the result is one. Compatibility checking proves that every
other occurrence is either singleton or equal to this dimension. In
particular, a zero dimension combined with singleton dimensions resolves to
zero rather than one.
-/
@[expose, implicit_reducible]
def einsumAxisLength (pattern : EinsumPattern) (inputShapes : List Shape)
    (axis : EinsumAxis) : Nat :=
  match (einsumAxisOccurrences pattern inputShapes).find? fun occurrence =>
      occurrence.1 == axis && occurrence.2 != 1 with
  | some occurrence => occurrence.2
  | none => 1

/-- The concrete output, in the axis order written by the user. -/
@[expose, implicit_reducible]
def einsumOutput (pattern : EinsumPattern)
    (inputShapes : List Shape) : Shape :=
  (einsumOutputAxes pattern inputShapes).map
    (einsumAxisLength pattern inputShapes)

/--
Whether repeated occurrences of a label inside one operand have equal
physical dimensions.
-/
@[expose, implicit_reducible]
def repeatedEinsumDimensionsAgree
    (axes : List EinsumAxis) (inputShape : Shape) : Bool :=
  let occurrences := axes.zip inputShape
  occurrences.all fun left =>
    occurrences.all fun right =>
      left.1 != right.1 || left.2 == right.2

/--
The Boolean repeated-label check is exactly pairwise equality of dimensions
for equal labels.
-/
theorem repeatedEinsumDimensionsAgree_eq_true_iff
    {axes : List EinsumAxis} {inputShape : Shape} :
    repeatedEinsumDimensionsAgree axes inputShape = true ↔
      ∀ left ∈ axes.zip inputShape, ∀ right ∈ axes.zip inputShape,
        left.1 = right.1 → left.2 = right.2 := by
  simp only [repeatedEinsumDimensionsAgree, List.all_eq_true, Bool.or_eq_true,
    bne_iff_ne, beq_iff_eq]
  constructor
  · intro h left hLeft right hRight hAxis
    exact
      (h left hLeft right hRight).resolve_left
        (not_ne_iff.mpr hAxis)
  · intro h left hLeft right hRight
    by_cases hAxis : left.1 = right.1
    · exact Or.inr (h left hLeft right hRight hAxis)
    · exact Or.inl hAxis

/-- Whether every axis in `left` occurs in `right`. -/
@[expose, implicit_reducible]
def einsumAxesSubset (left right : List EinsumAxis) : Bool :=
  left.all fun axis => right.contains axis

/-- The Boolean einsum-axis subset test exactly expresses list containment. -/
theorem einsumAxesSubset_eq_true_iff {left right : List EinsumAxis} :
    einsumAxesSubset left right = true ↔
      ∀ ⦃axis⦄, axis ∈ left → axis ∈ right := by
  simp [einsumAxesSubset]

/--
Concrete input dimensions are singleton or equal to their resolved logical
dimension.
-/
abbrev EinsumInputDimensions (pattern : EinsumPattern)
    (inputShapes : List Shape) (axisLength : EinsumAxis → Nat) : Prop :=
  List.Forall₂
    (fun inputShape axes =>
      List.Forall₂
        (fun dimension axis =>
          dimension = axisLength axis ∨ dimension = 1)
        inputShape axes)
    inputShapes (einsumInputAxes pattern inputShapes)

/--
Every resolved logical-axis length is justified by the input dimensions.

A length may be `1` when all occurrences are singleton. Otherwise one
physical occurrence supplies the resolved length. Combined with
`EinsumInputDimensions`, this rules out inventing a larger broadcast result
for an all-singleton axis.
-/
@[expose, implicit_reducible]
def einsumAxisLengths (pattern : EinsumPattern)
    (inputShapes : List Shape) (axisLength : EinsumAxis → Nat) : Bool :=
  (einsumGlobalAxes pattern inputShapes).all fun axis =>
    axisLength axis == 1 ||
      (einsumAxisOccurrences pattern inputShapes).any fun occurrence =>
        occurrence.1 == axis && occurrence.2 == axisLength axis

/-- Every operand satisfies the exact repeated-label dimension rule. -/
abbrev EinsumRepeatedDimensions (pattern : EinsumPattern)
    (inputShapes : List Shape) : Prop :=
  List.Forall₂
    (fun inputShape axes =>
      repeatedEinsumDimensionsAgree axes inputShape = true)
    inputShapes (einsumInputAxes pattern inputShapes)

/--
Computable form of the repeated-label invariant carried by a checked einsum.
-/
private def einsumRepeatedDimensionsValid (pattern : EinsumPattern)
    (inputShapes : List Shape) : Bool :=
  List.all₂
    (fun inputShape axes =>
      repeatedEinsumDimensionsAgree axes inputShape)
    inputShapes (einsumInputAxes pattern inputShapes)

/-- The Boolean repeated-dimension check is equivalent to its logical invariant. -/
private theorem einsumRepeatedDimensionsValid_eq_true_iff
    {pattern : EinsumPattern} {inputShapes : List Shape} :
    einsumRepeatedDimensionsValid pattern inputShapes = true ↔
      EinsumRepeatedDimensions pattern inputShapes := by
  simp [einsumRepeatedDimensionsValid, EinsumRepeatedDimensions]

/--
Computable form of the singleton-broadcasting invariant carried by a checked
einsum.
-/
private def einsumInputDimensionsValid (pattern : EinsumPattern)
    (inputShapes : List Shape) (axisLength : EinsumAxis → Nat) : Bool :=
  List.all₂
    (fun inputShape axes =>
      List.all₂
        (fun dimension axis =>
          dimension == axisLength axis || dimension == 1)
        inputShape axes)
    inputShapes (einsumInputAxes pattern inputShapes)

/-- The Boolean broadcast-dimension check is equivalent to its logical invariant. -/
private theorem einsumInputDimensionsValid_eq_true_iff
    {pattern : EinsumPattern} {inputShapes : List Shape}
    {axisLength : EinsumAxis → Nat} :
    einsumInputDimensionsValid pattern inputShapes axisLength = true ↔
      EinsumInputDimensions pattern inputShapes axisLength := by
  simp [einsumInputDimensionsValid, EinsumInputDimensions, Bool.or_eq_true]

/-- Removing duplicates under lawful equality always produces a duplicate-free list. -/
private theorem eraseDups_nodup {α : Type*} [BEq α] [LawfulBEq α]
    (items : List α) : items.eraseDups.Nodup := by
  cases items with
  | nil => simp
  | cons item items =>
      rw [List.eraseDups_cons]
      refine List.nodup_cons.mpr ⟨?_, ?_⟩
      · simp
      · exact
          eraseDups_nodup
            (items.filter fun other => !other == item)
termination_by items.length
decreasing_by
  apply Nat.lt_succ_of_le
  exact List.length_filter_le _ _

/--
An einsum call together with the invariants needed by tensor semantics and
lowering.

No normalized axes or shapes are stored independently: all are computed from
`pattern` and `inputShapes`, and these fields certify those computations.
-/
structure CheckedEinsum where
  /-- Parsed surface pattern accepted by the einsum checker. -/
  pattern : EinsumPattern
  /-- Physical operand shapes in source order. -/
  inputShapes : List Shape
  /-- Resolved length of each logical named or ellipsis axis. -/
  axisLength : EinsumAxis → Nat
  /-- The pattern contains exactly one input expression per operand shape. -/
  input_count : pattern.inputs.length = inputShapes.length
  /-- Every input and output expression belongs to the supported einsum grammar. -/
  surface_supported : einsumPatternSupported pattern = true
  /-- Repeated labels within one operand always index equal-sized dimensions. -/
  repeated_dimensions : EinsumRepeatedDimensions pattern inputShapes
  /-- Every physical dimension is its logical broadcast length or a singleton. -/
  input_dimensions :
    EinsumInputDimensions pattern inputShapes axisLength
  /-- Every non-singleton logical length is witnessed by an input occurrence. -/
  axis_lengths :
    einsumAxisLengths pattern inputShapes axisLength = true
  /-- Output labels are unique after ellipsis expansion. -/
  output_axes_nodup :
    (einsumOutputAxes pattern inputShapes).Nodup
  /-- Every output label is supplied by at least one input operand. -/
  output_axes_known :
    einsumAxesSubset
      (einsumOutputAxes pattern inputShapes)
      (einsumGlobalAxes pattern inputShapes) = true

namespace CheckedEinsum

/-- Expanded logical labels for every checked input tensor. -/
@[expose] def inputAxes (checked : CheckedEinsum) : List (List EinsumAxis) :=
  einsumInputAxes checked.pattern checked.inputShapes

/-- Expanded logical labels of the checked output tensor. -/
@[expose] def outputAxes (checked : CheckedEinsum) : List EinsumAxis :=
  einsumOutputAxes checked.pattern checked.inputShapes

/-- Distinct logical labels over which a complete assignment ranges. -/
@[expose] def globalAxes (checked : CheckedEinsum) : List EinsumAxis :=
  einsumGlobalAxes checked.pattern checked.inputShapes

/-- Global logical axes summed out because they are absent from the output. -/
@[expose] def contractedAxes (checked : CheckedEinsum) : List EinsumAxis :=
  checked.globalAxes.filter fun axis =>
    !checked.outputAxes.contains axis

/-- The global logical-axis list contains each label exactly once. -/
theorem global_axes_nodup (checked : CheckedEinsum) :
    checked.globalAxes.Nodup := by
  exact eraseDups_nodup _

/-- The inferred shape of the checked output tensor. -/
@[expose] def output (checked : CheckedEinsum) : Shape :=
  checked.outputAxes.map checked.axisLength

/-- Every checked output label belongs to the global input-label list. -/
theorem output_axis_mem_global (checked : CheckedEinsum) :
    ∀ ⦃axis⦄, axis ∈ checked.outputAxes → axis ∈ checked.globalAxes :=
  einsumAxesSubset_eq_true_iff.mp checked.output_axes_known

end CheckedEinsum

/-- Construct a source-located diagnostic for an unsupported einsum axis form. -/
private def unsupportedEinsumAxis
    (message : String) (span : Span) : Result Unit :=
  .error { code := .unsupportedEinsumAxis, message, span }

/-- Validate the restricted surface form of one physical einsum axis. -/
private def checkEinsumCompositeAxis (composite : CompositeAxis) : Result Unit :=
  match composite.semanticAxes with
  | [] =>
      unsupportedEinsumAxis
        "singleton and unit axes are not supported by einops einsum"
        composite.span
  | [axis] =>
      match axis.value with
      | .named _ => .ok ()
      | .anonymous _ _ =>
          unsupportedEinsumAxis
            "anonymous numeric axes are not supported by einops einsum"
            axis.span
      | .ellipsis =>
          if composite.parenthesized then
            unsupportedEinsumAxis
              "an einsum ellipsis must be a bare top-level axis"
              composite.span
          else
            .ok ()
      | .unit =>
          .error
            { code := .internalInvariant
              message := "a semantic einsum axis unexpectedly retained a unit axis"
              span := axis.span }
  | _ =>
      unsupportedEinsumAxis
        "grouped axes are not supported by einops einsum"
        composite.span

/-- Validate one einsum operand or output expression before shape checking. -/
private def checkEinsumExpression (expression : Expression) : Result Unit := do
  if expression.ellipsisCount ≤ 1 then
    for composite in expression.axes do
      checkEinsumCompositeAxis composite
  else
    .error
      { code := .invalidEllipsis
        message := "an einsum expression may contain only one ellipsis"
        span := expression.span }

/-- Validate every expression in an einsum pattern's surface syntax. -/
private def checkEinsumSurface (pattern : EinsumPattern) : Result Unit := do
  for input in pattern.inputs do
    checkEinsumExpression input
  checkEinsumExpression pattern.output

/-- Check each operand expression against its corresponding physical rank. -/
private def checkEinsumRanks :
    List Expression → List Shape → Nat → Result Unit
  | [], [], _ => .ok ()
  | expression :: expressions, inputShape :: inputShapes, operand => do
      if expression.ellipsisCount = 0 then
        if expression.axes.length = inputShape.length then
          checkEinsumRanks expressions inputShapes (operand + 1)
        else
          .error
            { code := .rankMismatch
              message :=
                s!"einsum operand {operand} has rank {inputShape.length}, " ++
                  s!"but its expression names {expression.axes.length} dimensions"
              span := expression.span }
      else if fixedEinsumRank expression ≤ inputShape.length then
        checkEinsumRanks expressions inputShapes (operand + 1)
      else
        .error
          { code := .rankMismatch
            message :=
              s!"einsum operand {operand} has rank {inputShape.length}, " ++
                s!"too small for {fixedEinsumRank expression} explicit dimensions"
            span := expression.span }
  | _, _, _ =>
      .error
        { code := .internalInvariant
          message := "einsum rank checking received different operand and shape counts"
          span := ⟨0, 0⟩ }

/--
Find the first repeated label whose two physical occurrences have different
lengths.
-/
private def firstRepeatedDimensionMismatch? :
    List (EinsumAxis × Nat) → Option (EinsumAxis × Nat × Nat)
  | [] => none
  | occurrence :: occurrences =>
      match occurrences.find? fun other =>
          other.1 == occurrence.1 && other.2 != occurrence.2 with
      | some other => some (occurrence.1, occurrence.2, other.2)
      | none => firstRepeatedDimensionMismatch? occurrences

/-- Validate repeated-label diagonal dimensions independently for each operand. -/
private def checkRepeatedEinsumDimensions :
    List Expression → List (List EinsumAxis) → List Shape → Nat → Result Unit
  | [], [], [], _ => .ok ()
  | expression :: expressions, axes :: remainingAxes,
      inputShape :: inputShapes, operand =>
      match firstRepeatedDimensionMismatch? (axes.zip inputShape) with
      | some (axis, firstLength, secondLength) =>
          .error
            { code := .repeatedEinsumDimensionMismatch
              message :=
                s!"einsum operand {operand} repeats {axis.description} with " ++
                  s!"different lengths {firstLength} and {secondLength}"
              span := expression.span }
      | none =>
          checkRepeatedEinsumDimensions
            expressions remainingAxes inputShapes (operand + 1)
  | _, _, _, _ =>
      .error
        { code := .internalInvariant
          message := "repeated-label checking received inconsistent operand data"
          span := ⟨0, 0⟩ }

/--
Find the first input occurrence that is neither singleton nor the resolved
logical-axis length.
-/
private def firstBroadcastMismatch? (axisLength : EinsumAxis → Nat) :
    List (EinsumAxis × Nat) → Option (EinsumAxis × Nat × Nat)
  | [] => none
  | occurrence :: occurrences =>
      let expected := axisLength occurrence.1
      if occurrence.2 = expected ∨ occurrence.2 = 1 then
        firstBroadcastMismatch? axisLength occurrences
      else
        some (occurrence.1, occurrence.2, expected)

/-- Find the first output label absent from every input expression. -/
private def firstUnknownOutputAxis? (pattern : EinsumPattern) :
    Option (Located String) :=
  let inputNames :=
    pattern.inputs.flatMap fun expression =>
      expression.namedAxes.map Located.value
  pattern.output.namedAxes.find? fun output =>
    !inputNames.contains output.value

/--
Check an einsum pattern against the concrete shapes of all input tensors.

Errors identify operand-count, surface, rank, repeated-label, output-label,
and broadcast failures separately. A successful value carries the facts used
by the independent contraction semantics.
-/
def checkEinsum (pattern : EinsumPattern)
    (inputShapes : List Shape) : Result CheckedEinsum := do
  if hCount : pattern.inputs.length = inputShapes.length then
    checkEinsumSurface pattern
    checkEinsumRanks pattern.inputs inputShapes 0
    let inputAxes := einsumInputAxes pattern inputShapes
    checkRepeatedEinsumDimensions pattern.inputs inputAxes inputShapes 0
    let outputNames := pattern.output.namedAxes.map Located.value
    if !outputNames.Nodup then
      .error
        { code := .duplicateEinsumOutputAxis
          message := "an einsum output label may occur only once"
          span := pattern.output.span }
    else
      match firstUnknownOutputAxis? pattern with
      | some unknown =>
          .error
            { code := .unknownEinsumOutputAxis
              message :=
                s!"output axis '{unknown.value}' does not occur in any einsum input"
              span := unknown.span }
      | none =>
          let axisLength := einsumAxisLength pattern inputShapes
          match
              firstBroadcastMismatch? axisLength
                (einsumAxisOccurrences pattern inputShapes) with
          | some (axis, actual, expected) =>
            .error
              { code := .incompatibleEinsumBroadcast
                message :=
                  s!"{axis.description} has incompatible lengths " ++
                    s!"{actual} and {expected}; one must be singleton"
                span := pattern.span }
          | none =>
              if hSurface : einsumPatternSupported pattern = true then
                if hRepeatedValid :
                    einsumRepeatedDimensionsValid pattern inputShapes = true then
                  let hRepeated :=
                    einsumRepeatedDimensionsValid_eq_true_iff.mp hRepeatedValid
                  if hDimensionsValid :
                      einsumInputDimensionsValid pattern inputShapes
                        axisLength = true then
                    let hDimensions :=
                      einsumInputDimensionsValid_eq_true_iff.mp hDimensionsValid
                    if hAxisLengths :
                        einsumAxisLengths pattern inputShapes axisLength = true then
                      if hOutputNodup :
                          (einsumOutputAxes pattern inputShapes).Nodup then
                        if hOutputKnown :
                            einsumAxesSubset
                              (einsumOutputAxes pattern inputShapes)
                              (einsumGlobalAxes pattern inputShapes) = true then
                          .ok
                            { pattern
                              inputShapes
                              axisLength
                              input_count := hCount
                              surface_supported := hSurface
                              repeated_dimensions := hRepeated
                              input_dimensions := hDimensions
                              axis_lengths := hAxisLengths
                              output_axes_nodup := hOutputNodup
                              output_axes_known := hOutputKnown }
                        else
                          .error
                            { code := .internalInvariant
                              message :=
                                "a validated einsum output contains an unknown expanded axis"
                              span := pattern.output.span }
                      else
                        .error
                          { code := .internalInvariant
                            message :=
                              "validated einsum output axes are unexpectedly duplicated"
                            span := pattern.output.span }
                    else
                      .error
                        { code := .internalInvariant
                          message :=
                            "validated einsum axis lengths lack an input source"
                          span := pattern.span }
                  else
                    .error
                      { code := .internalInvariant
                        message :=
                          "validated einsum dimensions do not satisfy broadcasting"
                        span := pattern.span }
                else
                  .error
                    { code := .internalInvariant
                      message :=
                        "validated repeated einsum labels disagree in dimension"
                      span := pattern.span }
              else
                .error
                  { code := .internalInvariant
                    message := "validated einsum syntax is marked unsupported"
                    span := pattern.span }
  else
    .error
      { code := .einsumOperandCountMismatch
        message :=
          s!"einsum pattern describes {pattern.inputs.length} operands, " ++
            s!"but {inputShapes.length} tensor shapes were supplied"
        span := pattern.span }

end TorchLean.Tensor.Internal.Check
