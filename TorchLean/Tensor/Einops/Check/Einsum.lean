/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Tensor.Einops.Check.Diagnostic
public import TorchLean.Tensor.Einops.Check.NormalForm
public import TorchLean.Tensor.Shape.Coord
public import TorchLean.Tensor.Einops.Syntax.Einsum -- shake: keep
public import TorchLean.Tensor.Einops.Syntax.Pack -- shake: keep
public import TorchLean.Tensor.Einops.Syntax.Transform -- shake: keep

/-!
# Einsum plans and executable checking

`EinsumPattern.check` checks a named-axis einsum pattern against finite tensor shapes and returns a
`Einsum`, the data and invariants that tensor semantics and lowering use. The checker
follows the shape behavior of einops v0.8.2:

* an input ellipsis expands to the dimensions that the named labels leave over,
* ellipsis dimensions are right-aligned between operands,
* dimensions attached to the same global label broadcast when either is one,
* repeated labels inside one operand must have equal dimensions, and they denote a diagonal,
* output labels must be distinct and must occur in an input.

The mathematical semantics accepts any finite number of named labels. The solitary name `_` is an
ordinary einsum label here, while `parse_shape` reads it as the wildcard for a skipped axis.

`Einsum` stores the parsed pattern, the input shapes and one total function giving the
length of each logical axis. Its invariants prove that each physical dimension is that resolved
length or a singleton, and that every resolved length other than one is the dimension of some
occurrence of the axis. The einsum elaborator builds the same structure for symbolic `Nat`
dimensions.

## Reference

The surface restrictions follow einops v0.8.2, pinned at commit
`8e911db71f2e693a0c434b041180388c685ed06f`. Its cap of 52 distinct labels comes from
`einops.einops._compactify_pattern_for_einsum`, which encodes labels as ASCII letters. The cap
belongs to that encoding, so nothing here enforces it. The Python front end leaves broadcasting
to the tensor backends, which is why the checker states the broadcast rule itself.
-/

public section

namespace TorchLean.Einops

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
Whether an einsum expression uses only the axis forms that einops v0.8.2 supports.

Empty expressions represent scalar tensors and are accepted. Every top-level position must hold
one named label or one bare ellipsis, with at most one ellipsis in the expression. Parentheses
around a single named label are harmless, as in the reference parser, but genuine compositions,
unit axes and numeric axes are rejected.
-/
@[expose, implicit_reducible]
def Expression.isEinsumSupported (expression : Expression) : Bool :=
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
def EinsumPattern.isSupported (pattern : EinsumPattern) : Bool :=
  pattern.inputs.all Expression.isEinsumSupported &&
    Expression.isEinsumSupported pattern.output

/-- The number of non-ellipsis tensor dimensions written in an expression. -/
@[expose, implicit_reducible]
def Expression.explicitRank (expression : Expression) : Nat :=
  expression.axes.length - expression.ellipsisCount

/--
The number of physical dimensions captured by one input ellipsis.

Rank checking proves that the subtraction is exact. For an expression without
an ellipsis the value is zero.
-/
@[expose, implicit_reducible]
def Expression.ellipsisRank (expression : Expression) (inputShape : Shape) : Nat :=
  if expression.ellipsisCount = 0 then
    0
  else
    inputShape.length - explicitRank expression

/--
The common ellipsis rank of an einsum call.

It is the largest rank captured by any operand ellipsis, and shorter ellipses are right-aligned
to it.
-/
@[expose, implicit_reducible]
def EinsumPattern.ellipsisRank (pattern : EinsumPattern) (inputShapes : List Shape) : Nat :=
  ((pattern.inputs.zip inputShapes).map fun input =>
      Expression.ellipsisRank input.1 input.2).foldl Nat.max 0

/-- Common ellipsis slots occupied by an operand with the given local rank. -/
@[expose, implicit_reducible]
def EinsumAxis.ellipsisSlots (commonRank operandRank : Nat) : List EinsumAxis :=
  (List.range operandRank).map fun index =>
    .ellipsis (commonRank - operandRank + index)

/-- Expanding an ellipsis never introduces duplicate logical axes. -/
theorem EinsumAxis.ellipsisSlots_nodup (commonRank operandRank : Nat) :
    (ellipsisSlots commonRank operandRank).Nodup := by
  apply List.nodup_range.map
  intro firstIndex secondIndex hEqual
  exact Nat.add_left_cancel (EinsumAxis.ellipsis.inj hEqual)

/--
Expand one supported expression to logical labels.

`ellipsisRank` is the local captured rank for an input and the common rank for the output.
Unsupported composites contribute no labels, a case that `surface_supported` excludes for checked
patterns.
-/
@[expose, implicit_reducible]
def Expression.einsumAxes (commonRank ellipsisRank : Nat)
    (expression : Expression) : List EinsumAxis :=
  expression.axes.flatMap fun composite =>
    match composite.semanticAxes with
    | [axis] =>
        match axis.value with
        | .named name => [.named name]
        | .ellipsis => EinsumAxis.ellipsisSlots commonRank ellipsisRank
        | .anonymous _ _ | .unit => []
    | _ => []

/-- Expanded logical labels for every input operand. -/
@[expose, implicit_reducible]
def EinsumPattern.inputAxes (pattern : EinsumPattern)
    (inputShapes : List Shape) : List (List EinsumAxis) :=
  let commonRank := ellipsisRank pattern inputShapes
  (pattern.inputs.zip inputShapes).map fun input =>
    Expression.einsumAxes commonRank
      (Expression.ellipsisRank input.1 input.2) input.1

/-- Expanded logical labels of the output tensor. -/
@[expose, implicit_reducible]
def EinsumPattern.outputAxes (pattern : EinsumPattern)
    (inputShapes : List Shape) : List EinsumAxis :=
  let commonRank := ellipsisRank pattern inputShapes
  Expression.einsumAxes commonRank commonRank pattern.output

/--
All logical labels in first-occurrence order.

Repeated input labels are collapsed here, but remain present in each
operand's expanded list where they impose diagonal indexing.
-/
@[expose, implicit_reducible]
def EinsumPattern.globalAxes (pattern : EinsumPattern)
    (inputShapes : List Shape) : List EinsumAxis :=
  (inputAxes pattern inputShapes).flatten.eraseDups

/-- Every physical input dimension paired with its expanded logical label. -/
@[expose, implicit_reducible]
def EinsumPattern.axisOccurrences (pattern : EinsumPattern)
    (inputShapes : List Shape) : List (EinsumAxis × Nat) :=
  ((inputAxes pattern inputShapes).zip inputShapes).flatMap fun input =>
    input.1.zip input.2

/--
The broadcast dimension of one logical label.

The first non-singleton occurrence determines the result, and the result is one when every
occurrence is singleton. Compatibility checking proves that every other occurrence is singleton
or equal to this dimension. A zero dimension next to singleton dimensions therefore resolves to
zero.
-/
@[expose, implicit_reducible]
def EinsumPattern.axisLength (pattern : EinsumPattern) (inputShapes : List Shape)
    (axis : EinsumAxis) : Nat :=
  match (axisOccurrences pattern inputShapes).find? fun occurrence =>
      occurrence.1 == axis && occurrence.2 != 1 with
  | some occurrence => occurrence.2
  | none => 1

/-- The concrete output, in the axis order written by the user. -/
@[expose, implicit_reducible]
def EinsumPattern.outputShape (pattern : EinsumPattern)
    (inputShapes : List Shape) : Shape :=
  (outputAxes pattern inputShapes).map
    (axisLength pattern inputShapes)

/--
Whether repeated occurrences of a label inside one operand have equal
physical dimensions.
-/
@[expose, implicit_reducible]
def EinsumPattern.repeatedDimensionsAgree
    (axes : List EinsumAxis) (inputShape : Shape) : Bool :=
  let occurrences := axes.zip inputShape
  occurrences.all fun left =>
    occurrences.all fun right =>
      left.1 != right.1 || left.2 == right.2

/--
The Boolean repeated-label check is exactly pairwise equality of dimensions
for equal labels.
-/
theorem EinsumPattern.repeatedDimensionsAgree_iff
    {axes : List EinsumAxis} {inputShape : Shape} :
    repeatedDimensionsAgree axes inputShape = true ↔
      ∀ left ∈ axes.zip inputShape, ∀ right ∈ axes.zip inputShape,
        left.1 = right.1 → left.2 = right.2 := by
  simp only [repeatedDimensionsAgree, List.all_eq_true, Bool.or_eq_true,
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

/--
Every physical input dimension is a singleton or the resolved length of its logical axis.
-/
abbrev EinsumPattern.InputDimensions (pattern : EinsumPattern)
    (inputShapes : List Shape) (axisLength : EinsumAxis → Nat) : Prop :=
  List.Forall₂
    (fun inputShape axes =>
      List.Forall₂
        (fun dimension axis =>
          dimension = axisLength axis ∨ dimension = 1)
        inputShape axes)
    inputShapes (inputAxes pattern inputShapes)

/--
Every global axis has resolved length one or the dimension of one of its occurrences.

Together with `InputDimensions`, the check forces length one on an axis whose occurrences
are all singleton, so no larger broadcast result can be invented for it.
-/
@[expose, implicit_reducible]
def EinsumPattern.axisLengthsJustified (pattern : EinsumPattern)
    (inputShapes : List Shape) (axisLength : EinsumAxis → Nat) : Bool :=
  (globalAxes pattern inputShapes).all fun axis =>
    axisLength axis == 1 ||
      (axisOccurrences pattern inputShapes).any fun occurrence =>
        occurrence.1 == axis && occurrence.2 == axisLength axis

/-- In every operand, repeated labels have equal dimensions. -/
abbrev EinsumPattern.RepeatedDimensions (pattern : EinsumPattern)
    (inputShapes : List Shape) : Prop :=
  List.Forall₂
    (fun inputShape axes =>
      repeatedDimensionsAgree axes inputShape = true)
    inputShapes (inputAxes pattern inputShapes)

/--
Computable form of the repeated-label invariant carried by a checked einsum.
-/
private def EinsumPattern.repeatedDimensionsValid (pattern : EinsumPattern)
    (inputShapes : List Shape) : Bool :=
  List.all₂
    (fun inputShape axes =>
      repeatedDimensionsAgree axes inputShape)
    inputShapes (inputAxes pattern inputShapes)

/-- The Boolean repeated-dimension check is equivalent to its logical invariant. -/
private theorem EinsumPattern.repeatedDimensionsValid_iff
    {pattern : EinsumPattern} {inputShapes : List Shape} :
    repeatedDimensionsValid pattern inputShapes = true ↔
      RepeatedDimensions pattern inputShapes := by
  simp [repeatedDimensionsValid, RepeatedDimensions]

/--
Computable form of the singleton-broadcasting invariant carried by a checked
einsum.
-/
private def EinsumPattern.inputDimensionsValid (pattern : EinsumPattern)
    (inputShapes : List Shape) (axisLength : EinsumAxis → Nat) : Bool :=
  List.all₂
    (fun inputShape axes =>
      List.all₂
        (fun dimension axis =>
          dimension == axisLength axis || dimension == 1)
        inputShape axes)
    inputShapes (inputAxes pattern inputShapes)

/-- The Boolean broadcast-dimension check is equivalent to its logical invariant. -/
private theorem EinsumPattern.inputDimensionsValid_iff
    {pattern : EinsumPattern} {inputShapes : List Shape}
    {axisLength : EinsumAxis → Nat} :
    inputDimensionsValid pattern inputShapes axisLength = true ↔
      InputDimensions pattern inputShapes axisLength := by
  simp [inputDimensionsValid, InputDimensions, Bool.or_eq_true]

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
An einsum call together with the invariants needed by tensor semantics and lowering.

Expanded axes and the output shape are computed from the stored `pattern`, `inputShapes` and
`axisLength`, and the proof fields certify those computations.
-/
structure Einsum where
  /-- Parsed surface pattern accepted by the einsum checker. -/
  pattern : EinsumPattern
  /-- Physical operand shapes in source order. -/
  inputShapes : List Shape
  /-- Resolved length of each logical named or ellipsis axis. -/
  axisLength : EinsumAxis → Nat
  /-- The pattern contains exactly one input expression per operand shape. -/
  input_count : pattern.inputs.length = inputShapes.length
  /-- Every input and output expression belongs to the supported einsum grammar. -/
  surface_supported : EinsumPattern.isSupported pattern = true
  /-- Repeated labels within one operand always index equal-sized dimensions. -/
  repeated_dimensions : EinsumPattern.RepeatedDimensions pattern inputShapes
  /-- Every physical dimension is its logical broadcast length or a singleton. -/
  input_dimensions :
    EinsumPattern.InputDimensions pattern inputShapes axisLength
  /-- Every non-singleton logical length is witnessed by an input occurrence. -/
  axis_lengths :
    EinsumPattern.axisLengthsJustified pattern inputShapes axisLength = true
  /-- Output labels are unique after ellipsis expansion. -/
  output_axes_nodup :
    (EinsumPattern.outputAxes pattern inputShapes).Nodup
  /-- Every output label is supplied by at least one input operand. -/
  output_axes_known :
    axesSubset
      (EinsumPattern.outputAxes pattern inputShapes)
      (EinsumPattern.globalAxes pattern inputShapes) = true

namespace Einsum

/-- Expanded logical labels for every checked input tensor. -/
@[expose] def inputAxes (checked : Einsum) : List (List EinsumAxis) :=
  EinsumPattern.inputAxes checked.pattern checked.inputShapes

/-- Expanded logical labels of the checked output tensor. -/
@[expose] def outputAxes (checked : Einsum) : List EinsumAxis :=
  EinsumPattern.outputAxes checked.pattern checked.inputShapes

/-- Distinct logical labels over which a complete assignment ranges. -/
@[expose] def globalAxes (checked : Einsum) : List EinsumAxis :=
  EinsumPattern.globalAxes checked.pattern checked.inputShapes

/-- Global logical axes summed out because they are absent from the output. -/
@[expose] def contractedAxes (checked : Einsum) : List EinsumAxis :=
  checked.globalAxes.filter fun axis =>
    !checked.outputAxes.contains axis

/-- The global logical-axis list contains each label exactly once. -/
theorem global_axes_nodup (checked : Einsum) :
    checked.globalAxes.Nodup := by
  exact eraseDups_nodup _

/-- The inferred shape of the checked output tensor. -/
@[expose] def output (checked : Einsum) : Shape :=
  checked.outputAxes.map checked.axisLength

/-- Every checked output label belongs to the global input-label list. -/
theorem output_axis_mem_global (checked : Einsum) :
    ∀ ⦃axis⦄, axis ∈ checked.outputAxes → axis ∈ checked.globalAxes :=
  axesSubset_eq_true_iff.mp checked.output_axes_known

end Einsum

/-- Construct a source-located diagnostic for an unsupported einsum axis form. -/
private def EinsumPattern.unsupportedAxis
    (message : String) (span : Span) : Result Unit :=
  .error { code := .unsupportedEinsumAxis, message, span }

/-- Validate the restricted surface form of one physical einsum axis. -/
private def EinsumPattern.checkComposite (composite : CompositeAxis) : Result Unit :=
  match composite.semanticAxes with
  | [] =>
      unsupportedAxis
        "singleton and unit axes are not supported by einops einsum"
        composite.span
  | [axis] =>
      match axis.value with
      | .named _ => .ok ()
      | .anonymous _ _ =>
          unsupportedAxis
            "anonymous numeric axes are not supported by einops einsum"
            axis.span
      | .ellipsis =>
          if composite.parenthesized then
            unsupportedAxis
              "an einsum ellipsis must be a bare top-level axis"
              composite.span
          else
            .ok ()
      | .unit =>
          .error
            { code := .invariantFailed
              message := "a semantic einsum axis unexpectedly retained a unit axis"
              span := axis.span }
  | _ =>
      unsupportedAxis
        "grouped axes are not supported by einops einsum"
        composite.span

/-- Validate one einsum operand or output expression before shape checking. -/
private def EinsumPattern.checkExpression (expression : Expression) : Result Unit := do
  if expression.ellipsisCount ≤ 1 then
    for composite in expression.axes do
      checkComposite composite
  else
    .error
      { code := .invalidEllipsis
        message := "an einsum expression may contain only one ellipsis"
        span := expression.span }

/-- Validate every expression in an einsum pattern's surface syntax. -/
private def EinsumPattern.checkSurface (pattern : EinsumPattern) : Result Unit := do
  for input in pattern.inputs do
    checkExpression input
  checkExpression pattern.output

/-- Check each operand expression against its corresponding physical rank. -/
private def EinsumPattern.checkRanks :
    List Expression → List Shape → Nat → Result Unit
  | [], [], _ => .ok ()
  | expression :: expressions, inputShape :: inputShapes, operand => do
      if expression.ellipsisCount = 0 then
        if expression.axes.length = inputShape.length then
          checkRanks expressions inputShapes (operand + 1)
        else
          .error
            { code := .rankMismatch
              message :=
                s!"einsum operand {operand} has rank {inputShape.length}, " ++
                  s!"but its expression names {expression.axes.length} dimensions"
              span := expression.span }
      else if Expression.explicitRank expression ≤ inputShape.length then
        checkRanks expressions inputShapes (operand + 1)
      else
        .error
          { code := .rankMismatch
            message :=
              s!"einsum operand {operand} has rank {inputShape.length}, " ++
                s!"too small for {Expression.explicitRank expression} explicit dimensions"
            span := expression.span }
  | _, _, _ =>
      .error
        { code := .invariantFailed
          message := "einsum rank checking received different operand and shape counts"
          span := ⟨0, 0⟩ }

/--
Find the first repeated label whose two physical occurrences have different
lengths.
-/
private def EinsumPattern.firstRepeatedMismatch? :
    List (EinsumAxis × Nat) → Option (EinsumAxis × Nat × Nat)
  | [] => none
  | occurrence :: occurrences =>
      match occurrences.find? fun other =>
          other.1 == occurrence.1 && other.2 != occurrence.2 with
      | some other => some (occurrence.1, occurrence.2, other.2)
      | none => firstRepeatedMismatch? occurrences

/-- Validate repeated-label diagonal dimensions independently for each operand. -/
private def EinsumPattern.checkRepeatedDimensions :
    List Expression → List (List EinsumAxis) → List Shape → Nat → Result Unit
  | [], [], [], _ => .ok ()
  | expression :: expressions, axes :: remainingAxes,
      inputShape :: inputShapes, operand =>
      match firstRepeatedMismatch? (axes.zip inputShape) with
      | some (axis, firstLength, secondLength) =>
          .error
            { code := .repeatedEinsumAxisMismatch
              message :=
                s!"einsum operand {operand} repeats {axis.description} with " ++
                  s!"different lengths {firstLength} and {secondLength}"
              span := expression.span }
      | none =>
          checkRepeatedDimensions
            expressions remainingAxes inputShapes (operand + 1)
  | _, _, _, _ =>
      .error
        { code := .invariantFailed
          message := "repeated-label checking received inconsistent operand data"
          span := ⟨0, 0⟩ }

/--
Find the first input occurrence that is neither singleton nor the resolved
logical-axis length.
-/
private def EinsumPattern.firstBroadcastMismatch? (axisLength : EinsumAxis → Nat) :
    List (EinsumAxis × Nat) → Option (EinsumAxis × Nat × Nat)
  | [] => none
  | occurrence :: occurrences =>
      let expected := axisLength occurrence.1
      if occurrence.2 = expected ∨ occurrence.2 = 1 then
        firstBroadcastMismatch? axisLength occurrences
      else
        some (occurrence.1, occurrence.2, expected)

/-- Find the first output label absent from every input expression. -/
private def EinsumPattern.firstUnknownOutputAxis? (pattern : EinsumPattern) :
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
def EinsumPattern.check (pattern : EinsumPattern)
    (inputShapes : List Shape) : Result Einsum := do
  if hCount : pattern.inputs.length = inputShapes.length then
    checkSurface pattern
    checkRanks pattern.inputs inputShapes 0
    let inputAxes := EinsumPattern.inputAxes pattern inputShapes
    checkRepeatedDimensions pattern.inputs inputAxes inputShapes 0
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
          let axisLength := EinsumPattern.axisLength pattern inputShapes
          match
              firstBroadcastMismatch? axisLength
                (axisOccurrences pattern inputShapes) with
          | some (axis, actual, expected) =>
            .error
              { code := .incompatibleEinsumBroadcast
                message :=
                  s!"{axis.description} has incompatible lengths " ++
                    s!"{actual} and {expected}; one must be singleton"
                span := pattern.span }
          | none =>
              if hSurface : isSupported pattern = true then
                if hRepeatedValid :
                    repeatedDimensionsValid pattern inputShapes = true then
                  let hRepeated :=
                    repeatedDimensionsValid_iff.mp hRepeatedValid
                  if hDimensionsValid :
                      inputDimensionsValid pattern inputShapes
                        axisLength = true then
                    let hDimensions :=
                      inputDimensionsValid_iff.mp hDimensionsValid
                    if hAxisLengths :
                        axisLengthsJustified pattern inputShapes axisLength = true then
                      if hOutputNodup :
                          (outputAxes pattern inputShapes).Nodup then
                        if hOutputKnown :
                            axesSubset
                              (outputAxes pattern inputShapes)
                              (globalAxes pattern inputShapes) = true then
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
                            { code := .invariantFailed
                              message :=
                                "a validated einsum output contains an unknown expanded axis"
                              span := pattern.output.span }
                      else
                        .error
                          { code := .invariantFailed
                            message :=
                              "validated einsum output axes are unexpectedly duplicated"
                            span := pattern.output.span }
                    else
                      .error
                        { code := .invariantFailed
                          message :=
                            "validated einsum axis lengths lack an input source"
                          span := pattern.span }
                  else
                    .error
                      { code := .invariantFailed
                        message :=
                          "validated einsum dimensions do not satisfy broadcasting"
                        span := pattern.span }
                else
                  .error
                    { code := .invariantFailed
                      message :=
                        "validated repeated einsum labels disagree in dimension"
                      span := pattern.span }
              else
                .error
                  { code := .invariantFailed
                    message := "validated einsum syntax is marked unsupported"
                    span := pattern.span }
  else
    .error
      { code := .einsumOperandCountMismatch
        message :=
          s!"einsum pattern describes {pattern.inputs.length} operands, " ++
            s!"but {inputShapes.length} tensor shapes were supplied"
        span := pattern.span }

end TorchLean.Einops
