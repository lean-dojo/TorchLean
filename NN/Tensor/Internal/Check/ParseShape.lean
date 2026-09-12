/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Tensor.Internal.Check.Transform
public import NN.Tensor.Internal.Check.Diagnostic -- shake: keep
public import NN.Tensor.Internal.Syntax.Ast -- shake: keep

/-!
# Checked `parse_shape`

`parse_shape` matches one pattern position to each physical tensor dimension
and returns the lengths of named positions in source order. An underscore or
an ellipsis-expanded position is skipped. Unit and positive anonymous axes
are accepted only when their literal length agrees with the corresponding
physical dimension.

The checker returns the existing ordered `SupplementaryLengths` type. No
separate plan structure is needed because this operation neither transforms
tensor values nor carries data into a later lowering stage.
-/

@[expose] public section

namespace TorchLean.Tensor.Internal.Check

open TorchLean.Tensor.Internal.Syntax

namespace ParseShape

/--
Expand the unique ellipsis to one skipped position per unmatched physical
dimension. A `none` entry records a physical dimension hidden by the
ellipsis; every other top-level pattern position is retained verbatim.

The concrete checker establishes the structural and rank preconditions under
which this list has exactly `inputRank` entries.
-/
def expandAxes (expression : Expression) (inputRank : Nat) :
    List (Option CompositeAxis) :=
  let ellipsisRank := inputRank - (expression.axes.length - 1)
  expression.axes.flatMap fun axis =>
    if axis.hasEllipsis then
      List.replicate ellipsisRank none
    else
      [some axis]

/--
Extract the named length contributed by one expanded pattern position.
Underscores, literals, units, and ellipsis-expanded positions contribute no
binding.
-/
def binding? (patternAxis : Option CompositeAxis) (dimension : Nat) :
    Option (String × Nat) :=
  match patternAxis with
  | none => none
  | some axis =>
      match axis.semanticAxes with
      | [item] =>
          match item.value with
          | .named name => if name == "_" then none else some (name, dimension)
          | _ => none
      | _ => none

/-- A returned binding always carries its paired physical dimension. -/
theorem binding?_eq_some_length {patternAxis : Option CompositeAxis}
    {dimension : Nat} {name : String} {length : Nat}
    (h : binding? patternAxis dimension = some (name, length)) :
    dimension = length := by
  cases patternAxis with
  | none => simp [binding?] at h
  | some axis =>
      cases hAxes : axis.semanticAxes with
      | nil => simp [binding?, hAxes] at h
      | cons item items =>
          cases items with
          | cons second rest => simp [binding?, hAxes] at h
          | nil =>
              cases hItem : item.value with
              | named axisName =>
                  by_cases hUnderscore : axisName == "_"
                  · simp [binding?, hAxes, hItem, hUnderscore] at h
                  · have hMapped := congrArg (Option.map Prod.snd) h
                    simpa [binding?, hAxes, hItem, hUnderscore] using hMapped
              | anonymous value occurrence => simp [binding?, hAxes, hItem] at h
              | unit => simp [binding?, hAxes, hItem] at h
              | ellipsis => simp [binding?, hAxes, hItem] at h

/--
The declarative dimension condition for one expanded parse-shape position.
Named axes and wildcards accept any dimension; units and numeric axes must
match their literal lengths.
-/
def AxisMatches (patternAxis : Option CompositeAxis) (dimension : Nat) : Prop :=
  match patternAxis with
  | none => True
  | some axis =>
      match axis.semanticAxes with
      | [] => dimension = 1
      | [item] =>
          match item.value with
          | .named _ => True
          | .anonymous value _ => dimension = value
          | .unit => dimension = 1
          | .ellipsis => False
      | _ => False

end ParseShape

namespace ParseShape.Impl

/-- Check one expanded parse-shape axis against its physical dimension. -/
def validateAxis (patternAxis : Option CompositeAxis)
    (dimension : Nat) : Result Unit :=
  match patternAxis with
  | none => .ok ()
  | some axis =>
      match axis.semanticAxes with
      | [] =>
          if dimension = 1 then
            .ok ()
          else
            .error
              { code := .dimensionMismatch
                message :=
                  s!"physical dimension {dimension} does not match unit axis length 1"
                span := axis.span }
      | [item] =>
          match item.value with
          | .named _ => .ok ()
          | .anonymous value _ =>
              if dimension = value then
                .ok ()
              else
                .error
                  { code := .dimensionMismatch
                    message :=
                      s!"physical dimension {dimension} does not match " ++
                        s!"anonymous axis length {value}"
                    span := item.span }
          | .unit =>
              if dimension = 1 then
                .ok ()
              else
                .error
                  { code := .dimensionMismatch
                    message :=
                      s!"physical dimension {dimension} does not match unit axis length 1"
                    span := item.span }
          | .ellipsis =>
              .error
                { code := .internalInvariant
                  message := "parse-shape ellipsis was not expanded"
                  span := item.span }
      | _ =>
          .error
            { code := .compositeParseShapeAxis
              message := "parse_shape does not permit composite axes"
              span := axis.span }

/-- Validate expanded parse-shape axes and physical dimensions in lockstep. -/
def validateAxes :
    List (Option CompositeAxis) → Shape → Result Unit
  | [], [] => .ok ()
  | patternAxis :: patternAxes, dimension :: dimensions =>
      match validateAxis patternAxis dimension with
      | .error diagnostic => .error diagnostic
      | .ok () => validateAxes patternAxes dimensions
  | _, _ =>
      .error
        { code := .internalInvariant
          message := "expanded parse-shape pattern and tensor shape have different ranks"
          span := ⟨0, 0⟩ }

/-- Successful validation of one expanded axis establishes its dimension match. -/
private theorem validateAxis_sound {patternAxis : Option CompositeAxis}
    {dimension : Nat} (h : validateAxis patternAxis dimension = .ok ()) :
    ParseShape.AxisMatches patternAxis dimension := by
  cases patternAxis with
  | none => trivial
  | some axis =>
      cases hAxes : axis.semanticAxes with
      | nil =>
          simp [validateAxis, hAxes] at h
          simpa [ParseShape.AxisMatches, hAxes] using h
      | cons item items =>
          cases items with
          | cons second rest =>
              simp [validateAxis, hAxes] at h
          | nil =>
              cases hItem : item.value with
              | named name =>
                  simp [ParseShape.AxisMatches, hAxes, hItem]
              | anonymous value occurrence =>
                  simp [validateAxis, hAxes, hItem] at h
                  simpa [ParseShape.AxisMatches, hAxes, hItem] using h
              | unit =>
                  simp [validateAxis, hAxes, hItem] at h
                  simpa [ParseShape.AxisMatches, hAxes, hItem] using h
              | ellipsis =>
                  simp [validateAxis, hAxes, hItem] at h

/-- Successful lockstep validation establishes the match invariant for every axis. -/
private theorem validateAxes_sound {patternAxes : List (Option CompositeAxis)}
    {inputShape : Shape} (h : validateAxes patternAxes inputShape = .ok ()) :
    List.Forall₂ ParseShape.AxisMatches patternAxes inputShape := by
  induction patternAxes generalizing inputShape with
  | nil =>
      cases inputShape with
      | nil => exact .nil
      | cons dimension dimensions => simp [validateAxes] at h
  | cons patternAxis patternAxes ih =>
      cases inputShape with
      | nil => simp [validateAxes] at h
      | cons dimension dimensions =>
          cases hAxis : validateAxis patternAxis dimension with
          | error diagnostic =>
              simp [validateAxes, hAxis] at h
          | ok value =>
              cases value
              have hRest : validateAxes patternAxes dimensions = .ok () := by
                simpa [validateAxes, hAxis] using h
              exact .cons (validateAxis_sound hAxis) (ih hRest)

end ParseShape.Impl

/--
Check a parsed `parse_shape` expression against a concrete tensor shape.

The result contains exactly the named, non-underscore axes in pattern order.
The checker rejects genuine composite axes, malformed ellipses, rank
mismatches, literal-length mismatches, and duplicate returned names.
-/
def checkParseShape (expression : Expression) (inputShape : Shape) :
    Result SupplementaryLengths :=
  if expression.hasCompositeAxes then
    .error
      { code := .compositeParseShapeAxis
        message := "parse_shape does not permit composite axes"
        span := expression.span }
  else if expression.ellipsisCount > 1 then
    .error
      { code := .invalidEllipsis
        message := "a parse_shape expression may contain only one ellipsis"
        span := expression.span }
  else if expression.hasParenthesizedEllipsis then
    .error
      { code := .invalidEllipsis
        message := "the parse_shape ellipsis may not be parenthesized"
        span := expression.span }
  else if expression.ellipsisCount = 0 &&
      expression.axes.length != inputShape.length then
    .error
      { code := .rankMismatch
        message := "tensor rank does not match the parse_shape pattern"
        span := expression.span }
  else if expression.ellipsisCount = 1 &&
      inputShape.length < expression.axes.length - 1 then
    .error
      { code := .rankMismatch
        message := "tensor rank is too small for the parse_shape ellipsis"
        span := expression.span }
  else
    let expanded := ParseShape.expandAxes expression inputShape.length
    match ParseShape.Impl.validateAxes expanded inputShape with
    | .error diagnostic => .error diagnostic
    | .ok () =>
        let bindings :=
          (expanded.zip inputShape).filterMap fun item =>
            ParseShape.binding? item.1 item.2
        if (SupplementaryLengths.names bindings).Nodup then
          .ok bindings
        else
          .error
            { code := .duplicateParseShapeAxis
              message := "parse_shape would return the same axis name more than once"
              span := expression.span }

/-- A successful check exposes validation, exact bindings, and unique names. -/
private theorem checkParseShape_success {expression : Expression}
    {inputShape : Shape} {bindings : SupplementaryLengths}
    (h : checkParseShape expression inputShape = .ok bindings) :
    ParseShape.Impl.validateAxes
        (ParseShape.expandAxes expression inputShape.length) inputShape =
        .ok () ∧
      bindings =
        ((ParseShape.expandAxes expression inputShape.length).zip inputShape).filterMap
          (fun item => ParseShape.binding? item.1 item.2) ∧
      (SupplementaryLengths.names bindings).Nodup := by
  unfold checkParseShape at h
  split at h <;> try contradiction
  split at h <;> try contradiction
  split at h <;> try contradiction
  split at h <;> try contradiction
  split at h <;> try contradiction
  dsimp only at h
  cases hValidation :
      ParseShape.Impl.validateAxes
        (ParseShape.expandAxes expression inputShape.length) inputShape with
  | error diagnostic =>
      rw [hValidation] at h
      contradiction
  | ok value =>
      cases value
      rw [hValidation] at h
      dsimp only at h
      split at h
      · rename_i hUnique
        simp only [Except.ok.injEq] at h
        exact ⟨rfl, h.symm, h ▸ hUnique⟩
      · contradiction

/--
A successful result is the left-to-right `filterMap` of the expanded pattern
zipped with physical dimensions. This states both the returned order and the
fact that ellipsis and wildcard positions are skipped rather than reordered.
-/
theorem checkParseShape_ordered_bindings {expression : Expression}
    {inputShape : Shape} {bindings : SupplementaryLengths}
    (h : checkParseShape expression inputShape = .ok bindings) :
    bindings =
      ((ParseShape.expandAxes expression inputShape.length).zip inputShape).filterMap
        fun item => ParseShape.binding? item.1 item.2 := by
  exact (checkParseShape_success h).2.1

/-- Successful parse-shape results never contain a duplicate axis name. -/
theorem checkParseShape_names_nodup {expression : Expression}
    {inputShape : Shape} {bindings : SupplementaryLengths}
    (h : checkParseShape expression inputShape = .ok bindings) :
    bindings.names.Nodup := by
  exact (checkParseShape_success h).2.2

/--
Every successful pattern position satisfies its declarative dimension
condition. In particular, unit and anonymous numeric axes agree with the
physical shape, including dimensions of length zero where allowed.
-/
theorem checkParseShape_axes_match {expression : Expression}
    {inputShape : Shape} {bindings : SupplementaryLengths}
    (h : checkParseShape expression inputShape = .ok bindings) :
    List.Forall₂ ParseShape.AxisMatches
      (ParseShape.expandAxes expression inputShape.length) inputShape := by
  exact ParseShape.Impl.validateAxes_sound (checkParseShape_success h).1

/--
Each returned `(name, length)` is obtained from a named expanded pattern
position paired with that exact physical tensor dimension.
-/
theorem checkParseShape_dimension_agreement {expression : Expression}
    {inputShape : Shape} {bindings : SupplementaryLengths}
    (h : checkParseShape expression inputShape = .ok bindings)
    {name : String} {length : Nat} (hBinding : (name, length) ∈ bindings) :
    ∃ patternAxis,
      (patternAxis, length) ∈
        (ParseShape.expandAxes expression inputShape.length).zip inputShape ∧
      ParseShape.binding? patternAxis length = some (name, length) := by
  rw [checkParseShape_ordered_bindings h] at hBinding
  simp only [List.mem_filterMap] at hBinding
  obtain ⟨⟨patternAxis, dimension⟩, hPosition, hValue⟩ := hBinding
  have hLength : dimension = length :=
    ParseShape.binding?_eq_some_length hValue
  subst dimension
  exact ⟨patternAxis, hPosition, hValue⟩

end TorchLean.Tensor.Internal.Check
