/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Tensor.Internal.Check.Normalize

/-!
# Transformation plans and executable checking

This module resolves elementary-axis lengths for `rearrange`, `repeat`, and
`reduce`. A composite input dimension may contain at most one axis whose
length is not supplied or fixed by an anonymous numeric axis. The remaining
length is inferred by exact division.

`TransformPlan` stores only the mathematical data consumed by semantics and
lowering: a normalized pattern, a total axis-length function, and the physical
output shape. Supplementary user input and partial assignments belong to the
concrete checking algorithm and deliberately do not survive in the plan. This
separation also permits elaborators to construct plans from symbolic `Nat`
expressions without imitating the concrete inference procedure.

When a known product is zero and the input dimension is also zero, the
missing factor is not uniquely determined. The certified checker rejects
that case rather than relying on the reference implementation's accidental
integer division by zero.
-/

public section

namespace TorchLean.Tensor.Internal.Check

/-- User-supplied lengths for named axes. -/
abbrev SupplementaryLengths := List (String × Nat)

namespace SupplementaryLengths

/-- Names in a supplementary-length list, preserving user order. -/
@[expose] def names (lengths : SupplementaryLengths) : List String :=
  lengths.map Prod.fst

/-- Look up the first supplied length with the given name. -/
@[expose] def lookup? (lengths : SupplementaryLengths) (name : String) : Option Nat :=
  (lengths.find? fun item => item.1 == name).map Prod.snd

end SupplementaryLengths

/-- A possibly incomplete assignment of elementary-axis lengths. -/
abbrev PartialAxisLengths := AxisId → Option Nat

namespace PartialAxisLengths

/-- Seed an assignment from anonymous literals and named supplementary lengths. -/
@[expose] def seed (supplementary : SupplementaryLengths) : PartialAxisLengths
  | .named name => supplementary.lookup? name
  | .anonymous value _ => some value
  | .ellipsis _ => none

/-- Update one elementary-axis length. -/
@[expose] def set (lengths : PartialAxisLengths) (axis : AxisId) (length : Nat) :
    PartialAxisLengths :=
  fun candidate => if candidate = axis then some length else lengths candidate

end PartialAxisLengths

/-- Multiply the resolved lengths of a list of elementary axes. -/
def resolvedProduct (lengths : PartialAxisLengths) :
    List AxisId → Option Nat
  | [] => some 1
  | axis :: axes => do
      let head ← lengths axis
      let tail ← resolvedProduct lengths axes
      pure (head * tail)

/--
The semantic data of a checked transformation.

Axis lengths are total because all resolution obligations are discharged
before a plan is constructed. This avoids exposing the checker's temporary
`Option`-valued assignment to semantics, lowerings, or symbolic elaboration.
-/
structure TransformPlan where
  /-- Parsed and ellipsis-expanded transformation with structural proofs. -/
  normalization : CheckedNormalization
  /-- Total length assignment for every logical elementary axis. -/
  axisLength : AxisId → Nat
  /-- Physical output induced by grouped output axes. -/
  output : Shape

namespace TransformPlan

/-- The normalized transformation underlying a checked plan. -/
@[expose] def normalized (plan : TransformPlan) : NormalizedTransform :=
  plan.normalization.value

/-- Every elementary axis relevant to the plan, with harmless cross-side repetition. -/
@[expose] def allAxes (plan : TransformPlan) : List AxisId :=
  plan.normalized.inputAxes ++ plan.normalized.outputAxes

/-- The product represented by one composite axis. -/
@[expose] def groupLength (plan : TransformPlan) (group : List AxisId) : Nat :=
  (group.map plan.axisLength).prod

/-- Input dimensions reconstructed from resolved elementary-axis lengths. -/
@[expose] def inferredInput (plan : TransformPlan) : Shape :=
  plan.normalized.inputGroups.map plan.groupLength

/-- Output dimensions reconstructed from resolved elementary-axis lengths. -/
@[expose] def inferredOutput (plan : TransformPlan) : Shape :=
  plan.normalized.outputGroups.map plan.groupLength

/-- Anonymous numeric axes retain their literal lengths. -/
@[expose] def literalAxesAgree (plan : TransformPlan) : Bool :=
  plan.allAxes.all fun axis =>
    match axis with
    | .anonymous value _ => plan.axisLength axis == value
    | _ => true

/-- The semantic facts certified for every transformation plan. -/
structure Valid (plan : TransformPlan) : Prop where
  /-- The underlying parsed transformation satisfies all structural checks. -/
  normalization : plan.normalized.Valid
  /-- Every anonymous numeric axis retains the length written in the pattern. -/
  literal_axes : plan.literalAxesAgree = true
  /-- Resolved logical-axis lengths reconstruct the physical input shape. -/
  input_shape : plan.inferredInput = plan.normalized.input
  /-- The stored output shape is exactly the shape induced by the output groups. -/
  output_shape : plan.output = plan.inferredOutput

end TransformPlan

/-- A transformation bundled with every invariant required by semantics and lowering. -/
structure CheckedTransform where
  /-- Total semantic plan produced by shape checking. -/
  value : TransformPlan
  /-- Proof that all lengths and physical shapes agree with the normalized pattern. -/
  valid : value.Valid

/-- Select unresolved axes from one normalized physical-axis group. -/
private def unknownAxes (lengths : PartialAxisLengths)
    (group : List AxisId) : List AxisId :=
  group.filter fun axis => (lengths axis).isNone

/-- Named axes used by a normalized pattern, preserving pattern order. -/
private def usedNamedAxes (normalized : NormalizedTransform) : List String :=
  (normalized.inputAxes ++ normalized.outputAxes).filterMap fun axis =>
    match axis with
    | .named name => some name
    | _ => none

/-- Whether every supplied name occurs in the normalized pattern. -/
private def supplementaryUsed (normalized : NormalizedTransform)
    (supplementary : SupplementaryLengths) : Bool :=
  supplementary.all fun item =>
    (usedNamedAxes normalized).contains item.1

/-- Whether every axis relevant to a normalized pattern has been resolved. -/
private def allAxesResolved (normalized : NormalizedTransform)
    (lengths : PartialAxisLengths) : Bool :=
  (normalized.inputAxes ++ normalized.outputAxes).all fun axis =>
    (lengths axis).isSome

/-- Whether resolved named axes retain every supplied length. -/
private def supplementaryAgree (lengths : PartialAxisLengths)
    (supplementary : SupplementaryLengths) : Bool :=
  supplementary.all fun item =>
    lengths (.named item.1) == some item.2

/--
Resolve at most one unknown factor per physical input dimension, checking
products and exact divisibility along the way.
-/
private def inferInputGroups (span : Syntax.Span) :
    List (List AxisId) → Shape → PartialAxisLengths → Result PartialAxisLengths
  | [], [], lengths => .ok lengths
  | group :: groups, dimension :: dimensions, lengths => do
      match unknownAxes lengths group with
      | [] =>
          let some product := resolvedProduct lengths group
            | .error
                { code := .internalInvariant
                  message := "a resolved input group contains an unresolved axis"
                  span }
          if product = dimension then
            inferInputGroups span groups dimensions lengths
          else
            .error
              { code := .dimensionMismatch
                message := s!"input dimension {dimension} does not equal expected product {product}"
                span }
      | [unknown] =>
          let knownAxes := group.filter fun axis => axis != unknown
          let some knownProduct := resolvedProduct lengths knownAxes
            | .error
                { code := .internalInvariant
                  message := "known factors contain an unresolved axis"
                  span }
          if knownProduct = 0 then
            if dimension = 0 then
              .error
                { code := .cannotInferAxis
                  message := "a missing factor cannot be inferred from a zero product"
                  span }
            else
              .error
                { code := .dimensionMismatch
                  message := s!"nonzero input dimension {dimension} has a zero known product"
                  span }
          else if dimension % knownProduct = 0 then
            let inferred := dimension / knownProduct
            inferInputGroups span groups dimensions (lengths.set unknown inferred)
          else
            .error
              { code := .dimensionMismatch
                message :=
                  s!"input dimension {dimension} is not divisible by known product {knownProduct}"
                span }
      | _ =>
          .error
            { code := .cannotInferAxis
              message := "a composite input axis has more than one unknown factor"
              span }
  | _, _, _ =>
      .error
        { code := .internalInvariant
          message := "normalized input groups and input shape have different ranks"
          span }

/-- Find the first axis whose length remains unresolved. -/
private def firstUnresolved? (lengths : PartialAxisLengths) :
    List AxisId → Option AxisId
  | [] => none
  | axis :: axes =>
      if (lengths axis).isSome then firstUnresolved? lengths axes else some axis

/-- Explain why a particular normalized axis still needs a length. -/
private def missingAxisMessage : AxisId → String
  | .named name => s!"axis '{name}' requires a supplied length"
  | .anonymous value _ => s!"anonymous axis {value} unexpectedly has no literal length"
  | .ellipsis index => s!"ellipsis axis {index} could not be inferred"

/--
Check a normalized transformation against concrete dimensions and supplied
named-axis lengths.
-/
def checkTransform (kind : TransformKind) (pattern : Syntax.TransformPattern)
    (inputShape : Shape) (supplementary : SupplementaryLengths := []) :
    Result CheckedTransform := do
  let normalization ← normalize kind pattern inputShape
  if supplementary.names.Nodup then
    let normalized := normalization.value
    if supplementaryUsed normalized supplementary = true then
      let seeded := PartialAxisLengths.seed supplementary
      let resolved ←
        inferInputGroups pattern.left.span normalized.inputGroups inputShape seeded
      match firstUnresolved? resolved normalized.outputAxes with
      | some axis =>
          .error
            { code := .missingAxisLength
              message := missingAxisMessage axis
              span := pattern.right.span }
      | none =>
          let axisLength := fun axis => (resolved axis).getD 0
          let output :=
            normalized.outputGroups.map fun group =>
              (group.map axisLength).prod
          let plan : TransformPlan :=
            { normalization
              axisLength
              output }
          if allAxesResolved normalized resolved = true then
            if hLiterals : plan.literalAxesAgree = true then
              if supplementaryAgree resolved supplementary = true then
                if hInput : plan.inferredInput = plan.normalized.input then
                  .ok
                    ⟨plan,
                      { normalization := normalization.valid
                        literal_axes := hLiterals
                        input_shape := hInput
                        output_shape := rfl }⟩
                else
                  .error
                    { code := .internalInvariant
                      message := "resolved elementary axes do not reconstruct the input shape"
                      span := pattern.left.span }
              else
                .error
                  { code := .internalInvariant
                    message := "a resolved named axis disagrees with its supplied length"
                    span := pattern.span }
            else
              .error
                { code := .internalInvariant
                  message := "a resolved anonymous axis disagrees with its literal length"
                  span := pattern.span }
          else
            .error
              { code := .internalInvariant
                message := "concrete checking left an elementary axis unresolved"
                span := pattern.span }
    else
      .error
        { code := .unusedAxisLength
          message := "a supplied axis length is not used by the transformation"
          span := pattern.span }
  else
    .error
      { code := .duplicateAxisLength
        message := "an axis length was supplied more than once"
        span := pattern.span }

end TorchLean.Tensor.Internal.Check
