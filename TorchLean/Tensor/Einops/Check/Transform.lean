/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Tensor.Einops.AxisTuple
public import TorchLean.Tensor.Einops.Check.NormalForm

/-!
# Transformation plans and executable checking

`TransformPattern.check` resolves the elementary-axis lengths of a `rearrange`, `repeat`
or `reduce` pattern. A composite input dimension may contain at most one axis
whose length is neither supplied nor fixed by an anonymous numeric axis, and
that length is inferred by exact division.

`TransformPlan` stores the mathematical data that semantics and lowering
consume: a normalized pattern, a total axis-length function and the physical
output shape. Supplied lengths and partial assignments belong to the concrete
checking algorithm and stay out of the plan, which lets elaborators build plans
from symbolic `Nat` expressions without imitating the concrete inference.

When a known product is zero and the input dimension is also zero, the missing
factor is not uniquely determined. The checker rejects that case with a
diagnostic, while the reference implementation reaches it only through an
accidental integer division by zero.
-/

public section

namespace TorchLean.Einops

/-- User-supplied lengths for named axes. -/
abbrev SupplementaryLengths := List (String × Nat)

namespace SupplementaryLengths

/-- Names in a supplementary-length list, preserving user order. -/
@[expose] def names (lengths : SupplementaryLengths) : List String :=
  lengths.map Prod.fst

end SupplementaryLengths

/-- A possibly incomplete assignment of elementary-axis lengths. -/
abbrev PartialAxisLengths := AxisId → Option Nat

namespace PartialAxisLengths

/-- Seed an assignment from anonymous literals and named supplementary lengths. -/
@[expose] def seed (supplementary : SupplementaryLengths) : PartialAxisLengths
  | .named name => supplementary.lookup name
  | .anonymous value _ => some value
  | .ellipsis _ => none

/-- Update one elementary-axis length. -/
@[expose] def set (lengths : PartialAxisLengths) (axis : AxisId) (length : Nat) :
    PartialAxisLengths :=
  fun candidate => if candidate = axis then some length else lengths candidate

end PartialAxisLengths

/-- Multiply the resolved lengths of a list of elementary axes. -/
def PartialAxisLengths.resolvedProduct (lengths : PartialAxisLengths) :
    List AxisId → Option Nat
  | [] => some 1
  | axis :: axes => do
      let head ← lengths axis
      let tail ← resolvedProduct lengths axes
      pure (head * tail)

/--
The semantic data of a checked transformation.

Axis lengths are total because every resolution obligation is discharged before
a plan is built, so the checker's temporary `Option`-valued assignment never
reaches semantics, lowering or symbolic elaboration.
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
@[expose] def normalized (plan : TransformPlan) : NormalForm :=
  plan.normalization.value

/-- Input axes followed by output axes, so an axis on both sides appears twice. -/
@[expose] def allAxes (plan : TransformPlan) : List AxisId :=
  plan.normalized.inputAxes ++ plan.normalized.outputAxes

/-- Input dimensions reconstructed from resolved elementary-axis lengths. -/
@[expose] def inferredInput (plan : TransformPlan) : Shape :=
  AxisTuple.groupedShape plan.axisLength plan.normalized.inputGroups

/-- Output dimensions reconstructed from resolved elementary-axis lengths. -/
@[expose] def inferredOutput (plan : TransformPlan) : Shape :=
  AxisTuple.groupedShape plan.axisLength plan.normalized.outputGroups

/-- Whether every anonymous numeric axis keeps its literal length. -/
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
structure Transform where
  /-- Total semantic plan produced by shape checking. -/
  value : TransformPlan
  /-- Proof that all lengths and physical shapes agree with the normalized pattern. -/
  valid : value.Valid

/-- Whether every supplied name occurs in the normalized pattern. -/
private def supplementaryUsed (normalized : NormalForm)
    (supplementary : SupplementaryLengths) : Bool :=
  let names := (normalized.inputAxes ++ normalized.outputAxes).filterMap fun axis =>
    match axis with
    | .named name => some name
    | _ => none
  supplementary.all fun item => names.contains item.1

/-- Whether every axis relevant to a normalized pattern has been resolved. -/
private def allAxesResolved (normalized : NormalForm)
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
private def inferInputGroups (span : Span) :
    List (List AxisId) → Shape → PartialAxisLengths → Result PartialAxisLengths
  | [], [], lengths => .ok lengths
  | group :: groups, dimension :: dimensions, lengths => do
      match group.filter (fun axis => (lengths axis).isNone) with
      | [] =>
          let some product := PartialAxisLengths.resolvedProduct lengths group
            | .error
                { code := .invariantFailed
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
          let some knownProduct := PartialAxisLengths.resolvedProduct lengths knownAxes
            | .error
                { code := .invariantFailed
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
        { code := .invariantFailed
          message := "normalized input groups and input shape have different ranks"
          span }

/-- Explain why a normalized axis has no resolved length. -/
private def missingAxisMessage : AxisId → String
  | .named name => s!"axis '{name}' requires a supplied length"
  | .anonymous value _ => s!"anonymous axis {value} unexpectedly has no literal length"
  | .ellipsis index => s!"ellipsis axis {index} could not be inferred"

/--
Normalize a transformation pattern against a concrete input shape and resolve
every axis length from the shape and the supplied named-axis lengths.

Each supplied name must occur in the pattern and be given once. Output axes that
the input does not determine need a supplied length.
-/
def TransformPattern.check (kind : TransformKind) (pattern : TransformPattern)
    (inputShape : Shape) (supplementary : SupplementaryLengths := []) :
    Result Transform := do
  let normalization ← normalize kind pattern inputShape
  if supplementary.names.Nodup then
    let normalized := normalization.value
    if supplementaryUsed normalized supplementary = true then
      let seeded := PartialAxisLengths.seed supplementary
      let resolved ←
        inferInputGroups pattern.left.span normalized.inputGroups inputShape seeded
      match normalized.outputAxes.find? (fun axis => (resolved axis).isNone) with
      | some axis =>
          .error
            { code := .missingAxisLength
              message := missingAxisMessage axis
              span := pattern.right.span }
      | none =>
          let axisLength := fun axis => (resolved axis).getD 0
          let output := AxisTuple.groupedShape axisLength normalized.outputGroups
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
                    { code := .invariantFailed
                      message := "resolved elementary axes do not reconstruct the input shape"
                      span := pattern.left.span }
              else
                .error
                  { code := .invariantFailed
                    message := "a resolved named axis disagrees with its supplied length"
                    span := pattern.span }
            else
              .error
                { code := .invariantFailed
                  message := "a resolved anonymous axis disagrees with its literal length"
                  span := pattern.span }
          else
            .error
              { code := .invariantFailed
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

end TorchLean.Einops
