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
# Concrete pack and unpack checking

The packing pattern has a fixed leading region, one `*`, and a fixed trailing
region. Inputs may have different ranks because `*` captures an arbitrary
middle shape. A checked plan records the common fixed shapes and proves that
every component decomposes as

```text
leadingShape ++ starShape ++ trailingShape.
```

The same `CheckedPack` certificate serves both directions. `checkPack` rejects
an empty input family and obtains the fixed shapes from its first tensor.
`checkUnpack` obtains them from the packed tensor, resolves requested star
shapes, and therefore also represents the strict empty-metadata case when the
packed axis has length zero.

Requested unpack dimensions are integers so `-1` can denote one inferred
dimension. Every other dimension must be nonnegative. Resolution rejects
multiple inferred dimensions, non-divisible residual lengths, underdetermined
zero products, and metadata whose segment lengths do not exactly partition
the packed axis.
-/

public section

namespace TorchLean.Tensor.Internal.Check

open TorchLean.Tensor.Internal.Syntax

/-- The number of fixed axes surrounding `*` in a packing pattern. -/
@[expose] def packFixedRank (pattern : PackPattern) : Nat :=
  pattern.before.length + pattern.after.length

/--
The middle shape captured by `*` in one component.

Checked components have enough dimensions for both fixed regions. The
subtraction therefore removes exactly those regions.
-/
@[expose] def packStarShape (pattern : PackPattern) (inputShape : Shape) : Shape :=
  (inputShape.drop pattern.before.length).take
    (inputShape.length - packFixedRank pattern)

/--
A pack/unpack shape plan with exactly the invariants used by semantics.

`leadingShape` and `trailingShape` are explicit because strict unpacking may
produce an empty component family, from which the fixed dimensions cannot be
recovered. Component star shapes, segment lengths, metadata, offsets, and the
packed output shape remain deterministic derived values.
-/
structure CheckedPack where
  /-- Parsed packing pattern shared by pack and unpack. -/
  pattern : PackPattern
  /-- Shapes of the unpacked components, in segment order. -/
  inputShapes : List Shape
  /-- Fixed dimensions preceding every component's star region. -/
  leadingShape : Shape
  /-- Fixed dimensions following every component's star region. -/
  trailingShape : Shape
  /-- The stored leading shape has the rank written before `*`. -/
  leading_rank : leadingShape.length = pattern.before.length
  /-- The stored trailing shape has the rank written after `*`. -/
  trailing_rank : trailingShape.length = pattern.after.length
  /-- Every component decomposes into the shared prefix, its star region, and suffix. -/
  input_shapes :
    ∀ component : Fin inputShapes.length,
      inputShapes.get component =
        leadingShape ++
          packStarShape pattern (inputShapes.get component) ++
          trailingShape

namespace CheckedPack

/-- The middle shape represented by one checked component. -/
@[expose] def starShape (checked : CheckedPack)
    (component : Fin checked.inputShapes.length) : Shape :=
  packStarShape checked.pattern (checked.inputShapes.get component)

/-- Pack metadata in component order. -/
@[expose] def metadata (checked : CheckedPack) : List Shape :=
  checked.inputShapes.map (packStarShape checked.pattern)

/-- Flattened lengths of all packed middle regions. -/
@[expose] def segmentLengths (checked : CheckedPack) : List Nat :=
  checked.metadata.map Shape.size

/-- The length of the concatenated packed axis. -/
@[expose] def packedAxisLength (checked : CheckedPack) : Nat :=
  checked.segmentLengths.sum

/-- Shape of the packed tensor. -/
@[expose] def output (checked : CheckedPack) : Shape :=
  checked.leadingShape ++ checked.packedAxisLength :: checked.trailingShape

/-- Pack metadata has one entry for every input component. -/
@[simp] theorem metadata_length (checked : CheckedPack) :
    checked.metadata.length = checked.inputShapes.length := by
  simp [metadata]

/-- Segment lengths have one entry for every input component. -/
@[simp] theorem segmentLengths_length (checked : CheckedPack) :
    checked.segmentLengths.length = checked.inputShapes.length := by
  simp [segmentLengths]

/--
Match each input component with its metadata segment.

The two lists have equal length by construction. Naming this equivalence
keeps the dependent component-to-segment transport identical in checking,
semantics, and lowering.
-/
def componentSegmentEquiv (checked : CheckedPack) :
    Fin checked.inputShapes.length ≃ Fin checked.segmentLengths.length :=
  finCongr checked.segmentLengths_length.symm

/--
Matching a component with its metadata segment preserves its position in the
component list.
-/
@[simp] theorem componentSegmentEquiv_val (checked : CheckedPack)
    (component : Fin checked.inputShapes.length) :
    (checked.componentSegmentEquiv component).val = component.val := by
  rfl

/-- Metadata at a component index is its checked star shape. -/
@[simp] theorem metadata_get (checked : CheckedPack)
    (component : Fin checked.metadata.length) :
    checked.metadata[component.val] =
      checked.starShape
        (Fin.cast checked.metadata_length component) := by
  simp [metadata, starShape]
  rfl

/-- A segment length is the row-major size of its metadata shape. -/
@[simp] theorem segmentLengths_get (checked : CheckedPack)
    (segment : Fin checked.segmentLengths.length) :
    checked.segmentLengths[segment.val] =
      Shape.size
        (checked.starShape
          (Fin.cast checked.segmentLengths_length segment)) := by
  simp [segmentLengths, metadata, starShape]
  rfl

/--
The segment selected by an input component has the row-major size of that
component's star region.
-/
@[simp] theorem segmentLength_eq_star_size (checked : CheckedPack)
    (component : Fin checked.inputShapes.length) :
    checked.segmentLengths[component.val] =
      Shape.size (checked.starShape component) := by
  simpa using
    checked.segmentLengths_get
      (Fin.cast checked.segmentLengths_length.symm component)

/--
Looking up the metadata segment selected for a component yields the size of
that component's star region.
-/
theorem segmentLengths_get_componentSegmentEquiv
    (checked : CheckedPack)
    (component : Fin checked.inputShapes.length) :
    checked.segmentLengths.get
        (checked.componentSegmentEquiv component) =
      Shape.size (checked.starShape component) := by
  simp [componentSegmentEquiv]

/--
Pack metadata is exactly the checked star-shape family in input order.

This extensional form is useful when clients consume metadata through finite
component indices rather than list lookup.
-/
theorem metadata_eq_ofFn (checked : CheckedPack) :
    checked.metadata =
      List.ofFn fun component : Fin checked.inputShapes.length =>
        checked.starShape component := by
  rw [metadata, ← List.ofFn_getElem_eq_map]
  rfl

/--
Flattening the star region into one axis preserves every component's number
of scalar entries.
-/
theorem input_size_eq_flattened (checked : CheckedPack)
    (component : Fin checked.inputShapes.length) :
    Shape.size (checked.inputShapes.get component) =
      Shape.size
        (checked.leadingShape ++
          Shape.size (checked.starShape component) ::
          checked.trailingShape) := by
  rw [checked.input_shapes component]
  simp [starShape, Shape.size_append]

/--
Reshaping one input component to its checked segment shape preserves the
number of entries.
-/
theorem input_size_eq_segment (checked : CheckedPack)
    (component : Fin checked.inputShapes.length) :
    Shape.size (checked.inputShapes.get component) =
      Shape.size
        (checked.leadingShape ++
          checked.segmentLengths.get
              (checked.componentSegmentEquiv component) ::
            checked.trailingShape) := by
  have hSegment :
      checked.segmentLengths.get
          (checked.componentSegmentEquiv component) =
        Shape.size (checked.starShape component) := by
    simp [componentSegmentEquiv]
  rw [checked.input_size_eq_flattened]
  rw [hSegment]

end CheckedPack

namespace Pack.Impl

/--
Validate rank and fixed-prefix/suffix agreement for every component in a pack
family.
-/
@[expose] def checkPackComponents (pattern : PackPattern)
    (leadingShape trailingShape : Shape) :
    List Shape → Nat → Result Unit
  | [], _ => .ok ()
  | inputShape :: inputShapes, component => do
      if packFixedRank pattern ≤ inputShape.length then
        if inputShape.take pattern.before.length = leadingShape then
          if inputShape.drop (inputShape.length - pattern.after.length) =
              trailingShape then
            let starShape := packStarShape pattern inputShape
            if inputShape = leadingShape ++ starShape ++ trailingShape then
              checkPackComponents pattern leadingShape trailingShape
                inputShapes (component + 1)
            else
              .error
                { code := .internalInvariant
                  message :=
                    s!"pack component {component} did not decompose after " ++
                      "its fixed dimensions were validated"
                  span := pattern.span }
          else
            .error
              { code := .packDimensionMismatch
                message :=
                  s!"pack component {component} has trailing dimensions " ++
                    "that disagree with the first component"
                span :=
                  match pattern.after.getLast? with
                  | some axis => axis.span
                  | none => pattern.packed }
        else
          .error
            { code := .packDimensionMismatch
              message :=
                s!"pack component {component} has leading dimensions " ++
                  "that disagree with the first component"
              span :=
                match pattern.before.head? with
                | some axis => axis.span
                | none => pattern.packed }
      else
        .error
          { code := .packRankMismatch
            message :=
              s!"pack component {component} has rank {inputShape.length}, " ++
                s!"but the pattern requires at least {packFixedRank pattern} fixed axes"
            span := pattern.span }

/--
Bundle validated component decompositions into the dependent `CheckedPack`
certificate consumed by semantics and lowering.
-/
@[expose] def certifyPackComponents (pattern : PackPattern)
    (inputShapes : List Shape) (leadingShape trailingShape : Shape) :
    Result CheckedPack := do
  checkPackComponents pattern leadingShape trailingShape inputShapes 0
  if hLeading : leadingShape.length = pattern.before.length then
    if hTrailing : trailingShape.length = pattern.after.length then
      if hInputs :
          ∀ component : Fin inputShapes.length,
            inputShapes.get component =
              leadingShape ++
                packStarShape pattern (inputShapes.get component) ++
                trailingShape then
        .ok
          { pattern
            inputShapes
            leadingShape
            trailingShape
            leading_rank := hLeading
            trailing_rank := hTrailing
            input_shapes := hInputs }
      else
        .error
          { code := .internalInvariant
            message := "validated pack components lack a certified decomposition"
            span := pattern.span }
    else
      .error
        { code := .internalInvariant
          message := "the checked trailing pack shape has the wrong rank"
          span := pattern.span }
  else
    .error
        { code := .internalInvariant
          message := "the checked leading pack shape has the wrong rank"
          span := pattern.span }

end Pack.Impl

/--
Check a nonempty family of tensor shapes for packing.

The first component determines fixed leading and trailing dimensions. Every
other component must agree at those positions, while its star region may have
any finite shape, including an empty shape or zero dimensions.
-/
@[expose] def checkPack (pattern : PackPattern)
    (inputShapes : List Shape) : Result CheckedPack := do
  match inputShapes with
  | [] =>
      .error
        { code := .emptyPackInput
          message := "pack requires at least one input tensor"
          span := pattern.packed }
  | firstShape :: _ =>
      if packFixedRank pattern ≤ firstShape.length then
        let leadingShape := firstShape.take pattern.before.length
        let trailingShape :=
          firstShape.drop (firstShape.length - pattern.after.length)
        Pack.Impl.certifyPackComponents
          pattern inputShapes leadingShape trailingShape
      else
        .error
          { code := .packRankMismatch
            message :=
              s!"pack component 0 has rank {firstShape.length}, " ++
                s!"but the pattern requires at least {packFixedRank pattern} fixed axes"
            span := pattern.span }

/-- Integer star shapes accepted by the public unpack checker. -/
abbrev RequestedShapes := List (List Int)

namespace Pack.Impl

/-- Compute one fully specified requested star-shape size. -/
@[expose] def requestedShapeProduct (requestedShape : List Int) : Nat :=
  (requestedShape.map Int.toNat).prod

/-- Multiply the known factors of a requested shape, treating `-1` as unknown. -/
@[expose] def requestedKnownFactor (requestedShape : List Int) : Nat :=
  (requestedShape.map fun dimension =>
    if dimension = -1 then 1 else dimension.toNat).prod

/-- Count inferred `-1` dimensions across the complete unpack request. -/
@[expose] def requestedInferredCount (requestedShapes : RequestedShapes) : Nat :=
  requestedShapes.flatten.count (-1)

/-- Replace the unique inferred dimension by its resolved natural-number value. -/
@[expose] def resolveRequestedShapesWith
    (inferred : Nat) (requestedShapes : RequestedShapes) : List Shape :=
  requestedShapes.map fun requestedShape =>
    requestedShape.map fun dimension =>
      if dimension = -1 then inferred else dimension.toNat

/-- Find the first unpack dimension smaller than the permitted sentinel `-1`. -/
@[expose] def firstInvalidRequestedDimension? :
    RequestedShapes → Option Int
  | [] => none
  | requestedShape :: requestedShapes =>
      match requestedShape.find? fun dimension => dimension < -1 with
      | some invalid => some invalid
      | none => firstInvalidRequestedDimension? requestedShapes

end Pack.Impl

/--
Resolve requested unpack metadata against one concrete packed-axis length.

The result contains only natural-number shapes. Success guarantees
computationally that their products sum to the packed-axis length; the
subsequent checked plan turns that equality into the segment partition used
by semantics.
-/
@[expose] def resolveUnpackShapes (span : Span) (packedAxisLength : Nat)
    (requestedShapes : RequestedShapes) : Result (List Shape) := do
  match Pack.Impl.firstInvalidRequestedDimension? requestedShapes with
  | some invalid =>
      .error
        { code := .invalidPackedShape
          message :=
            s!"unpack dimensions must be nonnegative or -1, not {invalid}"
          span }
  | none =>
      let inferredCount := Pack.Impl.requestedInferredCount requestedShapes
      if inferredCount > 1 then
        .error
          { code := .multipleInferredDimensions
            message := "unpack metadata may contain at most one inferred -1 dimension"
            span }
      else if inferredCount = 0 then
        let resolved :=
          Pack.Impl.resolveRequestedShapesWith 0 requestedShapes
        if (resolved.map Shape.size).sum = packedAxisLength then
          .ok resolved
        else
          .error
            { code := .packedAxisMismatch
              message :=
                s!"requested segment lengths sum to " ++
                  s!"{(resolved.map Shape.size).sum}, but the packed axis has " ++
                  s!"length {packedAxisLength}"
              span }
      else
        let knownTotal :=
          (requestedShapes.map fun requestedShape =>
            if requestedShape.contains (-1) then
              0
            else
              Pack.Impl.requestedShapeProduct requestedShape).sum
        if knownTotal ≤ packedAxisLength then
          let residual := packedAxisLength - knownTotal
          let some inferredShape :=
              requestedShapes.find? fun requestedShape =>
                requestedShape.contains (-1)
            | .error
                { code := .internalInvariant
                  message := "unpack inference count has no inferred shape"
                  span }
          let knownFactor :=
            Pack.Impl.requestedKnownFactor inferredShape
          if knownFactor = 0 then
            .error
              { code := .cannotInferPackedDimension
                message :=
                  if residual = 0 then
                    "an inferred unpack dimension is underdetermined by a zero product"
                  else
                    "a nonzero packed segment cannot have a zero known product"
                span }
          else if residual % knownFactor = 0 then
            let inferred := residual / knownFactor
            let resolved :=
              Pack.Impl.resolveRequestedShapesWith
                inferred requestedShapes
            if (resolved.map Shape.size).sum = packedAxisLength then
              .ok resolved
            else
              .error
                { code := .internalInvariant
                  message := "resolved unpack segments do not partition the packed axis"
                  span }
          else
            .error
              { code := .cannotInferPackedDimension
                message :=
                  s!"remaining packed length {residual} is not divisible by " ++
                    s!"the known factor {knownFactor}"
                span }
        else
          .error
            { code := .packedAxisMismatch
              message :=
                s!"known unpack segments require length {knownTotal}, " ++
                  s!"larger than packed-axis length {packedAxisLength}"
              span }

/--
Check strict unpack metadata against a concrete packed tensor shape.

Unlike the reference implementation's backend-deferred failures, every
segment bound and reshape size is validated before tensor execution.
-/
@[expose] def checkUnpack (pattern : PackPattern) (packedShape : Shape)
    (requestedShapes : RequestedShapes) : Result CheckedPack := do
  let expectedRank := pattern.before.length + 1 + pattern.after.length
  if hRank : packedShape.length = expectedRank then
    let packedAxis : Fin packedShape.length :=
      ⟨pattern.before.length, by omega⟩
    let packedAxisLength := packedShape.get packedAxis
    let resolvedStarShapes ←
      resolveUnpackShapes pattern.packed packedAxisLength requestedShapes
    let leadingShape := packedShape.take pattern.before.length
    let trailingShape := packedShape.drop (pattern.before.length + 1)
    let inputShapes :=
      resolvedStarShapes.map fun starShape =>
        leadingShape ++ starShape ++ trailingShape
    let checked ←
      Pack.Impl.certifyPackComponents
        pattern inputShapes leadingShape trailingShape
    if checked.output = packedShape then
      .ok checked
    else
      .error
        { code := .internalInvariant
          message := "resolved unpack metadata reconstructed a different packed shape"
          span := pattern.span }
  else
    .error
      { code := .unpackRankMismatch
        message :=
          s!"unpack input has rank {packedShape.length}, " ++
            s!"but the pattern requires rank {expectedRank}"
        span := pattern.span }

end TorchLean.Tensor.Internal.Check
