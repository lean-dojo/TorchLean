/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Tensor.Internal.Check.Pack
public import NN.Tensor.Internal.Representation.Segment
public import NN.Tensor.Internal.Representation.Fiber.Differential
public import NN.Tensor.Internal.Representation.Fiber -- shake: keep

/-!
# Coordinate semantics for pack and unpack

A checked packing pattern identifies one star region in every input shape.
Its dimensions may differ between components, but all leading and trailing
dimensions agree. The star region is flattened in row-major order and the
flattened segments are placed consecutively on one packed axis.

The central object in this module is `CheckedPack.packedCoordinateEquiv`.
It first reshapes each component's star region to its checked segment, then
uses the general segment-concatenation equivalence. `denotePack` reads through
the inverse equivalence, while `denoteUnpack` reads through the forward
equivalence. Both round-trip laws therefore hold for every scalar type,
including scalar components, zero-size shapes, and zero-length segments.

No tensor lowering is executed here. The later lowering theorem compares
reshape and concatenation primitives with this coordinate denotation.
-/

@[expose] public section

namespace TorchLean.Tensor.Internal

open scoped BigOperators

universe u

namespace Check.CheckedPack

/-- The dependent family of input tensors accepted by a checked pack plan. -/
abbrev InputTensors (checked : Check.CheckedPack) (α : Type u)
    [Storage α] :=
  (component : Fin checked.inputShapes.length) →
    Rep α (checked.inputShapes.get component)

/-- The packed tensor type determined by a checked pack plan. -/
abbrev OutputTensor (checked : Check.CheckedPack) (α : Type u)
    [Storage α] :=
  Rep α checked.output

/--
Reshape every component coordinate into the corresponding segment coordinate.

The equivalence is dependent because components may have different ranks and
star shapes. Row-major reshape preserves all scalar positions, while
`componentSegmentEquiv` keeps component and metadata order synchronized.
-/
def componentCoordinateEquiv (checked : Check.CheckedPack) :
    ((component : Fin checked.inputShapes.length) ×
      Coord (checked.inputShapes.get component)) ≃
      ((segment : Fin checked.segmentLengths.length) ×
        Coord
          (checked.leadingShape ++
            checked.segmentLengths.get segment ::
              checked.trailingShape)) :=
  Equiv.sigmaCongr checked.componentSegmentEquiv fun component =>
    (Rep.reshapeCoordEquiv
      (checked.input_size_eq_segment component)).symm

/--
The coordinate equivalence defining both pack and unpack.

On the input side, a coordinate records which component is selected and a
coordinate inside that component. On the output side, the component's
row-major star coordinate is placed in its consecutive segment of the packed
axis; leading and trailing coordinates are unchanged.
-/
def packedCoordinateEquiv (checked : Check.CheckedPack) :
    ((component : Fin checked.inputShapes.length) ×
      Coord (checked.inputShapes.get component)) ≃
      Coord checked.output :=
  checked.componentCoordinateEquiv.trans <|
    Rep.concatenateAxesCoordinateEquiv checked.leadingShape
      checked.trailingShape checked.segmentLengths

end Check.CheckedPack

namespace Semantics

open Check

/--
Pack a checked family by reading the unique component coordinate represented
by each packed output coordinate.
-/
def denotePack {α : Type u} [Storage α] (checked : CheckedPack)
    (inputTensors : checked.InputTensors α) : checked.OutputTensor α :=
  Rep.ofFn fun outputCoordinate =>
    let inputCoordinate := checked.packedCoordinateEquiv.symm outputCoordinate
    inputTensors inputCoordinate.1 inputCoordinate.2

/--
Unpack a tensor by embedding each requested component coordinate into its
checked segment of the packed axis.
-/
def denoteUnpack {α : Type u} [Storage α] (checked : CheckedPack)
    (packedTensor : checked.OutputTensor α) : checked.InputTensors α :=
  fun component =>
    Rep.ofFn fun inputCoordinate =>
      packedTensor (checked.packedCoordinateEquiv ⟨component, inputCoordinate⟩)

/-- Pack reads the component and coordinate selected by the packed coordinate. -/
@[simp, grind =] theorem denotePack_apply {α : Type u} [Storage α]
    (checked : CheckedPack)
    (inputTensors : checked.InputTensors α)
    (outputCoordinate : Coord checked.output) :
    denotePack checked inputTensors outputCoordinate =
      inputTensors
        (checked.packedCoordinateEquiv.symm outputCoordinate).1
        (checked.packedCoordinateEquiv.symm outputCoordinate).2 :=
  by
    simp [denotePack]

/-- Unpack reads the packed coordinate assigned to the requested component entry. -/
@[simp, grind =] theorem denoteUnpack_apply {α : Type u} [Storage α]
    (checked : CheckedPack)
    (packedTensor : checked.OutputTensor α)
    (component : Fin checked.inputShapes.length)
    (inputCoordinate : Coord (checked.inputShapes.get component)) :
    denoteUnpack checked packedTensor component inputCoordinate =
      packedTensor
        (checked.packedCoordinateEquiv ⟨component, inputCoordinate⟩) :=
  by
    simp [denoteUnpack]

/--
Unpacking a packed component family recovers every component tensor.
-/
@[simp, grind =] theorem denoteUnpack_denotePack {α : Type u}
    [Storage α]
    (checked : CheckedPack) (inputTensors : checked.InputTensors α) :
    denoteUnpack checked (denotePack checked inputTensors) =
      inputTensors := by
  funext component
  ext inputCoordinate
  simpa only [denoteUnpack_apply, denotePack_apply] using
    congrArg
      (fun componentCoordinate =>
        inputTensors componentCoordinate.1 componentCoordinate.2)
      (checked.packedCoordinateEquiv.symm_apply_apply
        ⟨component, inputCoordinate⟩)

/--
Packing a complete checked unpack recovers the original packed tensor.
-/
@[simp, grind =] theorem denotePack_denoteUnpack {α : Type u}
    [Storage α]
    (checked : CheckedPack) (packedTensor : checked.OutputTensor α) :
    denotePack checked (denoteUnpack checked packedTensor) =
      packedTensor := by
  ext outputCoordinate
  simpa only [denotePack_apply, denoteUnpack_apply] using
    congrArg packedTensor
      (checked.packedCoordinateEquiv.apply_symm_apply outputCoordinate)

/--
Packing both component families preserves their direct-sum finite pairing.

The coordinate equivalence partitions every packed coordinate into exactly
one component coordinate. Multiplication order is unchanged, so the theorem
holds over a potentially noncommutative semiring.
-/
@[grind =] theorem dot_denotePack_denotePack {R : Type u}
    [Storage R] [Semiring R]
    (checked : CheckedPack)
    (leftTensors rightTensors : checked.InputTensors R) :
    Rep.dot
        (denotePack checked leftTensors)
        (denotePack checked rightTensors) =
      ∑ component,
        Rep.dot (leftTensors component) (rightTensors component) := by
  simp only [Rep.dot, denotePack_apply]
  rw [← Fintype.sum_sigma']
  refine Fintype.sum_equiv checked.packedCoordinateEquiv.symm _ _ ?_
  intro outputCoordinate
  simp

end Semantics

end TorchLean.Tensor.Internal
