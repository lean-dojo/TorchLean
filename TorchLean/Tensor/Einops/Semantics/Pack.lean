/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Tensor.Einops.Check.Pack
public import TorchLean.Tensor.Rep.Segment
public import TorchLean.Tensor.Rep.Fiber.Mean -- shake: keep
public import TorchLean.Tensor.Rep.Fiber.Product -- shake: keep
public import TorchLean.Tensor.Rep.Fiber.Tie -- shake: keep

/-!
# Coordinate semantics for pack and unpack

A checked packing pattern identifies one star region in every input shape.
Its dimensions may differ between components, but all leading and trailing
dimensions agree. The star region is flattened in row-major order and the
flattened segments are placed consecutively on one packed axis.

Both directions rest on `Pack.packedCoordEquiv`, which reshapes
each component's star region to its checked segment and then applies the
general segment-concatenation equivalence. `denotePack` reads through the
inverse equivalence and `denoteUnpack` through the forward one, so both
round-trip laws hold for every scalar type, including scalar components,
zero-size shapes, and zero-length segments.

Nothing is lowered here. The pack and unpack kernels are proved equal to this
coordinate denotation, and they fill their row-major outputs without
intermediate reshape or concatenation tensors.
-/

@[expose] public section

namespace TorchLean.Einops

open scoped BigOperators
open Tensor (Rep)
open Shape (Coord)

universe u

namespace Pack

/-- The dependent family of input tensors accepted by a checked pack plan. -/
abbrev InputTensors (checked : Einops.Pack) (α : Type u)
    [Storage α] :=
  (component : Fin checked.inputShapes.length) →
    Rep α (checked.inputShapes.get component)

/-- The packed tensor type determined by a checked pack plan. -/
abbrev OutputTensor (checked : Einops.Pack) (α : Type u)
    [Storage α] :=
  Rep α checked.output

/--
Reshape every component coordinate into the corresponding segment coordinate.

The equivalence is dependent because components may have different ranks and
star shapes. Row-major reshape preserves all scalar positions, while
`segmentEquiv` keeps component and metadata order synchronized.
-/
def componentCoordEquiv (checked : Einops.Pack) :
    ((component : Fin checked.inputShapes.length) ×
      Coord (checked.inputShapes.get component)) ≃
      ((segment : Fin checked.segmentLengths.length) ×
        Coord
          (checked.leadingShape ++
            checked.segmentLengths.get segment ::
              checked.trailingShape)) :=
  Equiv.sigmaCongr checked.segmentEquiv fun component =>
    (Coord.reshapeEquiv
      (checked.input_size_eq_segment component)).symm

/--
The coordinate equivalence defining both pack and unpack.

An input coordinate records the selected component and a coordinate inside
it. The component's row-major star coordinate lands in its consecutive segment
of the packed axis, and the leading and trailing coordinates are unchanged.
-/
def packedCoordEquiv (checked : Einops.Pack) :
    ((component : Fin checked.inputShapes.length) ×
      Coord (checked.inputShapes.get component)) ≃
      Coord checked.output :=
  checked.componentCoordEquiv.trans <|
    Coord.concatEquiv checked.leadingShape
      checked.trailingShape checked.segmentLengths

end Pack

namespace Semantics

/--
Pack a checked family by reading the unique component coordinate represented
by each packed output coordinate.
-/
def denotePack {α : Type u} [Storage α] (checked : Pack)
    (inputTensors : checked.InputTensors α) : checked.OutputTensor α :=
  Rep.ofFn fun outputCoordinate =>
    let inputCoordinate := checked.packedCoordEquiv.symm outputCoordinate
    inputTensors inputCoordinate.1 inputCoordinate.2

/--
Unpack a tensor by embedding each requested component coordinate into its
checked segment of the packed axis.
-/
def denoteUnpack {α : Type u} [Storage α] (checked : Pack)
    (packedTensor : checked.OutputTensor α) : checked.InputTensors α :=
  fun component =>
    Rep.ofFn fun inputCoordinate =>
      packedTensor (checked.packedCoordEquiv ⟨component, inputCoordinate⟩)

/-- Pack reads the component and coordinate selected by the packed coordinate. -/
@[simp, grind =] theorem denotePack_apply {α : Type u} [Storage α]
    (checked : Pack)
    (inputTensors : checked.InputTensors α)
    (outputCoordinate : Coord checked.output) :
    denotePack checked inputTensors outputCoordinate =
      inputTensors
        (checked.packedCoordEquiv.symm outputCoordinate).1
        (checked.packedCoordEquiv.symm outputCoordinate).2 :=
  by
    simp [denotePack]

/-- Unpack reads the packed coordinate assigned to the requested component entry. -/
@[simp, grind =] theorem denoteUnpack_apply {α : Type u} [Storage α]
    (checked : Pack)
    (packedTensor : checked.OutputTensor α)
    (component : Fin checked.inputShapes.length)
    (inputCoordinate : Coord (checked.inputShapes.get component)) :
    denoteUnpack checked packedTensor component inputCoordinate =
      packedTensor
        (checked.packedCoordEquiv ⟨component, inputCoordinate⟩) :=
  by
    simp [denoteUnpack]

/--
Unpacking a packed component family recovers every component tensor.
-/
@[simp, grind =, einops =, einops_simps] theorem denoteUnpack_denotePack {α : Type u}
    [Storage α]
    (checked : Pack) (inputTensors : checked.InputTensors α) :
    denoteUnpack checked (denotePack checked inputTensors) =
      inputTensors := by
  funext component
  ext inputCoordinate
  simpa only [denoteUnpack_apply, denotePack_apply] using
    congrArg
      (fun componentCoordinate =>
        inputTensors componentCoordinate.1 componentCoordinate.2)
      (checked.packedCoordEquiv.symm_apply_apply
        ⟨component, inputCoordinate⟩)

/--
Packing a complete checked unpack recovers the original packed tensor.
-/
@[simp, grind =, einops =, einops_simps] theorem denotePack_denoteUnpack {α : Type u}
    [Storage α]
    (checked : Pack) (packedTensor : checked.OutputTensor α) :
    denotePack checked (denoteUnpack checked packedTensor) =
      packedTensor := by
  ext outputCoordinate
  simpa only [denotePack_apply, denoteUnpack_apply] using
    congrArg packedTensor
      (checked.packedCoordEquiv.apply_symm_apply outputCoordinate)

/--
Packing both component families preserves their direct-sum finite pairing.

The coordinate equivalence pairs each packed coordinate with exactly one
component coordinate, and every product keeps its factor order, so the
identity holds for any multiplication on an additive commutative monoid.
-/
@[grind =, einops =, einops_simps] theorem dot_denotePack_denotePack {R : Type u}
    [Storage R] [AddCommMonoid R] [Mul R]
    (checked : Pack)
    (leftTensors rightTensors : checked.InputTensors R) :
    Rep.dot
        (denotePack checked leftTensors)
        (denotePack checked rightTensors) =
      ∑ component,
        Rep.dot (leftTensors component) (rightTensors component) := by
  simp only [Rep.dot, denotePack_apply]
  rw [← Fintype.sum_sigma']
  refine Fintype.sum_equiv checked.packedCoordEquiv.symm _ _ ?_
  intro outputCoordinate
  simp

end Semantics

end TorchLean.Einops
