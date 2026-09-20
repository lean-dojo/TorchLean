/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Tensor.Internal.Semantics.Pack

/-!
# Direct native lowering for pack and unpack

The checked component-to-packed coordinate equivalence drives both kernels.
Packing fills one packed output array by reading the corresponding component
entry. Unpacking fills each component array by reading its corresponding
packed entry. Neither direction allocates slice, reshape, or concatenation
intermediates.

The compiler-correctness theorems compare these direct row-major kernels with
the independent coordinate semantics. The resulting inverse and adjoint laws
apply to arbitrary scalar types or semirings as appropriate; they do not
assume positive dimensions or a fixed tensor rank.
-/

@[expose] public section

namespace TorchLean.Tensor.Internal

open scoped BigOperators

universe u

namespace Lowering

open Check

/--
Execute pack by filling the packed row-major buffer directly.

Each output flat index is converted once to its certified component
coordinate, then read from that component's native tensor storage. No
reshaped component tensors or concatenation buffers are allocated.
-/
def packTensor {α : Type u} [Storage α] (checked : CheckedPack)
    (inputTensors : checked.InputTensors α) : checked.OutputTensor α :=
  Rep.ofFlatFn fun outputFlatIndex =>
    let outputCoordinate := Coord.unlinearize outputFlatIndex
    let inputCoordinate :=
      checked.packedCoordinateEquiv.symm outputCoordinate
    inputTensors inputCoordinate.1 inputCoordinate.2

/--
Execute unpack by filling every component buffer directly from the packed
tensor.

The checked coordinate equivalence maps each component flat index to its
unique packed location, so no slice or reshape buffer is constructed.
-/
def unpackTensor {α : Type u} [Storage α] (checked : CheckedPack)
    (packedTensor : checked.OutputTensor α) : checked.InputTensors α :=
  fun component =>
    Rep.ofFlatFn fun inputFlatIndex =>
      packedTensor <|
        checked.packedCoordinateEquiv
          ⟨component, Coord.unlinearize inputFlatIndex⟩

/--
The direct row-major pack kernel equals the independent coordinate
denotation.
-/
@[grind =] theorem packTensor_correct {α : Type u} [Storage α]
    (checked : CheckedPack)
    (inputTensors : checked.InputTensors α) :
    packTensor checked inputTensors =
      Semantics.denotePack checked inputTensors := by
  ext outputCoordinate
  simp only [packTensor, Rep.get_ofFlatFn, Semantics.denotePack_apply]
  rw [Coord.unlinearize_linearize]

/--
The direct row-major unpack kernel equals the independent coordinate
denotation.
-/
@[grind =] theorem unpackTensor_correct {α : Type u} [Storage α]
    (checked : CheckedPack)
    (packedTensor : checked.OutputTensor α) :
    unpackTensor checked packedTensor =
      Semantics.denoteUnpack checked packedTensor := by
  funext component
  ext inputCoordinate
  simp [unpackTensor, Semantics.denoteUnpack]

/-- Unpacking a lowered pack recovers every input tensor. -/
@[simp, grind =] theorem unpackTensor_packTensor {α : Type u}
    [Storage α]
    (checked : CheckedPack) (inputTensors : checked.InputTensors α) :
    unpackTensor checked (packTensor checked inputTensors) =
      inputTensors := by
  rw [packTensor_correct, unpackTensor_correct]
  exact Semantics.denoteUnpack_denotePack checked inputTensors

/-- Packing a complete lowered unpack recovers the packed tensor. -/
@[simp, grind =] theorem packTensor_unpackTensor {α : Type u}
    [Storage α]
    (checked : CheckedPack) (packedTensor : checked.OutputTensor α) :
    packTensor checked (unpackTensor checked packedTensor) =
      packedTensor := by
  rw [unpackTensor_correct, packTensor_correct]
  exact Semantics.denotePack_denoteUnpack checked packedTensor

/--
Pack is adjoint to unpack for the standard finite tensor pairing.

The sum ranges over the dependent component family. Multiplication order is
preserved, so a commutative scalar multiplication is not required.
-/
@[grind =] theorem dot_packTensor {R : Type u} [Storage R]
    [Semiring R]
    (checked : CheckedPack) (inputTensors : checked.InputTensors R)
    (packedTensor : checked.OutputTensor R) :
    Rep.dot (packTensor checked inputTensors) packedTensor =
      ∑ component,
        Rep.dot (inputTensors component)
          (unpackTensor checked packedTensor component) := by
  rw [packTensor_correct, unpackTensor_correct]
  simp only [Rep.dot, Semantics.denotePack_apply,
    Semantics.denoteUnpack_apply]
  rw [← Fintype.sum_sigma']
  refine Fintype.sum_equiv checked.packedCoordinateEquiv.symm _ _ ?_
  intro outputCoordinate
  simp

/--
Unpack is adjoint to pack with the packed tensor in the left pairing slot.

This is stated separately from `dot_packTensor` because it also preserves
multiplication order over noncommutative semirings.
-/
theorem dot_unpackTensor {R : Type u} [Storage R] [Semiring R]
    (checked : CheckedPack) (packedTensor : checked.OutputTensor R)
    (componentTensors : checked.InputTensors R) :
    (∑ component,
        Rep.dot (unpackTensor checked packedTensor component)
          (componentTensors component)) =
      Rep.dot packedTensor (packTensor checked componentTensors) := by
  rw [packTensor_correct, unpackTensor_correct]
  simp only [Rep.dot, Semantics.denotePack_apply,
    Semantics.denoteUnpack_apply]
  rw [← Fintype.sum_sigma']
  refine Fintype.sum_equiv checked.packedCoordinateEquiv _ _ ?_
  intro componentCoordinate
  simp only [List.get_eq_getElem, Sigma.eta]
  exact congrArg
    (fun value =>
      packedTensor
          (checked.packedCoordinateEquiv componentCoordinate) *
        value)
    ((congrArg
      (fun coordinate =>
        componentTensors coordinate.1 coordinate.2)
      (checked.packedCoordinateEquiv.symm_apply_apply
        componentCoordinate)).symm)

-- The packed pairing contains every top-level parameter and avoids the
-- dependent `Finset.sum` expression that automatic pattern inference rejects.
grind_pattern dot_unpackTensor =>
  Rep.dot packedTensor (packTensor checked componentTensors)

end Lowering

end TorchLean.Tensor.Internal
