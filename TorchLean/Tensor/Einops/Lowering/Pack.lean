/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Tensor.Einops.Semantics.Pack

/-!
# Direct native lowering for pack and unpack

The checked component-to-packed coordinate equivalence drives both kernels.
Packing fills one packed output buffer by reading the corresponding component
entry. Unpacking fills each component buffer by reading its corresponding
packed entry. Neither direction allocates slice, reshape, or concatenation
intermediates.

The compiler-correctness theorems compare these direct row-major kernels with the independent
coordinate semantics. The resulting inverse laws hold for every scalar type, and the adjoint laws
need only an additive commutative monoid with a multiplication. Both hold at every tensor rank,
zero-sized dimensions included.
-/

@[expose] public section

namespace TorchLean.Einops

open scoped BigOperators

universe u

namespace Lowering

open Tensor (Rep)
open Shape (Coord)

/--
Execute pack by filling the packed row-major buffer in one pass.

Each output flat index is converted once to its certified component
coordinate, then read from that component's native tensor storage. No
reshaped component tensors or concatenation buffers are allocated.
-/
def pack {α : Type u} [Storage α] (checked : Pack)
    (inputTensors : checked.InputTensors α) : checked.OutputTensor α :=
  Rep.ofFlatFn fun outputFlatIndex =>
    let outputCoordinate := Coord.unlinearize outputFlatIndex
    let inputCoordinate :=
      checked.packedCoordEquiv.symm outputCoordinate
    inputTensors inputCoordinate.1 inputCoordinate.2

/--
Return a component-indexed function that fills the requested component buffer from the packed
tensor.

The checked coordinate equivalence maps each component flat index to its
unique packed location, so no slice or reshape buffer is constructed.
Component buffers are built on demand and never memoized, so callers control which components
they evaluate and retain.
-/
def unpack {α : Type u} [Storage α] (checked : Pack)
    (packedTensor : checked.OutputTensor α) : checked.InputTensors α :=
  fun component =>
    Rep.ofFlatFn fun inputFlatIndex =>
      packedTensor <|
        checked.packedCoordEquiv
          ⟨component, Coord.unlinearize inputFlatIndex⟩

/--
The direct row-major pack kernel equals the independent coordinate
denotation.
-/
@[grind =, einops =, einops_simps] theorem pack_correct {α : Type u} [Storage α]
    (checked : Pack)
    (inputTensors : checked.InputTensors α) :
    pack checked inputTensors =
      Semantics.denotePack checked inputTensors := by
  ext outputCoordinate
  simp only [pack, Rep.get_ofFlatFn, Semantics.denotePack_apply]
  rw [Coord.unlinearize_linearize]

/--
The direct row-major unpack kernel equals the independent coordinate
denotation.
-/
@[grind =, einops =, einops_simps] theorem unpack_correct {α : Type u} [Storage α]
    (checked : Pack)
    (packedTensor : checked.OutputTensor α) :
    unpack checked packedTensor =
      Semantics.denoteUnpack checked packedTensor := by
  funext component
  ext inputCoordinate
  simp [unpack, Semantics.denoteUnpack]

/-- Unpacking a lowered pack recovers every input tensor. -/
@[simp, grind =, einops =] theorem unpack_pack {α : Type u}
    [Storage α]
    (checked : Pack) (inputTensors : checked.InputTensors α) :
    unpack checked (pack checked inputTensors) =
      inputTensors := by
  rw [pack_correct, unpack_correct]
  exact Semantics.denoteUnpack_denotePack checked inputTensors

/-- Packing a complete lowered unpack recovers the packed tensor. -/
@[simp, grind =, einops =] theorem pack_unpack {α : Type u}
    [Storage α]
    (checked : Pack) (packedTensor : checked.OutputTensor α) :
    pack checked (unpack checked packedTensor) =
      packedTensor := by
  rw [unpack_correct, pack_correct]
  exact Semantics.denotePack_denoteUnpack checked packedTensor

/--
Pack is adjoint to unpack for the standard finite tensor pairing.

The sum ranges over the dependent component family. Multiplication order is
preserved, so a commutative scalar multiplication is not required.
-/
@[grind =, einops =, einops_simps] theorem dot_pack {R : Type u} [Storage R]
    [AddCommMonoid R] [Mul R]
    (checked : Pack) (inputTensors : checked.InputTensors R)
    (packedTensor : checked.OutputTensor R) :
    Rep.dot (pack checked inputTensors) packedTensor =
      ∑ component,
        Rep.dot (inputTensors component)
          (unpack checked packedTensor component) := by
  rw [pack_correct, unpack_correct]
  simp only [Rep.dot, Semantics.denotePack_apply,
    Semantics.denoteUnpack_apply]
  rw [← Fintype.sum_sigma']
  refine Fintype.sum_equiv checked.packedCoordEquiv.symm _ _ ?_
  intro outputCoordinate
  simp

/--
Unpack is adjoint to pack with the packed tensor in the left pairing slot.

The statement is separate from `dot_pack` because it keeps the opposite order of the two
factors without imposing multiplication laws.
-/
@[einops_simps] theorem dot_unpack {R : Type u} [Storage R] [AddCommMonoid R] [Mul R]
    (checked : Pack) (packedTensor : checked.OutputTensor R)
    (componentTensors : checked.InputTensors R) :
    (∑ component,
        Rep.dot (unpack checked packedTensor component)
          (componentTensors component)) =
      Rep.dot packedTensor (pack checked componentTensors) := by
  rw [pack_correct, unpack_correct]
  simp only [Rep.dot, Semantics.denotePack_apply,
    Semantics.denoteUnpack_apply]
  rw [← Fintype.sum_sigma']
  refine Fintype.sum_equiv checked.packedCoordEquiv _ _ ?_
  intro componentCoordinate
  simp only [List.get_eq_getElem, Sigma.eta]
  exact congrArg
    (fun value =>
      packedTensor
          (checked.packedCoordEquiv componentCoordinate) *
        value)
    ((congrArg
      (fun coordinate =>
        componentTensors coordinate.1 coordinate.2)
      (checked.packedCoordEquiv.symm_apply_apply
        componentCoordinate)).symm)

-- The packed pairing contains every top-level parameter and avoids the
-- dependent `Finset.sum` expression that automatic pattern inference rejects.
grind_pattern dot_unpack =>
  Rep.dot packedTensor (pack checked componentTensors)
grind_pattern [einops] dot_unpack =>
  Rep.dot packedTensor (pack checked componentTensors)

end Lowering

end TorchLean.Einops
