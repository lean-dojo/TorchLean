/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Tensor.Internal.Semantics.Transform -- shake: keep

/-!
# Fused lowering for repeat

A checked repeat is executed by one row-major output fill. For each output
flat index, the checked axis projection computes the corresponding input
coordinate after forgetting every introduced axis. The kernel then reads that
entry directly from the input tensor's native array.

This implementation allocates no singleton reshape, broadcast tensor, or
permutation tensor. Its correctness theorem compares the flat-index kernel
with the independent coordinate denotation for every scalar type, rank, and
checked repeat pattern.
-/

@[expose] public section

namespace TorchLean.Tensor.Internal

universe u

namespace Lowering

open Check

/--
Repeat a tensor by filling one native output buffer from the certified
output-to-input coordinate map.
-/
def repeatTensor {α : Type u} [Storage α]
    (checked : CheckedTransform)
    (hKind : checked.value.normalized.kind = .repeat)
    (inputTensor : checked.InputTensor α) : checked.OutputTensor α :=
  Rep.ofFlatFn fun outputFlatIndex =>
    inputTensor <|
      checked.inputCoordinateOfOutput
        (checked.valid.normalization.input_axes_subset_output_of_repeat hKind)
        (Coord.unlinearize outputFlatIndex)

/--
The fused row-major repeat kernel equals the independent coordinate
denotation.
-/
@[grind =] theorem repeatTensor_correct {α : Type u}
    [Storage α]
    (checked : CheckedTransform)
    (hKind : checked.value.normalized.kind = .repeat)
    (inputTensor : checked.InputTensor α) :
    repeatTensor checked hKind inputTensor =
      Semantics.denoteRepeat checked hKind inputTensor := by
  ext outputCoordinate
  simp [repeatTensor, Semantics.denoteRepeat_apply]

end Lowering

end TorchLean.Tensor.Internal
