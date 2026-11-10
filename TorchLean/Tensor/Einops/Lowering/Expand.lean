/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Tensor.Einops.Semantics.Transform.RearrangeExpand -- shake: keep
public import TorchLean.Tensor.Einops.Semantics.Transform.Reduction -- shake: keep

/-!
# Fused lowering for repeat

A checked repeat is executed by one row-major output fill. For each output
flat index, the checked axis projection computes the corresponding input
coordinate after forgetting every introduced axis. The kernel then reads that
entry from the input tensor's storage.

The kernel allocates no singleton reshape, broadcast tensor, or permutation tensor. Its
correctness theorem compares the flat-index kernel with the independent coordinate denotation
for every scalar type, rank, and checked repeat pattern.
-/

@[expose] public section

namespace TorchLean.Einops

universe u

namespace Lowering

/--
Repeat a tensor by filling one native output buffer from the certified
output-to-input coordinate map.
-/
def expand {α : Type u} [Storage α]
    (checked : Transform)
    (hKind : checked.value.normalized.kind = .expand)
    (inputTensor : checked.InputTensor α) : checked.OutputTensor α :=
  Tensor.Rep.ofFlatFn fun outputFlatIndex =>
    inputTensor <|
      checked.inputOfOutput
        (checked.valid.normalization.input_subset_of_expand hKind)
        (Shape.Coord.unlinearize outputFlatIndex)

/--
The fused row-major repeat kernel equals the independent coordinate
denotation.
-/
@[grind =, einops =, einops_simps] theorem expand_correct {α : Type u}
    [Storage α]
    (checked : Transform)
    (hKind : checked.value.normalized.kind = .expand)
    (inputTensor : checked.InputTensor α) :
    expand checked hKind inputTensor =
      Semantics.denoteExpand checked hKind inputTensor := by
  ext outputCoordinate
  simp [expand, Semantics.denoteExpand_apply]

end Lowering

end TorchLean.Einops
