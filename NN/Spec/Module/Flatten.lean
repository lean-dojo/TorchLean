/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.TensorReductionShape
public import NN.Spec.Module.SpecModule

/-!
# Flatten module wrapper

`flatten_spec` converts a tensor of shape `s` into a vector of length `Shape.size s`.

Why is the output length computed at the type level?

- It reflects the spec contract: flattening is just a re-indexing, so the number of elements is
  determined entirely by the input shape.
- It prevents a common class of downstream mistakes (e.g. wiring a linear layer with the wrong
  feature dimension).

If you're thinking in PyTorch: this is `nn.Flatten()` in its simplest form (collapse all dims).
-/

@[expose] public section


namespace Spec

open Tensor
open ModSpec

-- Flatten module specification wrapper
/-- Wrap `flatten_spec` as an `NNModuleSpec` (`s -> (Shape.size s)`).

The `dimensions` metadata field is not meaningful for flatten because the output length depends on
the whole input shape; exporters should recompute the shape from the typed input.
-/
def FlattenModuleSpec (α : Type) [Context α] (s : Shape) :
  NNModuleSpec α s (.dim (Shape.size s) .scalar) :=
{ forward := fun x => flattenSpec x, kind := "Flatten", export_func := {
  toPyTorch := "nn.Flatten()",
  -- Metadata: flatten changes shape in a way that depends on the full input shape, so exporters
  -- should ignore this pair and use `Shape.size s`.
  dimensions := (0, 0)
} }

end Spec
