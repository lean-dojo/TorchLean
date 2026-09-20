/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.TensorReductionShape.ShapeChange
public import NN.Spec.Module.Core

/-!
# Flatten module wrapper

`flattenSpec` converts a tensor of shape `s` into a vector of length `Spec.Shape.size s`.

Why is the output length computed at the type level?

- It reflects the spec contract: flattening is just a re-indexing, so the number of elements is
  determined entirely by the input shape.
- It prevents a common class of downstream mistakes (e.g. wiring a linear layer with the wrong
  feature dimension).

If you're thinking in PyTorch: this is `nn.Flatten()` in its simplest form (collapse all dims).
-/

@[expose] public section


namespace Spec.Module

open TorchLean TorchLean.Tensor

-- Flatten module specification wrapper
/-- Wrap `flattenSpec` as an `Spec.Module` (`s -> (Spec.Shape.size s)`).

The `dimensions` metadata field is not meaningful for flatten because the output length depends on
the whole input shape; exporters should recompute the shape from the typed input.
-/
def flatten (α : Type) [TorchLean.Storage α] [Context α] (s : Shape) :
  Spec.Module α s ([(Spec.Shape.size s)]) :=
{ forward := fun x => flattenSpec x, kind := "Flatten", pythonExpr := "nn.Flatten()" }

end Spec.Module
