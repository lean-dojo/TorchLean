/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Tensor.Internal.Semantics.Transform.Geometry
public import NN.Tensor.Internal.Semantics.Transform.RearrangeRepeat
public import NN.Tensor.Internal.Semantics.Transform.Reduction

/-!
# Coordinate semantics for checked transformations

Stable import boundary for checked transform geometry and the independent
`rearrange`, `repeat`, and `reduce` denotations.

The source and output grouping conventions follow einops v0.8.2,
`einops.einops._reconstruct_from_shape_uncached` and
`einops.einops._apply_recipe`, at commit
`8e911db71f2e693a0c434b041180388c685ed06f`.
-/
