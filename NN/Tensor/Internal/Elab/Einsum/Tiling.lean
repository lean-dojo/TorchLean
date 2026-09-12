/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Tensor.Internal.Elab.Einsum.Tiling.Semantics
public import NN.Tensor.Internal.Elab.Einsum.Tiling.Width4
public import NN.Tensor.Internal.Elab.Einsum.Tiling.Width8

/-!
# Certified output tiling for generated einsum kernels

`Tiling.Semantics` proves contraction tiling for an arbitrary number of
lanes. `Tiling.Width4` and `Tiling.Width8` are concrete scalar-register
lowerings selected by static cost analysis. Importers use this facade and do
not depend on a particular machine width.
-/
