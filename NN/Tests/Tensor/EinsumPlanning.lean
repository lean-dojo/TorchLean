/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public meta import NN.Tensor.Internal.Elab.Einsum.Output.Planning
public meta import NN.Tensor.Internal.Elab.Einsum.Planning

/-!
# Einsum Planner Regression Tests

These elaboration-time checks pin output tiling boundaries and stable
contraction-axis ordering, which determine generated tensor traversal.
-/

public meta section

namespace NN.Tests.Tensor.EinsumPlanning

open TorchLean.Tensor.Internal.Elab.Impl

#guard einsumOutputTileWidth? 8 (some 8) 2 == none
#guard einsumOutputTileWidth? 8 (some 9) 2 == some 8
#guard einsumOutputTileWidth? 4 (some 9) 3 == none
#guard einsumOutputTileWidth? 4 none 2 == some 4

-- Equal stride costs preserve source order; differing costs still reorder axes.
#guard contractionAxisOrder [[.named "a"], [.named "b"]] [[2], [2]]
  [.named "a", .named "b"] == [.named "a", .named "b"]
#guard contractionAxisOrder [[.named "a", .named "b"]] [[2, 3]]
  [.named "b", .named "a"] == [.named "a", .named "b"]

end NN.Tests.Tensor.EinsumPlanning
